import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Gemini Assignment Bot", page_icon="🤖")
st.header("🤖 Useful AI Document Bot")

# --- Sidebar ---
with st.sidebar:
    st.title("Configuration")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    st.info("Get a free key at: https://aistudio.google.com/app/apikey")
    
    st.divider()
    file = st.file_uploader("Upload your PDF", type="pdf")

if file and gemini_api_key:
    # 1. Extract Text
    pdf_reader = PdfReader(file)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    # 2. Split Text (Smaller chunks help with rate limits)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # 3. Embedding Logic with Batching
    # FIX APPLIED: Using the active "gemini-embedding-001" model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=gemini_api_key
    )
    
    @st.cache_resource(show_spinner=False)
    def create_vector_store(_chunks, _key):
        try:
            batch_size = 5
            vector_store = None
            
            progress_bar = st.progress(0)
            for i in range(0, len(_chunks), batch_size):
                batch = _chunks[i : i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embeddings)
                else:
                    vector_store.add_texts(batch)
                
                # Sleep briefly to respect the free-tier quota
                time.sleep(0.5) 
                progress_bar.progress(min((i + batch_size) / len(_chunks), 1.0))
            
            return vector_store
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")
            return None

    vector_store = create_vector_store(chunks, gemini_api_key)

    if vector_store:
        # 4. LLM & Prompt
        # FIX APPLIED: Using the modern "gemini-2.5-flash" model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
            google_api_key=gemini_api_key
        )

        # Prompt Engineering Requirements for the Assignment
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant. Answer the user's question using ONLY the context provided.\n\n"
             "INSTRUCTIONS:\n"
             "1. Think step-by-step about how the context relates to the question.\n"
             "2. Provide a concise answer.\n"
             "3. If the answer is not in the text, say 'I cannot find this in the document.'\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ])

        chain = ({"context": vector_store.as_retriever() | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
                  "question": RunnablePassthrough()} 
                 | prompt | llm | StrOutputParser())

        user_query = st.text_input("Ask a question about your PDF:")
        if user_query:
            with st.spinner("Thinking..."):
                st.write("### ✅ Response:")
                st.write(chain.invoke(user_query))

elif not gemini_api_key:
    st.warning("Please enter your Gemini API Key in the sidebar.")
