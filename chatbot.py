import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="PDF Research Assistant", page_icon="📘")
st.header("📘 My Assignment Chatbot")

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("Settings & Documents")
    
    # It is best practice to allow the user to enter their own key 
    # or use st.secrets for a fully deployed version
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.divider()
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file and openai_api_key:
    # 1. Extract Text
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)

    # 3. Create Vector Store with Rate-Limit Handling
    # We use a batch process to avoid hitting OpenAI's Free Tier limits
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    @st.cache_resource(show_spinner=False)
    def create_vector_store(_chunks, _api_key):
        vector_store = None
        batch_size = 3  # Low batch size for Free Tier stability
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(_chunks), batch_size):
            batch = _chunks[i : i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embeddings)
            else:
                vector_store.add_texts(batch)
            
            # Update progress
            percent = min((i + batch_size) / len(_chunks), 1.0)
            progress_bar.progress(percent)
            status_text.text(f"Indexing chunks {i+1} to {min(i+batch_size, len(_chunks))}...")
            
            # Brief pause to respect Rate Limits
            time.sleep(1) 
            
        status_text.text("Indexing Complete!")
        return vector_store

    vector_store = create_vector_store(chunks, openai_api_key)

    # 4. LLM Configuration with Chain of Thought (CoT) Prompting
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0, 
        openai_api_key=openai_api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional research assistant. Answer the question ONLY using the provided PDF context.\n\n"
         "PROMPT ENGINEERING TECHNIQUES APPLIED:\n"
         "1. Chain of Thought: Before giving the final answer, perform a brief internal step-by-step analysis.\n"
         "2. Negative Constraints: If the information is not in the context, state that you cannot find it.\n"
         "3. Role Prompting: Act as a meticulous researcher.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Chat Interface
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        with st.spinner("Analyzing document..."):
            response = chain.invoke(user_question)
            st.write("### ✅ Answer:")
            st.write(response)

elif not openai_api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to start.")
