import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Config
st.set_page_config(page_title="Gemini PDF Bot", page_icon="🤖")
st.header("🤖 Useful AI Document Bot")

# --- Sidebar ---
with st.sidebar:
    st.title("Configuration")
    # Link to get key: https://aistudio.google.com/app/apikey
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    file = st.file_uploader("Upload your PDF Assignment", type="pdf")

if file and gemini_api_key:
    # 1. Extract Text from PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # 2. Split Text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=250
    )
    chunks = text_splitter.split_text(text)

    # 3. Setup Gemini Embeddings and Vector Store
    # This converts text into numbers so the bot can 'search' the PDF
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=gemini_api_key
    )
    
    @st.cache_resource(show_spinner=False)
    def create_vc(_chunks, _key):
        return FAISS.from_texts(_chunks, embeddings)

    vector_store = create_vc(chunks, gemini_api_key)

    # 4. Setup Gemini LLM (1.5 Flash is fast and free)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0, 
        google_api_key=gemini_api_key
    )

    # --- PROMPT ENGINEERING TECHNIQUES ---
    # We use: Role Prompting, Chain of Thought (CoT), and Negative Constraints
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a highly skilled academic assistant. Answer the user's question using ONLY the context provided.\n\n"
         "INSTRUCTIONS:\n"
         "1. Summarize the relevant facts from the document.\n"
         "2. Reason step-by-step how these facts answer the question.\n"
         "3. If the answer is NOT in the document, say: 'I'm sorry, that information is not in the uploaded PDF.'\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    # 5. Build the Chain
    retriever = vector_store.as_retriever()
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. User Interaction
    user_query = st.text_input("Ask a question about your document:")

    if user_query:
        with st.spinner("Gemini is thinking..."):
            response = chain.invoke(user_query)
            st.markdown("### ✅ Bot Response:")
            st.write(response)

elif not gemini_api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to begin.")
