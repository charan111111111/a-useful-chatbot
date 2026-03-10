import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Updated for OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.header("📘 My Assignment Chatbot")

# Securely get OpenAI Key from Streamlit Secrets
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file and openai_api_key:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted: text += extracted

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)

    # Use OpenAI Embeddings (Cloud based)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Use OpenAI Chat Model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

    # PROMPT ENGINEERING: Adding Chain of Thought (CoT)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional research assistant. Use the following pieces of context to answer the user's question.\n"
         "THINKING PROCESS: First, summarize the key facts from the context. Second, evaluate how they relate to the user's question. Third, provide a final concise answer.\n"
         "If the answer is not in the context, say 'I cannot find this in the document.'\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        response = chain.invoke(user_question)
        st.write(response)
elif not openai_api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to begin.")
