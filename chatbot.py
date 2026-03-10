import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.header("📘 My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file:

    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.from_texts(chunks, embeddings)

    llm = ChatOllama(
       model="gemma3:4b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant who answers questions ONLY from the provided PDF context.\n"
         "If the answer is not in the PDF text, reply:\n"
         "I can only answer questions related to the uploaded PDF document.\n\n"
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

    user_question = st.text_input("Ask your question:")

    if user_question:
        response = chain.invoke(user_question)
        st.write("### ✅ Answer:")
        st.write(response)