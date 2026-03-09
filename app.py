import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()

# Streamlit UI
st.title("RAG Chatbot")
st.write("Ask questions about your documents")

# Step 1: Load documents
loader = TextLoader("sample.txt")
documents = loader.load()

# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 5: Create retriever
retriever = vectorstore.as_retriever()

# Step 6: Load LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

# Step 7: Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Chat interface
query = st.text_input("Ask a question:")

if query:
    result = qa_chain.run(query)
    st.write("Answer:")
    st.write(result)