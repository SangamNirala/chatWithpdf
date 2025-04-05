import sys
import os
import streamlit as st
from dotenv import load_dotenv
import faiss
import numpy as np
from pathlib import Path

# PDF Processing Libraries
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# Google AI and LangChain
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Check your .env file or environment variables.")
    st.stop()

# Configure Google AI API
genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------ #
# TEXT EXTRACTION FUNCTIONS
# ------------------------------ #

def extract_text_from_pdf(pdf_files):
    """Extract text from PDFs (both digital and scanned)."""
    extracted_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text
            else:
                images = convert_from_bytes(pdf.read())
                for img in images:
                    extracted_text += pytesseract.image_to_string(img)
    return extracted_text

def split_text_into_chunks(text):
    """Split extracted text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    return splitter.split_text(text)

# ------------------------------ #
# FAISS VECTOR STORE IMPLEMENTATION
# ------------------------------ #

def store_in_vector_db(chunks, embeddings):
    """Store embeddings in FAISS index"""
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store(embeddings):
    """Load FAISS index from disk"""
    if Path("faiss_index").exists():
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

# ------------------------------ #
# CONVERSATIONAL CHAIN
# ------------------------------ #

# Update the ChatGoogleGenerativeAI model name in the get_conversational_chain function
def get_conversational_chain(retriever):
    """Create conversation chain with FAISS retriever"""
    prompt_template = PromptTemplate(
        template="""
        Answer concisely in 1-3 sentences using only the context.
        If unsure, say: "I don't have enough information."

        Context:\n{context}\n
        Question: \n{question}\n
        Answer:
        """,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
      
        model="models/gemma-3-1b-it",  
       
        temperature=0.3,
        max_output_tokens=150
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_len=3
    )

    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=False
    )

# ------------------------------ #
# STREAMLIT UI
# ------------------------------ #

def main():
    st.set_page_config(page_title="FAISS PDF Chatbot", page_icon="📚")
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Sidebar
    with st.sidebar:
        st.title("PDF Upload")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                if not pdf_docs:
                    st.error("Please upload PDF files first")
                    return
                
                raw_text = extract_text_from_pdf(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                store_in_vector_db(text_chunks, embeddings)
                st.success("Documents processed successfully!")

    # Main interface
    st.title("📚 Chat with PDFs using FAISS")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs to begin chatting"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Check if documents are processed
        vector_store = load_vector_store(embeddings)
        if not vector_store:
            st.error("Please process documents first")
            return
            
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_conversational_chain(vector_store.as_retriever())
                response = chain.invoke({"question": prompt, "chat_history": st.session_state.messages})
                st.write(response["answer"])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
