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

# LangChain & Transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLLM
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load Environment Variables
load_dotenv()

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
# FAISS VECTOR STORE
# ------------------------------ #

def store_in_vector_db(chunks, embeddings):
    """Store embeddings in FAISS index."""
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store(embeddings):
    """Load FAISS index from disk."""
    if Path("faiss_index").exists():
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

# ------------------------------ #
# HUGGINGFACE DISTILGPT2 WRAPPER
# ------------------------------ #

class HuggingFaceRunnable(BaseLLM):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _call(self, prompt: str, stop: list = None) -> str:
        response = self.pipeline(prompt, max_length=300, num_return_sequences=1, do_sample=True)
        return response[0]['generated_text']

# ------------------------------ #
# CONVERSATIONAL CHAIN
# ------------------------------ #

def get_conversational_chain(retriever):
    prompt_template = PromptTemplate(
        template="""
        Answer concisely in 1-3 sentences using only the context.
        If unsure, say: "I don't have enough information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

    model_pipeline = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2")
    hf_runnable = HuggingFaceRunnable(model_pipeline)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_len=3
    )

    return ConversationalRetrievalChain.from_llm(
        llm=hf_runnable,
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
    
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-nli-mean-tokens")

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("📄 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                if not pdf_docs:
                    st.error("Please upload PDF files first")
                    return
                
                raw_text = extract_text_from_pdf(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                store_in_vector_db(text_chunks, embeddings)
                st.success("✅ Documents processed successfully!")

    # Chat UI
    st.title("🤖 Chat with PDFs using DistilGPT-2")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs to begin chatting."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        vector_store = load_vector_store(embeddings)
        if not vector_store:
            st.error("Please process documents first.")
            return
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_conversational_chain(vector_store.as_retriever())
                response = chain.invoke({"question": prompt, "chat_history": st.session_state.messages})
                st.write(response["answer"])
        
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
