import sys
import os
import streamlit as st
from dotenv import load_dotenv

# SQLite3 Fix for ChromaDB
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# PDF Processing Libraries
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# Google AI and LangChain
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Metadata Filtering Module
from metadata_filtering import filter_documents_by_metadata, load_metadata

# Import Reranker
from reranking import Reranker  

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
                extracted_text += page_text  # Extract from digital PDFs
            else:
                images = convert_from_bytes(pdf.read())  # Convert scanned pages to images
                for img in images:
                    extracted_text += pytesseract.image_to_string(img)  # OCR for scanned PDFs

    return extracted_text

def split_text_into_chunks(text):
    """Split extracted text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# ------------------------------ #
# VECTOR STORE & EMBEDDINGS  
# ------------------------------ #

def store_in_vector_db(chunks):
    """Store extracted text embeddings in ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    chroma_db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="./chroma_db")
    chroma_db.persist()
    return chroma_db

def load_vector_store():
    """Load stored ChromaDB for retrieval."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ------------------------------ #
# CONVERSATIONAL CHAIN  
# ------------------------------ #

def get_conversational_chain(retriever):
    """Create a conversational chain with memory."""
    prompt_template = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the context, say: "The answer is not available in the context."

        Context:\n {context}\n
        Question: \n{question}\n
        Answer:
        """,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        client=genai,
        temperature=0.3
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}  # Fix input key issue
    )

# ------------------------------ #
# QUERY PROCESSING & HYBRID SEARCH WITH RERANKING  
# ------------------------------ #

def user_input(user_question, metadata_filters={}):
    """Process user queries with metadata filtering, hybrid search, and reranking."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    reranker = Reranker()  # Initialize reranker

    try:
        # Load ChromaDB
        chroma_db = load_vector_store()
        retriever = chroma_db.as_retriever()

        # Retrieve documents
        documents = retriever.get_relevant_documents(user_question)

        # Apply Metadata Filtering
        filtered_documents = filter_documents_by_metadata(documents, metadata_filters)

        # Hybrid Search (Vector + Keyword Search)
        dense_results = retriever.get_relevant_documents(user_question)  # Vector search
        keyword_results = [doc for doc in filtered_documents if user_question.lower() in doc.page_content.lower()]
        hybrid_results = dense_results + keyword_results  # Merge results

        # Apply Reranking
        reranked_docs = reranker.semantic_rerank(user_question, hybrid_results)

        # Setup Chat Chain (Ensure correct input format)
        if "chat_chain" not in st.session_state:
            st.session_state.chat_chain = get_conversational_chain(retriever)

        # Fix the Input Key Issue (Only pass "question" or restructure prompt)
        response = st.session_state.chat_chain.invoke(user_question)  # Pass as a string, not a dict

        return response["answer"] if isinstance(response, dict) else response

    except Exception as e:
        return f"Error in user_input(): {str(e)}"


# ------------------------------ #
# STREAMLIT UI & MAIN FUNCTION  
# ------------------------------ #

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="")

    # Sidebar - Upload PDFs
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdf(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                store_in_vector_db(text_chunks)
                st.success("PDFs processed successfully!")

    # Metadata Filtering Options
    metadata_file = st.sidebar.file_uploader("Upload Metadata JSON")
    metadata_filters = load_metadata(metadata_file) if metadata_file else {}

    # Main Chat Interface
    st.title("Chat with PDFs using Gemini")
    st.write("Upload some PDFs and ask me a question!")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Input
    if prompt := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt, metadata_filters)
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
