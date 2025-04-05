import sys
import os
import streamlit as st
from dotenv import load_dotenv

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/codespace/gemini_multipdf_chat/generativeai-446112-db826aef1f0c.json"

# Fix SQLite3 issue for ChromaDB
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Libraries for handling PDFs and images
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import io

# Google AI and LangChain components
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Extra functionality for metadata filtering and reranking
from metadata_filtering import filter_documents_by_metadata, load_metadata
from reranking import Reranker  

# Detecting the language of input text
from langdetect import detect

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Stop execution if API key or credentials are missing
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Check your .env file or environment variables.")
    st.stop()

if CREDENTIALS_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
else:
    st.error("GOOGLE_APPLICATION_CREDENTIALS not set. Check your .env file.")
    st.stop()

# Set up Google AI API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Google Cloud Vision client
from google.cloud import vision
client = vision.ImageAnnotatorClient()

# ------------------------------ #
# TEXT EXTRACTION FUNCTIONS  
# ------------------------------ #

def google_ocr(image):
    """Extract text from an image using Google Cloud Vision API. Supports multiple languages."""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        vision_image = vision.Image(content=img_byte_arr.getvalue())

        response = client.text_detection(image=vision_image, image_context={"language_hints": ["en", "hi", "bn", "zh"]})
        texts = response.text_annotations
        return texts[0].description if texts else ""
    except Exception as e:
        return f"Error in Google OCR: {str(e)}"

def extract_text_from_pdf(pdf_files):
    """Extract text from PDFs, whether they are digital or scanned."""
    extracted_text = ""

    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text
                else:
                    # If no text is found, assume it's a scanned document and use OCR
                    pdf.seek(0)
                    images = convert_from_bytes(pdf.read())
                    for img in images:
                        extracted_text += google_ocr(img)  
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    return extracted_text

def split_text_into_chunks(text):
    """Break down long text into smaller, manageable pieces."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# ------------------------------ #
# VECTOR STORE & EMBEDDINGS  
# ------------------------------ #

def store_in_vector_db(chunks):
    """Generate text embeddings and store them in ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    chroma_db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="./chroma_db")
    chroma_db.persist()
    return chroma_db

def load_vector_store():
    """Load the existing ChromaDB vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ------------------------------ #
# LANGUAGE DETECTION  
# ------------------------------ #

def detect_language(text):
    """Identify the language of a given text input."""
    try:
        return detect(text)
    except Exception:
        return "unknown"

# ------------------------------ #
# CONVERSATIONAL CHAIN  
# ------------------------------ #

def get_conversational_chain(retriever):
    """Set up a chatbot with memory to handle conversations."""
    prompt_template = PromptTemplate(
        template="""
        Answer the question as detailed as possible based on the provided context.
        If the answer isn't available, respond with: "The answer is not available in the context."

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
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

# ------------------------------ #
# QUERY PROCESSING & SEARCH  
# ------------------------------ #

def user_input(user_question, metadata_filters={}):
    """Process user queries using metadata filtering, hybrid search, and reranking."""
    try:
        detected_language = detect_language(user_question)
        st.write(f"Detected Language: {detected_language.upper()}")  

        # Load ChromaDB
        chroma_db = load_vector_store()
        retriever = chroma_db.as_retriever()
        reranker = Reranker()  

        # Retrieve relevant documents
        documents = retriever.get_relevant_documents(user_question)
        filtered_documents = filter_documents_by_metadata(documents, metadata_filters)

        # Perform hybrid search (vector + keyword search)
        dense_results = retriever.get_relevant_documents(user_question)
        keyword_results = [doc for doc in filtered_documents if user_question.lower() in doc.page_content.lower()]
        hybrid_results = dense_results + keyword_results  

        # Apply reranking to improve results
        reranked_docs = reranker.semantic_rerank(user_question, hybrid_results)

        # Initialize chat chain if not already set
        if "chat_chain" not in st.session_state:
            st.session_state.chat_chain = get_conversational_chain(retriever)

        # Get chatbot response
        response = st.session_state.chat_chain.invoke(user_question)

        return response["answer"] if isinstance(response, dict) else response

    except Exception as e:
        return f"Error in user_input(): {str(e)}"

# ------------------------------ #
# STREAMLIT UI & MAIN FUNCTION  
# ------------------------------ #

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Select PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing your files..."):
                raw_text = extract_text_from_pdf(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                store_in_vector_db(text_chunks)
                st.success("PDFs processed successfully!")

    metadata_file = st.sidebar.file_uploader("Upload Metadata JSON")
    metadata_filters = load_metadata(metadata_file) if metadata_file else {}

    st.title("Chat with your PDFs")
    st.write("Upload some PDFs and ask a question.")

    if prompt := st.chat_input("Type your question here..."):
        response = user_input(prompt, metadata_filters)
        st.write(response)

if __name__ == "__main__":
    main()
