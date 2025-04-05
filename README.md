# Gemini PDF Chatbot

Gemini PDF Chatbot is a Streamlit-based application that allows users to chat with a conversational AI model trained on PDF documents. The chatbot extracts information from uploaded PDF files and answers user questions based on the provided context.


<https://github.com/kaifcoder/gemini_multipdf_chat/assets/57701861/f6a841af-a92d-4e54-a4fd-4a52117e17f6>



## Introduction
Welcome to the **Gemini PDF Chatbot**! This innovative application leverages the power of Streamlit, Google Cloud Vision API, and LangChain to provide users with an interactive platform for querying information from PDF documents. Whether you're dealing with digital or scanned PDFs, this chatbot can extract text and facilitate meaningful conversations about the content.

## Key Features
- **Multi-PDF Upload**: Seamlessly upload multiple PDF files for processing.
- **Text Extraction**: Utilize advanced OCR capabilities to extract text from both digital and scanned documents.
- **Conversational Interface**: Engage with the chatbot to ask questions and receive contextually relevant answers.
- **Metadata Filtering**: Filter search results based on document metadata for enhanced relevance.
- **Reranking**: Improve the accuracy of search results through intelligent reranking mechanisms.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Required Packages**:
   Install the necessary Python libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Google Cloud Credentials**:
   - Create a service account in the Google Cloud Console and download the JSON key file.
   - Set the environment variable for Google Cloud credentials:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
     ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root directory and add your Google API key:
   ```plaintext
   GOOGLE_API_KEY=your_google_api_key
   ```

## Running the Application
1. **Start the Streamlit Application**:
   Launch the application with the following command:
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**:
   Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

3. **Upload PDFs and Ask Questions**:
   Use the sidebar to upload your PDF documents. Once uploaded, you can type your questions in the chat interface to receive answers based on the content of the PDFs.

## Contributing
We welcome contributions to enhance the functionality and performance of the Gemini PDF Chatbot. If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. For more details, please refer to the LICENSE file.

## Acknowledgments
- **Google Cloud Vision API**: For providing powerful OCR capabilities.
- **LangChain**: For enabling conversational AI functionalities.
- **ChromaDB**: For efficient document storage and retrieval.

Thank you for using the Gemini PDF Chatbot! We hope it enhances your productivity and makes working with PDFs easier and more interactive.
