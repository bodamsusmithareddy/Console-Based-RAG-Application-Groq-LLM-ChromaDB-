
# README for RAG Console-Based Application

## Overview
This application implements a **Retrieval-Augmented Generation (RAG)** pipeline that extracts text from PDF and PowerPoint files, indexes the content using semantic search with **ChromaDB**, and integrates with a Large Language Model (LLM) for answering user queries based on the indexed data.

## Features
- Extract text from PDF and PowerPoint files.
- Perform semantic search on the extracted text using ChromaDB and SentenceTransformer embeddings.
- Query the indexed content and generate responses using **Groq LLM**.
- Handle large-scale documents by chunking data into manageable pieces for efficient indexing and retrieval.

---

## Prerequisites

### Libraries
Ensure the following Python libraries are installed:
- `os` (standard library)
- `fitz` (from PyMuPDF)
- `python-pptx`
- `langchain`
- `sentence-transformers`
- `langchain-chroma`
- `langchain-groq`

Install dependencies using:
```bash
pip install PyMuPDF python-pptx langchain sentence-transformers langchain-chroma langchain-groq
```

### API Key
Set up your **Groq API Key**:
1. Obtain an API key from Groq.
2. Add it to the script in the `os.environ["GROQ_API_KEY"]` variable.

### Directory Structure
Ensure that the directory structure includes a folder for ChromaDB:
```
project_root/
  |-- chroma_db/
```
If the `chroma_db/` folder does not exist, the script will create it automatically.

---

## How to Use

### 1. Run the Application
Execute the script using:
```bash
python RAG_Application_Susmitha_Reddy_Bodam.py
```

### 2. File Upload and Text Extraction
- Input the path to a PDF or PowerPoint file when prompted.
- The script will extract text and display it on the console for verification.

### 3. Indexing
- The extracted text is indexed for semantic search using ChromaDB.
- Documents are chunked into logical pieces to improve retrieval quality.

### 4. Querying
- Enter a query when prompted. The application will:
  1. Perform a similarity search on the indexed content to retrieve relevant contexts.
  2. Use the retrieved context to generate an answer with Groq LLM.

### 5. Exit
- Type `exit` at any time to terminate the application.

---

## Key Functions

### Text Extraction
- **`extract_text_from_pptx(file_path)`**: Extracts text from PowerPoint files.
- **`extract_text_from_pdf(file_path)`**: Extracts text from PDF files.

### Embedding
- **`SentenceTransformerEmbedding`**: Wrapper for SentenceTransformer to embed documents and queries.

### Indexing
- **`create_index(data)`**: Creates a ChromaDB index from the extracted text.

### Query and Response
- **`query_index(db, query)`**: Searches for relevant contexts in the indexed database.
- **`generate_response(context, question)`**: Uses Groq LLM to generate answers from retrieved contexts.

### Main Application
- **`main()`**: Orchestrates the entire process of text extraction, indexing, querying, and generating responses.

---

## Error Handling
- Handles unsupported file formats with appropriate error messages.
- Manages exceptions during text extraction, indexing, and querying for robust performance.

---

## Example Usage
1. **Upload a file:**
   ```
   Enter the path to your file (.pdf or .pptx): example.pptx
   ```
2. **Query:**
   ```
   Enter your query: What is the main idea of slide 3?
   ```
3. **Response:**
   ```
   Response:
   The main idea of slide 3 is about...
   ```

---

