# Medical Information RAG Assistant

A document-grounded Medical Information RAG Assistant that allows users to upload one or more medical PDF documents and ask multiple general medical-information questions based only on the uploaded content.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks before generating an answer with Groq. Source document names and page numbers are displayed for supported answers.

> **Educational information only.** This application does not diagnose medical conditions or provide personalized treatment, medication dosage, or prescriptions.

---

## Features

- Upload one or more medical PDF documents
- Extract text from uploaded PDFs
- Clean extracted PDF text
- Split documents into overlapping chunks
- Generate semantic embeddings using Sentence Transformers
- Perform semantic similarity search using NumPy
- Retrieve relevant document chunks for each question
- Generate grounded answers using Groq
- Ask multiple questions without re-uploading the documents
- Display source document names and page numbers
- Handle questions outside the uploaded document knowledge base
- Streamlit-based web interface
- Medical safety and grounding instructions

---

## System Architecture

```text
                     User
                       │
                       ▼
              Upload Medical PDF
                       │
                       ▼
               PDF Text Extraction
                       │
                       ▼
                 Text Cleaning
                       │
                       ▼
                    Chunking
                       │
                       ▼
          Sentence Transformer Embeddings
                       │
                       ▼
              NumPy Vector Storage
                       │
                       ▼
                 User Question
                       │
                       ▼
                Query Embedding
                       │
                       ▼
           Cosine Similarity Search
                       │
                       ▼
             Relevant Document Chunks
                       │
                       ▼
                    Groq LLM
                       │
                       ▼
             Grounded Answer + Sources

-> Technologies Used

Python
Streamlit – web interface
PyPDF – PDF text extraction
LangChain Text Splitters – text chunking
Sentence Transformers – semantic embeddings
NumPy – cosine similarity search
Groq API – answer generation
python-dotenv – environment variable management

-> Project Structure
medical-information-rag-assistant/
│
├── app.py
├── document_processor.py
├── rag_system.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── medical_documents/       # Local development documents
├── vectorstore/             # Local/generated files if used during development
├── venv/                    # Python virtual environment
└── config.env               # Local API key configuration

-> How the Application Works
1. Upload Documents

The user uploads one or more medical PDF documents through the Streamlit interface.

2. PDF Processing

The uploaded PDFs are:

extracted page by page
cleaned
divided into overlapping text chunks
3. Embedding Generation

Each text chunk is converted into a semantic vector using a Sentence Transformer model.

4. Similarity Search

The application stores the generated embeddings in memory.

When the user asks a question:

The question is converted into an embedding.
Cosine similarity is calculated against the document embeddings.
The most relevant chunks are retrieved.
5. Context Construction

The retrieved chunks are combined with their:

document name
page number
text content
6. Answer Generation

The retrieved context is sent to the Groq language model.

The system instructs the model to answer using only the retrieved document context.

7. Source Attribution

For supported answers, the application displays the document name and page number associated with the retrieved information.

-> Example Usage

Upload a medical PDF and click Process Documents.

Example question:

What information about BMI is provided in this document?

The assistant retrieves the relevant content and generates a grounded answer with source information.

Multiple questions can be asked using the same uploaded documents without uploading them again.

-> Safety and Grounding

This application is designed as an educational medical-information assistant.

The system:

uses uploaded documents as its knowledge source
retrieves relevant document content before generation
avoids unsupported answers
does not diagnose users
does not provide personalized treatment
does not provide medication dosage or prescriptions
displays source information for supported answers
reports when information cannot be found in the uploaded documents

If the requested information is not supported by the uploaded documents, the system responds:

I could not find this information in the uploaded document(s).

This behavior is intentional and helps reduce unsupported or hallucinated answers.

-> Limitations

The quality of answers depends on:

the quality of the uploaded PDF
the information contained in the uploaded document
PDF text extraction quality
chunking quality
embedding quality
retrieval quality

The assistant cannot reliably answer information that is absent from the uploaded document.

-> Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/medical-information-rag-assistant.git
cd medical-information-rag-assistant

Create a virtual environment:

python -m venv venv

Activate it on Windows:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

-> API Configuration

Create a local file named:

config.env

Add your Groq API key:

GROQ_API_KEY=your_api_key_here

Never commit config.env or expose your API key publicly.

-> Run the Application

Start Streamlit:

python -m streamlit run app.py

Open the local Streamlit URL shown in the terminal.

-> Testing

Example supported questions depend on the uploaded document.

For example:

What is diabetes?
What are the risk factors?
What is BMI?
What complications are described?

The application also supports multiple questions from the same uploaded document.

For an unsupported question, the system should indicate that the requested information could not be found in the uploaded document.

-> Future Improvements

Hybrid keyword + semantic retrieval
Reranking of retrieved chunks
Improved medical PDF/table extraction
Better source highlighting
Retrieval evaluation metrics
Support for additional document formats
Improved conversation memory

-> Project Purpose

This project demonstrates the practical implementation of Retrieval-Augmented Generation for document-grounded medical information.

It covers:

document ingestion
PDF processing
text chunking
semantic embeddings
vector similarity retrieval
grounded LLM generation
source attribution
multi-question document interaction
medical safety guardrails
Deployment to a cloud platform
