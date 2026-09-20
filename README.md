# 🩺 Medical Information RAG Assistant

A document-grounded Medical Information RAG Assistant that allows users to upload medical PDF documents and ask multiple general medical-information questions based only on the uploaded content.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks before generating an answer. Source document names and page numbers are also shown for supported answers.

> ⚠️ Educational information only. This application does not diagnose medical conditions or provide personalized treatment, medication dosage, or prescriptions.

---

## ✨ Features

- Upload one or more medical PDF documents
- Extract text from uploaded PDFs
- Clean extracted PDF text
- Split documents into overlapping chunks
- Generate semantic embeddings using Sentence Transformers
- Store and search embeddings using FAISS
- Retrieve relevant chunks for each question
- Generate grounded answers using Groq
- Ask multiple questions without re-uploading the documents
- Display source document and page numbers
- Refuse unsupported questions when the answer is not found in the uploaded documents
- Streamlit-based web interface

---

## 🏗️ System Architecture

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
              FAISS Index
                  │
                  │
          ┌───────▼────────┐
          │ User Question  │
          └───────┬────────┘
                  │
                  ▼
           Query Embedding
                  │
                  ▼
        Retrieve Relevant Chunks
                  │
                  ▼
             Groq LLM
                  │
                  ▼
       Grounded Answer + Sources
