from io import BytesIO

import faiss
import numpy as np
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

MODEL_NAME = "all-MiniLM-L6-v2"


def clean_extracted_text(text):
    """Clean common PDF extraction artifacts."""

    text = re.sub(r"(?:/gid\d+){2,}", " ", text)

    text = re.sub(
        r"/parentleft\.case[^ \n]*",
        " ",
        text
    )

    text = re.sub(
        r"/parenright\.case[^ \n]*",
        " ",
        text
    )

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pages_from_pdf(file_bytes, filename):
    """Extract text from an uploaded PDF page by page."""

    reader = PdfReader(BytesIO(file_bytes))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        page_text = page.extract_text()

        if page_text and page_text.strip():

            cleaned_text = clean_extracted_text(
                page_text
            )

            if cleaned_text:

                pages.append({
                    "text": cleaned_text,
                    "source": filename,
                    "page": page_number
                })

    return pages


def create_chunks(pages):
    """Split pages into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


def create_vectorstore(uploaded_files):
    """
    Process uploaded PDF files and create an
    in-memory FAISS vector index.

    Returns:
        index
        metadata
        embedding_model
    """

    all_chunks = []

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        pages = extract_pages_from_pdf(
            file_bytes,
            uploaded_file.name
        )

        chunks = create_chunks(pages)

        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "No readable text was found in the uploaded PDF files."
        )

    print(
        f"Total chunks created: {len(all_chunks)}"
    )

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return (
        index,
        all_chunks,
        embedding_model
    )