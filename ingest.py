from pathlib import Path
import re

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

DOCUMENT_FOLDER = Path("medical_documents")
VECTORSTORE_FOLDER = Path("vectorstore")


# --------------------------------------------------
# CHUNKING SETTINGS
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

def clean_extracted_text(text):
    """
    Remove common PDF extraction artifacts
    while keeping the actual medical text.
    """

    # Remove repeated /gid.../gid... PDF artifacts
    text = re.sub(
        r"(?:/gid\d+){2,}",
        " ",
        text
    )

    # Remove parentleft/parenright extraction artifacts
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

    # Replace repeated whitespace
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Clean excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()

# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pages_from_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Each page stores:
    - text
    - source filename
    - page number
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text and page_text.strip():

            cleaned_text = clean_extracted_text(page_text)

            if cleaned_text:

                pages.append({
                    "text": cleaned_text,
                    "source": pdf_path.name,
                    "page": page_number
                })

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(pages):
    """
    Split page text into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("MEDICAL RAG - DOCUMENT INGESTION")
    print("=" * 60)

    # Find PDFs
    pdf_files = list(DOCUMENT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("ERROR: No PDF files found!")
        return

    print(f"\nFound {len(pdf_files)} PDF files.")

    # Store all chunks
    all_chunks = []

    # --------------------------------------------------
    # PROCESS PDFs
    # --------------------------------------------------

    for pdf_file in pdf_files:

        print("\n" + "-" * 60)
        print(f"Processing: {pdf_file.name}")
        print("-" * 60)

        # Extract pages
        pages = extract_pages_from_pdf(pdf_file)

        print(f"Pages extracted: {len(pages)}")

        # Create chunks
        chunks = create_chunks(pages)

        print(f"Chunks created: {len(chunks)}")

        # Add to complete collection
        all_chunks.extend(chunks)

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSING COMPLETE")
    print("=" * 60)

    print(f"Total chunks: {len(all_chunks)}")

    # --------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------

    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully.")

    # --------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------

    print("\nCreating embeddings...")
    print("This may take some time because there are many chunks.")

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("\nEmbeddings created successfully.")

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # --------------------------------------------------
    # CREATE FAISS INDEX
    # --------------------------------------------------

    print("\nCreating FAISS vector database...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS index created successfully.")

    print(f"Vectors stored in FAISS: {index.ntotal}")

    # --------------------------------------------------
    # SAVE FAISS INDEX
    # --------------------------------------------------

    VECTORSTORE_FOLDER.mkdir(exist_ok=True)

    index_path = VECTORSTORE_FOLDER / "medical_index.faiss"

    faiss.write_index(
        index,
        str(index_path)
    )

    # --------------------------------------------------
    # SAVE CHUNK METADATA
    # --------------------------------------------------

    metadata_path = VECTORSTORE_FOLDER / "metadata.pkl"

    with open(metadata_path, "wb") as file:

        pickle.dump(
            all_chunks,
            file
        )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FAISS VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Index file: {index_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Total vectors: {index.ntotal}")
    print(f"Embedding dimension: {dimension}")

    print("\nYour medical knowledge base is ready!")


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()