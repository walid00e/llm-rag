import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration (Must match your original ingest script)
DB_PATH = "./vector_db"


def add_pdf_to_db(file_path):
    """
    Takes a file path, processes the PDF, and appends it to the existing ChromaDB.
    """

    # 1. Validation
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False

    if not file_path.endswith(".pdf"):
        print("Error: This function only supports .pdf files right now.")
        return False

    print(f"--> Processing file: {file_path}")

    # 2. Load the Single PDF
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"--> Loaded {len(documents)} pages.")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return False

    # 3. Split Text (Must match original chunking settings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--> Created {len(chunks)} new text chunks.")

    # 4. Initialize Embedding Model
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 5. Add to existing ChromaDB
    # We initialize the DB pointing to the SAME folder.
    # It will open the existing database, not overwrite it.
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )

    print("--> Adding chunks to database...")
    db.add_documents(chunks)

    print(f"--> Success! {file_path} has been added to the knowledge base.")
    return True


if __name__ == "__main__":
    # Allow running from command line: python add_document.py path/to/file.pdf
    if len(sys.argv) < 2:
        print("Usage: python add_document.py <path_to_pdf>")
    else:
        file_to_add = sys.argv[1]
        add_pdf_to_db(file_to_add)