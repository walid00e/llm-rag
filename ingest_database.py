import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
DATA_PATH = "./Data"
DB_PATH = "./vector_db"


def create_vector_db():
    print(f"--> Loading PDFs from {DATA_PATH}...")
    # 1. Load Documents
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    if not documents:
        print("No documents found in the data folder.")
        return

    print(f"--> Loaded {len(documents)} pages. Splitting text...")

    # 2. Split Text (Chunking)
    # chunk_size=1000 characters is a good balance for study notes
    # chunk_overlap=100 ensures we don't cut sentences in half at boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--> Split into {len(chunks)} text chunks.")

    # 3. Embeddings (The "Translation" to Numbers)
    # We use a lightweight, local model. No API calls.
    print("--> Initializing Embedding Model (this runs locally)...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Store in ChromaDB
    print("--> Saving to Vector Database...")
    # This will create a folder named 'vector_db' in your current directory
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    print(f"--> Success! Database created at {DB_PATH}")


if __name__ == "__main__":
    create_vector_db()