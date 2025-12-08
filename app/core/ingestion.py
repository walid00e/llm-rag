from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from .storage import VectorStoreClient
from .models import IngestionResult

class IngestionPipeline:
    def __init__(self, vector_store: VectorStoreClient):
        self.vector_store = vector_store

    def load_pdf(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return pages

    def split_text(self, pages: List[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        return splitter.split_documents(pages)

    def embed_and_store(self, chunks: List[Document]):
        self.vector_store.add_embedded_documents(chunks)

    def ingest_pdf(self, file_path: str):
        pages = self.load_pdf(file_path)
        chunks = self.split_text(pages)
        self.embed_and_store(chunks)
