from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStoreClient:

    def __init__(self, collection_name: str = "langchain"):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            persist_directory="./vector_db_two"
        )

    def add_embedded_documents(
            self,
            documents: List[Document],
    ):
        self.db.add_documents(documents)

    def query(self,
              query: str,
              k: int = 3
    ):
        results = self.db.similarity_search_with_score(query, k)
        return results
