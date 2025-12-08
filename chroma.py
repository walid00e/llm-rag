from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="langchain",
    embedding_function=embedding_function,
    persist_directory="./vector_db_two"
)

db.add_documents([
    Document(page_content="new text", metadata={"source": "appendix"}),
    Document(page_content="other one", metadata={"source":"tweet"})
])

results = db.similarity_search("other", k=1)

for r in results:
    print(r.page_content)