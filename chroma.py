from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

db = Chroma(
    collection_name="langchain",
    embedding_function=embedding_function,
    persist_directory="./vector_db"
)

results = db.similarity_search("scaling law", k=3)

for r in results:
    print(r.page_content)

db.add_documents(
    [Document(page_content="new text", metadata={"source": "appendix"})]
)

db.persist()