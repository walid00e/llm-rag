from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name
        )

