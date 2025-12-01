import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Configuration
DB_PATH = "./vector_db"

# 1. Setup the Prompt Template
# This is the instructions we give the AI.
PROMPT_TEMPLATE = """
    You are a helpful study assistant. Use the following pieces of context to answer the question at the end.
    If the answer is not in the context, simply say "I don't know based on the provided material." Do not try to make up an answer.
    
    Context:
    {context}
    
    ---
    
    Question: {question}
"""


def query_rag(query_text):
    # 2. Prepare the Embedding Function
    # We must use the EXACT same model we used to ingest the data.
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 3. Connect to the Database
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # 4. Search for relevant data (Retrieval)
    # k=3 means "give me the top 3 most relevant chunks"
    results = db.similarity_search_with_score(query_text, k=3)

    # Combine the retrieved text into a single string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 5. Prepare the Prompt (Augmentation)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 6. Generate the Answer (with Ollama)
    # We use the model we pulled earlier
    model = OllamaLLM(model="llama3.2")

    print(f"\nThinking about: '{query_text}'...")
    response_text = model.invoke(prompt)

    # 7. Output the result
    print("\nResult:")
    print(response_text)

    # Optional: Show sources
    # print("\nSources used:")
    # for doc, score in results:
    #     print(f"- [Score: {score:.4f}] {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    # This allows us to run the script from the command line
    # Example: python rag_engine.py "What is attention?"
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The question to ask.")
    args = parser.parse_args()

    query_rag(args.query_text)