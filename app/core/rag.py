from langchain_core.prompts import ChatPromptTemplate
from .storage import VectorStoreClient
from .llm import LLM

class RAGEngine:
    def __init__(self, llm: LLM, prompt_instructions: str = "You are a helpful study assistant. Use the following pieces of context to answer the question at the end."):
        self.vector_store_client = VectorStoreClient()
        self.llm = llm
        self.prompt_instructions = prompt_instructions

    prompt_template = """
        {instructions}
        
        Context:
        {context}

        ---

        Question: {question}
    """

    def answer(self, query_text):
        # get the results from the db
        results = self.vector_store_client.query(query_text)
        # join the results into one text
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        # create the prompt template
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        # create the prompt
        prompt = prompt_template.format(instructions=self.prompt_instructions, context=context_text, question=query_text)
        # invoke the llm
        response = self.llm.invoke_llm(prompt)
        # return the response
        return response