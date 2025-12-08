from langchain_ollama import OllamaLLM

class LLM:
    def __init__(
            self,
            model: str = "llama3.2:latest",
            reasoning: bool = False,
            num_predict: int = 128,
            temperature: float = 1,
            top_k: int = 100,
            top_p: float = 1.0
    ):
        self.model = OllamaLLM(
            model=model,
            reasoning=reasoning,
            num_predict=num_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def invoke_llm(self, prompt:str):
        response_text = self.model.invoke(prompt)
        return response_text