from langchain_ollama import OllamaLLM

model = OllamaLLM(model='deepseek-r1')

print(model.invoke("who are you?"))
