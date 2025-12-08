from core.rag import RAGEngine
from core.llm import LLM
from core.ingestion import IngestionPipeline
from core.storage import VectorStoreClient
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
import os
import uuid
from pathlib import Path

path = "/home/walid/llm-rag-proj/Data/"

app = FastAPI()
vector_store_client = VectorStoreClient()
ingestion_pipeline = IngestionPipeline(vector_store_client)
llm = LLM()
engine = RAGEngine(llm)

@app.post("/ask")
async def process_text(request: Request):
    data = await request.json()
    input_prompt = data.get("prompt", "")
    output = engine.answer(input_prompt)
    return {"answer": output}

@app.post("/config_llm")
async def set_params(request: Request):
    data = await request.json()
    print(data)
    llm.__init__(
        model=data.get("model"),
        reasoning=data.get("reasoning"),
        temperature=data.get("temperature"),
        top_k=data.get("topK"),
        top_p=data.get("topP"),
        num_predict=data.get("numPredict")
    )
    engine.__init__(llm, prompt_instructions=data.get("instructions"))
    return {"status":"successfully changed the parameters"}

@app.get("/get_documents")
def get_documents():
    contents = os.listdir(path)
    return {"documents":contents}

@app.post("/upload_document")
async def post_document(file: UploadFile = File(...)):
    try:
        unique_id = uuid.uuid4().hex
        original_filename = Path(file.filename)
        new_filename = f"{unique_id}_{original_filename.name}"
        file_path = os.path.join(path, new_filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        ingestion_pipeline.ingest_pdf(file_path)
        return {
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


