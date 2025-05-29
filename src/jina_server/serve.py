from fastapi import FastAPI, Request, HTTPException
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel
from pydantic import BaseModel
from typing import List

# --- Pydantic Models ---
class EmbedRequestData(BaseModel):
    text: str

class EmbedResponseData(BaseModel):
    embedding: List[float]
    model_name: str

# --- Model Configuration ---
MODEL_NAME = "jinaai/jina-embeddings-v3"

# --- Lifespan Management (Model Loading & Cleanup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Lifespan: Application startup... Loading model {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")
    app.state.model = model
    app.state.model_name = MODEL_NAME
    print(f"Lifespan: Model '{MODEL_NAME}' loaded successfully.")

    # ThreadPoolExecutor for serialized inference
    app.state.inference_executor = ThreadPoolExecutor(max_workers=1)
    print("Lifespan: Inference executor created.")
    
    yield  # Application is now running
    
    print("Lifespan: Application shutdown...")
    app.state.inference_executor.shutdown(wait=True)
    print("Lifespan: Inference executor shut down.")
    print("Lifespan: Resources cleaned up.")

app = FastAPI(lifespan=lifespan)

# --- Synchronous Inference Wrapper ---
def _run_inference(model, text: str) -> List[float]:
    """
    Synchronous function to run model inference.
    To be executed in a ThreadPoolExecutor.
    """
    # .encode typically returns a list of embeddings (np.ndarray)
    # For a single text, we take the first element.
    embedding_np = model.encode([text])[0]
    return embedding_np.tolist() # Convert numpy array to Python list for JSON serialization

# --- API Endpoints ---
@app.post("/v1/embed", response_model=EmbedResponseData)
async def embed(payload: EmbedRequestData, request: Request):
    """
    Generates an embedding for the input text using the loaded Jina model.
    """
    model = request.app.state.model
    executor = request.app.state.inference_executor
    model_name = request.app.state.model_name

    loop = asyncio.get_event_loop()
    try:
        embedding_list = await loop.run_in_executor(
            executor,
            _run_inference, # Pass the synchronous wrapper function
            model,          # First argument to _run_inference
            payload.text    # Second argument to _run_inference
        )
        return EmbedResponseData(embedding=embedding_list, model_name=model_name)
    except Exception as e:
        print(f"Error during inference for text '{payload.text}': {e}")
        # Consider more specific error handling based on model exceptions
        raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)}")

if __name__ == "__main__":
    # This is a simple way to run the server for development.
    # For production, you would typically use a command like:
    # uvicorn src.jina_server.serve:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
