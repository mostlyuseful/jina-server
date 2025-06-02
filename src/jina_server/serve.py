# Jina Server - A FastAPI server for serving Jina embeddings
# Copyright (C) 2025 Maurice-Pascal Sonnemann <mpsonnemann@gmail.com>

import logging
import time
import yaml
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import Dict, List, Optional
from enum import Enum

# --- Pydantic Models ---
class TaskEnum(str, Enum):
    """
    Allowed task values for Jina embeddings.
    See: https://jina.ai/embeddings-v3/
    """
    retrieval_query = 'retrieval.query'
    retrieval_passage = 'retrieval.passage'
    separation = 'separation'
    classification = 'classification'
    text_matching = 'text-matching'

class EmbedRequestData(BaseModel):
    texts: list[str]
    task: Optional[TaskEnum] = None

    class Config:
        use_enum_values = True

class EmbedResponseData(BaseModel):
    embeddings: list[list[float]]
    model_name: str
    queue_duration_seconds: float
    embedding_duration_seconds: float

# --- Model Configuration ---
MODEL_NAME = "jinaai/jina-embeddings-v3"

def get_recommended_device():
    """
    Get the recommended device for running the model.
    """
    # Order: CUDA > MPS > CPU
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# --- Lifespan Management (Model Loading & Cleanup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger("uvicorn")
    logger.info("Lifespan: Importing transformers")
    from transformers import AutoModel
    logger.info(f"Lifespan: Application startup... Loading model {MODEL_NAME}")
    device = get_recommended_device()
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    model.to(device)
    app.state.model = model
    app.state.model_name = MODEL_NAME
    logger.info(f"Lifespan: Model '{MODEL_NAME}' loaded successfully.")

    # ThreadPoolExecutor for serialized inference
    app.state.inference_executor = ThreadPoolExecutor(max_workers=1)
    logger.info("Lifespan: Inference executor created.")

    yield  # Application is now running

    logger.info("Lifespan: Application shutdown...")
    app.state.inference_executor.shutdown(wait=True)
    logger.info("Lifespan: Inference executor shut down.")
    logger.info("Lifespan: Resources cleaned up.")

app = FastAPI(lifespan=lifespan)

# --- Synchronous Inference Wrapper ---
def _run_inference(model, texts: list[str], task: Optional[str] = None) -> list[list[float]]:
    """
    Synchronous function to run model inference.
    To be executed inPoolExecutorPoolExecutor.
    
    Args:
        model: The loaded Jina model
        texts: Input texts to embed
        task: Optional task name (one of 'retrieval.query', 'retrieval.passage', 
              'separation', 'classification', 'text-matching')
    """
    embedding_np = model.encode(texts, task=task)
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

    start_time = time.perf_counter()  # Record when request is received
    loop = asyncio.get_event_loop()
    try:
        # Record time just before queuing inference
        queue_start = time.perf_counter()
        embeddings_list = await loop.run_in_executor(
            executor,
            _run_inference, # Pass the synchronous wrapper function
            model,          # First argument to _run_inference
            payload.texts,   # Second argument to _run_inference
            payload.task    # Third argument to _run_inference
        )
        # Record time after inference completes
        inference_end = time.perf_counter()
        
        # Calculate durations
        queue_duration = queue_start - start_time  # Time from request to queuing
        embedding_duration = inference_end - queue_start  # Time from queuing to completion
        
        return EmbedResponseData(
            embeddings=embeddings_list,
            model_name=model_name,
            queue_duration_seconds=queue_duration,
            embedding_duration_seconds=embedding_duration
        )
    except Exception as e:
        print(f"Error during inference for texts: {e}")
        # Consider more specific error handling based on model exceptions
        raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)}")

@app.get("/")
async def index():
    from pathlib import Path
    template_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(template_path)

def find_log_config(log_fn = "uvicorn_log_config.yaml") -> Optional[Path]:
    """
    Find the log configuration file (default "uvicorn_log_config.yaml") in this directory or the parent directories.
    """
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        log_config_path = current_path / log_fn
        if log_config_path.exists():
            return log_config_path
        current_path = current_path.parent
    return None  # Not found

def get_log_config(log_fn = "uvicorn_log_config.yaml") -> Optional[Dict]:
    """
    Get the log configuration as a dictionary, parsed from the YAML file.
    """
    log_config_path = find_log_config(log_fn)
    if log_config_path:
        with open(log_config_path, 'r') as f:
            return yaml.safe_load(f)
    return None  # Not found

def main_serve():
    """
    Start the FastAPI server under Uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=get_log_config())

def main_download_model():
    """
    Download the model to the local cache.
    """
    from transformers import AutoModel
    AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=False)

if __name__ == "__main__":
    # This is a simple way to run the server for development.
    # For production, you would typically use a command like:
    # uvicorn src.jina_server.serve:app --host 0.0.0.0 --port 8000
    main_serve()
