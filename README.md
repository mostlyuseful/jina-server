# Jina Embeddings V3 Server

This project provides a FastAPI server for generating text embeddings using the Jina Embeddings V3 model.

I *just had to* create this to offload the embedding generation from my notebook (fans go brrr) to a server with a GPU and couldn't find a simple solution that met my needs. This server is designed to be easy to deploy and use with no extra frills.

## Features
- HTTP API for generating text embeddings
- Support for task-specific embeddings (retrieval, classification, etc.)
- GPU acceleration support via CUDA
- Detailed performance timing metrics
- Interactive demo page

Out of scope:
- No authentication or rate limiting
  If you need these features, consider using a reverse proxy like Nginx or Traefik.
- No caching - embeddings are always regenerated

## Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- NVIDIA drivers for GPU support (optional)

### Install Dependencies
```bash
# For CPU-only
uv sync --extra cpu

# For NVIDIA GPU support (CUDA 12.6+)
uv sync --extra cuda
```

## Running the Server

### For Development (Web UI + API)
```bash
uvicorn jina_server.serve:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` to access the demo Web UI

### Production with Docker
#### CPU-only container
```bash
docker build -t jina-server .
docker run -p 80:80 jina-server
```

#### NVIDIA GPU container
```bash
docker build -f Dockerfile.nvidia -t jina-server-gpu .
docker run --gpus all -p 80:80 jina-server-gpu
```

### Using Docker Compose
#### CPU Version
```bash
docker-compose -f docker-compose.cpu.yml up
```

#### NVIDIA GPU Version
```bash
docker-compose -f docker-compose.nvidia.yml up
```

## API Documentation

### Generate Embeddings
Endpoint: `POST /v1/embed`

Request Format:
```json
{
  "text": "Your text to embed",
  "task": "retrieval.query"  // Optional fine-tuned task parameter, field can be omitted
}
```

Valid Task Options:
- `retrieval.query`
- `retrieval.passage`
- `separation`
- `classification`
- `text-matching`

### Example API Usage
```bash
curl -X POST "http://localhost/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample text for embedding"}'
```

Response Example:
```json
{
  "embedding": [0.9876, ...],
  "model_name": "jinaai/jina-embeddings-v3",
  "queue_duration_seconds": 0.002541,
  "embedding_duration_seconds": 0.041536
}
```
