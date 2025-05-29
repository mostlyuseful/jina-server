# Use official Python 3.12 slim image
FROM python:3.12-slim

# Install uv
RUN pip install --no-cache-dir uv

# Setup user
RUN useradd -m -d /app -s /bin/bash jina-server
USER jina-server

# Set working directory
WORKDIR /app

# Copy required files
COPY . .

# Install project dependencies using uv
RUN uv sync --no-cache --extra cpu

# Download the model
RUN uv run download-model

# Expose default port
EXPOSE 80

# Run the server with logging config
CMD ["/app/.venv/bin/uvicorn", "jina_server.serve:app", "--host", "0.0.0.0", "--port", "80", "--log-config", "uvicorn_log_config.yaml"]
