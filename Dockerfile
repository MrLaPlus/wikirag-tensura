# Multi-stage production Dockerfile for WikiRAG
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "fastapi>=0.111.0" "uvicorn>=0.30.0" onnxruntime optimum

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and project configs
COPY src/ /app/src/
COPY projects/ /app/projects/
COPY eval/ /app/eval/
COPY pyproject.toml README.md .env.example /app/

# Install wikirag package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV WIKIRAG_HOST=0.0.0.0
ENV WIKIRAG_PORT=8000

CMD ["python", "-m", "wikirag", "serve", "--host", "0.0.0.0", "--port", "8000"]
