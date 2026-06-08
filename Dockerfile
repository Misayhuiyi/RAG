FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for sentence-transformers / jieba
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download models (optional — uncomment to bake models into image)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"

EXPOSE 8000

# Default: start server (override with --ingest-only for ingestion)
CMD ["python", "run.py"]
