# =============================================================================
# Production Dockerfile — Academic Research Intelligence System
# =============================================================================
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn gunicorn

# Copy application source code and frontend assets
COPY pyproject.toml .
COPY configs/ configs/
COPY data/ data/
COPY frontend/ frontend/
COPY results/ results/
COPY src/ src/
COPY scripts/ scripts/

EXPOSE 8000

CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]

