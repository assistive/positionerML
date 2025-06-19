# Multi-stage Dockerfile for Qwen 2.5-VL Service

# Base stage with CUDA support
FROM qwenllm/qwenvl:2-cu121 as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
COPY . .
RUN pip install -e .
CMD ["python", "scripts/run_service.py", "--auto-load-model"]

# Production stage
FROM base as production
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 qwenuser && chown -R qwenuser:qwenuser /app
USER qwenuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "scripts/run_service.py", "--auto-load-model"]
