FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml setup.py ./
RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    pip install -e .

# Create a lightweight runtime image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p logs .cache .embeddings

# Copy the installed packages and application code
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/llm_gateway.egg-info /app/llm_gateway.egg-info
COPY . .

# Create non-root user for security
RUN groupadd -r llmgateway && \
    useradd -r -g llmgateway llmgateway && \
    chown -R llmgateway:llmgateway /app

# Use non-root user
USER llmgateway

# Expose port
EXPOSE 8000

# Set default command
ENTRYPOINT ["python", "-m", "llm_gateway.cli.main"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1