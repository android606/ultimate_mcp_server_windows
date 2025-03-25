#!/bin/bash
# Fix environment variables by removing comments

# Cache settings
export CACHE_ENABLED="true"
export CACHE_TTL="86400"
export CACHE_DIR=".cache"
export CACHE_MAX_ENTRIES="10000"
export CACHE_FUZZY_MATCH="true"

# Logging settings
export LOG_LEVEL="INFO"
export LOG_FILE="logs/llm_gateway.log"
export USE_RICH_LOGGING="true"

# Server settings
export SERVER_NAME="LLM Gateway"
export SERVER_PORT="8019"
export SERVER_HOST="0.0.0.0"
export SERVER_WORKERS="4"
export SERVER_DEBUG="false"

# Provider settings
# Keep existing API keys
export OPENAI_DEFAULT_MODEL="gpt-4o-mini"
export ANTHROPIC_DEFAULT_MODEL="claude-3-5-haiku-latest"
export DEEPSEEK_DEFAULT_MODEL="deepseek-chat"
export GEMINI_DEFAULT_MODEL="gemini-2.0-flash-lite"
export OPENAI_MAX_TOKENS="8192"
export ANTHROPIC_MAX_TOKENS="200000"
export DEEPSEEK_MAX_TOKENS="8192"
export GEMINI_MAX_TOKENS="8192"

# Embedding settings
export EMBEDDING_CACHE_DIR=".embeddings"
export EMBEDDING_DEFAULT_MODEL="text-embedding-3-small"

# Request settings
export REQUEST_TIMEOUT="60"
export RATE_LIMIT_ENABLED="false"
export MAX_CONCURRENT_REQUESTS="20"

echo "Environment variables fixed. Now run the example:"
echo "source fix_env.sh && python examples/document_processing.py" 