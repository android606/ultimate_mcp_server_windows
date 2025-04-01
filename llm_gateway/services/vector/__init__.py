"""Vector database and embedding operations for LLM Gateway."""
from llm_gateway.services.vector.embeddings import (
    EmbeddingService,
    get_embedding_service,
)
from llm_gateway.services.vector.vector_service import (
    VectorCollection,
    VectorDatabaseService,
    get_vector_db_service,
)

# Create alias for compatibility
get_vector_database_service = get_vector_db_service

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "VectorCollection",
    "VectorDatabaseService",
    "get_vector_db_service",
    "get_vector_database_service",
]