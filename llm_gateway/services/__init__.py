"""Services for LLM Gateway."""
from llm_gateway.services.cache import (
    CacheService,
    CacheStats,
    get_cache_service,
    with_cache,
)
from llm_gateway.services.prompts import (
    PromptRepository,
    PromptTemplate,
    get_prompt_repository,
    render_prompt,
    render_prompt_template,
)
from llm_gateway.services.vector import (
    VectorCollection,
    VectorDatabaseService,
    get_vector_db_service,
)

# Providers and models
from llm_gateway.services.analytics import get_analytics_service

# Document processing
from llm_gateway.services.document import get_document_processor

# Prompt services
from llm_gateway.services.prompts import get_prompt_service

# Vector services
from llm_gateway.services.vector import get_vector_database_service, get_embedding_service

# RAG services
from llm_gateway.services.knowledge_base import (
    get_knowledge_base_manager,
    get_knowledge_base_retriever
)

__all__ = [
    # Cache service
    "CacheService",
    "CacheStats",
    "get_cache_service",
    "with_cache",
    
    # Vector service
    "VectorCollection", 
    "VectorDatabaseService",
    "get_vector_db_service",
    
    # Prompt service
    "PromptRepository",
    "get_prompt_repository",
    "PromptTemplate",
    "render_prompt",
    "render_prompt_template",

    # Providers and models
    "get_analytics_service",

    # Document processing
    "get_document_processor",

    # Prompt services
    "get_prompt_service",

    # Vector services
    "get_vector_database_service",
    "get_embedding_service",

    # RAG services
    "get_knowledge_base_manager",
    "get_knowledge_base_retriever"
]

# Initialize global service instances
_knowledge_base_manager = None
_knowledge_base_retriever = None
_rag_engine = None

def get_rag_engine():
    """Get or create a RAG engine instance.
    
    Returns:
        RAGEngine: RAG engine instance
    """
    global _rag_engine
    
    if _rag_engine is None:
        from llm_gateway.core import get_provider_manager
        from llm_gateway.services.knowledge_base.rag_engine import RAGEngine
        from llm_gateway.tools.optimization import get_optimization_service
        
        retriever = get_knowledge_base_retriever()
        provider_manager = get_provider_manager()
        optimization_service = get_optimization_service()
        analytics_service = get_analytics_service()
        
        _rag_engine = RAGEngine(
            retriever=retriever,
            provider_manager=provider_manager,
            optimization_service=optimization_service,
            analytics_service=analytics_service
        )
    
    return _rag_engine