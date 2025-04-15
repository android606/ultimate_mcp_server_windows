"""Services for LLM Gateway."""
# Keep only high-level service getters or core classes if needed
# from llm_gateway.services.analytics import get_analytics_service # Keep if used by other services here
# from llm_gateway.services.cache import CacheService, get_cache_service, with_cache # Keep if used by other services here
# from llm_gateway.services.document import get_document_processor # Keep if used by other services here
# from llm_gateway.services.knowledge_base import get_knowledge_base_manager, get_knowledge_base_retriever # REMOVE THESE
# from llm_gateway.services.prompts import PromptRepository, PromptTemplate, get_prompt_repository, get_prompt_service, render_prompt, render_prompt_template # Keep if used by other services here
# from llm_gateway.services.vector import VectorDatabaseService, get_embedding_service, get_vector_db_service # Keep if used by other services here

# Example: Only keep get_analytics_service if get_rag_engine needs it directly
from llm_gateway.services.analytics import get_analytics_service

# __all__ should only export symbols defined *in this file* or truly essential high-level interfaces
# Avoid re-exporting everything from submodules.
__all__ = [
    "get_analytics_service", # Only keep if it's fundamental/used here
    "get_rag_engine",        # get_rag_engine is defined below
    # Remove others unless absolutely necessary for this top-level init
    # "CacheService",
    # "get_cache_service",
    # "with_cache",
    # "VectorDatabaseService",
    # "get_vector_db_service",
    # "PromptRepository",
    # "get_prompt_repository",
    # "PromptTemplate",
    # "get_document_processor",
    # "get_prompt_service",
    # "get_embedding_service",
    # "get_knowledge_base_manager",
    # "get_knowledge_base_retriever"
]

# Initialize global service instances
# _knowledge_base_manager = None # Move initialization logic if needed
# _knowledge_base_retriever = None # Move initialization logic if needed
_rag_engine = None

def get_rag_engine():
    """Get or create a RAG engine instance.
    
    Returns:
        RAGEngine: RAG engine instance
    """
    global _rag_engine
    
    if _rag_engine is None:
        # Import dependencies *inside* the function to avoid top-level cycles
        from llm_gateway.core import get_provider_manager  # Assuming this doesn't import services
        from llm_gateway.services.knowledge_base import (
            get_knowledge_base_retriever,  # Import KB retriever here
        )
        from llm_gateway.services.knowledge_base.rag_engine import RAGEngine

        # Assuming OptimizationTools doesn't create cycles with services
        # This might need further investigation if OptimizationTools imports services
        from llm_gateway.tools.optimization import get_optimization_service
        
        # analytics_service is already imported at top-level
        # retriever = get_knowledge_base_retriever()
        # provider_manager = get_provider_manager()
        # optimization_service = get_optimization_service()
        # analytics_service = get_analytics_service()
        
        _rag_engine = RAGEngine(
            retriever=get_knowledge_base_retriever(),
            provider_manager=get_provider_manager(),
            optimization_service=get_optimization_service(),
            analytics_service=get_analytics_service() # Imported at top
        )
    
    return _rag_engine