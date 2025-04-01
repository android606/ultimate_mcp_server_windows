"""Knowledge base message handlers for MCP server."""
from typing import Any, Dict, List, Optional

from llm_gateway.core.models.knowledge_base import (
    KnowledgeBaseMetadata,
    RAGFeedbackRequest,
    RAGRequest,
    RAGResponse,
)
from llm_gateway.services.knowledge_base import get_rag_feedback_service, get_rag_service
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


async def handle_list_knowledge_bases(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle knowledge base listing request.
    
    Args:
        message: MCP message
        
    Returns:
        Response message
    """
    try:
        rag_service = get_rag_service()
        knowledge_bases = await rag_service.list_knowledge_bases()
        return {
            "status": "success",
            "knowledge_bases": [kb.dict() for kb in knowledge_bases]
        }
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        return {
            "status": "error", 
            "error": f"Failed to list knowledge bases: {str(e)}"
        }


async def handle_rag_query(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle RAG query request.
    
    Args:
        message: MCP message
        
    Returns:
        Response message
    """
    try:
        rag_service = get_rag_service()
        
        # Extract parameters from message
        knowledge_base_id = message.get("knowledge_base_id")
        if not knowledge_base_id:
            return {"status": "error", "error": "Missing knowledge_base_id"}
            
        # Convert message to request model
        params = message.get("params", {})
        rag_request = RAGRequest(
            query=params.get("query", ""),
            provider=params.get("provider"),
            model=params.get("model"),
            template=params.get("template"),
            max_tokens=params.get("max_tokens", 1000),
            temperature=params.get("temperature", 0.3),
            top_k=params.get("top_k"),
            retrieval_method=params.get("retrieval_method"),
            min_score=params.get("min_score"),
            metadata_filter=params.get("metadata_filter"),
            include_metadata=params.get("include_metadata", True),
            include_sources=params.get("include_sources", True),
            use_cache=params.get("use_cache", True),
            apply_feedback=params.get("apply_feedback", True),
            search_params=params.get("search_params")
        )
        
        # Generate response using RAG
        result = await rag_service.generate_with_rag(
            knowledge_base_name=knowledge_base_id,
            query=rag_request.query,
            provider=rag_request.provider,
            model=rag_request.model,
            template=rag_request.template or "rag_default",
            max_tokens=rag_request.max_tokens,
            temperature=rag_request.temperature,
            top_k=rag_request.top_k,
            retrieval_method=rag_request.retrieval_method,
            min_score=rag_request.min_score,
            metadata_filter=rag_request.metadata_filter,
            include_metadata=rag_request.include_metadata,
            include_sources=rag_request.include_sources,
            use_cache=rag_request.use_cache,
            apply_feedback=rag_request.apply_feedback,
            search_params=rag_request.search_params
        )
        
        return result
    except Exception as e:
        logger.error(f"Error querying knowledge base: {str(e)}")
        return {
            "status": "error", 
            "error": f"Failed to query knowledge base: {str(e)}"
        }


async def handle_rag_feedback(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle RAG feedback request.
    
    Args:
        message: MCP message
        
    Returns:
        Response message
    """
    try:
        feedback_service = get_rag_feedback_service()
        
        # Extract parameters from message
        knowledge_base_id = message.get("knowledge_base_id")
        if not knowledge_base_id:
            return {"status": "error", "error": "Missing knowledge_base_id"}
            
        # Convert message to request model
        params = message.get("params", {})
        feedback_request = RAGFeedbackRequest(
            query=params.get("query", ""),
            document_id=params.get("document_id", ""),
            is_relevant=params.get("is_relevant", False),
            feedback_type=params.get("feedback_type", "explicit"),
            notes=params.get("notes")
        )
        
        # Record feedback
        await feedback_service.record_retrieval_feedback(
            knowledge_base_name=knowledge_base_id,
            query=feedback_request.query,
            document_id=feedback_request.document_id,
            is_relevant=feedback_request.is_relevant,
            feedback_type=feedback_request.feedback_type
        )
        
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        return {
            "status": "error", 
            "error": f"Failed to record feedback: {str(e)}"
        }


async def handle_retrieve_documents(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document retrieval request.
    
    Args:
        message: MCP message
        
    Returns:
        Response message
    """
    try:
        rag_service = get_rag_service()
        
        # Extract parameters from message
        knowledge_base_id = message.get("knowledge_base_id")
        if not knowledge_base_id:
            return {"status": "error", "error": "Missing knowledge_base_id"}
            
        # Extract query parameters
        params = message.get("params", {})
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        min_score = params.get("min_score", 0.6)
        metadata_filter = params.get("metadata_filter")
        content_filter = params.get("content_filter")
        apply_feedback = params.get("apply_feedback", True)
        search_params = params.get("search_params")
        
        # Retrieve documents
        result = await rag_service.retriever.retrieve(
            knowledge_base_name=knowledge_base_id,
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
            content_filter=content_filter,
            apply_feedback=apply_feedback,
            search_params=search_params
        )
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {
            "status": "error", 
            "error": f"Failed to retrieve documents: {str(e)}"
        }


async def handle_retrieve_hybrid(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle hybrid document retrieval request.
    
    Args:
        message: MCP message
        
    Returns:
        Response message
    """
    try:
        rag_service = get_rag_service()
        
        # Extract parameters from message
        knowledge_base_id = message.get("knowledge_base_id")
        if not knowledge_base_id:
            return {"status": "error", "error": "Missing knowledge_base_id"}
            
        # Extract query parameters
        params = message.get("params", {})
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        min_score = params.get("min_score", 0.6)
        metadata_filter = params.get("metadata_filter")
        content_filter = params.get("content_filter")
        apply_feedback = params.get("apply_feedback", True)
        search_params = params.get("search_params")
        
        # Retrieve documents using hybrid search
        result = await rag_service.retriever.retrieve_hybrid(
            knowledge_base_name=knowledge_base_id,
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
            content_filter=content_filter,
            apply_feedback=apply_feedback,
            search_params=search_params
        )
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving documents with hybrid search: {str(e)}")
        return {
            "status": "error", 
            "error": f"Failed to retrieve documents with hybrid search: {str(e)}"
        } 