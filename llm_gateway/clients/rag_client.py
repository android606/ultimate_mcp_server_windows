"""High-level client for RAG (Retrieval-Augmented Generation) operations."""

from typing import Any, Dict, List, Optional

from llm_gateway.services.knowledge_base import (
    get_knowledge_base_manager,
    get_knowledge_base_retriever,
    get_rag_service,
)
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.clients.rag")

class RAGClient:
    """High-level client for RAG operations.
    
    This client provides a simplified interface for common RAG operations
    including knowledge base management, retrieval, and generation.
    """
    
    def __init__(self):
        """Initialize the RAG client."""
        self.kb_manager = get_knowledge_base_manager()
        self.kb_retriever = get_knowledge_base_retriever()
        self.rag_service = get_rag_service()
    
    async def create_knowledge_base(
        self,
        name: str,
        description: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Create a knowledge base.
        
        Args:
            name: The name of the knowledge base
            description: Optional description
            overwrite: Whether to overwrite an existing KB with the same name
            
        Returns:
            Result of the operation
        """
        logger.info(f"Creating knowledge base: {name}", emoji_key="processing")
        
        try:
            result = await self.kb_manager.create_knowledge_base(
                name=name,
                description=description,
                overwrite=overwrite
            )
            
            logger.success(f"Knowledge base created: {name}", emoji_key="success")
            return result
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {str(e)}", emoji_key="error")
            raise
    
    async def add_documents(
        self,
        knowledge_base_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_method: str = "semantic"
    ) -> Dict[str, Any]:
        """Add documents to the knowledge base.
        
        Args:
            knowledge_base_name: Name of the knowledge base to add to
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            chunk_size: Size of chunks to split documents into
            chunk_method: Method to use for chunking ('simple', 'semantic', etc.)
            
        Returns:
            Result of the operation
        """
        logger.info(f"Adding documents to knowledge base: {knowledge_base_name}", emoji_key="processing")
        
        try:
            result = await self.kb_manager.add_documents(
                knowledge_base_name=knowledge_base_name,
                documents=documents,
                metadatas=metadatas,
                chunk_size=chunk_size,
                chunk_method=chunk_method
            )
            
            added_count = result.get("added_count", 0)
            logger.success(f"Added {added_count} documents to knowledge base", emoji_key="success")
            return result
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}", emoji_key="error")
            raise
    
    async def list_knowledge_bases(self) -> List[Any]:
        """List all knowledge bases.
        
        Returns:
            List of knowledge base information
        """
        logger.info("Retrieving list of knowledge bases", emoji_key="processing")
        
        try:
            knowledge_bases = await self.kb_manager.list_knowledge_bases()
            return knowledge_bases
        except Exception as e:
            logger.error(f"Failed to list knowledge bases: {str(e)}", emoji_key="error")
            raise
    
    async def retrieve(
        self,
        knowledge_base_name: str,
        query: str,
        top_k: int = 3,
        retrieval_method: str = "vector"
    ) -> Dict[str, Any]:
        """Retrieve context for a given query.
        
        Args:
            knowledge_base_name: Name of the knowledge base to query
            query: The query text
            top_k: Number of results to retrieve
            retrieval_method: Method to use for retrieval ("vector", "hybrid", etc.)
            
        Returns:
            Retrieval results
        """
        logger.info(f"Retrieving context for query: '{query}'", emoji_key="processing")
        
        try:
            results = await self.kb_retriever.retrieve(
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                retrieval_method=retrieval_method
            )
            
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}", emoji_key="error")
            raise
    
    async def generate_with_rag(
        self,
        knowledge_base_name: str,
        query: str,
        provider: str = "gemini",
        model: Optional[str] = None,
        template: str = "rag_default",
        temperature: float = 0.3,
        top_k: int = 3,
        retrieval_method: str = "hybrid",
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Generate a response using RAG.
        
        Args:
            knowledge_base_name: Name of the knowledge base to use
            query: The query text
            provider: LLM provider to use
            model: Model to use (if None, uses provider default)
            template: Prompt template to use
            temperature: Sampling temperature
            top_k: Number of documents to retrieve
            retrieval_method: Method to use for retrieval
            include_sources: Whether to include source information
            
        Returns:
            RAG generation result
        """
        logger.info(f"Generating RAG response for: '{query}'", emoji_key="processing")
        
        try:
            result = await self.rag_service.generate_with_rag(
                knowledge_base_name=knowledge_base_name,
                query=query,
                provider=provider,
                model=model,
                template=template,
                temperature=temperature,
                top_k=top_k,
                retrieval_method=retrieval_method,
                include_sources=include_sources
            )
            
            return result
        except Exception as e:
            logger.error(f"Failed to call RAG service: {str(e)}", emoji_key="error")
            raise
    
    async def delete_knowledge_base(self, name: str) -> Dict[str, Any]:
        """Delete a knowledge base.
        
        Args:
            name: Name of the knowledge base to delete
            
        Returns:
            Result of the operation
        """
        logger.info(f"Deleting knowledge base: {name}", emoji_key="processing")
        
        try:
            result = await self.kb_manager.delete_knowledge_base(name=name)
            logger.success(f"Knowledge base {name} deleted successfully", emoji_key="success")
            return result
        except Exception as e:
            logger.error(f"Failed to delete knowledge base: {str(e)}", emoji_key="error")
            raise 