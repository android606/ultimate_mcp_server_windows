"""MCP tools for Retrieval-Augmented Generation (RAG)."""
from typing import Any, Dict, List, Optional

from llm_gateway.services import (
    get_knowledge_base_manager,
    get_knowledge_base_retriever,
    get_rag_engine,
)
from llm_gateway.tools.base import BaseTool, tool
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class RAGTools(BaseTool):
    """MCP tools for Retrieval-Augmented Generation (RAG)."""
    
    def __init__(self):
        """Initialize RAG tools."""
        self.kb_manager = get_knowledge_base_manager()
        self.kb_retriever = get_knowledge_base_retriever()
        self.rag_engine = get_rag_engine()
        logger.info("RAG tools initialized", extra={"emoji_key": "success"})
    
    @tool("create_knowledge_base")
    async def create_knowledge_base(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Create a new knowledge base for document storage and retrieval.
        
        Args:
            name: Knowledge base name
            description: Optional description of the knowledge base
            embedding_model: Optional embedding model to use (default is text-embedding-3-small)
            overwrite: Whether to overwrite an existing knowledge base with the same name
            
        Returns:
            Knowledge base creation result
        """
        result = await self.kb_manager.create_knowledge_base(
            name=name,
            description=description,
            embedding_model=embedding_model,
            overwrite=overwrite
        )
        
        return result
    
    @tool("list_knowledge_bases")
    async def list_knowledge_bases(self) -> Dict[str, Any]:
        """List all available knowledge bases.
        
        Returns:
            List of knowledge bases with metadata
        """
        result = await self.kb_manager.list_knowledge_bases()
        return result
    
    @tool("delete_knowledge_base")
    async def delete_knowledge_base(self, name: str) -> Dict[str, Any]:
        """Delete a knowledge base.
        
        Args:
            name: Knowledge base name
            
        Returns:
            Deletion result
        """
        result = await self.kb_manager.delete_knowledge_base(name)
        return result
    
    @tool("add_documents")
    async def add_documents(
        self,
        knowledge_base_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_method: str = "semantic",
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add documents to a knowledge base.
        
        Args:
            knowledge_base_name: Knowledge base name
            documents: List of document texts
            metadatas: Optional list of metadata for each document
            chunk_size: Size of chunks for document splitting
            chunk_overlap: Overlap between chunks
            chunk_method: Method for chunking (token, semantic, sentence)
            embedding_model: Optional embedding model to use
            
        Returns:
            Document addition result
        """
        result = await self.kb_manager.add_documents(
            knowledge_base_name=knowledge_base_name,
            documents=documents,
            metadatas=metadatas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_method=chunk_method,
            embedding_model=embedding_model
        )
        
        return result
    
    @tool("retrieve_context")
    async def retrieve_context(
        self,
        knowledge_base_name: str,
        query: str,
        top_k: int = 5,
        retrieval_method: str = "vector",
        min_score: float = 0.6,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant context from a knowledge base.
        
        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            top_k: Number of results to return
            retrieval_method: Retrieval method (vector, hybrid)
            min_score: Minimum similarity score
            metadata_filter: Optional metadata filter
            
        Returns:
            Retrieved context
        """
        if retrieval_method == "hybrid":
            result = await self.kb_retriever.retrieve_hybrid(
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                min_score=min_score,
                metadata_filter=metadata_filter
            )
        else:
            result = await self.kb_retriever.retrieve(
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                rerank=True,
                min_score=min_score,
                metadata_filter=metadata_filter
            )
        
        return result
    
    @tool("generate_with_rag")
    async def generate_with_rag(
        self,
        knowledge_base_name: str,
        query: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        template: str = "rag_default",
        max_tokens: int = 1000,
        temperature: float = 0.3,
        top_k: int = 5,
        retrieval_method: str = "vector",
        min_score: float = 0.6,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Generate a response using Retrieval-Augmented Generation (RAG).
        
        Args:
            knowledge_base_name: Knowledge base name
            query: User query
            provider: LLM provider (auto-selected if None)
            model: LLM model (auto-selected if None)
            template: RAG prompt template (rag_default, rag_with_sources, rag_summarize, rag_analysis)
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            top_k: Number of context documents to retrieve
            retrieval_method: Retrieval method (vector, hybrid)
            min_score: Minimum similarity score for retrieved documents
            include_sources: Whether to include sources in response
            
        Returns:
            Generated response with sources and metrics
        """
        result = await self.rag_engine.generate_with_rag(
            knowledge_base_name=knowledge_base_name,
            query=query,
            provider=provider,
            model=model,
            template=template,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            retrieval_method=retrieval_method,
            min_score=min_score,
            include_sources=include_sources
        )
        
        return result 