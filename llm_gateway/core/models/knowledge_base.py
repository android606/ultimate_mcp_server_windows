"""Models for knowledge base management."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeBaseMetadata(BaseModel):
    """Metadata for a knowledge base."""
    id: str = Field(..., description="ID of the knowledge base")
    name: str = Field(..., description="Name of the knowledge base")
    description: Optional[str] = Field(None, description="Description of the knowledge base")
    document_count: int = Field(..., description="Number of documents in the knowledge base")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    embedding_model: str = Field(..., description="Embedding model used")
    owners: List[str] = Field(default_factory=list, description="List of owner IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentSource(BaseModel):
    """Source information for a document in RAG results."""
    id: str = Field(..., description="Document ID")
    document: str = Field(..., description="Document content preview")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class RetrievalMethod(str, Enum):
    """Retrieval methods for RAG."""
    VECTOR = "vector"
    HYBRID = "hybrid"


class FeedbackType(str, Enum):
    """Types of feedback for RAG."""
    EXPLICIT = "explicit"  # User provided explicit feedback
    IMPLICIT = "implicit"  # System inferred feedback
    CORRECTION = "correction"  # User provided correction


class RAGRequest(BaseModel):
    """Request for RAG generation."""
    query: str = Field(..., description="Query text")
    provider: Optional[str] = Field(None, description="Provider name (auto-selected if None)")
    model: Optional[str] = Field(None, description="Model name (auto-selected if None)")
    template: Optional[str] = Field(None, description="RAG prompt template name")
    max_tokens: int = Field(1000, description="Maximum tokens for generation")
    temperature: float = Field(0.3, description="Temperature for generation")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve")
    retrieval_method: Optional[RetrievalMethod] = Field(None, description="Retrieval method")
    min_score: Optional[float] = Field(None, description="Minimum similarity score")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    include_metadata: bool = Field(True, description="Whether to include metadata in context")
    include_sources: bool = Field(True, description="Whether to include sources in response")
    use_cache: bool = Field(True, description="Whether to use cached responses")
    apply_feedback: bool = Field(True, description="Whether to apply feedback adjustments")
    search_params: Optional[Dict[str, Any]] = Field(None, description="Search parameters")
    
    class Config:
        use_enum_values = True


class RAGResponse(BaseModel):
    """Response from RAG generation."""
    status: str = Field(..., description="Status of the operation")
    query: str = Field(..., description="Original query")
    answer: Optional[str] = Field(None, description="Generated answer")
    sources: Optional[List[DocumentSource]] = Field(None, description="Document sources")
    knowledge_base: str = Field(..., description="Knowledge base name")
    provider: Optional[str] = Field(None, description="Provider used")
    model: Optional[str] = Field(None, description="Model used")
    used_document_ids: Optional[List[str]] = Field(None, description="Document IDs used in the response")
    message: Optional[str] = Field(None, description="Optional message (e.g., for errors)")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Operation metrics")


class RAGFeedbackRequest(BaseModel):
    """Request for providing feedback on RAG results."""
    query: str = Field(..., description="Original query")
    document_id: str = Field(..., description="Document ID")
    is_relevant: bool = Field(..., description="Whether the document is relevant")
    feedback_type: FeedbackType = Field(FeedbackType.EXPLICIT, description="Type of feedback")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    class Config:
        use_enum_values = True 