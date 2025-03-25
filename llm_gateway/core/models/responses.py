"""Response models for LLM Gateway."""
import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator

from llm_gateway.core.models.entities import TokenUsage


class BaseResponse(BaseModel):
    """Base response model."""
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields


class ErrorResponse(BaseResponse):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type")
    provider: Optional[str] = Field(None, description="Provider name")
    model: Optional[str] = Field(None, description="Model name")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "error": self.error,
            "error_type": self.error_type,
            "provider": self.provider,
            "model": self.model,
            "details": self.details,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
        }


class CompletionResponse(BaseResponse):
    """Text completion response model."""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    tokens: TokenUsage = Field(default_factory=TokenUsage, description="Token usage information")
    cost: float = Field(default=0.0, description="Estimated cost in USD")
    finish_reason: Optional[str] = Field(None, description="Reason for completion finish")
    cached: bool = Field(default=False, description="Whether the response was from cache")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional provider-specific data")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "processing_time": self.processing_time,
            "cached": self.cached,
        }
        
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
            
        if self.additional_data:
            result["additional_data"] = self.additional_data
            
        return result


class ChatCompletionResponse(BaseResponse):
    """Chat completion response model."""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    tokens: TokenUsage = Field(default_factory=TokenUsage, description="Token usage information")
    cost: float = Field(default=0.0, description="Estimated cost in USD")
    finish_reason: Optional[str] = Field(None, description="Reason for completion finish")
    cached: bool = Field(default=False, description="Whether the response was from cache")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call information")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool call information")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional provider-specific data")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "processing_time": self.processing_time,
            "cached": self.cached,
        }
        
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
            
        if self.function_call:
            result["function_call"] = self.function_call
            
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
            
        if self.additional_data:
            result["additional_data"] = self.additional_data
            
        return result


class StreamChunk(BaseModel):
    """Streaming chunk model."""
    text: str = Field(..., description="Chunk text")
    chunk_index: int = Field(..., description="Chunk index")
    finish_reason: Optional[str] = Field(None, description="Reason for completion finish")
    finished: bool = Field(default=False, description="Whether this is the final chunk")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "finished": self.finished,
        }
        
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
            
        return result


class EmbeddingResponse(BaseResponse):
    """Embedding response model."""
    embeddings: Union[List[List[float]], List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    tokens: TokenUsage = Field(default_factory=TokenUsage, description="Token usage information")
    cost: float = Field(default=0.0, description="Estimated cost in USD")
    cached: bool = Field(default=False, description="Whether the response was from cache")
    dimensions: int = Field(..., description="Number of dimensions in the embeddings")
    
    @root_validator(pre=True)
    def set_dimensions(cls, values):
        """Set dimensions based on embeddings."""
        embeddings = values.get("embeddings")
        if embeddings:
            if isinstance(embeddings[0], list):
                values["dimensions"] = len(embeddings[0])
            else:
                values["dimensions"] = len(embeddings)
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "embeddings": self.embeddings,
            "model": self.model,
            "provider": self.provider,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "processing_time": self.processing_time,
            "cached": self.cached,
            "dimensions": self.dimensions,
        }


class SummarizationResponse(BaseResponse):
    """Summarization response model."""
    summary: str = Field(..., description="Generated summary")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    tokens: TokenUsage = Field(default_factory=TokenUsage, description="Token usage information")
    cost: float = Field(default=0.0, description="Estimated cost in USD")
    format: str = Field(..., description="Summary format")
    cached: bool = Field(default=False, description="Whether the response was from cache")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "summary": self.summary,
            "model": self.model,
            "provider": self.provider,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "format": self.format,
            "processing_time": self.processing_time,
            "cached": self.cached,
        }


class DocumentChunk(BaseModel):
    """Document chunk model."""
    text: str = Field(..., description="Chunk text")
    index: int = Field(..., description="Chunk index")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "text": self.text,
            "index": self.index,
            "metadata": self.metadata or {},
        }


class DocumentResponse(BaseResponse):
    """Document processing response model."""
    operation: str = Field(..., description="Operation performed")
    result: Dict[str, Any] = Field(..., description="Operation result")
    model: Optional[str] = Field(None, description="Model used")
    provider: Optional[str] = Field(None, description="Provider used")
    tokens: Optional[TokenUsage] = Field(None, description="Token usage information")
    cost: float = Field(default=0.0, description="Estimated cost in USD")
    cached: bool = Field(default=False, description="Whether the response was from cache")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            "operation": self.operation,
            "result": self.result,
            "processing_time": self.processing_time,
            "cached": self.cached,
            "cost": self.cost,
        }
        
        if self.model:
            result["model"] = self.model
            
        if self.provider:
            result["provider"] = self.provider
            
        if self.tokens:
            result["tokens"] = self.tokens.to_dict()
            
        return result


class ExtractionResponse(BaseResponse):
    """Data extraction response model."""
    data: Dict[str, Any] = Field(..., description="Extracted data")
    extraction_type: str = Field(..., description="Type of extraction")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    tokens: TokenUsage = Field(default_factory=TokenUsage, description="Token usage information")
    cost: float = Field(default=0.0, description="Estimated cost in USD")
    format: str = Field(..., description="Output format")
    cached: bool = Field(default=False, description="Whether the response was from cache")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "data": self.data,
            "extraction_type": self.extraction_type,
            "model": self.model,
            "provider": self.provider,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "format": self.format,
            "processing_time": self.processing_time,
            "cached": self.cached,
        }