"""Request models for LLM Gateway."""
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator, validator

from llm_gateway.constants import TaskType
from llm_gateway.core.models.entities import Provider


class BaseRequest(BaseModel):
    """Base request model."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    timestamp: float = Field(default_factory=lambda: time.time(), description="Request timestamp")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields


class MessageRole(str, Enum):
    """Message role enumeration."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Chat message model."""
    role: MessageRole = Field(..., description="Message role")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Name of the author for function/tool messages")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call information")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool call information")
    
    @validator("role", pre=True)
    def validate_role(cls, v):
        """Validate and normalize role."""
        if isinstance(v, str):
            try:
                return MessageRole(v.lower())
            except ValueError as e:
                raise ValueError(f"Invalid message role: {v}") from e
        return v


class CompletionRequest(BaseRequest):
    """Text completion request model."""
    prompt: str = Field(..., description="Text prompt for completion")
    model: Optional[str] = Field(None, description="Model to use")
    provider: str = Field(default=Provider.OPENAI.value, description="Provider to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature parameter (0.0-1.0)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    task_type: Optional[str] = Field(default=TaskType.COMPLETION.value, description="Type of task")
    cache: bool = Field(default=True, description="Whether to use cache")
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v


class ChatCompletionRequest(BaseRequest):
    """Chat completion request model."""
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    model: Optional[str] = Field(None, description="Model to use")
    provider: str = Field(default=Provider.OPENAI.value, description="Provider to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature parameter (0.0-1.0)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    system: Optional[str] = Field(None, description="System prompt")
    task_type: Optional[str] = Field(default=TaskType.COMPLETION.value, description="Type of task")
    cache: bool = Field(default=True, description="Whether to use cache")
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v
    
    @validator("messages")
    def validate_messages(cls, v):
        """Validate messages."""
        if not v:
            raise ValueError("At least one message is required")
        return v
    
    @model_validator(mode="before")
    @classmethod
    def check_system_prompt(cls, values):
        """Check system prompt and messages for consistency."""
        messages = values.get("messages", [])
        system = values.get("system")
        
        # If system prompt is provided, ensure it's not also in messages
        if system and messages:
            # Check if there's already a system message
            has_system_message = any(
                msg.role == MessageRole.SYSTEM for msg in messages
            )
            if has_system_message:
                # If system message exists, don't use the separate system parameter
                values["system"] = None
        
        return values


class EmbeddingRequest(BaseRequest):
    """Embedding request model."""
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: Optional[str] = Field(None, description="Model to use")
    provider: str = Field(default=Provider.OPENAI.value, description="Provider to use")
    cache: bool = Field(default=True, description="Whether to use cache")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v


class SummarizationRequest(BaseRequest):
    """Summarization request model."""
    document: str = Field(..., description="Document to summarize")
    model: Optional[str] = Field(None, description="Model to use")
    provider: str = Field(default=Provider.OPENAI.value, description="Provider to use")
    max_length: Optional[int] = Field(None, description="Maximum summary length in words")
    format: str = Field(default="paragraph", description="Summary format (paragraph, bullet_points, key_points)")
    task_type: str = Field(default=TaskType.SUMMARIZATION.value, description="Type of task")
    cache: bool = Field(default=True, description="Whether to use cache")
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v
    
    @validator("format")
    def validate_format(cls, v):
        """Validate format."""
        valid_formats = ["paragraph", "bullet_points", "key_points"]
        if v not in valid_formats:
            raise ValueError(f"Invalid format: {v}. Valid options: {', '.join(valid_formats)}")
        return v


class ChunkingMethod(str, Enum):
    """Chunking method enumeration."""
    TOKEN = "token"
    CHARACTER = "character"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class DocumentRequest(BaseRequest):
    """Document processing request model."""
    document: str = Field(..., description="Document to process")
    operation: str = Field(..., description="Operation to perform (chunk, summarize, extract_entities, generate_qa)")
    model: Optional[str] = Field(None, description="Model to use")
    provider: str = Field(default=Provider.OPENAI.value, description="Provider to use")
    chunk_size: int = Field(default=1000, description="Chunk size for chunking operations")
    chunk_overlap: int = Field(default=100, description="Chunk overlap for chunking operations")
    chunking_method: ChunkingMethod = Field(default=ChunkingMethod.TOKEN, description="Chunking method")
    task_type: Optional[str] = Field(None, description="Type of task")
    cache: bool = Field(default=True, description="Whether to use cache")
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v
    
    @validator("operation")
    def validate_operation(cls, v):
        """Validate operation."""
        valid_operations = ["chunk", "summarize", "extract_entities", "generate_qa"]
        if v not in valid_operations:
            raise ValueError(f"Invalid operation: {v}. Valid options: {', '.join(valid_operations)}")
        return v
    
    @validator("chunking_method", pre=True)
    def validate_chunking_method(cls, v):
        """Validate and normalize chunking method."""
        if isinstance(v, str):
            try:
                return ChunkingMethod(v.lower())
            except ValueError as e:
                raise ValueError(f"Invalid chunking method: {v}") from e
        return v
    
    @model_validator(mode="before")
    @classmethod
    def set_task_type(cls, values):
        """Set task type based on operation if not provided."""
        task_type = values.get("task_type")
        operation = values.get("operation")
        
        if not task_type and operation:
            if operation == "summarize":
                values["task_type"] = TaskType.SUMMARIZATION.value
            elif operation == "extract_entities":
                values["task_type"] = TaskType.EXTRACTION.value
            elif operation == "generate_qa":
                values["task_type"] = TaskType.GENERATION.value
            else:
                values["task_type"] = TaskType.ANALYSIS.value
                
        return values


class ExtractionFormat(str, Enum):
    """Extraction format enumeration."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


class ExtractionRequest(BaseRequest):
    """Data extraction request model."""
    text: str = Field(..., description="Text to extract data from")
    extraction_type: str = Field(..., description="Type of extraction (json, table, key_value_pairs, schema)")
    model: Optional[str] = Field(None, description="Model to use")
    provider: str = Field(default=Provider.OPENAI.value, description="Provider to use")
    schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for structured extraction")
    format: ExtractionFormat = Field(default=ExtractionFormat.JSON, description="Output format")
    task_type: str = Field(default=TaskType.EXTRACTION.value, description="Type of task")
    cache: bool = Field(default=True, description="Whether to use cache")
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v
    
    @validator("extraction_type")
    def validate_extraction_type(cls, v):
        """Validate extraction type."""
        valid_types = ["json", "table", "key_value_pairs", "schema"]
        if v not in valid_types:
            raise ValueError(f"Invalid extraction type: {v}. Valid options: {', '.join(valid_types)}")
        return v
    
    @validator("format", pre=True)
    def validate_format(cls, v):
        """Validate and normalize format."""
        if isinstance(v, str):
            try:
                return ExtractionFormat(v.lower())
            except ValueError as e:
                raise ValueError(f"Invalid format: {v}") from e
        return v