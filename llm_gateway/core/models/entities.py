"""Entity models for LLM Gateway."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from llm_gateway.constants import Provider as ProviderEnum


class Provider(str, Enum):
    """Provider enumeration."""
    OPENAI = ProviderEnum.OPENAI.value
    ANTHROPIC = ProviderEnum.ANTHROPIC.value
    DEEPSEEK = ProviderEnum.DEEPSEEK.value
    GEMINI = ProviderEnum.GEMINI.value
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if provider name is valid.
        
        Args:
            value: Provider name to check
            
        Returns:
            Whether the provider name is valid
        """
        try:
            cls(value.lower())
            return True
        except ValueError:
            return False


class TokenUsage(BaseModel):
    """Token usage information."""
    input_tokens: int = Field(default=0, description="Number of input tokens")
    output_tokens: int = Field(default=0, description="Number of output tokens")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")
    
    @validator("total_tokens", always=True)
    def set_total_tokens(cls, v, values):
        """Set total tokens if not provided."""
        if v is None:
            return values.get("input_tokens", 0) + values.get("output_tokens", 0)
        return v
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens or (self.input_tokens + self.output_tokens)
        }


class ModelMetadata(BaseModel):
    """Metadata for an LLM model."""
    id: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Provider name")
    description: Optional[str] = Field(None, description="Model description")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens supported")
    created: Optional[int] = Field(None, description="Model creation timestamp")
    owned_by: Optional[str] = Field(None, description="Model owner")
    capabilities: Optional[List[str]] = Field(None, description="Model capabilities")
    pricing: Optional[Dict[str, float]] = Field(None, description="Pricing information")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional provider-specific information")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "provider": self.provider,
            "description": self.description,
            "max_tokens": self.max_tokens,
            "created": self.created,
            "owned_by": self.owned_by,
            "capabilities": self.capabilities,
            "pricing": self.pricing,
            "additional_info": self.additional_info,
        }


class LLMModel(BaseModel):
    """Model information."""
    id: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Provider name")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v
    
    def __str__(self) -> str:
        return f"{self.provider}/{self.id}"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "provider": self.provider,
        }


class ProviderConfig(BaseModel):
    """Provider configuration."""
    enabled: bool = Field(default=True, description="Whether the provider is enabled")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL for API requests")
    organization: Optional[str] = Field(None, description="Organization identifier")
    default_model: Optional[str] = Field(None, description="Default model to use")
    max_tokens: Optional[int] = Field(None, description="Default maximum tokens for completions")
    timeout: Optional[float] = Field(None, description="Timeout for API requests in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary without sensitive information.
        
        Returns:
            Dictionary representation
        """
        result = self.dict(exclude={"api_key"})
        # Add API key presence indicator
        result["api_key_configured"] = bool(self.api_key)
        return result


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    id: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Provider name")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for completions")
    temperature: Optional[float] = Field(None, description="Default temperature")
    timeout: Optional[float] = Field(None, description="Timeout for API requests in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model-specific parameters")
    
    @validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if not Provider.is_valid(v):
            raise ValueError(f"Invalid provider: {v}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "additional_params": self.additional_params,
        }