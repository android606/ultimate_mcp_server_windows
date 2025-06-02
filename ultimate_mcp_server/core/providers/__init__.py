"""Provider module for Ultimate MCP Server.

This module provides access to LLM providers and provider-specific functionality.
"""

from typing import Dict, Type, Optional, Any
import sys
import importlib

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import BaseProvider

class LazyProviderImporter:
    """Lazy importer for provider classes to avoid heavy startup dependencies."""
    
    def __init__(self, module_path: str, class_name: str):
        self.module_path = module_path
        self.class_name = class_name
        self._cached_class = None
    
    def get_provider_class(self) -> Type[BaseProvider]:
        """Get the provider class, importing only when needed."""
        if self._cached_class is None:
            try:
                module = importlib.import_module(self.module_path)
                self._cached_class = getattr(module, self.class_name)
            except ImportError as e:
                # If import fails, create a stub provider that shows the error
                from ultimate_mcp_server.utils import get_logger
                logger = get_logger("providers")
                logger.warning(f"Failed to import {self.class_name} from {self.module_path}: {e}")
                
                # Capture the error string before creating the class
                import_error_str = str(e)
                provider_name_str = self.class_name.replace("Provider", "").lower()
                
                # Create a factory function to create the stub class with captured values
                def create_unavailable_provider():
                    class UnavailableProvider(BaseProvider):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self._import_error = import_error_str
                            self.provider_name = provider_name_str
                        
                        async def initialize(self) -> bool:
                            self.logger.error(f"Provider {self.provider_name} is unavailable: {self._import_error}")
                            return False
                        
                        async def generate_completion(self, *args, **kwargs):
                            raise ImportError(f"Provider {self.provider_name} is unavailable: {self._import_error}")
                        
                        async def generate_completion_stream(self, *args, **kwargs):
                            raise ImportError(f"Provider {self.provider_name} is unavailable: {self._import_error}")
                        
                        async def list_models(self):
                            return []
                    
                    return UnavailableProvider
                
                self._cached_class = create_unavailable_provider()
        
        return self._cached_class

# Lazy provider registry - providers are only imported when first accessed
_LAZY_PROVIDERS = {
    Provider.OPENAI.value: LazyProviderImporter("ultimate_mcp_server.core.providers.openai", "OpenAIProvider"),
    Provider.ANTHROPIC.value: LazyProviderImporter("ultimate_mcp_server.core.providers.anthropic", "AnthropicProvider"),
    Provider.DEEPSEEK.value: LazyProviderImporter("ultimate_mcp_server.core.providers.deepseek", "DeepSeekProvider"),
    Provider.GEMINI.value: LazyProviderImporter("ultimate_mcp_server.core.providers.gemini", "GeminiProvider"),
    Provider.OPENROUTER.value: LazyProviderImporter("ultimate_mcp_server.core.providers.openrouter", "OpenRouterProvider"),
    Provider.GROK.value: LazyProviderImporter("ultimate_mcp_server.core.providers.grok", "GrokProvider"),
    Provider.OLLAMA.value: LazyProviderImporter("ultimate_mcp_server.core.providers.ollama", "OllamaProvider"),
    Provider.TOGETHER.value: LazyProviderImporter("ultimate_mcp_server.core.providers.together", "TogetherProvider"),
}

class LazyProviderRegistry:
    """Registry that provides lazy access to provider classes."""
    
    def __init__(self):
        self._cached_registry: Dict[str, Type[BaseProvider]] = {}
    
    def get(self, provider_name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name, loading it lazily."""
        if provider_name not in self._cached_registry:
            if provider_name in _LAZY_PROVIDERS:
                self._cached_registry[provider_name] = _LAZY_PROVIDERS[provider_name].get_provider_class()
            else:
                return None
        return self._cached_registry[provider_name]
    
    def keys(self):
        """Get all available provider names."""
        return _LAZY_PROVIDERS.keys()
    
    def items(self):
        """Get all provider items (lazy loaded)."""
        for name in _LAZY_PROVIDERS:
            yield name, self.get(name)
    
    def values(self):
        """Get all provider classes (lazy loaded)."""
        for name in _LAZY_PROVIDERS:
            yield self.get(name)
    
    def __contains__(self, item):
        """Check if provider exists."""
        return item in _LAZY_PROVIDERS
    
    def __getitem__(self, item):
        """Get provider by name."""
        result = self.get(item)
        if result is None:
            raise KeyError(f"Provider '{item}' not found")
        return result

# Global lazy provider registry
PROVIDER_REGISTRY = LazyProviderRegistry()

__all__ = ["PROVIDER_REGISTRY"]