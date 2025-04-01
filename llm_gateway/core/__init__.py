"""Core functionality for LLM Gateway."""
import asyncio

from llm_gateway.core.server import Gateway

# Add a provider manager getter function
_gateway_instance = None

async def async_init_gateway():
    """Asynchronously initialize gateway."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = Gateway("provider-manager")
        await _gateway_instance._initialize_providers()
    return _gateway_instance

def get_provider_manager():
    """Get the provider manager from the Gateway instance.
    
    Returns:
        Provider manager with initialized providers
    """
    global _gateway_instance
    
    if _gateway_instance is None:
        try:
            # Try to run in current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task in the current event loop
                asyncio.create_task(async_init_gateway())
            else:
                # Run in a new event loop
                _gateway_instance = Gateway("provider-manager")
                loop.run_until_complete(_gateway_instance._initialize_providers())
        except RuntimeError:
            # No event loop running, create one
            _gateway_instance = Gateway("provider-manager")
            asyncio.run(_gateway_instance._initialize_providers())
    
    # Return the providers dictionary as a "manager"
    return _gateway_instance.providers if _gateway_instance else {}

__all__ = ["Gateway", "get_provider_manager"]