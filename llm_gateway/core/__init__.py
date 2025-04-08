"""Core functionality for LLM Gateway."""
import asyncio
from typing import Optional

from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger

logger = get_logger(__name__)

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
                logger.warning("Gateway instance requested before async init completed.")
                return {}
            else:
                # Run in a new event loop (blocks)
                logger.info("Synchronously initializing gateway for get_provider_manager.")
                _gateway_instance = Gateway("provider-manager")
                loop.run_until_complete(_gateway_instance._initialize_providers())
        except RuntimeError:
            # No event loop running, create one (blocks)
            logger.info("Synchronously initializing gateway for get_provider_manager (new loop).")
            _gateway_instance = Gateway("provider-manager")
            asyncio.run(_gateway_instance._initialize_providers())
    
    # Return the providers dictionary as a "manager"
    return _gateway_instance.providers if _gateway_instance else {}

def get_gateway_instance() -> Optional[Gateway]:
    """Synchronously get the initialized gateway instance.
    
    Returns:
        The Gateway instance or None if it hasn't been initialized yet.
    """
    global _gateway_instance
    if _gateway_instance is None:
        logger.warning("get_gateway_instance() called before instance was initialized.")
    return _gateway_instance

__all__ = ["Gateway", "get_provider_manager"]