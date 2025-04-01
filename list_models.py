#!/usr/bin/env python3
import asyncio

from llm_gateway.core.server import Gateway


async def list_models():
    gateway = Gateway()
    await gateway._initialize_providers()
    print("Initialized providers")
    
    # Get the models from each provider
    for provider_name, provider in gateway.providers.items():
        print(f"\nProvider: {provider_name}")
        models = await provider.list_models()
        for model in models:
            print(f"  - {provider_name}:{model.get('id', model.get('name', 'unknown'))}")

if __name__ == "__main__":
    asyncio.run(list_models()) 