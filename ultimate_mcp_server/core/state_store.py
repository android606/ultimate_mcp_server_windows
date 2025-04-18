import asyncio
import os
import pickle
from typing import Any, Dict, Optional

import aiofiles


class StateStore:
    """Thread-safe, async-compatible state management for tools."""
    
    def __init__(self, persistence_dir: Optional[str] = None):
        self._in_memory_store: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._persistence_dir = persistence_dir
        if persistence_dir and not os.path.exists(persistence_dir):
            os.makedirs(persistence_dir)
    
    def _get_lock(self, namespace: str) -> asyncio.Lock:
        """Get or create a lock for a namespace."""
        if namespace not in self._locks:
            self._locks[namespace] = asyncio.Lock()
        return self._locks[namespace]
    
    async def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get a value from the store."""
        async with self._get_lock(namespace):
            if namespace not in self._in_memory_store:
                # Try to load from disk if persistence is enabled
                if self._persistence_dir:
                    await self._load_namespace(namespace)
                else:
                    self._in_memory_store[namespace] = {}
            
            return self._in_memory_store[namespace].get(key, default)
    
    async def set(self, namespace: str, key: str, value: Any) -> None:
        """Set a value in the store."""
        async with self._get_lock(namespace):
            if namespace not in self._in_memory_store:
                self._in_memory_store[namespace] = {}
            
            self._in_memory_store[namespace][key] = value
            
            # Persist immediately if enabled
            if self._persistence_dir:
                await self._persist_namespace(namespace)
    
    async def delete(self, namespace: str, key: str) -> None:
        """Delete a value from the store."""
        async with self._get_lock(namespace):
            if namespace in self._in_memory_store and key in self._in_memory_store[namespace]:
                del self._in_memory_store[namespace][key]
                
                # Persist the change if enabled
                if self._persistence_dir:
                    await self._persist_namespace(namespace)
    
    async def _persist_namespace(self, namespace: str) -> None:
        """Persist a namespace to disk."""
        if not self._persistence_dir:
            return
            
        file_path = os.path.join(self._persistence_dir, f"{namespace}.pickle")
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(pickle.dumps(self._in_memory_store[namespace]))
    
    async def _load_namespace(self, namespace: str) -> None:
        """Load a namespace from disk."""
        if not self._persistence_dir:
            self._in_memory_store[namespace] = {}
            return
            
        file_path = os.path.join(self._persistence_dir, f"{namespace}.pickle")
        if not os.path.exists(file_path):
            self._in_memory_store[namespace] = {}
            return
            
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                self._in_memory_store[namespace] = pickle.loads(data)
        except (pickle.PickleError, EOFError):
            # Handle corrupt data
            self._in_memory_store[namespace] = {} 