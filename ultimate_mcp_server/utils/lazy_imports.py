"""
Lazy Import Utilities for Ultimate MCP Server
============================================

This module provides utilities for lazy loading of heavy dependencies to improve startup time.
Heavy imports like numpy, pandas, cv2, torch, etc. are only loaded when actually needed.

Integrates with tool status tracking to provide real-time loading feedback.
"""

import importlib
import time
import asyncio
from typing import Any, Dict, Optional
try:
    from .tool_status import tool_status, ToolStatus
except ImportError:
    # Fallback if tool_status not available
    tool_status = None
    ToolStatus = None


class LazyModule:
    """Proxy module that imports on first attribute access."""
    
    def __init__(self, name: str, import_func, tool_name: Optional[str] = None):
        self._name = name
        self._import_func = import_func
        self._tool_name = tool_name or name
        self._module = None
        self._import_time = None
        self._loading = False
    
    def __getattr__(self, attr):
        if self._module is None and not self._loading:
            self._loading = True
            start_time = time.time()
            
            # Update status if tracking is available
            if tool_status:
                # Use asyncio.create_task if we're in an async context
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(tool_status.mark_loading(self._tool_name))
                except RuntimeError:
                    # Not in async context, continue without status
                    pass
            
            try:
                self._module = self._import_func()
                self._import_time = time.time() - start_time
                
                print(f"📦 Lazy loaded {self._name} in {self._import_time:.3f}s")
                
                # Mark as loaded
                if tool_status:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(tool_status.mark_loaded(self._tool_name))
                    except RuntimeError:
                        pass
                        
            except Exception as e:
                # Mark as failed
                if tool_status:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(tool_status.mark_failed(self._tool_name, str(e)))
                    except RuntimeError:
                        pass
                
                print(f"❌ Failed to import {self._name}: {e}")
                raise
            finally:
                self._loading = False
        
        return getattr(self._module, attr)
    
    def __repr__(self):
        if self._loading:
            return f"<LazyModule '{self._name}' (loading...)>"
        elif self._module is not None:
            load_time = f" loaded in {self._import_time:.3f}s" if self._import_time else ""
            return f"<LazyModule '{self._name}' (loaded{load_time})>"
        else:
            return f"<LazyModule '{self._name}' (not loaded)>"


class StatusAwareLazyModule(LazyModule):
    """Enhanced lazy module with full async status tracking."""
    
    async def async_getattr(self, attr):
        """Async version of getattr with proper status tracking."""
        if self._module is None and not self._loading:
            self._loading = True
            
            # Mark as loading
            if tool_status:
                await tool_status.mark_loading(self._tool_name)
            
            start_time = time.time()
            
            try:
                self._module = self._import_func()
                self._import_time = time.time() - start_time
                
                print(f"📦 Lazy loaded {self._name} in {self._import_time:.3f}s")
                
                # Mark as loaded
                if tool_status:
                    await tool_status.mark_loaded(self._tool_name)
                    
            except Exception as e:
                # Mark as failed
                if tool_status:
                    await tool_status.mark_failed(self._tool_name, str(e))
                
                print(f"❌ Failed to import {self._name}: {e}")
                raise
            finally:
                self._loading = False
        
        return getattr(self._module, attr)


class LazyImporter:
    """
    Lazy importer to defer heavy imports until actually needed.
    This can reduce startup time significantly for ML libraries.
    """
    
    def __init__(self):
        self._modules: Dict[str, LazyModule] = {}
        self._import_times: Dict[str, float] = {}
        
        # Register common heavy tools with status tracking
        if tool_status:
            self._register_heavy_tools()
    
    def _register_heavy_tools(self):
        """Register common heavy tools for status tracking."""
        heavy_tools = {
            'numpy': ['numpy'],
            'pandas': ['pandas'],
            'cv2': ['opencv-python'],
            'torch': ['torch'],
            'transformers': ['transformers'],
            'sklearn': ['scikit-learn'],
            'matplotlib': ['matplotlib'],
            'docling': ['docling'],
            'playwright': ['playwright'],
        }
        
        for tool_name, deps in heavy_tools.items():
            tool_status.register_tool(tool_name, deps)
    
    def lazy_import(self, module_name: str, package: Optional[str] = None, 
                   fallback=None, tool_name: Optional[str] = None):
        """
        Returns a lazy-loaded module that imports on first access.
        
        Args:
            module_name: Name of the module to import
            package: Package name for relative imports
            fallback: Value to return if import fails (None by default)
            tool_name: Name for status tracking (defaults to module_name)
        
        Usage:
            np = lazy_import('numpy')
            array = np.array([1, 2, 3])  # Import happens here
        """
        cache_key = f"{package}.{module_name}" if package else module_name
        tool_name = tool_name or module_name.split('.')[0]  # Use base module name
        
        if cache_key not in self._modules:
            def _import_on_access():
                try:
                    module = importlib.import_module(module_name, package)
                    return module
                except ImportError as e:
                    if fallback is not None:
                        print(f"⚠️  Failed to import {module_name}, using fallback: {e}")
                        return fallback
                    else:
                        print(f"❌ Failed to import {module_name}: {e}")
                        raise
            
            self._modules[cache_key] = LazyModule(cache_key, _import_on_access, tool_name)
        
        return self._modules[cache_key]
    
    def status_aware_import(self, module_name: str, package: Optional[str] = None,
                          tool_name: Optional[str] = None):
        """
        Returns a status-aware lazy module for async contexts.
        
        Usage:
            np_module = lazy.status_aware_import('numpy')
            np_array = await np_module.async_getattr('array')
        """
        cache_key = f"{package}.{module_name}" if package else module_name
        tool_name = tool_name or module_name.split('.')[0]
        
        if cache_key not in self._modules:
            def _import_on_access():
                return importlib.import_module(module_name, package)
            
            self._modules[cache_key] = StatusAwareLazyModule(cache_key, _import_on_access, tool_name)
        
        return self._modules[cache_key]
    
    def get_import_stats(self) -> Dict[str, float]:
        """Get timing statistics for all imported modules."""
        stats = {}
        for name, module in self._modules.items():
            if module._import_time is not None:
                stats[name] = module._import_time
        return stats
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get loading status summary."""
        if not tool_status:
            return {'status_tracking': 'disabled'}
        
        return tool_status.get_loading_summary()
    
    async def preload_modules(self, module_names: list, timeout: float = 30.0):
        """
        Preload specified modules in the background with status tracking.
        
        Args:
            module_names: List of module names to preload
            timeout: Maximum time to wait for all modules to load
        """
        async def preload_single(name: str):
            try:
                if tool_status:
                    await tool_status.mark_loading(name)
                
                module = self.lazy_import(name)
                # Trigger the import by accessing a common attribute
                if hasattr(module, '__version__'):
                    _ = module.__version__
                elif hasattr(module, '__name__'):
                    _ = module.__name__
                else:
                    # Just access the module to trigger import
                    _ = str(module)
                    
                if tool_status:
                    await tool_status.mark_loaded(name)
                    
            except Exception as e:
                if tool_status:
                    await tool_status.mark_failed(name, str(e))
                print(f"⚠️  Failed to preload {name}: {e}")
        
        async def preload_all():
            tasks = [preload_single(name) for name in module_names]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                print(f"⚠️  Module preloading timed out after {timeout}s")
        
        await preload_all()


# Global lazy importer instance
lazy = LazyImporter()

# Common heavy imports that should be lazy loaded
def get_numpy():
    """Get numpy with lazy loading and status tracking."""
    return lazy.lazy_import('numpy', tool_name='numpy')

def get_pandas():
    """Get pandas with lazy loading and status tracking."""
    return lazy.lazy_import('pandas', tool_name='pandas')

def get_cv2():
    """Get OpenCV with lazy loading and status tracking."""
    return lazy.lazy_import('cv2', tool_name='cv2')

def get_torch():
    """Get PyTorch with lazy loading and status tracking."""
    return lazy.lazy_import('torch', tool_name='torch')

def get_transformers():
    """Get transformers with lazy loading and status tracking."""
    return lazy.lazy_import('transformers', tool_name='transformers')

def get_sklearn():
    """Get scikit-learn with lazy loading and status tracking."""
    return lazy.lazy_import('sklearn', tool_name='sklearn')

def get_matplotlib():
    """Get matplotlib with lazy loading and status tracking."""
    return lazy.lazy_import('matplotlib.pyplot', fallback=None, tool_name='matplotlib')

def get_seaborn():
    """Get seaborn with lazy loading and status tracking."""
    return lazy.lazy_import('seaborn', fallback=None, tool_name='seaborn')

def get_docling():
    """Get docling with lazy loading and status tracking."""
    return lazy.lazy_import('docling', tool_name='docling')

def get_playwright():
    """Get playwright with lazy loading and status tracking."""
    return lazy.lazy_import('playwright', tool_name='playwright')

# Convenience function for common ML stack
async def preload_ml_stack():
    """Preload common ML libraries in background with status tracking."""
    ml_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'sklearn', 'cv2'
    ]
    await lazy.preload_modules(ml_modules)

async def preload_document_stack():
    """Preload document processing libraries."""
    doc_modules = ['docling', 'cv2', 'transformers']
    await lazy.preload_modules(doc_modules)

async def preload_browser_stack():
    """Preload browser automation libraries."""
    browser_modules = ['playwright']
    await lazy.preload_modules(browser_modules)

# Tool status integration functions
def get_tool_loading_status(tool_name: str) -> Dict[str, Any]:
    """Get loading status for a specific tool."""
    if not tool_status:
        return {'status_tracking': 'disabled'}
    return tool_status.get_tool_status(tool_name)

def get_all_tools_status() -> Dict[str, Any]:
    """Get status for all tools."""
    if not tool_status:
        return {'status_tracking': 'disabled'}
    return {
        'summary': tool_status.get_loading_summary(),
        'tools': tool_status.get_all_tools_status()
    }

# Usage examples and patterns
"""
USAGE PATTERNS:

1. Basic lazy import with status tracking:
   np = lazy.lazy_import('numpy')
   arr = np.array([1, 2, 3])  # numpy imported here, status tracked

2. Using convenience functions:
   np = get_numpy()
   pd = get_pandas()

3. Check loading status:
   status = get_tool_loading_status('numpy')
   if status.get('loading'):
       print(f"Numpy loading, {status.get('estimated_remaining_seconds')}s remaining")

4. Preload in background with status:
   await preload_ml_stack()  # Status tracked for each module

5. Status-aware lazy loading for async contexts:
   np_module = lazy.status_aware_import('numpy')
   np_array = await np_module.async_getattr('array')

6. Get overall status:
   summary = get_all_tools_status()
   print(f"{summary['summary']['loaded']}/{summary['summary']['total_tools']} tools loaded")

INTEGRATION WITH EXISTING CODE:

Replace existing imports:
# Before:
import numpy as np
import pandas as pd
import cv2

# After:
from ultimate_mcp_server.utils.lazy_imports import get_numpy, get_pandas, get_cv2
np = get_numpy()
pd = get_pandas()
cv2 = get_cv2()

# Check status before using heavy tools:
status = get_tool_loading_status('numpy')
if not status.get('available'):
    if status.get('loading'):
        raise RuntimeError(f"Numpy still loading, wait {status.get('estimated_remaining_seconds')}s")
    else:
        raise RuntimeError(f"Numpy unavailable: {status.get('error')}")

Or use the lazy importer directly:
from ultimate_mcp_server.utils.lazy_imports import lazy
np = lazy.lazy_import('numpy')
pd = lazy.lazy_import('pandas')
""" 