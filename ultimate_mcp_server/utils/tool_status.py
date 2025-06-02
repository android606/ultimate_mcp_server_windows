"""
Tool Status Management System for Ultimate MCP Server
===================================================

This module provides comprehensive tool status tracking and reporting, especially
important for lazy-loaded tools where users need to know if a tool is loading,
available, or failed.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta


class ToolStatus(Enum):
    """Tool loading/availability status."""
    NOT_LOADED = "not_loaded"      # Tool hasn't been loaded yet
    LOADING = "loading"            # Tool is currently loading
    LOADED = "loaded"              # Tool is loaded and ready
    FAILED = "failed"              # Tool failed to load
    UNAVAILABLE = "unavailable"    # Tool is permanently unavailable (missing deps)


@dataclass
class ToolInfo:
    """Information about a tool's status and loading."""
    name: str
    status: ToolStatus = ToolStatus.NOT_LOADED
    load_start_time: Optional[float] = None
    load_end_time: Optional[float] = None
    error_message: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    estimated_load_time: Optional[float] = None  # seconds
    retry_count: int = 0
    last_retry_time: Optional[float] = None
    
    @property
    def load_duration(self) -> Optional[float]:
        """Get the time it took to load the tool."""
        if self.load_start_time and self.load_end_time:
            return self.load_end_time - self.load_start_time
        return None
    
    @property
    def current_load_duration(self) -> Optional[float]:
        """Get current loading duration if tool is still loading."""
        if self.status == ToolStatus.LOADING and self.load_start_time:
            return time.time() - self.load_start_time
        return None
    
    @property
    def is_available(self) -> bool:
        """Check if tool is ready to use."""
        return self.status == ToolStatus.LOADED
    
    @property
    def is_loading(self) -> bool:
        """Check if tool is currently loading."""
        return self.status == ToolStatus.LOADING
    
    @property
    def estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining load time based on current progress."""
        if not self.is_loading or not self.estimated_load_time or not self.load_start_time:
            return None
        
        elapsed = time.time() - self.load_start_time
        return max(0, self.estimated_load_time - elapsed)


class ToolStatusManager:
    """
    Manages tool status tracking and reporting.
    Integrates with lazy imports to provide real-time status updates.
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self.loading_locks: Dict[str, asyncio.Lock] = {}
        self.status_listeners: List[Callable] = []
        
        # Estimated load times for common heavy tools (in seconds)
        self.load_time_estimates = {
            'convert_document': 8.0,    # docling, cv2, torch
            'ocr_image': 5.0,           # opencv, tesseract
            'execute_python': 12.0,     # pyodide browser setup
            'smart_browser': 6.0,       # playwright
            'marqo_search': 4.0,        # vector search
            'sql_databases': 3.0,       # database connections
            'excel_tools': 2.0,         # openpyxl, xlsxwriter
            'unified_memory': 7.0,      # ML embeddings
        }
    
    def register_tool(self, tool_name: str, dependencies: Optional[List[str]] = None, 
                     estimated_load_time: Optional[float] = None):
        """Register a tool for status tracking."""
        if tool_name not in self.tools:
            self.tools[tool_name] = ToolInfo(
                name=tool_name,
                dependencies=dependencies or [],
                estimated_load_time=estimated_load_time or self.load_time_estimates.get(tool_name)
            )
            self.loading_locks[tool_name] = asyncio.Lock()
    
    def get_tool_status(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed status information for a tool."""
        if tool_name not in self.tools:
            return {
                'name': tool_name,
                'status': 'unknown',
                'available': False,
                'error': 'Tool not registered'
            }
        
        tool = self.tools[tool_name]
        
        status_info = {
            'name': tool_name,
            'status': tool.status.value,
            'available': tool.is_available,
            'loading': tool.is_loading,
            'dependencies': tool.dependencies,
            'retry_count': tool.retry_count
        }
        
        # Add timing information
        if tool.load_duration:
            status_info['load_time_seconds'] = round(tool.load_duration, 3)
        
        if tool.current_load_duration:
            status_info['current_load_time_seconds'] = round(tool.current_load_duration, 3)
        
        if tool.estimated_remaining_time:
            status_info['estimated_remaining_seconds'] = round(tool.estimated_remaining_time, 1)
        
        if tool.estimated_load_time:
            status_info['estimated_total_load_seconds'] = tool.estimated_load_time
        
        # Add error information
        if tool.error_message:
            status_info['error'] = tool.error_message
            status_info['last_retry'] = tool.last_retry_time
        
        return status_info
    
    def get_all_tools_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all registered tools."""
        return {
            name: self.get_tool_status(name) 
            for name in self.tools.keys()
        }
    
    def get_tools_by_status(self, status: ToolStatus) -> List[str]:
        """Get list of tools with specific status."""
        return [
            name for name, tool in self.tools.items() 
            if tool.status == status
        ]
    
    async def mark_loading(self, tool_name: str) -> bool:
        """
        Mark a tool as loading. Returns False if already loading.
        Uses async lock to prevent race conditions.
        """
        if tool_name not in self.tools:
            self.register_tool(tool_name)
        
        async with self.loading_locks[tool_name]:
            tool = self.tools[tool_name]
            
            if tool.status == ToolStatus.LOADING:
                return False  # Already loading
            
            tool.status = ToolStatus.LOADING
            tool.load_start_time = time.time()
            tool.load_end_time = None
            
            await self._notify_status_change(tool_name, tool.status)
            return True
    
    async def mark_loaded(self, tool_name: str):
        """Mark a tool as successfully loaded."""
        if tool_name not in self.tools:
            return
        
        async with self.loading_locks[tool_name]:
            tool = self.tools[tool_name]
            tool.status = ToolStatus.LOADED
            tool.load_end_time = time.time()
            tool.error_message = None
            
            await self._notify_status_change(tool_name, tool.status)
    
    async def mark_failed(self, tool_name: str, error_message: str):
        """Mark a tool as failed to load."""
        if tool_name not in self.tools:
            self.register_tool(tool_name)
        
        async with self.loading_locks[tool_name]:
            tool = self.tools[tool_name]
            tool.status = ToolStatus.FAILED
            tool.load_end_time = time.time()
            tool.error_message = error_message
            tool.retry_count += 1
            tool.last_retry_time = time.time()
            
            await self._notify_status_change(tool_name, tool.status)
    
    async def mark_unavailable(self, tool_name: str, reason: str):
        """Mark a tool as permanently unavailable."""
        if tool_name not in self.tools:
            self.register_tool(tool_name)
        
        tool = self.tools[tool_name]
        tool.status = ToolStatus.UNAVAILABLE
        tool.error_message = reason
        
        await self._notify_status_change(tool_name, tool.status)
    
    async def retry_tool(self, tool_name: str) -> bool:
        """
        Attempt to retry loading a failed tool.
        Returns True if retry was initiated, False otherwise.
        """
        if tool_name not in self.tools:
            return False
        
        tool = self.tools[tool_name]
        
        # Only retry failed tools, and not too frequently
        if tool.status != ToolStatus.FAILED:
            return False
        
        # Rate limit retries (wait at least 30 seconds between retries)
        if (tool.last_retry_time and 
            time.time() - tool.last_retry_time < 30):
            return False
        
        # Reset status for retry
        tool.status = ToolStatus.NOT_LOADED
        tool.error_message = None
        
        return True
    
    def add_status_listener(self, listener: Callable):
        """Add a callback for status change notifications."""
        self.status_listeners.append(listener)
    
    async def _notify_status_change(self, tool_name: str, status: ToolStatus):
        """Notify all listeners of status changes."""
        for listener in self.status_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(tool_name, status)
                else:
                    listener(tool_name, status)
            except Exception as e:
                print(f"⚠️  Status listener error: {e}")
    
    def get_loading_summary(self) -> Dict[str, Any]:
        """Get a summary of current loading status."""
        all_tools = list(self.tools.keys())
        
        return {
            'total_tools': len(all_tools),
            'loaded': len(self.get_tools_by_status(ToolStatus.LOADED)),
            'loading': len(self.get_tools_by_status(ToolStatus.LOADING)),
            'failed': len(self.get_tools_by_status(ToolStatus.FAILED)),
            'not_loaded': len(self.get_tools_by_status(ToolStatus.NOT_LOADED)),
            'unavailable': len(self.get_tools_by_status(ToolStatus.UNAVAILABLE)),
            'loading_tools': self.get_tools_by_status(ToolStatus.LOADING),
            'failed_tools': self.get_tools_by_status(ToolStatus.FAILED)
        }


# Global status manager instance
tool_status = ToolStatusManager()


def track_tool_loading(tool_name: str):
    """
    Decorator to automatically track tool loading status.
    
    Usage:
        @track_tool_loading('convert_document')
        async def convert_document_tool(...):
            # Tool implementation
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Mark as loading
            loading_started = await tool_status.mark_loading(tool_name)
            
            if not loading_started:
                # Already loading, wait a bit and check status
                await asyncio.sleep(0.1)
                status_info = tool_status.get_tool_status(tool_name)
                if status_info['loading']:
                    raise RuntimeError(
                        f"Tool '{tool_name}' is currently loading. "
                        f"Please wait {status_info.get('estimated_remaining_seconds', '?')} seconds."
                    )
            
            try:
                result = await func(*args, **kwargs)
                await tool_status.mark_loaded(tool_name)
                return result
            except Exception as e:
                await tool_status.mark_failed(tool_name, str(e))
                raise
        
        return wrapper
    return decorator


# Integration with lazy imports
class StatusAwareLazyModule:
    """Lazy module that reports loading status."""
    
    def __init__(self, name: str, import_func, tool_name: str):
        self._name = name
        self._import_func = import_func
        self._tool_name = tool_name
        self._module = None
        self._loading = False
    
    async def __getattr__(self, attr):
        if self._module is None and not self._loading:
            self._loading = True
            await tool_status.mark_loading(self._tool_name)
            
            try:
                self._module = self._import_func()
                await tool_status.mark_loaded(self._tool_name)
            except Exception as e:
                await tool_status.mark_failed(self._tool_name, str(e))
                raise
            finally:
                self._loading = False
        
        return getattr(self._module, attr)


# MCP Tool for checking status
async def get_tool_status_mcp(tool_name: Optional[str] = None):
    """
    MCP tool to check tool loading status.
    
    Args:
        tool_name: Specific tool to check, or None for all tools
    
    Returns:
        Dictionary with tool status information
    """
    if tool_name:
        return {
            'success': True,
            'tool_status': tool_status.get_tool_status(tool_name)
        }
    else:
        return {
            'success': True,
            'summary': tool_status.get_loading_summary(),
            'all_tools': tool_status.get_all_tools_status()
        }


# Usage examples
"""
USAGE EXAMPLES:

1. Register tools at startup:
   tool_status.register_tool('convert_document', ['docling', 'cv2', 'torch'])
   tool_status.register_tool('ocr_image', ['opencv', 'tesseract'])

2. Use decorator for automatic tracking:
   @track_tool_loading('convert_document')
   async def convert_document_tool(...):
       # Implementation here

3. Check tool status:
   status = tool_status.get_tool_status('convert_document')
   if status['loading']:
       print(f"Tool loading, {status['estimated_remaining_seconds']}s remaining")

4. Get overview:
   summary = tool_status.get_loading_summary()
   print(f"{summary['loaded']}/{summary['total_tools']} tools loaded")

5. Add status change listener:
   def on_status_change(tool_name, status):
       print(f"Tool {tool_name} is now {status.value}")
   
   tool_status.add_status_listener(on_status_change)

6. Integration with MCP:
   # Add this as an MCP tool
   server.add_tool("get_tool_status", get_tool_status_mcp)
""" 