"""Playwright browser automation tools for LLM Gateway.

This module provides a comprehensive set of tools for browser automation using Playwright,
enabling actions like navigation, element interaction, screenshots, and more through a
standardized API compatible with LLM Gateway.
"""

import asyncio
import base64
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Page,
    Playwright,
    Response,
    async_playwright,
)

from llm_gateway.constants import TaskType
from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.playwright")

# --- Global browser session handling ---

_playwright_instance: Optional[Playwright] = None
_browser_instance: Optional[Browser] = None
_browser_context: Optional[BrowserContext] = None
_pages: Dict[str, Page] = {}
_current_page_id: Optional[str] = None
_snapshot_cache: Dict[str, Dict[str, Any]] = {}

async def _ensure_playwright():
    """Ensure Playwright is initialized and return the instance."""
    global _playwright_instance
    if not _playwright_instance:
        logger.info("Initializing Playwright")
        _playwright_instance = await async_playwright().start()
    return _playwright_instance

async def _ensure_browser(
    browser_name: str = "chromium",
    headless: bool = False,
    user_data_dir: Optional[str] = None,
    executable_path: Optional[str] = None
) -> Browser:
    """Ensure browser is launched and return the instance."""
    global _browser_instance, _playwright_instance
    
    if not _browser_instance:
        playwright = await _ensure_playwright()
        
        browser_type = getattr(playwright, browser_name.lower())
        if not browser_type:
            raise ToolError(
                status_code=400,
                detail=f"Unsupported browser type: {browser_name}. Use 'chromium', 'firefox', or 'webkit'."
            )
        
        launch_options = {
            "headless": headless
        }
        
        if executable_path:
            launch_options["executable_path"] = executable_path
            
        if user_data_dir:
            # Launch persistent context for chromium
            _browser_context = await browser_type.launch_persistent_context(
                user_data_dir=user_data_dir,
                **launch_options
            )
            # In persistent context mode, the browser is contained within the context
            _browser_instance = _browser_context.browser
        else:
            # Standard browser launch
            try:
                _browser_instance = await browser_type.launch(**launch_options)
            except Exception as e:
                if "executable doesn't exist" in str(e).lower():
                    raise ToolError(
                        status_code=500,
                        detail=f"Browser {browser_name} is not installed. Use browser_install tool to install it."
                    ) from e
                raise
                
        logger.info(
            f"Browser {browser_name} launched successfully",
            emoji_key="browser",
            headless=headless
        )
    
    return _browser_instance

async def _ensure_context(
    browser: Browser,
    user_data_dir: Optional[str] = None
) -> BrowserContext:
    """Ensure browser context is created and return the instance."""
    global _browser_context
    
    if not _browser_context:
        # If we're in persistent context mode, the context already exists
        if user_data_dir:
            # Find the "default" context that was created with the browser
            _browser_context = browser.contexts[0] if browser.contexts else None
            
        # Otherwise create a new context
        if not _browser_context:
            _browser_context = await browser.new_context()
            
        logger.info(
            "Browser context created",
            emoji_key="browser"
        )
    
    return _browser_context

async def _ensure_page() -> Tuple[str, Page]:
    """Ensure at least one page exists and return the current page ID and page."""
    global _current_page_id, _pages
    
    if not _current_page_id or _current_page_id not in _pages:
        # Create a new page if none exists
        context = await _ensure_context(
            await _ensure_browser()
        )
        
        page = await context.new_page()
        page_id = str(uuid.uuid4())
        _pages[page_id] = page
        _current_page_id = page_id
        
        # Set up page event handlers
        await _setup_page_event_handlers(page)
        
        logger.info(
            "New browser page created",
            emoji_key="browser",
            page_id=page_id
        )
    
    return _current_page_id, _pages[_current_page_id]

async def _setup_page_event_handlers(page: Page):
    """Set up event handlers for a page."""
    
    page.on("console", lambda msg: logger.debug(
        f"Console {msg.type}: {msg.text}",
        emoji_key="console"
    ))
    
    page.on("pageerror", lambda err: logger.error(
        f"Page error: {err}",
        emoji_key="error"
    ))
    
    page.on("dialog", lambda dialog: asyncio.create_task(
        dialog.dismiss()
    ))

async def _capture_snapshot(page: Page) -> Dict[str, Any]:
    """Capture page snapshot including accessibility tree."""
    
    # This function simulates the functionality of the TypeScript aria-snapshot
    # In a real implementation, we would use proper accessibility APIs
    
    snapshot_data = await page.evaluate("""() => {
        function getAccessibilityTree(element, depth = 0) {
            if (!element) return null;
            
            const role = element.getAttribute('role') || element.tagName.toLowerCase();
            const name = element.getAttribute('aria-label') || 
                        element.textContent?.trim() || 
                        element.getAttribute('alt') || 
                        element.getAttribute('title') || '';
                        
            const ref = 'ref-' + Math.random().toString(36).substring(2, 10);
            
            const result = {
                role,
                name: name.substring(0, 100), // Truncate long names
                ref,
                children: []
            };
            
            // Add more accessibility attributes as needed
            if (element.getAttribute('aria-selected'))
                result.selected = element.getAttribute('aria-selected') === 'true';
            
            if (element.getAttribute('aria-checked'))
                result.checked = element.getAttribute('aria-checked') === 'true';
            
            if (element.getAttribute('aria-expanded'))
                result.expanded = element.getAttribute('aria-expanded') === 'true';
                
            if (element.hasAttribute('disabled') || element.getAttribute('aria-disabled') === 'true')
                result.disabled = true;
                
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA')
                result.value = element.value;
                
            // Process children, but limit depth to avoid stack overflow
            if (depth < 20) {
                for (const child of element.children) {
                    const childTree = getAccessibilityTree(child, depth + 1);
                    if (childTree) result.children.push(childTree);
                }
            }
            
            return result;
        }
        
        return {
            url: window.location.href,
            title: document.title,
            tree: getAccessibilityTree(document.body)
        };
    }""")
    
    return snapshot_data

async def _find_element_by_ref(page: Page, ref: str) -> ElementHandle:
    """Find an element by its reference ID."""
    
    # This would query elements based on the ref attribute we added in the snapshot
    element = await page.query_selector(f"[data-ref='{ref}']")
    
    if not element:
        # In real implementation, the snapshot would add data-ref attributes
        # Since we can't do that in this demo, we'll raise an error
        raise ToolError(
            status_code=404,
            detail=f"Element with ref {ref} not found. This function relies on proper snapshot implementation."
        )
    
    return element

async def _clean_up_resources():
    """Clean up all Playwright resources."""
    global _browser_instance, _browser_context, _playwright_instance, _pages, _current_page_id
    
    # Close all pages
    for _page_id, page in list(_pages.items()):
        try:
            await page.close()
        except Exception:
            pass
    _pages = {}
    _current_page_id = None
    
    # Close browser context
    if _browser_context:
        try:
            await _browser_context.close()
        except Exception:
            pass
        _browser_context = None
    
    # Close browser
    if _browser_instance:
        try:
            await _browser_instance.close()
        except Exception:
            pass
        _browser_instance = None
        
    # Close playwright
    if _playwright_instance:
        try:
            await _playwright_instance.stop()
        except Exception:
            pass
        _playwright_instance = None
    
    logger.info("All browser resources cleaned up", emoji_key="cleanup")

@with_tool_metrics
@with_error_handling
async def browser_close() -> Dict[str, Any]:
    """Close the browser and clean up all resources.

    Closes all open tabs, the browser context, and the browser instance.
    This frees up system resources and should be called when browser automation is complete.
    
    Args:
        None

    Returns:
        A dictionary containing results:
        {
            "success": true,
            "message": "Browser closed successfully"
        }

    Raises:
        ToolError: If browser closing fails.
    """
    start_time = time.time()
    
    try:
        logger.info("Closing browser and cleaning up resources", emoji_key="browser")
        
        # Clean up all resources
        await _clean_up_resources()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "message": "Browser closed successfully",
            "processing_time": processing_time
        }
        
    except Exception as e:
        error_msg = f"Failed to close browser: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_install(
    browser_name: str = "chromium"
) -> Dict[str, Any]:
    """Install a Playwright browser.

    Installs the specified browser using Playwright's installation mechanism.
    This is useful when a browser is not already installed on the system.
    
    Args:
        browser_name: Name of the browser to install. Options: "chromium", "firefox", "webkit".
                     Default: "chromium".

    Returns:
        A dictionary containing installation results:
        {
            "success": true,
            "browser_name": "chromium",
            "message": "Browser installed successfully"
        }

    Raises:
        ToolError: If browser installation fails.
    """
    start_time = time.time()
    
    # Validate browser_name
    valid_browsers = ["chromium", "firefox", "webkit"]
    if browser_name not in valid_browsers:
        raise ToolInputError(
            f"Invalid browser name. Must be one of: {', '.join(valid_browsers)}",
            param_name="browser_name",
            provided_value=browser_name
        )
    
    try:
        logger.info(f"Installing browser: {browser_name}", emoji_key="install")
        
        # Use subprocess to run playwright install command
        import subprocess
        import sys
        
        # Get Python executable path
        python_executable = sys.executable
        
        # Run playwright install command
        process = await asyncio.create_subprocess_exec(
            python_executable, "-m", "playwright", "install", browser_name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_output = stderr.decode()
            raise ToolError(
                status_code=500,
                detail=f"Browser installation failed: {error_output}"
            )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "browser_name": browser_name,
            "message": f"Browser '{browser_name}' installed successfully",
            "processing_time": processing_time
        }
        
    except Exception as e:
        if isinstance(e, ToolError):
            raise
            
        error_msg = f"Failed to install browser: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_get_console_logs() -> Dict[str, Any]:
    """Get browser console logs from the current page.

    Retrieves JavaScript console logs (info, warnings, errors) from the current browser page.
    Useful for debugging JavaScript issues.
    
    Args:
        None

    Returns:
        A dictionary containing console logs:
        {
            "success": true,
            "logs": [                              # List of console log entries
                {
                    "type": "error",               # Log type: "log", "info", "warning", "error"
                    "text": "Reference error...",  # Log message text
                    "location": "https://..."      # URL where the log occurred
                },
                ...
            ]
        }

    Raises:
        ToolError: If retrieving console logs fails.
    """
    start_time = time.time()
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Get console logs
        # In a real implementation, we'd capture console logs using page.on("console")
        # Here, we'll simulate by evaluating a JavaScript function
        logs = await page.evaluate("""() => {
            // This is a simulation - in a real implementation,
            // logs would be captured via page.on("console") event handlers
            
            // Return last 50 logs from browser console API
            if (!window._consoleLogs) {
                window._consoleLogs = [];
                
                // Store original console methods
                const originalConsole = {
                    log: console.log,
                    info: console.info,
                    warn: console.warn,
                    error: console.error
                };
                
                // Override console methods to capture logs
                console.log = function() {
                    window._consoleLogs.push({
                        type: 'log',
                        text: Array.from(arguments).map(String).join(' '),
                        location: window.location.href,
                        timestamp: new Date().toISOString()
                    });
                    originalConsole.log.apply(console, arguments);
                };
                
                console.info = function() {
                    window._consoleLogs.push({
                        type: 'info',
                        text: Array.from(arguments).map(String).join(' '),
                        location: window.location.href,
                        timestamp: new Date().toISOString()
                    });
                    originalConsole.info.apply(console, arguments);
                };
                
                console.warn = function() {
                    window._consoleLogs.push({
                        type: 'warning',
                        text: Array.from(arguments).map(String).join(' '),
                        location: window.location.href,
                        timestamp: new Date().toISOString()
                    });
                    originalConsole.warn.apply(console, arguments);
                };
                
                console.error = function() {
                    window._consoleLogs.push({
                        type: 'error',
                        text: Array.from(arguments).map(String).join(' '),
                        location: window.location.href,
                        timestamp: new Date().toISOString()
                    });
                    originalConsole.error.apply(console, arguments);
                };
            }
            
            // Return logs (limit to last 50)
            return window._consoleLogs.slice(-50);
        }""")
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "logs": logs,
            "processing_time": processing_time
        }
        
    except Exception as e:
        error_msg = f"Failed to get console logs: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

# Register all tools in LLM Gateway
def register_playwright_tools(mcp_server) -> None:
    """Register all Playwright browser automation tools with the MCP server."""
    
    tools = [
        # Browser management
        browser_init,
        browser_close,
        browser_install,
        
        # Navigation
        browser_navigate,
        browser_back,
        browser_forward,
        browser_reload,
        browser_wait,
        
        # Tab management
        browser_tab_new,
        browser_tab_close,
        browser_tab_list,
        browser_tab_select,
        
        # Interaction
        browser_click,
        browser_type,
        browser_select,
        browser_checkbox,
        
        # Content extraction
        browser_get_text,
        browser_get_attributes,
        browser_get_console_logs,
        
        # Visual and file operations
        browser_screenshot,
        browser_pdf,
        browser_download_file,
        browser_upload_file,
        
        # Advanced
        browser_execute_javascript
    ]
    
    for tool in tools:
        mcp_server.tool(name=tool.__name__)(tool)
        logger.info(f"Registered Playwright tool: {tool.__name__}", emoji_key="⚙️")
    
    logger.success(f"Registered {len(tools)} Playwright browser automation tools", emoji_key="✅")

# --- Browser Control Tools ---

@with_tool_metrics
@with_error_handling
async def browser_init(
    browser_name: str = "chromium",
    headless: bool = False,
    user_data_dir: Optional[str] = None,
    executable_path: Optional[str] = None,
    default_timeout: int = 30000
) -> Dict[str, Any]:
    """Initializes a browser instance using Playwright.

    This tool allows you to customize browser settings and must be called before using other browser tools.
    If not called explicitly, other tools will use default settings.

    Args:
        browser_name: Browser to use. Options: "chromium" (Chrome), "firefox", or "webkit" (Safari).
                      Default: "chromium".
        headless: Whether to run the browser in headless mode (no GUI). 
                  Set to False to see the browser window. Default: False.
        user_data_dir: (Optional) Path to a user data directory to enable persistent sessions.
                       If not provided, a new temporary profile is created for each session.
        executable_path: (Optional) Path to custom browser executable instead of the bundled one.
        default_timeout: Timeout for browser operations in milliseconds. Default: 30000 (30 seconds).

    Returns:
        A dictionary containing initialization results:
        {
            "browser_name": "chromium",
            "headless": false,
            "user_data_dir": "/path/to/profile",  # If provided
            "browser_version": "115.0.5790.170",
            "success": true
        }

    Raises:
        ToolError: If browser initialization fails.
    """
    start_time = time.time()
    
    try:
        browser = await _ensure_browser(
            browser_name=browser_name,
            headless=headless,
            user_data_dir=user_data_dir,
            executable_path=executable_path
        )
        
        # Create context
        context = await _ensure_context(browser, user_data_dir)
        
        # Set default timeout
        context.set_default_timeout(default_timeout)
        
        # Get browser version
        version = await browser.version()
        
        processing_time = time.time() - start_time
        
        return {
            "browser_name": browser_name,
            "headless": headless,
            "user_data_dir": user_data_dir,
            "browser_version": version,
            "processing_time": processing_time,
            "success": True
        }
        
    except Exception as e:
        if isinstance(e, ToolError):
            raise
            
        raise ToolError(
            status_code=500,
            detail=f"Failed to initialize browser: {str(e)}"
        ) from e

@with_tool_metrics
@with_error_handling
async def browser_navigate(
    url: str,
    wait_until: str = "load",
    timeout: int = 30000,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Navigate to a URL in the browser.

    Opens the specified URL in the current browser tab, waiting for the page to load
    according to the specified criteria.

    Args:
        url: The URL to navigate to.
        wait_until: (Optional) When to consider navigation complete. Options:
                   - "load": Wait for the load event (default)
                   - "domcontentloaded": Wait for DOMContentLoaded event
                   - "networkidle": Wait for network to be idle
        timeout: (Optional) Maximum time to wait in milliseconds. Default: 30000 (30 seconds).
        capture_snapshot: (Optional) Whether to capture a page snapshot after navigation.
                        Default: True.

    Returns:
        A dictionary containing navigation results:
        {
            "url": "https://www.example.com", # Final URL after navigation (may differ due to redirects)
            "title": "Example Domain",        # Page title
            "status": 200,                    # HTTP status code
            "success": true,
            "snapshot": { ... }               # Page snapshot (if capture_snapshot=True)
        }

    Raises:
        ToolError: If navigation fails or times out.
    """
    start_time = time.time()
    
    # Validate URL
    if not url or not isinstance(url, str):
        raise ToolInputError("URL must be a non-empty string", param_name="url", provided_value=url)
    
    # Ensure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    
    # Validate wait_until
    valid_wait_options = ["load", "domcontentloaded", "networkidle"]
    if wait_until not in valid_wait_options:
        raise ToolInputError(
            f"Invalid wait_until value. Must be one of: {', '.join(valid_wait_options)}",
            param_name="wait_until",
            provided_value=wait_until
        )
    
    try:
        # Get or create page
        _, page = await _ensure_page()
        
        # Navigate to URL
        logger.info(f"Navigating to: {url}", emoji_key="browser")
        response: Optional[Response] = await page.goto(
            url=url,
            wait_until=wait_until,
            timeout=timeout
        )
        
        # Get navigation results
        final_url = page.url
        title = await page.title()
        status = response.status if response else None
        
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "url": final_url,
            "title": title,
            "status": status,
            "processing_time": processing_time,
            "success": True
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Successfully navigated to: {final_url} ({title})",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Navigation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            url=url
        )
        
        if "net::ERR_NAME_NOT_RESOLVED" in str(e):
            raise ToolError(status_code=404, detail=f"Could not resolve host: {url}") from e
        elif "net::ERR_CONNECTION_REFUSED" in str(e):
            raise ToolError(status_code=502, detail=f"Connection refused: {url}") from e
        elif "Timeout" in str(e):
            raise ToolError(status_code=408, detail=f"Navigation timed out after {timeout}ms: {url}") from e
        elif "ERR_ABORTED" in str(e):
            raise ToolError(status_code=499, detail=f"Navigation was aborted: {url}") from e
        else:
            raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_back(
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Navigate back to the previous page in browser history.

    Similar to clicking the browser's back button, this navigates to the previous page 
    in the current tab's history.

    Args:
        capture_snapshot: (Optional) Whether to capture a page snapshot after navigation.
                         Default: True.

    Returns:
        A dictionary containing navigation results:
        {
            "url": "https://www.example.com", # URL after navigation
            "title": "Example Domain",        # Page title after navigation
            "success": true,
            "snapshot": { ... }               # Page snapshot (if capture_snapshot=True)
        }

    Raises:
        ToolError: If navigation fails or no previous page exists in history.
    """
    start_time = time.time()
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Go back
        logger.info("Navigating back in history", emoji_key="browser")
        
        # Remember current URL before navigating back
        current_url = page.url
        
        response: Optional[Response] = await page.go_back()
        if not response:
            raise ToolError(
                status_code=400,
                detail="Could not navigate back - no previous page in history"
            )
        
        # Get navigation results
        final_url = page.url
        
        # If URLs are the same, navigation didn't actually happen
        if final_url == current_url:
            raise ToolError(
                status_code=400,
                detail="Could not navigate back - no previous page in history"
            )
            
        title = await page.title()
        
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "url": final_url,
            "title": title,
            "processing_time": processing_time,
            "success": True
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Successfully navigated back to: {final_url} ({title})",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, ToolError):
            raise
            
        error_msg = f"Navigation back failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_forward(
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Navigate forward to the next page in browser history.

    Similar to clicking the browser's forward button, this navigates to the next page 
    in the current tab's history.

    Args:
        capture_snapshot: (Optional) Whether to capture a page snapshot after navigation.
                         Default: True.

    Returns:
        A dictionary containing navigation results:
        {
            "url": "https://www.example.com", # URL after navigation
            "title": "Example Domain",        # Page title after navigation
            "success": true,
            "snapshot": { ... }               # Page snapshot (if capture_snapshot=True)
        }

    Raises:
        ToolError: If navigation fails or no next page exists in history.
    """
    start_time = time.time()
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Go forward
        logger.info("Navigating forward in history", emoji_key="browser")
        
        # Remember current URL before navigating forward
        current_url = page.url
        
        response: Optional[Response] = await page.go_forward()
        if not response:
            raise ToolError(
                status_code=400,
                detail="Could not navigate forward - no next page in history"
            )
        
        # Get navigation results
        final_url = page.url
        
        # If URLs are the same, navigation didn't actually happen
        if final_url == current_url:
            raise ToolError(
                status_code=400,
                detail="Could not navigate forward - no next page in history"
            )
            
        title = await page.title()
        
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "url": final_url,
            "title": title,
            "processing_time": processing_time,
            "success": True
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Successfully navigated forward to: {final_url} ({title})",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, ToolError):
            raise
            
        error_msg = f"Navigation forward failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_reload(
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Reload the current page.

    Similar to clicking the browser's refresh button, this reloads the current page.

    Args:
        capture_snapshot: (Optional) Whether to capture a page snapshot after reload.
                         Default: True.

    Returns:
        A dictionary containing reload results:
        {
            "url": "https://www.example.com", # URL after reload
            "title": "Example Domain",        # Page title after reload
            "success": true,
            "snapshot": { ... }               # Page snapshot (if capture_snapshot=True)
        }

    Raises:
        ToolError: If reload fails.
    """
    start_time = time.time()
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Reload
        logger.info("Reloading page", emoji_key="browser")
        
        await page.reload()
        
        # Get reload results
        final_url = page.url
        title = await page.title()
        
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "url": final_url,
            "title": title,
            "processing_time": processing_time,
            "success": True
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Successfully reloaded page: {final_url} ({title})",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Page reload failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_screenshot(
    full_page: bool = False,
    element_selector: Optional[str] = None,
    quality: int = 80,
    omit_background: bool = False
) -> Dict[str, Any]:
    """Take a screenshot of the current page or an element.

    Captures a screenshot of the current browser page, either the entire page, 
    the visible viewport, or a specific element.

    Args:
        full_page: (Optional) Whether to capture the entire scrollable page or just the visible viewport.
                  Default: False (only the visible viewport).
        element_selector: (Optional) CSS selector for capturing a specific element. If provided,
                       only that element will be captured.
        quality: (Optional) Image quality from 0-100 (JPEG compression quality). 
                Default: 80. Higher values = larger file size but better quality.
        omit_background: (Optional) Whether to hide default white background and allow capturing
                       screenshots with transparency. Default: False.

    Returns:
        A dictionary containing screenshot data:
        {
            "data": "base64-encoded-image-data",  # Base64-encoded image data
            "mime_type": "image/jpeg",            # Image MIME type
            "width": 1280,                        # Screenshot width in pixels
            "height": 720,                        # Screenshot height in pixels
            "success": true
        }

    Raises:
        ToolError: If screenshot capture fails.
        ToolInputError: If element_selector is provided but no matching element is found.
    """
    start_time = time.time()
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Validate quality
        if not 0 <= quality <= 100:
            raise ToolInputError(
                "Quality must be between 0 and 100",
                param_name="quality",
                provided_value=quality
            )
        
        # Prepare screenshot options
        screenshot_options = {
            "type": "jpeg",
            "quality": quality,
            "full_page": full_page,
            "omit_background": omit_background
        }
        
        # Take screenshot
        if element_selector:
            logger.info(f"Taking screenshot of element: {element_selector}", emoji_key="camera")
            element = await page.query_selector(element_selector)
            
            if not element:
                raise ToolInputError(
                    f"Element not found: {element_selector}",
                    param_name="element_selector",
                    provided_value=element_selector
                )
                
            screenshot_bytes = await element.screenshot(
                **{k: v for k, v in screenshot_options.items() if k != "full_page"}
            )
        else:
            logger.info(
                f"Taking {'full page' if full_page else 'viewport'} screenshot",
                emoji_key="camera"
            )
            screenshot_bytes = await page.screenshot(screenshot_options)
        
        # Convert to base64
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Get page size (viewport or full page)
        if full_page and not element_selector:
            # Get full page size
            dimensions = await page.evaluate("""() => {
                return {
                    width: document.documentElement.scrollWidth,
                    height: document.documentElement.scrollHeight
                }
            }""")
        elif element_selector:
            # Get element size
            dimensions = await page.evaluate("""(selector) => {
                const element = document.querySelector(selector);
                if (!element) return { width: 0, height: 0 };
                const { width, height } = element.getBoundingClientRect();
                return { width: Math.ceil(width), height: Math.ceil(height) }
            }""", element_selector)
        else:
            # Get viewport size
            viewport_size = page.viewport_size
            dimensions = {
                "width": viewport_size["width"],
                "height": viewport_size["height"]
            }
        
        processing_time = time.time() - start_time
        
        result = {
            "data": screenshot_base64,
            "mime_type": "image/jpeg",
            "width": dimensions["width"],
            "height": dimensions["height"],
            "processing_time": processing_time,
            "success": True
        }
            
        logger.success(
            f"Screenshot captured: {dimensions['width']}x{dimensions['height']}",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Screenshot failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_click(
    selector: str,
    button: str = "left",
    click_count: int = 1,
    delay: int = 0,
    position_x: Optional[float] = None,
    position_y: Optional[float] = None,
    modifiers: Optional[List[str]] = None,
    force: bool = False,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Click on an element in the page.

    Finds an element in the current page using a CSS selector and clicks it.
    Supports various click options including position, modifiers, and multi-clicks.

    Args:
        selector: CSS selector to find the element to click.
        button: (Optional) Mouse button to use. Options: "left", "right", "middle". Default: "left".
        click_count: (Optional) Number of clicks (1 for single-click, 2 for double-click). Default: 1.
        delay: (Optional) Delay between mousedown and mouseup in milliseconds. Default: 0.
        position_x: (Optional) X-coordinate relative to the element to click at. If omitted,
                  clicks at the element's center.
        position_y: (Optional) Y-coordinate relative to the element to click at. If omitted,
                  clicks at the element's center.
        modifiers: (Optional) Keyboard modifiers to press during click. Options: "Alt", "Control", 
                 "Meta", "Shift". Example: ["Control", "Shift"].
        force: (Optional) Whether to bypass actionability checks (visibility, enabled state, etc.)
              Default: False.
        capture_snapshot: (Optional) Whether to capture a page snapshot after click. Default: True.

    Returns:
        A dictionary containing click results:
        {
            "success": true,
            "element_description": "Button with text 'Submit'",  # Description of clicked element
            "snapshot": { ... }  # Page snapshot after click (if capture_snapshot=True)
        }

    Raises:
        ToolError: If the click operation fails.
        ToolInputError: If the selector doesn't match any elements.
    """
    start_time = time.time()
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    # Validate button
    valid_buttons = ["left", "right", "middle"]
    if button not in valid_buttons:
        raise ToolInputError(
            f"Invalid button. Must be one of: {', '.join(valid_buttons)}",
            param_name="button",
            provided_value=button
        )
    
    # Validate click_count
    if click_count < 1:
        raise ToolInputError(
            "Click count must be at least 1",
            param_name="click_count",
            provided_value=click_count
        )
    
    # Validate modifiers
    valid_modifiers = ["Alt", "Control", "Meta", "Shift"]
    if modifiers:
        for modifier in modifiers:
            if modifier not in valid_modifiers:
                raise ToolInputError(
                    f"Invalid modifier: {modifier}. Must be one of: {', '.join(valid_modifiers)}",
                    param_name="modifiers",
                    provided_value=modifiers
                )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Check if element exists
        element = await page.query_selector(selector)
        if not element:
            raise ToolInputError(
                f"No element found matching selector: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Get element description for better logging
        element_description = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return 'Unknown element';
            
            // Try various properties to get a useful description
            const text = element.innerText?.trim();
            const alt = element.getAttribute('alt')?.trim();
            const ariaLabel = element.getAttribute('aria-label')?.trim();
            const title = element.getAttribute('title')?.trim();
            const value = element instanceof HTMLInputElement ? element.value : null;
            const placeholder = element instanceof HTMLInputElement ? element.placeholder : null;
            const tagName = element.tagName.toLowerCase();
            const type = element instanceof HTMLInputElement ? element.type : null;
            
            // Construct description
            let description = tagName;
            if (type) description += `[type="${type}"]`;
            
            if (text && text.length <= 50) description += ` with text '${text}'`;
            else if (ariaLabel) description += ` with aria-label '${ariaLabel}'`;
            else if (title) description += ` with title '${title}'`;
            else if (alt) description += ` with alt '${alt}'`;
            else if (value) description += ` with value '${value}'`;
            else if (placeholder) description += ` with placeholder '${placeholder}'`;
            
            return description;
        }""", selector)
        
        # Prepare click options
        click_options = {
            "button": button,
            "clickCount": click_count,
            "delay": delay,
            "force": force
        }
        
        if modifiers:
            click_options["modifiers"] = modifiers
            
        if position_x is not None and position_y is not None:
            click_options["position"] = {"x": position_x, "y": position_y}
        
        # Click element
        logger.info(
            f"Clicking on {element_description} ({selector})",
            emoji_key="click",
            button=button,
            click_count=click_count
        )
        
        await page.click(selector, **click_options)
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            # Wait a bit for any animations or page changes to complete
            await asyncio.sleep(0.5)
            
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "element_description": element_description,
            "processing_time": processing_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Successfully clicked {element_description}",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Click operation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            selector=selector
        )
        
        if "TimeoutError" in str(e):
            raise ToolError(
                status_code=408,
                detail=f"Timeout while clicking on element: {selector}" 
            ) from e
        
        raise ToolError(status_code=500, detail=error_msg) from e
 
@with_tool_metrics
@with_error_handling
async def browser_type(
    selector: str,
    text: str,
    delay: int = 0,
    clear_first: bool = True,
    press_enter: bool = False,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Type text into an input element.

    Finds an input element in the current page using a CSS selector and types text into it.
    Can optionally clear the field first and/or press Enter after typing.

    Args:
        selector: CSS selector to find the input element.
        text: Text to type into the element.
        delay: (Optional) Delay between keystrokes in milliseconds. Default: 0.
               Setting a delay can help with rate-limited inputs or triggering JS events.
        clear_first: (Optional) Whether to clear the input field before typing. Default: True.
        press_enter: (Optional) Whether to press Enter after typing. Default: False.
        capture_snapshot: (Optional) Whether to capture a page snapshot after typing. Default: True.

    Returns:
        A dictionary containing type results:
        {
            "success": true,
            "element_description": "Input field with placeholder 'Email'",  # Description of element
            "text": "user@example.com",  # Text that was typed
            "snapshot": { ... }  # Page snapshot after typing (if capture_snapshot=True)
        }

    Raises:
        ToolError: If the type operation fails.
        ToolInputError: If the selector doesn't match any elements or matches a non-typeable element.
    """
    start_time = time.time()
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    # Validate text
    if not isinstance(text, str):
        raise ToolInputError(
            "Text must be a string",
            param_name="text",
            provided_value=text
        )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Check if element exists and is typeable
        element = await page.query_selector(selector)
        if not element:
            raise ToolInputError(
                f"No element found matching selector: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Get element description for better logging
        element_description = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return 'Unknown element';
            
            // Try various properties to get a useful description
            const label = element.labels && element.labels.length > 0 ? 
                          element.labels[0].textContent?.trim() : null;
            const ariaLabel = element.getAttribute('aria-label')?.trim();
            const name = element.getAttribute('name')?.trim();
            const placeholder = element.getAttribute('placeholder')?.trim();
            const id = element.id ? element.id : null;
            const tagName = element.tagName.toLowerCase();
            const type = element instanceof HTMLInputElement ? element.type : null;
            
            // Construct description
            let description = tagName;
            if (type) description += `[type="${type}"]`;
            
            if (label) description += ` with label '${label}'`;
            else if (ariaLabel) description += ` with aria-label '${ariaLabel}'`;
            else if (placeholder) description += ` with placeholder '${placeholder}'`;
            else if (name) description += ` with name '${name}'`;
            else if (id) description += ` with id '${id}'`;
            
            return description;
        }""", selector)
        
        # Check if element is typeable
        is_typeable = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return false;
            
            const tagName = element.tagName.toLowerCase();
            const isInput = tagName === 'input' && !['checkbox', 'radio', 'file', 'button', 'submit', 'reset', 'image'].includes(element.type);
            const isTextarea = tagName === 'textarea';
            const isContentEditable = element.hasAttribute('contenteditable') && element.getAttribute('contenteditable') !== 'false';
            
            return isInput || isTextarea || isContentEditable;
        }""", selector)
        
        if not is_typeable:
            raise ToolInputError(
                f"Element is not typeable: {element_description}",
                param_name="selector",
                provided_value=selector
            )
        
        # Clear field if requested
        if clear_first:
            await page.evaluate("""(selector) => {
                const element = document.querySelector(selector);
                if (element) {
                    if (element.tagName.toLowerCase() === 'input' || element.tagName.toLowerCase() === 'textarea') {
                        element.value = '';
                    } else if (element.hasAttribute('contenteditable')) {
                        element.textContent = '';
                    }
                }
            }""", selector)
        
        # Type text
        logger.info(
            f"Typing text into {element_description}: {text if len(text) < 30 else text[:27] + '...'}",
            emoji_key="keyboard",
            text_length=len(text)
        )
        
        await page.type(selector, text, delay=delay)
        
        # Press Enter if requested
        if press_enter:
            await page.press(selector, "Enter")
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            # Wait a bit for any animations or page changes to complete
            await asyncio.sleep(0.5)
            
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "element_description": element_description,
            "text": text,
            "processing_time": processing_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Successfully typed text into {element_description}",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Type operation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            selector=selector
        )
        
        if "TimeoutError" in str(e):
            raise ToolError(
                status_code=408,
                detail=f"Timeout while typing into element: {selector}"
            ) from e
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_select(
    selector: str,
    values: Union[str, List[str]],
    by: str = "value",
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Select options from a dropdown or multi-select element.

    Finds a <select> element in the current page and selects one or more options.
    Options can be selected by value, label, or index.

    Args:
        selector: CSS selector to find the select element.
        values: Value(s) to select. Single string or list of strings for multi-select.
                What these values match depends on the 'by' parameter.
        by: (Optional) How to match options. Options:
           - "value": Match option by its value attribute (default)
           - "label": Match option by its visible text
           - "index": Match option by its index (0-based)
        capture_snapshot: (Optional) Whether to capture a page snapshot after selection. Default: True.

    Returns:
        A dictionary containing selection results:
        {
            "success": true,
            "element_description": "Select dropdown with label 'Country'",  # Description of element
            "selected_values": ["US"],  # Values that were selected
            "selected_labels": ["United States"],  # Labels of selected options
            "snapshot": { ... }  # Page snapshot after selection (if capture_snapshot=True)
        }

    Raises:
        ToolError: If the select operation fails.
        ToolInputError: If the selector doesn't match a select element or values are invalid.
    """
    start_time = time.time()
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    # Validate by parameter
    valid_by_options = ["value", "label", "index"]
    if by not in valid_by_options:
        raise ToolInputError(
            f"Invalid 'by' parameter. Must be one of: {', '.join(valid_by_options)}",
            param_name="by",
            provided_value=by
        )
    
    # Normalize values to list
    if isinstance(values, str):
        values_list = [values]
    else:
        values_list = values
    
    # Validate values
    if not values_list:
        raise ToolInputError(
            "Values cannot be empty",
            param_name="values",
            provided_value=values
        )
    
    # If selecting by index, validate that all values are valid integers
    if by == "index":
        try:
            index_values = [int(v) for v in values_list]
            values_list = [str(v) for v in index_values]  # Convert back to strings for Playwright API
        except ValueError as e:
            raise ToolInputError(
                "When selecting by index, all values must be valid integers",
                param_name="values",
                provided_value=values
            ) from e
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Check if element exists and is a select
        element = await page.query_selector(selector)
        if not element:
            raise ToolInputError(
                f"No element found matching selector: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Check if element is a select
        is_select = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            return element && element.tagName.toLowerCase() === 'select';
        }""", selector)
        
        if not is_select:
            raise ToolInputError(
                f"Element is not a select: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Get element description for better logging
        element_description = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return 'Unknown element';
            
            // Try various properties to get a useful description
            const label = element.labels && element.labels.length > 0 ? 
                          element.labels[0].textContent?.trim() : null;
            const ariaLabel = element.getAttribute('aria-label')?.trim();
            const name = element.getAttribute('name')?.trim();
            const id = element.id ? element.id : null;
            
            // Construct description
            let description = 'Select dropdown';
            
            if (label) description += ` with label '${label}'`;
            else if (ariaLabel) description += ` with aria-label '${ariaLabel}'`;
            else if (name) description += ` with name '${name}'`;
            else if (id) description += ` with id '${id}'`;
            
            return description;
        }""", selector)
        
        # Select options based on the 'by' parameter
        if by == "index":
            # When selecting by index, convert values to integers
            await page.select_option(selector, index=[int(v) for v in values_list])
        elif by == "label":
            await page.select_option(selector, label=values_list)
        else:  # by == "value", the default
            await page.select_option(selector, value=values_list)
        
        # Get selected values and labels
        selected_info = await page.evaluate("""(selector) => {
            const select = document.querySelector(selector);
            const selectedOptions = Array.from(select.selectedOptions);
            return {
                values: selectedOptions.map(option => option.value),
                labels: selectedOptions.map(option => option.textContent.trim())
            };
        }""", selector)
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "element_description": element_description,
            "selected_values": selected_info["values"],
            "selected_labels": selected_info["labels"],
            "processing_time": processing_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
        
        # Format message based on number of selections
        if len(selected_info["labels"]) == 1:
            success_message = f"Selected option '{selected_info['labels'][0]}' in {element_description}"
        else:
            success_message = f"Selected {len(selected_info['labels'])} options in {element_description}"
            
        logger.success(
            success_message,
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Select operation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            selector=selector
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_checkbox(
    selector: str,
    check: bool = True,
    force: bool = False,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Check or uncheck a checkbox/radio button element.

    Finds a checkbox or radio button in the current page and sets its checked state.
    
    Args:
        selector: CSS selector to find the checkbox/radio element.
        check: (Optional) Whether to check (true) or uncheck (false) the element. Default: True.
        force: (Optional) Whether to bypass actionability checks (visibility, enabled state, etc.)
               Default: False.
        capture_snapshot: (Optional) Whether to capture a page snapshot after the action. Default: True.

    Returns:
        A dictionary containing results:
        {
            "success": true,
            "element_description": "Checkbox with label 'Agree to terms'",  # Description of element
            "checked": true,  # Final state of the checkbox
            "snapshot": { ... }  # Page snapshot after action (if capture_snapshot=True)
        }

    Raises:
        ToolError: If the operation fails.
        ToolInputError: If the selector doesn't match a checkbox/radio or the element isn't checkable.
    """
    start_time = time.time()
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Check if element exists
        element = await page.query_selector(selector)
        if not element:
            raise ToolInputError(
                f"No element found matching selector: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Check if element is a checkbox or radio
        is_checkable = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return false;
            
            const tagName = element.tagName.toLowerCase();
            const isCheckboxOrRadio = tagName === 'input' && 
                                     (element.type === 'checkbox' || element.type === 'radio');
            
            return isCheckboxOrRadio;
        }""", selector)
        
        if not is_checkable:
            raise ToolInputError(
                f"Element is not a checkbox or radio button: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Get element description for better logging
        element_description = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return 'Unknown element';
            
            // Try various properties to get a useful description
            const label = element.labels && element.labels.length > 0 ? 
                          element.labels[0].textContent?.trim() : null;
            const ariaLabel = element.getAttribute('aria-label')?.trim();
            const name = element.getAttribute('name')?.trim();
            const id = element.id ? element.id : null;
            const type = element.type;
            
            // Construct description
            let description = type === 'checkbox' ? 'Checkbox' : 'Radio button';
            
            if (label) description += ` with label '${label}'`;
            else if (ariaLabel) description += ` with aria-label '${ariaLabel}'`;
            else if (name) description += ` with name '${name}'`;
            else if (id) description += ` with id '${id}'`;
            
            return description;
        }""", selector)
        
        # Get current checked state
        current_state = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            return element ? element.checked : false;
        }""", selector)
        
        # Only perform action if needed
        if current_state != check:
            if check:
                logger.info(
                    f"Checking {element_description}",
                    emoji_key="checkbox"
                )
                await page.check(selector, force=force)
            else:
                logger.info(
                    f"Unchecking {element_description}",
                    emoji_key="checkbox"
                )
                await page.uncheck(selector, force=force)
        else:
            logger.info(
                f"{element_description} already {'checked' if check else 'unchecked'}",
                emoji_key="checkbox"
            )
        
        # Verify final state
        final_state = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            return element ? element.checked : false;
        }""", selector)
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "element_description": element_description,
            "checked": final_state,
            "processing_time": processing_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        action_desc = "Checked" if check else "Unchecked"
        if current_state != check:
            logger.success(
                f"{action_desc} {element_description}",
                emoji_key=TaskType.BROWSER.value,
                time=processing_time
            )
        else:
            logger.success(
                f"{element_description} was already {action_desc.lower()}",
                emoji_key=TaskType.BROWSER.value,
                time=processing_time
            )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"{'Check' if check else 'Uncheck'} operation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            selector=selector
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_get_text(
    selector: str,
    trim: bool = True,
    include_hidden: bool = False
) -> Dict[str, Any]:
    """Get the text content of an element.

    Finds an element in the current page and extracts its text content.
    
    Args:
        selector: CSS selector to find the element.
        trim: (Optional) Whether to trim whitespace from the text. Default: True.
        include_hidden: (Optional) Whether to include text from hidden elements. Default: False.
               When false, matches the text visible to users.

    Returns:
        A dictionary containing results:
        {
            "success": true,
            "element_description": "Heading 'Welcome to Example'",  # Description of element
            "text": "Welcome to Example",  # Text content of the element
            "html": "<h1>Welcome to Example</h1>"  # Inner HTML of the element
        }

    Raises:
        ToolError: If the operation fails.
        ToolInputError: If the selector doesn't match any element.
    """
    start_time = time.time()
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Check if element exists
        element = await page.query_selector(selector)
        if not element:
            raise ToolInputError(
                f"No element found matching selector: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Get text content
        text_content = await page.evaluate("""(selector, trim, includeHidden) => {
            const element = document.querySelector(selector);
            if (!element) return '';
            
            let text;
            if (includeHidden) {
                // Get all text including hidden elements
                text = element.textContent || '';
            } else {
                // Get only visible text
                text = element.innerText || '';
            }
            
            return trim ? text.trim() : text;
        }""", selector, trim, include_hidden)
        
        # Get element description and inner HTML
        element_info = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return { description: 'Unknown element', innerHTML: '' };
            
            // Get tag name
            const tagName = element.tagName.toLowerCase();
            
            // Get element type
            let elementType = tagName;
            if (tagName === 'input') elementType = `${tagName}[type="${element.type}"]`;
            if (tagName === 'h1' || tagName === 'h2' || tagName === 'h3' || 
                tagName === 'h4' || tagName === 'h5' || tagName === 'h6') elementType = 'Heading';
            if (tagName === 'p') elementType = 'Paragraph';
            if (tagName === 'a') elementType = 'Link';
            if (tagName === 'button') elementType = 'Button';
            if (tagName === 'span' || tagName === 'div') elementType = 'Element';
            
            // Get short text preview
            const textContent = element.textContent || '';
            const textPreview = textContent.trim().substring(0, 40) + 
                              (textContent.length > 40 ? '...' : '');
            
            // Build description
            let description = elementType;
            if (textPreview) description += ` '${textPreview}'`;
            
            return { 
                description, 
                innerHTML: element.innerHTML || ''
            };
        }""", selector)
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "element_description": element_info["description"],
            "text": text_content,
            "html": element_info["innerHTML"],
            "processing_time": processing_time
        }
            
        logger.success(
            f"Retrieved text from {element_info['description']}",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Get text operation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            selector=selector
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_get_attributes(
    selector: str,
    attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get attributes of an element.

    Finds an element in the current page and returns its attributes.
    Can either get all attributes or just the specified ones.
    
    Args:
        selector: CSS selector to find the element.
        attributes: (Optional) List of specific attribute names to retrieve.
                   If not provided, all attributes will be returned.

    Returns:
        A dictionary containing results:
        {
            "success": true,
            "element_description": "Link 'Learn More'",  # Description of element
            "attributes": {  # Dictionary of attribute name/value pairs
                "href": "https://example.com",
                "class": "btn btn-primary",
                "id": "learn-more-btn"
            }
        }

    Raises:
        ToolError: If the operation fails.
        ToolInputError: If the selector doesn't match any element.
    """
    start_time = time.time()
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Check if element exists
        element = await page.query_selector(selector)
        if not element:
            raise ToolInputError(
                f"No element found matching selector: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Get attributes
        attrs = await page.evaluate("""(selector, attributesList) => {
            const element = document.querySelector(selector);
            if (!element) return {};
            
            const attributes = {};
            
            // If specific attributes are requested, get only those
            if (attributesList && attributesList.length > 0) {
                for (const attr of attributesList) {
                    if (element.hasAttribute(attr)) {
                        attributes[attr] = element.getAttribute(attr);
                    }
                }
            } else {
                // Get all attributes
                for (const attr of element.attributes) {
                    attributes[attr.name] = attr.value;
                }
            }
            
            return attributes;
        }""", selector, attributes)
        
        # Get element description
        element_desc = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return 'Unknown element';
            
            // Get tag name
            const tagName = element.tagName.toLowerCase();
            
            // Get element type
            let elementType = tagName;
            if (tagName === 'input') elementType = `${tagName}[type="${element.type}"]`;
            if (tagName === 'h1' || tagName === 'h2' || tagName === 'h3' || 
                tagName === 'h4' || tagName === 'h5' || tagName === 'h6') elementType = 'Heading';
            if (tagName === 'p') elementType = 'Paragraph';
            if (tagName === 'a') elementType = 'Link';
            if (tagName === 'button') elementType = 'Button';
            if (tagName === 'span' || tagName === 'div') elementType = 'Element';
            
            // Get short text preview
            const textContent = element.textContent || '';
            const textPreview = textContent.trim().substring(0, 40) + 
                              (textContent.length > 40 ? '...' : '');
            
            // Build description
            let description = elementType;
            if (textPreview) description += ` '${textPreview}'`;
            
            return description;
        }""", selector)
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "element_description": element_desc,
            "attributes": attrs,
            "processing_time": processing_time
        }
            
        # Format message based on number of attributes
        attr_count = len(attrs)
        logger.success(
            f"Retrieved {attr_count} attribute{'s' if attr_count != 1 else ''} from {element_desc}",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Get attributes operation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            selector=selector
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_download_file(
    url: Optional[str] = None,
    selector: Optional[str] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    wait_for_download: bool = True,
    timeout: int = 60000,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Download a file from the current page or a URL.

    Downloads a file either by directly navigating to a URL or by clicking a download link/button.
    File is saved to the specified location or a default downloads folder.
    
    Args:
        url: (Optional) Direct URL to the file to download. If provided, navigates to this URL.
             Only one of 'url' or 'selector' should be provided.
        selector: (Optional) CSS selector for a download link or button to click.
                 Only one of 'url' or 'selector' should be provided.
        save_path: (Optional) Directory path where the file should be saved.
                  If not provided, saved to the default downloads directory.
        filename: (Optional) Custom filename for the downloaded file.
                 If not provided, uses the filename from the server or response headers.
        wait_for_download: (Optional) Whether to wait for the download to complete. Default: True.
        timeout: (Optional) Maximum time to wait for download in milliseconds. Default: 60000 (60 seconds).
        overwrite: (Optional) If True, overwrites any existing file with the same name.
                  If False, adds a number suffix to avoid overwrites. Default: False.

    Returns:
        A dictionary containing download results:
        {
            "success": true,
            "file_path": "/path/to/downloaded/file.pdf",   # Absolute path to the downloaded file
            "file_name": "file.pdf",                       # Filename of the saved file
            "file_size": 1048576,                          # File size in bytes
            "content_type": "application/pdf",             # MIME type if available
            "download_time": 2.34                          # Download time in seconds
        }

    Raises:
        ToolError: If the download fails.
        ToolInputError: If neither URL nor selector is provided, or if both are provided.
    """
    start_time = time.time()
    
    # Validate inputs
    if (not url and not selector) or (url and selector):
        raise ToolInputError(
            "Exactly one of 'url' or 'selector' must be provided",
            param_name="url/selector",
            provided_value={"url": url, "selector": selector}
        )
    
    if url and not isinstance(url, str):
        raise ToolInputError(
            "URL must be a string",
            param_name="url",
            provided_value=url
        )
        
    if selector and not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a string",
            param_name="selector",
            provided_value=selector
        )
    
    # Determine save directory
    if save_path:
        save_dir = Path(save_path)
    else:
        # Use default download directory
        save_dir = Path(os.path.expanduser("~")) / "Downloads"
        
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure directory is writable
    if not os.access(save_dir, os.W_OK):
        raise ToolInputError(
            f"Directory is not writable: {save_dir}",
            param_name="save_path",
            provided_value=str(save_dir)
        )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Set download location
        context = await _ensure_context(await _ensure_browser())
        await context.set_default_timeout(timeout)
        
        # Create a tracker for the download
        download_promise = context.expect_download()
        
        # Initiate download
        if url:
            # Direct URL download
            logger.info(f"Navigating to download URL: {url}", emoji_key="download")
            await page.goto(url)
        else:
            # Click selector to initiate download
            logger.info(f"Clicking element to initiate download: {selector}", emoji_key="download")
            element = await page.query_selector(selector)
            if not element:
                raise ToolInputError(
                    f"No element found matching selector: {selector}",
                    param_name="selector",
                    provided_value=selector
                )
            await element.click()
        
        # Wait for download to start
        download = await download_promise
        
        # Get suggested filename
        suggested_filename = download.suggested_filename()
        
        # Determine final filename
        if filename:
            final_filename = filename
        else:
            final_filename = suggested_filename
        
        # Create full path
        file_path = save_dir / final_filename
        
        # Handle filename conflicts
        if file_path.exists() and not overwrite:
            base_name = file_path.stem
            extension = file_path.suffix
            counter = 1
            while file_path.exists():
                new_name = f"{base_name}_{counter}{extension}"
                file_path = save_dir / new_name
                counter += 1
            final_filename = file_path.name
        
        logger.info(
            f"Downloading file: {final_filename}",
            emoji_key="download",
            path=str(file_path)
        )
        
        # Wait for download to complete if requested
        if wait_for_download:
            # Save file to specified path
            await download.save_as(file_path)
            
            # Get file info
            file_size = file_path.stat().st_size
            
            # Try to determine content type
            content_type = None
            try:
                import mimetypes
                content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            except Exception:
                content_type = "application/octet-stream"
        else:
            # Don't wait for download to complete
            # Start download but don't wait
            asyncio.create_task(download.save_as(file_path))
            
            # We don't know file size yet
            file_size = 0
            content_type = None
        
        download_time = time.time() - start_time
        
        result = {
            "success": True,
            "file_path": str(file_path.absolute()),
            "file_name": file_path.name,
            "file_size": file_size,
            "content_type": content_type,
            "download_time": download_time,
            "complete": wait_for_download
        }
            
        logger.success(
            f"File download {'completed' if wait_for_download else 'initiated'}: {file_path.name}",
            emoji_key=TaskType.BROWSER.value,
            time=download_time,
            file_size=file_size
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Download failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            url=url,
            selector=selector
        )
        
        if "TimeoutError" in str(e):
            raise ToolError(
                status_code=408,
                detail=f"Download timed out after {timeout}ms"
            ) from e
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_upload_file(
    selector: str,
    file_paths: Union[str, List[str]],
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Upload files to a file input element.

    Finds a file input element on the page and uploads one or more files to it.
    
    Args:
        selector: CSS selector to find the file input element.
        file_paths: Path(s) to the file(s) to upload. Can be a single string path or
                   a list of paths for multiple file upload.
        capture_snapshot: (Optional) Whether to capture a page snapshot after upload. Default: True.

    Returns:
        A dictionary containing upload results:
        {
            "success": true,
            "element_description": "File input",  # Description of the file input element
            "uploaded_files": [                   # List of uploaded files
                {
                    "name": "document.pdf",
                    "path": "/path/to/document.pdf",
                    "size": 1048576
                }
            ],
            "snapshot": { ... }  # Page snapshot after upload (if capture_snapshot=True)
        }

    Raises:
        ToolError: If the upload fails.
        ToolInputError: If the selector doesn't match a file input or files don't exist.
    """
    start_time = time.time()
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    # Normalize file_paths to list
    if isinstance(file_paths, str):
        file_paths_list = [file_paths]
    else:
        file_paths_list = file_paths
    
    # Validate file paths
    if not file_paths_list:
        raise ToolInputError(
            "File paths cannot be empty",
            param_name="file_paths",
            provided_value=file_paths
        )
    
    # Check if files exist
    for file_path in file_paths_list:
        if not os.path.exists(file_path):
            raise ToolInputError(
                f"File does not exist: {file_path}",
                param_name="file_paths",
                provided_value=file_path
            )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Check if element exists
        element = await page.query_selector(selector)
        if not element:
            raise ToolInputError(
                f"No element found matching selector: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Check if element is a file input
        is_file_input = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            return element && 
                   element.tagName.toLowerCase() === 'input' && 
                   element.type.toLowerCase() === 'file';
        }""", selector)
        
        if not is_file_input:
            raise ToolInputError(
                f"Element is not a file input: {selector}",
                param_name="selector",
                provided_value=selector
            )
        
        # Get element description
        element_desc = await page.evaluate("""(selector) => {
            const element = document.querySelector(selector);
            if (!element) return 'Unknown element';
            
            const label = element.labels && element.labels.length > 0 ? 
                          element.labels[0].textContent?.trim() : null;
            const ariaLabel = element.getAttribute('aria-label')?.trim();
            const name = element.getAttribute('name')?.trim();
            const id = element.id ? element.id : null;
            
            let description = 'File input';
            
            if (label) description += ` with label '${label}'`;
            else if (ariaLabel) description += ` with aria-label '${ariaLabel}'`;
            else if (name) description += ` with name '${name}'`;
            else if (id) description += ` with id '${id}'`;
            
            return description;
        }""", selector)
        
        # Upload files
        logger.info(
            f"Uploading {len(file_paths_list)} file(s) to {element_desc}",
            emoji_key="upload",
            files=file_paths_list
        )
        
        await page.set_input_files(selector, file_paths_list)
        
        # Get file information
        uploaded_files = []
        for file_path in file_paths_list:
            path_obj = Path(file_path)
            uploaded_files.append({
                "name": path_obj.name,
                "path": str(path_obj.absolute()),
                "size": path_obj.stat().st_size
            })
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "element_description": element_desc,
            "uploaded_files": uploaded_files,
            "processing_time": processing_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Successfully uploaded {len(file_paths_list)} file(s) to {element_desc}",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"File upload failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            selector=selector
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_pdf(
    full_page: bool = True,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    scale: float = 1.0,
    landscape: bool = False,
    prefer_css_page_size: bool = False,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Save the current page as a PDF file.

    Captures the current page as a PDF document and saves it to the specified location.
    
    Args:
        full_page: (Optional) Whether to include the full scrollable area of the page. Default: True.
        save_path: (Optional) Directory path where the PDF should be saved.
                  If not provided, saved to the default downloads directory.
        filename: (Optional) Custom filename for the PDF file.
                 If not provided, uses the page title or a timestamp-based name.
        scale: (Optional) Scale of the webpage rendering (0.1-2.0). Default: 1.0.
        landscape: (Optional) Whether to use landscape orientation. Default: False (portrait).
        prefer_css_page_size: (Optional) Whether to prefer page size as defined in CSS. Default: False.
        overwrite: (Optional) If True, overwrites any existing file with the same name.
                  If False, adds a number suffix to avoid overwrites. Default: False.

    Returns:
        A dictionary containing PDF generation results:
        {
            "success": true,
            "file_path": "/path/to/saved/file.pdf",  # Absolute path to the saved PDF
            "file_name": "file.pdf",                 # Filename of the saved PDF
            "file_size": 1048576,                    # File size in bytes
            "page_count": 5,                         # Number of pages in the PDF
            "url": "https://example.com"             # URL of the page that was captured
        }

    Raises:
        ToolError: If PDF generation fails.
    """
    start_time = time.time()
    
    # Validate scale
    if not 0.1 <= scale <= 2.0:
        raise ToolInputError(
            "Scale must be between 0.1 and 2.0",
            param_name="scale",
            provided_value=scale
        )
    
    # Determine save directory
    if save_path:
        save_dir = Path(save_path)
    else:
        # Use default download directory
        save_dir = Path(os.path.expanduser("~")) / "Downloads"
        
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure directory is writable
    if not os.access(save_dir, os.W_OK):
        raise ToolInputError(
            f"Directory is not writable: {save_dir}",
            param_name="save_path",
            provided_value=str(save_dir)
        )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Get page title and URL for naming
        page_title = await page.title()
        page_url = page.url
        
        # Sanitize page title for filename
        def sanitize_filename(name: str) -> str:
            # Replace invalid characters with underscores
            return re.sub(r'[\\/*?:"<>|]', "_", name)
        
        # Determine filename
        if filename:
            final_filename = filename
            if not final_filename.lower().endswith('.pdf'):
                final_filename += '.pdf'
        else:
            if page_title:
                # Use page title
                sanitized_title = sanitize_filename(page_title)
                # Truncate if too long
                if len(sanitized_title) > 100:
                    sanitized_title = sanitized_title[:97] + "..."
                final_filename = f"{sanitized_title}.pdf"
            else:
                # Use timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                final_filename = f"page-{timestamp}.pdf"
        
        # Create full path
        file_path = save_dir / final_filename
        
        # Handle filename conflicts
        if file_path.exists() and not overwrite:
            base_name = file_path.stem
            extension = file_path.suffix
            counter = 1
            while file_path.exists():
                new_name = f"{base_name}_{counter}{extension}"
                file_path = save_dir / new_name
                counter += 1
            final_filename = file_path.name
        
        # Set PDF options
        pdf_options = {
            "path": str(file_path),
            "printBackground": True,
            "scale": scale,
            "landscape": landscape,
            "preferCSSPageSize": prefer_css_page_size
        }
        
        if full_page:
            # Full page PDF (default)
            logger.info(
                f"Generating PDF of full page: {page_title or page_url}",
                emoji_key="pdf"
            )
            await page.pdf(pdf_options)
        else:
            # Viewport-only PDF
            logger.info(
                f"Generating PDF of viewport: {page_title or page_url}",
                emoji_key="pdf"
            )
            # For viewport only, we need to get the viewport size
            # and set the width and height in the PDF options
            viewport_size = page.viewport_size
            pdf_options["width"] = viewport_size["width"]
            pdf_options["height"] = viewport_size["height"]
            await page.pdf(pdf_options)
        
        # Get file info
        file_size = file_path.stat().st_size
        
        # Try to count pages in PDF
        page_count = None
        try:
            # This is a simplified approach and may not work for all PDFs
            # In a real implementation, we'd use a proper PDF library
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
                # Count occurrences of "/Page" in the PDF
                page_count = content.count(b"/Type /Page")
        except Exception:
            # If counting fails, just skip it
            pass
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "file_path": str(file_path.absolute()),
            "file_name": file_path.name,
            "file_size": file_size,
            "url": page_url,
            "processing_time": processing_time
        }
        
        if page_count is not None:
            result["page_count"] = page_count
            
        logger.success(
            f"Successfully saved PDF: {file_path.name} ({file_size / 1024:.1f} KB)",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time,
            file_size=file_size
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"PDF generation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_tab_new(
    url: Optional[str] = None,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Open a new browser tab.

    Creates a new browser tab, optionally navigating to a specified URL.
    
    Args:
        url: (Optional) URL to navigate to in the new tab. If not provided, opens a blank page.
        capture_snapshot: (Optional) Whether to capture a page snapshot after the tab is created.
                         Default: True.

    Returns:
        A dictionary containing results:
        {
            "success": true,
            "tab_id": "abc123",                          # ID of the new tab
            "tab_index": 2,                              # Index of the tab (1-based)
            "url": "https://example.com",                # URL of the new tab (blank if no URL provided)
            "total_tabs": 3,                             # Total number of open tabs
            "snapshot": { ... }                          # Page snapshot (if capture_snapshot=True)
        }

    Raises:
        ToolError: If tab creation fails.
    """
    start_time = time.time()
    
    try:
        # Ensure browser and context
        browser = await _ensure_browser()
        context = await _ensure_context(browser)
        
        # Create new page (tab)
        logger.info("Creating new browser tab", emoji_key="browser")
        page = await context.new_page()
        
        # Generate tab ID
        tab_id = str(uuid.uuid4())
        
        # Set up page event handlers
        await _setup_page_event_handlers(page)
        
        # Store in global tabs dictionary
        global _pages, _current_page_id
        _pages[tab_id] = page
        _current_page_id = tab_id
        
        # Navigate to URL if provided
        if url:
            logger.info(f"Navigating to URL in new tab: {url}", emoji_key="browser")
            await page.goto(url, wait_until="load")
        
        # Get tabs and index info
        all_tabs = list(_pages.keys())
        tab_index = all_tabs.index(tab_id) + 1  # 1-based index
        total_tabs = len(all_tabs)
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            _snapshot_cache[tab_id] = snapshot_data
        
        current_url = page.url
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "tab_id": tab_id,
            "tab_index": tab_index,
            "url": current_url,
            "total_tabs": total_tabs,
            "processing_time": processing_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"New tab created successfully (index: {tab_index}, url: {current_url})",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to create new tab: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_tab_close(
    tab_index: Optional[int] = None,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Close a browser tab.

    Closes a browser tab by index. If no index is provided, closes the current tab.
    
    Args:
        tab_index: (Optional) Index of the tab to close (1-based). 
                  If not provided, closes the current tab.
        capture_snapshot: (Optional) Whether to capture a page snapshot of the newly focused tab.
                         Default: True (only if tabs remain open).

    Returns:
        A dictionary containing results:
        {
            "success": true,
            "closed_tab_index": 2,                       # Index of the closed tab
            "current_tab_index": 1,                      # Index of the now-current tab
            "total_tabs": 2,                             # Total number of remaining tabs
            "snapshot": { ... }                          # Page snapshot (if capture_snapshot=True and tabs remain)
        }

    Raises:
        ToolError: If tab closing fails.
        ToolInputError: If the specified tab index is invalid.
    """
    start_time = time.time()
    
    global _pages, _current_page_id
    
    # Validate inputs
    if not _pages:
        raise ToolError(
            status_code=400,
            detail="No browser tabs are open"
        )
    
    if tab_index is not None:
        if not isinstance(tab_index, int) or tab_index < 1 or tab_index > len(_pages):
            raise ToolInputError(
                f"Invalid tab index. Must be between 1 and {len(_pages)}",
                param_name="tab_index",
                provided_value=tab_index
            )
    
    try:
        # Get all tabs
        all_tabs = list(_pages.keys())
        
        # Determine which tab to close
        if tab_index is None:
            # Close current tab
            tab_to_close_id = _current_page_id
            tab_to_close_index = all_tabs.index(tab_to_close_id) + 1  # 1-based index
        else:
            # Close specified tab
            tab_to_close_id = all_tabs[tab_index - 1]  # Convert to 0-based index
            tab_to_close_index = tab_index
            
        # Get tab to close
        page_to_close = _pages[tab_to_close_id]
        
        logger.info(
            f"Closing browser tab (index: {tab_to_close_index})",
            emoji_key="browser"
        )
        
        # Close the tab
        await page_to_close.close()
        
        # Remove from our dictionary
        _pages.pop(tab_to_close_id)
        
        # Update current tab if we closed the current one
        if tab_to_close_id == _current_page_id:
            # Set current tab to the first remaining tab, if any
            if _pages:
                _current_page_id = list(_pages.keys())[0]
            else:
                _current_page_id = None
        
        # Get updated tabs info
        remaining_tabs = list(_pages.keys())
        total_tabs = len(remaining_tabs)
        
        # Get current tab index
        current_tab_index = None
        if _current_page_id:
            current_tab_index = remaining_tabs.index(_current_page_id) + 1  # 1-based index
        
        # Capture snapshot if requested and we have tabs remaining
        snapshot_data = None
        if capture_snapshot and _current_page_id:
            current_page = _pages[_current_page_id]
            snapshot_data = await _capture_snapshot(current_page)
            _snapshot_cache[_current_page_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "closed_tab_index": tab_to_close_index,
            "total_tabs": total_tabs,
            "processing_time": processing_time
        }
        
        if current_tab_index:
            result["current_tab_index"] = current_tab_index
            
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Tab closed successfully. {total_tabs} tab(s) remaining.",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, ToolInputError):
            raise
            
        error_msg = f"Failed to close tab: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_tab_list() -> Dict[str, Any]:
    """List all open browser tabs.

    Returns information about all currently open browser tabs.
    
    Args:
        None

    Returns:
        A dictionary containing tab information:
        {
            "success": true,
            "tabs": [                                # List of tab information
                {
                    "index": 1,                      # Tab index (1-based)
                    "id": "abc123",                  # Tab ID
                    "url": "https://example.com",    # Tab URL
                    "title": "Example Domain",       # Tab title
                    "is_current": true               # Whether this is the current tab
                },
                ...
            ],
            "total_tabs": 3,                         # Total number of open tabs
            "current_tab_index": 1                   # Index of the current tab
        }

    Raises:
        ToolError: If listing tabs fails.
    """
    start_time = time.time()
    
    global _pages, _current_page_id
    
    try:
        # Get all tabs
        all_tabs = list(_pages.keys())
        total_tabs = len(all_tabs)
        
        if total_tabs == 0:
            return {
                "success": True,
                "tabs": [],
                "total_tabs": 0,
                "current_tab_index": None,
                "processing_time": time.time() - start_time
            }
        
        # Build tabs list
        tabs_info = []
        
        for i, tab_id in enumerate(all_tabs):
            page = _pages[tab_id]
            
            # Get tab info
            url = page.url
            title = await page.title()
            is_current = tab_id == _current_page_id
            
            tabs_info.append({
                "index": i + 1,  # 1-based index
                "id": tab_id,
                "url": url,
                "title": title,
                "is_current": is_current
            })
        
        # Get current tab index
        current_tab_index = None
        if _current_page_id:
            current_tab_index = all_tabs.index(_current_page_id) + 1  # 1-based index
        
        processing_time = time.time() - start_time
        
        logger.success(
            f"Listed {total_tabs} browser tabs",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return {
            "success": True,
            "tabs": tabs_info,
            "total_tabs": total_tabs,
            "current_tab_index": current_tab_index,
            "processing_time": processing_time
        }
        
    except Exception as e:
        error_msg = f"Failed to list tabs: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_tab_select(
    tab_index: int,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Select and activate a browser tab by index.

    Switches to the specified tab, making it the active tab.
    
    Args:
        tab_index: Index of the tab to select (1-based).
        capture_snapshot: (Optional) Whether to capture a page snapshot after switching tabs.
                         Default: True.

    Returns:
        A dictionary containing results:
        {
            "success": true,
            "tab_index": 2,                          # Index of the selected tab
            "tab_id": "def456",                      # ID of the selected tab
            "url": "https://example.org",            # URL of the selected tab
            "title": "Example.org",                  # Title of the selected tab
            "snapshot": { ... }                      # Page snapshot (if capture_snapshot=True)
        }

    Raises:
        ToolError: If tab selection fails.
        ToolInputError: If the specified tab index is invalid.
    """
    start_time = time.time()
    
    global _pages, _current_page_id
    
    # Validate inputs
    if not _pages:
        raise ToolError(
            status_code=400,
            detail="No browser tabs are open"
        )
    
    if not isinstance(tab_index, int) or tab_index < 1 or tab_index > len(_pages):
        raise ToolInputError(
            f"Invalid tab index. Must be between 1 and {len(_pages)}",
            param_name="tab_index",
            provided_value=tab_index
        )
    
    try:
        # Get all tabs
        all_tabs = list(_pages.keys())
        
        # Get tab to select
        tab_to_select_id = all_tabs[tab_index - 1]  # Convert to 0-based index
        page_to_select = _pages[tab_to_select_id]
        
        # If already current tab, just return success
        if tab_to_select_id == _current_page_id:
            url = page_to_select.url
            title = await page_to_select.title()
            
            logger.info(
                f"Tab {tab_index} is already the current tab",
                emoji_key="browser"
            )
            
            # Capture snapshot if requested
            snapshot_data = None
            if capture_snapshot:
                snapshot_data = await _capture_snapshot(page_to_select)
                _snapshot_cache[tab_to_select_id] = snapshot_data
            
            result = {
                "success": True,
                "tab_index": tab_index,
                "tab_id": tab_to_select_id,
                "url": url,
                "title": title,
                "processing_time": time.time() - start_time
            }
            
            if snapshot_data:
                result["snapshot"] = snapshot_data
                
            return result
        
        # Bring the page to front
        logger.info(
            f"Selecting browser tab (index: {tab_index})",
            emoji_key="browser"
        )
        
        await page_to_select.bring_to_front()
        
        # Update current tab
        _current_page_id = tab_to_select_id
        
        # Get tab info
        url = page_to_select.url
        title = await page_to_select.title()
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page_to_select)
            _snapshot_cache[tab_to_select_id] = snapshot_data
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "tab_index": tab_index,
            "tab_id": tab_to_select_id,
            "url": url,
            "title": title,
            "processing_time": processing_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Tab selected successfully (index: {tab_index}, url: {url})",
            emoji_key=TaskType.BROWSER.value,
            time=processing_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, ToolInputError):
            raise
            
        error_msg = f"Failed to select tab: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_execute_javascript(
    script: str,
    selector: Optional[str] = None,
    args: Optional[List[Any]] = None,
    timeout: int = 30000
) -> Dict[str, Any]:
    """Execute JavaScript code in the browser page.

    Runs arbitrary JavaScript code in the context of the current page.
    The code can interact with the page DOM and return data back to Python.
    
    Args:
        script: JavaScript code to execute.
        selector: (Optional) CSS selector. If provided, the script runs in the context
                 of the first element matching the selector.
        args: (Optional) List of arguments to pass to the script.
        timeout: (Optional) Maximum time to wait for script execution in milliseconds.
                Default: 30000 (30 seconds).

    Returns:
        A dictionary containing execution results:
        {
            "success": true,
            "result": {...},  # Value returned by the JavaScript code (serializable to JSON)
            "execution_time": 0.123  # Script execution time in seconds
        }

    Raises:
        ToolError: If script execution fails.
        ToolInputError: If the script is invalid or selector doesn't match any element.
    """
    start_time = time.time()
    
    # Validate script
    if not script or not isinstance(script, str):
        raise ToolInputError(
            "Script must be a non-empty string",
            param_name="script",
            provided_value=script
        )
    
    # Normalize args
    if args is None:
        args = []
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Set execution timeout
        page.set_default_timeout(timeout)
        
        # Execute script
        if selector:
            # Check if element exists
            element = await page.query_selector(selector)
            if not element:
                raise ToolInputError(
                    f"No element found matching selector: {selector}",
                    param_name="selector",
                    provided_value=selector
                )
                
            logger.info(
                f"Executing JavaScript on element: {selector}",
                emoji_key="javascript",
                script_length=len(script)
            )
            
            # Execute in element context
            result = await element.evaluate(script, *args)
        else:
            # Execute in page context
            logger.info(
                "Executing JavaScript in page context",
                emoji_key="javascript",
                script_length=len(script)
            )
            
            result = await page.evaluate(script, *args)
        
        execution_time = time.time() - start_time
        
        # Try to make result JSON-serializable
        try:
            # Test if result is JSON-serializable
            json.dumps(result)
            serialized_result = result
        except (TypeError, OverflowError):
            # If not serializable, convert to string
            if result is None:
                serialized_result = None
            else:
                serialized_result = str(result)
        
        logger.success(
            "JavaScript execution completed successfully",
            emoji_key=TaskType.BROWSER.value,
            time=execution_time
        )
        
        return {
            "success": True,
            "result": serialized_result,
            "execution_time": execution_time
        }
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"JavaScript execution failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error"
        )
        
        if "TimeoutError" in str(e):
            raise ToolError(
                status_code=408,
                detail=f"Script execution timed out after {timeout}ms"
            ) from e
        
        raise ToolError(status_code=500, detail=error_msg) from e

@with_tool_metrics
@with_error_handling
async def browser_wait(
    wait_type: str,
    value: str,
    timeout: int = 30000,
    state: Optional[str] = None,
    capture_snapshot: bool = True
) -> Dict[str, Any]:
    """Wait for specific conditions on the page before proceeding.

    Pauses execution until a specified condition is met, such as an element
    appearing, a URL changing, or a navigation completing.
    
    Args:
        wait_type: Type of wait condition. Options:
                 - "selector": Wait for an element matching a CSS selector
                 - "navigation": Wait for navigation to complete
                 - "url": Wait for URL to contain a specific string
                 - "function": Wait for a JavaScript function to return true
                 - "load_state": Wait for a certain load state
                 - "time": Wait for a specific amount of time (milliseconds)
        value: The value to wait for, based on wait_type:
              - For "selector": CSS selector string
              - For "navigation"/"url": URL string
              - For "function": JavaScript function body as string
              - For "load_state": State name (see state parameter)
              - For "time": Number of milliseconds as string
        timeout: (Optional) Maximum time to wait in milliseconds. Default: 30000 (30 seconds).
        state: (Optional) Specific state for "load_state" wait_type. Options:
              - "load": Wait for the 'load' event
              - "domcontentloaded": Wait for the 'DOMContentLoaded' event
              - "networkidle": Wait for network to be idle
              Default is "load" for "load_state" wait type.
        capture_snapshot: (Optional) Whether to capture a page snapshot after waiting.
                         Default: True.

    Returns:
        A dictionary containing wait results:
        {
            "success": true,
            "wait_time": 1.23,          # Actual time waited in seconds
            "wait_type": "selector",    # Type of wait performed
            "snapshot": { ... }         # Page snapshot after waiting (if capture_snapshot=True)
        }

    Raises:
        ToolError: If the wait condition is not met before the timeout.
        ToolInputError: If invalid wait_type or parameters are provided.
    """
    start_time = time.time()
    
    # Validate wait_type
    valid_wait_types = ["selector", "navigation", "url", "function", "load_state", "time"]
    if wait_type not in valid_wait_types:
        raise ToolInputError(
            f"Invalid wait_type. Must be one of: {', '.join(valid_wait_types)}",
            param_name="wait_type",
            provided_value=wait_type
        )
    
    # Validate value
    if not value and wait_type != "time":
        raise ToolInputError(
            "Value must be provided",
            param_name="value",
            provided_value=value
        )
    
    # Validate state for load_state
    valid_states = ["load", "domcontentloaded", "networkidle"]
    if wait_type == "load_state" and state and state not in valid_states:
        raise ToolInputError(
            f"Invalid state for load_state. Must be one of: {', '.join(valid_states)}",
            param_name="state",
            provided_value=state
        )
    
    try:
        # Get current page
        _, page = await _ensure_page()
        
        # Set default timeout for page operations
        page.set_default_timeout(timeout)
        
        # Perform wait based on wait_type
        if wait_type == "selector":
            logger.info(f"Waiting for selector: {value}", emoji_key="wait")
            await page.wait_for_selector(value, timeout=timeout)
            
        elif wait_type == "navigation":
            logger.info("Waiting for navigation to complete", emoji_key="wait")
            await page.wait_for_navigation(url=value if value else None, timeout=timeout)
            
        elif wait_type == "url":
            logger.info(f"Waiting for URL to contain: {value}", emoji_key="wait")
            await page.wait_for_url(f"**/*{value}*", timeout=timeout)
            
        elif wait_type == "function":
            logger.info("Waiting for JavaScript function to return true", emoji_key="wait")
            # Create function from string
            js_function = f"() => {{ {value} }}"
            await page.wait_for_function(js_function, timeout=timeout)
            
        elif wait_type == "load_state":
            load_state = state or "load"
            logger.info(f"Waiting for page load state: {load_state}", emoji_key="wait")
            await page.wait_for_load_state(load_state, timeout=timeout)
            
        elif wait_type == "time":
            try:
                # Convert value to milliseconds
                wait_ms = int(value)
                logger.info(f"Waiting for {wait_ms} milliseconds", emoji_key="wait")
                # Cap wait time to timeout for safety
                actual_wait_ms = min(wait_ms, timeout)
                await asyncio.sleep(actual_wait_ms / 1000)  # Convert to seconds for asyncio.sleep
            except ValueError as e:
                raise ToolInputError(
                    "For time wait_type, value must be a valid integer (milliseconds)",
                    param_name="value",
                    provided_value=value
                ) from e
        
        # Capture snapshot if requested
        snapshot_data = None
        if capture_snapshot:
            snapshot_data = await _capture_snapshot(page)
            page_id = _current_page_id or "unknown"
            _snapshot_cache[page_id] = snapshot_data
        
        wait_time = time.time() - start_time
        
        result = {
            "success": True,
            "wait_time": wait_time,
            "wait_type": wait_type,
            "processing_time": wait_time
        }
        
        if snapshot_data:
            result["snapshot"] = snapshot_data
            
        logger.success(
            f"Wait condition satisfied after {wait_time:.2f} seconds",
            emoji_key=TaskType.BROWSER.value,
            time=wait_time
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ToolError, ToolInputError)):
            raise
            
        error_msg = f"Wait operation failed: {str(e)}"
        logger.error(
            error_msg,
            emoji_key="error",
            wait_type=wait_type,
            value=value
        )
        
        if "TimeoutError" in str(e):
            raise ToolError(
                status_code=408,
                detail=f"Wait operation timed out after {timeout}ms: {wait_type}={value}"
            ) from e
        
        raise ToolError(status_code=500, detail=error_msg) from e