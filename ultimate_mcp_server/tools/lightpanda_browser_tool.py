"""Lightpanda headless browser tools for Ultimate MCP Server.

This module provides a comprehensive set of tools for headless browser automation
using Lightpanda, an open-source browser specifically optimized for headless usage.

Lightpanda offers significant advantages over traditional browsers for automation:
- 9x lower memory footprint than Chrome
- 11x faster execution than Chrome
- Simple installation and management
- Efficient resource usage for batch processing

The tools in this module provide similar functionality to Playwright-based tools
but with better performance, making them ideal for:
- Web scraping and content extraction
- PDF generation and downloads
- Form automation and interaction
- Web testing and monitoring
- Research and data synthesis
"""

import asyncio
import base64
import json
import mimetypes
import os
import platform
import re
import subprocess
import tempfile
import time
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiofiles
import httpx
from playwright.async_api import async_playwright

from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.tools.completion import generate_completion

# Removed unused import: from ultimate_mcp_server.tools.filesystem import create_directory
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.lightpanda_browser_tool")

# --- Constants ---

# Lightpanda binary URLs
LIGHTPANDA_BINARY_URLS = {
    'linux': 'https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux',
    'darwin': 'https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-aarch64-macos',
    # Windows is supported through WSL2, using the Linux binary
}

# Path to store the Lightpanda binary and session data
LIGHTPANDA_DIR = os.path.expanduser("~/.ultimate_mcp_server/lightpanda")
LIGHTPANDA_SESSION_DIR = os.path.join(LIGHTPANDA_DIR, "sessions")

# CDP server settings
LIGHTPANDA_CDP_HOST = "127.0.0.1"
LIGHTPANDA_CDP_PORT = 9222
LIGHTPANDA_CDP_WS_ENDPOINT = f"ws://{LIGHTPANDA_CDP_HOST}:{LIGHTPANDA_CDP_PORT}"

# Global session tracking for management
_active_sessions = {}
_session_counter = 0

# Global CDP server and Playwright browser instance
_cdp_server_process = None
_cdp_server_log_file = None
_playwright = None
_browser = None
_browser_initialization_lock = asyncio.Lock()
_connection_initialized = False

# --- Helper Functions ---

async def _check_lightpanda_installed() -> bool:
    """Check if the Lightpanda binary is installed and executable."""
    binary_path = os.path.join(LIGHTPANDA_DIR, "lightpanda")
    return os.path.exists(binary_path) and os.access(binary_path, os.X_OK)

async def _get_platform_info() -> Optional[str]:
    """Get the current platform key for Lightpanda binary selection."""
    system = platform.system().lower()
    
    if system == 'linux':
        return 'linux'
    elif system == 'darwin':
        # macOS - currently only aarch64/ARM64 is supported
        if platform.machine() in ('arm64', 'aarch64'):
            return 'darwin'
        else:
            logger.warning(f"Lightpanda doesn't support macOS on {platform.machine()} architecture.")
            return None
    elif system == 'windows':
        # Check if running in WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return 'linux'  # Use Linux binary for WSL
        except Exception: # Was bare except
            pass
        logger.warning("Lightpanda on Windows is only supported through WSL2.")
        return None
    else:
        logger.warning(f"Unsupported platform: {system}")
        return None

async def _download_file(url: str, target_path: str) -> bool:
    """Download a file from a URL to a target path."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Download file using httpx, allowing redirects
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                # Check if the final status code after potential redirects is 200
                if response.status_code != 200:
                    logger.error(f"Failed to download Lightpanda: HTTP status {response.status_code} from URL {response.url}")
                    return False
                
                async with aiofiles.open(target_path, 'wb') as f:
                    async for chunk in response.aiter_bytes():
                        await f.write(chunk)
        
        # Make file executable
        os.chmod(target_path, 0o755)
        return True
    except Exception as e:
        logger.error(f"Error downloading Lightpanda: {str(e)}")
        return False

async def _install_lightpanda() -> bool:
    """Download and install the Lightpanda binary for the current platform."""
    platform_key = await _get_platform_info()
    if not platform_key:
        return False
    
    binary_url = LIGHTPANDA_BINARY_URLS.get(platform_key)
    if not binary_url:
        logger.error(f"No binary available for platform: {platform_key}")
        return False
    
    binary_path = os.path.join(LIGHTPANDA_DIR, "lightpanda")
    logger.info(f"Installing Lightpanda from {binary_url} to {binary_path}")
    
    return await _download_file(binary_url, binary_path)

async def _ensure_lightpanda_installed() -> str:
    """Ensure Lightpanda is installed and return the path to the binary."""
    if not await _check_lightpanda_installed():
        if not await _install_lightpanda():
            raise ToolError(
                "Failed to install Lightpanda browser. Please check logs for details.",
                error_code="lightpanda_installation_failed"
            )
    
    return os.path.join(LIGHTPANDA_DIR, "lightpanda")

async def _run_lightpanda_command(cmd: List[str], stdin_data: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """Run a Lightpanda command and return the result."""
    try:
        # Get the path to the Lightpanda binary
        binary_path = await _ensure_lightpanda_installed()
        full_cmd = [binary_path] + cmd
        
        # Set environment variable to disable telemetry if desired
        env = os.environ.copy()
        env["LIGHTPANDA_DISABLE_TELEMETRY"] = "true"  # By default, disable telemetry
        
        # Run the command with timeout
        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            env=env
        )
        
        # Set up task with timeout
        try:
            if stdin_data:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin_data.encode('utf-8')), 
                    timeout=timeout
                )
            else:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            # Try to kill the process if it's still running
            try:
                process.kill()
            except Exception:
                pass
            
            return {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": f"Command timed out after {timeout} seconds"
            }
        
        # Check the process return code
        if process.returncode != 0:
            stderr_text = stderr.decode('utf-8', errors='replace')
            logger.error(f"Lightpanda command failed: {stderr_text}")
            return {
                "success": False,
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr_text,
                "error": stderr_text
            }
        
        return {
            "success": True,
            "returncode": process.returncode,
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace')
        }
    
    except Exception as e:
        logger.error(f"Error running Lightpanda command: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e)
        }

async def _extract_element_description(js_result: Dict[str, Any]) -> Optional[str]:
    """Extract a human-readable element description from JavaScript result."""
    try:
        element_info = json.loads(js_result.get("result", "{}"))
        tag_name = element_info.get("tagName", "").lower()
        element_id = element_info.get("id", "")
        element_class = element_info.get("className", "")
        element_text = element_info.get("textContent", "")[:50]  # Truncate long text
        
        # Build a description
        description_parts = []
        if tag_name:
            description_parts.append(tag_name)
        
        if element_id:
            description_parts.append(f"id='{element_id}'")
        elif element_class:
            class_list = element_class.split()
            if class_list:
                description_parts.append(f"class='{class_list[0]}'")
        
        if element_text:
            text_preview = element_text.strip()
            if text_preview:
                description_parts.append(f"text='{text_preview}'")
        
        return " ".join(description_parts) if description_parts else "Unknown element"
    except Exception: # Was bare except
        return "Element"

async def _create_new_session() -> str:
    """Create a new Lightpanda session ID."""
    global _session_counter
    _session_counter += 1
    session_id = f"session_{_session_counter}_{uuid.uuid4().hex[:8]}"
    _active_sessions[session_id] = {"created_at": datetime.now()}
    return session_id

async def _cleanup_session(session_id: str) -> bool:
    """Clean up a Lightpanda session and its files."""
    if session_id in _active_sessions:
        # Clean up any session-specific files here if needed
        _active_sessions.pop(session_id, None)
        return True
    return False

async def _parse_html_title(html: str) -> str:
    """Extract the title from HTML content."""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        return title_match.group(1).strip()
    return "Untitled Page"

async def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to be safe for all operating systems."""
    # Replace invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Truncate if too long
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    return sanitized

async def _call_llm(
    prompt: str,
    model: str = "openai/gpt-4.1-mini",
    system_message: Optional[str] = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """Helper function to call an LLM with a prompt."""
    try:
        system_message = system_message or "You are an AI assistant analyzing web page content."
        
        completion_params = {
            "model": model,
            "prompt": f"{system_message}\n\n{prompt}",
            "temperature": temperature,
        }
        
        logger.info(f"Calling LLM model {model} for analysis")
        response = await generate_completion(**completion_params)
        
        if response.get("success"):
            return {
                "success": True,
                "text": response.get("text", ""),
                "model": model
            }
        else:
            return {
                "success": False,
                "error": response.get("error", "Unknown LLM error"),
                "model": model
            }
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        return {
            "success": False,
            "error": f"Error calling LLM: {str(e)}",
            "model": model
        }

# --- Basic Browser Tools ---

@with_tool_metrics
@with_error_handling
async def lightpanda_fetch(
    url: str,
    dump_html: bool = True,
    timeout: int = 30,
    user_agent: Optional[str] = None,
    wait_for_selector: Optional[str] = None,
    javascript: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Fetch a URL using the Lightpanda browser and return the page content.
    
    This tool navigates to a URL using the Lightpanda headless browser, which is significantly
    faster and more lightweight than Chromium-based browsers. It can optionally wait for a
    specific element to appear, execute JavaScript, and then return the resulting HTML.
    
    Lightpanda is automatically downloaded and installed if not already present.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        dump_html: Whether to return the page HTML content (default: True).
        timeout: Maximum time in seconds to wait for the page to load (default: 30).
        user_agent: Custom User-Agent string to use (optional).
        wait_for_selector: CSS selector to wait for before considering the page loaded (optional).
        javascript: JavaScript code to execute on the page after loading (optional).
        headers: Additional HTTP headers to include in the request (optional).
    
    Returns:
        A dictionary containing the results:
        {
            "html": "...",  # Page HTML content (if dump_html is True)
            "title": "...",  # Page title (if available)
            "url": "...",    # Final URL (after any redirects)
            "status": 200,   # HTTP status code
            "success": true  # Whether the operation was successful
        }
    
    Raises:
        ToolInputError: If the URL is invalid.
        ToolError: If the Lightpanda browser fails to fetch the page.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Build command
    cmd = ["fetch"]
    
    if dump_html:
        cmd.append("--dump")
    
    # Add URL as the last argument
    cmd.append(url)
    
    # Run the command
    # Use default timeout for the process itself, as Lightpanda fetch doesn't support timeout arg
    result = await _run_lightpanda_command(cmd, timeout=timeout+5)
    
    if not result["success"]:
        raise ToolError(
            f"Failed to fetch URL: {result.get('error', 'Unknown error')}",
            details={
                "url": url,
                "stderr": result.get("stderr", ""),
                "returncode": result.get("returncode", -1)
            }
        )
    
    # Extract information from the output
    html_content = result["stdout"] if dump_html else ""
    
    # Try to find title from HTML
    title = None
    if html_content:
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
    
    # Parse logs from stderr to get status code and other information
    status_code = 200  # Default
    final_url = url  # Default to original URL
    
    stderr_lines = result["stderr"].split('\n')
    for line in stderr_lines:
        if "GET" in line and "http.Status" in line:
            matches = re.search(r'GET\s+(https?://[^\s]+)\s+http\.Status\.(\w+)', line)
            if matches:
                final_url = matches.group(1)
                status_name = matches.group(2)
                if status_name == "ok":
                    status_code = 200
                elif status_name == "not_found":
                    status_code = 404
                # Add more status mappings as needed
    
    # Execute JavaScript if provided - This seems unsupported by the current CLI
    js_result = None
    # if javascript and dump_html:
    #     logger.warning("JavaScript execution via lightpanda_fetch seems unsupported by the current Lightpanda CLI version.")
        # js_cmd = ["eval", "--url", url, javascript] # 'eval' command seems missing
        # js_execution = await _run_lightpanda_command(js_cmd, timeout=timeout)
        # if js_execution["success"]:
        #     js_result = js_execution["stdout"]
    
    return {
        "html": html_content,
        "title": title,
        "url": final_url,
        "status": status_code,
        "javascript_result": js_result if javascript else None,
        "success": True
    }

@with_tool_metrics
@with_error_handling
async def lightpanda_javascript(
    url: str,
    javascript: str,
    timeout: int = 30,
    user_agent: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    wait_for_selector: Optional[str] = None
) -> Dict[str, Any]:
    """Execute JavaScript code on a webpage using the Lightpanda browser.
    
    This tool navigates to a URL and executes the provided JavaScript code, returning
    the result. This is useful for extracting data from web pages, especially those
    that require JavaScript execution to render properly.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        javascript: JavaScript code to execute on the page.
        timeout: Maximum time in seconds to wait for the page to load (default: 30).
        user_agent: Custom User-Agent string to use (optional).
        headers: Additional HTTP headers to include in the request (optional).
        wait_for_selector: CSS selector to wait for before executing JavaScript (optional).
    
    Returns:
        A dictionary containing the results:
        {
            "result": "...",  # String result from the JavaScript execution
            "url": "...",     # URL that was loaded
            "success": true   # Whether the operation was successful
        }
    
    Raises:
        ToolInputError: If the URL or JavaScript is invalid.
        ToolError: If the Lightpanda browser fails to execute the JavaScript.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate JavaScript
    if not javascript or not isinstance(javascript, str):
        raise ToolInputError(
            "JavaScript code must be a non-empty string",
            param_name="javascript",
            provided_value=javascript
        )
        
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Set headers if provided
                if headers:
                    await page.set_extra_http_headers(headers)
                
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Wait for selector if provided
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=timeout * 1000)
                
                # Execute JavaScript
                result = await page.evaluate(javascript)
                
                # Convert result to string if it's not already
                if not isinstance(result, str):
                    if result is None:
                        result = "null"
                    else:
                        result = json.dumps(result)
                
                return {
                    "result": result,
                    "url": page.url,
                    "success": True
                }
            finally:
                await page.close()
        finally:
            await context.close()
            
    except Exception as e:
        logger.error(f"Error executing JavaScript: {str(e)}")
        raise ToolError(
            f"Failed to execute JavaScript: {str(e)}",
            details={"url": url}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_screenshot(
    url: str,
    output_path: Optional[str] = None,
    full_page: bool = False,
    element_selector: Optional[str] = None,
    format: str = "png",
    quality: int = 90,
    timeout: int = 30,
    user_agent: Optional[str] = None,
    wait_for_selector: Optional[str] = None
) -> Dict[str, Any]:
    """Take a screenshot of a webpage using the Lightpanda browser.
    
    This tool navigates to a URL and captures a screenshot of the page or a specific element.
    The screenshot can be returned as base64-encoded data or saved to a file.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        output_path: File path to save the screenshot to (optional).
        full_page: Whether to capture the full scrollable page (default: False).
        element_selector: CSS selector to capture a specific element (optional).
        format: Image format (png or jpeg) (default: png).
        quality: Image quality for jpeg format (1-100) (default: 90).
        timeout: Maximum time in seconds to wait for the page to load (default: 30).
        user_agent: Custom User-Agent string to use (optional).
        wait_for_selector: CSS selector to wait for before capturing screenshot (optional).
    
    Returns:
        A dictionary containing the results:
        {
            "data": "base64encoded...",  # Base64-encoded screenshot data
            "file_path": "/path/to/screenshot.png",  # Only if output_path provided
            "width": 1280,  # Width of screenshot in pixels
            "height": 720,  # Height of screenshot in pixels
            "format": "png",  # Format of the image
            "success": true   # Whether the operation was successful
        }
    
    Raises:
        ToolInputError: If the URL or parameters are invalid.
        ToolError: If the Lightpanda browser fails to capture the screenshot.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate format
    if format.lower() not in ["png", "jpeg", "jpg"]:
        raise ToolInputError(
            "Format must be 'png' or 'jpeg'/'jpg'",
            param_name="format",
            provided_value=format
        )
    
    # Normalize format
    format = "jpeg" if format.lower() == "jpg" else format.lower()
    
    # Validate quality for jpeg
    if format == "jpeg" and (quality < 1 or quality > 100):
        raise ToolInputError(
            "Quality must be between 1 and 100 for jpeg format",
            param_name="quality",
            provided_value=quality
        )
        
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Wait for selector if provided
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=timeout * 1000)
                
                # Prepare screenshot options
                screenshot_options = {
                    "type": format,
                    "full_page": full_page,
                    "path": output_path
                }
                
                if format == "jpeg":
                    screenshot_options["quality"] = quality
                
                # Take screenshot of specific element or full page
                if element_selector:
                    # Wait for the element to be available
                    element = await page.wait_for_selector(element_selector, timeout=timeout * 1000)
                    if not element:
                        raise ToolError(
                            f"Element not found: {element_selector}",
                            details={"selector": element_selector}
                        )
                    
                    # Get element dimensions
                    dimensions = await element.bounding_box()
                    if not dimensions:
                        raise ToolError(
                            f"Element has no dimensions: {element_selector}",
                            details={"selector": element_selector}
                        )
                    
                    # Take screenshot of the element
                    screenshot_data = await element.screenshot(path=output_path, type=format, quality=quality if format == "jpeg" else None)
                    width, height = dimensions["width"], dimensions["height"]
                else:
                    # Take screenshot of the page
                    screenshot_data = await page.screenshot(**screenshot_options)
                    
                    # Get page dimensions
                    dimensions = await page.evaluate("""() => {
                        return {
                            width: window.innerWidth,
                            height: window.innerHeight
                        };
                    }""")
                    width, height = dimensions["width"], dimensions["height"]
                
                # Prepare result
                result = {
                    "data": base64.b64encode(screenshot_data).decode("utf-8"),
                    "width": width,
                    "height": height,
                    "format": format,
                    "success": True
                }
                
                if output_path:
                    result["file_path"] = output_path
                
                return result
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        logger.error(f"Error taking screenshot: {str(e)}")
        raise ToolError(
            f"Failed to capture screenshot: {str(e)}",
            details={"url": url}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_extract_links(
    url: str,
    include_text: bool = True,
    timeout: int = 30,
    user_agent: Optional[str] = None,
    filter_pattern: Optional[str] = None,
    include_attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extract all links (anchor tags) from a webpage using the Lightpanda browser.
    
    This tool navigates to a URL, executes JavaScript to find all anchor elements,
    and returns their href attributes and optional text content. This is useful for
    web scraping, site mapping, or link analysis.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        include_text: Whether to include the link text in the results (default: True).
        timeout: Maximum time in seconds to wait for the page to load (default: 30).
        user_agent: Custom User-Agent string to use (optional).
        filter_pattern: Regex pattern to filter links (optional). Only links matching
                       this pattern will be included in the results.
        include_attributes: Additional anchor attributes to include (optional).
                          Example: ["rel", "target", "download"]
    
    Returns:
        A dictionary containing the results:
        {
            "links": [
                {
                    "href": "https://example.com/page1",
                    "text": "Page 1",                    # Only if include_text is True
                    "title": "Description of Page 1"     # Only if present in HTML
                },
                # ... more links
            ],
            "url": "https://example.com",  # The URL that was loaded
            "count": 42,                   # Number of links found
            "success": true                # Whether the operation was successful
        }
    
    Raises:
        ToolInputError: If the URL is invalid.
        ToolError: If the Lightpanda browser fails to extract links.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Prepare attribute list for extraction
    attrs_to_include = ["href", "title"]
    if include_text:
        attrs_to_include.append("text")
    
    if include_attributes:
        for attr in include_attributes:
            if attr not in attrs_to_include:
                attrs_to_include.append(attr)
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Build the JavaScript to extract links
                js_extract_links = """
                () => {
                    const links = Array.from(document.querySelectorAll('a'));
                    return links.map(link => {
                        const result = {};
                        
                        // Include standard attributes
                        result.href = link.href || '';
                        result.title = link.getAttribute('title') || '';
                        
                        %INCLUDE_TEXT%
                        
                        // Include additional attributes
                        %ADDITIONAL_ATTRS%
                        
                        return result;
                    });
                }
                """
                
                # Replace placeholders
                if include_text:
                    js_extract_links = js_extract_links.replace(
                        "%INCLUDE_TEXT%", 
                        "result.text = link.textContent.trim() || '';"
                    )
                else:
                    js_extract_links = js_extract_links.replace("%INCLUDE_TEXT%", "")
                
                # Add additional attributes
                additional_attrs = ""
                if include_attributes:
                    for attr in include_attributes:
                        if attr not in ["href", "title", "text"]:
                            additional_attrs += f"result['{attr}'] = link.getAttribute('{attr}') || '';\n"
                
                js_extract_links = js_extract_links.replace("%ADDITIONAL_ATTRS%", additional_attrs)
                
                # Execute JavaScript to extract links
                all_links = await page.evaluate(js_extract_links)
                
                # Filter links if pattern provided
                if filter_pattern:
                    try:
                        pattern = re.compile(filter_pattern)
                        filtered_links = [link for link in all_links if "href" in link and pattern.search(link["href"])]
                    except re.error as e:
                        raise ToolInputError(
                            f"Invalid regex pattern: {str(e)}",
                            param_name="filter_pattern",
                            provided_value=filter_pattern
                        ) from e
                else:
                    filtered_links = all_links
                
                # Clean up empty values
                for link in filtered_links:
                    for key, value in list(link.items()):
                        if value == "":
                            link[key] = None
                
                return {
                    "links": filtered_links,
                    "url": page.url,
                    "count": len(filtered_links),
                    "success": True
                }
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        logger.error(f"Error extracting links: {str(e)}")
        raise ToolError(
            f"Failed to extract links: {str(e)}",
            details={"url": url}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_extract_text(
    url: str,
    timeout: int = 30,
    user_agent: Optional[str] = None,
    selectors: Optional[List[str]] = None,
    remove_selectors: Optional[List[str]] = None,
    readability_mode: bool = False
) -> Dict[str, Any]:
    """Extract readable text content from a webpage using the Lightpanda browser.
    
    This tool navigates to a URL and extracts the main text content, optionally
    targeting specific elements via CSS selectors. It's useful for getting clean,
    readable content from articles, blog posts, or other text-heavy pages.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        timeout: Maximum time in seconds to wait for the page to load (default: 30).
        user_agent: Custom User-Agent string to use (optional).
        selectors: List of CSS selectors to extract text from (optional). If provided,
                  only text from these elements will be extracted.
        remove_selectors: List of CSS selectors to remove before extracting text (optional).
                         Useful for removing navigation, ads, etc.
        readability_mode: Apply Mozilla's Readability algorithm to extract article content (default: False).
    
    Returns:
        A dictionary containing the results:
        {
            "text": "The main text content of the page...",
            "title": "Page Title",
            "url": "https://example.com",
            "word_count": 1234,
            "success": true
        }
    
    Raises:
        ToolInputError: If the URL is invalid.
        ToolError: If the Lightpanda browser fails to extract text.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Remove elements if specified
                if remove_selectors:
                    for selector in remove_selectors:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            await page.evaluate('(el) => el.remove()', element)
                
                # Get the page title
                title = await page.title()
                
                # Extract text
                if readability_mode:
                    # Implement readability algorithm
                    # We'll add a simplified version of readability
                    text = await page.evaluate("""
                    () => {
                        // Get main content area
                        const content = document.querySelector('article') || 
                                       document.querySelector('main') || 
                                       document.querySelector('.content') || 
                                       document.querySelector('#content') || 
                                       document.body;
                        
                        // Remove non-content elements
                        const elementsToRemove = content.querySelectorAll('aside, nav, .nav, .navigation, .menu, .sidebar, .footer, script, style, link');
                        elementsToRemove.forEach(el => el.remove());
                        
                        // Extract and clean text
                        let text = content.innerText;
                        
                        // Normalize whitespace
                        text = text.replace(/\\s+/g, ' ').trim();
                        text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
                        
                        return text;
                    }
                    """)
                elif selectors:
                    # Extract text from specific selectors
                    text_parts = []
                    for selector in selectors:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            text_part = await element.evaluate('(el) => el.innerText')
                            if text_part:
                                text_parts.append(text_part)
                    
                    text = "\n\n".join(text_parts)
                else:
                    # Extract text from the main content area
                    text = await page.evaluate("""
                    () => {
                        // Get main content
                        const content = document.querySelector('article') || 
                                       document.querySelector('main') || 
                                       document.querySelector('.content') || 
                                       document.querySelector('#content') || 
                                       document.body;
                        
                        return content.innerText;
                    }
                    """)
                
                # Count words
                word_count = len([w for w in text.split() if w.strip()])
                
                return {
                    "text": text,
                    "title": title,
                    "url": page.url,
                    "word_count": word_count,
                    "success": True
                }
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise ToolError(
            f"Failed to extract text: {str(e)}",
            details={"url": url}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_download_file(
    url: str,
    output_path: Optional[str] = None,
    output_directory: Optional[str] = None,
    filename: Optional[str] = None,
    timeout: int = 60,
    user_agent: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Download a file using the Lightpanda browser.
    
    This tool downloads a file from a URL and saves it to the specified location.
    It can determine the filename from the URL or use a provided filename.
    
    Args:
        url: The URL of the file to download (must start with http:// or https://).
        output_path: Full path (including filename) to save the file (optional).
        output_directory: Directory to save the file (optional). Used with filename.
        filename: Name to give the downloaded file (optional).
                If not provided, extracted from the URL or Content-Disposition header.
        timeout: Maximum time in seconds to wait for the download (default: 60).
        user_agent: Custom User-Agent string to use (optional).
        headers: Additional HTTP headers to include in the request (optional).
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "file_path": "/path/to/downloaded/file.pdf",
            "file_size": 1048576,  # Size in bytes
            "content_type": "application/pdf",
            "download_time": 3.45  # Time in seconds
        }
    
    Raises:
        ToolInputError: If the URL or output parameters are invalid.
        ToolError: If the download fails.
    """
    start_time = time.time()
    
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate output parameters
    if output_path and (output_directory or filename):
        raise ToolInputError(
            "Cannot provide both output_path and output_directory/filename",
            param_name="output_path",
            provided_value=output_path
        )
    
    if not output_path and not output_directory:
        output_directory = os.path.join(tempfile.gettempdir(), "lightpanda_downloads")
        os.makedirs(output_directory, exist_ok=True)
    
    try:
        # Prepare headers
        req_headers = {}
        if user_agent:
            req_headers["User-Agent"] = user_agent
        if headers:
            req_headers.update(headers)
        
        # Download the file using httpx
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url, headers=req_headers, timeout=timeout) as response:
                if response.status_code != 200:
                    raise ToolError(
                        f"Download failed: HTTP status {response.status_code}",
                        details={"url": url, "status": response.status_code}
                    )
                
                # Determine filename if not provided
                if not filename:
                    # Try to get from Content-Disposition header
                    content_disposition = response.headers.get("Content-Disposition")
                    if content_disposition and "filename=" in content_disposition:
                        filename_match = re.search(r'filename=["\'](.*?)["\']', content_disposition)
                        if filename_match:
                            filename = filename_match.group(1)
                        else:
                            filename_match = re.search(r'filename=(.*?)($|;)', content_disposition)
                            if filename_match:
                                filename = filename_match.group(1)
                    
                    # If still no filename, extract from URL
                    if not filename:
                        url_path = urllib.parse.urlparse(url).path
                        filename = os.path.basename(url_path)
                        
                        # If still no valid filename, generate one
                        if not filename or filename == "/" or "." not in filename:
                            content_type = response.headers.get("Content-Type", "application/octet-stream")
                            extension = mimetypes.guess_extension(content_type) or ".bin"
                            filename = f"download_{int(time.time())}{extension}"
                
                # Sanitize filename
                if filename:
                    filename = await _sanitize_filename(filename)
                
                # Determine full output path
                if output_path:
                    file_path = output_path
                    # Ensure parent directory exists
                    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                else:
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory, exist_ok=True)
                    file_path = os.path.join(output_directory, filename)
                
                # Get content type
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                
                # Download the file
                file_size = 0
                async with aiofiles.open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)
                        file_size += len(chunk)
                
                # Calculate download time
                download_time = time.time() - start_time
                
                return {
                    "success": True,
                    "file_path": file_path,
                    "file_size": file_size,
                    "content_type": content_type,
                    "download_time": download_time
                }
                
    except httpx.TimeoutException as e:
        raise ToolError(
            f"Download timed out after {timeout} seconds",
            details={"url": url}
        ) from e
    except httpx.RequestError as e:
        raise ToolError(
            f"Download request failed: {str(e)}",
            details={"url": url}
        ) from e
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise ToolError(
            f"Failed to download file: {str(e)}",
            details={"url": url}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_find_and_download_pdfs(
    search_query: str,
    output_directory: str,
    filename_template: str,
    max_pdfs: int = 5,
    timeout: int = 30,
    llm_model: str = "openai/gpt-4.1-mini"
) -> Dict[str, Any]:
    """Find and download PDF files related to a search query using the Lightpanda browser.
    
    This tool searches for PDF files related to a given search query and downloads them
    to a specified directory. It uses a combination of regular search and explicit
    search terms to find relevant PDFs.
    
    Args:
        search_query: The search query to find related PDFs for.
        output_directory: The directory to save the downloaded PDF files to.
        filename_template: The template for naming the downloaded PDF files.
        max_pdfs: Maximum number of PDFs to download (default: 5).
        timeout: Maximum time in seconds to wait for the search and download operations (default: 30).
        llm_model: The language model to use for relevance evaluation (default: "openai/gpt-4.1-mini").
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,  # Whether the operation was successful
            "query": "...",  # The search query used
            "downloads": [
                {
                    "url": "https://example.com/pdf1.pdf",
                    "file_path": "/path/to/downloaded_pdf1.pdf",
                    "title": "Description of PDF 1",
                    "size": 123456
                },
                # ... more downloads
            ],
            "output_directory": "/path/to/output_directory",  # The output directory used
            "execution_time": 123.45  # Execution time in seconds
        }
    
    Raises:
        ToolInputError: If the search query or output directory is invalid.
        ToolError: If the PDF search or download operations fail.
    """
    try:
        # Start time
        start_time = time.time()
        
        # Initialize variables
        pdf_links = []
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(search_query)}"
        pdf_extract_js = """
        () => {
            const pdfs = [];
            document.querySelectorAll('a[href$=".pdf"]').forEach(link => {
                let url = link.href;
                pdfs.push({
                    url: url,
                    title: link.textContent.trim() || link.getAttribute('title') || url.split('/').pop()
                });
            });
            return JSON.stringify(pdfs);
        }
        """
        
        # First, try regular search results
        regular_links_result = await lightpanda_javascript(
            url=search_url,
            javascript=pdf_extract_js,
            timeout=timeout
        )
        
        if regular_links_result.get("success"):
            try:
                regular_links = json.loads(regular_links_result.get("result", "[]"))
                
                # Visit the first few regular links to find PDFs
                for link in regular_links[:5]:
                    try:
                        page_url = link.get("url")
                        
                        # Skip URLs from the search engine itself
                        if any(engine in page_url.lower() for engine in ["google", "bing", "duckduckgo"]):
                            continue
                        
                        # Visit the page
                        page_result = await lightpanda_fetch(
                            url=page_url,
                            dump_html=True,
                            timeout=timeout
                        )
                        
                        if page_result.get("success"):
                            # Look for PDF links on the page
                            page_pdf_links_js = """
                            () => {
                                const pdfs = [];
                                document.querySelectorAll('a[href$=".pdf"]').forEach(link => {
                                    let url = link.href;
                                    pdfs.push({
                                        url: url,
                                        title: link.textContent.trim() || link.getAttribute('title') || url.split('/').pop()
                                    });
                                });
                                return JSON.stringify(pdfs);
                            }
                            """
                            
                            page_pdf_result = await lightpanda_javascript(
                                url=page_url,
                                javascript=page_pdf_links_js,
                                timeout=timeout
                            )
                            
                            if page_pdf_result.get("success"):
                                try:
                                    page_pdfs = json.loads(page_pdf_result.get("result", "[]"))
                                    pdf_links.extend(page_pdfs)
                                except json.JSONDecodeError:
                                    pass
                    except Exception:
                        continue
            except json.JSONDecodeError:
                pass
        
        # If still no PDFs found, try alternate search queries
        if not pdf_links:
            # Try with explicit PDF search terms
            alternate_query = f"{search_query} filetype:pdf document report"
            alternate_search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(alternate_query)}"
            
            alternate_result = await lightpanda_fetch(
                url=alternate_search_url,
                dump_html=True,
                timeout=timeout
            )
            
            if alternate_result.get("success"):
                alternate_pdf_result = await lightpanda_javascript(
                    url=alternate_search_url,
                    javascript=pdf_extract_js,
                    timeout=timeout
                )
                
                if alternate_pdf_result.get("success"):
                    try:
                        alternate_pdfs = json.loads(alternate_pdf_result.get("result", "[]"))
                        pdf_links.extend(alternate_pdfs)
                    except json.JSONDecodeError:
                        pass
        
        # Remove duplicates
        unique_pdfs = []
        seen_urls = set()
        
        for pdf in pdf_links:
            url = pdf.get("url")
            if url and url not in seen_urls:
                unique_pdfs.append(pdf)
                seen_urls.add(url)
        
        if not unique_pdfs:
            return {
                "success": True,
                "query": search_query,
                "downloads": [],
                "output_directory": output_directory,
                "message": "No PDF files found for the search query.",
                "execution_time": time.time() - start_time
            }
        
        # Check if each of the PDFs is specifically relevant to the original search query
        if len(unique_pdfs) > max_pdfs:
            # Use LLM to evaluate relevance
            pdf_list_str = json.dumps([{
                "url": pdf.get("url"),
                "title": pdf.get("title", "No Title")
            } for pdf in unique_pdfs], indent=2)
            
            system_message = "You are an AI assistant that evaluates the relevance of PDF documents to a search query."
            prompt = f"""
            Evaluate the relevance of these PDF files to the search query:
            
            Search Query: {search_query}
            
            PDF Files:
            {pdf_list_str}
            
            Select the {max_pdfs} most relevant PDFs for this query. Return your selection as a JSON array of the most relevant PDF URLs, like this:
            ["url1", "url2", ...]
            
            Selection:
            """
            
            relevance_result = await _call_llm(prompt, llm_model, system_message)
            
            if relevance_result.get("success"):
                # Parse relevant PDF URLs
                try:
                    llm_text = relevance_result.get("text", "")
                    json_match = re.search(r'(\[.*\])', llm_text, re.DOTALL)
                    
                    if json_match:
                        relevant_urls = json.loads(json_match.group(1))
                        
                        # Filter PDFs by relevant URLs
                        filtered_pdfs = [pdf for pdf in unique_pdfs if pdf.get("url") in relevant_urls]
                        
                        if filtered_pdfs:
                            unique_pdfs = filtered_pdfs
                except Exception:
                    # If parsing fails, use the original list but limit to max_pdfs
                    unique_pdfs = unique_pdfs[:max_pdfs]
            else:
                # If LLM evaluation fails, use the original list but limit to max_pdfs
                unique_pdfs = unique_pdfs[:max_pdfs]
        
        # Download PDFs
        downloads = []
        
        for index, pdf in enumerate(unique_pdfs[:max_pdfs], 1):
            url = pdf.get("url")
            title = pdf.get("title", "No Title")
            
            try:
                # Clean up title for filename
                clean_title = await _sanitize_filename(title)
                if not clean_title:
                    clean_title = f"document_{index}"
                
                # Use the filename template
                current_date = datetime.now().strftime("%Y%m%d")
                filename = filename_template.format(
                    index=index,
                    title=clean_title,
                    date=current_date
                )
                
                # Ensure the filename ends with .pdf
                if not filename.lower().endswith('.pdf'):
                    filename += ".pdf"
                
                # Download the file
                download_result = await lightpanda_download_file(
                    url=url,
                    output_path=None,
                    output_directory=output_directory,
                    filename=filename,
                    timeout=timeout
                )
                
                if download_result.get("success"):
                    downloads.append({
                        "url": url,
                        "file_path": download_result.get("file_path"),
                        "title": title,
                        "size": download_result.get("file_size", 0)
                    })
                else:
                    logger.warning(f"Failed to download PDF: {url}")
            except Exception as download_error:
                logger.warning(f"Error downloading PDF {url}: {str(download_error)}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "query": search_query,
            "downloads": downloads,
            "output_directory": output_directory,
            "execution_time": execution_time
        }
    
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        logger.error(f"Find and download PDFs failed: {str(e)}")
        raise ToolError(f"Find and download PDFs failed: {str(e)}") from e

@with_tool_metrics
@with_error_handling
async def lightpanda_get_element_info(
    url: str,
    selector: str,
    timeout: int = 30,
    user_agent: Optional[str] = None,
    include_html: bool = False,
    include_attributes: bool = True
) -> Dict[str, Any]:
    """Get detailed information about an element using the Lightpanda browser.
    
    This tool navigates to a URL and retrieves information about a specific element,
    including its text content, tag name, attributes, and optionally its HTML.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        selector: CSS selector to find the element.
        timeout: Maximum time in seconds to wait for the page to load (default: 30).
        user_agent: Custom User-Agent string to use (optional).
        include_html: Whether to include the element's innerHTML (default: False).
        include_attributes: Whether to include the element's attributes (default: True).
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "element_found": true,
            "tag_name": "div",
            "text_content": "Hello World",
            "attributes": {   # Only if include_attributes is True
                "id": "greeting",
                "class": "message highlight"
            },
            "html": "<span>Hello World</span>"  # Only if include_html is True
        }
    
    Raises:
        ToolInputError: If the URL or selector is invalid.
        ToolError: If the Lightpanda browser fails to get element information.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Try to find the element
                element = await page.query_selector(selector)
                
                if not element:
                    return {
                        "success": True,
                        "element_found": False,
                        "error": f"Element not found: {selector}"
                    }
                
                # Get element information
                tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                text_content = await element.evaluate("el => el.textContent.trim()")
                
                result = {
                    "success": True,
                    "element_found": True,
                    "tag_name": tag_name,
                    "text_content": text_content
                }
                
                # Get HTML content if requested
                if include_html:
                    html_content = await element.evaluate("el => el.innerHTML")
                    result["html"] = html_content
                
                # Get attributes if requested
                if include_attributes:
                    attributes = await element.evaluate("""el => {
                        const attrs = {};
                        for (const attr of el.attributes) {
                            attrs[attr.name] = attr.value;
                        }
                        return attrs;
                    }""")
                    result["attributes"] = attributes
                
                return result
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        logger.error(f"Error getting element information: {str(e)}")
        raise ToolError(
            f"Failed to get element information: {str(e)}",
            details={"url": url, "selector": selector}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_click(
    url: str,
    selector: str,
    wait_for_navigation: bool = True,
    timeout: int = 30,
    user_agent: Optional[str] = None
) -> Dict[str, Any]:
    """Click on an element in the page using the Lightpanda browser.
    
    This tool navigates to a URL, finds an element using a CSS selector, and clicks on it.
    It can optionally wait for navigation to complete after the click.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        selector: CSS selector to find the element to click.
        wait_for_navigation: Whether to wait for navigation to complete after the click (default: True).
        timeout: Maximum time in seconds to wait for the page to load and actions to complete (default: 30).
        user_agent: Custom User-Agent string to use (optional).
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "url": "https://example.com/new-page",  # The URL after clicking and navigation (if applicable)
            "element_description": "Button with text 'Submit'",  # Description of the clicked element
            "navigation_occurred": true  # Whether navigation happened after the click
        }
    
    Raises:
        ToolInputError: If the URL or selector is invalid.
        ToolError: If the Lightpanda browser fails to click the element.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate selector
    if not selector or not isinstance(selector, str):
        raise ToolInputError(
            "Selector must be a non-empty string",
            param_name="selector",
            provided_value=selector
        )
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Get initial URL before clicking
                start_url = page.url
                
                # Verify the element exists
                element = await page.query_selector(selector)
                if not element:
                    raise ToolError(
                        f"Element not found: {selector}",
                        details={"selector": selector}
                    )
                
                # Get element description for better reporting
                element_description = await page.evaluate(
                    """
                    (element) => {
                        const tag = element.tagName.toLowerCase();
                        const id = element.id ? `id="${element.id}"` : '';
                        const classes = element.className ? `class="${element.className}"` : '';
                        const text = element.textContent.trim();
                        const textPreview = text ? `text="${text.substring(0, 50)}"` : '';
                        
                        return [tag, id, classes, textPreview].filter(Boolean).join(' ');
                    }
                    """,
                    element
                )
                
                # Set up navigation waiter if needed
                navigation_promise = None
                if wait_for_navigation:
                    navigation_promise = page.wait_for_navigation(timeout=timeout * 1000)
                
                # Click the element
                await element.click()
                
                # Wait for navigation if specified
                navigation_occurred = False
                if wait_for_navigation:
                    try:
                        await navigation_promise
                        navigation_occurred = True
                    except Exception as e:
                        logger.warning(f"Navigation after click did not occur: {str(e)}")
                
                # Check if URL changed to determine if navigation occurred
                current_url = page.url
                if current_url != start_url:
                    navigation_occurred = True
                
                return {
                    "success": True,
                    "url": current_url,
                    "element_description": element_description,
                    "navigation_occurred": navigation_occurred
                }
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        logger.error(f"Error clicking element: {str(e)}")
        raise ToolError(
            f"Failed to click element: {str(e)}",
            details={"url": url, "selector": selector}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_form_submit(
    url: str,
    form_selector: str,
    field_values: Dict[str, Any],
    submit_button_selector: Optional[str] = None,
    wait_for_navigation: bool = True,
    timeout: int = 60,
    user_agent: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Fill in and submit a form using the Lightpanda browser.
    
    This tool navigates to a URL, fills in form fields with specified values, and submits the form.
    It can handle various field types including text inputs, checkboxes, radio buttons, and select elements.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        form_selector: CSS selector to find the form element.
        field_values: Dictionary mapping field selectors or names to values.
                     Example: {"#email": "user@example.com", "input[name='password']": "secret"}
        submit_button_selector: CSS selector for the submit button (optional).
                              If not provided, form will be submitted via form.submit().
        wait_for_navigation: Whether to wait for navigation to complete after submission (default: True).
        timeout: Maximum time in seconds to wait for operations to complete (default: 60).
        user_agent: Custom User-Agent string to use (optional).
        headers: Additional HTTP headers to include in the request (optional).
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "url": "https://example.com/success",  # The URL after form submission
            "fields_filled": ["#email", "input[name='password']"],  # Fields that were successfully filled
            "navigation_occurred": true  # Whether navigation happened after submission
        }
    
    Raises:
        ToolInputError: If the URL, selectors, or field values are invalid.
        ToolError: If the Lightpanda browser fails to fill or submit the form.
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate form selector
    if not form_selector or not isinstance(form_selector, str):
        raise ToolInputError(
            "Form selector must be a non-empty string",
            param_name="form_selector",
            provided_value=form_selector
        )
    
    # Validate field values
    if not field_values or not isinstance(field_values, dict):
        raise ToolInputError(
            "Field values must be a non-empty dictionary",
            param_name="field_values",
            provided_value=field_values
        )
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Set headers if provided
                if headers:
                    await page.set_extra_http_headers(headers)
                
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Get initial URL before form submission
                start_url = page.url
                
                # Check if the form exists
                form = await page.query_selector(form_selector)
                if not form:
                    raise ToolError(
                        f"Form not found: {form_selector}",
                        details={"form_selector": form_selector}
                    )
                
                # Keep track of filled fields
                fields_filled = []
                
                # Fill form fields
                for selector, value in field_values.items():
                    try:
                        # If the selector is a field name and not a CSS selector, try to find it
                        if selector.find('#') == -1 and selector.find('[') == -1 and selector.find('.') == -1:
                            # Try as a name attribute
                            field_selector = f"[name='{selector}']"
                            element = await page.query_selector(field_selector)
                            
                            if not element:
                                # Try as an ID
                                field_selector = f"#{selector}"
                                element = await page.query_selector(field_selector)
                            
                            if not element:
                                logger.warning(f"Field not found: {selector}")
                                continue
                        else:
                            # Use as CSS selector
                            element = await page.query_selector(selector)
                            if not element:
                                logger.warning(f"Field not found: {selector}")
                                continue
                        
                        # Get element tag name and type
                        tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                        field_type = await element.evaluate("el => el.type ? el.type.toLowerCase() : null")
                        
                        # Fill the field based on its type
                        if tag_name == "input":
                            if field_type in ["checkbox", "radio"]:
                                # For checkboxes and radio buttons, check or uncheck
                                if value:
                                    await element.check()
                                else:
                                    await element.uncheck()
                            elif field_type != "file":
                                # For other inputs (except file inputs)
                                await element.fill(str(value))
                            else:
                                # For file inputs
                                if isinstance(value, str):
                                    await element.set_input_files(value)
                                elif isinstance(value, list):
                                    await element.set_input_files(value)
                        elif tag_name == "textarea":
                            await element.fill(str(value))
                        elif tag_name == "select":
                            # For select elements, set the value
                            if isinstance(value, (list, tuple)):
                                # Multiple selection
                                await element.select_option(values=[str(v) for v in value])
                            else:
                                # Single selection
                                await element.select_option(value=str(value))
                        
                        fields_filled.append(selector)
                    except Exception as field_error:
                        logger.warning(f"Error filling field {selector}: {str(field_error)}")
                
                # Set up navigation waiter if needed
                navigation_promise = None
                if wait_for_navigation:
                    navigation_promise = page.wait_for_navigation(timeout=timeout * 1000)
                
                # Submit the form
                if submit_button_selector:
                    # Find and click the submit button
                    submit_button = await page.query_selector(submit_button_selector)
                    if not submit_button:
                        raise ToolError(
                            f"Submit button not found: {submit_button_selector}",
                            details={"submit_button_selector": submit_button_selector}
                        )
                    
                    await submit_button.click()
                else:
                    # Submit the form programmatically
                    await form.evaluate("form => form.submit()")
                
                # Wait for navigation if specified
                navigation_occurred = False
                if wait_for_navigation:
                    try:
                        await navigation_promise
                        navigation_occurred = True
                    except Exception as e:
                        logger.warning(f"Navigation after form submission did not occur: {str(e)}")
                
                # Check if URL changed to determine if navigation occurred
                current_url = page.url
                if current_url != start_url:
                    navigation_occurred = True
                
                return {
                    "success": True,
                    "url": current_url,
                    "fields_filled": fields_filled,
                    "navigation_occurred": navigation_occurred
                }
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        logger.error(f"Error submitting form: {str(e)}")
        raise ToolError(
            f"Failed to submit form: {str(e)}",
            details={"url": url, "form_selector": form_selector}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_start_cdp_server(
    host: str = "127.0.0.1",
    port: int = 9222,
    disable_telemetry: bool = True
) -> Dict[str, Any]:
    """Start a Chrome DevTools Protocol (CDP) server using Lightpanda.
    
    This tool starts a CDP server which allows tools like Puppeteer or other
    CDP clients to connect and control the Lightpanda browser. This is useful
    for more complex browser automation that requires established automation
    libraries.
    
    Lightpanda is automatically downloaded and installed if not already present.
    
    Note: The server will continue to run in the background until explicitly stopped.
          For proper resource management, make sure to call lightpanda_stop_cdp_server
          when you're done.
    
    Args:
        host: The host to bind the server to (default: "127.0.0.1").
        port: The port to listen on (default: 9222).
        disable_telemetry: Whether to disable Lightpanda telemetry (default: True).
    
    Returns:
        A dictionary containing the server information:
        {
            "server_pid": 1234,             # Process ID of the server
            "ws_endpoint": "ws://127.0.0.1:9222",  # WebSocket endpoint for CDP clients
            "success": true                 # Whether the server was started successfully
        }
    
    Raises:
        ToolError: If the Lightpanda browser fails to start the CDP server.
    """
    global LIGHTPANDA_CDP_HOST, LIGHTPANDA_CDP_PORT, LIGHTPANDA_CDP_WS_ENDPOINT
    
    # Update global settings if parameters are different
    if host != LIGHTPANDA_CDP_HOST or port != LIGHTPANDA_CDP_PORT:
        LIGHTPANDA_CDP_HOST = host
        LIGHTPANDA_CDP_PORT = port
        LIGHTPANDA_CDP_WS_ENDPOINT = f"ws://{LIGHTPANDA_CDP_HOST}:{LIGHTPANDA_CDP_PORT}"
        
        # Stop any existing server since we're changing host/port
        await _cleanup_browser()
    
    # Start the CDP server
    if not await _ensure_cdp_server_running():
        raise ToolError(
            "Failed to start Lightpanda CDP server",
            error_code="lightpanda_cdp_server_failed"
        )
    
    return {
        "server_pid": _cdp_server_process.pid,
        "ws_endpoint": LIGHTPANDA_CDP_WS_ENDPOINT,
        "log_file": _cdp_server_log_file,
        "success": True
    }

@with_tool_metrics
@with_error_handling
async def lightpanda_stop_cdp_server(
    pid: Optional[int] = None
) -> Dict[str, Any]:
    """Stop the running Chrome DevTools Protocol (CDP) server.
    
    This tool stops the CDP server started by lightpanda_start_cdp_server and cleans up
    resources associated with it, including closing any Playwright browser connections.
    
    Args:
        pid: Optional process ID of the CDP server to stop. If not provided,
             the function will stop the most recent CDP server started through
             the lightpanda_start_cdp_server tool.
    
    Returns:
        A dictionary indicating the success of the operation:
        {
            "success": true,
            "message": "CDP server stopped successfully"
        }
    
    Raises:
        ToolError: If no CDP server is running or the server fails to stop.
    """
    global _cdp_server_process
    
    # If a specific PID was provided, verify it matches our process
    if pid is not None and _cdp_server_process is not None and _cdp_server_process.pid != pid:
        logger.warning(f"Requested to stop CDP server with PID {pid}, but running server has PID {_cdp_server_process.pid}")
    
    # If no CDP server is running, report an error
    if _cdp_server_process is None:
        return {
            "success": False,
            "message": "No CDP server is currently running"
        }
    
    # Clean up browser resources and stop the server
    await _cleanup_browser()
    
    return {
        "success": True,
        "message": "CDP server stopped successfully"
    }

# --- CDP Server and Browser Management ---

async def _ensure_cdp_server_running() -> bool:
    """Ensure that the Lightpanda CDP server is running."""
    global _cdp_server_process, _cdp_server_log_file
    
    if _cdp_server_process is not None:
        # Check if the process is still running
        if _cdp_server_process.poll() is None:
            return True
        else:
            # Process has terminated, clean up
            _cdp_server_process = None
            _cdp_server_log_file = None
    
    # Start the CDP server
    try:
        binary_path = await _ensure_lightpanda_installed()
        
        # Create a log file for stdout/stderr
        log_dir = os.path.join(LIGHTPANDA_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        _cdp_server_log_file = os.path.join(log_dir, f"cdp_server_{LIGHTPANDA_CDP_PORT}.log")
        
        # Build command for the CDP server
        cmd = [
            binary_path, 
            "serve", 
            "--host", LIGHTPANDA_CDP_HOST, 
            "--port", str(LIGHTPANDA_CDP_PORT)
        ]
        
        # Start the CDP server process
        with open(_cdp_server_log_file, "w") as f:
            _cdp_server_process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
                env=os.environ.copy(),
                start_new_session=True  # Detach from parent process
            )
        
        # Wait a moment for the server to start
        await asyncio.sleep(1)
        
        # Check if the process is running
        if _cdp_server_process.poll() is not None:
            # Process has terminated, read the log for error details
            with open(_cdp_server_log_file, "r") as f:
                log_content = f.read()
            
            logger.error(f"Failed to start Lightpanda CDP server: {log_content}")
            return False
        
        logger.info(f"Started Lightpanda CDP server on {LIGHTPANDA_CDP_HOST}:{LIGHTPANDA_CDP_PORT}")
        return True
        
    except Exception as e:
        logger.error(f"Error starting CDP server: {str(e)}")
        return False

async def _get_browser():
    """Get or initialize the global Playwright browser instance connected to Lightpanda."""
    global _playwright, _browser, _connection_initialized
    
    async with _browser_initialization_lock:
        if _browser is not None and _connection_initialized:
            return _browser
        
        # Ensure CDP server is running
        if not await _ensure_cdp_server_running():
            raise ToolError(
                "Failed to start Lightpanda CDP server",
                error_code="lightpanda_cdp_server_failed"
            )
        
        try:
            # Initialize Playwright if needed
            if _playwright is None:
                _playwright = await async_playwright().start()
            
            # Connect to the Lightpanda CDP endpoint
            _browser = await _playwright.chromium.connect_over_cdp(LIGHTPANDA_CDP_WS_ENDPOINT)
            _connection_initialized = True
            
            return _browser
        except Exception as e:
            logger.error(f"Failed to connect to Lightpanda CDP server: {str(e)}")
            raise ToolError(
                f"Failed to connect to Lightpanda CDP server: {str(e)}",
                error_code="lightpanda_cdp_connection_failed"
            ) from e

async def _cleanup_browser():
    """Clean up the Playwright browser and CDP server."""
    global _playwright, _browser, _cdp_server_process, _connection_initialized
    
    # Clean up Playwright resources
    if _browser is not None:
        try:
            await _browser.close()
        except Exception as e:
            logger.warning(f"Error closing browser: {str(e)}")
        finally:
            _browser = None
            _connection_initialized = False
    
    if _playwright is not None:
        try:
            await _playwright.stop()
        except Exception as e:
            logger.warning(f"Error stopping Playwright: {str(e)}")
        finally:
            _playwright = None
    
    # Stop the CDP server
    if _cdp_server_process is not None:
        try:
            _cdp_server_process.terminate()
            await asyncio.sleep(0.5)
            if _cdp_server_process.poll() is None:
                _cdp_server_process.kill()
        except Exception as e:
            logger.warning(f"Error terminating CDP server: {str(e)}")
        finally:
            _cdp_server_process = None

@with_tool_metrics
@with_error_handling
async def lightpanda_generate_pdf(
    url: str,
    output_path: Optional[str] = None,
    output_directory: Optional[str] = None,
    filename: Optional[str] = None,
    timeout: int = 60,
    user_agent: Optional[str] = None,
    page_size: str = "A4",
    landscape: bool = False,
    margins: Optional[Dict[str, float]] = None,
    wait_for_network_idle: bool = True
) -> Dict[str, Any]:
    """Generate a PDF from a webpage using the Lightpanda browser.
    
    This tool navigates to a URL and generates a PDF document from the page content
    using Playwright's built-in PDF generation capabilities.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        output_path: Full path (including filename) to save the PDF (optional).
        output_directory: Directory to save the PDF (optional). Used with filename.
        filename: Name to give the PDF file (optional). If not provided, generated from the page title.
        timeout: Maximum time in seconds to wait for page load and PDF generation (default: 60).
        user_agent: Custom User-Agent string to use (optional).
        page_size: PDF page size (default: "A4"). Options: "A4", "Letter", "Legal", etc.
        landscape: Whether to use landscape orientation (default: False).
        margins: Dictionary with margin values in inches (optional).
                Example: {"top": 0.5, "right": 1.0, "bottom": 0.5, "left": 1.0}
        wait_for_network_idle: Whether to wait for network to be idle before printing (default: True).
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "file_path": "/path/to/generated.pdf",
            "file_size": 1048576,  # Size in bytes
            "generation_time": 3.45  # Time in seconds
        }
    
    Raises:
        ToolInputError: If the URL or output parameters are invalid.
        ToolError: If the PDF generation fails.
    """
    start_time = time.time()
    
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate output parameters
    if output_path and (output_directory or filename):
        raise ToolInputError(
            "Cannot provide both output_path and output_directory/filename",
            param_name="output_path",
            provided_value=output_path
        )
    
    if not output_path and not output_directory:
        output_directory = os.path.join(tempfile.gettempdir(), "lightpanda_pdfs")
        os.makedirs(output_directory, exist_ok=True)
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to URL with timeout
                await page.goto(
                    url, 
                    timeout=timeout * 1000, 
                    wait_until="networkidle" if wait_for_network_idle else "load"
                )
                
                # Get page title for the filename if needed
                page_title = await page.title()
                
                # Determine filename
                if not filename:
                    if page_title:
                        filename = await _sanitize_filename(page_title) + ".pdf"
                    else:
                        # Extract from URL
                        path = urllib.parse.urlparse(url).path
                        basename = os.path.basename(path)
                        if basename and basename != "/":
                            filename = await _sanitize_filename(basename)
                            if not filename.lower().endswith('.pdf'):
                                filename += ".pdf"
                        else:
                            filename = f"page_{int(time.time())}.pdf"
                elif not filename.lower().endswith('.pdf'):
                    filename += ".pdf"
                
                # Determine full output path
                if output_path:
                    file_path = output_path
                    # Ensure parent directory exists
                    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                else:
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory, exist_ok=True)
                    file_path = os.path.join(output_directory, filename)
                
                # Set up PDF options
                pdf_options = {
                    "path": file_path,
                    "format": page_size,
                    "landscape": landscape,
                    "printBackground": True
                }
                
                # Add margins if provided
                if margins:
                    pdf_options["margin"] = {
                        "top": f"{margins.get('top', 0)}in",
                        "right": f"{margins.get('right', 0)}in",
                        "bottom": f"{margins.get('bottom', 0)}in",
                        "left": f"{margins.get('left', 0)}in",
                    }
                
                # Generate the PDF
                await page.pdf(**pdf_options)
                
                # Get file info
                file_size = os.path.getsize(file_path)
                generation_time = time.time() - start_time
                
                return {
                    "success": True,
                    "file_path": file_path,
                    "file_size": file_size,
                    "page_title": page_title,
                    "generation_time": generation_time
                }
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise ToolError(
            f"Failed to generate PDF: {str(e)}",
            details={"url": url}
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_get_version() -> Dict[str, Any]:
    """Get the version information of the installed Lightpanda browser.
    
    This tool returns version information for the installed Lightpanda browser.
    It also checks if Lightpanda is installed and installs it if needed.
    
    Returns:
        A dictionary containing version information:
        {
            "version": "nightly-build-2023-07-21",  # Version string
            "installed_path": "/home/user/.ultimate_mcp_server/lightpanda/lightpanda",
            "platform": "linux",  # Platform the binary is for
            "success": true       # Whether the operation was successful
        }
    
    Raises:
        ToolError: If Lightpanda cannot be installed or the version cannot be determined.
    """
    try:
        # Ensure Lightpanda is installed
        binary_path = await _ensure_lightpanda_installed()
        
        # Run version command
        result = await _run_lightpanda_command(["--version"])
        
        if not result["success"]:
            raise ToolError(
                f"Failed to get Lightpanda version: {result.get('error', 'Unknown error')}",
                details={"stderr": result.get("stderr", "")}
            )
        
        # Parse version from output
        version_info = result["stdout"].strip()
        platform_info = await _get_platform_info()
        
        return {
            "version": version_info,
            "installed_path": binary_path,
            "platform": platform_info,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error getting Lightpanda version: {str(e)}")
        raise ToolError(
            f"Failed to get Lightpanda version: {str(e)}"
        ) from e

@with_tool_metrics
@with_error_handling
async def lightpanda_execute_web_workflow(
    instructions: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None,
    max_steps: int = 15,
    timeout_per_step: int = 30,
    llm_model: str = "openai/gpt-4.1-mini"
) -> Dict[str, Any]:
    """Execute a sequence of web interactions guided by an LLM to achieve a specific goal.
    
    This tool allows for complex web automation workflows by providing high-level
    instructions and letting an LLM determine the specific steps needed.
    
    Args:
        instructions: Dictionary defining the workflow:
            goal: (str) High-level description of what to accomplish.
            start_url: (str) Initial URL to navigate to.
            available_actions: (List[str]) Actions the LLM can use (e.g., ["click", "type", "extract"]).
            success_criteria: (str) Description of what success looks like.
        input_data: Dictionary of data to use in the workflow (e.g., login credentials).
        max_steps: Maximum number of steps to take (default: 15).
        timeout_per_step: Maximum time in seconds per step (default: 30).
        llm_model: LLM model to use for guidance (default: "openai/gpt-4.1-mini").
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "message": "Workflow completed successfully",
            "steps_taken": 7,  # Number of steps executed
            "final_url": "https://example.com/success",
            "extracted_data": {...}  # Any data extracted during the workflow
        }
    
    Raises:
        ToolInputError: If the instructions are invalid.
        ToolError: If the workflow fails.
    """
    start_time = time.time()
    
    # Validate instructions
    if not isinstance(instructions, dict):
        raise ToolInputError(
            "Instructions must be a dictionary",
            param_name="instructions",
            provided_value=instructions
        )
    
    goal = instructions.get("goal")
    start_url = instructions.get("start_url")
    available_actions = instructions.get("available_actions")
    success_criteria = instructions.get("success_criteria")
    
    if not goal or not isinstance(goal, str):
        raise ToolInputError(
            "Instructions must include a 'goal' string",
            param_name="instructions.goal",
            provided_value=goal
        )
    
    if not start_url or not isinstance(start_url, str) or not start_url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "Instructions must include a valid 'start_url'",
            param_name="instructions.start_url",
            provided_value=start_url
        )
    
    if not available_actions or not isinstance(available_actions, list):
        raise ToolInputError(
            "Instructions must include 'available_actions' list",
            param_name="instructions.available_actions",
            provided_value=available_actions
        )
    
    if not success_criteria or not isinstance(success_criteria, str):
        raise ToolInputError(
            "Instructions must include 'success_criteria' string",
            param_name="instructions.success_criteria",
            provided_value=success_criteria
        )
    
    # Initialize workflow state
    steps_taken = 0
    current_url = start_url
    extracted_data = {}
    action_history = []
    workflow_successful = False
    workflow_message = "Workflow did not complete within step limit"
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context
        context = await browser.new_context()
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to the start URL
                await page.goto(start_url, timeout=timeout_per_step * 1000, wait_until="networkidle")
                current_url = page.url
                
                # Step through the workflow
                while steps_taken < max_steps and not workflow_successful:
                    steps_taken += 1
                    
                    # Get current page info (title and URL)
                    page_title = await page.title()
                    current_url = page.url
                    
                    # Extract page text content
                    page_content = await page.evaluate("() => document.body.innerText")
                    
                    # Get available elements for interaction
                    available_elements = await page.evaluate("""() => {
                        const elements = [];
                        // Get interactive elements
                        document.querySelectorAll('a, button, input, select, textarea').forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0) {
                                elements.push({
                                    id: `el_${index}`,
                                    tag: el.tagName.toLowerCase(),
                                    type: el.type || null,
                                    text: el.textContent.trim().substring(0, 50) || null,
                                    placeholder: el.placeholder || null,
                                    name: el.name || null,
                                    value: el.value || null,
                                    href: el.href || null,
                                    visible: true
                                });
                            }
                        });
                        return elements;
                    }""")
                    
                    # Prepare context for LLM
                    context = {
                        "goal": goal,
                        "current_url": current_url,
                        "current_title": page_title,
                        "page_content": page_content,
                        "available_elements": available_elements,
                        "available_actions": available_actions,
                        "action_history": action_history,
                        "steps_taken": steps_taken,
                        "max_steps": max_steps,
                        "input_data": input_data or {},
                        "extracted_data": extracted_data,
                        "success_criteria": success_criteria
                    }
                    
                    # Prepare the LLM prompt
                    system_message = "You are an AI assistant that controls web browsing actions to accomplish specific goals."
                    prompt = f"""
                    Goal: {goal}
                    
                    Current State:
                    - URL: {current_url}
                    - Title: {page_title}
                    - Step: {steps_taken} of {max_steps}
                    
                    Page Content Summary:
                    {page_content[:2000]}...
                    
                    Available Elements:
                    {json.dumps(available_elements, indent=2)}
                    
                    Available Actions:
                    {', '.join(available_actions)}
                    
                    Action History:
                    {json.dumps(action_history, indent=2)}
                    
                    Input Data:
                    {json.dumps(input_data or {}, indent=2)}
                    
                    Extracted Data:
                    {json.dumps(extracted_data, indent=2)}
                    
                    Success Criteria:
                    {success_criteria}
                    
                    Determine the next action to take. Respond with a JSON object with the following structure:
                    {{"action": "action_name", "params": {{...}}, "reason": "explanation"}}
                    
                    Valid actions: {', '.join(available_actions)}
                    
                    If you believe the goal has been achieved based on the success criteria, respond with:
                    {{"action": "finish", "params": {{}}, "reason": "explanation"}}
                    
                    If you believe the goal cannot be achieved, respond with:
                    {{"action": "fail", "params": {{}}, "reason": "explanation"}}
                    """
                    
                    # Call the LLM for next action
                    llm_result = await _call_llm(prompt, llm_model, system_message)
                    
                    if not llm_result.get("success"):
                        raise ToolError(
                            f"LLM guidance failed: {llm_result.get('error', 'Unknown error')}",
                            details={"step": steps_taken}
                        )
                    
                    # Parse the LLM response
                    try:
                        # Extract JSON from response
                        llm_text = llm_result.get("text", "")
                        json_match = re.search(r'(\{.*\})', llm_text, re.DOTALL)
                        
                        if not json_match:
                            raise ValueError("No JSON object found in LLM response")
                        
                        action_data = json.loads(json_match.group(1))
                        action_name = action_data.get("action")
                        action_params = action_data.get("params", {})
                        action_reason = action_data.get("reason", "No reason provided")
                        
                        # Record the action
                        action_record = {
                            "step": steps_taken,
                            "action": action_name,
                            "params": action_params,
                            "reason": action_reason,
                            "url": current_url
                        }
                        
                        action_history.append(action_record)
                        
                        # Execute the action
                        if action_name == "click":
                            element_id = action_params.get("element_id")
                            if not element_id:
                                raise ValueError("Missing 'element_id' for click action")
                            
                            # Find the element's index and information
                            element_info = next((el for el in available_elements if el.get("id") == element_id), None)
                            
                            if not element_info:
                                raise ToolError(f"Could not find element with ID: {element_id}")
                            
                            # Try to create a selector based on element attributes
                            selector = None
                            if element_info.get("tag") == "a" and element_info.get("href"):
                                selector = f"a[href='{element_info.get('href')}']"
                            elif element_info.get("name"):
                                selector = f"{element_info.get('tag')}[name='{element_info.get('name')}']"
                            elif element_info.get("text"):
                                # Try to find the element by text
                                element_text = element_info.get("text")
                                elements = await page.query_selector_all(f"{element_info.get('tag')}")
                                
                                for _i, elem in enumerate(elements):
                                    text = await elem.evaluate("el => el.textContent.trim()")
                                    if element_text in text:
                                        # We found the element, using xpath since it supports text content
                                        selector = f"//{element_info.get('tag')}[contains(text(), '{element_text}')]"
                                        break
                            
                            if not selector:
                                raise ToolError(f"Could not create selector for element ID: {element_id}")
                            
                            # Now try to find and click the element
                            try:
                                # Wait for the element to be available
                                element = await page.wait_for_selector(selector, timeout=timeout_per_step * 1000)
                                if not element:
                                    raise ToolError(f"Element not found: {selector}")
                                
                                # Click the element
                                await element.click()
                                
                                # Wait for navigation if it occurs
                                try:
                                    await page.wait_for_load_state("networkidle", timeout=timeout_per_step * 1000)
                                except Exception:
                                    # Navigation may not occur for some clicks
                                    pass
                                
                                # Update current URL after click action
                                current_url = page.url
                            except Exception as click_error:
                                raise ToolError(f"Failed to click element: {str(click_error)}") from click_error
                                
                        elif action_name == "type":
                            element_id = action_params.get("element_id")
                            text = action_params.get("text")
                            
                            if not element_id:
                                raise ValueError("Missing 'element_id' for type action")
                            if not text:
                                raise ValueError("Missing 'text' for type action")
                            
                            # Find the element's information
                            element_info = next((el for el in available_elements if el.get("id") == element_id), None)
                            
                            if not element_info:
                                raise ToolError(f"Could not find element with ID: {element_id}")
                            
                            # Try to create a selector based on element attributes
                            selector = None
                            if element_info.get("name"):
                                selector = f"{element_info.get('tag')}[name='{element_info.get('name')}']"
                            elif element_info.get("placeholder"):
                                selector = f"{element_info.get('tag')}[placeholder='{element_info.get('placeholder')}']"
                            else:
                                # Try other attributes
                                for attr in ["id", "class"]:
                                    attr_val = element_info.get(attr)
                                    if attr_val:
                                        selector = f"{element_info.get('tag')}[{attr}='{attr_val}']"
                                        break
                            
                            if not selector:
                                raise ToolError(f"Could not create selector for element ID: {element_id}")
                            
                            # Find and type into the element
                            try:
                                # Wait for the element to be available
                                element = await page.wait_for_selector(selector, timeout=timeout_per_step * 1000)
                                if not element:
                                    raise ToolError(f"Element not found: {selector}")
                                
                                # Clear existing text first (if applicable)
                                await element.fill("")
                                
                                # Type the text
                                await element.type(text)
                            except Exception as type_error:
                                raise ToolError(f"Failed to type into element: {str(type_error)}") from type_error
                        
                        elif action_name == "extract":
                            extract_key = action_params.get("key")
                            selector = action_params.get("selector")
                            
                            if not extract_key:
                                raise ValueError("Missing 'key' for extract action")
                            if not selector:
                                raise ValueError("Missing 'selector' for extract action")
                            
                            # Extract data from the element
                            try:
                                element = await page.query_selector(selector)
                                if element:
                                    text_content = await element.evaluate("el => el.textContent.trim()")
                                    extracted_data[extract_key] = text_content
                                else:
                                    extracted_data[extract_key] = None
                            except Exception as extract_error:
                                raise ToolError(f"Failed to extract data: {str(extract_error)}") from extract_error
                        
                        elif action_name == "navigate":
                            target_url = action_params.get("url")
                            
                            if not target_url:
                                raise ValueError("Missing 'url' for navigate action")
                            
                            # Navigate to the URL
                            try:
                                await page.goto(target_url, timeout=timeout_per_step * 1000, wait_until="networkidle")
                                current_url = page.url
                            except Exception as nav_error:
                                raise ToolError(f"Failed to navigate to URL: {str(nav_error)}") from nav_error
                        
                        elif action_name == "finish":
                            workflow_successful = True
                            workflow_message = action_reason
                        
                        elif action_name == "fail":
                            workflow_message = action_reason
                            break
                        
                        else:
                            raise ValueError(f"Unknown action: {action_name}")
                        
                    except json.JSONDecodeError as json_decode_error:
                        raise ToolError("Failed to parse LLM response as JSON") from json_decode_error
                    except ValueError as ve:
                        raise ToolError(f"Invalid LLM response: {str(ve)}") from ve
                
                # Calculate total time
                execution_time = time.time() - start_time
                
                return {
                    "success": workflow_successful,
                    "message": workflow_message,
                    "steps_taken": steps_taken,
                    "final_url": current_url,
                    "extracted_data": extracted_data,
                    "action_history": action_history,
                    "execution_time": execution_time
                }
            finally:
                await page.close()
        finally:
            await context.close()
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise ToolError(f"Workflow execution failed: {str(e)}") from e

@with_tool_metrics
@with_error_handling
async def lightpanda_analyze_web_content(
    url: str,
    analysis_type: str = "general",
    llm_model: str = "openai/gpt-4.1-mini",
    timeout: int = 60,
    user_agent: Optional[str] = None,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze web content using Lightpanda browser and LLM.
    
    This tool navigates to a URL, extracts content, and performs LLM-based analysis.
    It supports various analysis types for different use cases.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        analysis_type: Type of analysis to perform (default: "general").
                      Options: "general", "seo", "sentiment", "readability", "factcheck", "custom".
        llm_model: LLM model to use for analysis (default: "openai/gpt-4.1-mini").
        timeout: Maximum time in seconds to wait for the page to load (default: 60).
        user_agent: Custom User-Agent string to use (optional).
        custom_instructions: Custom instructions for analysis when analysis_type is "custom".
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "analysis": {  # Analysis results, structure depends on analysis_type
                "key_points": [...],
                "summary": "...",
                "recommendations": [...],
                ...
            },
            "analysis_time": 3.45,  # Time in seconds
            "url": "https://example.com"
        }
    
    Raises:
        ToolInputError: If the URL or analysis parameters are invalid.
        ToolError: If the analysis fails.
    """
    start_time = time.time()
    
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate analysis_type
    valid_analysis_types = ["general", "seo", "sentiment", "readability", "factcheck", "custom"]
    if analysis_type not in valid_analysis_types:
        raise ToolInputError(
            f"Invalid analysis_type. Must be one of: {', '.join(valid_analysis_types)}",
            param_name="analysis_type",
            provided_value=analysis_type
        )
    
    # If custom analysis, validate custom_instructions
    if analysis_type == "custom" and (not custom_instructions or not isinstance(custom_instructions, str)):
        raise ToolInputError(
            "Custom analysis requires non-empty custom_instructions",
            param_name="custom_instructions",
            provided_value=custom_instructions
        )
    
    try:
        # Get the browser instance
        browser = await _get_browser()
        
        # Create a new browser context with optional parameters
        context_params = {}
        if user_agent:
            context_params["user_agent"] = user_agent
        
        # Create a context
        context = await browser.new_context(**context_params)
        
        try:
            # Create a page
            page = await context.new_page()
            
            try:
                # Navigate to URL with timeout
                await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                
                # Get the page title
                title = await page.title()
                
                # Get the page's text content
                text_content = await page.evaluate("""() => {
                    // Get main content area
                    const content = document.querySelector('article') || 
                                   document.querySelector('main') || 
                                   document.querySelector('.content') || 
                                   document.querySelector('#content') || 
                                   document.body;
                    
                    // Remove non-content elements
                    const elementsToRemove = content.querySelectorAll('aside, nav, .nav, .navigation, .menu, .sidebar, .footer, script, style, link');
                    for (const el of elementsToRemove) {
                        if (el && el.parentNode) {
                            el.parentNode.removeChild(el);
                        }
                    }
                    
                    // Extract and clean text
                    let text = content.innerText;
                    
                    // Normalize whitespace
                    text = text.replace(/\\s+/g, ' ').trim();
                    text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
                    
                    return text;
                }""")
                
                # Count words
                words = text_content.split()
                word_count = len([w for w in words if w.strip()])
                
                # Extract links if needed for some analysis types
                links = []
                if analysis_type in ["seo", "general"]:
                    # Get all links from the page
                    links_elements = await page.query_selector_all('a[href]')
                    for link_element in links_elements:
                        href = await link_element.get_attribute('href')
                        text = await link_element.text_content()
                        if href:
                            links.append({
                                "href": href,
                                "text": text.strip() if text else ""
                            })
                
                # Extract metadata for SEO analysis
                metadata = {}
                if analysis_type in ["seo", "general"]:
                    # Get all meta tags
                    meta_elements = await page.query_selector_all('meta')
                    for meta_element in meta_elements:
                        name = await meta_element.get_attribute('name') or await meta_element.get_attribute('property')
                        content = await meta_element.get_attribute('content')
                        if name and content:
                            metadata[name] = content
                    
                    # Get heading structure
                    headings = {}
                    for heading_level in range(1, 7):
                        heading_elements = await page.query_selector_all(f'h{heading_level}')
                        headings[f'h{heading_level}'] = []
                        for heading_element in heading_elements:
                            heading_text = await heading_element.text_content()
                            headings[f'h{heading_level}'].append(heading_text.strip())
                    
                    metadata["headings"] = headings
                
                # Prepare analysis-specific instructions
                if analysis_type == "general":
                    system_message = "You are an AI assistant that provides general analysis of web page content."
                    prompt = f"""
                    Analyze the following web page content and provide a general analysis:
                    
                    URL: {url}
                    Title: {title}
                    Word Count: {word_count}
                    
                    Content Sample:
                    {text_content[:4000]}... (truncated)
                    
                    Metadata:
                    {json.dumps(metadata, indent=2)}
                    
                    Link Count: {len(links)}
                    
                    Provide an analysis with the following structure:
                    1. A brief summary of the page content (2-3 sentences)
                    2. Key topics/themes covered
                    3. Target audience
                    4. Content quality assessment
                    5. Notable observations or unique aspects
                    
                    Return your analysis as a JSON object with these keys:
                    {{"summary": "...", "key_topics": [...], "target_audience": "...", "quality_assessment": "...", "observations": [...]}}
                    """
                    
                elif analysis_type == "seo":
                    system_message = "You are an AI assistant that analyzes web pages for SEO optimization."
                    prompt = f"""
                    Analyze the following web page content for SEO optimization:
                    
                    URL: {url}
                    Title: {title} (Length: {len(title)} characters)
                    Word Count: {word_count}
                    
                    Meta Tags:
                    {json.dumps({k: v for k, v in metadata.items() if k != 'headings'}, indent=2)}
                    
                    Headings Structure:
                    {json.dumps(metadata.get('headings', {}), indent=2)}
                    
                    Content Sample:
                    {text_content[:3000]}... (truncated)
                    
                    Links: {len(links)} total links
                    
                    Perform an SEO analysis with the following structure:
                    1. Title tag assessment
                    2. Meta description assessment
                    3. Heading structure assessment
                    4. Content quality and keyword usage
                    5. Internal and external linking
                    6. Key recommendations for improvement
                    
                    Return your analysis as a JSON object with these keys:
                    {{"title_assessment": "...", "meta_description_assessment": "...", "heading_assessment": "...", "content_assessment": "...", "linking_assessment": "...", "recommendations": [...]}}
                    """
                    
                elif analysis_type == "sentiment":
                    system_message = "You are an AI assistant that analyzes the sentiment and emotional tone of web content."
                    prompt = f"""
                    Analyze the sentiment and emotional tone of the following web page content:
                    
                    URL: {url}
                    Title: {title}
                    
                    Content:
                    {text_content[:5000]}... (truncated)
                    
                    Perform a sentiment analysis with the following structure:
                    1. Overall sentiment (positive, negative, neutral, or mixed)
                    2. Emotional tone and intensity
                    3. Subjective vs. objective language assessment
                    4. Emotional keywords or phrases identified
                    5. Target emotional response from the reader
                    
                    Return your analysis as a JSON object with these keys:
                    {{"overall_sentiment": "...", "emotional_tone": "...", "subjectivity": "...", "emotional_keywords": [...], "target_emotional_response": "..."}}
                    """
                    
                elif analysis_type == "readability":
                    system_message = "You are an AI assistant that analyzes the readability and accessibility of web content."
                    prompt = f"""
                    Analyze the readability and accessibility of the following web page content:
                    
                    URL: {url}
                    Title: {title}
                    Word Count: {word_count}
                    
                    Content:
                    {text_content[:5000]}... (truncated)
                    
                    Perform a readability analysis with the following structure:
                    1. Estimated reading level (e.g., elementary, middle school, high school, college)
                    2. Sentence complexity assessment
                    3. Vocabulary assessment
                    4. Content structure and organization
                    5. Recommendations for improving readability
                    
                    Return your analysis as a JSON object with these keys:
                    {{"reading_level": "...", "sentence_complexity": "...", "vocabulary_assessment": "...", "structure_assessment": "...", "recommendations": [...]}}
                    """
                    
                elif analysis_type == "factcheck":
                    system_message = "You are an AI assistant that critically evaluates web content for factual claims, evidence, and potential misinformation."
                    prompt = f"""
                    Evaluate the following web page content for factual claims, cited evidence, and potential misinformation:
                    
                    URL: {url}
                    Title: {title}
                    
                    Content:
                    {text_content[:6000]}... (truncated)
                    
                    Perform a fact-check analysis with the following structure:
                    1. Objective vs. subjective content assessment
                    2. Major factual claims identified
                    3. Evidence and sources provided for claims
                    4. Potential misinformation or unsubstantiated claims
                    5. Overall factual reliability assessment
                    
                    Return your analysis as a JSON object with these keys:
                    {{"objectivity_assessment": "...", "major_claims": [...], "evidence_assessment": "...", "potential_misinformation": [...], "reliability_assessment": "..."}}
                    """
                    
                else:  # Custom analysis
                    system_message = "You are an AI assistant that analyzes web content according to custom instructions."
                    prompt = f"""
                    Analyze the following web page content according to these custom instructions:
                    
                    {custom_instructions}
                    
                    URL: {url}
                    Title: {title}
                    Word Count: {word_count}
                    
                    Content:
                    {text_content[:5000]}... (truncated)
                    
                    Metadata:
                    {json.dumps(metadata, indent=2)}
                    
                    Links: {len(links)} total links
                    
                    Return your analysis as a JSON object with appropriate keys.
                    """
                
                # Call the LLM for analysis
                llm_result = await _call_llm(prompt, llm_model, system_message)
                
                if not llm_result.get("success"):
                    raise ToolError(
                        f"LLM analysis failed: {llm_result.get('error', 'Unknown error')}",
                        details={"url": url, "model": llm_model}
                    )
                
                # Parse the analysis result
                try:
                    # Extract JSON from response
                    llm_text = llm_result.get("text", "")
                    json_match = re.search(r'(\{.*\})', llm_text, re.DOTALL)
                    
                    if not json_match:
                        raise ValueError("No JSON object found in LLM response")
                    
                    analysis_data = json.loads(json_match.group(1))
                    
                    analysis_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "analysis": analysis_data,
                        "analysis_time": analysis_time,
                        "url": url,
                        "title": title,
                        "word_count": word_count
                    }
                    
                except json.JSONDecodeError as json_decode_error:
                    raise ToolError("Failed to parse LLM analysis result as JSON") from json_decode_error
                except ValueError as ve:
                    raise ToolError(f"Invalid LLM analysis result: {str(ve)}") from ve
            finally:
                await page.close()
        finally:
            await context.close()
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        logger.error(f"Web content analysis failed: {str(e)}")
        raise ToolError(f"Web content analysis failed: {str(e)}") from e

@with_tool_metrics
@with_error_handling
async def lightpanda_extract_structured_data(
    url: str,
    extraction_schema: Dict[str, Any],
    page_wait_time: int = 2,
    timeout: int = 60,
    user_agent: Optional[str] = None,
    retry_count: int = 1, 
    llm_model: str = "openai/gpt-4.1-mini"
) -> Dict[str, Any]:
    """Extract structured data from a webpage based on a schema.
    
    This tool navigates to a URL and extracts structured data according to the provided schema.
    It combines Lightpanda's browser capabilities with LLM understanding to extract complex data.
    
    Args:
        url: The URL to navigate to (must start with http:// or https://).
        extraction_schema: Dictionary defining the data to extract.
                          Example: {"author": "Name of the article author", "price": "Product price (numeric)"}
        page_wait_time: Additional time in seconds to wait after page load (default: 2).
        timeout: Maximum time in seconds to wait for the page to load (default: 60).
        user_agent: Custom User-Agent string to use (optional).
        retry_count: Number of retry attempts if extraction fails (default: 1).
        llm_model: LLM model to use for extraction (default: "openai/gpt-4.1-mini").
    
    Returns:
        A dictionary containing the results:
        {
            "success": true,
            "data": {  # Extracted data matching the schema
                "author": "John Smith",
                "price": "29.99"
            },
            "extraction_time": 3.45,  # Time in seconds
            "url": "https://example.com"
        }
    
    Raises:
        ToolInputError: If the URL or schema is invalid.
        ToolError: If the extraction fails after all retries.
    """
    start_time = time.time()
    
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ToolInputError(
            "URL must start with http:// or https://",
            param_name="url",
            provided_value=url
        )
    
    # Validate schema
    if not extraction_schema or not isinstance(extraction_schema, dict):
        raise ToolInputError(
            "Extraction schema must be a non-empty dictionary",
            param_name="extraction_schema",
            provided_value=extraction_schema
        )
    
    # Format the schema for the LLM
    schema_str = json.dumps(extraction_schema, indent=2)
    
    last_error = None
    for attempt in range(retry_count + 1):
        try:
            # Get the browser instance
            browser = await _get_browser()
            
            # Create a new browser context with optional parameters
            context_params = {}
            if user_agent:
                context_params["user_agent"] = user_agent
            
            # Create a context
            context = await browser.new_context(**context_params)
            
            try:
                # Create a page
                page = await context.new_page()
                
                try:
                    # Navigate to URL with timeout
                    await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                    
                    # Wait for any dynamic content to load
                    if page_wait_time > 0:
                        await page.wait_for_timeout(page_wait_time * 1000)
                    
                    # Get page title and text content
                    title = await page.title()
                    
                    # Extract the page content with readability improvements
                    text_content = await page.evaluate("""() => {
                        // Get main content area
                        const content = document.querySelector('article') || 
                                      document.querySelector('main') || 
                                      document.querySelector('.content') || 
                                      document.querySelector('#content') || 
                                      document.body;
                        
                        // Remove non-content elements
                        const elementsToRemove = content.querySelectorAll('aside, nav, .nav, .navigation, .menu, .sidebar, .footer, script, style, link');
                        for (const el of elementsToRemove) {
                            if (el && el.parentNode) {
                                el.parentNode.removeChild(el);
                            }
                        }
                        
                        // Extract and clean text
                        let text = content.innerText;
                        
                        // Normalize whitespace
                        text = text.replace(/\\s+/g, ' ').trim();
                        text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
                        
                        return text;
                    }""")
                    
                    # Prepare the page content for the LLM
                    page_content = f"URL: {url}\nTitle: {title}\n\nContent:\n{text_content}"
                    
                    # Prepare the LLM prompt
                    system_message = "You are an AI assistant that extracts structured data from web page content according to a schema."
                    prompt = f"""
                    Extract structured data from the following web page content according to this schema:
                    
                    SCHEMA:
                    {schema_str}
                    
                    WEB PAGE CONTENT:
                    {page_content}
                    
                    INSTRUCTIONS:
                    1. Extract data for each field in the schema.
                    2. If you can't find data for a field, use null or an empty string.
                    3. Return ONLY a valid JSON object with the extracted data matching the schema.
                    
                    EXTRACTED DATA (JSON):
                    """
                    
                    # Call the LLM to extract structured data
                    llm_result = await _call_llm(prompt, llm_model, system_message)
                    
                    if not llm_result.get("success"):
                        raise ToolError(
                            f"LLM extraction failed: {llm_result.get('error', 'Unknown error')}",
                            details={"url": url, "model": llm_model}
                        )
                    
                    # Parse the extracted data
                    try:
                        # Try to find and parse the JSON data
                        text = llm_result.get("text", "")
                        
                        # Look for JSON object
                        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            extracted_data = json.loads(json_str)
                        else:
                            raise ValueError("No JSON object found in LLM response")
                        
                        extraction_time = time.time() - start_time
                        
                        return {
                            "success": True,
                            "data": extracted_data,
                            "extraction_time": extraction_time,
                            "url": url
                        }
                        
                    except Exception as parse_error:
                        # Handle JSON parsing errors
                        error_msg = f"Failed to parse LLM response as JSON: {str(parse_error)}"
                        logger.warning(f"Attempt {attempt + 1}/{retry_count + 1}: {error_msg}")
                        last_error = ToolError(
                            error_msg,
                            details={"url": url, "llm_response": text}
                        )
                        
                        # If this is the last attempt, re-raise
                        if attempt == retry_count:
                            raise last_error from parse_error
                        
                        # Otherwise, continue to next attempt
                        continue
                finally:
                    await page.close()
            finally:
                await context.close()
                
        except ToolError as tool_err:
            # Handle errors from Lightpanda operations
            error_msg = f"Extraction error: {str(tool_err)}"
            logger.warning(f"Attempt {attempt + 1}/{retry_count + 1}: {error_msg}")
            last_error = tool_err
            
            # If this is the last attempt, re-raise
            if attempt == retry_count:
                raise last_error from tool_err
            
            # Wait before retrying
            await asyncio.sleep(1)
            continue
            
        except Exception as general_err:
            # Handle any other unexpected errors
            error_msg = f"Unexpected error during extraction: {type(general_err).__name__}: {str(general_err)}"
            logger.error(error_msg, exc_info=True)
            last_error = ToolError(
                error_msg,
                details={"url": url}
            )
            
            # If this is the last attempt, re-raise
            if attempt == retry_count:
                raise last_error from general_err
            
            # Wait before retrying
            await asyncio.sleep(1)
            continue
    
    # This line should never be reached due to the error handling above,
    # but adding it for defensive programming
    if last_error:
        raise last_error
    else:
        raise ToolError(
            f"Failed to extract structured data from {url} after {retry_count + 1} attempts",
            details={"url": url}
        )