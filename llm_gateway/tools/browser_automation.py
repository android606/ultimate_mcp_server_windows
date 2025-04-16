"""Playwright browser automation tools for LLM Gateway.

This module provides a comprehensive set of tools for browser automation using Playwright,
enabling actions like navigation, element interaction, screenshots, and more through a
standardized API compatible with LLM Gateway.
"""

import asyncio
import base64
import json
import csv
import io
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Callable

import aiofiles
from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Locator,
    Page,
    Playwright,
    Response,
    async_playwright,
)

from llm_gateway.constants import TaskType
from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.tools.completion import generate_completion, chat_completion
from llm_gateway.tools.filesystem import create_directory
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
                f"Unsupported browser type: {browser_name}. Use 'chromium', 'firefox', or 'webkit'.",
                http_status_code=400
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
                         f"Browser {browser_name} is not installed. Use browser_install tool to install it.",
                         http_status_code=500
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
            message=f"Element with ref {ref} not found. This function relies on proper snapshot implementation.",
            http_status_code=404
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
                message=f"Browser installation failed: {error_output}",
                http_status_code=500
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
        
        raise ToolError(error_msg, http_status_code=500) from e

async def _get_simplified_page_state_for_llm(max_elements: int = 75) -> Dict[str, Any]:
    """Get a simplified version of the page state for LLM consumption."""
    script = f"""
    () => {{
        const MAX_ELEMENTS = {max_elements}; // Limit the number of elements extracted
        const MAX_TEXT_LEN = 100; // Limit text length per element
        const elements = [];
        let element_counter = 0;

        // Prioritize potentially relevant elements
        const selectors = [
            'a[href*="pdf"]', // PDF links first
            'a[href*="present"]', 'a[href*="event"]', 'a[href*="quarter"]', 'a[href*="investor"]', // Keywords
            'button', 'input[type="button"]', 'input[type="submit"]', // Buttons
            'a[href]' // All other links
        ];

        const seenElements = new Set(); // Avoid duplicates

        selectors.forEach(selector => {{
            document.querySelectorAll(selector).forEach((el) => {{
                if (element_counter >= MAX_ELEMENTS) return;
                if (seenElements.has(el)) return; // Skip if already processed

                // Basic visibility check (getBoundingClientRect is generally reliable)
                const rect = el.getBoundingClientRect();
                const isVisible = rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && rect.right <= (window.innerWidth || document.documentElement.clientWidth);
                // More complex check using computed style (can be slower)
                // const style = window.getComputedStyle(el);
                // const isVisible = style.display !== 'none' && style.visibility !== 'hidden' && el.offsetParent !== null;
                if (!isVisible) return;

                let description = el.textContent?.trim() ||
                                    el.getAttribute('aria-label') ||
                                    el.title ||
            let description = el.textContent?.trim() ||
                                el.getAttribute('aria-label') ||
                                el.title ||
                                el.value ||
                                el.alt ||
                                el.id ||
                                el.className.split(' ').filter(c=>c.length > 2)[0] || // First meaningful class name
                                el.tagName;

            if (description.length > MAX_TEXT_LEN) {{
                description = description.substring(0, MAX_TEXT_LEN - 3) + '...';
            }}

            const elementId = `el_${{element_counter++}}`; // Simple ID for LLM reference
            const elementInfo = {{
                id: elementId, // ID for LLM to refer to
                tag: el.tagName.toLowerCase(),
                text: description || '<no text>', // Ensure text is never empty
            }};

            // Add href for links, classify them
            if (el.href) {{
                const href = el.href;
                // Use a simpler regex check for PDF extension, case-insensitive
                if (/\\.pdf$/i.test(href)) {{
                    elementInfo.type = 'pdf_link';
                    elementInfo.href = href;
                }} else if (href.startsWith('http') || href.startsWith('/')) {{
                        elementInfo.type = 'nav_link';
                        // Avoid sending excessively long URLs to the LLM
                        elementInfo.href = href.length > 250 ? href.substring(0, 250) + '...' : href;
                }} else {{
                        // Exclude non-nav/pdf links like mailto:, javascript: for cleaner LLM context
                        return; // Skip adding 'other_link' types
                }}
            }} else {{
                    elementInfo.type = el.tagName.toLowerCase(); // button, input
            }}

            elements.push(elementInfo);
            seenElements.add(el); // Mark as seen
        }});
    }});

    // Limit total text summary to avoid huge prompts
    let bodyText = document.body.innerText || "";
    let textSummary = bodyText.substring(0, 2000); // Increased summary length slightly
    if (bodyText.length > 2000) textSummary += "... [text truncated]";

    return {{
        url: window.location.href,
        title: document.title || "No Title", // Ensure title is always a string
        elements: elements,
        text_summary: textSummary // Provide a text summary
    }};
    }}
    """
    try:
        page_id, page = await _ensure_page() # Ensure page exists
        # Use the existing low-level tool to execute JS
        result = await browser_execute_javascript(script=script)
        if result.get("success") and isinstance(result.get("result"), dict):
            page_state = result["result"]
            # Ensure title is always present, fallback to URL if empty
            if not page_state.get("title"):
                page_state["title"] = page_state.get("url", "Unknown URL")
            return page_state
        else:
            error_msg = result.get('error', 'Unknown JS execution error') if isinstance(result, dict) else "Invalid JS result"
            logger.error(f"Failed to get page state via JS for page {page_id}: {error_msg}")
            # Fallback using other tools
            fallback_url_res = await browser_execute_javascript("() => window.location.href")
            fallback_title_res = await browser_execute_javascript("() => document.title")
            fallback_text_res = await browser_get_text(selector="body")
            fallback_text = (fallback_text_res.get("text") or "")[:2000] + "... [fallback text truncated]" if fallback_text_res.get("success") else "Could not get body text."
            return {
                "url": fallback_url_res.get("result", "unknown_url"),
                "title": fallback_title_res.get("result") or "Unknown Title",
                "error": f"Failed to get detailed page state via JS ({error_msg}). Using fallback.",
                "elements": [],
                "text_summary": fallback_text
            }
    except Exception as e:
        logger.error(f"Exception getting simplified page state: {e}", exc_info=True)
        # Attempt graceful fallback even on exception
        try:
            page_id, page = await _ensure_page()
            page_url = page.url if page and not page.is_closed() else "unknown_url"
            page_title = await page.title() if page and not page.is_closed() else "Unknown Title"
            return {
                "url": page_url,
                "title": page_title,
                "error": f"Exception getting page state: {type(e).__name__}",
                "elements": [],
                "text_summary": "Failed to get page state due to exception."
            }
        except Exception as fallback_e:
             logger.critical(f"Critical failure getting page state fallback: {fallback_e}", exc_info=True)
             return {"url": "critical_error", "title": "critical_error", "error": "Critical failure getting page state.", "elements": []}
        
async def _call_browser_llm(
    messages: List[Dict[str, str]],
    model: str,
    task_description: str,
    expected_json: bool = True
) -> Optional[Dict[str, Any]]:
    logger.debug(f"Sending prompt to LLM ({model}) for {task_description}")
    if not messages:
        logger.error(f"LLM call attempted with empty messages list for {task_description}.")
        return {"action": "error", "error": "Internal error: Empty messages list."}

    try:
        # Extract prompt from the last user message
        user_messages = [m for m in messages if m.get('role') == 'user']
        if not user_messages:
            logger.error(f"No user messages found in message list for {task_description}")
            return {"action": "error", "error": "Internal error: No user messages in list."}
        
        prompt = user_messages[-1].get('content', '')
        if not prompt:
            logger.error(f"Empty prompt in last user message for {task_description}")
            return {"action": "error", "error": "Internal error: Empty prompt."}
        
        completion_params = {
            "model": model, 
            "prompt": prompt,
            "temperature": 0.1, # Low temperature for focused tasks
        }
        
        provider_name_for_debug = model.split('/')[0] if '/' in model else "openai" 
        if expected_json and provider_name_for_debug != "openai": 
             completion_params["additional_params"] = {"response_format": {"type": "json_object"}}
             logger.debug(f"Requesting JSON format for provider: {provider_name_for_debug}")
        elif expected_json and provider_name_for_debug == "openai":
             logger.warning(f"Skipping JSON format request for OpenAI provider due to potential issue (debugging). Expecting natural language JSON.")

        # Call the generate_completion tool instead of chat_completion
        logger.info(f"Calling generate_completion for {task_description} with model {model}")
        response = await generate_completion(**completion_params)

        # Extraction logic remains the same
        if response.get("success"):
            llm_text = response.get("text", "")
            if not llm_text:
                 logger.warning(f"LLM returned empty content for {task_description}")
                 if expected_json:
                     return {"action": "error", "error": "LLM returned empty content when JSON was expected."}
                 else:
                      return {"text": ""} # Return empty text if JSON wasn't required

            logger.debug(f"LLM Raw Response for {task_description}: {llm_text}")

            if expected_json:
                # Attempt to parse JSON, with robust cleaning
                try:
                    # Try direct parse first
                    try:
                        action_data = json.loads(llm_text)
                    except json.JSONDecodeError:
                        # If direct parse fails, try cleaning markdown fences
                        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", llm_text)
                        if match:
                            cleaned_output = match.group(1).strip()
                        else:
                            # Assume it might be plain JSON without fences but with surrounding text
                            cleaned_output = llm_text.strip()
                            # Attempt to extract first valid JSON object literal
                            brace_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
                            if brace_match:
                                cleaned_output = brace_match.group(0)
                            else:
                                 # If no object found, maybe it's just not JSON
                                 raise json.JSONDecodeError("No JSON object found", cleaned_output, 0)


                        if not cleaned_output:
                            logger.error(f"LLM returned empty JSON string after cleaning for {task_description}")
                            raise json.JSONDecodeError("Empty string after cleaning", cleaned_output, 0)

                        action_data = json.loads(cleaned_output)

                    if not isinstance(action_data, dict):
                        logger.error(f"LLM JSON response is not a dictionary for {task_description}: {type(action_data)}")
                        return {"action": "error", "error": f"LLM response was JSON but not a dictionary (type: {type(action_data)})."}

                    logger.debug(f"LLM parsed action for {task_description}: {action_data}")
                    return action_data

                except json.JSONDecodeError as json_err:
                    logger.error(f"LLM response for {task_description} not valid JSON: {json_err}\nRaw text: '''{llm_text}'''")
                    return {"action": "error", "error": f"LLM response was not valid JSON: {json_err}"}
                except Exception as parse_err:
                    logger.error(f"Error parsing LLM action JSON for {task_description}: {parse_err}\nRaw text: '''{llm_text}'''")
                    return {"action": "error", "error": f"Error parsing LLM JSON: {parse_err}"}
            else:
                 # Return raw text if JSON wasn't expected
                 return {"text": llm_text}
        else:
            # Use error details from the chat_completion response
            error_detail = response.get("error", "Unknown LLM call error")
            error_code = response.get("error_code", "LLM_CALL_FAILED")
            logger.error(f"LLM call failed for {task_description}: {error_code} - {error_detail}")
            # Include details if available in the response error
            details = response.get("details", {})
            return {"action": "error", "error": f"LLM call failed ({error_code}): {error_detail}", "details": details}

    except Exception as e:
        # Catch errors in setting up the call or unexpected issues
        logger.error(f"Exception preparing or processing LLM call for {task_description}: {e}", exc_info=True)
        return {"action": "error", "error": f"Exception during LLM interaction: {type(e).__name__}: {e}"}

def _sanitize_filename(name: str) -> str:
    """Internal helper to remove or replace characters invalid for filenames."""
    if not isinstance(name, str): name = str(name) # Ensure string
    name = name.strip()
    # Replace sequences of whitespace (including newline, tab etc) with a single underscore
    name = re.sub(r'\s+', '_', name)
    # Remove or replace characters typically invalid or problematic in filenames across OS
    # Removed: .,;: -- allow periods for extensions, maybe allow others if needed
    name = re.sub(r'[\\/*?"<>|()\'\[\]!@#$%^&={}]', '', name) # Be more aggressive in removal
    # Replace multiple consecutive underscores with a single one
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores that might result
    name = name.strip('_')
    # Handle case where name becomes empty after sanitization
    if not name:
        name = "sanitized_empty_name_" + str(uuid.uuid4())[:4] # Add unique suffix
    return name[:150] # Limit length for safety

async def _perform_web_search(
    query: str,
    engine: str = "google",
    num_results: int = 10 # Fetch more results initially for filtering
) -> List[Dict[str, str]]:
    """
    Performs a web search using the browser and extracts basic result info.
    Internal helper. Returns list of {"url": ..., "title": ..., "snippet": ...}.
    """
    logger.info(f"Performing web search on {engine} for: '{query}' (requesting top {num_results})")
    search_results = []
    selector_based_success = False

    try:
        # --- STEP 1: Try Selector-Based Approach --- 
        logger.debug(f"Attempting selector-based search on {engine}...")
        # Define engine specifics (copied/adapted from multi_engine_search_summary)
        search_urls = {"google": "https://www.google.com", "bing": "https://www.bing.com", "duckduckgo": "https://duckduckgo.com"}
        search_selectors = {"google": "textarea[name='q']", "bing": "input[name='q']", "duckduckgo": "input[name='q']"}
        results_selectors = {"google": "div.g", "bing": "li.b_algo", "duckduckgo": "article.result"} 
        link_selectors = {"google": "a[href]", "bing": "h2 > a", "duckduckgo": "a.result__a"}
        title_selectors = {"google": "h3", "bing": "h2", "duckduckgo": "h2 > a.result__a"} 
        snippet_selectors = {"google": "div[data-sncf~='1'], .VwiC3b", "bing": ".b_caption p", "duckduckgo": ".result__snippet"}

        engine_key = engine.lower()
        search_url = search_urls.get(engine_key, search_urls["google"])
        search_selector = search_selectors.get(engine_key, search_selectors["google"])
        results_selector = results_selectors.get(engine_key, results_selectors["google"])
        link_selector = link_selectors.get(engine_key, link_selectors["google"])
        title_selector = title_selectors.get(engine_key, title_selectors["google"])
        snippet_selector = snippet_selectors.get(engine_key, snippet_selectors["google"])

        # Execute search using low-level tools
        nav_res = await browser_navigate(url=search_url, wait_until="domcontentloaded", capture_snapshot=False)
        if not nav_res.get("success"): raise ToolError(f"Navigation failed: {nav_res.get('error')}")
        
        type_res = await browser_type(selector=search_selector, text=query, press_enter=True, capture_snapshot=False)
        if not type_res.get("success"): raise ToolError(f"Typing failed: {type_res.get('error')}")

        logger.info(f"Waiting for results selector: {results_selector}") 
        # Ensure we use the correct variable holding the engine-specific selector
        wait_res = await browser_wait(wait_type="selector", value=results_selector, timeout=15000, capture_snapshot=False) # Reduced timeout slightly
        if not wait_res.get("success"): raise ToolError(f"Waiting for results failed: {wait_res.get('error')}")
        
        await asyncio.sleep(2.0) # Let results settle

        # Extract results via JS 
        script = f"""
        () => {{
            const results = [];
            const resultElements = document.querySelectorAll('{results_selector}');
            const numToExtract = {num_results}; 
            console.log(`Found ${{resultElements.length}} potential results for '{results_selector}' on {engine_key}`);
            for (let i = 0; i < resultElements.length && results.length < numToExtract; i++) {{
                const el = resultElements[i];
                try {{
                    const linkEl = el.querySelector('{link_selector}');
                    const titleEl = el.querySelector('{title_selector}');
                    const snippetEl = el.querySelector('{snippet_selector}');
                    const url = linkEl?.href;
                    let title = titleEl?.innerText?.trim() || linkEl?.innerText?.trim();
                    let snippet = snippetEl?.innerText?.trim();
                    if (url && url.startsWith('http') && title && !url.includes('duckduckgo.com/y.js')) {{ 
                        title = title.replace(/^Cached - /i, '').trim();
                        snippet = snippet ? snippet.substring(0, 500) + (snippet.length > 500 ? '...' : '') : '';
                        results.push({{ url: url, title: title, snippet: snippet || '' }});
                    }}
                }} catch (e) {{ console.error('Error processing search result element:', e); }}
            }}
            return results;
        }}"""
        extract_res = await browser_execute_javascript(script=script)
        if extract_res.get("success") and isinstance(extract_res.get("result"), list):
            search_results = extract_res["result"]
            if search_results:
                 logger.info(f"Successfully extracted {len(search_results)} results from {engine} using selectors.")
                 selector_based_success = True
            else:
                 logger.warning(f"Selector-based JS extraction returned 0 results from {engine}.")
        else:
            logger.warning(f"Failed to extract results via JS from {engine} using selectors: {extract_res.get('error')}")
            # If JS extraction fails even after waiting, trigger fallback
            raise ToolError("Selector-based JS extraction failed")

    except (ToolError, ToolInputError, Exception) as selector_err:
        logger.warning(f"Selector-based search failed: {type(selector_err).__name__}: {selector_err}. Falling back to LLM-guided search.")
        
        # --- STEP 2: LLM-Guided Fallback --- 
        try:
            # We might be on the search engine homepage or results page depending on where the error occurred.
            page_id, current_page_obj = await _ensure_page()
            logger.info(f"LLM Fallback: Getting page state for {current_page_obj.url}")
            page_state = await _get_simplified_page_state_for_llm()
            if page_state.get("error"):
                 raise ToolError(f"LLM Fallback failed: Could not get page state - {page_state['error']}")
            
            # TODO: Implement LLM calls to find input, type, click submit, wait, get results
            # This requires significant changes, including potentially using a different LLM model
            # and carefully crafting prompts. For now, we'll just log the failure and return empty.
            
            # --- LLM Task 1: Find search input & type (Placeholder) ---
            logger.warning("LLM Fallback: Search input identification not implemented.")
            # llm_input_res = await _call_browser_llm(...) 
            # await browser_type(selector=llm_input_res['element_id'], ...)
            
            # --- LLM Task 2: Find submit & click (Placeholder) ---
            logger.warning("LLM Fallback: Submit button identification not implemented.")
            # llm_submit_res = await _call_browser_llm(...)
            # await browser_click(selector=llm_submit_res['element_id'], ...)
            
            # --- LLM Task 3: Wait & Extract Results (Placeholder) ---
            logger.warning("LLM Fallback: Results extraction not implemented.")
            # await asyncio.sleep(5) # Crude wait
            # page_state_results = await _get_simplified_page_state_for_llm()
            # llm_results_res = await _call_browser_llm(...) 
            # search_results = llm_results_res.get('results', [])
            
            # For now, return empty results if fallback is triggered
            search_results = [] 
            logger.error("LLM-guided search fallback is not fully implemented. Returning empty results.")
            
        except Exception as llm_fallback_err:
            logger.error(f"Error during LLM-guided search fallback: {type(llm_fallback_err).__name__}: {llm_fallback_err}", exc_info=True)
            search_results = [] # Ensure empty results on fallback error
            
    # Return results from either selector path or LLM fallback (currently empty for fallback)
    return search_results

async def _select_relevant_urls_llm(
    search_results: List[Dict[str, str]],
    selection_prompt: str,
    topic: str,
    max_urls: int,
    llm_model: str
) -> List[str]:
    """Uses LLM to filter/select relevant URLs from search results."""
    if not search_results:
        return []

    logger.info(f"Asking LLM ({llm_model}) to select up to {max_urls} relevant URLs for topic '{topic}'...")

    # Format search results for the prompt
    results_context = f"Topic: {topic}\n\nSearch Results:\n"
    for i, res in enumerate(search_results):
        results_context += f"{i+1}. URL: {res.get('url', 'N/A')}\n   Title: {res.get('title', 'N/A')}\n   Snippet: {res.get('snippet', 'N/A')}\n\n"

    system_prompt = "You are an AI assistant evaluating search results to select the most relevant sources for a research topic based on user criteria. Respond ONLY with a valid JSON object like {\"selected_urls\": [\"url1\", \"url2\", ...]}."
    user_prompt = selection_prompt.format(topic=topic, search_results_context=results_context, max_urls=max_urls) # Pass context and max_urls to prompt template
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    llm_response = await _call_browser_llm(messages, llm_model, f"selecting relevant URLs for {topic}", expected_json=True)

    selected_urls = []
    if llm_response and llm_response.get("action") != "error" and isinstance(llm_response.get("selected_urls"), list):
        selected_urls = [url for url in llm_response["selected_urls"] if isinstance(url, str) and url.startswith("http")]
        logger.info(f"LLM selected {len(selected_urls)} URLs based on criteria.")
        # Ensure we don't exceed max_urls requested by user
        selected_urls = selected_urls[:max_urls]
    else:
        logger.warning(f"LLM failed to select relevant URLs or returned invalid format. Error: {llm_response.get('error', 'Invalid response')}")

    return selected_urls


async def _extract_info_from_url_llm(
    url: str,
    extraction_prompt_or_schema: Union[str, Dict],
    topic: str, # Add topic for context
    llm_model: str
) -> Optional[Dict[str, Any]]:
    """
    Navigates to URL, gets content, uses LLM to extract data.
    Returns extracted data dict or None on failure. Internal helper.
    """
    logger.debug(f"Extracting information from URL: {url} related to topic '{topic}'")
    page_content_context = f"Failed to retrieve content for {url}." # Default
    extracted_data = None
    try:
        # Navigate & Get Content State
        nav_res = await browser_navigate(url=url, wait_until="load", timeout=30000, capture_snapshot=False)
        if not nav_res.get("success"): raise ToolError(f"Navigation failed: {nav_res.get('error')}")
        await asyncio.sleep(1.5)
        page_state = await _get_simplified_page_state_for_llm()
        if page_state.get("error"):
             body_text_res = await browser_get_text(selector="body")
             page_content_context = body_text_res.get("text", f"Failed extraction: {page_state['error']}")[:10000]
        else:
             page_content_context = f"URL: {page_state.get('url')}\nTitle: {page_state.get('title')}\n"
             page_content_context += f"Text Summary:\n{page_state.get('text_summary', '')[:8000]}\n" # Limit context more aggressively

        # Prepare LLM Prompt for Extraction
        system_prompt = "You are an AI assistant extracting specific information or data points from web page content based on user instructions or a schema. Respond ONLY with a valid JSON object containing the extracted data."
        if isinstance(extraction_prompt_or_schema, dict): # JSON Schema approach
            user_prompt = f"Extract data matching the following JSON schema from the web page content provided below. The content relates to the topic '{topic}'. Respond ONLY with the valid JSON data object.\n\nSchema:\n```json\n{json.dumps(extraction_prompt_or_schema, indent=2)}\n```\n\nWeb Page Content Context:\n---\n{page_content_context}\n---\n\nExtracted JSON Data:"
        else: # Natural Language Prompt approach
            user_prompt = f"{extraction_prompt_or_schema}\n\nUse the web page content context below, related to the topic '{topic}', to perform the extraction. Respond ONLY with a valid JSON data object containing the requested information.\n\nWeb Page Content Context:\n---\n{page_content_context}\n---\n\nExtracted JSON Data:"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # Call LLM for extraction
        extraction_result = await _call_browser_llm(messages, llm_model, f"extracting info from {url} for '{topic}'", expected_json=True)

        if extraction_result and extraction_result.get("action") != "error" and isinstance(extraction_result, dict):
            extracted_data = extraction_result # The result *is* the data dict
            logger.info(f"Successfully extracted info from {url}")
        else:
            raise ToolError(f"LLM data extraction failed: {extraction_result.get('error', 'No valid JSON returned')}")

        return extracted_data # Return the dictionary extracted by the LLM

    except (ToolError, ToolInputError) as e:
        logger.warning(f"Failed to extract info from {url}: {type(e).__name__}: {e}")
        return {"_extraction_error": str(e), "_url": url} # Return error marker
    except Exception as e:
        logger.error(f"Unexpected error extracting info from {url}: {e}", exc_info=True)
        return {"_extraction_error": f"Unexpected error: {type(e).__name__}", "_url": url} # Return error marker


async def _synthesize_report_llm(
    extracted_snippets: List[Dict[str, Any]],
    synthesis_prompt: str,
    topic: str,
    report_format: str,
    llm_model: str
) -> str:
    """Uses LLM to synthesize a report from extracted data snippets."""
    if not extracted_snippets:
        return "No information was successfully extracted from the sources to synthesize a report."

    logger.info(f"Synthesizing report for topic '{topic}' from {len(extracted_snippets)} snippets using LLM {llm_model}")

    # Format snippets for the prompt context
    context = f"Research Topic: {topic}\n\nExtracted Information Snippets:\n"
    snippet_limit = 15000 // len(extracted_snippets) if extracted_snippets else 500 # Dynamically limit snippet length based on count

    for i, snippet_data in enumerate(extracted_snippets):
        # Include URL if available, exclude error markers
        source_url = snippet_data.get("_url", "Unknown Source")
        actual_data = {k: v for k, v in snippet_data.items() if not k.startswith("_")}
        snippet_str = json.dumps(actual_data, indent=2, default=str) # Pretty print snippet data
        if len(snippet_str) > snippet_limit: snippet_str = snippet_str[:snippet_limit] + "... [truncated]"
        context += f"\n--- Snippet {i+1} from {source_url} ---\n{snippet_str}\n"

    # Prepare Synthesis Prompt
    system_prompt = f"You are an AI assistant synthesizing a research report based on provided information snippets. The desired report format is: {report_format}."
    user_prompt = synthesis_prompt.format(topic=topic, extracted_information_context=context) # Pass snippets to the prompt template
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    # Call LLM for Synthesis
    # Expecting text output here, not necessarily JSON
    llm_response = await _call_browser_llm(messages, llm_model, f"synthesizing report for {topic}", expected_json=False)

    if llm_response and llm_response.get("action") != "error" and isinstance(llm_response.get("text"), str):
        report_content = llm_response["text"].strip()
        logger.success(f"Successfully synthesized report for topic '{topic}'.")
        return report_content
    else:
        error_msg = f"LLM report synthesis failed: {llm_response.get('error', 'No text returned')}" if llm_response else "LLM call failed unexpectedly."
        logger.error(error_msg)
        return f"Error: Failed to synthesize report. {error_msg}"
    
async def _extract_single_data_point(
    page: Page, # Pass the Page object directly
    url: str,
    data_point_def: Dict[str, Any],
    llm_model: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts a single data point from the current page based on its definition.
    Handles both selector-based and LLM-based extraction.
    Internal helper for monitor_web_data_points.

    Args:
        page: The active Playwright Page object.
        url: The URL of the page (for context).
        data_point_def: Dictionary defining the data point ('name', 'identifier', 'extraction_method').
        llm_model: The LLM model identifier to use if method is 'llm'.

    Returns:
        Tuple (extracted_value: Optional[str], error_message: Optional[str])
    """
    name = data_point_def.get("name", "Unnamed Data Point")
    identifier = data_point_def.get("identifier")
    method = data_point_def.get("extraction_method", "selector").lower()
    attribute = data_point_def.get("attribute") # Optional attribute for selector method

    if not identifier:
        return None, f"Missing 'identifier' for data point '{name}'"

    logger.debug(f"Extracting data point '{name}' from {url} using method '{method}' (Identifier: '{identifier[:50]}...')")

    extracted_value = None
    error_message = None

    try:
        if method == "selector":
            if not isinstance(identifier, str):
                return None, f"'identifier' must be a CSS selector string for method 'selector' (data point '{name}')"

            # Use low-level tools for extraction
            if attribute:
                # Extract an attribute value
                attr_res = await browser_get_attributes(selector=identifier, attributes=[attribute])
                if attr_res.get("success"):
                    extracted_value = attr_res.get("attributes", {}).get(attribute) # Can be None if attribute doesn't exist
                    if extracted_value is None:
                         logger.warning(f"Attribute '{attribute}' not found for selector '{identifier}' on {url}")
                         # Return None, let condition check handle it, or report as error? Let's return None.
                else:
                    error_message = f"Failed to get attribute '{attribute}' for selector '{identifier}': {attr_res.get('error', 'Unknown error')}"
            else:
                # Extract text content
                text_res = await browser_get_text(selector=identifier, trim=True)
                if text_res.get("success"):
                    extracted_value = text_res.get("text")
                else:
                    error_message = f"Failed to get text for selector '{identifier}': {text_res.get('error', 'Unknown error')}"

        elif method == "llm":
            if not isinstance(identifier, str):
                return None, f"'identifier' must be a natural language prompt/description for method 'llm' (data point '{name}')"

            # Get page context for the LLM
            page_state = await _get_simplified_page_state_for_llm() # Use existing helper
            page_context = f"URL: {page_state.get('url', url)}\nTitle: {page_state.get('title', 'N/A')}\n"
            page_context += f"Text Summary:\n{page_state.get('text_summary', 'N/A')[:4000]}\n\n" # Limit context
            # Maybe add relevant elements? Could make prompt too long. Stick to text summary for now.

            system_prompt = "You are an AI assistant extracting specific data points from web page content based on user instructions. Respond ONLY with the extracted value as a plain string. If the value cannot be found or extracted, respond with the exact string 'VALUE_NOT_FOUND'."
            user_prompt = "Extract the following data point based on the provided page content:\n'{identifier}'\n\nWeb Page Content Context:\n---\n{page_content_context}\n---\n\nExtracted Value:"

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            llm_response = await _call_browser_llm(messages, llm_model, f"extracting '{name}' from {url}", expected_json=False) # Expect text

            if llm_response and llm_response.get("action") != "error" and isinstance(llm_response.get("text"), str):
                value = llm_response["text"].strip()
                if value == "VALUE_NOT_FOUND":
                    extracted_value = None # Explicitly not found by LLM
                    logger.info(f"LLM indicated value not found for data point '{name}' on {url}")
                else:
                    extracted_value = value
            else:
                error_message = f"LLM extraction failed for '{name}': {llm_response.get('error', 'No text returned')}"

        else:
            error_message = f"Invalid 'extraction_method' ('{method}') for data point '{name}'. Use 'selector' or 'llm'."

        # Ensure extracted value is string or None
        if extracted_value is not None and not isinstance(extracted_value, str):
             extracted_value = str(extracted_value)

        return extracted_value, error_message

    except (ToolError, ToolInputError) as e:
        # Errors from underlying browser tools
        logger.warning(f"Browser tool error extracting '{name}' from {url}: {e}")
        return None, f"Browser tool error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error extracting data point '{name}' from {url}: {e}", exc_info=True)
        return None, f"Unexpected error: {type(e).__name__}"


async def _evaluate_data_point_condition(
    data_point_name: str,
    current_value: Optional[str],
    condition_def: Dict[str, Any],
    previous_value: Optional[str],
    llm_model: str
) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluates a condition against the extracted data point value.
    Internal helper for monitor_web_data_points.

    Args:
        data_point_name: Name of the data point (for logging).
        current_value: The recently extracted value (string or None).
        condition_def: Dictionary defining the condition ('condition', 'condition_value', 'llm_condition_prompt').
        previous_value: The value from the previous run (Optional[str]).
        llm_model: LLM model identifier for 'llm_eval' condition.

    Returns:
        Tuple (condition_met: bool, status_description: str, error_message: Optional[str])
    """
    condition_type = condition_def.get("condition", "changed").lower()
    condition_value_param = condition_def.get("condition_value") # Can be string, number, pattern
    llm_prompt = condition_def.get("llm_condition_prompt")

    condition_met = False
    status_desc = f"Condition '{condition_type}' evaluated." # Default
    error_msg = None

    # Handle case where extraction failed (current_value is None)
    if current_value is None:
        status_desc = "Condition check skipped: Current value could not be extracted."
        # Should condition_met be false or null/error? Let's say false.
        condition_met = False
        error_msg = "Value extraction failed"
        return condition_met, status_desc, error_msg

    try:
        if condition_type == "changed":
            # Check if previous value exists and is different from current
            if previous_value is None:
                status_desc = "Condition 'changed': No previous value to compare."
                condition_met = True # Treat first run as 'changed'
            else:
                condition_met = (current_value != previous_value)
                status_desc = f"Condition 'changed': {'Detected' if condition_met else 'Not detected'} (Current: '{current_value[:30]}...', Previous: '{previous_value[:30]}...')"

        elif condition_type == "equals":
            expected_value = str(condition_value_param) if condition_value_param is not None else ""
            condition_met = (current_value == expected_value)
            status_desc = f"Condition 'equals': {'Met' if condition_met else 'Not met'} (Current: '{current_value}', Expected: '{expected_value}')"

        elif condition_type == "contains":
            substring = str(condition_value_param) if condition_value_param is not None else ""
            if not substring: raise ValueError("condition_value (substring) required for 'contains'")
            condition_met = (substring in current_value)
            status_desc = f"Condition 'contains': {'Met' if condition_met else 'Not met'} (Current: '{current_value[:50]}...', Substring: '{substring}')"

        elif condition_type in ["greater_than", "less_than", "ge", "le"]:
            # Attempt numeric comparison
            try:
                current_num = float(current_value)
                expected_num = float(condition_value_param)
                op_map = { "greater_than": lambda a, b: a > b, "less_than": lambda a, b: a < b, "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b }
                if condition_type not in op_map: raise ValueError(f"Unknown numeric comparison {condition_type}") # Should not happen
                condition_met = op_map[condition_type](current_num, expected_num)
                op_symbol = { "greater_than": ">", "less_than": "<", "ge": ">=", "le": "<=" }[condition_type]
                status_desc = f"Condition '{condition_type}': {'Met' if condition_met else 'Not met'} (Current: {current_num} {op_symbol} Expected: {expected_num})"
            except (ValueError, TypeError) as num_err:
                error_msg = f"Numeric conversion failed for condition '{condition_type}': Current='{current_value}', Expected='{condition_value_param}' ({num_err})"
                status_desc = f"Condition '{condition_type}': Failed ({error_msg})"
                condition_met = False

        elif condition_type == "regex_match":
            pattern = str(condition_value_param) if condition_value_param else ""
            if not pattern: raise ValueError("condition_value (regex pattern) required for 'regex_match'")
            try:
                condition_met = bool(re.search(pattern, current_value))
                status_desc = f"Condition 'regex_match': {'Met' if condition_met else 'Not met'} (Current: '{current_value[:50]}...', Pattern: '{pattern}')"
            except re.error as regex_err:
                error_msg = f"Invalid regex pattern '{pattern}': {regex_err}"
                status_desc = f"Condition 'regex_match': Failed ({error_msg})"
                condition_met = False

        elif condition_type == "llm_eval":
            if not llm_prompt: raise ValueError("llm_condition_prompt required for 'llm_eval'")
            # Prepare context for LLM evaluation
            eval_context = f"Current Value: {current_value}\n"
            if previous_value is not None: eval_context += f"Previous Value: {previous_value}\n"
            if condition_value_param is not None: eval_context += f"Reference Value/Criteria: {condition_value_param}\n"

            system_prompt = "You are an AI assistant evaluating a condition based on provided data and criteria. Respond ONLY with a valid JSON object: {\"condition_met\": true} or {\"condition_met\": false}."
            user_prompt = f"{llm_prompt}\n\nContext:\n---\n{eval_context}---\n\nJSON Response:"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            llm_response = await _call_browser_llm(messages, llm_model, f"evaluating condition for '{data_point_name}'", expected_json=True)

            if llm_response and llm_response.get("action") != "error" and isinstance(llm_response.get("condition_met"), bool):
                condition_met = llm_response["condition_met"]
                status_desc = f"Condition 'llm_eval': {'Met' if condition_met else 'Not met'} (Evaluated by LLM)"
            else:
                error_msg = f"LLM condition evaluation failed: {llm_response.get('error', 'Invalid response format - expected {{\"condition_met\": bool}}')}"
                status_desc = f"Condition 'llm_eval': Failed ({error_msg})"
                condition_met = False

        else:
            error_msg = f"Unsupported condition type: '{condition_type}'"
            status_desc = f"Condition check skipped: {error_msg}"
            condition_met = False

    except Exception as e:
        logger.error(f"Unexpected error evaluating condition '{condition_type}' for '{data_point_name}': {e}", exc_info=True)
        error_msg = f"Unexpected evaluation error: {type(e).__name__}"
        status_desc = f"Condition '{condition_type}': Failed ({error_msg})"
        condition_met = False

    logger.debug(f"Condition evaluation for '{data_point_name}': Type='{condition_type}', Met={condition_met}, Desc='{status_desc}', Error='{error_msg}'")
    return condition_met, status_desc, error_msg

async def _find_element_locator_for_workflow(
    page: Page,
    element_id: str,
    page_state_elements: List[Dict[str, Any]]
) -> Optional[Locator]:
    """
    Attempts to find a reliable Playwright Locator based on the element_id
    provided by the LLM, using the context from the captured page state.
    Internal helper for execute_web_workflow.

    Args:
        page: The current Playwright Page object.
        element_id: The 'el_N' identifier chosen by the LLM.
        page_state_elements: The list of elements captured by _get_simplified_page_state_for_llm
                             *at the time the LLM made its decision*.

    Returns:
        A Playwright Locator object if found, otherwise None.
    """
    target_element_info = next((el for el in page_state_elements if el.get("id") == element_id), None)

    if not target_element_info:
        logger.warning(f"Could not find element info in page state for ID: {element_id}")
        return None

    tag = target_element_info.get("tag", "*") # Default to any tag if missing
    text = target_element_info.get("text")
    el_type = target_element_info.get("type") # e.g., 'nav_link', 'pdf_link', 'button'

    logger.debug(f"Attempting to locate element {element_id}: tag={tag}, text='{text}', type={el_type}")

    locator = None
    try:
        # Strategy 1: Use text if available and reasonably specific
        if text and text != '<no text>' and len(text) > 2:
            try:
                # Try finding by text first (often robust for buttons, links)
                # Use exact=False for flexibility, filter by tag potentially
                locator = page.locator(f'{tag}:text-matches("{re.escape(text)}", "i")').first
                # Verify quickly if it exists
                await locator.count()
                if await locator.count() > 0:
                    logger.debug(f"Located {element_id} by text: '{text}'")
                    return locator
                else:
                     logger.debug(f"Locator by text '{text}' found 0 elements.")
                     locator = None # Reset if count is 0
            except Exception as text_loc_err:
                logger.warning(f"Locating {element_id} by text '{text}' failed: {text_loc_err}. Trying other methods.")
                locator = None # Reset locator

        # Strategy 2: Fallback using CSS selector derived from index (Less Reliable)
        # This assumes the element order from JS snapshot matches nth-of-type, which is fragile.
        # Only use as a last resort.
        if locator is None:
            match = re.match(r"el_(\d+)", element_id)
            if match:
                element_index = int(match.group(1))
                # Construct a potential nth-of-type selector based on common interactive tags
                fallback_selector = f":is(a[href], button, input[type='button'], input[type='submit'], input[type='text'], textarea, select)[{element_index}]"
                # Alternative: Use Playwright's nth() - might be slightly better
                # fallback_locator = page.locator(":is(a[href], button, input[type='button'], input[type='submit'], input[type='text'], textarea, select)").nth(element_index)

                try:
                    logger.debug(f"Falling back to index-based selector for {element_id}: index {element_index}")
                    locator = page.locator(":is(a[href], button, input[type='button'], input[type='submit'], input[type='text'], textarea, select)").nth(element_index)
                    await locator.count() # Check if it exists
                    if await locator.count() > 0:
                         logger.warning(f"Located {element_id} using less reliable index-based method.")
                         return locator
                    else:
                         logger.warning(f"Index-based locator for {element_id} found 0 elements.")
                         return None
                except Exception as idx_loc_err:
                    logger.error(f"Fallback index-based locator failed for {element_id}: {idx_loc_err}")
                    return None
            else:
                logger.error(f"Invalid element_id format for fallback: {element_id}")
                return None

    except Exception as e:
        logger.error(f"Unexpected error finding locator for {element_id}: {e}", exc_info=True)
        return None

    return None # Should have returned earlier if successful

async def _summarize_single_url(url: str, query: str, llm_model: str, url_info: Dict) -> Dict:
    """Internal helper to navigate, extract text, and summarize a single URL."""
    logger.debug(f"Summarizing URL: {url}")
    page_text = "Content extraction failed."
    summary = "Summary generation failed."
    error = None
    try:
        # Use existing low-level tools within the helper
        nav_res = await browser_navigate(url=url, wait_until="load", timeout=25000)
        if not nav_res.get("success"): raise ToolError(f"Navigation failed: {nav_res.get('error')}")
        await asyncio.sleep(1.5)

        text_res = await browser_get_text(selector="body")
        if text_res.get("success"):
             page_text = (text_res.get("text") or "")[:10000] # Limit context
        else:
             logger.warning(f"browser_get_text failed for {url}: {text_res.get('error')}")
             page_text = url_info.get("snippet", "Content extraction failed.") # Fallback to snippet

        if not page_text.strip():
             page_text = url_info.get("snippet", "No content available.")
             logger.warning(f"No text content found on {url}, using snippet: '{page_text[:100]}...'")

        system_prompt_sum = "You are an AI assistant that concisely summarizes web page text relevant to a user query."
        user_prompt_sum = f"Concisely summarize the key information from the following web page content, focusing on its relevance to the search query '{query}'. Output a brief 2-3 sentence summary only.\n\nPage Content:\n---\n{page_text}\n---\n\nConcise Summary:"
        messages_sum = [{"role": "system", "content": system_prompt_sum}, {"role": "user", "content": user_prompt_sum}]

        summary_res = await _call_browser_llm(messages_sum, llm_model, f"summarizing {url}", expected_json=False)

        if summary_res and summary_res.get("action") != "error" and isinstance(summary_res.get("text"), str):
            summary = summary_res["text"].strip() or "Summary generation returned empty."
        else:
            error_detail = summary_res.get('error', 'LLM did not return text.') if summary_res else 'LLM call failed.'
            summary = f"Summary generation failed: {error_detail}"
            error = summary # Assign error if summary failed

    except Exception as page_err:
        logger.error(f"Failed processing page {url} for summary: {type(page_err).__name__}: {page_err}", exc_info=False) # Keep log less noisy
        error = f"{type(page_err).__name__}: {str(page_err)}"
        summary = f"Failed to process page: {error}"

    # Return structured result including potential error
    return {
        "url": url,
        "title": url_info.get("title", "Title N/A"),
        "summary": summary,
        "source_engine": url_info.get("source_engines", []),
        "error": error # Will be None if successful
    }

async def _find_urls_dynamically(
    instructions: Dict[str, Any],
    max_urls: int = 100 # Safety limit
) -> List[str]:
    """
    Navigates listing pages to dynamically find URLs based on instructions.
    Internal helper for extract_structured_data_from_pages.
    """
    start_url = instructions.get("start_url")
    list_item_selector = instructions.get("list_item_selector") # Selector for the link element itself
    next_page_selector = instructions.get("next_page_selector") # Selector for the 'next page' button/link
    max_pages_to_crawl = instructions.get("max_pages_to_crawl", 5) # Limit pagination

    if not start_url or not list_item_selector:
        raise ToolInputError("Dynamic URL finding requires 'start_url' and 'list_item_selector' in instructions.")

    found_urls = set()
    current_crawl_url = start_url
    visited_crawl_urls = {start_url}

    logger.info(f"Starting dynamic URL discovery from: {start_url}")

    for page_num in range(max_pages_to_crawl):
        logger.info(f"Crawling page {page_num + 1}/{max_pages_to_crawl}: {current_crawl_url}")
        try:
            await browser_navigate(url=current_crawl_url, wait_until="domcontentloaded", capture_snapshot=False)
            await asyncio.sleep(1.5) # Wait for potential JS rendering

            # Extract links matching the item selector on the current page
            link_extraction_script = f"""
            () => {{
                const links = new Set(); // Use a Set to automatically handle duplicates on the page
                document.querySelectorAll('{list_item_selector}').forEach(el => {{
                    if (el.href && el.href.startsWith('http')) {{ // Ensure it's an absolute URL
                        links.add(el.href);
                    }}
                }});
                return Array.from(links);
            }}
            """
            extract_res = await browser_execute_javascript(script=link_extraction_script)
            page_links = extract_res.get("result", []) if extract_res.get("success") and isinstance(extract_res.get("result"), list) else []

            newly_found = 0
            for link in page_links:
                if link not in found_urls:
                    found_urls.add(link)
                    newly_found += 1
                    if len(found_urls) >= max_urls:
                         logger.info(f"Reached maximum URL limit ({max_urls}). Stopping discovery.")
                         return list(found_urls) # Stop early if max limit reached

            logger.info(f"Found {newly_found} new URLs on this page. Total unique URLs: {len(found_urls)}")

            # Find and click the next page element, if specified
            if next_page_selector:
                next_page_element_exists = await browser_execute_javascript(f"() => !!document.querySelector('{next_page_selector}')")
                if next_page_element_exists.get("result"):
                    logger.debug(f"Attempting to click next page selector: {next_page_selector}")
                    click_res = await browser_click(selector=next_page_selector, capture_snapshot=False)
                    if not click_res.get("success"):
                        logger.warning(f"Could not click next page selector ('{next_page_selector}'): {click_res.get('error')}. Stopping crawl.")
                        break # Stop if next click fails

                    await asyncio.sleep(2.5) # Wait for next page to load

                    # Get new URL and check if we've visited it before to detect loops
                    current_url_res = await browser_execute_javascript("() => window.location.href")
                    current_crawl_url = current_url_res.get("result")
                    if not current_crawl_url or current_crawl_url in visited_crawl_urls:
                         logger.warning(f"Next page navigation led to a visited URL ({current_crawl_url}) or failed. Stopping crawl.")
                         break
                    visited_crawl_urls.add(current_crawl_url)
                else:
                    logger.info("Next page selector not found. Assuming end of list.")
                    break # No next page found
            else:
                logger.info("No next_page_selector provided. Stopping crawl after one page.")
                break # Only crawl the first page if no next selector

        except (ToolError, ToolInputError, Exception) as crawl_err:
            logger.error(f"Error during URL discovery on page {page_num + 1} ({current_crawl_url}): {crawl_err}", exc_info=True)
            # Decide whether to stop or continue? Let's stop on error.
            raise ToolError(f"Failed during dynamic URL discovery: {crawl_err}") from crawl_err

    logger.info(f"Dynamic URL discovery finished. Found {len(found_urls)} unique URLs.")
    return list(found_urls)

async def _extract_data_from_single_page(
    url: str,
    extraction_schema_or_prompt: Union[str, Dict],
    llm_model: str,
    page_state_func: callable = _get_simplified_page_state_for_llm # Allow injecting state func if needed
) -> Dict[str, Any]:
    """
    Navigates to a URL, gets content, uses LLM to extract structured data.
    Internal helper for extract_structured_data_from_pages.
    Returns {"data": {...}} on success, {"error": "...", "url": url} on failure.
    """
    logger.debug(f"Extracting data from URL: {url}")
    page_content_context = "Could not retrieve page content." # Default context
    extraction_result = None
    try:
        # 1. Navigate
        nav_res = await browser_navigate(url=url, wait_until="load", timeout=30000, capture_snapshot=False)
        if not nav_res.get("success"):
            raise ToolError(f"Navigation failed: {nav_res.get('error', 'Unknown navigation error')}")
        await asyncio.sleep(1.5) # Wait for dynamic content

        # 2. Get Page State/Content
        page_state = await page_state_func() # Use the provided state function
        if page_state.get("error"):
             logger.warning(f"Using fallback content for {url} due to state extraction error: {page_state['error']}")
             # Try simple body text as a last resort
             body_text_res = await browser_get_text(selector="body")
             page_content_context = body_text_res.get("text", "Failed to get any page content.")[:10000] # Limit fallback context
        else:
             # Construct context from the simplified state
             page_content_context = f"URL: {page_state.get('url')}\nTitle: {page_state.get('title')}\n"
             page_content_context += f"Text Summary:\n{page_state.get('text_summary', '')[:4000]}\n\n" # Limit summary length
             page_content_context += f"Relevant Elements:\n"
             for el in page_state.get('elements', [])[:30]: # Limit elements in context
                 page_content_context += f"- {el.get('id')}: {el.get('tag')} '{el.get('text')}'\n"

        # 3. Prepare LLM Prompt
        system_prompt = "You are an AI assistant that extracts structured data from web page content based on user instructions or a schema. Respond ONLY with the valid JSON object containing the extracted data."
        if isinstance(extraction_schema_or_prompt, dict):
            # Assume it's a JSON schema
            user_prompt = f"Extract data matching the following JSON schema from the provided web page content. Respond ONLY with the valid JSON data object.\n\nSchema:\n```json\n{json.dumps(extraction_schema_or_prompt, indent=2)}\n```\n\nWeb Page Content Context:\n---\n{page_content_context}\n---\n\nExtracted JSON Data:"
        else:
            # Assume it's a natural language prompt
            user_prompt = f"{extraction_schema_or_prompt}\n\nUse the following web page content context to perform the extraction. Respond ONLY with the valid JSON data object.\n\nWeb Page Content Context:\n---\n{page_content_context}\n---\n\nExtracted JSON Data:"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # 4. Call LLM
        extraction_result = await _call_browser_llm(messages, llm_model, f"extracting data from {url}", expected_json=True)

        # 5. Validate Result
        if not extraction_result or extraction_result.get("action") == "error":
            raise ToolError(f"LLM data extraction failed: {extraction_result.get('error', 'No valid JSON returned')}")
        if not isinstance(extraction_result, dict):
             # Should be caught by _call_browser_llm, but double check
             raise ToolError(f"LLM extraction returned non-dictionary result: {type(extraction_result)}")

        # Successfully extracted data
        return {"data": extraction_result, "url": url}

    except Exception as e:
        logger.error(f"Failed to extract data from {url}: {type(e).__name__}: {e}", exc_info=False) # Log less verbosely for individual failures
        return {"error": f"{type(e).__name__}: {str(e)}", "url": url}

def _format_output_data(aggregated_data: List[Dict], output_format: str) -> Union[List[Dict], str]:
    """Formats the aggregated list of extracted data dictionaries."""
    if output_format == "csv_string":
        if not aggregated_data:
            return "" # Empty CSV for no data

        # Use the keys from the first item as headers
        # Ensure consistent order - use keys() which is ordered in Python 3.7+
        headers = list(aggregated_data[0].keys())
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)

        writer.writeheader()
        writer.writerows(aggregated_data)

        return output.getvalue()
    else: # Default to "json_list"
        return aggregated_data
    
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
        
        raise ToolError(error_msg, http_status_code=500) from e


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
        version = browser.version
        
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
            message=f"Failed to initialize browser: {str(e)}",
            http_status_code=500
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
            raise ToolError(f"Could not resolve host: {url}", http_status_code=404) from e
        elif "net::ERR_CONNECTION_REFUSED" in str(e):
            raise ToolError(f"Connection refused: {url}", http_status_code=502) from e
        elif "Timeout" in str(e):
            raise ToolError(f"Navigation timed out after {timeout}ms: {url}", http_status_code=408) from e
        elif "ERR_ABORTED" in str(e):
            raise ToolError(f"Navigation was aborted: {url}", http_status_code=499) from e
        else:
            raise ToolError(error_msg, http_status_code=500) from e

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
                message="Could not navigate back - no previous page in history",
                http_status_code=400
            )
        
        # Get navigation results
        final_url = page.url
        
        # If URLs are the same, navigation didn't actually happen
        if final_url == current_url:
            raise ToolError(
                message="Could not navigate back - no previous page in history",
                http_status_code=400
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
                message="Could not navigate forward - no next page in history",
                http_status_code=400
            )
        
        # Get navigation results
        final_url = page.url
        
        
        # If URLs are the same, navigation didn't actually happen
        if final_url == current_url:
            raise ToolError(
                message="Could not navigate forward - no next page in history",
                http_status_code=400
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
            screenshot_bytes = await page.screenshot(**screenshot_options)
        
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
            "click_count": click_count,
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
                message=f"Timeout while clicking on element: {selector}", # FIX: Use message
                http_status_code=408
            ) from e

        raise ToolError(message=error_msg, http_status_code=500) from e

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
                message=f"Timeout while typing into element: {selector}",
                http_status_code=408
            ) from e

        raise ToolError(message=error_msg, http_status_code=500) from e

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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
        text_content = await page.evaluate("""args => {
            const { selector, trim, includeHidden } = args; // <-- FIX: Destructure single argument
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
        }""", {"selector": selector, "trim": trim, "includeHidden": include_hidden}) # <-- FIX: Pass args as dict
        
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
            logger.info(f"Navigating to download URL: {url}", emoji_key=TaskType.DOWNLOAD.value)
            await page.goto(url)
        elif selector:
            logger.info(f"Clicking element to initiate download: {selector}", emoji_key=TaskType.DOWNLOAD.value)
            await page.click(selector, timeout=timeout) # Use timeout for click as well
        else:
            raise ToolInputError(
                "No URL or selector provided for download",
                param_name="url/selector",
                provided_value={"url": url, "selector": selector}
            )
        
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
            emoji_key=TaskType.DOWNLOAD.value,
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
            emoji_key=TaskType.DOWNLOAD.value,
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
                message=f"Download timed out after {timeout}ms",
                http_status_code=408
            ) from e
        
        raise ToolError(error_msg, http_status_code=500) from e

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
            emoji_key=TaskType.UPLOAD.value,
            selector=selector,
            files=file_paths_list
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
        
        # Correctly pass options using keyword argument expansion
        await page.pdf(**pdf_options)

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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
            message="No browser tabs are open",
            http_status_code=400
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
            message="No browser tabs are open",
            http_status_code=400
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
        
        raise ToolError(error_msg, http_status_code=500) from e

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
                message=f"Script execution timed out after {timeout}ms",
                http_status_code=408
            ) from e
        
        raise ToolError(error_msg, http_status_code=500) from e

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
                message=f"Wait operation timed out after {timeout}ms: {wait_type}={value}",
                http_status_code=408
            ) from e
        
        raise ToolError(error_msg, http_status_code=500) from e
    

# --- High-Level Abstract Tools ---

@with_tool_metrics
@with_error_handling
async def execute_web_workflow(
    instructions: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None, # Concrete values for the workflow
    browser_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes a sequence of browser interactions guided by an LLM to achieve a specific goal,
    configured entirely via the `instructions` dictionary.

    Args:
        instructions: Dictionary defining the workflow:
            start_url: (str) Required. Initial URL.
            workflow_goal_prompt: (str) Required. High-level objective for the LLM.
            available_actions: (List[str]) Required. Explicit list of actions LLM can request
                               (e.g., ["click", "type", "read_value", "scroll", "go_back", "finish_success", "finish_failure"]).
            llm_model: (str) Required. LLM model identifier ('provider/model_name') for guidance.
            max_steps: (Optional[int]) Max interaction steps. Default: 15.
            input_data_mapping: (Optional[Dict[str, str]]) Maps abstract names (e.g., "user_email")
                                used in prompts/actions to keys in the `input_data` parameter.
            element_finding_hints: (Optional[List[str]]) Keywords/descriptions of elements to help LLM.
            success_condition_prompt: (Optional[str]) LLM prompt to evaluate if goal is met (must request JSON {"goal_met": true/false}).
            step_prompts: (Optional[List[Dict]]) Sequence of prompts for multi-stage goals. Each dict: {"step_name": "...", "step_instruction_prompt": "..."}. (Advanced use).
        input_data: (Optional[Dict[str, Any]]) Dictionary containing concrete values (like credentials, form data) referenced indirectly via `input_data_mapping`.
        browser_options: (Optional) Options for browser_init (e.g., {"headless": True}).

    Returns:
        Dictionary summarizing the outcome:
        {
            "success": true/false,
            "final_status": "Goal achieved: ..." | "Max steps reached." | "Failed: ...",
            "steps_taken": int,
            "extracted_data": { "key": "value", ... }, # Data from 'read_value' actions
            "error": Optional[str],
            "message": "Workflow summary message."
        }
    """
    start_time = time.monotonic()

    # --- Validate Instructions & Inputs ---
    if not isinstance(instructions, dict): raise ToolInputError("Instructions must be a dictionary.")
    # Required instruction keys
    start_url = instructions.get("start_url")
    goal_prompt = instructions.get("workflow_goal_prompt")
    available_actions = instructions.get("available_actions")
    llm_model = instructions.get("llm_model") # Get model from instructions
    if not isinstance(start_url, str) or not start_url: raise ToolInputError("instructions['start_url'] (string) is required.")
    if not isinstance(goal_prompt, str) or not goal_prompt: raise ToolInputError("instructions['workflow_goal_prompt'] (string) is required.")
    if not isinstance(available_actions, list) or not available_actions: raise ToolInputError("instructions['available_actions'] (list) is required.")
    if not llm_model or not isinstance(llm_model, str) or '/' not in llm_model: raise ToolInputError("instructions['llm_model'] ('provider/model_name') is required.")

    # Optional instruction keys
    workflow_max_steps = instructions.get("max_steps", 15) # Default max_steps
    if not isinstance(workflow_max_steps, int) or workflow_max_steps < 1: raise ToolInputError("'max_steps' must be a positive integer.")
    input_data_map = instructions.get("input_data_mapping", {})
    if not isinstance(input_data_map, dict): raise ToolInputError("instructions['input_data_mapping'] must be a dictionary.")
    hints = instructions.get("element_finding_hints", [])
    if not isinstance(hints, list): raise ToolInputError("instructions['element_finding_hints'] must be a list.")
    success_condition_prompt = instructions.get("success_condition_prompt")
    if success_condition_prompt and not isinstance(success_condition_prompt, str): raise ToolInputError("instructions['success_condition_prompt'] must be a string if provided.")
    step_prompts = instructions.get("step_prompts") # Advanced feature
    if step_prompts and not isinstance(step_prompts, list): raise ToolInputError("instructions['step_prompts'] must be a list if provided.")

    input_values = input_data if isinstance(input_data, dict) else {}
    logger.info(f"Starting web workflow execution with LLM {llm_model}. Goal: '{goal_prompt[:50]}...'")

    # --- Setup ---
    browser_init_options = browser_options if isinstance(browser_options, dict) else {"headless": True}
    steps_taken = 0
    final_status = "Not started"
    action_history = []
    workflow_results = {} # Store data read by "read_value"
    browser_was_initialized_by_tool = False

    try:
        # 1. Initialize Browser (if needed)
        global _browser_instance
        if not _browser_instance or not _browser_instance.is_connected():
             init_res = await browser_init(**browser_init_options)
             if not init_res.get("success"): raise ToolError(f"Browser init failed: {init_res.get('error')}")
             browser_was_initialized_by_tool = True

        # 2. Navigate to Start URL
        logger.info(f"Navigating to start URL: {start_url}")
        await browser_navigate(url=start_url, wait_until="load", timeout=45000, capture_snapshot=False)
        await asyncio.sleep(2.0) # Wait for page settle

        # --- 3. Main Interaction Loop ---
        # Construct base system prompt incorporating goal and available actions
        system_prompt = f"""You are an AI assistant controlling a web browser to achieve this goal:
Goal: {goal_prompt}
Available Actions JSON format (Respond ONLY with one of these):
"""
        # Dynamically list available actions in the system prompt
        action_examples = {
            "click": '{"action": "click", "params": {"element_id": "el_N"}} /* Click element by ID */',
            "type": '{"action": "type", "params": {"element_id": "el_N", "text_ref": "<input_key>" | "text": "<literal_value>"}} /* Type data */',
            "read_value": '{"action": "read_value", "params": {"element_id": "el_N", "store_as": "<result_key>"}} /* Read element value/text */',
            "scroll": '{"action": "scroll", "params": {"direction": "down" | "up"}}',
            "go_back": '{"action": "go_back", "params": {}}',
            "finish_success": '{"action": "finish_success", "params": {"reason": "<why>", "result_ref": "<optional_result_key>"}}',
            "finish_failure": '{"action": "finish_failure", "params": {"reason": "<why>"}}'
            # Add more examples if needed for other custom actions
        }
        for act in available_actions:
             system_prompt += f"- {action_examples.get(act, act + ' (No example defined)')}\n" # Add examples if available
        system_prompt += "\nAnalyze page state (URL, title, summary, elements with IDs), goal, and history. Choose the SINGLE best JSON action."

        messages = [{"role": "system", "content": system_prompt}]

        for step in range(workflow_max_steps):
            steps_taken = step + 1
            page_id, current_page = await _ensure_page()
            current_url = current_page.url
            logger.info(f"Workflow Step {steps_taken}/{workflow_max_steps}. PageID: {page_id}, URL: {current_url}")
            action_history.append({"step": steps_taken, "url": current_url})

            page_state = await _get_simplified_page_state_for_llm()
            action_history[-1]['page_elements_count'] = len(page_state.get('elements', []))
            if page_state.get("error"):
                final_status = f"Error getting page state: {page_state['error']}. Aborting."; logger.error(final_status); break

            # Construct LLM User Prompt
            prompt_context = f"Current URL: {page_state['url']}\nTitle: {page_state['title']}\nSummary:\n{page_state['text_summary']}\n"
            prompt_context += f"\nOverall Goal: {goal_prompt}\n"
            # Add specific step instruction if using step_prompts (advanced)
            current_step_prompt = ""
            if step_prompts and step < len(step_prompts):
                 current_step_prompt = step_prompts[step].get("step_instruction_prompt", "")
                 if current_step_prompt:
                     prompt_context += f"\nCurrent Step Instruction: {current_step_prompt}\n"

            if workflow_results: prompt_context += f"Data collected so far: {json.dumps(workflow_results)}\n"
            if input_values: prompt_context += f"Available input data keys for 'text_ref': {list(input_values.keys())}\n"
            if hints: prompt_context += f"Element Hints: {hints}\n"

            prompt_context += "\nVisible Elements:\n"
            elements = page_state.get('elements', [])
            if not elements: prompt_context += "- None found.\n"
            else:
                 for el in elements: prompt_context += f"- {el.get('id')}: {el.get('tag')} (Type:{el.get('type')}) Text:'{el.get('text')}'\n"

            prompt_context += "\nWhat is the single best JSON action to take next to progress towards the goal?"
            messages.append({"role": "user", "content": prompt_context})
            if len(messages) > 12: messages = [messages[0]] + messages[-11:] # Limit context history

            # Get LLM Action Decision
            llm_action = await _call_browser_llm(messages, llm_model, f"workflow step {steps_taken} for '{goal_prompt[:30]}...'")
            if not llm_action or llm_action.get("action") == "error":
                final_status = f"LLM guidance failed: {llm_action.get('error', 'No action')}. Aborting."; logger.error(final_status); break
            messages.append({"role": "assistant", "content": json.dumps(llm_action)})

            action_name = llm_action.get("action")
            action_params = llm_action.get("params", {})
            action_history[-1]["llm_action"] = action_name; logger.info(f"LLM action: {action_name} {action_params}")

            # --- Execute Action ---
            try:
                if action_name not in available_actions:
                     raise ValueError(f"LLM chose an unavailable action: '{action_name}'. Allowed: {available_actions}")

                if action_name == "click":
                    element_id = action_params.get("element_id")
                    if not element_id: raise ValueError("Missing 'element_id' for click")
                    locator = await _find_element_locator_for_workflow(current_page, element_id, elements)
                    if not locator: raise ToolError(f"Could not reliably locate element {element_id} chosen by LLM.")
                    logger.info(f"Executing click on element identified as {element_id}")
                    await locator.click(timeout=15000)
                    await asyncio.sleep(3.0)

                elif action_name == "type":
                    element_id = action_params.get("element_id")
                    text_ref = action_params.get("text_ref")
                    literal_text = action_params.get("text")
                    if not element_id: raise ValueError("Missing 'element_id' for type")
                    if not text_ref and literal_text is None: raise ValueError("Missing 'text_ref' or 'text' for type")
                    if text_ref and literal_text is not None: raise ValueError("Provide either 'text_ref' or 'text'")

                    text_to_type = ""
                    if literal_text is not None: text_to_type = str(literal_text)
                    elif text_ref:
                         mapped_key = input_data_map.get(text_ref, text_ref)
                         if mapped_key not in input_values: raise ValueError(f"Input data key '{mapped_key}' (from ref '{text_ref}') not found.")
                         text_to_type = str(input_values[mapped_key])

                    locator = await _find_element_locator_for_workflow(current_page, element_id, elements)
                    if not locator: raise ToolError(f"Could not reliably locate element {element_id} for typing.")
                    logger.info(f"Executing type into element {element_id}")
                    await locator.fill("")
                    await locator.type(text_to_type, delay=50)
                    action_history[-1]['typed_ref'] = text_ref or 'literal'

                elif action_name == "read_value":
                    element_id = action_params.get("element_id")
                    store_as = action_params.get("store_as")
                    if not element_id: raise ValueError("Missing 'element_id' for read_value")
                    if not store_as: raise ValueError("Missing 'store_as' for read_value")

                    locator = await _find_element_locator_for_workflow(current_page, element_id, elements)
                    if not locator: raise ToolError(f"Could not reliably locate element {element_id} for reading.", http_status_code=400)
                    logger.info(f"Executing read_value from element {element_id}, storing as '{store_as}'")
                    value_read = await locator.input_value(timeout=5000) if await locator.evaluate("el => ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)", timeout=5000) else await locator.text_content(timeout=5000)
                    value_read = value_read.strip() if value_read else ""
                    workflow_results[store_as] = value_read
                    action_history[-1]['read_result'] = {store_as: value_read}
                    logger.info(f"Stored '{store_as}': '{value_read[:50]}...'")

                elif action_name == "scroll":
                     direction = action_params.get("direction", "down")
                     scroll_amount = 600 if direction == "down" else -600
                     logger.info(f"Scrolling {direction}"); await browser_execute_javascript(f"window.scrollBy(0, {scroll_amount})"); await asyncio.sleep(1.5)

                elif action_name == "go_back":
                     logger.info("Navigating back"); await browser_back(capture_snapshot=False); await asyncio.sleep(1.5)

                elif action_name == "finish_success":
                    reason = action_params.get('reason', 'Goal achieved')
                    result_ref = action_params.get('result_ref')
                    final_result_value = workflow_results.get(result_ref) if result_ref else workflow_results
                    final_status = f"Goal achieved: {reason}"
                    logger.success(final_status)
                    processing_time = time.monotonic() - start_time
                    return { # Successful exit
                        "success": True, "final_status": final_status, "steps_taken": steps_taken,
                        "extracted_data": final_result_value, "error": None,
                        "message": f"Workflow completed successfully: {reason}",
                        "processing_time": processing_time
                    }

                elif action_name == "finish_failure":
                    reason = action_params.get('reason', 'Goal could not be met')
                    final_status = f"Goal impossible: {reason}"
                    logger.warning(final_status)
                    processing_time = time.monotonic() - start_time
                    return { # Failure exit
                        "success": False, "final_status": final_status, "steps_taken": steps_taken,
                        "extracted_data": workflow_results, "error": reason,
                        "message": f"Workflow failed: {reason}",
                        "processing_time": processing_time
                    }

                else:
                    raise ValueError(f"Internal error: Unsupported action '{action_name}' selected by LLM.")

                # Optional: Add success condition check here if prompt provided
                if success_condition_prompt:
                    # TODO: Implement LLM call to check success condition based on current page_state
                    pass

            except (ToolError, ValueError, Exception) as exec_err:
                 final_status = f"Action '{action_name}' failed: {type(exec_err).__name__}: {exec_err}. Aborting."; logger.error(final_status, exc_info=True)
                 action_history[-1]["error"] = str(exec_err); break

        # Loop finished without explicit success/failure
        if steps_taken == workflow_max_steps: final_status = f"Max steps ({workflow_max_steps}) reached."; logger.warning(final_status)
        else: final_status = final_status if final_status != "Not started" else "Workflow ended unexpectedly."

        processing_time = time.monotonic() - start_time
        return { # Return failure if loop completes without finish action
            "success": False, "final_status": final_status, "steps_taken": steps_taken,
            "extracted_data": workflow_results, "error": final_status,
            "message": f"Workflow stopped after {steps_taken} steps. Status: {final_status}",
            "processing_time": processing_time
        }

    except (ToolInputError, ToolError) as e:
         logger.error(f"Error during web workflow setup or execution: {type(e).__name__}: {e}", exc_info=True)
         final_status = final_status if final_status != "Not started" else f"Setup Failed: {e}"
         return { # Return error structure
             "success": False, "final_status": final_status, "steps_taken": steps_taken,
             "extracted_data": workflow_results, "error": str(e),
             "message": f"Workflow failed: {e}",
             "processing_time": time.monotonic() - start_time
         }
    except Exception as e:
         logger.critical(f"Unexpected critical error during web workflow: {type(e).__name__}: {e}", exc_info=True)
         final_status = f"Critical Error: {type(e).__name__}"
         return {
             "success": False, "final_status": final_status, "steps_taken": steps_taken,
             "extracted_data": workflow_results, "error": f"Unexpected critical error: {str(e)}",
             "message": f"Workflow failed critically: {type(e).__name__}",
             "processing_time": time.monotonic() - start_time
         }
    finally:
        # Close browser ONLY if this tool instance initialized it
        if browser_was_initialized_by_tool:
            logger.info("Closing browser instance initialized by execute_web_workflow tool.")
            await browser_close()


@with_tool_metrics
@with_error_handling
async def extract_structured_data_from_pages(
    instructions: Dict[str, Any],
    # llm_model parameter removed - should be specified in instructions
    browser_options: Optional[Dict[str, Any]] = None,
    max_concurrent_pages: int = 5 # Concurrency limit for processing pages
) -> Dict[str, Any]:
    """
    Visits multiple web pages (provided directly or found dynamically via instructions)
    and extracts structured data from each using LLM guidance based on provided instructions.

    Args:
        instructions: Dictionary controlling the process:
            data_source: (dict) Required. Defines where to get URLs.
                source_type: (str) Required. "list" or "dynamic_crawl".
                urls: (Optional[List[str]]) Required if source_type is "list".
                crawl_config: (Optional[Dict]) Required if source_type is "dynamic_crawl".
                    start_url: (str) Required. URL of the first listing page.
                    list_item_selector: (str) Required. CSS selector for links to target pages.
                    next_page_selector: (Optional[str]) CSS selector for the 'next page' element.
                    max_pages_to_crawl: (Optional[int]) Max pagination steps (default 5).
                    max_urls_limit: (Optional[int]) Stop crawling after finding this many URLs (default 100).
            extraction_details: (dict) Required. Defines how to extract data.
                schema_or_prompt: (Union[str, Dict]) Required. LLM prompt describing fields OR JSON schema. Must instruct LLM to output ONLY JSON.
                extraction_llm_model: (str) Required. LLM model identifier (provider/model) for extraction.
            output_config: (dict) Required. Defines output format.
                format: (Optional[str]) "json_list" (default) or "csv_string".
                error_handling: (Optional[str]) "skip" (default) or "include_error". How to handle page-level errors.
        browser_options: (Optional) Options for browser_init (e.g., {"headless": True}).
        max_concurrent_pages: (Optional) Max number of pages to process in parallel. Default 5.

    Returns:
        Dictionary with results:
        {
            "success": true,
            "extracted_data": List[Dict] or str, # Depends on output_format
            "processed_urls_count": int,
            "successful_extractions_count": int,
            "errors": Dict[str, str], # url -> error message
            "message": "Summary message."
        }

    Raises:
        ToolInputError: For invalid instructions structure or parameters.
        ToolError: For critical errors (e.g., browser init, dynamic URL finding failure).
    """
    start_time = time.monotonic()

    # --- Validate Instructions Structure ---
    if not isinstance(instructions, dict): raise ToolInputError("Instructions must be a dictionary.")
    data_source = instructions.get("data_source")
    extraction_details = instructions.get("extraction_details")
    output_config = instructions.get("output_config")
    if not isinstance(data_source, dict): raise ToolInputError("instructions['data_source'] (dict) is required.")
    if not isinstance(extraction_details, dict): raise ToolInputError("instructions['extraction_details'] (dict) is required.")
    if not isinstance(output_config, dict): raise ToolInputError("instructions['output_config'] (dict) is required.")

    # Validate data_source
    source_type = data_source.get("source_type")
    if source_type not in ["list", "dynamic_crawl"]: raise ToolInputError("data_source['source_type'] must be 'list' or 'dynamic_crawl'.")
    if source_type == "list":
        source_urls = data_source.get("urls")
        if not isinstance(source_urls, list) or not source_urls: raise ToolInputError("data_source['urls'] (non-empty list) is required when source_type is 'list'.")
        if not all(isinstance(u, str) for u in source_urls): raise ToolInputError("All items in data_source['urls'] must be strings.")
    elif source_type == "dynamic_crawl":
        crawl_config = data_source.get("crawl_config")
        if not isinstance(crawl_config, dict): raise ToolInputError("data_source['crawl_config'] (dict) is required when source_type is 'dynamic_crawl'.")
        if not isinstance(crawl_config.get("start_url"), str) or not crawl_config["start_url"]: raise ToolInputError("crawl_config['start_url'] (string) is required.")
        if not isinstance(crawl_config.get("list_item_selector"), str) or not crawl_config["list_item_selector"]: raise ToolInputError("crawl_config['list_item_selector'] (string) is required.")
        # Optional crawl params have defaults in the helper

    # Validate extraction_details
    extraction_schema_or_prompt = extraction_details.get("schema_or_prompt")
    llm_model = extraction_details.get("extraction_llm_model") # Get model from instructions now
    if not extraction_schema_or_prompt: raise ToolInputError("extraction_details['schema_or_prompt'] is required.")
    if not llm_model or not isinstance(llm_model, str) or '/' not in llm_model: raise ToolInputError("extraction_details['extraction_llm_model'] ('provider/model_name') is required.")

    # Validate output_config
    output_format = output_config.get("format", "json_list").lower()
    error_handling = output_config.get("error_handling", "skip").lower()
    if output_format not in ["json_list", "csv_string"]: raise ToolInputError("output_config['format'] must be 'json_list' or 'csv_string'.")
    if error_handling not in ["skip", "include_error"]: raise ToolInputError("output_config['error_handling'] must be 'skip' or 'include_error'.")

    logger.info(f"Starting structured data extraction with LLM: {llm_model}")

    # --- Setup ---
    browser_init_options = browser_options if isinstance(browser_options, dict) else {"headless": True}
    browser_was_initialized_by_tool = False
    all_extracted_data = []
    errors_dict = {}
    target_urls = []

    try:
        # --- Initialize Browser ---
        global _browser_instance
        if not _browser_instance or not _browser_instance.is_connected():
             init_res = await browser_init(**browser_init_options)
             if not init_res.get("success"): raise ToolError(f"Browser init failed: {init_res.get('error')}", http_status_code=500)
             browser_was_initialized_by_tool = True

        # --- 1. Determine Target URLs ---
        if source_type == "list":
            target_urls = source_urls # Already validated
            logger.info(f"Processing {len(target_urls)} URLs provided directly.")
        else: # dynamic_crawl
            # Pass the validated crawl_config to the helper
            max_urls_limit = crawl_config.get("max_urls_limit", 100)
            target_urls = await _find_urls_dynamically(crawl_config, max_urls=max_urls_limit)
            if not target_urls:
                 logger.warning("Dynamic URL finding returned no URLs.")

        # --- 2. Process URLs Concurrently ---
        if not target_urls:
             logger.info("No target URLs to process.")
             # Return success with empty results
             return {
                 "success": True, "extracted_data": [], "processed_urls_count": 0,
                 "successful_extractions_count": 0, "errors": {},
                 "message": "No target URLs found or provided.", "processing_time": time.monotonic() - start_time
             }

        logger.info(f"Extracting data from {len(target_urls)} URLs concurrently (max: {max_concurrent_pages})...")
        semaphore = asyncio.Semaphore(max_concurrent_pages)
        tasks = []

        async def task_wrapper(url):
            async with semaphore:
                # Pass parameters explicitly to the helper
                return await _extract_data_from_single_page(url, extraction_schema_or_prompt, llm_model)

        tasks = [task_wrapper(url) for url in target_urls]
        results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result_or_exc in enumerate(results_from_gather):
            processed_url = target_urls[i]
            if isinstance(result_or_exc, Exception):
                err_msg = f"Task for {processed_url} raised exception: {type(result_or_exc).__name__}: {str(result_or_exc)}"
                logger.error(err_msg, exc_info=False)
                if error_handling == "include_error": errors_dict[processed_url] = str(result_or_exc)
            elif isinstance(result_or_exc, dict):
                if "data" in result_or_exc:
                    # Add source URL to the extracted data for context if needed
                    extracted_item = result_or_exc["data"]
                    extracted_item["_source_url"] = processed_url # Add metadata
                    all_extracted_data.append(extracted_item)
                elif "error" in result_or_exc:
                    err_msg = result_or_exc["error"]
                    logger.warning(f"Extraction failed for {processed_url}: {err_msg}")
                    if error_handling == "include_error": errors_dict[processed_url] = err_msg
                else:
                    err_msg = f"Unexpected result dict format for {processed_url}"
                    logger.error(err_msg)
                    if error_handling == "include_error": errors_dict[processed_url] = "Unknown error structure."
            else:
                err_msg = f"Unexpected result type ({type(result_or_exc)}) for {processed_url}"
                logger.error(err_msg)
                if error_handling == "include_error": errors_dict[processed_url] = "Internal error: Unexpected task result."

        # --- 3. Format Output ---
        formatted_output = _format_output_data(all_extracted_data, output_format)

    except (ToolInputError, ToolError) as e:
        logger.error(f"Error during structured data extraction: {type(e).__name__}: {e}", exc_info=True)
        raise # Re-raise critical setup/validation errors
    except Exception as e:
        logger.critical(f"Unexpected critical error during structured data extraction: {type(e).__name__}: {e}", exc_info=True)
        raise ToolError(f"Unexpected critical error: {e}", http_status_code=500) from e
    finally:
        # Close browser ONLY if this tool initialized it
        if browser_was_initialized_by_tool:
            logger.info("Closing browser instance initialized by extract_structured_data_from_pages tool.")
            await browser_close()

    processing_time = time.monotonic() - start_time
    success_count = len(all_extracted_data)
    error_count = len(errors_dict)
    msg = f"Processed {len(target_urls)} URLs. Successfully extracted data from {success_count}. Encountered {error_count} errors."
    logger.success(msg, time=processing_time)

    return {
        "success": True,
        "extracted_data": formatted_output, # List or CSV string
        "processed_urls_count": len(target_urls),
        "successful_extractions_count": success_count,
        "errors": errors_dict, # url -> error message map
        "message": msg,
        "processing_time": processing_time
    }


@with_tool_metrics
@with_error_handling
async def find_and_download_pdfs(
    topic: str, # The core subject, e.g., "AAPL", "Quantum Computing Research"
    instructions: Dict[str, Any], # Detailed instructions controlling behavior
    output_directory: str,
    llm_model: str = "openai/gpt-4.1-mini",
    max_exploration_steps: int = 20, # Default exploration steps
    browser_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Finds and downloads PDF files related to a topic by navigating websites using LLM guidance.

    This tool is highly configurable via the `instructions` parameter to adapt to
    different search strategies, site structures, PDF identification criteria, and
    download/naming conventions.

    Args:
        topic: The primary subject or entity for the search (e.g., stock ticker, research area).
        instructions: A dictionary defining the workflow:
            search_phase: (dict) Controls the initial search.
                search_query_template: (str) Python f-string template for the search query (e.g., "{topic} investor relations").
                target_site_identification_prompt: (str) LLM prompt to identify the main target site URL from search results. Must request JSON like {"target_url": "..."}.
                search_engine: (str) E.g., "google", "bing". Default: "google".
            exploration_phase: (dict) Controls site exploration.
                exploration_goal_prompt: (str) High-level goal for the LLM during exploration (can use {topic}).
                navigation_keywords: (List[str]) Keywords to prioritize in links/buttons for navigation.
                pdf_keywords: (List[str]) Keywords likely found near relevant PDF links.
                pdf_url_patterns: (List[str]) Regex patterns to identify potential PDF URLs. Default: [r'\\.pdf$'].
                max_steps: (int) Max exploration steps on the target site (overrides main arg if set).
            download_phase: (dict) Controls PDF download and naming.
                metadata_extraction_prompt: (str) LLM prompt to extract date and event type from PDF link context (must use {context}). Must request JSON like {"date": "YYYY-MM-DD", "event_type": "..."}.
                filename_template: (str) Python f-string template for the filename (e.g., "{date}_{topic}_{event_type}"). Requires 'date', 'topic', 'event_type'.
                required_metadata: (List[str]) Metadata keys that *must* be extracted for download (e.g., ["date", "event_type"]). Default: ["date", "event_type"].
        output_directory: Base directory to save downloaded PDFs. A subdirectory for the topic may be created.
        llm_model: (Optional) LLM model identifier for guidance. Default "openai/gpt-4.1-mini".
        max_exploration_steps: (Optional) Overall maximum LLM-guided exploration steps. Default 20.
        browser_options: (Optional) Dictionary of options for browser_init (e.g., {"headless": True}).

    Returns:
        Dictionary summarizing the outcome, found target site, downloaded files, status, etc.
    """
    start_time = time.monotonic()
    logger.info(f"Starting generic PDF search for topic: '{topic}' with model {llm_model}")

    # --- Validate Inputs ---
    if not topic or not isinstance(topic, str): raise ToolInputError("Topic must be non-empty string.", param_name="topic")
    if not output_directory or not isinstance(output_directory, str): raise ToolInputError("Output directory must be non-empty string.", param_name="output_directory")
    if not llm_model or not isinstance(llm_model, str) or '/' not in llm_model: raise ToolInputError("llm_model must be 'provider/model_name'.", param_name="llm_model")
    if not isinstance(instructions, dict): raise ToolInputError("Instructions must be a dictionary.", param_name="instructions")
    # Detailed structure validation
    if not isinstance(instructions.get('search_phase'), dict) or \
       not isinstance(instructions.get('exploration_phase'), dict) or \
       not isinstance(instructions.get('download_phase'), dict):
        raise ToolInputError("Instructions must contain 'search_phase', 'exploration_phase', and 'download_phase' dictionaries.")
    search_phase = instructions['search_phase']
    explore_phase = instructions['exploration_phase']
    download_phase = instructions['download_phase']
    if not isinstance(search_phase.get('search_query_template'), str) or \
       not isinstance(search_phase.get('target_site_identification_prompt'), str):
        raise ToolInputError("search_phase instructions missing 'search_query_template' or 'target_site_identification_prompt' (string).")
    if not isinstance(explore_phase.get('exploration_goal_prompt'), str):
         raise ToolInputError("exploration_phase instructions missing 'exploration_goal_prompt' (string).")
    if not isinstance(download_phase.get('metadata_extraction_prompt'), str) or \
       not isinstance(download_phase.get('filename_template'), str):
        raise ToolInputError("download_phase instructions missing 'metadata_extraction_prompt' or 'filename_template' (string).")
    required_metadata = download_phase.get("required_metadata", ["date", "event_type"])
    if not isinstance(required_metadata, list) or not all(isinstance(x, str) for x in required_metadata):
         raise ToolInputError("download_phase 'required_metadata' must be a list of strings.")

    # --- Setup ---
    topic_sanitized = _sanitize_filename(topic)
    base_output_dir = Path(output_directory).resolve() # Resolve path early
    topic_output_dir = base_output_dir / topic_sanitized
    browser_init_options = browser_options if isinstance(browser_options, dict) else {"headless": True} # Ensure dict or default
    max_steps = explore_phase.get("max_steps", max_exploration_steps) # Allow override

    downloaded_files_list = []
    target_site_url_found = None
    steps_taken = 0
    final_status = "Not started"
    action_history = []
    visited_urls = set()
    browser_was_initialized_by_tool = False

    try:
        # 1. Create Output Directory (using Filesystem tool)
        try:
            # Use await directly on the tool function
            dir_result = await create_directory(path=str(topic_output_dir))
            if not dir_result.get("success"):
                raise ToolError(f"Failed to create output dir '{topic_output_dir}': {dir_result.get('error')}", http_status_code=500)
            logger.info(f"Output directory ensured: {topic_output_dir}")
        except Exception as fs_err:
            # Catch errors from the filesystem tool call itself
            raise ToolError(f"Filesystem tool error creating directory '{topic_output_dir}': {type(fs_err).__name__}: {fs_err}") from fs_err

        # 2. Initialize Browser (if not already running)
        global _browser_instance # Need global to check instance
        if not _browser_instance or not _browser_instance.is_connected():
             init_res = await browser_init(**browser_init_options) # Use low-level tool
             if not init_res.get("success"): raise ToolError(f"Browser init failed: {init_res.get('error')}")
             browser_was_initialized_by_tool = True

        # --- 3. Search Phase ---
        search_engine = search_phase.get("search_engine", "google").lower()
        try:
             search_query = search_phase["search_query_template"].format(topic=topic)
        except KeyError:
             raise ToolInputError("search_query_template must contain {topic} placeholder.")
        logger.info(f"Search Phase: Engine='{search_engine}', Query='{search_query}'")

        search_urls = {"google": "https://www.google.com", "bing": "https://www.bing.com", "duckduckgo": "https://duckduckgo.com"}
        search_selectors = {"google": "textarea[name='q']", "bing": "input[name='q']", "duckduckgo": "input[name='q']"}
        wait_selectors = {"google": "#search", "bing": "#b_results", "duckduckgo": ".react-results--main, .serp__results"} 
        engine_key = search_engine.lower()
        search_url = search_urls.get(engine_key, search_urls["google"])
        search_selector = search_selectors.get(engine_key, search_selectors["google"])
        # wait_selector variable is removed or ignored

        # Use low-level browser tools
        logger.info(f"Navigating to {search_url} for search...")
        await browser_navigate(url=search_url, wait_until="domcontentloaded")
        logger.info(f"Typing search query '{search_query}' into {search_selector}...")
        await browser_type(selector=search_selector, text=search_query, press_enter=True, capture_snapshot=False)
        
        # Wait for initial page load
        await asyncio.sleep(2)
        
        # Dynamically detect search results container instead of using hardcoded selectors
        logger.info("Detecting search results container...")
        dynamic_selector = await _detect_search_results_element()
        if dynamic_selector:
            logger.info(f"Detected search results container: {dynamic_selector}")
            await browser_wait(wait_type="selector", value=dynamic_selector, timeout=5000)
        else:
            # Fallback to hardcoded selectors if dynamic detection fails
            actual_wait_selector = wait_selectors.get(search_engine.lower(), wait_selectors["google"])
            logger.info(f"Dynamic detection failed, using selector: {actual_wait_selector}...")
            try:
                await browser_wait(wait_type="selector", value=actual_wait_selector, timeout=5000)
            except Exception as wait_err:
                logger.warning(f"Failed waiting for selector {actual_wait_selector}: {wait_err}")
                # Continue anyway - we might still be able to extract content
        
        await asyncio.sleep(1.5) # Allow results to potentially render

        # --- 4. Target Site Identification ---
        logger.info("Extracting search results for Target Site ID LLM analysis...")
        page_state_search = await _get_simplified_page_state_for_llm()
        if page_state_search.get("error"): logger.warning(f"Error getting search page state: {page_state_search['error']}")
        # Construct context robustly even if state fetch had issues
        search_results_context = f"Search Page URL: {page_state_search.get('url', 'N/A')}\nTitle: {page_state_search.get('title', 'N/A')}\n"
        search_results_context += f"Text Summary:\n{page_state_search.get('text_summary', 'N/A')}\n\nVisible Elements:\n"
        for el in page_state_search.get('elements', [])[:20]: # Limit context
            search_results_context += f"- ID:{el.get('id')} Tag:{el.get('tag')} Type:{el.get('type')} Text:'{el.get('text')}' Href:{el.get('href')}\n"

        # Prepare prompts using templates from instructions
        system_prompt_target = "You are an AI assistant analyzing search results to find a specific target website URL based on user instructions."
        user_prompt_target = search_phase["target_site_identification_prompt"].format(
            search_term=search_query, search_results_summary=search_results_context
        )
        messages_target = [{"role": "system", "content": system_prompt_target}, {"role": "user", "content": user_prompt_target}]

        # Call LLM helper
        llm_target_result = await _call_browser_llm(messages_target, llm_model, f"finding target site for {topic}")

        # Process LLM response for target URL
        target_site_url_found = llm_target_result.get("target_url") if llm_target_result else None
        if not target_site_url_found or llm_target_result.get("action") == "error" or not isinstance(target_site_url_found, str):
            raise ToolError(f"LLM failed to identify target site URL. Error: {llm_target_result.get('error', 'Invalid or missing target_url in response')}")
        logger.info(f"LLM identified target site URL: {target_site_url_found}")

        # --- 5. Navigate to Target Site ---
        logger.info(f"Navigating to target site: {target_site_url_found}")
        nav_res_target = await browser_navigate(url=target_site_url_found, wait_until="load", timeout=45000)
        if not nav_res_target.get("success"): raise ToolError(f"Failed to navigate to target site {target_site_url_found}: {nav_res_target.get('error')}")
        current_url = nav_res_target.get('url', target_site_url_found) # Get final URL after redirects
        visited_urls.add(current_url)
        await asyncio.sleep(2.5) # Longer wait for complex sites

        # --- 6. Exploration Phase ---
        logger.info(f"Starting Exploration Phase on {current_url}")
        nav_keywords = explore_phase.get('navigation_keywords', [])
        pdf_keywords = explore_phase.get('pdf_keywords', [])
        pdf_url_patterns = explore_phase.get('pdf_url_patterns', [r'\.pdf$']) # Default regex
        pdf_url_regexes = [re.compile(p, re.IGNORECASE) for p in pdf_url_patterns]

        # Construct exploration system prompt dynamically
        system_prompt_explore = explore_phase['exploration_goal_prompt'].format(topic=topic) + "\n\n" + \
            f"Prioritize clicking links/buttons containing keywords: {nav_keywords}\n" + \
            f"Look for PDF download links (element type 'pdf_link' or href matching patterns like {pdf_url_patterns}) near keywords: {pdf_keywords}\n\n" + \
            """Available Actions (Respond ONLY with JSON):
{"action": "click", "params": {"element_id": "el_N"}} /* Click element by ID from list */
{"action": "scroll", "params": {"direction": "down" | "up"}} /* Scroll page */
{"action": "download_pdf", "params": {"url": "<pdf_url_from_page_elements>"}} /* Provide direct PDF URL found */
{"action": "go_back", "params": {}} /* Navigate back */
{"action": "goal_achieved", "params": {"reason": "<brief_reason_for_stopping>"}}
{"action": "goal_impossible", "params": {"reason": "<brief_reason_for_stopping>"}}"""
        messages_explore = [{"role": "system", "content": system_prompt_explore}]

        for step in range(max_steps):
            steps_taken = step + 1
            page_id, current_page_obj = await _ensure_page() # Get current page object
            current_url = current_page_obj.url # Update current URL from actual page state
            logger.info(f"Exploration Step {steps_taken}/{max_steps} for {topic}. PageID: {page_id}, URL: {current_url}")
            action_history.append({"step": steps_taken, "url": current_url})

            page_state = await _get_simplified_page_state_for_llm()
            if page_state.get("error"):
                final_status = f"Error getting page state: {page_state['error']}. Aborting exploration."; logger.error(final_status); break

            # Prepare context, limit size
            prompt_context = f"Current URL: {page_state['url']}\nTitle: {page_state['title']}\nSummary:\n{page_state.get('text_summary', '')[:1500]}\n\nElements:\n"
            elements = page_state.get('elements', [])
            if not elements: 
                prompt_context += "- None found.\n"
            else:
                 # Provide element info clearly
                 for el in elements: prompt_context += f"- {el.get('id')}: {el.get('tag')} (Type: {el.get('type', 'N/A')}) Text:'{el.get('text', '')}'{', Href:' + el.get('href', 'N/A') if el.get('href') else ''}\n"
            prompt_context += "\nBased on goal/instructions, what action JSON next?"

            # Add user message, manage history length
            messages_explore.append({"role": "user", "content": prompt_context})
            if len(messages_explore) > 12: messages_explore = [messages_explore[0]] + messages_explore[-11:] # Keep system + last 5 pairs

            # Get LLM guidance
            llm_action = await _call_browser_llm(messages_explore, llm_model, f"exploring {topic} (step {steps_taken})")
            if not llm_action or llm_action.get("action") == "error":
                final_status = f"LLM guidance failed: {llm_action.get('error', 'No action')}. Aborting."; logger.error(final_status); break
            messages_explore.append({"role": "assistant", "content": json.dumps(llm_action)}) # Add LLM response

            action_name = llm_action.get("action")
            action_params = llm_action.get("params", {})
            action_history[-1]["llm_action"] = action_name; logger.info(f"LLM action: {action_name} {action_params}")

            # --- Execute Action ---
            try:
                if action_name == "click":
                    element_id = action_params.get("element_id")
                    if not element_id: raise ValueError("Missing 'element_id'")
                    element_to_click = next((el for el in elements if el.get("id") == element_id), None)
                    if not element_to_click: raise ToolError(f"LLM chose non-existent element ID '{element_id}'")

                    # Attempt robust click using Playwright's locators based on text/attributes
                    click_selector = ""
                    element_text = element_to_click.get('text')
                    if element_text and element_text != '<no text>':
                         # Try clicking by text first
                         try:
                             logger.info(f"Attempting click (ID: {element_id}) using text: '{element_text}'")
                             # Use get_by_text with exact match option? Or default fuzzy? Default is often better.
                             await current_page_obj.get_by_text(element_text, exact=False).first.click(timeout=10000)
                             await asyncio.sleep(3) # Wait after click
                         except Exception as text_click_err:
                              logger.warning(f"Click by text failed for '{element_text}': {text_click_err}. Falling back to ID/selector guess.")
                              # Fallback selector (less reliable)
                              click_selector = f"#{element_id}" # Or the nth-of-type guess from previous version
                              await browser_click(selector=click_selector, capture_snapshot=False)
                              await asyncio.sleep(3)
                    else:
                         # If no text, fallback selector is needed
                         raise ToolError(f"Cannot click element ID '{element_id}' - no reliable text found for selector.")

                    # Update current URL after potential navigation
                    current_url_res = await browser_execute_javascript("() => window.location.href")
                    new_url = current_url_res.get("result", current_page_obj.url)
                    if new_url != current_url:
                        if new_url in visited_urls: logger.warning(f"Loop detected: Visited {new_url} again.")
                        else: current_url = new_url; visited_urls.add(current_url)

                elif action_name == "scroll":
                    direction = action_params.get("direction", "down")
                    scroll_amount = 600 if direction == "down" else -600
                    logger.info(f"Scrolling {direction}"); await browser_execute_javascript(f"window.scrollBy(0, {scroll_amount})"); await asyncio.sleep(1.5)

                elif action_name == "download_pdf":
                    pdf_url = action_params.get("url")
                    if not pdf_url or not isinstance(pdf_url, str): raise ValueError("'url' missing or invalid for download_pdf")

                    # Ensure URL seems valid and matches patterns if provided
                    if not any(regex.search(pdf_url) for regex in pdf_url_regexes):
                         logger.warning(f"LLM requested download of URL '{pdf_url}' which doesn't match expected PDF patterns {pdf_url_patterns}. Attempting anyway.")
                         # Could potentially skip download here if pattern match is strict requirement

                    logger.info(f"Asking LLM for metadata for PDF: {pdf_url}")
                    metadata_context = f"PDF URL: {pdf_url}\nFound on page: {current_url} ('{page_state['title']}')\nContext near link (if available from history):\n{messages_explore[-2]['content'][-1500:]}"
                    system_prompt_meta = "Extract metadata (date YYYY-MM-DD, event_type string) for PDF based on context. Respond ONLY with JSON: {\"date\": \"...\", \"event_type\": \"...\"} or defaults."
                    user_prompt_meta = download_phase["metadata_extraction_prompt"].format(context=metadata_context)
                    messages_meta = [{"role": "system", "content": system_prompt_meta}, {"role": "user", "content": user_prompt_meta}]
                    llm_meta_result = await _call_browser_llm(messages_meta, llm_model, f"extracting metadata for {topic}")

                    pdf_date_str = datetime.now().strftime('%Y-%m-%d')
                    pdf_event = "Unknown_Event"
                    # Check if metadata extraction was successful before using results
                    if llm_meta_result and llm_meta_result.get("action") != "error":
                         pdf_date_str_llm = llm_meta_result.get("date")
                         pdf_event_llm = llm_meta_result.get("event_type")
                         # Basic validation of returned values
                         if isinstance(pdf_date_str_llm, str) and len(pdf_date_str_llm) > 5: pdf_date_str = pdf_date_str_llm # Basic length check
                         if isinstance(pdf_event_llm, str) and pdf_event_llm: pdf_event = pdf_event_llm # Ensure not empty

                    try: pdf_date_iso = datetime.strptime(pdf_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                    except ValueError: pdf_date_iso = datetime.now().strftime('%Y-%m-%d'); logger.warning(f"Invalid date '{pdf_date_str}', using {pdf_date_iso}")

                    # Construct filename using template from instructions
                    try:
                         filename_base = _sanitize_filename(
                             download_phase["filename_template"].format(date=pdf_date_iso, topic=topic_sanitized, event_type=_sanitize_filename(pdf_event))
                         )
                         filename = f"{filename_base}.pdf"
                    except KeyError as fmt_err:
                         raise ToolInputError(f"Filename template needs keys: {fmt_err}. Available: date, topic, event_type.")

                    logger.info(f"Attempting download: '{pdf_event}' ({pdf_date_iso}) from {pdf_url} as {filename}")
                    dl_result = await browser_download_file(url=pdf_url, save_path=str(topic_output_dir), filename=filename, overwrite=False, timeout=120000) # Longer timeout for downloads
                    if not dl_result.get("success"): raise ToolError(f"Download failed: {dl_result.get('error')}")

                    final_filename = dl_result.get("file_name")
                    logger.info(f"Download successful: {final_filename}")
                    downloaded_files_list.append({"filename": final_filename, "url": pdf_url, "event_type": pdf_event, "date": pdf_date_iso})
                    action_history[-1]["downloaded"] = final_filename

                elif action_name == "go_back":
                     logger.info("Navigating back"); back_res = await browser_back(capture_snapshot=False)
                     if not back_res.get("success"): raise ToolError(f"Go back failed: {back_res.get('error')}")
                     current_url = back_res.get('url', current_page_obj.url); await asyncio.sleep(1.5)

                elif action_name == "goal_achieved" or action_name == "goal_impossible":
                    final_status = f"{action_name.replace('_', ' ').title()}: {action_params.get('reason', 'N/A')}"
                    logger.info(f"LLM indicated exploration end: {final_status}"); break
                else:
                    final_status = f"Unknown action '{action_name}' from LLM. Aborting."; logger.error(final_status); break

            except (ToolError, ValueError, Exception) as exec_err:
                 final_status = f"Action '{action_name}' failed: {type(exec_err).__name__}: {exec_err}. Aborting."; logger.error(final_status, exc_info=True)
                 action_history[-1]["error"] = str(exec_err); break # Exit loop on error

        # End of loop check
        if steps_taken == max_steps: final_status = f"Max exploration steps ({max_steps}) reached."; logger.warning(final_status)

    except (ToolError, ToolInputError, Exception) as e:
         logger.error(f"Error during generic PDF search for '{topic}': {type(e).__name__}: {e}", exc_info=True)
         final_status = final_status if final_status != "Not started" else f"Failed with error: {type(e).__name__}"
         # Ensure we return the error structure
         return {
             "success": False, "topic": topic, "error": f"{type(e).__name__}: {str(e)}",
             "target_site_url": target_site_url_found, "output_directory": str(topic_output_dir),
             "downloaded_files": downloaded_files_list, "steps_taken": steps_taken,
             "final_status": final_status, "message": f"Failed PDF search for '{topic}'."
         }
    finally:
        # Close browser ONLY if this tool initialized it
        if browser_was_initialized_by_tool:
            logger.info("Closing browser instance initialized by find_and_download_pdfs tool.")
            await browser_close() # Use the tool to ensure proper cleanup

    processing_time = time.monotonic() - start_time
    logger.info(f"Finished PDF search for '{topic}' in {processing_time:.2f}s. Found {len(downloaded_files_list)} PDFs.")

    return {
        "success": True, "topic": topic, "output_directory": str(topic_output_dir),
        "target_site_url": target_site_url_found, "downloaded_files": downloaded_files_list,
        "steps_taken": steps_taken, "final_status": final_status or "Reached max steps",
        "message": f"Successfully downloaded {len(downloaded_files_list)} PDF(s) for '{topic}'."
    }


@with_tool_metrics
@with_error_handling
async def multi_engine_search_summary(
    query: str,
    instructions: Dict[str, Any], # NEW: Replaces engines, num_results_per_engine
    llm_model: str = "openai/gpt-4.1-mini", # Model for summarization
    browser_options: Optional[Dict[str, Any]] = None,
    max_concurrent_summaries: int = 3 # Limit concurrent page processing
) -> Dict[str, Any]:
    """
    Performs a search query on multiple search engines (defined in instructions),
    extracts top results, visits selected pages, and generates an LLM summary
    for each based on a prompt defined in instructions.

    Args:
        query: The search query string.
        instructions: Dictionary controlling the search and summarization:
            search_params: (dict) Contains search engine config.
                engines: (List[str]) List of engines ("google", "bing", "duckduckgo").
                num_results_per_engine: (int) Number of results to fetch per engine.
            summarization_prompt: (str) LLM prompt template for summarizing. Must include placeholders for {query} and {page_content}.
            url_filter_keywords: (Optional[List[str]]) Keywords to require in URL/title/snippet before summarizing.
            min_content_length_for_summary: (Optional[int]) Minimum character length of extracted text to attempt summarization.
        llm_model: (Optional) LLM model identifier for summarization. Default "openai/gpt-4.1-mini".
        browser_options: (Optional) Dictionary of options for browser_init.
        max_concurrent_summaries: (Optional) Max number of URL summaries to process in parallel. Default 3.


    Returns:
        Dictionary containing summarized results and any errors encountered.
    """
    start_time = time.monotonic()
    logger.info(f"Starting multi-engine search summary for query: '{query}' using instructions.")

    # --- Validate Inputs ---
    if not query or not isinstance(query, str): raise ToolInputError("Query must be non-empty string.")
    if not isinstance(instructions, dict): raise ToolInputError("Instructions must be a dictionary.")
    if not llm_model or not isinstance(llm_model, str) or '/' not in llm_model: raise ToolInputError("llm_model must be 'provider/model_name'.")

    # Validate instructions structure
    search_params = instructions.get("search_params")
    summarization_prompt_template = instructions.get("summarization_prompt")
    url_filter_keywords = instructions.get("url_filter_keywords", [])
    min_content_len = instructions.get("min_content_length_for_summary", 100) # Default min length

    if not isinstance(search_params, dict): raise ToolInputError("instructions['search_params'] dictionary is required.")
    engines = search_params.get("engines")
    num_results_per_engine = search_params.get("num_results_per_engine")
    if not isinstance(engines, list) or not engines: raise ToolInputError("instructions['search_params']['engines'] (list) is required.")
    if not isinstance(num_results_per_engine, int) or num_results_per_engine < 1: raise ToolInputError("instructions['search_params']['num_results_per_engine'] (positive int) is required.")
    valid_engines = {"google", "bing", "duckduckgo"}; invalid_engines = [e for e in engines if e.lower() not in valid_engines]
    if invalid_engines: raise ToolInputError(f"Invalid engine(s): {invalid_engines}. Supported: {valid_engines}")
    if not isinstance(summarization_prompt_template, str) or "{query}" not in summarization_prompt_template or "{page_content}" not in summarization_prompt_template:
        raise ToolInputError("instructions['summarization_prompt'] (string) with {{query}} and {{page_content}} placeholders is required.")
    if not isinstance(url_filter_keywords, list): raise ToolInputError("instructions['url_filter_keywords'] must be a list of strings.")
    if not isinstance(min_content_len, int) or min_content_len < 0: raise ToolInputError("instructions['min_content_length_for_summary'] must be a non-negative integer.")

    # --- Setup ---
    browser_init_options = browser_options if isinstance(browser_options, dict) else {"headless": True}
    all_results_map: Dict[str, Dict[str, Any]] = {} # url -> {title, snippet, source_engines}
    errors_dict: Dict[str, str] = {} # engine/url -> error message
    browser_was_initialized_by_tool = False
    summarized_results = []

    try:
        # --- Initialize Browser ---
        global _browser_instance
        if not _browser_instance or not _browser_instance.is_connected():
             init_res = await browser_init(**browser_init_options)
             if not init_res.get("success"): raise ToolError(f"Browser init failed: {init_res.get('error')}")
             browser_was_initialized_by_tool = True

        # --- 1. Search on Each Engine ---
        # (Use the _perform_web_search helper function defined previously)
        seen_urls = set()
        for engine in engines:
            engine_name = engine.lower()
            try:
                 results = await _perform_web_search(query, engine=engine_name, num_results=num_results_per_engine)
                 for res in results:
                     url = res['url']
                     if url not in seen_urls:
                         all_results_map[url] = {"title": res['title'], "snippet": res['snippet'], "source_engines": [engine_name]}
                         seen_urls.add(url)
                     elif engine_name not in all_results_map[url]["source_engines"]:
                         all_results_map[url]["source_engines"].append(engine_name)
                         if len(res.get('title','')) > len(all_results_map[url].get('title','')): all_results_map[url]['title'] = res['title']
                         if len(res.get('snippet','')) > len(all_results_map[url].get('snippet','')): all_results_map[url]['snippet'] = res['snippet']
            except Exception as search_err:
                 logger.error(f"Search failed for {engine_name}: {search_err}", exc_info=True)
                 errors_dict[engine_name] = str(search_err)

        # --- 2. Filter URLs before Visiting ---
        urls_to_process = list(all_results_map.keys())
        if url_filter_keywords:
            urls_to_summarize = []
            keywords_lower = [k.lower() for k in url_filter_keywords]
            for url in urls_to_process:
                res_info = all_results_map[url]
                # Check keywords in URL, title, or snippet
                text_to_check = f"{url} {res_info.get('title','')} {res_info.get('snippet','')}".lower()
                if any(keyword in text_to_check for keyword in keywords_lower):
                    urls_to_summarize.append(url)
                else:
                     logger.debug(f"Skipping URL (filter keywords not found): {url}")
            logger.info(f"Filtered {len(urls_to_process)} URLs down to {len(urls_to_summarize)} based on keywords: {url_filter_keywords}")
        else:
            urls_to_summarize = urls_to_process

        if not urls_to_summarize:
             logger.warning("No URLs remaining after filtering to summarize.")
        else:
            logger.info(f"Attempting to visit and summarize {len(urls_to_summarize)} URLs concurrently (max: {max_concurrent_summaries})...")

            # --- 3. Visit and Summarize Filtered URLs Concurrently ---
            semaphore = asyncio.Semaphore(max_concurrent_summaries)
            tasks = []
            url_map_for_gather = {}
            task_counter = 0

            # --- Define the concurrent task using a nested helper ---
            async def summarize_task_wrapper(url):
                async with semaphore:
                    # This nested helper now uses the summarization prompt from instructions
                    logger.debug(f"Summarizing URL: {url}")
                    summary = "Summary generation failed."
                    error = None
                    page_text = "Content extraction failed."
                    try:
                        nav_res = await browser_navigate(url=url, wait_until="load", timeout=30000, capture_snapshot=False)
                        if not nav_res.get("success"): raise ToolError(f"Navigation failed: {nav_res.get('error')}")
                        await asyncio.sleep(1.5)

                        text_res = await browser_get_text(selector="body")
                        if text_res.get("success"):
                            page_text = (text_res.get("text") or "")
                        else:
                            logger.warning(f"browser_get_text failed for {url}: {text_res.get('error')}, using snippet.")
                            page_text = all_results_map[url].get("snippet", "Content extraction failed.")

                        if not page_text.strip() or len(page_text) < min_content_len:
                             logger.info(f"Skipping summary for {url}: Content too short ({len(page_text)} chars) or empty.")
                             summary = f"Skipped (content length {len(page_text)} < {min_content_len})"
                             # Not strictly an error, but not a generated summary
                        else:
                             page_text = page_text[:10000] # Limit context passed to LLM

                             system_prompt_sum = "You are an AI assistant that concisely summarizes web page text relevant to a user query."
                             # Use the prompt *template* from instructions
                             user_prompt_sum = summarization_prompt_template.format(query=query, page_content=page_text)
                             messages_sum = [{"role": "system", "content": system_prompt_sum}, {"role": "user", "content": user_prompt_sum}]

                             summary_res = await _call_browser_llm(messages_sum, llm_model, f"summarizing {url}", expected_json=False)

                             if summary_res and summary_res.get("action") != "error" and isinstance(summary_res.get("text"), str):
                                 summary = summary_res["text"].strip() or "Summary generation returned empty."
                             else:
                                 error_detail = summary_res.get('error', 'LLM did not return text.') if summary_res else 'LLM call failed.'
                                 summary = f"Summary generation failed: {error_detail}"
                                 error = summary

                    except Exception as page_err:
                        logger.error(f"Failed processing page {url} for summary: {type(page_err).__name__}: {page_err}", exc_info=False)
                        error = f"{type(page_err).__name__}: {str(page_err)}"
                        summary = f"Failed to process page: {error}"

                    return { # Return structure for the gather processing loop
                        "url": url,
                        "title": all_results_map[url].get("title", "N/A"),
                        "summary": summary,
                        "source_engine": all_results_map[url].get("source_engines", []),
                        "error": error
                    }
            # --- End of nested helper ---

            # Create tasks
            for url in urls_to_summarize:
                tasks.append(asyncio.create_task(summarize_task_wrapper(url)))
                url_map_for_gather[task_counter] = url
                task_counter += 1

            # Gather results
            summary_results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

            # Process gather results
            for i, result_or_exc in enumerate(summary_results_from_gather):
                processed_url = url_map_for_gather[i]
                if isinstance(result_or_exc, Exception):
                    err_msg = f"Summary task for {processed_url} failed: {type(result_or_exc).__name__}: {str(result_or_exc)}"
                    logger.error(err_msg, exc_info=False)
                    errors_dict[processed_url] = str(result_or_exc)
                    summarized_results.append({"url": processed_url, "title": all_results_map[processed_url].get("title", "N/A"), "summary": f"Failed: {type(result_or_exc).__name__}", "source_engine": all_results_map[processed_url].get("source_engines", []), "error": str(result_or_exc)})
                elif isinstance(result_or_exc, dict):
                    if result_or_exc.get("error"): errors_dict[processed_url] = result_or_exc["error"] # Capture summary-specific errors
                    summarized_results.append(result_or_exc)
                else:
                    err_msg = f"Unexpected result type ({type(result_or_exc)}) for {processed_url}"; logger.error(err_msg); errors_dict[processed_url] = err_msg

    except (ToolInputError, ToolError) as e:
         logger.error(f"Error during multi-engine search: {type(e).__name__}: {e}", exc_info=True)
         return {"success": False, "query": query, "results": summarized_results, "errors": errors_dict, "message": f"Search failed: {str(e)}"}
    except Exception as e:
         logger.critical(f"Unexpected critical error during multi-engine search: {type(e).__name__}: {e}", exc_info=True)
         return {"success": False, "query": query, "results": summarized_results, "errors": errors_dict, "message": f"Search failed critically: {type(e).__name__}: {e}"}
    finally:
        # Close browser ONLY if this tool initialized it
        if browser_was_initialized_by_tool:
            logger.info("Closing browser instance initialized by multi_engine_search_summary tool.")
            await browser_close()

    processing_time = time.monotonic() - start_time
    successful_summaries = len([r for r in summarized_results if not r.get("error")])
    msg = f"Summarized {successful_summaries} of {len(urls_to_summarize)} targeted results for query '{query}'."
    if errors_dict: msg += f" Encountered {len(errors_dict)} errors."
    logger.success(msg, time=processing_time)

    return {
        "success": True, "query": query, "results": summarized_results,
        "errors": errors_dict, "message": msg, "processing_time": processing_time
    }


@with_tool_metrics
@with_error_handling
async def monitor_web_data_points(
    instructions: Dict[str, Any],
    previous_values: Optional[Dict[str, str]] = None # Key format: "{url}::{data_point_name}"
    # Removed llm_model, browser_options, max_concurrent_pages from signature
) -> Dict[str, Any]:
    """
    Monitors specific data points across multiple web pages (defined in instructions),
    extracts values using specified methods (selector or LLM), and checks conditions
    or changes based on criteria defined in the instructions.

    Args:
        instructions: Dictionary controlling the monitoring process:
            monitoring_targets: (List[Dict]) Required. List of targets to monitor.
                Each target dict requires:
                url: (str) The URL of the page to visit.
                data_points: (List[Dict]) List of data points to check on this page.
                    Each data_point dict requires:
                    name: (str) Unique name for this data point.
                    identifier: (str) CSS selector OR natural language prompt for LLM.
                    extraction_method: (str) 'selector' or 'llm'.
                    attribute: (Optional[str]) HTML attribute for 'selector' method.
                    condition: (Optional[str]) Check type: 'changed', 'equals', 'contains',
                               'greater_than', 'less_than', 'ge', 'le', 'regex_match',
                               'llm_eval'. Default: 'changed'.
                    condition_value: (Optional[Any]) Value/pattern for comparison.
                    llm_condition_prompt: (Optional[str]) Prompt for 'llm_eval' condition.
            llm_config: (dict) Required. Defines LLM usage.
                model: (str) Required. LLM model identifier ('provider/model_name') for 'llm' extraction or 'llm_eval'.
            concurrency: (Optional[dict]) Defines parallel execution settings.
                max_concurrent_pages: (Optional[int]) Max URLs to process in parallel. Default 3.
            browser_options: (Optional[dict]) Options for browser_init (e.g., {"headless": True}). Default {"headless": True}.
        previous_values: (Optional[Dict[str, str]]) Dictionary holding the *last known* values
                         for data points, keyed as "{url}::{data_point_name}". Used for 'changed' condition.

    Returns:
        Dictionary with monitoring results:
        {
            "success": true,
            "results": { # Keyed by URL
                "https://example.com": {
                    "data_point_1_name": {
                        "current_value": "...", "condition_met": bool|None, "status": "...", "error": str|None
                    }, ...
                }, ...
            },
            "errors": List[str], # Page-level processing errors
            "message": "Monitoring complete summary."
        }
    """
    start_time = time.monotonic()

    # --- Validate Instructions Structure ---
    if not isinstance(instructions, dict): raise ToolInputError("Instructions must be a dictionary.")
    monitoring_targets = instructions.get("monitoring_targets")
    llm_config = instructions.get("llm_config")
    concurrency_config = instructions.get("concurrency", {}) # Optional
    browser_opts_from_instr = instructions.get("browser_options") # Optional

    if not isinstance(monitoring_targets, list) or not monitoring_targets: raise ToolInputError("instructions['monitoring_targets'] (non-empty list) is required.")
    if not isinstance(llm_config, dict) or not llm_config.get("model"): raise ToolInputError("instructions['llm_config'] (dict) with 'model' ('provider/model_name') is required.")
    llm_model = llm_config["model"] # Extract LLM model from instructions
    if not isinstance(concurrency_config, dict): raise ToolInputError("instructions['concurrency'] must be a dictionary if provided.")
    if browser_opts_from_instr and not isinstance(browser_opts_from_instr, dict): raise ToolInputError("instructions['browser_options'] must be a dictionary if provided.")
    # Further validation of monitoring_targets structure could be added if needed

    logger.info(f"Starting web data point monitoring for {len(monitoring_targets)} targets using LLM {llm_model}.")

    # --- Setup ---
    browser_options = browser_opts_from_instr if isinstance(browser_opts_from_instr, dict) else {"headless": True} # Use provided or default
    max_concurrent_pages = concurrency_config.get("max_concurrent_pages", 3) # Get concurrency from instructions or default
    if not isinstance(max_concurrent_pages, int) or max_concurrent_pages < 1: raise ToolInputError("concurrency['max_concurrent_pages'] must be a positive integer.")

    browser_was_initialized_by_tool = False
    previous_values_map = previous_values or {}
    monitoring_results = {} # url -> { data_point_name -> result }
    page_level_errors = []

    try:
        # --- Initialize Browser (if needed) ---
        global _browser_instance
        if not _browser_instance or not _browser_instance.is_connected():
             init_res = await browser_init(**browser_options)
             if not init_res.get("success"): raise ToolError(f"Browser init failed: {init_res.get('error')}")
             browser_was_initialized_by_tool = True

        # --- Process Targets Concurrently ---
        semaphore = asyncio.Semaphore(max_concurrent_pages)
        tasks = []

        async def process_target_url(target_def: Dict):
            """Task to process one target URL and its data points."""
            url = target_def.get("url")
            data_points = target_def.get("data_points", [])
            # Basic validation of the target structure
            if not url or not isinstance(url, str) or not isinstance(data_points, list) or not data_points:
                error_msg = f"Invalid target definition skipped (missing/invalid url or data_points): {str(target_def)[:150]}..."
                logger.error(error_msg)
                page_level_errors.append(error_msg)
                return None # Skip invalid target

            target_results = {}
            page_error_msg = None
            page_obj_for_target: Optional[Page] = None # Store page for reuse across data points

            try:
                # --- Navigate (once per URL) ---
                try:
                    page_id, page_obj_for_target = await _ensure_page() # Get a page
                    logger.debug(f"Navigating to target URL for monitoring: {url} (using page {page_id})")
                    nav_res = await browser_navigate(url=url, wait_until="load", timeout=30000, capture_snapshot=False)
                    if not nav_res.get("success"): raise ToolError(f"Navigation failed: {nav_res.get('error')}")
                    await asyncio.sleep(1.5) # Wait for JS
                except (ToolError, ToolInputError, Exception) as nav_err:
                    page_error_msg = f"Failed to navigate to {url}: {type(nav_err).__name__}: {str(nav_err)}"
                    logger.error(page_error_msg)
                    page_level_errors.append(page_error_msg)
                    return {url: {"page_error": page_error_msg}}

                # --- Extract and Evaluate Data Points (iteratively on the same page) ---
                for dp_def in data_points:
                    dp_name = dp_def.get("name")
                    if not dp_name or not isinstance(dp_name, str):
                         logger.warning(f"Data point definition missing/invalid 'name' in target {url}, skipping: {dp_def}")
                         # Store error for this specific data point within the URL's results
                         target_results[f"invalid_dp_{uuid.uuid4().hex[:6]}"] = {"error": "Invalid data point definition (missing name)"}
                         continue

                    dp_result = {"current_value": None, "condition_met": None, "status": "Not evaluated", "error": None}

                    # Extract value using helper
                    current_value, extract_err = await _extract_single_data_point(page_obj_for_target, url, dp_def, llm_model)
                    dp_result["current_value"] = current_value
                    if extract_err:
                        dp_result["error"] = f"Extraction failed: {extract_err}"
                        dp_result["status"] = "Extraction failed"
                        logger.warning(f"Extraction failed for '{dp_name}' on {url}: {extract_err}")
                    else:
                        # Evaluate condition if extraction succeeded
                        condition_def = {k: v for k, v in dp_def.items() if k in ['condition', 'condition_value', 'llm_condition_prompt']}
                        previous_val_key = f"{url}::{dp_name}" # Consistent key format
                        prev_val = previous_values_map.get(previous_val_key)
                        cond_met, cond_status, eval_err = await _evaluate_data_point_condition(dp_name, current_value, condition_def, prev_val, llm_model)
                        dp_result["condition_met"] = cond_met
                        dp_result["status"] = cond_status
                        if eval_err:
                            dp_result["error"] = f"Evaluation failed: {eval_err}"
                            logger.warning(f"Evaluation failed for '{dp_name}' on {url}: {eval_err}")

                    target_results[dp_name] = dp_result

                return {url: target_results} # Return results for this URL

            except Exception as target_proc_err:
                 err_msg = f"Unexpected error processing target {url}: {type(target_proc_err).__name__}: {str(target_proc_err)}"
                 logger.error(err_msg, exc_info=True)
                 page_level_errors.append(err_msg)
                 # Return page-level error associated with the URL
                 return {url: {"page_error": err_msg}}

        # --- Create and Gather Tasks ---
        async def task_wrapper(target_def):
            async with semaphore:
                return await process_target_url(target_def)

        tasks = [task_wrapper(target) for target in monitoring_targets]
        results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

        # Consolidate results
        for res_or_exc in results_from_gather:
            if isinstance(res_or_exc, Exception):
                err_msg = f"Monitoring task failed unexpectedly: {type(res_or_exc).__name__}: {str(res_or_exc)}"
                logger.error(err_msg, exc_info=False) # Less verbose for gather errors
                page_level_errors.append(err_msg)
            elif isinstance(res_or_exc, dict):
                monitoring_results.update(res_or_exc) # Merge results dicts {url: {dp_name: ...}}
            else:
                 err_msg = f"Unexpected result type ({type(res_or_exc)}) from monitoring task"
                 logger.error(err_msg)
                 page_level_errors.append(err_msg)

    except (ToolInputError, ToolError) as e:
         logger.error(f"Error during monitoring setup or execution: {type(e).__name__}: {e}", exc_info=True)
         # Add error to page_level_errors for reporting? Or re-raise? Let's re-raise setup errors.
         raise
    except Exception as e:
         logger.critical(f"Unexpected critical error during monitoring execution: {type(e).__name__}: {e}", exc_info=True)
         raise ToolError(f"Unexpected critical error: {e}") from e
    finally:
        # Close browser ONLY if this tool initialized it
        if browser_was_initialized_by_tool:
            logger.info("Closing browser instance initialized by monitor_web_data_points tool.")
            await browser_close()

    processing_time = time.monotonic() - start_time
    processed_count = len(monitoring_results)
    error_count = len(page_level_errors)
    msg = f"Monitoring complete. Processed {processed_count} of {len(monitoring_targets)} targets."
    if error_count > 0: msg += f" Encountered {error_count} page-level errors during processing."
    logger.success(msg, time=processing_time)

    return {
        "success": True,
        "results": monitoring_results, # {url: {dp_name: {current_value, condition_met, status, error}}}
        "errors": page_level_errors, # List of page-level errors encountered during navigation/setup
        "message": msg,
        "processing_time": processing_time
    }


@with_tool_metrics
@with_error_handling
async def research_and_synthesize_report(
    topic: str,
    instructions: Dict[str, Any],
    browser_options: Optional[Dict[str, Any]] = None, # Keep top-level browser_options for convenience
    max_concurrent_extractions: int = 3 # Keep concurrency control at top-level
) -> Dict[str, Any]:
    """
    Performs web research on a topic using specified search queries, selects relevant pages via LLM,
    extracts key information from those pages using LLM guidance, and synthesizes the findings
    into a report according to detailed instructions.

    Args:
        topic: The research topic.
        instructions: Dictionary controlling the workflow:
            research_goal_prompt: (str) Required. Overall objective of the research.
            search_phase: (dict) Required. Defines search parameters.
                search_queries: (List[str]) Required. List of search query templates (use "{topic}").
                search_engine: (Optional[str]) "google", "bing", etc. Default "google".
                num_search_results_per_query: (Optional[int]) Initial results to fetch per query. Default 10.
            site_selection_phase: (dict) Required. Defines how relevant sites are chosen.
                selection_prompt: (str) Required. LLM prompt to select relevant URLs (use "{topic}", "{search_results_context}", "{max_urls}"). Must request JSON {"selected_urls": [...]}.
                llm_model: (str) Required. LLM model for site selection ('provider/model_name'). # Renamed for clarity from site_selection_llm_model
                max_sites_to_visit: (int) Required. Max number of selected sites to process.
            extraction_phase: (dict) Required. Defines data extraction from visited sites.
                extraction_prompt_or_schema: (Union[str, Dict]) Required. LLM prompt or JSON schema for extraction (use "{topic}"). Must request JSON output.
                llm_model: (str) Required. LLM model for extraction ('provider/model_name'). # Renamed for clarity
            synthesis_phase: (dict) Required. Defines the final report generation.
                synthesis_prompt: (str) Required. LLM prompt for generating the report (use "{topic}", "{extracted_information_context}").
                llm_model: (str) Required. LLM model for synthesis ('provider/model_name'). # Renamed for clarity
                report_format_description: (str) Required. Description of the desired output format (guides the synthesis LLM).
            browser_options: (Optional[dict]) Options for browser_init (e.g., {"headless": True}). Merged with top-level.
            concurrency: (Optional[dict]) Defines parallel execution settings.
                 max_concurrent_extractions: (Optional[int]) Overrides top-level arg if set. Default 3.
        browser_options: (Optional) Options for browser_init. Applied over instructions' options.
        max_concurrent_extractions: (Optional) Max number of site extractions. Applied over instructions' options.

    Returns:
        Dictionary with the final report and process summary.
    """
    start_time = time.monotonic()
    logger.info(f"Starting research and synthesis report for topic: '{topic}'")

    # --- Validate Instructions Structure ---
    if not isinstance(instructions, dict): raise ToolInputError("Instructions must be a dictionary.")
    for phase_key in ["search_phase", "site_selection_phase", "extraction_phase", "synthesis_phase"]:
        if not isinstance(instructions.get(phase_key), dict): raise ToolInputError(f"Instructions missing/invalid '{phase_key}'.")

    search_phase = instructions['search_phase']
    site_selection_phase = instructions['site_selection_phase']
    extraction_phase = instructions['extraction_phase']
    synthesis_phase = instructions['synthesis_phase']

    # Search phase validation
    search_queries = search_phase.get("search_queries")
    if not isinstance(search_queries, list) or not search_queries: raise ToolInputError("search_phase['search_queries'] (list) required.")
    search_engine = search_phase.get("search_engine", "google").lower()
    num_search_results = search_phase.get("num_search_results_per_query", 10)
    if not isinstance(num_search_results, int) or num_search_results < 1: raise ToolInputError("num_search_results_per_query must be positive integer.")
    valid_engines = {"google", "bing", "duckduckgo"};
    if search_engine not in valid_engines: raise ToolInputError(f"search_engine invalid. Choose from {valid_engines}")

    # Site selection validation
    selection_prompt = site_selection_phase.get("selection_prompt")
    max_sites_to_visit = site_selection_phase.get("max_sites_to_visit")
    site_selection_llm_model = site_selection_phase.get("llm_model") # *** FIX: Get model for this phase ***
    if not isinstance(selection_prompt, str) or not selection_prompt: raise ToolInputError("site_selection_phase['selection_prompt'] (string) required.")
    if not isinstance(max_sites_to_visit, int) or max_sites_to_visit < 1: raise ToolInputError("max_sites_to_visit must be positive integer.")
    if not site_selection_llm_model or not isinstance(site_selection_llm_model, str) or '/' not in site_selection_llm_model: raise ToolInputError("site_selection_phase['llm_model'] ('provider/model_name') required.") # *** FIX: Validate model ***

    # Extraction phase validation
    extraction_prompt_or_schema = extraction_phase.get("extraction_prompt_or_schema")
    extraction_llm_model = extraction_phase.get("llm_model") # *** FIX: Get model for this phase ***
    if not extraction_prompt_or_schema: raise ToolInputError("extraction_phase['extraction_prompt_or_schema'] required.")
    if not extraction_llm_model or not isinstance(extraction_llm_model, str) or '/' not in extraction_llm_model: raise ToolInputError("extraction_phase['llm_model'] ('provider/model_name') required.") # *** FIX: Validate model ***

    # Synthesis phase validation
    synthesis_prompt_template = synthesis_phase.get("synthesis_prompt")
    synthesis_llm_model = synthesis_phase.get("llm_model") # *** FIX: Get model for this phase ***
    report_format_desc = synthesis_phase.get("report_format_description")
    if not isinstance(synthesis_prompt_template, str) or not synthesis_prompt_template: raise ToolInputError("synthesis_phase['synthesis_prompt'] (string) required.")
    if not synthesis_llm_model or not isinstance(synthesis_llm_model, str) or '/' not in synthesis_llm_model: raise ToolInputError("synthesis_phase['llm_model'] ('provider/model_name') required.") # *** FIX: Validate model ***
    if not isinstance(report_format_desc, str) or not report_format_desc: raise ToolInputError("report_format_description (string) required.")

    # --- Setup ---
    # Merge browser options: defaults < instructions < call parameters
    default_browser_opts = {"headless": True}
    instr_browser_opts = instructions.get("browser_options", {})
    final_browser_opts = {**default_browser_opts, **instr_browser_opts, **(browser_options or {})}

    # Concurrency: call parameters override instructions override tool default
    instr_concurrency = instructions.get("concurrency", {})
    final_max_concurrent = instr_concurrency.get("max_concurrent_extractions", max_concurrent_extractions)
    if not isinstance(final_max_concurrent, int) or final_max_concurrent < 1: raise ToolInputError("Concurrency must be a positive integer.")

    browser_was_initialized_by_tool = False
    all_search_results: List[Dict[str, str]] = []
    selected_urls_to_visit: List[str] = []
    extracted_snippets: List[Dict[str, Any]] = []
    errors_dict: Dict[str, str] = {} # stage/url -> error
    final_report: Optional[str] = None

    try:
        # --- Initialize Browser (if needed) ---
        global _browser_instance
        if not _browser_instance or not _browser_instance.is_connected():
             init_res = await browser_init(**final_browser_opts)
             if not init_res.get("success"):
                 raise ToolError(f"Browser init failed: {init_res.get('error')}")
             browser_was_initialized_by_tool = True

        # --- 1. Search Phase ---
        seen_urls = set()
        logger.info(f"Starting search phase using {len(search_queries)} queries on {search_engine}...")
        for query_template in search_queries:
            try: query = query_template.format(topic=topic)
            except KeyError: raise ToolInputError(f"Search query template '{query_template}' missing '{{topic}}' placeholder.")

            try:
                results = await _perform_web_search(query, engine=search_engine, num_results=num_search_results)
                new_found_count = 0
                for res in results:
                    if res.get('url') and res['url'] not in seen_urls:
                        all_search_results.append(res)
                        seen_urls.add(res['url'])
                        new_found_count += 1
                logger.debug(f"Query '{query}' yielded {new_found_count} new unique results.")
            except Exception as search_err:
                logger.error(f"Search failed for query '{query}': {search_err}", exc_info=False)
                errors_dict[f"Search: {query}"] = f"{type(search_err).__name__}: {str(search_err)}"

        if not all_search_results:
            raise ToolError("Web search phase yielded no usable results.", error_code="NO_SEARCH_RESULTS")
        logger.info(f"Collected {len(all_search_results)} unique search results.")


        # --- 2. Site Selection Phase ---
        logger.info(f"Starting site selection phase (LLM: {site_selection_llm_model})...")
        try:
            selected_urls_to_visit = await _select_relevant_urls_llm(
                all_search_results, selection_prompt, topic, max_sites_to_visit, site_selection_llm_model # *** FIX: Pass correct model ***
            )
            if not selected_urls_to_visit:
                raise ToolError("LLM failed to select any relevant URLs for extraction.", error_code="NO_RELEVANT_URLS_SELECTED")
            logger.info(f"LLM selected {len(selected_urls_to_visit)} URLs for extraction.")
        except Exception as select_err:
             errors_dict["Site Selection"] = f"{type(select_err).__name__}: {str(select_err)}"
             raise ToolError(f"Site selection failed: {select_err}") from select_err

        # --- 3. Information Extraction Phase ---
        logger.info(f"Starting information extraction from {len(selected_urls_to_visit)} URLs (LLM: {extraction_llm_model}, Max Concurrency: {final_max_concurrent})...")
        semaphore = asyncio.Semaphore(final_max_concurrent)
        tasks = []

        async def extraction_task_wrapper(url):
            async with semaphore:
                return await _extract_info_from_url_llm(url, extraction_prompt_or_schema, topic, extraction_llm_model)

        tasks = [extraction_task_wrapper(url) for url in selected_urls_to_visit]
        extraction_results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

        # Process extraction results (remains the same)
        for i, result_or_exc in enumerate(extraction_results_from_gather):
            processed_url = selected_urls_to_visit[i]
            if isinstance(result_or_exc, Exception):
                err_msg = f"Extraction task for {processed_url} failed: {type(result_or_exc).__name__}: {str(result_or_exc)}"
                logger.error(err_msg, exc_info=False)
                errors_dict[processed_url] = str(result_or_exc)
            elif isinstance(result_or_exc, dict):
                extract_err = result_or_exc.get("_extraction_error")
                if extract_err:
                    logger.warning(f"Extraction failed for {processed_url}: {extract_err}")
                    errors_dict[processed_url] = extract_err
                else:
                    actual_data = {k: v for k, v in result_or_exc.items() if not k.startswith("_")}
                    actual_data["_source_url"] = processed_url
                    extracted_snippets.append(actual_data)
            else:
                err_msg = f"Unexpected result type ({type(result_or_exc)}) from extraction task for {processed_url}"
                logger.error(err_msg); errors_dict[processed_url] = err_msg

        if not extracted_snippets:
             raise ToolError("Information extraction phase failed for all selected URLs.", error_code="EXTRACTION_FAILED_ALL")
        logger.info(f"Successfully extracted information snippets from {len(extracted_snippets)} URLs.")

        # --- 4. Synthesis Phase ---
        logger.info(f"Synthesizing final report (Format: {report_format_desc}, LLM: {synthesis_llm_model})...")
        try:
            final_report = await _synthesize_report_llm(
                extracted_snippets, synthesis_prompt_template, topic, report_format_desc, synthesis_llm_model # *** FIX: Pass correct model ***
            )
            if final_report.startswith("Error:"):
                 raise ToolError(f"Synthesis failed: {final_report}", error_code="SYNTHESIS_FAILED")
        except Exception as synth_err:
             errors_dict["Synthesis"] = f"{type(synth_err).__name__}: {str(synth_err)}"
             raise ToolError(f"Report synthesis failed: {synth_err}") from synth_err


    except (ToolInputError, ToolError) as e:
         logger.error(f"Error during research & synthesis for '{topic}': {type(e).__name__}: {e}", exc_info=True)
         return { "success": False, "topic": topic, "error": str(e), "report": None, "processed_urls": selected_urls_to_visit, "successful_extractions": len(extracted_snippets), "errors": errors_dict, "message": f"Research process failed: {e}" }
    except Exception as e:
         logger.critical(f"Unexpected critical error during research & synthesis for '{topic}': {type(e).__name__}: {e}", exc_info=True)
         return { "success": False, "topic": topic, "error": f"Unexpected critical error: {str(e)}", "report": None, "processed_urls": selected_urls_to_visit, "successful_extractions": len(extracted_snippets), "errors": errors_dict, "message": f"Research process failed critically: {type(e).__name__}" }
    finally:
        if browser_was_initialized_by_tool:
            logger.info("Closing browser instance initialized by research_and_synthesize_report tool.")
            await browser_close()

    processing_time = time.monotonic() - start_time
    msg = f"Research and synthesis for '{topic}' complete. Synthesized report based on info from {len(extracted_snippets)} sources."
    if errors_dict: msg += f" Encountered {len(errors_dict)} errors during processing."
    logger.success(msg, time=processing_time)

    return { # Return success structure
        "success": True, "topic": topic, "report": final_report,
        "processed_urls": [d.get("_source_url", "Unknown") for d in extracted_snippets], # URLs info was successfully extracted from
        "successful_extractions": len(extracted_snippets), "errors": errors_dict,
        "message": msg, "processing_time": processing_time
    }

async def _detect_search_results_element():
    """Dynamically detect search results container on the current page using JavaScript."""
    script = """() => {
        // Common patterns for search results containers
        const possibleSelectors = [
            // DuckDuckGo selectors
            '.react-results--main', '.serp__results', '#links', '.results', 
            // Google selectors
            '#search', '#rso', '.g',
            // Bing selectors
            '#b_results', '#b_content',
            // Generic selectors
            '[role="main"]', 'main', '.main-results', '.search-results'
        ];
        
        // Try each selector
        for (const selector of possibleSelectors) {
            const element = document.querySelector(selector);
            if (element) {
                // Check if it likely contains results (has multiple children or significant content)
                if (element.children.length > 2 || element.textContent.length > 100) {
                    return {
                        selector: selector,
                        found: true,
                        childCount: element.children.length
                    };
                }
            }
        }
        
        // If no predefined selector works, try to intelligently find results
        // Look for elements with multiple similar children (like a list of results)
        const potentialContainers = Array.from(document.querySelectorAll('div, section, main, ol, ul'))
            .filter(el => el.children.length >= 3);
            
        // Sort by number of children (more likely to be results container)
        potentialContainers.sort((a, b) => b.children.length - a.children.length);
        
        if (potentialContainers.length > 0) {
            const bestGuess = potentialContainers[0];
            let path = '';
            
            // Construct a selector path (not perfect but often works)
            if (bestGuess.id) {
                path = '#' + bestGuess.id;
            } else if (bestGuess.className) {
                path = '.' + bestGuess.className.split(' ')[0];
            } else {
                // Try to create a path using nth-child 
                let el = bestGuess;
                let tempPath = '';
                while (el && el !== document.body) {
                    const siblings = Array.from(el.parentNode.children);
                    const index = siblings.indexOf(el) + 1;
                    const tag = el.tagName.toLowerCase();
                    tempPath = `${tag}:nth-child(${index})${tempPath ? ' > ' + tempPath : ''}`;
                    el = el.parentNode;
                }
                path = tempPath;
            }
            
            return {
                selector: path,
                found: true,
                bestGuess: true,
                childCount: bestGuess.children.length
            };
        }
        
        return { found: false };
    }"""
    
    result = await browser_execute_javascript(script=script)
    if result.get("success") and result.get("result", {}).get("found"):
        return result["result"]["selector"]
    return None

