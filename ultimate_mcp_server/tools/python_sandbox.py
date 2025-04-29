# ultimate_mcp_server/tools/python_sandbox.py

"""Pyodide-backed sandbox tool for Ultimate MCP Server.

Provides a secure environment for executing Python code within a headless browser,
with stdout/stderr capture, package management, security controls, and optional REPL functionality.
"""

###############################################################################
# Standard library & typing
###############################################################################
import asyncio
import atexit
import collections
import json
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

###############################################################################
# Third‑party – runtime dependency only on Playwright
###############################################################################
# Use a type guard to handle potential ImportError
try:
    import playwright.async_api as pw
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pw = None # type: ignore
    PLAYWRIGHT_AVAILABLE = False

from ultimate_mcp_server.constants import TaskType
from ultimate_mcp_server.exceptions import (
    ProviderError,
    ToolError,
    ToolInputError,
)

# Import decorators needed for standalone functions
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.python_sandbox")

# Constants
COMMON_PACKAGES: list[str] = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "networkx",
]
MAX_SANDBOXES = 6            # LRU-evicted beyond this
GLOBAL_CONCURRENCY = 8       # concurrent exec() across sandboxes
MEM_LIMIT_MB = 512           # soft heap cap – page closed when breached
PYODIDE_CDN = (
    "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
)  # keep in sync with boot html


###############################################################################
# Browser / bookkeeping singletons
###############################################################################

_BROWSER: Optional[Any] = None
_PAGES: "collections.OrderedDict[str, PyodideSandbox]" = collections.OrderedDict()
_GLOBAL_SEM: Optional[asyncio.Semaphore] = None   # set lazily in _get_sandbox()

###############################################################################
# Pyodide boot HTML – includes io/traceback capture + watchdogs
###############################################################################

PYODIDE_BOOT_HTML = textwrap.dedent(
    f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <script src="{PYODIDE_CDN}pyodide.js"></script>
      </head>
      <body>
        <script>
          /* global loadPyodide */
          (async () => {{
            const t0 = performance.now();
            self.pyodide = await loadPyodide({{ indexURL: '{PYODIDE_CDN}' }});

            /* best-effort preload of common packages */
            await self.pyodide.loadPackage({json.dumps(COMMON_PACKAGES)}).catch(() => {{}});

            const BOOT_MS = performance.now() - t0;
            const now     = () => performance.now();
            const heapMB  = () =>
              (performance?.memory?.usedJSHeapSize ?? 0) / 1048576;

            // —————————————————— main handler ——————————————————
            self.addEventListener("message", async (ev) => {{
              const msg = ev.data;

              // Handle reset message to clear the REPL state
              if (msg.type === "reset") {{
                /* clear state without closing the page */
                if (self.pyodide?.globals?.has("_MCP_REPL_NS")) {{
                   self.pyodide.globals.delete("_MCP_REPL_NS");
                }}
                self.postMessage({{ id: msg.id, ok: true, cleared: true }});
                return;
              }}

              if (msg.type !== "exec") return;

              const reply = {{ id: msg.id }};
              try {{
                /* on-demand package / wheel loading (network-gated) */
                if (msg.packages?.length)
                  await self.pyodide.loadPackage(msg.packages);

                if (msg.wheels?.length) {{
                  await self.pyodide.loadPackage("micropip");
                  const micropip = pyodide.pyimport("micropip");
                  for (const whl of msg.wheels) await micropip.install(whl);
                }}

                /* wrap → capture stdout / stderr / result */
                const runner = `
import sys, io, contextlib, traceback, time, json
_stdout, _stderr = io.StringIO(), io.StringIO()

# REPL mode flag passed from JavaScript
repl_mode = ${{msg.repl_mode ? 'True' : 'False'}}

# Handle namespace based on mode
if repl_mode:
    # REPL mode - maintain state between calls
    if not pyodide.globals.has('_MCP_REPL_NS'):
        # Initialize if it doesn't exist
        pyodide.globals.set('_MCP_REPL_NS', {{'__name__': '__main__'}} )
    # Get the existing namespace object/proxy
    ns_proxy = pyodide.globals.get('_MCP_REPL_NS')
    # Convert proxy to a Python dict if needed, or use directly if API allows
    # For simplicity, let's assume runPython uses it correctly. If not, conversion needed.
    ns = ns_proxy # Use the persistent namespace proxy
else:
    # One-shot mode - fresh namespace each time
    ns = {{'__name__': '__main__'}}

# the next line receives the JS stringified code
code = json.loads(${{JSON.stringify(msg.code)}})

start = time.time()
try:
    with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
        # Execute code within the chosen namespace
        # Use the ns proxy directly for REPL mode if supported by exec
        exec(code, ns, ns)
    # Extract result from the namespace after execution
    result_key = 'result'
    rv = ns.get(result_key) if hasattr(ns, 'get') and ns.has(result_key) else None # Handle proxy access if needed
    ok = True
except Exception:
    rv, ok = traceback.format_exc(), False
elapsed = (time.time() - start) * 1000

# Explicitly convert result to standard Python type if it's a PyProxy
if 'pyodide' in sys.modules and isinstance(rv, pyodide.ffi.JsProxy):
    try:
        rv = rv.to_py()
    except Exception:
        rv = f"<Unserializable Pyodide Object: {{type(rv).__name__}}>"

__payload = dict(
    ok=ok,
    result=rv, # Already converted if needed
    stdout=_stdout.getvalue(),
    stderr=_stderr.getvalue(),
    elapsed=elapsed,
)
# Convert the final payload to JS object for postMessage
# json.dumps(__payload, default=str) # Ensure serializability
__payload`;

                const wall0 = now();
                const pyRes = await self.pyodide.runPythonAsync(runner);
                // Convert Python dict result (pyRes) to JS object before assigning
                const jsResult = pyRes.toJs ? pyRes.toJs({{ dict_converter: Object.fromEntries }}) : pyRes;
                Object.assign(reply, jsResult, {{ wall_ms: now() - wall0 }});
              }} catch (err) {{
                reply.ok = false;
                reply.error = err.toString();
              }}

              self.postMessage(reply);

              /* soft heap watchdog: checks JS heap only – not Python/RSS */
              if (heapMB() > {MEM_LIMIT_MB}) {{
                console.warn("Sandbox heap > " + heapMB().toFixed(0) + " MB – closing tab");
                setTimeout(() => window.close(), 0);
              }}
            }});

            self.postMessage({{ ready: true, boot_ms: BOOT_MS }});
          }})();
        </script>
      </body>
    </html>
    """
)

###############################################################################
# Sandbox wrapper
###############################################################################

@dataclass
class PyodideSandbox:
    """One Chromium tab with Pyodide runtime (optionally persistent)."""

    page: Any
    allow_network: bool = False
    allow_fs: bool = False
    ready_evt: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)  # Added timestamp
    last_used: float = field(default_factory=time.time)   # Added last used

    async def init(self):
        """Load boot HTML & wait for ready."""
        if not PLAYWRIGHT_AVAILABLE:
             raise RuntimeError("Playwright is not installed or available. Cannot initialize PyodideSandbox.")

        # Network interception --------------------------------------------------
        if not self.allow_network:
            async def _block(req: Any):
                # Allow CDN, block others
                # Convert URL to lowercase for case-insensitive comparison
                req_url_lower = req.url.lower()
                pyodide_cdn_lower = PYODIDE_CDN.lower()

                # Allow the base CDN URL and anything under it
                if req_url_lower.startswith(pyodide_cdn_lower):
                    try:
                         await req.continue_()
                    except pw.Error as e:
                         # Ignore errors continuing request (e.g., if page closed)
                         logger.debug(f"Ignoring playwright continue error: {e}")
                else:
                    # Block other requests
                    logger.debug(f"Blocking network request from sandbox: {req.url}")
                    try:
                        await req.abort()
                    except pw.Error as e:
                         # Ignore errors aborting (e.g., if request already handled or page closed)
                         logger.debug(f"Ignoring playwright abort error: {e}")

            try:
                await self.page.route("**/*", _block)
            except pw.Error as e:
                 logger.error(f"Failed to set up network blocking route: {e}", exc_info=True)
                 raise ToolError(f"Failed to configure sandbox network rules: {e}") from e


        # Boot page -------------------------------------------------------------
        await self.page.set_content(PYODIDE_BOOT_HTML)

        # Use a queue to handle messages reliably
        message_queue = asyncio.Queue()

        async def _on_msg(msg: Any):
            # Handle console messages if needed for debugging
            # logger.debug(f"Sandbox Console [{msg.type}]: {msg.text}")
            pass # Currently ignore console messages

        async def _on_page_message(payload: Any):
            # Handle messages posted from the page's JS context
            try:
                # Playwright gives the raw payload directly
                data = payload # Assume payload is already JS object/dict
                # Make sure it's a dict before proceeding
                if isinstance(data, dict):
                    # logger.debug(f"Sandbox Page Message Received: {data}") # Debug logging
                    await message_queue.put(data)
                else:
                    logger.warning(f"Received non-dict message from page: {type(payload)}")
            except Exception as e:
                 logger.error(f"Error processing page message: {e}", exc_info=True)


        self.page.on("console", _on_msg)
        # Listen for messages posted from within the page's JS
        self.page.expose_function("_handlePageMessage", _on_page_message)
        await self.page.evaluate("window.addEventListener('message', (event) => window._handlePageMessage(event.data));")


        # FS bridge if allowed --------------------------------------------------
        if self.allow_fs:
            # Ensure the listener uses the queue mechanism if needed, or direct callback
            await _listen_for_mcpfs_calls(self.page)
            await self._inject_mcpfs_stub()

        # Wait for the 'ready' message via the queue
        try:
            while True:
                data = await asyncio.wait_for(message_queue.get(), timeout=20)
                if data.get("ready"):
                    self.ready_evt.set()
                    logger.info(f"Pyodide sandbox ready (boot ms: {data.get('boot_ms', 'N/A')})")
                    break
        except asyncio.TimeoutError as e:
            logger.error("Timeout waiting for Pyodide sandbox 'ready' message.")
            await self.page.close() # Clean up page on failure
            raise ToolError("Pyodide sandbox failed to initialize within timeout.") from e
        except Exception as e:
            logger.error(f"Error during sandbox initialization wait: {e}", exc_info=True)
            await self.page.close() # Clean up page on failure
            raise ToolError(f"Unexpected error during sandbox initialization: {e}") from e

        # We might want to keep listening to messages *after* ready for execution results
        # The execute method will likely set up its own listener or use the queue.

    # ---------------------------------------------------------------------

    async def _inject_mcpfs_stub(self) -> None:
        """
        Creates a minimal stub module `mcpfs` inside the Pyodide interpreter.
        The stub forwards `.read_text / write_text / listdir` calls back to
        the host via postMessage → _listen_for_mcpfs_calls.
        """
        if not PLAYWRIGHT_AVAILABLE:
             return # Skip if playwright not available

        stub_code = r"""
import pyodide, types, asyncio, builtins, json
from js import self as _js               # window, or use dedicated postMessage if needed
_msg_id = 0
# Placeholder for futures keyed by message ID
_pending_mcpfs_futures = {}

async def _roundtrip(op, *a):
    global _msg_id, _pending_mcpfs_futures
    _msg_id += 1
    current_id = _msg_id
    fut = asyncio.get_running_loop().create_future()
    _pending_mcpfs_futures[current_id] = fut

    # Post message to the host (assuming host listens for type: mcpfs)
    _js.postMessage({ "type": "mcpfs", "id": current_id, "op": op, "args": a })

    # Wait for the host to post back a message with the same ID
    try:
        # Add a timeout to avoid waiting forever
        r = await asyncio.wait_for(fut, timeout=10.0) # 10 second timeout for FS ops
    except asyncio.TimeoutError:
        raise RuntimeError(f"Timeout waiting for host response for mcpfs op {op} (id: {current_id})")
    finally:
        # Clean up future
        del _pending_mcpfs_futures[current_id]

    if r is None: # Handle case where future might resolve to None unexpectedly
        raise RuntimeError(f"Received null response from host for mcpfs op {op}")
    if "error" in r:
        raise RuntimeError(r["error"])
    return r["result"]

# Callback function to resolve futures when host responds
def _mcpfs_callback(ev):
    global _pending_mcpfs_futures
    # Check if the message is from the host for mcpfs and has an ID we are waiting for
    if isinstance(ev.data, dict) and ev.data.get("type") == "mcpfs_response":
       msg_id = ev.data.get("id")
       if msg_id in _pending_mcpfs_futures:
           fut = _pending_mcpfs_futures.get(msg_id)
           if fut and not fut.done():
               # Resolve the future with the received data (pyodide.to_py might not be needed if host sends JSON)
               fut.set_result(ev.data) # Pass the whole response dict

# Register the callback listener in the JS environment
_js.addEventListener("message", _mcpfs_callback)

m = types.ModuleType("mcpfs")
# Define async functions that use the _roundtrip helper
async def read_text_async(p): return await _roundtrip("read", p)
async def write_text_async(p, t): return await _roundtrip("write", p, t)
async def listdir_async(p): return await _roundtrip("list", p)

# Assign the async functions to the module
m.read_text = read_text_async
m.write_text = write_text_async
m.listdir = listdir_async

import sys; sys.modules["mcpfs"] = m
"""
        # Ensure the page is ready before executing JS
        if not self.page:
            raise ToolError("Sandbox page is not available for injecting mcpfs stub.")
        if self.page.is_closed():
            raise ToolError("Sandbox page is not available for injecting mcpfs stub.")

        try:
             # Execute the stub code injection using evaluate to ensure completion
             await self.page.evaluate(f"""
                 (async () => {{
                     await self.pyodide.loadPackage(['micropip']); // Ensure micropip if needed by stub logic itself
                     // Wrap stub code execution in a try-catch within JS
                     try {{
                         await self.pyodide.runPythonAsync(`{stub_code}`);
                         console.log("mcpfs stub injected successfully.");
                     }} catch (err) {{
                         console.error("Error injecting mcpfs stub:", err);
                         // Optionally post an error message back to host
                         // self.postMessage({{ type: 'error', message: 'Failed to inject mcpfs stub: ' + err.toString() }});
                     }}
                 }})();
             """)
        except Exception as e:
             logger.error(f"Failed to execute mcpfs stub injection script: {e}", exc_info=True)
             raise ToolError(f"Could not inject filesystem bridge into sandbox: {e}") from e


    # ---------------------------------------------------------------------
    async def reset_repl_state(self) -> Dict[str, Any]:
        """Reset the REPL state in the sandbox."""
        if not PLAYWRIGHT_AVAILABLE:
             return {"ok": False, "error": "Sandbox page not available for reset."}
        if not self.page:
            return {"ok": False, "error": "Sandbox page not available for reset."}
        if self.page.is_closed():
             return {"ok": False, "error": "Sandbox page not available for reset."}

        reset_id = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_event_loop().create_future()

        # Use a queue for message handling
        response_queue = asyncio.Queue()

        async def _on_reset_response(payload: Any):
             try:
                 data = payload
                 if isinstance(data, dict):
                     if data.get("id") == reset_id:
                         await response_queue.put(data)
             except Exception as e:
                 logger.error(f"Error processing reset response: {e}", exc_info=True)
                 if not fut.done():
                     fut.set_exception(e) # Signal error if future is still waiting

        # Expose a temporary function or reuse the main message handler
        handler_name = f"_handleResetResponse_{reset_id}"
        await self.page.expose_function(handler_name, _on_reset_response)
        await self.page.evaluate(f"window.addEventListener('message', (event) => {{ if (event.data.id === '{reset_id}') window.{handler_name}(event.data); }});")


        try:
            # Send the reset message
            message_payload = {
                "type": "reset",
                "id": reset_id
            }
            await self.page.evaluate("postMessage", message_payload)

            # Wait for the response from the queue
            data = await asyncio.wait_for(response_queue.get(), timeout=5)

        except asyncio.TimeoutError:
             logger.warning(f"Timeout waiting for REPL reset confirmation (id: {reset_id})")
             return {"ok": False, "error": "Timeout waiting for reset confirmation."}
        except Exception as e:
             logger.error(f"Error during REPL reset operation: {e}", exc_info=True)
             return {"ok": False, "error": f"Error during reset: {e}"}
        finally:
             # Clean up the exposed function and listener if possible (best effort)
             try:
                 # Remove the specific listener if possible, or just unexpose the function
                 await self.page.evaluate(f"delete window.{handler_name};") # Basic cleanup
             except Exception as cleanup_err:
                 logger.debug(f"Ignoring cleanup error for reset handler: {cleanup_err}")

        return data # Return the confirmation data

    async def execute(
        self,
        code: str,
        packages: list[str] | None,
        wheels: list[str] | None,
        timeout_ms: int,
        repl_mode: bool = False,
    ) -> Dict[str, Any]:
        """Low‑level execution inside the tab."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ToolError("Sandbox page is not available for execution.")
        if not self.page:
             raise ToolError("Sandbox page is not available for execution.")
        if self.page.is_closed():
            raise ToolError("Sandbox page is not available for execution.")

        # Update last used time for LRU tracking
        self.last_used = time.time()

        # Use the global semaphore defined earlier
        if _GLOBAL_SEM is None: # Should have been set by _get_sandbox
             raise RuntimeError("Global sandbox semaphore not initialized.")

        async with _GLOBAL_SEM:
            exec_id = uuid.uuid4().hex
            fut: asyncio.Future = asyncio.get_event_loop().create_future()

            # Use a queue for message handling
            exec_response_queue = asyncio.Queue()

            async def _on_exec_response(payload: Any):
                try:
                    data = payload
                    if isinstance(data, dict):
                        if data.get("id") == exec_id:
                            await exec_response_queue.put(data)
                except Exception as e:
                    logger.error(f"Error processing execution response: {e}", exc_info=True)
                    if not fut.done(): # Check if future is still pending
                        # Set exception only if it hasn't been resolved or cancelled
                        try:
                           fut.set_exception(e)
                        except asyncio.InvalidStateError:
                           logger.debug("Future was already done when trying to set exception.")


            # Expose a temporary function or reuse main handler if possible
            handler_name = f"_handleExecResponse_{exec_id}"
            await self.page.expose_function(handler_name, _on_exec_response)
            await self.page.evaluate(f"window.addEventListener('message', (event) => {{ if (event.data.id === '{exec_id}') window.{handler_name}(event.data); }});")

            payload = {
                "type": "exec",
                "id": exec_id,
                "code": code,
                "packages": packages or [],
                "wheels": wheels or [],
                "repl_mode": repl_mode,
            }

            try:
                # Send the execution request
                await self.page.evaluate("postMessage", payload)
                # Wait for the response from the queue
                data: dict[str, Any] = await asyncio.wait_for(
                    exec_response_queue.get(), timeout=timeout_ms / 1000
                )
            except asyncio.TimeoutError as e:
                raise RuntimeError("Execution timed out") from e
            except Exception as e: # Catch other errors during wait/get
                 logger.error(f"Error waiting for sandbox execution result: {e}", exc_info=True)
                 raise ToolError(f"Error communicating with sandbox during execution: {e}") from e
            finally:
                # Clean up the exposed function (best effort)
                try:
                     await self.page.evaluate(f"delete window.{handler_name};")
                except Exception as cleanup_err:
                     logger.debug(f"Ignoring cleanup error for exec handler: {cleanup_err}")

            # Check response status before returning
            if not data.get("ok"):
                error_message = data.get("error", "Pyodide error")
                raise RuntimeError(error_message) # Raise runtime error on failure

            return data # Return successful data

###############################################################################
# Browser / sandbox lifecycle helpers – with LRU eviction
###############################################################################

async def _get_browser() -> Any:
    """Initializes and returns the shared Playwright browser instance."""
    global _BROWSER
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright is not installed. Cannot create browser for Pyodide sandbox.")

    browser_connected = False
    if _BROWSER is not None:
        browser_connected = _BROWSER.is_connected()

    if _BROWSER is None or not browser_connected:
        logger.info("Launching headless Chromium for Pyodide sandbox...")
        try:
            # Get the playwright instance - properly awaited
            playwright = await pw.async_playwright().start()
            # Then launch the browser
            launch_options = {
                "args": [
                    "--no-sandbox", # Often needed in containerized environments
                    "--disable-gpu", # Can help stability in headless
                    "--disable-dev-shm-usage" # Often needed in containers
                ]
            }
            _BROWSER = await playwright.chromium.launch(**launch_options)
            # Ensure graceful shutdown when the host process exits
            # Use a robust way to schedule async cleanup from sync context
            def _sync_cleanup():
                 try:
                      loop = asyncio.get_event_loop()
                      if loop.is_running():
                           asyncio.run_coroutine_threadsafe(_BROWSER.close(), loop)
                      else:
                           # If no loop is running, run in a new one (less ideal but necessary)
                           asyncio.run(_BROWSER.close())
                 except Exception as e:
                      logger.error(f"Error during Playwright browser cleanup: {e}")

            atexit.register(_sync_cleanup)
            logger.info("Headless Chromium launched successfully.")
        except Exception as e:
             logger.error(f"Failed to launch Playwright browser: {e}", exc_info=True)
             _BROWSER = None # Ensure it's None if launch fails
             raise ProviderError(f"Failed to launch browser for sandbox: {e}") from e

    return _BROWSER

async def _get_sandbox(session_id: str, **kwargs) -> PyodideSandbox:
    """Retrieves or creates a PyodideSandbox instance, managing LRU cache."""
    global _GLOBAL_SEM
    if not PLAYWRIGHT_AVAILABLE:
         raise RuntimeError("Playwright is not installed. Cannot get sandbox.")

    if _GLOBAL_SEM is None:
        _GLOBAL_SEM = asyncio.Semaphore(GLOBAL_CONCURRENCY)

    # --- Check Cache and Health ---
    sb = _PAGES.get(session_id)
    if sb is not None:
        # Check if page is still valid and connected
        page_valid = False
        if sb.page:
            if not sb.page.is_closed():
                page_valid = True

        if page_valid:
            logger.debug(f"Reusing existing sandbox session: {session_id}")
            _PAGES.move_to_end(session_id)  # Mark as recently used
            sb.last_used = time.time() # Update last used time
            return sb
        else:
            # Page is closed or invalid, remove from cache
            logger.warning(f"Removing closed/invalid sandbox session from cache: {session_id}")
            _PAGES.pop(session_id, None)
            # Fall through to create a new one

    # --- Evict LRU if Over Capacity ---
    while len(_PAGES) >= MAX_SANDBOXES:
        victim_id, victim_sb = _PAGES.popitem(last=False) # Pop oldest
        logger.info(f"Sandbox cache full ({len(_PAGES)}/{MAX_SANDBOXES}). Evicting LRU session: {victim_id}")
        try:
            page_needs_closing = False
            if victim_sb.page:
                if not victim_sb.page.is_closed():
                    page_needs_closing = True
            if page_needs_closing:
                await victim_sb.page.close()
        except Exception as e:
            logger.warning(f"Error closing evicted sandbox page {victim_id}: {e}")

    # --- Create New Sandbox ---
    logger.info(f"Creating new sandbox session: {session_id}")
    browser = await _get_browser() # Ensure browser is ready
    page: Optional[Any] = None
    try:
        page = await browser.new_page()
        # Optional: Add error listeners early
        def _log_page_error(exc):
            logger.error(f"Sandbox Page Error ({session_id}): {exc}", exc_info=exc)
        page.on("pageerror", _log_page_error)

        def _log_page_crash():
            logger.error(f"Sandbox Page Crashed ({session_id})!")
        page.on("crash", _log_page_crash)

        sb = PyodideSandbox(page=page, **kwargs)
        await sb.init() # Initialize (loads HTML, waits for ready)
        _PAGES[session_id] = sb
        logger.info(f"New sandbox session {session_id} created and ready.")
        return sb
    except Exception as e:
        logger.error(f"Failed to create or initialize new sandbox {session_id}: {e}", exc_info=True)
        # Cleanup potentially created page if init failed
        page_needs_closing = False
        if page:
            if not page.is_closed():
                page_needs_closing = True
        if page_needs_closing:
            await page.close()
        raise ProviderError(f"Failed to create sandbox {session_id}: {e}") from e


async def _close_all_sandboxes():
    """Gracefully close all active sandbox pages and the browser."""
    global _BROWSER
    global _PAGES
    logger.info("Closing all active Pyodide sandboxes...")
    page_close_tasks = []
    # Iterate over a copy of items to allow modification
    pages_to_close = list(_PAGES.items())
    for session_id, sb in pages_to_close:
        page_needs_closing = False
        if sb.page:
            if not sb.page.is_closed():
                page_needs_closing = True

        if page_needs_closing:
            logger.debug(f"Closing sandbox page: {session_id}")
            close_task = asyncio.create_task(sb.page.close())
            page_close_tasks.append(close_task)
        _PAGES.pop(session_id, None) # Remove from dict

    if page_close_tasks:
        gathered_results = await asyncio.gather(*page_close_tasks, return_exceptions=True)
        closed_count = 0
        for result in gathered_results:
            if not isinstance(result, Exception):
                closed_count += 1
            else:
                logger.warning(f"Exception during sandbox page close: {result}")
        logger.info(f"Closed {closed_count} of {len(page_close_tasks)} sandbox pages.")

    browser_needs_closing = False
    if _BROWSER:
        if _BROWSER.is_connected():
            browser_needs_closing = True

    if browser_needs_closing:
        logger.info("Closing Playwright browser...")
        try:
            await _BROWSER.close()
            logger.info("Playwright browser closed.")
        except Exception as e:
            logger.error(f"Error closing Playwright browser: {e}")
    _BROWSER = None
    _PAGES.clear()

# Optional: Register sandbox cleanup on shutdown
# Make sure this is called appropriately during your server's shutdown sequence
# e.g., using FastAPI lifespan or similar mechanism.
# register_shutdown_handler(_close_all_sandboxes)


###############################################################################
# mcpfs bridge – listens for postMessage & proxies to secure FS tool
###############################################################################

async def _listen_for_mcpfs_calls(page: Any):
    """Sets up listener for 'mcpfs' messages from the sandbox page."""
    if not PLAYWRIGHT_AVAILABLE:
        return

    async def _handle_mcpfs_message(payload: Any):
        """Processes 'mcpfs' request and sends 'mcpfs_response' back."""
        data = payload
        # Basic check for message type and structure
        is_mcpfs_message = False
        if isinstance(data, dict):
            if data.get("type") == "mcpfs":
                is_mcpfs_message = True

        if not is_mcpfs_message:
            # logger.debug(f"Ignoring non-mcpfs message: {data}") # Can be noisy
            return # Ignore messages not relevant to mcpfs

        call_id = data.get("id")
        op = data.get("op")
        args = data.get("args", [])

        # Basic validation
        if not call_id:
             logger.warning(f"Received invalid mcpfs message (missing id): {data}")
             return
        if not op:
             logger.warning(f"Received invalid mcpfs message (missing op): {data}")
             return

        response_payload: dict[str, Any] = {"type": "mcpfs_response", "id": call_id}
        try:
            # Dynamically import filesystem tools *inside* the handler
            # to potentially mitigate import cycle issues if they arise,
            # and ensure the latest version is used if modules are reloaded.
            from ultimate_mcp_server.tools import filesystem as fs

            logger.debug(f"MCPFS Bridge: Received op={op}, args={args}, id={call_id}")

            if op == "read":
                if len(args) != 1:
                    raise ValueError("read requires 1 argument (path)")
                # Assuming read_file returns {'content': [{'type':'text', 'text':...}]} or error structure
                path_arg = args[0]
                res = await fs.read_file(path_arg)
                # Check success and content format
                content_valid = False
                if res.get("success"):
                    content_list = res.get("content")
                    if isinstance(content_list, list):
                        if content_list: # Check if list is not empty
                           content_valid = True

                if content_valid:
                     # Extract text content from the standard response format
                     first_content_item = res["content"][0]
                     text_content = first_content_item.get("text", "<Error: Content format unexpected>")
                     response_payload["result"] = text_content
                elif not res.get("success"):
                    # Propagate error message from the filesystem tool
                    error_message = res.get("error", "Filesystem read error")
                    error_details = res.get("details")
                    raise ToolError(error_message, details=error_details)
                else:
                    raise ToolError("Filesystem read operation returned unexpected success format.")

            elif op == "write":
                if len(args) != 2:
                    raise ValueError("write requires 2 arguments (path, content)")
                path_arg = args[0]
                content_arg = args[1]
                res = await fs.write_file(path_arg, content_arg)
                if res.get("success"):
                     response_payload["result"] = True # Indicate success
                else:
                     error_message = res.get("error", "Filesystem write error")
                     error_details = res.get("details")
                     raise ToolError(error_message, details=error_details)

            elif op == "list":
                if len(args) != 1:
                    raise ValueError("list requires 1 argument (path)")
                path_arg = args[0]
                res = await fs.list_directory(path_arg)
                if res.get("success"):
                     # list_directory returns {'entries': [...]}
                     entries = res.get("entries", [])
                     response_payload["result"] = entries
                else:
                     error_message = res.get("error", "Filesystem list error")
                     error_details = res.get("details")
                     raise ToolError(error_message, details=error_details)
            else:
                raise ValueError(f"Unsupported FS op: {op}")

        except (ToolError, ToolInputError, ProviderError) as tool_exc:
            # Catch specific tool errors and format them
            logger.warning(f"MCPFS Bridge Error (op={op}, args={args}): {tool_exc}")
            error_type_name = type(tool_exc).__name__
            error_string = str(tool_exc)
            response_payload["error"] = f"{error_type_name}: {error_string}"
            # Check for details attribute and ensure serializability
            has_details = hasattr(tool_exc, 'details')
            if has_details:
                details_value = tool_exc.details
                if details_value:
                    try:
                        # Attempt to serialize details
                        serialized_details = json.dumps(details_value, default=str)
                        response_payload["details"] = json.loads(serialized_details)
                    except Exception:
                        response_payload["details"] = {"error": "Could not serialize error details"}
        except Exception as exc:
            # Catch broader exceptions
            logger.error(f"Unexpected MCPFS Bridge Error (op={op}, args={args}): {exc}", exc_info=True)
            error_string = str(exc)
            response_payload["error"] = f"Unexpected Host Error: {error_string}"

        # Send the response back to the page
        try:
             response_successful = 'error' not in response_payload
             logger.debug(f"MCPFS Bridge: Sending response id={call_id}, success={response_successful}")
             await page.evaluate("postMessage", response_payload)
        except Exception as post_err:
             # Log error if sending response back fails (page might have closed)
             logger.warning(f"Failed to send mcpfs response back to sandbox (id: {call_id}): {post_err}")


    # Expose the handler function to the page context
    handler_func_name = "_handleMcpFsMessage"
    try:
        await page.expose_function(handler_func_name, _handle_mcpfs_message)
        # Add an event listener within the page to call the exposed function
        await page.evaluate(f"window.addEventListener('message', (event) => window.{handler_func_name}(event.data));")
        logger.info("MCPFS message listener bridge established.")
    except Exception as e:
        logger.error(f"Failed to establish MCPFS message listener bridge: {e}", exc_info=True)
        raise ToolError(f"Could not set up filesystem bridge listener: {e}") from e


###############################################################################
# Standalone Tool Functions (Replaces the Class Methods)
###############################################################################

@with_tool_metrics
@with_error_handling
async def execute_python(
    code: str,
    packages: Optional[List[str]] = None,
    wheels: Optional[List[str]] = None,
    allow_network: bool = False,
    allow_fs: bool = False,
    session_id: Optional[str] = None,
    timeout_ms: int = 15_000,
    # Add ctx=None for compatibility with how decorators might pass context
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Python code in a Pyodide sandbox and return stdout, stderr, result & timings.

    This is a ONE-SHOT execution that doesn't maintain state between calls. Each call
    runs in a fresh namespace, even if you use the same session_id (which only persists
    loaded packages and wheels, not variables or definitions).

    For a persistent REPL-like experience where variables and definitions persist between
    calls, use the repl_python() function instead.

    Args:
        code: The Python code to execute in the sandbox.
        packages: Optional list of Pyodide packages to load before execution.
                Common packages like numpy, pandas, matplotlib, scipy, and networkx
                are preloaded by default.
        wheels: Optional list of Python wheel URLs to install via micropip.
                Only used if allow_network is True.
        allow_network: Whether to allow network access from the sandbox.
                    Default is False for security.
        allow_fs: Whether to allow access to the secure filesystem bridge.
                Default is False.
        session_id: Optional identifier for a persistent sandbox session.
                If None, a new random UUID is generated. Use the same ID
                to reuse a previously created sandbox environment (loaded packages
                will persist, but variable state won't).
        timeout_ms: Maximum execution time in milliseconds before timeout.
                Default is 15,000 (15 seconds).
        ctx: Optional context dictionary (often passed by MCP framework). Not directly used here but allows compatibility.

    Returns:
        A dictionary containing the execution results:
        {
            "stdout": "Standard output captured during execution",
            "stderr": "Standard error captured during execution",
            "result": Any value assigned to the 'result' variable in the code,
            "elapsed_py_ms": Time taken for Python execution in milliseconds,
            "elapsed_wall_ms": Wall clock time taken for the entire operation,
            "session_id": The session ID (useful for subsequent calls)
        }

    Raises:
        ToolInputError: If input parameters are invalid.
        ToolError: If execution fails, times out, or errors occur in the sandbox.
        ProviderError: For issues with the browser or sandbox environment.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ProviderError("Playwright library is not installed, Python sandbox tool is unavailable.", provider="python_sandbox")

    # Input validation
    is_code_str = isinstance(code, str)
    code_repr = repr(code) # Use repr for clarity
    if not code:
        raise ToolInputError(
            "Code must be a non-empty string.",
            param_name="code",
            provided_value=code_repr
        )
    if not is_code_str:
        raise ToolInputError(
            "Code must be a string.",
            param_name="code",
            provided_value=code_repr
        )

    is_timeout_int = isinstance(timeout_ms, int)
    if not is_timeout_int:
         raise ToolInputError("timeout_ms must be an integer.", param_name="timeout_ms", provided_value=timeout_ms)
    if timeout_ms <= 0:
         raise ToolInputError("timeout_ms must be a positive integer.", param_name="timeout_ms", provided_value=timeout_ms)

    # Add validation for other boolean flags if needed
    if not isinstance(allow_network, bool):
         raise ToolInputError("allow_network must be a boolean.", param_name="allow_network", provided_value=allow_network)
    if not isinstance(allow_fs, bool):
         raise ToolInputError("allow_fs must be a boolean.", param_name="allow_fs", provided_value=allow_fs)


    # -------- normalise inputs ------------------------------------------------
    packages_normalized = packages or []
    wheels_normalized = wheels or []
    current_session_id = session_id or uuid.uuid4().hex

    # -------- obtain / create sandbox ----------------------------------------
    try:
        # Ensure helpers are awaited correctly
        sandbox_options = {
            "allow_network": allow_network,
            "allow_fs": allow_fs,
        }
        sb = await _get_sandbox(
            current_session_id,
            **sandbox_options
        )
    except Exception as e:
        # Catch ProviderError from _get_browser or _get_sandbox creation failure
        if isinstance(e, ProviderError):
            raise # Re-raise ProviderError directly
        # Catch other init errors
        error_str = str(e)
        logger.error(f"Failed to get or initialize sandbox {current_session_id}: {e}", exc_info=True)
        raise ProviderError(
            f"Failed to initialize sandbox environment for session {current_session_id}: {error_str}",
            provider="python_sandbox",
            cause=e
        ) from e

    # -------- execute & time --------------------------------------------------
    t0 = time.perf_counter()
    try:
        # Pass repl_mode=False for one-shot execution
        data = await sb.execute(code, packages_normalized, wheels_normalized, timeout_ms, repl_mode=False)
    except asyncio.TimeoutError as e: # execute should raise RuntimeError for timeout now
        logger.warning(f"Code execution timed out after {timeout_ms}ms (session: {current_session_id})")
        error_details = {"timeout_ms": timeout_ms, "session_id": current_session_id}
        raise ToolError(
            f"Code execution timed out after {timeout_ms}ms",
            error_code="execution_timeout",
            details=error_details
        ) from e
    except RuntimeError as e: # Catch runtime errors from sb.execute (includes timeout and Pyodide errors)
        error_msg = str(e)
        error_details = {"error": error_msg, "session_id": current_session_id}
        if "timed out" in error_msg.lower():
             logger.warning(f"Code execution timed out after {timeout_ms}ms (session: {current_session_id})")
             error_details["timeout_ms"] = timeout_ms # Add timeout info
             raise ToolError(
                f"Code execution timed out after {timeout_ms}ms",
                error_code="execution_timeout",
                details=error_details
             ) from e
        else:
             # Don't need full traceback for sandbox error typically
             logger.error(f"Runtime error during code execution (session: {current_session_id}): {error_msg}", exc_info=False)
             raise ToolError(
                f"Runtime error during code execution: {error_msg}",
                error_code="runtime_error",
                details=error_details
             ) from e
    except Exception as e: # Catch unexpected errors during execution call
        error_msg = str(e)
        error_details = {"error": error_msg, "session_id": current_session_id}
        logger.error(f"Unexpected error during code execution (session: {current_session_id}): {error_msg}", exc_info=True)
        raise ToolError(
            f"Unexpected error during code execution: {error_msg}",
            error_code="execution_error",
            details=error_details
        ) from e

    perf_counter_end = time.perf_counter()
    wall_duration = perf_counter_end - t0
    wall_ms = int(wall_duration * 1000)

    # Log success with appropriate details
    stdout_content = data.get("stdout", "")
    stderr_content = data.get("stderr", "")
    log_details = {
        "session_id": current_session_id,
        "elapsed_ms": wall_ms,
        "packages_count": len(packages_normalized),
        "wheels_count": len(wheels_normalized),
        "stdout_len": len(stdout_content),
        "stderr_len": len(stderr_content)
    }
    logger.success(
        f"Python code executed successfully in sandbox (session: {current_session_id})",
        emoji_key=TaskType.CODE_EXECUTION.value,
        **log_details
    )

    # -------- shape unified response -----------------------------------------
    # Ensure result is serializable - Pyodide results might be proxies
    result_val = data.get("result")
    # Basic check for common non-serializable types if needed, but rely on error handling decorator mostly
    # if isinstance(result_val, (...)): result_val = repr(result_val)

    elapsed_py = data.get("elapsed", 0)

    return {
        "stdout": stdout_content,
        "stderr": stderr_content,
        "result": result_val, # Pass through, assume serializable or handled by caller/framework
        "elapsed_py_ms": int(elapsed_py),
        "elapsed_wall_ms": wall_ms,
        "session_id": current_session_id, # Return the session ID used
        "success": True # Indicate tool success (execution inside might have failed, check stderr/result)
    }


@with_tool_metrics
@with_error_handling
async def repl_python(
    code: str,
    packages: Optional[List[str]] = None,
    wheels: Optional[List[str]] = None,
    allow_network: bool = False,
    allow_fs: bool = False,
    handle: Optional[str] = None,
    timeout_ms: int = 15_000,
    reset: bool = False,
    # Add ctx=None for compatibility
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Python code in a persistent REPL-like sandbox environment where state is maintained.

    Unlike execute_python(), this function maintains variable definitions, function definitions,
    class definitions, and other Python state between calls when you reuse the same handle.
    This creates a true interactive programming environment similar to a Python REPL or
    Jupyter notebook.

    Usage example:
        # First call - define a variable
        r1 = await repl_python(code="x = 41")
        handle = r1["handle"]

        # Second call - use the variable from the previous call
        r2 = await repl_python(code="result = x + 1", handle=handle)
        assert r2["result"] == 42

        # Reset the state
        r3 = await repl_python(code="", handle=handle, reset=True)

        # Variable x should no longer exist
        r4 = await repl_python(code="result = x + 1", handle=handle) # This will raise NameError in sandbox
        # Check r4["stderr"] for the error

    Args:
        code: The Python code to execute in the REPL sandbox. Can be empty if only resetting.
        packages: Optional list of Pyodide packages to load before execution.
        wheels: Optional list of Python wheel URLs to install via micropip.
                Only used if allow_network is True.
        allow_network: Whether to allow network access from the sandbox.
        allow_fs: Whether to allow access to the secure filesystem bridge.
        handle: Optional REPL session handle from a previous call.
                If None, a new session is created and its handle returned.
        timeout_ms: Maximum execution time in milliseconds before timeout.
        reset: If True, reset the REPL's execution namespace before running the code.
               Useful for clearing variables and definitions in a session.
        ctx: Optional context dictionary (often passed by MCP framework). Not directly used here but allows compatibility.


    Returns:
        A dictionary containing the execution results:
        {
            "stdout": "Standard output captured during execution",
            "stderr": "Standard error captured during execution",
            "result": Any value assigned to the 'result' variable in the code,
            "elapsed_py_ms": Time taken for Python execution in milliseconds,
            "elapsed_wall_ms": Wall clock time taken for the entire operation,
            "handle": The session handle (use this for subsequent REPL calls)
        }

    Raises:
        ToolInputError: If input parameters are invalid.
        ToolError: If execution fails, times out, or errors occur in the sandbox.
        ProviderError: For issues with the browser or sandbox environment.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ProviderError("Playwright library is not installed, Python REPL tool is unavailable.", provider="python_sandbox")

    # Input validation
    # Allow empty code only if reset=True
    code_repr = repr(code)
    is_code_str = isinstance(code, str)
    if not is_code_str:
        raise ToolInputError(
            "Code must be a string.",
            param_name="code",
            provided_value=code_repr
        )
    if not code:
        if not reset:
            raise ToolInputError(
                "Code must be a non-empty string, unless reset=True.",
                param_name="code",
                provided_value=code_repr
            )

    is_timeout_int = isinstance(timeout_ms, int)
    if not is_timeout_int:
         raise ToolInputError("timeout_ms must be an integer.", param_name="timeout_ms", provided_value=timeout_ms)
    if timeout_ms <= 0:
         raise ToolInputError("timeout_ms must be a positive integer.", param_name="timeout_ms", provided_value=timeout_ms)

    if not isinstance(reset, bool):
         raise ToolInputError("reset must be a boolean.", param_name="reset", provided_value=reset)
    # Add validation for other boolean flags if needed
    if not isinstance(allow_network, bool):
         raise ToolInputError("allow_network must be a boolean.", param_name="allow_network", provided_value=allow_network)
    if not isinstance(allow_fs, bool):
         raise ToolInputError("allow_fs must be a boolean.", param_name="allow_fs", provided_value=allow_fs)

    # -------- normalise inputs ------------------------------------------------
    packages_normalized = packages or []
    wheels_normalized = wheels or []
    session_id = handle or uuid.uuid4().hex

    # -------- obtain / create sandbox ----------------------------------------
    try:
        # Ensure helpers are awaited correctly
        sandbox_options = {
            "allow_network": allow_network,
            "allow_fs": allow_fs,
        }
        sb = await _get_sandbox(
            session_id,
            **sandbox_options
        )
    except Exception as e:
         # Catch ProviderError from _get_browser or _get_sandbox creation failure
         if isinstance(e, ProviderError):
             raise # Re-raise ProviderError directly
         # Catch other init errors
         error_str = str(e)
         logger.error(f"Failed to get or initialize REPL sandbox {session_id}: {e}", exc_info=True)
         raise ProviderError(
            f"Failed to initialize REPL sandbox environment for session {session_id}: {error_str}",
            provider="python_sandbox",
            cause=e
         ) from e

    # -------- reset if requested ---------------------------------------------
    if reset:
        try:
            reset_result = await sb.reset_repl_state()
            if reset_result.get("ok"):
                logger.info(f"REPL state reset successfully for session {session_id}")
            else:
                # Log warning but don't necessarily fail the whole operation
                reset_error = reset_result.get('error', 'Unknown reason')
                logger.warning(f"Failed to reset REPL state for session {session_id}: {reset_error}")
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Error occurred during REPL state reset for session {session_id}: {error_str}", exc_info=True)
            # Decide if this should be fatal or just a warning - let's warn and continue

    # -------- execute & time (only if code is provided) ----------------------
    t0 = time.perf_counter()
    data = {} # Default empty data if code is empty (only reset was performed)

    if code: # Only execute if code was actually provided
        try:
            # Pass repl_mode=True for persistent execution
            data = await sb.execute(code, packages_normalized, wheels_normalized, timeout_ms, repl_mode=True)
        except asyncio.TimeoutError as e: # execute should raise RuntimeError now
            logger.warning(f"REPL execution timed out after {timeout_ms}ms (session: {session_id})")
            error_details = {"timeout_ms": timeout_ms, "session_id": session_id}
            raise ToolError(
                f"REPL code execution timed out after {timeout_ms}ms",
                error_code="execution_timeout",
                details=error_details
            ) from e
        except RuntimeError as e: # Catch runtime errors from sb.execute (includes timeout and Pyodide errors)
            error_msg = str(e)
            error_details = {"error": error_msg, "session_id": session_id}
            if "timed out" in error_msg.lower():
                 logger.warning(f"REPL execution timed out after {timeout_ms}ms (session: {session_id})")
                 error_details["timeout_ms"] = timeout_ms
                 raise ToolError(
                    f"REPL code execution timed out after {timeout_ms}ms",
                    error_code="execution_timeout",
                    details=error_details
                 ) from e
            else:
                 logger.error(f"Runtime error during REPL execution (session: {session_id}): {error_msg}", exc_info=False)
                 raise ToolError(
                    f"Runtime error during REPL code execution: {error_msg}",
                    error_code="runtime_error",
                    details=error_details
                 ) from e
        except Exception as e: # Catch unexpected errors during execution call
            error_msg = str(e)
            error_details = {"error": error_msg, "session_id": session_id}
            logger.error(f"Unexpected error during REPL execution (session: {session_id}): {error_msg}", exc_info=True)
            raise ToolError(
                f"Unexpected error during REPL code execution: {error_msg}",
                error_code="execution_error",
                details=error_details
            ) from e

    perf_counter_end = time.perf_counter()
    wall_duration = perf_counter_end - t0
    wall_ms = int(wall_duration * 1000)

    # Log success with appropriate details
    action = "executed"
    if not code:
        action = "accessed (reset only)"

    stdout_content = data.get("stdout", "")
    stderr_content = data.get("stderr", "")
    log_details = {
        "session_id": session_id,
        "reset_requested": reset,
        "elapsed_ms": wall_ms,
        "packages_count": len(packages_normalized),
        "wheels_count": len(wheels_normalized),
        "stdout_len": len(stdout_content),
        "stderr_len": len(stderr_content)
    }
    logger.success(
        f"Python code {action} successfully in REPL sandbox (session: {session_id})",
        emoji_key=TaskType.CODE_EXECUTION.value,
        **log_details
    )

    # -------- shape unified response -----------------------------------------
    result_val = data.get("result")
    elapsed_py = data.get("elapsed", 0)

    return {
        "stdout": stdout_content,
        "stderr": stderr_content,
        "result": result_val,
        "elapsed_py_ms": int(elapsed_py),
        "elapsed_wall_ms": wall_ms,
        "handle": session_id,  # Return the handle used/created
        "success": True
    }

