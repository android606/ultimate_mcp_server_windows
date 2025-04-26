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
import playwright.async_api as pw  # type: ignore

from ultimate_mcp_server.constants import TaskType
from ultimate_mcp_server.exceptions import (
    ProviderError,
    ToolError,
    ToolInputError,
)
from ultimate_mcp_server.tools.base import BaseTool, tool, with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.python_js_sandbox")

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

_BROWSER: Optional[pw.Browser] = None
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
                delete self.pyodide.globals.get("_MCP_REPL_NS");
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
    if '_MCP_REPL_NS' not in globals():
        _MCP_REPL_NS = {{'__name__': '__main__'}}
    ns = _MCP_REPL_NS
else:
    # One-shot mode - fresh namespace each time
    ns = {{'__name__': '__main__'}}

# the next line receives the JS stringified code
code = json.loads(${{JSON.stringify(msg.code)}})

start = time.time()
try:
    with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
        exec(code, ns, ns)
    rv, ok = ns.get('result', None), True
except Exception:
    rv, ok = traceback.format_exc(), False
elapsed = (time.time() - start) * 1000

__payload = dict(
    ok=ok,
    result=rv,
    stdout=_stdout.getvalue(),
    stderr=_stderr.getvalue(),
    elapsed=elapsed,
)
__payload`;

                const wall0 = now();
                const pyRes = await self.pyodide.runPythonAsync(runner);
                Object.assign(reply, pyRes, {{ wall_ms: now() - wall0 }});
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

    page: pw.Page
    allow_network: bool = False
    allow_fs: bool = False
    ready_evt: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)  # Added timestamp
    last_used: float = field(default_factory=time.time)   # Added last used

    async def init(self):
        """Load boot HTML & wait for ready."""
        # Network interception --------------------------------------------------
        if not self.allow_network:
            async def _block(req: pw.Request):
                if req.url == f"{PYODIDE_CDN}pyodide.js":
                    await req.continue_()
                else:
                    try:
                        await req.abort()
                    except pw.Error:
                        pass
            await self.page.route("**/*", _block)

        # Boot page -------------------------------------------------------------
        await self.page.set_content(PYODIDE_BOOT_HTML)

        async def _on_msg(msg):
            if msg.type == "message" and msg.json_value().get("ready"):
                self.ready_evt.set()
        self.page.on("message", _on_msg)

        # FS bridge if allowed --------------------------------------------------
        if self.allow_fs:
            await _listen_for_mcpfs_calls(self.page)
            await self._inject_mcpfs_stub()

        await asyncio.wait_for(self.ready_evt.wait(), timeout=20)

    # ---------------------------------------------------------------------
    
    async def _inject_mcpfs_stub(self) -> None:
        """
        Creates a minimal stub module `mcpfs` inside the Pyodide interpreter.
        The stub forwards `.read_text / write_text / listdir` calls back to
        the host via postMessage → _listen_for_mcpfs_calls.
        """
        stub_code = r"""
import pyodide, types, asyncio, builtins, json
from js import self as _js               # window
_msg_id = 0
async def _roundtrip(op, *a):
    global _msg_id
    _msg_id += 1
    fut = asyncio.get_running_loop().create_future()
    def _cb(ev):
        if ev.data.get("id") == _msg_id:
            fut.set_result(pyodide.to_py(ev.data))
            _js.removeEventListener("message", _cb)
    _js.addEventListener("message", _cb)
    _js.postMessage({ "type": "mcpfs", "id": _msg_id, "op": op, "args": a })
    r = await fut
    if "error" in r: raise RuntimeError(r["error"])
    return r["result"]

m = types.ModuleType("mcpfs")
m.read_text  = lambda p: _roundtrip("read",  p)
m.write_text = lambda p, t: _roundtrip("write", p, t)
m.listdir    = lambda p: _roundtrip("list", p)
import sys; sys.modules["mcpfs"] = m
"""
        await self.page.evaluate("(p)=>self.postMessage(p)", {
            "type": "exec",
            "id": "__mcpfs_stub__",
            "code": stub_code,
            "packages": [],
            "wheels": [],
        })

    # ---------------------------------------------------------------------
    async def reset_repl_state(self) -> Dict[str, Any]:
        """Reset the REPL state in the sandbox."""
        reset_id = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        
        async def _handler(msg):
            if msg.type != "message":
                return
            data = msg.json_value()
            if data.get("id") == reset_id:
                fut.set_result(data)
        
        self.page.on("message", _handler)
        await self.page.evaluate("postMessage", {
            "type": "reset",
            "id": reset_id
        })
        
        try:
            data = await asyncio.wait_for(fut, timeout=5)
        finally:
            self.page.off("message", _handler)
            
        return data
    
    async def execute(
        self,
        code: str,
        packages: list[str] | None,
        wheels: list[str] | None,
        timeout_ms: int,
        repl_mode: bool = False,
    ) -> Dict[str, Any]:
        """Low‑level execution inside the tab."""
        # Update last used time for LRU tracking
        self.last_used = time.time()
        
        async with _GLOBAL_SEM:
            exec_id = uuid.uuid4().hex
            fut: asyncio.Future = asyncio.get_event_loop().create_future()

            async def _handler(msg):
                if msg.type != "message":
                    return
                data = msg.json_value()
                if data.get("id") == exec_id:
                    fut.set_result(data)
            self.page.on("message", _handler)
            payload = {
                "type": "exec",
                "id": exec_id,
                "code": code,
                "packages": packages or [],
                "wheels": wheels or [],
                "repl_mode": repl_mode,
            }
            await self.page.evaluate("postMessage", payload)
            try:
                data: dict[str, Any] = await asyncio.wait_for(
                    fut, timeout=timeout_ms / 1000
                )
            except asyncio.TimeoutError as e:
                raise RuntimeError("Execution timed out") from e
            finally:
                self.page.off("message", _handler)
            if not data.get("ok"):
                raise RuntimeError(data.get("error", "Pyodide error"))
            return data

###############################################################################
# Browser / sandbox lifecycle helpers – with LRU eviction
###############################################################################

async def _get_browser() -> pw.Browser:
    global _BROWSER
    if _BROWSER is None:
        logger.info("Launching headless Chromium …")
        # Get the playwright instance - properly awaited
        playwright = await pw.async_playwright()
        # Then launch the browser
        _BROWSER = await playwright.chromium.launch(
            args=["--no-sandbox"]
        )
        # ensure graceful shutdown when the host process exits
        atexit.register(lambda: asyncio.run_coroutine_threadsafe(_BROWSER.close(), asyncio.new_event_loop()))
    return _BROWSER

async def _get_sandbox(session_id: str, **kwargs) -> PyodideSandbox:
    global _GLOBAL_SEM
    if _GLOBAL_SEM is None:
        _GLOBAL_SEM = asyncio.Semaphore(GLOBAL_CONCURRENCY)
    # Re‑use existing or create new ------------------------------------------------
    sb = _PAGES.get(session_id)
    if sb is not None and not sb.page.is_closed():
        _PAGES.move_to_end(session_id)  # mark as recently used
        return sb

    # Evict LRU if over capacity -----------------------------------------------
    if len(_PAGES) >= MAX_SANDBOXES:
        victim_id, victim_sb = _PAGES.popitem(last=False)
        try:
            await victim_sb.page.close()
        except Exception:  # pragma: no cover  – best effort
            pass
        logger.info("Evicted sandbox %s", victim_id)

    # New sandbox --------------------------------------------------------------
    browser = await _get_browser()
    page = await browser.new_page()
    sb = PyodideSandbox(page=page, **kwargs)
    await sb.init()
    _PAGES[session_id] = sb
    return sb

###############################################################################
# mcpfs bridge – listens for postMessage & proxies to secure FS tool
###############################################################################

async def _listen_for_mcpfs_calls(page: pw.Page):
    async def _handler(msg):
        if msg.type != "message":
            return
        data = msg.json_value()
        if data.get("type") != "mcpfs":
            return
        call_id = data["id"]
        op = data["op"]
        args = data.get("args", [])
        payload: dict[str, Any] = {"id": call_id}
        try:
            from ultimate_mcp_server.tools import filesystem as fs  # type: ignore

            if op == "read":
                res = await fs.read_file(args[0])
                payload["result"] = res["content"][0]["text"]
            elif op == "write":
                await fs.write_file(args[0], args[1])
                payload["result"] = True
            elif op == "list":
                res = await fs.list_directory(args[0])
                payload["result"] = res["entries"]
            else:
                raise ValueError(f"Unsupported FS op: {op}")
        except Exception as exc:  # pylint: disable=broad-except
            payload["error"] = str(exc)
        await page.evaluate("postMessage", payload)
    page.on("message", _handler)

###############################################################################
# Main tool class
###############################################################################

class PythonSandboxTool(BaseTool):
    """Provides Python code execution in a secure Pyodide-based sandbox environment."""
    
    tool_name = "python_sandbox"
    description = "Execute Python code securely in a browser-based Pyodide sandbox."
    
    @tool(name="execute_python")
    @with_tool_metrics
    @with_error_handling
    async def execute_python(
        self,
        code: str,
        packages: Optional[List[str]] = None,
        wheels: Optional[List[str]] = None,
        allow_network: bool = False,
        allow_fs: bool = False,
        session_id: Optional[str] = None,
        timeout_ms: int = 15_000,
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
        # Input validation
        if not code or not isinstance(code, str):
            raise ToolInputError(
                "Code must be a non-empty string.",
                param_name="code",
                provided_value=code
            )
        
        # -------- normalise inputs ------------------------------------------------
        packages = packages or []
        wheels = wheels or []
        session_id = session_id or uuid.uuid4().hex

        # -------- obtain / create sandbox ----------------------------------------
        try:
            sb = await _get_sandbox(
                session_id,
                allow_network=allow_network,
                allow_fs=allow_fs,
            )
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize sandbox environment: {str(e)}",
                provider="pyodide_sandbox",
                cause=e
            ) from e

        # -------- execute & time --------------------------------------------------
        t0 = time.perf_counter()
        try:
            data = await sb.execute(code, packages, wheels, timeout_ms, repl_mode=False)
        except asyncio.TimeoutError as e:
            logger.warning(f"Code execution timed out after {timeout_ms}ms", exc_info=True)
            raise ToolError(
                f"Code execution timed out after {timeout_ms}ms",
                error_code="execution_timeout",
                details={"timeout_ms": timeout_ms}
            ) from e
        except RuntimeError as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                raise ToolError(
                    f"Code execution timed out after {timeout_ms}ms",
                    error_code="execution_timeout",
                    details={"timeout_ms": timeout_ms}
                ) from e
            else:
                raise ToolError(
                    f"Runtime error during code execution: {error_msg}",
                    error_code="runtime_error",
                    details={"error": error_msg}
                ) from e
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error during code execution: {error_msg}", exc_info=True)
            raise ToolError(
                f"Unexpected error during code execution: {error_msg}",
                error_code="execution_error",
                details={"error": error_msg}
            ) from e
        
        wall_ms = int((time.perf_counter() - t0) * 1000)

        # Log success with appropriate details
        logger.success(
            "Python code executed successfully in sandbox",
            emoji_key=TaskType.CODE_EXECUTION.value,
            session_id=session_id,
            elapsed_ms=wall_ms,
            packages_count=len(packages),
            output_size=len(data.get("stdout", "")) + len(data.get("stderr", ""))
        )

        # -------- shape unified response -----------------------------------------
        return {
            "stdout": data.get("stdout", ""),
            "stderr": data.get("stderr", ""),
            "result": data.get("result"),
            "elapsed_py_ms": int(data.get("elapsed", 0)),
            "elapsed_wall_ms": wall_ms,
            "session_id": session_id,
        }

    @tool(name="repl_python")
    @with_tool_metrics
    @with_error_handling
    async def repl_python(
        self,
        code: str,
        packages: Optional[List[str]] = None,
        wheels: Optional[List[str]] = None,
        allow_network: bool = False,
        allow_fs: bool = False,
        handle: Optional[str] = None,
        timeout_ms: int = 15_000,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Python code in a persistent REPL-like sandbox environment where state is maintained.

        Unlike execute_python(), this function maintains variable definitions, function definitions,
        class definitions, and other Python state between calls when you reuse the same handle.
        This creates a true interactive programming environment similar to a Python REPL or
        Jupyter notebook.

        Usage example:
            # First call - define a variable
            r1 = await repl_python("x = 41")
            handle = r1["handle"]

            # Second call - use the variable from the previous call
            r2 = await repl_python("result = x + 1", handle=handle)
            assert r2["result"] == 42

        Args:
            code: The Python code to execute in the REPL sandbox.
            packages: Optional list of Pyodide packages to load before execution.
            wheels: Optional list of Python wheel URLs to install via micropip.
                    Only used if allow_network is True.
            allow_network: Whether to allow network access from the sandbox.
            allow_fs: Whether to allow access to the secure filesystem bridge.
            handle: Optional REPL session handle from a previous call.
                    If None, a new session is created.
            timeout_ms: Maximum execution time in milliseconds before timeout.
            reset: Whether to reset the REPL state before execution.
                Useful when you want to clear all variables but keep the same session.

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
        # Input validation
        if not code or not isinstance(code, str):
            raise ToolInputError(
                "Code must be a non-empty string.",
                param_name="code",
                provided_value=code
            )
        
        # -------- normalise inputs ------------------------------------------------
        packages = packages or []
        wheels = wheels or []
        session_id = handle or uuid.uuid4().hex

        # -------- obtain / create sandbox ----------------------------------------
        try:
            sb = await _get_sandbox(
                session_id,
                allow_network=allow_network,
                allow_fs=allow_fs,
            )
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize sandbox environment: {str(e)}",
                provider="pyodide_sandbox",
                cause=e
            ) from e
        
        # -------- reset if requested ---------------------------------------------
        if reset:
            try:
                await sb.reset_repl_state()
                logger.info(f"REPL state reset for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to reset REPL state: {str(e)}", exc_info=True)
                # Continue anyway, since this is just a convenience feature

        # -------- execute & time --------------------------------------------------
        t0 = time.perf_counter()
        try:
            data = await sb.execute(code, packages, wheels, timeout_ms, repl_mode=True)
        except asyncio.TimeoutError as e:
            logger.warning(f"REPL code execution timed out after {timeout_ms}ms", exc_info=True)
            raise ToolError(
                f"REPL code execution timed out after {timeout_ms}ms",
                error_code="execution_timeout",
                details={"timeout_ms": timeout_ms}
            ) from e
        except RuntimeError as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                raise ToolError(
                    f"REPL code execution timed out after {timeout_ms}ms",
                    error_code="execution_timeout",
                    details={"timeout_ms": timeout_ms}
                ) from e
            else:
                raise ToolError(
                    f"Runtime error during REPL code execution: {error_msg}",
                    error_code="runtime_error",
                    details={"error": error_msg}
                ) from e
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error during REPL code execution: {error_msg}", exc_info=True)
            raise ToolError(
                f"Unexpected error during REPL code execution: {error_msg}",
                error_code="execution_error",
                details={"error": error_msg}
            ) from e
        
        wall_ms = int((time.perf_counter() - t0) * 1000)

        # Log success with appropriate details
        logger.success(
            "Python code executed successfully in REPL sandbox",
            emoji_key=TaskType.CODE_EXECUTION.value,
            session_id=session_id,
            elapsed_ms=wall_ms,
            packages_count=len(packages),
            output_size=len(data.get("stdout", "")) + len(data.get("stderr", ""))
        )

        # -------- shape unified response -----------------------------------------
        return {
            "stdout": data.get("stdout", ""),
            "stderr": data.get("stderr", ""),
            "result": data.get("result"),
            "elapsed_py_ms": int(data.get("elapsed", 0)),
            "elapsed_wall_ms": wall_ms,
            "handle": session_id,  # Return as "handle" for clarity
        }