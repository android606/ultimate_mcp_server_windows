"""
Smart Browser - Playwright-powered web automation tool for Ultimate MCP Server.

Provides enterprise-grade web automation with comprehensive features for scraping,
testing, and browser automation tasks with built-in security, resilience, and ML capabilities.

FEATURES
────────────────────────────────────────────────────────────────────────────────
✓ 1  Enterprise audit log (hash-chained JSONL in ~/.smart_browser/audit.log)
✓ 2  Secret vault / ENV bridge → get_secret("env:FOO") / get_secret("vault:kv/data/foo#bar")
✓ 3  Headful toggle + optional VNC remote viewing (HEADLESS=0, VNC=1)
✓ 4  Proxy rotation from PROXY_POOL for IP diversity
✓ 5  AES-GCM-encrypted cookie jar for secure persistence (SB_STATE_KEY=<b64-key>)
✓ 6  Human-like jitter on UI actions with risk-aware timing
✓ 7  Resilient "chaos-monkey" retries with idempotent re-play
✓ 8  Async TAB POOL for parallel scraping with concurrency control (SB_MAX_TABS)
✓ 9  Pluggable HTML summarizer (trafilatura ▸ readability-lxml ▸ fallback)
✓ 10 Download helper with SHA-256 verification and audit logging
✓ 11 PDF / Excel auto-table extraction after download
✓ 12 Smart element locator with multiple fallback strategies
✓ 13 Adaptive selector learning with per-site SQLite cache
✓ 14 LLM-powered page state analysis and action recommendation
✓ 15 Natural-language macro runner (ReAct-style plan → act → reflect loop)
✓ 16 Universal search across multiple engines (Yandex, Bing, DuckDuckGo)
✓ 17 Form-filling with secure credential handling
✓ 18 Element state extraction and DOM mapping
✓ 19 Multi-tab parallel URL processing
✓ 20 Browser lifecycle management with secure shutdown

Fully integrated with Ultimate MCP Server's error handling, metrics tracking,
and tool registration systems for seamless usage within the MCP ecosystem.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import concurrent.futures
import difflib
import functools
import hashlib
import json
import os
import random
import re
import signal
import sqlite3
import subprocess
import sys
import time
import threading
import queue
import unicodedata
import urllib.parse
from contextlib import asynccontextmanager, closing
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from playwright.async_api import (
    Browser,
    BrowserContext,
    Locator,
    Page,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.tools.base import BaseTool, tool, with_error_handling, with_tool_metrics
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.tools.filesystem import create_directory, write_file_content
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.smart_browser")

# Thread pool for CPU-bound tasks
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, os.cpu_count() * 2 + 4),
    thread_name_prefix="sb_worker"
)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  FILESYSTEM & ENCRYPTION & DB
# ══════════════════════════════════════════════════════════════════════════════
_HOME = Path.home() / ".smart_browser"
_HOME.mkdir(parents=True, exist_ok=True)
_STATE_FILE = _HOME / "storage_state.enc"
_LOG_FILE = _HOME / "audit.log"
_SELDB_FILE = _HOME / "selectors.db"
_last_hash: str | None = None
_log_lock = asyncio.Lock()  # Added lock for thread safety in audit logging
_db_pool = queue.Queue(maxsize=5)  # Maximum 5 concurrent connections
_db_lock = threading.RLock()  # Reentrant lock for thread safety

# Versioned cipher format - add "SB1" prefix to distinguish versions
CIPHER_VERSION = b"SB1"

# Maintain a single SQLite connection per process
_db_connection = None
_db_lock = asyncio.Lock()

def _get_db_connection():
    """Get a SQLite connection from the pool or create a new one"""
    try:
        # Try to get a connection from the pool
        return _db_pool.get(block=False)
    except queue.Empty:
        # Create a new connection if the pool is empty
        conn = sqlite3.connect(_SELDB_FILE, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

def _return_db_connection(conn):
    """Return a connection to the pool"""
    try:
        _db_pool.put(conn, block=False)
    except queue.Full:
        # Close the connection if the pool is full
        conn.close()

def _key() -> bytes | None:
    k = os.getenv("SB_STATE_KEY", "")
    if not k:
        return None
    
    try:
        decoded = base64.b64decode(k)
        # Validate key length for AES-GCM (16/24/32 bytes)
        if len(decoded) not in (16, 24, 32):
            logger.warning("Invalid AES-GCM key length. Must be 16, 24, or 32 bytes after base64 decode.")
            return None
        return decoded
    except Exception:
        logger.warning("Invalid base64 in SB_STATE_KEY")
        return None


def _enc(buf: bytes) -> bytes:  # AES-GCM optional
    k = _key()
    if not k:
        # Log warning if encryption was expected but not possible
        if os.getenv("SB_STATE_KEY"):
            logger.warning("Encryption requested but not available. Storing in cleartext.")
        return buf
    
    # Use a unique nonce for each encryption
    nonce = os.urandom(12)
    # Add version prefix to support future cipher changes
    return CIPHER_VERSION + nonce + AESGCM(k).encrypt(nonce, buf, None)


def _dec(buf: bytes) -> bytes:
    k = _key()
    if not k:
        return buf
    
    try:
        # Check for versioned format
        if buf.startswith(CIPHER_VERSION):
            # Skip version prefix
            version_len = len(CIPHER_VERSION)
            nonce = buf[version_len:version_len+12]
            ciphertext = buf[version_len+12:]
            return AESGCM(k).decrypt(nonce, ciphertext, None)
        # Legacy format (unversioned) - for backward compatibility
        else:
            return AESGCM(k).decrypt(buf[:12], buf[12:], None)
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        # Remove corrupt state file
        _STATE_FILE.unlink(missing_ok=True)
        return b"{}"  # Return empty JSON object


# ── selector-cache DB ─────────────────────────────────────────────────────────
def _init_db():
    with closing(_get_db_connection()) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS selectors(
                site TEXT NOT NULL, 
                key TEXT NOT NULL, 
                css TEXT NOT NULL DEFAULT '', 
                xpath TEXT NOT NULL DEFAULT '',
                score REAL DEFAULT 1,
                PRIMARY KEY(site, key, css, xpath)
            )"""
        )
        con.commit()


_init_db()


def _sel_key(role: str | None, name: str | None) -> str:
    # Normalize Unicode to NFC form for consistent key generation
    if role:
        role = unicodedata.normalize('NFC', role)
    if name:
        name = unicodedata.normalize('NFC', name)
    return f"{role or ''}::{name or ''}".strip(":")


async def _get_best_selectors(site: str, key: str) -> list[tuple[str | None, str | None]]:
    """Get the best selectors for a site/key combination"""
    conn = None
    try:
        conn = _get_db_connection()
        with _db_lock:
            rows = conn.execute(
                "SELECT css,xpath FROM selectors WHERE site=? AND key=? ORDER BY score DESC LIMIT 5",
                (site, key)
            ).fetchall()
        return rows
    except sqlite3.Error as e:
        logger.error(f"SQLite error in _get_best_selectors: {e}")
        return []
    finally:
        if conn:
            _return_db_connection(conn)

async def _bump_selector(site: str, key: str, css: str | None, xpath: str | None):
    """Increment the score for a selector"""
    conn = None
    try:
        conn = _get_db_connection()
        with _db_lock:
            conn.execute(
                """INSERT INTO selectors(site,key,css,xpath,score)
                VALUES (?,?,?,?,1)
                ON CONFLICT(site,key,COALESCE(css,''),COALESCE(xpath,'')) 
                DO UPDATE SET score=score+1""",
                (site, key, css, xpath)
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"SQLite error in _bump_selector: {e}")
    finally:
        if conn:
            _return_db_connection(conn)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  AUDIT LOG (hash-chained)
# ══════════════════════════════════════════════════════════════════════════════
def _sanitize_for_log(obj: Any) -> Any:
    """Sanitize values to prevent log injection attacks"""
    if isinstance(obj, str):
        # Remove newlines and control characters to prevent log injection
        return re.sub(r'[\n\r\t\0]', ' ', obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_for_log(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_log(item) for item in obj]
    return obj


async def _log(event: str, **details):
    global _last_hash
    ts = int(time.time() * 1000)
    
    # Sanitize details to prevent log injection
    sanitized_details = _sanitize_for_log(details)
    
    # Map certain events to task types - using string literals instead of enum
    emoji_key = None
    if event.startswith("browser_"):
        emoji_key = "browser"  # String literal instead of TaskType.BROWSER.value
    elif event == "navigate":
        emoji_key = "browse"  # String literal instead of TaskType.BROWSE.value
    
    # Use lock to ensure thread safety
    async with _log_lock:
        entry = {
            "ts": ts, 
            "event": event, 
            "details": sanitized_details, 
            "prev": _last_hash,
            "emoji_key": emoji_key
        }
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        h = hashlib.sha256(payload).hexdigest()
        
        # Use atomic file write for extra safety
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": h, **entry}) + "\n")
            f.flush()
            os.fsync(f.fileno())  # Ensure write is persisted to disk
            
        _last_hash = h


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESILIENT RETRY DECORATOR
# ══════════════════════════════════════════════════════════════════════════════
def resilient(max_attempts: int = 3, backoff: float = 0.3):
    """
    Decorator for async functions; retries on TimeoutError / PlaywrightError.
    """
    def wrap(fn):
        async def inner(*a, **kw):
            attempt = 0
            while True:
                try:
                    return await fn(*a, **kw)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts or isinstance(e, KeyboardInterrupt):
                        await _log("retry_fail", func=fn.__name__, err=str(e))
                        raise
                    jitter = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    await _log("retry", func=fn.__name__, attempt=attempt, sleep=round(jitter, 2))
                    await asyncio.sleep(jitter)
        return inner
    return wrap


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SECRET VAULT BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
# Allowlist for Vault paths to prevent SSRF
_ALLOWED_VAULT_PATHS = set(
    path.strip() for path in 
    os.getenv("VAULT_ALLOWED_PATHS", "secret/data/,kv/data/").split(",")
    if path.strip()
)

def get_secret(path_key: str) -> str:
    """
    Retrieve secret.  path_key formats:
        env:VAR_NAME          → reads from environment
        vault:secret/path#k   → HashiCorp Vault KV v2
    """
    if path_key.startswith("env:"):
        var = path_key[4:]
        val = os.getenv(var)
        if val is None:
            raise KeyError(f"ENV secret {var} not set")
        return val

    if path_key.startswith("vault:"):
        try:
            import hvac  # type: ignore
        except ImportError as e:
            raise RuntimeError("hvac not installed for Vault access") from e

        addr = os.getenv("VAULT_ADDR")
        token = os.getenv("VAULT_TOKEN")
        if not addr or not token:
            raise RuntimeError("VAULT_ADDR and VAULT_TOKEN env vars must be set")
        
        # Extract path and key parts securely, handling KV v1 paths
        vault_path = path_key[6:]
        
        # For KV v2, path contains '#' to separate key
        # For KV v1, path might contain '#' in the secret name
        if "#" in vault_path:
            try:
                path, key = vault_path.split("#", 1)
            except ValueError as e:
                # This shouldn't happen given the if condition, but just in case
                raise ValueError(f"Invalid vault path format: {vault_path}") from e
        else:
            # Handle KV v1 paths without separating key
            raise ValueError("Vault path must include key with # separator")
        
        # Verify path against allowlist
        allowed = False
        for allowed_prefix in _ALLOWED_VAULT_PATHS:
            if path.startswith(allowed_prefix):
                allowed = True
                break
                
        if not allowed:
            raise ValueError(f"Access to Vault path '{path}' is not allowed")
            
        client = hvac.Client(url=addr, token=token)
        if not client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

        # Extract mount point and subpath
        mount, sub = (path.split("/", 1) + [""])[:2]
        
        # Access the secret
        try:
            if "/data/" in path:  # KV v2
                data = client.secrets.kv.v2.read_secret_version(mount_point=mount, path=sub)
                return data["data"]["data"][key]
            else:  # KV v1
                data = client.secrets.kv.v1.read_secret(mount_point=mount, path=sub)
                return data["data"][key]
        except KeyError as e:
            raise KeyError(f"Key {key} not found at {path}") from e
    raise ValueError("Unknown secret scheme")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLAYWRIGHT LIFECYCLE (proxy, VNC, cookies)
# ══════════════════════════════════════════════════════════════════════════════
_pw = None
_browser: Browser | None = None
_ctx: BrowserContext | None = None
_vnc_proc: subprocess.Popen | None = None
_js_lib_cached: Set[str] = set()
_js_lib_lock = asyncio.Lock()  # Lock for preventing race conditions on script injection


async def get_browser_context(headless: bool | None = None, use_incognito: bool = False) -> tuple[BrowserContext, Browser]:
    """
    Get or create a browser context.
    
    Args:
        headless: Whether to run the browser in headless mode (without UI).
        use_incognito: Whether to create a new incognito context.
        
    Returns:
        Tuple of (BrowserContext, Browser)
    """
    global _pw, _browser, _ctx
    if _ctx and not use_incognito:
        return _ctx, _browser  # type: ignore[arg-type]

    headless_env = os.getenv("HEADLESS")
    headless = (
        True
        if headless_env is None
        else headless_env.lower() not in ("0", "false", "no") and headless_env.strip() != ""
    )
    _start_vnc()
    
    # Initialize playwright if not already done
    if not _pw:
        _pw = await async_playwright().start()
    
    # Launch browser if not already launched
    if not _browser:
        # Configure browser launch to handle DNS correctly
        _browser = await _pw.chromium.launch(
            headless=headless,
            proxy=None,  # Proxy is set per context for better control
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--window-size=1280,1024",
                "--enable-features=NetworkService,NetworkServiceInProcess",
                "--dns-prefetch-disable",  # Mitigate synchronous DNS issue
            ],
        )
        
    # Use incognito context if requested (prevents service worker leaks)
    if use_incognito:
        incognito_ctx = await _browser.new_context(
            viewport={"width": 1280, "height": 1024}
        )
        await _log("browser_incognito_context")
        return incognito_ctx, _browser
    
    # Create main context if it doesn't exist
    if not _ctx:
        proxy_setting = _choose_proxy()
        _ctx = await _browser.new_context(
            viewport={"width": 1280, "height": 1024}, 
            storage_state=_load_state() or None,
            proxy=proxy_setting if proxy_setting else None
        )
        await _log("browser_start", headless=headless, proxy=str(proxy_setting))
        
        # Set up periodic cleanup (clear cookies, storage, etc. to avoid bloat)
        asyncio.create_task(_context_maintenance_loop(_ctx))
        
    return _ctx, _browser

def _choose_proxy() -> str | None:
    pool = [p.strip() for p in os.getenv("PROXY_POOL", "").split(";") if p.strip()]
    if not pool:
        return None
        
    # Validate proxy format before returning
    proxy = random.choice(pool)
    
    # Basic validation to ensure this is a valid proxy URL
    try:
        parsed = urlparse(proxy)
        if parsed.scheme not in ('http', 'https', 'socks5', 'socks5h'):
            logger.warning(f"Invalid proxy scheme: {parsed.scheme} in {proxy}")
            return None
            
        # Handle credentials in proxy URLs properly
        # If credentials exist, reformat the URL to ensure Playwright parses it correctly
        if '@' in proxy and parsed.netloc:
            # Check for attempts to manipulate the proxy via hashtags
            if '#' in proxy:
                logger.warning(f"Invalid proxy URL containing '#': {proxy}")
                return None
                
            # Extract and rebuild to ensure proper format
            auth_part, host_part = parsed.netloc.split('@', 1)
            proxy = f"{parsed.scheme}://{host_part}"
            
            # Add credentials as separate parameters
            username, password = auth_part.split(':', 1) if ':' in auth_part else (auth_part, '')
            return {
                "server": proxy,
                "username": username,
                "password": password
            }
            
        return proxy
    except Exception as e:
        logger.warning(f"Error validating proxy URL {proxy}: {e}")
        return None


def _get_allowed_domains() -> list[str]:
    """Get list of domains allowed for proxy usage"""
    domains = os.getenv("PROXY_ALLOWED_DOMAINS", "").split(",")
    return [d.strip() for d in domains if d.strip()]


def _is_domain_allowed(url: str) -> bool:
    """Check if domain is allowed for proxy usage to prevent MITM risks"""
    allowed = _get_allowed_domains()
    if not allowed:  # If no restrictions specified, allow all
        return True
    
    domain = urlparse(url).netloc
    # Normalize domain to help prevent IDN homograph attacks
    domain = _normalize_domain(domain)
    return any(domain.endswith(d) for d in allowed)


def _start_vnc():
    global _vnc_proc
    if _vnc_proc or os.getenv("VNC") != "1":
        return
        
    # Only start VNC if explicit password is provided
    vnc_password = os.getenv("VNC_PASSWORD")
    if not vnc_password:
        logger.warning("VNC=1 but no VNC_PASSWORD set. VNC server not started for security.")
        return
        
    try:
        # Use password-protected VNC
        _vnc_proc = subprocess.Popen(
            ["x11vnc", "-display", os.getenv("DISPLAY", ":0"), 
             "-passwd", vnc_password, "-forever", "-localhost"],  # Only allow local connections
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("VNC server started with password protection")
    except FileNotFoundError:
        logger.warning("x11vnc not found")


def _cleanup_vnc():
    global _vnc_proc
    if _vnc_proc:
        _vnc_proc.terminate()
        try:
            _vnc_proc.wait(timeout=5)  # Wait for process to terminate
        except subprocess.TimeoutExpired:
            _vnc_proc.kill()  # Force kill if it doesn't terminate
        _vnc_proc = None


def _load_state() -> dict[str, Any] | None:
    if not _STATE_FILE.exists():
        return None
    try:
        return json.loads(_dec(_STATE_FILE.read_bytes()))
    except Exception as e:
        logger.error(f"Failed to load browser state: {e}")
        _STATE_FILE.unlink(missing_ok=True)
        return None


async def _save_state(ctx: BrowserContext):
    try:
        state = await ctx.storage_state()
        _STATE_FILE.write_bytes(_enc(json.dumps(state).encode()))
    except Exception as e:
        logger.error(f"Failed to save browser state: {e}")

# Context manager for tabs, ensuring proper cleanup
@asynccontextmanager
async def _tab_context(ctx: BrowserContext):
    """
    Create a new page in the given browser context and yield it, ensuring it's properly closed.
    
    Args:
        ctx: The browser context in which to create the page.
        
    Yields:
        The newly created page.
    """
    # Create a new page in the provided context
    page = await ctx.new_page()
    try:
        await _log("page_open")
        yield page
    finally:
        # Always ensure page is closed, even on exception
        if not page.is_closed():
            await page.close()
        await _log("page_close")


async def _context_maintenance_loop(ctx: BrowserContext):
    """Periodically clean context to prevent resource leaks"""
    while True:
        try:
            # Wait for 30 minutes before cleaning
            await asyncio.sleep(30 * 60)
            
            # Save state first
            await _save_state(ctx)
            
            # Clear cookies but preserve storage state
            await ctx.clear_cookies()
            
            # Clear other types of storage
            await ctx.clear_permission_overrides()
            
            # Check service workers and other resources
            pages = ctx.pages
            logger.info(f"Maintenance: {len(pages)} pages open, cleaning context")
            
        except Exception as e:
            logger.error(f"Error in context maintenance: {e}")


async def shutdown():
    global _pw, _browser, _ctx, _vnc_proc
    try:
        if _ctx:
            await _save_state(_ctx)
            await _ctx.close()
        if _browser:
            await _browser.close()
        if _pw:
            await _pw.stop()
        _cleanup_vnc()
        await _log("browser_shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        _pw = _browser = _ctx = None


# Register shutdown with atexit and signal handlers
atexit.register(lambda: asyncio.run(shutdown()))

# Register signal handlers for graceful shutdown
def _signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down...")
    asyncio.create_task(shutdown())
    # Give shutdown a chance to run, then exit
    time.sleep(2)
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TAB POOL FOR PARALLELISM
# ══════════════════════════════════════════════════════════════════════════════
class TabPool:
    """Runs callables that need a fresh Page in parallel, bounded by SB_MAX_TABS."""

    def __init__(self, max_tabs: int | None = None):
        self.sem = asyncio.Semaphore(
            max_tabs if max_tabs is not None else int(os.getenv("SB_MAX_TABS", "5"))
        )
        # Track active tasks to ensure cleanup
        self.active_tasks = set()
        self.task_lock = asyncio.Lock()

    async def _run(self, fn: Callable[[Page], Awaitable[Any]]) -> Any:
        # Create a task-specific timeout for circuit breaker effect
        timeout_seconds = int(os.getenv("SB_TAB_TIMEOUT", "300"))  # 5-minute default
        
        # Function to execute with proper resource cleanup
        async def execute_with_context():
            # Track the current task
            task = asyncio.current_task()
            if task:
                async with self.task_lock:
                    self.active_tasks.add(task)
                    
            try:
                # Acquire semaphore first
                async with self.sem:
                    # Use incognito context to prevent service worker leaks
                    ctx, _ = await get_browser_context(use_incognito=True)
                    
                    try:
                        async with _tab_context(ctx) as p:
                            return await fn(p)
                    finally:
                        # Always close the incognito context to clean up service workers
                        await ctx.close()
            except Exception as e:
                await _log("page_error", error=str(e))
                return {"error": str(e), "success": False}
            finally:
                # Always remove task from tracking
                if task:
                    async with self.task_lock:
                        self.active_tasks.discard(task)
        
        # Execute with timeout for circuit breaker effect
        try:
            return await asyncio.wait_for(execute_with_context(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            await _log("tab_timeout", timeout_seconds=timeout_seconds)
            return {"error": f"Operation timed out after {timeout_seconds}s", "success": False}

    async def map(self, fns: Sequence[Callable[[Page], Awaitable[Any]]]) -> List[Any]:
        results = []
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._run(fn)) for fn in fns]
        # Collect results in a deterministic order
        for task in tasks:
            results.append(task.result())
        return results

    async def cancel_all(self):
        """Cancel all active tasks in the pool"""
        async with self.task_lock:
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
        
        # Wait for cancellations to complete
        await asyncio.sleep(0.1)

tab_pool = TabPool()

# ══════════════════════════════════════════════════════════════════════════════
# 6.  HUMAN-LIKE JITTER (bot evasion)
# ══════════════════════════════════════════════════════════════════════════════
def _normalize_domain(domain: str) -> str:
    """Normalize domain to help prevent homograph attacks"""
    # Convert to punycode to handle IDN homographs
    try:
        return domain.encode('idna').decode('ascii').lower()
    except Exception:
        return domain.lower()


def _risk_factor(url: str) -> float:
    """
    Calculate risk factor for a domain to adjust wait times.
    Higher values for sensitive domains that employ bot detection.
    """
    d = urlparse(url).netloc
    norm_domain = _normalize_domain(d)
    
    # Risk domains from configuration or defaults
    high_risk_domains_str = os.getenv(
        "HIGH_RISK_DOMAINS", 
        "facebook.com,linkedin.com,glassdoor.com,instagram.com,twitter.com,x.com,"
        "ticketmaster.com,cloudflare.com,recaptcha.net,amazon.com"
    )
    high_risk_domains = [x.strip() for x in high_risk_domains_str.split(",") if x.strip()]
    
    # Normalize configured domains too
    high_risk = [_normalize_domain(x) for x in high_risk_domains]
    
    if any(norm_domain.endswith(x) for x in high_risk):
        return 2.0
    
    # Dynamic risk increase based on previous page interactions
    # The more interactions on a page, the longer we should wait
    # This simulates a human slowing down on complex pages
    return 1.0


async def _pause(page: Page, base_ms: tuple[int, int] = (100, 400)):
    factor = _risk_factor(page.url or "")
    
    # Add jitter based on perceived page complexity
    try:
        # Estimate page complexity based on number of interactive elements
        element_count = await page.evaluate(
            """() => {
                return document.querySelectorAll('a, button, input, select, textarea').length;
            }"""
        )
        # Scale factor based on page complexity (1.0-1.5)
        complexity_factor = min(1.0 + (element_count / 500), 1.5)
        factor *= complexity_factor
    except Exception:
        # Fallback if we can't evaluate page complexity
        pass
        
    ms = random.uniform(*base_ms) * factor
    await asyncio.sleep(ms / 1000)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SMART LOCATOR + ADAPTIVE SELECTOR LEARNING
# ══════════════════════════════════════════════════════════════════════════════
class SmartLocator:
    def __init__(self, page: Page, timeout: int = 4000):
        self.p, self.t = page, timeout
        self.site = urlparse(page.url or "").netloc

    async def _try_learned(self, key: str):
        for css, xpath in await _get_best_selectors(self.site, key):
            if css:
                try:
                    el = self.p.locator(css).first
                    await el.wait_for(state="visible", timeout=500)
                    return el
                except PlaywrightTimeoutError:
                    continue
            if xpath:
                try:
                    el = self.p.locator(f"xpath={xpath}").first
                    await el.wait_for(state="visible", timeout=500)
                    return el
                except PlaywrightTimeoutError:
                    continue
        return None
    
    async def _by_css(self, css: str | None) -> Locator | None:
        """Get element by CSS selector"""
        if not css:
            return None
        try:
            locator = self.p.locator(css).first
            await locator.wait_for(state="attached", timeout=self.t)
            return locator
        except Exception:
            return None
            
    async def _by_xpath(self, xpath: str | None) -> Locator | None:
        """Get element by XPath"""
        if not xpath:
            return None
        try:
            locator = self.p.locator(f"xpath={xpath}").first
            await locator.wait_for(state="attached", timeout=self.t)
            return locator
        except Exception:
            return None    

    async def _by_role(self, role: str | None, name: str | None) -> Locator | None:
        if not role:
            return None
        try:
            # Fix: Get locator first, then wait for it
            locator = self.p.get_by_role(role, name=name, exact=True).first
            await locator.wait_for(state="visible", timeout=self.t)
            return locator
        except PlaywrightTimeoutError:
            return None

    async def _by_label(self, name: str | None) -> Locator | None:
        if not name:
            return None
        try:
            # Fix: Get locator first, then wait for it
            locator = self.p.get_by_label(name, exact=True).first
            await locator.wait_for(state="visible", timeout=self.t)
            return locator
        except PlaywrightTimeoutError:
            return None

    async def _by_text(self, name: str | None) -> Locator | None:
        if not name:
            return None
        try:
            # Fix: Get locator first, then wait for it
            locator = self.p.get_by_text(name, exact=True).first
            await locator.wait_for(state="visible", timeout=self.t)
            return locator
        except PlaywrightTimeoutError:
            return None

    async def _fuzzy(self, name: str | None) -> Locator | None:
        if not name:
            return None
            
        # Normalize the input text for comparison
        name_norm = unicodedata.normalize('NFC', name)
            
        texts = [
            unicodedata.normalize('NFC', t.strip()[:128])
            for t in await self.p.locator("xpath=//*[normalize-space(string())!='']").all_text_contents()
        ]
        best = difflib.get_close_matches(name_norm, texts, n=1, cutoff=0.55)
        if not best:
            return None
            
        locator = self.p.locator(f"text='{best[0]}'").first
        
        # Fix: Wait for visibility before returning
        try:
            await locator.wait_for(state="visible", timeout=500)
            return locator
        except PlaywrightTimeoutError:
            return None

    async def locate(
        self, *, name: str | None = None, role: str | None = None, css: str | None = None, xpath: str | None = None
    ) -> Locator:
        """Find an element using various strategies with error handling"""
        key = _sel_key(role, name)
        
        # Try learned selectors first
        if name or role:
            try:
                learned = await self._try_learned(key)
                if learned:
                    await _log("locator_learned_hit", key=key)
                    return learned
            except Exception as e:
                logger.debug(f"Error in _try_learned: {e}")
        
        # Try various strategies with proper error handling
        strategies = [
            # Define locator lookup functions that won't return None
            lambda: self._by_role(role, name),
            lambda: self._by_label(name),
            lambda: self._by_text(name),
            lambda: self._fuzzy(name),
            lambda: self._by_css(css),
            lambda: self._by_xpath(xpath),
        ]
        
        for strategy_fn in strategies:
            try:
                element = await strategy_fn()
                if element:
                    # Record successful selector
                    try:
                        await _bump_selector(self.site, key, css or None, xpath or None)
                    except Exception as e:
                        logger.debug(f"Error in _bump_selector: {e}")
                    
                    # Log success and return element
                    try:
                        await _log("locator_success", 
                                  method=strategy_fn.__name__, 
                                  target=name or role or css or xpath)
                    except Exception as e:
                        logger.debug(f"Error in _log: {e}")
                        
                    return element
            except Exception as e:
                logger.debug(f"Error in strategy {strategy_fn.__name__}: {e}")
                continue
        
        # All strategies failed, handle gracefully
        await _log("locator_fail", target=name or role or css or xpath)
        
        # Return a default element that won't cause await errors
        fallback_selector = css or xpath or "body"
        return self.p.locator(fallback_selector)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SMART ACTIONS WITH RETRY SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

@resilient(max_attempts=3, backoff=0.5)
async def smart_click(page: Page, **kw):
    """Click on an element with retry and error handling"""
    await _pause(page)
    
    try:
        # Get the element using SmartLocator with fallback handling
        locator = await SmartLocator(page).locate(**kw)
        
        # Check if element exists and is clickable before clicking
        if await locator.count() > 0:
            # Wait for element to be visible and enabled
            await locator.wait_for(state="visible")
            await locator.click()
            await _log("click", success=True, **kw)
            return True
        else:
            # Element not found, log failure
            await _log("click", success=False, error="Element not found", **kw)
            return False
    except Exception as e:
        # Log error and return failure
        await _log("click", success=False, error=str(e), **kw)
        return False
    

@resilient(max_attempts=3, backoff=0.5)
async def smart_type(page: Page, text: str, press_enter: bool = False, **kw):
    """Type text into an element with retry and error handling"""
    await _pause(page)
    
    # Handle secret placeholders
    secret_text = False
    
    if text.startswith("secret:"):
        try:
            text = get_secret(text[7:])
            secret_text = True
        except (KeyError, ValueError) as e:
            # If secret retrieval fails, log error
            await _log("type", value="*** [secret retrieval failed]", error=str(e), enter=press_enter, **kw)
            # Keep the original text
    
    try:
        # Get the element using SmartLocator with fallback handling
        locator = await SmartLocator(page).locate(**kw)
        
        # Check if element exists before typing
        if await locator.count() > 0:
            # Wait for element to be visible and enabled
            await locator.wait_for(state="visible")
            
            # Clear the input field first
            await locator.fill("")
            
            # Type the text with a delay to appear more human-like
            await locator.type(text, delay=25)
            
            # Press Enter if requested
            if press_enter:
                await locator.press("Enter")
            
            # Always mask sensitive information in logs
            if secret_text or len(text) > 6:
                log_value = "***"
            else:
                log_value = text
                
            await _log("type", value=log_value, success=True, enter=press_enter, **kw)
            return True
        else:
            # Element not found, log failure
            await _log("type", success=False, error="Element not found", **kw)
            return False
    except Exception as e:
        # Log error and return failure
        await _log("type", success=False, error=str(e), **kw)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DOWNLOAD HELPER WITH DOCUMENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
async def _compute_hash_in_thread(data: bytes) -> str:
    """Compute SHA-256 hash in a thread to avoid blocking the event loop"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _thread_pool, lambda: hashlib.sha256(data).hexdigest()
    )


async def _extract_tables(path: Path) -> list:
    """Extract tables from document files asynchronously"""
    ext = path.suffix.lower()
    
    # Use a process pool for heavy document processing
    loop = asyncio.get_running_loop()
    
    try:
        if ext == ".pdf":
            # Use a separate process for tabula to prevent JVM memory leaks
            def extract_pdf_tables():
                import tabula  # type: ignore
                dfs = tabula.read_pdf(str(path), pages="all", multiple_tables=True)
                return [df.to_dict(orient="records") for df in dfs]
                
            return await loop.run_in_executor(_thread_pool, extract_pdf_tables)
            
        if ext in (".xls", ".xlsx"):
            def extract_excel_tables():
                import pandas as pd  # type: ignore
                xl = pd.read_excel(str(path), sheet_name=None)
                return [{ "sheet": name, "rows": df.to_dict(orient="records")} for name, df in xl.items()]
                
            return await loop.run_in_executor(_thread_pool, extract_excel_tables)
            
        if ext == ".csv":
            def extract_csv_tables():
                import pandas as pd  # type: ignore
                df = pd.read_csv(str(path))
                return [df.to_dict(orient="records")]
                
            return await loop.run_in_executor(_thread_pool, extract_csv_tables)
    except Exception as e:
        await _log("table_extract_error", file=str(path), err=str(e))
    return []


@resilient()
async def smart_download(page: Page, target: Dict[str, Any], dest_dir: str | Path | None = None) -> Dict[str, Any]:
    # Ensure dest_dir is a directory path
    if dest_dir is None:
        dest_dir = _HOME / "downloads"
    else:
        dest_dir = Path(dest_dir)
        # Ensure dest_dir isn't accidentally treating a file as a directory
        if dest_dir.suffix:  # Has extension, likely a file
            parent_dir = dest_dir.parent
            await create_directory(str(parent_dir))
            out_path_template = dest_dir  # Keep the original path as a template
        else:
            await create_directory(str(dest_dir))
            out_path_template = None
    
    # Create directory if needed
    if not out_path_template:
        await create_directory(str(dest_dir))
    
    # Download the file
    async with page.expect_download() as dl_info:
        await smart_click(page, **target)
    dl = await dl_info.value
    fname = dl.suggested_filename
    
    # Determine the output path
    if out_path_template:
        out_path = out_path_template
    else:
        out_path = dest_dir / fname
        
    # Save the file
    await dl.save_as(str(out_path))
    
    # Read file data and compute hash in a separate thread
    data = out_path.read_bytes()
    sha = await _compute_hash_in_thread(data)
    
    # Extract tables if needed (in background thread)
    tables = await _extract_tables(out_path)
    
    # Prepare response
    info = {
        "file": str(out_path), 
        "sha256": sha, 
        "size": len(data),
        "tables_extracted": bool(tables)
    }
    if tables:
        info["tables"] = tables[:3]  # avoid huge payload
    await _log("download", **info)
    return info


# ── PDF-crawler helpers ───────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^A-Za-z0-9\u0080-\uFFFF]+")

def _slugify(text: str, max_len: int = 60) -> str:
    """
    Create a URL-friendly version of a string, preserving Unicode characters.
    """
    # First normalize unicode
    text = unicodedata.normalize('NFC', text)
    # Replace spaces and punctuation with hyphens
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    # Ensure we have a valid slug
    return (slug or "file")[:max_len]


def _get_dir_slug(path: str) -> str:
    """Get a slug from directory components of the path"""
    parts = Path(urlparse(path).path).parts
    # Use last two directory parts if available
    if len(parts) > 2:
        return f"{_slugify(parts[-3], 20)}-{_slugify(parts[-2], 20)}"
    elif len(parts) > 1:
        return _slugify(parts[-2], 40)
    return ""


async def _fetch_html(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=15)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
            return r.text
    except Exception:
        pass
    return ""


def _extract_links(base: str, html: str) -> tuple[list[str], list[str]]:
    pdfs, pages = [], []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(base, a["href"])
        if href.lower().endswith(".pdf"):
            pdfs.append(href)
        elif urllib.parse.urlparse(href).netloc == urllib.parse.urlparse(base).netloc:
            # Don't add PDFs to the pages list
            if not href.lower().endswith(".pdf"):
                pages.append(href.split("#")[0])
    return pdfs, pages


# Rate limiting for crawlers
class RateLimiter:
    def __init__(self, rate_limit: float = 1.0):
        """Initialize rate limiter with requests per second"""
        self.rate_limit = rate_limit
        self.last_request = 0
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait until allowed to make a request"""
        async with self.lock:
            now = time.time()
            # Calculate time to wait
            if self.last_request > 0:
                wait_time = max(0, (1 / self.rate_limit) - (now - self.last_request))
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            self.last_request = time.time()


async def crawl_for_pdfs(
    start_url: str,
    include_regex: str | None = None,
    max_depth: int = 2,
    max_pdfs: int = 100,
) -> list[str]:
    inc_re = re.compile(include_regex, re.I) if include_regex else None
    seen_urls, pdf_urls = set(), set()  # Use set to avoid duplicates
    queue: list[tuple[str, int]] = [(start_url, 0)]
    
    # Fix: Add visit limit as safeguard against infinite crawl
    max_visits = max(max_pdfs * 10, 500)  # Reasonable upper bound
    visit_count = 0
    
    # Rate limiter for politeness (2 requests per second max)
    rate_limiter = RateLimiter(2.0)
    
    async with httpx.AsyncClient(
        follow_redirects=True, 
        timeout=30.0,
        headers={"User-Agent": "Mozilla/5.0 (compatible;)"}
    ) as cli:
        while queue and len(pdf_urls) < max_pdfs and visit_count < max_visits:
            url, depth = queue.pop(0)
            if url in seen_urls or depth > max_depth:
                continue
            seen_urls.add(url)
            visit_count += 1
            
            # Respect rate limits
            await rate_limiter.acquire()
            
            html = await _fetch_html(cli, url)
            if not html:
                continue
                
            pdfs, pages = _extract_links(url, html)
            for p in pdfs:
                # Only process new PDFs
                if p not in pdf_urls:
                    if inc_re is None or inc_re.search(p):
                        pdf_urls.add(p)
                        if len(pdf_urls) >= max_pdfs:
                            break
                            
            # Add pages breadth-first
            if depth < max_depth:
                for p in pages:
                    if p not in seen_urls:
                        queue.append((p, depth + 1))
    
    # Convert set back to list for the return value
    return list(pdf_urls)[:max_pdfs]


async def _download_pdf_http(url: str, dest_dir: Path, seq: int) -> dict[str, str | int]:
    # Create a more descriptive slug using directory info
    dir_slug = _get_dir_slug(url)
    file_slug = _slugify(Path(urllib.parse.urlparse(url).path).stem)
    
    # Create a unique name with directory context
    if dir_slug:
        name = f"{seq:03d}_{dir_slug}_{file_slug}.pdf"
    else:
        name = f"{seq:03d}_{file_slug}.pdf"
        
    # Ensure dest_dir exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / name
    
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=60.0,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/pdf,*/*",
        }
    ) as cli:
        try:
            r = await cli.get(url, timeout=60)
            if r.status_code != 200:
                raise ToolError(f"download failed – {r.status_code}")
                
            # Verify content type is PDF
            content_type = r.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
                logger.warning(f"Downloaded file might not be a PDF. Content-Type: {content_type}")
                
            # Write content and compute hash asynchronously
            out_path.write_bytes(r.content)
            sha = await _compute_hash_in_thread(r.content)
            
            await _log("download_pdf_http", url=url, file=str(out_path), sha=sha)
            return {"url": url, "file": str(out_path), "size": len(r.content), "sha256": sha}
        except Exception as e:
            logger.error(f"Error downloading PDF {url}: {e}")
            raise ToolError(f"Download failed: {str(e)}") from e


# ── OSS documentation crawler helpers ────────────────────────────────────────
_DOC_EXTS = (".html", ".htm", "/")          # what we'll allow
_DOC_STOP_PAT = re.compile(r"\.(png|jpg|jpeg|gif|svg|css|js|zip|tgz|gz)$", re.I)

def _looks_like_docs(url: str) -> bool:
    """Heuristic: URL contains 'docs'/'readthedocs' or ends with an allowed ext."""
    u = url.lower()
    return (
        ("readthedocs" in u) or
        ("/docs" in u) or
        u.endswith(_DOC_EXTS)
    ) and not _DOC_STOP_PAT.search(u)

async def _pick_docs_root(pkg: str) -> str | None:
    """Use existing search_web to grab the best docs link."""
    hits = await search_web(f"{pkg} documentation", engine="bing", max_results=15)
    for h in hits:
        if _looks_like_docs(h["url"]):
            return h["url"]
    return hits[0]["url"] if hits else None

async def _grab_readable(client: httpx.AsyncClient, url: str) -> str:
    """Fetch HTML & run existing summariser for readability."""
    try:
        r = await client.get(url, timeout=15)
        if r.status_code != 200:
            return ""
        return _summarize_html(r.text, max_len=20_000)  # reuse helper from section 10
    except Exception:
        return ""

async def crawl_docs_site(root: str, max_pages: int = 40) -> list[tuple[str, str]]:
    """
    BFS within the same domain, limited to *max_pages*;
    returns list of (url, readable_text).
    """
    start_netloc = urllib.parse.urlparse(root).netloc
    seen, q, out = set(), [root], []
    
    # Fix: Add visit limit as safeguard against infinite crawl
    max_visits = max(max_pages * 5, 200)
    visit_count = 0
    
    # Rate limiter (3 requests per second max)
    rate_limiter = RateLimiter(3.0)
    
    async with httpx.AsyncClient(
        follow_redirects=True, 
        headers={"User-Agent": "Mozilla/5.0 (compatible;)"},
        timeout=20.0
    ) as cli:
        while q and len(out) < max_pages and visit_count < max_visits:
            url = q.pop(0)
            if url in seen:
                continue
            seen.add(url)
            visit_count += 1
            
            # Respect rate limits
            await rate_limiter.acquire()
            
            text = await _grab_readable(cli, url)
            if text:
                out.append((url, text))
                
            # simple link extraction
            try:
                # Respect rate limits again for the HTML fetch
                await rate_limiter.acquire()
                
                html = await cli.get(url, timeout=10)
                soup = BeautifulSoup(html.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = urllib.parse.urljoin(url, a["href"])
                    if urllib.parse.urlparse(href).netloc == start_netloc and _looks_like_docs(href):
                        if len(out) + len(q) < max_pages and href not in seen:
                            q.append(href)
            except Exception:
                pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PAGE STATE EXTRACTION WITH HTML SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════════
_HELPER_CDN = "https://cdn.jsdelivr.net/npm/htmlparser@1.7.7/lib/htmlparser.min.js"
_CDN_VERSION_CHECK = """
if (typeof HtmlParser === 'undefined' || typeof HtmlParser.version !== 'string' || 
    !HtmlParser.version.startsWith('1.0.')) {
    console.error('Invalid HtmlParser version');
    throw new Error('Library version mismatch');
}
"""

async def _ensure_helper(page: Page):
    async with _js_lib_lock:  # Prevent race conditions
        if _HELPER_CDN in _js_lib_cached:
            return
        
        try:
            # Fix: Use a more reliable approach for script loading
            await page.evaluate(f"""
                () => {{
                    return new Promise((resolve, reject) => {{
                        if (window._helperLoaded) {{
                            resolve(true);
                            return;
                        }}
                        
                        const script = document.createElement('script');
                        script.src = '{_HELPER_CDN}';
                        script.onload = () => {{
                            try {{
                                if (typeof HtmlParser === 'undefined') {{
                                    reject(new Error('HtmlParser not defined'));
                                    return;
                                }}
                                window._helperLoaded = true;
                                resolve(true);
                            }} catch (e) {{
                                reject(e);
                            }}
                        }};
                        script.onerror = (e) => reject(new Error('Script load error'));
                        document.head.appendChild(script);
                    }});
                }}
            """)
            _js_lib_cached.add(_HELPER_CDN)
        except Exception as e:
            logger.error(f"Helper script loading failed: {e}")
            # Continue without the helper - will fall back to basic functionality


def _summarize_html(html: str, max_len: int = 4000) -> str:
    """
    Extract readable content from HTML with memory limits.
    """
    # Set a reasonable limit for initial HTML processing
    if len(html) > 2_000_000:  # 2MB max
        html = html[:2_000_000]
        
    try:
        import trafilatura  # type: ignore
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted:
            return extracted[:max_len]
    except Exception:
        pass
        
    try:
        from readability import Document  # type: ignore
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        
        # Process in chunks to limit memory usage
        text = ""
        chunk_size = 100_000  # 100KB chunks
        for i in range(0, len(summary_html), chunk_size):
            chunk = summary_html[i:i+chunk_size]
            text += re.sub(r"<[^>]+>", " ", unescape(chunk))
            
            # Early exit if we have enough text
            if len(text) >= max_len * 2:
                break
                
        # Clean up whitespace and trim
        return re.sub(r"\s+", " ", text).strip()[:max_len]
    except Exception:
        pass
        
    # Fallback: process raw body text in chunks
    text = ""
    chunk_size = 50_000  # 50KB chunks for raw HTML
    for i in range(0, min(len(html), 1_000_000), chunk_size):
        chunk = html[i:i+chunk_size]
        text += re.sub(r"<[^>]+>", " ", unescape(chunk))
        
        # Early exit if we have enough text
        if len(text) >= max_len * 2:
            break
            
    return re.sub(r"\s+", " ", text).strip()[:max_len]


async def get_page_state(page: Page, max_nodes: int = 120) -> dict[str, Any]:
    """
    Extract the current state of the page including selectable elements.
    Independent of external JavaScript libraries.
    """
    try:
        # Get page HTML
        html = await page.content()
        
        # Extract text summary from HTML
        summary = _summarize_html(html)
        
        # Extract interactive elements using pure JavaScript
        elems = await page.evaluate(
            f"""() => {{
                const vis = e => {{
                    const r = e.getBoundingClientRect();
                    return r.width && r.height && r.top < window.innerHeight && r.left < window.innerWidth;
                }};
                
                let out = [], id = 0;
                document.querySelectorAll('a,button,input,textarea,select,[role]').forEach(el => {{
                    if (out.length >= {max_nodes} || !vis(el)) return;
                    
                    const o = {{
                        id: `el_${{id++}}`,
                        tag: el.tagName.toLowerCase(),
                        role: el.getAttribute('role') || '',
                        text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 120)
                    }};
                    
                    if (el.href) o.href = el.href.slice(0, 300);
                    out.push(o);
                }});
                
                return out;
            }}"""
        )
        
        return {
            "url": page.url, 
            "title": await page.title(), 
            "elements": elems, 
            "text_summary": summary
        }
    except Exception as e:
        logger.error(f"Error in get_page_state: {e}")
        # Return minimal page state to avoid further errors
        return {
            "url": page.url or "",
            "title": await page.title() or "Unknown",
            "elements": [],
            "text_summary": "Error extracting page content"
        }

# ══════════════════════════════════════════════════════════════════════════════
# 11.  LLM BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
async def _call_llm(
    messages: Sequence[dict[str, str]], model: str = "gpt-4o", expect_json: bool = False, temperature: float = 0.1
) -> dict[str, Any]:
    """Call LLM with improved JSON handling"""
    prompt = messages[-1]["content"]
    try:
        # Add explicit JSON instructions to the prompt
        if expect_json:
            prompt = f"{prompt}\n\nIMPORTANT: Your response must be valid JSON. Do not include any text outside the JSON. Format your response as either a JSON object (starting with {{ and ending with }}) or a JSON array (starting with [ and ending with ])."
            
        resp = await generate_completion(
            provider=Provider.OPENAI.value,
            model=model, 
            prompt=prompt, 
            temperature=temperature
        )
        
        if not resp.get("success"):
            return {"error": resp.get("error", "LLM failure")}
            
        txt = resp["text"]
        
        if not expect_json:
            return {"text": txt}
        
        # Extract JSON using progressive methods
        try:
            # First try direct parsing
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON with regex
                json_matches = re.findall(r'(\{.+?\}|\[.+?\])', txt, re.DOTALL)
                if json_matches:
                    for match in json_matches:
                        try:
                            return json.loads(match)
                        except json.JSONDecodeError:
                            continue
                            
                # If no valid JSON found, return error
                return {"error": "Could not parse JSON from response", "raw": txt[:1000]}
        except Exception as e:
            return {"error": f"JSON parsing error: {e}", "raw": txt[:1000]}
    except Exception as e:
        logger.error(f"LLM call error: {str(e)}")
        return {"error": f"LLM call error: {str(e)}"}

async def ask_llm(page: Page, user_instruction: str, model: str = "gpt-4o") -> dict[str, Any]:
    state = await get_page_state(page)
    messages = [
        {"role": "system", "content": "You are a web-browsing assistant that answers with compact JSON."},
        {
            "role": "user",
            "content": f"PAGE_STATE:\n{json.dumps(state)}\n\nTASK:\n{user_instruction}\n\n"
            'Respond ONLY with JSON like {"action":"click","selector":"..."} etc.',
        },
    ]
    return await _call_llm(messages, model=model, expect_json=True)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  NATURAL-LANGUAGE MACRO RUNNER
# ══════════════════════════════════════════════════════════════════════════════
ALLOWED_ACTIONS = {"click", "type", "wait", "download", "extract", "finish"}


async def _plan(page_state: dict[str, Any], task: str, model: str = "gpt-4o") -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You are a planning agent that outputs ONLY valid JSON."},
        {
            "role": "user",
            "content": f"""Page state:
{json.dumps(page_state)}

Task: "{task}"

Return a JSON list of steps. Each step is an object with:
  action: one of {sorted(ALLOWED_ACTIONS)}
  target: {{name/role/css/xpath...}}  (omit for wait/finish)
  [text]: string (for type)
  [enter]: bool
  [ms]: integer (for wait)
  [dest]: string (for download)
Example:
[{{"action":"click","target":{{"role":"button","name":"Accept"}}}},
 {{"action":"finish"}}]""",
        },
    ]
    
    # Set expect_json to True for proper parsing
    result = await _call_llm(messages, model=model, expect_json=True)
    if "error" in result:
        raise RuntimeError(f"Planner error: {result['error']}")
    
    # Handle different return formats
    steps = []
    if isinstance(result, list):
        steps = result
    elif isinstance(result, dict) and "steps" in result:
        steps = result["steps"]
    else:
        raise RuntimeError("Unexpected plan format from LLM")
    
    # Validate actions
    for st in steps:
        if not isinstance(st, dict) or "action" not in st:
            raise ValueError(f"Invalid step format: {st}")
        if st["action"] not in ALLOWED_ACTIONS:
            raise ValueError(f"Illegal action {st['action']}")
    
    return steps


async def run_macro(page: Page, task: str, max_rounds: int = 5, model: str = "gpt-4o"):
    results = []
    
    try:
        for i in range(max_rounds):
            state = await get_page_state(page)
            plan = await _plan(state, task, model)
            await _log("macro_plan", round=i + 1, steps=plan)
            
            # Execute the plan
            step_results = await run_steps(page, plan)
            results.extend(step_results)
            
            # Check if we're done
            if any(s["action"] == "finish" for s in plan):
                await _log("macro_complete", rounds=i + 1)
                return results
                
        # If we get here, we ran out of rounds
        await _log("macro_exceeded_rounds", max_rounds=max_rounds)
        return results
    except Exception as e:
        # Log the error and ensure we return any results we have
        await _log("macro_error", error=str(e))
        if not results:
            raise  # Re-raise if we have no results
        return results


# ── autopilot planner ────────────────────────────────────────────────────────
# Tool references without "self." prefix
_AVAILABLE_TOOLS = {
    # tool-name          : (callback_ref, brief arg schema)
    "search"             : ("search",         {"query": "str", "engine": "str"}),
    "browse"             : ("browse_url",     {"url": "str"}),
    "click"              : ("click_and_extract", {"url":"str","target":"dict"}),
    "fill_form"          : ("fill_form",      {"url":"str","form_fields":"list"}),
    "download"           : ("download_file",  {"url":"str","target":"dict","dest_dir":"str"}),
    "run_macro"          : ("execute_macro",  {"url":"str","task":"str"}),
    "download_site_pdfs" : ("download_site_pdfs", {"start_url":"str"}),
}

_PLANNER_SYS = (
    "You are an orchestrator that maps a user TASK into a JSON list of steps.\n"
    "Only use the tools listed, keep args minimal JSON."
)

async def _plan_autopilot(task: str, prior_results: list = None) -> list[dict[str, Any]]:
    tools_desc = {k:v[1] for k,v in _AVAILABLE_TOOLS.items()}
    
    # Include prior results in prompt if available
    prior_results_str = ""
    if prior_results:
        prior_results_str = f"\n\nPrior step results:\n{json.dumps(prior_results, indent=2)}\n"
    
    plan_req = (
        f"TOOLS:\n{json.dumps(tools_desc)}\n\n"
        f"TASK:\n{task}\n{prior_results_str}\n"
        'Return ONLY JSON like [{"tool":"search","args":{"query":"..."}}, ...]'
    )
    msg = [{"role":"system","content":_PLANNER_SYS},{"role":"user","content":plan_req}]
    resp = await _call_llm(msg, expect_json=True)
    
    # Handle response format
    if isinstance(resp, list):
        return resp
    elif isinstance(resp, dict):
        if "error" in resp:
            raise ToolError(f"Plan generation failed: {resp['error']}")
        elif "steps" in resp:
            return resp["steps"]
    
    raise ToolError("Unexpected plan format from LLM")


# ══════════════════════════════════════════════════════════════════════════════
# 13.  STEP RUNNER
# ══════════════════════════════════════════════════════════════════════════════
async def run_steps(page: Page, steps: Sequence[dict[str, Any]]):
    results = []
    loc = SmartLocator(page)
    
    for s in steps:
        step_result = dict(s)  # Copy the step to preserve action, etc.
        act = s["action"]
        
        try:
            if act == "click":
                await (await loc.locate(**s["target"])).click()
            elif act == "type":
                await smart_type(page, s["text"], press_enter=s.get("enter", False), **s["target"])
            elif act == "wait":
                await page.wait_for_timeout(int(s.get("ms", 1000)))
            elif act == "download":
                step_result["result"] = await smart_download(page, s["target"], s.get("dest"))
            elif act == "extract":
                step_result["result"] = await page.eval_on_selector_all(
                    s["selector"], "(els)=>els.map(e=>e.innerText)")
            elif act == "finish":
                pass
            else:
                raise ValueError(f"Unknown action {act}")
                
            step_result["success"] = True
        except Exception as e:
            step_result["success"] = False
            step_result["error"] = str(e)
            
        await _log("step", action=act, detail=step_result)
        results.append(step_result)
        
        # Stop processing if a step failed
        if not step_result.get("success", True) and act != "wait":
            break
            
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 14.  UNIVERSAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
async def search_web(
    query: str, engine: str = "bing", max_results: int = 10
) -> List[Dict[str, str]]:
    engine = engine.lower()
    qs = urllib.parse.quote_plus(query)
    
    # Updated URLs with randomized parameters to avoid detection
    timestamp = int(time.time()*1000)
    urls = {
        "bing": f"https://www.bing.com/search?q={qs}&count={max_results}&t={timestamp}",
        "duckduckgo": f"https://html.duckduckgo.com/html/?q={qs}&t={timestamp}",  # Use HTML version which is more reliable
        "yandex": f"https://yandex.com/search/?text={qs}&ncrnd={timestamp}",
    }
    
    # More robust selectors for each engine
    selectors = {
        "bing": ("li.b_algo", "h2>a", "h2>a", ".b_caption p"),
        "duckduckgo": (".result", "a.result__a", "a.result__a", ".result__snippet"),
        "yandex": (".serp-item", ".organic__url", ".organic__title-link", ".text-container"),
    }
    
    # Add user agents to look more like regular browsers
    user_agents = {
        "bing": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "duckduckgo": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        "yandex": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    }
    
    if engine not in urls:
        raise ToolError(
            f"Invalid search engine: {engine}",
            error_code="invalid_engine", 
            details={"valid_engines": list(urls.keys())}
        )
    
    # Use a dedicated incognito context for search with appropriate user agent
    browser_context_args = {  # noqa: F841
        "user_agent": user_agents.get(engine, user_agents["bing"]),
        "viewport": {"width": 1366, "height": 768},
    }
    
    ctx, _ = await get_browser_context(use_incognito=True)
    p = await ctx.new_page()
    
    try:
        # Set user agent
        await p.set_extra_http_headers({"User-Agent": user_agents.get(engine, user_agents["bing"])})
        
        # Add a small delay before visiting to appear more human-like
        await asyncio.sleep(0.5 + random.random())
        
        await p.goto(urls[engine], wait_until="domcontentloaded", timeout=30000)
        # Wait a bit for scripts to execute
        await asyncio.sleep(1.5 + random.random())
        
        # Try to handle consent dialogs
        try:
            for consent_btn in [
                "button:has-text('Accept')", 
                "button:has-text('I agree')",
                "button:has-text('Accept all')", 
                "button.cookie-consent__button",
                "button.consent-button"
            ]:
                if await p.locator(consent_btn).count() > 0:
                    await p.locator(consent_btn).click(timeout=2000)
                    await p.wait_for_load_state("networkidle")
                    break
        except Exception as e:
            logger.debug(f"No consent popup or error handling it: {e}")
        
        # Different result extraction based on engine
        if engine == "duckduckgo":
            # DuckDuckGo HTML version has a different structure
            results = await p.evaluate("""
                () => {
                    const results = [];
                    document.querySelectorAll('.result').forEach(el => {
                        const linkEl = el.querySelector('.result__a');
                        const snippetEl = el.querySelector('.result__snippet');
                        if (linkEl && linkEl.href) {
                            results.push({
                                url: linkEl.href,
                                title: linkEl.innerText.trim(),
                                snippet: snippetEl ? snippetEl.innerText.trim() : ''
                            });
                        }
                    });
                    return results;
                }
            """)
        else:
            # Use the selectors for other engines
            selector_parts = selectors[engine]
            main_selector = selector_parts[0]
            link_selector = selector_parts[1]
            title_selector = selector_parts[2]
            snippet_selector = selector_parts[3]
            
            results = await p.evaluate(f"""
                () => {{
                    const elements = Array.from(document.querySelectorAll('{main_selector}'));
                    return elements.slice(0, {max_results}).map(el => {{
                        const link = el.querySelector('{link_selector}');
                        const titleEl = el.querySelector('{title_selector}');
                        const snippetEl = el.querySelector('{snippet_selector}');
                        if (!link || !link.href || !link.href.startsWith('http')) return null;
                        return {{
                            url: link.href,
                            title: (titleEl ? titleEl.innerText : (link.innerText || '')).trim(),
                            snippet: (snippetEl ? snippetEl.innerText : '').trim()
                        }};
                    }}).filter(Boolean);
                }}
            """)
        
        # Filter out empty results
        results = [r for r in results if r and isinstance(r, dict) and "url" in r]
        
        if not results and engine == "yandex":
            # Check for CAPTCHA
            if await p.locator("form:has(img[src*='captcha'])").count() > 0:
                await _log("search_captcha", engine=engine)
                raise ToolError("Search engine shows CAPTCHA. Try a different engine.")
        
        await _log("search", engine=engine, q=query, n=len(results))
        return results
    finally:
        await p.close()
        await ctx.close()


# Improved direct file download function
async def _download_file_direct(url: str, dest_dir: Path, seq: int = 1) -> dict:
    """Download a file directly using httpx with better error handling"""
    try:
        # Parse URL components for better filename generation
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # Generate a unique filename if one can't be determined
        if not filename or not filename.strip():
            filename = f"file_{seq:03d}.pdf"
        elif not filename.lower().endswith('.pdf') and url.lower().endswith('.pdf'):
            filename = f"{filename}.pdf"
            
        # Create output path
        output_path = dest_dir / filename
        
        # Download with appropriate headers and timeout
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=60.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/pdf,*/*",
            }
        ) as client:
            response = await client.get(url)
            
            # Verify successful response
            if response.status_code != 200:
                return {
                    "url": url,
                    "error": f"HTTP error: {response.status_code}",
                    "success": False
                }
                
            # Write content to file
            with open(output_path, 'wb') as f:
                f.write(response.content)
                
            # Calculate hash
            file_hash = hashlib.sha256(response.content).hexdigest()
            
            return {
                "url": url,
                "file": str(output_path),
                "size": len(response.content),
                "sha256": file_hash,
                "success": True
            }
            
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "success": False
        }
    
# ══════════════════════════════════════════════════════════════════════════════
# 15.  TOOL CLASS FOR MCP SERVER INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
class SmartBrowserTool(BaseTool):
    """
    Powerful web automation tool with enterprise features.
    
    Features include:
    - Web browsing with proxy rotation and cookie persistence
    - Smart element location with machine learning
    - Human-like interaction with jitter for bot evasion
    - Natural language macro execution
    - Parallel tab processing
    - Web search capabilities
    - Document downloading and table extraction
    """
    
    tool_name = "smart_browser"
    description = "Powerful web automation tool with enterprise features."
    
    def __init__(self, mcp_server):
        """Initialize the tool with MCP server"""
        super().__init__(mcp_server)
        self.tab_pool = TabPool()
        self._last_activity = time.time()
        
        # Store task references
        self._inactivity_monitor_task = None
        self._browser_init_task = None
        
        # Check if we're in a server context (has lifespan)
        has_lifespan = hasattr(mcp_server, 'lifespan') if hasattr(mcp_server, 'mcp') else hasattr(mcp_server, 'lifespan')
        
        # Initialize async components safely, suppressing warning if we're in a server context
        # since the browser will be initialized during server startup
        self._init_async_components(suppress_warning=has_lifespan)
    
    def _init_async_components(self, suppress_warning=False):
        """Initialize async components safely without requiring an event loop"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we have a running loop, schedule tasks immediately
            self._inactivity_monitor_task = loop.create_task(self._inactivity_monitor())
            self._browser_init_task = loop.create_task(self._ensure_browser())
            logger.info("SmartBrowserTool async components initialized")
        except RuntimeError:
            # No running event loop - log a warning but continue
            if not suppress_warning:
                logger.warning("No running event loop detected for SmartBrowserTool initialization. " 
                             "Browser features will be initialized when first used.")
    
    async def _ensure_browser(self):
        """Ensure browser is initialized"""
        try:
            await get_browser_context()
            if not self._inactivity_monitor_task:
                # If we're initializing now, also initialize the inactivity monitor
                self._inactivity_monitor_task = asyncio.create_task(self._inactivity_monitor())
                logger.info("Initialized inactivity monitor during delayed browser initialization")
        except Exception as e:
            logger.error(f"Error initializing browser: {e}", exc_info=True)
    
    @tool(name="smart_browser.browse")
    @with_tool_metrics
    @with_error_handling
    async def browse_url(self, 
                         url: str, 
                         wait_for_selector: Optional[str] = None,
                         wait_for_navigation: bool = True) -> Dict[str, Any]:
        """
        Navigate to a URL and return information about the page.
        
        Args:
            url: URL to navigate to
            wait_for_selector: Optional CSS selector to wait for before considering page loaded
            wait_for_navigation: Whether to wait for network idle
            
        Returns:
            Dictionary with page information, including title, URL, and content summary
        """
        self._last_activity = time.time()
        if not url.startswith("http"):
            url = "https://" + url
            
        ctx, _ = await get_browser_context()
        
        async with _tab_context(ctx) as page:
            await _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle" if wait_for_navigation else "domcontentloaded")
            
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            
            state = await get_page_state(page)
                
            return {
                "success": True,
                "page_state": state
            }
    
    async def _inactivity_monitor(self):
        """Monitors browser inactivity and triggers shutdown."""
        # Configurable timeout (e.g., 10 minutes = 600 seconds)
        inactivity_timeout = int(os.getenv("SB_INACTIVITY_TIMEOUT", "600"))
        check_interval = 60 # Check every 60 seconds

        logger.info(f"Starting inactivity monitor. Timeout: {inactivity_timeout}s")

        while True:
            await asyncio.sleep(check_interval)
            # Check if browser is still active (might have been shut down manually/externally)
            global _browser
            if not _browser or not _browser.is_connected():
                logger.info("Inactivity monitor: Browser already closed. Exiting monitor.")
                break

            now = time.time()
            idle_time = now - self._last_activity
            # logger.debug(f"Inactivity check: Idle for {idle_time:.1f}s") # Optional debug log

            if idle_time > inactivity_timeout:
                logger.info(f"Browser inactive for over {inactivity_timeout}s. Shutting down.")
                should_exit = False
                try:
                    # Use create_task to avoid blocking the monitor loop itself
                    # if shutdown takes time
                    asyncio.create_task(shutdown())
                except Exception as e:
                    logger.error(f"Error during automatic shutdown: {e}")
                finally:
                    # Signal loop exit after finally block executes
                    should_exit = True
                
                # Exit the monitor loop once shutdown is initiated/attempted
                if should_exit:
                    break
        logger.info("Inactivity monitor stopped.")

    @tool(name="smart_browser.click")
    @with_tool_metrics
    @with_error_handling
    async def click_and_extract(self, 
                               url: str, 
                               target: Dict[str, Any],
                               wait_ms: int = 1000) -> Dict[str, Any]:
        """
        Navigate to a URL, click on a target element, and extract the resulting page state.
        
        Args:
            url: URL to navigate to
            target: Target element specification (can include name, role, css, or xpath)
            wait_ms: Milliseconds to wait after clicking
            
        Returns:
            Dictionary with page state after clicking
        """
        self._last_activity = time.time()
        ctx, _ = await get_browser_context()
        
        async with _tab_context(ctx) as page:
            await _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle")
            
            # Click the target
            await smart_click(page, **target)
            
            # Wait for specified time
            await page.wait_for_timeout(wait_ms)
            
            # Get page state
            state = await get_page_state(page)
            
            return {
                "success": True,
                "page_state": state
            }
    
    @tool(name="smart_browser.fill_form")
    @with_tool_metrics
    @with_error_handling
    async def fill_form(self, 
                        url: str, 
                        form_fields: List[Dict[str, Any]],
                        submit_button: Optional[Dict[str, Any]] = None,
                        return_result: bool = True) -> Dict[str, Any]:
        """
        Navigate to a URL and fill out a form.
        
        Args:
            url: URL to navigate to
            form_fields: List of field specifications, each with 'target' (locator) and 'text' (value)
            submit_button: Optional submit button specification
            return_result: Whether to return the resulting page state
            
        Returns:
            Dictionary with form submission result
        """
        self._last_activity = time.time()
        ctx, _ = await get_browser_context()
        
        async with _tab_context(ctx) as page:
            await _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle")
            
            # Fill each field
            for field in form_fields:
                if not isinstance(field, dict) or "target" not in field or "text" not in field:
                    raise ToolInputError("Each form field must have 'target' and 'text' keys")
                
                target = field["target"]
                text = field["text"]
                press_enter = field.get("press_enter", False)
                
                await smart_type(page, text, press_enter=press_enter, **target)
            
            # Submit form if a submit button is provided
            if submit_button:
                await smart_click(page, **submit_button)
                # Wait for navigation
                await page.wait_for_load_state("networkidle")
            
            result = {"success": True, "form_submitted": True}
            
            # Return page state if requested
            if return_result:
                result["page_state"] = await get_page_state(page)
                
            return result
    
    @tool(name="smart_browser.search")
    @with_tool_metrics
    @with_error_handling
    async def search(self, 
                    query: str, 
                    engine: str = "yandex",
                    max_results: int = 10) -> Dict[str, Any]:
        """
        Perform a web search using the specified search engine.
        
        Args:
            query: Search query
            engine: Search engine to use (yandex, bing, or duckduckgo)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        self._last_activity = time.time()
        if engine not in ("yandex", "bing", "duckduckgo"):
            raise ToolInputError("Engine must be one of: yandex, bing, duckduckgo")
        
        results = await search_web(query, engine, max_results)
        
        return {
            "success": True,
            "query": query,
            "engine": engine,
            "results": results,
            "result_count": len(results)
        }
    
    @tool(name="smart_browser.download")
    @with_tool_metrics
    @with_error_handling
    async def download_file(self, 
                           url: str, 
                           target: Dict[str, Any],
                           dest_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Navigate to a URL and download a file by clicking on a target element.
        
        Args:
            url: URL to navigate to
            target: Target element specification (can include name, role, css, or xpath)
            dest_dir: Optional destination directory for the download
            
        Returns:
            Dictionary with download information including file path and hash
        """
        self._last_activity = time.time()
        ctx, _ = await get_browser_context()
        
        async with _tab_context(ctx) as page:
            await _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle")
            
            # Download the file
            download_info = await smart_download(page, target, dest_dir)
            
            return {
                "success": True,
                "download": download_info
            }
    
    @tool(name="smart_browser.download_site_pdfs")
    @with_tool_metrics
    @with_error_handling
    async def download_site_pdfs(
        self,
        start_url: str,
        dest_subfolder: str = "site_pdfs",
        include_regex: Optional[str] = None,
        max_depth: int = 2,
        max_pdfs: int = 100,
    ) -> Dict[str, Any]:
        """
        Crawl *start_url* for PDF links, optionally filtered by *include_regex*,
        download them (HTTP, no browser), and save under
        ~/.smart_browser/downloads/<dest_subfolder>/.
        """
        self._last_activity = time.time()
        
        # Get PDF URLs either directly or by crawling
        if start_url.lower().endswith('.pdf'):
            # Direct PDF URL provided
            pdf_urls = [start_url]
        else:
            # Crawl for PDFs
            pdf_urls = await crawl_for_pdfs(start_url, include_regex, max_depth, max_pdfs)
        
        if not pdf_urls:
            return {"success": True, "pdf_count": 0, "files": [], "dest_dir": ""}

        # Create destination directory
        dest_dir = _HOME / "downloads" / dest_subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Define download task
        async def download_pdf(seq_url):
            seq, url = seq_url
            try:
                # Use a more reliable download method
                file_info = await _download_file_direct(url, dest_dir, seq)
                return file_info
            except Exception as e:
                await _log("download_pdf_error", url=url, err=str(e))
                return {"url": url, "error": str(e), "success": False}

        # Use rate limiting for downloads
        limiter = RateLimiter(1.0)  # 1 request per second
        tasks = []
        
        for i, url in enumerate(pdf_urls, 1):
            await limiter.acquire()
            tasks.append(download_pdf((i, url)))
            
        # Execute downloads in parallel
        files = await asyncio.gather(*tasks)
        
        # Filter successful downloads
        successful_files = [f for f in files if "file" in f and f.get("success", False)]

        return {
            "success": True,
            "pdf_count": len(successful_files),
            "dest_dir": str(dest_dir),
            "files": files,
        }


    @tool(name="smart_browser.collect_documentation")
    @with_tool_metrics
    @with_error_handling
    async def collect_documentation(
        self,
        package: str,
        max_pages: int = 40,
    ) -> Dict[str, Any]:
        """
        Auto-discovers and harvests the public documentation site of an
        open-source package/library/tool, concatenating readable text from
        up to *max_pages* pages into a scratch file (--- separator between pages).

        Returns the scratch file path plus basic stats.
        """
        self._last_activity = time.time()
        docs_root = await _pick_docs_root(package)
        if not docs_root:
            raise ToolError(f"Could not find docs site for '{package}'")

        pages = await crawl_docs_site(docs_root, max_pages=max_pages)
        if not pages:
            return {"success": False, "message": "Docs crawl yielded no pages."}

        # Try to create scratch directory with error handling
        scratch_dir = _HOME / "scratch"
        try:
            if not scratch_dir.exists():
                scratch_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Fallback to a local directory if permission issues
            logger.warning(f"Could not create directory {scratch_dir}: {e}")
            scratch_dir = Path("./scratch")
            scratch_dir.mkdir(parents=True, exist_ok=True)
        
        fname = f"{package.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.txt"
        fpath = scratch_dir / fname

        # build content
        parts = []
        for url, txt in pages:
            parts.append(f"URL: {url}\n\n{txt.strip()}\n---")
        combined = "\n".join(parts)

        await write_file_content(str(fpath), combined)

        await _log("docs_harvest", package=package, pages=len(pages), file=str(fpath))

        return {
            "success": True,
            "package": package,
            "pages": len(pages),
            "file": str(fpath),
        }
    
    @tool(name="smart_browser.run_macro")
    @with_tool_metrics
    @with_error_handling
    async def execute_macro(self, 
                           url: str, 
                           task: str,
                           model: str = "gpt-4o",
                           max_rounds: int = 5) -> Dict[str, Any]:
        """
        Navigate to a URL and execute a natural language task.
        
        Args:
            url: URL to navigate to
            task: Natural language description of the task to perform
            model: LLM model to use for task planning
            max_rounds: Maximum number of planning/execution rounds
            
        Returns:
            Dictionary with macro execution results
        """
        self._last_activity = time.time()
        ctx, _ = await get_browser_context()
        
        async with _tab_context(ctx) as page:
            await _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle")
            
            try:
                # Run the macro
                results = await run_macro(page, task, max_rounds, model)
                
                # Get the final page state
                final_state = await get_page_state(page)
                
                return {
                    "success": True,
                    "task": task,
                    "steps": results,
                    "final_state": final_state
                }
            except Exception as e:
                # Ensure we get the page state even on error
                try:
                    final_state = await get_page_state(page)
                except Exception:
                    final_state = {"error": "Could not extract page state after error"}
                    
                return {
                    "success": False,
                    "task": task,
                    "error": str(e),
                    "final_state": final_state
                }
    
# Continue with the autopilot method and remaining code
    @tool(name="smart_browser.autopilot")
    @with_tool_metrics
    @with_error_handling
    async def autopilot(
        self,
        task: str,
        scratch_subdir: str = "autopilot_runs",
        max_steps: int = 25
    ) -> Dict[str, Any]:
        """
        Accepts an arbitrary multi-step *task* description, asks the LLM to build a
        JSON plan using our internal tools, executes each step sequentially, stores
        intermediate JSON responses in a scratch file, and returns a summary.
        """
        self._last_activity = time.time()
        # Generate initial plan
        plan = await _plan_autopilot(task)
        if len(plan) > max_steps:
            raise ToolError(f"Plan too long ({len(plan)} steps)")

        scratch_dir = _HOME / scratch_subdir
        await create_directory(str(scratch_dir))
        run_id = datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
        log_fp = scratch_dir / f"{run_id}.jsonl"
        
        # Open file once for append mode
        with open(log_fp, "w", encoding="utf-8") as log_file:
            results, step_no = [], 0
            
            for step in plan:
                step_no += 1
                tool_name = step.get("tool")
                args = step.get("args", {})
                
                if tool_name not in _AVAILABLE_TOOLS:
                    error_msg = f"Unknown tool '{tool_name}' in plan"
                    # Log error but continue with next steps
                    entry = {"step": step_no, "tool": tool_name, "args": args, 
                            "success": False, "error": error_msg}
                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()
                    results.append(entry)
                    continue
                
                try:
                    # Properly resolve method from string name without "self." prefix
                    method_name = _AVAILABLE_TOOLS[tool_name][0]
                    fn = getattr(self, method_name)
                    
                    # Execute tool step
                    outcome = await fn(**args)
                    
                    # Create entry with full result information
                    entry = {
                        "step": step_no, 
                        "tool": tool_name, 
                        "args": args, 
                        "success": outcome.get("success", True),
                        "result": outcome
                    }
                    
                    # Record result and update for next planning step
                    results.append(entry)
                    
                    # Save to log
                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()
                    
                    # If the step failed, replan using prior results
                    if not outcome.get("success", True):
                        try:
                            # Replan with previous results context
                            remaining_plan = await _plan_autopilot(task, results)
                            if remaining_plan:
                                # Merge remaining tasks into plan for execution
                                plan[len(results):] = remaining_plan
                        except Exception as replan_error:
                            await _log("autopilot_replan_error", error=str(replan_error))
                except Exception as e:
                    # Log the error but continue with the next step
                    entry = {
                        "step": step_no, 
                        "tool": tool_name, 
                        "args": args, 
                        "success": False, 
                        "error": str(e)
                    }
                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()
                    results.append(entry)

        await _log("autopilot_run", task=task, steps=len(results), file=str(log_fp))
                
        return {
            "success": True,
            "steps_executed": len(results),
            "run_log": str(log_fp),
            "results": results[-3:]  # return only last few for brevity
        }
    
    @tool(name="smart_browser.parallel")
    @with_tool_metrics
    @with_error_handling
    async def parallel_process(self, 
                              urls: List[str], 
                              max_tabs: Optional[int] = None) -> Dict[str, Any]:
        """
        Process multiple URLs in parallel using the tab pool.
        
        Args:
            urls: List of URLs to process
            max_tabs: Maximum number of parallel tabs (defaults to SB_MAX_TABS env var or 5)
            
        Returns:
            Dictionary with results for each URL
        """
        self._last_activity = time.time()
        # Create tab pool with specified max tabs
        pool = TabPool(max_tabs)
        
        # Define function to process a single URL with proper async closure
        async def process_url(page: Page, *, url: str) -> Dict[str, Any]:
            """Proper async function to avoid lambda closure issues"""
            try:
                await _log("navigate", url=url)
                await page.goto(url, wait_until="networkidle")
                state = await get_page_state(page)
                return {"url": url, "state": state, "success": True}
            except Exception as e:
                await _log("parallel_url_error", url=url, error=str(e))
                return {"url": url, "error": str(e), "success": False}
        
        # Create a callable for each URL
        tasks = []
        for url in urls:
            u = url if url.startswith("http") else f"https://{url}"
            # Use partial to bind the url parameter and prevent closure issues
            tasks.append(functools.partial(process_url, url=u))
        
        # Execute all tasks in parallel
        results = await pool.map(tasks)
        
        return {
            "success": True,
            "results": results,
            "processed_count": len(results)
        }