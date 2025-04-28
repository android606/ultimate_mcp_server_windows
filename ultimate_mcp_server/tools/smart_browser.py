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
✓ 5  AES-GCM-encrypted cookie jar with AAD for secure persistence (SB_STATE_KEY=<b64-key>)
✓ 6  Human-like jitter on UI actions with risk-aware timing
✓ 7  Resilient "chaos-monkey" retries with idempotent re-play
✓ 8  Async TAB POOL for parallel scraping with concurrency control (SB_MAX_TABS)
✓ 9  Pluggable HTML summarizer (trafilatura ▸ readability-lxml ▸ fallback)
✓ 10 Download helper with SHA-256 verification and audit logging
✓ 11 PDF / Excel auto-table extraction after download (using ThreadPool)
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
import threading
import time
import unicodedata
import urllib.parse
from collections import deque  # Use deque for efficient queue operations
from contextlib import asynccontextmanager
from datetime import datetime, timezone  # Use timezone-aware datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlparse

import aiofiles  # For async file I/O
import httpx
from bs4 import BeautifulSoup
from cryptography.exceptions import InvalidTag  # Import specific exception
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Playwright imports
from playwright.async_api import (
    Browser,
    BrowserContext,
    Locator,
    Page,
    PlaywrightException,  # More general Playwright exception
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

# MCP Server imports (assuming these exist)
try:
    from ultimate_mcp_server.constants import Provider
    from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
    from ultimate_mcp_server.tools.base import (
        BaseTool,
        tool,
        with_error_handling,
        with_tool_metrics,
    )
    from ultimate_mcp_server.tools.completion import generate_completion
    from ultimate_mcp_server.tools.filesystem import create_directory, write_file_content
    from ultimate_mcp_server.utils import get_logger
except ImportError:
    # Provide dummy implementations if MCP server is not available
    # This allows the module to be imported/tested standalone to some extent
    print(
        "WARNING: Ultimate MCP Server components not found. Using dummy implementations.",
        file=sys.stderr,
    )

    # Dummy logger
    import logging

    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

    # Dummy exceptions
    class ToolError(Exception):
        pass

    class ToolInputError(ToolError):
        pass

    class ProviderError(Exception):
        pass

    # Dummy decorators and base class
    class BaseTool:
        def __init__(self, mcp_server=None):
            self.mcp_server = mcp_server

    def tool(name):
        return lambda func: func

    def with_tool_metrics(func):
        return func

    def with_error_handling(func):
        return func

    # Dummy constants and functions
    class Provider:
        OPENAI = type("Provider", (), {"value": "openai"})()

    async def generate_completion(**kwargs):
        return {"success": False, "error": "Dummy LLM response"}

    async def create_directory(path):
        os.makedirs(path, exist_ok=True)

    async def write_file_content(path, content):
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)


logger = get_logger("ultimate_mcp_server.tools.smart_browser")

# Thread pool for CPU-bound tasks (e.g., hashing, potentially some sync I/O fallbacks)
# Use reasonable defaults, ensure proper shutdown
_cpu_count = os.cpu_count() or 1  # Ensure cpu_count is at least 1
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, _cpu_count * 2 + 4), thread_name_prefix="sb_worker"
)

# Process pool for heavy CPU tasks like PDF/Excel extraction (avoids GIL issues)
# Disabled for now, requires careful handling of dependencies and pickling.
# Using ThreadPool for now, but recommend ProcessPool if Tabula/Pandas become bottlenecks.
# _process_pool = concurrent.futures.ProcessPoolExecutor(
#     max_workers=os.cpu_count() or 1
# )

# ══════════════════════════════════════════════════════════════════════════════
# 0.  FILESYSTEM & ENCRYPTION & DB
# ══════════════════════════════════════════════════════════════════════════════
_HOME = Path.home() / ".smart_browser"
try:
    _HOME.mkdir(parents=True, exist_ok=True)
    # Fix: Set restrictive permissions on the main directory
    os.chmod(_HOME, 0o700)
except OSError as e:
    logger.error(f"Failed to create or set permissions on {_HOME}: {e}")
    # Fallback or raise critical error depending on requirements
    # For now, log and continue, but functionality might be impaired.

_STATE_FILE = _HOME / "storage_state.enc"
_LOG_FILE = _HOME / "audit.log"
_SELDB_FILE = _HOME / "selectors.db"
_last_hash: str | None = None
_log_lock = asyncio.Lock()  # Lock for async audit log writing (protects _last_hash)

# Fix 1: Use threading.RLock for DB connection pool management as it's synchronous.
# The actual DB operations within async functions will use run_in_executor.
_db_conn_pool_lock = threading.RLock()
_db_connection: sqlite3.Connection | None = (
    None  # Single connection per process for simplicity with check_same_thread=False
)

# Versioned cipher format - add "SB1" prefix and use AAD
CIPHER_VERSION = b"SB1"
# Fix CWE-311: Add Associated Authenticated Data (AAD)
AAD_TAG = b"smart-browser-state-v1"


def _get_db_connection() -> sqlite3.Connection:
    """Get or create the single SQLite connection."""
    global _db_connection
    # Use threading lock for thread-safe access to the global connection variable
    with _db_conn_pool_lock:
        if _db_connection is None:
            try:
                # check_same_thread=False allows access from multiple threads (via run_in_executor)
                # but requires careful serialization of writes if WAL is not enough.
                _db_connection = sqlite3.connect(
                    _SELDB_FILE,
                    check_same_thread=False,
                    isolation_level=None,  # Autocommit mode
                    timeout=10,  # Wait 10 seconds if DB is locked
                )
                # Enable WAL mode for better concurrency
                _db_connection.execute("PRAGMA journal_mode=WAL")
                _db_connection.execute("PRAGMA foreign_keys = ON")
                _db_connection.execute("PRAGMA busy_timeout = 10000")  # Corresponds to timeout=10
                logger.info(f"Initialized SQLite DB connection to {_SELDB_FILE}")
            except sqlite3.Error as e:
                logger.critical(
                    f"Failed to connect to or initialize SQLite DB at {_SELDB_FILE}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Failed to initialize database: {e}") from e
        return _db_connection


def _close_db_connection():
    """Close the SQLite connection."""
    global _db_connection
    with _db_conn_pool_lock:
        if _db_connection is not None:
            try:
                _db_connection.close()
                logger.info("Closed SQLite DB connection.")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite DB connection: {e}")
            finally:
                _db_connection = None


# Register DB close on exit
atexit.register(_close_db_connection)


# --- Encryption ---
def _key() -> bytes | None:
    """Get and validate the AES-GCM key from environment variable."""
    k = os.getenv("SB_STATE_KEY", "")
    if not k:
        return None
    try:
        decoded = base64.b64decode(k)
        # Validate key length for AES-GCM (128, 192, or 256 bits)
        if len(decoded) not in (16, 24, 32):
            logger.warning(
                f"Invalid AES-GCM key length ({len(decoded)} bytes). Must be 16, 24, or 32 bytes after base64 decode. Encryption disabled."
            )
            return None
        return decoded
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid base64 in SB_STATE_KEY: {e}. Encryption disabled.")
        return None


def _enc(buf: bytes) -> bytes:
    """Encrypt data using AES-GCM with AAD."""
    k = _key()
    if not k:
        # Fix #11: Log if encryption was expected but not possible
        if os.getenv("SB_STATE_KEY"):
            logger.warning("Encryption key invalid or missing. Storing state in cleartext.")
        return buf  # Return plaintext if no valid key

    try:
        nonce = os.urandom(12)  # GCM standard nonce size
        encrypted_data = AESGCM(k).encrypt(nonce, buf, AAD_TAG)
        # Prepend version and nonce
        return CIPHER_VERSION + nonce + encrypted_data
    except Exception as e:
        logger.error(f"Encryption failed: {e}", exc_info=True)
        # Indicate failure - perhaps raise an error or return original buffer?
        # For safety, returning original might be unexpected. Raise or return None/Error indicator.
        # Let's raise to make the failure explicit.
        raise RuntimeError(f"Encryption failed: {e}") from e


def _dec(buf: bytes) -> bytes | None:
    """Decrypt data using AES-GCM with AAD, handling versioning."""
    k = _key()
    if not k:
        # If no key, assume data is plaintext (or handle as error if encryption is mandatory)
        logger.warning("No encryption key found. Assuming state file is cleartext.")
        return buf

    if not buf.startswith(CIPHER_VERSION):
        logger.warning("State file has an unknown or legacy format. Decryption attempt might fail.")
        # Attempt legacy decryption (without AAD, assuming 12-byte nonce prefix)
        # This is risky and might fail. Consider disallowing legacy format.
        # For now, we'll return None to force re-creation of state.
        logger.error("Legacy encryption format not supported. Please remove old state file.")
        _STATE_FILE.unlink(missing_ok=True)
        return None

    try:
        version_len = len(CIPHER_VERSION)
        nonce = buf[version_len : version_len + 12]
        ciphertext = buf[version_len + 12 :]
        # Fix CWE-359: Use the correct AAD during decryption
        return AESGCM(k).decrypt(nonce, ciphertext, AAD_TAG)
    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag (data tampered or wrong key?).")
        # Fix #11 / CWE-359: Remove corrupt state file and return None to signal failure
        _STATE_FILE.unlink(missing_ok=True)
        return None
    except Exception as e:
        logger.error(f"Decryption failed: {e}", exc_info=True)
        # Fix #11 / CWE-359: Remove corrupt state file and return None
        _STATE_FILE.unlink(missing_ok=True)
        return None


# --- Selector Cache DB ---
def _init_db_sync():
    """Synchronous DB initialization."""
    conn = None
    try:
        # Use the shared connection mechanism
        conn = _get_db_connection()
        # Use try-finally to ensure cursor is closed
        cursor = conn.cursor()
        try:
            # Use TEXT primary key for flexibility if needed, but separate columns are fine
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS selectors(
                    site TEXT NOT NULL,
                    key TEXT NOT NULL,
                    css TEXT NOT NULL, -- Store empty string if null
                    xpath TEXT NOT NULL, -- Store empty string if null
                    score REAL DEFAULT 1.0, -- Use REAL for score
                    last_used INTEGER DEFAULT (strftime('%s', 'now')), -- Timestamp
                    PRIMARY KEY(site, key, css, xpath)
                ) WITHOUT ROWID;"""  # Use WITHOUT ROWID if PK covers all access patterns
            )
            # Consider indexes for faster lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_selectors_site_key ON selectors (site, key, score DESC);"
            )
            # Commit is handled by isolation_level=None (autocommit)
            # conn.commit() # Not needed with autocommit
            logger.info("Selector DB schema initialized/verified.")
        finally:
            cursor.close()
    except sqlite3.Error as e:
        logger.critical(f"Failed to initialize selector DB schema: {e}", exc_info=True)
        # If DB init fails, the tool might not function correctly. Re-raise.
        raise RuntimeError(f"Failed to initialize selector database: {e}") from e
    # Note: Connection is not closed here; it's managed globally.


_init_db_sync()  # Initialize synchronously at startup


def _sel_key(role: str | None, name: str | None) -> str:
    """Generate a normalized key for selector caching."""
    # Normalize Unicode to NFC form
    norm_role = unicodedata.normalize("NFC", role) if role else ""
    norm_name = unicodedata.normalize("NFC", name) if name else ""
    # Limit key length to prevent excessively long keys
    key = f"{norm_role}::{norm_name}".strip(":")
    return key[:255]  # Limit key length


async def _get_best_selectors(site: str, key: str) -> list[tuple[str | None, str | None]]:
    """Get the best selectors for a site/key combination (runs DB query in thread pool)."""
    loop = asyncio.get_running_loop()

    def db_operation():
        conn = _get_db_connection()
        cursor = conn.cursor()
        try:
            # Fetch css/xpath, update last_used timestamp implicitly? Maybe not needed here.
            cursor.execute(
                "SELECT css, xpath FROM selectors WHERE site=? AND key=? ORDER BY score DESC LIMIT 5",
                (site, key),
            )
            rows = cursor.fetchall()
            # Convert empty strings back to None if needed by caller, though storing '' is fine
            return [(css if css else None, xpath if xpath else None) for css, xpath in rows]
        except sqlite3.Error as e:
            logger.error(f"SQLite error in _get_best_selectors for site={site}, key={key}: {e}")
            return []
        finally:
            cursor.close()

    # Fix #1, #8: Run synchronous DB operation in the thread pool
    return await loop.run_in_executor(_thread_pool, db_operation)


async def _bump_selector(site: str, key: str, css: str | None, xpath: str | None):
    """Increment the score for a selector (runs DB update in thread pool)."""
    # Normalize inputs: store empty strings instead of None in DB
    css_db = css or ""
    xpath_db = xpath or ""
    loop = asyncio.get_running_loop()

    def db_operation():
        conn = _get_db_connection()
        cursor = conn.cursor()
        try:
            # Use ON CONFLICT for atomic upsert, update score and last_used time
            cursor.execute(
                """INSERT INTO selectors(site, key, css, xpath, score, last_used)
                   VALUES (?, ?, ?, ?, 1.0, strftime('%s', 'now'))
                   ON CONFLICT(site, key, css, xpath) DO UPDATE SET
                     score = score + 1.0,
                     last_used = strftime('%s', 'now')
                   WHERE site = excluded.site AND key = excluded.key AND css = excluded.css AND xpath = excluded.xpath;""",
                (site, key, css_db, xpath_db),
            )
            # logger.debug(f"Bumped selector: site={site}, key={key}, css='{css_db}', xpath='{xpath_db}'")
        except sqlite3.Error as e:
            # Log error but don't crash the calling operation
            logger.error(f"SQLite error in _bump_selector for site={site}, key={key}: {e}")
        finally:
            cursor.close()
    await loop.run_in_executor(_thread_pool, db_operation)


async def _perform_selector_cleanup():
    """Runs SQL commands to decay scores and remove old/bad selectors."""
    loop = asyncio.get_running_loop()
    logger.info("Starting periodic selector cleanup...")

    def db_operation():
        conn = _get_db_connection()
        cursor = conn.cursor()
        deleted_low_score = 0
        deleted_old = 0
        decayed_count = 0
        try:
            # Decay scores slightly for selectors not used recently (e.g., > 7 days)
            # Avoid decaying scores that are already very low
            decay_cutoff_time = "(strftime('%s', 'now', '-7 days'))"
            cursor.execute(
                f"""UPDATE selectors
                   SET score = score * 0.90
                   WHERE last_used < {decay_cutoff_time} AND score > 0.1"""
            )
            decayed_count = cursor.rowcount

            # Delete selectors with very low scores (e.g., < 0.1)
            cursor.execute("DELETE FROM selectors WHERE score < 0.1")
            deleted_low_score = cursor.rowcount

            # Delete selectors not used for a long time (e.g., > 90 days)
            ninety_days_ago = "(strftime('%s', 'now', '-90 days'))"
            cursor.execute(f"DELETE FROM selectors WHERE last_used < {ninety_days_ago}")
            deleted_old = cursor.rowcount

            # Optional: Vacuum the database to reclaim space after deletions
            # cursor.execute("VACUUM;") # Can be slow, use cautiously

            logger.info(f"Selector cleanup finished. Decayed: {decayed_count}, Deleted (low score): {deleted_low_score}, Deleted (old): {deleted_old}")
            return True
        except sqlite3.Error as e:
            logger.error(f"SQLite error during selector cleanup: {e}")
            return False
        finally:
            cursor.close()

    await loop.run_in_executor(_thread_pool, db_operation)

_selector_cleanup_task_handle: Optional[asyncio.Task] = None


async def _selector_cleanup_task(interval_seconds: int = 24 * 60 * 60): # Default: Daily
    """Background task to periodically run selector cleanup."""
    logger.info(f"Selector cleanup task started. Running every {interval_seconds} seconds.")
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await _perform_selector_cleanup()
        except asyncio.CancelledError:
            logger.info("Selector cleanup task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in selector cleanup task loop: {e}", exc_info=True)
            # Avoid tight loop on error, wait before retrying
            await asyncio.sleep(60 * 5) # Wait 5 minutes after error

# ══════════════════════════════════════════════════════════════════════════════
# 1.  AUDIT LOG (hash-chained)
# ══════════════════════════════════════════════════════════════════════════════


def _sanitize_for_log(obj: Any) -> Any:
    """Sanitize values for JSON logging, preventing injection."""
    if isinstance(obj, str):
        # Fix CWE-79: Use JSON encoding to handle quotes, backslashes, control chars
        # Return the string content without the surrounding quotes added by dumps
        try:
            encoded = json.dumps(obj)
            return encoded[1:-1] if len(encoded) >= 2 else ""
        except TypeError:
            # Handle potential errors during encoding (e.g., surrogates)
            return "???"  # Placeholder for unencodable string data
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_log(v) for k, v in obj.items()}  # Ensure keys are strings
    elif isinstance(obj, list):
        return [_sanitize_for_log(item) for item in obj]
    # Allow safe basic types
    elif isinstance(obj, (int, float, bool, type(None))):
        return obj
    # Convert other types to string and sanitize
    else:
        try:
            s = str(obj)
            encoded = json.dumps(s)
            return encoded[1:-1] if len(encoded) >= 2 else ""
        except Exception:
            return "???"  # Placeholder for complex objects


_EVENT_EMOJI_MAP = {
    # --- Lifecycle ---
    "browser_start": "🚀",
    "browser_shutdown": "🛑",
    "browser_shutdown_complete": "🏁",
    "browser_context_create": "➕",
    "browser_incognito_context": "🕶️",
    "browser_context_close_shared": "➖",
    "browser_close": "🚪",
    "page_open": "📄",
    "page_close": "덮",
    "page_error": "🔥", # Error specific to a page operation in pool
    "tab_timeout": "⏱️", # Tab pool operation timeout
    "tab_cancelled": "🚫", # Tab pool operation cancelled
    "tab_error": "💥", # Generic error within a tab pool operation

    # --- Navigation & State ---
    "navigate": "➡️",
    "navigate_start": "➡️",
    "navigate_success": "✅",
    "navigate_fail_playwright": "❌",
    "navigate_fail_unexpected": "💣",
    "navigate_wait_selector_ok": "👌",
    "navigate_wait_selector_timeout": "⏳",
    "page_state_extracted": "ℹ️",
    "browse_fail_proxy_disallowed": "🛡️", # Proxy blocked navigation

    # --- Actions ---
    "click": "🖱️", # Generic click log (if used)
    "click_success": "🖱️✅",
    "click_fail_notfound": "🖱️❓",
    "click_fail_playwright": "🖱️❌",
    "click_fail_unexpected": "🖱️💣",
    "type": "⌨️", # Generic type log (if used)
    "type_success": "⌨️✅",
    "type_fail_secret": "⌨️🔑", # Secret resolution failure
    "type_fail_notfound": "⌨️❓",
    "type_fail_playwright": "⌨️❌",
    "type_fail_unexpected": "⌨️💣",
    "scroll": "↕️", # Generic scroll action

    # --- Locators ---
    "locator_learned_hit": "🧠",
    "locator_success": "🎯",
    "locator_hit": "🎯", # Alias for success?
    "locator_fail": "❓",

    # --- Downloads ---
    "download": "💾", # Generic download (smart_download result)
    "download_navigate": "🚚", # Navigate step before download
    "download_success": "💾✅",
    "download_fail_notfound": "💾❓",
    "download_fail_timeout": "💾⏱️",
    "download_fail_playwright": "💾❌",
    "download_fail_unexpected": "💾💣",
    "download_pdf_http": "📄💾", # Direct HTTP PDF download log
    "download_direct_success": "✨💾", # Successful direct HTTP download
    "download_pdf_error": "📄🔥", # Error during direct HTTP download
    "download_site_pdfs_complete": "📚✅", # Bulk PDF download finished

    # --- Extraction ---
    "table_extract_success": "📊✅",
    "table_extract_error": "📊❌",
    "docs_collected_success": "📖✅", # Documentation collection success
    "docs_harvest": "📖", # Alias? Keep consistent

    # --- Search ---
    "search": "🔍", # Generic search result summary
    "search_start": "🔍➡️",
    "search_complete": "🔍✅",
    "search_captcha": "🤖",
    "search_no_results_selector": "🤷",
    "search_error_playwright": "🔍❌",
    "search_error_unexpected": "🔍💣",

    # --- Macro / Autopilot ---
    "macro_plan": "📝", # Generic plan log
    "macro_plan_generated": "📝✅",
    "macro_plan_empty": "📝🤷",
    "macro_step_result": "▶️",
    "macro_complete": "🎉",
    "macro_finish_action": "🏁", # Explicit finish action
    "macro_error": "💥", # Generic macro error (if used)
    "macro_exceeded_rounds": "🔄",
    "macro_fail_step": "❌",
    "macro_error_tool": "🛠️💥",
    "macro_error_unexpected": "💣💥",
    "macro_navigate": "🗺️➡️", # Navigate step within a macro/tool
    "click_extract_navigate": "🖱️🗺️",
    "click_extract_success": "🖱️✅✨",
    "fill_form_navigate": "✍️🗺️",
    "fill_form_field": "✍️",
    "fill_form_submit": "✔️",
    "fill_form_success": "✍️✅",
    "autopilot_run": "🧑‍✈️",
    "autopilot_step_start": "▶️",
    "autopilot_step_success": "✅",
    "autopilot_step_fail": "❌",
    "autopilot_replan_success": "🧠🔄",
    "autopilot_replan_fail": "🧠❌",
    "autopilot_max_steps": "🚫🔄",
    "autopilot_plan_end": "🏁",
    "autopilot_critical_error": "💥🧑‍✈️",

    # --- Parallel Processing ---
    "parallel_navigate": "🚦➡️",
    "parallel_url_error": "🚦🔥",
    "parallel_process_complete": "🚦🏁",

    # --- System / Misc ---
    "retry": "⏳", # Waiting for retry
    "retry_fail": "⚠️", # Retry failed completely
    "retry_fail_unexpected": "💣⚠️", # Retry failed on unexpected error
    "retry_unexpected": "⏳💣", # Retry on unexpected error
    "llm_call_complete": "🤖💬",
    "db_cleanup_start": "🧹", # If cleanup start is logged
    "db_cleanup_complete": "🧹✅", # If cleanup end is logged
}

async def _log(event: str, **details):
    """Append a hash-chained entry to the audit log asynchronously."""
    global _last_hash
    # Use timezone-aware UTC time
    ts_iso = datetime.now(timezone.utc).isoformat()

    # Sanitize details
    sanitized_details = _sanitize_for_log(details)

    # Map event to emoji key
    emoji_key = _EVENT_EMOJI_MAP.get(event)  # Use .get for safety

    # Fix #9: Ensure atomicity of reading/updating _last_hash using the async lock
    async with _log_lock:
        # Read _last_hash within the lock
        current_last_hash = _last_hash

        entry = {
            "ts": ts_iso,
            "event": event,
            "details": sanitized_details,
            "prev": current_last_hash,
            "emoji": emoji_key,  # Using 'emoji' key now
        }
        # Use separators=(',', ':') for compact JSON, ensure keys are sorted for consistent hashing
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()

        log_entry_line = json.dumps({"hash": h, **entry}, separators=(",", ":")) + "\n"

        try:
            # Fix: Use aiofiles for async file writing
            async with aiofiles.open(_LOG_FILE, "a", encoding="utf-8") as f:
                await f.write(log_entry_line)
                await f.flush()  # Ensure it's written to OS buffer

            # Update _last_hash only after successful write+flush
            _last_hash = h
        except IOError as e:
            logger.error(f"Failed to write to audit log {_LOG_FILE}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing audit log: {e}", exc_info=True)


# Initialize _last_hash from existing log file if present (synchronously at startup)
def _init_last_hash():
    global _last_hash
    if _LOG_FILE.exists():
        try:
            # Read the last line efficiently
            with open(_LOG_FILE, "rb") as f:
                f.seek(-2, os.SEEK_END)  # Go to second-to-last byte
                while f.read(1) != b"\n":  # Find start of last line
                    f.seek(-2, os.SEEK_CUR)
                    if f.tell() == 0:  # Reached beginning of file
                        break
                last_line = f.readline().decode("utf-8")

            if last_line:
                last_entry = json.loads(last_line)
                _last_hash = last_entry.get("hash")
                logger.info(f"Initialized audit log chain from last hash: {_last_hash[:8]}...")
            else:
                logger.info("Audit log file found but is empty.")
        except Exception as e:
            logger.error(
                f"Failed to read last hash from audit log {_LOG_FILE}: {e}. Starting new chain."
            )
            _last_hash = None
    else:
        logger.info("No existing audit log found. Starting new chain.")


_init_last_hash()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESILIENT RETRY DECORATOR
# ══════════════════════════════════════════════════════════════════════════════
def resilient(max_attempts: int = 3, backoff: float = 0.3):
    """
    Decorator for async functions; retries on common transient errors like
    PlaywrightTimeoutError or network issues.
    """

    def wrap(fn):
        @functools.wraps(fn)
        async def inner(*a, **kw):
            attempt = 0
            while True:
                try:
                    # Add jitter *before* the attempt for less deterministic timing
                    if attempt > 0:
                        jitter_delay = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                        await asyncio.sleep(jitter_delay)

                    return await fn(*a, **kw)
                # Catch specific, retryable exceptions
                except (
                    PlaywrightTimeoutError,
                    PlaywrightException,
                    httpx.RequestError,
                    asyncio.TimeoutError,
                ) as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        func_name = getattr(fn, "__name__", "unknown_func")
                        await _log(
                            "retry_fail", func=func_name, attempts=max_attempts, error=str(e)
                        )
                        raise ToolError(
                            f"Operation '{func_name}' failed after {max_attempts} attempts: {e}"
                        ) from e

                    # Calculate delay for next attempt (moved sleep to start of loop)
                    delay = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    func_name = getattr(fn, "__name__", "unknown_func")
                    await _log(
                        "retry",
                        func=func_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep=round(delay, 2),
                        error=str(e),
                    )
                    # Sleep is handled at the start of the next iteration

                # Do not retry on non-transient errors or user interrupts
                except (ToolError, ValueError, TypeError, KeyError, KeyboardInterrupt):
                    raise  # Re-raise immediately
                except Exception as e:  # Catch unexpected errors
                    attempt += 1
                    if attempt >= max_attempts:
                        func_name = getattr(fn, "__name__", "unknown_func")
                        await _log(
                            "retry_fail_unexpected",
                            func=func_name,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        raise ToolError(
                            f"Operation '{func_name}' failed with unexpected error after {max_attempts} attempts: {e}"
                        ) from e

                    delay = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    func_name = getattr(fn, "__name__", "unknown_func")
                    await _log(
                        "retry_unexpected",
                        func=func_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep=round(delay, 2),
                        error=str(e),
                    )
                    # Sleep is handled at the start of the next iteration

        return inner

    return wrap


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SECRET VAULT BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
_ALLOWED_VAULT_PATHS = set(
    path.strip().rstrip("/") + "/"
    for path in os.getenv(  # Ensure trailing slash for prefix matching
        "VAULT_ALLOWED_PATHS", "secret/data/,kv/data/"
    ).split(",")
    if path.strip()
)


def get_secret(path_key: str) -> str:
    """
    Retrieve secret from environment or HashiCorp Vault.
    path_key formats:
        env:VAR_NAME          → reads from environment (e.g., "env:MY_API_KEY")
        vault:kv/path/to/secret#key  → reads 'key' from Vault KV v2 secret
                                        at 'kv/data/path/to/secret'
        vault:secret/path/to/secret#key → reads 'key' from Vault KV v1 secret
                                          at 'secret/path/to/secret'
    Requires VAULT_ADDR and VAULT_TOKEN environment variables for vault access.
    Requires 'hvac' library installed.
    """
    if path_key.startswith("env:"):
        var = path_key[4:]
        val = os.getenv(var)
        if val is None:
            raise ToolInputError(f"Environment variable secret '{var}' not set.")
        return val

    if path_key.startswith("vault:"):
        try:
            import hvac
        except ImportError as e:
            raise RuntimeError(
                "Python library 'hvac' is required for Vault access but not installed."
            ) from e

        addr = os.getenv("VAULT_ADDR")
        token = os.getenv("VAULT_TOKEN")
        if not addr or not token:
            raise RuntimeError(
                "VAULT_ADDR and VAULT_TOKEN environment variables must be set for Vault access."
            )

        # Fix #17: Use distinct variable names for parsing
        full_vault_uri = path_key[len("vault:") :]

        # Fix CWE-502: Basic check to prevent protocol specification in path
        if "://" in full_vault_uri:
            raise ValueError("Vault path cannot contain '://'. Specify address via VAULT_ADDR.")

        # Separate path and key
        if "#" not in full_vault_uri:
            raise ValueError(
                "Vault path must include the secret key separated by '#'. Format: vault:path/to/secret#key"
            )
        path_part, key_name = full_vault_uri.split("#", 1)
        path_part = path_part.strip("/")  # Normalize path

        # Determine Vault KV version and construct API path
        # KV v2 mount points often have '/data/' in the logical path users see,
        # but the API path needs `/data/` inserted after the mount point.
        # KV v1 paths are direct.
        # Heuristic: Assume KV v2 if 'data' is in the path segments after the mount? Risky.
        # Better: Try v2 first, then fall back to v1? Or require explicit v1/v2 marker?
        # Standard practice is KV v2 uses `/data/` in the API path. Let's assume that.

        # Try to split mount point and the rest of the path
        path_segments = path_part.split("/")
        if not path_segments:
            raise ValueError(f"Invalid Vault path format: '{path_part}'")

        mount_point = path_segments[0]
        secret_sub_path = "/".join(path_segments[1:])

        # Fix CWE-502: Validate the path_part against the allowlist more carefully
        # Ensure the *intended* secret path (e.g., "kv/data/my/app") starts with an allowed prefix
        # We check against the user-provided path_part before manipulating it for API calls.
        path_to_check = path_part + "/"  # Ensure trailing slash for prefix match
        allowed = any(path_to_check.startswith(prefix) for prefix in _ALLOWED_VAULT_PATHS)
        if not allowed:
            logger.warning(
                f"Access denied for Vault path '{path_part}'. Allowed prefixes: {_ALLOWED_VAULT_PATHS}"
            )
            raise ValueError(
                f"Access to Vault path '{path_part}' is not allowed by VAULT_ALLOWED_PATHS."
            )

        client = hvac.Client(url=addr, token=token)
        if not client.is_authenticated():
            raise RuntimeError(
                f"Vault authentication failed for address {addr}. Check VAULT_TOKEN."
            )

        # Try KV v2 read first
        try:
            api_path_v2 = f"{mount_point}/data/{secret_sub_path}"  # Standard KV v2 API structure
            logger.debug(
                f"Attempting Vault KV v2 read: mount='{mount_point}', path='{secret_sub_path}' (API: {api_path_v2})"
            )
            response = client.secrets.kv.v2.read_secret_version(
                mount_point=mount_point,  # Mount point name
                path=secret_sub_path,  # Path within the mount
            )
            secret_data = response["data"]["data"]
            if key_name in secret_data:
                return secret_data[key_name]
            else:
                raise KeyError(f"Key '{key_name}' not found in KV v2 secret at '{path_part}'")

        except hvac.exceptions.InvalidPath:
            # If InvalidPath, it might be a KV v1 secret or the path is wrong
            logger.debug(f"KV v2 path '{path_part}' not found, trying KV v1.")
            pass  # Fall through to try KV v1
        except KeyError as e:
            # If 'data' or 'data' key missing, likely not v2 or unexpected format
            logger.debug(
                f"Unexpected KV v2 response structure for '{path_part}': {e}. Trying KV v1."
            )
            pass  # Fall through to try KV v1
        except Exception as e:
            # Catch other potential hvac errors during v2 read
            logger.error(f"Error reading Vault KV v2 secret '{path_part}': {e}")
            raise RuntimeError(f"Failed to read Vault secret: {e}") from e

        # Try KV v1 read as fallback
        try:
            api_path_v1 = f"{mount_point}/{secret_sub_path}"  # KV v1 path is direct
            logger.debug(
                f"Attempting Vault KV v1 read: mount='{mount_point}', path='{secret_sub_path}' (API: {api_path_v1})"
            )
            response = client.secrets.kv.v1.read_secret(
                mount_point=mount_point, path=secret_sub_path
            )
            secret_data = response["data"]
            if key_name in secret_data:
                return secret_data[key_name]
            else:
                raise KeyError(f"Key '{key_name}' not found in KV v1 secret at '{path_part}'")

        except hvac.exceptions.InvalidPath:
            logger.error(f"Secret path not found in Vault (tried KV v2 and KV v1): '{path_part}'")
            raise KeyError(
                f"Secret path '{path_part}' not found in Vault."
            ) from None  # Chain explicitly
        except KeyError as e:
            # Handle cases where 'data' key is missing in v1 response or key_name is missing
            logger.error(
                f"Key '{key_name}' not found or unexpected KV v1 response structure for '{path_part}': {e}"
            )
            raise KeyError(
                f"Key '{key_name}' not found at path '{path_part}' (KV v1 attempt)."
            ) from e
        except Exception as e:
            logger.error(f"Error reading Vault KV v1 secret '{path_part}': {e}")
            raise RuntimeError(f"Failed to read Vault secret: {e}") from e

    raise ValueError(f"Unknown secret scheme or invalid path: {path_key}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLAYWRIGHT LIFECYCLE (proxy, VNC, cookies)
# ══════════════════════════════════════════════════════════════════════════════
_pw: Optional[async_playwright] = None
_browser: Optional[Browser] = None
_ctx: Optional[BrowserContext] = None
_vnc_proc: Optional[subprocess.Popen] = None
_js_lib_cached: Set[str] = set()  # Track injected JS libraries per context? Maybe global is fine.
_js_lib_lock = asyncio.Lock()  # Lock for script injection check
_playwright_lock = asyncio.Lock()  # Lock for playwright/browser/context initialization


def _get_proxy_config() -> Optional[Dict[str, Any]]:
    """Parses PROXY_POOL and returns a Playwright-compatible proxy dict if valid."""
    pool_str = os.getenv("PROXY_POOL", "")
    if not pool_str:
        return None

    proxies = [p.strip() for p in pool_str.split(";") if p.strip()]
    if not proxies:
        return None

    chosen_proxy = random.choice(proxies)
    logger.info(f"Attempting to use proxy: {chosen_proxy}")

    try:
        parsed = urlparse(chosen_proxy)

        # Basic validation
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"Invalid proxy URL format: '{chosen_proxy}'. Skipping proxy.")
            return None
        if parsed.scheme not in ("http", "https", "socks5", "socks5h"):
            logger.warning(
                f"Invalid proxy scheme '{parsed.scheme}' in '{chosen_proxy}'. Only http, https, socks5, socks5h supported. Skipping proxy."
            )
            return None
        # Prevent '#' characters which might interfere with parsing or security
        if "#" in chosen_proxy:
            logger.warning(
                f"Proxy URL '{chosen_proxy}' contains invalid character '#'. Skipping proxy."
            )
            return None

        # Fix #4: Always return the dictionary format expected by Playwright
        proxy_dict: Dict[str, Any] = {"server": f"{parsed.scheme}://{parsed.netloc}"}

        # Handle credentials embedded in the URL (e.g., user:pass@host:port)
        if parsed.username:
            proxy_dict["username"] = urllib.parse.unquote(parsed.username)
        if parsed.password:
            proxy_dict["password"] = urllib.parse.unquote(parsed.password)
            # Update server URL to remove credentials for the dict key
            proxy_dict["server"] = (
                f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
                if parsed.port
                else f"{parsed.scheme}://{parsed.hostname}"
            )

        logger.info(
            f"Using proxy configuration: {proxy_dict['server']} (auth {'present' if 'username' in proxy_dict else 'not present'})"
        )
        return proxy_dict

    except Exception as e:
        logger.warning(f"Error parsing proxy URL '{chosen_proxy}': {e}. Skipping proxy.")
        return None


def _get_allowed_domains() -> Optional[List[str]]:
    """Gets list of domains allowed for proxy usage. Returns None if all allowed."""
    domains_str = os.getenv("PROXY_ALLOWED_DOMAINS", "")
    if not domains_str or domains_str == "*":
        return None  # All domains allowed
    domains = [d.strip().lower() for d in domains_str.split(",") if d.strip()]
    # Normalize domains (e.g., ensure they start with a dot for subdomain matching)
    return [d if d.startswith(".") else "." + d for d in domains]


_PROXY_ALLOWED_DOMAINS = _get_allowed_domains()  # Cache at startup


def _is_domain_allowed_for_proxy(url: str) -> bool:
    """Checks if the URL's domain is allowed based on PROXY_ALLOWED_DOMAINS."""
    if _PROXY_ALLOWED_DOMAINS is None:
        return True  # All domains allowed

    try:
        domain = urlparse(url).netloc.lower()
        if not domain:
            return False  # Cannot determine domain

        # Check if the domain or any of its parent domains match the allowed list
        # E.g., if ".example.com" is allowed, "sub.example.com" should match.
        domain_parts = domain.split(".")
        for i in range(len(domain_parts)):
            sub_domain = "." + ".".join(domain_parts[i:])
            if sub_domain in _PROXY_ALLOWED_DOMAINS:
                return True
        return False
    except Exception as e:
        logger.warning(f"Error checking domain allowance for URL '{url}': {e}")
        return False  # Deny on error


async def get_browser_context(
    use_incognito: bool = False,
    context_args: Optional[Dict[str, Any]] = None,  # Accept extra args like user_agent
) -> tuple[BrowserContext, Browser]:
    """
    Get or create a browser context. Ensures Playwright and Browser are initialized.
    Uses an async lock to prevent race conditions during initialization.

    Args:
        use_incognito: If True, creates a new incognito context. Otherwise, returns shared context.
        context_args: Optional dictionary of arguments for Browser.new_context().

    Returns:
        Tuple of (BrowserContext, Browser)
    """
    global _pw, _browser, _ctx
    async with _playwright_lock:  # Lock ensures atomic initialization
        # Initialize Playwright if needed
        if not _pw:
            try:
                _pw = await async_playwright().start()
                logger.info("Playwright started.")
            except Exception as e:
                logger.critical(f"Failed to start Playwright: {e}", exc_info=True)
                raise RuntimeError(f"Failed to start Playwright: {e}") from e

        # Determine headless mode
        headless_env = os.getenv("HEADLESS", "true")  # Default to headless
        is_headless = headless_env.lower() not in ("0", "false", "no")

        # Start VNC only if not headless and VNC=1
        if not is_headless:
            _start_vnc()  # Start VNC if configured (non-blocking)

        # Launch Browser if needed
        if not _browser or not _browser.is_connected():
            if _browser:  # If disconnected, try closing first
                try:
                    await _browser.close()
                except Exception:
                    pass  # Ignore errors closing old browser
            try:
                _browser = await _pw.chromium.launch(
                    headless=is_headless,
                    # Proxy is set per-context
                    args=[
                        "--no-sandbox",  # Common requirement in Docker/Linux
                        "--disable-dev-shm-usage",  # Common requirement in Docker
                        "--disable-gpu",  # Often helps stability in headless
                        "--window-size=1280,1024",
                        # "--enable-features=NetworkService,NetworkServiceInProcess", # May not be needed
                        # "--dns-prefetch-disable", # Might help with some DNS issues
                    ],
                )
                logger.info(f"Browser launched (Headless: {is_headless}).")
                # Add cleanup hook specifically for the browser
                atexit.register(lambda: asyncio.run(_try_close_browser()))
            except PlaywrightException as e:
                logger.critical(f"Failed to launch browser: {e}", exc_info=True)
                raise RuntimeError(f"Failed to launch browser: {e}") from e

        # Handle context creation / retrieval
        default_args = {
            "viewport": {"width": 1280, "height": 1024},
            "locale": "en-US",
            "timezone_id": "UTC",
            "accept_downloads": True,
        }
        if context_args:
            default_args.update(context_args)

        # Incognito context: always create new
        if use_incognito:
            try:
                incognito_ctx = await _browser.new_context(**default_args)
                await _log("browser_incognito_context", args=default_args)
                # Add hook for proxy domain checking if proxy is enabled for this context
                if default_args.get("proxy"):
                    await _add_proxy_routing_rule(incognito_ctx, default_args["proxy"])
                return incognito_ctx, _browser
            except PlaywrightException as e:
                logger.error(f"Failed to create incognito context: {e}", exc_info=True)
                raise ToolError(f"Failed to create incognito context: {e}") from e

        # Shared context: create if doesn't exist or is closed
        if not _ctx or not _ctx.browser:  # Check if context is still associated with a browser
            # Check if context is closed - Playwright might not have a simple is_closed()
            # Checking .browser is a reasonable proxy.
            try:
                if _ctx:
                    await _ctx.close()  # Attempt to close old one if exists
            except Exception:
                pass

            try:
                # Load state before creating context
                loaded_state = await _load_state()  # Now uses async file read
                proxy_config = _get_proxy_config()  # Get proxy config

                final_context_args = default_args.copy()
                final_context_args["storage_state"] = loaded_state
                if proxy_config:
                    final_context_args["proxy"] = proxy_config

                _ctx = await _browser.new_context(**final_context_args)

                await _log(
                    "browser_context_create",
                    headless=is_headless,
                    proxy=bool(proxy_config),
                    args=final_context_args,
                )

                # Add proxy domain check hook if proxy is active
                if proxy_config:
                    await _add_proxy_routing_rule(_ctx, proxy_config)

                # Start maintenance loop only for the persistent context
                asyncio.create_task(_context_maintenance_loop(_ctx))

            except PlaywrightException as e:
                logger.critical(f"Failed to create shared browser context: {e}", exc_info=True)
                raise RuntimeError(f"Failed to create shared browser context: {e}") from e

        return _ctx, _browser


async def _add_proxy_routing_rule(context: BrowserContext, proxy_config: Dict[str, Any]):
    """Adds a routing rule to enforce PROXY_ALLOWED_DOMAINS if enabled."""
    if _PROXY_ALLOWED_DOMAINS is None:
        return  # No domain restrictions

    async def handle_route(route):
        request_url = route.request.url
        if not _is_domain_allowed_for_proxy(request_url):
            logger.warning(
                f"Proxy use blocked for disallowed domain: {request_url}. Aborting request."
            )
            try:
                await route.abort("accessdenied")  # Abort the request
            except PlaywrightException as e:
                logger.error(f"Error aborting route for {request_url}: {e}")
                # Try to continue without proxy as a fallback? Complex. Abort is safer.
        else:
            # Continue with the request (using the context's proxy settings)
            try:
                await route.continue_()
            except PlaywrightException as e:
                logger.error(f"Error continuing route for {request_url}: {e}")
                # Attempt to abort if continue fails
                try:
                    await route.abort()
                except Exception:
                    pass  # Ignore abort error

    try:
        # Route all network requests ('**/*')
        await context.route("**/*", handle_route)
        logger.info("Proxy domain restriction rule added to context.")
    except PlaywrightException as e:
        logger.error(f"Failed to add proxy domain routing rule: {e}")


def _start_vnc():
    """Starts X11VNC if VNC=1 and VNC_PASSWORD is set."""
    global _vnc_proc
    if _vnc_proc or os.getenv("VNC") != "1":
        return

    vnc_password = os.getenv("VNC_PASSWORD")
    if not vnc_password:
        logger.warning(
            "VNC=1 but VNC_PASSWORD env var not set. VNC server not started for security."
        )
        return

    display = os.getenv("DISPLAY", ":0")  # Default to :0 if not set

    try:
        # Check if x11vnc command exists
        if subprocess.run(["which", "x11vnc"], capture_output=True, text=True).returncode != 0:
            logger.warning("x11vnc command not found. Cannot start VNC server.")
            return

        # Prepare command
        cmd = [
            "x11vnc",
            "-display",
            display,
            "-passwd",
            vnc_password,
            "-forever",  # Keep running until killed
            "-localhost",  # Only allow connections from localhost (use SSH tunnel for remote access)
            "-nopw",  # Disable password prompt on server console
            "-quiet",  # Reduce logging
            "-noxdamage",  # May help with some compatibility issues
        ]

        # Fix: Use setsid to ensure VNC process is killed properly on termination
        preexec_fn = os.setsid if hasattr(os, "setsid") else None

        _vnc_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=preexec_fn,  # Create new session
        )
        logger.info(f"Password-protected VNC server started on display {display} (localhost only).")
        # Register cleanup function for VNC process
        atexit.register(_cleanup_vnc)

    except FileNotFoundError:
        logger.warning("x11vnc command not found. Cannot start VNC server.")
    except Exception as e:
        logger.error(f"Failed to start VNC server: {e}", exc_info=True)
        _vnc_proc = None


def _cleanup_vnc():
    """Terminates the VNC server process."""
    global _vnc_proc
    proc = _vnc_proc
    if proc and proc.poll() is None:  # Check if process exists and is running
        logger.info("Terminating VNC server process...")
        try:
            # Send SIGTERM to the process group (due to setsid)
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()  # Fallback for systems without killpg/getpgid

            proc.wait(timeout=5)  # Wait for termination
            logger.info("VNC server process terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("VNC server did not terminate gracefully. Sending SIGKILL.")
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process might have died between checks
            else:
                proc.kill()  # Fallback kill
            try:
                proc.wait(timeout=2)  # Wait briefly after kill
            except Exception:
                pass  # Ignore errors after kill
        except ProcessLookupError:
            logger.info("VNC process already terminated.")  # Process died before we could signal
        except Exception as e:
            logger.error(f"Error during VNC cleanup: {e}")
        finally:
            _vnc_proc = None


async def _load_state() -> dict[str, Any] | None:
    """Loads browser state from encrypted file asynchronously."""
    if not _STATE_FILE.exists():
        return None
    loop = asyncio.get_running_loop()
    try:
        # Read file asynchronously
        async with aiofiles.open(_STATE_FILE, "rb") as f:
            encrypted_data = await f.read()

        # Decrypt (synchronous CPU-bound operation, could move to executor if becomes bottleneck)
        # For typical state sizes, inline decryption is probably fine.
        decrypted_data = await loop.run_in_executor(_thread_pool, _dec, encrypted_data)

        # Fix #11 / CWE-359: _dec now returns None on failure
        if decrypted_data is None:
            logger.warning("Failed to decrypt browser state file. Starting with fresh state.")
            return None  # Signal failure

        return json.loads(decrypted_data)
    except FileNotFoundError:
        logger.info("Browser state file not found. Starting fresh.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse browser state JSON: {e}. State file might be corrupt.")
        _STATE_FILE.unlink(missing_ok=True)  # Remove corrupt file
        return None
    except Exception as e:
        logger.error(f"Failed to load browser state: {e}", exc_info=True)
        # Attempt removal on generic error too, might be corrupt
        _STATE_FILE.unlink(missing_ok=True)
        return None


async def _save_state(ctx: BrowserContext):
    """Saves browser state to encrypted file asynchronously."""
    if not ctx or not ctx.browser:  # Check if context is valid
        logger.warning("Attempted to save state for an invalid or closed context.")
        return
    loop = asyncio.get_running_loop()
    try:
        state = await ctx.storage_state()
        state_json = json.dumps(state).encode("utf-8")

        # Encrypt (synchronous CPU-bound, move to executor if needed)
        encrypted_data = await loop.run_in_executor(_thread_pool, _enc, state_json)

        # Write file asynchronously
        async with aiofiles.open(_STATE_FILE, "wb") as f:
            await f.write(encrypted_data)

        # Fix CWE-732: Ensure state file has restricted permissions
        await loop.run_in_executor(_thread_pool, os.chmod, _STATE_FILE, 0o600)

        # logger.debug("Browser state saved successfully.") # Optional debug log
    except PlaywrightException as e:
        logger.error(f"Failed to get storage state from browser context: {e}")
    except Exception as e:
        logger.error(f"Failed to save browser state: {e}", exc_info=True)


@asynccontextmanager
async def _tab_context(ctx: BrowserContext):
    """Async context manager for creating and cleaning up a Page."""
    page = None
    try:
        page = await ctx.new_page()
        await _log("page_open", context_id=id(ctx))  # Log context ID for correlation
        yield page
    except PlaywrightException as e:
        logger.error(f"Failed to create new page in context {id(ctx)}: {e}")
        # Raise a ToolError to indicate failure to the caller
        raise ToolError(f"Failed to create browser page: {e}") from e
    finally:
        if page and not page.is_closed():
            try:
                await page.close()
                await _log("page_close", context_id=id(ctx))
            except PlaywrightException as e:
                logger.warning(f"Error closing page in context {id(ctx)}: {e}")


async def _context_maintenance_loop(ctx: BrowserContext):
    """Periodically saves state and optionally clears cookies for the shared context."""
    # Interval for saving state (e.g., every 15 minutes)
    save_interval_seconds = 15 * 60
    clear_enabled = False  # Set to True to enable periodic clearing

    logger.info(f"Starting context maintenance loop for context {id(ctx)}.")

    while True:
        # Fix: Check if context is still connected/valid before proceeding
        if not ctx or not ctx.browser:
            logger.info(f"Context {id(ctx)} appears closed or invalid. Stopping maintenance loop.")
            break
        # Alternative check using Playwright's internal connection state if available
        # try:
        #     if not ctx._connection.is_connected: # Accessing private attribute is risky
        #         break
        # except AttributeError: pass # Ignore if attribute doesn't exist

        try:
            await asyncio.sleep(save_interval_seconds)

            if not ctx or not ctx.browser:
                break  # Re-check after sleep

            # Save state
            await _save_state(ctx)
            # logger.debug(f"Maintenance: State saved for context {id(ctx)}.") # Optional debug

            # Optional periodic cleanup (careful, might log user out)
            if clear_enabled:
                # Perform cleanup actions here if needed, e.g.:
                # await ctx.clear_cookies() # Be cautious with this
                # await ctx.clear_permissions()
                # logger.info(f"Maintenance: Cleared cookies/permissions for context {id(ctx)}.")
                pass

        except asyncio.CancelledError:
            logger.info(f"Context maintenance loop for {id(ctx)} cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in context maintenance loop for {id(ctx)}: {e}", exc_info=True)
            # Avoid tight loop on persistent error, wait before retrying
            await asyncio.sleep(60)


async def _try_close_browser():
    """Attempt to close the browser gracefully, used by atexit."""
    global _browser
    if _browser and _browser.is_connected():
        logger.info("Attempting to close browser via atexit handler...")
        try:
            await _browser.close()
            logger.info("Browser closed successfully.")
        except Exception as e:
            logger.error(f"Error closing browser during atexit: {e}")
        finally:
            _browser = None


async def shutdown():
    """Gracefully shut down Playwright, browser, context, and VNC."""
    global _pw, _browser, _ctx, _vnc_proc, _thread_pool, _selector_cleanup_task_handle
    logger.info("Initiating graceful shutdown...")

    # 0. Cancel background tasks first
    cleanup_task = _selector_cleanup_task_handle
    if cleanup_task and not cleanup_task.done():
        logger.info("Cancelling selector cleanup task during shutdown...")
        cleanup_task.cancel()
        # Don't necessarily await here, let gather handle it below? Or brief await.
        try:
            await asyncio.wait_for(cleanup_task, timeout=2.0)
        except Exception: 
            pass # Ignore errors during shutdown cancellation
        _selector_cleanup_task_handle = None

    # 1. Close Tab Pool tasks (if applicable, needs implementation)
    await tab_pool.cancel_all() # Assuming tab_pool is accessible

    # 2. Close shared context and save state
    async with _playwright_lock:  # Use lock to prevent races with get_browser_context
        ctx_to_close = _ctx
        _ctx = None  # Prevent new uses of the shared context
        if ctx_to_close and ctx_to_close.browser:
            try:
                await _save_state(ctx_to_close)
                await ctx_to_close.close()
                await _log("browser_context_close_shared")
            except Exception as e:
                logger.error(f"Error closing shared browser context: {e}")

        # 3. Close browser
        browser_to_close = _browser
        _browser = None
        if browser_to_close and browser_to_close.is_connected():
            try:
                await browser_to_close.close()
                await _log("browser_close")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")

        # 4. Stop Playwright
        pw_to_stop = _pw
        _pw = None
        if pw_to_stop:
            try:
                await pw_to_stop.stop()
                logger.info("Playwright stopped.")
            except Exception as e:
                logger.error(f"Error stopping Playwright: {e}")

    # 5. Cleanup VNC (synchronous, run outside lock)
    _cleanup_vnc()

    # 6. Shutdown thread pool
    logger.info("Shutting down thread pool...")
    _thread_pool.shutdown(wait=True)
    # If using ProcessPool:
    # if _process_pool: _process_pool.shutdown(wait=True)
    logger.info("Thread pool shut down.")

    await _log("browser_shutdown_complete")
    logger.info("Graceful shutdown complete.")


# Register shutdown hooks
_shutdown_initiated = False
_shutdown_lock = asyncio.Lock()


async def _initiate_shutdown():
    """Ensures shutdown runs only once."""
    global _shutdown_initiated
    async with _shutdown_lock:
        if not _shutdown_initiated:
            _shutdown_initiated = True
            await shutdown()


# Fix #2: Signal handler using call_soon_threadsafe
def _signal_handler(sig, frame):
    logger.info(f"Received signal {sig}. Initiating shutdown...")
    try:
        # Get the running loop safely
        loop = asyncio.get_event_loop()
        # Schedule the shutdown coroutine thread-safely
        if loop.is_running():
            loop.call_soon_threadsafe(asyncio.create_task, _initiate_shutdown())
        else:
            # If loop isn't running, run synchronously (may happen during interpreter shutdown)
            logger.warning(
                "Event loop not running during signal handling. Running shutdown synchronously."
            )
            asyncio.run(_initiate_shutdown())
    except RuntimeError as e:
        # Fallback if loop cannot be retrieved or task creation fails
        logger.error(
            f"Error scheduling shutdown from signal handler: {e}. Attempting synchronous shutdown."
        )
        try:
            asyncio.run(_initiate_shutdown())
        except Exception as final_e:
            logger.critical(f"Synchronous shutdown attempt failed: {final_e}")
    # Give shutdown a brief moment, then exit forcefully if needed
    # Note: Exiting here might cut off shutdown. Consider if this is desired.
    # time.sleep(3) # Allow some time for shutdown tasks
    # sys.exit(0) # Commented out: Allow shutdown to complete fully


# Register signal handlers
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except ValueError:
    logger.warning("Could not register signal handlers (likely not in main thread).")

# Register atexit hook (runs on normal exit)
# Note: atexit handlers run synchronously and might conflict with async shutdown
# It's often better to rely on explicit shutdown calls or signal handlers.
# However, keeping it as a fallback.
_atexit_shutdown_done = False


def _atexit_shutdown_sync():
    global _atexit_shutdown_done
    if not _atexit_shutdown_done:
        _atexit_shutdown_done = True
        logger.info("atexit hook triggered. Running shutdown.")
        try:
            # If loop exists and is running, schedule; otherwise, run sync
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cannot directly await here. Submit and wait briefly? Risky.
                logger.warning(
                    "Cannot reliably run async shutdown from synchronous atexit handler with running loop."
                )
                # Best effort: close DB connection synchronously
                _close_db_connection()
            else:
                asyncio.run(_initiate_shutdown())
        except RuntimeError:  # Loop might be closed
            _close_db_connection()  # Close DB at least
        except Exception as e:
            logger.error(f"Error during atexit shutdown: {e}")


# Commenting out atexit for now, relying on signals and explicit cleanup.
# Re-enable if necessary, but be aware of sync/async issues.
# atexit.register(_atexit_shutdown_sync)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TAB POOL FOR PARALLELISM
# ══════════════════════════════════════════════════════════════════════════════
class TabPool:
    """Runs async callables needing a Page in parallel, bounded by SB_MAX_TABS."""

    def __init__(self, max_tabs: int | None = None):
        try:
            default_max = int(os.getenv("SB_MAX_TABS", "5"))
        except ValueError:
            default_max = 5
            logger.warning(f"Invalid SB_MAX_TABS value. Using default: {default_max}")

        self.max_tabs = max_tabs if max_tabs is not None else default_max
        if self.max_tabs <= 0:
            logger.warning("max_tabs must be positive. Setting to 1.")
            self.max_tabs = 1

        self.sem = asyncio.Semaphore(self.max_tabs)
        self._active_contexts: Set[BrowserContext] = set()
        self._context_lock = asyncio.Lock()
        logger.info(f"TabPool initialized with max_tabs={self.max_tabs}")

    async def _run(self, fn: Callable[[Page], Awaitable[Any]]) -> Any:
        """Acquires semaphore, creates incognito context+page, runs fn, cleans up."""
        timeout_seconds_str = os.getenv("SB_TAB_TIMEOUT", "300")  # 5-minute default
        try:
            timeout_seconds = int(timeout_seconds_str)
        except ValueError:
            timeout_seconds = 300
            logger.warning(
                f"Invalid SB_TAB_TIMEOUT value '{timeout_seconds_str}'. Using default 300s."
            )

        incognito_ctx: Optional[BrowserContext] = None
        task = asyncio.current_task()  # Get current task for tracking/logging

        try:
            # Wrap the core logic in wait_for
            async with self.sem:  # Acquire semaphore before creating resources
                # Use incognito context for isolation
                incognito_ctx, _ = await get_browser_context(use_incognito=True)

                # Track the active context for potential cleanup on cancellation
                async with self._context_lock:
                    self._active_contexts.add(incognito_ctx)

                # Context manager handles page creation and closure
                async with _tab_context(incognito_ctx) as page:
                    # Execute the provided function with the page
                    result = await asyncio.wait_for(fn(page), timeout=timeout_seconds)
                    return result  # Return result on success

        except asyncio.TimeoutError:
            func_name = getattr(fn, "__name__", "anonymous_tab_function")
            await _log("tab_timeout", function=func_name, timeout=timeout_seconds, task_id=id(task))
            # Fix #Tab Leaks: Ensure context is closed even on timeout
            # The cancellation happens *outside* this block, need cleanup in finally.
            # We need to return an error indicator.
            return {"error": f"Tab operation timed out after {timeout_seconds}s", "success": False}
        except asyncio.CancelledError:
            func_name = getattr(fn, "__name__", "anonymous_tab_function")
            await _log("tab_cancelled", function=func_name, task_id=id(task))
            # Propagate cancellation, cleanup happens in finally
            raise
        except Exception as e:
            func_name = getattr(fn, "__name__", "anonymous_tab_function")
            await _log("tab_error", function=func_name, error=str(e), task_id=id(task))
            # Return error indicator
            return {"error": f"Tab operation failed: {e}", "success": False}
        finally:
            # Fix Resource Leak (Tab Leaks): Ensure the incognito context is *always* closed
            if incognito_ctx:
                async with self._context_lock:
                    self._active_contexts.discard(incognito_ctx)
                try:
                    await incognito_ctx.close()
                except PlaywrightException as close_err:
                    logger.warning(
                        f"Error closing incognito context {id(incognito_ctx)}: {close_err}"
                    )

    async def map(self, fns: Sequence[Callable[[Page], Awaitable[Any]]]) -> List[Any]:
        """Runs multiple functions in parallel using the tab pool."""
        if not fns:
            return []

        # Fix #3: Use asyncio.gather for compatibility with Python < 3.11
        tasks = [asyncio.create_task(self._run(fn)) for fn in fns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, logging any exceptions returned by gather
        processed_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                func_name = getattr(fns[i], "__name__", f"function_{i}")
                logger.error(
                    f"Error during TabPool.map for function '{func_name}': {res}", exc_info=res
                )
                processed_results.append(
                    {"error": f"Task execution failed: {res}", "success": False}
                )
            else:
                processed_results.append(res)

        return processed_results

    async def cancel_all(self):
        """Attempts to close all active contexts managed by the pool."""
        async with self._context_lock:
            contexts_to_close = list(self._active_contexts)  # Copy set for iteration
            self._active_contexts.clear()  # Clear the tracking set

        if not contexts_to_close:
            logger.info("TabPool cancel_all: No active contexts to close.")
            return

        logger.info(
            f"TabPool cancel_all: Attempting to close {len(contexts_to_close)} active contexts."
        )
        close_tasks = [asyncio.create_task(ctx.close()) for ctx in contexts_to_close]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)

        closed_count = 0
        error_count = 0
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                error_count += 1
                logger.warning(
                    f"Error closing context {id(contexts_to_close[i])} during cancel_all: {res}"
                )
            else:
                closed_count += 1
        logger.info(f"TabPool cancel_all: {closed_count} contexts closed, {error_count} errors.")


# Global instance (can be replaced if needed)
tab_pool = TabPool()

# ══════════════════════════════════════════════════════════════════════════════
# 6.  HUMAN-LIKE JITTER (bot evasion)
# ══════════════════════════════════════════════════════════════════════════════
_HIGH_RISK_DOMAINS: Optional[Set[str]] = None


def _get_high_risk_domains() -> Set[str]:
    """Parses and caches high-risk domains from environment."""
    global _HIGH_RISK_DOMAINS
    if _HIGH_RISK_DOMAINS is None:
        domains_str = os.getenv(
            "HIGH_RISK_DOMAINS",
            # Common sites known for bot detection
            ".google.com,.facebook.com,.linkedin.com,.glassdoor.com,"
            ".instagram.com,.twitter.com,.x.com,.reddit.com,"
            ".amazon.com,.ebay.com,.ticketmaster.com,.cloudflare.com,"
            ".datadome.co,.perimeterx.net,.recaptcha.net,.hcaptcha.com",
        )
        raw_domains = [d.strip().lower() for d in domains_str.split(",") if d.strip()]
        # Normalize to ensure leading dot for suffix matching
        _HIGH_RISK_DOMAINS = {d if d.startswith(".") else "." + d for d in raw_domains}
        logger.info(f"Initialized high-risk domains for jitter timing: {_HIGH_RISK_DOMAINS}")
    return _HIGH_RISK_DOMAINS


def _risk_factor(url: str) -> float:
    """Calculates a risk factor based on the URL's domain."""
    if not url:
        return 1.0
    try:
        domain = urlparse(url).netloc.lower()
        if not domain:
            return 1.0

        high_risk_set = _get_high_risk_domains()
        domain_parts = domain.split(".")
        for i in range(len(domain_parts)):
            sub_domain = "." + ".".join(domain_parts[i:])
            if sub_domain in high_risk_set:
                # logger.debug(f"High risk domain detected: {domain} (matched {sub_domain})")
                return 2.0  # Higher base factor for known sensitive domains
        return 1.0
    except Exception:
        return 1.0  # Default on parsing error


async def _pause(page: Page, base_ms_range: tuple[int, int] = (150, 500)):
    """Introduce a short, randomized pause, adjusted by URL risk factor."""
    if not page or page.is_closed():
        return

    risk = _risk_factor(page.url)
    min_ms, max_ms = base_ms_range

    # Calculate base delay
    base_delay_ms = random.uniform(min_ms, max_ms)

    # Apply risk factor
    adjusted_delay_ms = base_delay_ms * risk

    # Add slight extra jitter based on page complexity heuristic (optional)
    try:
        # Count interactive elements visible in viewport? Simpler: element count.
        element_count = await page.evaluate(
            "() => document.querySelectorAll('a, button, input, select, textarea, [role=button], [role=link], [onclick]').length"
        )
        complexity_factor = min(1.0 + (element_count / 500.0), 1.5)  # Scale 1.0 to 1.5
        adjusted_delay_ms *= complexity_factor
    except PlaywrightException:
        pass  # Ignore if evaluation fails

    # Ensure delay isn't excessively long (e.g., max 3 seconds)
    final_delay_ms = min(adjusted_delay_ms, 3000)
    final_delay_sec = final_delay_ms / 1000.0

    # logger.debug(f"Pausing for {final_delay_sec:.3f}s (Risk: {risk:.1f}, Base: {base_delay_ms:.0f}ms)")
    await asyncio.sleep(final_delay_sec)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SMART LOCATOR + ADAPTIVE SELECTOR LEARNING
# ══════════════════════════════════════════════════════════════════════════════
class ElementNotFoundError(ToolError):
    """Custom exception for when SmartLocator fails to find an element."""

    def __init__(self, strategies_tried: List[str], target_info: Dict[str, Any]):
        self.strategies_tried = strategies_tried
        self.target_info = target_info
        super().__init__(
            f"Element not found using strategies: {', '.join(strategies_tried)}. Target: {target_info}",
            error_code="element_not_found",
        )


class SmartLocator:
    def __init__(self, page: Page, timeout_ms: int = 5000):  # Default 5s timeout
        if not page or page.is_closed():
            raise ValueError("SmartLocator requires a valid, open Page object.")
        self.p = page
        self.t = timeout_ms
        try:
            self.site = urlparse(page.url or "").netloc
            if not self.site:
                logger.warning("SmartLocator could not determine site from page URL.")
        except Exception:
            self.site = ""
            logger.warning("SmartLocator failed to parse page URL.")

    async def _try_locator(
        self,
        locator: Locator,
        strategy_name: str,
        key: str,
        css: Optional[str] = None,
        xpath: Optional[str] = None,
    ) -> Optional[Locator]:
        """Tries to find and validate a locator, bumping score on success."""
        try:
            # Check visibility first - more reliable than attached sometimes
            # Use a shorter timeout for individual checks
            check_timeout = max(500, self.t / 5)  # e.g., 1 second if main timeout is 5s
            await locator.wait_for(state="visible", timeout=check_timeout)
            # Optional: Check if it's the *only* match for more specificity?
            # count = await locator.count()
            # if count == 1:
            if self.site and key:  # Only bump if we have site/key info
                # If successful, bump the score in the background
                # Use the original css/xpath values passed in, not generated ones
                asyncio.create_task(_bump_selector(self.site, key, css, xpath))
            return locator
        except PlaywrightTimeoutError:
            # logger.debug(f"Locator strategy '{strategy_name}' timed out waiting for visible.")
            return None
        except PlaywrightException:
            # Fix #5: Catch broader Playwright errors for this strategy
            # logger.debug(f"Locator strategy '{strategy_name}' failed: {e}")
            return None
        except Exception as e:
            # Catch any other unexpected error
            logger.warning(
                f"Unexpected error in locator strategy '{strategy_name}': {e}", exc_info=True
            )
            return None

    async def _by_learned(self, key: str) -> Optional[Locator]:
        if not self.site or not key:
            return None
        selectors = await _get_best_selectors(self.site, key)
        for css, xpath in selectors:
            if css:
                locator = self.p.locator(css).first
                found = await self._try_locator(locator, "learned_css", key=key, css=css)
                if found:
                    await _log("locator_hit", method="learned_css", key=key, selector=css)
                    return found
            if xpath:
                locator = self.p.locator(f"xpath={xpath}").first
                found = await self._try_locator(locator, "learned_xpath", key=key, xpath=xpath)
                if found:
                    await _log("locator_hit", method="learned_xpath", key=key, selector=xpath)
                    return found
        return None

    async def _by_css(self, css: Optional[str], key: str) -> Optional[Locator]:
        if not css:
            return None
        return await self._try_locator(self.p.locator(css).first, "css", key=key, css=css)

    async def _by_xpath(self, xpath: Optional[str], key: str) -> Optional[Locator]:
        if not xpath:
            return None
        return await self._try_locator(
            self.p.locator(f"xpath={xpath}").first, "xpath", key=key, xpath=xpath
        )

    async def _by_role(
        self, role: Optional[str], name: Optional[str], key: str
    ) -> Optional[Locator]:
        if not role:
            return None
        # Try exact match first
        locator_exact = self.p.get_by_role(role, name=name, exact=True).first
        found = await self._try_locator(locator_exact, f"role_exact_{role}_{name}", key=key)
        if found:
            return found
        # Fallback to non-exact if name provided
        if name:
            locator_inexact = self.p.get_by_role(role, name=name, exact=False).first
            return await self._try_locator(locator_inexact, f"role_inexact_{role}_{name}", key=key)
        return None

    async def _by_label(self, name: Optional[str], key: str) -> Optional[Locator]:
        if not name:
            return None
        locator_exact = self.p.get_by_label(name, exact=True).first
        found = await self._try_locator(locator_exact, f"label_exact_{name}", key=key)
        if found:
            return found
        locator_inexact = self.p.get_by_label(name, exact=False).first
        return await self._try_locator(locator_inexact, f"label_inexact_{name}", key=key)

    async def _by_text(self, name: Optional[str], key: str) -> Optional[Locator]:
        if not name:
            return None
        locator_exact = self.p.get_by_text(name, exact=True).first
        found = await self._try_locator(locator_exact, f"text_exact_{name}", key=key)
        if found:
            return found
        locator_inexact = self.p.get_by_text(name, exact=False).first
        return await self._try_locator(locator_inexact, f"text_inexact_{name}", key=key)

    async def _by_placeholder(self, name: Optional[str], key: str) -> Optional[Locator]:
        if not name:
            return None
        locator = self.p.get_by_placeholder(name).first  # Exact seems common here
        return await self._try_locator(locator, f"placeholder_{name}", key=key)

    async def _by_alt_text(self, name: Optional[str], key: str) -> Optional[Locator]:
        if not name:
            return None
        locator = self.p.get_by_alt_text(name).first  # Exact seems common here
        return await self._try_locator(locator, f"alt_text_{name}", key=key)

    async def _fuzzy(self, name: Optional[str], key: str) -> Optional[Locator]:
        """Find element by fuzzy text matching (use cautiously)."""
        if not name or len(name) < 3:
            return None  # Avoid matching on very short strings

        try:
            # Normalize the input text for comparison
            name_norm = unicodedata.normalize("NFC", name.lower())

            # Get text content of potentially relevant elements
            # Limit the scope to avoid performance hit
            candidate_elements = await self.p.locator(
                "a, button, input, label, h1, h2, h3, h4, span, strong, em, p, div[role~=button], div[role~=link]"
            ).all()

            texts_with_locators = []
            for el in candidate_elements[:200]:  # Limit number of elements checked
                try:
                    # Check visibility quickly
                    if not await el.is_visible(timeout=100):
                        continue
                    # Get text content, normalize, limit length
                    text = await el.text_content()
                    if text:
                        norm_text = unicodedata.normalize("NFC", text.strip().lower()[:128])
                        if norm_text:
                            texts_with_locators.append((norm_text, el))
                except PlaywrightException:
                    continue  # Ignore errors getting text/visibility for one element

            if not texts_with_locators:
                return None

            # Use difflib to find the best match
            candidates = [t[0] for t in texts_with_locators]
            matches = difflib.get_close_matches(
                name_norm, candidates, n=1, cutoff=0.6
            )  # Adjust cutoff as needed

            if not matches:
                return None

            best_match_text = matches[0]
            # Find the corresponding locator
            for text, locator in texts_with_locators:
                if text == best_match_text:
                    # Verify the found locator again before returning
                    return await self._try_locator(locator, f"fuzzy_{name}", key=key)
            return None  # Should not happen if match found

        except PlaywrightException as e:
            logger.debug(f"Fuzzy locator strategy failed for '{name}': {e}")
            return None
        except Exception as e:
            logger.warning(
                f"Unexpected error in fuzzy locator strategy for '{name}': {e}", exc_info=True
            )
            return None

    async def locate(
        self,
        *,
        name: Optional[str] = None,  # Text content, label, placeholder, alt text
        role: Optional[str] = None,  # ARIA role
        css: Optional[str] = None,  # CSS selector
        xpath: Optional[str] = None,  # XPath selector
    ) -> Locator:
        """
        Find an element using multiple strategies, returning the first valid match.
        Raises ElementNotFoundError if no element is found.

        Strategies (ordered):
        1. Learned Selectors (CSS/XPath from cache)
        2. Explicit CSS / XPath
        3. ARIA Role (exact then inexact)
        4. Label (exact then inexact)
        5. Placeholder Text
        6. Alt Text (for images)
        7. Text Content (exact then inexact)
        8. Fuzzy Text Match (if name provided)
        """
        key = _sel_key(role, name) if (role or name) else ""
        target_info = {"name": name, "role": role, "css": css, "xpath": xpath, "key": key}
        strategies_tried: List[str] = []

        # Define strategies as functions returning awaitables or None
        strategy_fns: List[Tuple[str, Callable[[], Awaitable[Optional[Locator]]]]] = []

        # 1. Learned Selectors
        if key:
            strategy_fns.append(("Learned", lambda: self._by_learned(key)))
        # 2. Explicit CSS/XPath
        if css:
            strategy_fns.append(("CSS", lambda: self._by_css(css, key)))
        if xpath:
            strategy_fns.append(("XPath", lambda: self._by_xpath(xpath, key)))
        # 3. Role
        if role:
            strategy_fns.append(("Role", lambda: self._by_role(role, name, key)))
        # 4. Label
        if name:
            strategy_fns.append(("Label", lambda: self._by_label(name, key)))
        # 5. Placeholder
        if name:
            strategy_fns.append(("Placeholder", lambda: self._by_placeholder(name, key)))
        # 6. Alt Text
        if name:
            strategy_fns.append(("Alt Text", lambda: self._by_alt_text(name, key)))
        # 7. Text Content
        if name:
            strategy_fns.append(("Text", lambda: self._by_text(name, key)))
        # 8. Fuzzy Match (last resort if name provided)
        if name:
            strategy_fns.append(("Fuzzy", lambda: self._fuzzy(name, key)))

        # Execute strategies sequentially
        for name, fn in strategy_fns:
            strategies_tried.append(name)
            try:
                element = await fn()
                if element:
                    await _log("locator_success", method=name, target=target_info)
                    return element  # Return the first successful locator
            except Exception as e:
                # Should be caught by _try_locator, but as a safeguard
                logger.warning(
                    f"Unexpected error during locator strategy '{name}': {e}", exc_info=True
                )

        # Fix #6: If all strategies fail, raise a specific error instead of returning 'body'
        await _log("locator_fail", tried=strategies_tried, target=target_info)
        raise ElementNotFoundError(strategies_tried, target_info)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SMART ACTIONS WITH RETRY SUPPORT
# ══════════════════════════════════════════════════════════════════════════════


@resilient(max_attempts=3, backoff=0.5)  # Retry on transient errors
async def smart_click(page: Page, **target_kwargs) -> bool:
    """
    Locates an element using SmartLocator and clicks it with human-like pauses.

    Args:
        page: The Playwright Page object.
        **target_kwargs: Keyword arguments for SmartLocator.locate()
                         (e.g., name, role, css, xpath).

    Returns:
        True if click was successful, False otherwise.

    Raises:
        ElementNotFoundError: If the element cannot be located.
        ToolError: For other unexpected errors during click.
    """
    loc = SmartLocator(page)
    log_target = {k: v for k, v in target_kwargs.items() if v is not None}
    try:
        element = await loc.locate(**target_kwargs)
        await _pause(page)  # Pause before interacting

        # Perform the click with built-in checks
        await element.click(timeout=loc.t)  # Use locator's timeout

        await _log("click_success", target=log_target)
        return True
    except ElementNotFoundError as e:
        # Element not found is not typically something to retry, re-raise
        await _log("click_fail_notfound", target=log_target, error=str(e))
        raise e  # Propagate the specific error
    except PlaywrightException as e:
        # Catch Playwright errors during the click itself (e.g., element obscured)
        # This might be caught by @resilient for retry if it's a TimeoutError etc.
        await _log("click_fail_playwright", target=log_target, error=str(e))
        # Let resilient handle retry, or raise if max attempts reached
        raise ToolError(f"Click failed due to Playwright error: {e}", details=log_target) from e
    except Exception as e:
        # Catch other unexpected errors
        await _log("click_fail_unexpected", target=log_target, error=str(e))
        raise ToolError(f"Unexpected error during click: {e}", details=log_target) from e


@resilient(max_attempts=3, backoff=0.5)
async def smart_type(
    page: Page, text: str, press_enter: bool = False, clear_before: bool = True, **target_kwargs
) -> bool:
    """
    Locates an element, types text into it, and optionally presses Enter.
    Handles secret resolution.

    Args:
        page: The Playwright Page object.
        text: The text to type. If starts with "secret:", resolves via get_secret().
        press_enter: If True, press Enter key after typing.
        clear_before: If True, clear the input field before typing.
        **target_kwargs: Keyword arguments for SmartLocator.locate().

    Returns:
        True if typing was successful, False otherwise.

    Raises:
        ElementNotFoundError: If the element cannot be located.
        ToolError: For secret resolution errors or other issues.
    """
    loc = SmartLocator(page)
    log_target = {k: v for k, v in target_kwargs.items() if v is not None}
    resolved_text = text

    if text.startswith("secret:"):
        secret_path = text[len("secret:") :]
        try:
            resolved_text = get_secret(text)  # Handles env: and vault:
            log_value = "***SECRET***"
        except (KeyError, ValueError, RuntimeError) as e:
            await _log("type_fail_secret", target=log_target, secret_ref=secret_path, error=str(e))
            raise ToolInputError(f"Failed to resolve secret '{secret_path}': {e}") from e
    else:
        # Log potentially sensitive text carefully
        log_value = resolved_text[:20] + "..." if len(resolved_text) > 23 else resolved_text

    try:
        element = await loc.locate(**target_kwargs)
        await _pause(page)  # Pause before interaction

        if clear_before:
            await element.fill("")  # Clear the field first

        # Type text with a slight delay between keystrokes for human-like appearance
        await element.type(resolved_text, delay=random.uniform(30, 80))

        if press_enter:
            await _pause(page, (50, 150))  # Short pause before Enter
            await element.press("Enter")

        await _log("type_success", target=log_target, value=log_value, entered=press_enter)
        return True
    except ElementNotFoundError as e:
        await _log("type_fail_notfound", target=log_target, value=log_value, error=str(e))
        raise e
    except PlaywrightException as e:
        await _log("type_fail_playwright", target=log_target, value=log_value, error=str(e))
        raise ToolError(
            f"Type operation failed due to Playwright error: {e}", details=log_target
        ) from e
    except Exception as e:
        await _log("type_fail_unexpected", target=log_target, value=log_value, error=str(e))
        raise ToolError(f"Unexpected error during type: {e}", details=log_target) from e


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DOWNLOAD HELPER WITH DOCUMENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


async def _run_in_thread(func, *args):
    """Helper to run synchronous function in the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_thread_pool, func, *args)


async def _compute_hash_async(data: bytes) -> str:
    """Compute SHA-256 hash in a thread pool."""
    return await _run_in_thread(lambda d: hashlib.sha256(d).hexdigest(), data)


async def _read_file_async(path: Path) -> bytes:
    """Read file bytes asynchronously using aiofiles."""
    async with aiofiles.open(path, mode="rb") as f:
        return await f.read()


async def _write_file_async(path: Path, data: bytes):
    """Write file bytes asynchronously using aiofiles."""
    async with aiofiles.open(path, mode="wb") as f:
        await f.write(data)


def _extract_tables_sync(path: Path) -> List[Dict]:
    """Synchronous table extraction logic (runs in thread pool)."""
    ext = path.suffix.lower()
    results = []
    try:
        if ext == ".pdf":
            try:
                import tabula  # type: ignore

                # Run tabula-java, requires Java runtime
                dfs = tabula.read_pdf(
                    str(path), pages="all", multiple_tables=True, pandas_options={"dtype": str}
                )
                if dfs:
                    results = [
                        {"type": "pdf_table", "page": i + 1, "rows": df.to_dict(orient="records")}
                        for i, df in enumerate(dfs)
                    ]
            except ImportError:
                logger.warning(
                    "Python library 'tabula-py' not installed. Cannot extract PDF tables."
                )
            except Exception as pdf_err:  # Catch Tabula-specific errors
                logger.warning(f"Tabula PDF table extraction failed for {path.name}: {pdf_err}")

        elif ext in (".xls", ".xlsx"):
            try:
                import pandas as pd  # type: ignore

                # Read all sheets
                xl = pd.read_excel(str(path), sheet_name=None, dtype=str)
                results = [
                    {
                        "type": "excel_sheet",
                        "sheet_name": name,
                        "rows": df.to_dict(orient="records"),
                    }
                    for name, df in xl.items()
                ]
            except ImportError:
                logger.warning(
                    "Python library 'pandas' and 'openpyxl'/'xlrd' not installed. Cannot extract Excel tables."
                )
            except Exception as excel_err:
                logger.warning(f"Pandas Excel table extraction failed for {path.name}: {excel_err}")

        elif ext == ".csv":
            try:
                import pandas as pd  # type: ignore

                df = pd.read_csv(str(path), dtype=str)
                results = [{"type": "csv_table", "rows": df.to_dict(orient="records")}]
            except ImportError:
                logger.warning("Python library 'pandas' not installed. Cannot extract CSV tables.")
            except Exception as csv_err:
                logger.warning(f"Pandas CSV table extraction failed for {path.name}: {csv_err}")

    except Exception as outer_err:
        # Catch errors loading dependencies or unexpected issues
        logger.error(f"Error during table extraction setup for {path.name}: {outer_err}")

    return results


async def _extract_tables_async(path: Path) -> list:
    """Extract tables from document files (PDF, Excel, CSV) using thread pool."""
    try:
        # Run the synchronous extraction logic in the thread pool
        tables = await _run_in_thread(_extract_tables_sync, path)
        if tables:
            await _log("table_extract_success", file=str(path), num_tables=len(tables))
        return tables
    except Exception as e:
        # Log error from the async wrapper/executor itself
        await _log("table_extract_error", file=str(path), error=str(e))
        return []


@resilient()  # Retry download initiation on transient errors
async def smart_download(
    page: Page, target: Dict[str, Any], dest_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Clicks a target element to initiate a download, saves the file,
    calculates its hash, extracts tables if applicable, and sets permissions.

    Args:
        page: The Playwright Page object.
        target: Target element specification for SmartLocator (e.g., name, role, css).
        dest_dir: Optional destination directory. Defaults to ~/.smart_browser/downloads.

    Returns:
        Dictionary with download info (path, hash, size, tables) or error.
    """
    # Determine destination directory
    if dest_dir is None:
        download_base_dir = _HOME / "downloads"
    else:
        download_base_dir = Path(dest_dir)

    # Fix CWE-732: Ensure base directory exists and has secure permissions if created
    if not download_base_dir.exists():
        logger.info(f"Creating download directory: {download_base_dir}")
        await _run_in_thread(lambda p: p.mkdir(parents=True, exist_ok=True), download_base_dir)
        await _run_in_thread(os.chmod, download_base_dir, 0o700)
    elif not os.access(download_base_dir, os.W_OK):
        logger.error(f"Download directory {download_base_dir} is not writable.")
        raise ToolError(f"Download directory not writable: {download_base_dir}")

    log_target = {k: v for k, v in target.items() if v is not None}

    try:
        # Initiate download by clicking the target element
        async with page.expect_download(
            timeout=60000
        ) as dl_info:  # Wait up to 60s for download start
            await smart_click(page, **target)  # Use smart_click for robust clicking

        dl = await dl_info.value  # Get the Download object
        suggested_fname = dl.suggested_filename or f"downloaded_file_{int(time.time())}"
        out_path = download_base_dir / suggested_fname

        # Save the file (Playwright handles the stream internally)
        await dl.save_as(out_path)

        # Fix CWE-732: Set permissions on the downloaded file (owner read/write)
        await _run_in_thread(os.chmod, out_path, 0o600)

        # Read file content asynchronously for hashing
        # This might be inefficient for very large files. Consider streaming hash if needed.
        try:
            file_data = await _read_file_async(out_path)
            file_size = len(file_data)
            sha256_hash = await _compute_hash_async(file_data)  # Hash in thread pool
        except Exception as read_hash_err:
            logger.error(f"Error reading or hashing downloaded file {out_path}: {read_hash_err}")
            # Proceed without hash/size if reading fails, but log the issue
            file_size = -1
            sha256_hash = "Error"

        # Extract tables in the background (don't block return)
        # Run extraction in thread pool to avoid blocking event loop
        table_extraction_task = asyncio.create_task(_extract_tables_async(out_path))

        # Wait for extraction and get results (with a timeout?)
        # For now, let it run in background and return immediately.
        # If tables are needed *in the result*, await the task here:
        try:
            tables = await asyncio.wait_for(
                table_extraction_task, timeout=120
            )  # 2 min timeout for extraction
        except asyncio.TimeoutError:
            logger.warning(f"Table extraction timed out for {out_path.name}")
            tables = []
        except Exception as extract_err:
            logger.error(f"Table extraction failed for {out_path.name}: {extract_err}")
            tables = []

        # Prepare result
        info = {
            "success": True,
            "file_path": str(out_path),
            "file_name": suggested_fname,
            "sha256": sha256_hash,
            "size_bytes": file_size,
            "url": dl.url,  # URL the download originated from
            "tables_extracted": bool(tables),
            "tables": tables[:5],  # Limit payload size
        }
        await _log("download_success", target=log_target, **info)
        return info

    except ElementNotFoundError as e:
        await _log("download_fail_notfound", target=log_target, error=str(e))
        raise ToolError(f"Download failed: Target element not found. {e}") from e
    except PlaywrightTimeoutError as e:
        await _log("download_fail_timeout", target=log_target, error=str(e))
        raise ToolError(
            f"Download failed: Timed out waiting for download to start or complete. {e}"
        ) from e
    except PlaywrightException as e:
        await _log("download_fail_playwright", target=log_target, error=str(e))
        # Let resilient handle retries if applicable
        raise ToolError(f"Download failed due to Playwright error: {e}") from e
    except Exception as e:
        await _log("download_fail_unexpected", target=log_target, error=str(e))
        raise ToolError(f"Unexpected error during download: {e}") from e


# --- PDF/Docs Crawler Helpers ---
_SLUG_RE = re.compile(r"[^a-z0-9\-_]+")  # Allow letters, numbers, hyphen, underscore


def _slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem/URL-friendly slug."""
    if not text:
        return "file"
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text).lower()
    # Replace invalid chars with hyphen
    slug = _SLUG_RE.sub("-", text).strip("-")
    # Remove consecutive hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    # Truncate
    slug = slug[:max_len].strip("-")
    return slug or "file"  # Fallback if slug becomes empty


def _get_dir_slug(url: str) -> str:
    """Generate a slug from the directory path of a URL."""
    try:
        path_parts = Path(urlparse(url).path).parts
        # Use last 2-3 non-empty directory parts, ignore filename part
        dir_parts = [p for p in path_parts[:-1] if p and p != "/"]
        if len(dir_parts) >= 2:
            return f"{_slugify(dir_parts[-2], 20)}-{_slugify(dir_parts[-1], 20)}"
        elif len(dir_parts) == 1:
            return _slugify(dir_parts[-1], 40)
        else:
            # Fallback using netloc if path is shallow
            netloc_slug = _slugify(urlparse(url).netloc, 40)
            return netloc_slug or "domain"
    except Exception:
        return "path"  # Fallback on error


async def _fetch_html(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """Fetch HTML content from a URL, handling errors."""
    try:
        # Use stream=True to check headers before downloading large content
        async with client.stream("GET", url, follow_redirects=True, timeout=20.0) as response:
            # Raise error for non-2xx responses
            response.raise_for_status()
            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                logger.debug(f"Skipping non-HTML content at {url} (Type: {content_type})")
                return None
            # Check content length (optional, avoid huge pages)
            content_length = int(response.headers.get("content-length", 0))
            if content_length > 5 * 1024 * 1024:  # 5 MB limit
                logger.warning(f"Skipping large HTML page at {url} (Size: {content_length} bytes)")
                return None

            # Read the content
            html = await response.aread()
            # Decode carefully, try common encodings
            try:
                return html.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return html.decode("iso-8859-1")
                except UnicodeDecodeError:
                    logger.warning(f"Could not decode HTML from {url} using utf-8 or iso-8859-1")
                    return None  # Give up if decoding fails

    except httpx.HTTPStatusError as e:
        # Log client/server errors, but don't treat as critical crawl failure
        if 400 <= e.response.status_code < 500:
            logger.info(f"Client error fetching {url}: {e.response.status_code}")
        elif 500 <= e.response.status_code < 600:
            logger.warning(f"Server error fetching {url}: {e.response.status_code}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Network error fetching {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
        return None


def _extract_links(base_url: str, html: str) -> Tuple[List[str], List[str]]:
    """Extract PDF links and same-domain page links from HTML."""
    pdfs: Set[str] = set()
    pages: Set[str] = set()
    try:
        soup = BeautifulSoup(html, "html.parser")
        base_netloc = urlparse(base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue

            try:
                # Resolve relative URLs
                abs_url = urllib.parse.urljoin(base_url, href)
                # Clean URL (remove fragment, normalize)
                parsed_url = urlparse(abs_url)
                clean_url = parsed_url._replace(fragment="").geturl()

                # Check if it's a PDF link
                if parsed_url.path.lower().endswith(".pdf"):
                    pdfs.add(clean_url)
                # Check if it's a same-domain HTML page (heuristic)
                elif parsed_url.netloc == base_netloc and (
                    parsed_url.path.lower().endswith((".html", ".htm", "/"))
                    or "." not in Path(parsed_url.path).name
                ):
                    # Avoid adding PDFs to the pages list
                    if not clean_url.lower().endswith(".pdf"):
                        pages.add(clean_url)

            except ValueError:
                logger.debug(f"Could not parse or join URL: base='{base_url}', href='{href}'")
            except Exception as link_err:
                logger.warning(f"Error processing link '{href}' on page {base_url}: {link_err}")

    except Exception as soup_err:
        logger.error(f"Error parsing HTML for links on {base_url}: {soup_err}")

    return list(pdfs), list(pages)


# Fix #14: Rate limiter logic corrected
class RateLimiter:
    """Limits async operations to a maximum rate (requests per second)."""

    def __init__(self, rate_limit: float = 1.0):
        if rate_limit <= 0:
            raise ValueError("Rate limit must be positive")
        self.rate_limit = rate_limit
        self.interval = 1.0 / rate_limit
        self.last_request_time = 0
        self.lock = asyncio.Lock()  # Ensure atomic check-sleep-update

    async def acquire(self):
        """Wait if necessary to maintain the rate limit."""
        async with self.lock:
            now = time.monotonic()  # Use monotonic clock for intervals
            time_since_last = now - self.last_request_time

            # Calculate time needed before next request is allowed
            time_to_wait = self.interval - time_since_last

            if time_to_wait > 0:
                # logger.debug(f"RateLimiter waiting for {time_to_wait:.3f}s")
                await asyncio.sleep(time_to_wait)
                # Update 'now' after sleeping
                now = time.monotonic()

            # Update last request time *after* waiting (or immediately if no wait needed)
            self.last_request_time = now


async def crawl_for_pdfs(
    start_url: str,
    include_regex: Optional[str] = None,
    max_depth: int = 2,
    max_pdfs: int = 100,
    max_pages_crawl: int = 500,  # Limit total pages visited
    rate_limit_rps: float = 2.0,  # Requests per second
) -> List[str]:
    """
    Crawls a website starting from start_url to find PDF links using BFS.

    Args:
        start_url: The URL to begin crawling.
        include_regex: Optional regex to filter PDF URLs.
        max_depth: Maximum depth for BFS crawl.
        max_pdfs: Maximum number of PDF URLs to collect.
        max_pages_crawl: Safety limit on the total number of pages fetched.
        rate_limit_rps: Max requests per second for politeness.

    Returns:
        List of found PDF URLs matching the criteria.
    """
    try:
        inc_re = re.compile(include_regex, re.I) if include_regex else None
    except re.error as e:
        raise ToolInputError(f"Invalid include_regex: {e}") from e

    # Use sets for efficient duplicate checking
    seen_urls: Set[str] = set()
    pdf_urls_found: Set[str] = set()
    # Fix #7: Use deque for efficient BFS queue
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    seen_urls.add(start_url)

    visit_count = 0
    rate_limiter = RateLimiter(rate_limit_rps)
    base_netloc = urlparse(start_url).netloc

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SmartBrowserBot/1.0; +http://example.com/bot)"
    }  # Be a good bot citizen

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
        while queue and len(pdf_urls_found) < max_pdfs and visit_count < max_pages_crawl:
            current_url, current_depth = queue.popleft()
            visit_count += 1

            # Respect rate limits before fetching
            await rate_limiter.acquire()

            logger.debug(f"Crawling [Depth:{current_depth}, Visited:{visit_count}]: {current_url}")
            html = await _fetch_html(client, current_url)
            if not html:
                continue  # Skip pages that fail to fetch or are not HTML

            pdfs, pages = _extract_links(current_url, html)

            # Process found PDFs
            for pdf_url in pdfs:
                if pdf_url not in pdf_urls_found:
                    if inc_re is None or inc_re.search(pdf_url):
                        pdf_urls_found.add(pdf_url)
                        logger.info(f"PDF found: {pdf_url}")
                        if len(pdf_urls_found) >= max_pdfs:
                            break  # Stop processing PDFs if limit reached
            if len(pdf_urls_found) >= max_pdfs:
                break  # Stop crawl if PDF limit reached

            # Add new, valid pages to the queue if depth allows
            if current_depth < max_depth:
                for page_url in pages:
                    # Basic filtering: stay on same domain, avoid seen URLs
                    if urlparse(page_url).netloc == base_netloc and page_url not in seen_urls:
                        seen_urls.add(page_url)
                        queue.append((page_url, current_depth + 1))

    if visit_count >= max_pages_crawl:
        logger.warning(f"PDF crawl stopped: Maximum page visit limit ({max_pages_crawl}) reached.")
    if len(pdf_urls_found) >= max_pdfs:
        logger.info(f"PDF crawl stopped: Maximum PDF limit ({max_pdfs}) reached.")

    return list(pdf_urls_found)


# Fix #15: _download_file_direct using async I/O and thread pool for hashing
async def _download_file_direct(url: str, dest_dir: Path, seq: int = 1) -> Dict:
    """
    Downloads a file directly using httpx with streaming and async I/O.
    Computes hash asynchronously. Returns dict with status.
    """
    output_path = None
    try:
        parsed_url = urlparse(url)
        filename_base = os.path.basename(parsed_url.path) if parsed_url.path else ""

        # Generate a slug if filename is empty or generic
        if not filename_base or filename_base == "/" or "." not in filename_base:
            dir_slug = _get_dir_slug(url)
            filename = f"{seq:03d}_{dir_slug}_{_slugify(filename_base or 'download')}"
            # Try to guess extension from Content-Type or URL later if needed
            # For now, assume PDF if URL ends with it, else common binary '.dat'
            filename += ".pdf" if url.lower().endswith(".pdf") else ".dat"
        else:
            # Use suggested filename, potentially prefixing with sequence
            filename = f"{seq:03d}_{_slugify(filename_base)}"

        output_path = dest_dir / filename
        # Ensure parent directory exists (should be handled by caller, but double-check)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "*/*",  # Accept anything
            "Accept-Encoding": "gzip, deflate, br",  # Allow compression
        }

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=120.0, headers=headers
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    return {
                        "url": url,
                        "error": f"HTTP error {response.status_code}",
                        "status_code": response.status_code,
                        "success": False,
                    }

                # Try to refine filename extension from Content-Disposition or Content-Type
                content_disposition = response.headers.get("content-disposition")
                if content_disposition:
                    match = re.search(r'filename="?([^"]+)"?', content_disposition)
                    if match:
                        output_path = dest_dir / f"{seq:03d}_{_slugify(match.group(1))}"

                content_type = response.headers.get("content-type", "").split(";")[0].strip()
                if content_type == "application/pdf" and not output_path.name.lower().endswith(
                    ".pdf"
                ):
                    output_path = output_path.with_suffix(".pdf")
                # Add more content-type to extension mappings if needed

                # Stream download to file asynchronously
                hasher = hashlib.sha256()
                bytes_written = 0
                async with aiofiles.open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        await f.write(chunk)
                        hasher.update(chunk)  # Hash while writing
                        bytes_written += len(chunk)

        # Finalize hash
        file_hash = hasher.hexdigest()

        # Set permissions
        await _run_in_thread(os.chmod, output_path, 0o600)

        await _log(
            "download_direct_success",
            url=url,
            file=str(output_path),
            size=bytes_written,
            sha256=file_hash,
        )
        return {
            "url": url,
            "file": str(output_path),
            "size": bytes_written,
            "sha256": file_hash,
            "success": True,
        }

    except httpx.RequestError as e:
        logger.warning(f"Network error downloading {url}: {e}")
        return {"url": url, "error": f"Network error: {e}", "success": False}
    except Exception as e:
        logger.error(f"Error downloading file {url} directly: {e}", exc_info=True)
        # Clean up partially downloaded file if it exists
        if output_path and output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        return {"url": url, "error": f"Download failed: {e}", "success": False}


# --- OSS Documentation Crawler Helpers ---
_DOC_EXTS = (".html", ".htm", "/")  # Allowed extensions/endings for doc pages
_DOC_STOP_PAT = re.compile(
    r"\.(png|jpg|jpeg|gif|svg|css|js|zip|tgz|gz|whl|exe|dmg|ico|woff|woff2|map|json|xml|txt)$", re.I
)  # Extensions to ignore


def _looks_like_docs_url(url: str) -> bool:
    """Heuristic check if a URL looks like a documentation page."""
    try:
        url_low = url.lower()
        parsed = urlparse(url_low)

        # Ignore URLs with query parameters that often indicate actions/APIs
        if parsed.query:
            return False

        # Ignore common non-doc paths
        if any(
            frag in parsed.path
            for frag in [
                "/api/",
                "/blog/",
                "/news/",
                "/community/",
                "/forum/",
                "/download/",
                "/install/",
                "/_static/",
                "/_images/",
            ]
        ):
            return False

        # Require common doc indicators or allowed extensions
        has_doc_indicator = (
            "docs" in url_low
            or "guide" in url_low
            or "tuto" in url_low
            or "ref" in url_low
            or "api" in parsed.path
        )  # Checking path specifically for api
        has_allowed_ending = url_low.endswith(_DOC_EXTS)

        # Check against stop pattern
        is_stopped = bool(_DOC_STOP_PAT.search(parsed.path))

        return (has_doc_indicator or has_allowed_ending) and not is_stopped
    except Exception:
        return False  # Error parsing URL


async def _pick_docs_root(pkg_name: str) -> Optional[str]:
    """Use web search to find the likely documentation root URL for a package."""
    try:
        # Use a reliable search engine
        logger.info(f"Searching for documentation root for package: '{pkg_name}'")
        search_results = await search_web(
            f"{pkg_name} documentation", engine="duckduckgo", max_results=10
        )

        if not search_results:
            search_results = await search_web(
                f"{pkg_name} user guide", engine="bing", max_results=5
            )

        for hit in search_results:
            url = hit.get("url")
            if url and _looks_like_docs_url(url):
                # Often the root is one level up from the first hit
                parsed = urlparse(url)
                # Heuristic: If path has multiple segments, try going up one level
                path_segments = [seg for seg in parsed.path.split("/") if seg]
                if len(path_segments) > 1:
                    root_url = parsed._replace(
                        path="/".join(path_segments[:-1]) + "/", query="", fragment=""
                    ).geturl()
                    # Test if this potential root is accessible? Overkill for now.
                    logger.info(f"Potential docs root (from path): {root_url}")
                    # Simple check: does it look like docs itself?
                    if _looks_like_docs_url(root_url):
                        return root_url

                # Otherwise, return the first plausible hit
                logger.info(f"Potential docs root (first hit): {url}")
                return url

        # Fallback if no good hits found
        logger.warning(
            f"Could not reliably determine documentation root for '{pkg_name}' via search."
        )
        return search_results[0]["url"] if search_results else None

    except ToolError as e:
        logger.error(f"Search failed while finding docs root for '{pkg_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error finding docs root for '{pkg_name}': {e}", exc_info=True)
        return None


def _summarize_html(html: str, max_len: int = 10000) -> str:
    """
    Extract readable text content from HTML using libraries or fallback.
    Limits memory usage by processing chunks if necessary.
    """
    if not html:
        return ""
    # Limit initial HTML size to avoid excessive memory use by parsers
    MAX_HTML_SIZE = 3 * 1024 * 1024  # 3MB limit
    if len(html) > MAX_HTML_SIZE:
        logger.warning(
            f"HTML content truncated from {len(html)} to {MAX_HTML_SIZE} bytes for summarization."
        )
        html = html[:MAX_HTML_SIZE]

    text = ""
    # 1. Try Trafilatura (often good for articles/main content)
    try:
        import trafilatura  # type: ignore

        # Settings to focus on main content, ignore comments/tables
        extracted = trafilatura.extract(
            html, include_comments=False, include_tables=False, favour_precision=True
        )
        if extracted and len(extracted) > 100:  # Basic check for non-trivial extraction
            text = extracted
            # logger.debug("Summarized HTML using Trafilatura")
    except ImportError:
        logger.debug("Trafilatura not installed, skipping.")
    except Exception as e:
        logger.warning(f"Trafilatura extraction failed: {e}")

    # 2. Try Readability-lxml if Trafilatura failed or didn't yield much
    if not text or len(text) < 200:
        try:
            from readability import Document  # type: ignore

            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            # Extract text from the summarized HTML using BeautifulSoup for robustness
            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            # logger.debug("Summarized HTML using Readability")
        except ImportError:
            logger.debug("Readability-lxml not installed, skipping.")
        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")

    # 3. Fallback: Basic text extraction using BeautifulSoup if others fail
    if not text or len(text) < 100:
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Remove script, style, nav, header, footer elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                element.decompose()
            # Get remaining text
            text = soup.get_text(separator=" ", strip=True)
            # logger.debug("Summarized HTML using BeautifulSoup fallback")
        except Exception as e:
            logger.warning(f"BeautifulSoup fallback extraction failed: {e}")
            # As a last resort, use regex (very basic)
            text = re.sub(r"<[^>]+>", " ", html)

    # Final cleanup: normalize whitespace and limit length
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text[:max_len]


async def _grab_readable(
    client: httpx.AsyncClient, url: str, rate_limiter: RateLimiter
) -> Optional[str]:
    """Fetch HTML using client and rate limiter, then extract readable text."""
    await rate_limiter.acquire()
    html = await _fetch_html(client, url)
    if html:
        # Run summarization in thread pool as it can be CPU intensive
        summary = await _run_in_thread(_summarize_html, html)
        return summary
    return None


async def crawl_docs_site(
    root_url: str, max_pages: int = 40, rate_limit_rps: float = 3.0
) -> List[Tuple[str, str]]:
    """
    BFS crawl a documentation site within the same domain, collecting readable text.

    Args:
        root_url: The starting URL for the documentation site.
        max_pages: Maximum number of pages to successfully fetch and process.
        rate_limit_rps: Max requests per second for politeness.

    Returns:
        List of tuples, where each tuple is (url, readable_text).
    """
    try:
        start_netloc = urlparse(root_url).netloc
        if not start_netloc:
            raise ValueError("Invalid root URL, cannot determine domain.")
    except ValueError as e:
        raise ToolInputError(f"Invalid root URL for documentation crawl: {root_url} ({e})") from e

    seen_urls: Set[str] = set()
    # Fix #7: Use deque
    queue: deque[str] = deque([root_url])
    seen_urls.add(root_url)
    output_pages: List[Tuple[str, str]] = []

    visit_count = 0
    # Fix #7: Add safety limit beyond max_pages
    max_visits = max(max_pages * 5, 200)
    rate_limiter = RateLimiter(rate_limit_rps)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SmartBrowserDocBot/1.0; +http://example.com/bot)"
    }

    logger.info(f"Starting documentation crawl from: {root_url} (Max pages: {max_pages})")

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
        while queue and len(output_pages) < max_pages and visit_count < max_visits:
            current_url = queue.popleft()
            visit_count += 1

            logger.debug(
                f"Crawling Doc [Collected:{len(output_pages)}, Visited:{visit_count}]: {current_url}"
            )

            # Get readable text (includes rate limiting and HTML fetching)
            readable_text = await _grab_readable(client, current_url, rate_limiter)

            if readable_text:
                output_pages.append((current_url, readable_text))
                if len(output_pages) >= max_pages:
                    break  # Stop if we've collected enough pages

                # Extract links only if we got content and need more pages
                # Re-fetch HTML (or reuse if _grab_readable returned it?) - Re-fetch is simpler
                await rate_limiter.acquire()  # Rate limit again for link extraction fetch
                html_for_links = await _fetch_html(client, current_url)
                if html_for_links:
                    _, page_links = _extract_links(current_url, html_for_links)

                    for link_url in page_links:
                        try:
                            # Validate link: same domain, looks like docs, not seen
                            parsed_link = urlparse(link_url)
                            if (
                                parsed_link.netloc == start_netloc
                                and _looks_like_docs_url(link_url)
                                and link_url not in seen_urls
                            ):
                                seen_urls.add(link_url)
                                queue.append(link_url)
                        except ValueError:
                            logger.debug(
                                f"Skipping invalid link URL found on {current_url}: {link_url}"
                            )

    if visit_count >= max_visits:
        logger.warning(f"Doc crawl stopped: Maximum visit limit ({max_visits}) reached.")
    logger.info(f"Documentation crawl finished. Collected {len(output_pages)} pages.")
    return output_pages


# ══════════════════════════════════════════════════════════════════════════════
# 10. PAGE STATE EXTRACTION WITH HTML SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════════
# Fix #Half-baked (_ensure_helper): Simplify - don't rely on external JS unless absolutely necessary.
# The get_page_state below uses Playwright's evaluate which is generally reliable.


async def get_page_state(page: Page, max_elements: int = 150) -> dict[str, Any]:
    """
    Extracts the current state of the page, including URL, title, summary,
    and details about visible interactive elements.

    Args:
        page: The Playwright Page object.
        max_elements: Maximum number of interactive elements to report.

    Returns:
        Dictionary representing the page state.
    """
    if not page or page.is_closed():
        return {
            "error": "Page is closed or invalid",
            "url": "",
            "title": "",
            "elements": [],
            "text_summary": "",
        }

    try:
        start_time = time.monotonic()
        # Get basic info
        page_url = page.url
        page_title = await page.title()

        # Get HTML content for summary (might be large)
        # Consider limiting content size if performance is an issue
        try:
            html = await page.content(timeout=10000)  # 10s timeout for content
            # Summarize in thread pool
            summary = await _run_in_thread(_summarize_html, html, max_len=5000)
        except PlaywrightTimeoutError:
            summary = "[Error: Timed out getting page content]"
            logger.warning(f"Timed out getting content for page state: {page_url}")
        except PlaywrightException as e:
            summary = f"[Error: Could not get page content: {e}]"
            logger.warning(f"Playwright error getting content for page state: {page_url}: {e}")

        # Extract interactive elements using JavaScript evaluation
        try:
            elements = await page.evaluate(f"""() => {{
                const interactiveSelectors = [
                    'a[href]', 'button', 'input:not([type="hidden"])',
                    'textarea', 'select', '[role="button"]', '[role="link"]',
                    '[role="checkbox"]', '[role="radio"]', '[role="tab"]',
                    '[role="menuitem"]', '[role="option"]', '[onclick]'
                ];
                // Function to check if element is reasonably visible in viewport
                const isVisible = elem => {{
                    if (!elem) return false;
                    const style = window.getComputedStyle(elem);
                    if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) < 0.1) {{
                        return false;
                    }}
                    const rect = elem.getBoundingClientRect();
                    const windowHeight = window.innerHeight || document.documentElement.clientHeight;
                    const windowWidth = window.innerWidth || document.documentElement.clientWidth;
                    // Check if element has size and is within viewport bounds (allowing partial visibility)
                    const vertInView = (rect.top <= windowHeight) && ((rect.top + rect.height) >= 0);
                    const horzInView = (rect.left <= windowWidth) && ((rect.left + rect.width) >= 0);
                    return rect.width > 0 && rect.height > 0 && vertInView && horzInView;
                }};

                let collectedElements = [];
                let elementIdCounter = 0;
                const uniqueElements = new Set(); // Track elements to avoid duplicates

                document.querySelectorAll(interactiveSelectors.join(', ')).forEach(el => {{
                    if (collectedElements.length >= {max_elements}) return;
                    // Skip if not visible or already added
                    if (!isVisible(el) || uniqueElements.has(el)) return;

                    const getText = (elem) => {{
                        let text = (elem.innerText || elem.value || elem.getAttribute('aria-label') || elem.getAttribute('title') || '');
                        return text.trim().replace(/\s+/g, ' ').slice(0, 150); // Normalize whitespace and limit length
                    }};

                    const elementData = {{
                        // id: `sb_el_${{elementIdCounter++}}`, // Simple counter ID
                        tag: el.tagName.toLowerCase(),
                        type: el.type ? el.type.toLowerCase() : null, // Input type
                        role: el.getAttribute('role'),
                        text: getText(el),
                        name: el.name || null, // Input name attribute
                        href: el.href ? el.href.slice(0, 300) : null, // Limit href length
                        placeholder: el.placeholder || null,
                    }};

                    // Add element data if it has some identifying info (text, name, role, etc.)
                    if (elementData.text || elementData.role || elementData.name || elementData.placeholder || elementData.tag === 'a' || elementData.tag === 'button') {{
                        collectedElements.push(elementData);
                        uniqueElements.add(el); // Mark as added
                    }}
                }});
                return collectedElements;
            }}""")
        except PlaywrightException as e:
            logger.warning(
                f"Could not evaluate script to get interactive elements on {page_url}: {e}"
            )
            elements = [{"error": "Failed to extract elements"}]

        end_time = time.monotonic()
        await _log(
            "page_state_extracted",
            url=page_url,
            duration_ms=int((end_time - start_time) * 1000),
            num_elements=len(elements),
        )

        return {"url": page_url, "title": page_title, "elements": elements, "text_summary": summary}

    except PlaywrightException as e:
        logger.error(f"Error getting page state for {page.url}: {e}", exc_info=True)
        # Return minimal state on error
        return {
            "url": page.url or "unknown",
            "title": "[Error getting title]",
            "elements": [{"error": f"Failed to get page state: {e}"}],
            "text_summary": "[Error getting summary]",
        }
    except Exception as e:
        logger.error(f"Unexpected error getting page state for {page.url}: {e}", exc_info=True)
        return {
            "url": page.url or "unknown",
            "title": "[Error getting title]",
            "elements": [{"error": f"Unexpected error getting page state: {e}"}],
            "text_summary": "[Error getting summary]",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 11. LLM BRIDGE
# ══════════════════════════════════════════════════════════════════════════════


def _extract_json_block(text: str) -> Optional[str]:
    """Tries to extract the first valid JSON object or array block from text."""
    # Regex to find JSON blocks: look for {..} or [..] possibly nested
    # This regex is simple and might fail on complex cases (e.g., strings containing brackets)
    # A more robust parser might be needed for complex LLM outputs.
    matches = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if matches:
        block = matches.group(0)
        # Basic validation: check if it starts/ends correctly
        if (block.startswith("{") and block.endswith("}")) or (
            block.startswith("[") and block.endswith("]")
        ):
            return block
    return None


async def _call_llm(
    messages: Sequence[Dict[str, str]],
    model: str = "gpt-4o",  # Consider making model configurable
    expect_json: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Union[Dict[str, Any], List[Any]]:  # Return type can be dict or list for JSON
    """
    Calls the LLM using the MCP server's completion tool.
    Handles message formatting and JSON parsing/validation if requested.

    Args:
        messages: List of message dictionaries (e.g., [{"role": "system", "content": "..."}, ...]).
        model: The LLM model to use.
        expect_json: If True, attempts to parse the response as JSON.
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens for the LLM response.

    Returns:
        - If expect_json is True: The parsed JSON object/list, or a dict with an "error" key.
        - If expect_json is False: A dict {"text": response_text} or {"error": ...}.
    """
    if not messages:
        return {"error": "No messages provided to LLM."}

    # Fix #12: Use the full message list. Assume generate_completion handles it.
    # If generate_completion only takes a string prompt, we need to format messages here.
    # Assuming generate_completion supports the `messages` structure:
    llm_args = {
        "provider": Provider.OPENAI.value,  # Assuming OpenAI, make configurable if needed
        "model": model,
        "messages": messages,  # Pass the list directly
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop_sequences": None,  # Add stop sequences if needed
    }

    # Add specific JSON instruction if needed
    if expect_json:
        # Append or modify the last user message to include JSON instructions
        last_message = messages[-1]
        if last_message["role"] == "user":
            json_instruction = (
                "\n\nIMPORTANT: Your response MUST be valid JSON. "
                "Output ONLY the JSON object or array, starting with `{` or `[` and ending with `}` or `]`. "
                "Do not include any introductory text, explanations, or markdown formatting like ```json."
            )
            # Create a new list to avoid modifying the original
            modified_messages = list(messages[:-1])
            modified_messages.append(
                {"role": "user", "content": last_message["content"] + json_instruction}
            )
            llm_args["messages"] = modified_messages
        else:
            # Cannot easily add instruction if last message isn't user. Log warning.
            logger.warning("Cannot add JSON instruction as last LLM message is not 'user'.")

    try:
        start_time = time.monotonic()
        resp = await generate_completion(**llm_args)
        duration = time.monotonic() - start_time
        await _log(
            "llm_call_complete",
            model=model,
            duration_ms=int(duration * 1000),
            success=resp.get("success", False),
        )

        if not resp.get("success"):
            error_msg = resp.get("error", "LLM call failed with no specific error message.")
            logger.error(f"LLM call failed: {error_msg}")
            return {"error": f"LLM API Error: {error_msg}"}

        raw_text = resp.get("text", "").strip()
        if not raw_text:
            return {"error": "LLM returned an empty response."}

        if not expect_json:
            return {"text": raw_text}

        # Fix #Quick-win #7: Strict JSON parsing
        # 1. Try parsing the whole response directly
        try:
            parsed_json = json.loads(raw_text)
            # Optional: Validate structure if needed (e.g., ensure it's a list of objects)
            return parsed_json
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed. Trying to extract JSON block.")
            # 2. Try extracting the first {...} or [...] block
            json_block = _extract_json_block(raw_text)
            if json_block:
                try:
                    parsed_json = json.loads(json_block)
                    logger.warning(
                        "LLM response contained non-JSON content. Extracted JSON block successfully."
                    )
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse extracted JSON block: {e}. Raw block: {json_block[:500]}..."
                    )
                    return {
                        "error": f"Could not parse JSON from LLM response (invalid JSON block extracted). Error: {e}",
                        "raw_response": raw_text[:1000],
                    }
            else:
                # 3. If no block found or parsing fails, return error
                logger.error(
                    f"Could not find or parse JSON in LLM response. Raw response: {raw_text[:1000]}..."
                )
                return {
                    "error": "Could not parse JSON from LLM response (no valid JSON block found).",
                    "raw_response": raw_text[:1000],
                }

    except ProviderError as e:
        logger.error(f"LLM Provider error: {e}", exc_info=True)
        return {"error": f"LLM Provider Error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
        return {"error": f"LLM call failed unexpectedly: {e}"}


async def ask_llm(page: Page, user_instruction: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Gets page state and asks LLM for the next action based on user instruction."""
    try:
        state = await get_page_state(page)
        if "error" in state:  # Handle case where page state extraction failed
            return {"error": f"Could not get page state: {state['error']}"}

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent web automation assistant. Based on the current page state "
                    "(URL, title, summary, interactive elements) and the user's task, determine the single "
                    "next action to perform. Respond ONLY with a valid JSON object describing the action. "
                    f"Allowed actions are: {', '.join(sorted(ALLOWED_ACTIONS))}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CURRENT PAGE STATE:\n"
                    f"URL: {state.get('url')}\n"
                    f"Title: {state.get('title')}\n"
                    f"Summary: {state.get('text_summary', 'N/A')}\n"
                    f"Interactive Elements:\n{json.dumps(state.get('elements', []), indent=2)}\n\n"
                    f"USER TASK:\n{user_instruction}\n\n"
                    "Respond ONLY with the JSON for the next single action (e.g., "
                    '{"action": "click", "target": {"name": "Login Button"}} or '
                    '{"action": "type", "target": {"role": "textbox", "name": "username"}, "text": "user@example.com"} or '
                    '{"action": "finish"}).'
                ),
            },
        ]
        # Expect a single JSON object representing the action
        return await _call_llm(
            messages, model=model, expect_json=True, temperature=0.0
        )  # Low temp for deterministic action
    except Exception as e:
        logger.error(f"Error in ask_llm: {e}", exc_info=True)
        return {"error": f"Failed to get LLM action suggestion: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# 12. NATURAL-LANGUAGE MACRO RUNNER
# ══════════════════════════════════════════════════════════════════════════════
ALLOWED_ACTIONS = {"click", "type", "wait", "download", "extract", "finish", "scroll"}


async def _plan(
    page_state: Dict[str, Any], task: str, model: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """Generates a sequence of browser actions based on page state and task."""

    action_details = """
    - "click": requires "target" (dict for SmartLocator: name, role, css, xpath).
    - "type": requires "target" (dict for SmartLocator) and "text" (string). Optional: "enter": bool (default false), "clear_before": bool (default true).
    - "wait": requires "ms" (integer milliseconds to wait).
    - "download": requires "target" (dict for SmartLocator to click). Optional: "dest" (string destination directory).
    - "extract": requires "selector" (CSS selector string). Returns text content of matching elements.
    - "scroll": requires "direction" ("up", "down", "bottom", "top"). Optional: "amount_px": int (for up/down).
    - "finish": takes no arguments, indicates the task is complete.
    """
    # Fix #Half-baked (download action params): Added 'dest' to description.

    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous web automation planner. Based on the user's task and the current page state "
                "(URL, title, summary, interactive elements), create a plan as a JSON list of action steps. "
                f"Use ONLY the allowed actions: {sorted(ALLOWED_ACTIONS)}. Follow the required arguments for each action.\n"
                f"ACTION DETAILS:\n{action_details}\n"
                "Your response MUST be ONLY the JSON list of steps, starting with `[` and ending with `]`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CURRENT PAGE STATE:\n"
                f"URL: {page_state.get('url')}\n"
                f"Title: {page_state.get('title')}\n"
                f"Summary: {page_state.get('text_summary', 'N/A')}\n"
                f"Interactive Elements:\n{json.dumps(page_state.get('elements', []), indent=2)}\n\n"
                f"USER TASK:\n{task}\n\n"
                "Generate the JSON list of steps to accomplish the task."
            ),
        },
    ]

    result = await _call_llm(messages, model=model, expect_json=True, temperature=0.0)

    if isinstance(result, dict) and "error" in result:
        raise ToolError(
            f"Planner LLM failed: {result['error']}", details=result.get("raw_response")
        )
    if not isinstance(result, list):
        raise ToolError(
            f"Planner LLM returned unexpected format (expected list): {type(result)}",
            details=str(result)[:500],
        )

    # Validate plan structure and actions
    validated_plan = []
    for i, step in enumerate(result):
        if not isinstance(step, dict) or "action" not in step:
            logger.warning(f"Invalid step format in plan (step {i + 1}): {step}")
            continue  # Skip invalid step format
        action = step.get("action")
        if action not in ALLOWED_ACTIONS:
            logger.warning(f"Invalid action '{action}' in plan (step {i + 1}): {step}")
            continue  # Skip invalid action
        validated_plan.append(step)

    if not validated_plan:
        raise ToolError("Planner LLM returned an empty or invalid plan.")

    return validated_plan


async def run_macro(
    page: Page, task: str, max_rounds: int = 7, model: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """
    Executes a multi-step task described in natural language using a plan-act loop.

    Args:
        page: The Playwright Page object.
        task: Natural language description of the task.
        max_rounds: Maximum number of plan-execute rounds.
        model: LLM model for planning.

    Returns:
        List of dictionaries, each representing the result of an executed step.
    """
    all_step_results = []
    current_task = task

    for i in range(max_rounds):
        round_num = i + 1
        logger.info(
            f"Macro Execution Round {round_num}/{max_rounds} | Task: {current_task[:100]}..."
        )
        try:
            state = await get_page_state(page)
            if "error" in state:
                raise ToolError(f"Failed to get page state for planning: {state['error']}")

            plan = await _plan(state, current_task, model)
            await _log("macro_plan_generated", round=round_num, task=current_task, steps=plan)

            if not plan:
                logger.warning(
                    f"Macro round {round_num}: Planner returned empty plan. Assuming finish."
                )
                await _log("macro_plan_empty", round=round_num, task=current_task)
                break  # Exit loop if plan is empty

            # Execute the planned steps
            step_results = await run_steps(page, plan)
            all_step_results.extend(step_results)

            # Check if 'finish' action was executed or if last step failed critically
            finished = any(s.get("action") == "finish" and s.get("success") for s in step_results)
            last_step_failed = (
                step_results
                and not step_results[-1].get("success")
                and step_results[-1].get("action") != "wait"
            )

            if finished:
                await _log("macro_finish_action", round=round_num)
                logger.info(
                    f"Macro finished successfully in round {round_num} via 'finish' action."
                )
                return all_step_results

            if last_step_failed:
                error_info = step_results[-1].get("error", "Unknown error")
                failed_action = step_results[-1].get("action", "unknown")
                await _log(
                    "macro_fail_step", round=round_num, action=failed_action, error=error_info
                )
                logger.warning(
                    f"Macro stopped in round {round_num} due to failed step: {failed_action} - {error_info}"
                )
                # Consider adding reflection/re-planning here based on failure
                # For now, stop execution on failure
                return all_step_results

            # Optional: Add reflection step here - analyze results and update task if needed
            # current_task = await _reflect(task, state, plan, step_results)

        except ToolError as e:
            await _log("macro_error_tool", round=round_num, task=current_task, error=str(e))
            logger.error(f"Macro round {round_num} failed with ToolError: {e}")
            all_step_results.append(
                {"action": "error", "success": False, "error": f"ToolError: {e}"}
            )
            return all_step_results  # Stop on tool errors
        except Exception as e:
            await _log("macro_error_unexpected", round=round_num, task=current_task, error=str(e))
            logger.error(
                f"Macro round {round_num} failed with unexpected error: {e}", exc_info=True
            )
            all_step_results.append(
                {"action": "error", "success": False, "error": f"Unexpected Error: {e}"}
            )
            return all_step_results  # Stop on unexpected errors

    # If loop completes without finishing
    await _log("macro_exceeded_rounds", max_rounds=max_rounds, task=task)
    logger.warning(
        f"Macro exceeded max rounds ({max_rounds}) without 'finish' action for task: {task}"
    )
    return all_step_results


# --- Autopilot Planner ---
# Note: Tool references MUST match the method names in SmartBrowserTool
_AVAILABLE_TOOLS = {
    # tool_name_for_llm : (SmartBrowserTool_method_name, brief_arg_schema_for_llm)
    "search_web": (
        "search",
        {"query": "str", "engine": "str (default yandex)", "max_results": "int (default 10)"},
    ),
    "browse_page": (
        "browse_url",
        {
            "url": "str",
            "wait_for_selector": "Optional[str]",
            "wait_for_navigation": "bool (default True)",
        },
    ),
    "click_element": (
        "click_and_extract",
        {
            "url": "str",
            "target": "dict (SmartLocator: name, role, css, xpath)",
            "wait_ms": "int (default 1000)",
        },
    ),
    "fill_form": (
        "fill_form",
        {
            "url": "str",
            "form_fields": "[{'target': dict, 'text': str, 'enter': bool?}]",
            "submit_button": "Optional[dict]",
        },
    ),
    "download_file_via_click": (
        "download_file",
        {"url": "str", "target": "dict (SmartLocator)", "dest_dir": "Optional[str]"},
    ),
    "run_page_macro": (
        "execute_macro",
        {"url": "str", "task": "str (natural language)", "max_rounds": "int (default 5)"},
    ),
    "download_all_pdfs_from_site": (
        "download_site_pdfs",
        {
            "start_url": "str",
            "dest_subfolder": "str (default site_pdfs)",
            "include_regex": "Optional[str]",
            "max_depth": "int (default 2)",
            "max_pdfs": "int (default 100)",
        },
    ),
    "collect_project_documentation": (
        "collect_documentation",
        {"package": "str (name of library/project)", "max_pages": "int (default 40)"},
    ),
    "process_urls_in_parallel": (
        "parallel_process",
        {"urls": "[str]", "max_tabs": "Optional[int]"},
    ),
}

_PLANNER_SYS = (
    "You are an AI orchestrator. Your goal is to create a plan to fulfill the user's TASK "
    "by selecting appropriate tools from the available list and specifying their arguments. "
    "The plan should be a JSON list of steps, where each step calls one tool.\n"
    "Analyze the TASK and any PRIOR RESULTS to decide the sequence of tool calls.\n"
    "Your response MUST be ONLY the JSON list of tool call objects, starting with `[` and ending with `]`.\n"
    "Do not include explanations or markdown."
)


async def _plan_autopilot(
    task: str, prior_results: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """Generates a multi-step plan using available tools for the autopilot feature."""
    tools_desc = {name: schema for name, (_, schema) in _AVAILABLE_TOOLS.items()}

    prior_results_summary = "None"
    if prior_results:
        # Summarize results to avoid huge prompts
        summaries = []
        for i, res in enumerate(
            prior_results[-3:], start=max(0, len(prior_results) - 3) + 1
        ):  # Show last 3 results
            tool = res.get("tool", "unknown")
            success = res.get("success", False)
            outcome = (
                "[Success]" if success else f"[Failed: {res.get('error', 'Unknown error')[:100]}]"
            )
            # Include key result snippets if available
            result_data = res.get("result", {})
            snippet = ""
            if isinstance(result_data, dict):
                if (
                    "results" in result_data
                    and isinstance(result_data["results"], list)
                    and result_data["results"]
                ):
                    snippet = f" (Found {len(result_data['results'])} items, e.g., {str(result_data['results'][0])[:100]}...)"
                elif "download" in result_data:
                    snippet = f" (Downloaded: {result_data['download'].get('file_name', 'N/A')})"
                elif "page_state" in result_data:
                    snippet = f" (Page: {result_data['page_state'].get('title', 'N/A')[:50]}...)"
                elif "file" in result_data:
                    snippet = f" (File: {Path(result_data['file']).name})"

            summaries.append(f"Step {i}: Tool={tool} -> {outcome}{snippet}")
        prior_results_summary = "\n".join(summaries)

    user_prompt = (
        f"AVAILABLE TOOLS:\n{json.dumps(tools_desc, indent=2)}\n\n"
        f"PRIOR RESULTS SUMMARY:\n{prior_results_summary}\n\n"
        f"USER TASK:\n{task}\n\n"
        "Generate the JSON plan (list of steps) to complete the task. "
        'Each step should be: {"tool": "<tool_name>", "args": {<arguments_dict>}}. '
        "Respond ONLY with the JSON list."
    )

    messages = [
        {"role": "system", "content": _PLANNER_SYS},
        {"role": "user", "content": user_prompt},
    ]

    response = await _call_llm(
        messages, expect_json=True, temperature=0.0, max_tokens=2048
    )  # Allow longer plans

    if isinstance(response, dict) and "error" in response:
        raise ToolError(
            f"Autopilot planner failed: {response['error']}", details=response.get("raw_response")
        )
    if not isinstance(response, list):
        raise ToolError(
            f"Autopilot planner returned unexpected format (expected list): {type(response)}",
            details=str(response)[:500],
        )

    # Basic validation of the plan structure
    validated_plan = []
    for i, step in enumerate(response):
        if (
            not isinstance(step, dict)
            or "tool" not in step
            or "args" not in step
            or not isinstance(step["args"], dict)
        ):
            logger.warning(f"Invalid step format in Autopilot plan (step {i + 1}): {step}")
            continue
        tool_name = step.get("tool")
        if tool_name not in _AVAILABLE_TOOLS:
            logger.warning(f"Unknown tool '{tool_name}' in Autopilot plan (step {i + 1}): {step}")
            continue
        validated_plan.append(step)

    if not validated_plan:
        raise ToolError("Autopilot planner returned an empty or invalid plan.")

    return validated_plan


# ══════════════════════════════════════════════════════════════════════════════
# 13. STEP RUNNER (for Macro execution)
# ══════════════════════════════════════════════════════════════════════════════
async def run_steps(page: Page, steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Executes a sequence of planned browser actions on a page."""
    results = []
    for step in steps:
        action = step.get("action")
        # Copy step data to store results, ensuring args are preserved
        step_result = step.copy()
        step_result["success"] = False  # Default to failure

        if not action:
            step_result["error"] = "Step missing 'action' field."
            results.append(step_result)
            continue  # Skip invalid step

        start_time = time.monotonic()
        try:
            if action == "click":
                target = step.get("target")
                if not target or not isinstance(target, dict):
                    raise ToolInputError("Missing 'target' dict for click action")
                await smart_click(page, **target)
                step_result["success"] = True

            elif action == "type":
                target = step.get("target")
                text = step.get("text")
                if not target or not isinstance(target, dict):
                    raise ToolInputError("Missing 'target' dict for type action")
                if text is None:
                    raise ToolInputError("Missing 'text' for type action")
                await smart_type(
                    page,
                    text,
                    press_enter=step.get("enter", False),
                    clear_before=step.get("clear_before", True),
                    **target,
                )
                step_result["success"] = True

            elif action == "wait":
                ms = step.get("ms")
                if ms is None:
                    raise ToolInputError("Missing 'ms' for wait action")
                await page.wait_for_timeout(int(ms))
                step_result["success"] = True  # Wait itself doesn't fail easily

            elif action == "download":
                target = step.get("target")
                if not target or not isinstance(target, dict):
                    raise ToolInputError("Missing 'target' dict for download action")
                # Use smart_download, result is already a dict
                download_outcome = await smart_download(page, target, step.get("dest"))
                step_result["result"] = download_outcome  # Store download info
                step_result["success"] = download_outcome.get(
                    "success", False
                )  # Reflect download success

            elif action == "extract":
                selector = step.get("selector")
                if not selector:
                    raise ToolInputError("Missing 'selector' for extract action")
                # Use page.eval_on_selector_all for robustness
                extracted_texts = await page.eval_on_selector_all(
                    selector, "(elements => elements.map(el => el.innerText || el.textContent))"
                )
                step_result["result"] = [t.strip() for t in extracted_texts if t]
                step_result["success"] = (
                    True  # Extraction itself succeeds even if no elements found
                )

            elif action == "scroll":
                direction = step.get("direction")
                amount_px = step.get("amount_px")
                if direction not in ["up", "down", "top", "bottom"]:
                    raise ToolInputError("Invalid scroll direction")
                if direction == "top":
                    await page.evaluate("() => window.scrollTo(0, 0)")
                elif direction == "bottom":
                    await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                elif direction == "up":
                    px = int(amount_px) if amount_px is not None else 500  # Default scroll amount
                    await page.evaluate("(px) => window.scrollBy(0, -px)", px)
                elif direction == "down":
                    px = int(amount_px) if amount_px is not None else 500  # Default scroll amount
                    await page.evaluate("(px) => window.scrollBy(0, px)", px)
                step_result["success"] = True

            elif action == "finish":
                logger.info("Macro execution: 'finish' action encountered.")
                step_result["success"] = True
                # No break here, let run_macro handle finish signal

            else:
                raise ValueError(f"Unknown action '{action}' in macro step.")

            step_result["duration_ms"] = int((time.monotonic() - start_time) * 1000)

        # Fix #6: Catch ElementNotFoundError specifically
        except ElementNotFoundError as e:
            step_result["error"] = f"Element not found for action '{action}': {e}"
            logger.warning(f"Step failed: {step_result['error']}")
            # Do not break here, let the macro decide based on failure

        # Catch other ToolErrors (e.g., Input, Secret resolution)
        except ToolError as e:
            step_result["error"] = f"ToolError during action '{action}': {e}"
            logger.warning(f"Step failed: {step_result['error']}")
            # Do not break here

        # Catch Playwright errors (e.g., click intercepted, navigation timeout)
        except PlaywrightException as e:
            step_result["error"] = f"Playwright error during action '{action}': {e}"
            logger.warning(f"Step failed: {step_result['error']}")
            # Do not break here

        # Catch unexpected errors
        except Exception as e:
            step_result["error"] = f"Unexpected error during action '{action}': {e}"
            logger.error(f"Step failed unexpectedly: {step_result['error']}", exc_info=True)
            # Maybe break on unexpected errors? For now, record and continue if possible.

        finally:
            # Always log the step outcome
            await _log("macro_step_result", **step_result)
            results.append(step_result)
            # If finish action was successful, add result and return immediately
            # Let the caller (run_macro) handle the finish state.
            # if action == "finish" and step_result["success"]:
            #     break

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 14. UNIVERSAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
@resilient(max_attempts=2, backoff=1.0)  # Retry search once on transient errors
async def search_web(
    query: str, engine: str = "bing", max_results: int = 10
) -> List[Dict[str, str]]:
    """
    Performs web search using Playwright against Bing, DuckDuckGo (HTML), or Yandex.

    Args:
        query: The search query string.
        engine: 'bing', 'duckduckgo', or 'yandex'.
        max_results: Maximum number of results desired.

    Returns:
        List of result dictionaries, each with 'url', 'title', 'snippet'.
    """
    engine = engine.lower()
    if engine not in ("bing", "duckduckgo", "yandex"):
        raise ToolInputError(
            f"Invalid search engine: {engine}. Use 'bing', 'duckduckgo', or 'yandex'."
        )

    # Simple query sanitization
    safe_query = re.sub(r"[^\w\s\-\.]", "", query).strip()  # Remove potentially harmful chars
    if not safe_query:
        raise ToolInputError("Search query is empty or invalid.")
    qs = urllib.parse.quote_plus(safe_query)

    # Use timestamp/random element to slightly vary URL
    nonce = random.randint(1000, 9999)

    urls = {
        "bing": f"https://www.bing.com/search?q={qs}&count={max_results}&form=QBLH&rdr=1&r={nonce}",  # Added form/rdr/r params
        "duckduckgo": f"https://html.duckduckgo.com/html/?q={qs}&r={nonce}",  # HTML version is simpler/more stable
        "yandex": f"https://yandex.com/search/?text={qs}&lr=10000&r={nonce}",  # Added location region (generic)
    }
    search_url = urls[engine]

    # Selectors adjusted for robustness
    selectors = {
        "bing": {  # Bing structure changes often
            "result_item": "li.b_algo",
            "link": "h2 > a",  # Primary link within title
            "title": "h2 > a",  # Title text usually within link
            "snippet": ".b_caption p",  # Snippet within caption block
            "backup_link": "a.tilk",  # Alternative link if h2>a fails
        },
        "duckduckgo": {  # DDG HTML version
            "result_item": "div.result",
            "link": "a.result__a",
            "title": "a.result__a",
            "snippet": "a.result__snippet",
        },
        "yandex": {
            "result_item": "li.serp-item[data-cid]",  # Items with content ID
            "link": "a.Link.Link_theme_outer.Path-Item",  # More specific link selector
            "title": "a.Link.Link_theme_outer.Path-Item",  # Title is often the link text
            # Snippet requires careful selection, might be complex
            "snippet": "div.OrganicTextContentSpan",  # Or explore other text containers
        },
    }
    sel = selectors[engine]

    # Unique user agents per engine
    user_agents = {
        "bing": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.51",  # Edge UA
        "duckduckgo": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",  # Safari UA
        "yandex": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",  # Chrome Linux UA
    }
    ua = user_agents[engine]

    # Fix #13: Pass user agent to the context
    # Use a dedicated incognito context for each search for isolation
    context_args = {"user_agent": ua, "locale": "en-US"}
    ctx, _ = await get_browser_context(use_incognito=True, context_args=context_args)
    page = None
    try:
        page = await ctx.new_page()
        # No need for set_extra_http_headers if UA is set on context

        await _log("search_start", engine=engine, query=query, url=search_url)
        await page.goto(
            search_url, wait_until="domcontentloaded", timeout=30000
        )  # Wait for DOM, not full network idle

        # Wait for results to appear (use a selector common to all engines?)
        # Wait for the main result item selector
        try:
            await page.wait_for_selector(sel["result_item"], state="visible", timeout=10000)
        except PlaywrightTimeoutError as e:
            logger.warning(
                f"Timed out waiting for results selector '{sel['result_item']}' on {engine} for query: {query}"
            )
            # Check for CAPTCHA before giving up
            captcha_found = await page.evaluate("""
                () => document.body.innerText.toLowerCase().includes('captcha') ||
                      document.querySelector('img[src*="captcha"]') !== null ||
                      document.querySelector('iframe[title*="captcha"]') !== null
            """)
            if captcha_found:
                await _log("search_captcha", engine=engine, query=query)
                raise ToolError(
                    f"Search failed: {engine} presented a CAPTCHA.", error_code="captcha_detected"
                ) from e
            # Otherwise, assume no results or page load issue
            await _log(
                "search_no_results_selector",
                engine=engine,
                query=query,
                selector=sel["result_item"],
            )
            return []  # Return empty list if results selector doesn't appear

        # Add a small random pause after load
        await asyncio.sleep(random.uniform(0.5, 1.5))

        # Try to dismiss cookie/consent banners aggressively
        consent_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Agree")',
            'button:has-text("Allow")',
            'button:has-text("Consent")',
            'button[id*="consent"]',
            'button[class*="consent"]',
            'button:has-text("I understand")',
            'button:has-text("Got it")',
        ]
        for btn_sel in consent_selectors:
            try:
                # Use wait_for with short timeout to avoid hanging
                button = page.locator(btn_sel).first
                await button.wait_for(state="visible", timeout=500)
                await button.click(delay=random.uniform(50, 100), timeout=1000)
                logger.info(f"Dismissed potential consent banner with selector: {btn_sel}")
                await asyncio.sleep(0.5)  # Wait briefly after click
                break  # Assume one banner is enough
            except (PlaywrightTimeoutError, PlaywrightException):
                pass  # Ignore if selector not found or click fails

        # Extract results using page.evaluate for efficiency
        results = await page.evaluate(
            f"""
            (sel, max_results) => {{
                const items = Array.from(document.querySelectorAll(sel.result_item));
                const results = [];
                for (let i = 0; i < Math.min(items.length, max_results); i++) {{
                    const item = items[i];
                    try {{
                        let linkEl = item.querySelector(sel.link);
                        // Fallback link selector for Bing
                        if (!linkEl && sel.backup_link) {{
                            linkEl = item.querySelector(sel.backup_link);
                        }}

                        let titleEl = item.querySelector(sel.title);
                        let snippetEl = item.querySelector(sel.snippet);

                        const url = linkEl ? linkEl.href : null;
                        let title = titleEl ? titleEl.innerText : (linkEl ? linkEl.innerText : '');
                        let snippet = snippetEl ? snippetEl.innerText : '';

                        // Clean data
                        if (url && url.trim() && !url.startsWith('javascript:')) {{
                             // Simple title cleaning
                             title = title.replace(/[\n\r\t]+/g, ' ').replace(/\s{(2,)}/g, ' ').trim();
                             // Simple snippet cleaning
                             snippet = snippet.replace(/[\n\r\t]+/g, ' ').replace(/\s{(2,)}/g, ' ').trim();

                             // Ensure some useful info exists
                             if (title || snippet) {{
                                 results.push({{
                                     url: url.trim(),
                                     title: title,
                                     snippet: snippet
                                 }});
                             }}
                        }}
                    }} catch (e) {{
                        // Ignore errors processing a single item
                        console.warn("Error processing search result item:", e);
                    }}
                }}
                return results;
            }}
        """,
            selectors[engine],
            max_results,
        )  # Pass selectors and max_results

        await _log("search_complete", engine=engine, query=query, num_results=len(results))
        return results

    except PlaywrightException as e:
        await _log("search_error_playwright", engine=engine, query=query, error=str(e))
        # Let resilient handle retry if applicable
        raise ToolError(f"Playwright error during search: {e}") from e
    except Exception as e:
        await _log("search_error_unexpected", engine=engine, query=query, error=str(e))
        raise ToolError(f"Unexpected error during search: {e}") from e
    finally:
        if page:
            await page.close()
        if ctx:
            await ctx.close()


# ══════════════════════════════════════════════════════════════════════════════
# 15.  TOOL CLASS FOR MCP SERVER INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
class SmartBrowserTool(BaseTool):
    """
    Advanced web automation tool using Playwright.

    Integrates browser control, smart element finding, LLM-based planning,
    search, download, and parallel processing capabilities within the MCP server framework.
    Includes features like audit logging, encrypted state, proxy rotation, and resilience.
    """

    tool_name = "smart_browser"
    description = "Performs advanced web automation tasks like browsing, interaction, search, download, and running complex macros."

    def __init__(self, mcp_server):
        """Initialize the tool, setting up async components and inactivity monitor."""
        super().__init__(mcp_server)
        self.tab_pool = TabPool()  # Use the shared instance for now
        self._last_activity = time.monotonic()  # Use monotonic clock
        self._inactivity_monitor_task: Optional[asyncio.Task] = None
        self._selector_cleanup_task_handle: Optional[asyncio.Task] = None
        self._init_lock = asyncio.Lock()
        self._is_initialized = False

        # Fix #Half-baked (MCP Lifespan Check): Check nested attribute correctly
        is_server_context = False
        if hasattr(mcp_server, "mcp") and mcp_server.mcp:
            is_server_context = hasattr(mcp_server.mcp, "lifespan")
        elif hasattr(mcp_server, "lifespan"):  # Check top-level attribute as fallback
            is_server_context = True

        # Defer async initialization until first use or explicit setup
        # This avoids issues if initialized without a running event loop.
        if is_server_context:
            logger.info(
                "SmartBrowserTool detected server context. Async components will init via lifespan or first use."
            )
            # Assume server will call an `async_setup` method if needed during lifespan.
        else:
            logger.info(
                "SmartBrowserTool initialized outside server context. Async components will init on first use."
            )
            # If not in server context, we might want to init immediately if a loop exists
            # self._schedule_async_init() # Optional: attempt immediate init if loop available

    async def _ensure_initialized(self):
        """Ensure async components like browser and monitor are started."""
        global _selector_cleanup_task_handle # Use global handle defined earlier
        if self._is_initialized:
            return
        async with self._init_lock:
            if self._is_initialized:  # Double-check after acquiring lock
                return

            logger.info(
                "Performing first-time initialization of SmartBrowserTool async components..."
            )
            # Ensure browser context exists (this handles Playwright/browser launch)
            await get_browser_context()

            # Start inactivity monitor if not already running
            if self._inactivity_monitor_task is None or self._inactivity_monitor_task.done():
                logger.info("Starting browser inactivity monitor...")
                timeout_str = os.getenv("SB_INACTIVITY_TIMEOUT", "600")  # 10 minutes default
                try:
                    timeout_sec = int(timeout_str)
                    if timeout_sec > 0:
                        self._inactivity_monitor_task = asyncio.create_task(
                            self._inactivity_monitor(timeout_sec)
                        )
                    else:
                        logger.info("Inactivity monitor disabled (timeout <= 0).")
                except ValueError:
                    logger.warning(
                        f"Invalid SB_INACTIVITY_TIMEOUT value '{timeout_str}'. Monitor disabled."
                    )

            # Start selector cleanup task if not already running
            if _selector_cleanup_task_handle is None or _selector_cleanup_task_handle.done(): # <<< MODIFIED Check global
                logger.info("Starting periodic selector cleanup task...")
                _selector_cleanup_task_handle = asyncio.create_task(_selector_cleanup_task()) # <<< MODIFIED Assign to glo
                
            self._is_initialized = True
            logger.info("SmartBrowserTool async components initialized successfully.")

    def _update_activity(self):
        """Updates the last activity timestamp."""
        self._last_activity = time.monotonic()

    async def _inactivity_monitor(self, timeout_seconds: int):
        """Monitors browser inactivity and triggers shutdown if idle for too long."""
        check_interval = 60  # Check every minute
        logger.info(
            f"Inactivity monitor started. Timeout: {timeout_seconds}s. Check interval: {check_interval}s."
        )

        while True:
            await asyncio.sleep(check_interval)
            # Check if browser/context still exist before calculating idle time
            # Use the lock to safely access global state
            async with _playwright_lock:
                browser_exists = _browser is not None and _browser.is_connected()
                context_exists = _ctx is not None and _ctx.browser is not None

            if not browser_exists and not context_exists:
                logger.info(
                    "Inactivity monitor: Browser/Context no longer active. Stopping monitor."
                )
                break

            idle_time = time.monotonic() - self._last_activity
            # logger.debug(f"Inactivity check: Idle for {idle_time:.1f}s")

            if idle_time > timeout_seconds:
                logger.info(
                    f"Browser inactive for {idle_time:.1f}s (threshold: {timeout_seconds}s). Initiating shutdown."
                )
                # Use the safe shutdown initiator
                try:
                    await _initiate_shutdown()
                except Exception as e:
                    logger.error(
                        f"Error during automatic shutdown triggered by inactivity: {e}",
                        exc_info=True,
                    )
            
            # Exit the monitor loop once shutdown is triggered
            break
        logger.info("Inactivity monitor stopped.")

    @tool(name="smart_browser.browse")
    @with_tool_metrics
    @with_error_handling
    async def browse_url(
        self, url: str, wait_for_selector: Optional[str] = None, wait_for_navigation: bool = True
    ) -> Dict[str, Any]:
        """
        Navigates to a URL, waits for load, and returns the page state.

        Args:
            url: The URL to browse (will prepend https:// if missing).
            wait_for_selector: Optional CSS selector to wait for after navigation.
            wait_for_navigation: If True (default), wait for 'networkidle' state, else 'domcontentloaded'.

        Returns:
            Dictionary containing 'success' (bool) and 'page_state' (dict).
        """
        await self._ensure_initialized()
        self._update_activity()

        if not isinstance(url, str) or not url.strip():
            raise ToolInputError("URL cannot be empty.")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        ctx, _ = await get_browser_context()  # Use shared context by default

        # Fix #10: Check proxy allowlist if proxy is active for this context
        if ctx.proxy and not _is_domain_allowed_for_proxy(url):
            proxy_server = ctx.proxy.get("server", "Unknown Proxy")
            error_msg = f"Navigation blocked: Domain for URL '{url}' is not in PROXY_ALLOWED_DOMAINS for proxy '{proxy_server}'."
            await _log("browse_fail_proxy_disallowed", url=url, proxy=proxy_server)
            raise ToolError(error_msg, error_code="proxy_domain_disallowed")

        async with _tab_context(ctx) as page:
            await _log("navigate_start", url=url)
            try:
                wait_until_state = "networkidle" if wait_for_navigation else "domcontentloaded"
                await page.goto(
                    url, wait_until=wait_until_state, timeout=60000
                )  # 60s navigation timeout

                if wait_for_selector:
                    try:
                        await page.wait_for_selector(
                            wait_for_selector, state="visible", timeout=15000
                        )  # 15s wait
                        await _log("navigate_wait_selector_ok", selector=wait_for_selector)
                    except PlaywrightTimeoutError:
                        await _log("navigate_wait_selector_timeout", selector=wait_for_selector)
                        # Don't fail the whole browse, just log the timeout
                        logger.warning(
                            f"Timed out waiting for selector '{wait_for_selector}' after navigation to {url}"
                        )

                await _pause(page, (50, 200))  # Short pause after load
                state = await get_page_state(page)
                await _log("navigate_success", url=url, title=state.get("title"))
                return {"success": True, "page_state": state}

            except PlaywrightException as e:
                await _log("navigate_fail_playwright", url=url, error=str(e))
                raise ToolError(f"Navigation or page load failed for {url}: {e}") from e
            except Exception as e:
                await _log("navigate_fail_unexpected", url=url, error=str(e))
                raise ToolError(f"Unexpected error browsing {url}: {e}") from e

    @tool(name="smart_browser.click")
    @with_tool_metrics
    @with_error_handling
    async def click_and_extract(
        self, url: str, target: Dict[str, Any], wait_ms: int = 1000
    ) -> Dict[str, Any]:
        """
        Navigates to a URL, clicks a target element, waits, and returns the new page state.

        Args:
            url: The URL to navigate to first.
            target: Target element specification for SmartLocator (e.g., name, role, css).
            wait_ms: Milliseconds to wait after the click (default 1000).

        Returns:
            Dictionary containing 'success' (bool) and 'page_state' (dict).
        """
        await self._ensure_initialized()
        self._update_activity()

        if not isinstance(target, dict) or not target:
            raise ToolInputError("Missing or invalid 'target' dictionary for click action.")

        # First, browse to the URL
        browse_result = await self.browse_url(url=url, wait_for_navigation=True)
        if not browse_result.get("success"):
            # If initial navigation fails, bubble up the error
            raise ToolError(
                f"Failed to load initial URL {url} before clicking.",
                details=browse_result.get("page_state"),
            )

        # Now perform the click on the current page (assuming browse_url leaves page open in context?)
        # This assumes browse_url uses the shared context, which it does.
        # We need the page object though. Let's re-implement slightly:
        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            # Navigate
            await _log("click_extract_navigate", url=url)
            await page.goto(url, wait_until="networkidle", timeout=60000)

            # Click the target
            await smart_click(page, **target)  # smart_click handles logging/errors

            # Wait after click
            if wait_ms > 0:
                await page.wait_for_timeout(wait_ms)

            # Wait for potential navigation/update after click
            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=10000
                )  # Wait up to 10s for idle
            except PlaywrightTimeoutError:
                logger.debug("Network did not become idle after click+wait. Proceeding anyway.")
                pass  # Ignore timeout, page might be dynamically updating

            # Get final page state
            await _pause(page, (50, 200))
            final_state = await get_page_state(page)
            await _log("click_extract_success", url=url, target=target)

            return {"success": True, "page_state": final_state}

    @tool(name="smart_browser.fill_form")
    @with_tool_metrics
    @with_error_handling
    async def fill_form(
        self,
        url: str,
        form_fields: List[Dict[str, Any]],
        submit_target: Optional[Dict[str, Any]] = None,
        wait_after_submit_ms: int = 2000,
    ) -> Dict[str, Any]:
        """
        Navigates to a URL, fills form fields, optionally clicks a submit button,
        and returns the final page state.

        Args:
            url: The URL containing the form.
            form_fields: List of field dicts. Each needs 'target' (for SmartLocator)
                         and 'text' (value to type). Can also include 'enter': bool, 'clear_before': bool.
            submit_target: Optional SmartLocator target dict for the submit button.
            wait_after_submit_ms: Milliseconds to wait after submitting (default 2000).

        Returns:
             Dictionary containing 'success' (bool) and 'page_state' (dict).
        """
        await self._ensure_initialized()
        self._update_activity()

        if not form_fields or not isinstance(form_fields, list):
            raise ToolInputError("'form_fields' must be a non-empty list.")

        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            # Navigate
            await _log("fill_form_navigate", url=url)
            await page.goto(url, wait_until="networkidle", timeout=60000)

            # Fill fields sequentially
            for i, field in enumerate(form_fields):
                if not isinstance(field, dict) or "target" not in field or "text" not in field:
                    raise ToolInputError(
                        f"Invalid form_field at index {i}: {field}. Requires 'target' and 'text'."
                    )

                target = field["target"]
                text = field["text"]
                await _log("fill_form_field", index=i, target=target)
                # smart_type handles logging of success/failure/secrets internally
                await smart_type(
                    page,
                    text,
                    press_enter=field.get("enter", False),
                    clear_before=field.get("clear_before", True),
                    **target,
                )
                await _pause(page, (50, 150))  # Small pause between fields

            # Click submit button if provided
            if submit_target:
                await _log("fill_form_submit", target=submit_target)
                await smart_click(page, **submit_target)
                # Wait for navigation/update after submit
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)  # Wait up to 15s
                except PlaywrightTimeoutError:
                    logger.debug(
                        "Network did not become idle after form submit. Proceeding anyway."
                    )
                if wait_after_submit_ms > 0:
                    await page.wait_for_timeout(wait_after_submit_ms)

            # Get final page state
            await _pause(page, (100, 300))
            final_state = await get_page_state(page)
            await _log(
                "fill_form_success",
                url=url,
                num_fields=len(form_fields),
                submitted=bool(submit_target),
            )

            return {"success": True, "page_state": final_state}

    @tool(name="smart_browser.search")
    @with_tool_metrics
    @with_error_handling
    async def search(
        self,
        query: str,
        engine: str = "bing",  # Default to Bing, often more stable
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Performs a web search using the specified engine and returns results.

        Args:
            query: The search query.
            engine: Search engine: 'bing', 'duckduckgo', or 'yandex'.
            max_results: Maximum number of results to return (default 10).

        Returns:
            Dictionary with 'success', 'query', 'engine', 'results' (list), 'result_count'.
        """
        await self._ensure_initialized()
        self._update_activity()

        if max_results <= 0:
            max_results = 10

        results = await search_web(query, engine=engine, max_results=max_results)

        return {
            "success": True,
            "query": query,
            "engine": engine,
            "results": results,
            "result_count": len(results),
        }

    @tool(name="smart_browser.download")
    @with_tool_metrics
    @with_error_handling
    async def download_file(
        self, url: str, target: Dict[str, Any], dest_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Navigates to a URL, clicks a target element to trigger a download,
        saves the file, and returns info (path, hash, size, tables).

        Args:
            url: URL to navigate to first.
            target: Target element specification for SmartLocator to click.
            dest_dir: Optional destination directory (defaults to ~/.smart_browser/downloads).

        Returns:
            Dictionary containing 'success' (bool) and 'download' info dict.
        """
        await self._ensure_initialized()
        self._update_activity()

        if not isinstance(target, dict) or not target:
            raise ToolInputError("Missing or invalid 'target' dictionary for download action.")

        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            # Navigate
            await _log("download_navigate", url=url)
            await page.goto(url, wait_until="networkidle", timeout=60000)

            # Initiate download via click
            download_info = await smart_download(
                page, target, dest_dir
            )  # Handles logging and errors

            # smart_download now returns the full info dict including success status
            if not download_info.get("success"):
                # If download failed, bubble up the error stored in the result
                raise ToolError(
                    f"Download failed: {download_info.get('error', 'Unknown reason')}",
                    details=download_info,
                )

            return {
                "success": True,
                "download": download_info,  # Return the detailed info from smart_download
            }

    @tool(name="smart_browser.download_site_pdfs")
    @with_tool_metrics
    @with_error_handling
    async def download_site_pdfs(
        self,
        start_url: str,
        dest_subfolder: Optional[str] = None,  # Optional subfolder name
        include_regex: Optional[str] = None,
        max_depth: int = 2,
        max_pdfs: int = 100,
        max_pages_crawl: int = 500,
        rate_limit_rps: float = 1.0,  # Slower default rate for bulk downloads
    ) -> Dict[str, Any]:
        """
        Crawls a site starting from *start_url* to find PDF links (optionally filtered
        by *include_regex* within *max_depth*). Downloads found PDFs (up to *max_pdfs*)
        using direct HTTP requests (no browser interaction needed for downloads) with
        rate limiting. Saves files under ~/.smart_browser/downloads/<dest_subfolder>/
        (subfolder defaults to slug of start_url domain).

        Args:
            start_url: URL to start crawling (can be domain or specific page).
            dest_subfolder: Optional name for subfolder within downloads directory.
            include_regex: Regex to filter PDF URLs.
            max_depth: Max crawl depth.
            max_pdfs: Max PDFs to download.
            max_pages_crawl: Safety limit on pages visited during crawl phase.
            rate_limit_rps: Download rate limit in requests/second.

        Returns:
            Dict with 'success', 'pdf_count', 'dest_dir', and 'files' list (each item
            is a dict with 'url', 'file', 'size', 'sha256', 'success', 'error'?).
        """
        await self._ensure_initialized()  # Ensure paths/logging are ready
        self._update_activity()  # Mark activity

        # Determine destination directory
        download_base = _HOME / "downloads"
        if dest_subfolder:
            # Sanitize subfolder name
            safe_subfolder = _slugify(dest_subfolder, max_len=50)
        else:
            # Default to slug of start URL domain
            try:
                safe_subfolder = _slugify(urlparse(start_url).netloc, max_len=50) or "pdfs"
            except ValueError:
                safe_subfolder = "pdfs"
        dest_dir = download_base / safe_subfolder

        # Create directory asynchronously with permissions check
        await create_directory(str(dest_dir))  # Assumes create_directory handles permissions
        # Double check permissions if needed: os.chmod(dest_dir, 0o700) via run_in_thread

        # 1. Find PDF URLs
        logger.info(f"Starting PDF crawl from: {start_url}")
        pdf_urls_to_download = await crawl_for_pdfs(
            start_url,
            include_regex,
            max_depth,
            max_pdfs,
            max_pages_crawl,
            rate_limit_rps=5.0,  # Faster crawl rate
        )
        if not pdf_urls_to_download:
            logger.info("No PDF URLs found or matched during crawl.")
            return {"success": True, "pdf_count": 0, "dest_dir": str(dest_dir), "files": []}

        logger.info(
            f"Found {len(pdf_urls_to_download)} PDF URLs to download. Starting downloads..."
        )

        # 2. Download PDFs directly using httpx and rate limiting
        limiter = RateLimiter(rate_limit_rps)  # Apply download rate limit
        download_tasks = []
        for i, pdf_url in enumerate(pdf_urls_to_download):
            # Create task: acquire limiter then call download function
            async def download_task(url, seq):
                await limiter.acquire()
                return await _download_file_direct(url, dest_dir, seq)

            download_tasks.append(asyncio.create_task(download_task(pdf_url, i + 1)))

        # Execute downloads concurrently
        download_results = await asyncio.gather(*download_tasks)

        successful_downloads = [res for res in download_results if res.get("success")]
        failed_downloads = [res for res in download_results if not res.get("success")]

        await _log(
            "download_site_pdfs_complete",
            start_url=start_url,
            total_found=len(pdf_urls_to_download),
            successful=len(successful_downloads),
            failed=len(failed_downloads),
            dest_dir=str(dest_dir),
        )

        return {
            "success": True,  # Overall operation succeeded, check individual file results
            "pdf_count": len(successful_downloads),
            "dest_dir": str(dest_dir),
            "files": download_results,  # Return results for all attempted downloads
        }

    @tool(name="smart_browser.collect_documentation")
    @with_tool_metrics
    @with_error_handling
    async def collect_documentation(
        self,
        package: str,  # Name of the package/library
        max_pages: int = 40,
        rate_limit_rps: float = 2.0,  # Rate limit for fetching doc pages
    ) -> Dict[str, Any]:
        """
        Searches for the documentation site of an open-source package, crawls it,
        extracts readable text from pages, and saves the combined text to a file.

        Args:
            package: The name of the package/library (e.g., "requests", "langchain").
            max_pages: Maximum number of documentation pages to fetch and process.
            rate_limit_rps: Max requests per second for crawling the doc site.

        Returns:
            Dict with 'success', 'package', 'pages_collected', 'file_path', 'root_url'.
        """
        await self._ensure_initialized()
        self._update_activity()

        # 1. Find documentation root URL
        docs_root = await _pick_docs_root(package)
        if not docs_root:
            raise ToolError(
                f"Could not automatically find a documentation site for '{package}'. Please provide a specific start URL."
            )

        # 2. Crawl the site and extract text
        pages_content = await crawl_docs_site(
            docs_root, max_pages=max_pages, rate_limit_rps=rate_limit_rps
        )
        if not pages_content:
            # Don't treat as error, just no content found
            logger.warning(
                f"Documentation crawl for '{package}' starting at {docs_root} yielded 0 pages with text."
            )
            return {
                "success": True,
                "package": package,
                "pages_collected": 0,
                "file_path": None,
                "root_url": docs_root,
                "message": "No content collected.",
            }

        # 3. Combine content and save to file
        # Create directory with permissions handling
        scratch_dir = _HOME / "docs_collected"
        await create_directory(str(scratch_dir))  # Assumes 700 permissions if created
        # Fallback needed if create_directory fails or isn't robust

        # Create filename
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_pkg_name = _slugify(package, max_len=40)
        fname = f"{safe_pkg_name}_docs_{now_str}.txt"
        fpath = scratch_dir / fname

        # Build combined content with separators
        combined_content = f"# Documentation for: {package}\n# Source Root: {docs_root}\n\n"
        page_separator = "\n\n" + ("=" * 80) + "\n\n"  # Clear separator
        for i, (url, text) in enumerate(pages_content):
            combined_content += f"## Page {i + 1}: {url}\n\n{text.strip()}{page_separator}"

        # Write asynchronously
        await write_file_content(str(fpath), combined_content)
        # Set permissions on file
        await _run_in_thread(os.chmod, fpath, 0o600)

        await _log(
            "docs_collected_success",
            package=package,
            root_url=docs_root,
            pages=len(pages_content),
            file=str(fpath),
        )

        return {
            "success": True,
            "package": package,
            "pages_collected": len(pages_content),
            "file_path": str(fpath),
            "root_url": docs_root,
        }

    @tool(name="smart_browser.run_macro")
    @with_tool_metrics
    @with_error_handling
    async def execute_macro(
        self, url: str, task: str, model: str = "gpt-4o", max_rounds: int = 7
    ) -> Dict[str, Any]:
        """
        Navigates to a URL and executes a natural language task using an LLM planner.

        Args:
            url: The starting URL.
            task: Natural language description of the task to perform.
            model: LLM model to use for planning (default gpt-4o).
            max_rounds: Maximum number of planning/execution rounds (default 7).

        Returns:
            Dictionary with 'success', 'task', 'steps' (list of results), 'final_page_state'.
        """
        await self._ensure_initialized()
        self._update_activity()

        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            # Navigate to start URL first
            await _log("macro_navigate", url=url)
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
            except PlaywrightException as nav_err:
                raise ToolError(
                    f"Macro failed: Could not navigate to initial URL {url}: {nav_err}"
                ) from nav_err

            # Run the macro using the plan-act loop
            results = await run_macro(
                page, task, max_rounds, model
            )  # Handles internal logging/errors

            # Get final page state, regardless of macro success/failure
            try:
                final_state = await get_page_state(page)
            except Exception as state_err:
                logger.error(f"Failed to get final page state after macro execution: {state_err}")
                final_state = {"error": f"Failed to get final page state: {state_err}"}

            # Determine overall success based on results (e.g., no errors, finish action hit)
            macro_success = True
            if not results or any(step.get("action") == "error" for step in results):
                macro_success = False
            # Check if last non-finish step failed? More nuanced check might be needed.

            return {
                "success": macro_success,
                "task": task,
                "steps": results,
                "final_page_state": final_state,
            }

    @tool(name="smart_browser.autopilot")
    @with_tool_metrics
    @with_error_handling
    async def autopilot(
        self,
        task: str,
        scratch_subdir: str = "autopilot_runs",
        max_steps: int = 15,  # Reduced default max steps
    ) -> Dict[str, Any]:
        """
        Executes an arbitrary multi-step task using LLM planning and available tools.
        Logs results to a file and handles replanning on failure.

        Args:
            task: Natural language description of the overall task.
            scratch_subdir: Subdirectory within ~/.smart_browser to store run logs.
            max_steps: Maximum number of tool execution steps allowed.

        Returns:
            Dict with 'success', 'steps_executed', 'run_log' path, 'final_results' list.
        """
        await self._ensure_initialized()
        self._update_activity()

        # Setup logging directory and file
        scratch_dir = _HOME / scratch_subdir
        await create_directory(str(scratch_dir))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = scratch_dir / f"autopilot_{run_id}.jsonl"
        await _run_in_thread(os.chmod, scratch_dir, 0o700)  # Ensure dir perms

        logger.info(f"Autopilot run started. Task: '{task[:100]}...'. Log: {log_path}")
        all_results = []
        current_task = task

        try:
            # Initial plan generation
            current_plan = await _plan_autopilot(current_task)

            # Execute plan step-by-step
            step_num = 0
            while step_num < max_steps and current_plan:
                step_num += 1
                step_to_execute = current_plan.pop(0)  # Get next step
                tool_name = step_to_execute.get("tool")
                args = step_to_execute.get("args", {})

                step_log_entry = {
                    "step": step_num,
                    "tool": tool_name,
                    "args": args,
                    "success": False,  # Default
                }

                if tool_name not in _AVAILABLE_TOOLS:
                    error_msg = f"Unknown tool '{tool_name}' requested in plan."
                    logger.error(error_msg)
                    step_log_entry["error"] = error_msg
                else:
                    method_name = _AVAILABLE_TOOLS[tool_name][0]
                    try:
                        # Get the actual method from self
                        tool_method = getattr(self, method_name)

                        await _log("autopilot_step_start", step=step_num, tool=tool_name, args=args)
                        self._update_activity()  # Update activity before long-running tool call
                        # Execute the tool method
                        outcome = await tool_method(**args)
                        self._update_activity()  # Update activity after tool call

                        # Record outcome
                        step_log_entry["success"] = outcome.get(
                            "success", True
                        )  # Assume success if key missing? Maybe False. Let's assume False.
                        step_log_entry["success"] = outcome.get(
                            "success", False
                        )  # Default to False if key missing
                        step_log_entry["result"] = outcome  # Store full result

                        if not step_log_entry["success"]:
                            step_log_entry["error"] = outcome.get(
                                "error", "Tool failed without specific error message."
                            )
                            await _log(
                                "autopilot_step_fail",
                                step=step_num,
                                tool=tool_name,
                                error=step_log_entry["error"],
                            )
                            logger.warning(
                                f"Autopilot Step {step_num} ({tool_name}) failed: {step_log_entry['error']}"
                            )
                            # Fix #16: Replanning logic
                            logger.info(f"Attempting to replan after failed step {step_num}...")
                            try:
                                # Pass current task and *all* prior results for context
                                new_plan_tail = await _plan_autopilot(
                                    current_task, all_results + [step_log_entry]
                                )
                                # Replace the rest of the plan queue
                                current_plan = new_plan_tail
                                logger.info(
                                    f"Replanning successful. New plan has {len(current_plan)} steps."
                                )
                                await _log("autopilot_replan_success", new_steps=len(current_plan))
                                # Continue to next iteration with the new plan
                                continue  # Skip appending the failed step result here? No, append failure, loop continues with new plan.
                            except Exception as replan_err:
                                logger.error(
                                    f"Replanning failed after step {step_num} failure: {replan_err}"
                                )
                                await _log("autopilot_replan_fail", error=str(replan_err))
                                # Stop execution if replanning fails
                                current_plan = []  # Clear plan to stop loop
                        else:
                            await _log(
                                "autopilot_step_success",
                                step=step_num,
                                tool=tool_name,
                                result_summary=str(outcome)[:200],
                            )

                    except ToolInputError as tie:
                        step_log_entry["error"] = f"Invalid arguments for tool '{tool_name}': {tie}"
                        logger.error(
                            f"Autopilot Step {step_num} failed: {step_log_entry['error']}",
                            exc_info=True,
                        )
                        current_plan = []  # Stop on bad input errors
                    except ToolError as te:
                        step_log_entry["error"] = f"Tool execution error for '{tool_name}': {te}"
                        logger.error(
                            f"Autopilot Step {step_num} failed: {step_log_entry['error']}",
                            exc_info=True,
                        )
                        # Allow potential replanning for tool errors
                    except Exception as e:
                        step_log_entry["error"] = (
                            f"Unexpected error executing tool '{tool_name}': {e}"
                        )
                        logger.critical(
                            f"Autopilot Step {step_num} failed unexpectedly: {step_log_entry['error']}",
                            exc_info=True,
                        )
                        current_plan = []  # Stop on unexpected errors

                # Append result of this step to overall results
                all_results.append(step_log_entry)

                # Asynchronously write log entry to file
                try:
                    async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                        await log_f.write(json.dumps(step_log_entry) + "\n")
                        await log_f.flush()
                except IOError as log_e:
                    logger.error(f"Failed to write autopilot log entry to {log_path}: {log_e}")

            # End of loop (max steps reached or plan empty)
            if step_num >= max_steps:
                logger.warning(f"Autopilot run stopped: Maximum step limit ({max_steps}) reached.")
                await _log("autopilot_max_steps", task=task, steps=step_num)
            elif not current_plan:
                logger.info(
                    f"Autopilot run finished: Plan completed or stopped after {step_num} steps."
                )
                await _log("autopilot_plan_end", task=task, steps=step_num)

            # Set file permissions after closing
            await _run_in_thread(os.chmod, log_path, 0o600)

            overall_success = all_results and all_results[-1].get(
                "success", False
            )  # Simplistic: success if last step succeeded

            return {
                "success": overall_success,
                "steps_executed": step_num,
                "run_log": str(log_path),
                "final_results": all_results[-min(len(all_results), 3) :],  # Return last 3 results
            }

        except Exception as autopilot_err:
            # Catch errors during initial planning or unexpected loop issues
            logger.critical(f"Autopilot run failed critically: {autopilot_err}", exc_info=True)
            await _log("autopilot_critical_error", task=task, error=str(autopilot_err))
            # Write error to log file if possible
            error_entry = {
                "step": 0,
                "success": False,
                "error": f"Autopilot critical failure: {autopilot_err}",
            }
            try:
                async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                    await log_f.write(json.dumps(error_entry) + "\n")
                await _run_in_thread(os.chmod, log_path, 0o600)
            except Exception:
                pass  # Ignore errors writing final error log

            raise ToolError(f"Autopilot failed: {autopilot_err}") from autopilot_err

    @tool(name="smart_browser.parallel")
    @with_tool_metrics
    @with_error_handling
    async def parallel_process(
        self,
        urls: List[str],
        action: str = "get_state",  # Action per URL: 'get_state', 'screenshot', etc.
        max_tabs: Optional[int] = None,  # Use pool's default if None
    ) -> Dict[str, Any]:
        """
        Processes multiple URLs in parallel using the tab pool.
        Currently supports 'get_state' action for each URL.

        Args:
            urls: List of URLs to process.
            action: Action to perform on each URL (currently 'get_state').
            max_tabs: Override the default max tabs for this specific operation.

        Returns:
            Dictionary with 'success', 'results' (list of outcomes per URL), 'processed_count'.
        """
        await self._ensure_initialized()
        self._update_activity()

        if not urls or not isinstance(urls, list):
            raise ToolInputError("Must provide a list of URLs.")
        if action != "get_state":
            raise ToolInputError(
                f"Unsupported parallel action: '{action}'. Currently only 'get_state' is supported."
            )

        # Use a temporary pool if max_tabs is specified, else the shared one
        pool = TabPool(max_tabs=max_tabs) if max_tabs is not None else self.tab_pool

        # Define the task function for each URL
        async def process_single_url(page: Page, *, url_to_process: str) -> Dict[str, Any]:
            """Task to navigate and get state for one URL."""
            try:
                # Prepend https:// if scheme is missing
                if not url_to_process.startswith(("http://", "https://")):
                    url_to_process = f"https://{url_to_process}"

                await _log("parallel_navigate", url=url_to_process)
                await page.goto(
                    url_to_process, wait_until="networkidle", timeout=45000
                )  # 45s timeout per page
                state = await get_page_state(page)
                return {"url": url_to_process, "success": True, "page_state": state}
            except PlaywrightException as e:
                await _log("parallel_url_error", url=url_to_process, error=str(e))
                return {"url": url_to_process, "success": False, "error": f"Playwright error: {e}"}
            except Exception as e:
                await _log("parallel_url_error", url=url_to_process, error=str(e))
                return {"url": url_to_process, "success": False, "error": f"Unexpected error: {e}"}

        # Create partial functions to pass URL correctly
        tasks_to_run = [functools.partial(process_single_url, url_to_process=url) for url in urls]

        # Run tasks using the pool's map function
        results = await pool.map(tasks_to_run)

        successful_count = sum(1 for r in results if r.get("success"))
        await _log(
            "parallel_process_complete",
            total_urls=len(urls),
            successful=successful_count,
            action=action,
        )

        return {
            "success": True,  # The parallel operation itself succeeded
            "results": results,
            "processed_count": len(results),
            "successful_count": successful_count,
        }

    # --- Lifecycle Methods for Server Integration ---
    async def async_setup(self):
        """Called by MCP server during startup (if using lifespan)."""
        logger.info("SmartBrowserTool async_setup called.")
        await self._ensure_initialized()

    async def async_teardown(self):
        """Called by MCP server during shutdown (if using lifespan)."""
        global _selector_cleanup_task_handle # Use global handle
        logger.info("SmartBrowserTool async_teardown called.")

        # Cancel the selector cleanup task
        cleanup_task = _selector_cleanup_task_handle # <<< MODIFIED Read global
        if cleanup_task and not cleanup_task.done():
            logger.info("Cancelling selector cleanup task...")
            cleanup_task.cancel()
            try:
                await asyncio.wait_for(cleanup_task, timeout=5.0) # Wait briefly for cancellation
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("Selector cleanup task did not finish cancelling gracefully.")
            except Exception as e:
                 logger.error(f"Error awaiting selector cleanup task cancellation: {e}")
            _selector_cleanup_task_handle = None # Clear handle

        # Use the safe shutdown initiator for browser etc.
        await _initiate_shutdown()