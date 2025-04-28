"""
Smart Browser - Playwright-powered web automation tool for Ultimate MCP Server.

Provides enterprise-grade web automation with comprehensive features for scraping,
testing, and browser automation tasks with built-in security, resilience, and ML capabilities.

FEATURES
────────────────────────────────────────────────────────────────────────────────
✓ 1  Enterprise audit log (hash-chained JSONL in ~/.smart_browser/audit.log)
✓ 2  Secret vault / ENV bridge → get_secret("env:FOO") / get_secret("vault:kv/data/foo#bar")
✓ 3  Headful toggle + optional VNC remote viewing (HEADLESS=0, VNC=1)
✓ 4  Proxy rotation from PROXY_POOL for IP diversity (PROXY_POOL="...")
✓ 5  AES-GCM-encrypted cookie jar with AAD (Requires SB_STATE_KEY set programmatically - Disabled by default)
✓ 6  Human-like jitter on UI actions with risk-aware timing
✓ 7  Resilient "chaos-monkey" retries with idempotent re-play
✓ 8  Async TAB POOL for parallel scraping with concurrency control (SB_MAX_TABS=N)
✓ 9  Pluggable HTML summarizer (trafilatura ▸ readability-lxml ▸ fallback) + Readability.js
✓ 10 Download helper with SHA-256 verification and audit logging
✓ 11 PDF / Excel auto-table extraction after download (using ThreadPool)
✓ 12 Enhanced element locator (Cache ▸ Heuristic ▸ LLM) with self-healing cache
✓ 13 DOM fingerprinting for selector cache validation
✓ 14 LLM-powered page state analysis and action recommendation / planning
✓ 15 Natural-language macro runner (ReAct-style plan → act loop)
✓ 16 Universal search across multiple engines (Yandex, Bing, DuckDuckGo)
✓ 17 Form-filling with secure credential handling
✓ 18 Element state extraction (including shadow DOM) and DOM mapping
✓ 19 Multi-tab parallel URL processing
✓ 20 Browser lifecycle management with secure shutdown & inactivity monitor

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
import textwrap  # Added
import threading
import time
import unicodedata
import urllib.parse
from collections import deque
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlparse

import aiofiles
import httpx
from bs4 import BeautifulSoup
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from playwright._impl._errors import Error as PlaywrightException
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError

# Playwright imports
from playwright.async_api import (
    Browser,
    BrowserContext,
    Frame,
    Locator,
    Page,
    async_playwright,
)

from ultimate_mcp_server.config import SmartBrowserConfig, get_config

# MCP Server imports (assuming these exist)
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider, parse_model_string
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.tools.base import (
    BaseTool,
    tool,
    with_error_handling,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.completion import chat_completion
from ultimate_mcp_server.tools.filesystem import (
    create_directory,
    delete_path,
    get_unique_filepath,
    read_binary_file,
    read_file,
    write_file,
)
from ultimate_mcp_server.utils import get_logger

# For loop binding and forked process detection
_pid = os.getpid()


def _loop():
    return asyncio.get_running_loop()


def _log_lock():
    """Returns the asyncio Lock dedicated to audit logging."""
    return _audit_log_lock

def _playwright_lock():
    return asyncio.Lock()


logger = get_logger("ultimate_mcp_server.tools.smart_browser")

try:
    import trafilatura  # type: ignore
except ImportError:
    trafilatura = None

try:
    from readability import Document  # type: ignore
except ImportError:
    Document = None


# ══════════════════════════════════════════════════════════════════════════════
# Global Variables for Configuration (Populated by _ensure_initialized)
# ══════════════════════════════════════════════════════════════════════════════
_sb_state_key_b64_global: Optional[str] = None
_sb_max_tabs_global: int = 5  # Default value
_sb_tab_timeout_global: int = 300  # Default value
_sb_inactivity_timeout_global: int = 600  # Default value
_headless_mode_global: bool = True  # Default value
_vnc_enabled_global: bool = False  # Default value
_vnc_password_global: Optional[str] = None  # Default value
_proxy_pool_str_global: str = ""  # Default value
_proxy_allowed_domains_str_global: str = "*"  # Default value
_vault_allowed_paths_str_global: str = "secret/data/,kv/data/"  # Default value
_max_widgets_global: int = 300  # Default value
_max_section_chars_global: int = 5000  # Default value
_dom_fp_limit_global: int = 20000  # Default value
_llm_model_locator_global: str = "gpt-4.1-mini"  # Default value
_retry_after_fail_global: int = 1  # Default value
_seq_cutoff_global: float = 0.72  # Default value
_area_min_global: int = 400  # Default value
_high_risk_domains_set_global: Set[str] = set()  # Populated during init

# Global Variables for Internal State
_pw: Optional[async_playwright] = None
_browser: Optional[Browser] = None
_ctx: Optional[BrowserContext] = None
_vnc_proc: Optional[subprocess.Popen] = None
_last_hash: str | None = None
_js_lib_cached: Set[str] = set()
_js_lib_lock = asyncio.Lock()
_audit_log_lock = asyncio.Lock()
_db_conn_pool_lock = threading.RLock()
_db_connection: sqlite3.Connection | None = None
_locator_cache_cleanup_task_handle: Optional[asyncio.Task] = None
_PROXY_CONFIG_DICT: Optional[Dict[str, Any]] = None  # Parsed proxy config
_PROXY_ALLOWED_DOMAINS_LIST: Optional[List[str]] = None  # Parsed allowed domains
_ALLOWED_VAULT_PATHS: Set[str] = set()  # Parsed allowed vault paths

# Thread pool for CPU-bound tasks & sync I/O execution
_cpu_count = os.cpu_count() or 1
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, _cpu_count * 2 + 4), thread_name_prefix="sb_worker"
)


def _get_pool():
    global _thread_pool, _pid
    if _pid != os.getpid():  # child after fork
        _thread_pool.shutdown(wait=False)
        _thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, _sb_max_tabs_global * 2), thread_name_prefix="sb_worker"
        )
        _pid = os.getpid()
    return _thread_pool


# ══════════════════════════════════════════════════════════════════════════════
# 0.  FILESYSTEM & ENCRYPTION & NEW LOCATOR DB
# ══════════════════════════════════════════════════════════════════════════════
_SB_INTERNAL_BASE_PATH_STR: Optional[str] = None # Will hold the absolute path resolved by FileSystemTool

_STATE_FILE: Optional[Path] = None
_LOG_FILE: Optional[Path] = None
_CACHE_DB: Optional[Path] = None
_READ_JS_CACHE: Optional[Path] = None


# --- Encryption ---
CIPHER_VERSION = b"SB1"
AAD_TAG = b"smart-browser-state-v1"


def _key() -> bytes | None:
    """Get AES-GCM key from the globally set config value."""
    if not _sb_state_key_b64_global:
        return None
    try:
        decoded = base64.b64decode(_sb_state_key_b64_global)
        if len(decoded) not in (16, 24, 32):
            logger.warning(
                f"Invalid SB State Key length: {len(decoded)} bytes. Need 16, 24, or 32."
            )
            return None
        return decoded
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid base64 SB State Key: {e}")
        return None


def _enc(buf: bytes) -> bytes:
    """Encrypt data using AES-GCM with AAD if key is set."""
    k = _key()
    if not k:
        # If key is not set, return the original buffer (no encryption)
        logger.debug("SB_STATE_KEY not set. Skipping encryption for state.")
        return buf
    # Key is set, proceed with encryption
    try:
        nonce = os.urandom(12)
        encrypted_data = AESGCM(k).encrypt(nonce, buf, AAD_TAG)
        return CIPHER_VERSION + nonce + encrypted_data
    except Exception as e:
        logger.error(f"Encryption failed: {e}", exc_info=True)
        raise RuntimeError(f"Encryption failed: {e}") from e


def _dec(buf: bytes) -> bytes | None:
    """Decrypt data using AES-GCM with AAD if key is set and buffer looks encrypted."""
    k = _key()
    if not k:
        # If key is not set, assume buffer is unencrypted plaintext
        logger.debug("SB_STATE_KEY not set. Assuming state is unencrypted.")
        # Basic check: does it look like JSON? (optional but helpful)
        try:
            if buf.strip().startswith(b"{") or buf.strip().startswith(b"["):
                return buf
            else:
                 logger.warning("Unencrypted state file doesn't look like JSON. Ignoring.")
                 return None
        except Exception:
             logger.warning("Error checking unencrypted state file format. Ignoring.")
             return None

    # Key is set, attempt decryption
    # Check if it starts with our cipher version header
    if not buf.startswith(CIPHER_VERSION):
        logger.warning("State file exists but lacks expected encryption header. Treating as legacy/invalid.")
        # Decide policy: return None (ignore), or try to parse as JSON? Let's ignore.
        _STATE_FILE.unlink(missing_ok=True) # Remove potentially corrupt/old file
        return None

    # Ensure buffer is long enough for header, nonce, and some ciphertext
    if len(buf) < len(CIPHER_VERSION) + 12 + 1:
        logger.error("State file too short to be valid encrypted data")
        return None

    # Parse fixed-length components
    _HDR, nonce, ciphertext = (
        buf[: len(CIPHER_VERSION)],
        buf[len(CIPHER_VERSION) : len(CIPHER_VERSION) + 12],
        buf[len(CIPHER_VERSION) + 12 :],
    )

    try:
        return AESGCM(k).decrypt(nonce, ciphertext, AAD_TAG)
    except InvalidTag: # Keep specific error handling
        logger.error("Decryption failed: Invalid tag (tampered/wrong key?)")
        # Option: Raise specific error or just return None/delete file
        _STATE_FILE.unlink(missing_ok=True)
        raise RuntimeError("State-file authentication failed (InvalidTag)") from None # Re-raise as critical
    except Exception as e:
        logger.error(f"Decryption failed: {e}.", exc_info=True)
        _STATE_FILE.unlink(missing_ok=True)
        return None

# --- Enhanced Locator Cache DB ---
def _get_db_connection() -> sqlite3.Connection:
    """Get or create the single shared SQLite connection."""
    global _db_connection
    with _db_conn_pool_lock:
        if _db_connection is None:
            try:
                _db_connection = sqlite3.connect(
                    _CACHE_DB,
                    check_same_thread=False,
                    isolation_level=None,
                    timeout=10,
                )
                _db_connection.execute("PRAGMA journal_mode=WAL")
                _db_connection.execute("PRAGMA synchronous=FULL")
                _db_connection.execute("PRAGMA busy_timeout = 10000")
                logger.info(f"Initialized SQLite DB connection to {_CACHE_DB}")
            except sqlite3.Error as e:
                logger.critical(
                    f"Failed to connect/init SQLite DB at {_CACHE_DB}: {e}", exc_info=True
                )
                raise RuntimeError(f"Failed to initialize database: {e}") from e
        return _db_connection


def _close_db_connection():
    """Close the SQLite connection."""
    global _db_connection
    with _db_conn_pool_lock:
        if _db_connection is not None:
            try:
                # Explicitly checkpoint WAL file before closing is good practice
                _db_connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.Error as e:
                logger.warning(f"Error during WAL checkpoint before closing DB: {e}")
            try:
                _db_connection.close()
                logger.info("Closed SQLite DB connection.")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite DB connection: {e}")
            finally:
                _db_connection = None


atexit.register(_close_db_connection)


def _init_locator_cache_db_sync():
    """Synchronous DB schema initialization for the locator cache."""
    conn = None
    try:
        conn = _get_db_connection()
        with closing(conn.cursor()) as cursor:
            # Create the table with composite primary key (key, dom_fp)
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS selector_cache(
                    key       TEXT,
                    selector  TEXT NOT NULL,
                    dom_fp    TEXT NOT NULL,
                    hits      INTEGER DEFAULT 1,
                    created_ts INTEGER DEFAULT (strftime('%s', 'now')),
                    last_hit  INTEGER DEFAULT (strftime('%s', 'now')),
                    PRIMARY KEY (key, dom_fp)
                );"""
            )

            # Check if we need to add the last_hit column (for existing installations)
            try:
                cursor.execute("SELECT last_hit FROM selector_cache LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Adding last_hit column to selector_cache table...")
                cursor.execute(
                    "ALTER TABLE selector_cache ADD COLUMN last_hit INTEGER DEFAULT(strftime('%s','now'))"
                )

            logger.info(f"Enhanced Locator cache DB schema initialized/verified at {_CACHE_DB}")
    except sqlite3.Error as e:
        logger.critical(f"Failed to initialize locator cache DB schema: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize locator cache database: {e}") from e


_init_locator_cache_db_sync()


def _cache_put_sync(key: str, selector: str, dom_fp: str) -> None:
    """Synchronous write/update to the locator cache."""
    try:
        conn = _get_db_connection()
        conn.execute(
            """INSERT INTO selector_cache(key, selector, dom_fp, created_ts, last_hit)
               VALUES (?, ?, ?, strftime('%s', 'now'), strftime('%s', 'now'))
               ON CONFLICT(key, dom_fp) DO UPDATE SET
                 hits = hits + 1,
                 last_hit = strftime('%s', 'now')
               WHERE key = excluded.key AND dom_fp = excluded.dom_fp;""",
            (key, selector, dom_fp),
        )
    except sqlite3.Error as e:
        logger.error(f"Failed to write to locator cache (key prefix={key[:8]}...): {e}")


def _cache_delete_sync(key: str) -> None:
    """Synchronously delete an entry from the locator cache by key."""
    try:
        conn = _get_db_connection()
        logger.debug(f"Deleting stale cache entry with key prefix: {key[:8]}...")
        cursor = conn.execute("DELETE FROM selector_cache WHERE key = ?", (key,))
        if cursor.rowcount > 0:
            logger.debug(f"Successfully deleted stale cache entry {key[:8]}...")
    except sqlite3.Error as e:
        logger.error(f"Failed to delete stale cache entry (key prefix={key[:8]}...): {e}")
    except Exception as e:
        logger.error(
            f"Unexpected error deleting cache entry (key prefix={key[:8]}...): {e}", exc_info=True
        )


def _cache_get_sync(key: str, dom_fp: str) -> Optional[str]:
    """Synchronous read from cache, checking fingerprint. Deletes stale entries."""
    row = None
    try:
        conn = _get_db_connection()
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT selector FROM selector_cache WHERE key=? AND dom_fp=?", (key, dom_fp)
            )
            row = cursor.fetchone()
            if row:
                # Update last_hit timestamp for this exact match
                conn.execute(
                    "UPDATE selector_cache SET last_hit = strftime('%s', 'now') WHERE key=? AND dom_fp=?",
                    (key, dom_fp),
                )
                return row[0]  # Return the selector

            # No match with this fingerprint, check if we need to clean up old entries
            cursor.execute("SELECT 1 FROM selector_cache WHERE key=? LIMIT 1", (key,))
            if cursor.fetchone():
                # Key exists but fingerprint mismatch - clean up old versions
                logger.debug(
                    f"Cache key '{key[:8]}...' found but DOM fingerprint mismatch. Deleting."
                )
                _cache_delete_sync(key)
    except sqlite3.Error as e:
        logger.error(f"Failed to read from locator cache (key={key}): {e}")

    return None  # Not found or error


# --- Locator Cache Cleanup ---
def _cleanup_locator_cache_db_sync(retention_days: int = 90) -> int:
    """Synchronously removes old entries from the locator cache DB."""
    deleted_count = 0
    if retention_days <= 0:
        logger.info("Locator cache cleanup skipped (retention_days <= 0).")
        return 0
    try:
        conn = _get_db_connection()
        cutoff_time = f"(strftime('%s', 'now', '-{retention_days} days'))"
        logger.info(
            f"Running locator cache cleanup: Removing entries older than {retention_days} days or with hits=0..."
        )
        with closing(conn.cursor()) as cursor:
            # Delete entries that are either old OR have 0 hits (abandoned entries)
            cursor.execute(
                f"DELETE FROM selector_cache WHERE created_ts < {cutoff_time} OR hits = 0"
            )
            deleted_count = cursor.rowcount
            # Optional: Vacuum based on deleted count or fixed interval
            if deleted_count > 500:  # Example threshold
                logger.info(f"Vacuuming locator cache DB after deleting {deleted_count} entries...")
                cursor.execute("VACUUM;")
        logger.info(f"Locator cache cleanup finished. Removed {deleted_count} old entries.")
        return deleted_count
    except sqlite3.Error as e:
        logger.error(f"Error during locator cache cleanup: {e}")
        return -1  # Indicate error
    except Exception as e:
        logger.error(f"Unexpected error during locator cache cleanup: {e}", exc_info=True)
        return -1


async def _locator_cache_cleanup_task(interval_seconds: int = 24 * 60 * 60):  # Default: Daily
    """Background task to periodically run locator cache cleanup."""
    if interval_seconds <= 0:
        logger.info("Locator cache cleanup task disabled (interval <= 0).")
        return
    logger.info(f"Locator cache cleanup task started. Running every {interval_seconds} seconds.")
    await asyncio.sleep(interval_seconds)  # Wait before first run
    while True:
        try:
            loop = asyncio.get_running_loop()
            # Run the synchronous DB operation in the thread pool
            # Add retention days config later if needed, use default 90 for now
            result_count = await loop.run_in_executor(_get_pool(), _cleanup_locator_cache_db_sync)
            if result_count < 0:
                logger.warning("Locator cache cleanup run encountered an error.")
            await asyncio.sleep(interval_seconds)  # Wait for next interval
        except asyncio.CancelledError:
            logger.info("Locator cache cleanup task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in locator cache cleanup task loop: {e}", exc_info=True)
            await asyncio.sleep(60 * 5)  # Wait 5 minutes after error before retrying


# ══════════════════════════════════════════════════════════════════════════════
# 1.  AUDIT LOG (hash-chained)
# ══════════════════════════════════════════════════════════════════════════════

# Add a salt for the hash chain at module level to prevent identical hashes
_salt = os.urandom(16)  # Generate a random salt at module load time


def _sanitize_for_log(obj: Any) -> Any:
    """Sanitize values for JSON logging, preventing injection."""
    if isinstance(obj, str):
        try:
            s = re.sub(r"[\x00-\x1f\x7f]", "", obj)  # Remove control characters
            encoded = json.dumps(s)
            return encoded[1:-1] if len(encoded) >= 2 else ""
        except TypeError:
            return "???"
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_log(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_log(item) for item in obj]
    elif isinstance(obj, (int, float, bool, type(None))):
        return obj
    else:
        try:
            s = str(obj)
            s = re.sub(r"[\x00-\x1f\x7f]", "", s)  # Remove control characters
            encoded = json.dumps(s)
            return encoded[1:-1] if len(encoded) >= 2 else ""
        except Exception:
            return "???"


_EVENT_EMOJI_MAP = {
    "browser_start": "🚀",
    "browser_shutdown": "🛑",
    "browser_shutdown_complete": "🏁",
    "browser_context_create": "➕",
    "browser_incognito_context": "🕶️",
    "browser_context_close_shared": "➖",
    "browser_close": "🚪",
    "page_open": "📄",
    "page_close": "덮",
    "page_error": "🔥",
    "tab_timeout": "⏱️",
    "tab_cancelled": "🚫",
    "tab_error": "💥",
    "navigate": "➡️",
    "navigate_start": "➡️",
    "navigate_success": "✅",
    "navigate_fail_playwright": "❌",
    "navigate_fail_unexpected": "💣",
    "navigate_wait_selector_ok": "👌",
    "navigate_wait_selector_timeout": "⏳",
    "page_state_extracted": "ℹ️",
    "browse_fail_proxy_disallowed": "🛡️",
    "click": "🖱️",
    "click_success": "🖱️✅",
    "click_fail_notfound": "🖱️❓",
    "click_fail_playwright": "🖱️❌",
    "click_fail_unexpected": "🖱️💣",
    "type": "⌨️",
    "type_success": "⌨️✅",
    "type_fail_secret": "⌨️🔑",
    "type_fail_notfound": "⌨️❓",
    "type_fail_playwright": "⌨️❌",
    "type_fail_unexpected": "⌨️💣",
    "scroll": "↕️",
    "locator_cache_hit": "⚡",
    "locator_heuristic_match": "🧠",
    "locator_llm_pick": "🤖🎯",
    "locator_fail_all": "❓❓",
    "locator_text_fallback": "✍️",
    "locator_success": "🎯",
    "locator_fail": "❓",
    "download": "💾",
    "download_navigate": "🚚",
    "download_success": "💾✅",
    "download_fail_notfound": "💾❓",
    "download_fail_timeout": "💾⏱️",
    "download_fail_playwright": "💾❌",
    "download_fail_unexpected": "💾💣",
    "download_pdf_http": "📄💾",
    "download_direct_success": "✨💾",
    "download_pdf_error": "📄🔥",
    "download_site_pdfs_complete": "📚✅",
    "table_extract_success": "📊✅",
    "table_extract_error": "📊❌",
    "docs_collected_success": "📖✅",
    "docs_harvest": "📖",
    "search": "🔍",
    "search_start": "🔍➡️",
    "search_complete": "🔍✅",
    "search_captcha": "🤖",
    "search_no_results_selector": "🤷",
    "search_error_playwright": "🔍❌",
    "search_error_unexpected": "🔍💣",
    "macro_plan": "📝",
    "macro_plan_generated": "📝✅",
    "macro_plan_empty": "📝🤷",
    "macro_step_result": "▶️",
    "macro_complete": "🎉",
    "macro_finish_action": "🏁",
    "macro_error": "💥",
    "macro_exceeded_rounds": "🔄",
    "macro_fail_step": "❌",
    "macro_error_tool": "🛠️💥",
    "macro_error_unexpected": "💣💥",
    "macro_navigate": "🗺️➡️",
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
    "parallel_navigate": "🚦➡️",
    "parallel_url_error": "🚦🔥",
    "parallel_process_complete": "🚦🏁",
    "retry": "⏳",
    "retry_fail": "⚠️",
    "retry_fail_unexpected": "💣⚠️",
    "retry_unexpected": "⏳💣",
    "llm_call_complete": "🤖💬",
    "readability_js_fetch": "📜💾",
}


async def _log(event: str, **details):
    """Append a hash-chained entry to the audit log asynchronously."""
    global _last_hash, _salt
    ts_iso = datetime.now(timezone.utc).isoformat()
    sanitized_details = _sanitize_for_log(details)
    emoji_key = _EVENT_EMOJI_MAP.get(event, "❓")
    async with _log_lock():
        current_last_hash = _last_hash
        entry = {
            "ts": ts_iso,
            "event": event,
            "details": sanitized_details,
            "prev": current_last_hash,
            "emoji": emoji_key,
        }
        payload = _salt + json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()
        log_entry_line = json.dumps({"hash": h, **entry}, separators=(",", ":")) + "\n"
        try:
            async with aiofiles.open(_LOG_FILE, "a", encoding="utf-8") as f:
                await f.write(log_entry_line)
                await f.flush()
                os.fsync(f.fileno())  # atomic persistence
            _last_hash = h
        except IOError as e:
            logger.error(f"Failed to write to audit log {_LOG_FILE}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing audit log: {e}", exc_info=True)


def _init_last_hash():
    global _last_hash
    if _LOG_FILE.exists():
        try:
            with open(_LOG_FILE, "rb") as f:
                data = f.read().splitlines()[-1:]  # safe even for 0/1 lines
            if data:
                last_entry = json.loads(data[0].decode("utf-8"))
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
    """Decorator for async functions; retries on common transient errors."""

    def wrap(fn):
        @functools.wraps(fn)
        async def inner(*a, **kw):
            attempt = 0
            while True:
                try:
                    if attempt > 0:
                        jitter_delay = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                        await asyncio.sleep(jitter_delay)
                    return await fn(*a, **kw)
                except (  # Specific, retryable exceptions
                    PlaywrightTimeoutError,
                    httpx.RequestError,
                    asyncio.TimeoutError,
                    # Some PlaywrightException subtypes might be retryable, but be cautious
                    # PlaywrightException, # Too broad? Might retry non-transient issues.
                ) as e:
                    # Check if the exception is specifically related to connection/network issues if possible
                    # For now, retry on Timeout and network errors
                    attempt += 1
                    func_name = getattr(fn, "__name__", "unknown_func")
                    if attempt >= max_attempts:
                        await _log(
                            "retry_fail", func=func_name, attempts=max_attempts, error=str(e)
                        )
                        raise ToolError(
                            f"Operation '{func_name}' failed after {max_attempts} attempts: {e}"
                        ) from e
                    delay = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    await _log(
                        "retry",
                        func=func_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep=round(delay, 2),
                        error=str(e),
                    )
                    # Sleep moved to start of loop
                except (
                    ToolError,
                    ValueError,
                    TypeError,
                    KeyError,
                    KeyboardInterrupt,
                    sqlite3.Error,
                ):  # Non-retryable errors
                    raise
                except Exception as e:  # Catch unexpected errors
                    attempt += 1
                    func_name = getattr(fn, "__name__", "unknown_func")
                    if attempt >= max_attempts:
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
                    await _log(
                        "retry_unexpected",
                        func=func_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep=round(delay, 2),
                        error=str(e),
                    )
                    # Sleep moved to start of loop

        return inner

    return wrap


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SECRET VAULT BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
def _update_vault_paths():
    """Parse the vault allowed paths string from global config into the global set."""
    global _ALLOWED_VAULT_PATHS
    _ALLOWED_VAULT_PATHS = set(
        path.strip().rstrip("/") + "/"
        for path in _vault_allowed_paths_str_global.split(",")
        if path.strip()
    )


def get_secret(path_key: str) -> str:
    """Retrieves secret from environment or HashiCorp Vault."""
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
            raise RuntimeError("'hvac' library required for Vault access.") from e

        # These are still read directly from env for security/bootstrapping
        addr = os.getenv("VAULT_ADDR")
        token = os.getenv("VAULT_TOKEN")
        if not addr or not token:
            raise RuntimeError("VAULT_ADDR/VAULT_TOKEN env vars must be set.")

        full_vault_uri = path_key[len("vault:") :]
        if "://" in full_vault_uri:
            raise ValueError("Vault path cannot contain '://'.")
        if "#" not in full_vault_uri:
            raise ValueError("Vault path must include '#key'.")
        path_part, key_name = full_vault_uri.split("#", 1)
        path_part = path_part.strip("/")

        # Ensure the global set is populated (should be by _ensure_initialized)
        if not _ALLOWED_VAULT_PATHS:
            _update_vault_paths()

        path_to_check = path_part + "/"
        if not any(path_to_check.startswith(prefix) for prefix in _ALLOWED_VAULT_PATHS):
            logger.warning(
                f"Access denied for Vault path '{path_part}'. Allowed: {_ALLOWED_VAULT_PATHS}"
            )
            raise ValueError(f"Access to Vault path '{path_part}' is not allowed.")

        client = hvac.Client(url=addr, token=token)
        if not client.is_authenticated():
            raise RuntimeError(f"Vault authentication failed for {addr}.")

        path_segments = path_part.split("/")
        if not path_segments:
            raise ValueError(f"Invalid Vault path format: '{path_part}'")
        mount_point, secret_sub_path = path_segments[0], "/".join(path_segments[1:])

        # Try KV v2 then KV v1
        try:  # KV v2
            resp_v2 = client.secrets.kv.v2.read_secret_version(
                mount_point=mount_point, path=secret_sub_path
            )
            data_v2 = resp_v2["data"]["data"]
            if key_name in data_v2:
                return data_v2[key_name]
            else:
                raise KeyError(f"Key '{key_name}' not found in KV v2 secret '{path_part}'")
        except hvac.exceptions.InvalidPath:
            pass  # Try v1
        except (KeyError, TypeError):
            pass  # Try v1 if structure unexpected
        except Exception as e:
            logger.error(f"Error reading Vault KV v2 '{path_part}': {e}")
            raise RuntimeError(f"Failed to read Vault secret: {e}") from e
        try:  # KV v1
            resp_v1 = client.secrets.kv.v1.read_secret(
                mount_point=mount_point, path=secret_sub_path
            )
            data_v1 = resp_v1["data"]
            if key_name in data_v1:
                return data_v1[key_name]
            else:
                raise KeyError(f"Key '{key_name}' not found in KV v1 secret '{path_part}'")
        except hvac.exceptions.InvalidPath:
            raise KeyError(
                f"Secret path '{path_part}' not found in Vault (tried KV v2 & v1)."
            ) from None
        except KeyError as e:
            raise KeyError(f"Key '{key_name}' not found at '{path_part}' (KV v1).") from e
        except Exception as e:
            logger.error(f"Error reading Vault KV v1 '{path_part}': {e}")
            raise RuntimeError(f"Failed to read Vault secret: {e}") from e

    raise ValueError(f"Unknown secret scheme or invalid path: {path_key}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLAYWRIGHT LIFECYCLE (proxy, VNC, cookies)
# ══════════════════════════════════════════════════════════════════════════════
def _update_proxy_settings():
    """Parse global proxy config strings into usable dict/list."""
    global _PROXY_CONFIG_DICT, _PROXY_ALLOWED_DOMAINS_LIST
    # Parse Proxy Pool String
    _PROXY_CONFIG_DICT = None
    if _proxy_pool_str_global:
        proxies = [p.strip() for p in _proxy_pool_str_global.split(";") if p.strip()]
        if proxies:
            chosen_proxy = random.choice(proxies)
            try:
                parsed = urlparse(chosen_proxy)
                if (
                    parsed.scheme
                    and parsed.netloc
                    and parsed.scheme in ("http", "https", "socks5", "socks5h")
                    and "#" not in chosen_proxy
                ):
                    proxy_dict: Dict[str, Any] = {"server": f"{parsed.scheme}://{parsed.netloc}"}
                    if parsed.username:
                        proxy_dict["username"] = urllib.parse.unquote(parsed.username)
                    if parsed.password:
                        proxy_dict["password"] = urllib.parse.unquote(parsed.password)
                        hostname_port = (
                            f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname
                        )
                        proxy_dict["server"] = f"{parsed.scheme}://{hostname_port}"
                    _PROXY_CONFIG_DICT = proxy_dict
                    logger.info(f"Proxy settings parsed: Using {proxy_dict.get('server')}")
                else:
                    logger.warning(f"Invalid proxy URL format/scheme: '{chosen_proxy}'.")
            except Exception as e:
                logger.warning(f"Error parsing proxy URL '{chosen_proxy}': {e}")
    # Parse Allowed Domains String
    if not _proxy_allowed_domains_str_global or _proxy_allowed_domains_str_global == "*":
        _PROXY_ALLOWED_DOMAINS_LIST = None
    else:
        domains = [
            d.strip().lower() for d in _proxy_allowed_domains_str_global.split(",") if d.strip()
        ]
        _PROXY_ALLOWED_DOMAINS_LIST = [d if d.startswith(".") else "." + d for d in domains]
        logger.info(f"Proxy allowed domains parsed: {_PROXY_ALLOWED_DOMAINS_LIST}")


def _get_proxy_config() -> Optional[Dict[str, Any]]:
    """Returns the globally cached parsed proxy dictionary."""
    # Note: Assumes _update_proxy_settings was called during init
    return _PROXY_CONFIG_DICT


def _is_domain_allowed_for_proxy(url: str) -> bool:
    """Checks if the URL's domain is allowed based on globally cached list."""
    # Note: Assumes _update_proxy_settings was called during init
    if _PROXY_ALLOWED_DOMAINS_LIST is None:
        return True
    try:
        domain = urlparse(url).netloc.lower()
        if not domain:
            return False
        domain_parts = domain.split(".")
        for i in range(len(domain_parts)):
            if ("." + ".".join(domain_parts[i:])) in _PROXY_ALLOWED_DOMAINS_LIST:
                return True
        return False
    except Exception:
        return False  # Deny on error


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        loop.call_soon_threadsafe(asyncio.create_task, coro)


async def _try_close_browser():
    """Attempt to close the browser gracefully via atexit."""
    global _browser
    if _browser and _browser.is_connected():
        logger.info("Attempting to close browser via atexit handler...")
        try:
            await _browser.close()
            logger.info("Browser closed successfully via atexit.")
        except Exception as e:
            logger.error(f"Error closing browser during atexit: {e}")
        finally:
            _browser = None


async def get_browser_context(
    use_incognito: bool = False,
    context_args: Optional[Dict[str, Any]] = None,
) -> tuple[BrowserContext, Browser]:
    """Get or create a browser context using global config values."""
    global _pw, _browser, _ctx
    async with _playwright_lock():
        if not _pw:
            try:
                _pw = await async_playwright().start()
                logger.info("Playwright started.")
            except Exception as e:
                raise RuntimeError(f"Failed to start Playwright: {e}") from e

        is_headless = _headless_mode_global
        if not is_headless:
            _start_vnc()  # Uses globals

        if not _browser or not _browser.is_connected():
            if _browser:
                await _browser.close()  # Ignore errors
            try:
                _browser = await _pw.chromium.launch(
                    headless=is_headless,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--window-size=1280,1024",
                    ],
                )
                logger.info(f"Browser launched (Headless: {is_headless}).")
                atexit.register(lambda: _run_sync(_try_close_browser()))
            except PlaywrightException as e:
                raise RuntimeError(f"Failed to launch browser: {e}") from e

        default_args = {
            "viewport": {"width": 1280, "height": 1024},
            "locale": "en-US",
            "timezone_id": "UTC",
            "accept_downloads": True,
        }
        if context_args:
            default_args.update(context_args)

        if use_incognito:
            try:
                incog_ctx = await _browser.new_context(**default_args)
                await _log("browser_incognito_context", args=default_args)
                if default_args.get("proxy"):
                    await _add_proxy_routing_rule(incog_ctx, default_args["proxy"])
                return incog_ctx, _browser
            except PlaywrightException as e:
                raise ToolError(f"Failed to create incognito context: {e}") from e

        if not _ctx or not _ctx.browser:
            if _ctx:
                await _ctx.close()  # Ignore errors
            try:
                loaded_state = await _load_state()  # Uses globals for key
                proxy_cfg = _get_proxy_config()  # Uses globals
                final_ctx_args = default_args.copy()
                final_ctx_args["storage_state"] = loaded_state
                if proxy_cfg:
                    final_ctx_args["proxy"] = proxy_cfg

                _ctx = await _browser.new_context(**final_ctx_args)
                await _log(
                    "browser_context_create",
                    headless=is_headless,
                    proxy=bool(proxy_cfg),
                    args={k: v for k, v in final_ctx_args.items() if k != "storage_state"},
                )
                if proxy_cfg:
                    await _add_proxy_routing_rule(_ctx, proxy_cfg)
                asyncio.create_task(_context_maintenance_loop(_ctx))
            except PlaywrightException as e:
                raise RuntimeError(f"Failed to create shared context: {e}") from e

        return _ctx, _browser


async def _add_proxy_routing_rule(context: BrowserContext, proxy_config: Dict[str, Any]):
    """Adds routing rule to enforce proxy domain restrictions if enabled."""
    if _PROXY_ALLOWED_DOMAINS_LIST is None:
        return

    async def handle_route(route):
        if not _is_domain_allowed_for_proxy(route.request.url):
            logger.warning(f"Proxy blocked for disallowed domain: {route.request.url}. Aborting.")
            try:
                await route.abort("accessdenied")
            except PlaywrightException as e:
                logger.error(f"Error aborting route: {e}")
        else:
            try:
                await route.continue_()
            except PlaywrightException as e:
                logger.error(f"Error continuing route: {e}")
            # Try abort as fallback? Risky, continue failure might mean connection issues
            # try: await route.abort() except Exception: pass

    try:
        await context.route("**/*", handle_route)
        logger.info("Proxy domain restriction rule added.")
    except PlaywrightException as e:
        logger.error(f"Failed to add proxy routing rule: {e}")


def _start_vnc():
    """Starts X11VNC if VNC enabled and password set."""
    global _vnc_proc
    if _vnc_proc or not _vnc_enabled_global:
        return
    vnc_pass = _vnc_password_global
    if not vnc_pass:  # Warning logged during config load/validation
        logger.debug("VNC start skipped: Password not set.")
        return
    display = os.getenv("DISPLAY", ":0")
    try:
        if subprocess.run(["which", "x11vnc"], capture_output=True).returncode != 0:
            logger.warning("x11vnc command not found. Cannot start VNC server.")
            return
        cmd = [
            "x11vnc",
            "-display",
            display,
            "-passwd",
            vnc_pass,
            "-forever",
            "-localhost",
            "-quiet",
            "-noxdamage",
        ]
        preexec_fn = os.setsid if hasattr(os, "setsid") else None
        _vnc_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=preexec_fn
        )
        logger.info(f"Password-protected VNC server started on display {display} (localhost only).")
        atexit.register(_cleanup_vnc)
    except FileNotFoundError:
        logger.warning("x11vnc command not found.")
    except Exception as e:
        logger.error(f"Failed to start VNC server: {e}", exc_info=True)
        _vnc_proc = None


def _cleanup_vnc():
    """Terminates the VNC server process."""
    global _vnc_proc
    proc = _vnc_proc
    if proc and proc.poll() is None:
        logger.info("Terminating VNC server process...")
        try:
            pgid = os.getpgid(proc.pid) if hasattr(os, "getpgid") else None
            if pgid and hasattr(os, "killpg"):
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=5)
            logger.info("VNC server process terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("VNC server SIGTERM timeout. Sending SIGKILL.")
            if pgid and hasattr(os, "killpg"):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Already dead
            else:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        except ProcessLookupError:
            logger.info("VNC process already terminated.")
        except Exception as e:
            logger.error(f"Error during VNC cleanup: {e}")
        finally:
            _vnc_proc = None

async def _load_state() -> dict[str, Any] | None:
    """Loads browser state asynchronously. Decryption runs in executor if needed."""
    if not _STATE_FILE.exists():
        return None
    loop = asyncio.get_running_loop()
    try:
        async with aiofiles.open(_STATE_FILE, "rb") as f:
            file_data = await f.read()

        # Determine if decryption is needed based on key existence
        k = _key()
        if not k:
            # No key, assume plaintext - _dec handles basic validation
            decrypted_data = await loop.run_in_executor(_get_pool(), _dec, file_data)
        else:
            # Key exists, attempt decryption via _dec
            decrypted_data = await loop.run_in_executor(_get_pool(), _dec, file_data)

        if decrypted_data is None:
            logger.warning("Failed to load or decrypt state data.")
            return None # Decryption/validation failed
        return json.loads(decrypted_data)
    except FileNotFoundError:
        logger.info("Browser state file not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse browser state JSON: {e}. Removing corrupt file.")
        _STATE_FILE.unlink(missing_ok=True)
        return None
    except Exception as e: # Catches errors from _dec like RuntimeError('State-file authentication failed')
        logger.error(f"Failed to load browser state: {e}", exc_info=True)
        _STATE_FILE.unlink(missing_ok=True)
        return None


async def _save_state(ctx: BrowserContext):
    """
    Saves browser state asynchronously using the MCP FileSystemTool.
    Encryption runs in executor if key is set.
    """
    if not ctx or not ctx.browser:
        logger.debug("Skipping save state: Invalid context or browser.")
        return
    loop = asyncio.get_running_loop()
    validated_fpath = None # Keep track for logging
    try:
        state = await ctx.storage_state()
        state_json = json.dumps(state).encode("utf-8")

        # Encryption is conditional within _enc based on _key()
        data_to_write = await loop.run_in_executor(
            _get_pool(), _enc, state_json
        )

        # --- Use FileSystemTool to write ---
        # Determine path to use (relative or absolute based on FileSystemTool needs)
        # Assuming FileSystemTool can handle absolute paths and validate them:
        file_path_to_write = str(_STATE_FILE)
        validated_fpath = file_path_to_write # Store for logging

        logger.debug(f"Attempting to save state to: {file_path_to_write} using filesystem tool.")

        # Call the filesystem write tool
        write_result = await write_file(
            path=file_path_to_write,
            content=data_to_write, # Pass bytes content
            # overwrite=True # Explicitly allow overwrite if needed
        )

        # Check the result from the filesystem tool
        if not isinstance(write_result, dict) or not write_result.get("success"):
            error_detail = write_result.get('error', 'Unknown error') if isinstance(write_result, dict) else 'Invalid response'
            logger.error(f"Failed to save browser state using filesystem tool. Reason: {error_detail}")
            raise ToolError(f"Failed to save browser state: {error_detail}")

        # Success log now includes path confirmed by tool
        actual_path = write_result.get("path", file_path_to_write)
        logger.debug(f"Browser state saved ({'encrypted' if _key() else 'unencrypted'}) to {actual_path} via filesystem tool.")
        # --- End FileSystemTool usage ---

    except PlaywrightException as e:
        logger.error(f"Failed to get storage state from Playwright: {e}")
    except ToolError as e: # Catch errors from write_file
         logger.error(f"Filesystem tool error saving state to {validated_fpath}: {e}", exc_info=True)
    except Exception as e: # Includes RuntimeError from _enc if encryption itself fails
        logger.error(f"Failed to save browser state (path: {validated_fpath}): {e}", exc_info=True)


@asynccontextmanager
async def _tab_context(ctx: BrowserContext):
    """Async context manager for creating and cleaning up a Page."""
    page = None
    try:
        page = await ctx.new_page()
        await _log("page_open", context_id=id(ctx))
        yield page
    except PlaywrightException as e:
        raise ToolError(f"Failed to create browser page: {e}") from e
    finally:
        if page and not page.is_closed():
            try:
                await page.close()
                await _log("page_close", context_id=id(ctx))
            except PlaywrightException as e:
                logger.warning(f"Error closing page {id(ctx)}: {e}")


async def _context_maintenance_loop(ctx: BrowserContext):
    """Periodically saves state for the shared context."""
    save_interval_seconds = 15 * 60
    logger.info(f"Starting context maintenance loop for context {id(ctx)}.")
    while True:
        if not ctx or not ctx.browser:
            logger.info(f"Context {id(ctx)} invalid. Stopping maintenance.")
            break
        try:
            await asyncio.sleep(save_interval_seconds)
            if not ctx or not ctx.browser:
                break
            await _save_state(ctx)
        except asyncio.CancelledError:
            logger.info(f"Context maintenance loop {id(ctx)} cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in context maintenance loop: {e}", exc_info=True)
            await asyncio.sleep(60)


async def shutdown():
    """Gracefully shut down Playwright, browser, context, VNC, and thread pool."""
    global _pw, _browser, _ctx, _vnc_proc, _thread_pool, _locator_cache_cleanup_task_handle
    logger.info("Initiating graceful shutdown...")

    # 1. Cancel background tasks
    if _locator_cache_cleanup_task_handle and not _locator_cache_cleanup_task_handle.done():
        logger.info("Cancelling locator cache cleanup task...")
        _locator_cache_cleanup_task_handle.cancel()
        try:
            await asyncio.wait_for(_locator_cache_cleanup_task_handle, timeout=2.0)
        except asyncio.CancelledError:
            logger.info("Locator cache cleanup task cancelled as expected during shutdown.") # Specific message
        except Exception as e: # Catch other errors like TimeoutError or custom exceptions
            logger.warning(f"Locator cache cleanup task cancellation timeout/error: {type(e).__name__}")
        finally: # Ensure handle is cleared even on error
            _locator_cache_cleanup_task_handle = None

    await tab_pool.cancel_all()  # Cancel tab pool tasks

    # 2. Close Playwright resources under lock
    async with _playwright_lock():
        ctx_to_close = _ctx
        _ctx = None
        if ctx_to_close and ctx_to_close.browser:
            try:
                await _save_state(ctx_to_close)
                await ctx_to_close.close()
                await _log("browser_context_close_shared")
            except Exception as e:
                logger.error(f"Error closing shared context: {e}")
        browser_to_close = _browser
        _browser = None
        if browser_to_close and browser_to_close.is_connected():
            try:
                await browser_to_close.close()
                await _log("browser_close")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
        pw_to_stop = _pw
        _pw = None
        if pw_to_stop:
            try:
                await pw_to_stop.stop()
                logger.info("Playwright stopped.")
            except Exception as e:
                logger.error(f"Error stopping Playwright: {e}")

    # 3. Cleanup Sync Resources
    _cleanup_vnc()
    logger.info("Shutting down thread pool...")
    _get_pool().shutdown(wait=True)
    logger.info("Thread pool shut down.")

    await _log("browser_shutdown_complete")
    logger.info("Graceful shutdown complete.")


_shutdown_initiated = False
_shutdown_lock = asyncio.Lock()


async def _initiate_shutdown():
    """Ensures shutdown runs only once."""
    global _shutdown_initiated
    async with _shutdown_lock:
        if not _shutdown_initiated:
            _shutdown_initiated = True
            await shutdown()


def _signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {sig}. Initiating shutdown...")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(asyncio.create_task, _initiate_shutdown())
        else:
            asyncio.run(_initiate_shutdown())
    except RuntimeError as e:
        logger.error(f"Error scheduling shutdown from signal: {e}. Attempting sync.")
    # Removed redundant asyncio.run call that could cause problems
    # try:
    #     asyncio.run(_initiate_shutdown())
    # except Exception as final_e:
    #     logger.critical(f"Sync shutdown attempt failed: {final_e}")


try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except ValueError:
    logger.warning("Could not register signal handlers (not main thread?).")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TAB POOL FOR PARALLELISM
# ══════════════════════════════════════════════════════════════════════════════
class TabPool:
    """Runs async callables needing a Page in parallel, bounded by global config."""

    def __init__(self, max_tabs: int | None = None):
        # Use global config value, allow override via argument
        self.max_tabs = max_tabs if max_tabs is not None else _sb_max_tabs_global
        if self.max_tabs <= 0:
            logger.warning(f"Invalid max_tabs ({self.max_tabs}), defaulting to 1.")
            self.max_tabs = 1
        self.sem = asyncio.Semaphore(self.max_tabs)
        self._active_contexts: Set[BrowserContext] = set()
        self._context_lock = asyncio.Lock()
        logger.info(f"TabPool initialized with max_tabs={self.max_tabs}")

    async def _run(self, fn: Callable[[Page], Awaitable[Any]]) -> Any:
        """Acquires semaphore, creates incognito context+page, runs fn, cleans up."""
        # Use global config value for timeout
        timeout_seconds = _sb_tab_timeout_global
        incognito_ctx: Optional[BrowserContext] = None
        task = asyncio.current_task()  # Get task for logging correlation

        try:
            # Wrap the core logic in wait_for
            async with self.sem:  # Acquire semaphore before creating resources
                # Use incognito context for isolation
                # Pass proxy config from global context if it exists? For now, incognito starts clean.
                incognito_ctx, _ = await get_browser_context(use_incognito=True)

                # Track the active context for potential cleanup on cancellation
                async with self._context_lock:
                    self._active_contexts.add(incognito_ctx)

                # Context manager handles page creation and closure
                async with _tab_context(incognito_ctx) as page:
                    # Execute the provided function with the page
                    result = await asyncio.wait_for(fn(page), timeout=timeout_seconds)
                    return result

        except asyncio.TimeoutError:
            func_name = getattr(fn, "__name__", "anon_tab_fn")
            await _log("tab_timeout", function=func_name, timeout=timeout_seconds, task_id=id(task))
            return {"error": f"Tab operation timed out after {timeout_seconds}s", "success": False}
        except asyncio.CancelledError:
            func_name = getattr(fn, "__name__", "anon_tab_fn")
            await _log("tab_cancelled", function=func_name, task_id=id(task))
            raise  # Propagate cancellation
        except Exception as e:
            func_name = getattr(fn, "__name__", "anon_tab_fn")
            await _log(
                "tab_error", function=func_name, error=str(e), task_id=id(task), exc_info=True
            )  # Log traceback
            return {"error": f"Tab operation failed: {e}", "success": False}
        finally:
            # Ensure the incognito context is *always* closed
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
        tasks = [asyncio.create_task(self._run(fn)) for fn in fns]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                func_name = getattr(fns[i], "__name__", f"fn_{i}")
                logger.error(f"Error in TabPool.map for '{func_name}': {res}", exc_info=res)
                processed_results.append({"error": f"Task failed: {res}", "success": False})
            else:
                processed_results.append(res)
        return processed_results

    async def cancel_all(self):
        """Attempts to close all active contexts managed by the pool."""
        contexts_to_close: List[BrowserContext] = []
        async with self._context_lock:
            contexts_to_close = list(self._active_contexts)
            self._active_contexts.clear()
        if not contexts_to_close:
            return
        logger.info(f"TabPool cancel_all: Closing {len(contexts_to_close)} active contexts.")
        close_tasks = [asyncio.create_task(ctx.close()) for ctx in contexts_to_close]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        errors = sum(1 for res in results if isinstance(res, Exception))
        if errors:
            logger.warning(f"TabPool cancel_all: {errors} errors closing contexts.")


tab_pool = TabPool()  # Instantiate the global instance


# ══════════════════════════════════════════════════════════════════════════════
# 6.  HUMAN-LIKE JITTER (bot evasion)
# ══════════════════════════════════════════════════════════════════════════════


def _risk_factor(url: str) -> float:
    """Calculates risk factor based on URL's domain using global config set."""
    # Uses _high_risk_domains_set_global
    if not url:
        return 1.0
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if not domain:
            return 1.0
        # Check if domain itself or parent domain is high risk
        domain_parts = domain.split(".")
        for i in range(len(domain_parts)):
            sub_domain = "." + ".".join(domain_parts[i:])
            if sub_domain in _high_risk_domains_set_global:  # Use global
                # logger.debug(f"High risk domain detected: {domain} (matched {sub_domain})")
                return 2.0  # Higher factor
        return 1.0
    except Exception:
        return 1.0  # Default on error


async def _pause(page: Page, base_ms_range: tuple[int, int] = (150, 500)):
    """Introduce a short, randomized pause, adjusted by URL risk factor."""
    if not page or page.is_closed():
        return
    risk = _risk_factor(page.url)  # Uses global config indirectly
    min_ms, max_ms = base_ms_range
    base_delay_ms = random.uniform(min_ms, max_ms)
    adjusted_delay_ms = base_delay_ms * risk
    try:  # Optional complexity factor based on interactive elements
        element_count = await page.evaluate(
            "() => document.querySelectorAll('a, button, input, select, textarea, [role=button], [role=link], [onclick]').length"
        )
        # Fix for SPAs (like Google) that return 0 elements due to shadow DOM
        element_count = max(element_count, 100) if element_count == 0 else element_count

        # Performance optimization: skip sleep on low-risk sites with few elements
        if risk == 1.0 and element_count < 50:
            return

        complexity_factor = min(1.0 + (element_count / 500.0), 1.5)
        adjusted_delay_ms *= complexity_factor
    except PlaywrightException:
        pass  # Ignore evaluation errors
    final_delay_ms = min(adjusted_delay_ms, 3000)  # Max 3 seconds
    await asyncio.sleep(final_delay_ms / 1000.0)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENHANCED LOCATOR HELPERS (Readability, Page Map, Heuristics)
# ══════════════════════════════════════════════════════════════════════════════

_READ_JS_WRAPPER = textwrap.dedent("""
    (html) => {
        const R = window.__sbReadability;
        if (!R || !html) return "";
        try {
            const doc = new DOMParser().parseFromString(html, "text/html");
            if (!doc || !doc.body || doc.body.innerHTML.trim() === '') return ""; // Robustness check
            const art = new R.Readability(doc).parse();
            return art ? art.textContent : "";
        } catch (e) { console.warn('Readability parsing failed:', e); return ""; }
    }
""")



async def _ensure_readability(page: Page) -> None:
    """
    Ensures Mozilla's Readability.js library is injected into the page,
    using the MCP FileSystemTool for caching.
    """
    if await page.evaluate("() => window.__sbReadability !== undefined"):
        return

    # --- Use FileSystemTool for Cache Read/Write ---
    # Assuming FileSystemTool can handle absolute path _READ_JS_CACHE
    cache_file_path = str(_READ_JS_CACHE)
    src: Optional[str] = None

    try:
        # Try reading from cache using filesystem tool
        try:
            read_result = await read_file(path=cache_file_path)
            if isinstance(read_result, dict) and read_result.get("success"):
                 # Extract content - assuming read_file result structure includes 'content' key
                 # or modifies its own response structure for create_tool_response
                 # Let's assume create_tool_response was used by read_file and content is list
                 if isinstance(read_result.get("content"), list) and len(read_result["content"]) > 0:
                     content_block = read_result["content"][0]
                     if content_block.get("type") == "text":
                          src = content_block.get("text")
                          logger.debug(f"Readability.js loaded from cache: {cache_file_path}")
                     else:
                          logger.warning(f"Cache file {cache_file_path} content block type is not 'text'.")
                 else:
                      logger.warning(f"Cache file {cache_file_path} read successfully but content format unexpected or empty.")
            # else: read_file failed, error logged by its decorator, src remains None

        except ToolError as e:
             # Handle specific "file not found" case gracefully
             error_code = getattr(e, 'error_code', '')
             error_details = getattr(e, 'details', {})
             # Check error code or details if FileSystemTool provides specific not found info
             is_not_found = (error_code == "PATH_NOT_FOUND" or
                             (isinstance(error_details, dict) and error_details.get("error_type") == "PATH_NOT_FOUND") or
                             "does not exist" in str(e).lower() or
                             "no such file" in str(e).lower())

             if is_not_found:
                 logger.info(f"Readability.js cache file not found ({cache_file_path}). Fetching from CDN.")
                 src = None # Explicitly ensure src is None
             else:
                 # Log other errors reading cache but proceed to fetch
                 logger.warning(f"Error reading Readability.js cache file {cache_file_path}: {e}. Will attempt fetch.")
                 src = None # Explicitly ensure src is None
        # If src is still None after cache attempt, fetch from CDN
        if src is None:
            logger.info("Fetching Readability.js from CDN (cache miss or error)...")
            try:
                async with httpx.AsyncClient() as client:
                    cdn_url = "https://cdnjs.cloudflare.com/ajax/libs/readability/0.5.0/Readability.js"
                    response = await client.get(cdn_url, timeout=15.0)
                    response.raise_for_status()
                    fetched_src = response.text
                    await _log("readability_js_fetch", url=cdn_url, size=len(fetched_src))

                if fetched_src:
                    # Write fetched content to cache using filesystem tool
                    try:
                        write_cache_result = await write_file(
                            path=cache_file_path,
                            content=fetched_src # Pass string content
                        )
                        if isinstance(write_cache_result, dict) and write_cache_result.get("success"):
                            logger.info(f"Saved Readability.js to cache: {cache_file_path}")
                            src = fetched_src # Use fetched content
                        else:
                            error_detail = write_cache_result.get('error', 'Unknown error') if isinstance(write_cache_result, dict) else 'Invalid response'
                            logger.warning(f"Failed to write Readability.js cache ({cache_file_path}): {error_detail}")
                            src = fetched_src # Still use fetched content even if cache write fails
                    except ToolError as write_err:
                         logger.warning(f"Filesystem tool error writing Readability.js cache ({cache_file_path}): {write_err}")
                         src = fetched_src # Still use fetched content
                    except Exception as write_unexpected:
                         logger.error(f"Unexpected error writing Readability.js cache: {write_unexpected}", exc_info=True)
                         src = fetched_src # Still use fetched content

                else:
                    logger.warning("Fetched empty content for Readability.js from CDN.")

            except httpx.RequestError as http_err:
                 logger.error(f"Failed to fetch Readability.js from CDN: {http_err}")
            except Exception as fetch_err:
                 logger.error(f"Unexpected error fetching/caching Readability.js: {fetch_err}", exc_info=True)

        # --- Script Injection Logic ---
        if src:
            wrapped_src = f"window.__sbReadability = (() => {{ {src}; return Readability; }})();"
            try:
                await page.add_script_tag(content=wrapped_src)
                logger.debug("Readability.js injected successfully.")
            except PlaywrightException as e:
                # ... (keep existing CSP/injection error handling) ...
                if "Content Security Policy" in str(e):
                    logger.warning(f"Could not inject Readability.js due to CSP on {page.url}. Proceeding without it.")
                else:
                    logger.error(f"Failed to inject Readability.js: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error during Readability.js injection: {e}", exc_info=True)
        else:
            logger.warning("Failed to load or fetch Readability.js source. Proceeding without it.")
        # --- End Script Injection Logic ---

    except Exception as e: # Catch errors in the outer try block
        logger.error(f"Unexpected error in _ensure_readability setup: {e}", exc_info=True)


async def _dom_fingerprint(page: Page) -> str:
    """Calculates a fingerprint of the page's visible text content using global config."""
    try:
        # Use global limit
        txt = await page.main_frame.evaluate(
            f"() => document.body.innerText.slice(0, {_dom_fp_limit_global})"
        )
        txt = (txt or "").strip()
        # Hashing is fast enough for this size, no need for thread pool
        return hashlib.sha256(txt.encode("utf-8", "ignore")).hexdigest()
    except PlaywrightException as e:
        logger.warning(f"Could not get text for DOM fingerprint: {e}")
        return hashlib.sha256(b"").hexdigest()  # Empty fingerprint


def _shadow_deep_js() -> str:  # Removed args, uses globals now
    """JS function string to find elements, traversing shadow DOM, using global config."""
    # Uses _max_widgets_global, _area_min_global
    return f"""
    (prefix) => {{
        const MAX = {_max_widgets_global};
        const MIN_AREA = {_area_min_global};
        
        const isVis = el => {{
            if (!el || !el.getBoundingClientRect) return false;
            try {{
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || 
                    style.opacity === '0' || el.hidden || !el.offsetParent) {{
                    return false;
                }}
                
                const rect = el.getBoundingClientRect();
                const hasSize = rect.width > 1 && rect.height > 1;
                const hasArea = rect.width * rect.height >= MIN_AREA;
                const isOnscreen = rect.bottom > 0 && rect.top < window.innerHeight && 
                                  rect.right > 0 && rect.left < window.innerWidth;
                
                return hasSize && (hasArea || el.tagName === 'A') && isOnscreen;
            }} catch (e) {{ 
                console.warn('Error in isVis:', e);
                return false; 
            }}
        }};
        
        const okTagRole = el => {{
            const tag = el.tagName.toLowerCase();
            const role = (el.getAttribute('role') || '').toLowerCase();
            
            const interactiveTags = ['a', 'button', 'input', 'select', 'textarea', 'option',
                                    'label', 'form', 'fieldset', 'details', 'summary',
                                    'dialog', 'menu', 'menuitem'];
            const interactiveRoles = ['button', 'link', 'checkbox', 'radio', 'menuitem', 'tab',
                                     'switch', 'option', 'searchbox', 'textbox', 'dialog'];
                                      
            // Interactive elements are always interesting
            if (interactiveTags.includes(tag) || interactiveRoles.includes(role)) return true;
            
            // Elements with specific attributes indicating interactivity
            if (el.onclick || el.href || el.getAttribute('tabindex') !== null ||
                el.getAttribute('contenteditable') === 'true') return true;
            
            // Non-empty container elements that have decent size
            if ((tag === 'div' || tag === 'section' || tag === 'span') && 
                el.innerText && el.innerText.trim().length > 0) {{
                const rect = el.getBoundingClientRect();
                if (rect.width * rect.height >= MIN_AREA) return true;
            }}
            
            // Images with good size are interesting
            if (tag === 'img' && el.alt) {{
                const rect = el.getBoundingClientRect();
                if (rect.width * rect.height >= MIN_AREA) return true;
            }}
            
            return false;
        }};
        
        const getText = el => {{
            try {{
                // For form elements, get value or placeholder
                if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {{
                    if (el.type === 'button' || el.type === 'submit') return el.value || '';
                    if (el.type === 'password') return 'Password field';
                    return el.placeholder || el.name || el.type || ''; 
                }}
                
                if (el.tagName === 'SELECT') return el.name || '';
                
                // For images, use alt text
                if (el.tagName === 'IMG') return el.alt || '';
                
                // Handle aria labels
                const ariaLabel = el.getAttribute('aria-label');
                if (ariaLabel) return ariaLabel;
                
                // Check related labels for inputs
                if (el.id && el.tagName === 'INPUT') {{
                    const labels = document.querySelectorAll(`label[for="${{el.id}}"]`);
                    if (labels.length > 0) return labels[0].textContent.trim();
                }}
                
                // Get text content without descendant script/style text
                let textContent = '';
                for (const node of el.childNodes) {{
                    if (node.nodeType === Node.TEXT_NODE) {{
                        textContent += node.textContent;
                    }}
                }}
                
                return textContent.trim() || el.innerText.trim() || '';
            }} catch (e) {{
                console.warn('Error in getText:', e);
                return '';
            }}
        }};
        
        const out = [];
        const q = [document.documentElement];
        const visited = new Set();
        let idx = 0;
        while (q.length > 0 && out.length < MAX) {{
            const node = q.shift();
            if (!node || visited.has(node)) continue;
            visited.add(node);
            if (okTagRole(node) && isVis(node)) {{
                try {{
                    const r = node.getBoundingClientRect();
                    const id = `${{prefix || ''}}el_${{idx++}}`;
                    node.dataset.sbId = id; // Add ID to element
                    out.push({{
                        id, tag: node.tagName.toLowerCase(), role: node.getAttribute("role") || "",
                        text: getText(node),
                        bbox: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]
                    }});
                }} catch (e) {{ console.warn('Error processing element:', node, e); }}
            }}
            const children = node.shadowRoot ? node.shadowRoot.children : node.children;
            if (children) {{
                for (let i = 0; i < children.length; i++) {{ if (!visited.has(children[i])) q.push(children[i]); }}
            }}
            if (node.tagName === 'IFRAME' && node.contentDocument && node.contentDocument.documentElement) {{
                 if (!visited.has(node.contentDocument.documentElement)) q.push(node.contentDocument.documentElement);
            }}
        }}
        return out;
    }}
    """


async def _build_page_map(page: Page) -> Tuple[Dict[str, Any], str]:
    """Builds a map of the page using global config values."""
    # Check if page already has a cached map with a valid fingerprint
    fp = await _dom_fingerprint(page)  # Calculate fingerprint first
    if hasattr(page, "_sb_page_map") and hasattr(page, "_sb_fp") and page._sb_fp == fp:
        return page._sb_page_map, fp

    await _ensure_readability(page)
    main_txt = ""
    elems: List[Dict[str, Any]] = []
    page_title = "[Error]"

    try:
        # Get main text
        html_content = await page.content()
        if html_content:
            main_txt = await page.evaluate(_READ_JS_WRAPPER, html_content) or ""
            if len(main_txt) < 200:  # Fallback

                def extract_basic_text(html):
                    soup = BeautifulSoup(html[:3_000_000], "lxml")
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    return soup.get_text(" ", strip=True)

                main_txt = await asyncio.get_running_loop().run_in_executor(
                    _get_pool(), extract_basic_text, html_content
                )
            main_txt = main_txt[:_max_section_chars_global]  # Use global limit
        else:
            logger.warning("Failed to get HTML content for page map.")

        # Extract elements
        js_func = _shadow_deep_js()  # Uses globals internally
        all_elems = []
        frame: Frame
        for i, frame in enumerate(page.frames):
            if frame.is_detached():
                continue
            try:
                frame_elems = await asyncio.wait_for(frame.evaluate(js_func, f"f{i}:"), timeout=5.0)
                all_elems.extend(frame_elems)
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout eval elements in frame {i} ({frame.url})")
            except PlaywrightException as e:
                logger.warning(f"Error eval elements in frame {i} ({frame.url}): {e}")
        elems = all_elems[:_max_widgets_global]  # Use global limit

        # Get title
        try:
            page_title = await page.title()
        except PlaywrightException:
            pass

    except PlaywrightException as e:
        logger.error(f"Could not build page map for {page.url}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error building page map: {e}", exc_info=True)

    page_map = {"url": page.url, "title": page_title, "main_text": main_txt, "elements": elems}

    # Cache the page map and fingerprint on the page object
    page._sb_page_map = page_map
    page._sb_fp = fp

    return (page_map, fp)


# Heuristic Matcher
_SM_GLOBAL = difflib.SequenceMatcher(
    autojunk=False
)  # Disable autojunk for potentially better short string matching


def _ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    _SM_GLOBAL.set_seqs(a, b)
    return _SM_GLOBAL.ratio()


def _heuristic_pick(pm: Dict[str, Any], hint: str, role: Optional[str]) -> Optional[str]:
    """Finds the best element ID based on text similarity and heuristics using global config."""
    # Uses _seq_cutoff_global
    if not hint or not pm or not pm.get("elements"):
        return None
    h_norm = unicodedata.normalize("NFC", hint).lower()
    best_id, best_score = None, -1.0
    for e in pm["elements"]:
        if not e or not isinstance(e, dict):
            continue
        el_id, el_text, el_role, el_tag = (
            e.get("id"),
            e.get("text", ""),
            e.get("role", ""),
            e.get("tag", ""),
        )
        if not el_id:
            continue
        if (
            role
            and role.lower() != el_role.lower()
            and not (role.lower() == "button" and el_tag.lower() == "button")
        ):
            continue  # Role filter

        score = _ratio(h_norm, unicodedata.normalize("NFC", el_text).lower())
        # Heuristic adjustments (same logic as before)
        if role and role.lower() == el_role.lower():
            score += 0.1
        hint_keywords = {
            "button",
            "submit",
            "link",
            "input",
            "download",
            "checkbox",
            "radio",
            "tab",
            "menu",
        }
        element_keywords = {el_role.lower(), el_tag.lower()}
        common_keywords = hint_keywords.intersection(
            w for w in h_norm.split() if w in hint_keywords
        )
        if common_keywords.intersection(element_keywords):
            score += 0.15
        if ("label for" in h_norm or "placeholder" in h_norm) and score > 0.6:
            score += 0.1
        if len(el_text) < 5 and len(h_norm) > 10:
            score -= 0.1
        if el_tag in ("div", "span") and not el_role:
            score -= 0.05

        if score > best_score:
            best_id, best_score = el_id, score

    # logger.debug(f"Heuristic pick: Hint='{hint}', Best ID='{best_id}', Score={best_score:.2f} (Cutoff: {_seq_cutoff_global})")
    return best_id if best_score >= _seq_cutoff_global else None  # Use global cutoff


# LLM Picker Helper
async def _llm_pick(pm: Dict[str, Any], task_hint: str, attempt: int) -> Optional[str]:
    """Asks the LLM to pick the best element ID based on page map and hint using global config."""
    # Uses _llm_model_locator_global
    if not pm or not task_hint:
        return None
    elements_summary = [
        f"id={el.get('id')} tag={el.get('tag')} role={el.get('role', ' ')} text='{el.get('text', ' ')}'"
        for el in pm.get("elements", [])
    ]
    system_prompt = textwrap.dedent("""
        You are an expert UI element selector. Based on the provided PAGE_MAP (URL, title, main_text, elements_summary)
        and the user's TASK_HINT, identify the single best element 'id' from the 'elements_summary' list that matches the hint.
        Focus on interactive elements (buttons, links, inputs) unless the hint clearly indicates otherwise.
        Consider the element's text, role, tag, and context provided by the main_text.

        Respond ONLY with a valid JSON object containing the chosen element ID, like this example:
        {"id": "f0:el_17"}

        If no suitable element is found or the choice is ambiguous, respond with:
        {"id": null}

        Do NOT include explanations, reasoning, or markdown formatting. Just the JSON object.
    """).strip()

    # User prompt with page context and task
    user_prompt = textwrap.dedent(f"""
        PAGE_MAP:
        url: {pm.get("url", "N/A")}
        title: {pm.get("title", "N/A")}
        main_text (summary): "{pm.get("main_text", "")[:1000]}..."
        elements_summary:
        {chr(10).join(elements_summary)}

        TASK_HINT: "{task_hint}"

        Attempt: {attempt}
        Choose the best element 'id' based *only* on the information provided above. Return JSON: {{"id": "..."}} or {{"id": null}}.
    """).strip()
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    res = await _call_llm(
        msgs, model=_llm_model_locator_global, expect_json=True, temperature=0.0, max_tokens=100
    )  # Use global model
    if isinstance(res, dict) and "id" in res:
        el_id = res.get("id")
        if el_id is None or (isinstance(el_id, str) and re.match(r"^(?:f\d+:)?el_\d+$", el_id)):
            return el_id
        else:
            logger.warning(f"LLM returned invalid ID format: {el_id} for hint '{task_hint}'")
            return None
    elif isinstance(res, dict) and "error" in res:
        logger.warning(f"LLM picker failed for hint '{task_hint}': {res['error']}")
        return None
    else:
        logger.warning(f"LLM picker returned unexpected format: {type(res)} for hint '{task_hint}'")
        return None


# Helper to get Locator from sb-id
async def _loc_from_id(page: Page, el_id: str) -> Locator:
    """Gets a Playwright Locator object from a data-sb-id attribute."""
    # No direct config dependencies
    if not el_id:
        raise ValueError("Element ID cannot be empty")

    # Properly escape the ID for CSS attribute selector
    # This handles IDs containing quotes or other special characters
    escaped_id = el_id.replace("\\", "\\\\").replace('"', '\\"')
    selector = f'[data-sb-id="{escaped_id}"]'

    if ":" in el_id and el_id.startswith("f"):
        try:
            frame_index = int(el_id.split(":", 1)[0][1:])
            if 0 <= frame_index < len(page.frames):
                return page.frames[frame_index].locator(selector).first
            else:
                logger.warning(
                    f"Frame index {frame_index} out of bounds for ID {el_id}. Falling back."
                )
        except (ValueError, IndexError):
            logger.warning(f"Could not parse frame index from ID {el_id}. Falling back.")
    return page.locator(selector).first


# ══════════════════════════════════════════════════════════════════════════════
# 7b. ENHANCED LOCATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════
class EnhancedLocator:
    """
    Unified locator using cache, heuristics, and LLM fallback with global config.
    Relies on task hints and page structure analysis rather than explicit selectors.
    """

    def __init__(self, page: Page):
        self.page: Page = page
        self.site: str = "unknown"
        try:
            parsed_url = urlparse(page.url or "")
            # Normalize site name (remove www.) for consistency
            self.site = parsed_url.netloc.lower().replace("www.", "")
            if not self.site:
                self.site = "unknown"  # Fallback if parsing fails
        except Exception:
            pass  # Ignore URL parsing errors

        # Internal state for page map caching within a single locate call
        self._pm: Dict[str, Any] | None = None
        self._pm_fp: str | None = None
        self._last_idle_check: float = 0.0  # Track last network idle check time

    async def _maybe_wait_for_idle(self, timeout: float = 1.5):
        """
        Waits for network idle state, throttled to avoid excessive waits.
        Useful before building page map to ensure dynamic content has loaded.
        """
        now = time.monotonic()
        # Only check network idle if more than ~1 second has passed since last check
        if now - self._last_idle_check > 1.0:
            try:
                await self.page.wait_for_load_state("networkidle", timeout=int(timeout * 1000))
                self._last_idle_check = time.monotonic()  # Update time on success
            except PlaywrightException:
                # Ignore timeout or other errors, proceed anyway
                self._last_idle_check = (
                    time.monotonic()
                )  # Still update time to prevent immediate re-check

    async def _get_page_map(self) -> Tuple[Dict[str, Any], str]:
        """
        Gets or refreshes the internal page map and DOM fingerprint.
        Calls the global _build_page_map helper which uses global config values.
        """
        # Wait briefly for network/rendering to settle before mapping
        await self._maybe_wait_for_idle()
        await asyncio.sleep(random.uniform(0.1, 0.25)) # Added small sleep

        # Call the global helper function which uses global config internally
        pm, fp = await _build_page_map(self.page)
        self._pm, self._pm_fp = pm, fp  # Cache within the instance for this locate() call
        return pm, fp

    async def _selector_cached(self, key: str, fp: str) -> Optional[Locator]:
        """
        Checks the locator cache for a valid selector matching the key and fingerprint.
        Runs synchronous DB operations in the thread pool. Deletes stale entries.
        """
        loop = asyncio.get_running_loop()
        # _cache_get_sync already handles fingerprint check and deletion of stale entries
        sel = await loop.run_in_executor(_get_pool(), _cache_get_sync, key, fp)
        if sel:
            # Found in cache with matching fingerprint, try to locate and verify visibility
            try:
                # Cache stores the selector string (e.g., '[data-sb-id="..."]')
                loc = await _loc_from_id(
                    self.page, sel.split('"')[1]
                )  # Extract ID for _loc_from_id
                # Use a very short timeout for cache hit verification
                await loc.wait_for(state="visible", timeout=500)
                await _log("locator_cache_hit", selector=sel, key=key[:8])
                # No need to bump hits here, _cache_put_sync does it on conflict
                return loc
            except (PlaywrightException, ValueError):  # Catch locator errors or ID parsing errors
                logger.debug(
                    f"Cached selector '{sel}' failed visibility/location check. Cache invalid."
                )
                # Deletion is now handled within _cache_get_sync on mismatch, but we can delete here too if needed
                # await loop.run_in_executor(_get_pool(), _cache_delete_sync, key)
        return None

    async def locate(
        self,
        task_hint: str,
        *,
        role: Optional[str] = None,  # Optional context for heuristic filtering
        timeout: int = 5000,  # Overall timeout for finding a visible element
    ) -> Locator:
        """
        Finds a visible element using cache, heuristics, and LLM fallback.
        Uses globally configured settings for timeouts, models, retries etc.

        Args:
            task_hint: Natural language description of the element/action.
            role: Optional role hint for heuristic filtering.
            timeout: Max milliseconds to wait for element visibility across all tiers.

        Returns:
            A visible Playwright Locator object.

        Raises:
            ValueError: If task_hint is empty.
            PlaywrightTimeoutError: If no suitable element is found and made visible within timeout.
        """
        if not task_hint or not task_hint.strip():
            raise ValueError("task_hint cannot be empty for EnhancedLocator")

        start_time = time.monotonic()
        timeout_sec = timeout / 1000.0

        # Generate cache key based on site, path (no query), and hint
        path = urlparse(self.page.url or "").path or "/"
        key_src = json.dumps(
            {"site": self.site, "path": path, "hint": task_hint.lower()}, sort_keys=True
        )
        cache_key = hashlib.sha256(key_src.encode()).hexdigest()

        loop = asyncio.get_running_loop()

        # Get initial fingerprint without building full map yet
        current_dom_fp = await _dom_fingerprint(self.page)

        # Tier 0: Cache Check
        # _selector_cached handles fingerprint check and potential deletion
        cached_loc = await self._selector_cached(cache_key, current_dom_fp)
        if cached_loc:
            return cached_loc  # Cache hit, return immediately

        # --- Cache miss or stale entry, proceed to active finding ---

        # Tier 1: Heuristic Match (using Page Map)
        # Get fresh map & fingerprint (only if cache missed)
        pm, current_dom_fp = await self._get_page_map()
        # Call global helper which uses global config (_seq_cutoff_global)
        heuristic_id = _heuristic_pick(pm, task_hint, role)
        if heuristic_id:
            try:
                loc = await _loc_from_id(self.page, heuristic_id)
                # Ensure element is in view before checking visibility
                await loc.scroll_into_view_if_needed()
                # Use a fraction of the total timeout for this tier's check
                await loc.wait_for(state="visible", timeout=max(1000, timeout // 3))
                # Success: Cache the found selector ([data-sb-id="..."])
                selector_str = f'[data-sb-id="{heuristic_id}"]'
                await loop.run_in_executor(
                    _get_pool(), _cache_put_sync, cache_key, selector_str, current_dom_fp
                )
                await _log("locator_heuristic_match", selector=heuristic_id, hint=task_hint)
                return loc
            except (PlaywrightException, ValueError):
                logger.debug(f"Heuristic pick '{heuristic_id}' failed visibility/location check.")
                # Don't delete from cache here, it wasn't added yet
                pass  # Fall through to LLM

        # Tier 2: LLM Pick + Retry
        # Use the page map already generated if heuristic failed
        # Loop incorporates global retry count (_retry_after_fail_global)
        for att in range(1, 2 + _retry_after_fail_global):
            elapsed_sec = time.monotonic() - start_time
            if elapsed_sec >= timeout_sec:
                raise PlaywrightTimeoutError(
                    f"EnhancedLocator timed out after {elapsed_sec:.1f}s before LLM attempt {att} for hint '{task_hint}'."
                )

            # Call global helper which uses global config (_llm_model_locator_global)
            llm_id = await _llm_pick(pm, task_hint, att)

            if not llm_id:  # LLM returned {"id": null} or failed
                logger.debug(
                    f"LLM pick attempt {att} returned no suitable ID for hint '{task_hint}'."
                )
                # Only refresh map if we intend to retry LLM
                if att <= _retry_after_fail_global:
                    logger.debug("Refreshing page map before LLM retry...")
                    pm, current_dom_fp = await self._get_page_map()
                    continue  # Try LLM again with the fresh map
                else:
                    break  # Exit loop if LLM gives up and no more retries configured

            # LLM returned an ID, try to locate and verify visibility
            try:
                loc = await _loc_from_id(self.page, llm_id)
                # Ensure element is in view before checking visibility
                await loc.scroll_into_view_if_needed()
                remaining_timeout_ms = max(500, timeout - int(elapsed_sec * 1000))
                await loc.wait_for(state="visible", timeout=remaining_timeout_ms)
                # Success! Cache the result.
                selector_str = f'[data-sb-id="{llm_id}"]'
                await loop.run_in_executor(
                    _get_pool(), _cache_put_sync, cache_key, selector_str, current_dom_fp
                )
                await _log("locator_llm_pick", selector=llm_id, attempt=att, hint=task_hint)
                return loc
            except (
                PlaywrightException,
                ValueError,
            ):  # Timeout or other error finding/parsing LLM ID
                logger.debug(
                    f"LLM pick '{llm_id}' (attempt {att}) failed visibility/location check."
                )
                # Only refresh map if we intend to retry LLM
                if att <= _retry_after_fail_global:
                    logger.debug("Refreshing page map before LLM retry...")
                    pm, current_dom_fp = await self._get_page_map()
                # Continue to next LLM attempt or loop exit

        # Tier 3: Fallback (Playwright's text selector) - Less reliable
        try:
            text_selector = f'text="{task_hint}"'
            loc = self.page.locator(text_selector).first
            elapsed_sec = time.monotonic() - start_time
            remaining_timeout_ms = max(500, timeout - int(elapsed_sec * 1000))
            if remaining_timeout_ms <= 0:
                raise PlaywrightTimeoutError("Timeout reached before text fallback check.")
            # Ensure element is in view before checking visibility
            await loc.scroll_into_view_if_needed()
            await loc.wait_for(state="visible", timeout=remaining_timeout_ms)
            await _log("locator_text_fallback", selector=text_selector, hint=task_hint)
            # Don't cache this less reliable method
            return loc
        except PlaywrightException:
            pass  # Final fallback failed, proceed to final error

        # All Tiers Failed
        elapsed_sec = time.monotonic() - start_time
        await _log("locator_fail_all", hint=task_hint[:120], duration_s=round(elapsed_sec, 1))
        # Raise a clear error indicating failure across all tiers
        raise PlaywrightTimeoutError(
            f"EnhancedLocator could not find a suitable visible element for hint: '{task_hint}' within {timeout_sec:.1f}s"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SMART ACTIONS (Using EnhancedLocator)
# ══════════════════════════════════════════════════════════════════════════════


@resilient(max_attempts=3, backoff=0.5)
async def smart_click(
    page: Page, task_hint: str, *, target_kwargs: Optional[Dict] = None, timeout_ms: int = 5000
) -> bool:
    """Locates an element using EnhancedLocator (via task_hint) and clicks it."""
    if not task_hint or not task_hint.strip():
        if target_kwargs and (target_kwargs.get("name") or target_kwargs.get("role")):
            name = target_kwargs.get("name", "")
            role = target_kwargs.get("role", "")
            task_hint = f"Click the {role or 'element'}" + (f" named '{name}'" if name else "")
            logger.warning(f"smart_click missing task_hint, generated: '{task_hint}'")
        else:
            raise ToolInputError("smart_click requires a non-empty 'task_hint'.")

    loc = EnhancedLocator(page)
    log_target = target_kwargs or {"hint": task_hint}
    try:
        element = await loc.locate(task_hint=task_hint, timeout=timeout_ms)
        element_id_for_cache = await element.get_attribute("data-sb-id") # Get ID FIRST

        # Ensure element is scrolled into view before clicking
        await element.scroll_into_view_if_needed()
        await _pause(page)

        # Perform the click
        await element.click(timeout=max(1000, timeout_ms // 2)) # Timeout for click action itself

        # Now cache using the ID obtained earlier
        # Optional: Re-calculate fingerprint *after* click if DOM might change drastically
        fp = await _dom_fingerprint(page)
        key_src = json.dumps(
            {
                "site": loc.site,
                "path": urlparse(page.url or "").path or "/",
                "hint": task_hint.lower(),
            },
            sort_keys=True,
        )
        cache_key = hashlib.sha256(key_src.encode()).hexdigest()
        if element_id_for_cache: # Only cache if we got an ID
            selector_str = f'[data-sb-id="{element_id_for_cache}"]'
            await asyncio.get_running_loop().run_in_executor(
                _get_pool(), _cache_put_sync, cache_key, selector_str, fp
            )

        await _log("click_success", target=log_target)
        return True
    except PlaywrightTimeoutError as e:  # Catch timeout from locate() or click()
        await _log("click_fail_notfound", target=log_target, error=str(e))
        raise ToolError(
            f"Click failed: Element not found or visible for hint '{task_hint}'. {e}",
            details=log_target,
        ) from e
    except PlaywrightException as e:
        await _log("click_fail_playwright", target=log_target, error=str(e))
        raise ToolError(f"Click failed due to Playwright error: {e}", details=log_target) from e
    except Exception as e:
        await _log("click_fail_unexpected", target=log_target, error=str(e))
        raise ToolError(f"Unexpected error during click: {e}", details=log_target) from e


@resilient(max_attempts=3, backoff=0.5)
async def smart_type(
    page: Page,
    task_hint: str,
    text: str,
    *,
    press_enter: bool = False,
    clear_before: bool = True,
    target_kwargs: Optional[Dict] = None,
    timeout_ms: int = 5000,
) -> bool:
    """Locates an element using EnhancedLocator (via task_hint), types text, optionally presses Enter."""
    if not task_hint or not task_hint.strip():
        if target_kwargs and (target_kwargs.get("name") or target_kwargs.get("role")):
            name = target_kwargs.get("name", "")
            role = target_kwargs.get("role", "input")
            task_hint = f"Type into the {role or 'element'}" + (f" named '{name}'" if name else "")
            logger.warning(f"smart_type missing task_hint, generated: '{task_hint}'")
        else:
            raise ToolInputError("smart_type requires a non-empty 'task_hint'.")

    loc = EnhancedLocator(page)
    log_target = target_kwargs or {"hint": task_hint}
    resolved_text = text
    log_value = (
        "***SECRET***"
        if text.startswith("secret:")
        else (text[:20] + "..." if len(text) > 23 else text)
    )

    if text.startswith("secret:"):
        secret_path = text[len("secret:") :]
        try:
            resolved_text = get_secret(secret_path)
        except (KeyError, ValueError, RuntimeError) as e:
            await _log("type_fail_secret", target=log_target, secret_ref=secret_path, error=str(e))
            raise ToolInputError(f"Failed to resolve secret '{secret_path}': {e}") from e

    try:
        element = await loc.locate(task_hint=task_hint, timeout=timeout_ms)

        # Fix 4: Add scroll into view (also needed for typing)
        element = await loc.locate(task_hint=task_hint, timeout=timeout_ms)
        element_id_for_cache = await element.get_attribute("data-sb-id") # Get ID FIRST

        # Add scroll into view (also needed for typing)
        await element.scroll_into_view_if_needed()
        await _pause(page)

        # Perform type action
        if clear_before:
            await element.fill("")
        await element.type(resolved_text, delay=random.uniform(30, 80))

        # Get fingerprint for caching even if element might be hidden
        fp = await _dom_fingerprint(page)

        # Fix 2: Handle press_enter with retry logic
        if press_enter:
            await _pause(page, (50, 150))
            try:
                # Add timeout and noWaitAfter for better press handling
                await element.press("Enter", timeout=1000, noWaitAfter=True)
            except PlaywrightException as e:
                logger.warning(f"Enter key press failed, trying click fallback: {e}")
                # Retry with click on same element if Enter press fails
                await smart_click(page, task_hint=task_hint, target_kwargs=target_kwargs)

            # Now cache using the ID obtained earlier
            # Optional: Re-calculate fingerprint *after* type/enter
            fp = await _dom_fingerprint(page)
            key_src = json.dumps(
                {
                    "site": loc.site,
                    "path": urlparse(page.url or "").path or "/",
                    "hint": task_hint.lower(),
                },
                sort_keys=True,
            )
            cache_key = hashlib.sha256(key_src.encode()).hexdigest()
            if element_id_for_cache: # Only cache if we got an ID
                selector_str = f'[data-sb-id="{element_id_for_cache}"]'
                await asyncio.get_running_loop().run_in_executor(
                    _get_pool(), _cache_put_sync, cache_key, selector_str, fp
                )

            await _log("type_success", target=log_target, value=log_value, entered=press_enter)
            return True
    except PlaywrightTimeoutError as e:  # Catch timeout from locate() or type()
        await _log("type_fail_notfound", target=log_target, value=log_value, error=str(e))
        raise ToolError(
            f"Type failed: Element not found or visible for hint '{task_hint}'. {e}",
            details=log_target,
        ) from e
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
    return await loop.run_in_executor(_get_pool(), func, *args)


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

                dfs = tabula.read_pdf(
                    str(path), pages="all", multiple_tables=True, pandas_options={"dtype": str}
                )
                if dfs:
                    results = [
                        {"type": "pdf_table", "page": i + 1, "rows": df.to_dict(orient="records")}
                        for i, df in enumerate(dfs)
                    ]
            except ImportError:
                logger.debug("tabula-py not installed. Cannot extract PDF tables.")
            except Exception as pdf_err:
                logger.warning(f"Tabula PDF extraction failed for {path.name}: {pdf_err}")
        elif ext in (".xls", ".xlsx"):
            try:
                import pandas as pd  # type: ignore

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
                logger.debug("pandas/openpyxl/xlrd not installed. Cannot extract Excel tables.")
            except Exception as excel_err:
                logger.warning(f"Pandas Excel extraction failed for {path.name}: {excel_err}")
        elif ext == ".csv":
            try:
                import pandas as pd  # type: ignore

                df = pd.read_csv(str(path), dtype=str)
                results = [{"type": "csv_table", "rows": df.to_dict(orient="records")}]
            except ImportError:
                logger.debug("pandas not installed. Cannot extract CSV tables.")
            except Exception as csv_err:
                logger.warning(f"Pandas CSV extraction failed for {path.name}: {csv_err}")
    except Exception as outer_err:
        logger.error(f"Error during table extraction setup for {path.name}: {outer_err}")
    return results


async def _extract_tables_async(path: Path) -> list:
    """Extract tables from document files (PDF, Excel, CSV) using thread pool."""
    try:
        tables = await asyncio.to_thread(_extract_tables_sync, path)
        if tables:
            await _log("table_extract_success", file=str(path), num_tables=len(tables))
        return tables
    except Exception as e:
        await _log("table_extract_error", file=str(path), error=str(e))
        return []


@resilient()
async def smart_download(
    page: Page,
    task_hint: str,
    dest_dir: Optional[Union[str, Path]] = None,
    target_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Clicks an element to initiate download, saves file via Playwright to a unique path
    determined by FileSystemTool, reads back using FileSystemTool, calculates hash,
    and extracts tables. Uses FileSystemTool to manage the download directory.

    Args:
        page: The Playwright Page object.
        task_hint: Natural language description of the download link/button.
        dest_dir: Optional destination directory path string (relative or absolute) for FileSystemTool.
        target_kwargs: Optional original target dict for logging/context.

    Returns:
        Dictionary with download info (path, hash, size, tables) or error.
    """
    final_dl_dir_path_str = "Unknown" # Keep track for logging
    try:
        # --- Determine and Prepare Download Directory using FileSystemTool ---
        if dest_dir:
            download_dir_path_str = str(dest_dir)
        else:
            # Default: Use a relative path structure recognizable by FileSystemTool
            # Example: Assumes 'smart_browser_root' is implicitly mapped or relative to an allowed base
            default_dl_subdir = "downloads"
            download_dir_path_str = f"smart_browser_root/{default_dl_subdir}"
            # --- OR if using absolute paths directly (requires _HOME to be allowed): ---
            # download_dir_path_str = str(_HOME / default_dl_subdir)

        logger.info(f"Ensuring download directory exists: '{download_dir_path_str}' using filesystem tool.")
        create_dir_result = await create_directory(path=download_dir_path_str)

        if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
            error_detail = create_dir_result.get('error', 'Unknown') if isinstance(create_dir_result, dict) else 'Invalid response'
            raise ToolError(f"Failed to prepare download directory '{download_dir_path_str}'. Filesystem tool error: {error_detail}")

        # Use the actual absolute path returned by the tool
        final_dl_dir_path_str = create_dir_result.get("path")
        if not final_dl_dir_path_str:
             logger.warning(f"create_directory did not return a path for '{download_dir_path_str}'. Using input path.")
             final_dl_dir_path_str = download_dir_path_str # Fallback, less ideal

        final_dl_dir_path = Path(final_dl_dir_path_str) # Convert to Path object for local use
        logger.info(f"Download directory confirmed/created at: {final_dl_dir_path}")

    except ToolError as e:
        logger.error(f"Error preparing download directory '{download_dir_path_str}': {e}", exc_info=True)
        raise ToolError(f"Could not prepare download directory: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error preparing download directory: {e}", exc_info=True)
        raise ToolError(f"An unexpected error occurred preparing download directory: {str(e)}") from e
    # --- End Directory Preparation ---

    log_target = target_kwargs or {"hint": task_hint}
    out_path: Optional[Path] = None # Define earlier for clarity

    try:
        # --- Initiate Download ---
        async with page.expect_download(timeout=60000) as dl_info:
            await smart_click(
                page, task_hint=task_hint, target_kwargs=target_kwargs, timeout_ms=10000
            )

        dl = await dl_info.value
        suggested_fname = dl.suggested_filename or f"download_{int(time.time())}.dat"
        safe_fname = re.sub(r"[^\w.\- ]", "_", suggested_fname)
        safe_fname = re.sub(r"\s+", "_", safe_fname).strip("._-")
        # --- Construct initial desired path ---
        initial_desired_path = final_dl_dir_path / (safe_fname or f"download_{int(time.time())}.dat")

        # --- Get Unique Path using FileSystemTool ---
        logger.debug(f"Requesting unique path based on: {initial_desired_path}")
        try:
            unique_path_result = await get_unique_filepath(path=str(initial_desired_path))
            if not isinstance(unique_path_result, dict) or not unique_path_result.get("success"):
                error_detail = unique_path_result.get('error', 'Unknown') if isinstance(unique_path_result, dict) else 'Invalid response'
                raise ToolError(f"Failed to get unique download path. Filesystem tool error: {error_detail}")

            final_unique_path_str = unique_path_result.get("path")
            if not final_unique_path_str:
                 raise ToolError("Filesystem tool get_unique_filepath succeeded but did not return a path.")

            out_path = Path(final_unique_path_str) # Use the unique path for saving
            logger.info(f"Determined unique download path: {out_path}")

        except ToolError as e:
            logger.error(f"Error determining unique download path based on '{initial_desired_path}': {e}", exc_info=True)
            raise ToolError(f"Could not determine unique save path for download: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting unique download path: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred finding a unique save path: {str(e)}") from e
        # --- End Getting Unique Path ---

        # --- Save Download using Playwright ---
        logger.info(f"Playwright saving download to unique path: {out_path}")
        await dl.save_as(out_path)
        logger.info(f"Playwright download save complete: {out_path}")

        # --- Read back, analyze (same as before, using out_path) ---
        file_data: Optional[bytes] = None
        file_size = -1
        sha256_hash = None
        read_back_error = None

        try:
            read_path_str = str(out_path)
            read_result = await read_binary_file(path=read_path_str) # Use the filesystem tool

            # ... (rest of the read_back logic is the same as previous answer) ...
            if isinstance(read_result, dict) and read_result.get("success"):
                if isinstance(read_result.get("content"), list) and len(read_result["content"]) > 0:
                     content_block = read_result["content"][0]
                     # Assume binary tool returns content like: {'type': 'binary', 'bytes': b'...'}
                     if content_block.get("type") == "binary" and isinstance(content_block.get("bytes"), bytes):
                          file_data = content_block.get("bytes")
                          file_size = len(file_data)
                     elif isinstance(content_block.get("text"), str): # Handle if read_binary_file falls back weirdly
                          logger.warning(f"read_binary_file returned text content for {read_path_str}")
                          read_back_error = "Read back tool returned unexpected text format for binary file."
                     else:
                          read_back_error = "Filesystem tool did not return expected binary content block format."
                else:
                    read_back_error = "Filesystem tool read succeeded but content format unexpected or empty."
            else:
                read_back_error = read_result.get('error', 'Filesystem tool failed to read back downloaded file.') if isinstance(read_result, dict) else 'Invalid response from read_binary_file'

        except ToolError as e:
            read_back_error = f"Filesystem tool error reading back {out_path}: {e}"
        except Exception as e:
            read_back_error = f"Unexpected error reading back {out_path}: {e}"

        if read_back_error:
             logger.error(f"Failed to read back downloaded file {out_path} for analysis. Error: {read_back_error}")
             # Return partial success but note the failure.
             info = {
                 "success": False, # Mark as overall failure if readback fails
                 "file_path": str(out_path),
                 "file_name": out_path.name,
                 "error": f"Download saved, but failed to read back for analysis: {read_back_error}",
                 "url": dl.url,
             }
             await _log("download_success_readback_fail", target=log_target, **info)
             # Raise ToolError to signal failure more clearly
             raise ToolError(info["error"], details=info)


        # --- Hashing and Table Extraction (same as before) ---
        sha256_hash = await _compute_hash_async(file_data)
        tables = []
        if out_path.suffix.lower() in (".pdf", ".xls", ".xlsx", ".csv"):
            try:
                table_extraction_task = asyncio.create_task(_extract_tables_async(out_path))
                tables = await asyncio.wait_for(table_extraction_task, timeout=120)
            except asyncio.TimeoutError:
                 logger.warning(f"Table extraction timed out for {out_path.name}")
                 # Cancel properly
                 if 'table_extraction_task' in locals() and not table_extraction_task.done():
                     table_extraction_task.cancel()
                     try: 
                         await asyncio.wait_for(table_extraction_task, timeout=1.0)
                     except Exception: 
                         pass
            except Exception as extract_err:
                logger.error(f"Table extraction failed for {out_path.name}: {extract_err}", exc_info=True)

        info = {
            "success": True,
            "file_path": str(out_path), # Return the final unique path
            "file_name": out_path.name,
            "sha256": sha256_hash,
            "size_bytes": file_size,
            "url": dl.url,
            "tables_extracted": bool(tables),
            "tables": tables[:5],
        }
        await _log("download_success", target=log_target, **info)
        return info

    # --- Error Handling (mostly same, ensure path context uses out_path if available) ---
    except (ToolInputError, ToolError) as e:
        log_event = "download_fail_other"
        await _log(log_event, target=log_target, error=str(e), path=str(out_path) if out_path else "N/A")
        raise
    except PlaywrightTimeoutError as e:
        await _log("download_fail_timeout", target=log_target, error=str(e), path=str(out_path) if out_path else "N/A")
        raise ToolError(f"Download operation timed out: {e}") from e
    except PlaywrightException as e:
        await _log("download_fail_playwright", target=log_target, error=str(e), path=str(out_path) if out_path else "N/A")
        raise ToolError(f"Download failed due to Playwright error: {e}") from e
    except Exception as e:
        await _log("download_fail_unexpected", target=log_target, error=str(e), path=str(out_path) if out_path else "N/A")
        raise ToolError(f"Unexpected error during download: {e}") from e


# --- PDF/Docs Crawler Helpers ---
_SLUG_RE = re.compile(r"[^a-z0-9\-_]+")


def _slugify(text: str, max_len: int = 60) -> str:
    if not text:
        return "file"
    slug = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    slug = _SLUG_RE.sub("-", slug).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_len].strip("-") or "file"


def _get_dir_slug(url: str) -> str:
    try:
        path_parts = [p for p in Path(urlparse(url).path).parts if p and p != "/"]
        if len(path_parts) >= 2:
            return f"{_slugify(path_parts[-2], 20)}-{_slugify(path_parts[-1], 20)}"
        elif len(path_parts) == 1:
            return _slugify(path_parts[-1], 40)
        else:
            return _slugify(urlparse(url).netloc, 40) or "domain"
    except Exception:
        return "path"


async def _fetch_html(
    client: httpx.AsyncClient, url: str, rate_limiter: Optional[RateLimiter] = None
) -> Optional[str]:
    """Fetch HTML using httpx client with optional rate limiting."""
    try:
        if rate_limiter:
            await rate_limiter.acquire()

        async with client.stream("GET", url, follow_redirects=True, timeout=20.0) as response:
            response.raise_for_status()
            if response.status_code == 204:
                return None
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                return None
            if int(response.headers.get("content-length", 0)) > 5 * 1024 * 1024:
                return None  # 5MB limit
            html = await response.aread()
            try:
                return html.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return html.decode("iso-8859-1")
                except UnicodeDecodeError:
                    logger.warning(f"Could not decode HTML from {url}")
                    return None
    except httpx.HTTPStatusError as e:
        if 400 <= e.response.status_code < 600:
            logger.debug(f"HTTP error fetching {url}: {e.response.status_code}")  # Debug level
        return None
    except httpx.RequestError as e:
        logger.warning(f"Network error fetching {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None


def _extract_links(base_url: str, html: str) -> Tuple[List[str], List[str]]:
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
                abs_url = urllib.parse.urljoin(base_url, href)
                parsed_url = urlparse(abs_url)
                clean_url = parsed_url._replace(fragment="").geturl()
                path_lower = parsed_url.path.lower()
                if path_lower.endswith(".pdf"):
                    pdfs.add(clean_url)
                elif parsed_url.netloc == base_netloc and (
                    path_lower.endswith((".html", ".htm", "/"))
                    or "." not in Path(parsed_url.path).name
                ):
                    if not path_lower.endswith(".pdf"):
                        pages.add(clean_url)
            except ValueError:
                pass
            except Exception as link_err:
                logger.warning(f"Error processing link '{href}' on {base_url}: {link_err}")
    except Exception as soup_err:
        logger.error(f"Error parsing HTML for links on {base_url}: {soup_err}")
    return list(pdfs), list(pages)


class RateLimiter:
    """Limits async operations to a maximum rate (requests per second)."""

    def __init__(self, rate_limit: float = 1.0):
        if rate_limit <= 0:
            raise ValueError("Rate limit must be positive")
        self.interval = 1.0 / rate_limit
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            time_since_last = now - self.last_request_time
            time_to_wait = self.interval - time_since_last
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
                now = time.monotonic()
            self.last_request_time = now


async def crawl_for_pdfs(
    start_url: str,
    include_regex: Optional[str] = None,
    max_depth: int = 2,
    max_pdfs: int = 100,
    max_pages_crawl: int = 500,
    rate_limit_rps: float = 2.0,
) -> List[str]:
    """Crawls a website using BFS to find PDF links."""
    try:
        inc_re = re.compile(include_regex, re.I) if include_regex else None
    except re.error as e:
        raise ToolInputError(f"Invalid include_regex: {e}") from e
    seen_urls: Set[str] = set()
    pdf_urls_found: Set[str] = set()
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    seen_urls.add(start_url)
    visit_count = 0
    rate_limiter = RateLimiter(rate_limit_rps)
    base_netloc = urlparse(start_url).netloc
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SmartBrowserBot/1.0; +http://example.com/bot)"
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
        while queue and len(pdf_urls_found) < max_pdfs and visit_count < max_pages_crawl:
            current_url, current_depth = queue.popleft()
            visit_count += 1
            # logger.debug(f"Crawling [Depth:{current_depth}, Visited:{visit_count}]: {current_url}")
            html = await _fetch_html(client, current_url, rate_limiter)
            if not html:
                continue
            pdfs, pages = _extract_links(current_url, html)
            for pdf_url in pdfs:
                if pdf_url not in pdf_urls_found:
                    if inc_re is None or inc_re.search(pdf_url):
                        pdf_urls_found.add(pdf_url)
                        logger.info(f"PDF found: {pdf_url}")
                        if len(pdf_urls_found) >= max_pdfs:
                            break
            if len(pdf_urls_found) >= max_pdfs:
                break
            if current_depth < max_depth:
                for page_url in pages:
                    if urlparse(page_url).netloc == base_netloc and page_url not in seen_urls:
                        seen_urls.add(page_url)
                        queue.append((page_url, current_depth + 1))
    if visit_count >= max_pages_crawl:
        logger.warning(f"PDF crawl stopped: Max pages ({max_pages_crawl}) reached.")
    if len(pdf_urls_found) >= max_pdfs:
        logger.info(f"PDF crawl stopped: Max PDFs ({max_pdfs}) reached.")
    return list(pdf_urls_found)


async def _download_file_direct(url: str, dest_dir_str: str, seq: int = 1) -> Dict:
    """
    Downloads a file directly using httpx, determines a unique filename via
    FileSystemTool, writes the content using FileSystemTool, and computes hash.

    Args:
        url: The URL to download from.
        dest_dir_str: The validated destination directory path string (from create_directory).
        seq: Sequence number for default filename generation.

    Returns:
        Dictionary with download results (success/error, path, size, hash, url).
    """
    final_output_path_str = None # Track path for potential deletion on error
    downloaded_content: Optional[bytes] = None

    try:
        # --- 1. Determine Initial Filename ---
        parsed_url = urlparse(url)
        fname_base = os.path.basename(parsed_url.path) if parsed_url.path else ""
        initial_filename: str
        if not fname_base or fname_base == "/" or "." not in fname_base:
            dir_slug = _get_dir_slug(url)
            initial_filename = f"{seq:03d}_{dir_slug}_{_slugify(fname_base or 'download')}"
            initial_filename += ".pdf" if url.lower().endswith(".pdf") else ".dat"
        else:
            initial_filename = f"{seq:03d}_{_slugify(fname_base)}"

        # Initial desired path *string*
        initial_desired_path = os.path.join(dest_dir_str, initial_filename)

        # --- 2. Download Content and Refine Filename ---
        headers = {
            "User-Agent": "Mozilla/5.0", "Accept": "*/*", "Accept-Encoding": "gzip, deflate, br"
        }
        refined_desired_path = initial_desired_path # Start with initial path

        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0, headers=headers) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    return {
                        "url": url, "error": f"HTTP {response.status_code}",
                        "status_code": response.status_code, "success": False,
                        "path": initial_desired_path # Report intended path on HTTP error
                    }

                # Refine filename based on headers *before* getting unique path
                content_disposition = response.headers.get("content-disposition")
                if content_disposition:
                    match = re.search(r'filename="?([^"]+)"?', content_disposition)
                    if match:
                         refined_filename = f"{seq:03d}_{_slugify(match.group(1))}"
                         refined_desired_path = os.path.join(dest_dir_str, refined_filename)

                content_type = response.headers.get("content-type", "").split(";")[0].strip()
                # Adjust suffix based on refined path
                current_stem, current_ext = os.path.splitext(refined_desired_path)
                if content_type == "application/pdf" and current_ext.lower() != ".pdf":
                    refined_desired_path = current_stem + ".pdf"

                # Download content to memory
                downloaded_content = await response.aread()
                bytes_read = len(downloaded_content)

        # --- 3. Get Unique Filepath using FileSystemTool ---
        logger.debug(f"Requesting unique path based on refined path: {refined_desired_path}")
        try:
            unique_path_result = await get_unique_filepath(path=refined_desired_path)
            if not isinstance(unique_path_result, dict) or not unique_path_result.get("success"):
                error_detail = unique_path_result.get('error', 'Unknown') if isinstance(unique_path_result, dict) else 'Invalid response'
                raise ToolError(f"Failed to get unique download path. Filesystem tool error: {error_detail}")

            final_output_path_str = unique_path_result.get("path")
            if not final_output_path_str:
                 raise ToolError("Filesystem tool get_unique_filepath succeeded but did not return a path.")
            logger.info(f"Determined unique download save path: {final_output_path_str}")

        except ToolError as e:
            logger.error(f"Error determining unique download path based on '{refined_desired_path}': {e}", exc_info=True)
            raise ToolError(f"Could not determine unique save path for download: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting unique download path: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred finding a unique save path: {str(e)}") from e

        # --- 4. Write Content using FileSystemTool ---
        try:
            write_result = await write_file(
                path=final_output_path_str,
                content=downloaded_content # Pass downloaded bytes
            )
            if not isinstance(write_result, dict) or not write_result.get("success"):
                error_detail = write_result.get('error', 'Unknown') if isinstance(write_result, dict) else 'Invalid response'
                raise ToolError(f"Filesystem tool failed to write downloaded file: {error_detail}")
            # Verify path returned matches if needed
            # final_output_path_str = write_result.get("path", final_output_path_str)

        except ToolError as e:
            logger.error(f"Failed to write downloaded file '{final_output_path_str}' using filesystem tool: {e}", exc_info=True)
            raise ToolError(f"Could not write downloaded file: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error writing downloaded file '{final_output_path_str}': {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred writing the downloaded file: {str(e)}") from e

        # --- 5. Compute Hash (from memory) ---
        hasher = hashlib.sha256()
        hasher.update(downloaded_content)
        file_hash = hasher.hexdigest()

        # Remove direct chmod calls

        await _log(
            "download_direct_success", url=url, file=final_output_path_str,
            size=bytes_read, sha256=file_hash
        )

        return {
            "url": url, "file": final_output_path_str, "size": bytes_read,
            "sha256": file_hash, "success": True,
        }

    except httpx.RequestError as e:
        logger.warning(f"Network error downloading {url}: {e}")
        return {"url": url, "error": f"Network error: {e}", "success": False, "path": final_output_path_str}
    except (ToolError, ToolInputError) as e: # Catch errors from FS tools or validation
         logger.error(f"Tool error downloading {url} directly: {e}", exc_info=True)
         # Attempt cleanup if write might have happened before error
         if final_output_path_str:
             try:
                 logger.warning(f"Attempting cleanup of potentially incomplete file: {final_output_path_str}")
                 await delete_path(final_output_path_str)
             except Exception as del_e:
                 logger.error(f"Cleanup failed for {final_output_path_str}: {del_e}")
         return {"url": url, "error": f"Download failed: {e}", "success": False, "path": final_output_path_str}
    except Exception as e:
        logger.error(f"Unexpected error downloading {url} directly: {e}", exc_info=True)
        # Attempt cleanup if write might have happened
        if final_output_path_str:
             try:
                 logger.warning(f"Attempting cleanup of potentially incomplete file: {final_output_path_str}")
                 await delete_path(final_output_path_str)
             except Exception as del_e:
                 logger.error(f"Cleanup failed for {final_output_path_str}: {del_e}")
        return {"url": url, "error": f"Download failed unexpectedly: {e}", "success": False, "path": final_output_path_str}


# --- OSS Documentation Crawler Helpers ---
_DOC_EXTS = (".html", ".htm", "/")
_DOC_STOP_PAT = re.compile(
    r"\.(png|jpg|jpeg|gif|svg|css|js|zip|tgz|gz|whl|exe|dmg|ico|woff|woff2|map|json|xml|txt)$", re.I
)


def _looks_like_docs_url(url: str) -> bool:
    try:
        url_low = url.lower()
        parsed = urlparse(url_low)
        if parsed.query:
            return False
        if any(
            f in parsed.path
            for f in [
                "/api/",
                "/blog/",
                "/news/",
                "/community/",
                "/forum/",
                "/download/",
                "/install/",
                "/_static/",
                "/_images/",
                "/assets/",
                "/media/",
            ]
        ):
            return False
        has_doc = (
            "docs" in url_low
            or "guide" in url_low
            or "tuto" in url_low
            or "ref" in url_low
            or "api" in parsed.path
            or "faq" in url_low
            or "howto" in url_low
            or "userguide" in url_low
        )
        has_end = url_low.endswith(_DOC_EXTS)
        return (has_doc or has_end) and not bool(_DOC_STOP_PAT.search(parsed.path))
    except Exception:
        return False


async def _pick_docs_root(pkg_name: str) -> Optional[str]:
    try:
        logger.info(f"Searching for docs root for package: '{pkg_name}'")
        # Try DDG first as it's often less noisy
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
                parsed = urlparse(url)
                path_segments = [seg for seg in parsed.path.split("/") if seg]
                if len(path_segments) > 1:
                    root_url = parsed._replace(
                        path="/".join(path_segments[:-1]) + "/", query="", fragment=""
                    ).geturl()
                    if _looks_like_docs_url(root_url):
                        logger.info(f"Found potential docs root: {root_url}")
                        return root_url
                logger.info(f"Found potential docs page: {url}")
                return url  # Return first plausible hit
        logger.warning(f"Could not reliably find docs root for '{pkg_name}'.")
        return search_results[0]["url"] if search_results else None
    except ToolError as e:
        logger.error(f"Search failed finding docs root for '{pkg_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error finding docs root for '{pkg_name}': {e}")
        return None


def _summarize_html_sync(html: str, max_len: int = 10000) -> str:
    """Synchronous HTML summarization (runs in thread pool)."""
    if not html:
        return ""
    MAX_HTML_SIZE = 3 * 1024 * 1024
    if len(html) > MAX_HTML_SIZE:
        html = html[:MAX_HTML_SIZE]
    text = ""
    try:  # Try Trafilatura
        if trafilatura is not None:
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=False, favor_precision=True
            )
            if extracted and len(extracted) > 100:
                text = extracted
    except Exception as e:
        logger.warning(f"Trafilatura failed: {e}")
    if not text or len(text) < 200:  # Try Readability-lxml
        try:
            if Document is not None:
                doc = Document(html)
                summary_html = doc.summary(html_partial=True)
                soup = BeautifulSoup(summary_html, "html.parser")
                text = soup.get_text(" ", strip=True)
        except Exception as e:
            logger.warning(f"Readability failed: {e}")
    if not text or len(text) < 100:  # Fallback: BeautifulSoup
        try:
            soup = BeautifulSoup(html, "lxml")  # Use lxml parser
            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "form",
                    "figure",
                    "figcaption",
                    "noscript",
                ]
            ):
                element.decompose()
            text = soup.get_text(" ", strip=True)
        except Exception as e:
            logger.warning(f"BeautifulSoup fallback failed: {e}")
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text[:max_len]


async def _grab_readable(
    client: httpx.AsyncClient, url: str, rate_limiter: RateLimiter
) -> Optional[str]:
    """Fetch HTML using client and rate limiter, then extract readable text."""
    html = await _fetch_html(client, url, rate_limiter)
    if html:
        return await _run_in_thread(_summarize_html_sync, html)  # Run sync summarizer in thread
    return None


async def crawl_docs_site(
    root_url: str, max_pages: int = 40, rate_limit_rps: float = 3.0
) -> List[Tuple[str, str]]:
    """BFS crawl a documentation site, collecting readable text."""
    try:
        start_netloc = urlparse(root_url).netloc
        assert start_netloc
    except (ValueError, AssertionError) as e:
        raise ToolInputError(f"Invalid root URL: {root_url}") from e
    seen_urls: Set[str] = set()
    queue: deque[str] = deque([root_url])
    seen_urls.add(root_url)
    output_pages: List[Tuple[str, str]] = []
    visit_count = 0
    max_visits = max(max_pages * 5, 200)
    rate_limiter = RateLimiter(rate_limit_rps)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SmartBrowserDocBot/1.0)"}
    logger.info(f"Starting documentation crawl from: {root_url} (Max pages: {max_pages})")
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
        while queue and len(output_pages) < max_pages and visit_count < max_visits:
            current_url = queue.popleft()
            visit_count += 1
            # logger.debug(f"Crawling Doc [{len(output_pages)}/{visit_count}]: {current_url}")
            readable_text = await _grab_readable(client, current_url, rate_limiter)
            if readable_text:
                output_pages.append((current_url, readable_text))
                if len(output_pages) >= max_pages:
                    break
                html_for_links = await _fetch_html(client, current_url, rate_limiter)
                if html_for_links:
                    _, page_links = _extract_links(current_url, html_for_links)
                    for link_url in page_links:
                        try:
                            parsed_link = urlparse(link_url)
                            if (
                                parsed_link.netloc == start_netloc
                                and _looks_like_docs_url(link_url)
                                and link_url not in seen_urls
                            ):
                                seen_urls.add(link_url)
                                queue.append(link_url)
                        except ValueError:
                            pass
    if visit_count >= max_visits:
        logger.warning(f"Doc crawl stopped: Max visits ({max_visits}) reached.")
    logger.info(f"Doc crawl finished. Collected {len(output_pages)} pages.")
    return output_pages


# ══════════════════════════════════════════════════════════════════════════════
# 10. PAGE STATE EXTRACTION (using EnhancedLocator's Page Map)
# ══════════════════════════════════════════════════════════════════════════════


async def get_page_state(page: Page, max_elements: Optional[int] = None) -> dict[str, Any]:
    """
    Extracts the current state of the page using the page map functionality.
    The number of elements is controlled by _MAX_WIDGETS.

    Args:
        page: The Playwright Page object.
        max_elements: (Deprecated/Ignored) Use _MAX_WIDGETS constant instead.

    Returns:
        Dictionary representing the page state (URL, title, main_text, elements).
    """
    if max_elements is not None:
        logger.warning("get_page_state 'max_elements' arg is deprecated, use _MAX_WIDGETS.")
    if not page or page.is_closed():
        return {"error": "Page closed", "url": "", "title": "", "elements": [], "main_text": ""}

    start_time = time.monotonic()
    try:
        # Use the core page map builder
        page_map, _ = await _build_page_map(page)  # Fingerprint not needed here
        duration = time.monotonic() - start_time
        await _log(
            "page_state_extracted",
            url=page_map.get("url"),
            duration_ms=int(duration * 1000),
            num_elements=len(page_map.get("elements", [])),
        )
        # Return the map directly, keys are already aligned (url, title, main_text, elements)
        return page_map
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(f"Error getting page state for {page.url}: {e}", exc_info=True)
        await _log("page_error", url=page.url, error=str(e), duration_ms=int(duration * 1000))
        return {
            "error": f"Failed to get page state: {e}",
            "url": page.url or "unknown",
            "title": "[Error]",
            "elements": [],
            "main_text": "",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 11. LLM BRIDGE
# ══════════════════════════════════════════════════════════════════════════════


def _extract_json_block(text: str) -> Optional[str]:
    """Tries to extract the first valid JSON object or array block from text."""
    # Try to find ```json ... ``` block first
    match_markdown = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if match_markdown:
        return match_markdown.group(1).strip()
    # Fallback to finding first {..} or [..]
    match_bare = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match_bare:
        block = match_bare.group(0)
        # Quick check for balanced brackets/braces (doesn't guarantee validity)
        if block.count("{") == block.count("}") and block.count("[") == block.count("]"):
            return block
    return None


# First we need to add a new rate-limit aware resilient decorator
def _llm_resilient(max_attempts: int = 3, backoff: float = 1.0):
    """Decorator for LLM API calls that handles rate limits with exponential backoff."""

    def wrap(fn):
        @functools.wraps(fn)
        async def inner(*a, **kw):
            attempt = 0
            while True:
                try:
                    if attempt > 0:
                        jitter_delay = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                        await asyncio.sleep(jitter_delay)
                    return await fn(*a, **kw)
                except ProviderError as e:
                    # Check specifically for rate limit errors
                    if (
                        "429" in str(e)
                        or "rate limit" in str(e).lower()
                        or "too many requests" in str(e).lower()
                    ):
                        attempt += 1
                        func_name = getattr(fn, "__name__", "unknown_func")
                        if attempt >= max_attempts:
                            logger.error(
                                f"LLM rate limit: Operation '{func_name}' failed after {max_attempts} attempts: {e}"
                            )
                            raise ToolError(
                                f"LLM rate-limit exceeded after {max_attempts} attempts: {e}"
                            ) from e

                        # Try to extract Retry-After header if available
                        retry_after = None
                        try:
                            # Look for Retry-After value in error message
                            match = re.search(r"retry[- ]after[: ]+(\d+)", str(e).lower())
                            if match:
                                retry_after = int(match.group(1))
                        except (ValueError, AttributeError):
                            pass

                        delay = retry_after or backoff * (2 ** (attempt - 1)) * random.uniform(
                            0.8, 1.2
                        )
                        logger.warning(
                            f"LLM rate limit hit. Retrying '{func_name}' after {delay:.2f}s (attempt {attempt}/{max_attempts})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Re-raise non-rate-limit provider errors
                        raise
                except Exception as e:  # Catch unexpected errors
                    # Only retry if it looks like a transient error
                    if (
                        isinstance(e, (httpx.RequestError, asyncio.TimeoutError))
                        or "timeout" in str(e).lower()
                    ):
                        attempt += 1
                        func_name = getattr(fn, "__name__", "unknown_func")
                        if attempt >= max_attempts:
                            logger.error(
                                f"LLM call: Operation '{func_name}' failed after {max_attempts} attempts: {e}"
                            )
                            raise ToolError(
                                f"LLM call failed after {max_attempts} attempts: {e}"
                            ) from e
                        delay = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                        logger.warning(
                            f"LLM transient error. Retrying '{func_name}' after {delay:.2f}s (attempt {attempt}/{max_attempts})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Re-raise non-retryable errors
                        raise

        return inner

    return wrap


@_llm_resilient(max_attempts=3, backoff=1.0)
async def _call_llm(
    messages: Sequence[Dict[str, str]],
    model: str = "gpt-4.1-mini",  # Default model for general calls
    expect_json: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Union[Dict[str, Any], List[Any]]:
    """Calls the LLM using MCP server's chat completion tool."""
    if not messages:
        return {"error": "No messages provided to LLM."}

    llm_args = {
        "model": model,
        "messages": messages.copy(), # Create a copy to avoid modifying the original
        "temperature": temperature,
        "max_tokens": max_tokens,
        "additional_params": {},
    }
    # Determine provider and update model name if prefixed
    llm_args["provider"] = Provider.OPENAI.value # Default
    if model:
        extracted_provider, extracted_model = parse_model_string(model)
        if extracted_provider:
            llm_args["provider"] = extracted_provider
            llm_args["model"] = extracted_model # Use the model name without the prefix

    use_json_instruction = False # Flag to add manual JSON instruction if native mode not used

    if expect_json:
        try:
            # Get the provider instance to check its capabilities
            provider_instance = await get_provider(llm_args["provider"])
            supports_native_json = (
                 (llm_args["provider"] == Provider.OPENAI.value and llm_args["model"].startswith("gpt-")) or
                 getattr(provider_instance, 'supports_json_response_format', False) # Add a hypothetical capability flag
            )
            if supports_native_json:
                logger.debug(f"Provider {llm_args['provider']} supports native JSON mode. Adding response_format to additional_params.")
                # Add response_format to *additional_params*, not directly to llm_args
                llm_args["additional_params"]["response_format"] = {"type": "json_object"}
                use_json_instruction = False # Don't add manual instruction
            else:
                logger.debug(f"Provider {llm_args['provider']} does not natively support JSON mode (or check failed). Will use manual instruction.")
                use_json_instruction = True # Need manual instruction
        except ProviderError as e:
             logger.warning(f"Could not get provider {llm_args['provider']} to check JSON support: {e}. Assuming manual instruction needed.")
             use_json_instruction = True # Fallback if provider check fails
        except Exception as e:
             logger.error(f"Unexpected error checking provider JSON support: {e}. Assuming manual instruction needed.", exc_info=True)
             use_json_instruction = True

    # If manual instruction needed, modify the messages list
    if use_json_instruction:
        json_instruction = (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "Start with `{` or `[` and end with `}` or `]`. "
            "Do not include ```json markers or explanations."
        )
        modified_messages = list(llm_args["messages"])
        if modified_messages and modified_messages[-1]["role"] == "user":
            last_content = modified_messages[-1]["content"]
            modified_messages[-1]["content"] = last_content + json_instruction
        else:
            modified_messages.append(
                {
                    "role": "user",
                    "content": "Please provide your response based on my previous messages."
                    + json_instruction,
                }
            )
        llm_args["messages"] = modified_messages # Use the modified list

    try:
        start_time = time.monotonic()

        # Ensure llm_args contains 'messages' key as expected by chat_completion
        if "messages" not in llm_args:
             logger.error("_call_llm was invoked without a 'messages' argument.")
             return {"error": "Internal error: _call_llm requires 'messages'."}

        resp = await chat_completion(**llm_args)
        duration = time.monotonic() - start_time

        await _log(
             "llm_call_complete",
             model=resp.get("model", model), # Use model from response if available
             duration_ms=int(duration * 1000),
             success=resp.get("success", False),
             cached=resp.get("cached_result", False) # Log if cached
        )

        if not resp.get("success"):
            error_msg = resp.get("error", "LLM call failed")
            # Attempt to get raw response if available from the error structure
            raw_resp_detail = resp.get("details", {}).get("raw_response") or resp.get("raw_response")
            return {"error": f"LLM API Error: {error_msg}", "raw_response": raw_resp_detail}

        # Handle response structure from chat_completion
        assistant_message = resp.get("message", {})
        content = assistant_message.get("content") # Get content, might be None
        raw_text = content.strip() if isinstance(content, str) else "" # Only strip if it's a string

        if not raw_text:
            return {"error": "LLM returned empty response content."}
        if not expect_json:
            return {"text": raw_text} # Return simple text if JSON not expected

        # Try parsing JSON (especially if native mode was used or instruction given)
        try:
            # First, try parsing the raw_text directly
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # If direct parse fails, try extracting block (common fallback)
            json_block = _extract_json_block(raw_text)
            if json_block:
                try:
                    parsed = json.loads(json_block)
                    logger.warning("LLM response had extra text; extracted JSON block.")
                    return parsed
                except json.JSONDecodeError as e:
                    err = f"Could not parse extracted JSON block: {e}. Block: {json_block[:500]}..."
                    return {"error": err, "raw_response": raw_text[:1000]}
            else: # No block found or extracted block invalid
                err = "Could not parse JSON from LLM response (no valid block found)."
                return {"error": err, "raw_response": raw_text[:1000]}

    except ProviderError as e:
        logger.error(f"LLM Provider error: {e}")
        # Extract raw response if available in the exception details
        raw_resp_detail = getattr(e, 'details', {}).get('raw_response')
        return {"error": f"LLM Provider Error: {e}", "raw_response": raw_resp_detail}
    except Exception as e:
        logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
        return {"error": f"LLM call failed unexpectedly: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# 12. NATURAL-LANGUAGE MACRO RUNNER / AUTOPILOT PLANNER
# ══════════════════════════════════════════════════════════════════════════════
ALLOWED_ACTIONS = {"click", "type", "wait", "download", "extract", "finish", "scroll"}


async def _plan_macro(
    page_state: Dict[str, Any], task: str, model: str = "gpt-4.1-mini"
) -> List[Dict[str, Any]]:
    """Generates a sequence of browser actions for the macro runner."""
    action_details = """
    Allowed actions and their arguments:
    - "click": requires "task_hint" (natural language description of element).
    - "type": requires "task_hint" (element description), "text" (string to type). Optional: "enter": bool, "clear_before": bool.
    - "wait": requires "ms" (integer milliseconds).
    - "download": requires "task_hint" (download link/button description). Optional: "dest" (string destination directory).
    - "extract": requires "selector" (CSS selector string). Returns text of matching elements.
    - "scroll": requires "direction" ("up", "down", "bottom", "top"). Optional: "amount_px": int.
    - "finish": takes no arguments. Signals task completion.
    """
    # Extract compact elements summary for prompt
    elements_summary = [
        f"id={el.get('id')} tag={el.get('tag')} role={el.get('role', ' ')} text='{el.get('text', ' ')}'"
        for el in page_state.get("elements", [])
    ]

    messages = [
        {
            "role": "system",
            "content": textwrap.dedent(f"""
                You are a web automation planner. Based on the user's TASK and the current PAGE_STATE
                (URL, title, summary, elements), create a plan as a JSON list of action steps.
                Use ONLY the allowed actions: {sorted(ALLOWED_ACTIONS)}. Provide arguments as needed.
                Use 'task_hint' for 'click', 'type', and 'download' actions.
                Your response MUST be ONLY the JSON list of steps `[...]`.
                ACTION DETAILS:\n{action_details}
            """).strip(),
        },
        {
            "role": "user",
            "content": textwrap.dedent(f"""
                CURRENT PAGE STATE:
                URL: {page_state.get("url")}
                Title: {page_state.get("title")}
                Summary: {page_state.get("main_text", "N/A")[:1000]}...
                Elements Summary:
                {chr(10).join(elements_summary)}

                USER TASK: {task}

                Generate the JSON list of steps. Use 'task_hint' based on element text/role/context.
            """).strip(),
        },
    ]

    result = await _call_llm(messages, model=model, expect_json=True, temperature=0.0)

    plan_list = None
    if isinstance(result, list):
        plan_list = result
    elif isinstance(result, dict) and "steps" in result and isinstance(result["steps"], list):
        logger.warning("LLM wrapped plan in 'steps' key. Extracting list.")
        plan_list = result["steps"]

    # --- Modify subsequent checks to use plan_list ---
    if isinstance(result, dict) and "error" in result: # Check original result for errors
        raise ToolError(
            f"Macro planner LLM failed: {result['error']}", details=result.get("raw_response")
        )

    # Check the extracted plan_list
    if plan_list is None: # Covers non-list/non-dict or dict without 'steps'
        raw_response_preview = str(result)[:500]
        raise ToolError(
            f"Macro planner LLM returned unexpected format: {type(result)}. "
            f"Preview: '{raw_response_preview}...'",
            details={"raw_response": raw_response_preview}
        )

    # Validate plan structure using plan_list
    validated_plan = []
    for i, step in enumerate(plan_list): # Use plan_list here
        if not isinstance(step, dict) or "action" not in step:
            logger.warning(f"Invalid step format in macro plan (step {i + 1}): {step}")
            continue
        action = step.get("action")
        if action not in ALLOWED_ACTIONS:
            logger.warning(f"Invalid action '{action}' in macro plan (step {i + 1}): {step}")
            continue
        validated_plan.append(step)

    if not validated_plan:
        # Include original LLM response in error if validation fails
        raw_response_preview = str(result)[:500]
        raise ToolError(
            "Macro planner returned an empty or invalid plan.",
             details={"raw_response": raw_response_preview}
        )
    return validated_plan

async def run_macro(
    page: Page, task: str, max_rounds: int = 7, model: str = "gpt-4.1-mini"
) -> List[Dict[str, Any]]:
    """Executes a multi-step task using a plan-act loop."""
    all_step_results = []
    current_task = task
    for i in range(max_rounds):
        round_num = i + 1
        logger.info(f"Macro Round {round_num}/{max_rounds} | Task: {current_task[:100]}...")
        try:
            state = await get_page_state(page)
            if "error" in state:
                raise ToolError(f"Failed to get page state: {state['error']}")
            plan = await _plan_macro(state, current_task, model)
            await _log("macro_plan_generated", round=round_num, task=current_task, steps=plan)
            if not plan:
                logger.warning(f"Macro round {round_num}: Empty plan.")
                await _log("macro_plan_empty", round=round_num)
                break
            step_results = await run_steps(page, plan)  # run_steps handles internal errors/logging
            all_step_results.extend(step_results)
            finished = any(s.get("action") == "finish" and s.get("success") for s in step_results)
            last_step_failed = (
                step_results
                and not step_results[-1].get("success")
                and step_results[-1].get("action") not in ("wait", "finish", "extract")
            )  # Don't stop on non-critical failures
            if finished:
                await _log("macro_finish_action", round=round_num)
                logger.info(f"Macro finished via 'finish' action in round {round_num}.")
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
                return all_step_results  # Stop on critical failure
        except ToolError as e:
            await _log("macro_error_tool", round=round_num, task=current_task, error=str(e))
            logger.error(f"Macro round {round_num} failed with ToolError: {e}")
            all_step_results.append(
                {"action": "error", "success": False, "error": f"ToolError: {e}"}
            )
            return all_step_results
        except Exception as e:
            await _log("macro_error_unexpected", round=round_num, task=current_task, error=str(e))
            logger.error(f"Macro round {round_num} failed unexpectedly: {e}", exc_info=True)
            all_step_results.append(
                {"action": "error", "success": False, "error": f"Unexpected Error: {e}"}
            )
            return all_step_results
    await _log("macro_exceeded_rounds", max_rounds=max_rounds, task=task)
    logger.warning(f"Macro exceeded max rounds ({max_rounds}) for task: {task}")
    return all_step_results


# Autopilot Planner
_AVAILABLE_TOOLS = {  # Maps LLM tool name to (ToolClass method name, brief args schema)
    "search_web": (
        "search",
        {"query": "str", "engine": "str (bing|duckduckgo|yandex)", "max_results": "int"},
    ),
    "browse_page": ("browse_url", {"url": "str", "wait_for_selector": "Optional[str]"}),
    "click_element": (
        "click_and_extract",
        {"url": "str", "task_hint": "str", "wait_ms": "int"},
    ),  # Uses task_hint
    "fill_form": (
        "fill_form",
        {
            "url": "str",
            "form_fields": "[{'task_hint': str, 'text': str, 'enter': bool?}]",
            "submit_hint": "Optional[str]",
        },
    ),  # Uses task_hint
    "download_file_via_click": (
        "download_file",
        {"url": "str", "task_hint": "str", "dest_dir": "Optional[str]"},
    ),  # Uses task_hint
    "run_page_macro": ("execute_macro", {"url": "str", "task": "str", "max_rounds": "int"}),
    "download_all_pdfs_from_site": (
        "download_site_pdfs",
        {
            "start_url": "str",
            "dest_subfolder": "Optional[str]",
            "include_regex": "Optional[str]",
            "max_depth": "int",
            "max_pdfs": "int",
        },
    ),
    "collect_project_documentation": (
        "collect_documentation",
        {"package": "str", "max_pages": "int"},
    ),
    "process_urls_in_parallel": (
        "parallel_process",
        {"urls": "[str]", "action": "str ('get_state')"},
    ),  # Only get_state for now
}

_PLANNER_SYS = textwrap.dedent("""
    You are an AI orchestrator. Your goal is to create a plan to fulfill the user's TASK
    by selecting appropriate tools from the available list and specifying their arguments.
    The plan should be a JSON list of steps, where each step calls one tool.
    Analyze the TASK and any PRIOR RESULTS to decide the sequence of tool calls.
    Your response MUST be ONLY the JSON list of tool call objects `[...]`.
    Do not include explanations or markdown. Use 'task_hint' where specified.
""").strip()


async def _plan_autopilot(
    task: str, prior_results: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """Generates a multi-step plan using available tools for the autopilot feature."""
    tools_desc = {name: schema for name, (_, schema) in _AVAILABLE_TOOLS.items()}
    prior_summary = "None"
    if prior_results:
        summaries = []
        for i, res in enumerate(prior_results[-3:], start=max(0, len(prior_results) - 3) + 1):
            tool = res.get("tool", "?")
            success = res.get("success", False)
            outcome = "[OK]" if success else "[FAIL]"
            details = str(res.get("result", res.get("error", "")))[:150]
            summaries.append(f"Step {i}: {tool} -> {outcome} ({details}...)")
        prior_summary = "\n".join(summaries)

    user_prompt = (
        f"AVAILABLE TOOLS:\n{json.dumps(tools_desc, indent=2)}\n\n"
        f"PRIOR RESULTS SUMMARY:\n{prior_summary}\n\n"
        f"USER TASK:\n{task}\n\n"
        "Generate the JSON plan (list of steps) to complete the task. "
        'Each step: {"tool": "<name>", "args": {<args_dict>}}. Respond ONLY with JSON list.'
    )

    messages = [
        {"role": "system", "content": _PLANNER_SYS},
        {"role": "user", "content": user_prompt},
    ]
    response = await _call_llm(messages, expect_json=True, temperature=0.0, max_tokens=2048)

    if isinstance(response, dict) and "error" in response:
        raise ToolError(f"Autopilot planner failed: {response['error']}")
    if not isinstance(response, list):
        raise ToolError(f"Autopilot planner returned non-list: {type(response)}")

    # Validate plan structure
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
        if step.get("tool") not in _AVAILABLE_TOOLS:
            logger.warning(
                f"Unknown tool '{step.get('tool')}' in Autopilot plan (step {i + 1}): {step}"
            )
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
        step_result = step.copy()
        step_result["success"] = False  # Default to failure
        start_time = time.monotonic()

        if not action:
            step_result["error"] = "Missing 'action'."
            results.append(step_result)
            continue

        try:
            if action == "click":
                hint = step.get("task_hint")
                target_fallback = step.get("target")  # Keep target for fallback hint generation
                if not hint:
                    if target_fallback and (
                        target_fallback.get("name") or target_fallback.get("role")
                    ):
                        name = target_fallback.get("name", "")
                        role = target_fallback.get("role", "")
                        hint = f"Click the {role or 'element'}" + (
                            f" named '{name}'" if name else ""
                        )
                    else:
                        raise ToolInputError("Click step needs 'task_hint' or 'target'.")
                await smart_click(page, task_hint=hint, target_kwargs=target_fallback)
                step_result["success"] = True

            elif action == "type":
                hint = step.get("task_hint")
                target_fallback = step.get("target")
                text = step.get("text")
                if not hint:
                    if target_fallback and (
                        target_fallback.get("name") or target_fallback.get("role")
                    ):
                        name = target_fallback.get("name", "")
                        role = target_fallback.get("role", "input")
                        hint = f"Type into the {role or 'element'}" + (
                            f" named '{name}'" if name else ""
                        )
                    else:
                        raise ToolInputError("Type step needs 'task_hint' or 'target'.")
                if text is None:
                    raise ToolInputError("Missing 'text' for type action")
                await smart_type(
                    page,
                    task_hint=hint,
                    text=text,
                    press_enter=step.get("enter", False),
                    clear_before=step.get("clear_before", True),
                    target_kwargs=target_fallback,
                )
                step_result["success"] = True

            elif action == "wait":
                ms = step.get("ms")
                assert ms is not None, "Missing 'ms' for wait"
                await page.wait_for_timeout(int(ms))
                step_result["success"] = True

            elif action == "download":
                hint = step.get("task_hint")
                target_fallback = step.get("target")
                if not hint:
                    if target_fallback and (
                        target_fallback.get("name") or target_fallback.get("role")
                    ):
                        name = target_fallback.get("name", "")
                        role = target_fallback.get("role", "")
                        hint = (
                            "Download link/button"
                            + (f" named '{name}'" if name else "")
                            + (f" with role '{role}'" if role else "")
                        )
                    else:
                        raise ToolInputError("Download step needs 'task_hint' or 'target'.")
                download_outcome = await smart_download(
                    page, task_hint=hint, dest_dir=step.get("dest"), target_kwargs=target_fallback
                )
                step_result["result"] = download_outcome
                step_result["success"] = download_outcome.get("success", False)

            elif action == "extract":
                selector = step.get("selector")
                assert selector, "Missing 'selector' for extract"
                extracted = await page.eval_on_selector_all(
                    selector, "(elements => elements.map(el => el.innerText || el.textContent))"
                )
                step_result["result"] = [t.strip() for t in extracted if t and t.strip()]
                step_result["success"] = True  # Success even if no elements found

            elif action == "scroll":
                direction = step.get("direction")
                amount = step.get("amount_px")
                if not direction or direction not in ["up", "down", "top", "bottom"]:
                    step_result["error"] = (
                        f"Invalid scroll direction: '{direction}'. Must be one of: up, down, top, bottom"
                    )
                    step_result["success"] = False
                    logger.warning(f"Scroll step failed: {step_result['error']}")
                else:
                    if direction == "top":
                        await page.evaluate("() => window.scrollTo(0, 0)")
                    elif direction == "bottom":
                        await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                    elif direction == "up":
                        await page.evaluate("(px) => window.scrollBy(0, -px)", int(amount or 500))
                    elif direction == "down":
                        await page.evaluate("(px) => window.scrollBy(0, px)", int(amount or 500))
                    step_result["success"] = True

            elif action == "finish":
                logger.info("Macro execution: 'finish' action.")
                step_result["success"] = True

            else:
                raise ValueError(f"Unknown action '{action}'")

            step_result["duration_ms"] = int((time.monotonic() - start_time) * 1000)

        except (PlaywrightTimeoutError, ToolError, ValueError, AssertionError, Exception) as e:
            err_type = type(e).__name__
            step_result["error"] = f"{err_type} during action '{action}': {e}"
            logger.warning(f"Step failed: {step_result['error']}")
            # Let run_macro decide whether to stop based on step_result["success"] = False

        finally:
            await _log("macro_step_result", **step_result)
            results.append(step_result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 14. UNIVERSAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════

# Maintain state for UA rotation
_ua_rotation_count = 0
_user_agent_pools = {
    "bing": deque(
        [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.51",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.36",
        ]
    ),
    "duckduckgo": deque(
        [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
        ]
    ),
    "yandex": deque(
        [
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0",
        ]
    ),
}


@resilient(max_attempts=2, backoff=1.0)
async def search_web(
    query: str, engine: str = "bing", max_results: int = 10
) -> List[Dict[str, str]]:
    """Performs web search using Playwright against Bing, DuckDuckGo (HTML), or Yandex."""
    global _ua_rotation_count

    engine = engine.lower()
    if engine not in ("bing", "duckduckgo", "yandex"):
        raise ToolInputError("Invalid engine. Use 'bing', 'duckduckgo', or 'yandex'.")
    safe_query = re.sub(r"[^\w\s\-\.]", "", query).strip()
    assert safe_query, "Search query invalid."
    qs = urllib.parse.quote_plus(safe_query)
    nonce = random.randint(1000, 9999)
    urls = {
        "bing": f"https://www.bing.com/search?q={qs}&count={max_results}&form=QBLH&rdr=1&r={nonce}",
        "duckduckgo": f"https://html.duckduckgo.com/html/?q={qs}&r={nonce}",  # Fixed DuckDuckGo URL
        "yandex": f"https://yandex.com/search/?text={qs}&lr=10000&r={nonce}",  # lr=10000 for generic region
    }
    selectors = {
        "bing": {
            "item": "li.b_algo",
            "link": "h2 > a",
            "title": "h2 > a",
            "snippet": ".b_caption p",
            "alt_link": "a.tilk",
        },
        "duckduckgo": {
            "item": "div.result",              # Main result container
            "link": "a.result__a",             # Link and often title
            "title": "a.result__a",            # Title element (same as link)
            "snippet": "a.result__snippet",    # Snippet element
        },
        "yandex": {
            "item": "li.serp-item[data-cid]",
            "link": "a.Link_theme_outer",
            "title": "a.Link_theme_outer",
            "snippet": "div.OrganicTextContentSpan",
        },
    }
    sel = selectors[engine]

    # Rotate UA strings every 20 requests
    _ua_rotation_count += 1
    if _ua_rotation_count % 20 == 0:
        # Rotate by moving first item to the end
        for pool in _user_agent_pools.values():
            if len(pool) > 1:
                pool.append(pool.popleft())

    ua = _user_agent_pools[engine][0]  # Get the current UA for this engine

    ctx, _ = await get_browser_context(
        use_incognito=True, context_args={"user_agent": ua, "locale": "en-US"}
    )
    page = None
    try:
        page = await ctx.new_page()
        await _log("search_start", engine=engine, query=query, url=urls[engine])
        await page.goto(urls[engine], wait_until="domcontentloaded", timeout=30000)

        # Handle meta refresh redirects for DuckDuckGo
        if engine == "duckduckgo":
            try:
                meta_refresh = await page.query_selector('meta[http-equiv="refresh"]')
                if meta_refresh:
                    content = await meta_refresh.get_attribute("content")
                    if content and "url=" in content.lower():
                        match = re.search(r'url=([^"]+)', content, re.IGNORECASE)
                        if match:
                            redirect_url = match.group(1)
                            logger.info(f"Following meta refresh to: {redirect_url}")
                            await page.goto(redirect_url, wait_until="domcontentloaded", timeout=20000)
                            await asyncio.sleep(0.5) # Small pause after redirect
            except PlaywrightException as e:
                logger.warning(f"Error checking/following meta refresh: {e}")

        try:
            await page.wait_for_selector(sel["item"], state="visible", timeout=10000)
        except PlaywrightTimeoutError as e:
            captcha_found = await page.evaluate(
                "() => document.body.innerText.toLowerCase().includes('captcha') || document.querySelector('iframe[title*=captcha]')"
            )
            if captcha_found:
                await _log("search_captcha", engine=engine, query=query)
                raise ToolError(f"{engine} CAPTCHA.", error_code="captcha_detected") from e
            await _log(
                "search_no_results_selector", engine=engine, query=query, selector=sel["item"]
            )
            return []
        await asyncio.sleep(random.uniform(0.5, 1.5))
        # Try dismissing common consent banners
        consent_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Agree")',
            'button:has-text("Consent")',
            'button[id*="consent"]',
        ]
        for btn_sel in consent_selectors:
            try:
                await page.locator(btn_sel).first.click(timeout=1000)
                await asyncio.sleep(0.3)
                break
            except PlaywrightException:
                pass

        results = await page.evaluate(
            """
            (args) => {
                const sel = args.sel; // Unpack from single arg
                const max_results = args.max_results; // Unpack from single arg
                const items = Array.from(document.querySelectorAll(sel.item));
                const results = [];
                for (let i = 0; i < Math.min(items.length, max_results); i++) {
                    const item = items[i];
                    try {
                        let linkEl = item.querySelector(sel.link) || (sel.alt_link ? item.querySelector(sel.alt_link) : null);
                        let titleEl = item.querySelector(sel.title) || linkEl; // Fallback title to link text
                        let snippetEl = item.querySelector(sel.snippet);
                        const url = linkEl ? linkEl.href : null;

                        // Get text efficiently, limiting length to reduce payload
                        let title = '';
                        if (titleEl) {
                            title = titleEl.textContent.trim();
                            title = title.substring(0, 300); // Limit title length
                        }

                        let snippet = '';
                        if (snippetEl) {
                            snippet = snippetEl.textContent.trim();
                            snippet = snippet.substring(0, 300); // Limit snippet length
                        }

                        if (url && url.trim() && !url.startsWith('javascript:')) {
                            title = title.replace(/[\\n\\r\\t]+/g, ' ').replace(/\\s{2,}/g, ' ').trim();
                            snippet = snippet.replace(/[\\n\\r\\t]+/g, ' ').replace(/\\s{2,}/g, ' ').trim();
                            if (title || snippet) { results.push({ url: url.trim(), title, snippet }); }
                        }
                    } catch (e) { /* Ignore single item error */ }
                }
                return results;
            }
        """,
            {"sel": sel, "max_results": max_results}, # Pass as a single dictionary argument
        )

        await _log("search_complete", engine=engine, query=query, num_results=len(results))
        return results
    except PlaywrightException as e:
        await _log("search_error_playwright", engine=engine, query=query, error=str(e))
        raise ToolError(f"Playwright error during search: {e}") from e
    except Exception as e:
        await _log("search_error_unexpected", engine=engine, query=query, error=str(e))
        raise ToolError(f"Unexpected error during search: {e}") from e
    finally:
        if page:
            await page.close()
        # Fixed context leak - always close context even if page creation fails
        await ctx.close()


# ══════════════════════════════════════════════════════════════════════════════
# 15.  TOOL CLASS FOR MCP SERVER INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
class SmartBrowserTool(BaseTool):
    """Advanced web automation tool using Playwright and Enhanced Locator."""

    tool_name = "smart_browser"
    description = "Performs advanced web automation: browse, click, type, search, download, run macros, collect docs."

    def __init__(self, mcp_server):
        super().__init__(mcp_server)
        self.tab_pool = tab_pool  # Use global instance
        self._last_activity = time.monotonic()
        self._inactivity_monitor_task: Optional[asyncio.Task] = None
        self._init_lock = asyncio.Lock()
        self._is_initialized = False
        self._config_cache = None  # Cache for config to avoid repeated loading
        # Check if running within MCP lifespan context (best effort)
        is_server_context = (
            hasattr(mcp_server, "mcp") and hasattr(mcp_server.mcp, "lifespan")
            if hasattr(mcp_server, "mcp")
            else hasattr(mcp_server, "lifespan")
        )
        if is_server_context:
            logger.info("SmartBrowserTool in server context. Async init via lifespan/first use.")
        else:
            logger.info("SmartBrowserTool outside server context. Async init on first use.")

    async def _ensure_initialized(self):
        """
        Ensure async components are ready, load configuration into global variables,
        and prepare internal storage directories using the FileSystemTool.
        This method is idempotent.
        """
        # Define all relevant globals that this function modifies or uses
        global _sb_state_key_b64_global, _sb_max_tabs_global, _sb_tab_timeout_global
        global _sb_inactivity_timeout_global, _headless_mode_global, _vnc_enabled_global
        global _vnc_password_global, _proxy_pool_str_global, _proxy_allowed_domains_str_global
        global _vault_allowed_paths_str_global, _max_widgets_global, _max_section_chars_global
        global _dom_fp_limit_global, _llm_model_locator_global, _retry_after_fail_global
        global _seq_cutoff_global, _area_min_global, _high_risk_domains_set_global
        global _locator_cache_cleanup_task_handle
        global _thread_pool # Include thread pool for potential reconfiguration
        global _SB_INTERNAL_BASE_PATH_STR, _STATE_FILE, _LOG_FILE, _CACHE_DB, _READ_JS_CACHE

        if self._is_initialized:
            return

        async with self._init_lock:
            if self._is_initialized:
                return

            logger.info("Performing first-time async initialization of SmartBrowserTool...")

            # --- Step 1: Load ALL SB config into globals ---
            try:
                if not self._config_cache:
                    config = get_config()
                    sb_config: SmartBrowserConfig = config.smart_browser
                    self._config_cache = sb_config
                else:
                    sb_config = self._config_cache

                _sb_state_key_b64_global = sb_config.sb_state_key_b64 or _sb_state_key_b64_global
                _sb_max_tabs_global = sb_config.sb_max_tabs or _sb_max_tabs_global
                _sb_tab_timeout_global = sb_config.sb_tab_timeout or _sb_tab_timeout_global
                _sb_inactivity_timeout_global = sb_config.sb_inactivity_timeout or _sb_inactivity_timeout_global
                _headless_mode_global = sb_config.headless_mode if sb_config.headless_mode is not None else _headless_mode_global
                _vnc_enabled_global = sb_config.vnc_enabled if sb_config.vnc_enabled is not None else _vnc_enabled_global
                _vnc_password_global = sb_config.vnc_password or _vnc_password_global
                _proxy_pool_str_global = sb_config.proxy_pool_str or _proxy_pool_str_global
                _proxy_allowed_domains_str_global = sb_config.proxy_allowed_domains_str or _proxy_allowed_domains_str_global
                _vault_allowed_paths_str_global = sb_config.vault_allowed_paths_str or _vault_allowed_paths_str_global
                _max_widgets_global = sb_config.max_widgets or _max_widgets_global
                _max_section_chars_global = sb_config.max_section_chars or _max_section_chars_global
                _dom_fp_limit_global = sb_config.dom_fp_limit or _dom_fp_limit_global
                _llm_model_locator_global = sb_config.llm_model_locator or _llm_model_locator_global
                _retry_after_fail_global = sb_config.retry_after_fail if sb_config.retry_after_fail is not None else _retry_after_fail_global
                _seq_cutoff_global = sb_config.seq_cutoff if sb_config.seq_cutoff is not None else _seq_cutoff_global
                _area_min_global = sb_config.area_min or _area_min_global
                _high_risk_domains_set_global = sb_config.high_risk_domains_set or _high_risk_domains_set_global

                logger.info("Smart Browser configuration loaded into global variables.")

                # Update dependent derived globals AFTER loading primary strings
                _update_proxy_settings() # Assumes this function uses the globals set above
                _update_vault_paths()  # Assumes this function uses the globals set above

                # Reconfigure thread pool based on potentially updated max_tabs
                # Check if the pool instance needs changing (e.g., if max_workers differs)
                # This is a simplified check; a more robust check might compare desired vs current max_workers
                current_max_workers = getattr(_thread_pool, '_max_workers', min(32, (_cpu_count or 1) * 2 + 4))
                desired_max_workers = min(32, _sb_max_tabs_global * 2) # Use updated global
                if current_max_workers != desired_max_workers:
                    logger.info(f"Reconfiguring thread pool max_workers from {current_max_workers} to {desired_max_workers} based on SB_MAX_TABS.")
                    _thread_pool.shutdown(wait=True) # Wait for existing tasks
                    _thread_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=desired_max_workers, thread_name_prefix="sb_worker"
                    )

            except AttributeError as e:
                logger.error(f"Error accessing expected Smart Browser config attributes: {e}. Using hardcoded defaults.")
                # Ensure derived settings are updated even with defaults
                _update_proxy_settings()
                _update_vault_paths()
            except Exception as e:
                logger.error(f"Unexpected error loading Smart Browser config: {e}. Using defaults.")
                # Ensure derived settings are updated even with defaults
                _update_proxy_settings()
                _update_vault_paths()
            # --- End Config Loading ---

            # --- Step 2: Prepare Internal Storage Directory using FileSystemTool ---
            try:
                # Define the *relative* path structure within an allowed base.
                # Assumes FileSystemTool is configured with '/home/ubuntu/ultimate_mcp_server/storage' as an allowed base.
                internal_storage_relative_path = "storage/smart_browser_internal"

                logger.info(f"Ensuring internal storage directory exists: '{internal_storage_relative_path}' using filesystem tool.")
                create_dir_result = await create_directory(path=internal_storage_relative_path)

                if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
                    error_detail = create_dir_result.get('error', 'Unknown') if isinstance(create_dir_result, dict) else 'Invalid response'
                    raise ToolError(f"Failed to prepare internal storage directory '{internal_storage_relative_path}'. Filesystem tool error: {error_detail}")

                resolved_base_path_str = create_dir_result.get("path")
                if not resolved_base_path_str:
                    raise ToolError(f"FileSystemTool create_directory did not return a path for '{internal_storage_relative_path}'.")

                _SB_INTERNAL_BASE_PATH_STR = resolved_base_path_str # Store the validated absolute path string
                internal_base_path = Path(_SB_INTERNAL_BASE_PATH_STR)

                # Define the absolute paths for internal files using the resolved base
                _STATE_FILE = internal_base_path / "storage_state.enc"
                _LOG_FILE = internal_base_path / "audit.log"
                _CACHE_DB = internal_base_path / "locator_cache2.db"
                _READ_JS_CACHE = internal_base_path / "readability.js"

                logger.info(f"Smart Browser internal files will use base: {internal_base_path}")

                # Initialize components that depend on these paths NOW that they are defined
                # Ensure these helper functions exist and use the global path variables correctly
                _init_last_hash() # Depends on _LOG_FILE
                _init_locator_cache_db_sync() # Depends on _CACHE_DB

            except ToolError as e:
                logger.critical(f"CRITICAL: Failed to initialize Smart Browser internal storage: {e}", exc_info=True)
                self._is_initialized = False # Mark as failed
                # Consider re-raising a more specific error if the tool cannot function without storage
                # raise RuntimeError("Smart Browser cannot initialize internal storage via FileSystemTool.") from e
                return # Stop initialization
            except Exception as e:
                logger.critical(f"CRITICAL: Unexpected error initializing Smart Browser internal storage: {e}", exc_info=True)
                self._is_initialized = False
                return # Stop initialization
            # --- End Internal Storage Preparation ---

            # --- Step 3: Initialize Browser Context (uses loaded globals) ---
            try:
                await get_browser_context() # Ensure browser/context are ready
                logger.info("Playwright browser and context initialized successfully.")
            except Exception as e:
                logger.critical(f"CRITICAL: Failed to initialize Playwright browser context: {e}", exc_info=True)
                self._is_initialized = False
                # Consider cleanup if browser started but context failed? get_browser_context might need internal cleanup logic.
                return # Stop initialization

            # --- Step 4: Start Background Tasks ---
            # Start inactivity monitor (uses global timeout value)
            if self._inactivity_monitor_task is None or self._inactivity_monitor_task.done():
                timeout_sec = _sb_inactivity_timeout_global
                if timeout_sec > 0:
                    logger.info(f"Starting browser inactivity monitor ({timeout_sec}s)...")
                    self._inactivity_monitor_task = asyncio.create_task(
                        self._inactivity_monitor(timeout_sec)
                    )
                else:
                    logger.info("Inactivity monitor disabled (timeout <= 0).")

            # Start locator cache cleanup task (uses global handle)
            if (_locator_cache_cleanup_task_handle is None or
                    _locator_cache_cleanup_task_handle.done()):
                # Use a configurable interval eventually, default to daily for now
                cleanup_interval_sec = 24 * 60 * 60 # Daily
                logger.info(f"Starting periodic locator cache cleanup task (interval: {cleanup_interval_sec}s)...")
                _locator_cache_cleanup_task_handle = asyncio.create_task(
                    _locator_cache_cleanup_task(interval_seconds=cleanup_interval_sec)
                )
            # --- End Background Tasks ---

            # --- Finalize Initialization ---
            self._is_initialized = True
            logger.info("SmartBrowserTool async components initialized successfully.")

    def _update_activity(self):
        self._last_activity = time.monotonic()

    async def _inactivity_monitor(self, timeout_seconds: int):
        """Monitors browser inactivity and triggers shutdown."""
        check_interval = 60
        logger.info(f"Inactivity monitor started. Timeout: {timeout_seconds}s.")
        while True:
            await asyncio.sleep(check_interval)
            async with _playwright_lock():
                browser_active = _browser is not None and _browser.is_connected()
            if not browser_active:
                logger.info("Inactivity monitor: Browser closed. Stopping monitor.")
                break
            idle_time = time.monotonic() - self._last_activity
            if idle_time > timeout_seconds:
                logger.info(f"Browser inactive for {idle_time:.1f}s. Initiating shutdown.")
                try:
                    await _initiate_shutdown()
                except Exception as e:
                    logger.error(f"Error during auto-shutdown: {e}", exc_info=True)
                break  # Exit monitor after triggering shutdown
        logger.info("Inactivity monitor stopped.")

    @tool(name="smart_browser.browse")
    @with_tool_metrics
    @with_error_handling
    async def browse_url(
        self, url: str, wait_for_selector: Optional[str] = None, wait_for_navigation: bool = True
    ) -> Dict[str, Any]:
        """Navigates to a URL, waits for load, and returns the page state."""
        await self._ensure_initialized()
        self._update_activity()
        if not isinstance(url, str) or not url.strip():
            raise ToolInputError("URL cannot be empty.")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        ctx, _ = await get_browser_context()
        # Check global proxy config and allowed domains list
        if _PROXY_CONFIG_DICT and _PROXY_ALLOWED_DOMAINS_LIST is not None and not _is_domain_allowed_for_proxy(url):
            proxy_server = _PROXY_CONFIG_DICT.get("server", "Configured Proxy") # Use global config dict
            error_msg = (
                f"Navigation blocked by PROXY_ALLOWED_DOMAINS for '{url}' via {proxy_server}."
            )
            await _log("browse_fail_proxy_disallowed", url=url, proxy=proxy_server)
            raise ToolError(error_msg, error_code="proxy_domain_disallowed")

        async with _tab_context(ctx) as page:
            await _log("navigate_start", url=url)
            try:
                wait_until_state = "networkidle" if wait_for_navigation else "domcontentloaded"
                await page.goto(url, wait_until=wait_until_state, timeout=60000)
                if wait_for_selector:
                    try:
                        await page.wait_for_selector(
                            wait_for_selector, state="visible", timeout=15000
                        )
                        await _log("navigate_wait_selector_ok", selector=wait_for_selector)
                    except PlaywrightTimeoutError:
                        logger.warning(
                            f"Timeout waiting for selector '{wait_for_selector}' at {url}"
                        )
                        await _log("navigate_wait_selector_timeout", selector=wait_for_selector)
                await _pause(page, (50, 200))
                state = await get_page_state(page)  # Uses new page map state
                await _log("navigate_success", url=url, title=state.get("title"))
                return {"success": True, "page_state": state}
            except PlaywrightException as e:
                await _log("navigate_fail_playwright", url=url, error=str(e))
                raise ToolError(f"Navigation failed for {url}: {e}") from e
            except Exception as e:
                await _log("navigate_fail_unexpected", url=url, error=str(e))
                raise ToolError(f"Unexpected error browsing {url}: {e}") from e

    @tool(name="smart_browser.click")
    @with_tool_metrics
    @with_error_handling
    async def click_and_extract(
        self,
        url: str,
        target: Optional[Dict[str, Any]] = None,
        task_hint: Optional[str] = None,
        wait_ms: int = 1000,
    ) -> Dict[str, Any]:
        """Navigates, clicks (using hint or target), waits, returns page state."""
        await self._ensure_initialized()
        self._update_activity()
        if not task_hint:
            if target and (target.get("name") or target.get("role")):
                name = target.get("name", "")
                role = target.get("role", "")
                task_hint = f"Click the {role or 'element'}" + (f" named '{name}'" if name else "")
            else:
                raise ToolInputError("Requires 'task_hint' or 'target' dict.")

        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            await _log("click_extract_navigate", url=url)
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await smart_click(
                page, task_hint=task_hint, target_kwargs=target, timeout_ms=10000
            )  # smart_click handles errors/logging
            if wait_ms > 0:
                await page.wait_for_timeout(wait_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except PlaywrightTimeoutError:
                logger.debug("Network idle wait timeout after click.")
            await _pause(page, (50, 200))
            final_state = await get_page_state(page)
            await _log("click_extract_success", url=url, hint=task_hint)
            return {"success": True, "page_state": final_state}

    @tool(name="smart_browser.fill_form")
    @with_tool_metrics
    @with_error_handling
    async def fill_form(
        self,
        url: str,
        form_fields: List[Dict[str, Any]],
        submit_hint: Optional[str] = None,
        submit_target: Optional[Dict[str, Any]] = None,
        wait_after_submit_ms: int = 2000,
    ) -> Dict[str, Any]:
        """Navigates, fills form fields (using hints), optionally submits, returns page state."""
        await self._ensure_initialized()
        self._update_activity()
        if not form_fields or not isinstance(form_fields, list):
            raise ToolInputError("'form_fields' must be a non-empty list.")
        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            await _log("fill_form_navigate", url=url)
            await page.goto(url, wait_until="networkidle", timeout=60000)
            try:
                # Wait for a common form element or the specific input if known
                await page.wait_for_selector('form, input[type="text"]', state='visible', timeout=5000)
                logger.debug("Found form elements, proceeding with field filling.")
            except PlaywrightTimeoutError:
                logger.warning("Did not find expected form elements quickly. Proceeding anyway.")

            for i, field in enumerate(form_fields):
                hint = field.get("task_hint")
                target_fallback = field.get("target")
                text = field.get("text")
                if not hint:
                    if target_fallback and (
                        target_fallback.get("name") or target_fallback.get("role")
                    ):
                        name = target_fallback.get("name", "")
                        role = target_fallback.get("role", "input")
                        hint = f"Input field for {name or role}"
                    else:
                        raise ToolInputError(f"Field {i} needs 'task_hint' or 'target'.")
                if text is None:
                    raise ToolInputError(f"Field {i} missing 'text'.")
                await _log("fill_form_field", index=i, hint=hint)
                await smart_type(
                    page,
                    task_hint=hint,
                    text=text,
                    press_enter=field.get("enter", False),
                    clear_before=field.get("clear_before", True),
                    target_kwargs=target_fallback,
                    timeout_ms=5000,
                )
                await _pause(page, (50, 150))

            if submit_hint or submit_target:
                final_submit_hint = submit_hint
                if not final_submit_hint:
                    if submit_target and (submit_target.get("name") or submit_target.get("role")):
                        name = submit_target.get("name", "")
                        role = submit_target.get("role", "button")
                        final_submit_hint = f"Submit button {name or role}"
                    else:
                        raise ToolInputError(
                            "Submit action needs 'submit_hint' or 'submit_target'."
                        )
                await _log("fill_form_submit", hint=final_submit_hint)
                await smart_click(
                    page, task_hint=final_submit_hint, target_kwargs=submit_target, timeout_ms=10000
                )
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except PlaywrightTimeoutError:
                    logger.debug("Network idle wait timeout after submit.")
                if wait_after_submit_ms > 0:
                    await page.wait_for_timeout(wait_after_submit_ms)

            await _pause(page, (100, 300))
            final_state = await get_page_state(page)
            await _log(
                "fill_form_success",
                url=url,
                num_fields=len(form_fields),
                submitted=bool(submit_hint or submit_target),
            )
            return {"success": True, "page_state": final_state}

    @tool(name="smart_browser.search")
    @with_tool_metrics
    @with_error_handling
    async def search(
        self, query: str, engine: str = "bing", max_results: int = 10
    ) -> Dict[str, Any]:
        """Performs a web search and returns results."""
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
        self,
        url: str,
        target: Optional[Dict[str, Any]] = None,
        task_hint: Optional[str] = None,
        dest_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Navigates, clicks (using hint or target) to download, saves file, returns info."""
        await self._ensure_initialized()
        self._update_activity()
        if not task_hint:
            if target and (target.get("name") or target.get("role")):
                name = target.get("name", "")
                role = target.get("role", "")
                task_hint = f"Download link/button {name or role}"
            else:
                raise ToolInputError("Requires 'task_hint' or 'target' dict.")

        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            await _log("download_navigate", url=url)
            await page.goto(url, wait_until="networkidle", timeout=60000)
            download_info = await smart_download(
                page, task_hint=task_hint, dest_dir=dest_dir, target_kwargs=target
            )
            if not download_info.get("success"):
                raise ToolError(
                    f"Download failed: {download_info.get('error', 'Unknown')}",
                    details=download_info,
                )
            return {"success": True, "download": download_info}


    @tool(name="smart_browser.download_site_pdfs")
    @with_tool_metrics
    @with_error_handling
    async def download_site_pdfs(
        self,
        start_url: str,
        dest_subfolder: Optional[str] = None,
        include_regex: Optional[str] = None,
        max_depth: int = 2,
        max_pdfs: int = 100,
        max_pages_crawl: int = 500,
        rate_limit_rps: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Crawls site, finds PDFs, downloads them directly using FileSystemTool for path management.

        Args:
            start_url: The URL to start crawling from.
            dest_subfolder: Optional subdirectory name for downloads (relative to an allowed base).
            include_regex: Optional regex to filter PDF URLs.
            max_depth: Maximum crawl depth.
            max_pdfs: Maximum number of PDFs to download.
            max_pages_crawl: Maximum number of pages to visit during crawl.
            rate_limit_rps: Download rate limit in requests per second.

        Returns:
            Dictionary with results, including count, destination directory, and individual file results.
        """
        await self._ensure_initialized()
        self._update_activity()
        final_dest_dir_str: Optional[str] = None # Track the validated dir path

        try:
            # --- Determine and Prepare Download Directory using FileSystemTool ---
            if dest_subfolder:
                safe_subfolder = _slugify(dest_subfolder, 50)
            else:
                safe_subfolder = _slugify(urlparse(start_url).netloc, 50) or "downloaded_pdfs" # Default folder name

            # Construct relative path for FileSystemTool (assuming a base like 'smart_browser_root' or similar)
            # Adjust 'smart_browser_root' if your FS tool uses a different base concept
            download_base_relative = "smart_browser_root/downloads"
            # --- OR if using absolute path directly (requires _HOME base to be allowed): ---
            # download_base_relative = str(_HOME / "downloads")

            dest_dir_relative_path = f"{download_base_relative}/{safe_subfolder}"

            logger.info(f"Ensuring download directory exists: '{dest_dir_relative_path}' using filesystem tool.")
            create_dir_result = await create_directory(path=dest_dir_relative_path)

            if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
                error_detail = create_dir_result.get('error', 'Unknown') if isinstance(create_dir_result, dict) else 'Invalid response'
                raise ToolError(f"Failed to prepare download directory '{dest_dir_relative_path}'. Filesystem tool error: {error_detail}")

            # Use the actual absolute path returned by the tool for passing to helpers
            final_dest_dir_str = create_dir_result.get("path")
            if not final_dest_dir_str:
                logger.warning(f"create_directory did not return a path for '{dest_dir_relative_path}'. Using input path.")
                final_dest_dir_str = dest_dir_relative_path # Fallback

            logger.info(f"Download directory confirmed/created at: {final_dest_dir_str}")
            # Removed direct mkdir/chmod calls

        except ToolError as e:
            logger.error(f"Error preparing download directory '{dest_dir_relative_path}': {e}", exc_info=True)
            raise ToolError(f"Could not prepare download directory: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error preparing download directory: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred preparing download directory: {str(e)}") from e
        # --- End Directory Preparation ---

        logger.info(f"Starting PDF crawl from: {start_url}")
        pdf_urls = await crawl_for_pdfs( # crawl_for_pdfs uses httpx, no FS access
            start_url, include_regex, max_depth, max_pdfs, max_pages_crawl, rate_limit_rps=5.0 # Use faster rate for crawl itself
        )

        if not pdf_urls:
            logger.info("No PDF URLs found during crawl.")
            return {"success": True, "pdf_count": 0, "dest_dir": final_dest_dir_str, "files": []}

        logger.info(f"Found {len(pdf_urls)} PDFs. Starting downloads (Rate: {rate_limit_rps}/s)...")
        limiter = RateLimiter(rate_limit_rps) # Use specified rate for downloads

        # --- Define Download Task ---
        async def download_task(url, seq):
            await limiter.acquire()
            # Pass the validated *string* path to the helper
            return await _download_file_direct(url, final_dest_dir_str, seq)
        # --- End Download Task ---

        # --- Run Downloads Concurrently ---
        download_tasks = [
            asyncio.create_task(download_task(url, i + 1)) for i, url in enumerate(pdf_urls)
        ]
        results = await asyncio.gather(*download_tasks) # Exceptions handled within _download_file_direct
        # --- End Downloads ---

        successful_downloads = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_downloads = [r for r in results if not isinstance(r, dict) or not r.get("success")]

        log_details = {
            "start_url": start_url, "found": len(pdf_urls),
            "successful": len(successful_downloads), "failed": len(failed_downloads),
            "dest_dir": final_dest_dir_str
        }
        # Optionally add first few errors to log details if needed
        if failed_downloads:
            log_details["errors_preview"] = [f"{res.get('url')}: {res.get('error')}" for res in failed_downloads[:3]]

        await _log("download_site_pdfs_complete", **log_details)

        # Return overall success=True, but include individual results
        return {
            "success": True, # Indicates the overall orchestration succeeded
            "pdf_count": len(successful_downloads),
            "failed_count": len(failed_downloads),
            "dest_dir": final_dest_dir_str,
            "files": results, # List containing dicts for each attempted download
        }


    @tool(name="smart_browser.collect_documentation")
    @with_tool_metrics
    @with_error_handling # This decorator should handle injecting/making filesystem tools available
    async def collect_documentation(
        self, package: str, max_pages: int = 40, rate_limit_rps: float = 2.0
    ) -> Dict[str, Any]:
        """
        Finds the documentation site for a package, crawls specified pages,
        extracts readable text content, and saves the combined content to a file
        within an allowed directory using the MCP FileSystem tools.

        Args:
            package: The name of the package to find documentation for.
            max_pages: Maximum number of documentation pages to crawl and extract.
            rate_limit_rps: Requests per second limit for crawling the docs site.

        Returns:
            A dictionary containing the success status, package name, number of pages
            collected, the absolute path to the saved file (if successful),
            and the root URL found for the documentation.
        """
        await self._ensure_initialized()
        self._update_activity()

        # 1. Find Documentation Root URL
        try:
            docs_root = await _pick_docs_root(package)
            if not docs_root:
                # _pick_docs_root already logs extensively, raise ToolError directly
                raise ToolError(f"Could not find a suitable documentation site for package '{package}'.")
        except ToolError as e:
            # Propagate ToolErrors from search/finding phase
            logger.error(f"Failed to find docs root for '{package}': {e}")
            raise # Re-raise the original error
        except Exception as e:
            # Catch unexpected errors during root finding
            logger.error(f"Unexpected error finding docs root for '{package}': {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred while searching for the docs root: {str(e)}") from e

        logger.info(f"Found potential docs root: {docs_root}. Starting crawl...")

        # 2. Crawl the Documentation Site
        try:
            pages_content = await crawl_docs_site(
                docs_root, max_pages=max_pages, rate_limit_rps=rate_limit_rps
            )
        except ToolInputError as e: # Catch errors like invalid URL from crawl_docs_site
            raise ToolInputError(f"Invalid documentation root URL '{docs_root}': {e}", param_name="docs_root", provided_value=docs_root) from e
        except Exception as e: # Catch unexpected crawl errors
            logger.error(f"Unexpected error crawling docs site {docs_root}: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred while crawling the documentation site: {str(e)}") from e

        if not pages_content:
            logger.info(f"No readable content collected from '{docs_root}' for package '{package}'.")
            return {
                "success": True, # Succeeded in crawling, but found nothing
                "package": package,
                "pages_collected": 0,
                "file_path": None,
                "root_url": docs_root,
                "message": f"Successfully crawled '{docs_root}', but no readable content pages were collected (or max_pages was 0).",
            }

        logger.info(f"Collected content from {len(pages_content)} pages for package '{package}'.")

        # 3. Prepare Output Path and Directory using FileSystemTool
        # Define a subdirectory within an assumed allowed base path.
        # This relies on FileSystemTool being configured with a suitable base (e.g., 'browser_demo_outputs').
        # The exact base directory isn't needed here, only the relative path for the tool.
        output_subdir_name = "docs_collected"
        # Construct the relative path for the directory creation tool
        # Choose a base directory known to be allowed by FileSystemConfig
        # For demo, let's assume 'browser_demo_outputs' is allowed. If not, adjust this.
        allowed_base_for_demo = "browser_demo_outputs"
        output_dir_relative_path = f"{allowed_base_for_demo}/{output_subdir_name}"

        try:
            # Ensure the output directory exists using the filesystem tool.
            # The tool should handle resolution relative to its allowed base paths.
            create_result = await create_directory(path=output_dir_relative_path)

            # Check the result carefully - it should return success status and potentially the resolved path
            if not isinstance(create_result, dict) or not create_result.get("success"):
                error_detail = create_result.get('error', 'Unknown') if isinstance(create_result, dict) else 'Invalid response from create_directory'
                logger.error(f"Filesystem tool failed to create directory '{output_dir_relative_path}'. Reason: {error_detail}")
                raise ToolError(f"Could not prepare output directory. Filesystem tool failed: {error_detail}")

            created_dir_path = create_result.get("path", output_dir_relative_path) # Get the actual path if returned
            logger.info(f"Ensured output directory exists via MCP tool: '{output_dir_relative_path}' resolved to '{created_dir_path}'")

        except ToolError as e:
            # Catch errors specifically from the create_directory tool call
            logger.error(f"Failed to ensure directory '{output_dir_relative_path}' exists using filesystem tool: {e}", exc_info=True)
            # Re-raise as a clear ToolError indicating failure to prepare output location
            raise ToolError(f"Could not prepare output directory '{output_dir_relative_path}': {str(e)}") from e
        except Exception as e:
            # Catch any other unexpected errors during directory preparation
            logger.error(f"Unexpected error preparing output directory '{output_dir_relative_path}': {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred preparing the output directory: {str(e)}") from e


        # 4. Prepare File Content
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_pkg = _slugify(package, 40)
        fname = f"{safe_pkg}_docs_{now}.txt"
        # Construct the relative path for the file writing tool
        fpath_relative = f"{output_dir_relative_path}/{fname}"

        sep = "\n\n" + ("=" * 80) + "\n\n"
        combined_content = f"# Docs for: {package}\n# Root: {docs_root}\n{sep}"
        try:
            combined_content += sep.join(
                # Ensure url and text are strings before joining
                f"## Page {i + 1}: {str(url)}\n\n{str(text).strip()}"
                for i, (url, text) in enumerate(pages_content)
            )
        except Exception as e:
            logger.error(f"Error combining documentation content for {package}: {e}", exc_info=True)
            raise ToolError(f"Internal error formatting documentation content: {str(e)}") from e


        # 5. Write File using FileSystemTool
        final_absolute_fpath = None
        try:
            # Use the write_file tool (or appropriate name from your filesystem module)
            write_result = await write_file(
                path=fpath_relative,
                content=combined_content,
                # Add overwrite=True if you want subsequent calls for the same package near the same time to overwrite
                # overwrite=True
            )
            # Check the result carefully
            if not isinstance(write_result, dict) or not write_result.get("success"):
                error_detail = write_result.get('error', 'Unknown') if isinstance(write_result, dict) else 'Invalid response from write_file'
                logger.error(f"Filesystem tool failed to write file '{fpath_relative}'. Reason: {error_detail}")
                raise ToolError(f"Could not write documentation file. Filesystem tool failed: {error_detail}")

            # --- Get the ACTUAL absolute path returned by the tool ---
            # This is crucial as the tool resolves the relative path against its base.
            final_absolute_fpath = write_result.get("path") # Use 'path' key, adjust if tool returns differently
            if not final_absolute_fpath:
                logger.warning(f"Filesystem tool write_file did not return the final absolute path for '{fpath_relative}'. Using relative path for logs/return value.")
                final_absolute_fpath = fpath_relative # Fallback, less ideal

            logger.info(f"Successfully wrote documentation to: {final_absolute_fpath} (requested relative: {fpath_relative})")

        except ToolError as e:
            # Catch errors specifically from the write_file tool call
            logger.error(f"Failed to write file '{fpath_relative}' using filesystem tool: {e}", exc_info=True)
            raise ToolError(f"Could not write documentation file '{fpath_relative}': {str(e)}") from e
        except Exception as e:
            # Catch any other unexpected errors during file writing
            logger.error(f"Unexpected error writing documentation file '{fpath_relative}': {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred writing the documentation file: {str(e)}") from e

        # 6. Log Success and Return Result
        await _log(
            "docs_collected_success",
            package=package,
            root=docs_root,
            pages=len(pages_content),
            file=str(final_absolute_fpath), # Log the actual path
        )

        return {
            "success": True,
            "package": package,
            "pages_collected": len(pages_content),
            "file_path": str(final_absolute_fpath), # Return the actual path
            "root_url": docs_root,
            "message": f"Successfully collected documentation for '{package}' ({len(pages_content)} pages) and saved to '{final_absolute_fpath}'."
        }


    @tool(name="smart_browser.run_macro")
    @with_tool_metrics
    @with_error_handling
    async def execute_macro(
        self,
        url: str,
        task: str,
        model: str = "gpt-4.1-mini",
        max_rounds: int = 7,
        timeout_seconds: int = 600,
    ) -> Dict[str, Any]:
        """Navigates to URL and executes a natural language task using LLM planner."""
        await self._ensure_initialized()
        self._update_activity()

        async def run_macro_inner():
            ctx, _ = await get_browser_context()
            async with _tab_context(ctx) as page:
                await _log("macro_navigate", url=url)
                try:
                    await page.goto(url, wait_until="networkidle", timeout=60000)
                except PlaywrightException as e:
                    raise ToolError(f"Macro nav failed: {e}") from e
                results = await run_macro(
                    page, task, max_rounds, model
                )  # Handles internal logs/errors
                try:
                    final_state = await get_page_state(page)
                except Exception as e:
                    final_state = {"error": f"Failed to get final state: {e}"}
                macro_success = bool(results) and all(
                    s.get("success", False) for s in results if s.get("action") != "error"
                )
                # More refined success check: did it finish or just stop without error?
                finished = any(s.get("action") == "finish" and s.get("success") for s in results)
                macro_success = (
                    macro_success or finished
                )  # Consider it success if it finished cleanly

                return {
                    "success": macro_success,
                    "task": task,
                    "steps": results,
                    "final_page_state": final_state,
                }

        try:
            return await asyncio.wait_for(run_macro_inner(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            await _log("macro_timeout", url=url, task=task, timeout=timeout_seconds)
            return {
                "success": False,
                "task": task,
                "error": f"Macro execution timed out after {timeout_seconds}s",
                "steps": [],
                "final_page_state": {"error": "Timeout occurred"},
            }


    @tool(name="smart_browser.autopilot")
    @with_tool_metrics
    @with_error_handling
    async def autopilot(
        self,
        task: str,
        scratch_subdir: str = "autopilot_runs",
        max_steps: int = 10,
        timeout_seconds: int = 1800,
    ) -> Dict[str, Any]:
        """Executes a complex multi-step task using LLM planning and available tools."""
        await self._ensure_initialized()
        self._update_activity()
        final_scratch_dir_str: Optional[str] = None
        log_path: Optional[Path] = None # Use Path object for log file handling

        try:
            # --- Prepare Scratch Directory using FileSystemTool ---
            # Construct relative path for FileSystemTool
            # Example: Assumes 'smart_browser_root' is implicitly mapped or relative to an allowed base
            scratch_base_relative = "smart_browser_root/scratch" # Or another configured allowed path
            # --- OR if using absolute path directly (requires _HOME base to be allowed): ---
            # scratch_base_relative = str(_HOME / "scratch")

            scratch_dir_relative_path = f"{scratch_base_relative}/{scratch_subdir}"

            logger.info(f"Ensuring autopilot scratch directory exists: '{scratch_dir_relative_path}' using filesystem tool.")
            create_dir_result = await create_directory(path=scratch_dir_relative_path)

            if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
                error_detail = create_dir_result.get('error', 'Unknown') if isinstance(create_dir_result, dict) else 'Invalid response'
                raise ToolError(f"Failed to prepare autopilot scratch directory '{scratch_dir_relative_path}'. Filesystem tool error: {error_detail}")

            final_scratch_dir_str = create_dir_result.get("path")
            if not final_scratch_dir_str:
                logger.warning(f"create_directory did not return a path for '{scratch_dir_relative_path}'. Using input path.")
                final_scratch_dir_str = scratch_dir_relative_path # Fallback

            final_scratch_dir_path = Path(final_scratch_dir_str) # Convert to Path for log path construction
            logger.info(f"Autopilot scratch directory confirmed/created at: {final_scratch_dir_path}")
            # --- End Directory Preparation ---

            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_filename = f"autopilot_{run_id}.jsonl"
            log_path = final_scratch_dir_path / log_filename # Construct log path using Path obj
            logger.info(f"Autopilot run '{run_id}' started. Task: '{task[:100]}...'. Log: {log_path}")

        except ToolError as e:
            logger.error(f"Error preparing autopilot scratch directory '{scratch_dir_relative_path}': {e}", exc_info=True)
            raise ToolError(f"Could not prepare scratch directory for autopilot run: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error preparing autopilot scratch directory: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred preparing the scratch directory: {str(e)}") from e

        # --- Autopilot Inner Logic ---
        async def autopilot_inner():
            all_results = []
            current_task = task

            try:
                current_plan = await _plan_autopilot(current_task)
                step_num = 0
                while step_num < max_steps and current_plan:
                    step_num += 1
                    step_to_execute = current_plan[0]
                    tool_name = step_to_execute.get("tool")
                    args = step_to_execute.get("args", {})
                    step_log = {"step": step_num, "tool": tool_name, "args": args, "success": False}

                    # --- Tool Execution Logic (remains largely the same) ---
                    if tool_name not in _AVAILABLE_TOOLS:
                        step_log["error"] = f"Unknown tool '{tool_name}'."
                        current_plan.pop(0)
                    else:
                        method_name = _AVAILABLE_TOOLS[tool_name][0]
                        try:
                            tool_method = getattr(self, method_name)
                            await _log("autopilot_step_start", step=step_num, tool=tool_name, args=args)
                            self._update_activity()
                            outcome = await tool_method(**args)
                            self._update_activity()
                            step_log["success"] = outcome.get("success", False)
                            step_log["result"] = outcome # Store full result

                            if step_log["success"] or not outcome.get("success"): # Pop if success OR failure (replanning handles next step)
                                current_plan.pop(0)

                            if not step_log["success"]:
                                step_log["error"] = outcome.get("error", "Tool failed")
                                # ... (keep replanning logic) ...
                                await _log("autopilot_step_fail", step=step_num, tool=tool_name, error=step_log["error"])
                                logger.warning(f"Autopilot Step {step_num} ({tool_name}) failed: {step_log['error']}")
                                logger.info(f"Attempting replan after failed step {step_num}...")
                                try:
                                    new_plan_tail = await _plan_autopilot(current_task, all_results + [step_log])
                                    current_plan = new_plan_tail # Replace remaining plan
                                    logger.info(f"Replanning successful. New plan: {len(current_plan)} steps.")
                                    await _log("autopilot_replan_success", new_steps=len(current_plan))
                                except Exception as replan_err:
                                    logger.error(f"Replanning failed: {replan_err}")
                                    await _log("autopilot_replan_fail", error=str(replan_err))
                                    current_plan = [] # Stop execution
                            else:
                                await _log("autopilot_step_success", step=step_num, tool=tool_name, result_summary=str(outcome)[:200])

                        except (ToolInputError, ToolError, ValueError, TypeError, AssertionError) as e:
                            step_log["error"] = f"{type(e).__name__} executing '{tool_name}': {e}"
                            logger.error(f"Autopilot Step {step_num} failed: {step_log['error']}", exc_info=True)
                            current_plan = [] # Stop
                        except Exception as e:
                            step_log["error"] = f"Unexpected error executing '{tool_name}': {e}"
                            logger.critical(f"Autopilot Step {step_num} failed unexpectedly: {step_log['error']}", exc_info=True)
                            current_plan = [] # Stop
                    # --- End Tool Execution Logic ---

                    all_results.append(step_log)
                    # --- Append to Log File (Keep direct aiofiles for append) ---
                    if log_path: # Only write if path was determined
                        try:
                            async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                                # Ensure result is serializable
                                log_entry_str = json.dumps(step_log, default=str) + "\n"
                                await log_f.write(log_entry_str)
                        except IOError as log_e:
                            logger.error(f"Failed to write autopilot log entry to {log_path}: {log_e}")
                    # --- End Log Append ---

                # --- Final logging after loop ---
                if step_num >= max_steps:
                    logger.warning(f"Autopilot max steps ({max_steps}) reached.")
                    await _log("autopilot_max_steps", task=task, steps=step_num)
                elif not current_plan and step_num > 0:
                    logger.info(f"Autopilot plan complete after {step_num} steps.")
                    await _log("autopilot_plan_end", task=task, steps=step_num)
                elif step_num == 0:
                    logger.warning("Autopilot did not execute any steps (empty initial plan?).")
                    await _log("autopilot_plan_end", task=task, steps=0)

                overall_success = bool(all_results) and all_results[-1].get("success", False)
                return {
                    "success": overall_success,
                    "steps_executed": step_num,
                    "run_log": str(log_path) if log_path else None, # Return string path
                    "final_results": all_results[-3:], # Return summary
                }

            except Exception as autopilot_err:
                logger.critical(f"Autopilot run failed critically: {autopilot_err}", exc_info=True)
                await _log("autopilot_critical_error", task=task, error=str(autopilot_err))
                error_entry = {"step": 0, "success": False, "error": f"Autopilot critical failure: {autopilot_err}"}
                if log_path: # Try logging error to file
                    try:
                        async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                            await log_f.write(json.dumps(error_entry, default=str) + "\n")
                    except Exception as final_log_e:
                        logger.error(f"Failed to write final error to autopilot log {log_path}: {final_log_e}")
                raise ToolError(f"Autopilot failed: {autopilot_err}") from autopilot_err

        # --- Timeout Wrapper ---
        try:
            return await asyncio.wait_for(autopilot_inner(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            error_msg = f"Autopilot execution timed out after {timeout_seconds}s"
            logger.error(error_msg)
            await _log("autopilot_timeout", task=task, timeout=timeout_seconds)
            # Try logging timeout error to file
            if log_path:
                try:
                    timeout_entry = {"step": -1, "success": False, "error": error_msg} # Use step -1 for timeout
                    async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                        await log_f.write(json.dumps(timeout_entry, default=str) + "\n")
                except Exception as timeout_log_e:
                    logger.error(f"Failed to write timeout error to autopilot log {log_path}: {timeout_log_e}")
            return {
                "success": False, "error": error_msg, "steps_executed": 0, # Or track steps executed before timeout
                "run_log": str(log_path) if log_path else None, "final_results": []
            }


    @tool(name="smart_browser.parallel")
    @with_tool_metrics
    @with_error_handling
    async def parallel_process(
        self, urls: List[str], action: str = "get_state", max_tabs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Processes multiple URLs in parallel using the tab pool (currently only 'get_state')."""
        await self._ensure_initialized()
        self._update_activity()
        if not urls or not isinstance(urls, list):
            raise ToolInputError("Requires a list of URLs.")
        if action != "get_state":
            raise ToolInputError("Only 'get_state' action supported currently.")
        pool = TabPool(max_tabs=max_tabs) if max_tabs is not None else self.tab_pool

        async def process_url(page: Page, *, url: str) -> Dict[str, Any]:
            try:
                full_url = url if url.startswith(("http://", "https://")) else f"https://{url}"
                await _log("parallel_navigate", url=full_url)
                await page.goto(full_url, wait_until="networkidle", timeout=45000)
                state = await get_page_state(page)
                return {"url": full_url, "success": True, "page_state": state}
            except PlaywrightException as e:
                await _log("parallel_url_error", url=full_url, error=str(e))
                return {"url": url, "success": False, "error": f"Playwright error: {e}"}
            except Exception as e:
                await _log("parallel_url_error", url=full_url, error=str(e))
                return {"url": url, "success": False, "error": f"Unexpected error: {e}"}

        tasks = [functools.partial(process_url, url=u) for u in urls]
        results = await pool.map(tasks)
        successful_count = sum(1 for r in results if r.get("success"))
        await _log(
            "parallel_process_complete", total=len(urls), successful=successful_count, action=action
        )
        return {
            "success": True,
            "results": results,
            "processed_count": len(results),
            "successful_count": successful_count,
        }

    # --- Lifecycle Methods ---
    async def async_setup(self):
        """Called by MCP server during startup."""
        logger.info("SmartBrowserTool async_setup.")
        await self._ensure_initialized()

    async def async_teardown(self):
        """Called by MCP server during shutdown."""
        logger.info("SmartBrowserTool async_teardown.")
        # Cancel inactivity monitor first
        if self._inactivity_monitor_task and not self._inactivity_monitor_task.done():
            self._inactivity_monitor_task.cancel()
            try:
                await asyncio.wait_for(self._inactivity_monitor_task, timeout=2.0)
            except asyncio.CancelledError:
                logger.info("Inactivity monitor task cancelled as expected during teardown.")
            except Exception as e:
                # Log other potential errors during wait_for
                logger.warning(f"Error waiting for inactivity monitor task cancellation: {e}")
        # Use the safe shutdown initiator
        await _initiate_shutdown()
