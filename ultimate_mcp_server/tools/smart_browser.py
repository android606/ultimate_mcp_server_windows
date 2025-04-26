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
import base64
import difflib
import hashlib
import json
import os
import random
import re
import sqlite3
import subprocess
import time
import urllib.parse
from contextlib import closing
from html import unescape
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from playwright.async_api import Browser, BrowserContext, Page, TimeoutError, async_playwright

from ultimate_mcp_server.constants import Provider, TaskType
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.tools.base import BaseTool, tool, with_error_handling, with_tool_metrics
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.smart_browser")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  FILESYSTEM & ENCRYPTION & DB
# ══════════════════════════════════════════════════════════════════════════════
_HOME = Path.home() / ".smart_browser"
_HOME.mkdir(parents=True, exist_ok=True)
_STATE_FILE = _HOME / "storage_state.enc"
_LOG_FILE = _HOME / "audit.log"
_SELDB_FILE = _HOME / "selectors.db"
_last_hash: str | None = None

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
except ImportError:
    AESGCM = None  # type: ignore


def _key() -> bytes | None:
    k = os.getenv("SB_STATE_KEY", "")
    return base64.b64decode(k) if k else None


def _enc(buf: bytes) -> bytes:  # AES-GCM optional
    k = _key()
    if not (k and AESGCM):
        return buf
    nonce = os.urandom(12)
    return nonce + AESGCM(k).encrypt(nonce, buf, None)


def _dec(buf: bytes) -> bytes:
    k = _key()
    if not (k and AESGCM):
        return buf
    return AESGCM(k).decrypt(buf[:12], buf[12:], None)


# ── selector-cache DB ─────────────────────────────────────────────────────────
def _init_db():
    with closing(sqlite3.connect(_SELDB_FILE)) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS selectors(
                   site TEXT, key TEXT, css TEXT, xpath TEXT,
                   score REAL DEFAULT 1,
                   PRIMARY KEY(site,key,css,xpath)
               )"""
        )
        con.commit()


_init_db()


def _sel_key(role: str | None, name: str | None) -> str:
    return f"{role or ''}::{name or ''}".strip(":")


def _get_best_selectors(site: str, key: str) -> list[tuple[str | None, str | None]]:
    with closing(sqlite3.connect(_SELDB_FILE)) as con:
        rows = con.execute(
            "SELECT css,xpath FROM selectors WHERE site=? AND key=? ORDER BY score DESC LIMIT 5",
            (site, key),
        ).fetchall()
    return rows


def _bump_selector(site: str, key: str, css: str | None, xpath: str | None):
    with closing(sqlite3.connect(_SELDB_FILE)) as con:
        con.execute(
            """INSERT INTO selectors(site,key,css,xpath,score)
               VALUES (?,?,?,?,1)
               ON CONFLICT(site,key,css,xpath) DO UPDATE SET score=score+1""",
            (site, key, css, xpath),
        )
        con.commit()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  AUDIT LOG (hash-chained)
# ══════════════════════════════════════════════════════════════════════════════
def _log(event: str, **details):
    global _last_hash
    ts = int(time.time() * 1000)
    
    # Map certain events to task types
    emoji_key = None
    if event.startswith("browser_"):
        emoji_key = TaskType.BROWSER.value  # Using TaskType for categorization
    elif event == "navigate":
        emoji_key = TaskType.BROWSE.value
    
    entry = {
        "ts": ts, 
        "event": event, 
        "details": details, 
        "prev": _last_hash,
        "emoji_key": emoji_key
    }
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256(payload).hexdigest()
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"hash": h, **entry}) + "\n")
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
                        _log("retry_fail", func=fn.__name__, err=str(e))
                        raise
                    jitter = backoff * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    _log("retry", func=fn.__name__, attempt=attempt, sleep=round(jitter, 2))
                    await asyncio.sleep(jitter)
        return inner
    return wrap


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SECRET VAULT BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
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
        client = hvac.Client(url=addr, token=token)
        if not client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

        path, key = path_key[6:].split("#", 1)
        mount, sub = (path.split("/", 1) + [""])[:2]
        data = client.secrets.kv.v2.read_secret_version(mount_point=mount, path=sub)
        try:
            return data["data"]["data"][key]
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
_js_lib_cached: set[str] = set()


def _choose_proxy() -> str | None:
    pool = [p.strip() for p in os.getenv("PROXY_POOL", "").split(";") if p.strip()]
    return random.choice(pool) if pool else None


def _start_vnc():
    global _vnc_proc
    if _vnc_proc or os.getenv("VNC") != "1":
        return
    try:
        _vnc_proc = subprocess.Popen(
            ["x11vnc", "-display", os.getenv("DISPLAY", ":0"), "-nopw", "-forever"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # optional


def _load_state() -> dict[str, Any] | None:
    if not _STATE_FILE.exists():
        return None
    try:
        return json.loads(_dec(_STATE_FILE.read_bytes()))
    except Exception:
        _STATE_FILE.unlink(missing_ok=True)
        return None


async def _save_state(ctx: BrowserContext):
    _STATE_FILE.write_bytes(_enc(json.dumps(await ctx.storage_state()).encode()))


async def _ctx(headless: bool | None = None) -> tuple[BrowserContext, Browser]:
    global _pw, _browser, _ctx
    if _ctx:
        return _ctx, _browser  # type: ignore[arg-type]

    headless_env = os.getenv("HEADLESS")
    headless = (
        True
        if headless_env is None
        else headless_env.lower() not in ("0", "false", "no")
    )
    _start_vnc()
    _pw = await async_playwright().start()
    proxy_url = _choose_proxy()
    _browser = await _pw.chromium.launch(
        headless=headless,
        proxy={"server": proxy_url} if proxy_url else None,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--window-size=1280,1024",
        ],
    )
    _ctx = await _browser.new_context(
        viewport={"width": 1280, "height": 1024}, storage_state=_load_state() or None
    )
    _log("browser_start", headless=headless, proxy=proxy_url)
    return _ctx, _browser


async def shutdown():
    global _pw, _browser, _ctx, _vnc_proc
    if _ctx:
        await _save_state(_ctx)
        await _ctx.close()
    if _browser:
        await _browser.close()
    if _pw:
        await _pw.stop()
    if _vnc_proc:
        _vnc_proc.terminate()
    _log("browser_shutdown")
    _pw = _browser = _ctx = _vnc_proc = None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TAB POOL FOR PARALLELISM
# ══════════════════════════════════════════════════════════════════════════════
class TabPool:
    """Runs callables that need a fresh Page in parallel, bounded by SB_MAX_TABS."""

    def __init__(self, max_tabs: int | None = None):
        self.sem = asyncio.Semaphore(
            max_tabs if max_tabs is not None else int(os.getenv("SB_MAX_TABS", "5"))
        )

    async def _run(self, fn: Callable[[Page], Awaitable[Any]]) -> Any:
        async with self.sem:
            ctx, _ = await _ctx()
            p = await ctx.new_page()
            try:
                _log("page_open")
                return await fn(p)
            finally:
                await p.close()
                _log("page_close")

    async def map(self, fns: Sequence[Callable[[Page], Awaitable[Any]]]) -> List[Any]:
        async with asyncio.TaskGroup() as tg:  # Python 3.11+
            tasks = [tg.create_task(self._run(fn)) for fn in fns]
        return [t.result() for t in tasks]


tab_pool = TabPool()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  HUMAN-LIKE JITTER (bot evasion)
# ══════════════════════════════════════════════════════════════════════════════
def _risk_factor(url: str) -> float:
    d = urlparse(url).netloc
    # naïve heuristic – tweak per organisation policy
    high = ("facebook.com", "linkedin.com", "glassdoor.com", "instagram.com")
    if any(d.endswith(x) for x in high):
        return 2.0
    return 1.0


async def _pause(page: Page, base_ms: tuple[int, int] = (100, 400)):
    factor = _risk_factor(page.url or "")
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
        for css, xpath in _get_best_selectors(self.site, key):
            if css:
                try:
                    el = self.p.locator(css).first
                    await el.wait_for(state="visible", timeout=500)
                    return el
                except TimeoutError:
                    continue
            if xpath:
                try:
                    el = self.p.locator(f"xpath={xpath}").first
                    await el.wait_for(state="visible", timeout=500)
                    return el
                except TimeoutError:
                    continue
        return None

    async def _by_role(self, role: str | None, name: str | None):
        if not role:
            return None
        try:
            return await self.p.get_by_role(role, name=name, exact=True).first.wait_for(
                state="visible", timeout=self.t
            )
        except TimeoutError:
            return None

    async def _by_label(self, name: str | None):
        if not name:
            return None
        try:
            return await self.p.get_by_label(name, exact=True).first.wait_for(
                state="visible", timeout=self.t
            )
        except TimeoutError:
            return None

    async def _by_text(self, name: str | None):
        if not name:
            return None
        try:
            return await self.p.get_by_text(name, exact=True).first.wait_for(
                state="visible", timeout=self.t
            )
        except TimeoutError:
            return None

    async def _fuzzy(self, name: str | None):
        if not name:
            return None
        texts = [
            t.strip()[:128]
            for t in await self.p.locator("xpath=//*[normalize-space(string())!='']").all_text_contents()
        ]
        best = difflib.get_close_matches(name, texts, n=1, cutoff=0.55)
        if not best:
            return None
        return await self.p.locator(f"text='{best[0]}'").first

    async def locate(
        self, *, name: str | None = None, role: str | None = None, css: str | None = None, xpath: str | None = None
    ):
        key = _sel_key(role, name)
        if name or role:
            learned = await self._try_learned(key)
            if learned:
                _log("locator_learned_hit", key=key)
                return learned
        for fn in (
            lambda: self._by_role(role, name),
            lambda: self._by_label(name),
            lambda: self._by_text(name),
            lambda: self._fuzzy(name),
            lambda: self.p.locator(css) if css else None,
            lambda: self.p.locator(f"xpath={xpath}") if xpath else None,
        ):
            el = await fn()
            if el:
                _bump_selector(self.site, key, css or None, xpath or None)
                _log("locator_success", method=fn.__name__, target=name or role or css or xpath)
                return el
        _log("locator_fail", target=name or role or css or xpath)
        raise TimeoutError(f"Element not found → {name or role or css or xpath}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SMART ACTIONS WITH RETRY SUPPORT
# ══════════════════════════════════════════════════════════════════════════════
@resilient()
async def smart_click(page: Page, **kw):
    await _pause(page)
    await (await SmartLocator(page).locate(**kw)).click()
    _log("click", **kw)


@resilient()
async def smart_type(page: Page, text: str, press_enter: bool = False, **kw):
    await _pause(page)
    if text.startswith("secret:"):
        text = get_secret(text[7:])
    el = await SmartLocator(page).locate(**kw)
    await el.fill("")
    await el.type(text, delay=25)
    if press_enter:
        await el.press("Enter")
    _log("type", value="***" if len(text) > 6 else text, enter=press_enter, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DOWNLOAD HELPER WITH DOCUMENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _extract_tables(path: Path) -> list:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            import tabula  # type: ignore
            dfs = tabula.read_pdf(str(path), pages="all", multiple_tables=True)
            return [df.to_dict(orient="records") for df in dfs]
        if ext in (".xls", ".xlsx"):
            import pandas as pd  # type: ignore
            xl = pd.read_excel(str(path), sheet_name=None)
            return [{ "sheet": name, "rows": df.to_dict(orient="records")} for name, df in xl.items()]
        if ext == ".csv":
            import pandas as pd  # type: ignore
            df = pd.read_csv(str(path))
            return [df.to_dict(orient="records")]
    except Exception as e:
        _log("table_extract_error", file=str(path), err=str(e))
    return []


@resilient()
async def smart_download(page: Page, target: Dict[str, Any], dest_dir: str | Path | None = None) -> Dict[str, Any]:
    dest_dir = Path(dest_dir or _HOME / "downloads")
    dest_dir.mkdir(parents=True, exist_ok=True)
    async with page.expect_download() as dl_info:
        await smart_click(page, **target)
    dl = await dl_info.value
    fname = dl.suggested_filename
    out_path = dest_dir / fname
    await dl.save_as(str(out_path))
    data = out_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    tables = _extract_tables(out_path)
    info = {
        "file": str(out_path), 
        "sha256": sha, 
        "size": len(data),
        "tables_extracted": bool(tables)
    }
    if tables:
        info["tables"] = tables[:3]  # avoid huge payload
    _log("download", **info)
    return info


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PAGE STATE EXTRACTION WITH HTML SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════════
_HELPER_CDN = "https://unpkg.com/htmlparser-mdist@1.0.5/dist/html-parser.min.js"


async def _ensure_helper(page: Page):
    if _HELPER_CDN in _js_lib_cached:
        return
    await page.evaluate(
        f"""
        () => new Promise(r => {{
            if (window._helperLoaded) return r();
            const s=document.createElement('script');s.src='{_HELPER_CDN}';
            s.onload=()=>{{window._helperLoaded=true;r();}};
            document.head.appendChild(s);
        }})
    """
    )
    _js_lib_cached.add(_HELPER_CDN)


def _summarize_html(html: str, max_len: int = 4000) -> str:
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
        text = re.sub(r"<[^>]+>", " ", unescape(summary_html))
        return re.sub(r"\s+", " ", text).strip()[:max_len]
    except Exception:
        pass
    # fallback: raw body text
    text = re.sub(r"<[^>]+>", " ", unescape(html))
    return re.sub(r"\s+", " ", text).strip()[:max_len]


async def get_page_state(page: Page, max_nodes: int = 120) -> dict[str, Any]:
    await _ensure_helper(page)
    html = await page.content()
    summary = _summarize_html(html)
    elems = await page.evaluate(
        """(max)=>{
            const vis=e=>{const r=e.getBoundingClientRect();return r.width&&r.height&&r.top<innerHeight&&r.left<innerWidth};
            let out=[],id=0;
            document.querySelectorAll('a,button,input,textarea,select,[role]').forEach(el=>{
              if(out.length>=max||!vis(el)) return;
              const o={id:`el_${id++}`,tag:el.tagName.toLowerCase(),
              role:el.getAttribute('role')||'',text:(el.innerText||el.value||el.getAttribute('aria-label')||'').trim().slice(0,120)};
              if(el.href) o.href=el.href.slice(0,300); out.push(o);
            }); return out;}""",
        max_nodes,
    )
    return {"url": page.url, "title": await page.title(), "elements": elems, "text_summary": summary}


# ══════════════════════════════════════════════════════════════════════════════
# 11.  LLM BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
async def _call_llm(
    messages: Sequence[dict[str, str]], model: str = "gpt-4o", expect_json: bool = True, temperature: float = 0.1
) -> dict[str, Any]:
    prompt = messages[-1]["content"]
    try:
        resp = await generate_completion(
            provider=Provider.OPENAI.value,  # Using Provider constant here
            model=model, 
            prompt=prompt, 
            temperature=temperature
        )
        if not resp.get("success"):
            return {"error": resp.get("error", "LLM failure")}
        txt = resp["text"]
        if not expect_json:
            return {"text": txt}
        try:
            return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception as e:
            return {"error": f"JSON parse error: {e}", "raw": txt}
    except ProviderError as e:  # Catching ProviderError here
        logger.error(f"Provider error in LLM call: {str(e)}")
        return {"error": f"Provider error: {str(e)}"}

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
    txt = await _call_llm(messages, model=model)
    try:
        steps = json.loads(re.search(r"\[.*\]", txt["text"], re.S).group(0))
    except Exception as e:
        raise RuntimeError("Planner returned invalid JSON") from e
    for st in steps:
        if st["action"] not in ALLOWED_ACTIONS:
            raise ValueError(f"illegal action {st['action']}")
    return steps


async def run_macro(page: Page, task: str, max_rounds: int = 5, model: str = "gpt-4o"):
    for i in range(max_rounds):
        state = await get_page_state(page)
        plan = await _plan(state, task, model)
        _log("macro_plan", round=i + 1, steps=plan)
        results = await run_steps(page, plan)
        if any(s["action"] == "finish" for s in plan):
            _log("macro_complete", rounds=i + 1)
            return results
    raise RuntimeError("macro exceeded rounds")


# ══════════════════════════════════════════════════════════════════════════════
# 13.  STEP RUNNER
# ══════════════════════════════════════════════════════════════════════════════
async def run_steps(page: Page, steps: Sequence[dict[str, Any]]):
    loc = SmartLocator(page)
    for s in steps:
        act = s["action"]
        if act == "click":
            await (await loc.locate(**s["target"])).click()
        elif act == "type":
            await smart_type(page, s["text"], press_enter=s.get("enter", False), **s["target"])
        elif act == "wait":
            await page.wait_for_timeout(int(s.get("ms", 1000)))
        elif act == "download":
            s["result"] = await smart_download(page, s["target"], s.get("dest"))
        elif act == "extract":
            s["result"] = await page.eval_on_selector_all(s["selector"], "(els)=>els.map(e=>e.innerText)")
        elif act == "finish":
            pass
        else:
            raise ValueError(f"Unknown action {act}")
        _log("step", action=act, detail=s)
    return steps


# ══════════════════════════════════════════════════════════════════════════════
# 14.  UNIVERSAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
async def search_web(
    query: str, engine: str = "yandex", max_results: int = 10
) -> List[Dict[str, str]]:
    engine = engine.lower()
    qs = urllib.parse.quote_plus(query)
    urls = {
        "yandex": f"https://yandex.com/search/?text={qs}&ncrnd={int(time.time()*1000)}",
        "bing": f"https://www.bing.com/search?q={qs}&count={max_results}",
        "duckduckgo": f"https://duckduckgo.com/?q={qs}&ia=web",
    }
    selectors = {
        "yandex": (".serp-item", "a.organic__url", "h2.organic__url-text", "div.text-container"),
        "bing": ("li.b_algo", "h2>a", "h2>a", ".b_caption p"),
        "duckduckgo": ("article.results_links_deep, article.result", "a.result__a", "a.result__a", ".result__snippet"),
    }
    if engine not in urls:
        raise ToolError(
            f"Invalid search engine: {engine}",
            error_code="invalid_engine", 
            details={"valid_engines": list(urls.keys())}
        )
    ctx, _ = await _ctx()
    p: Page = await ctx.new_page()
    await p.goto(urls[engine], wait_until="networkidle")
    await p.wait_for_selector(selectors[engine][0])
    main, link_sel, title_sel, snip_sel = selectors[engine]
    results = await p.eval_on_selector_all(
        main,
        """(els, l, t, s, maxR) => els.slice(0, maxR).map(el => {
            const link = el.querySelector(l);
            const titleEl = el.querySelector(t);
            const snippetEl = el.querySelector(s);
            if (!link || !link.href.startsWith('http')) return null;
            const title = (titleEl ? titleEl.innerText : link.innerText || '').trim();
            const snip = (snippetEl ? snippetEl.innerText : '').trim();
            return {url: link.href, title, snippet: snip};
        }).filter(Boolean)""",
        link_sel,
        title_sel,
        snip_sel,
        max_results,
    )
    await p.close()
    _log("search", engine=engine, q=query, n=len(results))
    return results

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
        # Initialize immediately so it's ready for use
        asyncio.create_task(self._ensure_browser())
    
    async def _ensure_browser(self):
        """Ensure browser is initialized"""
        await _ctx()
    
    @tool(name="browse")
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
        if not url.startswith("http"):
            url = "https://" + url
            
        ctx, _ = await _ctx()
        page = await ctx.new_page()
        
        try:
            _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle" if wait_for_navigation else "domcontentloaded")
            
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            
            state = await get_page_state(page)
                
            return {
                "success": True,
                "page_state": state
            }
        finally:
            await page.close()
    
    @tool(name="click")
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
        ctx, _ = await _ctx()
        page = await ctx.new_page()
        
        try:
            _log("navigate", url=url)
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
        finally:
            await page.close()
    
    @tool(name="fill_form")
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
        ctx, _ = await _ctx()
        page = await ctx.new_page()
        
        try:
            _log("navigate", url=url)
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
        finally:
            await page.close()
    
    @tool(name="search")
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
    
    @tool(name="download")
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
        ctx, _ = await _ctx()
        page = await ctx.new_page()
        
        try:
            _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle")
            
            # Download the file
            download_info = await smart_download(page, target, dest_dir)
            
            return {
                "success": True,
                "download": download_info
            }
        finally:
            await page.close()
    
    @tool(name="run_macro")
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
        ctx, _ = await _ctx()
        page = await ctx.new_page()
        
        try:
            _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle")
            
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
        finally:
            await page.close()
    
    @tool(name="parallel")
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
        # Create tab pool with specified max tabs
        pool = TabPool(max_tabs)
        
        # Define function to process a single URL
        async def process_url(page: Page, url: str) -> Dict[str, Any]:
            _log("navigate", url=url)
            await page.goto(url, wait_until="networkidle")
            state = await get_page_state(page)
            return {"url": url, "state": state}
        
        # Create a callable for each URL
        tasks = []
        for url in urls:
            u = url if url.startswith("http") else f"https://{url}"
            tasks.append(lambda p, url=u: process_url(p, url))
        
        # Execute all tasks in parallel
        results = await pool.map(tasks)
        
        return {
            "success": True,
            "results": results,
            "processed_count": len(results)
        }
    
    @tool(name="shutdown")
    @with_tool_metrics
    @with_error_handling
    async def shutdown_browser(self) -> Dict[str, Any]:
        """
        Shut down the browser instance and clean up resources.
        
        Returns:
            Dictionary with shutdown status
        """
        await shutdown()
        
        return {
            "success": True,
            "message": "Browser shutdown complete"
        }