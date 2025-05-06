#!/usr/bin/env python
"""Comprehensive HTML‑redline demo that exercises **every** change type.

Run this file after you have installed/linked the Ultimate‑MCP‑Server package
in editable mode (``pip install -e .``) or added the repo root to ``PYTHONPATH``.
It generates a single HTML file (``./redline_outputs/comprehensive_redline_demo.html``)
that you can open in any browser to see insertions (blue), deletions (red),
move‑targets/sources (green), attribute changes (orange) and inline word‑level
diffs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

# ---------------------------------------------------------------------------
# 1.  Make sure we can import the Ultimate‑MCP‑Server package from source.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root (../..)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 2.  Project‑level imports (raise immediately if the dev env is broken)
# ---------------------------------------------------------------------------
from ultimate_mcp_server.exceptions import ToolError  # noqa: E402
from ultimate_mcp_server.tools.filesystem import write_file  # noqa: E402
from ultimate_mcp_server.tools.text_redline_tools import (  # noqa: E402
    create_html_redline,  # noqa: E402
)
from ultimate_mcp_server.utils import get_logger  # noqa: E402

# ---------------------------------------------------------------------------
# 3.  Logger / console helpers
# ---------------------------------------------------------------------------
LOGGER = get_logger("demo.comprehensive_redline")
CONSOLE = Console()

# ---------------------------------------------------------------------------
# 4.  Demo input documents (original vs. modified)
# ---------------------------------------------------------------------------
ORIGINAL_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Demo Document</title>
    <meta name="description" content="A document to demonstrate redlining features">
</head>
<body>
    <h1>Project Documentation</h1>
    
    <div class="intro">
        <p>This project documentation covers all aspects of the Alpha system implementation.</p>
        <p>Last updated on January 15, 2025</p>
    </div>

    <h2>Executive Summary</h2>
    <p>The Alpha system provides robust data processing capabilities for enterprise applications.</p>
    <p>This documentation serves as the primary reference for developers and system architects.</p>

    <h2>Architecture Overview</h2>
    <p>The system follows a microservices architecture with the following components:</p>
    <ul>
        <li>Data ingestion layer</li>
        <li>Processing engine</li>
        <li>Storage layer</li>
        <li>API gateway</li>
    </ul>

    <h2>Implementation Details</h2>
    <p>Implementation follows the standard protocol described in section 5.2 of the technical specifications.</p>
    <p>All components must pass integration tests before deployment.</p>
    
    <h2>Deployment Process</h2>
    <p>Deployment occurs in three phases:</p>
    <ol>
        <li>Development environment validation</li>
        <li>Staging environment testing</li>
        <li>Production rollout</li>
    </ol>
    <p>Each phase requires approval from the technical lead.</p>

    <h2>Security Considerations</h2>
    <p>All data must be encrypted during transfer and at rest.</p>
    <p>Authentication uses OAuth 2.0 with JWT tokens.</p>
    <p>Regular security audits are conducted quarterly.</p>

    <table border="1">
        <tr>
            <th>Component</th>
            <th>Responsible Team</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>Data ingestion</td>
            <td>Data Engineering</td>
            <td>Complete</td>
        </tr>
        <tr>
            <td>Processing engine</td>
            <td>Core Systems</td>
            <td>In progress</td>
        </tr>
        <tr>
            <td>Storage layer</td>
            <td>Infrastructure</td>
            <td>Complete</td>
        </tr>
        <tr>
            <td>API gateway</td>
            <td>API Team</td>
            <td>Planning</td>
        </tr>
    </table>

    <h2>Appendix</h2>
    <p>For additional information, refer to the technical specifications document.</p>
    <p>Contact <a href="mailto:support@example.com">support@example.com</a> with any questions.</p>
</body>
</html>"""

MODIFIED_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Demo Document - 2025 Update</title>
    <meta name="description" content="A document to demonstrate all redlining features">
    <meta name="author" content="Documentation Team">
</head>
<body>
    <h1>Project Documentation</h1>
    
    <div class="intro">
        <p>This project documentation covers all aspects of the Alpha system implementation and integration.</p>
        <p>Last updated on May 5, 2025</p>
    </div>

    <h2>Appendix</h2>
    <p>For additional information, refer to the technical specifications document and API references.</p>
    <p>Contact <a href="mailto:technical-support@example.com">technical-support@example.com</a> with any questions.</p>

    <h2>Security Considerations</h2>
    <p>All data must be encrypted during transfer and at rest using AES-256 encryption.</p>
    <p>Authentication uses OAuth 2.0 with JWT tokens and optional two-factor authentication.</p>
    <p>Regular security audits are conducted quarterly by an independent security firm.</p>
    <p>Penetration testing is performed bi-annually.</p>

    <h2>Executive Summary</h2>
    <p>The Alpha system provides robust data processing capabilities for enterprise applications with enhanced performance.</p>
    <p>This documentation serves as the primary reference for developers, system architects, and operations teams.</p>
    <p>The system has been validated against ISO 27001 standards.</p>

    <h2>Architecture Overview</h2>
    <p>The system implements a cloud-native microservices architecture with the following components:</p>
    <ul>
        <li>Data ingestion layer with real-time processing</li>
        <li>Distributed processing engine</li>
        <li>Multi-region storage layer</li>
        <li>API gateway with rate limiting</li>
        <li>Monitoring and observability platform</li>
        <li>Disaster recovery system</li>
    </ul>

    <h2>Implementation Details</h2>
    <p>Implementation follows the enhanced protocol described in section 6.3 of the technical specifications.</p>
    <p>All components must pass integration and performance tests before deployment.</p>
    
    <table border="1">
        <tr>
            <th>Component</th>
            <th>Responsible Team</th>
            <th>Status</th>
            <th>Performance</th>
        </tr>
        <tr>
            <td>Data ingestion</td>
            <td>Data Engineering</td>
            <td>Complete</td>
            <td>Exceeds SLA</td>
        </tr>
        <tr>
            <td>Processing engine</td>
            <td>Core Systems</td>
            <td>Complete</td>
            <td>Meets SLA</td>
        </tr>
        <tr>
            <td>Storage layer</td>
            <td>Infrastructure</td>
            <td>Complete</td>
            <td>Meets SLA</td>
        </tr>
        <tr>
            <td>API gateway</td>
            <td>API Team</td>
            <td>Complete</td>
            <td>Exceeds SLA</td>
        </tr>
        <tr>
            <td>Monitoring platform</td>
            <td>DevOps</td>
            <td>Complete</td>
            <td>Meets SLA</td>
        </tr>
    </table>

    <h2>Scalability Considerations</h2>
    <p>The system is designed to scale horizontally with increasing load.</p>
    <p>Auto-scaling policies are configured for all compute resources.</p>
    <p>Database sharding is implemented for high-volume tenants.</p>
</body>
</html>"""

# ---------------------------------------------------------------------------
# 5.  Human‑readable change checklist (for demo output only)
# ---------------------------------------------------------------------------
CHANGE_SUMMARY: Dict[str, List[str]] = {
    "insertions": [
        "New <meta author> tag",
        "'and integration' added to intro paragraph",
        "AES‑256 wording added to encryption para",
        "Two‑factor authentication mention added",
        "Independent security firm phrase added",
        "Entire penetration‑testing paragraph added",
        "'with enhanced performance' in exec summary",
        "Audience now includes operations teams",
        "ISO‑27001 paragraph added",
        "'cloud‑native' adjective added",
        "Real‑time processing detail added",
        "'Distributed' processing engine detail",
        "Multi‑region storage detail",
        "Rate‑limiting mention in API gateway",
        "Two new architecture components",
        "Protocol reference bumped to 6.3",
        "Performance tests requirement added",
        "New PERFORMANCE column in table",
        "New Monitoring‑platform row",
        "Whole SCALABILITY section added",
    ],
    "deletions": [
        "API‑gateway status 'Planning' removed",
        "Deployment‑process section removed",
    ],
    "moves": [
        "Appendix moved before Security section",
        "Security section moved before Exec‑Summary",
    ],
    "updates": [
        "<title> suffixed with '2025 Update'",
        "Meta description tweaked",
        "Updated date to 5 May 2025",
        "Support e‑mail address changed",
        "Processing‑engine status updated",
    ],
}

# ---------------------------------------------------------------------------
# 6.  Async helper running the diff + reporting
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).with_suffix("").parent / "redline_outputs"
MARKDOWN_PATH = OUTPUT_DIR / "detected_redline_differences.md"


async def generate_redline() -> None:
    CONSOLE.print("\n[bold blue]Generating HTML redline…[/bold blue]")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = await create_html_redline(
            original_html=ORIGINAL_HTML,
            modified_html=MODIFIED_HTML,
            detect_moves=True,
            include_css=True,
            add_navigation=True,
            output_format="html",
            generate_markdown=True,
            markdown_path=str(MARKDOWN_PATH),
        )
    except Exception as exc:  # demo only
        LOGGER.error("Failed to generate redline", exc_info=True)
        CONSOLE.print(f"[red bold]Error:[/red bold] {escape(str(exc))}")
        return

    # ── Rich stats table ────────────────────────────────────────────────────────────
    stats_tbl = Table(title="Redline statistics", box=box.ROUNDED)
    stats_tbl.add_column("Metric", style="cyan")
    stats_tbl.add_column("Value", style="magenta")
    for k, v in result["stats"].items():
        stats_tbl.add_row(k.replace("_", " ").title(), str(v))
    stats_tbl.add_row("Processing time", f"{result['processing_time']:.3f}s")
    CONSOLE.print(stats_tbl)

    # ── manual checklist ────────────────────────────────────────────────────────────
    CONSOLE.print("\n[bold green]Manual checklist of expected changes[/bold green]")
    for cat, items in CHANGE_SUMMARY.items():
        CONSOLE.print(f"[cyan]{cat.title()}[/cyan] ({len(items)})")
        for idx, txt in enumerate(items, 1):
            CONSOLE.print(f"   {idx:>2}. {txt}")

    # ── write HTML diff ─────────────────────────────────────────────────────────────
    html_path = OUTPUT_DIR / "comprehensive_redline_demo.html"
    try:
        await write_file(path=str(html_path), content=result["redline_html"])
    except (ToolError, Exception) as exc:  # demo only
        LOGGER.warning("Unable to save HTML", exc_info=True)
        CONSOLE.print(f"\n[bold red]Warning:[/bold red] Could not save HTML — {exc}")
    else:
        LOGGER.info("Saved redline to %s", html_path)
        CONSOLE.print(f"\n[green]HTML written to:[/green] {html_path}")

    # ── ensure Markdown file exists (tool usually writes it already) ────────────────
    if not MARKDOWN_PATH.is_file() and "markdown_summary" in result:
        MARKDOWN_PATH.write_text(result["markdown_summary"], encoding="utf-8")
    if MARKDOWN_PATH.is_file():
        CONSOLE.print(f"[green]Markdown summary:[/green] {MARKDOWN_PATH}")


# ───────────────────────────── 7. entrypoint ────────────────────────────────────────
async def _amain() -> int:
    CONSOLE.rule("[white on blue]📝  Comprehensive Text-Redline Demo  📝")
    await generate_redline()
    CONSOLE.rule("Complete", style="green")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
