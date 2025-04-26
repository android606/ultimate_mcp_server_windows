"""Ultimate Document Processing Toolkit for MCP Server integrated as a BaseTool.

A comprehensive, fault-tolerant toolkit for document processing, providing:

1. Document Conversion
   - PDF/Office → Markdown/Text/HTML/JSON/doctags
   - Acceleration options (CPU/CUDA/MPS)
   - Page range and section filtering

2. HTML Processing
   - Fault-tolerant parsing with multiple back-ends
   - Aggressive repair of malformed HTML
   - Multi-stage extraction with automatic fallback
   - Resilient HTML → Markdown conversion

3. Document Chunking
   - Multiple strategies: token, character, section, paragraph
   - Configurable chunk size and overlap

4. Table Extraction & Conversion
   - Extract tables from documents in CSV, JSON, or pandas format
   - Convert HTML tables to markdown tables

5. Document Analysis
   - Section identification
   - Entity extraction
   - Metrics extraction
   - Risk flagging
   - QA Generation
   - Summarization
   - Classification

6. Batch Processing
   - Process multiple documents with multiple operations
   - Configurable concurrency

This unified toolkit combines Document Conversion, Universal Document-Processing,
and HTML-to-Markdown conversion into a single, coherent BaseTool class.
"""
from __future__ import annotations

###############################################################################
# Imports                                                                     #
###############################################################################
# Standard library imports
import asyncio
import csv
import hashlib
import html
import json
import os
import re
import tempfile
import textwrap
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Union,
)

# Third-party imports
import html2text
import readability
import trafilatura
from bs4 import BeautifulSoup, Tag
from rapidfuzz import fuzz

# Type checking imports
if TYPE_CHECKING:
    import pandas as pd
    import tiktoken
    from tiktoken import Encoding

# Try to import optional dependencies
try:
    from markdownify import markdownify as _markdownify_fallback
except ModuleNotFoundError:
    _markdownify_fallback = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None # type: ignore

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None # type: ignore

# Track if docling is available
_DOCLING_AVAILABLE = False

# Try to import basic document libraries for fallback
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

from ultimate_mcp_server.constants import Provider  # noqa: E402
from ultimate_mcp_server.exceptions import ToolError, ToolInputError  # noqa: E402
from ultimate_mcp_server.tools.base import (  # noqa: E402
    BaseTool,
    tool,
    with_error_handling,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.completion import generate_completion  # noqa: E402
from ultimate_mcp_server.utils import get_logger  # noqa: E402

# Setup loggers - BaseTool provides self.logger, but we keep one for module level if needed
_MODULE_LOG = get_logger("ultimate_mcp_server.tools.document_processing_tool")

# Try to import docling
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DoclingDocument, ImageRefMode
    _DOCLING_AVAILABLE = True
except ImportError as e:
    _MODULE_LOG.warning(f"Docling not available, will use basic fallback methods: {e}")


###############################################################################
# DocumentProcessingTool Class                                                #
###############################################################################

class DocumentProcessingTool(BaseTool):
    """
    Comprehensive document processing toolkit for MCP Server.

    Provides tools for document conversion, HTML processing, chunking,
    table extraction, document analysis, and batch processing.
    """

    tool_name = "document_processing"
    description = "A unified toolkit for document conversion, chunking, analysis, and HTML processing."

    def __init__(self, mcp_server):
        """Initialize the document processing tool."""
        super().__init__(mcp_server)
        
        # Track docling availability
        self._docling_available = _DOCLING_AVAILABLE
        if not self._docling_available:
            self.logger.warning("Docling library not available. Will use basic fallback methods for document conversion.")

        # ───────────────────── Acceleration Device Mapping ─────────────────────
        if _DOCLING_AVAILABLE:
            self._ACCEL_MAP = {
                "auto": AcceleratorDevice.AUTO,
                "cpu": AcceleratorDevice.CPU,
                "cuda": AcceleratorDevice.CUDA,
                "mps": AcceleratorDevice.MPS,
            }
        else:
            # Simple strings when docling isn't available
            self._ACCEL_MAP = {
                "auto": "auto",
                "cpu": "cpu",
                "cuda": "cuda",
                "mps": "mps",
            }

        # ───────────────────── Output Format Mapping ─────────────────────────
        # These will be used differently depending on whether docling is available
        self._VALID_FORMATS = {"markdown", "text", "html", "json", "doctags"}

        # ───────────────────── HTML Detection Patterns ─────────────────────────
        _RE_FLAGS = re.MULTILINE | re.IGNORECASE
        self.HTML_PATTERNS: Sequence[Pattern] = [
            re.compile(p, _RE_FLAGS)
            for p in (
                r"<\s*[a-zA-Z]+[^>]*>", # Opening tag
                r"<\s*/\s*[a-zA-Z]+\s*>", # Closing tag
                r"&[a-zA-Z]+;",         # HTML entity
                r"&#[0-9]+;",            # Numeric entity
                r"<!\s*DOCTYPE",         # Doctype declaration
                r"<!\s*--",              # HTML comment start
            )
        ]

        # ───────────────────── Domain Rules ─────────────────────────────────
        self.DOMAIN_RULES: Dict[str, Dict[str, Any]] = {
            # (Domain rules copied from original script - assuming these are correct)
            "generic": {
                "classification": {
                    "labels": ["Report", "Contract", "Presentation", "Memo", "Email", "Manual"],
                    "prompt_prefix": "Classify the document into exactly one of: "
                },
                "sections": {
                    "boundary_regex": r"^\s*(chapter\s+\d+|section\s+\d+|[A-Z][A-Za-z\s]{3,80})$",
                    "custom": []
                },
                "metrics": {
                    "metric_1": {"aliases": ["metric one", "m1"]},
                    "metric_2": {"aliases": ["metric two", "m2"]},
                },
                "risks": {
                    "Risk_A": r"risk a",
                    "Risk_B": r"risk b",
                },
            },
            "finance": {
                "classification": {
                    "labels": ["10-K", "Credit Agreement", "Investor Deck", "Press Release", "Board Minutes", "NDA", "LPA", "CIM"],
                    "prompt_prefix": "Identify the document type (finance domain): "
                },
                "sections": {
                    "boundary_regex": r"^\s*(item\s+\d+[a-z]?\.|[A-Z][A-Za-z\s]{3,80})$",
                    "custom": [
                        {"regex": r"item\s+1a?\.? .*business", "label": "Business"},
                        {"regex": r"item\s+1a\.? .*risk factors", "label": "Risk Factors"},
                        {"regex": r"item\s+7\.? .*management'?s discussion", "label": "MD&A"},
                        {"regex": r"covena[nv]ts", "label": "Covenants"},
                    ],
                },
                "metrics": {
                    "revenue": {"aliases": ["revenue", "net sales", "total sales", "sales revenue", "turnover"]},
                    "ebitda": {"aliases": ["ebitda", "adj. ebitda", "operating profit", "operating income"]},
                    "gross_profit": {"aliases": ["gross profit"]},
                    "net_income": {"aliases": ["net income", "net profit", "earnings"]},
                    "capex": {"aliases": ["capital expenditures", "capex"]},
                    "debt": {"aliases": ["total debt", "net debt", "long-term debt"]},
                },
                "risks": {
                    "Change_of_Control": r"change\s+of\s+control",
                    "ESG_Risk": r"(child\s+labor|environmental\s+violation|scope\s+3)",
                    "PII": r"(\bSSN\b|social security number|passport no)",
                },
            },
            "legal": {
                "classification": {
                    "labels": ["Contract", "NDA", "Lease", "Policy", "License", "Settlement"],
                    "prompt_prefix": "Classify the legal document into exactly one of: "
                },
                "sections": {
                    "boundary_regex": r"^\s*(article\s+\d+|section\s+\d+|[A-Z][A-Za-z\s]{3,80})$",
                    "custom": [
                        {"regex": r"definitions", "label": "Definitions"},
                        {"regex": r"termination", "label": "Termination"},
                        {"regex": r"confidentiality", "label": "Confidentiality"},
                    ],
                },
                "metrics": {}, # Legal domain might not have standard numeric metrics
                "risks": {
                    "Indemnity": r"indemnif(y|ication)",
                    "Liquidated_Damages": r"liquidated damages",
                    "Governing_Law_NY": r"governing law.*new york",
                    "Governing_Law_DE": r"governing law.*delaware",
                },
            },
            "medical": {
                "classification": {
                    "labels": ["Clinical Study", "Patient Report", "Lab Results", "Prescription", "Care Plan"],
                    "prompt_prefix": "Classify the medical document: "
                },
                "sections": {
                    "boundary_regex": r"^\s*(section\s+\d+|[A-Z][A-Za-z\s]{3,80})$",
                    "custom": [
                        {"regex": r"diagnosis", "label": "Diagnosis"},
                        {"regex": r"treatment", "label": "Treatment"},
                        {"regex": r"medications", "label": "Medications"},
                        {"regex": r"allergies", "label": "Allergies"}
                    ],
                },
                "metrics": {
                    "blood_pressure": {"aliases": ["blood pressure", "bp"]},
                    "heart_rate": {"aliases": ["heart rate", "hr"]},
                    "temperature": {"aliases": ["temperature", "temp"]},
                    "bmi": {"aliases": ["bmi", "body mass index"]},
                },
                "risks": {
                    "Allergy": r"allergic reaction",
                    "Contraindication": r"contraindicat(ed|ion)",
                    "Adverse_Event": r"adverse event",
                },
            },
        }

        # Initialize domain-specific rules
        self.ACTIVE_DOMAIN = os.getenv("DOC_DOMAIN", "generic")
        if self.ACTIVE_DOMAIN not in self.DOMAIN_RULES:
            self.logger.warning(f"Unknown DOC_DOMAIN '{self.ACTIVE_DOMAIN}', defaulting to 'generic'.")
            self.ACTIVE_DOMAIN = "generic"
        self.instruction_json: Dict[str, Any] = self.DOMAIN_RULES[self.ACTIVE_DOMAIN]

        # Compile regex patterns for sections, metrics, and risks
        try:
            self._BOUND_RX = re.compile(self.instruction_json["sections"].get("boundary_regex", r"$^"), re.M) # Default matches nothing
        except re.error as e:
            self.logger.error(f"Invalid boundary regex for domain {self.ACTIVE_DOMAIN}: {self.instruction_json['sections'].get('boundary_regex')}. Error: {e}")
            self._BOUND_RX = re.compile(r"$^") # Fallback

        self._CUSTOM_SECT_RX = []
        for d in self.instruction_json["sections"].get("custom", []):
            try:
                 self._CUSTOM_SECT_RX.append((re.compile(d["regex"], re.I), d["label"]))
            except re.error as e:
                 self.logger.error(f"Invalid custom section regex '{d['regex']}' for domain {self.ACTIVE_DOMAIN}: {e}")


        self._METRIC_RX: List[Tuple[str, re.Pattern]] = []
        for key, cfg in self.instruction_json.get("metrics", {}).items():
            aliases = cfg.get("aliases", [])
            if aliases: # Ensure aliases is not empty
                try:
                    # Sort aliases by length descending to match longer phrases first
                    sorted_aliases = sorted(aliases, key=len, reverse=True)
                    joined = "|".join(re.escape(a) for a in sorted_aliases)
                    if joined:
                        # Regex: case-insensitive alias, optional separator, numeric value with optional currency/commas/decimals
                        # Allowing for negative numbers and potential spaces after currency symbol
                        pattern = re.compile(rf"""
                            (?i)\b({joined})\b             # Alias (Group 1)
                            [\s:–-]*                       # Optional separator
                            ([$€£]?\s?                     # Optional currency symbol with optional space (Group 2 - part 1)
                             -?                           # Optional negative sign
                             \d[\d,.]*                     # Number (digits with optional commas/decimals)
                            )
                            """, re.VERBOSE | re.MULTILINE)
                        self._METRIC_RX.append((key, pattern))
                except re.error as e:
                    self.logger.error(f"Invalid metric regex for alias group '{key}' in domain {self.ACTIVE_DOMAIN}: {e}")


        self._RISK_RX = {}
        for t, pat_str in self.instruction_json.get("risks", {}).items():
            try:
                self._RISK_RX[t] = re.compile(pat_str, re.I)
            except re.error as e:
                 self.logger.error(f"Invalid risk regex for '{t}' in domain {self.ACTIVE_DOMAIN}: '{pat_str}'. Error: {e}")


        self._DOC_LABELS = self.instruction_json["classification"].get("labels", [])
        self._CLASS_PROMPT_PREFIX = self.instruction_json["classification"].get("prompt_prefix", "")

        # ────────────────── Content Type Detection Patterns ─────────────────────
        # Heuristic patterns and weights for detecting content type
        self.PATTERNS: Dict[str, List[Tuple[Pattern, float]]] = {
            "html": [
                (re.compile(r"<html", re.I), 5.0),
                (re.compile(r"<head", re.I), 4.0),
                (re.compile(r"<body", re.I), 4.0),
                (re.compile(r"</(div|p|span|a|li)>", re.I), 1.0),
                (re.compile(r"<[a-z][a-z0-9]*\s+[^>]*>", re.I), 0.8), # Generic opening tag with attributes
                (re.compile(r"<!DOCTYPE", re.I), 5.0),
                (re.compile(r"&\w+;"), 0.5), # Entities
            ],
            "markdown": [
                (re.compile(r"^#{1,6}\s+", re.M), 4.0),         # Headings
                (re.compile(r"^\s*[-*+]\s+", re.M), 2.0),      # Unordered lists
                (re.compile(r"^\s*\d+\.\s+", re.M), 2.0),    # Ordered lists
                (re.compile(r"`[^`]+`"), 1.5),                # Inline code
                (re.compile(r"^```", re.M), 5.0),             # Code blocks
                (re.compile(r"\*{1,2}[^*\s]+?\*{1,2}"), 1.0), # Emphasis/Strong
                (re.compile(r"!\[.*?\]\(.*?\)", re.M), 3.0),   # Images
                (re.compile(r"\[.*?\]\(.*?\)", re.M), 2.5),   # Links
                (re.compile(r"^>.*", re.M), 2.0),             # Blockquotes
                (re.compile(r"^-{3,}$", re.M), 3.0),           # Horizontal rules
            ],
            "code": [
                (re.compile(r"def\s+\w+\(.*\):"), 3.0),      # Python function
                (re.compile(r"class\s+\w+"), 3.0),           # Class definition (generic)
                (re.compile(r"import\s+|from\s+"), 3.0),     # Import statements (Pythonic)
                (re.compile(r"((function\s+\w+\(|const|let|var)\s*.*?=>|\b(document|console|window)\.)"), 3.0), # JS/Java function
                (re.compile(r"public\s+|private\s+|static\s+"), 2.5), # Java/C# keywords
                (re.compile(r"#include"), 3.0),             # C/C++ include
                (re.compile(r"<\?php"), 4.0),                # PHP tag
                (re.compile(r"console\.log"), 2.0),          # JS logging
                (re.compile(r";\s*$"), 1.0),                 # Semicolon line endings (C-like, JS)
                (re.compile(r"\b(var|let|const|int|float|string|bool)\b"), 1.5), # Common variable keywords
                (re.compile(r"//.*$"), 1.0),                 # Single line comment
                (re.compile(r"/\*.*?\*/", re.S), 1.5),        # Multi-line comment
            ]
        }
        # Patterns for specific language detection within code blocks
        self.LANG_PATTERNS: List[Tuple[Pattern, str]] = [
            (re.compile(r"(def\s+\w+\(.*?\):|import\s+|from\s+\S+\s+import)"), "python"),
            (re.compile(r"((function\s+\w+\(|const|let|var)\s*.*?=>|\b(document|console|window)\.)"), "javascript"),
            (re.compile(r"<(\w+)(.*?)>.*?</\1>", re.S), "html"), # More specific HTML check
            (re.compile(r"<\?php"), "php"),
            (re.compile(r"(public|private|protected)\s+(static\s+)?(void|int|String)"), "java"),
            (re.compile(r"#include\s+<"), "c/c++"),
            (re.compile(r"using\s+System;"), "c#"),
            (re.compile(r"(SELECT|INSERT|UPDATE|DELETE)\s+.*FROM", re.I), "sql"),
            (re.compile(r":\s+\w+\s*\{"), "css"), # Basic CSS block detection
            (re.compile(r"^[^:]+:\s* # YAML key-value", re.M | re.X), "yaml"), # Simple YAML key check
            (re.compile(r"\$\w+"), "shell/bash"), # Basic shell variable
        ]


        # Lazy loading state
        self._tiktoken_enc_instance: Union["Encoding", bool, None] = None

        # Map of operations for batch processing
        # Defined here to ensure methods are bound to the instance
        self.op_map: Dict[str, Callable[..., Awaitable[Any]]] = {
            "convert_document": self.convert_document_op,
            "chunk_document": self.chunk_document,
            "summarize_document": self.summarize_document,
            "extract_entities": self.extract_entities,
            "generate_qa_pairs": self.generate_qa_pairs,
            "classify_document": self.classify_document,
            "identify_sections": self.identify_sections,
            "extract_metrics": self.extract_metrics,
            "flag_risks": self.flag_risks,
            "canonicalise_entities": self.canonicalise_entities,
            "extract_tables": self.extract_tables,
            "clean_and_format_text_as_markdown": self.clean_and_format_text_as_markdown,
            "optimize_markdown_formatting": self.optimize_markdown_formatting,
            "detect_content_type": self.detect_content_type,
            "batch_format_texts": self.batch_format_texts, # Add batch formatting itself as an op if needed? Seems recursive. Removing.
        }

        # Markdown Processing Regex
        self._BULLET_RX = re.compile(r"^[•‣▪◦‧﹒∙·] ?", re.MULTILINE)

    ###############################################################################
    # Utility Methods (Private)                                                   #
    ###############################################################################
    # ───────────────────── Core Utilities ─────────────────────────────
    def _converter(self, device, threads: int):
        """Create a DocumentConverter with specified device and threads."""
        if not self._docling_available:
            self.logger.warning("Docling not available. Document conversion will use basic fallback methods.")
            return None
            
        opts = PdfPipelineOptions()
        opts.do_ocr = False  # Default to no OCR for speed, can be overridden if needed
        opts.generate_page_images = False
        opts.accelerator_options = AcceleratorOptions(num_threads=threads, device=device)
        try:
            # Assuming PdfFormatOption applies generally; adjust if other formats need specific options
            return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        except Exception as e:
            self.logger.error(f"Failed to initialize DocumentConverter: {e}", exc_info=True)
            raise ToolError("INITIALIZATION_FAILED", details={"component": "DocumentConverter", "error": str(e)}) from e

    def _tmp_path(self, src: str, fmt: str) -> Path:
        """Generate a temporary file path based on source and format."""
        name = os.path.basename(src.split("?")[0]) or "document"
        stem = os.path.splitext(name)[0]
        ext = "md" if fmt == "markdown" else fmt
        # Add a timestamp/hash to avoid potential collisions in high-concurrency scenarios
        timestamp = int(time.time() * 1000)
        return Path(tempfile.gettempdir()) / f"{stem}_{timestamp}.{ext}"

    def _metadata(self, doc) -> dict[str, Any]:
        """Extract metadata from a document."""
        try:
            if self._docling_available and isinstance(doc, DoclingDocument):
                # For docling's Doc object
                num_pages = doc.num_pages() if callable(getattr(doc, 'num_pages', None)) else 0
                has_tables = any(p.content.has_tables() for p in doc.pages)
                has_figures = any(p.content.has_figures() for p in doc.pages)
                has_sections = bool(doc.get_sections()) # Check if sections list is non-empty
                
                return {
                    "num_pages": num_pages,
                    "has_tables": has_tables,
                    "has_figures": has_figures,
                    "has_sections": has_sections,
                }
            else:
                # For fallback text content, minimal metadata
                if isinstance(doc, dict):
                    # If we got a simple dictionary from fallback method
                    return doc.get("metadata", {"num_pages": 0, "has_tables": False, "has_figures": False, "has_sections": False})
                return {"num_pages": 0, "has_tables": False, "has_figures": False, "has_sections": False}
        except Exception as e:
            self.logger.warning("Metadata collection failed: %s", e, exc_info=True)
            # Return default values on error
            return {"num_pages": 0, "has_tables": False, "has_figures": False, "has_sections": False}

    @contextmanager
    def _span(self, label: str):
        """Context manager for timing operations."""
        st = time.perf_counter()
        try:
            yield
        finally:
            self.logger.debug(f"{label} {(time.perf_counter() - st):.3f}s")

    def _json(self, obj: Any) -> str:
        """Utility to serialize objects to JSON."""
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def _hash(self, txt: str) -> str:
        """Generate SHA-1 hash of text."""
        return hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()

    # ───────────────────── Lazy Loading Helpers ─────────────────────────────
    def _lazy_import_tiktoken(self):
        """Lazy import tiktoken."""
        if self._tiktoken_enc_instance is not None: # Already attempted loading
             return

        if tiktoken is None:
            self.logger.warning("Optional dependency tiktoken not found. Falling back to character-based tokenization.")
            self._tiktoken_enc_instance = False # Mark as unavailable
            return

        try:
            # Consider making the encoding configurable
            encoding_name = os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
            self.logger.info(f"Lazy-loading tiktoken encoding: {encoding_name}")
            self._tiktoken_enc_instance = tiktoken.get_encoding(encoding_name)
            self.logger.info("Successfully lazy-loaded tiktoken encoder.")
        except Exception as e:
            self.logger.error(f"Failed to lazy-load tiktoken: {e}", exc_info=True)
            self._tiktoken_enc_instance = False # Mark as unavailable


    # ───────────────────── HTML Utilities ─────────────────────────────
    def _is_html_fragment(self, text: str) -> bool:
        """Check if text contains likely HTML markup."""
        # Limit checks for performance on very long texts
        check_len = min(len(text), 5000)
        sample = text[:check_len]
        return any(p.search(sample) for p in self.HTML_PATTERNS)

    def _best_soup(self, html_txt: str) -> Tuple[BeautifulSoup, str]:
        """Try progressively more forgiving parsers; fall back to empty soup."""
        parsers = ("html.parser", "lxml", "html5lib")
        last_exception = None
        for p_name in parsers:
            try:
                # Check if the parser is available (lxml, html5lib might not be installed)
                try:
                    return BeautifulSoup(html_txt, p_name), p_name
                except ImportError:
                    self.logger.debug(f"HTML parser '{p_name}' not installed, skipping.")
                    continue
                except Exception as e_parse: # Catch parsing errors specifically
                    last_exception = e_parse
                    self.logger.debug(f"HTML parsing with '{p_name}' failed: {e_parse}")
                    continue
            except Exception as e_general: # Catch other potential errors during soup creation
                 last_exception = e_general
                 self.logger.warning(f"Unexpected error creating BeautifulSoup with '{p_name}': {e_general}")
                 continue

        # last chance – wrap fragment then try html.parser one more time
        # This is often needed for snippets lacking <html> or <body> tags
        if last_exception: # Log only if standard parsers failed
            self.logger.warning(f"All standard HTML parsers failed ({last_exception}), attempting fragment parsing.")
        wrapped_html = f"<html><body>{html_txt}</body></html>"
        try:
             # Try html.parser first as it's built-in
             return BeautifulSoup(wrapped_html, "html.parser"), "html.parser-fragment"
        except Exception as e_frag:
             self.logger.error(f"Fragment parsing also failed: {e_frag}. Returning empty soup.", exc_info=True)
             # Return an empty soup object if everything fails
             return BeautifulSoup("", "html.parser"), "failed"


    def _clean_html(self, html_txt: str) -> Tuple[str, str]:
        """Remove dangerous/pointless elements & attempt structural repair."""
        soup, parser_used = self._best_soup(html_txt)
        if parser_used == "failed":
             self.logger.warning("HTML cleaning skipped due to parsing failure.")
             return html_txt, parser_used # Return original text if parsing failed

        # Elements to remove entirely
        tags_to_remove = ["script", "style", "svg", "iframe", "canvas", "noscript", "meta", "link", "form", "input", "button", "select", "textarea", "nav", "aside", "header", "footer", "video", "audio"]
        for el in soup(tags_to_remove):
            el.decompose()

        # Attributes considered unsafe or unnecessary
        unsafe_attrs = ["style", "onclick", "onload", "onerror", "onmouseover", "onmouseout", "target"] # Added target to avoid _blank issues sometimes

        # strip unsafe attrs and event handlers
        for tag in soup.find_all(True):
            attrs_to_remove = []
            current_attrs = list(tag.attrs.keys()) # Iterate over a copy of keys
            for attr in current_attrs:
                 attr_val_str = str(tag.get(attr, ''))
                 if (
                     attr in unsafe_attrs
                     or attr.startswith("on") # Generic event handlers
                     or attr.startswith("data-") # Often framework-specific, not content
                     or (attr == "src" and ("javascript:" in attr_val_str.lower() or "data:" in attr_val_str.lower()))
                     or (attr == "href" and attr_val_str.lower().startswith("javascript:"))
                 ):
                     attrs_to_remove.append(attr)
            for attr in attrs_to_remove:
                 if attr in tag.attrs: # Check if still exists (might be removed by previous step)
                     del tag[attr]

        # entity fix + whitespace collapse
        try:
            # Convert back to string, handle entities, collapse whitespace
            text = str(soup)
            text = html.unescape(text) # Decode HTML entities like &amp;
            text = re.sub(r"[ \t\r\f\v]+", " ", text) # Collapse horizontal whitespace
            text = re.sub(r"\n\s*\n", "\n\n", text) # Collapse multiple blank lines to one
            text = text.strip()
        except Exception as e:
            self.logger.error(f"Error during HTML text processing (unescape/regex): {e}", exc_info=True)
            # Return the current state of the soup as a string if post-processing fails
            try:
                return str(soup), parser_used
            except Exception as stringify_error:
                 self.logger.error(f"Could not even stringify soup after error: {stringify_error}")
                 return html_txt, parser_used # Fallback to original text


        return text, parser_used

    # ───────────────────── LLM Helper ─────────────────────────────
    async def _llm(self, *, prompt: str, provider: str = Provider.OPENAI.value, model: str | None = None,
                   temperature: float = 0.3, max_tokens: int | None = None, extra: Dict[str, Any] | None = None) -> str:
        """Generate text completion using LLM. Returns the generated text content."""
        try:
            # Ensure generate_completion is available and callable
            if not callable(generate_completion):
                raise ToolError("LLM_UNAVAILABLE", details={"reason": "generate_completion not available or not callable"})

            # Call generate_completion and expect a specific response structure
            # Assuming generate_completion returns a dict with a 'content' key for the text
            response_dict = await generate_completion(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                additional_params=extra or {}
            )

            if isinstance(response_dict, dict) and 'content' in response_dict:
                 llm_content = response_dict['content']
                 if isinstance(llm_content, str):
                      return llm_content.strip() # Return the stripped text content
                 else:
                      self.logger.warning(f"LLM response 'content' is not a string: {type(llm_content)}. Converting.")
                      return str(llm_content).strip()
            else:
                 # Handle unexpected response format
                 self.logger.error(f"LLM response has unexpected format: {response_dict}")
                 raise ToolError("LLM_INVALID_RESPONSE", details={"response_received": str(response_dict)})

        except ToolError as e: # Catch ToolErrors raised by generate_completion or above
             raise e # Re-raise specific ToolErrors
        except Exception as e:
             self.logger.error(f"LLM call failed: {e}", exc_info=True)
             raise ToolError("LLM_CALL_FAILED", details={"error": str(e)}) from e

    # ───────────────────── Markdown Processing Utils ─────────────────────
    def _sanitize(self, md: str) -> str:
        """Basic Markdown sanitization."""
        if not md:
            return ""
        md = md.replace("\u00a0", " ")  # Replace non-breaking space with regular space
        md = self._BULLET_RX.sub("- ", md) # Standardize initial bullet characters
        md = re.sub(r"\n{3,}", "\n\n", md) # Collapse excessive newlines
        md = re.sub(r" +$", "", md, flags=re.MULTILINE) # Remove trailing whitespace from lines
        md = re.sub(r"^[ \t]+", "", md, flags=re.MULTILINE) # Remove leading whitespace from lines (can be aggressive, check impact)
        md = re.sub(r"(^|\n)(#{1,6})([^#\s])", r"\1\2 \3", md) # Ensure space after heading hashes
        md = re.sub(r"```\s*\n", "```\n", md) # Normalize code block start fence
        md = re.sub(r"\n\s*```", "\n```", md) # Normalize code block end fence
        md = re.sub(r"^[*+]\s", "- ", md, flags=re.MULTILINE) # Standardize list bullets to '-'
        md = re.sub(r"^\d+\.\s", lambda m: f"{m.group(0).strip()} ", md, flags=re.MULTILINE) # Ensure space after ordered list numbers
        return md.strip() # Remove leading/trailing whitespace from the whole text


    def _improve(self, md: str) -> str:
        """Apply structural improvements to Markdown text."""
        if not md:
            return ""

        # Ensure blank lines around major block elements for better readability
        # Headings
        md = re.sub(r"(?<=\n\S[^\n]*)\n(#{1,6}\s+)", r"\n\n\1", md) # Add blank line before heading if needed
        md = re.sub(r"(#{1,6}\s+[^\n]*)\n(?=\S)", r"\1\n\n", md) # Add blank line after heading if needed

        # Code blocks
        md = re.sub(r"(?<=\n\S[^\n]*)\n(```)", r"\n\n\1", md) # Add blank line before code block fence
        md = re.sub(r"(```)\n(?=\S)", r"\1\n\n", md) # Add blank line after code block fence

        # Blockquotes
        md = re.sub(r"(?<=\n\S[^\n]*)\n(> )", r"\n\n\1", md) # Add blank line before blockquote
        md = re.sub(r"(\n> [^\n]*)\n(?=[^>\s])", r"\1\n\n", md) # Add blank line after blockquote if followed by non-quote/non-whitespace

        # Lists (Unordered and Ordered)
        # Ensure blank line before the start of a list block
        md = re.sub(r"(?<=\n\S[^\n]*)\n(\s*[-*+]\s+|\s*\d+\.\s+)", r"\n\n\1", md)
        # Ensure blank line after the end of a list block
        # This looks for the last list item followed by a line not starting as a list item (or whitespace)
        md = re.sub(r"(\n(\s*[-*+]\s+|\s*\d+\.\s+)[^\n]*)\n(?!\s*([-*+]|\d+\.)\s+)(\S)", r"\1\n\n\4", md)

        # Paragraphs: Ensure paragraphs are separated by exactly one blank line
        # This is tricky. Let's try replacing 3+ newlines with 2 first (done in _sanitize).
        # Then ensure blocks separated by a single newline get a double newline, UNLESS it's list items etc.
        # This needs careful negative lookaheads/lookbehinds, maybe simpler to split/join?

        # Split into lines, identify block types, then join with appropriate newlines.
        lines = md.splitlines()
        improved_lines = []
        
        # Add each line directly without additional processing
        # The regex substitutions above handle the proper spacing between elements
        for line in lines:
            improved_lines.append(line)

        # Re-join lines and apply final cleanup
        md = "\n".join(improved_lines)
        md = re.sub(r"\n{3,}", "\n\n", md) # Collapse excessive newlines again

        return md.strip()


    def _convert_html_table_to_markdown(self, table_tag: Tag) -> str:
        """Converts a single BeautifulSoup table Tag to a Markdown string."""
        md_rows = []
        header_cells = table_tag.find_all(['th', 'td'], recursive=False) # Check direct children first
        if not header_cells: # Check first tr if no direct th/td
            first_row = table_tag.find('tr')
            if first_row:
                header_cells = first_row.find_all(['th', 'td'])

        if not header_cells: # Still no header cells found in first row
             self.logger.debug("Table has no header cells identifiable in first row.")
             # Try to determine column count from the row with the most cells
             all_rows_tags = table_tag.find_all('tr')
             if not all_rows_tags: 
                 return "" # Empty table
             num_cols = 0
             for r in all_rows_tags:
                  num_cols = max(num_cols, len(r.find_all(['th', 'td'])))
             if num_cols == 0: 
                 return "" # Table with rows but no cells

             # Create a dummy header if none exists
             md_rows.append("| " + " | ".join(['Column'] * num_cols) + " |") # Generic header
             md_rows.append("| " + " | ".join(['---'] * num_cols) + " |")
        else:
            # Process identified header cells
            num_cols = len(header_cells)
            hdr = [" ".join(c.get_text(" ", strip=True).replace("|", "\\|").split()) for c in header_cells]
            md_rows.append("| " + " | ".join(hdr) + " |")
            md_rows.append("| " + " | ".join(['---'] * num_cols) + " |")

        # Process data rows (all 'tr' elements)
        body_rows = table_tag.find_all('tr')
        # Skip the first row if we used it for headers explicitly
        start_row_index = 1 if header_cells and body_rows and header_cells[0].find_parent('tr') == body_rows[0] else 0

        for r in body_rows[start_row_index:]:
            cells = r.find_all('td')
            # Handle cells within the row, pad/truncate to match header column count
            cell_texts = [" ".join(c.get_text(" ", strip=True).replace("|", "\\|").split()) for c in cells]
            # Pad if fewer cells than expected
            cell_texts.extend([""] * (num_cols - len(cells)))
            # Truncate if more cells than expected
            cell_texts = cell_texts[:num_cols]

            md_rows.append("| " + " | ".join(cell_texts) + " |")

        return "\n".join(md_rows)

    def _convert_html_tables_to_markdown(self, html_txt: str) -> str:
        """Finds HTML tables and replaces them with Markdown format."""
        soup, parser_used = self._best_soup(html_txt)
        if parser_used == "failed":
            self.logger.warning("Skipping HTML table conversion due to parsing failure.")
            return html_txt

        tables = soup.find_all("table")
        if not tables:
            return html_txt # No tables found

        self.logger.debug(f"Found {len(tables)} HTML tables to convert to Markdown.")
        for table_tag in tables:
            try:
                 md_table_str = self._convert_html_table_to_markdown(table_tag)
                 if md_table_str:
                      # Replace the table tag with the Markdown string.
                      # Important: Need to add newlines around it for proper rendering.
                      # Using a placeholder first avoids modifying the soup while iterating.
                      placeholder = BeautifulSoup(f"\n\n{md_table_str}\n\n", "html.parser")
                      table_tag.replace_with(placeholder)
                 else:
                      # Remove the table if conversion failed or resulted in empty string
                      table_tag.decompose()
            except Exception as e:
                 self.logger.error(f"Failed to convert a table to Markdown: {e}", exc_info=True)
                 # Optionally keep the original table HTML or remove it on error
                 # table_tag.decompose() # Uncomment to remove table on conversion error

        # Return the modified HTML (now with Markdown tables) as a string
        return str(soup)


    ###############################################################################
    # Document Conversion Tool                                                    #
    ###############################################################################
    @tool(name="convert_document", description="Convert document (PDF, Office) to specified format (markdown, text, html, json, doctags)")
    @with_tool_metrics
    @with_error_handling
    async def convert_document(
        self,
        document_path: str,
        output_format: str = "markdown",
        output_path: Optional[str] = None,
        save_to_file: bool = False,
        accelerator_device: str = "auto",
        num_threads: int = 4,
        page_range: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert document to specified format.

        Args:
            document_path: Path to the document to convert
            output_format: Format to convert to (markdown, text, html, json, doctags)
            output_path: Path to save the output file (if save_to_file is True)
            save_to_file: Whether to save the output to a file
            accelerator_device: Device to use for acceleration (auto, cpu, cuda, mps)
            num_threads: Number of threads to use for processing
            page_range: Range of pages to convert (e.g. "1-3,7,10") - Applied post-conversion if format allows.
            section_filter: Regex to filter sections (applied post-conversion for text-based formats)

        Returns:
            Dictionary with conversion results
        """
        t0 = time.time()

        doc_path = Path(document_path)
        if not doc_path.is_file():
             raise ToolInputError(f"Document not found at path: {document_path}", param_name="document_path")

        output_format = output_format.lower()
        if output_format not in self._VALID_FORMATS:
            raise ToolInputError(f"output_format must be one of {', '.join(self._VALID_FORMATS)}", param_name="output_format", provided_value=output_format)
        if accelerator_device.lower() not in self._ACCEL_MAP:
            raise ToolInputError(f"accelerator_device must be one of {', '.join(self._ACCEL_MAP)}", param_name="accelerator_device", provided_value=accelerator_device)

        device = self._ACCEL_MAP[accelerator_device.lower()]
        
        # Document conversion - with fallback path
        content = ""
        doc_metadata = {}
        
        try:
            self.logger.info(f"Starting conversion for {document_path} to {output_format}")
            
            if self._docling_available:
                # Use docling if available
                self.logger.info(f"Using docling with {accelerator_device} accelerator")
                conv = self._converter(device, num_threads)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, conv.convert, doc_path)

                if not result or not result.document:
                    raise ToolError("CONVERSION_FAILED", details={"document_path": document_path, "reason": "Converter returned empty result"})
                
                # Get docling document object and extract content based on format
                doc_obj = result.document
                doc_metadata = self._metadata(doc_obj)
                
                # Apply docling-specific filters
                if page_range:
                    # Code for docling page range filtering
                    # ...
                    pass  # Simplified for brevity
                
                # Get content based on output format
                if output_format == "markdown":
                    content = doc_obj.export_to_markdown()
                elif output_format == "text":
                    content = doc_obj.export_to_text()
                elif output_format == "html":
                    content = doc_obj.export_to_html()
                elif output_format == "json":
                    content = self._json(doc_obj.export_to_dict())
                elif output_format == "doctags":
                    content = doc_obj.export_to_doctags()
                
                # Save to file if requested
                if save_to_file:
                    fp = Path(output_path) if output_path else self._tmp_path(document_path, output_format)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    
                    if output_format == "markdown":
                        doc_obj.save_as_markdown(fp, image_mode=ImageRefMode.PLACEHOLDER)
                    elif output_format == "text":
                        # Save text as markdown file
                        doc_obj.save_as_markdown(fp, image_mode=ImageRefMode.PLACEHOLDER)
                    elif output_format == "html":
                        doc_obj.save_as_html(fp, image_mode=ImageRefMode.REFERENCED)
                    elif output_format == "json":
                        doc_obj.save_as_json(fp, image_mode=ImageRefMode.PLACEHOLDER)
                    elif output_format == "doctags":
                        doc_obj.save_as_doctags(fp)
                        
                    self.logger.info(f"Saved docling output to {fp}")
                
            else:
                # Use basic fallback methods when docling is not available
                self.logger.info(f"Using basic fallback conversion methods for {document_path}")
                fallback_result = {}
                
                # Choose conversion method based on file type
                suffix = doc_path.suffix.lower()
                if suffix == '.pdf':
                    fallback_result = await self._fallback_convert_pdf(doc_path)
                elif suffix == '.docx':
                    fallback_result = await self._fallback_convert_docx(doc_path)
                elif suffix in ['.txt', '.md', '.markdown', '.html', '.htm']:
                    fallback_result = await self._fallback_convert_text(doc_path)
                else:
                    # Try basic text fallback for unknown types
                    self.logger.warning(f"Unknown file type: {suffix}, attempting basic text extraction")
                    fallback_result = await self._fallback_convert_text(doc_path)
                
                # Extract content and metadata from fallback result
                content = fallback_result.get("content", "")
                doc_metadata = fallback_result.get("metadata", {})
                
                # Apply simple filters
                if section_filter and isinstance(content, str):
                    try:
                        pat = re.compile(section_filter, re.I | re.M)
                        blocks = re.split(r"\n\s*\n", content)
                        kept_blocks = [blk for blk in blocks if pat.search(blk)]
                        content = "\n\n".join(kept_blocks)
                        self.logger.info(f"Applied section filter: '{section_filter}'")
                    except Exception as e:
                        self.logger.warning(f"Failed to apply section filter: {e}")
                
                # For fallback, adapt output based on requested format
                if output_format == "html":
                    content = f"<html><body><pre>{html.escape(content)}</pre></body></html>"
                elif output_format == "json":
                    content = self._json({"content": content, "metadata": doc_metadata})
                
                # Save to file if requested
                if save_to_file:
                    fp = Path(output_path) if output_path else self._tmp_path(document_path, output_format)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    
                    if output_format == "html":
                        fp.write_text(content, encoding="utf-8")
                    elif output_format == "json":
                        fp.write_text(content, encoding="utf-8")
                    else:
                        # For markdown, text, and doctags in fallback mode, just save content as is
                        fp.write_text(content, encoding="utf-8")
                    
                    self.logger.info(f"Saved fallback output to {fp}")
            
            self.logger.info(f"Conversion successful for {document_path}.")
            
        except Exception as exc:
            self.logger.error(f"Conversion failed for {document_path}", exc_info=True)
            raise ToolError("CONVERSION_FAILED", details={"document_path": document_path, "error": str(exc)}) from exc

        elapsed = round(time.time() - t0, 3)
        resp: Dict[str, Any] = {
            "success": True,
            "content": content,
            "output_format": output_format,
            "processing_time": elapsed,
            "document_metadata": doc_metadata,
            "used_fallback": not self._docling_available,
        }
        if save_to_file and 'fp' in locals():
            resp["file_path"] = str(fp)

        self.logger.info(f"Completed conversion {document_path} → {output_format} in {elapsed}s")
        return resp

    # Wrapper for batch processing compatibility
    async def convert_document_op(self, document_path: str, *, output_format: str = "markdown", save_to_file: bool = False,
                                  output_path: Optional[str] = None, page_range: Optional[str] = None,
                                  section_filter: Optional[str] = None, **extra_kwargs) -> Dict[str, Any]:
        """Internal wrapper for convert_document used by batch processing."""
        # Propagate relevant kwargs from batch operation spec to the tool function
        res = await self.convert_document(
                document_path=document_path,
                output_format=output_format,
                save_to_file=save_to_file,
                output_path=output_path,
                page_range=page_range,
                section_filter=section_filter,
                accelerator_device=extra_kwargs.get("accelerator_device", "auto"),
                num_threads=extra_kwargs.get("num_threads", 4)
            )
        # Result format from convert_document is already compatible with batch needs
        return res

    async def _fallback_convert_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Basic PDF conversion using PyPDF2 when docling is not available."""
        if PyPDF2 is None:
            raise ToolError("DEPENDENCY_MISSING", details={"dependency": "PyPDF2", "for": "PDF conversion fallback"})
            
        try:
            self.logger.info(f"Using PyPDF2 fallback for PDF: {file_path}")
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for i in range(len(reader.pages)):
                    page_text = reader.pages[i].extract_text() or ""
                    pages.append(page_text)
                
                content = "\n\n".join(pages)
                metadata = {"num_pages": len(reader.pages), "has_tables": False, "has_figures": False, "has_sections": False}
                
                return {"content": content, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"PyPDF2 fallback failed: {e}", exc_info=True)
            raise ToolError("CONVERSION_FAILED", details={"file": str(file_path), "error": str(e)}) from e
    
    async def _fallback_convert_docx(self, file_path: Path) -> Dict[str, Any]:
        """Basic DOCX conversion using python-docx when docling is not available."""
        if docx is None:
            raise ToolError("DEPENDENCY_MISSING", details={"dependency": "python-docx", "for": "DOCX conversion fallback"})
            
        try:
            self.logger.info(f"Using python-docx fallback for DOCX: {file_path}")
            doc = docx.Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs]
            content = "\n".join(paragraphs)
            
            # Minimal metadata
            metadata = {
                "num_pages": 0,  # python-docx doesn't easily provide page count
                "has_tables": len(doc.tables) > 0,
                "has_figures": False,
                "has_sections": len(doc.sections) > 0
            }
            
            return {"content": content, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"python-docx fallback failed: {e}", exc_info=True)
            raise ToolError("CONVERSION_FAILED", details={"file": str(file_path), "error": str(e)}) from e
    
    async def _fallback_convert_text(self, file_path: Path) -> Dict[str, Any]:
        """Simple text file reading when docling is not available."""
        try:
            self.logger.info(f"Reading text file directly: {file_path}")
            content = file_path.read_text(encoding='utf-8', errors='replace')
            
            # Count pages by newlines (rough estimate)
            line_count = content.count('\n') + 1
            page_estimate = max(1, int(line_count / 40))  # ~40 lines per page
            
            metadata = {"num_pages": page_estimate, "has_tables": False, "has_figures": False, "has_sections": False}
            
            return {"content": content, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"Text file reading failed: {e}", exc_info=True)
            raise ToolError("CONVERSION_FAILED", details={"file": str(file_path), "error": str(e)}) from e

    ###############################################################################
    # Document Chunking Tool & Helpers                                            #
    ###############################################################################

    def _token_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by tokens, respecting sentence boundaries."""
        self._lazy_import_tiktoken()
        if self._tiktoken_enc_instance is False or not callable(getattr(self._tiktoken_enc_instance, 'encode', None)):
            self.logger.warning("Tiktoken not available or invalid, falling back to character chunking for token method.")
            # Estimate character size roughly (e.g., 4 chars per token)
            char_size = size * 4
            char_overlap = overlap * 4
            return self._char_chunks(doc, char_size, char_overlap)
        if not doc:
            return []

        enc = self._tiktoken_enc_instance # Type assertion helps type checkers
        try:
            tokens = enc.encode(doc, disallowed_special=()) # Allow special tokens for better boundary detection? Check effect.
        except Exception as e:
            self.logger.error(f"Tiktoken encoding failed: {e}. Falling back to character chunking.", exc_info=True)
            char_size = size * 4
            char_overlap = overlap * 4
            return self._char_chunks(doc, char_size, char_overlap)

        if not tokens:
            return []

        chunks: List[str] = []
        current_pos = 0
        n_tokens = len(tokens)

        # Define sentence ending tokens (heuristic, may vary by model/encoding)
        # Encode common sentence terminators to get their token IDs
        try:
             sentence_end_tokens = {enc.encode(p)[0] for p in (".", "?", "!", "\n")}
        except Exception as e:
             self.logger.warning(f"Could not encode sentence end markers: {e}. Using default IDs.")
             # Fallback common IDs for cl100k_base (APPROXIMATE!)
             sentence_end_tokens = {13, 30, 106, 198} # '.', '?', '!', '\n' token


        while current_pos < n_tokens:
            end_pos = min(current_pos + size, n_tokens)

            # If the chunk is already at the end, just take it
            if end_pos == n_tokens:
                best_split_pos = n_tokens
            else:
                # Try to find a sentence break within the lookback window before the end_pos
                # Look back up to 'overlap' tokens or a fixed amount, whichever is smaller
                lookback_distance = min(overlap, size // 4, end_pos - current_pos) # Sensible lookback
                search_start = max(current_pos, end_pos - lookback_distance)

                best_split_pos = end_pos # Default to hard cut if no break found
                # Iterate backwards from end_pos-1 to search_start
                for k in range(end_pos - 1, search_start -1, -1):
                    if tokens[k] in sentence_end_tokens:
                        best_split_pos = k + 1 # Split after the sentence end token
                        break

            chunk_token_ids = tokens[current_pos:best_split_pos]
            if not chunk_token_ids:
                 # This should not happen if logic is correct, but as a safeguard
                 if current_pos >= n_tokens: 
                     break
                 current_pos += 1 # Move past problematic position
                 continue

            try:
                 chunk_text = enc.decode(chunk_token_ids).strip()
                 if chunk_text: # Only add non-empty chunks
                      chunks.append(chunk_text)
            except Exception:
                 # If decoding fails, log and skip chunk
                 self.logger.error(f"Tiktoken decoding failed for tokens {current_pos}:{best_split_pos}. Skipping chunk.", exc_info=True)
                 # Attempt to recover the raw tokens as string? Probably not useful.

            # Move start position for next chunk
            next_start_pos = best_split_pos - overlap
            # Ensure forward progress: move at least one token, unless overlap is huge or at end
            current_pos = max(current_pos + 1, next_start_pos) if best_split_pos > current_pos else best_split_pos


        return chunks


    def _char_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by characters, respecting sentence/paragraph boundaries."""
        if not doc:
            return []

        chunks: List[str] = []
        current_pos = 0
        n_chars = len(doc)
        # Prefer splitting at sentence endings or paragraph breaks
        sentence_ends = (".", "?", "!", "\n\n", "\n") # Prioritize double newline
        # Less preferred but acceptable breaks (e.g., list items, commas)
        softer_breaks = (";", ":", ",", "\t", " ")

        while current_pos < n_chars:
            end_pos = min(current_pos + size, n_chars)

            # If the chunk is already at the end, take it
            if end_pos == n_chars:
                 best_split_pos = n_chars
            else:
                # Find the best split point by looking backward from end_pos
                best_split_pos = -1 # Sentinel value

                # Look back within a reasonable window (e.g., last 20% or 100 chars)
                lookback_window_start = max(current_pos, end_pos - int(size * 0.2), end_pos - 100)

                # 1. Search for preferred sentence/paragraph breaks backwards
                for marker in sentence_ends:
                     try:
                          # Find the last occurrence of the marker within the relevant part of the chunk
                          found_pos = doc.rfind(marker, lookback_window_start, end_pos)
                          if found_pos != -1:
                              # Position split *after* the marker
                              split_candidate = found_pos + len(marker)
                              # Update best_split_pos if this split is later than current best
                              if split_candidate > best_split_pos:
                                   best_split_pos = split_candidate
                     except Exception: 
                         pass # Ignore errors from rfind

                # 2. If no preferred break found, search for softer breaks backwards
                if best_split_pos == -1:
                     for marker in softer_breaks:
                          try:
                              found_pos = doc.rfind(marker, lookback_window_start, end_pos)
                              if found_pos != -1:
                                  split_candidate = found_pos + len(marker)
                                  if split_candidate > best_split_pos:
                                       best_split_pos = split_candidate
                          except Exception: 
                              pass

                # 3. If still no break found, use the hard cut-off point
                if best_split_pos == -1 or best_split_pos <= current_pos: # Ensure split is after current_pos
                     best_split_pos = end_pos


            actual_chunk_text = doc[current_pos:best_split_pos].strip()
            if actual_chunk_text: # Avoid adding empty chunks
                 chunks.append(actual_chunk_text)

            # Calculate next start position with overlap, ensuring forward progress
            next_start_pos = best_split_pos - overlap
            # Move at least one character forward unless overlap is huge or at end
            current_pos = max(current_pos + 1, next_start_pos) if best_split_pos > current_pos else best_split_pos

        return chunks


    def _paragraph_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by paragraphs, combining small ones up to size limit."""
        if not doc:
            return []
        # Split by one or more blank lines (more robust)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', doc) if p.strip()]
        if not paragraphs:
            return []

        chunks = []
        current_chunk_paragraphs: List[str] = []
        current_chunk_len = 0
        # Use character length for paragraph size check, unless tiktoken is available?
        # For simplicity and consistency with char_chunks, use char length.
        use_tiktoken_len = self._tiktoken_enc_instance and callable(getattr(self._tiktoken_enc_instance, 'encode', None))

        def get_len(text: str) -> int:
             if use_tiktoken_len:
                 try: 
                     return len(self._tiktoken_enc_instance.encode(text)) # type: ignore
                 except Exception: 
                     return len(text) # Fallback on error
             return len(text)

        for p in paragraphs:
            p_len = get_len(p)
            potential_new_len = current_chunk_len + (get_len("\n\n") if current_chunk_paragraphs else 0) + p_len

            # If adding the next paragraph exceeds size (and current chunk is not empty)
            if current_chunk_paragraphs and potential_new_len > size:
                # Finalize the current chunk
                chunks.append("\n\n".join(current_chunk_paragraphs))

                # Start new chunk, considering overlap.
                # Overlap logic for paragraphs is tricky. We can include the last few
                # paragraphs from the previous chunk or just start with the current paragraph.
                # Simple approach: Start new chunk with the current paragraph if it fits.
                if p_len <= size:
                     current_chunk_paragraphs = [p]
                     current_chunk_len = p_len
                else:
                     # Paragraph itself is too long. Add it as its own oversized chunk
                     # or split it further using character/token chunking.
                     self.logger.warning(f"Paragraph starting with '{p[:50]}...' (length {p_len}) exceeds chunk size {size}. Splitting paragraph.")
                     # Split the oversized paragraph using character chunking (respects 'size')
                     sub_chunks = self._char_chunks(p, size, overlap) # Use char chunking here
                     chunks.extend(sub_chunks)
                     # Reset current chunk tracking
                     current_chunk_paragraphs = []
                     current_chunk_len = 0

            # If current chunk is empty or adding the paragraph fits
            else:
                 # Handle case where the very first paragraph is too large
                 if not current_chunk_paragraphs and p_len > size:
                      self.logger.warning(f"First paragraph starting with '{p[:50]}...' (length {p_len}) exceeds chunk size {size}. Splitting paragraph.")
                      sub_chunks = self._char_chunks(p, size, overlap)
                      chunks.extend(sub_chunks)
                      current_chunk_paragraphs = [] # Ensure reset
                      current_chunk_len = 0
                 else:
                      # Add paragraph to current chunk
                      current_chunk_paragraphs.append(p)
                      current_chunk_len = potential_new_len


        # Add the last remaining chunk if it's not empty
        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        self.logger.info(f"Chunked into {len(chunks)} paragraphs/groups.")
        return chunks


    async def _section_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by identified sections. Falls back to paragraphs if sections fail."""
        try:
             section_result = await self.identify_sections(document=doc)
             # Check if section_result is the expected format and has sections
             if isinstance(section_result, dict) and section_result.get('success') and isinstance(section_result.get('sections'), list):
                  sections = section_result['sections']
             else:
                  self.logger.warning("Identify_sections did not return expected format. Falling back to paragraph chunking.")
                  # Run paragraph chunking in executor as it might be CPU-bound
                  loop = asyncio.get_running_loop()
                  return await loop.run_in_executor(None, self._paragraph_chunks, doc, size, overlap)

             if not sections:
                  self.logger.info("No sections identified, using paragraph chunking as fallback.")
                  loop = asyncio.get_running_loop()
                  return await loop.run_in_executor(None, self._paragraph_chunks, doc, size, overlap)

             # Extract text from each section. Optionally, include title.
             # Let's include the title for context.
             section_texts: List[str] = []
             for s in sections:
                 title = s.get("title", "").strip()
                 text = s.get("text", "").strip()
                 if text: # Only add sections with actual text content
                     full_section_text = f"# {title}\n\n{text}" if title and title != "Introduction" and title != "Main Content" else text
                     section_texts.append(full_section_text.strip())

             # If sections are too large, chunk them further.
             final_chunks = []
             loop = asyncio.get_running_loop()
             for text in section_texts:
                  # Use character length check, consistent with paragraph chunking fallback
                  if len(text) > size * 1.1: # Allow slightly larger sections before splitting
                      self.logger.warning(f"Section chunk starting with '{text[:50]}...' exceeds size limit ({len(text)} > {size}). Sub-chunking section using paragraphs.")
                      # Sub-chunk the large section using paragraph strategy (run in executor)
                      sub_chunks = await loop.run_in_executor(None, self._paragraph_chunks, text, size, overlap)
                      final_chunks.extend(sub_chunks)
                  elif text: # Add section if it fits and is not empty
                      final_chunks.append(text)

             return final_chunks

        except Exception as e:
             self.logger.error(f"Failed to get sections for chunking: {e}. Falling back to paragraph chunking.", exc_info=True)
             loop = asyncio.get_running_loop()
             return await loop.run_in_executor(None, self._paragraph_chunks, doc, size, overlap)


    @tool(name="chunk_document", description="Split document text into chunks using various strategies")
    @with_tool_metrics
    @with_error_handling
    async def chunk_document(self, document: str, *, chunk_size: int = 1000, chunk_method: str = "paragraph",
                             chunk_overlap: int = 0, chunk_strategy: Optional[str] = None) -> Dict[str, Any]:
        """Split document into chunks based on specified method and size.

        Args:
            document: Text content to chunk.
            chunk_size: Target maximum size of each chunk (meaning depends on method: tokens or characters).
            chunk_method: Chunking method ('token', 'character', 'section', 'paragraph').
            chunk_overlap: Number of tokens/characters to overlap between chunks (for token/char methods).
                           For paragraph/section, overlap is handled differently (less precise).
            chunk_strategy: Alias for chunk_method (for backward compatibility).

        Returns:
            Dictionary containing list of chunked text strings.
            Example: {"chunks": ["chunk 1 text...", "chunk 2 text..."], "success": True}
        """
        if not document or not isinstance(document, str):
             self.logger.warning("Chunking called with empty or invalid document input.")
             return {"chunks": [], "success": True} # Return success with empty list

        size = max(50, int(chunk_size)) # Minimum chunk size 50
        # Ensure overlap is reasonable relative to size
        overlap = max(0, min(int(chunk_overlap), size // 3))
        method = (chunk_strategy or chunk_method or "paragraph").lower() # Default to paragraph

        # Map method name to the actual chunking function (instance method)
        chunker_map: Dict[str, Callable[..., Union[List[str], Awaitable[List[str]]]]] = {
            "token": self._token_chunks,
            "character": self._char_chunks,
            "section": self._section_chunks, # This is async
            "paragraph": self._paragraph_chunks,
        }

        strat_func = chunker_map.get(method)
        if not strat_func:
             self.logger.warning(f"Unknown chunk_method '{method}'. Defaulting to 'paragraph'.")
             strat_func = self._paragraph_chunks
             method = "paragraph"


        self.logger.info(f"Chunking document using method='{method}', size={size}, overlap={overlap}")
        chunks: List[str] = []
        try:
            with self._span(f"chunk_document_{method}"):
                if asyncio.iscoroutinefunction(strat_func):
                    # Await async chunkers like _section_chunks directly
                    chunks = await strat_func(document, size, overlap) # type: ignore
                else:
                     # Run potentially CPU-bound sync chunkers in an executor thread
                     loop = asyncio.get_running_loop()
                     chunks = await loop.run_in_executor(
                          None, # Use default executor
                          strat_func, # The sync function (e.g., _token_chunks, _char_chunks, _paragraph_chunks)
                          document, size, overlap
                     )
        except Exception as e:
            self.logger.error(f"Error during chunking operation ({method}): {e}", exc_info=True)
            raise ToolError("CHUNKING_FAILED", details={"method": method, "error": str(e)}) from e


        # Filter out any potential empty strings resulting from chunking process
        final_chunks = [c for c in chunks if isinstance(c, str) and c]
        self.logger.info(f"Generated {len(final_chunks)} chunks.")

        return {"chunks": final_chunks, "success": True}


    ###############################################################################
    # HTML Processing Tools                                                       #
    ###############################################################################

    def _extract_readability(self, html_txt: str) -> str:
        """Extract main content using readability-lxml."""
        if not readability:
             self.logger.warning("Readability-lxml not installed. Cannot use readability extraction.")
             return "" # Return empty string, let caller handle fallback
        try:
            doc = readability.Document(html_txt)
            # Return the main content as HTML string
            return doc.summary(html_partial=True)
        except Exception as e:
             self.logger.warning(f"Readability extraction failed: {e}", exc_info=True)
             return "" # Return empty string on failure


    def _extract_trafilatura(self, html_txt: str) -> str:
        """Extract main content using trafilatura."""
        if not trafilatura:
            self.logger.warning("Trafilatura not installed. Cannot use trafilatura extraction.")
            return "" # Return empty string, let caller handle fallback
        try:
            # Configure trafilatura settings
            extracted = trafilatura.extract(
                html_txt,
                include_comments=False,
                include_tables=True,      # Keep table structure if possible
                favor_precision=True,   # Prioritize cleaner extraction over recall
                no_fallback=False,      # Allow fallback extractors within trafilatura if main fails
                output_format='html'      # Keep HTML structure for potential further processing
            )
            return extracted or "" # Return empty string if extraction yields None
        except Exception as e:
             self.logger.warning(f"Trafilatura extraction failed: {e}", exc_info=True)
             return "" # Return empty string on failure


    def _html_to_md_core(
        self,
        html_txt: str,
        links: bool,
        imgs: bool,
        tbls: bool,
        width: int,
    ) -> str:
        """Convert HTML to Markdown using primary and fallback libraries."""
        md_text = ""
        # Primary: html2text (generally robust)
        try:
             h = html2text.HTML2Text()
             h.ignore_links = not links
             h.ignore_images = not imgs
             h.ignore_tables = not tbls # html2text basic table support
             h.body_width = width if width > 0 else 0 # 0 means don't wrap lines
             h.unicode_snob = True # Prefer unicode chars
             h.escape_snob = True # Escape special markdown chars
             h.skip_internal_links = True # Don't include links like '#anchor'
             # Configure table handling (optional, defaults are usually ok)
             # h.table_def = "| %s |" # Example: change cell definition
             # h.th_mode = "UNDERLINED" # Example: use underline for headers

             # Explicitly handle table conversion if needed *before* html2text,
             # as html2text's table handling might be basic or undesirable.
             # If `tbls` is True, we might call _convert_html_tables_to_markdown first.
             # However, modifying the HTML then passing to html2text can be complex.
             # Let's rely on html2text's built-in table conversion for now if tbls=True.
             # If tbls=False, html2text will ignore them.

             md_text = h.handle(html_txt)
             self.logger.debug("html2text conversion successful.")
             return md_text.strip() # Return stripped markdown

        except Exception as e_html2text:
             self.logger.warning(f"html2text failed ({e_html2text}); attempting fallback with markdownify")

             # Fallback: markdownify (if installed)
             if _markdownify_fallback and callable(_markdownify_fallback):
                 try:
                     # Configure markdownify options
                     md_opts = {
                         "strip": ["script", "style", "meta", "link", "head", "iframe", "form", "button", "input", "select", "textarea", "nav", "aside", "header", "footer"],
                         "convert": ["a", "p", "img", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "blockquote", "code", "pre", "strong", "em", "table", "tr", "td", "th"],
                         "heading_style": "ATX", # Use '#' style headings
                         "bullets": "-",         # Use '-' for bullet lists
                         "strong_em_symbol": "*",# Use asterisks
                         "autolinks": False,      # Don't automatically convert URLs not in <a> tags
                     }
                     # Adjust options based on input parameters
                     if not links: 
                         md_opts['convert'].remove('a')
                     if not imgs:
                         md_opts['convert'].remove('img')
                     if not tbls:
                         md_opts['convert'].remove('table')
                         md_opts['convert'].remove('tr')
                         md_opts['convert'].remove('td')
                         md_opts['convert'].remove('th')

                     md_text = _markdownify_fallback(html_txt, **md_opts)
                     self.logger.debug("Markdownify fallback conversion successful.")
                     return md_text.strip() # Return stripped markdown

                 except Exception as e_markdownify:
                     self.logger.error(f"Markdownify fallback also failed: {e_markdownify}", exc_info=True)
                     # If both fail, raise an error to signal conversion failure
                     raise ToolError("MARKDOWN_CONVERSION_FAILED", details={"reason": "Both html2text and markdownify failed", "html2text_error": str(e_html2text), "markdownify_error": str(e_markdownify)}) from e_html2text

             else:
                 self.logger.error("html2text failed and markdownify fallback is not available.")
                 raise ToolError("MARKDOWN_CONVERSION_FAILED", details={"reason": "html2text failed, markdownify not installed", "error": str(e_html2text)}) from e_html2text


    @tool(name="clean_and_format_text_as_markdown", description="Convert plain text or HTML to clean Markdown, optionally extracting main content")
    @with_tool_metrics
    @with_error_handling
    async def clean_and_format_text_as_markdown(
        self,
        text: str,
        force_markdown_conversion: bool = False,
        extraction_method: str = "auto", # auto, readability, trafilatura, none
        preserve_tables: bool = True,
        preserve_links: bool = True,
        preserve_images: bool = False,
        max_line_length: int = 0, # 0 means no wrapping
    ) -> Dict[str, Any]:
        """Convert plain text or potentially messy HTML into well-formatted Markdown.

        Args:
            text: Input text (can be plain text or HTML).
            force_markdown_conversion: Treat input as HTML even if no obvious HTML tags are detected.
            extraction_method: Method to extract main content from HTML before conversion ('auto', 'readability', 'trafilatura', 'none'). 'auto' tries trafilatura then readability. 'none' cleans the full HTML.
            preserve_tables: Attempt to convert HTML tables to Markdown tables.
            preserve_links: Keep hyperlinks in the Markdown output.
            preserve_images: Keep image references in the Markdown output (often as placeholders or alt text).
            max_line_length: Wrap output lines to this length (0 for no wrapping). Affects readability.

        Returns:
            Dictionary with conversion results:
            - markdown_text: The resulting Markdown string.
            - was_html: Boolean indicating if HTML processing was performed.
            - parser_used: HTML parser backend used (e.g., 'html.parser', 'lxml').
            - extraction_method_used: Content extraction method actually used.
            - input_sha1: SHA1 hash of the original input text.
            - processing_time: Duration of the operation in seconds.
            - success: Boolean indicating success.
        """
        tic = time.perf_counter()
        if not text or not isinstance(text, str):
            raise ToolInputError("Input text must be a non-empty string", param_name="text")

        input_hash = self._hash(text)
        was_html = self._is_html_fragment(text) or force_markdown_conversion
        extractor_used, parser_used = "none", "none"
        md_text = ""

        if was_html:
            self.logger.info(f"Processing input as HTML. Extraction method: {extraction_method}")
            loop = asyncio.get_running_loop()
            # 1. Clean HTML (remove scripts, unsafe attrs, repair basic structure)
            try:
                # Run sync cleaning function in executor
                cleaned_html, parser_used = await loop.run_in_executor(None, self._clean_html, text)
                if parser_used == "failed":
                     # If cleaning failed drastically, attempt conversion on original text
                     self.logger.warning("HTML cleaning failed, attempting conversion on original text.")
                     cleaned_html = text # Use original text as fallback for conversion step
                else:
                     self.logger.debug(f"HTML cleaned using parser: {parser_used}")
            except Exception as e:
                 self.logger.error(f"Error during HTML cleaning: {e}", exc_info=True)
                 # If cleaning fails, proceed with original text for conversion? Or error out?
                 # Let's proceed with original text as a fallback.
                 self.logger.warning("Using original text for conversion due to cleaning error.")
                 cleaned_html = text
                 # raise ToolError("HTML_CLEANING_FAILED", details={"error": str(e)}) from e # Option: raise error instead


            # 2. Extract Main Content (Optional)
            extracted_html = cleaned_html # Default to cleaned HTML if no extraction
            extraction_method = extraction_method.lower()
            if extraction_method != "none":
                if extraction_method == "auto":
                    # Try Trafilatura first (often better for articles)
                    self.logger.debug("Attempting extraction with Trafilatura...")
                    extracted_t = await loop.run_in_executor(None, self._extract_trafilatura, cleaned_html)
                    # Basic check: did extraction yield significant content and not just whitespace?
                    if extracted_t and len(extracted_t.strip()) > len(cleaned_html.strip()) * 0.05 and len(extracted_t.strip()) > 50 :
                         extractor_used = "trafilatura"
                         extracted_html = extracted_t
                         self.logger.info("Extraction successful using Trafilatura.")
                    else:
                         # Fallback to Readability
                         self.logger.debug("Trafilatura yielded little/no content, trying Readability...")
                         extracted_r = await loop.run_in_executor(None, self._extract_readability, cleaned_html)
                         if extracted_r and len(extracted_r.strip()) > len(cleaned_html.strip()) * 0.05 and len(extracted_r.strip()) > 50:
                             extractor_used = "readability"
                             extracted_html = extracted_r
                             self.logger.info("Extraction successful using Readability.")
                         else:
                             self.logger.warning("Both Trafilatura and Readability failed or yielded little content. Using cleaned HTML without extraction.")
                             extractor_used = "none (auto failed)" # Indicate extraction was attempted but failed
                elif extraction_method == "readability":
                    extracted_r = await loop.run_in_executor(None, self._extract_readability, cleaned_html)
                    if extracted_r and extracted_r.strip():
                         extracted_html = extracted_r
                         extractor_used = "readability"
                         self.logger.info("Extraction successful using Readability.")
                    else:
                         self.logger.warning("Readability extraction failed or yielded empty content. Using cleaned HTML.")
                         extractor_used = "none (readability failed)"
                elif extraction_method == "trafilatura":
                    extracted_t = await loop.run_in_executor(None, self._extract_trafilatura, cleaned_html)
                    if extracted_t and extracted_t.strip():
                         extracted_html = extracted_t
                         extractor_used = "trafilatura"
                         self.logger.info("Extraction successful using Trafilatura.")
                    else:
                         self.logger.warning("Trafilatura extraction failed or yielded empty content. Using cleaned HTML.")
                         extractor_used = "none (trafilatura failed)"
                else:
                     self.logger.warning(f"Unknown extraction_method: {extraction_method}. No extraction performed.")
                     extractor_used = "none (unknown method)"
            else:
                 extractor_used = "none (explicitly skipped)"
                 self.logger.info("Extraction explicitly skipped (method='none').")

            # Ensure extracted_html is not empty before conversion
            if not extracted_html.strip():
                 self.logger.warning("HTML content became empty after cleaning/extraction. Result will be empty.")
                 md_text = ""
            else:
                 # 3. Convert (Extracted or Cleaned) HTML to Markdown
                 try:
                      # Call the core conversion logic (no longer cached at instance level)
                      # Run sync conversion function in executor
                      md_text = await loop.run_in_executor(
                          None,
                          self._html_to_md_core,
                          extracted_html,
                          preserve_links,
                          preserve_images,
                          preserve_tables,
                          max_line_length,
                      )
                      self.logger.debug("HTML to Markdown conversion step completed.")
                 except ToolError as e:
                      # Propagate errors from the conversion step
                      raise e
                 except Exception as e:
                      # Catch unexpected errors during conversion call
                      self.logger.error(f"Unexpected error during HTML to Markdown conversion call: {e}", exc_info=True)
                      raise ToolError("MARKDOWN_CONVERSION_FAILED", details={"reason": "Unexpected error in conversion call", "error": str(e)}) from e

        else:
            # Input is treated as plain text or already Markdown
            self.logger.info("Input treated as plain text/Markdown. Applying sanitization and improvements.")
            md_text = text # Start with the original text

        # 4. Sanitize and Improve Markdown (applied whether input was HTML or plain text)
        try:
            if md_text: # Only process if we have some text
                # Run potentially CPU-bound sync functions in executor
                loop = asyncio.get_running_loop()
                sanitized_md = await loop.run_in_executor(None, self._sanitize, md_text)
                improved_md = await loop.run_in_executor(None, self._improve, sanitized_md)

                # Apply line wrapping if specified and wasn't handled by html2text
                # html2text handles width internally if width > 0
                # Check if wrapping is needed *after* improvement steps
                if max_line_length > 0 and not (was_html and max_line_length > 0):
                     self.logger.debug(f"Applying textwrap with max_line_length={max_line_length}")
                     # Split into paragraphs, wrap each, rejoin. Preserve code blocks.
                     wrapped_parts = []
                     # Regex to split by blank lines OR code fences
                     blocks = re.split(r'(```[\s\S]*?```|(?:\n\s*){2,})', improved_md)
                     for block in blocks:
                         if block is None: 
                             continue
                         block_stripped = block.strip()
                         if not block_stripped:
                              # Keep double newlines (or more collapsed to double)
                              if block.count('\n') >= 2:
                                   wrapped_parts.append("\n\n")
                              continue
                         if block_stripped.startswith("```") and block_stripped.endswith("```"):
                             # Preserve code blocks as is
                             wrapped_parts.append(block)
                         elif re.match(r"^(#{1,6}\s|>|[-*+]|\d+\.)", block_stripped):
                              # Don't wrap headings, blockquotes, list items (let them flow naturally)
                              wrapped_parts.append(block)
                         else:
                             # Wrap normal paragraphs
                             wrapped_para = textwrap.fill(
                                 block_stripped,
                                 width=max_line_length,
                                 replace_whitespace=False, # Avoid merging spaces aggressively
                                 drop_whitespace=True,
                                 break_long_words=False, # Avoid breaking long words
                                 break_on_hyphens=True
                             )
                             wrapped_parts.append(wrapped_para)

                     # Join parts, ensuring correct spacing between them
                     final_wrapped_md = ""
                     for i, part in enumerate(wrapped_parts):
                          final_wrapped_md += part
                          # Add double newline after blocks unless it's the last one or next is empty
                          if i < len(wrapped_parts) - 1 and part.strip() and wrapped_parts[i+1].strip():
                               if not final_wrapped_md.endswith("\n\n"):
                                    if final_wrapped_md.endswith("\n"):
                                         final_wrapped_md += "\n"
                                    else:
                                         final_wrapped_md += "\n\n"

                     md_text = final_wrapped_md.strip()

                else:
                    md_text = improved_md # Use the improved version if no wrapping applied

            self.logger.debug("Markdown sanitization and improvement step completed.")
        except Exception as e:
            self.logger.error(f"Error during Markdown sanitization/improvement: {e}", exc_info=True)
            # Continue with potentially unpolished markdown, but log error
            # md_text already holds the result from conversion or original text
            # Optionally wrap this whole step in try/except and return intermediate result on failure.

        toc = time.perf_counter()
        return {
            "markdown_text": md_text,
            "was_html": was_html,
            "parser_used": parser_used,
            "extraction_method_used": extractor_used,
            "input_sha1": input_hash,
            "processing_time": round(toc - tic, 3),
            "success": True,
        }

    @tool(name="detect_content_type", description="Detect if text is primarily HTML, Markdown, code, or plain text")
    @with_tool_metrics
    @with_error_handling
    async def detect_content_type(self, text: str) -> Dict[str, Any]:
        """Detect whether input looks like HTML, Markdown, code, or plain text based on heuristic regex patterns.

        Args:
            text: The text content to analyze.

        Returns:
            Dictionary with detection results:
            - content_type: Detected type ('html', 'markdown', 'code', 'plain_text').
            - confidence: A heuristic confidence score (0.0-1.0).
            - details: Dictionary containing marker scores and detected language (for code).
            - success: Boolean indicating success.
        """
        if not isinstance(text, str): # Allow empty string, detect as plain_text
            raise ToolInputError("Input text must be a string", param_name="text")

        # Handle empty or very short text explicitly
        if len(text.strip()) < 10:
            return {
                "content_type": "plain_text",
                "confidence": 0.9,
                "details": {"reason": "Text too short"},
                "success": True,
             }

        details: Dict[str, Any] = {
            "html_score": 0.0,
            "markdown_score": 0.0,
            "code_score": 0.0,
            "detected_language": None,
        }
        scores = {k: 0.0 for k in ("html", "markdown", "code")}
        # Limit analysis to a portion of the text for performance
        text_sample = text[:10000] # Analyze first 10k chars

        # Calculate weighted marker scores based on sample
        for typ, pats in self.PATTERNS.items():
            if typ not in scores: 
                continue
            type_score = 0.0
            for pat, weight in pats:
                try:
                    # Use findall and weight by frequency and pattern weight
                    n_matches = len(pat.findall(text_sample))
                    if n_matches > 0:
                        # Normalize score by number of matches and weight
                        # Avoid extreme dominance by single pattern
                        type_score += min(n_matches * weight * 0.5, weight * 5) # Capped score per pattern
                except Exception as e:
                    self.logger.warning(f"Regex error during content type detection ({typ}, {pat.pattern}): {e}")
            # Normalize score by sample length? Less reliable, stick to weighted counts.
            scores[typ] = type_score
            details[f"{typ}_score"] = round(type_score, 2) # Store raw score in details

        # Heuristic for plain text: high if other scores are low
        total_marker_score = scores["html"] + scores["markdown"] + scores["code"]
        # Adjust plain_score calculation - make it less dominant initially
        plain_score = max(0, 10 - total_marker_score * 0.5) # Base score decreases as markers increase
        scores["plain_text"] = plain_score

        # Language detection for code (only if code score is reasonably high)
        # Adjust threshold based on observed scores
        code_detection_threshold = 5.0
        if scores["code"] > code_detection_threshold:
            self.logger.debug(f"Code score ({scores['code']:.2f}) > threshold {code_detection_threshold}, attempting language detection.")
            lang_scores: Dict[str, float] = {}
            for pat, lang in self.LANG_PATTERNS:
                 try:
                      n_lang_matches = len(pat.findall(text_sample))
                      if n_lang_matches > 0:
                           # Simple scoring: count matches for each language pattern group
                           lang_scores[lang] = lang_scores.get(lang, 0) + n_lang_matches
                 except Exception as e:
                      self.logger.warning(f"Regex error during language detection ({lang}, {pat.pattern}): {e}")

            if lang_scores:
                 # Select language with the highest score
                 detected_lang = max(lang_scores, key=lang_scores.get)
                 details["detected_language"] = detected_lang
                 details["language_scores"] = lang_scores
                 self.logger.debug(f"Detected language: {detected_lang} (Scores: {lang_scores})")
            else:
                 self.logger.debug("No specific language patterns matched significantly.")


        # Determine final content type and confidence
        if not scores: # Should not happen with plain_text included
             ctype = "plain_text"
             confidence = 0.1
        else:
             # Find type with highest score
             ctype = max(scores, key=scores.get) # type: ignore # scores keys are known strs

             # Calculate confidence (heuristic)
             top_score = scores[ctype]
             total_score = sum(scores.values())
             if total_score > 0:
                 # Confidence is proportion of top score, scaled
                 confidence = (top_score / total_score)
                 # Boost confidence if score disparity is large
                 sorted_scores = sorted(scores.values(), reverse=True)
                 if len(sorted_scores) > 1 and top_score > sorted_scores[1] * 2:
                      confidence = min(1.0, confidence * 1.2) # Increase confidence slightly
                 confidence = max(0.1, min(1.0, confidence)) # Clamp between 0.1 and 1.0
             elif ctype == 'plain_text': # If all scores were 0, confidence in plain is reasonably high
                 confidence = 0.8
             else:
                 confidence = 0.1 # Low confidence if total score is 0


        # If code is detected but no specific language, set type to 'code'
        if ctype == 'code' and not details.get("detected_language"):
             details["detected_language"] = "unknown"


        return {
            "content_type": ctype,
            "confidence": round(confidence, 3),
            "details": details,
            "success": True,
        }


    @tool(name="batch_format_texts", description="Format multiple texts (HTML/plain) to Markdown in parallel")
    @with_tool_metrics
    @with_error_handling
    async def batch_format_texts(
        self,
        texts: List[str],
        force_markdown_conversion: bool = False,
        extraction_method: str = "auto",
        max_concurrency: int = 5,
        preserve_tables: bool = True,
        preserve_links: bool = True,
        preserve_images: bool = False,
    ) -> Dict[str, Any]:
        """Applies 'clean_and_format_text_as_markdown' to a list of texts concurrently.

        Args:
            texts: A list of text strings (HTML or plain text) to format.
            force_markdown_conversion: Treat all inputs as HTML.
            extraction_method: Content extraction method for HTML ('auto', 'readability', 'trafilatura', 'none').
            max_concurrency: Maximum number of texts to process simultaneously.
            preserve_tables: Attempt to convert HTML tables to Markdown.
            preserve_links: Keep hyperlinks in the output.
            preserve_images: Keep image references in the output.


        Returns:
            Dictionary containing:
            - results: List of result dictionaries, one for each input text (including errors).
            - total_processing_time: Total time taken for the batch.
            - success_count: Number of texts successfully processed.
            - failure_count: Number of texts that failed processing.
            - success: Overall success status (always True if the batch operation itself runs).
        """
        if not texts or not isinstance(texts, list):
            raise ToolInputError("Input must be a non-empty list of strings", param_name="texts")
        if not all(isinstance(t, str) for t in texts):
             raise ToolInputError("All items in the 'texts' list must be strings", param_name="texts")

        if max_concurrency <= 0:
             self.logger.warning("max_concurrency must be positive, setting to 1.")
             max_concurrency = 1

        sem = asyncio.Semaphore(max_concurrency)
        tasks = []

        # Define the worker function
        async def _process_one(idx: int, txt: str):
            async with sem:
                self.logger.debug(f"Starting processing for text index {idx}")
                result_dict = {"original_index": idx} # Base result structure
                try:
                    # Call the single text processing method with specified options
                    res = await self.clean_and_format_text_as_markdown(
                        text=txt,
                        force_markdown_conversion=force_markdown_conversion,
                        extraction_method=extraction_method,
                        preserve_tables=preserve_tables,
                        preserve_links=preserve_links,
                        preserve_images=preserve_images,
                        max_line_length=0 # Defaulting to no wrap for batch, could be parameterized
                    )
                    # Merge the result into our tracking dict
                    result_dict.update(res)
                    # Ensure success flag is present and boolean
                    result_dict["success"] = bool(res.get("success", False)) # Default to False if missing or non-bool
                    if result_dict["success"]:
                         self.logger.debug(f"Successfully processed text index {idx}")

                except ToolInputError as e_input:
                     self.logger.warning(f"Input error for text index {idx}: {e_input}")
                     result_dict.update({"error": e_input.message, "success": False, "error_type": "ToolInputError", "error_code": e_input.code})
                except ToolError as e_tool:
                     self.logger.warning(f"Processing error for text index {idx}: {e_tool.code} - {e_tool.message}")
                     result_dict.update({"error": e_tool.message, "success": False, "error_code": e_tool.code, "error_type": "ToolError"})
                except Exception as e:
                     self.logger.error(f"Unexpected error processing text index {idx}: {e}", exc_info=True)
                     result_dict.update({"error": str(e), "success": False, "error_type": "Exception"})

                return result_dict

        tic = time.perf_counter()
        self.logger.info(f"Starting batch formatting for {len(texts)} texts with concurrency {max_concurrency}...")
        # Create tasks using the worker function
        for i, t in enumerate(texts):
             tasks.append(_process_one(i, t))

        # Gather results from all tasks
        all_results = await asyncio.gather(*tasks)
        toc = time.perf_counter()
        self.logger.info(f"Batch formatting completed in {toc - tic:.3f}s")

        # Sort results back into original order using the index we added
        all_results.sort(key=lambda r: r.get("original_index", -1))

        # Clean up the result list (remove temporary index) and count successes/failures
        final_results = []
        success_count = 0
        failure_count = 0
        for r in all_results:
            if r.get("success"): # Check the success flag we ensured exists
                success_count += 1
            else:
                failure_count += 1
            r.pop("original_index", None) # Remove the index key before returning
            final_results.append(r)


        return {
            "results": final_results,
            "total_processing_time": round(toc - tic, 3),
            "success_count": success_count,
            "failure_count": failure_count,
            "success": True, # The batch operation itself succeeded in running
        }

    @tool(name="optimize_markdown_formatting", description="Clean up and standardize existing Markdown text")
    @with_tool_metrics
    @with_error_handling
    async def optimize_markdown_formatting(
        self,
        markdown: str,
        normalize_headings: bool = False, # Adjust heading levels (e.g., make first heading h1)
        fix_lists: bool = True,        # Standardize list markers, ensure spacing
        fix_links: bool = True,        # Remove extra spaces in links like ] (
        add_line_breaks: bool = True,  # Ensure proper spacing around blocks (headings, paragraphs)
        compact_mode: bool = False,    # Reduce excessive blank lines
        max_line_length: int = 0,      # Wrap lines (0 for no wrapping)
    ) -> Dict[str, Any]:
        """Applies various cleaning and formatting rules to an existing Markdown string.

        Args:
            markdown: The Markdown text to optimize.
            normalize_headings: Adjust all heading levels so the top-level heading starts at h1.
            fix_lists: Standardize bullet markers to '-' and ensure correct spacing around list items.
            fix_links: Fix common issues like spaces between ']' and '('.
            add_line_breaks: Ensure appropriate blank lines between block elements (paragraphs, headings, lists).
            compact_mode: Reduce multiple blank lines down to a single blank line.
            max_line_length: Wrap lines to the specified length (0 disables wrapping).

        Returns:
            Dictionary containing:
            - optimized_markdown: The processed Markdown string.
            - changes_made: Dictionary indicating which types of fixes were applied.
            - processing_time: Duration of the operation.
            - success: Boolean indicating success.
        """
        if not markdown or not isinstance(markdown, str):
            raise ToolInputError("Input markdown must be a non-empty string", param_name="markdown")

        tic = time.perf_counter()
        orig_md = markdown # Keep original for comparison if needed
        md = markdown # Work on this copy

        changes: Dict[str, bool] = {
            "headings_normalized": False,
            "lists_fixed": False,
            "links_fixed": False,
            "line_breaks_added": False,
            "whitespace_adjusted": False,
            "line_wrapping_applied": False,
        }

        # --- Apply Fixes (Order can matter) ---

        # 1. Basic whitespace cleanup (run early)
        md_before_ws = md
        md = self._sanitize(md) # Use the existing sanitize function for basic cleanup
        if md != md_before_ws:
            changes["whitespace_adjusted"] = True

        # 2. Normalize Headings
        if normalize_headings:
            md_before_norm = md
            # Find the minimum heading level present
            levels = [len(m.group(1)) for m in re.finditer(r"^(#{1,6})\s", md, flags=re.MULTILINE)]
            if levels:
                min_level = min(levels)
                delta = min_level - 1 # Amount to shift up
                if delta > 0:
                    self.logger.debug(f"Normalizing headings: shifting by -{delta} levels.")
                    # Define replacement function to adjust hash count
                    def replace_heading(match):
                         hashes = match.group(1)
                         new_level = max(1, len(hashes) - delta) # Ensure level is at least 1
                         return f"{'#' * new_level} " # Ensure space is added back

                    md = re.sub(r"^(#{1,6})\s", replace_heading, md, flags=re.MULTILINE)
                    if md != md_before_norm: 
                        changes["headings_normalized"] = True

        # 3. Fix Links
        if fix_links:
            md_before_link = md
            # Remove space between ] and ( for inline links: ] ( -> ](
            md = re.sub(r"\][ \t]+\(", "](", md)
            # Remove space between ] and [ for reference links: ] [ -> ][
            md = re.sub(r"\][ \t]+\[", "][", md)
            # Could add more link fixes (e.g., URL encoding checks) if needed
            if md != md_before_link: 
                changes["links_fixed"] = True

        # 4. Add Line Breaks (apply _improve logic here) and Fix Lists
        # Combine list fixing and line break improvements as they are related to block spacing
        md_before_breaks = md
        # Standardize list markers first (part of _sanitize, but ensure consistency)
        md = re.sub(r"^[ \t]*[*+]\s", "- ", md, flags=re.MULTILINE) # Ensure '-' bullet
        md = re.sub(r"^[ \t]*(\d+\.)\s", r"\1 ", md, flags=re.MULTILINE) # Ensure space after number.

        if add_line_breaks or fix_lists:
             # Use the _improve logic for adding blank lines around blocks
             md = self._improve(md) # This handles headings, code, quotes, lists spacing
             changes["line_breaks_added"] = True # Assume _improve potentially adds breaks

        if fix_lists:
             # Additional list-specific fixes if needed (e.g., indentation) could go here
             # Currently, marker and spacing are handled by _sanitize and _improve
             pass # Placeholder if more list fixes needed

        if md != md_before_breaks:
             # Mark changes based on which feature was enabled
             if fix_lists: 
                 changes["lists_fixed"] = True
             # changes["line_breaks_added"] is already set if either was true
             changes["whitespace_adjusted"] = True # Spacing changes count as whitespace adjustment


        # 5. Compact Mode (Reduce excessive blank lines)
        if compact_mode:
            md_before_compact = md
            md = re.sub(r"\n{3,}", "\n\n", md) # Collapse 3+ newlines to exactly 2
            if md != md_before_compact: 
                changes["whitespace_adjusted"] = True

        # 6. Line Wrapping (Apply last)
        if max_line_length > 0:
            md_before_wrap = md
            self.logger.debug(f"Applying textwrap with max_line_length={max_line_length}")
            # Use the wrapping logic from clean_and_format_text_as_markdown's step 4
            wrapped_parts = []
            blocks = re.split(r'(```[\s\S]*?```|(?:\n\s*){2,})', md)
            for block in blocks:
                if block is None: 
                    continue
                block_stripped = block.strip()
                if not block_stripped:
                    if block.count('\n') >= 2: 
                        wrapped_parts.append("\n\n")
                    continue
                # Check if block should be preserved (code, heading, quote, list)
                if block_stripped.startswith("```") or re.match(r"^(#{1,6}\s|>|[-*+]\s|\d+\.\s)", block_stripped):
                    wrapped_parts.append(block) # Preserve formatting
                else:
                    # Wrap normal paragraphs
                    wrapped_para = textwrap.fill(
                        block, # Wrap the original block including leading/trailing spaces if any
                        width=max_line_length,
                        replace_whitespace=False, drop_whitespace=False, # Be less aggressive with whitespace
                        break_long_words=False, break_on_hyphens=True
                    )
                    wrapped_parts.append(wrapped_para)

            # Reconstruct the document
            md = ""
            for i, part in enumerate(wrapped_parts):
                md += part
                # Ensure appropriate spacing between reconstructed blocks
                if i < len(wrapped_parts) - 1:
                     if part.strip() and wrapped_parts[i+1].strip():
                          # Add double newline if not already ending with it
                          if not md.endswith("\n\n"):
                               md = md.rstrip() + "\n\n"
                     elif not md.endswith("\n"): # Ensure at least one newline if followed by blank lines
                          md += "\n"


            md = md.strip() # Final trim
            if md != md_before_wrap:
                 changes["line_wrapping_applied"] = True
                 changes["whitespace_adjusted"] = True


        # Final check for any whitespace changes not caught
        if not changes["whitespace_adjusted"] and md != orig_md:
             # Check if only whitespace differs (e.g., internal spacing)
             if md.split() != orig_md.split():
                  changes["whitespace_adjusted"] = True # Mark if non-whitespace content changed subtly
             elif md != orig_md:
                  changes["whitespace_adjusted"] = True # Mark if only whitespace distribution changed

        toc = time.perf_counter()
        return {
            "optimized_markdown": md,
            "changes_made": {k: v for k, v in changes.items() if v}, # Only report changes that occurred
            "processing_time": round(toc - tic, 3),
            "success": True,
        }


    ###############################################################################
    # Table Extraction Tool                                                       #
    ###############################################################################
    @tool(name="extract_tables", description="Extract tables from a document (PDF/Office) into CSV, JSON, or Pandas DataFrame format")
    @with_tool_metrics
    @with_error_handling
    async def extract_tables(self, document_path: str, *, table_mode: str = "csv", output_dir: Optional[str] = None,
                             accelerator_device: str = "auto", num_threads: int = 4) -> Dict[str, Any]:
        """Extracts tables found in a document and returns them in the specified format.

        Args:
            document_path: Path to the document (e.g., PDF, DOCX) containing tables.
            table_mode: Format for output tables ('csv', 'json', 'pandas').
                'csv': Each table is a string in CSV format.
                'json': Each table is a list of lists (rows of cells).
                'pandas': Each table is a pandas DataFrame (requires pandas installed).
            output_dir: If specified, saves each extracted table to a file in this directory.
            accelerator_device: Device for document conversion backend ('auto', 'cpu', 'cuda', 'mps').
            num_threads: Number of threads for document conversion backend.

        Returns:
            Dictionary containing:
            - tables: A list where each element is a table in the format specified by 'table_mode'.
            - saved_files: A list of paths to saved files (if output_dir was provided).
            - success: Boolean indicating success.
        """
        valid_modes = {"csv", "json", "pandas"}
        table_mode = table_mode.lower()
        if table_mode not in valid_modes:
            raise ToolInputError(f"table_mode must be one of {', '.join(valid_modes)}", param_name="table_mode", provided_value=table_mode)

        if table_mode == "pandas" and pd is None:
            raise ToolError("DEPENDENCY_MISSING", details={"dependency": "pandas", "feature": "extract_tables(mode='pandas')"})

        self.logger.info(f"Starting table extraction from {document_path}, mode='{table_mode}'")

        # --- Step 1: Convert document using Docling to get structured data ---
        # We need the Doc object, not just exported JSON, to access tables directly.
        doc_obj: Optional[DoclingDocument] = None
        try:
            doc_path = Path(document_path)
            if not doc_path.is_file():
                 raise ToolInputError(f"Document not found at path: {document_path}", param_name="document_path")

            device = self._ACCEL_MAP[accelerator_device.lower()]
            conv = self._converter(device, num_threads)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, conv.convert, doc_path)

            if result and result.document:
                 doc_obj = result.document
                 self.logger.info("Document converted successfully for table extraction.")
            else:
                 raise ToolError("CONVERSION_FAILED", details={"document_path": document_path, "reason": "Converter returned empty result"})

        except ToolError as te:
             raise te # Re-raise specific conversion errors
        except Exception as e:
             self.logger.error(f"Error during document conversion for table extraction: {e}", exc_info=True)
             raise ToolError("CONVERSION_FAILED", details={"document_path": document_path, "error": str(e)}) from e

        if not doc_obj: # Should be caught above, but safety check
             return {"tables": [], "saved_files": [], "success": False, "error": "Document object unavailable after conversion."}

        # --- Step 2: Extract tables from the Doc object ---
        # Docling's Doc object contains tables within PageContent
        tables_raw_data: List[List[List[str]]] = [] # Expecting list of tables, each table is list of rows, each row is list of strings
        try:
            for page in doc_obj.pages:
                if page.content and page.content.has_tables():
                    # Assuming get_tables() returns list of tables, where each table is list[list[str]]
                    page_tables = page.content.get_tables()
                    if page_tables:
                        # Basic validation of table structure
                        for tbl in page_tables:
                             if isinstance(tbl, list) and all(isinstance(row, list) for row in tbl):
                                  # Ensure cells are strings
                                  sanitized_tbl = [[str(cell) if cell is not None else "" for cell in row] for row in tbl]
                                  tables_raw_data.append(sanitized_tbl)
                             else:
                                  self.logger.warning(f"Skipping malformed table structure found on page {page.page_idx}: {type(tbl)}")

        except Exception as e:
             self.logger.error(f"Error accessing tables from Doc object: {e}", exc_info=True)
             # Continue if possible, might have partial extraction
             # Or raise ToolError("TABLE_EXTRACTION_FAILED", details={"error": str(e)}) from e

        if not tables_raw_data:
            self.logger.warning(f"No tables found or extracted from {document_path}")
            return {"tables": [], "saved_files": [], "success": True}

        self.logger.info(f"Extracted {len(tables_raw_data)} tables from document.")

        # --- Step 3: Format tables and optionally save ---
        output_tables: List[Any] = []
        saved_files: List[str] = []
        output_dir_path = Path(output_dir) if output_dir else None
        if output_dir_path:
             output_dir_path.mkdir(parents=True, exist_ok=True)

        for i, raw_table in enumerate(tables_raw_data):
            processed_table: Any = None
            file_ext = ""
            save_content: Union[str, Any] = "" # Content to be saved

            try:
                if table_mode == "csv":
                    # Use csv module for proper CSV formatting
                    output = StringIO()
                    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
                    writer.writerows(raw_table)
                    processed_table = output.getvalue()
                    file_ext = "csv"
                    save_content = processed_table

                elif table_mode == "json":
                    processed_table = raw_table # Already in list-of-lists format
                    file_ext = "json"
                    save_content = self._json(processed_table) # Convert to JSON string for saving

                elif table_mode == "pandas":
                     df = pd.DataFrame(raw_table)
                     # Simple heuristic: If first row looks like a header (more non-numeric?), use it.
                     # This is very basic. A more robust check might be needed.
                     if not df.empty and len(df) > 1:
                         first_row = df.iloc[0]
                         # Count non-numeric cells in first row vs second row
                         is_header = sum(1 for cell in first_row if not str(cell).replace('.','',1).isdigit()) > len(first_row)/2
                         if is_header:
                             df.columns = first_row # Set first row as header
                             df = df[1:].reset_index(drop=True) # Remove header row from data
                     processed_table = df
                     file_ext = "csv" # Pandas DataFrames usually saved as CSV
                     save_content = processed_table # Save the DataFrame directly

                output_tables.append(processed_table)

                # Save table to file if output_dir is set
                if output_dir_path and file_ext:
                     base_name = Path(document_path).stem
                     fp = output_dir_path / f"{base_name}_table_{i + 1}.{file_ext}"
                     try:
                         if isinstance(save_content, str):
                             fp.write_text(save_content, encoding="utf-8")
                         elif isinstance(save_content, pd.DataFrame):
                             save_content.to_csv(fp, index=False, encoding="utf-8")
                         saved_files.append(str(fp))
                         self.logger.debug(f"Saved table {i+1} to {fp}")
                     except Exception as e_save:
                         self.logger.error(f"Failed to save table {i+1} to {fp}: {e_save}", exc_info=True)
                         # Log error but continue processing other tables

            except Exception as e_format:
                 self.logger.error(f"Failed to format table {i} into '{table_mode}': {e_format}", exc_info=True)
                 continue # Skip this table

        self.logger.info(f"Successfully processed {len(output_tables)} tables into '{table_mode}' format.")
        return {"tables": output_tables, "saved_files": saved_files, "success": True}


    ###############################################################################
    # Document Analysis Tools                                                     #
    ###############################################################################

    @tool(name="identify_sections", description="Identify logical sections based on headings and structure")
    @with_tool_metrics
    @with_error_handling
    async def identify_sections(self, document: str) -> Dict[str, Any]:
        """Identifies logical sections in a document using regex patterns for headings.

        Args:
            document: The text content of the document.

        Returns:
            Dictionary containing:
            - sections: A list of identified sections, each a dict with:
                - title: The identified section title (or default).
                - text: The text content of the section (excluding title line).
                - position: The zero-based index of the section.
                - start_char: Start character index of the title line in the original document.
                - end_char: End character index of the section in the original document.
            - success: Boolean indicating success.
        """
        if not document or not isinstance(document, str):
            self.logger.warning("identify_sections called with empty or invalid input.")
            return {"sections": [], "success": True} # Return success with empty list


        sections_found: List[Dict[str, Any]] = []
        last_section_end = 0

        # Ensure _BOUND_RX is compiled
        if not hasattr(self, '_BOUND_RX') or not isinstance(self._BOUND_RX, re.Pattern):
             raise ToolError("INITIALIZATION_ERROR", details={"reason": "_BOUND_RX regex not compiled or invalid"})

        # Find all potential section boundaries (titles matching the regex)
        matches = list(self._BOUND_RX.finditer(document))

        # If no boundaries found, treat the whole document as one section
        if not matches:
            self.logger.info("No regex-based section boundaries found. Treating document as single section.")
            if document.strip(): # Only add if document is not just whitespace
                sections_found.append({
                    "title": "Main Content",
                    "text": document.strip(),
                    "position": 0,
                    "start_char": 0,
                    "end_char": len(document)
                })
        else:
            self.logger.info(f"Found {len(matches)} potential section boundaries based on regex.")
             # Add content before the first match as an initial section (e.g., Introduction)
            first_match_start = matches[0].start()
            if first_match_start > 0:
                initial_text = document[last_section_end:first_match_start].strip()
                if initial_text:
                     sections_found.append({
                         "title": "Introduction", # Generic title for content before first heading
                         "text": initial_text,
                         "position": 0,
                         "start_char": last_section_end,
                         "end_char": first_match_start
                     })
                     last_section_end = first_match_start


            # Process each match to create a section
            for i, match in enumerate(matches):
                title_raw = match.group(0).strip() # The matched heading line
                title_start_char = match.start()
                title_end_char = match.end()
                section_content_start = title_end_char

                # Determine the end of this section's content
                section_content_end = matches[i + 1].start() if i < len(matches) - 1 else len(document)

                # Extract section text (content *after* the title line up to the next title or end of doc)
                section_text = document[section_content_start:section_content_end].strip()

                # Determine the final section title (apply custom regex overrides)
                section_title = title_raw # Default to the matched heading text
                # Ensure _CUSTOM_SECT_RX is initialized and is a list
                if hasattr(self, '_CUSTOM_SECT_RX') and isinstance(self._CUSTOM_SECT_RX, list):
                     for pat, label in self._CUSTOM_SECT_RX:
                         if isinstance(pat, re.Pattern) and pat.search(title_raw):
                             section_title = label
                             self.logger.debug(f"Applied custom label '{label}' to section title '{title_raw}'.")
                             break # Use first matching custom label
                else:
                     self.logger.warning("_CUSTOM_SECT_RX not initialized correctly, skipping custom labels.")


                # Add the identified section only if it has content
                if section_text:
                     sections_found.append({
                         "title": section_title,
                         "text": section_text,
                         "position": len(sections_found), # Position based on order found
                         "start_char": title_start_char, # Start of the title line
                         "end_char": section_content_end # End of the section's content
                     })
                else:
                     self.logger.debug(f"Skipping section '{section_title}' because it has no content after the title.")

                last_section_end = section_content_end # Update position for next potential intro section (though unlikely needed here)


        return {"sections": sections_found, "success": True}


    @tool(name="extract_entities", description="Extract named entities (PERSON, ORG, etc.) using an LLM")
    @with_tool_metrics
    @with_error_handling
    async def extract_entities(self, document: str, entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extracts named entities from document text using an LLM prompt.

        Args:
            document: Text content to analyze.
            entity_types: Optional list of specific entity types to focus on (e.g., ["PERSON", "ORG"]).
                          If None, instructs LLM to extract common types.

        Returns:
            Dictionary containing:
            - entities: Dictionary where keys are entity types (uppercase)
                        and values are lists of unique entity strings found.
            - success: Boolean indicating success.
            - error: Error message if parsing or LLM call fails.
            - raw_llm_response: The raw text response from the LLM (for debugging).
        """
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")

        # Prepare context and prompt for LLM
        max_context = 3500 # Limit context sent to LLM to manage cost/latency
        context = document[:max_context]
        if len(document) > max_context:
            context += "\n..." # Indicate truncation
            self.logger.warning(f"Document truncated to ~{max_context} chars for entity extraction.")

        if entity_types and isinstance(entity_types, list):
             entity_focus = f"Extract only these entity types: {', '.join(entity_types)}."
        else:
             entity_focus = "Extract common named entity types such as PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, EVENT."

        prompt = f"""Please analyze the following text and extract all named entities.
{entity_focus}
Format the output ONLY as a valid JSON object. The keys should be the uppercase entity types (e.g., "PERSON", "ORGANIZATION"). The values should be lists of the unique entity strings found for each type.
Do not include any introductory text, explanations, apologies, or markdown formatting like ```json.

Text:
\"\"\"
{context}
\"\"\"

JSON Output:
"""
        self.logger.info(f"Requesting entity extraction from LLM. Entity focus: {entity_types or 'common'}")
        llm_response_raw = ""
        try:
            # Use lower temperature for more factual, structured output
            llm_response_raw = await self._llm(prompt=prompt, max_tokens=1500, temperature=0.1)
            self.logger.debug(f"LLM response received for entity extraction:\n{llm_response_raw}")

            # Attempt to parse the LLM response as JSON
            try:
                # LLMs sometimes add markdown fences, try to strip them
                json_match = re.search(r'```(?:json)?\s*([\s\S]+)\s*```', llm_response_raw)
                if json_match:
                     json_str = json_match.group(1).strip()
                     self.logger.debug("Stripped markdown fences from LLM response.")
                else:
                     # Assume raw response is JSON, trim whitespace just in case
                     json_str = llm_response_raw.strip()

                # Handle potential leading/trailing text before/after JSON object
                # Find the first '{' and last '}'
                start_brace = json_str.find('{')
                end_brace = json_str.rfind('}')
                if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
                     json_str = json_str[start_brace : end_brace + 1]
                else:
                     # If no braces found, parsing will likely fail below
                     self.logger.warning("Could not find JSON object boundaries ({...}) in LLM response.")


                entities_dict = json.loads(json_str)

                # --- Validate and sanitize the parsed structure ---
                if not isinstance(entities_dict, dict):
                    raise ValueError("LLM response parsed, but is not a JSON object (dictionary).")

                validated_entities: Dict[str, List[str]] = {}
                for key, value in entities_dict.items():
                    entity_type = str(key).upper().strip() # Normalize key
                    if not entity_type: 
                        continue # Skip empty keys

                    sanitized_values: List[str] = []
                    if isinstance(value, list):
                         for item in value:
                              if isinstance(item, str) and item.strip():
                                   sanitized_values.append(item.strip())
                              # Optionally handle dicts within list if LLM returns {"text": "..."}
                              elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                                   sanitized_values.append(item["text"].strip())

                    elif isinstance(value, str) and value.strip(): # Handle case where LLM returns a single string instead of list
                         sanitized_values.append(value.strip())

                    # Only add if list is not empty, store unique values
                    if sanitized_values:
                        validated_entities[entity_type] = sorted(list(set(sanitized_values)))

                self.logger.info(f"Successfully extracted and parsed entities for types: {list(validated_entities.keys())}")
                return {"entities": validated_entities, "success": True, "raw_llm_response": llm_response_raw}

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse LLM response for entities as valid JSON: {e}", exc_info=False) # Less noise in logs
                # Log the problematic response at debug level
                self.logger.debug(f"Raw LLM response causing parse error:\n{llm_response_raw}")
                # Return failure but include raw response for potential manual inspection
                return {"entities": {}, "error": f"Failed to parse LLM response: {e}", "raw_llm_response": llm_response_raw, "success": False}

        except ToolError as e: # Catch errors from _llm call itself
             self.logger.error(f"LLM call failed during entity extraction: {e}", exc_info=True)
             return {"entities": {}, "error": f"LLM call failed: {e.message}", "raw_llm_response": llm_response_raw, "success": False}
        except Exception as e:
             self.logger.error(f"Unexpected error during entity extraction: {e}", exc_info=True)
             return {"entities": {}, "error": f"Unexpected error: {e}", "raw_llm_response": llm_response_raw, "success": False}


    @tool(name="generate_qa_pairs", description="Generate question-answer pairs from the document using an LLM")
    @with_tool_metrics
    @with_error_handling
    async def generate_qa_pairs(self, document: str, num_questions: int = 5) -> Dict[str, Any]:
        """Generates question-answer pairs based on the document content using an LLM.

        Args:
            document: Source text content.
            num_questions: The desired number of QA pairs to generate.

        Returns:
            Dictionary containing:
            - qa_pairs: A list of dictionaries, each with 'question' and 'answer' keys.
            - success: Boolean indicating success.
            - error: Error message if generation or parsing fails.
            - raw_llm_response: The raw text response from the LLM (for debugging).
        """
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")
        if not isinstance(num_questions, int) or num_questions <= 0:
             raise ToolInputError("num_questions must be a positive integer", param_name="num_questions", provided_value=num_questions)


        max_context = 3800 # Limit context for QA generation
        context = document[:max_context]
        if len(document) > max_context:
             context += "\n..." # Indicate truncation
             self.logger.warning(f"Document truncated to ~{max_context} chars for QA generation.")

        prompt = f"""Based *only* on the information in the following text, generate exactly {num_questions} relevant and insightful question-answer pairs.
The questions should be answerable directly from the provided text. The answers should be factual and concise, quoting or summarizing information found in the text.
Format the output ONLY as a valid JSON list of objects. Each object in the list must have exactly two keys: "question" (string) and "answer" (string).
Do not include any introductory text, explanations, apologies, or markdown formatting like ```json.

Text:
\"\"\"
{context}
\"\"\"

JSON Output:
"""
        self.logger.info(f"Requesting {num_questions} QA pairs from LLM.")
        llm_response_raw = ""
        try:
            # Estimate token needs: ~100-150 tokens per QA pair
            llm_max_tokens = num_questions * 150
            llm_response_raw = await self._llm(prompt=prompt, max_tokens=llm_max_tokens, temperature=0.4) # Slightly higher temp for generation
            self.logger.debug(f"LLM response received for QA pairs:\n{llm_response_raw}")

            # Attempt to parse the LLM response as JSON list
            try:
                # Strip potential markdown code block fences
                json_match = re.search(r'```(?:json)?\s*([\s\S]+)\s*```', llm_response_raw)
                if json_match:
                     json_str = json_match.group(1).strip()
                     self.logger.debug("Stripped markdown fences from LLM response.")
                else:
                     json_str = llm_response_raw.strip()

                # Handle potential leading/trailing text before/after JSON list
                start_bracket = json_str.find('[')
                end_bracket = json_str.rfind(']')
                if start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket:
                     json_str = json_str[start_bracket : end_bracket + 1]
                else:
                     self.logger.warning("Could not find JSON list boundaries ([...]) in LLM response.")


                qa_list = json.loads(json_str)

                # Validate structure: list of dicts with 'question' and 'answer'
                if not isinstance(qa_list, list):
                     raise ValueError("LLM response parsed, but is not a JSON list.")

                validated_pairs: List[Dict[str, str]] = []
                for item in qa_list:
                    if isinstance(item, dict):
                         q = item.get("question")
                         a = item.get("answer")
                         # Ensure both are non-empty strings
                         if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
                              validated_pairs.append({"question": q.strip(), "answer": a.strip()})
                         else:
                              self.logger.warning(f"Skipping invalid QA pair item: {item}")
                    else:
                         self.logger.warning(f"Skipping non-dictionary item in QA list: {item}")

                if not validated_pairs:
                     # If validation removed all pairs, consider it a parsing failure in practice
                     raise ValueError("Parsed JSON list, but it contained no valid QA pairs.")

                self.logger.info(f"Successfully generated and parsed {len(validated_pairs)} valid QA pairs.")
                # Optionally trim/pad to num_questions? Let's return what was validly generated.
                # qa_pairs_final = validated_pairs[:num_questions]

                return {"qa_pairs": validated_pairs, "success": True, "raw_llm_response": llm_response_raw}

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse LLM response for QA pairs as valid JSON list: {e}", exc_info=False)
                self.logger.debug(f"Raw LLM response causing QA parse error:\n{llm_response_raw}")

                # Try simple regex fallback as a last resort (less reliable)
                pairs = []
                try:
                    extracted = re.findall(r'\{\s*"question":\s*"(.*?)",\s*"answer":\s*"(.*?)"\s*\}', llm_response_raw, re.DOTALL | re.IGNORECASE)
                    if extracted:
                        pairs = [{"question": q.strip().replace('\\"', '"'), "answer": a.strip().replace('\\"', '"')}
                                 for q, a in extracted if q.strip() and a.strip()]
                        if pairs:
                            self.logger.warning(f"JSON parsing failed, extracted {len(pairs)} pairs using regex fallback.")
                            return {"qa_pairs": pairs[:num_questions], "success": True, "warning": "Used regex fallback for parsing.", "raw_llm_response": llm_response_raw}
                except Exception as regex_e:
                    self.logger.error(f"Regex fallback for QA pairs also failed: {regex_e}")

                # If both JSON and regex fail
                return {"qa_pairs": [], "error": f"Failed to parse LLM response: {e}", "raw_llm_response": llm_response_raw, "success": False}

        except ToolError as e: # Catch errors from _llm call
             self.logger.error(f"LLM call failed during QA generation: {e}", exc_info=True)
             return {"qa_pairs": [], "error": f"LLM call failed: {e.message}", "raw_llm_response": llm_response_raw, "success": False}
        except Exception as e:
             self.logger.error(f"Unexpected error during QA generation: {e}", exc_info=True)
             return {"qa_pairs": [], "error": f"Unexpected error: {e}", "raw_llm_response": llm_response_raw, "success": False}


    @tool(name="summarize_document", description="Generate a concise summary of the document using an LLM")
    @with_tool_metrics
    @with_error_handling
    async def summarize_document(self, document: str, max_length: int = 150, focus: Optional[str] = None) -> Dict[str, Any]:
        """Generates a summary of the document text using an LLM.

        Args:
            document: Text content to summarize.
            max_length: Target maximum length of the summary in words (approximate).
            focus: Optional topic or aspect to focus the summary on.

        Returns:
            Dictionary containing:
            - summary: The generated summary string.
            - word_count: Approximate word count of the summary.
            - success: Boolean indicating success.
            - error: Error message if generation fails.
            - raw_llm_response: The raw text response from the LLM.
        """
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")
        if not isinstance(max_length, int) or max_length <= 10:
             raise ToolInputError("max_length must be a positive integer greater than 10", param_name="max_length", provided_value=max_length)


        max_context = 4000 # Context limit for summarization
        context = document[:max_context]
        if len(document) > max_context:
             context += "\n..." # Indicate truncation
             self.logger.warning(f"Document truncated to ~{max_context} chars for summarization.")

        focus_instruction = f" Please pay special attention to aspects related to: {focus}." if focus and isinstance(focus, str) and focus.strip() else ""

        prompt = f"""Please generate a concise and coherent summary of the following text.
The summary should be approximately {max_length} words long.{focus_instruction}
Capture the main points and key information accurately based *only* on the provided text. Do not add opinions or information not present in the text.
Output only the summary text itself, without any introductory phrases like "Here is the summary:", "This document discusses:", etc.

Text:
\"\"\"
{context}
\"\"\"

Summary:
"""
        self.logger.info(f"Requesting summary from LLM (max_length≈{max_length}, focus='{focus or 'none'}').")
        llm_response_raw = ""
        try:
            # Calculate appropriate max_tokens for LLM (words to tokens ratio ~0.75)
            # Add some buffer (e.g., 50%)
            llm_max_tokens = int(max_length / 0.65)

            summary_text = await self._llm(prompt=prompt, max_tokens=llm_max_tokens, temperature=0.5) # Temp 0.5 allows some fluency
            llm_response_raw = summary_text # Keep raw response before cleaning

            # Basic post-processing: remove potential leading boilerplate
            summary_text = re.sub(r"^(Here is a summary:|Summary:|The text discusses|This document is about)\s*:?\s*", "", summary_text, flags=re.IGNORECASE).strip()

            word_count = len(summary_text.split())
            self.logger.info(f"Generated summary with {word_count} words (target: {max_length}).")

            # Optional: Check if summary is too short/long? LLMs might ignore length constraint.
            if word_count < max_length * 0.5:
                 self.logger.warning(f"Generated summary is much shorter ({word_count} words) than requested ({max_length}).")
            elif word_count > max_length * 1.5:
                 self.logger.warning(f"Generated summary is much longer ({word_count} words) than requested ({max_length}).")


            return {
                "summary": summary_text,
                "word_count": word_count,
                "success": True,
                "raw_llm_response": llm_response_raw
            }

        except ToolError as e: # Catch errors from _llm call
             self.logger.error(f"LLM call failed during summarization: {e}", exc_info=True)
             return {"summary": "", "word_count": 0, "error": f"LLM call failed: {e.message}", "success": False, "raw_llm_response": llm_response_raw}
        except Exception as e:
             self.logger.error(f"Unexpected error during summarization: {e}", exc_info=True)
             return {"summary": "", "word_count": 0, "error": f"Unexpected error: {e}", "success": False, "raw_llm_response": llm_response_raw}


    @tool(name="classify_document", description="Classify the document into predefined or custom categories using an LLM")
    @with_tool_metrics
    @with_error_handling
    async def classify_document(self, document: str, custom_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classifies the document based on its content using domain-specific or custom labels via an LLM.

        Args:
            document: Text content to classify.
            custom_labels: Optional list of specific classification labels to use instead of the domain default.

        Returns:
            Dictionary containing:
            - classification: The label chosen by the LLM (or the closest match from the provided list).
            - confidence: A heuristic confidence score (0.0-1.0) based on fuzzy matching if LLM output needs correction.
            - raw_llm_output: The direct output from the LLM.
            - success: Boolean indicating success.
            - error: Error message if classification fails.
        """
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")

        # Determine labels and prompt prefix
        labels_to_use = custom_labels if isinstance(custom_labels, list) and custom_labels else self._DOC_LABELS
        prompt_prefix = self._CLASS_PROMPT_PREFIX if not custom_labels else "Classify the document using exactly one of the following labels: "

        if not labels_to_use:
             error_msg = f"No classification labels available for domain '{self.ACTIVE_DOMAIN}' and no custom labels provided."
             self.logger.error(error_msg)
             return {"classification": None, "confidence": 0.0, "raw_llm_output": "", "error": error_msg, "success": False}

        label_string = ', '.join(f'"{label}"' for label in labels_to_use) # Quote labels for clarity

        max_context = 3000 # Context limit for classification
        context = document[:max_context]
        if len(document) > max_context:
             context += "\n..." # Indicate truncation
             self.logger.warning(f"Document truncated to ~{max_context} chars for classification.")


        prompt = f"""{prompt_prefix}{label_string}.
Analyze the following document text and determine the single best classification label from the provided list.
Return ONLY the chosen label as a string, exactly as it appears in the list. Do not include any other text, explanation, or formatting.

Document Text:
\"\"\"
{context}
\"\"\"

Classification Label:
"""
        self.logger.info(f"Requesting classification from LLM using labels: {labels_to_use}")
        llm_response_raw = ""
        try:
            # Use very low temperature for classification task
            llm_response_raw = await self._llm(prompt=prompt, max_tokens=50, temperature=0.05)
            raw_output = llm_response_raw.strip().strip('"') # Remove surrounding quotes if any
            self.logger.debug(f"LLM raw classification response: '{raw_output}'")

            # --- Post-processing LLM Output ---
            best_match = None
            best_score = 0.0

            # 1. Check for exact match (case-insensitive comparison)
            for label in labels_to_use:
                 if raw_output.lower() == label.lower():
                     best_match = label # Use the canonical capitalization from labels_to_use
                     best_score = 1.0
                     self.logger.info(f"Exact match found: '{best_match}'")
                     break

            # 2. If no exact match, use fuzzy matching to find the closest label
            if best_match is None:
                 self.logger.debug(f"No exact match for '{raw_output}', performing fuzzy matching...")
                 fuzzy_scores = {}
                 for label in labels_to_use:
                     # Using fuzz.ratio - simple sequence similarity
                     score = fuzz.ratio(raw_output.lower(), label.lower()) / 100.0 # Normalize to 0-1
                     fuzzy_scores[label] = score
                     self.logger.debug(f"  - Score vs '{label}': {score:.2f}")

                 if fuzzy_scores:
                      # Find the label with the highest fuzzy score
                      potential_match = max(fuzzy_scores, key=fuzzy_scores.get)
                      best_score = fuzzy_scores[potential_match]

                      # Set a threshold for accepting the fuzzy match
                      fuzzy_threshold = 0.75 # Requires 75% similarity
                      if best_score >= fuzzy_threshold:
                           best_match = potential_match
                           self.logger.info(f"Fuzzy match selected: '{best_match}' (Score: {best_score:.2f})")
                      else:
                           self.logger.warning(f"Best fuzzy match '{potential_match}' has low score ({best_score:.2f} < {fuzzy_threshold}). Classification is uncertain. Returning raw LLM output.")
                           # If score is too low, classification is unreliable.
                           # Return the raw output from the LLM as the classification, with low confidence.
                           best_match = raw_output # Use the raw LLM output string
                           # Confidence remains the low best_score
                 else:
                     # Should not happen if labels_to_use is not empty, but handle defensively
                     self.logger.warning("Fuzzy matching yielded no scores. Returning raw LLM output.")
                     best_match = raw_output
                     best_score = 0.1 # Assign minimal confidence


            # If best_match is still None here (edge case), use raw output
            if best_match is None:
                 best_match = raw_output
                 best_score = 0.1 # Low confidence if no match logic worked

            return {
                "classification": best_match,
                "confidence": round(best_score, 3),
                "raw_llm_output": llm_response_raw, # Return the original raw response
                "success": True
            }

        except ToolError as e: # Catch errors from _llm call
             self.logger.error(f"LLM call failed during classification: {e}", exc_info=True)
             return {"classification": None, "confidence": 0.0, "raw_llm_output": llm_response_raw, "error": f"LLM call failed: {e.message}", "success": False}
        except Exception as e:
             self.logger.error(f"Unexpected error during classification: {e}", exc_info=True)
             return {"classification": None, "confidence": 0.0, "raw_llm_output": llm_response_raw, "error": f"Unexpected error: {e}", "success": False}


    @tool(name="extract_metrics", description="Extract numeric metrics based on domain-specific keywords and patterns")
    @with_tool_metrics
    @with_error_handling
    async def extract_metrics(self, document: str) -> Dict[str, Any]:
        """Extracts numeric metrics (e.g., revenue, EBITDA) from the document using regex patterns defined for the active domain.

        Args:
            document: Text content to analyze.

        Returns:
            Dictionary containing:
            - metrics: Dictionary where keys are metric names (e.g., 'revenue')
                       and values are lists of unique numeric values found for that metric.
            - success: Boolean indicating success.
            - error: Error message if initialization failed.
        """
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")

        # Ensure METRIC_RX is available and initialized
        if not hasattr(self, '_METRIC_RX') or not isinstance(self._METRIC_RX, list):
             self.logger.error("Metric regex patterns (_METRIC_RX) not initialized correctly.")
             return {"metrics": {}, "error": "Metric patterns not initialized", "success": False}

        extracted_metrics: Dict[str, List[float]] = {}
        self.logger.info(f"Starting metric extraction for domain '{self.ACTIVE_DOMAIN}'. Searching for {len(self._METRIC_RX)} metric types.")

        # Iterate through compiled regex patterns for the active domain
        for metric_name, pattern in self._METRIC_RX:
            if not isinstance(pattern, re.Pattern):
                 self.logger.warning(f"Skipping invalid pattern for metric '{metric_name}'.")
                 continue

            found_values_for_metric: List[float] = []
            try:
                # Find all matches for the pattern in the document
                # The regex captures (alias, value_string)
                matches = pattern.findall(document)

                if matches:
                    self.logger.debug(f"Found {len(matches)} potential matches for metric '{metric_name}'") # using pattern: {pattern.pattern}")
                    for match_groups in matches:
                        # Extract the value string (should be the second group)
                        if isinstance(match_groups, tuple) and len(match_groups) >= 2:
                            val_str = match_groups[1] # Group 2 contains the value string
                        elif isinstance(match_groups, str): # If findall returns only the value group
                            val_str = match_groups
                        else:
                            self.logger.debug(f"Unexpected match format for metric '{metric_name}': {match_groups}")
                            continue

                        # Clean the numeric string (remove currency, commas, trailing dots)
                        # Keep the negative sign and decimal point
                        val_str_cleaned = re.sub(r'[^\d.-]', '', str(val_str))
                        # Remove trailing dots that are not part of a decimal number
                        if val_str_cleaned.endswith('.') and val_str_cleaned.count('.') == 1:
                            val_str_cleaned = val_str_cleaned[:-1]
                        if not val_str_cleaned or val_str_cleaned == '-': # Skip empty strings or just '-'
                            continue

                        try:
                             # Convert cleaned string to float
                             value = float(val_str_cleaned)
                             found_values_for_metric.append(value)
                        except ValueError:
                             self.logger.debug(f"Could not convert extracted value '{val_str_cleaned}' (from '{val_str}') to float for metric '{metric_name}'. Skipping.")
                             continue # Skip if conversion fails

            except Exception as e:
                self.logger.error(f"Error processing regex pattern for metric '{metric_name}': {pattern.pattern}. Error: {e}", exc_info=True)
                # Continue to next metric even if one pattern fails

            # Add found values to the results dictionary (store unique values)
            if found_values_for_metric:
                # Use set for deduplication before storing
                unique_values = sorted(list(set(found_values_for_metric)))
                extracted_metrics[metric_name] = unique_values
                self.logger.info(f"Extracted {len(unique_values)} unique values for metric '{metric_name}': {unique_values}")


        return {"metrics": extracted_metrics, "success": True}


    @tool(name="flag_risks", description="Identify potential risk indicators using domain-specific regex patterns")
    @with_tool_metrics
    @with_error_handling
    async def flag_risks(self, document: str) -> Dict[str, Any]:
        """Flags potential risks or sensitive information in the document using regex patterns defined for the active domain.

        Args:
            document: Text content to analyze.

        Returns:
            Dictionary containing:
            - risks: Dictionary where keys are risk types (e.g., 'Change_of_Control')
                     and values are dicts containing:
                       - count: Number of times the risk pattern was found.
                       - sample_contexts: List of short text snippets surrounding the first few matches.
            - success: Boolean indicating success.
            - error: Error message if initialization failed.
        """
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")

        # Ensure RISK_RX is available and initialized
        if not hasattr(self, '_RISK_RX') or not isinstance(self._RISK_RX, dict):
             self.logger.error("Risk regex patterns (_RISK_RX) not initialized correctly.")
             return {"risks": {}, "error": "Risk patterns not initialized", "success": False}

        flagged_risks: Dict[str, Dict[str, Any]] = {}
        self.logger.info(f"Starting risk flagging for domain '{self.ACTIVE_DOMAIN}'. Searching for {len(self._RISK_RX)} risk types.")
        context_window = 50 # Characters before/after match for context snippet
        max_samples = 3 # Max context samples per risk type to keep response size manageable

        # Iterate through compiled risk patterns
        for risk_type, pattern in self._RISK_RX.items():
            if not isinstance(pattern, re.Pattern):
                 self.logger.warning(f"Skipping invalid pattern for risk '{risk_type}'.")
                 continue

            match_contexts: List[str] = []
            match_count = 0
            try:
                # Use finditer to get match objects with positions for context extraction
                matches_iterator = pattern.finditer(document)
                # Iterate through matches to count and get samples
                for match in matches_iterator:
                     match_count += 1
                     # Extract context only for the first few matches
                     if len(match_contexts) < max_samples:
                         match_start_pos = match.start()
                         match_end_pos = match.end()

                         # Calculate context boundaries, ensuring they are within document limits
                         context_start = max(0, match_start_pos - context_window)
                         context_end = min(len(document), match_end_pos + context_window)

                         # Extract context snippet
                         snippet = document[context_start:context_end]

                         # Clean up snippet (replace newlines with spaces, add ellipses for truncation)
                         snippet = snippet.replace("\n", " ").strip()
                         prefix = "..." if context_start > 0 else ""
                         suffix = "..." if context_end < len(document) else ""
                         # Optionally highlight the match within the snippet (e.g., with **match**)
                         # This requires careful index adjustment if prefix is added
                         # highlight_start = match_start_pos - context_start + len(prefix)
                         # highlight_end = match_end_pos - context_start + len(prefix)
                         # formatted_snippet = f"{prefix}{snippet[:highlight_start]}**{snippet[highlight_start:highlight_end]}**{snippet[highlight_end:]}{suffix}"
                         formatted_snippet = f"{prefix}{snippet}{suffix}" # Simpler version without highlight

                         match_contexts.append(formatted_snippet)

                # If any matches were found, store the result
                if match_count > 0:
                    self.logger.info(f"Flagged risk '{risk_type}' {match_count} times.")
                    flagged_risks[risk_type] = {
                        "count": match_count,
                        "sample_contexts": match_contexts
                    }

            except Exception as e:
                self.logger.error(f"Error processing regex pattern for risk '{risk_type}': {pattern.pattern}. Error: {e}", exc_info=True)
                # Continue to next risk type

        return {"risks": flagged_risks, "success": True}

    @tool(name="canonicalise_entities", description="Normalize and deduplicate a list of extracted entities using fuzzy matching")
    @with_tool_metrics
    @with_error_handling
    async def canonicalise_entities(self, entities_input: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes and attempts to merge similar entities from an extraction result using fuzzy matching.

        Args:
            entities_input: The dictionary result from `extract_entities`, expected to
                            have an 'entities' key mapping type to list of strings,
                            OR a list of entity dictionaries (e.g., from another NER tool)
                            where each dict has 'text' and 'type' keys.

        Returns:
            Dictionary containing:
            - canonicalized: Dictionary mapping entity type to a list of canonical entity dicts.
                             Each canonical dict includes 'text' (the chosen canonical form),
                             'count' (how many variants merged), 'type', 'variants' (list of original strings).
            - success: Boolean indicating success.
        """
        if not isinstance(entities_input, dict):
            raise ToolInputError("Input must be a dictionary.", param_name="entities_input")

        # --- Input Handling: Convert various formats to a List[Dict[str, Any]] ---
        entities_list: List[Dict[str, Any]] = []
        raw_entities = entities_input.get('entities')

        if isinstance(raw_entities, dict):
            # Input format: Dict[str, List[str]] (from our LLM extract_entities)
            for entity_type, text_list in raw_entities.items():
                 if isinstance(text_list, list):
                      for text in text_list:
                           if isinstance(text, str) and text.strip():
                                # Basic structure: {'text': ..., 'type': ...}
                                entities_list.append({"text": text.strip(), "type": str(entity_type).upper().strip()})
                 else:
                      self.logger.warning(f"Expected list for entity type '{entity_type}', got {type(text_list)}. Skipping.")
            self.logger.debug(f"Converted dict-based entity input (keys: {list(raw_entities.keys())}) to list format.")
        elif isinstance(raw_entities, list):
            # Input format: List[Dict[str, Any]] (e.g., from SpaCy or other NER tools)
            valid_items = True
            temp_list = []
            for item in raw_entities:
                if isinstance(item, dict) and isinstance(item.get('text'), str) and item['text'].strip() and isinstance(item.get('type'), str) and item['type'].strip():
                     temp_list.append({
                         "text": item['text'].strip(),
                         "type": item['type'].upper().strip(),
                         # Preserve other metadata if present (e.g., start/end char, score)
                         "metadata": {k: v for k, v in item.items() if k not in ['text', 'type']}
                     })
                else:
                     self.logger.warning(f"Skipping invalid item in entity list: {item}")
                     valid_items = False # Mark that some items were invalid, but continue

            entities_list = temp_list
            if not valid_items:
                 self.logger.warning("Some items in the input entity list were invalid or missing required keys ('text', 'type').")
            self.logger.debug(f"Processed list-based entity input ({len(entities_list)} valid items found).")
        else:
            raise ToolInputError("Input dictionary must contain an 'entities' key with either Dict[str, List[str]] or List[Dict[str, Any]].", param_name="entities_input")


        if not entities_list:
            self.logger.info("No valid entities provided to canonicalise.")
            return {"canonicalized": {}, "success": True}


        # --- Canonicalization Logic ---
        # Group entities by type first
        entities_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for entity in entities_list:
            etype = entity.get("type", "UNKNOWN")
            if etype not in entities_by_type:
                 entities_by_type[etype] = []
            entities_by_type[etype].append(entity)

        canonicalized_output: Dict[str, List[Dict[str, Any]]] = {}
        # Similarity threshold for merging (tune this based on results)
        # Higher value means more strict matching.
        similarity_threshold = 85 # e.g., "Acme Corp." vs "Acme Corporation" should merge

        # Process each entity type separately
        for entity_type, entity_group in entities_by_type.items():
            self.logger.debug(f"Canonicalising {len(entity_group)} entities of type '{entity_type}'...")
            # Sort entities by text length descending? Might help pick better canonical forms.
            # entity_group.sort(key=lambda x: len(x.get("text", "")), reverse=True)

            merged_entities: List[Dict[str, Any]] = []
            processed_indices = set()

            # Greedy clustering loop
            for i in range(len(entity_group)):
                if i in processed_indices:
                    continue

                current_entity = entity_group[i]
                # Use the text from this entity as the initial canonical form for the cluster
                canonical_form = current_entity.get("text", "")
                if not canonical_form: # Skip empty entities
                     processed_indices.add(i)
                     continue

                # Start a new cluster
                cluster_variants = [current_entity] # Store original dicts in cluster
                processed_indices.add(i)

                # Compare with remaining unprocessed entities of the same type
                for j in range(i + 1, len(entity_group)):
                    if j in processed_indices:
                        continue

                    other_entity = entity_group[j]
                    other_text = other_entity.get("text", "")
                    if not other_text:
                         processed_indices.add(j)
                         continue

                    # Calculate similarity score (case-insensitive)
                    # Consider using fuzz.token_sort_ratio for better handling of word order differences
                    score = fuzz.ratio(canonical_form.lower(), other_text.lower())

                    if score >= similarity_threshold:
                         # Add to the current cluster
                         cluster_variants.append(other_entity)
                         processed_indices.add(j)
                         self.logger.debug(f"  Merging '{other_text}' into cluster for '{canonical_form}' (score: {score:.0f})")
                         # Optional: Refine canonical form based on merged entities (e.g., choose longest)
                         # if len(other_text) > len(canonical_form):
                         #     canonical_form = other_text


                # Create the final canonical entry for this cluster
                if cluster_variants:
                     # Collect all unique text variants from the cluster
                     unique_texts = sorted(list(set(e.get("text") for e in cluster_variants)))
                     # TODO: Add logic to merge metadata if needed (e.g., average scores, combine positions)
                     # merged_meta = {}

                     canonical_entry = {
                         "text": canonical_form, # The chosen representative text
                         "count": len(cluster_variants), # How many original entities merged
                         "type": entity_type,
                         "variants": unique_texts # List of unique strings that were merged
                         # "merged_metadata": merged_meta
                     }
                     merged_entities.append(canonical_entry)

            # Sort the canonicalized list for this type (e.g., by count descending, then text)
            merged_entities.sort(key=lambda x: (-x.get("count", 0), x.get("text", "")))
            canonicalized_output[entity_type] = merged_entities
            self.logger.info(f"Canonicalised type '{entity_type}': {len(entity_group)} variants -> {len(merged_entities)} unique entities.")


        return {"canonicalized": canonicalized_output, "success": True}

    ###############################################################################
    # Batch Processing Tool                                                       #
    ###############################################################################

    @tool(name="process_document_batch", description="Process a batch of documents/texts through a pipeline of operations")
    @with_tool_metrics
    @with_error_handling
    async def process_document_batch(self, inputs: List[Dict[str, Any]], operations: List[Dict[str, Any]], max_concurrency: int = 5):
        """Processes a list of input dictionaries through a sequence of operations concurrently for each step.

        Args:
            inputs: List of input dictionaries. Each dictionary should contain the
                    initial data, typically under a key like "content" or "document_path".
            operations: List of operation specifications to apply sequentially to each input.
                Each operation dict must contain:
                - operation: Name of the tool method to call (e.g., "convert_document", "chunk_document").
                - output_key: The key under which to store the result in the input dict for subsequent steps.
                - params: Dictionary of parameters to pass to the operation function.
                Optional keys for operation dict:
                - input_key: Key in the input dict to use as the primary positional argument for the operation
                             (e.g., "document" or "document_path"). Defaults based on operation name.
                - input_keys_map: Dict mapping function parameter names to keys in the input dict.
                                  Allows passing multiple inputs from the dict as keyword arguments.
                                  e.g., {"document": "doc_content", "num_questions": "qa_count"}
                - promote_output: Optional string key within the operation's result dictionary.
                                  If set, the value of this key will be promoted to the top-level 'content' key
                                  in the processing state for the next step. Useful for chaining primary outputs.
                                  E.g., if chunk_document returns {"chunks": [...]} and promote_output is "chunks",
                                  the next step's default input will be the list of chunks.
            max_concurrency: Maximum number of input items to process in parallel *for each operation step*.

        Returns:
            List of dictionaries, representing the final state of each input dictionary after all operations.
            Each dictionary will contain the original input plus keys for each operation's output and an '_error_log'.
        """
        if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
            raise ToolInputError("Input 'inputs' must be a list of dictionaries.", param_name="inputs")
        if not isinstance(operations, list) or not all(isinstance(op, dict) for op in operations):
             raise ToolInputError("Input 'operations' must be a list of dictionaries.", param_name="operations")
        if max_concurrency <= 0:
             self.logger.warning("max_concurrency must be positive, setting to 1.")
             max_concurrency = 1


        # --- Initialize results state ---
        # Start with a deep copy? Shallow copy is usually sufficient if operations don't modify input dicts directly.
        results_state: List[Dict[str, Any]] = []
        for i, item in enumerate(inputs):
             state_item = item.copy() # Shallow copy of the input dict
             state_item["_original_index"] = i
             state_item["_error_log"] = [] # Log errors encountered for this item across steps
             state_item["_status"] = "pending" # Track item status
             results_state.append(state_item)


        self.logger.info(f"Starting batch processing for {len(inputs)} items through {len(operations)} operations.")

        # --- Apply operations sequentially across the batch ---
        for op_index, op_spec in enumerate(operations):
            op_name = op_spec.get("operation")
            op_output_key = op_spec.get("output_key")
            op_params = op_spec.get("params", {})
            op_input_key = op_spec.get("input_key") # Specific key for primary positional arg
            op_input_map = op_spec.get("input_keys_map", {}) # Map state keys to kwarg names
            op_promote = op_spec.get("promote_output") # Key in result to promote to state['content']

            # --- Validate operation specification ---
            if not op_name or not isinstance(op_name, str) or op_name not in self.op_map:
                error_msg = f"Invalid or unknown operation '{op_name}' specified at step {op_index}."
                self.logger.error(error_msg)
                # Mark all items as failed for this step if op spec is invalid
                for item_state in results_state:
                    if item_state["_status"] != "failed": # Avoid overwriting earlier failure
                         item_state["_error_log"].append(error_msg + " (Operation Skipped)")
                         item_state["_status"] = "failed"
                continue # Skip to the next operation in the list

            if not op_output_key or not isinstance(op_output_key, str):
                 error_msg = f"Missing or invalid 'output_key' for operation '{op_name}' at step {op_index}."
                 self.logger.error(error_msg)
                 for item_state in results_state:
                     if item_state["_status"] != "failed":
                          item_state["_error_log"].append(error_msg + " (Operation Skipped)")
                          item_state["_status"] = "failed"
                 continue # Skip to the next operation


            op_func = self.op_map[op_name]
            self.logger.info(f"--- Starting Batch Operation {op_index + 1}/{len(operations)}: '{op_name}' (Concurrency: {max_concurrency}) ---")

            # --- Define the worker function for this specific operation ---
            # Pass loop variables as default args to fix B023 closure issues
            async def _apply_op_to_item(
                item_state: Dict[str, Any],
                semaphore,  # Required parameter without default
                current_op_index=op_index,
                current_op_name=op_name,
                current_op_func=op_func,
                current_op_output_key=op_output_key,
                current_op_params=op_params,
                current_op_input_key=op_input_key,
                current_op_input_map=op_input_map,
                current_op_promote=op_promote,
            ):
                item_idx = item_state["_original_index"]
                # Skip if item already failed critically in a previous step
                if item_state["_status"] == "failed":
                     self.logger.debug(f"Skipping operation '{current_op_name}' for item {item_idx} due to previous failure.")
                     return item_state # Return unchanged state

                async with semaphore: # Use the passed semaphore instance
                    self.logger.debug(f"Applying operation '{current_op_name}' to item {item_idx}")
                    call_kwargs = {} # Arguments for the tool method call

                    try:
                        # --- Prepare arguments for the operation function ---
                        primary_input_value = None

                        # 1. Determine primary input key and value (for positional argument)
                        actual_input_key = current_op_input_key # Use specified key if provided
                        if not actual_input_key:
                            # Default primary input logic (e.g., 'document_path' for convert, 'content'/'document' otherwise)
                            if current_op_name == "convert_document" or current_op_name == "convert_document_op":
                                actual_input_key = "document_path"
                            elif current_op_name == "canonicalise_entities":
                                 actual_input_key = "entities_input" # Special case
                            elif current_op_name == "batch_format_texts":
                                 actual_input_key = "texts" # Special case
                            else:
                                # Default to 'document' if present, else 'content'
                                actual_input_key = "document" if "document" in item_state else "content"

                        if actual_input_key not in item_state:
                            raise ToolInputError(f"Required input key '{actual_input_key}' not found in state for item {item_idx}.", param_name=actual_input_key)
                        primary_input_value = item_state[actual_input_key]

                        # Determine the parameter name for the primary input (convention or inspection)
                        # Convention used here:
                        primary_arg_name = "document_path" if current_op_name.startswith("convert_document") else \
                                           "entities_input" if current_op_name == "canonicalise_entities" else \
                                           "texts" if current_op_name == "batch_format_texts" else \
                                           "document" # Default: 'document'

                        # Add primary input to kwargs (most tools take primary input as first arg, handled by convention)
                        # If the tool expects it as a keyword arg, it should also be in input_keys_map or params.
                        # We will pass it positionally if possible, or rely on kwargs.
                        # For simplicity, we'll prepare all args as kwargs.
                        call_kwargs[primary_arg_name] = primary_input_value

                        # 2. Handle mapped keyword inputs (input_keys_map)
                        if isinstance(current_op_input_map, dict):
                            for param_name, state_key in current_op_input_map.items():
                                if state_key not in item_state:
                                    raise ToolInputError(f"Mapped input key '{state_key}' (for param '{param_name}') not found in state for item {item_idx}.", param_name=state_key)
                                call_kwargs[param_name] = item_state[state_key]

                        # 3. Add explicit parameters from op_spec['params']
                        # These override mapped inputs if names clash
                        if isinstance(current_op_params, dict):
                            call_kwargs.update(current_op_params)

                        # --- Execute the operation ---
                        self.logger.debug(f"Calling {current_op_name} for item {item_idx} with kwargs: {list(call_kwargs.keys())}")
                        op_result = await current_op_func(**call_kwargs)

                        # --- Process result ---
                        if not isinstance(op_result, dict):
                             # Handle cases where tool might return non-dict on error/unexpectedly
                             raise ToolError("INVALID_RESULT_FORMAT", details={"step": current_op_index, "operation": current_op_name, "result_type": type(op_result).__name__, "item_index": item_idx})

                        # Store the entire result dict under the specified output key
                        item_state[current_op_output_key] = op_result

                        # Promote output key to 'content' for next step if requested
                        if current_op_promote and isinstance(current_op_promote, str):
                             if current_op_promote in op_result:
                                 item_state["content"] = op_result[current_op_promote]
                                 self.logger.debug(f"Promoted '{current_op_promote}' to 'content' for item {item_idx}")
                             else:
                                 self.logger.warning(f"Cannot promote key '{current_op_promote}' for item {item_idx}: key not found in result of '{current_op_name}'. Result keys: {list(op_result.keys())}")

                        # Check success flag within the result and update item status
                        if not op_result.get("success", False): # Be strict: success must be explicitly True
                             err_msg = op_result.get("error", f"Operation '{current_op_name}' reported failure.")
                             err_code = op_result.get("error_code", "PROCESSING_ERROR")
                             log_entry = f"Step {current_op_index+1} ({current_op_name}) Failed: [{err_code}] {err_msg}"
                             item_state["_error_log"].append(log_entry)
                             item_state["_status"] = "failed" # Mark item as failed for subsequent steps
                             self.logger.warning(f"Operation '{current_op_name}' failed for item {item_idx}: {err_msg}")
                        else:
                             item_state["_status"] = "processed" # Mark as processed for this step


                    except ToolInputError as tie:
                        error_msg = f"Step {current_op_index+1} ({current_op_name}) Input Error: [{tie.code}] {tie.message}"
                        self.logger.error(f"{error_msg} for item {item_idx}", exc_info=False) # Less verbose logging for input errors
                        item_state["_error_log"].append(error_msg)
                        item_state["_status"] = "failed"
                        # Store error details under the intended output key as well
                        item_state[current_op_output_key] = {"error": tie.message, "error_code": tie.code, "success": False}
                    except ToolError as te:
                        error_msg = f"Step {current_op_index+1} ({current_op_name}) Tool Error: [{te.code}] {te.message}"
                        self.logger.error(f"{error_msg} for item {item_idx}", exc_info=True)
                        item_state["_error_log"].append(error_msg)
                        item_state["_status"] = "failed"
                        item_state[current_op_output_key] = {"error": te.message, "error_code": te.code, "success": False}
                    except Exception as e:
                        error_msg = f"Step {current_op_index+1} ({current_op_name}) Unexpected Error: {str(e)}"
                        self.logger.error(f"{error_msg} for item {item_idx}", exc_info=True)
                        item_state["_error_log"].append(error_msg)
                        item_state["_status"] = "failed"
                        item_state[current_op_output_key] = {"error": str(e), "error_type": type(e).__name__, "success": False}

                    return item_state # Return the updated state

            # --- Run tasks for the current operation step ---
            step_semaphore = asyncio.Semaphore(max_concurrency) # Semaphore for this step
            step_tasks = []
            for item_state in results_state:
                 # Pass the current operation's details and the semaphore to the worker
                 task = _apply_op_to_item(
                     item_state,
                     step_semaphore,  # Pass semaphore as second parameter
                     current_op_index=op_index,
                     current_op_name=op_name,
                     current_op_func=op_func,
                     current_op_output_key=op_output_key,
                     current_op_params=op_params.copy(), # Pass copies to avoid closure issues?
                     current_op_input_key=op_input_key,
                     current_op_input_map=op_input_map.copy(),
                     current_op_promote=op_promote,
                 )
                 step_tasks.append(task)

            # Gather results for this step - updates results_state in place
            updated_states = await asyncio.gather(*step_tasks)
            # Overwrite results_state with the updated states returned by gather
            results_state = updated_states

            # Log summary after each step
            step_success_count = sum(1 for s in results_state if s.get("_status") == "processed")
            step_fail_count = sum(1 for s in results_state if s.get("_status") == "failed")
            self.logger.info(f"--- Finished Batch Operation {op_index + 1}: '{op_name}' (Success: {step_success_count}, Failed: {step_fail_count}) ---")


        # --- Final step: Clean up internal fields ---
        final_results = []
        for item_state in results_state:
             final_item = item_state.copy() # Copy again to avoid modifying original state list if reused
             final_item.pop("_original_index", None)
             final_item.pop("_status", None)
             # Keep error log for inspection
             final_results.append(final_item)


        self.logger.info(f"Batch processing finished for {len(inputs)} items.")
        # The return value is the list of final state dictionaries
        return final_results