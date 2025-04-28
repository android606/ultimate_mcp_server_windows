"""Ultimate Document Processing Toolkit for MCP Server integrated as a BaseTool.

A comprehensive, fault-tolerant toolkit for document processing, providing:

1. Document Conversion & OCR
   - PDF/Office/Image → Markdown/Text/HTML/JSON/doctags
   - Multiple extraction strategies: Docling, Direct Text, OCR, Hybrid modes
   - Advanced OCR with image preprocessing (denoise, deskew, thresholding)
   - LLM-based OCR text enhancement and error correction
   - Multi-language OCR support
   - PDF structure analysis (metadata, outline, font info, OCR needs estimation)
   - Optional OCR quality assessment
   - Acceleration options (CPU/CUDA/MPS) for Docling/potential future models
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
   - Extract tables from documents (Docling) in CSV, JSON, or pandas format
   - Convert HTML tables to markdown tables
   - Detect tables in images during OCR

5. Document Analysis
   - Section identification (regex-based)
   - Entity extraction (LLM-based)
   - Metrics extraction (regex-based)
   - Risk flagging (regex-based)
   - QA Generation (LLM-based)
   - Summarization (LLM-based)
   - Classification (LLM-based)
   - Canonicalization of entities

6. Batch Processing
   - Process multiple documents with multiple operations
   - Configurable concurrency

This unified toolkit combines Document Conversion, Universal Document-Processing,
HTML-to-Markdown conversion, and Advanced OCR/Text Enhancement capabilities
into a single, coherent BaseTool class.
"""

from __future__ import annotations

###############################################################################
# Imports                                                                     #
###############################################################################
# Standard library imports
import asyncio
import base64
import csv
import functools
import hashlib
import html
import io
import json
import math
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
    Set,
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
    import numpy as np
    import pandas as pd
    import tiktoken
    from PIL import Image as PILImage
    from tiktoken import Encoding

# ───────────────────── Optional Dependency Imports ─────────────────────────
# --- Core Document Processing ---
try:
    from markdownify import markdownify as _markdownify_fallback
except ModuleNotFoundError:
    _markdownify_fallback = None

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ModuleNotFoundError:
    pd = None  # type: ignore
    _PANDAS_AVAILABLE = False

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ModuleNotFoundError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False

# --- Docling (Advanced PDF/Office Conversion) ---
_DOCLING_AVAILABLE = False
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
except ImportError:
    pass  # Will log warning later

# --- Basic Fallback Document Libraries ---
try:
    import PyPDF2

    _PYPDF2_AVAILABLE = True
except ImportError:
    PyPDF2 = None
    _PYPDF2_AVAILABLE = False

try:
    import docx

    _DOCX_AVAILABLE = True
except ImportError:
    docx = None
    _DOCX_AVAILABLE = False

# --- OCR and Image Processing Dependencies ---
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter

    _PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageFilter = None  # type: ignore
    _PIL_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    import pytesseract

    _PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None  # type: ignore
    _PYTESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes, convert_from_path

    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_bytes = None  # type: ignore
    convert_from_path = None  # type: ignore
    _PDF2IMAGE_AVAILABLE = False

# --- Direct PDF Text Extraction Libraries ---
try:
    import pdfplumber

    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None  # type: ignore
    _PDFPLUMBER_AVAILABLE = False

try:
    import pymupdf  # PyMuPDF

    _PYMUPDF_AVAILABLE = True
except ImportError:
    pymupdf = None  # type: ignore
    _PYMUPDF_AVAILABLE = False

# Add these imports at the top with the other imports
try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    _TRAFILATURA_AVAILABLE = False

try:
    import readability
    _READABILITY_AVAILABLE = True
except ImportError:
    readability = None
    _READABILITY_AVAILABLE = False

# ───────────────────── MCP Server Imports ─────────────────────────
from ultimate_mcp_server.constants import Provider, TaskType  # noqa: E402
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError  # noqa: E402
from ultimate_mcp_server.tools.base import (  # noqa: E402
    BaseTool,
    tool,
    with_error_handling,
    with_retry,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.completion import generate_completion  # noqa: E402
from ultimate_mcp_server.utils import get_logger  # noqa: E402

# Setup loggers - BaseTool provides self.logger, but we keep one for module level if needed
_MODULE_LOG = get_logger("ultimate_mcp_server.tools.document_processing_tool")

DEFAULT_EXTRACTION_STRATEGY = "hybrid_direct_ocr"  # Robust default

###############################################################################
# DocumentProcessingTool Class                                                #
##############################################################################


class DocumentProcessingTool(BaseTool):
    """
    Comprehensive document processing toolkit for MCP Server.

    Provides tools for document conversion (including advanced OCR), HTML processing,
    chunking, table extraction, document analysis, and batch processing.
    Leverages Docling, OCR (Tesseract), direct text extraction (PyMuPDF/PDFPlumber),
    and LLMs for high-quality results across various document types.
    """

    tool_name = "document_processing"
    description = "A unified toolkit for document conversion (PDF/Office/Image), OCR, chunking, analysis, and HTML processing."

    def __init__(self, mcp_server):
        """Initialize the document processing tool."""
        super().__init__(mcp_server)

        # --- Dependency Availability Tracking ---
        self._docling_available = _DOCLING_AVAILABLE
        self._pypdf2_available = _PYPDF2_AVAILABLE
        self._docx_available = _DOCX_AVAILABLE
        self._pandas_available = _PANDAS_AVAILABLE
        self._tiktoken_available = _TIKTOKEN_AVAILABLE
        self._numpy_available = _NUMPY_AVAILABLE
        self._pil_available = _PIL_AVAILABLE
        self._cv2_available = _CV2_AVAILABLE
        self._pytesseract_available = _PYTESSERACT_AVAILABLE
        self._pdf2image_available = _PDF2IMAGE_AVAILABLE
        self._pdfplumber_available = _PDFPLUMBER_AVAILABLE
        self._pymupdf_available = _PYMUPDF_AVAILABLE
        self._trafilatura_available = _TRAFILATURA_AVAILABLE
        self._readability_available = _READABILITY_AVAILABLE

        # Log dependency status
        if not self._docling_available:
            self.logger.warning(
                "Docling library not available. Advanced PDF/Office conversion features (layout-aware markdown, etc.) will be disabled."
            )
        ocr_deps = {
            "Pillow": self._pil_available,
            "numpy": self._numpy_available,
            "opencv-python": self._cv2_available,
            "pytesseract": self._pytesseract_available,
            "pdf2image": self._pdf2image_available,
        }
        missing_ocr = [name for name, avail in ocr_deps.items() if not avail]
        if missing_ocr:
            self.logger.warning(
                f"Missing OCR dependencies: {', '.join(missing_ocr)}. OCR functionality will be limited or disabled."
            )

        direct_text_deps = {
            "pdfplumber": self._pdfplumber_available,
            "pymupdf": self._pymupdf_available,
        }
        missing_direct = [name for name, avail in direct_text_deps.items() if not avail]  # noqa: F841
        if not self._pdfplumber_available and not self._pymupdf_available:
            self.logger.warning(
                "Missing direct PDF text extraction libraries: pdfplumber and pymupdf. Direct text extraction strategies will be unavailable."
            )

        # ───────────────────── Acceleration Device Mapping (Docling) ──────────────
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
        # Docling handles all. OCR/Direct primarily produce text/markdown.
        self._VALID_FORMATS = {"markdown", "text", "html", "json", "doctags"}
        self._OCR_COMPATIBLE_FORMATS = {"text", "markdown"}

        # ───────────────────── Extraction Strategy Mapping ────────────────────
        self._VALID_EXTRACTION_STRATEGIES = {
            "docling",  # Layout-aware conversion via Docling
            "direct_text",  # Direct text extraction (PyMuPDF/PDFPlumber)
            "ocr",  # Full OCR pipeline (PDF->Image->Tesseract)
            "hybrid_direct_ocr",  # Try Direct Text, fallback to OCR
            # Future: "hybrid_docling_ocr", "auto"
        }
        self._DEFAULT_EXTRACTION_STRATEGY = DEFAULT_EXTRACTION_STRATEGY

        # ───────────────────── HTML Detection Patterns ─────────────────────────
        _RE_FLAGS = re.MULTILINE | re.IGNORECASE
        self.HTML_PATTERNS: Sequence[Pattern] = [
            re.compile(p, _RE_FLAGS)
            for p in (
                r"<\s*[a-zA-Z]+[^>]*>",  # Opening tag
                r"<\s*/\s*[a-zA-Z]+\s*>",  # Closing tag
                r"&[a-zA-Z]+;",  # HTML entity
                r"&#[0-9]+;",  # Numeric entity
                r"<!\s*DOCTYPE",  # Doctype declaration
                r"<!\s*--",  # HTML comment start
            )
        ]

        # ───────────────────── Domain Rules (Analysis) ───────────────────────
        self.DOMAIN_RULES: Dict[str, Dict[str, Any]] = {
            # (Domain rules copied from original script - assuming these are correct)
            "generic": {
                "classification": {
                    "labels": ["Report", "Contract", "Presentation", "Memo", "Email", "Manual"],
                    "prompt_prefix": "Classify the document into exactly one of: ",
                },
                "sections": {
                    "boundary_regex": r"^\s*(chapter\s+\d+|section\s+\d+|[A-Z][A-Za-z\s]{3,80})$",
                    "custom": [],
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
                    "labels": [
                        "10-K",
                        "Credit Agreement",
                        "Investor Deck",
                        "Press Release",
                        "Board Minutes",
                        "NDA",
                        "LPA",
                        "CIM",
                    ],
                    "prompt_prefix": "Identify the document type (finance domain): ",
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
                    "revenue": {
                        "aliases": [
                            "revenue",
                            "net sales",
                            "total sales",
                            "sales revenue",
                            "turnover",
                        ]
                    },
                    "ebitda": {
                        "aliases": ["ebitda", "adj. ebitda", "operating profit", "operating income"]
                    },
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
                    "prompt_prefix": "Classify the legal document into exactly one of: ",
                },
                "sections": {
                    "boundary_regex": r"^\s*(article\s+\d+|section\s+\d+|[A-Z][A-Za-z\s]{3,80})$",
                    "custom": [
                        {"regex": r"definitions", "label": "Definitions"},
                        {"regex": r"termination", "label": "Termination"},
                        {"regex": r"confidentiality", "label": "Confidentiality"},
                    ],
                },
                "metrics": {},  # Legal domain might not have standard numeric metrics
                "risks": {
                    "Indemnity": r"indemnif(y|ication)",
                    "Liquidated_Damages": r"liquidated damages",
                    "Governing_Law_NY": r"governing law.*new york",
                    "Governing_Law_DE": r"governing law.*delaware",
                },
            },
            "medical": {
                "classification": {
                    "labels": [
                        "Clinical Study",
                        "Patient Report",
                        "Lab Results",
                        "Prescription",
                        "Care Plan",
                    ],
                    "prompt_prefix": "Classify the medical document: ",
                },
                "sections": {
                    "boundary_regex": r"^\s*(section\s+\d+|[A-Z][A-Za-z\s]{3,80})$",
                    "custom": [
                        {"regex": r"diagnosis", "label": "Diagnosis"},
                        {"regex": r"treatment", "label": "Treatment"},
                        {"regex": r"medications", "label": "Medications"},
                        {"regex": r"allergies", "label": "Allergies"},
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
            self.logger.warning(
                f"Unknown DOC_DOMAIN '{self.ACTIVE_DOMAIN}', defaulting to 'generic'."
            )
            self.ACTIVE_DOMAIN = "generic"
        self.instruction_json: Dict[str, Any] = self.DOMAIN_RULES[self.ACTIVE_DOMAIN]

        # Compile regex patterns for sections, metrics, and risks
        try:
            self._BOUND_RX = re.compile(
                self.instruction_json["sections"].get("boundary_regex", r"$^"), re.M
            )  # Default matches nothing
        except re.error as e:
            self.logger.error(
                f"Invalid boundary regex for domain {self.ACTIVE_DOMAIN}: {self.instruction_json['sections'].get('boundary_regex')}. Error: {e}"
            )
            self._BOUND_RX = re.compile(r"$^")  # Fallback

        self._CUSTOM_SECT_RX = []
        for d in self.instruction_json["sections"].get("custom", []):
            try:
                self._CUSTOM_SECT_RX.append((re.compile(d["regex"], re.I), d["label"]))
            except re.error as e:
                self.logger.error(
                    f"Invalid custom section regex '{d['regex']}' for domain {self.ACTIVE_DOMAIN}: {e}"
                )

        self._METRIC_RX: List[Tuple[str, re.Pattern]] = []
        for key, cfg in self.instruction_json.get("metrics", {}).items():
            aliases = cfg.get("aliases", [])
            if aliases:  # Ensure aliases is not empty
                try:
                    sorted_aliases = sorted(aliases, key=len, reverse=True)
                    joined = "|".join(re.escape(a) for a in sorted_aliases)
                    if joined:
                        pattern = re.compile(
                            rf"""
                            (?i)\b({joined})\b             # Alias (Group 1)
                            [\s:–-]*                       # Optional separator
                            ([$€£]?\s?                     # Optional currency symbol with optional space (Group 2 - part 1)
                             -?                           # Optional negative sign
                             \d[\d,.]*                     # Number (digits with optional commas/decimals)
                            )
                            """,
                            re.VERBOSE | re.MULTILINE,
                        )
                        self._METRIC_RX.append((key, pattern))
                except re.error as e:
                    self.logger.error(
                        f"Invalid metric regex for alias group '{key}' in domain {self.ACTIVE_DOMAIN}: {e}"
                    )

        self._RISK_RX = {}
        for t, pat_str in self.instruction_json.get("risks", {}).items():
            try:
                self._RISK_RX[t] = re.compile(pat_str, re.I)
            except re.error as e:
                self.logger.error(
                    f"Invalid risk regex for '{t}' in domain {self.ACTIVE_DOMAIN}: '{pat_str}'. Error: {e}"
                )

        self._DOC_LABELS = self.instruction_json["classification"].get("labels", [])
        self._CLASS_PROMPT_PREFIX = self.instruction_json["classification"].get("prompt_prefix", "")

        # ────────────────── Content Type Detection Patterns ─────────────────────
        self.PATTERNS: Dict[str, List[Tuple[Pattern, float]]] = {
            "html": [
                (re.compile(r"<html", re.I), 5.0),
                (re.compile(r"<head", re.I), 4.0),
                (re.compile(r"<body", re.I), 4.0),
                (re.compile(r"</(div|p|span|a|li)>", re.I), 1.0),
                (
                    re.compile(r"<[a-z][a-z0-9]*\s+[^>]*>", re.I),
                    0.8,
                ),  # Generic opening tag with attributes
                (re.compile(r"<!DOCTYPE", re.I), 5.0),
                (re.compile(r"&\w+;"), 0.5),  # Entities
            ],
            "markdown": [
                (re.compile(r"^#{1,6}\s+", re.M), 4.0),  # Headings
                (re.compile(r"^\s*[-*+]\s+", re.M), 2.0),  # Unordered lists
                (re.compile(r"^\s*\d+\.\s+", re.M), 2.0),  # Ordered lists
                (re.compile(r"`[^`]+`"), 1.5),  # Inline code
                (re.compile(r"^```", re.M), 5.0),  # Code blocks
                (re.compile(r"\*{1,2}[^*\s]+?\*{1,2}"), 1.0),  # Emphasis/Strong
                (re.compile(r"!\[.*?\]\(.*?\)", re.M), 3.0),  # Images
                (re.compile(r"\[.*?\]\(.*?\)", re.M), 2.5),  # Links
                (re.compile(r"^>.*", re.M), 2.0),  # Blockquotes
                (re.compile(r"^-{3,}$", re.M), 3.0),  # Horizontal rules
            ],
            "code": [
                (re.compile(r"def\s+\w+\(.*\):"), 3.0),  # Python function
                (re.compile(r"class\s+\w+"), 3.0),  # Class definition (generic)
                (re.compile(r"import\s+|from\s+"), 3.0),  # Import statements (Pythonic)
                (
                    re.compile(
                        r"((function\s+\w+\(|const|let|var)\s*.*?=>|\b(document|console|window)\.)"
                    ),
                    3.0,
                ),  # JS/Java function
                (re.compile(r"public\s+|private\s+|static\s+"), 2.5),  # Java/C# keywords
                (re.compile(r"#include"), 3.0),  # C/C++ include
                (re.compile(r"<\?php"), 4.0),  # PHP tag
                (re.compile(r"console\.log"), 2.0),  # JS logging
                (re.compile(r";\s*$"), 1.0),  # Semicolon line endings (C-like, JS)
                (
                    re.compile(r"\b(var|let|const|int|float|string|bool)\b"),
                    1.5,
                ),  # Common variable keywords
                (re.compile(r"//.*$"), 1.0),  # Single line comment
                (re.compile(r"/\*.*?\*/", re.S), 1.5),  # Multi-line comment
            ],
        }
        # Patterns for specific language detection within code blocks
        self.LANG_PATTERNS: List[Tuple[Pattern, str]] = [
            (re.compile(r"(def\s+\w+\(.*?\):|import\s+|from\s+\S+\s+import)"), "python"),
            (
                re.compile(
                    r"((function\s+\w+\(|const|let|var)\s*.*?=>|\b(document|console|window)\.)"
                ),
                "javascript",
            ),
            (re.compile(r"<(\w+)(.*?)>.*?</\1>", re.S), "html"),  # More specific HTML check
            (re.compile(r"<\?php"), "php"),
            (re.compile(r"(public|private|protected)\s+(static\s+)?(void|int|String)"), "java"),
            (re.compile(r"#include\s+<"), "c/c++"),
            (re.compile(r"using\s+System;"), "c#"),
            (re.compile(r"(SELECT|INSERT|UPDATE|DELETE)\s+.*FROM", re.I), "sql"),
            (re.compile(r":\s+\w+\s*\{"), "css"),  # Basic CSS block detection
            (
                re.compile(r"^[^:]+:\s* # YAML key-value", re.M | re.X),
                "yaml",
            ),  # Simple YAML key check
            (re.compile(r"\$\w+"), "shell/bash"),  # Basic shell variable
        ]

        # ───────────────────── Lazy Loading State ─────────────────────────────
        self._tiktoken_enc_instance: Union["Encoding", bool, None] = (
            None  # None=not loaded, False=failed/unavailable, Encoding=loaded
        )

        # ───────────────────── Operation Map for Batch Processing ─────────────
        # Defined here to ensure methods are bound to the instance
        self.op_map: Dict[str, Callable[..., Awaitable[Any]]] = {
            # Core Conversion & Processing
            "convert_document": self.convert_document_op,  # Wrapper for batch
            "ocr_image": self.ocr_image,
            "enhance_ocr_text": self.enhance_ocr_text,
            # HTML & Markdown
            "clean_and_format_text_as_markdown": self.clean_and_format_text_as_markdown,
            "optimize_markdown_formatting": self.optimize_markdown_formatting,
            "detect_content_type": self.detect_content_type,
            # Chunking
            "chunk_document": self.chunk_document,
            # Analysis
            "summarize_document": self.summarize_document,
            "extract_entities": self.extract_entities,
            "generate_qa_pairs": self.generate_qa_pairs,
            "classify_document": self.classify_document,
            "identify_sections": self.identify_sections,
            "extract_metrics": self.extract_metrics,
            "flag_risks": self.flag_risks,
            "canonicalise_entities": self.canonicalise_entities,
            "analyze_pdf_structure": self.analyze_pdf_structure,
            # Tables
            "extract_tables": self.extract_tables,  # Currently relies on Docling
            # Batch Formatting (Handles multiple texts, not documents)
            "batch_format_texts": self.batch_format_texts,
        }

        # ───────────────────── Markdown Processing Regex ───────────────────────
        self._BULLET_RX = re.compile(r"^[•‣▪◦‧﹒∙·] ?", re.MULTILINE)

        # ───────────────────── OCR Caching (Placeholder) ───────────────────────
        # Simple dict cache for now, could be replaced with more sophisticated caching if needed
        self._OCR_CACHE: Dict[str, Any] = {}

    ###############################################################################
    # Utility Methods (Private)                                                   #
    ###############################################################################

    # ───────────────────── Core Utilities ─────────────────────────────

    def _get_docling_converter(self, device, threads: int):
        """Create a Docling DocumentConverter."""
        if not self._docling_available:
            self.logger.error("Docling is not available. Cannot create converter.")
            raise ToolError("DEPENDENCY_MISSING", details={"dependency": "docling"})

        opts = PdfPipelineOptions()
        opts.do_ocr = False  # Docling OCR off by default, we use Tesseract if needed
        opts.generate_page_images = False
        opts.accelerator_options = AcceleratorOptions(num_threads=threads, device=device)
        try:
            converter_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            return DocumentConverter(format_options=converter_options)
        except Exception as e:
            self.logger.error(f"Failed to initialize Docling DocumentConverter: {e}", exc_info=True)
            raise ToolError(
                "INITIALIZATION_FAILED", details={"component": "DocumentConverter", "error": str(e)}
            ) from e

    def _get_input_path_or_temp(
        self, document_path: Optional[str], document_data: Optional[bytes]
    ) -> Tuple[Path, bool]:
        """
        Gets a valid Path object for input. If data is provided, saves to a temp file.

        Returns:
            Tuple (Path object, boolean indicating if it's a temporary file).
        Raises:
            ToolInputError if neither path nor data is valid.
        """
        is_temp = False
        if document_path:
            path = Path(document_path)
            if not path.is_file():
                raise ToolInputError(
                    f"Input file not found: {document_path}", param_name="document_path"
                )
            return path, is_temp
        elif document_data:
            try:
                # Detect file type (basic guess) for suffix
                suffix = ".pdf"  # Default
                if document_data.startswith(b"%PDF"):
                    suffix = ".pdf"
                elif document_data[6:10] in (b"JFIF", b"Exif"):
                    suffix = ".jpg"
                elif document_data.startswith(b"\x89PNG\r\n\x1a\n"):
                    suffix = ".png"
                # Add more detections if needed (TIFF, etc.)

                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(document_data)
                    path = Path(tmp_file.name)
                is_temp = True
                self.logger.debug(f"Saved input data to temporary file: {path}")
                return path, is_temp
            except Exception as e:
                raise ToolError(
                    "TEMP_FILE_ERROR",
                    details={"error": f"Failed to save input data to temporary file: {e}"},
                ) from e
        else:
            raise ToolInputError("Either 'document_path' or 'document_data' must be provided.")

    @contextmanager
    def _handle_temp_file(self, path: Path, is_temp: bool):
        """Context manager to clean up temporary file."""
        try:
            yield path
        finally:
            if is_temp and path.exists():
                try:
                    path.unlink()
                    self.logger.debug(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    self.logger.warning(f"Failed to delete temporary file {path}: {e}")

    def _tmp_path(self, src: str, fmt: str) -> Path:
        """Generate a temporary file path for output based on source name and format."""
        # Use Path object methods for robust path handling
        src_path = Path(src.split("?")[0])  # Handle potential URLs
        name = src_path.name or "document"  # noqa: F841
        stem = src_path.stem
        ext = "md" if fmt == "markdown" else fmt
        timestamp = int(time.time() * 1000)
        # Ensure temp directory exists
        temp_dir = Path(tempfile.gettempdir())
        temp_dir.mkdir(exist_ok=True)
        return temp_dir / f"{stem}_{timestamp}.{ext}"

    def _get_docling_metadata(self, doc: "DoclingDocument") -> dict[str, Any]:
        """Extract metadata from a Docling document."""
        try:
            num_pages = doc.num_pages() if callable(getattr(doc, "num_pages", None)) else 0
            has_tables = any(
                p.content.has_tables() for p in doc.pages if hasattr(p, "content") and p.content
            )
            has_figures = any(
                p.content.has_figures() for p in doc.pages if hasattr(p, "content") and p.content
            )
            
            # Check for sections by looking for SectionHeaderItem instances in texts
            has_sections = False
            if hasattr(doc, "texts"):
                for item in doc.texts:
                    if hasattr(item, "label") and getattr(item, "label", None) == "section_header":
                        has_sections = True
                        break
            
            return {
                "num_pages": num_pages,
                "has_tables": has_tables,
                "has_figures": has_figures,
                "has_sections": has_sections,
            }
        except Exception as e:
            self.logger.warning(f"Docling metadata collection failed: {e}", exc_info=True)
            return {
                "num_pages": 0,
                "has_tables": False,
                "has_figures": False,
                "has_sections": False,
            }

    def _get_basic_metadata(self, text_content: str, num_pages: int = 0) -> dict[str, Any]:
        """Generate basic metadata for non-Docling content."""
        # Basic heuristics
        has_tables = "| --- |" in text_content or "\t" in text_content  # Very rough check
        has_figures = "![" in text_content  # Check for markdown image syntax
        has_sections = bool(
            re.search(r"^#{1,6}\s+", text_content, re.M)
        )  # Check for markdown headers
        return {
            "num_pages": num_pages,
            "has_tables": has_tables,
            "has_figures": has_figures,
            "has_sections": has_sections,
        }

    @contextmanager
    def _span(self, label: str):
        """Context manager for timing operations."""
        st = time.perf_counter()
        self.logger.debug(f"Starting span: {label}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - st
            self.logger.debug(f"Finished span: {label} ({elapsed:.3f}s)")

    def _json(self, obj: Any) -> str:
        """Utility to serialize objects to JSON."""
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def _hash(self, txt: str) -> str:
        """Generate SHA-1 hash of text."""
        return hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()

    # ───────────────────── Lazy Loading Helpers ─────────────────────────────
    def _lazy_import_tiktoken(self):
        """Lazy import tiktoken."""
        if self._tiktoken_enc_instance is not None:  # Already attempted loading
            return

        if not self._tiktoken_available:
            self.logger.warning(
                "Optional dependency tiktoken not found. Falling back to character-based tokenization."
            )
            self._tiktoken_enc_instance = False  # Mark as unavailable
            return

        try:
            encoding_name = os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
            self.logger.info(f"Lazy-loading tiktoken encoding: {encoding_name}")
            self._tiktoken_enc_instance = tiktoken.get_encoding(encoding_name)  # type: ignore
            self.logger.info("Successfully lazy-loaded tiktoken encoder.")
        except Exception as e:
            self.logger.error(f"Failed to lazy-load tiktoken: {e}", exc_info=True)
            self._tiktoken_enc_instance = False  # Mark as unavailable

    # ───────────────────── HTML Utilities ─────────────────────────────
    def _is_html_fragment(self, text: str) -> bool:
        """Check if text contains likely HTML markup."""
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
                except Exception as e_parse:  # Catch parsing errors specifically
                    last_exception = e_parse
                    self.logger.debug(f"HTML parsing with '{p_name}' failed: {e_parse}")
                    continue
            except Exception as e_general:  # Catch other potential errors during soup creation
                last_exception = e_general
                self.logger.warning(
                    f"Unexpected error creating BeautifulSoup with '{p_name}': {e_general}"
                )
                continue

        if last_exception:
            self.logger.warning(
                f"All standard HTML parsers failed ({last_exception}), attempting fragment parsing."
            )
        wrapped_html = f"<html><body>{html_txt}</body></html>"
        try:
            return BeautifulSoup(wrapped_html, "html.parser"), "html.parser-fragment"
        except Exception as e_frag:
            self.logger.error(
                f"Fragment parsing also failed: {e_frag}. Returning empty soup.", exc_info=True
            )
            return BeautifulSoup("", "html.parser"), "failed"

    def _clean_html(self, html_txt: str) -> Tuple[str, str]:
        """Remove dangerous/pointless elements & attempt structural repair."""
        soup, parser_used = self._best_soup(html_txt)
        if parser_used == "failed":
            self.logger.warning("HTML cleaning skipped due to parsing failure.")
            return html_txt, parser_used  # Return original text if parsing failed

        tags_to_remove = [
            "script",
            "style",
            "svg",
            "iframe",
            "canvas",
            "noscript",
            "meta",
            "link",
            "form",
            "input",
            "button",
            "select",
            "textarea",
            "nav",
            "aside",
            "header",
            "footer",
            "video",
            "audio",
        ]
        for el in soup(tags_to_remove):
            el.decompose()

        unsafe_attrs = [
            "style",
            "onclick",
            "onload",
            "onerror",
            "onmouseover",
            "onmouseout",
            "target",
        ]

        for tag in soup.find_all(True):
            attrs_to_remove = []
            current_attrs = list(tag.attrs.keys())
            for attr in current_attrs:
                attr_val_str = str(tag.get(attr, ""))
                if (
                    attr in unsafe_attrs
                    or attr.startswith("on")
                    or attr.startswith("data-")
                    or (
                        attr == "src"
                        and (
                            "javascript:" in attr_val_str.lower() or "data:" in attr_val_str.lower()
                        )
                    )
                    or (attr == "href" and attr_val_str.lower().startswith("javascript:"))
                ):
                    attrs_to_remove.append(attr)
            for attr in attrs_to_remove:
                if attr in tag.attrs:
                    del tag[attr]
        try:
            text = str(soup)
            text = html.unescape(text)
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = text.strip()
        except Exception as e:
            self.logger.error(
                f"Error during HTML text processing (unescape/regex): {e}", exc_info=True
            )
            try:
                return str(soup), parser_used
            except Exception as stringify_error:
                self.logger.error(f"Could not even stringify soup after error: {stringify_error}")
                return html_txt, parser_used  # Fallback to original text

        return text, parser_used

    # ───────────────────── LLM Helper ─────────────────────────────
    # (Inherited _llm method from BaseTool is sufficient)
    async def _llm(
        self,
        *,
        prompt: str,
        provider: str = Provider.OPENAI.value,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        """Generate text completion using LLM. Returns the generated text content."""
        # Ensure generate_completion is available and callable
        if not callable(generate_completion):
            self.logger.error("LLM generation function 'generate_completion' is not available.")
            raise ToolError(
                "LLM_UNAVAILABLE",
                details={"reason": "generate_completion not available or not callable"},
            )

        # Determine the provider and model
        # Use ANTHROPIC for enhancement tasks by default if provider not specified
        chosen_provider = provider

        try:
            # Prepare additional parameters
            additional_params = extra or {}

            response_dict = await generate_completion(
                prompt=prompt,
                provider=chosen_provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                additional_params=additional_params,
            )

            if isinstance(response_dict, dict):
                llm_content = response_dict.get("text") or response_dict.get("content")
                if llm_content is None:
                    self.logger.error(
                        f"LLM response missing 'text' or 'content' key: {response_dict}"
                    )
                    raise ToolError(
                        "LLM_INVALID_RESPONSE",
                        details={
                            "reason": "Missing content",
                            "response_received": str(response_dict),
                        },
                    )

                if isinstance(llm_content, str):
                    return llm_content.strip()
                else:
                    self.logger.warning(
                        f"LLM response content is not a string: {type(llm_content)}. Converting."
                    )
                    return str(llm_content).strip()
            else:
                self.logger.error(f"LLM response has unexpected format: {response_dict}")
                raise ToolError(
                    "LLM_INVALID_RESPONSE", details={"response_received": str(response_dict)}
                )

        except ProviderError as pe:  # Catch specific provider errors
            self.logger.error(
                f"LLM provider error during call ({chosen_provider}): {pe}", exc_info=True
            )
            raise ToolError(
                "LLM_PROVIDER_ERROR",
                details={"provider": chosen_provider, "error_code": pe.error_code, "error": str(pe)},
            ) from pe
        except ToolError as te:  # Catch other ToolErrors (e.g., LLM_INVALID_RESPONSE)
            raise te
        except Exception as e:
            self.logger.error(f"LLM call failed ({chosen_provider}): {e}", exc_info=True)
            raise ToolError(
                "LLM_CALL_FAILED", details={"provider": chosen_provider, "error": str(e)}
            ) from e

    # ───────────────────── Markdown Processing Utils ─────────────────────
    def _sanitize(self, md: str) -> str:
        """Basic Markdown sanitization."""
        if not md:
            return ""
        md = md.replace("\u00a0", " ")
        md = self._BULLET_RX.sub("- ", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = re.sub(r" +$", "", md, flags=re.MULTILINE)
        md = re.sub(r"^[ \t]+", "", md, flags=re.MULTILINE)
        md = re.sub(r"(^|\n)(#{1,6})([^#\s])", r"\1\2 \3", md)
        md = re.sub(r"```\s*\n", "```\n", md)
        md = re.sub(r"\n\s*```", "\n```", md)
        md = re.sub(r"^[*+]\s", "- ", md, flags=re.MULTILINE)
        md = re.sub(r"^\d+\.\s", lambda m: f"{m.group(0).strip()} ", md, flags=re.MULTILINE)
        return md.strip()

    def _improve(self, md: str) -> str:
        """Apply structural improvements to Markdown text."""
        if not md:
            return ""
        # Ensure blank lines around major block elements
        md = re.sub(
            r"(?<!\n)\n(#{1,6}\s+)", r"\n\n\1", md
        )  # Before heading (if not preceded by blank)
        md = re.sub(
            r"(#{1,6}\s+[^\n]*)\n(?!\n|#|```|>|[-*+]|\d+\.)", r"\1\n\n", md
        )  # After heading (if not followed by blank or block)
        md = re.sub(r"(?<!\n)\n(```)", r"\n\n\1", md)  # Before code fence
        md = re.sub(r"(```)\n(?!\n)", r"\1\n\n", md)  # After code fence
        md = re.sub(r"(?<!\n)\n(> )", r"\n\n\1", md)  # Before blockquote
        md = re.sub(r"(\n> [^\n]*)\n(?!\n|>)", r"\1\n\n", md)  # After blockquote
        md = re.sub(r"(?<!\n)\n(\s*[-*+]\s+|\s*\d+\.\s+)", r"\n\n\1", md)  # Before list start
        md = re.sub(
            r"(\n(\s*[-*+]\s+|\s*\d+\.\s+)[^\n]*)\n(?!\n|\s*([-*+]|\d+\.)\s+)", r"\1\n\n", md
        )  # After list end

        # Collapse excessive newlines again after improvements
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()

    def _convert_html_table_to_markdown(self, table_tag: Tag) -> str:
        """Converts a single BeautifulSoup table Tag to a Markdown string."""
        md_rows = []
        header_cells = table_tag.find_all(["th", "td"], recursive=False)
        if not header_cells:
            first_row = table_tag.find("tr")
            if first_row:
                header_cells = first_row.find_all(["th", "td"])

        if not header_cells:
            self.logger.debug("Table has no header cells identifiable in first row.")
            all_rows_tags = table_tag.find_all("tr")
            if not all_rows_tags:
                return ""
            num_cols = 0
            for r in all_rows_tags:
                num_cols = max(num_cols, len(r.find_all(["th", "td"])))
            if num_cols == 0:
                return ""
            md_rows.append("| " + " | ".join(["Column"] * num_cols) + " |")
            md_rows.append("| " + " | ".join(["---"] * num_cols) + " |")
        else:
            num_cols = len(header_cells)
            hdr = [
                " ".join(c.get_text(" ", strip=True).replace("|", "\\|").split())
                for c in header_cells
            ]
            md_rows.append("| " + " | ".join(hdr) + " |")
            md_rows.append("| " + " | ".join(["---"] * num_cols) + " |")

        body_rows = table_tag.find_all("tr")
        start_row_index = (
            1
            if header_cells and body_rows and header_cells[0].find_parent("tr") == body_rows[0]
            else 0
        )

        for r in body_rows[start_row_index:]:
            cells = r.find_all("td")
            cell_texts = [
                " ".join(c.get_text(" ", strip=True).replace("|", "\\|").split()) for c in cells
            ]
            cell_texts.extend([""] * (num_cols - len(cells)))
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
            return html_txt

        self.logger.debug(f"Found {len(tables)} HTML tables to convert to Markdown.")
        for table_tag in tables:
            try:
                md_table_str = self._convert_html_table_to_markdown(table_tag)
                if md_table_str:
                    placeholder = BeautifulSoup(f"\n\n{md_table_str}\n\n", "html.parser")
                    table_tag.replace_with(placeholder)
                else:
                    table_tag.decompose()
            except Exception as e:
                self.logger.error(f"Failed to convert a table to Markdown: {e}", exc_info=True)
                # table_tag.decompose() # Optionally remove table on error

        return str(soup)

    # ───────────────────── OCR & Image Helpers (Integrated) ───────────────────

    def _ocr_validate_file_path(
        self, file_path: str, expected_extension: Optional[str] = None
    ) -> Path:
        """Validates a file path exists and optionally has the expected extension."""
        if not file_path:
            raise ToolInputError("File path cannot be empty", param_name="file_path")

        path = Path(os.path.expanduser(os.path.normpath(file_path)))

        if not path.exists():
            raise ToolInputError(f"File not found: {path}", param_name="file_path")
        if not path.is_file():
            raise ToolInputError(f"Path is not a file: {path}", param_name="file_path")
        if expected_extension and not path.suffix.lower() == expected_extension.lower():
            raise ToolInputError(
                f"File does not have the expected extension ({expected_extension}): {path}",
                param_name="file_path",
            )
        return path

    def _ocr_get_task_type(self, extraction_method: str = "hybrid") -> str:
        """Returns the appropriate TaskType for OCR operations."""
        if extraction_method == "direct_text":
            return TaskType.TEXT_EXTRACTION.value
        # For OCR or Hybrid, consider it primarily an OCR task
        return TaskType.OCR.value

    def _ocr_check_dep(self, dep_name: str, is_available: bool, feature: str):
        """Checks if a required dependency is available, raising ToolError if not."""
        if not is_available:
            self.logger.error(f"Missing required dependency '{dep_name}' for feature '{feature}'.")
            raise ToolError(
                "DEPENDENCY_MISSING", details={"dependency": dep_name, "feature": feature}
            )

    def _ocr_preprocess_image(
        self, image: "PILImage.Image", preprocessing_options: Optional[Dict[str, Any]] = None
    ) -> "PILImage.Image":
        """Preprocesses an image for better OCR results."""
        if not self._pil_available:
            self.logger.warning("Pillow (PIL) not available. Skipping preprocessing.")
            return image  # Cannot proceed without PIL

        if not self._cv2_available or not self._numpy_available:
            self.logger.warning(
                "OpenCV or NumPy not available. Some preprocessing steps will be skipped."
            )
            # Continue with PIL-based enhancements if possible

        prep_opts = {  # Default options
            "denoise": True,
            "threshold": "otsu",
            "deskew": True,
            "enhance_contrast": True,
            "enhance_brightness": False,
            "enhance_sharpness": False,
            "apply_filters": [],
            "resize_factor": 1.0,
            **(preprocessing_options or {}),  # Merge user options
        }
        self.logger.debug(f"Applying preprocessing with options: {prep_opts}")

        # --- PIL Enhancements ---
        if self._pil_available:
            img_pil = image.copy()  # Work on a copy
            if prep_opts.get("enhance_brightness"):
                img_pil = ImageEnhance.Brightness(img_pil).enhance(1.3)
            if prep_opts.get("enhance_contrast"):
                img_pil = ImageEnhance.Contrast(img_pil).enhance(1.4)
            if prep_opts.get("enhance_sharpness"):
                img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.5)
            filters = prep_opts.get("apply_filters", [])
            for filter_name in filters:
                try:
                    if filter_name == "unsharp_mask":
                        img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
                    elif filter_name == "detail":
                        img_pil = img_pil.filter(ImageFilter.DETAIL)
                    elif filter_name == "edge_enhance":
                        img_pil = img_pil.filter(ImageFilter.EDGE_ENHANCE)
                    elif filter_name == "smooth":
                        img_pil = img_pil.filter(ImageFilter.SMOOTH)
                    else:
                        self.logger.warning(f"Unknown PIL filter requested: {filter_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to apply PIL filter '{filter_name}': {e}")
        else:
            img_pil = image  # Use original if PIL failed

        # --- OpenCV Processing ---
        if not self._cv2_available or not self._numpy_available:
            self.logger.warning("Skipping OpenCV preprocessing steps.")
            return img_pil  # Return PIL-enhanced image

        try:
            img_cv = np.array(img_pil)  # type: ignore
            # Ensure grayscale
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)  # type: ignore
            elif len(img_cv.shape) == 3 and img_cv.shape[2] == 4:  # Handle RGBA
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2GRAY)  # type: ignore
            else:
                gray = img_cv  # Assume already grayscale or single channel

            original_height, original_width = gray.shape[:2]
            processed_img = gray

            # Adaptive scaling
            resize_factor = prep_opts.get("resize_factor", 1.0)
            if resize_factor == 1.0:
                longest_edge = max(original_width, original_height)
                if longest_edge < 1500 and longest_edge > 0:
                    resize_factor = math.ceil(1500 / longest_edge * 10) / 10
                elif longest_edge > 3500:
                    resize_factor = math.floor(3500 / longest_edge * 10) / 10

            # Enhance contrast (using histogram equalization) - careful, can sometimes harm OCR
            if prep_opts.get("enhance_contrast", True):  # Re-evaluating if PIL contrast is enough
                processed_img = cv2.equalizeHist(processed_img)

            # Thresholding
            threshold_method = prep_opts.get("threshold", "otsu")
            if threshold_method == "otsu":
                _, processed_img = cv2.threshold(
                    processed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )  # type: ignore
            elif threshold_method == "adaptive":
                block_size = math.floor(min(processed_img.shape) / 30)
                block_size = max(3, block_size) | 1  # Ensure odd
                processed_img = cv2.adaptiveThreshold(
                    processed_img,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    2,
                )  # type: ignore
            # Else: no thresholding

            # Denoising
            if prep_opts.get("denoise", True):
                h_param = math.ceil(
                    10 * math.log10(max(10, min(original_width, original_height)))
                )  # Avoid log(0)
                processed_img = cv2.fastNlMeansDenoising(processed_img, None, h_param, 7, 21)  # type: ignore

            # Deskewing
            if prep_opts.get("deskew", True):
                try:
                    # Use inverted image for finding text blocks
                    coords = np.column_stack(
                        np.where(processed_img < 128)
                    )  # Find dark pixels (text)
                    if coords.size > 0:  # Check if any text pixels found
                        angle = cv2.minAreaRect(coords)[-1]  # type: ignore
                        if angle < -45:
                            angle = -(90 + angle)
                        else:
                            angle = -angle

                        if abs(angle) > 0.5:  # Only rotate if skew is significant
                            (h, w) = processed_img.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)  # type: ignore
                            processed_img = cv2.warpAffine(
                                processed_img,
                                M,
                                (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE,
                            )  # type: ignore
                            self.logger.debug(f"Deskewed image by {angle:.2f} degrees.")
                    else:
                        self.logger.debug("No significant content found for deskewing.")
                except Exception as e_deskew:
                    self.logger.warning(f"Deskewing failed: {e_deskew}. Using non-deskewed image.")

            # Resizing
            if resize_factor != 1.0:
                new_w = math.ceil(original_width * resize_factor)
                new_h = math.ceil(original_height * resize_factor)
                processed_img = cv2.resize(
                    processed_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC
                )  # type: ignore
                self.logger.debug(f"Resized image by factor {resize_factor:.2f} to {new_w}x{new_h}")

            # Convert back to PIL Image
            final_pil_image = Image.fromarray(processed_img)  # type: ignore
            return final_pil_image

        except Exception as e_cv:
            self.logger.error(f"OpenCV preprocessing failed: {e_cv}", exc_info=True)
            return img_pil  # Fallback to PIL-processed image

    def _ocr_run_tesseract(
        self, image: "PILImage.Image", ocr_language: str = "eng", ocr_config: str = ""
    ) -> str:
        """Extracts text from an image using Tesseract OCR."""
        self._ocr_check_dep("pytesseract", self._pytesseract_available, "OCR Text Extraction")
        self._ocr_check_dep("Pillow", self._pil_available, "OCR Text Extraction")

        try:
            custom_config = f"-l {ocr_language} {ocr_config}"
            self.logger.debug(f"Running Tesseract with config: {custom_config}")
            with self._span(f"pytesseract_ocr_{ocr_language}"):
                text = pytesseract.image_to_string(image, config=custom_config)  # type: ignore
            self.logger.debug(f"Tesseract extracted {len(text)} characters.")
            return text
        except pytesseract.TesseractNotFoundError as e:  # type: ignore
            self.logger.error(
                "Tesseract executable not found or not in PATH. Please install Tesseract."
            )
            raise ToolError(
                "DEPENDENCY_MISSING",
                details={"dependency": "Tesseract OCR Engine", "feature": "OCR Text Extraction"},
            ) from e
        except Exception as e:
            self.logger.error(f"Tesseract OCR extraction failed: {e}", exc_info=True)
            raise ToolError("OCR_FAILED", details={"engine": "Tesseract", "error": str(e)}) from e

    def _ocr_extract_text_from_pdf_direct(
        self, file_path: Path, start_page: int = 0, max_pages: int = 0
    ) -> Tuple[List[str], bool]:
        """Extracts text directly from a PDF file using PyMuPDF or PDFPlumber."""
        texts: List[str] = []
        has_text = False
        min_meaningful_chars_per_page = 50  # Heuristic threshold

        if self._pymupdf_available:
            self.logger.debug(f"Attempting direct text extraction with PyMuPDF for {file_path}")
            try:
                with pymupdf.open(file_path) as doc:  # type: ignore
                    total_pages = len(doc)
                    end_page = (
                        total_pages if max_pages <= 0 else min(start_page + max_pages, total_pages)
                    )
                    num_extracted = 0
                    for i in range(start_page, end_page):
                        try:
                            page = doc[i]
                            text = page.get_text() or ""
                            if len(text.strip()) >= min_meaningful_chars_per_page:
                                has_text = True
                            texts.append(text)
                            num_extracted += 1
                        except Exception as e_page:
                            self.logger.warning(
                                f"PyMuPDF: Error extracting text from page {i + 1}: {e_page}"
                            )
                            texts.append("")  # Append empty string for consistency
                    self.logger.debug(
                        f"PyMuPDF extracted text from {num_extracted} pages. Found meaningful text: {has_text}"
                    )
                    return texts, has_text
            except Exception as e_pymupdf:
                self.logger.warning(
                    f"PyMuPDF direct text extraction failed: {e_pymupdf}. Trying PDFPlumber..."
                )
                # Fall through to try pdfplumber if available

        if self._pdfplumber_available:
            self.logger.debug(f"Attempting direct text extraction with PDFPlumber for {file_path}")
            try:
                with pdfplumber.open(file_path) as pdf:  # type: ignore
                    total_pages = len(pdf.pages)
                    end_page = (
                        total_pages if max_pages <= 0 else min(start_page + max_pages, total_pages)
                    )
                    num_extracted = 0
                    for i in range(start_page, end_page):
                        try:
                            page = pdf.pages[i]
                            # Use more tolerant settings for slightly better layout preservation
                            text = (
                                page.extract_text(
                                    x_tolerance=2, y_tolerance=2, keep_blank_chars=True
                                )
                                or ""
                            )
                            if len(text.strip()) >= min_meaningful_chars_per_page:
                                has_text = True
                            texts.append(text)
                            num_extracted += 1
                        except Exception as e_page:
                            self.logger.warning(
                                f"PDFPlumber: Error extracting text from page {i + 1}: {e_page}"
                            )
                            texts.append("")
                    self.logger.debug(
                        f"PDFPlumber extracted text from {num_extracted} pages. Found meaningful text: {has_text}"
                    )
                    return texts, has_text
            except Exception as e_pdfplumber:
                self.logger.error(
                    f"PDFPlumber direct text extraction failed: {e_pdfplumber}", exc_info=True
                )
                raise ToolError(
                    "DIRECT_EXTRACTION_FAILED",
                    details={
                        "reason": "Both PyMuPDF (if tried) and PDFPlumber failed",
                        "error": str(e_pdfplumber),
                    },
                ) from e_pdfplumber
        else:
            # Neither library is available
            self.logger.error(
                "No direct PDF text extraction library (PyMuPDF or PDFPlumber) is available."
            )
            raise ToolError(
                "DEPENDENCY_MISSING",
                details={
                    "dependency": "PyMuPDF or PDFPlumber",
                    "feature": "Direct PDF Text Extraction",
                },
            )

    def _ocr_convert_pdf_to_images(
        self, file_path: Path, start_page: int = 0, max_pages: int = 0, dpi: int = 300
    ) -> List["PILImage.Image"]:
        """Converts pages of a PDF file path to PIL Image objects."""
        self._ocr_check_dep("pdf2image", self._pdf2image_available, "PDF to Image Conversion")
        self._ocr_check_dep("Pillow", self._pil_available, "PDF to Image Conversion")

        try:
            # pdf2image uses 1-based indexing
            first_page = start_page + 1
            last_page = None if max_pages <= 0 else first_page + max_pages - 1
            self.logger.debug(
                f"Converting PDF {file_path} to images (pages {first_page}-{last_page or 'end'}, dpi={dpi})"
            )

            with self._span(f"pdf2image_convert_path_p{first_page}-{last_page or 'end'}"):
                # Using context manager for temporary directory recommended by pdf2image
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_path(  # type: ignore
                        file_path,
                        dpi=dpi,
                        first_page=first_page,
                        last_page=last_page,
                        output_folder=temp_dir,
                        fmt="png",  # Use PNG for potentially better quality than default ppm
                        thread_count=max(1, os.cpu_count() // 2),  # Use multiple threads
                    )
            self.logger.info(f"Successfully converted {len(images)} pages from PDF to images.")
            return images  # type: ignore
        except Exception as e:
            # Catch specific pdf2image errors if possible, TBD
            self.logger.error(
                f"PDF to image conversion failed for path {file_path}: {e}", exc_info=True
            )
            raise ToolError(
                "PDF_CONVERSION_FAILED", details={"reason": "pdf2image failed", "error": str(e)}
            ) from e

    def _ocr_convert_pdf_bytes_to_images(
        self, pdf_bytes: bytes, start_page: int = 0, max_pages: int = 0, dpi: int = 300
    ) -> List["PILImage.Image"]:
        """Converts pages of PDF bytes to PIL Image objects."""
        self._ocr_check_dep("pdf2image", self._pdf2image_available, "PDF Bytes to Image Conversion")
        self._ocr_check_dep("Pillow", self._pil_available, "PDF Bytes to Image Conversion")

        try:
            first_page = start_page + 1
            last_page = None if max_pages <= 0 else first_page + max_pages - 1
            self.logger.debug(
                f"Converting PDF bytes to images (pages {first_page}-{last_page or 'end'}, dpi={dpi})"
            )

            with self._span(f"pdf2image_convert_bytes_p{first_page}-{last_page or 'end'}"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_bytes(  # type: ignore
                        pdf_bytes,
                        dpi=dpi,
                        first_page=first_page,
                        last_page=last_page,
                        output_folder=temp_dir,
                        fmt="png",
                        thread_count=max(1, os.cpu_count() // 2),
                    )
            self.logger.info(
                f"Successfully converted {len(images)} pages from PDF bytes to images."
            )
            return images  # type: ignore
        except Exception as e:
            self.logger.error(f"PDF bytes to image conversion failed: {e}", exc_info=True)
            raise ToolError(
                "PDF_CONVERSION_FAILED",
                details={"reason": "pdf2image failed for bytes", "error": str(e)},
            ) from e

    def _ocr_generate_cache_key(self, data: Any, prefix="ocr_cache", **kwargs) -> str:
        """Generate a cache key including relevant parameters."""
        key_parts = [prefix]
        if isinstance(data, Path):
            try:
                stat = data.stat()
                key_parts.append(f"path={data.resolve()}:{stat.st_mtime}:{stat.st_size}")
            except Exception:
                key_parts.append(f"path={str(data)}")  # Fallback if stat fails
        elif isinstance(data, bytes):
            key_parts.append(f"bytes_sha1={hashlib.sha1(data).hexdigest()}")
        elif isinstance(data, str) and len(data) > 100:  # Hash long strings
            key_parts.append(f"str_sha1={self._hash(data)}")
        else:
            key_parts.append(f"data={str(data)}")

        # Add sorted kwargs to ensure consistent key order
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={str(v)}")

        combined_key = "_".join(key_parts)
        # Hash the combined key if it's too long for some cache backends
        if len(combined_key) > 200:
            return f"{prefix}_hash_{self._hash(combined_key)}"
        return combined_key

    def _ocr_split_text_into_chunks(
        self, text: str, max_chunk_size: int = 8000, overlap: int = 200
    ) -> List[str]:
        """Splits text into chunks, trying to respect paragraphs and sentences."""
        if not text:
            return []

        max_chunk_size = max(1000, min(max_chunk_size, 15000))  # Sensible limits
        overlap = max(50, min(overlap, max_chunk_size // 4))
        min_chunk_size = overlap * 2  # Ensure chunks are larger than overlap

        chunks = []
        start_index = 0
        text_len = len(text)

        while start_index < text_len:
            end_index = min(start_index + max_chunk_size, text_len)

            # If we are near the end, just take the rest
            if end_index == text_len:
                chunk = text[start_index:end_index]
                if len(chunk.strip()) > min_chunk_size // 2:  # Add last chunk if not tiny
                    chunks.append(chunk)
                break

            # Find best split point backward from end_index
            best_split_index = -1
            # Try paragraph breaks first (\n\n)
            split_point = text.rfind("\n\n", max(start_index, end_index - overlap * 2), end_index)
            if split_point != -1 and split_point > start_index + min_chunk_size:
                best_split_index = split_point + 2  # Split after the double newline
            else:
                # Try sentence breaks (.?!) followed by space/newline
                sentence_match = list(
                    re.finditer(
                        r"[.?!](\s|\n)", text[max(start_index, end_index - overlap) : end_index]
                    )
                )
                if sentence_match:
                    split_offset = sentence_match[-1].end()
                    split_point = max(start_index, end_index - overlap) + split_offset
                    if split_point > start_index + min_chunk_size:
                        best_split_index = split_point

            # If no good sentence/paragraph break found, try newline or space
            if best_split_index == -1:
                split_point_newline = text.rfind(
                    "\n", max(start_index, end_index - overlap), end_index
                )
                split_point_space = text.rfind(
                    " ", max(start_index, end_index - overlap), end_index
                )
                split_point = max(split_point_newline, split_point_space)
                if split_point != -1 and split_point > start_index + min_chunk_size:
                    best_split_index = split_point + 1  # Split after space/newline

            # If still no good split, force split at end_index
            if best_split_index == -1:
                best_split_index = end_index

            # Extract chunk and add
            chunk = text[start_index:best_split_index]
            if len(chunk.strip()) > 0:
                chunks.append(chunk)

            # Calculate next start index with overlap
            next_start = max(start_index + 1, best_split_index - overlap)
            # Ensure next start isn't before the current chunk's true start if overlap is large
            next_start = max(
                start_index + min_chunk_size // 2, next_start
            )  # Guarantee forward progress
            # Don't re-process exactly the same content if split point was bad
            if next_start <= start_index:
                next_start = best_split_index

            start_index = next_start

        self.logger.debug(
            f"Split text ({text_len} chars) into {len(chunks)} chunks (max_size={max_chunk_size}, overlap={overlap})"
        )
        return chunks

    def _ocr_detect_tables(self, image: "PILImage.Image") -> List[Tuple[int, int, int, int]]:
        """Detects potential tables in an image using OpenCV."""
        if not self._cv2_available or not self._numpy_available or not self._pil_available:
            self.logger.warning("Cannot detect tables: OpenCV, NumPy, or Pillow not available.")
            return []

        try:
            img = np.array(image)  # type: ignore
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
            else:
                gray = img

            # Adaptive thresholding seems better for table line detection
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2
            )  # type: ignore

            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # type: ignore
            detected_horizontal = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
            )  # type: ignore
            cnts_h = cv2.findContours(
                detected_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )  # type: ignore
            cnts_h = cnts_h[0] if len(cnts_h) == 2 else cnts_h[1]

            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))  # type: ignore
            detected_vertical = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
            )  # type: ignore
            cnts_v = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore
            cnts_v = cnts_v[0] if len(cnts_v) == 2 else cnts_v[1]

            # Combine contours or find bounding boxes of line intersections
            # A simpler heuristic: look for large rectangular contours in the original threshold
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore

            table_regions = []
            img_area = img.shape[0] * img.shape[1]
            min_table_area = img_area * 0.03  # Minimum 3% of image area

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)  # type: ignore
                area = w * h
                aspect_ratio = w / max(1, h)

                # Filter based on size and aspect ratio (tables are often wider or squarish)
                if area > min_table_area and 0.3 <= aspect_ratio <= 5.0:
                    # Further check: does this region contain significant horizontal/vertical lines?
                    roi_h = detected_horizontal[y : y + h, x : x + w]
                    roi_v = detected_vertical[y : y + h, x : x + w]
                    if cv2.countNonZero(roi_h) > w * 0.5 and cv2.countNonZero(roi_v) > h * 0.5:  # type: ignore
                        table_regions.append((x, y, w, h))

            # Optional: Merge overlapping regions detected
            # ... (implementation omitted for brevity)

            self.logger.debug(f"Detected {len(table_regions)} potential table regions.")
            return table_regions
        except Exception as e:
            self.logger.error(f"Table detection failed: {e}", exc_info=True)
            return []

    def _ocr_crop_image(
        self, image: "PILImage.Image", region: Tuple[int, int, int, int]
    ) -> "PILImage.Image":
        """Crops an image to the specified region (x, y, width, height)."""
        self._ocr_check_dep("Pillow", self._pil_available, "Image Cropping")
        x, y, w, h = region
        return image.crop((x, y, x + w, y + h))

    def _ocr_is_text_mostly_noise(self, text: str, noise_threshold: float = 0.4) -> bool:
        """Determine if extracted text is mostly noise based on character distribution."""
        if not text or len(text) < 20:
            return False

        total_chars = len(text)
        # Count alphanumeric, whitespace, and common punctuation
        valid_chars = sum(
            1 for c in text if c.isalnum() or c.isspace() or c in ".,;:!?\"'()[]{}%/$£€-*+=<>@#&"
        )
        noise_ratio = 1.0 - (valid_chars / total_chars)

        is_noise = noise_ratio > noise_threshold
        if is_noise:
            self.logger.debug(
                f"Text flagged as noisy (Ratio: {noise_ratio:.2f} > {noise_threshold}): '{text[:100]}...'"
            )
        return is_noise

    def _ocr_is_likely_header_or_footer(self, text: str, line_length_threshold: int = 60) -> bool:
        """Determine if a text line is likely a header or footer (improved heuristics)."""
        text = text.strip()
        if not text or len(text) > line_length_threshold:
            return False

        # Contains page number (possibly with Page/Seite/etc.)
        if re.search(r"(page|seite|p\.?|s\.?)\s*\d+", text, re.I):
            return True
        if re.match(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", text):
            return True  # Just a number, possibly bracketed

        # Contains date patterns (allow variations)
        if re.search(
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}",
            text,
            re.I,
        ):
            return True
        if re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text):
            return True

        # Common keywords
        if re.search(r"^(confidential|internal use|draft|proprietary)", text, re.I):
            return True
        if re.search(r"^(copyright|\(c\)|©)", text, re.I):
            return True

        # Repeated characters (like --- or === used as separators)
        if len(set(text)) <= 3 and len(text) > 5:
            return True

        # Check if text is identical to lines at the very top/bottom of other pages (needs context, harder here)

        return False

    def _ocr_remove_headers_and_footers(self, text: str, max_lines_check: int = 5) -> str:
        """Removes likely headers and footers from the top/bottom of the text."""
        if not text:
            return text
        lines = text.splitlines()
        if len(lines) < max_lines_check * 2:
            return text  # Too short to reliably detect

        lines_to_remove_indices = set()

        # Check top lines
        for i in range(max_lines_check):
            if i < len(lines) and self._ocr_is_likely_header_or_footer(lines[i]):
                lines_to_remove_indices.add(i)
            else:
                # Stop checking top if a non-header/footer line is found early
                if i < max_lines_check // 2:
                    break

        # Check bottom lines
        for i in range(max_lines_check):
            idx = len(lines) - 1 - i
            if idx >= 0 and self._ocr_is_likely_header_or_footer(lines[idx]):
                # Avoid removing if it's the same index as a removed top line
                if idx not in lines_to_remove_indices:
                    lines_to_remove_indices.add(idx)
            else:
                if i < max_lines_check // 2:
                    break

        if not lines_to_remove_indices:
            return text

        self.logger.debug(f"Removing {len(lines_to_remove_indices)} potential header/footer lines.")
        result_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove_indices]

        # Remove leading/trailing blank lines potentially left after removal
        cleaned_text = "\n".join(result_lines).strip()
        return cleaned_text

    async def _ocr_enhance_text_chunk(
        self, chunk: str, output_format: str = "markdown", remove_headers: bool = False
    ) -> str:
        """Enhances OCR text using LLM to correct errors and improve formatting."""
        # --- Apply Basic Rule-based Cleaning First ---
        cleaned_text = chunk.strip()  # Work on a copy

        # Apply basic text cleanup rules
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Normalize multiple whitespace
        cleaned_text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", cleaned_text)
        # Normalize whitespace before LLM
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)  # Collapse multiple blank lines

        # Optional header/footer removal using rules *before* LLM
        if remove_headers:
            original_len = len(cleaned_text)
            cleaned_text = self._ocr_remove_headers_and_footers(cleaned_text)
            if len(cleaned_text) < original_len:
                self.logger.debug("Applied rule-based header/footer removal pre-LLM.")

        # Check for noise after initial cleaning
        if self._ocr_is_text_mostly_noise(cleaned_text):
            self.logger.warning(
                "Text chunk appears noisy after basic cleaning, LLM enhancement might be less effective."
            )
            # Return cleaned text directly? Or still try LLM? Let's still try LLM.

        # --- LLM Prompt Generation ---
        format_instruction = ""
        if output_format == "markdown":
            format_instruction = """
2. Format as clean, readable markdown:
   - Use appropriate heading levels (#, ##, etc.).
   - Format lists correctly (bulleted or numbered).
   - Apply emphasis (*italic*) and strong (**bold**) sparingly where appropriate.
   - Represent tabular data using markdown table syntax.
   - Use code blocks (```) for code snippets or equations if detected."""
        else:  # output_format == "text"
            format_instruction = """
2. Format as clean, readable plain text:
   - Ensure clear paragraph separation (double newline).
   - Maintain list structures with standard markers (e.g., -, 1.).
   - Avoid markdown syntax."""

        header_footer_instruction = (
            "Remove headers, footers, and page numbers."
            if remove_headers
            else "Preserve all content including potential headers/footers."
        )

        prompt = f"""You are an expert text processor specialized in correcting OCR errors. Please process the following text according to these instructions:

1. Fix OCR-induced errors:
   - Correct typos (e.g., 'rn' vs 'm', 'O' vs '0', 'l' vs '1').
   - Join words incorrectly split across lines.
   - Merge paragraphs that were artificially split.
   - Use context to resolve ambiguities and reconstruct the original meaning accurately.
{format_instruction}
3. Clean up formatting:
   - Remove redundant spaces and unnecessary line breaks within paragraphs.
   - Ensure consistent paragraph spacing.
   - {header_footer_instruction}

4. IMPORTANT: Preserve all meaningful content and the original structure as much as possible. Do not add information or summaries. Do not change the substance of the text.

Input Text:
```text
{cleaned_text}
```

Corrected Output ({output_format}):"""  # Ensure LLM knows the expected format tag

        try:
            self.logger.debug(
                f"Sending chunk (len={len(cleaned_text)}) to LLM for enhancement (format={output_format}, rm_hdrs={remove_headers})."
            )
            # Use a capable model like Claude or GPT-4 for best results
            provider = Provider.OPENAI.value 
            model = 'gpt-4.1-mini'

            # Estimate max tokens: base length + buffer (e.g., 30% for markdown, 10% for text)
            buffer_factor = 1.3 if output_format == "markdown" else 1.1
            # Estimate token count (rough approximation: 1 token ~ 3-4 chars)
            estimated_input_tokens = len(cleaned_text) // 3
            llm_max_tokens = (
                int(estimated_input_tokens * buffer_factor) + 500
            )  # Add fixed buffer too
            llm_max_tokens = max(1000, llm_max_tokens)  # Ensure minimum reasonable size
            # Cap max_tokens to avoid excessive cost/limits
            llm_max_tokens = min(llm_max_tokens, 15000)  # Adjust cap as needed

            enhanced_text = await self._llm(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=0.15,  # Low temperature for factual correction
                max_tokens=llm_max_tokens,
            )

            # --- Post-processing LLM Output ---
            # Remove potential LLM preamble/apologies
            enhanced_text = re.sub(
                r"^(Okay, |Here is |Sure, |Here\'s |Certainly, )[^:\n]*[:\n]\s*",
                "",
                enhanced_text,
                flags=re.IGNORECASE,
            )
            # Remove trailing explanations or markdown fences if they exist
            enhanced_text = re.sub(r"\n*```$", "", enhanced_text).strip()

            self.logger.debug(f"LLM enhancement returned text (len={len(enhanced_text)}).")
            return enhanced_text

        except ToolError as e:
            # If LLM call fails specifically, log and return the rule-based cleaned text
            self.logger.error(
                f"LLM text enhancement failed: {e.error_code} - {str(e)}. Returning pre-LLM cleaned text."
            )
            return cleaned_text  # Fallback
        except Exception as e:
            self.logger.error(f"Unexpected error during LLM text enhancement: {e}", exc_info=True)
            return cleaned_text  # Fallback

    async def _ocr_assess_text_quality(
        self, original_text: str, enhanced_text: str
    ) -> Dict[str, Any]:
        """Assesses the quality of OCR enhancement using LLM (from OCR script)."""
        if not original_text or not enhanced_text:
            return {"score": 0, "explanation": "Input text missing for assessment.", "examples": []}

        max_sample = 4000  # Limit context size for assessment prompt
        original_sample = original_text[:max_sample]
        enhanced_sample = enhanced_text[:max_sample]
        if len(original_text) > max_sample:
            original_sample += "\n... (truncated)"
        if len(enhanced_text) > max_sample:
            enhanced_sample += "\n... (truncated)"

        prompt = f"""Please assess the quality improvement from the 'Original OCR Text' to the 'Enhanced Text'. Focus on:
1. Correction of OCR errors (typos, spacing, broken words).
2. Improvement in formatting and readability (paragraphs, lists, structure).
3. Accuracy in preserving the original meaning and content.
4. Effectiveness of removing noise (like headers/footers if applicable).

Original OCR Text:
```
{original_sample}
```

Enhanced Text:
```
{enhanced_sample}
```

Provide your assessment ONLY in the following JSON format:
{{
  "score": <integer score 0-100, where 100 is perfect enhancement>,
  "explanation": "<brief explanation of the score, highlighting key improvements or remaining issues>",
  "examples": [
    "<example 1 of a specific correction or improvement>",
    "<example 2>",
    "<example 3 (optional)>"
  ]
}}
Do not add any text before or after the JSON object.
"""

        try:
            self.logger.debug("Requesting LLM quality assessment.")
            assessment_json_str = await self._llm(
                prompt=prompt,
                max_tokens=500,  # Assessment shouldn't be too long
            )

            # Attempt to parse the response as JSON
            try:
                # Clean potential markdown fences
                json_match = re.search(r"```(?:json)?\s*([\s\S]+)\s*```", assessment_json_str)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    json_str = assessment_json_str.strip()

                # Find first '{' and last '}'
                start_brace = json_str.find("{")
                end_brace = json_str.rfind("}")
                if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
                    json_str = json_str[start_brace : end_brace + 1]
                else:
                    raise ValueError("Could not find JSON object boundaries.")

                assessment_data = json.loads(json_str)

                # Validate structure
                if (
                    not isinstance(assessment_data, dict)
                    or "score" not in assessment_data
                    or "explanation" not in assessment_data
                    or "examples" not in assessment_data
                    or not isinstance(assessment_data["examples"], list)
                ):
                    raise ValueError("Parsed JSON has incorrect structure.")

                # Basic type validation
                assessment_data["score"] = int(assessment_data["score"])
                assessment_data["explanation"] = str(assessment_data["explanation"])
                assessment_data["examples"] = [str(ex) for ex in assessment_data["examples"]]

                self.logger.debug(f"Quality assessment received: Score {assessment_data['score']}")
                return assessment_data

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self.logger.error(
                    f"Failed to parse quality assessment JSON: {e}. Raw response:\n{assessment_json_str}"
                )
                return {
                    "score": None,
                    "explanation": f"Failed to parse LLM assessment response: {e}",
                    "examples": [],
                }

        except Exception as e:
            self.logger.error(f"Error during LLM quality assessment call: {e}", exc_info=True)
            return {
                "score": None,
                "explanation": f"Failed to get quality assessment from LLM: {e}",
                "examples": [],
            }

    async def _ocr_format_tables_in_text(self, text: str) -> str:
        """Detects and formats potential tables in text using markdown via LLM."""
        # This is a placeholder. A robust implementation would involve:
        # 1. Identifying potential table blocks (regex, layout analysis if available)
        # 2. Sending *only* those blocks to an LLM with a table formatting prompt.
        # 3. Replacing the original block with the formatted table.
        # For now, we assume the main enhancement chunk handled basic table formatting
        # if 'reformat_as_markdown' was True. A dedicated table tool might be better.
        self.logger.debug(
            "Table formatting within enhance_ocr_text is currently basic (relies on main LLM pass)."
        )
        # Simple check if markdown table syntax exists
        if "|\n| ---" in text:
            self.logger.debug("Markdown table syntax already detected in text.")
        return text

    async def _ocr_enhance_table_formatting(self, table_text: str):
        """Enhances formatting of a single table using LLM."""
        # (Helper intended for _ocr_format_tables_in_text, currently unused)
        prompt = f"""Format the following text block, which likely represents a table, into a clean markdown table. Preserve all data accurately.

Table Text:
```
{table_text}
```

Markdown Table:"""
        try:
            formatted_table = await self._llm(
                prompt=prompt,
                max_tokens=len(table_text) + 500,
            )
            # Basic check for markdown table structure
            if "|" in formatted_table and "\n| ---" in formatted_table:
                return "\n" + formatted_table.strip() + "\n"
            else:
                self.logger.warning(
                    "LLM did not return a valid markdown table format for the block."
                )
                return table_text  # Return original if LLM fails to format
        except Exception as e:
            self.logger.error(f"Error enhancing table format: {e}", exc_info=True)
            return table_text

    def _ocr_process_toc(self, toc: List) -> List[Dict[str, Any]]:
        """Processes a PDF table of contents (from PyMuPDF) into a nested structure."""
        if not toc:
            return []
        result = []
        stack = [(-1, result)]  # (level, children_list)
        for item in toc:
            level, title, page = item
            while stack[-1][0] >= level:
                stack.pop()
            entry = {"title": title, "page": page, "children": []}
            stack[-1][1].append(entry)
            stack.append((level, entry["children"]))
        return result

    # --- Fallback Conversion Helpers (Original) ---

    async def _fallback_convert_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Basic PDF conversion using PyPDF2 when docling is not available."""
        self._ocr_check_dep("PyPDF2", self._pypdf2_available, "Basic PDF Fallback Conversion")
        try:
            self.logger.info(f"Using PyPDF2 fallback for PDF: {file_path}")
            content = ""
            metadata = {
                "num_pages": 0,
                "has_tables": False,
                "has_figures": False,
                "has_sections": False,
                "is_fallback": True,
            }
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)  # type: ignore
                num_pages = len(reader.pages)
                metadata["num_pages"] = num_pages
                pages = [reader.pages[i].extract_text() or "" for i in range(num_pages)]
                content = "\n\n".join(pages)  # Add double newline as page separator
            metadata.update(self._get_basic_metadata(content, num_pages))
            return {"content": content, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"PyPDF2 fallback failed: {e}", exc_info=True)
            raise ToolError(
                "CONVERSION_FAILED",
                details={"file": str(file_path), "method": "PyPDF2 Fallback", "error": str(e)},
            ) from e

    async def _fallback_convert_docx(self, file_path: Path) -> Dict[str, Any]:
        """Basic DOCX conversion using python-docx when docling is not available."""
        self._ocr_check_dep("python-docx", self._docx_available, "DOCX Fallback Conversion")
        try:
            self.logger.info(f"Using python-docx fallback for DOCX: {file_path}")
            doc = docx.Document(file_path)  # type: ignore
            paragraphs = [para.text for para in doc.paragraphs]
            content = "\n".join(paragraphs)
            # Basic metadata
            metadata = {
                "num_pages": 0,  # Page count hard with python-docx
                "has_tables": len(doc.tables) > 0,
                "has_figures": False,  # Cannot easily detect figures
                "has_sections": len(doc.sections) > 0,
                "is_fallback": True,
            }
            metadata.update(self._get_basic_metadata(content))  # Add heuristic checks
            return {"content": content, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"python-docx fallback failed: {e}", exc_info=True)
            raise ToolError(
                "CONVERSION_FAILED",
                details={"file": str(file_path), "method": "python-docx Fallback", "error": str(e)},
            ) from e

    async def _fallback_convert_text(self, file_path: Path) -> Dict[str, Any]:
        """Simple text file reading."""
        try:
            self.logger.info(f"Reading text file directly: {file_path}")
            content = file_path.read_text(encoding="utf-8", errors="replace")
            line_count = content.count("\n") + 1
            page_estimate = max(1, int(line_count / 50))  # Rough estimate
            metadata = {"num_pages": page_estimate, "is_fallback": True}
            metadata.update(self._get_basic_metadata(content, page_estimate))
            return {"content": content, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"Text file reading failed: {e}", exc_info=True)
            raise ToolError(
                "CONVERSION_FAILED",
                details={"file": str(file_path), "method": "Direct Text Read", "error": str(e)},
            ) from e

    ###############################################################################
    # Document Conversion Tool (Refactored with OCR)                             #
    ###############################################################################
    @tool(
        name="convert_document",
        description="Convert document (PDF, Office, Image) to specified format using various strategies including OCR and LLM enhancement.",
    )
    @with_tool_metrics
    @with_error_handling  # Keep top-level error handling
    async def convert_document(
        self,
        document_path: Optional[str] = None,
        document_data: Optional[bytes] = None,
        output_format: str = "markdown",
        extraction_strategy: str = DEFAULT_EXTRACTION_STRATEGY,
        enhance_with_llm: bool = True,
        ocr_options: Optional[Dict] = None,  # Bundled OCR options
        output_path: Optional[str] = None,
        save_to_file: bool = False,
        page_range: Optional[str] = None,  # e.g., "1-3,7,10" (1-based for user)
        section_filter: Optional[str] = None,  # Regex filter
        # Docling specific (kept for compatibility, only used if strategy='docling')
        accelerator_device: str = "auto",
        num_threads: int = 4,
    ) -> Dict[str, Any]:
        """
        Convert documents (PDF, Office formats, Images) to various formats.

        Leverages different extraction strategies:
        - 'docling': Uses Docling library for layout-aware conversion (requires installation).
                     Supports PDF, DOCX, PPTX, etc. -> MD, HTML, JSON, Text, DocTags.
        - 'direct_text': Extracts text directly from PDF structure (requires PyMuPDF or PDFPlumber).
                         Fast, good for digital PDFs. Output is Text or Markdown.
        - 'ocr': Converts PDF/Image pages to images and uses Tesseract OCR.
                 Good for scanned documents or images. Output is Text or Markdown.
        - 'hybrid_direct_ocr': Tries 'direct_text' first. If text is poor or missing, falls back to 'ocr'. Recommended default.

        Args:
            document_path: Path to the document file (PDF, DOCX, PNG, JPG, etc.). Mutually exclusive with document_data.
            document_data: Document content as bytes. Mutually exclusive with document_path.
            output_format: Target format ('markdown', 'text', 'html', 'json', 'doctags').
                           Note: 'ocr'/'direct_text'/'hybrid' strategies primarily support 'text' and 'markdown'.
                           Requesting other formats with these strategies will likely default to 'markdown'.
            extraction_strategy: Method to use ('docling', 'direct_text', 'ocr', 'hybrid_direct_ocr').
            enhance_with_llm: If True (default), enhance raw text extracted via 'direct_text' or 'ocr' using an LLM
                              to fix errors and improve formatting. Does not apply to Docling outputs.
            ocr_options: Dictionary of options for OCR/Enhancement strategies:
                - language (str): Tesseract language(s) (e.g., "eng", "eng+fra"). Default: "eng".
                - dpi (int): Resolution for rendering PDF pages to images for OCR. Default: 300.
                - remove_headers (bool): Attempt to remove headers/footers during LLM enhancement. Default: False.
                - preprocessing (dict): Image preprocessing options for OCR (see `_ocr_preprocess_image`).
                                         Example: {"denoise": True, "threshold": "adaptive"}.
                - assess_quality (bool): If True, run LLM assessment comparing raw vs enhanced text (adds cost). Default: False.
            output_path: Path to save the output file (if save_to_file is True).
            save_to_file: Whether to save the output to a file.
            page_range: 1-based range of pages to process (e.g., "1-5,8,10-12"). Applied during extraction.
            section_filter: Regex pattern to filter content sections (applied after extraction, before enhancement).
            accelerator_device: Device for Docling acceleration ('auto', 'cpu', 'cuda', 'mps'). Only used if strategy='docling'.
            num_threads: Number of threads for Docling. Only used if strategy='docling'.

        Returns:
            Dictionary with conversion results:
            {
                "success": bool,
                "content": str | Dict, # The converted content (string for text/md/html, dict for json/doctags)
                "output_format": str, # The actual output format produced
                "processing_time": float, # Seconds
                "document_metadata": Dict, # Extracted metadata (pages, tables, etc.)
                "extraction_strategy_used": str, # The strategy actually executed
                "file_path": Optional[str], # Path where output was saved, if applicable
                "raw_text": Optional[str], # Raw extracted text before LLM enhancement (if applicable)
                "ocr_quality_metrics": Optional[Dict] # Result from quality assessment (if requested)
                "error": Optional[str], # Error message on failure
                "error_code": Optional[str] # ToolError code on failure
            }
        """
        t0 = time.time()
        strategy = extraction_strategy.lower()
        output_format = output_format.lower()
        ocr_options = ocr_options or {}

        # --- Input Validation ---
        if not document_path and not document_data:
            raise ToolInputError("Either 'document_path' or 'document_data' must be provided.")
        if document_path and document_data:
            raise ToolInputError("Provide either 'document_path' or 'document_data', not both.")
        if strategy not in self._VALID_EXTRACTION_STRATEGIES:
            raise ToolInputError(
                f"Invalid extraction_strategy. Choose from: {', '.join(self._VALID_EXTRACTION_STRATEGIES)}",
                param_name="extraction_strategy",
                provided_value=strategy,
            )
        if output_format not in self._VALID_FORMATS:
            raise ToolInputError(
                f"Invalid output_format. Choose from: {', '.join(self._VALID_FORMATS)}",
                param_name="output_format",
                provided_value=output_format,
            )

        # Validate strategy vs dependencies and requested format
        if strategy == "docling" and not self._docling_available:
            raise ToolError(
                "DEPENDENCY_MISSING",
                details={"dependency": "docling", "feature": "Docling extraction strategy"},
            )
        if strategy in ["direct_text", "hybrid_direct_ocr"] and not (
            self._pymupdf_available or self._pdfplumber_available
        ):
            raise ToolError(
                "DEPENDENCY_MISSING",
                details={
                    "dependency": "PyMuPDF or PDFPlumber",
                    "feature": "Direct Text extraction strategy",
                },
            )
        if strategy in ["ocr", "hybrid_direct_ocr"]:
            missing_deps = []
            if not self._pdf2image_available:
                missing_deps.append("pdf2image")
            if not self._pytesseract_available:
                missing_deps.append("pytesseract")
            if not self._pil_available:
                missing_deps.append("Pillow")
            # CV2/Numpy are optional for preprocessing but highly recommended
            if not self._cv2_available:
                self.logger.warning(
                    "OpenCV (cv2) not found, OCR image preprocessing will be limited."
                )
            if not self._numpy_available:
                self.logger.warning("NumPy not found, OCR image preprocessing will be limited.")
            if missing_deps:
                raise ToolError(
                    "DEPENDENCY_MISSING",
                    details={
                        "dependency": ", ".join(missing_deps),
                        "feature": "OCR extraction strategy",
                    },
                )

        # Adjust output format compatibility
        effective_output_format = output_format
        if strategy != "docling" and output_format not in self._OCR_COMPATIBLE_FORMATS:
            self.logger.warning(
                f"Output format '{output_format}' is not directly supported by strategy '{strategy}'. Defaulting to 'markdown'."
            )
            effective_output_format = "markdown"  # Default for text-based strategies

        # --- Prepare Input ---
        input_content: Union[Path, bytes]
        temp_file_path: Optional[Path] = None
        is_temp_file = False
        try:
            if document_path:
                input_path = Path(document_path)
                if not input_path.is_file():
                    raise ToolInputError(
                        f"Input file not found: {document_path}", param_name="document_path"
                    )
                input_content = input_path
                input_name = input_path.name
            elif document_data:
                input_content = document_data
                # Try to guess a name/suffix for temp file and logging
                input_name = "input_data"
                if document_data.startswith(b"%PDF"):
                    input_name += ".pdf"
                elif document_data[6:10] in (b"JFIF", b"Exif"):
                    input_name += ".jpg"
                elif document_data.startswith(b"\x89PNG\r\n\x1a\n"):
                    input_name += ".png"

                # Save to temp file needed for most libraries except potentially bytes-based pdf2image/pymupdf
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(input_name).suffix or ".bin"
                ) as tmp:
                    tmp.write(document_data)
                    temp_file_path = Path(tmp.name)
                is_temp_file = True
                # Use the temp path for operations needing a path
                input_content = temp_file_path
                self.logger.debug(f"Input data saved to temporary file: {temp_file_path}")
            else:
                raise ToolInputError("Missing input source.")  # Should be caught earlier

            input_suffix = Path(input_name).suffix.lower()
            is_pdf = input_suffix == ".pdf"
            is_image = input_suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
            is_office = input_suffix in [".docx", ".pptx", ".xlsx"]  # Basic check

            # Validate strategy vs input type
            if not is_pdf and strategy in ["direct_text", "hybrid_direct_ocr"]:
                self.logger.warning(
                    f"Strategy '{strategy}' is designed for PDFs. Input is '{input_suffix}'. Falling back to 'ocr' strategy."
                )
                strategy = "ocr"  # Force OCR for non-PDFs if direct/hybrid was chosen
            if not is_pdf and not is_image and strategy == "ocr":
                raise ToolInputError(
                    f"OCR strategy requires PDF or Image input, got '{input_suffix}'. Use 'docling' for Office files."
                )
            if is_office and strategy != "docling":
                raise ToolInputError(
                    f"Office files ('{input_suffix}') require 'docling' extraction strategy."
                )

            # --- Parse Page Range ---
            # Convert 1-based user input to 0-based internal indices
            pages_to_process: Optional[List[int]] = None
            total_doc_pages = 0  # Will be determined later if needed
            if page_range:
                try:
                    pages_set = set()
                    parts = page_range.split(",")
                    for part in parts:
                        part = part.strip()
                        if "-" in part:
                            start, end = map(int, part.split("-"))
                            if start < 1 or end < start:
                                raise ValueError("Invalid range")
                            pages_set.update(
                                range(start - 1, end)
                            )  # 0-based end-exclusive range matches list slicing
                        else:
                            page_num = int(part)
                            if page_num < 1:
                                raise ValueError("Page number must be positive")
                            pages_set.add(page_num - 1)  # 0-based index
                    pages_to_process = sorted(list(pages_set))
                    if not pages_to_process:
                        raise ValueError("No valid pages selected.")
                    self.logger.debug(
                        f"Parsed page range: {page_range} -> 0-based indices: {pages_to_process}"
                    )
                except ValueError as e:
                    raise ToolInputError(
                        f"Invalid page_range format: '{page_range}'. Use comma-separated numbers/ranges (e.g., '1-3,5,7-9'). Error: {e}",
                        param_name="page_range",
                    ) from e

            # Define result structure defaults
            result_content: Union[str, Dict] = ""
            doc_metadata: Dict[str, Any] = {}
            raw_text_pages: List[str] = []  # Store text per page before joining/enhancement
            final_raw_text: Optional[str] = None
            quality_metrics: Optional[Dict] = None
            strategy_used = strategy  # Track the actual strategy used

            # ======================== EXTRACTION STRATEGIES ========================

            # ------------------------ Docling Strategy ---------------------------
            if strategy == "docling":
                self.logger.info(f"Using 'docling' strategy for {input_name}")
                self._ocr_check_dep(
                    "docling", self._docling_available, "Docling extraction strategy"
                )
                if not isinstance(input_content, Path):
                    raise ToolError(
                        "INTERNAL_ERROR",
                        details={"reason": "Docling strategy requires a file path"},
                    )  # Should have temp file

                device = self._ACCEL_MAP[accelerator_device.lower()]
                conv = self._get_docling_converter(device, num_threads)
                loop = asyncio.get_running_loop()

                with self._span("docling_conversion"):
                    docling_result = await loop.run_in_executor(None, conv.convert, input_content)

                if not docling_result or not docling_result.document:
                    raise ToolError(
                        "CONVERSION_FAILED",
                        details={
                            "document": str(input_content),
                            "reason": "Docling converter returned empty result",
                        },
                    )

                doc_obj = docling_result.document
                doc_metadata = self._get_docling_metadata(doc_obj)
                total_doc_pages = doc_metadata.get("num_pages", 0)

                # Apply page filtering *after* conversion for Docling
                if pages_to_process is not None:
                    # Docling doesn't have direct page filtering after load AFAIK.
                    # We need to export page by page or filter the output string.
                    # Filtering the string output is simpler but less precise.
                    self.logger.warning(
                        "Page filtering for Docling strategy is applied heuristically to the output text/markdown."
                    )
                    # Fall through to post-processing filter

                # Export content based on format
                if effective_output_format == "markdown":
                    result_content = doc_obj.export_to_markdown()
                elif effective_output_format == "text":
                    result_content = doc_obj.export_to_text()
                elif effective_output_format == "html":
                    result_content = doc_obj.export_to_html()
                elif effective_output_format == "json":
                    result_content = self._json(
                        doc_obj.export_to_dict()
                    )  # Serialize dict to string
                elif effective_output_format == "doctags":
                    result_content = doc_obj.export_to_doctags()  # Already a string

                # If saving, use Docling's save methods for better image handling
                if save_to_file:
                    fp = (
                        Path(output_path)
                        if output_path
                        else self._tmp_path(str(input_content), effective_output_format)
                    )
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    img_mode = (
                        ImageRefMode.PLACEHOLDER
                        if effective_output_format in ["markdown", "text", "json"]
                        else ImageRefMode.REFERENCED
                    )
                    save_func_map = {
                        "markdown": functools.partial(
                            doc_obj.save_as_markdown, image_mode=img_mode
                        ),
                        "text": functools.partial(
                            doc_obj.save_as_markdown, 
                        ),
                        "html": functools.partial(doc_obj.save_as_html, image_mode=img_mode),
                        "json": functools.partial(doc_obj.save_as_json, image_mode=img_mode),
                        "doctags": functools.partial(doc_obj.save_as_doctags),
                    }
                    if effective_output_format in save_func_map:
                        with self._span(f"docling_save_{effective_output_format}"):
                            save_func_map[effective_output_format](fp)
                        self.logger.info(
                            f"Saved Docling output ({effective_output_format}) to {fp}"
                        )
                        doc_metadata["saved_output_path"] = str(fp)
                    else:
                        # Fallback for formats Docling might not save directly
                        fp.write_text(str(result_content), encoding="utf-8")
                        self.logger.info(f"Saved Docling output (generic) to {fp}")
                        doc_metadata["saved_output_path"] = str(fp)

            # ------------------------ Text/OCR Strategies ------------------------
            else:  # Strategies: direct_text, ocr, hybrid_direct_ocr
                if not is_pdf and not is_image:
                    # Should be caught earlier, but safeguard
                    raise ToolInputError(
                        f"Strategies '{strategy}' require PDF or Image input, got {input_suffix}"
                    )

                run_ocr = False
                run_direct = False

                if strategy == "direct_text":
                    run_direct = True
                elif strategy == "ocr":
                    run_ocr = True
                elif strategy == "hybrid_direct_ocr":
                    if not is_pdf:  # Cannot run direct on images
                        run_ocr = True
                        strategy_used = "ocr"  # Update used strategy
                        self.logger.info(
                            "Input is an image, using 'ocr' strategy for hybrid request."
                        )
                    else:
                        run_direct = True  # Attempt direct first for hybrid PDF

                # --- Stage 1: Extraction (Direct or OCR) ---
                page_limit = (
                    0 if pages_to_process is None else len(pages_to_process)
                )  # Max pages adjusted by filter

                if run_direct:
                    self.logger.info(f"Using 'direct_text' strategy for {input_name}")
                    try:
                        with self._span("direct_text_extraction"):
                            extracted_pages, has_meaningful_text = (
                                self._ocr_extract_text_from_pdf_direct(
                                    input_content,
                                    start_page=pages_to_process[0] if pages_to_process else 0,
                                    max_pages=page_limit,
                                )
                            )
                        total_doc_pages = len(
                            extracted_pages
                        )  # Crude page count from extraction attempt

                        # If hybrid, check if we need to fallback to OCR
                        if strategy == "hybrid_direct_ocr" and not has_meaningful_text:
                            self.logger.warning(
                                "Direct text extraction yielded minimal text. Falling back to OCR strategy."
                            )
                            run_ocr = True
                            strategy_used = "ocr"  # Update actual strategy used
                        elif not has_meaningful_text and strategy == "direct_text":
                            # Explicit direct strategy failed
                            raise ToolError(
                                "DIRECT_EXTRACTION_FAILED",
                                details={"reason": "No meaningful text found in PDF structure."},
                            )
                        else:
                            # Direct text succeeded (or hybrid is satisfied)
                            raw_text_pages = extracted_pages
                            # Filter pages if range was specified (needs adjustment if direct func didn't handle it)
                            if pages_to_process is not None:
                                # Assume _ocr_extract_text_from_pdf_direct respects page limits based on start/max
                                # If it returned all pages, we'd need to filter here based on indices.
                                # Let's assume it returned only the requested pages.
                                pass  # Already filtered by extraction function params
                            self.logger.info(
                                f"Direct text extraction successful for {len(raw_text_pages)} pages."
                            )

                    except ToolError as e:
                        if strategy == "hybrid_direct_ocr":
                            self.logger.warning(
                                f"Direct text extraction failed ({e.error_code}). Falling back to OCR strategy."
                            )
                            run_ocr = True
                            strategy_used = "ocr"
                        else:  # Direct strategy failed explicitly
                            raise e

                if run_ocr:  # Runs if strategy='ocr' or if hybrid fallback triggered
                    self.logger.info(f"Using 'ocr' strategy for {input_name}")
                    strategy_used = "ocr"  # Ensure strategy reflects OCR was used
                    ocr_lang = ocr_options.get("language", "eng")
                    ocr_dpi = ocr_options.get("dpi", 300)
                    ocr_prep_opts = ocr_options.get("preprocessing")

                    images: List["PILImage.Image"] = []
                    if is_pdf:
                        # Convert PDF pages to images
                        convert_func = (
                            self._ocr_convert_pdf_bytes_to_images
                            if isinstance(input_content, bytes)
                            else self._ocr_convert_pdf_to_images
                        )
                        input_for_convert = input_content  # Path or bytes
                        with self._span("pdf_to_images"):
                            images = convert_func(
                                input_for_convert,  # type: ignore
                                start_page=pages_to_process[0] if pages_to_process else 0,
                                max_pages=page_limit,
                                dpi=ocr_dpi,
                            )
                        total_doc_pages = len(images)  # Page count from conversion
                    elif is_image:
                        # Load the single image
                        if not self._pil_available:
                            raise ToolError(
                                "DEPENDENCY_MISSING",
                                details={"dependency": "Pillow", "feature": "Image loading"},
                            )
                        try:
                            if isinstance(input_content, Path):
                                img = Image.open(input_content)  # type: ignore
                            images = [img.convert("RGB")]  # Ensure RGB for consistency
                            total_doc_pages = 1
                        except Exception as e:
                            raise ToolError("IMAGE_LOAD_FAILED", details={"error": str(e)}) from e
                    else:
                        # Should not happen based on earlier checks
                        raise ToolError(
                            "INTERNAL_ERROR",
                            details={"reason": "OCR strategy called on unsupported file type"},
                        )

                    if not images:
                        raise ToolError(
                            "OCR_FAILED",
                            details={"reason": "No images generated or loaded for OCR."},
                        )

                    # Apply page range filtering to images if needed (e.g., if conversion returned all)
                    if pages_to_process is not None and is_pdf:
                        # Assuming convert function returned only requested pages. If not, filter here:
                        # filtered_images = [img for i, img in enumerate(images) if (pages_to_process[0] + i) in pages_to_process] # Complex mapping
                        # images = filtered_images
                        pass  # Assume filtering happened during conversion request

                    # Process images in parallel (Preprocess -> OCR)
                    processed_pages_text: List[str] = [""] * len(
                        images
                    )  # Initialize with empty strings

                    async def _process_ocr_page(idx: int, img: "PILImage.Image") -> Tuple[int, str]:
                        try:
                            with self._span(f"ocr_page_{idx}_preprocess"):
                                preprocessed_img = self._ocr_preprocess_image(img, ocr_prep_opts)
                            with self._span(f"ocr_page_{idx}_tesseract"):
                                page_text = self._ocr_run_tesseract(preprocessed_img, ocr_lang)
                            return idx, page_text
                        except Exception as page_err:
                            self.logger.error(
                                f"Error processing page {idx} for OCR: {page_err}", exc_info=True
                            )
                            return idx, ""  # Return empty string on page error

                    # Use asyncio.gather for concurrent page processing
                    # Limit concurrency? Tesseract can be CPU intensive. Use ThreadPoolExecutor implicitly?
                    # Let's run them concurrently via asyncio.gather, assuming underlying I/O or Tesseract process allows some parallelization.
                    # TODO: Consider explicit ThreadPoolExecutor if pure CPU bound.
                    tasks = [_process_ocr_page(i, img) for i, img in enumerate(images)]
                    page_results = await asyncio.gather(*tasks)

                    # Collect results in order
                    for idx, text in page_results:
                        processed_pages_text[idx] = text

                    raw_text_pages = processed_pages_text
                    self.logger.info(
                        f"OCR extraction successful for {len(raw_text_pages)} pages/images."
                    )

                # --- Stage 2: Post-Extraction Processing (Raw Text) ---
                if not raw_text_pages:
                    raise ToolError(
                        "EXTRACTION_FAILED",
                        details={
                            "reason": f"Strategy '{strategy_used}' did not yield any text content."
                        },
                    )

                # Combine raw pages into single string
                final_raw_text = "\n\n".join(
                    raw_text_pages
                ).strip()  # Use double newline as page separator

                # Apply section filter to raw text
                if section_filter:
                    try:
                        pat = re.compile(section_filter, re.I | re.M)
                        # Split by paragraphs/blocks, filter, rejoin
                        blocks = re.split(r"(\n\s*\n)", final_raw_text)  # Keep separators
                        kept_content = ""
                        for i in range(0, len(blocks), 2):  # Process content + separator
                            block = blocks[i]
                            separator = blocks[i + 1] if i + 1 < len(blocks) else ""
                            if block and pat.search(block):
                                kept_content += block + separator
                        final_raw_text = kept_content.strip()
                        self.logger.info(f"Applied section filter regex: '{section_filter}'")
                    except re.error as e:
                        self.logger.warning(
                            f"Invalid regex for section_filter: '{section_filter}'. Skipping filter. Error: {e}"
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to apply section filter: {e}")

                # --- Stage 3: LLM Enhancement (Optional) ---
                if enhance_with_llm:
                    self.logger.info("Applying LLM enhancement to extracted text.")
                    with self._span("llm_text_enhancement"):
                        # Split raw text into chunks for LLM
                        chunks = self._ocr_split_text_into_chunks(final_raw_text)
                        if not chunks:
                            self.logger.warning(
                                "Text became empty after filtering/splitting before LLM enhancement."
                            )
                            result_content = ""
                        else:
                            # Process chunks concurrently
                            enhancement_tasks = [
                                self._ocr_enhance_text_chunk(
                                    chunk,
                                    output_format=effective_output_format,
                                    remove_headers=ocr_options.get("remove_headers", False),
                                )
                                for chunk in chunks
                            ]
                            enhanced_chunks = await asyncio.gather(*enhancement_tasks)
                            result_content = "\n\n".join(
                                enhanced_chunks
                            ).strip()  # Rejoin enhanced chunks
                else:
                    # No LLM enhancement, result is the raw (potentially filtered) text
                    result_content = final_raw_text
                    # If markdown was requested, basic text-to-md conversion might be needed?
                    if effective_output_format == "markdown":
                        # Maybe wrap in pre tags? Or just return as is? Let's return as is.
                        self.logger.debug(
                            "LLM enhancement disabled, returning raw text for markdown format."
                        )

                # Generate metadata for text/ocr strategies
                doc_metadata = self._get_basic_metadata(str(result_content), total_doc_pages)

                # --- Stage 4: Quality Assessment (Optional) ---
                if enhance_with_llm and ocr_options.get("assess_quality", False):
                    self.logger.info("Performing OCR quality assessment.")
                    with self._span("ocr_quality_assessment"):
                        quality_metrics = await self._ocr_assess_text_quality(
                            final_raw_text or "", str(result_content)
                        )

            # ======================== POST-PROCESSING & RETURN ========================

            # Apply page filter to string output if Docling was used (heuristic)
            # This is less reliable than filtering during extraction
            if (
                strategy == "docling"
                and pages_to_process is not None
                and isinstance(result_content, str)
            ):
                self.logger.debug("Applying heuristic page filtering to Docling string output.")
                # Simple heuristic: Split by form feed character (\f) if present, or guess based on page numbers? Very unreliable.
                # Alternative: Could render markdown/html and count pages - complex.
                # Let's skip string filtering for Docling for now, rely on user understanding it's hard post-conversion.
                self.logger.warning(
                    "Heuristic page filtering for Docling output is not implemented. Full document content returned."
                )

            # Final content assignment
            final_content = result_content

            # Save non-docling output if requested
            if strategy != "docling" and save_to_file:
                fp = (
                    Path(output_path)
                    if output_path
                    else self._tmp_path(input_name, effective_output_format)
                )
                fp.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if isinstance(
                        final_content, dict
                    ):  # Should only be JSON from docling, but handle just in case
                        fp.write_text(self._json(final_content), encoding="utf-8")
                    else:  # String content
                        fp.write_text(str(final_content), encoding="utf-8")
                    self.logger.info(
                        f"Saved output ({effective_output_format}, strategy: {strategy_used}) to {fp}"
                    )
                    doc_metadata["saved_output_path"] = str(fp)
                except Exception as e:
                    self.logger.error(f"Failed to save output file to {fp}: {e}", exc_info=True)
                    # Continue without saving

            # --- Construct Final Response ---
            elapsed = round(time.time() - t0, 3)
            response: Dict[str, Any] = {
                "success": True,
                "content": final_content,
                "output_format": effective_output_format,
                "processing_time": elapsed,
                "document_metadata": doc_metadata,
                "extraction_strategy_used": strategy_used,
            }
            if final_raw_text is not None:
                response["raw_text"] = final_raw_text
            if quality_metrics is not None:
                response["ocr_quality_metrics"] = quality_metrics
            if "saved_output_path" in doc_metadata:
                response["file_path"] = doc_metadata["saved_output_path"]

            self.logger.info(
                f"Completed conversion '{input_name}' -> {effective_output_format} (strategy: {strategy_used}) in {elapsed}s"
            )
            return response

        finally:
            # Clean up temporary file if created
            if is_temp_file and temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                    self.logger.debug(f"Cleaned up temporary input file: {temp_file_path}")
                except OSError as e:
                    self.logger.warning(
                        f"Failed to delete temporary input file {temp_file_path}: {e}"
                    )

    # Wrapper for batch processing compatibility
    async def convert_document_op(
        self, *, output_key: str = "conversion_result", **kwargs
    ) -> Dict[str, Any]:
        """Internal wrapper for convert_document used by batch processing."""
        # We expect all necessary arguments (document_path/data, strategy, etc.) to be in kwargs
        # The batch processor will map keys from the item state to these arguments.
        try:
            # Remove output_key if present in kwargs, as it's for the batch processor itself
            kwargs.pop("output_key", None)
            # Call the main convert_document method
            res = await self.convert_document(**kwargs)
            # Result format from convert_document is already suitable
            return res
        except Exception as e:
            # Ensure errors are captured in a dictionary format compatible with batch processing
            self.logger.error(
                f"Error during batch operation call to convert_document: {e}", exc_info=True
            )
            error_code = (
                getattr(e, "code", "BATCH_OPERATION_FAILED")
                if isinstance(e, ToolError)
                else "BATCH_OPERATION_FAILED"
            )
            error_msg = getattr(e, "message", str(e)) if isinstance(e, ToolError) else str(e)
            return {"success": False, "error": error_msg, "error_code": error_code}

    ###############################################################################
    # Document Chunking Tool & Helpers                                            #
    ###############################################################################

    def _token_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by tokens, respecting sentence boundaries."""
        self._lazy_import_tiktoken()
        if self._tiktoken_enc_instance is False or not callable(
            getattr(self._tiktoken_enc_instance, "encode", None)
        ):
            self.logger.warning(
                "Tiktoken not available or invalid, falling back to character chunking for token method."
            )
            char_size = size * 4  # Rough estimate
            char_overlap = overlap * 4
            return self._char_chunks(doc, char_size, char_overlap)
        if not doc:
            return []

        enc = self._tiktoken_enc_instance  # Type assertion for clarity
        try:
            tokens = enc.encode(doc, disallowed_special=())
        except Exception as e:
            self.logger.error(
                f"Tiktoken encoding failed: {e}. Falling back to character chunking.", exc_info=True
            )
            char_size = size * 4
            char_overlap = overlap * 4
            return self._char_chunks(doc, char_size, char_overlap)

        if not tokens:
            return []

        chunks: List[str] = []
        current_pos = 0
        n_tokens = len(tokens)

        try:
            # Try to get sentence end tokens dynamically
            sentence_end_tokens = {enc.encode(p)[0] for p in (".", "?", "!", "\n")}
        except Exception as e:
            self.logger.warning(
                f"Could not encode sentence end markers: {e}. Using default IDs for cl100k_base."
            )
            sentence_end_tokens = {13, 30, 106, 198}  # Approx: '.', '?', '!', '\n'

        while current_pos < n_tokens:
            end_pos = min(current_pos + size, n_tokens)
            best_split_pos = end_pos

            if end_pos < n_tokens:  # Only look back if not at the very end
                lookback_distance = min(overlap, size // 4, end_pos - current_pos)
                search_start = max(current_pos, end_pos - lookback_distance)

                # Find the last sentence end token in the lookback window
                for k in range(end_pos - 1, search_start - 1, -1):
                    if tokens[k] in sentence_end_tokens:
                        best_split_pos = k + 1
                        break

            chunk_token_ids = tokens[current_pos:best_split_pos]
            if not chunk_token_ids:
                if current_pos >= n_tokens:
                    break
                current_pos += 1
                continue

            try:
                chunk_text = enc.decode(chunk_token_ids).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            except Exception:
                self.logger.error(
                    f"Tiktoken decoding failed for tokens {current_pos}:{best_split_pos}. Skipping chunk.",
                    exc_info=True,
                )

            # Move start position for next chunk, ensuring progress
            next_start_pos = best_split_pos - overlap
            current_pos = (
                max(current_pos + 1, next_start_pos)
                if best_split_pos > current_pos
                else best_split_pos
            )

        return chunks

    def _char_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by characters, respecting sentence/paragraph boundaries."""
        if not doc:
            return []

        chunks: List[str] = []
        current_pos = 0
        n_chars = len(doc)
        sentence_ends = (
            "\n\n",
            ". ",
            "? ",
            "! ",
            "\n",
        )  # Prioritize breaks followed by space/newline
        softer_breaks = ("; ", ": ", ", ", "\t", " ")

        while current_pos < n_chars:
            end_pos = min(current_pos + size, n_chars)
            best_split_pos = end_pos

            if end_pos < n_chars:
                best_found_pos = -1
                # Look back within a reasonable window
                lookback_window_start = max(current_pos, end_pos - int(size * 0.2), end_pos - 150)

                # Search for preferred breaks first
                for marker in sentence_ends:
                    found_pos = doc.rfind(marker, lookback_window_start, end_pos)
                    if found_pos != -1:
                        best_found_pos = max(best_found_pos, found_pos + len(marker))

                # If no preferred break found, search for softer breaks
                if best_found_pos == -1:
                    for marker in softer_breaks:
                        found_pos = doc.rfind(marker, lookback_window_start, end_pos)
                        if found_pos != -1:
                            best_found_pos = max(best_found_pos, found_pos + len(marker))

                # Use the best found position if it's valid
                if best_found_pos != -1 and best_found_pos > current_pos:
                    best_split_pos = best_found_pos
                # If no break found or found break is too early, use hard cut-off
                elif best_split_pos <= current_pos:  # Ensure split is after current_pos
                    best_split_pos = end_pos

            actual_chunk_text = doc[current_pos:best_split_pos].strip()
            if actual_chunk_text:
                chunks.append(actual_chunk_text)

            # Calculate next start position with overlap, ensuring forward progress
            next_start_pos = best_split_pos - overlap
            current_pos = (
                max(current_pos + 1, next_start_pos)
                if best_split_pos > current_pos
                else best_split_pos
            )

        return chunks

    def _paragraph_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by paragraphs, combining small ones up to size limit."""
        if not doc:
            return []
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]
        if not paragraphs:
            return []

        chunks = []
        current_chunk_paragraphs: List[str] = []
        current_chunk_len = 0
        # Use character length for size check for simplicity across methods
        use_tiktoken_len = False  # Keep consistent with char_chunks logic  # noqa: F841

        def get_len(text: str) -> int:
            return len(text)

        def is_markdown_table(text: str) -> bool:
            lines = text.strip().split("\n")
            return (
                len(lines) >= 2
                and all(line.strip().startswith("|") for line in lines[:2])
                and "|" in lines[0]
                and re.search(r"\|.*?(-{3,}|:{1,2}-{1,}:?).*?\|", lines[1]) is not None
            )

        for p in paragraphs:
            p_len = get_len(p)
            # Length calculation includes double newline separator
            potential_new_len = (
                current_chunk_len + (get_len("\n\n") if current_chunk_paragraphs else 0) + p_len
            )
            is_table = is_markdown_table(p)

            if current_chunk_paragraphs and potential_new_len > size and not is_table:
                # Finalize the current chunk
                chunks.append("\n\n".join(current_chunk_paragraphs))
                # Start new chunk, potentially with overlap (simplified: no overlap here)
                current_chunk_paragraphs = [p]
                current_chunk_len = p_len
            elif p_len > size and not is_table:
                # Paragraph itself is too long (and chunk is empty or paragraph doesn't fit)
                self.logger.warning(
                    f"Paragraph starting with '{p[:50]}...' (length {p_len}) exceeds chunk size {size}. Splitting paragraph using character chunking."
                )
                # Add previous chunk if exists
                if current_chunk_paragraphs:
                    chunks.append("\n\n".join(current_chunk_paragraphs))
                # Split the oversized paragraph
                sub_chunks = self._char_chunks(p, size, overlap)
                chunks.extend(sub_chunks)
                # Reset current chunk tracking
                current_chunk_paragraphs = []
                current_chunk_len = 0
            else:
                # Add paragraph to current chunk (fits, or is a table potentially exceeding size)
                current_chunk_paragraphs.append(p)
                current_chunk_len = potential_new_len

        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        self.logger.info(f"Chunked into {len(chunks)} paragraphs/groups.")
        return chunks

    async def _section_chunks(self, doc: str, size: int, overlap: int) -> List[str]:
        """Chunk document by identified sections. Falls back to paragraphs if sections fail."""
        try:
            section_result = await self.identify_sections(document=doc)
            if (
                isinstance(section_result, dict)
                and section_result.get("success")
                and isinstance(section_result.get("sections"), list)
            ):
                sections = section_result["sections"]
            else:
                self.logger.warning(
                    "identify_sections did not return expected format. Falling back to paragraph chunking."
                )
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self._paragraph_chunks, doc, size, overlap)

            if not sections:
                self.logger.info("No sections identified, using paragraph chunking as fallback.")
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self._paragraph_chunks, doc, size, overlap)

            section_texts: List[str] = []
            for s in sections:
                title = s.get("title", "").strip()
                text = s.get("text", "").strip()
                if text:  # Only add sections with actual text content
                    # Heuristic for adding title: Avoid adding generic "Introduction" or "Main Content" back in
                    use_title = title and title.lower() not in [
                        "introduction",
                        "main content",
                        "body",
                    ]
                    # Maybe use markdown heading? Assumes text is markdown-ish
                    full_section_text = f"## {title}\n\n{text}" if use_title else text
                    section_texts.append(full_section_text.strip())

            # Function to check if text contains tables (from paragraph chunker)
            def contains_markdown_table(text: str) -> bool:
                lines = text.strip().split("\n")
                return (
                    len(lines) >= 2
                    and all(line.strip().startswith("|") for line in lines[:2])
                    and "|" in lines[0]
                    and re.search(r"\|.*?(-{3,}|:{1,2}-{1,}:?).*?\|", lines[1]) is not None
                )

            # If sections are too large, sub-chunk them using paragraph strategy
            final_chunks = []
            loop = asyncio.get_running_loop()
            for text in section_texts:
                # Use character length check
                has_table = contains_markdown_table(text)
                # Allow tables to slightly exceed limit, but very large sections always split
                if len(text) > size * 1.1 and (not has_table or len(text) > size * 2):
                    self.logger.warning(
                        f"Section chunk starting with '{text[:50]}...' exceeds size limit ({len(text)} > {size}). Sub-chunking section using paragraphs."
                    )
                    sub_chunks = await loop.run_in_executor(
                        None, self._paragraph_chunks, text, size, overlap
                    )
                    final_chunks.extend(sub_chunks)
                elif text:  # Add section if it fits and is not empty
                    final_chunks.append(text)

            return final_chunks

        except Exception as e:
            self.logger.error(
                f"Failed to get sections for chunking: {e}. Falling back to paragraph chunking.",
                exc_info=True,
            )
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._paragraph_chunks, doc, size, overlap)

    @tool(
        name="chunk_document",
        description="Split document text into chunks using various strategies",
    )
    @with_tool_metrics
    @with_error_handling
    async def chunk_document(
        self,
        document: str,
        *,
        chunk_size: int = 1000,
        chunk_method: str = "paragraph",
        chunk_overlap: int = 0,
        chunk_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Split document into chunks based on specified method and size.

        Args:
            document: Text content to chunk.
            chunk_size: Target maximum size of each chunk (meaning depends on method: tokens or characters).
            chunk_method: Chunking method ('token', 'character', 'section', 'paragraph').
            chunk_overlap: Number of tokens/characters to overlap between chunks (for token/char methods).
                           Overlap logic for paragraph/section is heuristic/simplified.
            chunk_strategy: Alias for chunk_method (for backward compatibility).

        Returns:
            Dictionary containing list of chunked text strings.
            Example: {"chunks": ["chunk 1 text...", "chunk 2 text..."], "success": True}
        """
        if not document or not isinstance(document, str):
            self.logger.warning("Chunking called with empty or invalid document input.")
            return {"chunks": [], "success": True}

        size = max(100, int(chunk_size))  # Min chunk size 100
        overlap = max(0, min(int(chunk_overlap), size // 3))  # Ensure reasonable overlap
        method = (chunk_strategy or chunk_method or "paragraph").lower()

        chunker_map: Dict[str, Callable[..., Union[List[str], Awaitable[List[str]]]]] = {
            "token": self._token_chunks,
            "character": self._char_chunks,
            "section": self._section_chunks,  # Async
            "paragraph": self._paragraph_chunks,
        }

        strat_func = chunker_map.get(method)
        if not strat_func:
            self.logger.warning(f"Unknown chunk_method '{method}'. Defaulting to 'paragraph'.")
            strat_func = self._paragraph_chunks
            method = "paragraph"

        self.logger.info(
            f"Chunking document using method='{method}', size={size}, overlap={overlap}"
        )
        chunks: List[str] = []
        try:
            with self._span(f"chunk_document_{method}"):
                if asyncio.iscoroutinefunction(strat_func):
                    chunks = await strat_func(document, size, overlap)  # type: ignore
                else:
                    loop = asyncio.get_running_loop()
                    chunks = await loop.run_in_executor(None, strat_func, document, size, overlap)
        except Exception as e:
            self.logger.error(f"Error during chunking operation ({method}): {e}", exc_info=True)
            raise ToolError("CHUNKING_FAILED", details={"method": method, "error": str(e)}) from e

        final_chunks = [c for c in chunks if isinstance(c, str) and c]
        self.logger.info(f"Generated {len(final_chunks)} chunks.")
        return {"chunks": final_chunks, "success": True}

    ###############################################################################
    # HTML Processing Tools                                                       #
    ###############################################################################
    # [HTML methods _extract_readability, _extract_trafilatura, _html_to_md_core]
    # [and tools clean_and_format_text_as_markdown, detect_content_type, ]
    # [batch_format_texts, optimize_markdown_formatting remain unchanged ]
    # [ from the original DocumentProcessingTool code. ]

    def _extract_readability(self, html_txt: str) -> str:
        """Extract main content using readability-lxml."""
        if not readability:
            self.logger.warning(
                "Readability-lxml not installed. Cannot use readability extraction."
            )
            return ""
        try:
            # Increase cleaning efforts
            readability.htmls.DEFAULT_REGEXES["unlikelyCandidates"] = re.compile(
                readability.htmls.DEFAULT_REGEXES["unlikelyCandidates"].pattern
                + "|aside|footer|nav|sidebar|footnote|advertisement|related|recommend|share|social|comment|meta",
                re.I,
            )
            readability.htmls.DEFAULT_REGEXES["positive"] = re.compile(
                readability.htmls.DEFAULT_REGEXES["positive"].pattern
                + "|article|main|content|post|entry|body",
                re.I,
            )
            readability.htmls.DEFAULT_REGEXES["negative"] = re.compile(
                readability.htmls.DEFAULT_REGEXES["negative"].pattern
                + "|widget|menu|legal|promo|disclaimer",
                re.I,
            )

            doc = readability.Document(html_txt)
            # Return the main content as cleaned HTML string
            summary_html = doc.summary(html_partial=True)
            # Additional cleaning on the summary?
            soup, _ = self._best_soup(summary_html)
            return str(soup)
        except Exception as e:
            self.logger.warning(f"Readability extraction failed: {e}", exc_info=True)
            return ""

    def _extract_trafilatura(self, html_txt: str) -> str:
        """Extract main content using trafilatura."""
        if not trafilatura:
            self.logger.warning("Trafilatura not installed. Cannot use trafilatura extraction.")
            return ""
        try:
            # Configure trafilatura settings for better extraction
            extracted = trafilatura.extract(
                html_txt,
                include_comments=False,
                include_tables=True,
                favor_precision=True,  # Focus on quality over quantity
                deduplicate=True,  # Remove duplicate text sections
                target_language=None,  # Auto-detect language
                include_formatting=True,  # Try to keep basic formatting
                output_format="html",  # Output HTML for further processing
            )
            return extracted or ""
        except Exception as e:
            self.logger.warning(f"Trafilatura extraction failed: {e}", exc_info=True)
            return ""

    def _html_to_md_core(
        self, html_txt: str, links: bool, imgs: bool, tbls: bool, width: int
    ) -> str:
        """Convert HTML to Markdown using primary and fallback libraries."""
        md_text = ""
        # Primary: html2text
        try:
            h = html2text.HTML2Text()
            h.ignore_links = not links
            h.ignore_images = not imgs
            # Handle tables: If tbls is true, we might convert them separately first,
            # otherwise html2text's handling can be inconsistent.
            # Let's convert tables to MD *before* html2text if tbls=True.
            processed_html = html_txt
            if tbls:
                processed_html = self._convert_html_tables_to_markdown(html_txt)
                h.ignore_tables = (
                    False  # Let it process other content, MD tables should pass through
                )
            else:
                h.ignore_tables = True

            h.body_width = width if width > 0 else 0
            h.unicode_snob = True
            h.escape_snob = True
            h.skip_internal_links = True
            h.single_line_break = True  # Use single line breaks between paragraphs

            md_text = h.handle(processed_html)
            self.logger.debug("html2text conversion successful.")
            return md_text.strip()

        except Exception as e_html2text:
            self.logger.warning(
                f"html2text failed ({e_html2text}); attempting fallback with markdownify"
            )
            if _markdownify_fallback and callable(_markdownify_fallback):
                try:
                    md_opts = {
                        "strip": [
                            "script",
                            "style",
                            "meta",
                            "link",
                            "head",
                            "iframe",
                            "form",
                            "button",
                            "input",
                            "select",
                            "textarea",
                            "nav",
                            "aside",
                            "header",
                            "footer",
                        ],
                        "convert": [
                            "a",
                            "p",
                            "img",
                            "br",
                            "hr",
                            "h1",
                            "h2",
                            "h3",
                            "h4",
                            "h5",
                            "h6",
                            "li",
                            "ul",
                            "ol",
                            "blockquote",
                            "code",
                            "pre",
                            "strong",
                            "em",
                            "table",
                            "tr",
                            "td",
                            "th",
                        ],
                        "heading_style": "ATX",
                        "bullets": "-",
                        "strong_em_symbol": "*",
                        "autolinks": False,
                    }
                    if not links:
                        md_opts["convert"] = [tag for tag in md_opts["convert"] if tag != "a"]
                    if not imgs:
                        md_opts["convert"] = [tag for tag in md_opts["convert"] if tag != "img"]
                    if not tbls:
                        md_opts["convert"] = [
                            tag
                            for tag in md_opts["convert"]
                            if tag not in ["table", "tr", "td", "th"]
                        ]

                    md_text = _markdownify_fallback(html_txt, **md_opts)
                    self.logger.debug("Markdownify fallback conversion successful.")
                    return md_text.strip()
                except Exception as e_markdownify:
                    self.logger.error(
                        f"Markdownify fallback also failed: {e_markdownify}", exc_info=True
                    )
                    raise ToolError(
                        "MARKDOWN_CONVERSION_FAILED",
                        details={
                            "reason": "Both html2text and markdownify failed",
                            "html2text_error": str(e_html2text),
                            "markdownify_error": str(e_markdownify),
                        },
                    ) from e_html2text
            else:
                self.logger.error("html2text failed and markdownify fallback is not available.")
                raise ToolError(
                    "MARKDOWN_CONVERSION_FAILED",
                    details={
                        "reason": "html2text failed, markdownify not installed",
                        "error": str(e_html2text),
                    },
                ) from e_html2text

    @tool(
        name="clean_and_format_text_as_markdown",
        description="Convert plain text or HTML to clean Markdown, optionally extracting main content",
    )
    @with_tool_metrics
    @with_error_handling
    async def clean_and_format_text_as_markdown(
        self,
        text: str,
        force_markdown_conversion: bool = False,
        extraction_method: str = "auto",  # auto, readability, trafilatura, none
        preserve_tables: bool = True,
        preserve_links: bool = True,
        preserve_images: bool = False,
        max_line_length: int = 0,  # 0 means no wrapping
    ) -> Dict[str, Any]:
        """
        Convert plain text or HTML to clean Markdown, with options to extract main content.

        This tool handles:
        1. Content type auto-detection (HTML vs Markdown vs plain text)
        2. Main content extraction from HTML (removing boilerplate like navbars, sidebars, footers)
        3. Converting HTML to Markdown with options to preserve tables/links/images
        4. Cleaning up the result for consistent, readable Markdown

        Args:
            text: Input text (HTML, Markdown, or plain text)
            force_markdown_conversion: If True, always convert to markdown even if input looks like markdown
            extraction_method: How to extract content from HTML:
                            - "auto": Try best method automatically
                            - "readability": Use Mozilla's Readability algorithm
                            - "trafilatura": Use Trafilatura library
                            - "none": Don't extract, convert whole page
            preserve_tables: Whether to keep HTML tables in the output
            preserve_links: Whether to keep hyperlinks in the output
            preserve_images: Whether to keep image references in the output
            max_line_length: Line width for wrapping (0 means no wrapping)

        Returns:
            Dictionary with conversion results:
            {
                "success": bool,
                "markdown_text": str,
                "original_content_type": str,  # "html", "markdown", "text"
                "was_html": bool,              # Whether the input was detected as HTML
                "extraction_method_used": str, # Method actually used for extraction
                "processing_time": float,      # Seconds
            }
        """
        start_time = time.time()
        
        if not text or not isinstance(text, str):
            raise ToolInputError("Input text must be a non-empty string", param_name="text")
            
        # First, detect the content type
        content_type_result = await self.detect_content_type(text)
        input_type = content_type_result.get("content_type", "unknown")
        input_confidence = content_type_result.get("confidence", 0.0)
        
        was_html = (input_type == "html")
        extraction_method_used = "none"
        
        # Log detection results
        self.logger.debug(
            f"Content type detection: {input_type} (confidence: {input_confidence:.2f})"
        )
        
        # If detection confidence is very low, try to examine more text-specific indicators
        if input_confidence < 0.4:
            # Check for HTML doctype or common HTML patterns more explicitly
            if re.search(r"<!DOCTYPE\s+html|<html>|<body>|<div\s|<p>", text[:1000], re.IGNORECASE):
                was_html = True
                self.logger.debug("Low confidence detection, but found HTML markers - treating as HTML")
            # Check for markdown patterns more explicitly
            elif re.search(r"^#\s+|^\*\s+|^>\s+|^\d+\.\s+|\[.+\]\(.+\)", text[:1000], re.MULTILINE):
                was_html = False
                self.logger.debug("Low confidence detection, but found Markdown patterns - treating as Markdown")

        # Default output is the input (for non-HTML/no-conversion cases)
        md_text = text
        
        # If it's HTML or forced conversion is requested
        if was_html or force_markdown_conversion:
            # Record that we're handling HTML
            was_html = True
            
            # Determine which extraction method to use
            actual_extraction = extraction_method.lower()
            if actual_extraction == "auto":
                # Auto-selection logic: prefer Trafilatura if available, fall back to Readability
                if self._trafilatura_available:
                    actual_extraction = "trafilatura"
                    self.logger.debug("Auto-selected Trafilatura for HTML extraction")
                else:
                    actual_extraction = "readability"
                    self.logger.debug("Auto-selected Readability for HTML extraction (Trafilatura unavailable)")
            
            extraction_method_used = actual_extraction
            
            # Perform content extraction if requested (not 'none')
            if actual_extraction != "none":
                self.logger.debug(f"Extracting main content using {actual_extraction}")
                try:
                    if actual_extraction == "readability":
                        # First check if readability is available
                        if not self._readability_available:
                            self.logger.warning(
                                "Readability extraction requested but library not available, falling back to raw HTML"
                            )
                            actual_extraction = "none"
                        else:
                            extracted_text = self._extract_readability(text)
                            # If extraction fails, we'll get empty content
                            if not extracted_text or len(extracted_text) < 50:
                                self.logger.warning(
                                    f"Readability extraction failed or produced very short content ({len(extracted_text)} chars)"
                                )
                                # Fall back to no extraction
                                actual_extraction = "none"
                            else:
                                text = extracted_text
                                self.logger.debug(
                                    f"Readability extracted {len(text)} chars of HTML content"
                                )
                    
                    elif actual_extraction == "trafilatura":
                        # First check if trafilatura is available
                        if not self._trafilatura_available:
                            self.logger.warning(
                                "Trafilatura extraction requested but library not available, falling back to raw HTML"
                            )
                            actual_extraction = "none"
                        else:
                            extracted_text = self._extract_trafilatura(text)
                            # If extraction fails, we'll get empty content
                            if not extracted_text or len(extracted_text) < 50:
                                self.logger.warning(
                                    f"Trafilatura extraction failed or produced very short content ({len(extracted_text)} chars)"
                                )
                                # Fall back to no extraction
                                actual_extraction = "none"
                            else:
                                text = extracted_text
                                self.logger.debug(
                                    f"Trafilatura extracted {len(text)} chars of HTML content"
                                )
                    
                    # If extraction failed and we're falling back, record that
                    if actual_extraction == "none":
                        extraction_method_used = "none (extraction failed)"
                        # Use the original HTML
                        text = text
                
                except Exception as e:
                    self.logger.error(f"Error during HTML content extraction: {e}", exc_info=True)
                    extraction_method_used = "none (error during extraction)"
                    # Continue with the original HTML
            
            # Now convert to Markdown
            try:
                # Do the HTML to Markdown conversion with the requested options
                md_text = self._html_to_md_core(
                    text,
                    links=preserve_links,
                    imgs=preserve_images,
                    tbls=preserve_tables,
                    width=max_line_length,
                )
                
                # Clean up the markdown
                md_text = self._sanitize(md_text)
                
                # Check if the conversion was successful
                if not md_text or (len(md_text) < 50 and len(text) > 200):
                    self.logger.warning("HTML to Markdown conversion produced very short output")
                    # If HTML conversion failed but input has some markdown-like syntax already, 
                    # just return the cleaned input as a fallback
                    if input_type == "markdown" or re.search(r"^#|^\*|\[.+\]\(.+\)", text[:500], re.MULTILINE):
                        md_text = self._sanitize(text)
                        self.logger.debug("Falling back to cleaned original text with markdown-like syntax")
                
                # Apply additional Markdown improvements
                md_text = self._improve(md_text)
                
                # If we still have HTML tags in the output, try to clean further
                if re.search(r"<[a-z][a-z0-9]*\b[^>]*>", md_text):
                    self.logger.warning("HTML tags found in Markdown output - attempting cleanup")
                    # Try to strip remaining HTML tags
                    md_text = re.sub(r"<[^>]+>", "", md_text)
                    # Restore markdown headers, lists, and links that might have been damaged
                    md_text = re.sub(r"^(?<!\#)\s*([A-Z][^\n]+)$", r"## \1", md_text, flags=re.MULTILINE)
                    md_text = re.sub(r"^\s*[•·]?\s*([A-Z][^\n]*?[.!?])$", r"* \1", md_text, flags=re.MULTILINE)
                    
            except Exception as e:
                self.logger.error(f"Error converting HTML to Markdown: {e}", exc_info=True)
                # Fall back to just sanitizing the text as best we can
                md_text = self._sanitize(text)
        
        # If input was already Markdown and no conversion requested,
        # just clean it up for consistency
        elif input_type == "markdown" and not force_markdown_conversion:
            self.logger.debug("Input already appears to be Markdown, cleaning up for consistency")
            # Apply sanitization and improvements
            md_text = self._sanitize(md_text)
            md_text = self._improve(md_text)
        
        # For plain text, add minimal formatting based on structure
        elif input_type == "text" or input_type == "unknown":
            self.logger.debug("Input appears to be plain text, adding minimal formatting")
            # Apply basic formatting - convert paragraph breaks to double newlines
            md_text = re.sub(r"\n{3,}", "\n\n", text)
            # Look for potential headers (ALL CAPS lines or short lines followed by blank lines)
            md_text = re.sub(r"^([A-Z][A-Z\s]{10,}[A-Z])$", r"## \1", md_text, flags=re.MULTILINE)
            # Apply sanitization and improvements
            md_text = self._sanitize(md_text)
            md_text = self._improve(md_text)
            
        # Ensure we have clean output
        md_text = md_text.strip()
        
        # Apply line wrapping if requested
        if max_line_length > 0:
            try:
                wrapped_lines = []
                for line in md_text.split("\n"):
                    # Don't wrap lines that look like Markdown constructs
                    if (line.startswith("#") or line.startswith(">") or 
                        line.startswith("-") or line.startswith("*") or 
                        line.startswith("```") or re.match(r"^\d+\.\s", line) or
                        line.strip() == "" or line.startswith("|")):
                        wrapped_lines.append(line)
                    else:
                        wrapped_lines.append(textwrap.fill(
                            line, 
                            width=max_line_length,
                            break_long_words=False,
                            break_on_hyphens=False
                        ))
                md_text = "\n".join(wrapped_lines)
            except Exception as e:
                self.logger.error(f"Error during line wrapping: {e}")
                # Continue with unwrapped text

        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "markdown_text": md_text,
            "original_content_type": input_type,
            "was_html": was_html,
            "extraction_method_used": extraction_method_used,
            "processing_time": processing_time,
        }

    @tool(
        name="detect_content_type",
        description="Detect if text is primarily HTML, Markdown, code, or plain text",
    )
    @with_tool_metrics
    @with_error_handling
    async def detect_content_type(self, text: str) -> Dict[str, Any]:
        """
        Detect the primary content type of a text string.

        Uses pattern recognition and heuristics to determine if the provided text is:
        - HTML: Contains HTML tags, doctype declarations, or common HTML markers
        - Markdown: Contains Markdown syntax for headings, lists, links, etc.
        - Code: Contains code-like syntax (functions, statements, etc.)
        - Plain text: No significant markup of any kind detected

        Args:
            text: The text content to analyze

        Returns:
            Dictionary with detection results:
            {
                "success": bool,
                "content_type": str,  # "html", "markdown", "code", or "text"
                "confidence": float,  # 0.0-1.0 confidence score 
                "detection_criteria": List[str],  # Criteria that led to the classification
                "processing_time": float  # Seconds
            }
        """
        t0 = time.time()
        if not text or not isinstance(text, str):
            return {
                "success": False,
                "error": "Input text must be a non-empty string",
                "error_code": "INVALID_INPUT"
            }

        # Try to use Magika for content detection if available
        magika_result = None
        try:
            # Import magika only when needed to avoid startup dependency
            from magika import Magika
            
            # Create a Magika instance and detect content type
            magika = Magika()
            input_bytes = text.encode('utf-8')
            magika_result = magika.identify_bytes(input_bytes)
            
            # Get content type label from Magika
            detected_type = magika_result.output.label
            confidence = magika_result.score
            
            self.logger.debug(f"Magika detected content type: {detected_type} with confidence {confidence}")
            
            # Map Magika content types to our simpler categories
            if detected_type in ["text/html", "application/xhtml+xml"]:
                magika_type = "html"
                magika_confidence = confidence
            elif detected_type in ["text/markdown", "text/x-markdown"]:
                magika_type = "markdown"
                magika_confidence = confidence
            elif detected_type.startswith("text/x-") or detected_type in ["application/x-javascript", "application/json"]:
                magika_type = "code"
                magika_confidence = confidence
            elif detected_type == "text/plain":
                magika_type = "text"
                magika_confidence = confidence
            else:
                magika_type = None  # Unknown or unmapped type
                magika_confidence = 0.0
                
            if magika_type and magika_confidence > 0.85:
                # High confidence Magika result, return immediately
                return {
                    "success": True,
                    "content_type": magika_type,
                    "confidence": magika_confidence,
                    "detection_criteria": [f"Magika detection: {detected_type}"],
                    "detection_method": "magika",
                    "magika_details": {
                        "raw_type": detected_type,
                        "raw_confidence": confidence
                    },
                    "processing_time": time.time() - t0
                }
                
        except (ImportError, Exception) as e:
            # Handle both ImportError and any runtime errors
            if isinstance(e, ImportError):
                self.logger.debug("Magika not available, falling back to heuristic detection")
            else:
                self.logger.warning(f"Error using Magika for detection: {str(e)}, falling back to heuristic detection")
            magika_type = None
            magika_confidence = 0.0

        # Take a sample of text for efficiency with very large inputs
        # Using both beginning and middle portions as they often contain different indicators
        sample_size = 2000  # characters
        if len(text) > sample_size * 3:
            half_sample = sample_size // 2
            beginning = text[:sample_size] 
            middle = text[len(text)//2 - half_sample:len(text)//2 + half_sample]
            sample = beginning + "\n" + middle
        else:
            sample = text[:min(sample_size * 2, len(text))]
            
        # Content type detection criteria (weighted)
        html_criteria = []
        markdown_criteria = []
        code_criteria = []
        text_criteria = []
        
        # HTML Detection - Stronger checks first
        if re.search(r"<!DOCTYPE\s+html", sample, re.IGNORECASE):
            html_criteria.append("DOCTYPE declaration found")
            
        if re.search(r"<html\b[^>]*>", sample, re.IGNORECASE):
            html_criteria.append("<html> tag found")
            
        if re.search(r"<head\b[^>]*>", sample, re.IGNORECASE):
            html_criteria.append("<head> tag found")
            
        if re.search(r"<body\b[^>]*>", sample, re.IGNORECASE):
            html_criteria.append("<body> tag found")
            
        # Look for common HTML structural elements
        if re.search(r"<div\b[^>]*>|<span\b[^>]*>|<p\b[^>]*>|<a\b[^>]*>", sample, re.IGNORECASE):
            html_criteria.append("Common HTML elements found")
            
        # Look for meta tags, script tags, or style tags
        if re.search(r"<meta\b[^>]*>|<script\b[^>]*>|<style\b[^>]*>", sample, re.IGNORECASE):
            html_criteria.append("Meta, script, or style tags found")
            
        # Check for HTML comments
        if re.search(r"<!--.*?-->", sample, re.DOTALL):
            html_criteria.append("HTML comments found")
            
        # Check for HTML entities
        if re.search(r"&[a-z]+;|&#\d+;", sample):
            html_criteria.append("HTML entities found")
            
        # General tag density check
        tag_matches = re.findall(r"<[a-z][a-z0-9]*\b[^>]*>", sample, re.IGNORECASE)
        if len(tag_matches) > 5:
            html_criteria.append(f"High tag density ({len(tag_matches)} tags found)")
        
        # Check closing tags too
        closing_tags = re.findall(r"</[a-z][a-z0-9]*>", sample, re.IGNORECASE)
        if closing_tags:
            html_criteria.append("HTML closing tags found")
        
        # Markdown Detection
        # Headers
        if re.search(r"^#{1,6}\s+.+$", sample, re.MULTILINE):
            markdown_criteria.append("Markdown headers found")
        
        # Lists (unordered & ordered)
        if re.search(r"^[\s]*[-*+]\s+.+$", sample, re.MULTILINE):
            markdown_criteria.append("Markdown unordered lists found")
            
        if re.search(r"^[\s]*\d+\.\s+.+$", sample, re.MULTILINE):
            markdown_criteria.append("Markdown ordered lists found")
        
        # Blockquotes
        if re.search(r"^>\s+.+$", sample, re.MULTILINE):
            markdown_criteria.append("Markdown blockquotes found")
        
        # Inline formatting (bold, italic)
        if re.search(r"\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_", sample):
            markdown_criteria.append("Markdown emphasis formatting found")
        
        # Links
        if re.search(r"\[.+?\]\(.+?\)|\[.+?\]\[.+?\]|\[.+?\]:", sample):
            markdown_criteria.append("Markdown links found")
        
        # Code blocks
        if re.search(r"```[^`]*```|`[^`]+`", sample):
            markdown_criteria.append("Markdown code blocks or inline code found")
        
        # Horizontal rules
        if re.search(r"^(---|\*\*\*|___)\s*$", sample, re.MULTILINE):
            markdown_criteria.append("Markdown horizontal rules found")
            
        # Markdown tables
        if re.search(r"^\|[^|]+\|[^|]+\|.*$\n^\|-+\|-+\|", sample, re.MULTILINE):
            markdown_criteria.append("Markdown tables found")
        
        # Code Detection
        # Common programming language keywords
        code_keywords = r"\b(function|class|def|var|const|let|import|from|package|if|else|for|while|return|interface|implements|extends|public|private|static)\b"
        if re.search(code_keywords, sample):
            code_criteria.append("Programming language keywords found")
        
        # Common syntax patterns
        if re.search(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*{", sample):  # Function definitions
            code_criteria.append("Function definition patterns found")
            
        if re.search(r"[\w\.]+\s*\([^)]*\);?", sample):  # Function calls
            code_criteria.append("Function call patterns found")
        
        # Variable assignments
        if re.search(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=]", sample):
            code_criteria.append("Variable assignment patterns found")
        
        # Common language syntax
        if re.search(r"[\{\}\[\]<>]", sample) and len(re.findall(r"[\{\}\[\]<>]", sample)) > 10:
            code_criteria.append("High density of code-specific brackets and braces")
            
        # Language-specific imports/includes
        if re.search(r"import\s+[\w\.]+|from\s+[\w\.]+\s+import|#include\s+[<\"][\w\.]+[>\"]", sample):
            code_criteria.append("Import/include statements found")
            
        # Check for indentation patterns common in code
        indentation_lines = re.findall(r"^[ \t]+\S+.*$", sample, re.MULTILINE)
        if len(indentation_lines) > 5:
            code_criteria.append("Consistent code indentation patterns found")
        
        # Plain Text Detection (absence of other markers)
        # Check for paragraphs with punctuation but no markup
        text_paragraphs = re.findall(r"^[A-Z][^<>#`*_\[\]]+[.!?]$", sample, re.MULTILINE)
        if text_paragraphs and len(text_paragraphs) > 3:
            text_criteria.append("Multiple plain text paragraphs found")
            
        # Check for natural language patterns
        if re.search(r"\b(the|and|that|this|with|from|have|for)\b", sample, re.IGNORECASE) and not html_criteria and not code_criteria:
            text_criteria.append("Natural language common words found")
            
        # Check for sentence structures
        sentences = re.findall(r"[A-Z][^.!?]*[.!?]", sample)
        if len(sentences) > 5 and not html_criteria and not markdown_criteria and not code_criteria:
            text_criteria.append("Multiple natural language sentences found")
        
        # Decision logic - calculate confidence scores
        html_score = min(1.0, len(html_criteria) * 0.2)
        markdown_score = min(1.0, len(markdown_criteria) * 0.25)
        code_score = min(1.0, len(code_criteria) * 0.25)
        text_score = min(1.0, len(text_criteria) * 0.33)
        
        # HTML trumps all other scores if it has clear structural elements
        if html_score > 0.4 and any(["DOCTYPE" in c or "<html" in c or "<head" in c or "<body" in c for c in html_criteria]):
            html_score = max(html_score, 0.8)
        
        # Handle the case where text is both Markdown and code (e.g., README file with code examples)
        if markdown_score > 0.5 and code_score > 0.5:
            # If it has markdown-specific formatting, prefer markdown
            if any(["headers" in c or "blockquotes" in c or "tables" in c for c in markdown_criteria]):
                markdown_score = max(markdown_score, code_score + 0.1)
            else:
                code_score = max(code_score, markdown_score)
                
        # If content has HTML tags but also markdown features, it might be generated HTML
        if html_score > 0 and markdown_score > 0:
            if html_score > 0.6:  # Strong HTML indicators
                html_score = max(html_score, markdown_score + 0.2)
            
        # Get the highest score
        scores = {
            "html": html_score,
            "markdown": markdown_score,
            "code": code_score,
            "text": text_score
        }
        
        # If we have a Magika result but it wasn't confident enough to return immediately,
        # combine it with our heuristic scores
        if magika_type and magika_confidence > 0.5:
            self.logger.debug(f"Combining Magika result ({magika_type}, {magika_confidence}) with heuristic scores")
            # Boost the corresponding score based on Magika's confidence
            current_score = scores.get(magika_type, 0)
            boost_amount = 0.2 * magika_confidence  # Adjust the weight as needed
            scores[magika_type] = min(1.0, current_score + boost_amount)
            
        primary_type = max(scores, key=scores.get)
        confidence = scores[primary_type]
        
        # Return all criteria that supported the winning type
        criteria_map = {
            "html": html_criteria,
            "markdown": markdown_criteria,
            "code": code_criteria,
            "text": text_criteria
        }
        
        detection_criteria = criteria_map[primary_type]
        # Add Magika criteria if it matched our final type
        if magika_type == primary_type:
            detection_criteria.append(f"Magika detection confirmation ({magika_result.output.label})")
        
        processing_time = time.time() - t0
        
        result = {
            "success": True,
            "content_type": primary_type,
            "confidence": confidence,
            "detection_criteria": detection_criteria,
            "all_scores": scores,
            "processing_time": processing_time
        }
        
        # Include Magika details if available
        if magika_result:
            result["magika_details"] = {
                "type": magika_result.output.label,
                "confidence": magika_result.score,  # Access score from magika_result, not from output
                "matched_primary_type": magika_type == primary_type
            }
            
        return result

    @tool(
        name="batch_format_texts",
        description="Format multiple texts (HTML/plain) to Markdown in parallel",
    )
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
        """Applies 'clean_and_format_text_as_markdown' to a list of texts concurrently."""
        if not texts or not isinstance(texts, list):
            raise ToolInputError("Input must be a non-empty list of strings", param_name="texts")
        if not all(isinstance(t, str) for t in texts):
            raise ToolInputError(
                "All items in the 'texts' list must be strings", param_name="texts"
            )
        if max_concurrency <= 0:
            self.logger.warning("max_concurrency must be positive, setting to 1.")
            max_concurrency = 1

        sem = asyncio.Semaphore(max_concurrency)
        tasks = []

        async def _process_one(idx: int, txt: str):
            async with sem:
                self.logger.debug(f"Starting batch formatting for text index {idx}")
                result_dict = {"original_index": idx}
                try:
                    res = await self.clean_and_format_text_as_markdown(
                        text=txt,
                        force_markdown_conversion=force_markdown_conversion,
                        extraction_method=extraction_method,
                        preserve_tables=preserve_tables,
                        preserve_links=preserve_links,
                        preserve_images=preserve_images,
                        max_line_length=0,  # Batch usually doesn't need wrapping
                    )
                    result_dict.update(res)
                    result_dict["success"] = bool(res.get("success", False))
                    if result_dict["success"]:
                        self.logger.debug(f"Successfully batch formatted text index {idx}")

                except ToolInputError as e_input:
                    self.logger.warning(f"Input error for text index {idx}: {e_input}")
                    result_dict.update(
                        {
                            "error": str(e_input),
                            "success": False,
                            "error_type": "ToolInputError",
                            "error_code": e_input.error_code,
                        }
                    )
                except ToolError as e_tool:
                    self.logger.warning(
                        f"Processing error for text index {idx}: {e_tool.error_code} - {str(e_tool)}"
                    )
                    result_dict.update(
                        {
                            "error": str(e_tool),
                            "success": False,
                            "error_code": e_tool.error_code,
                            "error_type": "ToolError",
                        }
                    )
                except Exception as e:
                    self.logger.error(
                        f"Unexpected error processing text index {idx}: {e}", exc_info=True
                    )
                    result_dict.update(
                        {"error": str(e), "success": False, "error_type": "Exception"}
                    )
                return result_dict

        tic = time.perf_counter()
        self.logger.info(
            f"Starting batch formatting for {len(texts)} texts with concurrency {max_concurrency}..."
        )
        for i, t in enumerate(texts):
            tasks.append(_process_one(i, t))

        all_results = await asyncio.gather(*tasks)
        toc = time.perf_counter()
        self.logger.info(f"Batch formatting completed in {toc - tic:.3f}s")

        all_results.sort(key=lambda r: r.get("original_index", -1))
        final_results = []
        success_count = 0
        failure_count = 0
        for r in all_results:
            if r.get("success"):
                success_count += 1
            else:
                failure_count += 1
            r.pop("original_index", None)
            final_results.append(r)

        return {
            "results": final_results,
            "total_processing_time": round(toc - tic, 3),
            "success_count": success_count,
            "failure_count": failure_count,
            "success": True,  # Batch operation itself succeeded
        }

    @tool(
        name="optimize_markdown_formatting",
        description="Clean up and standardize existing Markdown text",
    )
    @with_tool_metrics
    @with_error_handling
    async def optimize_markdown_formatting(
        self,
        markdown: str,
        normalize_headings: bool = False,
        fix_lists: bool = True,
        fix_links: bool = True,
        add_line_breaks: bool = True,
        compact_mode: bool = False,
        max_line_length: int = 0,
    ) -> Dict[str, Any]:
        """
        Clean up and standardize Markdown formatting.

        Args:
            markdown: Input Markdown text (will auto-convert from HTML if detected)
            normalize_headings: If True, fix heading levels to be sequential
            fix_lists: If True, standardize lists (spacing and markers)
            fix_links: If True, clean up link formatting
            add_line_breaks: If True, add breaks between sections
            compact_mode: If True, minimize whitespace (overrides add_line_breaks)
            max_line_length: Line wrapping (0 = disable wrapping)

        Returns:
            Dictionary with optimization results:
            {
                "success": bool,
                "optimized_markdown": str,
                "changes_summary": str,
                "processing_time": float,
            }
        """
        t0 = time.time()
        # First, detect the content type
        content_type_result = await self.detect_content_type(markdown)
        input_type = content_type_result.get("content_type", "unknown")
        
        # If input is HTML, convert it to Markdown first
        actual_markdown = markdown
        conversion_note = ""
        
        if input_type == "html":
            self.logger.info("Input appears to be HTML, converting to Markdown first")
            conversion_note = "⚠️ Input was detected as HTML and automatically converted to Markdown. "
            
            # Use clean_and_format_text_as_markdown to convert HTML to Markdown
            conversion_result = await self.clean_and_format_text_as_markdown(
                text=markdown,
                force_markdown_conversion=True,
                extraction_method="auto",  # Try to extract main content
                preserve_tables=True,
                preserve_links=True
            )
            
            if conversion_result.get("success", False):
                actual_markdown = conversion_result.get("markdown_text", "")
                self.logger.success("Successfully converted HTML to Markdown")
            else:
                error_msg = conversion_result.get("error", "Unknown conversion error")
                self.logger.error(f"Failed to convert HTML to Markdown: {error_msg}")
                return {
                    "success": False,
                    "error": f"Input appears to be HTML but conversion to Markdown failed: {error_msg}",
                    "error_code": "FORMAT_CONVERSION_FAILED"
                }
        elif input_type != "markdown":
            self.logger.warning(f"Input doesn't appear to be Markdown (detected as {input_type})")
            conversion_note = f"⚠️ Input was detected as {input_type}, not Markdown. Results may be inconsistent. "

        # Implement markdown cleanup logic
        try:
            # Our original optimization logic
            optimized = actual_markdown
            changes = []

            if normalize_headings:
                # Uses a regex with a callback to adjust heading levels
                # Helper function to replace headings
                def replace_heading(match):
                    # Get the captured heading level (number of # characters)
                    level = len(match.group(1))
                    # Make sure it starts at h1 (one #) and increment from there
                    new_level = min(6, level)  # Cap at h6 (markdown standard)
                    # Return the replacement with the appropriate number of # chars
                    return "#" * new_level + match.group(2)

                # Find headings with regex and replace using the callback
                optimized = re.sub(
                    r"^(#{1,6})(\s.+)$", replace_heading, optimized, flags=re.MULTILINE
                )
                changes.append("Normalized heading levels")

            # Standardize list formatting
            if fix_lists:
                # Unordered lists (consistent markers and spacing)
                optimized = re.sub(
                    r"^([ \t]*)[-*+]([^ ])", r"\1- \2", optimized, flags=re.MULTILINE
                )
                # Ordered lists (add space after number if missing)
                optimized = re.sub(
                    r"^([ \t]*)\d+\.([^ ])", r"\1\\d+. \2", optimized, flags=re.MULTILINE
                )
                changes.append("Standardized list formatting")

            # Fix link formatting
            if fix_links:
                # Remove spaces between link components
                optimized = re.sub(r"\] \(", "](", optimized)
                changes.append("Fixed link formatting")

            # Line break management
            if compact_mode:
                # Compact mode: remove extra blank lines (more than 1)
                optimized = re.sub(r"\n{3,}", "\n\n", optimized)
                # Remove trailing whitespace
                optimized = re.sub(r"[ \t]+$", "", optimized, flags=re.MULTILINE)
                changes.append("Applied compact formatting (minimized whitespace)")
            elif add_line_breaks:
                # Add breaks between sections (after headings, before lists, etc.)
                # Ensure blank line after headings
                optimized = re.sub(r"(^#{1,6}[^\n]+)\n([^#\n])", r"\1\n\n\2", optimized, flags=re.MULTILINE)
                # Ensure blank line before lists
                optimized = re.sub(r"([^\n])\n([ \t]*[-*+])", r"\1\n\n\2", optimized, flags=re.MULTILINE)
                optimized = re.sub(r"([^\n])\n([ \t]*\d+\.)", r"\1\n\n\2", optimized, flags=re.MULTILINE)
                changes.append("Added line breaks for readability")

            # Line wrapping (if requested)
            if max_line_length > 0:
                # Split into paragraphs
                paragraphs = re.split(r"\n{2,}", optimized)
                wrapped_paragraphs = []
                
                # Helper to check if a line should be exempt from wrapping
                def should_not_wrap(line):
                    return (line.startswith("```") or
                            line.startswith("    ") or  # code indentation 
                            line.startswith("#") or  # headings
                            line.startswith(">") or  # blockquotes
                            line.startswith("- ") or  # list items
                            line.startswith("* ") or
                            line.startswith("+ ") or
                            re.match(r"^\d+\. ", line) or  # ordered lists
                            line.startswith("|"))  # table rows
                
                for para in paragraphs:
                    lines = para.split("\n")
                    current_block = []
                    
                    for line in lines:
                        # Don't wrap certain lines
                        if should_not_wrap(line):
                            # First flush any accumulated block
                            if current_block:
                                wrapped_text = textwrap.fill(
                                    " ".join(current_block),
                                    width=max_line_length,
                                    break_long_words=False,
                                    break_on_hyphens=False
                                )
                                wrapped_paragraphs.append(wrapped_text)
                                current_block = []
                            # Add the non-wrapped line
                            wrapped_paragraphs.append(line)
                        else:
                            current_block.append(line)
                    
                    # Wrap any remaining text
                    if current_block:
                        wrapped_text = textwrap.fill(
                            " ".join(current_block),
                            width=max_line_length,
                            break_long_words=False,
                            break_on_hyphens=False
                        )
                        wrapped_paragraphs.append(wrapped_text)
                
                optimized = "\n\n".join(wrapped_paragraphs)
                changes.append(f"Wrapped lines at {max_line_length} characters")

            # Clean up final result
            optimized = optimized.strip()
                        
            # Final general improvements for any markdown
            optimized = self._improve(optimized)
            changes.append("Applied general markdown improvements")

            return {
                "success": True,
                "optimized_markdown": optimized,
                "changes_summary": conversion_note + ", ".join(changes),
                "processing_time": time.time() - t0,
                "original_content_type": input_type
            }
        except Exception as e:
            self.logger.error(f"Markdown optimization failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    ###############################################################################
    # Table Extraction Tool                                                       #
    ###############################################################################
    @tool(
        name="extract_tables",
        description="Extract tables from a document (PDF/Office) into CSV, JSON, or Pandas format (Requires Docling Strategy)",
    )
    @with_tool_metrics
    @with_error_handling
    async def extract_tables(
        self,
        document_path: str,
        *,
        table_mode: str = "csv",
        output_dir: Optional[str] = None,
        accelerator_device: str = "auto",
        num_threads: int = 4,
    ) -> Dict[str, Any]:
        """Extracts tables found in a document using Docling and returns them.

        NOTE: This tool currently *requires* the 'docling' extraction strategy implicitly.
              It does not work with 'ocr' or 'direct_text' strategies yet. Table extraction
              from OCR'd text might be added in the future.

        Args:
            document_path: Path to the document (e.g., PDF, DOCX) containing tables.
            table_mode: Format for output tables ('csv', 'json', 'pandas').
            output_dir: If specified, saves each extracted table to a file in this directory.
            accelerator_device: Device for Docling backend ('auto', 'cpu', 'cuda', 'mps').
            num_threads: Number of threads for Docling backend.

        Returns:
            Dictionary containing:
            - tables: List of tables in the specified format.
            - saved_files: List of paths to saved files (if output_dir).
            - success: Boolean indicating success.
        """
        if not self._docling_available:
            raise ToolError(
                "DEPENDENCY_MISSING",
                details={
                    "dependency": "docling",
                    "feature": "Table Extraction (extract_tables tool)",
                },
            )

        valid_modes = {"csv", "json", "pandas"}
        table_mode = table_mode.lower()
        if table_mode not in valid_modes:
            raise ToolInputError(
                f"table_mode must be one of {', '.join(valid_modes)}",
                param_name="table_mode",
                provided_value=table_mode,
            )
        if table_mode == "pandas" and not self._pandas_available:
            raise ToolError(
                "DEPENDENCY_MISSING",
                details={"dependency": "pandas", "feature": "extract_tables(mode='pandas')"},
            )

        self.logger.info(
            f"Starting Docling-based table extraction from {document_path}, mode='{table_mode}'"
        )

        doc_obj: Optional[DoclingDocument] = None
        doc_path = self._ocr_validate_file_path(document_path)  # Reuse validator

        try:
            # --- Step 1: Convert document using Docling to get structured data ---
            device = self._ACCEL_MAP[accelerator_device.lower()]
            conv = self._get_docling_converter(device, num_threads)
            loop = asyncio.get_running_loop()
            with self._span("docling_table_conversion"):
                result = await loop.run_in_executor(None, conv.convert, doc_path)

            if result and result.document:
                doc_obj = result.document
                self.logger.info("Docling conversion successful for table extraction.")
            else:
                raise ToolError(
                    "CONVERSION_FAILED",
                    details={
                        "document_path": str(doc_path),
                        "reason": "Docling converter returned empty result",
                    },
                )

        except ToolError as te:
            raise te
        except Exception as e:
            self.logger.error(
                f"Error during Docling conversion for table extraction: {e}", exc_info=True
            )
            raise ToolError(
                "CONVERSION_FAILED", details={"document_path": str(doc_path), "error": str(e)}
            ) from e

        if not doc_obj:  # Should be caught above, but safety check
            return {
                "tables": [],
                "saved_files": [],
                "success": False,
                "error": "Document object unavailable after conversion.",
            }

        # --- Step 2: Extract tables from the Doc object ---
        tables_raw_data: List[List[List[str]]] = []
        try:
            with self._span("docling_table_extraction"):
                for page in doc_obj.pages:
                    if (
                        hasattr(page, "content")
                        and page.content
                        and callable(getattr(page.content, "has_tables", None))
                        and page.content.has_tables()
                    ):
                        page_tables = page.content.get_tables()
                        if page_tables:
                            for tbl in page_tables:
                                if isinstance(tbl, list) and all(
                                    isinstance(row, list) for row in tbl
                                ):
                                    sanitized_tbl = [
                                        [str(cell) if cell is not None else "" for cell in row]
                                        for row in tbl
                                    ]
                                    tables_raw_data.append(sanitized_tbl)
                                else:
                                    self.logger.warning(
                                        f"Skipping malformed table structure found on page {getattr(page, 'page_idx', 'unknown')}: {type(tbl)}"
                                    )
        except Exception as e:
            self.logger.error(f"Error accessing tables from Docling object: {e}", exc_info=True)
            # Continue if possible

        if not tables_raw_data:
            self.logger.warning(f"No tables found or extracted from {document_path} using Docling.")
            return {"tables": [], "saved_files": [], "success": True}

        self.logger.info(f"Extracted {len(tables_raw_data)} tables via Docling.")

        # --- Step 3: Format tables and optionally save ---
        output_tables: List[Any] = []
        saved_files: List[str] = []
        output_dir_path = Path(output_dir) if output_dir else None
        if output_dir_path:
            output_dir_path.mkdir(parents=True, exist_ok=True)

        with self._span("table_formatting_saving"):
            for i, raw_table in enumerate(tables_raw_data):
                processed_table: Any = None
                file_ext = ""
                save_content: Union[str, Any] = ""
                try:
                    if table_mode == "csv":
                        output = StringIO()
                        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
                        writer.writerows(raw_table)
                        processed_table = output.getvalue()
                        file_ext = "csv"
                        save_content = processed_table
                    elif table_mode == "json":
                        processed_table = raw_table
                        file_ext = "json"
                        save_content = self._json(processed_table)
                    elif table_mode == "pandas":
                        df = pd.DataFrame(raw_table)  # type: ignore
                        if not df.empty and len(df) > 1:
                            first_row = df.iloc[0]
                            # Basic header check: more than half non-numeric?
                            is_header = (
                                sum(
                                    1
                                    for cell in first_row
                                    if not str(cell).replace(".", "", 1).strip("-").isdigit()
                                )
                                > len(first_row) / 2
                            )
                            if is_header:
                                df.columns = first_row
                                df = df[1:].reset_index(drop=True)
                        processed_table = df
                        file_ext = "csv"
                        save_content = processed_table

                    output_tables.append(processed_table)

                    if output_dir_path and file_ext:
                        base_name = Path(document_path).stem
                        fp = output_dir_path / f"{base_name}_table_{i + 1}.{file_ext}"
                        try:
                            if isinstance(save_content, str):
                                fp.write_text(save_content, encoding="utf-8")
                            elif isinstance(save_content, pd.DataFrame):
                                save_content.to_csv(fp, index=False, encoding="utf-8")  # type: ignore
                            saved_files.append(str(fp))
                            self.logger.debug(f"Saved table {i + 1} to {fp}")
                        except Exception as e_save:
                            self.logger.error(
                                f"Failed to save table {i + 1} to {fp}: {e_save}", exc_info=True
                            )
                except Exception as e_format:
                    self.logger.error(
                        f"Failed to format table {i} into '{table_mode}': {e_format}", exc_info=True
                    )

        self.logger.info(
            f"Successfully processed {len(output_tables)} tables into '{table_mode}' format."
        )
        return {"tables": output_tables, "saved_files": saved_files, "success": True}

    ###############################################################################
    # Document Analysis Tools                                                     #
    ###############################################################################
    # [Analysis tools: identify_sections, extract_entities, generate_qa_pairs, ]
    # [summarize_document, classify_document, extract_metrics, flag_risks, ]
    # [canonicalise_entities remain largely unchanged from the original code. ]
    # [They operate on text input, which can come from any conversion strategy.]

    @tool(
        name="identify_sections",
        description="Identify logical sections based on headings and structure using regex",
    )
    @with_tool_metrics
    @with_error_handling
    async def identify_sections(self, document: str) -> Dict[str, Any]:
        """Identifies logical sections in a document using regex patterns for headings."""
        if not document or not isinstance(document, str):
            self.logger.warning("identify_sections called with empty or invalid input.")
            return {"sections": [], "success": True}

        sections_found: List[Dict[str, Any]] = []
        last_section_end = 0

        if not hasattr(self, "_BOUND_RX") or not isinstance(self._BOUND_RX, re.Pattern):
            raise ToolError(
                "INITIALIZATION_ERROR",
                details={"reason": "Section boundary regex _BOUND_RX not compiled"},
            )

        try:
            matches = list(self._BOUND_RX.finditer(document))
            if not matches:
                self.logger.info(
                    "No regex-based section boundaries found. Treating document as single section."
                )
                if document.strip():
                    sections_found.append(
                        {
                            "title": "Main Content",
                            "text": document.strip(),
                            "position": 0,
                            "start_char": 0,
                            "end_char": len(document),
                        }
                    )
            else:
                self.logger.info(
                    f"Found {len(matches)} potential section boundaries based on regex."
                )
                first_match_start = matches[0].start()
                if first_match_start > 0:
                    initial_text = document[last_section_end:first_match_start].strip()
                    if initial_text:
                        sections_found.append(
                            {
                                "title": "Introduction",
                                "text": initial_text,
                                "position": 0,
                                "start_char": last_section_end,
                                "end_char": first_match_start,
                            }
                        )
                        last_section_end = first_match_start

                for i, match in enumerate(matches):
                    title_raw = match.group(0).strip()
                    title_start_char = match.start()
                    title_end_char = match.end()
                    section_content_start = title_end_char
                    section_content_end = (
                        matches[i + 1].start() if i < len(matches) - 1 else len(document)
                    )
                    section_text = document[section_content_start:section_content_end].strip()

                    section_title = title_raw
                    if hasattr(self, "_CUSTOM_SECT_RX") and isinstance(self._CUSTOM_SECT_RX, list):
                        for pat, label in self._CUSTOM_SECT_RX:
                            if isinstance(pat, re.Pattern) and pat.search(title_raw):
                                section_title = label
                                self.logger.debug(
                                    f"Applied custom label '{label}' to section '{title_raw}'."
                                )
                                break
                    else:
                        self.logger.warning("_CUSTOM_SECT_RX not initialized correctly.")

                    if section_text:
                        sections_found.append(
                            {
                                "title": section_title,
                                "text": section_text,
                                "position": len(sections_found),
                                "start_char": title_start_char,
                                "end_char": section_content_end,
                            }
                        )
                    else:
                        self.logger.debug(f"Skipping section '{section_title}' (no content).")
                    last_section_end = section_content_end

        except Exception as e:
            self.logger.error(f"Error during section identification: {e}", exc_info=True)
            raise ToolError("SECTION_IDENTIFICATION_FAILED", details={"error": str(e)}) from e

        return {"sections": sections_found, "success": True}

    @tool(
        name="extract_entities",
        description="Extract named entities (PERSON, ORG, etc.) using an LLM",
    )
    @with_tool_metrics
    @with_error_handling
    async def extract_entities(
        self, document: str, entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extracts named entities from document text using an LLM prompt."""
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")

        max_context = 3800  # Limit context for LLM
        context = document[:max_context] + ("\n..." if len(document) > max_context else "")
        if len(document) > max_context:
            self.logger.warning(
                f"Document truncated to ~{max_context} chars for entity extraction."
            )

        entity_focus = (
            f"Extract only these entity types: {', '.join(entity_types)}."
            if entity_types
            else "Extract common named entity types (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, EVENT)."
        )

        prompt = f"""Analyze the following text and extract named entities.
{entity_focus}
Output ONLY a valid JSON object where keys are uppercase entity types (e.g., "PERSON") and values are lists of unique entity strings found for that type. Do not include explanations or markdown formatting.

Text:
\"\"\"
{context}
\"\"\"

JSON Output:
"""
        self.logger.info(
            f"Requesting entity extraction from LLM. Focus: {entity_types or 'common'}"
        )
        llm_response_raw = ""
        try:
            llm_response_raw = await self._llm(
                prompt=prompt, max_tokens=1500, temperature=0.1
            )
            self.logger.debug(f"LLM raw response for entities:\n{llm_response_raw}")

            # Parse JSON robustly
            json_str = llm_response_raw
            if "```" in json_str:
                json_match = re.search(r"```(?:json)?\s*([\s\S]+)\s*```", json_str)
                json_str = json_match.group(1).strip() if json_match else json_str
            start_brace = json_str.find("{")
            end_brace = json_str.rfind("}")
            if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
                json_str = json_str[start_brace : end_brace + 1]
            else:
                self.logger.warning(
                    "Could not find JSON object boundaries in LLM response for entities."
                )
                json_str = "{}"  # Default empty

            entities_dict = json.loads(json_str)
            if not isinstance(entities_dict, dict):
                raise ValueError("LLM response is not a JSON object.")

            validated_entities: Dict[str, List[str]] = {}
            for key, value in entities_dict.items():
                entity_type = str(key).upper().strip()
                if not entity_type:
                    continue
                sanitized_values: Set[str] = set()  # Use set for uniqueness directly
                items_to_process = (
                    value if isinstance(value, list) else [value]
                )  # Handle single string value
                for item in items_to_process:
                    text_val = None
                    if isinstance(item, str) and item.strip():
                        text_val = item.strip()
                    elif (
                        isinstance(item, dict)
                        and isinstance(item.get("text"), str)
                        and item["text"].strip()
                    ):
                        text_val = item["text"].strip()
                    if text_val:
                        sanitized_values.add(text_val)

                if sanitized_values:
                    validated_entities[entity_type] = sorted(list(sanitized_values))

            self.logger.info(
                f"Successfully extracted entities for types: {list(validated_entities.keys())}"
            )
            return {
                "entities": validated_entities,
                "success": True,
                "raw_llm_response": llm_response_raw,
            }

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response for entities as JSON: {e}")
            self.logger.debug(f"Raw LLM response causing parse error:\n{llm_response_raw}")
            return {
                "entities": {},
                "error": f"Failed to parse LLM response: {e}",
                "raw_llm_response": llm_response_raw,
                "success": False,
            }
        except ToolError as e:
            self.logger.error(f"LLM call failed during entity extraction: {e}", exc_info=True)
            return {
                "entities": {},
                "error": f"LLM call failed: {str(e)}",
                "raw_llm_response": llm_response_raw,
                "success": False,
                "error_code": e.error_code,
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during entity extraction: {e}", exc_info=True)
            return {
                "entities": {},
                "error": f"Unexpected error: {e}",
                "raw_llm_response": llm_response_raw,
                "success": False,
            }

    @tool(
        name="generate_qa_pairs",
        description="Generate question-answer pairs from the document using an LLM",
    )
    @with_tool_metrics
    @with_error_handling
    async def generate_qa_pairs(self, document: str, num_questions: int = 5) -> Dict[str, Any]:
        """Generates question-answer pairs based on the document content using an LLM."""
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")
        if not isinstance(num_questions, int) or num_questions <= 0:
            raise ToolInputError(
                "num_questions must be a positive integer",
                param_name="num_questions",
                provided_value=num_questions,
            )

        max_context = 3800
        context = document[:max_context] + ("\n..." if len(document) > max_context else "")
        if len(document) > max_context:
            self.logger.warning(f"Document truncated to ~{max_context} chars for QA generation.")

        prompt = f"""Based ONLY on the information in the following text, generate exactly {num_questions} relevant and insightful question-answer pairs.
Questions should be answerable directly from the text. Answers should be factual and concise, summarizing or quoting the text.
Format the output ONLY as a valid JSON list of objects. Each object must have keys "question" (string) and "answer" (string).
Do not include explanations or markdown formatting.

Text:
\"\"\"
{context}
\"\"\"

JSON Output:
"""
        self.logger.info(f"Requesting {num_questions} QA pairs from LLM.")
        llm_response_raw = ""
        try:
            llm_max_tokens = num_questions * 150 + 200  # Base + buffer
            llm_response_raw = await self._llm(
                prompt=prompt,
                max_tokens=llm_max_tokens,
                temperature=0.4,
            )
            self.logger.debug(f"LLM raw response for QA pairs:\n{llm_response_raw}")

            # Parse JSON list robustly
            json_str = llm_response_raw
            if "```" in json_str:
                json_match = re.search(r"```(?:json)?\s*([\s\S]+)\s*```", json_str)
                json_str = json_match.group(1).strip() if json_match else json_str
            start_bracket = json_str.find("[")
            end_bracket = json_str.rfind("]")
            if start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket:
                json_str = json_str[start_bracket : end_bracket + 1]
            else:
                self.logger.warning("Could not find JSON list boundaries in LLM response for QA.")
                json_str = "[]"

            qa_list = json.loads(json_str)
            if not isinstance(qa_list, list):
                raise ValueError("LLM response is not a JSON list.")

            validated_pairs: List[Dict[str, str]] = []
            for item in qa_list:
                if isinstance(item, dict):
                    q = item.get("question")
                    a = item.get("answer")
                    if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
                        validated_pairs.append({"question": q.strip(), "answer": a.strip()})
                    else:
                        self.logger.warning(f"Skipping invalid QA pair item: {item}")
                else:
                    self.logger.warning(f"Skipping non-dictionary item in QA list: {item}")

            if not validated_pairs:
                raise ValueError("Parsed JSON list contained no valid QA pairs.")

            self.logger.info(
                f"Successfully generated and parsed {len(validated_pairs)} valid QA pairs."
            )
            return {
                "qa_pairs": validated_pairs,
                "success": True,
                "raw_llm_response": llm_response_raw,
            }

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response for QA pairs as JSON list: {e}")
            self.logger.debug(f"Raw LLM response causing QA parse error:\n{llm_response_raw}")
            # Regex fallback attempt
            pairs = []
            try:
                extracted = re.findall(
                    r'\{\s*"question":\s*"(.*?)",\s*"answer":\s*"(.*?)"\s*\}',
                    llm_response_raw,
                    re.DOTALL | re.I,
                )
                pairs = [
                    {
                        "question": q.strip().replace('\\"', '"'),
                        "answer": a.strip().replace('\\"', '"'),
                    }
                    for q, a in extracted
                    if q.strip() and a.strip()
                ]
                if pairs:
                    self.logger.warning(
                        f"JSON parsing failed, extracted {len(pairs)} pairs using regex fallback."
                    )
                    return {
                        "qa_pairs": pairs[:num_questions],
                        "success": True,
                        "warning": "Used regex fallback for parsing.",
                        "raw_llm_response": llm_response_raw,
                    }
            except Exception as regex_e:
                self.logger.error(f"Regex fallback for QA pairs also failed: {regex_e}")
            # If both fail
            return {
                "qa_pairs": [],
                "error": f"Failed to parse LLM response: {e}",
                "raw_llm_response": llm_response_raw,
                "success": False,
            }
        except ToolError as e:
            self.logger.error(f"LLM call failed during QA generation: {e}", exc_info=True)
            return {
                "qa_pairs": [],
                "error": f"LLM call failed: {str(e)}",
                "raw_llm_response": llm_response_raw,
                "success": False,
                "error_code": e.error_code,
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during QA generation: {e}", exc_info=True)
            return {
                "qa_pairs": [],
                "error": f"Unexpected error: {e}",
                "raw_llm_response": llm_response_raw,
                "success": False,
            }

    @tool(
        name="summarize_document",
        description="Generate a concise summary of the document using an LLM",
    )
    @with_tool_metrics
    @with_error_handling
    async def summarize_document(
        self, document: str, max_length: int = 150, focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates a summary of the document text using an LLM."""
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")
        if not isinstance(max_length, int) or max_length <= 10:
            raise ToolInputError(
                "max_length must be a positive integer > 10",
                param_name="max_length",
                provided_value=max_length,
            )

        max_context = 4000
        context = document[:max_context] + ("\n..." if len(document) > max_context else "")
        if len(document) > max_context:
            self.logger.warning(f"Document truncated to ~{max_context} chars for summarization.")

        focus_instruction = (
            f" Focus particularly on aspects related to: {focus}."
            if focus and focus.strip()
            else ""
        )
        prompt = f"""Generate a concise, coherent summary of the following text, about {max_length} words long.{focus_instruction}
Capture the main points and key information accurately based ONLY on the provided text. Do not add external information or opinions.
Output ONLY the summary text itself, without any introductory phrases like "Here is the summary:".

Text:
\"\"\"
{context}
\"\"\"

Summary:
"""
        self.logger.info(
            f"Requesting summary from LLM (max_length≈{max_length}, focus='{focus or 'none'}')."
        )
        llm_response_raw = ""
        try:
            llm_max_tokens = int(max_length / 0.6)  # Estimate tokens from words + buffer
            summary_text = await self._llm(
                prompt=prompt,
                max_tokens=llm_max_tokens,
                temperature=0.5,
            )
            llm_response_raw = summary_text  # Store raw response

            # Clean potential boilerplate
            summary_text = re.sub(
                r"^(Here is a summary:|Summary:|The text discusses|This document is about)\s*:?\s*",
                "",
                summary_text,
                flags=re.I,
            ).strip()
            word_count = len(summary_text.split())
            self.logger.info(f"Generated summary with {word_count} words (target: {max_length}).")
            # Check length deviation
            if word_count < max_length * 0.5:
                self.logger.warning(
                    f"Summary is much shorter ({word_count} words) than requested ({max_length})."
                )
            elif word_count > max_length * 1.5:
                self.logger.warning(
                    f"Summary is much longer ({word_count} words) than requested ({max_length})."
                )

            return {
                "summary": summary_text,
                "word_count": word_count,
                "success": True,
                "raw_llm_response": llm_response_raw,
            }
        except ToolError as e:
            self.logger.error(f"LLM call failed during summarization: {e}", exc_info=True)
            return {
                "summary": "",
                "word_count": 0,
                "error": f"LLM call failed: {str(e)}",
                "success": False,
                "raw_llm_response": llm_response_raw,
                "error_code": e.error_code,
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during summarization: {e}", exc_info=True)
            return {
                "summary": "",
                "word_count": 0,
                "error": f"Unexpected error: {e}",
                "success": False,
                "raw_llm_response": llm_response_raw,
            }

    @tool(
        name="classify_document",
        description="Classify the document into predefined or custom categories using an LLM",
    )
    @with_tool_metrics
    @with_error_handling
    async def classify_document(
        self, document: str, custom_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Classifies the document based on its content using domain-specific or custom labels via an LLM."""
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")

        labels_to_use = (
            custom_labels if isinstance(custom_labels, list) and custom_labels else self._DOC_LABELS
        )
        prompt_prefix = (
            self._CLASS_PROMPT_PREFIX
            if not custom_labels
            else "Classify the document using exactly one of the following labels: "
        )
        if not labels_to_use:
            error_msg = f"No classification labels available for domain '{self.ACTIVE_DOMAIN}' and no custom labels provided."
            self.logger.error(error_msg)
            raise ToolError("CONFIGURATION_ERROR", details={"reason": error_msg})

        label_string = ", ".join(f'"{label}"' for label in labels_to_use)
        max_context = 3000
        context = document[:max_context] + ("\n..." if len(document) > max_context else "")
        if len(document) > max_context:
            self.logger.warning(f"Document truncated to ~{max_context} chars for classification.")

        prompt = f"""{prompt_prefix}{label_string}.
Analyze the following document text and determine the single best classification label from the provided list.
Return ONLY the chosen label as a string, exactly as it appears in the list. Do not include any other text or explanation.

Document Text:
\"\"\"
{context}
\"\"\"

Classification Label:
"""
        self.logger.info(f"Requesting classification from LLM using labels: {labels_to_use}")
        llm_response_raw = ""
        try:
            llm_response_raw = await self._llm(
                prompt=prompt,
                max_tokens=50,
                temperature=0.05,
            )
            raw_output = llm_response_raw.strip().strip('"')
            self.logger.debug(f"LLM raw classification response: '{raw_output}'")

            best_match = None
            best_score = 0.0
            # Check exact match (case-insensitive)
            for label in labels_to_use:
                if raw_output.lower() == label.lower():
                    best_match = label
                    best_score = 1.0
                    break
            # Fuzzy match if no exact match
            if best_match is None:
                self.logger.debug(
                    f"No exact match for '{raw_output}', performing fuzzy matching..."
                )
                fuzzy_scores = {
                    label: fuzz.ratio(raw_output.lower(), label.lower()) / 100.0
                    for label in labels_to_use
                }
                if fuzzy_scores:
                    potential_match = max(fuzzy_scores, key=fuzzy_scores.get)
                    best_score = fuzzy_scores[potential_match]
                    fuzzy_threshold = 0.75
                    if best_score >= fuzzy_threshold:
                        best_match = potential_match
                        self.logger.info(
                            f"Fuzzy match selected: '{best_match}' (Score: {best_score:.2f})"
                        )
                    else:
                        self.logger.warning(
                            f"Best fuzzy match '{potential_match}' ({best_score:.2f}) below threshold {fuzzy_threshold}. Returning raw LLM output."
                        )
                        best_match = raw_output
                else:
                    best_match = raw_output
                    best_score = 0.1
            if best_match is None:
                best_match = raw_output
                best_score = 0.1  # Fallback

            return {
                "classification": best_match,
                "confidence": round(best_score, 3),
                "raw_llm_output": llm_response_raw,
                "success": True,
            }
        except ToolError as e:
            self.logger.error(f"LLM call failed during classification: {e}", exc_info=True)
            return {
                "classification": None,
                "confidence": 0.0,
                "raw_llm_output": llm_response_raw,
                "error": f"LLM call failed: {str(e)}",
                "success": False,
                "error_code": e.error_code,
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during classification: {e}", exc_info=True)
            return {
                "classification": None,
                "confidence": 0.0,
                "raw_llm_output": llm_response_raw,
                "error": f"Unexpected error: {e}",
                "success": False,
            }

    @tool(
        name="extract_metrics",
        description="Extract numeric metrics based on domain-specific keywords and patterns",
    )
    @with_tool_metrics
    @with_error_handling
    async def extract_metrics(self, document: str) -> Dict[str, Any]:
        """Extracts numeric metrics using regex patterns defined for the active domain."""
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")
        if not hasattr(self, "_METRIC_RX") or not isinstance(self._METRIC_RX, list):
            raise ToolError(
                "INITIALIZATION_ERROR",
                details={"reason": "Metric regex patterns _METRIC_RX not initialized"},
            )

        extracted_metrics: Dict[str, List[float]] = {}
        self.logger.info(
            f"Starting metric extraction for domain '{self.ACTIVE_DOMAIN}' ({len(self._METRIC_RX)} types)."
        )
        for metric_name, pattern in self._METRIC_RX:
            if not isinstance(pattern, re.Pattern):
                self.logger.warning(f"Skipping invalid pattern for metric '{metric_name}'.")
                continue
            found_values: Set[float] = set()
            try:
                matches = pattern.findall(document)
                if matches:
                    self.logger.debug(
                        f"Found {len(matches)} potential matches for metric '{metric_name}'"
                    )
                for match_groups in matches:
                    val_str = (
                        match_groups[1]
                        if isinstance(match_groups, tuple) and len(match_groups) >= 2
                        else match_groups
                        if isinstance(match_groups, str)
                        else None
                    )
                    if val_str is None:
                        continue
                    val_str_cleaned = re.sub(r"[^\d.-]", "", str(val_str))
                    if val_str_cleaned.endswith("."):
                        val_str_cleaned = val_str_cleaned[:-1]
                    if not val_str_cleaned or val_str_cleaned == "-":
                        continue
                    try:
                        found_values.add(float(val_str_cleaned))
                    except ValueError:
                        self.logger.debug(
                            f"Could not convert value '{val_str_cleaned}' for metric '{metric_name}'."
                        )
            except Exception as e:
                self.logger.error(
                    f"Error processing regex for metric '{metric_name}': {e}", exc_info=True
                )
            if found_values:
                unique_values = sorted(list(found_values))
                extracted_metrics[metric_name] = unique_values
                self.logger.info(
                    f"Extracted {len(unique_values)} unique values for metric '{metric_name}': {unique_values}"
                )
        return {"metrics": extracted_metrics, "success": True}

    @tool(
        name="flag_risks",
        description="Identify potential risk indicators using domain-specific regex patterns",
    )
    @with_tool_metrics
    @with_error_handling
    async def flag_risks(self, document: str) -> Dict[str, Any]:
        """Flags potential risks using regex patterns defined for the active domain."""
        if not document or not isinstance(document, str):
            raise ToolInputError("Input document must be a non-empty string", param_name="document")
        if not hasattr(self, "_RISK_RX") or not isinstance(self._RISK_RX, dict):
            raise ToolError(
                "INITIALIZATION_ERROR",
                details={"reason": "Risk regex patterns _RISK_RX not initialized"},
            )

        flagged_risks: Dict[str, Dict[str, Any]] = {}
        self.logger.info(
            f"Starting risk flagging for domain '{self.ACTIVE_DOMAIN}' ({len(self._RISK_RX)} types)."
        )
        context_window = 50
        max_samples = 3

        for risk_type, pattern in self._RISK_RX.items():
            if not isinstance(pattern, re.Pattern):
                self.logger.warning(f"Skipping invalid pattern for risk '{risk_type}'.")
                continue
            match_contexts: List[str] = []
            match_count = 0
            try:
                for match in pattern.finditer(document):
                    match_count += 1
                    if len(match_contexts) < max_samples:
                        start, end = match.start(), match.end()
                        ctx_start, ctx_end = (
                            max(0, start - context_window),
                            min(len(document), end + context_window),
                        )
                        snippet = document[ctx_start:ctx_end].replace("\n", " ").strip()
                        prefix = "..." if ctx_start > 0 else ""
                        suffix = "..." if ctx_end < len(document) else ""
                        # Highlight match
                        hl_start = start - ctx_start + len(prefix)
                        hl_end = end - ctx_start + len(prefix)
                        formatted_snippet = f"{prefix}{snippet[:hl_start]}**{snippet[hl_start:hl_end]}**{snippet[hl_end:]}{suffix}"
                        match_contexts.append(formatted_snippet)
                if match_count > 0:
                    self.logger.info(f"Flagged risk '{risk_type}' {match_count} times.")
                    flagged_risks[risk_type] = {
                        "count": match_count,
                        "sample_contexts": match_contexts,
                    }
            except Exception as e:
                self.logger.error(
                    f"Error processing regex for risk '{risk_type}': {e}", exc_info=True
                )
        return {"risks": flagged_risks, "success": True}

    @tool(
        name="canonicalise_entities",
        description="Normalize and deduplicate extracted entities using fuzzy matching",
    )
    @with_tool_metrics
    @with_error_handling
    async def canonicalise_entities(self, entities_input: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes and attempts to merge similar entities from an extraction result."""
        if not isinstance(entities_input, dict):
            raise ToolInputError("Input must be a dictionary.", param_name="entities_input")

        entities_list: List[Dict[str, Any]] = []
        raw_entities = entities_input.get("entities")

        if isinstance(raw_entities, dict):  # Format: Dict[str, List[str]]
            for etype, text_list in raw_entities.items():
                if isinstance(text_list, list):
                    for text in text_list:
                        if isinstance(text, str) and text.strip():
                            entities_list.append(
                                {"text": text.strip(), "type": str(etype).upper().strip()}
                            )
                else:
                    self.logger.warning(
                        f"Expected list for entity type '{etype}', got {type(text_list)}. Skipping."
                    )
        elif isinstance(raw_entities, list):  # Format: List[Dict[str, Any]]
            for item in raw_entities:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                    and item["text"].strip()
                    and isinstance(item.get("type"), str)
                    and item["type"].strip()
                ):
                    entities_list.append(
                        {
                            "text": item["text"].strip(),
                            "type": item["type"].upper().strip(),
                            "metadata": {
                                k: v for k, v in item.items() if k not in ["text", "type"]
                            },
                        }
                    )
                else:
                    self.logger.warning(f"Skipping invalid item in entity list: {item}")
        else:
            raise ToolInputError(
                "Input dict must contain 'entities' key with either Dict[str, List[str]] or List[Dict[str, Any]].",
                param_name="entities_input",
            )

        if not entities_list:
            return {"canonicalized": {}, "success": True}

        entities_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for entity in entities_list:
            etype = entity.get("type", "UNKNOWN")
            entities_by_type.setdefault(etype, []).append(entity)

        canonicalized_output: Dict[str, List[Dict[str, Any]]] = {}
        similarity_threshold = 85  # Similarity score for merging

        for entity_type, entity_group in entities_by_type.items():
            self.logger.debug(
                f"Canonicalising {len(entity_group)} entities of type '{entity_type}'..."
            )
            merged_entities: List[Dict[str, Any]] = []
            processed_indices = set()

            for i in range(len(entity_group)):
                if i in processed_indices:
                    continue
                current_entity = entity_group[i]
                canonical_form = current_entity.get("text", "")
                if not canonical_form:
                    processed_indices.add(i)
                    continue

                cluster_variants = [current_entity]
                processed_indices.add(i)
                cluster_texts = {canonical_form}

                for j in range(i + 1, len(entity_group)):
                    if j in processed_indices:
                        continue
                    other_entity = entity_group[j]
                    other_text = other_entity.get("text", "")
                    if not other_text:
                        processed_indices.add(j)
                        continue

                    # Use token_sort_ratio for better handling of word order
                    score = fuzz.token_sort_ratio(canonical_form.lower(), other_text.lower())
                    if score >= similarity_threshold:
                        cluster_variants.append(other_entity)
                        cluster_texts.add(other_text)
                        processed_indices.add(j)
                        self.logger.debug(
                            f"  Merging '{other_text}' into cluster for '{canonical_form}' (score: {score:.0f})"
                        )
                        # Optional: Update canonical form to longest variant?
                        if len(other_text) > len(canonical_form):
                            canonical_form = other_text

                if cluster_variants:
                    # Final choice for canonical form: longest variant in the cluster
                    canonical_form = max(cluster_texts, key=len)
                    unique_texts = sorted(list(cluster_texts))
                    # TODO: Merge metadata (e.g., combine positions, average scores)
                    merged_entities.append(
                        {
                            "text": canonical_form,
                            "count": len(cluster_variants),
                            "type": entity_type,
                            "variants": unique_texts,
                        }
                    )

            merged_entities.sort(key=lambda x: (-x.get("count", 0), x.get("text", "")))
            canonicalized_output[entity_type] = merged_entities
            self.logger.info(
                f"Canonicalised type '{entity_type}': {len(entity_group)} variants -> {len(merged_entities)} unique entities."
            )

        return {"canonicalized": canonicalized_output, "success": True}

    ###############################################################################
    # New OCR-Specific Tools                                                      #
    ###############################################################################

    @tool(
        name="ocr_image",
        description="Extract text from an image file or data using OCR and optional LLM enhancement.",
    )
    @with_tool_metrics
    @with_retry(max_retries=2, retry_delay=1.5)
    @with_error_handling
    async def ocr_image(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[str] = None,  # Base64 encoded
        ocr_options: Optional[Dict] = None,  # Same options as convert_document
        enhance_with_llm: bool = True,
        output_format: str = "markdown",  # Primarily "text" or "markdown"
    ) -> Dict[str, Any]:
        """
        Performs OCR on a single image and optionally enhances the text with an LLM.

        Args:
            image_path: Path to the image file (e.g., PNG, JPG). Mutually exclusive with image_data.
            image_data: Base64-encoded image data. Mutually exclusive with image_path.
            ocr_options: Dictionary of options for OCR/Enhancement:
                - language (str): Tesseract language(s). Default: "eng".
                - preprocessing (dict): Image preprocessing options.
                - remove_headers (bool): Attempt header/footer removal (less effective on single images). Default: False.
                - assess_quality (bool): Run LLM quality assessment. Default: False.
                - detect_tables (bool): Attempt to detect tables in the image (used during markdown formatting). Default: True.
            enhance_with_llm: If True (default), enhance the raw OCR text using an LLM.
            output_format: Target format ('markdown' or 'text').

        Returns:
            Dictionary with OCR results:
            {
                "success": bool,
                "content": str, # Enhanced text
                "output_format": str,
                "processing_time": float,
                "document_metadata": Dict, # Basic metadata (page_count=1, tables_detected)
                "raw_text": Optional[str], # Raw OCR text
                "ocr_quality_metrics": Optional[Dict] # Quality assessment result
                "error": Optional[str],
                "error_code": Optional[str]
            }
        """
        t0 = time.time()
        ocr_opts = ocr_options or {}
        output_format = output_format.lower()
        if output_format not in self._OCR_COMPATIBLE_FORMATS:
            self.logger.warning(
                f"Output format '{output_format}' not ideal for image OCR. Using 'markdown'."
            )
            output_format = "markdown"

        # --- Dependency Checks ---
        self._ocr_check_dep("Pillow", self._pil_available, "Image OCR")
        self._ocr_check_dep("pytesseract", self._pytesseract_available, "Image OCR")
        # CV2/Numpy needed for advanced preprocessing/table detection
        if ocr_opts.get("preprocessing") and (not self._cv2_available or not self._numpy_available):
            self.logger.warning(
                "Preprocessing options provided but OpenCV/NumPy missing. Preprocessing limited."
            )
        if ocr_opts.get("detect_tables", True) and (
            not self._cv2_available or not self._numpy_available
        ):
            self.logger.warning("Table detection requires OpenCV/NumPy. Disabling table detection.")
            ocr_opts["detect_tables"] = False

        # --- Input Handling ---
        if not image_path and not image_data:
            raise ToolInputError("Either 'image_path' or 'image_data' must be provided.")
        if image_path and image_data:
            raise ToolInputError("Provide either 'image_path' or 'image_data', not both.")

        img: Optional["PILImage.Image"] = None
        input_name = "image_data"
        try:
            if image_path:
                img_path_obj = self._ocr_validate_file_path(image_path)  # Validate path
                input_name = img_path_obj.name
                with self._span(f"load_image_{input_name}"):
                    img = Image.open(img_path_obj)  # type: ignore
            elif image_data:
                try:
                    # Remove data URI prefix if present
                    if image_data.startswith("data:image"):
                        image_data = image_data.split(";base64,", 1)[1]
                    img_bytes = base64.b64decode(image_data)
                    with self._span("load_image_bytes"):
                        img = Image.open(io.BytesIO(img_bytes))  # type: ignore
                except Exception as e_b64:
                    raise ToolInputError(f"Invalid base64 image data: {e_b64}") from e_b64

            if img is None:
                raise ToolError(
                    "IMAGE_LOAD_FAILED", details={"reason": "Image could not be loaded."}
                )
            img = img.convert("RGB")  # Convert to RGB for consistency

            # --- OCR Pipeline ---
            with self._span("image_preprocessing"):
                preprocessed_img = self._ocr_preprocess_image(img, ocr_opts.get("preprocessing"))

            ocr_lang = ocr_opts.get("language", "eng")
            with self._span("tesseract_ocr"):
                raw_text = self._ocr_run_tesseract(preprocessed_img, ocr_lang)

            # --- LLM Enhancement ---
            final_content = raw_text
            if enhance_with_llm:
                with self._span("llm_image_text_enhancement"):
                    # remove_headers less relevant for single image, default False
                    remove_headers = ocr_opts.get("remove_headers", False)
                    # Table detection happens within enhancement if output is markdown
                    detect_tables = (
                        ocr_opts.get("detect_tables", True) and output_format == "markdown"
                    )

                    # Enhance using the same chunk processor (though likely only one chunk)
                    final_content = await self._ocr_enhance_text_chunk(
                        raw_text,
                        output_format=output_format,
                        remove_headers=remove_headers,  # Pass option
                    )
                    # TODO: Integrate table detection results better if needed?
                    # The enhance chunk currently doesn't use _detect_tables explicitly.
                    # Could run detect_tables here and pass info to the prompt? Complex.
            else:
                # If no enhancement and markdown requested, just return raw text
                final_content = raw_text

            # --- Quality Assessment ---
            quality_metrics = None
            if enhance_with_llm and ocr_opts.get("assess_quality", False):
                with self._span("ocr_quality_assessment"):
                    quality_metrics = await self._ocr_assess_text_quality(raw_text, final_content)

            # --- Metadata ---
            tables_detected = False
            if ocr_opts.get("detect_tables", True):
                # Run detection even if not formatting, just for metadata
                detected_regions = self._ocr_detect_tables(preprocessed_img)
                tables_detected = len(detected_regions) > 0

            doc_metadata = {
                "num_pages": 1,
                "has_tables": tables_detected,
                "has_figures": True,  # It's an image
                "has_sections": False,  # Unlikely from raw OCR
                "image_width": img.width,
                "image_height": img.height,
            }

            # --- Construct Response ---
            elapsed = round(time.time() - t0, 3)
            response = {
                "success": True,
                "content": final_content,
                "output_format": output_format,
                "processing_time": elapsed,
                "document_metadata": doc_metadata,
                "extraction_strategy_used": "ocr",  # Only strategy for images
            }
            if enhance_with_llm:
                response["raw_text"] = raw_text
            if quality_metrics:
                response["ocr_quality_metrics"] = quality_metrics

            self.logger.info(f"Completed OCR for '{input_name}' in {elapsed}s")
            return response

        except Exception as e:
            # Catch potential errors during image loading or processing
            self.logger.error(f"Error during image OCR for '{input_name}': {e}", exc_info=True)
            # Re-raise as ToolError or let with_error_handling catch it
            if isinstance(e, (ToolInputError, ToolError)):
                raise e
            raise ToolError(
                "IMAGE_OCR_FAILED", details={"input": input_name, "error": str(e)}
            ) from e
        finally:
            if img:
                img.close()
            if "preprocessed_img" in locals() and preprocessed_img:
                preprocessed_img.close()

    @tool(
        name="enhance_ocr_text",
        description="Enhance existing OCR text using LLM correction and formatting.",
    )
    @with_tool_metrics
    # Remove the problematic cache decorator
    # @with_cache(ttl=3600)  # Cache enhancement results
    @with_retry(max_retries=2, retry_delay=1.0)
    @with_error_handling
    async def enhance_ocr_text(
        self,
        text: str,
        output_format: str = "markdown",
        enhancement_options: Optional[
            Dict
        ] = None,  # e.g., {"remove_headers": bool, "detect_tables": bool}
    ) -> Dict[str, Any]:
        """
        Enhances existing OCR text using an LLM to correct errors and improve formatting.

        Args:
            text: The raw OCR text to enhance.
            output_format: Target format ('markdown' or 'text').
            enhancement_options: Dictionary of options:
                - remove_headers (bool): Attempt to remove headers/footers. Default: False.
                - detect_tables (bool): If output is markdown, attempt table formatting. Default: True.
                - assess_quality (bool): Run LLM quality assessment comparing input vs output. Default: False.

        Returns:
            Dictionary containing:
            {
                "success": bool,
                "content": str, # Enhanced text
                "output_format": str,
                "processing_time": float,
                "raw_text": str, # Original input text
                "ocr_quality_metrics": Optional[Dict], # Quality assessment result
                "error": Optional[str],
                "error_code": Optional[str]
            }
        """
        t0 = time.time()
        if not text or not isinstance(text, str):
            raise ToolInputError("Input 'text' must be a non-empty string", param_name="text")

        options = enhancement_options or {}
        output_format = output_format.lower()
        if output_format not in self._OCR_COMPATIBLE_FORMATS:
            self.logger.warning(
                f"Output format '{output_format}' not ideal for text enhancement. Using 'markdown'."
            )
            output_format = "markdown"

        try:
            # --- Enhancement Pipeline ---
            with self._span("llm_text_enhancement"):
                # Split if very large text
                max_direct_process_len = 15000  # Process directly if under this length
                if len(text) > max_direct_process_len:
                    self.logger.info(f"Splitting large text ({len(text)} chars) for enhancement.")
                    chunks = self._ocr_split_text_into_chunks(text)  # Use default chunk size
                else:
                    chunks = [text]  # Process as a single chunk

                if not chunks:
                    self.logger.warning("Input text became empty after splitting.")
                    final_content = ""
                else:
                    # Process chunks concurrently
                    enhancement_tasks = [
                        self._ocr_enhance_text_chunk(
                            chunk,
                            output_format=output_format,
                            remove_headers=options.get("remove_headers", False),
                        )
                        for chunk in chunks
                    ]
                    enhanced_chunks = await asyncio.gather(*enhancement_tasks)
                    final_content = "\n\n".join(enhanced_chunks).strip()  # Rejoin enhanced chunks

            # --- Table Formatting (if markdown) ---
            # Currently basic, relies on LLM pass. Could add _ocr_format_tables_in_text here.
            if output_format == "markdown" and options.get("detect_tables", True):
                # final_content = await self._ocr_format_tables_in_text(final_content) # Placeholder
                pass

            # --- Quality Assessment ---
            quality_metrics = None
            if options.get("assess_quality", False):
                with self._span("ocr_quality_assessment"):
                    quality_metrics = await self._ocr_assess_text_quality(text, final_content)

            # --- Construct Response ---
            elapsed = round(time.time() - t0, 3)
            response = {
                "success": True,
                "content": final_content,
                "output_format": output_format,
                "processing_time": elapsed,
                "raw_text": text,  # Return original input text
            }
            if quality_metrics:
                response["ocr_quality_metrics"] = quality_metrics

            self.logger.info(f"Completed OCR text enhancement in {elapsed}s")
            return response

        except Exception as e:
            self.logger.error(f"Error during OCR text enhancement: {e}", exc_info=True)
            if isinstance(e, (ToolInputError, ToolError)):
                raise e
            raise ToolError("TEXT_ENHANCEMENT_FAILED", details={"error": str(e)}) from e

    @tool(
        name="analyze_pdf_structure",
        description="Analyze PDF structure (metadata, outline, fonts, OCR needs) without full text extraction.",
    )
    @with_tool_metrics
    @with_retry(max_retries=2, retry_delay=1.0)
    @with_error_handling
    async def analyze_pdf_structure(
        self,
        file_path: Optional[str] = None,
        document_data: Optional[bytes] = None,
        extract_metadata: bool = True,
        extract_outline: bool = True,
        extract_fonts: bool = False,
        extract_images: bool = False,
        estimate_ocr_needs: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyzes the structure of a PDF file without performing full text extraction.

        Provides information about metadata, outline (table of contents), fonts,
        embedded images (estimated), and an assessment of whether OCR would be beneficial.
        Requires either PyMuPDF or PDFPlumber to be installed.

        Args:
            file_path: Path to the PDF file. Mutually exclusive with document_data.
            document_data: PDF content as bytes. Mutually exclusive with file_path.
            extract_metadata: Whether to extract document metadata (default: True).
            extract_outline: Whether to extract the document outline/TOC (default: True).
                           Note: Only supported if PyMuPDF is available.
            extract_fonts: Whether to extract font information (default: False).
                           Note: Only supported if PyMuPDF is available.
            extract_images: Whether to extract estimated image information (default: False).
                            Note: Only supported if PyMuPDF is available.
            estimate_ocr_needs: Whether to estimate if OCR would benefit this PDF (default: True).

        Returns:
            Dictionary containing analysis results. See OCR script docstring for structure example.

        Raises:
            ToolInputError: If input is invalid or neither PyMuPDF nor PDFPlumber is available.
            ToolError: If analysis fails.
        """
        t0 = time.time()

        # --- Dependency Check ---
        if not self._pymupdf_available and not self._pdfplumber_available:
            raise ToolError(
                "DEPENDENCY_MISSING",
                details={
                    "dependency": "PyMuPDF or PDFPlumber",
                    "feature": "PDF Structure Analysis",
                },
            )

        pdf_lib = "pymupdf" if self._pymupdf_available else "pdfplumber"
        self.logger.info(f"Analyzing PDF structure using {pdf_lib}.")

        # --- Input Handling ---
        input_path: Optional[Path] = None
        temp_file_used = False
        analysis_target: Union[Path, bytes]

        try:
            if file_path:
                input_path = self._ocr_validate_file_path(file_path, expected_extension=".pdf")
                analysis_target = input_path
                input_name = input_path.name
            elif document_data:
                # Most libraries work better with paths, create temp file
                input_path, temp_file_used = self._get_input_path_or_temp(None, document_data)
                analysis_target = input_path  # Use path for analysis
                input_name = input_path.name
            else:
                raise ToolInputError("Either 'file_path' or 'document_data' must be provided.")

            result: Dict[str, Any] = {
                "success": False,
                "file_info": input_name,
                "analysis_engine": pdf_lib,
                "processing_time": 0,
            }

            # --- Analysis using preferred library ---
            if pdf_lib == "pymupdf":
                self._ocr_check_dep("PyMuPDF", self._pymupdf_available, "PDF Structure Analysis")
                with pymupdf.open(analysis_target) as doc:  # type: ignore
                    result["page_count"] = len(doc)

                    if extract_metadata:
                        metadata = doc.metadata
                        result["metadata"] = {
                            k: metadata.get(k, "")
                            for k in [
                                "title",
                                "author",
                                "subject",
                                "keywords",
                                "creator",
                                "producer",
                                "creationDate",
                                "modDate",
                            ]
                        }

                    if extract_outline:
                        toc = doc.get_toc()
                        result["outline"] = self._ocr_process_toc(toc) if toc else []

                    if extract_fonts:
                        fonts: Set[str] = set()
                        embedded_fonts: Set[str] = set()
                        limit = min(10, len(doc))  # Sample pages
                        for i in range(limit):
                            for font in doc[i].get_fonts():
                                fonts.add(font[3])
                                embedded_fonts.add(font[3]) if font[
                                    2
                                ] else None  # font[3]=name, font[2]=embedded flag
                        result["font_info"] = {
                            "total_fonts": len(fonts),
                            "embedded_fonts": len(embedded_fonts),
                            "font_names": sorted(list(fonts)),
                        }

                    if extract_images:
                        img_count = 0
                        img_types: Dict[str, int] = {}
                        total_size = 0
                        limit = min(5, len(doc))  # Sample pages
                        for i in range(limit):
                            for img in doc[i].get_images(full=True):
                                img_count += 1
                                xref = img[0]
                                img_info = doc.extract_image(xref)
                                if img_info:
                                    img_types[img_info["ext"]] = (
                                        img_types.get(img_info["ext"], 0) + 1
                                    )
                                    total_size += len(img_info["image"])
                        est_total = int(img_count * (len(doc) / max(1, limit)))
                        avg_size_kb = (
                            int(total_size / max(1, img_count) / 1024) if img_count > 0 else 0
                        )
                        result["image_info"] = {
                            "sampled_images": img_count,
                            "estimated_total_images": est_total,
                            "image_types": img_types,
                            "average_size_kb": avg_size_kb,
                        }

                    if estimate_ocr_needs:
                        text_pages = 0
                        sample_size = min(10, len(doc))
                        min_chars = 50
                        for i in range(sample_size):
                            if len(doc[i].get_text("text").strip()) > min_chars:
                                text_pages += 1
                        text_ratio = text_pages / max(1, sample_size)
                        needs_ocr = (
                            text_ratio < 0.8
                        )  # Threshold: if less than 80% pages have decent text
                        confidence = "high" if text_ratio < 0.2 or text_ratio > 0.95 else "medium"
                        reason = (
                            "PDF appears scanned or image-based."
                            if needs_ocr and confidence == "high"
                            else "PDF likely contains extractable text."
                            if not needs_ocr and confidence == "high"
                            else "PDF contains a mix of text and potentially image-based pages."
                        )
                        result["ocr_assessment"] = {
                            "needs_ocr": needs_ocr,
                            "confidence": confidence,
                            "reason": reason,
                            "text_coverage_ratio": round(text_ratio, 2),
                        }

            elif pdf_lib == "pdfplumber":
                self._ocr_check_dep(
                    "pdfplumber", self._pdfplumber_available, "PDF Structure Analysis"
                )
                with pdfplumber.open(analysis_target) as pdf:  # type: ignore
                    result["page_count"] = len(pdf.pages)

                    if extract_metadata:
                        metadata = pdf.metadata
                        result["metadata"] = {
                            k: metadata.get(k.capitalize(), "")
                            for k in [
                                "title",
                                "author",
                                "subject",
                                "keywords",
                                "creator",
                                "producer",
                                "creationDate",
                                "modDate",
                            ]
                        }  # Keys often capitalized

                    # pdfplumber doesn't support outline, fonts, images easily
                    if extract_outline:
                        result["outline"] = []  # Not available
                    if extract_fonts:
                        result["font_info"] = {
                            "error": "Font extraction not supported by pdfplumber."
                        }
                    if extract_images:
                        result["image_info"] = {
                            "error": "Image extraction not supported by pdfplumber."
                        }

                    if estimate_ocr_needs:
                        text_pages = 0
                        sample_size = min(10, len(pdf.pages))
                        min_chars = 50
                        for i in range(sample_size):
                            if len((pdf.pages[i].extract_text() or "").strip()) > min_chars:
                                text_pages += 1
                        text_ratio = text_pages / max(1, sample_size)
                        needs_ocr = text_ratio < 0.8
                        confidence = "high" if text_ratio < 0.2 or text_ratio > 0.95 else "medium"
                        reason = (
                            "PDF appears scanned or image-based."
                            if needs_ocr and confidence == "high"
                            else "PDF likely contains extractable text."
                            if not needs_ocr and confidence == "high"
                            else "PDF contains a mix of text and potentially image-based pages."
                        )
                        result["ocr_assessment"] = {
                            "needs_ocr": needs_ocr,
                            "confidence": confidence,
                            "reason": reason,
                            "text_coverage_ratio": round(text_ratio, 2),
                        }

            # --- Finalize Result ---
            result["success"] = True
            result["processing_time"] = round(time.time() - t0, 3)
            self.logger.info(
                f"PDF structure analysis for '{input_name}' completed in {result['processing_time']:.3f}s using {pdf_lib}"
            )
            return result

        except Exception as e:
            self.logger.error(
                f"Error during PDF structure analysis for '{input_name}': {e}", exc_info=True
            )
            if isinstance(e, (ToolInputError, ToolError)):
                raise e
            raise ToolError(
                "PDF_ANALYSIS_FAILED", details={"input": input_name, "error": str(e)}
            ) from e
        finally:
            # Clean up temp file if we created one
            if temp_file_used and input_path and input_path.exists():
                try:
                    input_path.unlink()
                    self.logger.debug(f"Cleaned up temporary analysis file: {input_path}")
                except OSError as e_del:
                    self.logger.warning(
                        f"Failed to delete temporary analysis file {input_path}: {e_del}"
                    )

    ###############################################################################
    # Batch Processing Tool                                                       #
    ###############################################################################

    @tool(
        name="process_document_batch",
        description="Process a batch of documents/texts through a pipeline of operations",
    )
    @with_tool_metrics
    # No retry at this level; retries should be on individual operations if needed
    @with_error_handling  # Catch errors in setting up the batch itself
    async def process_document_batch(
        self,
        inputs: List[Dict[str, Any]],
        operations: List[Dict[str, Any]],
        max_concurrency: int = 5,
    ) -> List[Dict[str, Any]]:
        """Processes a list of input items through a sequence of operations concurrently.

        Each item in the `inputs` list is processed sequentially through the `operations` list.
        For each operation step, all items are processed concurrently up to `max_concurrency`.

        Args:
            inputs: List of input dictionaries. Each dictionary represents an item to process
                    and should contain initial data under keys like "document_path",
                    "document_data", or "content".
            operations: List of operation specifications. Each dict defines one step:
                - operation (str): Name of the tool method to call (must be in `self.op_map`).
                - output_key (str): Key to store the operation's result dictionary under in the item's state.
                - params (dict): Dictionary of fixed parameters for the operation.
                Optional keys for operation dict:
                - input_key (str): Key in the item's state dict holding the primary input
                                   (e.g., "document" for analysis, "document_path" for conversion).
                                   If omitted, defaults based on operation name convention.
                - input_keys_map (dict): Map parameter names of the tool function to keys
                                         in the item's state dict. Allows passing multiple dynamic inputs.
                                         e.g., {"document": "cleaned_text", "num_questions": "qa_count"}
                - promote_output (str): Key within the operation's result dict whose value should be
                                        promoted to the top-level `content` key for the *next* step's default input.
                                        e.g., for chunk_document result `{"chunks": [...]}`, `promote_output="chunks"`
                                        makes the list of chunks available as `content` for the next operation.
            max_concurrency: Maximum number of items to process in parallel *per operation step*.

        Returns:
            List of dictionaries, representing the final state of each input item after all
            operations. Each dict contains original input keys plus keys for each operation's
            output (`output_key`) and an `_error_log` list detailing any failures.
        """
        # --- Input Validation ---
        if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
            raise ToolInputError(
                "Input 'inputs' must be a list of dictionaries.", param_name="inputs"
            )
        if not isinstance(operations, list) or not all(isinstance(op, dict) for op in operations):
            raise ToolInputError(
                "Input 'operations' must be a list of dictionaries.", param_name="operations"
            )
        if max_concurrency <= 0:
            self.logger.warning("max_concurrency must be positive, setting to 1.")
            max_concurrency = 1
        if not inputs:
            self.logger.warning("Input list is empty. Nothing to process.")
            return []

        # --- Initialize Results State ---
        results_state: List[Dict[str, Any]] = []
        for i, item in enumerate(inputs):
            if not isinstance(item, dict):  # Ensure each item is a dict
                self.logger.error(f"Item at index {i} in inputs is not a dictionary. Skipping.")
                results_state.append(
                    {
                        "_original_index": i,
                        "_error_log": [f"Input item at index {i} was not a dictionary."],
                        "_status": "failed",
                        "input_data_preview": str(item)[:100],  # Log preview
                    }
                )
                continue
            state_item = item.copy()
            state_item["_original_index"] = i
            state_item["_error_log"] = []
            state_item["_status"] = "pending"  # pending -> processed -> failed
            results_state.append(state_item)

        self.logger.info(
            f"Starting batch processing for {len(inputs)} items through {len(operations)} operations."
        )

        # --- Apply Operations Sequentially ---
        for op_index, op_spec in enumerate(operations):
            op_name = op_spec.get("operation")
            op_output_key = op_spec.get("output_key")
            op_params = op_spec.get("params", {})
            op_input_key = op_spec.get("input_key")
            op_input_map = op_spec.get("input_keys_map", {})
            op_promote = op_spec.get("promote_output")

            # --- Validate Operation Spec ---
            if not op_name or not isinstance(op_name, str) or op_name not in self.op_map:
                error_msg = f"Invalid/unknown operation '{op_name}' at step {op_index + 1}."
                self.logger.error(error_msg + " Skipping step for all items.")
                for item_state in results_state:
                    if item_state["_status"] != "failed":
                        item_state["_error_log"].append(error_msg + " (Operation Skipped)")
                        item_state["_status"] = "failed"  # Mark as failed if step is invalid
                continue  # Skip this operation entirely

            if not op_output_key or not isinstance(op_output_key, str):
                error_msg = f"Missing/invalid 'output_key' for operation '{op_name}' at step {op_index + 1}."
                self.logger.error(error_msg + " Skipping step for all items.")
                for item_state in results_state:
                    if item_state["_status"] != "failed":
                        item_state["_error_log"].append(error_msg + " (Operation Skipped)")
                        item_state["_status"] = "failed"
                continue

            if not isinstance(op_params, dict):
                error_msg = f"Invalid 'params' (must be a dict) for operation '{op_name}' at step {op_index + 1}."
                self.logger.error(error_msg + " Skipping step for all items.")
                for item_state in results_state:
                    if item_state["_status"] != "failed":
                        item_state["_error_log"].append(error_msg + " (Operation Skipped)")
                        item_state["_status"] = "failed"
                continue

            op_func = self.op_map[op_name]
            step_label = f"Step {op_index + 1}/{len(operations)}: '{op_name}'"
            self.logger.info(f"--- Starting {step_label} (Concurrency: {max_concurrency}) ---")

            # --- Define Worker for this Operation ---
            async def _apply_op_to_item(
                item_state: Dict[str, Any], 
                semaphore, 
                step_label=step_label,
                op_name=op_name,
                op_input_key=op_input_key,
                op_output_key=op_output_key,
                op_params=op_params,
                op_input_map=op_input_map,
                op_promote=op_promote,
                op_func=op_func
            ) -> Dict[str, Any]:
                item_idx = item_state["_original_index"]
                if item_state["_status"] == "failed":
                    self.logger.debug(
                        f"Skipping {step_label} for item {item_idx} (already failed)."
                    )
                    return item_state

                async with semaphore:
                    self.logger.debug(f"Applying {step_label} to item {item_idx}")
                    call_kwargs = {}
                    primary_input_arg_name = None

                    try:
                        # --- Determine Primary Input ---
                        actual_input_key = op_input_key  # Explicit key takes precedence
                        if not actual_input_key:  # Determine default input key based on operation
                            if op_name.startswith("convert_document"):
                                actual_input_key = "document_path"  # Prefers path
                            elif op_name == "ocr_image":
                                actual_input_key = "image_path"  # Prefers path
                            elif op_name == "analyze_pdf_structure":
                                actual_input_key = "file_path"
                            elif op_name == "enhance_ocr_text":
                                actual_input_key = "text"
                            elif op_name == "canonicalise_entities":
                                actual_input_key = "entities_input"
                            elif op_name == "batch_format_texts":
                                actual_input_key = "texts"
                            else:
                                actual_input_key = (
                                    "document" if "document" in item_state else "content"
                                )  # Default cascade

                        # Check if primary input source exists (could be path or data)
                        primary_input_value = None
                        input_source_key = None  # Track which key held the data

                        # Prioritize specific path/data keys if they exist for relevant ops
                        if (
                            op_name.startswith("convert_document")
                            or op_name == "ocr_image"
                            or op_name == "analyze_pdf_structure"
                        ):
                            if "document_path" in item_state:
                                input_source_key = "document_path"
                            elif "file_path" in item_state:
                                input_source_key = "file_path"  # Alias for analysis
                            elif "image_path" in item_state:
                                input_source_key = "image_path"
                            elif "document_data" in item_state:
                                input_source_key = "document_data"
                            elif "image_data" in item_state:
                                input_source_key = "image_data"  # base64 string
                            # Fallback to the determined actual_input_key if specific ones not found
                            elif actual_input_key in item_state:
                                input_source_key = actual_input_key
                        else:  # For text-based operations
                            if actual_input_key in item_state:
                                input_source_key = actual_input_key
                            elif "content" in item_state:
                                input_source_key = "content"  # Common fallback

                        if input_source_key is None or input_source_key not in item_state:
                            raise ToolInputError(
                                f"Required input key ('{actual_input_key}' or alternatives) not found in state for item {item_idx}.",
                                param_name=actual_input_key,
                            )

                        primary_input_value = item_state[input_source_key]

                        # --- Determine Parameter Name for Primary Input ---
                        # Map the source key to the expected function parameter name
                        primary_param_map = {
                            "document_path": "document_path",
                            "file_path": "file_path",
                            "image_path": "image_path",
                            "document_data": "document_data",
                            "image_data": "image_data",
                            "text": "text",
                            "entities_input": "entities_input",
                            "texts": "texts",
                            "document": "document",
                            "content": "document",  # Map content to 'document' often
                        }
                        primary_input_arg_name = primary_param_map.get(input_source_key)

                        if not primary_input_arg_name:
                            # Should ideally map based on function signature inspection, but use convention
                            primary_input_arg_name = "document"  # Default guess
                            self.logger.warning(
                                f"Could not determine primary argument name for input key '{input_source_key}' for op '{op_name}'. Assuming '{primary_input_arg_name}'."
                            )

                        # Add primary input to kwargs
                        call_kwargs[primary_input_arg_name] = primary_input_value

                        # --- Handle Mapped Inputs ---
                        if isinstance(op_input_map, dict):
                            for param_name, state_key in op_input_map.items():
                                if state_key not in item_state:
                                    raise ToolInputError(
                                        f"Mapped input key '{state_key}' (for param '{param_name}') not found for item {item_idx}.",
                                        param_name=state_key,
                                    )
                                # Avoid overwriting primary input if mapped explicitly
                                if param_name != primary_input_arg_name:
                                    call_kwargs[param_name] = item_state[state_key]
                                elif (
                                    call_kwargs[primary_input_arg_name] != item_state[state_key]
                                ):  # Check if value differs
                                    self.logger.warning(
                                        f"Mapped input '{param_name}' conflicts with primary input for item {item_idx}. Using value from '{state_key}'."
                                    )
                                    call_kwargs[primary_input_arg_name] = item_state[state_key]

                        # --- Add Fixed Params (overrides mapped inputs) ---
                        if isinstance(op_params, dict):
                            # Ensure fixed params don't overwrite the primary input unless intended
                            for p_name, p_value in op_params.items():
                                if (
                                    p_name == primary_input_arg_name
                                    and primary_input_arg_name in call_kwargs
                                ):
                                    self.logger.warning(
                                        f"Fixed param '{p_name}' overrides primary input for item {item_idx}."
                                    )
                                call_kwargs[p_name] = p_value

                        # --- Execute Operation ---
                        self.logger.debug(
                            f"Calling {op_name} for item {item_idx} with args: {list(call_kwargs.keys())}"
                        )
                        # Pass output_key itself as a kwarg to the wrapper op if needed (e.g., convert_document_op)
                        if op_name.endswith("_op"):  # Convention for wrappers
                            call_kwargs["output_key"] = op_output_key

                        op_result = await op_func(**call_kwargs)

                        # --- Process Result ---
                        if not isinstance(op_result, dict):
                            raise ToolError(
                                "INVALID_RESULT_FORMAT",
                                details={
                                    "operation": op_name,
                                    "result_type": type(op_result).__name__,
                                    "item": item_idx,
                                },
                            )

                        item_state[op_output_key] = op_result  # Store full result

                        # Promote output if requested
                        if op_promote and isinstance(op_promote, str):
                            if op_promote in op_result:
                                item_state["content"] = op_result[op_promote]
                                self.logger.debug(
                                    f"Promoted '{op_promote}' to 'content' for item {item_idx}"
                                )
                            else:
                                self.logger.warning(
                                    f"Cannot promote key '{op_promote}' for item {item_idx}: key not found in result of '{op_name}'. Result keys: {list(op_result.keys())}"
                                )

                        # Update status based on success flag in result
                        if not op_result.get("success", False):
                            err_msg = op_result.get("error", f"Operation '{op_name}' failed.")
                            err_code = op_result.get("error_code", "PROCESSING_ERROR")
                            log_entry = f"{step_label} Failed: [{err_code}] {err_msg}"
                            item_state["_error_log"].append(log_entry)
                            item_state["_status"] = "failed"
                            self.logger.warning(
                                f"Operation '{op_name}' failed for item {item_idx}: {err_msg}"
                            )
                        else:
                            # If previously failed, keep failed status. Otherwise, mark processed.
                            if item_state["_status"] != "failed":
                                item_state["_status"] = "processed"

                    except ToolInputError as tie:
                        error_msg = f"{step_label} Input Error: [{tie.error_code}] {str(tie)}"
                        self.logger.error(f"{error_msg} for item {item_idx}", exc_info=False)
                        item_state["_error_log"].append(error_msg)
                        item_state["_status"] = "failed"
                        item_state[op_output_key] = {
                            "error": str(tie),
                            "error_code": tie.error_code,
                            "success": False,
                        }  # Store error info
                    except ToolError as te:
                        error_msg = f"{step_label} Tool Error: [{te.error_code}] {str(te)}"
                        self.logger.error(f"{error_msg} for item {item_idx}", exc_info=True)
                        item_state["_error_log"].append(error_msg)
                        item_state["_status"] = "failed"
                        item_state[op_output_key] = {
                            "error": str(te),
                            "error_code": te.error_code,
                            "success": False,
                        }
                    except Exception as e:
                        error_msg = f"{step_label} Unexpected Error: {type(e).__name__}: {str(e)}"
                        self.logger.error(f"{error_msg} for item {item_idx}", exc_info=True)
                        item_state["_error_log"].append(error_msg)
                        item_state["_status"] = "failed"
                        item_state[op_output_key] = {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "success": False,
                        }

                    return item_state  # Return updated state

            # --- Run Tasks for Current Step ---
            step_semaphore = asyncio.Semaphore(max_concurrency)
            step_tasks = [
                _apply_op_to_item(item_state, step_semaphore) for item_state in results_state
            ]
            updated_states = await asyncio.gather(*step_tasks)
            results_state = updated_states  # Update state list with results from this step

            # Log summary after step
            step_processed_count = sum(1 for s in results_state if s.get("_status") == "processed")
            step_fail_count = sum(1 for s in results_state if s.get("_status") == "failed")
            self.logger.info(
                f"--- Finished {step_label} (Processed: {step_processed_count}, Failed: {step_fail_count}) ---"
            )

        # --- Final Cleanup ---
        final_results = []
        for item_state in results_state:
            final_item = item_state.copy()
            final_item.pop("_original_index", None)
            # Optionally keep status? For now, remove it. User sees error log.
            final_item.pop("_status", None)
            # Keep _error_log
            final_results.append(final_item)

        self.logger.info(f"Batch processing finished for {len(inputs)} items.")
        return final_results
