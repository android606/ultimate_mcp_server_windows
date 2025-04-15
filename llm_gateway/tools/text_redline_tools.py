"""HTML redline tool for LLM Gateway.

This module provides tools for creating high-quality redlines (track changes) 
of HTML documents, similar to those used by legal firms and for SEC filings.
"""
import base64
import difflib
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from lxml import etree, html
from xmldiff import formatting, main

from llm_gateway.constants import TaskType
from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.redline")

# --- Constants for output styling ---
DEFAULT_CSS = """
/* Redline styling with improved visual treatment */
ins, .diff-insert { 
    color: #0066cc; 
    text-decoration: none;
    background-color: #e6f3ff;
    padding: 0 2px;
    border-radius: 2px;
    transition: background-color 0.2s;
}

ins:hover, .diff-insert:hover {
    background-color: #cce6ff;
}

del, .diff-delete { 
    color: #cc0000; 
    text-decoration: line-through;
    background-color: #ffebe6;
    padding: 0 2px;
    border-radius: 2px;
    transition: background-color 0.2s;
}

del:hover, .diff-delete:hover {
    background-color: #ffd6cc;
}

.diff-move-target { 
    color: #006633; 
    text-decoration: none;
    background-color: #e6fff0;
    padding: 0 2px;
    border-radius: 2px;
    transition: background-color 0.2s;
}

.diff-move-target:hover {
    background-color: #ccffe6;
}

.diff-move-source { 
    color: #006633; 
    text-decoration: line-through;
    background-color: #e6fff0;
    padding: 0 2px;
    border-radius: 2px;
    transition: background-color 0.2s;
}

.diff-move-source:hover {
    background-color: #ccffe6;
}

/* Navigation styling */
.redline-navigation {
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.redline-navigation button {
    border: none;
    background: #f8f8f8;
    padding: 5px 10px;
    margin: 0 5px;
    border-radius: 3px;
    cursor: pointer;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    transition: all 0.2s;
}

.redline-navigation button:hover {
    background: #e8e8e8;
}
"""

# --- XSLT for transforming xmldiff output to readable HTML ---
XMLDIFF_XSLT = """<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:diff="http://namespaces.shoobx.com/diff">
  <xsl:template match="@diff:delete">
    <del class="diff-delete"><xsl:apply-templates/></del>
  </xsl:template>

  <xsl:template match="@diff:insert">
    <ins class="diff-insert"><xsl:apply-templates/></ins>
  </xsl:template>

  <xsl:template match="@diff:move-to">
    <ins class="diff-move-target"><xsl:apply-templates/></ins>
  </xsl:template>

  <xsl:template match="@diff:move-from">
    <del class="diff-move-source"><xsl:apply-templates/></del>
  </xsl:template>

  <xsl:template match="@*|node()">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>
</xsl:stylesheet>
"""

@with_tool_metrics
@with_error_handling
async def create_html_redline(
    original_html: str,
    modified_html: str,
    detect_moves: bool = True,
    formatting_tags: Optional[List[str]] = None,
    ignore_whitespace: bool = True,
    include_css: bool = True,
    add_navigation: bool = True,
    output_format: str = "html",
    use_tempfiles: bool = False
) -> Dict[str, Any]:
    """Creates a high-quality redline (track changes) between two HTML documents.
    
    Generates a legal-style redline showing differences between original and modified HTML:
    - Deletions in RED with strikethrough
    - Additions in BLUE
    - Moved content in GREEN (if detect_moves=True)
    
    This tool preserves the original document structure and handles complex HTML
    including tables and nested elements. It's designed for comparing documents 
    like SEC filings, contracts, or other structured content.
    
    Args:
        original_html: The original/old HTML document
        modified_html: The modified/new HTML document
        detect_moves: Whether to identify and highlight moved content (vs treating moves 
                      as deletion+insertion). Default True.
        formatting_tags: List of HTML tags to treat as formatting (e.g., ['b', 'i', 'strong']).
                         Changes to these tags will be highlighted as formatting changes.
                         Default None (auto-detects common formatting tags).
        ignore_whitespace: Whether to ignore trivial whitespace differences. Default True.
        include_css: Whether to include default CSS for styling the redline. Default True.
        add_navigation: Whether to add JavaScript for navigating between changes. Default True.
        output_format: Output format, either "html" for full HTML document or "fragment" 
                       for just the body content. Default "html".
        use_tempfiles: Whether to use temporary files for large documents to reduce memory usage.
                      Default False.
                       
    Returns:
        A dictionary containing:
        {
            "redline_html": The HTML document with redline markups,
            "stats": {
                "total_changes": Total number of changes detected,
                "insertions": Number of insertions,
                "deletions": Number of deletions,
                "moves": Number of moved blocks (if detect_moves=True),
            },
            "processing_time": Time in seconds to generate the redline,
            "success": True if successful
        }
        
    Raises:
        ToolInputError: If input HTML is invalid or parameters are incorrect
        ToolError: If processing fails for other reasons
    """
    start_time = time.time()
    
    # Validate input
    if not original_html or not isinstance(original_html, str):
        raise ToolInputError("Original HTML must be a non-empty string.")
    if not modified_html or not isinstance(modified_html, str):
        raise ToolInputError("Modified HTML must be a non-empty string.")
    if output_format not in ["html", "fragment"]:
        raise ToolInputError(
            f"Invalid output_format: '{output_format}'. Must be 'html' or 'fragment'.",
            param_name="output_format",
            provided_value=output_format
        )
    
    # Set default formatting tags if not provided
    if formatting_tags is None:
        formatting_tags = ['b', 'strong', 'i', 'em', 'u', 'span', 'font', 'sub', 'sup']
    
    try:
        # For very large documents, use temporary files to reduce memory usage
        if use_tempfiles and (len(original_html) > 1_000_000 or len(modified_html) > 1_000_000):
            logger.info("Using temporary files for large document processing")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as orig_file, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as mod_file:
                
                # Write content to temp files
                orig_file.write(original_html)
                mod_file.write(modified_html)
                orig_file.flush()
                mod_file.flush()
                
                # Create Path objects for better path handling
                orig_path = Path(orig_file.name)
                mod_path = Path(mod_file.name)
                
                try:
                    # Parse from files instead of strings
                    original_doc = html.parse(str(orig_path))
                    modified_doc = html.parse(str(mod_path))
                    
                    if ignore_whitespace:
                        _normalize_whitespace(original_doc)
                        _normalize_whitespace(modified_doc)
                except Exception as e:
                    logger.warning(f"Failed to parse HTML from files: {str(e)}, falling back to in-memory parsing")
                    # Fall back to in-memory parsing
                    original_doc, modified_doc = _preprocess_html_docs(
                        original_html, 
                        modified_html,
                        ignore_whitespace=ignore_whitespace
                    )
                
                # Clean up temp files
                try:
                    orig_path.unlink()
                    mod_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary files: {str(e)}")
        else:
            # Parse HTML documents using standard in-memory approach
            original_doc, modified_doc = _preprocess_html_docs(
                original_html, 
                modified_html,
                ignore_whitespace=ignore_whitespace
            )
        
        # Generate diff and transform to redline HTML
        redline_html, diff_stats = _generate_redline(
            original_doc,
            modified_doc,
            detect_moves=detect_moves,
            formatting_tags=formatting_tags
        )
        
        # Post-process the output (add CSS, JS navigation if requested)
        redline_html = _postprocess_redline(
            redline_html,
            include_css=include_css,
            add_navigation=add_navigation,
            output_format=output_format
        )
        
        processing_time = time.time() - start_time
        
        # For very large HTML output, consider base64 encoding to improve transport efficiency
        redline_size = len(redline_html.encode('utf-8'))
        base64_encoded = None
        if redline_size > 10_000_000:  # 10MB
            logger.info(f"Large redline output detected ({redline_size/1_000_000:.2f} MB), providing base64 encoding")
            base64_encoded = base64.b64encode(redline_html.encode('utf-8')).decode('ascii')
        
        logger.success(
            f"Redline generated successfully ({diff_stats['total_changes']} changes)",
            emoji_key=TaskType.DOCUMENT.value,
            changes=diff_stats,
            time=processing_time
        )
        
        result = {
            "redline_html": redline_html,
            "stats": diff_stats,
            "processing_time": processing_time,
            "success": True
        }
        
        if base64_encoded:
            result["base64_encoded"] = base64_encoded
            result["encoding_info"] = "Base64 encoded UTF-8 for efficient transport of large document"
            
        return result
        
    except Exception as e:
        logger.error(f"Error generating redline: {str(e)}", exc_info=True)
        raise ToolError(
            f"Failed to generate redline: {str(e)}",
            error_code="REDLINE_GENERATION_ERROR",
            details={"error": str(e)}
        ) from e


def _preprocess_html_docs(
    original_html: str, 
    modified_html: str,
    ignore_whitespace: bool = True
) -> Tuple[etree._Element, etree._Element]:
    """Parses and preprocesses HTML documents for comparison.
    
    Args:
        original_html: Original HTML string
        modified_html: Modified HTML string
        ignore_whitespace: Whether to normalize whitespace
        
    Returns:
        Tuple of (original_doc, modified_doc) as lxml Element objects
        
    Raises:
        ToolError: If HTML parsing fails
    """
    try:
        # Try using external HTML tidy if available (better handling of malformed HTML)
        use_external_tidy = _check_tidy_available()
        
        if use_external_tidy:
            logger.info("Using external HTML tidy for preprocessing")
            original_html = _run_html_tidy(original_html)
            modified_html = _run_html_tidy(modified_html)
        
        # Parse HTML documents using lxml
        parser = html.HTMLParser()
        try:
            original_doc = html.fromstring(original_html, parser=parser)
            modified_doc = html.fromstring(modified_html, parser=parser)
        except Exception as e:
            # Fallback to BeautifulSoup if lxml fails (for malformed HTML)
            logger.warning(f"lxml parsing failed, falling back to BeautifulSoup: {str(e)}")
            original_soup = BeautifulSoup(original_html, 'html.parser')
            modified_soup = BeautifulSoup(modified_html, 'html.parser')
            original_html = str(original_soup)
            modified_html = str(modified_soup)
            original_doc = html.fromstring(original_html, parser=parser)
            modified_doc = html.fromstring(modified_html, parser=parser)
        
        # Normalize whitespace if requested
        if ignore_whitespace:
            _normalize_whitespace(original_doc)
            _normalize_whitespace(modified_doc)
            
        return original_doc, modified_doc
    
    except Exception as e:
        raise ToolError(
            f"Failed to parse HTML documents: {str(e)}",
            error_code="HTML_PARSING_ERROR"
        ) from e


def _check_tidy_available() -> bool:
    """Checks if HTML tidy is available on the system."""
    try:
        # Try to run html tidy with version flag
        result = subprocess.run(
            ["tidy", "-v"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=1
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _run_html_tidy(html_content: str) -> str:
    """Runs HTML content through tidy to clean and normalize it."""
    try:
        # Create a temp file for the content
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=False) as temp_file:
            temp_file.write(html_content)
            temp_file.flush()
            temp_path = Path(temp_file.name)
        
        # Run tidy with optimal settings for diff processing
        result = subprocess.run(  # noqa: F841
            [
                "tidy",
                "-q",                    # Quiet mode
                "-m",                    # Modify input file
                "--tidy-mark", "no",     # No tidy meta tag
                "--drop-empty-elements", "no",  # Don't drop empty elements
                "--wrap", "0",           # No line wrapping
                str(temp_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10                   # Timeout after 10 seconds
        )
        
        # Read the tidied content
        tidied_content = temp_path.read_text(encoding='utf-8')
        
        # Clean up
        temp_path.unlink()
        
        return tidied_content
    except Exception as e:
        logger.warning(f"HTML tidy failed: {str(e)}, using original content")
        return html_content


def _normalize_whitespace(doc: etree._Element) -> None:
    """Normalizes whitespace in text nodes.
    
    Args:
        doc: lxml Element tree to normalize
    """
    # Find all text nodes
    for element in doc.iter():
        if element.text is not None and element.tag not in ['script', 'style', 'pre']:
            # Normalize whitespace in text content
            element.text = re.sub(r'\s+', ' ', element.text).strip()
        if element.tail is not None:
            # Normalize whitespace in tail text
            element.tail = re.sub(r'\s+', ' ', element.tail).strip()


def _generate_redline(
    original_doc: etree._Element,
    modified_doc: etree._Element,
    detect_moves: bool = True,
    formatting_tags: List[str] = None
) -> Tuple[str, Dict[str, int]]:
    """Generates redline HTML showing differences between documents.
    
    Args:
        original_doc: Original document as lxml Element
        modified_doc: Modified document as lxml Element
        detect_moves: Whether to detect moved content
        formatting_tags: Tags to treat as formatting
        
    Returns:
        Tuple of (redline_html, stats_dict)
    """
    # Log the xmldiff version being used for diagnostic purposes
    try:
        import importlib.metadata
        xmldiff_version = importlib.metadata.version('xmldiff')
        logger.debug(f"Using xmldiff version: {xmldiff_version}")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        logger.debug("Could not determine xmldiff version")
    
    # Configure xmldiff formatter
    formatter = formatting.XMLFormatter(
        normalize=formatting.WS_BOTH if detect_moves else formatting.WS_NONE,
        pretty_print=True,
        text_tags=('p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', 'th', 'div', 'span', 'a'),
        formatting_tags=formatting_tags or [],
        diff_tag='diff',
        diff_attr_prefix='diff:'
    )
    
    # Track statistics
    insertions = 0
    deletions = 0
    moves = 0
    
    # Generate diff
    try:
        if detect_moves:
            # Use detailed diff with move detection
            diff_tree = main.diff_trees(
                original_doc, 
                modified_doc,
                formatter=formatter,
                ratio_mode=True  # Better for prose comparison
            )
            
            # Count operations for statistics
            for action in diff_tree:
                action_type = action[0]
                if action_type == 'insert':
                    insertions += 1
                elif action_type == 'delete':
                    deletions += 1
                elif action_type == 'move':
                    moves += 1
                    
            # Generate string output with diff markings
            diff_html = formatter.format(diff_tree, original_doc)
            
        else:
            # Simpler diff without move detection
            diff_html = main.diff_trees(
                original_doc, 
                modified_doc,
                formatter=formatter
            )
            
            # Count inserted and deleted attributes in output for statistics
            insertions = diff_html.count('diff:insert')
            deletions = diff_html.count('diff:delete')
            moves = 0
    except Exception as e:
        logger.error(f"xmldiff processing failed: {str(e)}", exc_info=True)
        raise ToolError(
            f"XML diff processing failed: {str(e)}",
            error_code="XMLDIFF_ERROR"
        ) from e
        
    # Apply XSLT to transform diff attributes into HTML elements
    try:
        xslt_doc = etree.fromstring(XMLDIFF_XSLT)
        transform = etree.XSLT(xslt_doc)
        
        # Parse the diff output and apply transformation
        diff_doc = etree.fromstring(diff_html.encode('utf-8'))
        result_doc = transform(diff_doc)
        
        # Convert back to string
        redline_html = etree.tostring(result_doc, encoding='unicode', pretty_print=True)
        
    except Exception as e:
        logger.error(f"XSLT transformation failed: {str(e)}", exc_info=True)
        # Fall back to the raw diff output if XSLT fails
        redline_html = diff_html
        logger.warning("Falling back to raw diff output without XSLT transformation")
    
    # Compile statistics
    stats = {
        "total_changes": insertions + deletions + moves,
        "insertions": insertions,
        "deletions": deletions,
        "moves": moves
    }
    
    return redline_html, stats


def _postprocess_redline(
    redline_html: str,
    include_css: bool = True,
    add_navigation: bool = True,
    output_format: str = "html"
) -> str:
    """Post-processes the redline HTML to add styling and navigation.
    
    Args:
        redline_html: The raw redline HTML
        include_css: Whether to include default CSS
        add_navigation: Whether to add JS navigation
        output_format: 'html' for full document or 'fragment' for body content
        
    Returns:
        Final redline HTML with styling and navigation
    """
    if output_format == "fragment":
        # Extract just the body content if fragment requested
        try:
            soup = BeautifulSoup(redline_html, 'html.parser')
            body = soup.body or soup
            redline_html = str(body)
            
            # Add minimal wrapper for CSS and JS if needed
            if include_css or add_navigation:
                redline_html = f"<div class='redline-container'>{redline_html}</div>"
                
                css_block = f"<style>{DEFAULT_CSS}</style>" if include_css else ""
                js_block = _get_navigation_js() if add_navigation else ""
                
                redline_html = f"{css_block}{redline_html}{js_block}"
        except Exception as e:
            logger.warning(f"Fragment extraction failed: {str(e)}, returning full HTML")
            # Fall back to full HTML if extraction fails
            output_format = "html"
    
    if output_format == "html":
        # Process as full HTML document
        try:
            soup = BeautifulSoup(redline_html, 'html.parser')
            
            # Find or create head element
            head = soup.head
            if not head:
                head = soup.new_tag("head")
                if soup.html:
                    soup.html.insert(0, head)
                else:
                    # Create html element if missing
                    html = soup.new_tag("html")
                    html.append(head)
                    if soup.body:
                        html.append(soup.body)
                        soup.body.extract()
                    soup.append(html)
            
            # Add CSS if requested
            if include_css:
                style = soup.new_tag("style")
                style.string = DEFAULT_CSS
                head.append(style)
            
            # Add navigation script if requested
            if add_navigation:
                script = soup.new_tag("script")
                script.string = _get_navigation_js().replace('<script>', '').replace('</script>', '')
                head.append(script)
                
                # Add navigation UI
                nav_div = soup.new_tag("div")
                nav_div["class"] = "redline-navigation"
                nav_div["style"] = "position:fixed; top:10px; right:10px; background:#f0f0f0; padding:10px; border-radius:5px; z-index:1000;"
                
                prev_btn = soup.new_tag("button")
                prev_btn["onclick"] = "goPrevChange()"
                prev_btn.string = "Previous Change"
                
                next_btn = soup.new_tag("button")
                next_btn["onclick"] = "goNextChange()"
                next_btn.string = "Next Change"
                
                stat_span = soup.new_tag("span")
                stat_span["id"] = "change-counter"
                stat_span["style"] = "margin-left:10px;"
                stat_span.string = "Change: - / -"
                
                nav_div.append(prev_btn)
                nav_div.append(next_btn)
                nav_div.append(stat_span)
                
                # Add to body
                body = soup.body
                if body:
                    body.insert(0, nav_div)
            
            redline_html = str(soup)
            
        except Exception as e:
            logger.warning(f"HTML post-processing failed: {str(e)}, returning original redline HTML")
            # If post-processing fails, return the original redline HTML
            pass
    
    return redline_html


def _get_navigation_js() -> str:
    """Returns JavaScript for navigating between changes."""
    return """<script>
// Find all changes in the document
function findAllChanges() {
    return document.querySelectorAll('ins, del, .diff-insert, .diff-delete, .diff-move-source, .diff-move-target');
}

// Global variables for navigation
let changes = [];
let currentIndex = -1;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    changes = Array.from(findAllChanges());
    updateCounter();
});

// Navigate to previous change
function goPrevChange() {
    if (changes.length === 0) {
        changes = Array.from(findAllChanges());
        if (changes.length === 0) return;
    }
    
    currentIndex = (currentIndex <= 0) ? changes.length - 1 : currentIndex - 1;
    navigateToChange();
}

// Navigate to next change
function goNextChange() {
    if (changes.length === 0) {
        changes = Array.from(findAllChanges());
        if (changes.length === 0) return;
    }
    
    currentIndex = (currentIndex >= changes.length - 1) ? 0 : currentIndex + 1;
    navigateToChange();
}

// Scroll to current change and update counter
function navigateToChange() {
    const change = changes[currentIndex];
    change.scrollIntoView({behavior: 'smooth', block: 'center'});
    
    // Highlight the current change
    changes.forEach(c => c.style.outline = '');
    change.style.outline = '2px solid orange';
    
    updateCounter();
}

// Update the change counter display
function updateCounter() {
    const counter = document.getElementById('change-counter');
    if (counter && changes.length > 0) {
        counter.textContent = `Change: ${currentIndex + 1} / ${changes.length}`;
    }
}
</script>"""


@with_tool_metrics
@with_error_handling
async def compare_documents_redline(
    original_text: str,
    modified_text: str,
    file_format: str = "auto",
    detect_moves: bool = True,
    output_format: str = "html",
    diff_level: str = "word"
) -> Dict[str, Any]:
    """Creates a redline comparison between two text documents (non-HTML).
    
    Generates a "track changes" style redline showing differences between original and modified text.
    Unlike create_html_redline, this function is for plain text, Markdown, LaTeX, etc.,
    not for pre-existing HTML documents.
    
    Args:
        original_text: The original/old text document
        modified_text: The modified/new text document
        file_format: Format of input documents. Options: "auto", "text", "markdown", "latex".
                     Default "auto" (attempts to detect format).
        detect_moves: Whether to identify and highlight moved content. Default True.
        output_format: Output format. Options: "html", "text". Default "html".
        diff_level: Granularity of diff. Options: "char", "word", "line". Default "word".
        
    Returns:
        A dictionary containing:
        {
            "redline": The redlined document in the requested format,
            "stats": {
                "total_changes": Total number of changes detected,
                "insertions": Number of insertions,
                "deletions": Number of deletions,
                "moves": Number of moved segments (if detect_moves=True),
            },
            "processing_time": Time in seconds to generate the redline,
            "success": True if successful
        }
        
    Raises:
        ToolInputError: If input parameters are invalid
        ToolError: If processing fails for other reasons
    """
    start_time = time.time()
    
    # Validate input
    if not original_text or not isinstance(original_text, str):
        raise ToolInputError("Original text must be a non-empty string.")
    if not modified_text or not isinstance(modified_text, str):
        raise ToolInputError("Modified text must be a non-empty string.")
    
    if file_format not in ["auto", "text", "markdown", "latex"]:
        raise ToolInputError(
            f"Invalid file_format: '{file_format}'. Must be 'auto', 'text', 'markdown', or 'latex'.",
            param_name="file_format"
        )
    
    if output_format not in ["html", "text"]:
        raise ToolInputError(
            f"Invalid output_format: '{output_format}'. Must be 'html' or 'text'.",
            param_name="output_format"
        )
    
    if diff_level not in ["char", "word", "line"]:
        raise ToolInputError(
            f"Invalid diff_level: '{diff_level}'. Must be 'char', 'word', or 'line'.",
            param_name="diff_level"
        )
    
    # Auto-detect format if needed
    if file_format == "auto":
        file_format = _detect_file_format(original_text)
        logger.info(f"Auto-detected file format: {file_format}")
    
    try:
        # Convert to HTML if needed for some formats
        if file_format == "markdown":
            try:
                # Try to import markdown library
                import markdown
                original_html = markdown.markdown(original_text)
                modified_html = markdown.markdown(modified_text)
                
                # Generate HTML redline
                result = await create_html_redline(
                    original_html=original_html,
                    modified_html=modified_html,
                    detect_moves=detect_moves,
                    output_format="html"
                )
                
                redline = result["redline_html"]
                stats = result["stats"]
                
            except ImportError:
                logger.warning("Markdown library not available, falling back to text diff")
                file_format = "text"
        
        # For plain text, use difflib or custom text diff
        if file_format == "text":
            redline, stats = _generate_text_redline(
                original_text, 
                modified_text,
                diff_level=diff_level,
                detect_moves=detect_moves,
                output_format=output_format
            )
            
        # For LaTeX, specialized handling could be added here
        if file_format == "latex":
            # For now, treat LaTeX as plain text
            redline, stats = _generate_text_redline(
                original_text, 
                modified_text,
                diff_level=diff_level,
                detect_moves=detect_moves,
                output_format=output_format
            )
            
        processing_time = time.time() - start_time
        
        logger.success(
            f"Document redline generated successfully ({stats['total_changes']} changes)",
            emoji_key=TaskType.DOCUMENT.value,
            changes=stats,
            time=processing_time,
            format=file_format
        )
        
        return {
            "redline": redline,
            "stats": stats,
            "processing_time": processing_time,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error generating document redline: {str(e)}", exc_info=True)
        raise ToolError(
            f"Failed to generate document redline: {str(e)}",
            error_code="REDLINE_GENERATION_ERROR",
            details={"error": str(e)}
        ) from e


def _detect_file_format(text: str) -> str:
    """Detects file format based on content.
    
    Args:
        text: The text content to analyze
        
    Returns:
        Detected format: "markdown", "latex", or "text"
    """
    # Check for Markdown indicators
    md_patterns = [
        r'^#\s+',      # Headers
        r'^-\s+',      # List items
        r'^!\[.+\]',   # Images
        r'\[.+\]\(.+\)',  # Links
        r'^\*\*.*\*\*$',  # Bold text
        r'^\*.*\*$',      # Italic text
    ]
    
    md_score = 0
    for pattern in md_patterns:
        if re.search(pattern, text, re.MULTILINE):
            md_score += 1
    
    # Check for LaTeX indicators
    latex_patterns = [
        r'\\begin{document}',
        r'\\section{',
        r'\\subsection{',
        r'\\usepackage{',
        r'\\documentclass',
        r'\$\$.+\$\$',  # Display math
        r'\$.+\$',      # Inline math
    ]
    
    latex_score = 0
    for pattern in latex_patterns:
        if re.search(pattern, text, re.MULTILINE):
            latex_score += 1
    
    # Determine format based on scores
    if latex_score >= 2:
        return "latex"
    elif md_score >= 2:
        return "markdown"
    else:
        return "text"


def _generate_text_redline(
    original_text: str,
    modified_text: str,
    diff_level: str = "word",
    detect_moves: bool = True,
    output_format: str = "html"
) -> Tuple[str, Dict[str, int]]:
    """Generates a redline comparison between two text documents.
    
    Args:
        original_text: Original text content
        modified_text: Modified text content
        diff_level: Level of diff granularity ('char', 'word', 'line')
        detect_moves: Whether to detect moved blocks
        output_format: Output format ('html' or 'text')
        
    Returns:
        Tuple of (redline, stats_dict)
    """
    # Split the text according to diff level
    if diff_level == "char":
        original_units = list(original_text)
        modified_units = list(modified_text)
    elif diff_level == "word":
        original_units = re.findall(r'\S+|\s+', original_text)
        modified_units = re.findall(r'\S+|\s+', modified_text)
    else:  # line
        original_units = original_text.splitlines(True)
        modified_units = modified_text.splitlines(True)
    
    # Calculate diff
    matcher = difflib.SequenceMatcher(None, original_units, modified_units)
    
    # Statistics
    insertions = 0
    deletions = 0
    moves = 0
    
    # Output in selected format
    if output_format == "html":
        # Build HTML output
        result = []
        
        # Add CSS if it's HTML
        css = """<style>
        ins { color: blue; text-decoration: none; }
        del { color: red; text-decoration: line-through; }
        .move { color: green; }
        </style>"""
        result.append(css)
        
        # Process each diff block
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Unchanged text
                result.append(''.join(original_units[i1:i2]))
            elif tag == 'replace':
                # Changed text (replacement)
                deletions += 1
                insertions += 1
                result.append(f'<del>{"".join(original_units[i1:i2])}</del>')
                result.append(f'<ins>{"".join(modified_units[j1:j2])}</ins>')
            elif tag == 'delete':
                # Deleted text
                deletions += 1
                result.append(f'<del>{"".join(original_units[i1:i2])}</del>')
            elif tag == 'insert':
                # Inserted text
                insertions += 1
                result.append(f'<ins>{"".join(modified_units[j1:j2])}</ins>')
        
        # Detect moves if requested
        if detect_moves:
            result_html = ''.join(result)
            result_html = _detect_text_moves(result_html)
            # Count moves (approximate by counting move classes)
            moves = result_html.count('class="move"')
            
            # Wrap with HTML document structure if missing
            if not result_html.strip().startswith("<!DOCTYPE") and not result_html.strip().startswith("<html"):
                result_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document Redline</title>
</head>
<body>
{result_html}
</body>
</html>"""
            
            redline = result_html
            
        else:
            # Skip move detection
            redline = ''.join(result)
            # Wrap with HTML document structure if missing
            if not redline.strip().startswith("<!DOCTYPE") and not redline.strip().startswith("<html"):
                redline = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document Redline</title>
</head>
<body>
{redline}
</body>
</html>"""
    
    else:  # text output format
        # Build plain text output with markers
        result = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Unchanged text
                result.append(''.join(original_units[i1:i2]))
            elif tag == 'replace':
                # Changed text (replacement)
                deletions += 1
                insertions += 1
                result.append(f'[-{"".join(original_units[i1:i2])}-]')
                result.append(f'{{+{"".join(modified_units[j1:j2])}+}}')
            elif tag == 'delete':
                # Deleted text
                deletions += 1
                result.append(f'[-{"".join(original_units[i1:i2])}-]')
            elif tag == 'insert':
                # Inserted text
                insertions += 1
                result.append(f'{{+{"".join(modified_units[j1:j2])}+}}')
        
        # Note: Move detection not implemented for text output
        redline = ''.join(result)
    
    # Compile statistics
    stats = {
        "total_changes": insertions + deletions + moves,
        "insertions": insertions,
        "deletions": deletions,
        "moves": moves
    }
    
    return redline, stats


def _detect_text_moves(html_diff: str) -> str:
    """Detects and marks moved content in HTML diff.
    
    This is a simplified implementation that identifies identical
    content that appears as both deleted and inserted, marking it as moved.
    
    Args:
        html_diff: HTML with <ins> and <del> tags
        
    Returns:
        HTML with moved content marked
    """
    # Parse HTML
    soup = BeautifulSoup(html_diff, 'html.parser')
    
    # Find all deletions and insertions
    deletions = soup.find_all('del')
    insertions = soup.find_all('ins')
    
    # Create content maps (text → elements)
    del_map = {}
    for el in deletions:
        content = el.get_text().strip()
        if content and len(content) > 10:  # Only consider substantial content
            if content not in del_map:
                del_map[content] = []
            del_map[content].append(el)
    
    # Look for matching content in insertions
    for el in insertions:
        content = el.get_text().strip()
        if content and content in del_map and len(content) > 10:
            # Found a potential move
            matching_dels = del_map[content]
            if matching_dels:
                # Mark as moved content
                el['class'] = el.get('class', []) + ['move']
                
                # Mark the first matching deletion as moved and remove from map
                matching_del = matching_dels.pop(0)
                matching_del['class'] = matching_del.get('class', []) + ['move']
                
                if not matching_dels:
                    del del_map[content]
    
    return str(soup)