"""HTML redline tool for LLM Gateway.

This module provides tools for creating high-quality redlines (track changes) 
of HTML documents, similar to those used by legal firms and for SEC filings.
"""
import base64
import difflib
import html as html_stdlib  # Import standard html library for escape with a clear alias
import markdown
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from bs4 import BeautifulSoup
from lxml import etree
from lxml import html as lxml_html  # Import lxml.html with an alias to avoid conflict
from xmldiff import formatting, main
# Import specific action types needed for subclassing
from xmldiff.actions import InsertNode, DeleteNode, MoveNode, UpdateAttrib, UpdateTextIn

from llm_gateway.constants import TaskType
from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.redline")

# --- Constants for output styling ---
# DEFAULT_CSS is removed - styling will be handled by Tailwind and custom style block

# --- Custom Formatter with _handle_str ---
class SafeXMLFormatter(formatting.XMLFormatter):
    """Subclass of XMLFormatter that handles all diff_trees output action types."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_actions = {
            'insertions': 0,
            'deletions': 0,
            'moves': 0,
        }
        # Add reference to the xpath mapping - will be useful for moves
        self.xpath_to_element = {}
        
    def _get_element_by_xpath(self, xpath, tree):
        """Helper to retrieve element by xpath, with caching."""
        if xpath in self.xpath_to_element:
            return self.xpath_to_element[xpath]
            
        # Support both string and tuple representations of XPath
        xpath_str = xpath if isinstance(xpath, str) else ''.join(xpath)
        
        try:
            # Attempt to find element directly
            elements = tree.xpath(xpath_str)
            if elements and len(elements) > 0:
                element = elements[0]
                self.xpath_to_element[xpath] = element
                return element
        except Exception as e:
            logger.debug(f"XPath evaluation failed: {str(e)} for xpath: {xpath_str}")
        
        return None
    
    def _handle_insert(self, action, parent):
        """Handle InsertNode actions - standard implementation with counter."""
        try:
            super()._handle_insert(action, parent)
            self.processed_actions['insertions'] += 1
        except Exception as e:
            logger.warning(f"Error in _handle_insert: {str(e)}")
                
    def _handle_delete(self, action, parent):
        """Handle DeleteNode actions - standard implementation with counter."""
        try:
            super()._handle_delete(action, parent)
            self.processed_actions['deletions'] += 1
        except Exception as e:
            logger.warning(f"Error in _handle_delete: {str(e)}")
            
    def _handle_move(self, action, parent):
        """Handle MoveNode actions - standard implementation with counter."""
        try:
            super()._handle_move(action, parent)
            self.processed_actions['moves'] += 1
        except Exception as e:
            logger.warning(f"Error in _handle_move: {str(e)}")
    
    def _handle_str(self, action, parent):
        """Handles string content that needs to be inserted into the tree."""
        try:
            # String content from action
            content = action[1] if len(action) > 1 else ""
            
            # No valid content, nothing to do
            if not content:
                return
                
            # Add the text content to the appropriate place
            if not parent.text and len(parent) == 0:
                # No children and no text yet - add as text
                parent.text = (parent.text or "") + content
            elif len(parent) > 0:
                # Has children - add to tail of last child
                last_child = parent[-1]
                last_child.tail = (last_child.tail or "") + content
            else:
                # Has text but no children - append to text
                parent.text = (parent.text or "") + content
        except Exception as e:
            logger.warning(f"Error in _handle_str: {str(e)}")
            
    def _handle_xpath_move(self, action, parent, tree):
        """Handles move operations expressed as XPath pairs."""
        try:
            # Expected format: (source_xpath, target_xpath)
            if len(action) < 2:
                return

            source_xpath = action[0]
            target_xpath = action[1]
            
            # Get elements by XPath
            source_element = self._get_element_by_xpath(source_xpath, tree)
            target_element = self._get_element_by_xpath(target_xpath, tree)
            
            if source_element is not None and target_element is not None:
                # Generate a unique move ID
                move_id = f"move-{hash(str(source_xpath))}-{hash(str(target_xpath))}"
                
                # Add diff:move-from attribute to source
                source_element.attrib['{http://namespaces.shoobx.com/diff}move-from'] = move_id
                
                # Add diff:move-to attribute to target
                target_element.attrib['{http://namespaces.shoobx.com/diff}move-to'] = move_id
                
                self.processed_actions['moves'] += 1
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error in _handle_xpath_move: {str(e)}")
            return False
            
    def _handle_xpath_delete(self, action, parent, tree):
        """Handles delete operations expressed as XPath."""
        try:
            # Expected format: (xpath_to_delete,)
            if len(action) < 1:
                return False
                
            xpath = action[0]
            element = self._get_element_by_xpath(xpath, tree)
            
            if element is not None:
                # Mark this element as deleted
                element.attrib['{http://namespaces.shoobx.com/diff}delete'] = 'true'
                self.processed_actions['deletions'] += 1
                return True
                
            return False
        except Exception as e:
            logger.warning(f"Error in _handle_xpath_delete: {str(e)}")
            return False
            
    def _handle_xpath_insert(self, action, parent, tree):
        """Handles insert operations expressed as XPath."""
        try:
            # Expected format could be (xpath_where_to_insert, content)
            if len(action) < 2:
                return False
                
            xpath = action[0]
            content = action[1]
            
            element = self._get_element_by_xpath(xpath, tree)
            
            if element is not None and isinstance(content, str):
                # This is tricky - we might need to create a new element
                # For now, mark the target with an insert attribute
                element.attrib['{http://namespaces.shoobx.com/diff}insert'] = 'true'
                
                # If it's just text, we can try to insert it
                if not element.text:
                    element.text = content
                else:
                    element.text += content
                    
                self.processed_actions['insertions'] += 1
                return True
                
            return False
        except Exception as e:
            logger.warning(f"Error in _handle_xpath_insert: {str(e)}")
            return False
    
    def _handle_generic_xpath_action(self, action, parent, tree):
        """Handles any XPath action that doesn't match specific patterns.
        
        This is a more general handler for XPath-based actions that don't fit into
        the standard move, delete, or insert patterns. These are often context-specific
        paths indicating changes to specific elements in the XML/HTML structure.
        
        Args:
            action: The action tuple with xpath as first element
            parent: The parent element
            tree: The source document tree
            
        Returns:
            True if the action was handled, False otherwise
        """
        try:
            xpath = action[0]
            
            # Ensure xpath is properly formatted for lxml
            # If it doesn't start with /, add it (for paths like 'html/body', etc.)
            if not xpath.startswith('/') and not xpath.startswith('./'):
                xpath = f"/{xpath}"
                
            # Try to find the target element - more robust matching
            element = None
            try:
                # First try direct xpath
                elements = tree.xpath(xpath)
                if elements and len(elements) > 0:
                    element = elements[0]
            except Exception:
                # If that fails, try more permissive approaches
                try:
                    # Try a more general approach to find the element
                    # For example, convert '/div/h1[1]' to '//div//h1[1]'
                    relaxed_xpath = xpath.replace('/', '//')
                    elements = tree.xpath(relaxed_xpath)
                    if elements and len(elements) > 0:
                        element = elements[0]
                except Exception:
                    # If still failing, try to extract just the last part of the path
                    # e.g., extract 'h1[1]' from '/div/h1[1]'
                    parts = xpath.strip('/').split('/')
                    if parts:
                        last_part = parts[-1]
                        try:
                            elements = tree.xpath(f"//{last_part}")
                            if elements and len(elements) > 0:
                                element = elements[0]
                        except Exception:
                            pass
            
            if element is None:
                return False
                
            # Determine action type based on content
            action_type = 'unknown'
            content = None
            
            # Try to infer action type from structure
            if len(action) > 1:
                content = action[1]
                if isinstance(content, str) and len(content) > 0:
                    # Check if second element is another xpath
                    if isinstance(content, str) and (content.startswith('/') or content.startswith('html') or content.startswith('div')):
                        action_type = 'move'  # Source -> Target XPath
                    else:
                        action_type = 'update'  # Could be updating text content
                elif isinstance(content, (list, tuple)) and len(content) > 0:
                    action_type = 'complex'  # Complex nested structure
                else:
                    action_type = 'reference'  # Just referencing the element
            else:
                action_type = 'reference'  # Just the xpath, no content
                
            # Process based on inferred action type
            if action_type == 'move' and content:
                # Handle as a move operation - mark both source and target
                # Try to find target element
                target_xpath = content
                if not target_xpath.startswith('/') and not target_xpath.startswith('./'):
                    target_xpath = f"/{target_xpath}"
                    
                try:
                    target_elements = tree.xpath(target_xpath)
                    if target_elements and len(target_elements) > 0:
                        target_element = target_elements[0]
                        # Generate a unique ID for this move
                        move_id = f"move-{hash(xpath)}-{hash(target_xpath)}"
                        
                        # Mark source and target
                        element.attrib['{http://namespaces.shoobx.com/diff}move-from'] = move_id
                        target_element.attrib['{http://namespaces.shoobx.com/diff}move-to'] = move_id
                        
                        self.processed_actions['moves'] += 1
                        return True
                except Exception:
                    # Fall back to regular update if move target can't be found
                    action_type = 'update'
            
            if action_type == 'update' and content:
                # Handle as text update
                if not element.text:
                    element.text = content
                else:
                    element.text += content
                
                # Mark as changed for visualization
                element.attrib['{http://namespaces.shoobx.com/diff}update'] = 'true'
                self.processed_actions['insertions'] += 1
                return True
                
            elif action_type == 'reference':
                # For simple references, just mark the element as referenced
                # This helps in visualization even if we don't modify it
                element.attrib['{http://namespaces.shoobx.com/diff}reference'] = 'true'
                return True
                
            elif action_type == 'complex':
                # Handle complex nested action by marking the element
                element.attrib['{http://namespaces.shoobx.com/diff}complex'] = 'true'
                return True
                
            # If we get here, we couldn't handle the action specifically
            # But we've at least marked the element, so return True
            element.attrib['{http://namespaces.shoobx.com/diff}generic'] = 'true'
            logger.debug(f"Handled generic XPath reference for: {xpath}")
            return True
            
        except Exception as e:
            logger.warning(f"Error in _handle_generic_xpath_action: {str(e)}")
            return False
    
    def handle_action(self, action, parent):
        """Intelligently handle all action types, including string and XPath references."""
        # Extract action information
        if not action or len(action) == 0:
            return

        action_type = action[0]
        
        # CASE 1: Handle standard action types via parent method
        if hasattr(action_type, '__name__'):
            action_name = action_type.__name__.lower()
            
            # Handle using standard methods if available
            if action_name == 'insertnode':
                self._handle_insert(action, parent)
                return
            elif action_name == 'deletenode':
                self._handle_delete(action, parent)
                return
            elif action_name == 'movenode':
                self._handle_move(action, parent)
                return
            elif action_name == 'updateattrib':
                self._handle_updateattrib(action, parent)
                return
            elif action_name == 'updatetextin':
                self._handle_updatetext(action, parent)
                return
        
        # CASE 2: Handle string content directly
        if action_type == 'str' or (isinstance(action_type, str) and action_type == 'str'):
            self._handle_str(action, parent)
            return
            
        # CASE 3: XPath action - any string action that looks like an XPath
        # This consolidated approach handles all XPath formats
        if isinstance(action_type, str) and (action_type.startswith('/') or action_type.startswith('html') or action_type.startswith('div')):
            # We have what appears to be an XPath
            # Try to handle it with our generic handler which will detect the appropriate action type
            if self._handle_generic_xpath_action(action, parent, self.source_doc):
                return
        
        # CASE 4: Tuple or list of paths/elements
        if isinstance(action_type, (tuple, list)):
            # Might be a complex nested action - try to infer what it is
            # For now, just log it
            logger.debug(f"Complex nested action encountered: {action[:2]}")
            return
            
        # If we get here, we couldn't handle the action
        logger.warning(f"Unhandled action type: {type(action_type)} - {str(action_type)[:50]}")

    def format(self, actions, source_doc):
        """Override to set source document and track stats before formatting."""
        self.source_doc = source_doc  # Store for XPath lookups
        self.processed_actions = {'insertions': 0, 'deletions': 0, 'moves': 0}
        return super().format(actions, source_doc)


# --- XSLT for transforming xmldiff output to readable HTML ---
XMLDIFF_XSLT = b"""<?xml version="1.0" encoding="UTF-8"?>
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
  
  <xsl:template match="@diff:update">
    <ins class="diff-update"><xsl:apply-templates/></ins>
  </xsl:template>
  
  <xsl:template match="@diff:reference">
    <span class="diff-reference"><xsl:apply-templates/></span>
  </xsl:template>
  
  <xsl:template match="@diff:complex">
    <span class="diff-complex"><xsl:apply-templates/></span>
  </xsl:template>
  
  <xsl:template match="@diff:generic">
    <span class="diff-generic"><xsl:apply-templates/></span>
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
                    original_doc = lxml_html.parse(str(orig_path))
                    modified_doc = lxml_html.parse(str(mod_path))
                    
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
        redline_html, diff_stats = await _generate_redline(
            original_doc,
            modified_doc,
            detect_moves=detect_moves,
            formatting_tags=formatting_tags
        )
        
        # Post-process the output (add CSS, JS navigation if requested)
        redline_html = await _postprocess_redline(
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
            emoji_key="update",
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
        parser = lxml_html.HTMLParser()
        try:
            original_doc = lxml_html.fromstring(original_html, parser=parser)
            modified_doc = lxml_html.fromstring(modified_html, parser=parser)
        except Exception as e:
            # Fallback to BeautifulSoup if lxml fails (for malformed HTML)
            logger.warning(f"lxml parsing failed, falling back to BeautifulSoup: {str(e)}")
            original_soup = BeautifulSoup(original_html, 'html.parser')
            modified_soup = BeautifulSoup(modified_html, 'html.parser')
            original_html = str(original_soup)
            modified_html = str(modified_soup)
            original_doc = lxml_html.fromstring(original_html, parser=parser)
            modified_doc = lxml_html.fromstring(modified_html, parser=parser)
        
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


# --- Helper to Normalize Diff Actions ---
def _normalize_diff_actions(raw_actions: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    """
    Filters and potentially normalizes the raw action list from diff_trees.

    Keeps known action types and logs/discards unknown ones.
    """
    normalized_actions = []
    # Define known action types/names expected by the formatter
    # Using lowercase string names for broader compatibility
    known_action_names = {'insert', 'delete', 'move', 'updateattrib', 'updatetext', 'str'}
    # Add specific action types if needed, e.g., from xmldiff.actions import InsertNode, ...
    # known_action_classes = {InsertNode, DeleteNode, MoveNode, UpdateAttrib, UpdateTextIn}

    for action in raw_actions:
        action_type = action[0]
        # Get a string representation of the action type name
        action_name = getattr(action_type, '__name__', str(action_type)).lower()

        # Check if the action name is one we know how to handle
        if action_name in known_action_names:
            # Basic approach: pass known actions through directly.
            # More advanced: Could consolidate adjacent 'str' actions here.
            normalized_actions.append(action)
        else:
            # Log discarded unknown actions for debugging
            # Log only the first few elements for brevity
            action_preview = repr(action[:2]) + ('...' if len(action) > 2 else '')
            logger.warning(
                f"Ignoring unknown action type '{action_name}' in diff tree. Action preview: {action_preview}",
                emoji_key="warning"
            )
            # Discard the unknown action

    return normalized_actions


async def _generate_redline(
    original_doc: etree._Element,
    modified_doc: etree._Element,
    detect_moves: bool = True,
    formatting_tags: List[str] = None
) -> Tuple[str, Dict[str, int]]:
    """Generates redline HTML showing differences between documents.
    Uses diff_trees with our enhanced SafeXMLFormatter to handle all action types,
    including XPath references and string fragments when detect_moves=True.
    Falls back to diff_texts if detect_moves=False.
    """
    # Log the xmldiff version being used for diagnostic purposes
    try:
        import importlib.metadata
        xmldiff_version = importlib.metadata.version('xmldiff')
        logger.debug(f"Using xmldiff version: {xmldiff_version}")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        logger.debug("Could not determine xmldiff version")
    
    # Configure the SafeXMLFormatter for processing the diff
    formatter = SafeXMLFormatter(
        normalize=formatting.WS_BOTH if detect_moves else formatting.WS_NONE,
        pretty_print=True,
        text_tags=('p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', 'th', 'div', 'span', 'a'),
        formatting_tags=formatting_tags or []
    )

    diff_html = ""
    stats = {"insertions": 0, "deletions": 0, "moves": 0, "total_changes": 0}

    try:
        if detect_moves:
            # Step 1: Get raw actions from diff_trees with detect_moves enabled
            raw_actions = main.diff_trees(
                original_doc,
                modified_doc,
                # Supply any necessary options
            )

            # Step 2: Apply the formatter which will process all action types
            # The formatter will automatically handle xpath references and string actions
            diff_html = formatter.format(raw_actions, original_doc)
            
            # Get statistics from our enhanced formatter
            stats["insertions"] = formatter.processed_actions.get('insertions', 0)
            stats["deletions"] = formatter.processed_actions.get('deletions', 0)
            stats["moves"] = formatter.processed_actions.get('moves', 0)

        else:
            # Use diff_texts when move detection is off
            original_string = etree.tostring(original_doc, encoding='unicode')
            modified_string = etree.tostring(modified_doc, encoding='unicode')

            # Use diff_texts directly - produces XML with diff attributes
            diff_html = main.diff_texts(
                original_string,
                modified_string,
                formatter=formatter  # Still pass our formatter for consistent handling
            )
            
            # Count stats from the output attributes or from formatter if available
            if hasattr(formatter, 'processed_actions'):
                stats["insertions"] = formatter.processed_actions.get('insertions', 0)
                stats["deletions"] = formatter.processed_actions.get('deletions', 0)
                stats["moves"] = formatter.processed_actions.get('moves', 0)
            else:
                # Fallback stats counting from XML attributes
                stats["insertions"] = diff_html.count('diff:insert')
                stats["deletions"] = diff_html.count('diff:delete')
                stats["moves"] = diff_html.count('diff:move-from') # Count move attributes

        # Update total changes stat
        stats["total_changes"] = stats["insertions"] + stats["deletions"] + stats["moves"]

    except Exception as e:
        logger.error(f"xmldiff processing failed: {str(e)}", exc_info=True)
        raise ToolError(
            f"XML diff processing failed: {str(e)}",
            error_code="XMLDIFF_ERROR"
        ) from e

    # Store intermediate XML for fallback
    intermediate_xml_for_fallback = diff_html
    
    # Apply XSLT transformation
    redline_html = ""
    try:
        xslt_doc = etree.fromstring(XMLDIFF_XSLT) # Already bytes
        transform = etree.XSLT(xslt_doc)
        parser = etree.HTMLParser(encoding='utf-8')
        # Operate on diff_html (which is XML string with diff:* attrs)
        diff_doc = etree.fromstring(diff_html.encode('utf-8'), parser=parser)
        result_doc = transform(diff_doc)
        redline_html = etree.tostring(result_doc, encoding='unicode', method='html', pretty_print=True)
    except Exception as e:
        logger.error(f"XSLT transformation failed: {str(e)}", exc_info=True)
        redline_html = intermediate_xml_for_fallback # Use the stored intermediate XML as fallback
        logger.warning("Falling back to raw diff output without XSLT transformation")

    return redline_html, stats


async def _postprocess_redline(
    redline_html: str,
    include_css: bool = True,
    add_navigation: bool = True,
    output_format: str = "html"
) -> str:
    """Post-processes the redline HTML to add Tailwind/Font styling and navigation.
    
    Args:
        redline_html: The raw redline HTML
        include_css: Whether to include default CSS
        add_navigation: Whether to add JS navigation
        output_format: 'html' for full document or 'fragment' for body content
        
    Returns:
        Final redline HTML with styling and navigation
    """
    final_html = redline_html # Start with the input

    # Use BeautifulSoup for manipulation
    try:
        soup = BeautifulSoup(final_html, 'html.parser')

        # --- Handle Fragment Output ---
        if output_format == "fragment":
            body_content = soup.body or soup # Get body or root if no body tag
            # Convert body content back to string
            final_html = ''.join(str(item) for item in body_content.contents)

            # If including styling/nav for fragment, wrap it and add head elements separately
            if include_css or add_navigation:
                head_elements = []
                if include_css:
                     # Tailwind CDN Script
                     head_elements.append('<script src="https://cdn.tailwindcss.com"></script>')
                     # Google Font Link (Newsreader)
                     head_elements.append('<link rel="preconnect" href="https://fonts.googleapis.com">')
                     head_elements.append('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
                     head_elements.append('<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet">')
                     # Custom CSS for redlines
                     head_elements.append(f"""<style type="text/tailwindcss">
                        @tailwind base;
                        @tailwind components;
                        @tailwind utilities;

                        @layer base {{
                            ins, .diff-insert {{
                                @apply text-blue-600 bg-blue-100 no-underline px-0.5 rounded-sm;
                            }}
                            ins:hover, .diff-insert:hover {{
                                @apply bg-blue-200;
                            }}
                            del, .diff-delete {{
                                @apply text-red-600 bg-red-100 line-through px-0.5 rounded-sm;
                            }}
                            del:hover, .diff-delete:hover {{
                                @apply bg-red-200;
                            }}
                            .diff-move-target {{
                                @apply text-green-700 bg-green-100 no-underline px-0.5 rounded-sm;
                            }}
                            .diff-move-target:hover {{
                                @apply bg-green-200;
                            }}
                            .diff-move-source {{
                                @apply text-green-700 bg-green-100 line-through px-0.5 rounded-sm;
                            }}
                            .diff-move-source:hover {{
                                @apply bg-green-200;
                            }}
                            .diff-update {{
                                @apply text-purple-600 bg-purple-100 no-underline px-0.5 rounded-sm;
                            }}
                            .diff-update:hover {{
                                @apply bg-purple-200;
                            }}
                            .diff-reference {{
                                @apply text-gray-600 border border-dotted border-gray-300 px-0.5 rounded-sm;
                            }}
                            .diff-complex {{
                                @apply text-orange-600 border border-dotted border-orange-300 px-0.5 rounded-sm;
                            }}
                            .diff-generic {{
                                @apply text-indigo-600 border border-dotted border-indigo-300 px-0.5 rounded-sm;
                            }}
                        }}
                    </style>""")
                if add_navigation:
                    js_content = _get_navigation_js().replace('<script>', '').replace('</script>', '')
                    head_elements.append(f"<script>{js_content}</script>")

                # Combine head elements, wrap fragment with prose div
                head_html = "\\n".join(head_elements)
                # Apply prose classes for typography - use responsive variants
                prose_classes = "prose max-w-none prose-sm sm:prose-base lg:prose-lg xl:prose-xl 2xl:prose-2xl"
                body_wrapper = f'<body class="font-[\'Newsreader\']"><div class="{prose_classes}">{final_html}</div></body>'

                # Reconstruct a minimal HTML structure
                final_html = f"<!DOCTYPE html><html><head>{head_html}</head>{body_wrapper}</html>"

        # --- Handle Full HTML Document Output ---
        elif output_format == "html":
            # Find or create head element
            head = soup.head
            if not head:
                head = soup.new_tag("head")
                if soup.html:
                    soup.html.insert(0, head)
                else: # Handle case where even <html> is missing
                    html_tag = soup.new_tag("html")
                    html_tag.append(head)
                    # Move existing top-level elements into the new html tag
                    for element in list(soup.contents):
                        if element is not head:
                            html_tag.append(element.extract())
                    soup.append(html_tag)

            # Add CSS/Font links if requested
            if include_css:
                # Tailwind CDN Script
                tw_script = soup.new_tag("script", src="https://cdn.tailwindcss.com")
                head.append(tw_script)
                # Google Font Link (Newsreader)
                font_preconnect1 = soup.new_tag("link", rel="preconnect", href="https://fonts.googleapis.com")
                font_preconnect2 = soup.new_tag("link", rel="preconnect", href="https://fonts.gstatic.com", crossorigin="")
                font_link = soup.new_tag("link", href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap", rel="stylesheet")
                head.append(font_preconnect1)
                head.append(font_preconnect2)
                head.append(font_link)
                # Custom CSS for redlines using Tailwind directives
                style_tag = soup.new_tag("style", type="text/tailwindcss")
                style_tag.string = """
                    @tailwind base;
                    @tailwind components;
                    @tailwind utilities;

                    @layer base {
                        ins, .diff-insert {
                            @apply text-blue-600 bg-blue-100 no-underline px-0.5 rounded-sm;
                        }
                        ins:hover, .diff-insert:hover {
                            @apply bg-blue-200;
                        }
                        del, .diff-delete {
                            @apply text-red-600 bg-red-100 line-through px-0.5 rounded-sm;
                        }
                        del:hover, .diff-delete:hover {
                            @apply bg-red-200;
                        }
                        .diff-move-target {
                            @apply text-green-700 bg-green-100 no-underline px-0.5 rounded-sm;
                        }
                        .diff-move-target:hover {
                            @apply bg-green-200;
                        }
                        .diff-move-source {
                            @apply text-green-700 bg-green-100 line-through px-0.5 rounded-sm;
                        }
                        .diff-move-source:hover {
                            @apply bg-green-200;
                        }
                        .diff-update {
                            @apply text-purple-600 bg-purple-100 no-underline px-0.5 rounded-sm;
                        }
                        .diff-update:hover {
                            @apply bg-purple-200;
                        }
                        .diff-reference {
                            @apply text-gray-600 border border-dotted border-gray-300 px-0.5 rounded-sm;
                        }
                        .diff-complex {
                            @apply text-orange-600 border border-dotted border-orange-300 px-0.5 rounded-sm;
                        }
                        .diff-generic {
                            @apply text-indigo-600 border border-dotted border-indigo-300 px-0.5 rounded-sm;
                        }
                    }
                """
                head.append(style_tag)


            # Add navigation script if requested
            if add_navigation:
                script_tag = soup.new_tag("script")
                script_tag.string = _get_navigation_js().replace('<script>', '').replace('</script>', '')
                # Append script to head or end of body? End of body is usually better for perf.
                target_for_script = soup.body or head # Add to body if exists, else head
                target_for_script.append(script_tag)

                # Add navigation UI (keep existing logic)
                nav_div = soup.find("div", class_="redline-navigation") # Check if exists
                if not nav_div:
                     nav_div = soup.new_tag("div")
                     nav_div["class"] = ["redline-navigation"] # Start with list
                     # Apply some basic Tailwind for the nav box
                     nav_div['class'].extend(['fixed', 'top-2', 'right-2', 'bg-gray-100', 'p-2', 'rounded', 'shadow', 'z-50', 'text-xs'])

                     # --- Safely Add Classes to Buttons ---
                     prev_btn = soup.new_tag("button", attrs={"onclick": "goPrevChange()"})
                     prev_classes = prev_btn.get('class', [])
                     if isinstance(prev_classes, str): prev_classes = prev_classes.split()
                     prev_btn['class'] = prev_classes + ['bg-white', 'hover:bg-gray-200', 'px-2', 'py-1', 'rounded', 'mr-1']
                     prev_btn.string = "Prev"

                     next_btn = soup.new_tag("button", attrs={"onclick": "goNextChange()"})
                     next_classes = next_btn.get('class', [])
                     if isinstance(next_classes, str): next_classes = next_classes.split()
                     next_btn['class'] = next_classes + ['bg-white', 'hover:bg-gray-200', 'px-2', 'py-1', 'rounded', 'mr-1']
                     next_btn.string = "Next"
                     # --- End Safe Class Handling ---

                     stat_span = soup.new_tag("span", attrs={"id": "change-counter", "class": "ml-2"})
                     stat_span.string = "-/-"

                     nav_div.append(prev_btn)
                     nav_div.append(next_btn)
                     nav_div.append(stat_span)

                     body = soup.body
                     if body:
                         body.insert(0, nav_div) # Insert nav at the beginning of body


            # Apply base font and prose styles to body/wrapper
            body = soup.body
            if body:
                # Add base font class - Safely
                body_classes = body.get('class', [])
                if isinstance(body_classes, str): body_classes = body_classes.split()
                if 'font-["Newsreader"]' not in body_classes:
                     body['class'] = body_classes + ['font-["Newsreader"]']

                # Wrap direct children of body in a prose div if not already structured
                content_wrapper = soup.new_tag("div")
                # Apply responsive prose classes
                content_wrapper['class'] = ["prose", "max-w-none", "prose-sm", "sm:prose-base", "lg:prose-lg", "xl:prose-xl", "2xl:prose-2xl"]

                # Find elements to wrap (all direct children except the nav div)
                elements_to_wrap = [child for child in body.contents if child is not nav_div and getattr(child, 'name', None)]

                # Avoid re-wrapping if already wrapped
                existing_prose_wrapper = body.find('div', class_='prose', recursive=False)
                if not existing_prose_wrapper:
                    for element in elements_to_wrap:
                        content_wrapper.append(element.extract()) # Move element into wrapper

                    # Insert the wrapper back into the body (after nav if present)
                    if nav_div and nav_div in body.contents:
                        nav_div.insert_after(content_wrapper)
                    else:
                        body.insert(0, content_wrapper) # Or at the beginning if no nav

            # Convert back to string
            final_html = str(soup)

    except Exception as e:
        logger.warning(f"HTML post-processing failed: {str(e)}, returning original redline HTML", exc_info=True)
        # If post-processing fails, return the original redline HTML passed in
        return redline_html # Return original unprocessed HTML

    return final_html


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
    
    # Handle special case: if original_text and modified_text are extremely similar strings
    # but we know they should be different (like LLM-generated content), 
    # use a different comparison method
    if original_text == modified_text:
        logger.warning("Original and modified texts are identical. No changes to show.")
        redline = modified_text
        stats = {"insertions": 0, "deletions": 0, "moves": 0, "total_changes": 0}
        processing_time = time.time() - start_time
        return {
            "redline": redline,
            "stats": stats,
            "processing_time": processing_time,
            "success": True
        }
    
    # If the texts are too similar but not identical (often the case with LLM outputs)
    # force character-level diff for better visualization
    if len(original_text) > 100 and len(modified_text) > 100:
        similarity_ratio = difflib.SequenceMatcher(None, original_text, modified_text).ratio()
        if similarity_ratio > 0.95:
            logger.info(f"Texts are very similar (ratio: {similarity_ratio:.2f}). Forcing character-level diff for better visualization.")
            diff_level = "char"
    
    redline = ""
    stats = {}

    try:
        # --- Handle HTML Output Generation ---
        if output_format == "html":
            original_html_input = ""
            modified_html_input = ""
            render_failed = False
            if file_format == "markdown":
                try:
                    # Convert Markdown to HTML first
                    original_html_input = markdown.markdown(original_text, extensions=['fenced_code', 'tables'])
                    modified_html_input = markdown.markdown(modified_text, extensions=['fenced_code', 'tables'])
                except Exception as md_err:
                    logger.warning(f"Markdown conversion failed: {md_err}. Falling back to text diff.", emoji_key="warning")
                    render_failed = True
            # If Markdown rendering failed or format is text/latex, treat as plain text for HTML diff
            if render_failed or file_format in ["text", "latex"]:
                # Wrap plain text in <pre> tags to preserve whitespace and formatting 
                # Use standard library html.escape, not lxml.html
                original_html_input = f"<pre>{html_stdlib.escape(original_text)}</pre>"
                modified_html_input = f"<pre>{html_stdlib.escape(modified_text)}</pre>"
                file_format = "text" # Treat as text for logging purposes

            # Now call create_html_redline with the (potentially converted) HTML
            result = await create_html_redline(
                original_html=original_html_input,
                modified_html=modified_html_input,
                detect_moves=detect_moves,
                output_format="html", # Ensure full doc for postprocessing
                include_css=True, # Enable Tailwind/Font styling
                add_navigation=True
            )
            redline = result.get("redline_html", "")
            stats = result.get("stats", {})

        # --- Handle Plain Text Output Generation ---
        else: # output_format == "text"
            # Use the existing text diff generator
            redline, stats = _generate_text_redline(
                original_text,
                modified_text,
                diff_level=diff_level,
                detect_moves=detect_moves, # Note: move detection is limited for text output
                output_format=output_format # Pass 'text'
            )

        # --- Final Processing ---
        processing_time = time.time() - start_time
        
        logger.success(
            f"Document redline generated successfully ({stats.get('total_changes', 0)} changes)",
            emoji_key="update", # Use valid key
            changes=stats,
            time=processing_time,
            format=file_format # Log the original (or detected) format
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
        # Enhanced word splitting to better handle punctuation separately
        # This improves diff visualization by treating punctuation as separate units
        original_units = re.findall(r'[^\w\s]|\w+|\s+', original_text)
        modified_units = re.findall(r'[^\w\s]|\w+|\s+', modified_text)
    else:  # line
        original_units = original_text.splitlines(True)
        modified_units = modified_text.splitlines(True)
    
    # Calculate diff with auto-junk detection disabled for better accuracy
    # Especially important for similar texts (like LLM outputs)
    matcher = difflib.SequenceMatcher(None, original_units, modified_units, autojunk=False)
    
    # For very similar texts, use a more sensitive matching algorithm
    # This helps detect small differences better
    if len(original_text) > 100 and len(modified_text) > 100:
        similarity_ratio = matcher.ratio()
        if similarity_ratio > 0.9 and diff_level != "char":
            # Switch to character level for high-similarity texts
            logger.debug(f"High similarity detected ({similarity_ratio:.2f}) - Using char-level matching for precision")
            original_units = list(original_text)
            modified_units = list(modified_text)
            matcher = difflib.SequenceMatcher(None, original_units, modified_units, autojunk=False)
    
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
                result.append(html_stdlib.escape(''.join(original_units[i1:i2])))
            elif tag == 'replace':
                # Changed text (replacement)
                deletions += 1
                insertions += 1
                result.append(f'<del>{html_stdlib.escape("".join(original_units[i1:i2]))}</del>')
                result.append(f'<ins>{html_stdlib.escape("".join(modified_units[j1:j2]))}</ins>')
            elif tag == 'delete':
                # Deleted text
                deletions += 1
                result.append(f'<del>{html_stdlib.escape("".join(original_units[i1:i2]))}</del>')
            elif tag == 'insert':
                # Inserted text
                insertions += 1
                result.append(f'<ins>{html_stdlib.escape("".join(modified_units[j1:j2]))}</ins>')
        
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
    
    # Add special note if no changes detected in very similar but different texts
    if insertions == 0 and deletions == 0 and original_text != modified_text:
        note = """
        <div style="background-color: #ffffcc; padding: 10px; margin: 10px 0; border: 1px solid #e6e600;">
        <strong>Note:</strong> The texts appear nearly identical with only subtle differences that may not be visible in this view.
        </div>
        """
        if output_format == 'html':
            # Add note to HTML redline
            if "</body>" in redline:
                redline = redline.replace("</body>", f"{note}</body>")
            else:
                redline += note
        else:
            # Add plain text note to text redline
            redline += "\n\nNote: The texts appear nearly identical with only subtle differences that may not be visible in this view."
    
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