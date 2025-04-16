"""HTML redline tool for LLM Gateway.

This module provides tools for creating high-quality redlines (track changes)
of HTML documents, similar to those used by legal firms and for SEC filings.

Uses xmldiff for structural comparison and move detection, and lxml/BeautifulSoup
for parsing and manipulation.
"""
import base64
import difflib
import html as html_stdlib # Import standard html library for escape with a clear alias
import markdown
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from bs4 import BeautifulSoup
from lxml import etree
from lxml import html as lxml_html # Import lxml.html with an alias to avoid conflict
from xmldiff import formatting, main
# Import specific action types needed for subclassing and handling
from xmldiff.actions import (
    DeleteNode, InsertNode, MoveNode, UpdateAttrib, UpdateTextIn, RenameAttrib, InsertAttrib, DeleteAttrib
)

from llm_gateway.constants import TaskType
from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.redline")

# --- Custom Formatter to Apply diff:* Attributes ---
class RedlineXMLFormatter(formatting.XMLFormatter):
    """
    Subclass of XMLFormatter that applies diff:* attributes to the source tree
    based on the actions provided by xmldiff.diff_trees.

    This formatter modifies the source tree *in place* by adding attributes like
    diff:insert, diff:delete, diff:move-from, diff:move-to, diff:update-*, etc.
    The resulting XML tree can then be transformed via XSLT to produce the final
    redline HTML.
    """
    DIFF_NS = "http://namespaces.shoobx.com/diff"
    DIFF = "{%s}" % DIFF_NS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_actions = {
            'insertions': 0,
            'deletions': 0,
            'moves': 0,
            'text_updates': 0,
            'attr_updates': 0,
            'other_changes': 0,
        }
        self.move_map = {} # Track moves: original_xpath -> target_xpath

    def _add_diff_attribute(self, element: etree._Element, attr_name: str, value: str = 'true'):
        """Helper to add a namespaced diff attribute."""
        if element is not None:
            element.set(f"{self.DIFF}{attr_name}", str(value))

    # --- Override Handlers to Apply Attributes ---

    def _handle_insert(self, action: InsertNode, parent: etree._Element):
        """Handles InsertNode actions by adding the node and marking it."""
        try:
            # super()._handle_insert(action, parent) # Don't call super, it modifies structure differently
            node_to_insert = action.node

            # Mark the node itself as inserted before adding
            self._add_diff_attribute(node_to_insert, 'insert')
            # Also mark children recursively? Maybe not needed if parent is marked.

            # Insert the node at the correct position
            if action.position is not None:
                parent.insert(action.position, node_to_insert)
            else:
                parent.append(node_to_insert) # Append if position not specified

            self.processed_actions['insertions'] += 1
            logger.debug(f"Handled InsertNode: added {node_to_insert.tag} to {parent.tag} at pos {action.position}")
        except Exception as e:
            logger.warning(f"Error in _handle_insert: {str(e)} processing action: {action}")

    def _handle_delete(self, action: DeleteNode, parent: etree._Element):
        """Handles DeleteNode actions by finding the node via XPath and marking it."""
        try:
            # super()._handle_delete(action, parent) # Don't call super, we mark instead of remove
            target_node = self._get_node_by_xpath(action.node, self.source_doc) # Find node in original doc
            if target_node is not None:
                # Check if this node was part of a move
                node_xpath = self.source_doc.getroottree().getpath(target_node)
                if node_xpath in self.move_map:
                    # This deletion is the source of a move
                    move_target_xpath = self.move_map[node_xpath]
                    move_id = f"move-{hash(node_xpath)}-{hash(move_target_xpath)}"
                    self._add_diff_attribute(target_node, 'move-from', move_id)
                    self.processed_actions['moves'] += 1 # Count here as the 'move'
                    logger.debug(f"Handled DeleteNode (as move source): marked {target_node.tag} at {node_xpath} with move-id {move_id}")
                else:
                    # It's a simple deletion
                    self._add_diff_attribute(target_node, 'delete')
                    self.processed_actions['deletions'] += 1
                    logger.debug(f"Handled DeleteNode: marked {target_node.tag} at {node_xpath} as deleted")
            else:
                 logger.warning(f"Could not find node to delete for action: {action.node}")

        except Exception as e:
            logger.warning(f"Error in _handle_delete: {str(e)} processing action: {action}")

    def _handle_move(self, action: MoveNode, parent: etree._Element):
        """Handles MoveNode actions by mapping source/target and marking later."""
        try:
            # Don't modify the tree here. Instead, record the move.
            # The actual marking happens when we process the corresponding
            # DeleteNode (for move-from) and InsertNode (for move-to).
            # However, xmldiff might represent moves differently in some cases.
            # Let's assume MoveNode implies a direct mapping.
            # We need the XPath of the source and target nodes IN THE ORIGINAL TREE

            source_node = self._get_node_by_xpath(action.node, self.source_doc)
            target_parent_node = self._get_node_by_xpath(action.target, self.source_doc) # Target parent

            if source_node is not None and target_parent_node is not None:
                 source_xpath = self.source_doc.getroottree().getpath(source_node)
                 # Construct target path based on parent and position
                 target_base_xpath = self.source_doc.getroottree().getpath(target_parent_node)
                 target_xpath = f"{target_base_xpath}/*[{action.position + 1}]" # Approximate target path

                 # Store this mapping to be used by _handle_delete and _handle_insert
                 self.move_map[source_xpath] = target_xpath
                 # Note: We don't increment 'moves' counter here, it's counted in _handle_delete/insert

                 logger.debug(f"Registered MoveNode: {source_xpath} -> {target_xpath}")

                 # We still need to handle the insertion part for the visual diff
                 # Let's try adding the node marked as move-to in the structure
                 moved_node_copy = etree.fromstring(etree.tostring(source_node)) # Create a copy
                 move_id = f"move-{hash(source_xpath)}-{hash(target_xpath)}"
                 self._add_diff_attribute(moved_node_copy, 'move-to', move_id)
                 target_parent_node.insert(action.position, moved_node_copy)

                 # And mark the original source node (will be handled by _handle_delete later)
                 self._add_diff_attribute(source_node, 'move-source-pending', move_id) # Temporary mark

            else:
                 logger.warning(f"Could not find source or target for MoveNode action: {action}")

            # super()._handle_move(action, parent) # Don't call super

        except Exception as e:
            logger.warning(f"Error in _handle_move: {str(e)} processing action: {action}")

    def _handle_update_text(self, action: UpdateTextIn, parent: etree._Element):
        """Handles UpdateTextIn actions by marking the node and adding diff text."""
        try:
            target_node = self._get_node_by_xpath(action.node, self.source_doc)
            if target_node is not None:
                original_text = target_node.text or ""
                new_text = action.text

                # Perform a simple word-level diff on the text itself
                text_diff_html = self._generate_inline_text_diff(original_text, new_text)

                # Clear existing text/children and replace with parsed diff html
                target_node.clear() # Remove existing children and text
                try:
                    # Parse the diff HTML fragment
                    # Use a dummy root to parse fragment
                    diff_elements = lxml_html.fragments_fromstring(text_diff_html)
                    if diff_elements:
                        # If the first element is a text node, assign it to target_node.text
                        if isinstance(diff_elements[0], str):
                             target_node.text = diff_elements[0]
                             start_index = 1
                        else:
                             target_node.text = None # Ensure no initial text if first is element
                             start_index = 0
                        # Append remaining elements as children
                        for elem in diff_elements[start_index:]:
                            if isinstance(elem, str):
                                # Append text to the tail of the last child
                                if len(target_node):
                                    last_child = target_node[-1]
                                    last_child.tail = (last_child.tail or "") + elem
                                else: # No children yet, append to text
                                    target_node.text = (target_node.text or "") + elem
                            else:
                                target_node.append(elem)

                except Exception as parse_err:
                     logger.warning(f"Failed to parse inline text diff HTML, falling back: {parse_err}. Diff HTML: {text_diff_html}")
                     # Fallback: just mark the node and set new text
                     self._add_diff_attribute(target_node, 'update-text')
                     target_node.text = new_text

                self.processed_actions['text_updates'] += 1
                logger.debug(f"Handled UpdateTextIn: node {target_node.tag}, new text applied.")
            else:
                logger.warning(f"Could not find node for UpdateTextIn action: {action.node}")

        except Exception as e:
            logger.warning(f"Error in _handle_update_text: {str(e)} processing action: {action}")

    def _handle_update_attrib(self, action: UpdateAttrib, parent: etree._Element):
        """Handles UpdateAttrib actions by marking the node and the attribute change."""
        try:
            target_node = self._get_node_by_xpath(action.node, self.source_doc)
            if target_node is not None:
                attr_name = action.name
                new_value = action.value
                old_value = target_node.get(attr_name, None)

                # Mark the node as having an attribute update
                self._add_diff_attribute(target_node, 'update-attrib', f"{attr_name}")
                # Optionally store old value:
                # self._add_diff_attribute(target_node, f'update-attrib-{attr_name}-old', str(old_value))

                # Apply the new attribute value
                target_node.set(attr_name, new_value)

                self.processed_actions['attr_updates'] += 1
                logger.debug(f"Handled UpdateAttrib: node {target_node.tag}, attr {attr_name} set to {new_value}")
            else:
                logger.warning(f"Could not find node for UpdateAttrib action: {action.node}")

        except Exception as e:
            logger.warning(f"Error in _handle_update_attrib: {str(e)} processing action: {action}")

    def _handle_rename_attrib(self, action: RenameAttrib, parent: etree._Element):
        """Handles RenameAttrib actions."""
        try:
            target_node = self._get_node_by_xpath(action.node, self.source_doc)
            if target_node is not None:
                old_name = action.old_name
                new_name = action.new_name
                value = target_node.get(old_name)

                if value is not None:
                    # Mark the node
                    self._add_diff_attribute(target_node, 'rename-attrib', f"{old_name}->{new_name}")
                    # Perform rename
                    del target_node.attrib[old_name]
                    target_node.set(new_name, value)
                    self.processed_actions['attr_updates'] += 1
                    logger.debug(f"Handled RenameAttrib: node {target_node.tag}, renamed {old_name} to {new_name}")
            else:
                logger.warning(f"Could not find node for RenameAttrib action: {action.node}")
        except Exception as e:
            logger.warning(f"Error in _handle_rename_attrib: {str(e)} processing action: {action}")

    def _handle_insert_attrib(self, action: InsertAttrib, parent: etree._Element):
        """Handles InsertAttrib actions."""
        try:
            target_node = self._get_node_by_xpath(action.node, self.source_doc)
            if target_node is not None:
                attr_name = action.name
                value = action.value
                 # Mark the node
                self._add_diff_attribute(target_node, 'insert-attrib', f"{attr_name}")
                # Add attribute
                target_node.set(attr_name, value)
                self.processed_actions['attr_updates'] += 1
                logger.debug(f"Handled InsertAttrib: node {target_node.tag}, added attr {attr_name}={value}")
            else:
                 logger.warning(f"Could not find node for InsertAttrib action: {action.node}")
        except Exception as e:
            logger.warning(f"Error in _handle_insert_attrib: {str(e)} processing action: {action}")

    def _handle_delete_attrib(self, action: DeleteAttrib, parent: etree._Element):
        """Handles DeleteAttrib actions."""
        try:
            target_node = self._get_node_by_xpath(action.node, self.source_doc)
            if target_node is not None:
                attr_name = action.name
                if attr_name in target_node.attrib:
                    # Mark the node
                    self._add_diff_attribute(target_node, 'delete-attrib', f"{attr_name}")
                    # Delete attribute (optional, could leave it but marked)
                    # del target_node.attrib[attr_name]
                    self.processed_actions['attr_updates'] += 1
                    logger.debug(f"Handled DeleteAttrib: node {target_node.tag}, marked attr {attr_name} for deletion")
            else:
                 logger.warning(f"Could not find node for DeleteAttrib action: {action.node}")
        except Exception as e:
             logger.warning(f"Error in _handle_delete_attrib: {str(e)} processing action: {action}")

    def _get_node_by_xpath(self, xpath: str, tree: etree._ElementTree) -> Optional[etree._Element]:
        """Helper to find a node by its XPath in the source document."""
        try:
            # Ensure tree is the root element if needed
            root = tree if isinstance(tree, etree._Element) else tree.getroot()
            # Lxml xpath needs element paths, not document paths usually
            # We assume xpath provided by xmldiff is relative to the root
            # If xpath starts with '/', adjust it
            # Note: xmldiff paths might already be absolute from the document root
            # Let's try both absolute and relative-from-root

            nodes = root.xpath(xpath)
            if nodes:
                return nodes[0]

            # Try adjusting if it looks like an absolute path from document
            if xpath.startswith('/'):
                 parts = xpath.split('/')[1:] # Skip first empty part
                 if parts and parts[0] == root.tag: # Matches root tag
                      relative_xpath = './' + '/'.join(parts[1:])
                      nodes = root.xpath(relative_xpath)
                      if nodes:
                           return nodes[0]

            logger.debug(f"Node not found for XPath: {xpath}")
            return None
        except Exception as e:
            logger.warning(f"XPath evaluation failed for '{xpath}': {e}")
            return None

    def _generate_inline_text_diff(self, text1: str, text2: str) -> str:
        """Generates simple HTML diff for text content using difflib."""
        text1 = text1 or ""
        text2 = text2 or ""
        # Use word level diffing for better readability
        matcher = difflib.SequenceMatcher(None, text1.split(), text2.split())
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result.append(html_stdlib.escape(' '.join(text1.split()[i1:i2])))
            elif tag == 'replace':
                result.append(f'<del class="diff-delete-text">{html_stdlib.escape(" ".join(text1.split()[i1:i2]))}</del>')
                result.append(f'<ins class="diff-insert-text">{html_stdlib.escape(" ".join(text2.split()[j1:j2]))}</ins>')
            elif tag == 'delete':
                result.append(f'<del class="diff-delete-text">{html_stdlib.escape(" ".join(text1.split()[i1:i2]))}</del>')
            elif tag == 'insert':
                result.append(f'<ins class="diff-insert-text">{html_stdlib.escape(" ".join(text2.split()[j1:j2]))}</ins>')
            if tag != 'equal': # Add space between diff tags if needed
                result.append(' ')
        # Join parts, ensuring spaces are handled correctly
        return ''.join(result).strip() # Join potentially separated by spaces and strip ends

    def handle_action(self, action: Any, parent: etree._Element):
        """Dispatch actions to specific handlers based on type."""
        action_type = type(action)

        handler_map = {
            InsertNode: self._handle_insert,
            DeleteNode: self._handle_delete,
            MoveNode: self._handle_move,
            UpdateTextIn: self._handle_update_text,
            UpdateAttrib: self._handle_update_attrib,
            RenameAttrib: self._handle_rename_attrib,
            InsertAttrib: self._handle_insert_attrib,
            DeleteAttrib: self._handle_delete_attrib,
            # Add other action types from xmldiff.actions if needed
        }

        handler = handler_map.get(action_type)

        if handler:
            try:
                # Pass the action object itself, not unpacked tuple
                handler(action, parent)
            except Exception as e:
                logger.error(f"Error executing handler for {action_type.__name__}: {e}", exc_info=True)
        else:
            # Fallback for unhandled known types or unexpected types
            logger.warning(f"Unhandled action type encountered: {action_type.__name__} - Action: {str(action)[:100]}")
            self.processed_actions['other_changes'] += 1
            # Optionally mark the parent or a relevant node as having an 'unknown' change
            # self._add_diff_attribute(parent, 'unknown-change', str(action_type.__name__))


    def format(self, actions: List[Any], source_doc: etree._ElementTree) -> etree._ElementTree:
        """
        Applies the diff actions to the source_doc by adding diff:* attributes.
        Returns the modified source_doc tree.
        """
        self.source_doc = source_doc # Store original tree for lookups
        self.processed_actions = {k: 0 for k in self.processed_actions} # Reset stats
        self.move_map = {} # Reset move map

        # We need the root element to pass as 'parent' initially, though actions might target elsewhere
        root = source_doc.getroot()

        # Process actions - they modify self.source_doc in place
        for action in actions:
            # The 'parent' argument here is tricky as actions contain their own target paths.
            # We pass the root, but handlers should use action.node/action.target XPath with self.source_doc
            self.handle_action(action, root)

        # Post-process moves: Find nodes marked 'move-source-pending' and add 'move-from'
        # This ensures the source node is marked even if the InsertNode for move-to wasn't processed first
        pending_moves = root.xpath(f"//*[@*[local-name()='move-source-pending']]")
        for node in pending_moves:
             move_id = node.get(f"{self.DIFF}move-source-pending")
             if move_id:
                 self._add_diff_attribute(node, 'move-from', move_id)
                 node.attrib.pop(f"{self.DIFF}move-source-pending") # Remove pending mark
                 self.processed_actions['moves'] += 1 # Count moves here finally


        # Return the modified source document tree
        return self.source_doc


# --- XSLT for transforming xmldiff output to readable HTML ---
# Added templates for update-*, move-source, move-target, and inline text diffs
XMLDIFF_XSLT = b"""<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:diff="http://namespaces.shoobx.com/diff"
    exclude-result-prefixes="diff">

  <!-- Copy everything by default -->
  <xsl:template match="@*|node()">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>

  <!-- Handle Insertions -->
  <xsl:template match="*[@diff:insert]">
    <ins class="diff-insert">
      <xsl:copy>
        <xsl:apply-templates select="@*|node()"/>
      </xsl:copy>
    </ins>
  </xsl:template>

  <!-- Handle Deletions (that are not move sources) -->
  <xsl:template match="*[@diff:delete and not(@diff:move-from)]">
    <del class="diff-delete">
      <xsl:copy>
        <xsl:apply-templates select="@*[not(starts-with(name(), 'diff:'))]|node()"/>
      </xsl:copy>
    </del>
  </xsl:template>

  <!-- Handle Move Targets -->
  <xsl:template match="*[@diff:move-to]">
    <ins class="diff-move-target" data-move-id="{@diff:move-to}">
      <xsl:copy>
        <!-- Remove diff attributes from the copy -->
        <xsl:apply-templates select="@*[not(starts-with(name(), 'diff:'))]|node()"/>
      </xsl:copy>
    </ins>
  </xsl:template>

  <!-- Handle Move Sources -->
  <xsl:template match="*[@diff:move-from]">
    <del class="diff-move-source" data-move-id="{@diff:move-from}">
      <xsl:copy>
        <!-- Remove diff attributes from the copy -->
        <xsl:apply-templates select="@*[not(starts-with(name(), 'diff:'))]|node()"/>
      </xsl:copy>
    </del>
  </xsl:template>

  <!-- Handle Text Updates (Mark the container, rely on inner ins/del for details) -->
  <xsl:template match="*[@diff:update-text]">
     <span class="diff-update-container"> <!-- Container for text change -->
         <xsl:copy>
             <xsl:apply-templates select="@*[not(starts-with(name(), 'diff:'))]|node()"/>
         </xsl:copy>
     </span>
  </xsl:template>

  <!-- Handle Attribute Updates (Mark the container) -->
  <xsl:template match="*[@diff:update-attrib or @diff:rename-attrib or @diff:insert-attrib or @diff:delete-attrib]">
     <span class="diff-attrib-change"> <!-- Mark element with attribute change -->
         <xsl:copy>
             <!-- Apply templates to attributes *except* diff ones -->
             <xsl:apply-templates select="@*[not(starts-with(name(), 'diff:'))]"/>
             <!-- Apply templates to child nodes -->
             <xsl:apply-templates select="node()"/>
         </xsl:copy>
     </span>
  </xsl:template>

  <!-- Strip diff:* attributes from the final output -->
  <xsl:template match="@diff:insert|@diff:delete|@diff:move-to|@diff:move-from|@diff:update-text|@diff:update-attrib|@diff:rename-attrib|@diff:insert-attrib|@diff:delete-attrib"/>

  <!-- Keep the inline diff tags generated by _generate_inline_text_diff -->
  <xsl:template match="ins[@class='diff-insert-text'] | del[@class='diff-delete-text']">
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
    - Deletions in RED with strikethrough (`<del class="diff-delete">`)
    - Additions in BLUE (`<ins class="diff-insert">`)
    - Moved content Source in GREEN strikethrough (`<del class="diff-move-source">`)
    - Moved content Target in GREEN background/underline (`<ins class="diff-move-target">`)
    - Text Updates marked inline (`<ins class="diff-insert-text">`, `<del class="diff-delete-text">`)
    - Attribute changes marked on element (`<span class="diff-attrib-change">`)

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
        ignore_whitespace: Whether to ignore trivial whitespace differences via xmldiff normalization.
                         Default True.
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
                "text_updates": Number of text content updates,
                "attr_updates": Number of attribute updates,
                "other_changes": Number of other/unhandled changes
            },
            "processing_time": Time in seconds to generate the redline,
            "success": True if successful
        }

    Raises:
        ToolInputError: If input HTML is invalid or parameters are incorrect
        ToolError: If processing fails for other reasons
    """
    start_time = time.time()

    # --- Input Validation (Same as before) ---
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

    # --- Default Formatting Tags (Same as before) ---
    if formatting_tags is None:
        formatting_tags = ['b', 'strong', 'i', 'em', 'u', 'span', 'font', 'sub', 'sup']

    try:
        # --- Preprocessing (Parsing, Tidying, Tempfiles) ---
        # Use existing _preprocess_html_docs but ensure it returns lxml trees
        original_doc_root, modified_doc_root = _preprocess_html_docs(
            original_html,
            modified_html,
            ignore_whitespace=False, # Let xmldiff handle whitespace normalization
            use_tempfiles=use_tempfiles
        )

        # Need ElementTree objects for xmldiff
        original_doc_tree = etree.ElementTree(original_doc_root)
        modified_doc_tree = etree.ElementTree(modified_doc_root)

        # --- Generate Diff using xmldiff ---
        redline_html, diff_stats = await _generate_redline_with_xmldiff(
            original_doc_tree,
            modified_doc_tree,
            detect_moves=detect_moves,
            formatting_tags=formatting_tags,
            normalize_whitespace=ignore_whitespace # Pass flag to xmldiff
        )

        # --- Post-process Output (CSS, Navigation) ---
        redline_html = await _postprocess_redline(
            redline_html,
            include_css=include_css,
            add_navigation=add_navigation,
            output_format=output_format
        )

        processing_time = time.time() - start_time

        # --- Result Formatting (Base64 for large output, Logging) ---
        redline_size = len(redline_html.encode('utf-8'))
        base64_encoded = None
        if redline_size > 10_000_000: # 10MB threshold
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
    ignore_whitespace: bool = True, # Keep param, but logic moved to xmldiff
    use_tempfiles: bool = False
) -> Tuple[etree._Element, etree._Element]:
    """Parses and preprocesses HTML documents for comparison.

    Args:
        original_html: Original HTML string
        modified_html: Modified HTML string
        ignore_whitespace: Flag (currently unused here, handled by xmldiff)
        use_tempfiles: Whether to use temp files for parsing large docs

    Returns:
        Tuple of (original_doc_root, modified_doc_root) as lxml Element objects

    Raises:
        ToolError: If HTML parsing fails
    """
    original_doc = None
    modified_doc = None

    # --- Use Temp Files if requested ---
    if use_tempfiles and (len(original_html) > 1_000_000 or len(modified_html) > 1_000_000):
        logger.info("Using temporary files for large document parsing")
        orig_path, mod_path = None, None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as orig_file, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as mod_file:
                orig_file.write(original_html)
                mod_file.write(modified_html)
                orig_path = Path(orig_file.name)
                mod_path = Path(mod_file.name)

            parser = lxml_html.HTMLParser(recover=True) # Use recover=True for robustness
            original_doc = lxml_html.parse(str(orig_path), parser=parser)
            modified_doc = lxml_html.parse(str(mod_path), parser=parser)
            logger.debug("Successfully parsed HTML from temporary files.")

        except Exception as e:
            logger.warning(f"Failed to parse HTML from files: {str(e)}, falling back to in-memory parsing")
            original_doc, modified_doc = None, None # Force fallback
        finally:
            # Clean up temp files
            try:
                if orig_path and orig_path.exists(): orig_path.unlink()
                if mod_path and mod_path.exists(): mod_path.unlink()
            except Exception as e_del:
                logger.warning(f"Failed to delete temporary files: {e_del}")

    # --- In-Memory Parsing (Default or Fallback) ---
    if original_doc is None or modified_doc is None:
        logger.debug("Using in-memory HTML parsing.")
        try:
            # Try HTML tidy if available for cleaning potentially malformed HTML
            use_external_tidy = _check_tidy_available()
            if use_external_tidy:
                logger.info("Using external HTML tidy for preprocessing")
                original_html = _run_html_tidy(original_html)
                modified_html = _run_html_tidy(modified_html)

            # Parse HTML documents using lxml with recovery
            parser = lxml_html.HTMLParser(recover=True, encoding='utf-8')
            try:
                original_doc_root = lxml_html.fromstring(original_html.encode('utf-8'), parser=parser)
                modified_doc_root = lxml_html.fromstring(modified_html.encode('utf-8'), parser=parser)
                 # Wrap roots in ElementTree if fromstring was used
                original_doc = etree.ElementTree(original_doc_root)
                modified_doc = etree.ElementTree(modified_doc_root)

            except (etree.XMLSyntaxError, etree.ParserError) as lxml_err:
                 logger.warning(f"lxml parsing failed ({lxml_err}), attempting BeautifulSoup fallback.")
                 try:
                    original_soup = BeautifulSoup(original_html, 'html.parser')
                    modified_soup = BeautifulSoup(modified_html, 'html.parser')
                    # Convert back to string and re-parse with lxml (BS often cleans it up)
                    original_doc_root = lxml_html.fromstring(str(original_soup).encode('utf-8'), parser=parser)
                    modified_doc_root = lxml_html.fromstring(str(modified_soup).encode('utf-8'), parser=parser)
                    original_doc = etree.ElementTree(original_doc_root)
                    modified_doc = etree.ElementTree(modified_doc_root)
                 except Exception as bs_err:
                     raise ToolError(f"Failed to parse HTML even with BeautifulSoup fallback: {bs_err}") from bs_err

            except Exception as e:
                raise ToolError(f"Failed during in-memory HTML parsing: {e}") from e

        except Exception as e:
            raise ToolError(f"Failed to parse HTML documents: {str(e)}", error_code="HTML_PARSING_ERROR") from e

    if original_doc is None or modified_doc is None:
         raise ToolError("HTML document parsing resulted in None.", error_code="HTML_PARSING_ERROR")

    # Return the root elements
    return original_doc.getroot(), modified_doc.getroot()


# --- Tidy Check/Run (Mostly unchanged, but ensure it handles encoding) ---
def _check_tidy_available() -> bool:
    """Checks if HTML tidy is available on the system."""
    try:
        result = subprocess.run(
            ["tidy", "-v"],
            capture_output=True, # Use capture_output for modern Python
            text=True, # Decode output as text
            timeout=2, # Increased timeout slightly
            check=False # Don't raise exception on non-zero exit
        )
        # Check stderr for version info, as -v often prints there
        return result.returncode == 0 and "HTML Tidy" in result.stderr
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError) as e:
        logger.debug(f"HTML tidy check failed: {e}")
        return False

def _run_html_tidy(html_content: str) -> str:
    """Runs HTML content through tidy to clean and normalize it."""
    tidied_content = html_content # Default to original if tidy fails
    temp_path = None
    try:
        # Use UTF-8 encoding explicitly
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(html_content)
            temp_path_str = temp_file.name # Get path before closing

        temp_path = Path(temp_path_str)

        # Run tidy with optimal settings for diff processing
        # Specify input/output encoding as UTF-8
        command = [
            "tidy",
            "-q",                   # Quiet mode
            "-m",                   # Modify input file in place
            "--tidy-mark", "no",    # No tidy meta tag
            "--drop-empty-elements", "no",
            "--wrap", "0",          # No line wrapping
            "--show-warnings", "no",# Suppress warnings
            "-utf8",                # Specify output encoding (and input assumed)
            str(temp_path)
        ]
        logger.debug(f"Running tidy command: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15, # Increased timeout for potentially larger files
            check=False, # Don't raise on error, check returncode
            encoding='utf-8' # Ensure consistent decoding
        )

        if result.returncode == 0 or result.returncode == 1: # Tidy returns 1 for warnings
             # Read the modified content back
            tidied_content = temp_path.read_text(encoding='utf-8')
            logger.info(f"HTML tidy completed (Return Code: {result.returncode}).")
            if result.stderr:
                 logger.debug(f"Tidy stderr: {result.stderr[:500]}...") # Log stderr for debugging
        else:
            logger.warning(f"HTML tidy failed with return code {result.returncode}. Stderr: {result.stderr[:500]}")
            # Keep original content if tidy failed badly

    except FileNotFoundError:
         logger.warning("HTML tidy command not found. Skipping tidy.")
    except (subprocess.SubprocessError, TimeoutError, OSError) as e:
        logger.warning(f"HTML tidy execution failed: {str(e)}, using original content")
    except Exception as e_read:
         logger.warning(f"Failed to read tidied file: {e_read}, using original content")
    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e_del:
                logger.warning(f"Failed to delete temporary tidy file: {e_del}")

    return tidied_content

async def _generate_redline_with_xmldiff(
    original_doc_tree: etree._ElementTree,
    modified_doc_tree: etree._ElementTree,
    detect_moves: bool = True,
    formatting_tags: Optional[List[str]] = None,
    normalize_whitespace: bool = True
) -> Tuple[str, Dict[str, int]]:
    """Generates redline HTML using xmldiff and the custom RedlineXMLFormatter."""

    try:
        import importlib.metadata
        xmldiff_version = importlib.metadata.version('xmldiff')
        logger.debug(f"Using xmldiff version: {xmldiff_version}")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        logger.debug("Could not determine xmldiff version")

    # --- Configure xmldiff ---
    # Note: xmldiff options have changed over versions. Adjust as needed.
    # Key options: diff_algo, unique_attributes, F (fast-match function)
    diff_options = {
        # 'fast_match': True, # Might speed up but reduce accuracy? Check xmldiff docs.
        'ratio_mode': 'accurate', # Or 'fast'
        'M': 0.6 # Similarity threshold for moves/updates
    }

    # --- Configure Formatter ---
    formatter = RedlineXMLFormatter(
        normalize=(formatting.WS_BOTH if normalize_whitespace else formatting.WS_NONE),
        pretty_print=False, # Pretty printing can interfere with whitespace diffs
        # Define tags where text content changes should be diffed inline
        text_tags=('p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', 'th', 'div', 'span', 'a', 'title', 'caption', 'label'),
        # Define tags considered purely formatting
        formatting_tags=formatting_tags or []
    )

    annotated_source_tree = None
    try:
        # --- Compute the Diff Actions ---
        logger.info("Computing diff actions using xmldiff.diff_trees...")
        actions = main.diff_trees(
            original_doc_tree,
            modified_doc_tree,
            diff_options=diff_options,
            # Pass detect_moves equivalent if option exists, else it's default behavior
        )
        logger.info(f"Found {len(actions)} raw diff actions.")

        # --- Apply Actions using Formatter ---
        # The formatter modifies the original_doc_tree in place
        logger.info("Applying diff actions using RedlineXMLFormatter...")
        annotated_source_tree = formatter.format(actions, original_doc_tree)
        logger.info("Diff actions applied, source tree annotated.")

        # Get statistics from the formatter
        stats = formatter.processed_actions
        stats["total_changes"] = sum(stats.values()) # Calculate total

    except Exception as e:
        logger.error(f"xmldiff processing failed: {str(e)}", exc_info=True)
        raise ToolError(f"XML diff processing failed: {str(e)}", error_code="XMLDIFF_ERROR") from e

    # --- Apply XSLT Transformation ---
    redline_html = ""
    try:
        logger.info("Applying XSLT transformation...")
        xslt_tree = etree.fromstring(XMLDIFF_XSLT)
        transform = etree.XSLT(xslt_tree)

        # Apply transformation to the annotated source tree
        result_doc = transform(annotated_source_tree)

        # Serialize the result to HTML string
        redline_html = etree.tostring(
            result_doc,
            encoding='unicode',
            method='html',
            pretty_print=True # Pretty print the final HTML output
        )
        logger.info("XSLT transformation successful.")

        # Basic validation: Check if output is non-empty HTML
        if not redline_html or not redline_html.strip().lower().startswith('<'):
             logger.warning(f"XSLT output seems invalid or empty. Length: {len(redline_html)}")
             # Fallback or raise error? For now, log and continue.
             # Maybe serialize the annotated tree without XSLT as fallback?
             if annotated_source_tree is not None:
                 logger.info("Falling back to annotated XML without XSLT transformation.")
                 redline_html = etree.tostring(annotated_source_tree, encoding='unicode', method='html', pretty_print=True)
             else:
                 raise ToolError("XSLT transformation failed and no annotated tree available.", error_code="XSLT_ERROR")

    except Exception as e:
        logger.error(f"XSLT transformation failed: {str(e)}", exc_info=True)
        # Fallback: Serialize the annotated tree directly if XSLT fails
        if annotated_source_tree is not None:
            logger.warning("Falling back to raw annotated diff output due to XSLT error.")
            try:
                redline_html = etree.tostring(annotated_source_tree, encoding='unicode', method='html', pretty_print=True)
            except Exception as fallback_err:
                 raise ToolError(f"XSLT failed and fallback serialization also failed: {fallback_err}", error_code="XSLT_FALLBACK_ERROR") from fallback_err
        else:
            raise ToolError(f"XSLT transformation failed: {str(e)}", error_code="XSLT_ERROR") from e

    return redline_html, stats


async def _postprocess_redline(
    redline_html: str,
    include_css: bool = True,
    add_navigation: bool = True,
    output_format: str = "html"
) -> str:
    """Post-processes the redline HTML to add Tailwind/Font styling and navigation."""

    if not redline_html:
        logger.warning("Received empty HTML for postprocessing. Returning empty string.")
        return ""

    final_html = redline_html # Start with the input

    try:
        # Use BeautifulSoup for robust manipulation, especially with potentially imperfect HTML from transform
        soup = BeautifulSoup(final_html, 'html.parser')

        # Ensure basic HTML structure exists if needed later
        if not soup.find('html'):
            # Wrap content in html/body if missing
            current_content = str(soup)
            soup = BeautifulSoup(f"<html><body>{current_content}</body></html>", 'html.parser')

        if not soup.head:
             head = soup.new_tag("head")
             if soup.html:
                 soup.html.insert(0, head)
             else: # Should not happen after above check, but defensive
                  soup.insert(0, head) # Add head at start
        if not soup.body:
             body = soup.new_tag("body")
             # Move existing content into body
             for element in list(soup.children):
                  if element.name not in ['html', 'head']:
                       body.append(element.extract())
             if soup.html:
                  soup.html.append(body)
             else:
                  soup.append(body) # Add body after head

        # --- Handle Fragment Output ---
        if output_format == "fragment":
            body_content = soup.body
            if body_content:
                 # Extract content, excluding the navigation div if added later
                 nav_div = body_content.find("div", class_="redline-navigation", recursive=False)
                 if nav_div:
                      nav_div.extract() # Remove nav if present in fragment source
                 final_html = body_content.decode_contents() # Get inner HTML of body
            else:
                 # Fallback if no body tag somehow
                 final_html = str(soup)

            # Note: Styling/JS for fragments is complex. Current approach adds them
            # back if requested, wrapping the fragment. This might not be ideal
            # for all use cases but provides styling.
            if include_css or add_navigation:
                head_elements_str = ""
                if include_css:
                    head_elements_str += '<script src="https://cdn.tailwindcss.com"></script>'
                    head_elements_str += '<link rel="preconnect" href="https://fonts.googleapis.com">'
                    head_elements_str += '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                    head_elements_str += '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet">'
                    head_elements_str += f'<style type="text/tailwindcss">{_get_tailwind_css()}</style>'
                if add_navigation:
                    js_content = _get_navigation_js().replace('<script>', '').replace('</script>', '')
                    head_elements_str += f"<script>{js_content}</script>"

                prose_classes = "prose max-w-none prose-sm sm:prose-base lg:prose-lg xl:prose-xl 2xl:prose-2xl"
                # Apply font *around* the prose wrapper
                body_wrapper = f'<body class="font-[\'Newsreader\']"><div class="{prose_classes}">{final_html}</div></body>'

                # Add navigation UI if requested for fragment context
                if add_navigation:
                     nav_html = """<div class="redline-navigation fixed top-2 right-2 bg-gray-100 p-2 rounded shadow z-50 text-xs">
                                     <button class="bg-white hover:bg-gray-200 px-2 py-1 rounded mr-1" onclick="goPrevChange()">Prev</button>
                                     <button class="bg-white hover:bg-gray-200 px-2 py-1 rounded mr-1" onclick="goNextChange()">Next</button>
                                     <span class="ml-2" id="change-counter">-/-</span>
                                   </div>"""
                     # Inject nav *inside* the body tag we created
                     body_wrapper = body_wrapper.replace('<body>', f'<body>{nav_html}')


                final_html = f"<!DOCTYPE html><html><head>{head_elements_str}</head>{body_wrapper}</html>"


        # --- Handle Full HTML Document Output ---
        elif output_format == "html":
            head = soup.head
            body = soup.body

            # Add CSS/Font links if requested
            if include_css:
                if not head.find("script", src="https://cdn.tailwindcss.com"):
                     head.append(BeautifulSoup('<script src="https://cdn.tailwindcss.com"></script>', 'html.parser'))
                if not head.find("link", href=lambda x: x and "fonts.googleapis.com" in x):
                     head.append(BeautifulSoup('<link rel="preconnect" href="https://fonts.googleapis.com">', 'html.parser'))
                     head.append(BeautifulSoup('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>', 'html.parser'))
                     head.append(BeautifulSoup('<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet">', 'html.parser'))
                if not head.find("style", type="text/tailwindcss"):
                     style_tag = soup.new_tag("style", type="text/tailwindcss")
                     style_tag.string = _get_tailwind_css()
                     head.append(style_tag)

            # Add navigation script if requested
            if add_navigation:
                if not body.find("script", string=lambda s: s and "findAllChanges" in s):
                     script_tag = soup.new_tag("script")
                     script_tag.string = _get_navigation_js().replace('<script>', '').replace('</script>', '')
                     body.append(script_tag) # Append script to end of body

                # Add navigation UI if it doesn't exist
                nav_div = body.find("div", class_="redline-navigation", recursive=False)
                if not nav_div:
                     nav_html = """<div class="redline-navigation fixed top-2 right-2 bg-gray-100 p-2 rounded shadow z-50 text-xs">
                                     <button class="bg-white hover:bg-gray-200 px-2 py-1 rounded mr-1" onclick="goPrevChange()">Prev</button>
                                     <button class="bg-white hover:bg-gray-200 px-2 py-1 rounded mr-1" onclick="goNextChange()">Next</button>
                                     <span class="ml-2" id="change-counter">-/-</span>
                                   </div>"""
                     body.insert(0, BeautifulSoup(nav_html, 'html.parser')) # Insert nav at the beginning of body

            # Apply base font and prose styles to body/wrapper
            if body:
                # Add base font class safely
                body_classes = body.get('class', [])
                if 'font-["Newsreader"]' not in body_classes:
                     body['class'] = body_classes + ['font-["Newsreader"]']

                # Wrap direct children of body (except nav) in a prose div if not already structured
                nav_div = body.find("div", class_="redline-navigation", recursive=False)
                existing_prose_wrapper = body.find('div', class_='prose', recursive=False)

                # Check if direct children need wrapping (i.e., not already inside a single container like 'prose')
                # This logic might need refinement depending on expected input structures.
                # Basic check: if body has multiple direct children (excluding nav/scripts/styles) and no prose div
                direct_children_count = len([c for c in body.contents if c.name and c is not nav_div and c.name not in ['script', 'style']])

                if not existing_prose_wrapper and direct_children_count > 1:
                     logger.debug("Wrapping body content in Tailwind Prose div.")
                     content_wrapper = soup.new_tag("div")
                     content_wrapper['class'] = ["prose", "max-w-none", "prose-sm", "sm:prose-base", "lg:prose-lg", "xl:prose-xl", "2xl:prose-2xl"]
                     elements_to_wrap = [child.extract() for child in body.contents if child is not nav_div and child.name and child.name not in ['script', 'style']]
                     for element in elements_to_wrap:
                         content_wrapper.append(element)

                     # Insert the wrapper back into the body (after nav if present)
                     if nav_div:
                         nav_div.insert_after(content_wrapper)
                     else:
                         body.insert(0, content_wrapper) # Or at the beginning if no nav
                elif existing_prose_wrapper:
                    logger.debug("Prose wrapper already exists, skipping auto-wrap.")
                else:
                    logger.debug("Body content seems structured (single child or no content), skipping auto-wrap.")

            # Convert back to string
            final_html = str(soup)

    except Exception as e:
        logger.warning(f"HTML post-processing failed: {str(e)}, returning original redline HTML", exc_info=True)
        # If post-processing fails, return the original redline HTML passed in
        return redline_html # Return original unprocessed HTML

    # Final check for basic validity
    if not final_html or not isinstance(final_html, str):
        logger.error("Post-processing resulted in empty or invalid HTML. Returning original.")
        return redline_html

    return final_html


def _get_tailwind_css() -> str:
    """Returns the Tailwind CSS block for redline styling."""
    # Combined styling for different diff types
    return """
        @tailwind base;
        @tailwind components;
        @tailwind utilities;

        @layer base {
            ins.diff-insert, ins.diff-insert-text { /* Added text */
                @apply text-blue-700 bg-blue-100 no-underline px-0.5 rounded-sm;
            }
            ins.diff-insert:hover, ins.diff-insert-text:hover {
                @apply bg-blue-200;
            }
            del.diff-delete, del.diff-delete-text { /* Deleted text */
                @apply text-red-700 bg-red-100 line-through px-0.5 rounded-sm;
            }
            del.diff-delete:hover, del.diff-delete-text:hover {
                @apply bg-red-200;
            }
            ins.diff-move-target { /* Moved text - new location */
                @apply text-green-800 bg-green-100 no-underline px-0.5 rounded-sm border border-green-300;
            }
            ins.diff-move-target:hover {
                @apply bg-green-200;
            }
            del.diff-move-source { /* Moved text - old location */
                @apply text-green-800 bg-green-100 line-through px-0.5 rounded-sm border border-dotted border-green-300;
            }
            del.diff-move-source:hover {
                @apply bg-green-200;
            }
             /* Container for elements with updated text (inline ins/del show details) */
            span.diff-update-container {
                /* Optional: Add subtle marker, e.g., border */
                /* @apply border-l-2 border-purple-300 pl-1; */
            }
            /* Container for elements with attribute changes */
            span.diff-attrib-change {
                /* Optional: Add subtle marker, e.g., dotted underline */
                 @apply border-b border-dotted border-orange-400;
            }
            /* Styles for other specific diff markers if needed */
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

def _get_navigation_js() -> str:
    """Returns JavaScript for navigating between changes."""
    # Added diff-move classes to querySelectorAll
    return """<script>
// Find all changes in the document
function findAllChanges() {
    return document.querySelectorAll('ins.diff-insert, del.diff-delete, ins.diff-move-target, del.diff-move-source, span.diff-attrib-change, ins.diff-insert-text, del.diff-delete-text');
}

// Global variables for navigation
let changes = [];
let currentIndex = -1;
let currentHighlight = null; // Store the currently highlighted element

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    try {
        changes = Array.from(findAllChanges());
        if (changes.length > 0) {
             currentIndex = 0; // Start at the first change
             updateCounter();
             // Optionally highlight the first change immediately
             // navigateToChange();
        } else {
             updateCounter(); // Show 0/0
        }
    } catch (e) {
        console.error("Error initializing redline navigation:", e);
    }
});

// Navigate to previous change
function goPrevChange() {
    if (changes.length === 0) return; // No changes found

    // If currentIndex is -1 (meaning no change selected yet), or 0, wrap to the end
    if (currentIndex <= 0) {
        currentIndex = changes.length - 1;
    } else {
        currentIndex--;
    }
    navigateToChange();
}

// Navigate to next change
function goNextChange() {
    if (changes.length === 0) return; // No changes found

    // If currentIndex is -1 (no change selected yet), or at the end, wrap to the beginning
    if (currentIndex < 0 || currentIndex >= changes.length - 1) {
        currentIndex = 0;
    } else {
        currentIndex++;
    }
    navigateToChange();
}

// Scroll to current change and update counter
function navigateToChange() {
    if (currentIndex < 0 || currentIndex >= changes.length) return; // Index out of bounds

    const change = changes[currentIndex];
    if (!change) return; // Safety check

    // Remove highlight from previous element
    if (currentHighlight) {
        currentHighlight.style.outline = '';
        currentHighlight.style.boxShadow = ''; // Remove box shadow too
    }

    // Scroll to the new change
    change.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Highlight the current change with a more visible effect
    change.style.outline = '2px solid orange';
    change.style.boxShadow = '0 0 5px 2px orange'; // Add a subtle shadow
    currentHighlight = change; // Store the new highlighted element

    updateCounter();
}

// Update the change counter display
function updateCounter() {
    const counter = document.getElementById('change-counter');
    if (counter) {
        if (changes.length > 0 && currentIndex >= 0) {
            counter.textContent = `${currentIndex + 1} / ${changes.length}`;
        } else if (changes.length > 0 && currentIndex === -1) {
             counter.textContent = `0 / ${changes.length}`; // Show 0 if not started
        }
        else {
            counter.textContent = `0 / 0`; // No changes found
        }
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
    diff_level: str = "word" # Used only for output_format='text'
) -> Dict[str, Any]:
    """Creates a redline comparison between two text documents (non-HTML).

    Generates a "track changes" style redline showing differences between original and modified text.
    Converts Markdown/Text to HTML and uses the HTML redline engine for `output_format='html'`.
    Uses basic text diff for `output_format='text'`.

    Args:
        original_text: The original/old text document
        modified_text: The modified/new text document
        file_format: Format of input documents. Options: "auto", "text", "markdown", "latex".
                     Default "auto" (attempts to detect format).
        detect_moves: Whether to identify and highlight moved content (for HTML output). Default True.
        output_format: Output format. Options: "html", "text". Default "html".
        diff_level: Granularity of diff for TEXT output only. Options: "char", "word", "line". Default "word".

    Returns:
        A dictionary containing:
        {
            "redline": The redlined document in the requested format,
            "stats": { ... } // Stats depend on the method used (HTML or Text)
            "processing_time": Time in seconds to generate the redline,
            "success": True if successful
        }

    Raises:
        ToolInputError: If input parameters are invalid
        ToolError: If processing fails for other reasons
    """
    start_time = time.time()

    # --- Input Validation ---
    if not original_text or not isinstance(original_text, str):
        raise ToolInputError("Original text must be a non-empty string.")
    if not modified_text or not isinstance(modified_text, str):
        raise ToolInputError("Modified text must be a non-empty string.")
    if file_format not in ["auto", "text", "markdown", "latex"]:
         raise ToolInputError("Invalid file_format.", param_name="file_format")
    if output_format not in ["html", "text"]:
        raise ToolInputError("Invalid output_format.", param_name="output_format")
    if diff_level not in ["char", "word", "line"]:
        raise ToolInputError("Invalid diff_level.", param_name="diff_level")

    # Auto-detect format if needed
    detected_format = file_format
    if file_format == "auto":
        detected_format = _detect_file_format(original_text)
        logger.info(f"Auto-detected file format: {detected_format}")

    # Handle identical input
    if original_text == modified_text:
        logger.warning("Original and modified texts are identical. No changes to show.")
        redline = modified_text if output_format == 'text' else f"<pre>{html_stdlib.escape(modified_text)}</pre>"
        # Wrap in basic HTML structure if format is html
        if output_format == 'html':
             redline = await _postprocess_redline(redline, include_css=True, add_navigation=False, output_format='html')

        stats = {"insertions": 0, "deletions": 0, "moves": 0, "text_updates": 0, "attr_updates": 0, "other_changes": 0, "total_changes": 0}
        processing_time = time.time() - start_time
        return {"redline": redline, "stats": stats, "processing_time": processing_time, "success": True}

    redline = ""
    stats = {}

    try:
        # --- HTML Output Generation (Primary path) ---
        if output_format == "html":
            original_html_input = ""
            modified_html_input = ""
            is_markdown = False

            if detected_format == "markdown":
                try:
                    logger.info("Converting Markdown to HTML for diffing...")
                    # Use common extensions for better conversion
                    extensions = ['fenced_code', 'tables', 'sane_lists', 'nl2br', 'footnotes']
                    original_html_input = markdown.markdown(original_text, extensions=extensions)
                    modified_html_input = markdown.markdown(modified_text, extensions=extensions)
                    is_markdown = True
                    logger.info("Markdown conversion successful.")
                except Exception as md_err:
                    logger.warning(f"Markdown conversion failed: {md_err}. Falling back to text diff.", emoji_key="warning")
                    # Fall through to text handling

            # Handle Text/LaTeX or failed Markdown conversion
            if not is_markdown:
                logger.info(f"Treating input as plain text (format: {detected_format}), wrapping in <pre> for HTML diff.")
                original_html_input = f"<pre>{html_stdlib.escape(original_text)}</pre>"
                modified_html_input = f"<pre>{html_stdlib.escape(modified_text)}</pre>"

            # Now call create_html_redline with the (potentially converted) HTML
            logger.info("Calling create_html_redline for comparison...")
            result = await create_html_redline(
                original_html=original_html_input,
                modified_html=modified_html_input,
                detect_moves=detect_moves,
                # Formatting tags less relevant for converted MD/Text, use defaults
                ignore_whitespace=True, # Ignore whitespace is generally good here
                output_format="html",   # Get full HTML doc
                include_css=True,
                add_navigation=True
            )
            redline = result.get("redline_html", "")
            stats = result.get("stats", {})

        # --- Plain Text Output Generation (Secondary path) ---
        else: # output_format == "text"
            logger.info(f"Generating plain text redline (diff_level: {diff_level}).")
            # Use the existing text diff generator, ensuring it *only* returns text
            redline, stats = _generate_text_redline(
                original_text,
                modified_text,
                diff_level=diff_level,
                detect_moves=False # Move detection not supported for plain text output
            )

        # --- Final Processing ---
        processing_time = time.time() - start_time

        logger.success(
            f"Document redline generated successfully ({stats.get('total_changes', 0)} changes)",
            emoji_key="update",
            changes=stats,
            time=processing_time,
            format=detected_format,
            output=output_format
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

# --- Detect File Format (Unchanged) ---
def _detect_file_format(text: str) -> str:
    # Check for Markdown indicators
    md_patterns = [r'^#\s+', r'^-\s+', r'^\*\s+', r'^>\s', r'`{1,3}', r'\*{1,2}[^*\s]', r'!\[.+\]\(.+\)', r'\[.+\]\(.+\)']
    md_score = sum(1 for pattern in md_patterns if re.search(pattern, text, re.MULTILINE))

    # Check for LaTeX indicators
    latex_patterns = [r'\\documentclass', r'\\begin\{document\}', r'\\section\{', r'\\usepackage\{', r'\$.+\$', r'\$\$.+\$\$']
    latex_score = sum(1 for pattern in latex_patterns if re.search(pattern, text, re.MULTILINE))

    # Basic HTML check (might overlap with MD)
    html_score = sum(1 for tag in ['<html', '<body', '<div', '<table'] if tag in text)

    if latex_score >= 2: return "latex"
    if md_score >= 3 or (md_score >= 1 and html_score < 2): return "markdown" # Prefer MD if ambiguous
    # If it looks like HTML, treat as text (as it should go through create_html_redline directly)
    # This function is mainly for non-HTML -> HTML conversion path
    return "text"


def _generate_text_redline(
    original_text: str,
    modified_text: str,
    diff_level: str = "word",
    detect_moves: bool = False, # Move detection not feasible for plain text
) -> Tuple[str, Dict[str, int]]:
    """Generates a plain text redline comparison using difflib.

    Args:
        original_text: Original text content
        modified_text: Modified text content
        diff_level: Level of diff granularity ('char', 'word', 'line')
        detect_moves: Ignored for text output.

    Returns:
        Tuple of (redline_text, stats_dict)
    """
    if diff_level == "char":
        original_units = list(original_text)
        modified_units = list(modified_text)
        joiner = ""
    elif diff_level == "word":
        original_units = re.findall(r'\S+\s*', original_text) # Split preserving trailing space
        modified_units = re.findall(r'\S+\s*', modified_text)
        joiner = ""
    else: # line
        original_units = original_text.splitlines(keepends=True)
        modified_units = modified_text.splitlines(keepends=True)
        joiner = ""

    matcher = difflib.SequenceMatcher(None, original_units, modified_units, autojunk=False)

    insertions = 0
    deletions = 0
    moves = 0 # Always 0 for text output

    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result.append(joiner.join(original_units[i1:i2]))
        elif tag == 'replace':
            deleted_text = joiner.join(original_units[i1:i2])
            inserted_text = joiner.join(modified_units[j1:j2])
            if deleted_text:
                result.append(f'[-{deleted_text}-]')
                deletions += 1
            if inserted_text:
                result.append(f'{{+{inserted_text}+}}')
                insertions += 1
        elif tag == 'delete':
            deleted_text = joiner.join(original_units[i1:i2])
            if deleted_text:
                 result.append(f'[-{deleted_text}-]')
                 deletions += 1
        elif tag == 'insert':
            inserted_text = joiner.join(modified_units[j1:j2])
            if inserted_text:
                 result.append(f'{{+{inserted_text}+}}')
                 insertions += 1

    redline = ''.join(result)

    # Compile statistics
    stats = {
        "total_changes": insertions + deletions, # Only ins/del counted
        "insertions": insertions,
        "deletions": deletions,
        "moves": 0,
        "text_updates": 0, # Not applicable for text diff
        "attr_updates": 0, # Not applicable
        "other_changes": 0, # Not applicable
    }

    return redline, stats