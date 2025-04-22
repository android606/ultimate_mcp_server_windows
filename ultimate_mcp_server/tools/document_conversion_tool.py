"""Document conversion tools for Ultimate MCP Server."""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)

# Import Docling components
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.document_conversion_tool")

@with_tool_metrics
@with_error_handling
async def convert_document(
    document_path: str,
    output_format: str = "markdown", 
    output_path: Optional[str] = None,
    save_to_file: bool = False,
    accelerator_device: str = "auto",
    num_threads: int = 4,
) -> Dict[str, Any]:
    """Converts documents (PDF, DOCX, HTML, etc.) to various formats using Docling.
    
    This tool provides document conversion capabilities powered by the Docling library,
    which can parse and process various document formats (including PDF, DOCX, HTML)
    and export them to formats like Markdown, JSON, HTML, and plain text.
    
    Use this tool when you need to extract structured content from documents while maintaining
    layout, tables, lists, headings, and other formatting elements. It's especially useful
    for converting complex PDF documents to markdown or other structured formats.
    
    Args:
        document_path: Path to a local file or a URL to a document to convert.
                      Supports: PDF, DOCX, HTML, XLSX, and more.
        output_format: Desired output format. Options:
                      "markdown" (default) - Markdown format with formatting preserved
                      "text" - Plain text without markup
                      "html" - HTML with structure preserved
                      "json" - Docling's JSON document representation
                      "doctags" - Docling's document token format
        output_path: Optional path where output should be saved if save_to_file is True.
                    If not provided, a temporary file will be created.
        save_to_file: Whether to save the output to a file. Default is False.
        accelerator_device: Device to use for acceleration. Options:
                          "auto" (default) - Will pick the best available option
                          "cpu" - CPU processing only
                          "cuda" - NVIDIA GPU acceleration
                          "mps" - Apple Silicon GPU acceleration
        num_threads: Number of threads to use for document processing. Default is 4.
    
    Returns:
        A dictionary containing the conversion result:
        {
            "success": True,
            "content": "# Document Title\\n\\nDocument content...",  # The converted content
            "output_format": "markdown",  # The format of the returned content
            "file_path": "/path/to/output.md",  # Only if save_to_file is True
            "processing_time": 1.23,  # Time taken in seconds
            "document_metadata": {  # Optional metadata if available
                "num_pages": 5,
                "has_tables": true,
                "has_figures": false, 
                "has_sections": true
            }
        }
        
        If an error occurs, the response will follow the standard error format.
    
    Raises:
        ToolInputError: If the document path is invalid, the output format is unsupported,
                       or if Docling is not properly installed.
        ToolError: If the conversion fails for any reason.
    
    Examples:
        Convert a PDF to markdown:
        >>> result = await convert_document("path/to/document.pdf")
        >>> print(result["content"])  # Markdown content
        
        Convert a web document to HTML and save to file:
        >>> result = await convert_document(
        ...     "https://example.com/document.pdf",
        ...     output_format="html",
        ...     save_to_file=True,
        ...     output_path="path/to/output.html"
        ... )
    """
    start_time = time.time()
    
    # Validate parameters
    valid_formats = ["markdown", "text", "html", "json", "doctags"]
    if output_format not in valid_formats:
        raise ToolInputError(
            f"Invalid output format: '{output_format}'. "
            f"Must be one of: {', '.join(valid_formats)}"
        )
    
    valid_accelerator_devices = ["auto", "cpu", "cuda", "mps"]
    if accelerator_device.lower() not in valid_accelerator_devices:
        raise ToolInputError(
            f"Invalid accelerator device: '{accelerator_device}'. "
            f"Must be one of: {', '.join(valid_accelerator_devices)}"
        )
    
    # Set up accelerator device
    if accelerator_device.lower() == "auto":
        device = AcceleratorDevice.AUTO
    elif accelerator_device.lower() == "cpu":
        device = AcceleratorDevice.CPU
    elif accelerator_device.lower() == "cuda":
        device = AcceleratorDevice.CUDA
    elif accelerator_device.lower() == "mps":
        device = AcceleratorDevice.MPS
    
    # Prepare pipeline options (without OCR)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Explicitly disable OCR
    pipeline_options.generate_page_images = False  # Don't generate page images
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads, 
        device=device
    )
    
    # Set up document converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Determine output path if saving to file
    output_file_path = None
    if save_to_file:
        if output_path:
            output_file_path = Path(output_path)
        else:
            # Create a temporary output path based on input document
            input_name = os.path.basename(document_path).split("?")[0]  # Handle URLs with query params
            input_stem = os.path.splitext(input_name)[0]
            extension = ".md" if output_format == "markdown" else f".{output_format}"
            output_file_path = Path(f"/tmp/{input_stem}{extension}")
        
        # Ensure the parent directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert the document
        logger.info(f"Converting document: {document_path}")
        result = doc_converter.convert(document_path)
        
        # Extract content based on requested format
        content = ""
        if output_format == "markdown":
            content = result.document.export_to_markdown()
        elif output_format == "text":
            content = result.document.export_to_text()
        elif output_format == "html":
            content = result.document.export_to_html()
        elif output_format == "json":
            content = json.dumps(result.document.export_to_dict())
        elif output_format == "doctags":
            content = result.document.export_to_document_tokens()
        
        # Save to file if requested
        if save_to_file and output_file_path:
            logger.info(f"Saving output to: {output_file_path}")
            
            if output_format == "markdown":
                result.document.save_as_markdown(
                    output_file_path,
                    image_mode=ImageRefMode.PLACEHOLDER
                )
            elif output_format == "text":
                result.document.save_as_markdown(
                    output_file_path,
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
            elif output_format == "html":
                result.document.save_as_html(
                    output_file_path,
                    image_mode=ImageRefMode.REFERENCED
                )
            elif output_format == "json":
                result.document.save_as_json(
                    output_file_path,
                    image_mode=ImageRefMode.PLACEHOLDER
                )
            elif output_format == "doctags":
                result.document.save_as_document_tokens(output_file_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare metadata
        try:
            metadata = {
                "num_pages": result.document.num_pages() if hasattr(result.document, 'num_pages') else 0,
                "has_tables": any(hasattr(p, 'content') and hasattr(p.content, 'has_tables') and p.content.has_tables() 
                                 for p in result.document.pages) if hasattr(result.document, 'pages') else False,
                "has_figures": any(hasattr(p, 'content') and hasattr(p.content, 'has_figures') and p.content.has_figures() 
                                  for p in result.document.pages) if hasattr(result.document, 'pages') else False,
                "has_sections": hasattr(result.document, 'get_sections') and len(result.document.get_sections()) > 0,
            }
        except Exception as e:
            logger.warning(f"Error extracting document metadata: {str(e)}")
            metadata = {
                "num_pages": 0,
                "has_tables": False,
                "has_figures": False,
                "has_sections": False
            }
        
        # Prepare response
        response = {
            "success": True,
            "content": content,
            "output_format": output_format,
            "processing_time": processing_time,
            "document_metadata": metadata,
        }
        
        # Add file path if saved
        if save_to_file and output_file_path:
            response["file_path"] = str(output_file_path)
        
        logger.info(
            f"Document conversion successful in {processing_time:.2f}s", 
            extra={"file": document_path, "format": output_format}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Document conversion failed: {str(e)}", exc_info=True)
        raise ToolError(
            f"Document conversion failed: {str(e)}",
            error_code="CONVERSION_FAILED",
            details={"document_path": document_path}
        ) from e