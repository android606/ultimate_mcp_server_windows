"""
Tool composition patterns for MCP servers.

This module demonstrates how to design tools that work together effectively
in sequences and patterns, making it easier for LLMs to understand how to
compose tools for multi-step operations.
"""
from typing import Any, Dict, List, Optional

from error_handling import non_empty_string, validate_inputs, with_error_handling
from tool_annotations import QUERY_TOOL, READONLY_TOOL


class DocumentProcessingExample:
    """
    Example of tool composition for document processing.
    
    This class demonstrates a pattern where multiple tools work together
    to process a document through multiple stages:
    1. Chunking - Break large document into manageable pieces
    2. Analysis - Process each chunk individually
    3. Aggregation - Combine results into a final output
    
    This pattern is ideal for working with large documents that exceed
    context windows.
    """
    
    def __init__(self, mcp_server):
        """Initialize with an MCP server instance."""
        self.mcp = mcp_server
        self._register_tools()
        
    def _register_tools(self):
        """Register document processing tools with the MCP server."""
        
        @self.mcp.tool(
            description=(
                "Split a document into manageable chunks for processing. "
                "This is the FIRST step in processing large documents that exceed context windows. "
                "After chunking, process each chunk separately with analyze_chunk()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Chunk research paper",
                    "description": "Split a research paper into chunks",
                    "input": {"document": "This is a long research paper...", "chunk_size": 1000},
                    "output": {
                        "chunks": ["Chunk 1...", "Chunk 2..."],
                        "chunk_count": 2,
                        "chunk_ids": ["doc123_chunk_1", "doc123_chunk_2"]
                    }
                }
            ]
        )
        @with_error_handling
        @validate_inputs(document=non_empty_string)
        async def chunk_document(
            document: str,
            chunk_size: int = 1000,
            overlap: int = 100,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Split a document into manageable chunks for processing.
            
            This tool is the first step in a multi-step document processing workflow:
            1. First, chunk the document with chunk_document() (this tool)
            2. Then, process each chunk with analyze_chunk()
            3. Finally, combine results with aggregate_chunks()
            
            Args:
                document: The document text to split
                chunk_size: Maximum size of each chunk in characters
                overlap: Number of characters to overlap between chunks
                ctx: Context object passed by the MCP server
                
            Returns:
                Dictionary containing the chunks and their metadata
            """
            # Simple chunking strategy - split by character count with overlap
            chunks = []
            chunk_ids = []
            doc_id = f"doc_{hash(document) % 10000}"
            
            # Create chunks with overlap
            for i in range(0, len(document), chunk_size - overlap):
                chunk_text = document[i:i + chunk_size]
                if chunk_text:
                    chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                    chunks.append(chunk_text)
                    chunk_ids.append(chunk_id)
            
            return {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "chunk_ids": chunk_ids,
                "document_id": doc_id,
                "next_step": "analyze_chunk"  # Hint for the next tool to use
            }
        
        @self.mcp.tool(
            description=(
                "Analyze a single document chunk. "
                "This is the SECOND step in the document processing workflow. "
                "Use after chunk_document() and before aggregate_chunks()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Analyze document chunk",
                    "description": "Analyze a single chunk from a research paper",
                    "input": {"chunk": "This chunk discusses methodology...", "chunk_id": "doc123_chunk_1"},
                    "output": {
                        "analysis": {"key_topics": ["methodology", "experiment design"]},
                        "chunk_id": "doc123_chunk_1"
                    }
                }
            ]
        )
        @with_error_handling
        @validate_inputs(chunk=non_empty_string)
        async def analyze_chunk(
            chunk: str,
            chunk_id: str,
            analysis_type: str = "general",
            ctx=None
        ) -> Dict[str, Any]:
            """
            Analyze a single document chunk.
            
            This tool is the second step in a multi-step document processing workflow:
            1. First, chunk the document with chunk_document()
            2. Then, process each chunk with analyze_chunk() (this tool)
            3. Finally, combine results with aggregate_chunks()
            
            Args:
                chunk: The text chunk to analyze
                chunk_id: The ID of the chunk (from chunk_document)
                analysis_type: Type of analysis to perform
                ctx: Context object passed by the MCP server
                
            Returns:
                Dictionary containing the analysis results
            """
            # Simulate chunk analysis
            word_count = len(chunk.split())
            key_sentences = [s.strip() for s in chunk.split(".") if len(s.strip()) > 40][:3]
            
            # Different analysis types
            analysis = {
                "word_count": word_count,
                "key_sentences": key_sentences,
            }
            
            # Add analysis-specific data
            if analysis_type == "sentiment":
                # Simulate sentiment analysis
                analysis["sentiment"] = "positive" if "good" in chunk.lower() else "neutral"
                analysis["sentiment_score"] = 0.75 if "good" in chunk.lower() else 0.5
            elif analysis_type == "entities":
                # Simulate entity extraction
                analysis["entities"] = [word for word in chunk.split() if word[0].isupper()][:5]
            
            return {
                "analysis": analysis,
                "chunk_id": chunk_id,
                "next_step": "aggregate_chunks"  # Hint for the next tool to use
            }
        
        @self.mcp.tool(
            description=(
                "Aggregate analysis results from multiple document chunks. "
                "This is the FINAL step in the document processing workflow. "
                "Use after analyzing individual chunks with analyze_chunk()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Aggregate analysis results",
                    "description": "Combine analysis results from multiple chunks",
                    "input": {
                        "analysis_results": [
                            {"analysis": {"key_topics": ["methodology"]}, "chunk_id": "doc123_chunk_1"},
                            {"analysis": {"key_topics": ["results"]}, "chunk_id": "doc123_chunk_2"}
                        ]
                    },
                    "output": {
                        "document_summary": "This document covers methodology and results...",
                        "overall_statistics": {"total_chunks": 2, "word_count": 2500}
                    }
                }
            ]
        )
        @with_error_handling
        async def aggregate_chunks(
            analysis_results: List[Dict[str, Any]],
            aggregation_type: str = "summary",
            ctx=None
        ) -> Dict[str, Any]:
            """
            Aggregate analysis results from multiple document chunks.
            
            This tool is the final step in a multi-step document processing workflow:
            1. First, chunk the document with chunk_document()
            2. Then, process each chunk with analyze_chunk()
            3. Finally, combine results with aggregate_chunks() (this tool)
            
            Args:
                analysis_results: List of analysis results from analyze_chunk
                aggregation_type: Type of aggregation to perform
                ctx: Context object passed by the MCP server
                
            Returns:
                Dictionary containing the aggregated results
            """
            # Validate input
            if not analysis_results or not isinstance(analysis_results, list):
                return {
                    "error": "Invalid analysis_results. Must provide a non-empty list of analysis results."
                }
            
            # Extract all analyses
            all_analyses = [result.get("analysis", {}) for result in analysis_results if "analysis" in result]
            total_chunks = len(all_analyses)
            
            # Calculate overall statistics
            total_word_count = sum(analysis.get("word_count", 0) for analysis in all_analyses)
            all_key_sentences = [sentence for analysis in all_analyses 
                                for sentence in analysis.get("key_sentences", [])]
            
            # Generate summary based on aggregation type
            if aggregation_type == "summary":
                summary = f"Document contains {total_chunks} chunks with {total_word_count} words total."
                if all_key_sentences:
                    summary += f" Key points include: {' '.join(all_key_sentences[:3])}..."
            elif aggregation_type == "sentiment":
                # Aggregate sentiment scores if available
                sentiment_scores = [analysis.get("sentiment_score", 0.5) for analysis in all_analyses 
                                   if "sentiment_score" in analysis]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
                sentiment_label = "positive" if avg_sentiment > 0.6 else "neutral" if avg_sentiment > 0.4 else "negative"
                summary = f"Document has an overall {sentiment_label} sentiment (score: {avg_sentiment:.2f})."
            else:
                summary = f"Aggregated {total_chunks} chunks with {total_word_count} total words."
            
            return {
                "document_summary": summary,
                "overall_statistics": {
                    "total_chunks": total_chunks,
                    "word_count": total_word_count,
                    "key_sentences_count": len(all_key_sentences)
                },
                "workflow_complete": True  # Indicate this is the end of the workflow
            }


class DataPipelineExample:
    """
    Example of tool composition for data processing pipelines.
    
    This class demonstrates a pattern where tools form a processing
    pipeline to transform, filter, and analyze data:
    1. Fetch - Get data from a source
    2. Transform - Clean and process the data
    3. Filter - Select relevant data
    4. Analyze - Perform analysis on filtered data
    
    This pattern is ideal for working with structured data that
    needs multiple processing steps.
    """
    
    def __init__(self, mcp_server):
        """Initialize with an MCP server instance."""
        self.mcp = mcp_server
        self._register_tools()
        
    def _register_tools(self):
        """Register data pipeline tools with the MCP server."""
        
        @self.mcp.tool(
            description=(
                "Fetch data from a source. "
                "This is the FIRST step in the data pipeline. "
                "Continue with transform_data() to clean the fetched data."
            ),
            annotations=QUERY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Fetch CSV data",
                    "description": "Fetch data from a CSV source",
                    "input": {"source_type": "csv", "source_path": "data/sales.csv"},
                    "output": {
                        "data": [{"date": "2023-01-01", "amount": 1200}, {"date": "2023-01-02", "amount": 950}],
                        "record_count": 2,
                        "schema": {"date": "string", "amount": "number"}
                    }
                }
            ]
        )
        @with_error_handling
        async def fetch_data(
            source_type: str,
            source_path: str,
            limit: Optional[int] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Fetch data from a source.
            
            This tool is the first step in a data processing pipeline:
            1. First, fetch data with fetch_data() (this tool)
            2. Then, clean the data with transform_data()
            3. Then, filter the data with filter_data()
            4. Finally, analyze the data with analyze_data()
            
            Args:
                source_type: Type of data source (csv, json, etc.)
                source_path: Path to the data source
                limit: Maximum number of records to fetch
                ctx: Context object passed by the MCP server
                
            Returns:
                Dictionary containing the fetched data and metadata
            """
            # Simulate fetching data from different sources
            if source_type == "csv":
                # Simulate CSV data
                data = [
                    {"date": "2023-01-01", "amount": 1200, "category": "electronics"},
                    {"date": "2023-01-02", "amount": 950, "category": "clothing"},
                    {"date": "2023-01-03", "amount": 1500, "category": "electronics"},
                    {"date": "2023-01-04", "amount": 800, "category": "food"},
                ]
            elif source_type == "json":
                # Simulate JSON data
                data = [
                    {"user_id": 101, "name": "Alice", "active": True, "last_login": "2023-01-10"},
                    {"user_id": 102, "name": "Bob", "active": False, "last_login": "2022-12-15"},
                    {"user_id": 103, "name": "Charlie", "active": True, "last_login": "2023-01-05"},
                ]
            else:
                # Default dummy data
                data = [{"id": i, "value": f"Sample {i}"} for i in range(1, 6)]
            
            # Apply limit if specified
            if limit and limit > 0:
                data = data[:limit]
            
            # Infer schema from first record
            schema = {}
            if data:
                first_record = data[0]
                for key, value in first_record.items():
                    value_type = "string"
                    if isinstance(value, (int, float)):
                        value_type = "number"
                    elif isinstance(value, bool):
                        value_type = "boolean"
                    schema[key] = value_type
            
            return {
                "data": data,
                "record_count": len(data),
                "schema": schema,
                "source_info": {"type": source_type, "path": source_path},
                "next_step": "transform_data"  # Hint for the next tool to use
            }
        
        @self.mcp.tool(
            description=(
                "Transform and clean data. "
                "This is the SECOND step in the data pipeline. "
                "Use after fetch_data() and before filter_data()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Transform sales data",
                    "description": "Clean and transform sales data",
                    "input": {
                        "data": [{"date": "2023-01-01", "amount": "1,200"}, {"date": "2023-01-02", "amount": "950"}],
                        "transformations": ["convert_dates", "normalize_numbers"]
                    },
                    "output": {
                        "transformed_data": [{"date": "2023-01-01", "amount": 1200}, {"date": "2023-01-02", "amount": 950}],
                        "transformation_log": ["Converted 2 dates", "Normalized 2 numbers"]
                    }
                }
            ]
        )
        @with_error_handling
        async def transform_data(
            data: List[Dict[str, Any]],
            transformations: List[str] = None,
            custom_transformations: Dict[str, str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Transform and clean data.
            
            This tool is the second step in a data processing pipeline:
            1. First, fetch data with fetch_data()
            2. Then, clean the data with transform_data() (this tool)
            3. Then, filter the data with filter_data()
            4. Finally, analyze the data with analyze_data()
            
            Args:
                data: List of data records to transform
                transformations: List of built-in transformations to apply
                custom_transformations: Dictionary of field->transform_expression
                ctx: Context object passed by the MCP server
                
            Returns:
                Dictionary containing the transformed data and transformation log
            """
            # Validate input
            if not data or not isinstance(data, list):
                return {
                    "error": "Invalid data. Must provide a non-empty list of records."
                }
            
            # Copy data to avoid modifying the original
            transformed_data = [record.copy() for record in data]
            transformation_log = []
            
            # Apply standard transformations if specified
            transformations = transformations or []
            for transform in transformations:
                if transform == "convert_dates":
                    # Convert date strings to standard format
                    date_count = 0
                    for record in transformed_data:
                        for key, value in record.items():
                            if isinstance(value, str) and ("date" in key.lower() or "time" in key.lower()):
                                # Simple date normalization (in real code, use datetime)
                                record[key] = value.replace("/", "-")
                                date_count += 1
                    transformation_log.append(f"Converted {date_count} dates")
                
                elif transform == "normalize_numbers":
                    # Convert number strings to actual numbers
                    number_count = 0
                    for record in transformed_data:
                        for key, value in record.items():
                            if isinstance(value, str) and any(c.isdigit() for c in value):
                                # Try to convert to number if it looks like one
                                try:
                                    # Remove commas and convert
                                    clean_value = value.replace(",", "")
                                    if "." in clean_value:
                                        record[key] = float(clean_value)
                                    else:
                                        record[key] = int(clean_value)
                                    number_count += 1
                                except ValueError:
                                    # Not a number after all, keep as string
                                    pass
                    transformation_log.append(f"Normalized {number_count} numbers")
            
            return {
                "transformed_data": transformed_data,
                "transformation_log": transformation_log,
                "record_count": len(transformed_data),
                "next_step": "filter_data"  # Hint for the next tool to use
            }
        
        # Additional tools for the data pipeline would be added here... 