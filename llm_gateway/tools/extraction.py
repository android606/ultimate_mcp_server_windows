"""Data extraction tools for LLM Gateway."""
import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from llm_gateway.config import config
from llm_gateway.constants import Provider, TaskType
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.cache import with_cache
from llm_gateway.tools.base import BaseTool, with_retry, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class ExtractionTools(BaseTool):
    """Data extraction tools for LLM Gateway."""
    
    tool_name = "extraction"
    description = "Tools for extracting structured data from text."
    
    def __init__(self, mcp_server):
        """Initialize the extraction tools.
        
        Args:
            mcp_server: MCP server instance
        """
        super().__init__(mcp_server)
        
    def _register_tools(self):
        """Register extraction tools with MCP server."""
        
        @self.mcp.tool()
        @with_cache(ttl=7 * 24 * 60 * 60)  # Cache for 7 days
        @with_tool_metrics
        async def extract_json(
            text: str,
            schema: Optional[Dict[str, Any]] = None,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            max_attempts: int = 3,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Extract structured JSON data from text.
            
            Args:
                text: Text to extract data from
                schema: Optional JSON schema to specify the expected output structure
                provider: LLM provider to use
                model: Model name (default based on provider)
                max_attempts: Maximum attempts to get valid JSON
                
            Returns:
                Dictionary containing extracted JSON and metadata
            """
            start_time = time.time()
            
            if not text:
                return {
                    "error": "No text provided",
                    "data": None,
                    "processing_time": 0.0,
                }
                
            # Prepare schema description
            schema_description = ""
            if schema:
                # Convert schema to a human-readable format
                schema_description = "The output should follow this structure:\n"
                
                # Format schema as JSON string with indentation
                schema_json = json.dumps(schema, indent=2)
                schema_description += f"```json\n{schema_json}\n```\n\n"
                
                # Add description of required fields
                if "properties" in schema:
                    required_fields = schema.get("required", [])
                    if required_fields:
                        schema_description += "Required fields: " + ", ".join(required_fields) + "\n\n"
                        
                    # Add descriptions of each property if available
                    for prop_name, prop_info in schema["properties"].items():
                        if "description" in prop_info:
                            schema_description += f"- {prop_name}: {prop_info['description']}\n"
            
            # Prepare prompt with instructions for clean JSON extraction
            prompt = f"""Extract structured information from the following text as JSON.
{schema_description}
The response should be ONLY a valid JSON object with no explanations, comments, or other text before or after it.
If information is not found in the text, use null for the corresponding field, but never omit required fields.

Text:
{text}

JSON output:"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            # Try extraction with multiple attempts if needed
            for attempt in range(max_attempts):
                try:
                    # Generate extraction
                    result = await provider_instance.generate_completion(
                        prompt=prompt,
                        model=model,
                        temperature=0.1,  # Low temperature for consistent extraction
                        max_tokens=4000,  # Allow for large JSON responses
                    )
                    
                    # Try to parse the JSON
                    extracted_data = self._parse_json_from_llm_response(result.text)
                    
                    if extracted_data:
                        # Validate against schema if provided
                        if schema and not self._validate_json_against_schema(extracted_data, schema):
                            # If invalid, try again if attempts remain
                            if attempt < max_attempts - 1:
                                logger.warning(
                                    f"JSON validation failed, retrying ({attempt+1}/{max_attempts})",
                                    emoji_key="warning"
                                )
                                continue
                        
                        # Successful extraction
                        processing_time = time.time() - start_time
                        
                        logger.success(
                            f"JSON extraction successful",
                            emoji_key=TaskType.EXTRACTION.value,
                            tokens={
                                "input": result.input_tokens,
                                "output": result.output_tokens
                            },
                            cost=result.cost,
                            time=processing_time
                        )
                        
                        return {
                            "data": extracted_data,
                            "model": result.model,
                            "provider": provider,
                            "tokens": {
                                "input": result.input_tokens,
                                "output": result.output_tokens,
                                "total": result.total_tokens,
                            },
                            "cost": result.cost,
                            "processing_time": processing_time,
                            "attempts": attempt + 1,
                        }
                        
                    elif attempt < max_attempts - 1:
                        # If JSON parsing failed, refine the prompt and try again
                        prompt = f"""Extract structured information from the following text as JSON.
{schema_description}
The response must be ONLY a valid JSON object with NO explanations, NO comments, and NO additional text.
Ensure correct JSON formatting with double quotes around keys and string values.
Do not include any markdown formatting symbols or backticks.

Here was my previous attempt that failed to produce valid JSON:
{result.text}

Text:
{text}

JSON output (ONLY valid JSON):"""
                
                except Exception as e:
                    logger.error(
                        f"JSON extraction error: {str(e)}",
                        emoji_key="error",
                        attempt=attempt+1
                    )
                    if attempt < max_attempts - 1:
                        # Wait briefly before retrying
                        await asyncio.sleep(1)
            
            # If we get here, all attempts failed
            processing_time = time.time() - start_time
            
            logger.error(
                f"JSON extraction failed after {max_attempts} attempts",
                emoji_key="error",
                time=processing_time
            )
            
            return {
                "error": "Failed to extract valid JSON",
                "raw_text": result.text if 'result' in locals() else None,
                "data": None,
                "model": result.model if 'result' in locals() else None,
                "provider": provider,
                "processing_time": processing_time,
                "attempts": max_attempts,
            }
        
        @self.mcp.tool()
        @with_cache(ttl=7 * 24 * 60 * 60)  # Cache for 7 days
        @with_tool_metrics
        async def extract_table(
            text: str,
            headers: Optional[List[str]] = None,
            format: str = "json",
            include_raw: bool = False,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Extract tabular data from text.
            
            Args:
                text: Text containing tabular data
                headers: Optional list of expected column headers
                format: Output format ('json', 'csv', or 'markdown')
                include_raw: Whether to include raw extracted text
                provider: LLM provider to use
                model: Model name (default based on provider)
                
            Returns:
                Dictionary containing extracted table and metadata
            """
            start_time = time.time()
            
            if not text:
                return {
                    "error": "No text provided",
                    "data": None,
                    "processing_time": 0.0,
                }
                
            # Prepare column headers description
            headers_description = ""
            if headers:
                headers_description = f"The table should have these columns: {', '.join(headers)}.\n"
                
            # Prepare format instructions
            if format == "json":
                format_instructions = """
Return the table as a JSON array of objects, where each object represents a row with column names as keys.
Example:
[
  {"Column1": "Value1", "Column2": "Value2", ...},
  {"Column1": "Value3", "Column2": "Value4", ...},
  ...
]
"""
            elif format == "csv":
                format_instructions = """
Return the table as CSV (comma-separated values) with headers in the first row.
Example:
Column1,Column2,...
Value1,Value2,...
Value3,Value4,...
"""
            elif format == "markdown":
                format_instructions = """
Return the table in Markdown format with headers.
Example:
| Column1 | Column2 | ... |
|---------|---------|-----|
| Value1  | Value2  | ... |
| Value3  | Value4  | ... |
"""
            else:
                format = "json"  # Default to JSON
                format_instructions = """
Return the table as a JSON array of objects, where each object represents a row with column names as keys.
"""
            
            # Prepare prompt
            prompt = f"""Extract the tabular data from the following text.
{headers_description}
{format_instructions}
The output should only contain the extracted table in {format.upper()} format with no additional text or explanations.

Text:
{text}

Extracted table ({format.upper()} format):"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            try:
                # Generate extraction
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=4000,  # Allow for potentially large tables
                )
                
                extracted_text = result.text.strip()
                processed_data = None
                
                # Process the extracted table according to format
                if format == "json":
                    processed_data = self._parse_json_from_llm_response(extracted_text)
                elif format == "csv":
                    processed_data = self._parse_csv_from_llm_response(extracted_text)
                elif format == "markdown":
                    processed_data = self._parse_markdown_table_from_llm_response(extracted_text)
                    
                # If parsing failed, try to extract tables directly from the text
                if not processed_data:
                    # Fallback to regex-based table extraction
                    processed_data = self._extract_tables_with_regex(text, format)
                
                processing_time = time.time() - start_time
                
                if processed_data:
                    logger.success(
                        f"Table extraction successful ({format} format, {len(processed_data)} rows)",
                        emoji_key=TaskType.EXTRACTION.value,
                        tokens={
                            "input": result.input_tokens,
                            "output": result.output_tokens
                        },
                        cost=result.cost,
                        time=processing_time
                    )
                    
                    response = {
                        "data": processed_data,
                        "format": format,
                        "model": result.model,
                        "provider": provider,
                        "tokens": {
                            "input": result.input_tokens,
                            "output": result.output_tokens,
                            "total": result.total_tokens,
                        },
                        "cost": result.cost,
                        "processing_time": processing_time,
                    }
                    
                    # Include raw extracted text if requested
                    if include_raw:
                        response["raw_text"] = extracted_text
                        
                    return response
                else:
                    logger.error(
                        f"Table extraction failed",
                        emoji_key="error",
                        time=processing_time
                    )
                    
                    return {
                        "error": f"Failed to extract table in {format} format",
                        "data": None,
                        "raw_text": extracted_text if include_raw else None,
                        "model": result.model,
                        "provider": provider,
                        "processing_time": processing_time,
                    }
                    
            except Exception as e:
                processing_time = time.time() - start_time
                
                logger.error(
                    f"Table extraction error: {str(e)}",
                    emoji_key="error",
                    time=processing_time
                )
                
                return {
                    "error": f"Table extraction error: {str(e)}",
                    "data": None,
                    "model": model,
                    "provider": provider,
                    "processing_time": processing_time,
                }
        
        @self.mcp.tool()
        @with_cache(ttl=30 * 24 * 60 * 60)  # Cache for 30 days
        @with_tool_metrics
        async def extract_key_value_pairs(
            text: str,
            expected_keys: Optional[List[str]] = None,
            normalize_keys: bool = True,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Extract key-value pairs from text.
            
            Args:
                text: Text to extract key-value pairs from
                expected_keys: Optional list of expected keys
                normalize_keys: Whether to normalize keys (lowercase, remove spaces)
                provider: LLM provider to use
                model: Model name (default based on provider)
                
            Returns:
                Dictionary containing extracted key-value pairs and metadata
            """
            start_time = time.time()
            
            if not text:
                return {
                    "error": "No text provided",
                    "data": {},
                    "processing_time": 0.0,
                }
                
            # Prepare expected keys description
            keys_description = ""
            if expected_keys:
                keys_description = f"Look for these specific keys: {', '.join(expected_keys)}.\n"
                
            # Prepare prompt
            prompt = f"""Extract all key-value pairs from the following text.
{keys_description}
Return the results as a JSON object where each key-value pair from the text becomes a property in the JSON.
If a key is mentioned without a clear value, use null as the value.
The output should be ONLY a valid JSON object with no explanations or additional text.

Text:
{text}

Key-value pairs (JSON format):"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            try:
                # Generate extraction
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=2000,  # Allow for potentially large number of key-value pairs
                )
                
                # Try to parse the JSON
                extracted_data = self._parse_json_from_llm_response(result.text)
                
                # Fallback to regex extraction if JSON parsing fails
                if not extracted_data:
                    # Extract key-value pairs using regex patterns
                    extracted_data = self._extract_key_value_pairs_with_regex(text, expected_keys)
                
                # Normalize keys if requested
                if normalize_keys and extracted_data:
                    normalized_data = {}
                    for key, value in extracted_data.items():
                        # Normalize key: lowercase, replace spaces with underscores
                        normalized_key = key.lower().replace(' ', '_').strip()
                        normalized_data[normalized_key] = value
                    extracted_data = normalized_data
                
                processing_time = time.time() - start_time
                
                if extracted_data:
                    logger.success(
                        f"Key-value extraction successful ({len(extracted_data)} pairs)",
                        emoji_key=TaskType.EXTRACTION.value,
                        tokens={
                            "input": result.input_tokens,
                            "output": result.output_tokens
                        },
                        cost=result.cost,
                        time=processing_time
                    )
                    
                    return {
                        "data": extracted_data,
                        "model": result.model,
                        "provider": provider,
                        "tokens": {
                            "input": result.input_tokens,
                            "output": result.output_tokens,
                            "total": result.total_tokens,
                        },
                        "cost": result.cost,
                        "processing_time": processing_time,
                    }
                else:
                    logger.error(
                        f"Key-value extraction failed",
                        emoji_key="error",
                        time=processing_time
                    )
                    
                    return {
                        "error": "Failed to extract key-value pairs",
                        "data": {},
                        "model": result.model,
                        "provider": provider,
                        "processing_time": processing_time,
                    }
                    
            except Exception as e:
                processing_time = time.time() - start_time
                
                logger.error(
                    f"Key-value extraction error: {str(e)}",
                    emoji_key="error",
                    time=processing_time
                )
                
                return {
                    "error": f"Key-value extraction error: {str(e)}",
                    "data": {},
                    "model": model,
                    "provider": provider,
                    "processing_time": processing_time,
                }

        @self.mcp.tool()
        @with_cache(ttl=14 * 24 * 60 * 60)  # Cache for 14 days
        @with_tool_metrics
        @with_retry(max_retries=2, retry_delay=1.0)
        async def extract_semantic_schema(
            text: str,
            target_format: str = "json_schema",
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Analyze text and extract its semantic structure as a schema.
            
            Args:
                text: Text to analyze
                target_format: Schema format ('json_schema', 'typescript', or 'python_dataclass')
                provider: LLM provider to use
                model: Model name (default based on provider)
                
            Returns:
                Dictionary containing extracted schema and metadata
            """
            start_time = time.time()
            
            if not text:
                return {
                    "error": "No text provided",
                    "schema": None,
                    "processing_time": 0.0,
                }
                
            # Validate target format
            valid_formats = ["json_schema", "typescript", "python_dataclass"]
            if target_format not in valid_formats:
                target_format = "json_schema"
                
            # Prepare format instructions
            if target_format == "json_schema":
                format_instructions = """
Create a JSON Schema (draft-07) that represents the data structure in the text.
Include types, required fields, and descriptions.
Example:
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Person's full name"
    },
    "age": {
      "type": "integer",
      "description": "Person's age in years"
    }
  },
  "required": ["name"]
}
"""
            elif target_format == "typescript":
                format_instructions = """
Create TypeScript interfaces that represent the data structure in the text.
Include proper types, optional fields, and JSDoc comments.
Example:
/**
 * Represents a person
 */
interface Person {
  /** Person's full name */
  name: string;
  /** Person's age in years */
  age?: number;
}
"""
            elif target_format == "python_dataclass":
                format_instructions = """
Create Python dataclasses that represent the data structure in the text.
Include proper types, default values, and docstrings.
Example:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Person:
    \"\"\"Represents a person.\"\"\"
    name: str  # Person's full name
    age: Optional[int] = None  # Person's age in years
```
"""
            
            # Prepare prompt
            prompt = f"""Analyze the following text and extract its semantic structure as a schema.
Identify entities, their properties, relationships, and data types.
{format_instructions}
The output should only contain the schema with no additional text or explanations.

Text:
{text}

Schema ({target_format}):"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            try:
                # Generate schema
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    temperature=0.2,  # Low temperature for consistent schema
                    max_tokens=3000,  # Allow for complex schemas
                )
                
                # Clean the response based on format
                schema = result.text.strip()
                
                # For JSON Schema, ensure it's valid JSON
                if target_format == "json_schema":
                    try:
                        # Try to parse as JSON to validate
                        json_schema = json.loads(schema)
                        schema = json.dumps(json_schema, indent=2)
                    except json.JSONDecodeError:
                        # If not valid JSON, extract JSON part using regex
                        schema = self._parse_json_from_llm_response(schema)
                        if not schema:
                            raise ValueError("Could not extract valid JSON schema")
                
                # For Python, clean up code blocks
                if target_format == "python_dataclass":
                    schema = re.sub(r'```python\s*|\s*```', '', schema)
                    
                # For TypeScript, clean up code blocks
                if target_format == "typescript":
                    schema = re.sub(r'```typescript\s*|\s*```', '', schema)
                
                processing_time = time.time() - start_time
                
                logger.success(
                    f"Schema extraction successful ({target_format})",
                    emoji_key=TaskType.EXTRACTION.value,
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                return {
                    "schema": schema,
                    "format": target_format,
                    "model": result.model,
                    "provider": provider,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                logger.error(
                    f"Schema extraction error: {str(e)}",
                    emoji_key="error",
                    time=processing_time
                )
                
                return {
                    "error": f"Schema extraction error: {str(e)}",
                    "schema": None,
                    "format": target_format,
                    "model": model,
                    "provider": provider,
                    "processing_time": processing_time,
                }
    
    def _parse_json_from_llm_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response text.
        
        This handles common issues with LLM-generated JSON:
        1. JSON embedded in markdown code blocks
        2. Extra text before or after the JSON
        3. Single quotes instead of double quotes
        4. Trailing commas in arrays/objects
        
        Args:
            text: LLM response text
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1).strip()
        
        # Try to find JSON object/array patterns
        json_object_match = re.search(r'(\{[\s\S]*\})', text)
        json_array_match = re.search(r'(\[[\s\S]*\])', text)
        
        if json_object_match:
            text = json_object_match.group(1)
        elif json_array_match:
            text = json_array_match.group(1)
        
        # Try parsing as is
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try fixing common JSON issues
        try:
            # Replace single quotes with double quotes (but not in already quoted strings)
            # This is a simplified fix and might not work for all cases
            text = re.sub(r"(?<![\"\'])\'(?![\"\'])", "\"", text)
            text = re.sub(r"(?<![\"\'])\'(?![\"\'])", "\"", text)
            
            # Remove trailing commas
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)
            
            # Try parsing again
            return json.loads(text)
        except json.JSONDecodeError:
            # If all attempts fail, return None
            return None
    
    def _parse_csv_from_llm_response(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse CSV from LLM response text.
        
        Args:
            text: LLM response text
            
        Returns:
            List of dictionaries (rows) or None if parsing fails
        """
        try:
            import csv
            from io import StringIO
            
            # Parse CSV
            csv_file = StringIO(text)
            reader = csv.DictReader(csv_file)
            
            # Convert to list of dictionaries
            rows = list(reader)
            
            # Clean up empty rows
            rows = [row for row in rows if any(v.strip() for v in row.values())]
            
            return rows
        except Exception:
            return None
    
    def _parse_markdown_table_from_llm_response(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse Markdown table from LLM response text.
        
        Args:
            text: LLM response text
            
        Returns:
            List of dictionaries (rows) or None if parsing fails
        """
        try:
            # Extract table content
            lines = text.strip().split('\n')
            
            # Need at least header, separator, and one data row
            if len(lines) < 3:
                return None
                
            # Extract headers
            header_match = re.match(r'\s*\|(.*)\|\s*', lines[0])
            if not header_match:
                return None
                
            header_cells = [h.strip() for h in header_match.group(1).split('|')]
            
            # Check separator line
            separator_match = re.match(r'\s*\|([-:\s\|]+)\|\s*', lines[1])
            if not separator_match:
                return None
                
            # Parse data rows
            rows = []
            for i in range(2, len(lines)):
                line = lines[i].strip()
                if not line or not line.startswith('|'):
                    continue
                    
                # Extract cells
                cells_match = re.match(r'\s*\|(.*)\|\s*', line)
                if cells_match:
                    cells = [c.strip() for c in cells_match.group(1).split('|')]
                    
                    # Create row dict
                    row = {}
                    for j, header in enumerate(header_cells):
                        if j < len(cells):
                            row[header] = cells[j]
                        else:
                            row[header] = ""
                            
                    rows.append(row)
            
            return rows
        except Exception:
            return None
    
    def _extract_tables_with_regex(self, text: str, format: str) -> Optional[List[Dict[str, Any]]]:
        """Extract tables using regex patterns.
        
        Args:
            text: Source text
            format: Desired output format
            
        Returns:
            Extracted table data or None if extraction fails
        """
        # Try to detect table structures in text
        # This is a simplified approach and may not work for all table formats
        
        # Look for Markdown tables
        md_table_pattern = r'\|(.+)\|\s*\n\|[-:\s\|]+\|\s*\n(\|.+\|\s*\n)+'
        md_tables = re.findall(md_table_pattern, text)
        
        if md_tables:
            # Extract first matched table
            table_text = md_tables[0]
            return self._parse_markdown_table_from_llm_response(table_text)
            
        # Look for CSV-like structures
        csv_pattern = r'([^,\n]+),([^,\n]+)(,[^,\n]+)*\n(([^,\n]+),([^,\n]+)(,[^,\n]+)*\n)+'
        csv_tables = re.findall(csv_pattern, text)
        
        if csv_tables:
            # Extract first matched table
            table_text = csv_tables[0]
            return self._parse_csv_from_llm_response(table_text)
            
        # If no tables found with regex, return None
        return None
    
    def _extract_key_value_pairs_with_regex(self, text: str, expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract key-value pairs using regex patterns.
        
        Args:
            text: Source text
            expected_keys: Optional list of expected keys
            
        Returns:
            Dictionary of extracted key-value pairs
        """
        # Define common key-value patterns
        patterns = [
            # Pattern 1: Key: Value
            r'([A-Za-z0-9_\s\-]+):\s*([^:\n]+)',
            
            # Pattern 2: Key = Value
            r'([A-Za-z0-9_\s\-]+)\s*=\s*([^=\n]+)',
            
            # Pattern 3: Key - Value
            r'([A-Za-z0-9_\s\-]+)\s*-\s*([^-\n]+)',
        ]
        
        # Extract using patterns
        extracted_pairs = {}
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                
                # Skip if we have expected keys and this isn't one of them
                if expected_keys and key not in expected_keys:
                    continue
                    
                # Add to results (overwriting if key already exists)
                extracted_pairs[key] = value
                
        # If we have expected keys, ensure all are present (with None for missing)
        if expected_keys:
            for key in expected_keys:
                if key not in extracted_pairs:
                    extracted_pairs[key] = None
                    
        return extracted_pairs
    
    def _validate_json_against_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate JSON data against a schema.
        
        Args:
            data: JSON data to validate
            schema: JSON schema
            
        Returns:
            True if validation succeeds, False otherwise
        """
        try:
            import jsonschema
            jsonschema.validate(instance=data, schema=schema)
            return True
        except ImportError:
            # jsonschema not available, try basic validation
            if "type" in schema:
                # Check basic type
                if schema["type"] == "object" and not isinstance(data, dict):
                    return False
                if schema["type"] == "array" and not isinstance(data, list):
                    return False
                    
            # Check required fields
            if "required" in schema and isinstance(data, dict):
                for field in schema["required"]:
                    if field not in data:
                        return False
                        
            return True
        except jsonschema.exceptions.ValidationError:
            return False
        except Exception:
            return False