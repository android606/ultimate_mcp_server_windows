"""Advanced extraction tools for LLM Gateway.

This module provides tools for structured extraction of data from text using LLMs.
These tools were previously defined in example scripts but are now part of the core library.
"""

import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
import jsonschema

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.core.models.requests import CompletionRequest
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.extraction")

class ExtractionTools:
    """Tools for structured extraction of data from text using LLMs."""
    
    def __init__(self, gateway=None):
        """Initialize extraction tools.
        
        Args:
            gateway: Optional gateway instance to register tools with
        """
        self.gateway = gateway
        # Register tools if gateway is provided
        if gateway:
            self.register_tools()
            
    def register_tools(self):
        """Register extraction tools with the gateway."""
        if not self.gateway:
            logger.warning("No gateway provided. Tools will not be registered.", emoji_key="warning")
            return
            
        # Register the extraction tools
        self.gateway.mcp.tool()(self.extract_json)
        self.gateway.mcp.tool()(self.extract_table)
        self.gateway.mcp.tool()(self.extract_key_value_pairs)
        self.gateway.mcp.tool()(self.extract_semantic_schema)
        
        logger.info("Extraction tools registered with gateway", emoji_key="info")
        
    async def extract_json(
        self,
        text: str,
        json_schema: Dict = None,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        validate_output: bool = True,
        ctx=None
    ) -> Dict[str, Any]:
        """Extract structured JSON data from text.
        
        Args:
            text: Text to extract from
            json_schema: Optional JSON schema to structure the output
            provider: LLM provider to use
            model: Model to use (if None, uses provider default)
            validate_output: Whether to validate output against schema
            ctx: Optional context passed from MCP server
            
        Returns:
            Dictionary containing extraction results
        """
        start_time = time.time()
        
        if not text:
            return {
                "error": "No text provided",
                "data": None,
                "processing_time": 0.0,
            }
            
        # Get the provider instance
        provider_instance = get_provider(provider)
        await provider_instance.initialize()
        
        # Prepare schema description
        schema_description = ""
        if json_schema:
            schema_description = f"Use this JSON schema to structure your response:\n{json.dumps(json_schema, indent=2)}\n"
        
        # Create prompt for extraction
        prompt = f"""Extract structured JSON data from the following text.
{schema_description}
Text to extract from:
```
{text}
```

Return ONLY valid JSON that represents the extracted data. Do not include any explanations.
"""
        
        # Extract the JSON
        result = await provider_instance.generate_completion(
            prompt=prompt,
            model=model,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=4000,  # Allow for large JSON responses
            response_format={"type": "json_object"}
        )
        
        # Get processing time
        processing_time = time.time() - start_time
        
        # Parse the JSON response
        try:
            extracted_data = json.loads(result.text)
            validation_result = {"valid": True, "errors": []}
            
            # Validate against schema if provided and validation is enabled
            if json_schema and validate_output:
                try:
                    jsonschema.validate(instance=extracted_data, schema=json_schema)
                except jsonschema.exceptions.ValidationError as e:
                    validation_result = {
                        "valid": False,
                        "errors": [str(e)]
                    }
            
            # Return successful result
            return {
                "data": extracted_data,
                "validation_result": validation_result,
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
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_text": result.text,
                "data": None,
                "model": result.model,
                "provider": provider,
                "processing_time": processing_time,
            }
    
    async def extract_table(
        self,
        text: str,
        headers: List[str] = None,
        formats: List[str] = None,
        extract_metadata: bool = False,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        ctx=None
    ) -> Dict[str, Any]:
        """Extract tabular data from text.
        
        Args:
            text: Text containing table to extract
            headers: Optional list of expected headers
            formats: Output formats to return (json, markdown, etc.)
            extract_metadata: Whether to extract table metadata
            provider: LLM provider to use
            model: Model to use (if None, uses provider default)
            ctx: Optional context passed from MCP server
            
        Returns:
            Dictionary containing table data and metadata
        """
        if formats is None:
            formats = ["json"]
        start_time = time.time()
        
        if not text:
            return {
                "error": "No text provided",
                "tables": [],
                "processing_time": 0.0,
            }
            
        # Get the provider instance
        provider_instance = get_provider(provider)
        await provider_instance.initialize()
        
        # Create headers guidance if provided
        headers_guidance = ""
        if headers:
            headers_guidance = f"The table should have these column headers: {', '.join(headers)}.\n"
        
        # Create prompt for extraction
        prompt = f"""Extract tables from the following text.

Text containing table:
```
{text}
```

{headers_guidance}
Return the table in JSON format where each row is an object and keys are column headers.
{f'Also include table metadata like title, notes, and context.' if extract_metadata else ''}

Format your response as a JSON object with this structure:
{{
  "title": "The table title if present",
  "json": [ 
    {{ "column1": "value", "column2": "value", ... }},
    ...
  ],
  "markdown": "The table in markdown format",
  "metadata": {{
    "notes": ["note1", "note2", ...],
    "source": "Source information if mentioned"
  }}
}}

Return ONLY valid JSON. Do not include any explanations.
"""
        
        # Extract the table
        result = await provider_instance.generate_completion(
            prompt=prompt,
            model=model,
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        # Get processing time
        processing_time = time.time() - start_time
        
        # Parse the JSON response
        try:
            extraction_result = json.loads(result.text)
            
            # Return successful result
            return {
                "data": extraction_result,
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
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse table JSON: {str(e)}",
                "raw_text": result.text,
                "data": None,
                "model": result.model,
                "provider": provider,
                "processing_time": processing_time,
            }
            
    async def extract_key_value_pairs(
        self,
        text: str,
        keys: List[str] = None,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        ctx=None
    ) -> Dict[str, Any]:
        """Extract key-value pairs from text.
        
        Args:
            text: Text to extract from
            keys: Optional list of keys to extract
            provider: LLM provider to use
            model: Model to use (if None, uses provider default)
            ctx: Optional context passed from MCP server
            
        Returns:
            Dictionary containing extracted key-value pairs
        """
        start_time = time.time()
        
        if not text:
            return {
                "error": "No text provided",
                "data": None,
                "processing_time": 0.0,
            }
            
        # Get the provider instance
        provider_instance = get_provider(provider)
        await provider_instance.initialize()
        
        # Create keys guidance if provided
        keys_guidance = ""
        if keys:
            keys_guidance = f"Specifically extract these keys: {', '.join(keys)}.\n"
        
        # Create prompt for extraction
        prompt = f"""Extract key-value pairs from the following text.

Text to extract from:
```
{text}
```

{keys_guidance}
Return the extraction as a JSON object with this structure:
{{
  "key_value_pairs": [
    {{ "key": "key1", "value": "value1" }},
    {{ "key": "key2", "value": "value2" }},
    ...
  ]
}}

Return ONLY valid JSON. Do not include any explanations.
"""
        
        # Extract the key-value pairs
        result = await provider_instance.generate_completion(
            prompt=prompt,
            model=model,
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        # Get processing time
        processing_time = time.time() - start_time
        
        # Parse the JSON response
        try:
            extraction_result = json.loads(result.text)
            
            # Return successful result
            return {
                "data": extraction_result,
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
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse key-value JSON: {str(e)}",
                "raw_text": result.text,
                "data": None,
                "model": result.model,
                "provider": provider,
                "processing_time": processing_time,
            }
            
    async def extract_semantic_schema(
        self,
        text: str,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        ctx=None
    ) -> Dict[str, Any]:
        """Extract a semantic schema from text.
        
        This attempts to identify entities, relationships, and attributes
        in the provided text and create a structured schema.
        
        Args:
            text: Text to extract from
            provider: LLM provider to use
            model: Model to use (if None, uses provider default)
            ctx: Optional context passed from MCP server
            
        Returns:
            Dictionary containing extracted schema
        """
        start_time = time.time()
        
        if not text:
            return {
                "error": "No text provided",
                "data": None,
                "processing_time": 0.0,
            }
            
        # Get the provider instance
        provider_instance = get_provider(provider)
        await provider_instance.initialize()
        
        # Create prompt for semantic schema extraction
        prompt = f"""Extract a semantic schema from the following text.
Identify entities, their attributes, and relationships between entities.

Text to analyze:
```
{text}
```

Return the extraction as a JSON object with this structure:
{{
  "entities": [
    {{
      "name": "EntityName",
      "attributes": [
        {{ "name": "AttributeName", "type": "AttributeType", "description": "Short description" }}
      ]
    }}
  ],
  "relationships": [
    {{
      "source": "SourceEntityName",
      "target": "TargetEntityName",
      "type": "RelationshipType",
      "cardinality": "OneToMany" 
    }}
  ]
}}

Valid relationship types include: "HasA", "IsA", "BelongsTo", "Contains", "Produces", etc.
Valid cardinalities include: "OneToOne", "OneToMany", "ManyToOne", "ManyToMany".

Return ONLY valid JSON. Do not include any explanations.
"""
        
        # Extract the semantic schema
        result = await provider_instance.generate_completion(
            prompt=prompt,
            model=model,
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        # Get processing time
        processing_time = time.time() - start_time
        
        # Parse the JSON response
        try:
            extraction_result = json.loads(result.text)
            
            # Return successful result
            return {
                "data": extraction_result,
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
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse semantic schema JSON: {str(e)}",
                "raw_text": result.text,
                "data": None,
                "model": result.model,
                "provider": provider,
                "processing_time": processing_time,
            }

async def extract_code_from_response(response_text: str, model: str = "openai:gpt-4o-mini", timeout: int = 15) -> str:
    """Extract code from response text using an LLM.
    
    This function takes a raw response text that may contain code mixed with explanations
    and extracts only the executable code part.
    
    Args:
        response_text: The raw response text from the model
        model: The model to use for extraction (default: openai:gpt-4o-mini)
        timeout: Timeout in seconds for the extraction request (default: 15)
        
    Returns:
        Extracted code or empty string if no code found
    """
    if not response_text:
        return ""
        
    extraction_prompt = f"""
Extract the complete, executable Python code from the following text. 
Return ONLY the code, with no additional text, explanations, or markdown formatting.
If there are multiple code snippets, combine them into a single coherent program.
If there is no valid Python code, return an empty string.

Text to extract from:
```
{response_text}
```

Python code (no markdown, no explanations, just the complete code):
"""
    
    try:
        # Get the provider
        provider_id = model.split(':')[0]
        provider = get_provider(provider_id)
        
        if not provider:
            logger.warning(f"Provider {provider_id} not available for code extraction", emoji_key="warning")
            return ""
        
        # Generate completion for extraction
        request = CompletionRequest(prompt=extraction_prompt, model=model)
        
        # Set a timeout for the completion request
        completion_task = provider.generate_completion(
            prompt=request.prompt,
            model=request.model
        )
        
        # Use timeout for extraction
        completion_result = await asyncio.wait_for(completion_task, timeout=timeout)
        
        extracted_code = completion_result.text.strip()
        
        # If the result starts with ```python or ```, strip it
        if extracted_code.startswith("```python"):
            extracted_code = extracted_code[len("```python"):].strip()
        elif extracted_code.startswith("```"):
            extracted_code = extracted_code[len("```"):].strip()
            
        # If the result ends with ```, strip it
        if extracted_code.endswith("```"):
            extracted_code = extracted_code[:-3].strip()
        
        return extracted_code
        
    except Exception as e:
        logger.warning(f"Error extracting code using LLM: {str(e)}", emoji_key="warning")
        return ""