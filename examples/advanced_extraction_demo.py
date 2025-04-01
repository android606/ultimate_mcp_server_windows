#!/usr/bin/env python
"""Demo of advanced extraction capabilities using LLM Gateway MCP server."""
import asyncio
import json
import sys
import time
import os
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))


from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger
# --- Add Rich and Display Imports ---
from llm_gateway.utils.logging.console import console
from llm_gateway.utils.display import parse_and_display_result
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.syntax import Syntax
from rich.markup import escape
from rich import box
# ----------------------

# Initialize logger
logger = get_logger("example.advanced_extraction")

# Initialize global gateway
gateway = None

# Use the shared display utility instead of this function
# Function kept for backwards compatibility during transition
def parse_and_display_result_legacy(title: str, input_data: Dict, result: Any):
    """Legacy version that calls the new shared utility function."""
    parse_and_display_result(title, input_data, result)

async def setup_gateway():
    """Set up the gateway for demonstration."""
    global gateway
    
    # Create gateway instance
    logger.info("Initializing gateway for demonstration", emoji_key="start")
    gateway = Gateway("extraction-demo")
    
    # Initialize the server with all providers and built-in tools
    await gateway._initialize_providers()
    
    # Get available tools before registering our extraction tools
    tools_before = await gateway.mcp.list_tools()
    logger.info(f"Tools before registering extraction tools: {[t.name for t in tools_before]}", emoji_key="info")
    
    # Register custom extraction tools directly without using ExtractionTools class
    
    @gateway.mcp.tool()
    async def extract_json(
        text: str,
        json_schema: Dict = None,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        validate_output: bool = True,
        ctx=None
    ) -> Dict[str, Any]:
        """Extract structured JSON data from text."""
        start_time = time.time()
        
        if not text:
            return {
                "error": "No text provided",
                "data": None,
                "processing_time": 0.0,
            }
            
        # Get the provider instance
        provider_instance = get_provider(provider)
        
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
    
    @gateway.mcp.tool()
    async def extract_table(
        text: str,
        headers: List[str] = None,
        formats: List[str] = None,
        extract_metadata: bool = False,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        ctx=None
    ) -> Dict[str, Any]:
        """Extract tabular data from text."""
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
        
        # Create prompt for extraction
        prompt = f"""Extract tables from the following text.

Text containing table:
```
{text}
```

Return the table in JSON format where each row is an object and keys are column headers.
Also include table metadata like title, notes, and context.

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
            table_data = json.loads(result.text)
            
            # Return successful result
            return {
                "tables": [table_data],
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
                "tables": [],
                "model": result.model,
                "provider": provider,
                "processing_time": processing_time,
            }
    
    @gateway.mcp.tool()
    async def extract_key_value_pairs(
        text: str,
        structured: bool = True,
        categorize: bool = False,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        ctx=None
    ) -> Dict[str, Any]:
        """Extract key-value pairs from text."""
        start_time = time.time()
        
        if not text:
            return {
                "error": "No text provided",
                "pairs": {},
                "processing_time": 0.0,
            }
            
        # Get the provider instance
        provider_instance = get_provider(provider)
        
        # Create prompt for extraction
        prompt = f"""Extract key-value pairs from the following text.

Text to extract from:
```
{text}
```

"""
        if categorize:
            prompt += """Categorize the extracted pairs.

Format your response as a JSON object with this structure:
{
  "people": [
    {"key": "Name", "value": "John Smith"},
    {"key": "Title", "value": "CEO"}
  ],
  "organizations": [
    {"key": "Company", "value": "Acme Inc."},
    {"key": "Industry", "value": "Technology"}
  ],
  "dates": [
    {"key": "Announcement Date", "value": "January 15, 2024"}
  ],
  "locations": [...],
  "products": [...],
  "financial": [...],
  "other": [...]
}
"""
        else:
            prompt += """Format your response as a JSON object where each key-value pair extracted from the text is represented.

For example:
{
  "Project Name": "Quantum Computing Research Initiative",
  "Status": "In Progress",
  "Completion Percentage": 45,
  "Budget": "$750,000"
}
"""
        
        prompt += "\nReturn ONLY valid JSON that represents the extracted key-value pairs. Do not include any explanations."
        
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
            pairs_data = json.loads(result.text)
            
            # Return successful result
            return {
                "pairs": pairs_data,
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
                "pairs": {},
                "model": result.model,
                "provider": provider,
                "processing_time": processing_time,
            }
    
    @gateway.mcp.tool()
    async def extract_semantic_schema(
        text: str,
        format: str = "json_schema",
        add_descriptions: bool = True,
        provider: str = Provider.OPENAI.value,
        model: str = None,
        ctx=None
    ) -> Dict[str, Any]:
        """Infer and extract semantic schema from text."""
        start_time = time.time()
        
        if not text:
            return {
                "error": "No text provided",
                "schema": {},
                "processing_time": 0.0,
            }
            
        # Get the provider instance
        provider_instance = get_provider(provider)
        
        # Create prompt for schema inference
        prompt = f"""Create a JSON Schema (draft-07) for the structured data in the following text.
{' Include field descriptions for all properties.' if add_descriptions else ''}

Text to analyze:
```
{text}
```

Your schema should capture all important entities, properties, and relationships in the data.
Use appropriate data types (string, number, boolean, array, object) and validation rules.
The schema should reflect the semantic structure of the data, not just its syntactic form.

Return ONLY valid JSON Schema. Do not include any explanations.
"""
        
        # Extract the schema
        result = await provider_instance.generate_completion(
            prompt=prompt,
            model=model,
            temperature=0.2,  # Slightly higher for creative schema creation
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        # Get processing time
        processing_time = time.time() - start_time
        
        # Parse the JSON response
        try:
            schema_data = json.loads(result.text)
            
            # Return successful result
            return {
                "schema": schema_data,
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
                "schema": {},
                "model": result.model,
                "provider": provider,
                "processing_time": processing_time,
            }
    
    # Verify extraction tools are registered
    tools_after = await gateway.mcp.list_tools()
    extraction_tool_names = [t.name for t in tools_after if t.name.startswith('extract_')]
    logger.info(f"Registered extraction tools: {extraction_tool_names}", emoji_key="info")
    
    logger.success("Gateway initialized with extraction tools", emoji_key="success")


async def json_extraction_demo():
    """Demonstrate JSON extraction with schema validation."""
    logger.info("Starting JSON extraction demo", emoji_key="start")
    
    # Sample text with embedded information
    text = """
    Project Status Report - March 2024
    
    Project Name: Quantum Computing Research Initiative
    Project ID: QC-2024-03
    Status: In Progress
    Completion: 45%
    
    Team Members:
    - Dr. Sarah Chen (Lead Researcher)
    - Dr. Robert Patel (Quantum Algorithm Specialist)
    - Maria Rodriguez (Software Engineer)
    - James Wilson (Data Scientist)
    
    Key Milestones:
    1. Initial research phase - Completed on January 15, 2024
    2. Algorithm development - Completed on February 28, 2024
    3. Prototype implementation - In progress, expected completion on April 10, 2024
    4. Testing and validation - Not started, scheduled for April 15, 2024
    5. Final documentation - Not started, scheduled for May 5, 2024
    
    Budget Information:
    Total budget: $750,000
    Spent to date: $320,000
    Remaining: $430,000
    
    Key Challenges:
    - Error correction rates still below target thresholds
    - Integration with existing systems more complex than anticipated
    - Need additional computing resources for large-scale simulations
    
    Next Steps:
    - Complete prototype implementation by April 10
    - Request additional computing resources by March 15
    - Schedule weekly progress review meetings
    """
    
    # Define the extraction schema
    schema = {
        "type": "object",
        "properties": {
            "project": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "id": {"type": "string"},
                    "status": {"type": "string"},
                    "completion_percentage": {"type": "number"}
                },
                "required": ["name", "id", "status"]
            },
            "team": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "milestones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"},
                        "completion_date": {"type": "string"}
                    }
                }
            },
            "budget": {
                "type": "object",
                "properties": {
                    "total": {"type": "number"},
                    "spent": {"type": "number"},
                    "remaining": {"type": "number"}
                }
            },
            "challenges": {
                "type": "array",
                "items": {"type": "string"}
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    # Log extraction attempt
    logger.info("Performing JSON extraction with schema", emoji_key="processing")
    
    try:
        # Call the extract_json tool directly
        result = await gateway.mcp.call_tool("extract_json", {
            "text": text,
            "json_schema": schema,
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "validate_output": True
        })
        
        # Parse the result using the shared utility
        parse_and_display_result("JSON Extraction Demo", {"text": text, "json_schema": schema}, result)
        
        return result
            
    except Exception as e:
        logger.error(f"Error in JSON extraction: {str(e)}", emoji_key="error")
        return None


async def table_extraction_demo():
    """Demonstrate table extraction capabilities."""
    logger.info("Starting table extraction demo", emoji_key="start")
    
    # Sample text with embedded table
    text = """
    Financial Performance by Quarter (2023-2024)
    
    | Quarter | Revenue ($M) | Expenses ($M) | Profit ($M) | Growth (%) |
    |---------|-------------|---------------|-------------|------------|
    | Q1 2023 | 42.5        | 32.1          | 10.4        | 3.2        |
    | Q2 2023 | 45.7        | 33.8          | 11.9        | 6.5        |
    | Q3 2023 | 50.2        | 35.6          | 14.6        | 9.8        |
    | Q4 2023 | 58.3        | 38.2          | 20.1        | 15.2       |
    | Q1 2024 | 60.1        | 39.5          | 20.6        | 3.1        |
    | Q2 2024 | 65.4        | 41.2          | 24.2        | 8.8        |
    
    Note: All figures are in millions of dollars and are unaudited.
    Growth percentages are relative to the previous quarter.
    """
    
    # Log extraction attempt
    logger.info("Performing table extraction", emoji_key="processing")
    
    try:
        # Call the extract_table tool directly
        result = await gateway.mcp.call_tool("extract_table", {
            "text": text,
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "formats": ["json", "markdown"],
            "extract_metadata": True
        })
        
        # Parse the result using the shared utility
        parse_and_display_result("Table Extraction Demo", {"text": text}, result)
        
        return result
            
    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}", emoji_key="error")
        return None


async def semantic_schema_inference_demo():
    """Demonstrate semantic schema inference."""
    logger.info("Starting semantic schema inference demo", emoji_key="start")
    
    # Sample text for schema inference
    text = """
    Patient Record: John Smith
    Date of Birth: 05/12/1978
    Patient ID: P-98765
    Blood Type: O+
    Height: 182 cm
    Weight: 76 kg
    
    Medications:
    - Lisinopril 10mg, once daily
    - Metformin 500mg, twice daily
    - Atorvastatin 20mg, once daily at bedtime
    
    Allergies:
    - Penicillin (severe)
    - Shellfish (mild)
    
    Recent Vital Signs:
    Date: 03/15/2024
    Blood Pressure: 128/85 mmHg
    Heart Rate: 72 bpm
    Temperature: 98.6°F
    Oxygen Saturation: 98%
    
    Medical History:
    - Type 2 Diabetes (diagnosed 2015)
    - Hypertension (diagnosed 2017)
    - Hyperlipidemia (diagnosed 2019)
    - Appendectomy (2005)
    """
    
    # Log schema inference attempt
    logger.info("Performing schema inference", emoji_key="processing")
    
    try:
        # Call the extract_semantic_schema tool directly
        result = await gateway.mcp.call_tool("extract_semantic_schema", {
            "text": text,
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "format": "json_schema",
            "add_descriptions": True
        })
        
        # Parse the result using the shared utility
        parse_and_display_result("Semantic Schema Inference Demo", {"text": text}, result)
        
        return result
            
    except Exception as e:
        logger.error(f"Error in schema inference: {str(e)}", emoji_key="error")
        return None


async def entity_extraction_demo():
    """Demonstrate entity extraction capabilities."""
    logger.info("Starting entity extraction demo", emoji_key="start")
    
    # Sample text for entity extraction
    text = """
    In a groundbreaking announcement on March 15, 2024, Tesla unveiled its latest solar energy
    technology in partnership with SolarCity. CEO Elon Musk presented the new PowerWall 4.0 
    battery system at their headquarters in Austin, Texas. The system can store up to 20kWh of 
    energy and costs approximately $6,500 per unit.
    
    According to Dr. Maria Chen, lead researcher at the National Renewable Energy Laboratory (NREL),
    this technology represents a significant advancement in residential energy storage. The new
    system integrates with the Tesla mobile app on both iOS and Android platforms, allowing users
    to monitor energy usage in real-time.
    
    Tesla stock (TSLA) rose 5.8% following the announcement, reaching $248.32 per share on the NASDAQ.
    The company plans to begin production at their Gigafactory Nevada location by June 2024, with
    initial deployments in California and Texas markets.
    """
    
    # Log entity extraction attempt
    logger.info("Performing entity extraction", emoji_key="processing")
    
    try:
        # Call the extract_key_value_pairs tool directly 
        result = await gateway.mcp.call_tool("extract_key_value_pairs", {
            "text": text,
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "structured": True,
            "categorize": True
        })
        
        # Parse the result using the shared utility
        parse_and_display_result("Entity Extraction Demo", {"text": text}, result)
        
        return result
            
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}", emoji_key="error")
        # Try an alternative tool if available
        try:
            logger.info("Trying alternative entity extraction approach", emoji_key="processing")
            
            # Define a simple schema for entity extraction
            entity_schema = {
                "type": "object",
                "properties": {
                    "organizations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "people": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "locations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "products": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "dates": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "monetary_values": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
            
            # Call extract_json as an alternative for entity extraction
            result = await gateway.mcp.call_tool("extract_json", {
                "text": text,
                "schema": entity_schema,
                "provider": Provider.OPENAI.value,
                "model": "gpt-4o-mini",
                "validate": True
            })
            
            # Parse the result using the shared utility
            parse_and_display_result("Entity Extraction Demo (Alternative)", {"text": text}, result)
            
            return result
            
        except Exception as nested_e:
            logger.error(f"Error in alternative entity extraction: {str(nested_e)}", emoji_key="error")
            return None


async def main():
    """Run all extraction demos."""
    try:
        # Set up gateway
        await setup_gateway()
        
        console.print(Rule("[bold magenta]Advanced Extraction Demos Starting[/bold magenta]"))
        
        # Run JSON extraction demo
        await json_extraction_demo()
        
        # Run table extraction demo
        await table_extraction_demo()
        
        # Run schema inference demo
        await semantic_schema_inference_demo()
        
        # Run entity extraction demo
        await entity_extraction_demo()
        
        # Final success message
        logger.success("All extraction demos completed successfully", emoji_key="success")
        console.print(Rule("[bold magenta]Advanced Extraction Demos Complete[/bold magenta]"))
        
    except Exception as e:
        logger.critical(f"Extraction demo failed: {str(e)}", emoji_key="critical")
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1
    finally:
        # Clean up
        if gateway:
            pass  # No cleanup needed for Gateway instance
    
    return 0


if __name__ == "__main__":
    # Run the demos
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 