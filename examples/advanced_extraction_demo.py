#!/usr/bin/env python
"""Demo of advanced extraction capabilities using LLM Gateway MCP server."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway
from llm_gateway.tools.extraction import ExtractionTools
from llm_gateway.utils import get_logger
from llm_gateway.utils.display import parse_and_display_result
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.advanced_extraction")

# Initialize global gateway
gateway = None

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
    
    # Register extraction tools using our ExtractionTools class
    extraction_tools = ExtractionTools(gateway)
    
    # Get available tools after registering extraction tools
    tools_after = await gateway.mcp.list_tools()
    logger.info(f"Tools after registering extraction tools: {[t.name for t in tools_after]}", emoji_key="info")
    
    # Create a set of the new tools added
    new_tools = set([t.name for t in tools_after]) - set([t.name for t in tools_before])
    logger.success(f"Added extraction tools: {new_tools}", emoji_key="success")
    
    return extraction_tools

async def run_json_extraction_example():
    """Demonstrate JSON extraction."""
    console.print(Rule("[bold blue]1. JSON Extraction Example[/bold blue]"))
    
    # Load sample text
    sample_path = Path(__file__).parent / "data" / "sample_event.txt"
    if not sample_path.exists():
        # Create a sample text for demonstration
        sample_text = """
        Tech Conference 2024
        Location: San Francisco Convention Center, 123 Tech Blvd, San Francisco, CA 94103
        Date: June 15-17, 2024
        Time: 9:00 AM - 6:00 PM daily
        
        Registration Fee: $599 (Early Bird: $499 until March 31)
        
        Keynote Speakers:
        - Dr. Sarah Johnson, AI Research Director at TechCorp
        - Mark Williams, CTO of FutureTech Industries
        - Prof. Emily Chen, MIT Computer Science Department
        
        Special Events:
        - Networking Reception: June 15, 7:00 PM - 10:00 PM
        - Hackathon: June 16, 9:00 PM - 9:00 AM (overnight)
        - Career Fair: June 17, 1:00 PM - 5:00 PM
        
        For more information, contact events@techconference2024.example.com or call (555) 123-4567.
        """
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        # Write sample text to file
        with open(sample_path, "w") as f:
            f.write(sample_text)
    else:
        # Read existing sample text
        with open(sample_path, "r") as f:
            sample_text = f.read()
    
    # Display sample text
    console.print(Panel(sample_text, title="Sample Event Text", border_style="blue"))
    
    # Define JSON schema for event
    event_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Event name"},
            "location": {
                "type": "object",
                "properties": {
                    "venue": {"type": "string"},
                    "address": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "zip": {"type": "string"}
                }
            },
            "dates": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "format": "date"},
                    "end": {"type": "string", "format": "date"}
                }
            },
            "time": {"type": "string"},
            "registration": {
                "type": "object",
                "properties": {
                    "regular_fee": {"type": "number"},
                    "early_bird_fee": {"type": "number"},
                    "early_bird_deadline": {"type": "string", "format": "date"}
                }
            },
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "title": {"type": "string"},
                        "organization": {"type": "string"}
                    }
                }
            },
            "special_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string", "format": "date"},
                        "time": {"type": "string"}
                    }
                }
            },
            "contact": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "phone": {"type": "string"}
                }
            }
        }
    }
    
    # Display JSON schema
    schema_json = json.dumps(event_schema, indent=2)
    console.print(Panel(
        Syntax(schema_json, "json", theme="monokai", line_numbers=True),
        title="Event JSON Schema",
        border_style="green"
    ))
    
    # Extract JSON using extract_json tool
    logger.info("Extracting structured JSON data from text...", emoji_key="processing")
    
    try:
        start_time = time.time()  # noqa: F841
        
        # Call the extract_json method
        result = await gateway.mcp.call_tool("extract_json", {
            "text": sample_text,
            "json_schema": event_schema,
            "provider": Provider.OPENAI.value,
            "validate_output": True
        })
        
        # Display the results using the utility function
        parse_and_display_result(
            title="JSON Extraction Results",
            input_data={"text": sample_text, "schema": event_schema},
            result=result,
            console=console
        )
        
    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}", emoji_key="error", exc_info=True)
        
    console.print()

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
        await run_json_extraction_example()
        
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