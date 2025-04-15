#!/usr/bin/env python
"""Document processing examples for LLM Gateway."""
import asyncio
import sys
from collections import namedtuple  # Import namedtuple
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule

from llm_gateway.constants import Provider
from llm_gateway.services.cache import get_cache_service

# Remove the old import
# from llm_gateway.tools.document import DocumentTools
# Add imports for the actual tool functions used
from llm_gateway.tools.document import (
    chunk_document,
    extract_entities,
    generate_qa_pairs,
    summarize_document,
)
from llm_gateway.utils import get_logger

# --- Import display utilities ---
from llm_gateway.utils.display import CostTracker, display_text_content_result  # Import CostTracker
from llm_gateway.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.document_processing")

# Create a simple structure for cost tracking from dict
TrackableResult = namedtuple("TrackableResult", ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"])

# Initialize FastMCP server
mcp = FastMCP("Document Processing Demo")

# Manually register the imported tool functions with the MCP instance
mcp.tool()(chunk_document)
mcp.tool()(summarize_document)
mcp.tool()(extract_entities)
mcp.tool()(generate_qa_pairs)

# Remove the instantiation of the old class
# document_tools = DocumentTools(mcp)

async def safe_tool_call(tool_name: str, args: dict) -> dict:
    try:
        result = await mcp.call_tool(tool_name, args)
        
        # Basic type checking and error handling
        if isinstance(result, dict) and result.get("error"):
            logger.error(f"Tool {tool_name} returned error: {result['error']}", emoji_key="error")
            return {"success": False, "error": result["error"]}
        
        # If successful, assume dict or potentially list (like chunks)
        return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"Exception calling {tool_name}: {e}", emoji_key="error", exc_info=True)
        return {"success": False, "error": str(e)}

async def demonstrate_document_processing(tracker: CostTracker): # Add tracker
    """Demonstrate document processing capabilities using Rich."""
    console.print(Rule("[bold blue]Document Processing Demonstration[/bold blue]"))
    logger.info("Starting document processing demonstration", emoji_key="start")
    
    # --- Clear Cache --- 
    try:
        logger.info("Clearing cache before demonstration...", emoji_key="cache")
        cache_service = get_cache_service()
        cache_service.clear()
        logger.success("Cache cleared successfully.", emoji_key="success")
    except Exception as e:
         logger.warning(f"Could not clear cache: {e}", emoji_key="warning")
    
    console.print()

    sample_document = """
    # Artificial Intelligence: An Overview
    
    Artificial Intelligence (AI) refers to systems or machines that mimic human intelligence to perform tasks and can iteratively improve themselves based on the information they collect. AI manifests in a number of forms including:
    
    ## Machine Learning
    
    Machine Learning is a subset of AI that enables a system to learn from data rather than through explicit programming. It involves algorithms that improve automatically through experience.
    
    Popular machine learning methods include:
    - Supervised Learning: Training on labeled data
    - Unsupervised Learning: Finding patterns in unlabeled data
    - Reinforcement Learning: Learning through trial and error
    
    ## Deep Learning
    
    Deep Learning is a subset of Machine Learning that uses neural networks with many layers (hence "deep") to analyze various factors of data. It is especially useful for:
    
    1. Image and speech recognition
    2. Natural language processing
    3. Recommendation systems
    
    ## Applications in Industry
    
    AI is transforming various industries:
    
    ### Healthcare
    
    In healthcare, AI is being used for:
    - Diagnosing diseases
    - Developing personalized treatment plans
    - Drug discovery
    - Managing medical records
    
    ### Finance
    
    In finance, AI applications include:
    - Fraud detection
    - Algorithmic trading
    - Risk assessment
    - Customer service automation
    
    ### Transportation
    
    The transportation industry uses AI for:
    - Autonomous vehicles
    - Traffic management
    - Predictive maintenance
    - Route optimization
    
    ## Ethical Considerations
    
    As AI becomes more prevalent, ethical considerations become increasingly important:
    
    - Privacy concerns
    - Algorithmic bias and fairness
    - Job displacement
    - Accountability for AI decisions
    - Security vulnerabilities
    
    ## Future Directions
    
    The future of AI might include:
    
    - More transparent and explainable models
    - Increased integration with robotics
    - Greater autonomy and self-improvement capabilities
    - Expanded creative applications
    
    As technology continues to advance, the potential applications and implications of AI will only grow in significance and impact across all aspects of society.
    """
    console.print(Panel(escape(sample_document[:500] + "..."), title="[cyan]Input Document Snippet[/cyan]", border_style="dim blue"))
    console.print()

    # --- Demonstrate Chunking --- 
    console.print(Rule("[cyan]Document Chunking[/cyan]"))
    logger.info("Demonstrating document chunking (method: token)", emoji_key="processing")
    chunk_args = {
        "document": sample_document,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "method": "token"
    }
    chunk_response = await safe_tool_call("chunk_document", chunk_args)

    if chunk_response["success"]:
        # Result might be a dict {chunks: []} or just the list []
        chunks = []
        raw_result = chunk_response["result"]
        if isinstance(raw_result, dict) and "chunks" in raw_result:
             chunks = raw_result["chunks"]
        elif isinstance(raw_result, list):
             chunks = raw_result # Assume list is the chunks directly
        else:
             logger.warning(f"Unexpected chunk result format: {type(raw_result)}", emoji_key="warning")

        # Handle potential TextContent objects within chunks list
        string_chunks = []
        for chunk in chunks:
            if hasattr(chunk, 'text'): 
                string_chunks.append(chunk.text)
            elif isinstance(chunk, str): 
                string_chunks.append(chunk)
            else: 
                string_chunks.append(str(chunk))

        logger.info(f"Chunked document into {len(string_chunks)} chunks.", emoji_key="success")
        if string_chunks:
            # Display first chunk preview
            preview = string_chunks[0][:300] + ("..." if len(string_chunks[0]) > 300 else "")
            console.print(Panel(
                escape(preview), 
                title="[bold]First Chunk Preview[/bold]", 
                subtitle=f"Total Chunks: {len(string_chunks)}", 
                border_style="blue"
            ))
        else:
             console.print("[yellow]No chunks were generated.[/yellow]")
    else:
        console.print(f"[bold red]Chunking Failed:[/bold red] {escape(chunk_response['error'])}")
    console.print()

    # --- Demonstrate Summarization --- 
    console.print(Rule("[cyan]Document Summarization[/cyan]"))
    logger.info("Demonstrating document summarization...", emoji_key="processing")
    summary_args = {
        "document": sample_document,
        "summary_length": "short", # concise, short, detailed
        "format": "paragraph", # bullet_points, paragraph
        "provider": Provider.OPENAI.value,
        "model": "gpt-4.1-mini"
    }
    summary_response = await safe_tool_call("summarize_document", summary_args)

    if summary_response["success"]:
        # Use the improved display utility to show the result
        display_text_content_result(
            "Generated Summary",
            summary_response["result"]
        )
        # Track cost
        if isinstance(summary_response["result"], dict) and all(k in summary_response["result"] for k in ["cost", "provider", "model"]) and "tokens" in summary_response["result"]:
            try:
                trackable = TrackableResult(
                    cost=summary_response["result"].get("cost", 0.0),
                    input_tokens=summary_response["result"].get("tokens", {}).get("input", 0),
                    output_tokens=summary_response["result"].get("tokens", {}).get("output", 0),
                    provider=summary_response["result"].get("provider", "unknown"),
                    model=summary_response["result"].get("model", "unknown"),
                    processing_time=summary_response["result"].get("processing_time", 0.0)
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(f"Could not track cost for summarization: {track_err}", exc_info=False)
    else:
        console.print(f"[bold red]Summarization Failed:[/bold red] {escape(summary_response['error'])}")
    console.print()

    # --- Demonstrate Entity Extraction --- 
    console.print(Rule("[cyan]Entity Extraction[/cyan]"))
    logger.info("Demonstrating entity extraction...", emoji_key="processing")
    entity_args = {
        "document": sample_document,
        "entity_types": ["field", "industry", "concept"], # Types more likely present
        "provider": Provider.OPENAI.value,
        "model": "gpt-4.1-mini"
    }
    entity_response = await safe_tool_call("extract_entities", entity_args)

    if entity_response["success"]:
        # Use the new display utility that better handles TextContent objects
        display_text_content_result(
            "Extracted Entities",
            entity_response["result"]
        )
        # Track cost
        if isinstance(entity_response["result"], dict) and all(k in entity_response["result"] for k in ["cost", "provider", "model"]) and "tokens" in entity_response["result"]:
            try:
                trackable = TrackableResult(
                    cost=entity_response["result"].get("cost", 0.0),
                    input_tokens=entity_response["result"].get("tokens", {}).get("input", 0),
                    output_tokens=entity_response["result"].get("tokens", {}).get("output", 0),
                    provider=entity_response["result"].get("provider", "unknown"),
                    model=entity_response["result"].get("model", "unknown"),
                    processing_time=entity_response["result"].get("processing_time", 0.0)
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(f"Could not track cost for entity extraction: {track_err}", exc_info=False)
    else:
        console.print(f"[bold red]Entity Extraction Failed:[/bold red] {escape(entity_response['error'])}")
    console.print()

    # --- Demonstrate Q&A Generation --- 
    console.print(Rule("[cyan]Question & Answer Generation[/cyan]"))
    logger.info("Demonstrating Q&A generation...", emoji_key="processing")
    qa_args = {
        "document": sample_document,
        "num_questions": 3,
        "question_types": ["factual", "conceptual"],
        "provider": Provider.OPENAI.value,
        "model": "gpt-4.1-mini"
    }
    # Use generate_qa_pairs instead of generate_qa
    qa_response = await safe_tool_call("generate_qa_pairs", qa_args)

    if qa_response["success"]:
        # Use the new display utility that better handles TextContent objects
        display_text_content_result(
            "Generated Q&A Pairs",
            qa_response["result"]
        )
        # Track cost
        if isinstance(qa_response["result"], dict) and all(k in qa_response["result"] for k in ["cost", "provider", "model"]) and "tokens" in qa_response["result"]:
            try:
                trackable = TrackableResult(
                    cost=qa_response["result"].get("cost", 0.0),
                    input_tokens=qa_response["result"].get("tokens", {}).get("input", 0),
                    output_tokens=qa_response["result"].get("tokens", {}).get("output", 0),
                    provider=qa_response["result"].get("provider", "unknown"),
                    model=qa_response["result"].get("model", "unknown"),
                    processing_time=qa_response["result"].get("processing_time", 0.0)
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(f"Could not track cost for Q&A generation: {track_err}", exc_info=False)
    else:
        console.print(f"[bold red]Q&A Generation Failed:[/bold red] {escape(qa_response['error'])}")
    console.print()

    # Display cost summary at the end
    tracker.display_summary(console)

async def main():
    """Run document processing demonstration."""
    tracker = CostTracker() # Instantiate tracker
    try:
        await demonstrate_document_processing(tracker) # Pass tracker
        
    except Exception as e:
        logger.critical(f"Document processing demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1
    
    logger.success("Document Processing Demo Finished Successfully!", emoji_key="complete")
    console.print(Rule("[bold magenta]Document Processing Demo Complete[/bold magenta]"))
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)