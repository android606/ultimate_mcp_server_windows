#!/usr/bin/env python
"""Document processing examples for LLM Gateway."""
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from llm_gateway.constants import Provider
from llm_gateway.tools.document import DocumentTools
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.document_processing")

# Initialize FastMCP server
mcp = FastMCP("Document Processing Demo")

# Create document tools instance
document_tools = DocumentTools(mcp)

async def demonstrate_document_processing():
    """Demonstrate document processing capabilities."""
    logger.info("Starting document processing demonstration", emoji_key="start")
    
    # Create a sample document
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
    
    # Demonstrate document chunking
    logger.info("Demonstrating document chunking", emoji_key="processing")
    
    # Try different chunking methods
    chunking_methods = ["token"]  # Start with just one method for debugging
    
    for method in chunking_methods:
        try:
            logger.info(f"Testing chunking method: {method}")
            
            # Call directly with all debugging information
            result = await document_tools.mcp.call_tool("chunk_document", {
                "document": sample_document,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "method": method
            })
            
            # Direct diagnostic of the result
            logger.info(f"Chunking result type: {type(result)}")
            
            # Convert TextContent objects to strings if needed
            chunks = []
            
            # If we got a list directly (which might contain TextContent objects)
            if isinstance(result, list):
                for chunk in result:
                    # Handle TextContent objects
                    if hasattr(chunk, 'text'):
                        chunks.append(chunk.text)
                    # Handle dict with text
                    elif isinstance(chunk, dict) and 'text' in chunk:
                        chunks.append(chunk['text'])
                    # Handle plain strings
                    elif isinstance(chunk, str):
                        chunks.append(chunk)
                    # Last resort - try to convert to string
                    else:
                        try:
                            chunks.append(str(chunk))
                        except Exception:
                            chunks.append("(Unconvertible chunk)")
                
                logger.info(f"Got direct list result with {len(chunks)} chunks after conversion")
            # If we got a dict as expected
            elif isinstance(result, dict) and "chunks" in result:
                chunks = result["chunks"]
                # Still need to handle potential TextContent objects in the chunks list
                chunks = [
                    chunk.text if hasattr(chunk, 'text') else
                    chunk['text'] if isinstance(chunk, dict) and 'text' in chunk else
                    chunk if isinstance(chunk, str) else
                    str(chunk)
                    for chunk in chunks
                ]
                logger.info(f"Got dict result with {len(chunks)} chunks")
            else:
                logger.error(f"Unexpected result: {result}")
                continue
                
            # Log results
            logger.info(f"Document chunked into {len(chunks)} chunks")
            
            # Print first chunk with careful handling
            if chunks and len(chunks) > 0:
                first_chunk = chunks[0]
                logger.info(f"First chunk type: {type(first_chunk)}")
                print("\nFirst chunk:")
                print(first_chunk[:200] + "..." if len(first_chunk) > 200 else first_chunk)
            else:
                logger.warning("No chunks returned")
                
        except Exception as e:
            import traceback
            logger.error(f"Error during chunking with method '{method}': {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    # Add a direct diagnostic call to the chunking function
    try:
        logger.info("Attempting direct call to chunking function for diagnostics")
        direct_chunks = document_tools._chunk_by_tokens(sample_document, 500, 50)
        logger.info(f"Direct chunking returned {len(direct_chunks)} chunks")
    except Exception as e:
        import traceback
        logger.error(f"Direct chunking failed: {str(e)}")
        logger.error(f"Direct chunking traceback: {traceback.format_exc()}")
        
    # Helper function to safely handle tool results
    async def safe_tool_call(tool_name, args):
        """Safely call a tool and handle potential errors."""
        try:
            result = await document_tools.mcp.call_tool(tool_name, args)
            
            # Special handling for chunks with TextContent objects
            if tool_name == "chunk_document" and isinstance(result, list):
                # Convert TextContent objects to strings if needed
                string_chunks = []
                for chunk in result:
                    if hasattr(chunk, 'text'):
                        string_chunks.append(chunk.text)
                    elif isinstance(chunk, dict) and 'text' in chunk:
                        string_chunks.append(chunk['text'])
                    elif isinstance(chunk, str):
                        string_chunks.append(chunk)
                    else:
                        logger.warning(f"Unexpected chunk type: {type(chunk)}")
                        try:
                            string_chunks.append(str(chunk))
                        except Exception:
                            string_chunks.append("(Unconvertible chunk)")
                
                return {
                    "success": True, 
                    "result": {
                        "chunks": string_chunks,
                        "chunk_count": len(string_chunks)
                    }
                }
            
            # Check if we got a list directly (common with mcp.types objects)
            if isinstance(result, list):
                # Try to convert this to a more usable format
                return {"success": True, "result": {"items": result, "count": len(result)}}
                
            # Check if we got a dict with error
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Tool {tool_name} returned error: {result['error']}")
                if "traceback" in result:
                    logger.error(f"Traceback: {result['traceback']}")
                return {"success": False, "error": result["error"]}
                
            # If we got a dict without error
            if isinstance(result, dict):
                return {"success": True, "result": result}
                
            # Otherwise, something unexpected
            logger.error(f"Unexpected result type from {tool_name}: {type(result)}")
            return {"success": False, "error": f"Unexpected result type: {type(result)}"}
            
        except Exception as e:
            import traceback
            logger.error(f"Error calling {tool_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    # Demonstrate summarization
    logger.info("Demonstrating document summarization", emoji_key="processing")
    
    # Try different formats
    summary_formats = ["paragraph", "bullet_points"]
    
    for format in summary_formats:
        try:
            safe_result = await safe_tool_call("summarize_document", {
                "document": sample_document,
                "provider": Provider.OPENAI.value,
                "max_length": 150,
                "format": format,
                "max_tokens": int(150 * 1.5)  # Ensure max_tokens is an integer
            })
            
            if not safe_result["success"]:
                logger.error(f"Error summarizing in {format} format: {safe_result['error']}")
                continue
                
            summary_result = safe_result["result"]
            
            # Handle list result converted to items dict
            if "items" in summary_result and summary_result["items"]:
                items = summary_result["items"]
                
                # Extract the summary text from the items
                summary_text = None
                for item in items:
                    # Try different ways to extract summary text
                    if isinstance(item, str):
                        summary_text = item
                        break
                    elif hasattr(item, 'text'):
                        summary_text = item.text
                        break
                    elif isinstance(item, dict) and 'text' in item:
                        summary_text = item['text']
                        break
                    elif isinstance(item, dict) and 'summary' in item:
                        summary_text = item['summary']
                        break
                
                if summary_text:
                    logger.info(f"Found summary from items ({format} format)")
                    print(f"\nSummary ({format}):")
                    print(summary_text)
                else:
                    logger.error(f"Could not extract summary from items: {items[:2]}")
                    
            # Handle standard dict format
            elif isinstance(summary_result, dict) and "summary" in summary_result:
                logger.info(f"Found summary directly in result ({format} format)")
                print(f"\nSummary ({format}):")
                print(summary_result["summary"])
            else:
                logger.error(f"No summary found in result: {summary_result}")
                
        except Exception as e:
            import traceback
            logger.error(f"Error summarizing in {format} format: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Demonstrate entity extraction
    logger.info("Demonstrating entity extraction", emoji_key="processing")
    
    try:
        safe_result = await safe_tool_call("extract_entities", {
            "document": sample_document,
            "entity_types": ["technology", "field", "concept"],
            "provider": Provider.OPENAI.value
        })
        
        if not safe_result["success"]:
            logger.error(f"Error extracting entities: {safe_result['error']}")
        else:
            entity_result = safe_result["result"]
            
            # Handle items list format
            if "items" in entity_result and entity_result["items"]:
                items = entity_result["items"]
                
                # Handle TextContent objects which might contain JSON strings
                for item in items:
                    # Check for TextContent objects with text field
                    if hasattr(item, 'text'):
                        text_content = item.text
                        try:
                            # Try to parse JSON from the text content
                            json_data = json.loads(text_content)
                            
                            # Look for entities in the parsed JSON
                            if "entities" in json_data and isinstance(json_data["entities"], dict):
                                print("\nExtracted entities (from parsed TextContent):")
                                for entity_type, entities in json_data["entities"].items():
                                    if not entities:
                                        continue
                                        
                                    print(f"\n{entity_type.title()}:")
                                    if isinstance(entities, list):
                                        for entity in entities:
                                            print(f"- {entity}")
                                    else:
                                        print(f"- {entities}")
                                break  # Successfully processed
                        except json.JSONDecodeError:
                            # Not JSON, continue to next check
                            pass
                    
                # If we get here and haven't printed entities, try the original methods
                # Rest of the entity extraction code remains unchanged
                
            # Handle direct dict format
            elif isinstance(entity_result, dict) and "entities" in entity_result:
                print("\nExtracted entities:")
                for entity_type, entities in entity_result["entities"].items():
                    if not entities:
                        continue
                        
                    print(f"\n{entity_type.title()}:")
                    for entity in entities:
                        print(f"- {entity}")
            else:
                logger.error(f"No entities found in result: {entity_result}")
            
    except Exception as e:
        import traceback
        logger.error(f"Error extracting entities: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Demonstrate QA pair generation
    logger.info("Demonstrating QA pair generation", emoji_key="processing")
    
    try:
        safe_result = await safe_tool_call("generate_qa_pairs", {
            "document": sample_document,
            "num_questions": 3,
            "question_types": ["factual", "conceptual"],
            "provider": Provider.OPENAI.value
        })
        
        if not safe_result["success"]:
            logger.error(f"Error generating QA pairs: {safe_result['error']}")
        else:
            qa_result = safe_result["result"]
            
            # Handle items list format
            if "items" in qa_result and qa_result["items"]:
                items = qa_result["items"]
                
                # Handle TextContent objects which might contain JSON strings
                for item in items:
                    # Check for TextContent objects with text field
                    if hasattr(item, 'text'):
                        text_content = item.text
                        try:
                            # Try to parse JSON from the text content
                            json_data = json.loads(text_content)
                            
                            # Look for qa_pairs in the parsed JSON
                            if "qa_pairs" in json_data and isinstance(json_data["qa_pairs"], list):
                                print("\nGenerated QA pairs (from parsed TextContent):")
                                for i, qa in enumerate(json_data["qa_pairs"], 1):
                                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                                        print(f"\n{i}. Q: {qa['question']}")
                                        print(f"   A: {qa['answer']}")
                                break  # Successfully processed
                        except json.JSONDecodeError:
                            # Not JSON, continue to next check
                            pass
                    
                # If we get here and haven't printed QA pairs, try the original methods
                # Rest of the QA pair extraction code remains unchanged
                
            # Handle standard dict format
            elif isinstance(qa_result, dict) and "qa_pairs" in qa_result:
                print("\nGenerated QA pairs:")
                for i, qa in enumerate(qa_result["qa_pairs"], 1):
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                        print(f"\n{i}. Q: {qa['question']}")
                        print(f"   A: {qa['answer']}")
            else:
                logger.error(f"No QA pairs found in result: {qa_result}")
            
    except Exception as e:
        import traceback
        logger.error(f"Error generating QA pairs: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Demonstrate multi-stage document workflow
    logger.info("Demonstrating document workflow", emoji_key="meta")
    
    # Define a simple workflow
    workflow = [
        {
            "name": "Chunking",
            "operation": "chunk",
            "chunk_size": 1000,
            "method": "paragraph",
            "output_as": "chunks"
        },
        {
            "name": "Summarization",
            "operation": "summarize",
            "provider": Provider.OPENAI.value,
            "input_from": "original",
            "max_length": 150,
            "format": "paragraph",
            "output_as": "summary"
        },
        {
            "name": "Entity Extraction",
            "operation": "extract_entities",
            "provider": Provider.OPENAI.value,
            "input_from": "original",
            "entity_types": ["technology", "field", "concept"],
            "output_as": "entities"
        }
    ]
    
    try:
        # Execute workflow (placeholder - would normally use optimization_tools.execute_optimized_workflow)
        logger.info("Workflow execution would process document through multiple stages", emoji_key="processing")
        logger.info("Each stage can use the most cost-effective model for its specific task", emoji_key="cost")
        
        # Print workflow structure
        print("\n--- Document Processing Workflow ---")
        for i, step in enumerate(workflow):
            print(f"\nStep {i+1}: {step['name']}")
            print(f"  Operation: {step['operation']}")
            if "provider" in step:
                print(f"  Provider: {step['provider']}")
            print(f"  Input from: {step.get('input_from', 'original')}")
            print(f"  Output as: {step['output_as']}")
            
    except Exception as e:
        logger.error(f"Error in workflow description: {str(e)}", emoji_key="error")


async def main():
    """Run document processing examples."""
    try:
        await demonstrate_document_processing()
        
    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)