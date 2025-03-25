#!/usr/bin/env python
"""Document processing examples for LLM Gateway."""
import asyncio
import sys
from pathlib import Path
import textwrap

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_gateway.constants import Provider
from llm_gateway.tools.document import DocumentTools
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.document_processing")


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
    
    # Create document tools instance
    document_tools = DocumentTools(None)
    
    # Demonstrate document chunking
    logger.info("Demonstrating document chunking", emoji_key="processing")
    
    # Try different chunking methods
    chunking_methods = ["token", "character", "paragraph", "semantic"]
    
    for method in chunking_methods:
        chunk_result = await document_tools.mcp.execute("chunk_document", {
            "document": sample_document,
            "chunk_size": 500,  # Characters or tokens depending on method
            "chunk_overlap": 50,
            "method": method
        })
        
        # Log result
        logger.success(
            f"Document chunked with {method} method",
            emoji_key="success",
            chunks=chunk_result["chunk_count"],
            time=chunk_result["processing_time"]
        )
        
        # Print first chunk
        if chunk_result["chunks"]:
            print(f"\n--- First chunk with {method} method ---")
            print(textwrap.fill(chunk_result["chunks"][0], width=80)[:200] + "...")
            print(f"... (plus {len(chunk_result['chunks'])-1} more chunks)")
    
    # Demonstrate summarization
    logger.info("Demonstrating document summarization", emoji_key="processing")
    
    # Try different formats
    summary_formats = ["paragraph", "bullet_points"]
    
    for format in summary_formats:
        try:
            summary_result = await document_tools.mcp.execute("summarize_document", {
                "document": sample_document,
                "provider": Provider.OPENAI.value,
                "max_length": 150,
                "format": format
            })
            
            # Log result
            logger.success(
                f"Document summarized in {format} format",
                emoji_key="success",
                tokens={
                    "input": summary_result["tokens"]["input"],
                    "output": summary_result["tokens"]["output"]
                },
                cost=summary_result["cost"],
                time=summary_result["processing_time"]
            )
            
            # Print summary
            print(f"\n--- Summary in {format} format ---")
            print(summary_result["summary"])
            
        except Exception as e:
            logger.error(f"Error summarizing in {format} format: {str(e)}", emoji_key="error")
    
    # Demonstrate entity extraction
    logger.info("Demonstrating entity extraction", emoji_key="processing")
    
    try:
        entity_result = await document_tools.mcp.execute("extract_entities", {
            "document": sample_document,
            "entity_types": ["technology", "field", "concept"],
            "provider": Provider.OPENAI.value
        })
        
        # Log result
        logger.success(
            f"Entities extracted from document",
            emoji_key="success",
            tokens={
                "input": entity_result["tokens"]["input"],
                "output": entity_result["tokens"]["output"]
            },
            cost=entity_result["cost"],
            time=entity_result["processing_time"]
        )
        
        # Print entities
        print("\n--- Extracted Entities ---")
        for entity_type, entities in entity_result["entities"].items():
            print(f"\n{entity_type.upper()}:")
            for entity in entities:
                print(f"- {entity}")
            
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}", emoji_key="error")
    
    # Demonstrate QA pair generation
    logger.info("Demonstrating QA pair generation", emoji_key="processing")
    
    try:
        qa_result = await document_tools.mcp.execute("generate_qa_pairs", {
            "document": sample_document,
            "num_questions": 3,
            "question_types": ["factual", "conceptual"],
            "provider": Provider.OPENAI.value
        })
        
        # Log result
        logger.success(
            f"QA pairs generated from document",
            emoji_key="success",
            tokens={
                "input": qa_result["tokens"]["input"],
                "output": qa_result["tokens"]["output"]
            },
            cost=qa_result["cost"],
            time=qa_result["processing_time"]
        )
        
        # Print QA pairs
        print("\n--- Generated QA Pairs ---")
        for i, qa_pair in enumerate(qa_result["qa_pairs"]):
            print(f"\nQ{i+1} ({qa_pair['type']}): {qa_pair['question']}")
            print(f"A: {qa_pair['answer']}")
            
    except Exception as e:
        logger.error(f"Error generating QA pairs: {str(e)}", emoji_key="error")
    
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