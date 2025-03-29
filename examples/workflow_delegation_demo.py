#!/usr/bin/env python
"""Workflow delegation example using LLM Gateway MCP server."""
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import Context, FastMCP

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.workflow_delegation")

# Initialize FastMCP server
mcp = FastMCP("Workflow Delegation Demo")

# Register meta tools directly
@mcp.tool()
async def analyze_task(
    task_description: str,
    available_providers: Optional[List[str]] = None,
    analyze_features: bool = True,
    analyze_cost: bool = True,
    ctx = None
) -> Dict[str, Any]:
    """Analyze a task and recommend suitable models."""
    start_time = time.time()
    
    # Mock implementation for demonstration
    if not available_providers:
        available_providers = [Provider.OPENAI.value, Provider.GEMINI.value, Provider.ANTHROPIC.value]
    
    # Analyze task type based on description
    task_type = "extraction" if "extract" in task_description.lower() else \
                "summarization" if "summarize" in task_description.lower() else \
                "generation"
    
    # Mock required features
    if "entities" in task_description.lower():
        required_features = ["entity_recognition", "classification"]
        features_explanation = "This task requires entity recognition capabilities to identify key concepts."
    elif "technical" in task_description.lower():
        required_features = ["domain_knowledge", "technical_understanding"]
        features_explanation = "This task requires technical domain knowledge to properly analyze content."
    else:
        required_features = ["text_processing"]
        features_explanation = "This is a general text processing task."
    
    # Generate recommendations
    recommendations = [
        {
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o",
            "explanation": "Best overall quality for complex tasks"
        },
        {
            "provider": Provider.ANTHROPIC.value,
            "model": "claude-3-opus-20240229",
            "explanation": "Excellent for technical content analysis"
        },
        {
            "provider": Provider.GEMINI.value,
            "model": "gemini-2.0-pro",
            "explanation": "Good balance of performance and cost"
        }
    ]
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "task_type": task_type,
        "required_features": required_features,
        "features_explanation": features_explanation,
        "recommendations": recommendations,
        "processing_time": processing_time
    }

@mcp.tool()
async def delegate_task(
    task_description: str,
    prompt: str,
    optimization_criteria: str = "balanced",
    available_providers: Optional[List[str]] = None,
    max_cost: Optional[float] = None,
    ctx = None
) -> Dict[str, Any]:
    """Delegate a task to the most appropriate provider."""
    start_time = time.time()
    
    # Mock implementation for demonstration
    if not available_providers:
        available_providers = [Provider.OPENAI.value, Provider.GEMINI.value]
    
    # Select a provider based on criteria
    if optimization_criteria == "cost":
        provider = Provider.GEMINI.value
        model = "gemini-2.0-flash-lite"
    elif optimization_criteria == "quality":
        provider = Provider.OPENAI.value
        model = "gpt-4o"
    else:  # balanced
        provider = Provider.OPENAI.value
        model = "gpt-4o-mini"
    
    # Get provider instance
    provider_instance = get_provider(provider)
    await provider_instance.initialize()
    
    # Generate completion
    result = await provider_instance.generate_completion(
        prompt=prompt,
        model=model,
        temperature=0.7,
        max_tokens=300
    )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "text": result.text,
        "provider": provider,
        "model": model,
        "processing_time": processing_time,
        "cost": result.cost,
        "tokens": {
            "input": result.input_tokens,
            "output": result.output_tokens,
            "total": result.total_tokens
        }
    }

@mcp.tool()
async def execute_workflow(
    workflow_steps: List[Dict[str, Any]],
    initial_input: str,
    max_concurrency: int = 1,
    ctx = None
) -> Dict[str, Any]:
    """Execute a multi-step workflow."""
    start_time = time.time()
    total_cost = 0.0
    
    # Initialize output collection
    outputs = {}
    
    # Initialize input for first step
    current_input = initial_input
    
    # Process each step sequentially (for demo)
    for step in workflow_steps:
        step_id = step.get("id", "unknown")
        operation = step.get("operation", "")
        provider_name = step.get("provider", Provider.OPENAI.value)
        model_name = step.get("model", "")
        parameters = step.get("parameters", {})
        output_as = step.get("output_as", step_id)
        
        # Check if we should use output from previous step
        if "input_from" in step and step["input_from"] in outputs:
            current_input = outputs[step["input_from"]]
        
        # Get provider instance
        provider = get_provider(provider_name)
        await provider.initialize()
        
        # Create prompts based on operation
        if operation == "summarize":
            prompt = f"Summarize the following text. {parameters.get('format', 'Keep it concise')}:\n\n{current_input}"
        elif operation == "extract_entities":
            entity_types = parameters.get("entity_types", ["organization", "person", "concept"])
            prompt = f"Extract the following entity types from the text: {', '.join(entity_types)}.\n\n{current_input}"
        elif operation == "generate_questions":
            prompt = f"Generate {parameters.get('question_count', 3)} {parameters.get('question_type', 'analytical')} questions about the following text:\n\n{current_input}"
        else:
            prompt = current_input
        
        # Generate completion
        result = await provider.generate_completion(
            prompt=prompt,
            model=model_name,
            temperature=0.7,
            max_tokens=500
        )
        
        # Store output
        outputs[output_as] = result.text
        
        # Add to total cost
        total_cost += result.cost
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "outputs": outputs,
        "processing_time": processing_time,
        "total_cost": total_cost
    }

@mcp.tool()
async def optimize_prompt(
    prompt: str,
    target_model: str,
    optimization_type: str = "general",
    provider: str = Provider.OPENAI.value,
    ctx = None
) -> Dict[str, Any]:
    """Optimize a prompt for a specific model."""
    # Get provider instance
    provider_instance = get_provider(provider)
    await provider_instance.initialize()
    
    # Create optimization prompt
    optimization_prompt = f"""
    I need to optimize this prompt for the {target_model} model:
    
    "{prompt}"
    
    Please rewrite this prompt to work optimally with {target_model}, 
    focusing on {optimization_type} optimization.
    
    Return ONLY the optimized prompt with no explanations.
    """
    
    # Generate optimized prompt
    result = await provider_instance.generate_completion(
        prompt=optimization_prompt,
        model=provider_instance.get_default_model(),
        temperature=0.7,
        max_tokens=300
    )
    
    # Return optimized prompt
    return {
        "original_prompt": prompt,
        "optimized_prompt": result.text.strip(),
        "target_model": target_model,
        "optimization_type": optimization_type,
        "cost": result.cost
    }

async def run_analyze_task_demo():
    """Demonstrate the analyze_task capability."""
    logger.info("Starting task analysis demo using real MCP tools", emoji_key="start")
    
    # Define a sample task
    task_description = "Extract key entities and relationships from a technical whitepaper about quantum computing"
    
    # Call the analyze_task tool via MCP
    logger.info(f"Analyzing task: {task_description}", emoji_key="processing")
    
    try:
        # Call the tool
        analysis_result = await mcp.call_tool("analyze_task", {
            "task_description": task_description,
            "available_providers": [Provider.OPENAI.value, Provider.GEMINI.value, Provider.ANTHROPIC.value],
            "analyze_features": True,
            "analyze_cost": True
        })
        
        # Parse the JSON if it's in a list
        analysis = {}
        if isinstance(analysis_result, list) and len(analysis_result) > 0:
            # Extract the first result
            first_item = analysis_result[0]
            # Check if it has text attribute (typical response format)
            if hasattr(first_item, 'text'):
                try:
                    analysis = json.loads(first_item.text)
                except json.JSONDecodeError:
                    # If can't parse as JSON, use text directly
                    analysis = {"task_type": "extraction", 
                               "required_features": ["entity_recognition"], 
                               "features_explanation": first_item.text,
                               "recommendations": [],
                               "processing_time": 0.0}
        elif isinstance(analysis_result, dict):
            analysis = analysis_result
        
        # Print the analysis
        logger.success("Task analysis completed", emoji_key="success")
        print("\n" + "-" * 80)
        print(f"Task Type: {analysis.get('task_type', 'Unknown')}")
        print(f"Required Features: {', '.join(analysis.get('required_features', []))}")
        print(f"Features Explanation: {analysis.get('features_explanation', 'No explanation available')}")
        print("\nRecommendations:")
        
        # Handle recommendations safely
        recommendations = analysis.get('recommendations', [])
        if isinstance(recommendations, list):
            for i, rec in enumerate(recommendations, 1):
                if isinstance(rec, dict):
                    provider = rec.get('provider', 'Unknown')
                    model = rec.get('model', 'Unknown')
                    explanation = rec.get('explanation', 'No explanation')
                    print(f"{i}. {provider} - {model}: {explanation}")
                else:
                    print(f"{i}. {rec}")
        
        print("-" * 80 + "\n")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing task: {str(e)}", emoji_key="error")
        return None


async def run_delegate_task_demo():
    """Demonstrate the delegate_task capability."""
    logger.info("Starting task delegation demo", emoji_key="start")
    
    # Define a task for delegation
    task_description = "Summarize the key advantages of quantum computing"
    prompt = """
    Quantum computing is an emerging technology that leverages quantum mechanics to process information in ways that classical computers cannot. Unlike classical computers that use bits (0s and 1s), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously due to a quantum property called superposition. This allows quantum computers to perform certain calculations exponentially faster than classical computers.
    
    Another key quantum property is entanglement, where qubits become correlated such that the state of one qubit instantly influences another, regardless of distance. This enables quantum computers to process complex problems more efficiently.
    
    Current applications being explored include cryptography (both breaking existing encryption and creating quantum-resistant algorithms), drug discovery (simulating molecular interactions), optimization problems (finding optimal solutions in complex systems), and machine learning (faster processing of certain algorithms).
    
    Despite these promising applications, quantum computers face significant challenges including maintaining quantum coherence (qubits are very sensitive to environmental disturbances), error correction (quantum states are fragile), and scaling (building systems with more qubits while maintaining their quantum properties).
    
    Companies like IBM, Google, Microsoft, and several startups are actively developing quantum computing hardware and software. The field continues to advance rapidly, with researchers working to overcome current limitations and build practical quantum computing systems.
    """
    
    # Call the delegate_task tool via MCP
    logger.info("Delegating summarization task to appropriate provider", emoji_key="processing")
    
    try:
        # Call the tool
        delegate_result = await mcp.call_tool("delegate_task", {
            "task_description": task_description,
            "prompt": prompt,
            "optimization_criteria": "balanced",
            "available_providers": [Provider.OPENAI.value, Provider.GEMINI.value],
            "max_cost": 0.05
        })
        
        # Parse the JSON if it's in a list
        result = {}
        if isinstance(delegate_result, list) and len(delegate_result) > 0:
            # Extract the first result
            first_item = delegate_result[0]
            # Check if it has text attribute
            if hasattr(first_item, 'text'):
                try:
                    result = json.loads(first_item.text)
                except json.JSONDecodeError:
                    result = {"text": first_item.text, 
                             "provider": Provider.OPENAI.value,
                             "model": "gpt-4o-mini",
                             "processing_time": 0.0,
                             "cost": 0.0}
        elif isinstance(delegate_result, dict):
            result = delegate_result
        
        # Print the delegation results safely
        logger.success("Task delegation completed", emoji_key="success")
        print("\n" + "-" * 80)
        print("Delegation Results:")
        print(f"Provider: {result.get('provider', 'Unknown')}")
        print(f"Model: {result.get('model', 'Unknown')}")
        print(f"Processing Time: {result.get('processing_time', 0.0):.2f}s")
        print(f"Cost: ${result.get('cost', 0.0):.6f}")
        print("\nSummary Result:")
        print(result.get('text', 'No text available'))
        print("-" * 80 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Error delegating task: {str(e)}", emoji_key="error")
        return None


async def run_workflow_demo():
    """Demonstrate the workflow execution capability with real tools."""
    logger.info("Starting workflow execution demo", emoji_key="start")
    
    # Sample document for processing
    document = """
    Quantum computing is an emerging technology that uses quantum mechanics to solve problems 
    that are too complex for classical computers. Unlike classical computing which uses bits
    (0s and 1s), quantum computing uses quantum bits or qubits that can exist in multiple states
    simultaneously due to a property called superposition. The field is advancing rapidly with
    companies like IBM, Google, and D-Wave leading development efforts. Recent breakthroughs
    include Google's quantum supremacy experiment and IBM's 127-qubit processor.
    
    One key application is in cryptography, where quantum computers could potentially break current
    encryption methods but also enable new quantum-resistant cryptographic techniques. Other promising
    applications include drug discovery, materials science, optimization problems, and machine learning.
    
    Challenges remain in scaling quantum systems, error correction, and maintaining quantum coherence.
    Despite these challenges, investment in the field continues to grow substantially.
    """
    
    # Define a real workflow
    workflow_steps = [
        {
            "id": "summarize",
            "name": "Summarize Document",
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "operation": "summarize",
            "parameters": {
                "max_length": 100,
                "format": "paragraph"
            },
            "output_as": "summary"
        },
        {
            "id": "entities",
            "name": "Extract Entities",
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "operation": "extract_entities",
            "parameters": {
                "entity_types": ["organization", "technology", "concept"]
            },
            "output_as": "entities"
        },
        {
            "id": "questions",
            "name": "Generate Questions",
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "operation": "generate_questions",
            "input_from": "summary",
            "parameters": {
                "question_count": 3,
                "question_type": "analytical"
            },
            "output_as": "questions"
        }
    ]
    
    # Execute the workflow
    logger.info("Executing workflow with 3 steps using MCP tools", emoji_key="processing")
    
    try:
        # Call the workflow execution tool
        workflow_result = await mcp.call_tool("execute_workflow", {
            "workflow_steps": workflow_steps,
            "initial_input": document,
            "max_concurrency": 1  # Sequential for demonstration
        })
        
        # Parse the JSON if it's in a list
        results = {}
        if isinstance(workflow_result, list) and len(workflow_result) > 0:
            # Extract the first result
            first_item = workflow_result[0]
            # Check if it has text attribute
            if hasattr(first_item, 'text'):
                try:
                    results = json.loads(first_item.text)
                except json.JSONDecodeError:
                    # If we can't parse JSON, create a basic structure
                    results = {
                        "outputs": {
                            "summary": first_item.text,
                            "entities": "JSON parsing error",
                            "questions": "Could not parse result"
                        },
                        "processing_time": 0.0,
                        "total_cost": 0.0
                    }
        elif isinstance(workflow_result, dict):
            results = workflow_result
        
        # Print the workflow results
        logger.success("Workflow execution completed", emoji_key="success")
        print("\n" + "-" * 80)
        print("Workflow Results:")
        
        # Print outputs from each step
        if "outputs" in results:
            # Print summary
            if "summary" in results["outputs"]:
                print("\n1. Document Summary:")
                print(results["outputs"]["summary"])
            
            # Print entities if available
            if "entities" in results["outputs"]:
                print("\n2. Extracted Entities:")
                entities = results["outputs"]["entities"]
                if isinstance(entities, str):
                    # Try to parse JSON if it's a string
                    try:
                        entities = json.loads(entities)
                    except Exception:
                        pass
                
                if isinstance(entities, dict):
                    for entity_type, entity_list in entities.items():
                        print(f"  {entity_type.capitalize()}: {', '.join(entity_list)}")
                else:
                    print(entities)
            
            # Print questions if available
            if "questions" in results["outputs"]:
                print("\n3. Generated Questions:")
                questions = results["outputs"]["questions"]
                
                if isinstance(questions, str):
                    # Try to parse JSON if it's a string
                    try:
                        questions = json.loads(questions)
                    except Exception:
                        # If can't parse, check if it's a formatted string with line breaks
                        if "\n" in questions:
                            questions = [q.strip() for q in questions.split("\n") if q.strip()]
                        else:
                            questions = [questions]
                
                if isinstance(questions, list):
                    for i, question in enumerate(questions, 1):
                        print(f"  Q{i}: {question}")
                else:
                    print(questions)
        
        # Print overall stats safely
        print("\nWorkflow Statistics:")
        print(f"  Total processing time: {results.get('processing_time', 0.0):.2f}s")
        print(f"  Total cost: ${results.get('total_cost', 0.0):.6f}")
        print("-" * 80 + "\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error executing workflow: {str(e)}", emoji_key="error")
        return None


async def run_prompt_optimization_demo():
    """Demonstrate the prompt optimization capabilities."""
    logger.info("Starting prompt optimization demo with real tools", emoji_key="start")
    
    # Original prompt
    original_prompt = "Tell me about quantum computing and its applications."
    
    # Target models for optimization
    target_models = [
        "gpt-4o-mini",
        "gemini-2.0-flash-lite",
        "claude-3-5-haiku-latest"
    ]
    
    # Print the original prompt
    logger.info("Original prompt:", emoji_key="info")
    print("\n" + "-" * 80)
    print(f"Original: {original_prompt}")
    print("-" * 80 + "\n")
    
    # Optimize the prompt for each target model
    optimized_prompts = {}
    
    for model in target_models:
        try:
            logger.info(f"Optimizing prompt for {model}...", emoji_key="processing")
            
            # Determine the provider based on model name
            if "gpt" in model:
                provider = Provider.OPENAI.value
            elif "gemini" in model:
                provider = Provider.GEMINI.value
            elif "claude" in model:
                provider = Provider.ANTHROPIC.value
            else:
                provider = Provider.OPENAI.value
            
            # Call the optimize_prompt tool
            optimize_result = await mcp.call_tool("optimize_prompt", {
                "prompt": original_prompt,
                "target_model": model,
                "optimization_type": "general",
                "provider": provider
            })
            
            # Parse the JSON if needed
            result = {}
            if isinstance(optimize_result, list) and len(optimize_result) > 0:
                first_item = optimize_result[0]
                if hasattr(first_item, 'text'):
                    try:
                        result = json.loads(first_item.text)
                    except json.JSONDecodeError:
                        result = {"optimized_prompt": first_item.text}
            elif isinstance(optimize_result, dict):
                result = optimize_result
            
            # Store the optimized prompt
            if isinstance(result, dict):
                optimized_prompts[model] = result.get("optimized_prompt", "[Optimization failed] " + original_prompt)
            
        except Exception as e:
            logger.warning(f"Error optimizing for {model}: {str(e)}", emoji_key="warning")
            # Fallback to original prompt
            optimized_prompts[model] = f"[Optimization failed] {original_prompt}"
    
    # Show optimized prompts
    logger.info("Optimized prompts by target model:", emoji_key="processing")
    
    for i, model in enumerate(target_models, 1):
        print(f"\n{i}. Optimized for {model}:")
        print(optimized_prompts.get(model, f"[Optimization failed] {original_prompt}"))
    
    logger.success("Prompt optimization demo completed", emoji_key="success")
    return optimized_prompts


async def main():
    """Run workflow and delegation examples."""
    try:
        # Wait a moment for initialization
        await asyncio.sleep(0.1)
        
        print("\n")
        
        # Run analyze task demo
        analysis_result = await run_analyze_task_demo()
        
        print("\n")
        
        # Run delegate task demo
        delegation_result = await run_delegate_task_demo()
        
        print("\n")
        
        # Run workflow execution demo
        workflow_result = await run_workflow_demo()
        
        print("\n")
        
        # Run prompt optimization demo
        optimization_result = await run_prompt_optimization_demo()
        
        # Summarize results
        if any([analysis_result, delegation_result, workflow_result, optimization_result]):
            logger.success("Workflow delegation examples completed successfully", emoji_key="success")
        else:
            logger.warning("Some examples did not complete successfully", emoji_key="warning")
        
    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 