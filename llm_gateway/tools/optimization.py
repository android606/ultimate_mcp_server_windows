"""Cost optimization tools for LLM Gateway."""
import asyncio
import time
from typing import Any, Dict, List, Optional

from llm_gateway.constants import COST_PER_MILLION_TOKENS, Provider
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class OptimizationTools:
    """Cost optimization tools for LLM Gateway."""
    
    def __init__(self, mcp_server):
        """Initialize the optimization tools.
        
        Args:
            mcp_server: MCP server instance
        """
        self.mcp = mcp_server
        self.logger = logger
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register cost optimization tools with MCP server."""
        
        @self.mcp.tool()
        async def estimate_cost(
            prompt: str,
            model: str,
            max_tokens: Optional[int] = None,
            include_output: bool = True,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Estimate the cost of a request without executing it.
            
            Args:
                prompt: Text prompt to send to the model
                model: Model name to use
                max_tokens: Maximum output tokens (if not provided, estimated from prompt)
                include_output: Whether to include output tokens in the estimate
                
            Returns:
                Dictionary containing cost estimate and token breakdown
            """
            # Estimate input tokens
            # Simple approximation: 1 token ≈ 4 characters
            input_tokens = len(prompt) // 4
            
            # Estimate output tokens if not provided
            if max_tokens is None:
                if include_output:
                    # Default estimate: output is about 50% of input size
                    output_tokens = input_tokens // 2
                else:
                    output_tokens = 0
            else:
                output_tokens = max_tokens if include_output else 0
            
            # Get cost rates for model
            cost_data = COST_PER_MILLION_TOKENS.get(model)
            if not cost_data:
                return {
                    "error": f"Unknown model: {model}",
                    "cost": 0.0,
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens
                    }
                }
            
            # Calculate cost
            input_cost = (input_tokens / 1_000_000) * cost_data["input"]
            output_cost = (output_tokens / 1_000_000) * cost_data["output"]
            total_cost = input_cost + output_cost
            
            # Return estimate
            return {
                "cost": total_cost,
                "breakdown": {
                    "input_cost": input_cost,
                    "output_cost": output_cost
                },
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "rate": {
                    "input": cost_data["input"],
                    "output": cost_data["output"]
                },
                "model": model,
                "is_estimate": True
            }
        
        @self.mcp.tool()
        async def compare_models(
            prompt: str,
            models: List[str],
            max_tokens: Optional[int] = None,
            include_output: bool = True,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Compare the cost of running a prompt across multiple models.
            
            Args:
                prompt: Text prompt to send to the models
                models: List of model names to compare
                max_tokens: Maximum output tokens (if not provided, estimated from prompt)
                include_output: Whether to include output tokens in the estimate
                
            Returns:
                Dictionary containing cost comparison
            """
            results = {}
            
            # Get estimates for each model
            for model in models:
                estimate = await estimate_cost(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    include_output=include_output
                )
                
                if "error" in estimate:
                    results[model] = {"error": estimate["error"]}
                else:
                    results[model] = {
                        "cost": estimate["cost"],
                        "tokens": estimate["tokens"],
                    }
            
            # Sort models by cost (ascending)
            sorted_models = sorted(
                [(m, r["cost"]) for m, r in results.items() if "error" not in r],
                key=lambda x: x[1]
            )
            
            # Return comparison
            return {
                "models": results,
                "ranking": [m for m, _ in sorted_models],
                "cheapest": sorted_models[0][0] if sorted_models else None,
                "most_expensive": sorted_models[-1][0] if sorted_models else None,
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
            }
        
        @self.mcp.tool()
        async def recommend_model(
            task_type: str,
            expected_input_length: int,
            expected_output_length: Optional[int] = None,
            required_capabilities: Optional[List[str]] = None,
            max_cost: Optional[float] = None,
            priority: str = "balanced",
            ctx=None
        ) -> Dict[str, Any]:
            """
            Recommend the most suitable model for a given task balancing cost and capability.
            
            Args:
                task_type: Type of task (summarization, generation, extraction, etc.)
                expected_input_length: Estimated input length in characters
                expected_output_length: Estimated output length in characters (if None, derived from input)
                required_capabilities: List of required model capabilities
                max_cost: Maximum acceptable cost for the request
                priority: Optimization priority (cost, quality, speed, or balanced)
                
            Returns:
                Dictionary containing recommended models
            """
            # Convert input length to tokens
            input_tokens = expected_input_length // 4
            
            # Estimate output tokens if not provided
            if expected_output_length is None:
                if task_type == "summarization":
                    output_tokens = input_tokens // 3  # Summaries are typically shorter
                elif task_type == "extraction":
                    output_tokens = input_tokens // 4  # Extraction is typically concise
                elif task_type == "generation":
                    output_tokens = input_tokens * 2  # Generation often creates more content
                else:
                    output_tokens = input_tokens  # Default 1:1 ratio
            else:
                output_tokens = expected_output_length // 4
            
            # Define capability requirements
            required_capabilities = required_capabilities or []
            
            # Model capability mapping (simplified)
            model_capabilities = {
                # OpenAI models
                "gpt-4o": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
                "gpt-4o-mini": ["reasoning", "coding", "knowledge", "instruction-following"],
                "gpt-3.5-turbo": ["coding", "knowledge", "instruction-following"],
                
                # Claude models
                "claude-3-opus-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
                "claude-3-sonnet-20240229": ["reasoning", "coding", "knowledge", "instruction-following"],
                "claude-3-haiku-20240307": ["knowledge", "instruction-following"],
                "claude-3-5-sonnet-20240620": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
                "claude-3-5-haiku-latest": ["reasoning", "knowledge", "instruction-following"],
                
                # Other models
                "deepseek-chat": ["coding", "knowledge", "instruction-following"],
                "deepseek-reasoner": ["reasoning", "math", "instruction-following"],
                "gemini-2.0-flash-lite": ["knowledge", "instruction-following"],
                "gemini-2.0-flash": ["knowledge", "instruction-following"],
                "gemini-2.0-pro": ["reasoning", "knowledge", "instruction-following", "math"],
            }
            
            # Model latency characteristics (lower is faster)
            model_speed = {
                # OpenAI models
                "gpt-4o": 3,
                "gpt-4o-mini": 2,
                "gpt-3.5-turbo": 1,
                
                # Claude models
                "claude-3-opus-20240229": 5,
                "claude-3-sonnet-20240229": 3,
                "claude-3-haiku-20240307": 2,
                "claude-3-5-sonnet-20240620": 3,
                "claude-3-5-haiku-latest": 2,
                
                # Other models
                "deepseek-chat": 2,
                "deepseek-reasoner": 3,
                "gemini-2.0-flash-lite": 1,
                "gemini-2.0-flash": 2,
                "gemini-2.0-pro": 3,
            }
            
            # Model quality characteristics (higher is better)
            model_quality = {
                # OpenAI models
                "gpt-4o": 9,
                "gpt-4o-mini": 7,
                "gpt-3.5-turbo": 5,
                
                # Claude models
                "claude-3-opus-20240229": 9,
                "claude-3-sonnet-20240229": 8,
                "claude-3-haiku-20240307": 6,
                "claude-3-5-sonnet-20240620": 9,
                "claude-3-5-haiku-latest": 7,
                
                # Other models
                "deepseek-chat": 6,
                "deepseek-reasoner": 7,
                "gemini-2.0-flash-lite": 5,
                "gemini-2.0-flash": 6,
                "gemini-2.0-pro": 8,
            }
            
            # Filter models by required capabilities
            qualified_models = []
            for model, capabilities in model_capabilities.items():
                if all(cap in capabilities for cap in required_capabilities):
                    # Calculate estimated cost
                    cost_data = COST_PER_MILLION_TOKENS.get(model)
                    if cost_data:
                        input_cost = (input_tokens / 1_000_000) * cost_data["input"]
                        output_cost = (output_tokens / 1_000_000) * cost_data["output"]
                        total_cost = input_cost + output_cost
                        
                        # Filter by max cost if specified
                        if max_cost is None or total_cost <= max_cost:
                            qualified_models.append({
                                "model": model,
                                "cost": total_cost,
                                "speed": model_speed.get(model, 3),  # Default to medium speed
                                "quality": model_quality.get(model, 5),  # Default to medium quality
                                "tokens": {
                                    "input": input_tokens,
                                    "output": output_tokens,
                                    "total": input_tokens + output_tokens
                                }
                            })
            
            # Sort models based on priority
            if priority == "cost":
                sorted_models = sorted(qualified_models, key=lambda m: m["cost"])
            elif priority == "quality":
                sorted_models = sorted(qualified_models, key=lambda m: -m["quality"])
            elif priority == "speed":
                sorted_models = sorted(qualified_models, key=lambda m: m["speed"])
            else:  # balanced
                # Create a composite score: (quality * 0.5) - (cost * 10) - (speed * 0.3)
                for model in qualified_models:
                    model["score"] = (model["quality"] * 0.5) - (model["cost"] * 10) - (model["speed"] * 0.3)
                sorted_models = sorted(qualified_models, key=lambda m: -m["score"])
            
            # Return recommendations
            return {
                "recommendations": sorted_models[:3],  # Top 3 models
                "all_qualified_models": sorted_models,
                "task_type": task_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "priority": priority,
                "required_capabilities": required_capabilities,
            }
        
        @self.mcp.tool()
        async def execute_optimized_workflow(
            documents: List[str],
            workflow: List[Dict[str, Any]],
            max_concurrency: int = 5,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Execute a multi-stage workflow optimized for cost and performance.
            
            Args:
                documents: List of document texts to process
                workflow: Workflow definition as a list of operation stages
                max_concurrency: Maximum concurrent documents to process
                
            Returns:
                Dictionary containing workflow results
            """
            start_time = time.time()
            
            if not documents:
                return {
                    "error": "No documents provided",
                    "results": [],
                    "processing_time": 0.0,
                }
                
            if not workflow:
                return {
                    "error": "No workflow stages defined",
                    "results": [],
                    "processing_time": 0.0,
                }
            
            # Initialize workflow context
            workflow_context = {
                "documents": documents,
                "results": [{} for _ in range(len(documents))],
                "errors": [[] for _ in range(len(documents))],
                "stage_metrics": [],
                "total_cost": 0.0,
            }
            
            # Process each stage of the workflow
            for stage_index, stage in enumerate(workflow):
                stage_name = stage.get("name", f"Stage {stage_index+1}")
                stage_operation = stage.get("operation")
                
                if not stage_operation:
                    logger.warning(
                        f"Skipping workflow stage {stage_index} - no operation defined",
                        emoji_key="warning"
                    )
                    continue
                
                # Log stage start
                logger.info(
                    f"Starting workflow stage: {stage_name}",
                    emoji_key="processing"
                )
                
                stage_start_time = time.time()
                stage_costs = []
                
                # Create tasks for documents
                document_tasks = []
                semaphore = asyncio.Semaphore(max_concurrency)
                
                async def process_document_in_stage(doc_index, stage_def, sem, costs):
                    async with sem:
                        doc_result = workflow_context["results"][doc_index]
                        
                        # Get input data for this stage
                        input_from = stage_def.get("input_from", "original")
                        if input_from == "original":
                            input_data = documents[doc_index]
                        elif input_from in doc_result:
                            input_data = doc_result[input_from]
                        else:
                            workflow_context["errors"][doc_index].append(
                                f"Input '{input_from}' not found for stage '{stage_name}'"
                            )
                            return None
                        
                        # Get operation parameters
                        operation = stage_def["operation"]
                        provider = stage_def.get("provider", Provider.OPENAI.value)
                        model = stage_def.get("model")
                        operation_params = stage_def.get("params", {})
                        
                        try:
                            # Import tool functions dynamically to avoid circular imports
                            from llm_gateway.tools.document import (
                                chunk_document,
                                extract_entities,
                                generate_qa_pairs,
                                summarize_document,
                            )
                            
                            # Execute operation
                            if operation == "chunk":
                                result = await chunk_document(
                                    document=input_data,
                                    **operation_params
                                )
                            elif operation == "summarize":
                                result = await summarize_document(
                                    document=input_data,
                                    provider=provider,
                                    model=model,
                                    **operation_params
                                )
                                # Track cost
                                if "cost" in result:
                                    costs.append(result["cost"])
                            elif operation == "extract_entities":
                                result = await extract_entities(
                                    document=input_data,
                                    provider=provider,
                                    model=model,
                                    **operation_params
                                )
                                # Track cost
                                if "cost" in result:
                                    costs.append(result["cost"])
                            elif operation == "generate_qa":
                                result = await generate_qa_pairs(
                                    document=input_data,
                                    provider=provider,
                                    model=model,
                                    **operation_params
                                )
                                # Track cost
                                if "cost" in result:
                                    costs.append(result["cost"])
                            else:
                                result = {
                                    "error": f"Unsupported operation: {operation}"
                                }
                                workflow_context["errors"][doc_index].append(
                                    f"Unsupported operation: {operation}"
                                )
                            
                            # Store result
                            output_as = stage_def.get("output_as", operation)
                            if isinstance(result, dict):
                                doc_result[output_as] = result
                            else:
                                doc_result[output_as] = {"result": result}
                            
                            return result
                            
                        except Exception as e:
                            error_msg = f"Error in {stage_name}: {str(e)}"
                            workflow_context["errors"][doc_index].append(error_msg)
                            logger.error(
                                f"Error processing document {doc_index} in stage '{stage_name}': {str(e)}",
                                emoji_key="error"
                            )
                            return {"error": error_msg}
                
                # Create tasks for each document
                for doc_index in range(len(documents)):
                    task = process_document_in_stage(doc_index, stage, semaphore, stage_costs)
                    document_tasks.append(task)
                
                # Execute tasks concurrently
                stage_results = await asyncio.gather(*document_tasks)
                
                # Calculate stage metrics
                stage_end_time = time.time()
                stage_duration = stage_end_time - stage_start_time
                stage_total_cost = sum(stage_costs)
                
                # Update workflow context
                workflow_context["total_cost"] += stage_total_cost
                workflow_context["stage_metrics"].append({
                    "stage": stage_name,
                    "duration": stage_duration,
                    "cost": stage_total_cost,
                })
                
                # Log stage completion
                logger.success(
                    f"Completed workflow stage: {stage_name}",
                    emoji_key="success",
                    time=stage_duration,
                    cost=stage_total_cost
                )
            
            # Calculate overall metrics
            processing_time = time.time() - start_time
            total_errors = sum(1 for errors in workflow_context["errors"] if errors)
            
            # Log workflow completion
            logger.success(
                f"Workflow completed: {len(workflow)} stages, {len(documents)} documents, " +
                f"{total_errors} documents with errors",
                emoji_key="success",
                time=processing_time,
                cost=workflow_context["total_cost"]
            )
            
            # Return workflow results
            return {
                "results": workflow_context["results"],
                "errors": workflow_context["errors"],
                "stage_metrics": workflow_context["stage_metrics"],
                "total_documents": len(documents),
                "documents_with_errors": total_errors,
                "total_cost": workflow_context["total_cost"],
                "processing_time": processing_time,
            }