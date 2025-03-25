"""Meta-tools for intelligent task delegation and workflow management."""
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from llm_gateway.constants import COST_PER_MILLION_TOKENS, Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.tools.base import BaseTool, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class MetaTools(BaseTool):
    """Meta-tools for intelligent task delegation and workflow management."""
    
    tool_name = "meta"
    description = "Tools for intelligent task delegation and workflow management."
    
    def __init__(self, mcp_server):
        """Initialize the meta-tools.
        
        Args:
            mcp_server: MCP server instance
        """
        super().__init__(mcp_server)
        
    def _register_tools(self):
        """Register meta-tools with MCP server."""
        
        @self.mcp.tool()
        @with_tool_metrics
        async def analyze_task(
            task_description: str,
            available_providers: Optional[List[str]] = None,
            analyze_features: bool = True,
            analyze_cost: bool = True,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Analyze a task and recommend the best provider and model based on requirements.
            
            Args:
                task_description: Description of the task to analyze
                available_providers: Optional list of available providers to consider
                analyze_features: Whether to analyze required features
                analyze_cost: Whether to analyze estimated cost
                
            Returns:
                Dictionary containing task analysis and recommendations
            """
            start_time = time.time()
            
            # Get available providers if not specified
            if available_providers is None:
                available_providers = [p.value for p in Provider]
            
            # Analyze task type
            task_type = self._analyze_task_type(task_description)
            
            # Analyze required features
            required_features = []
            features_explanation = ""
            
            if analyze_features:
                required_features, features_explanation = self._analyze_required_features(task_description)
            
            # Get provider options
            provider_options = await self._get_provider_options(
                task_type=task_type,
                required_features=required_features,
                available_providers=available_providers
            )
            
            # Analyze cost if requested
            cost_analysis = {}
            if analyze_cost:
                cost_analysis = self._analyze_cost(
                    task_description=task_description,
                    provider_options=provider_options,
                    task_type=task_type
                )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                provider_options=provider_options,
                cost_analysis=cost_analysis,
                task_type=task_type
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log completion
            logger.success(
                f"Task analysis completed: {task_type}",
                emoji_key="meta",
                time=processing_time
            )
            
            return {
                "task_type": task_type,
                "required_features": required_features,
                "features_explanation": features_explanation,
                "providers": provider_options,
                "cost_analysis": cost_analysis,
                "recommendations": recommendations,
                "processing_time": processing_time,
            }
        
        @self.mcp.tool()
        @with_tool_metrics
        async def delegate_task(
            task_description: str,
            prompt: str,
            optimization_criteria: str = "balanced",
            available_providers: Optional[List[str]] = None,
            max_cost: Optional[float] = None,
            model_preferences: Optional[Dict[str, Any]] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Delegate a task to the most appropriate provider based on criteria.
            
            Args:
                task_description: Description of the task
                prompt: Prompt or text to process
                optimization_criteria: Criteria for optimization (balanced, cost, quality, speed)
                available_providers: Optional list of available providers to consider
                max_cost: Maximum acceptable cost for the task
                model_preferences: Optional model preferences
                
            Returns:
                Dictionary containing the delegated task results
            """
            start_time = time.time()
            
            # Get available providers if not specified
            if available_providers is None:
                available_providers = [p.value for p in Provider]
            
            # Set default model preferences
            model_preferences = model_preferences or {}
            
            # Analyze the task
            analysis = await analyze_task(
                task_description=task_description,
                available_providers=available_providers,
                analyze_features=True,
                analyze_cost=True
            )
            
            # Get recommended provider and model
            provider, model = self._select_provider_and_model(
                analysis=analysis,
                optimization_criteria=optimization_criteria,
                max_cost=max_cost,
                model_preferences=model_preferences
            )
            
            # Log delegation decision
            logger.info(
                f"Delegating task to {provider}/{model}",
                emoji_key="meta",
                task_type=analysis["task_type"],
                criteria=optimization_criteria
            )
            
            # Execute the task with selected provider
            try:
                # Import completion tool
                
                # Get provider instance
                provider_instance = get_provider(provider)
                
                # Determine if we need system prompt based on task type
                system_prompt = None
                additional_params = {}
                
                if analysis["task_type"] == "summarization":
                    system_prompt = "Summarize the following text concisely while preserving the key information."
                elif analysis["task_type"] == "extraction":
                    system_prompt = "Extract and structure the requested information from the text."
                elif analysis["task_type"] == "classification":
                    system_prompt = "Classify the text according to the specified criteria."
                elif analysis["task_type"] == "translation":
                    system_prompt = "Translate the text accurately while preserving meaning and tone."
                elif analysis["task_type"] == "creative_writing":
                    system_prompt = "Create original, high-quality content based on the prompt."
                
                # Generate completion with selected provider
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    temperature=0.7,  # Default temperature
                    system=system_prompt,
                    **additional_params
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log success
                logger.success(
                    f"Task delegation successful: {provider}/{model}",
                    emoji_key="success",
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                return {
                    "text": result.text,
                    "provider": provider,
                    "model": model,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "task_type": analysis["task_type"],
                    "optimization_criteria": optimization_criteria,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Task delegation failed: {str(e)}",
                    emoji_key="error",
                    provider=provider,
                    model=model,
                    time=processing_time
                )
                
                # Try fallback provider if available
                fallback_provider = self._get_fallback_provider(provider, available_providers)
                if fallback_provider:
                    logger.info(
                        f"Trying fallback provider: {fallback_provider}",
                        emoji_key="warning"
                    )
                    
                    try:
                        # Get fallback provider instance
                        fallback_instance = get_provider(fallback_provider)
                        
                        # Generate completion with fallback provider
                        result = await fallback_instance.generate_completion(
                            prompt=prompt,
                            model=None,  # Use default model
                            temperature=0.7,
                            system=system_prompt,
                            **additional_params
                        )
                        
                        # Calculate processing time
                        processing_time = time.time() - start_time
                        
                        # Log success
                        logger.success(
                            f"Fallback provider successful: {fallback_provider}/{result.model}",
                            emoji_key="success",
                            tokens={
                                "input": result.input_tokens,
                                "output": result.output_tokens
                            },
                            cost=result.cost,
                            time=processing_time
                        )
                        
                        return {
                            "text": result.text,
                            "provider": fallback_provider,
                            "model": result.model,
                            "tokens": {
                                "input": result.input_tokens,
                                "output": result.output_tokens,
                                "total": result.total_tokens,
                            },
                            "cost": result.cost,
                            "task_type": analysis["task_type"],
                            "optimization_criteria": optimization_criteria,
                            "processing_time": processing_time,
                            "used_fallback": True,
                            "original_error": str(e)
                        }
                        
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback provider failed: {str(fallback_error)}",
                            emoji_key="error",
                            provider=fallback_provider
                        )
                
                # If we get here, both primary and fallback failed
                return {
                    "error": f"Task delegation failed: {str(e)}",
                    "provider": provider,
                    "model": model,
                    "task_type": analysis["task_type"],
                    "processing_time": processing_time,
                }
        
        @self.mcp.tool()
        @with_tool_metrics
        async def execute_workflow(
            workflow_steps: List[Dict[str, Any]],
            initial_input: str,
            max_concurrency: int = 3,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Execute a multi-step workflow with task delegation.
            
            Args:
                workflow_steps: List of workflow step definitions
                initial_input: Initial input text for the workflow
                max_concurrency: Maximum number of concurrent steps for parallel execution
                
            Returns:
                Dictionary containing workflow execution results
            """
            start_time = time.time()
            total_cost = 0.0
            
            # Validate workflow
            if not workflow_steps:
                return {
                    "error": "No workflow steps provided",
                    "results": {},
                }
            
            # Initialize workflow state
            workflow_state = {
                "input": initial_input,
                "steps": {},
                "current_step": 0,
                "outputs": {},
                "errors": {},
            }
            
            # Determine execution mode (sequential or parallel)
            sequential_execution = True
            for step in workflow_steps:
                if "depends_on" in step:
                    # If any step has dependencies, we need sequential execution
                    sequential_execution = True
                    break
            
            # Execute workflow
            if sequential_execution:
                # Sequential execution
                for i, step in enumerate(workflow_steps):
                    step_id = step.get("id", f"step_{i}")
                    step_name = step.get("name", f"Step {i}")
                    
                    # Update workflow state
                    workflow_state["current_step"] = i
                    
                    # Get input for this step
                    if "input_from" in step and step["input_from"] in workflow_state["outputs"]:
                        step_input = workflow_state["outputs"][step["input_from"]]
                    else:
                        step_input = workflow_state["input"]
                    
                    # Log step start
                    logger.info(
                        f"Starting workflow step {i+1}/{len(workflow_steps)}: {step_name}",
                        emoji_key="meta"
                    )
                    
                    try:
                        # Execute step
                        step_result = await self._execute_workflow_step(
                            step=step,
                            input_text=step_input,
                            workflow_state=workflow_state
                        )
                        
                        # Update workflow state
                        workflow_state["steps"][step_id] = step_result
                        workflow_state["outputs"][step_id] = step_result.get("text", "")
                        
                        # Update total cost
                        if "cost" in step_result:
                            total_cost += step_result["cost"]
                        
                        # Log step completion
                        logger.success(
                            f"Completed workflow step: {step_name}",
                            emoji_key="success",
                            cost=step_result.get("cost")
                        )
                        
                    except Exception as e:
                        # Log error
                        logger.error(
                            f"Workflow step failed: {step_name}: {str(e)}",
                            emoji_key="error"
                        )
                        
                        # Update workflow state
                        workflow_state["errors"][step_id] = str(e)
                        
                        # Check if step is critical
                        if step.get("critical", False):
                            # Stop workflow on critical step failure
                            break
            else:
                # Parallel execution
                # Group steps by dependencies
                step_groups = self._group_steps_by_dependencies(workflow_steps)
                
                # Execute step groups in sequence
                for _group_idx, step_group in enumerate(step_groups):
                    # Create tasks for each step in group
                    tasks = []
                    semaphore = asyncio.Semaphore(max_concurrency)
                    
                    async def process_step(step_idx, step, semaphore):
                        async with semaphore:
                            step_id = step.get("id", f"step_{step_idx}")
                            step_name = step.get("name", f"Step {step_idx}")
                            
                            # Get input for this step
                            if "input_from" in step and step["input_from"] in workflow_state["outputs"]:
                                step_input = workflow_state["outputs"][step["input_from"]]
                            else:
                                step_input = workflow_state["input"]
                            
                            # Log step start
                            logger.info(
                                f"Starting workflow step {step_idx+1}/{len(workflow_steps)}: {step_name}",
                                emoji_key="meta"
                            )
                            
                            try:
                                # Execute step
                                return await self._execute_workflow_step(
                                    step=step,
                                    input_text=step_input,
                                    workflow_state=workflow_state
                                ), step_id, step_name
                            except Exception as e:
                                # Log error
                                logger.error(
                                    f"Workflow step failed: {step_name}: {str(e)}",
                                    emoji_key="error"
                                )
                                
                                # Return error
                                return {"error": str(e)}, step_id, step_name
                    
                    # Create tasks for all steps in group
                    for step_idx, step in step_group:
                        tasks.append(process_step(step_idx, step, semaphore))
                    
                    # Execute group concurrently
                    group_results = await asyncio.gather(*tasks)
                    
                    # Process results
                    for step_result, step_id, step_name in group_results:
                        if "error" in step_result:
                            # Update workflow state with error
                            workflow_state["errors"][step_id] = step_result["error"]
                            
                            # Check if step is critical
                            step_idx = next((i for i, (idx, _) in enumerate(step_group) if idx == int(step_id.split("_")[1])), None)
                            if step_idx is not None and workflow_steps[step_idx].get("critical", False):
                                # Stop workflow on critical step failure
                                break
                        else:
                            # Update workflow state with result
                            workflow_state["steps"][step_id] = step_result
                            workflow_state["outputs"][step_id] = step_result.get("text", "")
                            
                            # Update total cost
                            if "cost" in step_result:
                                total_cost += step_result["cost"]
                            
                            # Log step completion
                            logger.success(
                                f"Completed workflow step: {step_name}",
                                emoji_key="success",
                                cost=step_result.get("cost")
                            )
            
            # Calculate final output
            final_output = None
            if workflow_steps:
                last_step = workflow_steps[-1]
                last_step_id = last_step.get("id", f"step_{len(workflow_steps)-1}")
                
                if last_step_id in workflow_state["outputs"]:
                    final_output = workflow_state["outputs"][last_step_id]
                elif workflow_state["outputs"]:
                    # Fall back to any output
                    final_output = next(iter(workflow_state["outputs"].values()))
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log workflow completion
            logger.success(
                f"Workflow execution completed ({len(workflow_steps)} steps)",
                emoji_key="success",
                time=processing_time,
                cost=total_cost
            )
            
            return {
                "success": len(workflow_state["errors"]) == 0,
                "outputs": workflow_state["outputs"],
                "final_output": final_output,
                "errors": workflow_state["errors"],
                "total_steps": len(workflow_steps),
                "completed_steps": len(workflow_state["steps"]),
                "total_cost": total_cost,
                "processing_time": processing_time,
            }
        
        @self.mcp.tool()
        @with_tool_metrics
        async def quality_check(
            text: str,
            original_task: str,
            quality_criteria: Optional[List[str]] = None,
            provider: str = Provider.ANTHROPIC.value,
            model: Optional[str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Perform a quality check on generated content.
            
            Args:
                text: Text to check
                original_task: Original task description
                quality_criteria: List of quality criteria to check
                provider: Provider to use for quality check
                model: Model to use for quality check
                
            Returns:
                Dictionary containing quality check results
            """
            start_time = time.time()
            
            # Set default quality criteria if not provided
            if not quality_criteria:
                quality_criteria = [
                    "accuracy",
                    "completeness",
                    "coherence",
                    "relevance",
                    "grammar",
                ]
            
            # Create prompt for quality check
            criteria_text = "\n".join([f"- {criterion}" for criterion in quality_criteria])
            prompt = f"""Evaluate the quality of the following text based on these criteria:
{criteria_text}

Original task: {original_task}

Text to evaluate:
{text}

For each criterion, provide a score from 1-10 and a brief explanation.
Also provide an overall score and summary of strengths and weaknesses.
Your response MUST be in valid JSON format with the following structure:
{{
  "criteria": {{
    "criterion1": {{
      "score": 8,
      "comments": "Explanation here"
    }},
    ...
  }},
  "overall_score": 7.5,
  "summary": "Overall assessment here",
  "strengths": ["Strength 1", "Strength 2"],
  "weaknesses": ["Weakness 1", "Weakness 2"],
  "recommendations": ["Recommendation 1", "Recommendation 2"]
}}"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            try:
                # Generate quality check
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    temperature=0.3,  # Low temperature for consistent evaluation
                )
                
                # Parse JSON response
                try:
                    evaluation = json.loads(result.text)
                except json.JSONDecodeError:
                    # Try to extract JSON with regex
                    import re
                    json_match = re.search(r'(\{[\s\S]*\})', result.text)
                    if json_match:
                        try:
                            evaluation = json.loads(json_match.group(1))
                        except Exception:
                            evaluation = {
                                "error": "Failed to parse evaluation JSON",
                                "raw_text": result.text
                            }
                    else:
                        evaluation = {
                            "error": "Failed to parse evaluation JSON",
                            "raw_text": result.text
                        }
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log success
                logger.success(
                    f"Quality check completed (score: {evaluation.get('overall_score', 'N/A')})",
                    emoji_key="success",
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                return {
                    "evaluation": evaluation,
                    "provider": provider,
                    "model": result.model,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Quality check failed: {str(e)}",
                    emoji_key="error",
                    provider=provider,
                    model=model,
                    time=processing_time
                )
                
                return {
                    "error": f"Quality check failed: {str(e)}",
                    "provider": provider,
                    "model": model,
                    "processing_time": processing_time,
                }
        
        @self.mcp.tool()
        @with_tool_metrics
        async def optimize_prompt(
            prompt: str,
            target_model: str,
            optimization_type: str = "general",
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Optimize a prompt for a specific model or task.
            
            Args:
                prompt: Original prompt to optimize
                target_model: Target model for optimization
                optimization_type: Type of optimization (general, factual, creative, coding)
                provider: Provider to use for optimization
                model: Model to use for optimization
                
            Returns:
                Dictionary containing optimized prompt
            """
            start_time = time.time()
            
            # Create prompt optimization instructions based on type
            if optimization_type == "factual":
                instructions = """
- Improve clarity and specificity for factual queries
- Add constraints to prevent hallucination
- Structure the prompt to encourage step-by-step reasoning
- Include explicit instructions to cite sources if needed
"""
            elif optimization_type == "creative":
                instructions = """
- Enhance creative elements while maintaining coherence
- Add specific stylistic guidelines if appropriate
- Include examples of desired tone and style
- Structure for creative exploration while maintaining constraints
"""
            elif optimization_type == "coding":
                instructions = """
- Add specific requirements for code format and style
- Include detailed specifications for inputs and outputs
- Request documentation and error handling
- Specify any libraries or frameworks to use
"""
            else:  # general
                instructions = """
- Improve clarity and remove ambiguity
- Add specificity and constraints
- Structure complex requests into clear steps
- Use concise and precise language
"""
            
            # Create meta-prompt for optimization
            meta_prompt = f"""Optimize the following prompt for the {target_model} model.

Optimization guidelines:
{instructions}

Original prompt:
{prompt}

Provide the optimized prompt along with a brief explanation of the changes made and how they improve effectiveness for {target_model}.
Format your response as JSON:
{{
  "optimized_prompt": "The improved prompt goes here",
  "explanation": "Explanation of changes and improvements",
  "key_improvements": ["Improvement 1", "Improvement 2", ...]
}}"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            try:
                # Generate optimized prompt
                result = await provider_instance.generate_completion(
                    prompt=meta_prompt,
                    model=model,
                    temperature=0.4,  # Moderate temperature for creativity with consistency
                )
                
                # Parse JSON response
                try:
                    optimization = json.loads(result.text)
                except json.JSONDecodeError:
                    # Try to extract JSON with regex
                    import re
                    json_match = re.search(r'(\{[\s\S]*\})', result.text)
                    if json_match:
                        try:
                            optimization = json.loads(json_match.group(1))
                        except Exception:
                            optimization = {
                                "error": "Failed to parse optimization JSON",
                                "raw_text": result.text
                            }
                    else:
                        optimization = {
                            "error": "Failed to parse optimization JSON",
                            "raw_text": result.text
                        }
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log success
                logger.success(
                    f"Prompt optimization completed for {target_model}",
                    emoji_key="success",
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                return {
                    "optimization": optimization,
                    "provider": provider,
                    "model": result.model,
                    "target_model": target_model,
                    "optimization_type": optimization_type,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Prompt optimization failed: {str(e)}",
                    emoji_key="error",
                    provider=provider,
                    model=model,
                    time=processing_time
                )
                
                return {
                    "error": f"Prompt optimization failed: {str(e)}",
                    "provider": provider,
                    "model": model,
                    "target_model": target_model,
                    "processing_time": processing_time,
                }
    
    def _analyze_task_type(self, task_description: str) -> str:
        """Analyze task description to determine task type.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Task type identifier
        """
        # Convert to lowercase for matching
        task_lower = task_description.lower()
        
        # Check for task type indicators
        if any(kw in task_lower for kw in ["summarize", "summary", "summarization", "summarize", "tldr"]):
            return "summarization"
            
        if any(kw in task_lower for kw in ["extract", "identify", "find all", "list the", "parse"]):
            return "extraction"
            
        if any(kw in task_lower for kw in ["classify", "categorize", "what type of", "which category"]):
            return "classification"
            
        if any(kw in task_lower for kw in ["translate", "translation", "convert to language"]):
            return "translation"
            
        if any(kw in task_lower for kw in ["write story", "create article", "generate content", "creative"]):
            return "creative_writing"
            
        if any(kw in task_lower for kw in ["write code", "implement function", "create program", "coding"]):
            return "coding"
            
        if any(kw in task_lower for kw in ["reason", "analyze", "evaluate", "assess", "interpret"]):
            return "reasoning"
            
        if any(kw in task_lower for kw in ["chat", "conversation", "discuss", "talk", "respond"]):
            return "conversation"
            
        # Default to general completion
        return "general"
    
    def _analyze_required_features(self, task_description: str) -> Tuple[List[str], str]:
        """Analyze task description to determine required features.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Tuple of (list of required features, explanation of requirements)
        """
        # Convert to lowercase for matching
        task_lower = task_description.lower()
        
        required_features = []
        explanation = ""
        
        # Check for specific requirements
        if any(kw in task_lower for kw in ["reason", "reasoning", "analyze", "complex", "nuanced"]):
            required_features.append("reasoning")
            explanation += "Task requires reasoning capabilities for analysis and complex logic. "
        
        if any(kw in task_lower for kw in ["code", "function", "programming", "implement", "algorithm"]):
            required_features.append("coding")
            explanation += "Task involves code generation or programming knowledge. "
        
        if any(kw in task_lower for kw in ["math", "calculate", "computation", "formula", "equation"]):
            required_features.append("math")
            explanation += "Task requires mathematical computation or understanding. "
        
        if any(kw in task_lower for kw in ["knowledge", "facts", "information", "domain"]):
            required_features.append("knowledge")
            explanation += "Task requires factual knowledge or domain expertise. "
        
        if any(kw in task_lower for kw in ["instruction", "specific format", "follow", "adhere", "precise"]):
            required_features.append("instruction-following")
            explanation += "Task requires precise instruction following or specific output formatting. "
        
        if any(kw in task_lower for kw in ["creative", "imagination", "original", "innovative", "novel"]):
            required_features.append("creativity")
            explanation += "Task requires creative or original thinking. "
        
        if any(kw in task_lower for kw in ["multi-step", "complex", "sophisticated", "difficult"]):
            required_features.append("complex-reasoning")
            explanation += "Task involves multi-step or complex reasoning. "
        
        # Default to instruction following if no specific features identified
        if not required_features:
            required_features.append("instruction-following")
            explanation = "Task requires basic instruction following."
        
        return required_features, explanation
    
    async def _get_provider_options(
        self,
        task_type: str,
        required_features: List[str],
        available_providers: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get provider options based on task type and required features.
        
        Args:
            task_type: Task type identifier
            required_features: List of required features
            available_providers: List of available providers
            
        Returns:
            Dictionary of provider options by provider
        """
        provider_options = {}
        
        # Model capability mapping
        model_capabilities = {
            # OpenAI models
            "gpt-4o": ["reasoning", "coding", "knowledge", "instruction-following", "math", "creativity", "complex-reasoning"],
            "gpt-4o-mini": ["reasoning", "coding", "knowledge", "instruction-following", "creativity"],
            "gpt-3.5-turbo": ["coding", "knowledge", "instruction-following"],
            
            # Claude models
            "claude-3-opus-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "math", "creativity", "complex-reasoning"],
            "claude-3-sonnet-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "creativity"],
            "claude-3-haiku-20240307": ["knowledge", "instruction-following"],
            "claude-3-5-sonnet-20240620": ["reasoning", "coding", "knowledge", "instruction-following", "math", "creativity", "complex-reasoning"],
            "claude-3-5-haiku-latest": ["reasoning", "knowledge", "instruction-following", "creativity"],
            
            # Other models
            "deepseek-chat": ["coding", "knowledge", "instruction-following"],
            "deepseek-reasoner": ["reasoning", "math", "instruction-following", "complex-reasoning"],
            "gemini-2.0-flash-lite": ["knowledge", "instruction-following"],
            "gemini-2.0-flash": ["knowledge", "instruction-following", "creativity"],
            "gemini-2.0-pro": ["reasoning", "knowledge", "instruction-following", "math", "creativity"],
        }
        
        # Get suitable models for each provider
        for provider_name in available_providers:
            # Skip providers we don't have capability data for
            if provider_name not in [Provider.OPENAI.value, Provider.ANTHROPIC.value, Provider.DEEPSEEK.value, Provider.GEMINI.value]:
                continue
                
            # Get provider instance
            try:
                provider = get_provider(provider_name)
                
                # Get available models
                models = await provider.list_models()
                
                # Filter models based on required features
                suitable_models = []
                
                for model_info in models:
                    model_id = model_info["id"]
                    
                    # Check if model is in capability mapping
                    if model_id in model_capabilities:
                        # Check if model has all required features
                        if all(feat in model_capabilities[model_id] for feat in required_features):
                            suitable_models.append({
                                "id": model_id,
                                "provider": provider_name,
                                "description": model_info.get("description", ""),
                                "capabilities": model_capabilities[model_id],
                            })
                
                # Add to options if any suitable models found
                if suitable_models:
                    provider_options[provider_name] = suitable_models
                    
            except Exception as e:
                logger.error(
                    f"Failed to get models for provider {provider_name}: {str(e)}",
                    emoji_key="error"
                )
        
        return provider_options
    
    def _analyze_cost(
        self,
        task_description: str,
        provider_options: Dict[str, List[Dict[str, Any]]],
        task_type: str
    ) -> Dict[str, Any]:
        """Analyze cost for task execution.
        
        Args:
            task_description: Description of the task
            provider_options: Provider options
            task_type: Task type identifier
            
        Returns:
            Dictionary of cost analysis
        """
        # Estimate token counts based on task type
        input_tokens = len(task_description.split()) * 1.3  # Rough estimate: words * 1.3
        
        # Estimate output tokens based on task type
        if task_type == "summarization":
            output_tokens = input_tokens * 0.5  # Summaries are typically shorter
        elif task_type == "extraction":
            output_tokens = input_tokens * 0.3  # Extraction is typically concise
        elif task_type == "creative_writing":
            output_tokens = input_tokens * 3.0  # Creative writing often generates more content
        elif task_type == "coding":
            output_tokens = input_tokens * 2.0  # Code generation often produces substantial output
        else:
            output_tokens = input_tokens  # Default 1:1 ratio
        
        # Calculate cost for each model
        cost_estimates = {}
        
        for provider_name, models in provider_options.items():
            provider_costs = []
            
            for model_info in models:
                model_id = model_info["id"]
                
                # Get cost rates for model
                cost_data = COST_PER_MILLION_TOKENS.get(model_id)
                if cost_data:
                    # Calculate cost
                    input_cost = (input_tokens / 1_000_000) * cost_data["input"]
                    output_cost = (output_tokens / 1_000_000) * cost_data["output"]
                    total_cost = input_cost + output_cost
                    
                    provider_costs.append({
                        "model": model_id,
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": total_cost,
                    })
            
            if provider_costs:
                # Sort by total cost
                provider_costs.sort(key=lambda x: x["total_cost"])
                cost_estimates[provider_name] = provider_costs
        
        # Generate overall cost summary
        cost_summary = {
            "lowest_cost": None,
            "highest_cost": None,
            "providers": cost_estimates,
        }
        
        # Find overall lowest and highest cost
        all_costs = []
        for provider_costs in cost_estimates.values():
            all_costs.extend(provider_costs)
            
        if all_costs:
            all_costs.sort(key=lambda x: x["total_cost"])
            cost_summary["lowest_cost"] = all_costs[0]
            cost_summary["highest_cost"] = all_costs[-1]
        
        return cost_summary
    
    def _generate_recommendations(
        self,
        provider_options: Dict[str, List[Dict[str, Any]]],
        cost_analysis: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """Generate recommendations based on provider options and cost analysis.
        
        Args:
            provider_options: Provider options
            cost_analysis: Cost analysis
            task_type: Task type identifier
            
        Returns:
            Dictionary of recommendations
        """
        recommendations = {
            "lowest_cost": None,
            "best_quality": None,
            "balanced": None,
            "fastest": None,
        }
        
        # Provider quality ratings (subjective)
        provider_quality = {
            "openai": {
                "gpt-4o": 9.5,
                "gpt-4o-mini": 8.5,
                "gpt-3.5-turbo": 7.5,
            },
            "anthropic": {
                "claude-3-opus-20240229": 9.5,
                "claude-3-sonnet-20240229": 9.0,
                "claude-3-haiku-20240307": 8.0,
                "claude-3-5-sonnet-20240620": 9.5,
                "claude-3-5-haiku-latest": 8.5,
            },
            "deepseek": {
                "deepseek-chat": 7.0,
                "deepseek-reasoner": 8.0,
            },
            "gemini": {
                "gemini-2.0-flash-lite": 7.0,
                "gemini-2.0-flash": 8.0,
                "gemini-2.0-pro": 9.0,
            },
        }
        
        # Provider speed ratings (subjective)
        provider_speed = {
            "openai": {
                "gpt-4o": 8.0,
                "gpt-4o-mini": 9.0,
                "gpt-3.5-turbo": 9.5,
            },
            "anthropic": {
                "claude-3-opus-20240229": 6.0,
                "claude-3-sonnet-20240229": 7.0,
                "claude-3-haiku-20240307": 8.5,
                "claude-3-5-sonnet-20240620": 7.0,
                "claude-3-5-haiku-latest": 9.0,
            },
            "deepseek": {
                "deepseek-chat": 8.5,
                "deepseek-reasoner": 7.5,
            },
            "gemini": {
                "gemini-2.0-flash-lite": 9.5,
                "gemini-2.0-flash": 9.0,
                "gemini-2.0-pro": 7.5,
            },
        }
        
        # Provider task specialization
        task_specialization = {
            "summarization": {
                "openai": ["gpt-4o", "gpt-4o-mini"],
                "anthropic": ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
                "gemini": ["gemini-2.0-pro"],
            },
            "extraction": {
                "openai": ["gpt-4o", "gpt-4o-mini"],
                "anthropic": ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
            },
            "classification": {
                "openai": ["gpt-4o", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-haiku-20240307", "claude-3-5-haiku-latest"],
                "gemini": ["gemini-2.0-flash"],
            },
            "translation": {
                "openai": ["gpt-4o", "gpt-3.5-turbo"],
                "deepseek": ["deepseek-chat"],
            },
            "creative_writing": {
                "openai": ["gpt-4o"],
                "anthropic": ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
                "gemini": ["gemini-2.0-pro"],
            },
            "coding": {
                "openai": ["gpt-4o"],
                "anthropic": ["claude-3-opus-20240229"],
                "deepseek": ["deepseek-chat"],
            },
            "reasoning": {
                "openai": ["gpt-4o"],
                "anthropic": ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
                "deepseek": ["deepseek-reasoner"],
            },
        }
        
        # Collect all suitable models
        all_models = []
        for provider, models in provider_options.items():
            for model in models:
                model_id = model["id"]
                quality = provider_quality.get(provider, {}).get(model_id, 7.0)  # Default quality
                speed = provider_speed.get(provider, {}).get(model_id, 7.0)  # Default speed
                
                # Check if model is specialized for task
                is_specialized = (
                    task_type in task_specialization and
                    provider in task_specialization[task_type] and
                    model_id in task_specialization[task_type][provider]
                )
                
                # Get cost if available
                cost = None
                if provider in cost_analysis.get("providers", {}) and cost_analysis["providers"][provider]:
                    for cost_data in cost_analysis["providers"][provider]:
                        if cost_data["model"] == model_id:
                            cost = cost_data["total_cost"]
                            break
                
                all_models.append({
                    "provider": provider,
                    "model": model_id,
                    "quality": quality,
                    "speed": speed,
                    "cost": cost,
                    "specialized": is_specialized,
                })
        
        # Generate recommendations if models available
        if all_models:
            # Lowest cost recommendation
            cost_models = [m for m in all_models if m["cost"] is not None]
            if cost_models:
                recommendations["lowest_cost"] = min(cost_models, key=lambda x: x["cost"])
            
            # Best quality recommendation
            quality_models = all_models.copy()
            # Prioritize specialized models
            specialized_models = [m for m in quality_models if m["specialized"]]
            if specialized_models:
                quality_models = specialized_models
                
            recommendations["best_quality"] = max(quality_models, key=lambda x: x["quality"])
            
            # Fastest recommendation
            recommendations["fastest"] = max(all_models, key=lambda x: x["speed"])
            
            # Balanced recommendation (consider quality, speed, and cost)
            # Calculate a balanced score
            for model in all_models:
                # Normalize cost (if available)
                normalized_cost = 0.0
                if model["cost"] is not None and cost_models:
                    max_cost = max(m["cost"] for m in cost_models)
                    min_cost = min(m["cost"] for m in cost_models)
                    cost_range = max(max_cost - min_cost, 0.0001)  # Avoid division by zero
                    normalized_cost = (model["cost"] - min_cost) / cost_range
                    # Invert so lower cost = higher score
                    normalized_cost = 1.0 - normalized_cost
                else:
                    normalized_cost = 0.5  # Middle score if cost unknown
                
                # Calculate balanced score
                # Quality: 40%, Speed: 30%, Cost: 30%
                model["balanced_score"] = (
                    (model["quality"] / 10.0) * 0.4 +
                    (model["speed"] / 10.0) * 0.3 +
                    normalized_cost * 0.3
                )
                
                # Bonus for specialized models
                if model["specialized"]:
                    model["balanced_score"] += 0.1
            
            # Get balanced recommendation
            recommendations["balanced"] = max(all_models, key=lambda x: x["balanced_score"])
        
        return recommendations
    
    def _select_provider_and_model(
        self,
        analysis: Dict[str, Any],
        optimization_criteria: str,
        max_cost: Optional[float] = None,
        model_preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Select provider and model based on analysis and criteria.
        
        Args:
            analysis: Task analysis
            optimization_criteria: Optimization criteria
            max_cost: Maximum cost
            model_preferences: Model preferences
            
        Returns:
            Tuple of (provider, model)
        """
        recommendations = analysis.get("recommendations", {})
        
        # Check model preferences first
        if model_preferences:
            preferred_provider = model_preferences.get("provider")
            preferred_model = model_preferences.get("model")
            
            if preferred_provider and preferred_model:
                # Check if preferred model is in options
                provider_options = analysis.get("providers", {})
                if preferred_provider in provider_options:
                    for model_info in provider_options[preferred_provider]:
                        if model_info["id"] == preferred_model:
                            return preferred_provider, preferred_model
            
            # If preferred provider specified without model
            if preferred_provider and not preferred_model:
                provider_options = analysis.get("providers", {})
                if preferred_provider in provider_options and provider_options[preferred_provider]:
                    # Use first model from preferred provider
                    return preferred_provider, provider_options[preferred_provider][0]["id"]
        
        # Select based on optimization criteria
        if optimization_criteria == "cost":
            # Select lowest cost option
            if recommendations.get("lowest_cost"):
                return (
                    recommendations["lowest_cost"]["provider"],
                    recommendations["lowest_cost"]["model"]
                )
        elif optimization_criteria == "quality":
            # Select best quality option
            if recommendations.get("best_quality"):
                # Check if within max cost
                if max_cost is not None and recommendations["best_quality"].get("cost") is not None:
                    if recommendations["best_quality"]["cost"] > max_cost:
                        # If over budget, try balanced option
                        if recommendations.get("balanced") and recommendations["balanced"].get("cost") is not None:
                            if recommendations["balanced"]["cost"] <= max_cost:
                                return (
                                    recommendations["balanced"]["provider"],
                                    recommendations["balanced"]["model"]
                                )
                
                return (
                    recommendations["best_quality"]["provider"],
                    recommendations["best_quality"]["model"]
                )
        elif optimization_criteria == "speed":
            # Select fastest option
            if recommendations.get("fastest"):
                return (
                    recommendations["fastest"]["provider"],
                    recommendations["fastest"]["model"]
                )
        
        # Default to balanced option
        if recommendations.get("balanced"):
            return (
                recommendations["balanced"]["provider"],
                recommendations["balanced"]["model"]
            )
        
        # Fallback to first available provider and model
        provider_options = analysis.get("providers", {})
        if provider_options:
            first_provider = next(iter(provider_options))
            if provider_options[first_provider]:
                return first_provider, provider_options[first_provider][0]["id"]
        
        # Last resort fallback
        return Provider.OPENAI.value, "gpt-3.5-turbo"
    
    def _get_fallback_provider(self, primary_provider: str, available_providers: List[str]) -> Optional[str]:
        """Get a fallback provider if primary provider fails.
        
        Args:
            primary_provider: Primary provider that failed
            available_providers: List of available providers
            
        Returns:
            Fallback provider or None if no suitable fallback
        """
        # Priority order for fallbacks
        fallback_order = [
            Provider.OPENAI.value,
            Provider.ANTHROPIC.value,
            Provider.GEMINI.value,
            Provider.DEEPSEEK.value,
        ]
        
        # Filter out primary provider
        fallbacks = [p for p in fallback_order if p != primary_provider and p in available_providers]
        
        return fallbacks[0] if fallbacks else None
    
    async def _execute_workflow_step(
        self,
        step: Dict[str, Any],
        input_text: str,
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step.
        
        Args:
            step: Step definition
            input_text: Input text for step
            workflow_state: Current workflow state
            
        Returns:
            Step execution result
            
        Raises:
            ValueError: If step is invalid
            Exception: If step execution fails
        """
        # Get step parameters
        step_type = step.get("type", "completion")
        provider = step.get("provider", Provider.OPENAI.value)
        model = step.get("model")
        
        if step_type == "completion":
            # Text completion step
            prompt = step.get("prompt", "")
            # Replace placeholders in prompt
            if "{input}" in prompt:
                prompt = prompt.replace("{input}", input_text)
            else:
                prompt = input_text
                
            # Get provider instance
            provider_instance = get_provider(provider)
            
            # Generate completion
            result = await provider_instance.generate_completion(
                prompt=prompt,
                model=model,
                temperature=step.get("temperature", 0.7),
                max_tokens=step.get("max_tokens"),
                system=step.get("system"),
                messages=step.get("messages"),
            )
            
            return {
                "text": result.text,
                "provider": provider,
                "model": result.model,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.total_tokens,
                },
                "cost": result.cost,
            }
            
        elif step_type == "summarize":
            # Import document tools
            
            # Use summarize_document tool
            result = await self.mcp.execute("summarize_document", {
                "document": input_text,
                "provider": provider,
                "model": model,
                "max_length": step.get("max_length", 300),
                "format": step.get("format", "paragraph"),
            })
            
            return result
            
        elif step_type == "extract_entities":
            # Use extract_entities tool
            result = await self.mcp.execute("extract_entities", {
                "document": input_text,
                "entity_types": step.get("entity_types", ["person", "organization", "location"]),
                "provider": provider,
                "model": model,
            })
            
            return result
            
        elif step_type == "extract_json":
            # Use extract_json tool
            result = await self.mcp.execute("extract_json", {
                "text": input_text,
                "schema": step.get("schema"),
                "provider": provider,
                "model": model,
            })
            
            return result
            
        elif step_type == "quality_check":
            # Use quality_check tool
            result = await self.execute("quality_check", {
                "text": input_text,
                "original_task": step.get("original_task", ""),
                "quality_criteria": step.get("quality_criteria"),
                "provider": provider,
                "model": model,
            })
            
            return result
            
        else:
            raise ValueError(f"Invalid step type: {step_type}")
    
    def _group_steps_by_dependencies(self, workflow_steps: List[Dict[str, Any]]) -> List[List[Tuple[int, Dict[str, Any]]]]:
        """Group workflow steps by dependencies for parallel execution.
        
        Args:
            workflow_steps: List of workflow step definitions
            
        Returns:
            List of groups, where each group is a list of (index, step) tuples
        """
        # Initialize dependency tracking
        step_deps = {}
        step_outputs = {}
        
        # Map step IDs to indices and identify dependencies
        for i, step in enumerate(workflow_steps):
            step_id = step.get("id", f"step_{i}")
            step_outputs[step_id] = i
            
            # Get dependencies
            deps = []
            if "depends_on" in step:
                deps = step["depends_on"] if isinstance(step["depends_on"], list) else [step["depends_on"]]
            elif "input_from" in step and step["input_from"] != "original":
                deps = [step["input_from"]]
                
            step_deps[i] = [step_outputs.get(dep) for dep in deps if step_outputs.get(dep) is not None]
        
        # Group steps by dependencies
        groups = []
        remaining = set(range(len(workflow_steps)))
        completed = set()
        
        while remaining:
            # Find steps with all dependencies satisfied
            current_group = []
            
            for i in list(remaining):
                if all(dep in completed for dep in step_deps.get(i, [])):
                    current_group.append((i, workflow_steps[i]))
                    remaining.remove(i)
            
            if not current_group:
                # No steps can be executed, likely due to circular dependencies
                # Add remaining steps to ensure we don't get stuck
                current_group = [(i, workflow_steps[i]) for i in remaining]
                remaining.clear()
            
            # Add group and update completed steps
            groups.append(current_group)
            completed.update(i for i, _ in current_group)
        
        return groups