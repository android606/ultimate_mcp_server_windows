"""
Tournament task implementations for asynchronous tournament execution.
"""
# Standard Library Imports
import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from llm_gateway.core.models.tournament import (
    ModelResponseData,
    TournamentData,
    TournamentRoundResult,
    TournamentStatus,
)
from llm_gateway.core.tournaments.manager import tournament_manager
from llm_gateway.core.tournaments.utils import (
    create_round_prompt,
    extract_thinking,
    generate_comparison_file,
    generate_synthesis_prompt,
    get_word_count,
    save_model_response,
)

# Import the tool function
from llm_gateway.tools.completion import generate_completion

logger = logging.getLogger(__name__)

async def run_tournament_async(tournament_id: str):
    """
    Asynchronous task orchestrating the tournament rounds.
    """
    # Add a small delay to mitigate potential race conditions on startup
    await asyncio.sleep(0.1) 
    
    logger.info(f"[TASK START] Starting execution for tournament {tournament_id}")
    # Ensure we load the latest state from disk
    logger.debug(f"[TASK DEBUG] Force reloading tournament {tournament_id} from disk")
    tournament = tournament_manager.get_tournament(tournament_id, force_reload=True)
    
    # Debug logging for tournament status
    if tournament:
        logger.debug(f"[TASK DEBUG] Tournament loaded successfully. ID: {tournament.tournament_id}")
        logger.debug(f"[TASK DEBUG] Tournament state: {tournament.status}, Round: {tournament.current_round}/{tournament.config.rounds}")
        logger.debug(f"[TASK DEBUG] Tournament config: {tournament.config.model_dump_json()}")
    else:
        logger.error(f"[TASK ERROR] Tournament {tournament_id} not found after reload")
        # Try to find the tournament by scanning all tournament directories
        storage_base = Path(__file__).parent.parent.parent.parent / "storage" / "tournaments"
        logger.debug(f"[TASK DEBUG] Scanning all tournament directories in: {storage_base}")
        
        # Look for the tournament in all subdirectories
        if storage_base.exists():
            for dir_path in storage_base.iterdir():
                if dir_path.is_dir():
                    state_file = dir_path / "tournament_state.json"
                    if state_file.exists():
                        try:
                            with open(state_file, 'r') as f:
                                data = json.load(f)
                                if data.get("tournament_id") == tournament_id:
                                    logger.info(f"[TASK INFO] Found tournament {tournament_id} in directory: {dir_path}")
                                    # Load the tournament properly through the manager
                                    tournament = tournament_manager.get_tournament(tournament_id, force_reload=True)
                                    if tournament:
                                        break
                        except (IOError, json.JSONDecodeError) as e:
                            logger.debug(f"[TASK DEBUG] Error reading {state_file}: {e}")
        
        # If we still haven't found the tournament, exit
        if not tournament:
            logger.error(f"[TASK ERROR] Tournament {tournament_id} not found in any directory. Exiting.")
            return
    
    logger.info(f"[TASK DEBUG] Tournament loaded successfully. Type: {tournament.config.tournament_type}, Rounds: {tournament.config.rounds}")
    
    try:
        # Initialize tournament state  
        if tournament.current_round < 0:  # Not started yet
            logger.debug("[TASK DEBUG] Initializing tournament state for round 0")
            tournament.current_round = 0
            tournament.rounds_results = []  # Ensure we have an empty list
            # Make sure tournament is marked as running (should already be set, but just in case)
            tournament.status = TournamentStatus.RUNNING
            logger.info(f"[TASK DEBUG] Tournament initialized with {len(tournament.config.models)} models")
            tournament_manager._save_tournament_state(tournament)
            logger.debug("[TASK DEBUG] Tournament state saved after initialization")
        
        # Process each round
        while tournament.current_round < tournament.config.rounds:
            current_round = tournament.current_round
            logger.info(f"[TASK DEBUG] Starting round {current_round}")
            
            # Initialize the round if it doesn't exist
            if len(tournament.rounds_results) <= current_round:
                logger.info(f"[TASK DEBUG] Initializing round {current_round} results")
                tournament.rounds_results.append(TournamentRoundResult(
                    round_num=current_round,
                    responses={}
                ))
                tournament_manager._save_tournament_state(tournament)
                logger.debug("[TASK DEBUG] Tournament state saved after round initialization")
            
            # Process this round
            logger.info(f"[TASK DEBUG] Processing round {current_round}")
            try:
                await process_round(tournament, current_round)
                logger.info(f"[TASK DEBUG] Round {current_round} processing completed")
            except Exception as e:
                logger.error(f"Error in round {current_round}: {e}", exc_info=True)
                tournament.status = TournamentStatus.FAILED
                tournament.error_message = f"Failed during round {current_round}: {str(e)}"
                tournament_manager._save_tournament_state(tournament)
                logger.debug("[TASK DEBUG] Tournament state saved after round failure")
                return
                
            # Save tournament state after round completes
            tournament_manager._save_tournament_state(tournament)
            logger.debug("[TASK DEBUG] Tournament state saved after round completion")
            
            # Move to next round
            logger.info(f"[TASK DEBUG] Advancing to round {current_round + 1}")
            tournament.current_round += 1
            tournament_manager._save_tournament_state(tournament)
            logger.debug("[TASK DEBUG] Tournament state saved after advancing to next round")
        
        # All rounds completed
        logger.info(f"[TASK DEBUG] All {tournament.config.rounds} rounds completed")
        tournament.status = TournamentStatus.COMPLETED
        tournament.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        logger.debug("[TASK DEBUG] Tournament state saved after completion")
        logger.info(f"Tournament {tournament_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error executing tournament: {e}", exc_info=True)
        tournament.status = TournamentStatus.FAILED
        tournament.error_message = str(e)
        tournament.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        logger.debug("[TASK DEBUG] Tournament state saved after error")
        
async def process_round(tournament: TournamentData, current_round: int) -> None:
    """
    Processes a single round of the tournament, running all tasks in parallel.
    """
    logger.info(f"[PROCESS ROUND] Starting round {current_round} for tournament {tournament.tournament_id}")
    
    try:
        # Get round results
        if current_round >= len(tournament.rounds_results):
            raise ValueError(f"Invalid round number: {current_round}")
        
        round_results = tournament.rounds_results[current_round]
        
        # Configure tournament type
        is_code_tournament = tournament.config.tournament_type == "code"  # noqa: F841
        extraction_model_id = tournament.config.extraction_model_id  # noqa: F841
        
        # Create tasks for all models
        model_tasks = []
        
        # Get previous round responses if this isn't the first round
        previous_round_responses = None
        if current_round > 0:
            previous_round = tournament.rounds_results[current_round - 1]
            previous_round_responses = {
                model_id: response.response_text 
                for model_id, response in previous_round.responses.items()
                if response.response_text
            }
        
        for model_config in tournament.config.models:
            model_id = model_config.model_id
            
            # Skip if already processed
            if model_id in round_results.responses:
                logger.info(f"[PROCESS ROUND] Skipping already processed model {model_id}")
                continue
                
            # Add the task
            task = process_model_task(
                tournament=tournament,
                model_id=model_id,
                round_num=current_round,
                previous_round_responses=previous_round_responses
            )
            model_tasks.append(task)
            logger.info(f"[PROCESS ROUND] Added task for model {model_id}")
        
        # Exit if no tasks to run
        if not model_tasks:
            logger.info(f"[PROCESS ROUND] No models to process for round {current_round}")
            return
            
        # Run all tasks in parallel
        logger.info(f"[PROCESS ROUND] Running {len(model_tasks)} model tasks in parallel")
        results = await asyncio.gather(*model_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            model_config = tournament.config.models[i]
            model_id = model_config.model_id
            
            # Handle exceptions
            if isinstance(result, Exception):
                logger.error(f"Error processing model {model_id}: {result}", exc_info=True)
                continue
                
            # Store result in the proper format
            response_data = ModelResponseData(
                model_id=model_id,
                round_num=current_round,
                response_text=result.get("response_text", ""),
                thinking_process=result.get("thinking", ""),
                metrics=result.get("metrics", {}),
                response_file_path=result.get("response_file", ""),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add response
            round_results.responses[model_id] = response_data
            tournament_manager._save_tournament_state(tournament)
            
        # Create comparison file for round
        comparison_content = generate_comparison_file(tournament, current_round)
        if comparison_content:
            round_dir = Path(tournament.storage_path) / f"round_{current_round}"
            round_dir.mkdir(exist_ok=True)
            comparison_file = round_dir / "model_comparison.md"
            
            with open(comparison_file, 'w', encoding='utf-8') as f:
                f.write(comparison_content)
                
            # Store the path back to the round results  
            round_results.comparison_file_path = str(comparison_file)
            tournament_manager._save_tournament_state(tournament)
            
        # Round successfully completed    
        logger.info(f"[PROCESS ROUND] Round {current_round} completed successfully")
        
    except Exception as e:
        logger.error(f"[PROCESS ROUND] Error processing round {current_round}: {e}", exc_info=True)
        raise

async def process_single_model(
    model_id: str,
    prompt: str,
    tournament_id: str,
    round_num: int,
    is_code_tournament: bool,
    extraction_model_id: Optional[str] = None
) -> ModelResponseData:
    """
    Handles the logic for calling a single model provider using the generate_completion tool.
    """
    start_time = time.monotonic()
    logger.info(f"[MODEL TASK] Processing model {model_id} for round {round_num}")
    
    # Get tournament to access storage path
    tournament = tournament_manager.get_tournament(tournament_id)
    if not tournament:
        raise ValueError(f"Tournament {tournament_id} not found")
    
    # Setup storage paths
    round_storage_path = Path(tournament.storage_path) / f"round_{round_num}"
    round_storage_path.mkdir(exist_ok=True, parents=True)
    
    response_data = ModelResponseData(model_id=model_id, round_num=round_num)
    extracted_code: Optional[str] = None  # noqa: F841
    file_extension = ".py" if is_code_tournament else ".md"
    
    provider_name = model_id.split('/')[0] if '/' in model_id else None # Infer provider from model_id if possible
    if not provider_name:
        logger.warning(f"[MODEL TASK] Could not infer provider from model_id: {model_id}. Attempting call without explicit provider.")
        # Note: generate_completion might fail if provider isn't specified and cannot be inferred

    try:
        # Use generate_completion tool
        logger.info(f"[MODEL TASK] Calling generate_completion for model {model_id} with prompt length {len(prompt)}")
        # Log prompt preview
        preview_length = 100
        prompt_preview = prompt[:preview_length] + "..." if len(prompt) > preview_length else prompt
        logger.info(f"[MODEL TASK] Prompt preview: {prompt_preview}")

        # Call the tool function directly
        completion_result_dict = await generate_completion(
            prompt=prompt,
            model=model_id, # Pass the full model ID
            provider=provider_name # Pass inferred provider
            # Add other params like max_tokens, temperature if needed/available in TournamentConfig
        )
        
        # Check for success
        if not completion_result_dict.get("success"):
            error_msg = completion_result_dict.get("error", "generate_completion tool indicated failure")
            raise RuntimeError(f"Completion failed for {model_id}: {error_msg}")

        # Extract data from the dictionary returned by the tool
        response_text = completion_result_dict.get("text", "")
        actual_model_used = completion_result_dict.get("model", model_id) # Use actual model if returned
        token_info = completion_result_dict.get("tokens", {})
        cost = completion_result_dict.get("cost", 0.0)
        processing_time_sec = completion_result_dict.get("processing_time", 0.0)
        latency_ms = int(processing_time_sec * 1000)

        # Log response preview
        response_preview = response_text[:preview_length] + "..." if len(response_text) > preview_length else response_text
        logger.info(f"[MODEL TASK] Response preview for {actual_model_used}: {response_preview}")

        # Extract metrics from the tool result
        completion_metrics = {
            "input_tokens": token_info.get("input"),
            "output_tokens": token_info.get("output"),
            "cost": cost,
            "latency_ms": latency_ms # Use processing_time from tool
        }

        # Process response - use async extract_thinking
        thinking = await extract_thinking(response_text)
        code_metrics = {} # Placeholder for potential future code analysis metrics

        # Save response to file with better naming pattern
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_model_id = re.sub(r'[^a-zA-Z0-9_\-.]', '_', actual_model_used) # Use actual model name
        safe_tournament_id = re.sub(r'[^a-zA-Z0-9_\-.]', '_', tournament_id)

        filename_base = f"tournament_{safe_tournament_id}_round-{round_num}_model-{safe_model_id}_{timestamp}"
        raw_response_path = round_storage_path / f"{filename_base}{file_extension}"

        raw_response_path.write_text(response_text or "", encoding="utf-8")

        # Create a more user-friendly version with added context
        readable_content = f"""# Tournament Response
**Tournament ID:** {tournament_id}
**Round:** {round_num}
**Model:** {actual_model_used}
**Timestamp:** {datetime.now().isoformat()}
**Tokens:** {completion_metrics.get('input_tokens', 'N/A')} in, {completion_metrics.get('output_tokens', 'N/A')} out
**Cost:** ${completion_metrics.get('cost', 0.0):.6f}
**Latency:** {completion_metrics.get('latency_ms', 'N/A')}ms

## Prompt
```
{prompt}
```

## Response
```
{response_text}
```
"""
        readable_path = round_storage_path / f"{filename_base}_readable{file_extension}"
        readable_path.write_text(readable_content, encoding="utf-8")

        logger.info(f"[MODEL TASK] Saved response to: {readable_path}")

        # Populate response data
        response_data.model_id = actual_model_used # Store the actual model used
        response_data.response_text = response_text
        response_data.thinking_process = thinking
        response_data.metrics = {**completion_metrics, **code_metrics}
        response_data.timestamp = datetime.now(timezone.utc)
        response_data.response_file_path = str(raw_response_path) # Store path to raw response
        response_data.metrics["total_processing_time_ms"] = int((time.monotonic() - start_time) * 1000) # Keep overall task time

        logger.info(f"[MODEL TASK] Finished processing model {actual_model_used} for round {round_num} in {response_data.metrics['total_processing_time_ms']}ms")

    except Exception as e:
        logger.error(f"[MODEL TASK] Error processing model {model_id}: {e}", exc_info=True)
        response_data.error = str(e)
    
    return response_data

async def run_single_round_task(tournament_id: str, round_num: int):
    """
    Task that runs a single round of the tournament, including LLM calls.
    """
    logger.info(f"[ROUND TASK START] Running round {round_num} for tournament {tournament_id}")
    tournament = tournament_manager.get_tournament(tournament_id, force_reload=True)
    
    # --- Check if tournament exists or was cancelled before proceeding --- 
    if not tournament:
        logger.error(f"[ROUND TASK FAIL] Tournament {tournament_id} not found at start of round {round_num}.")
        return
    if tournament.status == TournamentStatus.CANCELLED:
        logger.info(f"[ROUND TASK ABORT] Tournament {tournament_id} was cancelled. Stopping round {round_num}.")
        # Ensure round status reflects cancellation if it was running
        if round_num < len(tournament.rounds_results):
             round_result = tournament.rounds_results[round_num]
             if round_result.status == TournamentStatus.RUNNING:
                  round_result.status = TournamentStatus.CANCELLED
                  round_result.error = "Cancelled by user request during execution."
                  round_result.end_time = datetime.now(timezone.utc)
                  tournament_manager._save_tournament_state(tournament)
        return
    # -------------------------------------------------------------------
    
    if round_num >= len(tournament.rounds_results):
        logger.error(f"[ROUND TASK FAIL] Invalid round number {round_num} for tournament {tournament_id} state.")
        return
    
    round_result = tournament.rounds_results[round_num]
    
    try:
        # Mark round as running
        round_result.status = TournamentStatus.RUNNING
        round_result.start_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        logger.info(f"[ROUND TASK] Round {round_num} marked as running")
        
        # Get tournament config
        is_code_tournament = tournament.config.tournament_type == "code"
        extraction_model_id = tournament.config.extraction_model_id
        
        # Create prompt for this round
        prompt = create_round_prompt(tournament, round_num)
        
        # Create tasks for all configured models
        model_tasks = []
        for model_config in tournament.config.models:
            model_id = model_config.model_id
            
            # Skip if already processed
            if model_id in round_result.responses:
                logger.info(f"[ROUND TASK] Skipping already processed model {model_id}")
                continue
            
            # Add task for this model
            task = process_single_model(
                model_id=model_id,
                prompt=prompt,
                tournament_id=tournament_id,
                round_num=round_num,
                is_code_tournament=is_code_tournament,
                extraction_model_id=extraction_model_id
            )
            model_tasks.append(task)
            logger.info(f"[ROUND TASK] Added task for model {model_id}")
        
        # Exit if no tasks to run
        if not model_tasks:
            logger.info(f"[ROUND TASK] No models to process for round {round_num}")
            round_result.status = TournamentStatus.COMPLETED
            round_result.end_time = datetime.now(timezone.utc)
            tournament_manager._save_tournament_state(tournament)
            return
        
        # Run all model tasks in parallel
        logger.info(f"[ROUND TASK] Running {len(model_tasks)} model tasks in parallel")
        results = await asyncio.gather(*model_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            model_id = tournament.config.models[i].model_id
            
            # Handle exceptions
            if isinstance(result, Exception):
                logger.error(f"[ROUND TASK] Error processing model {model_id}: {result}", exc_info=True)
                continue
            
            # Store result
            round_result.responses[model_id] = result
            tournament_manager._save_tournament_state(tournament)
        
        # Create comparison file
        comparison_content = generate_comparison_file(tournament, round_num)
        if comparison_content:
            round_dir = Path(tournament.storage_path) / f"round_{round_num}"
            round_dir.mkdir(exist_ok=True)
            comparison_file = round_dir / "model_comparison.md"
            
            with open(comparison_file, 'w', encoding='utf-8') as f:
                f.write(comparison_content)
            
            # Store the path in round results
            round_result.comparison_file_path = str(comparison_file)
            tournament_manager._save_tournament_state(tournament)
        
        # Mark round as completed
        round_result.status = TournamentStatus.COMPLETED
        round_result.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        logger.info(f"[ROUND TASK COMPLETE] Round {round_num} for tournament {tournament_id} completed successfully")
        
        # If this was the last round, mark the tournament as completed
        if round_num == tournament.config.rounds - 1:
            tournament.status = TournamentStatus.COMPLETED
            tournament.end_time = datetime.now(timezone.utc)
            tournament_manager._save_tournament_state(tournament)
            logger.info(f"[ROUND TASK] Tournament {tournament_id} marked as completed after final round")
    
    except Exception as e:
        logger.error(f"[ROUND TASK ERROR] Error processing round {round_num}: {e}", exc_info=True)
        round_result.status = TournamentStatus.FAILED
        round_result.error = str(e)
        round_result.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        
        # Mark tournament as failed
        tournament.status = TournamentStatus.FAILED
        tournament.error_message = f"Failed during round {round_num}: {str(e)}"
        tournament.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament) 

async def process_model_task(
    tournament: TournamentData,
    model_id: str,
    round_num: int,
    previous_round_responses: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Process a single model task for the tournament using generate_completion tool.
    
    Args:
        tournament: Tournament data
        model_id: Model to use (e.g., 'openai/gpt-4o')
        round_num: Current round number
        previous_round_responses: Previous round responses (for rounds > 0)
        
    Returns:
        Model task result with response text and metrics
    """
    start_task_time = time.monotonic()
    # Infer provider from model_id format 'provider:model_name' or 'provider/model_name'
    provider_id = None
    if ':' in model_id:
        provider_id = model_id.split(':')[0]
    elif '/' in model_id: # Keep backward compatibility if '/' is used
        provider_id = model_id.split('/')[0]
        
    if not provider_id:
         logger.warning(f"[MODEL TASK] Could not infer provider from model_id: {model_id}. Attempting call without explicit provider.")
    
    try:
        logger.info(f"[MODEL TASK] Processing model {model_id} for round {round_num} (Provider: {provider_id})")
            
        # Generate prompt based on tournament type and round
        if round_num == 0:
            prompt = tournament.config.prompt
        else:
            prompt = generate_synthesis_prompt(tournament, previous_round_responses)
        
        # Generate completion using the tool
        logger.info(f"[MODEL TASK] Calling generate_completion for model {model_id} with prompt length {len(prompt)}")
        preview_length = 100
        prompt_preview = prompt[:preview_length] + "..." if len(prompt) > preview_length else prompt
        logger.info(f"[MODEL TASK] Prompt preview: {prompt_preview}")

        completion_result_dict = await generate_completion(
            prompt=prompt,
            model=model_id,
            provider=provider_id # Pass the inferred provider
            # Add other params like max_tokens, temperature if needed/available
        )

        # Check for success
        if not completion_result_dict.get("success"):
            error_msg = completion_result_dict.get("error", "generate_completion tool indicated failure")
            raise RuntimeError(f"Completion failed for {model_id}: {error_msg}")

        # Extract data from the result dictionary
        response_text = completion_result_dict.get("text", "")
        actual_model_used = completion_result_dict.get("model", model_id)
        token_info = completion_result_dict.get("tokens", {})
        cost = completion_result_dict.get("cost", 0.0)
        processing_time_sec = completion_result_dict.get("processing_time", 0.0)

        # Log response preview
        response_preview = response_text[:preview_length] + "..." if len(response_text) > preview_length else response_text
        logger.info(f"[MODEL TASK] Response preview for {actual_model_used}: {response_preview}")

        # Extract metrics from the tool result
        completion_metrics = {
            "input_tokens": token_info.get("input"),
            "output_tokens": token_info.get("output"),
            "cost": cost,
            "processing_time_ms": int(processing_time_sec * 1000) # Use tool's processing time
        }
        
        # Extract thinking/reasoning if present - use async extract_thinking
        thinking = await extract_thinking(response_text)
        
        # Save response to a file with timestamp - use async save_model_response
        response_file = await save_model_response(
            tournament=tournament,
            round_num=round_num,
            model_id=actual_model_used, # Use actual model name
            response_text=response_text,
            thinking=thinking
        )
        
        total_task_time_ms = int((time.monotonic() - start_task_time) * 1000)
        completion_metrics["total_task_time_ms"] = total_task_time_ms # Add overall task time

        logger.info(f"[MODEL TASK] Finished processing model {actual_model_used} for round {round_num} in {total_task_time_ms}ms (LLM time: {completion_metrics['processing_time_ms']}ms)")
        
        return {
            "model_id": actual_model_used, # Return actual model used
            "response_text": response_text,
            "thinking": thinking,
            "word_count": get_word_count(response_text),
            "metrics": completion_metrics,
            "response_file": str(response_file) # Ensure path is string
        }
    except Exception as e:
        logger.error(f"[MODEL TASK] Error processing model {model_id}: {str(e)}", exc_info=True)
        total_task_time_ms = int((time.monotonic() - start_task_time) * 1000)
        return {
            "model_id": model_id,
            "error": str(e),
            "response_text": f"Error generating response: {str(e)}",
            "thinking": None,
            "word_count": 0,
            "metrics": {
                "error": str(e), 
                "total_task_time_ms": total_task_time_ms,
                "processing_time_ms": None # LLM call failed
            },
            "response_file": None
        } 