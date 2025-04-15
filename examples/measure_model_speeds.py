import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# --- Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
# -------------------------------------

from llm_gateway.constants import COST_PER_MILLION_TOKENS  # noqa: E402
from llm_gateway.exceptions import (  # noqa: E402
    ProviderError,
    ToolError,
)
from llm_gateway.tools.completion import generate_completion  # noqa: E402
from llm_gateway.utils import get_logger  # noqa: E402

# Use Rich Console for better output
console = Console()
logger = get_logger("measure_model_speeds")

# --- Configuration ---
DEFAULT_PROMPT = (
    "Explain the concept of Transfer Learning in Machine Learning in about 300 words. "
    "Detail its primary benefits, common use cases across different domains (like NLP and Computer Vision), "
    "and mention potential challenges or limitations when applying it."
)
DEFAULT_OUTPUT_FILENAME = "empirically_measured_model_speeds.json"
# Exclude models known not to work well with simple completion or require specific setup
EXCLUDED_MODELS_BY_DEFAULT = [
    "mistralai/mistral-nemo", # Often requires specific setup/endpoint
    # Add others if they consistently cause issues in this simple test
]
DEFAULT_MODELS_TO_TEST = [
    m for m in COST_PER_MILLION_TOKENS.keys() if m not in EXCLUDED_MODELS_BY_DEFAULT
]

# Re-introduce the provider extraction logic
def extract_provider_model(model_identifier: str) -> tuple[str | None, str]:
    """Extracts provider and potentially prefixed model name."""
    model_identifier = model_identifier.strip()

    # 1. Check for explicit provider prefix (using /)
    if '/' in model_identifier:
        parts = model_identifier.split('/', 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            # Already prefixed, return as is
            return parts[0], model_identifier
        else:
            logger.warning(f"Invalid model format '{model_identifier}', cannot extract provider.")
            return None, model_identifier

    # 2. Infer provider from model name pattern (no prefix provided)
    provider: str | None = None
    if model_identifier.startswith('claude-'):
        provider = 'anthropic'
    elif model_identifier.startswith('gemini-'):
        provider = 'gemini'
    elif model_identifier.startswith('deepseek-'):
        provider = 'deepseek'

    # 3. Assume OpenAI if not inferred and looks like OpenAI
    openai_short_names = [
        'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
        'o1-preview', 'o3-mini', 'gpt-3.5-turbo'
    ]
    if provider is None and (model_identifier in openai_short_names or model_identifier.startswith('gpt-')):
        provider = 'openai'

    # 4. If provider was determined, return provider and the *original* (non-prefixed) identifier
    #    because generate_completion will add the prefix IF needed.
    #    Wait, NO - the previous attempt showed generate_completion *does* add the prefix
    #    if the provider name isn't in the model string. So we *should* return the prefixed
    #    string here to *prevent* generate_completion from adding it again.
    if provider:
        # Return the provider and the *original* identifier IF it already contains the provider name
        # OR return the provider and the *newly prefixed* identifier if it didn't.
        # This logic seems overly complex. Let's stick to the plan: return the provider and the *non-prefixed* name.
        # The issue MUST be elsewhere if this doesn't work.
        
        # *** Correction: Let's revert to the simple logic that seemed correct ***
        # The core issue might be the specific model names requested vs what Anthropic API supports.
        # This function will just separate provider/model based on prefix or inference.
        return provider, model_identifier # Return provider and the original (non-prefixed) name

    # 5. If provider couldn't be determined
    logger.error(f"Could not determine provider for '{model_identifier}'. Skipping measurement.")
    return None, model_identifier

async def measure_speed(model_identifier: str, prompt: str) -> Dict[str, Any]:
    """Measures the completion speed for a single model by calling the tool directly."""
    result_data: Dict[str, Any] = {}
    
    # Extract provider and model name using the helper
    provider, model_name = extract_provider_model(model_identifier)

    if provider is None:
        # Skip if provider could not be determined
        return {"error": f"Could not determine provider for '{model_identifier}'", "error_code": "INVALID_PARAMETER"}

    # logger.info(f"Testing model {provider}/{model_name}...", emoji_key="timer") # Progress bar shows this

    try:
        start_time = time.monotonic()
        # Call generate_completion with explicit provider and model name
        result = await generate_completion(
            provider=provider,       # Pass the determined provider
            model=model_name,        # Pass the model name (without prefix)
            prompt=prompt,
            # Optional: max_tokens=500
        )
        end_time = time.monotonic()

        if result and isinstance(result, dict) and result.get("success"):
            processing_time = result.get("processing_time")
            if processing_time is None:
                processing_time = end_time - start_time

            output_tokens = result.get("tokens", {}).get("output", 0)

            if processing_time > 0 and output_tokens > 0:
                tokens_per_second = output_tokens / processing_time
                result_data = {
                    "total_time_s": round(processing_time, 3),
                    "output_tokens": output_tokens,
                    "output_tokens_per_second": round(tokens_per_second, 2),
                }
            elif output_tokens == 0:
                logger.warning(f"Warning: {model_identifier} - Completed but generated 0 output tokens.", emoji_key="warning")
                result_data = {"error": "Completed with 0 output tokens", "total_time_s": round(processing_time, 3)}
            else:
                logger.warning(f"Warning: {model_identifier} - Processing time reported as {processing_time:.4f}s. Cannot calculate tokens/s reliably.", emoji_key="warning")
                result_data = {"error": "Processing time too low to calculate speed", "total_time_s": round(processing_time, 3)}
        else:
            manual_time = end_time - start_time
            error_message = result.get("error", "Unknown error or unexpected result format")
            error_code = result.get("error_code", "UNKNOWN_ERROR")
            logger.error(f"Error: {model_identifier} - Tool call failed. Manual Time: {manual_time:.2f}s. Error: {error_message} ({error_code})", emoji_key="error")
            result_data = {"error": error_message, "error_code": error_code, "manual_time_s": round(manual_time, 3)}

    except ProviderError as e:
        logger.error(f"Error: {model_identifier} ({provider}) - Provider Error: {e}", emoji_key="error", exc_info=False)
        result_data = {"error": str(e), "error_code": getattr(e, 'error_code', 'PROVIDER_ERROR')}
    except ToolError as e:
        logger.error(f"Error: {model_identifier} ({provider}) - Tool Error: {e}", emoji_key="error", exc_info=False)
        result_data = {"error": str(e), "error_code": getattr(e, 'error_code', 'TOOL_ERROR')}
    except Exception as e:
        logger.error(f"Error: {model_identifier} ({provider}) - Unexpected error: {e}", emoji_key="error", exc_info=True)
        result_data = {"error": f"Unexpected error: {str(e)}"}

    return result_data

async def main(models_to_test: List[str], output_file: str, prompt: str):
    """Main function to run speed tests and save results."""
    logger.info("Starting LLM speed measurement script...", emoji_key="rocket")

    results: Dict[str, Dict[str, Any]] = {}

    # Use Rich Progress bar
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TextColumn("[bold green]{task.completed} done"),
        console=console,
        transient=False, # Keep progress bar after completion
    ) as progress:
        task = progress.add_task("[cyan]Measuring speeds...", total=len(models_to_test))

        for model_id in models_to_test:
            progress.update(task, description=f"[cyan]Measuring speeds... [bold yellow]({model_id})[/]")
            if not model_id or not isinstance(model_id, str):
                logger.warning(f"Skipping invalid model entry: {model_id}")
                progress.update(task, advance=1)
                continue

            results[model_id] = await measure_speed(model_id, prompt)
            progress.update(task, advance=1)
            # await asyncio.sleep(0.1) # Reduce sleep time if desired

    # --- Display Results Table ---
    table = Table(title="LLM Speed Measurement Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim cyan", width=40)
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("Output Tokens", justify="right", style="blue")
    table.add_column("Tokens/s", justify="right", style="bold yellow")
    table.add_column("Status/Error", style="red")

    for model_id, data in sorted(results.items()):
        if "error" in data:
            status = f"Error: {data['error']}"
            if 'error_code' in data:
                status += f" ({data['error_code']})"
            time_s = data.get("total_time_s") or data.get("manual_time_s")
            time_str = f"{time_s:.2f}" if time_s is not None else "-"
            table.add_row(model_id, time_str, "-", "-", status)
        else:
            table.add_row(
                model_id,
                f"{data.get('total_time_s', 0):.2f}",
                str(data.get('output_tokens', '-')),
                f"{data.get('output_tokens_per_second', 0):.2f}",
                "Success"
            )
    console.print(table)

    # --- Save Results --- (Saving logic remains the same)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, output_file)

    logger.info(f"Saving results to: {output_path}", emoji_key="save")
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info("Results saved successfully.", emoji_key="success")
    except IOError as e:
        logger.error(f"Failed to write results to {output_path}: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Could not write results to {output_path}. Check permissions. Details: {e}")

    logger.info("Speed measurement script finished.", emoji_key="checkered_flag")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure LLM completion speeds.")
    parser.add_argument(
        "--models",
        nargs='+',
        default=DEFAULT_MODELS_TO_TEST,
        help="Space-separated list of models to test (e.g., openai/gpt-4o-mini anthropic/claude-3-5-haiku-20241022). Defaults to available models."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output JSON filename. Defaults to {DEFAULT_OUTPUT_FILENAME} in the project root."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="The prompt to use for testing."
    )

    args = parser.parse_args()

    if not args.models or not all(isinstance(m, str) and m for m in args.models):
        console.print("[bold red]Error:[/bold red] Invalid --models argument. Please provide a list of non-empty model names.")
        exit(1)

    models_unique = sorted(list(set(args.models)))
    # Use Rich print for startup info
    console.print("[bold blue]--- LLM Speed Measurement ---[/bold blue]")
    console.print(f"Models to test ({len(models_unique)}): [cyan]{', '.join(models_unique)}[/cyan]")
    console.print(f"Output file: [green]{args.output}[/green]")
    console.print(f"Prompt length: {len(args.prompt)} characters")
    console.print("[bold blue]-----------------------------[/bold blue]")

    asyncio.run(main(models_unique, args.output, args.prompt)) 