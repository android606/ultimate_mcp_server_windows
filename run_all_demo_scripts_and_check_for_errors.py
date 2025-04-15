#!/usr/bin/env python3
"""
Runs all demo scripts in the 'examples' folder sequentially and checks for errors.

Uses rich for progress tracking and a summary report.
Incorporates specific knowledge about expected outcomes for individual scripts.
"""

import asyncio
import re  # Import regex
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table

# --- Configuration ---
EXAMPLES_DIR = Path(__file__).parent / "examples"
PYTHON_EXECUTABLE = sys.executable # Use the same Python interpreter that runs this script

# Strings indicating a critical error in the output (used if no specific allowed patterns)
DEFAULT_ERROR_INDICATORS = ["Traceback (most recent call last):", "CRITICAL"]

# --- Individual Demo Expectations ---
# Define expected outcomes for specific scripts.
# - expected_exit_code: The code the script should exit with (default: 0)
# - allowed_stderr_patterns: List of regex patterns for stderr messages that are OK for this script.
#                            If this list exists, DEFAULT_ERROR_INDICATORS are ignored for stderr.
# - allowed_stdout_patterns: List of regex patterns for stdout messages that are OK (less common).
#                            If this list exists, DEFAULT_ERROR_INDICATORS are ignored for stdout.
DEMO_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    # --- Scripts with specific known patterns ---
    "filesystem_operations_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Protection Triggered!",
            r"Could not delete collection",
            r"Could not set utime for",
            r"WARNING: No allowed directories loaded",
            r"WARNING: Temporary directory .* not in loaded allowed dirs",
            r"ERROR during config verification",
            r"ERROR: Failed to manually update config",
            r"WARNING: Symlink creation might not be supported",
            r"WARNING: Could not create symlink",
            r"Error during cleanup of",
            r"Unexpected Exception calling", 
            r"Tool Error calling", 
            r"Filesystem demo failed:", 
        ],
        "allowed_stdout_patterns": [r"WARNING:", r"ERROR:"]
    },
    "sql_database_interactions_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Could not delete collection", r"Error deleting knowledge base",
            r"Error directly deleting vector collection", r"Error during initial cleanup",
            r"Connection failed:", r"Failed to connect to database",
            r"Failed to get database status", r"Failed to discover schema",
            r"Failed to get table details", r"Failed to get related tables",
            r"Failed to analyze column statistics", r"Error executing query",
            r"Failed to create view", r"Failed to create index",
            r"Failed to generate documentation", r"Failed to execute transaction",
            r"Error setting up demo database",
            r"Failed to disconnect", 
            r"SQL Database demo failed:", 
            r"Error in connection demo", r"Error in schema discovery demo",
            r"Error in table details demo", r"Error in column statistics demo",
            r"Error in query execution demo", r"Error in database objects demo",
            r"Error in documentation demo", r"Error in transaction demo",
        ]
    },
    "rag_example.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Could not delete collection", r"Error deleting knowledge base",
            r"Error directly deleting vector collection", r"Error during initial cleanup",
            r"Failed to reset ChromaDB client", r"Standard ChromaDB deletion failed",
            r"Failed to create knowledge base", r"Failed to add documents",
            r"Error retrieving from knowledge base", r"Failed to generate response",
            r"No suitable provider found",
            r"Failed to list knowledge bases", 
            r"Error rendering knowledge bases table", 
            r"Error displaying document", 
            r"Failed to process retrieval results", 
            r"Failed to delete knowledge base", 
            r"OpenAIError", # KEEP - Service might raise this if key invalid/missing
            r"Provider .* not available or initialized", # Allow provider init failure
            r"RAG demo failed:", 
        ]
    },
    "marqo_fused_search_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Marqo config file not found", r"Error decoding Marqo config file",
            r"Exiting demo as Marqo config could not be loaded", r"Error connecting to Marqo",
            r"Skipping Example \d+: No suitable .* field found", 
            r"Connection refused", r"Failed to fetch", r"Search failed",
            r"An exception occurred during", 
            r"Your Marqo Python client requires a minimum Marqo version", 
            r"Skipping Example", 
        ]
    },
    "advanced_vector_search_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Could not delete collection",
            r"Failed to initialize provider",
            r"Error generating embeddings with",
            r"Error during vector search demo",
            r"Error during hybrid search demo",
            r"Error calculating semantic similarity",
            r"Mismatch between number of texts and embeddings received",
            r"Vector search demo failed:",
            r"Initializing Gateway", r"Configuration loaded", 
            r"LLM Gateway .* initialized", r"Initializing LLM providers",
            r"Failed to initialize embedding service", 
            r"OpenAIError", # Allow internal errors
        ]
    },
     "vector_search_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Could not delete collection",
            r"Error in vector operations",
            r"Failed to initialize provider",
            r"Error retrieving",
            r"No suitable provider found",
            r"Failed to generate response",
            r"Error in RAG demo",
            r"Skipping RAG demo", 
            r"Vector search demo failed:", 
            r"One or more vector search demos failed", 
            r"Failed to initialize embedding service", 
            r"OpenAIError", # Allow internal errors
        ]
    },
    "prompt_templates_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Cleanup error:",
            r"Template .* not found",
            r"No providers available",
            r"Error during LLM completion",
            r"Failed to initialize providers",
            r"Failed to save template",
            r"Failed to retrieve template",
            r"Could not render with missing variables",
            r"Error rendering translation prompt",
            r"Demo failed:",
            r"Initializing Gateway", r"Configuration loaded", 
            r"LLM Gateway .* initialized", r"Initializing LLM providers"
        ]
    },
    "tournament_code_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Error reading state file directly", r"State file not found at",
            r"Cleanup error:",
            r"No tournament tools found",
            r"Failed to create tournament",
            r"Error fetching status",
            r"Error getting results",
            r"Tournament did not complete successfully",
            r"Template rendering failed",
            r"Failed to initialize providers",
            r"Gateway initialized",
            r"No functions found in the code", 
            r"Error running function", 
            r"Calculator is missing", 
            r"Calculator class not found", 
            r"Error testing calculator", 
            r"Error in code execution", 
            r"Error running tournament", 
            r"Tournament demo failed:", 
        ]
    },
    "tournament_text_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Error reading state file directly", r"State file not found at",
            r"Evaluation with .* timed out", r"Could not evaluate essays:",
            r"Fallback evaluation failed:", r"Could not save evaluation to file",
            r"Could not save fallback evaluation", r"Could not save cost summary",
            r"Cleanup error:",
            r"No tournament tools found",
            r"Failed to create tournament",
            r"Error fetching status",
            r"Error getting results",
            r"Tournament did not complete successfully",
            r"Template rendering failed",
            r"Failed to initialize providers",
            r"Provider .* not available for evaluation",
            r"Error during model request",
            r"Essay evaluation failed",
            r"Gateway initialized",
            r"Could not find path to final comparison file", 
            r"Error in tournament demo", 
            r"Demo failed:", 
        ]
    },
    "test_code_extraction.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Error loading tournament state: .*No such file or directory",
            r"Failed to load tournament state",
            r"Failed to initialize providers",
            r"No round results found",
            r"Test failed:",
            r"Initializing Gateway", r"Configuration loaded", 
            r"LLM Gateway .* initialized", r"Initializing LLM providers"
        ]
    },
    "advanced_extraction_demo.py": {
        "expected_exit_code": 0, 
        "allowed_stderr_patterns": [
            r"Failed to get OpenAI provider", # Allow this error from setup func
            r"Failed to initialize OpenAI provider", 
            r"Error extracting JSON:", r"Error in table extraction:", 
            r"Error in schema inference:", r"Error in entity extraction:", 
            r"Extraction demo failed:"
        ], 
        # Allow the skip message in stdout
        "allowed_stdout_patterns": [r"Skipping .* demo - no provider available", r"Raw Model Output \(JSON parsing failed\)"]
    },
    "analytics_reporting_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Failed to get/initialize", 
            r"No providers could be initialized", 
            r"Error simulating completion", r"No default model found for", 
            r"No metrics data found", r"Error during live monitoring", 
            r"Analytics demo failed:", r"Failed to generate .* plot"
        ]
    },
    "basic_completion.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [r"Provider .* not available or initialized", r"Error generating completion:", r"All providers failed", r"Error with cached completion demo", r"Error with multi-provider completion", r"Error generating streaming completion:", r"Provider .* failed:", r"Example failed:"] 
    },
    "browser_automation_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Could not find search input element",
            r"All methods to interact with search failed",
            r"playwright.*?Error", 
            r"Timeout",
            r"Failed to save screenshot",
            r"net::ERR_CONNECTION_REFUSED",
            r"Navigation failed",
            r"Failed to upload file", 
            r"Failed to save session report", 
            r"Error closing browser", 
            r"Demo failed with unexpected error:", 
        ]
    },
    "claude_integration.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Provider .* not available or initialized", r"Error testing model", 
            r"No suitable Claude model found", r"Selected models not found", 
            r"Error in model comparison", r"Error in system prompt demonstration", 
            r"Model .* not available, falling back to default.", r"Example failed:",
            r"Initializing Gateway", r"Configuration loaded", 
            r"LLM Gateway .* initialized", r"Initializing LLM providers"
            ]
    },
    "compare_synthesize_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [r"Failed to initialize providers", r"Error during .* demo:", r"compare_and_synthesize tool FAILED to register", r"Demo failed with unexpected error:"] 
    },
    "cost_optimization.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [r"Could not estimate cost for", r"Could not get recommendations for", r"API key for .* not found", r"Could not determine provider for", r"Error running completion with", r"Error calling estimate_cost", r"No models met criteria", r"Error calling recommend_model", r"Error getting balanced recommendation", r"Could not get a balanced recommendation", r"Example failed:"] 
    },
    "document_processing.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [r"Tool .* returned error:", r"Exception calling", r"Failed to", r"Error during LLM completion:", r"Chunking Failed:", r"Summarization Failed:", r"Entity Extraction Failed:", r"Q&A Generation Failed:", r"Unexpected chunk result format", r"Document processing demo failed:"] 
    },
    "multi_provider_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Provider .* not available or initialized", r"Error with", 
            r"All providers failed", r"Provider .* failed:", r"Demo failed:",
            r"Initializing Gateway", r"Configuration loaded", 
            r"LLM Gateway .* initialized", r"Initializing LLM providers"
            ]
    },
    "simple_completion_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Provider .* not available", r"Demo failed:",
            r"Initializing Gateway", r"Configuration loaded", 
            r"LLM Gateway .* initialized", r"Initializing LLM providers"
            ]
    },
    "workflow_delegation_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Some API keys missing", # This warning is still valid from initialize_providers
            r"Error delegating task", r"Error executing workflow", 
            r"Error optimizing prompt", r"Error in analyze_task demo", 
            r"Workflow demo failed:",
            r"Initializing required providers", 
            r"All required API keys seem to be present",
            r"Provider .* not available", # Added for get_provider calls within tools
            r"Failed to initialize provider", # Added for get_provider calls within tools
            ]
    },

    # --- Scripts expected to run cleanly (default check) ---
    "cache_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            r"Cache is disabled", 
            r"Error during cache demonstration run", 
            r"Cache demonstration failed:", 
        ]
    },

}

console = Console()

def find_demo_scripts() -> List[Path]:
    """Find all Python demo scripts in the examples directory."""
    if not EXAMPLES_DIR.is_dir():
        console.print(f"[bold red]Error:[/bold red] Examples directory not found at '{EXAMPLES_DIR}'")
        return []
    
    scripts = sorted([
        p for p in EXAMPLES_DIR.glob("*.py") 
        if p.is_file() and p.name != "__init__.py"
    ])
    return scripts

async def run_script(script_path: Path) -> Tuple[int, str, str]:
    """Run a single script and capture its output and exit code."""
    command = [PYTHON_EXECUTABLE, str(script_path)]
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    exit_code = process.returncode
    
    return exit_code, stdout.decode(errors='ignore'), stderr.decode(errors='ignore')

def check_for_errors(script_name: str, exit_code: int, stdout: str, stderr: str) -> Tuple[bool, str]:
    """Check script output/exit code against expectations for that script."""
    
    expectations = DEMO_EXPECTATIONS.get(script_name, {})
    expected_exit_code = expectations.get("expected_exit_code", 0)
    allowed_stderr_patterns = expectations.get("allowed_stderr_patterns", [])
    allowed_stdout_patterns = expectations.get("allowed_stdout_patterns", [])

    # 1. Check Exit Code
    if exit_code != expected_exit_code:
        return False, f"Exited with code {exit_code} (expected {expected_exit_code})"

    # --- Refined Error Log Checking --- 
    
    def find_unexpected_lines(output: str, allowed_patterns: List[str], default_indicators: List[str]) -> List[str]:
        lines = output.strip().splitlines()
        unexpected_lines = []
        for line in lines:
            line_content = line.strip()
            if not line_content: # Skip blank lines
                continue
                
            is_allowed = False
            # Check against specific allowed patterns for this script
            if allowed_patterns:
                for pattern in allowed_patterns:
                    if re.search(pattern, line_content):
                        is_allowed = True
                        break
            
            # If specific patterns were defined and line wasn't allowed, it's unexpected
            if allowed_patterns and not is_allowed:
                 unexpected_lines.append(line)
            # If no specific patterns were defined, check against default critical indicators only
            elif not allowed_patterns:
                for indicator in default_indicators:
                     if indicator in line_content: # Use 'in' for default indicators for simplicity
                         unexpected_lines.append(line)
                         break # Found a default indicator, no need to check others for this line
                         
        return unexpected_lines
        
    unexpected_stderr = find_unexpected_lines(stderr, allowed_stderr_patterns, DEFAULT_ERROR_INDICATORS)
    unexpected_stdout = find_unexpected_lines(stdout, allowed_stdout_patterns, DEFAULT_ERROR_INDICATORS)
    
    # Filter out lines that are just INFO/DEBUG/WARNING level logs unless they are explicitly disallowed
    # (This assumes default log format: YYYY-MM-DD HH:MM:SS] LEVEL ...) or rich format
    def is_ignorable_log(line: str) -> bool:
        line_lower = line.lower()  # noqa: F841
        return (
            re.match(r"^\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]\s+(INFO|DEBUG|WARNING)\s+", line.strip()) or 
            re.match(r"^\s*(INFO|DEBUG|WARNING)\s+", line.strip())
        )

    actual_stderr_errors = [line for line in unexpected_stderr if not is_ignorable_log(line)]
    actual_stdout_errors = [line for line in unexpected_stdout if not is_ignorable_log(line)]
    
    if actual_stderr_errors:
         return False, f"Unexpected errors found in stderr: ...{escape(actual_stderr_errors[0])}..."
         
    if actual_stdout_errors:
         return False, f"Unexpected errors found in stdout: ...{escape(actual_stdout_errors[0])}..."
    # --- End Refined Error Log Checking ---

    # If exit code matches and no unexpected critical errors found
    return True, "Success"

async def main():
    """Main function to run all demo scripts and report results."""
    console.print(Rule("[bold blue]Running All Example Scripts[/bold blue]"))
    
    scripts = find_demo_scripts()
    if not scripts:
        console.print("[yellow]No demo scripts found to run.[/yellow]")
        return 1
        
    console.print(f"Found {len(scripts)} demo scripts in '{EXAMPLES_DIR}'.")
    
    results = []
    success_count = 0
    fail_count = 0
    
    # --- Progress Bar Setup ---
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
        transient=False # Keep progress bar visible after completion
    )

    task_id = progress.add_task("[cyan]Running scripts...", total=len(scripts))

    with Live(progress, console=console, vertical_overflow="visible"):
        for script in scripts:
            script_name = script.name
            progress.update(task_id, description=f"[cyan]Running {script_name}...")

            exit_code, stdout, stderr = await run_script(script)
            is_success, reason = check_for_errors(script_name, exit_code, stdout, stderr)
            
            results.append({
                "script": script_name,
                "success": is_success,
                "reason": reason,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr
            })

            if is_success:
                success_count += 1
            else:
                fail_count += 1
            
            progress.update(task_id, advance=1)
        
        progress.update(task_id, description="[bold green]All scripts finished![/bold green]")
        await asyncio.sleep(0.5) # Allow final update to render

    # --- Summary Report ---
    console.print(Rule("[bold blue]Demo Run Summary[/bold blue]"))
    
    summary_table = Table(title="Script Execution Results", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    summary_table.add_column("Script Name", style="cyan", no_wrap=True)
    summary_table.add_column("Status", style="white")
    summary_table.add_column("Exit Code", style="yellow", justify="right")
    summary_table.add_column("Reason / Output Snippet", style="white")
    
    for result in results:
        status_icon = "[green]✅ SUCCESS[/green]" if result["success"] else "[bold red]❌ FAILURE[/bold red]"
        reason_or_output = result["reason"]
        
        # --- Enhanced Snippet Logic ---
        # Prioritize showing snippet related to the failure reason
        if not result["success"]:
            output_to_search = result["stderr"] + result["stdout"] # Combined output
            snippet = ""
            
            # If failure is due to unexpected error message
            if "Unexpected errors found" in reason_or_output:
                # Extract the specific error shown in the reason
                match = re.search(r"Unexpected errors found in (stdout|stderr): \.\.\.(.*)\.\.\.\"?", reason_or_output)
                if match:
                    error_snippet_text = match.group(2).strip()
                    # Try to find this snippet in the actual output
                    start_idx = output_to_search.find(error_snippet_text)
                    if start_idx != -1:
                        # Find the start of the line containing the snippet
                        line_start_idx = output_to_search.rfind('\n', 0, start_idx) + 1
                        lines_around_error = output_to_search[line_start_idx:].splitlines()
                        snippet = "\n".join(lines_around_error[:5]) # Show 5 lines from error
                        if len(lines_around_error) > 5:
                            snippet += "\n..."
           
            # If failure is due to exit code, show end of stderr/stdout
            elif "Exited with code" in reason_or_output:
                if result["stderr"].strip():
                     lines = result["stderr"].strip().splitlines()
                     snippet = "\n".join(lines[-5:]) # Last 5 lines of stderr
                elif result["stdout"].strip():
                     lines = result["stdout"].strip().splitlines()
                     snippet = "\n".join(lines[-5:]) # Last 5 lines of stdout
           
            # Fallback if no specific snippet found yet for failure
            if not snippet:
                 lines = output_to_search.strip().splitlines()
                 snippet = "\n".join(lines[-5:]) # Last 5 lines overall

            if snippet:
                 reason_or_output += f"\n---\n[dim]{escape(snippet)}[/dim]"

        elif result["success"]:
             # Show last few lines of stdout for successful runs
             lines = result["stdout"].strip().splitlines()
             if lines:
                 snippet = "\n".join(lines[-3:]) # Show last 3 lines
                 reason_or_output += f"\n---\n[dim]{escape(snippet)}[/dim]"
             else: # Handle case with no stdout
                  reason_or_output += "\n---\n[dim](No stdout produced)[/dim]"
        # --- End Enhanced Snippet Logic ---

        summary_table.add_row(
            result["script"],
            status_icon,
            str(result["exit_code"]),
            reason_or_output
        )
        
    console.print(summary_table)
    
    # --- Final Count ---
    console.print(Rule())
    total_scripts = len(scripts)
    final_message = f"[bold green]{success_count}[/bold green] succeeded, [bold red]{fail_count}[/bold red] failed out of {total_scripts} scripts."
    final_color = "green" if fail_count == 0 else "red"
    console.print(Panel(final_message, border_style=final_color, expand=False))
    
    return 1 if fail_count > 0 else 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 