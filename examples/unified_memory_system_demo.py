#!/usr/bin/env python
import asyncio
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt_id(val: Any, length: int = 8) -> str:
    """Return a short id string safe for logs."""
    s = str(val) if val is not None else "?"
    # Ensure slicing doesn't go out of bounds if string is shorter than length
    return s[: min(length, len(s))]


# --- Project Setup ---
# Add project root to path for imports when running as script
# Adjust this path if your script location relative to the project root differs
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    # Navigate up until we find a directory likely containing the project modules
    PROJECT_ROOT = SCRIPT_DIR
    while (
        not (PROJECT_ROOT / "ultimate_mcp_server").is_dir()
        and not (PROJECT_ROOT / "ultimate_mcp_client").is_dir()
        and PROJECT_ROOT.parent != PROJECT_ROOT
    ):  # Prevent infinite loop
        PROJECT_ROOT = PROJECT_ROOT.parent

    if (
        not (PROJECT_ROOT / "ultimate_mcp_server").is_dir()
        and not (PROJECT_ROOT / "ultimate_mcp_client").is_dir()
    ):
        print(
            f"Error: Could not reliably determine project root from {SCRIPT_DIR}.", file=sys.stderr
        )
        # Fallback: Add script dir anyway, maybe it's flat structure
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
            print(
                f"Warning: Added script directory {SCRIPT_DIR} to path as fallback.",
                file=sys.stderr,
            )
        else:
            sys.exit(1)  # Give up if markers not found after traversing up

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

except Exception as e:
    print(f"Error setting up sys.path: {e}", file=sys.stderr)
    sys.exit(1)

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.traceback import install as install_rich_traceback

# Tools and related components from unified_memory
from ultimate_mcp_server.tools.unified_memory_system import (
    # Initialization
    initialize_memory_system,
    # Workflow
    create_workflow,
    get_workflow_details,
    list_workflows,
    update_workflow_status,
    # Action
    record_action_start,
    record_action_completion,
    get_recent_actions,
    get_action_details,
    # Action Dependency Tools (NEW)
    add_action_dependency,
    get_action_dependencies,
    # Artifacts (NEW)
    record_artifact,
    get_artifacts,
    get_artifact_by_id,
    # Thought
    create_thought_chain,
    record_thought,
    get_thought_chain,
    # Core Memory
    store_memory,
    get_memory_by_id,
    update_memory,
    query_memories,
    create_memory_link, 
    get_linked_memories,
    search_semantic_memories,
    hybrid_search_memories,    
    get_working_memory,
    save_cognitive_state,
    load_cognitive_state,    
    focus_memory,
    optimize_working_memory,
    # Utilities/Enums/Exceptions needed
    DEFAULT_DB_PATH,
    DBConnection,
    MemoryLevel,
    MemoryType,
    ActionType,
    ActionStatus,
    WorkflowStatus,
    ThoughtType,
    ArtifactType,
    LinkType,
    ToolError,
    ToolInputError,
    # Defer these advanced features for now:
    # get_workflow_context,
    # auto_update_focus,
    # promote_memory_level,
    # consolidate_memories,
    # generate_reflection,
    # summarize_text,
    # delete_expired_memories,
    # compute_memory_statistics,
    # generate_workflow_report,
    # visualize_reasoning_chain,
    # visualize_memory_network,
)

# Utilities from the project
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.config import get_config

console = Console()
logger = get_logger("demo.unified_memory")
config = get_config()

install_rich_traceback(show_locals=False, width=console.width)

DEMO_DB_FILE: Optional[str] = None  # Global to hold the DB path being used


async def safe_tool_call(func, args: Dict, description: str, suppress_output: bool = False):
    """Helper to call a tool function, catch errors, and display results."""
    display_title = not suppress_output
    display_args = not suppress_output
    display_result_panel = not suppress_output

    if display_title:
        title = f"DEMO: {description}"
        console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]", style="blue"))
    if display_args:
        if args:
            console.print(f"[dim]Calling [bold cyan]{func.__name__}[/] with args:[/]")
            try:
                # Filter out db_path if it matches the global default for cleaner logs
                args_to_print = {
                    k: v for k, v in args.items() if k != "db_path" or v != DEMO_DB_FILE
                }
                args_repr = pretty_repr(args_to_print, max_length=120, max_string=100)
            except Exception:
                args_repr = str(args)[:300]
            console.print(args_repr)
        else:
            console.print(f"[dim]Calling [bold cyan]{func.__name__}[/] (no arguments)[/]")

    start_time = time.monotonic()
    result = None
    try:
        # Use the global DEMO_DB_FILE if db_path isn't explicitly in args
        if "db_path" not in args and DEMO_DB_FILE:
            args["db_path"] = DEMO_DB_FILE

        result = await func(**args)

        processing_time = time.monotonic() - start_time
        log_func = getattr(logger, "debug", print)
        log_func(f"Tool '{func.__name__}' execution time: {processing_time:.4f}s")

        if display_result_panel:
            success = isinstance(result, dict) and result.get("success", False)
            panel_title = f"[bold {'green' if success else 'yellow'}]Result: {func.__name__} {'✅' if success else '❔'}[/]"
            panel_border = "green" if success else "yellow"

            try:
                result_repr = pretty_repr(result, max_length=200, max_string=150)
            except Exception:
                result_repr = f"(Could not represent result of type {type(result)} fully)\n{str(result)[:500]}"

            console.print(
                Panel(result_repr, title=panel_title, border_style=panel_border, expand=False)
            )
            # Specific display logic for reports/visualizations can remain here if needed later
            # ...

        return result

    except (ToolInputError, ToolError) as e:
        processing_time = time.monotonic() - start_time
        log_func_error = getattr(logger, "error", print)
        log_func_error(f"Tool '{func.__name__}' failed: {e}", exc_info=False)
        if display_result_panel:
            error_title = f"[bold red]Error: {func.__name__} Failed ❌[/]"
            error_content = f"[bold red]{type(e).__name__}:[/] {escape(str(e))}"
            details = None
            if hasattr(e, "details") and e.details:
                details = e.details
            elif hasattr(e, "context") and e.context:
                details = e.context

            if details:
                try:
                    details_repr = pretty_repr(details)
                except Exception:
                    details_repr = str(details)
                error_content += f"\n\n[yellow]Details:[/]\n{details_repr}"
            console.print(Panel(error_content, title=error_title, border_style="red", expand=False))
        return {
            "success": False,
            "error": str(e),
            "error_code": getattr(e, "error_code", "TOOL_ERROR"),
            "error_type": type(e).__name__,
            "details": details or {},
            "isError": True,
        }
    except Exception as e:
        processing_time = time.monotonic() - start_time
        log_func_critical = getattr(logger, "critical", print)
        log_func_critical(f"Unexpected error calling '{func.__name__}': {e}", exc_info=True)
        if display_result_panel:
            console.print(f"\n[bold red]CRITICAL UNEXPECTED ERROR in {func.__name__}:[/bold red]")
            console.print_exception(show_locals=False)
        return {
            "success": False,
            "error": f"Unexpected: {str(e)}",
            "error_code": "UNEXPECTED_ERROR",
            "error_type": type(e).__name__,
            "details": {"traceback": traceback.format_exc()},
            "isError": True,
        }
    finally:
        if display_title:
            console.print()


# --- Demo Setup & Teardown ---


async def setup_demo_environment():
    """Initialize the memory system using the DEFAULT database file."""
    global DEMO_DB_FILE
    DEMO_DB_FILE = DEFAULT_DB_PATH
    log_func_info = getattr(logger, "info", print)
    log_func_info(f"Using default database for demo: {DEMO_DB_FILE}")
    console.print(
        Panel(
            f"Using default database: [cyan]{DEMO_DB_FILE}[/]\n"
            f"[yellow]NOTE:[/yellow] This demo will operate on the actual development database.",
            title="Demo Setup",
            border_style="yellow",
        )
    )

    init_result = await safe_tool_call(
        initialize_memory_system,
        {"db_path": DEMO_DB_FILE},
        "Initialize Memory System",
    )
    if not init_result or not init_result.get("success"):
        console.print(
            "[bold red]CRITICAL:[/bold red] Failed to initialize memory system. Aborting demo."
        )
        console.print(
            "[yellow]Check DB access and potentially API key configuration/network if init requires them.[/yellow]"
        )
        await cleanup_demo_environment()
        sys.exit(1)


async def cleanup_demo_environment():
    """Close DB connection."""
    global DEMO_DB_FILE
    log_func_info = getattr(logger, "info", print)
    log_func_warn = getattr(logger, "warning", print)

    try:
        await DBConnection.close_connection()
        log_func_info("Closed database connection.")
    except Exception as e:
        log_func_warn(f"Error closing DB connection during cleanup: {e}")

    if DEMO_DB_FILE:
        log_func_info(f"Demo finished using database: {DEMO_DB_FILE}")
        console.print(f"Demo finished using database: [dim]{DEMO_DB_FILE}[/dim]")
        DEMO_DB_FILE = None


# --- Individual Demo Sections ---


async def demonstrate_basic_workflows():
    """Demonstrate basic workflow CRUD and listing operations."""
    console.print(Rule("[bold green]1. Basic Workflow Operations[/bold green]", style="green"))

    # Create
    create_args = {
        "title": "Enhanced WF Demo",
        "goal": "Demonstrate core workflow, action, artifact, and memory linking.",
        "tags": ["enhanced", "demo", "core"],
    }
    wf_result = await safe_tool_call(create_workflow, create_args, "Create Enhanced Workflow")
    wf_id = wf_result.get("workflow_id") if wf_result.get("success") else None

    if not wf_id:
        console.print("[red]Cannot proceed: Failed to create workflow.[/red]")
        return None

    # Get Details
    await safe_tool_call(
        get_workflow_details, {"workflow_id": wf_id}, f"Get Workflow Details ({_fmt_id(wf_id)})"
    )

    # List (should show the one we created)
    await safe_tool_call(list_workflows, {"limit": 5}, "List Workflows (Limit 5)")

    # List Filtered by Tag
    await safe_tool_call(list_workflows, {"tag": "enhanced"}, "List Workflows Tagged 'enhanced'")

    # Update Status (to active for subsequent steps)
    await safe_tool_call(
        update_workflow_status,
        {"workflow_id": wf_id, "status": "active"},
        f"Ensure Workflow Status is Active ({_fmt_id(wf_id)})",
    )

    return wf_id


async def demonstrate_basic_actions(wf_id: Optional[str]):
    """Demonstrate basic action recording, completion, and retrieval."""
    console.print(Rule("[bold green]2. Basic Action Operations[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping action demo: No valid workflow ID provided.[/yellow]")
        return {}  # Return empty dict

    action_ids = {}

    # Record Action 1 Start (e.g., Planning)
    start_args_1 = {
        "workflow_id": wf_id,
        "action_type": ActionType.PLANNING.value,
        "reasoning": "Initial planning phase for the enhanced demo.",
        "title": "Plan Demo Steps",
        "tags": ["planning"],
    }
    action_res_1 = await safe_tool_call(
        record_action_start, start_args_1, "Record Action 1 Start (Planning)"
    )
    action_id_1 = action_res_1.get("action_id") if action_res_1.get("success") else None
    if action_id_1:
        action_ids["action1_id"] = action_id_1

    # Record Action 1 Completion
    complete_args_1 = {
        "action_id": action_id_1,
        "status": ActionStatus.COMPLETED.value,
        "summary": "Planning complete. Next step: data simulation.",
    }
    await safe_tool_call(
        record_action_completion,
        complete_args_1,
        f"Record Action 1 Completion ({_fmt_id(action_id_1)})",
    )

    # Record Action 2 Start (e.g., Tool Use - simulated)
    start_args_2 = {
        "workflow_id": wf_id,
        "action_type": ActionType.TOOL_USE.value,
        "reasoning": "Simulating data generation based on the plan.",
        "tool_name": "simulate_data",
        "tool_args": {"rows": 100, "type": "random"},
        "title": "Simulate Demo Data",
        "tags": ["data", "simulation"],
        "parent_action_id": action_id_1,  # Link to planning action
    }
    action_res_2 = await safe_tool_call(
        record_action_start, start_args_2, "Record Action 2 Start (Simulate Data)"
    )
    action_id_2 = action_res_2.get("action_id") if action_res_2.get("success") else None
    if action_id_2:
        action_ids["action2_id"] = action_id_2

    # Get Action Details (Multiple)
    if action_id_1 and action_id_2:
        await safe_tool_call(
            get_action_details,
            {"action_ids": [action_id_1, action_id_2]},
            "Get Action Details (Multiple Actions)",
        )
    elif action_id_1:
        await safe_tool_call(
            get_action_details,
            {"action_id": action_id_1},
            f"Get Action Details ({_fmt_id(action_id_1)})",
        )

    # Record Action 2 Completion (Failed example)
    complete_args_2 = {
        "action_id": action_id_2,
        "status": ActionStatus.FAILED.value,
        "summary": "Simulation failed due to resource limit.",
        "tool_result": {"error": "Timeout", "code": 504},
    }
    await safe_tool_call(
        record_action_completion,
        complete_args_2,
        f"Record Action 2 Completion (Failed - {_fmt_id(action_id_2)})",
    )

    # Get Recent Actions (should show both)
    await safe_tool_call(
        get_recent_actions, {"workflow_id": wf_id, "limit": 5}, "Get Recent Actions"
    )

    return action_ids


async def demonstrate_action_dependencies(wf_id: Optional[str], action_ids: Dict):
    """Demonstrate adding and retrieving action dependencies."""
    console.print(Rule("[bold green]3. Action Dependency Operations[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping dependency demo: No valid workflow ID.[/yellow]")
        return
    action1_id = action_ids.get("action1_id")
    action2_id = action_ids.get("action2_id")
    if not action1_id or not action2_id:
        console.print("[yellow]Skipping dependency demo: Need at least two action IDs.[/yellow]")
        return

    # Add Dependency (Action 2 requires Action 1)
    await safe_tool_call(
        add_action_dependency,
        {
            "source_action_id": action2_id,
            "target_action_id": action1_id,
            "dependency_type": "requires",
        },
        f"Add Dependency ({_fmt_id(action2_id)} requires {_fmt_id(action1_id)})",
    )

    # Get Dependencies for Action 1 (Should show Action 2 depends on it - Downstream)
    await safe_tool_call(
        get_action_dependencies,
        {"action_id": action1_id, "direction": "downstream"},
        f"Get Dependencies (Downstream of Action 1 - {_fmt_id(action1_id)})",
    )

    # Get Dependencies for Action 2 (Should show it depends on Action 1 - Upstream)
    await safe_tool_call(
        get_action_dependencies,
        {"action_id": action2_id, "direction": "upstream", "include_details": True},
        f"Get Dependencies (Upstream of Action 2 - {_fmt_id(action2_id)}, with Details)",
    )

    # Get Action 1 Details (Include Dependencies)
    await safe_tool_call(
        get_action_details,
        {"action_id": action1_id, "include_dependencies": True},
        f"Get Action 1 Details ({_fmt_id(action1_id)}), Include Dependencies",
    )


async def demonstrate_artifacts(wf_id: Optional[str], action_ids: Dict):
    """Demonstrate artifact recording and retrieval."""
    console.print(Rule("[bold green]4. Artifact Operations[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping artifact demo: No valid workflow ID provided.[/yellow]")
        return {}  # Return empty dict
    action1_id = action_ids.get("action1_id")  # Planning action
    action2_id = action_ids.get("action2_id")  # Failed simulation action

    artifact_ids = {}

    # Record Artifact 1 (e.g., Plan document from Action 1)
    artifact_args_1 = {
        "workflow_id": wf_id,
        "action_id": action1_id,
        "name": "demo_plan.txt",
        "artifact_type": ArtifactType.FILE.value,  # Use enum value
        "description": "Initial plan for the demo steps.",
        "path": "/path/to/demo_plan.txt",
        "content": "Step 1: Plan\nStep 2: Simulate\nStep 3: Analyze",  # Small content example
        "tags": ["planning", "document"],
        "is_output": False,
    }
    art_res_1 = await safe_tool_call(
        record_artifact, artifact_args_1, "Record Artifact 1 (Plan Doc)"
    )
    art_id_1 = art_res_1.get("artifact_id") if art_res_1.get("success") else None
    if art_id_1:
        artifact_ids["art1_id"] = art_id_1

    # Record Artifact 2 (e.g., Error log from Action 2)
    artifact_args_2 = {
        "workflow_id": wf_id,
        "action_id": action2_id,
        "name": "simulation_error.log",
        "artifact_type": ArtifactType.TEXT.value,
        "description": "Error log from the failed data simulation.",
        "content": "ERROR: Timeout waiting for resource. Code 504.",
        "tags": ["error", "log", "simulation"],
    }
    art_res_2 = await safe_tool_call(
        record_artifact, artifact_args_2, "Record Artifact 2 (Error Log)"
    )
    art_id_2 = art_res_2.get("artifact_id") if art_res_2.get("success") else None
    if art_id_2:
        artifact_ids["art2_id"] = art_id_2

    # Get Artifacts (List all for workflow)
    await safe_tool_call(
        get_artifacts, {"workflow_id": wf_id, "limit": 5}, "Get Artifacts (List for Workflow)"
    )

    # Get Artifacts (Filter by tag 'planning')
    await safe_tool_call(
        get_artifacts,
        {"workflow_id": wf_id, "tag": "planning"},
        "Get Artifacts (Filter by Tag 'planning')",
    )

    # Get Artifact by ID (Get the plan doc)
    if art_id_1:
        await safe_tool_call(
            get_artifact_by_id,
            {"artifact_id": art_id_1, "include_content": True},
            f"Get Artifact by ID ({_fmt_id(art_id_1)}, Include Content)",
        )

    return artifact_ids


async def demonstrate_thoughts_and_linking(
    wf_id: Optional[str], action_ids: Dict, artifact_ids: Dict
):
    """Demonstrate thought chains, recording thoughts, and linking them."""
    console.print(Rule("[bold green]5. Thought Operations & Linking[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping thought demo: No valid workflow ID provided.[/yellow]")
        return None
    action1_id = action_ids.get("action1_id")
    art1_id = artifact_ids.get("art1_id")  # Plan artifact

    # Create a new thought chain
    chain_args = {
        "workflow_id": wf_id,
        "title": "Analysis Thought Chain",
        "initial_thought": "Review the plan artifact.",
        "initial_thought_type": ThoughtType.PLAN.value,
    }
    chain_res = await safe_tool_call(
        create_thought_chain, chain_args, "Create New Thought Chain (Analysis)"
    )
    chain_id = chain_res.get("thought_chain_id") if chain_res.get("success") else None

    if not chain_id:
        console.print("[red]Cannot proceed: Failed to create analysis thought chain.[/red]")
        return None

    # Record a thought linked to the plan artifact
    thought_args_1 = {
        "workflow_id": wf_id,
        "thought_chain_id": chain_id,
        "content": "The plan seems straightforward but lacks detail on simulation parameters.",
        "thought_type": ThoughtType.CRITIQUE.value,
        "relevant_artifact_id": art1_id,  # Link to the plan artifact
    }
    thought_res_1 = await safe_tool_call(
        record_thought, thought_args_1, "Record Thought (Critique Linked to Artifact)"
    )
    thought1_id = thought_res_1.get("thought_id") if thought_res_1.get("success") else None

    # Record another thought linked to the failed action
    thought_args_2 = {
        "workflow_id": wf_id,
        "thought_chain_id": chain_id,
        "content": "The simulation failure needs investigation. Was it transient or configuration?",
        "thought_type": ThoughtType.QUESTION.value,
        "relevant_action_id": action_ids.get("action2_id"),  # Link to failed action
        "parent_thought_id": thought1_id,  # Link to previous thought
    }
    await safe_tool_call(
        record_thought, thought_args_2, "Record Thought (Question Linked to Action)"
    )

    # Get the thought chain details (should show linked thoughts)
    await safe_tool_call(
        get_thought_chain,
        {"thought_chain_id": chain_id},
        f"Get Analysis Thought Chain Details ({_fmt_id(chain_id)})",
    )

    return chain_id


async def demonstrate_memory_operations(wf_id: Optional[str], action_ids: Dict, thought_ids: Dict):
    """Demonstrate memory storage, querying, linking."""
    console.print(Rule("[bold green]6. Memory Operations & Querying[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping memory demo: No valid workflow ID provided.[/yellow]")
        return {}  # Return empty dict

    mem_ids = {}

    # Store Memory 1 (Related to Planning Action)
    store_args_1 = {
        "workflow_id": wf_id,
        "action_id": action_ids.get("action1_id"),
        "content": "The initial plan involves simulation and analysis.",
        "memory_type": MemoryType.SUMMARY.value,
        "memory_level": MemoryLevel.EPISODIC.value,
        "description": "Summary of initial plan",
        "tags": ["planning", "summary"],
        "generate_embedding": False,  # NO EMBEDDINGS YET
    }
    mem_res_1 = await safe_tool_call(store_memory, store_args_1, "Store Memory 1 (Plan Summary)")
    mem1_id = mem_res_1.get("memory_id") if mem_res_1.get("success") else None
    if mem1_id:
        mem_ids["mem1_id"] = mem1_id

    # Store Memory 2 (Related to Failed Action)
    store_args_2 = {
        "workflow_id": wf_id,
        "action_id": action_ids.get("action2_id"),
        "content": "Data simulation failed with a timeout error (Code 504).",
        "memory_type": MemoryType.OBSERVATION.value,
        "memory_level": MemoryLevel.EPISODIC.value,
        "description": "Simulation failure detail",
        "importance": 7.0,  # Failed actions might be important
        "tags": ["error", "simulation", "observation"],
        "generate_embedding": False,
    }
    mem_res_2 = await safe_tool_call(
        store_memory, store_args_2, "Store Memory 2 (Simulation Error)"
    )
    mem2_id = mem_res_2.get("memory_id") if mem_res_2.get("success") else None
    if mem2_id:
        mem_ids["mem2_id"] = mem2_id

    # Store Memory 3 (A more general fact)
    store_args_3 = {
        "workflow_id": wf_id,
        "content": "Timeout errors often indicate resource contention or configuration issues.",
        "memory_type": MemoryType.FACT.value,
        "memory_level": MemoryLevel.SEMANTIC.value,
        "description": "General knowledge about timeouts",
        "importance": 6.0,
        "confidence": 0.9,
        "tags": ["error", "knowledge", "fact"],
        "generate_embedding": False,
    }
    mem_res_3 = await safe_tool_call(store_memory, store_args_3, "Store Memory 3 (Timeout Fact)")
    mem3_id = mem_res_3.get("memory_id") if mem_res_3.get("success") else None
    if mem3_id:
        mem_ids["mem3_id"] = mem3_id

    # Link Memory 2 (Error) -> Memory 3 (Fact)
    if mem2_id and mem3_id:
        await safe_tool_call(
            create_memory_link,
            {
                "source_memory_id": mem2_id,
                "target_memory_id": mem3_id,
                "link_type": LinkType.REFERENCES.value,
                "description": "Error relates to general timeout knowledge",
            },
            f"Link Error ({_fmt_id(mem2_id)}) to Fact ({_fmt_id(mem3_id)})",
        )

        # Get Linked Memories for the Error Memory
        await safe_tool_call(
            get_linked_memories,
            {"memory_id": mem2_id, "direction": "outgoing", "include_memory_details": True},
            f"Get Outgoing Linked Memories for Error ({_fmt_id(mem2_id)})",
        )

    # Query Memories using FTS
    await safe_tool_call(
        query_memories,
        {"workflow_id": wf_id, "search_text": "simulation error timeout"},
        "Query Memories (FTS: 'simulation error timeout')",
    )

    # Query Memories by Importance Range
    await safe_tool_call(
        query_memories,
        {"workflow_id": wf_id, "min_importance": 6.5, "sort_by": "importance"},
        "Query Memories (Importance >= 6.5)",
    )

    # Query Memories by Memory Type
    await safe_tool_call(
        query_memories,
        {"workflow_id": wf_id, "memory_type": MemoryType.FACT.value},
        "Query Memories (Type: Fact)",
    )

    # Update Memory 1's tags
    if mem1_id:
        await safe_tool_call(
            update_memory,
            {"memory_id": mem1_id, "tags": ["planning", "summary", "initial_phase"]},
            f"Update Memory 1 Tags ({_fmt_id(mem1_id)})",
        )
        # Verify update
        await safe_tool_call(
            get_memory_by_id,
            {"memory_id": mem1_id},
            f"Get Memory 1 After Tag Update ({_fmt_id(mem1_id)})",
        )

    # Example: Record a thought linked to a memory
    if mem3_id and thought_ids:  # Assuming demonstrate_thoughts ran successfully
        thought_chain_id_str = thought_ids.get("main_chain_id")
        if not thought_chain_id_str:
            console.print("[yellow]Skipping thought link to memory: main_chain_id not found in thought_ids dict.[/yellow]")
        else:
            thought_args_link = {
                "workflow_id": wf_id,
                "thought_chain_id": thought_chain_id_str, # Pass the string ID
                "content": "Based on the general knowledge about timeouts, need to check server logs.",
                "thought_type": ThoughtType.PLAN.value,
                "relevant_memory_id": mem3_id,  # Link to the Fact memory
            }
            await safe_tool_call(
                record_thought,
                thought_args_link,
                f"Record Thought Linked to Memory ({_fmt_id(mem3_id)})",
            )
    elif not thought_ids:
        console.print("[yellow]Skipping thought link to memory: thought_ids dict is empty or None.[/yellow]")

    return mem_ids

async def demonstrate_embedding_and_search(wf_id: Optional[str], mem_ids: Dict, thought_ids: Dict):
    """Demonstrate embedding generation and semantic/hybrid search."""
    console.print(
        Rule("[bold green]7. Embedding & Semantic Search[/bold green]", style="green")
    )
    if not wf_id:
        console.print("[yellow]Skipping embedding demo: No valid workflow ID.[/yellow]")
        return # Return immediately if no workflow ID
    mem1_id = mem_ids.get("mem1_id") # Plan summary
    mem2_id = mem_ids.get("mem2_id") # Simulation error
    mem3_id = mem_ids.get("mem3_id") # Timeout fact

    if not mem1_id or not mem2_id or not mem3_id:
        console.print("[yellow]Skipping embedding demo: Missing required memory IDs from previous steps.[/yellow]")
        return # Return immediately if prerequisite memories are missing

    # 1. Update Memory 2 (Error) to generate embedding
    # This call should now succeed if the update_memory tool code was fixed.
    await safe_tool_call(
        update_memory,
        {
            "memory_id": mem2_id,
            "regenerate_embedding": True,
        },
        f"Update Memory 2 ({_fmt_id(mem2_id)}) to Generate Embedding"
    )

    # 2. Store a new memory WITH embedding generation enabled
    store_args_4 = {
        "workflow_id": wf_id,
        "content": "Investigating the root cause of the simulation timeout is the next priority.",
        "memory_type": MemoryType.PLAN.value,
        "memory_level": MemoryLevel.EPISODIC.value,
        "description": "Next step planning",
        "importance": 7.5,
        "tags": ["investigation", "planning", "error_handling"],
        "generate_embedding": True, # Explicitly enable
        # suggest_links is False by default, keep it simple here
    }
    mem_res_4 = await safe_tool_call(store_memory, store_args_4, "Store Memory 4 (Next Step Plan) with Embedding")
    mem4_id = mem_res_4.get("memory_id") if mem_res_4.get("success") else None
    if mem4_id: mem_ids["mem4_id"] = mem4_id # Add to our tracked IDs

    # Check if embedding was actually generated for Mem4
    if mem4_id:
        # Using suppress_output=True as we only care about the return value here
        mem4_details = await safe_tool_call(
            get_memory_by_id,
            {"memory_id": mem4_id},
            f"Check Memory 4 Details ({_fmt_id(mem4_id)})",
            suppress_output=True
        )
        if mem4_details and mem4_details.get("embedding_id"):
            console.print(f"[green]   -> Embedding ID confirmed for Memory 4: {_fmt_id(mem4_details['embedding_id'])}[/green]")
        else:
            # This might happen if the API key is invalid or embedding service has issues
            console.print(f"[yellow]   -> Warning: Embedding ID missing for Memory 4. Embedding generation likely failed.[/yellow]")
            console.print("[dim]      (Semantic/Hybrid search results may be limited.)[/dim]")


    # 3. Semantic Search (Will only work if embeddings were generated successfully above)
    # Query related to the failure
    await safe_tool_call(
        search_semantic_memories,
        {
            "workflow_id": wf_id,
            "query": "problems with simulation performance",
            "limit": 3,
            "threshold": 0.5 # Lower threshold slightly for demo
        },
        "Semantic Search: 'problems with simulation performance'"
    )

    # Query related to planning
    await safe_tool_call(
        search_semantic_memories,
        {
            "workflow_id": wf_id,
            "query": "next actions to take",
            "limit": 2,
            "memory_level": MemoryLevel.EPISODIC.value # Filter results
        },
        "Semantic Search: 'next actions to take' (Episodic only)"
    )

    # 4. Hybrid Search (Will combine keyword + semantic if embeddings exist)
    # Combining keyword and semantics for the error investigation
    await safe_tool_call(
        hybrid_search_memories,
        {
            "workflow_id": wf_id,
            "query": "investigate timeout simulation", # Mix of keywords
            "limit": 4,
            "semantic_weight": 0.6,
            "keyword_weight": 0.4,
            "tags": ["error"], # Add a filter
            "include_content": False, # Keep hybrid search results concise
        },
        "Hybrid Search: 'investigate timeout simulation' + tag 'error'"
    )

    # 5. Demonstrate link suggestions (Needs embedding for Mem3 and successful embedding for Mem5)
    # Update Mem3 (Timeout fact) to generate embedding
    await safe_tool_call(
        update_memory,
        { "memory_id": mem3_id, "regenerate_embedding": True }, # Use the FIXED update_memory
        f"Update Memory 3 ({_fmt_id(mem3_id)}) to Generate Embedding"
    )

    # --- Store Memory 5 (Corrected Approach) ---
    # First, record the hypothesis as a thought in the appropriate chain
    hypothesis_content = "Resource limits on the simulation server might be too low."
    # Retrieve chain_id safely from the passed dictionary
    # Ensure thought_ids is treated as a dictionary
    thought_chain_id = thought_ids.get("main_chain_id") if isinstance(thought_ids, dict) else None

    hypothesis_thought_id = None
    if thought_chain_id:
        thought_args_hyp = {
            "workflow_id": wf_id,
            "thought_chain_id": thought_chain_id,
            "content": hypothesis_content,
            "thought_type": ThoughtType.HYPOTHESIS.value, # Correct ThoughtType
            "relevant_memory_id": mem3_id # Link to the fact that prompted it
        }
        hyp_thought_res = await safe_tool_call(record_thought, thought_args_hyp, "Record Hypothesis Thought")
        hypothesis_thought_id = hyp_thought_res.get("thought_id") if hyp_thought_res.get("success") else None
    else:
        console.print("[yellow]Skipping hypothesis memory storage: Could not get thought chain ID.[/yellow]")

    # Now, optionally store this important reasoning step as a memory
    mem5_id = None
    mem_res_5 = None # Initialize mem_res_5
    if hypothesis_thought_id:
        store_args_5 = {
            "workflow_id": wf_id,
            "thought_id": hypothesis_thought_id, # Link memory back to the thought
            "content": hypothesis_content, # Store the hypothesis content
            "memory_type": MemoryType.REASONING_STEP.value, # Use REASONING_STEP instead of HYPOTHESIS
            "memory_level": MemoryLevel.SEMANTIC.value, # Store hypothesis reasoning as semantic
            "description": "Hypothesis on timeout cause (reasoning step)", # Clarify description
            "importance": 6.5,
            "confidence": 0.6, # Tentative confidence
            "tags": ["hypothesis", "resource", "error", "reasoning_step"], # Add appropriate tags
            "generate_embedding": True, # Try embedding the hypothesis text
            "suggest_links": True, # Explicitly ask for suggestions
            "max_suggested_links": 2, # Limit suggestions for demo clarity
        }
        mem_res_5 = await safe_tool_call(store_memory, store_args_5, "Store Memory 5 (Hypothesis Reasoning Step) - Suggest Links")
        mem5_id = mem_res_5.get("memory_id") if mem_res_5.get("success") else None
        if mem5_id: mem_ids["mem5_id"] = mem5_id # Add to tracked IDs only if successful

        # Check suggestions result
        if mem_res_5 and mem_res_5.get("success") and mem_res_5.get("suggested_links"):
            console.print(f"[cyan]   -> Link suggestions received for Memory 5:[/]")
            console.print(pretty_repr(mem_res_5["suggested_links"]))
        elif mem_res_5 and mem_res_5.get("success"):
             console.print(f"[dim]   -> No link suggestions returned for Memory 5.[/dim]")
        elif mem_res_5 and not mem_res_5.get("success"):
             console.print(f"[yellow]   -> Failed to store Memory 5, cannot check suggestions.[/yellow]")

    else:
        # Log if thought recording failed
        console.print("[yellow]Skipping Memory 5 storage: Hypothesis thought recording failed.[/yellow]")


async def demonstrate_state_and_working_memory(
    wf_id: str,
    mem_ids_dict: Dict[str, str],
    action_ids_dict: Dict[str, str],
    thought_ids_list: List[str],
    state_ids_dict: Dict[str, str] # Pass dict by reference to store state_id
):
    """Demonstrate saving/loading state and working memory operations."""
    console.print(
        Rule("[bold green]8. Cognitive State & Working Memory[/bold green]", style="green")
    )

    # Prepare IDs for saving state - use a subset of available memories
    working_mems = [
        mem_id for mem_id in [
            mem_ids_dict.get("mem2_id"), # Simulation error
            mem_ids_dict.get("mem3_id"), # Timeout fact
            mem_ids_dict.get("mem4_id"), # Next step plan
            mem_ids_dict.get("mem5_id")  # Hypothesis reasoning step
        ] if mem_id # Filter out None values if a memory wasn't created
    ]
    focus_mems = [mem_ids_dict.get("mem4_id")] if mem_ids_dict.get("mem4_id") else [] # Focus on the 'next step' memory
    context_actions = [
        action_id for action_id in [
            action_ids_dict.get("action1_id"), # Planning
            action_ids_dict.get("action2_id") # Failed simulation
        ] if action_id
    ]
    goal_thoughts = [thought_id for thought_id in thought_ids_list if thought_id] # Use the hypothesis thought as a 'current goal' proxy

    # 1. Save Cognitive State
    save_args = {
        "workflow_id": wf_id,
        "title": "State after simulation failure and hypothesis",
        "working_memory_ids": working_mems,
        "focus_area_ids": focus_mems,
        "context_action_ids": context_actions,
        "current_goal_thought_ids": goal_thoughts
    }
    state_res = await safe_tool_call(save_cognitive_state, save_args, "Save Cognitive State")
    state_id = state_res.get("state_id") if state_res.get("success") else None
    if state_id:
        state_ids_dict['saved_state_id'] = state_id # Store the ID in the passed dict
    else:
        console.print("[red]Cannot proceed with working memory demo: Failed to save state.[/red]")
        return

    # 2. Load Cognitive State (by ID)
    await safe_tool_call(
        load_cognitive_state,
        {"workflow_id": wf_id, "state_id": state_id},
        f"Load Cognitive State ({_fmt_id(state_id)}) by ID",
    )

    # 3. Load Cognitive State (Latest)
    await safe_tool_call(
        load_cognitive_state,
        {"workflow_id": wf_id}, # No state_id means load latest
        "Load Latest Cognitive State",
    )

    # --- Working Memory Operations using the saved state_id as the context_id ---
    context_id_for_demo = state_id
    console.print(f"\n[dim]Using saved state ID '{_fmt_id(context_id_for_demo)}' as context_id for working memory tests...[/dim]\n")

    # 4. Focus Memory (Focus on the 'hypothesis' memory if it exists)
    focus_target_id = mem_ids_dict.get("mem5_id")
    if focus_target_id:
        await safe_tool_call(
            focus_memory,
            {
                "memory_id": focus_target_id,
                "context_id": context_id_for_demo,
                "add_to_working": False # Assume it's already there from save_state
            },
            f"Focus Memory ({_fmt_id(focus_target_id)}) in Context ({_fmt_id(context_id_for_demo)})",
        )
    else:
         console.print("[yellow]Skipping focus memory test: Hypothesis memory ID not available.[/yellow]")

    # 5. Get Working Memory (Should reflect the saved state initially)
    await safe_tool_call(
        get_working_memory,
        {
            "context_id": context_id_for_demo,
            "include_links": False # Keep output cleaner for this demo step
        },
        f"Get Working Memory for Context ({_fmt_id(context_id_for_demo)})",
    )

    # 6. Optimize Working Memory (Reduce size, using 'balanced' strategy)
    # Check current size before optimizing
    wm_details = await safe_tool_call(
        get_working_memory,
        {"context_id": context_id_for_demo},
        "Get WM Size Before Optimization",
        suppress_output=True
    )
    current_wm_size = len(wm_details.get("working_memories", [])) if wm_details and wm_details.get("success") else 0

    if current_wm_size > 2: # Only optimize if we have more than 2 memories
        target_optimize_size = max(1, current_wm_size // 2) # Aim to reduce size significantly
        console.print(f"[cyan]   -> Optimizing working memory from {current_wm_size} down to {target_optimize_size}...[/cyan]")
        await safe_tool_call(
            # Import optimize_working_memory tool
            # from ultimate_mcp_server.tools.unified_memory_system import optimize_working_memory
            optimize_working_memory,
            {
                "context_id": context_id_for_demo,
                "target_size": target_optimize_size,
                "strategy": "balanced"
            },
            f"Optimize Working Memory (Context: {_fmt_id(context_id_for_demo)}, Target: {target_optimize_size})",
        )

        # Get Working Memory again to show the result
        await safe_tool_call(
            get_working_memory,
            {
                "context_id": context_id_for_demo,
                "include_links": False
            },
            f"Get Working Memory After Optimization (Context: {_fmt_id(context_id_for_demo)})",
        )
    else:
        console.print(f"[dim]Skipping working memory optimization: Current size ({current_wm_size}) is too small.[/dim]")

# --- Main Execution Logic ---
async def main():
    """Run the extended Unified Memory System demonstration suite."""
    console.print(
        Rule(
            "[bold magenta]Unified Memory System Tools Demo (Extended Basics)[/bold magenta]",
            style="white",
        )
    )
    exit_code = 0
    # Dictionaries to store IDs created during the demo
    wf_ids = {}
    action_ids = {}
    artifact_ids = {}
    thought_ids = {}  # Store chain ID
    mem_ids = {}

    try:
        await setup_demo_environment()

        # --- Run Demo Sections in Order ---
        wf_id = await demonstrate_basic_workflows()
        if wf_id:
            wf_ids["main_wf_id"] = wf_id

        action_ids = await demonstrate_basic_actions(wf_ids.get("main_wf_id"))

        await demonstrate_action_dependencies(wf_ids.get("main_wf_id"), action_ids)

        artifact_ids = await demonstrate_artifacts(wf_ids.get("main_wf_id"), action_ids)

        chain_id = await demonstrate_thoughts_and_linking(
            wf_ids.get("main_wf_id"), action_ids, artifact_ids
        )
        if chain_id:
            thought_ids["main_chain_id"] = chain_id

        mem_ids = await demonstrate_memory_operations(wf_ids.get("main_wf_id"), action_ids, thought_ids) # Pass thought_ids dict

        await demonstrate_embedding_and_search(wf_ids.get("main_wf_id"), mem_ids, thought_ids) # Pass thought_ids dictionary here

 # --- Retrieve necessary IDs from previous steps ---
        # Ensure these dictionaries are populated from previous steps
        main_wf_id = wf_ids.get("main_wf_id")
        main_chain_id = thought_ids.get("main_chain_id") # Analysis chain
        plan_action_id = action_ids.get("action1_id")
        sim_action_id = action_ids.get("action2_id") # Failed simulation
        mem1_id = mem_ids.get("mem1_id") # Plan summary
        mem2_id = mem_ids.get("mem2_id") # Simulation error
        mem3_id = mem_ids.get("mem3_id") # Timeout fact
        mem4_id = mem_ids.get("mem4_id") # Next step plan
        mem5_id = mem_ids.get("mem5_id") # Hypothesis reasoning step

        # Need the ID of the hypothesis thought itself
        hypothesis_thought_id = None
        if mem5_id and main_wf_id:
            # Get the memory details to find the linked thought ID
            mem5_details = await safe_tool_call(
                get_memory_by_id,
                {"memory_id": mem5_id},
                f"Get Memory 5 Details ({_fmt_id(mem5_id)}) for Thought ID",
                suppress_output=True # We just need the data
            )
            if mem5_details and mem5_details.get("success"):
                # --- FIX: Use the correct key 'thought_id' from the DB row ---
                hypothesis_thought_id = mem5_details.get("thought_id")
                # --- END FIX ---
                if hypothesis_thought_id:
                     console.print(f"[cyan]   -> Retrieved Hypothesis Thought ID: {_fmt_id(hypothesis_thought_id)}[/cyan]")
                else:
                    # Update warning slightly for clarity
                    console.print("[yellow]   -> Could not retrieve hypothesis thought ID from Memory 5 details (key 'thought_id' was missing or None).[/yellow]")

        # --- 8. Demonstrate Cognitive State & Working Memory ---
        # Check if we have enough data to proceed
        state_ids = {} # To store the state ID created
        if main_wf_id and mem1_id and mem2_id and mem3_id and mem4_id and plan_action_id and hypothesis_thought_id:
             await demonstrate_state_and_working_memory(
                 wf_id=main_wf_id,
                 mem_ids_dict=mem_ids, # Pass the whole dict
                 action_ids_dict=action_ids,
                 thought_ids_list=[hypothesis_thought_id], # Pass the specific goal/hypothesis thought
                 state_ids_dict=state_ids # Pass dict to store the created state_id
             )
        else:
            console.print(
                Rule(
                    "[bold yellow]8. Cognitive State & Working Memory Skipped[/bold yellow]",
                    style="yellow",
                )
            )
            console.print("[yellow]Skipping state/working memory demo: Missing required IDs from previous steps.[/yellow]")


        # --- Placeholder for remaining ADVANCED demos ---
        console.print(
            Rule(
                "[bold yellow]Advanced Demo Sections (Meta-Cognition, Maintenance, Reporting) Skipped[/bold yellow]",
                style="yellow",
            )
        )
        # await demonstrate_metacognition(...)
        # await demonstrate_maintenance_and_stats(...)
        # await demonstrate_reporting_and_viz(...)

        logger.success(
            "Extended Basic Unified Memory System Demo completed successfully!",
            emoji_key="complete",
        )
        console.print(Rule("[bold green]Extended Basic Demo Finished[/bold green]", style="green"))

    except Exception as e:
        logger.critical(f"Demo crashed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        exit_code = 1

    finally:
        # Clean up the demo environment
        console.print(Rule("Cleanup", style="dim"))
        await cleanup_demo_environment()

    return exit_code


if __name__ == "__main__":
    # Run the demo
    final_exit_code = asyncio.run(main())
    sys.exit(final_exit_code)
