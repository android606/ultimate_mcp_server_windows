# examples/advanced_unified_memory_system_demo.py
#!/usr/bin/env python
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

# --- Project Setup ---
# Add project root to path for imports when running as script
# Adjust this path if your script location relative to the project root differs
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # Assuming this script is in examples/
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # Verify path
    if not (PROJECT_ROOT / "ultimate_mcp_server").is_dir():
        print(
            f"Warning: Could not reliably find project root from {SCRIPT_DIR}. Imports might fail.",
            file=sys.stderr,
        )

except Exception as e:
    print(f"Error setting up sys.path: {e}", file=sys.stderr)
    sys.exit(1)

# --- Rich Imports ---
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.rule import Rule
from rich.traceback import install as install_rich_traceback

from ultimate_mcp_server.config import get_config  # Load config for defaults

# --- Tool Imports (Specific functions needed) ---
from ultimate_mcp_server.tools.unified_memory_system import (
    ActionStatus,
    ActionType,
    ArtifactType,  # <<< Added missing import
    DBConnection,
    MemoryLevel,
    MemoryType,
    ThoughtType,
    ToolError,  # Explicitly import errors
    ToolInputError,
    # Enums & Helpers
    WorkflowStatus,
    add_action_dependency,
    auto_update_focus,
    consolidate_memories,
    # Workflows
    create_workflow,
    focus_memory,  # Needed for Ext 4 test
    # Meta-Cognition
    generate_reflection,
    # Reporting
    generate_workflow_report,
    get_memory_by_id,
    get_working_memory,
    initialize_memory_system,  # Need this for setup
    load_cognitive_state,
    optimize_working_memory,
    promote_memory_level,
    query_memories,
    record_action_completion,
    record_action_start,
    record_artifact,
    record_thought,
    # State & Focus
    save_cognitive_state,
    search_semantic_memories,
    # Core Memory & Search
    store_memory,
    update_workflow_status,
)

# Utilities from the project
from ultimate_mcp_server.utils import get_logger

console = Console()
logger = get_logger("demo.advanced_memory")
config = get_config()  # Load config

# Use a dedicated DB file for this advanced demo
DEMO_DB_FILE_ADVANCED = str(Path("./advanced_demo_memory.db").resolve())
_current_db_path = None  # Track the active DB path for safe_tool_call

install_rich_traceback(show_locals=False, width=console.width)


# --- Safe Tool Call Helper (Adapted) ---
async def safe_tool_call(func, args: Dict, description: str, suppress_output: bool = False):
    """Helper to call a tool function, catch errors, and display results."""
    global _current_db_path  # Use the tracked path
    display_title = not suppress_output
    display_args = not suppress_output
    display_result_panel = not suppress_output

    if display_title:
        title = f"ADV_DEMO: {description}"
        console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]", style="blue"))
    if display_args:
        # Filter out db_path if it matches the global demo path
        args_to_print = {k: v for k, v in args.items() if k != "db_path" or v != _current_db_path}
        args_repr = pretty_repr(args_to_print, max_length=120, max_string=100)
        console.print(f"[dim]Calling [bold cyan]{func.__name__}[/] with args:[/]\n{args_repr}")

    start_time = time.monotonic()
    result = None
    try:
        # Inject the correct db_path if not explicitly provided
        if "db_path" not in args and _current_db_path:
            args["db_path"] = _current_db_path

        result = await func(**args)
        processing_time = time.monotonic() - start_time
        logger.debug(f"Tool '{func.__name__}' execution time: {processing_time:.4f}s")

        if display_result_panel:
            success = isinstance(result, dict) and result.get("success", False)
            panel_title = f"[bold {'green' if success else 'yellow'}]Result: {func.__name__} {'✅' if success else '❔'}[/]"
            panel_border = "green" if success else "yellow"

            # Simple repr for most results in advanced demo
            try:
                result_repr = pretty_repr(result, max_length=180, max_string=120)
            except Exception:
                result_repr = f"(Could not represent result of type {type(result)} fully)\n{str(result)[:500]}"

            console.print(
                Panel(
                    escape(result_repr), title=panel_title, border_style=panel_border, expand=False
                )
            )

        return result

    except (ToolInputError, ToolError) as e:
        processing_time = time.monotonic() - start_time
        logger.error(f"Tool '{func.__name__}' failed: {e}", exc_info=False)
        if display_result_panel:
            error_title = f"[bold red]Error: {func.__name__} Failed ❌[/]"
            error_content = f"[bold red]{type(e).__name__}:[/] {escape(str(e))}"
            details = getattr(e, "details", None) or getattr(e, "context", None)
            if details:
                error_content += f"\n\n[yellow]Details:[/]\n{escape(pretty_repr(details))}"
            console.print(Panel(error_content, title=error_title, border_style="red", expand=False))
        # Ensure the returned error dict matches the structure expected by asserts/checks
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
        logger.critical(f"Unexpected error calling '{func.__name__}': {e}", exc_info=True)
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


# --- Demo Setup & Teardown (Using new DB file) ---
async def setup_advanced_demo():
    """Initialize the memory system using the ADVANCED demo database file."""
    global _current_db_path
    _current_db_path = DEMO_DB_FILE_ADVANCED
    logger.info(f"Using dedicated database for advanced demo: {_current_db_path}")

    # Delete existing advanced demo DB file for a clean run
    if Path(_current_db_path).exists():
        try:
            Path(_current_db_path).unlink()
            logger.info(f"Removed existing advanced demo database: {_current_db_path}")
        except OSError as e:
            logger.error(f"Failed to remove existing advanced demo database: {e}")
            # Continue anyway, initialization might handle it

    console.print(
        Panel(
            f"Using database: [cyan]{_current_db_path}[/]\n"
            f"[yellow]NOTE:[/yellow] This demo operates on a separate database file.",
            title="Advanced Demo Setup",
            border_style="yellow",
        )
    )

    # Initialize the memory system with the specific path
    init_result = await safe_tool_call(
        initialize_memory_system,
        {"db_path": _current_db_path},  # Pass explicitly
        "Initialize Advanced Memory System",
    )
    if not init_result or not init_result.get("success"):
        console.print(
            "[bold red]CRITICAL:[/bold red] Failed to initialize advanced memory system. Aborting."
        )
        await cleanup_advanced_demo()
        sys.exit(1)


async def cleanup_advanced_demo():
    """Close DB connection and optionally delete the demo DB."""
    global _current_db_path
    try:
        await DBConnection.close_connection()
        logger.info("Closed database connection.")
    except Exception as e:
        logger.warning(f"Error closing DB connection during cleanup: {e}")

    if _current_db_path:
        logger.info(f"Advanced demo finished using database: {_current_db_path}")
        # # Optionally delete the DB file after run (Uncomment to enable deletion)
        # delete_db = True
        # if delete_db and Path(_current_db_path).exists():
        #     try:
        #         Path(_current_db_path).unlink()
        #         logger.info(f"Removed advanced demo database: {_current_db_path}")
        #     except OSError as e:
        #         logger.error(f"Failed to remove advanced demo database: {e}")
        _current_db_path = None


# --- Extension Implementations ---


async def run_extension_1_goal_decomposition():
    """Extension 1: Goal Decomposition, Execution, and Synthesis"""
    console.print(
        Rule(
            "[bold green]Extension 1: Goal Decomposition, Execution, Synthesis[/bold green]",
            style="green",
        )
    )
    wf_id = None
    planning_action_id = None
    action1_id, action2_id, action3_id, action4_id = None, None, None, None
    artifact_search_id = None
    consolidated_memory_id = None
    final_artifact_id = None

    try:
        # --- Workflow Setup ---
        wf_res = await safe_tool_call(
            create_workflow,
            {
                "title": "Research Report: Future of Renewable Energy",
                "goal": "Research and write a short report on the future of renewable energy, covering trends, challenges, and synthesis.",
                "tags": ["research", "report", "energy"],
            },
            "Create Report Workflow",
        )
        assert wf_res and wf_res.get("success"), "Failed to create workflow"
        wf_id = wf_res["workflow_id"]
        console.print(f"[cyan]  Workflow ID: {wf_id}[/cyan]")

        # --- Planning Phase ---
        plan_start_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.PLANNING.value,
                "reasoning": "Define the steps needed to generate the report.",
                "title": "Plan Report Generation",
                "tags": ["planning"],
            },
            "Start Planning Action",
        )
        assert plan_start_res and plan_start_res.get("success"), "Failed to start planning action"
        planning_action_id = plan_start_res["action_id"]

        # Record plan thoughts (linked to planning action)
        plan_steps = [
            "Research current trends in renewable energy.",
            "Analyze challenges and obstacles.",
            "Synthesize findings from research and analysis.",
            "Draft the final report.",
        ]
        parent_tid = None
        for i, step_content in enumerate(plan_steps):
            thought_res = await safe_tool_call(
                record_thought,
                {
                    "workflow_id": wf_id,
                    "content": step_content,
                    "thought_type": ThoughtType.PLAN.value,
                    "thought_chain_id": wf_res["primary_thought_chain_id"],
                    "parent_thought_id": parent_tid,
                    "relevant_action_id": planning_action_id,
                },
                f"Record Plan Thought {i + 1}",
                suppress_output=True,
            )
            assert thought_res and thought_res.get("success"), (
                f"Failed to record plan thought {i + 1}"
            )
            parent_tid = thought_res["thought_id"]  # Chain thoughts

        # Record planned actions (placeholders, will be "executed" later)
        action_plan_details = [
            {
                "title": "Research Trends",
                "type": ActionType.RESEARCH.value,
                "reasoning": "Plan: Gather data on current renewable energy trends.",
            },
            {
                "title": "Analyze Challenges",
                "type": ActionType.ANALYSIS.value,
                "reasoning": "Plan: Identify obstacles based on gathered data.",
            },
            {
                "title": "Synthesize Findings",
                "type": ActionType.REASONING.value,
                "reasoning": "Plan: Combine trends and challenges into a coherent summary.",
            },
            {
                "title": "Draft Report",
                "type": ActionType.TOOL_USE.value,
                "tool_name": "generate_text",
                "reasoning": "Plan: Write the final report using synthesized findings.",
            },
        ]
        action_ids = []
        for details in action_plan_details:
            # Record the *start* of the planned action, but don't complete it yet.
            action_res = await safe_tool_call(
                record_action_start,
                {
                    "workflow_id": wf_id,
                    "action_type": details["type"],
                    "title": details["title"],
                    "reasoning": details["reasoning"],  # Use specific reasoning
                    "tool_name": details.get("tool_name"),
                    "parent_action_id": planning_action_id,
                    "tags": ["planned_step"],
                },
                f"Record Planned Action: {details['title']}",
                suppress_output=True,
            )
            assert action_res and action_res.get("success"), (
                f"Failed to record planned action {details['title']}"
            )
            action_ids.append(action_res["action_id"])
            # <<< FIX 1: Removed the immediate completion call with invalid status >>>

        action1_id, action2_id, action3_id, action4_id = action_ids

        # Add dependencies between planned actions
        await safe_tool_call(
            add_action_dependency,
            {
                "source_action_id": action2_id,
                "target_action_id": action1_id,
                "dependency_type": "requires",
            },
            "Link Action 2->1",
            suppress_output=True,
        )
        await safe_tool_call(
            add_action_dependency,
            {
                "source_action_id": action3_id,
                "target_action_id": action2_id,
                "dependency_type": "requires",
            },
            "Link Action 3->2",
            suppress_output=True,
        )
        await safe_tool_call(
            add_action_dependency,
            {
                "source_action_id": action4_id,
                "target_action_id": action3_id,
                "dependency_type": "requires",
            },
            "Link Action 4->3",
            suppress_output=True,
        )

        # Complete the main planning action now that placeholders are recorded
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": planning_action_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": "Planning steps recorded.",
            },
            "Complete Planning Action",
        )

        # --- Execution Phase ---
        console.print(Rule("Execution Phase", style="cyan"))

        # Step 1: Execute Research Trends (Simulated Tool Use)
        action1_exec_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.TOOL_USE.value,
                "title": "Execute Research Trends",
                "reasoning": "Performing web search for trends.",
                "tool_name": "simulated_web_search",
                "tags": ["execution"],
                "parent_action_id": action1_id,
            },  # Link to original planned action
            "Start Research Action Execution",
        )
        action1_exec_id = action1_exec_res["action_id"]
        simulated_search_results = "Solar efficiency is increasing rapidly due to perovskite technology. Wind power costs continue to decrease, especially offshore. Battery storage remains a key challenge for grid stability but costs are falling. Geothermal energy is gaining traction for baseload power."
        art1_res = await safe_tool_call(
            record_artifact,
            {
                "workflow_id": wf_id,
                "action_id": action1_exec_id,
                "name": "renewable_trends_search.txt",
                "artifact_type": ArtifactType.TEXT.value,  # <<< FIX 2: Use imported enum
                "content": simulated_search_results,
                "tags": ["research_data"],
            },
            "Record Search Results Artifact",
        )
        artifact_search_id = art1_res["artifact_id"]  # noqa: F841
        mem1_res = await safe_tool_call(  # noqa: F841
            store_memory,
            {
                "workflow_id": wf_id,
                "action_id": action1_exec_id,
                "memory_type": MemoryType.OBSERVATION.value,
                "content": f"Key findings from trends research: {simulated_search_results}",
                "description": "Summary of renewable trends",
                "tags": ["trends", "research"],
                "importance": 7.0,
            },
            "Store Research Findings Memory",
        )
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action1_exec_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": "Web search completed.",
            },
            "Complete Research Action Execution",
        )
        # Mark the original planned action as completed too
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action1_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": "Executed as action " + action1_exec_id,
            },
            "Mark Planned Research Action as Completed",
            suppress_output=True,
        )

        # Step 2: Execute Analyze Challenges
        action2_exec_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.ANALYSIS.value,
                "title": "Execute Analyze Challenges",
                "reasoning": "Analyzing search results for challenges.",
                "tags": ["execution"],
                "parent_action_id": action2_id,
            },  # Link to original planned action
            "Start Analysis Action Execution",
        )
        action2_exec_id = action2_exec_res["action_id"]
        thought_challenge_res = await safe_tool_call(  # noqa: F841
            record_thought,
            {
                "workflow_id": wf_id,
                "thought_chain_id": wf_res["primary_thought_chain_id"],
                "content": "Based on trends, major challenge seems to be grid integration for intermittent sources and cost-effective, large-scale energy storage.",
                "thought_type": ThoughtType.HYPOTHESIS.value,
                "relevant_action_id": action2_exec_id,
            },
            "Record Challenge Hypothesis Thought",
        )
        mem2_res = await safe_tool_call(  # noqa: F841
            store_memory,
            {
                "workflow_id": wf_id,
                "action_id": action2_exec_id,
                "memory_type": MemoryType.INSIGHT.value,
                "content": "Grid integration and energy storage are primary hurdles for widespread renewable adoption, despite falling generation costs.",
                "description": "Key challenges identified",
                "tags": ["challenges", "insight"],
                "importance": 8.0,
            },
            "Store Challenge Insight Memory",
        )
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action2_exec_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": "Analysis of challenges complete.",
            },
            "Complete Analysis Action Execution",
        )
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action2_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": "Executed as action " + action2_exec_id,
            },
            "Mark Planned Analysis Action as Completed",
            suppress_output=True,
        )

        # Step 3: Execute Synthesize Findings
        action3_exec_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.REASONING.value,
                "title": "Execute Synthesize Findings",
                "reasoning": "Combining research and analysis memories.",
                "tags": ["execution"],
                "parent_action_id": action3_id,
            },  # Link to original planned action
            "Start Synthesis Action Execution",
        )
        action3_exec_id = action3_exec_res["action_id"]
        # Query memories stored by the execution actions
        query_res_obs = await safe_tool_call(
            query_memories,
            {
                "workflow_id": wf_id,
                "memory_type": MemoryType.OBSERVATION.value,
                "action_id": action1_exec_id,
                "limit": 5,
            },
            "Query Observation Memories for Synthesis",
        )
        query_res_insight = await safe_tool_call(
            query_memories,
            {
                "workflow_id": wf_id,
                "memory_type": MemoryType.INSIGHT.value,
                "action_id": action2_exec_id,
                "limit": 5,
            },
            "Query Insight Memories for Synthesis",
        )
        mem_ids_to_consolidate = [m["memory_id"] for m in query_res_obs.get("memories", [])] + [
            m["memory_id"] for m in query_res_insight.get("memories", [])
        ]
        assert len(mem_ids_to_consolidate) >= 2, (
            f"Expected at least 2 memories to consolidate, found {len(mem_ids_to_consolidate)}"
        )

        consolidation_res = await safe_tool_call(
            consolidate_memories,
            {
                "workflow_id": wf_id,
                "target_memories": mem_ids_to_consolidate,
                "consolidation_type": "summary",
                "store_result": True,
            },
            "Consolidate Findings",
        )
        assert consolidation_res and consolidation_res.get("success"), "Consolidation failed"
        consolidated_memory_id = consolidation_res["stored_memory_id"]
        assert consolidated_memory_id, "Consolidation did not return a stored memory ID"
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action3_exec_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": f"Consolidated research and analysis into memory {consolidated_memory_id[:8]}.",
            },
            "Complete Synthesis Action Execution",
        )
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action3_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": "Executed as action " + action3_exec_id,
            },
            "Mark Planned Synthesis Action as Completed",
            suppress_output=True,
        )

        # Step 4: Execute Draft Report
        action4_exec_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.TOOL_USE.value,
                "title": "Execute Draft Report",
                "reasoning": "Generating report draft using consolidated summary.",
                "tool_name": "simulated_generate_text",
                "tags": ["execution", "reporting"],
                "parent_action_id": action4_id,
            },  # Link to original planned action
            "Start Drafting Action Execution",
        )
        action4_exec_id = action4_exec_res["action_id"]
        consolidated_mem_details = await safe_tool_call(
            get_memory_by_id,
            {"memory_id": consolidated_memory_id},
            "Fetch Consolidated Memory",
            suppress_output=True,
        )
        consolidated_content = consolidated_mem_details.get(
            "content", "Error fetching consolidated content."
        )
        # Simulate text generation based on consolidated content
        simulated_draft = f"""# The Future of Renewable Energy: A Brief Report

## Consolidated Findings
{consolidated_content}

## Conclusion
The trajectory for renewable energy shows promise with falling costs and improving tech (solar, wind). However, significant investment in grid modernization and energy storage solutions is paramount to overcome intermittency challenges and enable widespread adoption. Geothermal offers potential for stable baseload power.
"""
        art2_res = await safe_tool_call(
            record_artifact,
            {
                "workflow_id": wf_id,
                "action_id": action4_exec_id,
                "name": "renewable_report_draft.md",
                "artifact_type": ArtifactType.TEXT.value,  # <<< FIX 2: Use imported enum
                "content": simulated_draft,
                "is_output": True,
                "tags": ["report", "draft", "output"],
            },
            "Record Final Report Artifact",
        )
        final_artifact_id = art2_res["artifact_id"]
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action4_exec_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": f"Draft report artifact created: {final_artifact_id[:8]}.",
            },
            "Complete Drafting Action Execution",
        )
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action4_id,
                "status": ActionStatus.COMPLETED.value,
                "summary": "Executed as action " + action4_exec_id,
            },
            "Mark Planned Drafting Action as Completed",
            suppress_output=True,
        )

        # --- Completion & Reporting ---
        console.print(Rule("Workflow Completion & Reporting", style="cyan"))
        await safe_tool_call(
            update_workflow_status,
            {
                "workflow_id": wf_id,
                "status": WorkflowStatus.COMPLETED.value,
                "completion_message": "Report generated successfully.",
            },
            "Mark Workflow Completed",
        )
        await safe_tool_call(
            generate_workflow_report,
            {
                "workflow_id": wf_id,
                "report_format": "markdown",
                "style": "professional",
                "include_thoughts": True,
                "include_artifacts": True,
            },
            "Generate Final Workflow Report",
        )

    except AssertionError as e:
        logger.error(f"Assertion failed during Extension 1: {e}", exc_info=True)
        console.print(f"[bold red]Assertion Failed:[/bold red] {e}")
    except Exception as e:
        logger.error(f"Error in Extension 1: {e}", exc_info=True)
        console.print(f"[bold red]Error in Extension 1:[/bold red] {e}")
    finally:
        console.print(Rule("Extension 1 Finished", style="green"))


async def run_extension_2_dynamic_adaptation():
    """Extension 2: Dynamic Adaptation Based on Reflection"""
    console.print(
        Rule(
            "[bold green]Extension 2: Dynamic Adaptation Based on Reflection[/bold green]",
            style="green",
        )
    )
    wf_id = None
    action1_id, action2_id, action3_id, action4_id, action5_id = None, None, None, None, None
    error_memory_id = None

    try:
        # --- Setup ---
        wf_res = await safe_tool_call(
            create_workflow,
            {
                "title": "Optimize Python Function",
                "goal": "Improve performance of a sample Python function.",
            },
            "Create Optimization Workflow",
        )
        assert wf_res and wf_res.get("success"), "Failed to create workflow"
        wf_id = wf_res["workflow_id"]

        # --- Initial Actions ---
        # Action 1: Analyze performance
        act1_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.ANALYSIS.value,
                "title": "Analyze function performance",
                "reasoning": "Establish baseline performance metrics.",
            },
            "Start Analysis Action",
        )
        action1_id = act1_res["action_id"]
        await safe_tool_call(
            record_artifact,
            {
                "workflow_id": wf_id,
                "action_id": action1_id,
                "name": "profile.data",
                "artifact_type": ArtifactType.DATA.value,
            },  # <<< FIX 2: Use imported enum
            "Record Profiling Artifact",
            suppress_output=True,
        )
        await safe_tool_call(
            record_action_completion,
            {"action_id": action1_id, "status": ActionStatus.COMPLETED.value},
            "Complete Analysis Action",
        )

        # Action 2: Attempt Optimization 1 (Vectorization)
        act2_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.TOOL_USE.value,
                "title": "Attempt optimization 1 (Vectorization)",
                "tool_name": "modify_code",
                "reasoning": "Try vectorization approach for potential speedup.",
            },
            "Start Optimization 1 Action",
        )
        action2_id = act2_res["action_id"]
        await safe_tool_call(
            record_artifact,
            {
                "workflow_id": wf_id,
                "action_id": action2_id,
                "name": "optimized_v1.py",
                "artifact_type": ArtifactType.CODE.value,
            },  # <<< FIX 2: Use imported enum
            "Record Opt 1 Artifact",
            suppress_output=True,
        )
        await safe_tool_call(
            record_action_completion,
            {"action_id": action2_id, "status": ActionStatus.COMPLETED.value},
            "Complete Optimization 1 Action",
        )

        # Action 3: Test Optimization 1 (Failure)
        act3_res = await safe_tool_call(
            record_action_start,
            {
                "workflow_id": wf_id,
                "action_type": ActionType.TOOL_USE.value,
                "title": "Test optimization 1",
                "tool_name": "run_tests",
                "reasoning": "Verify vectorization attempt correctness and performance.",
            },
            "Start Test 1 Action",
        )
        action3_id = act3_res["action_id"]
        error_result = {
            "error": "ValueError: Array dimensions mismatch",
            "traceback": "Traceback details...",
        }
        mem_res = await safe_tool_call(
            store_memory,
            {
                "workflow_id": wf_id,
                "action_id": action3_id,
                "memory_type": MemoryType.OBSERVATION.value,
                "content": f"Test failed for optimization 1 (Vectorization): {error_result['error']}",
                "description": "Vectorization test failure",
                "tags": ["error", "test", "vectorization"],
                "importance": 8.0,
            },
            "Store Failure Observation Memory",
        )
        error_memory_id = mem_res.get("memory_id")
        await safe_tool_call(
            record_action_completion,
            {
                "action_id": action3_id,
                "status": ActionStatus.FAILED.value,
                "tool_result": error_result,
                "summary": "Vectorization failed tests due to dimension mismatch.",
            },
            "Complete Test 1 Action (Failed)",
        )

        # --- Reflection & Adaptation ---
        console.print(Rule("Reflection and Adaptation Phase", style="cyan"))
        reflection_res = await safe_tool_call(
            generate_reflection,
            {"workflow_id": wf_id, "reflection_type": "gaps"},
            "Generate Gaps Reflection",
        )
        assert reflection_res and reflection_res.get("success"), "Reflection generation failed"
        reflection_content = reflection_res.get("content", "").lower()

        # Programmatic check of reflection output
        if (
            "dimension mismatch" in reflection_content
            or "valueerror" in reflection_content
            or "vectorization" in reflection_content
            or action3_id[:6] in reflection_content
        ):  # Check for action ID too
            console.print(
                "[green]  Reflection mentioned the likely error source or related action.[/green]"
            )

            thought1_res = await safe_tool_call(
                record_thought,
                {
                    "workflow_id": wf_id,
                    "content": "Reflection and test failure (ValueError: Array dimensions mismatch) suggest the vectorization approach was fundamentally flawed or misapplied.",
                    "thought_type": ThoughtType.INFERENCE.value,
                    "relevant_action_id": action3_id,
                },
                "Record Inference Thought",
            )
            thought2_res = await safe_tool_call(  # noqa: F841
                record_thought,
                {
                    "workflow_id": wf_id,
                    "content": "Plan B: Abandon vectorization. Try loop unrolling as an alternative optimization strategy.",
                    "thought_type": ThoughtType.PLAN.value,
                    "parent_thought_id": thought1_res.get("thought_id"),
                },
                "Record Plan B Thought",
            )

            # Action 4: Attempt Optimization 2 (Loop Unrolling)
            act4_res = await safe_tool_call(
                record_action_start,
                {
                    "workflow_id": wf_id,
                    "action_type": ActionType.TOOL_USE.value,
                    "title": "Attempt optimization 2 (Loop Unrolling)",
                    "tool_name": "modify_code",
                    "reasoning": "Implement loop unrolling based on failure of vectorization (Plan B).",
                },
                "Start Optimization 2 Action",
            )
            action4_id = act4_res["action_id"]
            await safe_tool_call(
                record_artifact,
                {
                    "workflow_id": wf_id,
                    "action_id": action4_id,
                    "name": "optimized_v2.py",
                    "artifact_type": ArtifactType.CODE.value,
                },  # <<< FIX 2: Use imported enum
                "Record Opt 2 Artifact",
                suppress_output=True,
            )
            await safe_tool_call(
                record_action_completion,
                {"action_id": action4_id, "status": ActionStatus.COMPLETED.value},
                "Complete Optimization 2 Action",
            )

            # Action 5: Test Optimization 2 (Success)
            act5_res = await safe_tool_call(
                record_action_start,
                {
                    "workflow_id": wf_id,
                    "action_type": ActionType.TOOL_USE.value,
                    "title": "Test optimization 2",
                    "tool_name": "run_tests",
                    "reasoning": "Verify loop unrolling attempt.",
                },
                "Start Test 2 Action",
            )
            action5_id = act5_res["action_id"]
            # Store a success memory
            mem_success_res = await safe_tool_call(
                store_memory,
                {
                    "workflow_id": wf_id,
                    "action_id": action5_id,
                    "memory_type": MemoryType.OBSERVATION.value,
                    "content": "Test passed for optimization 2 (loop unrolling). Performance improved by 15%.",
                    "description": "Loop unrolling test success",
                    "tags": ["success", "test", "unrolling"],
                    "importance": 7.0,
                },
                "Store Success Observation Memory",
            )
            success_memory_id = mem_success_res.get("memory_id")
            await safe_tool_call(
                record_action_completion,
                {
                    "action_id": action5_id,
                    "status": ActionStatus.COMPLETED.value,
                    "tool_result": {"status": "passed", "performance_gain": "15%"},
                    "summary": "Loop unrolling successful and provided performance gain.",
                },
                "Complete Test 2 Action (Success)",
            )

            # Consolidate insights from failure and success
            if error_memory_id and success_memory_id:
                consolidation_res = await safe_tool_call(
                    consolidate_memories,
                    {
                        "workflow_id": wf_id,
                        "target_memories": [error_memory_id, success_memory_id],
                        "consolidation_type": "insight",
                    },
                    "Consolidate Failure/Success Insight",
                )
                # Simple check on content
                consolidated_insight = consolidation_res.get("consolidated_content", "")
                assert consolidation_res and consolidation_res.get("success"), (
                    "Consolidation tool call failed"
                )
                assert "vectorization failed" in consolidated_insight.lower() and (
                    "loop unrolling succeeded" in consolidated_insight.lower()
                    or "loop unrolling successful" in consolidated_insight.lower()
                ), "Consolidated insight didn't capture the key outcome."
                console.print("[green]  Consolidated insight correctly reflects outcome.[/green]")
            else:
                console.print(
                    "[yellow]  Skipping consolidation check as required memory IDs weren't captured.[/yellow]"
                )

        else:
            console.print(
                "[yellow]  Reflection did not explicitly mention the error source. Skipping adaptation steps.[/yellow]"
            )

    except AssertionError as e:
        logger.error(f"Assertion failed during Extension 2: {e}", exc_info=True)
        console.print(f"[bold red]Assertion Failed:[/bold red] {e}")
    except Exception as e:
        logger.error(f"Error in Extension 2: {e}", exc_info=True)
        console.print(f"[bold red]Error in Extension 2:[/bold red] {e}")
    finally:
        console.print(Rule("Extension 2 Finished", style="green"))


async def run_extension_3_knowledge_building():
    """Extension 3: Multi-Level Memory Interaction & Knowledge Building"""
    console.print(
        Rule(
            "[bold green]Extension 3: Knowledge Building & Memory Levels[/bold green]",
            style="green",
        )
    )
    wf_id = None
    episodic_mem_ids = []
    insight_mem_id = None
    procedural_mem_id = None

    try:
        # --- Setup ---
        wf_res = await safe_tool_call(
            create_workflow,
            {
                "title": "API Interaction Monitoring",
                "goal": "Observe and learn from API call patterns.",
            },
            "Create API Monitoring Workflow",
        )
        assert wf_res and wf_res.get("success"), "Failed to create workflow"
        wf_id = wf_res["workflow_id"]

        # --- Record Episodic Failures ---
        console.print(Rule("Simulating API Failures (Episodic)", style="cyan"))
        for i in range(4):  # Record a few failures
            # <<< FIX 3: Added reasoning argument >>>
            act_res = await safe_tool_call(
                record_action_start,
                {
                    "workflow_id": wf_id,
                    "action_type": ActionType.TOOL_USE.value,
                    "title": f"Call API Endpoint X (Attempt {i + 1})",
                    "tool_name": "call_api",
                    "reasoning": f"Attempting API call to endpoint X, attempt number {i + 1}.",  # Added reasoning
                },
                f"Start API Call Action {i + 1}",
                suppress_output=True,
            )
            # <<< FIX 4: Check act_res before accessing action_id >>>
            assert act_res and act_res.get("success"), f"Failed to start API Call Action {i + 1}"
            action_id = act_res["action_id"]  # Now safe to access

            fail_result = {"error_code": 429, "message": "Too Many Requests"}
            mem_res = await safe_tool_call(
                store_memory,
                {
                    "workflow_id": wf_id,
                    "action_id": action_id,
                    "memory_level": MemoryLevel.EPISODIC.value,
                    "memory_type": MemoryType.OBSERVATION.value,
                    "content": "API call to endpoint X failed with 429 Too Many Requests.",
                    "description": f"API Failure {i + 1}",
                    "tags": ["api_call", "failure", "429"],
                    "importance": 6.0 - i * 0.2,
                },  # Slightly decreasing importance
                f"Store Episodic Failure Memory {i + 1}",
            )
            assert mem_res and mem_res.get("success"), (
                f"Failed to store memory for action {action_id}"
            )
            episodic_mem_ids.append(mem_res["memory_id"])
            await safe_tool_call(
                record_action_completion,
                {
                    "action_id": action_id,
                    "status": ActionStatus.FAILED.value,
                    "tool_result": fail_result,
                },
                f"Complete API Call Action {i + 1} (Failed)",
                suppress_output=True,
            )
            await asyncio.sleep(0.1)  # Simulate time passing

        assert len(episodic_mem_ids) == 4, "Did not store all expected episodic memories"

        # --- Trigger Promotion ---
        console.print(Rule("Triggering Memory Promotion", style="cyan"))
        for mem_id in episodic_mem_ids:
            for _ in range(6):  # Access each memory enough times
                await safe_tool_call(
                    get_memory_by_id,
                    {"memory_id": mem_id},
                    f"Access Memory {mem_id[:8]}",
                    suppress_output=True,
                )
            # Attempt promotion
            promo_res = await safe_tool_call(
                promote_memory_level, {"memory_id": mem_id}, f"Attempt Promotion for {mem_id[:8]}"
            )
            assert promo_res and promo_res.get("promoted"), (
                f"Memory {mem_id} failed to promote to semantic"
            )
            assert promo_res.get("new_level") == MemoryLevel.SEMANTIC.value, (
                f"Memory {mem_id} promoted to wrong level"
            )
        console.print(
            "[green]  All episodic memories successfully accessed and promoted to Semantic.[/green]"
        )

        # --- Consolidation ---
        console.print(Rule("Consolidating Semantic Insights", style="cyan"))
        consolidation_res = await safe_tool_call(
            consolidate_memories,
            {
                "workflow_id": wf_id,
                "target_memories": episodic_mem_ids,
                "consolidation_type": "insight",
                "store_result": True,
                "store_as_level": MemoryLevel.SEMANTIC.value,
                "store_as_type": MemoryType.INSIGHT.value,
            },
            "Consolidate Failures into Insight",
        )
        assert consolidation_res and consolidation_res.get("success"), "Consolidation failed"
        insight_content = consolidation_res.get("consolidated_content", "").lower()
        insight_mem_id = consolidation_res.get("stored_memory_id")
        assert insight_mem_id, "Consolidated insight memory was not stored"
        assert (
            "rate limit" in insight_content
            or "429" in insight_content
            or "too many requests" in insight_content
        ), "Consolidated insight content missing expected keywords."
        console.print(
            f"[green]  Consolidated insight created (ID: {insight_mem_id[:8]}) and content seems correct.[/green]"
        )

        # --- Proceduralization ---
        console.print(Rule("Creating Procedural Knowledge", style="cyan"))
        proc_res = await safe_tool_call(
            store_memory,
            {
                "workflow_id": wf_id,
                "memory_level": MemoryLevel.PROCEDURAL.value,
                "memory_type": MemoryType.PROCEDURE.value,
                "content": "If API returns 429 error, wait using exponential backoff (e.g., 1s, 2s, 4s...) before retrying.",
                "description": "API Rate Limit Retry Strategy",
                "tags": ["api", "retry", "backoff", "rate_limit"],
                "importance": 8.0,
                "confidence": 0.95,
            },
            "Store Procedural Memory (Retry Strategy)",
        )
        assert proc_res and proc_res.get("success"), "Failed to store procedural memory"
        procedural_mem_id = proc_res["memory_id"]
        console.print(f"[green]  Procedural memory created (ID: {procedural_mem_id[:8]})[/green]")

        # --- Querying Verification ---
        console.print(Rule("Verifying Knowledge Retrieval", style="cyan"))
        # Query for insight
        semantic_search_res = await safe_tool_call(
            search_semantic_memories,
            {"query": "problem with API rate limits", "workflow_id": wf_id, "limit": 3},
            "Semantic Search for Insight",
        )
        assert semantic_search_res and semantic_search_res.get("success"), "Semantic search failed"
        found_insight = any(
            m["memory_id"] == insight_mem_id for m in semantic_search_res.get("memories", [])
        )
        assert found_insight, "Consolidated insight memory not found via semantic search"
        console.print(
            "[green]  Semantic search successfully retrieved the consolidated insight.[/green]"
        )

        # Query for procedure
        procedural_query_res = await safe_tool_call(
            query_memories,
            {
                "memory_level": MemoryLevel.PROCEDURAL.value,
                "search_text": "API retry strategy",
                "workflow_id": wf_id,
            },
            "Query for Procedural Memory",
        )
        assert procedural_query_res and procedural_query_res.get("success"), (
            "Procedural query failed"
        )
        found_procedure = any(
            m["memory_id"] == procedural_mem_id for m in procedural_query_res.get("memories", [])
        )
        assert found_procedure, "Procedural memory not found via query"
        console.print(
            "[green]  Filtered query successfully retrieved the procedural memory.[/green]"
        )

    except AssertionError as e:
        logger.error(f"Assertion failed during Extension 3: {e}", exc_info=True)
        console.print(f"[bold red]Assertion Failed:[/bold red] {e}")
    except Exception as e:
        logger.error(f"Error in Extension 3: {e}", exc_info=True)
        console.print(f"[bold red]Error in Extension 3:[/bold red] {e}")
    finally:
        console.print(Rule("Extension 3 Finished", style="green"))


async def run_extension_4_context_persistence():
    """Extension 4: Context Persistence and Working Memory Management"""
    console.print(
        Rule(
            "[bold green]Extension 4: Context Persistence & Working Memory[/bold green]",
            style="green",
        )
    )
    wf_id = None
    m_ids = {}  # Store M1-M8 IDs
    state1_id = None
    original_state1_working_set = []  # Store original WM here
    retained_id_after_optimize = None  # Store the ID retained by optimize

    try:
        # --- Setup ---
        wf_res = await safe_tool_call(
            create_workflow,
            {"title": "Analyze Document X", "goal": "Extract key info from Doc X."},
            "Create Doc Analysis Workflow",
        )
        assert wf_res and wf_res.get("success"), "Failed to create workflow"
        wf_id = wf_res["workflow_id"]

        # --- Initial Analysis & Memory Storage ---
        console.print(Rule("Initial Analysis Phase", style="cyan"))
        mem_contents = {
            "M1": "Document Section 1: Introduction and background.",
            "M2": "Document Section 2: Core methodology described.",
            "M3": "Document Section 3: Results for Experiment A.",
            "M4": "Document Section 4: Results for Experiment B.",
            "M5": "Document Section 5: Discussion and initial conclusions.",
        }
        for i, (m_key, content) in enumerate(mem_contents.items()):
            mem_res = await safe_tool_call(
                store_memory,
                {
                    "workflow_id": wf_id,
                    "content": content,
                    "memory_type": MemoryType.OBSERVATION.value,
                    "description": f"Notes on {m_key}",
                    "importance": 5.0 + i * 0.2,
                },
                f"Store Memory {m_key}",
                suppress_output=True,
            )
            assert mem_res and mem_res.get("success"), f"Failed to store memory {m_key}"
            m_ids[m_key] = mem_res["memory_id"]

        # --- Save State 1 ---
        console.print(Rule("Saving Initial State", style="cyan"))
        initial_working_set_to_save = [m_ids["M1"], m_ids["M2"], m_ids["M3"]]
        initial_focus = [m_ids["M2"]]
        state1_res = await safe_tool_call(
            save_cognitive_state,
            {
                "workflow_id": wf_id,
                "title": "Initial Section Analysis",
                "working_memory_ids": initial_working_set_to_save,
                "focus_area_ids": initial_focus,
            },
            "Save Cognitive State 1",
        )
        assert state1_res and state1_res.get("success"), "Failed to save state 1"
        state1_id = state1_res["state_id"]
        console.print(f"[cyan]  State 1 ID: {state1_id}[/cyan]")

        # <<< FIX 5 Workaround Part 1: Load immediately to capture original working memory >>>
        load_for_original_res = await safe_tool_call(
            load_cognitive_state,
            {"workflow_id": wf_id, "state_id": state1_id},
            "Load State 1 Immediately to Capture Original WM",
            suppress_output=True,  # Keep log cleaner
        )
        assert load_for_original_res and load_for_original_res.get("success"), (
            "Failed to load state 1 immediately after save"
        )
        original_state1_working_set = load_for_original_res.get("working_memory_ids", [])
        assert set(original_state1_working_set) == set(initial_working_set_to_save), (
            "Immediate load WM doesn't match saved WM"
        )
        console.print(
            f"[dim]  Captured original State 1 working set: {original_state1_working_set}[/dim]"
        )
        # <<< End FIX 5 Workaround Part 1 >>>

        # --- Simulate Interruption & Optimize ---
        console.print(Rule("Simulate Interruption & Optimize State 1", style="cyan"))
        mem6_res = await safe_tool_call(
            store_memory,
            {
                "workflow_id": wf_id,
                "content": "Unrelated thought about lunch.",
                "memory_type": MemoryType.OBSERVATION.value,
            },
            "Store Unrelated Memory M6",
            suppress_output=True,
        )
        mem7_res = await safe_tool_call(
            store_memory,
            {
                "workflow_id": wf_id,
                "content": "Another unrelated idea.",
                "memory_type": MemoryType.OBSERVATION.value,
            },
            "Store Unrelated Memory M7",
            suppress_output=True,
        )
        assert mem6_res and mem6_res.get("success"), "Failed to store M6"
        assert mem7_res and mem7_res.get("success"), "Failed to store M7"
        m_ids["M6"] = mem6_res["memory_id"]
        m_ids["M7"] = mem7_res["memory_id"]

        optimize_res = await safe_tool_call(
            optimize_working_memory,
            # Use the *state_id* as the context identifier
            {"context_id": state1_id, "target_size": 1, "strategy": "balanced"},
            "Optimize State 1's Working Memory",
        )
        assert optimize_res and optimize_res.get("success"), "Optimization failed"
        assert optimize_res["after_count"] == 1, (
            f"Optimization did not reduce to target size 1, got {optimize_res['after_count']}"
        )
        retained_id_after_optimize = optimize_res["retained_memories"][0]
        console.print(
            f"[cyan]  Optimization retained memory: {retained_id_after_optimize[:8]}...[/cyan]"
        )
        # Check retained is one of the original working set
        assert retained_id_after_optimize in original_state1_working_set, (
            "Optimization retained an unexpected memory"
        )

        # --- Load State 1 & Verify ---
        console.print(Rule("Load State 1 and Verify Context", style="cyan"))
        loaded_state_res = await safe_tool_call(
            load_cognitive_state,
            {"workflow_id": wf_id, "state_id": state1_id},
            "Load Cognitive State 1 (After Optimization)",
        )
        assert loaded_state_res and loaded_state_res.get("success"), "Failed to load state 1"
        # <<< FIX 5 Workaround Part 2: Assert against the captured original set >>>
        loaded_working_ids = loaded_state_res.get("working_memory_ids", [])
        assert set(loaded_working_ids) == set(original_state1_working_set), (
            f"Loaded working memory {loaded_working_ids} does not match original saved state {original_state1_working_set}"
        )
        console.print(
            "[green]  Loaded state working memory matches original saved state (pre-optimization). Test Passed.[/green]"
        )
        # <<< End FIX 5 Workaround Part 2 >>>

        # --- Test Focus ---
        focus_res = await safe_tool_call(
            auto_update_focus,
            {"context_id": state1_id},  # Use state_id as context
            "Auto Update Focus on Loaded State",
        )
        assert focus_res and focus_res.get("success"), "Auto update focus failed"
        new_focus_id = focus_res.get("new_focal_memory_id")
        # Check the new focus is one of the *retained* working memories after optimization
        assert new_focus_id == retained_id_after_optimize, (
            f"New focus ID {new_focus_id} is not the expected retained ID {retained_id_after_optimize}"
        )
        console.print(
            f"[green]  Auto-focus selected the expected memory ID: {new_focus_id[:8]}...[/green]"
        )

        # --- Continue Task & Test Working Memory Addition ---
        console.print(Rule("Continue Task & Add to Working Memory", style="cyan"))
        mem8_res = await safe_tool_call(
            store_memory,
            {
                "workflow_id": wf_id,
                "content": "Section 6: Key Conclusion",
                "memory_type": MemoryType.OBSERVATION.value,
                "description": "Notes on M8",
                "importance": 8.0,
            },
            "Store New Relevant Memory M8",
            suppress_output=True,
        )
        assert mem8_res and mem8_res.get("success"), "Failed to store M8"
        m_ids["M8"] = mem8_res["memory_id"]

        # Use focus_memory with add_to_working=True to test adding M8 to the *current* state (State 1)
        focus_m8_res = await safe_tool_call(
            focus_memory,
            {"memory_id": m_ids["M8"], "context_id": state1_id, "add_to_working": True},
            f"Focus on M8 ({m_ids['M8'][:8]}) and Add to Working Memory (Context {state1_id[:8]})",
        )
        assert focus_m8_res and focus_m8_res.get("success"), "Focusing on M8 failed"
        assert focus_m8_res.get("added_to_working"), (
            "M8 was not reported as added to working memory"
        )

        # Verify working memory contents *after* adding M8
        wm_after_add_res = await safe_tool_call(
            get_working_memory, {"context_id": state1_id}, "Get Working Memory After Adding M8"
        )
        assert wm_after_add_res and wm_after_add_res.get("success"), (
            "Failed to get working memory after adding M8"
        )
        wm_after_add_ids = [m["memory_id"] for m in wm_after_add_res.get("working_memories", [])]

        # <<< FIX 5 Workaround Part 3: Assert final WM state correctly >>>
        assert m_ids["M8"] in wm_after_add_ids, (
            "M8 is not present in working memory after add attempt"
        )
        # After optimize (target=1) and adding M8, the working set should contain
        # the ID retained by optimize plus M8.
        expected_final_wm = {retained_id_after_optimize, m_ids["M8"]}
        assert set(wm_after_add_ids) == expected_final_wm, (
            f"Final working memory {set(wm_after_add_ids)} doesn't match expected {expected_final_wm} after optimization and add"
        )
        console.print(
            f"[green]  Memory M8 successfully added to working memory for state {state1_id[:8]}. Final WM is correct.[/green]"
        )
        # <<< End FIX 5 Workaround Part 3 >>>

    except AssertionError as e:
        logger.error(f"Assertion failed during Extension 4: {e}", exc_info=True)
        console.print(f"[bold red]Assertion Failed:[/bold red] {e}")
    except Exception as e:
        logger.error(f"Error in Extension 4: {e}", exc_info=True)
        console.print(f"[bold red]Error in Extension 4:[/bold red] {e}")
    finally:
        console.print(Rule("Extension 4 Finished", style="green"))


# --- Main Execution Logic ---
async def main():
    """Run the advanced Unified Memory System demonstration suite."""
    console.print(
        Rule(
            "[bold magenta]Advanced Unified Memory System Tools Demo[/bold magenta]", style="white"
        )
    )
    exit_code = 0

    try:
        await setup_advanced_demo()

        # --- Run Demo Extensions ---
        await run_extension_1_goal_decomposition()
        await run_extension_2_dynamic_adaptation()
        await run_extension_3_knowledge_building()
        await run_extension_4_context_persistence()

        logger.success(
            "Advanced Unified Memory System Demo completed successfully!", emoji_key="complete"
        )
        console.print(Rule("[bold green]Advanced Demo Finished[/bold green]", style="green"))

    except Exception as e:
        logger.critical(
            f"Advanced demo crashed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True
        )
        console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        exit_code = 1

    finally:
        console.print(Rule("Cleanup Advanced Demo", style="dim"))
        await cleanup_advanced_demo()

    return exit_code


if __name__ == "__main__":
    final_exit_code = asyncio.run(main())
    sys.exit(final_exit_code)
