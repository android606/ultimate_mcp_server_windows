"""
Utility functions for tournament functionality.
"""

import difflib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ultimate_mcp_server.core.models.tournament import (
    EvaluatorConfig,
    ModelResponseData,
    TournamentData,
    TournamentStatus,
)

# For file write, if using a tool:
from ultimate_mcp_server.tools.filesystem import write_file

logger = logging.getLogger(__name__)


def create_round_prompt(
    tournament: TournamentData,
    round_num: int,
    previous_round_variant_responses: Dict[
        str, ModelResponseData
    ],  # Now takes full ModelResponseData
    target_model_variant_id: Optional[str] = None,  # For per-variant system prompts etc. (future)
) -> str:
    """Creates the prompt for a specific round."""
    if round_num == 0:
        return tournament.config.prompt

    # --- Build prompt with previous round's responses ---
    base_prompt_header = f"""This is Round {round_num} of an iterative refinement process.
Original Problem:
---
{tournament.config.prompt}
---

In the previous round (Round {round_num - 1}), different model variants produced the following outputs.
Your goal is to synthesize the best aspects, address weaknesses, and produce a superior solution.
"""

    responses_section = []
    for variant_id, resp_data in previous_round_variant_responses.items():
        # For code tournaments, use extracted code if available and valid for prompting.
        # For text, use full response_text.
        content_to_show = ""
        if tournament.config.tournament_type == "code":
            # Prioritize clean extracted code for next round's prompt
            content_to_show = (
                resp_data.extracted_code if resp_data.extracted_code else resp_data.response_text
            )
            if (
                not content_to_show or len(content_to_show.strip()) < 10
            ):  # Heuristic for empty/trivial code
                content_to_show = resp_data.response_text  # Fallback to full text if code is bad
            content_to_show = (
                f"```python\n{content_to_show.strip()}\n```"
                if content_to_show
                else "[No valid code extracted]"
            )
        else:  # Text tournament
            content_to_show = (
                resp_data.response_text if resp_data.response_text else "[No response text]"
            )

        if resp_data.error:
            content_to_show += f"\n[Note: This variant encountered an error: {resp_data.error}]"

        # Show overall score if available
        score_info = ""
        if resp_data.overall_score is not None:
            score_info = f" (Overall Score: {resp_data.overall_score:.2f})"

        responses_section.append(
            f"--- Output from Variant: {variant_id}{score_info} ---\n{content_to_show.strip()}\n"
        )

    # --- Add type-specific instructions ---
    if tournament.config.tournament_type == "code":
        instructions = """
Carefully analyze all previous code solutions. Consider correctness, efficiency, readability, robustness, and how well they integrate good ideas.
Produce a NEW, complete Python implementation that is demonstrably better than any single prior solution.
Provide ONLY the Python code block, enclosed in triple backticks (```python ... ```).
Do not include any explanations outside of code comments.
"""
    else:  # Text tournament
        instructions = """
Analyze each previous response based on the original problem. Consider relevance, accuracy, completeness, clarity, conciseness, and style.
Synthesize the best aspects of ALL responses into a single, improved response.
Your new response should be superior to any individual response from the previous round.
You MAY optionally start your response with a brief (1-2 sentences) explanation of your synthesis choices, enclosed in <thinking>...</thinking> tags.
Then, provide the improved text response itself.
"""

    final_prompt = f"{base_prompt_header}\n{''.join(responses_section)}\n---Refinement Instructions---\n{instructions}"
    return final_prompt.strip()


async def extract_thinking(response_text: str) -> Optional[str]:
    """Extracts <thinking>...</thinking> block. More robust extraction can be added."""
    if not response_text:
        return None
    match = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


async def save_model_response_content(
    tournament_storage_path: Path,
    round_num: int,
    variant_id: str,
    response_text: Optional[str],
    extracted_code: Optional[str],
    thinking_process: Optional[str],
    metrics: Dict[str, Any],
    tournament_type: Literal["code", "text"],
) -> Dict[str, Optional[str]]:
    """Saves response text, extracted code, and metadata to files."""
    round_dir = tournament_storage_path / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    sanitized_variant_id = re.sub(r"[^a-zA-Z0-9_\-.]", "_", variant_id)
    base_filename = f"{sanitized_variant_id}_r{round_num}"

    # --- Main Markdown Report File ---
    md_content = f"# Response: {variant_id} - Round {round_num}\n\n"
    md_content += "## Metrics\n"
    for k, v in metrics.items():
        if isinstance(v, float):
            md_content += f"- **{k.replace('_', ' ').title()}:** {v:.4f}\n"
        else:
            md_content += f"- **{k.replace('_', ' ').title()}:** {v}\n"

    if thinking_process:
        md_content += f"\n## Thinking Process\n```\n{thinking_process}\n```\n"

    md_content += f"\n## Full Response Text\n```\n{response_text or '[No response text]'}\n```\n"

    if tournament_type == "code" and extracted_code:
        md_content += f"\n## Extracted Code\n```python\n{extracted_code}\n```\n"

    md_file_path = round_dir / f"{base_filename}_report.md"
    md_file_path.write_text(md_content, encoding="utf-8")

    saved_paths = {"markdown_file": str(md_file_path), "code_file": None}

    # --- Save Raw Extracted Code (if any) ---
    if tournament_type == "code" and extracted_code:
        code_file_path = round_dir / f"{base_filename}.py"
        code_file_path.write_text(extracted_code, encoding="utf-8")
        saved_paths["code_file"] = str(code_file_path)

    logger.debug(f"Saved response artifacts for {variant_id} to {round_dir}")
    return saved_paths


def generate_comparison_file_content(tournament: TournamentData, round_num: int) -> Optional[str]:
    if round_num < 0 or round_num >= len(tournament.rounds_results):
        return None
    round_result = tournament.rounds_results[round_num]
    if not round_result.responses:
        return None

    content = f"# Tournament Comparison Report - Round {round_num}\n\n"
    content += f"**Tournament:** {tournament.name} (ID: {tournament.tournament_id})\n"
    content += f"**Type:** {tournament.config.tournament_type}\n"
    content += f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n"

    content += "## Round Summary & Scores\n"
    content += (
        "| Variant ID | Overall Score | Key Metrics (e.g., Cost, Latency) | Evaluator Scores |\n"
    )
    content += (
        "|------------|---------------|-----------------------------------|------------------|\n"
    )

    sorted_responses = sorted(
        round_result.responses.items(),
        key=lambda item: item[1].overall_score if item[1].overall_score is not None else -1,
        reverse=True,
    )

    for variant_id, resp_data in sorted_responses:
        score_str = (
            f"{resp_data.overall_score:.2f}" if resp_data.overall_score is not None else "N/A"
        )
        cost = resp_data.metrics.get("cost", 0.0)
        latency = resp_data.metrics.get("latency_ms", "N/A")
        key_metrics = f"Cost: ${cost:.4f}, Latency: {latency}ms"

        eval_scores_str = (
            "; ".join(
                [
                    f"{eval_id}: {s_data.get('score', 'N/A')}"
                    for eval_id, s_data in resp_data.scores.items()
                ]
            )
            if resp_data.scores
            else "N/A"
        )

        content += f"| {variant_id} | {score_str} | {key_metrics} | {eval_scores_str} |\n"
    content += "\n"

    # --- Add Diffs (Proposal 6) ---
    if round_num > 0 and tournament.config.tournament_type == "code":
        content += "## Code Diffs from Previous Best (if applicable)\n"
        prev_round_best_code = None
        # Find best code from previous round (simplistic: first non-error, highest score)
        if round_num - 1 >= 0:
            prev_round_data = tournament.rounds_results[round_num - 1]
            best_prev_resp = max(
                filter(
                    lambda r: r.extracted_code and r.overall_score is not None,
                    prev_round_data.responses.values(),
                ),
                key=lambda r: r.overall_score,
                default=None,
            )
            if best_prev_resp:
                prev_round_best_code = best_prev_resp.extracted_code

        current_best_resp = max(
            filter(
                lambda r: r.extracted_code and r.overall_score is not None,
                round_result.responses.values(),
            ),
            key=lambda r: r.overall_score,
            default=None,
        )
        current_best_code = current_best_resp.extracted_code if current_best_resp else None

        if prev_round_best_code and current_best_code:
            diff = difflib.unified_diff(
                prev_round_best_code.splitlines(keepends=True),
                current_best_code.splitlines(keepends=True),
                fromfile=f"round_{round_num - 1}_best.py",
                tofile=f"round_{round_num}_best.py",
                lineterm="",
            )
            content += f"### Diff: Best of Round {round_num - 1} vs Best of Round {round_num}\n"
            content += "```diff\n"
            content += "".join(diff)
            content += "\n```\n\n"
        elif current_best_code:
            content += "Could not determine previous round's best code for diffing, or this is the first round with code.\n"
        # TODO: Add HTML diff for text tournaments if a library is available.

    content += "## Detailed Variant Responses\n"
    for variant_id, resp_data in sorted_responses:
        content += f"### Variant: {variant_id}\n"
        content += f"- **Original Model:** {resp_data.model_id_original}\n"
        content += (
            f"- **Overall Score:** {resp_data.overall_score:.2f}\n"
            if resp_data.overall_score is not None
            else "- **Overall Score:** N/A\n"
        )
        content += "#### Metrics:\n"
        for k, v in resp_data.metrics.items():
            content += f"  - {k}: {v}\n"
        content += "#### Evaluator Scores:\n"
        if resp_data.scores:
            for eval_id, s_data in resp_data.scores.items():
                content += f"  - **{eval_id}**: Score: {s_data.get('score', 'N/A')}\n    - Details: {s_data.get('details', 'N/A')[:200]}...\n"  # Truncate details
        else:
            content += "  - No scores available.\n"

        if resp_data.thinking_process:
            content += f"#### Thinking Process:\n```\n{resp_data.thinking_process}\n```\n"

        content_key = (
            "Extracted Code" if tournament.config.tournament_type == "code" else "Response Text"
        )
        code_lang_hint = "python" if tournament.config.tournament_type == "code" else ""
        actual_content = (
            resp_data.extracted_code
            if tournament.config.tournament_type == "code" and resp_data.extracted_code
            else resp_data.response_text
        )

        content += f"#### {content_key}:\n```{code_lang_hint}\n{actual_content or '[Content not available]'}\n```\n"
        if resp_data.response_file_path:  # Link to the full report for this variant
            # Make path relative to tournament storage root for portability
            try:
                tournament_root = Path(tournament.storage_path)
                relative_path = Path(resp_data.response_file_path).relative_to(
                    tournament_root.parent
                )  # one level up for `round_X/file`
                content += f"\n[View Full Variant Report](./{relative_path})\n"
            except ValueError:  # If not relative (e.g. absolute path)
                content += f"\n[View Full Variant Report]({resp_data.response_file_path})\n"

        content += "\n---\n"
    return content


def generate_leaderboard_file_content(tournament: TournamentData, round_num: int) -> Optional[str]:
    """Generates a leaderboard summary for the current round."""
    if round_num < 0 or round_num >= len(tournament.rounds_results):
        return None
    round_result = tournament.rounds_results[round_num]
    if not round_result.responses:
        return None

    content = f"# Leaderboard - Round {round_num}\n\n"
    content += f"**Tournament:** {tournament.name}\n"
    content += f"**Primary Metric(s):** {', '.join([e.evaluator_id for e in tournament.config.evaluators if e.primary_metric]) or 'Overall Score'}\n\n"

    content += "| Rank | Variant ID | Overall Score | Primary Metric Score(s) |\n"
    content += "|------|------------|---------------|-------------------------|\n"

    # Sort by overall_score, then by primary metric if tied (more complex sorting can be added)
    sorted_responses = sorted(
        round_result.responses.values(),
        key=lambda r: r.overall_score if r.overall_score is not None else -float("inf"),
        reverse=True,
    )

    for i, resp_data in enumerate(sorted_responses):
        rank = i + 1
        score_str = (
            f"{resp_data.overall_score:.2f}" if resp_data.overall_score is not None else "N/A"
        )

        primary_scores_list = []
        for eval_cfg in tournament.config.evaluators:
            if eval_cfg.primary_metric and eval_cfg.evaluator_id in resp_data.scores:
                primary_scores_list.append(
                    f"{eval_cfg.evaluator_id}: {resp_data.scores[eval_cfg.evaluator_id].get('score', 'N/A')}"
                )
        primary_metrics_str = "; ".join(primary_scores_list) or "N/A"

        content += (
            f"| {rank} | {resp_data.model_id_variant} | {score_str} | {primary_metrics_str} |\n"
        )

    return content


def calculate_weighted_score(
    scores: Dict[str, Dict[str, Any]], evaluator_configs: List[EvaluatorConfig]
) -> Optional[float]:
    """Calculates a single weighted overall score from multiple evaluator scores."""
    if not scores or not evaluator_configs:
        return None

    total_score = 0.0
    total_weight = 0.0

    for eval_cfg in evaluator_configs:
        eval_id = eval_cfg.evaluator_id
        if eval_id in scores:
            score_data = scores[eval_id]
            # Assuming 'score' is the primary numerical output of an evaluator
            numerical_score = score_data.get("score")
            if isinstance(numerical_score, (int, float)):
                total_score += numerical_score * eval_cfg.weight
                total_weight += eval_cfg.weight
            else:
                logger.warning(
                    f"Evaluator '{eval_id}' provided non-numeric score: {numerical_score}"
                )

    if total_weight == 0:
        # If no weights or no valid scores, average non-weighted if any scores present
        valid_scores = [
            s.get("score") for s in scores.values() if isinstance(s.get("score"), (int, float))
        ]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None

    return total_score / total_weight


def update_overall_best_response(tournament: TournamentData):
    """Identifies and updates the tournament's overall best response across all completed rounds."""
    current_best_score = -float("inf")
    if (
        tournament.overall_best_response
        and tournament.overall_best_response.overall_score is not None
    ):
        current_best_score = tournament.overall_best_response.overall_score

    new_best_found = False
    for round_result in tournament.rounds_results:
        if round_result.status == TournamentStatus.COMPLETED:
            for _, resp_data in round_result.responses.items():
                if resp_data.overall_score is not None and not resp_data.error:
                    if resp_data.overall_score > current_best_score:
                        tournament.overall_best_response = resp_data
                        current_best_score = resp_data.overall_score
                        new_best_found = True

    if new_best_found:
        logger.info(
            f"New overall best response for tournament '{tournament.name}' found: {tournament.overall_best_response.model_id_variant} with score {current_best_score:.2f}"
        )


def calculate_code_metrics(code: Optional[str]) -> dict:
    """
    Calculates basic metrics about a code string.
    """
    if not code:
        return {
            "code_lines": 0,
            "code_size_kb": 0.0,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
        }

    code_lines = code.count("\n") + 1
    code_size_bytes = len(code.encode("utf-8"))
    code_size_kb = round(code_size_bytes / 1024, 2)
    function_count = len(re.findall(r"\bdef\s+\w+", code))
    class_count = len(re.findall(r"\bclass\s+\w+", code))
    import_count = len(re.findall(r"^import\s+|\bfrom\s+", code, re.MULTILINE))

    return {
        "code_lines": code_lines,
        "code_size_kb": code_size_kb,
        "function_count": function_count,
        "class_count": class_count,
        "import_count": import_count,
    }


def generate_comparison_file(tournament: TournamentData, round_num: int) -> Optional[str]:
    """Generate a markdown comparison file for the given round.

    Args:
        tournament: The tournament data.
        round_num: The round number to generate the comparison for.

    Returns:
        The markdown content string, or None if data is missing.
    """
    if round_num < 0 or round_num >= len(tournament.rounds_results):
        logger.warning(f"Cannot generate comparison for invalid round {round_num}")
        return None

    round_result = tournament.rounds_results[round_num]
    if not round_result.responses:
        logger.warning(f"Cannot generate comparison for round {round_num}, no responses found.")
        return None

    previous_round = tournament.rounds_results[round_num - 1] if round_num > 0 else None
    is_code_tournament = tournament.config.tournament_type == "code"

    # Start with a comprehensive header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comparison_content = f"# Tournament Comparison - Round {round_num}\n\n"
    comparison_content += f"**Generated:** {timestamp}\n"
    comparison_content += f"**Tournament ID:** {tournament.tournament_id}\n"
    comparison_content += f"**Tournament Name:** {tournament.config.name}\n"
    comparison_content += f"**Type:** {tournament.config.tournament_type}\n"
    comparison_content += f"**Current Round:** {round_num} of {tournament.config.rounds}\n"
    comparison_content += (
        f"**Models:** {', '.join(model.model_id for model in tournament.config.models)}\n\n"
    )

    # Add original prompt section
    if round_num == 0:
        comparison_content += f"## Original Prompt\n\n```\n{tournament.config.prompt}\n```\n\n"
    else:
        # For later rounds, show what was provided to the models
        comparison_content += f"## Round {round_num} Prompt\n\n"
        # Get a sample prompt - all models get the same prompt in a round
        sample_prompt = create_round_prompt(tournament, round_num)
        comparison_content += f"```\n{sample_prompt[:500]}...\n```\n\n"

    # Summarize overall metrics
    comparison_content += "## Summary Metrics\n\n"
    comparison_content += "| Model | Tokens In | Tokens Out | Cost | Latency (ms) |\n"
    comparison_content += "|-------|-----------|------------|------|-------------|\n"

    for model_id, response_data in sorted(round_result.responses.items()):
        metrics = response_data.metrics
        tokens_in = metrics.get("input_tokens", "N/A")
        tokens_out = metrics.get("output_tokens", "N/A")
        cost = metrics.get("cost", "N/A")
        latency = metrics.get("latency_ms", "N/A")

        display_model_id = model_id.split(":")[-1] if ":" in model_id else model_id
        cost_display = f"${cost:.6f}" if isinstance(cost, (int, float)) else cost

        comparison_content += (
            f"| {display_model_id} | {tokens_in} | {tokens_out} | {cost_display} | {latency} |\n"
        )

    comparison_content += "\n## Detailed Model Responses\n\n"

    for model_id, response_data in sorted(round_result.responses.items()):
        metrics = response_data.metrics
        display_model_id = model_id.split(":")[-1] if ":" in model_id else model_id

        comparison_content += f"### {display_model_id}\n\n"

        # Display detailed metrics as a subsection
        comparison_content += "#### Metrics\n\n"
        tokens_in = metrics.get("input_tokens", "N/A")
        tokens_out = metrics.get("output_tokens", "N/A")
        total_tokens = metrics.get("total_tokens", "N/A")
        cost = metrics.get("cost", "N/A")
        latency = metrics.get("latency_ms", "N/A")

        comparison_content += (
            f"- **Tokens:** {tokens_in} in, {tokens_out} out, {total_tokens} total\n"
        )
        if isinstance(cost, (int, float)):
            comparison_content += f"- **Cost:** ${cost:.6f}\n"
        else:
            comparison_content += f"- **Cost:** {cost}\n"
        comparison_content += f"- **Latency:** {latency}ms\n"

        # Code-specific metrics
        if is_code_tournament:
            code_lines = metrics.get("code_lines", "N/A")
            code_size = metrics.get("code_size_kb", "N/A")
            comparison_content += f"- **Code Stats:** {code_lines} lines, {code_size} KB\n"

        comparison_content += "\n"

        # Display thinking process if available
        if response_data.thinking_process:
            comparison_content += "#### Thinking Process\n\n"
            comparison_content += f"```\n{response_data.thinking_process}\n```\n\n"

        # Display response content
        if is_code_tournament:
            comparison_content += "#### Extracted Code\n\n"
            comparison_content += "```python\n"
            comparison_content += response_data.extracted_code or "# No code extracted"
            comparison_content += "\n```\n\n"
        else:
            # For text tournaments, display the raw response
            comparison_content += "#### Response Text\n\n"
            comparison_content += "```\n"
            comparison_content += response_data.response_text or "[No response text]"
            comparison_content += "\n```\n\n"

        # Add link to the full response file
        if response_data.response_file_path:
            comparison_content += (
                f"[View full response file]({response_data.response_file_path})\n\n"
            )

    # Add a section comparing changes from previous round if this isn't round 0
    if previous_round and previous_round.responses:
        comparison_content += "## Changes from Previous Round\n\n"
        for model_id, response_data in sorted(round_result.responses.items()):
            if model_id in previous_round.responses:
                display_model_id = model_id.split(":")[-1] if ":" in model_id else model_id
                comparison_content += f"### {display_model_id}\n\n"

                # Compare metrics
                current_metrics = response_data.metrics
                previous_metrics = previous_round.responses[model_id].metrics

                current_tokens_out = current_metrics.get("output_tokens", 0)
                previous_tokens_out = previous_metrics.get("output_tokens", 0)
                token_change = (
                    current_tokens_out - previous_tokens_out
                    if isinstance(current_tokens_out, (int, float))
                    and isinstance(previous_tokens_out, (int, float))
                    else "N/A"
                )

                comparison_content += f"- **Token Change:** {token_change} tokens\n"

                # Note: Here you could add more sophisticated text comparison/diff
                comparison_content += "- Review the full responses to see detailed changes\n\n"

    return comparison_content.strip()


async def save_model_response(
    tournament: TournamentData,
    round_num: int,
    model_id: str,
    response_text: str,
    thinking: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Save model response to a file using standardized filesystem tools.

    Args:
        tournament: Tournament data
        round_num: Round number
        model_id: Model ID that generated this response
        response_text: The text response to save
        thinking: Optional thinking process from the model
        timestamp: Optional timestamp (defaults to current time if not provided)

    Returns:
        Path to saved response file
    """
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get path to tournament storage directory
    storage_dir = Path(tournament.storage_path)
    round_dir = storage_dir / f"round_{round_num}"
    round_dir.mkdir(exist_ok=True)

    # Create a safe filename from model ID
    safe_model_id = model_id.replace(":", "_").replace("/", "_")
    response_file = round_dir / f"{safe_model_id}_response.md"

    # Construct the markdown file with basic metadata header
    content = f"""# Response from {model_id}

## Metadata
- Tournament: {tournament.name}
- Round: {round_num}
- Model: {model_id}
- Timestamp: {timestamp}

## Response:

{response_text}
"""

    # Add thinking process if available
    if thinking:
        content += f"\n\n## Thinking Process:\n\n{thinking}\n"

    # Use the standard filesystem write tool
    try:
        # Properly use the async write_file tool
        result = await write_file(path=str(response_file), content=content)

        if not result.get("success", False):
            logger.warning(f"Standard write_file tool reported failure: {result.get('error')}")
            # Fall back to direct write
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(content)
    except Exception as e:
        logger.error(f"Error using standardized file writer: {e}. Using direct file write.")
        # Fall back to direct write in case of errors
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(content)

    return str(response_file)


def get_round_dir(tournament: TournamentData, round_num: int) -> Path:
    """Get the directory path for a specific tournament round.

    Args:
        tournament: The tournament data.
        round_num: The round number.

    Returns:
        Path to the round directory.
    """
    tournament_dir = Path(tournament.storage_path)
    round_dir = tournament_dir / f"round_{round_num}"
    return round_dir


def get_word_count(text: str) -> int:
    """Get the word count of a text string.

    Args:
        text: The text to count words in.

    Returns:
        The number of words.
    """
    if not text:
        return 0
    return len(text.split())


def generate_synthesis_prompt(
    tournament: TournamentData, previous_responses: Dict[str, str]
) -> str:
    """Generate the prompt for the synthesis round.

    Args:
        tournament: The tournament data
        previous_responses: A dictionary mapping model IDs to their responses

    Returns:
        The synthesis prompt for the next round.
    """
    # Letter used for referring to models to avoid bias
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Start with a base prompt instructing the model what to do
    prompt = f"""# {tournament.name} - Synthesis Round

Your task is to create an improved version based on the responses from multiple models.

Original task:
{tournament.config.prompt}

Below are responses from different models. Review them and create a superior response 
that combines the strengths of each model's approach while addressing any weaknesses.

"""

    # Add each model's response
    for i, (model_id, response) in enumerate(previous_responses.items()):
        if i < len(letters):
            letter = letters[i]
            model_name = model_id.split(":")[-1] if ":" in model_id else model_id

            prompt += f"""
## Model {letter} ({model_name}) Response:

{response}

"""

    # Add synthesis instructions
    prompt += """
# Your Task

Based on the responses above:

1. Create a single, unified response that represents the best synthesis of the information
2. Incorporate the strengths of each model's approach
3. Improve upon any weaknesses or omissions
4. Your response should be more comprehensive, accurate, and well-structured than any individual response

## Thinking Process
Start by briefly analyzing the strengths and weaknesses of each model's response, then explain your synthesis approach.

Example: "I synthesized the structured approach of Model A with the comprehensive detail from Model B, ensuring..."

"""

    return prompt
