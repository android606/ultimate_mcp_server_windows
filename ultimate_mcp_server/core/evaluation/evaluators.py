# --- core/evaluation/evaluators.py ---
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ultimate_mcp_server.core.evaluation.base import (
    EvaluationScore,
    Evaluator,
    register_evaluator,
)
from ultimate_mcp_server.core.models.tournament import ModelResponseData
from ultimate_mcp_server.tools.completion import generate_completion

# --- Import the sandbox execution tool ---
from ultimate_mcp_server.tools.python_sandbox import (
    ProviderError,
    ToolError,
    ToolInputError,
    execute_python,
)
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.evaluation.evaluators")


@register_evaluator
class LLMGraderEvaluator(Evaluator):
    evaluator_type = "llm_grader"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.grader_model_id = config.get("model_id", "anthropic/claude-3-5-haiku-20241022")
        self.rubric = config.get(
            "rubric",
            "Score the response on a scale of 0-100 for quality, relevance, and clarity. Explain your reasoning.",
        )
        self.score_extraction_regex_str = config.get(
            "score_extraction_regex", r"Score:\s*(\d{1,3})"
        )
        try:
            self.score_extraction_regex = re.compile(self.score_extraction_regex_str)
        except re.error as e:
            logger.error(
                f"Invalid regex for score_extraction_regex in LLMGrader: {self.score_extraction_regex_str}. Error: {e}"
            )
            self.score_extraction_regex = re.compile(r"Score:\s*(\d{1,3})")

    async def score(
        self,
        response_data: ModelResponseData,
        original_prompt: str,
        tournament_type: Literal["code", "text"],
    ) -> EvaluationScore:
        # ... (LLMGraderEvaluator code remains the same) ...
        content_to_grade = (
            response_data.extracted_code
            if tournament_type == "code" and response_data.extracted_code
            else response_data.response_text
        )

        if not content_to_grade:
            return EvaluationScore(score=0.0, details="No content to grade.")

        prompt = f"""Original Prompt:
{original_prompt}

Model Response to Evaluate:
---
{content_to_grade}
---

Rubric:
{self.rubric}

Please provide a score (0-100) and a brief justification. Format the score clearly, e.g., "Score: 90".
"""
        try:
            provider = self.grader_model_id.split("/")[0] if "/" in self.grader_model_id else None

            grader_response_dict = await generate_completion(
                prompt=prompt,
                model=self.grader_model_id,
                provider=provider,
                max_tokens=500,
                temperature=0.2,
            )  # Changed var name

            if not grader_response_dict.get("success"):  # Use new var name
                return EvaluationScore(
                    score=0.0, details=f"Grader LLM failed: {grader_response_dict.get('error')}"
                )

            grader_text = grader_response_dict.get("text", "")  # Use new var name

            score_match = self.score_extraction_regex.search(grader_text)
            numerical_score = 0.0
            if score_match:
                try:
                    numerical_score = float(score_match.group(1))
                    if not (0 <= numerical_score <= 100):
                        numerical_score = max(0.0, min(100.0, numerical_score))
                except ValueError:
                    logger.warning(
                        f"LLMGrader: Could not parse score from '{score_match.group(1)}'"
                    )
                except IndexError:
                    logger.warning(
                        f"LLMGrader: Regex '{self.score_extraction_regex_str}' matched but had no capture group 1."
                    )
            else:
                logger.warning(
                    f"LLMGrader: Could not find score pattern in grader response: {grader_text[:200]}"
                )

            return EvaluationScore(
                score=numerical_score,
                details=grader_text,
                metrics={"grader_cost": grader_response_dict.get("cost", 0)},  # Use new var name
            )

        except Exception as e:
            logger.error(f"LLMGrader failed: {e}", exc_info=True)
            return EvaluationScore(score=0.0, details=f"Error during LLM grading: {str(e)}")


@register_evaluator
class UnitTestEvaluator(Evaluator):
    evaluator_type = "unit_test"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        test_file_path_str = config.get("test_file_path")
        self.required_packages: List[str] = config.get("required_packages", [])  # For sandbox

        if not test_file_path_str:
            logger.warning(
                "UnitTestEvaluator: 'test_file_path' not provided in config. This evaluator may not function."
            )
            self.test_file_path = Path()
        else:
            self.test_file_path = Path(test_file_path_str)
        self.timeout_seconds = config.get("timeout_seconds", 30)  # Sandbox timeout is in ms

    async def score(
        self,
        response_data: ModelResponseData,
        original_prompt: str,  # Unused but part of interface
        tournament_type: Literal["code", "text"],
    ) -> EvaluationScore:
        if tournament_type != "code" or not response_data.extracted_code:
            return EvaluationScore(
                score=0.0,
                details="Unit test evaluator only applicable to code tournaments with extracted code.",
            )

        if (
            not self.test_file_path
            or not self.test_file_path.exists()
            or not self.test_file_path.is_file()
        ):
            details = f"Test file not found, not configured, or not a file: {self.test_file_path}"
            if not self.test_file_path.name:
                details = "Test file path not configured for UnitTestEvaluator."
            logger.warning(f"UnitTestEvaluator: {details}")
            return EvaluationScore(score=0.0, details=details)

        try:
            # Read the user's test code from the host filesystem
            user_test_code = self.test_file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"UnitTestEvaluator: Failed to read test file {self.test_file_path}: {e}")
            return EvaluationScore(score=0.0, details=f"Failed to read test file: {e}")

        # Combine the generated code and the user's test code into a single script
        # to be run in the sandbox.
        # The generated code will be defined first, then the test code.
        # We assume the test code can import/use things defined in the generated code.
        # A common pattern is for generated code to be in a module `solution` or similar.
        # Here, we'll just put them in the same global scope for simplicity.

        # Let's make the generated code importable as 'generated_solution'
        # and the test code able to 'from generated_solution import *' or specific functions/classes.
        # This requires the generated code to be structured as a module.
        # For now, a simpler approach: just concatenate.
        # More robust: write generated_code to solution.py, test_code to test_solution.py,
        # then run test_solution.py which imports solution.py. This is harder without a true sandbox FS.

        # --- Simpler approach: Inject generated code directly, then test code ---
        # Test code should be written to assume the generated code's functions/classes
        # are available in the global scope or importable from a predefined module name.
        # For Pyodide, defining them globally is easiest.

        # The `unittest_runner_script` will execute the combined code.
        # It will define the generated code, then the test code, then run unittest.

        generated_code_to_run = response_data.extracted_code

        # This script will be executed by python_sandbox.py
        # It needs to define the generated functions/classes, then define and run tests.
        # stdout from this script will be parsed for results.
        unittest_runner_script = f"""
# --- Generated Code from Model ---
{generated_code_to_run}
# --- End of Generated Code ---

# --- User's Test Code ---
{user_test_code}
# --- End of User's Test Code ---

# --- Unittest Execution ---
import unittest
import sys
import io # To capture unittest output

# Capture unittest's output to a string buffer instead of stderr
# This makes parsing easier and cleaner from the sandbox output.
suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
output_buffer = io.StringIO()
runner = unittest.TextTestRunner(stream=output_buffer, verbosity=2)
result = runner.run(suite)

# Print results in a parsable format to STDOUT
# The python_sandbox tool will capture this stdout.
print("UNIT_TEST_RESULTS_START") # Delimiter for easier parsing
print(f"TestsRun:{{result.testsRun}}")
print(f"Failures:{{len(result.failures)}}")
print(f"Errors:{{len(result.errors)}}")
pass_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0.0
print(f"PassRate:{{pass_rate:.4f}}")
print("UNIT_TEST_RESULTS_END")

# Also print the full unittest output (which was captured in output_buffer)
# This can go to stdout as well, or we can separate it.
print("\\n--- Unittest Full Output ---")
print(output_buffer.getvalue())
"""
        details_output = "Unit test execution details via Pyodide Sandbox:\n"
        pass_rate = 0.0
        tests_run = 0
        failures = 0
        errors = 0
        sandbox_stdout = ""
        sandbox_stderr = ""

        try:
            sandbox_result = await execute_python(
                code=unittest_runner_script,
                packages=self.required_packages,  # Pass packages needed by generated code or tests
                # wheels=... # If wheels are needed
                allow_network=False,  # Usually False for unit tests unless they test network code
                allow_fs=False,  # Usually False unless tests interact with mcpfs
                timeout_ms=self.timeout_seconds * 1000,
            )

            if sandbox_result.get("success"):
                sandbox_stdout = sandbox_result.get("stdout", "")
                sandbox_stderr = sandbox_result.get("stderr", "")  # Unittest output now in stdout
                details_output += f"Sandbox STDOUT:\n{sandbox_stdout}\n"
                if sandbox_stderr:  # Still log stderr if sandbox itself had issues
                    details_output += f"Sandbox STDERR:\n{sandbox_stderr}\n"

                # Parse metrics from sandbox_stdout
                # Use re.search with MULTILINE if parsing from a larger block
                run_match = re.search(r"TestsRun:(\d+)", sandbox_stdout)
                fail_match = re.search(r"Failures:(\d+)", sandbox_stdout)
                err_match = re.search(r"Errors:(\d+)", sandbox_stdout)
                rate_match = re.search(r"PassRate:([0-9.]+)", sandbox_stdout)

                if run_match:
                    tests_run = int(run_match.group(1))
                if fail_match:
                    failures = int(fail_match.group(1))
                if err_match:
                    errors = int(err_match.group(1))
                if rate_match:
                    pass_rate = float(rate_match.group(1))
                else:
                    logger.warning(
                        f"UnitTestEvaluator: Could not parse PassRate from sandbox stdout. Output: {sandbox_stdout[:500]}"
                    )
                    details_output += "Warning: Could not parse PassRate from output.\n"
            else:  # Sandbox execution itself failed
                error_msg = sandbox_result.get("error_message", "Sandbox execution failed")
                error_details = sandbox_result.get("error_details", {})
                details_output += (
                    f"Sandbox Execution Failed: {error_msg}\nDetails: {error_details}\n"
                )
                logger.error(
                    f"UnitTestEvaluator: Sandbox execution failed: {error_msg} - {error_details}"
                )
                pass_rate = 0.0

        except (
            ProviderError,
            ToolError,
            ToolInputError,
        ) as e:  # Catch errors from execute_python tool
            logger.error(f"UnitTestEvaluator: Error calling python_sandbox: {e}", exc_info=True)
            details_output += f"Error calling python_sandbox: {str(e)}\n"
            pass_rate = 0.0
        except Exception as e:  # Catch any other unexpected errors
            logger.error(f"UnitTestEvaluator: Unexpected error: {e}", exc_info=True)
            details_output += f"Unexpected error during unit test evaluation: {str(e)}\n"
            pass_rate = 0.0

        return EvaluationScore(
            score=pass_rate * 100,  # Score 0-100
            details=details_output,
            metrics={
                "tests_run": tests_run,
                "failures": failures,
                "errors": errors,
                "pass_rate": pass_rate,
                "sandbox_stdout_len": len(sandbox_stdout),
                "sandbox_stderr_len": len(sandbox_stderr),
            },
        )


@register_evaluator
class RegexMatchEvaluator(Evaluator):
    evaluator_type = "regex_match"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.patterns_str: List[str] = config.get("patterns", [])
        if not self.patterns_str or not isinstance(self.patterns_str, list):
            logger.error("RegexMatchEvaluator: 'patterns' (list of strings) is required in config.")
            self.patterns_str = []

        self.target_field: Literal["response_text", "extracted_code"] = config.get(
            "target_field", "response_text"
        )
        self.match_mode: Literal["all_must_match", "any_can_match", "proportion_matched"] = (
            config.get("match_mode", "all_must_match")
        )

        flag_options_str: Optional[List[str]] = config.get("regex_flag_options")
        self.regex_flags: int = 0
        if flag_options_str:
            for flag_name in flag_options_str:
                if hasattr(re, flag_name.upper()):
                    self.regex_flags |= getattr(re, flag_name.upper())
                else:
                    logger.warning(
                        f"RegexMatchEvaluator: Unknown regex flag '{flag_name}' specified."
                    )

        self.compiled_patterns: List[re.Pattern] = []
        for i, p_str in enumerate(
            self.patterns_str
        ):  # Use enumerate to get index for original string
            try:
                self.compiled_patterns.append(re.compile(p_str, self.regex_flags))
            except re.error as e:
                logger.error(
                    f"RegexMatchEvaluator: Invalid regex pattern '{p_str}' (index {i}): {e}. Skipping this pattern."
                )
                # Add a placeholder or skip to keep lengths consistent if needed,
                # or ensure patterns_str is filtered alongside compiled_patterns.
                # For simplicity now, compiled_patterns might be shorter if errors occur.

    async def score(
        self,
        response_data: ModelResponseData,
        original_prompt: str,
        tournament_type: Literal["code", "text"],
    ) -> EvaluationScore:
        # Iterate using original patterns_str for error reporting if compiled_patterns is shorter
        num_configured_patterns = len(self.patterns_str)

        if not self.compiled_patterns and self.patterns_str:  # Some patterns were invalid
            return EvaluationScore(
                score=0.0,
                details="No valid regex patterns could be compiled from configuration.",
                metrics={
                    "patterns_configured": num_configured_patterns,
                    "patterns_compiled": 0,
                    "patterns_matched": 0,
                },
            )
        if not self.compiled_patterns and not self.patterns_str:  # No patterns provided at all
            return EvaluationScore(
                score=0.0,
                details="No regex patterns configured for matching.",
                metrics={"patterns_configured": 0, "patterns_compiled": 0, "patterns_matched": 0},
            )

        content_to_check: Optional[str] = None
        if self.target_field == "extracted_code":
            content_to_check = response_data.extracted_code
        elif self.target_field == "response_text":
            content_to_check = response_data.response_text
        else:
            return EvaluationScore(
                score=0.0,
                details=f"Invalid target_field '{self.target_field}'.",
                metrics={"patterns_compiled": len(self.compiled_patterns), "patterns_matched": 0},
            )

        if content_to_check is None:
            return EvaluationScore(
                score=0.0,
                details=f"Target content field '{self.target_field}' is empty or None.",
                metrics={"patterns_compiled": len(self.compiled_patterns), "patterns_matched": 0},
            )

        num_matched = 0
        all_patterns_details: List[str] = []

        # Corrected loop over successfully compiled patterns
        for pattern_obj in self.compiled_patterns:
            if pattern_obj.search(content_to_check):
                num_matched += 1
                all_patterns_details.append(f"Pattern '{pattern_obj.pattern}': MATCHED")
            else:
                all_patterns_details.append(f"Pattern '{pattern_obj.pattern}': NOT MATCHED")

        final_score = 0.0
        num_effective_patterns = len(self.compiled_patterns)  # Base score on only valid patterns

        if num_effective_patterns == 0 and num_configured_patterns > 0:  # All patterns were invalid
            details_str = f"Target field: '{self.target_field}'. Mode: '{self.match_mode}'.\nAll {num_configured_patterns} configured regex patterns were invalid and could not be compiled."
            return EvaluationScore(
                score=0.0,
                details=details_str,
                metrics={
                    "patterns_configured": num_configured_patterns,
                    "patterns_compiled": 0,
                    "patterns_matched": 0,
                },
            )
        elif num_effective_patterns == 0 and num_configured_patterns == 0:  # No patterns configured
            details_str = f"Target field: '{self.target_field}'. Mode: '{self.match_mode}'.\nNo regex patterns configured."
            return EvaluationScore(
                score=0.0,
                details=details_str,
                metrics={"patterns_configured": 0, "patterns_compiled": 0, "patterns_matched": 0},
            )

        if self.match_mode == "all_must_match":
            final_score = 100.0 if num_matched == num_effective_patterns else 0.0
        elif self.match_mode == "any_can_match":
            final_score = 100.0 if num_matched > 0 else 0.0
        elif self.match_mode == "proportion_matched":
            final_score = (num_matched / num_effective_patterns) * 100.0

        details_str = f"Target field: '{self.target_field}'. Mode: '{self.match_mode}'.\n"
        details_str += f"Matched {num_matched} out of {num_effective_patterns} validly compiled patterns (from {num_configured_patterns} configured).\n"
        details_str += "\n".join(all_patterns_details)

        return EvaluationScore(
            score=final_score,
            details=details_str,
            metrics={
                "patterns_configured": num_configured_patterns,
                "patterns_compiled": num_effective_patterns,
                "patterns_matched": num_matched,
                "match_proportion_compiled": (num_matched / num_effective_patterns)
                if num_effective_patterns
                else 0.0,
            },
        )
