# llm_gateway/tools/docstring_refiner.py

import asyncio
import copy
import difflib
import json
import math
import random
import re
import time
import traceback
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeAlias, cast

from mcp.server.fastmcp import Context as McpContext
from mcp.types import JSONSchemaObject

# MCP and Pydantic Types
from mcp.types import Tool as McpToolDef
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.completion")

# JSON Schema and Patch Libraries
import jsonschema  # noqa: E402
from jsonpatch import JsonPatchException, apply_patch  # noqa: E402 
from jsonpointer import JsonPointerException  # noqa: E402
from jsonschema.exceptions import SchemaError as JsonSchemaValidationError  # noqa: E402

# Project Imports
from llm_gateway.constants import COST_PER_MILLION_TOKENS, Provider  # noqa: E402
from llm_gateway.exceptions import ProviderError, ToolError, ToolInputError  # noqa: E402
from llm_gateway.tools.base import with_error_handling, with_tool_metrics  # noqa: E402
from llm_gateway.tools.completion import generate_completion  # noqa: E402
from llm_gateway.utils import count_tokens, get_logger  # noqa: E402

# MCP Context Type Hint

logger = get_logger("llm_gateway.tools.docstring_refiner")

# --- Constants ---
DEFAULT_REFINEMENT_PROVIDER = Provider.OPENAI.value
DEFAULT_REFINEMENT_MODEL = "gpt-4.1"
FALLBACK_REFINEMENT_MODEL_PROVIDER = Provider.ANTHROPIC.value
FALLBACK_REFINEMENT_MODEL_NAME = "claude-3-7-sonnet-20250219"
MAX_CONCURRENT_TESTS = 4
MAX_TEST_QUERIES_PER_TOOL = 30
MAX_REFINEMENT_ITERATIONS = 5
SCHEMA_VALIDATION_LEVEL: Literal['none', 'basic', 'full'] = "full"
LLM_AGENT_TIMEOUT_SECONDS = 120.0

# Type Aliases
JsonDict: TypeAlias = Dict[str, Any]
TestStrategy: TypeAlias = Literal[
    "positive_required_only", "positive_optional_mix", "positive_all_optional",
    "negative_type", "negative_required", "negative_enum", "negative_format",
    "negative_range", "negative_length", "negative_pattern",
    "edge_empty", "edge_null", "edge_boundary_min", "edge_boundary_max",
    "llm_realistic_combo", "llm_ambiguity_probe", "llm_simulation_based"
]
ErrorPatternKey: TypeAlias = str
SchemaPath: TypeAlias = str
ProgressCallback: TypeAlias = Optional[Callable[['RefinementProgressEvent'], None]]

# --- Pydantic Models ---

class ErrorCategory(str, Enum):
    MISSING_DESCRIPTION = "Missing Description"
    AMBIGUOUS_DESCRIPTION = "Ambiguous Description"
    INCORRECT_DESCRIPTION = "Incorrect Description"
    MISSING_SCHEMA_CONSTRAINT = "Missing Schema Constraint"
    INCORRECT_SCHEMA_CONSTRAINT = "Incorrect Schema Constraint"
    OVERLY_RESTRICTIVE_SCHEMA = "Overly Restrictive Schema"
    TYPE_CONFUSION = "Schema Type Confusion"
    MISSING_EXAMPLE = "Missing Example"
    MISLEADING_EXAMPLE = "Misleading Example"
    INCOMPLETE_EXAMPLE = "Incomplete Example"
    PARAMETER_DEPENDENCY_UNCLEAR = "Unclear Parameter Dependencies"
    CONFLICTING_CONSTRAINTS = "Conflicting Constraints"
    AGENT_FORMULATION_ERROR = "Agent Formulation Error"
    SCHEMA_PREVALIDATION_FAILURE = "Schema Pre-validation Failure"
    TOOL_EXECUTION_ERROR = "Tool Execution Error"
    UNKNOWN = "Unknown Documentation Issue"

class TestCase(BaseModel):
    model_config = ConfigDict(frozen=True)
    strategy_used: TestStrategy
    arguments: JsonDict
    description: str
    targets_previous_failure: bool = False
    schema_validation_error: Optional[str] = None

class TestExecutionResult(BaseModel):
    test_case: TestCase
    start_time: float
    end_time: float
    success: bool
    result_preview: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    error_category_guess: Optional[ErrorCategory] = None
    error_details: Optional[JsonDict] = None

    @property
    def processing_time(self) -> float:
        return self.end_time - self.start_time

class ParameterSchemaPatch(BaseModel):
    parameter_name: str
    patch_ops: List[JsonDict] = Field(...)
    reasoning: str
    estimated_impact: Literal["high", "medium", "low"]

class GeneratedExample(BaseModel):
    args: JsonDict
    comment: str
    addresses_failure_pattern: Optional[ErrorPatternKey] = None

class RefinementAnalysis(BaseModel):
    overall_diagnosis: str
    error_pattern_analysis: Dict[ErrorPatternKey, str] = Field(default_factory=dict)
    identified_flaw_categories: List[ErrorCategory] = Field(default_factory=list)
    parameter_schema_patches: List[ParameterSchemaPatch] = Field(default_factory=list)
    general_suggestions: List[str] = Field(default_factory=list)
    improvement_confidence: float = Field(..., ge=0.0, le=1.0)
    remaining_ambiguity_analysis: Optional[str] = None
    hypothetical_error_resolution: str

class ProposedChanges(BaseModel):
    description: str
    schema_patches: List[ParameterSchemaPatch] = Field(default_factory=list)
    examples: List[GeneratedExample] = Field(default_factory=list)
    schema_modified: bool = False

    @model_validator(mode='after')
    def check_schema_modified(self):
        self.schema_modified = bool(self.schema_patches)
        return self

class AgentSimulationResult(BaseModel):
    task_description: str
    tool_selected: Optional[str] = None
    arguments_formulated: Optional[JsonDict] = None
    formulation_success: bool
    selection_error: Optional[str] = None
    formulation_error: Optional[str] = None
    reasoning: Optional[str] = None
    confidence_score: Optional[float] = None

class RefinementProgressEvent(BaseModel):
    tool_name: str
    iteration: int
    total_iterations: int
    stage: Literal[
        "starting_iteration", "agent_simulation", "test_generation",
        "test_execution_start", "test_execution_progress", "test_execution_end",
        "analysis_start", "analysis_end", "schema_patching", "winnowing",
        "iteration_complete", "tool_complete", "error"
    ]
    progress_pct: Optional[float] = None
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    details: Optional[JsonDict] = None

class RefinementIterationResult(BaseModel):
    iteration: int
    documentation_used: ProposedChanges
    schema_used: JSONSchemaObject
    agent_simulation_results: List[AgentSimulationResult]
    test_cases_generated: List[TestCase]
    test_results: List[TestExecutionResult]
    success_rate: float
    validation_failure_rate: float
    analysis: Optional[RefinementAnalysis] = None
    proposed_changes: Optional[ProposedChanges] = None
    applied_schema_patches: List[JsonDict] = Field(default_factory=list)
    description_diff: Optional[str] = None
    schema_diff: Optional[str] = None

class RefinedToolResult(BaseModel):
    tool_name: str
    original_schema: JSONSchemaObject
    iterations: List[RefinementIterationResult]
    final_proposed_changes: ProposedChanges
    final_schema_after_patches: JSONSchemaObject
    initial_success_rate: float
    final_success_rate: float
    improvement_factor: float
    token_count_change: Optional[int] = None
    process_error: Optional[str] = None

class DocstringRefinementResult(BaseModel):
    report_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    refined_tools: List[RefinedToolResult]
    refinement_model_configs: List[JsonDict]
    total_iterations_run: int
    total_test_calls_attempted: int
    total_test_calls_failed: int
    total_schema_validation_failures: int
    total_agent_simulation_failures: int
    total_refinement_cost: float
    total_processing_time: float
    errors_during_refinement_process: List[str] = Field(default_factory=list)
    success: bool = True

class TestCaseGenerationStrategy(BaseModel):
    """
    Configuration controlling the number of test cases generated for each strategy.
    Adjust counts to focus testing on specific areas based on tool complexity or
    observed failure patterns. The total number of tests generated will be the sum
    of these counts, capped by MAX_TEST_QUERIES_PER_TOOL.
    """
    # --- Positive Strategies ---
    positive_required_only: int = Field(
        default=2, ge=0,
        description="Number of tests using only required parameters."
    )
    positive_optional_mix: int = Field(
        default=3, ge=0,
        description="Number of tests using required parameters plus a random subset of optional parameters."
    )
    positive_all_optional: int = Field(
        default=1, ge=0,
        description="Number of tests using required parameters plus all available optional parameters."
    )

    # --- Negative Strategies (Schema Violations) ---
    negative_type: int = Field(
        default=2, ge=0,
        description="Number of tests providing a parameter with the wrong data type (e.g., string for integer)."
    )
    negative_required: int = Field(
        default=2, ge=0,
        description="Number of tests deliberately missing one required parameter."
    )
    negative_enum: int = Field(
        default=1, ge=0,
        description="Number of tests providing a value not listed in a parameter's 'enum' constraint."
    )
    negative_format: int = Field(
        default=1, ge=0,
        description="Number of tests providing a value violating a parameter's 'format' constraint (e.g., invalid email, date)."
    )
    negative_range: int = Field(
        default=1, ge=0,
        description="Number of tests providing a numeric value outside 'minimum'/'maximum' constraints."
    )
    negative_length: int = Field(
        default=1, ge=0,
        description="Number of tests providing a string/array violating 'minLength'/'maxLength' constraints."
    )
    negative_pattern: int = Field(
        default=1, ge=0,
        description="Number of tests providing a string violating a parameter's 'pattern' (regex) constraint."
    )

    # --- Edge Case Strategies ---
    edge_empty: int = Field(
        default=1, ge=0,
        description="Number of tests using empty values (empty string, list, object) where applicable."
    )
    edge_null: int = Field(
        default=1, ge=0,
        description="Number of tests using null/None for optional parameters or those explicitly allowing null."
    )
    edge_boundary_min: int = Field(
        default=1, ge=0,
        description="Number of tests using the exact 'minimum' or 'minLength' value."
    )
    edge_boundary_max: int = Field(
        default=1, ge=0,
        description="Number of tests using the exact 'maximum' or 'maxLength' value."
    )

    # --- LLM-Driven Strategies ---
    llm_realistic_combo: int = Field(
        default=3, ge=0,
        description="Number of tests generated by an LLM simulating realistic, potentially complex, combinations of parameters based on documentation."
    )
    llm_ambiguity_probe: int = Field(
        default=2, ge=0,
        description="Number of tests generated by an LLM specifically trying to probe potential ambiguities or underspecified aspects of the documentation."
    )
    llm_simulation_based: int = Field(
        default=1, ge=0,
        description="Number of tests derived directly from failed agent simulation attempts (if simulation is run)."
    )

    @property
    def total_requested(self) -> int:
        """Calculate the total number of tests requested by this configuration."""
        return sum(getattr(self, field_name) for field_name in self.model_fields)
    
# --- Helper Functions ---

async def _estimate_cost(prompt_tokens: int, completion_tokens: int, model_id: str) -> float:
    """Estimate cost based on token counts and model ID."""
    cost_data = COST_PER_MILLION_TOKENS.get(model_id)
    resolved_model_key = model_id
    if not cost_data and '/' in model_id:
        potential_short_key = model_id.split('/')[-1]
        cost_data = COST_PER_MILLION_TOKENS.get(potential_short_key)
        if cost_data: 
            resolved_model_key = potential_short_key

    if not cost_data:
        logger.warning(f"Cost data unavailable for model '{model_id}' (using key '{resolved_model_key}'). Cannot estimate cost.")
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * cost_data["input"]
    output_cost = (completion_tokens / 1_000_000) * cost_data["output"]
    return input_cost + output_cost

def _create_diff(original: str, proposed: str, fromfile: str, tofile: str) -> Optional[str]:
    """Creates a unified diff string, truncated if necessary."""
    if original == proposed: 
        return None
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile=fromfile, tofile=tofile, lineterm="\n"
    ))
    if not diff_lines: 
        return None

    diff_str = "".join(diff_lines)
    MAX_DIFF_LEN = 8000
    if len(diff_str) > MAX_DIFF_LEN:
        diff_str = diff_str[:MAX_DIFF_LEN] + "\n... [Diff Truncated] ..."
    return diff_str

def _create_schema_diff(original_schema: JsonDict, patched_schema: JsonDict) -> Optional[str]:
    """Creates a diff between two schema dictionaries."""
    return _create_diff(
        json.dumps(original_schema, indent=2, sort_keys=True),
        json.dumps(patched_schema, indent=2, sort_keys=True),
        "original_schema.json",
        "proposed_schema.json"
    )

def _apply_schema_patches(base_schema: JsonDict, patch_suggestions: List[ParameterSchemaPatch]) -> Tuple[JsonDict, List[str], List[JsonDict]]:
    """Applies validated patches, returns patched schema, errors, and applied patches."""
    patched_schema = copy.deepcopy(base_schema)
    errors = []
    successfully_applied_patches: List[JsonDict] = []
    all_ops: List[JsonDict] = []

    for suggestion in patch_suggestions:
        all_ops.extend(suggestion.patch_ops)

    if not all_ops: 
        return patched_schema, errors, successfully_applied_patches

    try:
        result_schema = apply_patch(patched_schema, all_ops, inplace=False)

        try:
            jsonschema.Draft7Validator.check_schema(result_schema)
            if "properties" in result_schema and "required" in result_schema:
                props_set = set(result_schema.get("properties", {}).keys())
                req_set = set(result_schema.get("required", []))
                missing_req = req_set - props_set
                if missing_req:
                        raise JsonPatchException(f"Patched schema invalid: Required properties {missing_req} not defined.")
            logger.debug(f"Applied and validated {len(all_ops)} schema patch operations.")
            successfully_applied_patches = all_ops
            return result_schema, errors, successfully_applied_patches
        except (JsonSchemaValidationError, JsonPatchException) as val_err:
                errors.append(f"Schema validation failed after patching: {val_err}. Patches: {all_ops}")
                logger.error(f"Schema validation failed after patching: {val_err}. Patches: {all_ops}")
                return copy.deepcopy(base_schema), errors, []
        else:
            logger.warning("jsonschema not available, skipping post-patch validation.")
            successfully_applied_patches = all_ops
            return result_schema, errors, successfully_applied_patches

    except (JsonPatchException, JsonPointerException) as e:
        errors.append(f"Failed to apply schema patches: {e}. Patches: {all_ops}")
        logger.error(f"Schema patch application failed: {e}. Patches: {all_ops}")
        return copy.deepcopy(base_schema), errors, []
    except Exception as e:
        errors.append(f"Unexpected error applying schema patches: {e}. Patches: {all_ops}")
        logger.error(f"Unexpected error applying schema patches: {e}", exc_info=True)
        return copy.deepcopy(base_schema), errors, []

def _validate_args_against_schema(args: JsonDict, schema: JSONSchemaObject) -> Optional[str]:
    """Validate arguments against schema using jsonschema."""
    try:
        jsonschema.validate(instance=args, schema=schema)
        return None
    except jsonschema.ValidationError as e:
        return f"Schema validation failed: {e.message} (Path: {'/'.join(map(str, e.path))})"
    except Exception as e:
        logger.error(f"Unexpected error during jsonschema validation: {e}", exc_info=True)
        return f"Unexpected validation error: {str(e)}"

def _create_value_for_schema(schema: JsonDict, strategy: str = "positive", history: Optional[List[Any]] = None) -> Any:
    """Generates a single value based on JSON schema type, format, constraints, and strategy."""
    schema_type = schema.get("type", "any")
    fmt = schema.get("format")
    enum = schema.get("enum")
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    minLength = schema.get("minLength")
    maxLength = schema.get("maxLength")
    pattern = schema.get("pattern")
    history = history or [] # Avoid generating same value repeatedly

    # --- Negative Strategies ---
    if strategy == "type_mismatch":
        if schema_type == "string": 
            return random.choice([12345, True, None, ["list"], {"key": "val"}])
        if schema_type == "integer": 
            return random.choice(["not_int", 123.45, True, None])
        if schema_type == "number": 
            return random.choice(["not_num", True, None])
        if schema_type == "boolean": 
            return random.choice(["true", 1, 0, None])
        if schema_type == "array":
            return {"not": "a list"}
        if schema_type == "object": 
            return ["not", "an", "object"]
        return "mismatched_type" # Fallback
    if strategy == "negative_required": 
        return ... # Handled by skipping
    if strategy.startswith("negative_constraint") or strategy.startswith("negative_enum") or \
       strategy.startswith("negative_format") or strategy.startswith("negative_range") or \
       strategy.startswith("negative_length") or strategy.startswith("negative_pattern"):
        # Simplified negative constraint generation
        if enum:
            return f"not_in_enum_{random.randint(1,100)}"
        if fmt == "date": 
            return "2024/13/32" # Invalid date format
        if fmt == "email": 
            return "user@domain." # Invalid email
        if fmt == "uri": 
            return "bad-schema://example com"
        if pattern: 
            return "!@#$%^&*()" # Likely violates most patterns
        if minimum is not None:
            return minimum - (1 if schema_type == "integer" else 0.1)
        if maximum is not None:
            return maximum + (1 if schema_type == "integer" else 0.1)
        if minLength is not None:
            return "a" * max(0, minLength - 1)
        if maxLength is not None: 
            return "a" * (maxLength + 1)
        return "constraint_violation_value" # Fallback

    # --- Edge Cases ---
    if strategy == "edge_empty":
        if schema_type == "string": 
            return ""
        if schema_type == "array":
            return []
        if schema_type == "object": 
            return {}
    if strategy == "edge_null": 
        return None
    if strategy == "edge_boundary_min" and minimum is not None:
        return minimum
    if strategy == "edge_boundary_max" and maximum is not None: 
        return maximum

    # --- Positive Cases (Default) ---
    if enum:
        return random.choice(enum)
    val = None
    if schema_type == "string":
        if fmt == "date":
            val = datetime.now().strftime("%Y-%m-%d")
        elif fmt == "date-time":
            val = datetime.now().isoformat() + "Z"
        elif fmt == "email":
            val = f"test.{random.randint(100,999)}@example.com"
        elif fmt == "uri": 
            val = f"https://example.com/item{random.randint(1,99)}"
        elif pattern: 
            val = "ABC-123" # Generic placeholder for pattern
        else: 
            val = f"Example String {random.randint(1,100)}"
        # Respect length constraints for positive cases
        if maxLength is not None: 
            val = val[:maxLength]
        if minLength is not None and len(val) < minLength: 
            val = val.ljust(minLength, 'X')
    elif schema_type == "integer":
        low = int(minimum if minimum is not None else 0)
        high = int(maximum if maximum is not None else 100)
        val = random.randint(low, high)
    elif schema_type == "number":
        low = float(minimum if minimum is not None else 0.0)
        high = float(maximum if maximum is not None else 100.0)
        val = round(random.uniform(low, high), 4)
    elif schema_type == "boolean":
        val = random.choice([True, False])
    elif schema_type == "array":
        minItems = schema.get("minItems", 0)
        maxItems = schema.get("maxItems", 3)
        num = random.randint(minItems, maxItems)
        item_schema = schema.get("items", {"type": "string"})
        val = [_create_value_for_schema(item_schema, "positive", history + [num]) for _ in range(num)]
    elif schema_type == "object":
        props = schema.get("properties", {f"key{random.randint(1,2)}": {"type": "string"}})
        required = set(schema.get("required", []))
        obj = {}
        for p_name, p_schema in props.items():
             if p_name in required or random.choice([True, False]):
                  obj[p_name] = _create_value_for_schema(p_schema, "positive", history + [p_name])
        val = obj
    elif schema_type == "null": 
        val = None
    else: 
        val = f"any_value_{random.randint(1,5)}" # any or unknown

    if val in history and len(history) < 5: # Simple cycle breaker
        return _create_value_for_schema(schema, strategy, history + [val])
    return val


def _select_test_strategies(
    num_tests: int,
    schema: JSONSchemaObject,
    previous_results: Optional[List['TestExecutionResult']] = None # Use forward ref
) -> List[Tuple[TestStrategy, int]]:
    """
    Selects test strategies and counts based on schema complexity and previous results.

    Prioritizes strategies likely to uncover errors based on schema features
    and past failures, then distributes the requested number of tests proportionally.

    Args:
        num_tests: Total number of test cases to generate across all strategies.
        schema: The JSON Schema object for the tool's input.
        previous_results: Optional list of results from the previous refinement iteration.

    Returns:
        A list of tuples, where each tuple contains a TestStrategy and the
        integer count of tests to generate for that strategy. The sum of counts
        will equal num_tests.
    """
    # --- Base Weights ---
    # Start with default weights representing general importance/likelihood of issues
    weights: Dict[TestStrategy, float] = {
        # Positive Cases (essential baseline)
        "positive_required_only": 2.0,
        "positive_optional_mix": 3.0,
        "positive_all_optional": 1.0,
        # Negative Schema Violations (common issues)
        "negative_type": 2.5,
        "negative_required": 2.5,
        "negative_enum": 1.5,
        "negative_format": 1.5,
        "negative_range": 1.5,
        "negative_length": 1.5,
        "negative_pattern": 1.0,
        # Edge Cases (important for robustness)
        "edge_empty": 1.0,
        "edge_null": 1.0,
        "edge_boundary_min": 1.5,
        "edge_boundary_max": 1.5,
        # LLM-Driven (exploratory)
        "llm_realistic_combo": 3.0, # Often useful complex cases
        "llm_ambiguity_probe": 2.0,
        "llm_simulation_based": 1.5
    }

    # --- Adjust Weights Based on Schema Complexity ---
    props = schema.get("properties", {})
    required_params = set(schema.get("required", []))
    optional_params = set(props.keys()) - required_params
    has_constraints = False

    if not props: # Very simple schema (e.g., just a single root type)
        weights["positive_required_only"] *= 1.5 # Focus on basic valid input
        weights["negative_type"] *= 1.5
        for k in list(weights.keys()): # Reduce weight of complex strategies
             if k.startswith("negative_constraint") or k.startswith("llm_") or k == "positive_optional_mix" or k == "positive_all_optional":
                 weights[k] *= 0.1
    else:
        # Adjust based on optional params
        if len(optional_params) == 0: # No optional params
            weights["positive_optional_mix"] = 0
            weights["positive_all_optional"] = 0
        elif len(optional_params) > 4: # Many optional params
            weights["positive_optional_mix"] *= 1.5
            weights["positive_all_optional"] *= 1.5
            weights["llm_realistic_combo"] *= 1.2 # More combos to test

        # Adjust based on constraints found
        constraint_counts = defaultdict(int)
        for param_schema in props.values():
            if not isinstance(param_schema, dict):
                continue # Skip invalid schema parts
            if "enum" in param_schema:
                 constraint_counts["enum"] += 1
                 has_constraints = True
            if "format" in param_schema:
                 constraint_counts["format"] += 1
                 has_constraints = True
            if "minimum" in param_schema or "maximum" in param_schema: 
                constraint_counts["range"] += 1
                has_constraints = True
            if "minLength" in param_schema or "maxLength" in param_schema:
                 constraint_counts["length"] += 1
                 has_constraints = True
            if "pattern" in param_schema:
                 constraint_counts["pattern"] += 1
                 has_constraints = True

        # Boost corresponding negative strategies based on constraint counts
        if constraint_counts["enum"] > 0:
            weights["negative_enum"] *= (1 + 0.5 * constraint_counts["enum"])
        if constraint_counts["format"] > 0:
            weights["negative_format"] *= (1 + 0.5 * constraint_counts["format"])
        if constraint_counts["range"] > 0:
             weights["negative_range"] *= (1 + 0.5 * constraint_counts["range"])
             weights["edge_boundary_min"] *= 1.2
             weights["edge_boundary_max"] *= 1.2
        if constraint_counts["length"] > 0:
            weights["negative_length"] *= (1 + 0.5 * constraint_counts["length"])
        if constraint_counts["pattern"] > 0:
            weights["negative_pattern"] *= (1 + 0.5 * constraint_counts["pattern"])

        # Boost ambiguity probe if schema has many properties or constraints
        if len(props) > 8 or has_constraints:
            weights["llm_ambiguity_probe"] *= 1.3

        # Reduce weight for missing required if none exist
        if not required_params:
            weights["negative_required"] = 0


    # --- Adjust Weights Based on Previous Errors ---
    if previous_results:
        error_codes: Dict[str, int] = defaultdict(int)
        validation_failures = 0
        strategy_failures: Dict[TestStrategy, int] = defaultdict(int)

        for r in previous_results:
            if r.test_case.schema_validation_error:
                validation_failures += 1
            if not r.success:
                code = r.error_code or "UNKNOWN_EXEC_ERROR"
                error_codes[code] += 1
                strategy_failures[r.test_case.strategy_used] += 1

        total_failures = sum(error_codes.values())
        if total_failures > 0:
            logger.debug(f"Adjusting weights based on {total_failures} previous failures...")
            # Increase weight for strategies that previously failed
            for strategy, fail_count in strategy_failures.items():
                 if strategy in weights:
                      weights[strategy] *= (1 + (fail_count / total_failures) * 2.0) # Strong boost for failing strategies

            # Increase weights for strategies targeting common error types
            for code, count in error_codes.items():
                ratio = count / total_failures
                boost_factor = 1 + ratio * 1.5 # Moderate boost based on frequency

                if "VALIDATION_ERROR" in code or "INVALID_PARAMETER" in code or code == "SCHEMA_VALIDATION_FAILED_PRE_CALL":
                    weights["negative_type"] *= boost_factor
                    weights["negative_required"] *= boost_factor
                    weights["negative_enum"] *= boost_factor
                    weights["negative_format"] *= boost_factor
                    weights["negative_range"] *= boost_factor
                    weights["negative_length"] *= boost_factor
                    weights["negative_pattern"] *= boost_factor
                    weights["edge_empty"] *= (1 + ratio * 0.5) # Slight boost for edges
                    weights["edge_boundary_min"] *= (1 + ratio * 0.5)
                    weights["edge_boundary_max"] *= (1 + ratio * 0.5)
                elif "TypeError" in code:
                    weights["negative_type"] *= (1 + ratio * 2.0) # Stronger boost for type errors
                elif "PROVIDER_ERROR" in code or "TOOL_EXECUTION_FAILED" in code or "UNEXPECTED" in code:
                    # These might indicate complex interactions or doc ambiguity
                    weights["llm_realistic_combo"] *= boost_factor
                    weights["llm_ambiguity_probe"] *= (1 + ratio * 1.5)

            # Boost constraint/type tests if schema pre-validation failed often
            if validation_failures > 0 and len(previous_results) > 0:
                 validation_fail_ratio = validation_failures / len(previous_results)
                 boost = 1 + validation_fail_ratio * 1.5
                 weights["negative_type"] *= boost
                 weights["negative_enum"] *= boost
                 weights["negative_format"] *= boost
                 weights["negative_range"] *= boost
                 weights["negative_length"] *= boost
                 weights["negative_pattern"] *= boost


    # --- Normalize Weights and Calculate Counts ---
    # Filter out strategies with zero weight
    active_strategies = {k: v for k, v in weights.items() if v > 0}
    if not active_strategies:
        # Fallback: distribute evenly among positive strategies if all weights zeroed out
        positive_strategies = [s for s in TestStrategy.__args__ if s.startswith("positive_")]
        if not positive_strategies: 
            positive_strategies = ["positive_required_only"] # Absolute fallback
        logger.warning("All strategy weights became zero, falling back to positive strategies.")
        active_strategies = {s: 1.0 for s in positive_strategies}

    total_weight = sum(active_strategies.values())
    if total_weight <= 0: # Prevent division by zero
        # Distribute evenly among active (or fallback positive) if total weight is zero
        num_active = len(active_strategies)
        base_count = num_tests // num_active
        remainder = num_tests % num_active
        final_counts = [(cast(TestStrategy, s), base_count) for s in active_strategies]
        for i in range(remainder): 
            final_counts[i] = (final_counts[i][0], final_counts[i][1] + 1)
        logger.debug(f"Selected test strategies (Even distribution due to zero total weight): {final_counts}")
        return final_counts

    # Calculate ideal counts based on weights
    ideal_counts: Dict[TestStrategy, float] = {
        strategy: (weight / total_weight) * num_tests
        for strategy, weight in active_strategies.items()
    }

    # Allocate integer counts, prioritizing strategies with larger fractional parts
    allocated_counts: Dict[TestStrategy, int] = {s: math.floor(c) for s, c in ideal_counts.items()}
    remainders = {s: c - math.floor(c) for s, c in ideal_counts.items()}
    tests_allocated = sum(allocated_counts.values())
    tests_remaining = num_tests - tests_allocated

    # Distribute remaining tests based on largest remainders
    sorted_by_remainder = sorted(remainders.items(), key=lambda item: item[1], reverse=True)

    for i in range(tests_remaining):
        strategy_to_increment = sorted_by_remainder[i % len(sorted_by_remainder)][0]
        allocated_counts[strategy_to_increment] += 1

    # Format final result, filtering out strategies with zero count
    final_counts_tuples: List[Tuple[TestStrategy, int]] = [
        (strategy, count) for strategy, count in allocated_counts.items() if count > 0
    ]

    # Final sanity check to ensure total count matches num_tests
    final_sum = sum(count for _, count in final_counts_tuples)
    if final_sum != num_tests:
        logger.warning(f"Final test count ({final_sum}) doesn't match requested ({num_tests}). Adjusting...")
        # Simple adjustment: add/remove from the strategy with the median count
        if final_counts_tuples:
            final_counts_tuples.sort(key=lambda x:x[1]) # Sort by count
            median_idx = len(final_counts_tuples) // 2
            diff = num_tests - final_sum
            new_count = max(0, final_counts_tuples[median_idx][1] + diff)
            final_counts_tuples[median_idx] = (final_counts_tuples[median_idx][0], new_count)
            # Re-filter zero counts if adjustment made a count zero
            final_counts_tuples = [(s, c) for s, c in final_counts_tuples if c > 0]
        else: # Should not happen if num_tests > 0
            logger.error("Cannot adjust test counts, final_counts_tuples is empty.")


    logger.debug(f"Selected test strategies: {final_counts_tuples}")
    return final_counts_tuples

async def _generate_test_cases(
    tool_name: str,
    tool_schema: JSONSchemaObject,
    tool_description: str,
    num_tests: int,
    refinement_model_config: Dict[str, Any],
    validation_level: str,
    previous_results: Optional[List['TestExecutionResult']] = None, # Forward reference
    agent_sim_results: Optional[List['AgentSimulationResult']] = None # Forward reference
) -> Tuple[List[TestCase], float]:
    """
    Generate diverse test cases using adaptive strategy, schema validation, and simulation guidance.

    Args:
        tool_name: Name of the tool.
        tool_schema: JSON schema for the tool's input.
        tool_description: Current description of the tool.
        num_tests: Target number of test cases to generate.
        refinement_model_config: LLM configuration for generating realistic tests.
        validation_level: Level of schema validation ('none', 'basic', 'full').
        previous_results: Results from the previous iteration to guide strategy.
        agent_sim_results: Results from agent simulation to guide strategy.

    Returns:
        Tuple containing a list of generated TestCase objects and the estimated cost
        of LLM calls during test generation.
    """
    total_cost = 0.0
    test_cases: List[TestCase] = []
    seen_args_hashes = set() # Track generated args to avoid exact duplicates

    # --- 1. Select Test Strategies Adaptively ---
    logger.debug(f"Selecting test strategies for {tool_name} based on schema and previous results...")
    # Ensure _select_test_strategies handles potential errors and returns a valid list
    try:
        strategy_counts = _select_test_strategies(num_tests, tool_schema, previous_results)
    except Exception as e:
        logger.error(f"Failed to select test strategies for {tool_name}: {e}. Falling back to basic.", exc_info=True)
        # Fallback strategy: focus on positive cases
        strategy_counts = [("positive_required_only", max(1, num_tests // 2)),
                           ("positive_optional_mix", num_tests - max(1, num_tests // 2))]

    # --- Schema Information ---
    schema_properties = tool_schema.get("properties", {})
    required_params = set(tool_schema.get("required", []))
    all_param_names = list(schema_properties.keys())
    param_schemas = {name: schema_properties.get(name, {}) for name in all_param_names}

    # --- Helper to Add Unique Test Case ---
    def add_unique_test_case(strategy: TestStrategy, args: JsonDict, description: str, targets_failure: bool = False):
        """Adds unique test case with pre-validation."""
        # Create a hashable representation of the arguments
        try:
            # Sort dictionary items for consistent hashing
            items_tuple = tuple(sorted(args.items()))
            args_hash = hash(items_tuple)
        except TypeError:
            # Fallback if args contain unhashable types (less reliable duplicate check)
            try:
                args_hash = hash(json.dumps(args, sort_keys=True))
            except TypeError:
                logger.warning(f"Could not hash arguments for duplicate check: {args}")
                args_hash = random.random() # Use random hash, might allow duplicates

        if args_hash not in seen_args_hashes:
            validation_error = None
            # Perform validation based on level
            should_validate = (validation_level == 'full') or \
                              (validation_level == 'basic' and strategy.startswith('positive_'))
            if should_validate and schema_properties:
                validation_error = _validate_args_against_schema(args, tool_schema)
                if validation_error:
                    logger.debug(f"Schema validation failed for generated args (Strategy: {strategy}): {validation_error}. Args: {args}")

            test_cases.append(TestCase(
                strategy_used=strategy,
                arguments=args,
                description=description,
                targets_previous_failure=targets_failure,
                schema_validation_error=validation_error if validation_level == "full" else None
            ))
            seen_args_hashes.add(args_hash)
            return True
        return False

    # --- 2. Generate Schema-Based Test Cases ---
    logger.debug(f"Generating schema-based test cases for {tool_name}...")
    if schema_properties:
        for strategy, count in strategy_counts:
            if strategy.startswith("llm_"): 
                continue # Handle LLM strategies separately

            generated_count = 0
            attempts = 0
            max_attempts = count * 5 # Try multiple times to get unique/valid cases

            while generated_count < count and attempts < max_attempts:
                attempts += 1
                case: Optional[JsonDict] = None # Use Optional type hint
                targeted_param: Optional[str] = None
                strategy_desc = f"Test case for strategy: {strategy}"
                targets_failure = False # Default

                # --- Build the 'case' dictionary based on 'strategy' ---
                # (This section implements the logic based on strategy name)
                try:
                    if strategy == "positive_required_only":
                        case = {p: _create_value_for_schema(param_schemas[p], "positive") for p in required_params}
                        strategy_desc = "Valid call: only required parameters."
                    elif strategy == "positive_optional_mix":
                        case = {p: _create_value_for_schema(param_schemas[p], "positive") for p in required_params}
                        optionals = [p for p in all_param_names if p not in required_params]
                        num_opts = random.randint(1, len(optionals)) if optionals else 0
                        for pname in random.sample(optionals, k=num_opts):
                            case[pname] = _create_value_for_schema(param_schemas[pname], "positive")
                        strategy_desc = "Valid call: required + some optional parameters."
                    elif strategy == "positive_all_optional":
                         case = {p: _create_value_for_schema(param_schemas[p], "positive") for p in required_params}
                         optionals = [p for p in all_param_names if p not in required_params]
                         for pname in optionals:
                             case[pname] = _create_value_for_schema(param_schemas[pname], "positive")
                         strategy_desc = "Valid call: required + all optional parameters."
                    elif strategy == "negative_required":
                        if not required_params: 
                            continue # Skip if no required params
                        targeted_param = random.choice(list(required_params))
                        case = {p: _create_value_for_schema(param_schemas[p], "positive")
                                for p in required_params if p != targeted_param}
                        strategy_desc = f"Missing required parameter: '{targeted_param}'."
                        targets_failure = True
                    elif strategy.startswith("negative_") or strategy.startswith("edge_"):
                        # Determine which parameter to target for this specific negative/edge strategy
                        strategy_type = strategy.split('_', 1)[1] # e.g., 'type', 'enum', 'empty'
                        suitable_params = []
                        if strategy_type == "type": 
                            suitable_params = all_param_names
                        elif strategy_type == "enum": 
                            suitable_params = [p for p,s in param_schemas.items() if "enum" in s]
                        elif strategy_type == "format": 
                            suitable_params = [p for p,s in param_schemas.items() if "format" in s]
                        elif strategy_type == "range": 
                            suitable_params = [p for p,s in param_schemas.items() if "minimum" in s or "maximum" in s]
                        elif strategy_type == "length": 
                            suitable_params = [p for p,s in param_schemas.items() if "minLength" in s or "maxLength" in s]
                        elif strategy_type == "pattern": 
                            suitable_params = [p for p,s in param_schemas.items() if "pattern" in s]
                        elif strategy_type == "empty": 
                            suitable_params = [p for p,s in param_schemas.items() if s.get("type") in ["string", "array", "object"]]
                        elif strategy_type == "null": 
                            suitable_params = [p for p,s in param_schemas.items() if s.get("type") == "null" or "null" in s.get("type", []) or p not in required_params]
                        elif strategy_type == "boundary_min": 
                            suitable_params = [p for p,s in param_schemas.items() if "minimum" in s]
                        elif strategy_type == "boundary_max": 
                            suitable_params = [p for p,s in param_schemas.items() if "maximum" in s]

                        if not suitable_params: 
                            continue # Cannot apply this strategy
                        targeted_param = random.choice(suitable_params)
                        value = _create_value_for_schema(param_schemas[targeted_param], strategy) # Pass full strategy name

                        if value is ...: 
                            continue # Sentinel means skip this attempt

                        case = {targeted_param: value}
                        # Fill other required params correctly
                        for pname in required_params:
                            if pname != targeted_param:
                                 case[pname] = _create_value_for_schema(param_schemas[pname], "positive")
                        strategy_desc = f"Strategy '{strategy}' targeting parameter '{targeted_param}'."
                        targets_failure = strategy.startswith("negative_") # Negative strategies target failure

                    # Add the generated case if valid for the strategy
                    if case is not None: # Ensure case was generated
                        if add_unique_test_case(cast('TestStrategy', strategy), case, strategy_desc, targets_failure):
                            generated_count += 1
                except Exception as gen_err:
                     logger.warning(f"Error generating test case for strategy '{strategy}' on tool '{tool_name}': {gen_err}", exc_info=True)
                     attempts += 1 # Count as attempt even if generation failed

    # --- 3. Generate LLM-Based Test Cases ---
    logger.debug(f"Generating LLM-based test cases for {tool_name}...")
    llm_strategy_counts = {s: c for s, c in strategy_counts if s.startswith("llm_")}
    for llm_strategy, count in llm_strategy_counts.items():
        if count <= 0: 
            continue

        # Prepare prompt based on strategy
        sim_summary = ""
        if agent_sim_results:
             sim_summary += "Insights from Agent Simulation (Task -> Success -> Reasoning/Error):\n"
             for r in agent_sim_results[:2]: # Limit context
                  sim_summary += f'- "{r.task_description}" -> {r.formulation_success} -> {r.reasoning or r.formulation_error or 'N/A'}\n'

        if llm_strategy == "llm_realistic_combo":
            focus = "Generate realistic, potentially complex, combinations of parameters an agent might plausibly use based on the description and schema."
        elif llm_strategy == "llm_ambiguity_probe":
            focus = "Generate argument sets specifically designed to probe potential ambiguities, edge cases, or underspecified aspects suggested by the description, schema, or simulation insights."
        elif llm_strategy == "llm_simulation_based":
             focus = "Based *specifically* on the Agent Simulation Insights provided, generate argument sets that reflect the kinds of formulations (both successful and potentially flawed) the simulated agent attempted."
             if not agent_sim_results: 
                 focus = "Generate realistic combinations (simulation results unavailable)." # Fallback if no sim results
        else:
             focus = "Generate diverse and realistic argument sets." # Default

        schema_str = json.dumps(tool_schema, indent=2)
        prompt = f"""Analyze the MCP tool documentation and any provided simulation insights. Generate exactly {count} diverse argument sets. {focus} Base generation *only* on the provided documentation and simulation insights.

Tool Description: {tool_description or '(None)'}
Input Schema:
```json
{schema_str}
```
{sim_summary if llm_strategy != 'llm_realistic_combo' else ''}
Output ONLY a valid JSON list of {count} argument dictionaries.""" # Don't overload realistic prompt with sim results unless probing

        try:
            result = await generate_completion(
                prompt=prompt, **refinement_model_config, temperature=0.8, # Higher creativity
                max_tokens=min(4000, 350 * count + 500), # Allow more tokens
                additional_params={"response_format": {"type": "json_object"}} if refinement_model_config.get("provider") == Provider.OPENAI.value else None
            )
            if not result.get("success"):
                 logger.warning(f"LLM failed test gen ({llm_strategy}) for {tool_name}: {result.get('error')}")
                 continue # Skip this strategy attempt if LLM failed
            total_cost += result.get("cost", 0.0)
            args_text = result["text"]
            try:
                args_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", args_text.strip())
                llm_args_list = json.loads(args_text_cleaned)
                if isinstance(llm_args_list, list):
                    for item in llm_args_list:
                        if isinstance(item, dict):
                            # Pass targets_failure=True if generated from sim failures? Difficult to determine reliably here. Default to False.
                            add_unique_test_case(cast(TestStrategy, llm_strategy), item, f"LLM generated ({llm_strategy})", targets_failure=False)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed parse LLM ({llm_strategy}) args: {e}. Raw: {args_text}", exc_info=False)
        except Exception as e:
            logger.error(f"Error during LLM ({llm_strategy}) test gen for {tool_name}: {e}", exc_info=True)


    # --- 4. Ensure Minimum Tests & Apply Cap ---
    if not test_cases:
        logger.warning(f"No test cases generated for {tool_name}, creating a default positive case.")
        # Create a simple default positive case if nothing else worked
        default_case = {p: _create_value_for_schema(param_schemas[p], "positive") for p in required_params} if required_params else {}
        add_unique_test_case("positive_required_only", default_case, "Default positive case (fallback)")

    # Apply the final cap based on the original num_tests request
    final_cases = test_cases[:num_tests]
    logger.debug(f"Final generated {len(final_cases)} test cases for tool '{tool_name}' (requested: {num_tests}).")

    return final_cases, total_cost

async def _execute_tool_tests(
    mcp_instance: Any,
    tool_name: str,
    test_cases: List[TestCase] # Use V6/Apex++ test case type
) -> Tuple[List[TestExecutionResult], float]:
    """
    Executes a list of test cases against a specified tool concurrently,
    capturing detailed results, errors, and estimated costs.

    Args:
        mcp_instance: The MCP server instance (e.g., FastMCP) with a `call_tool` method.
        tool_name: The name of the tool to execute.
        test_cases: A list of TestCase objects containing arguments and metadata.

    Returns:
        A tuple containing:
        - A list of TestExecutionResult objects detailing the outcome of each test.
        - The total estimated cost (float) accumulated from successful tool calls
          that reported a cost.
    """
    # Use the correct constant name defined previously
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TESTS)

    async def run_single_test(test_case: TestCase) -> Tuple[TestExecutionResult, float]:
        """Executes a single test case and returns its result and cost."""
        async with semaphore:
            start_time = time.time()
            success = False
            result_preview = None
            error_message = None
            error_type = None
            error_code = None
            error_details = None
            test_run_cost = 0.0
            call_result: Optional[Dict[str, Any]] = None # Store raw result for cost extraction

            # Determine if execution should be skipped due to pre-validation failure
            # Only skip if validation failed AND it wasn't specifically a negative test
            # designed to potentially trigger validation errors.
            is_negative_strategy = test_case.strategy_used.startswith("negative_")
            skip_execution = test_case.schema_validation_error and not is_negative_strategy

            if skip_execution:
                error_message = f"Skipped Execution: Schema pre-validation failed ({test_case.schema_validation_error})"
                error_code = "SCHEMA_VALIDATION_FAILED_PRE_CALL"
                error_type = "ToolInputError" # Treat as input error
                logger.debug(f"Skipping execution for test case (Strategy: {test_case.strategy_used}) due to pre-validation error: {test_case.arguments}")
            else:
                # --- Execute the tool ---
                try:
                    # Make the actual tool call via the MCP instance
                    call_result = await mcp_instance.call_tool(
                        tool_name=tool_name,
                        arguments=test_case.arguments
                    )

                    # --- Analyze Result ---
                    if isinstance(call_result, dict):
                        # Check for standard MCP ToolResult error format first
                        is_mcp_tool_error = call_result.get("isError", False)
                        # Then check for JSON-RPC error format
                        is_jsonrpc_error = "error" in call_result and "id" in call_result

                        if is_mcp_tool_error or is_jsonrpc_error:
                             # Handle error structure
                             error_data = call_result.get("error", {})
                             if not isinstance(error_data, dict): # Handle cases where error might be just a string
                                 error_message = str(error_data)
                                 error_code = "TOOL_REPORTED_ERROR_STRING"
                                 error_type = "ToolError"
                             else:
                                 error_message = error_data.get("message", "Tool reported an unknown error.")
                                 error_code = error_data.get("error_code", "TOOL_REPORTED_ERROR") # Get specific code
                                 error_type = error_data.get("error_type", "ToolError") # Get specific type
                                 error_details = error_data.get("details") # Get details dict
                        else: # Success case
                            success = True
                            # Create a preview string, excluding verbose fields
                            preview_items = []
                            for k, v in call_result.items():
                                if k in ["content", "raw_response", "_meta", "success", "isError", "cost", "tokens"]: 
                                    continue
                                try:
                                     v_repr = repr(v)
                                except Exception:
                                     v_repr = "<unrepresentable>"
                                preview_items.append(f"{k}: {v_repr[:40]}{'...' if len(v_repr)>40 else ''}")
                            result_preview = "{ " + ", ".join(preview_items) + " }"
                            # Extract cost if available
                            if call_result.get("cost") is not None:
                                 try: 
                                     test_run_cost = float(call_result["cost"])
                                 except (ValueError, TypeError): 
                                     pass

                    else: # Non-dict success result (less common for complex tools)
                        success = True
                        try:
                            result_preview = repr(call_result)[:80] + ('...' if len(repr(call_result)) > 80 else '')
                        except Exception:
                             result_preview = "<unrepresentable result>"

                # --- Handle Exceptions During Execution ---
                except (ToolInputError, ProviderError, ToolError) as known_exec_err:
                    # These errors are often raised by the tool's own logic or wrappers
                    error_message = f"{type(known_exec_err).__name__}: {str(known_exec_err)}"
                    error_type = type(known_exec_err).__name__
                    error_code = getattr(known_exec_err, 'error_code', "TOOL_EXECUTION_FAILED")
                    error_details = getattr(known_exec_err, 'details', None)
                    logger.warning(f"Tool execution caught known error for {tool_name}: {error_message}")
                except Exception as exec_err:
                    # Catch unexpected errors during the call_tool process
                    error_message = f"Unexpected Execution Error: {type(exec_err).__name__}: {str(exec_err)}"
                    error_type = type(exec_err).__name__
                    error_code = "UNEXPECTED_EXECUTION_ERROR"
                    error_details = {"traceback": traceback.format_exc()}
                    logger.error(f"Tool execution caught unexpected error for {tool_name}: {error_message}", exc_info=True)

            end_time = time.time()

            # --- Create Result Object ---
            return TestExecutionResult(
                test_case=test_case,
                start_time=start_time,
                end_time=end_time,
                success=success,
                result_preview=result_preview,
                error_message=error_message,
                error_type=error_type,
                error_code=error_code,
                error_details=error_details
                # error_category_guess could be added here or later by analysis
            ), test_run_cost # Return cost associated with this specific run

    # --- Gather results and costs ---
    if not test_cases:
        logger.warning(f"No test cases provided to execute for tool '{tool_name}'.")
        return [], 0.0

    test_tasks = [run_single_test(tc) for tc in test_cases]
    # Use return_exceptions=True to ensure all tasks complete even if some fail
    results_with_costs = await asyncio.gather(*test_tasks, return_exceptions=True)

    # Process results, handling potential exceptions from gather itself
    final_results: List[TestExecutionResult] = []
    accumulated_cost = 0.0
    for i, res_or_exc in enumerate(results_with_costs):
        test_case_used = test_cases[i] # Get corresponding input test case
        if isinstance(res_or_exc, Exception):
            # An error occurred in run_single_test *outside* its try/except, or gather failed
            logger.error(f"Gather caught exception for test case {i} (Strategy: {test_case_used.strategy_used}): {res_or_exc}", exc_info=res_or_exc)
            # Create a TestExecutionResult indicating this failure
            fail_result = TestExecutionResult(
                test_case=test_case_used,
                start_time=time.time(), end_time=time.time(), # Times are inaccurate here
                success=False,
                error_message=f"Gather Exception: {type(res_or_exc).__name__}: {str(res_or_exc)}",
                error_type=type(res_or_exc).__name__,
                error_code="GATHER_EXECUTION_ERROR",
                error_details={"traceback": traceback.format_exception(res_or_exc)} # Use specific formatter
            )
            final_results.append(fail_result)
        elif isinstance(res_or_exc, tuple) and len(res_or_exc) == 2:
            # Successfully got (TestExecutionResult, cost) tuple
            exec_result, run_cost = res_or_exc
            final_results.append(exec_result)
            accumulated_cost += run_cost
        else:
            # Unexpected result format from gather/run_single_test
             logger.error(f"Unexpected result type from gather for test case {i}: {type(res_or_exc)}")
             fail_result = TestExecutionResult(
                 test_case=test_case_used,
                 start_time=time.time(), end_time=time.time(),
                 success=False,
                 error_message=f"Internal Error: Unexpected result type from task execution ({type(res_or_exc).__name__})",
                 error_type="InternalError", error_code="UNEXPECTED_GATHER_RESULT"
             )
             final_results.append(fail_result)


    logger.debug(f"Executed {len(final_results)} tests for tool '{tool_name}'. Accumulated cost: ${accumulated_cost:.6f}")
    return final_results, accumulated_cost

async def _analyze_and_propose(
    tool_name: str,
    iteration: int,
    current_docs: ProposedChanges, # Documentation used for this iteration's tests
    current_schema: JSONSchemaObject, # Schema used for this iteration's tests
    test_results: List[TestExecutionResult], # Results from this iteration
    refinement_model_configs: List[Dict[str, Any]], # List of LLM configs for analysis
    original_schema: JSONSchemaObject, # Original schema for context (optional use)
    validation_level: str # Validation level used ('none', 'basic', 'full')
) -> Tuple[Optional['RefinementAnalysis'], Optional['ProposedChanges'], float]:
    """
    Analyzes test failures, proposes documentation/schema improvements using an
    ensemble of LLMs, generates relevant examples, and calculates costs.

    Args:
        tool_name: Name of the tool being analyzed.
        iteration: Current refinement iteration number.
        current_docs: The documentation (ProposedChanges) used for the tests in this iteration.
        current_schema: The schema used for validation in this iteration.
        test_results: List of TestExecutionResult objects from this iteration.
        refinement_model_configs: List of LLM configurations for the analysis ensemble.
        original_schema: The initial schema of the tool (for reference).
        validation_level: The schema validation level used during testing.

    Returns:
        A tuple containing:
        - Optional[RefinementAnalysis]: The analysis results (or None if no failures).
        - Optional[ProposedChanges]: The proposed documentation/schema changes (or None if no changes suggested).
        - float: The total estimated cost incurred during analysis and example generation.
    """
    total_analysis_cost = 0.0
    failures = [r for r in test_results if not r.success]
    success_cases = [r for r in test_results if r.success]

    # --- 1. Generate Examples (Based on current iteration's results) ---
    generated_examples: List[GeneratedExample] = []
    # Always try to generate examples, even if no failures, to enrich docs
    logger.debug(f"Iter {iteration}, Tool '{tool_name}': Generating examples based on {len(success_cases)} successes and {len(failures)} failures.")
    try:
        # Use the primary model config for example generation
        examples, example_cost = await _generate_examples(
            tool_name=tool_name,
            description=current_docs.description,
            schema=current_schema, # Use schema corresponding to current docs
            success_cases=success_cases,
            failure_cases=failures,
            refinement_model_config=refinement_model_configs[0]
        )
        generated_examples = examples
        total_analysis_cost += example_cost
        logger.debug(f"Iter {iteration}, Tool '{tool_name}': Generated {len(generated_examples)} examples (Cost: ${example_cost:.6f}).")
    except Exception as e:
        logger.error(f"Iter {iteration}, Tool '{tool_name}': Failed to generate examples: {e}", exc_info=True)
        # Continue without generated examples if generation fails

    # --- 2. Perform Analysis (Only if Failures Occurred) ---
    if not failures:
        logger.info(f"Iter {iteration}, Tool '{tool_name}': No failures detected. Skipping analysis, proposing current docs with new examples.")
        analysis = RefinementAnalysis(
            overall_diagnosis="No errors detected in test runs.",
            improvement_confidence=1.0,
            hypothetical_error_resolution="N/A - No errors to resolve."
        )
        # Create proposal with current description and newly generated examples
        proposed_changes = ProposedChanges(
            description=current_docs.description,
            schema_patches=[], # No schema changes proposed
            examples=generated_examples # Include any examples generated
        )
        return analysis, proposed_changes, total_analysis_cost

    # --- Failures occurred, proceed with analysis ---
    logger.info(f"Iter {iteration}, Tool '{tool_name}': Analyzing {len(failures)} failures...")

    # --- Format Error Summary for LLM ---
    error_summary_text = ""
    error_patterns: Dict[ErrorPatternKey, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "examples": []})
    for _i, failure in enumerate(failures):
        # Create a more robust key: Type:Code:ParamTarget (if available)
        err_key_parts = [failure.error_type or "UnknownType", failure.error_code or "UnknownCode"]
        targeted_param = None
        if failure.error_details and "param_name" in failure.error_details:
            targeted_param = failure.error_details["param_name"]
        elif failure.test_case.strategy_used.startswith(("negative_", "edge_")):
            if "parameter '" in failure.error_message: # Try to extract from message
                match = re.search(r"parameter '([^']+)'", failure.error_message)
                if match: 
                    targeted_param = match.group(1)
            elif len(failure.test_case.arguments) == 1: # Assume single arg was target
                targeted_param = list(failure.test_case.arguments.keys())[0]
        if targeted_param: 
            err_key_parts.append(f"param:{targeted_param}")
        err_key = ":".join(err_key_parts)

        pattern_info = error_patterns[err_key]
        pattern_info["count"] += 1
        if len(pattern_info["examples"]) < 3: # Store max 3 examples per pattern
             pattern_info["examples"].append({
                 "strategy": failure.test_case.strategy_used,
                 "args": failure.test_case.arguments,
                 "error": failure.error_message,
                 "validation_error": failure.test_case.schema_validation_error,
                 "details": failure.error_details
             })

    error_summary_text += "Observed Failure Patterns (Grouped by ErrorType:ErrorCode:ParamTarget):\n"
    for pattern, info in error_patterns.items():
        error_summary_text += f"- Pattern '{pattern}' ({info['count']} occurrences):\n"
        for ex in info["examples"]:
            val_err = f" (SchemaPreFail: {ex['validation_error']})" if ex['validation_error'] else ""
            details_preview = f" Details: {json.dumps(ex['details'], default=str)[:60]}..." if ex['details'] else ""
            error_summary_text += f"  - Strategy: {ex['strategy']}{val_err}\n"
            error_summary_text += f"    Args: {json.dumps(ex['args'], default=str)[:80]}...\n    Error: {ex['error'][:100]}...{details_preview}\n"

    # --- Build Analysis Prompt ---
    # Use the schema and description that were actually used for testing this iteration
    schema_str = json.dumps(current_schema, indent=2)
    # Include current examples in the context for the analysis LLM
    current_examples_str = json.dumps([e.model_dump() for e in current_docs.examples], indent=2)

    prompt = f"""# Task: Refine MCP Tool Documentation (Apex++ Analysis)

You are an expert technical writer and API designer diagnosing LLM agent failures when using an MCP tool. Your goal is to identify *root causes* in the documentation (description, schema, examples) and propose specific, structured improvements (JSON Patches for schema, revised description, targeted examples).

## Tool Documentation Used in Failed Tests (Iteration {iteration})

**Tool Name:** `{tool_name}`

**Description:**
```
{current_docs.description or '(No description provided)'}
```

**Input Schema Used (JSON Schema):**
```json
{schema_str}
```

**Examples Provided to Agent:**
```json
{current_examples_str or '(None provided)'}
```

## Observed Failures ({len(failures)} failures / {len(test_results)} tests)

{error_summary_text}

## Analysis & Suggestion Task (Respond ONLY with JSON)

**1. Diagnose Root Causes:**
   - Analyze failure patterns. Link each pattern to *specific flaws* in the *documentation used* (description, schema properties like type/description/format/enum, or examples). Explain the reasoning clearly.
   - Identify specific schema weaknesses and description weaknesses. Categorize the primary flaws using `ErrorCategory` enum values.

**2. Propose Specific Improvements:**
   - **Overall Diagnosis:** Summarize key documentation issues.
   - **Parameter Schema Patches (JSON Patch RFC 6902):** For parameters needing improvement, provide JSON Patch operations targeting their definition within `inputSchema/properties/param_name`. Focus on `add` or `replace` for `description`, `type`, `format`, `enum`, `example`, `minimum`, `maximum`, etc. Include reasoning and estimate the impact ('high', 'medium', 'low') of each patch set.
   - **General Suggestions:** List other documentation improvements (e.g., 'Add section on common errors').
   - **Generate Usage Examples:** Provide 2-3 *new*, clear examples of *correct* usage designed to *prevent* the observed failures or illustrate complex cases. Include a `comment` and link to the `addresses_failure_pattern` (ErrorPatternKey) if applicable.
   - **Propose Revised Description:** Write a complete, improved main description incorporating necessary clarifications.
   - **Confidence & Resolution:** Estimate your confidence (0.0-1.0) that applying *all* changes will resolve *observed errors*. Analyze if the proposed changes directly address the root causes of major failure patterns.

## Output Format (Strict JSON Object - Adhere precisely):

```json
{{
  "refinement_analysis": {{
    "overall_diagnosis": "(String) Root cause analysis.",
    "error_pattern_analysis": {{ "(String) ErrorPatternKey": "(String) Analysis linking pattern to docs.", ... }},
    "identified_flaw_categories": ["(String: ErrorCategory enum value)", ... ],
    "parameter_schema_patches": [
      {{
        "parameter_name": "(String)",
        "patch_ops": [ {{ "op": "replace", "path": "/properties/param_name/description", "value": "New desc." }}, ... ],
        "reasoning": "(String)",
        "estimated_impact": "(String: 'high'|'medium'|'low')"
      }}, ...
    ],
    "general_suggestions": [ "(String)", ... ],
    "improvement_confidence": "(Float) 0.0-1.0",
    "hypothetical_error_resolution": "(String) Analysis of proposed effectiveness."
  }},
  "proposed_documentation": {{
    "description": "(String) Complete, revised main description.",
    "examples": [ {{ "args": {{...}}, "comment": "...", "addresses_failure_pattern": "(Optional[String] ErrorPatternKey)" }}, ... ]
  }}
}}
```

JSON Output:"""

    # --- Run Analysis with Ensemble ---

    async def run_single_analysis(config: Dict[str, Any]) -> Tuple[Optional[RefinementAnalysis], Optional[ProposedChanges], float]:
        """Runs analysis using one LLM config, returning structured results and cost."""
        analysis_model: Optional[RefinementAnalysis] = None
        prop_changes: Optional[ProposedChanges] = None
        cost = 0.0
        model_identifier = f"{config.get('provider', 'unknown')}/{config.get('model', 'unknown')}"
        try:
            logger.debug(f"Sending analysis prompt (Iter {iteration}, Tool '{tool_name}') to {model_identifier}")
            result = await generate_completion(
                prompt=prompt, **config, temperature=0.1, max_tokens=4000, # Generous token limit
                additional_params={"response_format": {"type": "json_object"}} if config.get("provider") == Provider.OPENAI.value else None
            )
            cost = result.get("cost", 0.0)
            if not result.get("success"): 
                raise ToolError(f"LLM analysis call failed: {result.get('error')}")
            analysis_text = result["text"]
            logger.debug(f"Raw analysis response from {model_identifier}: {analysis_text[:500]}...")
            try:
                analysis_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", analysis_text.strip())
                analysis_data = json.loads(analysis_text_cleaned)
                # Validate and structure using Pydantic models
                analysis_model = RefinementAnalysis(**analysis_data.get("refinement_analysis", {}))
                prop_changes = ProposedChanges(**analysis_data.get("proposed_documentation", {}))
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                logger.error(f"Failed parse/validate analysis JSON from {model_identifier}: {e}. Raw: {analysis_text}", exc_info=True)
                diag = f"Parse Error from {model_identifier}: {e}. Raw: {analysis_text[:200]}..."
                analysis_model = RefinementAnalysis(overall_diagnosis=diag, improvement_confidence=0.0, hypothetical_error_resolution="Parse failed")
                prop_changes = ProposedChanges(description=current_docs.description, examples=generated_examples) # Fallback
        except Exception as e:
            logger.error(f"Error during analysis execution with {model_identifier}: {e}", exc_info=True)
            diag = f"Failed execution for {model_identifier}: {e}"
            analysis_model = RefinementAnalysis(overall_diagnosis=diag, improvement_confidence=0.0, hypothetical_error_resolution="Execution failed")
            prop_changes = ProposedChanges(description=current_docs.description, examples=generated_examples)
        return analysis_model, prop_changes, cost

    analysis_tasks = [run_single_analysis(config) for config in refinement_model_configs]
    results_from_ensemble = await asyncio.gather(*analysis_tasks)

    # --- Synthesize/Select Final Analysis & Proposal ---
    # Strategy: Use the result with the highest confidence score. Ties broken by first model.
    best_analysis: Optional[RefinementAnalysis] = None
    best_proposed_changes: Optional[ProposedChanges] = ProposedChanges(
        description=current_docs.description, examples=generated_examples, schema_patches=[] # Start with current + new examples
    )
    highest_confidence = -1.0

    for res_analysis, res_prop_changes, res_cost in results_from_ensemble:
        total_analysis_cost += res_cost # Accumulate cost from all attempts
        if res_analysis and res_analysis.improvement_confidence >= highest_confidence:
            # Update if confidence is higher, or if it's the first valid result
            if res_analysis.improvement_confidence > highest_confidence or best_analysis is None:
                highest_confidence = res_analysis.improvement_confidence
                best_analysis = res_analysis
                # Use the proposed changes corresponding to the best analysis
                # Ensure examples from generation phase are carried over if analysis didn't propose any
                if res_prop_changes:
                     if not res_prop_changes.examples:
                          res_prop_changes.examples = generated_examples
                     best_proposed_changes = res_prop_changes
                else:
                    # If proposal object itself is None, create default with generated examples
                    best_proposed_changes = ProposedChanges(
                        description=current_docs.description, # Fallback desc
                        examples=generated_examples,
                        schema_patches=[]
                    )

    # Ensure we always have a proposal object, even if all analysis failed
    if best_proposed_changes is None:
        best_proposed_changes = ProposedChanges(description=current_docs.description, examples=generated_examples, schema_patches=[])
    # Ensure examples are populated if somehow missed
    if not best_proposed_changes.examples:
         best_proposed_changes.examples = generated_examples

    return best_analysis, best_proposed_changes, total_analysis_cost

async def _generate_examples(
    tool_name: str,
    description: str,
    schema: JSONSchemaObject,
    success_cases: List[TestExecutionResult],
    failure_cases: List[TestExecutionResult],
    refinement_model_config: Dict[str, Any]
) -> Tuple[List[GeneratedExample], float]:
    """Generate examples addressing failures and showcasing success."""
    # (Implementation from V6 - Reuse)
    # ... Assume V6 implementation is present ...
    # NOTE: For brevity, omitting full copy-paste. Assume V6 logic is used.
    cost = 0.0
    if not success_cases and not failure_cases: 
        return [], cost
    schema_str = json.dumps(schema, indent=2)
    success_str = "\n".join([f"- Args: {json.dumps(r.test_case.arguments, default=str)}" for r in success_cases[:2]])
    failure_str = "Examples of Failed Calls (Args -> Error):\n" + "\n".join([f"- {json.dumps(r.test_case.arguments, default=str)} -> {r.error_message}" for r in failure_cases[:2]]) if failure_cases else "None"

    prompt = f"""Given the MCP tool documentation, successful usage examples, and observed failures:

**Tool Name:** `{tool_name}`
**Description:** {description or '(None)'}
**Schema:**
```json
{schema_str}
```
**Successful Args:**
```json
{success_str or '(None observed)'}
```
**Failed Args & Errors:**
```
{failure_str or '(None observed)'}
```

Generate 2-4 diverse, clear usage examples as a JSON list. Each example MUST be a dictionary `{{ "args": {{...}}, "comment": "...", "addresses_failure_pattern": "Optional[ErrorKey]" }}`.
- Prioritize examples that demonstrate how to *avoid* the observed failures.
- Include examples showcasing common or important successful use cases.
- Ensure the `args` are valid according to the schema.
- The `comment` should explain the purpose or key aspect demonstrated by the example.
- If an example addresses a specific failure pattern key (from the failure summary, e.g., "ToolInputError:INVALID_PARAMETER:param:some_param"), include that key in `addresses_failure_pattern`.

Output ONLY the valid JSON list."""

    try:
        result = await generate_completion(
            prompt=prompt, **refinement_model_config, temperature=0.4, max_tokens=1500, # Increased token limit
            additional_params={"response_format": {"type": "json_object"}} if refinement_model_config.get("provider") == Provider.OPENAI.value else None
        )
        if not result.get("success"):
            raise ToolError(f"LLM example gen fail: {result.get('error')}")
        cost += result.get("cost", 0.0)
        examples_text = result["text"]
        try:
            examples_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", examples_text.strip())
            examples_list_raw = json.loads(examples_text_cleaned)
            # Validate structure using Pydantic
            validated_examples = [GeneratedExample(**ex) for ex in examples_list_raw if isinstance(ex, dict)]
            return validated_examples, cost
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.error(f"Failed parse/validate examples JSON: {e}. Raw: {examples_text}", exc_info=True)
            return [], cost
    except Exception as e:
        logger.error(f"Error generating examples: {e}", exc_info=True)
        return [], cost



async def _winnow_documentation(
    tool_name: str,
    current_docs: ProposedChanges,
    current_schema: JSONSchemaObject,
    refinement_model_config: Dict[str, Any]
) -> Tuple[ProposedChanges, float]:
    """Simplifies documentation after refinement stabilizes."""
    # (Implementation from V6 - Reuse)
    # ... Assume V6 implementation ...
    # NOTE: For brevity, omitting full copy-paste. Assume V6 logic is used.
    cost = 0.0
    logger.info(f"Winnowing documentation for '{tool_name}'...")
    schema_str = json.dumps(current_schema, indent=2)
    examples_str = json.dumps([ex.model_dump() for ex in current_docs.examples], indent=2)

    prompt = f"""Winnow documentation for tool '{tool_name}'. Make description concise, keep 1-2 best examples demonstrating core use and avoiding past pitfalls. Preserve schema patches.
Current Desc: {current_docs.description}
Schema: {schema_str}
Current Examples: {examples_str}
Output JSON: {{"description": "concise_desc", "examples": [...]}}""" # Examples field only
    try:
        result = await generate_completion(prompt=prompt, **refinement_model_config, temperature=0.2, max_tokens=1800, additional_params={"response_format": {"type": "json_object"}} if refinement_model_config.get("provider") == Provider.OPENAI.value else None)
        if not result.get("success"): 
            raise ToolError(f"LLM winnow fail: {result.get('error')}")
        cost += result.get("cost", 0.0)
        winnow_text = result["text"]
        try:
            winnow_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", winnow_text.strip())
            winnow_data = json.loads(winnow_text_cleaned)
            winnowed_proposal = ProposedChanges(
                description=winnow_data.get("description", current_docs.description),
                schema_patches=current_docs.schema_patches, # Keep existing patches
                examples=[GeneratedExample(**ex) for ex in winnow_data.get("examples", []) if isinstance(ex, dict)]
            )
            return winnowed_proposal, cost
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.error(f"Failed parse/validate winnowing JSON: {e}. Raw: {winnow_text}", exc_info=True)
            return current_docs.model_copy(deep=True), cost
    except Exception as e:
        logger.error(f"Error winnowing {tool_name}: {e}", exc_info=True)
        return current_docs.model_copy(deep=True), cost


# --- Main Tool Function ---

@with_tool_metrics
@with_error_handling
async def refine_tool_documentation(
    tool_names: Optional[List[str]] = None,
    refine_all_available: bool = False,
    generation_config: Optional[JsonDict] = None,
    max_iterations: int = 1,
    refinement_model_config: Optional[JsonDict] = None,
    analysis_ensemble_configs: Optional[List[JsonDict]] = None,
    validation_level: Optional[Literal['none', 'basic', 'full']] = None, # Made optional, default below
    enable_winnowing: bool = True,
    progress_callback: ProgressCallback = None,
    ctx: McpContext = None
) -> JsonDict:
    """
    Apex++ Synergetic: Autonomously refines MCP tool documentation via agent simulation,
    schema-aware adaptive testing, validated JSON Patch schema evolution, ensemble analysis,
    failure-driven example generation, optional winnowing, and detailed reporting.

    Args:
        tool_names: Specific tool names (e.g., ["server:tool"]) to refine. Use EITHER this OR `refine_all_available`.
        refine_all_available: If True, refine all registered tools. Overrides `tool_names`. Default False.
        generation_config: (Optional) Dict controlling test case counts per strategy (`TestCaseGenerationStrategy`).
        max_iterations: (Optional) Max internal test-analyze-patch-suggest cycles per tool. Default 1. Max capped.
        refinement_model_config: (Optional) Primary LLM config for analysis/generation. Uses capable default if None.
        analysis_ensemble_configs: (Optional) List of additional LLM configs for ensemble analysis.
        validation_level: (Optional) How strictly to validate generated test args ('none', 'basic', 'full'). Defaults to 'full' if jsonschema available, else 'none'.
        enable_winnowing: (Optional) If True, run final pass to make documentation concise. Default True.
        progress_callback: (Optional) Async function to call with `RefinementProgressEvent` updates.
        ctx: MCP context object (required).

    Returns:
        Dictionary structured by `DocstringRefinementResult`.
    """
    start_time = time.time()
    refined_tools_list: List[RefinedToolResult] = []
    total_tests_attempted_overall = 0
    total_tests_failed_overall = 0
    total_schema_validation_failures = 0
    total_agent_simulation_failures = 0
    total_refinement_cost_overall = 0.0
    refinement_process_errors_overall: List[str] = []
    total_iterations_run_all_tools = 0

    # Determine effective validation level
    eff_validation_level = validation_level or ('full')

    async def _emit_progress(event_data: Dict[str, Any]):
        if progress_callback:
            try: 
                await progress_callback(RefinementProgressEvent(**event_data))
            except Exception as cb_err: 
                logger.warning(f"Progress callback failed: {cb_err}")

    # --- Validate Inputs and Context ---
    if not ctx or not hasattr(ctx, 'mcp') or ctx.mcp is None:
        raise ToolError("MCP context (ctx.mcp) is required.", error_code="CONTEXT_MISSING")
    mcp_instance = ctx.mcp

    all_tools_dict: Dict[str, McpToolDef] = {}
    if hasattr(mcp_instance, '_tools'): 
        all_tools_dict = mcp_instance._tools
    elif hasattr(ctx, 'request_context') and ctx.request_context and hasattr(ctx.request_context, 'lifespan_context'):
        all_tools_dict = ctx.request_context.lifespan_context.get('tools', {})

    if not all_tools_dict:
        logger.warning("No tools registered in server context.")
        return DocstringRefinementResult(refined_tools=[], refinement_model_configs=[], total_iterations_run=0, total_test_calls_attempted=0, total_test_calls_failed=0, total_schema_validation_failures=0, total_agent_simulation_failures=0, total_refinement_cost=0.0, total_processing_time=time.time()-start_time, success=True).model_dump()

    if not refine_all_available and (not tool_names or not isinstance(tool_names, list)):
        raise ToolInputError("Either 'tool_names' list must be provided or 'refine_all_available' must be True.")

    target_tool_names = list(all_tools_dict.keys()) if refine_all_available else tool_names
    missing = [name for name in target_tool_names if name not in all_tools_dict]
    if missing: 
        raise ToolInputError(f"Tools not found: {', '.join(missing)}. Available: {list(all_tools_dict.keys())}")

    if not target_tool_names:
         return DocstringRefinementResult(refined_tools=[], refinement_model_configs=[], total_iterations_run=0, total_test_calls_attempted=0, total_test_calls_failed=0, total_schema_validation_failures=0, total_agent_simulation_failures=0, total_refinement_cost=0.0, total_processing_time=time.time()-start_time, success=True).model_dump()

    # --- Prepare Configs ---
    try: 
        test_gen_strategy = TestCaseGenerationStrategy(**(generation_config or {}))
    except ValidationError as e: 
        raise ToolInputError(f"Invalid generation_config: {e}") from e
    num_tests_per_tool = min(sum(getattr(test_gen_strategy, f) for f in test_gen_strategy.model_fields), MAX_TEST_QUERIES_PER_TOOL)
    max_iterations = max(1, min(max_iterations, MAX_REFINEMENT_ITERATIONS))
    default_config = {"provider": DEFAULT_REFINEMENT_PROVIDER, "model": DEFAULT_REFINEMENT_MODEL}
    primary_refinement_config = refinement_model_config or default_config
    analysis_configs = [primary_refinement_config]
    if analysis_ensemble_configs: 
        analysis_configs.extend([c for c in analysis_ensemble_configs if isinstance(c, dict) and c != primary_refinement_config])

    logger.info(f"Starting Apex++ refinement for {len(target_tool_names)} tools...")

    # --- Function to process a single tool ---
    async def process_single_tool(tool_name: str) -> RefinedToolResult:
        nonlocal total_refinement_cost_overall, total_tests_attempted_overall, total_tests_failed_overall
        nonlocal total_schema_validation_failures, total_agent_simulation_failures, total_iterations_run_all_tools
        nonlocal refinement_process_errors_overall

        await _emit_progress({"tool_name": tool_name, "iteration": 0, "total_iterations": max_iterations, "stage": "starting_iteration", "message": "Starting refinement"})
        tool_def = all_tools_dict.get(tool_name)
        # Basic validation
        if not tool_def or not hasattr(tool_def, 'name'): 
            return RefinedToolResult(tool_name=tool_name, iterations=[], original_schema={}, final_proposed_changes=ProposedChanges(description="ERROR: Tool definition missing"), final_schema_after_patches={}, initial_success_rate=0, final_success_rate=0, improvement_factor=0, process_error="Tool definition missing")
        original_desc = getattr(tool_def, 'description', '') or ''
        original_schema = getattr(tool_def, 'inputSchema', getattr(tool_def, 'input_schema', None))
        if not isinstance(original_schema, dict): 
            return RefinedToolResult(tool_name=tool_name, iterations=[], original_schema={}, final_proposed_changes=ProposedChanges(description=original_desc), final_schema_after_patches={}, initial_success_rate=0, final_success_rate=0, improvement_factor=0, process_error="Invalid original schema")

        tool_iterations_results: List[RefinementIterationResult] = []
        current_proposed_changes = ProposedChanges(description=original_desc, examples=[]) # Start with original desc
        current_schema_for_iter = copy.deepcopy(original_schema)
        accumulated_patches: List[JsonDict] = []
        initial_success_rate = -1.0
        final_success_rate = -1.0
        tool_process_error: Optional[str] = None

        try:
            for iteration in range(1, max_iterations + 1):
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "total_iterations": max_iterations, "stage": "starting_iteration", "message": f"Starting Iteration {iteration}"})
                schema_before_patch = copy.deepcopy(current_schema_for_iter) # Schema used for this iter

                # 1. Agent Simulation
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "agent_simulation", "message": "Simulating agent usage..."})
                agent_sim_results, sim_cost = await _simulate_agent_usage(
                    tool_name, current_proposed_changes, current_schema_for_iter, primary_refinement_config
                )
                total_refinement_cost_overall += sim_cost
                current_agent_sim_failures = sum(1 for r in agent_sim_results if not r.formulation_success)
                total_agent_simulation_failures += current_agent_sim_failures
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "agent_simulation", "message": f"Simulation complete ({current_agent_sim_failures} failures)."})

                # 2. Generate Test Cases
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "test_generation", "message": "Generating test cases..."})
                previous_results_for_guidance = tool_iterations_results[-1].test_results if tool_iterations_results else None
                test_cases, test_gen_cost = await _generate_test_cases(
                    tool_name=tool_name, tool_schema=current_schema_for_iter,
                    tool_description=current_proposed_changes.description, num_tests=num_tests_per_tool,
                    refinement_model_config=primary_refinement_config,
                    validation_level=eff_validation_level,
                    previous_results=previous_results_for_guidance,
                    agent_sim_results=agent_sim_results
                )
                total_refinement_cost_overall += test_gen_cost
                iter_validation_failures = sum(1 for tc in test_cases if tc.schema_validation_error)
                total_schema_validation_failures += iter_validation_failures
                valid_test_cases_for_exec = [tc for tc in test_cases if eff_validation_level != 'full' or tc.schema_validation_error is None or tc.strategy_used.startswith("negative_")]
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "test_generation", "message": f"Generated {len(test_cases)} cases ({iter_validation_failures} validation fails). Executing {len(valid_test_cases_for_exec)}."})

                # 3. Execute Tests
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "test_execution_start", "message": f"Executing {len(valid_test_cases_for_exec)} tests..."})
                test_results, test_exec_cost = await _execute_tool_tests(
                    mcp_instance=mcp_instance, tool_name=tool_name, test_cases=valid_test_cases_for_exec
                )
                total_refinement_cost_overall += test_exec_cost
                total_tests_attempted_overall += len(test_results)
                current_failures = sum(1 for r in test_results if not r.success)
                total_tests_failed_overall += current_failures
                current_success_rate = (len(test_results) - current_failures) / len(test_results) if test_results else 1.0
                if iteration == 1: 
                    initial_success_rate = current_success_rate
                final_success_rate = current_success_rate
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "test_execution_end", "message": f"Execution complete ({current_failures} failures).", "details": {"failures": current_failures, "success_rate": current_success_rate}})

                # 4. Analyze & Propose
                iter_analysis: Optional[RefinementAnalysis] = None
                iter_proposal_changes: Optional[ProposedChanges] = current_proposed_changes.model_copy(deep=True)
                iter_schema_patches: List[ParameterSchemaPatch] = []
                analysis_cost = 0.0
                if current_failures > 0 or iteration == 1:
                    await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "analysis_start", "message": "Analyzing failures and proposing improvements..."})
                    iter_analysis, iter_proposal_changes, analysis_cost = await _analyze_and_propose(
                        tool_name=tool_name, iteration=iteration, current_docs=current_proposed_changes,
                        current_schema=current_schema_for_iter, test_results=test_results,
                        refinement_model_configs=analysis_configs, # Use ensemble
                        original_schema=original_schema, validation_level=eff_validation_level
                    )
                    total_refinement_cost_overall += analysis_cost
                    iter_schema_patches = iter_proposal_changes.schema_patches if iter_proposal_changes else []
                    await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "analysis_end", "message": f"Analysis complete (Confidence: {iter_analysis.improvement_confidence:.2f if iter_analysis else 'N/A'})."})
                else:
                    iter_analysis = RefinementAnalysis(overall_diagnosis="No errors this iteration.", improvement_confidence=1.0, hypothetical_error_resolution="N/A")
                    iter_proposal_changes = current_proposed_changes # Keep current

                # 5. Apply Schema Patches In-Memory
                schema_after_patch = schema_before_patch
                schema_apply_errors: List[str] = []
                applied_patches_this_iter: List[JsonDict] = []
                if iter_schema_patches:
                    await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "schema_patching", "message": f"Applying {len(iter_schema_patches)} schema patches..."})
                    schema_after_patch, schema_apply_errors, applied_patches_this_iter = _apply_schema_patches(
                        schema_before_patch, iter_schema_patches
                    )
                    if schema_apply_errors: 
                        await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "error", "message": f"Schema patch errors: {schema_apply_errors}"})
                    accumulated_patches.extend(applied_patches_this_iter)

                # 6. Calculate Diffs
                iter_diff_desc = _create_diff(current_proposed_changes.description, iter_proposal_changes.description, f"desc_iter_{iteration-1}", f"desc_iter_{iteration}")
                iter_diff_schema = _create_schema_diff(schema_before_patch, schema_after_patch)

                # 7. Store Iteration Result
                iter_result = RefinementIterationResult(
                    iteration=iteration,
                    documentation_used=current_proposed_changes.model_copy(deep=True),
                    schema_used=schema_before_patch,
                    agent_simulation_results=agent_sim_results,
                    test_cases_generated=test_cases,
                    test_results=test_results,
                    success_rate=current_success_rate,
                    validation_failure_rate=(iter_validation_failures / len(test_cases)) if test_cases else 0.0,
                    analysis=iter_analysis,
                    proposed_changes=iter_proposal_changes.model_copy(deep=True) if iter_proposal_changes else None,
                    applied_schema_patches=applied_patches_this_iter,
                    description_diff=iter_diff_desc,
                    schema_diff=iter_diff_schema
                )
                tool_iterations_results.append(iter_result)
                total_iterations_run_all_tools += 1

                # 8. Check Stopping Conditions
                confidence = iter_analysis.improvement_confidence if iter_analysis else 0.0
                stop_reason = ""
                if current_failures == 0 and iter_validation_failures == 0: 
                    stop_reason = "100% success & validation"
                elif confidence < 0.2: 
                    stop_reason = f"low analysis confidence ({confidence:.2f})"
                elif iteration > 1:
                     prev_success = tool_iterations_results[-2].success_rate
                     prev_valid = 1.0 - tool_iterations_results[-2].validation_failure_rate
                     curr_valid = 1.0 - iter_result.validation_failure_rate
                     if current_success_rate <= prev_success and curr_valid <= prev_valid:
                         stop_reason = "success/validation rates stagnated"

                if stop_reason:
                    logger.info(f"Iter {iteration}: Stopping refinement for '{tool_name}' - Reason: {stop_reason}.")
                    await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "iteration_complete", "message": f"Stopping: {stop_reason}"})
                    # Apply winnowing if enabled and stopping condition met
                    if enable_winnowing:
                        await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "winnowing", "message": "Attempting winnowing..."})
                        winnowed_proposal, winnow_cost = await _winnow_documentation(
                             tool_name, iter_proposal_changes or current_proposed_changes, # Use last proposal
                             schema_after_patch, # Use final schema state for winnowing context
                             primary_refinement_config
                        )
                        total_refinement_cost_overall += winnow_cost
                        iter_proposal_changes = winnowed_proposal # Replace proposal with winnowed version
                        await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "winnowing", "message": "Winnowing complete."})
                    break # Exit iteration loop

                # 9. Prepare for Next Iteration
                current_proposed_changes = iter_proposal_changes or current_proposed_changes
                current_schema_for_iter = schema_after_patch
                await _emit_progress({"tool_name": tool_name, "iteration": iteration, "stage": "iteration_complete", "message": "Preparing for next iteration"})

            # --- End Iteration Loop ---

        except Exception as tool_proc_error:
            msg = f"Failed refining tool '{tool_name}': {type(tool_proc_error).__name__}: {str(tool_proc_error)}"
            logger.error(msg, exc_info=True)
            tool_process_error = msg

        # --- Assemble Final Result for this Tool ---
        final_tool_result = None
        if tool_iterations_results:
            final_iter = tool_iterations_results[-1]
            # Final proposed changes are those from the last successful proposal step
            final_docs = final_iter.proposed_changes or final_iter.documentation_used
            # Final schema is the one used AFTER the last successful patch application
            final_schema = current_schema_for_iter # This holds the state after the last iter's patch
            improvement = ((final_success_rate - initial_success_rate) / (1.0001 - initial_success_rate)) if initial_success_rate >= 0 and final_success_rate >= 0 and initial_success_rate < 1 else 0.0

            final_tool_result = RefinedToolResult(
                 tool_name=tool_name, original_schema=original_schema, iterations=tool_iterations_results,
                 final_proposed_changes=final_docs, # Contains desc, examples, LAST proposed patches
                 final_proposed_schema_patches=accumulated_patches, # All patches successfully applied
                 final_schema_after_patches=final_schema,
                 initial_success_rate=initial_success_rate if initial_success_rate >= 0 else 0.0,
                 final_success_rate=final_success_rate if final_success_rate >= 0 else 0.0,
                 improvement_factor=improvement,
                 token_count_change=(count_tokens(original_desc) - count_tokens(final_docs.description)),
                 process_error=tool_process_error
             )
        elif tool_process_error: # Handle case where loop failed entirely
             final_tool_result = RefinedToolResult(
                 tool_name=tool_name, iterations=[], original_schema=original_schema,
                 final_proposed_changes=ProposedChanges(description=original_desc),
                 final_proposed_schema_patches=[], final_schema_after_patches=original_schema,
                 initial_success_rate=0.0, final_success_rate=0.0, improvement_factor=0.0,
                 process_error=tool_process_error
             )

        await _emit_progress({"tool_name": tool_name, "iteration": max_iterations, "stage": "tool_complete", "message": "Refinement process finished."})
        return final_tool_result

    # --- Parallel Tool Processing ---
    processing_tasks = [process_single_tool(name) for name in target_tool_names]
    results_from_gather = await asyncio.gather(*processing_tasks, return_exceptions=True)

    # --- Process Results & Accumulate Totals ---
    for res in results_from_gather:
         if isinstance(res, RefinedToolResult):
              refined_tools_list.append(res)
              if res.process_error: 
                  refinement_process_errors_overall.append(res.process_error)
              # Accumulate costs/stats from the result object if needed (already tracked globally though)
         elif isinstance(res, Exception):
              err_msg = f"Unexpected error during parallel processing: {type(res).__name__}: {str(res)}"
              logger.error(err_msg, exc_info=res)
              refinement_process_errors_overall.append(err_msg)

    # --- Final Overall Result ---
    processing_time = time.time() - start_time
    overall_success = not bool(refinement_process_errors_overall)
    final_result = DocstringRefinementResult(
        refined_tools=refined_tools_list,
        refinement_model_configs=analysis_configs, # Use cleaned list
        total_iterations_run=total_iterations_run_all_tools,
        total_test_calls_attempted=total_tests_attempted_overall,
        total_test_calls_failed=total_tests_failed_overall,
        total_schema_validation_failures=total_schema_validation_failures,
        total_agent_simulation_failures=total_agent_simulation_failures,
        total_refinement_cost=total_refinement_cost_overall,
        total_processing_time=processing_time,
        errors_during_refinement_process=refinement_process_errors_overall,
        success=overall_success
    )

    logger.success("Apex++ Docstring Refinement V6 completed.", time=processing_time)
    await _emit_progress({"tool_name": "ALL", "iteration": max_iterations, "total_iterations": max_iterations, "stage": "tool_complete", "message": "Overall refinement process finished.", "details": {"success": overall_success}})

    return final_result.model_dump()

    # --- Parallel Tool Processing ---
    # Wrap process_single_tool to handle potential None return if validation fails early
    async def safe_process_single_tool(name: str) -> Optional[RefinedToolResult]:
         try:
             return await process_single_tool(name)
         except Exception as e:
              # Log error specific to this tool's processing task
              logger.error(f"Critical error during processing task for tool '{name}': {e}", exc_info=True)
              # Record the error in the overall process errors
              nonlocal refinement_process_errors_overall
              refinement_process_errors_overall.append(f"Tool '{name}': {type(e).__name__}: {str(e)}")
              return None # Indicate failure for this tool

    processing_tasks = [safe_process_single_tool(name) for name in target_tool_names]
    results_from_gather = await asyncio.gather(*processing_tasks) # Exceptions inside task are handled

    # Filter out None results and add to the final list
    refined_tools_list = [res for res in results_from_gather if res is not None]
    # Update overall error list with errors caught during gather/task execution
    # (already handled by appending within safe_process_single_tool)

    # --- Final Overall Result ---
    processing_time = time.time() - start_time
    overall_success = not bool(refinement_process_errors_overall)
    final_result = DocstringRefinementResult(
        refined_tools=refined_tools_list,
        refinement_model_configs=analysis_configs, # Use final list
        total_iterations_run=total_iterations_run_all_tools,
        total_test_calls_attempted=total_tests_attempted_overall,
        total_test_calls_failed=total_tests_failed_overall,
        total_schema_validation_failures=total_schema_validation_failures,
        total_agent_simulation_failures=total_agent_simulation_failures,
        total_refinement_cost=total_refinement_cost_overall,
        total_processing_time=processing_time,
        errors_during_refinement_process=refinement_process_errors_overall,
        success=overall_success
    )

    logger.success("Docstring Refinement completed.", time=processing_time)
    await _emit_progress({
        "tool_name": "ALL", "iteration": max_iterations, "total_iterations": max_iterations,
        "stage": "tool_complete", "message": "Overall refinement process finished.",
        "details": {"success": overall_success, "total_tools": len(target_tool_names)}
    })

    return final_result.model_dump()

async def _generate_examples(
    tool_name: str,
    description: str,
    schema: JSONSchemaObject,
    success_cases: List['TestExecutionResult'], # Use forward reference
    failure_cases: List['TestExecutionResult'], # Use forward reference
    refinement_model_config: Dict[str, Any],
    validation_level: Literal['full', 'basic'] = 'full'
) -> Tuple[List['GeneratedExample'], float]: # Use forward reference
    """
    Generates illustrative usage examples for a tool using an LLM.

    It considers the tool's documentation, successful execution arguments, and
    arguments that led to failures to create examples that demonstrate correct usage
    and potentially address common pitfalls.

    Args:
        tool_name: Name of the tool.
        description: Current description of the tool.
        schema: Current JSON schema of the tool's input.
        success_cases: List of successful test execution results.
        failure_cases: List of failed test execution results.
        refinement_model_config: LLM configuration for generating the examples.

    Returns:
        A tuple containing:
        - A list of GeneratedExample objects.
        - The estimated cost (float) of the LLM call for example generation.
    """
    cost = 0.0
    # Don't bother generating examples if there's nothing to base them on
    # or if the schema itself is minimal (no properties likely means no args needed)
    if not schema.get("properties") and not success_cases and not failure_cases:
        logger.debug(f"Skipping example generation for '{tool_name}': No properties, successes, or failures provided.")
        return [], cost
    if not description and not schema.get("properties"):
         logger.debug(f"Skipping example generation for '{tool_name}': No description or properties.")
         return [], cost


    # --- Format Inputs for Prompt ---
    schema_str = json.dumps(schema, indent=2)

    # Limit the number of examples shown in the prompt to manage context size
    max_prompt_examples = 3
    success_examples_str = "\n".join([f"- Args: {json.dumps(r.test_case.arguments, default=str)}"
                                      for r in success_cases[:max_prompt_examples]])
    failure_examples_str = ""
    if failure_cases:
        failure_examples_str += "Examples of Failed Calls (Args -> Error Type: Error Code -> Message):\n"
        for r in failure_cases[:max_prompt_examples]:
             error_info = f"{r.error_type or 'UnknownType'}:{r.error_code or 'UnknownCode'} -> {r.error_message or 'No message'}"
             failure_examples_str += f"- {json.dumps(r.test_case.arguments, default=str)} -> {error_info[:150]}...\n" # Limit error message length

    # --- Construct Prompt ---
    prompt = f"""# Task: Generate Usage Examples for MCP Tool

Given the documentation for the MCP tool '{tool_name}', along with examples of successful and failed attempts to use it by an LLM agent, generate 2-4 diverse and clear usage examples.

## Tool Documentation

**Description:**
{description or '(No description provided)'}

**Input Schema:**
```json
{schema_str}
```

## Observed Usage

**Successful Args Examples:**
```json
{success_examples_str or '(None provided or none succeeded)'}
```

**Failed Args & Errors Examples:**
```
{failure_examples_str or '(None observed)'}
```

## Instructions

1.  Create 2-4 concise, valid usage examples based on the schema and description.
2.  **Prioritize examples that demonstrate how to *avoid* the observed failures.** If failures involved specific parameters or constraints, show the *correct* way to use them.
3.  Include examples showcasing common or important successful use cases reflected in the `Successful Args Examples`.
4.  Ensure the `args` in your examples are valid according to the provided JSON schema.
5.  For each example, provide a brief `comment` explaining its purpose or what key aspect it demonstrates (e.g., "Minimal call with only required fields", "Using the 'advanced_filter' optional parameter", "Correctly formatting the date string to avoid errors").
6.  If an example specifically addresses a failure pattern (like an error code or type from the failures list), optionally include the relevant `addresses_failure_pattern` key (e.g., "ToolInputError:INVALID_PARAMETER:param:date").

## Output Format

Respond ONLY with a single, valid JSON list. Each element in the list must be an object with the following keys:
- `args`: (Object) The valid arguments for the tool call.
- `comment`: (String) A brief explanation of the example.
- `addresses_failure_pattern`: (Optional[String]) The key identifying the failure pattern this example helps avoid.

Example Output:
```json
[
  {{
    "args": {{ "required_param": "value1" }},
    "comment": "Minimal usage with only the required parameter."
  }},
  {{
    "args": {{ "required_param": "value2", "optional_param": true, "filter": {{"type": "user", "min_score": 0.8}} }},
    "comment": "Using optional parameters and a nested filter object.",
    "addresses_failure_pattern": "ToolInputError:INVALID_PARAMETER:param:filter"
  }},
  {{
    "args": {{ "required_param": "value3", "format": "YYYY-MM-DD" }},
    "comment": "Demonstrating the correct 'format' enum value."
  }}
]
```

JSON Output:
"""

    # --- Call LLM ---
    try:
        logger.debug(f"Generating examples for tool '{tool_name}' using model '{refinement_model_config.get('model')}'...")
        result = await generate_completion(
            prompt=prompt,
            **refinement_model_config,
            temperature=0.4, # Moderate temperature for slightly diverse examples
            max_tokens=1500, # Allow decent space for examples
            # Use JSON mode if supported by the provider
            additional_params={"response_format": {"type": "json_object"}} if refinement_model_config.get("provider") == Provider.OPENAI.value else None
        )
        if not result.get("success"):
            raise ToolError(f"LLM call failed during example generation: {result.get('error')}")

        cost += result.get("cost", 0.0)
        examples_text = result["text"]
        logger.debug(f"Raw examples response from LLM: {examples_text[:500]}...")

        # --- Parse and Validate Response ---
        try:
            # Clean potential markdown fences first
            examples_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", examples_text.strip())
            # Attempt to parse the cleaned text
            examples_list_raw = json.loads(examples_text_cleaned)

            # Validate structure using Pydantic
            validated_examples: List[GeneratedExample] = []
            if isinstance(examples_list_raw, list):
                for i, ex_data in enumerate(examples_list_raw):
                    if isinstance(ex_data, dict):
                        try:
                            # Validate each example dictionary
                            validated_ex = GeneratedExample(**ex_data)
                            # Optional: Further validate example 'args' against the tool's schema
                            if validation_level != 'none':
                                validation_error = _validate_args_against_schema(validated_ex.args, schema)
                                if validation_error:
                                     logger.warning(f"LLM generated example {i+1} for '{tool_name}' failed schema validation: {validation_error}. Skipping example.")
                                     continue # Skip examples that don't validate against the schema
                            validated_examples.append(validated_ex)
                        except ValidationError as pydantic_err:
                            logger.warning(f"Skipping invalid example structure at index {i} for '{tool_name}': {pydantic_err}. Data: {ex_data}")
                        except Exception as val_err: # Catch other potential validation errors
                            logger.warning(f"Error processing example item {i} for '{tool_name}': {val_err}. Data: {ex_data}")
                    else:
                        logger.warning(f"Skipping non-dictionary item in examples list for '{tool_name}' at index {i}.")
            else:
                # LLM did not return a list as requested
                raise ValueError(f"LLM response was not a JSON list as expected. Type: {type(examples_list_raw)}")

            logger.info(f"Successfully generated and validated {len(validated_examples)} examples for '{tool_name}'.")
            return validated_examples, cost

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Catch errors during parsing or validation
            logger.error(f"Failed to parse or validate LLM examples JSON for '{tool_name}': {e}. Raw response: {examples_text}", exc_info=True)
            # Return empty list and cost incurred so far
            return [], cost

    except Exception as e:
        # Catch errors during the generate_completion call or other unexpected issues
        logger.error(f"Error generating examples for tool '{tool_name}': {e}", exc_info=True)
        return [], cost # Return empty list and accumulated cost

async def _simulate_agent_usage(
    tool_name: str,
    current_docs: 'ProposedChanges', # Use forward reference for ProposedChanges
    current_schema: JSONSchemaObject,
    refinement_model_config: Dict[str, Any],
    num_simulations: int = 3
) -> Tuple[List['AgentSimulationResult'], float]: # Use forward reference
    """
    Simulates an LLM agent attempting to select and use the specified tool
    for various tasks, based ONLY on the provided documentation.

    Args:
        tool_name: Name of the tool being evaluated.
        current_docs: The ProposedChanges object containing the current description and examples.
        current_schema: The current JSON schema for the tool's input.
        refinement_model_config: LLM configuration for running the simulation.
        num_simulations: The number of different task scenarios to simulate.

    Returns:
        A tuple containing:
        - List[AgentSimulationResult]: Results of each simulation attempt.
        - float: Estimated cost incurred by the simulation LLM calls.
    """
    sim_results: List[AgentSimulationResult] = []
    total_cost = 0.0

    # --- Define Generic Task Scenarios ---
    # These tasks are designed to be potentially solvable by various tools,
    # forcing the simulation agent to decide if *this specific tool* is appropriate.
    generic_tasks = [
        f"Perform the core function described for the '{tool_name}' tool.",
        f"I need to process data related to '{tool_name}'. What arguments should I use?",
        f"Use the '{tool_name}' tool to handle the following input: [Provide Plausible Input Here - LLM will adapt].",
        f"Is '{tool_name}' the right tool for analyzing recent performance metrics?",
        f"How would I use '{tool_name}' to get information about 'example_entity'?",
        f"Configure '{tool_name}' for a quick, low-detail execution.",
        f"Execute '{tool_name}' with all optional parameters specified for maximum detail.",
        f"Compare the output of '{tool_name}' with a hypothetical alternative approach.",
        f"Use tool '{tool_name}' to extract key information about a user request.",
        f"Generate a report using the '{tool_name}' tool based on last week's data."
    ]
    # Select a diverse subset of tasks for simulation
    tasks_to_simulate = random.sample(generic_tasks, k=min(num_simulations, len(generic_tasks)))
    if not tasks_to_simulate:
        logger.warning(f"No tasks selected for agent simulation for tool '{tool_name}'.")
        return [], 0.0

    # --- Prepare Documentation Snippets for Prompt ---
    schema_str = json.dumps(current_schema, indent=2)
    # Limit examples in prompt to save tokens, focus on description/schema
    examples_str = json.dumps([ex.model_dump() for ex in current_docs.examples[:1]], indent=2) if current_docs.examples else "(No examples provided)"

    # --- Run Simulation for each Task ---
    for task_desc in tasks_to_simulate:
        prompt = f"""You are an AI assistant evaluating how to use an MCP tool based *only* on its provided documentation to accomplish a specific task.

**Your Assigned Task:** "{task_desc}"

**Tool Documentation Available:**

Tool Name: `{tool_name}`

Description:
```
{current_docs.description or '(No description provided)'}
```

Input Schema (JSON Schema):
```json
{schema_str}
```

Usage Examples:
```json
{examples_str}
```

**Your Evaluation Steps:**

1.  **Tool Selection:** Based *only* on the documentation, decide if `{tool_name}` is the *most appropriate* tool available to you for the assigned task. Explain your reasoning. If not appropriate, state why and set `tool_selected` to `null`.
2.  **Argument Formulation:** If you selected `{tool_name}`, formulate the *exact* arguments (as a JSON object) you would pass to it, strictly following the Input Schema and using information from the Description and Examples. If you cannot formulate valid arguments due to documentation issues (ambiguity, missing info, conflicting constraints), explain the problem clearly. Set `arguments_formulated` to `null` if formulation fails.
3.  **Reasoning:** Detail your step-by-step thought process for both selection and formulation. Highlight any parts of the documentation that were unclear, ambiguous, or potentially misleading.
4.  **Confidence:** Estimate your confidence (0.0 to 1.0) that the formulated arguments (if any) are correct and will lead to successful tool execution based *solely* on the documentation.
5.  **Success Flags:** Set `formulation_success` to `true` only if you successfully formulated arguments you believe are correct based on the documentation. Set `selection_error` or `formulation_error` with a brief explanation if applicable.

**Output Format:** Respond ONLY with a single, valid JSON object adhering precisely to this structure:

```json
{{
  "tool_selected": "{tool_name}" or null,
  "arguments_formulated": {{...}} or null,
  "formulation_success": true or false,
  "reasoning": "(String) Your detailed step-by-step reasoning for selection and formulation.",
  "confidence_score": "(Float) 0.0-1.0",
  "selection_error": "(String) Reason tool was deemed inappropriate, or null.",
  "formulation_error": "(String) Reason arguments could not be formulated correctly, or null."
}}
```

JSON Output:
"""

        try:
            logger.debug(f"Running agent simulation for task: '{task_desc}' (Tool: {tool_name})")
            result = await generate_completion(
                prompt=prompt,
                **refinement_model_config,
                temperature=0.4, # Balance creativity and adherence for simulation
                max_tokens=2000, # Allow space for reasoning
                additional_params={"response_format": {"type": "json_object"}} if refinement_model_config.get("provider") == Provider.OPENAI.value else None
            )
            total_cost += result.get("cost", 0.0)

            if not result.get("success"):
                raise ToolError(f"LLM simulation call failed: {result.get('error')}")

            sim_text = result["text"]
            logger.debug(f"Raw simulation response for task '{task_desc}': {sim_text[:500]}...")

            # --- Parse and Validate Response ---
            try:
                sim_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", sim_text.strip())
                sim_data = json.loads(sim_text_cleaned)

                # Validate required fields manually or using Pydantic partial validation if needed
                if not all(k in sim_data for k in ["formulation_success", "reasoning"]):
                    raise ValueError("Simulation response missing required fields.")

                # Use Pydantic model for structured data, handling potential missing optional keys
                sim_result_obj = AgentSimulationResult(
                    task_description=task_desc,
                    tool_selected=sim_data.get("tool_selected"),
                    arguments_formulated=sim_data.get("arguments_formulated"),
                    formulation_success=sim_data.get("formulation_success", False), # Default to False
                    reasoning=sim_data.get("reasoning"),
                    confidence_score=sim_data.get("confidence_score"),
                    selection_error=sim_data.get("selection_error"),
                    formulation_error=sim_data.get("formulation_error")
                )
                sim_results.append(sim_result_obj)

            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
                logger.error(f"Failed to parse/validate simulation result JSON for task '{task_desc}': {e}. Raw: {sim_text}", exc_info=True)
                # Append an error result
                sim_results.append(AgentSimulationResult(
                    task_description=task_desc,
                    formulation_success=False,
                    formulation_error=f"Failed to parse simulation response: {e}. Raw: {sim_text[:200]}...",
                    reasoning=f"LLM output was not valid JSON or did not match expected structure. Raw: {sim_text[:200]}..."
                ))

        except Exception as e:
            # Catch errors during the generate_completion call or other unexpected issues
            logger.error(f"Error during agent simulation LLM call for task '{task_desc}': {e}", exc_info=True)
            sim_results.append(AgentSimulationResult(
                task_description=task_desc,
                formulation_success=False,
                formulation_error=f"Simulation LLM call failed: {type(e).__name__}: {str(e)}",
                reasoning="The LLM call to simulate agent behavior failed."
            ))

    logger.info(f"Agent simulation completed for tool '{tool_name}' ({len(sim_results)} scenarios).")
    return sim_results, total_cost

async def _winnow_documentation(
    tool_name: str,
    current_docs: ProposedChanges, # Input includes description, examples, patches
    current_schema: JSONSchemaObject, # Pass schema for context
    refinement_model_config: Dict[str, Any]
) -> Tuple[ProposedChanges, float]: # Return type includes patches
    """
    Simplifies refined tool documentation for conciseness after stability is reached.

    Uses an LLM to rewrite the description and select the most essential examples,
    while preserving the existing (validated) schema patches.

    Args:
        tool_name: Name of the tool being winnowed.
        current_docs: The ProposedChanges object containing the latest refined
                      description, examples, and schema patches.
        current_schema: The JSON schema corresponding to the current_docs (used for context).
        refinement_model_config: LLM configuration for the winnowing task.

    Returns:
        A tuple containing:
        - ProposedChanges: A new object with the potentially more concise description
                          and pruned examples, but the *same* schema patches as input.
                          Returns a copy of input `current_docs` on failure.
        - float: Estimated cost of the LLM call during winnowing.
    """
    cost = 0.0
    logger.info(f"Winnowing documentation for stable tool '{tool_name}'...")

    # Prepare context for the winnowing LLM
    schema_str = json.dumps(current_schema, indent=2)
    # Include all current examples as context for the LLM to choose from
    examples_str = json.dumps([ex.model_dump() for ex in current_docs.examples], indent=2)

    # --- Construct Winnowing Prompt ---
    prompt = f"""# Task: Winnow Stable MCP Tool Documentation

The documentation for tool `{tool_name}` has been iteratively refined and is considered stable (high success rate in tests). Your goal is to make it more **concise** and **efficient** for an LLM agent to process, while preserving all *essential* information and correctness.

## Current Documentation (Input)

**Description:**
```
{current_docs.description or '(No description provided)'}
```

**Input Schema (for context):**
```json
{schema_str}
```

**Current Examples:**
```json
{examples_str or '(None provided)'}
```

## Instructions

1.  **Rewrite Description:** Create a concise version of the description.
    *   Focus on the core purpose and *critical* parameters/constraints.
    *   Remove redundancy or verbose explanations if the schema itself is clear enough (assume the schema is now relatively accurate).
    *   Retain essential warnings or critical usage notes.
2.  **Select Essential Examples:** Choose only the **1 or 2 most informative examples** from the `Current Examples` list.
    *   Prioritize examples that illustrate the most common use case or clarify a previously identified point of confusion (even if not explicitly stated, choose diverse ones).
    *   Ensure selected examples are minimal yet correct according to the schema. Keep comments concise.
3.  **Do NOT modify the schema itself.** Your output only includes the revised description and pruned examples.

## Output Format (Strict JSON Object):

```json
{{
  "description": "(String) The concise, winnowed description.",
  "examples": [
    {{ "args": {{...}}, "comment": "Concise comment for example 1" }},
    // (Optional) Include a second example if truly necessary for clarity
    {{ "args": {{...}}, "comment": "Concise comment for example 2" }}
  ]
}}
```

JSON Output:
"""

    # --- Call LLM for Winnowing ---
    try:
        logger.debug(f"Sending winnowing prompt for '{tool_name}' to {refinement_model_config.get('model')}")
        result = await generate_completion(
            prompt=prompt,
            **refinement_model_config,
            temperature=0.1, # Low temperature for precise editing
            max_tokens=2000, # Ample space, but expect shorter output
            additional_params={"response_format": {"type": "json_object"}} if refinement_model_config.get("provider") == Provider.OPENAI.value else None
        )

        if not result.get("success"):
            raise ToolError(f"LLM winnowing call failed: {result.get('error')}")

        cost += result.get("cost", 0.0)
        winnow_text = result["text"]
        logger.debug(f"Raw winnowing response: {winnow_text[:500]}...")

        # --- Parse and Validate Response ---
        try:
            winnow_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", winnow_text.strip())
            winnow_data = json.loads(winnow_text_cleaned)

            # Validate the structure and content using Pydantic (for examples)
            # We create a new ProposedChanges object, preserving original schema patches
            winnowed_proposal = ProposedChanges(
                description=winnow_data.get("description", current_docs.description), # Fallback
                schema_patches=current_docs.schema_patches, # IMPORTANT: Keep original patches
                examples=[GeneratedExample(**ex) for ex in winnow_data.get("examples", []) if isinstance(ex, dict)]
            )

            # Basic sanity check on example count
            if len(winnowed_proposal.examples) > 3: # Allow maybe 3 max, even if asked for 1-2
                logger.warning(f"Winnowing returned {len(winnowed_proposal.examples)} examples, expected 1-2. Keeping all.")

            logger.info(f"Successfully winnowed documentation for '{tool_name}'.")
            return winnowed_proposal, cost

        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.error(f"Failed to parse/validate winnowing JSON for '{tool_name}': {e}. Raw: {winnow_text}", exc_info=True)
            # On failure, return the *original* ProposedChanges object, indicating winnowing failed
            return current_docs.model_copy(deep=True), cost

    except Exception as e:
        logger.error(f"Error during winnowing LLM call or processing for '{tool_name}': {e}", exc_info=True)
        # Return the *original* ProposedChanges object on any execution error
        return current_docs.model_copy(deep=True), cost
