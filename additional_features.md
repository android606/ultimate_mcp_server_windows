# --- Command Line Execution ---

# 1. Ensure the Ultimate MCP Server is running in one terminal:
#    (Activate venv)
#    ultimate-mcp-server run

# 2. Ensure your .env file is configured with available API keys.

# 3. In another terminal, run the test orchestrator script:
#    (Activate venv)
#    python run_all_demo_scripts_and_check_for_errors.py [OPTIONS]

# Example Options (Check the script's argparse setup for exact flags):
#    --tests-to-run <test_name_pattern> # Run only specific tests
#    --include-providers <provider1,provider2> # Only run tests involving these providers
#    --output-format <simple|detailed|html> # Control output verbosity/format
#    --fail-fast # Stop on the first failure

# --- Example Code Snippet (Illustrating the runner's purpose) ---
# This Python code shows how you might import and use the runner function if needed,
# based on the example provided in the source.

from run_all_demo_scripts_and_check_for_errors import run_test_suite # Import the main function

# Define test parameters
test_params = {
    "tests_to_run": "all",  # Or a pattern like "browser*" to run only browser tests
    "include_providers": ["openai", "anthropic"], # Limit tests to these providers if keys are set
    "exclude_providers": ["gemini"], # Explicitly exclude providers
    "output_format": "detailed", # Get detailed console output
    "fail_fast": False, # Continue running tests even if one fails
    # Add other parameters supported by run_test_suite if needed
}

print(f"Starting test suite with parameters: {test_params}")

# Execute the test suite
results = run_test_suite(**test_params)

# Process and display the results summary
print("\n--- Test Suite Summary ---")
print(f"Total Tests Attempted: {results.get('total_tests', 0)}")
print(f"Passed: {results.get('passed', 0)}")
print(f"Failed: {results.get('failed', 0)}")
print(f"Skipped (e.g., missing keys/deps): {results.get('skipped', 0)}")
print("--------------------------")

# The HTML report (if generated) provides detailed logs
html_report_content = results.get("html_report")
if html_report_content:
    report_path = "test_suite_report.html"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report_content)
        print(f"Detailed HTML report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving HTML report: {e}")
else:
    # Handle case where HTML report wasn't generated based on output_format
    if test_params.get("output_format") == "html":
         print("HTML report was requested but not found in results.")

# Note: This code block assumes you can import and run the test suite function
# directly. The primary intended use is likely via the command line script.