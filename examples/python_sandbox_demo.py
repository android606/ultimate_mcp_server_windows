#!/usr/bin/env python
"""Python sandbox execution demo using Ultimate MCP Server."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
# These imports need to be below sys.path modification, which is why they have noqa comments
from rich import box  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from ultimate_mcp_server.constants import TaskType  # noqa: E402

# Project imports
from ultimate_mcp_server.core.server import Gateway  # noqa: E402
from ultimate_mcp_server.tools.python_js_sandbox import PythonSandboxTool  # noqa: E402
from ultimate_mcp_server.utils import get_logger  # noqa: E402
from ultimate_mcp_server.utils.display import CostTracker  # noqa: E402
from ultimate_mcp_server.utils.logging.console import console  # noqa: E402

# Initialize logger
logger = get_logger("example.python_sandbox_demo")


async def run_basic_execution(sandbox_tool):
    """Demonstrate basic one-shot Python code execution."""
    console.print(Rule("[bold blue]Basic Python Execution[/bold blue]"))
    logger.info("Demonstrating basic Python code execution", emoji_key="start")
    
    # Simple code to execute
    code = """
import numpy as np
import math

# Calculate some values
numbers = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(numbers)
std_dev = np.std(numbers)
sqrt_sum = math.sqrt(np.sum(numbers))

# Print results
print(f"Numbers: {numbers}")
print(f"Mean: {mean_value:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Square Root of Sum: {sqrt_sum:.2f}")

# Store result for return
result = {
    "mean": float(mean_value),
    "std_dev": float(std_dev),
    "sqrt_sum": float(sqrt_sum)
}
"""
    
    try:
        # Execute the code
        logger.info("Executing basic Python code...", emoji_key=TaskType.CODE_EXECUTION.value)
        start_time = time.time()
        
        result = await sandbox_tool.execute_python(
            code=code,
            packages=["numpy"],  # numpy is already preloaded, but specifying for clarity
            timeout_ms=10000
        )
        
        execution_time = time.time() - start_time
        
        # Format and display result
        stdout = result.get("stdout", "").strip()
        result_value = result.get("result")
        
        # Log success
        logger.success(
            "Code execution completed successfully",
            emoji_key="success",
            elapsed_ms=execution_time * 1000,
            session_id=result.get("session_id")
        )
        
        # Display code
        console.print(Panel(
            code.strip(),
            title="[bold cyan]Python Code[/bold cyan]",
            border_style="blue",
            expand=False
        ))
        
        # Display stdout
        if stdout:
            console.print(Panel(
                escape(stdout),
                title="[bold green]Standard Output[/bold green]",
                border_style="green",
                expand=False
            ))
        
        # Display returned result
        if result_value:
            # Format result nicely if it's a dictionary
            if isinstance(result_value, dict):
                result_table = Table(title="Return Value", box=box.SIMPLE, show_header=True)
                result_table.add_column("Metric", style="cyan")
                result_table.add_column("Value", style="yellow")
                
                for key, value in result_value.items():
                    # Format floating point values
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    result_table.add_row(key, formatted_value)
                
                console.print(result_table)
            else:
                # For non-dict results, just display as text
                console.print(Panel(
                    str(result_value),
                    title="[bold magenta]Return Value[/bold magenta]",
                    border_style="magenta",
                    expand=False
                ))
        
        # Display execution stats
        stats_table = Table(title="Execution Statistics", box=box.SIMPLE, show_header=False)
        stats_table.add_column("Metric", style="blue")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Python Execution Time", f"{result.get('elapsed_py_ms', 0):.2f} ms")
        stats_table.add_row("Total Wall Time", f"{result.get('elapsed_wall_ms', 0):.2f} ms")
        stats_table.add_row("Session ID", result.get("session_id", "N/A"))
        console.print(stats_table)
        
    except Exception as e:
        logger.error(f"Error during basic code execution: {str(e)}", emoji_key="error", exc_info=True)


async def run_repl_execution(sandbox_tool):
    """Demonstrate REPL-mode execution with state preservation."""
    console.print(Rule("[bold blue]REPL Mode with State Preservation[/bold blue]"))
    logger.info("Demonstrating REPL mode with persistent state", emoji_key="start")
    
    # First code execution - define class and variables
    setup_code = """
import numpy as np
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, data):
        self.data = np.array(data)
    
    def calculate_stats(self):
        return {
            "mean": float(np.mean(self.data)),
            "median": float(np.median(self.data)),
            "std_dev": float(np.std(self.data)),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data))
        }
    
    def transform(self, operation="square"):
        if operation == "square":
            return self.data ** 2
        elif operation == "sqrt":
            return np.sqrt(self.data)
        elif operation == "log":
            return np.log(self.data)
        else:
            raise ValueError(f"Unknown operation: {operation}")

# Create an analyzer with some data
data = [10, 25, 17, 32, 8, 42, 30]
analyzer = DataAnalyzer(data)

print(f"Initialized DataAnalyzer with data: {data}")
"""
    
    # Second code execution - use the previously defined class and data
    analysis_code = """
# Use the previously defined analyzer
stats = analyzer.calculate_stats()
print(f"Data statistics: {stats}")

# Transform the data
squared_data = analyzer.transform("square")
print(f"Squared data: {squared_data}")

# Store results for return
result = {
    "original_stats": stats,
    "squared_data": squared_data.tolist()
}
"""
    
    try:
        # First execution - setup
        logger.info("REPL Step 1: Initializing with class definition and data", emoji_key=TaskType.CODE_EXECUTION.value)
        repl_result1 = await sandbox_tool.repl_python(
            code=setup_code,
            packages=["numpy", "matplotlib"],
            timeout_ms=15000
        )
        
        # Get handle for persistent session
        handle = repl_result1.get("handle")
        logger.info(f"REPL session initialized with handle: {handle}", emoji_key="info")
        
        # Display first execution output
        console.print(Panel(
            escape(repl_result1.get("stdout", "").strip()),
            title="[bold cyan]REPL Step 1 Output[/bold cyan]",
            border_style="cyan",
            expand=False
        ))
        
        # Second execution - analysis using persistent state
        logger.info("REPL Step 2: Using persistent state to analyze data", emoji_key=TaskType.CODE_EXECUTION.value)
        repl_result2 = await sandbox_tool.repl_python(
            code=analysis_code,
            handle=handle,  # Reuse the same session handle
            timeout_ms=15000
        )
        
        # Display second execution output
        console.print(Panel(
            escape(repl_result2.get("stdout", "").strip()),
            title="[bold green]REPL Step 2 Output[/bold green]",
            border_style="green",
            expand=False
        ))
        
        # Display returned result from second execution
        result_value = repl_result2.get("result")
        if result_value and isinstance(result_value, dict):
            # Create a nice table for the statistics
            stats_table = Table(title="Data Analysis Results", box=box.SIMPLE)
            
            # Original stats section
            stats_table.add_section()
            stats_table.add_row("[bold]Original Statistics[/bold]", "")
            
            if "original_stats" in result_value:
                for key, value in result_value["original_stats"].items():
                    stats_table.add_row(key, f"{value:.2f}")
            
            # Transformed data section
            stats_table.add_section()
            stats_table.add_row("[bold]Squared Data[/bold]", "")
            
            if "squared_data" in result_value:
                squared_str = ", ".join(f"{val:.1f}" for val in result_value["squared_data"])
                stats_table.add_row("Values", squared_str)
            
            console.print(stats_table)
        
        # Log success
        logger.success(
            "REPL execution completed successfully",
            emoji_key="success",
            handle=handle
        )
        
    except Exception as e:
        logger.error(f"Error during REPL execution: {str(e)}", emoji_key="error", exc_info=True)


async def run_data_visualization(sandbox_tool):
    """Demonstrate creating data visualizations with matplotlib."""
    console.print(Rule("[bold blue]Data Visualization with Matplotlib[/bold blue]"))
    logger.info("Demonstrating data visualization capabilities", emoji_key="start")
    
    # Code to generate a visualization
    viz_code = """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# Create sample data
np.random.seed(42)
dates = pd.date_range('20230101', periods=20)
df = pd.DataFrame({
    'Category A': np.random.randn(20).cumsum(),
    'Category B': np.random.randn(20).cumsum(),
    'Category C': np.random.randn(20).cumsum(),
}, index=dates)

# Create a basic visualization
plt.figure(figsize=(10, 6))
df.plot()
plt.title('Random Walk Comparison')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# Instead of saving to file, capture and convert to text
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Save plot to a buffer
buffer = BytesIO()
canvas = FigureCanvasAgg(plt.gcf())
canvas.print_png(buffer)
buffer.seek(0)

# Generate a base64 encoded version of this plot
b64_img = base64.b64encode(buffer.read()).decode('utf-8')

# Print some info about the dataframe
print("DataFrame Summary:")
print(df.describe())

# Return a dict with the visualization and data
result = {
    "b64_image": b64_img,
    "data_shape": df.shape,
    "data_summary": df.describe().to_dict()
}
"""
    
    try:
        # Execute the visualization code
        logger.info("Generating data visualization...", emoji_key=TaskType.CODE_EXECUTION.value)
        viz_result = await sandbox_tool.execute_python(
            code=viz_code,
            packages=["numpy", "matplotlib", "pandas"],
            timeout_ms=20000
        )
        
        # Log success
        logger.success(
            "Visualization generation completed successfully",
            emoji_key="success",
            elapsed_ms=viz_result.get("elapsed_wall_ms")
        )
        
        # Display stdout (dataframe description)
        stdout = viz_result.get("stdout", "").strip()
        if stdout:
            console.print(Panel(
                escape(stdout),
                title="[bold cyan]Data Summary[/bold cyan]",
                border_style="cyan",
                expand=False
            ))
        
        # Get the result with the base64 encoded image
        result_value = viz_result.get("result")
        if result_value and isinstance(result_value, dict) and "b64_image" in result_value:
            b64_img = result_value["b64_image"]
            
            # Note about the image viewing
            console.print("[yellow]Note: In a real application environment, you would decode and display the base64 image.[/yellow]")
            console.print(f"[green]A base64-encoded visualization was successfully generated ({len(b64_img)//1024} KB)[/green]")
            
            # Display data shape
            if "data_shape" in result_value:
                console.print(f"[cyan]Data Shape: {result_value['data_shape']}[/cyan]")
        
        # Display execution stats
        stats_table = Table(title="Visualization Execution Statistics", box=box.SIMPLE, show_header=False)
        stats_table.add_column("Metric", style="blue")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Python Execution Time", f"{viz_result.get('elapsed_py_ms', 0):.2f} ms")
        stats_table.add_row("Total Wall Time", f"{viz_result.get('elapsed_wall_ms', 0):.2f} ms")
        console.print(stats_table)
        
    except Exception as e:
        logger.error(f"Error during visualization generation: {str(e)}", emoji_key="error", exc_info=True)


async def run_network_access_demo(sandbox_tool):
    """Demonstrate optional network access capabilities."""
    console.print(Rule("[bold blue]Network Access Capabilities[/bold blue]"))
    logger.info("Demonstrating network access capabilities (when enabled)", emoji_key="start")
    
    # Code that attempts to access network resources
    network_code = """
import requests
import json

# Function to make a request and handle errors gracefully
def fetch_data(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return {
            "status": response.status_code,
            "content_type": response.headers.get('content-type', ''),
            "data": response.json() if 'application/json' in response.headers.get('content-type', '') else None,
            "success": True
        }
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {"success": False, "error": str(e)}

# Try to fetch data from a public API (JSONPlaceholder - a common test API)
api_url = "https://jsonplaceholder.typicode.com/todos/1"
result = fetch_data(api_url)

if result["success"]:
    print(f"Successfully fetched data from {api_url}")
    print(f"Status code: {result['status']}")
    print(f"Content type: {result['content_type']}")
    if result["data"]:
        print(f"Data: {json.dumps(result['data'], indent=2)}")
else:
    print(f"Failed to fetch data: {result.get('error', 'Unknown error')}")
"""
    
    try:
        # Execute once without network access (this should fail)
        logger.info("Executing with network access disabled (should fail)", emoji_key="info")
        no_network_result = await sandbox_tool.execute_python(
            code=network_code,
            allow_network=False,  # Explicitly disable network access
            packages=["requests"],
            timeout_ms=15000
        )
        
        # Display the result (should contain an error)
        console.print(Panel(
            escape(no_network_result.get("stdout", "").strip()),
            title="[bold red]With Network Access Disabled[/bold red]",
            border_style="red",
            expand=False
        ))
        
        # Now execute with network access enabled
        logger.info("Executing with network access enabled", emoji_key=TaskType.CODE_EXECUTION.value)
        network_result = await sandbox_tool.execute_python(
            code=network_code,
            allow_network=True,  # Enable network access
            packages=["requests"],
            timeout_ms=15000
        )
        
        # Display the result (should show successful API call)
        console.print(Panel(
            escape(network_result.get("stdout", "").strip()),
            title="[bold green]With Network Access Enabled[/bold green]",
            border_style="green",
            expand=False
        ))
        
        # Log success
        logger.success(
            "Network access demonstration completed",
            emoji_key="success"
        )
        
    except Exception as e:
        logger.error(f"Error during network access demo: {str(e)}", emoji_key="error", exc_info=True)


async def run_sandbox_performance_test(sandbox_tool):
    """Test performance of the sandbox with various computational tasks."""
    console.print(Rule("[bold blue]Sandbox Performance Testing[/bold blue]"))
    logger.info("Testing sandbox performance with different computational tasks", emoji_key="start")
    
    # Performance test code - includes multiple different computational tasks
    perf_code = """
import numpy as np
import time
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Dictionary to store performance results
performance_results = {}

# 1. Basic numpy operations
print("Testing basic NumPy operations...")
start_time = time.time()
size = 1000
arr1 = np.random.random((size, size))
arr2 = np.random.random((size, size))
result = np.dot(arr1, arr2)
array_ops_time = time.time() - start_time
performance_results["numpy_matrix_multiply"] = array_ops_time
print(f"NumPy {size}x{size} matrix multiplication: {array_ops_time:.4f} seconds")

# 2. Pandas operations
print("\\nTesting Pandas operations...")
start_time = time.time()
rows = 100000
df = pd.DataFrame({
    'A': np.random.randn(rows),
    'B': np.random.randn(rows),
    'C': np.random.choice(['X', 'Y', 'Z'], rows),
    'D': np.random.randint(0, 100, rows)
})
# Perform some typical pandas operations
grouped = df.groupby('C').agg({
    'A': ['mean', 'median', 'std'],
    'B': ['min', 'max', 'sum'],
    'D': ['count', 'mean']
})
pandas_time = time.time() - start_time
performance_results["pandas_group_agg"] = pandas_time
print(f"Pandas operations on {rows} rows: {pandas_time:.4f} seconds")

# 3. Statistical operations
print("\\nTesting statistical operations...")
start_time = time.time()
sample_size = 10000
# Generate some distributions
dist1 = np.random.normal(0, 1, sample_size)
dist2 = np.random.normal(0.5, 1.5, sample_size)
# Run various statistical tests
t_test = stats.ttest_ind(dist1, dist2)
corr = np.corrcoef(dist1, dist2)[0, 1]
ks_test = stats.ks_2samp(dist1, dist2)
stats_time = time.time() - start_time
performance_results["statistical_tests"] = stats_time
print(f"Statistical operations: {stats_time:.4f} seconds")

# 4. Pure Python operations (for comparison)
print("\\nTesting pure Python operations...")
start_time = time.time()
result = 0
iterations = 1000000
for i in range(iterations):
    result += i ** 2 / (i + 1)
python_time = time.time() - start_time
performance_results["pure_python_loop"] = python_time
print(f"Pure Python loop ({iterations} iterations): {python_time:.4f} seconds")

# Return the results
result = performance_results
print("\\nAll performance tests completed.")
"""
    
    try:
        # Execute the performance test code
        logger.info("Running performance tests...", emoji_key=TaskType.CODE_EXECUTION.value)
        perf_result = await sandbox_tool.execute_python(
            code=perf_code,
            packages=["numpy", "pandas", "scipy", "matplotlib"],
            timeout_ms=60000  # Longer timeout for performance tests
        )
        
        # Log success
        logger.success(
            "Performance tests completed successfully",
            emoji_key="success",
            elapsed_ms=perf_result.get("elapsed_wall_ms")
        )
        
        # Display stdout (performance results)
        stdout = perf_result.get("stdout", "").strip()
        if stdout:
            console.print(Panel(
                escape(stdout),
                title="[bold cyan]Performance Test Results[/bold cyan]",
                border_style="cyan",
                expand=False
            ))
        
        # Display results in a table
        result_value = perf_result.get("result")
        if result_value and isinstance(result_value, dict):
            perf_table = Table(title="Performance Summary", box=box.SIMPLE)
            perf_table.add_column("Operation", style="blue")
            perf_table.add_column("Time (seconds)", style="green", justify="right")
            
            for operation, time_value in result_value.items():
                perf_table.add_row(operation, f"{time_value:.4f}")
            
            console.print(perf_table)
        
        # Display execution stats
        stats_table = Table(title="Execution Statistics", box=box.SIMPLE, show_header=False)
        stats_table.add_column("Metric", style="blue")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Python Execution Time", f"{perf_result.get('elapsed_py_ms', 0):.2f} ms")
        stats_table.add_row("Total Wall Time", f"{perf_result.get('elapsed_wall_ms', 0):.2f} ms")
        console.print(stats_table)
        
    except Exception as e:
        logger.error(f"Error during performance testing: {str(e)}", emoji_key="error", exc_info=True)


async def main():
    """Run Python sandbox demo examples."""
    console.print("[bold blue]Python Sandbox Demonstration[/bold blue]")
    logger.info("Starting Python Sandbox demonstration", emoji_key="start")
    
    try:
        # Initialize CostTracker (used in other demos, kept for consistency)
        tracker = CostTracker()
        
        # Create Gateway instance
        gateway = Gateway("python-sandbox-demo", register_tools=False)
        
        # Initialize Python Sandbox Tool
        sandbox_tool = PythonSandboxTool(gateway)
        logger.info("Initialized Python Sandbox Tool", emoji_key="info")
        
        # Run each demo with spacing between them
        await run_basic_execution(sandbox_tool)
        console.print()
        
        await run_repl_execution(sandbox_tool)
        console.print()
        
        await run_data_visualization(sandbox_tool)
        console.print()
        
        await run_network_access_demo(sandbox_tool)
        console.print()
        
        await run_sandbox_performance_test(sandbox_tool)
        console.print()
        
        # Display final summary (would show costs if this were an LLM tool)
        tracker.display_summary(console)
        
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    
    logger.success("Python Sandbox Demo Finished Successfully!", emoji_key="complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)