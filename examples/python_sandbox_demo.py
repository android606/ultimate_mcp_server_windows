#!/usr/bin/env python
"""Demonstration script for PythonSandboxTool in Ultimate MCP Server."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich imports for nice UI
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.exceptions import ToolError
from ultimate_mcp_server.tools.python_js_sandbox import PythonSandboxTool
from ultimate_mcp_server.utils import get_logger

# Initialize Rich console and logger
console = Console()
logger = get_logger("demo.python_sandbox")

# Install rich tracebacks for better error display
install_rich_traceback(show_locals=False, width=console.width)

# --- Demo Helper Functions ---

def display_result(title: str, result: Dict[str, Any], code_str: Optional[str] = None) -> None:
    """Display execution result with enhanced formatting."""
    console.print(Rule(f"[bold cyan]{escape(title)}[/bold cyan]"))

    if code_str:
        console.print(Panel(
            Syntax(code_str.strip(), "python", theme="monokai", line_numbers=True, word_wrap=True),
            title="Executed Code",
            border_style="blue",
            padding=(1, 2)
        ))
    
    if not result.get("success", True):  # Most results don't have a success field
        error_msg = result.get("error", "Unknown error")
        console.print(Panel(
            f"[bold red]:x: Operation Failed:[/]\n{escape(error_msg)}",
            title="Error",
            border_style="red",
            padding=(1, 2),
            expand=False
        ))
        return
    
    # Create output panel for stdout/stderr
    output_parts = []
    
    if stdout := result.get("stdout", ""):
        output_parts.append(f"[bold green]STDOUT:[/]\n{escape(stdout)}")
    
    if stderr := result.get("stderr", ""):
        if output_parts:
            output_parts.append("\n")
        output_parts.append(f"[bold red]STDERR:[/]\n{escape(stderr)}")
    
    if output_parts:
        console.print(Panel(
            "\n".join(output_parts),
            title="Output",
            border_style="yellow",
            padding=(1, 2)
        ))
    
    # Display result value if present
    if "result" in result and result["result"] is not None:
        console.print(Panel(
            Syntax(str(result["result"]), "python", theme="monokai", line_numbers=False, word_wrap=True),
            title="Result Value",
            border_style="green",
            padding=(1, 2)
        ))
    
    # Display execution stats
    stats_table = Table(title="Execution Statistics", box=box.ROUNDED, show_header=False, padding=(0, 1), border_style="dim")
    stats_table.add_column("Metric", style="cyan", justify="right")
    stats_table.add_column("Value", style="white")
    
    if "elapsed_py_ms" in result:
        stats_table.add_row("Python Execution Time", f"{result['elapsed_py_ms']:.2f} ms")
    if "elapsed_wall_ms" in result:
        stats_table.add_row("Wall Clock Time", f"{result['elapsed_wall_ms']:.2f} ms")
    if "session_id" in result:
        stats_table.add_row("Session ID", result["session_id"])
    if "handle" in result:
        stats_table.add_row("REPL Handle", result["handle"])
    
    console.print(stats_table)
    console.print()  # Add spacing

# --- Demo Functions ---

async def basic_execution_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate basic Python code execution."""
    console.print(Rule("[bold green]1. Basic Python Execution Demo[/bold green]", style="green"))
    logger.info("Starting basic execution demo")
    
    # Simple Python code
    code = """
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

# Calculate area of a circle with radius 5
area = calculate_circle_area(5)
print(f"The area of a circle with radius 5 is {area:.2f}")

# Assign to result for return value
result = f"Circle area calculation complete. Area = {area:.4f}"
"""

    console.print("[cyan]Executing basic Python code...[/]")
    
    try:
        with console.status("[bold cyan]Running Python code...", spinner="dots"):
            execution_result = await python_tool.execute_python(
                code=code,
                timeout_ms=5000
            )
        
        display_result("Basic Python Execution", execution_result, code)
        
    except Exception as e:
        logger.error(f"Basic execution demo failed: {e}")
        console.print(f"[bold red]:x: Execution Error:[/]\n{escape(str(e))}")
    
    console.print()  # Spacing

async def package_loading_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate loading Python packages."""
    console.print(Rule("[bold green]2. Package Loading Demo[/bold green]", style="green"))
    logger.info("Starting package loading demo")
    
    # NumPy example
    numpy_code = """
import numpy as np

# Create a simple array
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
print(f"Standard deviation: {np.std(arr)}")

# Create a 3x3 identity matrix
identity = np.eye(3)
print(f"3x3 Identity matrix:\\n{identity}")

result = "NumPy operations completed successfully"
"""

    console.print("[cyan]Executing code with NumPy package...[/]")
    
    try:
        with console.status("[bold cyan]Running Python code with NumPy...", spinner="dots"):
            numpy_result = await python_tool.execute_python(
                code=numpy_code,
                packages=["numpy"],
                timeout_ms=10000  # Give it more time for package loading
            )
        
        display_result("NumPy Package Demo", numpy_result, numpy_code)
        
        # Now try with Pandas (which depends on NumPy)
        console.print("[cyan]Now executing code with Pandas (depends on NumPy)...[/]")
        
        pandas_code = """
import numpy as np
import pandas as pd

# Create a simple DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)

# Basic stats
print("\\nSummary statistics for Age:")
print(df['Age'].describe())

result = df.to_dict()  # Return the DataFrame as a dictionary
"""

        with console.status("[bold cyan]Running Python code with Pandas...", spinner="dots"):
            pandas_result = await python_tool.execute_python(
                code=pandas_code,
                packages=["numpy", "pandas"],
                timeout_ms=15000  # Pandas can take longer to load
            )
        
        display_result("Pandas Package Demo", pandas_result, pandas_code)
        
    except Exception as e:
        logger.error(f"Package loading demo failed: {e}")
        console.print(f"[bold red]:x: Package Loading Error:[/]\n{escape(str(e))}")
    
    console.print()  # Spacing

async def repl_mode_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate REPL mode with persistent state."""
    console.print(Rule("[bold green]3. REPL Mode Demo[/bold green]", style="green"))
    logger.info("Starting REPL mode demo")
    
    repl_handle = None
    
    try:
        # Step 1: Define variables and functions
        step1_code = """
# Define a class
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name} and I'm {self.age} years old."

# Create a list of people
people = [
    Person("Alice", 30),
    Person("Bob", 25),
    Person("Charlie", 35)
]

# Define a function
def find_person(name):
    return next((p for p in people if p.name == name), None)

print("Defined Person class and created a list of people")
print("Also defined find_person() function")

# No explicit result
"""

        console.print("[cyan]Step 1: Defining variables, classes, and functions in REPL...[/]")
        
        with console.status("[bold cyan]Running first REPL code block...", spinner="dots"):
            step1_result = await python_tool.repl_python(
                code=step1_code,
                timeout_ms=5000
            )
        
        display_result("REPL Step 1: Define Variables and Functions", step1_result, step1_code)
        repl_handle = step1_result.get("handle")
        
        # Step 2: Use previously defined variables and functions
        step2_code = """
# Find a person from the list defined in the previous step
alice = find_person("Alice")
print(alice.greet())

# Add a new person
people.append(Person("David", 40))
print(f"Now we have {len(people)} people in our list")

# Set result to be the list of names
result = [person.name for person in people]
"""

        console.print("[cyan]Step 2: Using previously defined variables and functions...[/]")
        
        with console.status("[bold cyan]Running second REPL code block...", spinner="dots"):
            step2_result = await python_tool.repl_python(
                code=step2_code,
                handle=repl_handle,
                timeout_ms=5000
            )
        
        display_result("REPL Step 2: Use Previously Defined State", step2_result, step2_code)
        
        # Step 3: Add an import and use more complex operations
        step3_code = """
import random

# Randomly select a person
selected = random.choice(people)
print(f"Randomly selected: {selected.name}")

# Update ages
for person in people:
    person.age += 1

print("Everyone is now one year older:")
for person in people:
    print(f"{person.name} is now {person.age} years old")

# Count total age
total_age = sum(person.age for person in people)
result = f"Total age of all {len(people)} people: {total_age}"
"""

        console.print("[cyan]Step 3: Adding imports and performing more operations...[/]")
        
        with console.status("[bold cyan]Running third REPL code block...", spinner="dots"):
            step3_result = await python_tool.repl_python(
                code=step3_code,
                handle=repl_handle,
                timeout_ms=5000
            )
        
        display_result("REPL Step 3: Adding Imports", step3_result, step3_code)
        
        # Step 4: Reset the REPL and show that state is cleared
        step4_code = """
# Try to access previously defined variables
try:
    print(f"People list length: {len(people)}")
    print("State persisted successfully")
    result = True
except NameError as e:
    print(f"Error: {e}")
    print("State has been reset as expected")
    result = False
"""

        console.print("[cyan]Step 4: Resetting REPL state...[/]")
        
        with console.status("[bold cyan]Resetting REPL and testing state...", spinner="dots"):
            step4_result = await python_tool.repl_python(
                code=step4_code,
                handle=repl_handle,
                reset=True,  # Reset the REPL state
                timeout_ms=5000
            )
        
        display_result("REPL Step 4: Reset State", step4_result, step4_code)
        
    except Exception as e:
        logger.error(f"REPL mode demo failed: {e}")
        console.print(f"[bold red]:x: REPL Mode Error:[/]\n{escape(str(e))}")
    
    console.print()  # Spacing

async def error_handling_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate error handling."""
    console.print(Rule("[bold green]4. Error Handling Demo[/bold green]", style="green"))
    logger.info("Starting error handling demo")
    
    # Code with syntax error
    syntax_error_code = """
def calculate_ratio(a, b)
    # Missing colon after function definition
    return a / b

result = calculate_ratio(10, 2)
"""

    console.print("[cyan]Executing code with syntax error...[/]")
    
    try:
        with console.status("[bold cyan]Running code with syntax error...", spinner="dots"):
            syntax_result = await python_tool.execute_python(
                code=syntax_error_code,
                timeout_ms=5000
            )
        
        display_result("Syntax Error Handling", syntax_result, syntax_error_code)
    except ToolError as e:
        logger.warning(f"Caught expected ToolError: {e}")
        console.print(Panel(
            f"[bold red]Expected Error:[/]\n{escape(str(e))}",
            title="Syntax Error Result",
            border_style="yellow",
            padding=(1, 2)
        ))
    except Exception as e:
        logger.error(f"Unexpected error type: {e}")
        console.print(f"[bold red]:x: Unexpected Error Type:[/]\n{escape(str(e))}")
    
    # Code with runtime error
    runtime_error_code = """
def calculate_ratio(a, b):
    return a / b

# Division by zero will cause a runtime error
result = calculate_ratio(10, 0)
"""

    console.print("[cyan]Executing code with runtime error (division by zero)...[/]")
    
    try:
        with console.status("[bold cyan]Running code with runtime error...", spinner="dots"):
            runtime_result = await python_tool.execute_python(
                code=runtime_error_code,
                timeout_ms=5000
            )
        
        display_result("Runtime Error Handling", runtime_result, runtime_error_code)
        
    except Exception as e:
        logger.error(f"Error handling demo failed: {e}")
        console.print(f"[bold red]:x: Error Handling Demo Failed:[/]\n{escape(str(e))}")
    
    console.print()  # Spacing

async def timeout_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate timeout handling."""
    console.print(Rule("[bold green]5. Timeout Handling Demo[/bold green]", style="green"))
    logger.info("Starting timeout handling demo")
    
    # Code with infinite loop
    timeout_code = """
import time

print("Starting computation that will time out...")
# This will run until timeout occurs
for i in range(1000000):
    # Do some work and print progress every 10000 iterations
    if i % 10000 == 0:
        print(f"Iteration {i}...")
        time.sleep(0.1)  # Sleep to ensure we hit the timeout

print("This line should never be reached due to timeout")
result = "Completed successfully"
"""

    console.print("[cyan]Executing code that should time out (short timeout)...[/]")
    
    try:
        with console.status("[bold cyan]Running code with 2-second timeout...", spinner="dots"):
            timeout_result = await python_tool.execute_python(
                code=timeout_code,
                timeout_ms=2000  # Only give it 2 seconds
            )
        
        # This should not be reached
        display_result("Timeout Should Have Occurred", timeout_result, timeout_code)
        console.print("[bold red]Error: Expected a timeout but code completed![/]")
    except ToolError as e:
        if "timeout" in str(e).lower():
            logger.info("Caught expected timeout error")
            console.print(Panel(
                f"[bold green]:heavy_check_mark: Timeout successfully detected:[/]\n{escape(str(e))}",
                title="Timeout Handling",
                border_style="green",
                padding=(1, 2)
            ))
        else:
            logger.error(f"Expected timeout error but got different error: {e}")
            console.print(f"[bold yellow]Unexpected Error (expected timeout):[/]\n{escape(str(e))}")
    except Exception as e:
        logger.error(f"Timeout demo failed with unexpected error type: {e}")
        console.print(f"[bold red]:x: Unexpected Error Type:[/]\n{escape(str(e))}")
    
    console.print()  # Spacing

async def network_access_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate network access control."""
    console.print(Rule("[bold green]6. Network Access Control Demo[/bold green]", style="green"))
    logger.info("Starting network access demo")
    
    # Network access denied code
    network_denied_code = """
try:
    import urllib.request
    
    print("Attempting to fetch data from the internet...")
    response = urllib.request.urlopen('https://httpbin.org/get')
    data = response.read().decode('utf-8')
    print(f"Data fetched successfully:\\n{data[:100]}...")
    result = "Network request successful"
except Exception as e:
    print(f"Network error: {e}")
    result = f"Network request failed: {str(e)}"
"""

    console.print("[cyan]Executing code with network access denied (default)...[/]")
    
    try:
        with console.status("[bold cyan]Running code with network access denied...", spinner="dots"):
            denied_result = await python_tool.execute_python(
                code=network_denied_code,
                timeout_ms=10000
            )
        
        display_result("Network Access Denied (Default)", denied_result, network_denied_code)
        
        # Network access allowed code (same code)
        console.print("[cyan]Now executing the same code with network access allowed...[/]")
        
        with console.status("[bold cyan]Running code with network access allowed...", spinner="dots"):
            allowed_result = await python_tool.execute_python(
                code=network_denied_code,
                allow_network=True,  # Now allow network
                timeout_ms=10000
            )
        
        display_result("Network Access Allowed", allowed_result, network_denied_code)
        
    except Exception as e:
        logger.error(f"Network access demo failed: {e}")
        console.print(f"[bold red]:x: Network Access Demo Failed:[/]\n{escape(str(e))}")
    
    console.print()  # Spacing

async def filesystem_access_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate filesystem access."""
    console.print(Rule("[bold green]7. Filesystem Access Demo[/bold green]", style="green"))
    logger.info("Starting filesystem access demo")
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        f.write("Hello from the file system!\nThis is a test file created for the demo.")
        temp_file_path = f.name
    
    try:
        # Filesystem access code
        fs_code = f"""
import os

# Get the path to the test file
file_path = "{temp_file_path.replace('\\', '\\\\')}"  # Escape backslashes for Windows paths

try:
    # Try to read the file using standard Python
    print(f"Attempting to read file: {{file_path}}")
    
    with open(file_path, 'r') as f:
        content = f.read()
        print(f"File content:\\n{{content}}")
    
    # Try to write to the file
    with open(file_path, 'a') as f:
        f.write("\\nAppended by Python sandbox!")
    
    print("File updated successfully")
    result = "Filesystem operations completed"
except Exception as e:
    print(f"Filesystem error: {{e}}")
    result = f"Filesystem operations failed: {{str(e)}}"
"""

        console.print("[cyan]Executing code with filesystem access denied (default)...[/]")
        
        with console.status("[bold cyan]Running code with filesystem access denied...", spinner="dots"):
            denied_result = await python_tool.execute_python(
                code=fs_code,
                timeout_ms=10000
            )
        
        display_result("Filesystem Access Denied (Default)", denied_result, fs_code)
        
        # Now try with mcpfs module
        mcpfs_code = """
try:
    # Try using the mcpfs module (only available when allow_fs=True)
    import mcpfs
    
    # Create a test file
    test_file = "/tmp/python_sandbox_test.txt"
    mcpfs.write_text(test_file, "Hello from mcpfs!\nThis is a secure filesystem access test.")
    print(f"Successfully wrote to {test_file}")
    
    # Read it back
    content = mcpfs.read_text(test_file)
    print(f"File content from mcpfs.read_text():\\n{content}")
    
    # List directory
    dir_listing = mcpfs.listdir("/tmp")
    print(f"Directory listing of /tmp: {dir_listing}")
    
    result = "Secure filesystem operations completed via mcpfs"
except ImportError as e:
    print(f"mcpfs module not available: {e}")
    result = "mcpfs module not available (expected when allow_fs=False)"
except Exception as e:
    print(f"Secure filesystem error: {e}")
    result = f"Secure filesystem operations failed: {str(e)}"
"""

        console.print("[cyan]Now trying with secure filesystem (mcpfs) access denied...[/]")
        
        with console.status("[bold cyan]Running code with mcpfs access denied...", spinner="dots"):
            mcpfs_denied_result = await python_tool.execute_python(
                code=mcpfs_code,
                timeout_ms=10000
            )
        
        display_result("Secure Filesystem (mcpfs) Access Denied", mcpfs_denied_result, mcpfs_code)
        
        console.print("[cyan]Now trying with secure filesystem (mcpfs) access allowed...[/]")
        
        with console.status("[bold cyan]Running code with mcpfs access allowed...", spinner="dots"):
            mcpfs_allowed_result = await python_tool.execute_python(
                code=mcpfs_code,
                allow_fs=True,  # Now allow filesystem
                timeout_ms=10000
            )
        
        display_result("Secure Filesystem (mcpfs) Access Allowed", mcpfs_allowed_result, mcpfs_code)
        
    except Exception as e:
        logger.error(f"Filesystem access demo failed: {e}")
        console.print(f"[bold red]:x: Filesystem Access Demo Failed:[/]\n{escape(str(e))}")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except OSError:
            pass
    
    console.print()  # Spacing

async def data_visualization_demo(python_tool: PythonSandboxTool) -> None:
    """Demonstrate data visualization capabilities."""
    console.print(Rule("[bold green]8. Data Visualization Demo[/bold green]", style="green"))
    logger.info("Starting data visualization demo")
    
    # Matplotlib code
    matplotlib_code = """
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'b-', label='sin(x)')
plt.plot(x, y2, 'r--', label='cos(x)')
plt.title('Sine and Cosine Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Save plot to base64 for display
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
img_str = base64.b64encode(buffer.read()).decode('utf-8')

# Print the base64 string (could be used to display image in HTML)
print("Generated plot as base64 string (first 100 chars):")
print(img_str[:100] + "...")

# Also display data summary
print("\\nData summary:")
print(f"x range: {x.min()} to {x.max()}")
print(f"sin(x) range: {y1.min():.4f} to {y1.max():.4f}")
print(f"cos(x) range: {y2.min():.4f} to {y2.max():.4f}")

# Return the full base64 string as result
result = f"data:image/png;base64,{img_str}"
"""

    console.print("[cyan]Executing code with matplotlib visualization...[/]")
    
    try:
        with console.status("[bold cyan]Running data visualization code...", spinner="dots"):
            viz_result = await python_tool.execute_python(
                code=matplotlib_code,
                packages=["numpy", "matplotlib"],
                timeout_ms=15000
            )
        
        # Special handling for base64 image result
        result_copy = viz_result.copy()
        if "result" in result_copy and isinstance(result_copy["result"], str) and result_copy["result"].startswith("data:image/png;base64,"):
            result_copy["result"] = f"[Base64 image data - {len(result_copy['result'])} chars]"
        
        display_result("Matplotlib Visualization", result_copy, matplotlib_code)
        
        # Display a note about image handling
        console.print(Panel(
            "[yellow]Note:[/] In a real application, the base64 image data returned as the result "
            "could be embedded in HTML for display. For this demo, we're just showing that "
            "matplotlib works in the sandbox and can generate visualizations.",
            border_style="yellow",
            padding=(1, 2)
        ))
        
    except Exception as e:
        logger.error(f"Data visualization demo failed: {e}")
        console.print(f"[bold red]:x: Data Visualization Demo Failed:[/]\n{escape(str(e))}")
    
    console.print()  # Spacing

async def main() -> int:
    """Run the Python Sandbox tools demo."""
    console.print(Rule("[bold magenta]Python Sandbox Tools Demo[/bold magenta]"))
    
    exit_code = 0
    
    try:
        # Create the Gateway and PythonSandboxTool
        gateway = Gateway("python-sandbox-demo", register_tools=False)
        python_tool = PythonSandboxTool(gateway)
        
        # Run the demonstrations
        await basic_execution_demo(python_tool)
        await package_loading_demo(python_tool)
        await repl_mode_demo(python_tool)
        await error_handling_demo(python_tool)
        await timeout_demo(python_tool)
        await network_access_demo(python_tool)
        await filesystem_access_demo(python_tool)
        await data_visualization_demo(python_tool)
        
        # Summary
        console.print(Rule("[bold green]Demo Completed Successfully[/bold green]", style="green"))
        console.print("[green]All Python Sandbox demonstrations completed successfully.[/]")
        
    except Exception as e:
        logger.critical(f"Demo failed with unexpected error: {e}")
        console.print(f"[bold red]CRITICAL ERROR: {escape(str(e))}[/]")
        exit_code = 1
    
    return exit_code

if __name__ == "__main__":
    # Setup logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)