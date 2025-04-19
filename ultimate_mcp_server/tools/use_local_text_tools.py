import asyncio
import inspect
import re
import shlex
import shutil  # Added import
from typing import Any, Dict, List, Optional, Union

from ultimate_mcp_server.exceptions import ToolExecutionError, ToolInputError
from ultimate_mcp_server.tools.base import BaseTool, tool, with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.local_text")


class LocalTextTools(BaseTool):
    """Provides tools for executing local command-line text processing utilities.

    This class enables safe, controlled access to powerful command-line text processing
    utilities like `ripgrep` (rg), `awk`, `sed`, and `jq` from within the Ultimate MCP Server.
    It creates a bridge between high-level application logic and low-level text processing
    capabilities, with robust security validation and error handling.

    Key Features:
    - Thread-safe, async-compatible execution of text processing utilities
    - Comprehensive security validation to prevent command injection and dangerous operations
    - Flexible input handling (string data, files, or directories)
    - Consistent error reporting and timeout management
    - Detailed output with stdout, stderr, and exit code information

    Supported Tools:
    - ripgrep (rg): Fast, recursive text pattern searching with regex support
    - awk: Pattern scanning and text processing language for structured data
    - sed: Stream editor for filtering and transforming text
    - jq: Lightweight, flexible command-line JSON processor

    Input Handling:
    - Each tool accepts input either as a raw string (`input_data`) passed via stdin
      or by targeting a file (`input_file`) or directory (`input_dir`).
    - **Crucially**, when using `input_file` or `input_dir`, the corresponding path
      *must* be explicitly included as part of the command arguments string (`args_str`).
      The tool methods themselves do not automatically append these paths to the command.

    Security Model:
    The class implements a strict security validation system that:
    - Prevents path traversal attacks by rejecting absolute paths and '..' sequences
    - Blocks command injection by filtering shell metacharacters and command substitution
    - Disables destructive operations (e.g., sed's in-place editing capability)
    - Enforces proper argument structure and usage patterns

    Security Note:
    Executing arbitrary command-line arguments provided by an LLM carries inherent
    security risks. Ensure the environment where this gateway runs is properly sandboxed
    or has appropriate security measures in place to prevent unintended system modifications
    or data exposure, especially if tools like `sed -i` (in-place edit) are allowed.

    Usage Pattern:
    ```python
    # Example: Use ripgrep to search for a pattern in a file
    text_tools = LocalTextTools()
    result = await text_tools.run_ripgrep(
        args_str="'error' --json -C 2 logs/app.log",  # Must include file path in args
        input_file=True,  # Flag indicating we're targeting a file (path in args_str)
        timeout=10.0
    )

    # Example: Process JSON data with jq
    data = '{"name": "Alice", "age": 30, "roles": ["admin", "user"]}'
    result = await text_tools.run_jq(
        args_str=".roles[]",  # Extract array elements
        input_data=data,      # Pass JSON string as input
        timeout=5.0
    )
    ```
    """

    COMMANDS = ["rg", "awk", "sed", "jq"]

    def __init__(self, mcp: Optional[Any] = None):
        super().__init__(mcp)
        self._check_command_availability()

    def _check_command_availability(self):
        """Checks if required commands are available in PATH and logs warnings."""
        for cmd in self.COMMANDS:
            if shutil.which(cmd) is None:
                logger.warning(
                    f"Command '{cmd}' not found in PATH. '{self.__class__.__name__}' tool '{cmd}' will fail if called.",
                    emoji_key="warning"
                )
            else:
                 logger.debug(f"Command '{cmd}' found in PATH.")

    def _validate_tool_arguments(self, cmd_name: str, args_list: List[str], is_file: bool = False, is_dir: bool = False) -> None:
        """Validates command arguments for security, preventing dangerous operations and command injection.
        
        This method implements critical security validation for all command-line text tools, enforcing
        restrictions on command arguments to prevent various attack vectors including path traversal,
        command injection, and potentially destructive operations.
        
        Security checks include:
        1. Path safety validation
           - Rejects absolute paths (starting with '/')
           - Blocks path traversal attempts containing '..'
           - Prevents access outside the workspace directory
        
        2. Command injection prevention
           - Blocks shell metacharacters used for command chaining (;, |, &, etc.)
           - Prevents command substitution via backticks (`) or $() syntax
           - Rejects attempts to use redirection operators (>, >>, <)
        
        3. Tool-specific safety restrictions
           - For 'sed': Blocks the '-i' flag that enables destructive in-place file modification
           - For file operations: Ensures that target paths are properly relative and sandboxed
        
        Args:
            cmd_name: The command being validated (e.g., "rg", "awk", "sed")
            args_list: List of command arguments to validate
            is_file: Flag indicating whether args are operating on a file
            is_dir: Flag indicating whether args are operating on a directory
            
        Raises:
            ToolInputError: When dangerous or insecure arguments are detected, with detailed 
                           error messages indicating the specific security violation
        
        Note:
            This method is deliberately strict in its validation to prioritize security over
            convenience. Some legitimate advanced uses might be blocked, but this is considered
            an acceptable trade-off for preventing potential security issues.
        """
        # Block absolute paths and path traversal attempts 
        for arg in args_list:
            # Skip options (arguments starting with -)
            if arg.startswith('-'):
                continue
                
            # Check for common command injection patterns
            if any(char in arg for char in ['`', '$(']):
                raise ToolInputError(
                    f"Command substitution characters detected in argument: '{arg}'", 
                    param_name="args_str", 
                    provided_value=arg
                )
            
            # Check for shell metacharacters that could enable command chaining
            if any(re.search(fr'[^\\]{re.escape(char)}', arg) for char in [';', '|', '&', '>', '<']):
                raise ToolInputError(
                    f"Shell metacharacters detected in argument: '{arg}'", 
                    param_name="args_str", 
                    provided_value=arg
                )
                
            # Path safety checks - relevant for file targets which could be in any argument position
            # Block absolute paths
            if arg.startswith('/'):
                raise ToolInputError(
                    f"Absolute paths are not allowed: '{arg}'", 
                    param_name="args_str", 
                    provided_value=arg
                )
                
            # Block path traversal attempts
            if '..' in arg.split('/'):
                raise ToolInputError(
                    f"Path traversal detected (contains '..'): '{arg}'", 
                    param_name="args_str", 
                    provided_value=arg
                )
        
        # Command-specific security checks
        if cmd_name == 'sed':
            # Block 'sed -i' (in-place editing) which can modify files
            for _i, arg in enumerate(args_list):
                if arg in ['-i', '--in-place'] or arg.startswith('-i') or arg.startswith('--in-place='):
                    raise ToolInputError(
                        f"In-place file editing is not allowed with sed. Remove '{arg}'.",
                        param_name="args_str",
                        provided_value=arg
                    )

        # Additional checks can be added here for other commands
        # if cmd_name == 'some_other_cmd': ...
        
        # Validate at least one path-like argument is provided when operating on files/dirs
        if (is_file or is_dir) and not any(not arg.startswith('-') for arg in args_list):
            raise ToolInputError(
                f"A path target must be included in arguments when using {'file' if is_file else 'directory'} input mode.",
                param_name="args_str"
            )

    async def _run_command(
        self, 
        cmd_name: str, 
        args_str: str, 
        input_data: Optional[str] = None, 
        input_file: Optional[str] = None, 
        input_dir: Optional[str] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Executes a local command-line text utility with robust input handling and security validation.
        
        This core method powers all text utility tools, providing a consistent interface for executing
        local commands like ripgrep, awk, sed, and jq with proper input redirection, argument parsing,
        timeout handling, and comprehensive error reporting.
        
        Key features:
        - Flexible input: Accepts data via stdin, file path, or directory path
        - Command argument validation: Security checks to prevent command injection and dangerous operations
        - Proper subprocess management: Async execution with timeout handling
        - Detailed result reporting: Returns stdout, stderr, exit code, and success status
        
        The method follows this execution process:
        1. Validates the command's availability in the system PATH
        2. Parses and validates input arguments for security concerns
        3. Constructs and executes the command as an async subprocess
        4. Handles timeout with graceful process termination
        5. Collects and formats results, including rich error information
        
        Security safeguards include:
        - Rejection of absolute paths and path traversal attempts
        - Blocking of shell metacharacters that could enable command chaining
        - Prevention of command substitution characters
        - Explicit prohibition of dangerous operations (e.g., sed's in-place edit flag)
        
        Args:
            cmd_name: The base command to execute (e.g., "rg", "awk", "sed", "jq")
            args_str: String containing all command arguments, properly quoted
            input_data: Text string to pass to the command via stdin
            input_file: Flag indicating args_str targets a file (path must be in args_str)
            input_dir: Flag indicating args_str targets a directory (path must be in args_str)
            timeout: Maximum execution time in seconds before terminating the process
            
        Returns:
            Dictionary containing:
            - stdout: The command's standard output as a string
            - stderr: The command's standard error output as a string  
            - exit_code: Integer exit code from the command
            - success: Boolean indicating if the command completed successfully (exit_code == 0)
            - error: Error message string if success is False, otherwise None
            
        Raises:
            ToolInputError: For invalid or insecure input parameters
            ToolExecutionError: For command not found, timeout, or other execution failures
            
        Important:
            When using input_file or input_dir, the file/directory path MUST be included
            in the args_str parameter. This method does not automatically append the path.
        """
        
        # Check for command existence (already logged warning in __init__)
        if shutil.which(cmd_name) is None:
             raise ToolExecutionError(f"Command '{cmd_name}' not found. Ensure it is installed and in the system PATH.")

        if sum(p is not None for p in [input_data, input_file, input_dir]) != 1:
            raise ToolInputError("Exactly one of 'input_data', 'input_file', or 'input_dir' must be provided.")

        # Argument String Parsing (Do this first)
        try:
            args_list = shlex.split(args_str)
        except ValueError as e:
            raise ToolInputError(f"Invalid arguments string (check quoting/escaping): {e}", param_name="args_str", provided_value=args_str) from e

        # --- Input Validation --- 
        try:
            self._validate_tool_arguments(cmd_name, args_list, input_file is not None, input_dir is not None)
        except ToolInputError:
             raise # Re-raise validation errors
        # --- End Input Validation --- 

        full_cmd_list = [cmd_name] + args_list
        
        # Determine the target description for logging
        target_desc = "stdin"
        if input_file:
             target_desc = f"file '{input_file}'"
        elif input_dir:
             target_desc = f"directory '{input_dir}'"

        stdin_content = None
        if input_data is not None:
            stdin_content = input_data.encode() # Pass data via stdin
            logger.debug(f"Running command '{' '.join(full_cmd_list)}' with string input via {target_desc}.")
        else: # input_file or input_dir is not None
             # Path should be within args_str - rely on sandboxing for path safety within args_str
             logger.debug(f"Running command '{' '.join(full_cmd_list)}' targeting path included in args (input source: {target_desc}).")


        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *full_cmd_list,
                stdin=asyncio.subprocess.PIPE if stdin_content else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=stdin_content), 
                timeout=timeout
            )
            
            exit_code = process.returncode
            # Decode stdout/stderr safely, replacing errors
            stdout = stdout_bytes.decode('utf-8', errors='replace') 
            stderr = stderr_bytes.decode('utf-8', errors='replace')

            # Treat exit code 0 as success. Note that some tools might use other codes
            # for non-error conditions (e.g., rg exit code 1 means matches found).
            success = exit_code == 0
            error_message = None

            if not success:
                # Include exit code and snippet of stderr in the error message
                stderr_snippet = stderr[:200].strip() # First 200 chars of stderr
                error_message = f"Command failed with exit code {exit_code}."
                if stderr_snippet:
                     error_message += f" Stderr: '{stderr_snippet}...'"
                logger.warning(f"Command '{cmd_name}' failed with exit code {exit_code}. Stderr: {stderr[:500]}...")
            else:
                 logger.info(f"Command '{cmd_name}' executed successfully (exit code {exit_code}).")

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "success": success, # True only if exit code is 0
                "error": error_message # Contains message only if success is False
            }

        except FileNotFoundError as e: # Should be caught by shutil.which earlier, but keep as safeguard
            logger.error(f"Command not found during execution: {cmd_name}")
            raise ToolExecutionError(f"Command '{cmd_name}' not found. Ensure it is installed and in the system PATH.") from e
        except asyncio.TimeoutError as e:
            logger.error(f"Command '{cmd_name}' timed out after {timeout} seconds.")
            if process:
                try:
                    process.terminate()
                    await asyncio.sleep(0.1) # Give a brief moment for termination
                    if process.returncode is None: # Still running?
                        process.kill()
                    await process.wait() # Ensure process is cleaned up
                except ProcessLookupError:
                    pass # Process already terminated
                except Exception as term_exc:
                     logger.error(f"Error terminating timed-out process: {term_exc}")
            raise ToolExecutionError(f"Command '{cmd_name}' timed out after {timeout} seconds.") from e
        except Exception as e:
            logger.error(f"Error executing command '{cmd_name}': {e}", exc_info=True)
            # Improve error message slightly
            raise ToolExecutionError(f"Unexpected error executing command '{cmd_name}': {type(e).__name__}: {str(e)}") from e


    @tool(name="run_ripgrep", description="Run the ripgrep (rg) command to search text patterns in files or directories.")
    @with_tool_metrics
    @with_error_handling
    async def run_ripgrep(
        self,
        args_str: str,
        input_data: Optional[str] = None,
        input_file: Optional[str] = None,
        input_dir: Optional[str] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Executes the 'rg' (ripgrep) command for fast, recursive text pattern searching.

        When to Use:
        Use this tool to efficiently search for regular expressions or fixed strings within 
        large text inputs, specific files, or entire directory trees. It's significantly
        faster than traditional `grep` for many use cases, especially in codebases.
        Ideal for finding specific log entries, code snippets, or occurrences of patterns.

        Input Handling:
        - `input_data`: Provide text directly via stdin. Do *not* include a path in `args_str`.
        - `input_file`/`input_dir`: Indicate the target type. The actual file/directory path 
          *must* be part of the `args_str` (e.g., `args_str="'pattern' path/to/search"`).

        Common `rg` Arguments (include in `args_str`):
          `'pattern'`: The regex pattern or fixed string to search for.
          `path`: File or directory path to search.
          `-i`, `--ignore-case`: Perform case-insensitive matching.
          `-v`, `--invert-match`: Select non-matching lines.
          `-l`, `--files-with-matches`: Print only the names of files containing matches.
          `-c`, `--count`: Print only the count of matches per file.
          `-A NUM`, `--after-context NUM`: Show NUM lines of context after a match.
          `-B NUM`, `--before-context NUM`: Show NUM lines of context before a match.
          `-C NUM`, `--context NUM`: Show NUM lines of context around a match.
          `--json`: Output results in a line-delimited JSON format.
          `-t type`: Search only files of the specified type (e.g., `py`, `js`, `md`).
          `-g glob`: Include/exclude files/directories matching the glob pattern (e.g., `-g '*.py'`).

        Exit Codes & Success:
        - `0`: Matches were found.
        - `1`: No matches were found.
        - `2`: An error occurred (e.g., invalid arguments, inaccessible file).
        The `success` field in the return dictionary is *only* True if the exit code is 0.
        An exit code of 1 (no matches) will result in `success: false` and an error message.

        Args:
            args_str (str): A string containing the full command-line arguments for `rg`, 
                            including the pattern, flags, and the target path if using 
                            `input_file` or `input_dir`. 
                            Example: `"--json 'error|warn' -C 1 logs/app.log"`
            input_data (Optional[str]): String data to pipe into `rg`'s stdin. Use this 
                                      *instead* of `input_file`/`input_dir`.
            input_file (Optional[str]): Flag indicating `args_str` targets a single file. Path must be in `args_str`.
            input_dir (Optional[str]): Flag indicating `args_str` targets a directory. Path must be in `args_str`.
            timeout (float): Maximum execution time in seconds. Defaults to 30.0.

        Returns:
            Dict[str, Any]: A dictionary containing the execution results:
            {
                "stdout": (str) The standard output from the `rg` command.
                "stderr": (str) The standard error output. Contains errors or warnings.
                "exit_code": (int) The command's exit code (0=matches, 1=no matches, 2=error).
                "success": (bool) True if `exit_code` is 0, False otherwise.
                "error": (Optional[str]) An error message if `exit_code` is non-zero, 
                         including the exit code and a snippet of stderr if available.
            }

        Raises:
            ToolInputError: If exactly one input type (`input_data`, `input_file`, `input_dir`) 
                          is not provided, or if `args_str` is invalidly formatted.
            ToolExecutionError: If the `rg` command is not found, times out, or encounters 
                              an unexpected execution error.
        """
        return await self._run_command("rg", args_str, input_data, input_file, input_dir, timeout)

    @tool(name="run_awk", description="Run the awk command for pattern scanning and processing language.")
    @with_tool_metrics
    @with_error_handling
    async def run_awk(
        self,
        args_str: str,
        input_data: Optional[str] = None,
        input_file: Optional[str] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Executes the 'awk' command for processing text based on patterns and actions.

        When to Use:
        Use `awk` for field-based text processing. It excels at manipulating columns 
        in structured text files (like CSVs or space-delimited logs), calculating sums 
        or averages across lines, and reformatting text based on specific patterns.

        Input Handling:
        - `input_data`: Provide text directly via stdin. Do *not* include a filename in `args_str`.
        - `input_file`: Indicate the target type. The actual input filename(s) 
          *must* be part of the `args_str` (e.g., `args_str="'program' input.txt"`).

        Common `awk` Arguments (include in `args_str`):
          `'program'`: The AWK script, enclosed in single quotes (e.g., '{ print $1, $3 }'). 
                       This is typically the first argument.
          `filename(s)`: One or more input filenames to process.
          `-F fs`: Define the input field separator (e.g., `-F ','` for CSV).
          `-v var=value`: Assign a variable before the script begins execution.

        Args:
            args_str (str): A string containing the full `awk` script and arguments, including 
                            the script itself and the target filename(s) if using `input_file`.
                            Example: `"-F ':' '{ count[$1]++ } END { for (c in count) print c, count[c] }' /etc/passwd"`
            input_data (Optional[str]): String data to pipe into `awk`'s stdin. Use this 
                                      *instead* of `input_file`.
            input_file (Optional[str]): Flag indicating `args_str` targets one or more files. Path(s) must be in `args_str`.
            timeout (float): Maximum execution time in seconds. Defaults to 30.0.

        Returns:
            Dict[str, Any]: A dictionary containing the execution results:
            {
                "stdout": (str) The standard output from the `awk` command.
                "stderr": (str) The standard error output. Contains errors or warnings.
                "exit_code": (int) The command's exit code (typically 0 for success).
                "success": (bool) True if `exit_code` is 0, False otherwise.
                "error": (Optional[str]) An error message if `exit_code` is non-zero, 
                         including the exit code and a snippet of stderr if available.
            }

        Raises:
            ToolInputError: If exactly one input type (`input_data`, `input_file`) 
                          is not provided, or if `args_str` is invalidly formatted.
            ToolExecutionError: If the `awk` command is not found, times out, or encounters 
                              an unexpected execution error.
        """
        # input_dir is less common for awk directly, usually handled by shell piping (find ... | awk)
        return await self._run_command("awk", args_str, input_data, input_file, None, timeout) 

    @tool(name="run_sed", description="Run the sed command for stream editing text.")
    @with_tool_metrics
    @with_error_handling
    async def run_sed(
        self,
        args_str: str,
        input_data: Optional[str] = None,
        input_file: Optional[str] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Executes the 'sed' (Stream Editor) command for text transformations.

        When to Use:
        Use `sed` for performing line-by-line editing tasks on text streams or files, 
        such as substitutions (`s/old/new/`), deletions (`/pattern/d`), insertions, 
        or printing specific lines (`-n 'address'p`). It's powerful for targeted 
        modifications based on patterns or line numbers.

        Input Handling:
        - `input_data`: Provide text directly via stdin. Do *not* include a filename in `args_str`.
        - `input_file`: Indicate the target type. The actual input filename 
          *must* be part of the `args_str` (e.g., `args_str="'script' input.txt"`).

        Common `sed` Arguments (include in `args_str`):
          `'script'`: The `sed` script or command (e.g., 's/foo/bar/g', '/^DEBUG/d'). Often the first argument.
          `filename`: Input filename.
          `-e script`: Add a script to the commands to be executed (allows multiple scripts).
          `-f script-file`: Read `sed` commands from a file.
          `-i[SUFFIX]`: Edit files in-place. **Use with extreme caution!** Optionally creates a backup if SUFFIX is provided.
          `-n`: Suppress automatic printing of each line; only print lines explicitly requested (e.g., with the `p` command).
          `-E` or `-r`: Use extended regular expressions (syntax varies slightly by `sed` version).

        Args:
            args_str (str): A string containing the full `sed` script and arguments, including 
                            the script/command and the target filename if using `input_file`.
                            Example: `"-n -e '/error/Ip' -e '/warn/Ip' app.log"` (I flag for case-insensitive)
            input_data (Optional[str]): String data to pipe into `sed`'s stdin. Use this 
                                      *instead* of `input_file`.
            input_file (Optional[str]): Flag indicating `args_str` targets a file. Path must be in `args_str`.
            timeout (float): Maximum execution time in seconds. Defaults to 30.0.

        Returns:
            Dict[str, Any]: A dictionary containing the execution results:
            {
                "stdout": (str) The standard output from the `sed` command.
                "stderr": (str) The standard error output. Contains errors or warnings.
                "exit_code": (int) The command's exit code (typically 0 for success).
                "success": (bool) True if `exit_code` is 0, False otherwise.
                "error": (Optional[str]) An error message if `exit_code` is non-zero, 
                         including the exit code and a snippet of stderr if available.
            }

        Raises:
            ToolInputError: If exactly one input type (`input_data`, `input_file`) 
                          is not provided, or if `args_str` is invalidly formatted.
            ToolExecutionError: If the `sed` command is not found, times out, or encounters 
                              an unexpected execution error.
        """
        # input_dir is less common for sed directly
        return await self._run_command("sed", args_str, input_data, input_file, None, timeout)

    @tool(name="run_jq", description="Run the jq command for processing JSON data.")
    @with_tool_metrics
    @with_error_handling
    async def run_jq(
        self,
        args_str: str,
        input_data: Optional[str] = None,
        input_file: Optional[str] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Executes the 'jq' command for querying, filtering, and transforming JSON data.

        When to Use:
        Use `jq` whenever you need to extract specific values, filter arrays/objects, 
        or restructure JSON data provided either as a string or within a file. It's essential
        for working with JSON outputs from APIs or other tools.

        Input Handling:
        - `input_data`: Provide JSON text directly via stdin. Do *not* include a filename in `args_str`.
        - `input_file`: Indicate the target type. The actual input JSON filename 
          *must* be part of the `args_str` (e.g., `args_str="'filter' data.json"`). Input *must* be valid JSON.

        Common `jq` Arguments (include in `args_str`):
          `'filter'`: The `jq` filter expression (e.g., '.users[] | select(.active==true) | .name'). 
                      Often the first argument.
          `filename`: Input JSON filename.
          `-c`: Compact output (JSON objects on a single line).
          `-r`: Raw output (strips quotes from string results).
          `-s`: Slurp mode (reads the entire input stream into a single JSON array).
          `--arg name value`: Define a string variable accessible within the filter.
          `--argjson name json_value`: Define a JSON variable accessible within the filter.
          
        Common Filter Patterns:
          `.property`: Access a property value (e.g., `.name`)
          `.property.subproperty`: Access nested properties (e.g., `.user.address.city`)
          `.items[]`: Iterate over array elements (e.g., `.users[]`)
          `select(condition)`: Filter elements based on a condition (e.g., `select(.age > 30)`)
          `map(...)`: Transform each element in an array (e.g., `map(.name)`)
          `{new_key: .old_key}`: Reshape/restructure objects
          `if ... then ... else ... end`: Conditional logic
          
        Performance Considerations:
        - For very large JSON files, consider using selective filters that limit output early
        - The `-c` flag produces more compact output, useful for large result sets
        - Memory usage can be high with deeply nested JSON or large arrays
        - For processing multiple large files, consider separate jq calls instead of slurp mode

        Example Filter Patterns:
        ```
        # Extract specific fields from array elements
        '.items[] | {id: .id, name: .name}'
        
        # Filter array items by condition
        '.users[] | select(.active==true and .age > 25)'
        
        # Count items matching a condition
        '.[] | select(.status=="completed") | length'
        
        # Group and aggregate data
        'group_by(.category) | map({category: .[0].category, count: length})'
        
        # Complex transformation with conditionals
        '.items | map(if .price > 100 then {name, price, category: "premium"} else {name, price, category: "standard"} end)'
        ```

        Args:
            args_str (str): A string containing the full `jq` filter and arguments, including 
                            the filter and the target filename if using `input_file`.
                            Example: `"-r '.items[] | select(.price > 100) | .name' products.json"`
            input_data (Optional[str]): String containing valid JSON data to pipe into `jq`'s stdin. 
                                      Use this *instead* of `input_file`.
            input_file (Optional[str]): Flag indicating `args_str` targets a file containing JSON. Path must be in `args_str`.
            timeout (float): Maximum execution time in seconds. Defaults to 30.0.

        Returns:
            Dict[str, Any]: A dictionary containing the execution results:
            {
                "stdout": (str) The standard output from the `jq` command (usually JSON or raw strings).
                "stderr": (str) The standard error output. Contains errors (e.g., JSON parsing, filter syntax).
                "exit_code": (int) The command's exit code (typically 0 for success, non-zero for errors).
                "success": (bool) True if `exit_code` is 0, False otherwise.
                "error": (Optional[str]) An error message if `exit_code` is non-zero, 
                         including the exit code and a snippet of stderr if available.
            }

        Raises:
            ToolInputError: If exactly one input type (`input_data`, `input_file`) 
                          is not provided, or if `args_str` is invalidly formatted.
            ToolExecutionError: If the `jq` command is not found, times out, or encounters 
                              an unexpected execution error (including JSON parsing errors if input is invalid).
        """
        # input_dir is less common for jq directly
        return await self._run_command("jq", args_str, input_data, input_file, None, timeout)

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Dynamically generate tool definitions based on decorated methods."""
        tools_list = []
        for name, method in inspect.getmembers(self, predicate=inspect.iscoroutinefunction):
            if hasattr(method, '_tool') and method._tool:
                tool_info = {
                    "name": getattr(method, '_tool_name', name),
                    "description": getattr(method, '_tool_description', inspect.getdoc(method)),
                    "parameters": self._generate_schema_from_method(method) 
                }
                # Add resource info if available
                if hasattr(method, '_resource_type'):
                     tool_info['resource_type'] = method._resource_type
                     tool_info['allow_creation'] = method._allow_creation
                     tool_info['require_existence'] = method._require_existence
                     
                tools_list.append(tool_info)
        return tools_list

    # Need to implement or inherit _generate_schema_from_method if not in BaseTool
    # For now, assuming BaseTool provides it or we add a basic version
    def _generate_schema_from_method(self, method):
        # Placeholder: Implement schema generation based on method signature and docstring
        # This might involve inspecting type hints and parsing docstrings
        sig = inspect.signature(method)
        schema = {"type": "object", "properties": {}, "required": []}
        
        # Simple parsing for args and required status based on defaults
        for name, param in sig.parameters.items():
            if name == 'self':
                continue # Skip self parameter
            
            param_type = "string" # Default assumption
            if param.annotation is not inspect.Parameter.empty:
                # Map Python types to JSON schema types (simplified)
                py_type = param.annotation
                if py_type is str: 
                    param_type = "string"
                elif py_type is int: 
                    param_type = "integer"
                elif py_type is float: 
                    param_type = "number"
                elif py_type is bool: 
                    param_type = "boolean"
                elif py_type is list or getattr(py_type, '__origin__', None) is list: 
                    param_type = "array"
                elif py_type is dict or getattr(py_type, '__origin__', None) is dict: 
                    param_type = "object"
                # Handle Optional[T]
                elif getattr(py_type, '__origin__', None) is Union and type(None) in getattr(py_type, '__args__', []):
                    # Extract the non-None type
                    non_none_type = next(t for t in getattr(py_type, '__args__', []) if t is not type(None))
                    if non_none_type is str: 
                        param_type = "string"
                    elif non_none_type is int: 
                        param_type = "integer"
                    elif non_none_type is float: 
                        param_type = "number"
                    elif non_none_type is bool: 
                        param_type = "boolean"
                    # Add more complex type mappings if needed
                    
            schema["properties"][name] = {"type": param_type}
            # Attempt to get description from docstring (very basic)
            doc = inspect.getdoc(method)
            if doc:
                # Look for ':param name:' or 'Args:\n    name:' patterns
                desc_match = re.search(rf"(?:Args:\s*(?:.*\n)*?\s*{name}:\s*(.*))|(?::param\s+{name}:\s*(.*))", doc)
                if desc_match:
                     description = desc_match.group(1) or desc_match.group(2)
                     if description:
                          schema["properties"][name]["description"] = description.strip()

            if param.default is inspect.Parameter.empty:
                schema["required"].append(name)
                
        return schema 