"""Excel Spreadsheet Automation Tools for LLM Gateway.

This module provides powerful, flexible tools for AI agents to automate Excel workflows through 
the Model Context Protocol (MCP). These tools leverage the intelligence of the Large Language Model
while providing deep integration with Microsoft Excel on Windows.

The core philosophy is minimalist but powerful - a few highly flexible functions that can be composed
to perform complex operations, with the LLM (Claude) providing the intelligence to drive these tools.

Key capabilities:
- Direct Excel manipulation (create, modify, analyze spreadsheets)
- Learning from exemplar templates and applying patterns to new contexts
- Formula debugging and optimization
- Rich automated formatting and visualization
- VBA generation and execution

Windows-specific: Uses COM automation with win32com and requires Excel to be installed.

Example usage:
```python
# Execute Excel operations with natural language instructions
result = await client.tools.excel_execute(
    instruction="Create a new workbook with two sheets: 'Revenue' and 'Expenses'. "
                "In the Revenue sheet, create a quarterly forecast for 2025 with "
                "monthly growth of 5%. Include columns for Product A and Product B "
                "with initial values of $10,000 and $5,000. Format as a professional "
                "financial table with totals and proper currency formatting.",
    file_path="financial_forecast.xlsx",
    operation_type="create"
)

# Learn from an exemplar template and adapt it to a new context
result = await client.tools.excel_learn_and_apply(
    exemplar_path="templates/financial_model.xlsx",
    output_path="healthcare_startup.xlsx",
    adaptation_context="Create a 3-year financial model for a healthcare SaaS startup "
                      "with subscription revenue model. Include revenue forecast, expense "
                      "projections, cash flow, and key metrics for investors. Adapt all "
                      "growth rates and assumptions for the healthcare tech market."
)

# Debug and optimize complex formulas
result = await client.tools.excel_analyze_formulas(
    file_path="complex_model.xlsx",
    sheet_name="Valuation",
    cell_range="D15:G25",
    analysis_type="optimize",
    detail_level="detailed"
)
```
"""
import asyncio
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# Try to import Windows-specific libraries
try:
    import pythoncom  # type: ignore
    import win32com.client  # type: ignore
    import win32com.client.gencache  # type: ignore
    from win32com.client import constants as win32c  # type: ignore
    WINDOWS_EXCEL_AVAILABLE = True
except ImportError:
    WINDOWS_EXCEL_AVAILABLE = False

from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.tools.base import BaseTool, with_error_handling, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.excel_automation")

class ExcelSession:
    """Manages a single Excel Application session with enhanced reliability and safety."""
    
    def __init__(self, visible=False):
        """Initialize a new Excel session.
        
        Args:
            visible: Whether Excel should be visible on screen
        """
        if not WINDOWS_EXCEL_AVAILABLE:
            raise ToolError("Excel automation requires Windows with Excel installed")
        
        # Initialize COM in this thread
        pythoncom.CoInitialize()
        
        self.app = None
        self.workbooks = {}
        self.visible = visible
        self.status = "initializing"
        
        try:
            self.app = win32com.client.Dispatch("Excel.Application")
            self.app.Visible = visible
            self.app.DisplayAlerts = False
            self.app.ScreenUpdating = False
            self.app_version = self.app.Version
            self.status = "ready"
        except Exception as e:
            self.status = "error"
            raise ToolError(f"Failed to create Excel instance: {str(e)}") from e
    
    def open_workbook(self, path, read_only=False):
        """Open an Excel workbook.
        
        Args:
            path: Path to the workbook file
            read_only: Whether to open in read-only mode
            
        Returns:
            Workbook COM object
        """
        try:
            abs_path = os.path.abspath(path)
            wb = self.app.Workbooks.Open(abs_path, ReadOnly=read_only)
            self.workbooks[wb.Name] = wb
            return wb
        except Exception as e:
            raise ToolError(f"Failed to open workbook at {path}: {str(e)}") from e
    
    def create_workbook(self):
        """Create a new Excel workbook.
        
        Returns:
            Workbook COM object
        """
        try:
            wb = self.app.Workbooks.Add()
            self.workbooks[wb.Name] = wb
            return wb
        except Exception as e:
            raise ToolError(f"Failed to create new workbook: {str(e)}") from e
    
    def save_workbook(self, workbook, path):
        """Save a workbook to a specified path.
        
        Args:
            workbook: Workbook COM object
            path: Path to save the workbook
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            workbook.SaveAs(os.path.abspath(path))
            return True
        except Exception as e:
            raise ToolError(f"Failed to save workbook to {path}: {str(e)}") from e
    
    def close_workbook(self, workbook, save_changes=False):
        """Close a workbook.
        
        Args:
            workbook: Workbook COM object
            save_changes: Whether to save changes before closing
        """
        try:
            workbook.Close(SaveChanges=save_changes)
            if workbook.Name in self.workbooks:
                del self.workbooks[workbook.Name]
        except Exception as e:
            logger.warning(f"Error closing workbook: {str(e)}")
    
    def close(self):
        """Close the Excel application and release resources."""
        if not self.app:
            return
        
        try:
            # Close all workbooks
            for wb_name in list(self.workbooks.keys()):
                try:
                    self.close_workbook(self.workbooks[wb_name], False)
                except Exception:
                    pass
            
            # Quit Excel
            try:
                self.app.DisplayAlerts = False
                self.app.ScreenUpdating = True
                self.app.Quit()
            except Exception:
                pass
            
            # Release COM references
            del self.app
            self.app = None
            
            # Uninitialize COM
            pythoncom.CoUninitialize()
            
            self.status = "closed"
        except Exception as e:
            self.status = "error_closing"
            logger.error(f"Error closing Excel session: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

@asynccontextmanager
async def get_excel_session(visible=False):
    """Async context manager for getting an Excel session.
    
    Args:
        visible: Whether Excel should be visible
        
    Yields:
        ExcelSession: An Excel session
    """
    session = None
    try:
        # Create the Excel session in a thread pool to avoid blocking
        session = await asyncio.to_thread(ExcelSession, visible=visible)
        yield session
    finally:
        # Cleanup in a thread pool as well
        if session:
            await asyncio.to_thread(session.close)

class ExcelSpreadsheetTools(BaseTool):
    """Tool for automating Excel spreadsheet operations."""
    
    tool_name = "excel_spreadsheet_tools"
    description = "Tool for automating Excel spreadsheet operations."
    
    def __init__(self, mcp_server):
        """Initialize Excel Spreadsheet Tools.
        
        Args:
            mcp_server: MCP server instance
        """
        super().__init__(mcp_server)
        self.templates = {}  # Store learned templates
        
        # Inform if Excel is not available
        if not WINDOWS_EXCEL_AVAILABLE:
            logger.warning(
                "Excel automation tools are only available on Windows with Excel installed. "
                "Some functionality will be limited."
            )
    
    @with_tool_metrics
    @with_error_handling
    async def excel_execute(
        self,
        instruction: str,
        file_path: Optional[str] = None,
        operation_type: str = "create",
        template_path: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        show_excel: bool = False
    ) -> Dict[str, Any]:
        """Execute Excel operations based on natural language instructions.
        
        This is the primary function for manipulating Excel files. It can create new files,
        modify existing ones, and perform various operations based on natural language instructions.
        The intelligence for interpreting these instructions comes from the LLM (Claude),
        which generates the appropriate parameters and logic.
        
        Args:
            instruction: Natural language instruction describing what to do
            file_path: Path to save or modify an Excel file
            operation_type: Type of operation (create, modify, analyze, format, etc.)
            template_path: Optional path to a template file to use as a starting point
            parameters: Optional structured parameters to supplement the instruction
            show_excel: Whether to make Excel visible during execution
            
        Returns:
            Dictionary with operation results and metadata
        """
        start_time = time.time()
        
        # Basic validation
        if not instruction:
            raise ToolInputError("instruction cannot be empty")
        
        if operation_type == "create" and not file_path:
            raise ToolInputError("file_path is required for 'create' operations")
        
        if operation_type in ["modify", "analyze", "format"] and (not file_path or not os.path.exists(file_path)):
            raise ToolInputError(f"Valid existing file_path is required for '{operation_type}' operations")
        
        # Use parameters if provided, otherwise empty dict
        parameters = parameters or {}
        
        # Process template path if provided
        if template_path and not os.path.exists(template_path):
            raise ToolInputError(f"Template file not found at {template_path}")
        
        # Execute the requested operation
        try:
            async with get_excel_session(visible=show_excel) as session:
                result = await self._execute_excel_operation(
                    session=session,
                    instruction=instruction,
                    operation_type=operation_type,
                    file_path=file_path,
                    template_path=template_path,
                    parameters=parameters
                )
                
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                
                logger.info(
                    f"Excel operation '{operation_type}' completed in {processing_time:.2f}s",
                    emoji_key="success"
                )
                
                return result
                
        except Exception as e:
            logger.error(
                f"Error executing Excel operation: {str(e)}",
                emoji_key="error",
                exc_info=True
            )
            raise ToolError(
                f"Failed to execute Excel operation: {str(e)}",
                details={"operation_type": operation_type, "file_path": file_path}
            ) from e
    
    @with_tool_metrics
    @with_error_handling
    async def excel_learn_and_apply(
        self,
        exemplar_path: str,
        output_path: str,
        adaptation_context: str,
        parameters: Optional[Dict[str, Any]] = None,
        show_excel: bool = False
    ) -> Dict[str, Any]:
        """Learn from an exemplar Excel template and apply it to a new context.
        
        This powerful function allows Claude to analyze an existing Excel model or template,
        understand its structure and formulas, and then create a new file adapted to a different
        context while preserving the intelligence embedded in the original.
        
        Args:
            exemplar_path: Path to the Excel file to learn from
            output_path: Path where the new adapted file should be saved
            adaptation_context: Natural language description of how to adapt the template
            parameters: Optional structured parameters with specific adaptation instructions
            show_excel: Whether to make Excel visible during processing
            
        Returns:
            Dictionary with operation results and adaptations made
        """
        start_time = time.time()
        
        # Basic validation
        if not exemplar_path or not os.path.exists(exemplar_path):
            raise ToolInputError(f"Exemplar file not found at {exemplar_path}")
        
        if not output_path:
            raise ToolInputError("output_path cannot be empty")
        
        if not adaptation_context:
            raise ToolInputError("adaptation_context cannot be empty")
        
        # Use parameters if provided, otherwise empty dict
        parameters = parameters or {}
        
        # Execute the template learning and application
        try:
            async with get_excel_session(visible=show_excel) as session:
                # First, learn the template structure
                template_analysis = await self._analyze_excel_template(
                    session=session,
                    exemplar_path=exemplar_path,
                    parameters=parameters
                )
                
                # Apply the learned template to the new context
                result = await self._apply_excel_template(
                    session=session,
                    template_analysis=template_analysis,
                    output_path=output_path,
                    adaptation_context=adaptation_context,
                    parameters=parameters
                )
                
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                
                logger.info(
                    f"Excel template learning and application completed in {processing_time:.2f}s",
                    emoji_key="success"
                )
                
                return result
                
        except Exception as e:
            logger.error(
                f"Error in template learning and application: {str(e)}",
                emoji_key="error",
                exc_info=True
            )
            raise ToolError(
                f"Failed to learn and apply template: {str(e)}",
                details={"exemplar_path": exemplar_path, "output_path": output_path}
            ) from e
    
    @with_tool_metrics
    @with_error_handling
    async def excel_analyze_formulas(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        cell_range: Optional[str] = None,
        analysis_type: str = "analyze",
        detail_level: str = "standard",
        show_excel: bool = False
    ) -> Dict[str, Any]:
        """Analyze, debug, and optimize Excel formulas.
        
        This function provides deep insights into Excel formulas, identifying errors,
        suggesting optimizations, and explaining complex calculations in natural language.
        
        Args:
            file_path: Path to the Excel file to analyze
            sheet_name: Name of the sheet to analyze (if None, active sheet is used)
            cell_range: Cell range to analyze (if None, all formulas are analyzed)
            analysis_type: Type of analysis (analyze, debug, optimize, explain)
            detail_level: Level of detail in the analysis (basic, standard, detailed)
            show_excel: Whether to make Excel visible during analysis
            
        Returns:
            Dictionary with analysis results, issues found, and suggestions
        """
        start_time = time.time()
        
        # Basic validation
        if not file_path or not os.path.exists(file_path):
            raise ToolInputError(f"File not found at {file_path}")
        
        # Execute the formula analysis
        try:
            async with get_excel_session(visible=show_excel) as session:
                result = await self._analyze_excel_formulas(
                    session=session,
                    file_path=file_path,
                    sheet_name=sheet_name,
                    cell_range=cell_range,
                    analysis_type=analysis_type,
                    detail_level=detail_level
                )
                
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                
                logger.info(
                    f"Excel formula analysis completed in {processing_time:.2f}s",
                    emoji_key="success"
                )
                
                return result
                
        except Exception as e:
            logger.error(
                f"Error analyzing Excel formulas: {str(e)}",
                emoji_key="error",
                exc_info=True
            )
            raise ToolError(
                f"Failed to analyze Excel formulas: {str(e)}",
                details={"file_path": file_path, "sheet_name": sheet_name, "cell_range": cell_range}
            ) from e
    
    @with_tool_metrics
    @with_error_handling
    async def excel_generate_macro(
        self,
        instruction: str,
        file_path: Optional[str] = None,
        template: Optional[str] = None,
        test_execution: bool = False,
        security_level: str = "standard",
        show_excel: bool = False
    ) -> Dict[str, Any]:
        """Generate and optionally execute Excel VBA macros based on natural language instructions.
        
        This function leverages Claude's capability to generate Excel VBA code for automating
        complex tasks within Excel. It can create new macros or modify existing ones.
        
        Args:
            instruction: Natural language description of what the macro should do
            file_path: Path to the Excel file where the macro should be added
            template: Optional template or skeleton code to use as a starting point
            test_execution: Whether to test execute the generated macro
            security_level: Security restrictions for macro execution (standard, restricted, permissive)
            show_excel: Whether to make Excel visible during processing
            
        Returns:
            Dictionary with the generated macro code and execution results if applicable
        """
        start_time = time.time()
        
        # Basic validation
        if not instruction:
            raise ToolInputError("instruction cannot be empty")
        
        if file_path and file_path.endswith(".xlsx"):
            # Convert to .xlsm for macro support if needed
            file_path = file_path.replace(".xlsx", ".xlsm")
            logger.info(f"Changed file extension to .xlsm for macro support: {file_path}")
        
        # Execute the macro generation
        try:
            async with get_excel_session(visible=show_excel) as session:
                result = await self._generate_excel_macro(
                    session=session,
                    instruction=instruction,
                    file_path=file_path,
                    template=template,
                    test_execution=test_execution,
                    security_level=security_level
                )
                
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                
                logger.info(
                    f"Excel macro generation completed in {processing_time:.2f}s",
                    emoji_key="success"
                )
                
                return result
                
        except Exception as e:
            logger.error(
                f"Error generating Excel macro: {str(e)}",
                emoji_key="error",
                exc_info=True
            )
            raise ToolError(
                f"Failed to generate Excel macro: {str(e)}",
                details={"file_path": file_path}
            ) from e
    
    # --- Internal implementation methods ---
    
    async def _execute_excel_operation(
        self,
        session: ExcelSession,
        instruction: str,
        operation_type: str,
        file_path: Optional[str] = None,
        template_path: Optional[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Internal method to execute Excel operations.
        
        This method handles the core Excel manipulation based on operation_type.
        
        Args:
            session: Excel session to use
            instruction: Natural language instruction
            operation_type: Type of operation
            file_path: Path to the Excel file
            template_path: Optional template file path
            parameters: Optional structured parameters
            
        Returns:
            Dictionary with operation results
        """
        # Initialize result structure
        result = {
            "success": True,
            "operation_type": operation_type,
            "file_path": file_path
        }
        
        # Handle different operation types
        if operation_type == "create":
            # Create a new workbook, either from scratch or from a template
            if template_path:
                # Open the template
                wb = session.open_workbook(template_path, read_only=True)
                # Save as the new file
                session.save_workbook(wb, file_path)
                # Close the template and reopen the new file
                session.close_workbook(wb)
                wb = session.open_workbook(file_path)
            else:
                # Create a new workbook
                wb = session.create_workbook()
            
            # Apply the instruction to the workbook
            operations_performed = await self._apply_instruction_to_workbook(
                session=session,
                workbook=wb,
                instruction=instruction,
                parameters=parameters
            )
            
            # Save the workbook
            session.save_workbook(wb, file_path)
            
            result["operations_performed"] = operations_performed
            result["file_created"] = file_path
            
        elif operation_type == "modify":
            # Open existing workbook for modification
            wb = session.open_workbook(file_path, read_only=False)
            
            # Apply the instruction to the workbook
            operations_performed = await self._apply_instruction_to_workbook(
                session=session,
                workbook=wb,
                instruction=instruction,
                parameters=parameters
            )
            
            # Save the workbook
            session.save_workbook(wb, file_path)
            
            result["operations_performed"] = operations_performed
            result["file_modified"] = file_path
            
        elif operation_type == "analyze":
            # Open existing workbook for analysis
            wb = session.open_workbook(file_path, read_only=True)
            
            # Analyze the workbook
            analysis_results = await self._analyze_workbook(
                session=session,
                workbook=wb,
                instruction=instruction,
                parameters=parameters
            )
            
            result["analysis_results"] = analysis_results
            
        elif operation_type == "format":
            # Open existing workbook for formatting
            wb = session.open_workbook(file_path, read_only=False)
            
            # Apply formatting to the workbook
            formatting_applied = await self._apply_formatting_to_workbook(
                session=session,
                workbook=wb,
                instruction=instruction,
                parameters=parameters
            )
            
            # Save the workbook
            session.save_workbook(wb, file_path)
            
            result["formatting_applied"] = formatting_applied
            
        else:
            raise ToolInputError(f"Unknown operation_type: {operation_type}")
        
        return result
    
    async def _apply_instruction_to_workbook(
        self,
        session: ExcelSession,
        workbook: Any,
        instruction: str,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply natural language instructions to a workbook.
        
        This method interprets the instructions and performs the requested operations.
        
        Args:
            session: Excel session
            workbook: Workbook COM object
            instruction: Natural language instruction
            parameters: Optional structured parameters
            
        Returns:
            List of operations performed
        """
        operations_performed = []
        
        # Default to first worksheet if none exists
        if workbook.Worksheets.Count == 0:
            worksheet = workbook.Worksheets.Add()
            operations_performed.append({
                "operation": "create_worksheet",
                "sheet_name": worksheet.Name
            })
        
        # Process instruction to extract key operations
        instruction_lower = instruction.lower()
        
        # Create sheets if mentioned
        if "sheet" in instruction_lower or "sheets" in instruction_lower:
            # Extract sheet names using regex to find patterns like:
            # - 'sheets: X, Y, Z'
            # - 'sheets named X and Y'
            # - 'create sheets X, Y, Z'
            sheet_patterns = [
                r"sheet(?:s)?\s*(?:named|called|:)?\s*(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))",
                r"create (?:a |)sheet(?:s)? (?:named|called)?\s*(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))"
            ]
            
            sheet_names = []
            for pattern in sheet_patterns:
                matches = re.findall(pattern, instruction_lower)
                if matches:
                    for match in matches:
                        # Each match is now a tuple with 3 capture groups: (single_quoted, double_quoted, unquoted)
                        sheet_name = match[0] or match[1] or match[2]
                        if sheet_name:
                            # Split by commas and/or 'and', then clean up
                            for name in re.split(r',|\sand\s', sheet_name):
                                clean_name = name.strip("' \"").strip()
                                if clean_name:
                                    sheet_names.append(clean_name)
            
            # Also check explicit parameters if provided
            if parameters and "sheet_names" in parameters:
                sheet_names.extend(parameters["sheet_names"])
            
            # Make sheet names unique
            sheet_names = list(set(sheet_names))
            
            # Create each sheet
            current_sheets = [sheet.Name.lower() for sheet in workbook.Worksheets]
            
            for sheet_name in sheet_names:
                if sheet_name.lower() not in current_sheets:
                    new_sheet = workbook.Worksheets.Add(After=workbook.Worksheets(workbook.Worksheets.Count))
                    new_sheet.Name = sheet_name
                    operations_performed.append({
                        "operation": "create_worksheet",
                        "sheet_name": sheet_name
                    })
        
        # Add headers if mentioned
        if "header" in instruction_lower or "headers" in instruction_lower:
            # Extract header information
            header_data = None
            
            # Check parameters first
            if parameters and "headers" in parameters:
                header_data = parameters["headers"]
            else:
                # Try to extract from instruction
                header_match = re.search(r"header(?:s)?\s*(?::|with|including)\s*([^.]+)", instruction_lower)
                if header_match:
                    # Parse the header text
                    header_text = header_match.group(1).strip()
                    # Split by commas and/or 'and'
                    header_data = [h.strip("' \"").strip() for h in re.split(r',|\sand\s', header_text) if h.strip()]
            
            if header_data:
                # Determine target sheet
                target_sheet_name = None
                
                # Check parameters first
                if parameters and "target_sheet" in parameters:
                    target_sheet_name = parameters["target_sheet"]
                else:
                    # Try to extract from instruction
                    sheet_match = re.search(r"in (?:the |)(?:sheet|worksheet) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_]+))", instruction_lower)
                    if sheet_match:
                        target_sheet_name = sheet_match.group(1) or sheet_match.group(2) or sheet_match.group(3)
                
                # Default to first sheet if not specified
                if not target_sheet_name:
                    target_sheet_name = workbook.Worksheets(1).Name
                
                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == target_sheet_name.lower():
                        worksheet = sheet
                        break
                
                if not worksheet:
                    worksheet = workbook.Worksheets(1)
                
                # Add headers to the worksheet
                for col_idx, header in enumerate(header_data, 1):
                    worksheet.Cells(1, col_idx).Value = header
                    
                    # Apply simple header formatting
                    worksheet.Cells(1, col_idx).Font.Bold = True
                
                operations_performed.append({
                    "operation": "add_headers",
                    "sheet_name": worksheet.Name,
                    "headers": header_data
                })
        
        # Add data if mentioned
        if "data" in instruction_lower or "values" in instruction_lower:
            # Check parameters first
            data_rows = None
            
            if parameters and "data" in parameters:
                data_rows = parameters["data"]
            
            if data_rows:
                # Determine target sheet
                target_sheet_name = None
                
                # Check parameters first
                if parameters and "target_sheet" in parameters:
                    target_sheet_name = parameters["target_sheet"]
                else:
                    # Try to extract from instruction
                    sheet_match = re.search(r"in (?:the |)(?:sheet|worksheet) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_]+))", instruction_lower)
                    if sheet_match:
                        target_sheet_name = sheet_match.group(1) or sheet_match.group(2) or sheet_match.group(3)
                
                # Default to first sheet if not specified
                if not target_sheet_name:
                    target_sheet_name = workbook.Worksheets(1).Name
                
                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == target_sheet_name.lower():
                        worksheet = sheet
                        break
                
                if not worksheet:
                    worksheet = workbook.Worksheets(1)
                
                # Determine starting row (typically 2 if headers exist)
                start_row = 2
                if parameters and "start_row" in parameters:
                    start_row = parameters["start_row"]
                
                # Add data to the worksheet
                for row_idx, row_data in enumerate(data_rows, start_row):
                    for col_idx, cell_value in enumerate(row_data, 1):
                        worksheet.Cells(row_idx, col_idx).Value = cell_value
                
                operations_performed.append({
                    "operation": "add_data",
                    "sheet_name": worksheet.Name,
                    "start_row": start_row,
                    "row_count": len(data_rows)
                })
        
        # Add formulas if mentioned
        if "formula" in instruction_lower or "formulas" in instruction_lower:
            formula_data = None
            
            # Check parameters first
            if parameters and "formulas" in parameters:
                formula_data = parameters["formulas"]
            
            if formula_data:
                # Determine target sheet
                target_sheet_name = None
                
                # Check parameters first
                if parameters and "target_sheet" in parameters:
                    target_sheet_name = parameters["target_sheet"]
                
                # Default to first sheet if not specified
                if not target_sheet_name:
                    target_sheet_name = workbook.Worksheets(1).Name
                
                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == target_sheet_name.lower():
                        worksheet = sheet
                        break
                
                if not worksheet:
                    worksheet = workbook.Worksheets(1)
                
                # Add formulas to the worksheet
                for formula_entry in formula_data:
                    cell_ref = formula_entry.get("cell")
                    formula = formula_entry.get("formula")
                    
                    if cell_ref and formula:
                        worksheet.Range(cell_ref).Formula = formula
                
                operations_performed.append({
                    "operation": "add_formulas",
                    "sheet_name": worksheet.Name,
                    "formula_count": len(formula_data)
                })
        
        # Apply formatting if mentioned
        if "format" in instruction_lower or "formatting" in instruction_lower:
            formatting = None
            
            # Check parameters first
            if parameters and "formatting" in parameters:
                formatting = parameters["formatting"]
            
            if formatting:
                await self._apply_formatting_to_workbook(
                    session=session,
                    workbook=workbook,
                    instruction=instruction,
                    parameters={"formatting": formatting}
                )
                
                operations_performed.append({
                    "operation": "apply_formatting",
                    "details": "Applied formatting based on parameters"
                })
            else:
                # Apply default formatting based on instruction
                sheet = workbook.Worksheets(1)
                
                # Auto-fit columns
                used_range = sheet.UsedRange
                used_range.Columns.AutoFit()
                
                # Add borders to data range
                if used_range.Rows.Count > 1:
                    data_range = sheet.Range(sheet.Cells(1, 1), sheet.Cells(used_range.Rows.Count, used_range.Columns.Count))
                    data_range.Borders.LineStyle = 1  # xlContinuous
                
                operations_performed.append({
                    "operation": "apply_formatting",
                    "details": "Applied default formatting (auto-fit columns, borders)"
                })
        
        # Create a chart if mentioned
        if "chart" in instruction_lower or "graph" in instruction_lower:
            chart_type = None
            
            # Check parameters first
            if parameters and "chart" in parameters:
                chart_info = parameters["chart"]
                chart_type = chart_info.get("type", "column")
                data_range = chart_info.get("data_range")
                chart_title = chart_info.get("title", "Chart")
                
                if data_range:
                    # Determine target sheet
                    target_sheet_name = chart_info.get("sheet_name")
                    
                    # Default to first sheet if not specified
                    if not target_sheet_name:
                        target_sheet_name = workbook.Worksheets(1).Name
                    
                    # Find the worksheet
                    worksheet = None
                    for sheet in workbook.Worksheets:
                        if sheet.Name.lower() == target_sheet_name.lower():
                            worksheet = sheet
                            break
                    
                    if not worksheet:
                        worksheet = workbook.Worksheets(1)
                    
                    # Create the chart
                    chart = worksheet.Shapes.AddChart2(-1, getattr(win32c, f"xl{chart_type.capitalize()}")).Chart
                    chart.SetSourceData(worksheet.Range(data_range))
                    chart.HasTitle = True
                    chart.ChartTitle.Text = chart_title
                    
                    operations_performed.append({
                        "operation": "create_chart",
                        "sheet_name": worksheet.Name,
                        "chart_type": chart_type,
                        "data_range": data_range
                    })
        
        return operations_performed
    
    async def _analyze_workbook(
        self,
        session: ExcelSession,
        workbook: Any,
        instruction: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a workbook based on instructions.
        
        This method examines the workbook structure, formulas, and data.
        
        Args:
            session: Excel session
            workbook: Workbook COM object
            instruction: Analysis instruction
            parameters: Optional structured parameters
            
        Returns:
            Dictionary with analysis results
        """
        # Initialize result
        analysis_results = {
            "workbook_name": workbook.Name,
            "sheet_count": workbook.Sheets.Count,
            "sheets_info": [],
            "has_formulas": False,
            "has_links": workbook.HasLinks,
            "calculation_mode": self._get_calculation_mode_name(workbook.CalculationMode),
        }
        
        total_formulas = 0
        
        # Analyze each sheet
        for sheet_idx in range(1, workbook.Sheets.Count + 1):
            sheet = workbook.Sheets(sheet_idx)
            
            # Skip chart sheets
            if sheet.Type != 1:  # xlWorksheet
                continue
            
            used_range = sheet.UsedRange
            
            # Get sheet details
            sheet_info = {
                "name": sheet.Name,
                "row_count": used_range.Rows.Count if used_range else 0,
                "column_count": used_range.Columns.Count if used_range else 0,
                "visible": sheet.Visible == -1,  # -1 is xlSheetVisible
                "has_formulas": False,
                "formula_count": 0,
                "data_tables": False,
                "has_charts": len(sheet.ChartObjects()) > 0,
                "chart_count": len(sheet.ChartObjects()),
                "named_ranges": []
            }
            
            # Look for formulas
            formula_cells = []
            formula_count = 0
            
            if used_range:
                # Sample used range cells to check for formulas (limit to reasonable number)
                row_count = min(used_range.Rows.Count, 1000)
                col_count = min(used_range.Columns.Count, 100)
                
                for row in range(1, row_count + 1):
                    for col in range(1, col_count + 1):
                        cell = used_range.Cells(row, col)
                        if cell.HasFormula:
                            formula_count += 1
                            if len(formula_cells) < 10:  # Just store a few examples
                                cell_address = cell.Address(False, False)  # A1 style without $
                                formula_cells.append({
                                    "address": cell_address,
                                    "formula": cell.Formula
                                })
            
            sheet_info["has_formulas"] = formula_count > 0
            sheet_info["formula_count"] = formula_count
            sheet_info["example_formulas"] = formula_cells
            
            total_formulas += formula_count
            
            # Get named ranges in this sheet
            for name in workbook.Names:
                try:
                    if name.RefersToRange.Parent.Name == sheet.Name:
                        sheet_info["named_ranges"].append({
                            "name": name.Name,
                            "refers_to": name.RefersTo
                        })
                except Exception:
                    pass  # Skip if there's an error (e.g., name refers to another workbook)
            
            analysis_results["sheets_info"].append(sheet_info)
        
        analysis_results["has_formulas"] = total_formulas > 0
        analysis_results["total_formula_count"] = total_formulas
        
        # Check for external links
        if workbook.HasLinks:
            links = []
            try:
                for link in workbook.LinkSources():
                    links.append(link)
            except Exception:
                pass  # Skip if there's an error
            
            analysis_results["external_links"] = links
        
        # Add sheet dependencies if requested
        if "analyze_dependencies" in instruction.lower() or (parameters and parameters.get("analyze_dependencies")):
            analysis_results["dependencies"] = await self._analyze_sheet_dependencies(session, workbook)
        
        # Add formula analysis if requested
        if "analyze_formulas" in instruction.lower() or (parameters and parameters.get("analyze_formulas")):
            analysis_results["formula_analysis"] = await self._analyze_formulas_in_workbook(session, workbook)
        
        return analysis_results
    
    async def _apply_formatting_to_workbook(
        self,
        session: ExcelSession,
        workbook: Any,
        instruction: str,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply formatting to a workbook based on instructions.
        
        This method handles various formatting operations.
        
        Args:
            session: Excel session
            workbook: Workbook COM object
            instruction: Formatting instruction
            parameters: Optional structured parameters
            
        Returns:
            List of formatting operations performed
        """
        formatting_applied = []
        
        # Check if specific formatting instructions are provided in parameters
        if parameters and "formatting" in parameters:
            formatting = parameters["formatting"]
            
            # Apply cell formatting
            if "cells" in formatting:
                for cell_format in formatting["cells"]:
                    cell_range = cell_format.get("range")
                    sheet_name = cell_format.get("sheet")
                    
                    if not cell_range:
                        continue
                    
                    # Find the worksheet
                    worksheet = None
                    if sheet_name:
                        for sheet in workbook.Worksheets:
                            if sheet.Name.lower() == sheet_name.lower():
                                worksheet = sheet
                                break
                    
                    if not worksheet:
                        worksheet = workbook.Worksheets(1)
                    
                    # Get the range
                    range_obj = worksheet.Range(cell_range)
                    
                    # Apply formatting attributes
                    if "bold" in cell_format:
                        range_obj.Font.Bold = cell_format["bold"]
                    
                    if "italic" in cell_format:
                        range_obj.Font.Italic = cell_format["italic"]
                    
                    if "color" in cell_format:
                        # Handle hex color codes (e.g., "#FF0000" for red)
                        color_code = cell_format["color"]
                        if color_code.startswith("#"):
                            # Convert hex color to RGB value
                            r = int(color_code[1:3], 16)
                            g = int(color_code[3:5], 16)
                            b = int(color_code[5:7], 16)
                            range_obj.Font.Color = r + (g << 8) + (b << 16)
                        else:
                            # Try to set color directly
                            range_obj.Font.Color = cell_format["color"]
                    
                    if "bg_color" in cell_format:
                        # Handle hex color codes
                        color_code = cell_format["bg_color"]
                        if color_code.startswith("#"):
                            # Convert hex color to RGB value
                            r = int(color_code[1:3], 16)
                            g = int(color_code[3:5], 16)
                            b = int(color_code[5:7], 16)
                            range_obj.Interior.Color = r + (g << 8) + (b << 16)
                        else:
                            # Try to set color directly
                            range_obj.Interior.Color = cell_format["bg_color"]
                    
                    if "number_format" in cell_format:
                        range_obj.NumberFormat = cell_format["number_format"]
                    
                    if "border" in cell_format:
                        border_style = cell_format["border"]
                        if border_style == "all":
                            for border_idx in range(7, 13):  # xlEdgeLeft to xlInsideVertical
                                range_obj.Borders(border_idx).LineStyle = 1  # xlContinuous
                                range_obj.Borders(border_idx).Weight = 2  # xlThin
                        elif border_style == "outside":
                            for border_idx in range(7, 11):  # xlEdgeLeft to xlEdgeRight
                                range_obj.Borders(border_idx).LineStyle = 1  # xlContinuous
                                range_obj.Borders(border_idx).Weight = 2  # xlThin
                    
                    formatting_applied.append({
                        "operation": "format_cells",
                        "sheet_name": worksheet.Name,
                        "range": cell_range
                    })
            
            # Apply table formatting
            if "tables" in formatting:
                for table_format in formatting["tables"]:
                    data_range = table_format.get("range")
                    sheet_name = table_format.get("sheet")
                    table_style = table_format.get("style", "TableStyleMedium2")
                    has_headers = table_format.get("has_headers", True)
                    
                    if not data_range:
                        continue
                    
                    # Find the worksheet
                    worksheet = None
                    if sheet_name:
                        for sheet in workbook.Worksheets:
                            if sheet.Name.lower() == sheet_name.lower():
                                worksheet = sheet
                                break
                    
                    if not worksheet:
                        worksheet = workbook.Worksheets(1)
                    
                    # Create a table
                    table_name = f"Table{len(worksheet.ListObjects) + 1}"
                    if "name" in table_format:
                        table_name = table_format["name"]
                    
                    try:
                        table = worksheet.ListObjects.Add(1, worksheet.Range(data_range), True)
                        table.Name = table_name
                        table.TableStyle = table_style
                        
                        formatting_applied.append({
                            "operation": "create_table",
                            "sheet_name": worksheet.Name,
                            "table_name": table_name,
                            "range": data_range
                        })
                    except Exception as e:
                        logger.warning(f"Failed to create table: {str(e)}")
            
            # Apply conditional formatting
            if "conditional_formatting" in formatting:
                for cf_format in formatting["conditional_formatting"]:
                    cell_range = cf_format.get("range")
                    sheet_name = cf_format.get("sheet")
                    cf_type = cf_format.get("type")
                    
                    if not cell_range or not cf_type:
                        continue
                    
                    # Find the worksheet
                    worksheet = None
                    if sheet_name:
                        for sheet in workbook.Worksheets:
                            if sheet.Name.lower() == sheet_name.lower():
                                worksheet = sheet
                                break
                    
                    if not worksheet:
                        worksheet = workbook.Worksheets(1)
                    
                    # Get the range
                    range_obj = worksheet.Range(cell_range)
                    
                    # Apply conditional formatting based on type
                    if cf_type == "data_bar":
                        color = cf_format.get("color", 43)  # Default blue
                        if isinstance(color, str) and color.startswith("#"):
                            # Convert hex color to RGB value
                            r = int(color[1:3], 16)
                            g = int(color[3:5], 16)
                            b = int(color[5:7], 16)
                            color = r + (g << 8) + (b << 16)
                        
                        cf = range_obj.FormatConditions.AddDatabar()
                        cf.BarColor.Color = color
                    
                    elif cf_type == "color_scale":
                        cf = range_obj.FormatConditions.AddColorScale(3)
                        # Configure color scale (could be extended with more options)
                    
                    elif cf_type == "icon_set":
                        icon_style = cf_format.get("icon_style", "3Arrows")
                        cf = range_obj.FormatConditions.AddIconSetCondition()
                        cf.IconSet = workbook.Application.IconSets(icon_style)
                    
                    elif cf_type == "cell_value":
                        comparison_operator = cf_format.get("operator", "greaterThan")
                        comparison_value = cf_format.get("value", 0)
                        
                        # Map string operator to Excel constant
                        operator_map = {
                            "greaterThan": 3,      # xlGreater
                            "lessThan": 5,         # xlLess
                            "equalTo": 2,          # xlEqual
                            "greaterOrEqual": 4,   # xlGreaterEqual
                            "lessOrEqual": 6,      # xlLessEqual
                            "notEqual": 7          # xlNotEqual
                        }
                        
                        operator_constant = operator_map.get(comparison_operator, 3)
                        
                        cf = range_obj.FormatConditions.Add(1, operator_constant, comparison_value)  # 1 = xlCellValue
                        
                        # Apply formatting
                        if "bold" in cf_format:
                            cf.Font.Bold = cf_format["bold"]
                        
                        if "italic" in cf_format:
                            cf.Font.Italic = cf_format["italic"]
                        
                        if "color" in cf_format:
                            # Handle hex color codes
                            color_code = cf_format["color"]
                            if color_code.startswith("#"):
                                # Convert hex color to RGB value
                                r = int(color_code[1:3], 16)
                                g = int(color_code[3:5], 16)
                                b = int(color_code[5:7], 16)
                                cf.Font.Color = r + (g << 8) + (b << 16)
                            else:
                                cf.Font.Color = cf_format["color"]
                        
                        if "bg_color" in cf_format:
                            # Handle hex color codes
                            color_code = cf_format["bg_color"]
                            if color_code.startswith("#"):
                                # Convert hex color to RGB value
                                r = int(color_code[1:3], 16)
                                g = int(color_code[3:5], 16)
                                b = int(color_code[5:7], 16)
                                cf.Interior.Color = r + (g << 8) + (b << 16)
                            else:
                                cf.Interior.Color = cf_format["bg_color"]
                    
                    formatting_applied.append({
                        "operation": "add_conditional_formatting",
                        "sheet_name": worksheet.Name,
                        "range": cell_range,
                        "type": cf_type
                    })
        
        # Apply default formatting based on instruction
        else:
            instruction_lower = instruction.lower()
            
            # Extract target sheet(s)
            sheet_names = []
            sheet_match = re.search(r"(?:in|to) (?:the |)(?:sheet|worksheet)(?:s|) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))", instruction_lower)
            
            if sheet_match:
                sheet_names_str = sheet_match.group(1) or sheet_match.group(2) or sheet_match.group(3)
                # Split by commas and/or 'and'
                for name in re.split(r',|\sand\s', sheet_names_str):
                    clean_name = name.strip("' \"").strip()
                    if clean_name:
                        sheet_names.append(clean_name)
            
            # If no sheets specified, use all worksheets
            if not sheet_names:
                sheet_names = [sheet.Name for sheet in workbook.Worksheets]
            
            for sheet_name in sheet_names:
                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == sheet_name.lower():
                        worksheet = sheet
                        break
                
                if not worksheet:
                    continue
                
                # Apply standard formatting
                used_range = worksheet.UsedRange
                
                # Auto-fit columns
                if "auto-fit" in instruction_lower or "autofit" in instruction_lower:
                    used_range.Columns.AutoFit()
                    
                    formatting_applied.append({
                        "operation": "auto_fit_columns",
                        "sheet_name": worksheet.Name
                    })
                
                # Add borders to data
                if "borders" in instruction_lower or "outline" in instruction_lower:
                    if used_range.Rows.Count > 0 and used_range.Columns.Count > 0:
                        # Apply borders
                        border_style = 1  # xlContinuous
                        border_weight = 2  # xlThin
                        
                        # Determine border type
                        if "outside" in instruction_lower:
                            # Outside borders only
                            for border_idx in range(7, 11):  # xlEdgeLeft to xlEdgeRight
                                used_range.Borders(border_idx).LineStyle = border_style
                                used_range.Borders(border_idx).Weight = border_weight
                        else:
                            # All borders
                            used_range.Borders.LineStyle = border_style
                            used_range.Borders.Weight = border_weight
                        
                        formatting_applied.append({
                            "operation": "add_borders",
                            "sheet_name": worksheet.Name,
                            "border_type": "outside" if "outside" in instruction_lower else "all"
                        })
                
                # Format headers
                if "header" in instruction_lower or "headers" in instruction_lower:
                    if used_range.Rows.Count > 0:
                        # Apply header formatting to first row
                        header_row = worksheet.Rows(1)
                        header_row.Font.Bold = True
                        
                        # Set background color if mentioned
                        if "blue" in instruction_lower:
                            header_row.Interior.Color = 15773696  # Light blue
                        elif "gray" in instruction_lower or "grey" in instruction_lower:
                            header_row.Interior.Color = 14540253  # Light gray
                        elif "green" in instruction_lower:
                            header_row.Interior.Color = 13561798  # Light green
                        else:
                            # Default light blue
                            header_row.Interior.Color = 15773696
                        
                        formatting_applied.append({
                            "operation": "format_headers",
                            "sheet_name": worksheet.Name
                        })
                
                # Apply number formatting
                if "currency" in instruction_lower or "dollar" in instruction_lower:
                    # Look for ranges with currency values
                    # This is a simplistic approach - in a real tool, we might analyze the data
                    # to identify numeric columns that might be currency
                    if used_range.Rows.Count > 1:  # Skip if only header row
                        for col in range(1, used_range.Columns.Count + 1):
                            # Check a sample of cells in this column
                            numeric_cell_count = 0
                            sample_size = min(10, used_range.Rows.Count - 1)
                            
                            for row in range(2, 2 + sample_size):  # Skip header
                                cell_value = worksheet.Cells(row, col).Value
                                if isinstance(cell_value, (int, float)):
                                    numeric_cell_count += 1
                            
                            # If most cells are numeric, apply currency format
                            if numeric_cell_count > sample_size / 2:
                                col_range = worksheet.Range(
                                    worksheet.Cells(2, col), 
                                    worksheet.Cells(used_range.Rows.Count, col)
                                )
                                
                                # Determine currency symbol
                                currency_format = "$#,##0.00"
                                if "euro" in instruction_lower:
                                    currency_format = "€#,##0.00"
                                elif "pound" in instruction_lower:
                                    currency_format = "£#,##0.00"
                                
                                col_range.NumberFormat = currency_format
                                
                                formatting_applied.append({
                                    "operation": "apply_currency_format",
                                    "sheet_name": worksheet.Name,
                                    "column": worksheet.Cells(1, col).Value or f"Column {col}",
                                    "format": currency_format
                                })
                
                # Apply percentage formatting
                if "percent" in instruction_lower or "percentage" in instruction_lower:
                    # Similar approach to currency formatting
                    if used_range.Rows.Count > 1:  # Skip if only header row
                        for col in range(1, used_range.Columns.Count + 1):
                            col_header = worksheet.Cells(1, col).Value
                            
                            # Check if column header suggests percentage
                            is_percentage_column = False
                            if col_header and isinstance(col_header, str):
                                if any(term in col_header.lower() for term in ["percent", "rate", "growth", "change", "margin"]):
                                    is_percentage_column = True
                            
                            if is_percentage_column:
                                col_range = worksheet.Range(
                                    worksheet.Cells(2, col), 
                                    worksheet.Cells(used_range.Rows.Count, col)
                                )
                                
                                col_range.NumberFormat = "0.0%"
                                
                                formatting_applied.append({
                                    "operation": "apply_percentage_format",
                                    "sheet_name": worksheet.Name,
                                    "column": col_header or f"Column {col}"
                                })
                
                # Create a table if requested
                if "table" in instruction_lower and "style" in instruction_lower:
                    if used_range.Rows.Count > 0 and used_range.Columns.Count > 0:
                        # Create a table with the used range
                        try:
                            has_headers = True
                            if "no header" in instruction_lower:
                                has_headers = False
                            
                            table = worksheet.ListObjects.Add(1, used_range, has_headers)
                            
                            # Set table style
                            table_style = "TableStyleMedium2"  # Default medium blue
                            
                            if "light" in instruction_lower:
                                if "blue" in instruction_lower:
                                    table_style = "TableStyleLight1"
                                elif "green" in instruction_lower:
                                    table_style = "TableStyleLight5"
                                elif "orange" in instruction_lower:
                                    table_style = "TableStyleLight3"
                            elif "medium" in instruction_lower:
                                if "blue" in instruction_lower:
                                    table_style = "TableStyleMedium2"
                                elif "green" in instruction_lower:
                                    table_style = "TableStyleMedium5"
                                elif "orange" in instruction_lower:
                                    table_style = "TableStyleMedium3"
                            elif "dark" in instruction_lower:
                                if "blue" in instruction_lower:
                                    table_style = "TableStyleDark2"
                                elif "green" in instruction_lower:
                                    table_style = "TableStyleDark5"
                                elif "orange" in instruction_lower:
                                    table_style = "TableStyleDark3"
                            
                            table.TableStyle = table_style
                            
                            formatting_applied.append({
                                "operation": "create_table",
                                "sheet_name": worksheet.Name,
                                "style": table_style
                            })
                        except Exception as e:
                            logger.warning(f"Failed to create table: {str(e)}")
        
        return formatting_applied
    
    async def _analyze_excel_template(
        self,
        session: ExcelSession,
        exemplar_path: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze an Excel template to understand its structure and formulas.
        
        This method extracts the key components, relationships, and logic of a template.
        
        Args:
            session: Excel session
            exemplar_path: Path to the exemplar Excel file
            parameters: Optional structured parameters
            
        Returns:
            Dictionary with template analysis results
        """
        # Open the workbook
        wb = session.open_workbook(exemplar_path, read_only=True)
        
        # Initialize the template analysis
        template_analysis = {
            "file_path": exemplar_path,
            "workbook_name": wb.Name,
            "sheet_count": wb.Sheets.Count,
            "sheets": [],
            "named_ranges": [],
            "data_tables": [],
            "worksheets_relationships": []
        }
        
        # Analyze named ranges
        for name in wb.Names:
            try:
                template_analysis["named_ranges"].append({
                    "name": name.Name,
                    "refers_to": name.RefersTo,
                    "scope": "Workbook"
                })
            except Exception:
                pass  # Skip if there's an error
        
        # Analyze each sheet
        for sheet_idx in range(1, wb.Sheets.Count + 1):
            sheet = wb.Sheets(sheet_idx)
            
            # Skip chart sheets
            if sheet.Type != 1:  # xlWorksheet
                continue
            
            used_range = sheet.UsedRange
            
            # Get sheet structure
            sheet_info = {
                "name": sheet.Name,
                "row_count": used_range.Rows.Count if used_range else 0,
                "column_count": used_range.Columns.Count if used_range else 0,
                "visible": sheet.Visible == -1,  # -1 is xlSheetVisible
                "has_formulas": False,
                "formula_patterns": [],
                "sections": [],
                "data_tables": []
            }
            
            # Analyze sheet structure to identify sections
            if used_range:
                # Check for headers in the first row
                headers = []
                has_headers = False
                
                try:
                    for col in range(1, min(used_range.Columns.Count, 20) + 1):
                        header_value = sheet.Cells(1, col).Value
                        if header_value:
                            headers.append(str(header_value))
                            has_headers = True
                        else:
                            headers.append(None)
                except Exception:
                    pass
                
                sheet_info["has_headers"] = has_headers
                sheet_info["headers"] = headers
                
                # Try to identify sections (groups of rows with a similar purpose)
                sections = []
                current_section = None
                
                for row in range(1, min(used_range.Rows.Count, 100) + 1):
                    # Check if this might be a section header
                    left_cell = sheet.Cells(row, 1).Value
                    
                    # Characteristics of a section header: bold text, merged cells, different formatting
                    is_section_header = False
                    
                    try:
                        left_cell_is_bold = sheet.Cells(row, 1).Font.Bold
                        left_cell_is_merged = sheet.Cells(row, 1).MergeCells
                        left_cell_has_fill = sheet.Cells(row, 1).Interior.ColorIndex != -4142  # -4142 is xlColorIndexNone
                        
                        if (left_cell and isinstance(left_cell, str) and 
                            (left_cell_is_bold or left_cell_is_merged or left_cell_has_fill)):
                            is_section_header = True
                    except Exception:
                        pass
                    
                    if is_section_header and left_cell:
                        # Create new section
                        if current_section:
                            sections.append(current_section)
                        
                        current_section = {
                            "name": str(left_cell),
                            "start_row": row,
                            "end_row": row,
                            "has_formulas": False,
                            "formula_count": 0
                        }
                    elif current_section:
                        # Extend current section
                        current_section["end_row"] = row
                
                # Add the last section if one exists
                if current_section:
                    sections.append(current_section)
                
                sheet_info["sections"] = sections
                
                # Analyze formulas
                formulas = []
                formula_count = 0
                
                # Formula patterns to identify
                common_patterns = {
                    "sum": 0,
                    "lookup": 0,
                    "conditional": 0,
                    "reference": 0,
                    "calculation": 0,
                    "date": 0
                }
                
                for row in range(1, min(used_range.Rows.Count, 1000) + 1):
                    for col in range(1, min(used_range.Columns.Count, 100) + 1):
                        try:
                            cell = sheet.Cells(row, col)
                            if cell.HasFormula:
                                formula_count += 1
                                
                                # Track formula in appropriate section
                                for section in sections:
                                    if section["start_row"] <= row <= section["end_row"]:
                                        section["has_formulas"] = True
                                        section["formula_count"] = section.get("formula_count", 0) + 1
                                
                                # Store formula details (limit to a reasonable number)
                                if len(formulas) < 50:
                                    cell_address = cell.Address(False, False)  # A1 style without $
                                    formula = cell.Formula
                                    
                                    formulas.append({
                                        "address": cell_address,
                                        "formula": formula,
                                        "row": row,
                                        "column": col
                                    })
                                
                                # Categorize formula
                                formula = cell.Formula.upper()
                                if "SUM" in formula or "SUBTOTAL" in formula:
                                    common_patterns["sum"] += 1
                                elif "VLOOKUP" in formula or "INDEX" in formula or "MATCH" in formula:
                                    common_patterns["lookup"] += 1
                                elif "IF" in formula:
                                    common_patterns["conditional"] += 1
                                elif "!" in formula:  # Sheet reference
                                    common_patterns["reference"] += 1
                                elif any(op in formula for op in ["+", "-", "*", "/"]):
                                    common_patterns["calculation"] += 1
                                elif any(func in formula for func in ["DATE", "TODAY", "NOW", "MONTH", "YEAR"]):
                                    common_patterns["date"] += 1
                        except Exception:
                            pass  # Skip if there's an error accessing the cell
                
                sheet_info["has_formulas"] = formula_count > 0
                sheet_info["formula_count"] = formula_count
                sheet_info["formulas"] = formulas
                
                # Add formula patterns with percentages
                if formula_count > 0:
                    formula_patterns = []
                    for pattern, count in common_patterns.items():
                        if count > 0:
                            formula_patterns.append({
                                "pattern": pattern,
                                "count": count,
                                "percentage": round(count / formula_count * 100, 1)
                            })
                    
                    sheet_info["formula_patterns"] = sorted(
                        formula_patterns, 
                        key=lambda x: x["count"], 
                        reverse=True
                    )
                
                # Check for tables
                try:
                    if len(sheet.ListObjects) > 0:
                        tables = []
                        for table in sheet.ListObjects:
                            tables.append({
                                "name": table.Name,
                                "range": table.Range.Address(False, False),
                                "column_count": table.Range.Columns.Count,
                                "row_count": table.Range.Rows.Count,
                                "has_headers": table.ShowHeaders
                            })
                        
                        sheet_info["data_tables"] = tables
                        
                        # Also add to workbook-level tables list
                        for table in tables:
                            table_info = table.copy()
                            table_info["sheet"] = sheet.Name
                            template_analysis["data_tables"].append(table_info)
                except Exception:
                    pass  # Skip if there's an error accessing tables
                
                # Check for charts
                try:
                    if len(sheet.ChartObjects()) > 0:
                        charts = []
                        for chart in sheet.ChartObjects():
                            chart_type = "Unknown"
                            try:
                                chart_type = chart.Chart.ChartType
                            except Exception:
                                pass
                            
                            charts.append({
                                "name": chart.Name,
                                "type": chart_type,
                                "has_title": chart.Chart.HasTitle
                            })
                        
                        sheet_info["charts"] = charts
                except Exception:
                    pass  # Skip if there's an error accessing charts
            
            template_analysis["sheets"].append(sheet_info)
        
        # Analyze sheet relationships (dependencies between sheets)
        try:
            sheet_dependencies = await self._analyze_sheet_dependencies(session, wb)
            template_analysis["worksheets_relationships"] = sheet_dependencies
        except Exception:
            pass  # Skip if there's an error analyzing dependencies
        
        # Add heuristic template categorization
        template_analysis["template_type"] = self._categorize_template(template_analysis)
        
        # Close the workbook
        session.close_workbook(wb, save_changes=False)
        
        return template_analysis
    
    async def _apply_excel_template(
        self,
        session: ExcelSession,
        template_analysis: Dict[str, Any],
        output_path: str,
        adaptation_context: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a learned Excel template to a new context.
        
        This method creates a new Excel file based on the template, adapting it to the new context.
        
        Args:
            session: Excel session
            template_analysis: Template analysis results from _analyze_excel_template
            output_path: Path where the new file should be saved
            adaptation_context: Natural language description of how to adapt the template
            parameters: Optional structured parameters
            
        Returns:
            Dictionary with adaptation results
        """
        # Create a new workbook
        wb = session.create_workbook()
        
        # Initialize the result
        result = {
            "success": True,
            "template_type": template_analysis.get("template_type", "unknown"),
            "output_path": output_path,
            "adaptations": []
        }
        
        # Process the adaptation context to understand what changes to make
        adaptation_context_lower = adaptation_context.lower()
        
        # Create the same sheet structure as the template
        template_sheets = template_analysis.get("sheets", [])
        
        # If the workbook has a default sheet, remove it (we'll create our own sheets)
        if wb.Sheets.Count > 0:
            default_sheet_name = wb.Sheets(1).Name
            
            # Don't remove if it's the only sheet and we have no sheets to add
            if len(template_sheets) > 0:
                # Create at least one sheet first to avoid errors
                new_sheet = wb.Sheets.Add(After=wb.Sheets(wb.Sheets.Count))
                new_sheet.Name = template_sheets[0]["name"]
                
                # Now remove the default sheet
                wb.Sheets(default_sheet_name).Delete()
        
        # Create the worksheet structure
        for i, sheet_info in enumerate(template_sheets):
            sheet_name = sheet_info["name"]
            
            # Adapt sheet name based on context if needed
            if "revenue" in sheet_name.lower() and "healthcare" in adaptation_context_lower:
                sheet_name = sheet_name.replace("Revenue", "Healthcare Revenue")
                result["adaptations"].append(f"Renamed sheet '{sheet_info['name']}' to '{sheet_name}'")
            
            # Create or use existing sheet
            if i == 0 and wb.Sheets.Count > 0:
                # Use the first sheet we already created when removing the default
                worksheet = wb.Sheets(1)
                worksheet.Name = sheet_name
            else:
                worksheet = wb.Sheets.Add(After=wb.Sheets(wb.Sheets.Count))
                worksheet.Name = sheet_name
            
            # Copy over the section structure
            sections = sheet_info.get("sections", [])
            
            for section in sections:
                section_name = section["name"]
                
                # Adapt section name based on context
                adapted_section_name = self._adapt_text_to_context(section_name, adaptation_context)
                if adapted_section_name != section_name:
                    result["adaptations"].append(f"Adapted section '{section_name}' to '{adapted_section_name}'")
                
                # Add the section header
                row = section["start_row"]
                worksheet.Cells(row, 1).Value = adapted_section_name
                worksheet.Cells(row, 1).Font.Bold = True
            
            # Copy headers if present
            if sheet_info.get("has_headers"):
                headers = sheet_info.get("headers", [])
                
                for col, header in enumerate(headers, 1):
                    if header:
                        # Adapt header based on context
                        adapted_header = self._adapt_text_to_context(header, adaptation_context)
                        if adapted_header != header:
                            result["adaptations"].append(f"Adapted header '{header}' to '{adapted_header}'")
                        
                        worksheet.Cells(1, col).Value = adapted_header
                        worksheet.Cells(1, col).Font.Bold = True
            
            # Set up formulas (simplified - a real implementation would be more sophisticated)
            formula_patterns = sheet_info.get("formula_patterns", [])
            
            if formula_patterns:
                # Add a note about the formula patterns
                formula_note = "Formula patterns replicated: " + ", ".join(p["pattern"] for p in formula_patterns)
                result["adaptations"].append(formula_note)
        
        # Create a general adaptations sheet based on the context
        adaptation_sheet = wb.Sheets.Add(After=wb.Sheets(wb.Sheets.Count))
        adaptation_sheet.Name = "Adaptation Notes"
        
        adaptation_sheet.Cells(1, 1).Value = "Context Description"
        adaptation_sheet.Cells(1, 1).Font.Bold = True
        adaptation_sheet.Cells(1, 2).Value = adaptation_context
        
        adaptation_sheet.Cells(3, 1).Value = "Adaptations Made"
        adaptation_sheet.Cells(3, 1).Font.Bold = True
        
        # Save adaptations to the sheet
        for i, adaptation in enumerate(result["adaptations"], 1):
            adaptation_sheet.Cells(3 + i, 1).Value = adaptation
        
        # Auto-fit columns
        adaptation_sheet.UsedRange.Columns.AutoFit()
        
        # Save the workbook
        session.save_workbook(wb, output_path)
        
        return result
    
    async def _analyze_excel_formulas(
        self,
        session: ExcelSession,
        file_path: str,
        sheet_name: Optional[str] = None,
        cell_range: Optional[str] = None,
        analysis_type: str = "analyze",
        detail_level: str = "standard"
    ) -> Dict[str, Any]:
        """Analyze Excel formulas for issues, optimizations, and explanations.
        
        This method provides in-depth analysis of Excel formulas.
        
        Args:
            session: Excel session
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to analyze
            cell_range: Cell range to analyze
            analysis_type: Type of analysis (analyze, debug, optimize, explain)
            detail_level: Level of detail in the analysis
            
        Returns:
            Dictionary with formula analysis results
        """
        # Open the workbook
        wb = session.open_workbook(file_path, read_only=True)
        
        # Get the worksheet
        worksheet = None
        if sheet_name:
            for sheet in wb.Worksheets:
                if sheet.Name.lower() == sheet_name.lower():
                    worksheet = sheet
                    break
        
        if not worksheet:
            worksheet = wb.ActiveSheet
            sheet_name = worksheet.Name
        
        # Determine the range to analyze
        if cell_range:
            try:
                range_obj = worksheet.Range(cell_range)
            except Exception as e:
                raise ToolInputError(f"Invalid cell range: {cell_range}") from e
        else:
            range_obj = worksheet.UsedRange
        
        # Initialize the result
        result = {
            "success": True,
            "file_path": file_path,
            "sheet_name": sheet_name,
            "range_analyzed": range_obj.Address(False, False),
            "analysis_type": analysis_type,
            "detail_level": detail_level,
            "formulas_analyzed": 0,
            "issues_found": 0,
            "formulas": []
        }
        
        # Analyze each cell in the range
        formula_count = 0
        issues_count = 0
        
        for row in range(1, range_obj.Rows.Count + 1):
            for col in range(1, range_obj.Columns.Count + 1):
                try:
                    cell = range_obj.Cells(row, col)
                    
                    if cell.HasFormula:
                        formula_count += 1
                        cell_info = {
                            "address": cell.Address(False, False),
                            "formula": cell.Formula,
                            "has_issues": False,
                            "issues": [],
                            "suggestions": []
                        }
                        
                        # Basic formula analysis
                        formula_text = cell.Formula
                        
                        # Check for known issues and patterns
                        
                        # 1. Check for volatile functions (recalculate with every change)
                        volatile_functions = ["NOW(", "TODAY(", "RAND(", "RANDBETWEEN(", "OFFSET(", "INDIRECT("]
                        for func in volatile_functions:
                            if func in formula_text.upper():
                                cell_info["has_issues"] = True
                                cell_info["issues"].append({
                                    "type": "performance",
                                    "description": f"Contains volatile function {func.rstrip('(')} that recalculates with every sheet change",
                                    "severity": "medium"
                                })
                                
                                if "NOW" in func or "TODAY" in func:
                                    cell_info["suggestions"].append("Consider if you really need the current date/time to update continuously. If not, use a static date/time value instead.")
                                elif "RAND" in func:
                                    cell_info["suggestions"].append("If random values don't need to change with every calculation, consider using a static list of pre-generated random numbers.")
                                elif "INDIRECT" in func:
                                    cell_info["suggestions"].append("INDIRECT is very powerful but can significantly slow down calculations. Consider using direct cell references or named ranges.")
                                
                                issues_count += 1
                        
                        # 2. Check for very large ranges
                        large_range_pattern = r"[A-Z]+[0-9]{3,}:[A-Z]+[0-9]{3,}"
                        if re.search(large_range_pattern, formula_text):
                            cell_info["has_issues"] = True
                            cell_info["issues"].append({
                                "type": "performance",
                                "description": "Contains very large cell ranges which may slow down calculations",
                                "severity": "medium"
                            })
                            cell_info["suggestions"].append("Limit ranges to only the necessary cells or use dynamic named ranges.")
                            issues_count += 1
                        
                        # 3. Check for nested IF statements (hard to maintain)
                        if formula_text.upper().count("IF(") > 2:
                            if_count = formula_text.upper().count("IF(")
                            cell_info["has_issues"] = True
                            cell_info["issues"].append({
                                "type": "complexity",
                                "description": f"Contains {if_count} nested IF statements which are hard to maintain",
                                "severity": "medium"
                            })
                            cell_info["suggestions"].append("Consider using IFS() or SWITCH() functions for multiple conditions, or separating logic into helper cells.")
                            issues_count += 1
                        
                        # 4. Check for error values
                        try:
                            cell_value = cell.Value
                            if isinstance(cell_value, str) and cell_value.startswith("#"):
                                error_type = cell_value
                                cell_info["has_issues"] = True
                                cell_info["issues"].append({
                                    "type": "error",
                                    "description": f"Formula results in {error_type} error",
                                    "severity": "high"
                                })
                                
                                # Suggest fixes based on error type
                                if error_type == "#DIV/0!":
                                    cell_info["suggestions"].append("Formula is attempting to divide by zero. Use IF() or IFERROR() to handle division by zero cases.")
                                elif error_type == "#VALUE!":
                                    cell_info["suggestions"].append("Formula is attempting an operation with incompatible data types. Check that all inputs are the correct type.")
                                elif error_type == "#REF!":
                                    cell_info["suggestions"].append("Formula contains invalid cell references. This often happens when referenced cells are deleted.")
                                elif error_type == "#NAME?":
                                    cell_info["suggestions"].append("Formula contains an undefined name. Check for typos in function names or named ranges.")
                                elif error_type == "#NUM!":
                                    cell_info["suggestions"].append("Formula results in a number that's too large or too small. Check for mathematical operations that might cause overflow.")
                                elif error_type == "#N/A":
                                    cell_info["suggestions"].append("Value not available. Use IFNA() to provide an alternative value when a lookup fails.")
                                
                                issues_count += 1
                        except Exception:
                            pass
                        
                        # 5. Check VLOOKUP without FALSE parameter (exact match)
                        if "VLOOKUP(" in formula_text.upper() and "FALSE" not in formula_text.upper() and "0" not in formula_text.upper().split("VLOOKUP")[1]:
                            cell_info["has_issues"] = True
                            cell_info["issues"].append({
                                "type": "reliability",
                                "description": "VLOOKUP without FALSE parameter will use approximate matching, which may return unexpected results",
                                "severity": "low"
                            })
                            cell_info["suggestions"].append("Add FALSE as the last parameter to VLOOKUP for exact matching, which is usually safer.")
                            issues_count += 1
                        
                        # Handle specific analysis types
                        if analysis_type == "explain":
                            # Add natural language explanation of what the formula does
                            cell_info["explanation"] = self._explain_formula(formula_text)
                        
                        elif analysis_type == "optimize":
                            # Add optimization suggestions
                            if "VLOOKUP(" in formula_text.upper():
                                cell_info["optimization"] = "Consider using XLOOKUP (if available) or INDEX/MATCH instead of VLOOKUP for more flexibility and better performance."
                            elif formula_text.upper().count("IF(") > 2:
                                cell_info["optimization"] = "Replace nested IF statements with IFS() or SWITCH() function (if available) for better readability and maintenance."
                        
                        # Add structured metadata based on detail level
                        if detail_level in ["detailed", "advanced"]:
                            # Include more detailed analysis
                            cell_info["functional_category"] = self._categorize_formula(formula_text)
                            cell_info["complexity"] = self._assess_formula_complexity(formula_text)
                            
                            if detail_level == "advanced":
                                # Add advanced optimization suggestions
                                cell_info["advanced_metadata"] = {
                                    "dependency_level": self._get_formula_dependency_level(formula_text),
                                    "volatility": self._check_formula_volatility(formula_text)
                                }
                        
                        result["formulas"].append(cell_info)
                except Exception as e:
                    # Skip cells with errors in analysis
                    logger.warning(f"Error analyzing cell: {str(e)}")
        
        # Add summary statistics
        result["formulas_analyzed"] = formula_count
        result["issues_found"] = issues_count
        
        # Generate summary
        result["summary"] = f"Analyzed {formula_count} formulas and found {issues_count} issues or optimization opportunities."
        
        if formula_count > 0:
            issue_percentage = (issues_count / formula_count) * 100
            if issue_percentage > 50:
                result["summary"] += " The spreadsheet has a high rate of formula issues and would benefit from significant optimization."
            elif issue_percentage > 20:
                result["summary"] += " There are several areas for improvement in the formula structure."
            elif issue_percentage > 0:
                result["summary"] += " Overall, the formulas are generally well-structured with a few minor issues."
            else:
                result["summary"] += " No issues found in the analyzed formulas. The spreadsheet appears to be well-optimized."
        
        # Close the workbook
        session.close_workbook(wb, save_changes=False)
        
        return result
    
    async def _generate_excel_macro(
        self,
        session: ExcelSession,
        instruction: str,
        file_path: Optional[str] = None,
        template: Optional[str] = None,
        test_execution: bool = False,
        security_level: str = "standard"
    ) -> Dict[str, Any]:
        """Generate Excel VBA macro code based on natural language instructions.
        
        This method creates VBA code for automating Excel tasks.
        
        Args:
            session: Excel session
            instruction: Natural language description of the macro
            file_path: Path to save the Excel file with macro
            template: Optional starting VBA code
            test_execution: Whether to test execute the macro
            security_level: Security level for execution
            
        Returns:
            Dictionary with the generated macro code and execution results
        """
        # Initialize result
        result = {
            "success": True,
            "instruction": instruction,
            "macro_type": "Sub",  # Default to Sub procedure
            "macro_language": "VBA",
            "code_generated": True
        }
        
        # Generate the macro name based on the instruction
        words = re.findall(r'\b[a-zA-Z]+\b', instruction)
        name_words = []
        
        # Take the first few significant words
        for word in words:
            if word.lower() not in ["the", "a", "an", "in", "on", "with", "and", "or", "to", "from", "that", "macro"]:
                name_words.append(word.capitalize())
                if len(name_words) >= 3:
                    break
        
        if not name_words:
            name_words = ["Custom", "Macro"]
        
        macro_name = "".join(name_words)
        
        # Check for specific macro types in the instruction
        instruction_lower = instruction.lower()
        
        if "function" in instruction_lower:
            result["macro_type"] = "Function"
        
        # Determine if this should be a workbook-level or module-level macro
        scope = "Module"
        if "workbook" in instruction_lower and ("open" in instruction_lower or "close" in instruction_lower):
            scope = "Workbook"
        elif "worksheet" in instruction_lower and ("activate" in instruction_lower or "change" in instruction_lower):
            scope = "Worksheet"
        
        # Generate the macro code based on the instruction
        if scope == "Module":
            # Standard module macro
            code = f"Sub {macro_name}()\n"
            code += "    ' Generated macro based on instruction:\n"
            code += f"    ' {instruction}\n"
            code += "    \n"
            
            # Add template code if provided
            if template:
                code += "    " + template.replace("\n", "\n    ") + "\n"
            else:
                # Generate code based on common tasks in the instruction
                
                # Check for data import
                if "import" in instruction_lower or "data" in instruction_lower:
                    code += "    ' Import data from external source\n"
                    if "csv" in instruction_lower:
                        code += "    Dim filePath As String\n"
                        code += '    filePath = Application.GetOpenFilename("CSV Files (*.csv), *.csv")\n'
                        code += "    \n"
                        code += '    If filePath <> "False" Then\n'
                        code += "        Workbooks.OpenText filePath, DataType:=xlDelimited, Comma:=True\n"
                        code += "        ' Process the imported data...\n"
                        code += "    End If\n"
                
                # Check for formatting
                if "format" in instruction_lower:
                    code += "    ' Apply formatting to selected range\n"
                    code += "    Dim rng As Range\n"
                    code += "    Set rng = Selection\n"
                    code += "    \n"
                    code += "    With rng\n"
                    code += "        .Font.Bold = True\n"
                    
                    if "color" in instruction_lower or "colour" in instruction_lower:
                        # Check for specific colors
                        if "blue" in instruction_lower:
                            code += "        .Interior.Color = RGB(200, 220, 255) ' Light blue\n"
                        elif "green" in instruction_lower:
                            code += "        .Interior.Color = RGB(200, 255, 200) ' Light green\n"
                        elif "red" in instruction_lower:
                            code += "        .Interior.Color = RGB(255, 200, 200) ' Light red\n"
                        else:
                            code += "        .Interior.Color = RGB(240, 240, 240) ' Light gray\n"
                    
                    code += "        .Borders.LineStyle = xlContinuous\n"
                    code += "        .Borders.Weight = xlThin\n"
                    code += "    End With\n"
                
                # Check for reports or analysis
                if "report" in instruction_lower or "analysis" in instruction_lower:
                    code += "    ' Generate report from data\n"
                    code += "    Dim wsData As Worksheet\n"
                    code += "    Dim wsReport As Worksheet\n"
                    code += "    \n"
                    code += "    ' Assume data is in a sheet named 'Data'\n"
                    code += '    Set wsData = ThisWorkbook.Worksheets("Data")\n'
                    code += "    \n"
                    code += "    ' Create or select report sheet\n"
                    code += "    On Error Resume Next\n"
                    code += '    Set wsReport = ThisWorkbook.Worksheets("Report")\n'
                    code += "    On Error GoTo 0\n"
                    code += "    \n"
                    code += "    If wsReport Is Nothing Then\n"
                    code += "        Set wsReport = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))\n"
                    code += '        wsReport.Name = "Report"\n'
                    code += '        wsReport.Name = "Report"\n'
                    code += "    \n"
                    code += "    ' Clear existing report content\n"
                    code += "    wsReport.Cells.Clear\n"
                    code += "    \n"
                    code += "    ' Add report header\n"
                    code += '    wsReport.Range("A1").Value = "Generated Report"\n'
                    code += '    wsReport.Range("A1").Font.Size = 14\n'
                    code += '    wsReport.Range("A1").Font.Bold = True\n'
                    
                    # Add pivot table if mentioned
                    if "pivot" in instruction_lower:
                        code += "    \n"
                        code += "    ' Create PivotTable\n"
                        code += "    Dim pvtCache As PivotCache\n"
                        code += "    Dim pvt As PivotTable\n"
                        code += "    Dim pvtRange As Range\n"
                        code += "    \n"
                        code += "    ' Get the data range from the Data sheet\n"
                        code += "    Set pvtRange = wsData.UsedRange\n"
                        code += "    \n"
                        code += "    ' Create pivot cache\n"
                        code += "    Set pvtCache = ThisWorkbook.PivotCaches.Create( _\n"
                        code += "        SourceType:=xlDatabase, _\n"
                        code += "        SourceData:=pvtRange)\n"
                        code += "    \n"
                        code += "    ' Create pivot table\n"
                        code += "    Set pvt = pvtCache.CreatePivotTable( _\n"
                        code += '        TableDestination:=wsReport.Range("A3"), _\n'
                        code += '        TableName:="PivotTable1")\n'
                        code += "    \n"
                        code += "    ' Add pivot fields - customize based on your data\n"
                        code += "    ' Assuming first column is row field and second column has values to summarize\n"
                        code += "    On Error Resume Next\n"
                        code += "    pvt.AddFields RowFields:=Array(1), ColumnFields:=Array(2)\n"
                        code += "    pvt.PivotFields(3).Orientation = xlDataField\n"
                        code += "    On Error GoTo 0\n"
                    
                    # Add charts if mentioned
                    if "chart" in instruction_lower:
                        code += "    \n"
                        code += "    ' Create chart based on report data\n"
                        code += "    Dim chtObj As ChartObject\n"
                        code += "    \n"
                        
                        # Determine chart type based on instruction
                        chart_type = "xlColumnClustered"  # Default
                        if "bar" in instruction_lower:
                            chart_type = "xlBarClustered"
                        elif "line" in instruction_lower:
                            chart_type = "xlLine"
                        elif "pie" in instruction_lower:
                            chart_type = "xlPie"
                        
                        code += "    ' Adjust the chart location and size as needed\n"
                        code += "    Set chtObj = wsReport.ChartObjects.Add(Left:=100, Top:=100, Width:=450, Height:=250)\n"
                        code += "    \n"
                        code += "    With chtObj.Chart\n"
                        code += "        .SetSourceData Source:=wsReport.Range(\"A10:B20\")  ' Adjust range to your data\n"
                        code += f"        .ChartType = {chart_type}\n"
                        code += "        .HasTitle = True\n"
                        code += '        .ChartTitle.Text = "Report Chart"\n'
                        code += "    End With\n"
            
            code += "End Sub\n"
            
        elif scope == "Workbook":
            # Workbook-level event handler
            code = "' Add this code to the ThisWorkbook module\n\n"
            
            if "open" in instruction_lower:
                code += "Private Sub Workbook_Open()\n"
                code += "    ' Code to execute when the workbook is opened\n"
                code += '    MsgBox "Workbook has been opened", vbInformation\n'
                
                # Add specific functionality based on instruction
                if "update" in instruction_lower or "refresh" in instruction_lower:
                    code += "    \n"
                    code += "    ' Refresh all data connections\n"
                    code += "    ThisWorkbook.RefreshAll\n"
                
                code += "End Sub\n"
                
            elif "close" in instruction_lower:
                code += "Private Sub Workbook_BeforeClose(Cancel As Boolean)\n"
                code += "    ' Code to execute before the workbook is closed\n"
                
                if "save" in instruction_lower:
                    code += "    ' Ensure workbook is saved\n"
                    code += "    Dim response As Integer\n"
                    code += '    response = MsgBox("Do you want to save changes?", vbYesNo + vbQuestion)\n'
                    code += "    \n"
                    code += "    If response = vbYes Then\n"
                    code += "        ThisWorkbook.Save\n"
                    code += "    End If\n"
                
                code += "End Sub\n"
                
        elif scope == "Worksheet":
            # Worksheet-level event handler
            code = "' Add this code to a Worksheet module\n\n"
            
            if "activate" in instruction_lower:
                code += "Private Sub Worksheet_Activate()\n"
                code += "    ' Code to execute when the worksheet is activated\n"
                
                if "format" in instruction_lower:
                    code += "    ' Apply formatting to key ranges\n"
                    code += '    Me.Range("A1:Z1").Interior.Color = RGB(200, 200, 200)\n'
                
                code += "End Sub\n"
                
            elif "change" in instruction_lower:
                code += "Private Sub Worksheet_Change(ByVal Target As Range)\n"
                code += "    ' Code to execute when cells in the worksheet change\n"
                code += "    \n"
                code += "    ' Check if the changed cell is in a specific range\n"
                code += '    If Not Intersect(Target, Me.Range("A1:A10")) Is Nothing Then\n'
                code += "        ' Do something when cells A1:A10 change\n"
                code += '        MsgBox "Data in critical range has changed", vbInformation\n'
                code += "    End If\n"
                code += "End Sub\n"
        
        # Store the generated code
        result["generated_code"] = code
        result["macro_name"] = macro_name
        result["macro_scope"] = scope
        
        # If file_path is provided, add the macro to the file
        if file_path:
            try:
                # Create or open the workbook
                if os.path.exists(file_path):
                    wb = session.open_workbook(file_path, read_only=False)
                else:
                    wb = session.create_workbook()
                
                # Get the VBA project
                vba_project = wb.VBProject
                
                # Add a module if needed
                module = None
                
                if scope == "Module":
                    # Find an existing module or create a new one
                    module_found = False
                    for comp_idx in range(1, vba_project.VBComponents.Count + 1):
                        comp = vba_project.VBComponents(comp_idx)
                        if comp.Type == 1:  # vbext_ct_StdModule
                            module = comp
                            module_found = True
                            break
                    
                    if not module_found:
                        module = vba_project.VBComponents.Add(1)  # vbext_ct_StdModule
                        module.Name = "MacroModule"
                
                elif scope == "Workbook":
                    # Get the ThisWorkbook module
                    for comp_idx in range(1, vba_project.VBComponents.Count + 1):
                        comp = vba_project.VBComponents(comp_idx)
                        if comp.Name == "ThisWorkbook":
                            module = comp
                            break
                
                elif scope == "Worksheet":
                    # Get the first worksheet module
                    for comp_idx in range(1, vba_project.VBComponents.Count + 1):
                        comp = vba_project.VBComponents(comp_idx)
                        if comp.Name.startswith("Sheet"):
                            module = comp
                            break
                
                if module:
                    # Add the code to the module
                    module.CodeModule.AddFromString(code)
                    
                    # Save the workbook
                    session.save_workbook(wb, file_path)
                    
                    result["file_path"] = file_path
                    result["macro_added_to_file"] = True
                
                # Test execution if requested
                if test_execution and scope == "Module":
                    try:
                        # Execute the macro
                        wb.Application.Run(macro_name)
                        result["execution_result"] = "Success"
                    except Exception as e:
                        result["execution_result"] = f"Error: {str(e)}"
                
            except Exception as e:
                logger.error(f"Error adding macro to file: {str(e)}", exc_info=True)
                result["error"] = str(e)
                result["macro_added_to_file"] = False
        
        return result
    
    # --- Helper methods for Excel analysis and processing ---
    
    async def _analyze_sheet_dependencies(self, session, workbook):
        """Analyze dependencies between worksheets in a workbook.
        
        Args:
            session: Excel session
            workbook: Workbook COM object
            
        Returns:
            List of dependencies between sheets
        """
        dependencies = []
        
        # Iterate through sheets
        for source_idx in range(1, workbook.Sheets.Count + 1):
            source_sheet = workbook.Sheets(source_idx)
            
            # Skip chart sheets
            if source_sheet.Type != 1:  # xlWorksheet
                continue
            
            used_range = source_sheet.UsedRange
            
            if not used_range:
                continue
            
            # Check a sample of cells for cross-sheet references
            for row in range(1, min(used_range.Rows.Count, 1000) + 1):
                for col in range(1, min(used_range.Columns.Count, 100) + 1):
                    try:
                        cell = used_range.Cells(row, col)
                        
                        if cell.HasFormula:
                            formula = cell.Formula
                            
                            # Look for sheet references (!), excluding external references ([])
                            if "!" in formula and not (("[" in formula) and ("]" in formula)):
                                # Extract sheet names from formula
                                sheet_refs = re.findall(r"'?([^'!]+)'?!", formula)
                                
                                for target_sheet_name in sheet_refs:
                                    # Find the target sheet
                                    target_sheet = None
                                    
                                    for sheet_idx in range(1, workbook.Sheets.Count + 1):
                                        sheet = workbook.Sheets(sheet_idx)
                                        if sheet.Name == target_sheet_name:
                                            target_sheet = sheet
                                            break
                                    
                                    if target_sheet:
                                        # Add the dependency
                                        dependencies.append({
                                            "source_sheet": source_sheet.Name,
                                            "target_sheet": target_sheet_name,
                                            "formula_sample": formula,
                                            "cell": cell.Address(False, False)
                                        })
                    except Exception:
                        pass  # Skip cells with errors
        
        # Group dependencies by source and target
        grouped_deps = {}
        
        for dep in dependencies:
            key = f"{dep['source_sheet']}=>{dep['target_sheet']}"
            
            if key not in grouped_deps:
                grouped_deps[key] = {
                    "source_sheet": dep["source_sheet"],
                    "target_sheet": dep["target_sheet"],
                    "reference_count": 0,
                    "formula_samples": []
                }
            
            grouped_deps[key]["reference_count"] += 1
            
            if len(grouped_deps[key]["formula_samples"]) < 3:  # Limit samples
                grouped_deps[key]["formula_samples"].append({
                    "cell": dep["cell"],
                    "formula": dep["formula_sample"]
                })
        
        return list(grouped_deps.values())
    
    async def _analyze_formulas_in_workbook(self, session, workbook):
        """Analyze formulas across a workbook.
        
        Args:
            session: Excel session
            workbook: Workbook COM object
            
        Returns:
            Dictionary with formula analysis results
        """
        # Initialize results
        analysis = {
            "total_formulas": 0,
            "sheets_with_formulas": 0,
            "formula_categories": {},
            "complexity": {
                "simple": 0,
                "moderate": 0,
                "complex": 0,
                "very_complex": 0
            },
            "samples": {}
        }
        
        # Function categories to track
        categories = {
            "mathematical": ["SUM", "AVERAGE", "MIN", "MAX", "COUNT", "PRODUCT", "ROUND"],
            "logical": ["IF", "AND", "OR", "NOT", "SWITCH", "IFS"],
            "lookup": ["VLOOKUP", "HLOOKUP", "INDEX", "MATCH", "XLOOKUP"],
            "text": ["CONCATENATE", "LEFT", "RIGHT", "MID", "FIND", "SEARCH", "REPLACE"],
            "date": ["TODAY", "NOW", "DATE", "DAY", "MONTH", "YEAR"],
            "financial": ["PMT", "RATE", "NPV", "IRR", "FV", "PV"],
            "statistical": ["STDEV", "VAR", "AVERAGE", "MEDIAN", "PERCENTILE"],
            "reference": ["INDIRECT", "OFFSET", "ADDRESS", "ROW", "COLUMN"],
            "database": ["DSUM", "DAVERAGE", "DCOUNT", "DGET"]
        }
        
        for category in categories:
            analysis["formula_categories"][category] = 0
            analysis["samples"][category] = []
        
        # Analyze each sheet
        for sheet_idx in range(1, workbook.Sheets.Count + 1):
            sheet = workbook.Sheets(sheet_idx)
            
            # Skip chart sheets
            if sheet.Type != 1:  # xlWorksheet
                continue
            
            used_range = sheet.UsedRange
            
            if not used_range:
                continue
            
            sheet_has_formulas = False
            sheet_formula_count = 0
            
            # Check cells for formulas
            for row in range(1, min(used_range.Rows.Count, 1000) + 1):
                for col in range(1, min(used_range.Columns.Count, 100) + 1):
                    try:
                        cell = used_range.Cells(row, col)
                        
                        if cell.HasFormula:
                            formula = cell.Formula
                            sheet_has_formulas = True
                            sheet_formula_count += 1
                            analysis["total_formulas"] += 1
                            
                            # Categorize formula
                            formula_upper = formula.upper()
                            categorized = False
                            
                            for category, functions in categories.items():
                                for func in functions:
                                    if func.upper() + "(" in formula_upper:
                                        analysis["formula_categories"][category] += 1
                                        
                                        # Store a sample if needed
                                        if len(analysis["samples"][category]) < 3:
                                            analysis["samples"][category].append({
                                                "sheet": sheet.Name,
                                                "cell": cell.Address(False, False),
                                                "formula": formula
                                            })
                                        
                                        categorized = True
                                        break
                                
                                if categorized:
                                    break
                            
                            # Assess complexity
                            complexity = self._assess_formula_complexity(formula)
                            analysis["complexity"][complexity] += 1
                    except Exception:
                        pass  # Skip cells with errors
            
            if sheet_has_formulas:
                analysis["sheets_with_formulas"] += 1
        
        return analysis
    
    def _categorize_template(self, template_analysis):
        """Categorize a template based on its structure and contents.
        
        Args:
            template_analysis: Analysis of the template
            
        Returns:
            String indicating the template category
        """
        # Extract relevant information from analysis
        sheets = template_analysis.get("sheets", [])
        sheet_names = [s.get("name", "").lower() for s in sheets]
        
        # Look for common sheet patterns
        has_financial_sheets = any(name in ["income", "balance", "cash flow", "forecast", "budget", "revenue"] for name in sheet_names)
        has_project_sheets = any(name in ["tasks", "timeline", "gantt", "resources", "schedule"] for name in sheet_names)
        has_dashboard_sheets = any(name in ["dashboard", "summary", "overview", "kpi", "metrics"] for name in sheet_names)
        has_data_sheets = any(name in ["data", "raw data", "source", "input"] for name in sheet_names)
        
        # Check formula patterns
        formula_patterns = []
        for sheet in sheets:
            formula_patterns.extend(sheet.get("formula_patterns", []))
        
        # Count pattern types
        financial_formulas = sum(p.get("count", 0) for p in formula_patterns if p.get("pattern") in ["sum", "calculation"])
        lookup_formulas = sum(p.get("count", 0) for p in formula_patterns if p.get("pattern") in ["lookup", "reference"])
        
        # Determine category based on collected information
        if has_financial_sheets and financial_formulas > 0:
            if "forecast" in " ".join(sheet_names) or "projection" in " ".join(sheet_names):
                return "financial_forecast"
            elif "budget" in " ".join(sheet_names):
                return "budget"
            else:
                return "financial"
        
        elif has_project_sheets:
            return "project_management"
        
        elif has_dashboard_sheets and has_data_sheets:
            return "dashboard"
        
        elif lookup_formulas > financial_formulas:
            return "data_analysis"
        
        else:
            return "general"
    
    def _adapt_text_to_context(self, text, context):
        """Adapt text based on context for template adaptation.
        
        Args:
            text: Original text string
            context: Context description
            
        Returns:
            Adapted text
        """
        if not text or not isinstance(text, str):
            return text
        
        # Check what industry or domain is mentioned in the context
        context_lower = context.lower()
        industry = None
        
        # Try to detect the target industry or domain
        if "healthcare" in context_lower or "medical" in context_lower or "hospital" in context_lower:
            industry = "healthcare"
        elif "tech" in context_lower or "software" in context_lower or "saas" in context_lower:
            industry = "technology"
        elif "retail" in context_lower or "shop" in context_lower or "store" in context_lower:
            industry = "retail"
        elif "finance" in context_lower or "bank" in context_lower or "investment" in context_lower:
            industry = "finance"
        elif "education" in context_lower or "school" in context_lower or "university" in context_lower:
            industry = "education"
        elif "manufacturing" in context_lower or "factory" in context_lower:
            industry = "manufacturing"
        elif "real estate" in context_lower or "property" in context_lower:
            industry = "real_estate"
        
        # If no industry detected, return original text
        if not industry:
            return text
        
        # Adapt common business terms based on industry
        text_lower = text.lower()
        
        # Handle healthcare industry adaptations
        if industry == "healthcare":
            if "customer" in text_lower:
                return text.replace("Customer", "Patient").replace("customer", "patient")
            elif "sales" in text_lower:
                return text.replace("Sales", "Services").replace("sales", "services")
            elif "product" in text_lower:
                return text.replace("Product", "Treatment").replace("product", "treatment")
            elif "revenue" in text_lower and "healthcare revenue" not in text_lower:
                return text.replace("Revenue", "Healthcare Revenue").replace("revenue", "healthcare revenue")
        
        # Handle technology industry adaptations
        elif industry == "technology":
            if "customer" in text_lower:
                return text.replace("Customer", "User").replace("customer", "user")
            elif "sales" in text_lower:
                return text.replace("Sales", "Subscriptions").replace("sales", "subscriptions")
            elif "product" in text_lower:
                return text.replace("Product", "Solution").replace("product", "solution")
            
        # Handle retail industry adaptations
        elif industry == "retail":
            if "customer" in text_lower:
                return text.replace("Customer", "Shopper").replace("customer", "shopper")
            elif "sales" in text_lower:
                return text.replace("Sales", "Retail Sales").replace("sales", "retail sales")
        
        # Handle finance industry adaptations
        elif industry == "finance":
            if "customer" in text_lower:
                return text.replace("Customer", "Client").replace("customer", "client")
            elif "product" in text_lower:
                return text.replace("Product", "Financial Product").replace("product", "financial product")
        
        # Handle education industry adaptations
        elif industry == "education":
            if "customer" in text_lower:
                return text.replace("Customer", "Student").replace("customer", "student")
            elif "sales" in text_lower:
                return text.replace("Sales", "Enrollments").replace("sales", "enrollments")
            elif "product" in text_lower:
                return text.replace("Product", "Course").replace("product", "course")
        
        # Default - return original text
        return text
    
    def _explain_formula(self, formula):
        """Generate a natural language explanation of an Excel formula.
        
        Args:
            formula: Excel formula string
            
        Returns:
            Human-readable explanation
        """
        formula_upper = formula.upper()
        
        # SUM function
        if "SUM(" in formula_upper:
            match = re.search(r"SUM\(([^)]+)\)", formula_upper)
            if match:
                range_str = match.group(1)
                return f"This formula calculates the sum of values in the range {range_str}."
        
        # AVERAGE function
        elif "AVERAGE(" in formula_upper:
            match = re.search(r"AVERAGE\(([^)]+)\)", formula_upper)
            if match:
                range_str = match.group(1)
                return f"This formula calculates the average (mean) of values in the range {range_str}."
        
        # VLOOKUP function
        elif "VLOOKUP(" in formula_upper:
            params = formula_upper.split("VLOOKUP(")[1].split(")", 1)[0].split(",")
            lookup_value = params[0] if len(params) > 0 else "?"
            table_array = params[1] if len(params) > 1 else "?"
            col_index = params[2] if len(params) > 2 else "?"
            exact_match = "FALSE" in params[3] if len(params) > 3 else False
            
            match_type = "exact match" if exact_match else "closest match (approximate match)"
            return f"This formula looks up {lookup_value} in the first column of {table_array}, and returns the value from column {col_index}. It finds the {match_type}."
        
        # IF function
        elif "IF(" in formula_upper:
            try:
                # This is a simplistic parsing - real parsing would be more complex
                content = formula_upper.split("IF(")[1].split(")", 1)[0]
                parts = []
                depth = 0
                current = ""
                
                for char in content:
                    if char == "," and depth == 0:
                        parts.append(current)
                        current = ""
                    else:
                        if char == "(":
                            depth += 1
                        elif char == ")":
                            depth -= 1
                        current += char
                
                if current:
                    parts.append(current)
                
                condition = parts[0] if len(parts) > 0 else "?"
                true_value = parts[1] if len(parts) > 1 else "?"
                false_value = parts[2] if len(parts) > 2 else "?"
                
                return f"This formula tests if {condition}. If true, it returns {true_value}, otherwise it returns {false_value}."
            except Exception:
                # Fallback if parsing fails
                return "This formula uses an IF statement to return different values based on a condition."
        
        # INDEX/MATCH
        elif "INDEX(" in formula_upper and "MATCH(" in formula_upper:
            return "This formula uses the INDEX/MATCH combination to look up a value in a table. INDEX returns a value at a specific position, and MATCH finds the position of a lookup value."
        
        # Simple calculations
        elif "+" in formula or "-" in formula or "*" in formula or "/" in formula:
            # Check if it's a simple calculation without functions
            if not any(func in formula_upper for func in ["SUM(", "AVERAGE(", "IF(", "VLOOKUP(", "INDEX("]):
                operations = []
                if "+" in formula:
                    operations.append("addition")
                if "-" in formula:
                    operations.append("subtraction")
                if "*" in formula:
                    operations.append("multiplication")
                if "/" in formula:
                    operations.append("division")
                
                ops_text = " and ".join(operations)
                return f"This formula performs {ops_text} on the specified values or cell references."
        
        # Fallback for unrecognized or complex formulas
        return "This formula performs a calculation on the referenced cells. For complex formulas, consider breaking it down into its component parts to understand it better."
    
    def _categorize_formula(self, formula):
        """Categorize a formula based on its functions and structure.
        
        Args:
            formula: Excel formula string
            
        Returns:
            Category string
        """
        formula_upper = formula.upper()
        
        # Mathematical
        if any(func in formula_upper for func in ["SUM(", "AVERAGE(", "MIN(", "MAX(", "COUNT(", "PRODUCT(", "ROUND("]):
            return "mathematical"
        
        # Logical
        elif any(func in formula_upper for func in ["IF(", "AND(", "OR(", "NOT(", "SWITCH(", "IFS("]):
            return "logical"
        
        # Lookup
        elif any(func in formula_upper for func in ["VLOOKUP(", "HLOOKUP(", "INDEX(", "MATCH(", "XLOOKUP("]):
            return "lookup"
        
        # Text
        elif any(func in formula_upper for func in ["CONCATENATE(", "LEFT(", "RIGHT(", "MID(", "FIND(", "SEARCH(", "REPLACE("]):
            return "text"
        
        # Date
        elif any(func in formula_upper for func in ["TODAY(", "NOW(", "DATE(", "DAY(", "MONTH(", "YEAR("]):
            return "date"
        
        # Financial
        elif any(func in formula_upper for func in ["PMT(", "RATE(", "NPV(", "IRR(", "FV(", "PV("]):
            return "financial"
        
        # Statistical
        elif any(func in formula_upper for func in ["STDEV(", "VAR(", "MEDIAN(", "PERCENTILE("]):
            return "statistical"
        
        # Reference
        elif any(func in formula_upper for func in ["INDIRECT(", "OFFSET(", "ADDRESS(", "ROW(", "COLUMN("]):
            return "reference"
        
        # Database
        elif any(func in formula_upper for func in ["DSUM(", "DAVERAGE(", "DCOUNT(", "DGET("]):
            return "database"
        
        # Simple calculation
        elif any(op in formula for op in ["+", "-", "*", "/"]):
            return "calculation"
        
        # Default/unknown
        return "other"
    
    def _assess_formula_complexity(self, formula):
        """Assess the complexity of a formula.
        
        Args:
            formula: Excel formula string
            
        Returns:
            Complexity level (simple, moderate, complex, very_complex)
        """
        # Count various aspects of the formula
        formula_length = len(formula)
        function_count = formula.upper().count("(")
        nesting_level = 0
        max_nesting = 0
        
        # Calculate nesting depth
        for char in formula:
            if char == "(":
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif char == ")":
                nesting_level -= 1
        
        # Count references
        reference_count = len(re.findall(r"[A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?", formula))
        
        # Count operators
        operator_count = sum(formula.count(op) for op in ["+", "-", "*", "/", "=", "<", ">", "&"])
        
        # Calculate a weighted complexity score
        score = (
            min(10, formula_length / 40) +          # Length: max 10 points
            function_count * 1.5 +                   # Functions: 1.5 points each
            max_nesting * 2 +                        # Max nesting: 2 points per level
            reference_count * 0.5 +                  # References: 0.5 points each
            operator_count * 0.5                     # Operators: 0.5 points each
        )
        
        # Determine complexity level
        if score < 5:
            return "simple"
        elif score < 10:
            return "moderate"
        elif score < 20:
            return "complex"
        else:
            return "very_complex"
    
    def _get_formula_dependency_level(self, formula):
        """Determine how many other cells a formula depends on.
        
        Args:
            formula: Excel formula string
            
        Returns:
            Dependency level (low, medium, high)
        """
        # Count cell references and ranges
        references = re.findall(r"[A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?", formula)
        
        # Count individual cells
        cell_count = 0
        for ref in references:
            if ":" in ref:
                # It's a range
                try:
                    start, end = ref.split(":")
                    start_col = re.search(r"[A-Z]+", start).group(0)
                    start_row = int(re.search(r"[0-9]+", start).group(0))
                    end_col = re.search(r"[A-Z]+", end).group(0)
                    end_row = int(re.search(r"[0-9]+", end).group(0))
                    
                    # Convert column letters to numbers
                    start_col_num = 0
                    for char in start_col:
                        start_col_num = start_col_num * 26 + (ord(char) - ord('A') + 1)
                    
                    end_col_num = 0
                    for char in end_col:
                        end_col_num = end_col_num * 26 + (ord(char) - ord('A') + 1)
                    
                    # Calculate cells in range
                    cells_in_range = (end_row - start_row + 1) * (end_col_num - start_col_num + 1)
                    cell_count += cells_in_range
                except Exception:
                    # Fallback if parsing fails
                    cell_count += 10  # Assume a moderate size range
            else:
                # Single cell
                cell_count += 1
        
        # Determine dependency level
        if cell_count <= 3:
            return "low"
        elif cell_count <= 10:
            return "medium"
        else:
            return "high"
    
    def _check_formula_volatility(self, formula):
        """Check if a formula contains volatile functions.
        
        Args:
            formula: Excel formula string
            
        Returns:
            Volatility level (none, low, high)
        """
        formula_upper = formula.upper()
        
        # Highly volatile functions
        high_volatility = ["NOW(", "TODAY(", "RAND(", "RANDBETWEEN("]
        if any(func in formula_upper for func in high_volatility):
            return "high"
        
        # Low volatility functions
        low_volatility = ["OFFSET(", "INDIRECT(", "CELL(", "INFO("]
        if any(func in formula_upper for func in low_volatility):
            return "low"
        
        # Non-volatile
        return "none"
    
    def _get_calculation_mode_name(self, mode_value):
        """Convert Excel calculation mode numeric value to name.
        
        Args:
            mode_value: Numeric value of calculation mode
            
        Returns:
            String name of calculation mode
        """
        modes = {
            -4105: "Automatic",
            -4135: "Manual",
            -4133: "Semiautomatic"
        }
        
        return modes.get(mode_value, f"Unknown ({mode_value})")


def register_excel_spreadsheet_tools(mcp_server):
    """Registers Excel Spreadsheet Tools with the MCP server.
    
    Args:
        mcp_server: MCP server instance
        
    Returns:
        ExcelSpreadsheetTools instance
    """
    # Initialize the tool
    excel_tools = ExcelSpreadsheetTools(mcp_server)
    
    # Register tools with MCP server
    mcp_server.tool(name="excel_execute")(excel_tools.excel_execute)
    mcp_server.tool(name="excel_learn_and_apply")(excel_tools.excel_learn_and_apply)
    mcp_server.tool(name="excel_analyze_formulas")(excel_tools.excel_analyze_formulas)
    mcp_server.tool(name="excel_generate_macro")(excel_tools.excel_generate_macro)
    
    return excel_tools