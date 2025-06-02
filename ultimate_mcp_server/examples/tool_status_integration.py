"""
Example: Integrating Tool Status with MCP Tools
==============================================

This example shows how to integrate the tool status tracking system with actual MCP tools
to provide users with real-time feedback about tool availability and loading progress.
"""

import asyncio
from typing import Optional, Dict, Any

# Import our utilities
from ultimate_mcp_server.utils.lazy_imports import (
    get_numpy, get_pandas, get_cv2, get_docling,
    get_tool_loading_status, get_all_tools_status,
    lazy
)
from ultimate_mcp_server.utils.tool_status import (
    tool_status, track_tool_loading, ToolStatus
)


# Example 1: Document conversion tool with status tracking
@track_tool_loading('convert_document')
async def convert_document_tool(
    document_path: str,
    output_format: str = "markdown"
) -> Dict[str, Any]:
    """
    Convert document with lazy loading and status tracking.
    
    This tool will report its loading status so clients know if it's:
    - Loading dependencies (docling, cv2, torch, etc.)
    - Ready to use
    - Failed to load
    """
    try:
        # Check if dependencies are loading
        docling_status = get_tool_loading_status('docling')
        cv2_status = get_tool_loading_status('cv2')
        
        if docling_status.get('loading') or cv2_status.get('loading'):
            loading_tools = []
            if docling_status.get('loading'):
                remaining = docling_status.get('estimated_remaining_seconds', '?')
                loading_tools.append(f"docling (~{remaining}s)")
            if cv2_status.get('loading'):
                remaining = cv2_status.get('estimated_remaining_seconds', '?')
                loading_tools.append(f"opencv (~{remaining}s)")
            
            return {
                'success': False,
                'status': 'loading_dependencies',
                'message': f"Loading dependencies: {', '.join(loading_tools)}",
                'retry_after_seconds': max(
                    docling_status.get('estimated_remaining_seconds', 0),
                    cv2_status.get('estimated_remaining_seconds', 0)
                )
            }
        
        # Lazy load dependencies (will use cached versions if already loaded)
        docling = get_docling()
        cv2 = get_cv2()
        
        # Your actual document conversion logic here
        # For example:
        # converter = docling.DocumentConverter()
        # result = converter.convert(document_path)
        
        return {
            'success': True,
            'status': 'completed',
            'output_format': output_format,
            'message': 'Document converted successfully'
        }
        
    except Exception as e:
        return {
            'success': False,
            'status': 'error',
            'error': str(e)
        }


# Example 2: Data analysis tool with numpy/pandas
@track_tool_loading('data_analysis')
async def analyze_data_tool(
    data_path: str,
    analysis_type: str = "summary"
) -> Dict[str, Any]:
    """
    Data analysis tool with status tracking for numpy/pandas.
    """
    try:
        # Check status of required libraries
        np_status = get_tool_loading_status('numpy')
        pd_status = get_tool_loading_status('pandas')
        
        # If either is loading, inform the user
        if np_status.get('loading') or pd_status.get('loading'):
            loading_info = []
            total_wait = 0
            
            if np_status.get('loading'):
                remaining = np_status.get('estimated_remaining_seconds', 3)
                loading_info.append(f"numpy (~{remaining}s)")
                total_wait = max(total_wait, remaining)
                
            if pd_status.get('loading'):
                remaining = pd_status.get('estimated_remaining_seconds', 3)
                loading_info.append(f"pandas (~{remaining}s)")
                total_wait = max(total_wait, remaining)
            
            return {
                'success': False,
                'status': 'loading_dependencies',
                'message': f"Loading data libraries: {', '.join(loading_info)}",
                'retry_after_seconds': total_wait
            }
        
        # Check if libraries failed to load
        if not np_status.get('available') or not pd_status.get('available'):
            failed_libs = []
            if not np_status.get('available'):
                failed_libs.append(f"numpy: {np_status.get('error', 'unknown error')}")
            if not pd_status.get('available'):
                failed_libs.append(f"pandas: {pd_status.get('error', 'unknown error')}")
            
            return {
                'success': False,
                'status': 'dependencies_failed',
                'error': f"Required libraries failed to load: {'; '.join(failed_libs)}"
            }
        
        # Load libraries (should be instant if already loaded)
        np = get_numpy()
        pd = get_pandas()
        
        # Your actual data analysis logic here
        # data = pd.read_csv(data_path)
        # result = np.mean(data.values)
        
        return {
            'success': True,
            'status': 'completed',
            'analysis_type': analysis_type,
            'message': 'Data analysis completed successfully'
        }
        
    except Exception as e:
        return {
            'success': False,
            'status': 'error',
            'error': str(e)
        }


# Example 3: Tool status reporting MCP tool
async def get_tool_status_mcp(tool_name: Optional[str] = None) -> Dict[str, Any]:
    """
    MCP tool for checking tool loading status.
    
    Args:
        tool_name: Specific tool to check, or None for all tools
    
    Returns:
        Tool status information with user-friendly messages
    """
    try:
        if tool_name:
            status = get_tool_loading_status(tool_name)
            
            if status.get('status_tracking') == 'disabled':
                return {
                    'success': True,
                    'message': 'Tool status tracking is not enabled',
                    'tool_status': 'unknown'
                }
            
            # Add user-friendly status message
            if status.get('loading'):
                remaining = status.get('estimated_remaining_seconds')
                if remaining:
                    message = f"Tool '{tool_name}' is loading, approximately {remaining} seconds remaining"
                else:
                    current_time = status.get('current_load_time_seconds', 0)
                    message = f"Tool '{tool_name}' is loading (current: {current_time:.1f}s)"
            elif status.get('available'):
                load_time = status.get('load_time_seconds')
                if load_time:
                    message = f"Tool '{tool_name}' is ready (loaded in {load_time:.3f}s)"
                else:
                    message = f"Tool '{tool_name}' is ready"
            elif status.get('status') == 'failed':
                retry_count = status.get('retry_count', 0)
                error = status.get('error', 'Unknown error')
                message = f"Tool '{tool_name}' failed to load (attempts: {retry_count}): {error}"
            elif status.get('status') == 'unavailable':
                error = status.get('error', 'Unknown reason')
                message = f"Tool '{tool_name}' is permanently unavailable: {error}"
            else:
                message = f"Tool '{tool_name}' is not loaded"
            
            return {
                'success': True,
                'tool_name': tool_name,
                'status': status.get('status', 'unknown'),
                'available': status.get('available', False),
                'loading': status.get('loading', False),
                'message': message,
                'details': status
            }
        
        else:
            # Get all tools status
            all_status = get_all_tools_status()
            
            if all_status.get('status_tracking') == 'disabled':
                return {
                    'success': True,
                    'message': 'Tool status tracking is not enabled',
                    'summary': 'Status tracking disabled'
                }
            
            summary = all_status.get('summary', {})
            total = summary.get('total_tools', 0)
            loaded = summary.get('loaded', 0)
            loading = summary.get('loading', 0)
            failed = summary.get('failed', 0)
            
            # Create user-friendly summary message
            status_parts = []
            if loaded > 0:
                status_parts.append(f"{loaded} loaded")
            if loading > 0:
                loading_tools = summary.get('loading_tools', [])
                status_parts.append(f"{loading} loading ({', '.join(loading_tools)})")
            if failed > 0:
                failed_tools = summary.get('failed_tools', [])
                status_parts.append(f"{failed} failed ({', '.join(failed_tools)})")
            
            message = f"Tools status: {', '.join(status_parts)} out of {total} total"
            
            return {
                'success': True,
                'message': message,
                'summary': summary,
                'all_tools': all_status.get('tools', {})
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to get tool status: {str(e)}"
        }


# Example 4: Tool preloading function
async def preload_tools_mcp(tool_categories: Optional[list] = None) -> Dict[str, Any]:
    """
    MCP tool to preload common tool categories.
    
    Args:
        tool_categories: List of categories like ['ml', 'document', 'browser']
    
    Returns:
        Preloading status and progress
    """
    try:
        from ultimate_mcp_server.utils.lazy_imports import (
            preload_ml_stack, preload_document_stack, preload_browser_stack
        )
        
        if not tool_categories:
            tool_categories = ['ml', 'document']  # Default categories
        
        preload_tasks = []
        category_names = []
        
        if 'ml' in tool_categories:
            preload_tasks.append(preload_ml_stack())
            category_names.append('ML libraries (numpy, pandas, sklearn, etc.)')
        
        if 'document' in tool_categories:
            preload_tasks.append(preload_document_stack())
            category_names.append('Document processing (docling, cv2, transformers)')
        
        if 'browser' in tool_categories:
            preload_tasks.append(preload_browser_stack())
            category_names.append('Browser automation (playwright)')
        
        if not preload_tasks:
            return {
                'success': False,
                'error': 'No valid tool categories specified'
            }
        
        # Start preloading
        await asyncio.gather(*preload_tasks, return_exceptions=True)
        
        # Get final status
        summary = get_all_tools_status().get('summary', {})
        
        return {
            'success': True,
            'message': f"Preloading completed for: {', '.join(category_names)}",
            'categories_loaded': tool_categories,
            'final_status': summary
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Preloading failed: {str(e)}"
        }


# Example 5: Integration with MCP server registration
def register_status_aware_tools(mcp_server):
    """
    Example of how to register tools with the MCP server.
    """
    # Register the main working tools
    mcp_server.add_tool("convert_document", convert_document_tool)
    mcp_server.add_tool("analyze_data", analyze_data_tool)
    
    # Register status and management tools
    mcp_server.add_tool("get_tool_status", get_tool_status_mcp)
    mcp_server.add_tool("preload_tools", preload_tools_mcp)
    
    # Register common heavy tools for status tracking
    heavy_tools = {
        'convert_document': ['docling', 'cv2', 'torch'],
        'ocr_image': ['cv2', 'tesseract'],
        'analyze_data': ['numpy', 'pandas'],
        'smart_browser': ['playwright'],
        'execute_python': ['pyodide']
    }
    
    for tool_name, deps in heavy_tools.items():
        tool_status.register_tool(tool_name, deps)


# Example usage script
async def main():
    """Example usage of the status-aware tools."""
    print("🚀 Starting Ultimate MCP Server with status tracking...")
    
    # Check initial status
    print("\n📊 Initial tool status:")
    status_result = await get_tool_status_mcp()
    print(status_result['message'])
    
    # Try to use a heavy tool
    print("\n🔧 Attempting to convert document...")
    doc_result = await convert_document_tool("example.pdf")
    print(f"Result: {doc_result['message']}")
    
    if not doc_result['success'] and doc_result.get('retry_after_seconds'):
        wait_time = doc_result['retry_after_seconds']
        print(f"⏳ Waiting {wait_time} seconds for dependencies to load...")
        await asyncio.sleep(wait_time)
        
        # Try again
        doc_result = await convert_document_tool("example.pdf")
        print(f"Retry result: {doc_result['message']}")
    
    # Check final status
    print("\n📊 Final tool status:")
    final_status = await get_tool_status_mcp()
    print(final_status['message'])


if __name__ == "__main__":
    # Run example
    asyncio.run(main())


"""
EXPECTED USER EXPERIENCE:

1. Client calls convert_document tool
2. If dependencies are loading:
   - Returns: {"status": "loading_dependencies", "retry_after_seconds": 8}
   - Client can show loading indicator and retry after specified time

3. Client calls get_tool_status
   - Returns: "Tool 'docling' is loading, approximately 6 seconds remaining"
   - Client can show progress to user

4. Client retries convert_document after waiting
   - Returns: {"status": "completed", "message": "Document converted successfully"}
   - Tool is now ready for immediate use

5. Subsequent calls are instant (dependencies cached)

This provides transparent feedback to users about what's happening behind the scenes.
""" 