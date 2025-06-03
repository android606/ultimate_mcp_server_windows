#!/usr/bin/env python3
"""
Custom virtual environment builder for Ultimate MCP Server.
Extends EnvBuilder to create a more robust virtual environment,
particularly on Windows systems.
"""

import os
import sys
import platform
from pathlib import Path
import venv
import subprocess


class UltimateMCPEnvBuilder(venv.EnvBuilder):
    """
    Enhanced virtual environment builder for Ultimate MCP Server.
    
    This builder ensures:
    1. The virtual environment's Scripts or bin directory is properly added to PATH
    2. Git HOME environment variable is properly set on Windows
    3. Python environment detection works correctly
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as venv.EnvBuilder."""
        self.with_pip = not kwargs.pop('without_pip', False)
        super().__init__(*args, **kwargs)
    
    def post_setup(self, context):
        """
        Customize the environment after it's been created.
        
        Args:
            context: The context from EnvBuilder with paths and executables
        """
        # Let the base class do its setup first
        super().post_setup(context)
        
        # Customize activation scripts
        self._customize_activation_scripts(context)
        
        # Create a .env file in the virtual environment to set environment variables
        self._create_env_file(context)
        
        # Install essential packages if pip is available
        if self.with_pip:
            self._install_essential_packages(context)
    
    def _customize_activation_scripts(self, context):
        """
        Customize activation scripts to ensure proper PATH setup.
        
        Args:
            context: The context with environment paths
        """
        # Paths to activation scripts
        is_windows = platform.system() == 'Windows'
        if is_windows:
            # Windows scripts
            batch_script = Path(context.bin_path) / 'activate.bat'
            ps_script = Path(context.bin_path) / 'Activate.ps1'
            
            # Modify batch script
            if batch_script.exists():
                self._modify_batch_activation(batch_script)
            
            # Modify PowerShell script
            if ps_script.exists():
                self._modify_ps_activation(ps_script)
        else:
            # Unix scripts
            bash_script = Path(context.bin_path) / 'activate'
            
            # Modify bash script
            if bash_script.exists():
                self._modify_bash_activation(bash_script)
    
    def _modify_batch_activation(self, script_path):
        """
        Modify the Windows batch activation script to ensure PATH is set correctly.
        
        Args:
            script_path: Path to the activation.bat file
        """
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check if our modifications are already there
        if "REM Ultimate MCP Server customization" in content:
            return
            
        # Add our custom code to ensure PATH is set correctly
        custom_content = """
REM Ultimate MCP Server customization
set "PATH=%VIRTUAL_ENV%\\Scripts;%PATH%"
REM Ensure HOME is set for Git
if not defined HOME (
    if defined USERPROFILE (
        set "HOME=%USERPROFILE%"
    ) else (
        if defined HOMEDRIVE (
            if defined HOMEPATH (
                set "HOME=%HOMEDRIVE%%HOMEPATH%"
            )
        )
    )
)
"""
        
        # Find the insertion point (after setting VIRTUAL_ENV)
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('set "VIRTUAL_ENV='):
                # Insert our custom content after this line
                lines.insert(i + 1, custom_content)
                break
        
        # Write the modified content back
        with open(script_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def _modify_ps_activation(self, script_path):
        """
        Modify the PowerShell activation script to ensure PATH is set correctly.
        
        Args:
            script_path: Path to the Activate.ps1 file
        """
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check if our modifications are already there
        if "# Ultimate MCP Server customization" in content:
            return
            
        # Add our custom code to ensure PATH is set correctly
        custom_content = """
# Ultimate MCP Server customization
$env:PATH = "$VirtualEnv\\Scripts;" + $env:PATH

# Ensure HOME is set for Git
if (-not (Test-Path env:HOME)) {
    if (Test-Path env:USERPROFILE) {
        $env:HOME = $env:USERPROFILE
    } elseif ((Test-Path env:HOMEDRIVE) -and (Test-Path env:HOMEPATH)) {
        $env:HOME = $env:HOMEDRIVE + $env:HOMEPATH
    }
}
"""
        
        # Find the insertion point (after setting VIRTUAL_ENV)
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('$VirtualEnv = '):
                # Find the end of the initialization block
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == '':
                        # Insert our custom content after the initialization
                        lines.insert(j, custom_content)
                        break
                break
        
        # Write the modified content back
        with open(script_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def _modify_bash_activation(self, script_path):
        """
        Modify the bash activation script to add our custom environment variables.
        
        Args:
            script_path: Path to the activate file
        """
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check if our modifications are already there
        if "# Ultimate MCP Server customization" in content:
            return
            
        # Add our custom code
        custom_content = """
# Ultimate MCP Server customization
export UMCP_SKIP_ENV_CHECK=1
"""
        
        # Find where to insert our code (before the deactivate function definition)
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('deactivate () {'):
                # Insert our custom content before the deactivate function
                lines.insert(i, custom_content)
                break
        
        # Write the modified content back
        with open(script_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def _create_env_file(self, context):
        """
        Create a .env file in the virtual environment to set environment variables.
        
        Args:
            context: The context with environment paths
        """
        env_file = Path(context.env_dir) / '.env'
        
        with open(env_file, 'w') as f:
            f.write("""# Ultimate MCP Server environment variables
# This file is automatically loaded by Ultimate MCP Server
UMCP_SKIP_ENV_CHECK=1
""")
    
    def _install_essential_packages(self, context):
        """
        Install essential packages for Ultimate MCP Server.
        
        Args:
            context: The context with environment executables
        """
        try:
            # Get the pip executable
            pip_exe = Path(context.bin_path) / ('pip.exe' if platform.system() == 'Windows' else 'pip')
            
            # Install essential packages
            subprocess.check_call([
                str(pip_exe), 'install', '-U',
                'pytest-asyncio', 'pytest-cov', 'pytest-mock', 'anyio'
            ])
        except Exception as e:
            print(f"Warning: Failed to install essential packages: {e}")


def create_ultimate_mcp_venv(env_dir, system_site_packages=False, clear=False, with_pip=True):
    """
    Create a virtual environment for Ultimate MCP Server.
    
    Args:
        env_dir: Directory to create the environment in
        system_site_packages: Whether to give access to system site-packages
        clear: Whether to clear the directory if it exists
        with_pip: Whether to include pip in the environment
    
    Returns:
        Path to the created environment
    """
    env_path = Path(env_dir)
    
    # If the directory exists and clear=True, remove it
    if clear and env_path.exists():
        import shutil
        print(f"Clearing existing environment at {env_path}...")
        try:
            shutil.rmtree(env_path)
        except (PermissionError, OSError) as e:
            print(f"Warning: Failed to remove existing directory: {e}")
            print("You may need to run this script with administrator permissions.")
            # On Windows, try a more aggressive approach
            if platform.system() == 'Windows':
                try:
                    os.system(f'rmdir /S /Q "{env_path}"')
                except Exception as e:
                    print(f"Failed to remove directory using system command: {e}")
    
    # Check if we can write to the parent directory
    parent_dir = env_path.parent
    try:
        test_file = parent_dir / ".venv_write_test"
        with open(test_file, 'w') as f:
            f.write("test")
        test_file.unlink()  # Remove the test file
    except (PermissionError, OSError):
        print(f"Warning: No write access to {parent_dir}")
        print("You may need to run this script with administrator permissions.")
    
    try:
        # Create the custom builder
        builder = UltimateMCPEnvBuilder(
            system_site_packages=system_site_packages,
            clear=clear,
            with_pip=with_pip,
            upgrade_deps=True  # Upgrade pip to the latest version
        )
        
        # Create the environment
        builder.create(env_dir)
        
        # Verify the environment was created correctly
        if platform.system() == 'Windows':
            python_exe = env_path / 'Scripts' / 'python.exe'
        else:
            python_exe = env_path / 'bin' / 'python'
            
        if not python_exe.exists():
            print(f"Warning: Python executable not found at {python_exe}")
            print("The environment may not have been created correctly.")
            print("Falling back to standard venv module...")
            
            # Fallback to standard venv module
            import venv
            venv.create(env_dir, with_pip=with_pip, clear=clear, 
                       system_site_packages=system_site_packages,
                       upgrade_deps=True)
            
            # Check again
            if not python_exe.exists():
                print(f"ERROR: Failed to create virtual environment at {env_path}")
                print("Please try creating it manually:")
                print(f"python -m venv {env_path} {'--system-site-packages' if system_site_packages else ''}")
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        print("Falling back to standard venv module...")
        
        try:
            # Fallback to standard venv module
            import venv
            venv.create(env_dir, with_pip=with_pip, clear=clear, 
                       system_site_packages=system_site_packages,
                       upgrade_deps=True)
        except Exception as e2:
            print(f"Failed to create environment with standard venv module: {e2}")
            print("Please try creating it manually:")
            print(f"python -m venv {env_path} {'--system-site-packages' if system_site_packages else ''}")
    
    return env_path


if __name__ == '__main__':
    # Allow this script to be run directly to create a virtual environment
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create a virtual environment for Ultimate MCP Server'
    )
    parser.add_argument(
        'env_dir', help='Directory to create the environment in'
    )
    parser.add_argument(
        '--system-site-packages', action='store_true',
        help='Give access to the system site-packages'
    )
    parser.add_argument(
        '--clear', action='store_true',
        help='Clear the environment directory if it exists'
    )
    parser.add_argument(
        '--without-pip', action='store_true',
        help='Don\'t install pip in the environment'
    )
    
    args = parser.parse_args()
    
    print(f"Creating Ultimate MCP Server virtual environment in {args.env_dir}...")
    env_path = create_ultimate_mcp_venv(
        args.env_dir,
        system_site_packages=args.system_site_packages,
        clear=args.clear,
        with_pip=not args.without_pip
    )
    
    print(f"Virtual environment created successfully at {env_path}")
    print("\nTo activate the environment:")
    if platform.system() == 'Windows':
        print(f"    {env_path}\\Scripts\\activate.bat  (for cmd.exe)")
        print(f"    {env_path}\\Scripts\\Activate.ps1  (for PowerShell)")
    else:
        print(f"    source {env_path}/bin/activate  (for bash/zsh)") 