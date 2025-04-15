"""Prompt template service for managing and rendering prompt templates."""
import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from llm_gateway.config import get_config
from llm_gateway.exceptions import PromptTemplateError
from llm_gateway.utils.logging import get_logger

logger = get_logger(__name__)

# Singleton instance
_prompt_service = None


def get_prompt_service():
    """Get the global prompt service instance."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service


class PromptService:
    """Service for managing and rendering prompt templates."""
    
    def __init__(self):
        """Initialize the prompt service.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates: Dict[str, str] = {}
        try:
            config = get_config()
            self.templates_dir = config.prompt_templates_directory
            logger.info(f"Initializing PromptService. Looking for templates in: {self.templates_dir}")
            self._load_templates()
        except Exception as e:
            logger.error(f"Failed to initialize PromptService: {e}", exc_info=True)
            # Allow service to exist even if loading fails, get_template will raise errors
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Read templates from files
        self._read_templates()
        logger.info(f"Prompt service initialized with {len(self.templates)} templates")
    
    def _load_templates(self):
        """Loads all .txt files from the templates directory."""
        if not Path(self.templates_dir).is_dir():
            logger.warning(f"Prompt templates directory not found or not a directory: {self.templates_dir}")
            return

        loaded_count = 0
        for filepath in Path(self.templates_dir).glob('*.txt'):
            try:
                template_name = filepath.stem # Use filename without extension as name
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.templates[template_name] = content
                logger.debug(f"Loaded prompt template: {template_name}")
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load prompt template {filepath.name}: {e}")

        if loaded_count > 0:
             logger.info(f"Successfully loaded {loaded_count} prompt templates.")
        else:
             logger.info("No prompt templates found or loaded.")
    
    def _read_templates(self) -> None:
        """Read templates from files in the templates directory."""
        try:
            template_files = list(Path(self.templates_dir).glob("*.json"))
            logger.info(f"Found {len(template_files)} template files")
            
            for template_file in template_files:
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        templates_data = json.load(f)
                    
                    # Add templates from file
                    for template_name, template_content in templates_data.items():
                        if isinstance(template_content, str):
                            self.templates[template_name] = template_content
                        elif isinstance(template_content, dict) and "text" in template_content:
                            self.templates[template_name] = template_content["text"]
                        else:
                            logger.warning(f"Invalid template format for {template_name}")
                            
                    logger.info(f"Loaded templates from {template_file.name}")
                except Exception as e:
                    logger.error(f"Error loading template file {template_file.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading templates: {str(e)}")
    
    def _save_templates(self) -> None:
        """Save all templates to disk."""
        try:
            # Group templates by category
            categorized_templates: Dict[str, Dict[str, Any]] = {}
            
            for template_name, template_text in self.templates.items():
                # Extract category from template name (before first _)
                parts = template_name.split("_", 1)
                category = parts[0] if len(parts) > 1 else "general"
                
                if category not in categorized_templates:
                    categorized_templates[category] = {}
                
                categorized_templates[category][template_name] = template_text
            
            # Save each category to its own file
            for category, templates in categorized_templates.items():
                file_path = Path(self.templates_dir) / f"{category}_templates.json"
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(templates, f, indent=2)
                
                logger.info(f"Saved {len(templates)} templates to {file_path.name}")
                
        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")
    
    def get_template(self, template_name: str) -> Optional[str]:
        """Get a prompt template by name.
        
        Args:
            template_name: Template name
            
        Returns:
            Template text or None if not found
        """
        return self.templates.get(template_name)
    
    def get_all_templates(self) -> Dict[str, str]:
        """Get all templates.
        
        Returns:
            Dictionary of template name to template text
        """
        return self.templates.copy()
    
    def register_template(self, template_name: str, template_text: str) -> bool:
        """Register a new template or update an existing one.
        
        Args:
            template_name: Template name
            template_text: Template text
            
        Returns:
            True if successful
        """
        try:
            self.templates[template_name] = template_text
            
            # Schedule template save
            asyncio.create_task(self._async_save_templates())
            
            return True
        except Exception as e:
            logger.error(f"Error registering template {template_name}: {str(e)}")
            return False
    
    async def _async_save_templates(self) -> None:
        """Save templates asynchronously."""
        self._save_templates()
    
    def remove_template(self, template_name: str) -> bool:
        """Remove a template.
        
        Args:
            template_name: Template name
            
        Returns:
            True if removed, False if not found
        """
        if template_name in self.templates:
            del self.templates[template_name]
            
            # Schedule template save
            asyncio.create_task(self._async_save_templates())
            
            return True
        return False
    
    def render_template(
        self, 
        template_name: str, 
        variables: Dict[str, Any]
    ) -> Optional[str]:
        """Render a template with variables.
        
        Args:
            template_name: Template name
            variables: Variables to substitute
            
        Returns:
            Rendered template or None if error
        """
        template = self.get_template(template_name)
        if not template:
            logger.warning(f"Template {template_name} not found")
            return None
        
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in template {template_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {str(e)}")
            return None 

# Global instance
_prompt_manager_instance = None
_prompt_manager_lock = threading.Lock()

def get_prompt_manager() -> PromptService:
    """Gets the singleton PromptManager instance."""
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        with _prompt_manager_lock:
            if _prompt_manager_instance is None:
                _prompt_manager_instance = PromptService()
    return _prompt_manager_instance

# Example Usage
if __name__ == '__main__':
    from llm_gateway.utils.logging import setup_logging

    setup_logging(log_level="DEBUG")

    # Create dummy templates dir and file for example
    EXAMPLE_TEMPLATES_DIR = Path("./temp_prompt_templates_example")
    EXAMPLE_TEMPLATES_DIR.mkdir(exist_ok=True)
    (EXAMPLE_TEMPLATES_DIR / "greeting.txt").write_text("Hello, {{name}}! How are you today?")
    (EXAMPLE_TEMPLATES_DIR / "summary.txt").write_text("Summarize the following text:\n\n{{text}}")

    # Set env var to use this temp dir
    os.environ['GATEWAY_PROMPT_TEMPLATES_DIR'] = str(EXAMPLE_TEMPLATES_DIR.resolve())
    os.environ['GATEWAY_FORCE_CONFIG_RELOAD'] = 'true' # Force reload

    try:
        manager = get_prompt_manager()
        print(f"Templates directory: {manager.templates_dir}")
        print(f"Available templates: {manager.list_templates()}")

        greeting_template = manager.get_template('greeting')
        print(f"Greeting Template: {greeting_template}")

        try:
            manager.get_template('non_existent')
        except PromptTemplateError as e:
            print(f"Caught expected error: {e}")

    finally:
        # Clean up
        import shutil
        shutil.rmtree(EXAMPLE_TEMPLATES_DIR)
        print(f"Cleaned up {EXAMPLE_TEMPLATES_DIR}")
        if 'GATEWAY_PROMPT_TEMPLATES_DIR' in os.environ:
            del os.environ['GATEWAY_PROMPT_TEMPLATES_DIR']
        if 'GATEWAY_FORCE_CONFIG_RELOAD' in os.environ:
            del os.environ['GATEWAY_FORCE_CONFIG_RELOAD']