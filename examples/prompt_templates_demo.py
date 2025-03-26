#!/usr/bin/env python
"""Prompt templates and repository demonstration for LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.prompts import (
    get_prompt_repository,
    PromptTemplate,
    render_prompt,
    render_prompt_template
)
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.prompt_templates")


async def demonstrate_prompt_templates():
    """Demonstrate prompt template creation and rendering."""
    logger.info("Starting prompt template demonstration", emoji_key="start")
    
    # Simple prompt template
    template_text = """
You are an expert in {{field}}. 
Please explain {{concept}} in simple terms that a {{audience}} could understand.
"""

    # Create a prompt template
    template = PromptTemplate(
        template=template_text,
        template_id="simple_explanation",
        description="A template for generating simple explanations of concepts"
    )
    
    # Display template details
    logger.info(
        f"Created prompt template: {template.template_id}",
        emoji_key="template",
        format=template.format
    )
    
    # Render the template with variables
    variables = {
        "field": "artificial intelligence",
        "concept": "neural networks",
        "audience": "high school student"
    }
    
    # Render using helper function
    rendered_prompt = template.render(variables)
    
    logger.info(
        "Template rendered successfully",
        emoji_key="success",
        variables=list(variables.keys())
    )
    
    # Display rendered template
    print("\n" + "-" * 80)
    print("TEMPLATE:")
    print(template_text)
    print("\nVARIABLES:")
    for key, value in variables.items():
        print(f"  {key}: {value}")
    print("\nRENDERED PROMPT:")
    print(rendered_prompt)
    print("-" * 80 + "\n")
    
    # Create a more complex template with conditional blocks
    complex_template = """
{% if system_message %}
{{system_message}}
{% else %}
You are a helpful assistant that provides accurate information.
{% endif %}

{% if context %}
Here is some context to help you answer:
{{context}}
{% endif %}

USER: {{query}}

Please respond with:
{% for item in response_items %}
- {{item}}
{% endfor %}
"""
    
    # Create complex template object with properly defined required_vars
    complex_template_obj = PromptTemplate(
        template=complex_template,
        template_id="complex_assistant",
        description="A complex assistant template with conditionals and loops",
        required_vars=["system_message", "query", "response_items", "context"]  # Explicitly define required vars, excluding 'item' which is a loop variable
    )
    
    # Complex variables
    complex_variables = {
        "system_message": "You are an expert in climate science who explains concepts clearly and objectively.",
        "query": "What are the main causes of climate change?",
        "context": """
Recent data shows that global temperatures have risen by about 1.1°C since pre-industrial times.
The IPCC Sixth Assessment Report (2021) states that human activities are unequivocally the main driver
of climate change, primarily through greenhouse gas emissions. CO2 levels have increased by 48% since 
the industrial revolution, reaching levels not seen in at least 800,000 years.
""",
        "response_items": [
            "A summary of the main causes based on scientific consensus",
            "The role of greenhouse gases (CO2, methane, etc.) in climate change",
            "Human activities that contribute most significantly to emissions",
            "Natural vs anthropogenic factors and their relative impact",
            "Regional variations in climate change impacts"
        ]
    }
    
    # Render complex template
    complex_rendered = complex_template_obj.render(complex_variables)
    
    logger.info(
        "Complex template rendered successfully",
        emoji_key="success",
        template_id=complex_template_obj.template_id
    )
    
    # Display complex template rendering
    print("\n" + "-" * 80)
    print("COMPLEX TEMPLATE RENDERING")
    print("-" * 80)
    print("RENDERED RESULT:")
    print(complex_rendered)
    print("-" * 80 + "\n")
    
    # Demonstrate rendering with missing/partial variables 
    # (template should use defaults for missing values)
    missing_variables = {
        "query": "How can individuals reduce their carbon footprint?",
        "response_items": [
            "Daily lifestyle changes with significant impact",
            "Transportation choices and alternatives",
            "Home energy consumption reduction strategies"
        ]
        # system_message and context are intentionally missing to demonstrate fallback behavior
    }
    
    try:
        # This might fail due to missing required variables
        missing_rendered = complex_template_obj.render(missing_variables)
        
        logger.info(
            "Template rendered with missing variables",
            emoji_key="info",
            missing=["system_message", "context"]
        )
        
        # Display rendering with missing variables
        print("\n" + "-" * 80)
        print("TEMPLATE WITH MISSING VARIABLES")
        print("-" * 80)
        print("RENDERED RESULT:")
        print(missing_rendered)
        print("-" * 80 + "\n")
    except ValueError as e:
        logger.warning(f"Could not render with missing variables: {str(e)}", emoji_key="warning")
    
    return template, complex_template_obj


async def demonstrate_prompt_repository():
    """Demonstrate saving and retrieving templates from repository."""
    logger.info("Starting prompt repository demonstration", emoji_key="start")
    
    # Get repository
    repo = get_prompt_repository()
    
    # Check repository path
    logger.info(f"Prompt repository path: {repo.base_dir}", emoji_key="info")
    
    # List existing prompts (if any)
    prompts = await repo.list_prompts()
    if prompts:
        logger.info(f"Found {len(prompts)} existing prompts: {', '.join(prompts)}", emoji_key="info")
    else:
        logger.info("No existing prompts found in repository", emoji_key="info")
    
    # Create a new prompt template for saving
    translation_template = """
Translate the following {{source_language}} text into {{target_language}}:

TEXT: {{text}}

The translation should be:
- Accurate and faithful to the original
- Natural in the target language
- Preserve the tone and style of the original

TRANSLATION:
"""
    
    template = PromptTemplate(
        template=translation_template,
        template_id="translation_prompt",
        description="A template for translation tasks",
        metadata={
            "author": "LLM Gateway",
            "version": "1.0",
            "supported_languages": ["English", "Spanish", "French", "German", "Japanese"]
        }
    )
    
    # Save to repository
    template_dict = template.to_dict()
    
    logger.info(
        f"Saving template '{template.template_id}' to repository",
        emoji_key="save",
        metadata=template.metadata
    )
    
    save_result = await repo.save_prompt(template.template_id, template_dict)
    
    if save_result:
        logger.success(
            f"Template '{template.template_id}' saved successfully",
            emoji_key="success"
        )
    else:
        logger.error(
            f"Failed to save template '{template.template_id}'",
            emoji_key="error"
        )
        return
    
    # Retrieve the saved template
    logger.info(f"Retrieving template '{template.template_id}' from repository", emoji_key="loading")
    
    retrieved_dict = await repo.get_prompt(template.template_id)
    
    if retrieved_dict:
        # Convert back to PromptTemplate object
        retrieved_template = PromptTemplate.from_dict(retrieved_dict)
        
        logger.success(
            f"Retrieved template '{retrieved_template.template_id}' successfully",
            emoji_key="success",
            metadata=retrieved_template.metadata
        )
        
        # Render the retrieved template
        variables = {
            "source_language": "English",
            "target_language": "Spanish",
            "text": "Machine learning is transforming how we interact with technology."
        }
        
        rendered = retrieved_template.render(variables)
        
        # Display rendered template
        print("\n" + "-" * 80)
        print("RETRIEVED AND RENDERED TEMPLATE")
        print("-" * 80)
        print("TEMPLATE ID: " + retrieved_template.template_id)
        print("DESCRIPTION: " + retrieved_template.description)
        print("METADATA: " + str(retrieved_template.metadata))
        print("\nRENDERED RESULT:")
        print(rendered)
        print("-" * 80 + "\n")
        
    else:
        logger.error(
            f"Failed to retrieve template '{template.template_id}'",
            emoji_key="error"
        )
    
    # List prompts again to confirm addition
    updated_prompts = await repo.list_prompts()
    logger.info(
        f"Repository now contains {len(updated_prompts)} prompts: {', '.join(updated_prompts)}",
        emoji_key="info"
    )
    
    # Optionally: Delete the prompt at the end of demo
    # Uncomment to keep prompt in repository for future use
    delete_result = await repo.delete_prompt(template.template_id)
    if delete_result:
        logger.info(
            f"Deleted template '{template.template_id}' from repository",
            emoji_key="cleaning"
        )


async def demonstrate_llm_with_templates():
    """Demonstrate using templates with LLM for completions."""
    logger.info("Starting LLM with templates demonstration", emoji_key="start")
    
    # Get OpenAI provider
    api_key = decouple_config("OPENAI_API_KEY", default=None)
    provider = get_provider(Provider.OPENAI.value, api_key=api_key)
    await provider.initialize()
    
    # Create a template for question answering
    qa_template = """
You are a concise and helpful assistant.

QUESTION: {{question}}

Please provide an accurate answer that is helpful and direct.
If you need to make any assumptions, state them clearly.

ANSWER:
"""
    
    # Create template object
    qa_template_obj = PromptTemplate(
        template=qa_template,
        template_id="qa_template"
    )
    
    # Questions to ask
    questions = [
        "What is the capital of France?",
        "How does a transformer neural network work?"
    ]
    
    for question in questions:
        # Prepare variables
        variables = {"question": question}
        
        # Render template
        rendered_prompt = qa_template_obj.render(variables)
        
        logger.info(
            f"Generating completion for question: '{question}'",
            emoji_key="question"
        )
        
        # Generate completion
        start_time = time.time()
        result = await provider.generate_completion(
            prompt=rendered_prompt,
            temperature=0.3,
            max_tokens=150
        )
        completion_time = time.time() - start_time
        
        # Display results
        logger.success(
            f"Completion generated in {completion_time:.2f}s",
            emoji_key="success",
            tokens=f"{result.input_tokens} input, {result.output_tokens} output",
            cost=result.cost
        )
        
        print("\n" + "-" * 80)
        print(f"QUESTION: {question}")
        print("\nANSWER:")
        print(result.text.strip())
        print("-" * 80 + "\n")
    
    # Create a chain of templates for a more complex workflow
    system_template = "You are an {{role}} who {{expertise}}."
    
    user_template = """
I need information about {{topic}}.
Specifically, I want to know about {{aspect}}.
"""
    
    # Create a combined template
    combined_template = f"""
{system_template}

{user_template}

Please provide a {{response_type}} response, focusing on the most important points.
"""
    
    combined_obj = PromptTemplate(
        template=combined_template,
        template_id="combined_template"
    )
    
    # Variables for the combined template
    combined_variables = {
        "role": "expert historian",
        "expertise": "specializes in ancient civilizations",
        "topic": "the Roman Empire",
        "aspect": "its fall and the key factors that led to it",
        "response_type": "concise"
    }
    
    # Render combined template
    combined_rendered = combined_obj.render(combined_variables)
    
    logger.info(
        "Using combined template for complex workflow",
        emoji_key="template",
        variables=list(combined_variables.keys())
    )
    
    # Generate completion with the combined template
    start_time = time.time()
    result = await provider.generate_completion(
        prompt=combined_rendered,
        temperature=0.3,
        max_tokens=250
    )
    completion_time = time.time() - start_time
    
    # Display results
    logger.success(
        "Complex template completion generated",
        emoji_key="success",
        tokens=f"{result.input_tokens} input, {result.output_tokens} output",
        cost=result.cost,
        time=f"{completion_time:.2f}s"
    )
    
    print("\n" + "-" * 80)
    print("COMPLEX TEMPLATE CHAIN")
    print("-" * 80)
    print("RENDERED TEMPLATE:")
    print(combined_rendered)
    print("\nGENERATED RESPONSE:")
    print(result.text.strip())
    print("-" * 80 + "\n")


async def main():
    """Run prompt templates and repository demonstration."""
    try:
        # First demonstrate basic template operations
        templates = await demonstrate_prompt_templates()
        
        print("\n" + "=" * 80 + "\n")
        
        # Then demonstrate prompt repository
        await demonstrate_prompt_repository()
        
        print("\n" + "=" * 80 + "\n")
        
        # Finally demonstrate using templates with LLMs
        await demonstrate_llm_with_templates()
        
    except Exception as e:
        logger.critical(f"Prompt templates demonstration failed: {str(e)}", emoji_key="critical")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 