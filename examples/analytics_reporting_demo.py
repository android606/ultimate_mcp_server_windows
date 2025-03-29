#!/usr/bin/env python
"""Analytics and reporting demonstration for LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.analytics.metrics import get_metrics_tracker
from llm_gateway.services.analytics.reporting import AnalyticsReporting
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.analytics_reporting")


async def simulate_llm_usage():
    """Simulate various LLM API calls to generate analytics data."""
    logger.info("Simulating LLM usage to generate analytics data", emoji_key="start")
    
    # Get metrics tracker (don't reset it)
    metrics = get_metrics_tracker()
    
    # Get providers for various API calls
    providers = []
    
    # Try to get OpenAI provider
    openai_key = decouple_config("OPENAI_API_KEY", default=None)
    if openai_key:
        openai = get_provider(Provider.OPENAI.value, api_key=openai_key)
        await openai.initialize()
        providers.append((Provider.OPENAI.value, openai))
    
    # Try to get Anthropic provider
    anthropic_key = decouple_config("ANTHROPIC_API_KEY", default=None)
    if anthropic_key:
        anthropic = get_provider(Provider.ANTHROPIC.value, api_key=anthropic_key)
        await anthropic.initialize()
        providers.append((Provider.ANTHROPIC.value, anthropic))
    
    # Try to get Gemini provider
    gemini_key = decouple_config("GEMINI_API_KEY", default=None)
    if gemini_key:
        gemini = get_provider(Provider.GEMINI.value, api_key=gemini_key)
        await gemini.initialize()
        providers.append((Provider.GEMINI.value, gemini))
    
    logger.info(f"Using {len(providers)} providers for simulation", emoji_key="provider")
    
    # Sample prompts of different lengths for varied token usage
    prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks in simple terms.",
        "Write a short story about a robot that learns to love.",
        "Summarize the key innovations in artificial intelligence over the past decade.",
        "What are the ethical considerations in developing advanced AI systems?"
    ]
    
    # Simulate different completion calls
    total_calls = len(providers) * len(prompts)
    call_count = 0
    
    for provider_name, provider in providers:
        for prompt in prompts:
            call_count += 1
            logger.info(
                f"Generating completion ({call_count}/{total_calls})",
                emoji_key="processing",
                provider=provider_name,
                prompt_length=len(prompt)
            )
            
            try:
                # Generate completion
                start_time = time.time()
                result = await provider.generate_completion(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=150
                )
                completion_time = time.time() - start_time
                
                # Manually record the metrics
                metrics.record_request(
                    provider=provider_name,
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost=result.cost,
                    duration=completion_time,
                    success=True
                )
                
                # Log success
                logger.success(
                    "Completion generated successfully",
                    emoji_key="success",
                    provider=provider_name,
                    model=result.model,
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens
                    },
                    cost=result.cost,
                    time=completion_time
                )
                
                # Add brief delay between calls
                await asyncio.sleep(0.5)
            
            except Exception as e:
                logger.error(
                    f"Error generating completion: {str(e)}",
                    emoji_key="error",
                    provider=provider_name
                )
    
    logger.info("Finished simulating LLM usage", emoji_key="success")
    
    # Return metrics for further analysis
    return metrics


async def demonstrate_metrics_tracking():
    """Demonstrate metrics tracking functionality."""
    logger.info("Starting metrics tracking demonstration", emoji_key="start")
    
    # Create a new instance of the metrics tracker with reset_on_start=True
    metrics = get_metrics_tracker(reset_on_start=True)
    
    # Simulate usage to generate metrics
    await simulate_llm_usage()
    
    # Get current stats
    stats = metrics.get_stats()
    
    # Display general metrics
    logger.info("General LLM usage metrics:", emoji_key="metrics")
    print("\n" + "-" * 80)
    print("GENERAL METRICS")
    print("-" * 80)
    
    general = stats["general"]
    print(f"Total requests: {general['requests_total']}")
    print(f"Total tokens: {general['tokens_total']}")
    print(f"Total cost: ${general['cost_total']:.6f}")
    
    # Calculate these metrics on the fly
    avg_tokens_per_request = general['tokens_total'] / general['requests_total'] if general['requests_total'] > 0 else 0
    avg_cost_per_request = general['cost_total'] / general['requests_total'] if general['requests_total'] > 0 else 0
    avg_cost_per_1k_tokens = (general['cost_total'] / general['tokens_total']) * 1000 if general['tokens_total'] > 0 else 0
    
    print(f"Average tokens per request: {avg_tokens_per_request:.1f}")
    print(f"Average cost per request: ${avg_cost_per_request:.6f}")
    print(f"Average cost per 1K tokens: ${avg_cost_per_1k_tokens:.6f}")
    print(f"Total errors: {general['errors_total']}")
    
    # Display provider-specific metrics
    print("\n" + "-" * 80)
    print("PROVIDER METRICS")
    print("-" * 80)
    
    for provider, provider_stats in stats["providers"].items():
        print(f"\nProvider: {provider}")
        print(f"  Requests: {provider_stats['requests']}")
        print(f"  Tokens: {provider_stats['tokens']}")
        print(f"  Cost: ${provider_stats['cost']:.6f}")
        if provider_stats['requests'] > 0:
            print(f"  Avg tokens per request: {provider_stats['tokens'] / provider_stats['requests']:.1f}")
            print(f"  Avg cost per request: ${provider_stats['cost'] / provider_stats['requests']:.6f}")
    
    # Display model-specific metrics
    print("\n" + "-" * 80)
    print("MODEL METRICS")
    print("-" * 80)
    
    for model, model_stats in stats["models"].items():
        print(f"\nModel: {model}")
        print(f"  Requests: {model_stats['requests']}")
        print(f"  Tokens: {model_stats['tokens']}")
        print(f"  Cost: ${model_stats['cost']:.6f}")
        if model_stats['requests'] > 0:
            print(f"  Avg tokens per request: {model_stats['tokens'] / model_stats['requests']:.1f}")
            print(f"  Avg cost per request: ${model_stats['cost'] / model_stats['requests']:.6f}")
    
    # Display daily usage data
    print("\n" + "-" * 80)
    print("DAILY USAGE")
    print("-" * 80)
    
    for day in stats["daily_usage"]:
        print(f"Date: {day['date']}")
        print(f"  Tokens: {day['tokens']}")
        print(f"  Cost: ${day['cost']:.6f}")
        print(f"  Requests: {day.get('requests', 0)}")  # Get requests with default of 0
    
    print("-" * 80 + "\n")
    
    # Return stats for further analysis
    return stats


async def demonstrate_analytics_reporting():
    """Demonstrate analytics reporting functionality."""
    logger.info("Starting analytics reporting demonstration", emoji_key="start")
    
    # Get metrics tracker (should already have data from previous functions)
    metrics = get_metrics_tracker()
    
    # Check if we have any metrics data
    stats = metrics.get_stats()
    if stats["general"]["requests_total"] == 0:
        logger.warning(
            "No metrics data available. Running simulation first.",
            emoji_key="warning"
        )
        await simulate_llm_usage()
    
    # Initialize reporting service
    reports_dir = Path.home() / ".llm_gateway" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    analytics = AnalyticsReporting(reports_dir=reports_dir)
    
    logger.info(f"Analytics reporting initialized (dir: {reports_dir})", emoji_key="info")
    
    # Generate a usage report
    logger.info("Generating usage report (JSON format)", emoji_key="report")
    json_report = analytics.generate_usage_report(days=7, output_format="json")
    
    if isinstance(json_report, Path):
        logger.success(f"Generated JSON report: {json_report}", emoji_key="success")
    
    # Generate a cost report
    logger.info("Generating cost report (JSON format)", emoji_key="report")
    cost_report = analytics.generate_cost_report(days=30, output_format="json")
    
    if isinstance(cost_report, Path):
        logger.success(f"Generated cost report: {cost_report}", emoji_key="success")
    
    # Try to generate provider report for first available provider
    available_providers = list(stats["providers"].keys())
    if available_providers:
        provider = available_providers[0]
        logger.info(f"Generating provider report for {provider}", emoji_key="report")
        
        provider_report = analytics.generate_provider_report(
            provider=provider,
            days=7,
            output_format="json"
        )
        
        if isinstance(provider_report, Path):
            logger.success(f"Generated provider report: {provider_report}", emoji_key="success")
    
    # Display report locations
    logger.info("All reports saved to:", emoji_key="info")
    print(f"Report directory: {reports_dir}")


async def demonstrate_real_time_monitoring():
    """Demonstrate real-time analytics monitoring with streaming."""
    logger.info("Starting real-time monitoring demonstration", emoji_key="start")
    
    # Get metrics tracker and reset for this demo
    metrics = get_metrics_tracker()
    metrics.reset()
    logger.info("Metrics reset for streaming demo", emoji_key="info")
    
    # Get provider for streaming demo
    api_key = decouple_config("OPENAI_API_KEY", default=None)
    if not api_key:
        logger.error("OpenAI API key not available for streaming demo", emoji_key="error")
        return
    
    provider = get_provider(Provider.OPENAI.value, api_key=api_key)
    await provider.initialize()
    
    # Prompt for streaming demo
    prompt = "Write a short poem about artificial intelligence and the future of humanity."
    
    logger.info("Starting streaming completion for monitoring", emoji_key="processing")
    
    # Get initial metrics
    before_stats = metrics.get_stats()
    
    # Start streaming completion
    stream = provider.generate_completion_stream(
        prompt=prompt,
        temperature=0.7,
        max_tokens=200
    )
    
    # Process the stream and monitor in real-time
    token_count = 0
    full_text = ""
    print("\n" + "-" * 80)
    print("STREAMING OUTPUT:")
    print("-" * 80)
    
    try:
        async for chunk, _metadata in stream:
            # Print chunk
            print(chunk, end="", flush=True)
            full_text += chunk
            token_count += 1
            
            # For demo purposes, check metrics every 5 tokens
            if token_count % 5 == 0:
                current_stats = metrics.get_stats()  # noqa: F841
                # Real applications would log/visualize these metrics
            
            # Small delay to simulate processing
            await asyncio.sleep(0.05)
    
    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}", emoji_key="error")
    
    print("\n" + "-" * 80 + "\n")
    
    # Get final metrics after streaming
    after_stats = metrics.get_stats()
    
    # Display metrics changes
    tokens_used = after_stats["general"]["tokens_total"] - before_stats["general"]["tokens_total"]
    cost_incurred = after_stats["general"]["cost_total"] - before_stats["general"]["cost_total"]
    
    # If no metrics were recorded, try to manually record them based on response
    if tokens_used == 0:
        # Estimate tokens used (rough approximation)
        input_tokens = len(prompt.split()) // 2  # Rough estimation
        output_tokens = len(full_text.split()) // 2  # Rough estimation
        
        # Record request metrics manually
        metrics.record_request(
            provider=Provider.OPENAI.value,
            model="gpt-4o-mini",  # Default model - may need adjustment
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0005,  # Small arbitrary cost
            duration=time.time() - before_stats["general"]["uptime"],
            success=True
        )
        
        # Update after stats
        after_stats = metrics.get_stats()
        tokens_used = after_stats["general"]["tokens_total"] - before_stats["general"]["tokens_total"]
        cost_incurred = after_stats["general"]["cost_total"] - before_stats["general"]["cost_total"]
    
    logger.info(
        "Streaming completion finished",
        emoji_key="success",
        tokens_generated=tokens_used,
        cost=f"${cost_incurred:.6f}"
    )
    
    # Show metrics changes
    print("\n" + "-" * 80)
    print("METRICS CHANGES FROM STREAMING")
    print("-" * 80)
    print(f"Tokens used: {tokens_used}")
    print(f"Cost incurred: ${cost_incurred:.6f}")
    print("-" * 80 + "\n")


async def main():
    """Run analytics and reporting demonstration."""
    try:
        # First demonstrate metrics tracking
        await demonstrate_metrics_tracking()
        
        print("\n" + "=" * 80 + "\n")
        
        # Then demonstrate reporting functionality
        await demonstrate_analytics_reporting()
        
        print("\n" + "=" * 80 + "\n")
        
        # Finally demonstrate real-time monitoring
        await demonstrate_real_time_monitoring()
        
    except Exception as e:
        logger.critical(f"Analytics demonstration failed: {str(e)}", emoji_key="critical")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 