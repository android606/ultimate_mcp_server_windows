"""Integration tests for LLM Gateway workflows."""
import json
from typing import Any, Dict, List, Optional

import pytest
from pytest import MonkeyPatch

from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway
from llm_gateway.tools.meta import MetaTools
from llm_gateway.utils import get_logger

logger = get_logger("test.integration.workflow")


@pytest.fixture
async def workflow_gateway(monkeypatch: MonkeyPatch) -> Gateway:
    """Create a gateway with mocked tool execution for workflow testing."""
    gateway = Gateway(name="workflow-test")
    
    # Mock tool execution
    async def mock_execute(tool_name, params):
        # Simulate document processing tools
        if tool_name == "summarize_document":
            return {
                "summary": f"Summary of document: {params['document'][:30]}...",
                "model": params.get("model", "default-model"),
                "provider": params.get("provider", "default-provider"),
                "tokens": {"input": 100, "output": 50, "total": 150},
                "cost": 0.01,
                "processing_time": 0.5
            }
        elif tool_name == "extract_entities":
            return {
                "data": {
                    "person": ["John Doe", "Jane Smith"],
                    "organization": ["Acme Corp", "Example Inc."],
                    "location": ["New York", "London"]
                },
                "model": params.get("model", "default-model"),
                "provider": params.get("provider", "default-provider"),
                "tokens": {"input": 80, "output": 40, "total": 120},
                "cost": 0.008,
                "processing_time": 0.3
            }
        elif tool_name == "chunk_document":
            document = params["document"]
            # Simple chunking for testing
            chunks = []
            chunk_size = params.get("chunk_size", 1000)
            for i in range(0, len(document), chunk_size):
                chunks.append(document[i:i + chunk_size])
            return {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "method": params.get("method", "token"),
                "processing_time": 0.1
            }
        elif tool_name == "extract_json":
            return {
                "data": {"extracted": "data", "from": params["text"][:20]},
                "model": params.get("model", "default-model"),
                "provider": params.get("provider", "default-provider"),
                "tokens": {"input": 50, "output": 30, "total": 80},
                "cost": 0.004,
                "processing_time": 0.2
            }
        # Delegate to actual implementation for meta tools
        else:
            # For unknown tools, return mock response
            return {
                "result": f"Mock result for {tool_name}",
                "params": params
            }
            
    monkeypatch.setattr(gateway.mcp, "execute", mock_execute)
    
    return gateway


@pytest.fixture
def meta_tools(workflow_gateway: Gateway) -> MetaTools:
    """Initialize MetaTools for testing."""
    return MetaTools(workflow_gateway)


class TestWorkflows:
    """Tests for workflow execution."""
    
    async def test_analyze_task(self, meta_tools: MetaTools, workflow_gateway: Gateway):
        """Test task analysis."""
        logger.info("Testing analyze_task", emoji_key="test")
        
        # Register tools
        meta_tools._register_tools()
        
        # Execute analyze_task
        result = await workflow_gateway.mcp.execute("analyze_task", {
            "task_description": "Summarize this document about quantum physics",
            "available_providers": ["openai", "anthropic", "gemini"],
            "analyze_features": True,
            "analyze_cost": True
        })
        
        # Check result structure
        assert "task_type" in result
        assert result["task_type"] == "summarization"  # Should detect summarization
        assert "required_features" in result
        assert "providers" in result
        assert "recommendations" in result
        
    async def test_delegate_task(self, meta_tools: MetaTools, workflow_gateway: Gateway, monkeypatch: MonkeyPatch):
        """Test task delegation."""
        logger.info("Testing delegate_task", emoji_key="test")
        
        # Mock the analyze_task execution
        async def mock_analyze_task(**params):
            return {
                "task_type": "summarization",
                "required_features": ["instruction-following"],
                "providers": {
                    "openai": [
                        {"id": "gpt-4o-mini", "provider": "openai"}
                    ],
                    "gemini": [
                        {"id": "gemini-2.0-flash-lite", "provider": "gemini"}
                    ]
                },
                "recommendations": {
                    "lowest_cost": {"provider": "gemini", "model": "gemini-2.0-flash-lite"},
                    "best_quality": {"provider": "openai", "model": "gpt-4o-mini"},
                    "balanced": {"provider": "gemini", "model": "gemini-2.0-flash-lite"}
                }
            }
            
        # Override the function import in the meta tools module
        import types
        meta_tools.analyze_task = types.MethodType(mock_analyze_task, meta_tools)
        
        # Execute delegate_task
        result = await workflow_gateway.mcp.execute("delegate_task", {
            "task_description": "Summarize this document",
            "prompt": "Provide a concise summary of this scientific paper...",
            "optimization_criteria": "cost"
        })
        
        # Check result
        assert "text" in result
        assert "provider" in result
        assert "model" in result
        assert "cost" in result
        assert "task_type" in result
        assert result["task_type"] == "summarization"
        assert result["provider"] == "gemini"  # Should select lowest cost
        
    async def test_execute_workflow(self, meta_tools: MetaTools, workflow_gateway: Gateway):
        """Test workflow execution."""
        logger.info("Testing execute_workflow", emoji_key="test")
        
        # Define a test workflow
        workflow_steps = [
            {
                "name": "Document Chunking",
                "operation": "chunk",
                "input_from": "original",
                "output_as": "chunks"
            },
            {
                "name": "Summarization",
                "operation": "summarize",
                "provider": "gemini",
                "model": "gemini-2.0-flash-lite",
                "input_from": "original",
                "output_as": "summary"
            },
            {
                "name": "Entity Extraction",
                "operation": "extract_entities",
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "input_from": "original",
                "output_as": "entities"
            }
        ]
        
        # Execute workflow
        result = await workflow_gateway.mcp.execute("execute_workflow", {
            "workflow_steps": workflow_steps,
            "initial_input": "This is a test document for workflow execution",
            "max_concurrency": 2
        })
        
        # Check result
        assert "outputs" in result
        assert "final_output" in result
        assert "total_cost" in result
        assert "processing_time" in result
        assert "completed_steps" in result
        assert result["completed_steps"] == 3
        
        # Check that all steps produced outputs
        assert "chunks" in result["outputs"]
        assert "summary" in result["outputs"]
        assert "entities" in result["outputs"]
        
    async def test_quality_check(self, meta_tools: MetaTools, workflow_gateway: Gateway, monkeypatch: MonkeyPatch):
        """Test quality check tool."""
        logger.info("Testing quality_check", emoji_key="test")
        
        # Mock completion provider for quality check
        class MockProvider:
            async def generate_completion(self, prompt, **kwargs):
                return type("MockResponse", (), {
                    "text": json.dumps({
                        "criteria": {
                            "accuracy": {"score": 8, "comments": "Good accuracy"},
                            "completeness": {"score": 7, "comments": "Missing some details"},
                            "coherence": {"score": 9, "comments": "Very coherent"}
                        },
                        "overall_score": 8.0,
                        "summary": "Good quality overall",
                        "strengths": ["Well structured", "Clear explanation"],
                        "weaknesses": ["Minor omissions"],
                        "recommendations": ["Add more context"]
                    }),
                    "model": kwargs.get("model", "claude-3-sonnet-20240229"),
                    "provider": "anthropic",
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                    "cost": 0.02
                })
                
            async def initialize(self):
                return True
        
        # Mock get_provider to return our mock
        monkeypatch.setattr(
            "llm_gateway.core.providers.base.get_provider", 
            lambda *args, **kwargs: MockProvider()
        )
        
        # Execute quality_check
        result = await workflow_gateway.mcp.execute("quality_check", {
            "text": "This is a test document for quality checking",
            "original_task": "Write a summary of quantum physics",
            "quality_criteria": ["accuracy", "completeness", "coherence"],
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229"
        })
        
        # Check result
        assert "evaluation" in result
        assert "model" in result
        assert "provider" in result
        assert "tokens" in result
        assert "cost" in result
        assert "processing_time" in result
        
        # Check evaluation structure
        eval_data = result["evaluation"]
        assert "criteria" in eval_data
        assert "overall_score" in eval_data
        assert "summary" in eval_data
        assert "strengths" in eval_data
        assert "weaknesses" in eval_data
        assert "recommendations" in eval_data
        
        # Check specific criteria
        assert "accuracy" in eval_data["criteria"]
        assert "completeness" in eval_data["criteria"]
        assert "coherence" in eval_data["criteria"]


class TestOptimizationWorkflows:
    """Tests for optimization-focused workflows."""
    
    async def test_optimize_prompt(self, meta_tools: MetaTools, workflow_gateway: Gateway, monkeypatch: MonkeyPatch):
        """Test prompt optimization."""
        logger.info("Testing optimize_prompt", emoji_key="test")
        
        # Mock completion provider for optimization
        class MockProvider:
            async def generate_completion(self, prompt, **kwargs):
                return type("MockResponse", (), {
                    "text": json.dumps({
                        "optimized_prompt": "Improved prompt: " + prompt.split("\n\n")[-1],
                        "explanation": "Made the prompt more specific and added constraints",
                        "key_improvements": ["Added specificity", "Removed ambiguity", "Added examples"]
                    }),
                    "model": kwargs.get("model", "gpt-4o"),
                    "provider": "openai",
                    "input_tokens": 150,
                    "output_tokens": 80,
                    "total_tokens": 230,
                    "cost": 0.015
                })
                
            async def initialize(self):
                return True
        
        # Mock get_provider to return our mock
        monkeypatch.setattr(
            "llm_gateway.core.providers.base.get_provider", 
            lambda *args, **kwargs: MockProvider()
        )
        
        # Execute optimize_prompt
        result = await workflow_gateway.mcp.execute("optimize_prompt", {
            "prompt": "Explain quantum physics",
            "target_model": "claude-3-haiku-20240307",
            "optimization_type": "factual",
            "provider": "openai",
            "model": "gpt-4o"
        })
        
        # Check result
        assert "optimization" in result
        assert "provider" in result
        assert "model" in result
        assert "target_model" in result
        assert "optimization_type" in result
        assert "tokens" in result
        assert "cost" in result
        
        # Check optimization structure
        opt_data = result["optimization"]
        assert "optimized_prompt" in opt_data
        assert "explanation" in opt_data
        assert "key_improvements" in opt_data
        assert len(opt_data["key_improvements"]) > 0
        
    async def test_model_comparison(self, workflow_gateway: Gateway, monkeypatch: MonkeyPatch):
        """Test model comparison workflow."""
        logger.info("Testing model comparison workflow", emoji_key="test")
        
        # Mock multi_completion execution
        async def mock_multi_completion(**params):
            providers = params.get("providers", [])
            results = {}
            
            for provider_config in providers:
                provider = provider_config.get("provider", "openai")
                model = provider_config.get("model", "default-model")
                provider_key = f"{provider}/{model}"
                
                results[provider_key] = {
                    "success": True,
                    "text": f"Response from {provider}/{model}",
                    "model": model,
                    "provider": provider,
                    "tokens": {
                        "input": 100,
                        "output": 50 + providers.index(provider_config) * 10,  # Different outputs
                        "total": 150 + providers.index(provider_config) * 10
                    },
                    "cost": 0.01 + providers.index(provider_config) * 0.005
                }
            
            return {
                "results": results,
                "successful_count": len(providers),
                "total_providers": len(providers)
            }
            
        monkeypatch.setattr(workflow_gateway.mcp, "execute", 
            lambda tool, params: mock_multi_completion(**params) if tool == "multi_completion" else None
        )
        
        # Execute comparison workflow
        result = await workflow_gateway.mcp.execute("multi_completion", {
            "prompt": "Explain the implications of quantum computing for cryptography",
            "providers": [
                {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3},
                {"provider": "anthropic", "model": "claude-3-haiku-20240307", "temperature": 0.3},
                {"provider": "gemini", "model": "gemini-2.0-pro", "temperature": 0.3}
            ]
        })
        
        # Check result
        assert "results" in result
        assert "successful_count" in result
        assert "total_providers" in result
        assert result["successful_count"] == 3
        assert len(result["results"]) == 3
        
        # Check individual results
        for provider_key, provider_result in result["results"].items():
            assert "text" in provider_result
            assert "model" in provider_result
            assert "provider" in provider_result
            assert "tokens" in provider_result
            assert "cost" in provider_result