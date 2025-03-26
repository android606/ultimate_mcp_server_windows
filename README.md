# LLM Gateway MCP Server

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/mpctechdebt/mcp)

**A Model Context Protocol (MCP) server enabling intelligent delegation from high-capability AI agents to cost-effective LLMs**

![Illustration](https://github.com/Dicklesworthstone/llm_gateway_mcp_server/blob/main/F402AFA6-0BE7-4F14-909A-0A3E67E11DF4.png)

[Getting Started](#getting-started) •
[Key Features](#key-features) •
[Usage Examples](#usage-examples) •
[Architecture](#architecture) •

</div>

## What is LLM Gateway?

LLM Gateway is an MCP-native server that enables intelligent task delegation from advanced AI agents like Claude 3.7 Sonnet to more cost-effective models like Gemini Flash 2.0 Lite. It provides a unified interface to multiple Large Language Model (LLM) providers while optimizing for cost, performance, and quality.

### MCP-Native Architecture

The server is built on the [Model Context Protocol (MCP)](https://github.com/mpctechdebt/mcp), making it specifically designed to work with AI agents like Claude. All functionality is exposed through MCP tools that can be directly called by these agents, creating a seamless workflow for AI-to-AI delegation.

### Primary Use Case: AI Agent Task Delegation

The primary design goal of LLM Gateway is to allow sophisticated AI agents like Claude 3.7 Sonnet to intelligently delegate tasks to less expensive models:

```plaintext
                          delegates to
┌─────────────┐ ────────────────────────► ┌───────────────────┐         ┌──────────────┐
│ Claude 3.7  │                           │   LLM Gateway     │ ───────►│ Gemini Flash │
│   (Agent)   │ ◄──────────────────────── │    MCP Server     │ ◄───────│ DeepSeek     │
└─────────────┘      returns results      └───────────────────┘         │ GPT-4o-mini  │
                                                                        └──────────────┘
```

**Example workflow:**

1. Claude identifies that a document needs to be summarized (an expensive operation with Claude)
2. Claude delegates this task to LLM Gateway via MCP tools
3. LLM Gateway routes the summarization task to Gemini Flash (10-20x cheaper than Claude)
4. The summary is returned to Claude for higher-level reasoning and decision-making
5. Claude can then focus its capabilities on tasks that truly require its intelligence

This delegation pattern can save 70-90% on API costs while maintaining output quality.

## Why Use LLM Gateway?

### 🔄 AI-to-AI Task Delegation

The most powerful use case is enabling advanced AI agents to delegate routine tasks to cheaper models:

- Have Claude 3.7 use GPT-4o-mini for initial document summarization
- Let Claude use Gemini 2.0 Flash light for data extraction and transformation
- Allow Claude to orchestrate a multi-stage workflow across different providers
- Enable Claude to choose the right model for each specific sub-task

### 💰 Cost Optimization

API costs for advanced models can be substantial. LLM Gateway helps reduce costs by:

- Routing appropriate tasks to cheaper models (e.g., $0.01/1K tokens vs $0.15/1K tokens)
- Implementing advanced caching to avoid redundant API calls
- Tracking and optimizing costs across providers
- Enabling cost-aware task routing decisions

### 🔄 Provider Abstraction

Avoid provider lock-in with a unified interface:

- Standard API for OpenAI, Anthropic (Claude), Google (Gemini), and DeepSeek
- Consistent parameter handling and response formatting
- Ability to swap providers without changing application code
- Protection against provider-specific outages and limitations

### 📄 Document Processing at Scale

Process large documents efficiently:

- Break documents into semantically meaningful chunks
- Process chunks in parallel across multiple models
- Extract structured data from unstructured text
- Generate summaries and insights from large texts

## Key Features

### MCP Protocol Integration

- **Native MCP Server**: Built on the Model Context Protocol for AI agent integration
- **MCP Tool Framework**: All functionality exposed through standardized MCP tools
- **Tool Composition**: Tools can be combined for complex workflows
- **Tool Discovery**: Support for tool listing and capability discovery

### Intelligent Task Delegation

- **Task Routing**: Analyze tasks and route to appropriate models
- **Provider Selection**: Choose provider based on task requirements
- **Cost-Performance Balancing**: Optimize for cost, quality, or speed
- **Delegation Tracking**: Monitor delegation patterns and outcomes

### Advanced Caching

- **Multi-level Caching**: Multiple caching strategies:
  - Exact match caching
  - Semantic similarity caching
  - Task-aware caching
- **Persistent Cache**: Disk-based persistence with fast in-memory access
- **Cache Analytics**: Track savings and hit rates

### Document Tools

- **Smart Chunking**: Multiple chunking strategies:
  - Token-based chunking
  - Semantic boundary detection
  - Structural analysis
- **Document Operations**:
  - Summarization
  - Entity extraction
  - Question generation
  - Batch processing

### Structured Data Extraction

- **JSON Extraction**: Extract structured JSON with schema validation
- **Table Extraction**: Extract tables in multiple formats
- **Key-Value Extraction**: Extract key-value pairs from text
- **Semantic Schema Inference**: Generate schemas from text

## Usage Examples

### Claude Using LLM Gateway for Document Analysis

This example shows how Claude can use the LLM Gateway to process a document by delegating tasks to cheaper models:

```python
import asyncio
from mcp.client import Client

async def main():
    # Claude would use this client to connect to the LLM Gateway
    client = Client("http://localhost:8000")
    
    # Claude can identify a document that needs processing
    document = "... large document content ..."
    
    # Step 1: Claude delegates document chunking
    chunks_response = await client.tools.chunk_document(
        document=document,
        chunk_size=1000,
        method="semantic"
    )
    print(f"Document divided into {chunks_response['chunk_count']} chunks")
    
    # Step 2: Claude delegates summarization to a cheaper model
    summaries = []
    total_cost = 0
    for i, chunk in enumerate(chunks_response["chunks"]):
        # Use Gemini Flash (much cheaper than Claude)
        summary = await client.tools.summarize_document(
            document=chunk,
            provider="gemini",
            model="gemini-2.0-flash-lite",
            format="paragraph"
        )
        summaries.append(summary["summary"])
        total_cost += summary["cost"]
        print(f"Processed chunk {i+1} with cost ${summary['cost']:.6f}")
    
    # Step 3: Claude delegates entity extraction to another cheap model
    entities = await client.tools.extract_entities(
        document=document,
        entity_types=["person", "organization", "location", "date"],
        provider="openai",
        model="gpt-3.5-turbo"
    )
    total_cost += entities["cost"]
    
    print(f"Total delegation cost: ${total_cost:.6f}")
    # Claude would now process these summaries and entities using its advanced capabilities
    
    # Close the client when done
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Provider Comparison for Decision Making

```python
# Claude can compare outputs from different providers for critical tasks
responses = await client.tools.multi_completion(
    prompt="Explain the implications of quantum computing for cryptography.",
    providers=[
        {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3},
        {"provider": "anthropic", "model": "claude-3-haiku-20240307", "temperature": 0.3},
        {"provider": "gemini", "model": "gemini-2.0-pro", "temperature": 0.3}
    ]
)

# Claude could analyze these responses and decide which is most accurate
for provider_key, result in responses["results"].items():
    if result["success"]:
        print(f"{provider_key} Cost: ${result['cost']}")
```

### Cost-Optimized Workflow

```python
# Claude can define and execute complex multi-stage workflows
workflow = [
    {
        "name": "Initial Analysis",
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
    },
    {
        "name": "Question Generation",
        "operation": "generate_qa",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "input_from": "summary",
        "output_as": "questions"
    }
]

# Execute the workflow
results = await client.tools.execute_optimized_workflow(
    documents=[document],
    workflow=workflow
)

print(f"Workflow completed in {results['processing_time']:.2f}s")
print(f"Total cost: ${results['total_cost']:.6f}")
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm_gateway_mcp_server.git
cd llm_gateway_mcp_server

# Install with pip
pip install -e .

# Or install with optional dependencies
pip install -e .[all]
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# API Keys (at least one provider required)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key

# Server Configuration
SERVER_PORT=8000
SERVER_HOST=127.0.0.1

# Logging Configuration
LOG_LEVEL=INFO
USE_RICH_LOGGING=true

# Cache Configuration
CACHE_ENABLED=true
CACHE_TTL=86400
```

### Running the Server

```bash
# Start the MCP server
python -m llm_gateway.cli.main run

# Or with Docker
docker compose up
```

Once running, the server will be available at `http://localhost:8000`.

## Cost Savings With Delegation

Using LLM Gateway for delegation can yield significant cost savings:

| Task | Claude 3.7 Direct | Delegated to Cheaper LLM | Savings |
|------|-------------------|--------------------------|---------|
| Summarizing 100-page document | $4.50 | $0.45 (Gemini Flash) | 90% |
| Extracting data from 50 records | $2.25 | $0.35 (GPT-3.5) | 84% |
| Generating 20 content ideas | $0.90 | $0.12 (DeepSeek) | 87% |
| Processing 1,000 customer queries | $45.00 | $7.50 (Mixed delegation) | 83% |

These savings are achieved while maintaining high-quality outputs by letting Claude focus on high-level reasoning and orchestration while delegating mechanical tasks to cost-effective models.

## Architecture

### How MCP Integration Works

The LLM Gateway is built natively on the Model Context Protocol:

1. **MCP Server Core**: The gateway implements a full MCP server
2. **Tool Registration**: All capabilities are exposed as MCP tools
3. **Tool Invocation**: Claude and other AI agents can directly invoke these tools
4. **Context Passing**: Results are returned in MCP's standard format

This ensures seamless integration with Claude and other MCP-compatible agents.

### Component Diagram

```plaintext
┌─────────────┐         ┌───────────────────┐         ┌──────────────┐
│  Claude 3.7 │ ────────► LLM Gateway MCP   │ ────────► LLM Providers│
│   (Agent)   │ ◄──────── Server & Tools    │ ◄──────── (Multiple)   │
└─────────────┘         └───────┬───────────┘         └──────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │  Completion   │  │   Document    │  │  Extraction   │        │
│  │    Tools      │  │    Tools      │  │    Tools      │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │  Optimization │  │  Core MCP     │  │  Analytics    │        │
│  │    Tools      │  │   Server      │  │    Tools      │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │    Cache      │  │    Vector     │  │    Prompt     │        │
│  │   Service     │  │   Service     │  │   Service     │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow for Delegation

When Claude delegates a task to LLM Gateway:

1. Claude sends an MCP tool invocation request
2. The Gateway receives the request via MCP protocol
3. The appropriate tool processes the request
4. The caching service checks if the result is already cached
5. If not cached, the optimization service selects the appropriate provider/model
6. The provider layer sends the request to the selected LLM API
7. The response is standardized, cached, and metrics are recorded
8. The MCP server returns the result to Claude

## Detailed Feature Documentation

### Provider Integration

- **Multi-Provider Support**: First-class support for:
  - OpenAI (GPT-3.5, GPT-4o, GPT-4o mini)
  - Anthropic (Claude 3 Opus, Sonnet, Haiku, Claude 3.5 series)
  - Google (Gemini Pro, Gemini Flash)
  - DeepSeek (DeepSeek-Chat, DeepSeek-Coder)
  - Extensible architecture for adding new providers

- **Model Management**:
  - Automatic model selection based on task requirements
  - Model performance tracking
  - Fallback mechanisms for provider outages

### Cost Optimization

- **Intelligent Routing**: Automatically selects models based on:
  - Task complexity requirements
  - Budget constraints
  - Performance priorities
  - Historical performance data

- **Advanced Caching System**:
  - Multiple caching strategies (exact, semantic, task-based)
  - Configurable TTL per task type
  - Persistent cache with fast in-memory lookup
  - Cache statistics and cost savings tracking

### Document Processing

- **Smart Document Chunking**:
  - Multiple chunking strategies (token-based, semantic, structural)
  - Overlap configuration for context preservation
  - Handles very large documents efficiently

- **Document Operations**:
  - Summarization (with configurable formats)
  - Entity extraction
  - Question-answer pair generation
  - Batch processing with concurrency control

### Data Extraction

- **Structured Data Extraction**:
  - JSON extraction with schema validation
  - Table extraction (JSON, CSV, Markdown formats)
  - Key-value pair extraction
  - Semantic schema inference

### Vector Operations

- **Embedding Service**:
  - Efficient text embedding generation
  - Embedding caching to reduce API costs
  - Batched processing for performance

- **Semantic Search**:
  - Find semantically similar content
  - Configurable similarity thresholds
  - Fast vector operations

### System Features

- **Rich Logging**:
  - Beautiful console output with [Rich](https://github.com/Textualize/rich)
  - Emoji indicators for different operations
  - Detailed context information
  - Performance metrics in log entries

- **Streaming Support**:
  - Consistent streaming interface across all providers
  - Token-by-token delivery
  - Cost tracking during stream

## Real-World Use Cases

### AI Agent Orchestration

Claude or other advanced AI agents can use LLM Gateway to:

- Delegate routine tasks to cheaper models
- Process large documents in parallel
- Extract structured data from unstructured text
- Generate drafts for review and enhancement

### Enterprise Document Processing

Process large document collections efficiently:

- Break documents into meaningful chunks
- Distribute processing across optimal models
- Extract structured data at scale
- Implement semantic search across documents

### Research and Analysis

Research teams can use LLM Gateway to:

- Compare outputs from different models
- Process research papers efficiently
- Extract structured information from studies
- Track token usage and optimize research budgets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Model Context Protocol](https://github.com/mpctechdebt/mcp) for the foundation of the API
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [Pydantic](https://docs.pydantic.dev/) for data validation
- All the LLM providers making their models available via API
