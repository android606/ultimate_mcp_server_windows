# Ultimate MCP Server

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/modelcontextprotocol)

**A Model Context Protocol (MCP) server enabling intelligent delegation from high-capability AI agents to cost-effective LLMs**

![Illustration](https://github.com/Dicklesworthstone/ultimate_mcp_server/blob/main/illustration.webp)

[Getting Started](#getting-started) •
[Key Features](#key-features) •
[Usage Examples](#usage-examples) •
[Architecture](#architecture) •

</div>

## What is Ultimate MCP Server?

Ultimate MCP Server is an MCP-native server that enables intelligent task delegation from advanced AI agents like Claude 3.7 Sonnet to more cost-effective models like Gemini Flash 2.0 Lite. It provides a unified interface to multiple Large Language Model (LLM) providers while optimizing for cost, performance, and quality.

Beyond LLM delegation, it offers a comprehensive suite of tools spanning cognitive memory systems, browser automation, Excel manipulation, database interactions, document processing, and much more - essentially a complete AI agent operating system exposing dozens of powerful capabilities through the MCP protocol.

### The Vision: AI-Driven Resource Optimization

At its core, Ultimate MCP Server represents a fundamental shift in how we interact with AI systems. Rather than using a single expensive model for all tasks, it enables an intelligent hierarchy where:

- Advanced models like Claude 3.7 focus on high-level reasoning, orchestration, and complex tasks
- Cost-effective models handle routine processing, extraction, and mechanical tasks
- The overall system achieves near-top-tier performance at a fraction of the cost
- Specialized tools handle domain-specific functions like browser automation, Excel manipulation, or database queries
- Complex cognitive architectures enable persistent agent memory and reasoning

This approach mirrors how human organizations work — specialists handle complex decisions while delegating routine tasks to others with the right skills for those specific tasks.

### MCP-Native Architecture

The server is built on the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol), making it specifically designed to work with AI agents like Claude. All functionality is exposed through MCP tools that can be directly called by these agents, creating a seamless workflow for AI-to-AI delegation.

### Primary Use Case: AI Agent Ecosystem

The Ultimate MCP Server enables sophisticated AI agents like Claude 3.7 Sonnet to intelligently orchestrate a comprehensive set of tools:

```plaintext
                          delegates to
┌─────────────┐ ────────────────────────► ┌───────────────────┐         ┌──────────────┐
│ Claude 3.7  │                           │   Ultimate MCP     │ ───────►│ LLM Providers│
│   (Agent)   │ ◄──────────────────────── │     Server        │ ◄───────│ Specialized  │
└─────────────┘      returns results      └───────────────────┘         │ Tools        │
                                                │                        └──────────────┘
                                                ▼
                      ┌─────────────────────────────────────────────┐
                      │                                             │
                      │  • Browser Automation                       │
                      │  • Excel Spreadsheet Manipulation           │
                      │  • Cognitive Memory System                  │
                      │  • Database Interactions                    │
                      │  • Document Processing & OCR                │
                      │  • File System Operations                   │
                      │  • Entity Relations & Knowledge Graphs      │
                      │  • Audio Transcription                      │
                      │  • RAG & Vector Search                      │
                      │  • Text Classification                      │
                      │  • Command-Line Text Tools (Ripgrep, JQ...) │
                      │  • Dynamic API Integration                  │
                      │  • Meta Tools for Self-Improvement          │
                      │                                             │
                      └─────────────────────────────────────────────┘
```

**Example workflow:**

1. Claude identifies that a task requires multiple specialized capabilities
2. Claude delegates these tasks to Ultimate MCP Server via MCP tools
3. Ultimate MCP Server provides access to specialized tools (browser automation, database queries, etc.)
4. If LLM processing is needed, Ultimate MCP Server routes tasks to optimal cost-effective providers
5. The results are returned to Claude for high-level reasoning and decision-making
6. Claude maintains orchestration while leveraging the full suite of specialized capabilities

This delegation pattern enables complex workflows while saving 70-90% on API costs compared to using Claude for all operations.

## Why Use Ultimate MCP Server?

### 🔄 Comprehensive AI Agent Toolkit

The most powerful use case is enabling advanced AI agents to access an extensive ecosystem of tools:

- Access rich cognitive memory systems for persistent agent memory
- Perform complex web automation tasks with Playwright integration
- Manipulate and analyze Excel spreadsheets with deep integration
- Interact with databases through SQL operations
- Process documents with OCR capabilities
- Perform sophisticated vector search operations
- Utilize specialized text processing and classification
- Leverage command-line tools like ripgrep, awk, sed, jq for powerful text processing
- Dynamically integrate external REST APIs
- Use meta tools for self-discovery and optimization

### 💰 Cost Optimization

API costs for advanced models can be substantial. Ultimate MCP Server helps reduce costs by:

- Routing appropriate tasks to cheaper models (e.g., $0.01/1K tokens vs $0.15/1K tokens)
- Implementing advanced caching to avoid redundant API calls
- Tracking and optimizing costs across providers
- Enabling cost-aware task routing decisions
- Handling routine processing with specialized non-LLM tools

### 🔄 Provider Abstraction

Avoid provider lock-in with a unified interface:

- Standard API for OpenAI, Anthropic (Claude), Google (Gemini), Grok, DeepSeek, and OpenRouter
- Consistent parameter handling and response formatting
- Ability to swap providers without changing application code
- Protection against provider-specific outages and limitations

### 📄 Comprehensive Document and Data Processing

Process documents and data efficiently:

- Break documents into semantically meaningful chunks
- Process chunks in parallel across multiple models
- Extract structured data from unstructured text
- Generate summaries and insights from large texts
- Convert formats (HTML to Markdown, documents to structured data)
- Apply OCR to images and PDFs with LLM enhancement

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

### Provider Integration

- **Multi-Provider Support**: First-class support for:
  - OpenAI (gpt-4.1-mini, GPT-4o, GPT-4o mini)
  - Anthropic (Claude 3.7 series)
  - Google (Gemini Pro, Gemini Flash, Gemini Flash Light)
  - DeepSeek (DeepSeek-Chat, DeepSeek-Reasoner)
  - xAI (Grok models)
  - OpenRouter (access to various models)
  - Extensible architecture for adding new providers

- **Model Management**:
  - Automatic model selection based on task requirements
  - Model performance tracking
  - Fallback mechanisms for provider outages

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

### Secure Filesystem Operations

- **Path Management**:
  - Robust path validation and normalization
  - Symlink security verification to prevent traversal attacks
  - Configurable allowed directories for security boundaries

- **File Operations**:
  - Read files with proper encoding detection
  - Write files with proper directory validation
  - Smart text replacement for editing existing files
  - Detailed file metadata retrieval

- **Directory Operations**:
  - Directory creation with recursive support
  - Directory listing with detailed metadata
  - Hierarchical directory tree visualization
  - File and directory movement with security checks

- **Search Capabilities**:
  - Recursive file and directory searching
  - Case-insensitive pattern matching
  - Exclude patterns for filtering results

- **Security Features**:
  - Enforcement of allowed directory restrictions
  - Path normalization to prevent directory traversal attacks
  - Parent directory validation for write operations
  - Symlink target verification

### Autonomous Tool Documentation Refiner

The MCP server includes an autonomous documentation refinement system that improves the quality of tool documentation through an intelligent, iterative process:

- **Agent Simulation**: Tests how LLMs interpret documentation to identify ambiguities
- **Adaptive Testing**: Generates and executes diverse test cases based on tool schemas
- **Failure Analysis**: Uses LLM ensembles to diagnose documentation weaknesses
- **Smart Improvements**: Proposes targeted enhancements to descriptions, schemas, and examples
- **Iterative Refinement**: Continuously improves documentation until tests consistently pass
- **(See dedicated section for more details)**

### Browser Automation with Playwright
  - Enable agents to interact with websites: navigate, click, type, scrape data, take screenshots, generate PDFs, download/upload files, and execute JavaScript via Playwright integration.
  - Perform complex web research tasks across multiple search engines
  - Extract structured data from websites using intelligent patterns
  - Monitor specific data points on websites for changes
  - Find and download documents based on intelligent criteria
  - Synthesize multi-source research into comprehensive reports

### Cognitive & Agent Memory System

- **Multi-Level Memory Hierarchy**:
  - Working memory for current context
  - Episodic memory for experiences and observations
  - Semantic memory for facts and knowledge
  - Procedural memory for skills and workflows

- **Knowledge Management**:
  - Store and retrieve memories with rich metadata
  - Create associative memory links with relationship types
  - Track importance, confidence, and access patterns
  - Automatically suggest related memories

- **Workflow Tracking**:
  - Record and monitor agent actions and artifacts
  - Capture reasoning chains and thought processes
  - Manage dependencies between actions
  - Generate comprehensive workflow reports and visualizations

- **Smart Memory Operations**:
  - Consolidate related memories into higher-level insights
  - Generate reflections on agent behavior and learning
  - Optimize working memory based on relevance and recency
  - Decay less important memories over time

### Excel Spreadsheet Automation

- **Direct Spreadsheet Manipulation**:
  - Create, modify, and format Excel spreadsheets with natural language instructions
  - Analyze and debug complex formulas
  - Apply professional formatting and visualization
  - Export sheets to CSV and import data from various formats

- **Template Learning and Application**:
  - Learn from exemplar templates and adapt to new contexts
  - Preserve formula logic while updating with new data
  - Generate flexible templates based on instructions
  - Apply complex formatting patterns consistently

- **VBA Macro Generation**:
  - Generate VBA code from natural language instructions
  - Enable complex automation sequences within Excel
  - Test and secure macro execution
  - Provide explanations and documentation of generated code

### Structured Data Extraction

- **JSON Extraction**: Extract structured JSON with schema validation
- **Table Extraction**: Extract tables in multiple formats
- **Key-Value Extraction**: Extract key-value pairs from text
- **Semantic Schema Inference**: Generate schemas from text

### Tournament Mode

- **Code and Text Competitions**: Support for running tournament-style competitions
- **Multiple Models**: Compare outputs from different models simultaneously
- **Performance Metrics**: Evaluate and track model performance
- **Results Storage**: Persist tournament results for further analysis

### SQL Database Interactions

- **Query Execution**: Run SQL queries against various database types
- **Schema Analysis**: Analyze database schemas and recommend optimizations
- **Data Exploration**: Explore and visualize database contents
- **Query Generation**: Generate SQL from natural language descriptions

### Entity Relation Graphs

- **Entity Extraction**: Identify entities in text documents
- **Relationship Mapping**: Discover and map relationships between entities
- **Knowledge Graph Construction**: Build persistent knowledge graphs
- **Graph Querying**: Extract insights from entity relationship graphs

### Advanced Vector Operations

- **Semantic Search**: Find semantically similar content across documents
- **Vector Storage**: Efficient storage and retrieval of vector embeddings
- **Hybrid Search**: Combine keyword and semantic search capabilities (e.g., via Marqo)
- **Batched Processing**: Efficiently process large datasets

### Retrieval-Augmented Generation (RAG)

- **Contextual Generation**:
  - Augments LLM prompts with relevant retrieved information
  - Improves factual accuracy and reduces hallucinations
  - Integrates with vector search and document stores

- **Workflow Integration**:
  - Seamlessly combine document retrieval with generation tasks
  - Customizable retrieval and generation strategies

### Audio Transcription

- **Speech-to-Text**: Convert audio recordings to text
- **Speaker Diarization**: Identify different speakers in conversations
- **Transcript Enhancement**: Clean and format transcripts for readability
- **Multi-language Support**: Transcribe audio in various languages

### Text Classification

- **Custom Classifiers**: Build and apply text classification models
- **Multi-label Classification**: Assign multiple categories to text
- **Confidence Scoring**: Provide probability scores for classifications
- **Batch Processing**: Classify large document collections efficiently

### OCR Tools

- **Extract Text from PDF/Images**: Extract and enhance text using direct extraction or OCR, leveraging LLMs for correction.
- **Preprocessing**: Options for denoising, thresholding, deskewing images.
- **Structure Analysis**: Get PDF structure information.
- **Batch Processing**: Handle multiple documents concurrently.
- **(See dedicated section for installation and examples)**

### Text Redline Tools

- **HTML Redline Generation**: Create visual diffs between text/HTML, highlighting changes with formatting, move detection.
- **Document Comparison**: Compare various formats (text, HTML, Markdown) with intuitive redlines.

### HTML to Markdown Conversion

- **Intelligent Format Detection**: Auto-detect content type (HTML, markdown, code, text).
- **Content Extraction**: Use strategies like readability/trafilatura to get main content, filter boilerplate, preserve structure.
- **Markdown Optimization**: Clean, normalize, fix syntax, improve readability.

### Workflow Optimization Tools

- **Cost Estimation/Comparison**: Estimate API costs, compare models, recommend based on cost/quality/speed.
- **Model Selection**: Analyze task requirements, filter by capabilities/budget, provide recommendations.
- **Workflow Execution**: Define and run multi-stage pipelines with dependencies, parallel execution, optimal routing.

### Local Text Processing Tools

- **Offline Text Operations**: Clean, normalize, format text without API calls.
- **Command-Line Integration**: Securely use tools like `ripgrep`, `awk`, `sed`, `jq` via MCP tools.

### Model Performance Benchmarking

- **Speed/Latency Measurement**: Track response times, tokens/second, latency stats.
- **Performance Profiles**: Generate detailed profiles, track consistency, compare advertised vs. actual performance.
- **Empirical Optimization**: Make data-driven model choices based on real performance data.

### Server-Sent Events (SSE) Support

- **Streaming Completions**: Real-time token-by-token updates.
- **Progress Monitoring**: Track long-running operations, multi-stage workflows.
- **Event-Based Architecture**: Subscribe to specific event types.

### Multi-Model Synthesis

- **Comparative Analysis**: Side-by-side output analysis, identify strengths/weaknesses, contradictions/agreements.
- **Response Synthesis**: Combine best elements, generate meta-responses, create consensus outputs.
- **Collaborative Reasoning**: Implement multi-model reasoning chains, use specialized models for steps.

### Extended Model Support

- **Grok Integration**: Native support for xAI's Grok models.
- **DeepSeek Support**: Optimized handling for DeepSeek coding/conversational models.
- **OpenRouter Integration**: Access a wide variety of models via OpenRouter.

### Meta Tools for Self-Improvement & Dynamic Integration

- **Tool Information and Discovery**: Query available tools, get details, discover dynamically.
- **Tool Usage Recommendations**: AI-optimized advice, combination recommendations.
- **External API Integration**: Dynamically register external REST APIs via OpenAPI specs, convert endpoints to MCP tools.
- **Self-Improvement and Documentation**: Auto-generate LLM-optimized tool docs (see Autonomous Refiner).

### Analytics and Reporting

- **Usage Metrics Tracking**: Monitor tokens, costs, requests, success rates across providers/models.
- **Real-Time Monitoring**: Live stats, active requests, cost accumulation.
- **Detailed Reporting**: Generate historical reports, compare efficiency, export data.
- **Optimization Insights**: Identify cost savings, inefficient patterns, get recommendations.

### Prompt Templates and Management

- **Jinja2-Based Templates**: Reusable templates with variables, conditionals, composition.
- **Prompt Repository**: Persistent storage, categorization, versioning.
- **Metadata and Documentation**: Rich metadata, input/output docs, usage tracking.
- **Template Optimization**: Testing, comparison, iterative improvement.

### Error Handling and Resilience

- **Intelligent Retry Logic**: Automatic retries with exponential backoff for specific errors (rate limits, server issues).
- **Fallback Mechanisms**: Switch to alternate providers on failure, cache responses during downtime.
- **Detailed Error Reporting**: Comprehensive error info, categorization, tracking.
- **Validation and Prevention**: Input validation, security checks, misuse prevention.

### System Features

- **Rich Logging**: Formatted console output, performance metrics.
- **Health Monitoring**: Health check endpoints, resource monitoring.
- **Command-Line Interface**: Server management, tool interaction, status checks.

## Getting Started

### Installation

```bash
# Install uv if you don't already have it:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server

# Install in venv using uv:
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[all]" # Use ".[base]" for core functionality without all tools
```
*Note: Some tools like OCR have optional dependencies. Install with `uv pip install -e ".[ocr]"` or `uv pip install -e ".[all]"`.*

### Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# API Keys (at least one provider required)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_google_ai_studio_key # Or GOOGLE_APPLICATION_CREDENTIALS for service accounts
DEEPSEEK_API_KEY=your_deepseek_key
OPENROUTER_API_KEY=your_openrouter_key
# X_API_KEY=your_grok_key # If using Grok

# Server Configuration (Defaults shown)
SERVER_PORT=8013
SERVER_HOST=127.0.0.1 # Use 0.0.0.0 for external access/Docker

# Logging Configuration (Defaults shown)
LOG_LEVEL=INFO
USE_RICH_LOGGING=true

# Cache Configuration (Defaults shown)
CACHE_ENABLED=true
CACHE_TTL=86400 # 24 hours in seconds
# CACHE_TYPE=memory # Other options might include 'redis', 'diskcache'
# REDIS_URL=redis://localhost:6379/0 # If using Redis cache
```

### Running the Server

```bash
# Start the MCP server with all registered tools
python -m ultimate.cli.main run
# Or use the installed CLI command:
ultimate-mcp-server run

# Start the server with specific tools only
ultimate-mcp-server run --include-tools generate_completion read_file write_file

# Start the server excluding specific tools
ultimate-mcp-server run --exclude-tools browser_automation

# Start with Docker (ensure .env file is present or pass vars)
docker compose up
```

Once running, the server will be available at `http://localhost:8013` (or the configured host/port).

## Usage Examples

This section shows how an MCP client (like Claude 3.7) would invoke specific tools provided by the Ultimate MCP Server. Assume an initialized `mcp.client.Client` instance named `client`.

### Basic Completion

```python
import asyncio
from mcp.client import Client

async def basic_completion_example():
    client = Client("http://localhost:8013")
    response = await client.tools.completion(
        prompt="Write a short poem about a robot learning to dream.",
        provider="openai",
        model="gpt-4.1-mini",
        max_tokens=100,
        temperature=0.7
    )
    if response["success"]:
        print(f"Completion: {response['completion']}")
        print(f"Cost: ${response['cost']:.6f}")
    else:
        print(f"Error: {response['error']}")
    await client.close()

# asyncio.run(basic_completion_example())
```

### Claude Using Ultimate MCP Server for Document Analysis (Delegation)

```python
# Example showing delegation to cheaper models for sub-tasks
async def document_analysis_example():
    client = Client("http://localhost:8013")
    document = "... large document content ..." # Assume this is loaded

    # Step 1: Claude delegates document chunking (local operation)
    chunks_response = await client.tools.chunk_document(
        document=document, chunk_size=1000, method="semantic"
    )
    print(f"Document divided into {chunks_response['chunk_count']} chunks")

    # Step 2: Claude delegates summarization to a cheaper model
    summaries = []
    total_cost = 0.0
    for i, chunk in enumerate(chunks_response["chunks"]):
        summary = await client.tools.summarize_document(
            document=chunk,
            provider="gemini", # Delegate to cheaper model
            model="gemini-2.0-flash-lite",
            format="paragraph"
        )
        if summary["success"]:
            summaries.append(summary["summary"])
            total_cost += summary["cost"]
            print(f"Processed chunk {i+1} summary with cost ${summary['cost']:.6f}")
        else:
            print(f"Chunk {i+1} summarization failed: {summary['error']}")

    # Step 3: Claude delegates entity extraction to another cheap model
    entities_response = await client.tools.extract_entities(
        document=document,
        entity_types=["person", "organization", "location", "date"],
        provider="openai", # Delegate to another cost-effective model
        model="gpt-4.1-mini"
    )
    if entities_response["success"]:
        total_cost += entities_response["cost"]
        print(f"Extracted entities with cost ${entities_response['cost']:.6f}")
        # Claude would now process these summaries and entities using its advanced capabilities
    else:
        print(f"Entity extraction failed: {entities_response['error']}")

    print(f"Total estimated delegation cost: ${total_cost:.6f}")
    await client.close()

# asyncio.run(document_analysis_example())
```

### Browser Automation for Research

```python
# Use Playwright integration to perform web research
async def browser_research_example():
    client = Client("http://localhost:8013")
    result = await client.tools.research_and_synthesize_report(
        topic="Latest advances in quantum computing for financial applications",
        instructions={
            "search_query": "quantum computing financial applications 2024",
            "search_engines": ["google", "bing"],
            "urls_to_include": ["arxiv.org", "nature.com", "ibm.com/quantum"],
            "max_urls": 5,
            "focus_areas": ["algorithmic trading", "risk modeling", "encryption"],
            "report_format": "markdown",
            "report_length": "comprehensive",
            "llm_model": "gemini/gemini-2.0-pro" # Model used for synthesis
        }
    )
    if result["success"]:
        print(f"Research report generated with {len(result['extracted_data'])} sources")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(result['report'])
    else:
        print(f"Browser research failed: {result['error']}")
    await client.close()

# asyncio.run(browser_research_example())
```

### Cognitive Memory System Usage

```python
# Example interactions with the agent memory system
async def cognitive_memory_example():
    client = Client("http://localhost:8013")
    # Create a workflow
    workflow = await client.tools.create_workflow(
        title="Financial Research Project", goal="Produce investment recommendations"
    )
    workflow_id = workflow["workflow_id"]

    # Store a fact
    memory = await client.tools.store_memory(
        workflow_id=workflow_id, content="IBM Eagle has 127 qubits (Nov 2023)",
        memory_type="fact", memory_level="semantic", importance=8.5, tags=["quantum", "ibm"]
    )
    print(f"Stored memory ID: {memory['memory_id']}")

    # Search memories
    search_results = await client.tools.hybrid_search_memories(
        query="quantum hardware advances", workflow_id=workflow_id, memory_type="fact"
    )
    print(f"Found {len(search_results['results'])} relevant memories.")

    # Generate reflection
    reflection = await client.tools.generate_reflection(workflow_id=workflow_id, reflection_type="insights")
    print(f"Reflection generated: {reflection['reflection'][:100]}...")
    await client.close()

# asyncio.run(cognitive_memory_example())
```

### Excel Spreadsheet Automation

```python
# Create/modify Excel files using natural language
async def excel_automation_example():
    client = Client("http://localhost:8013")
    result = await client.tools.excel_execute(
        instruction="Create a 5-year financial projection: revenue growth 15% annually, COGS 40% revenue, OPEX growth 7%. Add charts.",
        file_path="financial_model.xlsx", operation_type="create", show_excel=False
    )
    if result["success"]:
        print(f"Excel operation successful: {result['message']}")
        print(f"File saved at: {result['output_file_path']}")
    else:
        print(f"Excel operation failed: {result['error']}")
    await client.close()

# asyncio.run(excel_automation_example())
```

### Multi-Provider Comparison

```python
# Get completions from multiple models for comparison
async def multi_provider_completion_example():
    client = Client("http://localhost:8013")
    multi_response = await client.tools.multi_completion(
        prompt="What are the main benefits of using the MCP protocol?",
        providers=[
            {"provider": "openai", "model": "gpt-4.1-mini"},
            {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
            {"provider": "gemini", "model": "gemini-2.0-flash-lite"}
        ],
        temperature=0.5
    )
    if multi_response["success"]:
        print("Multi-completion results:")
        for provider_key, result in multi_response["results"].items():
            if result["success"]:
                print(f"--- {provider_key} (Cost: ${result['cost']:.6f}) ---")
                print(f"Completion: {result['completion'][:150]}...") # Truncated
            else:
                print(f"--- {provider_key} Error: {result['error']} ---")
    else:
        print(f"Multi-completion failed: {multi_response['error']}")
    await client.close()

# asyncio.run(multi_provider_completion_example())
```

### Cost-Optimized Workflow Execution

```python
# Define and run a multi-step workflow with potential optimization
async def optimized_workflow_example():
    client = Client("http://localhost:8013")
    document = "... document content for workflow ..."
    workflow = [
        {"stage_id": "summarize", "tool_name": "summarize_document",
         "params": {"provider": "gemini", "model": "gemini-2.0-flash-lite", "document": document}},
        {"stage_id": "extract_entities", "tool_name": "extract_entities",
         "params": {"provider": "openai", "model": "gpt-4.1-mini", "document": document}},
        {"stage_id": "generate_qa", "tool_name": "generate_qa",
         "params": {"provider": "deepseek", "model": "deepseek-chat", "document": "${summarize.summary}"},
         "depends_on": ["summarize"]}
    ]
    results = await client.tools.execute_optimized_workflow(workflow=workflow)

    if results["success"]:
        print(f"Workflow completed in {results['processing_time']:.2f}s")
        print(f"Total cost: ${results['total_cost']:.6f}")
        # Access stage results via results['stage_outputs']
    else:
        print(f"Workflow execution failed: {results['error']}")
    await client.close()

# asyncio.run(optimized_workflow_example())
```

### Entity Relation Graph Example

```python
async def entity_graph_example():
    client = Client("http://localhost:8013")
    doc = "Amazon acquired Anthropic for $4 billion to compete with Microsoft and Google. Dario Amodei leads Anthropic."
    entity_graph = await client.tools.extract_entity_relations(
        document=doc,
        entity_types=["organization", "person", "money"],
        relationship_types=["acquired", "worth", "compete_with", "lead"],
        include_visualization=False # Set True for graphical output if supported
    )
    if entity_graph["success"]:
        print("Entity Graph Data:", entity_graph["graph_data"])
        # You might query this graph data using another tool or process it directly
    else:
        print(f"Entity relation extraction failed: {entity_graph['error']}")
    await client.close()

# asyncio.run(entity_graph_example())
```

### Structured Data Extraction (JSON)

```python
async def json_extraction_example():
    client = Client("http://localhost:8013")
    text_with_data = "User John Doe (john.doe@example.com) created account on 2024-07-15. ID: 12345."
    desired_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}, "email": {"type": "string", "format": "email"},
            "creation_date": {"type": "string", "format": "date"}, "user_id": {"type": "integer"}
        }, "required": ["name", "email", "creation_date", "user_id"]
    }
    json_response = await client.tools.extract_json(
        document=text_with_data, json_schema=desired_schema,
        provider="openai", model="gpt-4.1-mini" # Choose a model good at structured output
    )
    if json_response["success"]:
        print(f"Extracted JSON: {json_response['json_data']}")
        print(f"Cost: ${json_response['cost']:.6f}")
    else:
        print(f"JSON Extraction Error: {json_response['error']}")
    await client.close()

# asyncio.run(json_extraction_example())
```

### Retrieval-Augmented Generation (RAG) Query

```python
# Assumes documents are indexed in a vector store accessible by the server
async def rag_query_example():
    client = Client("http://localhost:8013")
    rag_response = await client.tools.rag_query( # Assuming tool name is rag_query
        query="What were the key findings in the latest financial report?",
        # index_name="financial_reports", # Optional: specify index
        # top_k=3, # Optional: number of docs to retrieve
        provider="anthropic", model="claude-3-5-haiku-20241022" # Model for generation
    )
    if rag_response["success"]:
        print(f"RAG Answer:\n{rag_response['answer']}")
        # print(f"Sources: {rag_response['sources']}") # If sources are returned
        print(f"Cost: ${rag_response['cost']:.6f}")
    else:
        print(f"RAG Error: {rag_response['error']}")
    await client.close()

# asyncio.run(rag_query_example())
```

### Running a Model Tournament

```python
async def model_tournament_example():
    client = Client("http://localhost:8013")
    tournament_response = await client.tools.run_model_tournament(
        task_type="code_generation",
        prompt="Write a Python function for factorial.",
        competitors=[
            {"provider": "openai", "model": "gpt-4.1-mini"},
            {"provider": "anthropic", "model": "claude-3-opus-20240229"},
            {"provider": "deepseek", "model": "deepseek-coder"}
        ],
        evaluation_criteria=["correctness", "efficiency", "readability"]
    )
    if tournament_response["success"]:
        print("Tournament Results:")
        for rank, result in enumerate(tournament_response.get("ranking", [])):
            print(f"  {rank+1}. {result['provider']}/{result['model']} - Score: {result.get('score', 'N/A'):.2f}")
        print(f"Total Cost: ${tournament_response['total_cost']:.6f}")
    else:
        print(f"Tournament Error: {tournament_response['error']}")
    await client.close()

# asyncio.run(model_tournament_example())
```

### Listing Available Tools (Meta Tool)

```python
async def list_tools_example():
    client = Client("http://localhost:8013")
    list_tools_response = await client.tools.list_tools() # Assuming tool name list_tools
    if list_tools_response["success"]:
        print("Available Tools:")
        for tool_name, tool_info in list_tools_response["tools"].items():
            print(f"- {tool_name}: {tool_info.get('description', 'No description')[:80]}...")
    else:
        print(f"Error listing tools: {list_tools_response['error']}")
    await client.close()

# asyncio.run(list_tools_example())
```

### Local Command-Line Text Processing (e.g., ripgrep)

```python
async def local_cli_tool_example():
    client = Client("http://localhost:8013")
    ripgrep_result = await client.tools.run_ripgrep(
        args_str="-i --json 'error|exception' -C 1", # Search case-insensitively, JSON output, 1 line context
        input_data="Log line 1\nMAJOR Error in module X\nLog line 3\nMinor exception occurred\nLog line 5"
    )
    if ripgrep_result["success"]:
        print("Ripgrep Output (stdout):")
        print(ripgrep_result["stdout"])
    else:
        print(f"Ripgrep Error: {ripgrep_result['error']} - {ripgrep_result['stderr']}")
    await client.close()

# asyncio.run(local_cli_tool_example())
```

### Dynamic API Integration Example

```python
# Example assumes the server has a tool to register/call dynamic APIs
async def dynamic_api_example():
    client = Client("http://localhost:8013")
    # Register Petstore API
    reg_response = await client.tools.register_api(
        api_name="petstore", openapi_url="https://petstore.swagger.io/v2/swagger.json"
    )
    if not reg_response["success"]:
        print(f"API Registration failed: {reg_response['error']}")
        await client.close()
        return
    print(f"Registered {reg_response['tools_count']} petstore tools.")

    # Call a tool derived from the API
    pet_response = await client.tools.call_dynamic_tool(
        tool_name="petstore_getPetById", inputs={"petId": 1} # Tool name likely derived
    )
    if pet_response["success"]:
        print("Pet details:", pet_response["response_body"])
    else:
        print(f"Dynamic tool call failed: {pet_response['error']}")

    # Unregister (optional cleanup)
    await client.tools.unregister_api(api_name="petstore")
    await client.close()

# asyncio.run(dynamic_api_example())
```

### OCR Usage Example

```python
# Requires 'ocr' extras installed: pip install 'ultimate-mcp-server[ocr]'
async def ocr_example():
    client = Client("http://localhost:8013")
    # Assume 'scanned_document.pdf' exists
    pdf_text_result = await client.tools.extract_text_from_pdf(
        file_path="scanned_document.pdf", # Provide path accessible by the server
        extraction_method="ocr_only", # Force OCR
        reformat_as_markdown=True,
        llm_correction_model={"provider": "openai", "model": "gpt-4.1-mini"} # Optional LLM correction
    )
    if pdf_text_result["success"]:
        print("Extracted PDF Text (Markdown):")
        print(pdf_text_result["text"][:500] + "...") # Print first 500 chars
        print(f"Cost (incl. correction if used): ${pdf_text_result['cost']:.6f}")
    else:
        print(f"PDF OCR failed: {pdf_text_result['error']}")

    # Assume 'image_receipt.png' exists
    image_text_result = await client.tools.process_image_ocr(
        image_path="image_receipt.png", # Provide path accessible by the server
        preprocessing_options={"denoise": True, "deskew": True}
    )
    if image_text_result["success"]:
         print("\nExtracted Image Text:")
         print(image_text_result["text"])
         print(f"Cost: ${image_text_result['cost']:.6f}")
    else:
        print(f"Image OCR failed: {image_text_result['error']}")

    await client.close()

# Make sure to create dummy files or provide actual paths for this example
# asyncio.run(ocr_example())
```

## Autonomous Documentation Refiner

The Ultimate MCP Server includes a powerful feature for autonomously analyzing, testing, and refining the documentation of registered MCP tools. This feature, implemented in `ultimate/tools/docstring_refiner.py`, helps improve the usability and reliability of tools when invoked by Large Language Models (LLMs) like Claude.

### How It Works

The documentation refiner follows a methodical, iterative approach:

1.  **Agent Simulation**: Simulates how an LLM agent would interpret the current documentation to identify potential ambiguities or missing information
2.  **Adaptive Test Generation**: Creates diverse test cases based on the tool's schema, simulation results, and failures from previous iterations
3.  **Schema-Aware Testing**: Validates generated test cases against the schema before execution, then executes valid tests against the actual tools
4.  **Ensemble Failure Analysis**: Uses multiple LLMs to analyze failures in the context of the documentation used for that test run
5.  **Structured Improvement Proposals**: Generates specific improvements to the description, schema (as JSON Patch operations), and usage examples
6.  **Validated Schema Patching**: Applies and validates proposed schema patches in-memory
7.  **Iterative Refinement**: Repeats the cycle until tests consistently pass or a maximum iteration count is reached
8.  **Optional Winnowing**: Performs a final pass to streamline documentation while preserving critical information

### Benefits

- **Reduces Manual Effort**: Automates the tedious process of improving tool documentation
- **Improves Agent Performance**: Creates clearer, more precise documentation that helps LLMs correctly use tools
- **Identifies Edge Cases**: Discovers ambiguities and edge cases human writers might miss
- **Increases Consistency**: Establishes a consistent documentation style across all tools
- **Adapts to Feedback**: Learns from test failures to target specific documentation weaknesses
- **Schema Evolution**: Incrementally improves schemas with validated patches
- **Detailed Reporting**: Provides comprehensive reports on the entire refinement process

### Limitations and Considerations

- **Cost & Time**: Performs multiple LLM calls per iteration per tool, which can be expensive and time-consuming
- **Resource Intensive**: May require significant computational resources for large tool ecosystems
- **LLM Dependency**: Quality of improvements depends on the capabilities of the LLMs used
- **Schema Complexity**: Generating correct JSON Patches for complex schemas can be challenging
- **Maintenance Complexity**: The system has many dependencies which can make maintenance more difficult

### When to Use

This feature is particularly valuable when:

- You have many tools that are frequently accessed by LLM agents
- You're seeing a high rate of tool usage failures due to misinterpretation of documentation
- You're expanding your tool ecosystem and need to ensure consistent documentation quality
- You want to improve agent performance without modifying the underlying tools

### Usage Example (Server-Side)

This tool is typically run server-side, not called directly by an external agent via MCP.

```python
# Example of how to invoke the refiner from within the server environment
# This code would typically reside in a maintenance script or admin interface

# from ultimate_mcp_server.tools.docstring_refiner import refine_tool_documentation
# from ultimate_mcp_server.core import mcp_context # Assuming context setup

async def run_refiner():
    # Assuming mcp_context is properly initialized with tools and config
    print("Starting documentation refinement...")
    result = await refine_tool_documentation(
        tool_names=["browser_navigate", "extract_json"], # Refine specific tools
        max_iterations=2,
        refinement_model_config={"provider": "openai", "model": "gpt-4o"},
        enable_winnowing=True,
        ctx=mcp_context # Pass the MCP context
    )
    print("Refinement complete.")
    # Process results (e.g., save updated docstrings/schemas)
    if result["success"]:
        print(f"Refined {len(result['refined_tools'])} tools.")
        # Access detailed report: result['report']
    else:
        print(f"Refinement failed: {result['error']}")

# In a real scenario, you'd run this within the server's async loop or a dedicated script
# asyncio.run(run_refiner())
```

## Example Library and Testing Framework

The Ultimate MCP Server includes an extensive collection of 35+ end-to-end examples (`examples/*.py`) that serve as both comprehensive documentation and integration tests. These examples demonstrate real-world use cases and ensure all components work together correctly.

### Example Structure and Organization

Examples are organized by functionality (model integration, specific tools, workflows, advanced features) and are structured as standalone applications using the `mcp.client` to interact with a running server instance. They leverage the `Rich` library for clear, formatted console output.

### Rich Visual Output

Examples provide informative output using tables, syntax highlighting, progress bars, and panels to clearly show operations, results, costs, and performance statistics.

### Running Examples and Tests

The examples can be run individually:

```bash
# Ensure the server is running in another terminal
python examples/simple_completion_demo.py
python examples/browser_automation_demo.py --headless # Example with args
```

A comprehensive test suite script orchestrates running all examples:

```bash
# Run the complete test suite against a running server
python run_all_demo_scripts_and_check_for_errors.py
```

This script automatically discovers examples, runs them, validates outcomes against expected patterns (handling known benign messages like missing API keys), and generates a detailed report.

These examples serve as living documentation and a robust validation mechanism for the server's capabilities.

## CLI Commands

Ultimate MCP Server provides a command-line interface:

```bash
# Show available commands
ultimate-mcp-server --help

# Start the server (see Getting Started)
ultimate-mcp-server run [options]

# List available providers configured via environment variables
ultimate-mcp-server providers

# List available tools registered by the server
ultimate-mcp-server tools [--category CATEGORY] [--verbose]

# Test connectivity and basic generation for a provider
ultimate-mcp-server test openai --model gpt-4.1-mini

# Generate a completion directly from the CLI
ultimate-mcp-server complete --provider anthropic --model claude-3-5-sonnet-20241022 --prompt "Explain MCP."

# Check cache status and statistics
ultimate-mcp-server cache --status [--clear]
```

Use `ultimate-mcp-server COMMAND --help` for specific command options.

## Advanced Configuration

Configuration is primarily managed through environment variables (or a `.env` file). Key options beyond API keys:

### Server Configuration

- `SERVER_HOST`: (Default: `127.0.0.1`) Network interface. Use `0.0.0.0` for external/Docker.
- `SERVER_PORT`: (Default: `8013`) Listening port.
- `API_PREFIX`: (Default: `/`) URL prefix for API endpoints.

### Tool Filtering (at startup)

- `--include-tools`: Comma-separated list of tools to *only* include.
- `--exclude-tools`: Comma-separated list of tools to exclude.
- Example: `ultimate-mcp-server run --include-tools read_file,write_file,completion`

### Logging Configuration

- `LOG_LEVEL`: (Default: `INFO`) Verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `USE_RICH_LOGGING`: (Default: `true`) Use formatted console logs. `false` for plain text.
- `LOG_FORMAT`: (Optional) Custom log format string.
- `LOG_TO_FILE`: (Optional, e.g., `gateway.log`) Path for file logging.

### Cache Configuration

- `CACHE_ENABLED`: (Default: `true`) Global cache switch.
- `CACHE_TTL`: (Default: `86400`) Default cache item lifetime (seconds).
- `CACHE_TYPE`: (Default: `memory`) Backend (`memory`, `redis`, `diskcache` - check implementation).
- `CACHE_MAX_SIZE`: (Optional) Max items or memory size.
- `REDIS_URL`: (Required if `CACHE_TYPE=redis`) e.g., `redis://localhost:6379/0`.

### Provider Timeouts & Retries

- `PROVIDER_TIMEOUT`: (Default: `120`) Default request timeout (seconds).
- `PROVIDER_MAX_RETRIES`: (Default: `3`) Default retries on failure.
- Specific provider overrides may exist (e.g., `OPENAI_TIMEOUT`).

### Tool-Specific Configuration

- Some tools may have dedicated env vars (e.g., `MARQO_URL`, `PLAYWRIGHT_BROWSER_TYPE`). Check tool documentation/code.

*Restart the server after changing environment variables.*

## Deployment Considerations

For production or robust deployments:

### 1. Running as a Background Service

Use a process manager like `systemd` (Linux), `supervisor`, or Docker restart policies (`unless-stopped`, `always`) to ensure the server runs continuously and restarts on failure.

### 2. Using a Reverse Proxy (Nginx/Caddy/Apache)

Highly recommended for:
- **HTTPS/SSL Termination:** Handle certificates securely.
- **Load Balancing:** Distribute traffic across multiple server instances.
- **Path Routing:** Map external URLs to the internal server.
- **Security Headers:** Add HSTS, CSP, etc.
- **Access Control:** Implement IP allow-listing or gateway authentication.
- **Buffering/Caching:** Potential additional performance benefits.

*Example Nginx `location` block (simplified):*
```nginx
location /ultimate-mcp-server/ { # Match your desired public path
    proxy_pass http://127.0.0.1:8013/; # Point to the internal server
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    # Add proxy timeouts, buffer settings, etc.
}
```

### 3. Container Orchestration (Kubernetes/Swarm)

- **Health Checks:** Configure liveness/readiness probes using an endpoint like `/healthz`.
- **Configuration:** Use ConfigMaps/Secrets for env vars and API keys.
- **Resource Limits:** Define CPU/memory requests and limits.
- **Service Discovery:** Use the orchestrator's built-in mechanisms.

### 4. Resource Allocation

- Ensure sufficient **RAM**, especially for caching and large document processing.
- Monitor **CPU usage**, especially under heavy load.

## Cost Savings With Delegation

Using Ultimate MCP Server for delegation can yield significant cost savings:

| Task                            | Claude 3.7 Direct | Delegated to Cheaper LLM | Savings |
| :------------------------------ | :---------------- | :----------------------- | :------ |
| Summarizing 100-page document   | $4.50             | $0.45 (Gemini Flash)     | 90%     |
| Extracting data from 50 records | $2.25             | $0.35 (gpt-4.1-mini)     | 84%     |
| Generating 20 content ideas     | $0.90             | $0.12 (DeepSeek)         | 87%     |
| Processing 1,000 customer queries | $45.00            | $7.50 (Mixed delegation) | 83%     |

*(Costs are illustrative and depend on actual token counts and current provider pricing)*

These savings are achieved while maintaining high-quality outputs by letting advanced models focus on high-level reasoning and orchestration, while delegating mechanical or simpler tasks to cost-effective models or specialized tools.

## Why AI-to-AI Delegation Matters

The strategic importance extends beyond simple cost savings:

### Democratizing Advanced AI Capabilities

- Makes top-tier AI reasoning accessible at lower blended costs.
- Enables organizations with budget constraints to leverage advanced AI.

### Economic Resource Optimization

- Reserves expensive models for complex reasoning, creativity, and orchestration.
- Uses cost-effective models or specialized tools for routine processing, extraction, etc.
- Achieves near-top-tier system performance at a fraction of the cost.
- Makes API costs more controllable and predictable.

### Sustainable AI Architecture

- Reduces unnecessary consumption of high-end computational resources.
- Promotes a tiered approach matching capabilities to requirements.
- Enables experimentation that might be cost-prohibitive otherwise.

### Technical Evolution Path

- Moves from monolithic AI calls to distributed, multi-agent/multi-model workflows.
- Enables AI-driven orchestration of complex processing pipelines.
- Builds foundations for AI systems that can reason about their own resource usage.

### The Future of AI Efficiency

- Points toward AI systems that actively manage and optimize resource use.
- Positions high-capability models as intelligent orchestrators.
- Enables more sophisticated, self-organizing AI workflows.

This vision represents the next frontier in practical AI deployment, moving beyond using single, expensive models for every task.

## Architecture

### How MCP Integration Works

The Ultimate MCP Server is built natively on the Model Context Protocol:

1.  **MCP Server Core**: Implements a compliant MCP server endpoint.
2.  **Tool Registration**: All capabilities (LLM access, browser, file system, etc.) are exposed as standard MCP tools with defined schemas.
3.  **Tool Invocation**: MCP clients (like Claude) can discover and invoke these tools using standard MCP requests.
4.  **Context Passing**: Parameters are passed, and results are returned in MCP's standard structured format.

This ensures seamless integration with MCP-compatible agents.

### Component Diagram

```plaintext
┌─────────────┐         ┌────────────────────────────┐         ┌──────────────┐
│  Claude 3.7 │ MCP Req ► Ultimate MCP Server        │ API Req ►│ LLM Providers│
│   (Agent)   │◄─────────│ (Core Server & Tool Impl.) │◄─────────│ (Multiple)   │
└─────────────┘ MCP Resp  └──────────────┬─────────────┘ API Resp └───────┬──────┘
                                         │                                │
                                         ▼ Tool Execution                 │ Calls to
┌─────────────────────────────────────────────────────────────────────────┴───────────────┐
│ Internal Services & Tool Categories:                                                    │
│                                                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│  │  Completion   │  │   Document    │  │  Extraction   │  │ Browser Tools │             │
│  │    (LLM)      │  │    Tools      │  │    Tools      │  │ (Playwright)  │             │
│  └───────┬───────┘  └───────────────┘  └───────┬───────┘  └───────────────┘             │
│          │                                     │                                        │
│  ┌───────▼───────┐  ┌───────────────┐  ┌───────▼───────┐  ┌───────────────┐             │
│  │ Optimization  │  │ Cognitive Mem │  │ Vector / RAG  │  │ Excel Tools   │             │
│  │ & Routing     │  │ System        │  │ Service       │  │               │             │
│  └───────┬───────┘  └───────────────┘  └───────────────┘  └───────────────┘             │
│          │                                                                              │
│  ┌───────▼───────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│  │ Provider Abstr│  │ Filesystem    │  │ SQL DB Tools  │  │ Audio Tools   │             │
│  │ (API Clients) │  │ Tools         │  │               │  │               │             │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘             │
│                                                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│  │ Caching Svc   │  │ Analytics Svc │  │ Prompt Mgmt   │  │ OCR Tools     │             │
│  │               │  │               │  │               │  │               │             │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘             │
│                                                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│  │ Tournament    │  │ Entity Graph  │  │ Text Classify │  │ Meta / CLI    │             │
│  │ Tools         │  │ Tools         │  │ Tools         │  │ Tools / DynAPI│             │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘             │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow for Delegation

1.  An MCP agent (e.g., Claude) identifies a task suitable for delegation or requiring a specialized tool.
2.  The agent sends an MCP tool invocation request to the Ultimate MCP Server endpoint.
3.  The server's MCP Core receives and parses the request, identifying the target tool and parameters.
4.  The corresponding Tool Implementation is invoked.
5.  The Caching Service checks for a valid cached response for this request. If found, it's returned (step 9).
6.  If not cached:
    *   If it's an LLM task, the Optimization Service selects the best provider/model based on configuration, cost, and task needs. The Provider Abstraction layer formats and sends the request to the chosen LLM API.
    *   If it's a specialized tool (Browser, Filesystem, Excel, etc.), the relevant module executes the required action locally or interacts with the target system (e.g., database, Playwright).
7.  The response (from LLM or specialized tool) is received.
8.  The response is standardized, metrics are recorded by the Analytics Service, and the result is stored in the Cache Service.
9.  The MCP Core formats the final result according to the MCP specification.
10. The server sends the MCP response back to the requesting agent.

## Real-World Use Cases

### Advanced AI Agent Capabilities

Enable agents like Claude to perform complex tasks by giving them access to tools for:
- Persistent memory across sessions (Cognitive Memory).
- Web research, data scraping, form filling (Browser Automation).
- Financial modeling, data analysis in spreadsheets (Excel Automation).
- Querying and manipulating databases (SQL Tools).
- Processing PDFs, images (OCR, Document Tools).
- Building and querying knowledge graphs (Entity Relation Tools).
- Transcribing meetings or audio notes (Audio Transcription).
- Searching internal documents semantically (Vector Search / RAG).
- Running command-line utilities for text processing (CLI Tools).

### Enterprise Workflow Automation

Build automated processes combining multiple capabilities:
- Intelligent document workflows: OCR -> Extract -> Classify -> Store in DB -> Summarize.
- Research assistants: Search web -> Scrape data -> Summarize articles -> Store in memory -> Generate report.
- Financial analysis pipelines: Fetch data (DB/Web) -> Analyze in Excel -> Generate charts/reports.
- Customer support augmentation: Transcribe call -> Classify issue -> Search knowledge base (RAG) -> Draft response.

### Data Processing and Integration

- Extract structured data (JSON, tables) from unstructured documents or web pages.
- Build knowledge graphs from text collections.
- Transform and normalize data using SQL, Excel, or local text tools.
- Classify large document sets automatically.
- Create searchable semantic knowledge bases.

### Research and Analysis

- Automate literature reviews and data gathering from web/databases.
- Compare different model outputs for specific analytical tasks.
- Process large datasets or research papers efficiently.
- Extract structured findings from studies.
- Track research budgets with cost analytics.
- Maintain persistent research notes and findings (Cognitive Memory).

### Document Intelligence

- End-to-end PDF/image processing: OCR -> LLM Enhancement -> Structure Extraction -> Classification -> Summarization -> Indexing (Vector Search).

### Financial Analysis and Modeling

- Automate creation/maintenance of complex Excel models.
- Access/analyze financial data from databases.
- Gather market data via browser automation.
- Find relevant insights in reports using semantic search.
- Track investment theses and supporting evidence (Cognitive Memory).

## Security Considerations

When deploying and operating the Ultimate MCP Server:

1.  **API Key Management:** Use environment variables or secrets management tools. Never hardcode keys. Restrict permissions on `.env` files. Rotate keys periodically.
2.  **Network Exposure:** Use `SERVER_HOST=127.0.0.1` unless external access is needed. If exposed, use a reverse proxy and firewall rules to restrict access to trusted sources.
3.  **Authentication/Authorization:** The server may lack built-in auth. Rely on network security (firewalls, VPNs) or reverse proxy authentication (Basic Auth, OAuth2 proxy). Ensure only authorized clients can reach the endpoint.
4.  **Rate Limiting:** Implement rate limiting at the reverse proxy or gateway level to prevent DoS and control costs.
5.  **Input Validation:** Be cautious if tools interpret inputs directly (especially filesystem, SQL, browser eval). Sanitize where appropriate. MCP structure provides some safety.
6.  **Dependency Security:** Regularly update dependencies (`uv pip install --upgrade ...`) and use scanning tools (`pip-audit`, Dependabot).
7.  **Logging:** Be mindful that `DEBUG` logs might contain sensitive data (prompts/responses). Configure `LOG_LEVEL` appropriately and secure log files.
8.  **Filesystem Tool Security:** Configure `ALLOWED_DIRS` carefully if using filesystem tools to prevent access outside designated areas. Ensure the server process runs with appropriate (least privilege) user permissions.
9.  **Browser Tool Security:** Be aware of risks if automating interactions with untrusted websites (potential for malicious scripts, though Playwright provides sandboxing). Avoid storing sensitive session data if possible.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

- [Model Context Protocol](https://github.com/modelcontextprotocol) for the API foundation.
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output.
- [Pydantic](https://docs.pydantic.dev/) for robust data validation.
- [uv](https://github.com/astral-sh/uv) for fast Python package management.
- [FastAPI](https://fastapi.tiangolo.com/) as the underlying web framework.
- [Playwright](https://playwright.dev/) for powerful browser automation.
- All the LLM providers for their APIs.