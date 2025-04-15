# LLM Gateway MCP Server

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/mpctechdebt/mcp)

**A Model Context Protocol (MCP) server enabling intelligent delegation from high-capability AI agents to cost-effective LLMs**

![Illustration](https://github.com/Dicklesworthstone/llm_gateway_mcp_server/blob/main/illustration.webp)

[Getting Started](#getting-started) •
[Key Features](#key-features) •
[Usage Examples](#usage-examples) •
[Architecture](#architecture) •

</div>

## What is LLM Gateway?

LLM Gateway is an MCP-native server that enables intelligent task delegation from advanced AI agents like Claude 3.7 Sonnet to more cost-effective models like Gemini Flash 2.0 Lite. It provides a unified interface to multiple Large Language Model (LLM) providers while optimizing for cost, performance, and quality.

### The Vision: AI-Driven Resource Optimization

At its core, LLM Gateway represents a fundamental shift in how we interact with AI systems. Rather than using a single expensive model for all tasks, it enables an intelligent hierarchy where:

- Advanced models like Claude 3.7 focus on high-level reasoning, orchestration, and complex tasks
- Cost-effective models handle routine processing, extraction, and mechanical tasks
- The overall system achieves near-top-tier performance at a fraction of the cost

This approach mirrors how human organizations work — specialists handle complex decisions while delegating routine tasks to others with the right skills for those specific tasks.

### MCP-Native Architecture

The server is built on the [Model Context Protocol (MCP)](https://github.com/mpctechdebt/mcp), making it specifically designed to work with AI agents like Claude. All functionality is exposed through MCP tools that can be directly called by these agents, creating a seamless workflow for AI-to-AI delegation.

### Primary Use Case: AI Agent Task Delegation

The primary design goal of LLM Gateway is to allow sophisticated AI agents like Claude 3.7 Sonnet to intelligently delegate tasks to less expensive models:

```plaintext
                          delegates to
┌─────────────┐ ────────────────────────► ┌───────────────────┐         ┌──────────────┐
│ Claude 3.7  │                           │   LLM Gateway     │ ───────►│ Gemini Flash │
│   (Agent)   │ ◄──────────────────────── │    MCP Server     │ ◄───────│ DeepSeek     │
└─────────────┘      returns results      └───────────────────┘         │ gpt-4.1-mini  │
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

- Have Claude 3.7 use gpt-4.1-mini for initial document summarization
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

### Secure Filesystem Operations

- **File Access Controls**: Restrict operations to allowed directories
- **Smart File Editing**: Text-based file editing with whitespace-insensitive matching
- **Comprehensive Operations**:
  - Read and write files with proper encoding handling
  - Edit existing files with smart pattern matching
  - Create, list, and traverse directories
  - Move files and directories securely
- **Security Features**:
  - Path validation and normalization
  - Symlink security verification
  - Parent directory existence checking

### Browser Automation with Playwright
  - Enable agents to interact with websites: navigate, click, type, scrape data, take screenshots, generate PDFs, download/upload files, and execute JavaScript via Playwright integration.

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

### Advanced Vector Operations

- **Semantic Search**: Find semantically similar content across documents
- **Vector Storage**: Efficient storage and retrieval of vector embeddings
- **Hybrid Search**: Combine keyword and semantic search capabilities
- **Batched Processing**: Efficiently process large datasets

### Retrieval-Augmented Generation (RAG)

- **Contextual Generation**:
  - Augments LLM prompts with relevant retrieved information
  - Improves factual accuracy and reduces hallucinations
  - Integrates with vector search and document stores

- **Workflow Integration**:
  - Seamlessly combine document retrieval with generation tasks
  - Customizable retrieval and generation strategies

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

### Local Text Processing

- **Offline Operations**:
  - Provides tools for text manipulation that run locally, without API calls
  - Includes functions for cleaning, formatting, and basic analysis
  - Useful for pre-processing text before sending to LLMs or post-processing results

## OCR Tools

The LLM Gateway includes powerful OCR (Optical Character Recognition) tools that leverage LLMs to improve text extraction from PDFs and images:

- **Extract Text from PDF**: Extract and enhance text from PDF documents using direct extraction or OCR.
- **Process Image OCR**: Extract and enhance text from images with preprocessing options.
- **Enhance OCR Text**: Improve existing OCR text using LLM-based correction and formatting.
- **Analyze PDF Structure**: Get information about a PDF's structure without full text extraction.
- **Batch Process Documents**: Process multiple documents with concurrent execution.

### OCR Installation

OCR tools require additional dependencies. Install them with:

```bash
pip install 'llm-gateway[ocr]'
```

### OCR Usage Examples

```python
# Extract text from a PDF file with LLM correction
result = await client.tools.extract_text_from_pdf(
    file_path="document.pdf",
    extraction_method="hybrid",  # Try direct text extraction first, fall back to OCR if needed
    max_pages=5,
    reformat_as_markdown=True
)

# Process an image file with custom preprocessing
result = await client.tools.process_image_ocr(
    image_path="scan.jpg",
    preprocessing_options={
        "denoise": True,
        "threshold": "adaptive",
        "deskew": True
    },
    ocr_language="eng+fra"  # Multi-language support
)

# Workflow with OCR and summarization
workflow = [
    {
        "stage_id": "extract_text",
        "tool_name": "extract_text_from_pdf",
        "params": {
            "file_path": "/path/to/document.pdf",
            "reformat_as_markdown": True
        }
    },
    {
        "stage_id": "summarize",
        "tool_name": "summarize_document",
        "params": {
            "document": "${extract_text.text}",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022"
        },
        "depends_on": ["extract_text"]
    }
]

workflow_result = await client.tools.execute_optimized_workflow(
    workflow=workflow
)
```

## Usage Examples

### Claude Using LLM Gateway for Document Analysis

This example shows how Claude can use the LLM Gateway to process a document by delegating tasks to cheaper models:

```python
import asyncio
from mcp.client import Client

async def main():
    # Claude would use this client to connect to the LLM Gateway
    client = Client("http://localhost:8013")
    
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
        model="gpt-4.1-mini"
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
        {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.3},
        {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "temperature": 0.3},
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
        "model": "gpt-4.1-mini",
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

### Document Chunking

To break a large document into smaller, manageable chunks:

```python
large_document = "... your very large document content ..."

chunking_response = await client.tools.chunk_document(
    document=large_document,
    chunk_size=500,     # Target size in tokens
    overlap=50,         # Token overlap between chunks
    method="semantic"   # Or "token", "structural"
)

if chunking_response["success"]:
    print(f"Document divided into {chunking_response['chunk_count']} chunks.")
    # chunking_response['chunks'] contains the list of text chunks
else:
    print(f"Error: {chunking_response['error']}")
```

### Multi-Provider Completion

To get completions for the same prompt from multiple providers/models simultaneously for comparison:

```python
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
            print(f"--- {provider_key} ---")
            print(f"Completion: {result['completion']}")
            print(f"Cost: ${result['cost']:.6f}")
        else:
            print(f"--- {provider_key} Error: {result['error']} ---")
else:
    print(f"Multi-completion failed: {multi_response['error']}")
```

### Structured Data Extraction (JSON)

To extract information from text into a specific JSON schema:

```python
text_with_data = "User John Doe (john.doe@example.com) created an account on 2024-07-15. His user ID is 12345."

desired_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "creation_date": {"type": "string", "format": "date"},
        "user_id": {"type": "integer"}
    },
    "required": ["name", "email", "creation_date", "user_id"]
}

json_response = await client.tools.extract_json(
    document=text_with_data,
    json_schema=desired_schema,
    provider="openai", # Choose a provider capable of structured extraction
    model="gpt-4.1-mini"
)

if json_response["success"]:
    print(f"Extracted JSON: {json_response['json_data']}")
    print(f"Cost: ${json_response['cost']:.6f}")
else:
    print(f"Error: {json_response['error']}")
```

### Retrieval-Augmented Generation (RAG) Query

To ask a question using RAG, where the system retrieves relevant context before generating an answer (assuming relevant documents have been indexed):

```python
rag_response = await client.tools.rag_query( # Assuming a tool name like rag_query
    query="What were the key findings in the latest financial report?",
    # Parameters to control retrieval, e.g.:
    # index_name="financial_reports",
    # top_k=3, 
    provider="anthropic",
    model="claude-3-5-haiku-20241022" # Model to generate the answer based on context
)

if rag_response["success"]:
    print(f"RAG Answer:\n{rag_response['answer']}")
    # Potentially include retrieved sources: rag_response['sources']
    print(f"Cost: ${rag_response['cost']:.6f}")
else:
    print(f"Error: {rag_response['error']}")
```

### Fused Search (Keyword + Semantic)

To perform a hybrid search combining keyword relevance and semantic similarity using Marqo:

```python
fused_search_response = await client.tools.fused_search( # Assuming a tool name like fused_search
    query="impact of AI on software development productivity",
    # Parameters for Marqo index and tuning:
    # index_name="tech_articles",
    # keyword_weight=0.3, # Weight for keyword score (0.0 to 1.0)
    # semantic_weight=0.7, # Weight for semantic score (0.0 to 1.0)
    # top_n=5,
    # filter_string="year > 2023"
)

if fused_search_response["success"]:
    print(f"Fused Search Results ({len(fused_search_response['results'])} hits):")
    for hit in fused_search_response["results"]:
        print(f" - Score: {hit['_score']:.4f}, ID: {hit['_id']}, Content: {hit.get('text', '')[:100]}...")
else:
    print(f"Error: {fused_search_response['error']}")
```

### Local Text Processing

To perform local, offline text operations without calling an LLM API:

```python
# Assuming a tool that bundles local text functions
local_process_response = await client.tools.process_local_text( 
    text="  Extra   spaces   and\nnewlines\t here.  ",
    operations=[
        {"action": "trim_whitespace"},
        {"action": "normalize_newlines"},
        {"action": "lowercase"}
    ]
)

if local_process_response["success"]:
    print(f"Processed Text: '{local_process_response['processed_text']}'")
else:
    print(f"Error: {local_process_response['error']}")
```

### Browser Automation Example: Getting Started and Basic Interaction

```python
# Agent uses the gateway to open a browser, navigate, and extract text

# Initialize the browser (optional, defaults can be used)
init_response = await client.tools.browser_init(headless=True) # Run without GUI
if not init_response["success"]:
    print(f"Browser init failed: {init_response.get('error')}")
    # Handle error...

# Navigate to a page
nav_response = await client.tools.browser_navigate(
    url="https://example.com",
    wait_until="load"
)
if nav_response["success"]:
    print(f"Navigated to: {nav_response['url']}, Title: {nav_response['title']}")
    # Agent can use the snapshot for context: nav_response['snapshot']
else:
    print(f"Navigation failed: {nav_response.get('error')}")
    # Handle error...

# Extract the heading text
text_response = await client.tools.browser_get_text(selector="h1")
if text_response["success"]:
    print(f"Extracted text: {text_response['text']}")

# Close the browser when done
close_response = await client.tools.browser_close()
print(f"Browser closed: {close_response['success']}")

### Running a Model Tournament

To compare the outputs of multiple models on a specific task (e.g., code generation):

```python
# Assuming a tournament tool
tournament_response = await client.tools.run_model_tournament(
    task_type="code_generation",
    prompt="Write a Python function to calculate the factorial of a number.",
    competitors=[
        {"provider": "openai", "model": "gpt-4.1-mini"},
        {"provider": "anthropic", "model": "claude-3-opus-20240229"}, # Higher-end model for comparison
        {"provider": "deepseek", "model": "deepseek-coder"}
    ],
    evaluation_criteria=["correctness", "efficiency", "readability"],
    # Optional: ground_truth="def factorial(n): ..." 
)

if tournament_response["success"]:
    print("Tournament Results:")
    # tournament_response['results'] would contain rankings, scores, outputs
    for rank, result in enumerate(tournament_response.get("ranking", [])):
        print(f"  {rank+1}. {result['provider']}/{result['model']} - Score: {result['score']:.2f}")
    print(f"Total Cost: ${tournament_response['total_cost']:.6f}")
else:
    print(f"Error: {tournament_response['error']}")

```

*(More tool examples can be added here...)*

## Getting Started

### Installation

```bash
# Install uv if you don't already have it:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/llm_gateway_mcp_server.git
cd llm_gateway_mcp_server

# Install in venv using uv:
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[all]"
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# API Keys (at least one provider required)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENROUTER_API_KEY=your_openrouter_key

# Server Configuration
SERVER_PORT=8013
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
# Start the MCP server with all tools
python -m llm_gateway.cli.main run

# Start the server with specific tools only
python -m llm_gateway.cli.main run --include-tools generate_completion read_file write_file

# Start the server excluding specific tools
python -m llm_gateway.cli.main run --exclude-tools browser_automation

# Or with Docker
docker compose up
```

Once running, the server will be available at `http://localhost:8013`.

## CLI Commands

LLM Gateway comes with a command-line interface for server management and tool interaction:

```bash
# Show available commands
llm-gateway --help

# Start the server
llm-gateway run [options]

# List available providers
llm-gateway providers

# List available tools
llm-gateway tools [--category CATEGORY]

# Test a provider
llm-gateway test openai --model gpt-4.1-mini

# Generate a completion
llm-gateway complete --provider anthropic --prompt "Hello, world!"

# Check cache status
llm-gateway cache --status
```

Each command has additional options that can be viewed with `llm-gateway COMMAND --help`.

## Advanced Configuration

While the `.env` file is convenient for basic setup, the LLM Gateway offers more detailed configuration options primarily managed through environment variables.

### Server Configuration

- `SERVER_HOST`: (Default: `127.0.0.1`) The network interface the server listens on. Use `0.0.0.0` to listen on all interfaces (necessary for Docker or external access).
- `SERVER_PORT`: (Default: `8013`) The port the server listens on.
- `API_PREFIX`: (Default: `/`) The URL prefix for the API endpoints.

### Tool Filtering

The LLM Gateway allows selectively choosing which MCP tools to register, helping manage complexity or reduce resource usage:

```bash
# List all available tools
llm-gateway tools

# List tools in a specific category
llm-gateway tools --category filesystem

# Start the server with only specific tools
llm-gateway run --include-tools read_file write_file generate_completion

# Start the server excluding specific tools
llm-gateway run --exclude-tools browser_automation marqo_fused_search
```

This feature is particularly useful when:
- You need a lightweight version of the gateway for a specific purpose
- Some tools are causing conflicts or excessive resource usage
- You want to restrict what capabilities are available to agents

### Logging Configuration

- `LOG_LEVEL`: (Default: `INFO`) Controls the verbosity of logs. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- `USE_RICH_LOGGING`: (Default: `true`) Use Rich library for colorful, formatted console logs. Set to `false` for plain text logs (better for file redirection or some log aggregation systems).
- `LOG_FORMAT`: (Optional) Specify a custom log format string.
- `LOG_TO_FILE`: (Optional, e.g., `gateway.log`) Path to a file where logs should also be written.

### Cache Configuration

- `CACHE_ENABLED`: (Default: `true`) Enable or disable caching globally.
- `CACHE_TTL`: (Default: `86400` seconds, i.e., 24 hours) Default Time-To-Live for cached items. Specific tools might override this.
- `CACHE_TYPE`: (Default: `memory`) The type of cache backend. Options might include `memory`, `redis`, `diskcache`. (*Note: Check current implementation for supported types*).
- `CACHE_MAX_SIZE`: (Optional) Maximum number of items or memory size for the cache.
- `REDIS_URL`: (Required if `CACHE_TYPE=redis`) Connection URL for the Redis cache server (e.g., `redis://localhost:6379/0`).

### Provider Timeouts & Retries

- `PROVIDER_TIMEOUT`: (Default: `120` seconds) Default timeout for requests to LLM provider APIs.
- `PROVIDER_MAX_RETRIES`: (Default: `3`) Default number of retries for failed provider requests (e.g., due to temporary network issues or rate limits).
- Specific provider timeouts/retries might be configurable via dedicated variables like `OPENAI_TIMEOUT`, `ANTHROPIC_MAX_RETRIES`, etc. (*Note: Check current implementation*).

### Tool-Specific Configuration

- Some tools might have their own specific environment variables for configuration (e.g., `MARQO_URL` for fused search, default chunking parameters). Refer to the documentation or source code of individual tools.

*Always ensure your environment variables are set correctly before starting the server. Changes often require a server restart.* 

## Deployment Considerations

While running the server directly with `python` or `docker compose up` is suitable for development and testing, consider the following for more robust or production deployments:

### 1. Running as a Background Service

To ensure the gateway runs continuously and restarts automatically on failure or server reboot, use a process manager:

- **`systemd` (Linux):** Create a service unit file (e.g., `/etc/systemd/system/llm-gateway.service`) to manage the process. This allows commands like `sudo systemctl start|stop|restart|status llm-gateway`.
- **`supervisor`:** A popular process control system written in Python. Configure `supervisord` to monitor and control the gateway process.
- **Docker Restart Policies:** If using Docker (standalone or Compose), configure appropriate restart policies (e.g., `unless-stopped` or `always`) in your `docker run` command or `docker-compose.yml` file.

### 2. Using a Reverse Proxy (Nginx/Caddy/Apache)

Placing a reverse proxy in front of the LLM Gateway is highly recommended:

- **HTTPS/SSL Termination:** The proxy can handle SSL certificates (e.g., using Let's Encrypt with Caddy or Certbot with Nginx/Apache), encrypting traffic between clients and the proxy.
- **Load Balancing:** If you need to run multiple instances of the gateway for high availability or performance, the proxy can distribute traffic among them.
- **Path Routing:** Map external paths (e.g., `https://api.yourdomain.com/llm-gateway/`) to the internal gateway server (`http://localhost:8013`).
- **Security Headers:** Add important security headers (like CSP, HSTS).
- **Buffering/Caching:** Some proxies offer additional request/response buffering or caching capabilities.

*Example Nginx `location` block (simplified):*
```nginx
location /llm-gateway/ {
    proxy_pass http://127.0.0.1:8013/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    # Add configurations for timeouts, buffering, etc.
}
```

### 3. Container Orchestration (Kubernetes/Swarm)

If deploying in a containerized environment:

- **Health Checks:** Implement and configure health check endpoints (e.g., the `/healthz` mentioned earlier) in your deployment manifests so the orchestrator can monitor the service's health.
- **Configuration:** Use ConfigMaps and Secrets (Kubernetes) or equivalent mechanisms to manage environment variables and API keys securely, rather than hardcoding them in images or relying solely on `.env` files.
- **Resource Limits:** Define appropriate CPU and memory requests/limits for the gateway container to ensure stable performance and prevent resource starvation.
- **Service Discovery:** Utilize the orchestrator's service discovery mechanisms instead of hardcoding IP addresses or hostnames.

### 4. Resource Allocation

- Ensure the host machine or container has sufficient **RAM**, especially if using in-memory caching or processing large documents/requests.
- Monitor **CPU usage**, particularly under heavy load or when multiple complex operations run concurrently.

## Cost Savings With Delegation

Using LLM Gateway for delegation can yield significant cost savings:

| Task | Claude 3.7 Direct | Delegated to Cheaper LLM | Savings |
|------|-------------------|--------------------------|---------|
| Summarizing 100-page document | $4.50 | $0.45 (Gemini Flash) | 90% |
| Extracting data from 50 records | $2.25 | $0.35 (gpt-4.1-mini) | 84% |
| Generating 20 content ideas | $0.90 | $0.12 (DeepSeek) | 87% |
| Processing 1,000 customer queries | $45.00 | $7.50 (Mixed delegation) | 83% |

These savings are achieved while maintaining high-quality outputs by letting Claude focus on high-level reasoning and orchestration while delegating mechanical tasks to cost-effective models.

## Why AI-to-AI Delegation Matters

The strategic importance of AI-to-AI delegation extends beyond simple cost savings:

### Democratizing Advanced AI Capabilities

By enabling powerful models like Claude 3.7, GPT-4o, and others to delegate effectively, we:
- Make advanced AI capabilities accessible at a fraction of the cost
- Allow organizations with budget constraints to leverage top-tier AI capabilities
- Enable more efficient use of AI resources across the industry

### Economic Resource Optimization

AI-to-AI delegation represents a fundamental economic optimization:
- Complex reasoning, creativity, and understanding are reserved for top-tier models
- Routine data processing, extraction, and simpler tasks go to cost-effective models
- The overall system achieves near-top-tier performance at a fraction of the cost
- API costs become a controlled expenditure rather than an unpredictable liability

### Sustainable AI Architecture

This approach promotes more sustainable AI usage:
- Reduces unnecessary consumption of high-end computational resources
- Creates a tiered approach to AI that matches capabilities to requirements
- Allows experimental work that would be cost-prohibitive with top-tier models only
- Creates a scalable approach to AI integration that can grow with business needs

### Technical Evolution Path

LLM Gateway represents an important evolution in AI application architecture:
- Moving from monolithic AI calls to distributed, multi-model workflows
- Enabling AI-driven orchestration of complex processing pipelines
- Creating a foundation for AI systems that can reason about their own resource usage
- Building toward self-optimizing AI systems that make intelligent delegation decisions

### The Future of AI Efficiency

LLM Gateway points toward a future where:
- AI systems actively manage and optimize their own resource usage
- Higher-capability models serve as intelligent orchestrators for entire AI ecosystems
- AI workflows become increasingly sophisticated and self-organizing
- Organizations can leverage the full spectrum of AI capabilities in cost-effective ways

This vision of efficient, self-organizing AI systems represents the next frontier in practical AI deployment, moving beyond the current pattern of using single models for every task.

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
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │  Tournament   │  │    Code       │  │   Multi-Agent │        │
│  │     Tools     │  │  Extraction   │  │  Coordination │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │   RAG Tools   │  │ Local Text    │  │  Meta Tools   │        │
│  │               │  │    Tools      │  │               │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐                           │
│  │ Browser Tools │  │ Filesystem    │                           │
│  │ (Playwright)  │  │    Tools      │                           │
│  └───────────────┘  └───────────────┘                           │
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
  - OpenAI (gpt-4.1-mini, GPT-4o, GPT-4o mini)
  - Anthropic (Claude 3.7 series)
  - Google (Gemini Pro, Gemini Flash, Gemini Flash Light)
  - DeepSeek (DeepSeek-Chat, DeepSeek-Reasoner)
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

### Tournament and Benchmarking

- **Model Competitions**:
  - Run competitions between different models and configurations
  - Compare code generation capabilities across providers
  - Generate statistical performance reports
  - Store competition results for historical analysis

- **Code Extraction**:
  - Extract clean code from model responses
  - Analyze and validate extracted code
  - Support for multiple programming languages

### Vector Operations

- **Embedding Service**:
  - Efficient text embedding generation
  - Embedding caching to reduce API costs
  - Batched processing for performance

- **Semantic Search**:
  - Find semantically similar content
  - Configurable similarity thresholds
  - Fast vector operations

- **Advanced Fused Search (Marqo)**:
  - Leverages Marqo for combined keyword and semantic search
  - Tunable weighting between keyword and vector relevance
  - Supports complex filtering and faceting

### Retrieval-Augmented Generation (RAG)

- **Contextual Generation**:
  - Augments LLM prompts with relevant retrieved information
  - Improves factual accuracy and reduces hallucinations
  - Integrates with vector search and document stores

- **Workflow Integration**:
  - Seamlessly combine document retrieval with generation tasks
  - Customizable retrieval and generation strategies

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

### Local Text Processing

- **Offline Operations**:
  - Provides tools for text manipulation that run locally, without API calls
  - Includes functions for cleaning, formatting, and basic analysis
  - Useful for pre-processing text before sending to LLMs or post-processing results

### Browser Automation (Playwright)
- **Capabilities:** Enables agents to control a web browser instance (Chromium, Firefox, WebKit) via Playwright.
- **Actions:** Supports navigation, clicking elements, typing text, selecting options, handling checkboxes, taking screenshots (full page, element, viewport), generating PDFs, downloading/uploading files, executing JavaScript, and managing browser tabs.
- **State Management:** Maintains browser sessions, contexts, and pages. Provides tools to initialize, close, and install browsers.
- **Agent Feedback:** Many tools return a `snapshot` of the page's accessibility tree, URL, and title after an action, giving the agent context about the resulting page state.
- **Configuration:** Allows setting headless mode, user data directories for persistence, timeouts, and specific browser executables.

### Meta Operations

- **Introspection and Management**:
  - Tools for querying server capabilities and status
  - May include functions for managing configurations or tool settings dynamically
  - Facilitates more complex agent interactions and self-management

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

- **Health Monitoring**:
  - Endpoint health checks (/healthz)
  - Resource usage monitoring
  - Provider availability tracking
  - Error rate statistics

- **Command-Line Interface**:
  - Rich interactive CLI for server management
  - Direct tool invocation from command line
  - Configuration management
  - Cache and server status inspection

## Tool Usage Examples

This section provides examples of how an MCP client (like Claude 3.7) would invoke specific tools provided by the LLM Gateway. These examples assume you have an initialized `mcp.client.Client` instance named `client` connected to the gateway.

### Basic Completion

To get a simple text completion from a chosen provider:

```python
response = await client.tools.completion(
    prompt="Write a short poem about a robot learning to dream.",
    provider="openai",  # Or "anthropic", "gemini", "deepseek"
    model="gpt-4.1-mini", # Specify the desired model
    max_tokens=100,
    temperature=0.7
)

if response["success"]:
    print(f"Completion: {response['completion']}")
    print(f"Cost: ${response['cost']:.6f}")
else:
    print(f"Error: {response['error']}")
```

### Document Summarization

To summarize a piece of text, potentially delegating to a cost-effective model:

```python
document_text = "... your long document content here ..."

summary_response = await client.tools.summarize_document(
    document=document_text,
    provider="gemini",
    model="gemini-2.0-flash-lite", # Using a cheaper model for summarization
    format="bullet_points", # Options: "paragraph", "bullet_points"
    max_length=150 # Target summary length in tokens (approximate)
)

if summary_response["success"]:
    print(f"Summary:\n{summary_response['summary']}")
    print(f"Cost: ${summary_response['cost']:.6f}")
else:
    print(f"Error: {summary_response['error']}")
```

### Entity Extraction

To extract specific types of entities from text:

```python
text_to_analyze = "Apple Inc. announced its quarterly earnings on May 5th, 2024, reporting strong iPhone sales from its headquarters in Cupertino."

entity_response = await client.tools.extract_entities(
    document=text_to_analyze,
    entity_types=["organization", "date", "product", "location"],
    provider="openai",
    model="gpt-4.1-mini"
)

if entity_response["success"]:
    print(f"Extracted Entities: {entity_response['entities']}")
    print(f"Cost: ${entity_response['cost']:.6f}")
else:
    print(f"Error: {entity_response['error']}")
```

### Executing an Optimized Workflow

To run a multi-step workflow where the gateway optimizes model selection for each step:

```python
doc_content = "... content for workflow processing ..."

workflow_definition = [
    {
        "name": "Summarize",
        "operation": "summarize_document",
        "provider_preference": "cost", # Prioritize cheaper models
        "params": {"format": "paragraph"},
        "input_from": "original",
        "output_as": "step1_summary"
    },
    {
        "name": "ExtractKeywords",
        "operation": "extract_keywords", # Assuming an extract_keywords tool exists
        "provider_preference": "speed",
        "params": {"count": 5},
        "input_from": "step1_summary",
        "output_as": "step2_keywords"
    }
]

workflow_response = await client.tools.execute_optimized_workflow(
    documents=[doc_content],
    workflow=workflow_definition
)

if workflow_response["success"]:
    print("Workflow executed successfully.")
    print(f"Results: {workflow_response['results']}") # Contains outputs like step1_summary, step2_keywords
    print(f"Total Cost: ${workflow_response['total_cost']:.6f}")
    print(f"Processing Time: {workflow_response['processing_time']:.2f}s")
else:
    print(f"Workflow Error: {workflow_response['error']}")
```

### Listing Available Tools (Meta Tool)

To dynamically discover the tools currently registered and available on the gateway:

```python
# Assuming a meta-tool for listing capabilities
list_tools_response = await client.tools.list_tools()

if list_tools_response["success"]:
    print("Available Tools:")
    for tool_name, tool_info in list_tools_response["tools"].items():
        print(f"- {tool_name}: {tool_info.get('description', 'No description')}")
        # You might also get parameters, etc.
else:
    print(f"Error listing tools: {list_tools_response['error']}")

```

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

### Model Benchmarking and Selection

Organizations can use the tournament features to:

- Run controlled competitions between different models
- Generate quantitative performance metrics
- Make data-driven decisions on model selection
- Build custom model evaluation frameworks

## Security Considerations

When deploying and operating the LLM Gateway, consider the following security aspects:

1.  **API Key Management:**
    *   **Never hardcode API keys** in your source code.
    *   Use environment variables (`.env` file for local development, system environment variables, or secrets management tools like HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager for production).
    *   Ensure the `.env` file (if used) has strict file permissions (readable only by the user running the gateway).
    *   Rotate keys periodically and revoke any suspected compromised keys immediately.

2.  **Network Exposure & Access Control:**
    *   By default, the server binds to `127.0.0.1`, only allowing local connections. Only change `SERVER_HOST` to `0.0.0.0` if you intend to expose it externally, and ensure proper controls are in place.
    *   **Use a reverse proxy** (Nginx, Caddy, etc.) to handle incoming connections. This allows you to manage TLS/SSL encryption, apply access controls (e.g., IP allow-listing), and potentially add gateway-level authentication.
    *   Employ **firewall rules** on the host machine or network to restrict access to the `SERVER_PORT` only from trusted sources (like the reverse proxy or specific internal clients).

3.  **Authentication & Authorization:**
    *   The gateway itself may not have built-in user authentication. Access control typically relies on network security (firewalls, VPNs) and potentially authentication handled by a reverse proxy (e.g., Basic Auth, OAuth2 proxy).
    *   Ensure that only authorized clients (like your trusted AI agents or applications) can reach the gateway endpoint.

4.  **Rate Limiting & Abuse Prevention:**
    *   Implement **rate limiting** at the reverse proxy level or using dedicated middleware to prevent denial-of-service attacks or excessive API usage (which can incur high costs).

5.  **Input Validation:**
    *   While LLM inputs are generally text, be mindful if any tools interpret inputs in ways that could lead to vulnerabilities (e.g., if a tool were to execute code based on input). Sanitize or validate inputs where appropriate for the specific tool's function.

6.  **Dependency Security:**
    *   Regularly update dependencies (`uv pip install --upgrade ...` or similar) to patch known vulnerabilities in third-party libraries.
    *   Consider using security scanning tools (like `pip-audit` or GitHub Dependabot alerts) to identify vulnerable dependencies.

7.  **Logging:**
    *   Be aware that `DEBUG` level logging might log full prompts and responses, potentially including sensitive information. Configure `LOG_LEVEL` appropriately for your environment and ensure log files have proper permissions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Model Context Protocol](https://github.com/mpctechdebt/mcp) for the foundation of the API
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [Pydantic](https://docs.pydantic.dev/) for data validation
- [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management
- All the LLM providers making their models available via API
