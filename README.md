# Ultimate MCP Server

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/mpctechdebt/mcp)

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

The server is built on the [Model Context Protocol (MCP)](https://github.com/mpctechdebt/mcp), making it specifically designed to work with AI agents like Claude. All functionality is exposed through MCP tools that can be directly called by these agents, creating a seamless workflow for AI-to-AI delegation.

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

### 💰 Cost Optimization

API costs for advanced models can be substantial. Ultimate MCP Server helps reduce costs by:

- Routing appropriate tasks to cheaper models (e.g., $0.01/1K tokens vs $0.15/1K tokens)
- Implementing advanced caching to avoid redundant API calls
- Tracking and optimizing costs across providers
- Enabling cost-aware task routing decisions
- Handling routine processing with specialized non-LLM tools

### 🔄 Provider Abstraction

Avoid provider lock-in with a unified interface:

- Standard API for OpenAI, Anthropic (Claude), Google (Gemini), and DeepSeek
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

### Autonomous Tool Documentation Refiner

The MCP server includes an autonomous documentation refinement system that improves the quality of tool documentation through an intelligent, iterative process:

- **Agent Simulation**: Tests how LLMs interpret documentation to identify ambiguities
- **Adaptive Testing**: Generates and executes diverse test cases based on tool schemas
- **Failure Analysis**: Uses LLM ensembles to diagnose documentation weaknesses
- **Smart Improvements**: Proposes targeted enhancements to descriptions, schemas, and examples
- **Iterative Refinement**: Continuously improves documentation until tests consistently pass

This feature significantly reduces manual documentation maintenance, improves tool usability for LLM agents, and helps identify edge cases that human writers might miss. While powerful, it does consume LLM resources and works best for tools with structured input schemas.

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

The Ultimate MCP Server includes powerful OCR (Optical Character Recognition) tools that leverage LLMs to improve text extraction from PDFs and images:

- **Extract Text from PDF**: Extract and enhance text from PDF documents using direct extraction or OCR.
- **Process Image OCR**: Extract and enhance text from images with preprocessing options.
- **Enhance OCR Text**: Improve existing OCR text using LLM-based correction and formatting.
- **Analyze PDF Structure**: Get information about a PDF's structure without full text extraction.
- **Batch Process Documents**: Process multiple documents with concurrent execution.

### OCR Installation

OCR tools require additional dependencies. Install them with:

```bash
pip install 'ultimate-mcp-server[ocr]'
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

### Claude Using Ultimate MCP Server for Document Analysis

This example shows how Claude can use the Ultimate MCP Server to process a document by delegating tasks to cheaper models:

```python
import asyncio
from mcp.client import Client

async def main():
    # Claude would use this client to connect to the Ultimate MCP Server
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

### Browser Automation for Research

```python
# Use Playwright integration to perform web research
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
        "llm_model": "gemini/gemini-2.0-pro"
    }
)

print(f"Research report generated with {len(result['extracted_data'])} sources")
print(f"Processing time: {result['processing_time']:.2f}s")
print(result['report'])
```

### Cognitive Memory System Usage

```python
# Create a workflow to represent a coherent agent session
workflow = await client.tools.create_workflow(
    title="Financial Research Project",
    description="Research on quantum computing impacts on finance",
    goal="Produce investment recommendations based on quantum computing trends"
)

# Record an agent's action and store the thought process
action = await client.tools.record_action_start(
    workflow_id=workflow["workflow_id"],
    action_type="research",
    reasoning="Need to gather information on quantum computing financial applications",
    title="Initial research on quantum finance"
)

# Create a thought chain to capture the agent's reasoning process
thought_chain = await client.tools.create_thought_chain(
    workflow_id=workflow["workflow_id"],
    title="Investment thesis development",
    initial_thought="Quantum computing may create arbitrage opportunities in markets",
    initial_thought_type="hypothesis"
)

# Add a follow-up thought to the chain
thought = await client.tools.record_thought(
    workflow_id=workflow["workflow_id"],
    thought_chain_id=thought_chain["thought_chain_id"],
    content="Quantum computing excels at optimization problems relevant to portfolio construction",
    thought_type="evidence"
)

# Store an important fact in semantic memory
memory = await client.tools.store_memory(
    workflow_id=workflow["workflow_id"],
    content="IBM's Eagle processor achieved 127 qubits in November 2023",
    memory_type="fact",
    memory_level="semantic",
    importance=8.5,
    tags=["quantum_computing", "hardware", "ibm"]
)

# Later, search for relevant memories
search_results = await client.tools.hybrid_search_memories(
    query="recent quantum hardware advances",
    workflow_id=workflow["workflow_id"],
    memory_type="fact",
    semantic_weight=0.7,
    keyword_weight=0.3
)

# Generate a reflection on the research process
reflection = await client.tools.generate_reflection(
    workflow_id=workflow["workflow_id"],
    reflection_type="insights"
)
```

### Excel Spreadsheet Automation

```python
# Create a financial model in Excel with natural language instructions
result = await client.tools.excel_execute(
    instruction="Create a 5-year financial projection model with revenue growth of 15% annually, "
               "cost of goods at 40% of revenue, operating expenses growing at 7% annually, "
               "and include charts for revenue, profit, and key ratios. Format as a professional "
               "financial statement with proper headers and currency formatting.",
    file_path="financial_model.xlsx",
    operation_type="create",
    show_excel=False  # Run Excel in the background
)

# Learn from an existing template and adapt it to a new scenario
adaptation_result = await client.tools.excel_learn_and_apply(
    exemplar_path="templates/saas_financial_model.xlsx",
    output_path="healthcare_saas_model.xlsx",
    adaptation_context="Adapt this SaaS financial model template for a healthcare SaaS startup "
                      "with slower customer acquisition but higher lifetime value. The healthcare "
                      "regulatory environment requires higher compliance costs and longer sales cycles. "
                      "Assume 3-year projections with seed funding of $2M in year 1."
)

# Analyze and optimize complex formulas in an existing model
formula_analysis = await client.tools.excel_analyze_formulas(
    file_path="financial_model.xlsx",
    sheet_name="Valuation",
    cell_range="D15:G25",
    analysis_type="optimize",
    detail_level="detailed"
)
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

### Entity Relation Graph Example

```python
# Extract entity relationships from a document
entity_graph = await client.tools.extract_entity_relations(
    document="Amazon announced today that it has acquired Anthropic, a leading AI research company. "
             "The deal, worth $4 billion, will help Amazon compete with Microsoft and Google in the AI race. "
             "Anthropic CEO Dario Amodei will continue to lead the company as a subsidiary of Amazon.",
    entity_types=["organization", "person", "money", "date"],
    relationship_types=["acquired", "worth", "compete_with", "lead"],
    include_visualization=True
)

# Query the entity graph
acquisition_info = await client.tools.query_entity_graph(
    query="Which companies has Amazon acquired and for how much?",
    graph_data=entity_graph["graph_data"]
)
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
```

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

## Autonomous Documentation Refiner

The Ultimate MCP Server includes a powerful feature for autonomously analyzing, testing, and refining the documentation of registered MCP tools. This feature, implemented in `ultimate/tools/docstring_refiner.py`, helps improve the usability and reliability of tools when invoked by Large Language Models (LLMs) like Claude.

### How It Works

The documentation refiner follows a methodical, iterative approach:

1. **Agent Simulation**: Simulates how an LLM agent would interpret the current documentation to identify potential ambiguities or missing information
2. **Adaptive Test Generation**: Creates diverse test cases based on the tool's schema, simulation results, and failures from previous iterations
3. **Schema-Aware Testing**: Validates generated test cases against the schema before execution, then executes valid tests against the actual tools
4. **Ensemble Failure Analysis**: Uses multiple LLMs to analyze failures in the context of the documentation used for that test run
5. **Structured Improvement Proposals**: Generates specific improvements to the description, schema (as JSON Patch operations), and usage examples
6. **Validated Schema Patching**: Applies and validates proposed schema patches in-memory
7. **Iterative Refinement**: Repeats the cycle until tests consistently pass or a maximum iteration count is reached
8. **Optional Winnowing**: Performs a final pass to streamline documentation while preserving critical information

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

### Usage Example

```python
from ultimate_mcp_server.tools.docstring_refiner import refine_tool_documentation

# Refine specific tools
result = await refine_tool_documentation(
    tool_names=["search_tool", "data_processor"],
    max_iterations=3,
    refinement_model_config={"model": "gpt-4", "provider": "openai"},
    enable_winnowing=True,
    ctx=mcp_context
)

# Or refine all available tools
result = await refine_tool_documentation(
    refine_all_available=True,
    max_iterations=2,
    ctx=mcp_context
)
```

### Text Redline Tools

- **HTML Redline Generation**:
  - Create visual diff comparisons between text or HTML documents
  - Highlight insertions, deletions, and text changes with professional formatting
  - Support for move detection and proper attribution of changes
  - Generate standalone HTML output with navigation and styling
  - Compare documents at word or character level granularity

- **Document Comparison**:
  - Compare original and modified text with intelligent format detection
  - Support for various document formats including plain text, HTML, and Markdown
  - Produce visually intuitive redlines highlighting all differences
  - Track insertions, deletions, moves, and attribute changes

### HTML to Markdown Conversion

- **Intelligent Format Detection**:
  - Automatically detect content type (HTML, markdown, code, plain text)
  - Apply appropriate conversion strategies based on detected format
  - Preserve important structural elements during conversion

- **Content Extraction**:
  - Use multiple strategies (readability, trafilatura) to extract the main content
  - Filter out boilerplate, navigation, and unwanted elements
  - Preserve tables, links, and other important structures
  - Support batch processing of multiple documents

- **Markdown Optimization**:
  - Clean and normalize markdown formatting
  - Fix common markdown syntax issues
  - Improve readability with proper spacing and structure
  - Customize output format with various options

### Workflow Optimization

- **Cost Estimation and Comparison**:
  - Estimate LLM API costs before execution
  - Compare costs across different models for the same prompt
  - Recommend optimal models based on task requirements
  - Balance cost, quality, and speed considerations

- **Model Selection**:
  - Analyze task requirements to recommend suitable models
  - Filter by required capabilities (reasoning, coding, knowledge)
  - Consider budget constraints and quality requirements
  - Provide reasoned recommendations with justifications

- **Workflow Execution**:
  - Define and execute complex multi-stage processing pipelines
  - Handle dependencies between workflow stages
  - Support parallel execution of independent stages
  - Process data through different models and tools with optimal routing

### Local Text Processing

- **Offline Text Operations**:
  - Process text without API calls to LLMs
  - Clean and normalize text content
  - Handle whitespace, line breaks, and special characters
  - Convert between text formats and encodings

- **Text Normalization**:
  - Standardize formatting across documents
  - Remove redundant whitespace and control characters
  - Normalize quotes, dashes, and special symbols
  - Prepare text for further processing by LLMs

### Model Performance Benchmarking

The server includes a sophisticated system for measuring and comparing the empirical performance of different LLM providers and models:

- **Speed Measurement**: 
  - Track response times across models and providers
  - Measure tokens-per-second processing rates
  - Record latency statistics for different request types
  - Compare performance under varying load conditions

- **Performance Profiles**:
  - Generate detailed performance profiles for each model
  - Track consistency and reliability metrics
  - Compare advertised vs. actual performance
  - Identify optimal contexts for each model

- **Empirical Optimization**:
  - Make data-driven model selection decisions
  - Match model capabilities to task requirements
  - Optimize for speed, cost, or quality based on real data
  - Create performance benchmarks for system monitoring

```python
# Measure model speeds across providers
measurement_results = await client.tools.measure_model_speeds(
    prompt="Explain the process of photosynthesis in 100 words.",
    models=[
        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        {"provider": "openai", "model": "gpt-4.1-mini"},
        {"provider": "gemini", "model": "gemini-2.0-flash-lite"}
    ],
    iterations=5,  # Run multiple times for reliability
    include_details=True
)

# View performance results
for model_key, stats in measurement_results["results"].items():
    print(f"{model_key}:")
    print(f"  Average response time: {stats['avg_response_time']:.2f}s")
    print(f"  Tokens per second: {stats['tokens_per_second']:.2f}")
    print(f"  Cost per 1K tokens: ${stats['cost_per_1k_tokens']:.5f}")
    print(f"  Performance score: {stats['performance_score']:.1f}/10")
```

### Server-Sent Events (SSE) Support

The Ultimate MCP Server provides real-time streaming capabilities through Server-Sent Events (SSE):

- **Streaming Completions**:
  - Receive token-by-token updates in real-time
  - Enable progressive rendering of LLM responses
  - Implement responsive user interfaces with immediate feedback
  - Optimize perceived performance with partial results

- **Progress Monitoring**:
  - Track progress of long-running operations
  - Receive status updates for multi-stage workflows
  - Monitor resource usage during processing
  - Get immediate notification of completion

- **Event-Based Architecture**:
  - Subscribe to specific event types
  - Receive targeted notifications for relevant updates
  - Filter events based on custom criteria
  - Implement advanced event handling patterns

```python
# Example of using SSE client to stream completions
from ultimate_mcp_server.clients.sse import SSEClient

async def stream_completion():
    sse_client = SSEClient("http://localhost:8013/sse/completion")
    
    # Request parameters
    params = {
        "prompt": "Write a short story about a robot learning to paint.",
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "stream": True
    }
    
    # Connect to the SSE endpoint
    await sse_client.connect("/generate", params)
    
    # Process events as they arrive
    async for event in sse_client.events():
        if event.event == "token":
            # Display token as it arrives
            print(event.data, end="", flush=True)
        elif event.event == "completion":
            # Process the full completion
            print("\n\nFull completion received!")
            print(f"Total tokens: {event.data.get('total_tokens')}")
            print(f"Cost: ${event.data.get('cost'):.6f}")
        elif event.event == "error":
            print(f"\nError: {event.data.get('message')}")
```

### Multi-Model Synthesis

Beyond simply comparing model outputs, the Ultimate MCP Server provides advanced capabilities for synthesizing responses from multiple models:

- **Comparative Analysis**:
  - Analyze outputs from multiple models side by side
  - Identify strengths and weaknesses in each response
  - Detect contradictions and agreement points
  - Assess factual consistency across models

- **Response Synthesis**:
  - Combine best elements from multiple responses
  - Generate meta-responses that incorporate multiple perspectives
  - Create consensus outputs from diverse model inputs
  - Identify and resolve conflicts between model outputs

- **Collaborative Reasoning**:
  - Implement multi-model reasoning chains
  - Use specialized models for different reasoning steps
  - Build complex problem-solving workflows across models
  - Create ensemble approaches for critical applications

```python
# Compare and synthesize responses from multiple models
synthesis_result = await client.tools.compare_and_synthesize(
    prompt="What are the key ethical considerations in developing general AI systems?",
    models=[
        {"provider": "anthropic", "model": "claude-3-5-opus-20240229", "weight": 0.5},
        {"provider": "openai", "model": "gpt-4o", "weight": 0.3},
        {"provider": "gemini", "model": "gemini-2.0-pro", "weight": 0.2}
    ],
    synthesis_type="comprehensive",  # Options: basic, consensus, comprehensive
    include_comparisons=True
)

# Display the synthesized response and analysis
print(f"Synthesized Response:\n{synthesis_result['synthesized_response']}\n")
print("Comparative Analysis:")
for finding in synthesis_result["comparative_analysis"]:
    print(f"- {finding}")
```

### Extended Model Support

The Ultimate MCP Server includes support for a growing ecosystem of LLM providers, including newer and specialized models:

- **Grok Integration**:
  - Native support for xAI's Grok models
  - Optimization for Grok's specific capabilities and features
  - Dedicated parameter handling for optimal Grok performance
  - Full integration with the MCP server framework

- **DeepSeek Support**:
  - Support for DeepSeek's specialized coding and conversational models
  - Optimized parameter handling for DeepSeek models
  - Cost-efficient routing of appropriate tasks to DeepSeek
  - Integration with caching and analytics systems

- **OpenRouter Integration**:
  - Access to a wide variety of models through OpenRouter
  - Unified interface for specialized and niche models
  - Backup access to primary providers through alternative channels
  - Expanded model coverage without direct API integrations

```python
# Use Grok models through the unified interface
grok_response = await client.tools.generate_completion(
    prompt="Explain how transformer models work from first principles.",
    provider="xai",
    model="grok-1",
    temperature=0.7,
    max_tokens=500
)

# Use DeepSeek's specialized coding model
code_response = await client.tools.generate_completion(
    prompt="Write a Python function that implements the A* search algorithm.",
    provider="deepseek",
    model="deepseek-coder",
    temperature=0.3,
    max_tokens=800
)

# Access models through OpenRouter
openrouter_response = await client.tools.generate_completion(
    prompt="Write a detailed analysis of the current state of AI safety research.",
    provider="openrouter",
    model="or:meta/llama-3-70b-instruct",  # Access Meta's Llama 3 through OpenRouter
    temperature=0.5,
    max_tokens=1000
)
```

### Comprehensive Testing Framework

The Ultimate MCP Server includes a sophisticated testing infrastructure to ensure reliability and compatibility:

- **Automated Test Suite**:
  - 35+ comprehensive end-to-end tests
  - Validation of all key functionality
  - Cross-provider compatibility testing
  - Performance and reliability benchmarks

- **Intelligent Test Configuration**:
  - Environment-aware test execution
  - Graceful handling of missing API keys
  - Pattern-based error classification
  - Detailed test reporting

- **Continuous Validation**:
  - Track compatibility with updated models and APIs
  - Verify functionality across different environments
  - Ensure consistent behavior with varying inputs
  - Maintain reliability across server updates

```python
# Run the comprehensive test suite
from run_all_demo_scripts_and_check_for_errors import run_test_suite

results = run_test_suite(
    tests_to_run="all",  # Or specify individual tests
    include_providers=["openai", "anthropic", "gemini"],
    output_format="detailed",
    fail_fast=False
)

# View test results
print(f"Tests run: {results['total_tests']}")
print(f"Passed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Skipped: {results['skipped']} (missing dependencies or API keys)")

# Export detailed test report
with open("test_report.html", "w") as f:
    f.write(results["html_report"])
```

## Getting Started

### Installation

```bash
# Install uv if you don't already have it:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/ultimate_mcp_server.git
cd ultimate_mcp_server

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
python -m ultimate.cli.main run

# Start the server with specific tools only
python -m ultimate.cli.main run --include-tools generate_completion read_file write_file

# Start the server excluding specific tools
python -m ultimate.cli.main run --exclude-tools browser_automation

# Or with Docker
docker compose up
```

Once running, the server will be available at `http://localhost:8013`.

## CLI Commands

Ultimate MCP Server comes with a command-line interface for server management and tool interaction:

```bash
# Show available commands
ultimate-mcp-server --help

# Start the server
ultimate-mcp-server run [options]

# List available providers
ultimate-mcp-server providers

# List available tools
ultimate-mcp-server tools [--category CATEGORY]

# Test a provider
ultimate-mcp-server test openai --model gpt-4.1-mini

# Generate a completion
ultimate-mcp-server complete --provider anthropic --prompt "Hello, world!"

# Check cache status
ultimate-mcp-server cache --status
```

Each command has additional options that can be viewed with `ultimate-mcp-server COMMAND --help`.

## Advanced Configuration

While the `.env` file is convenient for basic setup, the Ultimate MCP Server offers more detailed configuration options primarily managed through environment variables.

### Server Configuration

- `SERVER_HOST`: (Default: `127.0.0.1`) The network interface the server listens on. Use `0.0.0.0` to listen on all interfaces (necessary for Docker or external access).
- `SERVER_PORT`: (Default: `8013`) The port the server listens on.
- `API_PREFIX`: (Default: `/`) The URL prefix for the API endpoints.

### Tool Filtering

The Ultimate MCP Server allows selectively choosing which MCP tools to register, helping manage complexity or reduce resource usage:

```bash
# List all available tools
ultimate-mcp-server tools

# List tools in a specific category
ultimate-mcp-server tools --category filesystem

# Start the server with only specific tools
ultimate-mcp-server run --include-tools read_file write_file generate_completion

# Start the server excluding specific tools
ultimate-mcp-server run --exclude-tools browser_automation marqo_fused_search
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

- **`systemd` (Linux):** Create a service unit file (e.g., `/etc/systemd/system/ultimate-mcp-server.service`) to manage the process. This allows commands like `sudo systemctl start|stop|restart|status ultimate-mcp-server`.
- **`supervisor`:** A popular process control system written in Python. Configure `supervisord` to monitor and control the gateway process.
- **Docker Restart Policies:** If using Docker (standalone or Compose), configure appropriate restart policies (e.g., `unless-stopped` or `always`) in your `docker run` command or `docker-compose.yml` file.

### 2. Using a Reverse Proxy (Nginx/Caddy/Apache)

Placing a reverse proxy in front of the Ultimate MCP Server is highly recommended:

- **HTTPS/SSL Termination:** The proxy can handle SSL certificates (e.g., using Let's Encrypt with Caddy or Certbot with Nginx/Apache), encrypting traffic between clients and the proxy.
- **Load Balancing:** If you need to run multiple instances of the gateway for high availability or performance, the proxy can distribute traffic among them.
- **Path Routing:** Map external paths (e.g., `https://api.yourdomain.com/ultimate-mcp-server/`) to the internal gateway server (`http://localhost:8013`).
- **Security Headers:** Add important security headers (like CSP, HSTS).
- **Buffering/Caching:** Some proxies offer additional request/response buffering or caching capabilities.

*Example Nginx `location` block (simplified):*
```nginx
location /ultimate-mcp-server/ {
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

Using Ultimate MCP Server for delegation can yield significant cost savings:

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
- Specialized tasks are handled by purpose-built tools rather than general-purpose LLMs
- The overall system achieves near-top-tier performance at a fraction of the cost
- API costs become a controlled expenditure rather than an unpredictable liability

### Sustainable AI Architecture

This approach promotes more sustainable AI usage:
- Reduces unnecessary consumption of high-end computational resources
- Creates a tiered approach to AI that matches capabilities to requirements
- Allows experimental work that would be cost-prohibitive with top-tier models only
- Creates a scalable approach to AI integration that can grow with business needs

### Technical Evolution Path

Ultimate MCP Server represents an important evolution in AI application architecture:
- Moving from monolithic AI calls to distributed, multi-model workflows
- Enabling AI-driven orchestration of complex processing pipelines
- Creating a foundation for AI systems that can reason about their own resource usage
- Building toward self-optimizing AI systems that make intelligent delegation decisions

### The Future of AI Efficiency

Ultimate MCP Server points toward a future where:
- AI systems actively manage and optimize their own resource usage
- Higher-capability models serve as intelligent orchestrators for entire AI ecosystems
- AI workflows become increasingly sophisticated and self-organizing
- Organizations can leverage the full spectrum of AI capabilities in cost-effective ways

This vision of efficient, self-organizing AI systems represents the next frontier in practical AI deployment, moving beyond the current pattern of using single models for every task.

## Architecture

### How MCP Integration Works

The Ultimate MCP Server is built natively on the Model Context Protocol:

1. **MCP Server Core**: The gateway implements a full MCP server
2. **Tool Registration**: All capabilities are exposed as MCP tools
3. **Tool Invocation**: Claude and other AI agents can directly invoke these tools
4. **Context Passing**: Results are returned in MCP's standard format

This ensures seamless integration with Claude and other MCP-compatible agents.

### Component Diagram

```plaintext
┌─────────────┐         ┌───────────────────┐         ┌──────────────┐
│  Claude 3.7 │ ────────► Ultimate MCP      │ ────────► LLM Providers│
│   (Agent)   │ ◄──────── Server            │ ◄──────── (Multiple)   │
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
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Browser Tools │  │ Filesystem    │  │ Cognitive     │        │
│  │ (Playwright)  │  │    Tools      │  │ Memory System │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Excel         │  │ SQL Database  │  │ Entity        │        │
│  │ Automation    │  │ Interactions  │  │ Relation Graph│        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Audio         │  │ OCR           │  │ Text          │        │
│  │ Transcription │  │ Tools         │  │ Classification│        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow for Delegation

When Claude delegates a task to Ultimate MCP Server:

1. Claude sends an MCP tool invocation request
2. The Gateway receives the request via MCP protocol
3. The appropriate tool processes the request
4. The caching service checks if the result is already cached
5. If not cached, the optimization service selects the appropriate provider/model or specialized tool
6. If using an LLM, the provider layer sends the request to the selected LLM API
7. If using a specialized tool, the relevant module processes the request directly
8. The response is standardized, cached, and metrics are recorded
9. The MCP server returns the result to Claude

## Real-World Use Cases

### Advanced AI Agent Capabilities

Claude or other advanced AI agents can use Ultimate MCP Server to:

- Maintain persistent memory across sessions with the cognitive memory system
- Perform complex web research and automation with browser tools
- Create and analyze spreadsheets and financial models with Excel tools
- Access and manipulate databases with SQL tools
- Extract and process information from documents with OCR and document tools
- Build knowledge graphs with entity relation tools
- Generate and refine audio transcriptions
- Classify and analyze text across numerous dimensions

### Enterprise Workflow Automation

Organizations can leverage the Ultimate MCP Server for:

- Creating automated processes that combine web interactions, document processing, and data analysis
- Building intelligent document workflows that extract, transform, and load data
- Developing research assistants that gather, analyze, and synthesize information from multiple sources
- Creating financial modeling and analysis pipelines using Excel automation
- Maintaining knowledge bases with persistent cognitive memory

### Data Processing and Integration

Process and integrate data from multiple sources:

- Extract structured data from unstructured documents
- Create connections between entities and concepts using relationship graphs
- Transform and normalize data using SQL and spreadsheet tools
- Apply classification and categorization to large document collections
- Build searchable knowledge bases with vector search capabilities

### Research and Analysis

Research teams can use Ultimate MCP Server to:

- Automate web research across multiple sources
- Compare outputs from different models
- Process research papers efficiently
- Extract structured information from studies
- Track token usage and optimize research budgets
- Maintain persistent research memory and knowledge graphs

### Document Intelligence

Create end-to-end document processing systems:

- Apply OCR to extract text from images and PDFs
- Structure and normalize extracted information
- Classify documents automatically
- Identify key entities and relationships
- Generate summaries and insights
- Store results in searchable, queryable systems

### Financial Analysis and Modeling

Financial professionals can utilize:

- Excel automation for creating and maintaining complex financial models
- Database tools for accessing and analyzing financial data
- Browser automation for gathering market information
- Vector search for finding relevant financial insights
- Cognitive memory for tracking financial reasoning and decisions

## Cost Savings With Delegation

Using Ultimate MCP Server for delegation can yield significant cost savings:

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
- Specialized tasks are handled by purpose-built tools rather than general-purpose LLMs
- The overall system achieves near-top-tier performance at a fraction of the cost
- API costs become a controlled expenditure rather than an unpredictable liability

### Sustainable AI Architecture

This approach promotes more sustainable AI usage:
- Reduces unnecessary consumption of high-end computational resources
- Creates a tiered approach to AI that matches capabilities to requirements
- Allows experimental work that would be cost-prohibitive with top-tier models only
- Creates a scalable approach to AI integration that can grow with business needs

### Technical Evolution Path

Ultimate MCP Server represents an important evolution in AI application architecture:
- Moving from monolithic AI calls to distributed, multi-model workflows
- Enabling AI-driven orchestration of complex processing pipelines
- Creating a foundation for AI systems that can reason about their own resource usage
- Building toward self-optimizing AI systems that make intelligent delegation decisions

### The Future of AI Efficiency

Ultimate MCP Server points toward a future where:
- AI systems actively manage and optimize their own resource usage
- Higher-capability models serve as intelligent orchestrators for entire AI ecosystems
- AI workflows become increasingly sophisticated and self-organizing
- Organizations can leverage the full spectrum of AI capabilities in cost-effective ways

This vision of efficient, self-organizing AI systems represents the next frontier in practical AI deployment, moving beyond the current pattern of using single models for every task.

## Getting Started

### Installation

```bash
# Install uv if you don't already have it:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/ultimate_mcp_server.git
cd ultimate_mcp_server

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
python -m ultimate.cli.main run

# Start the server with specific tools only
python -m ultimate.cli.main run --include-tools generate_completion read_file write_file

# Start the server excluding specific tools
python -m ultimate.cli.main run --exclude-tools browser_automation

# Or with Docker
docker compose up
```

Once running, the server will be available at `http://localhost:8013`.

## CLI Commands

Ultimate MCP Server comes with a command-line interface for server management and tool interaction:

```bash
# Show available commands
ultimate-mcp-server --help

# Start the server
ultimate-mcp-server run [options]

# List available providers
ultimate-mcp-server providers

# List available tools
ultimate-mcp-server tools [--category CATEGORY]

# Test a provider
ultimate-mcp-server test openai --model gpt-4.1-mini

# Generate a completion
ultimate-mcp-server complete --provider anthropic --prompt "Hello, world!"

# Check cache status
ultimate-mcp-server cache --status
```

Each command has additional options that can be viewed with `ultimate-mcp-server COMMAND --help`.

## Advanced Configuration

While the `.env` file is convenient for basic setup, the Ultimate MCP Server offers more detailed configuration options primarily managed through environment variables.

### Server Configuration

- `SERVER_HOST`: (Default: `127.0.0.1`) The network interface the server listens on. Use `0.0.0.0` to listen on all interfaces (necessary for Docker or external access).
- `SERVER_PORT`: (Default: `8013`) The port the server listens on.
- `API_PREFIX`: (Default: `/`) The URL prefix for the API endpoints.

### Tool Filtering

The Ultimate MCP Server allows selectively choosing which MCP tools to register, helping manage complexity or reduce resource usage:

```bash
# List all available tools
ultimate-mcp-server tools

# List tools in a specific category
ultimate-mcp-server tools --category filesystem

# Start the server with only specific tools
ultimate-mcp-server run --include-tools read_file write_file generate_completion

# Start the server excluding specific tools
ultimate-mcp-server run --exclude-tools browser_automation marqo_fused_search
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

- **`systemd` (Linux):** Create a service unit file (e.g., `/etc/systemd/system/ultimate-mcp-server.service`) to manage the process. This allows commands like `sudo systemctl start|stop|restart|status ultimate-mcp-server`.
- **`supervisor`:** A popular process control system written in Python. Configure `supervisord` to monitor and control the gateway process.
- **Docker Restart Policies:** If using Docker (standalone or Compose), configure appropriate restart policies (e.g., `unless-stopped` or `always`) in your `docker run` command or `docker-compose.yml` file.

### 2. Using a Reverse Proxy (Nginx/Caddy/Apache)

Placing a reverse proxy in front of the Ultimate MCP Server is highly recommended:

- **HTTPS/SSL Termination:** The proxy can handle SSL certificates (e.g., using Let's Encrypt with Caddy or Certbot with Nginx/Apache), encrypting traffic between clients and the proxy.
- **Load Balancing:** If you need to run multiple instances of the gateway for high availability or performance, the proxy can distribute traffic among them.
- **Path Routing:** Map external paths (e.g., `https://api.yourdomain.com/ultimate-mcp-server/`) to the internal gateway server (`http://localhost:8013`).
- **Security Headers:** Add important security headers (like CSP, HSTS).
- **Buffering/Caching:** Some proxies offer additional request/response buffering or caching capabilities.

*Example Nginx `location` block (simplified):*
```nginx
location /ultimate-mcp-server/ {
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Model Context Protocol](https://github.com/mpctechdebt/mcp) for the foundation of the API
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [Pydantic](https://docs.pydantic.dev/) for data validation
- [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management
- All the LLM providers making their models available via API

### Meta Tools for Self-Improvement

- **Tool Information and Discovery**:
  - Query available MCP tools and their capabilities
  - Get detailed information on specific tool parameters and usage
  - Dynamically discover new tools and their functionalities
  - Access tool registries and categorizations

- **Tool Usage Recommendations**:
  - Receive AI-optimized advice for specific task types
  - Get task-specific tool combination recommendations
  - Obtain provider and model recommendations per task
  - Access best practices for common workflows

- **Multi-Provider Comparison**:
  - Compare outputs from multiple providers for the same prompt
  - Analyze differences in generated content across models
  - Synthesize best responses using weighted criteria
  - Generate comprehensive comparison reports

- **External API Integration**:
  - Dynamically register external REST APIs via OpenAPI specifications
  - Convert API endpoints into MCP-compatible tools
  - Perform authenticated API calls through standard tool interfaces
  - Maintain and refresh API integrations as needed

- **Self-Improvement and Documentation**:
  - Automatically generate LLM-optimized tool documentation
  - Analyze tool usage patterns for improvement
  - Get detailed instructions on complex tool ecosystems
  - Generate walkthroughs for common use cases

### Command-Line Text Tools

- **Local Text Processing**:
  - Execute powerful local text processing utilities directly through MCP tools
  - Access native performance and capabilities without API calls
  - Use established UNIX text processing paradigms securely

- **Ripgrep Integration**:
  - Search text patterns with high-performance regex capabilities
  - Process large files or entire directory trees efficiently
  - Get context around matches with sophisticated formatting
  - Filter by file types, patterns, and other attributes

- **AWK Processing**:
  - Transform text data with powerful pattern-action processing
  - Operate on columnar data with field-based operations
  - Perform calculations and aggregations across text data
  - Generate reports and reformatted output

- **Stream Editing with SED**:
  - Transform text with line-based pattern substitutions
  - Perform complex find-and-replace operations
  - Extract specific patterns from text data
  - Apply transformations to matched patterns

- **JSON Processing with JQ**:
  - Query and transform JSON data with specialized syntax
  - Extract, filter, and reformat JSON structures
  - Manipulate complex nested JSON objects and arrays
  - Convert between JSON formats and other representations

## Usage Examples

### Meta Tools for Tool Discovery

```python
# Get information about available tools
tools_info = await client.tools.get_tool_info()
print(f"Available tools: {len(tools_info['tools'])}")
for tool in tools_info['tools']:
    print(f"- {tool['name']}: {tool['description'][:100]}...")

# Get detailed information about a specific tool
tool_details = await client.tools.get_tool_info(tool_name="extract_entities")
print(f"Tool: {tool_details['name']}")
print(f"Description: {tool_details['description']}")
print(f"Parameters: {tool_details.get('parameters', {})}")

# Get task-specific tool recommendations
recommendations = await client.tools.get_tool_recommendations(
    task="extract names, dates, and organizations from a collection of news articles",
    constraints={"max_cost": 0.02, "priority": "speed"}
)

print(f"Recommended tools for the task:")
for tool in recommendations["tools"]:
    print(f"- {tool['tool']}: {tool['reason']}")

# Get LLM-specific instructions on how to use tools
instructions = await client.tools.get_llm_instructions(
    tool_name="chunk_document",
    task_type="summarization"
)
print(instructions["instructions"])
```

### Local Command-Line Text Processing

```python
# Use ripgrep to search for patterns in text
ripgrep_result = await client.tools.run_ripgrep(
    args_str="-i --json 'error|exception' -C 2",
    input_data="Line 1: Normal log\nLine 2: Error occurred in module X\nLine 3: Exception details\nLine 4: Normal operation resumed"
)
print(f"Found {ripgrep_result['stdout'].count('match')} matches")

# Use jq to process JSON data
jq_result = await client.tools.run_jq(
    args_str="'.items[] | select(.price > 50) | {name: .name, price: .price}'",
    input_data='{"items": [{"name": "Item A", "price": 30}, {"name": "Item B", "price": 75}, {"name": "Item C", "price": 120}]}'
)
print(jq_result["stdout"])

# Use awk to process columnar data
awk_result = await client.tools.run_awk(
    args_str="-F ',' '{ sum += $2 } END { print \"Total: \" sum }'",
    input_data="Product A,42\nProduct B,18\nProduct C,73"
)
print(f"Result: {awk_result['stdout'].strip()}")

# Use sed to transform text
sed_result = await client.tools.run_sed(
    args_str="'s/important/critical/g; s/error/ERROR/g'",
    input_data="This is an important warning.\nAn error occurred in the system."
)
print(sed_result["stdout"])
```

### Dynamic API Integration

```python
# Register an external API using its OpenAPI specification
api_registration = await client.tools.register_api(
    api_name="petstore",
    openapi_url="https://petstore.swagger.io/v2/swagger.json",
    cache_ttl=3600  # Cache responses for an hour
)

print(f"Registered {api_registration['tools_count']} API tools:")
for tool in api_registration['tools_registered']:
    print(f"- {tool}")

# List all registered APIs
apis = await client.tools.list_registered_apis()
for api_name, api_info in apis["apis"].items():
    print(f"{api_name}: {api_info['tools_count']} endpoints, base URL: {api_info['base_url']}")

# Call a dynamically registered API endpoint
pet_result = await client.tools.call_dynamic_tool(
    tool_name="petstore_getPetById",
    inputs={"petId": 123}
)
print(f"Retrieved pet: {pet_result['name']} (status: {pet_result['status']})")

# Create a new resource via the API
new_pet = await client.tools.call_dynamic_tool(
    tool_name="petstore_addPet",
    inputs={
        "body": {
            "id": 0,
            "name": "Fluffy",
            "category": {"id": 1, "name": "Dogs"},
            "status": "available"
        }
    }
)
print(f"Created new pet with ID: {new_pet['id']}")

# Unregister the API when no longer needed
unregister_result = await client.tools.unregister_api(api_name="petstore")
print(f"Unregistered {unregister_result['tools_count']} tools")
```

## Example Library and Testing Framework

The Ultimate MCP Server includes an extensive collection of 35+ end-to-end examples that serve as both comprehensive documentation and integration tests. These examples demonstrate real-world use cases and ensure all components work together correctly.

### Example Structure and Organization

The examples are organized by functionality and cover the entire range of tools available in the Ultimate MCP Server:

- **Model Integration Examples**: Demonstrate working with various LLM providers
  - `simple_completion_demo.py` - Basic completion functionality
  - `claude_integration_demo.py` - Working with Anthropic Claude models
  - `grok_integration_demo.py` - Integration with Grok models
  - `multi_provider_demo.py` - Using multiple providers simultaneously

- **Tool-Specific Examples**: Showcase individual specialized tools
  - `browser_automation_demo.py` - Web automation with Playwright
  - `audio_transcription_demo.py` - Speech-to-text conversion
  - `entity_relation_graph_demo.py` - Entity extraction and relationship mapping
  - `filesystem_operations_demo.py` - Secure file access and manipulation
  - `sql_database_interactions_demo.py` - Database queries and schema analysis
  - `text_redline_demo.py` - Document comparison and visualization

- **Optimization and Workflow Examples**: Illustrate complex orchestration
  - `cost_optimization.py` - Model selection based on cost/performance
  - `workflow_delegation_demo.py` - Multi-step task delegation across models
  - `research_workflow_demo.py` - End-to-end research processes

- **Advanced AI Features**: Demonstrate cutting-edge capabilities
  - `rag_example.py` - Retrieval-augmented generation
  - `tournament_code_demo.py` - Model competitions for code generation
  - `meta_api_demo.py` - Dynamic API integration

Each example is structured as a standalone application that:

1. Sets up the necessary environment
2. Initializes the Ultimate MCP Server or specific tools
3. Executes demonstrations with real inputs and outputs
4. Provides rich console output with progress tracking
5. Handles errors gracefully with proper logging

### Rich Visual Output

All examples use the [Rich](https://github.com/Textualize/rich) library to provide beautiful, informative console output with:

- Color-coded results and progress indicators
- Detailed tables with operation statistics
- Formatted JSON and code syntax highlighting
- Progress bars for long-running operations
- Panels to display results in an organized manner

Example console output for the `browser_automation_demo.py`:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━ Browser Initialization Demo ━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️ Starting browser initialization

┌────────┬────────────────────────────────┐
│ Metric │ Value                          │
├────────┼────────────────────────────────┤
│ Status │ Success                        │
│ Browser│ chromium                       │
└────────┴────────────────────────────────┘

╭──────────────────────── Page Screenshot ────────────────────────╮
│ Full page: false                                                │
│ Quality: 80                                                     │
│ Size: 268.42 KB                                                 │
╰───────────────────────────────────────────────────────────────── ╯
```

For model tournament examples, the output includes detailed tracking of model performances, cost statistics, and even visualizations of the competition results:

```
╭────────────────────── Tournament Results ───────────────────────╮
│ [1] claude-3-5-haiku-20241022: Score 8.7/10                    │
│     Cost: $0.00013                                             │
│                                                                │
│ [2] gpt-4.1-mini: Score 7.9/10                                │
│     Cost: $0.00021                                             │
│                                                                │
│ [3] gemini-2.0-flash-lite: Score 6.8/10                       │
│     Cost: $0.00008                                             │
╰────────────────────────────────────────────────────────────────╯
```

### Customizing Examples

The examples are designed to be easily adapted for your specific use cases:

- **API Keys**: Examples automatically use API keys from your environment
- **Model Selection**: Change the model variables to use different LLM providers
- **Input Data**: Modify the input text/prompts to match your domain
- **Parameters**: Adjust the parameters to experiment with different settings
- **Workflow Steps**: Customize multi-step workflows for your specific needs

Many examples include command line arguments to customize behavior:

```bash
# Run with a custom model
python examples/tournament_code_demo.py --task calculator

# Use your own input files
python examples/text_redline_demo.py --input-file1 my_document.txt --input-file2 revised_document.txt

# Adjust browser automation settings
python examples/browser_automation_demo.py --headless --max-steps 3
```

### Learning from Examples

These examples serve multiple educational purposes:

1. **Code Patterns**: Learn best practices for structuring AI applications
2. **Tool Selection**: Understand which tools to use for specific tasks
3. **Parameter Tuning**: See optimal parameters for different scenarios
4. **Error Handling**: Learn robust error handling patterns for AI applications
5. **Cost Optimization**: Discover strategies to minimize API costs
6. **Integration Techniques**: Understand how to combine multiple tools

For developers building their own applications with Ultimate MCP Server, these examples provide production-ready patterns that can be directly adapted to solve real-world problems.

### Comprehensive Testing Framework

The examples are designed to work together as a complete test suite using the `run_all_demo_scripts_and_check_for_errors.py` script. This script:

- Automatically discovers and runs all example scripts sequentially
- Tracks progress with detailed console output
- Validates expected outcomes against specific success criteria
- Handles known limitations (like missing API keys)
- Generates detailed reports of all test results

The testing framework includes sophisticated error detection that distinguishes between:

- Actual errors that indicate failures
- Expected messages that are part of normal operation
- Warnings that occur when optional features are unavailable

Example of the test framework configuration:

```python
# Configuration for a specific test
"sql_database_interactions_demo.py": {
    "expected_exit_code": 0,
    "allowed_stderr_patterns": [
        # Known data type limitations (not errors)
        r"Could not compute statistics for column customers\.signup_date: 'str' object has no attribute 'isoformat'",
        # Example database connection scenarios
        r"Connection failed: \(sqlite3\.OperationalError\) unable to open database file",
        # Standard setup messages
        r"Configuration not yet loaded\. Loading now\.\.\."
    ]
}
```

### Running the Example Suite

To run all examples and validate the entire Ultimate MCP Server functionality:

```bash
# Run the complete test suite
python run_all_demo_scripts_and_check_for_errors.py

# Run a specific example
python examples/browser_automation_demo.py

# Run with specific arguments
python examples/text_redline_demo.py --input-file1 doc1.txt --input-file2 doc2.txt
```

The test suite generates a comprehensive report showing which examples succeeded, which failed, and detailed information about any issues encountered.

These examples serve as living documentation, showing exactly how to use each feature of the Ultimate MCP Server in real-world scenarios while ensuring all components work together correctly.

### Analytics and Reporting

The Ultimate MCP Server includes a comprehensive analytics and reporting system that helps track and visualize your LLM usage:

- **Usage Metrics Tracking**: 
  - Monitor token usage across providers and models
  - Track costs with detailed breakdowns
  - Analyze request patterns and performance
  - Record success rates and error statistics

- **Real-Time Monitoring**:
  - View live usage statistics as they occur
  - Monitor active requests and processing
  - Get alerts for unusual patterns or errors
  - Watch cost accumulation in real-time

- **Detailed Reporting**:
  - Generate comprehensive cost reports by provider/model
  - View historical trends and usage patterns
  - Compare efficiency across different providers
  - Export data for external analysis

- **Optimization Insights**:
  - Identify cost-saving opportunities
  - Detect inefficient token usage patterns
  - Get recommendations for model selection
  - Find optimal prompt patterns

```python
# Get analytics metrics
metrics = get_metrics_tracker()
stats = metrics.get_stats()

# Display provider cost breakdown
for provider, data in stats["providers"].items():
    print(f"{provider}: ${data['cost']:.6f} (Tokens: {data['tokens']:,})")

# Get historical usage patterns
daily_usage = stats["daily_usage"]
for day in daily_usage:
    print(f"Date: {day['date']}, Requests: {day['requests']}, Cost: ${day['cost']:.6f}")

# Generate a cost report for a specific time period
report = await client.tools.generate_cost_report(
    start_date="2024-07-01",
    end_date="2024-07-31",
    group_by=["provider", "model"],
    include_charts=True,
    format="markdown"
)
```

### Prompt Templates and Management

The server includes a sophisticated prompt templating system that enables advanced prompt engineering:

- **Jinja2-Based Templates**:
  - Create reusable prompt templates with variables
  - Include conditionals and loops in templates
  - Compose templates from smaller components
  - Implement default values and fallbacks

- **Prompt Repository**:
  - Save and retrieve templates from persistent storage
  - Categorize templates by purpose or domain
  - Version control your prompts
  - Share templates across applications

- **Metadata and Documentation**:
  - Add rich metadata to your templates
  - Document expected inputs and outputs
  - Track template authorship and usage
  - Include usage examples and best practices

- **Template Optimization**:
  - Test templates with different inputs
  - Compare template performance
  - Iteratively improve prompts
  - Track token usage across template versions

```python
# Create a prompt template
template = PromptTemplate(
    template="""
You are an expert in {{field}}. 
Please explain {{concept}} in simple terms that a {{audience}} could understand.
""",
    template_id="simple_explanation",
    description="Generate simple explanations of complex concepts",
    metadata={"author": "AI Team", "version": "1.0"}
)

# Save to repository
repo = get_prompt_repository()
await repo.save_prompt(template.template_id, template.to_dict())

# Use the template with variables
rendered_prompt = template.render({
    "field": "quantum physics",
    "concept": "entanglement",
    "audience": "high school student"
})

# Generate completion using the rendered prompt
result = await client.tools.generate_completion(
    prompt=rendered_prompt,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)
```

### Error Handling and Resilience

Ultimate MCP Server implements sophisticated error handling to ensure reliable operations even in challenging conditions:

- **Intelligent Retry Logic**:
  - Automatically retry failed provider requests
  - Implement exponential backoff for rate limits
  - Distinguish between retryable and non-retryable errors
  - Gracefully handle provider outages

- **Fallback Mechanisms**:
  - Switch to alternate providers when primary fails
  - Degrade gracefully when resources are unavailable
  - Implement alternative paths for critical operations
  - Cache responses to serve during downtime

- **Detailed Error Reporting**:
  - Capture comprehensive error information
  - Categorize errors for analysis
  - Track error patterns over time
  - Generate actionable error reports

- **Validation and Prevention**:
  - Validate inputs before sending to providers
  - Check token limits and request constraints
  - Ensure secure operations in filesystem tools
  - Prevent common misuse patterns

```python
# Error handling configuration
error_handling_config = {
    "max_retries": 3,
    "backoff_factor": 1.5,
    "retry_on": ["rate_limit", "server_error", "connection_error"],
    "fallback_providers": ["openai", "anthropic", "gemini"],
    "cache_on_failure": True
}

# Generate completion with robust error handling
try:
    result = await client.tools.generate_completion_with_fallback(
        prompt="Explain quantum computing to a high school student",
        provider="anthropic", # Primary provider
        model="claude-3-5-sonnet-20241022",
        error_handling=error_handling_config
    )
except Exception as e:
    # Handle ultimate failure after all retries and fallbacks
    print(f"Operation failed after exhausting all options: {e}")
```


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

The Ultimate MCP Server includes powerful OCR (Optical Character Recognition) tools that leverage LLMs to improve text extraction from PDFs and images:

- **Extract Text from PDF**: Extract and enhance text from PDF documents using direct extraction or OCR.
- **Process Image OCR**: Extract and enhance text from images with preprocessing options.
- **Enhance OCR Text**: Improve existing OCR text using LLM-based correction and formatting.
- **Analyze PDF Structure**: Get information about a PDF's structure without full text extraction.
- **Batch Process Documents**: Process multiple documents with concurrent execution.

### OCR Installation

OCR tools require additional dependencies. Install them with:

```bash
pip install 'ultimate-mcp-server[ocr]'
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

### Claude Using Ultimate MCP Server for Document Analysis

This example shows how Claude can use the Ultimate MCP Server to process a document by delegating tasks to cheaper models:

```python
import asyncio
from mcp.client import Client

async def main():
    # Claude would use this client to connect to the Ultimate MCP Server
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

## Autonomous Documentation Refiner

The Ultimate MCP Server includes a powerful feature for autonomously analyzing, testing, and refining the documentation of registered MCP tools. This feature, implemented in `ultimate/tools/docstring_refiner.py`, helps improve the usability and reliability of tools when invoked by Large Language Models (LLMs) like Claude.

### How It Works

The documentation refiner follows a methodical, iterative approach:

1. **Agent Simulation**: Simulates how an LLM agent would interpret the current documentation to identify potential ambiguities or missing information
2. **Adaptive Test Generation**: Creates diverse test cases based on the tool's schema, simulation results, and failures from previous iterations
3. **Schema-Aware Testing**: Validates generated test cases against the schema before execution, then executes valid tests against the actual tools
4. **Ensemble Failure Analysis**: Uses multiple LLMs to analyze failures in the context of the documentation used for that test run
5. **Structured Improvement Proposals**: Generates specific improvements to the description, schema (as JSON Patch operations), and usage examples
6. **Validated Schema Patching**: Applies and validates proposed schema patches in-memory
7. **Iterative Refinement**: Repeats the cycle until tests consistently pass or a maximum iteration count is reached
8. **Optional Winnowing**: Performs a final pass to streamline documentation while preserving critical information

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

### Usage Example

```python
from ultimate_mcp_server.tools.docstring_refiner import refine_tool_documentation

# Refine specific tools
result = await refine_tool_documentation(
    tool_names=["search_tool", "data_processor"],
    max_iterations=3,
    refinement_model_config={"model": "gpt-4", "provider": "openai"},
    enable_winnowing=True,
    ctx=mcp_context
)

# Or refine all available tools
result = await refine_tool_documentation(
    refine_all_available=True,
    max_iterations=2,
    ctx=mcp_context
)
```


## Getting Started

### Installation

```bash
# Install uv if you don't already have it:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/ultimate_mcp_server.git
cd ultimate_mcp_server

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
python -m ultimate.cli.main run

# Start the server with specific tools only
python -m ultimate.cli.main run --include-tools generate_completion read_file write_file

# Start the server excluding specific tools
python -m ultimate.cli.main run --exclude-tools browser_automation

# Or with Docker
docker compose up
```

Once running, the server will be available at `http://localhost:8013`.

## CLI Commands

Ultimate MCP Server comes with a command-line interface for server management and tool interaction:

```bash
# Show available commands
ultimate-mcp-server --help

# Start the server
ultimate-mcp-server run [options]

# List available providers
ultimate-mcp-server providers

# List available tools
ultimate-mcp-server tools [--category CATEGORY]

# Test a provider
ultimate-mcp-server test openai --model gpt-4.1-mini

# Generate a completion
ultimate-mcp-server complete --provider anthropic --prompt "Hello, world!"

# Check cache status
ultimate-mcp-server cache --status
```

Each command has additional options that can be viewed with `ultimate-mcp-server COMMAND --help`.

## Advanced Configuration

While the `.env` file is convenient for basic setup, the Ultimate MCP Server offers more detailed configuration options primarily managed through environment variables.

### Server Configuration

- `SERVER_HOST`: (Default: `127.0.0.1`) The network interface the server listens on. Use `0.0.0.0` to listen on all interfaces (necessary for Docker or external access).
- `SERVER_PORT`: (Default: `8013`) The port the server listens on.
- `API_PREFIX`: (Default: `/`) The URL prefix for the API endpoints.

### Tool Filtering

The Ultimate MCP Server allows selectively choosing which MCP tools to register, helping manage complexity or reduce resource usage:

```bash
# List all available tools
ultimate-mcp-server tools

# List tools in a specific category
ultimate-mcp-server tools --category filesystem

# Start the server with only specific tools
ultimate-mcp-server run --include-tools read_file write_file generate_completion

# Start the server excluding specific tools
ultimate-mcp-server run --exclude-tools browser_automation marqo_fused_search
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

- **`systemd` (Linux):** Create a service unit file (e.g., `/etc/systemd/system/ultimate-mcp-server.service`) to manage the process. This allows commands like `sudo systemctl start|stop|restart|status ultimate-mcp-server`.
- **`supervisor`:** A popular process control system written in Python. Configure `supervisord` to monitor and control the gateway process.
- **Docker Restart Policies:** If using Docker (standalone or Compose), configure appropriate restart policies (e.g., `unless-stopped` or `always`) in your `docker run` command or `docker-compose.yml` file.

### 2. Using a Reverse Proxy (Nginx/Caddy/Apache)

Placing a reverse proxy in front of the Ultimate MCP Server is highly recommended:

- **HTTPS/SSL Termination:** The proxy can handle SSL certificates (e.g., using Let's Encrypt with Caddy or Certbot with Nginx/Apache), encrypting traffic between clients and the proxy.
- **Load Balancing:** If you need to run multiple instances of the gateway for high availability or performance, the proxy can distribute traffic among them.
- **Path Routing:** Map external paths (e.g., `https://api.yourdomain.com/ultimate-mcp-server/`) to the internal gateway server (`http://localhost:8013`).
- **Security Headers:** Add important security headers (like CSP, HSTS).
- **Buffering/Caching:** Some proxies offer additional request/response buffering or caching capabilities.

*Example Nginx `location` block (simplified):*
```nginx
location /ultimate-mcp-server/ {
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

Using Ultimate MCP Server for delegation can yield significant cost savings:

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

Ultimate MCP Server represents an important evolution in AI application architecture:
- Moving from monolithic AI calls to distributed, multi-model workflows
- Enabling AI-driven orchestration of complex processing pipelines
- Creating a foundation for AI systems that can reason about their own resource usage
- Building toward self-optimizing AI systems that make intelligent delegation decisions

### The Future of AI Efficiency

Ultimate MCP Server points toward a future where:
- AI systems actively manage and optimize their own resource usage
- Higher-capability models serve as intelligent orchestrators for entire AI ecosystems
- AI workflows become increasingly sophisticated and self-organizing
- Organizations can leverage the full spectrum of AI capabilities in cost-effective ways

This vision of efficient, self-organizing AI systems represents the next frontier in practical AI deployment, moving beyond the current pattern of using single models for every task.

## Architecture

### How MCP Integration Works

The Ultimate MCP Server is built natively on the Model Context Protocol:

1. **MCP Server Core**: The gateway implements a full MCP server
2. **Tool Registration**: All capabilities are exposed as MCP tools
3. **Tool Invocation**: Claude and other AI agents can directly invoke these tools
4. **Context Passing**: Results are returned in MCP's standard format

This ensures seamless integration with Claude and other MCP-compatible agents.

### Component Diagram

```plaintext
┌─────────────┐         ┌───────────────────┐         ┌──────────────┐
│  Claude 3.7 │ ────────► Ultimate MCP Server MCP   │ ────────► LLM Providers│
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

When Claude delegates a task to Ultimate MCP Server:

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

This section provides examples of how an MCP client (like Claude 3.7) would invoke specific tools provided by the Ultimate MCP Server. These examples assume you have an initialized `mcp.client.Client` instance named `client` connected to the gateway.

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

Claude or other advanced AI agents can use Ultimate MCP Server to:

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

Research teams can use Ultimate MCP Server to:

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

When deploying and operating the Ultimate MCP Server, consider the following security aspects:

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