"""Entity relationship graph tools for LLM Gateway."""
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import networkx as nx
    from pyvis.network import Network
    HAS_VISUALIZATION_LIBS = True
except ImportError:
    HAS_VISUALIZATION_LIBS = False

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.exceptions import ProviderError, ToolError, ToolInputError
from llm_gateway.tools.base import with_cache, with_error_handling, with_retry, with_tool_metrics
from llm_gateway.tools.completion import generate_completion
from llm_gateway.tools.document import chunk_document
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.entity_graph")

class GraphStrategy(Enum):
    """Strategies for entity graph extraction."""
    STANDARD = "standard"        # Basic prompt-based extraction
    MULTISTAGE = "multistage"    # Process in stages: entities first, then relationships
    CHUNKED = "chunked"          # Process large texts in chunks and merge results
    INCREMENTAL = "incremental"  # Build graph incrementally from existing graph
    STRUCTURED = "structured"    # Use structured examples for consistent extraction
    STRICT_SCHEMA = "strict_schema"  # Use a predefined schema of entities and relationships

class OutputFormat(Enum):
    """Output formats for entity graphs."""
    JSON = "json"                # Standard JSON
    NETWORKX = "networkx"        # NetworkX graph object
    RDF = "rdf"                  # Resource Description Framework
    CYTOSCAPE = "cytoscape"      # Cytoscape.js format
    D3 = "d3"                    # D3.js force graph format
    NEO4J = "neo4j"              # Neo4j Cypher queries

class VisualizationFormat(Enum):
    """Visualization formats for entity graphs."""
    NONE = "none"                # No visualization
    HTML = "html"                # Interactive HTML (Pyvis)
    SVG = "svg"                  # Static SVG
    PNG = "png"                  # Static PNG
    DOT = "dot"                  # GraphViz DOT format

# Global schemas for common domains
COMMON_SCHEMAS = {
    "business": {
        "entities": [
            {"type": "Person", "attributes": ["name", "title", "role"]},
            {"type": "Organization", "attributes": ["name", "industry", "location"]},
            {"type": "Product", "attributes": ["name", "category", "price"]},
            {"type": "Location", "attributes": ["name", "address", "type"]},
            {"type": "Event", "attributes": ["name", "date", "location"]},
        ],
        "relationships": [
            {"type": "WORKS_FOR", "source_types": ["Person"], "target_types": ["Organization"]},
            {"type": "PRODUCES", "source_types": ["Organization"], "target_types": ["Product"]},
            {"type": "COMPETES_WITH", "source_types": ["Organization", "Product"], "target_types": ["Organization", "Product"]},
            {"type": "LOCATED_IN", "source_types": ["Organization", "Person"], "target_types": ["Location"]},
            {"type": "FOUNDED", "source_types": ["Person"], "target_types": ["Organization"]},
            {"type": "ACQUIRED", "source_types": ["Organization"], "target_types": ["Organization"]},
            {"type": "SUPPLIES", "source_types": ["Organization"], "target_types": ["Organization"]},
            {"type": "PARTNERS_WITH", "source_types": ["Organization"], "target_types": ["Organization"]},
            {"type": "INVESTS_IN", "source_types": ["Organization", "Person"], "target_types": ["Organization"]},
            {"type": "ATTENDS", "source_types": ["Person"], "target_types": ["Event"]},
            {"type": "HOSTS", "source_types": ["Organization"], "target_types": ["Event"]},
        ],
    },
    "academic": {
        "entities": [
            {"type": "Researcher", "attributes": ["name", "affiliation", "field"]},
            {"type": "Institution", "attributes": ["name", "type", "location"]},
            {"type": "Publication", "attributes": ["title", "date", "journal", "impact_factor"]},
            {"type": "Concept", "attributes": ["name", "field", "definition"]},
            {"type": "Dataset", "attributes": ["name", "size", "source"]},
            {"type": "Research_Project", "attributes": ["name", "duration", "funding"]},
        ],
        "relationships": [
            {"type": "AFFILIATED_WITH", "source_types": ["Researcher"], "target_types": ["Institution"]},
            {"type": "AUTHORED", "source_types": ["Researcher"], "target_types": ["Publication"]},
            {"type": "CITES", "source_types": ["Publication"], "target_types": ["Publication"]},
            {"type": "INTRODUCES", "source_types": ["Publication"], "target_types": ["Concept"]},
            {"type": "COLLABORATES_WITH", "source_types": ["Researcher"], "target_types": ["Researcher"]},
            {"type": "USES", "source_types": ["Publication", "Researcher"], "target_types": ["Dataset", "Concept"]},
            {"type": "BUILDS_ON", "source_types": ["Concept", "Publication"], "target_types": ["Concept"]},
            {"type": "FUNDS", "source_types": ["Institution"], "target_types": ["Research_Project"]},
            {"type": "WORKS_ON", "source_types": ["Researcher"], "target_types": ["Research_Project"]},
        ],
    },
    "medical": {
        "entities": [
            {"type": "Patient", "attributes": ["id", "age", "gender"]},
            {"type": "Physician", "attributes": ["name", "specialty", "affiliation"]},
            {"type": "Condition", "attributes": ["name", "icd_code", "severity"]},
            {"type": "Medication", "attributes": ["name", "dosage", "manufacturer"]},
            {"type": "Procedure", "attributes": ["name", "code", "duration"]},
            {"type": "Healthcare_Facility", "attributes": ["name", "type", "location"]},
        ],
        "relationships": [
            {"type": "DIAGNOSED_WITH", "source_types": ["Patient"], "target_types": ["Condition"]},
            {"type": "TREATED_BY", "source_types": ["Patient"], "target_types": ["Physician"]},
            {"type": "PRESCRIBED", "source_types": ["Physician"], "target_types": ["Medication"]},
            {"type": "TAKES", "source_types": ["Patient"], "target_types": ["Medication"]},
            {"type": "TREATS", "source_types": ["Medication", "Procedure"], "target_types": ["Condition"]},
            {"type": "PERFORMED", "source_types": ["Physician"], "target_types": ["Procedure"]},
            {"type": "UNDERWENT", "source_types": ["Patient"], "target_types": ["Procedure"]},
            {"type": "WORKS_AT", "source_types": ["Physician"], "target_types": ["Healthcare_Facility"]},
            {"type": "ADMITTED_TO", "source_types": ["Patient"], "target_types": ["Healthcare_Facility"]},
            {"type": "INTERACTS_WITH", "source_types": ["Medication"], "target_types": ["Medication"]},
            {"type": "CONTRAINDICATES", "source_types": ["Condition"], "target_types": ["Medication"]},
        ],
    },
    "legal": {
        "entities": [
            {"type": "Person", "attributes": ["name", "role", "jurisdiction"]},
            {"type": "Legal_Entity", "attributes": ["name", "type", "jurisdiction"]},
            {"type": "Document", "attributes": ["name", "type", "date", "status"]},
            {"type": "Obligation", "attributes": ["description", "deadline", "status"]},
            {"type": "Claim", "attributes": ["description", "value", "status"]},
            {"type": "Asset", "attributes": ["description", "value", "type"]},
            {"type": "Court", "attributes": ["name", "jurisdiction", "type"]},
            {"type": "Law", "attributes": ["name", "jurisdiction", "date"]},
        ],
        "relationships": [
            {"type": "PARTY_TO", "source_types": ["Person", "Legal_Entity"], "target_types": ["Document"]},
            {"type": "HAS_OBLIGATION", "source_types": ["Person", "Legal_Entity"], "target_types": ["Obligation"]},
            {"type": "OWNS", "source_types": ["Person", "Legal_Entity"], "target_types": ["Asset"]},
            {"type": "CLAIMS", "source_types": ["Person", "Legal_Entity"], "target_types": ["Claim"]},
            {"type": "REPRESENTED_BY", "source_types": ["Person", "Legal_Entity"], "target_types": ["Person"]},
            {"type": "REFERENCED_IN", "source_types": ["Law", "Document"], "target_types": ["Document"]},
            {"type": "ADJUDICATED_BY", "source_types": ["Claim", "Document"], "target_types": ["Court"]},
            {"type": "REGULATES", "source_types": ["Law"], "target_types": ["Legal_Entity", "Obligation"]},
            {"type": "TRANSFERS", "source_types": ["Document"], "target_types": ["Asset"]},
            {"type": "AUTHORIZES", "source_types": ["Document"], "target_types": ["Person", "Legal_Entity"]},
        ],
    },
}

# Entity relationship detection prompts
SYSTEM_PROMPTS = {
    "entity_detection": """You are an expert entity extraction system. Your task is to identify and extract named entities from the input text with high precision. Follow these guidelines:

1. Focus on identifying complete entity mentions
2. Classify entities into appropriate types
3. Merge mentions of the same entity
4. Include precise position information when possible
5. Only extract entities actually mentioned in the text
6. Do not hallucinate entities that aren't clearly present

Output should be in valid JSON format with entities grouped by type.""",

    "relationship_detection": """You are an expert relationship extraction system. Your task is to identify meaningful connections between entities in the text. Follow these guidelines:

1. Only identify relationships between entities that are explicitly stated or strongly implied
2. Capture the semantic relationship type accurately
3. Identify the direction of the relationship (source → target)
4. Include supporting evidence from the text
5. Assign a confidence score based on how explicitly the relationship is stated
6. Do not invent relationships not supported by the text

Output should be in valid JSON with relationship records linking entity IDs.""",

    "multilingual": """You are an expert entity and relationship extraction system with multilingual capabilities. Extract entities and relationships from text in any language. First identify the language, then apply language-specific extraction patterns to identify:

1. Named entities (people, organizations, locations, etc.)
2. The relationships between these entities
3. Evidence for each relationship from the text

Be attentive to language-specific naming patterns, grammatical structures, and relationship indicators.""",

    "temporal": """You are a specialized entity and relationship extraction system with focus on temporal information. Your goal is to extract:

1. Entities with their temporal attributes (founding dates, birth dates, etc.)
2. Relationships between entities with temporal context
3. Changes in relationships over time
4. Sequence of events
5. Duration information

For each entity and relationship, capture explicit or implicit time information as precisely as possible.""",
}

FEW_SHOT_EXAMPLES = {
    "business": {
        "text": """Apple Inc., founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976, announced its latest iPhone model yesterday at its headquarters in Cupertino, California. CEO Tim Cook showcased the device, which competes with Samsung's Galaxy series. 
        
The company has partnered with TSMC to manufacture the A15 Bionic chip that powers the new iPhone. Meanwhile, Google, led by Sundar Pichai, continues to dominate the search engine market with products that compete with Apple's offerings.""",
        "entities": [
            {"id": "ent1", "name": "Apple Inc.", "type": "Organization", "mentions": [{"text": "Apple Inc.", "pos": [0, 10]}, {"text": "Apple", "pos": [226, 231]}]},
            {"id": "ent2", "name": "Steve Jobs", "type": "Person", "mentions": [{"text": "Steve Jobs", "pos": [22, 32]}]},
            {"id": "ent3", "name": "Steve Wozniak", "type": "Person", "mentions": [{"text": "Steve Wozniak", "pos": [34, 47]}]},
            {"id": "ent4", "name": "Ronald Wayne", "type": "Person", "mentions": [{"text": "Ronald Wayne", "pos": [53, 65]}]},
            {"id": "ent5", "name": "iPhone", "type": "Product", "mentions": [{"text": "iPhone", "pos": [95, 101]}, {"text": "iPhone", "pos": [324, 330]}]},
            {"id": "ent6", "name": "Cupertino", "type": "Location", "mentions": [{"text": "Cupertino, California", "pos": [129, 149]}]},
            {"id": "ent7", "name": "Tim Cook", "type": "Person", "mentions": [{"text": "Tim Cook", "pos": [156, 164]}]},
            {"id": "ent8", "name": "Samsung", "type": "Organization", "mentions": [{"text": "Samsung", "pos": [201, 208]}]},
            {"id": "ent9", "name": "Galaxy", "type": "Product", "mentions": [{"text": "Galaxy series", "pos": [210, 223]}]},
            {"id": "ent10", "name": "TSMC", "type": "Organization", "mentions": [{"text": "TSMC", "pos": [261, 265]}]},
            {"id": "ent11", "name": "A15 Bionic", "type": "Product", "mentions": [{"text": "A15 Bionic chip", "pos": [281, 295]}]},
            {"id": "ent12", "name": "Google", "type": "Organization", "mentions": [{"text": "Google", "pos": [348, 354]}]},
            {"id": "ent13", "name": "Sundar Pichai", "type": "Person", "mentions": [{"text": "Sundar Pichai", "pos": [365, 378]}]}
        ],
        "relationships": [
            {"id": "rel1", "source": "ent2", "target": "ent1", "type": "FOUNDED", "confidence": 0.95, "evidence": "Apple Inc., founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976", "temporal": {"year": 1976}},
            {"id": "rel2", "source": "ent3", "target": "ent1", "type": "FOUNDED", "confidence": 0.95, "evidence": "Apple Inc., founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976", "temporal": {"year": 1976}},
            {"id": "rel3", "source": "ent4", "target": "ent1", "type": "FOUNDED", "confidence": 0.95, "evidence": "Apple Inc., founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976", "temporal": {"year": 1976}},
            {"id": "rel4", "source": "ent7", "target": "ent1", "type": "CEO_OF", "confidence": 0.9, "evidence": "CEO Tim Cook showcased the device"},
            {"id": "rel5", "source": "ent1", "target": "ent5", "type": "PRODUCES", "confidence": 0.9, "evidence": "Apple Inc.... announced its latest iPhone model"},
            {"id": "rel6", "source": "ent1", "target": "ent6", "type": "HEADQUARTERS_IN", "confidence": 0.8, "evidence": "its headquarters in Cupertino, California"},
            {"id": "rel7", "source": "ent5", "target": "ent9", "type": "COMPETES_WITH", "confidence": 0.8, "evidence": "which competes with Samsung's Galaxy series"},
            {"id": "rel8", "source": "ent1", "target": "ent10", "type": "PARTNERS_WITH", "confidence": 0.9, "evidence": "The company has partnered with TSMC"},
            {"id": "rel9", "source": "ent10", "target": "ent11", "type": "MANUFACTURES", "confidence": 0.9, "evidence": "TSMC to manufacture the A15 Bionic chip"},
            {"id": "rel10", "source": "ent11", "target": "ent5", "type": "COMPONENT_OF", "confidence": 0.9, "evidence": "A15 Bionic chip that powers the new iPhone"},
            {"id": "rel11", "source": "ent13", "target": "ent12", "type": "LEADS", "confidence": 0.85, "evidence": "Google, led by Sundar Pichai"},
            {"id": "rel12", "source": "ent12", "target": "ent1", "type": "COMPETES_WITH", "confidence": 0.7, "evidence": "with products that compete with Apple's offerings"}
        ]
    },
    "academic": {
        "text": """Dr. Jennifer Chen from Stanford University published a groundbreaking paper in Nature on quantum computing applications in drug discovery. Her research, funded by the National Science Foundation, built upon earlier work by Dr. Richard Feynman. 

Chen collaborated with Dr. Michael Layton at MIT, who provided the dataset used in their experiments. Their publication has been cited by researchers at IBM's Quantum Computing division led by Dr. Sarah Johnson.""",
        "entities": [
            {"id": "ent1", "name": "Jennifer Chen", "type": "Researcher", "attributes": {"affiliation": "Stanford University"}, "mentions": [{"text": "Dr. Jennifer Chen", "pos": [0, 16]}]},
            {"id": "ent2", "name": "Stanford University", "type": "Institution", "mentions": [{"text": "Stanford University", "pos": [22, 41]}]},
            {"id": "ent3", "name": "Nature", "type": "Publication", "type_specific": "Journal", "mentions": [{"text": "Nature", "pos": [78, 84]}]},
            {"id": "ent4", "name": "Quantum computing applications in drug discovery", "type": "Publication", "type_specific": "Paper", "mentions": [{"text": "paper", "pos": [71, 76]}]},
            {"id": "ent5", "name": "National Science Foundation", "type": "Institution", "type_specific": "Funding Organization", "mentions": [{"text": "National Science Foundation", "pos": [124, 152]}]},
            {"id": "ent6", "name": "Richard Feynman", "type": "Researcher", "mentions": [{"text": "Dr. Richard Feynman", "pos": [178, 196]}]},
            {"id": "ent7", "name": "Michael Layton", "type": "Researcher", "attributes": {"affiliation": "MIT"}, "mentions": [{"text": "Dr. Michael Layton", "pos": [223, 241]}]},
            {"id": "ent8", "name": "MIT", "type": "Institution", "mentions": [{"text": "MIT", "pos": [245, 248]}]},
            {"id": "ent9", "name": "Drug discovery dataset", "type": "Dataset", "mentions": [{"text": "dataset", "pos": [264, 271]}]},
            {"id": "ent10", "name": "IBM", "type": "Institution", "type_specific": "Company", "mentions": [{"text": "IBM", "pos": [334, 337]}]},
            {"id": "ent11", "name": "IBM Quantum Computing division", "type": "Institution", "type_specific": "Research Division", "mentions": [{"text": "IBM's Quantum Computing division", "pos": [334, 365]}]},
            {"id": "ent12", "name": "Sarah Johnson", "type": "Researcher", "mentions": [{"text": "Dr. Sarah Johnson", "pos": [375, 392]}]}
        ],
        "relationships": [
            {"id": "rel1", "source": "ent1", "target": "ent2", "type": "AFFILIATED_WITH", "confidence": 0.95, "evidence": "Dr. Jennifer Chen from Stanford University"},
            {"id": "rel2", "source": "ent1", "target": "ent4", "type": "AUTHORED", "confidence": 0.95, "evidence": "Dr. Jennifer Chen from Stanford University published a groundbreaking paper"},
            {"id": "rel3", "source": "ent4", "target": "ent3", "type": "PUBLISHED_IN", "confidence": 0.9, "evidence": "published a groundbreaking paper in Nature"},
            {"id": "rel4", "source": "ent5", "target": "ent4", "type": "FUNDED", "confidence": 0.85, "evidence": "Her research, funded by the National Science Foundation"},
            {"id": "rel5", "source": "ent4", "target": "ent6", "type": "BUILDS_ON", "confidence": 0.8, "evidence": "built upon earlier work by Dr. Richard Feynman"},
            {"id": "rel6", "source": "ent1", "target": "ent7", "type": "COLLABORATED_WITH", "confidence": 0.9, "evidence": "Chen collaborated with Dr. Michael Layton at MIT"},
            {"id": "rel7", "source": "ent7", "target": "ent8", "type": "AFFILIATED_WITH", "confidence": 0.9, "evidence": "Dr. Michael Layton at MIT"},
            {"id": "rel8", "source": "ent7", "target": "ent9", "type": "PROVIDED", "confidence": 0.85, "evidence": "who provided the dataset used in their experiments"},
            {"id": "rel9", "source": "ent4", "target": "ent9", "type": "USES", "confidence": 0.8, "evidence": "dataset used in their experiments"},
            {"id": "rel10", "source": "ent11", "target": "ent4", "type": "CITES", "confidence": 0.85, "evidence": "Their publication has been cited by researchers at IBM's Quantum Computing division"},
            {"id": "rel11", "source": "ent12", "target": "ent11", "type": "LEADS", "confidence": 0.9, "evidence": "IBM's Quantum Computing division led by Dr. Sarah Johnson"}
        ]
    }
}

@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1.0)
@with_cache(ttl=24 * 60 * 60)  # Cache results for 24 hours by default
@with_error_handling
async def extract_entity_graph(
    text: str,
    entity_types: Optional[List[str]] = None,
    relation_types: Optional[List[str]] = None,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    include_evidence: bool = True,
    include_attributes: bool = True,
    include_positions: bool = True, 
    include_temporal_info: bool = True,
    max_entities: int = 100,
    max_relations: int = 200,
    min_confidence: float = 0.6,
    domain: Optional[str] = None,  # e.g., "business", "academic", "medical", "legal"
    output_format: Union[str, OutputFormat] = OutputFormat.JSON,
    visualization_format: Union[str, VisualizationFormat] = VisualizationFormat.HTML,
    strategy: Union[str, GraphStrategy] = GraphStrategy.STANDARD,
    example_entities: Optional[List[Dict[str, Any]]] = None,
    example_relationships: Optional[List[Dict[str, Any]]] = None,
    custom_entity_schema: Optional[Dict[str, Any]] = None,
    custom_relationship_schema: Optional[Dict[str, Any]] = None,
    existing_graph: Optional[Dict[str, Any]] = None,
    context_window: Optional[int] = None,
    language: Optional[str] = None, 
    automatic_coreference: bool = True,
    chunk_size: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    normalize_entities: bool = True,
    sort_by: str = "confidence",  # Options: "confidence", "centrality", "mentions"
    max_tokens_per_request: Optional[int] = None,
    enable_reasoning: bool = False,
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extracts entities and their relationships from text, building a comprehensive knowledge graph.
    
    This tool analyzes unstructured text to identify entities and the semantic relationships 
    between them, creating a structured knowledge graph representation. It supports multiple 
    strategies for extraction, including chunking for large documents, multi-stage processing,
    predefined schemas, and incremental graph building.
    
    Args:
        text: The input text to analyze.
        entity_types: Optional list of entity types to focus on (e.g., ["Person", "Organization", 
                    "Location", "Date", "Product", "Event"]). If None, extracts all entity types.
        relation_types: Optional list of relationship types to extract (e.g., ["works_for", 
                      "located_in", "founder_of"]). If None, extracts all relationship types.
        provider: The LLM provider (e.g., "openai", "anthropic", "gemini"). Defaults to "openai".
        model: The specific model ID. If None, the provider's default model is used.
        include_evidence: Whether to include text snippets supporting each relationship. Default True.
        include_attributes: Whether to extract and include entity attributes. Default True.
        include_positions: Whether to include position information for entity mentions. Default True.
        include_temporal_info: Whether to extract temporal context for relationships. Default True.
        max_entities: Maximum number of entities to extract. Default 100.
        max_relations: Maximum number of relations to extract. Default 200.
        min_confidence: Minimum confidence score (0.0-1.0) for relationships. Default 0.6.
        domain: Optional domain to use specialized extraction schemas (e.g., "business", "academic").
               Uses predefined schemas for common entity and relationship types in that domain.
        output_format: Desired output format. Options: "json", "networkx", "rdf", "cytoscape", etc.
                      Default "json".
        visualization_format: Format for visualization output. Options: "none", "html", "svg", etc.
                            Default "html".
        strategy: Extraction strategy to use. Options:
                  - "standard": Basic extraction in one step.
                  - "multistage": Extract entities first, then relationships.
                  - "chunked": Split large text into chunks, process separately, then merge.
                  - "incremental": Add to an existing graph.
                  - "structured": Use structured examples for consistent extraction.
                  - "strict_schema": Use a predefined schema of entities and relationships.
                  Default "standard".
        example_entities: Optional list of example entities to guide extraction format.
        example_relationships: Optional list of example relationships to guide extraction format.
        custom_entity_schema: Optional custom schema for entity types and attributes.
        custom_relationship_schema: Optional custom schema for relationship types.
        existing_graph: Optional existing graph data to augment (for incremental strategy).
        context_window: Optional maximum context window size (in tokens) for the model.
                       If None, estimated from model. Used for chunking strategy.
        language: Optional language specification for multilingual extraction.
        automatic_coreference: Whether to attempt automatic coreference resolution. Default True.
        chunk_size: Optional custom chunk size for chunked processing strategy.
        custom_prompt: Optional custom prompt for extraction. Use placeholders {text}, 
                     {entity_instructions}, and {relationship_instructions} if specified.
        system_prompt: Optional custom system prompt to override defaults.
        normalize_entities: Whether to normalize entity names (capitalize, remove duplicates). 
                          Default True.
        sort_by: How to sort entities and relationships in output. Options: "confidence", 
               "centrality", "mentions". Default "confidence".
        max_tokens_per_request: Optional maximum token limit for each request to the LLM.
        enable_reasoning: Whether to enable step-by-step reasoning for complex extractions. Default False.
        additional_params: Additional provider-specific parameters.
        
    Returns:
        A dictionary containing the entity graph data and metadata:
        {
            "entities": [
                {
                    "id": "ent1",
                    "name": "Tim Cook",
                    "type": "Person",
                    "mentions": [{"text": "Tim Cook", "pos": [10, 18]}, ...],
                    "attributes": {"role": "CEO", "company": "Apple"}, # If include_attributes is True
                    "centrality": 0.75 # Added during post-processing
                },
                ...
            ],
            "relationships": [
                {
                    "id": "rel1",
                    "source": "ent1",
                    "target": "ent2", 
                    "type": "CEO_of",
                    "confidence": 0.95,
                    "evidence": "Tim Cook, the CEO of Apple, announced...",
                    "temporal": {"start": "2011-08-24"} # If include_temporal_info is True
                },
                ...
            ],
            "metadata": {
                "entity_count": 10,
                "relationship_count": 15,
                "entity_types": ["Person", "Organization", ...],
                "relation_types": ["works_for", "located_in", ...],
                "processing_strategy": "multistage",
                "extraction_date": "2025-04-15T15:30:45Z",
                "stats": {
                    "density": 0.23,
                    "average_degree": 2.5,
                    "diameter": 4
                }
            },
            "visualization": {
                "html": "...", # If visualization_format is "html"
                "url": "file://..." # Local path to visualization file
            },
            "query_interface": { # Utility functions for graph queries
                "find_path": "Function to find paths between entities",
                "find_entity": "Function to search for entities",
                "get_subgraph": "Function to extract subgraphs"
            },
            "model": "model-used",
            "provider": "provider-name",
            "tokens": {
                "input": 350,
                "output": 180,
                "total": 530
            },
            "cost": 0.000412,
            "processing_time": 2.34,
            "success": true
        }
        
    Raises:
        ToolInputError: If the text is empty or parameters are invalid.
        ProviderError: If the provider is unavailable or extraction fails.
        ToolError: For parsing or other processing errors.
    """
    start_time = time.time()
    
    # --- Input Validation ---
    if not text or not isinstance(text, str):
        raise ToolInputError("Text must be a non-empty string.")
    
    # Validate enum parameters
    if isinstance(output_format, str):
        try:
            output_format = OutputFormat(output_format.lower())
        except ValueError as e:
            valid_formats = [f.value for f in OutputFormat]
            raise ToolInputError(
                f"Invalid output_format: '{output_format}'. Valid options: {valid_formats}"
            ) from e
    
    if isinstance(visualization_format, str):
        try:
            visualization_format = VisualizationFormat(visualization_format.lower())
        except ValueError as e:
            valid_formats = [f.value for f in VisualizationFormat]
            raise ToolInputError(
                f"Invalid visualization_format: '{visualization_format}'. Valid options: {valid_formats}"
            ) from e
    
    if isinstance(strategy, str):
        try:
            strategy = GraphStrategy(strategy.lower())
        except ValueError as e:
            valid_strategies = [f.value for f in GraphStrategy]
            raise ToolInputError(
                f"Invalid strategy: '{strategy}'. Valid options: {valid_strategies}"
            ) from e
    
    # Validate numeric parameters
    if not isinstance(min_confidence, (int, float)) or min_confidence < 0 or min_confidence > 1:
        raise ToolInputError("min_confidence must be a number between 0 and 1.")
    
    # Validate domain if specified
    if domain and domain not in COMMON_SCHEMAS:
        valid_domains = list(COMMON_SCHEMAS.keys())
        raise ToolInputError(
            f"Invalid domain: '{domain}'. Valid options: {valid_domains}"
        )
    
    # Check for required parameters based on strategy
    if strategy == GraphStrategy.INCREMENTAL and not existing_graph:
        raise ToolInputError(
            "The 'incremental' strategy requires an existing_graph parameter."
        )
    
    # Check visualization library availability
    if visualization_format != VisualizationFormat.NONE and not HAS_VISUALIZATION_LIBS:
        logger.warning(
            "Visualization libraries (networkx, pyvis) not available. Falling back to 'none' format."
        )
        visualization_format = VisualizationFormat.NONE
    
    # --- Initialize Configuration ---
    # Get provider instance
    try:
        provider_instance = await get_provider(provider)
    except Exception as e:
        raise ProviderError(
            f"Failed to initialize provider '{provider}': {str(e)}",
            provider=provider,
            cause=e
        ) from e
    
    # Set default additional params
    additional_params = additional_params or {}
    
    # Determine context window if not specified
    if not context_window:
        # Estimate based on model
        model_context_estimates = {
            "gpt-4.1-mini": 128000,
            "gpt-4o": 128000,
            "claude-3-5-sonnet": 200000,
            "claude-3-opus": 200000,
            "gemini-2.0-pro": 128000,
            "gemini-2.5-pro": 128000,
            "gemini-2.0-flash": 128000,
        }
        
        # Try to find a match based on model name (rough estimate)
        if model:
            for model_key, window_size in model_context_estimates.items():
                if model_key.lower() in model.lower():
                    context_window = window_size
                    break
        
        # If still not determined, use a conservative default
        if not context_window:
            context_window = 16000  # Conservative default
    
    # Initialize schema based on domain if specified
    schema = None
    if domain and domain in COMMON_SCHEMAS:
        schema = COMMON_SCHEMAS[domain]
    elif custom_entity_schema and custom_relationship_schema:
        schema = {
            "entities": custom_entity_schema,
            "relationships": custom_relationship_schema
        }
    
    # --- Process Text Based on Strategy ---
    # This is where we branch based on the selected strategy
    if strategy == GraphStrategy.STANDARD:
        extraction_result = await _perform_standard_extraction(
            text=text,
            provider_instance=provider_instance,
            model=model,
            entity_types=entity_types,
            relation_types=relation_types,
            include_evidence=include_evidence,
            include_attributes=include_attributes,
            include_positions=include_positions,
            include_temporal_info=include_temporal_info,
            min_confidence=min_confidence,
            max_entities=max_entities,
            max_relations=max_relations,
            schema=schema,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            language=language,
            example_entities=example_entities,
            example_relationships=example_relationships,
            enable_reasoning=enable_reasoning,
            additional_params=additional_params
        )
    
    elif strategy == GraphStrategy.MULTISTAGE:
        extraction_result = await _perform_multistage_extraction(
            text=text,
            provider_instance=provider_instance,
            model=model,
            entity_types=entity_types,
            relation_types=relation_types,
            include_evidence=include_evidence,
            include_attributes=include_attributes,
            include_positions=include_positions,
            include_temporal_info=include_temporal_info,
            min_confidence=min_confidence,
            max_entities=max_entities,
            max_relations=max_relations,
            schema=schema,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            language=language,
            automatic_coreference=automatic_coreference,
            example_entities=example_entities,
            example_relationships=example_relationships,
            enable_reasoning=enable_reasoning,
            additional_params=additional_params
        )
    
    elif strategy == GraphStrategy.CHUNKED:
        extraction_result = await _perform_chunked_extraction(
            text=text,
            provider_instance=provider_instance,
            model=model,
            entity_types=entity_types,
            relation_types=relation_types,
            include_evidence=include_evidence,
            include_attributes=include_attributes,
            include_positions=include_positions,
            include_temporal_info=include_temporal_info,
            min_confidence=min_confidence,
            max_entities=max_entities, 
            max_relations=max_relations,
            schema=schema,
            context_window=context_window,
            chunk_size=chunk_size,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            language=language,
            example_entities=example_entities,
            example_relationships=example_relationships,
            enable_reasoning=enable_reasoning,
            additional_params=additional_params
        )
    
    elif strategy == GraphStrategy.INCREMENTAL:
        extraction_result = await _perform_incremental_extraction(
            text=text,
            existing_graph=existing_graph,
            provider_instance=provider_instance,
            model=model,
            entity_types=entity_types,
            relation_types=relation_types,
            include_evidence=include_evidence,
            include_attributes=include_attributes,
            include_positions=include_positions,
            include_temporal_info=include_temporal_info,
            min_confidence=min_confidence,
            max_entities=max_entities,
            max_relations=max_relations,
            schema=schema,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            language=language,
            additional_params=additional_params
        )
    
    elif strategy == GraphStrategy.STRUCTURED:
        # Use example-based extraction with strict formatting
        if not example_entities and not example_relationships and domain:
            # Use domain examples if available
            if domain in FEW_SHOT_EXAMPLES:
                example_entities = FEW_SHOT_EXAMPLES[domain]["entities"]
                example_relationships = FEW_SHOT_EXAMPLES[domain]["relationships"]
        
        extraction_result = await _perform_structured_extraction(
            text=text,
            provider_instance=provider_instance,
            model=model,
            entity_types=entity_types,
            relation_types=relation_types,
            include_evidence=include_evidence,
            include_attributes=include_attributes,
            include_positions=include_positions,
            include_temporal_info=include_temporal_info,
            min_confidence=min_confidence,
            max_entities=max_entities,
            max_relations=max_relations,
            schema=schema,
            example_entities=example_entities,
            example_relationships=example_relationships,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            language=language,
            additional_params=additional_params
        )
    
    elif strategy == GraphStrategy.STRICT_SCHEMA:
        # Schema-guided extraction
        if not schema:
            raise ToolInputError(
                "The 'strict_schema' strategy requires either a domain or custom schema definitions."
            )
        
        extraction_result = await _perform_schema_guided_extraction(
            text=text,
            provider_instance=provider_instance,
            model=model,
            schema=schema,
            include_evidence=include_evidence,
            include_attributes=include_attributes,
            include_positions=include_positions,
            include_temporal_info=include_temporal_info,
            min_confidence=min_confidence,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            language=language,
            additional_params=additional_params
        )
    
    # --- Post-processing ---
    # Normalize entity names if requested
    if normalize_entities:
        extraction_result = _normalize_entities(extraction_result)
    
    # Add computed graph metrics
    extraction_result = _add_graph_metrics(extraction_result, sort_by)
    
    # Generate visualization if requested
    visualization_data = None
    if visualization_format != VisualizationFormat.NONE:
        visualization_data = _generate_visualization(
            extraction_result, 
            visualization_format
        )
    
    # Create query interface
    query_interface = _create_query_interface(extraction_result)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Format output according to requested format
    formatted_result = _format_output(extraction_result, output_format)
    
    # Prepare final result
    final_result = {
        "entities": formatted_result["entities"],
        "relationships": formatted_result["relationships"],
        "metadata": {
            **formatted_result.get("metadata", {}),
            "entity_count": len(formatted_result["entities"]),
            "relationship_count": len(formatted_result["relationships"]),
            "entity_types": sorted(list(set(e.get("type") for e in formatted_result["entities"] if "type" in e))),
            "relation_types": sorted(list(set(r.get("type") for r in formatted_result["relationships"] if "type" in r))),
            "processing_strategy": strategy.value,
            "extraction_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "provider": provider,
        "model": extraction_result["model"],
        "tokens": extraction_result["tokens"],
        "cost": extraction_result["cost"],
        "processing_time": processing_time,
        "success": True
    }
    
    # Add visualization if generated
    if visualization_data:
        final_result["visualization"] = visualization_data
    
    # Add query interface
    if query_interface:
        final_result["query_interface"] = query_interface
    
    # Log success
    logger.success(
        f"Entity graph extraction completed successfully with {strategy.value} strategy using {provider}/{extraction_result['model']}",
        emoji_key="graph",
        entities=final_result["metadata"]["entity_count"],
        relationships=final_result["metadata"]["relationship_count"],
        cost=extraction_result["cost"],
        time=processing_time
    )
    
    return final_result

# --- Strategy Implementation Functions ---

def _validate_graph_data(
    graph_data: Dict[str, Any], 
    min_confidence: float,
    max_entities: int,
    max_relations: int
) -> Dict[str, Any]:
    """Validates and cleans up extracted graph data.
    
    Args:
        graph_data: The raw graph data from extraction
        min_confidence: Minimum confidence threshold for relationships
        max_entities: Maximum number of entities to keep
        max_relations: Maximum number of relationships to keep
        
    Returns:
        Cleaned and validated graph data
    """
    # Ensure required keys exist
    if "entities" not in graph_data:
        graph_data["entities"] = []
    if "relationships" not in graph_data:
        graph_data["relationships"] = []
    
    # Validate entities
    valid_entities = []
    entity_ids = set()
    
    for i, entity in enumerate(graph_data["entities"]):
        # Skip if missing required fields
        if not isinstance(entity, dict) or "name" not in entity or "type" not in entity:
            continue
        
        # Generate ID if missing
        if "id" not in entity:
            entity["id"] = f"ent{i+1}"
        
        # Ensure unique IDs
        original_id = entity["id"]
        if original_id in entity_ids:
            counter = 1
            while f"{original_id}_{counter}" in entity_ids:
                counter += 1
            entity["id"] = f"{original_id}_{counter}"
        
        entity_ids.add(entity["id"])
        
        # Ensure mentions is a valid list if included
        if "mentions" in entity:
            if not isinstance(entity["mentions"], list):
                entity["mentions"] = []
            else:
                # Validate each mention
                valid_mentions = []
                for mention in entity["mentions"]:
                    if isinstance(mention, dict) and "text" in mention:
                        # Ensure position is valid if included
                        if "pos" in mention:
                            pos = mention["pos"]
                            if (isinstance(pos, list) and len(pos) == 2 and 
                                isinstance(pos[0], (int, float)) and isinstance(pos[1], (int, float))):
                                valid_mentions.append(mention)
                            else:
                                # Fix invalid position
                                valid_mentions.append({"text": mention["text"]})
                        else:
                            valid_mentions.append(mention)
                
                entity["mentions"] = valid_mentions
        
        # Ensure attributes is a valid dictionary if included
        if "attributes" in entity:
            if not isinstance(entity["attributes"], dict):
                entity["attributes"] = {}
            else:
                # Clean up attribute values (ensure they're strings, numbers, or booleans)
                cleaned_attributes = {}
                for key, value in entity["attributes"].items():
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_attributes[key] = value
                    elif value is None:
                        continue  # Skip None values
                    else:
                        # Convert complex values to strings
                        try:
                            cleaned_attributes[key] = str(value)
                        except (TypeError, ValueError):
                            pass  # Skip if conversion fails
                
                entity["attributes"] = cleaned_attributes
        
        valid_entities.append(entity)
    
    # Limit entities if needed
    if len(valid_entities) > max_entities:
        valid_entities = valid_entities[:max_entities]
    
    # After limiting entities, get final set of valid IDs
    valid_entity_ids = {e["id"] for e in valid_entities}
    
    # Validate relationships
    valid_relationships = []
    relationship_signatures = set()  # Track (source, target, type) to avoid duplicates
    
    for i, relation in enumerate(graph_data["relationships"]):
        # Skip if missing required fields
        if not isinstance(relation, dict) or "source" not in relation or "target" not in relation:
            continue
        
        # Generate type if missing
        if "type" not in relation:
            relation["type"] = "RELATED_TO"
        
        # Generate ID if missing
        if "id" not in relation:
            relation["id"] = f"rel{i+1}"
        
        # Skip if referencing invalid entities
        if relation["source"] not in valid_entity_ids or relation["target"] not in valid_entity_ids:
            continue
        
        # Check confidence
        confidence = relation.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5  # Default confidence
            relation["confidence"] = confidence
        
        # Skip if below threshold
        if confidence < min_confidence:
            continue
        
        # Check for duplicate relationship (same source, target, and type)
        signature = (relation["source"], relation["target"], relation["type"])
        if signature in relationship_signatures:
            continue
        
        relationship_signatures.add(signature)
        
        # Validate evidence if included
        if "evidence" in relation and not isinstance(relation["evidence"], str):
            try:
                relation["evidence"] = str(relation["evidence"])
            except Exception:
                relation.pop("evidence")  # Remove invalid evidence
        
        # Validate temporal info if included
        if "temporal" in relation:
            if not isinstance(relation["temporal"], dict):
                relation.pop("temporal")
            else:
                # Ensure temporal values are strings or numbers
                cleaned_temporal = {}
                for key, value in relation["temporal"].items():
                    if isinstance(value, (str, int, float)):
                        cleaned_temporal[key] = value
                
                if cleaned_temporal:
                    relation["temporal"] = cleaned_temporal
                else:
                    relation.pop("temporal")
        
        valid_relationships.append(relation)
    
    # Limit relationships if needed
    if len(valid_relationships) > max_relations:
        # Sort by confidence first
        valid_relationships.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        valid_relationships = valid_relationships[:max_relations]
    
    # Return validated data
    return {
        "entities": valid_entities,
        "relationships": valid_relationships
    }

async def _perform_standard_extraction(
    text: str,
    provider_instance: Any,
    model: Optional[str],
    entity_types: Optional[List[str]],
    relation_types: Optional[List[str]],
    include_evidence: bool,
    include_attributes: bool,
    include_positions: bool,
    include_temporal_info: bool,
    min_confidence: float,
    max_entities: int,
    max_relations: int,
    schema: Optional[Dict[str, Any]],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    language: Optional[str],
    example_entities: Optional[List[Dict[str, Any]]],
    example_relationships: Optional[List[Dict[str, Any]]],
    enable_reasoning: bool = False,
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Performs extraction in a single step."""
    # Build extraction prompt
    entity_types_str = ""
    if entity_types:
        entity_types_str = "Entity types to extract:\n" + "\n".join([f"- {t}" for t in entity_types])
    
    relation_types_str = ""
    if relation_types:
        relation_types_str = "Relationship types to extract:\n" + "\n".join([f"- {t}" for t in relation_types])
    
    # Schema guidance if provided
    schema_guidance = ""
    if schema:
        schema_guidance = "Use the following schema for extraction:\n\n"
        
        # Entity schema
        schema_guidance += "Entity Types:\n"
        for entity_type in schema.get("entities", []):
            schema_guidance += f"- {entity_type['type']}"
            if "attributes" in entity_type:
                schema_guidance += f" (Attributes: {', '.join(entity_type['attributes'])})"
            schema_guidance += "\n"
        
        # Relationship schema
        schema_guidance += "\nRelationship Types:\n"
        for rel_type in schema.get("relationships", []):
            source_types = ", ".join(rel_type.get("source_types", ["Any"]))
            target_types = ", ".join(rel_type.get("target_types", ["Any"]))
            schema_guidance += f"- {rel_type['type']} (From: {source_types}, To: {target_types})\n"
    
    # Examples formatting
    examples_str = ""
    if example_entities and example_relationships:
        examples_str = "\n\nEXAMPLES:\n\nExample entities:"
        examples_str += json.dumps(example_entities[:3], indent=2)
        examples_str += "\n\nExample relationships:"
        examples_str += json.dumps(example_relationships[:3], indent=2)
    
    # Language specification
    language_instruction = ""
    if language:
        language_instruction = f"The text is in {language}. Extract entities and relationships accordingly."
    
    # Reasoning steps instruction
    reasoning_instruction = ""
    if enable_reasoning:
        reasoning_instruction = """
Before extracting the final entities and relationships, follow these reasoning steps:
1. Identify all potential entity mentions in the text
2. Group mentions that refer to the same entity
3. Determine the most appropriate entity type for each entity
4. Identify explicit relationship statements in the text
5. Infer implicit relationships based on context
6. Assign appropriate relationship types
7. Determine relationship directionality (source → target)
8. Assess confidence for each extraction
"""
    
    # Format instructions for the output
    format_instructions = f"""
Extract entities and their relationships from the text below.
{entity_types_str}
{relation_types_str}
{schema_guidance}
{language_instruction}
{reasoning_instruction}

Format your response as JSON with the following structure:
{{
  "entities": [
    {{
      "id": "ent1", 
      "name": "entity_name",
      "type": "entity_type",
      {"mentions": [{"text": "mention_text", "pos": [start_pos, end_pos]}]," if include_positions else ""}
      {"attributes": {"attribute_name": "value"}," if include_attributes else ""}
    }}
  ],
  "relationships": [
    {{
      "id": "rel1",
      "source": "ent1",
      "target": "ent2",
      "type": "relationship_type",
      "confidence": 0.95,
      {"evidence": "text_supporting_relationship"," if include_evidence else ""}
      {"temporal": {"start": "timestamp", "end": "timestamp"}," if include_temporal_info else ""}
    }}
  ]
}}

Only include relationships with confidence ≥ {min_confidence}.
Limit to approximately {max_entities} most important entities and {max_relations} most significant relationships.
Ensure entity IDs are consistent and correctly referenced in relationships.
{examples_str}
"""

    # Use custom prompt if provided
    if custom_prompt:
        prompt = custom_prompt.format(
            text=text,
            entity_instructions=entity_types_str,
            relationship_instructions=relation_types_str,
            format_instructions=format_instructions
        )
    else:
        prompt = f"{format_instructions}\n\nText to analyze:\n{text}"
    
    # Use custom system prompt if provided, otherwise use default
    sys_prompt = system_prompt or SYSTEM_PROMPTS.get("entity_detection", "")
    
    # Execute extraction
    try:
        # Set temperature low for deterministic extraction
        temperature = additional_params.pop("temperature", 0.1)
        
        # Generate extraction using standardized completion tool
        completion_result = await generate_completion(
            prompt=prompt,
            model=model,
            provider=provider_instance.__class__.__name__.lower(), # Extract provider name from instance
            temperature=temperature,
            max_tokens=4000,  # Allow sufficient tokens for complex graphs
            additional_params={
                "system_prompt": sys_prompt if sys_prompt else None,
                **additional_params
            }
        )
        
        # Check if completion was successful
        if not completion_result.get("success", False):
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Entity graph extraction failed: {error_message}", 
                provider=provider_instance.__class__.__name__,
                model=model or "default"
            )
        
        # Parse response
        try:
            # Extract JSON from the response
            json_match = re.search(r'(\{.*\})', completion_result["text"], re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in the response.")
            
            graph_data = json.loads(json_match.group(0))
            
            # Validate and clean up the extracted data
            graph_data = _validate_graph_data(
                graph_data, 
                min_confidence, 
                max_entities, 
                max_relations
            )
            
            # Add model metadata to result
            graph_data["model"] = completion_result["model"]
            graph_data["tokens"] = completion_result["tokens"]
            graph_data["cost"] = completion_result["cost"]
            
            return graph_data
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ToolError(
                f"Failed to parse entity graph extraction: {str(e)}",
                error_code="PARSING_ERROR",
                details={"response_text": completion_result["text"]}
            ) from e
            
    except Exception as e:
        if isinstance(e, ProviderError):
            raise # Re-raise provider errors as-is
            
        # Convert other exceptions to provider error
        error_model = model or "default"
        raise ProviderError(
            f"Entity graph extraction failed for model '{error_model}': {str(e)}",
            provider=provider_instance.__class__.__name__,
            model=error_model,
            cause=e
        ) from e

async def _perform_multistage_extraction(
    text: str,
    provider_instance: Any,
    model: Optional[str],
    entity_types: Optional[List[str]],
    relation_types: Optional[List[str]],
    include_evidence: bool,
    include_attributes: bool,
    include_positions: bool,
    include_temporal_info: bool,
    min_confidence: float,
    max_entities: int,
    max_relations: int,
    schema: Optional[Dict[str, Any]],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    language: Optional[str],
    automatic_coreference: bool,
    example_entities: Optional[List[Dict[str, Any]]],
    example_relationships: Optional[List[Dict[str, Any]]],
    enable_reasoning: bool = False,
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Performs extraction in multiple stages: entities first, then relationships."""
    # Track token usage and cost
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    # --- Stage 1: Entity Extraction ---
    entity_types_str = ""
    if entity_types:
        entity_types_str = "Entity types to extract:\n" + "\n".join([f"- {t}" for t in entity_types])
    
    # Schema guidance for entities
    schema_guidance = ""
    if schema and "entities" in schema:
        schema_guidance = "Use the following entity schema for extraction:\n\n"
        for entity_type in schema.get("entities", []):
            schema_guidance += f"- {entity_type['type']}"
            if "attributes" in entity_type:
                schema_guidance += f" (Attributes: {', '.join(entity_type['attributes'])})"
            schema_guidance += "\n"
    
    # Examples formatting for entities
    examples_str = ""
    if example_entities:
        examples_str = "\n\nEXAMPLES:\n\nExample entities:"
        examples_str += json.dumps(example_entities[:3], indent=2)
    
    # Language specification
    language_instruction = ""
    if language:
        language_instruction = f"The text is in {language}. Extract entities accordingly."
    
    # Format instructions for entity extraction
    entity_format_instructions = f"""
Extract all entities from the text below.
{entity_types_str}
{schema_guidance}
{language_instruction}

Format your response as JSON with the following structure:
{{
  "entities": [
    {{
      "id": "ent1", 
      "name": "entity_name",
      "type": "entity_type",
      {"mentions": [{"text": "mention_text", "pos": [start_pos, end_pos]}]," if include_positions else ""}
      {"attributes": {"attribute_name": "value"}," if include_attributes else ""}
    }}
  ]
}}

Limit to approximately {max_entities} most important entities.
Ensure entity IDs are unique and descriptive (e.g., "person_john_smith").
{examples_str}
"""

    # Use custom entity prompt if provided
    if custom_prompt and "{entity_instructions}" in custom_prompt:
        entity_prompt = custom_prompt.format(
            text=text,
            entity_instructions=entity_types_str,
            format_instructions=entity_format_instructions
        )
    else:
        entity_prompt = f"{entity_format_instructions}\n\nText to analyze:\n{text}"
    
    # Use custom system prompt or default entity detection prompt
    entity_sys_prompt = system_prompt or SYSTEM_PROMPTS.get("entity_detection", "")
    
    # Extract entities
    try:
        # Set temperature low for deterministic extraction
        temperature = additional_params.pop("temperature", 0.1)
        
        # Generate entity extraction using standardized completion tool
        entity_completion_result = await generate_completion(
            prompt=entity_prompt,
            model=model,
            provider=provider_instance.__class__.__name__.lower(),
            temperature=temperature,
            max_tokens=2000,
            additional_params={
                "system_prompt": entity_sys_prompt if entity_sys_prompt else None,
                **additional_params
            }
        )
        
        # Check if completion was successful
        if not entity_completion_result.get("success", False):
            error_message = entity_completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Entity extraction failed: {error_message}", 
                provider=provider_instance.__class__.__name__,
                model=model or "default"
            )
        
        # Track usage
        total_input_tokens += entity_completion_result["tokens"]["input"]
        total_output_tokens += entity_completion_result["tokens"]["output"]
        total_cost += entity_completion_result["cost"]
        
        # Parse entity response
        try:
            # Extract JSON from the response
            json_match = re.search(r'(\{.*\})', entity_completion_result["text"], re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in the entity extraction response.")
            
            entity_data = json.loads(json_match.group(0))
            
            if "entities" not in entity_data or not isinstance(entity_data["entities"], list):
                raise ValueError("Invalid entity extraction result: missing 'entities' array.")
            
            entities = entity_data["entities"]
            
            # Limit to max_entities if needed
            if len(entities) > max_entities:
                entities = entities[:max_entities]
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ToolError(
                f"Failed to parse entity extraction: {str(e)}",
                error_code="PARSING_ERROR",
                details={"response_text": entity_completion_result["text"]}
            ) from e
        
        # --- Stage 2: Relationship Extraction ---
        relation_types_str = ""
        if relation_types:
            relation_types_str = "Relationship types to extract:\n" + "\n".join([f"- {t}" for t in relation_types])
        
        # Schema guidance for relationships
        rel_schema_guidance = ""
        if schema and "relationships" in schema:
            rel_schema_guidance = "Use the following relationship schema for extraction:\n\n"
            for rel_type in schema.get("relationships", []):
                source_types = ", ".join(rel_type.get("source_types", ["Any"]))
                target_types = ", ".join(rel_type.get("target_types", ["Any"]))
                rel_schema_guidance += f"- {rel_type['type']} (From: {source_types}, To: {target_types})\n"
        
        # Examples formatting for relationships
        rel_examples_str = ""
        if example_relationships:
            rel_examples_str = "\n\nEXAMPLES:\n\nExample relationships:"
            rel_examples_str += json.dumps(example_relationships[:3], indent=2)
        
        # Format entity list for relationship extraction
        entity_list = ""
        for entity in entities:
            entity_id = entity.get("id", "")
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            entity_list += f"- ID: {entity_id}, Name: {entity_name}, Type: {entity_type}\n"
        
        # Format instructions for relationship extraction
        relationship_format_instructions = f"""
Extract relationships between the entities listed below from the text.
{relation_types_str}
{rel_schema_guidance}
{language_instruction}

ENTITIES:
{entity_list}

Format your response as JSON with the following structure:
{{
  "relationships": [
    {{
      "id": "rel1",
      "source": "ent1", # Must be an ID from the entity list above
      "target": "ent2", # Must be an ID from the entity list above
      "type": "relationship_type",
      "confidence": 0.95,
      {"evidence": "text_supporting_relationship"," if include_evidence else ""}
      {"temporal": {"start": "timestamp", "end": "timestamp"}," if include_temporal_info else ""}
    }}
  ]
}}

Only include relationships with confidence ≥ {min_confidence}.
Limit to approximately {max_relations} most significant relationships.
Only use entity IDs from the provided entity list above.
{rel_examples_str}
"""

        # Use custom relationship prompt if provided
        if custom_prompt and "{relationship_instructions}" in custom_prompt:
            relationship_prompt = custom_prompt.format(
                text=text,
                relationship_instructions=relation_types_str,
                format_instructions=relationship_format_instructions
            )
        else:
            relationship_prompt = f"{relationship_format_instructions}\n\nText to analyze:\n{text}"
        
        # Use custom system prompt or default relationship detection prompt
        relationship_sys_prompt = system_prompt or SYSTEM_PROMPTS.get("relationship_detection", "")
        
        # Extract relationships
        relationship_completion_result = await generate_completion(
            prompt=relationship_prompt,
            model=model,
            provider=provider_instance.__class__.__name__.lower(),
            temperature=temperature,
            max_tokens=2000,
            additional_params={
                "system_prompt": relationship_sys_prompt if relationship_sys_prompt else None,
                **additional_params
            }
        )
        
        # Check if completion was successful
        if not relationship_completion_result.get("success", False):
            error_message = relationship_completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Relationship extraction failed: {error_message}", 
                provider=provider_instance.__class__.__name__,
                model=model or "default"
            )
        
        # Track usage
        total_input_tokens += relationship_completion_result["tokens"]["input"]
        total_output_tokens += relationship_completion_result["tokens"]["output"]
        total_cost += relationship_completion_result["cost"]
        
        # Parse relationship response
        try:
            # Extract JSON from the response
            json_match = re.search(r'(\{.*\})', relationship_completion_result["text"], re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in the relationship extraction response.")
            
            relationship_data = json.loads(json_match.group(0))
            
            if "relationships" not in relationship_data or not isinstance(relationship_data["relationships"], list):
                raise ValueError("Invalid relationship extraction result: missing 'relationships' array.")
            
            relationships = relationship_data["relationships"]
            
            # Limit to max_relations if needed
            if len(relationships) > max_relations:
                relationships = relationships[:max_relations]
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ToolError(
                f"Failed to parse relationship extraction: {str(e)}",
                error_code="PARSING_ERROR",
                details={"response_text": relationship_completion_result["text"]}
            ) from e
        
        # --- Stage 3: Coreference Resolution (Optional) ---
        # If automatic_coreference is enabled, perform an additional step to resolve references
        if automatic_coreference and len(entities) > 1:
            # This would be a separate step to resolve coreferences and merge entities
            # For simplicity, we'll skip the actual implementation here
            pass
        
        # --- Combine Results ---
        combined_result = {
            "entities": entities,
            "relationships": relationships,
            "model": entity_completion_result["model"],  # Use model from first call
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
            "cost": total_cost
        }
        
        return combined_result
        
    except Exception as e:
        # Convert to provider error
        error_model = model or "default"
        raise ProviderError(
            f"Multi-stage entity graph extraction failed for model '{error_model}': {str(e)}",
            provider=provider_instance.__class__.__name__,
            model=error_model,
            cause=e
        ) from e

async def _perform_chunked_extraction(
    text: str,
    provider_instance: Any,
    model: Optional[str],
    entity_types: Optional[List[str]],
    relation_types: Optional[List[str]],
    include_evidence: bool,
    include_attributes: bool,
    include_positions: bool,
    include_temporal_info: bool,
    min_confidence: float,
    max_entities: int,
    max_relations: int,
    schema: Optional[Dict[str, Any]],
    context_window: int,
    chunk_size: Optional[int],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    language: Optional[str],
    example_entities: Optional[List[Dict[str, Any]]],
    example_relationships: Optional[List[Dict[str, Any]]],
    enable_reasoning: bool = False,
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Chunks large text and processes each chunk separately, then merges results."""
    # Track token usage and cost
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    # Determine chunk size (in tokens)
    if not chunk_size:
        # Estimate reasonable chunk size based on context window
        # Leave room for prompt and response
        prompt_tokens_estimate = 500
        response_tokens_estimate = 1500
        available_tokens = context_window - prompt_tokens_estimate - response_tokens_estimate
        
        # Use 80% of available tokens to be safe
        chunk_size = int(available_tokens * 0.8)
        
        # Cap at a reasonable size
        chunk_size = min(chunk_size, 12000)
    
    # Use the chunk_document tool to split the text
    try:
        chunk_response = await chunk_document(
            document=text,
            chunk_size=chunk_size,
            overlap=100,  # Overlap to maintain context across chunks
            method="semantic"  # Try to split at semantic boundaries
        )
        
        chunks = chunk_response.get("chunks", [])
        
        if not chunks:
            raise ToolError(
                "Failed to chunk document for processing.",
                error_code="CHUNKING_ERROR"
            )
            
        # Log chunking info
        logger.info(
            f"Chunked document into {len(chunks)} parts (avg {len(chunks[0]) // 4} tokens per chunk)."
        )
        
    except Exception as e:
        # If chunking fails, fall back to standard extraction
        logger.warning(
            f"Chunking failed: {str(e)}. Falling back to standard extraction.",
            exc_info=True
        )
        
        # Try to process the entire text
        return await _perform_standard_extraction(
            text=text,
            provider_instance=provider_instance,
            model=model,
            entity_types=entity_types,
            relation_types=relation_types,
            include_evidence=include_evidence,
            include_attributes=include_attributes,
            include_positions=include_positions,
            include_temporal_info=include_temporal_info,
            min_confidence=min_confidence,
            max_entities=max_entities,
            max_relations=max_relations,
            schema=schema,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            language=language,
            example_entities=example_entities,
            example_relationships=example_relationships,
            enable_reasoning=enable_reasoning,
            additional_params=additional_params
        )
    
    # Process each chunk separately
    chunk_results = []
    
    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            # Extract from this chunk
            chunk_result = await _perform_standard_extraction(
                text=chunk,
                provider_instance=provider_instance,
                model=model,
                entity_types=entity_types,
                relation_types=relation_types,
                include_evidence=include_evidence,
                include_attributes=include_attributes,
                include_positions=include_positions,
                include_temporal_info=include_temporal_info,
                min_confidence=min_confidence,
                # Use higher limits per chunk since we'll merge and deduplicate later
                max_entities=max_entities * 2,
                max_relations=max_relations * 2,
                schema=schema,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                language=language,
                example_entities=example_entities,
                example_relationships=example_relationships,
                enable_reasoning=enable_reasoning,
                additional_params=additional_params
            )
            
            # Track usage
            total_input_tokens += chunk_result.get("tokens", {}).get("input", 0)
            total_output_tokens += chunk_result.get("tokens", {}).get("output", 0)
            total_cost += chunk_result.get("cost", 0.0)
            
            # Add chunk metadata
            chunk_result["chunk_index"] = i
            chunk_result["chunk_text"] = chunk
            
            chunk_results.append(chunk_result)
            
        except Exception as e:
            # Log error but continue with other chunks
            logger.error(
                f"Error processing chunk {i+1}: {str(e)}",
                exc_info=True
            )
    
    # If all chunks failed, raise error
    if not chunk_results:
        raise ToolError(
            "All document chunks failed to process.",
            error_code="CHUNK_PROCESSING_ERROR"
        )
    
    # Merge results from all chunks
    merged_result = _merge_chunk_results(
        chunk_results,
        max_entities=max_entities,
        max_relations=max_relations,
        min_confidence=min_confidence
    )
    
    # Add token usage and cost
    merged_result["model"] = chunk_results[0]["model"]  # Use model from first chunk
    merged_result["tokens"] = {
        "input": total_input_tokens,
        "output": total_output_tokens,
        "total": total_input_tokens + total_output_tokens,
    }
    merged_result["cost"] = total_cost
    
    return merged_result

async def _perform_incremental_extraction(
    text: str,
    existing_graph: Dict[str, Any],
    provider_instance: Any,
    model: Optional[str],
    entity_types: Optional[List[str]],
    relation_types: Optional[List[str]],
    include_evidence: bool,
    include_attributes: bool,
    include_positions: bool,
    include_temporal_info: bool,
    min_confidence: float,
    max_entities: int,
    max_relations: int,
    schema: Optional[Dict[str, Any]],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    language: Optional[str],
    additional_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Incremental extraction that builds on an existing graph."""
    # Validate existing graph structure
    if not isinstance(existing_graph, dict) or "entities" not in existing_graph or "relationships" not in existing_graph:
        raise ToolInputError(
            "Invalid existing_graph format. Must contain 'entities' and 'relationships' keys."
        )
    
    existing_entities = existing_graph.get("entities", [])
    existing_relationships = existing_graph.get("relationships", [])
    
    if not isinstance(existing_entities, list) or not isinstance(existing_relationships, list):
        raise ToolInputError(
            "Invalid existing_graph format. 'entities' and 'relationships' must be lists."
        )
    
    # Extract existing entity IDs and names for reference
    existing_entity_ids = {e.get("id") for e in existing_entities if "id" in e}
    existing_entity_names = {e.get("name").lower(): e.get("id") for e in existing_entities if "name" in e}
    
    # Format existing entity list for context
    entity_context = ""
    for entity in existing_entities[:50]:  # Limit to avoid huge prompts
        entity_id = entity.get("id", "")
        entity_name = entity.get("name", "")
        entity_type = entity.get("type", "")
        entity_context += f"- ID: {entity_id}, Name: {entity_name}, Type: {entity_type}\n"
    
    # Entity types to focus on
    entity_types_str = ""
    if entity_types:
        entity_types_str = "Entity types to extract:\n" + "\n".join([f"- {t}" for t in entity_types])
    
    # Relationship types to focus on
    relation_types_str = ""
    if relation_types:
        relation_types_str = "Relationship types to extract:\n" + "\n".join([f"- {t}" for t in relation_types])
    
    # Schema guidance if provided
    schema_guidance = ""
    if schema:
        schema_guidance = "Use the following schema for extraction:\n\n"
        
        # Entity schema
        if "entities" in schema:
            schema_guidance += "Entity Types:\n"
            for entity_type in schema.get("entities", []):
                schema_guidance += f"- {entity_type['type']}"
                if "attributes" in entity_type:
                    schema_guidance += f" (Attributes: {', '.join(entity_type['attributes'])})"
                schema_guidance += "\n"
        
        # Relationship schema
        if "relationships" in schema:
            schema_guidance += "\nRelationship Types:\n"
            for rel_type in schema.get("relationships", []):
                source_types = ", ".join(rel_type.get("source_types", ["Any"]))
                target_types = ", ".join(rel_type.get("target_types", ["Any"]))
                schema_guidance += f"- {rel_type['type']} (From: {source_types}, To: {target_types})\n"
    
    # Language specification
    language_instruction = ""
    if language:
        language_instruction = f"The text is in {language}. Extract entities and relationships accordingly."
    
    # Format instructions for incremental extraction
    incremental_instructions = f"""
TASK: Extract new entities and relationships from the text below, and connect them to the existing knowledge graph.

EXISTING ENTITIES:
{entity_context}

{entity_types_str}
{relation_types_str}
{schema_guidance}
{language_instruction}

IMPORTANT INSTRUCTIONS:
1. Extract new entities not present in the existing list
2. Extract relationships between new entities
3. Extract relationships between new entities and existing entities
4. Use existing entity IDs when referencing existing entities
5. Create new IDs for new entities that follow the same format

Format your response as JSON with the following structure:
{{
  "new_entities": [
    {{
      "id": "ent_new_1", 
      "name": "entity_name",
      "type": "entity_type",
      {"mentions": [{"text": "mention_text", "pos": [start_pos, end_pos]}]," if include_positions else ""}
      {"attributes": {"attribute_name": "value"}," if include_attributes else ""}
    }}
  ],
  "new_relationships": [
    {{
      "id": "rel_new_1",
      "source": "entity_id",  # Can be existing or new entity ID
      "target": "entity_id",  # Can be existing or new entity ID
      "type": "relationship_type",
      "confidence": 0.95,
      {"evidence": "text_supporting_relationship"," if include_evidence else ""}
      {"temporal": {"start": "timestamp", "end": "timestamp"}," if include_temporal_info else ""}
    }}
  ]
}}

Only include relationships with confidence ≥ {min_confidence}.
"""

    # Use custom prompt if provided
    if custom_prompt:
        prompt = custom_prompt.format(
            text=text,
            entity_instructions=entity_types_str,
            relationship_instructions=relation_types_str,
            format_instructions=incremental_instructions,
            existing_entities=entity_context
        )
    else:
        prompt = f"{incremental_instructions}\n\nText to analyze:\n{text}"
    
    # Use custom system prompt if provided, otherwise use default
    sys_prompt = system_prompt or SYSTEM_PROMPTS.get("entity_detection", "")
    
    # Execute extraction
    try:
        # Set temperature low for deterministic extraction
        temperature = additional_params.pop("temperature", 0.1)
        
        # Generate incremental extraction
        result = await provider_instance.generate_completion(
            prompt=prompt,
            system_prompt=sys_prompt if sys_prompt else None,
            model=model,
            temperature=temperature,
            max_tokens=3000,
            **additional_params
        )
        
        # Parse response
        try:
            # Extract JSON from the response
            json_match = re.search(r'(\{.*\})', result.text, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in the incremental extraction response.")
            
            incremental_data = json.loads(json_match.group(0))
            
            # Validate response structure
            if "new_entities" not in incremental_data or "new_relationships" not in incremental_data:
                raise ValueError("Invalid incremental extraction result: missing 'new_entities' or 'new_relationships'.")
            
            new_entities = incremental_data.get("new_entities", [])
            new_relationships = incremental_data.get("new_relationships", [])
            
            # Validate and fix entity references
            validated_entities, validated_relationships = _validate_incremental_data(
                new_entities,
                new_relationships,
                existing_entity_ids,
                existing_entity_names
            )
            
            # Combine with existing graph
            combined_entities = existing_entities + validated_entities
            combined_relationships = existing_relationships + validated_relationships
            
            # Deduplicate and limit if needed
            if len(combined_entities) > max_entities:
                # Prioritize existing entities
                remaining_slots = max(0, max_entities - len(existing_entities))
                if remaining_slots > 0:
                    combined_entities = existing_entities + validated_entities[:remaining_slots]
                else:
                    combined_entities = existing_entities[:max_entities]
            
            if len(combined_relationships) > max_relations:
                # Prioritize existing relationships
                remaining_slots = max(0, max_relations - len(existing_relationships))
                if remaining_slots > 0:
                    combined_relationships = existing_relationships + validated_relationships[:remaining_slots]
                else:
                    combined_relationships = existing_relationships[:max_relations]
            
            # Prepare final result
            combined_result = {
                "entities": combined_entities,
                "relationships": combined_relationships,
                "incremental_stats": {
                    "new_entities_found": len(validated_entities),
                    "new_relationships_found": len(validated_relationships),
                    "total_entities": len(combined_entities),
                    "total_relationships": len(combined_relationships)
                },
                "model": result.model,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.total_tokens,
                },
                "cost": result.cost
            }
            
            return combined_result
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ToolError(
                f"Failed to parse incremental extraction result: {str(e)}",
                error_code="PARSING_ERROR",
                details={"response_text": result.text}
            ) from e
            
    except Exception as e:
        # Convert to provider error
        error_model = model or "default"
        raise ProviderError(
            f"Incremental entity graph extraction failed for model '{error_model}': {str(e)}",
            provider=provider_instance.__class__.__name__,
            model=error_model,
            cause=e
        ) from e

async def _perform_structured_extraction(
    text: str,
    provider_instance: Any,
    model: Optional[str],
    entity_types: Optional[List[str]],
    relation_types: Optional[List[str]],
    include_evidence: bool,
    include_attributes: bool,
    include_positions: bool,
    include_temporal_info: bool,
    min_confidence: float,
    max_entities: int,
    max_relations: int,
    schema: Optional[Dict[str, Any]],
    example_entities: Optional[List[Dict[str, Any]]],
    example_relationships: Optional[List[Dict[str, Any]]],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    language: Optional[str],
    additional_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Performs extraction using structured examples for consistent formatting."""
    # Ensure we have examples
    if not example_entities or not example_relationships:
        raise ToolError(
            "The 'structured' strategy requires example_entities and example_relationships.",
            error_code="MISSING_EXAMPLES"
        )
    
    # Entity types to focus on
    entity_types_str = ""
    if entity_types:
        entity_types_str = "Entity types to extract:\n" + "\n".join([f"- {t}" for t in entity_types])
    
    # Relationship types to focus on
    relation_types_str = ""
    if relation_types:
        relation_types_str = "Relationship types to extract:\n" + "\n".join([f"- {t}" for t in relation_types])
    
    # Schema guidance if provided
    schema_guidance = ""
    if schema:
        schema_guidance = "Use the following schema for extraction:\n\n"
        
        # Entity schema
        if "entities" in schema:
            schema_guidance += "Entity Types:\n"
            for entity_type in schema.get("entities", []):
                schema_guidance += f"- {entity_type['type']}"
                if "attributes" in entity_type:
                    schema_guidance += f" (Attributes: {', '.join(entity_type['attributes'])})"
                schema_guidance += "\n"
        
        # Relationship schema
        if "relationships" in schema:
            schema_guidance += "\nRelationship Types:\n"
            for rel_type in schema.get("relationships", []):
                source_types = ", ".join(rel_type.get("source_types", ["Any"]))
                target_types = ", ".join(rel_type.get("target_types", ["Any"]))
                schema_guidance += f"- {rel_type['type']} (From: {source_types}, To: {target_types})\n"
    
    # Language specification
    language_instruction = ""
    if language:
        language_instruction = f"The text is in {language}. Extract entities and relationships accordingly."
    
    # Format example_entities and example_relationships as structured examples
    examples_str = "EXAMPLES:\n\nExample 1 input text:\n"
    
    # Try to find a matching example in FEW_SHOT_EXAMPLES
    example_found = False
    for _domain_name, example in FEW_SHOT_EXAMPLES.items():
        if example_entities == example["entities"] and example_relationships == example["relationships"]:
            examples_str += example["text"] + "\n\n"
            example_found = True
            break
    
    if not example_found:
        examples_str += "This is an example text.\n\n"
    
    examples_str += "Example 1 output:\n```json\n{\n"
    examples_str += '  "entities": ' + json.dumps(example_entities, indent=2) + ",\n"
    examples_str += '  "relationships": ' + json.dumps(example_relationships, indent=2) + "\n"
    examples_str += "}\n```\n\n"
    
    # Format instructions
    structured_instructions = f"""
Extract entities and their relationships from the text below following EXACTLY the same format as the examples.
{entity_types_str}
{relation_types_str}
{schema_guidance}
{language_instruction}

{examples_str}

Your output must strictly follow the same format as the example above, including:
1. The exact same attribute names and structure
2. The same ID naming convention
3. The same level of detail in each field

Only include relationships with confidence ≥ {min_confidence}.
Limit to approximately {max_entities} most important entities and {max_relations} most significant relationships.
"""

    # Use custom prompt if provided
    if custom_prompt:
        prompt = custom_prompt.format(
            text=text,
            entity_instructions=entity_types_str,
            relationship_instructions=relation_types_str,
            format_instructions=structured_instructions,
            examples=examples_str
        )
    else:
        prompt = f"{structured_instructions}\n\nText to analyze:\n{text}"
    
    # Use custom system prompt if provided, otherwise use default
    sys_prompt = system_prompt or SYSTEM_PROMPTS.get("entity_detection", "")
    
    # Execute extraction
    try:
        # Set temperature very low for consistent formatting
        temperature = additional_params.pop("temperature", 0.1)
        
        # Generate structured extraction
        result = await provider_instance.generate_completion(
            prompt=prompt,
            system_prompt=sys_prompt if sys_prompt else None,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            **additional_params
        )
        
        # Parse response
        try:
            # Extract JSON from the response
            json_match = re.search(r'(\{.*\})', result.text, re.DOTALL)
            if not json_match:
                # Try to find JSON within code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', result.text, re.DOTALL)
                
            if not json_match:
                raise ValueError("No valid JSON found in the structured extraction response.")
            
            graph_data = json.loads(json_match.group(1))
            
            # Validate and clean up the extracted data
            graph_data = _validate_graph_data(
                graph_data, 
                min_confidence, 
                max_entities, 
                max_relations
            )
            
            # Add model metadata to result
            graph_data["model"] = result.model
            graph_data["tokens"] = {
                "input": result.input_tokens,
                "output": result.output_tokens,
                "total": result.total_tokens,
            }
            graph_data["cost"] = result.cost
            
            return graph_data
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ToolError(
                f"Failed to parse structured entity graph extraction: {str(e)}",
                error_code="PARSING_ERROR",
                details={"response_text": result.text}
            ) from e
            
    except Exception as e:
        # Convert to provider error
        error_model = model or "default"
        raise ProviderError(
            f"Structured entity graph extraction failed for model '{error_model}': {str(e)}",
            provider=provider_instance.__class__.__name__,
            model=error_model,
            cause=e
        ) from e

async def _perform_schema_guided_extraction(
    text: str,
    provider_instance: Any,
    model: Optional[str],
    schema: Dict[str, Any],
    include_evidence: bool,
    include_attributes: bool,
    include_positions: bool,
    include_temporal_info: bool,
    min_confidence: float,
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    language: Optional[str],
    additional_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Performs extraction strictly according to a predefined schema."""
    # Entity schema string
    entity_schema_str = "ENTITY TYPES:\n"
    for entity_type in schema.get("entities", []):
        entity_schema_str += f"- {entity_type['type']}"
        if "attributes" in entity_type:
            entity_schema_str += f" (Attributes: {', '.join(entity_type['attributes'])})"
        entity_schema_str += "\n"
    
    # Relationship schema string
    rel_schema_str = "RELATIONSHIP TYPES:\n"
    for rel_type in schema.get("relationships", []):
        source_types = ", ".join(rel_type.get("source_types", ["Any"]))
        target_types = ", ".join(rel_type.get("target_types", ["Any"]))
        rel_schema_str += f"- {rel_type['type']} (From: {source_types}, To: {target_types})\n"
    
    # Language specification
    language_instruction = ""
    if language:
        language_instruction = f"The text is in {language}. Extract entities and relationships accordingly."
    
    # Format instructions for schema-guided extraction
    schema_instructions = f"""
Extract entities and relationships from the text STRICTLY according to the provided schema.
{language_instruction}

{entity_schema_str}

{rel_schema_str}

RULES:
1. ONLY extract entities of the types specified in the schema
2. ONLY extract relationships of the types specified in the schema
3. ONLY create relationships between entity types as defined in the schema
4. Assign appropriate attributes to entities as defined in the schema
5. If the text contains entities or relationships outside the schema, DO NOT include them

Format your response as JSON with the following structure:
{{
  "entities": [
    {{
      "id": "ent1", 
      "name": "entity_name",
      "type": "entity_type", # MUST match one of the types in the schema
      {"mentions": [{"text": "mention_text", "pos": [start_pos, end_pos]}]," if include_positions else ""}
      {"attributes": {{# ONLY include attributes defined in the schema
        "attribute_name": "value"
      }}," if include_attributes else ""}
    }}
  ],
  "relationships": [
    {{
      "id": "rel1",
      "source": "ent1", # ID of a source entity with a type allowed by the schema
      "target": "ent2", # ID of a target entity with a type allowed by the schema
      "type": "relationship_type", # MUST match one of the types in the schema
      "confidence": 0.95,
      {"evidence": "text_supporting_relationship"," if include_evidence else ""}
      {"temporal": {"start": "timestamp", "end": "timestamp"}," if include_temporal_info else ""}
    }}
  ]
}}

Only include relationships with confidence ≥ {min_confidence}.
Ensure all entities and relationships strictly conform to the schema.
"""

    # Use custom prompt if provided
    if custom_prompt:
        prompt = custom_prompt.format(
            text=text,
            entity_schema=entity_schema_str,
            relationship_schema=rel_schema_str,
            format_instructions=schema_instructions
        )
    else:
        prompt = f"{schema_instructions}\n\nText to analyze:\n{text}"
    
    # Use custom system prompt if provided, otherwise use default
    sys_prompt = system_prompt or SYSTEM_PROMPTS.get("entity_detection", "")
    
    # Execute extraction
    try:
        # Set temperature very low for deterministic schema-guided extraction
        temperature = additional_params.pop("temperature", 0.1)
        
        # Generate schema-guided extraction
        result = await provider_instance.generate_completion(
            prompt=prompt,
            system_prompt=sys_prompt if sys_prompt else None,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            **additional_params
        )
        
        # Parse response
        try:
            # Extract JSON from the response
            json_match = re.search(r'(\{.*\})', result.text, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in the schema-guided extraction response.")
            
            graph_data = json.loads(json_match.group(0))
            
            # Validate entities against schema
            valid_entity_types = {entity_type["type"] for entity_type in schema.get("entities", [])}
            entities = []
            
            for entity in graph_data.get("entities", []):
                if "type" in entity and entity["type"] in valid_entity_types:
                    # Filter attributes to only include those in schema
                    if include_attributes and "attributes" in entity:
                        entity_type = entity["type"]
                        valid_attributes = next(
                            (e.get("attributes", []) for e in schema.get("entities", []) if e["type"] == entity_type),
                            []
                        )
                        
                        filtered_attributes = {
                            k: v for k, v in entity["attributes"].items() if k in valid_attributes
                        }
                        
                        entity["attributes"] = filtered_attributes
                    
                    entities.append(entity)
            
            # Validate relationships against schema
            valid_relationship_types = {rel_type["type"] for rel_type in schema.get("relationships", [])}
            allowed_source_target = {
                rel_type["type"]: (rel_type.get("source_types", []), rel_type.get("target_types", []))
                for rel_type in schema.get("relationships", [])
            }
            
            # Build entity ID to type mapping
            entity_id_to_type = {
                entity["id"]: entity["type"] for entity in entities if "id" in entity and "type" in entity
            }
            
            relationships = []
            
            for relation in graph_data.get("relationships", []):
                if "type" not in relation or relation["type"] not in valid_relationship_types:
                    continue
                
                if "source" not in relation or "target" not in relation:
                    continue
                
                # Check if source and target entities exist and are of allowed types
                source_id = relation["source"]
                target_id = relation["target"]
                
                if source_id not in entity_id_to_type or target_id not in entity_id_to_type:
                    continue
                
                source_type = entity_id_to_type[source_id]
                target_type = entity_id_to_type[target_id]
                relation_type = relation["type"]
                
                allowed_source_types, allowed_target_types = allowed_source_target.get(relation_type, ([], []))
                
                # If either list is empty, assume any type is allowed
                if (not allowed_source_types or source_type in allowed_source_types) and \
                   (not allowed_target_types or target_type in allowed_target_types):
                    relationships.append(relation)
            
            # Prepare validated result
            validated_result = {
                "entities": entities,
                "relationships": relationships,
                "model": result.model,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.total_tokens,
                },
                "cost": result.cost
            }
            
            return validated_result
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ToolError(
                f"Failed to parse schema-guided extraction result: {str(e)}",
                error_code="PARSING_ERROR",
                details={"response_text": result.text}
            ) from e
            
    except Exception as e:
        # Convert to provider error
        error_model = model or "default"
        raise ProviderError(
            f"Schema-guided entity graph extraction failed for model '{error_model}': {str(e)}",
            provider=provider_instance.__class__.__name__,
            model=error_model,
            cause=e
        ) from e

# --- Helper Functions ---

def _merge_chunk_results(
    chunk_results: List[Dict[str, Any]],
    max_entities: int,
    max_relations: int,
    min_confidence: float
) -> Dict[str, Any]:
    """Merges results from multiple text chunks, handling duplicates and overlaps."""
    # Initialize merged collections
    merged_entities = []
    merged_relationships = []
    entity_map = {}  # Maps entity names to their IDs
    seen_entity_ids = set()
    seen_relationship_signatures = set()  # (source, target, type) tuples
    
    # First pass - collect and deduplicate entities
    for chunk_result in chunk_results:
        chunk_entities = chunk_result.get("entities", [])
        
        for entity in chunk_entities:
            if "id" not in entity or "name" not in entity or "type" not in entity:
                continue
                
            # Normalize entity name for matching
            normalized_name = entity["name"].lower().strip()
            
            # If we've seen this entity before (by name)
            if normalized_name in entity_map:
                existing_id = entity_map[normalized_name]
                existing_idx = next(i for i, e in enumerate(merged_entities) if e["id"] == existing_id)
                existing_entity = merged_entities[existing_idx]
                
                # Merge mentions if available
                if "mentions" in entity and "mentions" in existing_entity:
                    # Use a set to deduplicate based on text
                    mention_texts = {m["text"] for m in existing_entity["mentions"]}
                    for mention in entity["mentions"]:
                        if mention["text"] not in mention_texts:
                            existing_entity["mentions"].append(mention)
                            mention_texts.add(mention["text"])
                
                # Merge attributes if available
                if "attributes" in entity and "attributes" in existing_entity:
                    existing_entity["attributes"].update(entity["attributes"])
            else:
                # New entity - add to merged list
                new_id = f"ent{len(merged_entities) + 1}"
                new_entity = {**entity, "id": new_id}
                
                merged_entities.append(new_entity)
                entity_map[normalized_name] = new_id
                seen_entity_ids.add(new_id)
    
    # Second pass - collect and deduplicate relationships
    for chunk_result in chunk_results:
        chunk_relationships = chunk_result.get("relationships", [])
        
        for relationship in chunk_relationships:
            if "source" not in relationship or "target" not in relationship or "type" not in relationship:
                continue
                
            # Map source and target to new entity IDs if needed
            source_entity = next((e for e in chunk_result.get("entities", []) if e["id"] == relationship["source"]), None)
            target_entity = next((e for e in chunk_result.get("entities", []) if e["id"] == relationship["target"]), None)
            
            if not source_entity or not target_entity:
                continue
                
            source_name = source_entity["name"].lower().strip()
            target_name = target_entity["name"].lower().strip()
            
            if source_name in entity_map and target_name in entity_map:
                new_source_id = entity_map[source_name]
                new_target_id = entity_map[target_name]
                relation_type = relationship["type"]
                
                # Check if this relationship already exists
                signature = (new_source_id, new_target_id, relation_type)
                if signature in seen_relationship_signatures:
                    continue
                
                # Get confidence
                confidence = relationship.get("confidence", 1.0)
                if confidence < min_confidence:
                    continue
                
                # Add new relationship
                new_id = f"rel{len(merged_relationships) + 1}"
                new_relationship = {
                    **relationship,
                    "id": new_id,
                    "source": new_source_id,
                    "target": new_target_id
                }
                
                merged_relationships.append(new_relationship)
                seen_relationship_signatures.add(signature)
    
    # Limit to max entities and relationships if needed
    if len(merged_entities) > max_entities:
        merged_entities = merged_entities[:max_entities]
        
        # Keep only relationships that reference existing entities
        valid_entity_ids = {e["id"] for e in merged_entities}
        merged_relationships = [
            r for r in merged_relationships 
            if r["source"] in valid_entity_ids and r["target"] in valid_entity_ids
        ]
    
    if len(merged_relationships) > max_relations:
        # Sort by confidence and take top max_relations
        merged_relationships = sorted(
            merged_relationships, 
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )[:max_relations]
    
    # Return merged result
    return {
        "entities": merged_entities,
        "relationships": merged_relationships
    }

def _validate_incremental_data(
    new_entities: List[Dict[str, Any]],
    new_relationships: List[Dict[str, Any]],
    existing_entity_ids: Set[str],
    existing_entity_names: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validates and fixes entities and relationships from incremental extraction."""
    validated_entities = []
    validated_relationships = []
    
    # Assign new entity IDs and track mappings
    id_mapping = {}  # Maps original IDs to new IDs
    new_entity_names = {}  # Maps normalized names to new IDs
    
    # Process entities
    for entity in new_entities:
        if "id" not in entity or "name" not in entity or "type" not in entity:
            continue
        
        original_id = entity["id"]
        entity_name = entity["name"].lower().strip()
        
        # Check if this entity already exists by name
        if entity_name in existing_entity_names:
            id_mapping[original_id] = existing_entity_names[entity_name]
            continue  # Skip adding this entity, just map its ID
        
        # Check if this entity name is already among the new entities
        if entity_name in new_entity_names:
            id_mapping[original_id] = new_entity_names[entity_name]
            continue  # Skip adding duplicate entity
        
        # Create new unique ID
        new_id = f"ent_new_{len(validated_entities) + 1}"
        while new_id in existing_entity_ids:
            new_id = f"ent_new_{len(validated_entities) + len(existing_entity_ids) + 1}"
        
        # Create validated entity with new ID
        validated_entity = {**entity, "id": new_id}
        validated_entities.append(validated_entity)
        
        # Track mappings
        id_mapping[original_id] = new_id
        new_entity_names[entity_name] = new_id
    
    # Process relationships
    for relationship in new_relationships:
        if "source" not in relationship or "target" not in relationship or "type" not in relationship:
            continue
        
        original_source = relationship["source"]
        original_target = relationship["target"]
        
        # Map source and target IDs
        source_id = original_source
        target_id = original_target
        
        # If source references a new entity, map to new ID
        if original_source in id_mapping:
            source_id = id_mapping[original_source]
        
        # If target references a new entity, map to new ID
        if original_target in id_mapping:
            target_id = id_mapping[original_target]
        
        # Skip if either source or target is invalid
        if source_id not in existing_entity_ids and source_id not in {e["id"] for e in validated_entities}:
            continue
            
        if target_id not in existing_entity_ids and target_id not in {e["id"] for e in validated_entities}:
            continue
        
        # Create validated relationship with new IDs
        new_id = f"rel_new_{len(validated_relationships) + 1}"
        validated_relationship = {
            **relationship,
            "id": new_id,
            "source": source_id,
            "target": target_id
        }
        
        validated_relationships.append(validated_relationship)
    
    return validated_entities, validated_relationships

def _normalize_entities(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes entity names and attributes, resolving duplicates."""
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    
    if not entities:
        return extraction_result
    
    # Create name normalization mapping
    normalized_names = {}
    normalized_entities = []
    id_mapping = {}  # Old ID to new ID mapping
    
    for entity in entities:
        if "name" not in entity:
            continue
            
        # Normalize name (capitalize properly, remove extra whitespace)
        orig_name = entity["name"]
        normalized_name = " ".join(orig_name.split())  # Remove extra whitespace
        
        # Apply basic capitalization rules (can be made more sophisticated)
        if entity.get("type") in ["Person", "Organization", "Location", "Event"]:
            # Title case for proper nouns
            normalized_name = " ".join(word.capitalize() for word in normalized_name.split())
        
        # Check if we've seen this normalized name before
        if normalized_name.lower() in normalized_names:
            # Get the existing entity ID and map this ID to it
            existing_id = normalized_names[normalized_name.lower()]
            id_mapping[entity["id"]] = existing_id
            
            # Find the existing entity to merge attributes
            existing_entity = next(e for e in normalized_entities if e["id"] == existing_id)
            
            # Merge mentions if available
            if "mentions" in entity and "mentions" in existing_entity:
                existing_mentions = {m["text"] for m in existing_entity["mentions"]}
                for mention in entity["mentions"]:
                    if mention["text"] not in existing_mentions:
                        existing_entity["mentions"].append(mention)
            
            # Merge attributes if available
            if "attributes" in entity and "attributes" in existing_entity:
                existing_entity["attributes"].update(entity["attributes"])
        else:
            # New normalized entity
            new_entity = {**entity, "name": normalized_name}
            normalized_entities.append(new_entity)
            normalized_names[normalized_name.lower()] = entity["id"]
    
    # Update relationships to use normalized entity IDs
    normalized_relationships = []
    for relationship in relationships:
        if "source" not in relationship or "target" not in relationship:
            continue
            
        source_id = relationship["source"]
        target_id = relationship["target"]
        
        # Map to new IDs if needed
        if source_id in id_mapping:
            source_id = id_mapping[source_id]
        
        if target_id in id_mapping:
            target_id = id_mapping[target_id]
        
        # Create normalized relationship
        normalized_relationship = {
            **relationship,
            "source": source_id,
            "target": target_id
        }
        
        normalized_relationships.append(normalized_relationship)
    
    # Return normalized result
    normalized_result = {
        **extraction_result,
        "entities": normalized_entities,
        "relationships": normalized_relationships
    }
    
    return normalized_result

def _add_graph_metrics(extraction_result: Dict[str, Any], sort_by: str = "confidence") -> Dict[str, Any]:
    """Adds computed graph metrics and sorts entities and relationships."""
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    
    if not entities or not relationships:
        return extraction_result
    
    # Create NetworkX graph for analysis (if available)
    try:
        if HAS_VISUALIZATION_LIBS:
            G = nx.DiGraph()
            
            # Add nodes
            for entity in entities:
                G.add_node(entity["id"], label=entity["name"], type=entity["type"])
            
            # Add edges
            for rel in relationships:
                G.add_edge(
                    rel["source"],
                    rel["target"],
                    label=rel["type"],
                    confidence=rel.get("confidence", 1.0)
                )
            
            # Calculate graph metrics
            metrics = {}
            
            # Basic graph properties
            metrics["node_count"] = G.number_of_nodes()
            metrics["edge_count"] = G.number_of_edges()
            metrics["density"] = nx.density(G)
            
            # Average degree
            metrics["avg_in_degree"] = sum(dict(G.in_degree()).values()) / max(1, len(G))
            metrics["avg_out_degree"] = sum(dict(G.out_degree()).values()) / max(1, len(G))
            
            # Try to calculate diameter (only for connected graphs)
            try:
                # Convert to undirected for diameter calculation
                undirected_G = G.to_undirected()
                if nx.is_connected(undirected_G):
                    metrics["diameter"] = nx.diameter(undirected_G)
            except nx.NetworkXError:
                pass
            
            # Calculate node centrality metrics
            try:
                centrality = nx.betweenness_centrality(G)
                degree_centrality = nx.degree_centrality(G)
                
                # Add centrality to entities
                for entity in entities:
                    entity_id = entity["id"]
                    entity["centrality"] = centrality.get(entity_id, 0.0)
                    entity["degree_centrality"] = degree_centrality.get(entity_id, 0.0)
            except nx.NetworkXError:
                # Fallback if centrality calculation fails
                for entity in entities:
                    entity["centrality"] = 0.0
                    entity["degree_centrality"] = 0.0
            
            # Add metrics to result
            extraction_result["metrics"] = metrics
    except Exception as exc:
        # If NetworkX processing fails or is unavailable, continue without metrics
        logger.debug(f"NetworkX metrics calculation failed: {exc}")
        pass
    
    # Sort entities based on requested criteria
    if sort_by == "centrality" and all("centrality" in e for e in entities):
        entities.sort(key=lambda x: x.get("centrality", 0.0), reverse=True)
    elif sort_by == "mentions" and all("mentions" in e for e in entities):
        entities.sort(key=lambda x: len(x.get("mentions", [])), reverse=True)
    
    # Sort relationships by confidence
    relationships.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    
    # Update sorted entities and relationships
    extraction_result["entities"] = entities
    extraction_result["relationships"] = relationships
    
    return extraction_result

def _generate_visualization(
    extraction_result: Dict[str, Any],
    format: VisualizationFormat
) -> Dict[str, Any]:
    """Generates a visualization of the entity graph."""
    if not HAS_VISUALIZATION_LIBS or format == VisualizationFormat.NONE:
        return None
        
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    
    if not entities or not relationships:
        return None
    
    # Create visualization data
    viz_data = {}
    
    if format == VisualizationFormat.HTML:
        # Generate interactive HTML using Pyvis
        try:
            # Create network
            net = Network(notebook=False, height="750px", width="100%")
            
            # Configure network
            net.toggle_physics(True)
            net.show_buttons(filter_=['physics'])
            net.set_options("""
            {
              "nodes": {
                "shape": "dot",
                "size": 20,
                "font": {
                  "size": 12,
                  "face": "Arial"
                }
              },
              "edges": {
                "font": {
                  "size": 10,
                  "align": "middle"
                },
                "smooth": {
                  "type": "continuous"
                },
                "arrows": {
                  "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                  }
                }
              },
              "physics": {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                  "springLength": 150,
                  "avoidOverlap": 1
                }
              }
            }
            """)
            
            # Add nodes with colors by type
            entity_types = set(e.get("type", "Unknown") for e in entities)
            type_colors = {}
            
            # Generate colors for each entity type
            colors = [
                "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
                "#1abc9c", "#34495e", "#e67e22", "#8e44ad", "#d35400"
            ]
            
            for i, entity_type in enumerate(entity_types):
                type_colors[entity_type] = colors[i % len(colors)]
            
            # Add nodes
            for entity in entities:
                entity_id = entity["id"]
                entity_label = entity["name"]
                entity_type = entity.get("type", "Unknown")
                
                # Get attributes to add to node title (hover text)
                attributes = entity.get("attributes", {})
                title = f"Type: {entity_type}<br>"
                
                if attributes:
                    for attr, value in attributes.items():
                        title += f"{attr}: {value}<br>"
                
                # Use centrality for node size if available
                size = 25
                if "centrality" in entity:
                    # Scale centrality to reasonable node size
                    size = 15 + (entity["centrality"] * 50)
                
                net.add_node(
                    entity_id,
                    label=entity_label,
                    title=title,
                    color=type_colors.get(entity_type, "#cccccc"),
                    size=size
                )
            
            # Add edges
            for rel in relationships:
                source_id = rel["source"]
                target_id = rel["target"]
                rel_type = rel.get("type", "")
                
                # Skip if source or target doesn't exist
                if source_id not in [e["id"] for e in entities] or target_id not in [e["id"] for e in entities]:
                    continue
                
                # Get confidence for edge width
                confidence = rel.get("confidence", 1.0)
                width = 1 + (confidence * 3)  # Scale confidence to width
                
                # Get evidence for edge title (hover text)
                title = f"Type: {rel_type}<br>Confidence: {confidence:.2f}"
                if "evidence" in rel:
                    title += f"<br>Evidence: {rel['evidence']}"
                
                net.add_edge(
                    source_id,
                    target_id,
                    title=title,
                    label=rel_type,
                    width=width
                )
            
            # Generate HTML
            temp_html_path = os.path.join(tempfile.gettempdir(), f"graph_{uuid.uuid4()}.html")
            net.save_graph(temp_html_path)
            
            # Read the file
            with open(temp_html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            # Add visualization data
            viz_data["html"] = html_content
            viz_data["url"] = f"file://{temp_html_path}"
            
        except Exception as e:
            # Fall back to no visualization if there's an error
            viz_data["error"] = f"Failed to generate HTML visualization: {str(e)}"
            
    elif format == VisualizationFormat.SVG:
        # Generate SVG using NetworkX/Matplotlib
        try:
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for entity in entities:
                G.add_node(entity["id"], label=entity["name"], type=entity["type"])
            
            # Add edges
            for rel in relationships:
                G.add_edge(rel["source"], rel["target"], label=rel["type"])
            
            # Create temporary file for SVG
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 10))
            
            # Calculate positions using spring layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Draw nodes with colors by type
            entity_types = set(e.get("type", "Unknown") for e in entities)
            type_colors = {}
            
            # Generate colors for each entity type
            colors = [
                "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", 
                "#1abc9c", "#34495e", "#e67e22", "#8e44ad", "#d35400"
            ]
            
            for i, entity_type in enumerate(entity_types):
                type_colors[entity_type] = colors[i % len(colors)]
            
            # Get node colors
            node_colors = [type_colors.get(G.nodes[n]["type"], "#cccccc") for n in G.nodes]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_size=500,
                node_color=node_colors,
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                width=1.0,
                alpha=0.5,
                arrows=True,
                arrowsize=15
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                labels={n: G.nodes[n]["label"] for n in G.nodes},
                font_size=10
            )
            
            # Draw edge labels
            edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=8
            )
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=t)
                for t, color in type_colors.items()
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save as SVG
            temp_svg_path = os.path.join(tempfile.gettempdir(), f"graph_{uuid.uuid4()}.svg")
            plt.savefig(temp_svg_path, format="svg")
            plt.close()
            
            # Read the file
            with open(temp_svg_path, "r", encoding="utf-8") as f:
                svg_content = f.read()
            
            # Add visualization data
            viz_data["svg"] = svg_content
            viz_data["url"] = f"file://{temp_svg_path}"
            
        except Exception as e:
            # Fall back to no visualization if there's an error
            viz_data["error"] = f"Failed to generate SVG visualization: {str(e)}"
    
    elif format == VisualizationFormat.DOT:
        # Generate GraphViz DOT format
        try:
            dot_lines = ["digraph G {"]
            
            # Graph attributes
            dot_lines.append('  graph [rankdir=LR, fontname="Arial", nodesep=0.8];')
            dot_lines.append('  node [shape=ellipse, style=filled, fontname="Arial"];')
            dot_lines.append('  edge [fontname="Arial"];')
            
            # Generate node definitions
            entity_types = set(e.get("type", "Unknown") for e in entities)
            type_colors = {}
            
            # Generate colors for each entity type
            colors = [
                "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", 
                "#1abc9c", "#34495e", "#e67e22", "#8e44ad", "#d35400"
            ]
            
            for i, entity_type in enumerate(entity_types):
                type_colors[entity_type] = colors[i % len(colors)]
            
            # Add nodes
            for entity in entities:
                entity_id = entity["id"]
                entity_label = entity["name"].replace('"', "'")  # Use single quotes instead of escaping
                entity_type = entity.get("type", "Unknown")
                
                color = type_colors.get(entity_type, "#cccccc")
                
                dot_lines.append(f'  "{entity_id}" [label="{entity_label}", fillcolor="{color}"];')
            
            # Add edges
            for rel in relationships:
                source_id = rel["source"]
                target_id = rel["target"]
                rel_type = rel.get("type", "")
                
                # Skip if source or target doesn't exist
                if source_id not in [e["id"] for e in entities] or target_id not in [e["id"] for e in entities]:
                    continue
                
                rel_label = rel_type.replace('"', "'")  # Use single quotes instead
                
                dot_lines.append(f'  "{source_id}" -> "{target_id}" [label="{rel_label}"];')
            
            # Close graph
            dot_lines.append("}")
            
            # Join the DOT content
            dot_content = "\n".join(dot_lines)
            
            # Create file
            temp_dot_path = os.path.join(tempfile.gettempdir(), f"graph_{uuid.uuid4()}.dot")
            with open(temp_dot_path, "w", encoding="utf-8") as f:
                f.write(dot_content)
            
            # Add visualization data
            viz_data["dot"] = dot_content
            viz_data["url"] = f"file://{temp_dot_path}"
            
        except Exception as e:
            # Fall back to no visualization if there's an error
            viz_data["error"] = f"Failed to generate DOT visualization: {str(e)}"
    
    # Return visualization data
    return viz_data

def _format_output(
    extraction_result: Dict[str, Any],
    format: OutputFormat
) -> Dict[str, Any]:
    """Formats the extraction result in the requested output format."""
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    
    if format == OutputFormat.JSON:
        # Already in JSON format, just return
        return extraction_result
    
    elif format == OutputFormat.NETWORKX:
        # Convert to NetworkX graph
        if HAS_VISUALIZATION_LIBS:
            G = nx.DiGraph()
            
            # Add nodes with attributes
            for entity in entities:
                G.add_node(
                    entity["id"],
                    **{k: v for k, v in entity.items() if k != "id"}
                )
            
            # Add edges with attributes
            for rel in relationships:
                G.add_edge(
                    rel["source"],
                    rel["target"],
                    **{k: v for k, v in rel.items() if k not in ["source", "target", "id"]}
                )
            
            # Return formatted result with graph
            return {
                "entities": entities,
                "relationships": relationships,
                "graph": G,
                "metadata": extraction_result.get("metadata", {})
            }
        else:
            # NetworkX not available
            return {
                "error": "NetworkX output format requested but library not available.",
                "entities": entities,
                "relationships": relationships
            }
    
    elif format == OutputFormat.RDF:
        # Convert to RDF triples format
        try:
            triples = []
            
            # Create entity URIs
            entity_uris = {}
            for entity in entities:
                entity_id = entity["id"]
                entity_type = entity.get("type", "Thing")
                entity_name = entity.get("name", "").replace(" ", "_")
                
                # Create URI (simple format for example)
                uri = f"urn:entity:{entity_type}:{entity_name}"
                entity_uris[entity_id] = uri
                
                # Add entity type triple
                triples.append((uri, "rdf:type", f"ont:{entity_type}"))
                
                # Add entity name triple
                triples.append((uri, "rdfs:label", f'"{entity["name"]}"'))
                
                # Add entity attributes
                if "attributes" in entity:
                    for attr, value in entity["attributes"].items():
                        # Convert attribute name to predicate
                        predicate = f"ont:{attr.replace(' ', '_')}"
                        
                        # Format value based on type
                        if isinstance(value, (int, float)):
                            obj = str(value)
                        else:
                            obj = f'"{value}"'
                        
                        triples.append((uri, predicate, obj))
            
            # Create relationship triples
            for rel in relationships:
                source_id = rel["source"]
                target_id = rel["target"]
                rel_type = rel.get("type", "related_to")
                
                # Get entity URIs
                if source_id in entity_uris and target_id in entity_uris:
                    source_uri = entity_uris[source_id]
                    target_uri = entity_uris[target_id]
                    
                    # Create predicate (simple format)
                    predicate = f"ont:{rel_type.replace(' ', '_')}"
                    
                    # Add relationship triple
                    triples.append((source_uri, predicate, target_uri))
            
            # Return formatted result
            return {
                "entities": entities,
                "relationships": relationships,
                "rdf_triples": triples,
                "metadata": extraction_result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "error": f"Failed to format as RDF: {str(e)}",
                "entities": entities,
                "relationships": relationships
            }
    
    elif format == OutputFormat.CYTOSCAPE:
        # Convert to Cytoscape.js format
        try:
            cytoscape_elements = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            for entity in entities:
                entity_id = entity["id"]
                
                node = {
                    "data": {
                        "id": entity_id,
                        "label": entity.get("name", ""),
                        "type": entity.get("type", "")
                    }
                }
                
                # Add attributes if available
                if "attributes" in entity:
                    for attr, value in entity["attributes"].items():
                        node["data"][attr] = value
                
                # Add centrality if available
                if "centrality" in entity:
                    node["data"]["centrality"] = entity["centrality"]
                
                cytoscape_elements["nodes"].append(node)
            
            # Add edges
            for rel in relationships:
                source_id = rel["source"]
                target_id = rel["target"]
                
                edge = {
                    "data": {
                        "id": rel.get("id", f"e_{source_id}_{target_id}"),
                        "source": source_id,
                        "target": target_id,
                        "label": rel.get("type", ""),
                        "confidence": rel.get("confidence", 1.0)
                    }
                }
                
                cytoscape_elements["edges"].append(edge)
            
            # Return formatted result
            return {
                "entities": entities,
                "relationships": relationships,
                "cytoscape": cytoscape_elements,
                "metadata": extraction_result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "error": f"Failed to format as Cytoscape: {str(e)}",
                "entities": entities,
                "relationships": relationships
            }
    
    elif format == OutputFormat.D3:
        # Convert to D3.js force graph format
        try:
            d3_data = {
                "nodes": [],
                "links": []
            }
            
            # Add nodes
            for entity in entities:
                node = {
                    "id": entity["id"],
                    "name": entity.get("name", ""),
                    "group": entity.get("type", "")
                }
                
                # Add centrality if available
                if "centrality" in entity:
                    node["centrality"] = entity["centrality"]
                
                d3_data["nodes"].append(node)
            
            # Add links
            for rel in relationships:
                link = {
                    "source": rel["source"],
                    "target": rel["target"],
                    "type": rel.get("type", ""),
                    "value": rel.get("confidence", 1.0) * 10  # Scale for D3 link strength
                }
                
                d3_data["links"].append(link)
            
            # Return formatted result
            return {
                "entities": entities,
                "relationships": relationships,
                "d3": d3_data,
                "metadata": extraction_result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "error": f"Failed to format as D3: {str(e)}",
                "entities": entities,
                "relationships": relationships
            }
    
    elif format == OutputFormat.NEO4J:
        # Convert to Neo4j Cypher queries
        try:
            cypher_queries = []
            
            # Create entity nodes
            for entity in entities:
                entity_id = entity["id"]
                entity_label = entity.get("type", "Entity")
                
                # Prepare properties
                properties = {
                    "name": entity.get("name", ""),
                    "id": entity_id
                }
                
                # Add attributes if available
                if "attributes" in entity:
                    for attr, value in entity["attributes"].items():
                        properties[attr] = value
                
                # Format properties for Cypher
                props_str = ", ".join([
                    f"`{k}`: {json.dumps(v)}" for k, v in properties.items()
                ])
                
                # Create Cypher query for node
                query = f'CREATE (n:{entity_label} {{{props_str}}})'
                cypher_queries.append(query)
            
            # Create relationship edges
            for rel in relationships:
                source_id = rel["source"]
                target_id = rel["target"]
                rel_type = rel.get("type", "RELATED_TO")
                
                # Prepare properties
                properties = {
                    "confidence": rel.get("confidence", 1.0)
                }
                
                if "evidence" in rel:
                    properties["evidence"] = rel["evidence"]
                
                # Format properties for Cypher
                props_str = ", ".join([
                    f"`{k}`: {json.dumps(v)}" for k, v in properties.items()
                ])
                
                # Create Cypher query for relationship
                query = f'MATCH (a), (b) WHERE a.id = "{source_id}" AND b.id = "{target_id}" CREATE (a)-[r:{rel_type} {{{props_str}}}]->(b)'
                cypher_queries.append(query)
            
            # Return formatted result
            return {
                "entities": entities,
                "relationships": relationships,
                "neo4j_queries": cypher_queries,
                "metadata": extraction_result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "error": f"Failed to format as Neo4j: {str(e)}",
                "entities": entities,
                "relationships": relationships
            }
    
    else:
        # Unknown format, return default
        return extraction_result

def _create_query_interface(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a query interface for interacting with the extracted graph."""
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    
    if not entities or not relationships:
        return None
    
    # Define query interface functions
    query_interface = {}
    
    # Create NetworkX graph for querying
    if HAS_VISUALIZATION_LIBS:
        G = nx.DiGraph()
        
        # Add nodes
        for entity in entities:
            G.add_node(
                entity["id"],
                **{k: v for k, v in entity.items() if k != "id"}
            )
        
        # Add edges
        for rel in relationships:
            G.add_edge(
                rel["source"],
                rel["target"],
                **{k: v for k, v in rel.items() if k not in ["source", "target", "id"]}
            )
        
        # Define interface functions
        def find_entity(name=None, entity_type=None, attribute=None, attribute_value=None):
            """Finds entities matching the specified criteria."""
            results = []
            
            for entity in entities:
                matches = True
                
                if name and name.lower() not in entity.get("name", "").lower():
                    matches = False
                
                if entity_type and entity.get("type") != entity_type:
                    matches = False
                
                if attribute and attribute_value:
                    if "attributes" not in entity or entity.get("attributes", {}).get(attribute) != attribute_value:
                        matches = False
                
                if matches:
                    results.append(entity)
            
            return results
        
        def find_path(source_name, target_name):
            """Finds paths between entities with the given names."""
            # Find source and target entities
            source_entities = [e for e in entities if source_name.lower() in e.get("name", "").lower()]
            target_entities = [e for e in entities if target_name.lower() in e.get("name", "").lower()]
            
            if not source_entities or not target_entities:
                return {"error": "Source or target entity not found."}
            
            # Use the first matching entity of each
            source_entity = source_entities[0]
            target_entity = target_entities[0]
            
            source_id = source_entity["id"]
            target_id = target_entity["id"]
            
            # Find all simple paths between source and target
            try:
                paths = list(nx.all_simple_paths(G, source_id, target_id, cutoff=4))
            except nx.NetworkXError:
                paths = []
            
            # Format path results
            path_results = []
            
            for path in paths:
                path_entities = []
                path_relationships = []
                
                # Find entities in path
                for node_id in path:
                    entity = next((e for e in entities if e["id"] == node_id), None)
                    if entity:
                        path_entities.append(entity)
                
                # Find relationships between consecutive entities
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    
                    rel = next((r for r in relationships if r["source"] == source and r["target"] == target), None)
                    if rel:
                        path_relationships.append(rel)
                
                path_results.append({
                    "entities": path_entities,
                    "relationships": path_relationships,
                    "length": len(path) - 1
                })
            
            return {
                "source": source_entity,
                "target": target_entity,
                "paths": path_results
            }
        
        def get_connected_entities(entity_name, direction="both", max_distance=1):
            """Gets entities connected to the specified entity."""
            # Find the entity
            matching_entities = [e for e in entities if entity_name.lower() in e.get("name", "").lower()]
            
            if not matching_entities:
                return {"error": "Entity not found."}
            
            # Use the first matching entity
            entity = matching_entities[0]
            entity_id = entity["id"]
            
            # Find connected entities
            connected = []
            
            if direction in ["outgoing", "both"]:
                # Find targets of outgoing relationships
                outgoing = [
                    (r, next((e for e in entities if e["id"] == r["target"]), None))
                    for r in relationships 
                    if r["source"] == entity_id
                ]
                
                connected.extend([
                    {
                        "entity": target_entity,
                        "relationship": rel,
                        "direction": "outgoing"
                    }
                    for rel, target_entity in outgoing
                    if target_entity
                ])
            
            if direction in ["incoming", "both"]:
                # Find sources of incoming relationships
                incoming = [
                    (r, next((e for e in entities if e["id"] == r["source"]), None))
                    for r in relationships 
                    if r["target"] == entity_id
                ]
                
                connected.extend([
                    {
                        "entity": source_entity,
                        "relationship": rel,
                        "direction": "incoming"
                    }
                    for rel, source_entity in incoming
                    if source_entity
                ])
            
            return {
                "entity": entity,
                "connected": connected
            }
        
        # Add functions to interface
        query_interface["find_entity"] = find_entity
        query_interface["find_path"] = find_path
        query_interface["get_connected_entities"] = get_connected_entities
        
        # Add descriptions
        interface_descriptions = {
            "find_entity": "Finds entities matching criteria like name, type, or attribute value.",
            "find_path": "Finds paths between two entities specified by name.",
            "get_connected_entities": "Gets entities directly connected to the specified entity."
        }
        
        query_interface["descriptions"] = interface_descriptions
    
    return query_interface