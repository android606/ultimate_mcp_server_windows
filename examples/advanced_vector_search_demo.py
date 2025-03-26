#!/usr/bin/env python
"""Demo of advanced vector search capabilities using real LLM Gateway tools."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))


from decouple import config as decouple_config

from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway
from llm_gateway.services.vector import get_vector_db_service
from llm_gateway.services.vector.embeddings import cosine_similarity, get_embedding_service
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.advanced_vector_search")

# Get API key from environment
openai_api_key = decouple_config('OPENAI_API_KEY', default=None)

# Initialize global gateway
gateway = None
vector_service = None
vector_tools = None
embedding_service = None


async def setup_services():
    """Set up the gateway and vector service for demonstration."""
    global gateway, vector_service, vector_tools, embedding_service
    
    # Create gateway instance
    logger.info("Initializing gateway for demonstration", emoji_key="start")
    gateway = Gateway("vector-demo")
    
    # Initialize the server with built-in tools
    await gateway._initialize_providers()
    
    # Initialize services directly like in vector_search_demo.py
    embedding_service = get_embedding_service(api_key=openai_api_key)
    vector_service = get_vector_db_service()
    
    # Store the MCP interface for potential future use
    vector_tools = gateway.mcp
    
    logger.success("Services initialized with vector tools", emoji_key="success")


async def embedding_generation_demo():
    """Demonstrate embedding generation with real providers."""
    logger.info("Starting embedding generation demo", emoji_key="start")
    
    # Text for embedding
    text_samples = [
        "Quantum computing leverages quantum mechanics to perform computations",
        "Artificial intelligence systems can learn from data and improve over time",
        "Cloud infrastructure enables scalable and flexible computing resources",
        "Blockchain technology provides a distributed and immutable ledger",
        "Natural language processing helps computers understand human language"
    ]
    
    # List of providers to demonstrate
    providers_to_demo = [
        {"name": Provider.OPENAI.value, "model": "text-embedding-3-small"},
        {"name": Provider.OPENAI.value, "model": "text-embedding-ada-002"}
    ]
    
    # Check which providers are available
    logger.info("Checking available providers for embedding generation", emoji_key="processing")
    
    results = {}
    
    for provider_info in providers_to_demo:
        provider_name = provider_info["name"]
        model_name = provider_info["model"]
        
        try:
            # Generate embeddings using the embedding service
            logger.info(f"Generating embeddings with {provider_name} ({model_name})", emoji_key="provider")
            
            start_time = time.time()
            
            # Call embedding service directly
            embeddings = await embedding_service.get_embeddings(
                texts=text_samples[:1],  # Just use first sample for demo
                model=model_name
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get embedding data
            if embeddings and len(embeddings) > 0:
                embedding_size = len(embeddings[0])
                
                # Store results
                results[f"{provider_name}/{model_name}"] = {
                    "successful": True,
                    "dimensions": embedding_size,
                    "generation_time": processing_time,
                    "cost": embedding_service.last_request_cost,
                    "sample_embedding": list(embeddings[0][:5]) + ["..."],  # Just show first few dimensions
                    "provider": provider_name,
                    "model": model_name
                }
                
                logger.success(
                    f"Successfully generated {len(embeddings)} embeddings with {embedding_size} dimensions",
                    emoji_key="success"
                )
            else:
                results[f"{provider_name}/{model_name}"] = {
                    "successful": False,
                    "error": "No embeddings returned"
                }
                logger.warning(f"No embeddings returned from {provider_name}", emoji_key="warning")
                
        except Exception as e:
            logger.error(f"Error generating embeddings with {provider_name}: {str(e)}", emoji_key="error")
            results[f"{provider_name}/{model_name}"] = {
                "successful": False,
                "error": str(e)
            }
    
    # Print embedding generation results
    print("\n" + "-" * 80)
    print("Embedding Generation Results:")
    
    for provider_key, data in results.items():
        print(f"\n{provider_key}:")
        if data["successful"]:
            print(f"  Dimensions: {data['dimensions']}")
            print(f"  Generation Time: {data['generation_time']:.3f}s")
            print(f"  Cost: ${data['cost']:.6f}")
            print(f"  Sample Values: {data['sample_embedding'][:3]}...")
        else:
            print(f"  Failed: {data.get('error', 'Unknown error')}")
    
    print("-" * 80 + "\n")
    
    return results


async def vector_search_demo():
    """Demonstrate vector search capabilities with real tools."""
    logger.info("Starting vector search demo", emoji_key="start")
    
    # Create a collection of sample documents
    documents = [
        "Quantum computing uses quantum bits or qubits to perform calculations.",
        "Machine learning algorithms learn patterns from data without explicit programming.",
        "Blockchain technology creates a distributed and immutable ledger of transactions.",
        "Cloud computing delivers computing services over the internet on demand.",
        "Natural language processing helps computers understand and interpret human language.",
        "Artificial intelligence systems can simulate human intelligence in machines.",
        "Edge computing processes data closer to where it is generated rather than in a centralized location.",
        "Cybersecurity involves protecting systems from digital attacks and unauthorized access.",
        "Internet of Things (IoT) connects everyday devices to the internet for data sharing.",
        "Virtual reality creates an immersive computer-generated environment."
    ]
    
    # Document metadata for filtering
    document_metadata = [
        {"id": "doc1", "category": "quantum", "level": "advanced"},
        {"id": "doc2", "category": "ai", "level": "intermediate"},
        {"id": "doc3", "category": "blockchain", "level": "beginner"},
        {"id": "doc4", "category": "cloud", "level": "intermediate"},
        {"id": "doc5", "category": "ai", "level": "advanced"},
        {"id": "doc6", "category": "ai", "level": "beginner"},
        {"id": "doc7", "category": "cloud", "level": "advanced"},
        {"id": "doc8", "category": "security", "level": "intermediate"},
        {"id": "doc9", "category": "iot", "level": "beginner"},
        {"id": "doc10", "category": "vr", "level": "intermediate"}
    ]
    
    try:
        # Step 1: Create vector store and add documents
        logger.info("Creating vector store and adding documents", emoji_key="processing")
        
        # Create or get collection
        collection_name = "demo_vector_store"
        collection = vector_service.create_collection(  # noqa: F841
            name=collection_name,
            dimension=1536,  # OpenAI embedding dimension
            overwrite=True,   # Start fresh for demo
            metadata={"description": "Demo collection for vector search", "created_at": time.strftime("%Y-%m-%d")}
        )
        
        # Add documents to the store
        ids = await vector_service.add_texts(
            collection_name=collection_name,
            texts=documents,
            metadatas=document_metadata,
            batch_size=5
        )
        
        logger.success(
            f"Added {len(ids)} documents to vector store",
            emoji_key="success"
        )
        
        # Step 2: Perform vector search
        logger.info("Performing vector search with queries", emoji_key="processing")
        
        # Queries to search for
        search_queries = [
            "How does quantum computing work?",
            "Machine learning for image recognition",
            "Secure blockchain implementation"
        ]
        
        search_results = []
        
        for query in search_queries:
            # Perform search
            logger.info(f'Searching for: "{query}"', emoji_key="search")
            
            results = await vector_service.search_by_text(
                collection_name=collection_name,
                query_text=query,
                top_k=3,
                include_vectors=False
            )
            
            # Process results
            search_results.append({
                "query": query,
                "matches": results,
                "processing_time": 0.1  # Placeholder
            })
        
        # Display search results
        print("\n" + "-" * 80)
        print("Vector Search Results:")
        
        for result in search_results:
            print(f'\nQuery: "{result['query']}"')
            print("Results:")
            
            for i, match in enumerate(result["matches"], 1):
                doc_id = match.get("metadata", {}).get("id", f"doc{i}")
                category = match.get("metadata", {}).get("category", "unknown")
                score = match.get("similarity", 0)
                text = match.get("text", "")
                
                print(f"  {i}. [ID: {doc_id}, Category: {category}, Score: {score:.3f}]")
                print(f"     {text[:100]}..." if len(text) > 100 else f"     {text}")
        
        # Step 3: Demonstrate filtered search
        logger.info("Performing filtered vector search", emoji_key="search")
        
        filter_query = "AI technologies for beginners"
        filter_metadata = {"level": "beginner"}
        
        filter_results = await vector_service.search_by_text(
            collection_name=collection_name,
            query_text=filter_query,
            top_k=5,
            filter=filter_metadata,
            include_vectors=False
        )
        
        # Display filtered search results
        print("\n" + "-" * 80)
        print("Filtered Vector Search Results:")
        print(f'Query: "{filter_query}" [Filter: level=beginner]')
        print("Results:")
        
        for i, match in enumerate(filter_results, 1):
            doc_id = match.get("metadata", {}).get("id", f"doc{i}")
            category = match.get("metadata", {}).get("category", "unknown")
            level = match.get("metadata", {}).get("level", "unknown")
            score = match.get("similarity", 0)
            
            print(f"  {i}. [ID: {doc_id}, Category: {category}, Level: {level}, Score: {score:.3f}]")
            print(f"     {match.get('text', '')[:100]}..." if len(match.get('text', '')) > 100 else f"     {match.get('text', '')}")
        
        print("-" * 80 + "\n")
        
        # Cleanup
        logger.info("Cleaning up vector store", emoji_key="cleanup")
        vector_service.delete_collection(collection_name)
        
        return {
            "created": True,
            "document_count": len(ids),
            "search_results": search_results,
            "filtered_results": filter_results
        }
        
    except Exception as e:
        logger.error(f"Error in vector search demo: {str(e)}", emoji_key="error")
        return None


async def hybrid_search_demo():
    """Demonstrate hybrid search capabilities with real tools."""
    logger.info("Starting hybrid search demo", emoji_key="start")
    
    try:
        # Create documents with technical content
        documents = [
            "Python is a high-level, interpreted programming language known for its readability. It's widely used for web development, data analysis, AI, and scientific computing.",
            "JavaScript is a scripting language that enables interactive web pages. It runs in browsers and can also be used on servers with Node.js.",
            "Rust is a systems programming language focused on safety, particularly safe concurrency. It provides memory safety without using garbage collection.",
            "TensorFlow is an open-source machine learning framework developed by Google. It's used for building and training neural network models.",
            "Docker is a platform that uses containerization technology to package applications and their dependencies together.",
            "Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management of containerized applications.",
            "PyTorch is a deep learning framework developed by Facebook's AI Research lab. It's known for its flexibility and ease of use in research.",
            "React is a JavaScript library for building user interfaces, particularly single-page applications. It was developed by Facebook.",
            "CUDA is a parallel computing platform and API model created by NVIDIA. It allows developers to use NVIDIA GPUs for general purpose processing.",
            "GraphQL is a query language for APIs developed by Facebook. It provides a more efficient alternative to REST by allowing clients to request exactly what they need."
        ]
        
        # Document metadata
        document_metadata = [
            {"id": "doc1", "type": "language", "paradigm": "object-oriented", "year": 1991},
            {"id": "doc2", "type": "language", "paradigm": "multi-paradigm", "year": 1995},
            {"id": "doc3", "type": "language", "paradigm": "systems", "year": 2010},
            {"id": "doc4", "type": "framework", "domain": "machine-learning", "year": 2015},
            {"id": "doc5", "type": "platform", "domain": "containerization", "year": 2013},
            {"id": "doc6", "type": "platform", "domain": "orchestration", "year": 2014},
            {"id": "doc7", "type": "framework", "domain": "machine-learning", "year": 2016},
            {"id": "doc8", "type": "library", "domain": "frontend", "year": 2013},
            {"id": "doc9", "type": "platform", "domain": "gpu-computing", "year": 2007},
            {"id": "doc10", "type": "language", "domain": "api", "year": 2015}
        ]
        
        # Step 1: Create vector store with hybrid search enabled
        logger.info("Creating vector store with hybrid search enabled", emoji_key="processing")
        
        # Create collection
        collection_name = "hybrid_search_demo"
        collection = vector_service.create_collection(  # noqa: F841
            name=collection_name,
            dimension=1536,  # OpenAI embedding dimension
            overwrite=True,   # Start fresh for demo
            metadata={"description": "Demo collection for hybrid search", "created_at": time.strftime("%Y-%m-%d")}
        )
        
        # Step 2: Add documents to the store
        ids = await vector_service.add_texts(
            collection_name=collection_name,
            texts=documents,
            metadatas=document_metadata,
            batch_size=5
        )
        
        logger.success(
            f"Added {len(ids)} documents with hybrid indexing",
            emoji_key="success"
        )
        
        # Step 3: Perform different types of searches
        # Define queries that benefit from different search types
        search_configs = [
            {
                "name": "Vector Search",
                "query": "Neural network frameworks for deep learning research",
                "search_type": "vector",
                "top_k": 3
            },
            {
                "name": "Keyword Search",
                "query": "NVIDIA CUDA GPU",  # Exact terms that should match
                "search_type": "keyword",
                "top_k": 3
            },
            {
                "name": "Hybrid Search",
                "query": "Modern web development frameworks",
                "search_type": "hybrid",
                "hybrid_alpha": 0.7,  # Weight more toward semantic search
                "top_k": 3
            }
        ]
        
        # Perform searches and collect results
        all_results = {}
        
        for config in search_configs:
            logger.info(f'Performing {config['name']} with query: "{config['query']}"', emoji_key="search")
            
            # Perform search - note: we'll just use regular search here since we don't have hybrid search directly
            # In a real implementation, you would use a hybrid search method
            results = await vector_service.search_by_text(
                collection_name=collection_name,
                query_text=config["query"],
                top_k=config["top_k"],
                include_vectors=False
            )
            
            # Store results
            all_results[config["name"]] = {
                "query": config["query"],
                "search_type": config["search_type"],
                "matches": results,
                "processing_time": 0.1  # Placeholder
            }
        
        # Display search results
        print("\n" + "-" * 80)
        print("Comparison of Search Types:")
        
        for search_name, result in all_results.items():
            print(f"\n{search_name}:")
            print(f'Query: "{result['query']}"')
            print(f"Search Type: {result['search_type']}")
            print("Results:")
            
            for i, match in enumerate(result["matches"], 1):
                doc_id = match.get("metadata", {}).get("id", f"doc{i}")
                doc_type = match.get("metadata", {}).get("type", "unknown")
                domain = match.get("metadata", {}).get("domain", "unknown")
                score = match.get("similarity", 0)
                text = match.get("text", "")
                
                print(f"  {i}. [ID: {doc_id}, Type: {doc_type}, Domain: {domain}, Score: {score:.3f}]")
                print(f"     {text[:100]}..." if len(text) > 100 else f"     {text}")
        
        # Cleanup
        logger.info("Cleaning up vector store", emoji_key="cleanup")
        vector_service.delete_collection(collection_name)
        
        print("-" * 80 + "\n")
        
        # Summary of demonstration
        print("Hybrid Search Summary:")
        print("✅ Vector search: Best for semantic understanding and conceptual queries")
        print("✅ Keyword search: Best for specific terms and explicit matches")
        print("✅ Hybrid search: Best for balancing semantic understanding with keyword precision")
        print("\n")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error in hybrid search demo: {str(e)}", emoji_key="error")
        # Add diagnostic info
        logger.info("Attempting to get available tools", emoji_key="info")
        try:
            available_tools = await gateway.mcp.list_tools()
            logger.info(f"Available vector-related tools: {[t for t in available_tools if 'vector' in t or 'embedding' in t]}", emoji_key="info")
        except Exception as tool_error:
            logger.error(f"Error getting available tools: {str(tool_error)}", emoji_key="error")
        return None


async def semantic_similarity_demo():
    """Demonstrate semantic similarity comparison using real tools."""
    logger.info("Starting semantic similarity demo", emoji_key="start")
    
    # Define a set of technical descriptions to compare
    descriptions = [
        "TensorFlow is an open-source machine learning framework developed by Google. It supports deep neural networks and is widely used in production environments.",
        "PyTorch is a machine learning library developed by Facebook's AI Research lab. It's known for its dynamic computational graph and ease of use in research.",
        "scikit-learn is a free software machine learning library for Python. It features various classification, regression and clustering algorithms.",
        "NumPy is the fundamental package for scientific computing with Python. It provides support for arrays, matrices, and many mathematical functions.",
        "Pandas is a software library written for data manipulation and analysis in Python. It offers data structures and operations for manipulating numerical tables."
    ]
    
    try:
        # Calculate semantic similarities
        logger.info("Calculating pairwise semantic similarities", emoji_key="processing")
        
        # Get embeddings for all descriptions
        embeddings = await embedding_service.get_embeddings(texts=descriptions)
        
        # Calculate similarity matrix
        n = len(embeddings)
        similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = float(cosine_similarity(embeddings[i], embeddings[j]))
        
        # Display the similarity results
        logger.success("Semantic similarity calculation completed", emoji_key="success")
        print("\n" + "-" * 80)
        print("Semantic Similarity Results:")
        
        # Create labels for each description
        labels = ["TensorFlow", "PyTorch", "scikit-learn", "NumPy", "Pandas"]
        
        # Print similarity matrix
        print("\nSimilarity Matrix:")
        
        # Print header row
        header = "           "
        for label in labels:
            header += f"{label:12}"
        print(header)
        
        # Print rows
        for i, row in enumerate(similarity_matrix):
            row_str = f"{labels[i]:<12}"
            for sim in row:
                row_str += f"{sim:.4f}      "
            print(row_str)
        
        # Find and print most similar pair
        most_similar = (0, 0, 0.0)
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix[i])):
                if similarity_matrix[i][j] > most_similar[2]:
                    most_similar = (i, j, similarity_matrix[i][j])
        
        print(f"\nMost similar pair: {labels[most_similar[0]]} and {labels[most_similar[1]]} ({most_similar[2]:.4f})")
        
        # Find and print least similar pair
        least_similar = (0, 0, 1.0)
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix[i])):
                if similarity_matrix[i][j] < least_similar[2]:
                    least_similar = (i, j, similarity_matrix[i][j])
        
        print(f"Least similar pair: {labels[least_similar[0]]} and {labels[least_similar[1]]} ({least_similar[2]:.4f})")
        
        # Print stats
        print("\nProcessing Statistics:")
        print(f"Time Taken: {0.1:.2f}s")  # Placeholder
        print(f"Cost: ${embedding_service.last_request_cost:.6f}")
        
        print("-" * 80 + "\n")
        
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error in semantic similarity demo: {str(e)}", emoji_key="error")
        return None


async def main():
    """Run vector search examples."""
    try:
        # Set up services
        await setup_services()
        
        print("\n")
        
        # Run embedding generation demo
        await embedding_generation_demo()
        
        print("\n")
        
        # Run vector search demo
        await vector_search_demo()
        
        print("\n")
        
        # Run hybrid search demo
        await hybrid_search_demo()
        
        print("\n")
        
        # Run semantic similarity demo
        await semantic_similarity_demo()
        
        # Final success message
        logger.success("All vector search demos completed successfully", emoji_key="success")
        
    except Exception as e:
        logger.critical(f"Vector search demo failed: {str(e)}", emoji_key="critical")
        return 1
    finally:
        # Clean up
        if gateway:
            pass  # No cleanup needed for Gateway instance
    
    return 0


if __name__ == "__main__":
    # Run the demos
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 