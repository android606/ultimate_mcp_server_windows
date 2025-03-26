#!/usr/bin/env python
"""Vector database and semantic search demonstration for LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.vector import get_embedding_service, get_vector_db_service
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.vector_search")


async def demonstrate_vector_operations():
    """Demonstrate basic vector database operations."""
    logger.info("Starting vector database demonstration", emoji_key="start")
    
    # Get OpenAI API key for embeddings
    api_key = decouple_config("OPENAI_API_KEY", default=None)
    if not api_key:
        logger.critical("OpenAI API key is required for this demo", emoji_key="critical")
        return False
    
    # Get embedding service with API key
    embedding_service = get_embedding_service(api_key=api_key)
    
    # Get vector database service
    vector_db = get_vector_db_service()
    
    # Check if services are available
    logger.info(f"Vector database using ChromaDB: {vector_db.use_chromadb}", emoji_key="info")
    logger.info(f"Vector database storage: {vector_db.base_dir}", emoji_key="info")
    
    # Collection name for this demo
    collection_name = "semantic_search_demo"
    
    # Create or get collection
    logger.info(f"Creating collection: {collection_name}", emoji_key="vector")
    collection = vector_db.create_collection(  # noqa: F841
        name=collection_name,
        dimension=1536,  # OpenAI embedding dimension
        overwrite=True,   # Start fresh for demo
        metadata={"description": "Demo collection for semantic search", "created_at": time.strftime("%Y-%m-%d")}
    )
    
    # Sample documents for the demonstration
    documents = [
        "Machine learning is a field of study in artificial intelligence concerned with the development of algorithms that can learn from data.",
        "Natural language processing (NLP) is a subfield of linguistics and AI focused on interactions between computers and human language.",
        "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "Transformer models have revolutionized natural language processing with their self-attention mechanism.",
        "Vector databases store and retrieve high-dimensional vectors for tasks like semantic search and recommendation systems.",
        "Embeddings are numerical representations that capture semantic meanings and relationships between objects.",
        "Clustering algorithms group data points into clusters based on similarity metrics.",
        "Reinforcement learning is about how software agents should take actions to maximize cumulative reward.",
        "Knowledge graphs represent knowledge in graph form with entities as nodes and relationships as edges."
    ]
    
    # Add documents to the collection
    logger.info(f"Adding {len(documents)} documents to the collection", emoji_key="processing")
    start_time = time.time()
    
    # Generate IDs for the documents
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Add metadata for each document
    document_metadata = [
        {"domain": "machine_learning", "type": "concept", "word_count": len(doc.split())}
        for i, doc in enumerate(documents)
    ]
    
    try:
        # Add documents to vector DB with embeddings
        ids = await vector_db.add_texts(
            collection_name=collection_name,
            texts=documents,
            metadatas=document_metadata,
            ids=document_ids,
            embedding_model=None  # Use default model
        )
        
        elapsed_time = time.time() - start_time
        logger.success(
            f"Added {len(ids)} documents to collection in {elapsed_time:.2f}s",
            emoji_key="success",
            ids=ids
        )
        
        # Perform a simple text-based search
        query = "How do neural networks work?"
        logger.info(f"Performing semantic search with query: '{query}'", emoji_key="search")
        
        start_time = time.time()
        results = await vector_db.search_by_text(
            collection_name=collection_name,
            query_text=query,
            top_k=3,
            include_vectors=False
        )
        elapsed_time = time.time() - start_time
        
        # Display search results
        logger.success(
            f"Search completed in {elapsed_time:.4f}s, found {len(results)} results",
            emoji_key="success"
        )
        
        print("\n" + "-" * 80)
        print(f"Search query: '{query}'")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(f"\nResult #{i+1} (Similarity: {result['similarity']:.4f})")
            print(f"Text: {result['text']}")
            print(f"Metadata: {result['metadata']}")
        
        print("-" * 80 + "\n")
        
        # Demonstrate metadata filtering
        filter_query = "embeddings"
        logger.info(f"Searching with metadata filtering, query: '{filter_query}'", emoji_key="filter")
        
        # First create metadata filter
        domain_filter = {"domain": "machine_learning"}
        
        results = await vector_db.search_by_text(
            collection_name=collection_name,
            query_text=filter_query,
            top_k=2,
            filter=domain_filter
        )
        
        # Display filtered results
        logger.success(
            f"Filtered search completed, found {len(results)} results",
            emoji_key="success"
        )
        
        print("\n" + "-" * 80)
        print(f"Filtered search (domain: machine_learning) with query: '{filter_query}'")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(f"\nResult #{i+1} (Similarity: {result['similarity']:.4f})")
            print(f"Text: {result['text']}")
            print(f"Metadata: {result['metadata']}")
        
        print("-" * 80 + "\n")
        
        # Demonstrate direct embedding generation
        logger.info("Demonstrating direct embedding generation", emoji_key="vector")
        
        sample_text = "Semantic search helps find conceptually similar content."
        
        # Generate embedding
        start_time = time.time()
        embedding = await embedding_service.get_embedding(sample_text)
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"Generated embedding in {elapsed_time:.4f}s",
            emoji_key="vector",
            shape=f"{len(embedding)} dimensions"
        )
        
        # Show embedding truncated (first 5 dimensions)
        print(f"Sample embedding (first 5 dimensions): {embedding[:5]}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in vector operations: {str(e)}", emoji_key="error")
        return False


async def demonstrate_llm_with_vector_retrieval():
    """Demonstrate using vector search with LLM for retrieval-augmented generation."""
    logger.info("Starting retrieval-augmented generation demo", emoji_key="start")
    
    # Get OpenAI API key for embeddings and completions
    api_key = decouple_config("OPENAI_API_KEY", default=None)
    if not api_key:
        logger.critical("OpenAI API key is required for this demo", emoji_key="critical")
        return False
    
    # Get vector DB service with API key - explicitly disable ChromaDB
    vector_db = get_vector_db_service()
    
    # Get OpenAI provider for completions
    provider = get_provider(Provider.OPENAI.value, api_key=api_key)
    await provider.initialize()
    
    # Collection name (use the same as in previous function)
    collection_name = "semantic_search_demo"
    
    # Check if collection exists
    collection = vector_db.get_collection(collection_name)
    if collection is None:
        logger.warning(
            f"Collection '{collection_name}' not found. Please run the vector operations demo first.",
            emoji_key="warning"
        )
        return False
    
    # User question
    question = "What is the difference between deep learning and neural networks?"
    
    try:
        logger.info(f"User question: '{question}'", emoji_key="question")
        
        # Step 1: Perform vector search to find relevant context
        logger.info("Retrieving relevant documents from vector DB", emoji_key="search")
        
        start_time = time.time()
        search_results = await vector_db.search_by_text(
            collection_name=collection_name,
            query_text=question,
            top_k=3
        )
        search_time = time.time() - start_time
        
        # Extract relevant context from search results
        context_texts = [result["text"] for result in search_results]
        context = "\n\n".join(context_texts)
        
        logger.info(
            f"Retrieved {len(search_results)} documents in {search_time:.4f}s",
            emoji_key="success",
            similarity=f"{search_results[0]['similarity']:.4f} (best match)"
        )
        
        # Step 2: Create a prompt that includes the context and question
        prompt = f"""Answer the following question based on the provided context:

Context:
{context}

Question: {question}

Answer:"""
        
        # Step 3: Generate completion using the augmented prompt
        logger.info("Generating completion with retrieved context", emoji_key="processing")
        
        start_time = time.time()
        result = await provider.generate_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=200
        )
        completion_time = time.time() - start_time
        
        # Display results
        logger.success(
            "Retrieval-augmented generation completed",
            emoji_key="success",
            tokens=f"{result.input_tokens} input, {result.output_tokens} output",
            cost=result.cost,
            time=f"{completion_time:.2f}s"
        )
        
        print("\n" + "-" * 80)
        print("RETRIEVAL-AUGMENTED GENERATION RESULTS")
        print("-" * 80)
        print(f"Question: {question}")
        print("\nRetrieved Contexts:")
        
        for i, text in enumerate(context_texts):
            print(f"\n[{i+1}] {text}")
        
        print("\nGenerated Answer:")
        print(result.text.strip())
        print("-" * 80 + "\n")
        
        # Compare with direct completion (no retrieval)
        logger.info("Generating completion WITHOUT retrieved context (for comparison)", emoji_key="processing")
        
        direct_prompt = f"Question: {question}\n\nAnswer:"
        
        start_time = time.time()
        direct_result = await provider.generate_completion(
            prompt=direct_prompt,
            temperature=0.3,
            max_tokens=200
        )
        direct_time = time.time() - start_time
        
        # Display comparison
        logger.info(
            "Direct completion (no retrieval) completed",
            emoji_key="success",
            tokens=f"{direct_result.input_tokens} input, {direct_result.output_tokens} output",
            cost=direct_result.cost,
            time=f"{direct_time:.2f}s"
        )
        
        print("\n" + "-" * 80)
        print("DIRECT COMPLETION (NO RETRIEVAL)")
        print("-" * 80)
        print(f"Question: {question}")
        print("\nGenerated Answer (without context):")
        print(direct_result.text.strip())
        print("-" * 80 + "\n")
        
        # Show comparison
        total_rag_tokens = result.input_tokens + result.output_tokens
        total_direct_tokens = direct_result.input_tokens + direct_result.output_tokens
        
        logger.info(
            "Comparison summary:",
            emoji_key="info",
            rag={"tokens": total_rag_tokens, "cost": result.cost},
            direct={"tokens": total_direct_tokens, "cost": direct_result.cost}
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error in RAG demonstration: {str(e)}", emoji_key="error")
        return False


async def main():
    """Run vector database and search demonstration."""
    try:
        # First demonstrate basic vector operations
        vector_op_success = await demonstrate_vector_operations()
        
        # If vector operations failed, exit early
        if not vector_op_success:
            logger.warning("Vector operations demo failed, skipping RAG demo", emoji_key="warning")
            return 1
            
        print("\n" + "=" * 80 + "\n")
        
        # Then demonstrate retrieval-augmented generation
        rag_success = await demonstrate_llm_with_vector_retrieval()
        
        if not rag_success:
            logger.warning("Retrieval-augmented generation demo failed", emoji_key="warning")
            return 1
        
    except Exception as e:
        logger.critical(f"Vector demonstration failed: {str(e)}", emoji_key="critical")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.success("All demonstrations completed successfully!", emoji_key="success")
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 