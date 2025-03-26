"""Embedding generation service for vector operations."""
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from llm_gateway.config import get_env
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Cache for embeddings to avoid repeated API calls."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".llm_gateway" / "embeddings"
            
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.cache = {}
        
        logger.info(
            f"Embeddings cache initialized (directory: {self.cache_dir})",
            emoji_key="cache"
        )
        
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate a cache key for text and model.
        
        Args:
            text: Text to embed
            model: Embedding model name
            
        Returns:
            Cache key
        """
        # Create a hash based on text and model
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{model}_{text_hash}"
        
    def _get_cache_file_path(self, key: str) -> Path:
        """Get cache file path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Cache file path
        """
        return self.cache_dir / f"{key}.npy"
        
    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get embedding from cache.
        
        Args:
            text: Text to embed
            model: Embedding model name
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._get_cache_key(text, model)
        
        # Check in-memory cache first
        if key in self.cache:
            return self.cache[key]
            
        # Check disk cache
        cache_file = self._get_cache_file_path(key)
        if cache_file.exists():
            try:
                embedding = np.load(str(cache_file))
                # Add to in-memory cache
                self.cache[key] = embedding
                return embedding
            except Exception as e:
                logger.error(
                    f"Failed to load embedding from cache: {str(e)}",
                    emoji_key="error"
                )
                
        return None
        
    def set(self, text: str, model: str, embedding: np.ndarray) -> None:
        """Set embedding in cache.
        
        Args:
            text: Text to embed
            model: Embedding model name
            embedding: Embedding vector
        """
        key = self._get_cache_key(text, model)
        
        # Add to in-memory cache
        self.cache[key] = embedding
        
        # Save to disk
        cache_file = self._get_cache_file_path(key)
        try:
            np.save(str(cache_file), embedding)
        except Exception as e:
            logger.error(
                f"Failed to save embedding to cache: {str(e)}",
                emoji_key="error"
            )
            
    def clear(self) -> None:
        """Clear the embedding cache."""
        # Clear in-memory cache
        self.cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(
                    f"Failed to delete cache file {cache_file}: {str(e)}",
                    emoji_key="error"
                )
                
        logger.info(
            "Embeddings cache cleared",
            emoji_key="cache"
        )


class EmbeddingService:
    """Service for generating embeddings from text."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key
            cache_dir: Directory to store embedding cache
        """
        # Only initialize once for singleton
        if self._initialized:
            return
            
        # Get API key
        self.api_key = api_key or get_env("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided for embedding service",
                emoji_key="warning"
            )
            
        # Initialize client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Create embedding cache
        self.cache = EmbeddingCache(cache_dir)
        
        # Set default model
        self.default_model = "text-embedding-3-small"
        
        # Track stats
        self.total_embeddings = 0
        self.cache_hits = 0
        self.api_calls = 0
        self.last_request_cost = 0.0
        
        self._initialized = True
        
        logger.info(
            "Embedding service initialized",
            emoji_key="provider"
        )
        
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """Get embedding for text.
        
        Args:
            text: Text to embed
            model: Embedding model name
            use_cache: Whether to use the cache
            
        Returns:
            Embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        model = model or self.default_model
        self.total_embeddings += 1
        
        # Check cache if enabled
        if use_cache:
            cached_embedding = self.cache.get(text, model)
            if cached_embedding is not None:
                self.cache_hits += 1
                self.last_request_cost = 0.0  # Cache hit, no cost
                return cached_embedding
        
        # Generate embedding via API
        start_time = time.time()
        self.api_calls += 1
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text,
                encoding_format="float"
            )
            
            # Extract embedding
            embedding = np.array(response.data[0].embedding)
            
            # Calculate and update cost
            # Cost calculation based on model and token count
            token_count = len(text) / 4  # Rough estimate of token count
            
            # Set cost based on model
            if model == "text-embedding-3-small":
                # $0.02 per 1M tokens
                self.last_request_cost = (token_count / 1_000_000) * 0.02
            elif model == "text-embedding-3-large":
                # $0.13 per 1M tokens
                self.last_request_cost = (token_count / 1_000_000) * 0.13
            elif model == "text-embedding-ada-002":
                # $0.10 per 1M tokens
                self.last_request_cost = (token_count / 1_000_000) * 0.10
            else:
                # Default cost estimate
                self.last_request_cost = (token_count / 1_000_000) * 0.05
            
            # Log success
            processing_time = time.time() - start_time
            logger.debug(
                f"Embedding generated ({len(embedding)} dimensions)",
                emoji_key="provider",
                model=model,
                time=processing_time
            )
            
            # Cache the result
            if use_cache:
                self.cache.set(text, model, embedding)
                
            return embedding
            
        except Exception as e:
            # Log error
            processing_time = time.time() - start_time
            logger.error(
                f"Failed to generate embedding: {str(e)}",
                emoji_key="error",
                model=model,
                time=processing_time
            )
            raise
            
    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        use_cache: bool = True,
        batch_size: int = 100,
        max_concurrency: int = 5
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            use_cache: Whether to use the cache
            batch_size: Maximum batch size for API calls
            max_concurrency: Maximum number of concurrent API calls
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        model = model or self.default_model
        self.total_embeddings += len(texts)
        
        # Create batch processing function
        async def process_batch(batch_texts):
            # Check cache for each text
            embeddings = []
            texts_to_embed = []
            cache_indices = []
            
            # Check cache first
            for i, text in enumerate(batch_texts):
                if use_cache:
                    cached_embedding = self.cache.get(text, model)
                    if cached_embedding is not None:
                        self.cache_hits += 1
                        embeddings.append((i, cached_embedding))
                    else:
                        texts_to_embed.append(text)
                        cache_indices.append(i)
                else:
                    texts_to_embed.append(text)
                    cache_indices.append(i)
            
            # If all embeddings were cached, return them
            if not texts_to_embed:
                # Sort embeddings by original index
                self.last_request_cost = 0.0  # All cache hits, no cost
                return [emb for _, emb in sorted(embeddings, key=lambda x: x[0])]
                
            # Generate embeddings for remaining texts
            start_time = time.time()
            self.api_calls += 1
            
            try:
                response = await self.client.embeddings.create(
                    model=model,
                    input=texts_to_embed,
                    encoding_format="float"
                )
                
                # Calculate and update cost
                # Estimate total tokens in batch
                total_tokens = sum(len(text) / 4 for text in texts_to_embed)
                
                # Set cost based on model
                if model == "text-embedding-3-small":
                    # $0.02 per 1M tokens
                    self.last_request_cost = (total_tokens / 1_000_000) * 0.02
                elif model == "text-embedding-3-large":
                    # $0.13 per 1M tokens
                    self.last_request_cost = (total_tokens / 1_000_000) * 0.13
                elif model == "text-embedding-ada-002":
                    # $0.10 per 1M tokens
                    self.last_request_cost = (total_tokens / 1_000_000) * 0.10
                else:
                    # Default cost estimate
                    self.last_request_cost = (total_tokens / 1_000_000) * 0.05
                
                # Process API response
                for i, embedding_data in enumerate(response.data):
                    original_idx = cache_indices[i]
                    embedding = np.array(embedding_data.embedding)
                    
                    # Cache the result
                    if use_cache:
                        self.cache.set(texts_to_embed[i], model, embedding)
                        
                    embeddings.append((original_idx, embedding))
                    
                # Log success
                processing_time = time.time() - start_time
                logger.debug(
                    f"Batch embedding generated ({len(texts_to_embed)} texts)",
                    emoji_key="provider",
                    model=model,
                    time=processing_time
                )
                
                # Sort embeddings by original index
                return [emb for _, emb in sorted(embeddings, key=lambda x: x[0])]
                
            except Exception as e:
                # Log error
                processing_time = time.time() - start_time
                logger.error(
                    f"Failed to generate batch embeddings: {str(e)}",
                    emoji_key="error",
                    model=model,
                    time=processing_time
                )
                raise
        
        # Split texts into batches
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await process_batch(batch)
                
        # Create tasks for each batch
        tasks = [process_with_semaphore(batch) for batch in batches]
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        embeddings = []
        for batch_result in batch_results:
            embeddings.extend(batch_result)
            
        return embeddings
        
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_embeddings": self.total_embeddings,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "cache_hit_ratio": self.cache_hits / max(1, self.total_embeddings),
            "default_model": self.default_model,
            "last_request_cost": self.last_request_cost,
        }
        
    async def find_similar(
        self,
        query: str,
        texts: List[str],
        model: Optional[str] = None,
        top_k: int = 3,
        similarity_threshold: float = 0.7,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Find texts similar to query.
        
        Args:
            query: Query text
            texts: List of texts to search
            model: Embedding model name
            top_k: Number of top matches to return
            similarity_threshold: Minimum similarity score
            use_cache: Whether to use the cache
            
        Returns:
            List of similar texts with similarity scores
            
        Raises:
            Exception: If similarity search fails
        """
        # Get embeddings
        query_embedding = await self.get_embedding(query, model, use_cache)
        text_embeddings = await self.get_embeddings(texts, model, use_cache)
        
        # Calculate similarities
        similarities = []
        for i, text_embedding in enumerate(text_embeddings):
            # Cosine similarity
            similarity = cosine_similarity(query_embedding, text_embedding)
            similarities.append((i, similarity, texts[i]))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and limit to top_k
        results = []
        for i, similarity, text in similarities[:top_k]:
            if similarity >= similarity_threshold:
                results.append({
                    "index": i,
                    "text": text,
                    "similarity": similarity,
                })
                
        return results
        
    def reset_stats(self) -> None:
        """Reset service statistics."""
        self.total_embeddings = 0
        self.cache_hits = 0
        self.api_calls = 0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    # Ensure vectors are normalized
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)


# Singleton instance getter
def get_embedding_service(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> EmbeddingService:
    """Get the embedding service singleton instance.
    
    Args:
        api_key: OpenAI API key (defaults to environment variable)
        cache_dir: Directory to store embedding cache
        
    Returns:
        EmbeddingService singleton instance
    """
    return EmbeddingService(api_key, cache_dir)