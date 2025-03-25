"""Vector database service for semantic search."""
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from llm_gateway.services.vector.embeddings import cosine_similarity, get_embedding_service
from llm_gateway.utils import get_logger

logger = get_logger(__name__)

# Try to import chromadb, but don't fail if not available
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Try to import hnswlib, but don't fail if not available
try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


class VectorCollection:
    """A collection of vectors with metadata."""
    
    def __init__(
        self,
        name: str,
        dimension: int = 1536,
        similarity_metric: str = "cosine",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a vector collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            similarity_metric: Similarity metric (cosine, dot, or euclidean)
            metadata: Optional metadata for the collection
        """
        self.name = name
        self.dimension = dimension
        self.similarity_metric = similarity_metric
        self.metadata = metadata or {}
        
        # Initialize storage
        self.vectors = []
        self.ids = []
        self.metadatas = []
        
        # Create embedding service
        self.embedding_service = get_embedding_service()
        
        # Initialize search index
        self._init_search_index()
        
        logger.info(
            f"Vector collection '{name}' created ({dimension} dimensions)",
            emoji_key="vector"
        )
        
    def _init_search_index(self):
        """Initialize search index based on available libraries."""
        self.index_type = "numpy"  # Fallback
        self.index = None
        
        # Try to use HNSW for fast search if available
        if HNSWLIB_AVAILABLE:
            try:
                self.index = hnswlib.Index(space=self._get_hnswlib_space(), dim=self.dimension)
                self.index.init_index(max_elements=1000, ef_construction=200, M=16)
                self.index.set_ef(50)  # Search accuracy parameter
                self.index_type = "hnswlib"
                logger.debug(
                    f"Using HNSW index for collection '{self.name}'",
                    emoji_key="vector"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize HNSW index: {str(e)}. Falling back to numpy.",
                    emoji_key="warning"
                )
                self.index = None
    
    def _get_hnswlib_space(self) -> str:
        """Get HNSW space based on similarity metric.
        
        Returns:
            HNSW space name
        """
        if self.similarity_metric == "cosine":
            return "cosine"
        elif self.similarity_metric == "dot":
            return "ip"  # Inner product
        elif self.similarity_metric == "euclidean":
            return "l2"
        else:
            return "cosine"  # Default
    
    def add(
        self,
        vectors: Union[List[List[float]], np.ndarray],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add vectors to the collection.
        
        Args:
            vectors: Vectors to add
            ids: Optional IDs for the vectors (generated if not provided)
            metadatas: Optional metadata for each vector
            
        Returns:
            List of vector IDs
        """
        # Ensure vectors is a numpy array
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # Ensure metadatas is a list of dicts
        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]
        
        # Add to storage
        for _i, (vector, id, metadata) in enumerate(zip(vectors, ids, metadatas, strict=False)):
            self.vectors.append(vector)
            self.ids.append(id)
            self.metadatas.append(metadata)
        
        # Update index if using HNSW
        if self.index_type == "hnswlib" and self.index is not None:
            try:
                # Resize index if needed
                if len(self.vectors) > self.index.get_max_elements():
                    new_size = max(1000, len(self.vectors) * 2)
                    self.index.resize_index(new_size)
                
                # Add vectors to index
                start_idx = len(self.vectors) - len(vectors)
                for i, vector in enumerate(vectors):
                    self.index.add_items(vector, start_idx + i)
            except Exception as e:
                logger.error(
                    f"Failed to update HNSW index: {str(e)}",
                    emoji_key="error"
                )
                # Rebuild index
                self._rebuild_index()
        
        logger.debug(
            f"Added {len(vectors)} vectors to collection '{self.name}'",
            emoji_key="vector"
        )
        
        return ids
    
    def _rebuild_index(self):
        """Rebuild the search index from scratch."""
        if not HNSWLIB_AVAILABLE or not self.vectors:
            return
            
        try:
            # Re-initialize index
            self.index = hnswlib.Index(space=self._get_hnswlib_space(), dim=self.dimension)
            self.index.init_index(max_elements=max(1000, len(self.vectors) * 2), ef_construction=200, M=16)
            self.index.set_ef(50)
            
            # Add all vectors
            vectors_array = np.array(self.vectors, dtype=np.float32)
            self.index.add_items(vectors_array, np.arange(len(self.vectors)))
            
            logger.info(
                f"Rebuilt HNSW index for collection '{self.name}'",
                emoji_key="vector"
            )
        except Exception as e:
            logger.error(
                f"Failed to rebuild HNSW index: {str(e)}",
                emoji_key="error"
            )
            self.index = None
            self.index_type = "numpy"
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of results with scores and metadata
        """
        # Ensure query_vector is a numpy array
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Filter vectors based on metadata if needed
        if filter:
            filtered_indices = self._apply_filter(filter)
            if not filtered_indices:
                return []
        else:
            filtered_indices = list(range(len(self.vectors)))
        
        # Perform search based on index type
        if self.index_type == "hnswlib" and self.index is not None and not filter:
            # Use HNSW for fast search (only if no filter)
            try:
                start_time = time.time()
                labels, distances = self.index.knn_query(query_vector, k=min(top_k, len(self.vectors)))
                search_time = time.time() - start_time
                
                # Convert distances to similarities based on metric
                if self.similarity_metric == "cosine" or self.similarity_metric == "dot":
                    similarities = 1.0 - distances[0]  # Convert distance to similarity
                else:
                    similarities = 1.0 / (1.0 + distances[0])  # Convert distance to similarity
                
                # Format results
                results = []
                for _i, (label, similarity) in enumerate(zip(labels[0], similarities, strict=False)):
                    results.append({
                        "id": self.ids[label],
                        "similarity": float(similarity),
                        "metadata": self.metadatas[label],
                        "vector": self.vectors[label].tolist(),
                    })
                
                logger.debug(
                    f"HNSW search completed in {search_time:.6f}s",
                    emoji_key="vector",
                    time=search_time
                )
                
                return results
            except Exception as e:
                logger.error(
                    f"HNSW search failed: {str(e)}. Falling back to numpy.",
                    emoji_key="error"
                )
                # Fall back to numpy search
        
        # Numpy-based search (slower but always works)
        start_time = time.time()
        
        # Calculate similarities
        results = []
        for idx in filtered_indices:
            vector = self.vectors[idx]
            
            # Calculate similarity based on metric
            if self.similarity_metric == "cosine":
                similarity = cosine_similarity(query_vector, vector)
            elif self.similarity_metric == "dot":
                similarity = np.dot(query_vector, vector)
            elif self.similarity_metric == "euclidean":
                similarity = 1.0 / (1.0 + np.linalg.norm(query_vector - vector))
            else:
                similarity = cosine_similarity(query_vector, vector)
            
            results.append({
                "id": self.ids[idx],
                "similarity": float(similarity),
                "metadata": self.metadatas[idx],
                "vector": vector.tolist(),
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit to top_k
        results = results[:top_k]
        
        search_time = time.time() - start_time
        logger.debug(
            f"Numpy search completed in {search_time:.6f}s",
            emoji_key="vector",
            time=search_time
        )
        
        return results
    
    def _apply_filter(self, filter: Dict[str, Any]) -> List[int]:
        """Apply metadata filter to get matching indices.
        
        Args:
            filter: Metadata filter
            
        Returns:
            List of matching indices
        """
        filtered_indices = []
        for i, metadata in enumerate(self.metadatas):
            # Simple equality filter for now
            match = True
            for k, v in filter.items():
                if k not in metadata or metadata[k] != v:
                    match = False
                    break
            if match:
                filtered_indices.append(i)
        return filtered_indices
    
    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search by text query.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            filter: Optional metadata filter
            model: Embedding model name
            
        Returns:
            List of results with scores and metadata
        """
        # Get query embedding
        query_embedding = await self.embedding_service.get_embedding(query_text, model)
        
        # Search with the embedding
        return self.search(query_embedding, top_k, filter)
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete vectors from the collection.
        
        Args:
            ids: IDs of vectors to delete
            filter: Metadata filter for vectors to delete
            
        Returns:
            Number of vectors deleted
        """
        if ids is None and filter is None:
            return 0
        
        # Get indices to delete
        indices_to_delete = set()
        
        # Add indices by ID
        if ids:
            for i, id in enumerate(self.ids):
                if id in ids:
                    indices_to_delete.add(i)
        
        # Add indices by filter
        if filter:
            filtered_indices = self._apply_filter(filter)
            indices_to_delete.update(filtered_indices)
        
        # Delete vectors (in reverse order to avoid index issues)
        indices_to_delete = sorted(indices_to_delete, reverse=True)
        for idx in indices_to_delete:
            del self.vectors[idx]
            del self.ids[idx]
            del self.metadatas[idx]
        
        # Rebuild index if using HNSW
        if self.index_type == "hnswlib" and self.index is not None:
            self._rebuild_index()
        
        logger.info(
            f"Deleted {len(indices_to_delete)} vectors from collection '{self.name}'",
            emoji_key="vector"
        )
        
        return len(indices_to_delete)
    
    def save(self, directory: Union[str, Path]) -> bool:
        """Save collection to disk.
        
        Args:
            directory: Directory to save to
            
        Returns:
            True if successful
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save vectors
            vectors_array = np.array(self.vectors, dtype=np.float32)
            np.save(str(directory / "vectors.npy"), vectors_array)
            
            # Save IDs and metadata
            with open(directory / "data.json", "w") as f:
                json.dump({
                    "name": self.name,
                    "dimension": self.dimension,
                    "similarity_metric": self.similarity_metric,
                    "metadata": self.metadata,
                    "ids": self.ids,
                    "metadatas": self.metadatas,
                }, f)
            
            logger.info(
                f"Saved collection '{self.name}' to {directory}",
                emoji_key="vector"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to save collection: {str(e)}",
                emoji_key="error"
            )
            return False
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> "VectorCollection":
        """Load collection from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded collection
            
        Raises:
            FileNotFoundError: If collection files not found
            ValueError: If collection data is invalid
        """
        directory = Path(directory)
        
        # Check if files exist
        vectors_file = directory / "vectors.npy"
        data_file = directory / "data.json"
        
        if not vectors_file.exists() or not data_file.exists():
            raise FileNotFoundError(f"Collection files not found in {directory}")
        
        try:
            # Load vectors
            vectors_array = np.load(str(vectors_file))
            vectors = [vectors_array[i] for i in range(len(vectors_array))]
            
            # Load data
            with open(data_file, "r") as f:
                data = json.load(f)
            
            # Create collection
            collection = cls(
                name=data["name"],
                dimension=data["dimension"],
                similarity_metric=data["similarity_metric"],
                metadata=data["metadata"]
            )
            
            # Set data
            collection.ids = data["ids"]
            collection.metadatas = data["metadatas"]
            collection.vectors = vectors
            
            # Rebuild index
            collection._rebuild_index()
            
            logger.info(
                f"Loaded collection '{collection.name}' from {directory} ({len(vectors)} vectors)",
                emoji_key="vector"
            )
            
            return collection
        except Exception as e:
            logger.error(
                f"Failed to load collection: {str(e)}",
                emoji_key="error"
            )
            raise ValueError(f"Failed to load collection: {str(e)}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "name": self.name,
            "dimension": self.dimension,
            "similarity_metric": self.similarity_metric,
            "vectors_count": len(self.vectors),
            "index_type": self.index_type,
            "metadata": self.metadata,
        }
    
    def clear(self) -> None:
        """Clear all vectors from the collection."""
        self.vectors = []
        self.ids = []
        self.metadatas = []
        
        # Reset index
        self._init_search_index()
        
        logger.info(
            f"Cleared collection '{self.name}'",
            emoji_key="vector"
        )


class VectorDatabaseService:
    """Vector database service for semantic search."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(VectorDatabaseService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        use_chromadb: Optional[bool] = None
    ):
        """Initialize the vector database service.
        
        Args:
            base_dir: Base directory for storage
            use_chromadb: Whether to use ChromaDB (if available)
        """
        # Only initialize once for singleton
        if self._initialized:
            return
            
        # Set base directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path.home() / ".llm_gateway" / "vector_db"
            
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if ChromaDB should be used
        self.use_chromadb = use_chromadb if use_chromadb is not None else CHROMADB_AVAILABLE
        
        # Initialize ChromaDB client if used
        self.chroma_client = None
        if self.use_chromadb and CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.base_dir / "chromadb"),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info(
                    "Using ChromaDB for vector storage",
                    emoji_key="vector"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize ChromaDB: {str(e)}. Falling back to local storage.",
                    emoji_key="error"
                )
                self.use_chromadb = False
        else:
            self.use_chromadb = False
            
        # Collections
        self.collections = {}
        
        # Get embedding service
        self.embedding_service = get_embedding_service()
        
        self._initialized = True
        
        logger.info(
            f"Vector database service initialized (base_dir: {self.base_dir}, use_chromadb: {self.use_chromadb})",
            emoji_key="vector"
        )
    
    def create_collection(
        self,
        name: str,
        dimension: int = 1536,
        similarity_metric: str = "cosine",
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Union[VectorCollection, Any]:
        """Create a new collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            similarity_metric: Similarity metric (cosine, dot, or euclidean)
            metadata: Optional metadata for the collection
            overwrite: Whether to overwrite existing collection
            
        Returns:
            Created collection
            
        Raises:
            ValueError: If collection already exists and overwrite is False
        """
        # Check if collection already exists
        if name in self.collections and not overwrite:
            raise ValueError(f"Collection '{name}' already exists")
            
        # Create collection
        if self.use_chromadb and self.chroma_client is not None:
            # ChromaDB collection
            try:
                # Delete if exists and overwrite is True
                if overwrite and name in [c.name for c in self.chroma_client.list_collections()]:
                    self.chroma_client.delete_collection(name)
                
                # Create collection
                collection = self.chroma_client.create_collection(
                    name=name,
                    metadata=metadata or {}
                )
                
                logger.info(
                    f"Created ChromaDB collection '{name}'",
                    emoji_key="vector"
                )
                
                self.collections[name] = collection
                return collection
            except Exception as e:
                logger.error(
                    f"Failed to create ChromaDB collection: {str(e)}. Falling back to local storage.",
                    emoji_key="error"
                )
                # Fall back to local storage
        
        # Local storage collection
        collection = VectorCollection(
            name=name,
            dimension=dimension,
            similarity_metric=similarity_metric,
            metadata=metadata
        )
        
        self.collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> Optional[Union[VectorCollection, Any]]:
        """Get a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            Collection or None if not found
        """
        # Check if collection is already loaded
        if name in self.collections:
            return self.collections[name]
            
        # Try to load from disk
        if self.use_chromadb and self.chroma_client is not None:
            # Check if ChromaDB collection exists
            try:
                collections = self.chroma_client.list_collections()
                if name in [c.name for c in collections]:
                    collection = self.chroma_client.get_collection(name)
                    self.collections[name] = collection
                    return collection
            except Exception as e:
                logger.error(
                    f"Failed to get ChromaDB collection: {str(e)}",
                    emoji_key="error"
                )
        
        # Try to load local collection
        collection_dir = self.base_dir / "collections" / name
        if collection_dir.exists():
            try:
                collection = VectorCollection.load(collection_dir)
                self.collections[name] = collection
                return collection
            except Exception as e:
                logger.error(
                    f"Failed to load collection '{name}': {str(e)}",
                    emoji_key="error"
                )
        
        return None
    
    def list_collections(self) -> List[str]:
        """List all collection names.
        
        Returns:
            List of collection names
        """
        collection_names = set(self.collections.keys())
        
        # Add collections from ChromaDB
        if self.use_chromadb and self.chroma_client is not None:
            try:
                for collection in self.chroma_client.list_collections():
                    collection_names.add(collection.name)
            except Exception as e:
                logger.error(
                    f"Failed to list ChromaDB collections: {str(e)}",
                    emoji_key="error"
                )
        
        # Add collections from disk
        collections_dir = self.base_dir / "collections"
        if collections_dir.exists():
            for path in collections_dir.iterdir():
                if path.is_dir() and (path / "data.json").exists():
                    collection_names.add(path.name)
        
        return list(collection_names)
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful
        """
        # Remove from loaded collections
        if name in self.collections:
            del self.collections[name]
        
        # Delete from ChromaDB
        if self.use_chromadb and self.chroma_client is not None:
            try:
                self.chroma_client.delete_collection(name)
            except Exception as e:
                logger.error(
                    f"Failed to delete ChromaDB collection: {str(e)}",
                    emoji_key="error"
                )
        
        # Delete from disk
        collection_dir = self.base_dir / "collections" / name
        if collection_dir.exists():
            try:
                import shutil
                shutil.rmtree(collection_dir)
            except Exception as e:
                logger.error(
                    f"Failed to delete collection directory: {str(e)}",
                    emoji_key="error"
                )
                return False
        
        logger.info(
            f"Deleted collection '{name}'",
            emoji_key="vector"
        )
        
        return True
    
    async def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embedding_model: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Add texts to a collection.
        
        Args:
            collection_name: Collection name
            texts: Texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for the texts
            embedding_model: Embedding model name
            batch_size: Maximum batch size for embedding generation
            
        Returns:
            List of document IDs
            
        Raises:
            ValueError: If collection not found
        """
        # Get or create collection
        collection = self.get_collection(collection_name)
        if collection is None:
            collection = self.create_collection(collection_name)
        
        # Generate embeddings
        embeddings = await self.embedding_service.get_embeddings(
            texts=texts,
            model=embedding_model,
            batch_size=batch_size
        )
        
        # Add to collection
        if self.use_chromadb and isinstance(collection, chromadb.Collection):
            # ChromaDB collection
            try:
                # Generate IDs if not provided
                if ids is None:
                    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
                
                # Ensure metadatas is provided
                if metadatas is None:
                    metadatas = [{} for _ in range(len(texts))]
                
                # Add to ChromaDB collection
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(
                    f"Added {len(texts)} documents to ChromaDB collection '{collection_name}'",
                    emoji_key="vector"
                )
                
                return ids
            except Exception as e:
                logger.error(
                    f"Failed to add documents to ChromaDB collection: {str(e)}",
                    emoji_key="error"
                )
                raise
        else:
            # Local collection
            return collection.add(
                vectors=embeddings,
                ids=ids,
                metadatas=[{"text": text, **(meta or {})} for text, meta in zip(texts, metadatas or [{} for _ in range(len(texts))], strict=False)]
            )
    
    async def search_by_text(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[str] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """Search a collection by text query.
        
        Args:
            collection_name: Collection name
            query_text: Text query
            top_k: Number of results to return
            filter: Optional metadata filter
            embedding_model: Embedding model name
            include_vectors: Whether to include vectors in results
            
        Returns:
            List of search results
            
        Raises:
            ValueError: If collection not found
        """
        # Get collection
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(
            text=query_text,
            model=embedding_model
        )
        
        # Search collection
        if self.use_chromadb and isinstance(collection, chromadb.Collection):
            # ChromaDB collection
            try:
                # Convert filter to ChromaDB format if provided
                chroma_filter = self._convert_to_chroma_filter(filter) if filter else None
                
                # Search ChromaDB collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=chroma_filter,
                    include=["documents", "metadatas", "distances", "embeddings" if include_vectors else None]
                )
                
                # Format results
                formatted_results = []
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": 1.0 - float(results["distances"][0][i]),  # Convert distance to similarity
                    }
                    
                    if include_vectors and "embeddings" in results:
                        result["vector"] = results["embeddings"][0][i]
                        
                    formatted_results.append(result)
                
                return formatted_results
            except Exception as e:
                logger.error(
                    f"Failed to search ChromaDB collection: {str(e)}",
                    emoji_key="error"
                )
                raise
        else:
            # Local collection
            results = await collection.search_by_text(
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                model=embedding_model
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result["id"],
                    "text": result["metadata"].get("text", ""),
                    "metadata": {k: v for k, v in result["metadata"].items() if k != "text"},
                    "similarity": result["similarity"],
                }
                
                if include_vectors:
                    formatted_result["vector"] = result["vector"]
                    
                formatted_results.append(formatted_result)
                
            return formatted_results
    
    def _convert_to_chroma_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert filter to ChromaDB format.
        
        Args:
            filter: Filter dictionary
            
        Returns:
            ChromaDB-compatible filter
        """
        # Simple equality filter for now
        return filter
    
    def save_all_collections(self) -> int:
        """Save all local collections to disk.
        
        Returns:
            Number of collections saved
        """
        saved_count = 0
        collections_dir = self.base_dir / "collections"
        collections_dir.mkdir(parents=True, exist_ok=True)
        
        for name, collection in self.collections.items():
            if not self.use_chromadb or not isinstance(collection, chromadb.Collection):
                # Only save local collections
                collection_dir = collections_dir / name
                if collection.save(collection_dir):
                    saved_count += 1
        
        logger.info(
            f"Saved {saved_count} collections to disk",
            emoji_key="vector"
        )
        
        return saved_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary of statistics
        """
        collection_names = self.list_collections()
        collection_stats = {}
        
        for name in collection_names:
            collection = self.get_collection(name)
            if collection:
                if isinstance(collection, VectorCollection):
                    collection_stats[name] = collection.get_stats()
                else:
                    # ChromaDB collection
                    try:
                        count = collection.count()
                        collection_stats[name] = {
                            "name": name,
                            "vectors_count": count,
                            "type": "chromadb",
                        }
                    except Exception:
                        collection_stats[name] = {
                            "name": name,
                            "type": "chromadb",
                        }
        
        return {
            "collections_count": len(collection_names),
            "collections": collection_stats,
            "base_dir": str(self.base_dir),
            "using_chromadb": self.use_chromadb,
            "embedding_service": self.embedding_service.get_stats(),
        }


# Singleton instance getter
def get_vector_db_service(
    base_dir: Optional[Union[str, Path]] = None,
    use_chromadb: Optional[bool] = None
) -> VectorDatabaseService:
    """Get the vector database service singleton instance.
    
    Args:
        base_dir: Base directory for storage
        use_chromadb: Whether to use ChromaDB (if available)
        
    Returns:
        VectorDatabaseService singleton instance
    """
    return VectorDatabaseService(base_dir, use_chromadb)