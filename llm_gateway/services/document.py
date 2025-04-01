"""Document processing service for chunking and analyzing text documents."""
import re
from typing import List

from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Service for processing documents."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(DocumentProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the document processor."""
        # Only initialize once for singleton
        if getattr(self, "_initialized", False):
            return
            
        logger.info("Document processor initialized", extra={"emoji_key": "success"})
        self._initialized = True
    
    async def chunk_document(
        self,
        document: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        method: str = "token"
    ) -> List[str]:
        """Split a document into chunks.
        
        Args:
            document: Document text
            chunk_size: Size of chunks (approximate)
            chunk_overlap: Overlap between chunks
            method: Chunking method (token, sentence, semantic)
            
        Returns:
            List of document chunks
        """
        if not document:
            return []
            
        logger.debug(
            f"Chunking document using method '{method}' (size: {chunk_size}, overlap: {chunk_overlap})",
            extra={"emoji_key": "processing"}
        )
        
        if method == "semantic":
            return await self._chunk_semantic(document, chunk_size, chunk_overlap)
        elif method == "sentence":
            return await self._chunk_by_sentence(document, chunk_size, chunk_overlap)
        else:
            # Default to token-based chunking
            return await self._chunk_by_tokens(document, chunk_size, chunk_overlap)
    
    async def _chunk_by_tokens(
        self,
        document: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Split document into chunks by approximate token count.
        
        Args:
            document: Document text
            chunk_size: Size of chunks (approximate tokens)
            chunk_overlap: Overlap between chunks (approximate tokens)
            
        Returns:
            List of document chunks
        """
        # Simple token estimation (split by whitespace)
        words = document.split()
        
        # No words, return empty list
        if not words:
            return []
            
        # Simple chunking
        chunks = []
        start = 0
        
        while start < len(words):
            # Calculate end position with potential overlap
            end = min(start + chunk_size, len(words))
            
            # Create chunk
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - chunk_overlap
            
            # Avoid getting stuck at the end
            if start >= len(words) - chunk_overlap:
                break
        
        logger.debug(
            f"Split document into {len(chunks)} chunks by token",
            extra={"emoji_key": "processing"}
        )
        
        return chunks
    
    async def _chunk_by_sentence(
        self,
        document: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Split document into chunks by sentences.
        
        Args:
            document: Document text
            chunk_size: Size of chunks (approximate tokens)
            chunk_overlap: Overlap between chunks (approximate tokens)
            
        Returns:
            List of document chunks
        """
        # Simple sentence splitting (not perfect but works for most cases)
        sentence_delimiters = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_delimiters, document)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # No sentences, return empty list
        if not sentences:
            return []
            
        # Chunk by sentences, trying to reach target size
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # Estimate size in tokens (approximate)
            sentence_size = len(sentence.split())
            
            # If adding this sentence exceeds the chunk size and we have content,
            # finalize the current chunk
            if current_chunk and current_size + sentence_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_chunk = []
                
                # Add sentences from the end of previous chunk for overlap
                for s in reversed(current_chunk):
                    s_size = len(s.split())
                    if overlap_size + s_size <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += s_size
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            # Add current sentence
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.debug(
            f"Split document into {len(chunks)} chunks by sentence",
            extra={"emoji_key": "processing"}
        )
        
        return chunks
    
    async def _chunk_semantic(
        self,
        document: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Split document into chunks by semantic meaning.
        
        Args:
            document: Document text
            chunk_size: Size of chunks (approximate tokens)
            chunk_overlap: Overlap between chunks (approximate tokens)
            
        Returns:
            List of document chunks
        """
        # For simplicity, this implementation is similar to sentence chunking
        # but with paragraph awareness
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
        
        # Fallback to sentence chunking if no clear paragraphs
        if len(paragraphs) <= 1:
            return await self._chunk_by_sentence(document, chunk_size, chunk_overlap)
        
        # Process each paragraph and create semantic chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            # Estimate size in tokens
            paragraph_size = len(paragraph.split())
            
            # If paragraph is very large, chunk it further
            if paragraph_size > chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Chunk large paragraph by sentences
                paragraph_chunks = await self._chunk_by_sentence(
                    paragraph, chunk_size, chunk_overlap
                )
                chunks.extend(paragraph_chunks)
                continue
            
            # If adding this paragraph exceeds the chunk size and we have content,
            # finalize the current chunk
            if current_chunk and current_size + paragraph_size > chunk_size:
                chunks.append("\n\n".join(current_chunk))
                
                # Start new chunk with last paragraph for better context
                if current_chunk[-1] != paragraph and len(current_chunk) > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[-1].split())
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add current paragraph
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        logger.debug(
            f"Split document into {len(chunks)} chunks semantically",
            extra={"emoji_key": "processing"}
        )
        
        return chunks


# Singleton instance
_document_processor = None


def get_document_processor() -> DocumentProcessor:
    """Get or create a document processor instance.
    
    Returns:
        DocumentProcessor: Document processor instance
    """
    global _document_processor
    
    if _document_processor is None:
        _document_processor = DocumentProcessor()
        
    return _document_processor 