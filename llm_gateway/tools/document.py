"""Document processing tools for LLM Gateway."""
import asyncio
import json
import re
import time
import traceback
from typing import Any, Dict, List, Optional

# Optional imports for semantic chunking
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _semantic_libs_available = True
except ImportError:
    _semantic_libs_available = False

# Optional import for accurate token counting
try:
    import tiktoken
    _tiktoken_available = True
except ImportError:
    _tiktoken_available = False

from llm_gateway.constants import Provider, TaskType
from llm_gateway.exceptions import ProviderError, ToolError, ToolInputError
from llm_gateway.services.cache import with_cache
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.tools.completion import generate_completion
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.document")

# --- Standalone Helper Functions (Moved out of class) --- 

def _improve_chunk_coherence(chunk_text: str, target_size: int) -> List[str]:
    """Improve the semantic coherence of a chunk..."""
    try:
        logger.debug(f"Improving chunk coherence: text_length={len(chunk_text)}, target_size={target_size}")
        if not chunk_text or len(chunk_text) < 50: 
            return [chunk_text]
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        sentences = [s for s in re.split(sentence_pattern, chunk_text) if s and s.strip()]
        logger.debug(f"Split into {len(sentences)} sentences")
        if len(sentences) <= 3: 
            return [chunk_text]
        
        groups = []
        current_group = [sentences[0]]
        transition_words = {"however", "nevertheless", "conversely", "meanwhile", 
                          "furthermore", "additionally", "consequently", "therefore",
                          "thus", "hence", "accordingly", "subsequently"}
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            if not sentence or not sentence.strip(): 
                continue
            words = sentence.split()
            if not words:
                current_group.append(sentence)
                continue
            sentence_start = ' '.join(words[:min(3, len(words))]).lower()
            has_transition = any(tw in sentence_start for tw in transition_words)
            is_short = len(words) < 5
            if has_transition or (not is_short and len(current_group) >= 3):
                groups.append(current_group)
                current_group = [sentence]
            else:
                current_group.append(sentence)
        if current_group: 
            groups.append(current_group)
        logger.debug(f"Created {len(groups)} sentence groups")
        if not groups: 
            return [chunk_text]
        
        improved_chunks = []
        current_combined = []
        current_size = 0
        for group in groups:
            group_sentences = [s.strip() for s in group if s and s.strip()]
            if not group_sentences: 
                continue
            group_text = ' '.join(group_sentences)
            group_size = len(group_text)
            if current_size + group_size > target_size and current_combined:
                combined_text = ' '.join(current_combined)
                if combined_text: 
                    improved_chunks.append(combined_text)
                current_combined = [group_text]
                current_size = group_size
            else:
                current_combined.append(group_text)
                current_size += group_size
        if current_combined:
            combined_text = ' '.join(current_combined)
            if combined_text: 
                improved_chunks.append(combined_text)
        logger.debug(f"Created {len(improved_chunks)} improved chunks")
        return improved_chunks if improved_chunks else [chunk_text]
    except Exception as e:
        logger.error(f"Error in _improve_chunk_coherence: {str(e)}\n{traceback.format_exc()}")
        return [chunk_text]

def _estimate_tokens(document: str) -> List[str]:
    """Estimate tokenization..."""
    try:
        if not document: 
            return []
        words = re.findall(r'\w+|[^\w\s]', document)
        if not words: 
             logger.warning("No words found in document for token estimation")
             return [document] # Return whole doc as one token if no words
        tokens = []
        for word in words:
            if not word: 
                continue
            if word.isdigit():
                for i in range(0, len(word), 3):
                    tokens.append(word[i:i+3])
            else:
                if len(word) <= 4:
                    tokens.append(word)
                else:
                    remaining = word
                    while remaining:
                        piece_size = min(4 + (len(remaining) % 3), len(remaining))
                        tokens.append(remaining[:piece_size])
                        remaining = remaining[piece_size:]
        if not tokens: 
             logger.warning("No tokens generated from words")
             return [document] # Return whole doc if tokenization failed
        logger.debug(f"Estimated {len(tokens)} tokens from {len(words)} words")
        return tokens
    except Exception as e:
        logger.error(f"Error in _estimate_tokens: {str(e)}\n{traceback.format_exc()}")
        return [document]

def _chunk_by_token_estimation(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk document by estimated token count..."""
    try:
        if not document: 
            return [document]
        estimated_tokens = _estimate_tokens(document) 
        if not estimated_tokens: 
            return [document]
        logger.debug(f"Estimated tokens: {len(estimated_tokens)} for document of length {len(document)}")
        chunks = []
        i = 0
        while i < len(estimated_tokens):
            chunk_end = min(i + chunk_size, len(estimated_tokens))
            if chunk_end < len(estimated_tokens):
                search_start = max(i, chunk_end - (chunk_size // 10))
                for j in range(chunk_end, search_start, -1):
                    # Check bounds before accessing estimated_tokens[j]
                    if j >= 0 and j < len(estimated_tokens) and estimated_tokens[j] in [".", "?", "!"]:
                        chunk_end = j + 1
                        break
            token_slice = estimated_tokens[i:chunk_end]
            if not token_slice: 
                i += max(1, chunk_size - chunk_overlap)
                continue
            current_chunk = ""
            for token in token_slice:
                if not token: 
                    continue
                if token.startswith("'") or token in [",", ".", ":", ";", "!", "?"] or not current_chunk:
                    current_chunk += token
                else:
                    current_chunk += " " + token
            current_chunk = current_chunk.strip()
            if not current_chunk:
                i += max(1, chunk_size - chunk_overlap)
                continue
            if len(current_chunk) > chunk_size // 4 and len(current_chunk) > 100:
                try:
                    improved_chunks = _improve_chunk_coherence(current_chunk, chunk_size) 
                    chunks.extend(improved_chunks)
                except Exception as e:
                    logger.error(f"Error improving chunk coherence: {str(e)}, using original chunk")
                    chunks.append(current_chunk)
            else:
                chunks.append(current_chunk)
            next_position = max(1, chunk_size - chunk_overlap)
            i += next_position
            logger.debug(f"Moving to next chunk position: {i} (advanced by {next_position})")
        if not chunks: 
             logger.warning("No chunks created via token estimation, returning original")
             return [document]
        logger.debug(f"Created {len(chunks)} chunks via token estimation")
        return chunks
        # --- End of Restored Logic --- 
    except Exception as e:
         logger.error(f"Error in _chunk_by_token_estimation: {str(e)}\n{traceback.format_exc()}")
         return [document]

def _chunk_by_tokens(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk document by token count using tiktoken, prioritizing sentence boundaries.

    Requires the 'tiktoken' library to be installed. Falls back to character chunking
    if tiktoken is unavailable.

    Args:
        document: Document text
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens (must be < chunk_size)

    Returns:
        List of document chunks
    """
    global _tiktoken_available
    if not _tiktoken_available:
        logger.warning("tiktoken library not found. Falling back to character chunking for _chunk_by_tokens.")
        # Fallback to character chunking if tiktoken is missing
        return _chunk_by_characters(document, chunk_size * 4, chunk_overlap * 4) # Estimate char size

    if chunk_overlap >= chunk_size:
         logger.warning(f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). Setting overlap to {chunk_size // 5}.")
         chunk_overlap = chunk_size // 5 # Ensure overlap is smaller

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(document)
        num_tokens = len(tokens)
        if num_tokens == 0:
            return []

        logger.debug(f"Tokenized document: {num_tokens} tokens")
        chunks = []
        start_index = 0

        # Define sentence ending punctuation tokens (using common token representations)
        try:
             end_punctuations = {'.', '?', '', '\n'}
             end_punctuation_tokens = {
                 t for p in end_punctuations for t in encoding.encode(p, allowed_special='all')
             }
             if '\n\n' in encoding._special_tokens:
                  end_punctuation_tokens.add(encoding._special_tokens['\n\n'])
             logger.debug(f"Sentence end tokens identified: {end_punctuation_tokens}")
        except Exception as enc_ex:
             logger.warning(f"Could not encode sentence end punctuations: {enc_ex}. Using basic set.")
             end_punctuation_tokens = {encoding.encode(p)[0] for p in ['.', '?', ''] if encoding.encode(p)}


        while start_index < num_tokens:
            target_end_index = min(start_index + chunk_size, num_tokens)
            best_end_index = target_end_index

            if target_end_index < num_tokens:
                look_back_limit = max(start_index, target_end_index - max(1, min(100, chunk_size // 5)))
                found_boundary = False
                for i in range(target_end_index - 1, look_back_limit - 1, -1):
                    if i < num_tokens and tokens[i] in end_punctuation_tokens:
                        best_end_index = i + 1
                        found_boundary = True
                        break
                if not found_boundary and target_end_index > 0 and tokens[target_end_index-1] in end_punctuation_tokens:
                     best_end_index = target_end_index

            current_chunk_tokens = tokens[start_index:best_end_index]
            if not current_chunk_tokens: 
                break
            current_chunk = encoding.decode(current_chunk_tokens).strip()

            if current_chunk: 
                chunks.append(current_chunk)

            next_start_index = max(start_index + 1, best_end_index - chunk_overlap)
            if next_start_index <= start_index:
                 next_start_index = start_index + 1
            start_index = next_start_index

        logger.debug(f"Created {len(chunks)} chunks via token chunking.")
        return chunks if chunks else ([document] if document else [])

    except Exception as e:
        logger.error(f"Error in _chunk_by_tokens: {str(e)}\n{traceback.format_exc()}")
        return _chunk_by_characters(document, chunk_size * 4, chunk_overlap * 4)

def _chunk_by_characters(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk document by character count, prioritizing sentence boundaries.

    Args:
        document: Document text
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters (must be < chunk_size)

    Returns:
        List of document chunks
    """
    try:
        chunks = []
        i = 0
        doc_len = len(document)
        if doc_len == 0:
            return []

        if chunk_overlap >= chunk_size:
             logger.warning(f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). Setting overlap to {chunk_size // 5}.")
             chunk_overlap = chunk_size // 5

        while i < doc_len:
            target_end_index = min(i + chunk_size, doc_len)
            best_end_index = target_end_index

            if target_end_index < doc_len:
                search_start = max(i, target_end_index - max(20, min(100, chunk_size // 5)))
                search_text = document[search_start:target_end_index]
                boundaries = ['. ', '? ', '! ', '\n\n', '\n- ', '\n* ']
                found_ends = [search_text.rfind(b) for b in boundaries if search_text.rfind(b) != -1]

                if found_ends:
                    last_boundary_pos_in_search = max(found_ends)
                    potential_end = search_start + last_boundary_pos_in_search + 2
                    if potential_end > i + (chunk_size // 10):
                         best_end_index = potential_end

            current_chunk = document[i:best_end_index].strip()
            if current_chunk:
                 chunks.append(current_chunk)

            next_start_index = max(i + 1, best_end_index - chunk_overlap)
            if next_start_index <= i:
                 next_start_index = i + 1
            i = next_start_index

        return chunks if chunks else ([document] if document else [])
    except Exception as e:
        logger.error(f"Error in _chunk_by_characters: {str(e)}\n{traceback.format_exc()}")
        return [document] if document else []

def _chunk_by_paragraphs(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk document by paragraphs, preserving paragraphs and adding overlap.

    Args:
        document: Document text
        chunk_size: Target maximum characters per chunk (soft limit).
        chunk_overlap: Number of *paragraphs* to overlap (0 or 1 recommended).

    Returns:
        List of document chunks.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', document) if p.strip()]
    if not paragraphs: 
        return [document] if document.strip() else []

    chunks = []
    current_chunk_paras = []
    current_len = 0
    join_str = "\n\n"
    join_len = len(join_str)
    para_overlap = 1 if chunk_overlap > 0 else 0
    last_paragraph_for_overlap = None

    for _i, para in enumerate(paragraphs):
        para_len = len(para)

        if para_len > chunk_size:
            if current_chunk_paras:
                chunks.append(join_str.join(current_chunk_paras))
            chunks.append(para)
            last_paragraph_for_overlap = para
            current_chunk_paras = []
            current_len = 0
            continue

        projected_len = current_len + para_len + (join_len if current_chunk_paras else 0)
        if current_chunk_paras and projected_len > chunk_size:
            chunks.append(join_str.join(current_chunk_paras))
            overlap_paras = [last_paragraph_for_overlap] if para_overlap > 0 and last_paragraph_for_overlap else []
            current_chunk_paras = overlap_paras + [para]
            current_len = sum(len(p) for p in current_chunk_paras) + (join_len * (len(current_chunk_paras) -1) if len(current_chunk_paras)>1 else 0)
        else:
            if not current_chunk_paras and para_overlap > 0 and last_paragraph_for_overlap and last_paragraph_for_overlap is not para :
                 current_chunk_paras.append(last_paragraph_for_overlap)
                 current_len += len(last_paragraph_for_overlap)
            current_chunk_paras.append(para)
            current_len += para_len + (join_len if len(current_chunk_paras) > 1 else 0)

        last_paragraph_for_overlap = para

    if current_chunk_paras:
        final_chunk_text = join_str.join(current_chunk_paras)
        if not chunks or chunks[-1] != final_chunk_text:
             chunks.append(final_chunk_text)

    return chunks if chunks else ([document] if document.strip() else [])

async def _chunk_by_semantic_boundaries(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk document by semantic boundaries using sentence embeddings. Includes overlap.

    Requires 'sentence-transformers' and 'scikit-learn' libraries.
    Falls back to paragraph chunking if libraries are unavailable or an error occurs.

    Args:
        document: Document text
        chunk_size: Target chunk size in characters (approximate).
        chunk_overlap: Number of *sentences* to overlap.

    Returns:
        List of document chunks.
    """
    global _semantic_libs_available
    if not _semantic_libs_available:
        logger.warning("Semantic libraries not found, falling back to paragraph chunking.")
        return _chunk_by_paragraphs(document, chunk_size, chunk_overlap)

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        sentences = [s.strip() for s in re.split(sentence_pattern, document) if s and s.strip()]
        if not sentences: 
            return [document] if document.strip() else []

        sentence_embeddings = model.encode(sentences)
        topic_boundaries = set()
        if len(sentence_embeddings) > 1:
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                emb1 = sentence_embeddings[i].reshape(1, -1)
                emb2 = sentence_embeddings[i+1].reshape(1, -1)
                sim = cosine_similarity(emb1, emb2)[0][0]
                similarities.append(sim)

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                variance = sum((s - avg_sim) ** 2 for s in similarities) / len(similarities)
                std_sim = variance ** 0.5 if variance >= 0 else 0 # Handle potential floating point issues
                threshold = avg_sim - (std_sim * 1.0) # Tune this multiplier
                topic_boundaries = {i for i, sim in enumerate(similarities) if sim < threshold}
                logger.debug(f"Semantic boundaries identified at sentence indices: {sorted(list(topic_boundaries))}")

        chunks = []
        current_chunk_sentences = []
        last_chunk_end_index = -1 # Keep track of where the previous chunk ended *before* overlap

        for i in range(len(sentences)):
            # Determine start index for this potential chunk (includes overlap)
            current_start_index = max(0, last_chunk_end_index + 1 - chunk_overlap) if chunks else 0
            current_chunk_sentences = sentences[current_start_index : i+1]
            current_chunk_text = " ".join(current_chunk_sentences)

            # Check if we should end the chunk
            should_end = False
            # Use boundary index i (end of sentence i, start of sentence i+1 is boundary)
            if i in topic_boundaries and len(current_chunk_text) > chunk_size // 2: # Boundary found, substantial content
                 should_end = True
            elif len(current_chunk_text) > chunk_size * 1.25: # Exceeded target size considerably
                 should_end = True
            elif i == len(sentences) - 1: # Last sentence
                 should_end = True

            if should_end:
                 # Finalize the text for this chunk (using sentences up to index i)
                 final_chunk_sentences = sentences[current_start_index : i+1]
                 final_chunk_text = " ".join(final_chunk_sentences).strip()

                 if final_chunk_text:
                     chunks.append(final_chunk_text)
                 last_chunk_end_index = i # Record where this chunk ended (sentence index)
                 # Next iteration will calculate overlap based on this

        logger.debug(f"Created {len(chunks)} semantic chunks.")
        return chunks if chunks else ([document] if document.strip() else [])

    except Exception as e:
        logger.error(f"Error in _chunk_by_semantic_boundaries: {e}\n{traceback.format_exc()}")
        return _chunk_by_paragraphs(document, chunk_size, chunk_overlap) # Fallback

# --- Standalone Tool Functions --- 

# Removed DocumentTools class and _register_tools method

# Removed @self.mcp.tool(), added @with_tool_metrics, @with_error_handling
@with_tool_metrics 
@with_error_handling
@with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
async def chunk_document(
    document: str,
    chunk_size: int = 1000,
    chunk_method: str = "semantic",
    chunk_overlap: int = 0
) -> List[str]:
    """Splits a large document into smaller, manageable text chunks.

    This tool is essential for processing documents that exceed the context window limits
    of LLMs. It breaks down text using various strategies to maintain coherence.

    Args:
        document: The full text of the document to be chunked.
        chunk_size: The target size for each chunk. Interpretation depends on `chunk_method`:
                    - For "token": Approximate number of tokens (requires tiktoken).
                    - For "character": Number of characters.
                    - For "paragraph": Soft maximum number of characters per chunk (keeps paragraphs whole).
                    - For "semantic": Approximate number of characters (uses sentence embeddings, requires sentence-transformers).
                    Defaults to 1000.
        chunk_method: The strategy for splitting the document:
                      - "token": Chunks by token count, attempting to respect sentence boundaries.
                                 Best for strict LLM input limits.
                      - "character": Chunks by character count, attempting to respect sentence boundaries.
                                   Useful when token counting is unavailable or less critical.
                      - "paragraph": Splits into paragraphs, then groups paragraphs into chunks near `chunk_size`.
                                   Preserves paragraph structure.
                      - "semantic": (Recommended for coherence) Splits into sentences and groups them based
                                    on semantic similarity using embeddings. Requires extra libraries.
                                    Overlap is measured in *sentences* for this method.
                      Defaults to "semantic".
        chunk_overlap: The number of units (tokens, characters, paragraphs, or sentences depending on `chunk_method`)
                       to overlap between consecutive chunks. Helps maintain context across chunks.
                       Set to 0 for no overlap. Default is 0.
                       Note: For "paragraph" and "semantic", overlap > 0 usually means 1 paragraph/sentence overlap.

    Returns:
        A list of strings, where each string is a chunk of the original document.

    Raises:
        ValueError: If an invalid `chunk_method` is provided.
        Exception: For unexpected errors during chunking.
    """
    start_time = time.time()
    
    # Super defensive check for document
    if not document or not isinstance(document, str):
        logger.warning(f"Invalid document provided: {type(document)}")
        empty_doc = "" if document is None else str(document)
        return [empty_doc] if empty_doc else []
    
    # Normalize parameters
    try:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            chunk_size = 1000
            logger.warning(f"Invalid chunk_size corrected to {chunk_size}")
            
        chunk_overlap = int(chunk_overlap)
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            chunk_overlap = min(100, chunk_size // 10)
            logger.warning(f"Invalid chunk_overlap corrected to {chunk_overlap}")
            
        if not isinstance(chunk_method, str):
            chunk_method = "semantic"
            logger.warning(f"Invalid chunk_method corrected to {chunk_method}")
    except (TypeError, ValueError) as e:
        logger.error(f"Parameter normalization error: {str(e)}")
        chunk_size = 1000
        chunk_overlap = 100
        chunk_method = "semantic"
    
    try:
        # Log input parameters for debugging
        logger.info(
            f"Starting chunking with method={chunk_method}, size={chunk_size}, overlap={chunk_overlap}, doc_length={len(document)}"
        )
        
        # Select chunking method - call standalone helpers
        if chunk_method == "token":
            chunks = _chunk_by_tokens(document, chunk_size, chunk_overlap)
        elif chunk_method == "character":
            chunks = _chunk_by_characters(document, chunk_size, chunk_overlap)
        elif chunk_method == "paragraph":
            chunks = _chunk_by_paragraphs(document, chunk_size, chunk_overlap)
        elif chunk_method == "semantic":
            chunks = await _chunk_by_semantic_boundaries(document, chunk_size, chunk_overlap)
        else:
            logger.warning(f"Unknown chunking method '{chunk_method}', defaulting to 'token'.")
            chunks = _chunk_by_tokens(document, chunk_size, chunk_overlap)
        
        # Verify chunks - make sure we got a list
        if not isinstance(chunks, list):
            logger.error(f"Chunking returned non-list type: {type(chunks)}")
            chunks = [document] if document else []
        
        # Final validation to ensure we have proper string chunks 
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk:  # Skip empty chunks
                continue
                
            if not isinstance(chunk, str):
                logger.warning(f"Chunk {i} is not a string: {type(chunk)}, converting")
                try:
                    chunk = str(chunk)
                except Exception:
                    logger.error(f"Could not convert chunk {i} to string, skipping")
                    continue
                    
            valid_chunks.append(chunk)
            
        # If we lost all chunks in validation, return the original document
        if not valid_chunks and document:
            logger.warning("No valid chunks after validation, using original document")
            valid_chunks = [document]
        
        # Log chunk info for debugging
        for i, chunk in enumerate(valid_chunks[:5]):  # Log first 5 chunks only
            logger.debug(f"Chunk {i}: length={len(chunk)}, starts_with={chunk[:30]}...")
        
        processing_time = time.time() - start_time
        
        # Log result
        logger.info(
            f"Chunked document into {len(valid_chunks)} chunks using {chunk_method} method",
            emoji_key=TaskType.SUMMARIZATION.value,
            time=processing_time
        )
        
        return valid_chunks
        
    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        logger.error(
            f"Error in chunk_document: {str(e)}\n{err_trace}",
            emoji_key="error"
        )
        # Return document as single chunk if there's an error
        if document:
            return [document]
        else:
            return []

@with_cache(ttl=24 * 60 * 60)
@with_tool_metrics
@with_error_handling
async def summarize_document(
    document: str,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    max_length: Optional[int] = None,
    summary_format: str = "paragraph",
    is_chunk: bool = False,
    chunk_index: Optional[int] = None,
    total_chunks: Optional[int] = None
) -> Dict[str, Any]:
    """Generates a concise summary of the provided text using an LLM.

    Useful for condensing large documents or text chunks into key points.
    Often used after `chunk_document` to summarize parts of a larger document.
    Results are cached for 24 hours based on the input document and parameters.

    Args:
        document: The text content to summarize.
        provider: The name of the LLM provider (e.g., "openai", "anthropic"). Defaults to "openai".
        model: The specific model ID (e.g., "openai/gpt-4.1-mini"). Uses provider default if None.
        max_length: (Optional) Target maximum length for the summary (in tokens).
        summary_format: (Optional) Desired format for the summary (e.g., "paragraph", "bullet points"). Default "paragraph".
        is_chunk: (Internal Use/Optional) Set to True if the input `document` is a chunk of a larger document.
        chunk_index: (Internal Use/Optional) If `is_chunk` is True, the index of this chunk (1-based).
        total_chunks: (Internal Use/Optional) If `is_chunk` is True, the total number of chunks.

    Returns:
        A dictionary containing the summary and metadata:
        {
            "summary": "The generated summary text...",
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": { ... },    # Token usage for the summarization task
            "cost": 0.000088,   # Estimated cost in USD
            "processing_time": 3.14, # Execution time in seconds
            "cached_result": false, # True if served from cache
            "success": true
        }

    Raises:
        ProviderError: If the provider is unavailable or the LLM request fails.
        ToolError: For other internal errors.
    """
    start_time = time.time()
    logger.info(f"Summarizing document (length: {len(document)}) with {provider}/{model or 'default'}")

    # Construct prompt
    prompt = f"Summarize the following text in {summary_format}" 
    if max_length:
        prompt += f" with a target maximum length of {max_length} tokens" 
    if is_chunk and chunk_index is not None and total_chunks is not None:
        prompt += f" (This is chunk {chunk_index} of {total_chunks}). Focus on the main points of this specific chunk."
    else:
        prompt += "."
    prompt += f"\n\nText:\n{document}"

    try:
        # Determine max_tokens based on max_length (approximate)
        # This is a rough estimate, might need refinement. Assume 1 word ~ 1.5 tokens
        max_tokens_limit = int(max_length * 1.5) if max_length else None 

        # Use the standardized generate_completion tool
        completion_result = await generate_completion(
            prompt=prompt,
            model=model,
            provider=provider,
            temperature=0.3, # Lower temperature for factual summary
            max_tokens=max_tokens_limit # Pass calculated max_tokens
        )
        
        processing_time = time.time() - start_time
        
        # Check if completion was successful
        if not completion_result.get("success", False):
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Document summarization failed: {error_message}", 
                provider=provider,
                model=model or "default"
            )
        
        summary_text = completion_result["text"]
        if not summary_text:
             raise ToolError(message="LLM response for summary was empty.")

        logger.success(
            f"Document summarized successfully with {provider}/{completion_result['model']}",
            emoji_key=TaskType.SUMMARIZATION.value,
            cost=completion_result["cost"],
            time=processing_time
        )
        return {
            "summary": summary_text,
            "model": completion_result["model"],
            "provider": provider,
            "tokens": completion_result["tokens"],
            "cost": completion_result["cost"],
            "processing_time": processing_time,
            # "cached_result": False, # Let cache decorator handle this
            "success": True
        }
            
    except Exception as e:
         error_model_detail = model or "default"
         if isinstance(e, ProviderError) and hasattr(e, 'model') and e.model:
             error_model_detail = e.model
         final_error_model = f"{provider}/{error_model_detail}"
         
         if isinstance(e, (ProviderError, ToolError, ToolInputError)):
             if isinstance(e, ProviderError) and not hasattr(e, 'model'): 
                 e.model = final_error_model
             raise 
         else:
             raise ProviderError(
                 f"Summarization failed for model '{final_error_model}': {str(e)}", 
                 provider=provider, 
                 model=final_error_model,
                 cause=e
             ) from e

@with_cache(ttl=24 * 60 * 60)
@with_tool_metrics
@with_error_handling
async def extract_entities(
    document: str,
    entity_types: List[str],
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """Extracts specific types of named entities (e.g., names, places, dates) from text using an LLM.

    Use this tool to identify and pull out structured information from unstructured text.
    Can operate on full documents or individual chunks.
    Results are cached for 24 hours.

    Args:
        document: The text content to extract entities from.
        entity_types: A list of strings specifying the types of entities to extract
                      (e.g., ["Person Name", "Organization", "Location", "Date", "Product Name", "Email Address"]).
                      Be specific for better results.
        provider: The name of the LLM provider. Defaults to "openai".
        model: The specific model ID. Uses provider default if None.
        output_format: (Optional) Desired format for the results ("list" or "json").
                       - "list": Returns entities grouped by type: `{"Person Name": ["Alice", "Bob"], ...}`
                       - "json": Attempts to return a structured JSON string (model dependent).
                       Defaults to "json".

    Returns:
        A dictionary containing the extracted entities and metadata:
        {
            "entities": { ... }, # Structure depends on output_format
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": { ... },
            "cost": 0.000095,
            "processing_time": 2.8,
            "cached_result": false,
            "success": true
        }

    Raises:
        ToolInputError: If `entity_types` is empty or invalid.
        ProviderError: If the provider/LLM fails.
        ToolError: For other internal errors.
    """
    start_time = time.time()
    if not entity_types or not isinstance(entity_types, list):
        raise ToolInputError("entity_types must be a non-empty list of strings.", param_name="entity_types", provided_value=entity_types)
    
    logger.info(f"Extracting entities ({', '.join(entity_types)}) from document (length: {len(document)}) with {provider}/{model or 'default'}")
    
    # --- Improved Prompt Construction --- 
    entities_str = ", ".join(entity_types)
    # Create a dynamic example based on the first 2-3 requested types
    example_entities = {}
    if "person" in entity_types:
         example_entities["person"] = ["Alice", "Dr. Smith"]
    if "organization" in entity_types:
         example_entities["organization"] = ["Example Corp"]
    if "location" in entity_types and "person" not in example_entities:
         # Add location only if we don't have person/org already for a concise example
         example_entities["location"] = ["Paris", "Main Street"]
    
    # Ensure at least one example type if possible
    if not example_entities and entity_types:
         example_entities[entity_types[0]] = ["Value1", "Value2"]
         
    example_json = json.dumps(example_entities, indent=2)

    prompt = f"""You are an expert entity extraction system.
Extract the following types of entities from the provided text: {entities_str}.
Respond ONLY with a valid JSON object. The keys in the JSON object MUST be the requested entity types ({entities_str}). The value for each key MUST be a list of strings, where each string is an extracted entity of that type found in the text.
Only include keys for entity types that were actually found in the text. If no entities of any of the requested types are found, respond ONLY with an empty JSON object: {{}}.

Example Output Format:
```json
{example_json}
```

Text to analyze:
{document}

JSON Output:""" # Added priming token
    # --- End Improved Prompt Construction --- 

    try:
        # Use the standardized generate_completion tool
        completion_result = await generate_completion(
            prompt=prompt,
            model=model,
            provider=provider,
            temperature=0.1, # Keep temperature low for factual task
            additional_params={
                "response_format": {"type": "json_object"} if provider == Provider.OPENAI.value else None
            }
        )
        
        processing_time = time.time() - start_time
        
        # Check if completion was successful
        if not completion_result.get("success", False):
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Entity extraction failed: {error_message}", 
                provider=provider,
                model=model or "default"
            )
        
        extracted_text = completion_result["text"]
        logger.info(f"Raw LLM output for entity extraction:\n{extracted_text}") 
        
        if not extracted_text:
             raise ToolError(message="LLM response for entity extraction was empty.")

        entities_data = {} # Initialize as empty dict
        
        # Attempt to parse based on requested format (defaulting to JSON)
        try:
            extracted_text_cleaned = re.sub(r"^\s*```json\n?|\n?```\s*$", "", extracted_text.strip())
            parsed_data = json.loads(extracted_text_cleaned)
            
            logger.info(f"Parsed entity data (before filtering): {parsed_data}")
            
            # Ensure it's a dict
            if not isinstance(parsed_data, dict):
                 raise ValueError("LLM did not return a JSON object.")
                 
            # --- RE-ENABLE FILTERING --- 
            entities_data = {k: v for k, v in parsed_data.items() if k in entity_types and v}
            # --- END RE-ENABLE --- 
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM output as JSON for entity extraction: {e}. Raw text: {extracted_text}")
            raise ToolError(message=f"Failed to parse LLM output as JSON: {e}. Output: {extracted_text}", error_code="PARSING_ERROR") from e
        except Exception as parse_err:
             logger.warning(f"Failed to parse LLM output for entity extraction ({output_format}): {parse_err}. Raw text: {extracted_text}")
             raise ToolError(message=f"Failed to parse LLM output ({output_format}): {parse_err}. Output: {extracted_text}", error_code="PARSING_ERROR") from parse_err

        logger.info(f"Final entities data (after filtering): {entities_data}")
        
        logger.success(
            f"Entities extracted successfully with {provider}/{completion_result['model']}",
            emoji_key=TaskType.EXTRACTION.value,
            cost=completion_result["cost"],
            time=processing_time
        )
        return {
            "entities": entities_data,
            "model": completion_result["model"],
            "provider": provider,
            "tokens": completion_result["tokens"],
            "cost": completion_result["cost"],
            "processing_time": processing_time,
            # "cached_result": False, # Let cache decorator handle this
            "success": True
        }
            
    except Exception as e:
        error_model_detail = model or "default"
        if isinstance(e, ProviderError) and hasattr(e, 'model') and e.model:
             error_model_detail = e.model
        final_error_model = f"{provider}/{error_model_detail}"
         
        if isinstance(e, (ProviderError, ToolError, ToolInputError)):
             if isinstance(e, ProviderError) and not hasattr(e, 'model'): 
                 e.model = final_error_model
             raise 
        else:
             raise ProviderError(
                 f"Entity extraction failed for model '{final_error_model}': {str(e)}", 
                 provider=provider, 
                 model=final_error_model,
                 cause=e
             ) from e

@with_tool_metrics
@with_error_handling
async def generate_qa_pairs(
    document: str,
    num_pairs: int = 5,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    temperature: float = 0.5 # Keep temperature as param
) -> Dict[str, Any]:
    """Generates question-answer pairs based on the provided text content using an LLM.

    Useful for creating study materials, verifying comprehension, or generating training data
    for question-answering systems. Can operate on full documents or chunks.

    Args:
        document: The text content to generate Q&A pairs from.
        num_pairs: The desired number of question-answer pairs to generate. Default 5.
        provider: The name of the LLM provider. Defaults to "openai".
        model: The specific model ID. Uses provider default if None.
        temperature: The temperature for generating the response. Default 0.5.

    Returns:
        A dictionary containing the generated Q&A pairs and metadata:
        {
            "qa_pairs": [
                {"question": "What is...?", "answer": "It is..."},
                {"question": "Why did...?", "answer": "Because..."},
                ...
            ],
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": { ... },
            "cost": 0.000110,
            "processing_time": 4.5,
            "success": true
        }

    Raises:
        ProviderError: If the provider/LLM fails.
        ToolError: For other internal errors, including failure to parse the LLM output.
    """
    start_time = time.time()
    logger.info(f"Generating {num_pairs} Q&A pairs from document (length: {len(document)}) with {provider}/{model or 'default'}")
    
    # Construct prompt
    prompt = f"Generate exactly {num_pairs} question and answer pairs based on the following text. " \
             f"Format the output STRICTLY as a JSON list of objects, where each object has a 'question' key and an 'answer' key. " \
             f"Ensure the questions are relevant to the main topics and the answers are accurate according to the text.\\n\\n" \
             f"Text:\\n{document}"

    try:
        # Use the standardized generate_completion tool
        completion_result = await generate_completion(
            prompt=prompt,
            model=model,
            provider=provider,
            temperature=temperature # Use passed temperature
        )
        
        processing_time = time.time() - start_time
        
        # Check if completion was successful
        if not completion_result.get("success", False):
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"QA generation failed: {error_message}", 
                provider=provider,
                model=model or "default"
            )
        
        qa_text = completion_result["text"]
        if not qa_text:
             raise ToolError(message="LLM response for Q&A generation was empty.") # Use message kwarg
             
        qa_pairs_data = []
        try:
            # IMPROVED Regex to handle optional whitespace around fences
            qa_text = re.sub(r"^\s*```json\n?|\n?```\s*$", "", qa_text.strip())
            qa_pairs_data = json.loads(qa_text)
            if not isinstance(qa_pairs_data, list):
                raise ValueError("LLM did not return a JSON list.")
            # Basic validation of list items
            validated_pairs = []
            for item in qa_pairs_data:
                 if isinstance(item, dict) and 'question' in item and 'answer' in item:
                      validated_pairs.append({
                           "question": str(item['question']).strip(), 
                           "answer": str(item['answer']).strip()
                      })
                 else:
                      logger.warning(f"Skipping invalid Q&A item format: {item}")
            
            if not validated_pairs:
                 raise ValueError("No valid Q&A objects found in the list.")
                 
            qa_pairs_data = validated_pairs # Use only validated pairs
                          
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Q&A pairs from LLM output: {e}. Raw output:\n{qa_text}")
            raise ToolError(message=f"Failed to parse generated Q&A pairs from the LLM response: {e}. Output: {qa_text}", error_code="PARSING_ERROR") from e

        logger.success(
            f"Q&A pairs generated successfully with {provider}/{completion_result['model']}",
            emoji_key=TaskType.QA.value,
            cost=completion_result["cost"],
            time=processing_time,
            pairs_generated=len(qa_pairs_data)
        )
        return {
            "qa_pairs": qa_pairs_data,
            "model": completion_result["model"],
            "provider": provider,
            "tokens": completion_result["tokens"],
            "cost": completion_result["cost"],
            "processing_time": processing_time,
            "success": True
        }

    except Exception as e:
        error_model_detail = model or "default"
        if isinstance(e, ProviderError) and hasattr(e, 'model') and e.model:
             error_model_detail = e.model
        final_error_model = f"{provider}/{error_model_detail}"

        if isinstance(e, (ProviderError, ToolError, ToolInputError)):
             if isinstance(e, ProviderError) and not hasattr(e, 'model'): 
                 e.model = final_error_model
             raise 
        else:
             raise ProviderError(
                 f"Q&A generation failed for model '{final_error_model}': {str(e)}", 
                 provider=provider, 
                 model=final_error_model,
                 cause=e
             ) from e

@with_tool_metrics
@with_error_handling
async def process_document_batch(
    documents: List[str],
    operations: List[Dict[str, Any]],
    max_concurrency: int = 5
) -> List[Dict[str, Any]]:
    """Processes a batch of documents through a sequence of operations (e.g., chunk, summarize).

    This tool orchestrates multiple document processing steps (like chunking followed by summarizing)
    across a list of documents, running operations concurrently where possible.

    NOTE: This is a higher-level orchestration tool. Consider using `execute_optimized_workflow`
          from the optimization tools for more complex, cost/model-aware workflows.

    Args:
        documents: A list of strings, where each string is a document to process.
        operations: A list of dictionaries, each defining an operation to perform sequentially.
                    Each operation dict should contain:
                    - 'operation': Name of the tool function to call (e.g., 'chunk_document', 'summarize_document').
                    - 'params': Dictionary of parameters for the tool function (excluding the main document input).
                    - 'input_key': (Optional) Key in the results dict of the *previous* operation to use as input.
                                     If omitted or first operation, uses the original document.
                    - 'output_key': Key under which to store the result of this operation.
        max_concurrency: Maximum number of documents to process in parallel for each operation stage. Default 5.

    Returns:
        A list of dictionaries, one for each input document, containing the results of all operations applied to it.
        Example for one document:
        {
            "original_document_index": 0,
            "chunked_content": ["chunk1...", "chunk2..."], # Result of 'chunk_document' stored under 'chunked_content'
            "summary": "Overall summary...",             # Result of 'summarize_document'
            "error": null                             # Error message if any operation failed for this document
        }

    Raises:
        ToolError: For invalid operation definitions or major processing failures.
                   Individual document/operation errors are captured in the results list.
    """
    start_time = time.time()
    logger.info(f"Starting batch processing for {len(documents)} documents with {len(operations)} operations.")

    # Map operation names to functions
    op_map = {
        "chunk_document": chunk_document,
        "summarize_document": summarize_document,
        "extract_entities": extract_entities,
        "generate_qa_pairs": generate_qa_pairs
    }

    # Initialize results structure
    batch_results = [
        {"original_document_index": i, "error": None}
        for i in range(len(documents))
    ]

    # Use original documents as the first input
    current_inputs = documents

    for op_index, op_def in enumerate(operations):
        op_name = op_def.get("operation")
        op_params = op_def.get("params", {})
        input_key = op_def.get("input_key") # If None, uses current_inputs
        output_key = op_def.get("output_key")

        if not op_name or op_name not in op_map or not output_key:
            raise ToolInputError(f"Invalid operation definition at index {op_index}: Must have valid 'operation' ({list(op_map.keys())}) and 'output_key'.")
        
        op_func = op_map[op_name]
        logger.info(f"Executing Operation {op_index + 1}: {op_name} (Output Key: {output_key}) on {len(current_inputs)} items...")

        tasks = []
        semaphore = asyncio.Semaphore(max_concurrency)
        results_for_next_stage = [None] * len(documents) # Prepare storage for this stage's output

        async def run_operation(index, input_data, _semaphore=semaphore, _input_key=input_key, 
                               _op_name=op_name, _op_func=op_func, _op_params=op_params, 
                               _results_for_next_stage=results_for_next_stage):
            async with _semaphore:
                if batch_results[index]["error"]: # Skip if previous stage failed for this doc
                    return None 
                
                # Determine the actual input for this operation
                actual_input = None
                if _input_key:
                     # Find the result from the previous stage for this document index
                     prev_result = batch_results[index].get(_input_key)
                     if prev_result is None:
                          error_msg = f"Input key '{_input_key}' not found in results for document {index} for operation '{_op_name}'"
                          batch_results[index]["error"] = error_msg
                          logger.warning(error_msg)
                          return None
                     actual_input = prev_result
                else:
                     # Use the item from the current_inputs list (original doc or result of prev op)
                     actual_input = input_data
                
                if actual_input is None: # Should not happen if logic above is correct, but safety check
                     error_msg = f"Could not determine input for operation '{_op_name}' on document {index}."
                     batch_results[index]["error"] = error_msg
                     logger.error(error_msg)
                     return None

                try:
                    # Assume the first arg is the main input (document/text)
                    # This might need adjustment for tools with different signatures
                    if _op_name == "chunk_document":
                        result = await _op_func(document=actual_input, **_op_params)
                    elif _op_name in ["summarize_document", "extract_entities", "generate_qa_pairs"]:
                         # These tools return dicts, we usually want the main result field
                         result_dict = await _op_func(document=actual_input, **_op_params)
                         if result_dict.get("success"):
                              if _op_name == "summarize_document":
                                   result = result_dict.get("summary")
                              elif _op_name == "extract_entities":
                                   result = result_dict.get("entities")
                              elif _op_name == "generate_qa_pairs":
                                   result = result_dict.get("qa_pairs")
                              else:
                                   result = result_dict # Fallback
                         else:
                              raise ToolError(status_code=500, detail=result_dict.get("error", "Operation failed"))
                    else:
                        # Default call structure - might need refinement
                        result = await _op_func(actual_input, **_op_params)
                        
                    # Store result for the next stage
                    _results_for_next_stage[index] = result 
                    return result
                except Exception as e:
                    error_msg = f"Error during '{_op_name}' on document {index}: {type(e).__name__}: {str(e)}"
                    batch_results[index]["error"] = error_msg # Store error at the document level
                    logger.error(error_msg, exc_info=False) # Log error but don't need full trace always
                    return None # Indicate failure for this document

        # Create tasks based on the correct input source for this stage
        _input_key = input_key  # Bind loop variable
        _op_name = op_name  # Bind loop variable
        if _input_key:
             # Input comes from previous results, use batch_results index
             tasks = [run_operation(i, None) for i in range(len(documents))] 
        else:
             # Input comes from current_inputs list (original docs or results list from prior stage)
             tasks = [run_operation(i, current_inputs[i]) for i in range(len(current_inputs))] 
             
        op_results = await asyncio.gather(*tasks)
        
        # Store results in the main batch_results structure
        for i, res in enumerate(op_results):
            if batch_results[i]["error"] is None: # Only store if no error occurred for this doc
                 if res is not None: # Check if the operation itself succeeded
                    batch_results[i][output_key] = res
                 else:
                      # If res is None but no error was set, something went wrong internally
                      if batch_results[i]["error"] is None:
                           batch_results[i]["error"] = f"Operation '{_op_name}' did not return a result for document {i}."
        
        # Prepare inputs for the *next* operation stage
        # If the current op failed for a doc, the input for next stage will be None
        current_inputs = results_for_next_stage 

    processing_time = time.time() - start_time
    logger.success(f"Batch processing finished in {processing_time:.2f} seconds.")
    return batch_results