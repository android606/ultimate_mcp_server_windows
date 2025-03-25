"""Document processing tools for LLM Gateway."""
import asyncio
import re
import time
from typing import Any, Dict, List, Optional

from llm_gateway.constants import Provider, TaskType
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.cache import with_cache
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class DocumentTools:
    """Document processing tools for LLM Gateway."""
    
    def __init__(self, mcp_server):
        """Initialize the document tools.
        
        Args:
            mcp_server: MCP server instance
        """
        self.mcp = mcp_server
        self.logger = logger
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register document processing tools with MCP server."""
        
        @self.mcp.tool()
        @with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
        async def chunk_document(
            document: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 100,
            method: str = "token",
            ctx=None
        ) -> Dict[str, Any]:
            """
            Chunk a document into smaller pieces for processing.
            
            Args:
                document: Document text to chunk
                chunk_size: Target size of each chunk
                chunk_overlap: Number of tokens/chars to overlap between chunks
                method: Chunking method (token, character, paragraph, or semantic)
                
            Returns:
                Dictionary containing chunks and metadata
            """
            start_time = time.time()
            
            # Super defensive check for document
            if not document or not isinstance(document, str):
                logger.warning(f"Invalid document provided: {type(document)}")
                empty_doc = "" if document is None else str(document)
                return {
                    "error": "Document is empty or invalid",
                    "chunks": [empty_doc] if empty_doc else [],
                    "chunk_count": 1 if empty_doc else 0,
                    "processing_time": 0.0,
                }
            
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
                    
                if not isinstance(method, str):
                    method = "token"
                    logger.warning(f"Invalid method corrected to {method}")
            except (TypeError, ValueError) as e:
                logger.error(f"Parameter normalization error: {str(e)}")
                chunk_size = 1000
                chunk_overlap = 100
                method = "token"
            
            try:
                # Log input parameters for debugging
                logger.info(
                    f"Starting chunking with method={method}, size={chunk_size}, overlap={chunk_overlap}, doc_length={len(document)}"
                )
                
                # Select chunking method
                if method == "token":
                    chunks = self._chunk_by_tokens(document, chunk_size, chunk_overlap)
                elif method == "character":
                    chunks = self._chunk_by_characters(document, chunk_size, chunk_overlap)
                elif method == "paragraph":
                    chunks = self._chunk_by_paragraphs(document, chunk_size, chunk_overlap)
                elif method == "semantic":
                    chunks = await self._chunk_by_semantic_boundaries(document, chunk_size, chunk_overlap)
                else:
                    chunks = self._chunk_by_tokens(document, chunk_size, chunk_overlap)
                
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
                    f"Chunked document into {len(valid_chunks)} chunks using {method} method",
                    emoji_key=TaskType.SUMMARIZATION.value,
                    time=processing_time
                )
                
                return {
                    "chunks": valid_chunks,
                    "chunk_count": len(valid_chunks),
                    "method": method,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                import traceback
                err_trace = traceback.format_exc()
                logger.error(
                    f"Error in chunk_document: {str(e)}\n{err_trace}",
                    emoji_key="error"
                )
                # Return document as single chunk if there's an error
                if document:
                    return {
                        "error": f"Chunking failed: {str(e)}",
                        "traceback": err_trace,
                        "chunks": [document],
                        "chunk_count": 1,
                        "method": method,
                        "processing_time": time.time() - start_time,
                    }
                else:
                    return {
                        "error": f"Chunking failed: {str(e)}",
                        "traceback": err_trace,
                        "chunks": [],
                        "chunk_count": 0,
                        "method": method,
                        "processing_time": time.time() - start_time,
                    }
            
        @self.mcp.tool()
        @with_cache(ttl=7 * 24 * 60 * 60)  # Cache for 7 days
        async def summarize_document(
            document: str,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            max_length: int = 300,
            format: str = "paragraph",
            ctx=None
        ) -> Dict[str, Any]:
            """
            Summarize a document using the specified provider.
            
            Args:
                document: Document text to summarize
                provider: LLM provider to use
                model: Model name (default based on provider)
                max_length: Maximum summary length in words
                format: Summary format (paragraph, bullet_points, or key_points)
                
            Returns:
                Dictionary containing summary and metadata
            """
            start_time = time.time()
            
            if not document:
                return {
                    "error": "Document is empty",
                    "summary": "",
                    "processing_time": 0.0,
                }
                
            # Prepare prompt based on requested format
            if format == "bullet_points":
                prompt = f"""Summarize the following document as a list of bullet points. Keep the summary concise, focusing on the most important information, and limit it to approximately {max_length} words.

Document:
{document}

Summary (as bullet points):"""
            elif format == "key_points":
                prompt = f"""Extract the key points from the following document. Identify the most important information, main arguments, and conclusions. Limit the summary to approximately {max_length} words.

Document:
{document}

Key points:"""
            else:  # paragraph format
                prompt = f"""Provide a concise summary of the following document in paragraph form. Focus on the main ideas, key information, and conclusions. Limit the summary to approximately {max_length} words.

Document:
{document}

Summary:"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            # Generate summary
            result = await provider_instance.generate_completion(
                prompt=prompt,
                model=model,
                temperature=0.3,  # Lower temperature for more focused summary
                max_tokens=int(max_length * 1.5),  # Convert words to tokens (approximate)
            )
            
            processing_time = time.time() - start_time
            
            # Log result
            logger.success(
                f"Document summarized successfully with {provider}/{result.model}",
                emoji_key=TaskType.SUMMARIZATION.value,
                tokens={
                    "input": result.input_tokens,
                    "output": result.output_tokens
                },
                cost=result.cost,
                time=processing_time
            )
            
            return {
                "summary": result.text,
                "model": result.model,
                "provider": provider,
                "format": format,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.total_tokens,
                },
                "cost": result.cost,
                "processing_time": processing_time,
            }
            
        @self.mcp.tool()
        @with_cache(ttl=14 * 24 * 60 * 60)  # Cache for 14 days
        async def extract_entities(
            document: str,
            entity_types: Optional[List[str]] = None,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Extract entities from a document.
            
            Args:
                document: Document text to analyze
                entity_types: Types of entities to extract (defaults to ["person", "organization", "location", "date", "number"])
                provider: LLM provider to use
                model: Model name (default based on provider)
                
            Returns:
                Dictionary containing extracted entities and metadata
            """
            start_time = time.time()
            
            if not document:
                return {
                    "error": "Document is empty",
                    "entities": {},
                    "processing_time": 0.0,
                }
                
            # Set default entity types if None
            if entity_types is None:
                entity_types = ["person", "organization", "location", "date", "number"]
            
            # Validate entity types
            valid_types = [
                "person", "organization", "location", "date", "time", 
                "number", "currency", "product", "event", "work", "law",
                "language", "facility", "url", "email"
            ]
            
            entity_types = [et for et in entity_types if et in valid_types]
            if not entity_types:
                entity_types = ["person", "organization", "location"]
                
            # Prepare prompt
            prompt = f"""Extract all entities of the following types from the document: {', '.join(entity_types)}.
Return the entities in JSON format, with each entity type as a key and a list of unique entities as the value.
If an entity appears multiple times, only include it once in the list. Include any relevant qualifiers or descriptions with the entities.

For example:
{{
  "person": ["John Smith (CEO)", "Jane Doe (researcher)"],
  "organization": ["Acme Corp", "XYZ University"],
  "location": ["New York City", "Paris, France"],
  ...
}}

Document:
{document}

Extracted entities (JSON format):"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            # Extract entities
            result = await provider_instance.generate_completion(
                prompt=prompt,
                model=model,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000,   # Allow for potentially lengthy entity lists
            )
            
            processing_time = time.time() - start_time
            
            # Parse JSON response
            import json
            try:
                response_text = result.text.strip()
                
                # Extract JSON part if response includes other text
                json_match = re.search(r"({[\s\S]*})", response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
                    
                entities = json.loads(json_str)
                
                # Ensure all requested entity types exist in response
                for entity_type in entity_types:
                    if entity_type not in entities:
                        entities[entity_type] = []
                        
                # Log result
                logger.success(
                    f"Extracted {sum(len(entities[et]) for et in entities)} entities " +
                    f"across {len(entities)} types",
                    emoji_key=TaskType.EXTRACTION.value,
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                return {
                    "entities": entities,
                    "model": result.model,
                    "provider": provider,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except json.JSONDecodeError:
                logger.error(
                    "Failed to parse JSON response from entity extraction",
                    emoji_key="error"
                )
                
                # Return raw text if JSON parsing fails
                return {
                    "error": "Failed to parse structured entities",
                    "raw_text": result.text,
                    "model": result.model,
                    "provider": provider,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                logger.error(
                    f"Error extracting entities: {str(e)}",
                    emoji_key="error"
                )
                return {
                    "error": str(e),
                    "entities": {},
                    "model": model,
                    "provider": provider,
                    "tokens": {"input": 0, "output": 0, "total": 0},
                    "cost": 0.0,
                    "processing_time": time.time() - start_time,
                }
                
        @self.mcp.tool()
        @with_cache(ttl=14 * 24 * 60 * 60)  # Cache for 14 days
        async def generate_qa_pairs(
            document: str,
            num_questions: int = 5,
            question_types: List[str] = ["factual", "conceptual"],
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Generate question-answer pairs from a document.
            
            Args:
                document: Document text to analyze
                num_questions: Number of QA pairs to generate
                question_types: Types of questions to generate
                provider: LLM provider to use
                model: Model name (default based on provider)
                
            Returns:
                Dictionary containing QA pairs and metadata
            """
            start_time = time.time()
            
            if not document:
                return {
                    "error": "Document is empty",
                    "qa_pairs": [],
                    "processing_time": 0.0,
                }
                
            # Validate question types
            valid_types = ["factual", "conceptual", "open-ended", "analytical", "procedural"]
            question_types = [qt for qt in question_types if qt in valid_types]
            if not question_types:
                question_types = ["factual", "conceptual"]
                
            # Prepare prompt
            prompt = f"""Generate {num_questions} question-answer pairs based on the following document. 
Include a mix of these question types: {', '.join(question_types)}.
Return the QA pairs in JSON format, where each pair has a "question", "answer", and "type" field.

For example:
[
  {{
    "question": "What is the main topic of the document?",
    "answer": "The document discusses the effects of climate change on marine ecosystems.",
    "type": "conceptual"
  }},
  ...
]

Document:
{document}

QA pairs (JSON format):"""
            
            # Get provider instance
            provider_instance = get_provider(provider)
            
            # Generate QA pairs
            result = await provider_instance.generate_completion(
                prompt=prompt,
                model=model,
                temperature=0.5,  # Moderate temperature for creativity but consistency
                max_tokens=num_questions * 150,  # Approximate token count for response
            )
            
            processing_time = time.time() - start_time
            
            # Parse JSON response
            import json
            try:
                response_text = result.text.strip()
                
                # Extract JSON part if response includes other text
                json_match = re.search(r"(\[[\s\S]*\])", response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
                    
                qa_pairs = json.loads(json_str)
                
                # Log result
                logger.success(
                    f"Generated {len(qa_pairs)} QA pairs",
                    emoji_key=TaskType.GENERATION.value,
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                return {
                    "qa_pairs": qa_pairs,
                    "model": result.model,
                    "provider": provider,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except json.JSONDecodeError:
                logger.error(
                    "Failed to parse JSON response from QA generation",
                    emoji_key="error"
                )
                
                # Return raw text if JSON parsing fails
                return {
                    "error": "Failed to parse structured QA pairs",
                    "raw_text": result.text,
                    "model": result.model,
                    "provider": provider,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                logger.error(
                    f"Error generating QA pairs: {str(e)}",
                    emoji_key="error"
                )
                return {
                    "error": str(e),
                    "qa_pairs": [],
                    "model": model,
                    "provider": provider,
                    "tokens": {"input": 0, "output": 0, "total": 0},
                    "cost": 0.0,
                    "processing_time": time.time() - start_time,
                }
                
        @self.mcp.tool()
        @with_cache(ttl=7 * 24 * 60 * 60)  # Cache for 7 days
        async def process_document_batch(
            documents: List[str],
            operation: str,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            max_concurrency: int = 5,
            operation_params: Optional[Dict[str, Any]] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Process a batch of documents with the specified operation.
            
            Args:
                documents: List of document texts to process
                operation: Operation to perform (summarize, extract_entities, generate_qa)
                provider: LLM provider to use
                model: Model name (default based on provider)
                max_concurrency: Maximum number of concurrent operations
                operation_params: Additional parameters for the operation
                
            Returns:
                Dictionary containing results for each document
            """
            start_time = time.time()
            
            if not documents:
                return {
                    "error": "No documents provided",
                    "results": [],
                    "processing_time": 0.0,
                }
            
            try:
                # Set default operation params
                operation_params = operation_params or {}
                
                # Validate operation
                valid_operations = ["summarize", "extract_entities", "generate_qa"]
                if operation not in valid_operations:
                    return {
                        "error": f"Invalid operation: {operation}. " +
                                f"Valid options: {', '.join(valid_operations)}",
                        "results": [],
                        "processing_time": 0.0,
                    }
                    
                # Log batch info
                logger.info(f"Processing batch of {len(documents)} documents with operation: {operation}")
                
                # Create tasks for each document
                tasks = []
                semaphore = asyncio.Semaphore(max_concurrency)
                
                async def process_document(doc_index: int, doc_text: str):
                    async with semaphore:
                        try:
                            logger.debug(f"Processing document {doc_index}, length={len(doc_text)}")
                            
                            if operation == "summarize":
                                tool_result = await summarize_document(
                                    document=doc_text,
                                    provider=provider,
                                    model=model,
                                    **operation_params
                                )
                            elif operation == "extract_entities":
                                tool_result = await extract_entities(
                                    document=doc_text,
                                    provider=provider,
                                    model=model,
                                    **operation_params
                                )
                            elif operation == "generate_qa":
                                tool_result = await generate_qa_pairs(
                                    document=doc_text,
                                    provider=provider,
                                    model=model,
                                    **operation_params
                                )
                            else:
                                tool_result = {"error": f"Unsupported operation: {operation}"}
                                
                            return {
                                "document_index": doc_index,
                                "result": tool_result,
                                "success": "error" not in tool_result,
                            }
                        except Exception as e:
                            import traceback
                            err_trace = traceback.format_exc()
                            logger.error(
                                f"Error processing document {doc_index}: {str(e)}\n{err_trace}",
                                emoji_key="error"
                            )
                            return {
                                "document_index": doc_index,
                                "result": {
                                    "error": str(e),
                                    "traceback": err_trace
                                },
                                "success": False,
                            }
                
                for i, doc in enumerate(documents):
                    tasks.append(process_document(i, doc))
                    
                # Process documents concurrently
                results = await asyncio.gather(*tasks)
                
                processing_time = time.time() - start_time
                
                # Calculate success rate and statistics
                success_count = sum(1 for r in results if r["success"])
                
                # Log result
                logger.success(
                    f"Batch processed {len(documents)} documents " +
                    f"({success_count} successful, {len(documents) - success_count} failed)",
                    emoji_key="processing",
                    time=processing_time
                )
                
                # Return results sorted by document index
                sorted_results = sorted(results, key=lambda r: r["document_index"])
                
                return {
                    "results": sorted_results,
                    "success_count": success_count,
                    "failure_count": len(documents) - success_count,
                    "operation": operation,
                    "provider": provider,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                import traceback
                err_trace = traceback.format_exc()
                logger.error(
                    f"Error in batch processing: {str(e)}\n{err_trace}",
                    emoji_key="error"
                )
                return {
                    "error": f"Batch processing failed: {str(e)}",
                    "traceback": err_trace,
                    "results": [],
                    "success_count": 0,
                    "failure_count": len(documents),
                    "operation": operation,
                    "provider": provider,
                    "processing_time": time.time() - start_time,
                }

    def _chunk_by_tokens(self, document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk document by estimated token count using a more accurate tokenization.
        
        Args:
            document: Document text
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of document chunks
        """
        try:
            # Use tiktoken if available for accurate tokenization
            try:
                import tiktoken
                # Use cl100k_base encoding which is used by most recent models
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(document)
                
                logger.debug(f"Tokenized document: {len(tokens)} tokens")
                
                chunks = []
                i = 0
                while i < len(tokens):
                    # Get current chunk
                    chunk_end = min(i + chunk_size, len(tokens))
                    
                    # Try to end at a sentence boundary if possible
                    if chunk_end < len(tokens):
                        # Find a period or newline in the last 20% of the chunk
                        look_back_size = min(chunk_size // 5, 100)  # Look back up to 100 tokens or 20%
                        for j in range(chunk_end, max(i, chunk_end - look_back_size), -1):
                            # Check if token corresponds to period, question mark, exclamation mark, or newline
                            token_text = encoding.decode([tokens[j]])
                            if token_text in [".", "?", "!", "\n"]:
                                # Found a good break point, but make sure we include it
                                chunk_end = j + 1
                                break
                    
                    # Decode the chunk back to text
                    current_chunk = encoding.decode(tokens[i:chunk_end])
                    
                    # Improve semantic coherence of large chunks
                    if len(current_chunk) > chunk_size // 4:
                        improved_chunks = self._improve_chunk_coherence(current_chunk, chunk_size)
                        chunks.extend(improved_chunks)
                    else:
                        chunks.append(current_chunk)
                    
                    # Move to next chunk with overlap
                    i += max(1, chunk_size - chunk_overlap)  # Ensure we make progress
                
                return chunks
                
            except ImportError:
                # Fallback to our token estimator if tiktoken not available
                logger.info("tiktoken not available, falling back to token estimation")
                return self._chunk_by_token_estimation(document, chunk_size, chunk_overlap)
                
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            logger.error(f"Error in _chunk_by_tokens: {str(e)}\n{err_trace}")
            # Return document as a single chunk rather than failing completely
            return [document]

    def _improve_chunk_coherence(self, chunk_text: str, target_size: int) -> List[str]:
        """Improve the semantic coherence of a chunk by finding better split points.
        
        Args:
            chunk_text: Text of the chunk to improve
            target_size: Target size for chunks
            
        Returns:
            List of improved chunks
        """
        try:
            # Log input for debugging
            logger.debug(f"Improving chunk coherence: text_length={len(chunk_text)}, target_size={target_size}")
            
            # Defensive check - return original if empty or tiny
            if not chunk_text or len(chunk_text) < 50:
                return [chunk_text]
            
            # Split into sentences
            sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
            sentences = re.split(sentence_pattern, chunk_text)
            
            # Filter out empty sentences
            sentences = [s for s in sentences if s and s.strip()]
            
            logger.debug(f"Split into {len(sentences)} sentences")
            
            # If only a few sentences, not much we can do
            if len(sentences) <= 3:
                return [chunk_text]
            
            # Find natural groupings of sentences
            groups = []
            current_group = [sentences[0]]
            
            # Group by keywords and semantic transitions
            transition_words = {"however", "nevertheless", "conversely", "meanwhile", 
                              "furthermore", "additionally", "consequently", "therefore",
                              "thus", "hence", "accordingly", "subsequently"}
            
            for i in range(1, len(sentences)):
                sentence = sentences[i]
                if not sentence or not sentence.strip():
                    continue
                    
                # Check for transition indicators
                words = sentence.split()
                if not words:
                    current_group.append(sentence)
                    continue
                
                sentence_start = ' '.join(words[:min(3, len(words))]).lower()
                has_transition = any(tw in sentence_start for tw in transition_words)
                
                # Check sentence length - very short sentences often continue previous thought
                is_short = len(words) < 5
                
                # Start new group on transitions or long content shifts
                if has_transition or (not is_short and len(current_group) >= 3):
                    groups.append(current_group)
                    current_group = [sentence]
                else:
                    current_group.append(sentence)
            
            # Add the last group
            if current_group:
                groups.append(current_group)
            
            logger.debug(f"Created {len(groups)} sentence groups")
            
            # If no groups were created, return original
            if not groups:
                return [chunk_text]
            
            # Combine groups into chunks of appropriate size
            improved_chunks = []
            current_combined = []
            current_size = 0
            
            for group in groups:
                # Join the group sentences with spaces instead of blindly
                group_sentences = [s.strip() for s in group if s and s.strip()]
                if not group_sentences:
                    continue
                    
                group_text = ' '.join(group_sentences)
                group_size = len(group_text)
                
                if current_size + group_size > target_size and current_combined:
                    combined_text = ' '.join(current_combined)
                    if combined_text:  # Don't add empty chunks
                        improved_chunks.append(combined_text)
                    current_combined = [group_text]
                    current_size = group_size
                else:
                    current_combined.append(group_text)
                    current_size += group_size
            
            # Add the last combined group
            if current_combined:
                combined_text = ' '.join(current_combined)
                if combined_text:  # Don't add empty chunks
                    improved_chunks.append(combined_text)
            
            logger.debug(f"Created {len(improved_chunks)} improved chunks")
            
            # In case we end up with no chunks (rare), return the original
            if not improved_chunks:
                return [chunk_text]
                
            return improved_chunks
            
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            logger.error(f"Error in _improve_chunk_coherence: {str(e)}\n{err_trace}")
            # Return original text as a single chunk to avoid failing
            return [chunk_text]

    def _chunk_by_token_estimation(self, document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk document by estimated token count when tiktoken is not available.
        
        Args:
            document: Document text
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of document chunks
        """
        try:
            # Safety check
            if not document:
                return [document]
            
            # Tokenize document
            estimated_tokens = self.estimate_tokens(document)
            
            # Make sure we have at least some tokens
            if not estimated_tokens:
                logger.warning("No tokens estimated, returning document as single chunk")
                return [document]
                
            logger.debug(f"Estimated tokens: {len(estimated_tokens)} for document of length {len(document)}")
            
            # Create chunks based on estimated tokens
            chunks = []
            i = 0
            while i < len(estimated_tokens):
                # Get current chunk end index
                chunk_end = min(i + chunk_size, len(estimated_tokens))
                
                # Try to end at a sentence boundary if within reasonable distance
                if chunk_end < len(estimated_tokens):
                    # Find a good breakpoint in the last 10% of the chunk
                    search_start = max(i, chunk_end - (chunk_size // 10))
                    for j in range(chunk_end, search_start, -1):
                        if j >= 0 and j < len(estimated_tokens) and estimated_tokens[j] in [".", "?", "!"]:
                            chunk_end = j + 1
                            break
                
                # Safely reconstruct text from tokens
                token_slice = estimated_tokens[i:chunk_end]
                if not token_slice:  # Skip empty slices
                    i += max(1, chunk_size - chunk_overlap)
                    continue
                
                # Rebuild text carefully
                current_chunk = ""
                for token in token_slice:
                    if not token:  # Skip empty tokens
                        continue
                    # Add proper spacing
                    if token.startswith("'") or token in [",", ".", ":", ";", "!", "?"]:
                        current_chunk += token
                    else:
                        current_chunk += " " + token
                current_chunk = current_chunk.strip()
                
                # Skip empty chunks
                if not current_chunk:
                    i += max(1, chunk_size - chunk_overlap)
                    continue
                
                # Apply semantic coherence improvements only for significant chunks
                if len(current_chunk) > chunk_size // 4 and len(current_chunk) > 100:
                    try:
                        improved_chunks = self._improve_chunk_coherence(current_chunk, chunk_size)
                        chunks.extend(improved_chunks)
                    except Exception as e:
                        logger.error(f"Error improving chunk coherence: {str(e)}, using original chunk")
                        chunks.append(current_chunk)
                else:
                    chunks.append(current_chunk)
                
                # Move to next chunk with overlap
                next_position = max(1, chunk_size - chunk_overlap)
                i += next_position  # Ensure we make progress
                logger.debug(f"Moving to next chunk position: {i} (advanced by {next_position})")
            
            # Ensure we returned something
            if not chunks:
                logger.warning("No chunks created, returning original document")
                return [document]
                
            logger.debug(f"Created {len(chunks)} chunks via token estimation")
            return chunks
            
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            logger.error(f"Error in _chunk_by_token_estimation: {str(e)}\n{err_trace}")
            # Return document as a single chunk rather than failing
            return [document]
        
    def estimate_tokens(self, document: str) -> List[str]:
        """Estimate tokenization of text for chunk size calculation.
        
        Args:
            document: Text content to estimate tokens for
            
        Returns:
            List of estimated tokens
        """
        try:
            # Safety check
            if not document:
                return []
                
            # Split text into words
            words = re.findall(r'\w+|[^\w\s]', document)
            
            # Safety check
            if not words:
                logger.warning("No words found in document")
                return [document]  # Return the whole document as a token
            
            tokens = []
            for word in words:
                # Skip empty words
                if not word:
                    continue
                    
                # Handle different token types
                if word.isdigit():
                    # Numbers are typically 1 token per 2-3 digits
                    for i in range(0, len(word), 3):
                        tokens.append(word[i:i+3])
                else:
                    # Regular words are split by subword tokenization
                    # This is a simplified approximation
                    if len(word) <= 4:
                        tokens.append(word)
                    else:
                        # Split longer words into subword pieces
                        remaining = word
                        while remaining:
                            # Take chunks of 4-6 chars as common subword sizes
                            piece_size = min(4 + (len(remaining) % 3), len(remaining))
                            tokens.append(remaining[:piece_size])
                            remaining = remaining[piece_size:]
            
            # Safety check - make sure we have tokens
            if not tokens:
                logger.warning("No tokens generated from words")
                return [document]  # Return the whole document as a token
                
            logger.debug(f"Estimated {len(tokens)} tokens from {len(words)} words")
            return tokens
            
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            logger.error(f"Error in estimate_tokens: {str(e)}\n{err_trace}")
            # Return minimal set of tokens to prevent failure
            return [document]

    def _chunk_by_characters(self, document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk document by character count.
        
        Args:
            document: Document text
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of document chunks
        """
        chunks = []
        i = 0
        while i < len(document):
            # Get current chunk
            chunk_end = min(i + chunk_size, len(document))
            
            # Try to end at sentence boundary
            if chunk_end < len(document):
                # Look for sentence boundary within the last 20% of the chunk
                search_start = chunk_end - (chunk_size // 5)
                search_text = document[search_start:chunk_end]
                
                # Find last sentence boundary
                sentence_end = max(
                    search_text.rfind('. '),
                    search_text.rfind('? '),
                    search_text.rfind('! '),
                    search_text.rfind('\n\n')
                )
                
                if sentence_end != -1:
                    chunk_end = search_start + sentence_end + 2  # Include the period and space
            
            # Extract current chunk
            current_chunk = document[i:chunk_end]
            chunks.append(current_chunk)
            
            # Move to next chunk with overlap
            i += chunk_size - chunk_overlap
            
            # Ensure we make progress
            if i <= 0:
                i = chunk_end
        
        return chunks

    def _chunk_by_paragraphs(self, document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk document by paragraphs.
        
        Args:
            document: Document text
            chunk_size: Maximum characters per chunk
            chunk_overlap: Not used in this method
            
        Returns:
            List of document chunks
        """
        # Split document into paragraphs
        paragraphs = re.split(r'\n\s*\n', document)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_size = len(paragraph)
            
            # If adding this paragraph exceeds chunk size and we already have content,
            # finish the current chunk
            if current_size + paragraph_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # If a single paragraph is larger than chunk size, split it
            if paragraph_size > chunk_size:
                # Split the paragraph using character chunking
                paragraph_chunks = self._chunk_by_characters(paragraph, chunk_size, chunk_overlap)
                for p_chunk in paragraph_chunks:
                    chunks.append(p_chunk)
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_size += paragraph_size + 4  # Add 4 for the newlines
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    async def _chunk_by_semantic_boundaries(self, document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk document by semantic boundaries based on content analysis.
        
        This uses multiple techniques to identify semantic boundaries:
        1. Structure-based: Headers, lists, and other formatting elements
        2. Content-based: Topic shifts and semantic units
        3. NLP-based: If available, uses embeddings to ensure semantic coherence
        
        Args:
            document: Document text
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of document chunks
        """
        # Try to use sentence transformers for embeddings if available
        embedding_available = False
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
            embedding_available = True
        except ImportError:
            # Continue without embeddings
            pass
        
        # 1. Initial structural analysis
        # Define patterns for structural elements
        patterns = {
            'header': r'(?:^|\n)(?:#{1,6}|\*{1,3}|=+|-+|\d+\.) +(.+?)(?:\n|$)',
            'list_item': r'(?:^|\n)(?:[-*+]|\d+\.) +(.+?)(?:\n|$)',
            'paragraph': r'(?:^|\n)(.+?)(?:\n\s*\n|$)',
            'code_block': r'(?:^|\n)```.*?\n(.+?)```(?:\n|$)',
            'table': r'(?:^|\n)(?:\|.+?\|)(?:\n\|[-:]+\|[-:]+\|)+(?:\n\|.+?\|)+',
            "quote": r"(?:^|\n)>+(.+?)(?:\n\s*\n|$)",
        }
        
        # Identify structural elements and their positions
        structure_elements = []
        for elem_type, pattern in patterns.items():
            for match in re.finditer(pattern, document, re.DOTALL):
                structure_elements.append({
                    'type': elem_type,
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(0)
                })
        
        # Sort elements by position
        structure_elements.sort(key=lambda x: x['start'])
        
        # 2. Identify potential semantic boundaries
        # First, split into sentences
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, document)
        
        # Calculate sentence embeddings if available
        sentence_embeddings = []
        sample_indices = []
        if embedding_available:
            # Only embed a subset of sentences if document is very large
            if len(sentences) > 500:
                # Sample sentences throughout the document
                step = max(1, len(sentences) // 500)
                sample_indices = list(range(0, len(sentences), step))
                sample_sentences = [sentences[i] for i in sample_indices]
                sentence_embeddings = model.encode(sample_sentences)
            else:
                sample_indices = list(range(len(sentences)))
                sentence_embeddings = model.encode(sentences)
        
        # 3. Identify topic shifts using embeddings
        topic_boundaries = []
        if embedding_available and len(sentence_embeddings) > 1:
            # Calculate cosine similarity between adjacent sentences
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                sim = cosine_similarity(
                    [sentence_embeddings[i]], 
                    [sentence_embeddings[i + 1]]
                )[0][0]
                similarities.append(sim)
            
            # Identify significant drops in similarity
            # (indicates potential topic shift)
            avg_sim = sum(similarities) / len(similarities)
            std_sim = (sum((s - avg_sim) ** 2 for s in similarities) / len(similarities)) ** 0.5
            threshold = avg_sim - std_sim
            
            for i, sim in enumerate(similarities):
                if sim < threshold and i < len(sample_indices) - 1:
                    topic_boundaries.append(sample_indices[i])
        
        # 4. Create chunks based on both structural and semantic boundaries
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Combine all boundaries
        all_boundaries = []
        
        # Add structural boundaries
        for elem in structure_elements:
            if elem['type'] == 'header':
                all_boundaries.append({'pos': elem['start'], 'weight': 1.0})
            elif elem['type'] == 'paragraph':
                all_boundaries.append({'pos': elem['start'], 'weight': 0.5})
        
        # Add topic shift boundaries
        for pos in topic_boundaries:
            all_boundaries.append({'pos': pos, 'weight': 0.8})
        
        # Sort boundaries by position
        all_boundaries.sort(key=lambda x: x['pos'])
        
        # Split document into sections at boundaries
        if not all_boundaries:
            # No clear boundaries found, fall back to paragraph chunking
            return self._chunk_by_paragraphs(document, chunk_size, chunk_overlap)
        
        # Create sections from boundaries
        sections = []
        last_pos = 0
        
        for boundary in all_boundaries:
            if boundary['pos'] > last_pos:
                section_text = document[last_pos:boundary['pos']].strip()
                if section_text:
                    sections.append(section_text)
                last_pos = boundary['pos']
        
        # Add final section
        if last_pos < len(document):
            section_text = document[last_pos:].strip()
            if section_text:
                sections.append(section_text)
        
        # Process sections into chunks
        for section in sections:
            section_size = len(section)
            
            # If adding this section exceeds chunk size and we already have content,
            # finish the current chunk
            if current_size + section_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                # Include the last section from the previous chunk for context
                if current_chunk and chunk_overlap > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[-1])
                else:
                    current_chunk = []
                    current_size = 0
            
            # If a single section is larger than chunk size, split it
            if section_size > chunk_size:
                # If current chunk has content, finish it first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split the section using paragraph chunking
                section_chunks = self._chunk_by_paragraphs(section, chunk_size, chunk_overlap)
                for s_chunk in section_chunks:
                    chunks.append(s_chunk)
            else:
                # Add section to current chunk
                current_chunk.append(section)
                current_size += section_size + 4  # Add 4 for the newlines
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks