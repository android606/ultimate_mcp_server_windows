"""Text processing utilities for LLM Gateway."""
import re
import string
from typing import Any, Dict, List, Optional

from llm_gateway.utils import get_logger

logger = get_logger(__name__)


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
        add_ellipsis: Whether to add ellipsis to truncated text
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
        
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    
    # Find the last sentence boundary in the truncated text
    last_boundary = max(
        truncated.rfind('. '), 
        truncated.rfind('? '), 
        truncated.rfind('! '),
        truncated.rfind('\n\n')
    )
    
    if last_boundary > max_length * 0.8:  # Only truncate at boundary if it's not too short
        truncated = truncated[:last_boundary + 1]
    
    # Add ellipsis if requested and text was truncated
    if add_ellipsis and len(text) > len(truncated):
        truncated = truncated.rstrip() + "..."
        
    return truncated


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate the number of tokens in text.
    
    Args:
        text: Text to count tokens for
        model: Optional model name to use for counting
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
        
    # Try to use tiktoken if available (accurate for OpenAI models)
    try:
        import tiktoken
        
        # Select encoding based on model
        if model and model.startswith("gpt-4o"):
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif model and "claude" in model.lower():
            # For Claude, use cl100k_base as approximation
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base (used by most recent models)
            encoding = tiktoken.get_encoding("cl100k_base")
            
        return len(encoding.encode(text))
        
    except ImportError:
        # Fallback to character-based estimation if tiktoken is not available
        return _estimate_tokens_by_chars(text)


def _estimate_tokens_by_chars(text: str) -> int:
    """Estimate tokens based on character count and heuristics.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Character-based estimation (rough approximation)
    avg_chars_per_token = 4.0
    
    # Count characters
    char_count = len(text)
    
    # Account for whitespace more efficiently representing tokens
    whitespace_count = sum(1 for c in text if c.isspace())
    
    # Count numbers (numbers are often encoded efficiently)
    digit_count = sum(1 for c in text if c.isdigit())
    
    # Adjust total count based on character types
    adjusted_count = char_count + (whitespace_count * 0.5) - (digit_count * 0.5)
    
    # Estimate tokens
    return max(1, int(adjusted_count / avg_chars_per_token))


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_whitespace: bool = False,
    remove_urls: bool = False,
    remove_numbers: bool = False,
) -> str:
    """Normalize text with various options.
    
    Args:
        text: Text to normalize
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation
        remove_whitespace: Replace multiple whitespace with single space
        remove_urls: Remove URLs
        remove_numbers: Remove numbers
        
    Returns:
        Normalized text
    """
    if not text:
        return text
        
    # Convert to lowercase
    if lowercase:
        text = text.lower()
        
    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
        
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
        
    # Normalize whitespace
    if remove_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text


def extract_key_phrases(text: str, max_phrases: int = 5, min_word_length: int = 3) -> List[str]:
    """Extract key phrases from text using statistical methods.
    
    Args:
        text: Source text
        max_phrases: Maximum number of phrases to extract
        min_word_length: Minimum word length to consider
        
    Returns:
        List of key phrases
    """
    if not text:
        return []
        
    # Normalize text
    normalized = normalize_text(
        text,
        lowercase=True,
        remove_punctuation=False,
        remove_whitespace=True,
        remove_urls=True,
    )
    
    # Split into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', normalized)
    
    # Extract phrases (simple noun phrases)
    phrases = []
    for sentence in sentences:
        # Find potential noun phrases
        np_matches = re.finditer(
            r'\b((?:(?:[A-Za-z]+\s+){0,2}[A-Za-z]{%d,})|(?:[A-Za-z]{%d,}))\b' % 
            (min_word_length, min_word_length),
            sentence
        )
        for match in np_matches:
            phrase = match.group(0).strip()
            if len(phrase.split()) <= 3:  # Limit to 3-word phrases
                phrases.append(phrase)
    
    # Count phrase frequency
    phrase_counts = {}
    for phrase in phrases:
        if phrase in phrase_counts:
            phrase_counts[phrase] += 1
        else:
            phrase_counts[phrase] = 1
    
    # Sort by frequency
    sorted_phrases = sorted(
        phrase_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Return top phrases
    return [phrase for phrase, _ in sorted_phrases[:max_phrases]]


def split_text_by_similarity(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks of similar size at natural boundaries.
    
    Args:
        text: Text to split
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text]
    
    # Define boundary patterns in order of preference
    boundaries = [
        r'\n\s*\n',  # Double newline (paragraph)
        r'\.\s+[A-Z]',  # End of sentence
        r',\s+',  # Comma with space
        r'\s+',  # Any whitespace
    ]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Determine end position for this chunk
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of the text, find a good boundary
        if end < len(text):
            # Try each boundary pattern in order
            for pattern in boundaries:
                # Search for the boundary pattern before the end position
                search_area = text[max(start, end - chunk_size // 4):end]
                matches = list(re.finditer(pattern, search_area))
                
                if matches:
                    # Found a good boundary, adjust end position
                    match_end = matches[-1].end()
                    end = max(start, end - chunk_size // 4) + match_end
                    break
        
        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move to the next chunk with overlap
        start = end - overlap
    
    return chunks


def sanitize_text(text: str, allowed_tags: Optional[List[str]] = None) -> str:
    """Sanitize text by removing potentially harmful elements.
    
    Args:
        text: Text to sanitize
        allowed_tags: Optional list of allowed HTML tags
        
    Returns:
        Sanitized text
    """
    if not text:
        return text
    
    # Remove script tags and content
    text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text)
    
    # Remove style tags and content
    text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', text)
    
    # Remove comments
    text = re.sub(r'<!--.*?-->', '', text)
    
    # Handle HTML tags based on allowed_tags
    if allowed_tags:
        # Allow specified tags but remove all others
        allowed_pattern = '|'.join(allowed_tags)  # noqa: F841
        
        # Function to process tag matches
        def tag_replacer(match):
            tag = match.group(1).lower()
            if tag in allowed_tags:
                return match.group(0)
            else:
                return ''
                
        # Replace tags not in allowed_tags
        text = re.sub(r'<(\w+)(?:\s[^>]*)?(?:\/?>|>.*?<\/\1>)', tag_replacer, text)
    else:
        # Remove all HTML tags
        text = re.sub(r'<[^>]*>', '', text)
    
    # Convert HTML entities
    text = _convert_html_entities(text)
    
    return text


def _convert_html_entities(text: str) -> str:
    """Convert common HTML entities to characters.
    
    Args:
        text: Text with HTML entities
        
    Returns:
        Text with entities converted to characters
    """
    # Define common HTML entities
    entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&nbsp;': ' ',
    }
    
    # Replace each entity
    for entity, char in entities.items():
        text = text.replace(entity, char)
    
    # Handle numeric entities
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&#x([0-9a-f]+);', lambda m: chr(int(m.group(1), 16)), text)
    
    return text


def extract_structured_data(text: str, patterns: Dict[str, str]) -> Dict[str, Any]:
    """Extract structured data from text using regex patterns.
    
    Args:
        text: Source text
        patterns: Dictionary of field name to regex pattern
        
    Returns:
        Dictionary of extracted data
    """
    if not text:
        return {}
    
    result = {}
    
    # Apply each pattern
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # If the pattern has groups, use the first group
            if match.groups():
                result[field] = match.group(1).strip()
            else:
                result[field] = match.group(0).strip()
    
    return result


def find_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using character n-grams.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = normalize_text(text1, lowercase=True, remove_whitespace=True)
    text2 = normalize_text(text2, lowercase=True, remove_whitespace=True)
    
    # Generate character trigrams
    def get_trigrams(s):
        return set(s[i:i+3] for i in range(len(s) - 2))
        
    trigrams1 = get_trigrams(text1)
    trigrams2 = get_trigrams(text2)
    
    # Find common trigrams
    common = trigrams1.intersection(trigrams2)
    
    # Calculate Jaccard similarity
    if not trigrams1 and not trigrams2:
        return 1.0  # Both strings are too short for trigrams
    
    return len(common) / max(1, len(trigrams1.union(trigrams2)))


def get_text_stats(text: str) -> Dict[str, Any]:
    """Get statistical information about text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of text statistics
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "estimated_tokens": 0,
        }
    
    # Character count
    char_count = len(text)
    
    # Word count
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    # Sentence count
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Paragraph count
    paragraphs = re.split(r'\n\s*\n', text)
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
    # Average sentence length (in words)
    avg_sentence_length = word_count / max(1, sentence_count)
    
    # Estimated tokens
    estimated_tokens = count_tokens(text)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_word_length": round(avg_word_length, 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "estimated_tokens": estimated_tokens,
    }


def preprocess_text(text: str) -> str:
    """Preprocesses text for classification tasks.
    
    Args:
        text: Text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not text:
        return text
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove control characters
    text = re.sub(r'[\x00-\x09\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Remove excessive punctuation repetition
    text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)
    
    # Truncate if extremely long (preserve beginning and end)
    max_chars = 100000  # Reasonable limit to prevent token explosion
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + " [...text truncated...] " + text[-half:]
        
    return text