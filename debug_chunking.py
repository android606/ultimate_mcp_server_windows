#!/usr/bin/env python
"""Diagnostic script for debugging document chunking."""
import sys
import traceback
import asyncio

# Create a simple test document
SAMPLE_DOCUMENT = """
# Artificial Intelligence: An Overview

Artificial Intelligence (AI) refers to systems or machines that mimic human intelligence to perform tasks and can iteratively improve themselves based on the information they collect. AI manifests in a number of forms including:

## Machine Learning

Machine Learning is a subset of AI that enables a system to learn from data rather than through explicit programming. It involves algorithms that improve automatically through experience.
"""

def debug_tokenization():
    """Debug the tokenization process which seems to be the source of the error."""
    print("\n=== DEBUGGING TOKENIZATION ===")
    
    try:
        # Try to import tiktoken to see if it's available
        try:
            import tiktoken
            print("tiktoken is available")
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(SAMPLE_DOCUMENT)
            print(f"tiktoken encoded document into {len(tokens)} tokens")
            
            # Test accessing tokens to find the error
            for i in range(min(10, len(tokens))):
                token_text = encoding.decode([tokens[i]])
                print(f"Token {i}: {token_text!r}")
                
        except ImportError:
            print("tiktoken is NOT available, will use fallback tokenization")
            
        # Try our fallback tokenization
        print("\nTesting fallback tokenization:")
        import re
        words = re.findall(r'\w+|[^\w\s]', SAMPLE_DOCUMENT)
        print(f"Found {len(words)} words/tokens")
        
        # Print a sample of tokens
        print("\nSample tokens from fallback tokenization:")
        for i, word in enumerate(words[:20]):
            print(f"Token {i}: {word!r}")
            
        # Try to reconstruct text from tokens
        print("\nTesting text reconstruction:")
        sample_tokens = words[:30]
        reconstructed = ""
        for token in sample_tokens:
            if token.startswith("'") or token in [",", ".", ":", ";", "!", "?"]:
                reconstructed += token
            else:
                reconstructed += " " + token
        print(f"Reconstructed text: {reconstructed!r}")
        
    except Exception as e:
        print(f"ERROR in tokenization: {str(e)}")
        print(traceback.format_exc())
        
def debug_chunk_by_tokens():
    """Directly debug the _chunk_by_tokens function."""
    print("\n=== DEBUGGING CHUNK_BY_TOKENS ===")
    
    try:
        # First try tiktoken version
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(SAMPLE_DOCUMENT)
        
        print(f"Encoded {len(tokens)} tokens")
        
        # Manual chunking logic
        chunk_size = 200
        chunk_overlap = 50
        chunks = []
        i = 0
        
        print(f"Chunking with size={chunk_size}, overlap={chunk_overlap}")
        
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
                    print(f"Checking token {j}: {token_text!r}")
                    if token_text in [".", "?", "!", "\n"]:
                        # Found a good break point, but make sure we include it
                        chunk_end = j + 1
                        print(f"Found break at token {j}: {token_text!r}")
                        break
            
            # Decode the chunk back to text
            current_chunk = encoding.decode(tokens[i:chunk_end])
            chunks.append(current_chunk)
            
            print(f"Created chunk {len(chunks)}: {len(current_chunk)} chars, starts with {current_chunk[:30]!r}")
            
            # Move to next chunk with overlap
            i += max(1, chunk_size - chunk_overlap)  # Ensure we make progress
            
        print(f"Created {len(chunks)} chunks")
        
    except Exception as e:
        print(f"ERROR in chunk_by_tokens: {str(e)}")
        print(traceback.format_exc())
        
def debug_improve_chunk_coherence():
    """Debug the _improve_chunk_coherence function."""
    print("\n=== DEBUGGING IMPROVE_CHUNK_COHERENCE ===")
    
    try:
        chunk_text = SAMPLE_DOCUMENT
        target_size = 500
        
        print(f"Input text length: {len(chunk_text)}")
        
        # Split into sentences
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        import re
        sentences = re.split(sentence_pattern, chunk_text)
        
        print(f"Split into {len(sentences)} sentences")
        
        # Print first few sentences
        for i, sentence in enumerate(sentences[:5]):
            print(f"Sentence {i}: {sentence[:50]}...")
        
        # If only a few sentences, not much we can do
        if len(sentences) <= 3:
            print("Too few sentences, would return original chunk")
            return
        
        # Find natural groupings of sentences
        groups = []
        current_group = [sentences[0]]
        
        # Group by keywords and semantic transitions
        transition_words = {"however", "nevertheless", "conversely", "meanwhile", 
                          "furthermore", "additionally", "consequently", "therefore",
                          "thus", "hence", "accordingly", "subsequently"}
        
        print("\nGrouping sentences:")
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            
            # Check for transition indicators
            words = sentence.split()
            if not words:
                print(f"Empty sentence at index {i}")
                continue
                
            sentence_start = ' '.join(words[:3]).lower()
            has_transition = any(tw in sentence_start for tw in transition_words)
            
            # Check sentence length - very short sentences often continue previous thought
            is_short = len(words) < 5
            
            # Start new group on transitions or long content shifts
            if has_transition or (not is_short and len(current_group) >= 3):
                groups.append(current_group)
                current_group = [sentence]
                print(f"Starting new group at sentence {i}")
            else:
                current_group.append(sentence)
                
        # Add the last group
        if current_group:
            groups.append(current_group)
            
        print(f"Created {len(groups)} sentence groups")
        
        # Combine groups into chunks of appropriate size
        improved_chunks = []
        current_combined = []
        current_size = 0
        
        for group_idx, group in enumerate(groups):
            group_text = ' '.join(group)
            group_size = len(group_text)
            
            print(f"Group {group_idx}: {len(group)} sentences, {group_size} chars")
            
            if current_size + group_size > target_size and current_combined:
                improved_chunks.append(' '.join(current_combined))
                print(f"Created improved chunk {len(improved_chunks)}")
                current_combined = [group_text]
                current_size = group_size
            else:
                current_combined.append(group_text)
                current_size += group_size
                
        # Add the last combined group
        if current_combined:
            improved_chunks.append(' '.join(current_combined))
            print(f"Created final improved chunk {len(improved_chunks)}")
            
        print(f"Created {len(improved_chunks)} improved chunks")
        
    except Exception as e:
        print(f"ERROR in improve_chunk_coherence: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("=== DOCUMENT CHUNKING DIAGNOSTIC TOOL ===")
    print(f"Sample document length: {len(SAMPLE_DOCUMENT)} characters")
    
    # Run diagnostic tests
    debug_tokenization()
    debug_chunk_by_tokens()
    debug_improve_chunk_coherence()
    
    print("\nDiagnostic complete") 