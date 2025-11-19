"""
Tokenizer Module

Tokenizes text and filters by token length.
"""

import logging
import tiktoken
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


def tokenize_samples(
    samples: List[Dict[str, Any]],
    encoding_name: str = "cl100k_base",
    min_token_length: int = 10,
    max_token_length: int = 8192
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Tokenize samples and filter by token length.
    
    Args:
        samples: List of sample dictionaries
        encoding_name: Tiktoken encoding name
        min_token_length: Minimum token count
        max_token_length: Maximum token count
        
    Returns:
        Tuple of (tokenized_samples, tokenization_metrics):
        - tokenized_samples: List of tokenized samples with token counts
        - tokenization_metrics: Dictionary with tokenization statistics
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.error(f"Error loading encoding {encoding_name}: {e}")
        raise
    
    tokenized_samples = []
    dropped_samples = 0
    token_lengths = []
    drop_reason_counts = Counter()
    
    logger.info(f"Tokenizing {len(samples)} samples...")
    
    for sample in samples:
        text = sample.get('text', '')
        if not text:
            dropped_samples += 1
            drop_reason_counts['empty_text'] += 1
            sample['drop_reason'] = 'empty_text'
            continue
        
        # Tokenize
        try:
            tokens = enc.encode(text)
            token_count = len(tokens)
            token_lengths.append(token_count)
            
            # Filter by token length
            if token_count < min_token_length:
                dropped_samples += 1
                drop_reason_counts['too_few_tokens'] += 1
                sample['drop_reason'] = 'too_few_tokens'
                continue
            
            if token_count > max_token_length:
                dropped_samples += 1
                drop_reason_counts['too_many_tokens'] += 1
                sample['drop_reason'] = 'too_many_tokens'
                continue
            
            # Add token information
            sample['tokens'] = tokens
            sample['token_count'] = token_count
            sample['drop_reason'] = None
            tokenized_samples.append(sample)
            
        except Exception as e:
            logger.warning(f"Error tokenizing sample: {e}")
            dropped_samples += 1
            drop_reason_counts['tokenization_error'] += 1
            sample['drop_reason'] = 'tokenization_error'
            continue
    
    # Log statistics
    if token_lengths:
        avg_tokens = sum(token_lengths) / len(token_lengths)
        min_tokens = min(token_lengths)
        max_tokens = max(token_lengths)
        median_tokens = int(np.median(token_lengths)) if token_lengths else 0
        logger.info(f"Tokenized samples: {len(tokenized_samples)}/{len(samples)}")
        logger.info(f"Dropped samples: {dropped_samples} ({dropped_samples/len(samples)*100:.2f}%)")
        logger.info(f"Token statistics - Avg: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
    else:
        avg_tokens = 0
        min_tokens = 0
        max_tokens = 0
        median_tokens = 0
    
    # Create metrics dictionary
    metrics = {
        'total_samples': len(samples),
        'kept_samples': len(tokenized_samples),
        'dropped_samples': dropped_samples,
        'drop_reason_counts': dict(drop_reason_counts),
        'avg_token_length': avg_tokens,
        'median_token_length': median_tokens,
        'min_token_length': min_tokens,
        'max_token_length': max_tokens,
        'token_lengths': token_lengths  # List for box plot visualization
    }
    
    return tokenized_samples, metrics

