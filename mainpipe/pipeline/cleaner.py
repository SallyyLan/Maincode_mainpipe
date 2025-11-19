"""
Text Cleaner Module

Cleans and normalizes text data.
"""

import re
import logging
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def clean_text(
    text: str,
    min_char_length: int = 50,
    max_char_length: int = 100000,
    remove_html: bool = True,
    normalize_whitespace: bool = True,
    lowercase: bool = False
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        min_char_length: Minimum character length
        max_char_length: Maximum character length
        remove_html: Whether to remove HTML tags
        normalize_whitespace: Whether to normalize whitespace
        lowercase: Whether to lowercase the text
        
    Returns:
        Tuple of (cleaned_text, metadata_dict)
    """
    metadata = {
        'drop_reason': None,
        'original_length': len(text),
        'cleaned_length': 0
    }
    
    # Remove HTML/markup
    if remove_html:
        try:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text(separator=' ')
        except Exception as e:
            logger.warning(f"Error parsing HTML: {e}")
    
    # Remove non-printable characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    # Optional lowercase
    if lowercase:
        text = text.lower()
    
    # Check length constraints
    if len(text) < min_char_length:
        metadata['drop_reason'] = 'too_short'
        return None, metadata
    
    if len(text) > max_char_length:
        metadata['drop_reason'] = 'too_long'
        return None, metadata
    
    metadata['cleaned_length'] = len(text)
    return text, metadata


def clean_samples(
    samples: List[Dict[str, Any]],
    min_char_length: int = 50,
    max_char_length: int = 100000,
    remove_html: bool = True,
    normalize_whitespace: bool = True,
    lowercase: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Clean a list of samples.
    
    Args:
        samples: List of sample dictionaries
        min_char_length: Minimum character length
        max_char_length: Maximum character length
        remove_html: Whether to remove HTML tags
        normalize_whitespace: Whether to normalize whitespace
        lowercase: Whether to lowercase the text
        
    Returns:
        Tuple of (cleaned_samples, cleaning_metrics):
        - cleaned_samples: List of cleaned samples with metadata
        - cleaning_metrics: Dictionary with cleaning statistics
    """
    cleaned_samples = []
    drop_reasons = Counter()
    total_samples = len(samples)
    
    logger.info(f"Cleaning {total_samples} samples...")
    
    for sample in samples:
        text = sample.get('text', '')
        if not text:
            drop_reasons['empty_text'] += 1
            continue
        
        cleaned_text, metadata = clean_text(
            text,
            min_char_length=min_char_length,
            max_char_length=max_char_length,
            remove_html=remove_html,
            normalize_whitespace=normalize_whitespace,
            lowercase=lowercase
        )
        
        if cleaned_text is None:
            drop_reasons[metadata['drop_reason']] += 1
            continue
        
        # Update sample with cleaned text and metadata
        cleaned_sample = sample.copy()
        cleaned_sample['text'] = cleaned_text
        cleaned_sample['char_length'] = metadata['cleaned_length']
        cleaned_sample['drop_reason'] = None
        cleaned_samples.append(cleaned_sample)
    
    # Log statistics
    dropped = total_samples - len(cleaned_samples)
    logger.info(f"Cleaned samples: {len(cleaned_samples)}/{total_samples}")
    logger.info(f"Dropped samples: {dropped} ({dropped/total_samples*100:.2f}%)")
    logger.info(f"Drop reasons: {dict(drop_reasons)}")
    
    # Create metrics dictionary
    metrics = {
        'total_samples': total_samples,
        'kept_samples': len(cleaned_samples),
        'dropped_samples': dropped,
        'drop_reasons': dict(drop_reasons)
    }
    
    return cleaned_samples, metrics

