"""
Deduplicator Module

Removes duplicate samples using MinHash or hash-based methods.
"""

import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


def canonicalize_text(text: str) -> str:
    """
    Normalize text for dedupe / caching.

    Lowercase, trim, collapse whitespace, and strip simple punctuation repeats.
    """
    if not text:
        return ""
    text = text.strip().lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove repeated punctuation sequences (e.g., \"!!!\")
    text = re.sub(r"([!?,.])\\1+", r"\\1", text)
    return text


def compute_content_hash(text: str) -> str:
    """Compute SHA256 of canonical text."""
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def ensure_sample_fingerprint(sample: Dict[str, Any]) -> Optional[str]:
    """
    Ensure a sample carries canonical text and hash, returning the hash.
    """
    text = sample.get("text", "")
    if not text:
        return None

    canonical = sample.get("canonical_text")
    if not canonical:
        canonical = canonicalize_text(text)
        sample["canonical_text"] = canonical

    content_hash = sample.get("content_hash")
    if not content_hash and canonical:
        content_hash = compute_content_hash(canonical)
        sample["content_hash"] = content_hash

    return content_hash


def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """
    Create a MinHash for a text sample.
    
    Args:
        text: Input text
        num_perm: Number of permutations for MinHash
        
    Returns:
        MinHash object
    """
    m = MinHash(num_perm=num_perm)
    # Split text into shingles (3-grams)
    words = text.split()
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])
        m.update(shingle.encode('utf8'))
    return m


def deduplicate_minhash(samples: List[Dict[str, Any]], num_perm: int = 128) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplicate samples using MinHash with LSH (Locality Sensitive Hashing).
    
    Uses MinHashLSH to achieve O(n) complexity instead of O(n²) by using
    hash-based indexing to only compare similar documents.
    
    Args:
        samples: List of sample dictionaries
        num_perm: Number of permutations for MinHash
        
    Returns:
        Tuple of (deduplicated_samples, metrics_dict):
        - deduplicated_samples: List of deduplicated samples
        - metrics_dict: Dictionary with deduplication statistics
    """
    # Initialize LSH index with threshold 0.9 (Jaccard similarity)
    # LSH automatically handles the indexing for efficient similarity search
    lsh = MinHashLSH(threshold=0.9, num_perm=num_perm)
    
    # Store MinHash objects for candidates that pass LSH query
    minhash_store = {}
    deduplicated = []
    duplicates = 0
    
    logger.info(f"Deduplicating {len(samples)} samples using MinHash with LSH...")
    
    for i, sample in enumerate(samples):
        # Progress logging every 5k samples for better visibility
        if i % 5000 == 0 and i > 0:
            logger.info(f"Processed {i} samples, found {duplicates} duplicates so far...")
        
        text = sample.get('text', '')
        if not text:
            continue

        ensure_sample_fingerprint(sample)
        canonical_text = sample.get('canonical_text', text)

        m = create_minhash(canonical_text, num_perm)
        
        # Query LSH for candidate duplicates (returns indices of similar documents)
        candidates = lsh.query(m)
        
        # Check candidates for actual similarity (Jaccard > 0.9)
        is_duplicate = False
        if candidates:
            # Verify similarity with exact Jaccard calculation
            for candidate_idx in candidates:
                candidate_m = minhash_store[candidate_idx]
                similarity = m.jaccard(candidate_m)
                if similarity > 0.9:
                    is_duplicate = True
                    duplicates += 1
                    break
        
        if not is_duplicate:
            # Insert into LSH index and store MinHash
            lsh.insert(i, m)
            minhash_store[i] = m
            sample['is_duplicate'] = False
            deduplicated.append(sample)
        else:
            sample['is_duplicate'] = True
    
    logger.info(f"Deduplicated samples: {len(deduplicated)}/{len(samples)}")
    logger.info(f"Duplicates removed: {duplicates} ({duplicates/len(samples)*100:.2f}%)")
    
    # Create metrics dictionary
    metrics = {
        'total_samples': len(samples),
        'kept_samples': len(deduplicated),
        'duplicates_removed': duplicates,
        'duplicate_rate': duplicates / len(samples) if len(samples) > 0 else 0.0
    }
    
    return deduplicated, metrics


def deduplicate_hash(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplicate samples using exact hash matching.
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Tuple of (deduplicated_samples, metrics_dict):
        - deduplicated_samples: List of deduplicated samples
        - metrics_dict: Dictionary with deduplication statistics
    """
    seen_hashes = set()
    deduplicated = []
    duplicates = 0
    
    logger.info(f"Deduplicating {len(samples)} samples using hash-based method...")
    
    for sample in samples:
        text = sample.get('text', '')
        if not text:
            continue

        text_hash = ensure_sample_fingerprint(sample)
        if text_hash is None:
            continue
        
        if text_hash in seen_hashes:
            duplicates += 1
            sample['is_duplicate'] = True
        else:
            seen_hashes.add(text_hash)
            sample['is_duplicate'] = False
            deduplicated.append(sample)
    
    logger.info(f"Deduplicated samples: {len(deduplicated)}/{len(samples)}")
    logger.info(f"Duplicates removed: {duplicates} ({duplicates/len(samples)*100:.2f}%)")
    
    # Create metrics dictionary
    metrics = {
        'total_samples': len(samples),
        'kept_samples': len(deduplicated),
        'duplicates_removed': duplicates,
        'duplicate_rate': duplicates / len(samples) if len(samples) > 0 else 0.0
    }
    
    return deduplicated, metrics


def deduplicate(
    samples: List[Dict[str, Any]],
    method: str = "minhash",
    num_perm: int = 128
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplicate samples using specified method.
    
    Args:
        samples: List of sample dictionaries
        method: Deduplication method ("minhash" or "hash")
        num_perm: Number of permutations for MinHash (only used if method="minhash")
        
    Returns:
        Tuple of (deduplicated_samples, metrics_dict):
        - deduplicated_samples: List of deduplicated samples
        - metrics_dict: Dictionary with deduplication statistics
    """
    if method is None or str(method).lower() == "none":
        logger.info("Skipping deduplication step (method='none').")
        for sample in samples:
            ensure_sample_fingerprint(sample)
            sample['is_duplicate'] = False
        metrics = {
            'total_samples': len(samples),
            'kept_samples': len(samples),
            'duplicates_removed': 0,
            'duplicate_rate': 0.0
        }
        return samples, metrics
    method = str(method).lower()
    if method == "minhash":
        return deduplicate_minhash(samples, num_perm)
    elif method == "hash":
        return deduplicate_hash(samples)
    else:
        raise ValueError(f"Unknown deduplication method: {method}")

