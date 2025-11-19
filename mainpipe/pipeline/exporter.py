"""
Exporter Module

Exports processed data in various formats: clean shards, tokenized shards, and training-ready formats.
"""

import json
import logging
import numpy as np
import hashlib
import random
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse
import os
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum for a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_tokenizer_info(encoding_name: str) -> Dict[str, Any]:
    """Extract tokenizer name, version, and vocabulary checksum."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
        # Get vocabulary size as a proxy for checksum
        vocab_size = enc.n_vocab if hasattr(enc, 'n_vocab') else len(enc._mergeable_ranks)
        # Create a simple checksum from encoding name and vocab size
        checksum_str = f"{encoding_name}_{vocab_size}"
        checksum = hashlib.md5(checksum_str.encode()).hexdigest()
        
        return {
            "name": encoding_name,
            "version": "tiktoken",  # tiktoken doesn't expose version easily
            "vocabulary_checksum": checksum,
            "vocab_size": vocab_size
        }
    except Exception as e:
        logger.warning(f"Could not get tokenizer info: {e}")
        return {
            "name": encoding_name,
            "version": "unknown",
            "vocabulary_checksum": "unknown",
            "vocab_size": 0
        }


def determine_source_from_url(url: Optional[str]) -> str:
    """Extract source from URL domain or default to 'web'."""
    if not url:
        return "web"
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if domain:
            # Remove www. prefix if present
            domain = domain.replace("www.", "")
            # Take top-level domain
            parts = domain.split(".")
            if len(parts) >= 2:
                return parts[-2] if len(parts) > 2 else domain
        return "web"
    except Exception:
        return "web"


def add_missing_fields(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add missing required fields to samples."""
    for sample in samples:
        # Add source from URL or default to "web"
        if 'source' not in sample:
            sample['source'] = determine_source_from_url(sample.get('url'))
        
        # Add license (default to "unknown")
        if 'license' not in sample:
            sample['license'] = "unknown"
        
        # Add dedupe_hash from content_hash
        if 'dedupe_hash' not in sample and 'content_hash' in sample:
            sample['dedupe_hash'] = sample['content_hash']
        elif 'dedupe_hash' not in sample:
            # Generate hash if missing
            text = sample.get('text', '')
            if text:
                sample['dedupe_hash'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Add pii_types from pii.pii_types
        if 'pii_types' not in sample:
            pii_info = sample.get('pii', {})
            if isinstance(pii_info, dict):
                sample['pii_types'] = pii_info.get('pii_types', [])
            else:
                sample['pii_types'] = []
        
        # Add pii_redacted flag
        if 'pii_redacted' not in sample:
            pii_info = sample.get('pii', {})
            if isinstance(pii_info, dict):
                sample['pii_redacted'] = pii_info.get('has_pii', False)
            else:
                sample['pii_redacted'] = False
    
    return samples


def redact_pii_in_text(text: str) -> str:
    """Redact PII from text for inspection reports."""
    # Simple redaction - replace common PII patterns
    import re
    # Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', '[EMAIL_REDACTED]', text)
    # Phone
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE_REDACTED]', text)
    # Credit card
    text = re.sub(r'\b(?:\d[ -]?){13,16}\b', '[CARD_REDACTED]', text)
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    return text


# ============================================================================
# Pre-condition Validation
# ============================================================================

def validate_preconditions(
    samples: List[Dict[str, Any]],
    tokenizer_name: str,
    min_tokens: int = 20,
    min_chars: int = 80
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validate that all samples have required fields and meet pre-conditions.
    
    Returns:
        Tuple of (validated_samples, validation_report)
    """
    logger.info("Validating pre-conditions for export...")
    
    required_fields = {
        'text': 0,
        'tokens': 0,
        'token_count': 0,
        'language': 0,
        'source': 0,
        'license': 0,
        'char_length': 0,
        'dedupe_hash': 0,
        'pii_types': 0,
        'pii_redacted': 0
    }
    
    missing_fields = {field: 0 for field in required_fields}
    invalid_samples = 0
    validated_samples = []
    
    # First, add missing fields
    samples = add_missing_fields(samples)
    
    for sample in samples:
        is_valid = True
        
        # Check required fields
        for field in required_fields:
            if field not in sample or sample[field] is None:
                missing_fields[field] += 1
                is_valid = False
            else:
                required_fields[field] += 1
        
        # Check min length thresholds
        token_count = sample.get('token_count', 0)
        char_length = sample.get('char_length', 0)
        
        if token_count < min_tokens:
            is_valid = False
            sample['drop_reason'] = f'too_few_tokens_{token_count}'
        
        if char_length < min_chars:
            is_valid = False
            sample['drop_reason'] = f'too_few_chars_{char_length}'
        
        # Check that tokens is a list/array
        tokens = sample.get('tokens', [])
        if not isinstance(tokens, (list, np.ndarray)) or len(tokens) == 0:
            is_valid = False
            sample['drop_reason'] = 'empty_tokens'
        
        # Check that pii_types is a list
        if not isinstance(sample.get('pii_types', []), list):
            sample['pii_types'] = []
        
        if is_valid:
            validated_samples.append(sample)
        else:
            invalid_samples += 1
    
    # Get tokenizer info
    tokenizer_info = get_tokenizer_info(tokenizer_name)
    
    validation_report = {
        'total_samples': len(samples),
        'valid_samples': len(validated_samples),
        'invalid_samples': invalid_samples,
        'missing_fields': {k: v for k, v in missing_fields.items() if v > 0},
        'field_counts': required_fields,
        'min_tokens_threshold': min_tokens,
        'min_chars_threshold': min_chars,
        'tokenizer_info': tokenizer_info
    }
    
    logger.info(f"Pre-condition validation: {len(validated_samples)}/{len(samples)} samples valid")
    
    return validated_samples, validation_report


# ============================================================================
# Consistency Pass
# ============================================================================

def perform_consistency_pass(
    samples: List[Dict[str, Any]],
    tokenizer_name: str,
    eos_token: int = 100257
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform final consistency pass: validate tokens, apply EOS, check redaction.
    
    Returns:
        Tuple of (consistent_samples, consistency_report)
    """
    logger.info("Performing consistency pass...")
    
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
        # Get valid token range (0 to vocab_size-1 typically)
        vocab_size = enc.n_vocab if hasattr(enc, 'n_vocab') else len(enc._mergeable_ranks)
        max_valid_token = vocab_size - 1
    except Exception as e:
        logger.warning(f"Could not load tokenizer for validation: {e}")
        enc = None
        max_valid_token = 100000  # Fallback
    
    consistent_samples = []
    drop_reason_counts = Counter()
    pii_type_counts = Counter()
    duplicate_stats = {
        'total_duplicates': 0,
        'duplicate_rate': 0.0
    }
    
    out_of_vocab_count = 0
    empty_sequences = 0
    
    for sample in samples:
        tokens = sample.get('tokens', [])
        if not tokens or len(tokens) == 0:
            empty_sequences += 1
            drop_reason_counts['empty_sequence'] += 1
            continue
        
        # Check for out-of-vocab tokens
        if enc is not None:
            invalid_tokens = [t for t in tokens if not isinstance(t, (int, np.integer)) or t < 0 or t > max_valid_token]
            if invalid_tokens:
                out_of_vocab_count += 1
                drop_reason_counts['out_of_vocab'] += 1
                continue
        
        # Check PII redaction (basic check - no raw emails, phones, etc.)
        text = sample.get('text', '')
        if text:
            import re
            # Check for unredacted PII patterns
            has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', text))
            has_phone = bool(re.search(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text))
            has_card = bool(re.search(r'\b(?:\d[ -]?){13,16}\b', text))
            
            if has_email or has_phone or has_card:
                # If PII detected but pii_redacted is False, this is a problem
                if not sample.get('pii_redacted', False):
                    drop_reason_counts['unredacted_pii'] += 1
                    continue
        
        # Count PII types
        pii_types = sample.get('pii_types', [])
        if pii_types:
            pii_type_counts.update(pii_types)
        
        # Check duplicates
        if sample.get('is_duplicate', False):
            duplicate_stats['total_duplicates'] += 1
        
        # Check drop reason
        drop_reason = sample.get('drop_reason')
        if drop_reason:
            drop_reason_counts[drop_reason] += 1
            continue
        
        consistent_samples.append(sample)
    
    duplicate_stats['duplicate_rate'] = duplicate_stats['total_duplicates'] / len(samples) if samples else 0.0
    
    consistency_report = {
        'total_samples': len(samples),
        'consistent_samples': len(consistent_samples),
        'dropped_samples': len(samples) - len(consistent_samples),
        'drop_reason_counts': dict(drop_reason_counts),
        'pii_type_counts': dict(pii_type_counts),
        'duplicate_stats': duplicate_stats,
        'out_of_vocab_count': out_of_vocab_count,
        'empty_sequences': empty_sequences
    }
    
    logger.info(f"Consistency pass: {len(consistent_samples)}/{len(samples)} samples passed")
    
    return consistent_samples, consistency_report


# ============================================================================
# Chunking & Block Formation
# ============================================================================

def chunk_and_form_blocks(
    samples: List[Dict[str, Any]],
    block_size: int = 2048,
    eos_token: int = 100257
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Chunk samples using Strategy A: Concatenation → Fixed Blocks.
    
    Concatenate documents with EOS between them, then split into fixed-size blocks.
    
    Returns:
        Tuple of (blocks, chunking_metadata)
    """
    logger.info(f"Chunking {len(samples)} samples into blocks of size {block_size}...")
    
    # Concatenate all tokens with EOS between documents
    all_tokens = []
    sample_metadata = []  # Track which original samples contributed to each block
    
    for sample in samples:
        tokens = sample.get('tokens', [])
        if not tokens:
            continue
        
        # Add document tokens
        all_tokens.extend(tokens)
        # Add EOS token between documents
        all_tokens.append(eos_token)
    
    # Remove trailing EOS if present
    if all_tokens and all_tokens[-1] == eos_token:
        all_tokens.pop()
    
    # Get aggregate metadata from samples (for blocks that span multiple samples)
    languages = [s.get('language', 'en') for s in samples if s.get('language')]
    most_common_language = Counter(languages).most_common(1)[0][0] if languages else 'en'
    sources = [s.get('source', 'web') for s in samples if s.get('source')]
    most_common_source = Counter(sources).most_common(1)[0][0] if sources else 'web'
    total_char_length = sum(s.get('char_length', 0) for s in samples)
    
    # Split into fixed-size blocks
    blocks = []
    block_id = 0
    
    for i in range(0, len(all_tokens), block_size):
        block_tokens = all_tokens[i:i + block_size]
        
        if len(block_tokens) < block_size:
            # Last block might be shorter - pad or skip based on policy
            # For now, we'll include it if it's at least 50% of block_size
            if len(block_tokens) < block_size // 2:
                break
        
        # Create block sample with metadata
        block_sample = {
            'tokens': block_tokens,
            'token_count': len(block_tokens),
            'sample_id': f"block_{block_id:08d}",
            'block_index': block_id,
            'is_continuation': (i > 0),  # All blocks after first are continuations
            'chunk_index': 0,  # Not used in Strategy A, but included for compatibility
            'chunking_strategy': 'concatenation_fixed_blocks',
            # Aggregate metadata (since blocks may span multiple samples)
            'language': most_common_language,
            'source': most_common_source,
            'url': None,  # Can't preserve individual URLs in concatenated blocks
            'char_length': len(block_tokens) * 4  # Rough estimate: ~4 chars per token
        }
        
        blocks.append(block_sample)
        block_id += 1
    
    chunking_metadata = {
        'strategy': 'concatenation_fixed_blocks',
        'block_size': block_size,
        'eos_token': eos_token,
        'total_input_samples': len(samples),
        'total_blocks': len(blocks),
        'total_tokens': len(all_tokens),
        'avg_tokens_per_block': len(all_tokens) / len(blocks) if blocks else 0
    }
    
    logger.info(f"Created {len(blocks)} blocks from {len(samples)} samples")
    
    return blocks, chunking_metadata


# ============================================================================
# Shuffle & Determinism
# ============================================================================

def shuffle_samples(samples: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """Shuffle samples deterministically using a fixed seed."""
    logger.info(f"Shuffling {len(samples)} samples with seed {seed}...")
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    return shuffled


# ============================================================================
# Sharding with Metadata
# ============================================================================

def create_shards_with_metadata(
    blocks: List[Dict[str, Any]],
    output_dir: Path,
    shard_size: int = 10000,
    shard_max_size_mb: int = 500,
    tokenizer_info: Dict[str, Any] = None,
    shuffle_seed: int = 42,
    consistency_report: Dict[str, Any] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create shards with metadata files.
    
    Returns:
        Tuple of (shard_metadata_list, sharding_summary)
    """
    logger.info(f"Creating shards from {len(blocks)} blocks...")
    
    output_path = output_dir / 'training_ready'
    output_path.mkdir(parents=True, exist_ok=True)
    
    shard_metadata_list = []
    num_shards = (len(blocks) + shard_size - 1) // shard_size
    
    total_tokens = 0
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(blocks))
        shard_blocks = blocks[start_idx:end_idx]
        
        shard_filename = f"training_shard_{shard_idx:05d}.jsonl"
        shard_path = output_path / shard_filename
        
        # Calculate statistics
        token_counts = [b['token_count'] for b in shard_blocks]
        shard_tokens = sum(token_counts)
        total_tokens += shard_tokens
        
        # Write shard file
        with open(shard_path, 'w', encoding='utf-8') as f:
            for block in shard_blocks:
                # Export block with required fields
                export_block = {
                    'tokens': block['tokens'],
                    'token_count': block['token_count'],
                    'language': block.get('language', 'en'),
                    'source': block.get('source', 'web'),
                    'url': block.get('url'),
                    'char_length': block.get('char_length', 0),
                    'sample_id': block.get('sample_id', f"block_{shard_idx}_{start_idx}"),
                    'is_continuation': block.get('is_continuation', False),
                    'chunk_index': block.get('chunk_index', 0)
                }
                f.write(json.dumps(export_block, ensure_ascii=False) + '\n')
        
        # Create metadata file
        meta_filename = f"training_shard_{shard_idx:05d}.meta.json"
        meta_path = output_path / meta_filename
        
        # Prepare summaries
        dedupe_summary = {
            'total_samples': len(shard_blocks),
            'duplicate_rate': consistency_report.get('duplicate_stats', {}).get('duplicate_rate', 0.0) if consistency_report else 0.0
        }
        
        pii_summary = {
            'samples_with_pii': sum(1 for b in shard_blocks if b.get('pii_redacted', False)),
            'pii_type_counts': consistency_report.get('pii_type_counts', {}) if consistency_report else {}
        }
        
        drop_reasons_summary = consistency_report.get('drop_reason_counts', {}) if consistency_report else {}
        
        shard_meta = {
            'shard_index': shard_idx,
            'total_shards': num_shards,
            'num_samples': len(shard_blocks),
            'num_tokens': shard_tokens,
            'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'tokenizer': tokenizer_info or {},
            'dedupe_summary': dedupe_summary,
            'pii_summary': pii_summary,
            'drop_reasons_summary': drop_reasons_summary,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0',  # Could be extracted from config
            'shuffle_seed': shuffle_seed
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(shard_meta, f, indent=2, ensure_ascii=False)
        
        shard_metadata_list.append({
            'shard_index': shard_idx,
            'filename': shard_filename,
            'meta_filename': meta_filename,
            'path': str(shard_path),
            'meta_path': str(meta_path),
            'num_samples': len(shard_blocks),
            'num_tokens': shard_tokens
        })
        
        logger.info(f"Created shard {shard_idx + 1}/{num_shards}: {len(shard_blocks)} samples, {shard_tokens} tokens")
    
    sharding_summary = {
        'total_shards': num_shards,
        'total_samples': len(blocks),
        'total_tokens': total_tokens,
        'avg_samples_per_shard': len(blocks) / num_shards if num_shards > 0 else 0,
        'avg_tokens_per_shard': total_tokens / num_shards if num_shards > 0 else 0
    }
    
    return shard_metadata_list, sharding_summary


# ============================================================================
# Mixture Definition
# ============================================================================

def generate_mixture_json(
    shard_metadata_list: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """Generate mixture.json with domain definitions and sampling weights."""
    logger.info("Generating mixture.json...")
    
    # For now, use a single "web" domain
    # In the future, this could be enhanced to detect domains from URLs
    shard_filenames = [s['filename'] for s in shard_metadata_list]
    
    mixture = {
        'domains': [
            {
                'name': 'web',
                'shards': shard_filenames,
                'weight': 1.0,
                'license': 'unknown'
            }
        ],
        'total_shards': len(shard_filenames),
        'total_weight': 1.0
    }
    
    mixture_path = output_dir / 'mixture.json'
    with open(mixture_path, 'w', encoding='utf-8') as f:
        json.dump(mixture, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated mixture.json with {len(shard_filenames)} shards")
    
    return mixture


# ============================================================================
# Validation & QA
# ============================================================================

def perform_validation(
    shard_metadata_list: List[Dict[str, Any]],
    output_dir: Path,
    tokenizer_name: str,
    min_tokens: int = 20
) -> Dict[str, Any]:
    """Perform validation and QA checks on shards."""
    logger.info("Performing validation and QA checks...")
    
    validation_dir = output_dir / 'validation'
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'shard_validations': [],
        'schema_errors': [],
        'token_range_errors': [],
        'pii_recheck_errors': [],
        'read_speed_tests': []
    }
    
    total_samples_audited = 0
    samples_to_audit = 200  # Minimum samples to audit
    
    for shard_meta in shard_metadata_list:
        shard_path = Path(shard_meta['path'])
        shard_errors = []
        
        # Schema validation
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                sample_count = 0
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line)
                        # Check required fields
                        required = ['tokens', 'token_count', 'language', 'source']
                        for field in required:
                            if field not in sample:
                                shard_errors.append(f"Line {line_num}: missing field '{field}'")
                        
                        # Check token range
                        tokens = sample.get('tokens', [])
                        token_count = sample.get('token_count', 0)
                        if len(tokens) != token_count:
                            shard_errors.append(f"Line {line_num}: token_count mismatch")
                        
                        if token_count < min_tokens:
                            validation_results['token_range_errors'].append({
                                'shard': shard_meta['filename'],
                                'line': line_num,
                                'token_count': token_count
                            })
                        
                        sample_count += 1
                    except json.JSONDecodeError as e:
                        shard_errors.append(f"Line {line_num}: JSON decode error: {e}")
                
                # Random audit
                if sample_count > 0:
                    audit_size = min(samples_to_audit // len(shard_metadata_list), sample_count)
                    # Re-read file for audit
                    with open(shard_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        audit_indices = random.sample(range(len(lines)), min(audit_size, len(lines)))
                        for idx in audit_indices:
                            sample = json.loads(lines[idx])
                            # Basic PII re-check
                            text = sample.get('text', '')
                            if text:
                                import re
                                has_pii = bool(
                                    re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', text) or
                                    re.search(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
                                )
                                if has_pii and not sample.get('pii_redacted', False):
                                    validation_results['pii_recheck_errors'].append({
                                        'shard': shard_meta['filename'],
                                        'line': idx + 1
                                    })
                            total_samples_audited += 1
                
        except Exception as e:
            shard_errors.append(f"Error reading shard: {e}")
        
        # Read speed test
        start_time = time.time()
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            read_time = time.time() - start_time
            read_speed = line_count / read_time if read_time > 0 else 0
            validation_results['read_speed_tests'].append({
                'shard': shard_meta['filename'],
                'samples': line_count,
                'read_time_seconds': read_time,
                'samples_per_second': read_speed
            })
        except Exception as e:
            validation_results['read_speed_tests'].append({
                'shard': shard_meta['filename'],
                'error': str(e)
            })
        
        validation_results['shard_validations'].append({
            'shard': shard_meta['filename'],
            'errors': shard_errors,
            'valid': len(shard_errors) == 0
        })
        
        if shard_errors:
            validation_results['schema_errors'].extend(shard_errors)
    
    validation_results['total_samples_audited'] = total_samples_audited
    validation_results['validation_status'] = 'passed' if len(validation_results['schema_errors']) == 0 else 'failed'
    
    # Save validation report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    validation_report_path = validation_dir / f'validation_report_{timestamp}.json'
    with open(validation_report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Validation complete: {validation_results['validation_status']}")
    logger.info(f"Audited {total_samples_audited} samples across {len(shard_metadata_list)} shards")
    
    return validation_results


# ============================================================================
# Dataset Index
# ============================================================================

def generate_dataset_index(
    shard_metadata_list: List[Dict[str, Any]],
    output_dir: Path,
    tokenizer_info: Dict[str, Any],
    mixture: Dict[str, Any],
    validation_results: Dict[str, Any],
    sharding_summary: Dict[str, Any],
    consistency_report: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generate dataset_index.json with all metadata."""
    logger.info("Generating dataset_index.json...")
    
    # Compute checksums for all shards
    shard_paths_with_checksums = []
    for shard_meta in shard_metadata_list:
        shard_path = Path(shard_meta['path'])
        if shard_path.exists():
            checksum = compute_checksum(shard_path)
            shard_paths_with_checksums.append({
                'path': shard_meta['filename'],
                'checksum': checksum,
                'num_samples': shard_meta['num_samples'],
                'num_tokens': shard_meta['num_tokens']
            })
    
    # Safety summary
    safety_summary = {
        'pii_detection_rate': consistency_report.get('pii_type_counts', {}).__len__() / len(shard_metadata_list) if shard_metadata_list else 0.0,
        'pii_type_counts': consistency_report.get('pii_type_counts', {}) if consistency_report else {},
        'duplicate_rate': consistency_report.get('duplicate_stats', {}).get('duplicate_rate', 0.0) if consistency_report else 0.0
    }
    
    dataset_index = {
        'shards': shard_paths_with_checksums,
        'total_tokens': sharding_summary.get('total_tokens', 0),
        'total_samples': sharding_summary.get('total_samples', 0),
        'tokenizer': tokenizer_info,
        'mixture_file': 'mixture.json',
        'pipeline_version': '1.0',
        'build_timestamp': datetime.now().isoformat(),
        'validation_status': validation_results.get('validation_status', 'unknown'),
        'safety_summary': safety_summary,
        'num_shards': len(shard_paths_with_checksums)
    }
    
    dataset_index_path = output_dir / 'dataset_index.json'
    with open(dataset_index_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_index, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated dataset_index.json with {len(shard_paths_with_checksums)} shards")
    
    return dataset_index


# ============================================================================
# Main Export Function
# ============================================================================

def export_training_ready(
    samples: List[Dict[str, Any]],
    output_dir: str,
    config: Dict[str, Any] = None,
    tokenizer_name: str = "cl100k_base"
):
    """
    Export training-ready format following comprehensive specification.
    
    Args:
        samples: List of tokenized sample dictionaries
        output_dir: Output directory
        config: Configuration dictionary with training export settings
        tokenizer_name: Tokenizer encoding name
    """
    if config is None:
        config = {}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    block_size = config.get('training_block_size', 2048)
    eos_token = config.get('training_eos_token', 100257)
    shuffle_seed = config.get('training_shuffle_seed', 42)
    shard_size = config.get('training_shard_size', 10000)
    shard_max_size_mb = config.get('training_shard_max_size_mb', 500)
    min_tokens = config.get('min_token_length', 20)
    min_chars = config.get('min_char_length', 80)
    
    logger.info("=" * 80)
    logger.info("Starting training-ready export")
    logger.info("=" * 80)
    
    # Step 1: Validate pre-conditions
    logger.info("\nStep 1: Validating pre-conditions...")
    validated_samples, validation_report = validate_preconditions(
        samples, tokenizer_name, min_tokens, min_chars
    )
    
    # Step 2: Perform consistency pass
    logger.info("\nStep 2: Performing consistency pass...")
    consistent_samples, consistency_report = perform_consistency_pass(
        validated_samples, tokenizer_name, eos_token
    )
    
    # Step 3: Chunk and form blocks
    logger.info("\nStep 3: Chunking and forming blocks...")
    blocks, chunking_metadata = chunk_and_form_blocks(
        consistent_samples, block_size, eos_token
    )
    
    # Step 4: Shuffle deterministically
    logger.info("\nStep 4: Shuffling samples...")
    shuffled_blocks = shuffle_samples(blocks, shuffle_seed)
    
    # Step 5: Create shards with metadata
    logger.info("\nStep 5: Creating shards with metadata...")
    tokenizer_info = get_tokenizer_info(tokenizer_name)
    shard_metadata_list, sharding_summary = create_shards_with_metadata(
        shuffled_blocks, output_path, shard_size, shard_max_size_mb,
        tokenizer_info, shuffle_seed, consistency_report
    )
    
    # Step 6: Generate mixture.json
    logger.info("\nStep 6: Generating mixture.json...")
    mixture = generate_mixture_json(shard_metadata_list, output_path)
    
    # Step 7: Perform validation
    logger.info("\nStep 7: Performing validation and QA...")
    validation_results = perform_validation(
        shard_metadata_list, output_path, tokenizer_name, min_tokens
    )
    
    # Step 8: Generate dataset_index.json
    logger.info("\nStep 8: Generating dataset_index.json...")
    dataset_index = generate_dataset_index(
        shard_metadata_list, output_path, tokenizer_info, mixture,
        validation_results, sharding_summary, consistency_report
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Training-ready export completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Exported {len(shuffled_blocks)} blocks in {len(shard_metadata_list)} shards")
    logger.info(f"Total tokens: {sharding_summary.get('total_tokens', 0):,}")
    logger.info(f"Output directory: {output_path}")


# ============================================================================
# Legacy Export Functions (kept for backward compatibility)
# ============================================================================

def export_clean_shards(
    samples: List[Dict[str, Any]],
    output_dir: str,
    shard_size: int = 10000,
    compress: bool = False
):
    """
    Export cleaned text samples as sharded JSONL files.
    
    Args:
        samples: List of cleaned sample dictionaries
        output_dir: Output directory
        shard_size: Number of samples per shard
        compress: Whether to compress output (not implemented yet)
    """
    output_path = Path(output_dir) / 'clean_shard'
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_shards = (len(samples) + shard_size - 1) // shard_size
    
    logger.info(f"Exporting {len(samples)} samples to {num_shards} clean shards...")
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(samples))
        shard_samples = samples[start_idx:end_idx]
        
        shard_filename = f"clean_shard_{shard_idx:05d}.jsonl"
        shard_path = output_path / shard_filename
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            for sample in shard_samples:
                # Export only relevant fields for clean shards
                export_sample = {
                    'text': sample.get('text', ''),
                    'url': sample.get('url'),
                    'char_length': sample.get('char_length'),
                    'language': sample.get('language')
                }
                f.write(json.dumps(export_sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported shard {shard_idx + 1}/{num_shards}: {len(shard_samples)} samples to {shard_filename}")
    
    logger.info(f"Exported {num_shards} clean shards to {output_path}")


def export_tokenized_shards(
    samples: List[Dict[str, Any]],
    output_dir: str,
    shard_size: int = 10000
):
    """
    Export tokenized samples as sharded NPZ files.
    
    Args:
        samples: List of tokenized sample dictionaries
        output_dir: Output directory
        shard_size: Number of samples per shard
    """
    output_path = Path(output_dir) / 'tokenized_shard'
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_shards = (len(samples) + shard_size - 1) // shard_size
    
    logger.info(f"Exporting {len(samples)} samples to {num_shards} tokenized shards...")
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(samples))
        shard_samples = samples[start_idx:end_idx]
        
        # Prepare arrays
        tokens_list = []
        token_counts = []
        metadata = []
        
        for sample in shard_samples:
            if 'tokens' in sample:
                tokens_list.append(sample['tokens'])
                token_counts.append(sample.get('token_count', len(sample['tokens'])))
                
                # Store metadata as JSON string
                metadata_dict = {
                    'url': sample.get('url'),
                    'char_length': sample.get('char_length'),
                    'language': sample.get('language'),
                    'original_index': sample.get('original_index')
                }
                metadata.append(json.dumps(metadata_dict))
        
        # Convert to numpy arrays (pad tokens to same length or use variable length)
        # For variable length, we'll store as list of arrays
        max_len = max(len(t) for t in tokens_list) if tokens_list else 0
        
        # Store as list of arrays (variable length)
        tokens_array = np.array([np.array(t, dtype=np.int32) for t in tokens_list], dtype=object)
        token_counts_array = np.array(token_counts, dtype=np.int32)
        metadata_array = np.array(metadata, dtype=object)
        
        shard_filename = f"tokenized_shard_{shard_idx:05d}.npz"
        shard_path = output_path / shard_filename
        
        np.savez_compressed(
            shard_path,
            tokens=tokens_array,
            token_counts=token_counts_array,
            metadata=metadata_array
        )
        
        logger.info(f"Exported shard {shard_idx + 1}/{num_shards}: {len(shard_samples)} samples to {shard_filename}")
    
    logger.info(f"Exported {num_shards} tokenized shards to {output_path}")


def export(
    samples: List[Dict[str, Any]],
    output_dir: str,
    shard_size: int = 10000,
    export_clean: bool = True,
    export_tokenized: bool = True,
    export_training: bool = True,
    compress: bool = False,
    config: Dict[str, Any] = None
):
    """
    Main export function.
    
    Args:
        samples: List of processed sample dictionaries
        output_dir: Output directory
        shard_size: Number of samples per shard
        export_clean: Whether to export clean shards
        export_tokenized: Whether to export tokenized shards
        export_training: Whether to export training-ready shards
        compress: Whether to compress output
        config: Configuration dictionary (for training export)
    """
    logger.info(f"Exporting {len(samples)} samples to {output_dir}...")
    
    if export_clean:
        export_clean_shards(samples, output_dir, shard_size, compress)
    
    if export_tokenized:
        # Only export if samples have tokens
        if any('tokens' in s for s in samples):
            export_tokenized_shards(samples, output_dir, shard_size)
        else:
            logger.warning("Samples do not have tokens, skipping tokenized export")
    
    if export_training:
        # Only export if samples have tokens
        if any('tokens' in s for s in samples):
            tokenizer_name = config.get('tokenizer_encoding', 'cl100k_base') if config else 'cl100k_base'
            export_training_ready(samples, output_dir, config, tokenizer_name)
        else:
            logger.warning("Samples do not have tokens, skipping training-ready export")
    
    logger.info("Export completed")
