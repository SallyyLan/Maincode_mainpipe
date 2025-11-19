"""
Data Loader Module

Loads JSONL data and filters by language.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
from langdetect import detect, DetectorFactory

logger = logging.getLogger(__name__)

# Set seed for reproducible results
DetectorFactory.seed = 0


def load_jsonl(file_path: str, target_language: str = "en") -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Load JSONL file and filter by language using langdetect library.
    
    Args:
        file_path: Path to JSONL file
        target_language: Target language code (default: "en")
        
    Returns:
        Tuple of (samples, language_distribution):
        - samples: List of samples with metadata including language detection results
        - language_distribution: Dictionary mapping language codes to counts
    """
    samples = []
    language_distribution = Counter()
    total_samples = 0
    english_samples = 0
    
    logger.info(f"Loading JSONL file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num} lines...")
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    url = data.get('url')
                    
                    if not text:
                        continue
                    
                    total_samples += 1
                    
                    # Detect language using langdetect
                    try:
                        detected_lang = detect(text)
                        language_distribution[detected_lang] += 1
                        
                        # Filter for target language
                        if detected_lang == target_language:
                            english_samples += 1
                            samples.append({
                                'text': text,
                                'url': url,
                                'language': detected_lang,
                                'original_index': total_samples - 1
                            })
                    except Exception as e:
                        logger.warning(f"Could not detect language for sample {line_num}: {e}")
                        language_distribution['unknown'] += 1
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise
    
    # Log statistics
    logger.info(f"Total samples loaded: {total_samples}")
    logger.info(f"English samples: {english_samples} ({english_samples/total_samples*100:.2f}%)")
    logger.info(f"Language distribution: {dict(language_distribution.most_common(10))}")
    
    return samples, dict(language_distribution)

