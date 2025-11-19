"""
Perplexity Evaluation Module

Computes sample-based perplexity proxy using a small language model (GPT-2) to assess
linguistic quality of the dataset.
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers or torch not available. Perplexity evaluation will be disabled.")


def _determine_device(device_preference: str = "auto") -> str:
    """Determine which device to use for model inference."""
    if not TRANSFORMERS_AVAILABLE:
        return "cpu"
    
    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_preference


def _load_model_and_tokenizer(model_name: str = "gpt2", device: str = "auto") -> Tuple[Any, Any]:
    """
    Load GPT-2 model and tokenizer.
    
    Args:
        model_name: HuggingFace model name (default: "gpt2")
        device: Device to use ("auto", "cpu", "cuda", "mps")
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers and torch are required for perplexity evaluation")
    
    device_str = _determine_device(device)
    
    logger.info(f"Loading {model_name} model on {device_str}...")
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to(device_str)
        model.eval()  # Set to evaluation mode
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Successfully loaded {model_name} on {device_str}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def _compute_sample_perplexity(
    text: str,
    model: Any,
    tokenizer: Any,
    device: str,
    max_length: int = 1024
) -> Optional[float]:
    """
    Compute perplexity for a single text sample.
    
    Args:
        text: Input text string
        model: Loaded GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: Device string ("cpu", "cuda", "mps")
        max_length: Maximum sequence length to process
        
    Returns:
        Perplexity score or None if computation fails
    """
    if not text or not text.strip():
        return None
    
    try:
        # Tokenize the text
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        input_ids = encodings.input_ids.to(device)
        
        # Skip if sequence is too short (need at least 2 tokens for perplexity)
        if input_ids.size(1) < 2:
            return None
        
        # Compute log-likelihood
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # Negative log-likelihood is in outputs.loss
            # Perplexity = exp(loss)
            loss = outputs.loss.item()
            perplexity = np.exp(loss)
        
        return perplexity
        
    except Exception as e:
        logger.warning(f"Failed to compute perplexity for sample: {e}")
        return None


def compute_perplexity(
    samples: List[Dict[str, Any]],
    model_name: str = "gpt2",
    sample_size: int = 100,
    device: str = "auto",
    max_length: int = 1024,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute sample-based perplexity proxy using GPT-2 small.
    
    Args:
        samples: List of processed sample dictionaries (must have 'text' field)
        model_name: HuggingFace model name (default: "gpt2")
        sample_size: Number of samples to evaluate
        device: Device preference ("auto", "cpu", "cuda", "mps")
        max_length: Maximum sequence length for model (default: 1024)
        seed: Random seed for sampling
        
    Returns:
        Dictionary containing:
        - enabled: bool
        - model_name: str
        - samples_evaluated: int
        - mean_perplexity: float
        - median_perplexity: float
        - std_perplexity: float
        - min_perplexity: float
        - max_perplexity: float
        - p25_perplexity: float
        - p75_perplexity: float
        - per_sample_scores: List[float]
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers/torch not available. Skipping perplexity evaluation.")
        return {
            "enabled": False,
            "error": "transformers or torch not available"
        }
    
    if not samples:
        logger.warning("No samples provided for perplexity evaluation.")
        return {
            "enabled": False,
            "error": "no samples provided"
        }
    
    # Sample subset of samples
    sample_size = min(sample_size, len(samples))
    random.seed(seed)
    sampled_indices = random.sample(range(len(samples)), sample_size)
    sampled_samples = [samples[i] for i in sampled_indices]
    
    logger.info(f"Computing perplexity on {sample_size} samples using {model_name}...")
    
    # Load model and tokenizer
    try:
        device_str = _determine_device(device)
        model, tokenizer = _load_model_and_tokenizer(model_name, device_str)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "enabled": False,
            "error": str(e)
        }
    
    # Compute perplexity for each sample
    per_sample_scores = []
    successful_evaluations = 0
    
    for idx, sample in enumerate(sampled_samples):
        text = sample.get('text', '')
        if not text:
            continue
        
        perplexity = _compute_sample_perplexity(
            text, model, tokenizer, device_str, max_length
        )
        
        if perplexity is not None:
            per_sample_scores.append(perplexity)
            successful_evaluations += 1
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{sample_size} samples...")
    
    if not per_sample_scores:
        logger.warning("No valid perplexity scores computed.")
        return {
            "enabled": True,
            "model_name": model_name,
            "samples_evaluated": 0,
            "error": "no valid scores computed"
        }
    
    # Compute aggregate statistics
    scores_array = np.array(per_sample_scores)
    
    metrics = {
        "enabled": True,
        "model_name": model_name,
        "samples_evaluated": successful_evaluations,
        "mean_perplexity": float(np.mean(scores_array)),
        "median_perplexity": float(np.median(scores_array)),
        "std_perplexity": float(np.std(scores_array)),
        "min_perplexity": float(np.min(scores_array)),
        "max_perplexity": float(np.max(scores_array)),
        "p25_perplexity": float(np.percentile(scores_array, 25)),
        "p75_perplexity": float(np.percentile(scores_array, 75)),
        "per_sample_scores": [float(x) for x in per_sample_scores]
    }
    
    logger.info(
        f"Perplexity evaluation complete: "
        f"mean={metrics['mean_perplexity']:.2f}, "
        f"median={metrics['median_perplexity']:.2f}, "
        f"std={metrics['std_perplexity']:.2f}"
    )
    
    return metrics


__all__ = ["compute_perplexity"]

