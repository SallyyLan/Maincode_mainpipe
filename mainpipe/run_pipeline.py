#!/usr/bin/env python3
"""
Main Pipeline Runner

Orchestrates the entire data processing pipeline.
"""

import os
import sys
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List

from pipeline.loader import load_jsonl
from pipeline.cleaner import clean_samples
from pipeline.deduplicator import deduplicate
from pipeline.tokenizer import tokenize_samples
from pipeline.safety import filter_samples
from pipeline.perplexity import compute_perplexity
from pipeline.inspector import inspect
from pipeline.exporter import export

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _record_throughput_stage(
    throughput_metrics: Dict[str, Any],
    stage_name: str,
    processed_samples: int,
    start_time: float
) -> None:
    """Record throughput metrics for a pipeline stage."""
    if throughput_metrics is None:
        return
    
    duration = max(time.perf_counter() - start_time, 1e-9)
    samples_per_sec = processed_samples / duration if duration > 0 else 0.0
    stage_entry = {
        "stage": stage_name,
        "processed_samples": processed_samples,
        "duration_sec": duration,
        "samples_per_sec": samples_per_sec
    }
    throughput_metrics.setdefault("stages", []).append(stage_entry)
    logger.info(
        "[Throughput] %s handled %s samples in %.2fs (%.2f samples/sec)",
        stage_name,
        f"{processed_samples:,}",
        duration,
        samples_per_sec
    )


def run_pipeline(config_path: str):
    """
    Run the complete pipeline.
    
    Args:
        config_path: Path to configuration YAML file
    """
    logger.info("=" * 80)
    logger.info("Starting Mainpipe LLM Dataset Pipeline")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Get input file path (relative to config file or absolute)
    config_dir = Path(config_path).parent
    input_file = config.get('input_file', 'Mainpipe Data v1.jsonl')
    if not Path(input_file).is_absolute():
        # Try relative to config directory first
        potential_path = config_dir / input_file
        if potential_path.exists():
            input_file = potential_path
        else:
            # Try parent directory (for data files)
            potential_path = config_dir.parent / Path(input_file).name
            if potential_path.exists():
                input_file = potential_path
            else:
                # Use original path and let it fail with clear error
                input_file = config_dir / input_file
    
    output_dir = config.get('output_dir', 'output')
    reports_dir = config.get('reports_dir', 'reports')
    
    # Ensure output directories exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    throughput_metrics: Dict[str, Any] = {"stages": []}
    pipeline_start = time.perf_counter()
    
    try:
        # Step 1: Load data
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Loading JSONL data")
        logger.info("=" * 80)
        stage_start = time.perf_counter()
        samples, language_distribution = load_jsonl(
            str(input_file),
            target_language=config.get('target_language', 'en')
        )
        _record_throughput_stage(
            throughput_metrics,
            "load_jsonl",
            len(samples),
            stage_start
        )
        logger.info(f"Loaded {len(samples)} English samples")
        
        # Step 2: Clean text
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Cleaning text")
        logger.info("=" * 80)
        stage_start = time.perf_counter()
        samples, cleaning_metrics = clean_samples(
            samples,
            min_char_length=config.get('min_char_length', 50),
            max_char_length=config.get('max_char_length', 100000),
            remove_html=True,
            normalize_whitespace=True,
            lowercase=False
        )
        _record_throughput_stage(
            throughput_metrics,
            "clean_text",
            len(samples),
            stage_start
        )
        logger.info(f"After cleaning: {len(samples)} samples")
        
        # Step 3: Safety filtering
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Safety filtering (toxicity & PII)")
        logger.info("=" * 80)
        stage_start = time.perf_counter()
        samples, safety_metrics = filter_samples(
            samples,
            enabled=config.get('safety_enabled', True),
            toxicity_threshold=config.get('safety_toxicity_threshold', 1.5),
            blocklist=config.get('safety_blocklist'),
            max_chars_scanned=config.get('safety_max_chars_scanned', 10000),
            toxicity_char_cap=config.get('safety_toxicity_char_cap', 768),
            toxicity_window_chars=config.get('safety_toxicity_window_chars', 512),
            toxicity_window_stride=config.get('safety_toxicity_window_stride', 0.5),
            toxicity_window_padding=config.get('safety_toxicity_window_padding', 64),
            toxicity_max_windows=config.get('safety_toxicity_max_windows', 3),
            toxicity_suspect_windows=config.get('safety_toxicity_suspect_windows', 5),
            prefilter_enabled=config.get('safety_prefilter_enabled', True),
            prefilter_safe_skip_toxicity=config.get('safety_prefilter_safe_skip_toxicity', True),
            uppercase_ratio_limit=config.get('safety_uppercase_ratio_limit', 0.65),
            symbol_ratio_limit=config.get('safety_symbol_ratio_limit', 0.45),
            max_repeated_chars=config.get('safety_max_repeated_chars', 4),
            uppercase_penalty=config.get('safety_uppercase_penalty', 0.5),
            symbol_penalty=config.get('safety_symbol_penalty', 0.4),
            repeat_penalty=config.get('safety_repeat_penalty', 0.4),
            repeated_punct_penalty=config.get('safety_repeated_punct_penalty', 0.3),
            obfuscated_profanity_penalty=config.get('safety_obfuscated_profanity_penalty', 0.6),
            enable_pii=config.get('safety_enable_pii', True),
            pii_entities=config.get('safety_pii_entities', ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "SSN", "IP_ADDRESS"]),
            pii_max_matches=config.get('safety_pii_max_matches', 2),
        )
        _record_throughput_stage(
            throughput_metrics,
            "safety_filtering",
            len(samples),
            stage_start
        )
        logger.info(f"After safety filtering: {len(samples)} samples")
        
        # Step 4: Deduplicate
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Deduplicating samples")
        logger.info("=" * 80)
        stage_start = time.perf_counter()
        samples, dedup_metrics = deduplicate(
            samples,
            method=config.get('deduplication_method', 'minhash'),
            num_perm=config.get('minhash_num_perm', 128)
        )
        _record_throughput_stage(
            throughput_metrics,
            "deduplication",
            len(samples),
            stage_start
        )
        logger.info(f"After deduplication: {len(samples)} samples")
        
        # Step 5: Tokenize
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Tokenizing samples")
        logger.info("=" * 80)
        stage_start = time.perf_counter()
        samples, tokenization_metrics = tokenize_samples(
            samples,
            encoding_name=config.get('tokenizer_encoding', 'cl100k_base'),
            min_token_length=config.get('min_token_length', 10),
            max_token_length=config.get('max_token_length', 8192)
        )
        _record_throughput_stage(
            throughput_metrics,
            "tokenization",
            len(samples),
            stage_start
        )
        logger.info(f"After tokenization: {len(samples)} samples")
        
        # Step 5.5: Compute perplexity (if enabled)
        perplexity_metrics = None
        if config.get('calculate_perplexity', False):
            logger.info("\n" + "=" * 80)
            logger.info("Step 5.5: Computing perplexity evaluation")
            logger.info("=" * 80)
            stage_start = time.perf_counter()
            perplexity_metrics = compute_perplexity(
                samples,
                model_name=config.get('perplexity_model_name', 'gpt2'),
                sample_size=config.get('perplexity_sample_size', 100),
                device=config.get('perplexity_device', 'auto'),
                max_length=config.get('perplexity_max_length', 1024),
                seed=config.get('perplexity_seed', 42)
            )
            if perplexity_metrics and perplexity_metrics.get('enabled', False):
                logger.info(f"Perplexity evaluation complete: {perplexity_metrics.get('samples_evaluated', 0)} samples evaluated")
                _record_throughput_stage(
                    throughput_metrics,
                    "perplexity_evaluation",
                    perplexity_metrics.get('samples_evaluated', 0),
                    stage_start
                )
            else:
                logger.warning("Perplexity evaluation was not completed")
        
        # Step 6: Generate metrics and visualizations
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Generating metrics and visualizations")
        logger.info("=" * 80)
        inspect(
            samples,
            reports_dir=reports_dir,
            generate_plots=config.get('generate_visualizations', True),
            safety_metrics=safety_metrics,
            language_distribution=language_distribution,
            cleaning_metrics=cleaning_metrics,
            deduplication_metrics=dedup_metrics,
            tokenization_metrics=tokenization_metrics,
            perplexity_metrics=perplexity_metrics,
            throughput_metrics=throughput_metrics
        )
        
        # Step 7: Export
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: Exporting processed data")
        logger.info("=" * 80)
        stage_start = time.perf_counter()
        export(
            samples,
            output_dir=output_dir,
            shard_size=config.get('shard_size', 10000),
            export_clean=config.get('export_clean_shards', True),
            export_tokenized=config.get('export_tokenized_shards', True),
            export_training=config.get('export_training_ready', True),
            compress=config.get('compress_output', False),
            config=config
        )
        _record_throughput_stage(
            throughput_metrics,
            "export",
            len(samples),
            stage_start
        )
        
        total_duration = max(time.perf_counter() - pipeline_start, 1e-9)
        final_samples = len(samples)
        stage_throughputs: List[float] = [
            stage.get('samples_per_sec', 0.0)
            for stage in throughput_metrics.get('stages', [])
        ]
        throughput_metrics['overall'] = {
            "total_duration_sec": total_duration,
            "final_samples": final_samples,
            "average_samples_per_sec": final_samples / total_duration if total_duration > 0 else 0.0,
            "peak_stage_samples_per_sec": max(stage_throughputs) if stage_throughputs else 0.0
        }
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Final sample count: {len(samples)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Reports directory: {reports_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_pipeline(config_path)

