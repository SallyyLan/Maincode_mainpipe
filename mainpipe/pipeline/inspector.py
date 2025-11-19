"""
Inspector Module

Generates metrics and visualizations for the processed dataset.
"""

import json
import csv
import logging
from typing import List, Dict, Any
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    import numpy as _np

    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_make_json_serializable(v) for v in obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def calculate_statistics(
    samples: List[Dict[str, Any]], 
    safety_metrics: Dict[str, Any] = None,
    deduplication_metrics: Dict[str, Any] = None,
    cleaning_metrics: Dict[str, Any] = None,
    tokenization_metrics: Dict[str, Any] = None,
    perplexity_metrics: Dict[str, Any] = None,
    language_distribution: Dict[str, int] = None,
    throughput_metrics: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculate statistics from processed samples.
    
    Args:
        samples: List of processed sample dictionaries (KEPT samples only)
        safety_metrics: Safety filtering metrics dictionary
        deduplication_metrics: Deduplication metrics dictionary
        cleaning_metrics: Cleaning metrics dictionary
        tokenization_metrics: Tokenization metrics dictionary
        perplexity_metrics: Perplexity evaluation metrics dictionary
        
    Returns:
        Dictionary of statistics with clear metric definitions
    """
    stats = {
        'total_samples': len(samples),
        'token_lengths': [],
        'char_lengths': [],
        'pii_hit_rates': Counter(),
        'drop_reasons': Counter(),
        'toxicity_scores': []
    }
    
    # Calculate statistics from KEPT samples only
    for sample in samples:
        # Token lengths
        if 'token_count' in sample:
            stats['token_lengths'].append(sample['token_count'])
        
        # Character lengths
        if 'char_length' in sample:
            stats['char_lengths'].append(sample['char_length'])
        elif 'text' in sample:
            stats['char_lengths'].append(len(sample['text']))
        
        # PII hit rates (from kept samples - will be empty if PII samples were dropped)
        if 'pii' in sample and sample['pii'].get('has_pii', False):
            for pii_type in sample['pii'].get('pii_types', []):
                stats['pii_hit_rates'][pii_type] += 1
        
        # Drop reasons (from kept samples - will be empty if dropped samples were removed)
        if 'drop_reason' in sample and sample['drop_reason']:
            stats['drop_reasons'][sample['drop_reason']] += 1
        
        # Toxicity scores
        if 'toxicity' in sample:
            score = sample['toxicity'].get('score', 0.0)
            stats['toxicity_scores'].append(score)
    
    # Calculate summary statistics with clear sample set labels
    # Use safety_metrics for more accurate rates if available
    if safety_metrics and safety_metrics.get('enabled', False):
        total_for_rates = safety_metrics.get('total_samples', stats['total_samples'])
        toxic_dropped = safety_metrics.get('toxic_dropped', 0)
        pii_dropped = safety_metrics.get('pii_dropped', 0)
        pii_detection_rate = pii_dropped / total_for_rates if total_for_rates > 0 else 0.0
        toxicity_rate = toxic_dropped / total_for_rates if total_for_rates > 0 else 0.0
    else:
        # Fallback to calculating from sample fields
        pii_detection_rate = sum(stats['pii_hit_rates'].values()) / stats['total_samples'] if stats['total_samples'] > 0 else 0
        # Count samples with toxicity score above threshold
        toxicity_rate = len([s for s in stats['toxicity_scores'] if s >= 1.5]) / stats['total_samples'] if stats['total_samples'] > 0 else 0
    
    # Use deduplication_metrics if available, otherwise calculate from samples
    if deduplication_metrics:
        duplicate_rate = deduplication_metrics.get('duplicate_rate', 0.0)
    else:
        duplicate_rate = sum(stats['duplicate_markers']) / len(stats['duplicate_markers']) if stats['duplicate_markers'] else 0
    
    summary = {
        'total_samples': stats['total_samples'],  # KEPT samples only
        'duplicate_rate': duplicate_rate,
        'pii_detection_rate': pii_detection_rate,  # Based on safety_total_samples
        'toxicity_rate': toxicity_rate,  # Based on safety_total_samples
        '_metric_definitions': {
            'total_samples': 'Number of samples in final output (KEPT samples only)',
            'avg_char_length': 'Average character length of KEPT samples',
            'avg_token_length': 'Average token count of KEPT samples',
            'avg_chars_scanned': 'Average characters scanned during safety filtering (ALL samples entering safety stage)',
            'pii_type_counts': 'Count of PII type OCCURRENCES (not samples). A single sample can have multiple PII types.',
            'pii_dropped': 'Number of SAMPLES dropped due to PII (a sample with multiple PII types counts as 1)',
            'drop_reason_counts': 'Aggregated count of samples dropped across all pipeline stages with stage prefixes (cleaning_, safety_, tokenization_, deduplication_)',
        }
    }
    
    # Token length stats (from KEPT samples)
    if stats['token_lengths']:
        summary['avg_token_length'] = np.mean(stats['token_lengths'])
        summary['median_token_length'] = np.median(stats['token_lengths'])
        summary['min_token_length'] = np.min(stats['token_lengths'])
        summary['max_token_length'] = np.max(stats['token_lengths'])
    
    # Character length stats (from KEPT samples)
    if stats['char_lengths']:
        summary['avg_char_length'] = np.mean(stats['char_lengths'])
        summary['median_char_length'] = np.median(stats['char_lengths'])
    
    # PII type counts: Use safety_metrics (from ALL samples entering safety) instead of kept samples
    # This is important because samples with PII were dropped, so they're not in the kept samples list
    if safety_metrics and safety_metrics.get('pii_type_counts'):
        # pii_type_counts counts OCCURRENCES of PII types, not samples
        # A single sample can have multiple PII types (e.g., both EMAIL and PHONE)
        summary['pii_type_counts'] = safety_metrics.get('pii_type_counts', {})
    else:
        # Fallback to kept samples (will be empty if PII samples were dropped)
        summary['pii_type_counts'] = dict(stats['pii_hit_rates'])
    
    # Aggregate drop reason counts from all pipeline stages with stage prefixes
    # This provides a unified view of all drop reasons across the pipeline
    aggregated_drop_reasons = {}
    
    # Cleaning stage drops
    if cleaning_metrics and cleaning_metrics.get('drop_reasons'):
        for reason, count in cleaning_metrics.get('drop_reasons', {}).items():
            aggregated_drop_reasons[f'cleaning_{reason}'] = count
    else:
        # Fallback to kept samples (will be empty if dropped samples were removed)
        for reason, count in dict(stats['drop_reasons']).items():
            if reason in ['too_short', 'too_long']:  # Only cleaning reasons
                aggregated_drop_reasons[f'cleaning_{reason}'] = count
    
    # Safety stage drops
    if safety_metrics and safety_metrics.get('enabled', False):
        toxic_dropped = safety_metrics.get('toxic_dropped', 0)
        pii_dropped = safety_metrics.get('pii_dropped', 0)
        if toxic_dropped > 0:
            aggregated_drop_reasons['safety_toxic'] = toxic_dropped
        if pii_dropped > 0:
            aggregated_drop_reasons['safety_contains_pii'] = pii_dropped
    
    # Tokenization stage drops
    if tokenization_metrics and tokenization_metrics.get('drop_reason_counts'):
        for reason, count in tokenization_metrics.get('drop_reason_counts', {}).items():
            aggregated_drop_reasons[f'tokenization_{reason}'] = count
    
    # Deduplication stage drops
    if deduplication_metrics:
        duplicates_removed = deduplication_metrics.get('duplicates_removed', 0)
        if duplicates_removed > 0:
            aggregated_drop_reasons['deduplication_duplicate'] = duplicates_removed
    
    summary['drop_reason_counts'] = aggregated_drop_reasons
    
    # Include perplexity metrics if available
    if perplexity_metrics:
        summary['perplexity_pipeline'] = perplexity_metrics
    
    if language_distribution:
        total_lang = sum(language_distribution.values())
        lang_percentages = {
            lang: (count / total_lang) if total_lang > 0 else 0.0
            for lang, count in language_distribution.items()
        }
        summary['language_distribution'] = {
            "counts": language_distribution,
            "percentages": lang_percentages
        }
    
    if throughput_metrics:
        summary['throughput_metrics'] = throughput_metrics
    
    return summary


def plot_token_length_histogram(token_lengths: List[int], output_path: str):
    """Plot histogram of token lengths."""
    plt.figure(figsize=(12, 6))
    plt.hist(token_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Token Lengths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved token length histogram to {output_path}")


def plot_char_length_histogram(char_lengths: List[int], output_path: str):
    """Plot histogram of character lengths."""
    plt.figure(figsize=(12, 6))
    plt.hist(char_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Character Lengths (After Processing)')
    plt.xlim(0, 20000)  # Limit x-axis to 20000
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved character length histogram to {output_path}")


def plot_pii_hit_rates(pii_type_counts: Dict[str, int], output_path: str):
    """Plot bar chart of PII hit rates by type."""
    if not pii_type_counts:
        logger.warning("No PII data to plot")
        return
    
    plt.figure(figsize=(12, 6))
    types = list(pii_type_counts.keys())
    counts = list(pii_type_counts.values())
    
    plt.bar(types, counts, edgecolor='black', alpha=0.7)
    plt.xlabel('PII Type')
    plt.ylabel('Count')
    plt.title('PII Detection Rates by Type')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PII hit rates chart to {output_path}")


def plot_drop_reasons(drop_reason_counts: Dict[str, int], output_path: str):
    """Plot summary of drop reasons."""
    if not drop_reason_counts:
        logger.warning("No drop reasons to plot")
        return
    
    plt.figure(figsize=(12, 6))
    reasons = list(drop_reason_counts.keys())
    counts = list(drop_reason_counts.values())
    
    plt.barh(reasons, counts, edgecolor='black', alpha=0.7)
    plt.xlabel('Count')
    plt.ylabel('Drop Reason')
    plt.title('Sample Drop Reasons Summary')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved drop reasons summary to {output_path}")


def plot_language_distribution(language_distribution: Dict[str, int], output_path: str):
    """Plot bar chart of top 5 languages by count."""
    if not language_distribution:
        logger.warning("No language distribution data to plot")
        return
    
    # Sort by count descending and keep only top 5
    sorted_langs = sorted(language_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
    languages = [lang for lang, _ in sorted_langs]
    counts = [count for _, count in sorted_langs]
    
    plt.figure(figsize=(12, 6))
    plt.bar(languages, counts, edgecolor='black', alpha=0.7)
    plt.xlabel('Language Code')
    plt.ylabel('Count')
    plt.title('Top 5 Languages Distribution (Before Filtering)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved language distribution to {output_path}")


def plot_toxic_removals(flagged_token_counts: Dict[str, int], output_path: str):
    """Plot horizontal bar chart of toxic removal reasons sorted by count."""
    if not flagged_token_counts:
        logger.warning("No toxic removal data to plot")
        return
    
    # Sort by count ascending for barh (highest at top - last item in list appears at top)
    sorted_tokens = sorted(flagged_token_counts.items(), key=lambda x: x[1], reverse=False)
    tokens = [token for token, _ in sorted_tokens]
    counts = [count for _, count in sorted_tokens]
    
    plt.figure(figsize=(10, max(6, len(tokens) * 0.3)))
    bars = plt.barh(tokens, counts, edgecolor='black', alpha=0.7, color='#e74c3c')
    plt.xlabel('Count')
    plt.ylabel('Flagged Token')
    plt.title('Toxic Removal Reasons (Sorted by Count)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        plt.text(count, bar.get_y() + bar.get_height()/2, f' {count:,}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved toxic removals chart to {output_path}")


def plot_pii_removals(pii_type_counts: Dict[str, int], output_path: str):
    """Plot horizontal bar chart of PII removal reasons sorted by count."""
    if not pii_type_counts:
        logger.warning("No PII removal data to plot")
        return
    
    # Sort by count ascending for barh (highest at top - last item in list appears at top)
    sorted_pii = sorted(pii_type_counts.items(), key=lambda x: x[1], reverse=False)
    pii_types = [pii_type for pii_type, _ in sorted_pii]
    counts = [count for _, count in sorted_pii]
    
    plt.figure(figsize=(10, max(6, len(pii_types) * 0.4)))
    bars = plt.barh(pii_types, counts, edgecolor='black', alpha=0.7, color='#f39c12')
    plt.xlabel('Count')
    plt.ylabel('PII Type')
    plt.title('PII Removal Reasons (Sorted by Count)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        plt.text(count, bar.get_y() + bar.get_height()/2, f' {count:,}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PII removals chart to {output_path}")


def plot_cleaning_drop_reasons(drop_reasons_dict: Dict[str, int], output_path: str):
    """Plot vertical bar chart of cleaning drop reasons."""
    if not drop_reasons_dict:
        logger.warning("No cleaning drop reasons to plot")
        return
    
    reasons = list(drop_reasons_dict.keys())
    counts = list(drop_reasons_dict.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(reasons, counts, edgecolor='black', alpha=0.7, color=['#e67e22', '#c0392b'])
    plt.xlabel('Drop Reason')
    plt.ylabel('Count')
    plt.title('Cleaning Drop Reasons')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, count, f'{count:,}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cleaning drop reasons to {output_path}")


def plot_tokenization_stats(token_lengths: List[int], output_path: str):
    """Plot box plot of tokenization statistics with annotations."""
    if not token_lengths:
        logger.warning("No token length data to plot")
        return
    
    avg = np.mean(token_lengths)
    median = np.median(token_lengths)
    min_val = np.min(token_lengths)
    max_val = np.max(token_lengths)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([token_lengths], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    # Add annotations
    stats_text = f'Avg: {avg:.1f}\nMedian: {median:.1f}\nMin: {min_val}\nMax: {max_val}'
    ax.text(1.15, ax.get_ylim()[1] * 0.9, stats_text, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='top')
    
    ax.set_ylabel('Token Count')
    ax.set_title('Tokenization Statistics Distribution')
    ax.set_xticklabels(['Token Lengths'])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved tokenization statistics to {output_path}")


def plot_perplexity_distribution(per_sample_scores: List[float], output_path: str):
    """Plot histogram of perplexity distribution with statistics annotations."""
    if not per_sample_scores:
        logger.warning("No perplexity data to plot")
        return
    
    scores_array = np.array(per_sample_scores)
    mean_score = np.mean(scores_array)
    median_score = np.median(scores_array)
    std_score = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    p25_score = np.percentile(scores_array, 25)
    p75_score = np.percentile(scores_array, 75)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(per_sample_scores, bins=50, edgecolor='black', alpha=0.7, color='#9b59b6')
    
    # Add vertical lines for statistics
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    ax.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
    ax.axvline(p25_score, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'P25: {p25_score:.2f}')
    ax.axvline(p75_score, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'P75: {p75_score:.2f}')
    
    # Add statistics text box
    stats_text = (
        f'Mean: {mean_score:.2f}\n'
        f'Median: {median_score:.2f}\n'
        f'Std: {std_score:.2f}\n'
        f'Min: {min_score:.2f}\n'
        f'Max: {max_score:.2f}\n'
        f'P25: {p25_score:.2f}\n'
        f'P75: {p75_score:.2f}'
    )
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right')
    
    ax.set_xlabel('Perplexity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Perplexity Distribution (Sample-Based Proxy)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved perplexity distribution to {output_path}")


def plot_throughput_overview(
    stage_entries: List[Dict[str, Any]],
    overall_stats: Dict[str, Any],
    output_path: str
):
    """Plot throughput metrics across pipeline stages."""
    if not stage_entries:
        logger.warning("No throughput metrics to plot")
        return
    
    stages = [entry.get('stage', 'stage') for entry in stage_entries]
    samples_per_sec = [entry.get('samples_per_sec', 0.0) for entry in stage_entries]
    durations = [entry.get('duration_sec', 0.0) for entry in stage_entries]
    processed = [entry.get('processed_samples', 0) for entry in stage_entries]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(stages, samples_per_sec, color='#1abc9c', edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Samples per Second')
    ax1.set_xlabel('Pipeline Stage')
    ax1.set_title('Pipeline Throughput by Stage')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Annotate bars with processed samples
    for bar, count, duration in zip(bars, processed, durations):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,} samples\n{duration:.1f}s",
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Add overall stats text box if available
    if overall_stats:
        text_lines = [
            f"Total Duration: {overall_stats.get('total_duration_sec', 0.0):.1f}s",
            f"Final Samples: {overall_stats.get('final_samples', 0):,}",
            f"Avg Throughput: {overall_stats.get('average_samples_per_sec', 0.0):.2f}/s",
            f"Peak Stage Throughput: {overall_stats.get('peak_stage_samples_per_sec', 0.0):.2f}/s"
        ]
        ax1.text(
            0.98,
            0.95,
            "\n".join(text_lines),
            transform=ax1.transAxes,
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved throughput overview to {output_path}")


def plot_metrics_table(all_metrics: Dict[str, Any], output_path: str):
    """Plot metrics summary table visualization."""
    # Prepare table data
    table_data = []
    
    # Extract key metrics
    if 'deduplication_metrics' in all_metrics and all_metrics['deduplication_metrics']:
        dedup = all_metrics['deduplication_metrics']
        table_data.append(['Samples After Deduplication', f"{dedup.get('kept_samples', 0):,}"])
        table_data.append(['Duplicates Removed', f"{dedup.get('duplicates_removed', 0):,}"])
        table_data.append(['Duplicate Rate', f"{dedup.get('duplicate_rate', 0.0)*100:.2f}%"])
    
    if 'cleaning_metrics' in all_metrics and all_metrics['cleaning_metrics']:
        clean = all_metrics['cleaning_metrics']
        table_data.append(['Samples After Cleaning', f"{clean.get('kept_samples', 0):,}"])
    
    if 'tokenization_metrics' in all_metrics and all_metrics['tokenization_metrics']:
        tok = all_metrics['tokenization_metrics']
        table_data.append(['Samples After Tokenization', f"{tok.get('kept_samples', 0):,}"])
        table_data.append(['Avg Token Length', f"{tok.get('avg_token_length', 0):.1f}"])
        table_data.append(['Median Token Length', f"{tok.get('median_token_length', 0):.1f}"])
    
    if 'safety_metrics' in all_metrics and all_metrics['safety_metrics']:
        safety = all_metrics['safety_metrics']
        if safety.get('enabled', False):
            table_data.append(['Samples After Safety Filtering', f"{safety.get('kept_samples', 0):,}"])
            table_data.append(['Toxic Samples Dropped', f"{safety.get('toxic_dropped', 0):,}"])
            table_data.append(['PII Samples Dropped', f"{safety.get('pii_dropped', 0):,}"])
            table_data.append(['PII Detection Rate', f"{all_metrics.get('pii_detection_rate', 0.0)*100:.2f}%"])
            table_data.append(['Toxicity Rate', f"{all_metrics.get('toxicity_rate', 0.0)*100:.2f}%"])
    
    if 'language_distribution' in all_metrics and all_metrics['language_distribution']:
        lang_dist = all_metrics['language_distribution']
        if isinstance(lang_dist, dict) and 'counts' in lang_dist:
            counts = lang_dist['counts']
        else:
            counts = lang_dist
        total_lang = sum(counts.values())
        table_data.append(['Total Samples Loaded', f"{total_lang:,}"])
        table_data.append(['Samples After Language Filtering', f"{counts.get('en', 0):,}"])
    
    table_data.append(['Final Output Samples', f"{all_metrics.get('total_samples', 0):,}"])
    
    # Calculate retention rate
    if 'language_distribution' in all_metrics and all_metrics['language_distribution']:
        lang_dist = all_metrics['language_distribution']
        if isinstance(lang_dist, dict) and 'counts' in lang_dist:
            counts = lang_dist['counts']
        else:
            counts = lang_dist
        total_input = sum(counts.values())
        final_output = all_metrics.get('total_samples', 0)
        if total_input > 0:
            retention_rate = (final_output / total_input) * 100
            table_data.append(['Retention Rate', f"{retention_rate:.2f}%"])
    
    if 'throughput_metrics' in all_metrics and all_metrics['throughput_metrics']:
        overall = all_metrics['throughput_metrics'].get('overall', {})
        table_data.append([
            'Avg Throughput',
            f"{overall.get('average_samples_per_sec', 0.0):.2f} samples/sec"
        ])
        table_data.append([
            'Peak Stage Throughput',
            f"{overall.get('peak_stage_samples_per_sec', 0.0):.2f} samples/sec"
        ])
    
    if not table_data:
        logger.warning("No metrics data available for table")
        return
    
    # Create table
    fig, ax = plt.subplots(figsize=(12, max(8, len(table_data) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Pipeline Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics table to {output_path}")


def generate_visualizations(
    samples: List[Dict[str, Any]], 
    reports_dir: str,
    language_distribution: Dict[str, int] = None,
    cleaning_metrics: Dict[str, Any] = None,
    deduplication_metrics: Dict[str, Any] = None,
    tokenization_metrics: Dict[str, Any] = None,
    safety_metrics: Dict[str, Any] = None,
    perplexity_metrics: Dict[str, Any] = None,
    throughput_metrics: Dict[str, Any] = None
):
    """
    Generate all visualizations.
    
    Args:
        samples: List of processed sample dictionaries
        reports_dir: Directory to save visualizations
        language_distribution: Language distribution dictionary from loader
        cleaning_metrics: Cleaning metrics dictionary
        deduplication_metrics: Deduplication metrics dictionary
        tokenization_metrics: Tokenization metrics dictionary
        safety_metrics: Safety filtering metrics dictionary
        perplexity_metrics: Perplexity evaluation metrics dictionary
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations subdirectory
    viz_path = reports_path / 'visualizations'
    viz_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating visualizations...")
    
    # Extract data for plotting
    token_lengths = [s.get('token_count', 0) for s in samples if 'token_count' in s]
    char_lengths = [s.get('char_length', len(s.get('text', ''))) for s in samples]
    
    pii_type_counts = Counter()
    for sample in samples:
        if 'pii' in sample and sample['pii'].get('has_pii', False):
            for pii_type in sample['pii'].get('pii_types', []):
                pii_type_counts[pii_type] += 1
    
    drop_reason_counts = Counter()
    for sample in samples:
        if 'drop_reason' in sample and sample['drop_reason']:
            drop_reason_counts[sample['drop_reason']] += 1
    
    # Generate existing plots (move to visualizations subdirectory)
    if token_lengths:
        plot_token_length_histogram(token_lengths, str(viz_path / 'token_length_histogram.png'))
    
    if char_lengths:
        plot_char_length_histogram(char_lengths, str(viz_path / 'char_length_histogram.png'))
    
    if pii_type_counts:
        plot_pii_hit_rates(dict(pii_type_counts), str(viz_path / 'pii_hit_rates.png'))
    
    if drop_reason_counts:
        plot_drop_reasons(dict(drop_reason_counts), str(viz_path / 'drop_reasons_summary.png'))
    
    # Generate new visualizations
    # 1. Language distribution (pre-filtering) - top 5 only
    if language_distribution:
        plot_language_distribution(language_distribution, str(viz_path / 'language_distribution.png'))
    
    # 2. Safety removal reasons
    if safety_metrics and safety_metrics.get('enabled', False):
        flagged_token_counts = safety_metrics.get('flagged_token_counts', {})
        if flagged_token_counts:
            plot_toxic_removals(flagged_token_counts, str(viz_path / 'toxic_removals.png'))
        
        pii_type_counts_safety = safety_metrics.get('pii_type_counts', {})
        if pii_type_counts_safety:
            plot_pii_removals(pii_type_counts_safety, str(viz_path / 'pii_removals.png'))
    
    # 3. Cleaning drop reasons
    if cleaning_metrics:
        drop_reasons = cleaning_metrics.get('drop_reasons', {})
        if drop_reasons:
            plot_cleaning_drop_reasons(drop_reasons, str(viz_path / 'cleaning_drop_reasons.png'))
    
    # 4. Perplexity distribution
    if perplexity_metrics and perplexity_metrics.get('enabled', False):
        per_sample_scores = perplexity_metrics.get('per_sample_scores', [])
        if per_sample_scores:
            plot_perplexity_distribution(per_sample_scores, str(viz_path / 'perplexity_distribution.png'))
    
    # 5. Throughput overview
    if throughput_metrics and throughput_metrics.get('stages'):
        plot_throughput_overview(
            throughput_metrics.get('stages', []),
            throughput_metrics.get('overall', {}),
            str(viz_path / 'throughput_overview.png')
        )
    
    # 6. Metrics table
    all_metrics = {
        'total_samples': len(samples),
        'language_distribution': language_distribution,
        'cleaning_metrics': cleaning_metrics,
        'deduplication_metrics': deduplication_metrics,
        'tokenization_metrics': tokenization_metrics,
        'safety_metrics': safety_metrics,
        'pii_detection_rate': safety_metrics.get('pii_dropped', 0) / safety_metrics.get('total_samples', 1) if safety_metrics and safety_metrics.get('total_samples', 0) > 0 else 0.0,
        'toxicity_rate': safety_metrics.get('toxic_dropped', 0) / safety_metrics.get('total_samples', 1) if safety_metrics and safety_metrics.get('total_samples', 0) > 0 else 0.0,
        'throughput_metrics': throughput_metrics
    }
    plot_metrics_table(all_metrics, str(viz_path / 'metrics_summary_table.png'))


def export_metrics(
    samples: List[Dict[str, Any]], 
    reports_dir: str, 
    safety_metrics: Dict[str, Any] = None,
    deduplication_metrics: Dict[str, Any] = None,
    cleaning_metrics: Dict[str, Any] = None,
    tokenization_metrics: Dict[str, Any] = None,
    perplexity_metrics: Dict[str, Any] = None,
    language_distribution: Dict[str, int] = None,
    throughput_metrics: Dict[str, Any] = None
):
    """
    Export metrics to JSON and CSV.
    
    Args:
        samples: List of processed sample dictionaries
        reports_dir: Directory to save metrics
        safety_metrics: Safety filtering metrics dictionary
        deduplication_metrics: Deduplication metrics dictionary
        cleaning_metrics: Cleaning metrics dictionary
        tokenization_metrics: Tokenization metrics dictionary
        perplexity_metrics: Perplexity evaluation metrics dictionary
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Calculating and exporting metrics...")
    
    stats = calculate_statistics(
        samples, 
        safety_metrics=safety_metrics, 
        deduplication_metrics=deduplication_metrics,
        cleaning_metrics=cleaning_metrics,
        tokenization_metrics=tokenization_metrics,
        perplexity_metrics=perplexity_metrics,
        language_distribution=language_distribution,
        throughput_metrics=throughput_metrics
    )
    if safety_metrics:
        stats['safety_pipeline'] = safety_metrics
    if tokenization_metrics:
        stats['tokenization_pipeline'] = tokenization_metrics
    
    # Export JSON
    json_path = reports_path / 'metrics.json'
    serializable_stats = _make_json_serializable(stats)
    with open(json_path, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    logger.info(f"Exported metrics to {json_path}")
    
    # Export CSV (flattened)
    csv_path = reports_path / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in stats.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    writer.writerow([f"{key}.{sub_key}", sub_value])
            elif isinstance(value, list):
                writer.writerow([key, len(value)])
            else:
                writer.writerow([key, value])
    logger.info(f"Exported metrics to {csv_path}")


def inspect(
    samples: List[Dict[str, Any]],
    reports_dir: str,
    generate_plots: bool = True,
    safety_metrics: Dict[str, Any] = None,
    language_distribution: Dict[str, int] = None,
    cleaning_metrics: Dict[str, Any] = None,
    deduplication_metrics: Dict[str, Any] = None,
    tokenization_metrics: Dict[str, Any] = None,
    perplexity_metrics: Dict[str, Any] = None,
    throughput_metrics: Dict[str, Any] = None
):
    """
    Main inspection function.
    
    Args:
        samples: List of processed sample dictionaries
        reports_dir: Directory to save reports
        generate_plots: Whether to generate visualizations
        safety_metrics: Safety filtering metrics dictionary
        language_distribution: Language distribution dictionary from loader
        cleaning_metrics: Cleaning metrics dictionary
        deduplication_metrics: Deduplication metrics dictionary
        tokenization_metrics: Tokenization metrics dictionary
        perplexity_metrics: Perplexity evaluation metrics dictionary
    """
    if generate_plots:
        generate_visualizations(
            samples, 
            reports_dir,
            language_distribution=language_distribution,
            cleaning_metrics=cleaning_metrics,
            deduplication_metrics=deduplication_metrics,
            tokenization_metrics=tokenization_metrics,
            safety_metrics=safety_metrics,
            perplexity_metrics=perplexity_metrics,
            throughput_metrics=throughput_metrics
        )
    
    export_metrics(
        samples, 
        reports_dir, 
        safety_metrics=safety_metrics, 
        deduplication_metrics=deduplication_metrics,
        cleaning_metrics=cleaning_metrics,
        tokenization_metrics=tokenization_metrics,
        perplexity_metrics=perplexity_metrics,
        language_distribution=language_distribution,
        throughput_metrics=throughput_metrics
    )

