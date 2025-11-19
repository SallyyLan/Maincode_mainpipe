# Mainpipe LLM Dataset Pipeline

A containerized, end-to-end Python pipeline for processing LLM training datasets. The pipeline loads JSONL data, filters, cleans, deduplicates, tokenizes, and applies safety checks before exporting training-ready sharded files.

## Features

- **Language Filtering**: Filters English-only samples using language detection
- **Text Cleaning**: Removes HTML, normalizes whitespace, filters by length
- **Deduplication**: Uses MinHash or hash-based methods to remove duplicates
- **Tokenization**: Tokenizes text using OpenAI's tiktoken (cl100k_base encoding)
- **Safety Filtering**: Lightweight, rule-based toxicity + PII guardrail that uses character windows, blocklists, and regex detection
- **Metrics & Visualizations**: Generates histograms, charts, statistics, and throughput summaries in `reports/`
- **Perplexity Evaluation (Optional)**: Sample-based perplexity scoring for QA using GPT-style models
- **Validation & Manifesting**: Block-level QA, schema validation, and signed manifests (`dataset_index.json`, `mixture.json`, validation reports)
- **Multiple Export Formats**: 
  - Clean text shards (`clean_shard/clean_shard_*.jsonl`)
  - Tokenized shards (`tokenized_shard/tokenized_shard_*.npz`)
  - Training-ready exports with metadata + manifest artifacts (`training_ready/training_shard_*.jsonl`)

## Installation

### Local Installation

1. Clone or navigate to the project directory:
```bash
cd mainpipe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Installation

Build the Docker image:
```bash
docker build -t mainpipe .
```

## Usage

### Local Usage

Run the pipeline with a configuration file:
```bash
python run_pipeline.py config.yaml
```

The runner writes detailed stage logs plus throughput metrics to `pipeline.log`.

### Docker Usage

Run the pipeline in a container:
```bash
docker run -v $(pwd):/app/data -v $(pwd)/output:/app/output -v $(pwd)/reports:/app/reports mainpipe config.yaml
```

Or if your data is in the current directory:
```bash
docker run -v $(pwd):/app mainpipe config.yaml
```

## Configuration

Edit `config.yaml` to customize pipeline parameters:

- **Input/Output**: File paths and directories
- **Text Filtering**: Min/max character and token lengths
- **Language Detection**: Confidence threshold and target language
- **Deduplication**: Method (minhash or hash) and parameters
- **Safety Filtering**: Rule-based toxicity scoring thresholds, blocklist customization, character heuristics, PII entity toggles, scan limits, and local window settings
- **Export Settings**: Shard size and export format options
- **Perplexity & QA**: Enable perplexity sampling, control model/device, and toggle validation artifacts
- **Reporting**: Enable/disable visualization generation and adjust throughput recording

### Example Configuration

```yaml
input_file: "Mainpipe Data v1.jsonl"
output_dir: "output"
min_char_length: 50
max_char_length: 100000
min_token_length: 10
max_token_length: 8192
shard_size: 10000
generate_visualizations: true
calculate_perplexity: true
perplexity_model_name: "gpt2"
perplexity_sample_size: 200
perplexity_device: "auto"        # "cpu", "cuda", or "auto"
perplexity_max_length: 1024
perplexity_seed: 42
export_clean_shards: true
export_tokenized_shards: true
export_training_ready: true
```

### Safety-Specific Options

```yaml
safety_enabled: true
safety_max_chars_scanned: 10000        # Upper bound for PII regex scans
safety_toxicity_threshold: 1.5         # Absolute sum of rule weights that gates toxicity
safety_toxicity_char_cap: 768          # Rule heuristics inspect at most 768 chars per sample
safety_toxicity_window_chars: 512      # Window size used when sampling long documents
safety_prefilter_enabled: true
safety_prefilter_safe_skip_toxicity: true
safety_uppercase_ratio_limit: 0.65     # Uppercase ratio before penalties apply
safety_symbol_ratio_limit: 0.45        # Symbol ratio before penalties apply
safety_max_repeated_chars: 4           # Repeated char run length that triggers a penalty
safety_uppercase_penalty: 0.5          # Penalty added when caps ratio is exceeded
safety_symbol_penalty: 0.4             # Penalty added when symbol ratio is exceeded
safety_repeat_penalty: 0.4             # Penalty added when repeated chars are detected
safety_repeated_punct_penalty: 0.3     # Penalty added for !!!/??? sequences
safety_obfuscated_profanity_penalty: 0.6 # Penalty added when obfuscated profanity is detected
safety_blocklist:
  - "hate"
  - "kill"
  - "murder"
safety_enable_pii: true
safety_pii_entities:
  - EMAIL_ADDRESS
  - PHONE_NUMBER
  - CREDIT_CARD
  - SSN
  - IP_ADDRESS
safety_pii_max_matches: 2              # Stop scanning once this many matches are found
```

New knobs worth calling out:
- `safety_blocklist` accepts either a list (weight = 1.0) or a `{token: weight}` mapping.
- `safety_max_chars_scanned` caps how much text is inspected for PII; `safety_toxicity_char_cap` limits the toxicity heuristics to a smaller local window for speed.
- `safety_*_penalty` entries let you bias the heuristic scoring without code changes.
- `safety_pii_entities` toggles specific regexes so you only pay for what you need.
- `toxicity_window_stride`, `toxicity_window_padding`, and `safety_toxicity_max_windows` (see `config.yaml`) control how much overlap the safety windows use on long documents.

### Perplexity & Validation Options

```yaml
calculate_perplexity: true
perplexity_model_name: "gpt2-medium"
perplexity_sample_size: 128
perplexity_device: "cuda"          # falls back to CPU automatically
perplexity_max_length: 1024

export_training_ready: true
training_block_size: 2048
training_shard_size: 10000
training_shuffle_seed: 42
training_shard_max_size_mb: 500
```

Perplexity sampling is optional; when enabled the pipeline logs sample coverage and writes summary stats into `reports/metrics.json`. Training-ready export always executes the validation + manifest steps (dataset index, mixture, validation report) so downstream consumers can verify shards independently.

## Pipeline Steps

1. **Load**: Reads JSONL file and filters by language
2. **Clean**: Removes HTML, normalizes text, filters by length
3. **Safety Filter**: Runs toxicity + PII heuristics with optional prefilter skip
4. **Deduplicate**: Removes duplicates using configurable MinHash/hash strategies
5. **Tokenize**: Tokenizes text, enforces token-length bounds
6. **Perplexity (Optional)**: Samples documents to compute model perplexity for QA
7. **Inspect**: Writes metrics, visualizations, throughput summaries, and safety stats
8. **Export & QA**: Writes clean/tokenized/training-ready shards, runs shard validation, and generates manifests (`mixture.json`, `dataset_index.json`, validation reports)

## Output Structure

```
output/
├── clean_shard/
│   ├── clean_shard_00000.jsonl        # Cleaned text shards
│   └── clean_shard_00001.jsonl
├── tokenized_shard/
│   ├── tokenized_shard_00000.npz      # Tokenized numpy arrays
│   └── tokenized_shard_00001.npz
├── training_ready/
│   ├── training_shard_00000.jsonl     # Training-ready format + metadata
│   └── training_shard_00001.jsonl
├── validation/
│   └── validation_report_*.json       # Schema + PII audit results
├── dataset_index.json                 # Signed manifest with shard checksums
└── mixture.json                       # Mixture weights + export metadata

reports/
├── metrics.json                    # Exported metrics
├── metrics.csv                     # Exported metrics (CSV)
├── token_length_histogram.png
├── char_length_histogram.png
├── language_scores_histogram.png
├── duplicate_markers.png
├── pii_hit_rates.png
└── drop_reasons_summary.png
```

`reports/metrics.json` still contains the aggregate dataset stats, including a `safety_pipeline` section, throughput summaries per stage, and (when enabled) perplexity evaluation results. Validation artifacts (`output/validation/*.json`, `dataset_index.json`, `mixture.json`) allow downstream consumers to verify shard integrity before training.

## Dependencies

- **Core**: pandas, pyyaml
- **Language Detection**: langdetect
- **Text Cleaning**: beautifulsoup4, lxml
- **Deduplication**: datasketch
- **Tokenization**: tiktoken
- **Safety**: built-in regex + heuristics (no extra dependencies)
- **Visualization**: matplotlib, seaborn
- **ML**: (none by default — install project-specific extras as needed)

## Scaling Considerations

For larger datasets, consider:

- **Ray**: Distributed processing for parallel map/filter operations
- **Streaming**: Replace single-file I/O with streaming from S3/GCS
- **Caching**: Cache intermediate stages for resume/fail recovery
- **Optimized I/O**: Use DuckDB or Parquet for better I/O performance
- **Parallel Tokenization**: Use parallel BPE tokenizer implementations

See `reports/report.md` for detailed scaling strategies.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

