# Text Generation Evaluation Script

This document provides guidance on running and interpreting the optimized evaluation script (`eval_script_optimized.py`) for assessing text generation quality.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

```bash
pip install pandas numpy nltk gensim torch transformers scikit-learn tqdm datasets
```

## Running the Script

The script now uses a configuration file approach for easier management of evaluation settings.

1. **Basic Usage**:
   ```bash
   python eval_script_optimized.py --config config.json
   ```

2. **Configuration File**:
   Create a `config.json` file with your desired settings. Here's a template:
   ```json
   {
       "evaluation": {
           "type": "both",
           "data_files": {
               "real": "path/to/real_data.csv",
               "synthetic": "path/to/synthetic_data.csv"
           },
           "sampling": {
               "num_samples": -1,
               "batch_size": 32,
               "perplexity": {
                   "enabled": true,
                   "sample_size": 1000,
                   "random_seed": 42
               },
               "coherence": {
                   "enabled": true,
                   "sample_size": 5000,
                   "random_seed": 42
               }
           }
       },
       "model_settings": {
           "max_text_length": 1024,
           "device": "cuda",
           "lda": {
               "num_topics": 10,
               "passes": 5
           }
       },
       "logging": {
           "log_dir": "logs",
           "console_verbosity": "minimal"
       }
   }
   ```

3. **Configuration Options**:
   - `evaluation.type`: Choose "real", "synthetic", or "both"
   - `data_files`: Paths to your input data files (CSV or JSON)
   - `sampling`:
     - `num_samples`: Total samples to use (-1 for all)
     - `batch_size`: Processing batch size
     - `perplexity`: Settings for perplexity evaluation sampling
     - `coherence`: Settings for topic coherence evaluation sampling
   - `model_settings`: Model-specific parameters
   - `logging`: Logging preferences

4. **Sampling Control**:
   - Both perplexity and coherence evaluations support random sampling
   - Enable/disable sampling independently
   - Configure sample sizes and random seeds
   - Helps reduce computation time while maintaining statistical significance

## Output Structure

The script creates two types of output:

### 1. Logs Directory
- Located in `./logs/`
- Contains detailed timestamped logs (`eval_log_YYYYMMDD_HHMMSS.txt`)
- Tracks progress, warnings, and errors
- Useful for debugging and monitoring long runs

### 2. Results File
- Named `evaluation_results_YYYYMMDD_HHMMSS.json`
- Contains structured results from all evaluations
- JSON format for easy parsing and analysis

## Understanding the Results

The evaluation consists of four main components:

### 1. Topic Coherence
```json
"topic_coherence": {
    "real": 0.XXXX,
    "synthetic": 0.XXXX
}
```
- Higher scores indicate more coherent and interpretable topics
- Compare synthetic score to real score (closer = better)
- Typical range: 0.3 to 0.7

### 2. Statistical Properties
```json
"statistical_properties": {
    "real_avg_length": XXX.XX,
    "synth_avg_length": XXX.XX,
    "real_ttr": 0.XXXX,
    "synth_ttr": 0.XXXX
}
```
- `avg_length`: Average document length in words
- `ttr`: Type-Token Ratio (vocabulary richness)
  - Higher = more diverse vocabulary
  - Should be similar between real and synthetic

### 3. Perplexity
```json
"perplexity": {
    "real": XXX.XX,
    "synthetic": XXX.XX
}
```
- Lower scores indicate more natural language
- Synthetic should be close to real
- GPT-2 baseline on natural text: ~20-60

### 4. Classification Results
```json
"classification": {
    "precision": X.XX,
    "recall": X.XX,
    "f1-score": X.XX,
    "accuracy": X.XX
}
```
- Closer to 0.5 accuracy is better
- Shows how distinguishable synthetic text is from real
- Perfect score of 0.5 means indistinguishable

## Memory Management

The script includes several memory optimization features:
- Batch processing for large datasets
- Regular garbage collection
- CUDA memory management for GPU usage
- Progress bars for long-running operations

## Configuration Best Practices

1. **Sampling Strategies**:
   - For large datasets (>10k texts):
     - Enable perplexity sampling with 1000-2000 samples
     - Enable coherence sampling with 3000-5000 samples
   - For small datasets (<1000 texts):
     - Disable sampling to use full dataset
   - Always use the same random seed for reproducibility

2. **Performance Optimization**:
   - Adjust `batch_size` based on available memory
   - Use smaller sample sizes for initial testing
   - Enable GPU acceleration when available
   - Set appropriate `max_text_length` for your use case

3. **Common Issues**:
   - If running out of memory:
     - Reduce batch size
     - Enable sampling with smaller sample sizes
     - Reduce max_text_length
   - If evaluation is too slow:
     - Enable sampling
     - Increase batch size (if memory allows)
     - Use GPU acceleration

4. **Version Control**:
   - Keep different config files for different experiments
   - Name config files descriptively (e.g., `eval_config_large.json`)
   - Document sampling parameters used in experiments

## Troubleshooting

Common issues and solutions:

1. **Out of Memory (OOM)**:
   - Reduce `BATCH_SIZE`
   - Reduce `NUM_SAMPLES` for testing
   - Free up GPU memory

2. **Slow Processing**:
   - Increase `BATCH_SIZE` if memory allows
   - Ensure GPU is being utilized (check DEVICE output)
   - Close other memory-intensive applications

3. **CUDA Errors**:
   - Script will automatically fall back to CPU
   - Ensure latest CUDA drivers are installed
   - Monitor GPU memory usage

## Best Practices

1. **Testing New Data**:
   - Start with small `NUM_SAMPLES` (~100)
   - Gradually increase to full dataset
   - Monitor memory usage

2. **Long Runs**:
   - Use screen or tmux for long runs
   - Monitor the logs directory
   - Check intermediate results

3. **Results Analysis**:
   - Compare all metrics between real and synthetic
   - Look for consistent patterns
   - Consider multiple runs for stability

## Contributing

When modifying the script:
1. Maintain the batch processing structure
2. Add appropriate logging
3. Include error handling
4. Clean up resources properly
5. Update this documentation as needed

## Data sources and configuration

This section explains where the evaluation script obtains data by default and how to configure both real and synthetic data sources.

1. Default real data (Hugging Face):
    - By default, the notebook and the optimized script call `load_dataset("cnn_dailymail", "3.0.0", split="train")` to fetch the CNN/DailyMail training split from the Hugging Face `datasets` library.
    - In the notebook this is invoked through the `load_real_data` function. In the optimized script the same function `load_real_data(num_samples, logger)` performs this step.
    - To change the amount of data used for testing, set the `NUM_SAMPLES` constant near the top of `eval_script_optimized.py` to an integer (e.g., `NUM_SAMPLES = 1000`) or `-1` to use the entire split.

2. Using a local file for REAL data (CSV/JSON/NDJSON):
    - If you want to use your own real dataset instead of the Hugging Face dataset, replace the body of `load_real_data` with code that reads your file. Example (CSV with an `article` column):

       ```python
       import pandas as pd

       def load_real_data(num_samples, logger):
             logger.info('Loading real data from local CSV...')
             df = pd.read_csv('path/to/real_articles.csv')
             texts = df['article'].tolist()
             if num_samples != -1:
                   texts = texts[:num_samples]
             return texts
       ```

    - For JSON/NDJSON, use `pd.read_json('path/to/file.json', lines=True)` or iterate over the file to extract the text field.

3. Configuring SYNTHETIC data
    - The optimized script currently creates a placeholder synthetic dataset by modifying the real data (`text.replace("CNN", "GMN")`). Replace that block with code that loads your generated samples.
    - Example: load generated samples from `generated.csv` where the generated article column is `generated_article`:

       ```python
       import pandas as pd

       gemma_df = pd.read_csv('path/to/generated.csv')
       synthetic_articles = gemma_df['generated_article'].dropna().astype(str).tolist()
       if NUM_SAMPLES != -1:
             synthetic_articles = synthetic_articles[:NUM_SAMPLES]
       ```

    - If your synthetic outputs are stored in multiple files or a directory, you can aggregate them using `glob`:

       ```python
       import glob
       synthetic_files = glob.glob('path/to/synthetic_dir/*.json')
       synthetic_articles = []
       for f in synthetic_files:
             df = pd.read_json(f, lines=True)
             synthetic_articles.extend(df['generated_article'].astype(str).tolist())
       ```

4. Tips for matching dataset sizes and stratification
    - Several evaluation steps (e.g., classification) assume roughly balanced classes. If your synthetic set is much smaller than real, consider sampling or duplicating examples to avoid class imbalance. For example:

       ```python
       if len(synthetic_articles) < len(real_articles):
             synthetic_articles = synthetic_articles * (len(real_articles)//len(synthetic_articles))
             synthetic_articles = synthetic_articles[:len(real_articles)]
       ```

5. Environment / config file pattern
    - For reproducible runs, consider adding a small `config.yaml` or `params.json` that contains `NUM_SAMPLES`, paths to real/synthetic data, `BATCH_SIZE`, and `DEVICE`. Read it at the top of `eval_script_optimized.py` and override constants.

6. Example minimal change to `main()` in `eval_script_optimized.py` to load local files:

```python
# replace the synthetic loading block
real_articles = load_real_data(NUM_SAMPLES, logger)
# Load real from CSV instead by modifying load_real_data as shown earlier
gemma_df = pd.read_csv('data/my_generated.csv')
synthetic_articles = gemma_df['generated_article'].dropna().astype(str).tolist()
```

7. Validation and quick checks
    - Before running the full evaluation, validate both lists:

    ```python
    print('Real count:', len(real_articles))
    print('Synthetic count:', len(synthetic_articles))
    print('Sample real:', real_articles[0][:200])
    print('Sample synthetic:', synthetic_articles[0][:200])
    ```

    - This helps catch parsing or encoding issues early.


### New CLI options (updated)

The optimized script now supports command-line arguments so you can choose which datasets to evaluate and supply local files.

- `--evaluate {real,synthetic,both}` (default: `real`) — choose which dataset(s) to evaluate. By default the script evaluates only the Hugging Face CNN/DailyMail real dataset.
- `--real-file PATH` — path to a local real data file (CSV or line-delimited JSON) with an `article` column/field.
- `--synthetic-file PATH` — path to a local synthetic data file (CSV or line-delimited JSON) with a `generated_article` column/field.
- `--num-samples N` — number of samples to use (`-1` for all). Matches the `NUM_SAMPLES` setting.
- `--batch-size N` — batch size for processing. Matches the `BATCH_SIZE` setting.

Example: evaluate only the CNN real dataset (default):

```powershell
python eval_script_optimized.py
```

Example: evaluate both datasets, providing a synthetic CSV file:

```powershell
python eval_script_optimized.py --evaluate both --synthetic-file data/generated.csv --num-samples 1000
```

If you pass `--real-file` or `--synthetic-file`, the script will attempt to read CSV files (expecting `article` or `generated_article` columns respectively) or JSON (line-delimited JSON with those fields).

If `--evaluate` is left at the default (`real`) the script will not attempt to run the classifier (which requires both datasets) and will only compute fidelity metrics (topic coherence, statistical properties, perplexity) for the real CNN/DailyMail data.