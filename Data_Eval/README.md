# Text Generation Evaluation Script

This document provides guidance on running and interpreting the optimized evaluation script (`eval_script_optimized.py`) for assessing text generation quality.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

```bash
pip install pandas numpy nltk gensim torch transformers scikit-learn tqdm datasets
```

## Running the Script

1. **Basic Usage**:
   ```bash
   python eval_script_optimized.py
   ```

2. **Configuration**:
   The script contains several configurable constants at the top:
   - `NUM_SAMPLES`: Set to -1 for full dataset, or specify a number for testing
   - `BATCH_SIZE`: Default is 32, adjust based on your memory capacity
   - `MAX_TEXT_LENGTH`: Maximum text length for GPT-2 (default: 1024)

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