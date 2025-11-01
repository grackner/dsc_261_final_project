"""
Topic Coherence Evaluation Module

Provides functions for evaluating topic coherence of text collections using LDA modeling.
Designed for easy importing and use in notebooks or other scripts.

Example usage:
    from coherence_test import compute_topic_coherence, process_text_collection

    # Using raw texts
    texts = ["document 1", "document 2", ...]
    coherence_result = compute_topic_coherence(texts, sample_size=1000)
    
    # Or with pandas DataFrame
    df = pd.read_csv("data.csv")
    coherence_result = compute_topic_coherence(
        df['text_column'],
        sample_size=5000,
        num_topics=15,
        random_seed=42
    )
"""
from __future__ import annotations

from typing import Union, List, Dict, Optional
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from datetime import datetime
from tqdm import tqdm

TextCollection = Union[List[str], pd.Series, np.ndarray]

def process_text_collection(
    texts: TextCollection,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    silent: bool = False
) -> List[List[str]]:
    """
    Preprocess a collection of texts: tokenize, clean, and optionally sample.
    
    Args:
        texts: Collection of texts (list, pandas Series, or numpy array)
        sample_size: Optional number of texts to randomly sample
        random_seed: Random seed for sampling
        silent: If True, suppress progress bar
    
    Returns:
        List of preprocessed text tokens
    """
    # Convert to list if needed
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    
    # Sample if requested
    if sample_size and len(texts) > sample_size:
        np.random.seed(random_seed)
        texts = list(np.random.choice(texts, size=sample_size, replace=False))
    
    # Preprocess
    stop_words = set(stopwords.words('english'))
    processed = []
    
    iterator = tqdm(texts, desc="Processing texts", disable=silent)
    for text in iterator:
        if not isinstance(text, str):
            continue
        tokens = word_tokenize(text.lower())
        filtered = [
            word for word in tokens
            if word.isalpha() and word not in stop_words 
            and word not in string.punctuation
        ]
        if filtered:
            processed.append(filtered)
            
    return processed

def compute_topic_coherence(
    texts: TextCollection,
    sample_size: Optional[int] = None,
    num_topics: int = 10,
    no_below: int = 5,
    no_above: float = 0.5,
    random_seed: int = 42,
    num_passes: int = 5,
    silent: bool = False
) -> Dict:
    """
    Compute topic coherence score for a collection of texts.
    
    Args:
        texts: Collection of texts to analyze
        sample_size: Optional number of texts to randomly sample
        num_topics: Number of topics for LDA model
        no_below: Min document frequency for dictionary filtering
        no_above: Max document frequency for dictionary filtering
        random_seed: Random seed for reproducibility
        num_passes: Number of passes for LDA training
        silent: If True, suppress progress bars
        
    Returns:
        Dictionary containing:
            - coherence_score: The calculated coherence score
            - model_info: Information about the LDA model
            - data_info: Information about the processed texts
            - parameters: The parameters used for the calculation
            - timestamp: When the analysis was performed
    """
    # Process texts
    processed_texts = process_text_collection(
        texts, 
        sample_size=sample_size,
        random_seed=random_seed,
        silent=silent
    )
    
    if not processed_texts:
        return {
            'coherence_score': 0.0,
            'error': 'No valid texts after processing',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Create dictionary and corpus
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    if not corpus:
        return {
            'coherence_score': 0.0,
            'error': 'No valid documents after dictionary filtering',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_seed,
        passes=num_passes
    )
    
    # Calculate coherence
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=processed_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    coherence_score = coherence_model.get_coherence()
    
    # Return comprehensive results
    return {
        'coherence_score': coherence_score,
        'model_info': {
            'num_topics': num_topics,
            'dictionary_size': len(dictionary),
            'num_docs': len(corpus)
        },
        'data_info': {
            'input_texts': len(texts),
            'processed_texts': len(processed_texts),
            'sampled': sample_size is not None
        },
        'parameters': {
            'sample_size': sample_size,
            'no_below': no_below,
            'no_above': no_above,
            'num_passes': num_passes,
            'random_seed': random_seed
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Optional function to save results
def save_coherence_results(results: Dict, prefix: str = "coherence_results") -> str:
    """
    Save coherence results to a JSON file.
    
    Args:
        results: Results dictionary from compute_topic_coherence
        prefix: Prefix for the output filename
        
    Returns:
        Path to the saved file
    """
    import json
    output_file = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    return output_file

# For command-line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate topic coherence for a text dataset')
    parser.add_argument('input_file', help='Path to input file (CSV or JSON)')
    parser.add_argument('--column', default='article', help='Column name containing text (default: article)')
    parser.add_argument('--sample-size', type=int, help='Number of texts to sample (optional)')
    parser.add_argument('--num-topics', type=int, default=10, help='Number of topics for LDA (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Load data
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    else:
        df = pd.read_json(args.input_file, lines=True)
    
    # Compute coherence
    results = compute_topic_coherence(
        texts=df[args.column],
        sample_size=args.sample_size,
        num_topics=args.num_topics,
        random_seed=args.seed
    )
    
    # Save and print results
    output_file = save_coherence_results(results)
    print(f"\nCoherence Score: {results['coherence_score']:.4f}")
    print(f"Results saved to {output_file}")