"""
Topic Coherence Evaluation Module

Implements functions for the following tasks:
- **Text Preprocessing**: Tokenizes, lowercases, removes stopwords and punctuation using NLTK.
- **Topic Modeling**: Trains an LDA model with 10 topics using Gensim.
- **Coherence Calculation**: Computes the c_v coherence score for the LDA topics.

Other notes:
- **Sampling**: Optionally samples a subset of texts before processing.
- **Input Formats**: Supports CSV or JSON input files with configurable text column.
- **Output**: Saves coherence score and metadata as a timestamped JSON file.

Example usage (CLI):

    python coherence_test.py input_data.csv --column article --sample-size 1000

This will load texts from the specified column, preprocess and sample them, compute topic coherence, and save results.
"""

import argparse
import json
import string
from datetime import datetime

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def process_texts(texts, sample_size=None):
    """
    Preprocess a list of texts: lowercase, tokenize, remove stopwords and punctuation.
    Optionally sample a subset of texts.
    Returns a list of lists of tokens.
    """
    stop_words = set(stopwords.words('english'))
    processed = []

    # Optionally sample texts
    if sample_size and len(texts) > sample_size:
        texts = np.random.choice(texts, size=sample_size, replace=False)

    for text in tqdm(texts, desc="Processing"):
        if not isinstance(text, str):
            continue
        tokens = word_tokenize(text.lower())
        filtered = [
            word for word in tokens
            if word.isalpha() and word not in stop_words and word not in string.punctuation
        ]
        if filtered:
            processed.append(filtered)

    return processed


def calculate_coherence(texts):
    """
    Calculate the c_v topic coherence score for a list of tokenized texts.
    Trains a 10-topic LDA model using Gensim.
    """
    # Create dictionary and corpus
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        random_state=42,
        passes=5
    )

    # Calculate coherence
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    return coherence_model.get_coherence()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate topic coherence for a text dataset'
    )
    parser.add_argument('input_file', help='Path to input file (CSV or JSON)')
    parser.add_argument('--column', default='article', help='Column name containing text (default: article)')
    parser.add_argument('--sample-size', type=int, help='Number of texts to sample (optional)')
    args = parser.parse_args()

    # Load data
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    else:
        df = pd.read_json(args.input_file, lines=True)

    texts = df[args.column].dropna().astype(str).tolist()
    print(f"Loaded {len(texts)} texts from {args.input_file}")

    # Process and evaluate
    processed_texts = process_texts(texts, args.sample_size)
    coherence_score = calculate_coherence(processed_texts)

    # Save results
    results = {
        'file': args.input_file,
        'total_texts': len(texts),
        'processed_texts': len(processed_texts),
        'sample_size': args.sample_size,
        'coherence_score': coherence_score,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    output_file = f"coherence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCoherence Score: {coherence_score:.4f}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
