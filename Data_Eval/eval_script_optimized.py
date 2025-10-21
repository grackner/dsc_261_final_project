import pandas as pd
import numpy as np
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import logging
import time
from tqdm import tqdm
import gc
import os
from datetime import datetime

# --- Setup Logging ---
def setup_logging():
    """Configure logging to both file and console with timestamps."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eval_log_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- Constants ---
NUM_SAMPLES = -1  # Use -1 for full dataset
BATCH_SIZE = 32   # For batch processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LENGTH = 1024  # Maximum text length for GPT-2

# --- Data Loading and Preprocessing ---
def load_real_data(num_samples, logger):
    """Loads the CNN/DailyMail dataset with progress tracking."""
    logger.info("Loading CNN/DailyMail dataset...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
        if num_samples != -1:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        df = pd.DataFrame(dataset)
        logger.info(f"Successfully loaded {len(df)} articles")
        return df['article'].tolist()
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_batch(texts, stop_words):
    """Process a batch of texts."""
    preprocessed = []
    for text in texts:
        if not isinstance(text, str):
            continue
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            word for word in tokens
            if word.isalpha() and word not in stop_words and word not in string.punctuation
        ]
        preprocessed.append(filtered_tokens)
    return preprocessed

def preprocess_text(text_list, logger):
    """Batch process texts with progress tracking."""
    logger.info("Preprocessing text data...")
    try:
        stop_words = set(stopwords.words('english'))
        preprocessed_texts = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(text_list), BATCH_SIZE)):
            batch = text_list[i:i + BATCH_SIZE]
            preprocessed_batch = preprocess_batch(batch, stop_words)
            preprocessed_texts.extend(preprocessed_batch)
            
        logger.info(f"Successfully preprocessed {len(preprocessed_texts)} texts")
        return preprocessed_texts
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

# --- Evaluation Functions ---
def evaluate_topic_coherence(processed_texts, logger):
    """Enhanced topic coherence evaluation with progress tracking."""
    logger.info("Evaluating Topic Coherence...")
    try:
        if not processed_texts:
            logger.warning("Not enough data to evaluate topic coherence")
            return 0.0

        dictionary = Dictionary(processed_texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        if not corpus:
            logger.warning("Corpus is empty after filtering")
            return 0.0

        logger.info("Training LDA model...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=10,
            random_state=42,
            passes=5
        )
        
        logger.info("Calculating Coherence Score...")
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        logger.info(f"Coherence Score (C_v): {coherence_score:.4f}")
        
        # Clean up to free memory
        del lda_model
        del coherence_model
        gc.collect()
        
        return coherence_score
    except Exception as e:
        logger.error(f"Error in topic coherence evaluation: {str(e)}")
        raise

def evaluate_statistical_properties(real_texts, synthetic_texts, logger):
    """Enhanced statistical properties evaluation with better memory management."""
    logger.info("Evaluating Statistical Properties...")
    try:
        # Process in batches to manage memory
        real_lengths = []
        real_vocab = set()
        real_token_count = 0
        
        for i in range(0, len(real_texts), BATCH_SIZE):
            batch = real_texts[i:i + BATCH_SIZE]
            batch_lengths = [len(doc.split()) for doc in batch]
            real_lengths.extend(batch_lengths)
            
            # Update vocabulary and token count
            for doc in batch:
                tokens = doc.lower().split()
                real_vocab.update(tokens)
                real_token_count += len(tokens)
        
        # Same for synthetic texts
        synth_lengths = []
        synth_vocab = set()
        synth_token_count = 0
        
        for i in range(0, len(synthetic_texts), BATCH_SIZE):
            batch = synthetic_texts[i:i + BATCH_SIZE]
            batch_lengths = [len(doc.split()) for doc in batch]
            synth_lengths.extend(batch_lengths)
            
            for doc in batch:
                tokens = doc.lower().split()
                synth_vocab.update(tokens)
                synth_token_count += len(tokens)
        
        # Calculate metrics
        real_ttr = len(real_vocab) / real_token_count if real_token_count else 0
        synth_ttr = len(synth_vocab) / synth_token_count if synth_token_count else 0
        
        # Log results
        logger.info("\nStatistical Properties Results:")
        logger.info(f"Real data - Avg length: {np.mean(real_lengths):.2f} (Std: {np.std(real_lengths):.2f})")
        logger.info(f"Synthetic data - Avg length: {np.mean(synth_lengths):.2f} (Std: {np.std(synth_lengths):.2f})")
        logger.info(f"Real data - TTR: {real_ttr:.4f}")
        logger.info(f"Synthetic data - TTR: {synth_ttr:.4f}")
        
        # Clean up
        del real_vocab, synth_vocab
        gc.collect()
        
        return {
            'real_avg_length': np.mean(real_lengths),
            'synth_avg_length': np.mean(synth_lengths),
            'real_ttr': real_ttr,
            'synth_ttr': synth_ttr
        }
    except Exception as e:
        logger.error(f"Error in statistical properties evaluation: {str(e)}")
        raise

def evaluate_perplexity_batch(texts, model, tokenizer, device):
    """Evaluate perplexity for a batch of texts."""
    total_nll = 0
    total_tokens = 0

    for text in texts:
        if not text or not isinstance(text, str):
            continue

        encodings = tokenizer(
            text,
            return_tensors='pt',
            max_length=MAX_TEXT_LENGTH,
            truncation=True
        )
        input_ids = encodings.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss * input_ids.size(1)
            
        total_nll += nll.item()
        total_tokens += input_ids.size(1)
        
    return total_nll, total_tokens

def run_perplexity_evaluation(real_texts, synthetic_texts, logger):
    """Enhanced perplexity evaluation with batch processing and memory management."""
    logger.info("Starting Perplexity Evaluation...")
    try:
        logger.info("Loading GPT-2 model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
        model.eval()

        # Evaluate real texts
        logger.info("Evaluating real texts perplexity...")
        real_total_nll = 0
        real_total_tokens = 0
        
        for i in tqdm(range(0, len(real_texts), BATCH_SIZE)):
            batch = real_texts[i:i + BATCH_SIZE]
            nll, tokens = evaluate_perplexity_batch(batch, model, tokenizer, DEVICE)
            real_total_nll += nll
            real_total_tokens += tokens

        # Evaluate synthetic texts
        logger.info("Evaluating synthetic texts perplexity...")
        synth_total_nll = 0
        synth_total_tokens = 0
        
        for i in tqdm(range(0, len(synthetic_texts), BATCH_SIZE)):
            batch = synthetic_texts[i:i + BATCH_SIZE]
            nll, tokens = evaluate_perplexity_batch(batch, model, tokenizer, DEVICE)
            synth_total_nll += nll
            synth_total_tokens += tokens

        # Calculate final perplexity scores
        real_perplexity = torch.exp(torch.tensor(real_total_nll / real_total_tokens))
        synth_perplexity = torch.exp(torch.tensor(synth_total_nll / synth_total_tokens))

        logger.info(f"Real Data Perplexity: {real_perplexity.item():.4f}")
        logger.info(f"Synthetic Data Perplexity: {synth_perplexity.item():.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return real_perplexity.item(), synth_perplexity.item()
    except Exception as e:
        logger.error(f"Error in perplexity evaluation: {str(e)}")
        raise

def evaluate_downstream_task(real_texts, synthetic_texts, logger):
    """Enhanced downstream task evaluation with batch processing."""
    logger.info("Starting Downstream Task Evaluation...")
    try:
        # Prepare data
        data = real_texts + synthetic_texts
        labels = [0] * len(real_texts) + [1] * len(synthetic_texts)

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )

        # Vectorize in batches
        logger.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer(max_features=5000)
        
        # Fit vectorizer on training data in batches
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            batch = X_train[i:i + BATCH_SIZE]
            if i == 0:
                X_train_tfidf = vectorizer.fit_transform(batch)
            else:
                X_train_tfidf = torch.sparse.cat([
                    X_train_tfidf,
                    vectorizer.transform(batch)
                ], dim=0)

        # Transform test data
        X_test_tfidf = vectorizer.transform(X_test)

        # Train and evaluate
        logger.info("Training classifier...")
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_train_tfidf, y_train)

        logger.info("Evaluating classifier...")
        y_pred = classifier.predict(X_test_tfidf)
        
        # Log results
        report = classification_report(y_test, y_pred)
        logger.info("\nClassification Report (0=Real, 1=Synthetic):")
        logger.info("\n" + report)

        # Clean up
        del X_train_tfidf, X_test_tfidf
        gc.collect()

        return report
    except Exception as e:
        logger.error(f"Error in downstream task evaluation: {str(e)}")
        raise

def main():
    # Setup logging
    logger = setup_logging()
    start_time = time.time()
    
    try:
        logger.info("Starting evaluation pipeline...")
        logger.info(f"Using device: {DEVICE}")
        
        # Load and preprocess data
        real_articles = load_real_data(NUM_SAMPLES, logger)
        
        # Replace this with your actual synthetic data loading
        synthetic_articles = [text.replace("CNN", "GMN") for text in real_articles[:len(real_articles)//2]]
        
        processed_real = preprocess_text(real_articles, logger)
        processed_synthetic = preprocess_text(synthetic_articles, logger)
        
        # Run evaluations
        logger.info("\n" + "="*20 + " FIDELITY EVALUATION " + "="*20)
        
        topic_scores = {
            'real': evaluate_topic_coherence(processed_real, logger),
            'synthetic': evaluate_topic_coherence(processed_synthetic, logger)
        }
        
        stat_properties = evaluate_statistical_properties(
            real_articles,
            synthetic_articles,
            logger
        )
        
        perplexity_scores = run_perplexity_evaluation(
            real_articles,
            synthetic_articles,
            logger
        )
        
        logger.info("\n" + "="*20 + " UTILITY EVALUATION " + "="*20)
        classification_results = evaluate_downstream_task(
            real_articles,
            synthetic_articles,
            logger
        )
        
        # Save results
        results = {
            'topic_coherence': topic_scores,
            'statistical_properties': stat_properties,
            'perplexity': {
                'real': perplexity_scores[0],
                'synthetic': perplexity_scores[1]
            },
            'classification': classification_results
        }
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"\nResults saved to {results_file}")
        logger.info(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()