import pandas as pd
import numpy as np
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
import torch
from torch.cuda.amp import autocast # Added for mixed precision
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Swapped LogisticRegression for the much faster SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import logging
import time
from tqdm import tqdm
import gc
import os
from datetime import datetime
import argparse
import json
import multiprocessing
from functools import partial
import random # Added for sampling

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
MAX_TEXT_LENGTH = 1024
EVAL_SAMPLE_SIZE = -1 # New: Sample size for expensive evals
# Use 1 fewer core than total to leave one for system tasks
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)
# Use a smaller model for faster perplexity
PERPLEXITY_MODEL = 'distilgpt2' 

# --- Data Loading and Preprocessing ---
def load_real_data(num_samples, logger):
    """Loads the CNN/DailyMail dataset with progress tracking."""
    logger.info("Loading CNN/DailyMail dataset from Hugging Face...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
        if num_samples != -1 and num_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        df = pd.DataFrame(dataset)
        logger.info(f"Successfully loaded {len(df)} articles")
        return df['article'].tolist()
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def load_articles_from_file(filepath, column_name, logger, num_samples):
    """Helper to load articles from CSV or JSONL file."""
    logger.info(f"Loading data from {filepath} using column '{column_name}'")
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json') or filepath.endswith('.jsonl'):
            df = pd.read_json(filepath, lines=True)
        else:
            raise ValueError("Unsupported file format. Use .csv or .jsonl")
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in {filepath}. Available: {list(df.columns)}")
            
        articles = df[column_name].dropna().astype(str).tolist()
        
        if num_samples != -1 and num_samples < len(articles):
            # Shuffle before sampling to get a random subset from local file
            random.seed(42)
            random.shuffle(articles)
            articles = articles[:num_samples]
            
        logger.info(f"Loaded {len(articles)} articles.")
        return articles
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        raise

def _preprocess_worker(text, stop_words):
    """Worker function for parallel preprocessing."""
    if not isinstance(text, str):
        return []
    tokens = simple_preprocess(text, deacc=True) 
    return [word for word in tokens if word not in stop_words]

def preprocess_text(text_list, logger):
    """Batch process texts in parallel using multiprocessing."""
    logger.info("Preprocessing text data...")
    if not text_list:
        logger.warning("No texts to preprocess.")
        return []
    try:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
        logger.info(f"Starting preprocessing with {NUM_CORES} workers...")
        
        worker_func = partial(_preprocess_worker, stop_words=stop_words)
        
        preprocessed_texts = []
        with multiprocessing.Pool(NUM_CORES) as pool:
            with tqdm(total=len(text_list), desc="Preprocessing") as pbar:
                for result in pool.imap(worker_func, text_list, chunksize=100):
                    preprocessed_texts.append(result)
                    pbar.update(1)
        
        preprocessed_texts = [text for text in preprocessed_texts if text]
            
        logger.info(f"Successfully preprocessed {len(preprocessed_texts)} texts")
        return preprocessed_texts
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

# --- Evaluation Functions ---
def evaluate_topic_coherence(processed_texts, logger):
    """Enhanced topic coherence evaluation with multicore LDA."""
    logger.info("Evaluating Topic Coherence...")
    try:
        if not processed_texts:
            logger.warning("No preprocessed texts available. Skipping topic coherence.")
            return 0.0

        dictionary = Dictionary(processed_texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        if not corpus:
            logger.warning("Corpus is empty after filtering. Skipping topic coherence.")
            return 0.0

        logger.info(f"Training LDA model with {NUM_CORES} workers...")
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=10,
            random_state=42,
            passes=5,
            workers=NUM_CORES
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
        
        del lda_model, coherence_model, dictionary, corpus
        gc.collect()
        
        return coherence_score
    except Exception as e:
        logger.error(f"Error in topic coherence evaluation: {str(e)}")
        raise

def evaluate_statistical_properties(real_texts, synthetic_texts, logger):
    """Evaluates statistical properties. Handles empty lists."""
    logger.info("Evaluating Statistical Properties...")

    def get_stats(texts):
        if not texts:
            return 0.0, 0.0, 0.0
        lengths = []
        vocab = set()
        token_count = 0
        for doc in texts:
            if not isinstance(doc, str): continue
            tokens = doc.lower().split()
            lengths.append(len(tokens))
            vocab.update(tokens)
            token_count += len(tokens)
        avg_len = np.mean(lengths) if lengths else 0.0
        std_len = np.std(lengths) if lengths else 0.0
        ttr = len(vocab) / token_count if token_count else 0.0
        del vocab
        gc.collect()
        return avg_len, std_len, ttr

    try:
        real_avg_len, real_std, real_ttr = get_stats(real_texts)
        synth_avg_len, synth_std, synth_ttr = get_stats(synthetic_texts)
        
        logger.info("\nStatistical Properties Results:")
        logger.info(f"Real data - Avg length: {real_avg_len:.2f} (Std: {real_std:.2f}) | TTR: {real_ttr:.4f}")
        logger.info(f"Synthetic data - Avg length: {synth_avg_len:.2f} (Std: {synth_std:.2f}) | TTR: {synth_ttr:.4f}")

        return {
            'real_avg_length': real_avg_len,
            'real_std_length': real_std,
            'real_ttr': real_ttr,
            'synth_avg_length': synth_avg_len,
            'synth_std_length': synth_std,
            'synth_ttr': synth_ttr
        }
    except Exception as e:
        logger.error(f"Error in statistical properties evaluation: {str(e)}")
        raise

def evaluate_perplexity_batch(texts, model, tokenizer, device):
    """
    *** OPTIMIZED with Mixed Precision ***
    Evaluate perplexity for a *batch* of texts using parallel GPU processing.
    """
    valid_texts = [t for t in texts if t and isinstance(t, str)]
    if not valid_texts:
        return 0, 0

    try:
        encodings = tokenizer(
            valid_texts,
            return_tensors='pt',
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
            padding=True,
            return_attention_mask=True
        )

        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            # *** ADDED: autocast for fp16/mixed precision ***
            with autocast(dtype=torch.float16 if device == "cuda" else torch.bfloat16):
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                mean_nll = outputs.loss.item()
        
        num_tokens = (labels != -100).sum().item()
        total_nll = mean_nll * num_tokens
            
        return total_nll, num_tokens
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in perplexity batch: {e}. Skipping batch.")
        if "out of memory" in str(e) and device == "cuda":
            torch.cuda.empty_cache()
        return 0, 0


def run_perplexity_evaluation(real_texts, synthetic_texts, logger):
    """Enhanced perplexity evaluation with true batch processing."""
    logger.info("Starting Perplexity Evaluation...")
    try:
        logger.info(f"Loading perplexity model: {PERPLEXITY_MODEL}...")
        tokenizer = GPT2Tokenizer.from_pretrained(PERPLEXITY_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = GPT2LMHeadModel.from_pretrained(PERPLEXITY_MODEL).to(DEVICE)
        model.eval()

        real_perplexity, synth_perplexity = None, None

        if real_texts:
            logger.info(f"Evaluating real texts perplexity on {len(real_texts)} samples...")
            real_total_nll = 0
            real_total_tokens = 0
            
            for i in tqdm(range(0, len(real_texts), BATCH_SIZE), desc="Perplexity (Real)"):
                batch = real_texts[i:i + BATCH_SIZE]
                nll, tokens = evaluate_perplexity_batch(batch, model, tokenizer, DEVICE)
                real_total_nll += nll
                real_total_tokens += tokens
            
            if real_total_tokens > 0:
                real_perplexity = float(torch.exp(torch.tensor(real_total_nll / real_total_tokens)).item())
                logger.info(f"Real Data Perplexity: {real_perplexity:.4f}")
            else:
                logger.warning("No tokens processed for real data perplexity.")

        if synthetic_texts:
            logger.info(f"Evaluating synthetic texts perplexity on {len(synthetic_texts)} samples...")
            synth_total_nll = 0
            synth_total_tokens = 0
            
            for i in tqdm(range(0, len(synthetic_texts), BATCH_SIZE), desc="Perplexity (Synthetic)"):
                batch = synthetic_texts[i:i + BATCH_SIZE]
                nll, tokens = evaluate_perplexity_batch(batch, model, tokenizer, DEVICE)
                synth_total_nll += nll
                synth_total_tokens += tokens

            if synth_total_tokens > 0:
                synth_perplexity = float(torch.exp(torch.tensor(synth_total_nll / synth_total_tokens)).item())
                logger.info(f"Synthetic Data Perplexity: {synth_perplexity:.4f}")
            else:
                logger.warning("No tokens processed for synthetic data perplexity.")

        del model, tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return real_perplexity, synth_perplexity
    except Exception as e:
        logger.error(f"Error in perplexity evaluation: {str(e)}")
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        raise

def evaluate_downstream_task(real_texts, synthetic_texts, logger):
    """
    *** OPTIMIZED with SGDClassifier ***
    Downstream task evaluation with correct, scalable classifier.
    """
    logger.info("Starting Downstream Task Evaluation...")
    try:
        if not real_texts or not synthetic_texts:
            logger.warning("Insufficient data for downstream evaluation. Skipping.")
            return None

        data = real_texts + synthetic_texts
        labels = [0] * len(real_texts) + [1] * len(synthetic_texts)

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )

        logger.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        logger.info(f"Fitting vectorizer on {len(X_train)} training documents...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        logger.info(f"Transforming {len(X_test)} test documents...")
        X_test_tfidf = vectorizer.transform(X_test)

        # *** OPTIMIZATION: Use SGDClassifier for speed ***
        logger.info("Training classifier with SGD...")
        classifier = SGDClassifier(
            loss='log_loss',  # This makes it a Logistic Regression
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            n_jobs=NUM_CORES, # Use all cores
            early_stopping=True # Stop when validation score isn't improving
        )
        classifier.fit(X_train_tfidf, y_train)
        # *** End of Optimization ***

        logger.info("Evaluating classifier...")
        y_pred = classifier.predict(X_test_tfidf)
        
        report = classification_report(y_test, y_pred, target_names=['Real (0)', 'Synthetic (1)'], output_dict=True)
        logger.info("\nClassification Report (0=Real, 1=Synthetic):")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Real (0)', 'Synthetic (1)'])}")

        del X_train_tfidf, X_test_tfidf, vectorizer, classifier
        gc.collect()

        return report
    except Exception as e:
        logger.error(f"Error in downstream task evaluation: {str(e)}")
        raise

def main():
    logger = setup_logging()
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Optimized evaluation script for text generation')
    parser.add_argument('--evaluate', choices=['real', 'synthetic', 'both'], default='real',
                        help='Which datasets to evaluate. Default: real (CNN/DailyMail)')
    parser.add_argument('--real-file', type=str, default=None, help='Path to local real data file (CSV/JSONL)')
    parser.add_argument('--real-col', type=str, default='article', help='Column name for real articles')
    parser.add_argument('--synthetic-file', type=str, default=None, help='Path to local synthetic data file (CSV/JSONL)')
    parser.add_argument('--synthetic-col', type=str, default='generated_article', help='Column name for synthetic articles')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES, help='Number of samples to load (-1 for all)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for processing')
    
    # --- New Arguments for Optimization ---
    parser.add_argument('--eval-sample-size', type=int, default=EVAL_SAMPLE_SIZE,
                        help='Number of samples for *expensive* evals (perplexity, topic). -1 for all.')
    parser.add_argument('--skip-perplexity', action='store_true',
                        help='Skip the (very slow) perplexity evaluation entirely.')
    
    args = parser.parse_args()

    # Override module-level constants
    global NUM_SAMPLES, BATCH_SIZE, EVAL_SAMPLE_SIZE
    NUM_SAMPLES = args.num_samples
    BATCH_SIZE = args.batch_size
    EVAL_SAMPLE_SIZE = args.eval_sample_size

    try:
        logger.info("Starting evaluation pipeline...")
        logger.info(f"Using device: {DEVICE}")
        logger.info(f"Batch size: {BATCH_SIZE}, Num samples loaded: {NUM_SAMPLES}")
        logger.info(f"Expensive eval sample size: {EVAL_SAMPLE_SIZE}")
        if args.skip_perplexity:
            logger.warning("--- PERPLEXITY EVALUATION IS SKIPPED ---")

        real_articles = []
        synthetic_articles = []

        if args.evaluate in ('real', 'both'):
            if args.real_file:
                real_articles = load_articles_from_file(args.real_file, args.real_col, logger, NUM_SAMPLES)
            else:
                real_articles = load_real_data(NUM_SAMPLES, logger)

        if args.evaluate in ('synthetic', 'both'):
            if args.synthetic_file:
                synthetic_articles = load_articles_from_file(args.synthetic_file, args.synthetic_col, logger, NUM_SAMPLES)
            else:
                logger.warning("No synthetic file provided for evaluation.")

        if not real_articles and not synthetic_articles:
            logger.error("No data loaded. Exiting.")
            return

        # --- Create Subsets for Expensive Evals ---
        real_subset = real_articles
        synth_subset = synthetic_articles
        
        if EVAL_SAMPLE_SIZE != -1:
            if real_articles and len(real_articles) > EVAL_SAMPLE_SIZE:
                logger.info(f"Sampling {EVAL_SAMPLE_SIZE} real articles for expensive evals.")
                random.seed(42)
                real_subset = random.sample(real_articles, EVAL_SAMPLE_SIZE)
            
            if synthetic_articles and len(synthetic_articles) > EVAL_SAMPLE_SIZE:
                logger.info(f"Sampling {EVAL_SAMPLE_SIZE} synthetic articles for expensive evals.")
                random.seed(42)
                synth_subset = random.sample(synthetic_articles, EVAL_SAMPLE_SIZE)

        # --- Preprocessing (on subsets) ---
        processed_real = []
        processed_synthetic = []
        
        if real_subset:
            logger.info(f"Processing {len(real_subset)} real articles...")
            processed_real = preprocess_text(real_subset, logger)
        if synth_subset:
            logger.info(f"Processing {len(synth_subset)} synthetic articles...")
            processed_synthetic = preprocess_text(synth_subset, logger)

        results = {}

        # --- Fidelity Evaluations ---
        logger.info("\n" + "="*20 + " FIDELITY EVALUATION " + "="*20)
        # Topic Coherence (on processed subsets)
        if processed_real:
            results['topic_coherence_real'] = evaluate_topic_coherence(processed_real, logger)
        if processed_synthetic:
            results['topic_coherence_synthetic'] = evaluate_topic_coherence(processed_synthetic, logger)

        # Statistical properties (on FULL datasets - this is fast)
        results['statistical_properties'] = evaluate_statistical_properties(real_articles, synthetic_articles, logger)

        # Perplexity (on subsets, and skippable)
        if not args.skip_perplexity:
            if real_subset or synth_subset:
                perp_real, perp_synth = run_perplexity_evaluation(real_subset, synth_subset, logger)
                results['perplexity'] = {'real': perp_real, 'synthetic': perp_synth}
            else:
                logger.info("Skipping perplexity: no data in subsets.")
                results['perplexity'] = None
        else:
            results['perplexity'] = "SKIPPED"

        # --- Utility Evaluation ---
        logger.info("\n" + "="*20 + " UTILITY EVALUATION " + "="*20)
        # Downstream task (on FULL datasets)
        if args.evaluate == 'both' and real_articles and synthetic_articles:
            results['classification'] = evaluate_downstream_task(real_articles, synthetic_articles, logger)
        else:
            if args.evaluate == 'both':
                logger.warning("Downstream task skipped: requires both real and synthetic data.")
            results['classification'] = None

        # --- Save Results ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        
        def default_json_serializer(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=default_json_serializer)

        logger.info(f"\nResults saved to {results_file}")
        logger.info(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")

    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
    finally:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data. {e}")
        
    main()