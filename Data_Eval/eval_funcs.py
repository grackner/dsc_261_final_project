"""
Synthetic Data Quality — Minimal Importable Module

Implements separate functions for metrics 2a–2d with the constraints you requested:
- **No data loading/cleaning**: All functions accept arrays/series/lists of strings.
- **Tokenization**: Always uses `facebook/opt-125m`'s fast tokenizer.
- **Embeddings**: Mean-pooled last hidden states from OPT-125m (via `AutoModel`).
- **Perplexity**: Uses OPT-125m (via `AutoModelForCausalLM`).
- **Wasserstein**: Sliced Wasserstein on OPT embeddings.
- **Classification**: `LogisticRegressionCV` (solver='saga') with cross-validation.
- **Single module with separate functions**: caching minimizes repeated model loads.

Example (notebook):

    import numpy as np, pandas as pd
    from synthetic_data_quality import (
        perplexity_for_corpora,
        wasserstein_distance_embeddings,
        classify_real_vs_synth,
        compute_stat_properties,
        compare_stat_properties,
        compute_opt_embeddings,
    )

    real  = df_real["article"].values      # ndarray/Series/list of strings
    synth = df_synth["article"].values

    # 2a: Perplexity (corpus-level per set)
    ppl = perplexity_for_corpora(real, synth, batch_size=8, max_length=2048)

    # 2b: Wasserstein on embeddings
    w = wasserstein_distance_embeddings(real, synth, n_projections=128, max_length=2048)

    # 2c: Classification with LogisticRegressionCV(saga)
    clf_res = classify_real_vs_synth(real, synth, cv=5, max_length=2048)

    # 2d: Stats (tokenized with OPT)
    stats = compare_stat_properties(real, synth, max_length=2048)

    # Optional: reuse embeddings across metrics
    Er = compute_opt_embeddings(real, max_length=2048)
    Es = compute_opt_embeddings(synth, max_length=2048)

"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Dict, Tuple, Union

import numpy as np
import pandas as pd

TextArray = Union[pd.Series, np.ndarray, List[str]]

# OPT-125m context window (used to cap requested lengths without extra model loads)
_OPT_MAX_CTX = 2048

# -----------------------------------------------------------------------------
# Utilities: tokenizer/model caches and simple helpers
# -----------------------------------------------------------------------------

def _to_list(texts: TextArray) -> List[str]:
    """Accept pd.Series, np.ndarray, or list[str] and return a Python list[str]."""
    if isinstance(texts, pd.Series):
        return texts.tolist()
    if isinstance(texts, np.ndarray):
        return texts.tolist()
    if isinstance(texts, list):
        return texts
    if isinstance(texts, Iterable):
        return list(texts)
    raise TypeError("Expected pandas.Series, numpy.ndarray, list[str], or iterable of strings.")


@lru_cache(maxsize=1)
def _get_opt_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


@lru_cache(maxsize=1)
def _get_opt_lm_model():
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _get_opt_base_model():
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/opt-125m")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


def _batch_encode(texts: List[str], max_length: int) -> dict:
    tok = _get_opt_tokenizer()
    # Cap max_length to model context for safety
    max_length = min(int(max_length), _OPT_MAX_CTX)
    return tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _mean_pool(last_hidden_state, attention_mask):
    """Mean pool over the sequence length, masking pads."""
    import torch
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


# -----------------------------------------------------------------------------
# 2d. Statistical properties (using OPT tokenizer)
# -----------------------------------------------------------------------------

def compute_stat_properties(texts: TextArray, max_length: int = 2048) -> Dict[str, float]:
    """Corpus statistics with OPT tokenization.

    Returns:
        avg_len_tokens, std_len_tokens, avg_len_chars, ttr, hapax_ratio
    """
    docs = _to_list(texts)
    if len(docs) == 0:
        return {
            'avg_len_tokens': 0.0,
            'std_len_tokens': 0.0,
            'avg_len_chars': 0.0,
            'ttr': 0.0,
            'hapax_ratio': 0.0,
        }

    import torch
    # Cap to context window
    eff_max_len = min(int(max_length), _OPT_MAX_CTX)
    enc = _batch_encode(docs, max_length=eff_max_len)
    input_ids = enc["input_ids"]  # (B, L)
    attn = enc["attention_mask"]  # (B, L)

    lengths = attn.sum(dim=1).to(dtype=torch.float32).cpu().numpy()
    avg_len_tokens = float(lengths.mean())
    std_len_tokens = float(lengths.std())
    avg_len_chars = float(np.mean([len(t) for t in docs]))

    masked_ids = input_ids.masked_select(attn.bool()).cpu().numpy()
    if masked_ids.size == 0:
        ttr = 0.0
        hapax_ratio = 0.0
    else:
        unique, counts = np.unique(masked_ids, return_counts=True)
        ttr = float(len(unique) / masked_ids.size)
        hapax_ratio = float((counts == 1).sum() / len(unique))

    return {
        'avg_len_tokens': avg_len_tokens,
        'std_len_tokens': std_len_tokens,
        'avg_len_chars': avg_len_chars,
        'ttr': ttr,
        'hapax_ratio': hapax_ratio,
    }


def compare_stat_properties(real_texts: TextArray, synth_texts: TextArray, max_length: int = 2048) -> Dict[str, Dict[str, float]]:
    """Side-by-side stats for real and synthetic corpora."""
    eff_max_len = min(int(max_length), _OPT_MAX_CTX)
    return {
        'real': compute_stat_properties(real_texts, max_length=eff_max_len),
        'synthetic': compute_stat_properties(synth_texts, max_length=eff_max_len),
    }


# -----------------------------------------------------------------------------
# 2a. Perplexity with OPT-125m (with progress/ETA prints)
# -----------------------------------------------------------------------------

def _batch_loss(input_ids, attention_mask) -> Tuple[float, int]:
    import torch
    model, device = _get_opt_lm_model()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mean_nll = float(out.loss.detach().cpu().item())
    n_tokens = int((labels != -100).sum().item())
    total_nll = mean_nll * n_tokens
    return total_nll, n_tokens


def perplexity_for_corpora(
    real_texts: TextArray,
    synth_texts: TextArray,
    batch_size: int = 8,
    max_length: int = 2048,
) -> Dict[str, Dict[str, float]]:
    """Compute **corpus-level** perplexity for each corpus using OPT-125m.

Implementation is intentionally simple: truncate each doc to `max_length`,
pad within batch, compute NLL and aggregate over tokens per corpus.

**Note:** `max_length` is automatically capped at the model's maximum
context window (`model.config.max_position_embeddings`, 2048 for OPT-125m).
If you pass a larger value, it will be reduced to that cap.

Prints progress with estimated time remaining (ETA) based on observed
average batch time. ETA is approximate.

Returns: { 'real': {'corpus_ppl': ...}, 'synthetic': {'corpus_ppl': ...} }
"""
    import math
    import time

    real_docs = _to_list(real_texts)
    synth_docs = _to_list(synth_texts)

    # Enforce model's max context to prevent CUDA/CUBLAS failures
    model, _device = _get_opt_lm_model()
    model_max_ctx = int(getattr(model.config, 'max_position_embeddings', _OPT_MAX_CTX))
    eff_max_len = min(int(max_length), model_max_ctx)

    # Progress accounting across both corpora
    total_batches = ((len(real_docs) + batch_size - 1) // batch_size) + \
                    ((len(synth_docs) + batch_size - 1) // batch_size)
    processed_batches = 0
    t0 = time.perf_counter()

    print(f"[perplexity] device={_device} batch_size={batch_size} max_length(requested)={max_length} max_length(effective)={eff_max_len}", flush=True)
    print(f"[perplexity] num_docs: real={len(real_docs)} synthetic={len(synth_docs)} total_batches={total_batches}", flush=True)

    def _tick():
        # Print progress/ETA every few batches
        nonlocal processed_batches
        if total_batches == 0 or processed_batches == 0:
            return
        if (processed_batches % 5) != 0 and processed_batches != total_batches:
            return
        elapsed = time.perf_counter() - t0
        avg = elapsed / processed_batches
        remaining = max(0, total_batches - processed_batches)
        eta = remaining * avg
        print(
            f"[perplexity] progress {processed_batches}/{total_batches} | elapsed={elapsed:.1f}s | avg/batch={avg:.2f}s | ETA~{eta:.1f}s",
            flush=True,
        )

    def _corpus_ppl(docs: List[str]) -> float:
        nonlocal processed_batches
        if len(docs) == 0:
            print("[perplexity] empty corpus; skipping.", flush=True)
            return 0.0
        total_nll = 0.0
        total_tokens = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            enc = _batch_encode(batch, max_length=eff_max_len)
            nll, ntok = _batch_loss(enc["input_ids"], enc["attention_mask"])  # sums across batch
            total_nll += nll
            total_tokens += ntok
            processed_batches += 1
            _tick()
        return float(np.exp(total_nll / total_tokens)) if total_tokens else 0.0

    res_real = _corpus_ppl(real_docs)
    res_synth = _corpus_ppl(synth_docs)

    elapsed_total = (time.perf_counter() - t0)
    print(f"[perplexity] done in {elapsed_total:.1f}s", flush=True)

    return {
        'real': {'corpus_ppl': res_real},
        'synthetic': {'corpus_ppl': res_synth},
    }


# -----------------------------------------------------------------------------
# OPT embeddings (used by 2b, 2c). Exposed for reuse.
# -----------------------------------------------------------------------------

def compute_opt_embeddings(
    texts: TextArray,
    batch_size: int = 8,
    max_length: int = 2048,
    *,
    verbose: bool = False,
    label: str = "corpus",
) -> np.ndarray:
    """Return OPT-125m mean-pooled embeddings for each text as an ndarray (N, H).
    Truncates to `max_length` tokens for simplicity (capped at model context).
    If `verbose`, prints progress and ETA for the embedding pass.
    """
    import torch
    from torch.utils.data import DataLoader

    docs = _to_list(texts)
    if len(docs) == 0:
        return np.zeros((0, 768), dtype=np.float32)  # OPT-125m hidden size

    model, device = _get_opt_base_model()
    model_max_ctx = int(getattr(model.config, 'max_position_embeddings', _OPT_MAX_CTX))
    eff_max_len = min(int(max_length), model_max_ctx)

    if verbose:
        import time
        print(f"[embed:{label}] device={device} batch_size={batch_size} "
              f"max_length(req)={max_length} max_length(eff)={eff_max_len} "
              f"num_docs={len(docs)}", flush=True)
        t0 = time.perf_counter()

    class _Dataset:
        def __init__(self, xs): self.xs = xs
        def __len__(self): return len(self.xs)
        def __getitem__(self, idx): return self.xs[idx]

    def _collate(batch_texts: List[str]):
        return _batch_encode(batch_texts, max_length=eff_max_len)

    dl = DataLoader(_Dataset(docs), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total_batches = (len(docs) + batch_size - 1) // batch_size
    processed = 0

    outs = []
    with torch.no_grad():
        for enc in dl:
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
            pooled = _mean_pool(out.last_hidden_state, attn)  # (B, H)
            outs.append(pooled.detach().cpu().numpy())
            processed += 1
            if verbose and (processed % 5 == 0 or processed == total_batches):
                import time
                elapsed = time.perf_counter() - t0
                avg = elapsed / processed
                rem = max(0, total_batches - processed)
                eta = rem * avg
                print(f"[embed:{label}] progress {processed}/{total_batches} | "
                      f"elapsed={elapsed:.1f}s | avg/batch={avg:.2f}s | ETA~{eta:.1f}s",
                      flush=True)
    if verbose:
        import time
        print(f"[embed:{label}] done in {time.perf_counter()-t0:.1f}s", flush=True)

    return np.vstack(outs)


# -----------------------------------------------------------------------------
# 2b. Sliced Wasserstein distance on OPT embeddings (with progress prints)
# -----------------------------------------------------------------------------

def wasserstein_distance_embeddings(
    real_texts: TextArray,
    synth_texts: TextArray,
    n_projections: int = 128,
    batch_size: int = 8,
    max_length: int = 2048,
    seed: int = 42,
) -> Dict[str, float | List[float]]:
    """Compute sliced Wasserstein distance between embedding sets.

    1) Embed docs via OPT-125m mean-pooling
    2) Draw `n_projections` random unit vectors u
    3) Project both sets and compute 1-D Wasserstein distance per projection
    4) Return average and the list

    Prints progress for embeddings and projection loop.
    """
    from scipy.stats import wasserstein_distance
    import time

    # Verbose embedding passes
    Er = compute_opt_embeddings(real_texts, batch_size=batch_size, max_length=max_length, verbose=True, label="real")
    Es = compute_opt_embeddings(synth_texts, batch_size=batch_size, max_length=max_length, verbose=True, label="synthetic")
    if Er.shape[0] == 0 or Es.shape[0] == 0:
        return {'mean_distance': 0.0, 'distances': []}

    rng = np.random.default_rng(seed)
    d = Er.shape[1]
    dists = []
    t0 = time.perf_counter()
    for i in range(n_projections):
        u = rng.normal(size=(d, 1))
        u /= np.linalg.norm(u) + 1e-12
        r_proj = Er @ u
        s_proj = Es @ u
        dists.append(float(wasserstein_distance(r_proj.ravel(), s_proj.ravel())))
        if (i + 1) % 10 == 0 or (i + 1) == n_projections:
            elapsed = time.perf_counter() - t0
            avg = elapsed / (i + 1)
            rem = max(0, n_projections - (i + 1))
            eta = rem * avg
            print(f"[wasserstein] projections {i+1}/{n_projections} | "
                  f"elapsed={elapsed:.1f}s | avg/proj={avg:.3f}s | ETA~{eta:.1f}s",
                  flush=True)

    return {'mean_distance': float(np.mean(dists)), 'distances': dists}


# -----------------------------------------------------------------------------
# 2c. Classification (real vs synthetic) using LogisticRegressionCV(saga) (with progress prints)
# -----------------------------------------------------------------------------

def classify_real_vs_synth(
    real_texts: TextArray,
    synth_texts: TextArray,
    test_size: float = 0.2,
    batch_size: int = 8,
    max_length: int = 2048,
    cv: int = 5,
    Cs: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0),
    seed: int = 42,
) -> Dict[str, object]:
    """Train a light classifier on OPT embeddings with cross-validated LogisticRegression(saga).

    Returns:
      - 'metrics': accuracy, macro_f1, roc_auc
      - 'report': sklearn classification_report (dict)
      - 'embeddings_shape': (N, H)
      - 'classifier': fitted sklearn Pipeline (StandardScaler + LogisticRegressionCV)

    Prints progress for embedding passes and timing for model fit.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
    import time

    print(f"[classify] batch_size={batch_size} max_length={max_length} cv={cv} Cs={list(Cs)}", flush=True)
    Er = compute_opt_embeddings(real_texts, batch_size=batch_size, max_length=max_length, verbose=True, label="real")
    Es = compute_opt_embeddings(synth_texts, batch_size=batch_size, max_length=max_length, verbose=True, label="synthetic")

    if Er.shape[0] == 0 or Es.shape[0] == 0:
        return {'metrics': {}, 'report': {}, 'embeddings_shape': (0, 0)}

    X = np.vstack([Er, Es])
    y = np.array([0] * Er.shape[0] + [1] * Es.shape[0])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    lr = LogisticRegressionCV(
        solver='saga',
        penalty='l2',
        Cs=list(Cs),
        cv=cv,
        scoring='roc_auc',
        max_iter=5000,
        n_jobs=-1,
        refit=True,
    )
    clf = make_pipeline(StandardScaler(), lr)
    print("[classify] fitting LogisticRegressionCV...", flush=True)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    print(f"[classify] fit done in {time.perf_counter()-t0:.1f}s", flush=True)

    yhat = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(yte, yhat)),
        'macro_f1': float(f1_score(yte, yhat, average='macro')),
        'roc_auc': float(roc_auc_score(yte, proba)),
    }
    report = classification_report(yte, yhat, target_names=['real', 'synthetic'], output_dict=True)

    return {
        'metrics': metrics,
        'report': report,
        'embeddings_shape': (int(X.shape[0]), int(X.shape[1])),
        'classifier': clf,
    }
