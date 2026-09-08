"""
Feature Engineering Module (TF-IDF)
===================================
This module handles feature extraction for the baseline model using Scikit-Learn's
Term Frequency - Inverse Document Frequency (TF-IDF) Vectorizer.

Key Anti-Leakage Rule:
----------------------
The TF-IDF vectorizer is strictly fitted ONLY on the training split (X_train).
The test split (X_test) and any new incoming user reviews are transformed
using the pre-fitted vectorizer.
"""

import os
import joblib
import pandas as pd
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


def build_tfidf_features(
    train_texts: pd.Series,
    test_texts: Optional[pd.Series] = None,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    save_path: Optional[str] = "models/tfidf_vectorizer.joblib",
) -> Tuple[spmatrix, Optional[spmatrix], TfidfVectorizer]:
    """
    Fit TF-IDF vectorizer on training text only and transform test text.

    Parameters:
    -----------
    train_texts : pd.Series or list of str
        Training text data (used to fit and transform).
    test_texts : pd.Series or list of str, optional
        Held-out testing text data (transformed only, preventing data leakage).
    max_features : int, default=5000
        Maximum vocabulary size for top term frequencies.
    ngram_range : tuple of (int, int), default=(1, 2)
        Unigrams and bigrams.
    min_df : int, default=2
        Minimum document frequency.
    save_path : str, optional
        Filepath to save the fitted vectorizer for inference in Streamlit.

    Returns:
    --------
    X_train_tfidf : scipy.sparse.csr_matrix
        TF-IDF feature matrix for training data.
    X_test_tfidf : scipy.sparse.csr_matrix or None
        TF-IDF feature matrix for testing data.
    vectorizer : TfidfVectorizer
        Fitted Scikit-Learn vectorizer.
    """
    print(f"[FeatureEngineering] Fitting TfidfVectorizer (max_features={max_features}, ngram_range={ngram_range}, min_df={min_df})...")
    # In tiny samples, ensure min_df doesn't exceed document count
    actual_min_df = min(min_df, max(1, len(train_texts) // 2))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=actual_min_df,
        sublinear_tf=True,
    )

    # 1. Fit and transform on training data ONLY
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    print(f"[FeatureEngineering] X_train_tfidf shape: {X_train_tfidf.shape}")

    # 2. Transform test data without fitting (zero leakage)
    X_test_tfidf = None
    if test_texts is not None:
        print("[FeatureEngineering] Transforming X_test without fitting (no data leakage)...")
        X_test_tfidf = vectorizer.transform(test_texts)
        print(f"[FeatureEngineering] X_test_tfidf shape: {X_test_tfidf.shape}")

    # 3. Save fitted vectorizer for reuse in Streamlit and inference
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(vectorizer, save_path)
        print(f"[FeatureEngineering] Saved fitted vectorizer to: {save_path}")

    return X_train_tfidf, X_test_tfidf, vectorizer


def load_tfidf_vectorizer(path: str = "models/tfidf_vectorizer.joblib") -> TfidfVectorizer:
    """
    Load a pre-fitted TfidfVectorizer from disk.

    Parameters:
    -----------
    path : str
        Path to the saved vectorizer.

    Returns:
    --------
    TfidfVectorizer
        Loaded vectorizer ready for inference.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at: {path}")
    return joblib.load(path)


if __name__ == "__main__":
    print("Testing feature_engineering module...")
    sample_train = pd.Series([
        "delicious gourmet coffee great taste delicious coffee",
        "terrible stale coffee product not good at all",
        "best dog food product highly recommended coffee",
    ])
    sample_test = pd.Series([
        "great coffee taste delicious",
        "awful product not recommended",
    ])
    X_tr, X_te, vec = build_tfidf_features(sample_train, sample_test, max_features=100, min_df=1, save_path=None)
    assert X_tr.shape[0] == 3
    assert X_te.shape[0] == 2
    print("Feature engineering test passed successfully!")
