"""
Baseline Model Module (TF-IDF + Logistic Regression)
====================================================
This module implements the baseline classification model using Scikit-Learn's
LogisticRegression combined with TF-IDF features.

Beginner-friendly explanation:
------------------------------
Logistic Regression calculates a linear decision boundary over TF-IDF word weights
and outputs a probability score using the sigmoid function:
P(Helpful = 1 | x) = 1 / (1 + e^-(w·x + b))
"""

import os
import joblib
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from scipy.sparse import spmatrix


def train_baseline_model(
    X_train: spmatrix,
    y_train: np.ndarray,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
    save_path: Optional[str] = "models/baseline_model.joblib",
) -> LogisticRegression:
    """
    Train a Logistic Regression model on TF-IDF features with class weighting.

    Parameters:
    -----------
    X_train : scipy.sparse matrix
        Training TF-IDF feature matrix.
    y_train : np.ndarray or pd.Series
        Binary training labels (0 or 1).
    class_weight : str, default='balanced'
        Adjusts weights inversely proportional to class frequencies to combat 83:17 imbalance.
    random_state : int, default=42
        Random seed for reproducibility.
    max_iter : int, default=1000
        Maximum solver iterations.
    save_path : str, optional
        Path to save the serialized model.

    Returns:
    --------
    LogisticRegression
        Fitted Scikit-Learn Logistic Regression model.
    """
    print(f"[BaselineModel] Training LogisticRegression (class_weight='{class_weight}')...")
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train, y_train)
    print(f"[BaselineModel] Training completed. Classes: {model.classes_}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"[BaselineModel] Saved baseline model to: {save_path}")

    return model


def load_baseline_model(path: str = "models/baseline_model.joblib") -> LogisticRegression:
    """
    Load saved Logistic Regression baseline model from disk.

    Parameters:
    -----------
    path : str
        Path to the saved joblib file.

    Returns:
    --------
    LogisticRegression
        Loaded model ready for inference.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline model not found at: {path}")
    return joblib.load(path)


def predict_baseline(
    model: LogisticRegression,
    X: spmatrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict binary class labels and probability of Helpful (class 1).

    Parameters:
    -----------
    model : LogisticRegression
        Trained model.
    X : spmatrix
        TF-IDF features.

    Returns:
    --------
    y_pred : np.ndarray
        Binary predictions (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for Helpful class (class 1).
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


if __name__ == "__main__":
    print("Testing baseline_model module...")
    from scipy.sparse import csr_matrix
    X_toy = csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.1, 0.9]])
    y_toy = np.array([1, 0, 1, 0])
    clf = train_baseline_model(X_toy, y_toy, save_path=None)
    preds, probs = predict_baseline(clf, X_toy)
    print("Predictions:", preds)
    print("Probabilities:", probs)
    print("Baseline model test passed successfully!")
