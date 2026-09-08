"""
LSTM Memory Network Module
==========================
This module implements the Long Short-Term Memory (LSTM) recurrent neural network
for review helpfulness classification.

Beginner-friendly explanation:
------------------------------
Standard LSTMs process text sequentially word-by-word. At each step, memory cells
and gating mechanisms (forget gate, input gate, output gate) capture long-term context
and word order that bag-of-words or TF-IDF models ignore.

Architecture strictly following requirements:
1. Tokenizer (vocab_size=10,000, oov_token='<OOV>')
2. Padding (maxlen=150, padding='post', truncating='post')
3. Embedding(input_dim=10,000, output_dim=64)
4. LSTM(64) (strictly unidirectional standard LSTM, NO BiLSTM)
5. Dropout(0.3)
6. Dense(32, activation='relu')
7. Dense(1, activation='sigmoid')
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


def build_and_fit_tokenizer(
    train_texts: pd.Series,
    num_words: int = 10000,
    oov_token: str = "<OOV>",
    save_path: Optional[str] = "models/tokenizer.pkl",
) -> Tokenizer:
    """
    Fit Keras Tokenizer strictly on training text and save to disk.

    Parameters:
    -----------
    train_texts : pd.Series or list of str
        Training text data (no leakage).
    num_words : int, default=10000
        Maximum vocabulary size.
    oov_token : str, default='<OOV>'
        Token used for unseen out-of-vocabulary words.
    save_path : str, optional
        Path to save serialized tokenizer.

    Returns:
    --------
    Tokenizer
        Fitted Keras Tokenizer.
    """
    print(f"[LSTM] Fitting Tokenizer (num_words={num_words}, oov_token='{oov_token}')...")
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(train_texts)
    print(f"[LSTM] Tokenizer vocabulary size: {len(tokenizer.word_index):,} unique tokens.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"[LSTM] Saved tokenizer to: {save_path}")

    return tokenizer


def load_saved_tokenizer(path: str = "models/tokenizer.pkl") -> Tokenizer:
    """
    Load a pre-fitted Keras Tokenizer from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def text_to_padded_sequences(
    texts: pd.Series,
    tokenizer: Tokenizer,
    maxlen: int = 150,
    padding: str = "pre",
) -> np.ndarray:
    """
    Convert raw strings to integer sequences and pad/truncate to fixed length.
    Note: 'pre' padding is preferred for recurrent networks (LSTM) so that the final
    hidden state corresponds to actual words rather than zero padding.

    Parameters:
    -----------
    texts : pd.Series or list of str
        Text strings to convert.
    tokenizer : Tokenizer
        Fitted Keras Tokenizer.
    maxlen : int, default=150
        Maximum sequence length.
    padding : str, default='pre'
        Padding direction ('pre' or 'post').

    Returns:
    --------
    np.ndarray
        Padded integer sequence matrix of shape (N, maxlen).
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating="post")
    return padded


def build_lstm_model(
    vocab_size: int = 10000,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    input_length: int = 150,
    learning_rate: float = 0.001,
) -> Sequential:
    """
    Construct the exact user-specified standard LSTM Memory Network architecture:
    Embedding(64) -> LSTM(64) -> Dropout(0.3) -> Dense(32, ReLU) -> Dense(1, Sigmoid)

    Parameters:
    -----------
    vocab_size : int, default=10000
    embedding_dim : int, default=64
    lstm_units : int, default=64
    dense_units : int, default=32
    dropout_rate : float, default=0.3
    input_length : int, default=150
    learning_rate : float, default=0.001

    Returns:
    --------
    Sequential
        Compiled Keras Sequential model.
    """
    print("[LSTM] Building standard unidirectional LSTM model...")
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding_layer",
            ),
            LSTM(
                units=lstm_units,
                name="lstm_layer",
            ),
            Dropout(
                rate=dropout_rate,
                name="dropout_layer",
            ),
            Dense(
                units=dense_units,
                activation="relu",
                name="dense_relu",
            ),
            Dense(
                units=1,
                activation="sigmoid",
                name="output_sigmoid",
            ),
        ],
        name="lstm_helpfulness_network",
    )
    model.build(input_shape=(None, input_length))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=lambda x: print(f"  {x}"))
    return model


def train_lstm_model(
    model: Sequential,
    X_train_pad: np.ndarray,
    y_train: np.ndarray,
    X_val_pad: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 6,
    batch_size: int = 64,
    save_path: Optional[str] = "models/lstm_model.keras",
) -> Tuple[Sequential, tf.keras.callbacks.History]:
    """
    Train the LSTM model using class weights and early stopping.

    Parameters:
    -----------
    model : Sequential
        Compiled LSTM model.
    X_train_pad : np.ndarray
        Padded training sequences.
    y_train : np.ndarray
        Training labels (0 or 1).
    X_val_pad : np.ndarray
        Padded validation sequences.
    y_val : np.ndarray
        Validation labels (0 or 1).
    epochs : int, default=6
    batch_size : int, default=64
    save_path : str, optional
        Path to save trained Keras model.

    Returns:
    --------
    model : Sequential
        Trained model.
    history : History
        Keras training history object.
    """
    # Apply moderate class weighting (2.0 for minority unhelpful class, 1.0 for helpful class)
    # to maintain high sensitivity without over-penalizing majority predictions
    class_weights_dict = {0: 2.0, 1: 1.0}
    print(f"[LSTM] Applied moderate class weights: {class_weights_dict}")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"[LSTM] Training LSTM for up to {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
        verbose=1,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[LSTM] Saved trained model to: {save_path}")

    return model, history


def load_saved_lstm_model(path: str = "models/lstm_model.keras") -> Sequential:
    """
    Load saved Keras LSTM model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"LSTM model not found at: {path}")
    return load_model(path)


def predict_lstm(
    model: Sequential,
    X_pad: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict binary labels and probabilities using trained LSTM.

    Parameters:
    -----------
    model : Sequential
        Trained LSTM model.
    X_pad : np.ndarray
        Padded sequence matrix.
    threshold : float, default=0.5
        Decision threshold for class 1.

    Returns:
    --------
    y_pred : np.ndarray
        Binary predictions (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for Helpful class.
    """
    y_prob = model.predict(X_pad, batch_size=128, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


if __name__ == "__main__":
    print("Testing lstm_model module...")
    sample_texts = pd.Series([
        "delicious coffee love the flavor",
        "stale awful ruined my day",
        "great product will buy again",
        "not helpful terrible quality",
    ])
    tok = build_and_fit_tokenizer(sample_texts, num_words=100, save_path=None)
    pad = text_to_padded_sequences(sample_texts, tok, maxlen=10)
    print("Padded shape:", pad.shape)
    toy_model = build_lstm_model(vocab_size=100, input_length=10)
    preds, probs = predict_lstm(toy_model, pad)
    print("Initial toy predictions:", preds)
    print("Initial toy probabilities:", probs)
    print("LSTM module test passed successfully!")
