"""
Explainability Module (LIME Feature Attribution)
================================================
This module implements Local Interpretable Model-agnostic Explanations (LIME)
for text classification.

Beginner-friendly explanation:
------------------------------
LIME explains individual predictions by perturbing the review text (turning words on/off),
observing changes in the model's output probabilities, and fitting a local interpretable
linear surrogate to quantify each word's contribution to:
- Positive contribution (weight > 0): pushes the prediction towards "Helpful"
- Negative contribution (weight < 0): pushes the prediction towards "Not Helpful"
"""

import os
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np
from lime.lime_text import LimeTextExplainer

from src.preprocessing import clean_text
from src.lstm_model import text_to_padded_sequences


class ReviewHelpfulnessLIMEExplainer:
    """
    Wrapper for LIME text explainability for both Baseline and LSTM models.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        if class_names is None:
            class_names = ["Not Helpful", "Helpful"]
        self.class_names = class_names
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            bow=False,
            split_expression=r"\W+",
            random_state=42,
        )

    @staticmethod
    def get_baseline_predict_fn(vectorizer, model) -> Callable[[List[str]], np.ndarray]:
        """
        Create a prediction function compatible with LIME for the Baseline model.
        LIME passes a list of text strings. The function must return an (N, 2) probability array.
        """
        def predict_fn(texts: List[str]) -> np.ndarray:
            # Clean each string
            cleaned = [clean_text(t, remove_stopwords=True) for t in texts]
            X_tfidf = vectorizer.transform(cleaned)
            probs = model.predict_proba(X_tfidf)
            return probs

        return predict_fn

    @staticmethod
    def get_lstm_predict_fn(tokenizer, model, maxlen: int = 150) -> Callable[[List[str]], np.ndarray]:
        """
        Create a prediction function compatible with LIME for the LSTM model.
        Returns an (N, 2) probability array [P(Not Helpful), P(Helpful)].
        """
        def predict_fn(texts: List[str]) -> np.ndarray:
            cleaned = [clean_text(t, remove_stopwords=True) for t in texts]
            padded = text_to_padded_sequences(cleaned, tokenizer, maxlen=maxlen)
            helpful_prob = model.predict(padded, batch_size=128, verbose=0).flatten()
            unhelpful_prob = 1.0 - helpful_prob
            # Stack into (N, 2)
            probs = np.column_stack([unhelpful_prob, helpful_prob])
            return probs

        return predict_fn

    def explain_review(
        self,
        text: str,
        predict_fn: Callable[[List[str]], np.ndarray],
        num_features: int = 8,
        num_samples: int = 150,
    ) -> Dict[str, Any]:
        """
        Explain a review's prediction using LIME.

        Parameters:
        -----------
        text : str
            Input review text.
        predict_fn : Callable
            Function taking list of strings and returning (N, 2) probabilities.
        num_features : int, default=8
            Number of top contributing words to extract.
        num_samples : int, default=150
            Number of perturbation samples.

        Returns:
        --------
        dict
            Explanation summary including predicted class, probabilities,
            word contributions, and HTML visualization string.
        """
        # Get baseline prediction on the single text
        probs = predict_fn([text])[0]
        unhelpful_prob = float(probs[0])
        helpful_prob = float(probs[1])
        predicted_class_idx = int(np.argmax(probs))
        predicted_label = self.class_names[predicted_class_idx]
        confidence = float(np.max(probs))

        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=[1],  # Explain contribution towards class 1 ("Helpful")
        )

        word_weights = exp.as_list(label=1)
        positive_contributors = [(w, round(wt, 4)) for w, wt in word_weights if wt > 0]
        negative_contributors = [(w, round(wt, 4)) for w, wt in word_weights if wt < 0]

        html_repr = exp.as_html()

        return {
            "predicted_label": predicted_label,
            "predicted_class_idx": predicted_class_idx,
            "confidence": round(confidence, 4),
            "helpful_probability": round(helpful_prob, 4),
            "unhelpful_probability": round(unhelpful_prob, 4),
            "word_weights": word_weights,
            "positive_contributors": positive_contributors,
            "negative_contributors": negative_contributors,
            "html_representation": html_repr,
        }


if __name__ == "__main__":
    print("Testing explainability module...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    toy_texts = [
        "great delicious organic snack wonderful quality",
        "terrible stale broken bag bad taste",
        "excellent fresh roasted beans love it",
        "awful waste of money never buying again",
    ]
    toy_labels = np.array([1, 0, 1, 0])
    vec = TfidfVectorizer().fit(toy_texts)
    clf = LogisticRegression().fit(vec.transform(toy_texts), toy_labels)

    explainer = ReviewHelpfulnessLIMEExplainer()
    p_fn = explainer.get_baseline_predict_fn(vec, clf)
    res = explainer.explain_review("great snack wonderful taste", p_fn, num_features=4, num_samples=50)

    print("Predicted Label:", res["predicted_label"])
    print("Confidence:", res["confidence"])
    print("Positive Contributors:", res["positive_contributors"])
    print("Negative Contributors:", res["negative_contributors"])
    print("Explainability test passed successfully!")
