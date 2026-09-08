"""
End-to-End NLP Training and Evaluation Pipeline
===============================================
Project Title:
"Product Review Helpfulness Prediction with LSTM Memory Networks and Explainable Feature Attribution"

Pipeline Overview:
1. Data Loading: Read Amazon Fine Food Reviews.
2. Target Creation & EDA: Compute HelpfulnessRatio, inspect class distribution,
   handle zero-denominator rows, and save EDA figures.
3. Text Preprocessing: Clean Summary + Text, preserve critical negation words.
4. Stratified Split: 80% Train, 20% Test (stratified by class).
5. Word Clouds: Generate All, Helpful, and Not Helpful word clouds.
6. Baseline Model: Fit TF-IDF on train only, train Logistic Regression, evaluate on test.
7. LSTM Model: Fit Tokenizer on train only, pad sequences, train standard LSTM(64), evaluate on test.
8. Comparative Evaluation: Calculate test set Accuracy, Precision, Recall, F1, ROC-AUC,
   Confusion Matrices, ROC Curves, and Model Comparison bar chart.
9. LIME Explainability: Generate and save feature attribution reports for sample reviews.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import project modules
from src.data_loader import load_raw_data, get_dataset_statistics
from src.target_creation import (
    analyze_threshold_distribution,
    create_helpfulness_target,
    prepare_labeled_sample,
)
from src.preprocessing import clean_text, combine_and_clean_reviews
from src.feature_engineering import build_tfidf_features
from src.baseline_model import train_baseline_model, predict_baseline
from src.lstm_model import (
    build_and_fit_tokenizer,
    text_to_padded_sequences,
    build_lstm_model,
    train_lstm_model,
    predict_lstm,
)
from src.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    plot_training_history,
    save_evaluation_results,
)
from src.wordclouds import generate_all_wordclouds
from src.explainability import ReviewHelpfulnessLIMEExplainer


def generate_eda_figures(df: pd.DataFrame, figures_dir: str = "outputs/figures") -> None:
    """
    Generate initial EDA charts for dataset analysis:
    - Vote distribution
    - Helpfulness ratio distribution
    - Star Rating (Score) vs Helpfulness
    - Class distribution
    """
    os.makedirs(figures_dir, exist_ok=True)
    print("\n[Pipeline - EDA] Generating exploratory data analysis figures...")

    # Filter voted reviews
    voted = df[df["HelpfulnessDenominator"] > 0].copy()
    valid_mask = voted["HelpfulnessNumerator"] <= voted["HelpfulnessDenominator"]
    valid = voted[valid_mask].copy()
    valid["HelpfulnessRatio"] = valid["HelpfulnessNumerator"] / valid["HelpfulnessDenominator"]

    # 1. Helpfulness Ratio Distribution
    plt.figure(figsize=(7, 4.5))
    sns.histplot(valid["HelpfulnessRatio"], bins=20, kde=True, color="#1976D2")
    plt.title("Distribution of Helpfulness Ratio (Voted Reviews)", fontweight="bold")
    plt.xlabel("Helpfulness Ratio (Numerator / Denominator)", fontweight="semibold")
    plt.ylabel("Review Count", fontweight="semibold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "helpfulness_ratio_distribution.png"), dpi=300)
    plt.close()

    # 2. Score vs Helpfulness
    if "Score" in valid.columns:
        plt.figure(figsize=(7, 4.5))
        valid["Helpful_Label"] = (valid["HelpfulnessRatio"] >= 0.5).map({True: "Helpful (>=0.5)", False: "Not Helpful (<0.5)"})
        sns.countplot(data=valid, x="Score", hue="Helpful_Label", palette=["#D32F2F", "#388E3C"])
        plt.title("Product Star Rating (Score) vs Helpfulness", fontweight="bold")
        plt.xlabel("Star Rating (1 to 5)", fontweight="semibold")
        plt.ylabel("Number of Reviews", fontweight="semibold")
        plt.legend(title="Class", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "score_vs_helpfulness.png"), dpi=300)
        plt.close()

    # 3. Class Distribution
    plt.figure(figsize=(6, 4.5))
    class_counts = (valid["HelpfulnessRatio"] >= 0.5).value_counts()
    ax = sns.barplot(
        x=["Helpful (1)", "Not Helpful (0)"],
        y=[class_counts.get(True, 0), class_counts.get(False, 0)],
        palette=["#388E3C", "#D32F2F"],
    )
    plt.title("Class Distribution (Initial Threshold = 0.5)", fontweight="bold")
    plt.ylabel("Review Count", fontweight="semibold")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:,}\n({h/len(valid)*100:.1f}%)", (p.get_x() + p.get_width() / 2.0, h / 2), ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=300)
    plt.close()

    print(f"[Pipeline - EDA] Saved EDA figures to '{figures_dir}/'")


def run_full_pipeline(
    data_path: str = "data/Reviews.csv",
    sample_size: int = 25000,
    min_denominator: int = 2,
    epochs: int = 6,
    batch_size: int = 64,
    random_state: int = 42,
) -> None:
    """
    Run complete NLP project pipeline incrementally.
    """
    print("=" * 80)
    print("PRODUCT REVIEW HELPFULNESS PREDICTION WITH LSTM MEMORY NETWORKS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: Data Loading & Exploration
    # -------------------------------------------------------------------------
    print("\n--- STAGE 1: Data Loading & Dataset Overview ---")
    raw_df = load_raw_data(data_path)
    stats = get_dataset_statistics(raw_df)
    print(f"Dataset Overview Statistics: {stats}")

    # Generate EDA figures and sensitivity report
    generate_eda_figures(raw_df)
    analyze_threshold_distribution(raw_df, output_path="outputs/evaluation/eda_threshold_report.json")

    # -------------------------------------------------------------------------
    # STAGE 2: Target Creation & Validation (Anti-Leakage Safeguard)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 2: Target Creation & Leakage Prevention ---")
    features_df, y = prepare_labeled_sample(
        raw_df,
        sample_size=sample_size,
        min_denominator=min_denominator,
        threshold=0.5,
        random_state=random_state,
    )
    # Memory management: free up raw dataframe
    del raw_df

    # -------------------------------------------------------------------------
    # STAGE 3: Text Preprocessing (Negation Preservation)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 3: Text Preprocessing ---")
    clean_texts = combine_and_clean_reviews(
        features_df,
        summary_col="Summary",
        text_col="Text",
        remove_stopwords=True,
    )

    # Generate Word Clouds
    print("\n--- STAGE 3b: Word Cloud Generation ---")
    generate_all_wordclouds(clean_texts, y, output_dir="outputs/wordclouds")

    # -------------------------------------------------------------------------
    # STAGE 4: Stratified Train/Test Split
    # -------------------------------------------------------------------------
    print("\n--- STAGE 4: Stratified Train/Test Split (80% Train, 20% Test) ---")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        clean_texts,
        y.values,
        test_size=0.20,
        stratify=y.values,
        random_state=random_state,
    )
    print(f"Training split size: {len(X_train_text):,} reviews")
    print(f"Testing split size : {len(X_test_text):,} reviews")
    print(f"Test class distribution: Class 1 = {(y_test == 1).sum():,}, Class 0 = {(y_test == 0).sum():,}")

    # -------------------------------------------------------------------------
    # STAGE 5: Baseline Model (TF-IDF + Logistic Regression)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 5: Baseline Model (TF-IDF + Logistic Regression) ---")
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
        train_texts=X_train_text,
        test_texts=X_test_text,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        save_path="models/tfidf_vectorizer.joblib",
    )

    baseline_clf = train_baseline_model(
        X_train=X_train_tfidf,
        y_train=y_train,
        class_weight="balanced",
        random_state=random_state,
        save_path="models/baseline_model.joblib",
    )

    # Evaluate Baseline on Test Set
    baseline_preds, baseline_probs = predict_baseline(baseline_clf, X_test_tfidf)
    baseline_metrics = calculate_metrics(
        y_true=y_test,
        y_pred=baseline_preds,
        y_prob=baseline_probs,
        model_name="Baseline (TF-IDF + LR)",
    )
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=baseline_preds,
        model_name="Baseline (TF-IDF + LR)",
        save_path="outputs/figures/baseline_confusion_matrix.png",
    )

    # -------------------------------------------------------------------------
    # STAGE 6: Main Model (LSTM Memory Network)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 6: Main Model (Standard LSTM Memory Network) ---")
    tokenizer = build_and_fit_tokenizer(
        train_texts=X_train_text,
        num_words=10000,
        oov_token="<OOV>",
        save_path="models/tokenizer.pkl",
    )

    maxlen = 100
    X_train_pad = text_to_padded_sequences(X_train_text, tokenizer, maxlen=maxlen, padding="pre")
    X_test_pad = text_to_padded_sequences(X_test_text, tokenizer, maxlen=maxlen, padding="pre")

    lstm_net = build_lstm_model(
        vocab_size=10000,
        embedding_dim=64,
        lstm_units=64,
        dense_units=32,
        dropout_rate=0.3,
        input_length=maxlen,
    )

    lstm_net, history = train_lstm_model(
        model=lstm_net,
        X_train_pad=X_train_pad,
        y_train=y_train,
        X_val_pad=X_test_pad,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        save_path="models/lstm_model.keras",
    )

    # Plot training and validation loss/accuracy curves
    plot_training_history(history, save_path="outputs/figures/lstm_training_history.png")

    # Evaluate LSTM on Test Set
    lstm_preds, lstm_probs = predict_lstm(lstm_net, X_test_pad)
    lstm_metrics = calculate_metrics(
        y_true=y_test,
        y_pred=lstm_preds,
        y_prob=lstm_probs,
        model_name="LSTM Memory Network",
    )
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=lstm_preds,
        model_name="LSTM Memory Network",
        save_path="outputs/figures/lstm_confusion_matrix.png",
    )

    # -------------------------------------------------------------------------
    # STAGE 7: Model Comparison & Visualizations
    # -------------------------------------------------------------------------
    print("\n--- STAGE 7: Model Comparison & ROC Analysis ---")
    comparison_list = [baseline_metrics, lstm_metrics]
    comp_df = save_evaluation_results(
        metrics_list=comparison_list,
        json_path="outputs/evaluation/model_comparison.json",
        csv_path="outputs/evaluation/model_comparison.csv",
    )
    plot_model_comparison(comp_df, save_path="outputs/figures/model_comparison_chart.png")
    plot_roc_curves(
        y_true=y_test,
        baseline_probs=baseline_probs,
        lstm_probs=lstm_probs,
        baseline_auc=baseline_metrics["roc_auc"],
        lstm_auc=lstm_metrics["roc_auc"],
        save_path="outputs/figures/roc_curve_comparison.png",
    )

    # -------------------------------------------------------------------------
    # STAGE 8: Explainable Feature Attribution (LIME)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 8: LIME Explainability Demonstration ---")
    explainer = ReviewHelpfulnessLIMEExplainer()
    base_predict_fn = explainer.get_baseline_predict_fn(vectorizer, baseline_clf)
    lstm_predict_fn = explainer.get_lstm_predict_fn(tokenizer, lstm_net, maxlen=maxlen)

    sample_helpful = "Outstanding organic coffee! Rich aroma, smooth dark roast flavor, and very fresh beans. Highly recommended."
    sample_unhelpful = "Terrible. Box arrived smashed, completely stale and bad taste. Waste of money, never buy."

    os.makedirs("outputs/explanations", exist_ok=True)
    exp_helpful = explainer.explain_review(sample_helpful, lstm_predict_fn, num_features=6, num_samples=100)
    with open("outputs/explanations/lime_sample_helpful.html", "w", encoding="utf-8") as f:
        f.write(exp_helpful["html_representation"])

    exp_unhelpful = explainer.explain_review(sample_unhelpful, lstm_predict_fn, num_features=6, num_samples=100)
    with open("outputs/explanations/lime_sample_unhelpful.html", "w", encoding="utf-8") as f:
        f.write(exp_unhelpful["html_representation"])

    print(f"[Pipeline - LIME] Sample Helpful review prediction: {exp_helpful['predicted_label']} (confidence={exp_helpful['confidence']:.2f})")
    print(f"  Top positive words: {exp_helpful['positive_contributors']}")
    print(f"[Pipeline - LIME] Sample Unhelpful review prediction: {exp_unhelpful['predicted_label']} (confidence={exp_unhelpful['confidence']:.2f})")
    print(f"  Top negative words: {exp_unhelpful['negative_contributors']}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY! ALL ARTIFACTS AND MODELS SAVED.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLP Helpfulness Prediction Pipeline")
    parser.add_argument("--sample_size", type=int, default=25000, help="Number of reviews to sample for training")
    parser.add_argument("--min_denominator", type=int, default=2, help="Minimum denominator for target creation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for LSTM")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for LSTM")
    args = parser.parse_args()

    run_full_pipeline(
        sample_size=args.sample_size,
        min_denominator=args.min_denominator,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
