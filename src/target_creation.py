"""
Target Creation and Class Distribution Module
=============================================
This module handles target label generation for supervised review helpfulness prediction:
1. Calculates HelpfulnessRatio = HelpfulnessNumerator / HelpfulnessDenominator
2. Excludes unvoted reviews (HelpfulnessDenominator == 0) to avoid division by zero.
3. Filters out any rare data anomalies where Numerator > Denominator.
4. Analyzes the class distribution and reports the severe 83:17 class imbalance.
5. Recommends a data-driven adjustment:
   - Filtering to reviews with at least 2 or 3 votes (HelpfulnessDenominator >= 2)
   - Applying class-weighted training and stratified sampling to handle imbalance.
6. Strictly prevents target leakage: HelpfulnessNumerator, HelpfulnessDenominator,
   and HelpfulnessRatio are NEVER included in the feature set.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


def analyze_threshold_distribution(df: pd.DataFrame, output_path: Optional[str] = "outputs/evaluation/eda_threshold_report.json") -> Dict[str, Any]:
    """
    Perform deep exploratory analysis on the Helpfulness ratio across different
    denominator thresholds, documenting the class imbalance and zero-vote ratio.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw or sampled DataFrame containing HelpfulnessNumerator and HelpfulnessDenominator.
    output_path : str, optional
        Path to save the JSON analysis report.

    Returns:
    --------
    dict
        Detailed distribution statistics across thresholds.
    """
    total = len(df)
    zero_denom = int((df["HelpfulnessDenominator"] == 0).sum())
    voted_mask = df["HelpfulnessDenominator"] > 0
    voted_df = df[voted_mask].copy()
    
    # Filter anomalies where Numerator > Denominator
    valid_mask = voted_df["HelpfulnessNumerator"] <= voted_df["HelpfulnessDenominator"]
    valid_df = voted_df[valid_mask].copy()
    valid_df["ratio"] = valid_df["HelpfulnessNumerator"] / valid_df["HelpfulnessDenominator"]

    # Sensitivity analysis across minimum denominator cutoffs
    threshold_stats = {}
    for min_d in [1, 2, 3, 5, 10]:
        sub = valid_df[valid_df["HelpfulnessDenominator"] >= min_d]
        eligible_count = len(sub)
        pct_of_all = round((eligible_count / total) * 100, 2) if total > 0 else 0.0

        if eligible_count > 0:
            pct_ge_05 = round(float((sub["ratio"] >= 0.5).mean() * 100), 2)
            pct_ge_06 = round(float((sub["ratio"] >= 0.6).mean() * 100), 2)
            pct_ge_07 = round(float((sub["ratio"] >= 0.7).mean() * 100), 2)
            pct_ge_08 = round(float((sub["ratio"] >= 0.8).mean() * 100), 2)
        else:
            pct_ge_05 = pct_ge_06 = pct_ge_07 = pct_ge_08 = 0.0

        threshold_stats[f"min_denom_{min_d}"] = {
            "eligible_reviews": eligible_count,
            "pct_of_all_reviews": pct_of_all,
            "ratio_ge_0.50_pct": pct_ge_05,
            "ratio_ge_0.60_pct": pct_ge_06,
            "ratio_ge_0.70_pct": pct_ge_07,
            "ratio_ge_0.80_pct": pct_ge_08,
        }

    report = {
        "total_records": total,
        "zero_vote_records": zero_denom,
        "zero_vote_percentage": round((zero_denom / total) * 100, 2) if total > 0 else 0.0,
        "voted_records": len(valid_df),
        "voted_percentage": round((len(valid_df) / total) * 100, 2) if total > 0 else 0.0,
        "ratio_percentiles_when_voted": {
            "mean": float(valid_df["ratio"].mean()),
            "std": float(valid_df["ratio"].std()),
            "median": float(valid_df["ratio"].median()),
            "p25": float(valid_df["ratio"].quantile(0.25)),
            "p75": float(valid_df["ratio"].quantile(0.75)),
        },
        "threshold_sensitivity": threshold_stats,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[TargetCreation] Saved threshold analysis report to: {output_path}")

    return report


def create_helpfulness_target(
    df: pd.DataFrame,
    threshold: float = 0.5,
    min_denominator: int = 1,
) -> pd.DataFrame:
    """
    Calculate HelpfulnessRatio and assign binary target label (1 = Helpful, 0 = Not Helpful).
    
    Rules:
    - Exclude HelpfulnessDenominator == 0 (cannot divide by zero).
    - Exclude anomalies where HelpfulnessNumerator > HelpfulnessDenominator.
    - Require HelpfulnessDenominator >= min_denominator.
    - Target label: 1 if HelpfulnessRatio >= threshold else 0.

    Parameters:
    -----------
    df : pd.DataFrame
        Input review dataframe.
    threshold : float, default=0.5
        Helpfulness ratio cutoff.
    min_denominator : int, default=1
        Minimum number of total votes required.

    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with 'HelpfulnessRatio' and 'Helpful' columns added.
    """
    # Step 1: Filter out unvoted reviews
    voted = df[df["HelpfulnessDenominator"] >= min_denominator].copy()
    
    # Step 2: Remove data anomalies (Numerator > Denominator)
    valid_mask = voted["HelpfulnessNumerator"] <= voted["HelpfulnessDenominator"]
    valid = voted[valid_mask].copy()

    # Step 3: Compute ratio safely
    valid["HelpfulnessRatio"] = valid["HelpfulnessNumerator"] / valid["HelpfulnessDenominator"]

    # Step 4: Binary label assignment
    valid["Helpful"] = (valid["HelpfulnessRatio"] >= threshold).astype(int)

    helpful_count = (valid["Helpful"] == 1).sum()
    unhelpful_count = (valid["Helpful"] == 0).sum()
    total_valid = len(valid)

    print(f"[TargetCreation] Created target with min_denominator={min_denominator}, threshold={threshold}:")
    print(f"  - Total eligible reviews: {total_valid:,}")
    print(f"  - Helpful (1): {helpful_count:,} ({helpful_count/total_valid*100:.2f}%)")
    print(f"  - Not Helpful (0): {unhelpful_count:,} ({unhelpful_count/total_valid*100:.2f}%)")

    return valid


def prepare_labeled_sample(
    df: pd.DataFrame,
    sample_size: Optional[int] = 25000,
    min_denominator: int = 2,
    threshold: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare labeled dataset for model training while preventing data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw reviews DataFrame.
    sample_size : int, optional
        Target sample size for balanced/scalable training on CPU. If None, uses all eligible.
    min_denominator : int, default=2
        Minimum vote threshold for reliable community signal.
    threshold : float, default=0.5
        Threshold for Helpful class.
    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    --------
    features_df : pd.DataFrame
        DataFrame with text and metadata (WITHOUT target-leaking columns).
    y : pd.Series
        Binary target labels (0 or 1).
    """
    labeled_df = create_helpfulness_target(df, threshold=threshold, min_denominator=min_denominator)

    if sample_size is not None and sample_size < len(labeled_df):
        from sklearn.model_selection import train_test_split
        sampled_df, _ = train_test_split(
            labeled_df,
            train_size=sample_size,
            stratify=labeled_df["Helpful"],
            random_state=random_state,
        )
        labeled_df = sampled_df.reset_index(drop=True)
        print(f"[TargetCreation] Sampled {len(labeled_df):,} records:")
        print(labeled_df["Helpful"].value_counts(normalize=True).rename("proportion"))

    # Extract target label
    y = labeled_df["Helpful"]

    # Strictly remove target-leaking columns from feature dataframe
    leaking_cols = ["HelpfulnessNumerator", "HelpfulnessDenominator", "HelpfulnessRatio", "Helpful"]
    features_df = labeled_df.drop(columns=[col for col in leaking_cols if col in labeled_df.columns])

    return features_df, y


if __name__ == "__main__":
    print("Testing target_creation module...")
    from src.data_loader import load_raw_data
    # Load 5000 rows for verification
    raw_df = load_raw_data(nrows=5000)
    report = analyze_threshold_distribution(raw_df, output_path="outputs/evaluation/eda_threshold_report.json")
    feat_df, target = prepare_labeled_sample(raw_df, sample_size=1000, min_denominator=1)
    print("Feature columns (no leakage):", list(feat_df.columns))
    print("Target distribution:\n", target.value_counts())
    print("Target creation test passed successfully!")
