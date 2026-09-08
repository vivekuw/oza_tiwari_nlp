"""
Data Loader Module
==================
This module is responsible for loading the Amazon Fine Food Reviews dataset,
verifying required columns, handling missing text/summary values, and providing
summary statistics about the dataset.

Dataset Source: Stanford SNAP / Kaggle
URL: https://snap.stanford.edu/data/web-FineFoods.html
     https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
"""

import os
import pandas as pd
from typing import Optional, List


# Columns present in the raw Reviews.csv
EXPECTED_COLUMNS = [
    "Id",
    "ProductId",
    "UserId",
    "ProfileName",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
    "Score",
    "Time",
    "Summary",
    "Text",
]


def load_raw_data(
    file_path: str = "data/Reviews.csv",
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load raw review data from CSV with error handling and validation.

    Parameters:
    -----------
    file_path : str
        Path to the Reviews.csv file.
    nrows : int, optional
        Number of rows to load (useful for testing and fast prototyping).
    usecols : list of str, optional
        Specific columns to read to save memory.

    Returns:
    --------
    pd.DataFrame
        Loaded pandas DataFrame with missing values filled.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path}. "
            f"Please ensure Reviews.csv is located in the data/ directory."
        )

    print(f"[DataLoader] Reading dataset from '{file_path}'...")
    df = pd.read_csv(
        file_path,
        nrows=nrows,
        usecols=usecols,
        dtype={
            "HelpfulnessNumerator": "int32",
            "HelpfulnessDenominator": "int32",
            "Score": "int32",
        },
    )
    print(f"[DataLoader] Successfully loaded {len(df):,} records with columns: {list(df.columns)}")

    # Handle missing values in text columns safely
    if "Summary" in df.columns:
        null_summaries = df["Summary"].isnull().sum()
        if null_summaries > 0:
            print(f"[DataLoader] Found {null_summaries} missing Summary values. Filling with empty string.")
            df["Summary"] = df["Summary"].fillna("")

    if "Text" in df.columns:
        null_texts = df["Text"].isnull().sum()
        if null_texts > 0:
            print(f"[DataLoader] Found {null_texts} missing Text values. Filling with empty string.")
            df["Text"] = df["Text"].fillna("")

    return df


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Compute basic dataset statistics for EDA and reporting.

    Parameters:
    -----------
    df : pd.DataFrame
        Loaded review DataFrame.

    Returns:
    --------
    dict
        Dictionary containing key dataset metrics.
    """
    total_reviews = len(df)
    unique_users = df["UserId"].nunique() if "UserId" in df.columns else 0
    unique_products = df["ProductId"].nunique() if "ProductId" in df.columns else 0
    zero_votes = (df["HelpfulnessDenominator"] == 0).sum() if "HelpfulnessDenominator" in df.columns else 0
    voted_reviews = total_reviews - zero_votes

    return {
        "total_reviews": int(total_reviews),
        "unique_users": int(unique_users),
        "unique_products": int(unique_products),
        "zero_vote_reviews": int(zero_votes),
        "zero_vote_percentage": round((zero_votes / total_reviews) * 100, 2) if total_reviews > 0 else 0.0,
        "voted_reviews": int(voted_reviews),
        "voted_percentage": round((voted_reviews / total_reviews) * 100, 2) if total_reviews > 0 else 0.0,
    }


if __name__ == "__main__":
    # Test data loading on a small sample
    print("Testing data_loader module...")
    sample_df = load_raw_data(nrows=1000)
    stats = get_dataset_statistics(sample_df)
    print("Dataset sample statistics:", stats)
    print("Data loader test passed successfully!")
