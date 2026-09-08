"""
Text Preprocessing Module
=========================
This module handles all NLP text cleaning operations in a clean, beginner-friendly manner:
1. Combines review Summary + Text
2. Handles missing text/summary values
3. Converts text to lowercase
4. Removes HTML tags (e.g. <br />, <p>)
5. Removes URLs (http://, https://, www)
6. Removes punctuation and special characters (retaining words and spaces)
7. Normalizes excess whitespace
8. Tokenizes text
9. Removes standard NLTK stopwords while PRESERVING critical negation words
   (e.g., 'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot')
   to maintain review sentiment and semantic integrity.
"""

import re
import html
import nltk
from nltk.corpus import stopwords
import pandas as pd
from typing import List, Optional, Union

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Define critical negation words to preserve in text
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "cannot",
    "without",
    "hardly",
    "barely",
    "scarcely",
    "n't",
}

# Build custom stopword set that excludes negation words
BASE_STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS

# Compiled regular expressions for fast, beginner-friendly cleaning
HTML_TAG_PATTERN = re.compile(r"<.*?>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
PUNCTUATION_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Union[str, float], remove_stopwords: bool = True) -> str:
    """
    Clean a single text string through beginner-friendly standard NLP steps.

    Parameters:
    -----------
    text : str or float
        Raw review text (handles None / NaN safely).
    remove_stopwords : bool, default=True
        Whether to filter out stopwords (preserving negations).

    Returns:
    --------
    str
        Cleaned text string.
    """
    # 1. Handle missing values
    if not isinstance(text, str) or text is None:
        return ""

    # 2. Unescape HTML entities (e.g., &amp; -> &, &quot; -> ")
    cleaned = html.unescape(text)

    # 3. Remove HTML tags (<br />, <p>, etc.)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)

    # 4. Remove URLs
    cleaned = URL_PATTERN.sub(" ", cleaned)

    # 5. Convert to lowercase
    cleaned = cleaned.lower()

    # 6. Remove special characters, digits, and punctuation (keep letters and spaces)
    cleaned = PUNCTUATION_PATTERN.sub(" ", cleaned)

    # 7. Normalize whitespace
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    # 8. Tokenize and optionally remove stopwords while keeping negations
    if remove_stopwords:
        tokens = cleaned.split()
        filtered_tokens = [token for token in tokens if token not in CUSTOM_STOPWORDS]
        cleaned = " ".join(filtered_tokens)

    return cleaned


def combine_and_clean_reviews(
    df: pd.DataFrame,
    summary_col: str = "Summary",
    text_col: str = "Text",
    remove_stopwords: bool = True,
) -> pd.Series:
    """
    Combine Summary + Text into a single clean text series.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing review columns.
    summary_col : str, default='Summary'
        Name of summary column.
    text_col : str, default='Text'
        Name of text column.
    remove_stopwords : bool, default=True
        Whether to remove stopwords (keeping negations).

    Returns:
    --------
    pd.Series
        Cleaned, combined review text strings.
    """
    # Safely handle missing text / summary
    summary_series = df[summary_col].fillna("").astype(str) if summary_col in df.columns else pd.Series([""] * len(df))
    text_series = df[text_col].fillna("").astype(str) if text_col in df.columns else pd.Series([""] * len(df))

    # Combine Summary and Text with a space/period
    print(f"[Preprocessing] Combining '{summary_col}' and '{text_col}' columns...")
    combined_raw = summary_series.str.cat(text_series, sep=". ").str.strip()

    # Apply text cleaning
    print(f"[Preprocessing] Cleaning {len(combined_raw):,} review texts...")
    cleaned_series = combined_raw.apply(lambda t: clean_text(t, remove_stopwords=remove_stopwords))
    
    return cleaned_series


if __name__ == "__main__":
    print("Testing preprocessing module...")
    sample_text = (
        "<html>Great product! But it did <b>NOT</b> arrive on time. "
        "Visit https://amazon.com for info. I would never buy again! &amp; price was $25.99.</html>"
    )
    cleaned = clean_text(sample_text)
    print("Original text:\n", sample_text)
    print("Cleaned text:\n", cleaned)

    # Validate that negation words ('not', 'never') are preserved
    assert "not" in cleaned, "Error: 'not' was incorrectly removed!"
    assert "never" in cleaned, "Error: 'never' was incorrectly removed!"
    assert "<" not in cleaned and ">" not in cleaned, "Error: HTML tags were not removed!"
    assert "http" not in cleaned, "Error: URLs were not removed!"
    print("Preprocessing test passed successfully! Negations preserved.")
