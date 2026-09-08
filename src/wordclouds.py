"""
Word Cloud Generation Module
============================
This module generates visual word clouds for review texts across three categories:
1. All Reviews (general vocabulary)
2. Helpful Reviews (terms frequently associated with high helpfulness votes)
3. Not Helpful Reviews (terms frequently associated with low helpfulness votes)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Optional

from src.preprocessing import CUSTOM_STOPWORDS


def generate_wordcloud(
    text_corpus: str,
    title: str,
    save_path: str,
    colormap: str = "Blues",
    max_words: int = 150,
) -> None:
    """
    Generate and save a single high-resolution word cloud.

    Parameters:
    -----------
    text_corpus : str
        Combined text string.
    title : str
        Plot title.
    save_path : str
        Output image file path.
    colormap : str, default='Blues'
        Matplotlib color palette.
    max_words : int, default=150
        Maximum words to display.
    """
    if not text_corpus.strip():
        print(f"[WordCloud] Warning: Empty text corpus for '{title}'. Skipping.")
        return

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        colormap=colormap,
        stopwords=CUSTOM_STOPWORDS,
        max_words=max_words,
        random_state=42,
        collocations=False,
    ).generate(text_corpus)

    plt.figure(figsize=(10, 5.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, pad=15, fontweight="bold")
    plt.tight_layout(pad=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[WordCloud] Saved wordcloud to: {save_path}")


def generate_all_wordclouds(
    clean_texts: pd.Series,
    labels: pd.Series,
    output_dir: str = "outputs/wordclouds",
) -> None:
    """
    Generate all 3 required word clouds:
    1. All reviews
    2. Helpful reviews (Helpful == 1)
    3. Not Helpful reviews (Helpful == 0)

    Parameters:
    -----------
    clean_texts : pd.Series
        Cleaned review text series.
    labels : pd.Series
        Binary helpfulness labels (0 or 1).
    output_dir : str
        Directory to save wordcloud image files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[WordCloud] Generating Word Cloud for ALL reviews...")
    all_text = " ".join(clean_texts.astype(str))
    generate_wordcloud(
        all_text,
        title="Word Cloud: All Reviews",
        save_path=os.path.join(output_dir, "all_reviews_wordcloud.png"),
        colormap="Blues",
    )

    print("[WordCloud] Generating Word Cloud for HELPFUL reviews...")
    helpful_mask = labels == 1
    helpful_text = " ".join(clean_texts[helpful_mask].astype(str))
    generate_wordcloud(
        helpful_text,
        title="Word Cloud: Helpful Reviews (HelpfulnessRatio >= 0.5)",
        save_path=os.path.join(output_dir, "helpful_wordcloud.png"),
        colormap="Greens",
    )

    print("[WordCloud] Generating Word Cloud for NOT HELPFUL reviews...")
    not_helpful_mask = labels == 0
    not_helpful_text = " ".join(clean_texts[not_helpful_mask].astype(str))
    generate_wordcloud(
        not_helpful_text,
        title="Word Cloud: Not Helpful Reviews (HelpfulnessRatio < 0.5)",
        save_path=os.path.join(output_dir, "not_helpful_wordcloud.png"),
        colormap="Reds",
    )


if __name__ == "__main__":
    print("Testing wordclouds module...")
    sample_texts = pd.Series([
        "great flavor delicious tea wonderful organic product",
        "horrible taste waste of money stale bitter terrible",
    ])
    sample_labels = pd.Series([1, 0])
    generate_all_wordclouds(sample_texts, sample_labels, output_dir="outputs/wordclouds")
    print("Wordclouds module test passed successfully!")
