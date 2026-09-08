"""
Streamlit Web Dashboard: Product Review Helpfulness Prediction
==============================================================
Project Title:
"Product Review Helpfulness Prediction with LSTM Memory Networks and Explainable Feature Attribution"

Satisfies all 16 user requirements:
1. Project title
2. Dataset overview
3. Key statistics
4. Interactive EDA charts
5. Word Cloud
6. Helpful/Not Helpful class distribution
7. Model performance comparison
8. Confusion Matrix
9. ROC Curve
10. New review prediction
11. LIME explanation
12. Key findings
13. Challenges
14. Applications
15. Future scope
16. Conclusion
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Product Review Helpfulness Prediction",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F9FAFB, #F3F4F6);
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .callout-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .badge-helpful {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-unhelpful {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_tools():
    """
    Load pre-trained models and preprocessing tools cached in memory.
    """
    from src.baseline_model import load_baseline_model
    from src.feature_engineering import load_tfidf_vectorizer
    from src.lstm_model import load_saved_lstm_model, load_saved_tokenizer
    from src.explainability import ReviewHelpfulnessLIMEExplainer

    vectorizer = load_tfidf_vectorizer("models/tfidf_vectorizer.joblib")
    baseline_clf = load_baseline_model("models/baseline_model.joblib")
    tokenizer = load_saved_tokenizer("models/tokenizer.pkl")
    lstm_net = load_saved_lstm_model("models/lstm_model.keras")
    explainer = ReviewHelpfulnessLIMEExplainer()

    return vectorizer, baseline_clf, tokenizer, lstm_net, explainer


@st.cache_data
def load_evaluation_data():
    """
    Load saved evaluation JSON and EDA threshold statistics.
    """
    comp_data = []
    if os.path.exists("outputs/evaluation/model_comparison.json"):
        with open("outputs/evaluation/model_comparison.json", "r", encoding="utf-8") as f:
            comp_data = json.load(f)

    eda_report = {}
    if os.path.exists("outputs/evaluation/eda_threshold_report.json"):
        with open("outputs/evaluation/eda_threshold_report.json", "r", encoding="utf-8") as f:
            eda_report = json.load(f)

    return comp_data, eda_report


# =============================================================================
# HEADER SECTION
# =============================================================================
st.markdown('<div class="main-title">Product Review Helpfulness Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">LSTM Memory Networks & Explainable Feature Attribution (LIME) on Amazon Fine Food Reviews</div>',
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation")
menu_choice = st.sidebar.radio(
    "Go to Section:",
    [
        "1. Project Overview & Stats",
        "2. Exploratory Data Analysis & Class Balance",
        "3. Word Clouds",
        "4. Model Performance Comparison",
        "5. Diagnostic Curves (Confusion & ROC)",
        "6. Live Review Predictor & LIME Explanation",
        "7. Academic Insights & Conclusion",
    ],
)

# Quick System Status in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
models_exist = (
    os.path.exists("models/baseline_model.joblib")
    and os.path.exists("models/tfidf_vectorizer.joblib")
    and os.path.exists("models/lstm_model.keras")
    and os.path.exists("models/tokenizer.pkl")
)
if models_exist:
    st.sidebar.success("All models & tokenizers loaded")
else:
    st.sidebar.warning("Models are currently training...")

comp_data, eda_report = load_evaluation_data()

# =============================================================================
# TAB 1: OVERVIEW & KEY STATS
# =============================================================================
if menu_choice == "1. Project Overview & Stats":
    st.header("Project Overview & Academic Context")
    
    st.markdown(
        """
        ### Academic Background & Problem Statement
        In e-commerce platforms like Amazon, customer purchasing decisions are heavily influenced by user-submitted reviews. 
        However, with hundreds of reviews per product, shoppers face information overload. E-commerce platforms rely on a 
        **Helpfulness Voting Mechanism** where users vote whether a review was helpful or not.
        
        **Goal:** Predict whether an Amazon fine food review is **Helpful (1)** or **Not Helpful (0)** using only its textual content 
        (`Summary` and `Text`).
        
        - **Dataset Source**: Stanford Network Analysis Project (SNAP) & Kaggle
        - **URL**: [https://snap.stanford.edu/data/web-FineFoods.html](https://snap.stanford.edu/data/web-FineFoods.html) | [Kaggle Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
        - **Total Records**: 568,454 reviews across 74,258 products from 256,059 users.
        """
    )

    st.markdown("### Key Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)

    total_rec = eda_report.get("total_records", 568454)
    voted_rec = eda_report.get("voted_records", 298400)
    zero_rec = eda_report.get("zero_vote_records", 270052)
    zero_pct = eda_report.get("zero_vote_percentage", 47.51)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{total_rec:,}</div>
                <div class="metric-lbl">Total Reviews</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{voted_rec:,}</div>
                <div class="metric-lbl">Voted Reviews (52.5%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{zero_rec:,}</div>
                <div class="metric-lbl">Unvoted Reviews ({zero_pct}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-val">83.2% vs 16.8%</div>
                <div class="metric-lbl">Class Imbalance (H vs NH)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="callout-box">
            <b>Anti-Leakage Safeguard:</b> The features <code>HelpfulnessNumerator</code>, <code>HelpfulnessDenominator</code>, 
            and <code>HelpfulnessRatio</code> are strictly excluded from the feature space. Classification relies purely on the 
            natural language text semantics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Data Schema")
    st.table(
        pd.DataFrame(
            [
                {"Column": "Id", "Type": "Integer", "Description": "Unique review record index (1 to 568,454)"},
                {"Column": "ProductId", "Type": "String", "Description": "Amazon ASIN identifier"},
                {"Column": "UserId", "Type": "String", "Description": "Reviewer user ID"},
                {"Column": "ProfileName", "Type": "String", "Description": "Reviewer profile name"},
                {"Column": "HelpfulnessNumerator", "Type": "Integer", "Description": "Count of users who voted the review helpful (Target calculation only)"},
                {"Column": "HelpfulnessDenominator", "Type": "Integer", "Description": "Total count of users who voted on helpfulness (Target calculation only)"},
                {"Column": "Score", "Type": "Integer", "Description": "Customer rating (1 to 5 stars)"},
                {"Column": "Time", "Type": "Timestamp", "Description": "Review post date"},
                {"Column": "Summary", "Type": "Text", "Description": "Short review title / headline"},
                {"Column": "Text", "Type": "Text", "Description": "Full review body text (Primary NLP input)"},
            ]
        )
    )

# =============================================================================
# TAB 2: EDA & CLASS DISTRIBUTION
# =============================================================================
elif menu_choice == "2. Exploratory Data Analysis & Class Balance":
    st.header("Exploratory Data Analysis & Class Balance Inspection")
    
    st.markdown(
        """
        ### Target Creation Formula
        $$\\text{HelpfulnessRatio} = \\frac{\\text{HelpfulnessNumerator}}{\\text{HelpfulnessDenominator}}$$
        
        - Rows with $\\text{HelpfulnessDenominator} = 0$ (47.51% of dataset) cannot yield a valid ratio due to division by zero and are excluded.
        - Binary target: $\\text{Helpful} = 1$ if $\\text{Ratio} \\ge 0.5$, else $0$.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("outputs/figures/class_distribution.png"):
            st.image("outputs/figures/class_distribution.png", caption="Helpful vs Not Helpful Class Distribution (5:1 Imbalance)")
        elif os.path.exists("outputs/figures/vote_distribution.png"):
            st.image("outputs/figures/vote_distribution.png", caption="Vote Distribution")
    with col2:
        if os.path.exists("outputs/figures/helpfulness_ratio_distribution.png"):
            st.image("outputs/figures/helpfulness_ratio_distribution.png", caption="Helpfulness Ratio Distribution (Concentrated at 1.0)")

    st.markdown("### Sensitivity Analysis across Minimum Vote Cutoffs")
    st.markdown(
        "Reviews with only 1 total vote are noisy (a single vote pushes the ratio to either 1.0 or 0.0). "
        "The table below documents how class proportions behave as we filter by minimum votes:"
    )

    sens_data = [
        {"Min Votes (Denominator)": ">= 1 (All voted)", "Eligible Reviews": "298,400 (52.5%)", "Helpful (>= 0.5)": "83.21%", "Not Helpful (< 0.5)": "16.79%"},
        {"Min Votes (Denominator)": ">= 2", "Eligible Reviews": "185,648 (32.7%)", "Helpful (>= 0.5)": "85.16%", "Not Helpful (< 0.5)": "14.84%"},
        {"Min Votes (Denominator)": ">= 3", "Eligible Reviews": "124,167 (21.8%)", "Helpful (>= 0.5)": "82.51%", "Not Helpful (< 0.5)": "17.49%"},
        {"Min Votes (Denominator)": ">= 5", "Eligible Reviews": "67,467 (11.9%)", "Helpful (>= 0.5)": "81.70%", "Not Helpful (< 0.5)": "18.30%"},
        {"Min Votes (Denominator)": ">= 10", "Eligible Reviews": "24,982 (4.4%)", "Helpful (>= 0.5)": "81.80%", "Not Helpful (< 0.5)": "18.20%"},
    ]
    st.table(pd.DataFrame(sens_data))

    if os.path.exists("outputs/figures/score_vs_helpfulness.png"):
        st.image("outputs/figures/score_vs_helpfulness.png", caption="Product Star Rating vs Review Helpfulness")

# =============================================================================
# TAB 3: WORD CLOUDS
# =============================================================================
elif menu_choice == "3. Word Clouds":
    st.header("Word Clouds: Review Vocabulary Analysis")
    st.markdown(
        """
        Word clouds visualizes the most frequent vocabulary across three subsets:
        1. **All Reviews**: Overall vocabulary of gourmet foods, coffee, tea, flavors, and packaging.
        2. **Helpful Reviews**: Terms in reviews voted helpful by the community (detailed descriptions, taste, ingredients, comparisons).
        3. **Not Helpful Reviews**: Terms in unhelpful reviews (short rants, shipping complaints, generic praise, or extreme negativity).
        """
    )

    wc_type = st.radio("Select Word Cloud Category:", ["All Reviews", "Helpful Reviews", "Not Helpful Reviews"], horizontal=True)

    if wc_type == "All Reviews":
        img_path = "outputs/wordclouds/all_reviews_wordcloud.png"
        caption = "Word Cloud: All Reviews (Filtered Stopwords, Preserved Negations)"
    elif wc_type == "Helpful Reviews":
        img_path = "outputs/wordclouds/helpful_wordcloud.png"
        caption = "Word Cloud: Helpful Reviews (HelpfulnessRatio >= 0.5)"
    else:
        img_path = "outputs/wordclouds/not_helpful_wordcloud.png"
        caption = "Word Cloud: Not Helpful Reviews (HelpfulnessRatio < 0.5)"

    if os.path.exists(img_path):
        st.image(img_path, caption=caption, use_container_width=True)
    else:
        st.info(f"Word cloud image '{img_path}' will be generated by the training pipeline.")

# =============================================================================
# TAB 4: MODEL COMPARISON
# =============================================================================
elif menu_choice == "4. Model Performance Comparison":
    st.header("Model Architecture & Performance Comparison")

    st.markdown(
        """
        ### Models Under Comparison
        1. **Baseline Model: TF-IDF + Logistic Regression**
           - Vectorizer: N-grams (1, 2), max 5,000 features, sublinear TF.
           - Classifier: Scikit-learn LogisticRegression with balanced class weights.
        2. **Main Model: Standard LSTM Memory Network**
           - Tokenizer (10,000 vocab) $\\rightarrow$ Padding (maxlen=150) $\\rightarrow$ `Embedding(64)` $\\rightarrow$ `LSTM(64)` $\\rightarrow$ `Dropout(0.3)` $\\rightarrow$ `Dense(32, ReLU)` $\\rightarrow$ `Dense(1, Sigmoid)`.
           - Optimizer: Adam (lr=0.001), Loss: Binary Crossentropy, EarlyStopping.
        """
    )

    if comp_data:
        st.subheader("Test Set Performance Metrics (Strictly Evaluated on Held-Out Test Set)")
        rows = []
        for item in comp_data:
            rows.append(
                {
                    "Model": item["model_name"],
                    "Accuracy": f"{item['accuracy'] * 100:.2f}%",
                    "Precision": f"{item['precision'] * 100:.2f}%",
                    "Recall": f"{item['recall'] * 100:.2f}%",
                    "F1-Score": f"{item['f1'] * 100:.2f}%",
                    "Macro F1": f"{item.get('macro_f1', 0.0) * 100:.2f}%",
                    "ROC-AUC": f"{item['roc_auc']:.4f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if os.path.exists("outputs/figures/model_comparison_chart.png"):
            st.image("outputs/figures/model_comparison_chart.png", caption="Grouped Bar Chart: Model Performance Comparison")
    else:
        st.info("Evaluation metrics will appear once pipeline completes.")

# =============================================================================
# TAB 5: DIAGNOSTIC CURVES
# =============================================================================
elif menu_choice == "5. Diagnostic Curves (Confusion & ROC)":
    st.header("Diagnostic Curves & Error Analysis")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("outputs/figures/baseline_confusion_matrix.png"):
            st.image("outputs/figures/baseline_confusion_matrix.png", caption="Confusion Matrix: Baseline Model")
    with col2:
        if os.path.exists("outputs/figures/lstm_confusion_matrix.png"):
            st.image("outputs/figures/lstm_confusion_matrix.png", caption="Confusion Matrix: LSTM Model")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        if os.path.exists("outputs/figures/roc_curve_comparison.png"):
            st.image("outputs/figures/roc_curve_comparison.png", caption="Combined ROC Curves (Baseline vs LSTM)")
    with col4:
        if os.path.exists("outputs/figures/lstm_training_history.png"):
            st.image("outputs/figures/lstm_training_history.png", caption="LSTM Training History (Loss & Accuracy across Epochs)")

# =============================================================================
# TAB 6: LIVE PREDICTOR & LIME
# =============================================================================
elif menu_choice == "6. Live Review Predictor & LIME Explanation":
    st.header("Interactive Review Helpfulness Predictor & LIME Attribution")
    st.markdown(
        """
        Enter a custom product review (or choose a preset) and select an NLP model.
        The system will predict its helpfulness and compute **Local Interpretable Model-agnostic Explanations (LIME)** 
        to show exact word contributions!
        """
    )

    # Preset examples
    preset_choice = st.selectbox(
        "Select a Preset Review or write your own below:",
        [
            "-- Custom Review --",
            "Preset 1: Thorough & Informative (Helpful)",
            "Preset 2: Generic Vague Review (Unhelpful)",
            "Preset 3: Harsh But Detailed Critique (Helpful)",
            "Preset 4: One-liner rant with no substance (Unhelpful)",
        ],
    )

    preset_texts = {
        "Preset 1: Thorough & Informative (Helpful)": (
            "Outstanding organic dark roast coffee",
            "I have been brewing espresso for over 5 years. These beans are freshly roasted with an intact bloom. "
            "The crema is thick, hazelnut-colored, with deep chocolate undertones and zero burnt bitterness. "
            "Packaging has a reliable one-way degassing valve. Highly recommended for French press or AeroPress.",
        ),
        "Preset 2: Generic Vague Review (Unhelpful)": (
            "ok",
            "It was ok i guess, kids liked it. Maybe buy again.",
        ),
        "Preset 3: Harsh But Detailed Critique (Helpful)": (
            "Misleading ingredient list and artificial sweetener aftertaste",
            "Do not be fooled by the 'all natural' marketing on the front label. Reading the nutritional panel reveals "
            "maltitol and sucralose as primary sweetening agents. It causes serious digestive discomfort if consumed in "
            "quantities over 25g. The texture is chalky and leaves a lingering bitter chemical aftertaste.",
        ),
        "Preset 4: One-liner rant with no substance (Unhelpful)": (
            "HATE IT",
            "Total junk! Arrived late and broken! Never again!",
        ),
    }

    if preset_choice != "-- Custom Review --":
        default_summary, default_text = preset_texts[preset_choice]
    else:
        default_summary = "Rich aromatic French roast"
        default_text = "The beans have a great sheen without being too oily. Full-bodied taste with notes of dark chocolate."

    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        user_summary = st.text_input("Review Summary (Headline):", value=default_summary)
        selected_model_name = st.selectbox("Select Model:", ["LSTM Memory Network", "Baseline (TF-IDF + LR)"])
        num_features = st.slider("Top LIME Features to Display:", min_value=4, max_value=12, value=6)
    with col_in2:
        user_text = st.text_area("Full Review Text:", value=default_text, height=130)

    if st.button("Predict Helpfulness & Explain with LIME", type="primary"):
        combined_input = f"{user_summary}. {user_text}".strip()
        if not combined_input:
            st.warning("Please enter review text.")
        elif not models_exist:
            st.error("Models are currently still compiling/training. Please wait a moment and try again.")
        else:
            with st.spinner("Computing prediction and generating LIME attribution..."):
                try:
                    vectorizer, baseline_clf, tokenizer, lstm_net, explainer = load_models_and_tools()

                    if selected_model_name == "LSTM Memory Network":
                        predict_fn = explainer.get_lstm_predict_fn(tokenizer, lstm_net)
                    else:
                        predict_fn = explainer.get_baseline_predict_fn(vectorizer, baseline_clf)

                    result = explainer.explain_review(
                        combined_input,
                        predict_fn,
                        num_features=num_features,
                        num_samples=120,
                    )

                    st.markdown("---")
                    st.subheader("Prediction Result")

                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        if result["predicted_label"] == "Helpful":
                            st.markdown('<div class="badge-helpful">PREDICTION: HELPFUL (1)</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="badge-unhelpful">PREDICTION: NOT HELPFUL (0)</div>', unsafe_allow_html=True)
                    with c2:
                        st.metric("Model Confidence", f"{result['confidence'] * 100:.1f}%")
                    with c3:
                        st.write(f"**P(Helpful):** `{result['helpful_probability']:.3f}` | **P(Not Helpful):** `{result['unhelpful_probability']:.3f}`")
                        st.progress(result["helpful_probability"])

                    st.markdown("---")
                    st.subheader("Explainable Feature Attribution (LIME)")
                    st.markdown(
                        "Green bars represent words that push the prediction towards **Helpful**. "
                        "Red bars represent words that push the prediction towards **Not Helpful**."
                    )

                    pos_words = result["positive_contributors"]
                    neg_words = result["negative_contributors"]

                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.write("**Top Positive Features (Pushes towards Helpful):**")
                        if pos_words:
                            df_pos = pd.DataFrame(pos_words, columns=["Word", "LIME Weight"])
                            st.dataframe(df_pos, use_container_width=True)
                        else:
                            st.write("None")
                    with col_neg:
                        st.write("**Top Negative Features (Pushes towards Not Helpful):**")
                        if neg_words:
                            df_neg = pd.DataFrame(neg_words, columns=["Word", "LIME Weight"])
                            st.dataframe(df_neg, use_container_width=True)
                        else:
                            st.write("None")

                    st.markdown("#### LIME Interactive HTML Highlight")
                    st.components.v1.html(result["html_representation"], height=350, scrolling=True)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

# =============================================================================
# TAB 7: ACADEMIC DISCUSSION & CONCLUSION
# =============================================================================
elif menu_choice == "7. Academic Insights & Conclusion":
    st.header("Academic Synthesis & Project Discussion")

    st.markdown(
        """
        ### 1. Key Findings
        - **Vote Scarcity**: 47.51% of all reviews receive exactly 0 helpfulness votes, indicating severe sparsity in explicit user feedback on e-commerce platforms.
        - **Class Imbalance**: Among voted reviews, positive helpfulness votes dominate (~83:17 ratio). Supervised models without class weighting degenerate into majority-class classifiers.
        - **Linguistic Predictors of Helpfulness**:
          - Informative reviews frequently feature domain-specific terminology (e.g., roast profile, packaging details, ingredients, specific brewing instructions).
          - Unhelpful reviews frequently consist of very short, emotional outbursts (e.g., "awful", "hate it", "damaged in mail") that provide little value to subsequent buyers.
        - **Sequential vs Linear Modeling**:
          - The Baseline Logistic Regression provides high interpretability and strong linear separation via TF-IDF n-grams.
          - The standard LSTM memory network successfully captures word sequence dependencies and sentiment qualifiers (e.g., negations such as "not good").

        ### 2. Challenges Encountered
        - **Target Leakage Risk**: Because the target label is derived from `HelpfulnessNumerator` and `HelpfulnessDenominator`, these raw columns must be strictly excluded from input features to avoid trivial target leakage.
        - **Small Vote Noise**: Reviews with only 1 total vote ($1/1 = 1.0$ or $0/1 = 0.0$) introduce high variance. Filtering by minimum denominator threshold ($\ge 2$) provides much cleaner signal.
        - **Subjectivity**: Helpfulness is intrinsically subjective; a concise review might be helpful for one user but inadequate for another.

        ### 3. Real-World Applications
        - **Intelligent Review Sorting**: E-commerce platforms can sort newly posted reviews by predicted helpfulness before any community votes accumulate ("cold start" review ranking).
        - **Review Quality Feedback for Authors**: Provide real-time suggestions to reviewers (e.g., "Add specific details about flavor, packaging, or brewing to make your review more helpful").
        - **Spam & Low-Effort Content Filtering**: Automatically flag superficial one-liners for secondary moderation.

        ### 4. Future Scope
        - Incorporating reviewer reputation and historical helpfulness track record.
        - Multilingual extension across non-English product markets.
        - Exploring aspect-based sentiment extraction to highlight specific product attributes (price, durability, flavor).

        ### 5. Conclusion
        This project successfully demonstrated an end-to-end beginner-friendly NLP pipeline for review helpfulness prediction:
        1. Explored 568,454 Amazon reviews and formulated a leakage-free binary helpfulness target.
        2. Cleaned text and preserved critical negation words to maintain semantic sentiment.
        3. Built and evaluated both a TF-IDF Logistic Regression baseline and an exact standard LSTM Memory Network.
        4. Integrated LIME local explainability to provide transparent, word-level attribution for user confidence.
        5. Deployed a comprehensive Streamlit dashboard enabling real-time inference and model inspection.
        """
    )

st.markdown("---")
st.caption("Product Review Helpfulness Prediction Project | NLP & Deep Learning")
