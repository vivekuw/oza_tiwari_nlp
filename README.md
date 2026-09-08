# Product Review Helpfulness Prediction with LSTM Memory Networks and Explainable Feature Attribution

An end-to-end, academic-grade Natural Language Processing (NLP) project built with Python to predict whether an Amazon fine food review is **Helpful (1)** or **Not Helpful (0)** using text semantics, with local interpretability via **LIME (Local Interpretable Model-agnostic Explanations)** and an interactive **Streamlit Dashboard**.

---

## 1. Dataset Source and Overview

- **Source**: Stanford Network Analysis Project (SNAP) & Kaggle
- **Dataset**: Amazon Fine Food Reviews (~568,454 reviews)
- **URL**: [https://snap.stanford.edu/data/web-FineFoods.html](https://snap.stanford.edu/data/web-FineFoods.html) | [Kaggle Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Columns**: `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`.

---

## 2. Problem Formulation & Anti-Leakage Safeguard

### Target Creation
$$\text{HelpfulnessRatio} = \frac{\text{HelpfulnessNumerator}}{\text{HelpfulnessDenominator}}$$

- **Zero-Denominator Exclusion**: Reviews with $\text{HelpfulnessDenominator} = 0$ (47.51% of all records) have no community feedback and are excluded from supervised classification.
- **Binary Target**:
  - $\text{Helpful} = 1$ if $\text{HelpfulnessRatio} \ge 0.5$
  - $\text{Not Helpful} = 0$ if $\text{HelpfulnessRatio} < 0.5$
- **Class Imbalance**: Among voted reviews, ~83.2% are Helpful vs ~16.8% Not Helpful (a ~5:1 imbalance). The pipeline applies balanced class weights in both Logistic Regression and Keras LSTM to prevent majority-class collapse.
- **Strict Anti-Leakage Guarantee**: `HelpfulnessNumerator`, `HelpfulnessDenominator`, and `HelpfulnessRatio` are strictly excluded from the feature space. Classification relies purely on the text content (`Summary` + `Text`).

---

## 3. NLP Pipeline Architecture

```
Reviews.csv
    │
    ▼
Data Cleaning & Missing Value Imputation
    │
    ▼
Exploratory Data Analysis (EDA) & Zero-Denominator Filtering
    │
    ▼
Text Preprocessing (HTML & URL strip, lowercase, punctuation, stopwords with negation preservation)
    │
    ▼
Stratified Train/Test Split (80% Train, 20% Test)
    │
    ├─────────────────────────────────────────────┐
    ▼                                             ▼
Feature Engineering (TF-IDF)              Keras Tokenizer + Padding (maxlen=150)
    │                                             │
    ▼                                             ▼
Baseline Model: Logistic Regression       Main Model: LSTM Memory Network
(balanced class weights)                  Embedding(64) -> LSTM(64) -> Dropout(0.3)
    │                                     -> Dense(32, ReLU) -> Dense(1, Sigmoid)
    │                                             │
    └──────────────────────┬──────────────────────┘
                           ▼
             Model Evaluation on Test Set
             (Accuracy, Precision, Recall, F1, ROC-AUC,
              Confusion Matrices, ROC Curves, Loss/Acc Curves)
                           │
                           ▼
             LIME Explainability & Attribution
                           │
                           ▼
             Interactive Streamlit Web Dashboard
```

---

## 4. Models Implemented

### Baseline Model
- **Representation**: Scikit-Learn `TfidfVectorizer` (max 5,000 features, unigrams & bigrams, sublinear TF scaling).
- **Classifier**: `LogisticRegression(class_weight='balanced', random_state=42)`.

### Main Model (LSTM Memory Network)
- **Tokenization**: Keras `Tokenizer(num_words=10000, oov_token='<OOV>')` fitted strictly on training data.
- **Sequence Padding**: `pad_sequences(maxlen=150, padding='post', truncating='post')`.
- **Architecture**:
  - `Embedding(input_dim=10000, output_dim=64, input_length=150)`
  - `LSTM(64)` (unidirectional recurrent memory network)
  - `Dropout(0.3)`
  - `Dense(32, activation='relu')`
  - `Dense(1, activation='sigmoid')`
- **Optimization**: Adam ($\text{lr} = 0.001$), Binary Crossentropy, EarlyStopping on validation loss.

---

## 5. Explainable Feature Attribution (LIME)

Using `LimeTextExplainer`:
- Perturbs review text dynamically and tracks output probability shifts.
- Deconstructs individual predictions into:
  - **Positive Contributors** (weights > 0): words driving the prediction towards **Helpful**.
  - **Negative Contributors** (weights < 0): words driving the prediction towards **Not Helpful**.
- Generates interactive HTML highlights for visual review inspection.

---

## 6. Directory Structure

```
oza_tiwari_nlp/
│
├── data/
│   └── Reviews.csv                   # Raw Amazon Fine Food Reviews dataset
│
├── models/                           # Serialized model artifacts
│   ├── baseline_model.joblib         # Fitted Logistic Regression
│   ├── tfidf_vectorizer.joblib       # Fitted TF-IDF Vectorizer
│   ├── lstm_model.keras              # Trained Keras LSTM model
│   └── tokenizer.pkl                 # Fitted Keras Tokenizer
│
├── outputs/
│   ├── evaluation/
│   │   ├── model_comparison.json     # Test metrics JSON
│   │   ├── model_comparison.csv      # Test metrics CSV
│   │   └── eda_threshold_report.json # Denominator distribution report
│   ├── figures/
│   │   ├── baseline_confusion_matrix.png
│   │   ├── lstm_confusion_matrix.png
│   │   ├── roc_curve_comparison.png
│   │   ├── model_comparison_chart.png
│   │   ├── lstm_training_history.png
│   │   ├── helpfulness_ratio_distribution.png
│   │   ├── score_vs_helpfulness.png
│   │   └── class_distribution.png
│   ├── wordclouds/
│   │   ├── all_reviews_wordcloud.png
│   │   ├── helpful_wordcloud.png
│   │   └── not_helpful_wordcloud.png
│   └── explanations/
│       ├── lime_sample_helpful.html
│       └── lime_sample_unhelpful.html
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Loading & validation
│   ├── target_creation.py            # Target ratio & zero-vote handling
│   ├── preprocessing.py              # Text cleaning & negation preservation
│   ├── feature_engineering.py        # TF-IDF vectorization
│   ├── baseline_model.py             # Logistic regression baseline
│   ├── lstm_model.py                 # Standard LSTM network
│   ├── evaluation.py                 # Metrics & diagnostic plots
│   ├── wordclouds.py                 # Word cloud generator
│   └── explainability.py             # LIME wrapper
│
├── run_pipeline.py                   # Incremental pipeline runner script
├── app.py                            # Streamlit web dashboard
├── README.md                         # Documentation
└── .gitignore
```

---

## 7. How to Run

### Step 1: Run the Training & Evaluation Pipeline
```bash
python run_pipeline.py --sample_size 25000 --epochs 5 --batch_size 64
```
This script will:
1. Load `Reviews.csv` and validate schema.
2. Inspect and report the class balance and save EDA figures.
3. Preprocess text and generate word clouds (All, Helpful, Not Helpful).
4. Perform stratified train/test split (80/20).
5. Train and evaluate the Baseline Logistic Regression.
6. Train and evaluate the standard LSTM Memory Network with early stopping.
7. Save all trained models, tokenizers, confusion matrices, ROC curves, and LIME explanations.

### Step 2: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with all 16 project features:
- Interactive EDA charts & class balance sensitivity
- High-resolution Word Clouds
- Model performance tables and grouped comparison bar charts
- Confusion matrices and combined ROC curves
- Real-time review helpfulness predictor with confidence gauges
- Live LIME word contribution graphs and highlighted text spans
- Academic discussion, challenges, applications, and conclusions

---

## 8. Academic Findings & Conclusions

1. **Information Density Predicts Helpfulness**: Reviews that articulate specific product characteristics (flavor notes, bean roast level, ingredients, comparative value) consistently achieve higher helpfulness ratings.
2. **Short Emotional Outbursts are Voted Down**: One-line negative rants (e.g. "Arrived damaged, hate it") are consistently rated unhelpful by the Amazon community.
3. **Class Weighting is Essential**: Due to the severe 83:17 positive bias in community voting, unweighted models collapse into majority-class predictors. Applying balanced class weights ensures both classes are learned reliably.
