# Academic Project Documentation & Theoretical Guide

**Project Title:** Product Review Helpfulness Prediction with LSTM Memory Networks and Explainable Feature Attribution  
**Domain:** Natural Language Processing (NLP), Deep Learning, and Explainable AI (XAI)  
**Dataset:** Amazon Fine Food Reviews (Stanford SNAP / Kaggle)  
**Target Variable:** Review Helpfulness (Binary: 1 = Helpful, 0 = Not Helpful)  
**Primary Models:** TF-IDF + Logistic Regression (Baseline) vs. Standard LSTM Memory Network (Main Model)  
**Explainability Technique:** Local Interpretable Model-agnostic Explanations (LIME)  

---

> **Note for Handwritten Submissions:**  
> This document contains the complete theoretical, mathematical, and architectural foundations required for handwriting an academic mini-project record or project report. Every section is written in structured, step-by-step prose with explicit mathematical formulas, algorithmic derivations, and illustrative text examples.

---

## 1. Problem Statement & Problem Formulation

### 1.1 Context & Background
In contemporary e-commerce ecosystems (e.g., Amazon, Flipkart), customer purchase decisions rely profoundly on user-generated product reviews. A single popular consumer item can accumulate thousands of customer reviews. Reading every review is cognitively impractical for potential buyers, leading to severe **information overload**.

To assist consumers, platforms introduce community-driven **Helpfulness Voting Mechanisms**, allowing shoppers to vote on whether a review was informative by asking: *"Was this review helpful to you? (Yes / No)"*.

### 1.2 The Problem Statement
While helpfulness voting is effective for mature reviews, it suffers from two acute systemic limitations:
1. **The Cold-Start Problem for Newly Posted Reviews**: When a new review is published, it has zero community votes. Valuable, detailed reviews remain buried at the bottom of the review list until shoppers stumble upon them and vote.
2. **Review Sparsity & Scarcity**: Across large datasets, nearly half of all reviews never receive any votes.

**Core Research Objective:**  
Develop an automated, supervised Natural Language Processing system that analyzes the text semantics (`Summary` and `Text`) of an incoming product review and accurately predicts whether the review will be **Helpful (Class 1)** or **Not Helpful (Class 0)** prior to the accumulation of any community feedback.

---

### 1.3 Mathematical Formulation of the Target Label

The raw Amazon dataset provides two integer voting tallies for each review:
- $\text{HelpfulnessNumerator}$ ($N_h$): Count of users who voted that the review was helpful.
- $\text{HelpfulnessDenominator}$ ($D_h$): Total count of users who voted on the review ($D_h = \text{Helpful Votes} + \text{Unhelpful Votes}$).

#### A. The Helpfulness Ratio
The continuous measure of review utility is defined as:
$$\text{HelpfulnessRatio} = \frac{N_h}{D_h} = \frac{\text{HelpfulnessNumerator}}{\text{HelpfulnessDenominator}}$$
where $0 \le N_h \le D_h$.

#### B. The Zero-Denominator Boundary Condition
For reviews that have never been evaluated by other users:
$$D_h = 0 \implies \text{HelpfulnessRatio} = \frac{N_h}{0} \quad (\text{Undefined / Division by Zero})$$
- In the Amazon Fine Food dataset (568,454 total records), **270,052 reviews (47.51%)** have $D_h = 0$.
- **Academic Rule**: Unvoted reviews cannot be labeled objectively. They represent unlabeled data and must be excluded from the supervised classification dataset.

#### C. Binary Label Assignment
For eligible voted reviews ($D_h \ge 1$ or $D_h \ge 2$), a decision boundary threshold $\tau = 0.5$ is applied:
$$y = \begin{cases} 
1 & \text{if } \text{HelpfulnessRatio} \ge 0.5 \quad (\text{Helpful}) \\
0 & \text{if } \text{HelpfulnessRatio} < 0.5 \quad (\text{Not Helpful})
\end{cases}$$

---

### 1.4 Critical Implementation Safeguards

#### 1. Target Leakage Prevention
If $N_h$, $D_h$, or $\text{HelpfulnessRatio}$ are supplied to the feature matrix $X$, the machine learning model will achieve a trivial 100% accuracy by merely checking if $\frac{N_h}{D_h} \ge 0.5$.  
**Anti-Leakage Mandate**: $N_h$, $D_h$, and $\text{HelpfulnessRatio}$ are strictly isolated for target creation and purged entirely from the feature matrix $X$. The models are trained exclusively on textual tokens.

#### 2. Severe Class Imbalance Handling
Among voted reviews with $D_h \ge 1$, the empirical distribution is:
- **Helpful ($y = 1$)**: $83.21\%$
- **Not Helpful ($y = 0$)**: $16.79\%$  

This represents a severe **5:1 class imbalance**. A naive model predicting all reviews as "Helpful" would achieve $83.2\%$ accuracy while completely failing to detect unhelpful reviews. To solve this, the pipeline applies:
- **Balanced Class Weighting**: Penalizing misclassifications on minority Class 0 more heavily during loss calculation.
- **Stratified Sampling**: Preserving exact class ratios across training and test splits.
- **Comprehensive Evaluation**: Prioritizing Precision, Recall, Macro F1, and ROC-AUC over simple Accuracy.

---

## 2. Theory of Text Preprocessing Techniques

Raw human-written text contains substantial noise: HTML layout markup, URLs, arbitrary punctuation, casing discrepancies, and high-frequency non-informative grammatical particles. Text preprocessing transforms irregular raw strings into structured, informative linguistic tokens.

```
Raw Review Text (HTML + URLs + Punctuation + Mixed Case)
    │
    ▼ [1. Missing Value Imputation]
    │
    ▼ [2. HTML Entity Unescaping & Tag Removal]
    │
    ▼ [3. URL Removal]
    │
    ▼ [4. Case Normalization (Lowercasing)]
    │
    ▼ [5. Special Character & Punctuation Filtering]
    │
    ▼ [6. Whitespace Normalization]
    │
    ▼ [7. Lexical Tokenization]
    │
    ▼ [8. Stopword Filtering with Negation Preservation]
    │
Clean Semantic Token Sequence
```

---

### 2.1 Missing Value Imputation
- **Theoretical Need**: Real-world databases frequently have missing (`null` / `NaN`) entries in text fields. Passing `null` values causes runtime exceptions in string processing routines.
- **Mechanism**: The dataset features both `Summary` (headline) and `Text` (full body). If `Summary` is missing, it is replaced with an empty string `""`. The combined text is constructed as:
  $$\text{CombinedText} = \text{clean}(\text{Summary}) \oplus \text{". "} \oplus \text{clean}(\text{Text})$$
  This ensures headline context and detailed body sentences are integrated without losing records.

---

### 2.2 HTML Entity Unescaping & Tag Stripping
- **Theoretical Need**: Amazon reviews scraped from web interfaces contain raw HTML markup (e.g., `<br />`, `<p>`, `&amp;`, `&quot;`). These tags are artifacts of web formatting and carry no semantic relevance to product quality or helpfulness.
- **Mechanism**:
  1. *Entity Unescaping*: Converts character entities back to literal characters:
     $$\text{"&amp;"} \to \text{"&"}, \quad \text{"&quot;"} \to \text{'"'}, \quad \text{"&#39;"} \to \text{"'"}$$
  2. *Regular Expression Tag Removal*: Replaces all matching patterns of `<.*?>` with a blank space:
     $$\text{Pattern: } \mathcal{R}_{\text{HTML}} = \text{r"<.*?>"}$$
     $$\text{Example: } \text{"Great coffee!<br /><br />Loved it."} \to \text{"Great coffee! Loved it."}$$

---

### 2.3 URL Removal
- **Theoretical Need**: Reviewers occasionally paste external hyperlinks (e.g., promotional blogs, tracking links, manufacturer sites). URLs have high lexical entropy, creating unique out-of-vocabulary tokens that bloat the feature space without providing generalizable linguistic meaning.
- **Mechanism**: Matches HTTP/HTTPS protocols and `www` prefixes using regular expressions:
  $$\text{Pattern: } \mathcal{R}_{\text{URL}} = \text{r"https?://\S+|www\.\S+"}$$
  $$\text{Example: } \text{"Check out www.beans.com for deals"} \to \text{"Check out for deals"}$$

---

### 2.4 Case Normalization (Lowercasing)
- **Theoretical Need**: Standard computational vocabulary models treat strings with identical semantic meaning as distinct tokens if their casing differs (e.g., `"Coffee"`, `"coffee"`, and `"COFFEE"` would become three separate feature dimensions).
- **Mechanism**: Every character $c$ is projected to its lowercase equivalent:
  $$c \mapsto \text{lowercase}(c), \quad \forall c \in \text{String}$$
  This reduces vocabulary dimensionality and aggregates term statistics reliably.

---

### 2.5 Special Character and Punctuation Filtering
- **Theoretical Need**: Non-alphabetic symbols (e.g., `$`, `%`, `*`, `@`, `#`, `!`, `?`) introduce sparseness. While exclamation points denote emphasis, in beginner-level bag-of-words and basic sequence models, punctuation creates irregular token splits (e.g., `"product!"` vs `"product"`).
- **Mechanism**: Replaces all non-alphabetic characters with whitespace:
  $$\text{Pattern: } \mathcal{R}_{\text{punct}} = \text{r"[^a-zA-Z\s]"}$$
  $$\text{Example: } \text{"Price was \$25.99 -- amazing!"} \to \text{"Price was amazing"}$$

---

### 2.6 Whitespace Normalization
- **Theoretical Need**: Punctuation and tag stripping leave behind consecutive whitespace characters, tabs (`\t`), and carriage returns (`\n`).
- **Mechanism**: Collapses sequences of whitespace into a single standard space:
  $$\text{Pattern: } \mathcal{R}_{\text{space}} = \text{r"\s+"} \to \text{" "}$$
  Followed by string stripping (`strip()`) to remove leading and trailing spaces.

---

### 2.7 Lexical Tokenization
- **Theoretical Need**: Machine learning algorithms cannot operate directly on continuous character strings; they require discrete units of meaning known as **tokens**.
- **Mechanism**: Word tokenization segments a normalized sentence into an ordered sequence of words:
  $$S = "w_1 \ w_2 \ w_3 \ \dots \ w_T" \implies \mathcal{T} = [w_1, w_2, \dots, w_T]$$
  In our beginner-friendly implementation, whitespace splitting ($\text{split()}$) over normalized strings produces clean lexical tokens.

---

### 2.8 Stopword Removal with Negation Word Preservation (Crucial NLP Technique)

#### The Standard Stopword Principle
In any human language, certain grammatical words occur with extreme frequency (e.g., `"the"`, `"is"`, `"at"`, `"which"`, `"on"`). According to **Zipf's Law**, these functional words dominate word counts but carry minimal discriminatory information regarding domain content. Removing them reduces noise and compresses the vocabulary.

#### The Problem with Naive Stopword Removal in Sentiment and Quality Analysis
Standard stopword corpora (such as NLTK's English stopword list) include negation words:
$$\text{Negations in Default Lists: } \{\text{"not"}, \text{"no"}, \text{"nor"}, \text{"neither"}, \text{"never"}, \text{"cannot"}\}$$

If naive stopword removal is applied:
$$\text{Raw:} \quad \text{"This coffee is not good, I will never buy it again."}$$
$$\text{Naive Filtered:} \quad \text{"coffee"} \ \mathbf{\xc3\x97} \ \text{"good"} \ \mathbf{\xc3\x97} \ \text{"buy"}$$
The semantic polarity is completely inverted from **negative** to **positive** (`"coffee good buy"`), severely corrupting both the baseline classifier and the LSTM memory network!

#### The Implemented Solution
We construct a customized stopword dictionary $\mathcal{S}_{\text{custom}}$ that explicitly preserves all negation words:
$$\mathcal{N} = \{\text{"not"}, \text{"no"}, \text{"never"}, \text{"neither"}, \text{"nor"}, \text{"none"}, \text{"cannot"}, \text{"without"}, \text{"hardly"}, \text{"barely"}, \text{"scarcely"}\}$$
$$\mathcal{S}_{\text{custom}} = \mathcal{S}_{\text{NLTK}} \setminus \mathcal{N}$$

$$\text{Preserved Filtered Result:} \quad \text{"coffee \textbf{not} good \textbf{never} buy"}$$
This guarantees that linguistic qualifiers and contextual polarity remain intact.

---

## 3. Feature Engineering: TF-IDF Vectorization Theory

For traditional machine learning algorithms (such as Logistic Regression), variable-length text sequences must be transformed into fixed-length numeric vectors.

```
Cleaned Text Document d
    │
    ▼
Term Frequency (TF) Calculation: How often does word t appear in document d?
    │
    ▼
Inverse Document Frequency (IDF) Calculation: How rare/informative is word t across corpus D?
    │
    ▼
TF-IDF Weight = TF(t, d) * IDF(t, D)
    │
    ▼
L2 Normalization: Project vector to unit Euclidean sphere
```

---

### 3.1 Term Frequency ($TF$)
Term Frequency measures the local importance of a word $t$ within a specific document $d$:
$$TF(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
where $f_{t, d}$ is the raw count of token $t$ in document $d$.

To prevent extremely long reviews from dominating due to high raw counts, **Sublinear TF Scaling** is applied:
$$TF_{\text{sublinear}}(t, d) = \begin{cases} 
1 + \log(f_{t, d}) & \text{if } f_{t, d} > 0 \\
0 & \text{otherwise}
\end{cases}$$

---

### 3.2 Inverse Document Frequency ($IDF$)
If a word appears in almost every review across the entire Amazon catalog (e.g., `"food"`, `"product"`, `"taste"`), its ability to discriminate between helpful and unhelpful reviews is minimal. Inverse Document Frequency penalizes common words and rewards rare, specialized words:
$$IDF(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
where:
- $|D| = N$ is the total number of reviews in the training split.
- $|\{d \in D : t \in d\}|$ is the document frequency ($DF$), i.e., the count of reviews containing term $t$.
- The $+1$ terms prevent division by zero and ensure non-negative weights (smooth IDF).

---

### 3.3 The Combined TF-IDF Weight
The final numeric weight assigned to term $t$ in document $d$ is the product:
$$\text{TF-IDF}(t, d) = TF_{\text{sublinear}}(t, d) \times IDF(t, D)$$

---

### 3.4 L2 Euclidean Normalization
To ensure that review length does not bias classification scores, each document vector $\mathbf{v}_d$ is normalized to unit length:
$$\mathbf{x}_d = \frac{\mathbf{v}_d}{\|\mathbf{v}_d\|_2} = \frac{\mathbf{v}_d}{\sqrt{\sum_{k=1}^M v_{d, k}^2}}$$
After L2 normalization, the Euclidean length of every review vector is exactly $\|\mathbf{x}_d\|_2 = 1$.

---

### 3.5 N-Gram Range: Unigrams & Bigrams
- **Unigram ($n=1$)**: Single isolated words (e.g., `"coffee"`, `"delicious"`, `"stale"`).
- **Bigram ($n=2$)**: Contiguous pairs of two words (e.g., `"dark roast"`, `"not good"`, `"waste money"`).
By setting `ngram_range=(1, 2)`, the model captures local phrase associations that single words cannot convey on their own.

---

### 3.6 Vocabulary Truncation & Anti-Leakage Guard
- `max_features = 5000`: Restricts the feature matrix to the top 5,000 most informative terms across the corpus, discarding rare spelling typos.
- **Strict Anti-Leakage Protocol**:
  $$\text{Fitted on } X_{\text{train}} \text{ only}: \quad \text{vectorizer.fit}(X_{\text{train}})$$
  $$\text{Transformed without fitting}: \quad X_{\text{test\_tfidf}} = \text{vectorizer.transform}(X_{\text{test}})$$
  No token frequencies or IDF weights from the test split ever influence the training matrix.

---

## 4. Machine Learning Baseline: Logistic Regression

Logistic Regression serves as the industry-standard linear baseline for text classification.

```
Input TF-IDF Vector x (5,000 dims)
    │
    ▼ [Linear Combination: z = w^T x + b]
    │
    ▼ [Sigmoid Activation: sigma(z) = 1 / (1 + e^-z)]
    │
Predicted Probability P(Helpful = 1 | x)
    │
    ▼ [Decision Threshold tau = 0.5]
Binary Prediction: 1 (Helpful) or 0 (Not Helpful)
```

---

### 4.1 Linear Formulation
Given a review TF-IDF vector $\mathbf{x} = [x_1, x_2, \dots, x_M]^T \in \mathbb{R}^M$ ($M = 5,000$), the linear decision boundary calculates the dot product with learned weight vector $\mathbf{w} \in \mathbb{R}^M$ and bias scalar $b \in \mathbb{R}$:
$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^M w_j x_j + b$$

---

### 4.2 The Sigmoid (Logistic) Function
To convert the unbounded real number $z \in (-\infty, +\infty)$ into a valid probability $P \in (0, 1)$, the **Sigmoid Function** $\sigma(z)$ is applied:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

Properties of the Sigmoid Function:
- $\lim_{z \to +\infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma(0) = 0.5$
- Symmetry: $\sigma(-z) = 1 - \sigma(z)$
- Derivative: $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$

The predicted probabilities for the two classes are:
$$P(Y = 1 \mid \mathbf{x}) = \hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)$$
$$P(Y = 0 \mid \mathbf{x}) = 1 - \hat{y} = 1 - \sigma(\mathbf{w}^T \mathbf{x} + b)$$

---

### 4.3 Objective Function: Weighted Binary Cross-Entropy Loss
The model parameters $\mathbf{w}$ and $b$ are estimated by minimizing the **Binary Cross-Entropy (Log Loss)** function over $N$ training instances. To handle the 83:17 class imbalance, class weights $c_0$ and $c_1$ are incorporated directly into the loss:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \Big[ c_1 \cdot y_i \log(\hat{y}_i) + c_0 \cdot (1 - y_i) \log(1 - \hat{y}_i) \Big] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

where:
- $y_i \in \{0, 1\}$ is the true ground-truth label.
- $\hat{y}_i \in (0, 1)$ is the model's predicted probability of being Helpful.
- $c_1$ and $c_0$ are class weights computed as $c_k = \frac{N}{2 \cdot N_k}$.
- $\frac{1}{2C} \|\mathbf{w}\|_2^2$ is the **L2 Ridge Regularization Penalty**, which prevents weights from growing excessively large and overfitting.
- The objective is convex and optimized globally using the **L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)** quasi-Newton optimization algorithm.

---

## 5. Deep Learning Algorithm: LSTM Memory Networks

While TF-IDF treats a review as an unordered bag of words, human language is inherently sequential. The order of words, syntactic modifiers, and distance between phrases convey critical information. The **Long Short-Term Memory (LSTM)** network is a specialized recurrent neural network designed to capture long-range contextual dependencies across sequences.

```
Raw Review Text
    │
    ▼ [Tokenizer: Map words to integer vocabulary indices: 1 to 10,000]
Sequence of Integers: [45, 12, 804, 3, ...]
    │
    ▼ [Sequence Padding: Pre-padding to fixed length T = 100]
Padded Sequence Tensor: [0, 0, ..., 45, 12, 804, 3]  Shape: (Batch, 100)
    │
    ▼ [Embedding Layer: Dense continuous projection, dim = 64]
Embedding Tensor: (Batch, 100, 64)
    │
    ▼ [Standard LSTM Layer: 64 hidden units with gating cells]
LSTM Output (Final Hidden State h_T): (Batch, 64)
    │
    ▼ [Dropout Layer: Rate = 0.3 to prevent co-adaptation]
    │
    ▼ [Dense Projection Layer: 32 units, ReLU activation]
Dense Tensor: (Batch, 32)
    │
    ▼ [Output Layer: 1 unit, Sigmoid activation]
Scalar Probability: P(Helpful = 1 | x) in [0, 1]
```

---

### 5.1 Tokenization & Integer Sequence Mapping
The Keras Tokenizer builds a dictionary $\mathcal{V}$ mapping the top $V = 10,000$ most frequent training words to distinct integer identifiers ($1, 2, \dots, V$). Any token outside this vocabulary is assigned the Out-of-Vocabulary token index: $\text{oov\_token} = \text{"<OOV>"}$.
$$\text{"delicious coffee"} \implies [142, \ 18]$$

---

### 5.2 Sequence Padding Theory: The Critical Pre-Padding Principle
Neural network batch processing requires all input sequences in a mini-batch to have identical temporal dimensions ($T = 100$). Reviews shorter than $T$ must be padded with integer $0$.

#### Post-Padding vs. Pre-Padding in LSTMs
- **Post-Padding (`padding='post'`)**: Real words appear first; padding zeros are appended at the end:
  $$\mathbf{x} = [w_1, w_2, w_3, \dots, w_k, \mathbf{0, 0, 0, \dots, 0}]$$
  *The Failure Mode*: An LSTM processes tokens sequentially from left to right. When `return_sequences=False`, the network returns its final hidden state $h_T$ at step $T = 100$. If the review ends at word $k=20$, the model executes **80 consecutive time steps of pure zero inputs**! By the time it reaches step 100, the recurrent state has been diluted by zeros, causing severe performance degradation.
- **Pre-Padding (`padding='pre'`) [Implemented Solution]**: Padding zeros are prepended to the front; real words appear at the end:
  $$\mathbf{x} = [\mathbf{0, 0, 0, \dots, 0}, w_1, w_2, w_3, \dots, w_k]$$
  The LSTM encounters neutral zero states first, and the final timesteps process the *actual ending words* of the review. The final hidden state $h_T$ captures the complete contextual summary of the text.

---

### 5.3 The Word Embedding Layer
A discrete one-hot vector representation of size $10,000$ is sparse, high-dimensional, and orthogonal (the dot product between any two different words is 0, failing to capture semantic similarity).

The **Embedding Layer** projects each discrete word index $w \in \{1, \dots, V\}$ into a continuous, dense vector space of dimension $d = 64$:
$$\mathbf{e}_t = \mathbf{E}[w_t] = \mathbf{W}_e^T \mathbf{v}_{w_t} \in \mathbb{R}^{64}$$
where $\mathbf{W}_e \in \mathbb{R}^{V \times 64}$ is the learnable embedding weight matrix. During training, words appearing in similar contexts develop similar geometric orientations in the 64-dimensional latent space.

---

### 5.4 The Mathematical Mechanics of the LSTM Cell

#### Why Vanilla RNNs Fail (The Vanishing Gradient Problem)
In a standard Simple RNN, the hidden state update is:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
During Backpropagation Through Time (BPTT), computing gradients over long sequences involves repeated multiplication of the Jacobian matrix:
$$\frac{\partial h_T}{\partial h_1} = \prod_{j=2}^T \frac{\partial h_j}{\partial h_{j-1}}$$
Because the derivative of $\tanh$ is bounded in $(0, 1]$, repeated matrix multiplication causes the gradient to decay exponentially toward zero as $T$ increases. Consequently, early words in the review are forgotten.

#### The LSTM Solution: The Cell State Highway
The LSTM overcomes vanishing gradients by maintaining an internal **Cell State** $\mathbf{C}_t$, which functions as an informational conveyor belt with linear additive interactions. Information is regulated via three non-linear gating mechanisms:

```
                      Cell State C_{t-1} ──────[ * ]──────────────────(+)───────> Cell State C_t
                                                ^                      ^
                                                │ f_t                  │ i_t * \tilde{C}_t
                                            [Forget Gate]         [Input Gate]
                                                │                      │
Previous Hidden State h_{t-1} ──┬───────────────┴──────────────────────┼──────────[Output Gate]──[ * ]──> Hidden State h_t
                                │                                      │                 │         ^
Current Input x_t ──────────────┴──────────────────────────────────────┴─────────────────┴─────────┘ \tanh(C_t)
```

#### Gate 1: The Forget Gate ($\mathbf{f}_t$)
Decides what fraction of past memory to retain or discard:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
- If $f_{t, j} \approx 1$: Fully retain memory component $j$.
- If $f_{t, j} \approx 0$: Completely wipe out memory component $j$.

#### Gate 2: The Input Gate ($\mathbf{i}_t$) & Candidate Memory ($\tilde{\mathbf{C}}_t$)
Decides what new information from current token $\mathbf{x}_t$ should be stored:
1. *Input Gate Activation*:
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
2. *Candidate Cell State*:
   $$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
   The candidate values are bounded in $[-1, 1]$ via the hyperbolic tangent function.

#### Updating the Cell State ($\mathbf{C}_t$)
The new cell state is computed by combining forgotten past memory with scaled new candidate memory:
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
where $\odot$ denotes the Hadamard (element-wise) product. Because updates are additive, error gradients can flow backwards across hundreds of time steps without exponentially decaying!

#### Gate 3: The Output Gate ($\mathbf{o}_t$) & Hidden State ($\mathbf{h}_t$)
Determines what information from the cell state should be emitted as the output hidden state:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$
The final hidden state vector $\mathbf{h}_T \in \mathbb{R}^{64}$ summarizes the entire semantic progression of the review.

---

### 5.5 Regularization: Dropout Layer ($p = 0.3$)
To prevent complex co-adaptation of features where specific recurrent neurons memorize individual training reviews, a **Dropout Layer** randomly sets $30\%$ of the hidden state elements to zero during training:
$$\mathbf{r} \sim \text{Bernoulli}(1 - p), \quad \mathbf{h}_{\text{drop}} = \frac{1}{1 - p} (\mathbf{h}_T \odot \mathbf{r})$$
During inference, dropout is disabled, and the full expected network weights are utilized.

---

### 5.6 Dense Projection Layer (ReLU Activation)
The regularized hidden vector is projected into a 32-dimensional non-linear representation:
$$\mathbf{a} = \text{ReLU}(\mathbf{W}_d \mathbf{h}_{\text{drop}} + \mathbf{b}_d) = \max(0, \ \mathbf{W}_d \mathbf{h}_{\text{drop}} + \mathbf{b}_d)$$
The **Rectified Linear Unit (ReLU)** introduces non-saturating non-linearity, allowing the model to learn high-level feature combinations while maintaining fast gradient backpropagation.

---

### 5.7 Output Classification Layer
The final classification layer projects the 32-dimensional vector to a single scalar logit, transformed by the sigmoid function into a probability:
$$\hat{y} = \sigma(\mathbf{w}_{\text{out}}^T \mathbf{a} + b_{\text{out}}) = \frac{1}{1 + e^{-(\mathbf{w}_{\text{out}}^T \mathbf{a} + b_{\text{out}})}}$$

---

### 5.8 Optimization: The Adam Algorithm
The network weights $\Theta$ are updated using **Adam (Adaptive Moment Estimation)**, which maintains individual adaptive learning rates for each parameter by computing exponentially decaying moving averages of past gradients ($m_t$) and squared gradients ($v_t$):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{First Moment / Momentum})$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{Second Moment / Uncentered Variance})$$

Bias-corrected estimates:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:
$$\Theta_{t+1} = \Theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
Hyperparameters: Learning rate $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$.

---

## 6. Explainable AI (XAI) Theory: LIME

Deep learning NLP models are often criticized as opaque **"black boxes"**. While the model outputs a probability $\hat{y} = 0.91$, users cannot verify *why* the model made that determination. **LIME (Local Interpretable Model-agnostic Explanations)** bridges this gap by providing local word-level feature attribution.

```
Original Input Review x: "Outstanding organic coffee! Rich aroma, smooth dark roast flavor..."
    │
    ▼ [Generate Perturbed Samples: Randomly drop/mask subsets of words]
Sample 1: "Outstanding coffee! aroma, dark roast flavor..."
Sample 2: "organic coffee! Rich, smooth flavor..."
Sample 3: "Outstanding! Rich aroma, smooth..."
    │
    ▼ [Query Black-Box Model: Compute prediction probabilities for all perturbations]
P(Helpful): [0.84, 0.72, 0.65, ...]
    │
    ▼ [Exponential Kernel Weighting: Weight samples by proximity to original review]
Proximity Weight pi_x(z) = exp(-dist(x, z)^2 / sigma^2)
    │
    ▼ [Fit Local Interpretable Linear Surrogate Model]
Minimize: Loss(f, g, pi_x) + Complexity(g)
    │
    ▼ [Extract Feature Weights (Word Contributions)]
Positive Contributors (+): Words driving prediction toward Helpful (Green)
Negative Contributors (-): Words driving prediction toward Not Helpful (Red)
```

---

### 6.1 The Core Philosophy of LIME
LIME operates on a foundational insight: **Even if a global decision boundary across the entire dataset is highly non-linear and complex, the local decision surface in the immediate neighborhood of a single specific review can be faithfully approximated by a simple linear model.**

---

### 6.2 Mathematical Formulation of LIME

Let:
- $f: \mathcal{X} \to [0, 1]$ be the complex black-box model (the LSTM or Logistic Regression).
- $x$ be the original review text being explained.
- $g \in G$ be an interpretable surrogate model (e.g., a sparse linear regressor):
  $$g(z') = w_0 + \sum_{j=1}^d w_j z'_j$$
  where $z' \in \{0, 1\}^d$ is a binary vector indicating the presence ($1$) or absence ($0$) of word $j$ in the perturbed sentence.

#### The LIME Optimization Objective:
$$\xi(x) = \arg\min_{g \in G} \ \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

Where:
1. **Fidelity Loss ($\mathcal{L}$)**: Measures how well the linear surrogate $g$ approximates the black-box model $f$ within the local neighborhood:
   $$\mathcal{L}(f, g, \pi_x) = \sum_{z, z' \in \mathcal{Z}} \pi_x(z) \Big( f(z) - g(z') \Big)^2$$
2. **Proximity Measure ($\pi_x(z)$)**: An exponential smoothing kernel that weights each perturbed sample $z$ by its cosine distance from the original review $x$:
   $$\pi_x(z) = \exp\left( -\frac{D_{\text{cosine}}(x, z)^2}{\sigma^2} \right)$$
   Perturbations that preserve most original words receive heavy weights; perturbations that alter the sentence drastically receive near-zero weights.
3. **Model Complexity Penalty ($\Omega(g)$)**: Enforces sparsity by restricting the explanation to the top $K$ most influential words (e.g., $K = 6$ or $8$).

---

### 6.3 Interpreting LIME Feature Attribution Weights
After solving the weighted ridge regression problem, each word $j$ receives a learned coefficient $w_j$:
- **Positive Weight ($w_j > 0$)**: The presence of word $j$ increases the probability that the review is **Helpful** (pushes toward Class 1).
- **Negative Weight ($w_j < 0$)**: The presence of word $j$ decreases the probability, pushing the prediction toward **Not Helpful** (pushes toward Class 0).

#### Practical Review Example from Our Pipeline:
- **Review**: *"Outstanding organic coffee! Rich aroma, smooth dark roast flavor, and very fresh beans. Highly recommended."*
- **Model Output**: Predicted Class = **Helpful (1)**, Confidence = **91.0%**.
- **LIME Word Attribution**:
  - `Highly`: $+0.1489$ (Strongest positive driver)
  - `fresh`: $+0.0902$
  - `recommended`: $+0.0755$
  - `flavor`: $+0.0718$
  - `beans`: $+0.0691$
  - `Outstanding`: $+0.0595$
- **Theoretical Takeaway**: The model has legitimately learned that concrete, sensory descriptors and explicit quality recommendations correlate with community helpfulness, rather than relying on spurious statistical artifacts.

---

## 7. Model Evaluation Metrics & Theoretical Definitions

All evaluation metrics are computed strictly on the held-out test split ($N_{\text{test}} = 3,000$ reviews) to reflect real-world generalization performance.

```
                           Actual Class (Ground Truth)
                           Class 1 (Helpful)       Class 0 (Not Helpful)
Predicted    Class 1       True Positive (TP)      False Positive (FP)
Class        Class 0       False Negative (FN)     True Negative (TN)
```

---

### 7.1 Confusion Matrix Elements
- **True Positives ($TP$)**: Actual helpful reviews correctly predicted as Helpful.
- **True Negatives ($TN$)**: Actual unhelpful reviews correctly predicted as Not Helpful.
- **False Positives ($FP$) [Type I Error]**: Actual unhelpful reviews incorrectly predicted as Helpful.
- **False Negatives ($FN$) [Type II Error]**: Actual helpful reviews incorrectly predicted as Not Helpful.

---

### 7.2 Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- *Limitation*: Under class imbalance ($83\%$ class 1), a model predicting only class 1 scores $83\%$ accuracy despite having zero ability to detect unhelpful reviews ($TN = 0$). Accuracy must always be complemented by precision, recall, and ROC-AUC.

---

### 7.3 Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$
- *Meaning*: Of all reviews predicted by the model to be Helpful, what fraction was genuinely helpful? High precision guarantees that buyers are not recommended low-quality reviews.

---

### 7.4 Recall (Sensitivity)
$$\text{Recall} = \frac{TP}{TP + FN}$$
- *Meaning*: Of all genuinely helpful reviews in the dataset, what fraction did the model successfully retrieve? High recall guarantees that valuable customer insights are not overlooked.

---

### 7.5 F1-Score
The harmonic mean of Precision and Recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
Unlike the arithmetic mean, the harmonic mean heavily penalizes models that sacrifice one metric to inflate the other.

---

### 7.6 Macro-Averaged F1-Score
To ensure transparent evaluation under class imbalance, Macro F1 computes the unweighted average of F1 scores across both classes:
$$F_{1, \text{class } 0} = \frac{2 \cdot TN}{2 \cdot TN + FP + FN}$$
$$F_{1, \text{class } 1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
$$\text{Macro } F_1 = \frac{F_{1, \text{class } 0} + F_{1, \text{class } 1}}{2}$$
This forces the model to perform well on the minority "Not Helpful" class rather than solely optimizing for the majority class.

---

### 7.7 Receiver Operating Characteristic (ROC) & ROC-AUC
- **True Positive Rate ($TPR$)**: $TPR = \frac{TP}{TP + FN}$ (Identical to Recall).
- **False Positive Rate ($FPR$)**: $FPR = \frac{FP}{FP + TN} = 1 - \text{Specificity}$.
- **The ROC Curve**: Plots $TPR$ (y-axis) against $FPR$ (x-axis) across all possible continuous decision thresholds $\tau \in [0, 1]$.
- **ROC-AUC (Area Under Curve)**:
  $$\text{AUC} = \int_0^1 TPR(FPR) \, d(FPR)$$
  - $\text{AUC} = 0.50$: Equivalent to random coin tossing.
  - $\text{AUC} = 1.00$: Perfect rank-ordering classification.
  - *Interpretation*: The probability that the classifier ranks a randomly chosen helpful review higher than a randomly chosen unhelpful review.

---

## 8. Summary Comparison Table of Implemented Models

| Theoretical Dimension | Baseline Model | Main Deep Learning Model |
| :--- | :--- | :--- |
| **Model Name** | TF-IDF + Logistic Regression | Standard LSTM Memory Network |
| **Input Representation** | Sparse TF-IDF Vector ($5,000$ dimensions) | Dense Integer Sequences ($T = 100$, Pre-padded) |
| **Word Order Modeling** | Local only via Bigrams ($n=1, 2$) | Full temporal sequential dependency modeling |
| **Embedding Type** | Discrete statistical weights (TF $\times$ IDF) | Continuous 64-dimensional learned embeddings |
| **Computational Complexity** | $\mathcal{O}(N \cdot M)$ — Extremely fast linear algebra | $\mathcal{O}(N \cdot T \cdot d^2)$ — Recurrent tensor transformations |
| **Test Accuracy** | **$77.63\%$** | **$84.40\%$** |
| **Test Precision** | **$92.28\%$** | **$86.60\%$** |
| **Test Recall** | **$80.47\%$** | **$96.63\%$** |
| **Test F1-Score** | **$85.97\%$** | **$91.34\%$** |
| **Test ROC-AUC** | **$0.7807$** | **$0.7590$** |
| **Interpretability** | Global linear weights + Local LIME | Local LIME feature attribution |

---

## 9. Key Academic Findings, Challenges & Real-World Applications

### 9.1 Key Academic Findings
1. **Vote Scarcity & Sparsity**: $47.51\%$ of all e-commerce reviews receive exactly 0 community votes, confirming that user feedback is sparse and an automated text-based prediction system is essential.
2. **Linguistic Determinants of Utility**: Helpful reviews are characterized by information density—specific sensory terms, brewing instructions, comparative evaluations, and packaging details. Unhelpful reviews are overwhelmingly brief, emotional rants (e.g., *"hate it"*, *"damaged in shipping"*).
3. **Pre-Padding is Critical for LSTMs**: In fixed-length sequence models, pre-padding ensures the final hidden state represents the end of the text rather than zero padding, preventing gradient vanishing.

### 9.2 Real-World Engineering Applications
- **Review Cold-Start Ranking**: Sort newly posted reviews by predicted helpfulness immediately, ensuring shoppers see the most informative reviews before votes accumulate.
- **Review Writing Assistant**: Provide live authoring suggestions (e.g., *"Adding details about product durability or flavor will increase the helpfulness of your review"*).
- **Spam and Low-Effort Review Detection**: Automatically flag low-information one-liner reviews for moderation.

---

## 10. Guide for Handwriting this Document

When copying this document into an academic record, lab journal, or project report:
1. **Section 1**: Copy the Problem Statement, the Helpfulness Ratio formula, the Zero-Denominator condition, and the anti-leakage rule.
2. **Section 2**: Draw the Preprocessing Flowchart (shown at the beginning of Section 2), followed by the 8 preprocessing steps with equations and the negation preservation example.
3. **Section 3**: Write the TF, IDF, and TF-IDF formulas with definitions of $N$ and $DF$.
4. **Section 4**: Write the linear equation, the Sigmoid function with its graph sketch, and the Weighted Binary Cross-Entropy formula.
5. **Section 5**: Draw the LSTM architecture pipeline and the LSTM Cell diagram. Write down the equations for Forget Gate ($f_t$), Input Gate ($i_t$), Candidate ($\tilde{C}_t$), Cell State ($C_t$), and Output Gate ($o_t$). Explain why pre-padding is superior to post-padding.
6. **Section 6**: Draw the LIME workflow diagram and state the optimization objective $\xi(x) = \arg\min \mathcal{L}(f, g, \pi_x) + \Omega(g)$ and the exponential kernel formula.
7. **Section 7 & 8**: Copy the Confusion Matrix table, the metric formulas (Accuracy, Precision, Recall, F1, ROC-AUC), and the final model comparison table with real test scores.
