# NLP_c1_m1

## Supervised ML and sentiment analysis

### Direct definition

**Supervised machine learning** is a learning paradigm where a model learns a mapping from inputs x to targets y using a labeled dataset {(x(i), y(i))} and a loss function that measures prediction error.

**Sentiment analysis** is a supervised NLP task that assigns a sentiment label (e.g., positive, negative, neutral) to text, using features derived from text and a model trained to predict sentiment labels.

### Concept intuition

- Supervised ML: imagine a teacher showing many examples (input → correct answer). The model is a student that updates its internal rules to match the teacher's answers. Over time the student generalizes to unseen examples.
- Sentiment analysis: the model learns how words, phrases, and patterns map to an opinion. Words like "excellent", "great", "hate" act as signals; the model learns which signals push a text toward positive or negative sentiment.
- Why it matters: sentiment analysis is everywhere — product reviews, social media monitoring, customer support routing. It’s a canonical supervised NLP problem that connects raw text to actionable labels.

Practical framing:

- Input x: raw text (sentence, review).
- Feature extraction: tokenization → vectorization (bag-of-words, TF-IDF, embeddings).
- Model: logistic regression, SVM, or neural network.
- Output y: discrete label (binary or multi-class) or continuous sentiment score.

### Mathematical breakdown

Problem statement:

- Dataset: D = {(x(1), y(1)), ..., (x(m), y(m))} where y ∈ {0,1} for binary sentiment.
- Model f(x; θ) produces probability p = P(y=1 | x; θ).

Logistic regression probability (sigmoid on linear score):

```
z = w^T x + b
p = sigmoid(z) = 1 / (1 + exp(-z))
```

Binary cross-entropy loss for one example:

```
L(y, p) = -[ y * log(p) + (1 - y) * log(1 - p) ]
```

Average loss over dataset:

```
J(θ) = (1/m) * sum_{i=1..m} L(y(i), p(i))
```

Gradient update (SGD step):

```
w := w - alpha * dw
b := b - alpha * db

where dw = (1/m) * X^T (p - y)
      db = (1/m) * sum(p - y)
```

Variable explanations:

- x: input feature vector for a text (e.g., TF-IDF or embedding).
- w: weight vector.
- b: bias scalar.
- z: linear score.
- p: predicted probability for class 1.
- y: true label (0 or 1).
- J(θ): cost function averaged over examples.
- alpha: learning rate.

Why this works:

- The linear score z measures evidence for positive sentiment.
- Sigmoid converts score to probability in (0,1).
- Cross-entropy penalizes confident wrong predictions more heavily and provides smooth gradients for optimization.

### Code and practical application

Minimal NumPy logistic regression on tiny bag-of-words features

```python
import numpy as np

# Tiny dataset
texts = ["I love this product", "This is terrible", "Amazing experience", "I hate it"]
labels = np.array([1, 0, 1, 0])  # 1 positive, 0 negative

# Very simple bag-of-words vectorizer
vocab = {}
def vectorize(text):
    vec = np.zeros(len(vocab))
    for tok in text.lower().split():
        if tok in vocab:
            vec[vocab[tok]] += 1
    return vec

# build vocab
for t in texts:
    for tok in t.lower().split():
        if tok not in vocab:
            vocab[tok] = len(vocab)

X = np.array([vectorize(t) for t in texts])  # shape (4, V)

# initialize weights
w = np.zeros(X.shape[1])
b = 0.0
lr = 0.5
epochs = 200

def sigmoid(z): return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    z = X.dot(w) + b
    p = sigmoid(z)
    m = X.shape[0]
    dw = (1/m) * X.T.dot(p - labels)
    db = (1/m) * np.sum(p - labels)
    w -= lr * dw
    b -= lr * db

# test
for t in texts:
    x = vectorize(t)
    print(t, sigmoid(w.dot(x) + b) > 0.5)
```

TensorFlow 2 binary classifier with TF-IDF preprocessing (sketch)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

texts = ["I love this product", "This is terrible", "Amazing experience", "I hate it"]
labels = np.array([1,0,1,0])

# TF-IDF via Keras TextVectorization producing integer counts then adapt to TF-IDF externally
vectorizer = layers.TextVectorization(output_mode='tf_idf')
vectorizer.adapt(texts)
X = vectorizer(np.array(texts))

model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=50, verbose=0)
print(model.predict(X).round(2))
```

Practical workflow:

- Start with basic bag-of-words or TF-IDF. Rapidly prototype with logistic regression.
- If performance plateaus, move to word embeddings + neural models (CNN, LSTM, Transformer).
- Use validation set and metrics like accuracy, precision, recall, F1, ROC-AUC depending on class balance and business need.

### Visualization and geometric intuition

- Feature space geometry: bag-of-words or TF-IDF maps each text to a high-dimensional vector. Each coordinate is presence/importance of a word. Sentiment classes often lie in different regions of this space.
- Linear separator: logistic regression finds a hyperplane (w) that separates positive and negative examples. w points toward the positive side; b shifts the plane.
- Sigmoid curve: maps linear score to probability smoothly; gradient is largest near p=0.5 so learning focuses on uncertain predictions.
- Loss surface: cross-entropy with a linear model is convex, so global minimum exists. Visualizing J(w) in 2D shows bowl-shaped surface when restricted to two weights.
- Gradients: dw points in direction that increases agreement with labels. For positive examples with p < 1, (p - y) negative reduces loss by increasing dot(w,x).

Visualization tip: plot 2D PCA of TF-IDF vectors, color by label, and overlay decision boundary from logistic regression projected back into 2D.

Quick PCA + boundary sketch (NumPy + matplotlib idea):

- Compute PCA to 2D.
- Train logistic on original features.
- For many points on 2D grid, map back to original space via inverse transform approx or project classifier by training new logistic on 2D PCA coords for visualization.

### Common pitfalls and tips

- Pitfall: Using raw text without cleaning/tokenization can introduce noise. Tip: normalize case, remove punctuation carefully, consider negation handling ("not good").
- Pitfall: High-dimensional sparse features cause overfitting with small datasets. Tip: regularize (L2), reduce vocab size, use TF-IDF, or switch to embeddings.
- Pitfall: Imbalanced classes (many neutral/positive vs few negative) cause misleading accuracy. Tip: use balanced metrics (precision, recall, F1) and resampling or class-weighting.
- Pitfall: Treating TF-IDF vectors as sequences. Tip: models like logistic or SVM operate on fixed vectors; if sequence info matters, use RNNs/Transformers.
- Tip: Start simple (logistic + TF-IDF), add complexity only when needed.
- Tip: Track learning curves (train vs validation loss) to detect underfit/overfit.
- Tip: Beware of label noise; sentiment labels can be subjective. Consider multiple annotators or soft labels.

### Interview ready insights

- Explain why logistic regression is a reasonable baseline: linear, fast, interpretable weights (which words push toward sentiment).
- Be able to derive gradient of cross-entropy with sigmoid: dw = (1/m) X^T (p - y).
- Discuss feature engineering trade-offs: bag-of-words vs TF-IDF vs embeddings; when each is preferred.
- Mention evaluation choices: accuracy vs F1 vs AUC and why class balance matters.
- Explain how to handle negation: simple rule-based flip tokens or include bigrams to capture "not good".
- Be ready to justify regularization choices and hyperparameters: why L2 helps with sparse high-dim text features.
- Discuss domain shift: model trained on tweets may fail on product reviews — need domain adaptation or re-training.

### Practice exercises

1. Basic logistic regression from scratch
    - Task: Implement logistic regression with L2 regularization on the IMDb small sample (or create >200 synthetic labeled sentences) using NumPy. Report accuracy and plot training loss.
    - Hint: L2 adds term (lambda/(2m)) * ||w||^2 to loss; gradient adds (lambda/m) * w to dw.
2. TF-IDF vs Bag-of-Words comparison
    - Task: Using scikit-learn, vectorize a small sentiment dataset with CountVectorizer and TfidfVectorizer, train logistic regression, compare accuracy and top positive/negative features.
    - Hint: Use LogisticRegression with class_weight='balanced' if classes skewed.
3. Visualize decision boundary in 2D
    - Task: Take a small TF-IDF dataset, reduce features to 2D with PCA, train logistic on 2D PCA coords, plot data points and boundary. Then train logistic on full features and project predicted labels to PCA plot to compare.
    - Hint: Use sklearn.decomposition.PCA and matplotlib contourf for decision boundary.
4. Negation handling experiment
    - Task: Create a small set where negation flips sentiment ("good" vs "not good"). Compare model performance with unigram features vs bigrams vs a simple negation tokenization that joins "not" with following word ("not_good").
    - Hint: In tokenization, replace "not X" with "not_X" before vectorization.
5. Move to embeddings
    - Task: Use Keras TextVectorization (output_mode='int'), an embedding layer, and a simple Dense classifier (embedding → GlobalAveragePooling → Dense) on a small dataset. Compare performance vs TF-IDF + logistic.
    - Hint: GlobalAveragePooling1D reduces sequence of embedding vectors to fixed-size by averaging.

Short walkthrough for exercise 1 (hint + steps)

- Build vocab, vectorize texts into X.
- Initialize w, b zeros.
- Compute z = X.dot(w) + b; p = sigmoid(z).
- Compute loss L = -(1/m) * sum(y*log(p) + (1-y)*log(1-p)) + (lambda/(2m)) * sum(w^2).
- Compute dw = (1/m)*X.T.dot(p - y) + (lambda/m)*w; db = (1/m)*sum(p - y).
- Gradient descent updates for epochs; monitor loss.

---

## Vocabulary and feature extraction

### 1. Direct definition

**Vocabulary**: the set of distinct tokens (words, subwords, characters) your model recognizes.

**Feature extraction**: the process of converting raw text into numeric vectors that capture information useful for a learning algorithm.

### 2. Concept intuition: what it is and why it matters

- Text is symbolic; ML expects numbers. Vocabulary defines the dictionary and feature extraction converts text to coordinates in a vector space.
- Choice of vocabulary and features determines which patterns are visible to the model (e.g., sentiment words, negation, idioms). Good choices make models simpler, generalize better, and train faster. Poor choices force the model to learn complicated mappings or overfit noise.
- Real-world trade-offs:
    - Small vocab (top-k words) → faster, less overfitting, but misses rare signals.
    - Large vocab → captures nuance, increases sparsity and memory.
    - Subword tokens (BPE, WordPiece) handle OOV words and morphologies.
    - Character-level handles typos and languages with productive morphology.

Analogy: vocabulary is the pixel grid you choose to represent an image; feature extraction is deciding whether you store raw pixels, edges, or SIFT descriptors.

### 3. Mathematical breakdown

Common vector representations:

One-hot encoding (dimension = Vocab size)

```
given vocab size V and token index i in [0..V-1]:
one_hot[i] = 1
all other indices = 0
```

Bag-of-words (count vector for document)

```
x_j = count of token j in document
x = [x_0, x_1, ..., x_{V-1}]  (shape V)
```

Term Frequency (TF)

```
tf_j = x_j / sum_k x_k
```

TF-IDF

```
idf_j = log( (N + 1) / (df_j + 1) ) + 1
tfidf_j = tf_j * idf_j

where:
N = number of documents
df_j = number of documents containing token j
```

Continuous embeddings (pretrained or learned)

```
Given embedding matrix E (V x d),
token i -> vector e_i = E[i]  (dimension d)
document vector = aggregate(e_{i1}, e_{i2}, ... ) (mean, sum, or weighted)
```

Why TF-IDF works: TF captures local importance in doc; IDF downweights common tokens (stopwords) that carry less discriminative signal.

Gradient/optimization note (for learned embeddings):

- Embedding matrix E parameters are updated via gradient of loss L wrt E: dL/dE[i] accumulates contributions from occurrences of token i in batch.

### 4. Code & practical application

A. Building vocab and simple features (NumPy + Python)

```python
# build vocab and count vectors
texts = ["I love this product", "Not good, I hate it", "Amazing product and experience"]

# simple tokenizer (space + lowercase)
def tokenize(s): return s.lower().replace(",", "").replace(".", "").split()

# build vocab (min_freq threshold)
from collections import Counter
min_freq = 1
cnt = Counter(tok for t in texts for tok in tokenize(t))
vocab = {w:i for i,(w,f) in enumerate(sorted(cnt.items(), key=lambda x:-x[1]) if True else []) if f>=min_freq}

def count_vector(text, vocab):
    vec = [0]*len(vocab)
    for tok in tokenize(text):
        idx = vocab.get(tok)
        if idx is not None:
            vec[idx] += 1
    return np.array(vec, dtype=float)

X_counts = np.array([count_vector(t, vocab) for t in texts])
print("Vocab:", vocab)
print("Counts:\\n", X_counts)
```

B. TF-IDF from scratch (NumPy)

```python
import numpy as np
def tfidf(X_counts):
    N = X_counts.shape[0]
    df = np.sum(X_counts>0, axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1.0      # smoothing variant
    tf = X_counts / (np.sum(X_counts, axis=1, keepdims=True) + 1e-9)
    return tf * idf

X_tfidf = tfidf(X_counts)
```

C. Simple embedding-based classifier with TensorFlow (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Example dataset
texts = ["I love this product", "I hate this", "Not good", "Amazing experience"]
labels = [1, 0, 0, 1]

# TextVectorization to integer tokens
vectorizer = layers.TextVectorization(output_mode='int', max_tokens=1000)
vectorizer.adapt(texts)
vocab_size = len(vectorizer.get_vocabulary())

# Model: embedding -> global average -> dense
inputs = layers.Input(shape=(1,), dtype='string')
x = vectorizer(inputs)                             # shape (batch, seq_len)
x = layers.Embedding(input_dim=vocab_size, output_dim=32)(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(texts), np.array(labels), epochs=20, verbose=0)
```

D. Subword tokenization (high-level description)

- Use BPE/WordPiece: build merges/subword vocabulary from corpus; allows lowercase tokenization of "unhappiness" -> "un", "##happi", "##ness".

### 5. Visualization / Geometry

- One-hot vectors: orthogonal axes in V-dimensional space; each token sits at a unique axis. Distances convey nothing about similarity.
- Bag-of-words/TF-IDF: documents are sparse vectors in V-space; similar documents cluster by cosine similarity if they share important tokens.
- Embeddings: tokens are points in low-d d-space; geometric proximity encodes semantic or syntactic similarity (cosine distance). Averaging embeddings maps variable-length text to a single point; this loses order but retains topic/sentiment signal.
- Visual recipe:
    - Compute TF-IDF vectors for documents.
    - Use PCA or t-SNE to project to 2D.
    - Plot points colored by label; examine clusters and nearest neighbors.
- Decision boundaries: for linear classifiers with TF-IDF, model learns a hyperplane in V-space; projected to 2D PCA, boundary is a line approximately separating classes.

Geometric intuition for IDF: common tokens make all docs similar along those axes; IDF scales down those axes so discriminative axes dominate geometry.

### 6. Common pitfalls & tips

- Pitfall: naïve tokenization. Contracted/negation, punctuation, emojis, casing matter. Tip: choose tokenizer appropriate to data (tweet tokenizer, spaCy, Hugging Face tokenizers).
- Pitfall: out-of-vocabulary (OOV) tokens. Tip: use subword tokenization or assign an <UNK> token and consider character models for heavy OOV settings.
- Pitfall: extremely large vocab → memory blowup. Tip: prune by frequency, use min_df, or hashing trick.
- Pitfall: sparse high-d vectors with small datasets → overfitting. Tip: L2 regularization, dimensionality reduction, or use embeddings with pretrained initialization.
- Pitfall: losing negation. Tip: add rules to join "not_X" or include bigrams/trigrams to capture short-range order.
- Tip: prefer normalized TF or TF-IDF for many classical models, use embeddings + pooling or sequence models when order/phrasing matters.
- Tip: always inspect top features (highest weights) from linear models to sanity-check what the model learned.

### 7. Interview-ready insights

- Explain differences: one-hot (unique identity), BOW (counts), TF-IDF (importance), embeddings (dense semantic vectors). Give when to use each.
- Be ready to justify subword tokenization: reduces OOV, handles morphology, lowers vocab size while preserving expressiveness.
- Discuss hashing trick: maps tokens to fixed-size vector via hash function to bound memory at cost of collisions.
- Explain why TF-IDF + linear models often beat small neural networks on small datasets: TF-IDF is strong feature engineering that reduces the need to learn token importance from scratch.
- Show how to inspect model weights: in logistic regression, top positive/negative features reveal which tokens drive predictions.
- Give complexity tradeoffs: embedding lookup cost O(1) per token, TF-IDF cost depends on sparse operations; memory for embedding matrix is V*d.

### 8. Practice exercises

Build vocab, TF, TF-IDF, and compare cosine similarity

- Task: Given a small set of 8-12 sentences, implement tokenization, vocab with min_freq=1, compute TF and TF-IDF, then compute cosine similarity between all pairs. Find the most similar pair per TF and per TF-IDF and explain differences.
- Hints: normalize tokens, handle zero vectors, cosine sim = (a·b)/(||a||*||b||).

Negation handling experiment

- Task: Create a dataset with sentences that include negation. Implement three pipelines:
A) unigram tokenizer
B) unigram + bigrams
C) negation-aware tokenization that converts "not X" to "not_X" for the next 1 token
Train a small logistic regression on TF-IDF features and compare accuracy on a held-out set.
- Hint: scikit-learn CountVectorizer/TfidfVectorizer supports ngram_range. For negation, simple rule-based replacement before vectorization suffices.

Subword vs word vocab

- Task: Using a small custom corpus with rare and compound words (e.g., "datascientist", "data-scientist", "data scientist"), toy-implement a simple BPE loop: find most frequent adjacent pair of characters/subtokens and merge, repeat K times. Compare resulting tokens and show how words get segmented.
- Hint: operate on token sequences like ["d a t a s c i e n t i s t"] and count adjacent pairs.

Embedding aggregation effects

- Task: Use Keras TextVectorization + Embedding. For a set of sentences, compute:
    - mean pooling
    - sum pooling
    - max pooling
    Visualize resulting 2D PCA for each pooling method and comment which pooling best separates sentiment classes for your toy dataset.
- Hint: use GlobalAveragePooling1D, Lambda for sum, and GlobalMaxPooling1D.

Feature debugging

- Task: Train a logistic regression on TF-IDF for a small sentiment set. Extract top 10 positive and top 10 negative weights. For each, inspect example sentences to check if the model learned sensible cues or spurious correlations.
- Hint: sklearn LogisticRegression.coef_ returns weight vector aligned with feature names from vectorizer.get_feature_names_out().

Short walkthrough for Exercise 1 (cosine similarity):

```python
# after building X_tfidf (N x V)
from numpy.linalg import norm
def cosine_sim(a,b):
    if norm(a)==0 or norm(b)==0:
        return 0.0
    return float(a.dot(b) / (norm(a)*norm(b)))

# compute all pair similarities
sims = [[cosine_sim(X_tfidf[i], X_tfidf[j]) for j in range(N)] for i in range(N)]
```

---

## Negative and positive frequencies

### Direct definition

**Negative and positive frequencies** are counts or weighted counts of terms or features in text that indicate negative or positive sentiment respectively; they measure how often sentiment-bearing tokens (or their proxies) occur in texts and are used to create features or scores for sentiment analysis.

### Concept intuition: what it is and why it matters

- Frequency is the simplest signal: words like "great", "excellent" (positive) or "terrible", "hate" (negative) appear more often in texts with matching sentiment. Counting these occurrences gives a fast, interpretable signal for classifiers and rule-based scorers.
- Positive/negative frequencies can be raw counts, normalized counts (term frequency), or weighted counts (TF-IDF, log-odds). They help detect polarity at document, sentence, or token level and are often combined with negation handling, intensifiers, and lexicon-based scores.
- Why it matters: frequency-based features are robust baselines, easy to inspect, cheap to compute, and often effective on small datasets. They also make models explainable — you can point to which words drove a prediction.

### Mathematical breakdown

Raw positive / negative frequency per document

```
pos_freq(d) = sum_{t in V_pos} count(t, d)
neg_freq(d) = sum_{t in V_neg} count(t, d)
```

Normalized frequency (term frequency)

```
pos_tf(d) = pos_freq(d) / total_tokens(d)
neg_tf(d) = neg_freq(d) / total_tokens(d)
```

Simple polarity score (difference or ratio)

```
polarity_diff(d) = pos_tf(d) - neg_tf(d)
polarity_ratio(d) = (pos_tf(d) + eps) / (neg_tf(d) + eps)
polarity_logodds(d) = log( (pos_freq(d)+0.5) / (neg_freq(d)+0.5) )
```

Lexicon-weighted sum (weights from sentiment lexicon)

```
pos_weighted(d) = sum_{t in tokens(d)} w_pos(t)
neg_weighted(d) = sum_{t in tokens(d)} w_neg(t)
where w_pos(t) = sentiment_score(t) if sentiment_score>0 else 0
      w_neg(t) = -sentiment_score(t) if sentiment_score<0 else 0
```

PMI-based association (word -> positive class)

```
PMI(word, pos) = log( P(word & pos) / (P(word) * P(pos)) )
Estimate probabilities from counts across corpus.
```

Log-odds ratio for discriminative weighting between classes

```
log_odds_t = log( (count_t_pos + alpha) / (total_pos + alpha*V) )
            - log( (count_t_neg + alpha) / (total_neg + alpha*V) )
```

Variables:

- V_pos, V_neg: sets of known positive/negative tokens (lexicon) or tokens discovered from data.
- count(t,d): count of token t in document d.
- total_tokens(d): length of document d.
- eps: small constant to avoid divide-by-zero.
- alpha: smoothing constant (e.g., 1) to avoid zero counts.
- total_pos / total_neg: total token counts in positive/negative corpora.
- V: vocabulary size.

Why these formulas:

- Differences and ratios capture balance of evidence; log transforms stabilize and turn ratios into additive features.
- PMI and log-odds highlight words strongly associated with a class beyond raw frequency, reducing bias from very common words.

### Code & practical application

A. Build positive/negative frequency features (NumPy / plain Python)

```python
from collections import Counter
import numpy as np

# toy data
texts = [
    "I love this product, it is amazing and excellent",
    "This is terrible, I hate it and it's awful",
    "Good quality but poor support",
    "Absolutely fantastic experience"
]
labels = [1, 0, 1, 1]  # 1 positive, 0 negative

# simple sentiment lexicon (toy)
pos_lex = {"love", "amazing", "excellent", "good", "fantastic"}
neg_lex = {"terrible", "hate", "awful", "poor", "bad"}

def tokenize(s): return s.lower().replace("'", "").replace(".", "").replace(",", "").split()

def pos_neg_freqs(text, pos_lex, neg_lex):
    toks = tokenize(text)
    c = Counter(toks)
    pos_freq = sum(c[t] for t in pos_lex if t in c)
    neg_freq = sum(c[t] for t in neg_lex if t in c)
    total = len(toks)
    return pos_freq, neg_freq, pos_freq/total, neg_freq/total

features = [pos_neg_freqs(t, pos_lex, neg_lex) for t in texts]
X = np.array(features)  # columns: pos_freq, neg_freq, pos_tf, neg_tf
print(X)
```

B. Polarity score and smoothing

```python
def polarity_logodds(pos_freq, neg_freq, smoothing=0.5):
    return np.log((pos_freq + smoothing) / (neg_freq + smoothing))

for f in features:
    pos_f, neg_f = f[0], f[1]
    print("log-odds:", polarity_logodds(pos_f, neg_f))
```

C. Log-odds ratio from corpus counts (discriminative feature)

```python
# compute counts across labeled corpus
from collections import defaultdict
count_pos = defaultdict(int)
count_neg = defaultdict(int)
total_pos = total_neg = 0

for text, label in zip(texts, labels):
    for tok in set(tokenize(text)):  # document frequency per class
        if label==1:
            count_pos[tok] += 1
            total_pos += 1
        else:
            count_neg[tok] += 1
            total_neg += 1

V = len(set(tok for t in texts for tok in tokenize(t)))
alpha = 1.0

log_odds = {}
for tok in set(tok for t in texts for tok in tokenize(t)):
    p_pos = (count_pos[tok] + alpha) / (total_pos + alpha*V)
    p_neg = (count_neg[tok] + alpha) / (total_neg + alpha*V)
    log_odds[tok] = np.log(p_pos / p_neg)

# sort tokens most indicative of positive or negative
sorted_pos = sorted(log_odds.items(), key=lambda x: -x[1])[:10]
sorted_neg = sorted(log_odds.items(), key=lambda x: x[1])[:10]
print("Top positive-indicating tokens:", sorted_pos)
print("Top negative-indicating tokens:", sorted_neg)
```

D. Use frequencies as features in a classifier (scikit-learn example sketch)

```python
# X_freq: columns [pos_tf, neg_tf, polarity_diff, log_odds_doc]
# Fit logistic regression using these engineered features alongside TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# assemble small feature matrix (illustration)
polarity_diff = [f[2]-f[3] for f in features]
logodds_doc = [polarity_logodds(f[0], f[1]) for f in features]
X_feat = np.vstack([ [f[2], f[3]] for f in features ]).T  # shape (n_samples,2)
X_feat = np.vstack([ [f[2], f[3], pd, lo] for f,pd,lo in zip(features, polarity_diff, logodds_doc)])
# normalize, train
scaler = StandardScaler(); Xs = scaler.fit_transform(X_feat)
clf = LogisticRegression().fit(Xs, labels)
```

Practical workflow notes:

- Start by building lexicon-based frequency features as quick baselines.
- Combine lexicon counts with global discriminative scores (log-odds) and TF-IDF or embeddings for stronger models.
- Use smoothing to avoid extreme ratios for zero counts.
- Consider document length normalization (TF) to compare short vs long texts fairly.

### Visualization / Geometry

- Feature space: using pos_tf and neg_tf as 2D features, plots show documents arranged by their positive vs negative signal. The diagonal (pos_tf ≈ neg_tf) is neutral; high pos_tf and low neg_tf indicates positive sentiment.
- Polarity as a single axis: project documents to polarity_diff or log-odds to rank sentiment intensity. This creates a 1D ordering from strongly negative to strongly positive.
- Heatmap of token log-odds: tokens form a spectrum; you can plot bar charts for top positive/negative tokens to inspect discriminative signals.
- Decision boundary intuition: a linear classifier on features [pos_tf, neg_tf] learns a line that roughly splits the 2D plane; on the 1D polarity axis, it learns a threshold.

Visualization recipe:

- Scatter pos_tf vs neg_tf with point color = true label.
- Plot histograms of polarity_diff per class to check separability.
- Bar chart top-k tokens by log-odds.

### Common pitfalls & tips

- Pitfall: lexicon incompleteness — many domain-specific sentiment words won’t be in your lexicon. Tip: augment with corpus-derived indicators (log-odds, PMI) or use embeddings for semantic generalization.
- Pitfall: ignoring negation and modifiers — "not good" flips polarity but pos_freq still counts "good". Tip: detect negation scope and invert or tag following tokens, or include bigrams/phrases.
- Pitfall: treating raw counts without normalization — long documents naturally have higher counts. Tip: use TF normalization or divide by document length.
- Pitfall: zero counts causing infinite ratios/logs. Tip: use smoothing constants (Laplace or 0.5) before log transforms.
- Pitfall: spurious correlations — tokens may correlate with labels for non-sentiment reasons (e.g., product names). Tip: inspect top log-odds tokens and remove or control for metadata.
- Tip: Combine lexicon frequencies with model-learned features — frequencies give interpretability; learned features give adaptability.

### Interview-ready insights

- Be ready to derive and explain the log-odds formula and why smoothing is required. Demonstrate how log-odds converts multiplicative ratios into additive, easier-to-use features.
- Explain PMI vs log-odds: PMI measures association beyond independence expectation; log-odds compares relative probabilities across classes and is more stable with class-conditional smoothing.
- Explain why normalization matters: raw counts conflate document length with prevalence; TF or TF-IDF mitigates this.
- Describe practical pipeline: build lexicon counts → normalize → compute log-odds/P MI or train discriminative log-odds from labeled corpus → combine with TF-IDF/embeddings → train classifier.
- Discuss trade-offs: lexicon frequencies are interpretable but brittle; corpus-derived discriminative scores capture dataset specifics but can overfit.

### Practice exercises

Compute per-document pos/neg frequencies and polarity

- Task: Given a dataset of 200 labeled short reviews and a small sentiment lexicon, compute pos_freq, neg_freq, pos_tf, neg_tf, polarity_diff, and polarity_logodds for each document. Plot polarity_diff histograms for positive and negative classes and report overlap.
- Hints: Use smoothing=0.5 for log-odds; normalize by token count for TF.

Log-odds token ranking

- Task: From the labeled corpus, compute token-level log-odds scores (using document-frequency per class with Laplace smoothing alpha=1). Output top 20 positive and top 20 negative tokens. Manually inspect for spurious tokens and propose fixes.
- Hints: Use set(tokenize(doc)) per document to compute document-frequency, not raw counts, to reduce length bias.

Negation experiment with frequencies

- Task: Create variants of your preprocessing:
A) simple unigram counts,
B) negation-aware tokenization (convert "not good" → "not_good"),
C) use bigrams.
For each, compute pos/neg features and train a logistic regression on these features (only frequency-based features). Compare accuracy and precision/recall.
- Hints: Implement negation by scanning for negation words ("not", "no", "never") and joining the next 1–3 tokens.

PMI vs log-odds comparison

- Task: Compute PMI(word, pos) and log-odds(word) for tokens with minimum df >= 5. Rank tokens by each metric and compare top-20 lists. Explain differences.
- Hints: Estimate probabilities from document counts: P(word & pos) ≈ count(word in pos-docs) / N_total.

Build a hybrid classifier

- Task: Create a feature set combining:
    - TF-IDF vector
    - pos_tf and neg_tf
    - top-50 token log-odds values (document frequency per doc as indicators)
    Train a logistic regression and compare to TF-IDF-only baseline. Report which tokens improved performance.
- Hints: Use sklearn Pipeline to concatenate sparse TF-IDF matrix with dense engineered features (use scipy.sparse.hstack).

Short walkthrough for Exercise 2 (log-odds ranking):

```python
# inputs: docs (list of strings), labels (0/1)
from collections import defaultdict
alpha = 1.0
count_pos_doc = defaultdict(int)
count_neg_doc = defaultdict(int)
total_pos_docs = sum(1 for l in labels if l==1)
total_neg_docs = sum(1 for l in labels if l==0)

for doc, label in zip(docs, labels):
    toks = set(tokenize(doc))
    if label==1:
        for t in toks: count_pos_doc[t] += 1
    else:
        for t in toks: count_neg_doc[t] += 1

V = len(set(t for d in docs for t in tokenize(d)))
log_odds = {}
for t in set(t for d in docs for t in tokenize(d)):
    p_pos = (count_pos_doc[t] + alpha) / (total_pos_docs + alpha*V)
    p_neg = (count_neg_doc[t] + alpha) / (total_neg_docs + alpha*V)
    log_odds[t] = np.log(p_pos / p_neg)

# sort
top_pos = sorted(log_odds.items(), key=lambda x: -x[1])[:20]
top_neg = sorted(log_odds.items(), key=lambda x: x[1])[:20]
```

---

## Feature extraction with frequencies

### Direct definition

**Feature extraction with frequencies** is the process of converting text into numeric features by counting or weighting occurrences of tokens (words, subwords, characters, n‑grams) and transforming those counts into signals (raw counts, term frequency, TF‑IDF, log‑odds, PMI, polarity scores) that a model can use.

### Concept intuition

- Frequencies are first-order signals: they measure evidence for concepts (topics, sentiment) by how often signal-bearing tokens appear.
- They are fast, interpretable, and often surprisingly effective as baseline features for classification tasks.
- Frequency features expose the model to important tokens directly (e.g., "great" → positive evidence). The job of downstream learning is then mainly to weight and combine these evidences, rather than to discover token importance from scratch.
- Use-cases: quick baselines, interpretable models, feature engineering for small-data scenarios, hybrid systems combining lexicons and learned features.

### Mathematical breakdown (clean, copy-paste formulas)

Raw count vector for a document d:

```
x_j(d) = count(token_j, d)
```

Term Frequency (normalized by doc length):

```
tf_j(d) = x_j(d) / sum_k x_k(d)
```

Inverse Document Frequency (smoothed):

```
idf_j = log((N + 1) / (df_j + 1)) + 1.0
```

TF-IDF:

```
tfidf_j(d) = tf_j(d) * idf_j
```

Document-level positive / negative counts (given lexicons V_pos, V_neg):

```
pos_freq(d) = sum_{t in V_pos} x_t(d)
neg_freq(d) = sum_{t in V_neg} x_t(d)
pos_tf(d) = pos_freq(d) / total_tokens(d)
neg_tf(d) = neg_freq(d) / total_tokens(d)
polarity_diff(d) = pos_tf(d) - neg_tf(d)
polarity_logodds(d) = log((pos_freq(d) + s) / (neg_freq(d) + s))
```

Token-level log-odds ratio (document-frequency, Laplace smoothing):

```
p(token|class=pos) = (df_pos(token) + alpha) / (total_pos_docs + alpha * V)
p(token|class=neg) = (df_neg(token) + alpha) / (total_neg_docs + alpha * V)
log_odds_token = log( p(token|pos) / p(token|neg) )
```

Pointwise Mutual Information (PMI) for token vs positive class:

```
PMI(token, pos) = log( P(token & pos) / (P(token) * P(pos)) )
estimate from counts: P(token & pos) ≈ df_pos(token) / N
P(token) ≈ df(token) / N ; P(pos) = total_pos_docs / N
```

Notes on smoothing and stability:

- Use alpha >= 1 (Laplace) or 0.5 (additive smoothing) to avoid zeros in denominators and extreme log values.
- Use small constants eps for TF normalization denominators to avoid division-by-zero.

### Code & practical application (copy-paste friendly)

Tokenization, vocab, count matrix (NumPy + plain Python)

```python
import numpy as np
from collections import Counter, defaultdict

# toy documents
docs = [
    "I love this product it is amazing",
    "This is terrible I hate it",
    "Good quality but poor support",
    "Absolutely fantastic experience",
    "Not good at all, very bad support"
]

def tokenize(text):
    return text.lower().replace(".", "").replace(",", "").split()

# build vocab (min_freq = 1)
min_freq = 1
counter = Counter(tok for d in docs for tok in tokenize(d))
vocab = {tok:i for i,(tok,c) in enumerate([x for x in counter.items() if x[1]>=min_freq])}
V = len(vocab)

# counts matrix (N x V)
N = len(docs)
X_counts = np.zeros((N, V), dtype=float)
for i,d in enumerate(docs):
    for t in tokenize(d):
        if t in vocab:
            X_counts[i, vocab[t]] += 1

```

TF and TF-IDF from scratch

```python
# term frequency (tf)
doc_lengths = X_counts.sum(axis=1, keepdims=True)  # shape (N,1)
tf = X_counts / (doc_lengths + 1e-9)

# idf
df = np.sum(X_counts > 0, axis=0)                   # document frequency per token
idf = np.log((N + 1) / (df + 1)) + 1.0              # smoothed idf

# tf-idf
tfidf = tf * idf

```

Positive/negative frequency features + polarity

```python
# toy lexicons
pos_lex = {"love","amazing","good","fantastic"}
neg_lex = {"terrible","hate","poor","bad","not"}  # include 'not' if treating as negative cue

def pos_neg_features(docs, vocab, pos_lex, neg_lex):
    pos_freqs = []
    neg_freqs = []
    for d in docs:
        toks = tokenize(d)
        c = Counter(toks)
        pos_freq = sum(c[t] for t in pos_lex if t in c)
        neg_freq = sum(c[t] for t in neg_lex if t in c)
        total = max(len(toks), 1)
        pos_freqs.append(pos_freq)
        neg_freqs.append(neg_freq)
    pos_freqs = np.array(pos_freqs, dtype=float)
    neg_freqs = np.array(neg_freqs, dtype=float)
    pos_tf = pos_freqs / (np.array([len(tokenize(d)) for d in docs]) + 1e-9)
    neg_tf = neg_freqs / (np.array([len(tokenize(d)) for d in docs]) + 1e-9)
    polarity_diff = pos_tf - neg_tf
    polarity_logodds = np.log((pos_freqs + 0.5) / (neg_freqs + 0.5))
    return pos_freqs, neg_freqs, pos_tf, neg_tf, polarity_diff, polarity_logodds

pos_freqs, neg_freqs, pos_tf, neg_tf, polarity_diff, polarity_logodds = \\
    pos_neg_features(docs, vocab, pos_lex, neg_lex)

```

Token-level log-odds ranking (document frequency based)

```python
def token_log_odds(docs, labels, alpha=1.0):
    # labels: 1 for pos, 0 for neg; N docs
    total_pos_docs = sum(1 for l in labels if l==1)
    total_neg_docs = sum(1 for l in labels if l==0)
    Vset = set(tok for d in docs for tok in tokenize(d))
    V = len(Vset)

    df_pos = defaultdict(int)
    df_neg = defaultdict(int)
    for d,label in zip(docs, labels):
        toks = set(tokenize(d))
        for t in toks:
            if label==1:
                df_pos[t] += 1
            else:
                df_neg[t] += 1

    log_odds = {}
    for t in Vset:
        p_pos = (df_pos[t] + alpha) / (total_pos_docs + alpha * V)
        p_neg = (df_neg[t] + alpha) / (total_neg_docs + alpha * V)
        log_odds[t] = np.log(p_pos / p_neg)
    return log_odds

# example labels (toy)
labels = [1, 0, 1, 1, 0]
log_odds = token_log_odds(docs, labels, alpha=1.0)
sorted_pos = sorted(log_odds.items(), key=lambda x: -x[1])
sorted_neg = sorted(log_odds.items(), key=lambda x: x[1])

```

Combining frequency features with TF-IDF into a classifier (scikit-learn sketch)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF sparse matrix via sklearn
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(docs)  # shape (N, V_tfidf)

# engineered dense features (N x F)
dense_feats = np.vstack([pos_tf, neg_tf, polarity_diff, polarity_logodds]).T

# normalize dense features
scaler = StandardScaler()
dense_scaled = scaler.fit_transform(dense_feats)

# combine (sparse + dense)
X_combined = hstack([X_tfidf, dense_scaled])

# labels as array
y = np.array(labels)

clf = LogisticRegression(max_iter=1000).fit(X_combined, y)

```

### Visualization / Geometry

- 2D feature views:
    - Plot pos_tf vs neg_tf: each document is a point; neutral docs near the diagonal; positives cluster high pos_tf, negatives high neg_tf.
    - Project TF-IDF vectors via PCA/t-SNE to 2D and color by polarity to see cluster structure.
- Token spectrum:
    - Bar chart tokens sorted by token-level log-odds: left = strongly negative, right = strongly positive. This visual is a sanity check for what the model will rely on.
- Decision boundary intuition:
    - If classifier uses only (pos_tf, neg_tf), decision boundary is a line: w1*pos_tf + w2*neg_tf + b = 0. On the 1D polarity axis (pos_tf - neg_tf) the model reduces to a threshold.
- Loss/gradient intuition:
    - Frequency features are usually dense low-dimensional vectors when aggregated (e.g., 4 features pos/neg/log-odds/pol_diff). Gradients with respect to these features show which direction reduces loss: increasing weights for pos tokens moves positive-labeled docs closer to positive side.

### Common pitfalls & tips

- Pitfall: Counting negation words as negative without scope handling (e.g., "not good" counted as both neg and good). Tip: handle negation by joining tokens (not_good) or using bigrams/short-window markers.
- Pitfall: Long documents dominate raw counts. Tip: use normalized TF or length normalization.
- Pitfall: Zero counts lead to infinite logs. Tip: always use smoothing (alpha, add-0.5) before ratio/log computations.
- Pitfall: Lexicons don't cover domain-specific sentiment words. Tip: compute corpus log-odds or PMI to expand lexicon automatically.
- Pitfall: Overfitting to spurious tokens (product names, user IDs). Tip: inspect top log-odds tokens and remove obvious metadata or include additional regularization.
- Tip: Combine frequency-based features with TF-IDF or embeddings to get interpretability + generalization.
- Tip: Use document-frequency (df) rather than raw token counts when computing token-level association to reduce influence of long docs.

### Interview-ready insights

- Explain and derive why smoothing is necessary for log-odds: to avoid division by zero and to control variance for rare tokens.
- Describe difference between TF-IDF and log-odds: TF-IDF weights token importance within the corpus structure; log-odds expresses token discriminativeness between classes. Combine both for stronger models.
- When and why to prefer frequency features: small datasets, need for interpretability, quick baseline, or hybrid systems where lexicon signals are crucial.
- Explain PMI vs log-odds: PMI measures association relative to independence; log-odds compares conditional probabilities between classes and is more directly discriminative.
- Be ready to show how to extract top k positive/negative tokens from a trained linear model (inspect coef_ aligned with feature names) and explain what they reveal.

### Practice exercises (with hints and short walkthroughs)

Compute and visualize document polarity distribution

- Task: Given 500 labeled short reviews, implement pos_freq, neg_freq, pos_tf, neg_tf, polarity_diff, polarity_logodds. Plot histograms of polarity_diff for positive and negative classes and compute overlap (e.g., KS statistic or simply means & stds).
- Hint: Use smoothed log-odds with s=0.5. Normalize counts by token length for tf measures.

Token log-odds ranking and cleaning

- Task: From the labeled dataset, compute token-level log-odds (use df per class, Laplace alpha=1). Output top-50 positive and negative tokens. Manually inspect and remove likely spurious tokens (product names, reviewer handles). Recompute and observe change.
- Hint: Use set(tokenize(doc)) when updating df counters.

Negation handling comparison

- Task: Build three pipelines and compare classification performance (only frequency-based features):
A) unigram counts,
B) bigrams included,
C) negation-aware (convert "not X" → "not_X" for next token).
Train logistic regression with these features and compare precision/recall.
- Hint: Use scikit-learn CountVectorizer with ngram_range, and implement negation by a simple regex transform before vectorization.

Hybrid feature ablation

- Task: Train three logistic models: (1) TF-IDF only, (2) TF-IDF + pos/neg frequency features, (3) TF-IDF + pos/neg + top-100 token log-odds indicators. Report which combination improves validation F1 and which added features help most.
- Hint: Use sklearn FeatureUnion or sparse hstack to combine sparse and dense features.

PMI vs log-odds comparison

- Task: For tokens with df >= 5, compute PMI(token,pos) and log_odds(token). Compare the top 20 lists and explain tokens that rank differently.
- Hint: Estimate probabilities using document counts; sort by metric and inspect examples where PMI highlights rare but strongly associated tokens.

Quick walkthrough for Exercise 1 (synthesized steps):

```python
# assume docs: list[str], labels: list[int]
# 1) compute pos/neg lexicons (or use provided)
# 2) compute pos_freq, neg_freq per doc as shown earlier
# 3) compute pos_tf = pos_freq / doc_length
# 4) compute polarity_diff = pos_tf - neg_tf
# 5) plot histograms using matplotlib:
#    plt.hist(polarity_diff_for_pos, bins=30, alpha=0.6, label='pos')
#    plt.hist(polarity_diff_for_neg, bins=30, alpha=0.6, label='neg')
#    plt.legend()
```

---

## Preprocessing

### Direct definition

Preprocessing is the set of deterministic text transformations and cleaning steps applied to raw text to produce normalized tokens or sequences that downstream models and feature extractors can reliably consume.

### Concept intuition

Preprocessing turns messy, real-world text into predictable, comparable inputs so models learn signal instead of noise. It removes or normalizes surface variation (case, punctuation, repeated characters, URLs, numbers), handles token boundaries (tokenization, subword splits), and encodes structural cues (sentence boundaries, negation scope, emojis). Good preprocessing improves sample efficiency, reduces spurious correlations, and makes lexical features and embeddings more stable across domains.

### Mathematical / algorithmic breakdown

Normalization mapping

```
normalize(text) -> text'
text' = lower(text)  or  preserve_case(text)
text' = replace_urls(text', "<URL>")
text' = replace_numbers(text', "<NUM>")
text' = unicode_normalize(text')
```

Tokenization (space-based / rule-based / subword)

```
tokens = tokenize(text')  # returns [t1, t2, ..., tL]
space_tokenize: split on whitespace
regex_tokenize: use pattern r"\\w+|[^\\s\\w]"
subword_tokenize: apply BPE/WordPiece to get subword ids
```

Vocabulary / indexing

```
vocab = {tok: idx for tok in top_k_tokens_or_subwords}
ids = [vocab.get(tok, unk_id) for tok in tokens]
```

Sequence handling

```
pad_or_truncate(ids, max_len) -> ids_padded
mask = [1 if i < length else 0 for i in range(max_len)]
```

Optional feature transforms

```
counts = count_vector(tokens)           # bag-of-words
tf = counts / sum(counts)               # term frequency
tfidf = tf * idf_vector                 # TF-IDF weighting
embeddings = E[ids_padded]              # lookup to produce (max_len, d)
pooled = pool(embeddings)               # mean/sum/max -> (d,)
```

Special handling examples

```
negation_scope(tokens, window=3) -> mark tokens "not_good"
emoji_map(":)") -> "<SMILE>"
repeat_norm("soooo") -> "soo" or "so<rep>"
```

### Code & practical application (copy-paste ready)

Robust tokenizer + normalization (Python)

```python
import re
import unicodedata

URL_RE = re.compile(r"https?://\\S+|www\\.\\S+")
NUM_RE = re.compile(r"\\d+([.,]\\d+)*")
PUNC_RE = re.compile(r"[^\\w\\s]")

def normalize_text(s, lower=True):
    s = unicodedata.normalize("NFKC", s)
    s = URL_RE.sub(" <URL> ", s)
    s = NUM_RE.sub(" <NUM> ", s)
    s = s.replace("\\n", " ").strip()
    if lower:
        s = s.lower()
    return s

def simple_tokenize(s):
    s = normalize_text(s)
    # keep words and contractions and emojis
    tokens = re.findall(r"\\w+'?\\w+|\\w+|[^\\s\\w]", s)
    return tokens

# example
text = "I absolutely love this! Visit <https://x.com>. Not good :("
print(simple_tokenize(text))

```

Negation marking (scope-based simple rule)

```python
NEG_WORDS = {"not","no","never","n't"}
STOPPERS = {".","!", "?"}

def negation_mark(tokens, window=3):
    out = []
    neg = False
    neg_count = 0
    for t in tokens:
        if t in NEG_WORDS:
            neg = True
            neg_count = 0
            out.append(t)
            continue
        if neg:
            out.append("NEG_" + t)
            neg_count += 1
            if t in STOPPERS or neg_count >= window:
                neg = False
        else:
            out.append(t)
    return out

# example
tokens = simple_tokenize("I do not like this movie. It was bad")
print(negation_mark(tokens))

```

Subword tokenization example using sentencepiece (conceptual snippet)

```python
# install sentencepiece and train on corpus, then:
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='m.model')
ids = sp.encode("unbelievable", out_type=int)
pieces = sp.encode("unbelievable", out_type=str)

```

End-to-end preprocessing pipeline (vectorize + pad + embed)

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
def texts_to_padded_ids(texts, vocab, unk_id=1, max_len=128):
    id_seqs = []
    for t in texts:
        toks = simple_tokenize(t)
        ids = [vocab.get(tok, unk_id) for tok in toks][:max_len]
        id_seqs.append(ids)
    return pad_sequences(id_seqs, maxlen=max_len, padding='post', truncating='post', value=0)

# example vocab (build from train set)
# embed = tf.keras.layers.Embedding(input_dim=len(vocab)+2, output_dim=128)

```

### Visualization and geometric intuition

- Token normalization reduces noisy axes in vector space; mapping "Dog", "dog!", "doggg" to the same or nearby token reduces spurious spread in embedding space.
- Tokenization determines basis vectors: one-hot axes for tokens or subwords. Subword tokens cluster related morphological variants, creating nearby vectors in embedding space so averages/sums produce coherent document points.
- Padding and masks shape computation graphs: mask zeros remove padded contributions so pooled vectors represent real tokens only.
- Negation-marked tokens shift document vectors in sentiment direction by flipping sign-like indicators (e.g., "good" → "NEG_good") so classifier weights treat them differently.
- Visual recipe: take TF-IDF or embedding-pooled vectors before and after preprocessing (e.g., with vs without negation marking), project to 2D with PCA/t-SNE, and observe class separability changes.

### Common pitfalls & tips

- Pitfall: Over-normalizing (removing case or punctuation) that destroys signals (proper nouns, sarcasm markers). Tip: test with/without certain normalizations and keep features like emojis or punctuation if they carry information.
- Pitfall: Removing numbers or URLs blindly loses signal (order tracking IDs, product versions). Tip: replace with tokens <NUM> and <URL> rather than deleting.
- Pitfall: Poor tokenization for languages or noisy text (social media). Tip: use language-specific tokenizers or pretrained subword tokenizers (SentencePiece, Hugging Face tokenizers).
- Pitfall: Applying train-time vocabulary to new domain leads to many UNK tokens. Tip: use subwords or open vocab (character n-grams) and monitor OOV rate.
- Pitfall: Incorrect padding/masking that lets padded zeros affect pooling/attention. Tip: always provide mask arrays to models and use mask-aware pooling.
- Tip: Keep preprocessing deterministic and saved as a reusable pipeline (serialization) so training and serving match exactly.
- Tip: Log preprocessing statistics: OOV rate, avg length, most common tokens, freq of <NUM>/<URL>, and negation counts.

### Interview-ready insights

- Be ready to explain tokenization trade-offs: word vs subword vs char, memory vs expressivity, and OOV handling.
- Explain why subword tokenization helps and how BPE/WordPiece merges frequent pairs to balance vocabulary size and expressivity.
- Discuss normalization choices: NFKC unicode, lowercasing vs case-preserving, URL/number placeholders, and why deterministic mapping is critical for production.
- Explain masking and padding: how attention or pooling must ignore pads and how mask propagation works in frameworks.
- Describe practical pitfalls in deployment: unseen tokens, different unicode normalization, inconsistent preprocessing between training and inference.
- Show knowledge of evaluation: measure preprocessing impact by ablation (train with and without a step), report delta on validation metrics.

### Practice exercises

1. Build and evaluate a preprocessing pipeline
- Task: Implement normalization, tokenization, negation marking, and vocab building on a 1k-sample reviews corpus. Measure OOV rate when using only top-10k tokens. Plot length distribution before and after cleaning.
- Hints: Save vocab to JSON; compute percent tokens mapped to UNK.
1. Negation scope ablation
- Task: Create three pipelines: (A) no negation handling, (B) negation join next token (not_good), (C) negation marker prefix (NEG_token for next 3 tokens). Train TF-IDF + logistic baseline and report change in F1.
- Hints: Use sklearn TfidfVectorizer on transformed text.
1. Subword vs word comparison
- Task: Train a small BPE model (SentencePiece) with vocab sizes {2k, 8k, 32k} on the corpus. Tokenize held-out text and compute average token length per word and OOV rate for pure word vocab. Train an embedding + classifier for each setting and report validation accuracy and model size.
- Hints: Use sentencepiece.SentencePieceTrainer; compare embedding matrix sizes (V*d).
1. Masking correctness test
- Task: Create variable-length sequences, pad to max_len, compute masked mean pooling manually and compare to TensorFlow/Keras GlobalAveragePooling1D with mask argument. Show that unmasked pooling biases results.
- Hints: Implement masked_mean = sum(emb*mask)/sum(mask) per sample.
1. Preprocessing robustness checks
- Task: Build small scripts that perturb inputs (add random emojis, repeat characters, uppercase noise, insert URLs) and measure model confidence change for a trained sentiment classifier. Identify preprocessing steps that stabilize predictions the most.
- Hints: For each perturbation, compute average change in predicted probability magnitude.

---

## Logistic regression overview

### Definition

**Logistic regression** is a supervised classification model that predicts the probability of a binary outcome by applying a sigmoid function to a linear combination of input features. It learns weights that linearly separate classes in feature space and outputs calibrated probabilities.

### Concept intuition

- At its core logistic regression asks: how much evidence does the input provide for the positive class? Each input feature contributes evidence weighted by a learned coefficient. The model sums weighted evidence, produces a score, then converts that score to a probability between 0 and 1 using the sigmoid.
- Analogy: each feature is a witness giving a vote; weights represent the reliability and direction of each witness; the linear score aggregates votes and the sigmoid turns the aggregated vote into a confidence level.
- Why it matters: logistic regression is a simple, fast, interpretable baseline for classification tasks (including sentiment analysis). It connects directly to probabilistic modeling, convex optimization, regularization, and feature engineering. Many modern pipelines use logistic regression as a baseline or as the final linear head on top of learned representations (embeddings).

### Mathematical breakdown

Model equations (single example x in R^d):

```
z = w^T x + b
p = sigmoid(z) = 1 / (1 + exp(-z))
y_hat = 1 if p >= 0.5 else 0
```

Binary cross-entropy loss for one example:

```
L(y, p) = -[ y * log(p) + (1 - y) * log(1 - p) ]
```

Dataset average loss (m examples):

```
J(w, b) = (1/m) * sum_{i=1..m} L(y(i), p(i))
```

Gradient of loss w.r.t. parameters (vectorized forms):

```
z = X.dot(w) + b             # shape (m,)
p = sigmoid(z)               # shape (m,)

dw = (1/m) * X.T.dot(p - y)  # shape (d,)
db = (1/m) * sum(p - y)      # scalar
```

With L2 regularization (weight decay lambda):

```
J_reg = J + (lambda/(2*m)) * ||w||^2
dw_reg = dw + (lambda/m) * w
```

Why these formulas work:

- The sigmoid links a linear model to probability space.
- Cross-entropy is the negative log-likelihood for Bernoulli labels; minimizing it maximizes likelihood.
- The gradient p - y is the residual in probability space; multiplying by X sums feature-weighted residuals across the dataset, giving the direction to change weights to reduce error.

Derivation sketch (single example):

- dL/dz = p - y (classic result).
- dL/dw = (p - y) * x, averaging over examples gives dw formula.

### Code and practical application

NumPy implementation from scratch (train with gradient descent)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_numpy(X, y, lr=0.1, epochs=1000, lam=0.0):
    m, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for epoch in range(epochs):
        z = X.dot(w) + b
        p = sigmoid(z)
        dw = (1/m) * X.T.dot(p - y) + (lam/m) * w
        db = (1/m) * np.sum(p - y)
        w -= lr * dw
        b -= lr * db
    return w, b

# tiny example
X = np.array([[1,2],[1,-1],[2,1],[ -1,-2 ]], dtype=float)
y = np.array([1,1,1,0], dtype=float)
w,b = train_logistic_numpy(X,y, lr=0.5, epochs=2000)
print("w,b", w, b)
```

Scikit-learn quick baseline

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
clf.fit(X, y)
print("coef:", clf.coef_, "intercept:", clf.intercept_)
```

Keras example using TF embeddings pipeline

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Suppose X_tfidf is (N, V) sparse dense matrix and y is labels
inputs = layers.Input(shape=(V,))
outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_tfidf, y, batch_size=32, epochs=10)
```

Using logistic regression as a linear classifier on top of embeddings (PyTorch sketch)

```python
import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

model = LinearClassifier(d)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()
```

Practical workflow notes:

- Start with TF-IDF + logistic for quick baselines.
- Use L2 regularization and standardized features (scale dense features).
- For high-dimensional sparse X (bag-of-words), prefer solvers that work with sparse matrices (scikit-learn handles sparse CSR).
- When using embeddings, freeze or fine-tune embeddings depending on data size.

### Visualization and geometric intuition

- Geometry: logistic regression finds a hyperplane {x | w^T x + b = 0} that separates feature space into two half-spaces. w points toward the region of positive class; the magnitude of w controls margin steepness.
- Sigmoid geometry: score z maps along the real line; near z=0 the sigmoid gradient is largest, so examples near the hyperplane drive learning most.
- Loss surface: for linear + cross-entropy the loss is convex in w and b; visualize 2D slices of J(w1,w2) as a convex bowl. Adding L2 shifts and smooths the minimum, penalizing large weights.
- Gradients: dw is a feature-weighted average of residuals (p - y). In geometry, dw points toward the direction reducing misclassification mass on one side of the hyperplane.
- Visual recipe:
    - Project data to 2D with PCA or t-SNE and plot points colored by class.
    - Draw decision boundary line from learned w,b in 2D projection (train a new logistic on projected coords for true boundary).
    - Plot sigmoid output for sample points along the normal to the hyperplane to see probability transition.

### Common pitfalls and tips

- Pitfall: interpreting coefficients without considering feature scaling. Tip: standardize or normalize features; then coefficients are comparable.
- Pitfall: overfitting high-dimensional sparse data without regularization. Tip: use L2 (ridge) or L1 (sparse) regularization; use C parameter in scikit-learn (inverse of lambda).
- Pitfall: reporting accuracy on imbalanced data. Tip: use precision, recall, F1, ROC-AUC and class-weighting or resampling when classes are imbalanced.
- Pitfall: using default solvers on extremely large sparse matrices. Tip: choose solvers optimized for sparsity (liblinear, saga) and use sparse inputs.
- Tip: inspect top positive and negative coefficients to sanity-check model behavior.
- Tip: use learning curves to detect underfitting vs overfitting and adjust regularization or add features accordingly.
- Tip: add interaction features or n-grams if single tokens miss important context (e.g., "not good").

### Interview ready insights

- Be able to derive dL/dz = p - y and then dL/dw = (p - y) x. Show the short derivation: sigmoid derivative and chain rule.
- Explain why cross-entropy is chosen: it is the negative log-likelihood of Bernoulli model and yields convex objective for linear model.
- Discuss regularization: L2 penalizes large weights and corresponds to Gaussian prior on weights; L1 encourages sparse weights and can perform feature selection.
- Explain decision boundary and how probability threshold choice affects precision/recall trade-off.
- Discuss calibration: logistic regression outputs calibrated probabilities better than many other models because it directly models log-odds linearly; nonetheless calibration can be improved with temperature scaling or Platt scaling.
- Explain link to other models: logistic regression is a single-layer neural network with sigmoid activation and cross-entropy loss; softmax generalizes it to multi-class classification.
- Discuss numerical stability: use log-sum-exp tricks or stable sigmoid implementations to avoid overflow when z is large in magnitude.

### Practice exercises

Implement logistic regression from scratch with L2 regularization and plot training loss

- Task: Use a small TF-IDF matrix from 2k movie reviews. Implement gradient descent with different learning rates and lambdas. Plot loss vs epochs and validation accuracy.
- Hints: monitor dw norms; try lr in {0.1, 0.01, 0.001} and lambda in {0, 0.01, 0.1}.

Inspect coefficients for interpretability

- Task: Train sklearn LogisticRegression on a TF-IDF representation. Extract top 30 positive and top 30 negative tokens and check for spurious tokens.
- Hints: match coef_ order with vectorizer.get_feature_names_out().

Class imbalance experiment

- Task: Create an imbalanced dataset (90% negatives). Train logistic regression with and without class_weight='balanced'. Compare precision, recall, F1 for positive class.
- Hints: use sklearn.metrics.classification_report.

Threshold tuning

- Task: Train logistic regression and compute predicted probabilities on validation set. Sweep threshold from 0.01 to 0.99 and plot precision-recall and F1 vs threshold. Choose threshold maximizing F1 or matching business metric.
- Hints: use sklearn.metrics.precision_recall_curve.

From logistic regression to a neural classifier

- Task: Replace TF-IDF inputs with average word embeddings and train a Keras logistic head (Embedding -> GlobalAveragePooling -> Dense sigmoid). Compare performance with TF-IDF baseline and comment on feature richness and data needs.
- Hints: freeze vs fine-tune embeddings; compare when training data is small vs large.

Short walkthrough for Exercise 1 (skeleton)

```python
# assume X_train (m x d), y_train (m,), X_val, y_val
w = np.zeros(d); b = 0.0
for epoch in range(epochs):
    z = X_train.dot(w) + b
    p = 1 / (1 + np.exp(-z))
    loss = -np.mean(y_train*np.log(p+1e-12) + (1-y_train)*np.log(1-p+1e-12)) + (lam/(2*m))*np.sum(w*w)
    dw = (1/m)*X_train.T.dot(p - y_train) + (lam/m)*w
    db = np.mean(p - y_train)
    w -= lr * dw
    b -= lr * db
    # compute val acc using p_val = sigmoid(X_val.dot(w)+b)
```

---

## Logistic regression training

### Definition

Logistic regression training is the process of estimating the parameters (weights w and bias b) of a logistic model so that the model’s predicted probabilities match labeled binary outcomes; training minimizes a loss (binary cross-entropy) over a dataset using optimization (gradient descent, stochastic methods, or specialized solvers).

### Intuition: what training is doing and why it matters

- The model computes a linear score z = w^T x + b that measures evidence for the positive class. The sigmoid turns that score into a probability p in (0,1). Training moves w and b so that p is close to observed labels y across the training set.
- Training is repeatedly: predict → measure error → compute gradient (how each parameter affects error) → update parameters to reduce error. For large datasets we use minibatches (stochastic gradient descent) so updates are noisy but cheap and often generalize better.
- Why it matters: good training yields calibrated probabilities, robust separation between classes, and weights that reflect feature importance. Training choices (learning rate, batch size, regularization, optimizer) strongly affect convergence speed, generalization, and stability.

### Mathematical breakdown

Model forward pass:

```
z = w^T x + b
p = sigmoid(z) = 1 / (1 + exp(-z))
```

Binary cross-entropy loss for one example:

```
L(y, p) = - [ y * log(p) + (1 - y) * log(1 - p) ]
```

Average loss over m examples:

```
J(w,b) = (1/m) * sum_{i=1..m} L(y(i), p(i))
```

Vectorized gradient (no regularization):

```
z = X.dot(w) + b        # shape (m,)
p = sigmoid(z)          # shape (m,)
dw = (1/m) * X.T.dot(p - y)   # shape (d,)
db = (1/m) * sum(p - y)       # scalar
```

Add L2 regularization (weight decay lambda):

```
J_reg = J + (lambda/(2*m)) * ||w||^2
dw_reg = dw + (lambda/m) * w
db_reg = db
```

SGD / minibatch update (learning rate alpha):

```
w := w - alpha * dw_batch
b := b - alpha * db_batch
```

Batch size B effect:

- Full-batch (B = m): exact gradient, deterministic updates, expensive per step.
- Minibatch (1 < B < m): tradeoff between noise and compute.
- SGD (B = 1): noisy updates, can escape shallow local plateaus, cheap per step.

Convergence diagnostics:

- Monitor training and validation loss curves.
- If training loss decreases and validation loss increases → overfitting (increase regularization, reduce model capacity, get more data).
- If both losses plateau high → underfitting or learning rate too small.

### Practical code: from-scratch, sklearn, PyTorch, TensorFlow

Minimal NumPy logistic regression training (batch gradient descent)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_numpy(X, y, lr=0.1, epochs=1000, lam=0.0, batch_size=None, verbose=False):
    m, d = X.shape
    w = np.zeros(d)
    b = 0.0
    if batch_size is None:
        batch_size = m
    for epoch in range(epochs):
        # shuffle
        perm = np.random.permutation(m)
        Xs = X[perm]; ys = y[perm]
        for i in range(0, m, batch_size):
            Xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            B = Xb.shape[0]
            z = Xb.dot(w) + b
            p = sigmoid(z)
            dw = (1.0/B) * Xb.T.dot(p - yb) + (lam/m) * w
            db = (1.0/B) * np.sum(p - yb)
            w -= lr * dw
            b -= lr * db
        if verbose and (epoch % (epochs//10 + 1) == 0):
            z_all = X.dot(w) + b
            p_all = sigmoid(z_all)
            loss = -np.mean(y*np.log(p_all+1e-12) + (1-y)*np.log(1-p_all+1e-12)) + (lam/(2*m))*np.sum(w*w)
            print(f"epoch {epoch} loss {loss:.4f}")
    return w, b

# Usage example:
# X: (m,d) numpy float, y: (m,) with 0/1 labels
```

Quick scikit-learn training (recommended baseline)

```python
from sklearn.linear_model import LogisticRegression
# For sparse TF-IDF use solver='saga' or 'liblinear' for efficiency on high-d sparse inputs
clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
clf.fit(X_train, y_train)
# coef = clf.coef_.ravel(), intercept = clf.intercept_[0]
```

PyTorch training loop (minibatch + optimizer)

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

class LogisticModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

def train_torch(X, y, lr=1e-3, epochs=20, batch_size=32, weight_decay=0.0, device='cpu'):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = LogisticModel(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(ds)
        # optionally compute val metrics here
    return model

# Note: for numerical stability with logits, prefer BCEWithLogitsLoss + raw logits.
```

TensorFlow / Keras training (dense input)

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def build_and_train_tf(X_train, y_train, X_val=None, y_val=None, lr=1e-3, lam=1e-3, epochs=10, batch_size=32):
    m, d = X_train.shape
    inputs = layers.Input(shape=(d,))
    outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(lam))(inputs)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val) if X_val is not None else None,
              epochs=epochs, batch_size=batch_size, verbose=2)
    return model

# For sparse TF-IDF, convert to dense or use tf.data pipeline with sparse tensors.

```

Numerical-stability note (use logits where possible)

- Using raw logits z and BCEWithLogitsLoss (PyTorch) or from_logits=True in TF is numerically stable:

```python
# PyTorch: loss_fn = nn.BCEWithLogitsLoss(); pred_logits = linear(x); loss = loss_fn(pred_logits, y)
# TF: tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

### Visualization & geometric intuition for training dynamics

- Decision boundary evolution: plot 2D PCA/t-SNE projection of training points; after each epoch draw a linear decision boundary learned on the projected plane (or train a separate logistic on projected coords). Watch boundary rotate and margin widen as loss decreases.
- Loss surface slices: for two-parameter toy problem, plot J(w1, w2) contours to see convex bowl and descent path of w updates (gradient steps move roughly orthogonally to contour lines).
- Gradient magnitude: plot ||dw|| over epochs — typically large early, then decreases as you approach optimum; spikes indicate too-large learning rate or noisy minibatches.
- Calibration / reliability plot: split predicted probabilities into bins and plot average predicted probability vs actual positive fraction to check calibration; training can affect calibration (regularization and class imbalance matter).
- Learning curves: plot training and validation loss vs epochs. Typical signs:
    - Both high and similar → underfitting; increase capacity/features or decrease regularization.
    - Training low, validation high → overfitting; increase regularization, gather more data, or early-stop.
    - Both decreasing and close → healthy training.

### Common pitfalls, practical tips, and hyperparameter guidance

- Feature scaling: always standardize dense features (zero mean, unit variance) before training; coefficient magnitudes depend on feature scale. For sparse TF-IDF, scaling is usually not needed.
- Regularization: use L2 weight decay. In scikit-learn C is inverse regularization (C = 1/lambda). Tune lambda {1e-4, 1e-3, 1e-2, 1e-1, 1.0}.
- Learning rate: too large → divergence or oscillation; too small → slow training. Typical ranges: for NumPy SGD try 0.1–1.0 for small d, for Adam try 1e-3 to 1e-4.
- Batch size: small batches (32–128) often generalize better; large batches need smaller learning rates.
- Early stopping: use validation loss to stop and avoid overfitting.
- Class imbalance: use class weights or re-sampling. In scikit-learn use class_weight='balanced' or pass class_weight dict.
- Check gradients: implement gradient checks (finite differences) for from-scratch implementations to ensure correctness.
- Numerical stability: clip logits or use loss functions that accept logits (BCEWithLogitsLoss / from_logits=True).
- Sparse data: use solvers and libraries that accept CSR sparse matrices (sklearn LogisticRegression handles sparse input).
- Interpretability: inspect top-k positive and negative coefficients to sanity-check model behavior.
- Use mini-experiments: start with small subset and short epochs to iterate hyperparameters quickly.

### Interview-ready insights

- Derive dL/dz = p - y and dL/dw = (p - y) * x; vectorized gradient dw = X^T (p - y) / m.
- Explain why cross-entropy is negative log-likelihood of Bernoulli and leads to convex objective for linear model.
- Discuss tradeoffs: full-batch vs minibatch, SGD noise benefits for generalization, Adam vs SGD with momentum.
- Explain L2 as Gaussian prior on weights; L1 encourages sparsity and feature selection.
- Describe numerical stability: prefer logits + stable loss implementation to avoid overflow in exp() for large z.
- Explain calibration: logistic gives calibrated probabilities when model assumptions roughly hold; regularization and class imbalance affect calibration.
- Discuss practical tuning: scale features, tune learning rate and lambda, use early stopping, inspect coef_ for sanity.

### Practice exercises (small, practical, with hints)

From-scratch gradient check and training

- Task: Implement train_logistic_numpy with analytic gradients; verify correctness with numeric finite differences on a toy example.
- Hint: for finite difference of w_j use (J(w+eps e_j) - J(w-eps e_j)) / (2*eps) and compare to dw_j.

Minibatch vs full-batch comparison

- Task: Train on the same dataset with batch_size = full, 256, 32, 1. Plot training loss vs iterations and final validation accuracy. Report which converges faster and which generalizes best.
- Hint: keep total number of gradient evaluations similar across runs for fair comparison.

Regularization sweep

- Task: Train logistic (sklearn or NumPy) across lambdas {0, 1e-4, 1e-3, 1e-2, 1e-1}. Plot validation F1 vs lambda and show coefficient norms ||w||. Explain relationship between lambda and coefficient magnitude.

Use logits for numerical stability

- Task: Re-implement PyTorch training using BCEWithLogitsLoss and compare training stability to using sigmoid + BCELoss. Observe gradients and loss values for examples with large z.

Calibration check and temperature scaling

- Task: Train logistic model, compute probability calibration curve (10 bins), then compute temperature scaling: find T>0 minimizing NLL on validation when scaling logits z' = z / T. Show before/after calibration plots.
- Hint: temperature is a single scalar optimized with validation loss; use scipy.optimize or simple grid search.

Real-data mini-project

- Task: Use a small TF-IDF representation of 2k movie reviews. Train a logistic model (sklearn) with class_weight if needed. Report precision, recall, F1, ROC-AUC. Extract top 20 positive and negative tokens and comment on whether they are meaningful; remove spurious tokens and retrain to measure change.
- Hint: use sklearn TfidfVectorizer(max_features=10000, ngram_range=(1,2)).

---

## Visualizing tweets and logistic regression models

### 1. Direct definition

Visualizing tweets and logistic regression models means (a) exploring and plotting tweet-level text features (token counts, TF‑IDF, embeddings, metadata) to understand data distributions and signals, and (b) visualizing a trained logistic regression model’s decision surface, coefficients, and calibration so you can interpret how the model separates positive vs negative classes.

### 2. Concept intuition: what we want to see and why it matters

- Tweets are short, noisy, and idiosyncratic (hashtags, mentions, emojis, URLs, contractions, slang). Visualizations help you discover frequent tokens, long-tail tokens, OOV rate, and whether preprocessing (negation handling, emoji mapping) changes separability.
- For logistic regression, visualizations show which features (words, n‑grams, embedding dims) push predictions, how confident predictions are, and where errors happen. This supports debugging (spurious tokens, label noise), feature engineering, and communicating model behavior to stakeholders.
- Good visuals: token frequency bars, cumulative coverage, Zipf plots, word clouds, heatmaps of confusion by token, 2D projections of TF‑IDF or embedding spaces, decision boundary overlays in a 2D projection, coefficient bar charts, calibration plots, and per-tweet explanation plots (e.g., showing token contributions).

### 3. Data preparation and preprocessing specific to tweets (practical rules)

- Normalize: unicode NFKC, lowercasing (or preserve if you want proper nouns), replace URLs with <URL>, numbers with <NUM>, mentions with <USER>, hashtags either keep or split (#Happy -> "happy").
- Tokenize: use a tokenizer that keeps emojis and handles contractions (TweetTokenizer from NLTK or Hugging Face tokenizers).
- Special handling:
    - Emojis: map to sentiment tags (":)" -> <EMO_SMILE>) or keep as tokens.
    - Hashtags: either keep raw "#cat" or split camel case / remove '#'.
    - Mentions and URLs: replace with placeholders to avoid leaking identities.
    - Repeated characters: normalize runs ("sooooo" -> "soo" or "so<rep>").
    - Negation: either mark scope (NEG_token) or include bigrams to capture "not good".
- Build vocab: set min_df or min_freq, consider n-grams (1,2) for short text to capture negation/bigrams.
- Feature choices: CountVectorizer / TfidfVectorizer; or embeddings (pretrained tweet-specific embeddings if available). For logistic regression baseline, TF‑IDF + n‑grams is effective.

### 4. Useful visualizations and what they tell you (with why + brief code sketches)

Token frequency bar (top-k tokens)

- What: show top 30 tokens by corpus count.
- Why: reveals stopwords, hashtags, emoticons, domain words.
- Sketch:

```python
# df_tokens from counts: token,count
top = df_tokens.head(30)
sns.barplot(x='count', y='token', data=top)
plt.title("Top 30 tokens")
```

Cumulative coverage (top-k coverage)

- What: fraction of total tokens covered by top‑k tokens.
- Why: decide vocab cutoff; shows long tail.
- Sketch:

```python
cdf = df_tokens['count'].cumsum() / df_tokens['count'].sum()
plt.plot(df_tokens['rank'], cdf)
plt.xscale('log'); plt.xlabel('rank'); plt.ylabel('coverage')
```

Zipf log-log plot

- What: log(rank) vs log(freq).
- Why: check Zipf law; spot anomalies if top tokens deviate.
- Sketch:

```python
plt.loglog(df_tokens['rank'], df_tokens['count'], marker='.')
plt.xlabel('rank'); plt.ylabel('frequency')
```

Word clouds for positive/negative classes

- What: word clouds built from tweets by label.
- Why: quick qualitative check of sentiment tokens and spurious words.
- Sketch: use wordcloud.WordCloud on concatenated text for each class.

Token-level log-odds / discriminative ranking

- What: compute log-odds(df_pos + alpha / total_pos, df_neg + alpha / total_neg) and bar top tokens.
- Why: shows which tokens are most predictive for each class.
- Sketch:

```python
# compute log_odds dict; show top positive and negative tokens
```

TF‑IDF / embedding 2D projection (PCA / t‑SNE / UMAP)

- What: project tweet vectors to 2D, color by label, overlay misclassified examples.
- Why: see cluster structure, overlap, and hard examples.
- Sketch:

```python
from sklearn.decomposition import PCA
X = vectorizer.fit_transform(tweets)        # TF-IDF sparse
X2 = PCA(n_components=2).fit_transform(X.toarray())  # for small data
sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=labels)
```

- Use t-SNE/UMAP when non-linear structure is expected. For sparse TF‑IDF, convert to dense or use truncated SVD (TruncatedSVD) instead of PCA.

Decision boundary visualization in 2D projection

- What: overlay logistic regression decision contour on 2D projection (trained separately on projected features or approximate boundary).
- Why: see how linear separator interacts with data clusters and what errors look like.
- Sketch:

```python
# Fit logistic on X2 (2D) to get exact 2D boundary; plot contourf over grid
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression().fit(X2, labels)
# grid -> predict_proba -> contourf
```

Coefficient bar chart for top features

- What: show the top K positive and top K negative coefficients from TF‑IDF logistic model.
- Why: interpret which words drive decisions.
- Sketch:

```python
feat_names = vectorizer.get_feature_names_out()
coefs = clf.coef_.ravel()
top_pos = np.argsort(coefs)[-20:]
top_neg = np.argsort(coefs)[:20]
# plot bars for top_pos and top_neg
```

Calibration plot and reliability curve

- What: bin predicted probabilities and plot mean predicted vs observed fraction.
- Why: check if model probabilities are well-calibrated (important for downstream decisions).
- Sketch: use sklearn.calibration.calibration_curve or calibration_plot.
1. Per-example token contribution (explainer)
- What: show token contributions to logit = w^T x + b for a tweet (or use SHAP/LIME).
- Why: clear local explanation—what tokens pushed prediction.
- Sketch:

```python
# For TF-IDF sparse vector x and coef w:
contribs = x.toarray().ravel() * coefs
# show tokens with top positive/negative contributions
```

### 5. Full practical pipeline (preprocess → visualize → train → inspect) with runnable snippets

Assumptions: tweets list, labels 0/1. Minimal libraries: sklearn, pandas, seaborn, matplotlib, wordcloud, umap-learn (optional), shap (optional).

Preprocess + tokenization:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import re, unicodedata

def normalize_tweet(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'https?://\\S+|www\\.\\S+', ' <URL> ', s)
    s = re.sub(r'@\\w+', ' <USER> ', s)
    s = re.sub(r'\\d+([.,]\\d+)*', ' <NUM> ', s)
    s = s.replace('\\n',' ')
    s = s.lower()
    return s

def basic_tokenizer(s):
    s = normalize_tweet(s)
    # keep emojis and words, minimal split
    tokens = re.findall(r"\\w+'?\\w+|\\w+|[^\\s\\w]", s)
    return tokens

# TF-IDF vectorizer with basic token pattern (sklearn can do its own tokenization)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, token_pattern=r"\\w+'?\\w+|\\w+|[^\\s\\w]")
X_tfidf = vectorizer.fit_transform([normalize_tweet(t) for t in tweets])

```

Top token visualization:

```python
import numpy as np, pandas as pd
from collections import Counter

# corpus token counts (use vectorizer vocabulary mapping)
names = vectorizer.get_feature_names_out()
counts = np.asarray(X_tfidf.sum(axis=0)).ravel()
df_feats = pd.DataFrame({'token': names, 'count': counts})
df_feats = df_feats.sort_values('count', ascending=False)
sns.barplot(y='token', x='count', data=df_feats.head(30))
plt.title("Top TF-IDF-summed tokens")

```

Train logistic regression baseline:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42, stratify=labels)
clf = LogisticRegression(max_iter=2000, solver='saga', penalty='l2', C=1.0)
clf.fit(X_train, y_train)
print("val acc", clf.score(X_val, y_val))

```

Coefficient inspection:

```python
coefs = clf.coef_.ravel()
top_pos_idx = np.argsort(coefs)[-20:][::-1]
top_neg_idx = np.argsort(coefs)[:20]
for idx in top_pos_idx[:10]:
    print(names[idx], coefs[idx])
for idx in top_neg_idx[:10]:
    print(names[idx], coefs[idx])

```

2D projection + decision boundary (TruncatedSVD recommended for sparse TF-IDF)

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=0)
X_svd = svd.fit_transform(X_tfidf)   # (n_samples,50)
# then use UMAP/t-SNE or PCA for two dims:
import umap
X2 = umap.UMAP(n_components=2, random_state=0).fit_transform(X_svd)

# train logistic on X2 (to visualize actual 2D decision boundary)
clf2 = LogisticRegression().fit(X2, labels)

# scatter + contour
plt.figure(figsize=(8,6))
sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=labels, alpha=0.6, palette='coolwarm')
# grid
xx, yy = np.meshgrid(np.linspace(X2[:,0].min(), X2[:,0].max(), 200),
                     np.linspace(X2[:,1].min(), X2[:,1].max(), 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf2.predict_proba(grid)[:,1].reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.3)
plt.title("2D projection with logistic decision contour")

```

Per-tweet token contribution explanation (simple, exact for linear TF‑IDF logistic):

```python
import numpy as np

def explain_tweet(idx, clf, vectorizer, X_tfidf):
    x = X_tfidf[idx]
    feat_idx = x.nonzero()[1]
    contributions = x.data * clf.coef_.ravel()[feat_idx]
    tokens = vectorizer.get_feature_names_out()[feat_idx]
    pairs = sorted(zip(tokens, contributions), key=lambda x: -abs(x[1]))[:20]
    return pairs, clf.predict_proba(x)[:,1][0]

pairs, prob = explain_tweet(5, clf, vectorizer, X_tfidf)
print("prob", prob)
for tok, cont in pairs:
    print(tok, cont)

```

Calibration plot:

```python
from sklearn.calibration import calibration_curve
y_prob = clf.predict_proba(X_val)[:,1]
frac_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=10)
plt.plot(mean_pred, frac_pos, marker='o')
plt.plot([0,1],[0,1], '--', color='gray')
plt.xlabel('Mean predicted prob'); plt.ylabel('Fraction of positives')
plt.title('Calibration curve')

```

SHAP explanations for linear model (optional but useful)

```python
# shap.LinearExplainer works well for linear models
import shap
explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_val[:50])  # small subset
shap.summary_plot(shap_values, features=X_val[:50].toarray(), feature_names=vectorizer.get_feature_names_out())

```

### 6. Common pitfalls and practical tips when visualizing tweets & logistic models

- Pitfall: projecting high‑dim data to 2D loses information, so decision boundary in 2D is only an approximation. Fix: train a separate linear model in projected 2D to visualize an exact 2D boundary or annotate that projection is approximate.
- Pitfall: converting sparse TF‑IDF to dense for PCA/t‑SNE can be memory heavy. Fix: use TruncatedSVD (LSA) for sparse matrices then t‑SNE / UMAP.
- Pitfall: interpreting raw top-frequency tokens can confuse common tokens with discriminative tokens. Combine frequency and log‑odds to find discriminative tokens.
- Pitfall: ignoring negation/sarcasm. Visualization may show misleading token importance; handle negation and mark sarcasm if possible, examine misclassified examples manually.
- Tip: show misclassified examples on plots (different marker shape) to inspect where the model fails.
- Tip: overlay token contributions on each tweet (color code positive vs negative contributions) to debug per-example reasoning.
- Tip: for short texts, include bigrams or char n‑grams to capture phrases and contractions.

### 7. Interview-ready insights

- Explain why TruncatedSVD is used for TF‑IDF projection (works on sparse matrices, approximates PCA) and why UMAP/t‑SNE are used after an SVD reduction for visualization.
- Be able to justify TF‑IDF + logistic baseline for tweets: robust, interpretable, sparse-friendly, and performant on small/medium datasets.
- Explain how to interpret logistic coefficients: positive coefficient means feature increases log-odds by w_j * x_j; for TF‑IDF x_j is weight so contribution is additive.
- Describe how to compute token-level contributions and why a linear model makes this exact and efficient (no need for approximations).
- Explain calibration curve meaning: a well-calibrated model’s predicted probability equals observed frequency; logistic regression often yields reasonably calibrated probabilities but verify and adjust with temperature scaling if needed.

### 8. Practice exercises

Explore top tokens and discriminative tokens

- Task: On a 2k tweet labeled dataset, compute top 50 tokens by corpus frequency and top 50 by token-level log‑odds. Plot both lists; write a short paragraph about differences and any surprising tokens.
- Hint: compute df_pos and df_neg using set(tokenize(doc)) per doc to reduce length bias.

Visualize TF‑IDF clusters and misclassifications

- Task: Vectorize tweets with TF‑IDF (1,2-grams, min_df=5), reduce with TruncatedSVD → UMAP to 2D, plot points colored by true label and mark misclassified points from a logistic model. List 10 misclassified tweets and inspect them for noise/negation.
- Hint: use clf.predict on TF‑IDF matrix to get predictions; overlay shapes on scatter.

Decision boundary sanity check

- Task: Fit logistic on TF‑IDF projected to 2D (SVD→2 components) and plot decision contour. Then compute accuracy of this 2D logistic vs original high-dim logistic and comment on differences.
- Hint: the 2D logistic is not equivalent but useful for visualization; expect lower performance.

Explain a prediction

- Task: For 20 random tweets in validation set, compute token contributions using the method above and produce an HTML/text report showing tweet, predicted prob, and top 5 positive and negative contributing tokens. Evaluate whether the explanation matches human intuition.
- Hint: sort contributions by value and show tokens with highest magnitude.

Calibration and threshold tuning

- Task: Compute calibration plot and Brier score for your logistic model. Then sweep thresholds 0.01→0.99 to maximize F1. Report threshold that maximizes F1 and resultant precision/recall.
- Hint: use sklearn.metrics.precision_recall_curve.

Robustness to perturbations

- Task: For 100 validation tweets, create perturbed variants: (a) add random emoji at end, (b) insert "<USER>" mention, (c) replace one token with <NUM>. Measure change in predicted probability averaged over perturbation types and identify which preprocessing reduces sensitivity most.
- Hint: run predictions before/after and compute mean absolute change in prob.

---

## Logistic regression testing

### Definition

Logistic regression testing is the process of evaluating a trained logistic regression model on held-out data to measure predictive performance, calibration, robustness, and error modes; it includes metrics, statistical tests, confidence intervals, threshold tuning, and targeted tests (e.g., slice/edge-case evaluation).

### Why it matters

Testing shows whether the model’s learned mapping generalizes to new examples, how confident its probabilities are, where it fails, and whether it meets product or regulatory requirements. Good testing prevents deploying models that appear accurate on training data but fail in the real world due to data shift, class imbalance, or spurious correlations.

### Core evaluation metrics (formulas and meaning)

- Accuracy:

```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- Precision, Recall, F1:

```
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)
```

- ROC AUC:

```
AUC = area under ROC curve (TPR vs FPR over thresholds)
```

- Average Precision (AP) / PR AUC: area under precision-recall curve.
- Log loss (binary cross-entropy):

```
logloss = - (1/N) * sum_i [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
```

- Brier score (calibration / mean squared error of probs):

```
Brier = (1/N) * sum_i (p_i - y_i)^2
```

- Calibration (reliability curve): compute for bins b:

```
mean_pred_b = mean(p_i for i in bin b)
frac_pos_b = mean(y_i for i in bin b)
plot mean_pred_b vs frac_pos_b
```

- Confidence intervals for metrics (approximate, bootstrap):

```
repeat B times: resample test set with replacement, compute metric -> build percentile CI
```

### Practical testing checklist (ordered)

1. Hold-out split and cross-validation
    - Use stratified split for imbalanced labels.
    - Use k-fold or repeated CV for stable estimates when data is limited.
2. Baselines and chance
    - Compare to simple baselines: majority-class, random, rule-based lexicon, TF‑IDF+logistic trained earlier.
    - Report uplift vs baseline.
3. Threshold selection
    - Sweep threshold t over [0,1] and compute precision/recall/F1; pick threshold per business metric (F1, max precision at min recall, cost-weighted).
    - Show PR curve and ROC curve.
4. Calibration check
    - Produce reliability plot and Brier score.
    - If miscalibrated, apply temperature scaling or isotonic regression using a validation set.
5. Error analysis
    - Confusion matrix, per-class metrics.
    - Inspect false positives and false negatives; sample examples and look for patterns (negation, sarcasm, domain terms).
6. Slice testing (data subpopulations)
    - Evaluate on important slices: short vs long tweets, tweets with negation, tweets with emojis, by language, by user demographics (if allowed).
    - Report metrics per slice and per metadata bucket.
7. Robustness tests
    - Perturbations: add emojis, replace tokens with <NUM>, shuffle token order, misspellings, adversarial tokens.
    - Measure average change in predicted probability and classification flip rate.
8. Stability & uncertainty
    - Bootstrap metrics for confidence intervals.
    - If model used in critical systems, compute prediction intervals or use ensembling for uncertainty.
9. Fairness & bias checks (if applicable)
    - Measure disparate impact across protected groups and verify no harmful bias introduced by vocab artifacts or labels.
    - Flag high errors on any subgroup.
10. Regression testing for deployment
    - Compare new model vs production model on the same test set; require improvement on key metrics or pass additional checks before promotion.

### Code: concise test workflow (NumPy / scikit-learn style)

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, log_loss, brier_score_loss,
                             precision_recall_curve)
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
p_test = clf.predict_proba(X_test)[:,1]
y_pred = (p_test >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
auc = roc_auc_score(y_test, p_test)
ll = log_loss(y_test, p_test)
brier = brier_score_loss(y_test, p_test)

# threshold sweep for best F1
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, p_test)
f1s = 2*precisions*recalls/(precisions+recalls+1e-12)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
```

### Visualization checks (what to plot and why)

- ROC curve and AUC: global discrimination ability.
- Precision-Recall curve: useful with class imbalance; shows trade-off.
- Reliability (calibration) plot: predicted prob vs observed fraction.
- Confusion matrix heatmap: class error distribution.
- Metric-by-slice bar charts: expose hidden failures.
- Prediction distribution histograms per class: shows if model pushes probs to extremes or stays uncertain.
- Scatter of probability change under perturbations: robustness signal.

### Common pitfalls & mitigation

- Pitfall: test on data used for training/tuning. Mitigation: strict train/validation/test separation and a separate calibration set if applying temperature scaling.
- Pitfall: single metric blindness (accuracy) on imbalanced data. Mitigation: report precision, recall, F1, PR AUC.
- Pitfall: optimistic estimates due to leakage (user IDs, timestamps). Mitigation: audit features, remove leakage, use time-based splits if appropriate.
- Pitfall: poor calibration. Mitigation: temperature scaling, Platt scaling, isotonic regression on validation set.
- Pitfall: metric instability on small test sets. Mitigation: use cross-validation or bootstrap to estimate CI.
- Pitfall: ignoring slices where model fails. Mitigation: slice testing and targeted improvements.
- Pitfall: not testing runtime/latency/memory. Mitigation: include performance tests for production constraints.

### Interview-ready insights

- Explain difference between discrimination (AUC) and calibration (Brier / reliability curve).
- Describe threshold selection: business-driven; maximize F1 or set threshold to meet recall/precision constraints.
- Explain bootstrap for confidence intervals: resample test set, compute metric distribution, report percentiles.
- Explain temperature scaling: learn scalar T on validation to minimize NLL on validation; apply p = sigmoid(z / T).
- Describe why PR curve is better than ROC when positives are rare: ROC can be overly optimistic; PR focuses on precision at high recall.
- Explain leakage and why time-based splitting prevents optimistic performance for temporally evolving data.

### Practice exercises (small, actionable)

1. Full test-suite on a tweet classifier
    - Task: Given a trained TF‑IDF+logistic model and test tweets, implement: accuracy, precision/recall/F1, ROC AUC, log loss, Brier score, calibration plot, and threshold sweep to pick threshold maximizing F1. Save best threshold.
    - Hint: use sklearn.metrics and calibration_curve.
2. Bootstrap CI for F1
    - Task: Implement B=1000 bootstrap resamples of the test set, compute F1 each time, and report 95% CI.
    - Hint: sample indices with replacement; use np.random.choice with size=N.
3. Slice testing: negation and emoji
    - Task: Create two slices: tweets containing negation tokens and tweets containing emojis. Compute metrics per slice and compare to overall. Report slices with >10% relative drop in F1.
    - Hint: use regex to detect emojis and negation words.
4. Robustness perturbation test
    - Task: For 200 random test tweets, create perturbed versions (append a neutral emoji, replace one token with <NUM>, random misspell one token). Compute average absolute change in predicted probability and flip rate (label change). Report which perturbation type breaks predictions most.
    - Hint: measure mean(|p_orig - p_pert|) and fraction(predict(orig) != predict(pert)).
5. Calibration fix with temperature scaling
    - Task: Using validation split, fit temperature T to minimize NLL on validation. Apply T to test logits and compare NLL and Brier before/after.
    - Hint: use scipy.optimize.minimize on scalar T > 0; optimize NLL = -sum(y*log(sigmoid(z/T)) + (1-y)*log(1-sigmoid(z/T))).

---

## Logistic regression - cost function

### Direct definition

The cost function for logistic regression (binary) is the average binary cross-entropy (negative log-likelihood) between true labels and predicted probabilities, optionally plus a regularization term that penalizes large weights.

### Concept intuition

- The model predicts a probability p = sigmoid(z) where z = w^T x + b. The cost measures how "surprised" the model is by the true labels: confident wrong predictions are punished heavily, confident correct predictions are rewarded (low loss).
- Minimizing cost moves model parameters to make predicted probabilities match empirical label frequencies. Cross-entropy is the natural loss because logistic regression is a Bernoulli probabilistic model and cross-entropy = negative log-likelihood.
- Regularization (L2 or L1) adds a penalty that stabilizes learning, reduces overfitting in high-dimensional/sparse text features, and corresponds to adding a prior on weights (Gaussian for L2).

### Mathematical breakdown

Model definitions:

```
z(i) = w^T x(i) + b
p(i) = sigmoid(z(i)) = 1 / (1 + exp(-z(i)))
```

Binary cross-entropy loss for one example:

```
L(i) = -[ y(i) * log(p(i)) + (1 - y(i)) * log(1 - p(i)) ]
```

Average cost over m examples:

```
J(w,b) = (1/m) * sum_{i=1..m} L(i)
```

L2 regularized cost (weight decay lambda):

```
J_reg(w,b) = J(w,b) + (lambda / (2*m)) * ||w||^2
```

Gradients (vectorized, no regularization):

```
z = X.dot(w) + b           # shape (m,)
p = sigmoid(z)             # shape (m,)
dw = (1/m) * X.T.dot(p - y)  # shape (d,)
db = (1/m) * sum(p - y)      # scalar
```

With L2:

```
dw_reg = dw + (lambda/m) * w
db_reg = db
```

Numerical-stable log-loss expression using logits z directly (avoids computing p first):

```
L(i) = max(0, z) - z*y + log(1 + exp(-abs(z)))
```

Vectorized average:

```
J = (1/m) * sum( max(0,z) - z*y + log(1 + exp(-abs(z))) )
```

This form prevents overflow when z is very large positive or negative.

Variable explanations:

- X: data matrix (m × d), x(i) rows.
- y: labels (m,) with 0/1 values.
- w: weight vector (d,).
- b: bias scalar.
- z: logit (real-valued score).
- p: predicted probability for class 1.
- lambda: L2 regularization strength.

Why gradient = X^T(p - y)/m:

- dL/dz = p - y (chain-rule from cross-entropy + sigmoid).
- dL/dw = x * (dL/dz). Summing/averaging over dataset gives dw formula.

### Code & practical application (NumPy + stable loss + gradient)

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss_and_grad(X, y, w, b, lam=0.0):
    # X: (m,d), y: (m,), w: (d,), b: scalar
    m = X.shape[0]
    z = X.dot(w) + b                  # logits
    # numerically stable loss per example (logistic loss)
    loss_terms = np.maximum(0, z) - z * y + np.log1p(np.exp(-np.abs(z)))
    loss = loss_terms.mean() + (lam / (2*m)) * np.sum(w*w)
    # probabilities (safe for gradient)
    p = sigmoid(z)
    error = p - y                      # (m,)
    dw = (X.T.dot(error)) / m + (lam / m) * w
    db = np.sum(error) / m
    return loss, dw, db

# tiny sanity check
X = np.array([[1.0, 2.0], [1.0, -1.0], [2.0, 1.0], [-1.0, -2.0]])
y = np.array([1, 1, 1, 0], dtype=float)
w = np.zeros(X.shape[1]); b = 0.0
loss, dw, db = logistic_loss_and_grad(X, y, w, b, lam=0.1)
print(loss, dw, db)
```

Practical training loop (mini-batch SGD sketch):

```python
# assume X_train, y_train
w = np.zeros(d); b = 0.0
lr = 0.1; lam = 0.01; epochs = 100; batch_size = 64
for epoch in range(epochs):
    perm = np.random.permutation(len(y))
    for i in range(0, len(y), batch_size):
        idx = perm[i:i+batch_size]
        Xb, yb = X[idx], y[idx]
        loss, dw, db = logistic_loss_and_grad(Xb, yb, w, b, lam)
        w -= lr * dw
        b -= lr * db
```

Framework tip: prefer using logits + stable loss functions:

- PyTorch: use BCEWithLogitsLoss on raw linear output (no sigmoid).
- TensorFlow: BinaryCrossentropy(from_logits=True).

### Visualization / Geometry

- Loss surface: for linear model and cross-entropy the cost is convex in w and b; visualize 2D slices to see a single bowl-shaped minimum. Gradient descent moves downhill along the steepest axis then fine-tunes toward minimum.
- Sigmoid geometry: z on real line; loss penalizes according to distance from correct side; maximum gradient magnitude occurs near z ≈ 0 (p ≈ 0.5). Examples near the decision boundary dominate updates.
- Gradient interpretation: dw is a weighted sum of feature vectors where weights are (p - y). Positive residuals (p>y) push w opposite to features present; negative residuals (p<y) push w along feature directions present in positive examples.
- Visual recipe: plot predicted probabilities vs logits; plot training and validation loss curves; plot magnitude of dw over epochs to check convergence.

### Common pitfalls & tips

- Numerical instability when computing log(p) or log(1-p): use stable logits formulation or library loss functions.
- Forgetting regularization in gradient update (apply lambda/m to dw).
- Mis-scaling features: unscaled features make regularization and learning rate hard to tune; standardize dense features. For sparse TF-IDF features, scaling is often unnecessary.
- Using inappropriate lambda: too large → underfit, too small → overfit. Cross-validate.
- Confusing likelihood vs cross-entropy sign: minimizing cross-entropy = maximizing log-likelihood.
- Using sigmoid + BCELoss (non-logit) can be less stable than logits + BCEWithLogitsLoss.
- Batch-size and learning-rate coupling: larger batches often need smaller learning rates.

### Interview-ready insights

- Derive quickly: dL/dz = sigmoid(z) - y, then dL/dw = x*(sigmoid(z)-y). Vectorized dw = X^T(p-y)/m.
- Explain why cross-entropy is used: it's the negative log-likelihood for Bernoulli labels and yields convex optimization for linear model.
- Explain L2 regularization effect: adds (lambda/2m)||w||^2 to cost and (lambda/m)w to gradient; corresponds to Gaussian prior on w.
- Show numerically-stable loss: L = max(0,z) - z*y + log(1+exp(-abs(z))). Mention why this avoids overflow.
- Distinguish losses: log-loss (cross-entropy) produces calibrated probabilities, unlike hinge loss; but hinge loss (SVM) focuses on margin.

### Practice exercises

1. Implement cross-entropy and gradient from scratch and verify gradient numerically:
    - Use finite differences for a random w to compare analytic dw to numerical derivative for a single parameter.
2. Implement training with and without L2 regularization on TF‑IDF features:
    - Compare validation F1 and coefficient norms ||w|| across lambdas {0, 1e-4, 1e-3, 1e-2}.
3. Stability experiment:
    - Create logits z with values ±100, compute naive log-loss via -[y*log(sigmoid(z)) + (1-y)log(1-sigmoid(z))] and compare to stable formula; show numerical breakdown.
4. Minibatch behavior:
    - Train with batch sizes {1, 32, full} keeping total gradient evaluations similar; plot training loss vs wall-clock iterations and compare generalization.

Hints:

- For numeric gradient check use eps=1e-6 and compare relative error.
- For stability test, use y=1 with z=100; naive p = sigmoid(100) ≈ 1.0 -> log(1-p) underflows; stable formula avoids that.

---

## Logistic regression - gradient

### Definition

The gradient for logistic regression is the vector of partial derivatives of the loss (binary cross‑entropy, optionally plus regularization) with respect to the model parameters (weights w and bias b). It tells you how to change each parameter to most rapidly reduce the loss.

### Intuition: what the gradient means and why it matters

- Each training example produces a prediction probability p = sigmoid(z) where z = w^T x + b. The scalar error term (p − y) measures how much the model “over‑predicted” (positive) or “under‑predicted” (negative) the true label.
- The gradient for w is a weighted sum of input feature vectors x, where each x is scaled by the scalar residual (p − y). Thus the gradient moves weights in directions that reduce systematic residuals across the dataset.
- The bias gradient is the average residual; it shifts the entire decision threshold.
- Practically, the gradient is the engine of training: optimizers (SGD, Adam) use it to update parameters. Understanding its form explains convergence speed, why examples near the decision boundary matter most, and how regularization modifies updates.

### Mathematical derivation

Model and loss:

```
z(i) = w^T x(i) + b
p(i) = sigmoid(z(i)) = 1 / (1 + exp(-z(i)))
L(i) = -[ y(i) * log(p(i)) + (1 - y(i)) * log(1 - p(i)) ]
J(w,b) = (1/m) * sum_{i=1..m} L(i)
```

Key derivative (single example):

- derivative of loss w.r.t. logit z:

```
dL/dz = p - y
```

- derivative w.r.t. weight vector w (single example):

```
dL/dw = (p - y) * x
```

- derivative w.r.t. bias b (single example):

```
dL/db = (p - y)
```

Vectorized gradient over m examples:

```
z = X.dot(w) + b         # shape (m,)
p = sigmoid(z)           # shape (m,)
error = p - y            # shape (m,)

dw = (1/m) * X.T.dot(error)   # shape (d,)
db = (1/m) * sum(error)       # scalar
```

With L2 regularization (λ):

```
J_reg = J + (λ / (2*m)) * ||w||^2
dw_reg = dw + (λ / m) * w
db_reg = db
```

Numerically stable forms:

- compute loss per example using:

```
loss_i = max(0, z) - z*y + log(1 + exp(-abs(z)))
```

- compute p with a stable sigmoid implementation or compute gradients using logits and stable functions in framework loss APIs (BCEWithLogitsLoss / BinaryCrossentropy(from_logits=True)).

### Practical code: analytic gradient, minibatches, and gradient check

NumPy: gradient computation + SGD update

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_gradients(X, y, w, b, lam=0.0):
    # X: (m,d), y: (m,), w: (d,), b: scalar
    m = X.shape[0]
    z = X.dot(w) + b
    p = sigmoid(z)
    error = p - y                       # (m,)
    dw = (X.T.dot(error)) / m           # (d,)
    db = np.sum(error) / m              # scalar
    if lam > 0:
        dw = dw + (lam / m) * w
    return dw, db

# SGD update
def sgd_step(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b

```

PyTorch: using logits and BCEWithLogitsLoss (preferred for stability)

```python
import torch
import torch.nn as nn

# linear layer outputs logits; use BCEWithLogitsLoss which combines sigmoid + stable loss
model = nn.Linear(d, 1)   # produces logits
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.0) # weight_decay = L2

# training step (single batch)
Xb = torch.tensor(X_batch, dtype=torch.float32)
yb = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1)
logits = model(Xb)                      # shape (B,1)
loss = loss_fn(logits, yb)              # computes stable loss
loss.backward()                         # computes gradients dLoss/dw in model.parameters()
opt.step(); opt.zero_grad()

```

Numeric gradient check (finite differences) — compare analytic dw to numeric approx

```python
def numeric_grad_check(X, y, w, b, lam=0.0, eps=1e-6):
    d = w.size
    _, dw_analytic, db_analytic = None, *compute_gradients(X, y, w, b, lam)
    num_dw = np.zeros_like(w)
    for j in range(d):
        w_pos = w.copy(); w_pos[j] += eps
        w_neg = w.copy(); w_neg[j] -= eps
        loss_pos = logistic_loss(X, y, w_pos, b, lam)
        loss_neg = logistic_loss(X, y, w_neg, b, lam)
        num_dw[j] = (loss_pos - loss_neg) / (2*eps)
    # numeric db
    b_pos = b + eps; b_neg = b - eps
    db_num = (logistic_loss(X, y, w, b_pos, lam) - logistic_loss(X, y, w, b_neg, lam)) / (2*eps)
    return num_dw, db_num

```

Note: implement logistic_loss with numerically stable formula.

### Visualization & geometric intuition for gradients

- Example influence: plot (p − y) distribution across training examples. Points with p near 0.5 have larger gradients and drive learning more; points with p ≈ y (confident correct) contribute near-zero gradient.
- Gradient direction: for a positive residual (p > y), dw is X^T * positive residuals, so features present in those examples push weights opposite their direction (reduce their positive logit contribution).
- Gradient magnitude over time: plot ||dw|| (L2 norm of gradient) per epoch — it usually decreases as model converges. Large spikes indicate noisy updates or too-large learning rate.
- Example plots:
    - Histogram of residuals (p − y) at epoch t.
    - Heatmap of average contribution per feature: mean_over_examples[(p − y) * x_j].
    - Loss contour (2D toy): show gradient vector at current w pointing downhill on contour lines.

Visual recipe:

- Reduce features to 2D (PCA) and train logistic in 2D; show decision boundary, color points by (p − y), draw arrows proportional to gradient contribution of each data point projected back to 2D.

### Common pitfalls & practical tips

- Pitfall: forgetting to average gradient over the batch (division by batch size). Tip: always divide by batch size m (or let framework do it).
- Pitfall: applying regularization to bias b. Tip: do not regularize bias; apply L2 only to w.
- Pitfall: numerical instability computing sigmoid on large |z|. Tip: use logits + framework stable loss or implement stable sigmoid (e.g., exp clamping or use np.where to compute).
- Pitfall: using too-large learning rate causing gradient explosion or oscillation. Tip: monitor loss and ||dw||, reduce lr or use adaptive optimizer (Adam) with appropriate lr.
- Pitfall: not scaling dense features. Tip: standardize dense features so gradient magnitudes are comparable across dimensions.
- Pitfall: sparse high‑dim data with tiny gradients in many coordinates — use appropriate solvers or sparse-aware updates (scikit-learn or sparse SGD).
- Tip: gradient clipping can stabilize training when using noisy mini‑batches.
- Tip: for very imbalanced labels, scale loss via class weights or use weighted gradients: multiply per-example loss/gradient by weight.

### Interview-ready insights

- Derive quickly: dL/dz = sigmoid(z) − y; then dL/dw = x * (sigmoid(z) − y); vectorized dw = X^T (p − y) / m.
- Explain why examples near the decision boundary dominate learning: sigmoid derivative is largest around z ≈ 0, so (p − y) magnitude is largest for uncertain predictions.
- Explain effect of L2 on gradient: adds (λ/m) * w to dw which pulls weights toward zero (weight decay).
- Explain numeric stability: prefer logits + BCEWithLogitsLoss/BinaryCrossentropy(from_logits=True) to avoid overflow in exp.
- Explain per-example weighted gradient for class imbalance: multiply per-example loss (and thus p − y) by class weight.

### Practice exercises

Compute and plot residuals

- Task: On a TF‑IDF representation and trained logistic model, compute residuals r_i = p_i − y_i. Plot histogram of r_i for train and validation. Interpret shape and changes over epochs.
- Hint: Early epochs have wide residual spread; convergence tightens distribution around 0.

Finite-difference gradient check

- Task: Implement numeric_grad_check for a small synthetic dataset (d ≤ 5). Compare relative error between analytic and numeric dw; ensure error < 1e-6.
- Hint: use small eps (1e-6) and compute relative error |num − analytic| / (|num| + |analytic| + 1e-12).

Mini-batch gradient dynamics

- Task: Train logistic with batch sizes {1, 32, full} and plot ||dw|| versus iteration for first 2000 updates. Explain differences in noise and convergence speed.
- Hint: compute dw per batch using compute_gradients and record its norm.

Regularization effect on gradient

- Task: Train same model with λ = 0 and λ = 1e-2. Plot L2 norm of w and average gradient norm over epochs. Show that regularized gradient has extra (λ/m)w pulling weights down.

Per-feature gradient contribution

- Task: For a linear TF‑IDF model, compute per-feature contribution to the gradient: g_j = (1/m) * sum_i (p_i − y_i) * x_{i,j}. Display top 20 features by absolute g_j; inspect whether they match top coefficient magnitudes.
- Hint: g_j is exactly dw_j. Compare sorted lists.

---