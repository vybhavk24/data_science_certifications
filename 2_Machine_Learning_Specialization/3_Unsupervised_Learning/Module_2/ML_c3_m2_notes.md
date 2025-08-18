# ML_c3_m2

## Recommender Systems: Making Recommendations

### 1. Prerequisites

- Linear algebra: vectors, dot products, matrix factorization
- Basic statistics: mean, variance
- Similarity measures: cosine similarity, Pearson correlation
- Python & NumPy (optionally pandas, scikit-learn)

### 2. What Is a Recommender System?

A recommender system suggests items (products, movies, articles) that a user is likely to engage with. Broadly:

- **Content-Based Filtering**: recommends items similar to those a user liked, based on item features.
- **Collaborative Filtering (CF)**: leverages the tastes of many users—“you and I rate similarly, so you might like what I like.”
- **Hybrid Models**: combine content and collaborative signals for better coverage.

Real-world uses:

- E-commerce (“Customers who bought X also bought Y”)
- Video platforms (“Because you watched A, here’s B”)
- News personalization

### 3. Content-Based Filtering

### 3.1 Intuition

Build a profile of each user based on features of items they’ve consumed. Recommend new items whose feature vectors align with the user’s profile.

### 3.2 Math & Code

1. Represent items with feature vectors
    
    ```python
    # Example: TF-IDF features of movie plots or one-hot genres
    item_features = np.array([
        [1,0,1,0],  # genres for movie1
        [0,1,0,1],  # movie2
        …
    ])  # shape (n_items, n_features)
    ```
    
2. Build user profile as weighted average of liked-item features
    
    ```python
    # R[u,i] is 1 if user u liked item i, else 0
    user_profile[u] = (R[u] @ item_features) / max(1, R[u].sum())
    ```
    
3. Score new items by cosine similarity between `user_profile[u]` and each item’s features
    
    ```python
    def cosine_sim(a, b):
        return (a @ b) / (np.linalg.norm(a)*np.linalg.norm(b))
    
    scores = [cosine_sim(user_profile[u], item_features[i])
              for i in range(n_items)]
    recs = np.argsort(scores)[-k:]  # top-k recommendations
    ```
    

### 3.3 Limitations & Tips

- Cold-start: new items with no ratings still need well-engineered features.
- Over-specialization: users see only very similar items. Mix in exploration.

### 4. Collaborative Filtering

### 4.1 Memory-Based CF

### User-Based

- **Intuition**: find users “nearest” to target user by similarity of their rating vectors; aggregate their ratings.
- **Formula**
    
    ```python
    # R: user × item rating matrix, with 0 for unknown
    # sim(u, v) = Pearson correlation between R[u] and R[v]
    pred_rating = sum( sim(u,v) * R[v,i] for v in neighbors ) \
                  / sum( |sim(u,v)| for v in neighbors )
    ```
    

### Item-Based

- **Intuition**: find items similar to a target item by co-ratings, then predict based on a user’s other ratings.
- **Formula**
    
    ```python
    # sim(i, j) = cosine_similarity(R[:,i], R[:,j])
    pred_rating = sum( sim(i,j) * R[u,j] for j in similar_items ) \
                  / sum( |sim(i,j)| for j in similar_items )
    ```
    

### 4.2 Model-Based CF: Matrix Factorization

### Intuition

Map users and items into the same low-dimensional latent space so that the dot-product predicts ratings.

### Cost Function

```python
# X: (n_users, n_items), with zeros for unknowns
# P: (n_users, k) user factors
# Q: (n_items, k) item factors
# λ: regularization strength

J = sum_{u,i: X[u,i]>0} ( X[u,i] - P[u]·Q[i] )**2 \
    + λ * ( ||P||_F^2 + ||Q||_F^2 )
```

### Gradient Descent Updates

```python
for u,i in observed_ratings:
    err = X[u,i] - P[u].dot(Q[i])
    P[u] += α * ( err * Q[i] - λ * P[u] )
    Q[i] += α * ( err * P[u] - λ * Q[i] )
```

### Python Snippet

```python
def matrix_factorization(X, k, alpha, lam, iters):
    n_users, n_items = X.shape
    P = np.random.normal(scale=0.1, size=(n_users, k))
    Q = np.random.normal(scale=0.1, size=(n_items, k))
    rows, cols = X.nonzero()
    for _ in range(iters):
        for u,i in zip(rows, cols):
            err = X[u,i] - P[u].dot(Q[i])
            P[u] += alpha * (err * Q[i] - lam * P[u])
            Q[i] += alpha * (err * P[u] - lam * Q[i])
    return P, Q
```

### 4.3 Advanced Variants

- **Alternating Least Squares (ALS)**: solve for P fixing Q, then Q fixing P—fast via linear algebra.
- **Bias terms**: incorporate global mean, user bias `b_u`, and item bias `b_i`.
- **Implicit feedback**: treat views/clicks as positive signals; use confidence weighting.

### 5. Real-World Workflow

1. **Data ingestion**: collect user–item interactions (ratings, clicks, purchases).
2. **Preprocessing**: filter inactive users/items, split train/test.
3. **Baseline**: start with popularity (top-N most popular items).
4. **Implement memory-based CF**: evaluate RMSE or ranking metrics (precision@k, recall@k).
5. **Build matrix factorization**: tune `k`, `λ`, learning rate.
6. **Hybrid & context**: add content features or time/context signals.
7. **Deploy**: precompute top-N lists offline; store factors in feature store; serve via API.
8. **Monitor**: track click-through rate (CTR), conversion lift, A/B test new models.

### 6. Evaluation Metrics

- **Rating prediction**: RMSE, MAE
- **Top-N recommendation**:
    - Precision@k, Recall@k
    - Mean Average Precision (MAP)
    - Normalized Discounted Cumulative Gain (NDCG@k)

```python
def precision_at_k(preds, ground_truth, k):
    hits = len(set(preds[:k]) & set(ground_truth))
    return hits / k
```

### 7. Geometric Insight

- User and item factors live in a k-dimensional “taste space.”
- A user’s coordinates reflect preference axes (e.g., “action vs. drama,” “novelty vs. popularity”).
- Dot-product = projection of user onto item direction = predicted affinity.

### 8. Practice Problems

1. **MovieLens 100K**
    - Load dataset (`u.data`), build rating matrix.
    - Implement item-based CF with cosine similarity; evaluate RMSE.
2. **Matrix Factorization**
    - Factorize MovieLens with k=10,20; tune `λ` by grid search.
    - Compute precision@10 for top-10 movie recommendations per user.
3. **Hybrid Model**
    - Augment CF with item content: genre one-hot vectors.
    - At prediction time, blend CF score and content-based score (weighted sum); optimize weights.
4. **Cold-Start Experiment**
    - Simulate new users: train on a subset of ratings, hold out first 5 ratings, recommend from cold-start using content only.
    - Measure hit rate@5.

---

## Using Per-Item Features in Recommender Systems

Below is a comprehensive outline of every sub-topic—from the fundamentals to cutting-edge research—you’ll want to master when incorporating per-item features into recommendation engines. Formulas (when shown) are in plain code blocks so you can copy them directly into Notion.

### 1. Motivation & Overview

- Why item features matter
    - Cold-start for new items
    - Explainability via human-readable attributes
    - Reducing overfitting with shared parameters
- Comparison to pure collaborative filtering

### 2. Item Feature Representation

- **Raw vs. engineered features**
- **Numerical features** (price, rating, duration)
- **Categorical features** (genres, categories, tags)
- **Text features** (TF-IDF vector, pretrained embeddings)
- **Image features** (CNN embeddings from product photos)
- **Temporal/event features** (release date, recency)
- **Multi-modal fusion** (concatenation, late fusion)

### 3. Content-Based Filtering (Pure Feature Model)

- Building a user profile as average of consumed item features
- Scoring by similarity
    
    ```python
    # simple cosine score
    score = (user_profile @ item_features[i]) / \
            (||user_profile|| * ||item_features[i]||)
    ```
    
- Pros & cons, cold-start behavior

### 4. Hybrid Architectures

1. **Weighted Hybrid**
    - Combine CF score and feature-based score via a weighted sum
2. **Switching Hybrid**
    - Choose CF or content branch based on data availability
3. **Feature Augmentation**
    - Treat CF latent factors as “features” in a second-stage model
4. **Ensemble & Stacking**
    - Blend multiple models (GBDT, FM, NN)

### 5. Feature-Based Matrix Factorization

- Map item features `x_i` → latent vector `q_i` via weight matrix `W`
- Prediction
    
    ```python
    q_i = W.T.dot(x_i)                      # item latent
    r_hat = p_u.dot(q_i)                    # p_u is user latent
    ```
    
- Loss function
    
    ```python
    loss = (r_ui - p_u.dot(W.T.dot(x_i)))**2 \
         + lambda*(||p_u||^2 + ||W||^2)
    ```
    
- SGD updates for `p_u` and `W`

### 6. Factorization Machines (FMs)

- Model all 1st- and 2nd-order feature interactions
- Prediction formula
    
    ```python
    y_hat = w0 \
          + sum_i(w_i * x_i) \
          + sum_i(sum_j>i( (v_i·v_j) * x_i * x_j ))
    ```
    
- Training via SGD or ALS
- Cold-start and sparse data handling

### 7. Field-Aware Factorization Machines (FFMs)

- Extend FMs by learning field-specific embeddings
- Improved accuracy on categorical data
- Typical use in ad-tech and click-prediction

### 8. Deep Hybrid Models

- **Wide & Deep** (LR + DNN)
- **DeepFM** (FM + DNN share embeddings)
- **Neural Factorization Machines** (higher-order interactions)
- **Autoencoder-based** content reconstructions
- **Graph Neural Networks** over item–feature graphs

### 9. Feature Engineering & Interaction Modeling

- Handcrafting cross features vs. automatic interactions
- Embedding dimension selection
- Normalization & binning of numerical features
- Dealing with high cardinality categories (hashing, embeddings)

### 10. Cold-Start & Zero-Shot Recommendations

- Pure feature-only models for brand-new items
- Meta-learning approaches to generalize to unseen feature combinations

### 11. Interpretability & Explainability

- Feature importance scores (e.g., SHAP, integrated gradients)
- Local explanations: why did we recommend item X?
- Global insight: which features drive latent dimensions?

### 12. Scalability & Performance

- Approximate nearest-neighbor search on feature-augmented vectors (FAISS, Annoy)
- Mini-batch and asynchronous SGD for large feature matrices
- Online vs. batch feature processing

### 13. Production & Deployment Considerations

- Designing a feature store for item attributes
- Real-time inference pipelines
- Offline retraining cadence & model versioning
- Monitoring drift in feature distributions

### 14. Evaluation & Experimentation

- Offline metrics: RMSE, precision@K, NDCG@K for feature models
- Ablation studies: measuring impact of each feature group
- Online A/B testing for hybrid feature-driven recommenders

---

## Collaborative Filtering Algorithms

### 1. Overview & Motivation

- Collaborative filtering (CF) predicts a user’s interest in items by leveraging patterns of past user–item interactions.
- Core idea: similar users rate similar items similarly, or items co-rated by users can recommend each other.
- Two main categories:
    1. Memory-based CF
    2. Model-based CF

### 2. Memory-Based CF

### 2.1 User-Based Collaborative Filtering

- Find users whose rating patterns correlate with the target user.
- Aggregate their ratings to predict the target user’s unknown ratings.

```python
# similarity(u,v): precomputed cosine or Pearson between user u and user v
predicted_rating = sum(similarity(u, v) * R[v, i] for v in neighbors) \
                   / sum(abs(similarity(u, v)) for v in neighbors)
```

### 2.2 Item-Based Collaborative Filtering

- Compute similarity between items based on all users’ ratings.
- Predict a user’s rating for item i by looking at her ratings on items similar to i.

```python
# similarity(i,j): precomputed cosine or Pearson between item i and j
predicted_rating = sum(similarity(i, j) * R[u, j] for j in similar_items) \
                   / sum(abs(similarity(i, j)) for j in similar_items)
```

### 2.3 Similarity Measures

- Cosine similarity
    
    ```python
    def cosine_sim(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ```
    
- Pearson correlation (mean-centered)
    
    ```python
    def pearson_sim(a, b):
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        return (a_centered @ b_centered) \
               / (np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    ```
    

### 2.4 Neighborhood Selection

- k-Nearest Neighbors: pick top-k most similar users/items.
- Similarity threshold: only use neighbors with sim > θ.
- Tradeoff: larger k → smoother predictions but more noise.

### 3. Model-Based CF

### 3.1 Matrix Factorization (MF)

- Factorize the user–item rating matrix (R) into user factors (P) and item factors (Q).
- Low-dimensional latent vectors capture taste dimensions.

```python
# R[u, i] is observed if >0
loss = sum((R[u, i] - P[u].dot(Q[i]))**2
           for u, i in observed) \
       + lambda_ * (np.linalg.norm(P)**2 + np.linalg.norm(Q)**2)
```

- Stochastic gradient descent updates:

```python
for u, i in observed:
    err = R[u, i] - P[u].dot(Q[i])
    P[u] += alpha * (err * Q[i] - lambda_ * P[u])
    Q[i] += alpha * (err * P[u] - lambda_ * Q[i])
```

### 3.2 Alternating Least Squares (ALS)

- Fix (Q), solve for (P) via regularized least squares; then fix (P), solve for (Q).

```python
# P = argmin_P ||R - P Q^T||^2 + lambda||P||^2  (closed form via linear algebra)
# Q = argmin_Q ||R - P Q^T||^2 + lambda||Q||^2
```

### 3.3 Bias-Aware MF

- Account for global mean (\mu), user bias (b_u), item bias (b_i):

```python
pred = mu + b_u + b_i + P[u].dot(Q[i])
loss = sum((R[u,i] - pred)**2 for u,i in observed) \
     + lambda*(||P||^2 + ||Q||^2 + ||b_u||^2 + ||b_i||^2)
```

### 3.4 Implicit Feedback Models

- Treat view/clicks as positive signals; introduce confidence (c_{ui}).
- Optimize:

```python
loss = sum(c_ui * (p_u.dot(q_i) - 1)**2
           for u,i in all_pairs) \
     + lambda*(||P||^2 + ||Q||^2)
```

### 3.5 Probabilistic Matrix Factorization (PMF)

- Assume Gaussian priors on (P) and (Q).
- Optimize log-posterior via SGD or variational methods.

### 4. Advanced Variants

- SVD++: integrates implicit feedback into MF.
- TimeSVD++: models temporal dynamics of user biases and item biases.
- Neural CF: learn non-linear interactions with neural networks (e.g., Neural Collaborative Filtering).
- Graph-based CF: represent users/items as nodes and perform graph convolutions or random walks.

### 5. Practical Considerations

- Data sparsity: apply shrinkage, smoothing, or side-information to alleviate.
- Cold-start: combine CF with content features (hybrid approach).
- Scalability: use approximate nearest neighbors (e.g., FAISS) for large-scale similarity.
- Hyperparameter tuning: latent dimension (k), regularization (\lambda), learning rate (\alpha).
- Evaluation: RMSE/MAE for rating prediction; precision@K, recall@K, NDCG@K for ranking.

### 6. Code Walkthrough

1. Load user–item matrix into a sparse structure.
2. Precompute similarity matrix (user×user or item×item).
3. Implement prediction functions for user- and item-based CF.
4. Build MF training loop with SGD or ALS.
5. Evaluate on held-out test set.

---

## Collaborative Filtering with Binary Labels (favs, likes, clicks)

Below is a complete roadmap—from first principles to state-of-the-art—covering every angle of building recommender systems when feedback is binary (1 = favourite/like/click, 0 = unknown). All formulas are in plain code blocks for easy copy-paste into Notion.

### 1. Nature of Binary Feedback

- Definition
    - Positive signal only: user explicitly “liked” or “clicked.”
    - Absence of signal means unknown preference, not true negative.
- Challenges
    - One-class problem: only positives, no explicit negatives.
    - High sparsity: most entries are 0 (unknown).
    - Ambiguity: non-interaction could be “didn’t see” or “not interested.”

### 2. Data Representation

- User–Item Binary Interaction Matrix
    
    ```python
    R = np.zeros((n_users, n_items))  # default unknown
    R[user, item] = 1                 # user clicked/liked/faved
    ```
    
- Confidence or weight matrix (optional)
    
    ```python
    # map binary signal to confidence c_ui
    C = 1 + alpha * R
    ```
    

### 3. Baseline Popularity Models

- Global popularity ranking
    
    ```python
    popularity = R.sum(axis=0)      # count of positives per item
    top_items = np.argsort(popularity)[::-1]
    ```
    
- Per-user baseline: recommend global top-N to everyone

### 4. Memory-Based CF with Binary Data

### 4.1 Similarity Measures

- Jaccard similarity
    
    ```python
    def jaccard(u, v):
        intersect = np.logical_and(R[u], R[v]).sum()
        union     = np.logical_or(R[u], R[v]).sum()
        return intersect / max(1, union)
    ```
    
- Cosine on binary vectors
    
    ```python
    def cosine(u, v):
        return (R[u] @ R[v]) / (np.linalg.norm(R[u]) * np.linalg.norm(R[v]))
    ```
    

### 4.2 User-Based Prediction

```python
neighbors = top_k_similar_users(u, similarity)
pred_score = sum(sim(u, v) * R[v, i] for v in neighbors) \
           / sum(abs(sim(u, v)) for v in neighbors)
```

### 4.3 Item-Based Prediction

```python
sims = item_item_similarity_matrix(R)
pred_score = sum(sims[i, j] * R[u, j] for j in top_k_items(i)) \
           / sum(abs(sims[i, j]) for j in top_k_items(i))
```

### 5. Model-Based CF for Implicit Feedback

### 5.1 Weighted Matrix Factorization (Hu et al.)

- Prediction
    
    ```python
    r_hat = p_u.dot(q_i)
    ```
    
- Loss with confidence
    
    ```python
    loss = sum(C[u,i] * (R[u,i] - p_u.dot(q_i))**2
               for u,i in all_pairs) \
         + lambda * (||P||^2 + ||Q||^2)
    ```
    
- SGD updates
    
    ```python
    for u,i in observed:
        c = C[u,i]
        err = R[u,i] - P[u].dot(Q[i])
        P[u] += alpha * (c * err * Q[i] - lambda * P[u])
        Q[i] += alpha * (c * err * P[u] - lambda * Q[i])
    ```
    

### 5.2 Alternating Least Squares (ALS)

- Closed-form update for user factors
    
    ```python
    P[u] = inv(Q.T * C_u * Q + lambda*I) * Q.T * C_u * r_u
    ```
    
- Similarly update each item factor

### 6. Pairwise Ranking (BPR)

- Optimizes that observed positives rank above unobserved
- Sample triple (u, i_pos, i_neg) where R[u,i_pos]=1, R[u,i_neg]=0
- Prediction difference
    
    ```python
    x_uij = p_u.dot(q_pos - q_neg)
    ```
    
- BPR loss and gradients
    
    ```python
    loss = -log(sigmoid(x_uij)) + lambda*(||p_u||^2 + ||q_pos||^2 + ||q_neg||^2)
    # gradient steps follow from ∂loss/∂parameters
    ```
    

### 7. Pointwise Logistic Models

- Treat recommendation as binary classification
- Prediction with logistic sigmoid
    
    ```python
    r_hat = sigmoid(p_u.dot(q_i) + b_u + b_i)
    ```
    
- Binary cross-entropy loss
    
    ```python
    loss = -sum(R[u,i]*log(r_hat) + (1-R[u,i])*log(1-r_hat))
         + lambda*(||P||^2 + ||Q||^2 + ||b||^2)
    ```
    

### 8. Negative Sampling Strategies

- Uniform sampling of i_neg
- Popularity-based sampling to choose “hard negatives”
- Dynamic sampling: pick negatives with high current score

### 9. Evaluation Metrics for Binary Feedback

- Ranking-oriented metrics (no RMSE)
    - Precision@K
    - Recall@K
    - AUC (area under ROC curve)
    - MAP (mean average precision)
    - NDCG@K

```python
def precision_at_k(preds, actuals, k):
    return len(set(preds[:k]) & set(actuals)) / k
```

### 10. Hybridizing Binary CF with Item Features

- Use feature-parameterized Q from “feature matrix factorization”
- Jointly optimize implicit CF loss with feature mapping W

```python
q_i = W.T.dot(x_i)
r_hat = p_u.dot(q_i)
# use confidence-weighted loss as in section 5.1
```

### 11. Neural Collaborative Filtering

- Replace dot product with neural network
- Input: one-hot user vector, one-hot item vector (or embeddings)
- Output: probability of click/like
- Loss: binary cross-entropy

### 12. Scalability & Production

- Approximate nearest neighbors for real-time retrieval (FAISS)
- Batch vs online retraining cadence
- Incremental updates for new interactions
- Feature store for user and item metadata

---

## Mean Normalization

### 1. Why Mean Normalize Ratings

Mean normalization centers the rating data by removing average biases.

- It corrects for users who give consistently high or low ratings.
- It adjusts for items that generally receive high or low scores.
- Centered data often converges faster in matrix factorization.

### 2. User- and Item-Wise Centering

### 2.1 User Mean Normalization

Subtract each user’s average rating from their ratings:

```python
# R is user×item matrix with zeros for unknowns
user_means = np.true_divide(R.sum(axis=1), (R!=0).sum(axis=1))
R_centered = R.copy()
for u in range(n_users):
    R_centered[u, R[u]!=0] -= user_means[u]
```

### 2.2 Item Mean Normalization

Subtract each item’s average rating from its ratings:

```python
item_means = np.true_divide(R.sum(axis=0), (R!=0).sum(axis=0))
R_centered = R.copy()
for i in range(n_items):
    R_centered[R[:,i]!=0, i] -= item_means[i]
```

### 2.3 Combined Baseline Correction

Remove both global, user, and item biases:

```python
# global mean
mu = R[R!=0].mean()

# user and item biases
b_u = user_means - mu
b_i = item_means - mu

# center each rating
R_centered = R.copy()
for u,i in observed:
    R_centered[u,i] = R[u,i] - mu - b_u[u] - b_i[i]
```

### 3. Integrating with Matrix Factorization

After centering, factorize the residual matrix:

```python
# minimize over P, Q
loss = sum((R_centered[u,i] - P[u].dot(Q[i]))**2
           for u,i in observed) \
     + lambda*(||P||^2 + ||Q||^2)
```

At prediction time, add biases back:

```python
r_hat = mu + b_u[u] + b_i[i] + P[u].dot(Q[i])
```

### 4. Code Snippet: Full Pipeline

```python
# compute global, user, item means
mu       = R[R!=0].mean()
user_means = np.true_divide(R.sum(1), (R!=0).sum(1))
item_means = np.true_divide(R.sum(0), (R!=0).sum(0))
b_u = user_means - mu
b_i = item_means - mu

# center ratings
R_centered = R.copy()
for u,i in zip(*R.nonzero()):
    R_centered[u,i] = R[u,i] - mu - b_u[u] - b_i[i]

# initialize factors
P = np.random.normal(0, 0.1, (n_users, k))
Q = np.random.normal(0, 0.1, (n_items, k))

# SGD on centered data
for _ in range(iters):
    for u,i in zip(*R_centered.nonzero()):
        err = R_centered[u,i] - P[u].dot(Q[i])
        P[u] += alpha * (err * Q[i] - lam * P[u])
        Q[i] += alpha * (err * P[u] - lam * Q[i])

# prediction adds biases back
def predict(u, i):
    return mu + b_u[u] + b_i[i] + P[u].dot(Q[i])
```

### 5. Benefits & Trade-Offs

- Speeds up convergence by centering data.
- Captures baseline effects explicitly.
- Requires computing and storing bias terms.
- May underfit if biases change over time or context.

### 6. Practical Tips

- Recompute means and biases on each data update.
- Clip centered values to handle outliers.
- Combine mean normalization with regularized bias terms in MF.
- For implicit feedback, use confidence-weighted centering.

---

## TensorFlow Implementation of Collaborative Filtering

### Overview

Build a Keras model with user and item embedding layers, combine them via dot product (plus optional biases), and train with mean-squared error (for ratings) or binary cross-entropy (for clicks/likes).

### 1. Prerequisites

- Python 3.7+
- TensorFlow 2.x (`pip install tensorflow`)
- NumPy, pandas (for data handling)

### 2. Data Preparation

1. Encode user and item IDs as consecutive integers
2. Create arrays:
    - `user_ids`: shape `(n_interactions,)`
    - `item_ids`: shape `(n_interactions,)`
    - `ratings` or `labels`: shape `(n_interactions,)`
3. Split into train and test sets

```python
import numpy as np
from sklearn.model_selection import train_test_split

# example raw data
user_ids_raw = [...]
item_ids_raw = [...]
ratings     = [...]      # for explicit feedback
labels      = [...]      # for binary feedback

# map to indices
user_map = {u: idx for idx, u in enumerate(set(user_ids_raw))}
item_map = {i: idx for idx, i in enumerate(set(item_ids_raw))}

user_ids = np.array([user_map[u] for u in user_ids_raw])
item_ids = np.array([item_map[i] for i in item_ids_raw])

# choose target
target = np.array(ratings)        # or np.array(labels)

# train/test split
u_train, u_test, i_train, i_test, y_train, y_test = train_test_split(
    user_ids, item_ids, target, test_size=0.2, random_state=42
)
```

### 3. Model Definition

### 3.1 Hyperparameters

```python
num_users    = len(user_map)
num_items    = len(item_map)
latent_dim   = 32        # dimensionality of embeddings
reg_lambda   = 1e-6      # regularization strength
```

### 3.2 Embedding Layers & Biases

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# inputs
user_input = layers.Input(shape=(), dtype=tf.int32, name='user_id')
item_input = layers.Input(shape=(), dtype=tf.int32, name='item_id')

# embeddings
user_emb = layers.Embedding(
    input_dim=num_users, output_dim=latent_dim,
    embeddings_regularizer=tf.keras.regularizers.l2(reg_lambda),
    name='user_embed'
)(user_input)

item_emb = layers.Embedding(
    input_dim=num_items, output_dim=latent_dim,
    embeddings_regularizer=tf.keras.regularizers.l2(reg_lambda),
    name='item_embed'
)(item_input)

# optional bias terms
user_bias = layers.Embedding(
    input_dim=num_users, output_dim=1,
    embeddings_regularizer=tf.keras.regularizers.l2(reg_lambda),
    name='user_bias'
)(user_input)

item_bias = layers.Embedding(
    input_dim=num_items, output_dim=1,
    embeddings_regularizer=tf.keras.regularizers.l2(reg_lambda),
    name='item_bias'
)(item_input)
```

### 3.3 Dot-Product & Prediction

```python
# flatten embeddings
user_vec = layers.Flatten()(user_emb)
item_vec = layers.Flatten()(item_emb)
u_bias   = layers.Flatten()(user_bias)
i_bias   = layers.Flatten()(item_bias)

# interaction score
dot = layers.Dot(axes=1)([user_vec, item_vec])

# final prediction: dot + biases
pred = layers.Add()([dot, u_bias, i_bias])
```

### 3.4 Build & Compile

```python
model = Model(inputs=[user_input, item_input], outputs=pred)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',                     # for explicit ratings
    # loss='binary_crossentropy',   # for clicks/likes
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

### 4. Training Loop

```python
history = model.fit(
    x=[u_train, i_train],
    y=y_train,
    validation_data=([u_test, i_test], y_test),
    batch_size=1024,
    epochs=20,
    verbose=2
)
```

### 5. Evaluation & Prediction

- **Explicit feedback**: report RMSE on test set
- **Implicit feedback**: convert outputs to probabilities with sigmoid, then evaluate AUC, precision@K

```python
# for binary case: wrap pred in sigmoid
sigmoid = tf.keras.activations.sigmoid
probs = sigmoid(model.predict([u_test, i_test]))

# compute AUC
auc = tf.keras.metrics.AUC()
auc.update_state(y_test, probs)
print('Test AUC:', auc.result().numpy())
```

- **Top-K recommendations** for a user:
    
    ```python
    user_idx = 10
    items = np.arange(num_items)
    user_vecs = np.full_like(items, user_idx)
    
    scores = model.predict([user_vecs, items]).flatten()
    top_k = np.argsort(scores)[-10:][::-1]
    ```
    

### 6. Under the Hood: Loss Formula

```python
# prediction
r_hat = dot + u_bias + i_bias

# MSE loss for explicit ratings
loss = mean((r_true - r_hat)**2)
       + reg_lambda * (||user_emb||^2 + ||item_emb||^2)

# binary cross-entropy for clicks/likes
loss = -mean(r_true * log(sigmoid(r_hat))
             + (1-r_true)*log(1-sigmoid(r_hat)))
       + reg_lambda * (||user_emb||^2 + ||item_emb||^2)
```

### 7. Practical Tips

- Initialize embeddings with small random values (Keras does this by default).
- Monitor overfitting—add dropout or increase `reg_lambda` if needed.
- Adjust `latent_dim` based on dataset size (larger means more capacity).
- Use learning-rate schedules (e.g., `ReduceLROnPlateau`).
- For large datasets, build a `tf.data.Dataset` pipeline for performance.

---

## Finding Related Items

### 1. Motivation & Use Cases

Finding related items lets you power “Customers also viewed,” “Similar products,” or “You might also like” widgets. It boosts engagement, cross-sells products, and improves discovery by surfacing items with high affinity to a given item.

### 2. Similarity Measures

- Cosine similarity
- Pearson correlation
- Jaccard coefficient (for binary interactions)
- Euclidean distance (after normalization)
- Mahalanobis distance (for correlated features)

```python
def cosine_sim(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

```python
def jaccard_sim(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(1, union)
```

### 3. Content-Based Similarity

1. Represent items by feature vectors (e.g., TF-IDF for text, one-hot for categories).
2. Compute pairwise similarity on these vectors.
3. For each item, precompute the top-K similar items.

```python
from sklearn.metrics.pairwise import cosine_similarity

# item_features: shape (n_items, n_features)
sim_matrix = cosine_similarity(item_features)
top_k = np.argsort(sim_matrix[i])[::-1][1:k+1]  # skip self
```

### 4. Item-Item Collaborative Filtering

1. Build a user–item interaction matrix `R` (ratings, clicks, purchases).
2. Compute similarity between item columns of `R`.
3. For a query item `i`, recommend items with highest similarity.

```python
# R: shape (n_users, n_items)
item_sim = cosine_similarity(R.T)
related = np.argsort(item_sim[i])[::-1][1:k+1]
```

1. Optionally weight similarities by item popularity or inverse user frequency to reduce popularity bias.

### 5. Embedding-Based Retrieval

- Learn item embeddings via matrix factorization or neural networks.
- Similarity in embedding space often captures latent co-occurrence patterns better than raw features.

```python
# Q: item latent matrix, shape (n_items, k)
sim_matrix = cosine_similarity(Q)
related = np.argsort(sim_matrix[i])[::-1][1:k+1]
```

- For neural models, you can extract the final hidden representation for each item.

### 6. Graph-Based Methods

- Build an undirected graph where nodes are items and edges connect co-viewed or co-purchased pairs.
- Edge weight = frequency of co-occurrence or normalized lift.
- Use graph algorithms:
    - Random walk with restart to score neighbors
    - Personalized PageRank seeded at item `i`
    - Graph Neural Networks to learn richer item embeddings

### 7. Scalability & Approximate Nearest Neighbors

When `n_items` is large, computing full pairwise sims is expensive:

- Locality Sensitive Hashing (LSH) for cosine or Jaccard
- FAISS (Facebook), Annoy (Spotify), HNSW for vector indexing
- Precompute index and run approximate top-K queries at serving time

```python
import faiss

index = faiss.IndexFlatIP(k)           # inner product (for cosine if vectors are normalized)
index.add(Q)                           # Q is item embeddings
_, related = index.search(Q[i:i+1], k+1)
related = related[0][1:]               # drop the query itself
```

### 8. Evaluation Metrics

- Precision@K, Recall@K
- Mean Average Precision (MAP@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Hit Rate or Coverage for catalogs

```python
def precision_at_k(preds, actuals, k):
    return len(set(preds[:k]) & set(actuals)) / k
```

### 9. Practical Tips

- Normalize feature vectors (unit length) before cosine similarity.
- Apply shrinkage to similarity scores to down-weight low-support pairs:
    
    ```python
    sim_shrunk = (n_common / (n_common + beta)) * raw_sim
    ```
    
- Filter out overly popular items to reduce bias.
- Cache precomputed top-K lists offline; update regularly as data drifts.
- Blend multiple signals (content, CF, graph) via weighted sum or rank-level ensemble.

---

## Collaborative Filtering vs Content-Based Filtering

### 1. Prerequisites

- Vector representations of items and users
- Similarity measures (cosine, Pearson)
- Basic matrix operations and averaging
- Python & NumPy for code examples

### 2. High-Level Intuition

Content-based filtering recommends new items by matching item attributes to a user’s known preferences. It’s like saying, “You enjoyed mystery novels, so here’s another mystery.”

Collaborative filtering relies on user behavior patterns—users who agreed in the past will agree in the future. It’s like, “People with similar tastes to yours also liked this book.”

### 3. Content-Based Filtering

### 3.1 User Profile Construction

Combine features of items a user has liked into a single vector:

```python
# X: item_features matrix (n_items × f)
# R[u]: binary or weighted preferences of user u over items
user_profile = R[u].dot(X) / max(1, R[u].sum())
```

### 3.2 Scoring New Items

Compute similarity between the user profile and each candidate item:

```python
def cosine_sim(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_sim(user_profile, X[i]) for i in range(n_items)]
```

### 3.3 Pros & Cons

Pros

- No need for other users’ data
- Explains recommendations via item features
- Cold-start for users (with initial likes)

Cons

- Limited discovery beyond known features
- Requires good feature engineering
- Over-specialization (little novelty)

### 4. Collaborative Filtering

### 4.1 Memory-Based CF

### User-Based

Predict user u’s rating of item i by aggregating ratings of similar users:

```python
# sim_uv: precomputed similarity between user u and v
pred = sum(sim_uv * R[v, i] for v in neighbors) \
     / sum(abs(sim_uv) for v in neighbors)
```

### Item-Based

Predict by looking at similar items:

```python
# sim_ij: similarity between item i and j
pred = sum(sim_ij * R[u, j] for j in top_k_items(i)) \
     / sum(abs(sim_ij) for j in top_k_items(i))
```

### 4.2 Model-Based CF (Matrix Factorization)

Learn low-dimensional embeddings `P` (users) and `Q` (items) so that `P[u]·Q[i]` approximates `R[u,i]`:

```python
# loss over observed ratings
loss = sum((R[u,i] - P[u].dot(Q[i]))**2 for (u,i) in observed) \
     + lambda*(||P||^2 + ||Q||^2)
```

Use SGD to update `P[u]` and `Q[i]`.

### 4.3 Pros & Cons

Pros

- Captures patterns beyond explicit features
- Can recommend unexpected items
- Scales to large datasets with MF or ALS

Cons

- Cold-start problem for new users/items
- Harder to explain than content models
- Suffers when data is extremely sparse

### 5. Side-by-Side Comparison

| Aspect | Content-Based | Collaborative |
| --- | --- | --- |
| Data Needed | Item features + user history | User–item interaction matrix |
| Cold-Start (Item) | Good (features available) | Poor (no interactions yet) |
| Cold-Start (User) | Depends on initial likes | Poor (no history) |
| Discovery | Limited to known feature space | Can suggest novel items |
| Explainability | High (feature-driven) | Low (latent factors) |
| Scalability | Depends on feature dims | Can be heavy for memory CF |
| Over-specialization | High (too similar) | Lower (learns from others) |

### 6. Hybrid Approaches

Combine both methods to offset weaknesses:

- Weighted blend of scores
- Switching: use content for cold-start, CF otherwise
- Feature augmentation: add CF latent factors to content model

### 7. Practice Problems

1. **Implement Both Methods**
    - Use MovieLens 100K: build a content model using genres, and an item-based CF using ratings. Compare Precision@10.
2. **Cold-Start Simulation**
    - Hold out 5 new movies and recommend them to users. Evaluate content vs CF performance.
3. **Hybrid Tuner**
    - Blend content and CF scores with weight α. Sweep α from 0 to 1 and plot validation precision to find the optimal mix.
4. **Explainability Exercise**
    - For a few recommendations from each model, list the top 3 features or users driving the suggestion.

---

## Deep Learning for Content-Based Filtering

Below is a complete roadmap—from fundamentals to state-of-the-art—on using deep neural networks to power content-based recommendation. All formulas and snippets live in plain code blocks so you can copy them straight into Notion.

### 1. Motivation & Overview

Content-based filtering relies on item attributes to drive recommendations. Traditional methods (TF-IDF, one-hot vectors) struggle to capture rich semantics in text, images, or audio. Deep learning lets us learn dense, task-specific embeddings that reflect intricate item similarities, enabling:

- Better cold-start handling for new items
- Richer semantic matching
- Multi-modal fusion of text, images, audio

### 2. Item Input Types

- Textual metadata (titles, descriptions, reviews)
- Visual media (product photos, video frames)
- Audio snippets (music clips, podcasts)
- Structured attributes (categories, tags, numeric features)
- Multi-modal combos (text + image, text + audio)

### 3. Neural Encoders for Item Features

### 3.1 Text Encoders

- Convolutional Neural Networks (CNNs) over word embeddings
- Recurrent Neural Networks (LSTM, GRU) for sequence modeling
- Transformer encoders (BERT, RoBERTa, DistilBERT) for contextual embeddings

```python
# Example: encode text with a simple 1D CNN
inputs = Input(shape=(max_len,), dtype='int32')
emb   = Embedding(vocab_size, embed_dim)(inputs)
conv  = Conv1D(filters=128, kernel_size=5, activation='relu')(emb)
pool  = GlobalMaxPool1D()(conv)
text_embed = Dense(latent_dim, activation='relu')(pool)
```

### 3.2 Image Encoders

- Pretrained CNNs (ResNet, EfficientNet) fine-tuned or frozen
- Vision Transformers (ViT) for patch-based embeddings

```python
# Example: use a pretrained ResNet backbone
base_model   = tf.keras.applications.ResNet50(
    include_top=False, input_shape=(224,224,3), pooling='avg')
img_outputs  = base_model(img_inputs)
img_embed    = Dense(latent_dim, activation='relu')(img_outputs)
```

### 3.3 Audio & Other Modalities

- 1D CNNs or spectrogram-based 2D CNNs for audio
- Graph Neural Networks for structured metadata graphs

### 4. Learning Item Embeddings

- **Siamese Networks**: train two identical encoders to map similar items close together via contrastive loss
- **Autoencoders**: reconstruct raw features, use bottleneck as embedding
- **Triplet Networks**: optimize anchor-positive vs anchor-negative distances

```python
# Contrastive loss example
def contrastive_loss(y_true, d):
    margin = 1.0
    return y_true * K.square(d) + (1 - y_true) * K.square(K.maximum(margin - d, 0))
```

### 5. User Profile Modeling

Aggregate item embeddings from a user’s history to form a user vector:

- **Average pooling**
- **Recurrent models** (LSTM over chronological clicks)
- **Self-Attention/Transformer** to weigh interactions differently

```python
# Simple user vector by averaging item embeds
user_history_embeds = EmbeddingSequenceLayer()(history_item_ids)
user_vector = tf.reduce_mean(user_history_embeds, axis=1)
```

### 6. Scoring & Recommendation

### 6.1 Dot-Product Scoring

```python
score = tf.reduce_sum(user_vector * item_vector, axis=-1, keepdims=True)
```

### 6.2 Neural Scoring (MLP)

```python
concat = Concatenate()([user_vector, item_vector])
x      = Dense(128, activation='relu')(concat)
score  = Dense(1, activation=None)(x)
```

### 7. Loss Functions

### 7.1 Pointwise Regression / Classification

```python
# Regression for explicit ratings
loss = tf.reduce_mean((y_true - score)**2)

# Binary classification for clicks/likes
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, score)
```

### 7.2 Pairwise Ranking (BPR)

```python
# x_uij = score(u,i_pos) - score(u,i_neg)
loss = -tf.reduce_mean(tf.math.log_sigmoid(x_uij))
```

### 7.3 Listwise Objectives

- Softmax over candidates, cross-entropy against ground truth
- Listwise losses (e.g., LambdaRank, ListNet)

### 8. Training Strategies

- **Negative Sampling**: sample unobserved items per user
- **Batch construction**: group by user or random triples
- **Pretraining**: initialize from general embedding models (BERT, ResNet)
- **Fine-tuning**: gradually unfreeze pretrained layers

### 9. Multi-Modal Fusion

- **Early fusion**: concatenate embeddings from different modalities
- **Late fusion**: compute modality-specific scores, then blend
- **Co-attention**: allow cross-modal attention between text and image

```python
# Early fusion example
combined = Concatenate()([text_embed, img_embed])
joint    = Dense(latent_dim, activation='relu')(combined)
```

### 10. Cold-Start & Zero-Shot

- New items: generate embedding from raw features only
- Meta-learning approaches (MAML) to adapt quickly with few examples

### 11. Interpretability & Explainability

- Attention weights over text tokens or past items
- Saliency maps on images (Grad-CAM)
- Post-hoc methods: LIME, SHAP on embedding inputs

### 12. Scalability & Retrieval

- Precompute item embeddings, index with Approximate Nearest Neighbor (FAISS, Annoy)
- Use HNSW or product quantization for billion-scale catalogs
- Online inference: embed incoming items/users, query index

---

## Scalable Recommendations for Large Catalogues

### 1. Why Scaling Matters

- When you have millions (or billions) of items, brute-force scoring every item per user becomes infeasible.
- You need low-latency (few tens of milliseconds) online predictions under tight memory budgets.
- A two-stage recall + ranking pipeline lets you prune the catalogue before applying heavy models.

### 2. Two-Stage Recommendation Pipeline

1. Candidate Retrieval (Recall)
    - Quickly pull a small subset (e.g. 100–1,000) of “likely” items
    - Must be sub-linear (log-time or constant) wrt catalogue size
2. Candidate Ranking
    - Compute rich features (context, time, item metadata)
    - Score top-k with a more expensive model (GBDT, neural net)

### 3. Candidate Retrieval Techniques

| Method | Strengths | Trade-Offs |
| --- | --- | --- |
| Popularity Heuristic | Very fast; cold-start safe | Bias towards popular items |
| Inverted Index | Exact text/feature matching | Doesn’t capture semantics |
| Approx. Nearest Neighbors (ANN) | Semantic recall via embeddings | Slight recall drop; extra index cost |

### 3.1 Approximate Nearest Neighbor (ANN)

- Map users & items into a shared embedding space
- Build an index that finds top-k nearest item vectors to a user vector in sub-linear time

Popular libraries:

- Faiss (Facebook AI)
- Annoy (Spotify)
- ScaNN (Google)

### 3.1.1 HNSW with Faiss (Python Example)

```python
import numpy as np
import faiss

# Suppose item_embeddings is (n_items × d) float32 array
d = item_embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32)      # 32 neighbors per node
index.hnsw.efConstruction = 200         # construction time/search accuracy trade-off
index.add(item_embeddings)              # build index

# Build user_profile (1 × d) as before
_, topk_ids = index.search(user_profile.reshape(1, d), k=100)
```

- index.search runs in ~O(log n) time
- efSearch controls query accuracy vs speed

### 4. Candidate Generation Strategies

- **Popularity + Recency**: Blend global popularity with trending scores
- **Session-Based Retrieval**: Use last-N interactions (e.g. co-view co-click)
- **User-User CF (ANN on user vectors)**
- **Item-Item CF (ANN on item embeddings)**

Mix and match: e.g. take 50 popular, 30 ann-recalled, 20 session-based.

### 5. End-to-End Architecture

1. **Offline ETL**
    - Update item embeddings nightly
    - Rebuild or incrementally update ANN index
2. **Feature Store**
    - Materialize user profiles, session features, item metadata
3. **Online Service**
    - Recall via ANN + heuristics (5–10 ms)
    - Fetch features for recalled IDs
    - Rank top 100 with GBDT or neural model (20–30 ms)
4. **Monitoring & Feedback**
    - Log impressions & interactions
    - Periodically re-train embeddings and ranking models

### 6. Hands-On Exercise

1. Choose MovieLens-1M as your catalogue.
2. Compute 32-D item embeddings (e.g. via matrix factorization or averaging genre/tfidf).
3. Build a Faiss HNSW index and recall top 50 for each user profile.
4. Train a simple ranking model (logistic regression + features: timestamp, popularity, genre overlap).
5. Measure Recall@50 and Precision@10 versus a brute-force cosine baseline.

---

## Ethical Use of a Recommender System

The core of ethical recommendation is ensuring that your system balances user benefit, business goals, and societal values. This means designing for fairness, transparency, privacy, accountability, diversity, and user autonomy at every stage of the pipeline.

### 1. Fairness

- Identify bias sources
    - Data imbalance (over-representation of popular items or demographics)
    - Feedback loops (popular items get more visibility)
- Mitigation strategies
    - Re-weight training data or loss function to reduce disparate impact
    - Adversarial debiasing layers that remove sensitive-attribute signals
    - Post-hoc reranking to ensure demographic or item-category parity
- Metrics to track
    - Disparate Impact Ratio
    - Equality of Opportunity (e.g., equal true-positive rates across groups)
    - Gini or Shannon entropy for item distribution

### 2. Transparency & Explainability

- Explain recommendations in real-time
    - Feature-level explanations (“You saw sci-fi shows, so this was scored highly”)
    - Example-based explanations (“Users like you also watched…”)
- Auditability
    - Log model inputs, outputs, and decision paths
    - Regularly inspect for drift or unexplained anomalies
- Tools
    - LIME or SHAP for per-user explanations
    - Model cards or datasheets for full-system documentation

### 3. Privacy & Data Protection

- Data minimization
    - Collect only the history and metadata needed for quality recall/ranking
- Anonymization & Aggregation
    - k-anonymity for user identifiers
    - Differential privacy noise injection during embedding updates
- Consent & Control
    - Opt-in/out toggles for personalized suggestions
    - Transparent data retention and deletion policies

### 4. Accountability & Governance

- Roles & responsibilities
    - Define who owns fairness metrics, privacy audits, and incident response
- Monitoring & Alerts
    - Real-time dashboards for engagement vs. bias indicators
    - Automated alerts when a protected group’s relevance drops below threshold
- Human-in-the-loop
    - Manual review of flagged recommendations
    - Periodic ethics board assessments

### 5. Diversity & Novelty

- Avoid echo chambers
    - Inject serendipity: random exploration slots in each slate
    - Maximum Marginal Relevance (MMR) reranking for topical diversity
- Metrics
    - Coverage (fraction of catalog shown over time)
    - Intra-list diversity (average pairwise dissimilarity)

### 6. Practical Interview & Job-Readiness Tips

| Topic | What to Know | Example Question |
| --- | --- | --- |
| Fairness Metrics | Definitions, computation, trade-offs | “How would you detect and mitigate popularity bias?” |
| Explainable Models | LIME/SHAP basics, UI design | “Design an API that returns an explanation with each rec.” |
| Privacy Techniques | k-anonymity vs. differential privacy | “How would you protect user data in a streaming service?” |
| Monitoring Frameworks | Dashboards, alert thresholds, human-in-loop workflows | “Outline an A/B test to measure fairness regressions.” |

### 7. Hands-On Exercises

1. **Bias Audit Notebook**
    - On MovieLens or your own data, compute recommendation rates per demographic.
    - Apply a re-weighting of training samples and compare parity metrics.
2. **Explainable Rec Demo**
    - Build a simple item-based recommender and integrate SHAP.
    - Create a small web UI showing feature contributions for each recommendation.
3. **Differential Privacy in MF**
    - Inject Gaussian noise into gradient updates of matrix factorization.
    - Measure RMSE vs. privacy parameter ε trade-off.
4. **Diversity-Constrained Ranking**
    - Implement MMR reranking over top-100 candidates.
    - Plot precision@10 vs. intra-list diversity curves for different λ values.

---

## TensorFlow Implementation of Content-Based Filtering

This guide walks through a simple content-based recommender built with TensorFlow 2.x. You’ll see how to:

1. Load or define item feature vectors
2. Build a user profile from liked items
3. Compute cosine similarities in TensorFlow
4. Produce top-K recommendations

All code is copy-paste-ready and designed for your personal knowledge base.

### 1. Setup and Prerequisites

```bash
pip install tensorflow numpy
```

```python
import numpy as np
import tensorflow as tf
```

- TensorFlow 2.x
- NumPy for synthetic data or data loading
- Basic familiarity with `tf.Tensor` operations

### 2. Prepare Item Features

Assume `n_items` items each described by an `f`-dimensional feature vector (e.g., TF-IDF on text, one-hot genres, or precomputed embeddings).

```python
n_items, f = 10000, 50
# Example: random features for demonstration
item_features = np.random.random((n_items, f)).astype(np.float32)

# Convert to a TensorFlow constant
item_feats_tf = tf.constant(item_features)  # shape: (n_items, f)
```

You can replace the random matrix with real features loaded via Pandas or tf.data.

### 3. Build the User Profile

Given a user’s history—a list of item IDs they've liked—you form a weighted average of the corresponding feature vectors.

```python
# Example liked item IDs and optional weights
liked_ids = np.array([12, 305, 4780, 999])       # user clicked/liked these
weights   = np.array([1.0, 0.5, 1.0, 1.5])       # optional importance weights

# Gather liked features and compute weighted average
liked_feats = tf.gather(item_feats_tf, liked_ids)     # shape: (len(liked_ids), f)
w = tf.reshape(weights, (-1, 1))                      # shape: (len(liked_ids), 1)

user_profile = tf.reduce_sum(liked_feats * w, axis=0) / tf.reduce_sum(w)
# user_profile: shape (f,)
```

If you have only binary feedback, set all `weights = 1.0`.

### 4. Cosine Similarity Scoring

We compute cosine similarity between the single `user_profile` vector and all `n_items` vectors in one batch.

```python
def cosine_similarity(matrix, vector):
    # matrix: (n_items, f), vector: (f,)
    matrix_norms = tf.norm(matrix, axis=1)            # (n_items,)
    vector_norm  = tf.norm(vector)                    # scalar
    dots         = tf.tensordot(matrix, vector, axes=1)  # (n_items,)
    return dots / (matrix_norms * vector_norm + 1e-8)

scores = cosine_similarity(item_feats_tf, user_profile)  # shape: (n_items,)
```

### 5. Extract Top-K Recommendations

Use `tf.math.top_k` for efficient selection.

```python
K = 10
topk = tf.math.top_k(scores, k=K)

recommended_ids    = topk.indices.numpy()   # item IDs
recommended_scores = topk.values.numpy()    # similarity scores

print("Top-K Recommendations:")
for idx, score in zip(recommended_ids, recommended_scores):
    print(f"Item {idx} — score {score:.4f}")
```

You now have the `K` most content-similar items to the user profile.

### 6. Wrapping as a Reusable Function

```python
@tf.function
def recommend(items: tf.Tensor,
              user_ids: tf.Tensor,
              user_weights: tf.Tensor,
              top_k: int = 10):
    # Build profile
    feats = tf.gather(items, user_ids)           # (m, f)
    w     = tf.reshape(user_weights, (-1, 1))    # (m, 1)
    profile = tf.reduce_sum(feats * w, axis=0) / tf.reduce_sum(w)

    # Score all items
    sims = cosine_similarity(items, profile)     # (n_items,)
    top  = tf.math.top_k(sims, k=top_k)
    return top.indices, top.values

# Example usage
ids, vals = recommend(item_feats_tf,
                      tf.constant(liked_ids),
                      tf.constant(weights, dtype=tf.float32),
                      top_k=5)
```

Marking the function with `@tf.function` compiles it to a graph for speed in production.

---