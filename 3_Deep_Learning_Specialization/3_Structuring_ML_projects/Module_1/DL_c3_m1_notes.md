# DL_c3_m1

## Orthogonalization in ML Strategy

Orthogonalization means decoupling your system’s components so that changes in one part don’t ripple unpredictably into others. In ML projects, it ensures you can isolate, debug, and iterate on each element—data, features, model architecture, and evaluation—independently.

### 1. Direct Definition

Orthogonalization is the practice of structuring your ML pipeline so that each variable you tune or component you modify impacts only one “axis” of performance (e.g., model capacity, features, data quantity), making it easy to attribute gains or errors to the right change.

### 2. Concept Intuition

- Why it matters:Prevents confounding: You know exactly which change drove performance up or down.Speeds debugging: When error spikes, you trace it to a single module (data-cleaning, feature-engineering, architecture, etc.).Streamlines collaboration: Team members own discrete subsystems—data engineers, feature engineers, modelers—without stepping on each other’s toes.

### 3. Mathematical Breakdown

At its core, orthogonalization shows up in how you decompose error and track gaps:

```python
# Let:
#   J_train  = training set error
#   J_dev    = development set error

generalization_gap = J_dev - J_train
```

If you treat feature engineering and model capacity as orthogonal axes, you can write:

```python
# Total error breakdown (conceptually):
J_dev ≈ bias_error + variance_error + data_mismatch_error + noise
```

- bias_error : error_train
- variance_error: generalization_gap
- data_mismatch_error: difference between dev set and real-world distribution
- noise : irreducible error

By modifying only one term at a time, you ensure orthogonality.

### 4. Code & Practical Application

### a. Fix your feature pipeline while tuning model capacity

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Split once and hold feature pipeline fixed
X, y = np.random.randn(1000, 10), np.random.randint(0, 2, size=1000)
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build and freeze scaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_dev_scaled   = scaler.transform(X_dev)

# 3. Sweep over model capacity (C parameter)
results = {}
for C in [0.01, 0.1, 1, 10]:
    model = LogisticRegression(C=C, solver='lbfgs', max_iter=200)
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)
    dev_acc   = model.score(X_dev_scaled, y_dev)
    results[C] = (train_acc, dev_acc)

print("C parameter vs (train_acc, dev_acc):", results)
```

This code isolates model-capacity (C) without re-tuning the scaler or features—an orthogonal experiment.

### 5. Visualization / Geometry

```
                 Vary model capacity
               ↑
 error          •———•———•———•
 (accuracy)     |   |   |   |
 axis           •   •   •   •   ← Fixed feature pipeline
               |   |   |   |
               •———•———•———•
```

- Vertical axis: model capacity
- Horizontal axis: you hold feature preprocessing constant
- Each dot: one experiment run

By moving only up/down, you see how capacity alone shifts error, without any diagonal confounding of feature changes.

### 6. Common Pitfalls & Tips

- Pitfall: Re-running data cleaning or augmentation inside your model-sweeps.Tip: Precompute and serialize transformed data so feature work stays fixed.
- Pitfall: Mixing hyperparameter search for features and architecture in one loop.Tip: Two-stage search—first fix features ↦ adjust model; then fix model ↦ adjust features.
- Pitfall: Leaking dev-set information into preprocessing.Tip: Fit scalers/tokenizers only on train set; apply “transform” to dev/test.

### 7. Interview-Ready Insights

- Definition: “Orthogonalization means decoupling axes of experimentation so you can attribute performance changes to one factor at a time.”
- Why: Reduces debugging time and keeps modular code.
- Example: Freeze your feature pipeline when tuning learning rate or model size.
- Advanced: Discuss multi-factor experiments with grid searches vs. sequential orthogonalization and trade-offs in compute cost.

### 8. Practice Exercises

1. **Dev/Test Split Isolation**
    - Load the UCI Iris dataset. Split 70/30 train/dev.
    - Build a feature pipeline: StandardScaler + PCA (n_components=2).
    - Freeze the pipeline, then train a small neural net with Keras varying only the number of hidden units [4, 8, 16].
    - Plot train vs. dev accuracy for each hidden-unit count.
    
    *Hint:* Use `Pipeline` from scikit-learn and call `pipeline.fit_transform(X_train)` only once.
    
2. **Two-Stage Search**
    - Write two loops:a) First loop: keep model architecture fixed, search best dropout rate [0.1, 0.3, 0.5].b) Second loop: freeze dropout, search best number of layers [1,2,3].
    - Report which combo achieves highest dev accuracy.
3. **Error Decomposition Report**
    - For your best model above, compute and report:
        - Training error
        - Dev error
        - Generalization gap
    - Interpret which term (bias vs. variance) is dominant, and propose your next orthogonal experiment.

---

## Single Number Evaluation Metric

A single number evaluation metric is a scalar summary that captures your model’s performance on a task. It lets you compare, track, and optimize models with one clear objective.

### 1. Direct Definition

A single number evaluation metric reduces model performance to one value—such as accuracy, F1-score, mean squared error, or AUC—so you have an agreed-upon target for hyperparameter tuning, model selection, and production monitoring.

### 2. Concept Intuition

- Why it matters:Alignment: Everyone (data scientists, engineers, stakeholders) speaks the same language about “how good” your model is.Simplified decisions: One metric drives early stopping, model checkpoints, and A/B tests.Trade-off clarity: When you adjust thresholds, architectures, or data, you immediately see the impact on your key metric.

### 3. Mathematical Breakdown

For a binary classification task, key metrics and their formulas:

```python
# Let:
#   TP = true positives
#   TN = true negatives
#   FP = false positives
#   FN = false negatives

accuracy  = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP)

recall    = TP / (TP + FN)

f1_score  = 2 * (precision * recall) / (precision + recall)
```

For a regression task:

```python
# y_true, y_pred are arrays of true and predicted values

mean_squared_error = sum((y_true - y_pred)**2) / len(y_true)

mean_absolute_error = sum(abs(y_true - y_pred)) / len(y_true)
```

Each formula collapses a confusion matrix or residual vector into one number.

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)

# Sample data
y_true = np.array([1,0,1,1,0,1,0,0,1,0])
y_pred = np.array([1,0,1,0,0,1,0,1,1,0])
y_proba= np.array([0.9,0.2,0.8,0.4,0.1,0.95,0.3,0.7,0.85,0.05])

# Classification metrics
print("Accuracy:",  accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:",    recall_score(y_true, y_pred))
print("F1 score:",  f1_score(y_true, y_pred))
print("AUC-ROC:",   roc_auc_score(y_true, y_proba))

# Regression metrics (toy continuous labels)
y_true_reg = np.array([2.5, 3.0, 4.5, 5.0])
y_pred_reg = np.array([2.7, 2.8, 4.0, 5.4])

print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
print("MAE:", mean_absolute_error(y_true_reg, y_pred_reg))
```

This snippet shows how to compute and log your chosen metric end-to-end.

### 5. Visualization / Geometry

```
                ┌─────────┐
                │ Confusion│
                │  Matrix  │
┌─────────────┐ │ TP FP   │
│ Actual Pos  │─┤        ├─ Pred Pos
│             │ │ FN TN   │
└─────────────┘ └─────────┘
```

- Moving the decision threshold changes FP vs. FN trade-off.
- Plotting precision vs. recall or ROC curves shows you geometry of trade-offs.
- The AUC is literally the area under the ROC curve—a single scalar summarizing all thresholds.

### 6. Common Pitfalls & Tips

- Pitfall: Using accuracy on a 99:1 imbalance.
    
    Tip: Switch to precision/recall or F1 to surface rare‐class performance.
    
- Pitfall: Over-optimizing one metric at the expense of others (e.g., high recall but tiny precision).
    
    Tip: Define minimum acceptable floors for secondary metrics.
    
- Pitfall: Ignoring business KPIs (e.g., cost per false alarm).
    
    Tip: Map your metric back to dollars, user experience, or risk before finalizing.
    

### 7. Interview-Ready Insights

- Definition: “A single number metric is your north star—one scalar that aligns model development, evaluation, and business goals.”
- Trade-offs: “Accuracy is easy but brittle under imbalance. F1 balances precision and recall. AUC captures performance across all thresholds.”
- Advanced: “For multi‐class problems, micro vs. macro averaging changes how you weight rare classes. For ranking tasks, consider MAP or NDCG.”

### 8. Practice Exercises

1. Implement F1-score from scratch given TP, FP, FN.
    
    *Hint:* Avoid division by zero when TP + FP = 0.
    
2. On the UCI Wine Quality dataset:
    - Train a small classifier to predict “good” vs. “bad” wine.
    - Compare accuracy, F1, and AUC.
    - Which metric would you optimize if false positives (classifying bad wine as good) cost $10 each?
3. Imbalanced Dataset Simulation:
    - Generate 1 000 samples with 95% class 0, 5% class 1.
    - Fit logistic regression.
    - Compute accuracy vs. recall.
    - Plot a bar chart showing how accuracy can mislead you.
4. Regression Thresholding:
    - For a regression task, decide a cutoff (e.g., y_true > 3⟶“high”).
    - Convert to binary and compute classification metrics.
    - Reflect on metric selection for mixed tasks.

---

## Satisficing and Optimizing Metrics

Satisficing and optimizing metrics form a two‐metric strategy: you first ensure your model meets minimum requirements on a “must‐have” metric (satisficing), then you push to improve a secondary “nice‐to‐have” metric (optimizing).

### 1. Direct Definition

Satisficing metric: a threshold you require your model to hit before anything else (e.g., recall ≥ 0.75).

Optimizing metric: once the threshold is met, you tune and compare models based on this metric (e.g., F1‐score or precision).

### 2. Concept Intuition

When you juggle multiple goals—say, catching as many fraud cases as possible without swamping analysts—you need a guardrail plus a dial:

- Satisficing metric acts as a safety net: ensures you don’t sacrifice critical business needs.
- Optimizing metric drives fine‐tuning: focuses on efficiency, cost, or user experience after the floor is met.

This decoupling aligns stakeholders: product needs (must catch fraud) and engineering goals (minimize false alarms).

### 3. Mathematical Breakdown

```python
# Given:
#  TP, FP, FN, TN counts from predictions

recall    = TP / (TP + FN)          # Satisficing metric
precision = TP / (TP + FP)
f1_score  = 2 * precision * recall / (precision + recall)  # Optimizing metric
```

Workflow:

1. Check recall ≥ R_thresh.
2. If yes, compute f1_score and compare models.
3. Discard any model with recall < R_thresh, regardless of F1.

This decomposition ensures orthogonality between “meeting business requirement” and “driving performance.”

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split

# Generate imbalanced data
X, y = make_classification(
    n_samples=2000, n_features=20, weights=[0.9, 0.1],
    flip_y=0.01, random_state=0
)

X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.3, random_state=0
)

models = {
    'C=0.01': LogisticRegression(C=0.01, solver='lbfgs', max_iter=200),
    'C=0.1':  LogisticRegression(C=0.1, solver='lbfgs', max_iter=200),
    'C=1':    LogisticRegression(C=1, solver='lbfgs', max_iter=200),
    'C=10':   LogisticRegression(C=10, solver='lbfgs', max_iter=200),
}

R_THRESH = 0.75
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_dev_pred = model.predict(X_dev)
    rec = recall_score(y_dev, y_dev_pred)
    if rec < R_THRESH:
        # Model fails satisficing requirement
        results[name] = {'recall': rec, 'f1_score': None}
    else:
        f1 = f1_score(y_dev, y_dev_pred)
        results[name] = {'recall': rec, 'f1_score': f1}

print("Model comparison (must satisfy recall ≥ {:.2f}):".format(R_THRESH))
for name, metrics in results.items():
    print(f"{name} → recall: {metrics['recall']:.2f}, f1_score: {metrics['f1_score']}")
```

This snippet trains four models, discards any below the recall threshold, and ranks the rest by F1.

### 5. Visualization / Geometry

```
        ↑ F1 score
        |
   high •          •       •
        |      •
        |
   low  •___•________•_____  → Recall
           .75 (satisficing threshold)
```

- X‐axis: recall. Vertical dotted line at R_THRESH.
- Only points to the right qualify.
- Y‐axis: F1 score; higher is better for optimizing metric.
- You first filter horizontally, then pick the highest dot vertically.

### 6. Common Pitfalls & Tips

- Pitfall: Setting satisficing threshold too high before any tuning.Tip: Calibrate with a simple baseline; choose a realistic threshold.
- Pitfall: Ignoring model complexity when optimizing secondary metric.Tip: Among models that satisfy, consider model size or latency as a tertiary metric.
- Pitfall: Satisficing the wrong metric (e.g., accuracy in imbalanced data).Tip: Always link your threshold to a business imperative (cost of missed fraud).

### 7. Interview-Ready Insights

- Definition in a sentence:“Use a satisficing metric to guarantee business constraints, then optimize a secondary metric for efficiency or cost.”
- Why it matters:“Separates must-haves from nice-to-haves, streamlining hyperparameter searches and aligning cross-functional teams.”
- Real-world example:“In medical diagnosis, require sensitivity ≥ 0.9 (satisficing) then maximize specificity or F1 among those models.”
- Advanced nuance:“You can generalize to multi-objective satisficing: satisfy multiple thresholds, then use a weighted sum or hierarchy for optimization.”

### 8. Practice Exercises

1. **Threshold Tuning**
    - On the sklearn breast cancer dataset, require precision ≥ 0.85.
    - Sweep decision thresholds on predicted probabilities from 0.1 to 0.9.
    - Select the threshold that satisfies the precision constraint and maximizes recall.
2. **Two-Stage Grid Search**
    - Using `RandomForestClassifier`, first grid‐search `min_samples_leaf` to satisfy recall ≥ 0.8.
    - Among those, grid‐search `n_estimators` to maximize ROC‐AUC.
    - Report best parameter pair and associated metrics.
3. **Business-Aligned Metric Mapping**
    - Imagine each false negative costs $100, each false positive costs $10.
    - Define a new single number:
        
        ```python
        cost = FN * 100 + FP * 10
        ```
        
    - Set a sati sfice threshold on `cost ≤ 500`.
    - Optimize on F1 among models that meet this budget.

---

## Train/Dev/Test Distribution

Splitting your dataset into distinct training, development (dev/validation), and test sets is foundational for reliable model building, hyperparameter tuning, and unbiased evaluation.

### 1. Direct Definition

A train/dev/test distribution partitions your data into three non-overlapping subsets:

- **Training set**: used to fit model parameters.
- **Development (dev/validation) set**: used to tune hyperparameters and make model-selection decisions.
- **Test set**: held out completely until final evaluation to estimate real-world performance.

### 2. Concept Intuition

Separating data into three groups prevents information leakage and overfitting:

- When you tune hyperparameters on the dev set, you “see” those examples—so you can’t trust dev scores as a final gauge.
- The test set acts as an unseen benchmark, simulating how the model performs on fresh data.
- Clear separation keeps your evaluation orthogonal: each metric (training error, dev error, test error) reflects different stages of model development and generalization.

### 3. Mathematical Breakdown

Let the full dataset have (m) examples. Define proportions:

```python
m_train = floor(m * p_train)
m_dev   = floor(m * p_dev)
m_test  = m - m_train - m_dev
```

where typically `p_train = 0.7–0.8`, `p_dev = 0.1–0.15`, `p_test = 0.1–0.15`.

Errors on each set:

```python
J_train = (1/m_train) * sum(loss(f(x_i), y_i) for (x_i,y_i) in train)
J_dev   = (1/m_dev)   * sum(loss(f(x_i), y_i) for (x_i,y_i) in dev)
J_test  = (1/m_test)  * sum(loss(f(x_i), y_i) for (x_i,y_i) in test)
```

Key gaps:

```python
# Generalization gap during tuning
gap_dev = J_dev - J_train

# Final reported gap
gap_test = J_test - J_train
```

By keeping dev/test separate, you ensure `gap_dev` guides hyperparameter choices, and `gap_test` remains an unbiased performance estimate.

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 1. Load data
data = load_boston()
X, y = data.data, data.target

# 2. First split: train_val + test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# 3. Second split: train + dev
X_train, X_dev, y_train, y_dev = train_test_split(
    X_trainval, y_trainval, test_size=0.1765,  # 0.1765*0.85 ≈ 0.15 overall
    random_state=42
)

# 4. Verify sizes
print("Train:", X_train.shape[0],
      "Dev:", X_dev.shape[0],
      "Test:", X_test.shape[0])
```

**Engineering choices**:

- Use a fixed `random_state` for reproducibility.
- Stratify splits for classification to preserve class balances.
- Store split data artifacts (e.g., serialized arrays or TFRecords) to avoid re-splitting in pipelines.

### 5. Visualization / Geometry

```
Full Dataset (m examples)
┌───────────────────────────────────────────────┐
│   Train set    │   Dev set    │   Test set   │
│  p_train ≈ 70% │ p_dev ≈ 15%  │ p_test ≈ 15% │
└───────────────────────────────────────────────┘

Error curve:
   J
   ↑           • J_dev
   │        /
   │     • J_train
   │    /
   │
   │              • J_test
   └─────────────────────────→ model complexity
```

- As complexity grows, training error falls fastest.
- Dev error shows “sweet spot” for hyperparameters.
- Test error confirms whether the chosen complexity generalizes.

### 6. Common Pitfalls & Tips

- Pitfall: **Leaking dev/test info** when standardizing or augmenting data.
    
    Tip: Fit scalers/augmenters only on the train set and apply to dev/test (“transform” only).
    
- Pitfall: **Non-representative splits** if data are time-series or clustered.
    
    Tip: For time-series, use chronological splits; for clusters (e.g., users), split by group.
    
- Pitfall: **Insufficient dev/test size** leading to noisy estimates.
    
    Tip: Aim for at least a few hundred examples in dev/test to stabilize error estimates.
    
- Pitfall: **Reusing the test set** during iterative development.
    
    Tip: Lock the test set and only evaluate on it once, after all tuning is complete.
    

### 7. Interview-Ready Insights

- **Why three sets?**
    
    “Dev set guides hyperparameter tuning; test set simulates unseen data for final performance guarantees.”
    
- **Choosing proportions:**
    
    “Standard splits are 70/15/15 or 80/10/10, but if data are scarce, use k-fold cross-validation on train+dev and reserve a small held-out test set.”
    
- **Advanced nuance:**
    
    “In production, you may add a monitoring “live” set: periodically hold back a slice of real-time data to detect drift without affecting dev/test.”
    
- **Real-world example:**
    
    “In a search-ranking system, you might split by user session ID, ensuring all queries from one user land in the same subset to prevent leakage.”
    

### 8. Practice Exercises

1. **Stratified Split on Imbalanced Data**
    - Load the sklearn `load_breast_cancer` dataset.
    - Perform a 70/15/15 stratified split preserving class ratios.
    - Verify class distributions in each subset.
2. **Time-Series Split**
    - Simulate a time series: `X = np.arange(1000).reshape(-1,1)`, `y = sin(X) + noise`.
    - Split the first 70% as train, next 15% as dev, final 15% as test.
    - Plot true vs. predicted for a simple linear regression on each subset and compare errors.
3. **Cross-Validation vs. Hold-out**
    - On the UCI Wine Quality dataset, compare dev error from a single 80/10/10 split vs. 5-fold CV (on 90% of data) with a held-out 10% test.
    - Report mean dev error for both methods and final test error.
    - Reflect on variance in dev estimates.
4. **Distribution Mismatch Simulation**
    - Create `X` from two 2D Gaussians for train/dev, but shift the mean for test.
    - Train a classifier on train/dev split; evaluate on test.
    - Compute `gap_dev` and `gap_test`; analyze the impact of distribution shift.

---

## Size of Dev and Test Sets

Choosing the right sizes for your development (dev) and test sets balances reliable performance estimates against having enough data to train a robust model.

### 1. Direct Definition

The size of the dev and test sets refers to the number of examples (or percentage of the full dataset) reserved for hyperparameter tuning (dev set) and for final, unbiased evaluation (test set).

### 2. Concept Intuition

- A larger dev set gives you **lower-variance** estimates when you compare models or tune hyperparameters.
- A larger test set gives you **higher-confidence** in your final performance report.
- But every example you allocate to dev/test is one less example for training—potentially hurting model quality.

You trade off **training data quantity** vs. **evaluation reliability**.

### 3. Mathematical Breakdown

Let

```python
m      # total examples
p_train  # fraction for training
p_dev    # fraction for dev
p_test   # fraction for test
```

Then

```python
m_train = floor(m * p_train)
m_dev   = floor(m * p_dev)
m_test  = m - m_train - m_dev
```

Typical choices:

```python
p_train = 0.7 to 0.8
p_dev   = 0.1 to 0.15
p_test  = 0.1 to 0.15
```

Variance of an error estimate scales roughly as

```python
var(J) ∝ 1 / m_subset
```

so doubling m_dev halves the estimate’s variance.

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data
X, y = fetch_california_housing(return_X_y=True)
m = X.shape[0]

# Define proportions
p_test = 0.15
p_dev  = 0.15
p_train = 1 - p_test - p_dev

# First split: train_val + test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=p_test, random_state=0
)

# Second split: train + dev
dev_fraction_of_trainval = p_dev / (p_train + p_dev)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_trainval, y_trainval,
    test_size=dev_fraction_of_trainval,
    random_state=0
)

# Report sizes
print(f"Total examples: {m}")
print(f"Train: {X_train.shape[0]} ({X_train.shape[0]/m:.2%})")
print(f"Dev:   {X_dev.shape[0]} ({X_dev.shape[0]/m:.2%})")
print(f"Test:  {X_test.shape[0]} ({X_test.shape[0]/m:.2%})")
```

This code cleanly carves out the three sets according to your chosen fractions.

### 5. Visualization / Geometry

| Split | Proportion | Examples (if m=10,000) |
| --- | --- | --- |
| Train | 70% | 7,000 |
| Dev | 15% | 1,500 |
| Test | 15% | 1,500 |

```
Full Dataset ─────────────────────────────────────────────────
│██████████████████████████████████████████████████████████│
 Train (70%) ────────────────│ Dev (15%) │ Test (15%) │
```

- The longer the colored bar for dev/test, the lower the **scatter** (variance) in performance estimates.

### 6. Common Pitfalls & Tips

- Pitfall: **Dev/Test too small** → noisy hyperparameter decisions or unreliable final metrics.
    
    Tip: Ensure at least a few hundred examples in each set; if not possible, use cross-validation on train+dev.
    
- Pitfall: **Dev/Test too large** → starving the model of training data.
    
    Tip: If you have millions of examples, you can afford 20% combined; for thousands, shrink to 10–20%.
    
- Pitfall: **Fixed proportions for time-series or grouped data** can break chronology or leak information.
    
    Tip: Use time-based splits (e.g., oldest 70% for train, next 15% dev, latest 15% test) or group splits (e.g., by user or session).
    

### 7. Interview-Ready Insights

- “For small datasets (<10 K), I’d allocate 10% dev and 10% test, then perform k-fold CV on the 80% training split to stabilize my estimates.”
- “On massive datasets (>1 M), I might slice off only 5% for dev and 5% for test since variance is already low, maximizing training data.”
- “In time-series forecasting, I never random-split. I roll forward: use the first 70% of timestamps for training, next 15% for dev, final 15% to simulate future test.”

### 8. Practice Exercises

1. Load `load_breast_cancer` from scikit-learn.
    - Try three schemes:a) 70/15/15 random splitb) 80/10/10 random splitc) 60/20/20 random split
    - For each, train a simple logistic regression and record dev AUC.
    - Analyze how dev-set variability changes with split size.
2. On a dataset of your choice with <2 000 examples:
    - Perform 5-fold CV on an 80% training slice, reserving 20% as test.
    - Compare the CV mean and standard deviation vs. a single 70/15/15 split’s dev error.
3. Time-Series Simulation:
    - Generate a sine wave with noise over 1 000 points.
    - Use a sliding-window regressor (window=10).
    - Split chronologically 70/15/15.
    - Train, tune on dev, and evaluate on test.
    - Plot error vs. timestamp; comment on test behavior at edges.

---

## When to Change Dev/Test Sets and Metrics

Knowing **when** to update your data splits or evaluation metrics is as critical as building models. You want your dev/test sets and metrics to reflect the true business goal and production data—otherwise you’ll tune to the wrong target.

### 1. Direct Definition

You change your dev/test sets when their distributions or feature representations no longer match the production (or future) data.

You change your evaluation metric when the business objective, error‐cost trade‐offs, or problem statement shifts.

### 2. Concept Intuition

- Distribution drift: if new users, sensors, or markets alter the feature space, your original dev/test no longer simulate real use.
- Business pivot: a new KPI (e.g., cost of false negatives spikes) means your old metric (say, accuracy) is misaligned.
- Model lifecycle: as you add features or collect more data, your splits may become unbalanced or too small relative to training.
- Risk management: revisiting splits/metrics after each major release prevents silent performance decay.

Updating splits and metrics keeps your evaluation orthogonal and aligned—so every improvement you measure truly moves the needle.

### 3. Mathematical Breakdown

1. Measure distribution shift
    
    ```python
    # Let X_dev, X_prod be feature matrices
    from scipy.stats import ks_2samp
    p_values = [ks_2samp(X_dev[:,i], X_prod[:,i]).pvalue for i in range(X_dev.shape[1])]
    ```
    
    A low p‐value (<0.05) in many features signals drift → time to resplit.
    
2. Generalization gaps
    
    ```python
    # J_train, J_dev, J_test defined as before
    gap_dev  = J_dev - J_train
    gap_test = J_test - J_train
    # If gap_test ≫ gap_dev, test set no longer reflects future data.
    ```
    
3. Metric realignment
    
    ```python
    # Original metric: accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # New cost‐based metric:
    cost = FN * C_FN + FP * C_FP   # define C_FN, C_FP per business
    ```
    

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from sklearn.metrics import recall_score, precision_score

# Simulate production drift
X_full, y_full = np.random.randn(5000,10), np.random.randint(0,2,5000)
X_prod, y_prod = X_full[4000:], y_full[4000:]
X_old, y_old   = X_full[:4000], y_full[:4000]

# 1. Check drift between old dev and new prod
X_trainval, X_dev_old, y_trainval, y_dev_old = train_test_split(
    X_old, y_old, test_size=0.2, random_state=0
)
drift_pvals = [ks_2samp(X_dev_old[:,i], X_prod[:,i]).pvalue for i in range(X_full.shape[1])]
if sum(p < 0.05 for p in drift_pvals) > 3:
    print("Feature drift detected. Resplitting data.")

# 2. Resplit including new data
X_new = np.vstack([X_old, X_prod])
y_new = np.hstack([y_old, y_prod])
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_new, y_new, test_size=0.15, random_state=1
)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=1
)
print("Resplit sizes:", X_train.shape, X_dev.shape, X_test.shape)

# 3. Switch metric to cost if false negatives expensive
C_FN, C_FP = 100, 10
y_dev_pred = np.random.randint(0,2,size=y_dev.shape)
cost = ((y_dev != y_dev_pred) & (y_dev==1)).sum()*C_FN + ((y_dev != y_dev_pred) & (y_dev==0)).sum()*C_FP
print("Dev cost:", cost)
```

- Step 1 detects drift via Kolmogorov–Smirnov tests on features.
- Step 2 merges old + new data and recreates splits.
- Step 3 computes a cost‐based metric aligned with updated business stakes.

### 5. Visualization / Geometry

```
 Time →
 ┌────────────────────────────────────────────────────────┐
 │ Original Dev (red)            Production Data (blue) │
 │ ● ● ● ●  ● ● ●                   ● ● ● ● ● ● ● ● ●    │
 └────────────────────────────────────────────────────────┘
      ↓ Drift emerges                 ↓ Resplit dev/test

 Metric evolution:

   Business need  Accuracy  Cost-based
   ------------   --------  ----------
   Phase 1        0.85      1,200
   Phase 2        0.82      900   ← optimized cost
```

- The red vs. blue points show how dev no longer covers production distribution.
- Table shows how the cost‐metric improves alignment with the new business phase.

### 6. Common Pitfalls & Tips

- Pitfall: **Resplitting too often**, causing your dev/test to shrink and training to starve.
    
    Tip: Only resplit after significant drift or quarterly product changes.
    
- Pitfall: **Overfitting to new test** once you change metrics.
    
    Tip: Lock the updated test & metric, and treat it as sacred for final evaluation.
    
- Pitfall: **Ignoring metric stability**—metrics can fluctuate on small sets.
    
    Tip: Ensure dev/test each have ≥ 500 examples or use bootstrap confidence intervals.
    
- Pitfall: **Changing metric without stakeholder buy‐in**.
    
    Tip: Document why the new metric captures ROI, risk, or user experience better.
    

### 7. Interview-Ready Insights

- “I monitor feature distributions in production vs. dev via statistical tests; if drift exceeds a threshold, I merge new data and recreate splits.”
- “I only change metrics when business costs or user impact shifts—for example, moving from accuracy to a dollar‐cost metric when false negatives become critical.”
- “After updating dev/test or metrics, I freeze them to prevent iterative overfitting—like locking the fire escape after you know the safe route.”

### 8. Practice Exercises

1. **Drift Detection & Resplit**
    - Simulate a 2D Gaussian train/dev. Create a prod set with mean shifted by 1.
    - Write a function that runs KS tests on each dimension and resplits if ≥ 1 feature fails.
2. **Metric Pivot**
    - Use the sklearn `load_breast_cancer` data. Initially optimize recall ≥ 0.9 then F1.
    - Midway, simulate a new cost: cost = 50×FN + 5×FP.
    - Re-evaluate your models under the new cost metric and report which model now wins.
3. **Bootstrap Confidence**
    - For a fixed dev set of 1 000 examples, bootstrap 1000 replicates of dev error under accuracy.
    - Compute 95% confidence interval.
    - Discuss whether you’d increase or decrease dev size based on the interval width.
4. **Stakeholder Alignment**
    - Draft a one-page summary justifying a metric change from AUC to “expected daily dollar loss.”
    - Include charts of how model rankings change under both metrics.

---

## Why Human-Level Performance

Benchmarking against human-level performance means measuring how well skilled humans perform on your ML task and using that as a reference point. It helps you detect irreducible error, set realistic goals, and guide error decomposition.

### 1. Direct Definition

Human-level performance is the error rate (or accuracy) achieved by competent human annotators on the same task and dataset. It serves as an empirical proxy for the Bayes optimal error.

### 2. Concept Intuition

- Why it matters:Ceiling for improvement: If your model’s error ≈ human error, you’re close to the irreducible limit—further gains are tiny or impossible.Error decomposition anchor: Human error approximates Bayes error, letting you break down model shortcomings into avoidable bias and variance.Resource planning: Once you hit human-level, investing in more data or bigger models yields diminishing returns; you may need better features or domain knowledge instead.

### 3. Mathematical Breakdown

```python
# Let:
#   error_human  = human-level error rate
#   J_train      = training set error
#   J_dev        = development set error

# Approximate Bayes (irreducible) error:
error_bayes ≈ error_human

# Avoidable bias:
bias    = J_train - error_bayes

# Variance:
variance = J_dev - J_train

# Total reducible error:
reducible_error = bias + variance

# Confirm:
J_dev ≈ error_bayes + bias + variance
```

Here, `error_bayes` acts as the lowest error achievable; the gap between `J_train` and `error_bayes` is bias, and the gap between `J_dev` and `J_train` is variance.

### 4. Code & Practical Application

```python
import numpy as np

# Simulate measured errors
error_human = 0.05   # 5% human error on the task
J_train     = 0.15   # 15% training error for your model
J_dev       = 0.20   # 20% dev error

error_bayes      = error_human
bias             = J_train - error_bayes
variance         = J_dev - J_train
reducible_error  = bias + variance

print(f"Bayes error ≈ human error: {error_bayes:.2%}")
print(f"Avoidable bias: {bias:.2%}")
print(f"Variance: {variance:.2%}")
print(f"Total reducible error: {reducible_error:.2%}")
print(f"Check: bayes + bias + variance = {error_bayes + reducible_error:.2%} ≈ {J_dev:.2%}")
```

**Engineering choices**:

- Measure human error on your exact dev or test set (same images, same questions).
- Use multiple annotators and majority vote to reduce label noise.
- Document the human-level benchmark as part of project guidelines.

### 5. Visualization / Geometry

```
Error decomposition on dev set
│
│ Bayes error (human):      ■■■■■                          5%
│ Avoidable bias:           ■■■■■■■■■■■■■■■■■■■■          10%
│ Variance:                 ■■■■■■■■■■■■■■■■■■■■■■■■■■■   5%
│
└───────────────────────────────────────────────────────────→ Components add to 20% dev error
```

- The human-level bar shows the irreducible floor.
- The bias bar is the “gap” from irreducible to train error.
- The variance bar is the “gap” from train to dev error.

### 6. Common Pitfalls & Tips

- Pitfall: Estimating human error on a small sample, yielding high variance.
    
    Tip: Use at least a few hundred examples and multiple annotators to stabilize the estimate.
    
- Pitfall: Using crowd-workers for specialized tasks (radiology, legal), where non-experts underperform.
    
    Tip: Benchmark with domain experts; non-expert error can overstate Bayes error.
    
- Pitfall: Treating human error as zero if experts always agree.
    
    Tip: Even experts disagree; measure the residual disagreement to approximate irreducible noise.
    

### 7. Interview-Ready Insights

- “I always ask, ‘How well do humans do on this exact task?’ That gives me a floor—if my model’s error is above that, I know I have avoidable bias or variance to tackle.”
- “Human-level error approximates the Bayes error; it tells me when to stop scaling up model size and instead focus on feature quality or domain-specific improvements.”
- “In medical imaging, expert radiologists achieve around 85–90% accuracy. If my model hits that range, further gains require new modalities or deeper clinical context, not just more data.”

### 8. Practice Exercises

1. **Compute Decomposition**
    - Take the MNIST dataset. Assume human error ≈ 0.5% (literature value).
    - Train a simple two-layer neural net, record training and dev error.
    - Compute bias, variance, and reducible error.
2. **Human Error Measurement**
    - Randomly select 300 examples from a classification dataset of your choice.
    - Have 3 friends label them. Compute majority-vote error vs. ground truth.
    - Report the human-level error estimate and confidence interval.
3. **Expert vs. Novice Comparison**
    - On a sentiment analysis task, compare labeled sentiment by domain experts vs. Mechanical Turk workers.
    - Compute error rates for both groups. Discuss which to use as your irreducible benchmark and why.
4. **Stopping Criterion**
    - Using UCI CIFAR-10, plot model dev error vs. training set size for a small CNN.
    - Mark the human-level error (around 5%) on the plot. Identify where the learning curve plateaus near human performance.

---

## Avoidable Bias

Avoidable bias is the portion of your model’s error that you can eliminate by choosing a more powerful hypothesis or model and by fitting it better. It’s the gap between the training error and the irreducible (Bayes) error, often approximated by human‐level error.

### 1. Direct Definition

Avoidable bias = training set error − Bayes error (approximated by human‐level error).

### 2. Concept Intuition

- What it is: the error due to underfitting—when your model is too simple or not trained enough to capture the underlying data patterns.
- Why it matters: identifying avoidable bias tells you when to increase model capacity, add features, or train longer. Without this insight, you might waste effort on regularization or data collection when you really need a bigger network.

### 3. Mathematical Breakdown

```python
# Let:
#   error_human  = human-level error ≈ Bayes error
#   J_train      = model’s training error rate

avoidable_bias = J_train - error_human
```

Breaking down:

- error_human: irreducible error floor (e.g., 5% for a vision task).
- J_train: fraction of misclassified examples on the train set.
- When avoidable_bias is high, you have room to reduce error by improving model fitting or capacity.

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import zero_one_loss

# 1. Load and split data
X, y = load_digits(return_X_y=True)
X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 2. Define human error (literature or measured)
error_human = 0.005  # 0.5% human error on digit recognition

# 3. Train a simple model
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=0)
model.fit(X_train, y_train)

# 4. Compute training error
y_train_pred = model.predict(X_train)
J_train = zero_one_loss(y_train, y_train_pred)  # fraction misclassified

# 5. Compute avoidable bias
avoidable_bias = J_train - error_human

print(f"Training error (J_train): {J_train:.2%}")
print(f"Avoidable bias   : {avoidable_bias:.2%}")
```

Use this pattern to measure your avoidable bias after any training run. If it’s large, consider:

- Increasing network width/depth
- Training more epochs
- Adding richer features

### 5. Visualization / Geometry

```
Error breakdown on training set
│
│ irreducible error (human): ■■■■        0.5%
│ avoidable bias         : ■■■■■■■■■■■■■■ 5.0%
│
└──────────────────────────────────────────→ Components of J_train = 5.5%
```

In a bar chart:

| Component | Error (%) |
| --- | --- |
| irreducible (Bayes) | 0.5 |
| avoidable bias | 5.0 |
| total training error (J_train) | 5.5 |

### 6. Common Pitfalls & Tips

- Pitfall: using zero human error as Bayes floor.Tip: always measure or use literature values for human‐level performance.
- Pitfall: attributing high dev error solely to variance without checking training error.Tip: if both train and dev error are high, bias is the culprit.
- Pitfall: increasing regularization when bias is dominant.Tip: reduce or remove regularization, or enlarge your model instead.

### 7. Interview-Ready Insights

- Definition: “Avoidable bias is the gap between training error and the irreducible error floor—shows how much error you can remove by using a more powerful model or better optimization.”
- Diagnosis: “Compute training error, subtract human‐level error; if the result is large, you need to boost capacity or train longer.”
- Real-world example: “In image classification, if your CNN’s train error is 8% but radiologists err at 2%, you have 6% avoidable bias—so you might deepen the network or improve feature maps.”

### 8. Practice Exercises

1. Train a logistic regression vs. a two‐layer neural network on MNIST. For each, compute J_train and avoidable bias assuming 0.5% human error. Interpret which model reduces avoidable bias best.
2. On a custom toy dataset (e.g., moons or circles), vary polynomial feature degree [1,2,3,4] with a logistic regressor. Plot training error vs. degree. Compute avoidable bias each time and identify the optimum capacity.
3. Implement a function `compute_avoidable_bias(model, X_train, y_train, error_human)` that returns the bias rate and logs a warning if bias > 5%, suggesting capacity increase. Use it in a simple grid search over hidden units [10,50,100].

---

## Understanding Human-Level Performance

Anchoring your model’s goals to how well humans perform on the exact same task and data gives you an empirical floor (Bayes error) and guides where to focus your efforts.

### 1. Direct Definition

Human-level performance is the accuracy (or error rate) achieved by expert human annotators on the same dataset and task. It approximates the irreducible (Bayes) error, serving as a benchmark for your model’s maximum attainable performance.

### 2. Concept Intuition

- Why measure it:
    - Establish your ceiling—if model error ≈ human error, you’re near irreducible noise.
    - Drive error analysis—separate avoidable bias (underfitting) from variance (overfitting).
    - Allocate effort—once you hit human-level, further gains require new data modalities or feature engineering, not just bigger networks.
- How to measure:
    - Select a representative subset of data examples.
    - Have multiple experts label each example independently.
    - Compare each label against the majority vote of the others (leave-one-out) to estimate error.

### 3. Mathematical Breakdown

```python
# m = number of examples
# k = number of human annotators per example
# labels[i][j] = label by annotator j on example i

def majority_vote(labels_i):
    counts = {}
    for label in labels_i:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)

human_mismatches = 0
total_labels     = m * k

for i in range(m):
    mv = majority_vote(labels[i])
    for j in range(k):
        if labels[i][j] != mv:
            human_mismatches += 1

error_human = human_mismatches / total_labels
# irreducible error ≈ error_human
```

- `error_human` approximates the Bayes error.
- Use this in error decomposition:
    
    ```python
    avoidable_bias = J_train - error_human
    variance       = J_dev   - J_train
    ```
    

### 4. Code & Practical Application

```python
from collections import Counter
import numpy as np

# Simulate 5 examples each labeled by 3 annotators
labels = [
    [1, 1, 0],  # two agree on 1
    [0, 0, 0],
    [2, 2, 2],
    [1, 2, 1],
    [0, 1, 0],
]

m = len(labels)
k = len(labels[0])

def majority_vote(lst):
    return Counter(lst).most_common(1)[0][0]

# Compute human error
mismatches = 0
for example in labels:
    mv = majority_vote(example)
    mismatches += sum(int(lab != mv) for lab in example)

error_human = mismatches / (m * k)
print(f"Estimated human error: {error_human:.2%}")

# Integrate into your training pipeline
# Assume J_train and J_dev from your model
J_train = 0.12  # 12% train error
J_dev   = 0.18  # 18% dev error

avoidable_bias = J_train - error_human
variance       = J_dev   - J_train

print(f"Avoidable bias: {avoidable_bias:.2%}")
print(f"Variance      : {variance:.2%}")
```

### 5. Visualization / Geometry

```
Error Decomposition on Dev Set (20%)

│ human-level (irreducible)  ■■■■■■■■■■■■■■■■■■■ 5%
│ avoidable bias            ■■■■■■■■■■■■■■■■■■■■■■■■■■ 7%
│ variance                  ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 8%
```

- The human-level bar marks the irreducible floor.
- Bias is the gap from human to train error.
- Variance is the train→dev gap.

### 6. Common Pitfalls & Tips

- Pitfall: too few examples → high variance in `error_human`.
    
    Tip: annotate at least several hundred examples for stability.
    
- Pitfall: non-experts inflate error.
    
    Tip: recruit domain experts or calibrate crowd labels with a gold subset.
    
- Pitfall: majority vote hides systematic bias.
    
    Tip: inspect disagreements—if experts split systematically, your irreducible error may be higher.
    

### 7. Interview-Ready Insights

- “I benchmark against experts to approximate Bayes error. If model error is still above human error, I focus on capacity or feature improvements; if it’s at human-level, I shift to richer data or domain knowledge.”
- “In NLP tasks, inter-annotator agreement of ~90% often means irreducible error ~10%. Models hitting 90% accuracy are essentially saturating the task.”
- “Sometimes models exceed average human performance by exploiting dataset artifacts. Always qualitatively inspect errors against human annotations.”

### 8. Practice Exercises

1. Collect human labels
    - Choose 200 examples from a classification dataset.
    - Have 3 friends label each. Compute `error_human` with leave-one-out majority vote.
2. Error decomposition
    - Train a small CNN on CIFAR-10.
    - Measure `error_human ≈ 5%` (literature). Compute avoidable bias and variance.
3. Annotator number vs. variance
    - Simulate 50 examples with k∈{1,3,5} annotators.
    - Plot standard deviation of estimated `error_human` vs. k. Decide how many annotators you need for <1% uncertainty.
4. Qualitative audit
    - For examples where human annotators disagree, inspect model predictions.
    - Are these edge-cases, label noise, or model mistakes? Document findings.

---

## Surpassing Human-Level Performance

Surpassing human‐level performance means your model’s error rate dips below the human annotator benchmark on the same task and data. It signals you’ve out-learned typical experts—if the result is genuine and generalizable.

### 1. Direct Definition

Surpassing human‐level performance occurs when

```python
error_model < error_human
```

where `error_model` is your model’s measured error on a held-out test set and `error_human` is the human-level error baseline.

### 2. Concept Intuition

- Why it matters:
    - You’ve closed the gap to Bayes error—or possibly found shortcuts that humans don’t use.
    - It flags diminishing returns on architecture scaling; further gains require new data or modalities.
    - It raises questions about fairness, trust, and unintended behaviors since models may exploit dataset artifacts.
- What it reveals:
    - True mastery of patterns in your data.
    - Potential overfitting to quirks—test whether performance holds under new conditions (out-of-distribution, adversarial).
    - A strategic inflection point: shift from brute-force scaling to robustness, interpretability, and monitoring.

### 3. Mathematical Breakdown

```python
# Given:
error_human = H   # e.g., 0.05 (5%)
error_model = M   # e.g., 0.04 (4%)

# Surpassed human level if:
M < H

# Margin of improvement:
delta = H - M      # 0.05 - 0.04 = 0.01 (1% absolute)
```

To test significance, use a paired bootstrap:

```python
# errors is an array of (model_correct, human_correct) booleans per example
diffs = model_correct.astype(int) - human_correct.astype(int)
# bootstrap diffs to get confidence interval on mean(diffs)
```

If the 95% CI of `mean(diffs)` lies entirely above zero, the model truly beats humans.

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.utils import resample

# Simulated correct flags for 1000 test examples
human_correct = np.random.binomial(1, 0.95, size=1000)
model_correct = np.random.binomial(1, 0.96, size=1000)

# Compute point estimate
error_human = 1 - human_correct.mean()
error_model = 1 - model_correct.mean()
print(f"Human error: {error_human:.2%}, Model error: {error_model:.2%}")

# Bootstrap significance test
n_boot = 10000
boot_diffs = []
for _ in range(n_boot):
    idx = resample(np.arange(1000))
    h = human_correct[idx].mean()
    m = model_correct[idx].mean()
    boot_diffs.append((m - h))
ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
print(f"95% CI of (model_acc - human_acc): [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**Industry tip**: Always freeze your test set and hold out a *second* sanity‐check set (or conduct an A/B test in production) to confirm gains aren’t due to data leakage.

### 5. Visualization / Geometry

```
Error Rate (%)
   ↑
6  │            ┌─────────┐
   │            │  Human  │   Bayes floor
5  │            │  Level  │ ■■■■■■
   │            └─────────┘
4  │       ■■■■■■  Model (test)
   │       ■■■■■■
3  │
   └────────────────────────────→ Model complexity / training size
```

- Human‐level bar shows irreducible floor.
- Model curve dipping below indicates surpassing human performance.
- Watch for over‐steep drops that may signal data artifacts rather than genuine generalization.

### 6. Common Pitfalls & Tips

- Pitfall: **Data leakage**—model sees test labels indirectly via preprocessing.
    
    Tip: Separate raw feature pipelines and audit your splits.
    
- Pitfall: **Evaluating on too small a test set**, making wins noise.
    
    Tip: Ensure ≥1 000 examples or use bootstrap CIs to verify.
    
- Pitfall: **Exploiting dataset artifacts** (watermarks in images, metadata in text).
    
    Tip: Conduct out-of-distribution tests or adversarial evaluations.
    
- Pitfall: **Ignoring real-world constraints**—model may be larger, slower, or costlier than human experts.
    
    Tip: Measure latency, memory, and cost per inference alongside accuracy.
    

### 7. Interview-Ready Insights

- “Beating human‐level error is a milestone but not the finish line. It can reflect true generalization or hidden shortcuts.”
- “I use bootstrap confidence intervals on paired human vs. model predictions to confirm significance.”
- “After surpassing human performance, I pivot to robustness tests: domain shift, adversarial attacks, fairness audits, and production A/B tests.”
- “Real‐world value demands measuring throughput and cost—human experts may adapt in real time, while models need retraining.”

### 8. Practice Exercises

1. Paired Bootstrap on MNIST
    - Train a small CNN and record per‐example correctness on test.
    - Simulate human correctness at 99.2%.
    - Bootstrap the difference in accuracy and report the 95% CI. Conclude if your model statistically beats humans.
2. Artifact Detection
    - On a subset of ImageNet, add a colored border to each image class.
    - Train a classifier that “beats human” by exploiting the border.
    - Demonstrate how out-of-distribution images (no border) collapse model performance below human level.
3. Latency vs. Accuracy Trade-off
    - Quantify human labeling speed (e.g., 5 images/sec).
    - Benchmark your best model’s inference throughput and latency.
    - Plot throughput vs. accuracy, and discuss at what point you’d choose model over human in production.
4. A/B Test Simulation
    - Simulate a task with human vs. model decision streams.
    - Assign 5 000 examples to “human pipeline” and 5 000 to “model pipeline.”
    - Compare error rates and compute p-value for difference.

---

## Improving Your Model Performance

Raising your model’s performance is an iterative, strategic process: you diagnose current errors, decide *which*improvements to make, and measure their impact—always in an orthogonal, reproducible way.

### 1. Direct Definition

Improving model performance means systematically reducing your chosen evaluation metric (e.g., error, loss) or increasing your metric of interest (e.g., accuracy, F1-score) by applying targeted changes to architecture, training procedure, data, or hyperparameters.

### 2. Concept Intuition

When your model sits at a plateau, you need a playbook:

- Diagnose: use error analysis to break down *why* your model misbehaves (high bias vs. high variance, specific classes, edge cases).
- Prioritize: rank potential fixes by expected impact and engineering cost.
- Iterate orthogonally: change one “axis” at a time—model size, learning rate, data augmentation—so you know exactly what moved the needle.
- Validate: always test on the frozen dev/test sets (or A/B in production) to guard against overfitting optimizations to stale data.

This mirrors how top ML teams ship reliable, incremental gains in production.

### 3. Mathematical Breakdown

### Error Decomposition

```python
# Let:
error_bayes   = human_error      # irreducible floor
J_train       = train_error
J_dev         = dev_error

avoidable_bias = J_train  - error_bayes
variance       = J_dev    - J_train
```

High bias → model too simple or under-trained.

High variance → model too complex or not enough data/regularization.

### L2 Regularization

Adds a penalty on weight magnitude to reduce overfitting:

```python
J_reg(w) = J(w) + (λ/2m) * sum_i w_i^2

# Gradient update (for weight w_j):
dw_j = ∂J/∂w_j + (λ/m)*w_j
w_j ← w_j - α * dw_j
```

- λ controls regularization strength
- α is learning rate

### Dropout

Randomly zeroes activations at rate `p` during training:

```python
# Forward pass for layer output a:
mask = np.random.binomial(1, 1-p, size=a.shape)
a_drop = a * mask
# At test time: scale activations by (1-p):
a_test = a * (1-p)
```

### 4. Code & Practical Application

Below is a baseline on Fashion-MNIST, then three orthogonal improvements:

1. Increase capacity
2. Add L2 regularization
3. Use data augmentation + dropout

```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 1. Load data
(x_train, y_train), (x_dev, y_dev) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_dev = x_train/255.0, x_dev/255.0
x_train = x_train[..., np.newaxis]
x_dev   = x_dev[..., np.newaxis]

# Helper: compile & train model
def train_model(model, epochs=10, use_aug=False):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    if not use_aug:
        return model.fit(x_train, y_train,
                         validation_data=(x_dev, y_dev),
                         epochs=epochs, verbose=2)
    # Data augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True
    )
    datagen.fit(x_train)
    return model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        validation_data=(x_dev, y_dev), epochs=epochs, verbose=2
    )

# Baseline model
baseline = models.Sequential([
    layers.Flatten(input_shape=(28,28,1)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
print("Baseline performance:")
hist_base = train_model(baseline)

# 1) Increase capacity
big_model = models.Sequential([
    layers.Flatten(input_shape=(28,28,1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64,  activation='relu'),
    layers.Dense(10,  activation='softmax')
])
print("\nLarger model performance:")
hist_big   = train_model(big_model)

# 2) Add L2 regularization to big model
l2_model = models.Sequential([
    layers.Flatten(input_shape=(28,28,1)),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(10, activation='softmax')
])
print("\nL2-regularized model performance:")
hist_l2    = train_model(l2_model)

# 3) Use dropout + data augmentation
aug_model = models.Sequential([
    layers.Flatten(input_shape=(28,28,1)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
print("\nAugmented + dropout model performance:")
hist_aug   = train_model(aug_model, use_aug=True)
```

Each training run holds other factors constant. You can then compare dev accuracies to see which change helped most.

### 5. Visualization / Geometry

```
Validation Accuracy vs. Epochs
┌───────────────────────────────────┐
│0.92 ●                     ● hist_aug
│0.90   ●                 ● hist_l2
│0.88     ●             ● hist_big
│0.86       ●         ● hist_base
│0.84         ●     ●
│0.82           ● ──●───────────────→ Epochs
└───────────────────────────────────┘
```

- Curves separated vertically show orthogonal improvements.
- Observe overfitting: if train ↑ but dev ↓, you need regularization or more data.

### 6. Common Pitfalls & Tips

- Pitfall: tuning many hyperparameters at once → confounded results.
    
    Tip: change one axis (capacity, regularization, data) per experiment.
    
- Pitfall: forgetting to freeze your dev/test split → you inadvertently overfit to it.
    
    Tip: serialize split IDs and preprocessing pipelines, then never touch them.
    
- Pitfall: using massive L2 (λ too large) → underfitting.
    
    Tip: sweep λ logarithmically (e.g., [1e-4, 1e-3, 1e-2]).
    
- Pitfall: data augmentation on dev/test → invalid evaluation.
    
    Tip: apply augmentation only to the training set.
    

### 7. Interview-Ready Insights

- “I start with a simple baseline, then incrementally add capacity, regularization, or data augmentation—measuring dev-set impact after each change.”
- “I decompose errors into bias vs. variance to decide whether to grow the model (if bias) or add regularization/data (if variance).”
- “I use grid/random search for hyperparameters but only after narrowing down which axis (learning rate, λ, dropout) matters most.”
- “In production, I also profile latency and memory—sometimes I trade a 0.2% accuracy boost for a 2× speedup by pruning or quantizing the model.”

### 8. Practice Exercises

1. **Bias vs. Variance Sweep**
    - On MNIST, train a 3-layer MLP with hidden sizes [32, 64, 128, 256].
    - Plot both train and dev error vs. model size. Identify where bias vs. variance dominate.
2. **Regularization Grid Search**
    - For your best MLP above, grid-search L2 λ over [1e-4,1e-3,1e-2,1e-1].
    - Keep dropout at 0.5. Plot dev accuracy vs. λ on a log scale.
3. **Data Augmentation Impact**
    - Implement random rotations, shifts, and flips on CIFAR-10.
    - Compare dev accuracy of a simple CNN with vs. without augmentation.
    - Quantify the % improvement and training-time overhead.
4. **Learning Rate Schedule**
    - Train a small CNN on Fashion-MNIST with:a) constant LR = 0.001b) exponential decay LR starting at 0.001, decay=0.9 per epochc) cosine annealing schedule
    - Plot dev loss vs. epoch for each and decide which schedule converges fastest with lowest loss.

---