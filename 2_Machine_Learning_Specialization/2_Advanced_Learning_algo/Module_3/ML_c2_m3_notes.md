# ML_c2_m3

## Topic: Evaluating a Model

### Pre-requisites

- Clean train, validation, and test splits
- Understanding of model outputs (class labels vs. probabilities vs. continuous values)
- Familiarity with basic Python and NumPy or pandas for data handling

### 1. Why Model Evaluation Matters

Accurate evaluation tells you how well your model will perform on unseen data.

- Guards against overfitting (model memorizes training data)
- Helps compare different algorithms or hyperparameter settings
- Informs business and safety decisions by quantifying performance

### 2. Data Splitting and Validation Strategies

### 2.1 Hold-Out Validation

```
Split data into:
  train (e.g. 60–80%)
  validation (e.g. 10–20%)
  test (e.g. 10–20%)
```

- Quick and simple
- Results depend on one random split

### 2.2 K-Fold Cross-Validation

```
for fold in 1..K:
  train on K-1 folds
  test on 1 fold
aggregate performance over all folds
```

- Reduces variance of estimate
- Common choice: K=5 or 10

### 2.3 Stratified Sampling

- Preserve class ratios in each fold for classification
- Ensures minority classes appear in every split

### 2.4 Nested Cross-Validation

- Outer loop for performance estimation
- Inner loop for hyperparameter tuning
- Prevents information leakage from validation to test

### 3. Common Pitfalls

- **Data Leakage**: using information from the test set during training
- **Temporal Leakage**: training on future data in time‐series tasks
- **Imbalanced Data**: accuracy can be misleading if one class dominates

### 4. Classification Metrics

### 4.1 Confusion Matrix

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

### 4.2 Accuracy & Error Rate

```
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
```

### 4.3 Precision & Recall

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

- Precision measures correctness of positive predictions
- Recall measures coverage of actual positives

### 4.4 F1 Score

```
F1 = 2 * (precision * recall) / (precision + recall)
```

- Harmonic mean of precision and recall
- Useful for imbalanced classes

### 4.5 ROC Curve & AUC

- **ROC Curve**: plot true positive rate vs. false positive rate at different thresholds
- **AUC**: area under ROC; 1.0 is perfect, 0.5 is random

### 4.6 Precision-Recall Curve

- Better for highly imbalanced data
- Plot precision vs. recall across thresholds

### 4.7 Specificity & Balanced Accuracy

```
specificity = TN / (TN + FP)
balanced_accuracy = (recall + specificity) / 2
```

- Captures performance on both classes equally

### 5. Regression Metrics

### 5.1 Mean Squared Error (MSE)

```
MSE = sum((y_pred - y_true)^2) / m
```

### 5.2 Root Mean Squared Error (RMSE)

```
RMSE = sqrt(MSE)
```

### 5.3 Mean Absolute Error (MAE)

```
MAE = sum(|y_pred - y_true|) / m
```

### 5.4 R-Squared (Coefficient of Determination)

```
R2 = 1 - sum((y_true - y_pred)^2)/sum((y_true - y_mean)^2)
```

- 1.0 means perfect fit; can be negative

### 5.5 Mean Absolute Percentage Error (MAPE)

```
MAPE = sum(|(y_true - y_pred)/y_true|) / m * 100
```

- Expressed as a percentage; watch out for zero targets

### 6. Calibration and Probabilistic Predictions

### 6.1 Calibration Curve

- Plot predicted probability vs. actual frequency
- Perfect calibration lies on the diagonal

### 6.2 Brier Score

```
Brier = sum((p_pred - y_true)^2) / m
```

- Lower is better; measures mean squared error of probability forecasts

### 7. Bias-Variance Tradeoff

### 7.1 Learning Curves

- Plot training vs. validation error as function of training set size
- High bias: both errors high and converge
- High variance: training error low, validation error high

### 7.2 Underfitting vs. Overfitting

- **Underfitting**: too simple model, high bias
- **Overfitting**: too complex model, high variance

### 8. Model Comparison & Statistical Tests

### 8.1 Paired t-Test on CV Scores

- Compare two models’ cross-validation scores
- Null hypothesis: no difference in means

### 8.2 Bootstrap Confidence Intervals

- Resample test set with replacement
- Compute metric on each sample
- Derive confidence intervals for the metric

### 9. Hyperparameter Tuning & Model Selection

### 9.1 Grid Search

```
for each combination in grid:
  evaluate via CV
select best parameters by highest metric
```

### 9.2 Random Search

- Sample hyperparameter combinations at random
- Often finds good settings faster

### 9.3 Bayesian Optimization

- Build a surrogate model of the metric
- Select next point to evaluate by acquisition function (e.g. Expected Improvement)

### 9.4 Nested Cross-Validation

- Inner CV for tuning
- Outer CV for unbiased performance estimate

### 10. Multi-Metric & Custom Metrics

- Combine precision, recall, and cost into a single score
- Example: weighted sum
    
    ```
    custom_score = 0.7*recall + 0.3*precision
    ```
    
- Cost-sensitive learning: penalize specific mistakes more heavily

### 11. Practical Deployment Considerations

### 11.1 Inference Latency & Throughput

- Measure time per prediction and batch throughput
- Balance model complexity with real‐time constraints

### 11.2 Model Size & Memory Footprint

- Count parameters and size on disk
- Use pruning or quantization for resource‐constrained environments

### 11.3 Fairness & Ethical Metrics

- Equal opportunity: equal recall across subgroups
- Demographic parity: equal positive rate across subgroups

### 11.4 Monitoring in Production

- Track data drift and performance metrics over time
- Set alerts for performance degradation

### 12. Tools & Code Examples

### 12.1 Scikit-Learn Example

```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Classification metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
```

### 12.2 TensorFlow/Keras Example

```python
import tensorflow as tf

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Evaluate on test set
test_metrics = model.evaluate(test_ds)
```

---

## Model Selection and Data Splitting Strategies

### 1. What Is Model Selection?

Model selection is the process of choosing the best algorithm and hyperparameter configuration that will generalize well to unseen data. It sits between training (fitting model parameters) and final evaluation (using the test set).

- Defines a search space of candidate models and their hyperparameters.
- Uses validation performance to compare and rank candidates.
- Finalizes the chosen model only after unbiased testing on the hold-out set.

### 2. Training, Validation, and Test Sets

Partition your dataset into three distinct subsets to prevent overfitting and obtain reliable performance estimates:

- Training Set
    - Used exclusively for fitting model parameters.
    - Commonly 60–80% of the total data.
- Validation Set
    - Guides model comparison and hyperparameter tuning.
    - Typically 10–20% of the data.
    - Should never leak into the test set.
- Test Set
    - Provides the final unbiased estimate of generalization error.
    - Only used once, after all tuning and selection are complete.
    - Usually 10–20% of the data.

Ensure stratification for classification problems to maintain class ratios across splits.

### 3. Cross-Validation Techniques

Cross-Validation (CV) yields more stable performance estimates, especially when data is limited:

- K-Fold CV
    - Split data into *K* equal folds.
    - Train on *K–1* folds; validate on the remaining fold.
    - Repeat for all folds and average metrics.
    - Common choices: *K* = 5 or 10.
- Stratified K-Fold
    - Maintains the class distribution in each fold.
    - Crucial for imbalanced classification tasks.
- Leave-One-Out CV
    - Each sample serves as the validation once.
    - Very low bias but high computational cost.

### 4. Nested Cross-Validation for Unbiased Tuning

Prevent “information leakage” from validation into final performance estimates:

1. **Outer Loop**: Split data into *K₁* folds.
2. **Inner Loop**: For each outer training split, perform K₂-fold CV to tune hyperparameters.
3. Evaluate the tuned model on the outer test fold.
4. Repeat for all outer folds and average results.

Nested CV provides an unbiased estimate of generalization after hyperparameter tuning.

### 5. Scikit-Learn Workflow Example

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Initial hold-out split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Define nested CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid,
    cv=inner_cv,
    scoring='accuracy'
)

# 3. Outer CV for unbiased performance
outer_scores = []

for train_idx, val_idx in outer_cv.split(X_trainval, y_trainval):
    X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

    grid.fit(X_tr, y_tr)
    best_model = grid.best_estimator_

    val_preds = best_model.predict(X_val)
    outer_scores.append(accuracy_score(y_val, val_preds))

print("Nested CV accuracy:", sum(outer_scores)/len(outer_scores))

# 4. Final fit on combined training+validation, then test
best_global = grid.best_estimator_
best_global.fit(X_trainval, y_trainval)
test_preds = best_global.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, test_preds))
```

### 6. Practical Tips

- Always set `random_state` for reproducibility.
- In time-series tasks, replace random splits with chronologically ordered train/validation/test sets.
- For very small datasets, favor cross-validation over a single hold-out split.
- Never peek at the test set until all tuning and selection are done.
- Log split ratios, seeds, and CV strategies for clarity—useful for interviews and reproducibility.

---

## Diagnosing Bias and Variance

### Pre-requisites

- Basic understanding of train/validation/test splits
- Familiarity with model error concepts (underfitting vs. overfitting)
- Python, scikit-learn, and Matplotlib or Seaborn for plotting

### 1. Fundamental Definitions

1. What is **Bias**?
    - Systematic error from overly simple models
    - Underfitting: model can’t capture underlying patterns
2. What is **Variance**?
    - Sensitivity of model predictions to different training sets
    - Overfitting: model learns noise as if it were signal
3. **Irreducible Noise**
    - Inherent randomness in the data
    - Cannot be removed by any model

### 2. Bias–Variance Decomposition

The expected squared prediction error at a point x can be split into three parts:

```
Expected_squared_error = Bias^2 + Variance + Irreducible_Noise
```

- **Bias²**Average squared difference between model’s predictions and the true values.
- **Variance**Variability of model predictions across different training samples.
- **Irreducible_Noise**Variance inherent to the data generation process.

### 3. Error Patterns

| Scenario | Training Error | Validation Error | Diagnosis |
| --- | --- | --- | --- |
| Underfitting | High | High (≈ training error) | High bias |
| Overfitting | Low | High (≫ training error) | High variance |
| Good Fit | Low | Low | Balanced model |

### 4. Diagnostic Plots

### 4.1 Learning Curves

Show error vs. training set size:

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    cv=5,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    scoring='neg_mean_squared_error'
)

train_err = -train_scores.mean(axis=1)
val_err   = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_err, label='Train Error')
plt.plot(train_sizes, val_err,   label='Validation Error')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.legend()
plt.show()
```

- **High Bias**: both curves flatten at high error
- **High Variance**: big gap between train and validation curves

### 4.2 Validation Curves

Show error vs. a hyperparameter:

```python
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1, 10]
train_scores, val_scores = validation_curve(
    model, X, y,
    param_name='alpha',
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)

train_err = 1 - train_scores.mean(axis=1)
val_err   = 1 - val_scores.mean(axis=1)

plt.semilogx(param_range, train_err, label='Train Error')
plt.semilogx(param_range, val_err,   label='Validation Error')
plt.xlabel('alpha')
plt.ylabel('Error')
plt.legend()
plt.show()
```

- **Increasing Complexity**
    - Both errors high → increase complexity
    - Train low, val high → reduce complexity or add regularization

### 4.3 Bootstrap Estimates

Estimate variance of error metric:

```python
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

scores = []
for _ in range(1000):
    X_bs, y_bs = resample(X_test, y_test)
    preds = model.predict(X_bs)
    scores.append(mean_squared_error(y_bs, preds))

print('MSE mean:', np.mean(scores))
print('MSE 95% CI:', np.percentile(scores, [2.5, 97.5]))
```

### 5. Remedies and Best Practices

### 5.1 If You See High Bias

- Increase model complexity (deeper trees, more features)
- Reduce regularization strength
- Add polynomial or interaction features
- Train neural nets longer or with more layers

### 5.2 If You See High Variance

- Simplify model (prune trees, reduce polynomial degree)
- Increase regularization (ridge, lasso, dropout)
- Gather more data or augment existing data
- Use ensemble methods (bagging to reduce variance)
- Feature selection to drop noisy inputs

### 6. Advanced Topics

1. **Regularization Path Analysis**
    - Plot coefficient shrinkage vs. penalty strength
2. **Bias–Variance in Ensembles**
    - Bagging reduces variance
    - Boosting reduces bias
3. **Neural Network Specifics**
    - Early stopping and dropout to control variance
    - Batch normalization to stabilize learning
4. **Estimating Irreducible Noise**
    - Use highly flexible models on large data to approximate noise floor

### 7. Automated Diagnostic Tools

- **Yellowbrick** LearningCurve and ValidationCurve visualizers
- **MLflow** or **TensorBoard** for tracking metrics over experiments
- **Eli5** for inspecting model weights and seeing overfitting signs

### 8. Interview-Ready Summary

1. Start with definitions of bias, variance, irreducible noise.
2. Show the decomposition formula in plain text.
3. Describe error patterns in train vs. validation.
4. Demonstrate diagnostic plots you’d build.
5. Offer precise, actionable remedies for each case.

### 9. Practical Example

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt

# 1. Split data
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 2. Fit ridge with varying alpha
alphas = [0.01, 0.1, 1, 10, 100]
train_err, val_err = [], []

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_tr, y_tr)
    train_err.append(((y_tr - model.predict(X_tr))**2).mean())
    val_err.append(((y_val - model.predict(X_val))**2).mean())

# 3. Plot
plt.semilogx(alphas, train_err, label='Train MSE')
plt.semilogx(alphas, val_err,   label='Validation MSE')
plt.xlabel('alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

- Low alpha: low bias, high variance
- High alpha: high bias, low variance

---

## Regularizing Bias and Variance

### Pre-requisites

- Understanding of bias–variance trade-off
- Familiarity with loss functions and gradient descent
- Basic Python and scikit-learn or TensorFlow for code examples

### 1. Recap of Bias and Variance

When you evaluate a model:

- High bias means underfitting: model too simple to capture patterns
- High variance means overfitting: model learns noise as if it were signal

Regularization shifts the bias–variance balance:

- Increasing regularization strength reduces variance but raises bias
- Decreasing regularization strength reduces bias but raises variance

### 2. Regularization Techniques to Reduce Variance

### 2.1 L2 Regularization (Ridge)

Adds a penalty proportional to the square of weights to the loss:

```
loss = original_loss + lambda * sum(W[i]^2 for i in all weights)
```

Gradient update with weight decay:

```
W = W - lr * (dW + lambda * W)
```

- Tends to shrink weights smoothly
- Improves generalization by discouraging large weights

### Code Snippet (scikit-learn)

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)  # alpha is lambda
model.fit(X_train, y_train)
```

### 2.2 L1 Regularization (Lasso)

Penalizes the absolute value of weights, encouraging sparsity:

```
loss = original_loss + lambda * sum(abs(W[i]) for i in all weights)
```

- Drives some weights exactly to zero
- Useful for feature selection

### Code Snippet (scikit-learn)

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.01)
model.fit(X_train, y_train)
```

### 2.3 Elastic Net

Combines L1 and L2 penalties:

```
loss = original_loss
     + alpha * l1_ratio * sum(abs(W))
     + alpha * (1 - l1_ratio) * sum(W^2)
```

- Balances sparsity and weight smoothing
- Tune `alpha` (strength) and `l1_ratio` (mix)

### Code Snippet (scikit-learn)

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, y_train)
```

### 2.4 Dropout (Neural Networks)

Randomly zeroes a fraction of activations each training step:

```
during training:
  A = activation(Z)
  mask = random_uniform(shape=A.shape) > dropout_rate
  A_dropout = A * mask
during inference:
  A_dropout = A * (1 - dropout_rate)
```

- Prevents co-adaptation of neurons
- Acts like an ensemble of sub-networks

### Code Snippet (TensorFlow)

```python
import tensorflow as tf

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

### 2.5 Data Augmentation

Artificially enlarge dataset by transforming inputs:

- Images: random flips, crops, rotations, color jitter
- Text: synonym replacement, back-translation
- Time series: jitter, scaling, slicing

Reduces variance by exposing model to more diverse examples.

### Code Snippet (Keras ImageDataGenerator)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  horizontal_flip=True
)
train_gen = aug.flow(X_train, y_train, batch_size=32)
```

### 2.6 Early Stopping

Halt training when validation loss stops improving:

```python
monitor = 'val_loss'
patience = 5
if no improvement in val_loss for patience epochs:
    stop training and restore best weights
```

- Prevents over-training on noisy patterns

### Code Snippet (Keras Callback)

```python
callback = tf.keras.callbacks.EarlyStopping(
  monitor='val_loss',
  patience=3,
  restore_best_weights=True
)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=50, callbacks=[callback])
```

### 2.7 Batch Normalization

Normalizes layer inputs to zero mean and unit variance:

```
mu = mean(x_batch)
sigma2 = var(x_batch)
x_norm = (x_batch - mu) / sqrt(sigma2 + eps)
y = gamma * x_norm + beta
```

- Stabilizes and accelerates training
- Has a slight regularizing effect

### Code Snippet (Keras)

```python
tf.keras.layers.BatchNormalization()
```

### 3. Techniques to Reduce Bias

When underfitting, you can:

- Increase model complexity
    - Deepen neural network, add more neurons
    - Use higher-degree polynomial features
- Decrease regularization strength
    - Lower lambda in L1/L2
- Add or engineer better features
- Switch to more expressive models
    - From linear to tree-based or ensembles

### Example: Removing L2 Penalty

```python
model = Ridge(alpha=0.0)  # equivalent to ordinary least squares
```

### 4. Advanced Regularization Approaches

- Weight decay vs explicit L2: weight decay applies decay directly during updates
- DropConnect: randomly zeroes weights instead of activations
- Mixup: train on convex combinations of inputs and labels
- Cutout, CutMix for image regularization
- Adversarial training as a regularizer

### 5. Choosing Regularization Strength

- Use cross-validation to select lambda or dropout rate
- Plot validation error vs regularization hyperparameter
- Monitor training and validation curves for under/overfitting patterns

### 6. Common Pitfalls

- Over-regularizing leads to underfitting
- Forgetting to disable dropout at inference
- Mixing up weight decay with L2 penalty in optimizers
- Applying batch norm incorrectly on small batch sizes

### 7. Interview Tips

- Explain how L2 regularization adds a term to the Hessian, making it better conditioned
- Derive the weight update rule with weight decay
- Compare L1 vs L2 effects on feature selection and model sparsity

---

## Establishing a Baseline Level of Performance

### 1. Why a Baseline Matters

Before you build complex models, you need a reference point. A solid baseline helps you:

- Quantify how much benefit your fancy model actually brings
- Detect data leaks or evaluation mistakes early
- Choose sensible performance targets for iterative improvements

### 2. Key Steps to Build a Baseline

1. Define the problem type and select the evaluation metric.
2. Choose one or more trivial models or heuristics as your baseline.
3. Evaluate them rigorously (cross‐validation or hold-out).
4. Record and visualize baseline results.
5. Use those results to set “must-beat” thresholds for future models.

### 3. Problem Types & Typical Baselines

| Task | Metric | Baseline Model |
| --- | --- | --- |
| Regression | RMSE, MAE | Predict mean (\bar{y}); DummyRegressor (mean/median) |
| Binary Class. | Accuracy, F1, AUC | Predict majority class; DummyClassifier (“most_frequent”) |
| Multi-class | Accuracy, F1-macro | Predict most frequent label |
| Ranking | NDCG, MAP | Random ranking; rank by a single feature |
| Time Series | MAPE, RMSE | Naïve forecast: (y_{t+1}=y_t) |

### 4. Code Examples

### 4.1 Regression Baseline with scikit-learn

```python
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Dummy regressor that always predicts the training mean
baseline = DummyRegressor(strategy="mean")

scores = cross_val_score(
    baseline, X_train, y_train,
    scoring="neg_root_mean_squared_error",
    cv=5
)

print("Baseline RMSE:", -np.mean(scores))
```

### 4.2 Classification Baseline

```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

# Always predict the most frequent class
clf_baseline = DummyClassifier(strategy="most_frequent")

scores = cross_val_score(
    clf_baseline, X_train, y_train,
    scoring="f1",
    cv=5
)

print("Baseline F1-score:", scores.mean())
```

### 5. Visualizing & Interpreting Baselines

- Plot training vs. validation error of your baseline.
- Compare with a “random” or “feature-only” baseline if you have domain heuristics.
- If your baseline outperforms a simple model (e.g., a single decision tree), you know something’s off with your implementation or data.

### 6. Interview-Ready Talking Points

- Explain why beating a baseline is the first sanity check in any ML pipeline.
- Discuss how a trivial model (e.g., mean predictor) may inadvertently capture strong class imbalance or leakage.
- Show how cross-validation on a dummy model can expose data-splitting issues.

---

## Learning Curves

Learning curves show how a model’s performance evolves as you vary the amount of training data or the number of training iterations. They’re essential for diagnosing whether you need more data, a more complex model, or stronger regularization.

### 1. What Is a Learning Curve?

A learning curve plots a performance metric (e.g., accuracy, loss, RMSE) on the y-axis against:

- The size of the training set (data-based curve)
- The number of training epochs or iterations (time-based curve)

By comparing training and validation curves, you can see where your model underfits or overfits.

### 2. Why Use Learning Curves?

- Detect underfitting when both curves converge at poor performance
- Detect overfitting when training performance is much better than validation
- Estimate gains from adding more data or training longer
- Guide decisions on model complexity and regularization

### 3. Typical Shapes and Diagnoses

| Scenario | Training Curve | Validation Curve | Diagnosis | Remedy |
| --- | --- | --- | --- | --- |
| High bias | Plateaus at high loss | Mirrors training | Underfitting | Increase model capacity, reduce reg. |
| High variance | Low loss | High loss | Overfitting | Add regularization, get more data |
| Good fit | Low loss, stable | Low loss, converging | Balanced bias-variance | Maintain settings, consider slight reg. |
| Data deficiency | Large gap, both high | Slowly decreasing | Not enough training examples | Gather more data |

### 4. Data-Based Learning Curves with scikit-learn

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Define model and metric
model = RandomForestClassifier(n_estimators=100)
train_sizes = np.linspace(0.1, 1.0, 5)

# Compute learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)
val_std    = np.std(val_scores, axis=1)

# Plot
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std,
                 alpha=0.1, color='C0')
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std,
                 alpha=0.1, color='C1')
plt.plot(train_sizes, train_mean, 'o-', color='C0', label='Training score')
plt.plot(train_sizes, val_mean,   'o-', color='C1', label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### 5. Epoch-Based Learning Curves with Keras

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Build simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train with validation split
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

# Plot loss
plt.plot(history.history['loss'],    label='Training loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### 6. Interpreting the Curves

- If training and validation loss both plateau high: model too simple
- If training loss is low but validation loss rises: model overfits
- If validation loss keeps decreasing slowly: more data may help
- Watch for noisy validation curves—consider smoothing or larger batches

### 7. Common Pitfalls

- Using non-representative subsets for small training sizes
- Forgetting to shuffle or stratify when sampling training fractions
- Relying on a single split—prefer cross-validation for stability
- Misreading metric scales (e.g., accuracy vs. log-loss)

---

## Bias, Variance, and Neural Networks

### 1. Overview of the Bias–Variance Tradeoff

High bias means your model is too simple and underfits the data, yielding large training and validation errors.

High variance means your model is too flexible, overfits training noise, and performs poorly on unseen data.

In neural networks, capacity (depth, width, activation) governs where you sit on the bias–variance spectrum.

Every design choice—from architecture to regularization—shifts this balance.

### 2. Manifestation in Neural Networks

- Underfitting (high bias): shallow networks, excessive regularization, few parameters.
- Overfitting (high variance): very deep/wide networks, insufficient regularization, noisy labels.
- Modern NNs can interpolate data (zero training error) yet still generalize, thanks to implicit regularization.
- The *double descent* curve shows that after a certain capacity, test error can drop again.

### 3. Diagnosing Bias vs. Variance

1. Learning curves
    - Plot training vs. validation loss or accuracy over epochs or data size.
2. Weight distribution
    - Excessively large weights hint at overfitting.
3. Sensitivity to noise
    - Add label noise; high-variance models’ performance degrades sharply.
4. Validation gap
    - Wide gap = variance; both high = bias.

### 4. Techniques to Reduce Variance

| Technique | Effect on Variance | Additional Notes |
| --- | --- | --- |
| L2 Weight Decay | Shrinks weights smoothly | (L = L_0 + \lambda \sum_i w_i^2) |
| Dropout | Breaks co-adaptation | Drop rate 0.2–0.5 typical |
| Batch Normalization | Slight implicit regularizer | Normalizes activations per mini-batch |
| Data Augmentation | Exposes model to diversity | Images, text, time-series transforms |
| Early Stopping | Stops before overfit | Monitor validation loss with patience |
| Ensembling | Averages multiple models | Bagging or snapshot ensembles |

### 4.1 L2 Weight Decay in Keras

```python
import tensorflow as tf

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### 4.2 Dropout and Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

model = tf.keras.Sequential([
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')
])

early_stop = EarlyStopping(monitor='val_loss', patience=5,
                           restore_best_weights=True)

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, validation_split=0.2,
          epochs=100, callbacks=[early_stop])
```

### 5. Techniques to Reduce Bias

- Increase capacity
    1. Add layers or neurons
    2. Use richer activations (e.g., Swish, GELU)
- Decrease regularization strength (lower (\lambda), dropout rate)
- Feature engineering and embedding rich representations
- Pretrain or transfer-learn from larger models

### 6. The Double Descent Phenomenon

After the interpolation threshold (zero training error), increasing capacity can actually *reduce* test error again.

This challenges the classic U-shaped bias–variance curve in high-capacity regimes.

Implication: very large neural nets can generalize well even when they have far more parameters than data points.

### 7. Interview Talking Points

- Derive how weight decay adds (\lambda I) to the Hessian, improving conditioning.
- Explain dropout as an approximate model ensemble.
- Discuss implicit regularization by SGD and batch norm.
- Contrast classical bias–variance tradeoff with modern double descent.

---

## Iterative ML Development Loop: Building a Spam Classifier

### 1. Problem Definition & Success Criteria

Defining the goal clearly sets the stage for everything that follows.

- Business objective: Filter out unwanted spam messages from user inboxes.
- ML goal: Predict whether a message is “spam” (1) or “ham” (0).
- Success metrics:
    - Precision on the spam class (minimize false positives).
    - Recall on the spam class (minimize false negatives).
    - F1-score as a single consolidated metric.
- Minimum acceptable threshold: F1-score ≥ 0.90 on held-out test set.

### 2. Data Collection & Management

Gathering and versioning your dataset prevents surprises later.

- Data source: public “SMS Spam Collection” CSV or Enron email dataset.
- Snapshot the raw CSV into a data folder under version control.
- Record dataset schema: columns “label” and “text”.
- Ensure reproducible splits by fixing random seeds.

### 3. Exploratory Data Analysis & Preparation

Understand your data before modeling.

- Class imbalance: count of ham vs spam.
- Text length distribution: boxplots of token counts.
- Common words per class: word cloud or top-20 tokens.
- Clean and normalize:
    - Lowercase, strip punctuation, collapse whitespace.
    - Remove or encode URLs, email addresses, numbers.
- Feature engineering:
    1. Bag-of-words with unigrams and bigrams.
    2. TF-IDF weighting to down-weigh common terms.
    3. Optionally, add hand-crafted features (e.g., message length, presence of “free” or “win”).
- Split data:
    
    ```python
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    ```
    

### 4. Baseline Model & Quick Win

Set a “must-beat” performance using trivial heuristics.

- DummyClassifier predicting the majority class:
    
    ```python
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report
    
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_feat, y_train)
    preds = baseline.predict(X_test_feat)
    print(classification_report(y_test, preds))
    ```
    
- Record results:
    
    
    | Metric | Baseline Score |
    | --- | --- |
    | Precision | 0.00 |
    | Recall | 0.00 |
    | F1-score | 0.00 |
- Any non-zero F1 shows your real model is adding value.

### 5. Model Selection & Training

Iterate through increasingly powerful models.

### 5.1 Pipeline with TF-IDF + Logistic Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("clf", LogisticRegression(C=1.0, solver="liblinear"))
])

pipeline.fit(X_train, y_train)
```

### 5.2 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "tfidf__max_df": [0.75, 1.0],
    "clf__C": [0.1, 1, 10]
}

search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)
best_model = search.best_estimator_
```

### 6. Evaluation & Diagnostics

Thoroughly analyze where your model succeeds and fails.

- Classification report on test set:
    
    ```python
    from sklearn.metrics import classification_report
    
    preds = best_model.predict(X_test)
    print(classification_report(y_test, preds))
    ```
    
- Confusion matrix heatmap.
- Precision–Recall curve to choose operating threshold:
    
    ```python
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    proba = best_model.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    ```
    
- Learning curves to spot under/overfitting:
    
    ```python
    from sklearn.model_selection import learning_curve
    import numpy as np
    
    sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=5, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    ```
    

### 7. Deployment & Monitoring

Put the model into production and watch it in action.

- Serialize with `joblib.dump(best_model, "spam_clf.pkl")`.
- Build a minimal Flask API:
    
    ```python
    from flask import Flask, request
    import joblib
    
    app = Flask(__name__)
    model = joblib.load("spam_clf.pkl")
    
    @app.route("/predict", methods=["POST"])
    def predict():
        text = request.json["text"]
        label = model.predict([text])[0]
        return {"label": int(label)}
    
    if __name__ == "__main__":
        app.run()
    ```
    
- Monitor:
    - Track daily spam rate and false-positive alerts.
    - Log misclassified samples for human review.
    - Alert on data drift (e.g., vocabulary shifts).

### 8. Feedback Loop & Continuous Improvement

Keep improving your classifier based on real-world feedback.

- Collect manually corrected labels and append to training data.
- Schedule periodic retraining pipelines (e.g., monthly).
- Introduce new features: word embeddings, message metadata.
- Evaluate concept drift by comparing feature distributions over time.

---

## Error Analysis in Machine Learning

### 1. Direct Summary

Error analysis is the systematic process of examining model mistakes—quantitatively and qualitatively—to uncover their root causes, prioritize fixes, and iteratively improve performance.

### 2. Why Error Analysis Matters

- Reveals blind spots that global metrics (e.g., accuracy or RMSE) can hide.
- Guides targeted interventions—feature engineering, data collection, model adjustments.
- Ensures you’re fixing the most impactful errors first for maximum ROI.

### 3. Core Steps in Error Analysis

1. **Collect Errors**
    - Generate predictions on a hold-out or validation set.
    - Record instances where predictions differ from true labels.
2. **Quantify & Categorize**
    - Build a confusion matrix or residual distribution.
    - Group errors by type: false positives vs. false negatives, high-residual buckets, etc.
3. **Slice-Based Analysis**
    - Define data slices: by feature value (e.g., short vs. long messages), user segment, timestamp.
    - Compute error metrics per slice to spot weak areas.
4. **Qualitative Inspection**
    - Manually review representative misclassified examples.
    - Note patterns: ambiguous text, typos, formatting issues.
5. **Hypothesize Root Causes**
    - Link error patterns to data issues, feature gaps, model capacity, or bias.
6. **Prioritize Interventions**
    - Estimate error volume × business impact for each category.
    - Tackle high-priority buckets first.
7. **Implement & Validate Fixes**
    - Engineer new features, augment data, adjust class weights.
    - Re-evaluate to confirm error reduction on targeted slices.

### 4. Quantitative Techniques

| Technique | Purpose |
| --- | --- |
| Confusion Matrix | Counts true/false positives & negatives |
| Precision–Recall Curve | Evaluates balance at different thresholds |
| ROC Curve | Measures trade-off between TPR and FPR |
| Calibration Curve | Checks probability estimates vs. truth |
| Error Rate by Slice Table | Compares error rates across segments |

### 5. Visualization Examples

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham","spam"])
disp.plot(cmap="Blues")
plt.show()
```

### 6. Slice-Based Error Analysis

```python
import pandas as pd

df = pd.DataFrame({
    "text": X_test_texts,
    "true": y_test,
    "pred": y_pred
})

# Define a slice: messages shorter than 20 chars
short_slice = df[df["text"].str.len() < 20]
error_rate = (short_slice["true"] != short_slice["pred"]).mean()
print(f"Error rate on short messages: {error_rate:.2%}")

```

### 7. Qualitative Inspection Checklist

- Is the text too short or too long?
- Does it contain unusual tokens (URLs, slang, emojis)?
- Are there systematic typos or obfuscations (e.g., “Fr33”, “w1n”)?
- Could domain knowledge help (e.g., common spam phrases)?

### 8. Advanced Tools for Root-Cause Analysis

- **SHAP Values**: identify which features drive a prediction for each error.
- **LIME**: locally interpretable explanations for individual mispredictions.
- **Embedding Visualization**: t-SNE or UMAP to cluster correct vs. incorrect.
- **Error Typology**: build a taxonomy of error causes and tag examples accordingly.

### 9. Prioritization Framework

1. Estimate volume of each error type (count or rate).
2. Assign a business impact score (e.g., cost per false positive).
3. Compute a priority score = volume × impact.
4. Rank error categories and allocate your development time accordingly.

### 10. Continuous Error Monitoring

- Integrate error-slice dashboards (Plotly Dash, Grafana) into your CI/CD pipeline.
- Set alerts when error rates spike on critical slices (e.g., VIP users).
- Automate sample collection of recent mispredictions for weekly review.

---

## Adding Data: Expanding and Enriching Your Dataset

### 1. Why Add More Data?

Increasing the training set often reduces variance, improves generalization, and uncovers rare patterns.

- More examples help neural networks learn complex decision boundaries.
- Fresh data can correct bias caused by unbalanced or outdated samples.
- Diminishing returns kick in—monitor performance gains per new batch.

### 2. Data Sources

Where to get additional labeled or unlabeled samples:

- Internal logs and user-generated content (emails, chat transcripts).
- Public datasets (Kaggle, UCI, academic corpora).
- Third-party APIs (social media streams, web scraping with ethical and legal review).
- Crowdsourcing platforms (Amazon Mechanical Turk, Labelbox).
- Partnerships or data marketplaces.

### 3. Ensuring Data Quality

Adding noisy or biased samples can hurt more than help. Key practices:

- Label auditing: sample new entries for manual verification.
- De-duplication and canonicalization to avoid redundant examples.
- Balanced sampling if classes are skewed (oversample minority or undersample majority).
- Schema and format checks woven into ingestion pipelines.

### 4. Data Augmentation Techniques

### 4.1 Text Augmentation

```python
import nlpaug.augmenter.word as naw

augmenter = naw.SynonymAug(aug_src='wordnet')
augmented_text = augmenter.augment("Win free tickets now!")  # e.g., "Win complimentary tickets now!"
```

- Synonym replacement, random swap, back-translation.
- Be cautious: heavy augmentation can introduce semantic drift.

### 4.2 Image Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
train_gen = aug.flow(X_train, y_train, batch_size=32)
```

- Crop, flip, color jitter, CutMix, Mixup.
- Useful for vision-based spam filtering (e.g., image attachments).

### 4.3 Time-Series & Tabular

- Jitter: add small noise to numeric features.
- Window slicing for sequences.
- SMOTE or ADASYN to synthetically balance classes.

### 5. Synthetic Data Generation

When real data is scarce or privacy-sensitive:

- **GANs** for images and even text embeddings.
- **Language models** (e.g., GPT) to generate plausible text messages.
- **Simulation engines** to create user-behavior logs.

Pitfalls: distribution mismatch, hallucinations, and potential privacy leaks.

### 6. Active Learning & Semi-Supervised Learning

Maximize annotation ROI by labeling only the most informative samples:

1. **Uncertainty sampling**: pick examples where the model’s confidence is lowest.
2. **Query by committee**: compare disagreement among multiple models.
3. **Pseudo-labeling**: assign labels to high-confidence unlabeled data, then retrain.

These methods can halve your labeling costs while boosting performance.

### 7. Data Versioning & Pipelines

As data grows, track versions and lineage:

- Store raw snapshots in object stores (e.g., S3, GCS).
- Use tools like DVC, MLflow, or Delta Lake for dataset version control.
- Automate ingestion, validation, and feature computation (Airflow, Prefect, Kubeflow).

### 8. Monitoring, Feedback & Drift Detection

New data can shift your model’s operational envelope:

- Continuously monitor feature distributions vs. train set (e.g., Kolmogorov–Smirnov test).
- Track live performance metrics and error-slice rates on incoming data.
- Set alerts for vocabulary shifts, class-ratio changes, or API-level schema breaks.

### 9. Impact on Bias–Variance

- Adding diverse data primarily cuts variance by smoothing decision boundaries.
- Targeted data collection (e.g., for rare spam patterns) can also address bias.
- Always re-evaluate learning curves after each data increment to quantify gains.

### 10. Interview-Ready Talking Points

- Explain how data augmentation artificially increases sample diversity.
- Contrast active learning vs. random sampling for labeling efficiency.
- Discuss synthetic data risks: overfitting to generated artifacts vs. real-world validity.
- Describe how data versioning ensures reproducibility and auditability.

---

## Transfer Learning: Leveraging Data from a Different Task

### 1. Direct Answer

Transfer learning lets you reuse knowledge (features or weights) learned on a source task to boost performance on your target task—especially when your target data is scarce or expensive to label.

### 2. When and Why to Use Transfer Learning

- You have limited labeled data for your target problem.
- A large, related dataset (e.g., ImageNet, Wikipedia) is already modeled.
- Your tasks share underlying structure (e.g., object shapes in images, language patterns in text).
- You want to accelerate training and improve generalization.

### 3. Types of Transfer Learning

| Category | Definition |
| --- | --- |
| Inductive TL | Source and target tasks differ; target data is labeled. |
| Transductive TL | Source and target tasks are the same; domains differ (e.g., different feature space). |
| Unsupervised TL | Neither source nor target tasks have labels; transfer unsupervised representations. |

### 4. Common Transfer Approaches

1. Feature Extraction
    - Freeze a pre-trained model’s layers.
    - Use its internal activations as fixed embeddings.
    - Train only a new classifier head on your data.
2. Fine-Tuning
    - Initialize with pre-trained weights.
    - Unfreeze some top layers (or the whole network).
    - Retrain at a lower learning rate to adapt representations.
3. Adapter Modules
    - Insert small trainable layers between frozen layers.
    - Retain most pre-trained weights unchanged.
    - Achieve efficient task adaptation with fewer parameters.
4. Multi-Task Learning
    - Train jointly on source and target tasks with shared backbone.
    - Leverage simultaneous gradients to regularize representations.

### 5. Workflow Steps

1. **Select Pre-trained Model**
    - Vision: ResNet, EfficientNet, MobileNet, Vision Transformer
    - NLP: BERT, RoBERTa, GPT, T5
2. **Decide Transfer Strategy**
    - Feature extraction if data < 1,000 samples
    - Fine-tuning for moderate data (1k–10k)
    - Full retraining only if target dataset ≫ source
3. **Prepare Your Data**
    - Resize/crop images to model’s input size
    - Tokenize/text-normalize to match pre-trained tokenizer
    - Align label formats (one-hot, integer indices)
4. **Configure Training**
    - Choose optimizers and learning rates: typically 10× lower than training from scratch
    - Set up early stopping to guard against overfitting
    - Use regularization (dropout, weight decay) to smooth adaptations
5. **Evaluate and Iterate**
    - Monitor validation metrics closely
    - Experiment with different numbers of unfrozen layers
    - Compare against training from scratch to quantify transfer gains

### 6. Code Examples

### 6.1 Keras/TensorFlow: Image Classification

```python
import tensorflow as tf

# Load pre-trained ResNet50 without top
base = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(224,224,3)
)
base.trainable = False  # feature extraction

# Add your classifier head
inputs = tf.keras.Input(shape=(224,224,3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=10)
```

### 6.2 PyTorch: Fine-Tuning a Transformer for Text

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Unfreeze last two encoder layers
for name, param in model.named_parameters():
    if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# Training loop (simplified)
for batch in DataLoader(train_dataset, batch_size=16):
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs, labels=batch['label'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 7. Common Pitfalls

- Negative transfer: source and target are too dissimilar, harming performance.
- Catastrophic forgetting: fine-tuning erases useful pre-trained features.
- Overfitting when unfreezing too many layers on small datasets.
- Mismatched preprocessing (e.g., tokenization or image normalization).

### 8. Interview Talking Points

- Contrast feature extraction vs. fine-tuning and when to choose each.
- Discuss learning-rate scaling and layer freezing strategies.
- Explain how adapter modules or LoRA can reduce parameter costs.
- Describe domain adaptation techniques (e.g., adversarial alignment).

---

## Full Cycle of a Machine Learning Project

### 1. Problem Definition & Business Understanding

Clarifying the goal and success criteria ensures alignment across teams.

- Align on the business objective (e.g., increase click-through rate by 5% or reduce churn by 10%).
- Translate into an ML task: classification, regression, ranking, or anomaly detection.
- Define target metric(s) and acceptable thresholds (AUC ≥ 0.85, RMSE ≤ 2.0, F1 ≥ 0.90).
- Identify constraints: latency, interpretability, compliance, cost of errors.

### 2. Data Acquisition & Versioning

Gathering, storing, and tracking data is the foundation of reproducibility.

1. **Identify Sources**
    - Internal databases, logs, event streams, third-party APIs, or public datasets.
2. **Ingest & Store**
    - Snapshot raw inputs in object storage (S3, GCS) or a data lake.
    - Tag each snapshot with a version ID, timestamp, and schema.
3. **Version Control**
    - Use tools like DVC or Delta Lake to track data lineage.
    - Record transformations, cleaning steps, and label sources.

### 3. Exploratory Data Analysis & Preprocessing

Understanding your data early prevents surprises later.

- Visualize distributions, correlations, missing values, and class imbalance.
- Clean and normalize: handle missing entries, correct typos, standardize formats.
- Split data reproducibly: train/validation/test or time-based splits for time series.
- Automate checks (schema, null rates, outlier detection) in your pipeline.

### 4. Feature Engineering & Selection

Transform raw data into meaningful signals for your model.

- Create domain-informed features: date parts, aggregations, text embeddings, or interaction terms.
- Encode categorical variables: one-hot, target encoding, or embeddings.
- Scale numeric features (standardization or min–max) if required by your algorithms.
- Use feature‐selection techniques (variance thresholding, mutual information, or L1-based selection) to reduce noise.

### 5. Baseline Modeling

Establish simple reference models to set “must-beat” benchmarks.

- Regression: predict the mean or median using `DummyRegressor`.
- Classification: predict majority class with `DummyClassifier`.
- Quick-win models: linear models or shallow decision trees on a core feature set.
- Record baseline metrics and visualize their training/validation curves.

### 6. Model Selection & Training

Iterate through a range of algorithms and tune hyperparameters.

1. **Choose Candidates**
    - Tree-based (Random Forest, XGBoost), linear models, SVMs, or neural networks.
2. **Set Up Cross-Validation**
    - k-fold, stratified, or time-series splits based on data characteristics.
3. **Hyperparameter Search**
    - Grid search, random search, or Bayesian optimization with tools like Optuna or Hyperopt.
4. **Experiment Tracking**
    - Log parameters, metrics, and artifacts in MLflow, Weights & Biases, or a custom dashboard.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, None]
}
search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)
```

### 7. Evaluation & Error Analysis

Deeply diagnose errors to guide next steps.

- Compute confusion matrix, precision–recall curve, ROC curve, and calibration plots.
- Perform slice-based analysis: error rates across user segments, time buckets, or feature ranges.
- Conduct qualitative review of representative mispredictions to uncover common patterns.
- Prioritize fixes by error volume × business impact.

### 8. Model Deployment & Serving

Turn your trained model into a production service.

- Serialize artifacts (`joblib.dump`, `torch.save`, or Keras `.h5`).
- Containerize the model server with Docker or use serverless inference (AWS Lambda, Azure Functions).
- Expose prediction endpoints via REST, gRPC, or streaming pipelines.
- Implement batch scoring pipelines for large-scale inference.

```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(payload: dict):
    features = payload["features"]
    label = model.predict([features])[0]
    return {"prediction": int(label)}
```

### 9. Monitoring & Maintenance

Ensure your model remains accurate and reliable over time.

- Track live metrics: prediction latency, throughput, error rates, and business KPIs.
- Monitor data drift and concept drift via statistical tests or embedding distances.
- Log incoming data distributions and alert when deviations exceed thresholds.
- Capture real-world labels (ground truth) to measure actual performance.

### 10. Feedback Loop & Continuous Improvement

Use production insights to refine your model and pipeline.

- Schedule automated retraining when data drift or performance degradation is detected.
- Incorporate newly labeled examples—especially from high-impact error slices.
- Revisit feature engineering and algorithm choices based on error patterns.
- Update success criteria and business alignment as objectives evolve.

---

## Fairness, Bias, and Ethics in Machine Learning

### 1. Why Fairness and Ethics Matter

Machine learning systems increasingly shape decisions in hiring, lending, policing, and healthcare. If they embed unfair biases or opaque logic, they can perpetuate discrimination, erode trust, and cause real harm. Embedding fairness and ethics throughout your ML workflow safeguards users, aligns with legal requirements, and strengthens the societal impact of your models.

### 2. Sources of Bias

- Data Collection
    - Sample bias: under- or over-representation of certain groups
    - Label bias: human annotators’ prejudices reflected in training labels
- Feature Engineering
    - Proxy features: innocuous-looking variables that correlate with sensitive traits (e.g., ZIP code → race)
- Algorithmic Learning
    - Optimization objectives that prioritize overall accuracy over subgroup performance
    - Model complexity that overfits majority-group patterns
- Deployment Context
    - Feedback loops: model outputs influencing future data (credit scores, policing patrols)

### 3. Fairness Definitions & Metrics

| Fairness Notion | Definition | When to Use |  |  |
| --- | --- | --- | --- | --- |
| Statistical Parity | P(ŷ=1 | A=“group1”) ≈ P(ŷ=1 | A=“group2”) | Hiring rate parity across demographics |
| Equalized Odds | TPR(A=“group1”) ≈ TPR(A=“group2”) and FPR(A=“group1”) ≈ FPR(A=“group2”) | Criminal risk assessment, equal error rates |  |  |
| Equal Opportunity | TPR(A=“group1”) ≈ TPR(A=“group2”) | Focus on equal recall for the positive class |  |  |
| Predictive Parity | PPV(A=“group1”) ≈ PPV(A=“group2”) | Loan default prediction, consistent precision |  |  |
| Calibration | P(Y=1 | ŷ, A=“group1”) ≈ P(Y=1 | ŷ, A=“group2”) | Probabilistic forecasts across subgroups |
| Individual Fairness | Similar individuals receive similar predictions | Personalized pricing or recommendations |  |  |

### 4. Detecting Bias in Your Pipeline

1. **Data Audit**
    - Compare feature distributions and label rates across sensitive groups.
    - Visualize with histograms, boxplots, or cumulative distribution plots.
2. **Metric Evaluation**
    - Compute fairness metrics on your validation set.
    - Track overall and subgroup performance side by side.
3. **Error Slice Analysis**
    
    ```python
    import pandas as pd
    
    df = pd.DataFrame({
      "pred": preds,
      "true": y_test,
      "gender": demo["gender"]
    })
    error_by_group = (
      df.assign(error=lambda d: d.pred != d.true)
        .groupby("gender")["error"]
        .mean()
    )
    print(error_by_group)
    ```
    
4. **Automated Tools**
    - Fairlearn for visualizations and mitigation strategies.
    - IBM AI Fairness 360 for pre-, in-, and post-processing algorithms.

### 5. Bias Mitigation Techniques

### 5.1 Pre-processing

- Reweighing: assign weights to samples so groups are balanced.
- Data augmentation: oversample under-represented groups or synthesize new examples.

### 5.2 In-processing

- Fairness regularizers: add penalty terms to your loss (e.g., equality of odds loss).
- Adversarial debiasing: train a model to predict the sensitive attribute from embeddings and penalize that ability.

### 5.3 Post-processing

- Threshold optimization: learn group-specific decision thresholds to equalize metrics.
- Reject option classification: defer decisions on examples near the decision boundary.

### Code Snippet: Fairlearn Threshold Optimizer

```python
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression

# Train a base classifier
clf = LogisticRegression().fit(X_train, y_train)

# Optimize thresholds for equalized odds
opt = ThresholdOptimizer(
    estimator=clf,
    constraints="equalized_odds",
    prefit=True
)
opt.fit(X_val, y_val, sensitive_features=demo_val["race"])

preds_oo = opt.predict(X_test, sensitive_features=demo_test["race"])
```

### 6. Broader Ethical Considerations

- Privacy and Consent
    - Minimize personally identifiable information
    - Employ differential privacy or federated learning when needed
- Transparency and Explainability
    - Provide model cards or data sheets describing limitations and intended use
    - Use interpretable models or explanation tools (SHAP, LIME) for high-stakes decisions
- Accountability and Governance
    - Define clear ownership for model outcomes
    - Establish review boards or ethical committees for periodic audits
- Human Oversight
    - Implement “human-in-the-loop” for appeal or manual review on critical cases

### 7. Interview-Ready Talking Points

- Trade-offs between fairness metrics: e.g., statistical parity vs. equalized odds cannot both hold under unequal base rates.
- Examples of real-world failures: COMPAS recidivism score bias, facial recognition misdiagnosis across skin tones.
- Regulatory landscape: GDPR’s right to explanation, proposed EU AI Act risk categories.
- Ethical frameworks: Principles from ACM’s Code of Ethics or IEEE’s Ethically Aligned Design.

---

## Error Metrics for Skewed Datasets

### Pre-requisites

- Understanding of classification basics and confusion matrix
- Familiarity with train/validation/test splits
- Python with scikit-learn for code examples

### 1. Why Skewed Data Is Challenging

When one class (e.g., “negative”) far outweighs the other (e.g., “positive”), accuracy becomes misleading. A model that always predicts the majority class can have high accuracy but zero utility for detecting the rare class.

### 2. Confusion Matrix Refresher

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

### 3. Core Metrics

### 3.1 Precision

- Fraction of positive predictions that are correct.

```
precision = TP / (TP + FP)
```

### 3.2 Recall (Sensitivity)

- Fraction of actual positives detected.

```
recall = TP / (TP + FN)
```

### 3.3 F1-Score

- Harmonic mean of precision and recall.

```
f1 = 2 * (precision * recall) / (precision + recall)
```

### 3.4 Specificity

- Fraction of actual negatives correctly identified.

```
specificity = TN / (TN + FP)
```

### 3.5 Balanced Accuracy

- Average of recall and specificity.

```
balanced_accuracy = (recall + specificity) / 2
```

### 4. Threshold-Independent Metrics

### 4.1 ROC AUC

- Area Under the Receiver Operating Characteristic curve.
- Plots true positive rate (recall) vs. false positive rate (1 – specificity) as threshold varies.
- Good overall ranking metric but can mask poor performance on rare class.

### 4.2 Precision–Recall AUC

- Area under the precision–recall curve.
- More informative when positives are scarce.
- Baseline PR AUC for a random model equals positive prevalence.

### 5. Advanced Single-Number Metrics

### 5.1 Matthews Correlation Coefficient (MCC)

- Correlation between observed and predicted binary labels; handles imbalance well.

```
mcc = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
```

### 5.2 Cohen’s Kappa

- Agreement measure that adjusts for chance agreement.

```
total = TP + TN + FP + FN
po = (TP + TN) / total
pe = ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)) / (total*total)
kappa = (po - pe) / (1 - pe)
```

### 6. Sample-Weighted Metrics

When positive class is very rare, weight examples inversely to class frequency.

```python
from sklearn.metrics import f1_score

# y_true and y_pred are arrays of 0/1
f1 = f1_score(y_true, y_pred, average='weighted')
```

- `average='macro'`: unweighted mean of per-class scores
- `average='micro'`: global counts to compute precision/recall

### 7. Cost-Sensitive and Custom Metrics

### 7.1 Weighted Loss

In model training, assign higher penalty to misclassifying rare class.

```
loss = - (w_pos * y*log(p) + w_neg * (1-y)*log(1-p))
```

- `w_pos = total / (2 * positives)`
- `w_neg = total / (2 * negatives)`

### 7.2 Custom Business Metric

If false positives cost $10 and false negatives cost $100:

```
cost = 10*FP + 100*FN
```

### 8. Calibration Metrics

For skewed data with probability outputs, check calibration.

### 8.1 Brier Score

- Mean squared error of predicted probabilities.

```
brier = sum((p_pred - y_true)^2) / N
```

### 8.2 Expected Calibration Error (ECE)

1. Bin predictions by confidence (e.g., 10 bins).
2. Compute |accuracy_in_bin - avg_confidence_in_bin| for each bin.
3. Weighted average by bin size.

### 9. Multi-Class Imbalance

Extend metrics to more than two classes:

- Per-class precision, recall, F1 (report as table).
- Macro-averaged: average of per-class metrics.
- Weighted-averaged: weighted by support (number of true instances).
- One-vs-rest ROC AUC for each class.

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_true, y_pred,
    target_names=['class0','class1','class2'],
    digits=4
))
```

### 10. Practical Code Example

```python
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

# Compute confusion matrix values
cm = confusion_matrix(y_true, y_pred)
TP = cm[1,1]; TN = cm[0,0]
FP = cm[0,1]; FN = cm[1,0]

# Basic metrics
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)
bal_acc = (rec + TN/(TN+FP)) / 2

# AUC metrics
roc_auc = roc_auc_score(y_true, y_scores)
pr_auc  = average_precision_score(y_true, y_scores)

print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"Bal Acc:   {bal_acc:.3f}")
print(f"ROC AUC:   {roc_auc:.3f}")
print(f"PR AUC:    {pr_auc:.3f}")
```

### 11. Common Pitfalls

- Relying on accuracy when classes are imbalanced.
- Interpreting ROC AUC without considering base rates.
- Using macro-average when one class dominates support.
- Ignoring calibration errors when decisions depend on probability thresholds.

### 12. Best Practices

- Report a combination of metrics: precision, recall, F1, PR AUC, MCC.
- Use stratified splits or cross-validation to preserve class ratios.
- Tune thresholds based on business costs or desired recall/precision trade-off.
- Visualize confusion matrix and PR/ROC curves to see trade-offs.

### 13. Interview-Ready Summary

1. Explain why accuracy fails on skewed data.
2. Define precision, recall, F1, balanced accuracy, and MCC with code-block formulas.
3. Compare ROC AUC vs. PR AUC and when to use each.
4. Discuss calibration (Brier score, ECE) when probabilities matter.
5. Show how to weight metrics and optimize thresholds for business impact.

---

## Trading Off Precision and Recall

Most directly, you control the balance between precision and recall by adjusting the probability threshold for your classifier, examining the precision–recall curve, and picking the cutoff that best matches your business or performance goals (often via an F-beta score or recall/precision constraint).

## Prerequisites

- Familiarity with precision and recall
- Ability to generate prediction probabilities (`predict_proba`)
- Basic use of NumPy and scikit-learn

### 1. Role of the Decision Threshold

Every probabilistic classifier outputs a score (p\in[0,1]). By default you label as positive when (p \ge 0.5). Changing that cutoff shifts precision and recall:

- Raising the threshold → fewer positives → precision ↑, recall ↓
- Lowering the threshold → more positives → precision ↓, recall ↑

### 2. Precision–Recall Curve

Compute precision and recall at every possible threshold to visualize trade-offs.

```python
from sklearn.metrics import precision_recall_curve

# y_true: true labels, y_scores: model probabilities for positive class
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
```

- `precisions[i]` and `recalls[i]` correspond to `thresholds[i]`
- Plot `recalls` on x-axis, `precisions` on y-axis

### 3. Composite Metric: F-beta Score

Use F-beta to weight recall ((\beta>1)) or precision ((\beta<1)):

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)

f_beta = (1 + beta**2) * (precision * recall) \
         / (beta**2 * precision + recall)
```

- F₁ (beta=1): equal weight
- F₂ favors recall twice as much as precision
- F₀.₅ favors precision twice as much as recall

### 4. Finding the Optimal Threshold

### 4.1 Maximize F₁

```python
import numpy as np

# compute F1 at each threshold
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
best_idx   = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
```

### 4.2 Meet Recall Constraint

```python
desired_recall = 0.8
# find last threshold where recall >= desired_recall
idx = np.where(recalls >= desired_recall)[0][-1]
threshold_for_recall = thresholds[idx]
precision_at_recall = precisions[idx]
```

### 5. Visualizing the Trade-off

```python
import matplotlib.pyplot as plt

plt.plot(recalls, precisions, label='PR Curve')
plt.scatter(recalls[best_idx], precisions[best_idx],
            marker='o', color='red',
            label=f'Best F1 @ {best_thresh:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.legend(); plt.show()
```

### 6. Threshold Selection Strategies

- Maximize F-beta that reflects business priorities
- Impose minimum recall (or precision) and pick highest precision (or recall) above that bar
- Use cost-based threshold:choose threshold minimizing expected cost
    
    ```
    cost = C_fp * FP + C_fn * FN
    ```
    
- Calibrated thresholds via cross-validation to avoid overfitting

### 7. Code Example: End-to-End

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

# Train
model = LogisticRegression().fit(X_train, y_train)
y_scores = model.predict_proba(X_val)[:,1]

# Compute curve
precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)

# Compute F1 and pick threshold
f1_scores = 2 * precisions * recalls / (precisions + recalls)
best_idx = np.argmax(f1_scores)
opt_thresh = thresholds[best_idx]

print(f"Optimal threshold for max F1: {opt_thresh:.3f}")
print(f"Precision at optimal threshold: {precisions[best_idx]:.3f}")
print(f"Recall at optimal threshold:    {recalls[best_idx]:.3f}")
```

### 8. Interview-Ready Summary

1. Explain that threshold tuning shifts precision/recall balance.
2. Show how to compute and plot a precision–recall curve.
3. Define the F-beta formula in code blocks and describe its impact.
4. Demonstrate finding the threshold that maximizes F-beta or meets a fixed recall/precision.
5. Mention cost-based threshold selection and calibration considerations.

---