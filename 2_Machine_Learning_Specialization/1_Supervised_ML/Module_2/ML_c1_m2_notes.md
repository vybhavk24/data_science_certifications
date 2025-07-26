# ML_c1_w2

## Multiple Linear Regression: Handling Multiple Features

### 1. Problem Formulation

We now extend univariate linear regression (one feature) to multiple features (n features). Our goal is to fit an n-dimensional hyperplane that best predicts y from inputs x¹, x², …, xⁿ.

### 2. Hypothesis Representation

Using a design matrix X ∈ ℝᵐˣ⁽ⁿ⁺¹⁾ (m examples, n features + bias):

```
h_θ(x) = θ₀·1 + θ₁·x¹ + θ₂·x² + … + θₙ·xⁿ
```

Vectorized form:

```
h_θ(x) = X · θ
```

- X’s first column is all ones (for θ₀, the intercept).
- θ ∈ ℝⁿ⁺¹ is the parameter vector.

### 3. Cost Function

We measure fit with mean squared error over m examples:

$$ J(θ) ;=; \frac{1}{2m}\sum_{i=1}^{m}\bigl(h_θ(x^{(i)}) - y^{(i)}\bigr)^2 $$

Vectorized:

$$ J(θ) = \frac{1}{2m},(X θ - y)^T (X θ - y) $$

### 4. Gradient Descent Updates

To minimize (J(θ)), we update all parameters in lock-step using the gradient:

$$ θ := θ - α;\frac{1}{m};X^T,(X θ - y) $$

- (α) is the learning rate.
- Each update uses all m examples (batch GD).

### 5. Intuition: Hyperplane in n-Dimensional Space

- Think of each feature axis as a coordinate in n-dimensional space.
- θ₁…θₙ define the orientation of the hyperplane; θ₀ shifts it up/down.
- Gradient (X^T(Xθ - y)) computes how much to tilt/shove the hyperplane along each feature axis.
- Subtracting the scaled gradient nudges the plane toward smaller error.

### 6. Python Implementation

```python
import numpy as np

# Assume X_raw: m×n feature matrix, y: m×1 target vector
m, n = X_raw.shape

# 1. Add bias term
X = np.c_[np.ones((m, 1)), X_raw]      # shape: (m, n+1)

# 2. Initialize parameters
theta = np.zeros(n+1)

# 3. Cost function
def compute_cost(X, y, theta):
    errors = X.dot(theta) - y
    return (1/(2*m)) * errors.T.dot(errors)

# 4. Batch gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    cost_history = []
    for _ in range(num_iters):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# 5. Run training
alpha = 0.01
iters = 500
theta_final, costs = gradient_descent(X, y, theta, alpha, iters)
print("Trained parameters:", theta_final)
```

### 7. Practical Tips

- **Feature Scaling**: Normalize each feature to zero mean and unit variance. This prevents large-scale features dominating the gradient and speeds up convergence.
- **Learning Rate Tuning**: Start with α ∈ [0.001, 0.1]; adjust by monitoring cost descent.
- **Convergence Check**: Stop when the change in cost between iterations falls below a small threshold ε.
- **Compare to Normal Equation**: For small n, you can solve$$θ = (X^T X)^{-1} X^T y$$in one step—no iteration required.

---

## Vectorization – Part 1: Hypothesis and Cost in Matrix Form

---

### Overview

Vectorization means replacing explicit loops over examples or features with matrix and vector operations. In linear regression, instead of summing errors one example at a time, we use whole-array math. This not only makes code concise but leverages highly optimized linear algebra libraries for speed.

By thinking in terms of matrices and vectors, you treat your entire dataset as one object. This shift in perspective reveals patterns and operations at the “big picture” level, so you can derive formulas and code that naturally scale to thousands or millions of examples.

### Key Math and Formulas

### Hypothesis (Predictions)

```
# Vectorized prediction for all m examples
predictions = X · theta
```

- X is an m×(n+1) matrix (first column ones for bias)
- theta is an (n+1)×1 parameter vector
- predictions is an m×1 vector of hθ(x⁽ⁱ⁾)

### Cost Function

```
J(theta) = (1 / (2 * m)) * (predictions - y)^T · (predictions - y)
```

- y is the m×1 vector of true labels
- (predictions - y) is the m×1 error vector
- Transpose and multiply sums squared errors in one step

### Gradient (Preview for Next Part)

```
gradient = (1 / m) * X^T · (predictions - y)
```

This single product computes all partial derivatives ∂J/∂θⱼ at once.

### Naïve Loop vs. Vectorized Code

### Loop-Based Cost Computation

```python
def compute_cost_loop(X, y, theta):
    m = len(y)
    total_error = 0
    for i in range(m):
        pred = theta[0] + theta[1] * X[i,1]  # for single feature
        total_error += (pred - y[i])**2
    return total_error / (2 * m)
```

### Vectorized Cost Computation

```python
def compute_cost_vec(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)                      # shape: (m, 1)
    errors = predictions - y                        # shape: (m, 1)
    return (1 / (2 * m)) * errors.T.dot(errors)     # scalar
```

The vectorized version handles all examples in two or three lines versus dozens of loop iterations.

### Shapes and Dimensions

- X: m×(n+1) (m examples, n features + bias)
- theta: (n+1)×1
- predictions: m×1
- errors: m×1
- gradient: (n+1)×1

Keeping track of these shapes ensures your dot products align and flags mistakes early.

### Why Vectorization Matters in Practice

- **Performance**: Matrix operations use optimized C/Fortran libraries (BLAS/LAPACK).
- **Readability**: Code closely follows math, making it easier to verify.
- **Scalability**: One line can handle millions of examples or thousands of features.
- **Maintainability**: Fewer chances for indexing errors when no loops are involved.

---

## Vectorization – Part 2: Gradient and Parameter Updates

### Recap and Goals

You’ve seen how to compute predictions and cost in one fell swoop using matrix operations. Now, let’s vectorize the gradient descent step so we update all parameters simultaneously without any explicit loops. This will prepare you for large-scale ML workflows and interview questions on efficient implementations.

### Vectorized Gradient Formula

In component form, each parameter update is:

```
θⱼ := θⱼ − α * (1/m) * Σᵢ [ (hθ(x⁽ⁱ⁾) − y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾ ]
```

Vectorized, that becomes:

```
gradient = (1 / m) * Xᵀ · (predictions − y)    # shape: (n+1, 1)
θ := θ − α * gradient
```

- Xᵀ is (n+1)×m
- (predictions − y) is m×1
- gradient matches θ’s shape: (n+1)×1

This single matrix multiply computes all partial derivatives ∂J/∂θⱼ in one go.

### Python Implementation

```python
import numpy as np

def gradient_descent_vec(X, y, theta, alpha, num_iters):
    m = len(y)
    history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = X.dot(theta)               # shape: (m, 1)
        errors = predictions - y                 # shape: (m, 1)
        gradient = (1 / m) * X.T.dot(errors)     # shape: (n+1, 1)
        theta = theta - alpha * gradient         # update all θ at once

        history[i] = compute_cost_vec(X, y, theta)

    return theta, history
```

Key points:

- `X.T.dot(errors)` replaces a loop over features.
- No indexing by θⱼ; linear algebra does it for you.
- Recording cost in `history` helps visualize convergence.

### Shapes and Sanity Checks

- X: m×(n+1)
- y: m×1
- theta: (n+1)×1
- predictions: m×1
- errors: m×1
- gradient: (n+1)×1

Always print `X.shape`, `theta.shape`, and `gradient.shape` early in debugging to catch mismatches.

---

## Gradient Descent for Multiple Linear Regression

### Pre-requisites

You should be comfortable with:

- Basic linear algebra (vectors, dot products, matrix multiplication)
- The hypothesis and cost function for univariate linear regression
- Python and NumPy fundamentals

If any of these feel shaky, review simple vector–matrix multiplication and the single-feature cost function first.

### Intuition: Descending the Error Surface

Imagine you’re hiking down into a valley blindfolded. At each step you feel the slope under your feet and take a small step downhill. In gradient descent, our “elevation” is the cost (mean squared error), and our “position” is the parameter vector θ. We compute the slope (gradient) and move a bit in the negative gradient direction to reduce error.

### Key Formulas

```python
# Hypothesis (predictions for all m examples)
predictions = X.dot(theta)           # shape: (m, 1)

# Cost function J(θ)
errors = predictions - y             # shape: (m, 1)
J = (1 / (2 * m)) * errors.T.dot(errors)   # scalar

# Gradient vector for all n+1 parameters
gradient = (1 / m) * X.T.dot(errors)      # shape: (n+1, 1)

# Update rule
theta = theta - alpha * gradient
```

- X is an m×(n+1) matrix (column of 1s for bias + n features)
- y is an m×1 vector of targets
- θ is an (n+1)×1 parameter vector
- α (alpha) is the learning rate

Each update shifts θ slightly to reduce the overall error.

### Why It Works

- errors.T.dot(errors) computes Σᵢ(hθ(x⁽ⁱ⁾) − y⁽ⁱ⁾)² in one shot
- X.T.dot(errors) produces a vector whose jth entry is Σᵢ(hθ(x⁽ⁱ⁾) − y⁽ⁱ⁾)·xⱼ⁽ⁱ⁾
- Dividing by m averages the effect, so the step size α controls convergence speed

Vectorization bundles all partial derivatives and example loops into highly optimized linear algebra calls.

### Real-World ML Workflow

1. Load and split your dataset (train/validation/test).
2. Normalize features (mean zero, unit variance) so gradient descent converges smoothly.
3. Initialize θ (often zeros or small random values).
4. Run gradient descent, track cost each iteration, and stop when improvements plateau.
5. Evaluate on validation/test sets; adjust α or add regularization if needed.

Under the hood, many libraries (scikit-learn’s `LinearRegression` by default uses normal equations but can switch to gradient-based solvers) follow this pattern.

### Visual Insight

If n=1 (two parameters: bias θ₀ and slope θ₁), J(θ₀,θ₁) forms a convex bowl. Contour lines are ellipses. At each iteration, the gradient vector points uphill. Moving opposite takes you toward the center (minimum). For n>1, the intuition holds in higher dimensions.

### Python Practice Problems

1. Implement gradient descent for a tiny synthetic dataset with two features:
    
    ```python
    import numpy as np
    
    # Dataset: [size_in_sqft, num_bedrooms] → price
    X = np.array([[1, 800, 2],
                  [1, 1500, 3],
                  [1, 1200, 2],
                  [1, 2000, 4]], dtype=float)  # first col=1 for bias
    y = np.array([[150],
                  [260],
                  [220],
                  [340]], dtype=float)
    
    theta = np.zeros((3,1))
    alpha = 0.0001
    num_iters = 1000
    # Task: write gradient descent loop, print final theta and cost_history plot
    ```
    
    Hint: remember to record cost every 100 iterations and plot using matplotlib.
    
2. Experiment with different learning rates (e.g., 0.00001, 0.001, 0.01). Observe how α affects convergence speed and stability.
3. Extend your code to include L2 regularization (ridge). Adjust the gradient and cost formulas to include (λ/m)*θ for j≥1.

---

## Feature Scaling – Part 1: Why and When to Scale Features

### Pre-requisites

Before diving in, ensure you’re comfortable with:

- Basic statistics (mean, standard deviation, min, max)
- Vector and matrix notation (feature matrix X, shape m×n)
- Gradient descent mechanics (why gradient steps depend on feature scale)

### Why Feature Scaling Matters

When features have wildly different ranges—say, “age” in [0,100] and “income” in [0,100,000]—gradient descent updates become unbalanced:

- Large-scale features dominate the gradient and force tiny learning rates.
- Convergence slows or oscillates along steep dimensions.
- Distance-based algorithms (K-means, K-nearest neighbors) bias toward large-magnitude features.

Feature scaling brings all inputs to comparable ranges, improving:

- Speed of convergence in gradient-based optimizers
- Stability and conditioning of the optimization problem
- Performance of distance-sensitive methods

### Common Scaling Techniques

1. **Standardization (Z-score)**
    
    Centers data to zero mean and scales to unit variance:
    
    ```
    x_scaled = (x - mean(x)) / std(x)
    ```
    
    - mean(x): average of the feature
    - std(x): standard deviation of the feature
2. **Min–Max Scaling**
    
    Rescales features to a fixed range [0, 1] (or [a, b]):
    
    ```
    x_scaled = (x - min(x)) / (max(x) - min(x))
    ```
    
3. **Mean Normalization**
    
    Centers data around zero but keeps range proportional:
    
    ```
    x_scaled = (x - mean(x)) / (max(x) - min(x))
    ```
    

### How Scaling Helps in ML Workflows

- **Gradient Descent:**
    
    Balanced feature scales let you pick a larger learning rate without divergence.
    
- **Regularization:**
    
    L1/L2 penalties apply equally across features when scales match.
    
- **Distance-Based Models:**
    
    K-means and KNN treat all dimensions fairly, avoiding bias toward large-range features.
    
- **Feature Selection:**
    
    You can compare weights or importance directly when features share scale.
    

### Visualization Example

Below we create two synthetic features with different ranges and plot before/after standardization.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Synthetic data
np.random.seed(0)
x1 = 100 * np.random.rand(100)       # range [0,100]
x2 = np.random.rand(100)             # range [0,1]
X = np.column_stack((x1, x2))

# Standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Plot before scaling
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c='blue')
plt.xlabel('x1 (0–100)'); plt.ylabel('x2 (0–1)')
plt.title('Before Scaling')

# Plot after scaling
plt.subplot(1,2,2)
plt.scatter(X_std[:,0], X_std[:,1], c='green')
plt.xlabel('x1_scaled'); plt.ylabel('x2_scaled')
plt.title('After Standardization')
plt.tight_layout()
plt.show()
```

### Python Implementation Snippet

Implement standardization and min–max scaling from scratch:

```python
import numpy as np

def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu) / sigma

def min_max_scale(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

# Usage
X_std = standardize(X)
X_mm  = min_max_scale(X)
```

### Practice Problems

1. **Scale and Train**
    - Load the Boston housing dataset’s features.
    - Split into train/test sets.
    - Apply standardization on training features and transform test features.
    - Train linear regression with gradient descent (use scaled data) and compare convergence to unscaled data.
2. **Compare Scaling Methods**
    - Take any 3-feature dataset (e.g., diabetes).
    - Apply standardization, min–max, and mean normalization.
    - For each, train K-means (K=3) and visualize cluster assignments.
    - Analyze how scaling choice alters cluster shapes.
3. **Custom Scaling Function**
    - Write a function `scale_features(X, method, params)` that accepts
        - `method` ∈ {'standard', 'minmax', 'meannorm'}
        - `params` dict (e.g., range for min–max)
    - Return scaled X and parameters needed to inverse transform.

---

## Feature Scaling – Part 2: Implementations and Advanced Techniques

### 1. Integrating Scaling into Gradient Descent

Before running gradient descent, scale your feature matrix once—don’t rescale every epoch. Vectorized update on standardized data looks like this:

```python
# Precompute scaling parameters
mu    = X.mean(axis=0)
sigma = X.std(axis=0)
Xs    = (X - mu) / sigma

# Gradient descent on scaled Xs
m, n = Xs.shape
theta = np.zeros(n)
alpha = 0.01
for _ in range(num_iters):
    gradient = (1/m) * Xs.T.dot(Xs.dot(theta) - y)
    theta   = theta - alpha * gradient
```

Storing `mu` and `sigma` lets you inverse‐transform predictions back to the original scale:

```python
y_pred_original = Xs.dot(theta) * sigma_y + mu_y
```

### 2. Handling Categorical Features

When you one-hot encode categorical variables, each dummy feature naturally lives in {0,1}. You only need to scale **numerical** columns so they align with those dummies:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose  import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = ['age','income']
cat_cols = ['gender','region']

preprocessor = ColumnTransformer([
  ('num', StandardScaler(),   num_cols),
  ('cat', OneHotEncoder(),   cat_cols)
])

pipeline = Pipeline([
  ('preprocess', preprocessor),
  ('model',      LinearRegression())
])
```

This avoids distortions and prevents dummy variables from skewing distance or coefficient magnitudes.

### 3. Advanced Scalers Comparison

| Scaler | Method | Pros | Cons |
| --- | --- | --- | --- |
| StandardScaler | (x − µ)/σ | Centers mean to 0, unit variance | Sensitive to outliers |
| MinMaxScaler | (x − min)/(max − min) | Bounds data to [0,1] | Outliers compress other values |
| RobustScaler | (x − median)/IQR | Robust against outliers | Outputs unbounded |
| PowerTransformer | Box-Cox or Yeo-Johnson | Reduces skew, makes data more normal | Box-Cox needs positive data; complexity |
| QuantileTransformer | Rank-based mapping to uniform/normal | Shapes arbitrary distributions | Can distort feature relationships |

### Applying Them in Code

```python
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer

scalers = {
  'robust':          RobustScaler(),
  'yeo_johnson':     PowerTransformer(method='yeo-johnson'),
  'quantile_norm':   QuantileTransformer(output_distribution='normal')
}

for name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X)
    # e.g., inspect histograms or plug into a model
```

### 4. Practice Challenge

- Build a pipeline incorporating each advanced scaler and run cross-validation on a regression task.
- Plot feature distributions before and after each transform to visually assess skew reduction.
- Compare model metrics (R², RMSE) across scalers and explain why some outperform others on skewed data.

---

## Feature Scaling – Part 3: Best Practices, Preventing Leakage, and Streaming

### 1. Preventing Data Leakage in Cross‐Validation

To avoid peeking at test data, fit your scaler inside each train fold rather than beforehand. Use an **estimator pipeline** with cross‐validation:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model    import Ridge
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE:", -scores.mean())
```

This ensures each fold computes its own `mean` and `std`, eliminating leakage and producing unbiased performance estimates.

### 2. Chaining Scaling with Feature Selection

Combining scaling with dimensionality reduction or sparsity can improve both interpretability and performance:

- **PCA after Standardization**
    
    Standardize, then apply PCA to decorrelate and reduce dimensionality.
    
- **Lasso (L1) Regularization**
    
    Scaling ensures the penalty treats each coefficient equally; wrap in a pipeline to select and scale in one shot.
    

```python
from sklearn.decomposition      import PCA
from sklearn.linear_model      import Lasso
from sklearn.pipeline          import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca',   PCA(n_components=10)),
    ('lasso', Lasso(alpha=0.01))
])
pipe.fit(X_train, y_train)
selected_features = pipe.named_steps['lasso'].coef_ != 0
```

### 3. Scaling for Streaming and Online Learning

When data arrives in batches or streams, leverage **incremental scalers** that update statistics on the fly:

```python
from sklearn.preprocessing       import StandardScaler
from sklearn.linear_model        import SGDRegressor

scaler = StandardScaler(with_mean=True, with_std=True)
model  = SGDRegressor(alpha=0.001, max_iter=1, warm_start=True)

for X_batch, y_batch in stream_batches():
    scaler.partial_fit(X_batch)
    X_scaled = scaler.transform(X_batch)
    model.partial_fit(X_scaled, y_batch)
```

This pattern maintains a running mean and variance, allowing your model to adapt without restarting.

### 4. Advanced Use Cases

| Technique | Purpose | When to Use |
| --- | --- | --- |
| BatchNorm / LayerNorm | Normalize activations in deep networks | Training deep neural nets to stabilize gradients |
| PCA Whitening | Decorrelate and whiten features | Preprocessing for algorithms sensitive to covariance |
| Group‐wise Scaling | Scale within subgroups (e.g., time series) | Panel data where each group has different scale |
| Supervised Scaling | Use target statistics for scaling | Rare categories or when feature–target relationship is strong |
- **BatchNorm / LayerNorm** stabilize training in CNNs and Transformers.
- **PCA Whitening** transforms features to have identity covariance.
- **Group‐wise Scaling** can use `DataFrame.groupby(...).transform(...)` to apply scaling per entity.
- **Supervised Scaling** (e.g., target encoding) leverages `mean_target` within each category but risks leakage—always fit within CV folds.

### Practice Challenge

1. Implement nested cross‐validation with a pipeline that scales, selects via PCA, then trains a model. Report average accuracy.
2. Compare model performance on skewed data using:
    - No scaling
    - StandardScaler
    - QuantileTransformerTrain a random forest and analyze metric differences.
3. Simulate a data stream and build an online pipeline using `partial_fit` for both scaler and SGDClassifier. Measure how error evolves with each batch.

---

## Checking Gradient Descent for Convergence

### 1. Why Monitor Convergence?

Before you trust your learned parameters, you need to ensure gradient descent has actually settled into a minimum. Without checks, you might stop too early—or waste time running needless iterations. Convergence monitoring lets you automate stopping, pick sensible learning rates, and diagnose oscillations or divergence. In production pipelines this becomes essential for reproducibility and resource efficiency.

### 2. Common Convergence Criteria

- **Cost Change Threshold:** Stop when the absolute or relative decrease in the loss `J(θ)` falls below a small ε.
- **Parameter Shift Threshold:** Detect when `‖θᵏ⁺¹ − θᵏ‖` is tiny, indicating minimal updates.
- **Gradient Norm Threshold:** Halt when `‖∇J(θ)‖` drops below a set tolerance, implying you’re near a stationary point.
- **Validation Plateau:** For supervised tasks, use early stopping by monitoring validation loss; stop if it hasn’t improved for *p* iterations.

### 3. Implementing Stopping Rules

### 3.1 Cost‐Difference Stopping

```python
prev_cost = float('inf')
for i in range(max_iters):
    cost = compute_cost(X, y, theta)
    if abs(prev_cost - cost) < tol:
        print(f"Converged at iteration {i}")
        break
    prev_cost = cost
    theta -= alpha * gradient(X, y, theta)
```

### 3.2 Parameter‐Shift Stopping

```python
for i in range(max_iters):
    update = alpha * gradient(X, y, theta)
    if np.linalg.norm(update) < tol:
        print(f"Parameter change below tol at iter {i}")
        break
    theta -= update
```

### 3.3 Gradient‐Norm Stopping

```python
for i in range(max_iters):
    grad = gradient(X, y, theta)
    if np.linalg.norm(grad) < grad_tol:
        print(f"Gradient norm below tol at iter {i}")
        break
    theta -= alpha * grad
```

### 3.4 Early Stopping on Validation

```python
best_val, patience = float('inf'), 0
for i in range(max_iters):
    theta -= alpha * gradient(X_train, y_train, theta)
    val_loss = compute_cost(X_val, y_val, theta)
    if val_loss < best_val:
        best_val, patience = val_loss, 0
    else:
        patience += 1
    if patience >= p:
        print(f"Validation stalled after {i} iters")
        break
```

### 4. Full Example: Unified Gradient Descent

```python
def gradient_descent(X, y, theta, alpha, max_iters, tol, grad_tol):
    history = {'cost': [], 'grad_norm': []}
    prev_cost = compute_cost(X, y, theta)

    for i in range(max_iters):
        grad = gradient(X, y, theta)
        cost = compute_cost(X, y, theta)

        history['cost'].append(cost)
        history['grad_norm'].append(np.linalg.norm(grad))

        if abs(prev_cost - cost) < tol or np.linalg.norm(grad) < grad_tol:
            print(f"Stopped at iter {i}")
            break

        theta -= alpha * grad
        prev_cost = cost

    return theta, history
```

### 5. Visualization and Diagnostics

Plot both the cost curve and gradient-norm curve against iterations to verify smooth decline and absence of oscillations. A logarithmic scale on the vertical axis often reveals plateaus or sudden jumps. Overlay multiple runs with different learning rates to choose the most stable setup.

### 6. Practical Tips

- Scale features first to speed convergence and avoid tiny gradients.
- Choose tolerances relative to initial cost or gradient norm (e.g., `tol = 1e-6 * initial_cost`).
- Monitor both training and validation curves to catch overfitting.
- Combine with learning-rate schedules or momentum when plain GD stalls.
- Log metrics in real time for long runs (e.g., using TensorBoard or a simple CSV).

---

## Choosing the Learning Rate

### 1. Why the Learning Rate Matters

Gradient descent updates

```
θ ← θ – α·∇J(θ)
```

hinge critically on α. Too small, and convergence crawls; too large, and steps overshoot or diverge. Picking α well balances speed and stability—vital for training efficiency, reproducible results, and passing interviews on optimizer tuning.

### 2. Signs of Under- and Over-Shooting

- Under-shooting (α too small):
    - Cost decreases very slowly.
    - Learning curve almost flat.
- Over-shooting (α too large):
    - Cost oscillates or increases.
    - Gradient norms blow up.

Plotting loss vs. iterations will quickly reveal which regime you’re in.

### 3. Manual and Automated Strategies

### 3.1 Manual Search

1. Pick an initial α (e.g., 1e-3).
2. Run a few dozen iterations; observe loss.
3. If loss plateaus, multiply α by 3–10; if it diverges, divide by 3–10.
4. Repeat until you find a stable, fast-converging regime.

### 3.2 Grid Search / Cross-Validation

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('sgd',   SGDRegressor(max_iter=1000, tol=1e-4))
])

param_grid = {'sgd__eta0': [1e-4, 1e-3, 1e-2, 1e-1]}
search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error')
search.fit(X_train, y_train)
print("Best α:", search.best_params_)
```

### 3.3 Learning-Rate Schedules

- **Step Decay:** Reduce α by a factor every *k* epochs.
- **Exponential Decay:**
    
    ```
    α_t = α_0 · exp(–k·t)
    ```
    
- **Inverse Time Decay:**
    
    ```
    α_t = α_0 / (1 + decay_rate·t)
    ```
    

### 3.4 Adaptive Optimizers

Methods like **Adam**, **RMSprop**, and **Adagrad** adjust per-parameter learning rates automatically:

| Optimizer | Key Idea | Default α Range |
| --- | --- | --- |
| Adagrad | Accumulates squared gradients, slows down over time | 1e-2 to 1e-1 |
| RMSprop | Exponential moving average of squared grads | 1e-3 |
| Adam | Combines momentum + RMSprop | 1e-3 |

### 4. Visual Diagnostics

1. Sweep α over logarithmic scale (e.g., 1e-5 to 1).
2. For each α, run fixed iterations and record final loss.
3. Plot loss vs. α on a log-log or semilog plot.

```python
alphas = np.logspace(-5, 0, 20)
losses = []
for a in alphas:
    θ, history = gradient_descent(X, y, zeros, a, max_iters=200)
    losses.append(history['cost'][-1])

plt.semilogx(alphas, losses, marker='o')
plt.xlabel('α'); plt.ylabel('Final Loss')
plt.title('Loss vs. Learning Rate')
plt.show()
```

### 5. Practical Tips

- Always **scale** features first—unscaled data can make a “good” α useless.
- Tie your stopping tolerance to α (e.g., `tol = 1e-4 * α`).
- Combine schedules with early stopping on validation loss.
- In interviews, explain trade-offs: higher α speeds initial descent but risks divergence; schedules mitigate this tension.

### 6. Practice Exercises

1. **Alpha Sweep on Real Data**
    - Take a regression dataset, standardize it, and sweep α from 1e-6 to 1.
    - Plot convergence speed and final loss.
2. **Implement Decay Schedule**
    - Code inverse time decay in your gradient descent loop.
    - Compare convergence against fixed α.
3. **Compare Optimizers**
    - Train a simple neural network on MNIST using SGD, SGD+momentum, and Adam.
    - Track training/validation loss and discuss learning-rate choices for each.

---

## Feature Engineering: Crafting Powerful Inputs for Your Models

### 1. What Is Feature Engineering and Why It Matters

Feature engineering is the art and science of transforming raw data into meaningful signals that machine-learning algorithms can leverage effectively. Well-engineered features can:

- Expose underlying patterns and relationships
- Improve model accuracy and convergence speed
- Reduce overfitting by summarizing or regularizing information
- Turn a mediocre model into a high-performing one with minimal algorithmic changes

In interviews and real-world projects alike, strong feature engineering skills distinguish you as someone who understands both data and model behavior deeply.

### 2. Core Types of Feature Engineering

- **Domain-Driven Derivations**
    
    Leverage subject-matter knowledge to craft features.
    
    Examples:
    
    - In finance, compute debt-to-income ratios.
    - In IoT, derive moving averages of sensor readings.
- **Mathematical Transformations**
    
    Tame skew or nonlinearities:
    
    - Log, square-root, or Box-Cox transforms
    - Polynomial expansions (e.g., x², x·y)
- **Encoding Categorical Variables**
    - One-hot or dummy encoding for nominal vars
    - Ordinal encoding with meaningful order
    - Target (mean) encoding in high-cardinality cases (within CV folds)
- **Feature Interactions**
    
    Automatically or manually combine features:
    
    - Pairwise products
    - Ratios (e.g., price_per_square_foot)
    - Boolean flags (e.g., is_weekend & is_holiday)
- **Aggregations and Windowing**
    
    Summarize groups or time windows:
    
    - Group‐by customer_id and compute purchase counts
    - Rolling means, lags, and differences in time series
- **Extraction from Unstructured Data**
    - Text: TF-IDF, word embeddings, topic scores
    - Images: color histograms, edge detectors, pretrained CNN features
    - Dates: day_of_week, month, is_quarter_end

### 3. Workflow and Best Practices

1. **Explore and Visualize**
    - Understand distributions and correlations
    - Spot outliers, missing patterns, and potential transformations
2. **Ideate and Hypothesize**
    - Brainstorm features based on domain insights
    - Ask “What could reveal my target most directly?”
3. **Implement in Reusable Pipelines**
    - Encapsulate steps in `sklearn.Pipeline` or `FeatureUnion`
    - Version and document transformation logic
4. **Validate Rigorously**
    - Fit transformers only on training folds to avoid leakage
    - Use cross-validation to assess feature utility
5. **Iterate and Prune**
    - Measure feature importance or use regularization (Lasso, tree-based)
    - Drop redundant or noisy inputs

### 4. Code Examples: From Raw to Ready

### 4.1 Polynomial and Log Transforms

```python
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.pipeline      import Pipeline

poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('log',  FunctionTransformer(np.log1p))
])

X_transformed = poly_pipe.fit_transform(X_numeric)
```

### 4.2 Date/Time Feature Extraction

```python
import pandas as pd

def extract_datetime_features(df, col):
    dt = pd.to_datetime(df[col])
    return pd.DataFrame({
        'hour':        dt.dt.hour,
        'day_of_week': dt.dt.dayofweek,
        'month':       dt.dt.month,
        'is_weekend':  dt.dt.dayofweek >= 5
    })

datetime_feats = extract_datetime_features(df, 'timestamp')
```

### 4.3 Group-By Aggregations

```python
# For each user, compute total and mean purchase
agg = df.groupby('user_id')['purchase_amount'].agg(['sum','mean','count']).reset_index()
agg.columns = ['user_id','total_spent','avg_spent','purchase_count']
df = df.merge(agg, on='user_id', how='left')
```

### 4.4 Text Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_text = tfidf.fit_transform(df['review_text'])
```

### 5. Automated Tools and Libraries

- **Featuretools**: Automated deep feature synthesis for relational datasets
- **TSFresh**: Time-series feature extraction at scale
- **AutoML Platforms** (TPOT, H2O AutoML): Offer built-in feature preprocessing and selection

These can jump-start your pipeline, but always review and understand the generated features.

### 6. Practice Challenges

1. **Titanic Survival**
    - Engineer titles from names (`Mr`, `Mrs`, `Master`), cabin decks, and family sizes.
    - Compare model performance before and after your features.
2. **House Prices Regression**
    - Derive “age_of_house”, “total_sqft”, and “has_pool” flags.
    - Use polynomial interactions between lot size and number of rooms.
3. **Time-Series Sales Forecasting**
    - Create rolling means, previous-day differences, and holiday flags.
    - Feed into a gradient-boosting regressor and analyze feature importance.

---

## Polynomial Regression: Deep Dive for Mastery

### Pre-requisites

- Basic linear regression: hypothesis, cost function, normal equation
- Matrix and vector operations (dot products, transposes)
- Concept of overfitting vs. underfitting

If any feel shaky, review how a straight-line model fits data and how cost changes with parameters.

### 1. Intuition & Use Cases

Imagine fitting a straight ribbon to scattered points—it works only if points lie roughly on a line. If they curve, you need a bendable ribbon. Polynomial regression adds “bend” by including powers of your input.

Use cases:

- Modeling growth rates (e.g., population vs. time)
- Pricing curves (e.g., square footage vs. house price)
- Sensor calibrations where response isn’t linear

### 2. Mathematical Formulation

We expand a single feature `x` into degrees up to `d`. Your hypothesis becomes:

```
h(x) = θ0 + θ1·x + θ2·x^2 + … + θd·x^d
```

For `m` examples, build the design matrix `X_poly` where each row is `[1, x, x^2, …, x^d]`. Then solve for `θ` just like in linear regression.

### 3. Clean-Formulas & Why They Work

Normal equation (closed-form):

```
θ = inverse( X_polyᵀ · X_poly ) · X_polyᵀ · y
```

- `X_polyᵀ · X_poly` captures how powers of `x` correlate
- Inverting adjusts for those correlations to minimize squared error

Gradient descent on `X_poly` uses the same update rule:

```python
gradient = (1/m) * X_poly.T.dot(X_poly.dot(theta) - y)
theta    = theta - alpha * gradient
```

Scaling `X_poly` columns speeds convergence and keeps gradients balanced.

### 4. Visual & Geometric Insight

Plot your data points on a 2D scatter. A degree-2 model fits a parabola, degree-3 an S-shaped curve, and so on. Each added power is a new “bend” basis function.

Visualizing design matrix columns as independent curves shows how combining them reconstructs complex shapes.

### 5. Real-World ML Workflow

1. **Feature Engineering**: Use `PolynomialFeatures` in a pipeline.
2. **Scaling**: Apply `StandardScaler` after expansion to normalize each power term.
3. **Regularization**: Wrap with Ridge or Lasso to prevent overfitting high-degree terms.
4. **Model Selection**: Use k-fold cross-validation to pick best degree `d` and regularization strength.

### 6. Python Implementation Example

```python
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import PolynomialFeatures, StandardScaler
from sklearn.linear_model    import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# 1. Generate synthetic data
np.random.seed(0)
X = 6 * np.random.rand(100,1) - 3
y = 0.5*X**3 - X**2 + 2*X + np.random.randn(100,1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3. Pipeline: degree=3 polynomial + scaling + ridge regression
pipe = Pipeline([
  ('poly',  PolynomialFeatures(degree=3, include_bias=False)),
  ('scale', StandardScaler()),
  ('ridge', Ridge(alpha=1.0))
])

# 4. Cross-validate degree selection
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Mean CV RMSE:", np.sqrt(-scores.mean()))
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 7. Practice Problems

1. **Degree Sweep**
    - Generate `y = sin(x) + noise`.
    - Fit degrees 1–15, record validation RMSE, plot error vs. degree.
2. **Housing Prices**
    - Use California housing.
    - For feature `AveRooms`, compare linear vs. polynomial (deg 2–4) models with/without Ridge.
3. **Manual Expansion & GD**
    - Write a function to expand features to degree `d`.
    - Run gradient descent with cost-difference stopping and experiment with learning rates.

---