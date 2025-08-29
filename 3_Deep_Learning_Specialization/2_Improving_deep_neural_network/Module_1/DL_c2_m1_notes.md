# DL_c2_m1

## Train / Dev / Test Sets

### 1. Concept Intuition

Splitting your data into three disjoint sets—train, development (dev), and test—is the foundation of building models that generalize well.

- Train set: The model “sees” this data during optimization.
- Dev set: You tune hyperparameters and make design choices here. It stands between training and the final evaluation.
- Test set: You only touch this after everything is finalized. It gives you an honest estimate of how your model will perform on completely new data.

Why it matters: Without a proper split, you’ll overestimate how well your model works in the real world. The dev set prevents you from over-tuning to quirks of the train set, and the test set guards against accidentally peeking during development.

### 2. Mathematical Breakdown

At each split, you can compute an error metric — say, classification error or mean squared error. Let

- `m_train`, `m_dev`, `m_test` be the number of examples in train/dev/test.
- `J_train(θ)` be the cost on the train set.
- `J_dev(θ)` and `J_test(θ)` be the cost on the dev and test sets respectively.

```python
# Given predictions y_pred and labels y_true, classification error is:
def classification_error(y_pred, y_true):
    return np.mean(y_pred != y_true)

# Costs on each set:
error_train = classification_error(model.predict(X_train), y_train)
error_dev   = classification_error(model.predict(X_dev),   y_dev)
error_test  = classification_error(model.predict(X_test),  y_test)
```

Interpreting these:

- If `error_train` is low but `error_dev` is high → high variance (overfitting).
- If both `error_train` and `error_dev` are high → high bias (underfitting).
- `error_test` gives your final unbiased performance estimate once you’ve locked in hyperparameters.

### 3. Code & Practical Application

Below is a minimal NumPy + scikit-learn example splitting synthetic data into train/dev/test and computing errors.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Create synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=2,
                           n_informative=2, n_redundant=0,
                           random_state=42)

# 2. Split into train+dev and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# 3. Further split train+dev into train and dev
X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=1)
# Note: 0.25 x 0.8 = 0.2 so train/dev/test = 60/20/20

# 4. Train a simple model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Compute classification error
def classification_error(y_pred, y_true):
    return np.mean(y_pred != y_true)

for name, X_set, y_set in [
    ("Train", X_train, y_train),
    ("Dev",   X_dev,   y_dev),
    ("Test",  X_test,  y_test)
]:
    y_pred = model.predict(X_set)
    err    = classification_error(y_pred, y_set)
    print(f"{name} error: {err:.3f}")
```

This snippet shows how to structure your splits, train, and evaluate cleanly.

### 4. Visualization / Geometry

Imagine your data as points in 2D:

- Plot train points in blue, dev in green, test in red.
- See if the splits share the same distribution shape.

```python
import matplotlib.pyplot as plt

plt.scatter(X_train[:,0], X_train[:,1], c='blue', alpha=0.5, label='Train')
plt.scatter(X_dev[:,0],   X_dev[:,1],   c='green', alpha=0.5, label='Dev')
plt.scatter(X_test[:,0],  X_test[:,1],  c='red', alpha=0.5, label='Test')
plt.legend()
plt.title("Train / Dev / Test Split Visualization")
plt.show()
```

Geometric intuition: If train covers a different region of the feature space than dev/test, your model may perform poorly on unseen examples. A good random split keeps all sets representative of the same underlying distribution.

### 5. Common Pitfalls & Tips

- Data leakage: never derive features using the full dataset (e.g., scaling on all data first).
- Peeking at the test set: avoid using test performance to guide hyperparameter choices.
- Imbalanced splits: ensure class proportions stay roughly equal across splits.
- Too small dev/test sets: with only a handful of examples, your error estimates become noisy.
- Time-series or grouped data: random splits can break temporal or group coherence—use time-based or group-based splitting instead.

### 6. Practice Exercises

### Exercise 1: Manual Split

Given a small dataset

```python
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,1,1,1,1,0,1,0])
```

- Manually create a 60/20/20 train/dev/test split without using helper functions.
- Compute and print the mean target (proportion of 1’s) in each set.

Hint:

- Decide indices for each split.
- Use array slicing and `.mean()`.

### Exercise 2: Stratified Split

Use scikit-learn’s `train_test_split` with `stratify=y` to split the above data.

- Compare class ratios in train/dev/test.
- Why is stratification important when your classes are imbalanced?

### Exercise 3: Data Leakage Simulation

1. On the synthetic dataset earlier, perform `StandardScaler()` on the entire `X` before splitting.
2. Train and evaluate.
3. Now, redo the scaling by fitting the scaler *only* on `X_train` and transform dev/test with the same parameters.
4. Compare the errors.

Hint: Look for a small but real difference in dev/test performance due to leakage.

---

## Bias and Variance

### 1. Concept Intuition

Bias and variance quantify two fundamental sources of error in supervised learning.

- Bias measures how close your model’s predictions are to the true underlying function on average. High bias means the model is too simple and underfits (it can’t capture the patterns).
- Variance measures how much your model’s predictions fluctuate for different training sets. High variance means the model is too complex and overfits (it captures noise).

Balancing bias and variance is key. Too much bias → underfitting. Too much variance → overfitting. The sweet spot minimizes total error on new data.

### 2. Mathematical Breakdown

We decompose the expected squared error at a point (x) into three terms:

```python
Total_Error = Bias^2(x) + Variance(x) + Irreducible_Error(x)
```

Here’s how each term is computed (in expectation over training sets):

```python
# Given many models f_hat trained on different datasets, and true function f
# Let y_pred_i = f_hat_i(x) be predictions on the same x

# 1. Compute the average prediction across models
y_avg = np.mean([y_pred_i for y_pred_i in y_preds], axis=0)

# 2. Bias^2 at x
bias2 = (y_avg - f(x))**2

# 3. Variance at x
variance = np.mean([(y_pred_i - y_avg)**2 for y_pred_i in y_preds], axis=0)

# 4. Irreducible error (noise) = variance of true noise around f(x)
# Often estimated from data if you know noise level
```

- `bias2` shows the squared distance between the average model and the true function.
- `variance` shows how widely models differ around their average prediction.
- The **irreducible error** is noise you can’t model (measurement noise, randomness).

### 3. Code & Practical Application

We’ll generate 1D data from a nonlinear function, fit polynomial regressions of varying degrees, compute train/dev errors, and visualize bias–variance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Generate noisy data
np.random.seed(0)
m = 30
X = np.sort(5 * np.random.rand(m, 1), axis=0)
y = np.sin(X).ravel() + 0.3 * np.random.randn(m)

# 2. Define helper to fit-and-evaluate for a given degree
def eval_poly(degree, X_train, y_train, X_dev, y_dev):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_dev_poly   = poly.transform(X_dev)
    model = LinearRegression().fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_dev_pred   = model.predict(X_dev_poly)
    return mean_squared_error(y_train, y_train_pred), mean_squared_error(y_dev, y_dev_pred)

# 3. Split data into train/dev
split = int(0.6 * m)
X_train, y_train = X[:split], y[:split]
X_dev,   y_dev   = X[split:], y[split:]

# 4. Sweep degrees and collect errors
degrees = [1, 3, 5, 9]
train_err, dev_err = [], []
for d in degrees:
    te, de = eval_poly(d, X_train, y_train, X_dev, y_dev)
    train_err.append(te)
    dev_err.append(de)

# 5. Plot
plt.plot(degrees, train_err, marker='o', label='Train MSE')
plt.plot(degrees, dev_err,   marker='s', label='Dev MSE')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.show()
```

This code shows:

- **High bias** at low degrees: both train/dev errors are large.
- **High variance** at high degrees: train error drops, but dev error climbs.
- The valley around degree 3–5 often gives the best generalization.

### 4. Visualization / Geometry

```
Test Error
   ^
   |            .
   |         .     .
   |       .         .
   |     .             .
   |   .                 .
   +------------------------> Model Complexity
    Underfit     Optimal     Overfit
```

- Underfit region (left): your hypothesis space is too restricted (high bias).
- Overfit region (right): your hypothesis space is too flexible (high variance).
- Optimal point minimizes expected test error.

In geometric terms, bias pulls your model far from the true function, while variance makes it wiggle around depending on the sample.

### 5. Common Pitfalls & Tips

- Confusing train error with generalization: low train error alone doesn’t prove low variance.
- Relying on a single train/dev split: variance estimates can be noisy; consider k-fold cross-validation for more stability.
- Ignoring irreducible error: no amount of complexity can beat measurement noise.
- Overinterpreting small error differences: a tiny dev error drop might be just random fluctuation.
- Using overly powerful models by default: start simple, then add complexity only when needed.

### 6. Practice Exercises

### Exercise 1: k-Fold Cross-Validation for Bias–Variance

- Implement 5-fold cross-validation on the polynomial regression above.
- For each degree in `[1,3,5,9]`, collect the mean and standard deviation of dev MSE across folds.
- Plot error bars.*Provided hint*: use `KFold` from scikit-learn and loop over splits.

### Exercise 2: Noise Sensitivity

- Increase noise magnitude from `0.3` to `1.0` in the synthetic data.
- Repeat the bias–variance sweep.
- Observe how the irreducible error raises the floor for both train and dev MSE.

### Exercise 3: Visualize Model Variance

- Create three different training sets (by resampling noise) of size `m=30`.
- For a fixed degree (e.g., 9), fit models on each set and plot all three predictions on one plot with the true function.
- Compute the variance term at several x-values by measuring the spread of predictions.

---

## Best Recipe for Machine Learning

### 1. Concept Intuition

Think of machine learning like cooking a complex dish. You need quality ingredients (data), a clear recipe (workflow), the right cooking tools (algorithms), constant taste tests (evaluation), and iterative tweaks (hyperparameter tuning).

A well-defined recipe ensures you don’t skip steps, keeps experiments organized, prevents accidental “over-salting” (overfitting), and leads to reliably tasty results (generalizable models).

Key steps in the ML recipe:

- Frame the problem: Define inputs, outputs, and how you’ll measure success.
- Organize data: Split into train, dev, and test sets.
- Build a simple baseline model: Verify your pipeline works end-to-end.
- Evaluate and diagnose: Use train/dev errors to identify bias vs variance.
- Iterate orthogonally: Tackle one aspect at a time—features, model complexity, regularization, data volume.
- Final test: Once you’ve tuned on dev, get an honest assessment on the test set.

### 2. Mathematical Breakdown

At the heart is choosing model parameters θ and hyperparameters λ to minimize an evaluation metric on the dev set:

```python
# Let f(x; θ, λ) be our model predictions on input x
# Define J_train, J_dev as the chosen cost functions (e.g., MSE or classification error)
θ_star = argmin_θ  J_train( θ, λ )                # fit parameters on train set
λ_star = argmin_λ  J_dev( θ_star(λ), λ )         # choose hyperparameters via dev set
final_cost = J_test( θ_star(λ_star), λ_star )    # final evaluation on test set
```

Bias–variance diagnosis uses:

```python
bias2 ≈ J_train(θ_star)      # large ⇒ underfitting
variance ≈ J_dev(θ_star) - J_train(θ_star)   # large ⇒ overfitting
```

If bias2 is high, increase model capacity or features. If variance term is high, get more data or add regularization.

### 3. Code & Practical Application

Below is a skeleton using scikit-learn pipelines to illustrate the recipe end-to-end on the Iris dataset.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load data
X, y = load_iris(return_X_y=True)

# 2. Split into train/dev/test (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_dev, y_train, y_dev   = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)

# 3. Build a pipeline: scaling + SVM
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc",    SVC())
])

# 4. Quick baseline
pipeline.set_params(svc__kernel="linear", svc__C=1.0)
pipeline.fit(X_train, y_train)

# 5. Evaluate baseline
print("Baseline train acc:", accuracy_score(y_train, pipeline.predict(X_train)))
print("Baseline dev   acc:", accuracy_score(y_dev,   pipeline.predict(X_dev)))

# 6. Hyperparameter tuning (dev set)
param_grid = {"svc__C": [0.1, 1, 10], "svc__kernel": ["linear", "rbf"]}
grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(X_train, y_train)

# 7. Best model performance
best_model = grid.best_estimator_
print("Tuned train acc:", accuracy_score(y_train, best_model.predict(X_train)))
print("Tuned dev   acc:", accuracy_score(y_dev,   best_model.predict(X_dev)))

# 8. Final test evaluation (honest)
print("Final test  acc:", accuracy_score(y_test,  best_model.predict(X_test)))
```

This script walks through framing, splitting, baseline, dev-set tuning, and a final test.

### 4. Visualization / Geometry

```
   +------------+      +-----------+      +-----------+
   | Raw Data   | ---> | Train Set | ---> | Fit Model |
   +------------+      +-----------+      +-----------+
                           |
                           v
                       +--------+
                       | Dev Set|  <-- Hyperparameter tuning & bias/variance cheks
                       +--------+
                           |
                           v
                       +---------+
                       | Test Set|  <-- Final unbiased evaluation
                       +---------+
```

Geometric intuition:

- Each split samples the same input space.
- Dev-set performance traces a curve as you tweak capacity or regularization—look for the sweet spot where dev error is minimized without overfitting the train set.

### 5. Common Pitfalls & Tips

- Peeking at test set too early biases your final evaluation.
- Tuning multiple things at once muddies the cause-effect—change one variable at a time.
- Ignoring data preprocessing steps (normalization, feature encoding) can introduce pipeline leaks.
- Using overly small dev/test sets makes error estimates noisy—aim for at least a few hundred examples if possible.
- Not tracking experiments—use simple logging or tools like MLflow to compare runs.

### 6. Practice Exercises

### Exercise 1: Build Your Own Recipe

- Pick a public dataset (e.g., Wine Quality from UCI).
- Define your metric (accuracy, F1, RMSE).
- Manually split into 60/20/20.
- Create a pipeline with at least two preprocessing steps and one model.
- Train a baseline, compute train/dev errors, then tune one hyperparameter at a time.
- Finally, report test performance.

### Exercise 2: Bias vs Variance Walk

- On your pipeline above, log train and dev errors as you vary model complexity (e.g., tree depth in a DecisionTree).
- Plot those errors against complexity to visualize underfitting vs overfitting.

### Exercise 3: Danger of Data Leakage

- Introduce a “leaky” feature (e.g., one that uses test-set information when computing a statistic) in preprocessing.
- Train and compare performance with and without the leak.
- Observe the dev error collapse and why it fails on the true test set.

---

## Regularization

### 1. Concept Intuition

Regularization is a set of techniques that prevent a model from overfitting by constraining its complexity. Think of it as adding a gentle “brake” on the model’s parameters so they don’t swing wildly to fit every noise in your training data.

Key ideas:

- You accept a bit more bias (slightly underfit) in exchange for much lower variance (more stable predictions).
- Penalizing large weights forces the model to prefer simpler functions.
- Different regularizers (L2, L1, dropout) achieve this simplicity in different ways—shrinking all weights versus zeroing out some entirely versus randomly silencing neurons.

### 2. Mathematical Breakdown

### L2 Regularization (Weight Decay)

Let

- `m` = number of training examples
- `θ` = vector of weights (excluding bias)
- `J0(θ)` = original cost (e.g., mean squared error or cross-entropy)

The L2-regularized cost is:

```python
J(θ) = J0(θ) + (λ / (2*m)) * sum(θ_j**2 for each weight θ_j)
```

Gradient of this cost w.r.t. a weight θ_j becomes:

```python
dJ/dθ_j = dJ0/dθ_j + (λ / m) * θ_j
```

- The extra `(λ / m) * θ_j` term gently pulls each weight toward zero on every gradient step.
- You never penalize the bias term.

### L1 Regularization

L1 adds an absolute‐value penalty, which encourages exact zeros and thus sparsity:

```python
J(θ) = J0(θ) + (λ / m) * sum(abs(θ_j) for each weight θ_j)
```

Gradient (subgradient at zero) for each θ_j:

```python
dJ/dθ_j = dJ0/dθ_j + (λ / m) * sign(θ_j)
```

- Here `sign(θ_j)` is +1 if θ_j>0, –1 if θ_j<0, and any value in [–1,1] at θ_j=0.

### 3. Code & Practical Application

### NumPy Logistic Regression with L2

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_and_grad(X, y, θ, λ):
    m = X.shape[0]
    z = X.dot(θ)
    h = sigmoid(z)
    # Cross-entropy loss
    J0 = - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    # L2 penalty (exclude bias θ[0])
    J_reg = (λ / (2*m)) * np.sum(θ[1:]**2)
    J = J0 + J_reg

    # Gradient
    error = h - y
    grad0 = (1/m) * X[:,0].dot(error)                # bias term
    grad_rest = (1/m) * X[:,1:].T.dot(error) + (λ/m)*θ[1:]
    grad = np.concatenate(([grad0], grad_rest))
    return J, grad

# Example usage
np.random.seed(0)
X = np.random.randn(100, 3)    # 100 examples, 2 features + bias column
X[:,0] = 1                     # bias term
y = (np.random.rand(100) > 0.5).astype(int)
θ = np.zeros(3)
λ = 1.0

# One gradient step
J, grad = compute_cost_and_grad(X, y, θ, λ)
θ -= 0.1 * grad
print("Cost:", J, "Grad:", grad)
```

### PyTorch Weight Decay

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
    nn.Sigmoid()
)

# Use weight_decay argument for L2 penalty on all weights (bias not shrunk by default)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)
criterion = nn.BCELoss()

# Training loop (sketch)
for X_batch, y_batch in dataloader:
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()
```

### 4. Visualization / Geometry

1. **Constraint View**
    - L2: Your weight vector θ must lie inside a circle (‖θ‖₂ ≤ C).
    - L1: θ lies inside a diamond (‖θ‖₁ ≤ C).
2. **Level Sets + Contours**
    - Plot contours of the original loss and overlay the L2 circle. The minimum of the combined cost sits where a contour first touches the circle—pushing θ inward.

```python
# Pseudocode for contour + circle plot
#   - Grid of (θ1, θ2), compute J0 for each.
#   - Plot J0 contours.
#   - Plot circle of radius sqrt(C).
```

1. **Weight Shrinkage**
    - Track ‖θ‖ over training epochs for different λ values. You’ll see larger λ keeps weights much closer to zero.

### 5. Common Pitfalls & Tips

- Forgetting not to regularize bias terms; you usually only shrink weights.
- Mixing up weight decay (directly scales weights each step) and L2 penalty (adds to loss)—PyTorch’s `weight_decay` is true weight decay, some frameworks add `λ/m * θ` in gradients.
- Choosing λ purely by intuition: always tune λ on your dev set via grid search.
- Over-regularizing: too large λ → high bias underfitting.
- Combining L1 and L2 (Elastic Net) when you want both sparsity and stability.

### 6. Practice Exercises

### Exercise 1: Implement L1 vs L2 in NumPy

- Adapt the NumPy logistic code above to support L1 penalty.
- Train on a synthetic dataset and compare the number of near-zero weights for different λ values.

### Exercise 2: λ-Sweep with PyTorch

- Using a small feedforward network on MNIST (or Fashion-MNIST), train with `weight_decay` ∈ {0, 1e-4, 1e-3, 1e-2}.
- Plot train and dev accuracies versus λ.

### Exercise 3: Elastic Net

- Implement a combined penalty: `λ1‖θ‖₁ + (λ2/2)‖θ‖₂²` in NumPy.
- Compare model sparsity and generalization for various (λ1, λ2) pairs.

---

## Why Regularization Reduces Overfitting

### 1. Concept Intuition

Regularization tames overly complex models by discouraging extreme parameter values.

- Overfitting happens when a model memorizes noise in the training data, leading to wildly varying predictions on new inputs.
- By adding a penalty for large weights, regularization makes the model prefer simpler patterns that capture the true signal rather than noise.
- Think of it as adding friction to the learning process—weights can still move, but only if the gain in fitting the data outweighs the penalty for complexity.

### 2. Mechanism Breakdown

At its core, regularization augments the original cost function (`J0(θ)`) with a complexity term.

**L2 Regularization (Weight Decay)**

```python
J(θ) = J0(θ) + (λ / (2*m)) * sum(θ_j**2 for each weight θ_j)
```

- `λ` controls the trade-off between fitting the data and keeping weights small.
- On each gradient step, weights are nudged toward zero by an extra `λ/m * θ_j` term.
- Smaller weights produce smoother decision boundaries and reduce sensitivity to small fluctuations in the training set.

**L1 Regularization (Sparsity Penalty)**

```python
J(θ) = J0(θ) + (λ / m) * sum(abs(θ_j) for each weight θ_j)
```

- Encourages many weights to become exactly zero, effectively selecting a simpler subset of features.
- By zeroing out irrelevant parameters, the model has less flexibility to overfit noise.

### 3. Code & Practical Application

### NumPy Example: L2 vs No Regularization

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Create synthetic data
X, y = make_regression(n_samples=200, n_features=20, noise=20, random_state=0)

# Split into train/dev
split = int(0.7 * len(X))
X_train, y_train = X[:split], y[:split]
X_dev,   y_dev   = X[split:], y[split:]

# 1. Fit without regularization
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_dev_lr = lr.predict(X_dev)
mse_dev_lr = mean_squared_error(y_dev, pred_dev_lr)

# 2. Fit with L2 regularization (Ridge)
ridge = Ridge(alpha=10)  # alpha = λ
ridge.fit(X_train, y_train)
pred_dev_ridge = ridge.predict(X_dev)
mse_dev_ridge = mean_squared_error(y_dev, pred_dev_ridge)

print(f"Dev MSE without reg: {mse_dev_lr:.2f}")
print(f"Dev MSE with L2 reg: {mse_dev_ridge:.2f}")
```

- You’ll typically see a lower dev MSE with the regularized model when overfitting is present.

### 4. Visualization / Geometry

1. **Contour + Penalty Region**
    - Imagine elliptical contours of `J0(θ)` in the `(θ1, θ2)` plane.
    - Overlay a circle (`L2`) or diamond (`L1`) representing allowable weight magnitudes.
    - The optimal solution lies where a contour first touches this constraint region—pushing you toward simpler solutions.
2. **Weight Trajectories**
    - Plot each weight’s value over training epochs with and without regularization.
    - Regularized weights stay near zero, while unregularized weights swing widely to fit every data point.

### 5. Common Pitfalls & Tips

- Not tuning the regularization strength `λ`: too small has no effect; too large leads to underfitting.
- Forgetting to exclude bias terms from the penalty—regularizing biases can introduce unwanted shift constraints.
- Mixing up frameworks: some libraries’ “weight_decay” applies true weight decay, others add an L2 term to the loss.
- Using L1 when you need smooth shrinkage—L1 can cause unstable training if many weights flip sign near zero.

### 6. Practice Exercises

### Exercise 1: λ Sweep and Curves

- On the NumPy example above, vary `alpha` in `[0, 0.1, 1, 10, 100]`.
- Plot dev MSE versus `alpha`.
- Identify the range where regularization helps most.

### Exercise 2: L1 vs L2 Comparison

- Use `Lasso` (scikit-learn) alongside `Ridge`.
- Train both on the same dataset and report how many weights are zeroed by Lasso.
- Discuss when sparsity is more beneficial than smooth shrinkage.

### Exercise 3: Overfitting Visualization

- Generate a small 1D dataset from a noisy sinusoid.
- Fit a high-degree polynomial regression with and without L2 regularization.
- Plot both fitted curves against the true function and training points to see how regularization smooths the fit.

---

## Dropout Regularization

### 1. Concept Intuition

Dropout randomly “turns off” a subset of neurons during each training step.

- Imagine training many different thinned networks and averaging their predictions.
- By dropping units, you prevent co-adaptation—neurons can’t rely on specific peers, so each must learn robust, standalone features.
- At test time, you use the full network (with scaled weights) to approximate this ensemble’s average.

This ensemble view guards against overfitting: the model cannot memorize noise through complex co-dependencies, so it generalizes better.

### 2. Mathematical Breakdown

For one layer’s activations (a^{[l]}) and dropout keep probability (p):

1. Sample a binary mask
    
    ```python
    mask = (np.random.rand(*a.shape) < p).astype(float)
    ```
    
2. Apply mask and scale
    
    ```python
    a_drop = a * mask       # zero out dropped neurons
    a_drop /= p             # scale to maintain expectation
    ```
    
3. During backprop, gradients flow only through kept neurons:
    
    ```python
    dA = dA_next * mask / p
    ```
    

Because we divide by (p) at training time (“inverted dropout”), there’s no extra scaling at test time—forwarding uses the raw activations.

### 3. Code & Practical Application

### 3.1 NumPy Implementation (Forward & Backward)

```python
import numpy as np

def dropout_forward(A, keep_prob):
    mask = (np.random.rand(*A.shape) < keep_prob).astype(float)
    A_drop = (A * mask) / keep_prob
    cache = (mask, keep_prob)
    return A_drop, cache

def dropout_backward(dA_drop, cache):
    mask, keep_prob = cache
    dA = (dA_drop * mask) / keep_prob
    return dA

# Example usage in a single-layer network
np.random.seed(1)
A_prev = np.random.randn(5, 4)       # batch_size=5, neurons=4
A_drop, cache = dropout_forward(A_prev, keep_prob=0.8)
dA_prev = dropout_backward(A_drop, cache)
```

### 3.2 PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p_drop=0.5):
        super().__init__()
        self.fc1    = nn.Linear(input_dim, hidden_dim)
        self.relu   = nn.ReLU()
        self.dropout= nn.Dropout(p=p_drop)
        self.fc2    = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)      # active only in train mode
        x = self.fc2(x)
        return x

# Training sketch
model = SimpleNet(784, 128, 10, p_drop=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    # evaluate on dev/test without dropout
```

### 4. Visualization / Geometry

- **Network View**: At each mini-batch, you see a different sub-network. Over training, all possible subnetworks share weights.
- **Loss Surface Smoothing**: Dropout adds noise to the gradient updates, which helps the optimizer avoid sharp minima (often associated with poor generalization) and settle in wide, flat regions.
- **Ensemble Analogy**: Test-time inference approximates averaging predictions over an exponential number of thinned models, leading to a smoother decision boundary.

### 5. Common Pitfalls & Tips

- Forgetting to switch modes: In frameworks like PyTorch/TensorFlow, ensure `model.train()` enables dropout and `model.eval()` disables it.
- Setting (p) too high (e.g., 0.9 keep): overly aggressive dropout hurts learning. Typical keep rates: 0.8–0.5 for hidden layers, 0.9–0.8 for inputs.
- Applying dropout on small datasets: too much stochasticity can underfit.
- Combining dropout with Batch Normalization: order matters—apply dropout after batch-norm activations.
- Not tuning dropout rate via dev set: different architectures and data require different (p).

### 6. Practice Exercises

1. **NumPy Dropout from Scratch**
    - Integrate `dropout_forward` and `dropout_backward` into a two-layer neural network.
    - Train on a toy dataset (e.g., small spiral classification) and compare train/dev accuracy with and without dropout.
2. **Dropout Rate Sweep with PyTorch**
    - Train `SimpleNet` on MNIST for dropout rates in `[0.0, 0.25, 0.5, 0.75]`.
    - Plot train and dev accuracy versus dropout rate. Identify the sweet spot.
3. **Visualizing Ensemble Effect**
    - After training a dropout network, feed the same dev input multiple times in train mode, collect logits, and average them.
    - Compare this Monte Carlo average to a single eval-mode forward pass. Observe the variance reduction.

---

## Understanding Dropout

### 1. Concept Intuition

Dropout is a regularization technique that “drops” a random subset of neurons during each training step.

- Prevents co-adaptation: Neurons can’t rely on specific peers, so each one must learn features that work in many contexts.
- Acts like an ensemble: Training implicitly averages over many thinned networks, leading to smoother decision boundaries.
- Adds noise to gradients: Makes the loss surface flatter around minima, which correlates with better generalization.

In practice, dropout forces your network to be robust—no single neuron becomes indispensable, and the model can’t memorize idiosyncratic noise in the training set.

### 2. Mathematical Breakdown

For a layer’s pre-activation vector (z) and post-activation (a = g(z)), dropout with keep-probability `p` proceeds as:

```python
# 1. Sample a mask of zeros and ones
mask = (np.random.rand(*a.shape) < p).astype(float)

# 2. Zero out dropped units and scale up remaining ones
a_drop = (a * mask) / p

# 3. Store mask for backprop
cache = (mask, p)
```

During backprop, gradients through that layer get masked and scaled the same way:

```python
# dA_drop is gradient from layer above
mask, p = cache
dA = (dA_drop * mask) / p
```

Why divide by `p` (“inverted dropout”)? To keep the expected activation value the same at train and test time, so you don’t need to adjust anything during inference.

### 3. Code & Practical Application

### 3.1 NumPy from-Scratch: Two-Layer Net with Dropout

```python
import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    return dA * (Z > 0)

# Forward pass with dropout
def forward(X, W1, b1, W2, b2, keep_prob):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    mask1 = (np.random.rand(*A1.shape) < keep_prob).astype(float)
    A1_drop = A1 * mask1 / keep_prob

    Z2 = A1_drop.dot(W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid
    cache = (X, W1, b1, Z1, A1, mask1, keep_prob, W2, b2, Z2, A2)
    return A2, cache

# Backward pass with dropout
def backward(Y_pred, Y_true, cache):
    (X, W1, b1, Z1, A1, mask1, p, W2, b2, Z2, A2) = cache
    m = X.shape[0]

    # dZ2 and gradients for second layer
    dZ2 = (A2 - Y_true) / m
    dW2 = A1.dot(dZ2.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    # Backprop into hidden layer with dropout mask
    dA1_drop = W2.dot(dZ2)
    dA1 = dA1_drop * mask1 / p
    dZ1 = relu_backward(dA1, Z1)
    dW1 = X.T.dot(dZ1.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Example usage
np.random.seed(0)
X = np.random.randn(10, 5)      # 10 examples, 5 features
Y = (np.random.rand(10, 1) > .5).astype(float)

W1 = np.random.randn(5, 4)
b1 = np.zeros((4, 1))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

# Forward + backward
Y_pred, cache = forward(X, W1, b1, W2, b2, keep_prob=0.8)
grads = backward(Y_pred, Y, cache)
print("Computed gradients with dropout:", grads)
```

### 3.2 PyTorch: Dropout on MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Dataset and loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

# Model with dropout
class Net(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)      # dropout layer
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.drop(x)               # active only during train()
        x = self.fc2(x)
        return x

model = Net(p=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop snippet
for epoch in range(5):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    # Evaluate on dev set here
```

### 4. Visualization / Geometry

1. Sub-network Sampling
    
    ```
    Full Net:           A1 → A2 → A3
    Drop 30%: A1*_→ A2 → A3*_
    Drop 30%: A1 → A2*_→ A3
    ```
    
    Each arrowed path is a different thinned network. Training shares weights across these variations.
    
2. Loss Surface Smoothing
    - Without dropout: optimizer may hit sharp valleys.
    - With dropout: noisy gradient updates push you toward flatter regions—those generalize better.
3. Expected Activation
    - At train time: E[a_drop] = a
    - At test time: no mask, no scaling needed.

### 5. Common Pitfalls & Tips

- Forgetting mode switch: always call `model.train()` to activate dropout and `model.eval()` to disable it.
- Over-dropping: `keep_prob` too low (e.g., 0.3) can underfit—start around 0.8 for hidden layers.
- Input vs hidden: use gentler dropout on inputs (keep_prob ≥ 0.9) and stronger on deep layers.
- Dropout + BatchNorm: apply batch-norm before dropout to maintain stable activation statistics.
- Small datasets: dropout can add too much noise—consider simpler regularizers or data augmentation instead.

### 6. Practice Exercises

1. NumPy Network Comparison
    - Train the two-layer NumPy net above with `keep_prob` in `[1.0, 0.9, 0.7, 0.5]`.
    - Plot train/dev accuracy vs. `keep_prob`.
2. Monte Carlo Dropout Estimates
    - On a trained PyTorch model with dropout, run multiple forward passes in train mode on one input.
    - Compute mean and variance of the softmax outputs. Use this to estimate model uncertainty.
3. CIFAR-10 Dropout Sweep
    - Build a simple conv-net with `nn.Conv2d`, `ReLU`, `nn.Dropout2d`.
    - Vary dropout rates `[0.0, 0.2, 0.5]` and record test accuracy.

---

## Other Regularization Methods

Beyond L1/L2 and dropout, several complementary techniques help your models generalize better. Below we cover four widely used methods—each with intuition, math, code, visuals, pitfalls, and exercises.

### 1. Early Stopping

### 1.1 Concept Intuition

Early stopping halts training once performance on the dev set stops improving.

- The model begins by underfitting, then hits a sweet spot of low dev error, and eventually starts overfitting.
- Stopping at that sweet spot trades a small bias increase for a large variance reduction.

### 1.2 Mathematical Breakdown

Let

- $(\mathcal{L}_{\text{dev}}(t)) = dev loss  after  epoch (t).$
- $(t^* = \arg\min_t \mathcal{L}_{\text{dev}}(t)).$

Early stopping chooses parameters $(\theta_{t^*})$ instead of the final $(\theta_T)$.

### 1.3 Code & Practical Application

### Keras Example

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,           # wait 3 epochs after last improvement
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_dev, y_dev),
    epochs=100,
    callbacks=[early_stop]
)
```

### Manual Loop (PyTorch Sketch)

```python
best_dev = float('inf')
best_state = None
patience, trigger = 5, 0

for epoch in range(100):
    train_one_epoch()
    dev_loss = evaluate(X_dev, y_dev)
    if dev_loss < best_dev:
        best_dev, best_state, trigger = dev_loss, model.state_dict(), 0
    else:
        trigger += 1
        if trigger >= patience:
            model.load_state_dict(best_state)
            break
```

### 1.4 Visualization / Geometry

```
Dev Loss
  ^           .
  |         .   .
  |       .       Hit min (t*)
  |     .
  +--------------------> Epoch
             ↑ stop here
```

### 1.5 Common Pitfalls & Tips

- Too small patience → underfitting.
- Too large patience → wasted compute.
- Noisy dev curves: smooth with moving average.
- Always reset optimizer state if you reload best weights.

### 1.6 Practice Exercises

- Implement early stopping on your MNIST training loop manually.
- Plot train/dev loss and mark the stopping epoch.
- Experiment with patience ∈ {1,3,5} and compare final test accuracy.

### 2. Data Augmentation

### 2.1 Concept Intuition

Augmentation synthetically expands your dataset by applying label-preserving transforms (rotate, crop, color jitter).

- Exposes the model to more variations, reducing variance.
- Acts like adding new, realistic examples to the training set.

### 2.2 Mathematical Breakdown

Empirical risk over augmented set $(\tilde{\mathcal{D}}):$

$[ \hat{R}(\theta) ;=; \frac{1}{|\tilde{\mathcal{D}}|}\sum_{(x',y)\in \tilde{\mathcal{D}}} \ell\big(f(x';\theta),,y\big) ]$

where $(x') = (T(x)), (T)$ drawn from your augmentation distribution.

### 2.3 Code & Practical Application

### PyTorch Transforms

```python
from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor()
])

train_ds = datasets.CIFAR10(
    root='.', train=True, download=True, transform=augment
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
```

### TensorFlow ImageDataGenerator

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

model.fit(gen.flow(X_train, y_train, batch_size=32), epochs=20)
```

### 2.4 Visualization / Geometry

Show a grid of five augmented versions of a single input image. This illustrates how the model sees diverse views of the same class.

### 2.5 Common Pitfalls & Tips

- Over-augmenting: unrealistic transforms can confuse the model.
- Slow pipelines: use GPU-accelerated augmenters or prefetching.
- Inconsistent labels: ensure transforms preserve labels (no random erasing of digits to look like another).

### 2.6 Practice Exercises

- On CIFAR-10, apply mix of flips and color jitter. Measure dev accuracy with and without augmentation.
- Visualize 10 augmented samples for one image.
- Experiment with Cutout (randomly mask a square patch) and compare performance.

### 3. Batch Normalization

### 3.1 Concept Intuition

BatchNorm normalizes layer inputs to zero mean and unit variance per mini-batch, then applies learned scale and shift.

- Reduces internal covariate shift, stabilizes activations.
- Acts as a mild regularizer by adding noise from batch statistics.

### 3.2 Mathematical Breakdown

For activations (z^{(i)}) in a batch of size (m):

```python
mu = np.mean(z, axis=0)
var = np.var(z, axis=0)

z_norm = (z - mu) / np.sqrt(var + eps)
out    = gamma * z_norm + beta
```

- (\gamma,\beta) are trainable parameters.
- During backprop, gradients flow through normalization.

### 3.3 Code & Practical Application

### PyTorch

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # ...
)
```

### TensorFlow

```python
from tensorflow.keras.layers import BatchNormalization

x = BatchNormalization()(x)
```

### 3.4 Visualization / Geometry

- Plot activation distributions before and after BN across epochs.
- Show how mean/variance quickly stabilize around zero/one, reducing layer input drift.

### 3.5 Common Pitfalls & Tips

- Small batch sizes lead to noisy estimates—consider GroupNorm or LayerNorm.
- Place BN before activation (Conv → BN → ReLU).
- Do not combine with high dropout rates—BN noise + dropout noise can hurt convergence.

### 3.6 Practice Exercises

- Add BatchNorm to your feedforward MNIST model. Compare training speed and final dev accuracy.
- Track moving mean/variance over time and plot their evolution.
- Swap to LayerNorm and evaluate on small batches.

### 4. Label Smoothing

### 4.1 Concept Intuition

Label smoothing replaces one-hot targets with soft targets (e.g., 0.9 for true class, 0.1/(K–1) for others).

- Prevents the model from becoming over-confident.
- Reduces variance in the logits, improving calibration and generalization.

### 4.2 Mathematical Breakdown

For (K) classes and smoothing factor (\epsilon):

[ y_i^{\text{smooth}} = \begin{cases} 1 - \epsilon ;+; \epsilon/K, & \text{if } i=\text{true class}\ \epsilon/K, & \text{otherwise} \end{cases} ]

Cross-entropy uses these (y^{\text{smooth}}) instead of hard 0/1 labels.

### 4.3 Code & Practical Application

### PyTorch Example

```python
import torch.nn.functional as F

def smooth_cross_entropy(logits, target, eps=0.1):
    K = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    true_dist = torch.full_like(log_probs, eps / (K-1))
    true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    return torch.mean(-torch.sum(true_dist * log_probs, dim=1))
```

### TensorFlow

```python
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(..., loss=loss)
```

### 4.4 Visualization / Geometry

- Plot predicted softmax distributions with and without smoothing.
- Show calibration curves (confidence vs accuracy); smoothed models lie closer to the diagonal.

### 4.5 Common Pitfalls & Tips

- Too much smoothing (ε>0.2) can underfit.
- Don’t apply smoothing when you truly need one-hot for certain tasks (e.g., segmentation masks).
- Combine with other regularizers for best effect.

### 4.6 Practice Exercises

- Implement label smoothing in your classification code. Compare train/dev accuracy and calibration error.
- Sweep ε ∈ {0.0, 0.05, 0.1, 0.2} and plot test accuracy.
- Compute expected calibration error (ECE) for each ε to see the impact on confidence calibration.

---

## Normalizing Inputs

Normalizing inputs rescales features so they contribute equally during training. Without it, features with larger scales can dominate gradients, slowing convergence and harming model performance. By centering and scaling inputs, you stabilize training dynamics and help optimizers find good minima faster.

### 1. Common Normalization Methods

### 1.1 Z-score Standardization

Subtract the feature mean and divide by its standard deviation.

$[ z = \frac{x - \mu}{\sigma} ]$

After standardization, each feature has zero mean and unit variance.

### 1.2 Min-Max Scaling

Rescales features to lie in a fixed range, typically ([0,1]).

$[ x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} ]$

Preserves relative distances but is sensitive to outliers.

### 1.3 Whitening (PCA Whitening)

Decorrelates features by projecting onto principal components and scaling to unit variance.

$[ x_{\text{white}} = D^{-\tfrac12} E^\top (x - \mu) ]$

where (E) and (D) are eigenvectors and eigenvalues of the covariance matrix.

### 1.4 Per-Sample / Image Normalization

For images, normalize each pixel channel by channel-wise mean and standard deviation (computed over the training set).

### 2. Code & Practical Application

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torchvision import transforms
import tensorflow as tf

# 2.1 NumPy + scikit-learn
X = np.random.rand(100, 5)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)           # z-score

minmax = MinMaxScaler(feature_range=(0,1))
X_mm = minmax.fit_transform(X)           # min-max

# 2.2 PyTorch Image Normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# 2.3 TensorFlow Layer
normalization_layer = tf.keras.layers.Normalization()
normalization_layer.adapt(X)             # fit on training data
X_tf = normalization_layer(X)            # transform batch
```

### 3. Visualization / Geometry

```
Raw features      → cloud of points elongated along axes
After standardization → roughly spherical cloud
```

Spherical clusters indicate balanced scales, making gradient steps uniform in all directions.

### 4. Common Pitfalls & Tips

- Always fit scalers on the training set only, then apply to dev/test sets.
- Outliers skew min-max scaling; consider clipping or robust scaling (using medians and IQR).
- For streaming data, use incremental statistics or moving averages.
- Remember to invert transforms when interpreting model outputs or making predictions.

### 5. Practice Exercises

- Implement z-score and min-max scaling on the UCI Iris dataset. Compare classifier convergence rates.
- Visualize feature distributions before and after each normalization.
- Apply PCA whitening on a small synthetic dataset. Plot covariance matrices pre- and post-whitening.
- On CIFAR-10, measure training speed and final accuracy with and without per-channel image normalization.

---

## Vanishing and Exploding Gradients

Vanishing gradients occur when the derivatives in backpropagation shrink exponentially, making early layers learn extremely slowly. Exploding gradients happen when derivatives grow exponentially, causing unstable updates. Both issues stem from deep compositions of weight matrices and activation functions, and they can cripple training if left unaddressed.

### 1. Concept Intuition

Deep networks compute gradients via the chain rule, multiplying many small or large numbers together.

- If each term in the product is less than one, the overall gradient decays toward zero (vanishing).
- If each term exceeds one, the gradient blows up (exploding).

Imagine pushing a ball uphill with a very small or very large gear ratio—either you barely move it or you overshoot wildly.

### 2. Mathematical Breakdown

Given a loss $(\mathcal{L})$ and activations $(a^{(l)} = \sigma\big(z^{(l)}\big))$ where $(z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)})$, the gradient at layer (l) is:

$[ \frac{\partial \mathcal{L}}{\partial a^{(l)}}$ = $\Bigl(W^{(l+1),T},\mathrm{diag}\bigl(\sigma'!(z^{(l+1)})\bigr)\Bigr)$; $\frac{\partial \mathcal{L}}{\partial a^{(l+1)}}]$

Unrolling through (L) layers gives a product of up to (L) matrices and activation derivatives. If the spectral radii of these Jacobians are

- below 1 → gradients vanish
- above 1 → gradients explode

### 3. Code & Practical Application

### 3.1 Demonstrating Exploding Gradients in PyTorch

```python
import torch
import torch.nn as nn

# Simple 20-layer linear network
model = nn.Sequential(*[nn.Linear(100, 100) for _ in range(20)])
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# Dummy data
x = torch.randn(32, 100)
y = torch.randn(32, 100)

for i in range(5):
    optimizer.zero_grad()
    out = model(x)
    loss = nn.MSELoss()(out, y)
    loss.backward()
    # Measure gradient norm of first layer
    grad_norm = model[0].weight.grad.norm().item()
    print(f"Step {i}: grad_norm={grad_norm:.2e}")
    optimizer.step()
```

You’ll see gradient norms exploding after just a few steps.

### 3.2 Gradient Clipping to Control Explosion

```python
for i in range(5):
    optimizer.zero_grad()
    out = model(x); loss = nn.MSELoss()(out, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norm = model[0].weight.grad.norm().item()
    print(f"Clipped step {i}: grad_norm={grad_norm:.2e}")
    optimizer.step()
```

Clipping caps the gradient norm and stabilizes training.

### 4. Visualization / Geometry

```
Layer Index →
Grad Norm
  ^         Exploding: skyrockets after a few layers
  |          /\
  |         /
  |        /
  |       /  Vanishing: shrinks nearly to zero
  |      /
  +--------------------->
```

Plotting per-layer gradient norms reveals whether values trend toward zero or infinity.

### 5. Common Pitfalls & Tips

- Activation choice: avoid saturating sigmoids or tanh in very deep nets; prefer ReLU or variants.
- Initialization: use Xavier/Glorot for symmetric activations, He for ReLU, or orthogonal schemes to keep Jacobian spectral radius near one.
- Normalization layers: BatchNorm or LayerNorm rescale activations each batch, indirectly controlling gradient scale.
- Architectural fixes: residual or highway connections let gradients flow around problematic layers.
- Optimizer settings: too large a learning rate can amplify exploding gradients; clipping is essential for RNNs.

### 6. Practice Exercises

1. Build a 50-layer MLP with sigmoid activations. Track and plot gradient norms per layer—observe vanishing behavior.
2. Swap in ReLU and repeat. How do norms change?
3. Experiment with Xavier vs He initialization and visualize gradient distributions.
4. Implement a toy RNN on sequence data. Measure how gradient norms evolve over time steps and apply clipping.

---

## Weight Initialization for Deep Networks

Start by choosing initial weights that keep the variance of activations and gradients roughly constant as they flow through layers. Proper initialization prevents early saturation or explosion of signals, accelerates convergence, and sets you up for stable, deep training.

### 1. Why Initialization Matters

Getting your weight scales right from the start

- avoids activations saturating (all near zero or extreme values)
- ensures gradient norms neither vanish nor explode
- leads to faster convergence and better final accuracy

Without it, you spend epochs fighting poor signal propagation, masking your model’s capacity.

### 2. Common Initialization Schemes

### 2.1 Xavier (Glorot) Initialization

Designed for layers with symmetric activations like tanh.

- Sample weights from a uniform distribution between ±sqrt(6 / (fan_in + fan_out))
- Or from a normal distribution with std = sqrt(2 / (fan_in + fan_out))

This keeps forward and backward signal variances balanced.

### 2.2 He (Kaiming) Initialization

Optimized for ReLU and its variants.

- Sample weights from a normal distribution with std = sqrt(2 / fan_in)
- Or from a uniform distribution between ±sqrt(6 / fan_in)

This boosts the scale to account for half of units being zeroed by ReLU.

### 2.3 Orthogonal Initialization

Use a random orthogonal matrix (Q) scaled by a chosen gain.

- Generates weight matrices whose singular values are 1, preserving norm
- Good for very deep or recurrent networks

### 2.4 Zero and Small Constant Biases

Biases are usually initialized to zero or a small positive constant (e.g., 0.01)

- Zero biases avoid introducing unintended shifts early on
- Small positive biases on ReLU layers can help “fire” more neurons at the start

### 3. Code Examples

```python
# PyTorch example
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # Xavier uniform for tanh-style layers
        nn.init.xavier_uniform_(m.weight)
        # He normal for ReLU-style layers
        # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0.0)

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Linear(64*32*32, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10)
)
model.apply(init_weights)
```

```python
# TensorFlow / Keras example
from tensorflow.keras import layers, initializers

model = tf.keras.Sequential([
    layers.Conv2D(64, 3, padding='same',
                  kernel_initializer=initializers.HeNormal(),
                  bias_initializer='zeros',
                  activation='relu'),
    layers.Flatten(),
    layers.Dense(1000, kernel_initializer=initializers.GlorotUniform(),
                 bias_initializer=initializers.Zeros(),
                 activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 4. Practical Tips and Pitfalls

- Match initialization to activation: Xavier for sigmoid/tanh, He for ReLU.
- Don’t use the same scale for all layers if they have very different fan-in/fan-out.
- Always reinitialize weights when changing architecture depth or width.
- Watch out for small batch sizes: noisy gradient estimates can hide bad initialization.
- Combine with BatchNorm or LayerNorm to reduce sensitivity to exact initial scales.

### 5. Practice Exercises

- Build a 20-layer MLP on MNIST with ReLU. Compare convergence using Xavier vs He initializers.
- Swap to sigmoid activations and observe training collapse with He init, then fix with Xavier.
- Implement orthogonal initialization for each layer. Measure if deeper nets train faster or more stably.
- For a small CNN on CIFAR-10, record activation variances at each layer for different initializers.

---

## Numerical Approximation of Gradients

At its core, numerical gradient approximation uses finite differences to estimate how a function’s output changes with small tweaks to its inputs. It’s essential for debugging your backprop implementation and for interview questions on “gradient checking.”

### 1. Why and When to Use Numerical Gradients

- **Debugging**: Validate your analytic gradients in custom layers or loss functions.
- **Educational**: Build intuition for how derivatives work before diving into auto-diff.
- **Not for Training**: It’s too slow and sensitive to step size—use it only for checks, not real optimization.

### 2. Finite-Difference Formulas

1. **Forward Difference**
    
    gradient ≈ (f(x + ε) – f(x)) / ε
    
    - Easy but only first-order accurate (error ∝ ε).
2. **Backward Difference**
    
    gradient ≈ (f(x) – f(x – ε)) / ε
    
    - Same order as forward, mirrors it on the other side.
3. **Central Difference**
    
    gradient ≈ (f(x + ε) – f(x – ε)) / (2 × ε)
    
    - Second-order accurate (error ∝ ε²), more reliable for small ε.

Choose ε around 1e-4 to 1e-6 in most float32 settings.

### 3. Code Example: Gradient Check for a Single Scalar

```python
import numpy as np

def numeric_gradient(f, x, eps=1e-5):
    # f: function mapping R^n → R
    grad = np.zeros_like(x)
    for i in range(x.size):
        old_val = x.flat[i]

        x.flat[i] = old_val + eps
        fx_plus = f(x)

        x.flat[i] = old_val - eps
        fx_minus = f(x)

        grad.flat[i] = (fx_plus - fx_minus) / (2 * eps)
        x.flat[i] = old_val  # restore

    return grad

# Example: gradient of sum(x^3) should be 3*x^2
f = lambda x: np.sum(x**3)
x = np.array([1.0, 2.0, 3.0])
print("Numeric grad:", numeric_gradient(f, x))
print("True grad:   ", 3 * x**2)
```

### 4. Extending to Neural Networks

1. **Flatten Parameters**: Gather all weights/biases into a single vector θ.
2. **Compute Loss**: Define L(θ) on a small mini-batch.
3. **Numeric Gradient**: Call the routine above to get grad_num.
4. **Analytic Gradient**: Run your backprop to get grad_bp.
5. **Compare**:Assert that max(relative_error) < 1e-6.
    
    ```
    relative_error = |grad_num - grad_bp| / (|grad_num| + |grad_bp| + 1e-8)
    ```
    

### 5. Common Pitfalls & Tips

- ε too large → poor approximation; too small → floating-point noise dominates.
- Forgetting to restore the original parameter after each perturbation corrupts results.
- Checking on the full parameter vector is slow—sample a few indices instead.
- Always compute loss on the same data split for both numeric and analytic passes.

### 6. Visualization / Geometry

```
   f(x + ε)   ●
             |
             |
             |   slope ≈ (● - ○)/(ε)
             |
   f(x)      ○
             +----------→ x
```

Central difference uses points on both sides for a more symmetric slope estimate.

### 7. Practice Exercises

- Implement forward, backward, and central difference functions. Compare their errors on f(x)=sin(x).
- Write a small two-layer neural network from scratch (no frameworks), implement backprop, and validate every weight with numeric_gradient.
- Sample 10 random weights in a pretrained PyTorch model and verify gradients on a toy batch.
- Experiment with ε = [1e-2, 1e-4, 1e-6, 1e-8] and plot the error between numeric and analytic gradients.

---

## Gradient Checking

Gradient checking uses numerical finite differences to verify your backpropagation implementation. It’s a debugging tool—never use it for actual training since it’s too slow.

### 1. Concept Intuition

- Backprop gives you **analytic gradients** (`∂J/∂θ`) in closed form.
- Numerical gradients approximate these via small perturbations of each parameter:
    
    ```
    grad_num[i] ≈ (J(θ + ε·e_i) – J(θ – ε·e_i)) / (2·ε)
    ```
    
- Compare analytic vs numeric via a relative difference. If they match (e.g., <1e-6), your backprop is correct.

### 2. Mathematical Breakdown

Let θ be all your parameters flattened into a vector of size n. For each index i:

1. Define the basis vector e_i (1 at position i, else 0).
2. Compute loss with positive shift:
    
    ```python
    theta_plus  = theta.copy()
    theta_plus[i] += eps
    J_plus = compute_cost(X, Y, theta_plus)
    ```
    
3. Compute loss with negative shift:
    
    ```python
    theta_minus = theta.copy()
    theta_minus[i] -= eps
    J_minus = compute_cost(X, Y, theta_minus)
    ```
    
4. Central difference gives numeric gradient:
    
    ```python
    grad_num[i] = (J_plus - J_minus) / (2 * eps)
    ```
    
5. Analytic gradient from backprop is `grad_analytic[i]`.

Relative error per component:

```python
rel_error = abs(grad_num[i] - grad_analytic[i]) / (abs(grad_num[i]) + abs(grad_analytic[i]) + 1e-8)
```

Finally, check the maximum relative error across all i.

### 3. Code & Practical Application

Below is a NumPy example for a simple two-layer network. You’ll see how to:

- Flatten parameters
- Compute both numeric and analytic gradients
- Compare and report the maximum difference

```python
import numpy as np

# 3.1 Forward and cost for a 2-layer network
def forward_and_cost(X, Y, params):
    # params is a dict with 'W1','b1','W2','b2'
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    m = X.shape[0]

    Z1 = X.dot(W1) + b1
    A1 = np.maximum(0, Z1)           # ReLU
    Z2 = A1.dot(W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))       # Sigmoid

    # Cross-entropy cost
    cost = -np.sum(Y * np.log(A2 + 1e-8) + (1-Y)*np.log(1-A2 + 1e-8)) / m
    cache = (X, Z1, A1, Z2, A2)
    return cost, cache

# 3.2 Backward pass to get analytic grads
def backward(cache, params, Y):
    X, Z1, A1, Z2, A2 = cache
    W2 = params['W2']
    m = X.shape[0]

    dZ2 = (A2 - Y) / m
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    return grads

# 3.3 Utility: flatten and unflatten
def flatten_params(params):
    keys, shapes, flat_vals = [], [], []
    for k, v in params.items():
        keys.append(k)
        shapes.append(v.shape)
        flat_vals.append(v.ravel())
    theta = np.concatenate(flat_vals)
    return theta, keys, shapes

def unflatten_theta(theta, keys, shapes):
    params = {}
    offset = 0
    for k, shape in zip(keys, shapes):
        size = np.prod(shape)
        params[k] = theta[offset:offset+size].reshape(shape)
        offset += size
    return params

# 3.4 Gradient check routine
def gradient_check(X, Y, params, eps=1e-7, threshold=1e-6):
    # 1. Compute analytic gradients
    cost, cache = forward_and_cost(X, Y, params)
    grads = backward(cache, params, Y)

    # 2. Flatten parameters and grads
    theta, keys, shapes = flatten_params(params)
    grad_analytic, _, _ = flatten_params(grads)

    # 3. Compute numeric gradients
    grad_numeric = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus  = theta.copy(); theta_plus[i] += eps
        theta_minus = theta.copy(); theta_minus[i] -= eps

        params_plus  = unflatten_theta(theta_plus, keys, shapes)
        params_minus = unflatten_theta(theta_minus, keys, shapes)

        J_plus, _  = forward_and_cost(X, Y, params_plus)
        J_minus, _ = forward_and_cost(X, Y, params_minus)

        grad_numeric[i] = (J_plus - J_minus) / (2 * eps)

    # 4. Compute relative error
    numerator   = np.abs(grad_numeric - grad_analytic)
    denominator = np.abs(grad_numeric) + np.abs(grad_analytic) + 1e-8
    rel_errors  = numerator / denominator

    max_error = np.max(rel_errors)
    print("Max relative error:", max_error)
    if max_error < threshold:
        print("Gradient check passed!")
    else:
        print("Gradient check failed. Inspect your backprop.")

    return max_error

# 3.5 Example run
np.random.seed(0)
X_sample = np.random.randn(5, 3)
Y_sample = (np.random.rand(5, 1) > 0.5).astype(float)

# Initialize parameters
params = {
    'W1': np.random.randn(3, 4) * 0.01,
    'b1': np.zeros((1, 4)),
    'W2': np.random.randn(4, 1) * 0.01,
    'b2': np.zeros((1, 1))
}

gradient_check(X_sample, Y_sample, params)
```

### 4. Visualization / Geometry

```
Loss J(θ)
   ^
   |      ●      ●
   |    ●   ●     ← central diff uses both sides
   |  ●       ●
   +----------------→ θ
```

Plotting J(θ+ε), J(θ), J(θ−ε) illustrates the slope estimate.

### 5. Common Pitfalls & Tips

- **Restoring θ**: Always revert to the original parameter after each perturbation.
- **Step size ε**: 1e-7 is typical; too small → numerical noise, too large → coarse approximation.
- **Performance**: Checking every parameter is O(n·cost); sample a handful of indices when n is large.
- **Bias in grad shape**: Ensure you flatten and unflatten with consistent ordering.
- **Cost consistency**: Use the same mini-batch X,Y for both forward passes and backprop.

### 6. Practice Exercises

1. **Selective Check**: For a deeper 3-layer network, randomly pick 10 parameters and run gradient_check on them.
2. **Different ε**: Sweep ε in `[1e-4,1e-5,1e-6,1e-7]`, record max relative error, and plot error vs ε.
3. **Framework Check**: Extract parameters and analytic grads from a small PyTorch model on a toy batch. Use the numeric routine above to verify them.
4. **Vectorized Finite Differences**: Research and implement a vectorized approach that perturbs multiple θ’s at once for speed (using identity matrix slices).

---