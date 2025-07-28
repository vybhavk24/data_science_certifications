# ML_c1_m3

## Classification with Logistic Regression – Motivation

### Pre-requisites

- Understanding of univariate linear regression (hypothesis, parameters, cost)
- Basic probability concepts (odds, probabilities in [0,1])
- Comfort with vector–matrix notation for inputs and parameters

If you need a quick refresher on how linear regression predicts unbounded real values, revisit the hypothesis `h(x) = θ0 + θ1·x`.

### 1. Why Not Use Linear Regression for Classification?

- Linear regression outputs can be any real number, not constrained to a probability range.
- If you threshold a linear prediction (e.g., predict class 1 if `θᵀx > 0`), extreme values can dominate and cause erratic decision boundaries.
- Errors on points near the class boundary get treated the same as points far away, making training unstable.

We need a model that

1. Outputs values in [0,1] to represent class probabilities
2. Learns a smooth boundary that’s robust to outliers

### 2. Introducing the Sigmoid (Logistic) Function

The sigmoid squashes any real-valued input into a probability:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

- Input `z = θ0 + θ1·x1 + … + θn·xn` can be negative or positive
- Output `sigmoid(z)` always lies in (0,1)
- The S-shaped curve ensures gradual transitions near the decision threshold

### 3. Hypothesis and Interpretation

For a feature vector **x** and parameters **θ**, the logistic regression hypothesis is:

```
h_theta(x) = sigmoid( θ0 + θ1·x1 + … + θn·xn )
```

- `h_theta(x)` represents the estimated probability that `y = 1` given **x**
- We predict class 1 if `h_theta(x) ≥ 0.5`, else class 0

### 4. Decision Boundary

Setting `h_theta(x) = 0.5` yields:

```
θ0 + θ1·x1 + … + θn·xn = 0
```

This linear equation defines a hyperplane (line in 2D, plane in 3D) separating the two classes. Points on one side get probability above 0.5, on the other below 0.5.

### 5. Real-World Applications

- **Spam Detection:** Classify email as spam (1) or not (0) based on word frequencies.
- **Medical Diagnosis:** Predict disease presence from patient measurements.
- **Credit Scoring:** Estimate probability of default using financial features.
- **Click-Through Prediction:** Model whether an ad will be clicked (yes/no) from user context.

In each case, outputs need to be interpretable probabilities, and logistic regression provides that.

### 6. Visual Insight

- Plotting `sigmoid(z)` shows a smooth S-curve, with most “learning” happening in the transition region around 0.
- Overlay data points in 2D: see how the logistic curve yields a soft boundary before hard thresholding.

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 200)
plt.plot(z, sigmoid(z))
plt.xlabel('z = θᵀx')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()
```

### 7. Practice Problems

1. **Hands-On Sigmoid:**
    - Implement the `sigmoid` function from scratch.
    - Generate `z` values in [−5,5] and plot the curve.
2. **Simple Binary Classifier:**
    - Create a tiny dataset with 2D points and labels (0 or 1).
    - Manually choose `θ` values to separate them and plot the probability contours.
3. **Interpreting Odds:**
    - Show how `θᵀx = 2` corresponds to odds `e^2 : 1` (i.e., probability ≈ 0.88).
    - Compute probabilities for several `z` values and comment on the odds interpretation.

---

## Logistic Regression

### Pre-requisites

- Single-variable and multivariate linear regression
- The sigmoid (logistic) function concept
- Basic probability (odds, probabilities in [0,1])
- Gradient descent fundamentals
- Python and NumPy basics

If any feel shaky, review how linear regression predicts real values and how gradient descent updates parameters.

### 1. Intuition & Motivation

- **Goal**: Predict a binary outcome (0 or 1) and output a probability.
- **Why Not Linear?**
    - Linear model can output <0 or >1, not valid probabilities.
    - Thresholding a linear output gives a hard boundary and unstable training.
- **Analogy**: Think of a door with a dimmer switch.
    - The switch position (θᵀx) can be any number.
    - The sigmoid “dimmer” squashes it smoothly into brightness levels (0–1), letting you gauge “how likely” you are to open the door.

### 2. Hypothesis & Sigmoid Function

```
# Linear part (logit)
z = θ0 + θ1·x1 + θ2·x2 + … + θn·xn

# Sigmoid (maps z to probability in (0,1))
h_theta(x) = sigmoid(z)
sigmoid(z) = 1 / (1 + exp(-z))
```

- `h_theta(x)` is the estimated P(y=1 | x).
- Predict **1** if `h_theta(x) ≥ 0.5`, else **0**.

### 3. Decision Boundary

Setting `h_theta(x) = 0.5` gives the linear equation:

```
θ0 + θ1·x1 + θ2·x2 + … + θn·xn = 0
```

This hyperplane in n-dimensional space separates the two classes.

### 4. Cost Function: Log Loss (Cross-Entropy)

To train θ, we minimize the average negative log-likelihood:

```
J(theta) = -(1/m) * sum(
  y(i)*log(h_theta(x(i)))
  + (1 - y(i))*log(1 - h_theta(x(i)))
  for i = 1..m
)
```

- `y(i)` is 0 or 1.
- This cost punishes confident but wrong predictions heavily and is **convex**, so it has a single global minimum.

### 5. Gradient & Parameter Update

### 5.1 Vectorized Gradient

```
# Compute predictions
predictions = sigmoid( X.dot(theta) )   # shape: (m,1)

# Compute error vector
errors = predictions - y                 # shape: (m,1)

# Gradient of log-loss
gradient = (1/m) * X.T.dot(errors)       # shape: (n+1,1)
```

- `X` is m×(n+1) (first column ones)
- `theta` is (n+1)×1
- `y` is m×1

### 5.2 Gradient Descent Update

```
theta = theta - alpha * gradient
```

- `alpha` is the learning rate controlling step size.

### 6. Regularization

To prevent overfitting, add a penalty on θ₁…θₙ (not θ₀):

```
# L2 regularization (Ridge)
J_reg = J + (lambda/(2*m)) * sum(theta[j]^2 for j=1..n)

# Gradient with L2 penalty
gradient_reg[j] = gradient[j] + (lambda/m) * theta[j]   for j=1..n
gradient_reg[0] = gradient[0]                            # no penalty on bias
```

- `lambda` (λ) controls regularization strength.

### 7. Geometric & Visual Insights

- **Sigmoid Curve**
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    z = np.linspace(-10, 10, 200)
    plt.plot(z, 1/(1+np.exp(-z)))
    plt.xlabel('z = θᵀx'); plt.ylabel('sigmoid(z)')
    plt.title('Sigmoid Function'); plt.grid(True)
    plt.show()
    ```
    
    The S-shape shows smooth probability transitions.
    
- **Decision Boundary in 2D**
    
    Plot data points with `x1,x2` and overlay the line `θ0 + θ1·x1 + θ2·x2 = 0`.
    
- **Cost Surface**
    
    In simple cases (two parameters), visualize `J(theta0,theta1)` as a convex bowl.
    

### 8. Real-World ML Workflow

1. **Data Prep**
    - One-hot encode categoricals
    - Scale numeric features
2. **Model Pipeline**
    - Plug into pipeline: scaler → logistic regression (with `penalty='l2'`)
3. **Train/Validate**
    - Use k-fold cross-validation to pick `alpha` (learning rate) or `C` (inverse λ)
4. **Evaluate**
    - Metrics: accuracy, precision, recall, F1, ROC AUC
    - Calibration plots to check predicted probabilities
5. **Deploy**
    - Save model and scaler
    - Expose probability API for downstream systems

### 9. Python Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import accuracy_score, roc_auc_score

# 1. Generate synthetic data
np.random.seed(0)
m = 200
X = np.random.randn(m,2)
# Define true boundary and labels
y = (X[:,0] + 2*X[:,1] > 0).astype(int).reshape(-1,1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3. Scale
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# 4. Train logistic regression with L2 penalty
model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')
model.fit(X_train_s, y_train.ravel())

# 5. Predict & evaluate
y_pred   = model.predict(X_test_s)
y_proba  = model.predict_proba(X_test_s)[:,1]
acc       = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.3f}, ROC AUC: {roc_auc:.3f}")
```

### 10. Practice Problems

1. **From-Scratch Logistic GD**
    - Implement `sigmoid`, `compute_cost`, and `gradient_descent` for logistic regression.
    - Test on a small 2D dataset and plot decision boundary.
2. **Regularization Tuning**
    - Load the breast cancer dataset (`sklearn.datasets.load_breast_cancer`).
    - Use `LogisticRegression` with different `C` values (0.01, 0.1, 1, 10).
    - Plot validation ROC AUC vs. `C`.
3. **Imbalanced Data**
    - Create a dataset with a 90:10 class split.
    - Train logistic regression with and without `class_weight='balanced'`.
    - Compare precision, recall, and F1 scores.

### 11. Key Takeaways

- Logistic regression models P(y=1|x) with a sigmoid over a linear function.
- Log-loss is convex and well-suited for gradient-based optimization.
- Regularization (L2 or L1) prevents overfitting high-dimensional data.
- Real-world use spans spam detection, medical diagnosis, credit scoring, click-through prediction.
- Understanding the math, code, and workflow makes you interview-ready and ML-practical.

---

## Decision Boundaries

### Pre-requisites

- Binary classification fundamentals
- Geometry of hyperplanes and manifolds
- Probability thresholds (e.g., P(y=1|x) ≥ t)
- Python with NumPy and scikit-learn basics
- Plotting with Matplotlib

### 1. What Is a Decision Boundary?

A decision boundary is the locus of points in feature space where a classifier switches its prediction from one class to another. For binary problems, it’s the set of all x satisfying:

```
P(y = 1 | x) = threshold
```

Commonly the threshold is 0.5, but any value in (0,1) can be used to tune precision/recall trade-offs.

### 2. Linear Decision Boundaries

### 2.1 Logistic Regression

- Hypothesis:
    
    ```
    hθ(x) = sigmoid(θᵀx) = 1 / (1 + e^(−θᵀx))
    ```
    
- Boundary for threshold t:
    
    ```
    θᵀx = log( t / (1 − t) )
    ```
    
- For t = 0.5, simplifies to θᵀx = 0.

### 2.2 Perceptron & Linear SVM

- Perceptron: θᵀx + b = 0
- SVM: same hyperplane, but maximizes margin.
- Decision function sign:
    
    ```
    f(x) = sign(θᵀx + b)
    ```
    

### 3. Nonlinear Decision Boundaries

### 3.1 k-Nearest Neighbors

- Boundary is the union of Voronoi cell edges around each training point.
- Highly flexible, piecewise constant regions.

### 3.2 Kernel SVM

- Uses kernel φ(x) to map to higher dims.
- Boundary:
    
    ```
    ∑ᵢ αᵢ yᵢ K(x, xᵢ) + b = 0
    ```
    
- Can be circles, curves, complex manifolds.

### 3.3 Decision Trees

- Axis-aligned splits produce rectangular regions.
- Boundary is a union of axis-parallel segments or boxes.

### 3.4 Neural Networks

- With ReLU activations, boundary is a piecewise linear manifold.
- Depth and width control the number of linear pieces.

### 4. Geometric & Visual Intuition

- **2D Linear**: straight line separating points.
- **2D Nonlinear**: curves, ripples, boxes depending on model.
- **Margin (SVM)**: parallel lines on either side of boundary where f(x)=±1.

Example contour of P(y=1|x) from logistic regression:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# sample data
np.random.seed(0)
X = np.vstack([np.random.randn(50,2) + [2,2], np.random.randn(50,2) - [2,2]])
y = np.array([1]*50 + [0]*50)

model = LogisticRegression().fit(X, y)

# grid
xx, yy = np.meshgrid(np.linspace(-5,5,200), np.linspace(-5,5,200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:,1].reshape(xx.shape)

# plot
plt.contourf(xx, yy, probs, levels=20, cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, probs, levels=[0.5], colors='black')
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plt.title("Logistic Regression Decision Boundary")
plt.show()
```

### 5. Threshold Adjustment & Boundary Shift

Changing your probability threshold t moves the boundary:

- New boundary:
    
    ```
    θᵀx = log( t / (1 − t) )
    ```
    
- Higher t → boundary shifts toward positive samples (fewer positives, higher precision).
- Lower t → boundary shifts toward negative samples (more positives, higher recall).

### 6. Model Capacity & Boundary Complexity

| Model Type | Boundary Shape | Flexibility | Overfitting Risk |
| --- | --- | --- | --- |
| Logistic/Linear | Hyperplane (flat) | Low | Low |
| k-NN | Voronoi cells (jagged) | Very High | Very High |
| SVM (kernel) | Curved manifolds | Tunable via kernel & C | Medium–High |
| Decision Tree | Axis-aligned boxes | Tunable via depth | High if deep |
| Neural Network | Piecewise linear / nonlin | Very High | High without regularization |

### 7. High-Dimensional Challenges

- **Visualization**: Use pair-plots, PCA/TSNE projections, or slice through two features at a time.
- **Interpretability**: In high dims, analytic boundary equations get complex. Use LIME or SHAP to explain local decisions.
- **Scaling**: Boundary shape depends on feature scaling and regularization. Always preprocess consistently.

### 8. Interview-Ready Talking Points

1. **Derive** the logistic boundary from P(y=1|x)=0.5.
2. **Explain** how changing threshold affects false-positive/false-negative rates.
3. **Compare** linear vs kernel vs tree boundaries and their bias-variance trade-offs.
4. **Discuss** how regularization shapes the boundary (e.g., SVM C or logistic λ).
5. **Visualize** on a 2D scatter: sketch different boundary shapes.

### 9. Practice Exercises

1. **Plot multiple thresholds**
    - On a synthetic logistic dataset, overlay boundaries for t=0.3, 0.5, and 0.7.
2. **Compare classifiers**
    - Train logistic, k-NN (k=1, k=20), SVM (RBF), and a decision tree.
    - Plot all boundaries on the same 2D dataset.
3. **High-dim slice**
    - Load the Iris dataset.
    - Select two features, plot decision boundaries for each class vs rest.

---

## Cost Function for Logistic Regression

### Pre-requisites

- Basic probability concepts (logarithms, odds)
- Understanding of the sigmoid function
- Familiarity with hypothesis `h(x)` for logistic regression
- Vectorized operations in NumPy

### 1. Intuition: Why We Need a Special Cost

Using squared error (`(h(x) − y)²`) for classification leads to a non-convex optimization and poor gradient signals when predictions saturate.

Instead, we derive a cost from the **log-likelihood** of our binary labels, which:

- Penalizes confident but wrong predictions heavily
- Produces a **convex** surface with a single global minimum
- Gives clear gradient signals even when predictions are near 0 or 1

### 2. Per-Example Logistic Loss

For a single example `(x⁽ⁱ⁾, y⁽ⁱ⁾)` where `y` is 0 or 1, define:

```
h = sigmoid( θᵀ x )              # model’s estimated P(y=1 | x)

if y == 1:
    loss = -log(h)
else:  # y == 0
    loss = -log(1 - h)
```

Combined in one formula:

```
loss = -y * log(h) - (1 - y) * log(1 - h)
```

- When `y=1`, the second term drops out and we incur `log(h)`.
- When `y=0`, the first term drops out and we incur `log(1−h)`.

### 3. Overall Cost Function (Vectorized)

For `m` training examples:

```python
# X: m×(n+1) design matrix with bias column
# y: m×1 vector of labels (0 or 1)
# theta: (n+1)×1 parameter vector

# 1. Compute predictions
z = X.dot(theta)                  # shape: (m, 1)
h = 1 / (1 + np.exp(-z))          # sigmoid, shape: (m, 1)

# 2. Compute vectorized loss
term1 = y.T.dot(np.log(h))                             # shape: (1,1)
term2 = (1 - y).T.dot(np.log(1 - h))                   # shape: (1,1)
J = -(1 / m) * (term1 + term2).item()                  # scalar cost
```

### 4. Gradient of the Cost

The gradient with respect to all parameters θ is:

```python
gradient = (1 / m) * X.T.dot(h - y)   # shape: (n+1, 1)
```

Then update via gradient descent:

```python
theta = theta - alpha * gradient
```

### 5. Convexity and Optimization

- The logistic cost surface is **convex**, ensuring any gradient‐based method reaches the global minimum.
- In practice, we use:
    - **Batch gradient descent** (all examples per update)
    - **Stochastic** or **mini-batch** variants for large m
    - **Advanced optimizers** (Adam, RMSprop) for deep networks

### 6. Real-World ML Use

- **Binary classification**: spam detection, medical diagnosis, fraud detection
- As the **output layer** in neural networks with cross-entropy loss
- Provides **probabilistic outputs** for downstream decision-making

Most ML frameworks implement logistic loss under names like `binary_crossentropy`.

### 7. Visual Insight: Loss Curve

Plot per-example loss versus `h`:

```python
import numpy as np
import matplotlib.pyplot as plt

h = np.linspace(0.001, 0.999, 200)
loss_pos = -np.log(h)           # y=1
loss_neg = -np.log(1 - h)       # y=0

plt.plot(h, loss_pos, label='y=1 loss')
plt.plot(h, loss_neg, label='y=0 loss')
plt.xlabel('h = P(y=1|x)')
plt.ylabel('loss')
plt.title('Logistic Loss Curves')
plt.legend()
plt.show()
```

- As `h→0` for a positive label, `loss_pos` → ∞
- As `h→1` for a negative label, `loss_neg` → ∞

This harsh penalty drives learning away from confident mistakes.

### 8. Practice Problems

1. **Implement `compute_cost_logistic`**
    - Write a function that takes `X`, `y`, `theta` and returns `J`.
    - Test on a tiny dataset:
        
        ```python
        X = np.array([[1, 0],
                      [1, 2],
                      [1, 4]])
        y = np.array([[0],
                      [1],
                      [1]])
        theta = np.zeros((2,1))
        ```
        
    - Print cost and compare against manual calculation.
2. **Gradient Checking**
    - Numerically approximate gradient via finite differences:
        
        ```python
        eps = 1e-4
        approx_grad[j] = (J(theta + eps * e_j) - J(theta - eps * e_j)) / (2*eps)
        ```
        
    - Compare to your analytic `gradient`.
3. **Plot Decision Boundary and Loss Surface**
    - Create a 2D dataset separable by a line.
    - Compute and plot the cost surface over a grid of `(θ0, θ1)` for fixed `θ2`.
    - Overlay gradient descent path to the minimum.
4. **Regularized Logistic Cost**
    - Add L2 penalty:
        
        ```
        J_reg = J + (lambda/(2*m)) * sum(theta[j]^2 for j=1..n)
        ```
        
    - Update your cost and gradient functions, then train on a synthetic noisy dataset to see the effect.

---

## Simplified Cost Function for Logistic Regression

Logistic regression predicts probability with

```
h = sigmoid(z)
  where z = θᵀx
        sigmoid(z) = 1 / (1 + exp(-z))
```

### 1. Per-example Loss

For a single example `(x, y)` where `y` is 0 or 1, the loss is:

```
loss = -y * log(h) - (1 - y) * log(1 - h)
```

### 2. Total Cost over m Examples

Average the per-example loss across all `m` training examples:

```
J(θ) = (1 / m) * sum(
        -y(i) * log(h(i))
        - (1 - y(i)) * log(1 - h(i))
      ) for i = 1..m
```

### 3. Vectorized Form

Using matrix/vector notation where

- `X` is m×(n+1) feature matrix (including bias)
- `y` is m×1 label vector
- `h = sigmoid(X · θ)` is m×1 prediction vector

you can write:

```
J(θ) = -(1 / m) * [
         yᵀ · log(h)
         + (1 - y)ᵀ · log(1 - h)
       ]
```

---

## Gradient Descent and Logistic Regression

### Gradient Descent

Gradient descent iteratively updates parameters to minimize a cost function. You need a feature matrix `X`, label vector `y`, initial `theta`, learning rate `alpha`, and number of iterations `num_iters`.

```python
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        # Compute prediction and error
        preds = X.dot(theta)
        error = preds - y

        # Compute gradient and update theta
        gradient = (1 / m) * X.T.dot(error)
        theta = theta - alpha * gradient

    return theta
```

### Logistic Regression with Gradient Descent

Replace the linear prediction with a sigmoid, use the binary cross-entropy cost, then update via gradient descent.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = -(1/m) * np.sum(
        y * np.log(h) + (1 - y) * np.log(1 - h)
    )
    return cost

def logistic_gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    costs = []

    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta = theta - alpha * gradient

        # Optional: track cost for convergence plots
        costs.append(compute_cost(X, y, theta))

    return theta, costs
```

Key steps:

- Compute `h = sigmoid(X · θ)`
- Compute cost:
    
    ```
    cost = -(1/m) * sum( y*log(h) + (1-y)*log(1-h) )
    ```
    
- Update parameters:
    
    ```
    θ := θ − α * (1/m) * Xᵀ · (h − y)
    ```
    

### Logistic Regression with scikit-learn

scikit-learn handles optimization, regularization, and convergence checks for you.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and fit the model
model = LogisticRegression(
    penalty='none',       # no regularization
    solver='saga',        # supports no-penalty option
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions and probabilities
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, preds)
logloss = log_loss(y_test, probs)

print(f"Accuracy: {accuracy:.3f}")
print(f"Log Loss: {logloss:.3f}")
```

Common hyperparameters:

- `penalty` (e.g., `'l2'`, `'l1'`, or `'none'`)
- `C` (inverse of regularization strength)
- `solver` (e.g., `'lbfgs'`, `'liblinear'`, `'saga'`)
- `max_iter` (number of optimization steps)

---

## The Problem of Overfitting

### Prerequisites

- Basic understanding of supervised regression (linear or polynomial).
- Familiarity with training vs. validation/test splits.
- Knowledge of the mean squared error cost:
    
    ```
    J = (1/m) * sum((h(x_i) - y_i)^2)  for i = 1..m
    ```
    

### 1. Intuition: When Your Model Learns Too Much

Imagine fitting a smooth curve through noisy data points.

- A **simple model** (straight line) might miss subtle trends (“underfit”).
- A **complex model** (high-degree polynomial) can twist to hit every point exactly, including noise (“overfit”).

Overfitting means your model memorizes noise and outliers, so it performs well on training data but poorly on new data.

### 2. Key Math: Training vs. Generalization Error

Track two costs:

```
J_train = (1/m_train) * sum((h(x_i) - y_i)^2)   for training set
J_test  = (1/m_test)  * sum((h(x_j) - y_j)^2)   for hold-out set
```

- When complexity ↑:
    - J_train ↓ (model can flex to fit every point)
    - J_test ↓ until a point, then ↑ (model fits noise, hurts new data)

Plotting complexity vs. error gives the classic **U-shaped** test error curve, marking the sweet spot for model simplicity.

### 3. Real-World ML Workflows

In practice, overfitting shows up as:

- High accuracy on training logs but low validation accuracy.
- Validation loss plateauing or rising while training loss keeps falling.

Common defenses:

- **Cross-validation**: rotate your validation set to estimate generalization.
- **Regularization**: add penalty terms to cost (L1/L2).
- **Early stopping**: halt training when validation error stops improving.
- **More data**: noise averages out with larger samples.
- **Feature selection**: remove irrelevant or redundant inputs.

### 4. Visual or Geometric Insight

Plotting polynomial fits makes overfitting obvious:

1. Generate 20 noisy points from a sine curve.
2. Fit polynomials of degree 1, 3, and 15.
3. Visualize:
    - Degree 1: misses wiggles (underfit).
    - Degree 3: captures main shape (good fit).
    - Degree 15: loops through every noise bump (overfit).

### 5. Practice Problems

1. Polynomial Regression and Error Curves
    - Create synthetic data:
        
        ```python
        import numpy as np
        X = np.linspace(0, 1, 20)
        y = np.sin(2 * np.pi * X) + np.random.randn(20) * 0.2
        ```
        
    - For degrees d in `[1, 3, 5, 9, 15]`:
        - Build feature matrix `X_poly` with columns `[X**0, X**1, …, X**d]`.
        - Fit `theta` by normal equation or gradient descent.
        - Compute and plot `J_train` and `J_test` on a fixed test split.
2. Cross-Validation Split
    - Write a function `k_fold_split(X, y, k)` that yields (X_train, y_train, X_val, y_val) for each fold.
    - Use it to estimate average validation error for polynomial degrees and pick the best.
3. Early Stopping
    - Implement gradient descent that tracks validation loss each epoch.
    - Stop when validation loss hasn’t decreased for 5 consecutive epochs.
    - Compare the final degree-9 model with and without early stopping.

---

## Addressing Overfitting

### Prerequisites

- Understanding of overfitting vs. underfitting
- Familiarity with training, validation, and test splits
- Basic cost function and gradient descent

### 1. Intuition: Why We Constrain Our Models

If a model memorizes every noisy fluctuation in training data, it fails on new examples.

Think of a student who rote-learns practice questions without grasping core concepts. They ace the homework but stumble on fresh problems.

### 2. Core Techniques

### 2.1 Regularization

Add a weight-shrinkage penalty to the cost so large coefficients get discouraged.

Linear regression with L2 (ridge) penalty:

```
J(theta) = (1/m) * sum((h(x_i) - y_i)^2 for i=1..m)
           + (lambda/(2*m)) * sum(theta_j^2 for j=1..n)
```

Logistic regression with L2 penalty:

```
J(theta) = -(1/m) * sum(y_i*log(h_i) + (1-y_i)*log(1-h_i) for i=1..m)
           + (lambda/(2*m)) * sum(theta_j^2 for j=1..n)
```

- lambda (a.k.a. alpha in some libraries) controls how much we shrink coefficients.
- Larger lambda → simpler model, higher bias, lower variance.

### 2.2 Early Stopping

Stop training when validation error stops improving:

1. Split off a validation set.
2. After each epoch, compute validation loss.
3. If validation loss hasn’t decreased for `patience` epochs, halt training.

### 2.3 Cross-Validation

Use k-fold cross-validation to get a reliable estimate of generalization:

1. Split data into k equal folds.
2. For each fold: train on k−1 folds, validate on the held-out fold.
3. Average validation errors across folds to compare models or hyperparameters.

### 2.4 Data Augmentation

Expand your dataset by applying transformations:

- In images: rotations, flips, crops, color shifts
- In text: synonym replacement, paraphrasing
- In time series: jittering, scaling, time warping

This teaches models to focus on robust patterns, not noise.

### 2.5 Feature Selection & Dimensionality Reduction

Remove or combine inputs that add little new information:

- Manual selection guided by domain knowledge
- Automated methods like recursive feature elimination
- PCA to project data into a lower‐dimensional space

### 2.6 Dropout (Neural Networks)

Randomly “drop” a fraction of units each training step so the network can’t rely on any single pathway:

```
layer_output = Dropout(rate=0.5)(previous_layer)
```

This forces redundant representations and reduces co-adaptation of features.

### 3. Real-World ML Workflows

### scikit-learn Example: Ridge Regression

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

model = Ridge(alpha=1.0)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print('Average CV MSE:', -scores.mean())
```

### Keras Example: Early Stopping & Dropout

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[es])
```

### 4. Visual Insight

Plot training and validation error as you vary lambda in ridge regression:

1. For each `lambda` in `[0.01, 0.1, 1, 10, 100]`, fit a ridge model.
2. Record `train_mse` and `val_mse`.
3. Plot `log(lambda)` on the x-axis and both errors on the y-axis.

You’ll see training error rise with `lambda` while validation error forms a U-shape, highlighting the optimal regularization strength.

### 5. Practice Problems

1. Ridge vs. OLS on Noisy Data
    - Generate linear data with Gaussian noise.
    - Fit OLS and ridge (with `alpha` in `[0, 0.1, 1, 10]`).
    - Plot train/test MSE vs. `alpha` and identify the best lambda.
2. Early Stopping in a Custom Loop
    - Implement a gradient-descent loop for logistic regression.
    - Track validation loss each epoch and stop when it hasn’t improved for 5 epochs.
    - Compare final test accuracy with and without early stopping.
3. k-Fold Cross-Validation from Scratch
    - Write `k_fold_split(X, y, k)` yielding train/val splits.
    - Use it to evaluate polynomial regression models of varying degree.
    - Plot average validation error vs. polynomial degree.

---

## Cost Function with Regularization

### Prerequisites

- Familiarity with un-regularized cost functions (linear or logistic).
- Understanding of why overfitting happens (high-variance models).
- Basic Python and NumPy for implementation.

### 1. Intuition: Shrinking Coefficients to Fight Overfitting

Regularization adds a penalty for large weights. Think of your model like a rubber sheet stretched over data points:

- Without regularization, it can bulge wildly to hit every point.
- With regularization, the sheet is tethered, so it stays smoother and focuses on real trends, not noise.

### 2. Regularized Cost Formulas

### 2.1 Linear Regression (L2/Ridge)

```
J(theta) = (1 / (2*m)) * sum((h(x_i) - y_i)^2 for i = 1..m)
         + (lambda / (2*m)) * sum(theta_j^2 for j = 1..n)
```

- First term: mean squared error
- Second term: penalty on all weights except theta_0
- `lambda` controls strength of penalty (larger => simpler model)

### 2.2 Logistic Regression (L2)

```
J(theta) = -(1 / m) * sum(
             y_i * log(h(x_i))
             + (1 - y_i) * log(1 - h(x_i))
           for i = 1..m)
         + (lambda / (2*m)) * sum(theta_j^2 for j = 1..n)
```

- First term: binary cross-entropy
- Second term: same weight penalty
- Exclude bias term `theta_0` from regularization

### 3. Python Implementation Examples

### 3.1 Linear Regression Cost with L2

```python
import numpy as np

def compute_cost_linear_reg(X, y, theta, lambda_):
    m = len(y)
    preds = X.dot(theta)
    error = preds - y
    cost = (1 / (2*m)) * np.sum(error**2)
    reg  = (lambda_ / (2*m)) * np.sum(theta[1:]**2)
    return cost + reg
```

### 3.2 Logistic Regression Cost with L2

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic_reg(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    term1 = -y.dot(np.log(h))
    term2 = -(1 - y).dot(np.log(1 - h))
    cost  = (1 / m) * (term1 + term2)
    reg   = (lambda_ / (2*m)) * np.sum(theta[1:]**2)
    return cost + reg
```

### 4. Real-World ML Use

- **scikit-learn Ridge/Lasso** for linear models
- **`LogisticRegression(penalty='l2', C=1/lambda)`** for classification
- Hyperparameter `lambda` (or `C` in sklearn) chosen via cross-validation
- Prevents coefficient explosion and improves generalization

### 5. Visual or Geometric Insight

- In weight-space, unregularized cost contours are ellipses; regularization adds circular contours (L2) that pull the optimum toward zero.
- Plotting coefficients vs. `log(lambda)` shows them shrinking as `lambda` increases.

### 6. Practice Problems

1. **Implement and Compare**
    - Generate synthetic linear data with noise.
    - Compute `compute_cost_linear_reg` for `lambda_` in `[0, 0.1, 1, 10]`.
    - Plot training vs. validation MSE vs. `lambda_`.
2. **Grid Search for Logistic**
    - Use `compute_cost_logistic_reg` inside a simple loop to pick `lambda_` that minimizes validation loss on a small binary dataset.
3. **Analytic vs. Numeric Gradient**
    - Derive gradient of the regularized cost.
    - Implement both analytic and finite-difference checks to verify your gradient for logistic regression.

---

## Regularized Linear Regression

### Prerequisites

- Basic linear regression (hypothesis, cost, normal equation)
- Gradient descent mechanics
- Understanding of overfitting vs. underfitting
- NumPy and scikit-learn basics

If any feel shaky, review:

```
# Linear regression cost
J(θ) = (1/(2*m)) * sum((h(x_i) - y_i)^2)  for i = 1..m

# Gradient descent update
θ := θ - α * (1/m) * Xᵀ · (X·θ - y)
```

### 1. Intuition: Taming Overfitting by Penalty

Think of fitting a rubber sheet over pegs (data points).

- Without regularization, the sheet bulges to touch every peg (fits noise).
- With a tether (penalty), it stays smoother and ignores small bumps.

Regularization adds a penalty term that **shrinks** large coefficients, keeping the model simple and robust.

### 2. Types of Regularization

| Penalty | Name | Effect |
| --- | --- | --- |
| L2 | Ridge | Shrinks all weights toward zero evenly |
| L1 | Lasso | Can drive some weights exactly to zero |
| L1 + L2 | ElasticNet | Combines shrinkage and sparsity |

### 3. Cost Formulas

### 3.1 Ridge Regression (L2)

```
J(θ) = (1/(2*m)) * sum((h(x_i) - y_i)^2)
     + (λ/(2*m)) * sum(θ_j^2)   for j = 1..n
```

- Excludes `θ_0` (bias) from the penalty if you prefer.
- `λ` controls strength: larger → simpler model.

### 3.2 Lasso Regression (L1)

```
J(θ) = (1/(2*m)) * sum((h(x_i) - y_i)^2)
     + (λ/(m)) * sum(abs(θ_j))   for j = 1..n
```

- Encourages exact zeros in `θ`, performing feature selection.

### 3.3 Elastic Net

```
J(θ) = (1/(2*m)) * sum((h(x_i) - y_i)^2)
     + (λ1/(m)) * sum(abs(θ_j))
     + (λ2/(2*m)) * sum(θ_j^2)
```

- Mixes L1 and L2 for flexible control.

### 4. Gradients and Updates

### 4.1 Ridge Gradient

```python
# Compute gradient (excluding bias θ[0] from penalty)
error    = X.dot(theta) - y                # shape: (m,1)
grad     = (1/m) * X.T.dot(error)          # shape: (n+1,1)
grad[1:] += (λ/m) * theta[1:]              # add penalty for j>=1

theta = theta - α * grad
```

### 4.2 Lasso Gradient (Subgradient)

```python
# For each j>=1:
subgrad_j = (1/m) * sum((h(x_i)-y_i)*x_i_j) + (λ/m)*sign(theta_j)
# sign(0) can be any value in [-1,1]
```

In practice, use specialized solvers (coordinate descent) for L1.

### 5. Closed-Form Solution for Ridge

When `XᵀX + λI` is invertible:

```
θ = inverse( XᵀX + λ * I ) · Xᵀ · y
```

- `I` is the identity matrix of size (n+1)×(n+1) with `I[0,0] = 0` if you skip bias regularization.

### 6. Python Examples

### 6.1 NumPy Gradient Descent for Ridge

```python
import numpy as np

def ridge_gradient_descent(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    cost_history = []
    for _ in range(num_iters):
        preds = X.dot(theta)
        error = preds - y

        grad = (1/m) * X.T.dot(error)
        grad[1:] += (lambda_/m) * theta[1:]

        theta = theta - alpha * grad
        # Optional: record cost (use compute_cost_linear_reg from earlier)
        cost_history.append(compute_cost_ridge(X, y, theta, lambda_))
    return theta, cost_history

def compute_cost_ridge(X, y, theta, lambda_):
    m = len(y)
    error = X.dot(theta) - y
    cost  = (1/(2*m)) * np.sum(error**2)
    reg   = (lambda_/(2*m)) * np.sum(theta[1:]**2)
    return cost + reg
```

### 6.2 scikit-learn Usage

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

# Example: Ridge regression pipeline
pipe_ridge = Pipeline([
    ('scale', StandardScaler()),    # important before penalty
    ('ridge', Ridge(alpha=1.0))
])
scores = cross_val_score(pipe_ridge, X, y, cv=5, scoring='neg_mean_squared_error')
print("Ridge CV RMSE:", np.sqrt(-scores.mean()))

# Lasso example
pipe_lasso = Pipeline([
    ('scale', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])
# ElasticNet example
pipe_en = Pipeline([
    ('scale', StandardScaler()),
    ('en', ElasticNet(alpha=0.1, l1_ratio=0.5))
])
```

### 7. Real-World Applications

- **High-dimensional data** (p >> n): Lasso selects features automatically.
- **Multicollinearity**: Ridge spreads weight among correlated features.
- **Model interpretability**: Sparse models from Lasso aid understanding.
- **Baseline models**: Regularized linear models are fast to train and tune.

### 8. Visual Insights

1. **Coefficient Paths**
    - Plot each `θ_j` vs. `log(λ)` to see shrinkage (Ridge: smooth, Lasso: paths hit zero).
2. **Cost Contours**
    - Unregularized: elongated ellipses.
    - Ridge: circles intersect ellipses → smaller optimum.
    - Lasso: diamonds (L1 ball) intersect ellipses → some axes exactly zero.

### 9. Practice Problems

1. **Ridge vs. OLS on Noisy Data**
    
    ```python
    # Generate linear data with noise
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100)
    # Add polynomial features degree=10
    X_poly = np.c_[X, X**2, ..., X**10]
    ```
    
    - Fit OLS (no regularization) and Ridge for λ in [0, 0.1, 1, 10].
    - Plot train/test MSE vs. λ and identify best λ.
2. **Lasso Feature Selection**
    - On a synthetic dataset with 20 features, only 5 truly informative.
    - Fit Lasso with cross-validated `alpha`.
    - Report which features are selected (non-zero weights).
3. **Elastic Net Tuning**
    - Create a pipeline that grid-searches over `alpha` and `l1_ratio`.
    - Use 5-fold CV to minimize MSE.
    - Visualize validation error as a heatmap over the grid.

---

## Regularized Logistic Regression

### Prerequisites

- Understanding of standard logistic regression (sigmoid hypothesis, log-loss cost)
- Gradient descent and Newton’s method for optimization
- The concept of overfitting versus underfitting
- Familiarity with NumPy and scikit-learn

### 1. Intuition: Keeping the Decision Boundary Stable

Imagine drawing a line to separate two groups of points.

Without regularization, the line can twist to correctly classify every point—even noise.

Adding a penalty smooths that boundary, preventing it from contorting around outliers.

### 2. Types of Regularization

| Penalty | Name | Effect |
| --- | --- | --- |
| L2 | Ridge-style | Shrinks all weights toward zero evenly |
| L1 | Lasso-style | Drives some weights exactly to zero |
| Combined | Elastic Net | Balances shrinkage with sparsity |

### 3. Cost Functions

### 3.1 L2 Regularization

```
J(θ) = −(1/m) ∑[ y_i·log(hθ(x_i)) + (1−y_i)·log(1−hθ(x_i)) ]
       + (λ/(2m)) ∑θ_j^2    for j = 1…n
```

- Exclude the bias term θ₀ from the penalty if desired.
- λ controls the strength of shrinkage.

### 3.2 L1 Regularization

```
J(θ) = −(1/m) ∑[ y_i·log(hθ(x_i)) + (1−y_i)·log(1−hθ(x_i)) ]
       + (λ/m) ∑|θ_j|      for j = 1…n
```

- Encourages exact zeros in θ, offering built-in feature selection.

### 3.3 Elastic Net

```
J(θ) = −(1/m) ∑[ y_i·log(hθ(x_i)) + (1−y_i)·log(1−hθ(x_i)) ]
       + (λ1/m) ∑|θ_j|
       + (λ2/(2m)) ∑θ_j^2
```

- Combines L1’s sparsity with L2’s smooth shrinkage.

### 4. Gradients and Updates

### 4.1 L2 Gradient Descent

```python
# Compute predictions
z    = X.dot(theta)
h    = 1 / (1 + np.exp(-z))

# Gradient (exclude bias from penalty)
grad = (1/m) * X.T.dot(h - y)
grad[1:] += (lambda_/m) * theta[1:]

theta -= alpha * grad
```

### 4.2 L1 Subgradient

```
For j ≥ 1:
   grad_j = (1/m) ∑(hθ(x_i) − y_i)·x_i_j
            + (λ/m)·sign(θ_j)

sign(0) ∈ [−1, +1], so specialized solvers (coordinate descent) are used.
```

### 5. Newton’s Method with L2 Penalty

Newton’s method updates parameters using the Hessian:

```
H = (1/m) XᵀR X  + (λ/m) I*
θ := θ − H⁻¹ · [ (1/m) Xᵀ(h − y) + (λ/m) θ* ]
```

- R is a diagonal matrix with h(i)(1−h(i)).
- I* has its [0,0] entry set to zero if bias isn’t penalized.
- Converges in fewer steps but each step is costlier.

### 6. Python Examples

### 6.1 NumPy Gradient Descent for Ridge Logistic

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ridge_logistic_gd(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    cost_history = []
    for _ in range(num_iters):
        z    = X.dot(theta)
        h    = sigmoid(z)
        grad = (1/m) * X.T.dot(h - y)
        grad[1:] += (lambda_/m) * theta[1:]
        theta -= alpha * grad
        cost_history.append(compute_cost_logistic_ridge(X, y, theta, lambda_))
    return theta, cost_history

def compute_cost_logistic_ridge(X, y, theta, lambda_):
    m = len(y)
    z    = X.dot(theta)
    h    = sigmoid(z)
    term = -y*np.log(h) - (1-y)*np.log(1-h)
    reg  = (lambda_/(2*m)) * np.sum(theta[1:]**2)
    return (1/m)*np.sum(term) + reg
```

### 6.2 scikit-learn Pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('logreg', LogisticRegression(penalty='l2', C=1.0))
])

# Grid search over C = 1/λ
param_grid = {'logreg__C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print("Best C:", grid.best_params_, "Accuracy:", grid.best_score_)
```

### 7. Real-World Applications

- Text classification where vocabulary size is huge (L1 picks key words)
- Medical diagnosis with correlated biomarkers (L2 balances weights)
- Credit scoring where interpretability and stability matter
- Any large-scale binary classification with noisy, high-dimensional features

### 8. Visual Insights

- Decision boundary complexity reduces as λ increases
- Coefficient magnitude paths versus log(λ):
    - L2 paths shrink smoothly
    - L1 paths hit zero and stay flat

### 9. Practice Problems

1. Apply unregularized and L2-regularized logistic regression on a 2D dataset.
    - Plot the decision boundary for λ = 0, 0.1, 1, 10.
2. On a dataset with 100 features but only 10 informative:
    - Fit L1-regularized logistic regression and report which features remain non-zero.
3. Build an Elastic Net logistic pipeline:
    - Grid search over `l1_ratio` and `C` to maximize AUC.
    - Visualize results as a heatmap.

---