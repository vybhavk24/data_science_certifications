# DL_c1_m2

## Binary Classification with Neural Networks

### 1. Concept Intuition

Binary classification is the task of assigning one of two labels (0 or 1) to each example. In deep learning, you build a network that outputs a probability between 0 and 1, then pick a threshold (usually 0.5) to decide the class.

Why it matters:

- Many real-world problems are binary (spam vs. not spam, disease vs. healthy, churn vs. stay).
- It’s the foundation for multi-class, multi-label, and more complex architectures.

Think of it like a smart gatekeeper: features come in, the network computes a “score,” and the sigmoid function squashes it into a probability. If that probability passes your cutoff, the gate swings one way; otherwise it swings the other.

### 2. Mathematical Breakdown

### Forward Propagation

```python
# z = linear combination
z = np.dot(w.T, x) + b

# a = activation (sigmoid)
a = 1 / (1 + np.exp(-z))
```

- `x` is an (n_features, m_examples) matrix
- `w` is (n_features, 1), `b` is scalar
- `a` is (1, m_examples), giving P(y=1|x)

### Loss Function: Binary Cross-Entropy

For a single example:

```python
L(a, y) = -[ y * np.log(a) + (1 - y) * np.log(1 - a) ]
```

Vectorized cost over m examples:

```python
cost = -(1/m) * np.sum( y * np.log(a) + (1 - y) * np.log(1 - a) )
```

- `y` is (1, m), containing 0 or 1
- The minus sign ensures minimization pulls predicted probs toward true labels

### Backward Propagation (Gradients)

```python
dz = a - y                   # (1, m)
dw = (1/m) * np.dot(x, dz.T) # (n_features, 1)
db = (1/m) * np.sum(dz)      # scalar
```

- `dz` measures how far sigmoid output is from the true label
- `dw`, `db` tell us how to shift weights/bias to reduce loss

### 3. Code & Practical Application

### A. From-Scratch Logistic Regression (NumPy)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_params(n_features):
    w = np.zeros((n_features, 1))
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    # forward
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = - (1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    # backward
    dZ = A - Y
    dw = (1/m) * np.dot(X, dZ.T)
    db = (1/m) * np.sum(dZ)
    return {"dw": dw, "db": db}, cost

def optimize(w, b, X, Y, num_iters, lr):
    for i in range(num_iters):
        grads, cost = propagate(w, b, X, Y)
        w -= lr * grads["dw"]
        b -= lr * grads["db"]
        if i % 100 == 0:
            print(f"Iteration {i}, cost: {cost:.4f}")
    return w, b

# Toy dataset
from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=200, n_features=2,
                           n_informative=2, n_redundant=0,
                           random_state=1)
X, Y = X.T, Y.reshape(1, -1)

w, b = initialize_params(X.shape[0])
w, b = optimize(w, b, X, Y, num_iters=1000, lr=0.1)
```

### B. TensorFlow/Keras Example

```python
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(2,))
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X.T, Y.ravel(), epochs=50, batch_size=16)
```

This runs the same logistic regression under the hood, but with built-in optimizers and GPU support.

### 4. Visualization / Geometry

- **Decision boundary:** In 2D, it’s a line defined by `w1*x1 + w2*x2 + b = 0`. Points on one side get P > 0.5, the other side P < 0.5.
- **Loss surface:** For two weights, imagine a 3D bowl-shaped surface where the height is the cost. Gradient descent “rolls” down that bowl.

Simple ASCII sketch of a line separator:

```
   y
   ^
   |   +      +        +
   |           \       /
   |  +         \     /   Class = 1
   |             \   /
   |--------------\-/-------------
   |              / \
   |    Class=0  /   \    +
   +--------------------------------> x
```

As you tweak `w` and `b`, that line rotates/translates to best separate the +s and 0s.

### 5. Common Pitfalls & Tips

- Not scaling inputs: large feature magnitudes make the sigmoid saturate (gradients ≈ 0).
- Learning rate too high/low: causes divergence or painfully slow convergence.
- Numeric stability: use `np.clip(a, 1e-10, 1-1e-10)` before `log` to avoid `inf`.
- Threshold choice: 0.5 works when classes are balanced; consider precision/recall trade-offs or ROC curves otherwise.
- Sigmoid saturation: for deeper nets, prefer ReLU in hidden layers; keep sigmoid only at the output.

### 6. Practice Exercises

1. **Implement from scratch**
    - Code everything above without peeking. Generate a 2D toy dataset, train your model, and plot the decision boundary.
2. **Threshold tuning**
    - On your trained model, compute precision, recall, and F1 for thresholds `[0.1,0.2,…,0.9]`. Plot F1 vs. threshold.
3. **Hidden layer extension**
    - Build a small neural net with one hidden layer of 4 units (ReLU) and sigmoid output. Train on the same data, compare accuracy and boundary shape.
4. **Numeric stability hack**
    - Modify your loss computation to clip activation values before the log. Show that it prevents NaNs if you very low learning rate or extreme initial weights.
5. **Class imbalance scenario**
    - Create a dataset with 90% of class 0, 10% class 1. Train logistic regression and inspect accuracy vs. recall. Then adjust the threshold to maximize recall on class 1.

---

## Logistic Regression

### 1. Intuition and Overview

Logistic regression is a linear model for binary outcomes. It learns a weighted combination of input features and maps that sum through a sigmoid function to produce a probability between 0 and 1.

Use cases range from spam detection and medical diagnosis to customer churn prediction.

### 2. Model Definition

### 2.1 Linear Combination

The model computes a score (z) for each example by combining features linearly:

```python
z = w.T @ x + b
```

- `x` is an n-dimensional feature vector
- `w` is an n-dimensional weight vector
- `b` is a scalar bias

### 2.2 Sigmoid Activation

Convert the raw score into a probability with the sigmoid function:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Output `a = sigmoid(z)` represents (P(y=1 \mid x)).

### 3. Odds, Log-Odds, and Probability

- **Odds**: ratio of event happening vs. not happening
    
    ```python
    odds = a / (1 - a)
    ```
    
- **Log-odds (logit)**: the inverse sigmoid
    
    ```python
    logit = np.log(odds)
    ```
    
- Logistic regression models the log-odds as a linear function of features.

### 4. Loss Function

### 4.1 Likelihood

Maximize the likelihood of observed labels under the model:

```python
L = ∏(i=1 to m) P(yᵢ | xᵢ; w, b)
  = ∏ (aᵢ^yᵢ * (1 - aᵢ)^(1 - yᵢ))
```

### 4.2 Negative Log-Likelihood (Cross-Entropy)

Minimize the average negative log-likelihood:

```python
cost = - (1/m) * sum(y * log(a) + (1 - y) * log(1 - a))
```

- `y` is 0 or 1
- `a` is predicted probability

### 5. Gradient Computation

Compute gradients for weight and bias to perform gradient descent.

```python
dz = a - y                 # shape (1, m)
dw = (1/m) * X @ dz.T      # shape (n, 1)
db = (1/m) * np.sum(dz)    # scalar
```

- `dz` measures prediction error
- `dw`, `db` guide parameter updates

### 6. Optimization Algorithms

1. **Batch Gradient Descent**Update using the full dataset each step.
2. **Stochastic Gradient Descent (SGD)**Update on one example at a time.
3. **Mini-Batch Gradient Descent**Update on small batches for a middle ground.
4. **Newton’s Method / IRLS**Uses second-order derivatives to converge faster.
5. **Quasi-Newton (BFGS, L-BFGS)**Approximate Hessian for large-scale problems.

### 7. Regularization

Add penalty terms to prevent overfitting.

- **L2 (Ridge):**
    
    ```python
    cost_reg = cost + (λ/(2*m)) * np.sum(w**2)
    ```
    
- **L1 (Lasso):**
    
    ```python
    cost_reg = cost + (λ/m) * np.sum(np.abs(w))
    ```
    
- **Elastic Net:**
    
    ```python
    cost_reg = cost + (λ1/m)*sum(abs(w)) + (λ2/(2*m))*sum(w**2)
    ```
    

Control `λ` to trade off bias and variance. In scikit-learn, `C = 1/λ`.

### 8. Multiclass Extensions

1. **One-vs-Rest (OvR):** train one binary classifier per class.
2. **Softmax (Multinomial Logistic):**Minimize multinomial cross-entropy.
    
    ```python
    z_k = w_k.T @ x + b_k  for each class k
    a_k = exp(z_k) / sum_j exp(z_j)
    ```
    

### 9. Implementation from Scratch (NumPy)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize(n):
    w = np.zeros((n, 1))
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    Z = w.T @ X + b
    A = sigmoid(Z)
    cost = - (1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    dz = A - Y
    dw = (1/m) * X @ dz.T
    db = (1/m) * np.sum(dz)
    return {'dw': dw, 'db': db}, cost

def optimize(w, b, X, Y, lr, iterations):
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        w -= lr * grads['dw']
        b -= lr * grads['db']
        if i % 100 == 0:
            print(f"Iter {i}: cost {cost:.4f}")
    return w, b

def predict(w, b, X, threshold=0.5):
    A = sigmoid(w.T @ X + b)
    return (A > threshold).astype(int)
```

### 10. Scikit-Learn Usage

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X.T, Y.ravel(), test_size=0.2, random_state=0
)

clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    class_weight='balanced'
)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
```

Adjust `penalty`, `C`, `solver`, and `class_weight` for your data.

### 11. TensorFlow/Keras Example

```python
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(n_features,))
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X.T, Y.ravel(), epochs=50, batch_size=32)
```

For multiclass, use `Dense(n_classes, activation='softmax')` and `loss='sparse_categorical_crossentropy'`.

### 12. Numerical Stability

- Clip predictions before log to avoid `inf`:
    
    ```python
    A = np.clip(A, 1e-15, 1-1e-15)
    ```
    
- Use the log-sum-exp trick in softmax implementations.

### 13. Feature Engineering & Preprocessing

- **Scaling**: standardize or normalize features
- **Polynomial / Interaction Terms**: capture non-linearities
- **One-Hot Encoding**: for categorical variables
- **Dimensionality Reduction**: PCA or feature selection

### 14. Handling Class Imbalance

- **Resampling**: oversample minority or undersample majority
- **Class Weights**: adjust penalty for each class
- **Threshold Tuning**: pick operating point based on precision/recall trade-off

### 15. Evaluation Metrics

- **Accuracy**: overall correctness
- **Precision & Recall**: focus on type of error
- **F1 Score**: harmonic mean of precision and recall
- **ROC & AUC**: threshold-independent ranking quality
- **Calibration Curve & Brier Score**: probability reliability

### 16. Interpretability

- **Coefficients** `w`: each feature’s log-odds impact
- **Odds Ratio**: `exp(w_i)` shows multiplicative change in odds
- **Feature Importance**: use standardized coefficients or L1 sparsity

### 17. Common Pitfalls & Best Practices

- Forgetting to scale inputs leads to poor convergence.
- Ignoring regularization can cause overfitting or weight explosion.
- Using default threshold (0.5) without checking class balance.
- Neglecting probability calibration, especially in risk-sensitive domains.
- Overinterpreting coefficients without considering feature correlation.

### 18. Practical Exercises

1. Implement logistic regression from scratch and compare batch vs. mini-batch SGD.
2. Extend your implementation to support L1, L2, and elastic-net regularization.
3. Generate an imbalanced dataset; explore resampling vs. class weights vs. threshold tuning.
4. Calibrate predicted probabilities with Platt scaling and isotonic regression.
5. Build a softmax classifier from scratch and derive its gradient formulas.
6. Experiment with Newton’s method (IRLS) and compare convergence to gradient descent.

---

## Logistic Regression Cost Function

### 1. Intuition

Logistic regression learns parameters that make predicted probabilities match true labels.

A cost function measures how “off” those probabilities are. Minimizing this cost pushes the model to output high probabilities for true positives and low probabilities for true negatives.

### 2. From Likelihood to Cost

Logistic regression models each label (y_i\in{0,1}) with probability

```
P(y_i | x_i; w,b) = a_i^y_i * (1 - a_i)^(1 - y_i)
```

where

```python
z_i = w.T @ x_i + b
a_i = 1 / (1 + exp(-z_i))    # predicted probability
```

The likelihood of the entire dataset is the product of individual probabilities:

```
L(w,b) = ∏_{i=1 to m} [ a_i^y_i * (1 - a_i)^(1 - y_i) ]
```

Maximizing the likelihood is equivalent to minimizing the negative log-likelihood (the cost).

### 3. Binary Cross-Entropy Cost

By taking negative log and averaging over (m) examples, we get the binary cross-entropy cost:

```python
J(w,b) = - (1/m) * sum(
    y_i * log(a_i)
  + (1 - y_i) * log(1 - a_i)
  for i in range(m)
)
```

- when (y_i=1), cost term is (-log(a_i))
- when (y_i=0), cost term is (-log(1-a_i))

### 4. Vectorized Form

Let `X` be shape `(n_features, m)` and `Y` be `(1, m)`. Compute:

```python
Z = w.T @ X + b                      # shape (1, m)
A = 1 / (1 + exp(-Z))                # shape (1, m)
cost = - (1/m) * sum( Y * log(A)
                   + (1 - Y) * log(1 - A) )
```

Vectorization yields concise code and leverages optimized linear algebra.

### 5. Gradients of the Cost

To update parameters via gradient descent, compute partial derivatives:

```python
dZ = A - Y                           # shape (1, m)
dw = (1/m) * X @ dZ.T                # shape (n_features, 1)
db = (1/m) * sum(dZ)                 # scalar
```

These gradients tell you how to adjust `w` and `b` to reduce the cost.

### 6. Regularized Cost

Prevent overfitting by adding penalty terms:

- L2 (ridge) regularization
    
    ```python
    J_reg = J + (λ/(2*m)) * sum(w_j**2 for j in range(n_features))
    ```
    
- L1 (lasso) regularization
    
    ```python
    J_reg = J + (λ/m) * sum(abs(w_j) for j in range(n_features))
    ```
    
- Elastic net
    
    ```python
    J_reg = J
          + (λ1/m) * sum(abs(w_j) for j in range(n_features))
          + (λ2/(2*m)) * sum(w_j**2 for j in range(n_features))
    ```
    

In scikit-learn, `C = 1/λ` controls regularization strength.

### 7. Numerical Stability

Log operations on values extremely close to 0 or 1 cause `-inf`. Clip predicted probabilities:

```python
A = clip(A, 1e-15, 1 - 1e-15)
```

For softmax (multiclass), use the log-sum-exp trick to stabilize exponentials.

### 8. Convexity and Second-Order Methods

The binary cross-entropy cost is convex in `w` and `b`. You can use:

- **Newton’s Method / IRLS**
    
    ```python
    D = diag((A * (1 - A)).flatten())      # shape (m, m)
    H = (1/m) * X.T @ D @ X                # Hessian matrix
    grad = (1/m) * X @ (A - Y).T           # gradient vector
    theta_new = theta_old - inv(H) @ grad  # update both w and b
    ```
    
- **Quasi-Newton (BFGS/L-BFGS)** for large-scale problems without forming full Hessian.

### 9. Multiclass Cross-Entropy Cost

For (K) classes, use softmax activation and extend cost to:

```python
for each example i:
  z_i = W @ x_i + b                    # shape (K,)
  a_i = exp(z_i) / sum(exp(z_i))       # shape (K,)
J = - (1/m) * sum(
      sum(y_i_k * log(a_i_k) for k in range(K))
    for i in range(m)
)
```

This generalizes binary cross-entropy to multiple classes.

### 10. Implementation Snippet

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_and_grads(w, b, X, Y, λ=0.0):
    m = X.shape[1]
    Z = w.T @ X + b
    A = sigmoid(Z)
    A = np.clip(A, 1e-15, 1 - 1e-15)
    cost = - (1/m) * np.sum(Y * np.log(A)
                         + (1 - Y) * np.log(1 - A))
    if λ > 0:
        cost += (λ/(2*m)) * np.sum(w**2)
    dZ = A - Y
    dw = (1/m) * X @ dZ.T + (λ/m) * w
    db = (1/m) * np.sum(dZ)
    return cost, dw, db
```

### 11. Common Pitfalls

- forgetting to clip `A` before taking `log`
- omitting regularization when data is high-dimensional
- mixing up `m` (examples) and `n` (features) in formulas
- using the wrong shape for bias when vectorizing
- misinterpreting convexity—gradient descent still needs a sensible learning rate

---

## Gradient Descent

### 1. Intuition

Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a cost function.

Imagine standing on a smooth hill and trying to reach the lowest point. At each step, you look at the steepness (the gradient) and move downhill in that direction. Over many small steps, you end up at or near the lowest valley.

### 2. Basic Algorithm

### 2.1 Cost Function and Gradient

A cost function `J(θ)` measures how far your predictions are from the true labels. The gradient `∇J(θ)` is a vector of partial derivatives showing how `J` changes with each parameter.

```python
# For logistic regression or linear regression:
J(θ) = (1/m) * sum((hθ(xᵢ) - yᵢ)**2 for i in range(m))
# gradient
dJ/dθ_j = (1/m) * sum((hθ(xᵢ) - yᵢ) * xᵢ_j for i in range(m))
```

### 2.2 Parameter Update Rule

At each iteration, update parameters `θ` by stepping opposite the gradient:

```python
θ = θ - α * ∇J(θ)
```

- `α` is the learning rate
- A small `α` takes tiny steps (slow but stable)
- A large `α` may overshoot or diverge

### 3. Gradient Descent Variants

### 3.1 Batch Gradient Descent

Uses the entire training set to compute the gradient at each step.

```python
for each iteration:
    compute ∇J(θ) using all m examples
    θ = θ - α * ∇J(θ)
```

- Pros: stable gradient direction
- Cons: slow on large datasets

### 3.2 Stochastic Gradient Descent (SGD)

Updates parameters using one training example at a time.

```python
for each epoch:
    for each example (xᵢ, yᵢ):
        compute gradient using xᵢ, yᵢ
        θ = θ - α * ∇Jᵢ(θ)
```

- Pros: faster per update, can escape shallow local minima
- Cons: noisy convergence

### 3.3 Mini-Batch Gradient Descent

Splits data into small batches to balance variance and efficiency.

```python
for each epoch:
    shuffle training data
    for each batch of size B:
        compute gradient on batch
        θ = θ - α * ∇J_batch(θ)
```

- Pros: uses vectorized operations, smoother convergence than SGD

### 4. Advanced Optimizers

### 4.1 Momentum

Accelerates convergence by adding a fraction of the previous update to the current one.

```python
v = β * v + α * ∇J(θ)
θ = θ - v
```

- `β` (momentum term) typically 0.9

### 4.2 Nesterov Accelerated Gradient

Looks ahead by applying momentum before gradient calculation.

```python
v_prev = v
v = β * v - α * ∇J(θ + β * v_prev)
θ = θ + v
```

### 4.3 AdaGrad

Adapts learning rate per parameter based on past gradients.

```python
G = G + ∇J(θ)**2
θ = θ - (α / sqrt(G + ε)) * ∇J(θ)
```

- `ε` prevents division by zero

### 4.4 RMSProp

Modifies AdaGrad to decay past gradient contributions.

```python
E[g²] = ρ * E[g²] + (1 - ρ) * ∇J(θ)**2
θ = θ - (α / sqrt(E[g²] + ε)) * ∇J(θ)
```

- `ρ` typically 0.9

### 4.5 Adam

Combines momentum and RMSProp for robust performance.

```python
m = β1 * m + (1 - β1) * ∇J(θ)
v = β2 * v + (1 - β2) * ∇J(θ)**2
m_hat = m / (1 - β1**t)
v_hat = v / (1 - β2**t)
θ = θ - (α / (sqrt(v_hat) + ε)) * m_hat
```

- Defaults: `β1=0.9`, `β2=0.999`, `ε=1e-8`

### 5. NumPy Implementation from Scratch

```python
import numpy as np

def gradient_descent(X, Y, θ, α, iterations, batch_size=None):
    m = X.shape[1]
    for it in range(iterations):
        if batch_size:
            # mini-batch variant
            idx = np.random.choice(m, batch_size, replace=False)
            X_batch, Y_batch = X[:, idx], Y[:, idx]
        else:
            # batch gradient descent
            X_batch, Y_batch = X, Y

        Z = θ.T @ X_batch
        A = 1 / (1 + np.exp(-Z))    # sigmoid example
        dZ = A - Y_batch
        dθ = (1/X_batch.shape[1]) * X_batch @ dZ.T

        θ = θ - α * dθ

        if it % 100 == 0:
            cost = - (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
            print(f"Iter {it}: cost {cost:.4f}")

    return θ
```

### 6. Learning Rate and Schedules

### 6.1 Fixed Learning Rate

Keep `α` constant. Simple but may require manual tuning.

### 6.2 Time-Based Decay

Reduce `α` over time:

```python
α_t = α0 / (1 + decay_rate * t)
```

### 6.3 Exponential Decay

```python
α_t = α0 * exp(-decay_rate * t)
```

### 6.4 Warm Restarts

Periodically reset `α` to escape local minima:

```python
if t % T_0 == 0: α = α0
else: α = α * 0.5
```

### 7. Convergence and Troubleshooting

- **Learning rate too high**: cost diverges or oscillates
- **Learning rate too low**: slow convergence, stuck at poor solution
- **Vanishing/exploding gradients**: in deep nets prefer ReLU and batch normalization
- **Saddle points**: momentum or adaptive optimizers help escape
- **Non-convex landscapes**: only guaranteed global optimum for convex problems

### 8. Interview-Ready Talking Points

1. Difference between batch, SGD, and mini-batch in terms of convergence speed and stability.
2. Why momentum accelerates learning and reduces oscillation along steep ravines.
3. How Adam adaptively combines momentum and per-parameter learning rates.
4. Trade-offs of fixed vs. decaying learning rates and need for tuning.

### 9. Practical Exercises

1. Implement batch, SGD, and mini-batch gradient descent and compare convergence on a toy dataset.
2. Add momentum and observe changes in the loss curve.
3. Experiment with learning rate schedules and plot how cost decreases over epochs.
4. Compare Adam, RMSProp, and AdaGrad on sparse vs. dense data.

---

## Derivatives

### 1. Intuition and Definition

Derivatives measure how a function changes as its input changes.

Think of driving a car: your speedometer shows the instantaneous rate of change of distance with respect to time. In math, the derivative of (f(x)) at a point (x) is the slope of the tangent line at that point.

The formal limit definition is:

```python
f_prime(x) = limit as h->0 of (f(x + h) - f(x)) / h
```

### 2. Notation

Common ways to write a derivative of `y = f(x)`:

- `f_prime(x)`
- `df_dx`
- `dy_dx`
- `D[f][x]`

For higher dimensions (multivariable) you’ll see partial derivatives:

- `∂f/∂x`
- `partial_f_partial_x`

### 3. Basic Differentiation Rules

1. Constant rule
    
    A constant has zero slope.
    
    ```python
    d_dx of C = 0
    ```
    
2. Power rule
    
    ```python
    d_dx of x**n = n * x**(n - 1)
    ```
    
3. Constant multiple
    
    ```python
    d_dx of [C * f(x)] = C * f_prime(x)
    ```
    
4. Sum/difference
    
    ```python
    d_dx of [f(x) + g(x)] = f_prime(x) + g_prime(x)
    d_dx of [f(x) - g(x)] = f_prime(x) - g_prime(x)
    ```
    
5. Product rule
    
    ```python
    d_dx of [f(x) * g(x)] = f_prime(x) * g(x) + f(x) * g_prime(x)
    ```
    
6. Quotient rule
    
    ```python
    d_dx of [f(x) / g(x)] =
      (f_prime(x) * g(x) - f(x) * g_prime(x)) / g(x)**2
    ```
    
7. Chain rule
    
    ```python
    d_dx of f(g(x)) = f_prime(g(x)) * g_prime(x)
    ```
    

### 4. Common Examples

1. Polynomial
    
    ```python
    f(x) = 3*x**4 - 5*x**2 + 7
    f_prime(x) = 12*x**3 - 10*x
    ```
    
2. Exponential
    
    ```python
    f(x) = exp(x)
    f_prime(x) = exp(x)
    ```
    
3. Logarithm
    
    ```python
    f(x) = log(x)
    f_prime(x) = 1 / x
    ```
    
4. Trigonometric
    
    ```python
    f(x) = sin(x)
    f_prime(x) = cos(x)
    
    f(x) = cos(x)
    f_prime(x) = -sin(x)
    ```
    
5. Composite
    
    ```python
    f(x) = sin(3*x**2 + 2)
    # inner = 3*x**2 + 2, inner_prime = 6*x
    f_prime(x) = cos(3*x**2 + 2) * 6*x
    ```
    

### 5. Higher-Order Derivatives

- Second derivative measures curvature.
- Notation: `f_double_prime(x)` or `d2f_dx2`.

Example for `f(x) = x**3`:

```python
f_prime(x) = 3*x**2
f_double_prime(x) = 6*x
```

Use sign of second derivative to detect concavity and inflection points.

### 6. Partial Derivatives (Multivariable)

For `f(x, y)`, hold one variable constant:

```python
partial_f_partial_x = d/dx of f(x, y)
partial_f_partial_y = d/dy of f(x, y)
```

Example:

```python
f(x, y) = x**2 * y + sin(y)
partial_f_partial_x = 2*x*y
partial_f_partial_y = x**2 + cos(y)
```

### 7. Gradient, Jacobian, Hessian

- Gradient (`grad f`) is the vector of all partials for a scalar field.
- Jacobian is the matrix of first derivatives for a vector-valued function.
- Hessian is the matrix of second partial derivatives for a scalar field.

```python
# Gradient of f(x, y)
grad_f = [partial_f_partial_x, partial_f_partial_y]

# Hessian of f(x, y)
hessian_f = [
  [d2f_dx2, d2f_dxdy],
  [d2f_dydx, d2f_dy2]
]
```

### 8. Implicit Differentiation

When `y` is defined implicitly by an equation `F(x, y) = 0`, differentiate both sides:

Example:

```python
# Given x**2 + y**2 = 1
2*x + 2*y * dy_dx = 0
dy_dx = -x / y
```

### 9. Directional Derivatives and Total Derivative

- Directional derivative along a unit vector `u`:

```python
D_u f(x) = grad_f(x) @ u
```

- Total derivative of `f` approximates change in all directions:

```python
df ≈ grad_f(x).T @ dx_vector
```

### 10. Numerical Differentiation

Finite difference approximations:

1. Forward difference
    
    ```python
    f_prime(x) ≈ (f(x + h) - f(x)) / h
    ```
    
2. Backward difference
    
    ```python
    f_prime(x) ≈ (f(x) - f(x - h)) / h
    ```
    
3. Central difference (more accurate)
    
    ```python
    f_prime(x) ≈ (f(x + h) - f(x - h)) / (2*h)
    ```
    

Choose `h` small (e.g., `1e-5`) but not too small to avoid rounding errors.

### 11. Applications in Optimization

- Critical points where derivative equals zero.
- Use first and second derivatives to classify minima, maxima, and saddle points.
- Gradient descent uses the gradient to drive parameters downhill:

```python
theta = theta - learning_rate * grad_J(theta)
```

### 12. Differentiability and Exceptions

- Not all functions are differentiable (e.g., `abs(x)` at `x=0`).
- Points of non-differentiability often correspond to corners or cusps.

### 13. Automatic Differentiation (AD)

AD combines symbolic and numeric methods to compute exact derivatives in code.

Popular in ML frameworks (TensorFlow, PyTorch).

### 14. Practice Exercises

1. Differentiate `f(x) = x**5 - 4*x + 1`.
2. Compute gradient of `f(x, y, z) = x*y + y*z + z*x`.
3. Use implicit differentiation on `x*y + sin(y) = y`.
4. Implement central difference to approximate derivative of `exp(x)`.
5. Find critical points of `f(x) = x**3 - 3*x + 2` and classify them.

---

## Computation Graph

### 1. Intuition

A computation graph is a way to break down a complex mathematical expression into a network of simple operations.

Each operation (node) takes inputs, computes an output, and passes that output along edges to subsequent nodes.

This representation makes it easy to execute the expression (forward pass) and compute gradients (backward pass) automatically.

### 2. Core Concepts

- A **node** represents either an input, a constant, or an operation (addition, multiplication, activation, etc.).
- An **edge** carries a value from one node to another.
- The graph is typically directed and acyclic (DAG).
- You evaluate the graph by traversing nodes in topological order.

### 3. Graph Components

- **Input nodes** hold raw data or parameters (weights, biases).
- **Operation nodes** perform elementary functions:
    - Addition, subtraction
    - Multiplication, division
    - Element-wise functions (sigmoid, ReLU, log)
- **Output nodes** represent the final result or loss.

### 4. Forward Pass

1. Assign values to input nodes.
2. Visit each operation node in order so that its inputs are already computed.
3. Compute the node’s output and store it.
4. Continue until you reach the output node.

Example for `z = x * w + b`:

```python
# assume x, w, b are scalars or arrays
u = x * w         # multiplication node
z = u + b         # addition node
```

### 5. Backward Pass (Automatic Differentiation)

Use the chain rule to propagate gradients from output back to inputs:

1. Initialize gradient at output node: `dz = 1`.
2. For each node in reverse topological order, compute local gradients.
3. Multiply incoming gradient by local derivative to get gradient wrt each input.
4. Accumulate gradients for nodes with multiple children.

Example for `z = x * w + b`:

```python
# forward
u = x * w
z = u + b

# backward
dz = 1
du = dz * 1           # derivative of (u + b) wrt u
db = dz * 1           # derivative of (u + b) wrt b

dx = du * w           # derivative of (x * w) wrt x
dw = du * x           # derivative of (x * w) wrt w
```

### 6. Static vs Dynamic Graphs

- **Static graph** (TensorFlow 1.x, Theano): build the entire graph before any computation.
    - Pros: global optimizations, deployment optimizations
    - Cons: less intuitive debugging, rigid control flow
- **Dynamic graph** (PyTorch, TensorFlow Eager): graph is built on-the-fly during execution.
    - Pros: native Python control flow, easy debugging
    - Cons: less opportunity for whole-graph optimizations

### 7. Example: Simple Neural Unit

Compute `y = sigmoid(x1 * w1 + x2 * w2 + b)` and its gradients.

```python
# forward pass
u1 = x1 * w1
u2 = x2 * w2
s = u1 + u2 + b
y = 1 / (1 + exp(-s))

# backward pass
dy = dL_dy         # incoming gradient from loss
ds = dy * y * (1 - y)

du1 = ds * 1
du2 = ds * 1
db = ds * 1

dx1 = du1 * w1
dw1 = du1 * x1

dx2 = du2 * w2
dw2 = du2 * x2
```

### 8. Implementing a Computation Graph from Scratch

```python
class Node:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.value = None
        self.grad = 0

    def forward(self):
        pass

    def backward(self):
        pass

class Multiply(Node):
    def __init__(self, x, w):
        super().__init__()
        self.inputs = [x, w]
        x.outputs.append(self)
        w.outputs.append(self)

    def forward(self):
        x, w = self.inputs
        self.value = x.value * w.value

    def backward(self):
        x, w = self.inputs
        x.grad += self.grad * w.value
        w.grad += self.grad * x.value

# Build graph
x = Node(); w = Node(); b = Node()
x.value, w.value, b.value = 2.0, 3.0, 1.0

u = Multiply(x, w)
z = Add(u, b)
y = Sigmoid(z)

# forward pass
for node in [x, w, b, u, z, y]:
    node.forward()

# backward pass
y.grad = 1.0
for node in [y, z, u, x, w, b]:
    node.backward()
```

### 9. Framework Examples

- **TensorFlow 2.x**
    
    ```python
    import tensorflow as tf
    
    x = tf.Variable(2.0)
    w = tf.Variable(3.0)
    b = tf.Variable(1.0)
    
    with tf.GradientTape() as tape:
        y = tf.sigmoid(x * w + b)
    
    grads = tape.gradient(y, [x, w, b])
    ```
    
- **PyTorch**
    
    ```python
    import torch
    
    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    
    y = torch.sigmoid(x * w + b)
    y.backward()
    
    dx, dw, db = x.grad, w.grad, b.grad
    ```
    

### 10. Practice Exercises

1. Build a computation graph for `f(x, y) = x*y + sin(x)`, compute forward and backward manually.
2. Extend your Node class to support division and log.
3. Implement second-order derivatives for a small graph.
4. Compare static vs dynamic graph performance on a toy deep network.
5. Profile memory usage of forward and backward passes with and without checkpointing.

---

## Derivatives via Computation Graphs

### 1. Why Computation Graphs for Derivatives?

Computation graphs turn any mathematical expression into nodes and edges. That structure lets us

- Run a **forward pass** to compute outputs
- Run a **backward pass** to compute all partial derivatives via the chain rule

This is the backbone of automatic differentiation in ML libraries.

### 2. Graph Anatomy

- **Input nodes** hold variables or constants
- **Operation nodes** perform elementary steps (add, multiply, sin, exp, …)
- **Edges** carry values forward and gradients backward
- The graph is a DAG (directed acyclic graph)

Example expression:

```
f(x, y) = x * y + sin(x)
```

Breaks into nodes:

1. `u = x * y`
2. `v = sin(x)`
3. `f = u + v`

### 3. Forward Pass

1. Assign values to `x` and `y`.
2. Compute `u = x * y`.
3. Compute `v = sin(x)`.
4. Compute `f = u + v`.

At each step store both the **value** and references to input nodes.

### 4. Backward Pass (Reverse-Mode)

1. Set `df/df = 1` at output node.
2. Visit nodes in reverse topological order.
3. At each node, multiply its upstream gradient by the local derivative and add to its inputs’ gradients.

Chain rule at work:

```python
# for f = u + v
df_du = 1      # ∂f/∂u
df_dv = 1      # ∂f/∂v

# for u = x * y
du_dx = y      # ∂u/∂x
du_dy = x      # ∂u/∂y

# for v = sin(x)
dv_dx = cos(x) # ∂v/∂x
```

### 5. Node-Level Derivative Rules

```python
# Add node: z = a + b
dz_da = 1
dz_db = 1

# Multiply node: z = a * b
dz_da = b
dz_db = a

# Sin node: z = sin(a)
dz_da = cos(a)

# Exp node: z = exp(a)
dz_da = exp(a)

# Log node: z = log(a)
dz_da = 1 / a
```

During backward pass, each node uses its local rule to push gradients to its inputs.

### 6. Example 1: Simple Linear

Function:

```python
f(x) = w * x + b
```

### Forward

```python
u = w * x
f = u + b
```

### Backward

```python
df_du = 1
df_db = 1

du_dw = x
du_dx = w

# accumulate gradients
df_dw = df_du * du_dw  # = 1 * x
df_dx = df_du * du_dx  # = 1 * w
df_db = df_db          # = 1
```

### 7. Example 2: Composite

Function:

```python
f(x) = sin(w * x + b)
```

### Forward

```python
u = w * x
v = u + b
f = sin(v)
```

### Backward

```python
df_dv    = cos(v) * 1
dv_du    = 1
du_dw    = x
du_dx    = w

# chain them
df_dw = df_dv * dv_du * du_dw   # = cos(v) * x
df_dx = df_dv * dv_du * du_dx   # = cos(v) * w
df_db = df_dv * dv_du           # = cos(v) * 1
```

### 8. Example 3: Two-Variable with Sum

Function:

```python
f(x, y) = x * y + sin(x)
```

### Forward

```python
u = x * y
v = sin(x)
f = u + v
```

### Backward

```python
# output
df_du = 1
df_dv = 1

# u = x * y
du_dx = y
du_dy = x

# v = sin(x)
dv_dx = cos(x)

# chain
df_dx = df_du * du_dx + df_dv * dv_dx   # = y + cos(x)
df_dy = df_du * du_dy                   # = x
```

### 9. Build Your Own Graph Engine (Python)

```python
class Node:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.value = None
        self.grad  = 0.0

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Add(Node):
    def __init__(self, a, b):
        super().__init__()
        self.inputs = [a, b]
        a.outputs.append(self); b.outputs.append(self)

    def forward(self):
        a, b = self.inputs
        self.value = a.value + b.value

    def backward(self):
        for inp in self.inputs:
            inp.grad += 1 * self.grad

class Mul(Node):
    def __init__(self, a, b):
        super().__init__()
        self.inputs = [a, b]
        a.outputs.append(self); b.outputs.append(self)

    def forward(self):
        a, b = self.inputs
        self.value = a.value * b.value

    def backward(self):
        a, b = self.inputs
        a.grad += b.value * self.grad
        b.grad += a.value * self.grad

class Sin(Node):
    def __init__(self, a):
        super().__init__()
        self.inputs = [a]
        a.outputs.append(self)

    def forward(self):
        import math
        self.value = math.sin(self.inputs[0].value)

    def backward(self):
        import math
        a = self.inputs[0]
        a.grad += math.cos(a.value) * self.grad

# usage:
x, y, w, b = Node(), Node(), Node(), Node()
x.value, y.value, w.value, b.value = 2.0, 3.0, 4.0, 1.0

u = Mul(x, y)
v = Sin(x)
f = Add(u, v)

# forward pass
for node in [x, y, w, b, u, v, f]:
    node.forward()

# backward pass
f.grad = 1.0
for node in [f, v, u, x, y]:
    node.backward()

print("df/dx:", x.grad)
print("df/dy:", y.grad)
```

### 10. Automatic Differentiation Workflow

1. **Build graph** of operations as you compute forward.
2. **Topologically sort** nodes.
3. **Forward**: call `node.forward()` in sorted order.
4. **Backward**: set `output.grad = 1` then call `node.backward()` in reverse order.
5. **Read** `.grad` on input nodes.

This yields exact derivatives up to machine precision.

### 11. Higher-Order Derivatives

To get second derivatives (Hessian):

1. Compute first `.grad` via backward pass.
2. Treat each gradient as a new graph output and run backward again to get partials of partials.
3. Or implement a dual-number system (forward-over-reverse or reverse-over-forward).

### 12. Framework Examples

**PyTorch**

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
f = x * y + torch.sin(x)

f.backward()
print(x.grad)   # df/dx = y + cos(x)
print(y.grad)   # df/dy = x
```

**TensorFlow 2.x**

```python
import tensorflow as tf

x = tf.Variable(2.0)
y = tf.Variable(3.0)
with tf.GradientTape() as tape:
    f = x * y + tf.sin(x)

grads = tape.gradient(f, [x, y])
print(grads)    # [y + cos(x), x]
```

### 13. Pitfalls & Tips

- **Zero gradients** if you forget to reset `.grad` before each backward.
- **Multiple paths**: always accumulate (`+=`) gradients when a node feeds into many children.
- **Order matters**: ensure reverse topological order in backward pass.
- **Numerical stability**: watch for operations like `log` on zero or division by zero—handle in forward or backward carefully.

### 14. Practice Exercises

1. Manually build a graph for `f(a,b,c)= a*b + c*exp(a)` and compute `(df/da, df/db, df/dc)`.
2. Extend the engine to support `Exp` and `Log` nodes and test gradients.
3. Compute Hessian of `f(x, y) = x*y + x**2` using two backward passes.
4. Implement checkpointing: drop some intermediate values in forward, recompute in backward.
5. Compare reverse-mode (compute gradients of one output wrt many inputs) vs forward-mode (compute many directional derivatives).

---

## Logistic Regression with Gradient Descent

### 1. Concept Intuition

Logistic regression models the probability of a binary outcome by applying a sigmoid to a linear combination of inputs. Gradient descent is the iterative procedure that tweaks those linear weights and bias to minimize how wrong our probability estimates are.

- Imagine starting with random weights on a 2-D dataset: the decision boundary is a random line.
- Each pass (epoch), you measure “how wrong” the model is (cost), compute “which way downhill” in weight-space the cost decreases (gradient), and move a tiny step in that direction.
- Over many steps, the line rotates and shifts to separate classes as best as possible.

This combination underlies most training loops in deep learning.

### 2. Mathematical Breakdown

### 2.1 Model and Prediction

```python
# Linear score
Z = w.T @ X + b            # shape (1, m)

# Sigmoid activation
A = 1 / (1 + np.exp(-Z))   # shape (1, m) gives P(y=1|x)
```

- `X` has shape `(n_features, m_examples)`.
- `w` is `(n_features, 1)`, `b` is scalar.

### 2.2 Cost Function (Binary Cross-Entropy)

```python
cost = - (1/m) * np.sum(
    Y * np.log(A)
  + (1 - Y) * np.log(1 - A)
)
```

- Penalizes confident but wrong predictions heavily.
- Convex in `(w, b)`—one global minimum.

### 2.3 Gradients

```python
dZ = A - Y                    # shape (1, m)
dw = (1/m) * X @ dZ.T         # shape (n_features, 1)
db = (1/m) * np.sum(dZ)       # scalar
```

- `dZ` is element-wise error.
- `dw`, `db` tell how cost changes with each weight and bias.

### 2.4 Parameter Update Rule

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

- `learning_rate` (α) controls step size: too large → overshoot; too small → slow.

### 3. Code & Practical Application

### 3.1 From-Scratch Training Loop

```python
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 1. Generate toy data
X, Y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2,
    random_state=42
)
X, Y = X.T, Y.reshape(1, -1)
m = X.shape[1]

# 2. Initialize parameters
w = np.zeros((X.shape[0], 1))
b = 0.0
learning_rate = 0.1
num_iters = 1000
costs = []

# 3. Training with batch gradient descent
for i in range(num_iters):
    Z = w.T @ X + b
    A = 1 / (1 + np.exp(-Z))
    cost = - (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dZ = A - Y
    dw = (1/m) * X @ dZ.T
    db = (1/m) * np.sum(dZ)

    # parameter update
    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 100 == 0:
        costs.append(cost)
        print(f"Iter {i}, cost: {cost:.4f}")

# 4. Plot cost over iterations
plt.plot(np.arange(len(costs)) * 100, costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs. Iteration")
plt.show()
```

### 3.2 Decision Boundary Visualization

```python
# grid for contour
xx, yy = np.meshgrid(
    np.linspace(X[0].min()-1, X[0].max()+1, 200),
    np.linspace(X[1].min()-1, X[1].max()+1, 200)
)
grid = np.c_[xx.ravel(), yy.ravel()].T
Z_grid = w.T @ grid + b
A_grid = 1 / (1 + np.exp(-Z_grid))
A_grid = A_grid.reshape(xx.shape)

plt.contourf(xx, yy, A_grid > 0.5, alpha=0.3)
plt.scatter(X[0], X[1], c=Y.ravel(), edgecolors='k')
plt.title("Decision Boundary")
plt.show()
```

### 4. Visualization / Geometry

- **Parameter space**: For two weights `(w1, w2)` with fixed `b`, you can contour the cost function and plot the path that gradient descent takes from the origin to the minimum.
- **Loss surface**: A convex “bowl” in `(w, b)` dimensions.
- **Decision boundary**: The line `w1·x1 + w2·x2 + b = 0` rotates/moves as `(w, b)` update.

Example contour sketch:

```
Cost
  ^
  |        .--o--.
  |      .'       '.
  |     /           \
  |    o    minima   \
  |   /               \
  | .'                 '
  +------------------------> w1
```

Gradient descent steps “roll” downhill.

### 5. Common Pitfalls & Tips

- **Unscaled inputs** slow or stall convergence. Always standardize or normalize features.
- **Learning rate**:
    - Too large: cost oscillates or explodes.
    - Too small: training is painfully slow.
    - Use a scheduler or try [10⁻³, 10⁻¹] range first.
- **Initialization**: zeros work for logistic regression, but for deeper nets use small random values.
- **Numerical stability**:before `log` to avoid `NaN`.
    
    ```python
    A = np.clip(A, 1e-15, 1-1e-15)
    ```
    
- **Stopping criterion**: instead of fixed iterations, stop when cost change < ε.

### 6. Practice Exercises

1. **Learning Rate Exploration**
    - Train with `learning_rate` ∈ {0.001, 0.01, 0.1, 1.0}.
    - Plot cost vs. iteration. Identify stable vs. divergent regimes.
2. **Feature Scaling Impact**
    - Create one feature with large variance (×100).
    - Train with and without standardization; compare convergence speeds.
3. **Mini-Batch Gradient Descent**
    - Modify loop to sample batches of size 32.
    - Track cost per batch and epoch. Compare stability and speed to batch GD.
4. **Momentum Extension**
    - Implement:
        
        ```python
        v_dw, v_db = 0, 0
        β = 0.9
        v_dw = β*v_dw + (1-β)*dw
        v_db = β*v_db + (1-β)*db
        w -= learning_rate * v_dw
        b -= learning_rate * v_db
        ```
        
    - Observe how the path to minimum smooths out.
5. **Early Stopping**
    - Split data into train/validation.
    - Stop training when validation cost increases for 10 consecutive checks.

---

## Gradient Descent on m Examples

### 1. Intuition & Overview

Batch gradient descent computes the gradient of the cost function using all m training examples at once.

- You start with initial parameters (weights and bias).
- In each iteration (step), you measure how “wrong” your model is on the entire dataset (cost).
- You compute the direction of steepest descent (gradient) by aggregating errors over all m examples.
- You update parameters in that direction by a small amount (learning rate).
- Repeat until the cost stops improving.

Because you use all examples every step, the trajectory is smooth and moves directly toward the global minimum—but each step can be costly when m is large.

### 2. Notation & Setup

- X: input data matrix, shape = (n_features, m_examples)
- Y: true labels vector, shape = (1, m_examples)
- w: weights vector, shape = (n_features, 1)
- b: bias (scalar)
- α: learning rate (step size)
- num_iters: total number of gradient descent iterations

All operations below assume you’ve imported NumPy as `np`.

### 3. Forward Pass & Cost

```python
# 1. Linear combination (vectorized over all m examples)
Z = w.T @ X + b        # shape = (1, m)

# 2. Activation (sigmoid for binary classification)
A = 1 / (1 + np.exp(-Z))   # shape = (1, m)

# 3. Cost (binary cross-entropy over m examples)
cost = - (1/m) * np.sum(
    Y * np.log(A) +
    (1 - Y) * np.log(1 - A)
)
```

- `Z` gives a raw score for each example.
- `A` is the predicted probability of class 1 for each example.
- `cost` aggregates the error across all m examples.

### 4. Gradient Computation

```python
# 1. Compute error term
dZ = A - Y              # shape = (1, m)

# 2. Compute gradients w.r.t parameters
dw = (1/m) * X @ dZ.T   # shape = (n_features, 1)
db = (1/m) * np.sum(dZ) # scalar
```

- `dZ` measures how far predictions are from ground truth for each example.
- `dw` and `db` are the average sensitivities of the cost to changes in w and b.

### 5. Parameter Update

```python
w = w - α * dw
b = b - α * db
```

- You move parameters **downhill** on the cost surface by α times the gradient.
- Repeat this update until convergence.

### 6. Full Training Loop

```python
def batch_gradient_descent(X, Y, w, b, α, num_iters):
    m = X.shape[1]
    costs = []

    for i in range(num_iters):
        # Forward pass
        Z = w.T @ X + b
        A = 1 / (1 + np.exp(-Z))

        # Cost
        cost = - (1/m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1 - A)
        )

        # Backward pass (gradients)
        dZ = A - Y
        dw = (1/m) * X @ dZ.T
        db = (1/m) * np.sum(dZ)

        # Parameter update
        w = w - α * dw
        b = b - α * db

        # Record and optionally print cost
        if i % 100 == 0:
            costs.append(cost)

    return w, b, costs
```

- **Inputs**: data `X, Y`, initial `w, b`, learning rate `α`, iterations.
- **Outputs**: optimized `w, b`, history of costs every 100 steps.

### 7. Convergence & Learning Rate

- α too large → cost oscillates or diverges.
- α too small → extremely slow convergence.
- Common strategy: try α in {1e-3, 1e-2, 1e-1} and plot cost vs iterations.
- You can also use a **learning rate schedule** that decays α over time:
    
    ```python
    α_t = α0 / (1 + decay_rate * t)
    ```
    

### 8. Variants of Gradient Descent

1. **Batch GD**
    - Uses all m examples every update.
    - Smooth trajectory, high per-step cost.
2. **Stochastic GD**
    - Uses batch_size = 1 (one example per update).
    - Noisy trajectory, low per-step cost, can escape shallow minima.
3. **Mini-Batch GD**
    - Uses batches of size b (e.g., 32 or 64).
    - Trade-off: smoother than stochastic, cheaper than batch.

### 9. Optimization Enhancements

### 9.1 Momentum

```python
v_dw, v_db = 0, 0
β = 0.9

v_dw = β * v_dw + (1 - β) * dw
v_db = β * v_db + (1 - β) * db

w = w - α * v_dw
b = b - α * v_db
```

- Accelerates convergence along consistent directions.

### 9.2 Nesterov Accelerated Gradient (NAG)

```python
# lookahead step
w_ahead = w - β * v_dw
b_ahead = b - β * v_db

# compute gradients at the lookahead
dZ = sigmoid(w_ahead.T @ X + b_ahead) - Y
dw = (1/m) * X @ dZ.T
db = (1/m) * np.sum(dZ)

# update velocity and parameters
v_dw = β * v_dw + α * dw
v_db = β * v_db + α * db

w = w - v_dw
b = b - v_db
```

### 9.3 Adagrad

```python
cache_dw += dw**2
cache_db += db**2

w = w - (α / (np.sqrt(cache_dw) + ε)) * dw
b = b - (α / (np.sqrt(cache_db) + ε)) * db
```

### 9.4 RMSProp

```python
decay = 0.9
cache_dw = decay * cache_dw + (1 - decay) * dw**2
cache_db = decay * cache_db + (1 - decay) * db**2

w = w - (α / (np.sqrt(cache_dw) + ε)) * dw
b = b - (α / (np.sqrt(cache_db) + ε)) * db
```

### 9.5 Adam (Adaptive Moment Estimation)

```python
# moment estimates
m_dw = β1 * m_dw + (1 - β1) * dw
m_db = β1 * m_db + (1 - β1) * db

v_dw = β2 * v_dw + (1 - β2) * dw**2
v_db = β2 * v_db + (1 - β2) * db**2

# bias correction
m_dw_corr = m_dw / (1 - β1**t)
m_db_corr = m_db / (1 - β1**t)
v_dw_corr = v_dw / (1 - β2**t)
v_db_corr = v_db / (1 - β2**t)

w = w - (α * m_dw_corr) / (np.sqrt(v_dw_corr) + ε)
b = b - (α * m_db_corr) / (np.sqrt(v_db_corr) + ε)
```

### 10. Practical Considerations

- Always **shuffle** your data before each epoch for mini-batch and stochastic GD.
- **Scale/normalize** features so they all lie in similar ranges.
- Clip `A` to `[ε, 1-ε]` before log to avoid NaNs:
    
    ```python
    A = np.clip(A, 1e-15, 1-1e-15)
    ```
    
- Initialize weights to small random values for deep nets; zeros are fine for simple logistic regression.
- Track both **training** and **validation** cost to diagnose overfitting.

### 11. Computational Complexity

- Batch GD per iteration:
    - Forward pass: O(n_features × m)
    - Backward pass: O(n_features × m)
- Mini-batch GD with batch_size b: cost per update reduces proportionally.
- Second-order methods may require O((n_features)²) for Hessian operations.

---

## Vectorization

### 1. What Is Vectorization?

Vectorization is the practice of replacing explicit loops in Python with array-based operations that run in optimized, low-level code.

- Python `for`loops incur overhead on every iteration
- Libraries like NumPy implement operations in C, processing entire arrays at once
- Vectorized code is not only more concise but typically 10×–100× faster

### 2. NumPy Universal Functions (ufuncs)

Universal functions apply an operation elementwise over entire arrays.

```python
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y = np.exp(x)        # compute e^x for every element
z = np.sin(x) + 5    # combine sin and scalar add
```

Common ufuncs: `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.exp`, `np.log`, `np.sqrt`, `np.maximum`, `np.minimum`.

### 3. Matrix and Vector Operations

Linear algebra routines in NumPy use BLAS/LAPACK under the hood.

```python
# vector dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_ab = a @ b       # equals 1*4 + 2*5 + 3*6 = 32

# matrix multiplication
M = np.array([[1, 2], [3, 4]])
N = np.array([[5, 6], [7, 8]])
P = M @ N            # shape (2,2)
```

- `@` or `np.dot` for multiplication
- `np.matmul` supports higher-dimensional stacks

### 4. Broadcasting Rules

Broadcasting lets operations work on arrays of different shapes by “stretching” dimensions.

- If shapes differ, NumPy aligns trailing axes
- A dimension of size 1 can be repeated to match the other array
- New axes can be introduced with `None` or `np.newaxis`

```python
X = np.array([[1,2,3],
              [4,5,6]])      # shape (2,3)
v = np.array([10,20,30])      # shape (3,)
Y = X + v                     # v is broadcast to (2,3)

# adding bias term to each column
b = np.array([[1],[2]])       # shape (2,1)
Z = X + b                     # b broadcast to (2,3)
```

### 5. Replacing Python Loops with Vectorized Code

Example: sum of squares of a 1D list vs vectorized array:

```python
# using a Python loop
nums = [1, 2, 3, 4, 5]
total = 0
for x in nums:
    total += x**2

# vectorized with NumPy
arr = np.array(nums)
total_vec = np.sum(arr**2)
```

Example: logistic regression forward pass on m examples:

```python
# inputs
X = np.random.randn(n_features, m)
w = np.random.randn(n_features, 1)
b = 0.0

# vectorized forward
Z = w.T @ X + b
A = 1 / (1 + np.exp(-Z))
```

No explicit loop over examples is needed.

### 6. Performance Comparison

Use `%timeit` in Jupyter or `time.perf_counter()`:

```python
import time, numpy as np

# prepare data
m = 10_000_000
x = np.random.randn(m)
start = time.perf_counter()
# vectorized
y = x * 2 + 3
end = time.perf_counter()
print("Vectorized:", end-start)

# loop
start = time.perf_counter()
y2 = [xi*2 + 3 for xi in x]
end = time.perf_counter()
print("Loop:", end-start)
```

Typically, vectorized version runs an order of magnitude faster.

### 7. Advanced Vectorization Patterns

1. Boolean masking
    
    ```python
    a = np.arange(10)
    mask = (a % 2 == 0)
    evens = a[mask]      # select only even elements
    ```
    
2. Fancy indexing
    
    ```python
    X = np.random.randn(5,5)
    rows = [0,2,4]
    cols = [1,3,0]
    sub = X[rows, cols]  # shape (3,)
    ```
    
3. Vectorized conditionals
    
    ```python
    x = np.linspace(-2,2,9)
    y = np.where(x > 0, x**2, x**3)
    ```
    
4. One-hot encoding
    
    ```python
    labels = np.array([0,2,1,3])
    num_classes = 4
    one_hot = np.eye(num_classes)[labels]
    ```
    

### 8. Memory Layout & In-Place Operations

- NumPy arrays can be **C-contiguous** (row-major) or **Fortran-contiguous** (column-major)
- Slicing often returns a **view**, not a copy
- In-place ops (`+=`, `=`) save memory but can lead to unintended side-effects

```python
a = np.arange(6).reshape(2,3)
b = a[:, :2]         # b is a view on a
b[:] = 99            # modifies a as well
```

Use `a.copy()` to force a standalone array.

### 9. GPU-Accelerated Vectorization

Libraries like CuPy mimic NumPy API on NVIDIA GPUs:

```python
import cupy as cp

x_gpu = cp.random.randn(1000_000)
y_gpu = cp.sin(x_gpu) * cp.exp(-x_gpu**2)
y = cp.asnumpy(y_gpu)   # move back to host memory
```

Frameworks such as JAX, PyTorch, TensorFlow also provide vectorized GPU ops.

### 10. JIT and Parallel Vectorization

### 10.1 Numba JIT

```python
from numba import njit
import numpy as np

@njit
def vector_add(a, b):
    return a + b

# now this runs at native speed
```

### 10.2 Multiprocessing / Dask

- Dask arrays let you work with chunks larger than memory
- `dask.array` API mirrors NumPy but distributes work across cores/machines

### 11. Real-World Vectorization Projects

- **Image processing**: apply filters via 2D convolution with `scipy.signal.convolve2d` or FFT
- **Signal processing**: vectorized Fourier transforms (`np.fft.fft`)
- **NLP**: build term-document matrices with sparse vector representations
- **Finance**: compute rolling statistics with array strides

---

## Vectorizing Logistic Regression

### 1. Why Vectorize?

Vectorization replaces Python loops with array operations, giving you

- concise code
- 10×–100× speedups via NumPy’s C routines
- clear mapping from math to code for clean Notion notes

### 2. Shapes & Notation

- `X` : input data, shape = (n_features, m_examples)
- `Y` : labels, shape = (1, m_examples)
- `w` : weights, shape = (n_features, 1)
- `b` : bias, scalar
- `m` : number of examples

### 3. Vectorized Forward Pass

```python
# Z = w^T X + b   shape = (1, m)
Z = w.T @ X + b

# A = sigmoid(Z)  shape = (1, m)
A = 1 / (1 + np.exp(-Z))
```

### 4. Vectorized Cost Computation

```python
# binary cross-entropy
cost = - (1/m) * np.sum(
    Y * np.log(A) +
    (1 - Y) * np.log(1 - A)
)
```

### 5. Vectorized Backward Pass (Gradients)

```python
# error term
dZ = A - Y             # shape = (1, m)

# gradients
dw = (1/m) * X @ dZ.T  # shape = (n_features, 1)
db = (1/m) * np.sum(dZ)
```

### 6. Parameter Update

```python
w = w - alpha * dw
b = b - alpha * db
```

### 7. Prediction Function

```python
def predict(X, w, b):
    Z = w.T @ X + b
    A = 1 / (1 + np.exp(-Z))
    return (A > 0.5).astype(int)
```

### 8. Full Vectorized Training Loop

```python
def train_logistic(X, Y, alpha, num_iters):
    n_features, m = X.shape
    w = np.zeros((n_features, 1))
    b = 0.0
    costs = []

    for i in range(num_iters):
        Z = w.T @ X + b
        A = 1 / (1 + np.exp(-Z))
        cost = - (1/m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1 - A)
        )

        dZ = A - Y
        dw = (1/m) * X @ dZ.T
        db = (1/m) * np.sum(dZ)

        w -= alpha * dw
        b -= alpha * db

        if i % 100 == 0:
            costs.append(cost)

    return w, b, costs
```

### 9. Mini-Batch & Shuffling

```python
def iterate_mini_batches(X, Y, batch_size):
    m = X.shape[1]
    perm = np.random.permutation(m)
    X_shuffled = X[:, perm]
    Y_shuffled = Y[:, perm]

    for start in range(0, m, batch_size):
        end = start + batch_size
        yield X_shuffled[:, start:end], Y_shuffled[:, start:end]
```

- call `train_logistic` per batch inside each epoch
- trade off stability (batch) vs speed (stochastic)

### 10. Performance Tips

- standardize features: zero mean, unit variance
- clip `A` before `log`:
    
    ```python
    A = np.clip(A, 1e-15, 1 - 1e-15)
    ```
    
- choose `alpha` in [1e-3, 1e-1] and plot cost
- use `@` operator, not elementwise loops

### 11. Numerical Stability

```python
# avoid overflow in exp
Z = np.clip(Z, -500, 500)
A = 1 / (1 + np.exp(-Z))
```

### 12. Exercises

- compare batch vs mini-batch vs SGD on toy data
- add L2 regularization:
    
    ```python
    cost += (lambda_/ (2*m)) * np.sum(w**2)
    dw += (lambda_/m) * w
    
    ```
    
- implement momentum update
- experiment with learning rate schedules

---

## Vectorizing Logistic Regression’s Gradient Output

### 1. Why Vectorize?

- removes explicit Python loops over examples
- maps math directly to fast C‐level NumPy routines
- yields concise, copy-pasteable code for your Notion notes
- scales seamlessly from toy datasets to millions of examples

### 2. Notation & Shapes

- **X**: input features, shape = `(n_features, m_examples)`
- **Y**: true labels (0 or 1), shape = `(1, m_examples)`
- **w**: weights, shape = `(n_features, 1)`
- **b**: bias scalar
- **m**: number of examples (`m = X.shape[1]`)
- **A**: model output (sigmoid activations), shape = `(1, m_examples)`

### 3. Naive Loop Implementation (for contrast)

```python
dw = np.zeros_like(w)
db = 0.0

for i in range(m):
    xi = X[:, i].reshape(-1,1)   # shape = (n_features, 1)
    yi = Y[0, i]

    zi = float(w.T @ xi + b)     # scalar
    ai = 1 / (1 + np.exp(-zi))   # scalar

    dZi = ai - yi                # scalar
    dw += xi * dZi               # accumulate gradients
    db += dZi

dw = dw / m
db = db / m
```

- loops over `m` examples → slow when `m` is large

### 4. Fully Vectorized Gradient Computation

```python
# 1. Forward pass
Z  = w.T @ X + b                 # shape = (1, m)
A  = 1 / (1 + np.exp(-Z))        # sigmoid, shape = (1, m)

# 2. Compute error term
dZ = A - Y                       # shape = (1, m)

# 3. Compute gradients in one shot
dw = (1/m) * X @ dZ.T            # shape = (n_features, 1)
db = (1/m) * np.sum(dZ)          # scalar
```

- single matrix multiplication replaces the loop
- keep shapes straight: `X @ dZ.T` sums over examples

### 5. Vectorized Gradient with L2 Regularization

```python
lambda_ = 0.1

# error term
dZ = A - Y

# regularized gradients
dw = (1/m) * X @ dZ.T + (lambda_/m) * w
db = (1/m) * np.sum(dZ)
```

- adds `(λ/m)*w` to shrink weights toward zero
- no change to `db`

### 6. Numerical Stability Tricks

```python
# 1. Clip Z to avoid overflow in exp
Z = w.T @ X + b
Z = np.clip(Z, -500, 500)

# 2. Compute sigmoid safely
A = 1 / (1 + np.exp(-Z))

# 3. Clip A to avoid log(0) downstream
A = np.clip(A, 1e-15, 1-1e-15)
```

- prevents `exp` overflow/underflow
- prevents `log(0)` in cost or gradient checks

### 7. Full Training Loop (Vectorized)

```python
def train_logistic(X, Y, alpha, num_iters, lambda_=0.0):
    n_features, m = X.shape
    w = np.zeros((n_features, 1))
    b = 0.0
    costs = []

    for i in range(num_iters):
        # forward pass
        Z = w.T @ X + b
        Z = np.clip(Z, -500, 500)
        A = 1 / (1 + np.exp(-Z))
        A = np.clip(A, 1e-15, 1-1e-15)

        # cost (optional record)
        cost = - (1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
        if lambda_ > 0:
            cost += (lambda_/(2*m)) * np.sum(w**2)

        # backward pass
        dZ = A - Y
        dw = (1/m) * X @ dZ.T
        db = (1/m) * np.sum(dZ)

        # add regularization term
        if lambda_ > 0:
            dw += (lambda_/m) * w

        # parameter update
        w -= alpha * dw
        b -= alpha * db

        # record cost every 100 steps
        if i % 100 == 0:
            costs.append(cost)

    return w, b, costs
```

### 8. Mini-Batch & Stochastic Variants

```python
def iterate_minibatches(X, Y, batch_size):
    m = X.shape[1]
    perm = np.random.permutation(m)
    X_shuf = X[:, perm]
    Y_shuf = Y[:, perm]
    for k in range(0, m, batch_size):
        yield X_shuf[:, k:k+batch_size], Y_shuf[:, k:k+batch_size]
```

- in training loop, call this per epoch
- update `w, b` for each mini-batch

### 9. Debugging & Pitfalls

- assert shapes at each step:
    
    ```python
    assert Z.shape == (1, m)
    assert dw.shape == w.shape
    ```
    
- visualize cost curve → smooth decrease
- check gradient magnitudes; extremely large dw indicates too large α
- test on a tiny dataset where you can compute dw by hand

### 10. Exercises

- **Gradient Check**: numerically approximate gradients and compare to vectorized `dw, db`
- **Momentum**: add velocity terms to accelerate convergence
- **Adaptive Optimizers**: implement AdaGrad, RMSProp, Adam using vectorized updates
- **Regularization Sweep**: vary λ in {0.0, 0.01, 0.1, 1.0} and observe weight norms
- **Feature Scaling**: train with unscaled vs standardized X and compare convergence

---

## Broadcasting in Python (NumPy)

### What Is Broadcasting?

Broadcasting is NumPy’s way of performing element-wise operations on arrays of different shapes without explicit loops or manual replication. It “stretches” the smaller array along the mismatched dimensions so that arithmetic operations proceed as if both arrays had the same shape.

Benefits of broadcasting:

- Concise, loop-free code
- Leverages optimized C implementations for speed
- Simplifies common tasks (adding bias, scaling, distance computations)

### Broadcasting Rules

NumPy compares array shapes element by element, starting from the trailing (rightmost) dimensions. Two dimensions are compatible if:

1. They are equal, or
2. One of them is 1, or
3. One of the arrays has no dimension at that position (i.e., its shape is shorter).

If all dimensions are compatible, NumPy virtually “repeats” the array with size 1 along that axis to match the other shape.

### Step-by-Step Example

Suppose you have:

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape = (2, 3)

b = np.array([10, 20, 30])      # shape = (3,)
```

Operation:

```python
C = A + b
```

Broadcasting alignment:

```
 A.shape = (2, 3)
 b.shape =    (3)   → treated as (1, 3)
 b “stretched” to shape (2, 3)
```

Result:

```python
C = [[11, 22, 33],
     [14, 25, 36]]
```

### Rules Illustrated

1. Align trailing dims:
    - A is (2, 3)
    - b is (3) → align with last dim of A
2. Check compatibility:
    - dim1: 3 vs 3 → OK
    - dim0: 2 vs “no dim” → treat as 1 → OK
3. Stretch b from (1, 3) → (2, 3) by repeating the single row

### Common Broadcasting Patterns

### 1. Adding a Bias Vector to Each Column

```python
# X: features × examples
X = np.random.randn(5, 100)   # shape = (5, 100)
b = np.random.randn(5, 1)     # shape = (5, 1)

# add bias to each example
Z = X + b                     # b broadcasts from (5,1) to (5,100)
```

### 2. Normalizing Columns (Zero-mean)

```python
# compute column means
col_means = np.mean(X, axis=0, keepdims=True)  # shape = (1, 100)

# subtract per-column mean
X_centered = X - col_means                     # broadcasts (1,100) → (5,100)
```

### 3. Outer Operations (Pairwise Distances)

```python
# vector of length n
v = np.arange(4)              # shape = (4,)

# compute pairwise differences
D = v[:, np.newaxis] - v      # shapes: (4,1) and (4,) → result (4,4)
```

### Advanced Broadcasting Techniques

### Introducing New Axes

Use `np.newaxis` or `None` to add a singleton dimension.

```python
x = np.array([1,2,3])        # shape = (3,)
x_col = x[:, None]           # shape = (3,1)
x_row = x[None, :]           # shape = (1,3)
```

### Broadcasting in Higher Dimensions

```python
# A: (2,3,4), B: (3,4)
A = np.zeros((2,3,4))
B = np.ones((3,4))

# B treated as (1,3,4), broadcast to (2,3,4)
C = A + B                     # shape = (2,3,4)
```

### Pitfalls & Tips

- Silent errors:
    
    An unintended singleton dimension can still broadcast, leading to wrong results without an exception. Always verify shapes.
    
- Memory usage:
    
    Broadcasting doesn’t actually replicate data in memory, but operations may allocate large result arrays. Monitor array sizes.
    
- Performance:
    
    Vectorized broadcasting is fast, but chaining many operations can create intermediate arrays. Use in-place operations (e.g., `+=`) where safe.
    
- Always check with assertions:
    
    ```python
    assert X.shape[1] == b.shape[0]  # for X + b along axis 0
    ```
    

### Full Python Example

```python
import numpy as np

# 1. Create data
m, n = 1000, 5
X = np.random.randn(m, n)   # 1000 samples, 5 features

# 2. Feature-wise mean and std
mean = np.mean(X, axis=0)           # shape = (5,)
std  = np.std(X, axis=0)            # shape = (5,)

# 3. Standardize features
X_std = (X - mean) / std            # broadcasting (m,5) - (5,) → (m,5)

# 4. Add intercept term (column of ones)
ones = np.ones((m, 1))              # shape = (m,1)
X_aug = np.concatenate([ones, X_std], axis=1)  # result (m,6)

# 5. Compute predictions for logistic regression
w = np.random.randn(6, 1)           # weights
Z = X_aug @ w                       # shape = (m,1)
A = 1 / (1 + np.exp(-Z))            # sigmoid, broadcasts scalar operations

print("Shapes:", X_aug.shape, w.shape, Z.shape, A.shape)
```

### When Broadcasting Fails

If dimensions aren’t compatible under the rules, NumPy raises a `ValueError`:

```python
a = np.zeros((2,3))
b = np.zeros((2,2))
c = a + b   # ValueError: operands could not be broadcast together

```

Diagnose by printing `a.shape` and `b.shape`.

---