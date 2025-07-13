# Mathematics_c2_w3

## Regression with a Perceptron

### 1. Prerequisites

Make sure you already understand:

- Linear regression (model = dot-product + bias)
- Mean squared error loss and its gradients【Optimization Using Gradient Descent – least squares】
- Basic perceptron/classifier structure (weighted sum + bias + activation)

### 2. Conceptual Overview

A perceptron is the simplest neural network unit. It:

- Takes an input vector **x** = [x₁, x₂, …, xₙ].
- Computes a weighted sum plus bias:
    
    ```
    z = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
    ```
    
- Applies an **activation** σ(z) to produce output ŷ.

For **regression**, we choose the identity activation σ(z)=z. The perceptron becomes exactly a linear regression model:

```
y_pred = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
```

We then train the weights and bias to minimize the mean squared error between y_pred and the true y.

### 3. Model Formula

```
y_pred = w₁ * x₁ + w₂ * x₂ + … + wₙ * xₙ + b
```

- **w** = [w₁, w₂, …, wₙ]ᵀ are trainable weights.
- **b** is the trainable bias (intercept).
- **y_pred** is the model’s predicted continuous value.

### 4. Loss Function

We use Mean Squared Error (MSE) over m examples:

```
J(w, b) = (1 / (2m)) * Σ_{i=1..m} ( y_pred⁽ⁱ⁾ − y⁽ⁱ⁾ )²
```

- The factor 1/(2m) makes the gradient expressions cleaner (cancels the 2).
- y⁽ⁱ⁾ is the true target for the i-th observation.

### 5. Gradients

To update parameters by gradient descent, compute:

```
For each weight wⱼ:
∂J/∂wⱼ = (1/m) * Σ_{i=1..m} [ (y_pred⁽ⁱ⁾ − y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾ ]

For bias b:
∂J/∂b = (1/m) * Σ_{i=1..m} [ (y_pred⁽ⁱ⁾ − y⁽ⁱ⁾) ]
```

- `(y_pred⁽ⁱ⁾ − y⁽ⁱ⁾)` is the prediction error for example i.
- Multiplying by xⱼ⁽ⁱ⁾ measures how that feature contributes.

### 6. Gradient Descent Update Rules

Choose a learning rate α, then repeat until convergence:

```
wⱼ_new = wⱼ_old − α * (1/m) * Σ [ (y_pred⁽ⁱ⁾ − y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾ ]

b_new   = b_old   − α * (1/m) * Σ [ (y_pred⁽ⁱ⁾ − y⁽ⁱ⁾) ]
```

After each step, the perceptron’s hyperplane moves slightly closer to the best least-squares fit.

### 7. Python Example (Scratch Implementation)

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate toy data (m samples, n=2 features)
np.random.seed(0)
m = 100
X = 2 * np.random.rand(m, 2)                   # shape (m,2)
true_w = np.array([3.0, -2.0])                 # ground-truth weights
true_b = 4.0
y = X.dot(true_w) + true_b + np.random.randn(m) * 1.0

# 2. Initialize parameters
w = np.zeros(2)
b = 0.0
alpha = 0.05
n_iter = 500
costs = []

# 3. Training loop: gradient descent
for _ in range(n_iter):
    # Forward pass
    y_pred = X.dot(w) + b                      # shape (m,)
    error  = y_pred - y                        # shape (m,)

    # Compute gradients
    grad_w = (1/m) * X.T.dot(error)            # shape (2,)
    grad_b = (1/m) * np.sum(error)             # scalar

    # Update parameters
    w -= alpha * grad_w
    b -= alpha * grad_b

    # Track cost
    cost = (1/(2*m)) * np.sum(error**2)
    costs.append(cost)

print("Learned parameters:", w, b)
print("True parameters   :", true_w, true_b)

# 4. Plot cost over iterations
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("MSE Loss Convergence")
plt.show()
```

### 8. Geometric Interpretation

- The perceptron defines a hyperplane in feature space:
    
    ```
    { x | w·x + b = constant }
    ```
    
- Minimizing MSE “tilts” and “lifts” that plane to best slice through the cloud of points ((x^{(i)}, y^{(i)})).
- Each gradient step adjusts the normal vector **w** and offset **b** to reduce the vertical errors.

### 9. Practice Problems

1. **Single-feature fit**
    - Generate data y=5+2x+noise. Train a single-weight perceptron (n=1) by gradient descent. Plot fit.
2. **Feature scaling**
    - Add a feature x₂ = x₁². Fit w₁, w₂, b. Compare convergence with and without normalizing x₂.
3. **Batch vs Stochastic**
    - Convert the loop to update parameters per-example (stochastic GD). Compare the noise in cost curves.
4. **Regularized regression**
    - Add an L2 penalty λ·‖w‖² to J. Derive new gradients and implement ridge-type perceptron.

### 10. Real-World Application

- **House price prediction:** A perceptron maps features like square footage, number of rooms to price.
- **Demand forecasting:** Weather variables fed into a perceptron estimate electricity load.
- **Feature embeddings:** In deep models, each neuron acts like a perceptron combining inputs into one signal.

By viewing linear regression as training a perceptron with identity activation, you build a smooth bridge between classical statistics and neural network foundations.

---

## Regression with a Perceptron – Loss Function

### 1. Prerequisites

You should already understand:

- The perceptron model for regression:
    
    ```
    y_pred = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
    ```
    
- Why we need a “score” of how well predictions match targets.

### 2. Conceptual Overview

The **loss function** measures the discrepancy between the perceptron’s output (y_pred) and the true value (y).

- It assigns a non‐negative number to each prediction–target pair.
- A perfect prediction gives loss = 0.
- Larger errors incur larger penalties.

For regression, the most common choice is the **mean squared error** (MSE), which squares each error to penalize large misses.

### 3. Mean Squared Error (MSE) Loss

The MSE over a training set of m examples is:

```
J(w, b) = (1 / (2m)) * sum_{i=1..m} ( y_pred⁽ⁱ⁾ - y⁽ⁱ⁾ )²
```

- `y_pred⁽ⁱ⁾ = w·x⁽ⁱ⁾ + b`
- `y⁽ⁱ⁾` is the true target for the i-th example
- The factor `1/(2m)`
    - Averages over m samples
    - Cancels the 2 when differentiating

### 4. Breaking Down the Formula

1. **Error term** for example i:
    
    ```
    e⁽ⁱ⁾ = y_pred⁽ⁱ⁾ - y⁽ⁱ⁾
    ```
    
2. **Square the error**:Squaring amplifies large mistakes.
    
    ```
    (e⁽ⁱ⁾)²
    ```
    
3. **Sum over all examples**:
    
    ```
    sum_{i=1..m} (e⁽ⁱ⁾)²
    ```
    
4. **Normalize by 2m**:This gives the final average loss J(w, b).
    
    ```
    (1/(2m)) * sum_{i=1..m} (e⁽ⁱ⁾)²
    ```
    

### 5. Why Square the Error?

- **Convexity**: Squared error yields a smooth, convex loss surface—guaranteed single global minimum.
- **Differentiability**: The squared term is easy to differentiate, leading to simple gradient formulas.
- **Symmetry**: Positive and negative errors are penalized equally.

### 6. Python Code: Computing MSE

```python
import numpy as np

def mse_loss(w, b, X, y):
    """
    Computes the mean squared error loss.
    w: weight vector (n,)
    b: bias scalar
    X: feature matrix (m, n)
    y: targets vector (m,)
    """
    m = X.shape[0]
    y_pred = X.dot(w) + b             # predictions (m,)
    errors = y_pred - y               # errors (m,)
    return (1/(2*m)) * np.dot(errors, errors)

# Example usage
np.random.seed(0)
m, n = 5, 2
X = np.random.randn(m, n)
true_w = np.array([2.0, -1.0])
true_b = 0.5
y = X.dot(true_w) + true_b + np.random.randn(m) * 0.1

initial_w = np.zeros(n)
initial_b = 0.0
print("Initial MSE:", mse_loss(initial_w, initial_b, X, y))
print("True MSE   :", mse_loss(true_w, true_b, X, y))
```

### 7. Practice Problems

1. **By hand**: For data points
    
    ```
    (x, y): (1,2), (2,4), (3,5)
    ```
    
    and model y_pred = w·x + b, compute J(w,b) at (w=1, b=1).
    
2. **Code exercise**:
    - Write a function `mse_gradient(w, b, X, y)` that returns gradients ∂J/∂w and ∂J/∂b.
    - Use small random data to verify finite‐difference approximations.
3. **Visualize**:
    - Fix b=0.
    - Plot J(w,0) vs w over range [−2, 4] for the points above.
    - Observe the convex “bowl” shape.
4. **Compare losses**:
    - Implement mean absolute error (MAE):
        
        ```
        MAE(w,b) = (1/m) * sum |y_pred⁽ⁱ⁾ - y⁽ⁱ⁾|
        
        ```
        
    - Plot MSE vs MAE on the same data and discuss differences.

### 8. Real-World Application

In **housing price prediction**, squaring errors punishes huge over- or under-estimates (e.g., predicting $100k off incurs a 100× larger penalty than a $10k error).

MSE is standard in:

- Simple and polynomial regression
- Training layers in neural networks for continuous outputs
- Autoencoder reconstruction loss

---

## Regression with a Perceptron – Gradient Descent

### 1. Prerequisites

- A perceptron for regression has the form
    
    ```
    y_pred = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
    ```
    
- We measure fit with mean squared error:
    
    ```
    J(w,b) = (1/(2m)) · Σ (y_pred⁽ⁱ⁾ - y⁽ⁱ⁾)²
    ```
    

Ensure you know how to compute basic derivatives and the idea of updating parameters iteratively.

### 2. Conceptual Overview

Gradient descent finds the weight vector **w** and bias **b** that minimize the loss by taking repeated small steps in the direction that decreases the loss fastest.

At each step, we compute how the loss changes with each parameter (the gradient) and move opposite that direction, scaled by a learning rate α.

### 3. Deriving the Gradients

We need the partial derivatives of the loss with respect to each parameter.

For weight wⱼ:

```
dJ/dwⱼ = (1/m) · Σ_{i=1..m} [ (y_pred⁽ⁱ⁾ - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾ ]
```

- y_pred⁽ⁱ⁾ - y⁽ⁱ⁾ is the error for example i
- Multiplying by xⱼ⁽ⁱ⁾ weighs the error by feature j

For bias b:

```
dJ/db = (1/m) · Σ_{i=1..m} [ (y_pred⁽ⁱ⁾ - y⁽ⁱ⁾) ]
```

- Each error contributes equally to the shift in b

These formulas come from differentiating the squared error and cancelling the 2 with the 1/2m factor.

### 4. Gradient Descent Update Rules

Once we have the gradients, each iteration applies:

```
wⱼ_new = wⱼ_old - α · (1/m) · Σ [ (y_pred - y) · xⱼ ]

b_new   = b_old   - α · (1/m) · Σ [ (y_pred - y) ]
```

- α is the learning rate (step size)
- A small α yields slow, stable convergence; a large α may overshoot

In vector form, with X as an m×n matrix and y as an m-vector:

```
predictions = X·w_old + b_old·1
errors      = predictions - y

grad_w = (1/m) · Xᵀ · errors
grad_b = (1/m) · sum(errors)

w_new = w_old - α·grad_w
b_new = b_old - α·grad_b
```

### 5. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate data
np.random.seed(42)
m, n = 100, 2
X = np.random.randn(m, n)              # features
true_w = np.array([2.5, -1.0])
true_b = 0.5
y = X.dot(true_w) + true_b + np.random.randn(m) * 0.5

# 2. Initialize parameters
w = np.zeros(n)
b = 0.0
alpha = 0.1
iterations = 500
costs = []

# 3. Gradient descent loop
for _ in range(iterations):
    y_pred = X.dot(w) + b              # predictions
    error = y_pred - y                 # shape (m,)

    grad_w = (1/m) * X.T.dot(error)    # shape (n,)
    grad_b = (1/m) * np.sum(error)     # scalar

    w -= alpha * grad_w
    b -= alpha * grad_b

    cost = (1/(2*m)) * np.sum(error**2)
    costs.append(cost)

# 4. Results
print("Learned w, b:", w, b)
print("True   w, b:", true_w, true_b)

# 5. Plot cost convergence
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Gradient Descent Convergence")
plt.show()
```

### 6. Geometric Interpretation

- The loss surface J(w,b) is a convex “bowl” in the (w,b)-space.
- The gradient ∇J points uphill; stepping opposite moves you downhill.
- Over iterations, the hyperplane defined by (w,b) rotates and shifts to fit the data cloud better.

### 7. Practice Problems

1. By hand, perform three GD updates on
    
    ```
    f(w, b) = (1/2m) Σ (w·x⁽ⁱ⁾ + b - y⁽ⁱ⁾)²
    ```
    
    for points (x,y) = {(1,2), (2,3)} starting at w₀=0, b₀=0 with α=0.1.
    
2. Modify the Python code to use **stochastic gradient descent** (update on one random example per step). Plot the cost over iterations.
3. Add an L2 penalty λ·‖w‖² to the cost and derive new gradient formulas. Implement ridge regression via gradient descent and compare to ordinary least squares.
4. Extend to a third feature x₃ = x₁·x₂ and observe how the model adapts with gradient descent.

---

## Classification with a Perceptron

### 1. What You Need to Know First

You should already understand:

- Linear models for regression (weighted sum plus bias)
- Vector dot‐product and bias term:
    
    ```
    z = w · x + b
    ```
    
- The idea of an **activation** that maps this score into a prediction

### 2. Conceptual Overview

A perceptron for **classification** takes a weighted sum of inputs and outputs one of two labels (e.g. +1 or −1):

- It computes a score `z = w·x + b`.
- It applies the **sign activation**:
    
    ```
    y_pred = +1  if z ≥ 0
           = −1  if z <  0
    ```
    
- During training, it adjusts `w` and `b` to push misclassified points to the correct side of the decision boundary.

Imagine drawing a straight line (in 2D) or hyperplane (in n-D) that separates the two classes. The perceptron learning rule nudges that boundary whenever a point falls on the wrong side.

### 3. Model and Prediction

**Model formula** in copy-paste-friendly form:

```
z = w₁*x₁ + w₂*x₂ + … + wₙ*xₙ + b
y_pred = sign(z)          // returns +1 or −1
```

- `w` is the weight vector
- `x` is the input feature vector
- `b` is the bias (intercept)
- `sign(z)` maps the continuous score to a discrete class label

### 4. Perceptron Learning Rule

When a training example `(xᵢ, yᵢ)` is misclassified (`y_pred ≠ yᵢ`), update:

```
if yᵢ * (w · xᵢ + b) ≤ 0:
    w ← w + α * yᵢ * xᵢ
    b ← b + α * yᵢ
```

Step by step:

1. Compute the margin `m = yᵢ * (w · xᵢ + b)`.
2. If `m > 0`, the point is correctly classified (no update).
3. If `m ≤ 0`, it’s wrong or on the boundary, so:
    - Move `w` toward the input `xᵢ` if `yᵢ=+1`, or away if `yᵢ=−1`.
    - Adjust `b` by `α*yᵢ` to shift the boundary.

`α` (alpha) is the learning rate, usually set to 1 for the classic perceptron.

### 5. Geometric Intuition

- The hyperplane `w·x + b = 0` splits feature space into two halves.
- Points with `w·x + b > 0` are labeled +1; those with `< 0` are −1.
- A misclassified point lies on the wrong side; updating shifts the boundary closer to that point’s correct side.
- Over many updates, the perceptron homes in on a separating hyperplane (if classes are linearly separable).

### 6. Python Implementation

```python
import numpy as np

def perceptron_train(X, y, alpha=1.0, n_iter=1000):
    """
    X: array of shape (m, n)
    y: labels (+1 or -1) of shape (m,)
    Returns learned (w, b)
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(n_iter):
        errors = 0
        for i in range(m):
            activation = np.dot(w, X[i]) + b
            if y[i] * activation <= 0:       # Misclassified
                w += alpha * y[i] * X[i]
                b += alpha * y[i]
                errors += 1
        if errors == 0:
            break  # Perfect separation found
    return w, b

# Example usage on a toy dataset
np.random.seed(0)
# Generate two separable classes in 2D
X_pos = np.random.randn(50,2) + [2,2]
X_neg = np.random.randn(50,2) + [-2,-2]
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(50), -np.ones(50)])

w, b = perceptron_train(X, y, alpha=1.0, n_iter=100)
print("Learned weights:", w, "bias:", b)
```

### 7. Practice Problems

1. **By hand**, run one epoch of perceptron updates on the points:
    
    ```
    X = [(2,1), (−1,−1), (1,−2)], y = [+1, −1, +1], α=1
    ```
    
    Start from w=[0,0], b=0. Show each update.
    
2. **Visualize** the decision boundary:
    - After training on the toy dataset above, plot data points and the line `w·x+b=0`.
3. **Non‐separable data**:
    - Add noise so classes overlap. Observe that the perceptron may not converge.
    - Limit to a fixed number of passes and track training errors per epoch.
4. **Margin analysis**:
    - Modify the update rule to stop updating points once they lie beyond a fixed margin `γ>0` (perceptron with margin).

### 8. Real-World ML/DS Applications

- Early spam filters separated “spam” vs “ham” emails by keyword features.
- Simple image classification (e.g., distinguishing two digits) with raw pixel inputs.
- As a building block of multi-layer neural networks: each neuron is a perceptron with a different activation.

### 9. Limitations

- The classic perceptron only finds a solution if the data are **linearly separable**.
- It does not produce probabilities or handle non-separable data well.

---

## Classification with a Perceptron – The Sigmoid Function

### 1. What You Need to Know First

Before exploring the sigmoid function, you should already understand:

- A linear score from a perceptron:
    
    ```
    z = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
    ```
    
- How a hard threshold (sign function) turns that score into +1/−1.
- Why a smooth, probabilistic output can be more useful in practice.

### 2. Conceptual Overview

The **sigmoid function** (also called logistic function) turns any real-valued score `z` into a number between 0 and 1.

- Think of it as “squashing” the infinite range of `z` into a probability.
- Near `z=0`, it changes rapidly, giving maximum sensitivity.
- As `z→±∞`, it flattens out, reflecting saturated confidence (close to 0 or 1).

Using sigmoid instead of sign lets us:

- Interpret model outputs as probabilities.
- Define a smooth loss (cross‐entropy) for gradient-based training.
- Avoid non-differentiability of the hard threshold.

### 3. The Sigmoid Formula

```
σ(z) = 1 / (1 + exp(−z))
```

**Breakdown**

1. `exp(−z)`: takes the exponential of the negative score.
2. `1 + exp(−z)`: shifts that to always be ≥1.
3. `1 / (…)`: inverts, mapping large positive `z` to values near 1, large negative `z` to near 0.

### 4. Key Properties

- Output range: `σ(z) ∈ (0, 1)`
- `σ(0) = 0.5`
- **Monotonic**: strictly increasing in `z`
- **Derivative**: the slope of the sigmoid itself isThis identity makes backpropagation especially efficient.
    
    ```
    dσ/dz = σ(z) * (1 − σ(z))
    ```
    

### 5. From Sigmoid to Classification

1. **Probability output**:
    
    ```
    p(y=1 | x) = σ(z)
    ```
    
2. **Decision rule** (threshold at 0.5):
    
    ```
    y_pred = 1  if σ(z) ≥ 0.5
           = 0  otherwise
    ```
    
3. **Confidence**: distance from 0.5 reflects how sure the model is.

### 6. Real-World ML Applications

- **Logistic regression**: fits `w` and `b` by minimizing cross‐entropy loss on sigmoid outputs.
- **Neural networks**: use sigmoid (or variants) in output layers for binary classification (e.g., spam detection).
- **Calibration**: mapping raw scores (e.g., credit-risk model outputs) into well-calibrated probabilities.

### 7. Python Exercises

### 7.1 Implement and Plot Sigmoid

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot
zs = np.linspace(-10, 10, 200)
plt.plot(zs, sigmoid(zs), label='σ(z)')
plt.axvline(0, color='gray', linestyle='--')
plt.axhline(0.5, color='gray', linestyle='--')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.legend()
plt.show()
```

### 7.2 Derivative Check

```python
# Numeric vs analytic derivative
h = 1e-5
z0 = 2.0
numeric = (sigmoid(z0 + h) - sigmoid(z0 - h)) / (2*h)
analytic = sigmoid(z0) * (1 - sigmoid(z0))
print("Numeric dσ/dz:", numeric)
print("Analytic dσ/dz:", analytic)
```

### 7.3 Simple Logistic Regression

```python
# Toy binary dataset
np.random.seed(0)
m = 100
X = np.vstack([np.random.randn(m//2,2)+2, np.random.randn(m//2,2)-2])
y = np.hstack([np.ones(m//2), np.zeros(m//2)])

# Initialize
w = np.zeros(2)
b = 0.0
alpha, epochs = 0.1, 200

for _ in range(epochs):
    z = X.dot(w) + b
    p = sigmoid(z)
    # gradients of cross-entropy
    error = p - y
    grad_w = (1/m) * X.T.dot(error)
    grad_b = (1/m) * error.sum()
    w -= alpha * grad_w
    b -= alpha * grad_b

# Evaluate
probs = sigmoid(X.dot(w) + b)
preds = (probs >= 0.5).astype(int)
print("Accuracy:", (preds == y).mean())
```

### 8. Practice Problems

1. **By hand**, compute `σ(−2)`, `σ(0)`, and `σ(3)`.
2. **Plot** the derivative `σ(z)*(1−σ(z))` over z∈[−10,10] and identify its maximum.
3. **Implement** logistic regression from scratch on the Iris dataset’s setosa vs. versicolor petal length feature.
4. **Compare** sigmoid + cross‐entropy vs. perceptron sign rule on the same data: how do updates differ?

---

## Classification with a Perceptron – Gradient Descent

### 1. Prerequisites

You should already know:

- The perceptron model’s score and sign activation:
    
    ```
    z = w · x + b
    y_pred = sign(z)        // +1 or −1
    ```
    
- Basic vector operations (dot‐product, addition).
- The idea of gradient descent: updating parameters by moving opposite to the gradient of a loss.

### 2. Why Gradient Descent for a Perceptron?

The classic perceptron update applies only when a point is misclassified. We can view this as **batch gradient descent** on a specific loss, the **perceptron loss**, which measures how badly points violate the margin.

By framing the perceptron rule as gradient descent, we:

- Tie it to a clear loss function.
- Can extend to other losses (hinge, logistic).
- Leverage the same optimization framework used elsewhere in ML.

### 3. The Perceptron Loss Function

For a dataset of m examples ((x^{(i)}, y^{(i)})) with labels (y^{(i)}\in{-1,+1}), define the perceptron loss:

```
L_perceptron(w, b)
  = (1/m) * Σ_{i=1..m} max(0, - y^{(i)} · (w · x^{(i)} + b))
```

- If (y^{(i)}·z^{(i)}) is positive (correct side), the max is 0 (no loss).
- If it’s negative or zero (misclassified or on boundary), the loss is (-y^{(i)}·z^{(i)}).

This “hinge‐less” variant penalizes any example not on the correct side of the hyperplane.

### 4. Gradient of the Perceptron Loss

The perceptron loss is piecewise‐linear, but its subgradient w.r.t. (w) and (b) is:

```
Let M = { i | y^{(i)}·(w·x^{(i)} + b) ≤ 0 }  // misclassified or on boundary

∂L/∂w = -(1/m) * Σ_{i∈M} y^{(i)} * x^{(i)}
∂L/∂b = -(1/m) * Σ_{i∈M} y^{(i)}
```

- We sum only over the “active” points M that incur loss.
- The negative sign appears because we differentiate max(0, −y·z).

### 5. Gradient Descent Update Rules

With learning rate α:

```
w_new = w_old - α * (∂L/∂w)
      = w_old + (α/m) * Σ_{i∈M} y^{(i)} * x^{(i)}

b_new = b_old - α * (∂L/∂b)
      = b_old + (α/m) * Σ_{i∈M} y^{(i)}
```

This batch update accumulates the classic perceptron corrections over all misclassified points each epoch.

### 6. Python Implementation

```python
import numpy as np

def perceptron_gd(X, y, alpha=1.0, epochs=100):
    """
    Batch gradient‐descent training of a perceptron via perceptron loss.
    X: (m, n) feature matrix
    y: (m,) labels in {-1, +1}
    Returns (w, b) after training.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        # Compute scores and margins
        scores  = X.dot(w) + b            # shape (m,)
        margins = y * scores             # shape (m,)

        # Identify misclassified or on‐boundary points
        mask = margins <= 0              # boolean array (m,)

        # Compute gradient components
        grad_w = -(1/m) * (y[mask, None] * X[mask]).sum(axis=0)  # shape (n,)
        grad_b = -(1/m) * y[mask].sum()                          # scalar

        # Parameter update
        w += alpha * (-grad_w)  # or w = w - alpha*grad_w
        b += alpha * (-grad_b)

        # Early exit if no errors
        if not mask.any():
            break

    return w, b

# Example usage on separable 2D data
np.random.seed(0)
X_pos = np.random.randn(50,2) + [2,2]
X_neg = np.random.randn(50,2) + [-2,-2]
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(50), -np.ones(50)])

w_learned, b_learned = perceptron_gd(X, y, alpha=1.0, epochs=100)
print("Learned w:", w_learned, "b:", b_learned)
```

### 7. Practice Problems

1. By hand, for points
    
    ```
    X = [(2,1), (−1,−1), (1,−2)], y = [+1, −1, +1]
    ```
    
    compute one batch‐GD update of w and b with α=1 starting from w=(0,0), b=0.
    
2. Modify the code above to run **stochastic** updates (one random example per step) and compare the final hyperplane to the batch version.
3. Experiment with different α values (0.1, 0.5, 2.0) and observe convergence speed and stability.
4. Extend to the **hinge loss**
    
    ```
    L_hinge(w,b) = (1/m) * Σ max(0, 1 - y^{(i)}·(w·x^{(i)} + b))
    ```
    
    derive gradients and implement batch GD.
    

### 8. Real‐World ML/DS Applications

- Fast, online binary classification (spam vs. ham, sentiment analysis) on high‐dimensional sparse data.
- Foundation for kernelized algorithms (kernel perceptron).
- Basis for support vector machines when extended to hinge‐loss and regularization.

---

## Classification with a Perceptron – Calculating the Derivatives

### 1. Model and Loss Recap

We have a binary‐classification perceptron with

```
zᵢ = w · x⁽ⁱ⁾ + b       // raw score for example i
y_pred⁽ⁱ⁾ = sign(zᵢ)     // +1 or −1 prediction
```

The **perceptron loss** over m examples {(x⁽ⁱ⁾, y⁽ⁱ⁾)} is

```
L(w,b) = (1/m) * Σ_{i=1..m} max(0, −y⁽ⁱ⁾ · zᵢ)
       = (1/m) * Σ_{i=1..m} ℓᵢ
where ℓᵢ = max(0, −y⁽ⁱ⁾ (w·x⁽ⁱ⁾ + b))
```

– if example i is correctly classified (yᵢ·zᵢ>0), ℓᵢ=0; otherwise ℓᵢ = −yᵢ·zᵢ.

### 2. Subgradient w.r.t. the Score zᵢ

Because ℓᵢ = max(0, uᵢ) with uᵢ = −yᵢ·zᵢ, its derivative wrt uᵢ is:

```
dℓᵢ/duᵢ =
  0       if uᵢ < 0    (correctly classified → no loss)
  undefined at uᵢ=0    (we pick any subgradient ∈ [0,1])
  1       if uᵢ > 0    (misclassified → linear penalty)
```

By chain‐rule,

```
dℓᵢ/dzᵢ = (dℓᵢ/duᵢ) * (duᵢ/dzᵢ)
        = { 0         if yᵢ·zᵢ > 0
          { 1 * (−yᵢ)  if yᵢ·zᵢ ≤ 0
        = { 0         if yᵢ·zᵢ > 0
          { −yᵢ       if yᵢ·zᵢ ≤ 0
```

### 3. Derivative of the Loss w.r.t. w and b

Using the fact that

```
∂zᵢ/∂w = x⁽ⁱ⁾     and     ∂zᵢ/∂b = 1,
```

the (sub)gradients of the average loss L(w,b) = (1/m) Σℓᵢ are

```
∂L/∂w = (1/m) * Σ_{i=1..m} (dℓᵢ/dzᵢ) · (∂zᵢ/∂w)
      = (1/m) * Σ_{i: yᵢ·zᵢ ≤ 0} (−yᵢ) · x⁽ⁱ⁾
      = −(1/m) * Σ_{i ∈ M} yᵢ · x⁽ⁱ⁾

∂L/∂b = (1/m) * Σ_{i=1..m} (dℓᵢ/dzᵢ) · (∂zᵢ/∂b)
      = (1/m) * Σ_{i: yᵢ·zᵢ ≤ 0} (−yᵢ) · 1
      = −(1/m) * Σ_{i ∈ M} yᵢ
```

Here **M** = { i | yᵢ·(w·x⁽ⁱ⁾+b) ≤ 0 } is the set of misclassified or boundary examples.

### 4. Compact Formula

```
L(w, b) = (1/m) * Σ max(0, −y⁽ⁱ⁾ (w·x⁽ⁱ⁾ + b))

∂L/∂w = −(1/m) * Σ_{i: y⁽ⁱ⁾ (w·x⁽ⁱ⁾+b) ≤ 0} y⁽ⁱ⁾ · x⁽ⁱ⁾

∂L/∂b = −(1/m) * Σ_{i: y⁽ⁱ⁾ (w·x⁽ⁱ⁾+b) ≤ 0} y⁽ⁱ⁾
```

Use these subgradients in a batch gradient‐descent step:

```
w_new = w_old − α · ∂L/∂w
b_new = b_old − α · ∂L/∂b
```

### 5. Python Snippet

```python
import numpy as np

def perceptron_gradients(w, b, X, y):
    """
    Compute perceptron loss subgradients for batch X,y.
    X: (m,n), y: (m,) with labels ±1
    Returns grad_w (n,), grad_b (scalar)
    """
    m = X.shape[0]
    scores  = X.dot(w) + b             # (m,)
    mask    = y * scores <= 0         # boolean (m,) of misclassified
    # if none misclassified, gradient = 0
    if not mask.any():
        return np.zeros_like(w), 0.0

    grad_w = -(1/m) * (y[mask, None] * X[mask]).sum(axis=0)
    grad_b = -(1/m) * y[mask].sum()
    return grad_w, grad_b

# Example usage:
w = np.zeros(2); b = 0.0
X = np.array([[2,1],[-1,-1],[1,-2]])
y = np.array([ 1,  -1,   1])
grad_w, grad_b = perceptron_gradients(w, b, X, y)
print("∂L/∂w =", grad_w, "∂L/∂b =", grad_b)
```

---

## Classification with a Neural Network

### 1. What You Should Already Know

- How a perceptron computes a weighted sum plus bias and applies an activation (sign or sigmoid).
- The idea of loss functions (perceptron loss, cross-entropy) and gradient descent updates.
- Chain rule for derivatives and backpropagation basics from single-layer models.

### 2. Conceptual Overview

A neural network (often called a multi-layer perceptron, MLP) stacks multiple layers of neurons to learn complex, non-linear decision boundaries.

- Each **layer** applies a linear transformation (weights · inputs + bias) followed by a non-linear **activation**.
- Hidden layers let the network carve out curved regions in feature space, far beyond the straight line of a single perceptron.
- The **output layer** turns the final activations into class scores or probabilities (e.g., via softmax for multi-class).

In practice, you feed raw inputs forward through each layer, compute a loss comparing predictions to true labels, then propagate error gradients backward through the network to update every weight and bias.

### 3. Network Architecture and Forward Pass

### 3.1 Two-Layer Network (One Hidden Layer)

```
Inputs x (n features)
   ↓  Linear transform + bias           Hidden layer activations a¹ (h units)
z¹ = W¹ · x + b¹                a¹ = g(z¹)
   ↓  Linear transform + bias           Output layer scores z² (k classes)
z² = W² · a¹ + b²                ŷ = softmax(z²)
```

- `W¹` is an h×n weight matrix, `b¹` is an h-vector.
- `g(z)` is a non-linear activation (e.g., ReLU or sigmoid).
- `W²` is a k×h matrix, `b²` is a k-vector, and `softmax` converts scores to probabilities:

```
softmax(z)ⱼ = exp(zⱼ) / sum_{ℓ=1..k} exp(zℓ)
```

### 3.2 Step-by-Step

1. Compute hidden pre-activations:
    
    ```
    z¹ = W¹ · x + b¹
    ```
    
2. Apply activation:
    
    ```
    a¹ = g(z¹)        // e.g., g(u)=max(0, u) for ReLU
    ```
    
3. Compute output scores:
    
    ```
    z² = W² · a¹ + b²
    ```
    
4. Convert to probabilities:
    
    ```
    ŷ = softmax(z²)
    ```
    

### 4. Loss Function: Categorical Cross-Entropy

For a single example with true one-hot label y (length k) and predicted probability ŷ:

```
L(ŷ, y) = - Σ_{j=1..k} yⱼ * log(ŷⱼ)
```

Over m examples, the average loss is:

```
J = (1/m) * Σ_{i=1..m} [ - Σ_{j=1..k} y⁽ⁱ⁾ⱼ * log(ŷ⁽ⁱ⁾ⱼ) ]
```

- If the true class is t, that reduces to `−log(ŷₜ)`.
- This loss penalizes confident mistakes heavily and is smooth for gradient backpropagation.

### 5. Backpropagation: Computing Gradients

We compute gradients layer by layer using the chain rule.

### 5.1 Output Layer Gradients

```
δ² = ŷ - y                  // shape (k,)
∂J/∂W² = (1/m) * δ² · (a¹)ᵀ   // shape (k×h)
∂J/∂b² = (1/m) * sum over batch of δ²   // shape (k,)
```

### 5.2 Hidden Layer Gradients

```
δ¹ = (W²ᵀ · δ²) * g′(z¹)     // shape (h,)
∂J/∂W¹ = (1/m) * δ¹ · xᵀ      // shape (h×n)
∂J/∂b¹ = (1/m) * sum over batch of δ¹   // shape (h,)
```

- `g′(z¹)` is the element-wise derivative of the activation (e.g., ReLU′=1 if z>0 else 0).
- We propagate the error `δ²` backward through the weights and scale by the activation slope.

### 6. Parameter Update Rule

Using a learning rate α, apply gradient descent to each weight matrix and bias vector:

```
W¹_new = W¹_old - α * ∂J/∂W¹
b¹_new = b¹_old - α * ∂J/∂b¹
W²_new = W²_old - α * ∂J/∂W²
b²_new = b²_old - α * ∂J/∂b²
```

Repeat forward pass, loss, backprop, update until convergence.

### 7. Python Example: Two-Layer Network from Scratch

```python
import numpy as np

# Activation and its derivative
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# Forward and backward passes
def forward_backward(X, y, W1, b1, W2, b2):
    m = X.shape[0]
    # Forward
    z1 = X.dot(W1.T) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2.T) + b2
    y_hat = softmax(z2)

    # Loss
    loss = -np.sum(y * np.log(y_hat + 1e-8)) / m

    # Backward
    delta2 = (y_hat - y) / m               # shape (m,k)
    dW2 = delta2.T.dot(a1)                # shape (k,h)
    db2 = delta2.sum(axis=0)              # shape (k,)

    delta1 = delta2.dot(W2) * relu_grad(z1)  # shape (m,h)
    dW1    = delta1.T.dot(X)                 # shape (h,n)
    db1    = delta1.sum(axis=0)              # shape (h,)

    return loss, (dW1, db1, dW2, db2)

# Training loop on toy data
np.random.seed(0)
m, n, h, k = 200, 2, 5, 3
X = np.random.randn(m, n)
y_int = np.random.randint(k, size=m)
y = np.eye(k)[y_int]  # one-hot labels

# Initialize parameters
W1 = np.random.randn(h, n) * 0.1; b1 = np.zeros(h)
W2 = np.random.randn(k, h) * 0.1; b2 = np.zeros(k)
alpha, epochs = 0.5, 1000

for epoch in range(epochs):
    loss, grads = forward_backward(X, y, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = grads

    # Update
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss {loss:.4f}")

# Evaluate accuracy
probs = softmax(relu(X.dot(W1.T) + b1).dot(W2.T) + b2)
preds = np.argmax(probs, axis=1)
acc = (preds == y_int).mean()
print("Training accuracy:", acc)
```

### 8. Practice Problems

1. **Vary hidden size**: Change `h` to 10, 50 and observe impact on training loss and accuracy.
2. **Different activations**: Replace `relu` with `sigmoid` and adjust `sigmoid_grad`. Compare convergence.
3. **Mini-batch training**: Split X,y into batches of size 32 and update per batch. Track loss curve.
4. **Apply to real data**: Use the Iris dataset’s first two features and three classes. Plot decision regions.

### 9. Visual and Intuitive Insights

- **Decision boundaries**: Hidden units carve out piecewise-linear regions; the output layer mixes them by softmax.
- **Loss landscape**: Deep models have high-dimensional, non-convex surfaces—initialization and learning rate matter.
- **Backprop engine**: Frameworks like TensorFlow or PyTorch automate these derivative chains, but the core math is identical.

---

## Classification with a Neural Network – Minimizing Log‐Loss

### 1. What You Already Know

You’ve seen how a network makes predictions via forward passes and how it applies softmax to turn scores into class probabilities【Classification with Neural Network】.

You understand cross‐entropy as a loss that penalizes confident mistakes heavily. Now we’ll dive deep into that loss—often called log‐loss—and how to minimize it.

### 2. Conceptual Overview: Why Log‐Loss?

Log‐loss (categorical cross‐entropy) measures the distance between true labels and predicted probabilities.

- If the model is certain and correct (probability near 1), loss → 0.
- If it’s certain and wrong (probability near 0), loss → ∞ (huge penalty).

This drives the network to push correct‐class scores up and wrong‐class scores down.

### 3. The Log‐Loss Formula

For one example with true one‐hot label y and predicted probability ŷ:

```
L(ŷ, y) = − Σ_{j=1..k} yⱼ * log(ŷⱼ)
```

Over m examples, the average loss is:

```
J = (1/m) * Σ_{i=1..m} [ − Σ_{j=1..k} y⁽ⁱ⁾ⱼ * log(ŷ⁽ⁱ⁾ⱼ) ]
```

### 4. Breaking Down the Formula

- `ŷⱼ`: predicted probability of class j (from softmax).
- `yⱼ`: 1 if the true class is j, else 0.
- `yⱼ·log(ŷⱼ)` picks out the log‐probability of the true class.
- The outer negative sign and sum average the penalty over m samples.

### 5. Loss Surface and Geometric Intuition

- In score‐space, each example’s contribution is a smooth, convex “pit” over the correct class axis.
- Summing across samples creates a high‐dimensional convex bowl in parameter space (for a single softmax layer).
- Gradient descent “rolls” this surface downhill toward its unique global minimum.

### 6. Gradients and Backpropagation

### 6.1 Derivative w.r.t. Output Scores

For one example i:

```
δ²⁽ⁱ⁾ = ŷ⁽ⁱ⁾ − y⁽ⁱ⁾    // shape (k,)
```

This is the gradient of L w.r.t. the pre‐softmax scores z².

### 6.2 Gradients through the Network

- Hidden layer error:
    
    ```
    δ¹ = (W²ᵀ · δ²) * g′(z¹)
    ```
    
- Weight gradients:
    
    ```
    ∂J/∂W² = (1/m) * δ² · (a¹)ᵀ
    ∂J/∂W¹ = (1/m) * δ¹ · xᵀ
    ```
    
- Bias gradients:
    
    ```
    ∂J/∂b² = (1/m) * Σ δ²
    ∂J/∂b¹ = (1/m) * Σ δ¹
    ```
    

### 7. Update Rules with Gradient Descent

With learning rate α, for each layer:

```
W_new = W_old − α * ∂J/∂W
b_new = b_old − α * ∂J/∂b
```

Repeat forward pass, compute J, backprop to get gradients, then update.

### 8. Python Example: Training on Log‐Loss

```python
import numpy as np

def softmax(z):
    ex = np.exp(z - np.max(z, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

# One‐hot encode labels
def to_one_hot(y, k): return np.eye(k)[y]

# Forward and backward for one hidden layer
def forward_backward(X, y, W1,b1, W2,b2):
    m = X.shape[0]
    # Forward
    z1 = X.dot(W1.T) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2.T) + b2
    y_hat = softmax(z2)
    # Loss
    loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
    # Backprop
    delta2 = (y_hat - y) / m
    dW2 = delta2.T.dot(a1)
    db2 = delta2.sum(axis=0)
    delta1 = delta2.dot(W2) * relu_grad(z1)
    dW1 = delta1.T.dot(X)
    db1 = delta1.sum(axis=0)
    return loss, (dW1,db1,dW2,db2)

# Toy data
np.random.seed(1)
m, n, h, k = 200, 2, 5, 3
X = np.random.randn(m, n)
y_int = np.random.randint(k, size=m)
y = to_one_hot(y_int, k)

# Initialize
W1 = np.random.randn(h,n)*0.1; b1 = np.zeros(h)
W2 = np.random.randn(k,h)*0.1; b2 = np.zeros(k)
alpha, epochs = 0.5, 500

for epoch in range(epochs):
    loss, grads = forward_backward(X, y, W1,b1, W2,b2)
    dW1,db1,dW2,db2 = grads
    W1 -= alpha * dW1; b1 -= alpha * db1
    W2 -= alpha * dW2; b2 -= alpha * db2
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss:.4f}")

# Final accuracy
probs = softmax(relu(X.dot(W1.T)+b1).dot(W2.T)+b2)
preds = np.argmax(probs, axis=1)
print("Accuracy:", (preds==y_int).mean())
```

### 9. Practice Problems

1. By hand, compute L and δ² for a two‐class example with scores z²=[2,−1] and true label y=[1,0].
2. Implement batch vs. stochastic gradient descent on log‐loss and plot loss curves.
3. Compare convergence using ReLU vs. sigmoid activations in the hidden layer.
4. Apply this to the Iris dataset (3‐class) and visualize decision regions.

### 10. Real‐World Applications

- Image classifiers assign pixel inputs to object categories.
- Language models predict next‐word probabilities.
- Medical diagnoses output disease likelihoods for patient features.

Minimizing log‐loss with backprop is the core of nearly every modern deep‐learning classifier.

---

## Gradient Descent and Backpropagation

### 1. Prerequisites

Before tackling gradient descent and backpropagation, ensure you are comfortable with:

- First‐order partial derivatives and gradients for scalar functions 【Gradients】
- The chain rule for single‐variable and multivariable functions
- A basic feed-forward neural network architecture (layers of weighted sums + activations)【Classification with Neural Network】

### 2. Conceptual Overview

**Gradient descent** is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function by moving in the direction of steepest descent (negative gradient).

**Backpropagation** is the method we use to efficiently compute those gradients in a layered neural network by applying the chain rule backward through the network.

Together they form the core training loop for deep models:

1. **Forward pass**: compute predictions and loss.
2. **Backward pass** (backprop): compute gradients of loss w.r.t. each weight and bias.
3. **Parameter update**: apply gradient descent to adjust every parameter.
4. **Repeat** until convergence.

### 3. Formal Definitions

### 3.1 Gradient Descent Update Rule

For any parameter vector θ and loss function (J(θ)), gradient descent updates:

```
θ_new = θ_old - α * ∇J(θ_old)
```

- `θ_old`: current parameters
- `∇J(θ_old)`: gradient of loss wrt parameters
- `α`: learning rate (step size)
- `θ_new`: updated parameters after one step

### 3.2 Backpropagation via Chain Rule

In a network of L layers, each layer ℓ computes:

```
z⁽ˡ⁾ = W⁽ˡ⁾·a⁽ˡ⁻¹⁾ + b⁽ˡ⁾      // linear step
a⁽ˡ⁾ = g⁽ˡ⁾(z⁽ˡ⁾)               // activation
```

- `a⁽⁰⁾ = x` (input)
- `a⁽ᴸ⁾ = ŷ` (output probabilities via softmax)

Backprop computes **error signals** δ at each layer, starting from the output:

```
δ⁽ᴸ⁾ = ∂J/∂z⁽ᴸ⁾      // gradient of loss wrt output pre-activations
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ · δ⁽ˡ⁺¹⁾  *  g⁽ˡ⁾′(z⁽ˡ⁾)
```

Then gradients of weights and biases are:

```
∂J/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
∂J/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

### 4. Detailed Math and Formulas

### 4.1 Forward Pass Equations

For layer ℓ:

```
z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = g⁽ˡ⁾(z⁽ˡ⁾)
```

At the output layer L with softmax and one-hot label y:

```
ŷ   = softmax(z⁽ᴸ⁾)
J   = -(1/m) * Σ_{i=1..m} Σ_{j=1..k} y⁽ⁱ⁾_j * log(ŷ⁽ⁱ⁾_j)
```

### 4.2 Backpropagation Steps

1. **Output layer δ**
    
    For each example i, component j:
    
    ```
    δ⁽ᴸ⁾_j = ∂J/∂z⁽ᴸ⁾_j = ŷ_j - y_j
    ```
    
2. **Hidden layer δ**
    
    For ℓ = L−1…1:
    
    ```
    δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ · δ⁽ˡ⁺¹⁾  elementwise-multiplied by g⁽ˡ⁾′(z⁽ˡ⁾)
    ```
    
3. **Gradients**
    
    ```
    ∂J/∂W⁽ˡ⁾ = (1/m) * δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
    ∂J/∂b⁽ˡ⁾ = (1/m) * Σ_{i=1..m} δ⁽ˡ⁾(i)
    ```
    
4. **Parameter updates**
    
    ```
    W⁽ˡ⁾ ← W⁽ˡ⁾ − α * ∂J/∂W⁽ˡ⁾
    b⁽ˡ⁾ ← b⁽ˡ⁾ − α * ∂J/∂b⁽ˡ⁾
    ```
    

### 5. Real-World ML/DS Examples

- **Image classification**: thousands of parameters in convolutional layers learn features via backprop.
- **Language models**: recurrent or transformer networks update embeddings and attention weights with gradient descent.
- **Autoencoders**: encoder and decoder weights trained to minimize reconstruction log-loss or MSE.

### 6. Python Implementation Example

Below is a scratch two-layer network with ReLU and softmax, training on a toy dataset:

```python
import numpy as np

# Activations and derivatives
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

def softmax(z):
    ex = np.exp(z - np.max(z, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

# Forward and backprop
def forward_backward(X, y, W1,b1, W2,b2):
    m = X.shape[0]
    # Forward pass
    z1 = X.dot(W1.T) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2.T) + b2
    y_hat = softmax(z2)
    # Compute loss
    loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
    # Backprop output layer
    delta2 = (y_hat - y) / m                 # shape (m,k)
    dW2    = delta2.T.dot(a1)               # shape (k,h)
    db2    = delta2.sum(axis=0)             # shape (k,)
    # Backprop hidden layer
    delta1 = delta2.dot(W2) * relu_grad(z1)  # shape (m,h)
    dW1    = delta1.T.dot(X)                # shape (h,n)
    db1    = delta1.sum(axis=0)             # shape (h,)
    return loss, (dW1, db1, dW2, db2)

# Toy data
np.random.seed(0)
m, n, h, k = 100, 2, 10, 3
X = np.random.randn(m, n)
y_int = np.random.randint(k, size=m)
y = np.eye(k)[y_int]

# Initialize
W1 = np.random.randn(h, n) * 0.1; b1 = np.zeros(h)
W2 = np.random.randn(k, h) * 0.1; b2 = np.zeros(k)
alpha, epochs = 0.1, 500

for epoch in range(epochs):
    loss, grads = forward_backward(X, y, W1,b1, W2,b2)
    dW1,db1,dW2,db2 = grads
    # Update
    W1 -= alpha * dW1; b1 -= alpha * db1
    W2 -= alpha * dW2; b2 -= alpha * db2
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss:.4f}")

# Final accuracy
probs = softmax(relu(X.dot(W1.T)+b1).dot(W2.T)+b2)
preds = np.argmax(probs, axis=1)
print("Training accuracy:", (preds==y_int).mean())
```

### 7. Practice Problems

1. Extend this network to **three hidden layers** and verify that backprop still works.
2. Replace ReLU with **sigmoid** activation; derive and implement its derivative.
3. Implement **mini-batch** gradient descent (batch size 32) and compare convergence speed.
4. Apply the same code to the **Iris** or **MNIST** dataset and plot loss vs. epoch.

### 8. Geometric and Intuitive Interpretation

- **Gradient descent** follows the negative gradient vector downhill on a high-dimensional loss surface.
- **Backpropagation** efficiently computes how a small change in any weight influences the final loss by propagating error signals backward through the network’s computational graph.

---

## Optimization in Neural Networks – Newton’s Method

### 1. Prerequisites

Before exploring Newton’s method in neural nets, you should already understand:

- How to compute the **gradient** of a loss via backpropagation【Classification with Neural Network】
- The **Hessian** matrix of second derivatives for scalar functions【Partial Derivatives – Part 2】
- Gradient descent parameter updates

### 2. Conceptual Overview

Newton’s method uses both first (gradient) and second (curvature) information to jump toward a loss minimum faster than gradient descent.

- Instead of stepping a fixed fraction of the gradient, you scale that step by the inverse curvature.
- In a high-dimensional network, the Hessian encodes how each weight’s slope is changing with every other weight.
- The ideal update iswhere
    
    ```
    θ_new = θ_old − [H(θ_old)]⁻¹ · ∇J(θ_old)
    ```
    
    - θ = all network parameters (weights and biases) flattened into a vector
    - ∇J = gradient of loss
    - H = Hessian of loss

### 3. Newton’s Update Rule for Network Parameters

```
θ_new = θ_old − H(θ_old)⁻¹ · ∇J(θ_old)
```

- **θ_old**: current parameter vector
- **∇J(θ_old)**: gradient computed via backprop
- **H(θ_old)**: matrix of second derivatives, size (P×P) if you have P parameters
- **θ_new**: next iterate

**Step‐by‐step**:

1. Run a forward pass to compute loss J.
2. Backpropagate to get gradient ∇J.
3. Compute or approximate H.
4. Solve the linear system H·∆θ = ∇J for ∆θ.
5. Update θ ← θ − ∆θ.

### 4. Computing the Hessian

### 4.1 Hessian in One‐Layer Logistic Regression

For logistic regression (no hidden layers), X is m×n, w is n×1, bias b:

```
J(w,b) = -(1/m) Σ [ yᵢ·ln(pᵢ) + (1−yᵢ)·ln(1−pᵢ) ]
pᵢ     = sigmoid(w·xᵢ + b)
```

Let θ = [b; w], stack X with a column of ones to X_aug:

```
S = diag(pᵢ·(1−pᵢ))            // m×m

H = (1/m) * X_augᵀ · S · X_aug  // (n+1)×(n+1)
∇J = (1/m) * X_augᵀ · (p − y)   // (n+1)×1
θ_new = θ_old − H⁻¹ · ∇J
```

### 4.2 Hessian in Multi‐Layer Networks (Gauss–Newton)

For deep nets, the full Hessian is huge and expensive to invert. Common strategies:

- **Gauss–Newton approximation**: replace Hessian with JᵀJ, where J is the network’s Jacobian of outputs wrt parameters.
- **Quasi‐Newton methods** (BFGS, L‐BFGS): build low-rank approximations of H over iterations.
- **Hessian-vector products** via Pearlmutter’s trick, allowing Conjugate Gradient solves without forming H explicitly.

### 5. Worked Example: Newton’s Method for Logistic Regression

1. **Data and model**
    
    ```
    X_aug = [1 xᵢ]   // m×(n+1)
    θ = [b; w]      // (n+1)×1
    p = sigmoid(X_aug·θ)
    ```
    
2. **Compute gradient**
    
    ```
    ∇J = (1/m) * X_augᵀ · (p − y)
    ```
    
3. **Compute Hessian**
    
    ```
    S = diag(p · (1−p))
    H = (1/m) * X_augᵀ · S · X_aug
    ```
    
4. **Newton update**
    
    ```
    θ_new = θ_old − H⁻¹ · ∇J
    ```
    
5. **Iterate** until ∥∇J∥ is small.

### 6. Python Implementation

```python
import numpy as np

# 1. Simulate logistic data
np.random.seed(0)
m, n = 100, 2
X = np.random.randn(m, n)
X_aug = np.hstack([np.ones((m,1)), X])  # prepend bias
true_theta = np.array([0.5, 2.0, -1.0])  # [b, w1, w2]
z = X_aug.dot(true_theta)
y = (1 / (1 + np.exp(-z)) > 0.5).astype(float)

# 2. Sigmoid
sigmoid = lambda u: 1 / (1 + np.exp(-u))

# 3. Newton’s method loop
theta = np.zeros(n+1)
for i in range(10):
    z     = X_aug.dot(theta)
    p     = sigmoid(z)
    grad  = (1/m) * X_aug.T.dot(p - y)               # (n+1,)
    S     = np.diag(p * (1-p))                       # m×m
    H     = (1/m) * X_aug.T.dot(S).dot(X_aug)         # (n+1)×(n+1)
    delta = np.linalg.solve(H, grad)                 # solve H·∆=grad
    theta -= delta
    loss   = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
    print(f"Iter {i}, Loss = {loss:.4f}")

print("Estimated θ:", theta)
```

### 7. Practice Problems

1. **By hand**, derive H and ∇J for a single‐feature logistic model and perform one Newton update from θ₀=0.
2. **Implement** Newton’s method for the simple two‐layer network inClassification with Neural Network. Approximate H with Gauss–Newton.
3. **Compare**: run gradient descent vs Newton’s method on the logistic data above; plot loss vs iteration.
4. **Quasi‐Newton**: use SciPy’s `optimize.minimize(method='BFGS')` on the same loss and compare to your Newton solver.

### 8. Geometric Intuition

- **Gradient descent** steps move in the negative gradient direction with a fixed step size—like walking downhill with a fixed stride.
- **Newton’s method** measures the slope and curvature: it finds where the local quadratic approximation (tangent + curvature) has its minimum and jumps there directly.
- Near a convex minimum, Newton converges **quadratically** (very fast), but far from it or on non-convex surfaces it can diverge or get stuck.

### 9. Real‐World Applications

- **Small to medium-scale models** (e.g., logistic regression, shallow autoencoders) where exact Hessian inversion is feasible.
- **Bayesian neural networks**: Laplace approximation around a mode uses the inverse Hessian as a covariance estimate.
- **Second‐order optimizers** in deep learning libraries often use limited‐memory BFGS as an efficient appeal to Newton’s ideas.

---

## The Second Derivative

### 1. What You Need to Know First

You should already understand:

- The concept of a first derivative as the instantaneous slope of a function f(x):
    
    ```
    f′(x) = d/dx [ f(x) ]
    ```
    
- How to compute basic single‐variable derivatives (power, product, chain rules).

### 2. Conceptual Overview

The **second derivative** measures how the *first derivative* itself changes as x moves.

- If the first derivative f′(x) is the *slope* of the curve y=f(x), then the second derivative f″(x) is the *curvature* or *concavity*—how sharply that slope is rising or falling.
- In physical terms, if f(x) is position over time, f′(x) is velocity and f″(x) is acceleration.

### 3. Formal Definition and Formula

The second derivative is the derivative of the first derivative:

```
f″(x) = d²/dx² [ f(x) ] = d/dx [ f′(x) ]
```

Broken into two steps:

1. Compute the first derivative
    
    ```
    f′(x) = d/dx [ f(x) ]
    ```
    
2. Differentiate that result
    
    ```
    f″(x) = d/dx [ f′(x) ]
    ```
    

### 4. Breaking Down the Formula

1. **d²/dx²** notation indicates “take two derivatives w.r.t. x.”
2. **f′(x)** is itself a function of x; differentiating it measures its rate of change.
3. If f(x) = x³, then:
    - f′(x) = 3x²
    - f″(x) = d/dx [3x²] = 6x

### 5. Geometric Interpretation

- **f″(x) > 0**: the curve is *concave up* (cup-shaped)—slopes are increasing; the graph looks like a valley.
- **f″(x) < 0**: the curve is *concave down* (cap-shaped)—slopes are decreasing; the graph looks like a hill.
- **f″(x) = 0**: an inflection point—concavity switches sign.

Plotting f, f′, and f″ together shows how curvature relates to slope.

### 6. Role in Data Science and Machine Learning

- **Optimization tests**: In one‐dimensional problems, f′(x)=0 locates stationary points; f″(x) determines maxima or minima.
- **Hessian matrix**: For multivariable f, the Hessian generalizes second derivatives and drives Newton’s method updates.
- **Model curvature**: In logistic or neural‐network training, curvature information can speed up convergence (e.g., second‐order optimizers).

### 7. Worked Examples

### 7.1 Polynomial Example

Let

```
f(x) = x^4 - 2x^2 + 3x
```

1. First derivative:
    
    ```
    f′(x) = 4x^3 - 4x + 3
    ```
    
2. Second derivative:
    
    ```
    f″(x) = 12x^2 - 4
    ```
    

Test at x=1:

- f′(1)=4−4+3=3 (positive slope)
- f″(1)=12−4=8>0 (concave up → local minimum if f′=0)

### 7.2 Inflection Point

For

```
f(x) = x^3
```

- f′(x)=3x²
- f″(x)=6x
- f″(x)=0 at x=0 → inflection where concavity switches from down (x<0) to up (x>0).

### 8. Python Exercises

```python
import numpy as np

# Function and analytic second derivative
f  = lambda x: x**4 - 2*x**2 + 3*x
f2 = lambda x: 12*x**2 - 4

# Numeric second derivative via central differences
def numeric_second_derivative(f, x, h=1e-5):
    return (f(x+h) - 2*f(x) + f(x-h)) / h**2

# Test at several points
for x0 in [-2, 0, 1]:
    print(f"x={x0}: Analytic f″={f2(x0):.4f}, Numeric f″={numeric_second_derivative(f, x0):.4f}")
```

Plot f and its second derivative to visualize concavity:

```python
import matplotlib.pyplot as plt

xs = np.linspace(-2, 2, 400)
plt.plot(xs, f(xs), label='f(x)')
plt.plot(xs, f2(xs), label="f″(x)")
plt.axhline(0, color='gray', linestyle='--')
plt.legend(); plt.show()
```

### 9. Practice Problems

1. Compute by hand f″(x) for
    
    ```
    f(x) = e^(2x) * sin(x)
    ```
    
2. Identify and classify stationary points forusing f′ and f″.
    
    ```
    f(x) = x^3 - 6x^2 + 9x
    ```
    
3. Use Python to find where f″(x) changes sign for f(x)=x^5−5x^3+4x.
4. In SciPy, use `scipy.misc.derivative` to numerically compute second derivatives of an arbitrary function.

---

## The Hessian Matrix

### 1. Prerequisites

You should already understand:

- First‐order partial derivatives (∂f/∂x, ∂f/∂y)
- Second‐order partials (fₓₓ, fₓᵧ, fᵧₓ, fᵧᵧ)
- How the second derivative f″(x) measures curvature for single‐variable functions

### 2. Conceptual Overview

When you move from one variable to many, **curvature** has multiple directions. The Hessian collects all second‐order partials into a single matrix that describes how the slope changes in every axis direction and how axes interact.

- It tells you whether a point is a hill, a valley, or a saddle in multi‐dimensional space.
- In optimization, it guides Newton’s method and reveals the local shape of the loss surface.

### 3. Definition and Formula

For a twice‐differentiable function

```
f(x₁, x₂, …, xₙ)
```

the **Hessian matrix** H is the n×n matrix of all second partial derivatives:

```
H(f)(x) = [ ∂²f/∂xᵢ∂xⱼ ]  for i, j = 1..n

Example for n=2:

H(f)(x,y) = | fₓₓ(x,y)  fₓᵧ(x,y) |
            | fᵧₓ(x,y)  fᵧᵧ(x,y) |
```

**Step‐by‐step**:

1. Compute each first partial: fₓ = ∂f/∂x, fᵧ = ∂f/∂y.
2. Differentiate fₓ with respect to x → fₓₓ, and with respect to y → fₓᵧ.
3. Differentiate fᵧ with respect to x → fᵧₓ, and with respect to y → fᵧᵧ.
4. Assemble into the matrix above.

### 4. Interpreting the Hessian

- **Eigenvalues** of H at a point tell curvature signs in principle directions.
    - All positive → local minimum (valley)
    - All negative → local maximum (peak)
    - Mixed signs → saddle point
- **Symmetry**: Under continuity, fₓᵧ = fᵧₓ, so H is symmetric and has real eigenvalues.

### 5. Real‐World ML/DS Applications

- **Newton’s method**: update θ ← θ − H⁻¹ · ∇f for faster local convergence.
- **Laplace approximation** in Bayesian inference: use H⁻¹ as a covariance around the mode.
- **Curvature‐aware optimization**: trust‐region methods (Levenberg–Marquardt) adapt steps based on H.

### 6. Worked Example

Let

```
f(x, y) = x²y + 3xy² − sin(xy)
```

1. First partials:
    
    ```
    fₓ = 2x·y + 3y² − y·cos(xy)
    fᵧ = x² + 6x·y − x·cos(xy)
    ```
    
2. Second partials:
    
    ```
    fₓₓ = 2y + y²·sin(xy)
    fₓᵧ = 2x + 6y − (cos(xy) − xy·sin(xy))
    fᵧₓ = fₓᵧ      // by Clairaut’s theorem
    fᵧᵧ = 6x + x²·sin(xy)
    ```
    
3. Hessian:
    
    ```
    H = | 2y + y²·sin(xy)      2x + 6y − cos(xy) + x y·sin(xy) |
        | 2x + 6y − cos(xy) + x y·sin(xy)   6x + x²·sin(xy)    |
    ```
    

### 7. Python Implementation

```python
import numpy as np

def f_hessian(x, y):
    # Compute second partials
    f_xx = 2*y + (y**2)*np.sin(x*y)
    f_xy = 2*x + 6*y - np.cos(x*y) + x*y*np.sin(x*y)
    f_yy = 6*x + (x**2)*np.sin(x*y)
    # Hessian matrix
    return np.array([[f_xx, f_xy],
                     [f_xy, f_yy]])

# Evaluate at (1, 2)
H = f_hessian(1.0, 2.0)
print("Hessian at (1,2):\n", H)
# Eigenvalues to classify curvature
eigs = np.linalg.eigvals(H)
print("Eigenvalues:", eigs)
```

### 8. Practice Problems

1. By hand, compute the Hessian of
    
    ```
    f(x,y) = x³ + y³ - 3xy
    ```
    
2. Write a Python function to compute and return the Hessian for
    
    ```
    f(x,y,z) = x² + 2y² + 3z² + xy - yz + zx
    ```
    
3. Use the Hessian eigenvalues at a critical point to determine if it’s a min, max, or saddle.
4. Implement Newton’s method for a two‐variable function using the Hessian and compare to gradient descent.

### 9. Geometric Visualization

Imagine a 3D surface z=f(x,y). At a point, the Hessian tells you:

- How fast the surface curves **along x** (fₓₓ).
- How fast it curves **along y** (fᵧᵧ).
- How much the curvature in x and y **interacts** (fₓᵧ).

A positive definite Hessian makes the surface look like a “bowl” locally.

---

## Hessians and Concavity

### 1. What You Should Already Know

You should be familiar with:

- First‐order partial derivatives (∂f/∂x, ∂f/∂y)
- The Hessian matrix (collection of second‐order partials)
- How the single‐variable second derivative f″(x) tells concavity (cup vs. cap)

### 2. Conceptual Overview

In one dimension, concavity is about whether a curve is cup‐shaped (concave up) or cap‐shaped (concave down).

In multiple dimensions, the **Hessian** tells us the curvature in every direction at a point. By examining the Hessian’s sign pattern, we can tell if a surface is:

- A **valley** (concave up in all directions)
- A **peak** (concave down in all directions)
- A **saddle** (mix of up and down)

Concavity matters in optimization: convex (valley) regions guarantee a unique minimum.

### 3. The Hessian as a Curvature Matrix

For a twice‐differentiable function

```
f(x₁, x₂, …, xₙ)
```

the Hessian H is

```
H(f)(x) = [ ∂²f/∂xᵢ∂xⱼ ]  for i,j = 1..n
```

Example for n=2:

```
H(f)(x,y) = | fₓₓ  fₓᵧ |
            | fᵧₓ  fᵧᵧ |
```

Each entry captures how the slope in one direction changes when you move along another.

### 4. Definite Hessians and Concavity

The Hessian’s **definiteness** classifies concavity:

- **Positive definite** (all eigenvalues > 0)→ f is locally **convex** (cup‐shaped) ⇒ local minimum
- **Negative definite** (all eigenvalues < 0)→ f is locally **concave** (cap‐shaped) ⇒ local maximum
- **Indefinite** (mixed eigenvalues)→ f has a **saddle point** (up in some directions, down in others)
- **Semi‐definite** (some eigenvalues zero)→ flat directions; test is inconclusive

### 5. Checking Definiteness: Sylvester’s Criterion

For small n, you can check principal minors:

- n=2: compute
    
    ```
    D1 = fₓₓ
    D2 = det(H) = fₓₓ·fᵧᵧ − (fₓᵧ)²
    ```
    
    - If D1>0 and D2>0 ⇒ positive definite
    - If D1<0 and D2>0 ⇒ negative definite
    - If D2<0 ⇒ indefinite
- Larger n: check all leading principal minors or compute eigenvalues numerically.

### 6. Real‐World ML/DS Examples

1. **Loss surface analysis**
    
    A convex loss (positive‐definite Hessian everywhere) ensures gradient descent finds the unique global minimum.
    
2. **Newton’s method**
    
    Uses H⁻¹ to scale gradients. If H is indefinite, the update can head toward a saddle or maximum.
    
3. **Uncertainty estimation**
    
    In Bayesian models, the Hessian at a mode approximates the inverse covariance of parameter estimates.
    

### 7. Python Implementation

```python
import numpy as np

def hessian_2d(f_xx, f_xy, f_yy, x, y):
    """
    Constructs the Hessian matrix for a function at (x,y)
    given functions for second partials.
    """
    H = np.array([[f_xx(x,y), f_xy(x,y)],
                  [f_xy(x,y), f_yy(x,y)]])
    return H

# Example: f(x,y) = x^2 + 3xy + 2y^2
f_xx = lambda x,y: 2
f_xy = lambda x,y: 3
f_yy = lambda x,y: 4

H = hessian_2d(f_xx, f_xy, f_yy, 0, 0)
eigs = np.linalg.eigvals(H)
print("Hessian:\n", H)
print("Eigenvalues:", eigs)
# Check definiteness
print("Positive definite?" , np.all(eigs > 0))
```

This example builds a constant‐curvature surface; its Hessian eigenvalues (2±3±4) tell you concavity.

### 8. Practice Problems

1. By hand, compute and classify the Hessian of
    
    ```
    f(x,y) = x^3 + y^3 - 3xy
    ```
    
    at the point (1,1).
    
2. Write a Python function to:
    - Symbolically or numerically compute all second partials of
        
        ```
        f(x,y) = sin(x)·e^y + x*y^2
        ```
        
    - Evaluate its Hessian at (0,0) and check definiteness.
3. For
    
    ```
    f(x,y,z) = x^2 + 2y^2 - z^2 + xy - yz + zx
    ```
    
    implement code that returns its Hessian (3×3) and prints eigenvalues.
    
4. Using a convex quadratic form
    
    ```
    f(x,y) = ax^2 + bxy + cy^2
    ```
    
    derive conditions on a,b,c for positive definiteness.
    

### 9. Geometric Interpretation

- **Positive definite Hessian** looks like a smooth bowl around the point.
- **Negative definite Hessian** looks like an inverted bowl (peak).
- **Indefinite Hessian** looks like a saddle surface (uphill one way, downhill another).

Visualizing level curves helps:

- Elliptical contours for minima/maxima
- Hyperbolic contours for saddles

---

## Newton’s Method for Two Variables

### 1. Prerequisites

You should already understand:

- First‐order partial derivatives and the gradient ∇f(x,y) 【Gradients】.
- The Hessian matrix H(x,y) of second partials 【The Hessian Matrix】.
- Taylor’s theorem up to second order for single‐variable functions.

### 2. Conceptual Overview

Newton’s method uses a **local quadratic approximation** of your function f(x,y) to jump directly toward its stationary points (where ∇f=0):

- At each iterate ([x,y]), you build the second‐order Taylor expansion of f around that point.
- You find the minimizer of that quadratic model by solving a linear system involving the Hessian.
- If f is convex near the solution, these steps converge very quickly—often in just a few iterations.

### 3. Taylor Expansion in Two Variables

Around ((x₀,y₀)), the second‐order Taylor approximation of f is:

```
f(x, y) ≈ f(x₀,y₀)
  + fₓ(x₀,y₀)·(x−x₀)
  + fᵧ(x₀,y₀)·(y−y₀)
  + ½ [ fₓₓ(x₀,y₀)·(x−x₀)²
       +2·fₓᵧ(x₀,y₀)·(x−x₀)(y−y₀)
       + fᵧᵧ(x₀,y₀)·(y−y₀)² ]
```

- The linear terms involve the **gradient**.
- The quadratic terms involve the **Hessian**.

### 4. Newton’s Update Rule

By minimizing that quadratic model, the Newton step solves:

```
H(x₀,y₀) · [ Δx
            Δy ] = ∇f(x₀,y₀)
```

and updates:

```
x_new = x₀ − Δx
y_new = y₀ − Δy
```

In compact vector form:

```
θ_old = [ x₀
          y₀ ]

∇f   = [ fₓ
         fᵧ ]

H    = [ fₓₓ  fₓᵧ
         fₓᵧ  fᵧᵧ ]

θ_new = θ_old − H⁻¹ · ∇f
```

### 5. Step‐by‐Step Procedure

1. **Start** from an initial guess ((x₀,y₀)).
2. **Compute** the gradient (\nabla f) and Hessian (H) at ((x₀,y₀)).
3. **Solve** the linear system (H·Δθ = ∇f) for the step (Δθ = [Δx, Δy]).
4. **Update**:
    
    ```
    [x₁, y₁]ᵀ = [x₀, y₀]ᵀ − [Δx, Δy]ᵀ
    ```
    
5. **Repeat** until (|Δθ|) or (|\nabla f|) is below a chosen tolerance.

### 6. Worked Example

Minimize the convex quadratic

```
f(x,y) = (x − 2)² + (y + 3)²
```

1. **Gradient**:
    
    ```
    fₓ = 2·(x − 2)
    fᵧ = 2·(y + 3)
    ```
    
2. **Hessian** (constant):
    
    ```
    H = | 2  0 |
        | 0  2 |
    ```
    
3. **Newton step** solvesso
    
    ```
    [2 0; 0 2]·[Δx; Δy] = [2(x−2); 2(y+3)]
    ```
    
    ```
    Δx = (2(x−2)) / 2 = x − 2
    Δy = (2(y+3)) / 2 = y + 3
    ```
    
4. **Update**:You converge to ((2,−3)) in **one iteration**.
    
    ```
    x_new = x − (x − 2) = 2
    y_new = y − (y + 3) = −3
    ```
    

### 7. Python Implementation

```python
import numpy as np

# 1. Define f, gradient, Hessian
def f_grad(xy):
    x, y = xy
    return np.array([2*(x-2), 2*(y+3)])

def f_hess(xy):
    # constant Hessian
    return np.array([[2.0, 0.0],
                     [0.0, 2.0]])

# 2. Newton’s method
xy = np.array([0.0, 0.0])  # initial guess
for i in range(5):
    grad = f_grad(xy)
    H    = f_hess(xy)
    step = np.linalg.solve(H, grad)
    xy  -= step
    print(f"Iter {i}: xy = {xy}, |grad| = {np.linalg.norm(grad):.4f}")

print("Converged to", xy)
```

### 8. Practice Problems

1. Use Newton’s method to find the minimum of
    
    ```
    f(x,y) = x^3 + y^3 - 3x - 6y + 10
    ```
    
    starting from (0,0). Compute two iterations by hand.
    
2. Implement Newton’s method in Python for
    
    ```
    f(x,y) = x^2*y + y^2 - x
    ```
    
    and observe its convergence behavior for several starting points.
    
3. Compare convergence of gradient descent vs Newton’s method on the example
    
    ```
    f(x,y) = (x-1)^4 + (y+2)^4
    ```
    
    by coding both and plotting |gradient| vs iteration.
    
4. Solve a system of nonlinear equations via Newton’s method: find (x,y) satisfying
    
    ```
    { x^2 + y^2 = 5
    { x*y        = 1
    ```
    
    by minimizing g(x,y) = (x^2+y^2-5)^2 + (x*y-1)^2.
    

### 9. Geometric Intuition

- At each point, the Hessian defines a **local bowl** (or saddle) that best approximates f.
- Newton’s update jumps right to the bottom of that bowl.
- For non‐quadratic f, you “hop” around but rapidly zero in on the true minimum when the shape is locally convex.

### 10. Real‐World ML/DS Applications

- **Logistic regression**: the normal‐equation Newton update uses the Hessian of the log‐loss for exact parameter fitting.
- **Small neural networks**: Gauss–Newton or full Newton updates can accelerate convergence in shallow nets.
- **Bayesian inference**: Laplace’s method around a mode uses H⁻¹ as an approximation of posterior covariance.

---