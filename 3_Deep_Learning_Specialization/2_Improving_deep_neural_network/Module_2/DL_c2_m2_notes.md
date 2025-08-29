# DL_c2_m2

## Batch Gradient Descent

### 1. Concept intuition

Batch gradient descent is the process of updating model parameters by computing the gradient of the loss function over your entire training set at each step.

By looking at all examples before each update, you follow the true slope of the loss surface and slowly move toward the global minimum.

This approach contrasts with stochastic or mini-batch variants, trading off update frequency for a stable descent direction.

### 2. Mathematical breakdown

Let

- m be the number of training examples
- θ be the vector of parameters
- J(θ) be the cost (loss) function

Compute the cost over all examples:

```python
J(θ) = (1/m) * sum_{i=1..m} L(y^(i), ŷ^(i))
```

Here L is your per-example loss (e.g., squared error or cross-entropy).

The parameter update rule for batch gradient descent is:

```python
θ := θ - α * ∇J(θ)
```

where

- α is the learning rate
- ∇J(θ) is the gradient of J with respect to θ, computed over all m examples

Concretely, if J(θ) = (1/m) * sum (h_θ(x^(i)) - y^(i))^2, then

```python
gradient = (2/m) * X.T.dot(hθ(X) - y)
θ := θ - α * gradient
```

### 3. Code & practical application

Below is a NumPy implementation of batch gradient descent for linear regression on a toy dataset.

```python
import numpy as np

# Generate synthetic data
np.random.seed(0)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Add bias term
X_b = np.c_[np.ones((m, 1)), X]  # shape (m, 2)

# Hyperparameters
alpha = 0.1
n_iterations = 1000
theta = np.random.randn(2, 1)  # random initialization

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    return (1/m) * np.sum((predictions - y) ** 2)

for iteration in range(n_iterations):
    gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - alpha * gradients

final_cost = compute_cost(X_b, y, theta)
print("Estimated parameters:", theta.ravel())
print("Final cost:", final_cost)
```

This code

- builds X_b by concatenating a column of ones for the bias
- computes gradients over the full dataset
- updates θ in one shot per iteration

## 4. Visualization / Geometry

Imagine the loss surface as a bowl-shaped contour plot over θ₀ (bias) and θ₁ (weight):

```
  θ₀
   ^
   |
  / \       ← contour lines of constant J(θ)
 /   \
*     *    ← Starting θ at the rim
 \   /
  \ /
   *
   └────────> θ₁
```

At each step, batch gradient descent computes the exact steepest descent direction (the gradient) at your current (θ₀, θ₁) by looking at every training point. The update moves you straight “downhill” toward the global minimum at the center of the contours.

### 5. Common pitfalls & tips

- Choosing α too large can overshoot the minimum and diverge.
- Choosing α too small leads to very slow convergence.
- Computing the gradient over massive datasets can be slow or memory-intensive.
- Always shuffle or randomize your data ordering before splitting, though batch itself doesn’t sample subsets.
- Monitor cost over iterations; if it plateaus or oscillates, adjust α.

### 6. Practice

1. Implement batch gradient descent for logistic regression on a small binary classification dataset (e.g., sklearn’s `make_classification`).
    - Hint: your loss is cross-entropy:
        
        ```python
        J(θ) = -(1/m) * sum(y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z)))
        ```
        
    - Walkthrough: compute z = X_b.dot(theta), apply sigmoid, then gradients.
2. Plot cost vs. iteration for three different learning rates (0.01, 0.1, 1.0). Observe convergence behavior.
3. Use a contour plot to visualize how the parameters θ evolve over iterations for the linear regression example above.

---

## Mini-Batch Gradient Descent

### 1. Direct answer

Mini-batch gradient descent breaks your training set into small groups (batches) of examples and updates the parameters using the average gradient computed on each batch. It combines the stability of full-batch gradient descent with the speed and noise-induced exploration of stochastic gradient descent.

### 2. Why mini-batches matter

- faster per-update computation by leveraging matrix operations on GPUs
- smoother updates than pure stochastic (one-example) SGD
- more frequent updates than full batch—helps escape shallow local minima
- fits in memory when your dataset cannot be loaded all at once

### 3. Mathematical formulation

Let

- m be total number of examples
- b be the mini-batch size (e.g., 32, 64, 128)
- θ be parameters
- L the per-example loss

For each batch indexed by t:

1. select batch $(B_t\subset{1,\dots,m})$ of size b
2. compute cost on batch:$[ J_{B_t}(\theta) = \frac{1}{b}\sum_{i\in B_t}L\bigl(y^{(i)},\hat y^{(i)}\bigr) ]$
3. compute gradient on batch:$[ g_t = \nabla_\theta J_{B_t}(\theta) ]$
4. update parameters:$[ \theta := \theta - \alpha,g_t ]$

Because you only average over b examples, each gradient step is cheaper but noisier, injecting randomness that can help generalization.

### 4. NumPy implementation

```python
import numpy as np

# synthetic data
np.random.seed(42)
m = 500
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# add bias term
X_b = np.c_[np.ones((m, 1)), X]

# hyperparameters
learning_rate = 0.05
n_epochs = 50
batch_size = 64
n_batches = int(np.ceil(m / batch_size))

# initialize parameters
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    # shuffle indices each epoch
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_b_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # compute gradient on current mini-batch
        gradients = (2 / len(y_batch)) * X_batch.T.dot(X_batch.dot(theta) - y_batch)

        # parameter update
        theta -= learning_rate * gradients

final_cost = (1/m) * np.sum((X_b.dot(theta) - y) ** 2)
print("Learned theta:", theta.ravel())
print("Final cost:", final_cost)
```

### 5. Visualization / Geometric intuition

Imagine the loss surface contours like a landscape.

- full-batch descent computes the exact downhill direction at every step
- pure SGD bounces around a lot
- mini-batches give “noisy but smooth” steps that wiggle toward the valley faster and help jump over small bumps

### 6. Tips, pitfalls, and best practices

1. choose batch sizes that fit your hardware: powers of two often run fastest on GPUs
2. too small batches (e.g., ≤16) produce high variance updates and can slow convergence
3. too large batches (e.g., >1024) approach full-batch behavior and require more memory
4. pair mini-batches with learning-rate schedules (decay, warm-up) or adaptive optimizers (Adam)
5. always shuffle data each epoch to prevent cyclic patterns

### 7. Interview-ready insights

- explain runtime per epoch: (O(m,d)) remains same as batch, but memory and hardware utilization differ
- discuss trade-off between update noise and gradient accuracy
- mention how batch normalization and vectorized matrix operations depend on mini-batch structure
- articulate why modern frameworks default to mini-batches (hardware, convergence speed, generalization)

### 8. Practice exercises

1. implement mini-batch gradient descent for logistic regression on a binary dataset using cross-entropy loss
2. compare convergence curves for batch sizes [1, 32, 256, m] at a fixed learning rate
3. visualize parameter trajectories in a contour plot for two-parameter linear regression
4. integrate a simple learning-rate decay schedule and observe its effect on convergence

---

## Understanding Mini-Batch Gradient Descent

### 1. Direct definition

Mini-batch gradient descent splits your dataset into small batches of size b and, at each update step, computes the average gradient over one batch instead of the full dataset or a single example. This strikes a balance between the stability of full-batch descent and the speed (and exploration) of stochastic descent.

### 2. Why mini-batches matter

- They let you exploit fast matrix-matrix operations on modern hardware (GPUs/TPUs).
- They inject controlled noise into each update, which can help escape shallow minima and improve generalization.
- They require less memory per update than full-batch methods, making it possible to train large models on large datasets.
- They produce smoother convergence curves than purely stochastic updates.

### 3. Batch size, variance, and noise

When you pick a batch of size b:

- Your gradient estimate is`g = (1/b) * sum_{i in batch} ∇θ L(y⁽ⁱ⁾, ŷ⁽ⁱ⁾; θ)`
- If b = 1, you get the noisiest estimate (highest variance).
- If b = m (the full dataset), variance is zero, but each step is expensive.
- In practice, b between 32 and 256 often gives a good trade-off: variance low enough to converge quickly, with frequent updates.

### 4. Convergence behavior

- **Bias vs. variance trade-off**Small b → low bias in gradient direction (you still head downhill) but high variance → more zig-zag.Large b → low variance but slower updates and risk of getting stuck in narrow minima.
- **Learning rate scaling**Empirically, if you double the batch size, you can often double the learning rate without diverging. This keeps the “signal-to-noise ratio” of the update roughly constant.

### 5. Hardware and throughput

- GPUs/TPUs achieve peak throughput when data is fed in large contiguous blocks. Powers-of-two batch sizes (e.g., 32, 64, 128) often run fastest.
- Very large batches can exceed memory limits or underutilize the optimizer’s ability to explore the loss surface.
- Mixed-precision training pairs well with moderate-sized batches to maximize speed and stability.

### 6. Empirical rule of thumb for batch size

| Dataset size m | Batch size b suggestion | Notes |
| --- | --- | --- |
| Small (m≤10 k) | 16–64 | helps regularize via noise |
| Medium (10 k–1 M) | 32–256 | balances throughput and variance |
| Large (m>1 M) | 256–1024+ | scale up to hardware limits, adjust LR |

### 7. Code demo: measuring gradient variance

```python
import numpy as np

# toy data
np.random.seed(1)
m, d = 1000, 10
X = np.random.randn(m, d)
true_theta = np.arange(d).reshape(d, 1)
y = X.dot(true_theta) + np.random.randn(m, 1)*0.5

def batch_gradient(X_batch, y_batch, theta):
    preds = X_batch.dot(theta)
    return (2/len(y_batch)) * X_batch.T.dot(preds - y_batch)

theta = np.zeros((d, 1))
batch_sizes = [1, 32, 128, m]
variances = []

for b in batch_sizes:
    grads = []
    for _ in range(50):
        idx = np.random.choice(m, b, replace=False)
        g = batch_gradient(X[idx], y[idx], theta)
        grads.append(g.flatten())
    grads = np.stack(grads)
    variances.append(np.var(grads, axis=0).mean())

for b, var in zip(batch_sizes, variances):
    print(f"Batch size {b:4d}: avg gradient variance = {var:.5f}")
```

This script samples multiple mini-batches at different sizes and reports the average variance of the gradient estimate. You’ll observe variance dropping as batch size grows.

### 8. Visualization idea

1. Fix θ and plot several gradient vectors computed on random batches in the parameter space.
2. Show how small batches produce a “cloud” of gradient estimates around the true gradient, whereas large batches cluster tightly.

### 9. Best practices

- Always shuffle examples each epoch to break ordering effects.
- Start with a moderate batch size (32–128), tune learning rate first, then experiment with batch size.
- Combine mini-batches with learning-rate schedules (step decay, cosine annealing) or adaptive optimizers (Adam) for even better results.
- Monitor training loss and validation loss separately to catch overfitting early.

### 10. Interview-ready insights

- Explain the trade-off between computational efficiency and gradient accuracy.
- Discuss why controlled noise can improve generalization.
- Be ready to justify choice of batch size and how to adapt both batch size and learning rate as you scale up.

### 11. Practice exercises

1. Run the variance-measurement code on a real dataset (e.g., MNIST) to see how gradient variance scales with b in a neural network.
2. Plot training loss vs. number of parameter updates (not epochs) for different batch sizes.
3. Implement a learning-rate warm-up schedule that linearly increases the rate over the first few epochs when using large batches.

---

## Exponentially Weighted Averages

### 1. Direct definition

An exponentially weighted average (EWA) computes a smoothed version of a time series by giving more weight to recent observations and an exponentially decaying weight to older ones. Each new average combines the previous average and the current value with a smoothing factor α (alpha) between 0 and 1.

### 2. Intuition

- When α is close to 1, you rely mostly on the latest value, leading to a very responsive but noisy average.
- When α is close to 0, you rely heavily on the past average, yielding a very smooth but sluggish response.
- The “memory” of older values decays exponentially, so after k steps their influence is roughly (1 − α)^k.

### 3. Mathematical breakdown

Let

- v_t be the exponentially weighted average at time step t
- x_t be the raw observation at time step t
- α be the smoothing factor (0 < α < 1)

We initialize v_0 (often to 0 or to x_0), then for each new t:

v_t = α * x_t + (1 − α) * v_{t−1}

This recurrence keeps updating v_t in one pass, using only the current observation and the previous average.

### 4. Python implementation

```python
def exponential_weighted_average(xs, alpha, initial=None):
    """
    Compute exponentially weighted average of list xs using smoothing factor alpha.

    Args:
      xs        : list or array of observations
      alpha     : smoothing factor between 0 and 1
      initial   : starting value for v_0; if None, uses xs[0]

    Returns:
      list of same length as xs, containing EWA at each time step
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")

    v = initial if initial is not None else xs[0]
    ewav = [v]

    for x in xs[1:]:
        v = alpha * x + (1 - alpha) * v
        ewav.append(v)

    return ewav

# Example usage
raw = [10, 12, 11, 13, 12, 14]
smoothed = exponential_weighted_average(raw, alpha=0.3)
print("Raw values:     ", raw)
print("Smoothed values:", [round(val, 2) for val in smoothed])
```

### 5. Applications in optimization

In deep learning optimizers like momentum, RMSprop, and Adam, exponentially weighted averages track:

- gradients (first moment)
- squared gradients (second moment)

This smoothing reduces oscillation and adapts learning rates per parameter. For momentum, you maintain:

```python
momentum = beta * momentum + (1 - beta) * gradient
```

where beta plays the role of 1 − α in our EWA formula.

### 6. Tips and best practices

- Choose α based on how much past history you want to retain. A common choice for tracking gradients is β = 0.9 (which corresponds to α = 0.1).
- If your series has a strong trend, consider subtracting the initial bias: divide v_t by (1 − (1 − α)^t) to correct for the zero initialization effect in early steps.
- Check edge cases: if α is exactly 1, you ignore history; if α is 0, you never update from the initial value.
- Visualize different α values on your data to see the smoothing effect before committing to one.

### 7. Practice exercises

1. Plot EWAs for a noisy sine wave at α values [0.1, 0.3, 0.7] and compare smoothing.
2. Implement bias-corrected EWA:and show the difference in the first 20 steps.
    
    ```python
    v_biascorr = v_t / (1 - (1 - alpha) ** t)
    ```
    
3. Integrate EWA into your own SGD loop for a toy neural network and observe how momentum smooths out parameter updates.

### 8. Interview-ready insights

- Explain why EWAs are “online” (single-pass, constant memory) and ideal for streaming data.
- Discuss the bias correction term and why it matters during the first few iterations.
- Relate smoothing factor choices to the “effective window size,” which is roughly 1/α steps of memory.

---

## Understanding Exponentially Weighted Averages

### 1. Direct summary

An exponentially weighted average (EWA) is a running average that smooths a sequence by giving each new observation a fixed weight α and decaying the influence of past values by a factor of (1–α) at each step.

### 2. Intuitive view

- Think of a rolling average that “remembers” recent values more than old ones.
- At every time t you mix the new data point and the previous average in fixed proportions.
- The memory of an observation k steps ago is (1–α)^k, so older points fade out exponentially.

### 3. Core formula

Let

- x_t be the new data point at step t
- v_t be the EWA at step t
- α be the smoothing factor, 0 < α < 1

Initialize v_0 (often to x_0 or 0). Then for each t ≥ 1:

v_t = α * x_t + (1 – α) * v_{t–1}

### 4. Effective window size

Even though the formula spans all past steps, most of the weight is concentrated in a finite “window.”

| α value | Approximate window length (1/α) | Behavior |
| --- | --- | --- |
| 0.05 | 20 | Very smooth, slow to react |
| 0.1 | 10 | Moderate smoothing |
| 0.3 | ~ 3 | Quick reaction, more noise |
| 0.5 | 2 | Very responsive, noisy average |

### 5. Bias correction

Early on, v_t tends to be biased toward the initial value, because there isn’t enough history. You can correct it by dividing by the cumulative weight:

1. Compute raw average:
    
    v_t(raw) = α * x_t + (1 – α) * v_{t–1}(raw)
    
2. Compute bias factor:
    
    bias_correction = 1 – (1 – α)^t
    
3. Corrected average:
    
    v_t = v_t(raw) / bias_correction
    

This makes sure your EWA starts unbiased from the first step.

### 6. Python implementation

```python
def ewma(xs, alpha):
    """Return exponentially weighted moving average of xs with smoothing alpha."""
    v = xs[0]
    ewav = [v]
    for t, x in enumerate(xs[1:], start=1):
        v = alpha * x + (1 - alpha) * v
        bias = 1 - (1 - alpha) ** (t + 1)
        v_corrected = v / bias
        ewav.append(v_corrected)
    return ewav

# Example
raw = [5, 7, 6, 9, 8, 10]
smoothed = ewma(raw, alpha=0.2)
print("Raw:", raw)
print("EWA:", [round(v, 2) for v in smoothed])
```

### 7. Applications in deep learning

- **Momentum optimizer**
    
    Tracks a running average of past gradients to push updates in consistent directions:
    
    `v = β * v + (1 - β) * grad`
    
- **RMSprop and Adam**
    
    Use separate EWAs for squared gradients (second moment) to adapt per-parameter learning rates.
    

### 8. Visualization suggestions

- Plot raw versus smoothed data for different α values on the same axes.
- Show how the weight of a single impulse decays over t steps.
- Demonstrate bias correction by overlaying raw EWA and corrected EWA early in training.

### 9. Common pitfalls

- Picking α too small makes the EWA too sluggish to track changes.
- Picking α too large makes it nearly as noisy as the raw data.
- Forgetting bias correction can mislead you in the first few iterations.
- Using EWA on nonstationary data without resetting—old context may linger in v_t.

### 10. Interview insights

- Describe EWAs as “online algorithms” that run in O(1) time and constant memory.
- Highlight the trade-off between smoothness and responsiveness via α.
- Explain the need for bias correction and how it’s derived from geometric series.
- Relate EWA to classic signal-processing filters and their transfer functions.

### 11. Practice exercises

- Apply EWA to a noisy sine wave at α = 0.05, 0.2, 0.5 and compare.
- Use EWAs to track training and validation loss in a small neural network; experiment with and without bias correction.
- Implement a double exponential smoothing (to capture trend) and compare to single EWA.

---

## Gradient Descent with Momentum

### 1. Direct definition

Gradient descent with momentum augments the plain gradient update by accumulating a velocity term that “remembers” past gradients. At each step you update this velocity and then move parameters along it. This leads to faster convergence and smoother trajectories, especially in directions with high curvature.

### 2. Intuition

Momentum treats your parameter updates like a ball rolling down a hill:

- On steep slopes, it builds up speed by accumulating past gradients.
- In shallow, flat regions it keeps moving even when the raw gradient is small.
- When gradients oscillate across a ravine, momentum dampens the zig-zag by averaging successive steps.

### 3. Math breakdown

Let

- θ be the parameters vector
- g_t be the gradient of the loss w.r.t. θ at iteration t
- v_t be the velocity (momentum term) at iteration t
- α be the learning rate
- β be the momentum coefficient (0 ≤ β < 1, often 0.9)

Initialize

```
v_0 = 0
θ_0 = random or default
```

At each iteration t ≥ 1:

```
g_t = ∇θ Loss(θ_{t-1})
v_t = β * v_{t-1} + (1 - β) * g_t
θ_t = θ_{t-1} - α * v_t
```

Here, β controls how much past gradients influence the current velocity. A typical choice β = 0.9 gives a long “memory” of past updates.

### 4. Connection to exponentially weighted averages

The velocity v_t is exactly an exponentially weighted average of the gradients:

- Each new gradient contributes (1 – β) of its value.
- Past gradients decay by factor β each step.
- v_t tracks a smoothed direction of descent rather than the raw gradient at t.

### 5. NumPy implementation

```python
import numpy as np

# toy linear regression data
np.random.seed(0)
m = 200
X = 2 * np.random.rand(m, 1)
y = 3 + 4 * X + np.random.randn(m, 1)

# add bias term
X_b = np.c_[np.ones((m, 1)), X]
d = X_b.shape[1]

# hyperparameters
learning_rate = 0.1
momentum = 0.9
n_iterations = 1000

# initialize
theta = np.zeros((d, 1))
v = np.zeros((d, 1))

for iteration in range(1, n_iterations + 1):
    # compute gradient
    error = X_b.dot(theta) - y
    grad = (2 / m) * X_b.T.dot(error)

    # update velocity and parameters
    v = momentum * v + (1 - momentum) * grad
    theta = theta - learning_rate * v

print("Estimated parameters:", theta.ravel())
```

### 6. Geometric perspective

Imagine the loss surface as a narrow valley:

- Plain gradient descent bounces from one wall to the other, making slow progress along the valley floor.
- Momentum builds up speed along the valley direction and dampens oscillations, cutting diagonally through each swipe and reaching the minimum faster.

### 7. Tips and pitfalls

- Common β values are 0.8, 0.9, 0.99. Higher β means longer memory but slower response to changing gradients.
- If learning rate α is too large, momentum can cause divergence (overshooting). Reduce α when increasing β.
- Always monitor training curves; if loss oscillates, lower α or β.
- Combine momentum with mini-batches and learning-rate schedules (decay, warm-up) for best results.

### 8. Interview-ready insights

- Explain how momentum accelerates convergence in directions of consistent gradient and damps oscillation in high-curvature directions.
- Quantify effective “look-back” length: roughly 1 / (1 – β) steps of memory.
- Compare classical momentum to Nesterov accelerated gradient, which computes the gradient at a “look-ahead” position for even faster convergence.

### 9. Practice exercises

1. Implement gradient descent with momentum for logistic regression on a binary dataset and compare convergence to plain SGD.
2. Plot parameter trajectories on a 2D quadratic loss for plain GD vs. GD with momentum.
3. Experiment with β values [0.5, 0.8, 0.9, 0.99] and observe how quickly and smoothly the algorithm converges.
4. Combine momentum with a learning-rate decay schedule and report final test accuracy on a small neural network.

---

## RMSprop

### 1. Direct definition

RMSprop (Root Mean Square Propagation) is an adaptive learning-rate method that scales each parameter update by a moving average of recent squared gradients. It keeps the benefits of momentum-style smoothing but automatically shrinks learning rates for parameters with large gradients and increases them for parameters with small gradients.

### 2. Intuition

- If a parameter’s gradient has been large recently, RMSprop divides by a large moving average of squared gradients, shrinking its effective learning rate.
- If gradients have been small, the denominator is small, boosting that parameter’s learning rate.
- This per-parameter scaling speeds up training on problems where some weights see steeper slopes than others, and it avoids the diminishing learning-rate issue of plain Adagrad.

### 3. Math breakdown

Let

- θ be the parameter vector
- g_t be the gradient at time step t
- s_t be the running average of squared gradients
- α be the base learning rate
- β be the decay rate for the squared gradient average (commonly 0.9 or 0.99)
- ϵ be a small constant for numerical stability (e.g., 1e-8)

Initialize

```
s_0 = 0
θ_0 = initial parameters
```

At each iteration t ≥ 1:

```
g_t = ∇θ Loss(θ_{t−1})
s_t = β * s_{t−1} + (1 - β) * (g_t ◦ g_t)
θ_t = θ_{t−1} - α * g_t / (sqrt(s_t) + ϵ)
```

Here “◦” denotes element-wise multiplication and “sqrt” is element-wise square root.

### 4. Connection to exponentially weighted averages

The term s_t is exactly an exponentially weighted average of past squared gradients. You’re using that average to normalize the raw gradient at each step, effectively giving each parameter its own adaptive step size based on historical curvature.

### 5. NumPy implementation

```python
import numpy as np

# toy data: linear regression
np.random.seed(0)
m = 200
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1) * 0.5

# add bias term
X_b = np.c_[np.ones((m, 1)), X]
d = X_b.shape[1]

# hyperparameters
learning_rate = 0.01
beta = 0.9
epsilon = 1e-8
n_iterations = 1000

# initialize
theta = np.zeros((d, 1))
s = np.zeros((d, 1))

for iteration in range(1, n_iterations + 1):
    # compute gradient
    error = X_b.dot(theta) - y
    grad = (2 / m) * X_b.T.dot(error)

    # update running average of squared gradients
    s = beta * s + (1 - beta) * (grad * grad)

    # parameter update
    theta -= learning_rate * grad / (np.sqrt(s) + epsilon)

print("Estimated parameters:", theta.ravel())
```

### 6. Geometric/intuitive view

Imagine each parameter in a steep slope and flat floor in different directions. RMSprop adapts the step length so that in directions with steep curvature you take smaller steps and in flatter directions you take larger steps. This helps you dive quickly into valleys without overshooting along steep walls.

### 7. Tips and pitfalls

- Typical defaults: β = 0.9 or 0.99, ϵ = 1e-8, learning rate 0.001–0.01.
- If β is too small, the normalization will react too quickly to noisy gradients; if too large, it will change too slowly.
- Always include ϵ to prevent division by zero.
- Monitor effective learning rates (learning_rate / (sqrt(s_t)+ϵ)) to ensure they stay in a healthy range.
- Combine with mini-batches and learning-rate schedules for best performance.

### 8. Interview-ready insights

- Explain why Adagrad’s per-parameter rate decays too aggressively and how RMSprop fixes it by using an exponential window instead of a sum.
- Describe the trade-off controlled by β: smoothing versus adaptability.
- Be ready to compare RMSprop to Adam—both use EWAs but Adam also tracks a first moment (average gradient) and includes bias correction.

### 9. Practice exercises

1. Apply RMSprop to a small neural network on MNIST and plot training loss vs. epochs for different β values.
2. Visualize per-parameter adaptive rates over time for a toy quadratic problem.
3. Compare convergence speed and final test accuracy for plain SGD, momentum, RMSprop, and Adam on a shallow classification model.

---

## Adam Optimization Algorithm

### 1. Direct definition

Adam (Adaptive Moment Estimation) is an optimization algorithm that computes adaptive learning rates for each parameter by combining

- an exponentially weighted average of past gradients (first moment)
- an exponentially weighted average of past squared gradients (second moment)with bias-correction terms to stabilize updates.

### 2. Intuition

- Like momentum, Adam builds velocity in directions with consistent gradients, speeding progress.
- Like RMSprop, it rescales updates by the inverse root of the second moment, shrinking steps for parameters with large curvature.
- Bias corrections compensate for initializing both moment estimates at zero, ensuring reliable step sizes from the very first iteration.

### 3. Math breakdown

Let

- θ_t be the parameters at iteration t
- g_t = ∇θ Loss(θ_{t–1}) be the gradient at t
- m_t be the first moment estimate
- v_t be the second moment estimate
- α be the base learning rate
- β₁ and β₂ be decay rates for the first and second moments
- ε be a small constant for numerical stability

Initialize

```
m₀ = 0, v₀ = 0, t = 0
```

Each step:

1. t ← t + 1
2. m_t ← β₁ * m_{t–1} + (1 – β₁) * g_t
3. v_t ← β₂ * v_{t–1} + (1 – β₂) * (g_t ◦ g_t)
4. m̂_t ← m_t / (1 – β₁ᵗ) ← bias-corrected first moment
5. v̂_t ← v_t / (1 – β₂ᵗ) ← bias-corrected second moment
6. θ*t ← θ*{t–1} – α * m̂_t / (sqrt(v̂_t) + ε)

Here “◦” means element-wise multiplication, and sqrt and division are applied element-wise.

### 4. Hyperparameter defaults

| Hyperparameter | Typical Symbol | Default Value |
| --- | --- | --- |
| Learning rate | α | 0.001 |
| First moment decay rate | β₁ | 0.9 |
| Second moment decay rate | β₂ | 0.999 |
| Epsilon (numerical stability) | ε | 1e-8 |

### 5. NumPy implementation

```python
import numpy as np

# Toy linear regression data
np.random.seed(42)
m = 300
X = 2 * np.random.rand(m, 1)
y = 5 + 2 * X + np.random.randn(m, 1) * 0.5

# Add bias term
X_b = np.c_[np.ones((m, 1)), X]
d = X_b.shape[1]

# Hyperparameters
alpha = 0.001
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
n_iterations = 2000

# Initialization
theta = np.zeros((d, 1))
m_t = np.zeros((d, 1))
v_t = np.zeros((d, 1))
t = 0

for iteration in range(n_iterations):
    t += 1
    # Compute gradient
    error = X_b.dot(theta) - y
    grad = (2 / m) * X_b.T.dot(error)

    # Update biased first and second moment estimates
    m_t = beta1 * m_t + (1 - beta1) * grad
    v_t = beta2 * v_t + (1 - beta2) * (grad * grad)

    # Compute bias-corrected estimates
    m_hat = m_t / (1 - beta1**t)
    v_hat = v_t / (1 - beta2**t)

    # Update parameters
    theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

print("Estimated parameters:", theta.ravel())
```

### 6. Tips and pitfalls

- If training is unstable or loss diverges, try reducing α before touching β₁ or β₂.
- β₁ controls how quickly you forget past gradients; lower β₁ reacts faster to recent changes but yields noisier updates.
- β₂ controls smoothing of squared gradients; higher β₂ makes per-parameter rates more stable but slower to adapt.
- Never disable bias correction when using default β₁, β₂—without it early updates are overly small.
- Combine Adam with mini-batches for efficient, scalable training on large datasets.

### 7. Interview-ready insights

- Explain how Adam unifies momentum and adaptive scaling in one algorithm.
- Highlight the purpose of bias correction for m̂_t and v̂_t.
- Discuss scenarios where Adam may outperform SGD with momentum (sparse gradients, noisy problems).
- Know limitations: can sometimes fail to converge to the exact minimum or generalize as well as well-tuned SGD in some vision tasks.

### 8. Practice exercises

1. Train a small neural network on MNIST using SGD, SGD+momentum, RMSprop, and Adam. Compare training curves and final test accuracy.
2. Vary β₁ and β₂ across [0.8, 0.9, 0.99] and observe sensitivity in convergence speed.
3. Visualize per-parameter adaptive learning rates (α / (sqrt(v̂_t) + ε)) over time for a toy quadratic problem.

---

## Learning Rate Decay

### 1. Direct definition

Learning rate decay is a strategy that gradually reduces the learning rate (step size) during training. Instead of keeping a fixed learning rate α, you apply a schedule that lowers α over epochs or steps to fine-tune convergence and avoid oscillation near a minimum.

### 2. Why decay matters

- It lets you take large steps early on to explore the loss surface quickly.
- It slows down updates later to carefully settle into the minimum.
- It can improve generalization by preventing parameter updates from bouncing around in low-loss regions.
- It reduces the need to pick one “perfect” learning rate value up front.

### 3. Common decay schedules

| Schedule | Formula (epoch t) | Key hyperparameters | Notes |
| --- | --- | --- | --- |
| Time-based decay | α_t = α₀ / (1 + decay_rate * t) | initial α₀, decay_rate | Simple inverse-time scaling |
| Step decay | α_t = α₀ * drop_factor ^ floor(t / step_size) | drop_factor (e.g. 0.5), step_size | Sharp drops at fixed intervals |
| Exponential decay | α_t = α₀ * exp(− decay_rate * t) | initial α₀, decay_rate | Smooth, continuous exponential |
| 1/t decay | α_t = α₀ / t | initial α₀ | Quick early drop, slow later |
| Polynomial decay | α_t = (α₀ − α_end) * (1 − t / T)^power + α_end | α_end, T (total epochs), power | Flexible tail shape |
| Cosine annealing | α_t = α_min + 0.5 * (α₀ − α_min) * (1 + cos(π * t / T)) | α_min, T | Smooth drop to α_min at T |
| Cyclical learning rate | α_t cycles between α_low and α_high over steps | α_low, α_high, cycle_length | Periodic warm restarts to escape local minima |

### 4. Intuitive behavior

- Early training: higher α lets you make bold moves, jumping over small bumps.
- Mid training: moderate α balances progress and stability.
- Late training: small α avoids overshooting and fine-tunes parameters.
- Cyclical or cosine schedules can reintroduce larger α briefly to help jump out of shallow minima.

### 5. Python implementation (pure NumPy style)

```python
def get_learning_rate(initial_lr, epoch, schedule, **kwargs):
    """
    Compute learning rate for a given epoch using different schedules.
    schedule: 'time', 'step', 'exp', 'poly', 'cosine'
    kwargs for schedules:
      decay_rate, step_size, drop_factor, alpha_end, total_epochs, power, alpha_min
    """
    if schedule == 'time':
        decay = kwargs.get('decay_rate', 0.1)
        return initial_lr / (1 + decay * epoch)
    if schedule == 'step':
        drop = kwargs.get('drop_factor', 0.5)
        step = kwargs.get('step_size', 10)
        return initial_lr * drop ** (epoch // step)
    if schedule == 'exp':
        decay = kwargs.get('decay_rate', 0.05)
        return initial_lr * np.exp(-decay * epoch)
    if schedule == 'poly':
        alpha_end = kwargs.get('alpha_end', 0.0001)
        total = kwargs.get('total_epochs', 100)
        power = kwargs.get('power', 2.0)
        return (initial_lr - alpha_end) * (1 - epoch / total) ** power + alpha_end
    if schedule == 'cosine':
        alpha_min = kwargs.get('alpha_min', 0.0001)
        total = kwargs.get('total_epochs', 100)
        cos_inner = np.cos(np.pi * epoch / total)
        return alpha_min + 0.5 * (initial_lr - alpha_min) * (1 + cos_inner)
    raise ValueError("Unknown schedule")

# Example: print LR for first 20 epochs using step decay
initial_lr = 0.1
for e in range(20):
    lr = get_learning_rate(initial_lr, e, 'step', drop_factor=0.5, step_size=5)
    print(f"Epoch {e:2d}: lr = {lr:.5f}")
```

### 6. Framework snippet (PyTorch)

```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

model = ...              # your neural net
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

# Step decay: multiply lr by 0.1 every 30 epochs
scheduler_step = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing: drop from 0.1 to 0.001 over 100 epochs
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

for epoch in range(1, 101):
    train_one_epoch(...)
    validate(...)
    scheduler_step.step()
    # or scheduler_cosine.step()
    print("Epoch", epoch, "lr:", optimizer.param_groups[0]["lr"])
```

### 7. Best practices & pitfalls

- Tune your initial learning rate with a learning-rate finder (e.g., test increasing α and pick where loss just starts to diverge).
- Combine decay with momentum or adaptive optimizers for smooth convergence.
- Avoid too-aggressive drop factors or decay rates; if LR falls below a useful threshold, training stalls.
- For cyclic schedules, pick cycle length based on expected “plateau” durations.
- Always monitor both training and validation loss curves when applying decay.

### 8. Interview-ready insights

- Explain why decaying the learning rate can be viewed as annealing the optimization process, analogous to simulated annealing.
- Discuss trade-offs: constant LR can converge faster early but may oscillate later; decay trades early speed for late stability.
- Compare static schedules (step, exponential) to dynamic or metric-based decay (reduce LR on plateau when validation loss stops improving).
- Describe one-cycle policy: increase LR linearly then decrease, often yields state-of-the-art results in modern training.

### 9. Practice exercises

1. Implement and compare time-based, step, exponential, and cosine decay on a small neural network trained on MNIST. Plot LR vs. epoch and training loss curves.
2. Integrate a “reduce on plateau” scheduler: lower LR by a factor when validation loss hasn’t improved for N epochs. Observe its effect on overfitting.
3. Try the one-cycle policy: warm up LR over the first 10% of steps then cool down over the remaining 90%. Measure final validation accuracy on CIFAR-10.

---

## The Problem of Local Optima

### 1. Direct definition

A local optimum is a solution where the loss (or objective) value is lower than in its immediate neighborhood, but not necessarily the lowest possible over the entire parameter space. In non-convex landscapes—like those of deep neural networks—gradient-based methods can get “stuck” in these suboptimal valleys instead of reaching the global minimum.

### 2. Intuitive/geometric view

Imagine a mountainous terrain with many peaks and valleys. You drop a ball at some point and let it roll downhill. If it reaches a small basin surrounded by higher ridges, it’ll settle there even though a deeper valley exists elsewhere. That basin is a local optimum; the deepest valley is the global optimum.

### 3. Why it matters in deep learning

- Neural network loss surfaces are highly non-convex and full of plateaus, valleys, and saddle regions.
- Getting trapped in a shallow valley can slow convergence and degrade final model performance.
- Local optima aren’t always catastrophic—modern networks often find “good enough” minima—but understanding them helps design better optimizers and architectures.

### 4. Local minima vs. saddle points

- Local minima: points where all directions have non-negative curvature (like the bottom of a bowl).
- Saddle points: points where some directions curve down while others curve up (like a mountain pass).
- In high dimensions, saddle points are more common than strict local minima and can stall gradient descent just as badly.

### 5. Strategies to escape or avoid local optima

- Random restarts (train the same model multiple times with different initial weights).
- Mini-batch noise: stochastic gradient descent’s noisy updates can push you out of shallow traps.
- Momentum: builds velocity to roll through small bumps instead of stopping.
- Adaptive optimizers (RMSprop, Adam): automatically adjust per-parameter step sizes to overcome narrow valleys.
- Learning-rate schedules or warm restarts: temporarily boost step size to escape basins.
- Architectural choices (skip-connections, batch normalization): smooth the loss surface and reduce harmful curvature.

### 6. Code demo: toy non-convex function

```python
import numpy as np
import matplotlib.pyplot as plt

# Non-convex function: f(x) = x^4 - 3*x^2 + 2
def f(x):
    return x**4 - 3*x**2 + 2

def grad_f(x):
    return 4*x**3 - 6*x

def gradient_descent(initial_x, lr, n_steps):
    x = initial_x
    path = [x]
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        path.append(x)
    return np.array(path)

# Run with different initial points
inits = [-2.0, -0.5, 0.5, 2.0]
lr = 0.05
n_steps = 50

x_vals = np.linspace(-3, 3, 400)
plt.plot(x_vals, f(x_vals), color='gray')

for x0 in inits:
    path = gradient_descent(x0, lr, n_steps)
    plt.plot(path, f(path), marker='o', label=f'init {x0}')
plt.legend()
plt.title('Gradient Descent on a Non-Convex Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```

This plot shows how different initializations lead to different basins (local minima) on the same landscape.

### 7. Interview-ready insights

- Explain how high-dimensional spaces make saddle points more prevalent than local minima.
- Discuss why small-batch noise or momentum can help algorithms traverse shallow regions.
- Be ready to compare first-order methods (SGD, Adam) with second-order methods (Newton, quasi-Newton) for escaping curvature traps.

### 8. Practice exercises

1. Implement gradient descent with momentum on the toy function above and observe how momentum helps escape shallow valleys.
2. Visualize trajectories with and without Adam on a two-dimensional non-convex surface (e.g., the Rosenbrock function).
3. Experiment with cyclical learning-rate schedules on a small neural network and track how restarts impact escaping plateaus.

---