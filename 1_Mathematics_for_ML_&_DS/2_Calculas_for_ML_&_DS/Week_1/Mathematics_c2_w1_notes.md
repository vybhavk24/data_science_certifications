# Mathematics_c2_w1

## Why Calculus Is Required for Machine Learning and Data Science

### Conceptual Role of Calculus

Calculus lets us understand and work with continuous change. In machine learning and data science, most models are expressed as continuous functions—think of the loss curve of a neural network or the probability density of a model.

Calculus tools help us

- Quantify how small changes in input or parameters affect a model’s output.
- Find optima—minimum error or maximum likelihood—by locating where the slope of a function is zero.
- Compute areas under curves (integrals) when dealing with probabilities, expectations, and smoothing kernels.

Without calculus, we couldn’t efficiently train models or reason about continuous data distributions.

### Key Calculus Concepts in ML and DS

- Derivative: measures instantaneous rate of change of a function.
- Partial derivative & gradient: extend derivatives to multi-dimensional parameter spaces.
- Gradient descent: iterative optimization that uses gradients to minimize loss functions.
- Integral: sums infinitesimal contributions—used in probability (area under density) and feature smoothing.
- Chain rule: links complex function compositions—critical in backpropagation through deep networks.

### Core Formulas

### 1. Derivative Definition

```
f'(x) = lim(h → 0) [f(x + h) − f(x)] / h
```

- f′(x): the derivative of f at x.
- h: a very small increment in x.
- f(x + h) − f(x): change in function’s value when x increases by h.
- Divide by h: gives the rate of change per unit of x.
- Take limit as h approaches zero: ensures instantaneous rate.

Real-world use: slope of loss curve at a given parameter setting.

### 2. Gradient of a Multivariable Function

```
∇f(x₁, x₂, …, xₙ) = [∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂xₙ]
```

- ∇f: vector of all partial derivatives of f.
- ∂f/∂xᵢ: derivative of f holding all other variables constant.
- Each component shows sensitivity of f to that single variable.

Real-world use: tells you which direction in parameter space reduces loss fastest.

### 3. Gradient Descent Update Rule

```
θ_new = θ_old − α * ∇J(θ_old)
```

- θ_old: current parameters (weights) of your model.
- J(θ): cost (loss) function you want to minimize.
- ∇J(θ_old): gradient of the cost at current parameters.
- α (alpha): learning rate, a small positive scalar.
- θ_new: updated parameters after one iteration.

Real-world use: backbone of training linear/logistic regression, neural networks, etc.

### Real-World Examples in ML and DS

- Linear regression training: derivative of mean squared error dictates weight updates.
- Logistic regression: derivative of the sigmoid-loss function provides gradient for classification tasks.
- Neural network backpropagation: repeated use of chain rule and gradients to adjust millions of weights.
- Kernel density estimation: integrals to compute probabilities and smooth distributions.
- Regularization: derivative of penalty terms (L1, L2) added to loss to prevent overfitting.

### Practice Problems and Python Exercises

1. Compute by hand
    - Derivative of f(x) = 3x² + 5x − 2.
    - Derivative of f(x) = e^(2x) · x.
2. Gradient of a bivariate function
    - f(x,y) = x²·y + 4xy². Compute ∇f at (1,2).
3. Implement gradient descent for simple linear regression

```python
import numpy as np

# Generate toy data
X = np.linspace(0, 10, 50)
y = 2.5 * X + np.random.randn(50) * 2

# Add bias term
X_b = np.vstack([np.ones(len(X)), X]).T

# Hyperparameters
alpha = 0.01
n_iterations = 1000
theta = np.random.randn(2)

# Gradient descent loop
for i in range(n_iterations):
    gradients = 2/len(X_b) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - alpha * gradients

print("Learned parameters:", theta)
```

1. Compute definite integral
    - ∫₀¹ x³ dx by hand and verify using numerical integration in Python.

### Geometric Intuition

Imagine the graph of a loss function as a hilly landscape.

- Derivative at a point gives the slope of the tangent line—how steep the hill is.
- Gradient in higher dimensions points “uphill” most steeply. Taking the negative gradient means you walk “downhill” fastest to reach the minimum.
- Integrals measure the total accumulated area under a curve—used in expectations and probability mass.

By mastering these core calculus tools, you’ll have the mathematical foundation to derive, implement, and troubleshoot every step of model training and evaluation.

---

## Introduction to Derivatives

### What Is a Derivative?

A derivative measures how a function’s output changes as its input moves by a tiny amount.

Think of riding a bike up a hill: at each point the steepness tells you how hard you must pedal. The derivative is that “steepness” for any curve you draw on a graph.

In data science and machine learning, derivatives show how small tweaks in parameters change your model’s loss or output.

### Formal Definition of the Derivative

```
f'(x) = lim(h -> 0) (f(x + h) - f(x)) / h
```

- f′(x): the derivative of f at point x
- h: a very small increment in the input
- f(x + h) – f(x): change in the function’s value
- Divide by h: gives the change per unit of input
- lim(h -> 0): ensures we capture the instantaneous rate

### Geometric Interpretation

Imagine the graph of y = f(x).

- Draw two points: (x, f(x)) and (x + h, f(x + h)).
- Connect them with a straight line (the secant).
- As h shrinks, the secant pivots and approaches the tangent line at x.
- The slope of that tangent is f′(x).

This slope tells you the direction and steepness of the curve right at x.

### Why Slope Matters in ML and DS

- **Model training:** Derivatives of the loss function guide weight updates in gradient descent.
- **Sensitivity analysis:** Understand which inputs or parameters most affect your prediction.
- **Optimization:** Locate minima or maxima by finding points where f′(x) = 0.

Every time a neural network learns, it’s using derivatives to decide how to adjust millions of weights.

### Calculating a Slope by Hand

1. Choose a simple function, for example f(x) = x².
2. Plug into the definition:
    
    ```
    f'(x) = lim(h -> 0) ((x + h)² - x²) / h
    ```
    
3. Expand and simplify:
    
    ```
    = lim(h -> 0) (x² + 2xh + h² - x²) / h
    = lim(h -> 0) (2xh + h²) / h
    = lim(h -> 0) (2x + h)
    = 2x
    ```
    

So the derivative of x² is 2x.

### Practice Problems

1. By hand, compute the derivative of:
    - f(x) = 3x³
    - f(x) = 1/x
2. Estimate the derivative of f(x) = sin(x) at x = 0.5 using h = 0.01.

### Python Exercises

```python
import numpy as np

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# Example function
f = lambda x: x**2 + 2*x + 1

# Points to evaluate
xs = np.linspace(-3, 3, 100)
approx_slopes = [derivative(f, x) for x in xs]

# Print a few values
for x, slope in zip(xs[::20], approx_slopes[::20]):
    print(f"x={x:.2f}, approx f'={slope:.2f}")
```

- Plot f(x) and its tangent at x=1:

```python
import matplotlib.pyplot as plt

x0 = 1
slope0 = derivative(f, x0)
tangent = lambda x: f(x0) + slope0 * (x - x0)

x_plot = np.linspace(-3, 3, 200)
plt.plot(x_plot, f(x_plot), label="f(x)")
plt.plot(x_plot, tangent(x_plot), '--', label="tangent at x=1")
plt.scatter([x0], [f(x0)], color='red')
plt.legend()
plt.show()
```

---

## Derivatives and Tangents

### Quick Reminder: What Is a Derivative?

A derivative at a point measures the instantaneous rate of change of a function.

It tells you how steep the graph of y = f(x) is at a particular x-value.

That steepness is exactly the slope of the tangent line at that point.

### Tangent Line: Definition

A tangent line touches the curve at exactly one point and has the same instantaneous slope as the curve there.

If you zoom in infinitely close around that point, the curve and its tangent become indistinguishable.

### Formula for the Tangent Line

Given a function f and a point a, the equation of the tangent line at x = a is:

```
y_tangent(x) = f(a) + f'(a) * (x - a)
```

- y_tangent(x): predicted y on the tangent line for any x.
- f(a): the function’s value at the point of tangency.
- f'(a): derivative (slope) of the function at x = a.
- (x - a): horizontal distance from the tangency point.

### Deriving the Tangent-Line Formula

1. **Point-slope form of a line**
    
    Any line with slope m through (x₀, y₀) is
    
    ```
    y = y₀ + m * (x - x₀)
    ```
    
2. **Apply to our curve**
    - x₀ = a
    - y₀ = f(a)
    - m = f'(a)
3. **Result**
    
    ```
    y_tangent(x) = f(a) + f'(a) * (x - a)
    ```
    

### Geometric Intuition

- **Secant lines** connect two points on the curve.
- As the second point approaches the first (h → 0), the secant becomes the tangent.
- The slope of that limiting secant is the derivative f′(a).

Imagine walking on a hill: at each step, you align your direction to the hill’s flank exactly at your feet—that’s your instantaneous direction (tangent), not the average direction over a longer stretch (secant).

### Worked Example

Function: f(x) = x²

Point of tangency: a = 2

1. Compute f(a):
    
    f(2) = 2² = 4
    
2. Compute derivative f′(x) = 2x, so f′(2) = 4
3. Plug into formula:
    
    ```
    y_tangent(x) = 4 + 4 * (x - 2)
                 = 4 + 4x - 8
                 = 4x - 4
    ```
    

Tangent line at x = 2 is y = 4x − 4.

If you plot both y = x² and y = 4x − 4, they touch only at (2, 4).

### Real-World Machine-Learning Uses

- **Linear approximation**: Simplify complex loss surfaces locally.
- **Newton’s method**: Uses first and second derivatives to find roots/optima faster.
- **Taylor expansions**: Build higher-order approximations; tangent is the first-order term.
- **Interpretability**: At a given parameter setting, tangent slope shows sensitivity of loss to weight changes.

### Practice Problems

1. By hand, find the tangent line to f(x) = 3x³ − 5x at x = 1.
2. Compute f′(π/4) and equation of tangent for f(x) = sin(x).
3. Sketch f(x) = e^x and its tangent at x = 0.

### Python Exercise: Plotting Tangents

```python
import numpy as np
import matplotlib.pyplot as plt

# Define function and its derivative
f  = lambda x: x**2
df = lambda x: 2*x

# Point of tangency
a = 2
y0 = f(a)
m  = df(a)

# Tangent line
tangent = lambda x: y0 + m * (x - a)

# Plot
x_vals = np.linspace(0, 4, 200)
plt.plot(x_vals, f(x_vals), label='f(x) = x^2')
plt.plot(x_vals, tangent(x_vals), '--', label='tangent at x=2')
plt.scatter([a], [y0], color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and Tangent Line')
plt.show()
```

---

## Slopes, Maxima, and Minima

### Quick Refresher: Slope as a Derivative

At any point x on a curve y = f(x), the slope is the instantaneous rate of change.

That slope is exactly the derivative f′(x).

```
f′(x) = lim(h → 0) [f(x + h) − f(x)] / h
```

- As h → 0, the secant line between (x, f(x)) and (x+h, f(x+h)) becomes the tangent.
- The slope of that tangent is f′(x).

### Finding Extremes: Critical Points

To locate maxima and minima (peaks and valleys) of f(x):

1. **Compute the derivative** f′(x).
2. **Find critical points** by solving
    
    ```
    f′(x) = 0
    ```
    
3. **Classify each critical point** using the second derivative test.

### Second Derivative Test

```
Let x_c be a solution of f′(x) = 0.

If f″(x_c) > 0, f has a local minimum at x_c.
If f″(x_c) < 0, f has a local maximum at x_c.
If f″(x_c) = 0, test is inconclusive (higher-order derivatives needed).
```

- f″(x) measures concavity: curving up (>0) or curving down (<0).
- Local minimum looks like a “cup” (concave up); local maximum looks like a “cap” (concave down).

### Geometric Intuition

- **Slope = 0**: tangent is perfectly horizontal.
- **Concave up**: the curve bows upward—horizontal tangent is a valley.
- **Concave down**: the curve bows downward—horizontal tangent is a peak.

Imagine rolling a marble along the curve: it settles in a local minimum (cup) and can’t stay on a local maximum (cap).

### Real-World ML/DS Use Cases

- **Loss function optimization**: You minimize a loss by finding its minima.
- **Log-likelihood**: You maximize log-likelihood to fit probabilistic models.
- **Hyperparameter tuning**: Some methods treat validation error as a function of hyperparameters and search for its minimum.
- **Feature engineering**: Score metrics (e.g., AUC) may have peaks at certain parameter values.

### Worked Example

Function: f(x) = x³ − 3x + 1

1. Derivative:
    
    ```
    f′(x) = 3x² − 3
    ```
    
2. Solve f′(x) = 0:
    
    3x² − 3 = 0 → x² = 1 → x = ±1
    
3. Second derivative:
    
    ```
    f″(x) = 6x
    ```
    
    - f″(1) = 6 > 0 → local minimum at x = 1
    - f″(−1) = −6 < 0 → local maximum at x = −1
4. Values:
    - f(1) = (1)³ − 3·1 + 1 = −1
    - f(−1) = (−1)³ − 3·(−1) + 1 = 3

So the curve has a peak at (−1, 3) and a valley at (1, −1).

### Practice Problems

1. **By hand**
    
    a. f(x) = x⁴ − 4x²
    
    b. f(x) = sin(x) + x at x ∈ [−2π, 2π]
    
2. **Estimate numerically**
    
    Use h = 1e-5 to approximate f′ and f″ for f(x)=e^(−x²) at x = 0.5.
    
3. **Python exercise**
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define function and discrete derivative approximators
    f  = lambda x: x**3 - 3*x + 1
    df = lambda x, h=1e-5: (f(x+h)-f(x-h)) / (2*h)
    d2f= lambda x, h=1e-5: (f(x+h) - 2*f(x) + f(x-h)) / (h*h)
    
    # Find critical points by scanning
    xs = np.linspace(-2, 2, 1001)
    crits = [x for x in xs if abs(df(x)) < 1e-3]
    print("Approx critical points:", crits)
    
    # Classify
    for x_c in crits:
        print(x_c, "f'' ≈", d2f(x_c))
    
    # Plot function and mark extrema
    plt.plot(xs, f(xs), label='f(x)')
    for x_c in crits:
        plt.scatter([x_c], [f(x_c)],
                    color='red' if d2f(x_c)<0 else 'green')
    plt.title('Extrema of f(x)')
    plt.legend()
    plt.show()
    ```
    

### Visual Description

Imagine the cubic curve f(x)=x³−3x+1:

- A red dot at (−1, 3) sitting on a hillcap.
- A green dot at (1, −1) nestled in a valley.
- Flat tangent lines at those points.

---

## Concepts: Derivatives and Numerical Approximation

### 1. What Is a Derivative?

A derivative measures how fast a function’s output changes when you tweak its input by a tiny amount.

Think of a bicycle ride up a hill: the instantaneous steepness at each point tells you how hard you must pedal. In math, that steepness is the derivative of the hill’s elevation curve.

In machine learning, derivatives tell you how changing a model’s weight by a small amount affects its loss. This information drives every gradient-based optimization algorithm.

### 2. Formal Definition

```
f′(x) = limit as h → 0 of [f(x + h) − f(x)] / h
```

- f′(x): derivative of f at x
- h: a very small step in input
- f(x + h) − f(x): change in function output
- Divide by h: change per unit input
- Taking the limit as h goes to zero gives the instantaneous rate

### 3. Why Approximate Derivatives?

Most real-world functions (loss surfaces, probabilities) aren’t given by simple formulas you can differentiate by hand. Instead, you can only evaluate them at sample points.

Finite-difference methods let you approximate f′(x) from function values at nearby points. These approximations power numerical gradient checks and enable solving differential equations when an analytic solution is unavailable.

### 4. Finite-Difference Formulas

### 4.1 Forward Difference

```
f′(x) ≈ [f(x + h) − f(x)] / h
```

- uses f(x) and f(x + h)
- error term on the order of h

### 4.2 Backward Difference

```
f′(x) ≈ [f(x) − f(x − h)] / h
```

- uses f(x) and f(x − h)
- error term on the order of h

### 4.3 Central Difference

```
f′(x) ≈ [f(x + h) − f(x − h)] / (2 * h)
```

- uses symmetric points around x
- error term on the order of h², making it more accurate

### 5. Step-by-Step Breakdown of Central Difference

```
step 1: compute f(x + h)
step 2: compute f(x − h)
step 3: subtract: numerator = f(x + h) − f(x − h)
step 4: divide by (2 * h)
result: approximate f′(x)
```

Using points on both sides cancels first-order errors, yielding a second-order accurate estimate.

### 6. Real-World Examples

- **Gradient checking** in neural networks: compare analytic gradient to finite-difference approximation to catch coding bugs.
- **Numerical solvers** for ODE-based models (Neural ODEs): approximate derivatives of states when computing next time step.
- **Time series smoothing**: approximate velocity or acceleration from sampled data.

### 7. Python Exercises

```python
import numpy as np

def forward_diff(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h=1e-5):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Test function
f = lambda x: np.sin(x)

x0 = 0.7
print("True derivative:", np.cos(x0))
print("Forward diff :", forward_diff(f, x0))
print("Backward diff:", backward_diff(f, x0))
print("Central diff :", central_diff(f, x0))
```

Plot approximation error versus h to visualize convergence:

```python
import matplotlib.pyplot as plt

hs = np.logspace(-1, -8, 50)
errors = [abs(central_diff(f, x0, h) - np.cos(x0)) for h in hs]

plt.loglog(hs, errors, marker='o')
plt.xlabel('h')
plt.ylabel('Absolute error')
plt.title('Central Difference Error vs Step Size')
plt.show()
```

### 8. Practice Problems

- By hand, approximate the derivative of f(x)=x³ at x=1 using forward, backward, and central differences with h=0.1.
- Implement a gradient checker for a simple linear regression loss J(θ) = mean((Xθ−y)²), comparing analytic gradient to finite differences for each parameter.
- Use central differences to approximate second derivatives and compare to the analytic second derivative for f(x)=e^(−x²).

### 9. Geometric Intuition

Imagine the graph of f(x).

- Forward difference draws a secant to the right of x and measures its slope.
- Backward difference draws a secant to the left.
- Central difference averages both, aligning more closely with the tangent.

As h shrinks, all secant slopes converge to the tangent slope.

---

## Derivatives and Their Notations

### 1. Why Notation Matters

A derivative captures the instantaneous rate of change of one quantity with respect to another. In machine learning and data science, clear notation helps you

- Communicate mathematical ideas unambiguously
- Map abstract formulas to code and algorithms
- Keep track of variables when working with multivariable or time-dependent models

Different contexts favor different notations—knowing them makes you fluent in reading papers, writing proofs, and debugging code.

### 2. Common Derivative Notations

| Notation | Reads As | Used When |
| --- | --- | --- |
| `dy/dx` | “dee y by dee x” | Emphasizing relationship between y and x (Leibniz) |
| `f′(x)` | “f prime of x” | Compact form for single-variable f(x) (Lagrange) |
| `D_x f(x)` or `D f(x)` | “D sub x of f” or “D f of x” | Operator view; functional analysis (Euler) |
| `ẏ` (ẏ) | “y dot” | Time derivative in dynamics, t is independent variable (Newton) |
| `∂y/∂x` | “partial y by partial x” | Multivariable functions y(x, z, …) |
| `∂^2y/∂x^2` | “partial squared y by partial x squared” | Second partial derivative |
| `f″(x), f‴(x), …` | “f double prime,” “f triple prime,” … | Higher-order single-variable derivatives |

### 3. Breakdown of Each Notation

1. Leibniz notation
    
    ```
    dy/dx
    ```
    
    - `d`: stands for a differential (tiny change)
    - `y`, `x`: dependent and independent variables
    - Emphasizes the ratio of infinitesimal changes
2. Lagrange notation
    
    ```
    f′(x)
    ```
    
    - Single quote denotes first derivative
    - Double quote (`f″`) denotes second derivative
3. Euler’s operator notation
    
    ```
    D_x f(x)
    ```
    
    - `D_x`: operator that differentiates with respect to x
    - Useful when applying the same operator to many functions
4. Newton’s dot notation
    
    ```
    ẏ  (ẏ)
    ```
    
    - Dot over a function denotes derivative with respect to time (t)
    - Common in dynamical systems and physics
5. Partial derivative notation
    
    ```
    ∂f/∂x
    ```
    
    - ∂ (curly d) signals differentiation with other variables held constant
    - Second-order:
        
        ```
        ∂²f/∂x²
        ```
        

### 4. Notation Cheat Sheet

| Notation | Formula Code Block | Meaning | Context |
| --- | --- | --- | --- |
| Leibniz | `dy/dx` | Rate of change of y w.r.t. x | Calculus textbook, proofs |
| Lagrange | `f′(x)` | First derivative of f at x | Compact expressions |
| Euler | `D_x f(x)` | Operator form | Functional analysis |
| Newton | `ẏ` | d/dt of y(t) | ODEs, dynamics |
| Partial | `∂f/∂x` | Rate of change in multi-var. context | ML loss w.r.t. one weight |
| Second-order | `d²y/dx²` or `f″(x)` | Curvature or concavity | Maximum/minimum classification |

### 5. Practice Problems

1. Single-variable derivatives
    - Function: f(x) = 3x² + 2x + 1
    - Write derivative in:
        - Leibniz: `dy/dx = …`
        - Lagrange: `f′(x) = …`
        - Euler: `D_x f(x) = …`
    - Compute the expression by hand.
2. Time derivative
    - y(t) = e^(2t)
    - Express and compute ẏ in dot notation and Leibniz notation.
3. Partial derivatives
    - f(x, y) = x²·y + sin(xy)
    - Write ∂f/∂x and ∂f/∂y; compute both.
4. Python exercise (using SymPy)
    
    ```python
    import sympy as sp
    
    # Define symbols
    x, y, t = sp.symbols('x y t')
    f = 3*x**2 + 2*x + 1
    g = sp.exp(2*t)
    h = x**2*y + sp.sin(x*y)
    
    # Compute derivatives
    df_dx_leibniz = sp.diff(f, x)         # dy/dx
    df_dx_lagrange = sp.diff(f, x)        # f'(x)
    dg_dt_dot    = sp.diff(g, t)          # ẏ
    dh_dx_partial= sp.diff(h, x)          # ∂f/∂x
    dh_dy_partial= sp.diff(h, y)          # ∂f/∂y
    
    # Print results
    print("dy/dx  =", df_dx_leibniz)
    print("f'(x)  =", df_dx_lagrange)
    print("ẏ = d/dt of g(t) =", dg_dt_dot)
    print("∂f/∂x  =", dh_dx_partial)
    print("∂f/∂y  =", dh_dy_partial)
    ```
    

### 6. Geometric and Applied Intuition

- Any derivative notation—dy/dx, f′(x), ẏ—ultimately points to the slope of the tangent line on a graph.
- In ML:
    - `∂Loss/∂wᵢ` tells how steeply the loss changes if you tweak weight wᵢ.
    - `ẋ(t)` in continuous-time models shows how the hidden state evolves over time.
- Mastering notation helps you read research papers, follow backpropagation derivations, and implement gradients correctly in code.

---

## Common Derivatives of Line Functions

### 1. Conceptual Overview

A line is the simplest non-constant function in calculus. It has the form

```
f(x) = m * x + b
```

- **m** is the slope: how steeply the line rises (if positive) or falls (if negative).
- **b** is the y-intercept: where the line crosses the vertical axis.

Since a line’s steepness never changes, its derivative is constant everywhere. In other words, its tangent line at any point is the line itself.

### 2. Key Formulas

### 2.1 Derivative of a Linear Function

```
f(x) = m * x + b
f′(x) = m
```

- `f(x)`: original line.
- `m * x`: slope times input.
- `b`: constant offset.
- `f′(x) = m`: the rate of change is always m, regardless of x.

### 2.2 Derivative of a Constant Function

```
g(x) = c
g′(x) = 0
```

- `c`: any fixed number.
- Since g(x) never changes, its instantaneous rate of change is zero.

### 3. Step-by-Step Explanation

1. Start with
    
    ```
    f(x) = m * x + b
    ```
    
2. Apply the limit definition:
    
    ```
    f′(x) = limit as h→0 of [f(x+h) − f(x)] / h
          = limit as h→0 of [m*(x+h)+b − (m*x+b)] / h
          = limit as h→0 of [m*x + m*h + b − m*x − b] / h
          = limit as h→0 of [m*h] / h
          = limit as h→0 of m
          = m
    ```
    
3. No dependence on x remains, so the derivative is the constant m.

### 4. Real-World ML/DS Use Cases

- **Linear Regression Gradient**
    
    In simple linear regression, your model is
    
    ```
    y_pred = w * x + b
    ```
    
    The derivative of y_pred w.r.t. w is x (not constant), but the derivative w.r.t. b is 1.
    
    When computing the gradient of the mean squared error
    
    ```
    L = (1/n) ∑(y_pred − y_true)²
    ```
    
    the update rule for b uses that constant derivative 1, simplifying the sum.
    
- **First-Layer Weights in a Neural Network**
    
    A dense layer computes
    
    ```
    z = W x + b
    ```
    
    Its Jacobian w.r.t. b is a vector of ones: ∂z/∂b = 1. This constant structure speeds up backpropagation.
    

### 5. Practice Problems

1. By hand, find the derivative of:
    - `f(x) = 5x + 3`
    - `f(x) = −2x + 7`
    - `g(x) = 42` (constant function)
2. Confirm using finite differences for `f(x) = 5x + 3` at x = 10 with h = 1e-5.
3. In the linear regression loss
    
    ```
    J(w, b) = (1/n) ∑(w * x_i + b − y_i)²
    ```
    
    derive the partial derivative ∂J/∂b using the fact that d(b)/db = 1.
    

### 6. Python Exercises

```python
import numpy as np

# Define a line
m, b = 5.0, 3.0
f = lambda x: m * x + b

# Analytical derivative
print("Analytical f'(x):", m)

# Finite-difference approximation
def central_diff(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

for x0 in [0, 1, 10, -5]:
    approx = central_diff(f, x0)
    print(f"x0={x0:>2}, central_diff={approx:.5f}")
```

```python
# Linear regression example
# y = w*x + b, loss = mean squared error
X = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])  # true relationship: y=2*x+1
w, b = 0.0, 0.0
alpha = 0.1

for epoch in range(10):
    y_pred = w * X + b
    error = y_pred - y
    # gradient w.r.t. b uses derivative of (b) which is 1
    grad_b = (2/len(X)) * error.sum() * 1
    b -= alpha * grad_b
print("Updated b:", b)
```

### 7. Geometric Interpretation

Everywhere on the line `f(x) = m*x + b`, the tangent coincides with the line itself.

- Draw two points `(x, f(x))` and `(x+h, f(x+h))`.
- The secant slope is always `(m*(x+h)+b − (m*x+b)) / h = m`.
- As h→0, the secant becomes the tangent, still with slope = m.

---

## Common Derivatives of Quadratic Functions

### 1. Conceptual Overview

A quadratic function is a simple curve defined by

```
f(x) = a*x^2 + b*x + c
```

where a, b, and c are constants.

Its graph is a parabola opening upward if a > 0 or downward if a < 0. The steepness of the parabola changes with x, so its derivative is a linear function.

### 2. Formula for the Derivative

```
f(x) = a*x^2 + b*x + c
f′(x) = 2*a*x + b
```

- a*x^2: when you differentiate, the exponent 2 multiplies a, giving 2*a*x
- b*x: the derivative of b*x is b
- c: derivative of a constant is 0

Thus the instantaneous rate of change at any x is the line 2*a*x + b.

### 3. Derivation via the Limit Definition

```
f′(x) = lim(h → 0) [f(x + h) − f(x)] / h
       = lim(h → 0) [(a*(x+h)^2 + b*(x+h) + c)
                     − (a*x^2 + b*x + c)] / h
       = lim(h → 0) [a*(x^2+2xh+h^2) + b*x + b*h + c
                     − a*x^2 − b*x − c] / h
       = lim(h → 0) [2*a*x*h + a*h^2 + b*h] / h
       = lim(h → 0) [2*a*x + a*h + b]
       = 2*a*x + b
```

Breaking it down:

- Expand (x+h)^2 to x^2 + 2xh + h^2
- Cancel x^2, b*x, and c terms
- Factor out h, divide, then let h go to zero

### 4. Real-World ML/DS Use Cases

- Polynomial regression uses quadratics to fit curved trends in data.
- Feature engineering can add x^2 features; derivatives tell how sensitive predictions are to those features.
- In loss surfaces, a quadratic approximation (Taylor’s theorem) captures curvature near a minimum.
- Second-order optimization methods (Newton’s method) use second derivative (2*a) to adjust step sizes.

### 5. Practice Problems

1. By hand compute f′(x) for:
    - f(x) = 4x^2 + 3x − 5
    - f(x) = −2x^2 + 7x + 1
2. Find critical point of f(x) = 3x^2 − 12x + 4 by solving f′(x) = 0 and classify it via second derivative.
3. Numerically approximate f′(2) for f(x)=5x^2+2x+3 using central difference with h=0.01.

### 6. Python Exercises

```python
import numpy as np
import matplotlib.pyplot as plt

# Quadratic parameters
a, b, c = 2.0, -3.0, 1.0
f = lambda x: a*x**2 + b*x + c
df = lambda x: 2*a*x + b

# Plot function and its derivative
x_vals = np.linspace(-5, 5, 200)
plt.plot(x_vals, f(x_vals), label='f(x) = 2x^2 − 3x + 1')
plt.plot(x_vals, df(x_vals), '--', label="f'(x) = 4x − 3")
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.xlabel('x')
plt.title('Quadratic Function and Its Derivative')
plt.show()

# Find vertex (critical point)
x_vertex = -b / (2*a)
print("Vertex at x =", x_vertex, "f(x) =", f(x_vertex))
```

### 7. Geometric Interpretation

- The parabola’s slope increases linearly as you move away from the vertex.
- At x = −b/(2a) the slope 2*a*x + b is zero—this is the vertex (minimum or maximum).
- Plotting f and f′ together shows the parabola and the straight line of its instantaneous slopes.

---

## Common Derivatives of Higher-Degree Polynomials

### 1. Conceptual Overview

A polynomial of degree n has the form

```
f(x) = a_n*x^n + a_{n-1}*x^{n-1} + … + a_1*x + a_0
```

Each term a_k*x^k contributes to the overall shape. The derivative measures how steep f is at any x, and for polynomials the slope itself is again a polynomial of one degree lower.

### 2. The Power-Rule Generalization

**Power Rule:** For any term a*x^k,

```
d/dx [a*x^k] = a * k * x^(k-1)
```

Applying this to every term gives the derivative of the whole polynomial:

```
f′(x) = n*a_n*x^(n-1)
      + (n-1)*a_{n-1}*x^(n-2)
      + …
      + 1*a_1
      + 0
```

### 3. Step-by-Step Breakdown

1. **Identify each term** a_k*x^k.
2. **Multiply** the coefficient a_k by the exponent k.
3. **Reduce** the exponent by 1.
4. **Drop** constant terms (exponent 0) since their derivative is 0.

Example for f(x)=5x³−4x²+2x−7:

```
f′(x) = 5*3*x^(3-1) − 4*2*x^(2-1) + 2*1*x^(1-1) − 0
      = 15*x^2 − 8*x + 2
```

### 4. Real-World ML/DS Applications

- **Polynomial regression:** Fit data with curves; derivative predicts instantaneous slope of fitted model for sensitivity analysis.
- **Taylor series:** Approximate complex functions by polynomials; derivatives supply coefficients for linear and higher-order terms.
- **Feature engineering:** When you add x², x³ features, their derivatives inform how model output changes with those engineered features.
- **Loss surface analysis:** Near a solution, loss can be approximated by a quadratic or cubic; derivatives help design second-order optimization.

### 5. Worked Examples

| Polynomial | f′(x) |
| --- | --- |
| x³ − 6x² + x − 2 | 3x² − 12x + 1 |
| 4x⁴ + x³ − 3x + 5 | 16x³ + 3x² − 3 |
| −2x⁵ + 7x² + 4 | −10x⁴ + 14x |

### 6. Practice Problems

- By hand, compute derivatives of:
    1. f(x) = 2x⁵ − 3x³ + x
    2. g(x) = −x⁴ + 4x² − x + 10
    3. h(x) = 7
- Find and classify critical points of f(x)=x³−3x²+2x.
- Numerically approximate f′(1.5) for f(x)=x⁴−2x³+3x using central difference with h=0.001.

### 7. Python Exercises

```python
import numpy as np
import matplotlib.pyplot as plt

# Define polynomial and its derivative
coeffs = [2, -3, 0, 1, 0]             # represents 2x^4 - 3x^3 + 0x^2 + 1x + 0
f      = lambda x: np.polyval(coeffs, x)
df     = lambda x: np.polyval(np.polyder(coeffs), x)

# Plot both
x_vals = np.linspace(-2, 3, 300)
plt.plot(x_vals, f(x_vals), label='f(x)')
plt.plot(x_vals, df(x_vals), '--', label="f'(x)")
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.title('Polynomial and Its Derivative')
plt.show()
```

```python
# Gradient check for simple polynomial loss J(w) = (w^3 - 2w^2 + w - 5)^2
def loss(w): return (w**3 - 2*w**2 + w - 5)**2
def grad_analytic(w):
    # derivative of inside g(w)=w^3-2w^2+w-5 is 3w^2-4w+1, so dJ/dw=2*g(w)*g'(w)
    g = w**3 - 2*w**2 + w - 5
    gp = 3*w**2 - 4*w + 1
    return 2*g*gp

def grad_numeric(w, h=1e-5):
    return (loss(w+h)-loss(w-h)) / (2*h)

for w in [0, 1, 2]:
    print(w, grad_analytic(w), grad_numeric(w))
```

### 8. Geometric Interpretation

- A degree-n polynomial’s slope curve is degree n−1.
- Zeros of f′(x) mark where f(x) has peaks, valleys, or inflection if multiplicity >1.
- Plotting f and f′ together reveals how the sign of slope changes across each critical point.

---

## Common Derivatives – Other Power Functions

### 1. Conceptual Overview

Beyond integer-powered polynomials, many models and feature transforms use fractional or negative exponents.

Examples:

- Root transforms (√x, ³√x) to reduce skew in features.
- Reciprocal features (1/x) to capture diminishing returns.
- Box–Cox or custom power transforms xᵖ for p∈ℝ.

The **general power rule** handles all these cases in one stroke.

### 2. General Power Rule

```
f(x) = x^n
f′(x) = n * x^(n - 1)
```

- n can be any real number (integer, fraction, or negative).
- The derivative is a scaled power function of one degree lower.

### Derivation Sketch

1. Start with the limit definition for f(x)=xⁿ.
2. Use the generalized binomial expansion of (x+h)ⁿ.
3. Cancel terms and factor h.
4. Let h→0 to obtain n·xⁿ⁻¹.

### 3. Specific Examples

| f(x) | Code Block | f′(x) Code Block |
| --- | --- | --- |
| Square root | `x^(1/2)` | `(1/2) * x^(-1/2)` |
| Cube root | `x^(1/3)` | `(1/3) * x^(-2/3)` |
| Reciprocal | `x^(-1)` | `-1 * x^(-2)` |
| Reciprocal square | `x^(-2)` | `-2 * x^(-3)` |
| Fractional power | `x^(2/3)` | `(2/3) * x^(-1/3)` |
| Constant multiple | `a * x^n` | `a * n * x^(n-1)` |

### 4. Worked Example: √x

Function: f(x)=√x = x^(1/2)

1. Apply power rule:
    
    ```
    f′(x) = (1/2) * x^((1/2) - 1)
          = (1/2) * x^(-1/2)
    ```
    
2. Interpret: slope = 1/(2√x). As x grows, slope flattens; near zero, slope →∞.

### 5. Real-World ML/DS Uses

- **Feature engineering**: Transform skewed features via x^(1/2) or x^(-1).
- **Loss scaling**: Huber loss uses piecewise powers |r|ᵖ with p=1 or 2.
- **Box–Cox**: Find optimal p by differentiating a likelihood in power-transformed space.
- **Neural nets**: Activation penalties sometimes involve fractional powers (e.g., Lₚ-norms).

### 6. Practice Problems

1. By hand, compute derivatives:
    - f(x)=x^(3/5)
    - g(x)=x^(-3/2)
2. Find f′(4) for f(x)=x^(2/3).
3. Derive ∂/∂x of h(x)=5 * x^(−1) + 2 * x^(3/2).

### 7. Python Exercises

```python
import numpy as np

def deriv_power(x, n):
    return n * x**(n-1)

# Test a variety of powers
xs = np.array([0.5, 1.0, 2.0, 4.0])
powers = [0.5, 1/3, -1, -2, 2/3]

for p in powers:
    print(f"n={p}: f'(x)=", deriv_power(xs, p))
```

Compare to central difference:

```python
def central_diff(f, x, h=1e-6):
    return (f(x+h) - f(x-h)) / (2*h)

for p in powers:
    f = lambda x: x**p
    approx = central_diff(f, 2.0)
    exact  = deriv_power(2.0, p)
    print(f"p={p}, approx {approx:.6f}, exact {exact:.6f}")
```

### 8. Geometric Intuition

- **Fractional exponents** yield curves that grow slower than linears. Their slopes decrease faster.
- **Negative exponents** produce hyperbolas; slopes are steep near x=0 and flatten out for large x.
- The tangent at any x follows the same power-shape scaled by n.

---

## The Inverse Function and Its Derivative

### 1. What Is an Inverse Function?

An inverse function “undoes” what the original function does.

If you have a function

```
y = f(x)
```

then its inverse, denoted

```
x = f⁻¹(y)
```

satisfies

```
f(f⁻¹(y)) = y   and   f⁻¹(f(x)) = x
```

Think of f as a lock that takes key x and produces a code y. The inverse f⁻¹ is the unlock process that recovers x from y.

### 2. Conditions for Invertibility

- **One-to-one (injective):** Each x produces a unique y.
- **Onto (surjective):** Every y in the target is produced by some x.
- **Differentiable:** f′(x) exists and is nonzero on its domain, ensuring local invertibility.

When these hold, f is *bijective* and f⁻¹ exists and is smooth.

### 3. Notation

- Original function:
    
    ```
    y = f(x)
    ```
    
- Inverse function:
    
    ```
    x = f⁻¹(y)
    ```
    
- Derivative of the inverse:
    
    ```
    (f⁻¹)′(y)
    ```
    

### 4. Formula for the Derivative of an Inverse

When f is invertible and differentiable at x with f′(x) ≠ 0, then at y = f(x):

```
(f⁻¹)′(y) = 1 / [ f′( x ) ]
          = 1 / [ f′( f⁻¹(y) ) ]
```

```
# Copy-paste version
(f_inverse)'(y) = 1 / f'(f_inverse(y))
```

### Breakdown

- `f⁻¹(y)`: the x-value that maps to y under f.
- `f′(f⁻¹(y))`: slope of f at that x.
- Taking its reciprocal gives the slope of the inverse at y.

### 5. Derivation via Implicit Differentiation

1. Start with y = f(x).
2. Assume x = f⁻¹(y).
3. Differentiate both sides w.r.t. y:
    
    ```
    d/dy [ y ] = d/dy [ f( f⁻¹(y) ) ]
    1         = f′( f⁻¹(y) ) * (f⁻¹)′(y)
    ```
    
4. Solve for (f⁻¹)′(y):
    
    ```
    (f⁻¹)′(y) = 1 / f′( f⁻¹(y) )
    ```
    

### 6. Real-World ML/DS Use Cases

- **Logistic & Logit:**
    - f(x) = 1 / (1 + e^(−x)) → f⁻¹(y) = logit(y) = log( y / (1−y) )
    - Derivative of logit:
        
        ```
        (logit)'(y) = 1 / [ y*(1−y) ]
        ```
        
    - Crucial for transforming probabilities to log-odds in classification.
- **Normalizing Flows:**
    - Build complex distributions by applying invertible transforms whose Jacobian determinants (products of derivatives) are needed for density evaluation.
- **Feature Scaling & Whitening:**
    - PCA whitening applies a linear invertible transform; its inverse reconstructs original data.
- **Box–Cox Transform:**
    - Power transform f(x)= (x^λ −1)/λ has inverse f⁻¹(y)= (λ*y +1)^(1/λ). Its derivative appears in likelihood calculations.

### 7. Practice Problems

1. **Linear function**
    - f(x) = a*x + b
    - Inverse: f⁻¹(y) = (y − b)/a
    - Show (f⁻¹)′(y) = 1/a using the formula and by directly differentiating.
2. **Cubic function**
    - f(x) = x³
    - Inverse: f⁻¹(y) = y^(1/3)
    - Compute (f⁻¹)′(y) = (1/3)*y^(−2/3) and compare to 1 / f′(f⁻¹(y)).
3. **Logistic–logit pair**
    - f(x) = 1/(1+e^(−x)), y in (0,1)
    - f⁻¹(y) = log(y/(1−y))
    - Verify (f⁻¹)′(y) = 1/[y*(1−y)].

### 8. Python Exercises

```python
import numpy as np

# Example: f(x)=x**3
f      = lambda x: x**3
f_inv  = lambda y: np.sign(y) * np.abs(y)**(1/3)
f_prime = lambda x: 3*x**2

# Test points
ys = np.array([-8.0, -1.0, 0.0, 1.0, 8.0])

# Analytic derivative of inverse
d_inv_analytic = 1 / f_prime( f_inv(ys) )

# Numerical derivative
h = 1e-5
d_inv_numeric = (f_inv(ys+h) - f_inv(ys-h)) / (2*h)

for y, da, dn in zip(ys, d_inv_analytic, d_inv_numeric):
    print(f"y={y}, analytic={da:.5f}, numeric={dn:.5f}")
```

### 9. Geometric Interpretation

- Graph of f⁻¹ is the reflection of f across the line y = x.
- A steep slope in f (large f′) becomes a shallow slope (small derivative) in f⁻¹, since slope(f⁻¹) = 1/slope(f).
- At points where f′→0, the inverse’s slope →∞, indicating a vertical tangent.

---

## Derivative of Trigonometric Functions

### 1. Conceptual Overview

Trigonometric functions model periodic, wave-like patterns.

Their derivatives describe how those waves rise and fall at any instant.

In machine learning and data science, trig functions appear in:

- Signal processing (Fourier features)
- Time-series modeling (seasonality)
- Coordinate transforms (rotations)
- Activation functions (periodic kernels)

Understanding their derivatives is key for backpropagation through such components.

### 2. Fundamental Trigonometric Derivatives

| Function | Derivative Code Block | Breakdown |
| --- | --- | --- |
| sine | `d/dx [sin(x)] = cos(x)` | slope of sin is cos |
| cosine | `d/dx [cos(x)] = -sin(x)` | slope of cos is -sin |
| tangent | `d/dx [tan(x)] = sec^2(x)` | slope of tan is sec squared |
| cotangent | `d/dx [cot(x)] = -csc^2(x)` | slope of cot is -csc squared |
| secant | `d/dx [sec(x)] = sec(x)*tan(x)` | product of sec and tan |
| cosecant | `d/dx [csc(x)] = -csc(x)*cot(x)` | negative product of csc and cot |

### 3. Breakdown of Key Formulas

1. **sin → cos**
    
    ```
    d/dx [sin(x)] = cos(x)
    ```
    
    - sin(x) rises most steeply when x=0; cos(0)=1.
    - At peaks of sin, slope is zero because cos(π/2)=0.
2. **cos → –sin**
    
    ```
    d/dx [cos(x)] = -sin(x)
    ```
    
    - cos starts at 1 with zero slope because sin(0)=0.
    - Falling part of cos has negative slope.
3. **tan → sec²**
    
    ```
    d/dx [tan(x)] = sec(x)^2
    ```
    
    - tan(x)=sin(x)/cos(x). Use quotient rule to get sec²(x).
4. **sec → sec·tan**
    
    ```
    d/dx [sec(x)] = sec(x) * tan(x)
    ```
    
    - sec(x)=1/cos(x). Differentiate via reciprocal rule.

### 4. Geometric Intuition on the Unit Circle

- At any angle x, imagine a point moving around the unit circle.
- The vertical velocity of that point is cos(x)—the derivative of sin.
- The horizontal velocity (negative) is −sin(x)—the derivative of cos.
- For tan, the slope of the line from center to the circle’s tangent point grows with sec²(x).

### 5. Real-World ML/DS Use Cases

- **Fourier feature mapping:** Embedding x → [sin(wx), cos(wx)] for kernel machines; derivatives tune frequency w.
- **Seasonal time-series:** Backpropagate through sin/cos seasonal terms in RNNs and forecasting models.
- **Coordinate systems:** Rotational dynamics in physics-inspired models require sin/cos gradients.
- **Periodic regularizers:** Add sin/cos penalty terms; need their derivatives for optimization.

### 6. Practice Problems

1. By hand compute:
    - d/dx [sin(3x)]
    - d/dx [cos(5x + π/4)]
    - d/dx [tan(2x) + 3x]
2. Find second derivatives:
    - d²/dx² [sin(x)]
    - d²/dx² [cos(x)]
3. Show that derivative of tan(x) = sin(x)/cos(x) yields sec²(x) via quotient rule.

### 7. Python Exercises

```python
import numpy as np

# Define functions
f_sin = lambda x: np.sin(x)
f_cos = lambda x: np.cos(x)
f_tan = lambda x: np.tan(x)

# Analytical derivatives
df_sin = lambda x: np.cos(x)
df_cos = lambda x: -np.sin(x)
df_tan = lambda x: 1/np.cos(x)**2

# Numeric central difference
def central_diff(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

# Test at x0
x0 = 0.7
print("sin' analytical:", df_sin(x0), "numeric:", central_diff(f_sin, x0))
print("cos' analytical:", df_cos(x0), "numeric:", central_diff(f_cos, x0))
print("tan' analytical:", df_tan(x0), "numeric:", central_diff(f_tan, x0))
```

Plot function and derivative:

```python
import matplotlib.pyplot as plt

xs = np.linspace(-1.2, 1.2, 400)
plt.plot(xs, np.sin(xs), label='sin(x)')
plt.plot(xs, np.cos(xs), label='cos(x)')
plt.plot(xs, df_sin(xs), '--', label="sin'(x)=cos(x)")
plt.plot(xs, df_cos(xs), '--', label="cos'(x)=-sin(x)")
plt.legend()
plt.title('Trig Functions and Derivatives')
plt.show()
```

---

## Meaning of the Exponential Base (e)

### 1. What Is e?

An “exponential” grows (or decays) at a rate proportional to its current value. The constant e ≈ 2.71828 is the unique base that makes this growth rate equal to 1 at every point.

You encounter e in:

- Continuous compounding of interest
- Natural growth and decay processes
- The definition of the natural logarithm

### 2. Formal Definitions

### 2.1 Limit Definition

```
e = limit as n → ∞ of (1 + 1/n)^n
```

- As you compound 100%, 1/n times per period, the amount converges to e.

### 2.2 Series Definition

```
e = sum from k=0 to ∞ of 1 / k!
```

- k! is k × (k−1) × … × 1, with 0! defined as 1.
- Summing 1 + 1 + 1/2 + 1/6 + 1/24 + … converges to e.

### 3. Key Properties

- **Derivative of eˣ** remains eˣ, meaning its instantaneous growth rate equals its value.
- **Integral of 1/x** is ln|x|, the natural logarithm to base e.
- **Inverse relationship:** logₑ(x) (written as ln x) reverses the exponential.

```
d/dx [ e^x ] = e^x

∫ (1 / x) dx = ln(x) + C
```

### 4. Geometric Intuition

On the curve y = eˣ:

- The slope at any x is exactly y.
- If you zoom in at any point, the tangent line has the same shape of the function itself.
- This self-similarity underlies why e shows up in so many continuous processes.

### 5. Real-World ML/DS Uses

- **Logistic function:** σ(x) = 1 / (1 + e^(−x)) for binary classification.
- **Softmax:** e^(zᵢ) / Σ_j e^(zⱼ) for multi-class probability distributions.
- **Activation and gating:** e^(x) ensures smooth, non-negative transforms.
- **Probability distributions:** e^(−x²/2) in Gaussian density functions.
- **Loss functions:** e.g., exponential loss in boosting algorithms.

### 6. Practice Problems

1. By hand, show that
    
    ```
    limit as n→∞ of (1 + 1/n)^n = limit as h→0 of (1 + h)^(1/h)
    ```
    
2. Compute a partial sum of the series
    
    ```
    S_N = sum_{k=0}^N 1 / k!
    ```
    
    for N = 5, 10, 15 and compare to e.
    
3. Implement continuous compound growth in Python:
    
    ```python
    def continuous_compound(principal, rate, time, steps):
        dt = time / steps
        amount = principal
        for _ in range(steps):
            amount *= (1 + rate * dt)
        return amount
    
    # Compare for principal=100, rate=5%, time=1, as steps→∞
    ```
    
4. Plot y = eˣ and its tangent at x = 0 via Python and verify the slope equals 1.

### 7. Python Example: Approximating e

```python
import math

# Series approximation
def approx_e(N):
    return sum(1 / math.factorial(k) for k in range(N+1))

for N in [5, 10, 15, 20]:
    print(N, approx_e(N), "error:", abs(math.e - approx_e(N)))
```

---

## The Derivative of eˣ

### 1. Conceptual Understanding

When you take the derivative of eˣ, you ask: “If I increase x by a tiny amount, by how much does eˣ change?”

Because the base e is defined so that its rate of change equals its own value at every point, the slope of the curve y = eˣ at x is exactly eˣ itself.

### 2. Formula and Line-by-Line Breakdown

```
d/dx [ e^x ] = e^x
```

- `d/dx`: operator meaning “take the derivative with respect to x.”
- `e^x`: the function whose derivative we seek.
- `=`: the result of differentiation.
- `e^x`: shows that the slope at x is the same as the function’s value at x.

### 3. Proof via Limit Definition

```
d/dx [ e^x ]
  = limit as h → 0 of [ e^(x + h) − e^x ] / h
  = limit as h → 0 of [ e^x * e^h − e^x ] / h
  = limit as h → 0 of [ e^x * (e^h − 1) ] / h
  = e^x * limit as h → 0 of [ (e^h − 1) / h ]
  = e^x * 1
  = e^x
```

Step-by-step:

1. Factor out eˣ, since e^(x+h)=eˣ·eʰ.
2. Recognize the remaining limit limₕ→0 (eʰ−1)/h equals 1 by the definition of e.
3. Multiply back by eˣ.

### 4. Geometric Intuition

- The graph of y = eˣ is always rising.
- At each point (x, eˣ), draw the tangent line.
- Its slope equals the height eˣ: the curve “climbs” in proportion to where it is.
- Zoom in anywhere and the curve looks like its own tangent.

### 5. Real-World ML/DS Applications

- **Softmax and probabilities:** when computing gradients of softmax outputs, you differentiate e^(zᵢ).
- **Exponential moving averages:** smoothing signals uses update rules whose continuous analog involves derivatives of e^(−λt).
- **Neural ODEs:** hidden states evolve according to dh/dt = f(h), often linearized as dh/dt = a·h giving solutions h(t)=h₀·e^(a t).
- **Loss functions:** exponential loss in boosting algorithms L = exp(−y·f(x)) differentiates to L′ = −y·exp(−y·f(x))·f′(x).

### 6. Practice Problems

1. By hand, show that d/dx [5·e^(3x)] = 15·e^(3x).
2. Compute the derivative of f(x) = e^(2x) + 4·e^(−x).
3. Use central differences to approximate d/dx [e^x] at x = 2 with h = 1e-5 and compare to the analytical value.

### 7. Python Exercises

```python
import numpy as np

# Define function and analytic derivative
f  = lambda x: np.exp(x)
df = lambda x: np.exp(x)

# Numerical derivative (central difference)
def numeric_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

# Test at various x
for x0 in [0.0, 1.0, 2.0, -1.0]:
    print(f"x={x0}, analytic={df(x0):.6f}, numeric={numeric_derivative(f, x0):.6f}")
```

Plot the function and its tangent at x₀:

```python
import matplotlib.pyplot as plt

x0 = 1.0
y0 = np.exp(x0)
slope0 = np.exp(x0)
tangent = lambda x: y0 + slope0 * (x - x0)

x_vals = np.linspace(-2, 4, 300)
plt.plot(x_vals, f(x_vals), label='e^x')
plt.plot(x_vals, tangent(x_vals), '--', label='tangent at x=1')
plt.scatter([x0], [y0], color='red')
plt.legend()
plt.title('Function e^x and Its Tangent Line at x=1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

---

## Derivative of log(x)

### 1. Conceptual Overview

The natural logarithm ln(x) measures how many times you multiply e (≈2.718) to get x.

Its derivative tells you how the log’s output changes when x changes by a tiny amount.

In data science, ln(x) appears in log‐likelihoods, feature scaling, cross-entropy losses, and information measures.

### 2. Key Formula

```
d/dx [ ln(x) ] = 1 / x
```

- ln(x): the natural logarithm of x.
- d/dx: derivative with respect to x.
- 1/x: the instantaneous rate of change.

For a general base-b logarithm log_b(x):

```
d/dx [ log_b(x) ] = 1 / [ x * ln(b) ]
```

### 3. Step-by-Step Derivation via Implicit Differentiation

1. Start with y = ln(x).
2. Exponentiate both sides: e^y = x.
3. Differentiate w.r.t. x:
    
    ```
    d/dx [ e^y ] = d/dx [ x ]
    e^y * (dy/dx) = 1
    ```
    
4. Solve for dy/dx:
    
    ```
    dy/dx = 1 / e^y
          = 1 / x
    ```
    

Thus d/dx[ln(x)] = 1/x.

### 4. Geometric Intuition

- Graph of y = ln(x) is increasing and concave down.
- At any point (x₀, ln(x₀)), the tangent line has slope 1/x₀.
- As x grows, 1/x shrinks → the curve flattens out for large x.
- Near x=0⁺, 1/x→∞ → the curve shoots upward steeply.

### 5. Real-World ML/DS Applications

- **Log-likelihoods:** Maximizing ln p(data|θ) uses derivative 1/x to update parameters.
- **Cross-entropy loss:** d/dz [−y·ln(σ(z))] involves 1/σ(z) in backprop.
- **Feature transforms:** Taking ln of skewed features makes distributions more Gaussian; gradient-guided algorithms adjust features accordingly.
- **Information gain:** Entropy H = −∑p·ln(p); derivative −ln(p) −1 appears in decision-tree splits.

### 6. Practice Problems

1. By hand, compute d/dx [ ln(3x + 2) ].
2. Find the derivative of f(x) = ln(x² + 1).
3. Derive d/dx [ log₁₀(x) ] using the change-of-base formula.
4. Use finite differences to approximate d/dx[ln(x)] at x=2 with h=1e-5 and compare to 1/2.

### 7. Python Exercises

```python
import numpy as np

# Analytic derivative
dln = lambda x: 1/x

# Numeric central difference
def central_diff(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

f = np.log
x0 = 2.0

print("Analytic:", dln(x0))
print("Numeric :", central_diff(f, x0))
```

```python
import matplotlib.pyplot as plt

# Plot ln(x) and its tangent at x0
x0 = 2.0
y0 = np.log(x0)
slope = 1/x0
tangent = lambda x: y0 + slope * (x - x0)

xs = np.linspace(0.1, 5, 300)
plt.plot(xs, np.log(xs), label='ln(x)')
plt.plot(xs, tangent(xs), '--', label=f'tangent at x={x0}')
plt.scatter([x0], [y0], color='red')
plt.legend()
plt.title('ln(x) and Tangent Line')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

---

## Existence of Derivatives

### 1. What It Means for a Derivative to Exist

A function f is differentiable at x₀ if the limit that defines its derivative actually converges to a single finite number.

```
f′(x₀) = limit as h → 0 of [f(x₀ + h) − f(x₀)] / h
```

If that limit exists (same from left and right), we say f has a well-defined slope at x₀. If the limit fails—because left and right differ, blow up, or oscillate—then f is not differentiable there.

### 2. Differentiability vs. Continuity

- Differentiability implies continuity.If f′(x₀) exists, f must be continuous at x₀.
- Continuity does **not** imply differentiability.A function can be continuous yet have sharp corners or cusps where no tangent exists.

**Example:** f(x)=|x| is continuous everywhere but not differentiable at x=0 because slopes from left (-1) and right (+1) disagree.

### 3. Common Causes of Non-Existence

1. **Corners or Cusps**Sharp angle yields different one-sided slopes (absolute value, ReLU).
2. **Vertical Tangents**Slope → ±∞ (e.g., f(x)=x^(1/3) at x=0).
3. **Oscillations**Highly oscillatory behavior can prevent a single limit (Weierstrass function).
4. **Discontinuities**Jump breaks kill derivative at that point.

### 4. Theoretical Conditions

- If f′ exists on an open interval, f is continuous there.
- If f′ is continuous on [a, b], then f satisfies the Mean Value Theorem.
- Lipschitz-continuity (|f(x)−f(y)| ≤ K|x−y|) does not guarantee differentiability everywhere, but rules out infinite oscillations.

### 5. Real-World ML/DS Implications

- **ReLU activation:** f(x)=max(0,x) is non-differentiable at 0. In practice we pick a subgradient.
- **Absolute-error loss:** |y_pred−y_true| has a corner at zero error; optimization uses subgradients.
- **Numerical differentiation:** step size h must be small enough to approximate but not so small that noise or floating-point error dominates.

### 6. Practice Problems

1. By hand determine where each is differentiable:
    
    a. f(x)=|x|
    
    b. f(x)=x^(1/3)
    
    c. f(x)=if x≥0 then x² else −x
    
2. Show continuity but non-differentiability for f(x)=x·sin(1/x) at x=0 by evaluating one-sided difference quotients.
3. Prove that if f′ exists at x₀ then limₓ→ₓ₀ f(x)=f(x₀).

### 7. Python Exercises

```python
import numpy as np

def approx_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Test functions
functions = {
    'abs': abs,
    'cuberoot': lambda x: np.sign(x)*abs(x)**(1/3),
    'oscillate': lambda x: x*np.sin(1/x) if x!=0 else 0
}

xs = np.linspace(-0.01, 0.01, 21)
for name, func in functions.items():
    slopes = [approx_derivative(func, x) for x in xs]
    print(name, "approx slopes near 0:", np.round(slopes, 3))
```

Use the output to spot where the derivative “jumps” or spikes—indicating non-existence or vertical tangents.

### 8. Visual Description

- **Absolute value:** V-shaped graph; at the corner the two tangent directions do not match.
- **Cube root:** S-shaped but with a vertical tangent at 0 (slope →∞).
- **Oscillatory:** infinitely many small wiggles near 0 prevent settling on one tangent.

---

## Properties of the Derivative – Multiplication by a Scalar

### Conceptual Overview

When you multiply a function by a constant (scalar), you simply stretch or shrink its graph vertically.

The rule for derivatives tells us that this vertical scaling carries through to the slope: if you scale the function by c, you scale every tangent slope by the same factor.

### Key Formula

```
d/dx [ c * f(x) ] = c * f′(x)
```

- `c`: constant scalar
- `f(x)`: original function
- `f′(x)`: derivative of f at x
- `d/dx`: the differentiation operator

Multiplying f(x) by c multiplies its derivative by c as well.

### Line-by-Line Breakdown

1. Start with the scaled function:
    
    ```
    y = c * f(x)
    ```
    
2. Apply the limit definition of the derivative:
    
    ```
    y′ = lim(h→0) [c*f(x+h) − c*f(x)] / h
       = c * lim(h→0) [f(x+h) − f(x)] / h
       = c * f′(x)
    ```
    
3. Factor out c (since it doesn’t depend on x or h) before taking the limit.

### Geometric Intuition

- The graph of y = f(x) is stretched vertically by c to become y = c·f(x).
- Every tangent line’s slope is c times steeper (or shallower if 0<c<1) than before.
- If c is negative, the graph flips over the x-axis and slopes reverse sign.

### Real-World ML/DS Applications

- **Learning-rate scaling:** In gradient descent, you multiply the gradient by the learning rate α (a scalar). The update ruleuses this property: scaling the loss gradient by α scales the parameter step by α.
    
    ```
    θ_new = θ_old − α * ∇J(θ_old)
    ```
    
- **Loss weighting:** When combining multiple losses (e.g., L_total = λ₁L₁ + λ₂L₂), each loss’s gradient is weighted by its λ.
- **Feature scaling:** If you multiply an input feature by c, the derivative of your prediction with respect to that feature also scales by c.

### Practice Problems

1. By hand, differentiate each:
    - d/dx [5 * x²]
    - d/dx [−3 * sin(x)]
    - d/dx [2 * e^(3x)]
2. Given L(w) = 0.5 * (w³ − 2w + 1), find dL/dw using the scalar-multiplication property.
3. If g(x) = x³ and h(x) = 4·g(x), show h′(x) via both the scalar rule and direct expansion.

### Python Exercises

```python
import numpy as np

# Define f(x) and scalar c
f  = lambda x: np.sin(x)
c  = 3.0
h  = lambda x: c * f(x)

# Analytical derivatives
df = lambda x: np.cos(x)
dh = lambda x: c * df(x)

# Numerical derivative via central difference
def central_diff(func, x, eps=1e-5):
    return (func(x+eps) - func(x-eps)) / (2*eps)

# Test at multiple points
for x0 in [0.0, 1.0, 2.0]:
    print(f"x={x0:.1f}, analytic={dh(x0):.5f}, numeric={central_diff(h, x0):.5f}")
```

- **Extend:** Try c = −2.5, f(x) = x**2 + 3x, and confirm the scaling rule holds.

---

## Properties of the Derivative – Sum Rule

### Conceptual Overview

The sum rule states that the derivative of a sum of functions equals the sum of their derivatives.

If you have two curves, f(x) and g(x), and you “stack” them by adding their heights at each x, the instantaneous slope of the combined curve is just the sum of each slope.

This linearity is fundamental: it means differentiation respects addition.

### Key Formula

```
d/dx [ f(x) + g(x) ] = f′(x) + g′(x)
```

- `d/dx`: apply the derivative operator with respect to x
- `f(x) + g(x)`: original combined function
- `f′(x) + g′(x)`: sum of individual derivatives

### Formula Breakdown

1. **Apply limit definition** to F(x)=f(x)+g(x):
    
    ```
    F′(x) = lim(h→0)[F(x+h) − F(x)] / h
          = lim(h→0)[f(x+h)+g(x+h) − (f(x)+g(x))] / h
    ```
    
2. **Rearrange terms** to separate f and g:
    
    ```
    = lim(h→0)[(f(x+h)−f(x)) + (g(x+h)−g(x))] / h
    ```
    
3. **Split the fraction** (linearity of limits):
    
    ```
    = lim(h→0)[f(x+h)−f(x)]/h + lim(h→0)[g(x+h)−g(x)]/h
    = f′(x) + g′(x)
    ```
    

### Geometric Intuition

- Imagine two hills f and g. At each point x, you walk on both hills stacked on top of each other.
- Your instantaneous direction (slope) is the sum of how steep each hill is at that point.
- If one hill goes up at 2 units per x and the other at 3 units per x, the combined ascent is 5 units per x.

### Real-World ML/DS Applications

- **Composite loss functions:** L_total = L_data + λ·L_regularization.Backpropagating through L_total splits neatly into gradients of each term.
- **Feature combinations:** If a model’s score is f(feature1) + g(feature2), the sensitivity to x is additive.
- **Signal processing:** Summing two waveforms means you sum their instantaneous frequencies and amplitudes’ rates of change.

### Practice Problems

1. By hand, compute derivatives:
    - d/dx [ (3x² + 2x) + sin(x) ]
    - d/dx [ eˣ + ln(x) ]
    - d/dx [ x³ − 1/x + 5 ]
2. Verify that F′(x) = f′(x) + g′(x) holds for f(x)=x², g(x)=eˣ using finite differences at x=1 with h=1e-5.
3. In linear regression with L(w,b)=MSE(w,b)+α·∥w∥², write ∂L/∂w as the sum of its data-error term and regularization term.

### Python Exercises

```python
import numpy as np

# Define functions and their derivatives
f  = lambda x: x**2
g  = lambda x: np.sin(x)
df = lambda x: 2*x
dg = lambda x: np.cos(x)

# Combined function and derivative
F  = lambda x: f(x) + g(x)
dF = lambda x: df(x) + dg(x)

# Numeric derivative via central difference
def central_diff(func, x, h=1e-5):
    return (func(x+h) - func(x-h)) / (2*h)

# Test at multiple points
for x0 in [0.0, 1.0, 2.0]:
    print(f"x={x0}, analytic={dF(x0):.5f}, numeric={central_diff(F, x0):.5f}")
```

---

## Properties of the Derivative – Product Rule

### Conceptual Overview

When two quantities both change with x and you multiply them, the rate at which their product changes depends on how each one changes individually.

The product rule tells you to take one function times the derivative of the other, plus the derivative of the first times the second. This ensures no interaction is missed.

### Key Formula

```
d/dx [ f(x) * g(x) ] = f(x) * g′(x) + f′(x) * g(x)
```

- f(x): first function
- g(x): second function
- f′(x): derivative of f with respect to x
- g′(x): derivative of g with respect to x

### Line-by-Line Breakdown

1. **Start** with y = f(x)·g(x).
2. **Increment** x by h:
    
    ```
    y(x+h) = f(x+h)*g(x+h)
    ```
    
3. **Difference quotient**:
    
    ```
    [f(x+h)g(x+h) − f(x)g(x)] / h
    ```
    
4. **Add and subtract** f(x+h)·g(x):
    
    ```
    = [f(x+h)g(x+h) − f(x+h)g(x) + f(x+h)g(x) − f(x)g(x)] / h
    ```
    
5. **Group terms**:
    
    ```
    = [f(x+h)(g(x+h) − g(x)) + g(x)(f(x+h) − f(x))] / h
    ```
    
6. **Divide** and take limit h→0:
    
    ```
    = f(x) * lim[h→0][g(x+h)−g(x)]/h
      + g(x) * lim[h→0][f(x+h)−f(x)]/h
    = f(x)g′(x) + f′(x)g(x)
    ```
    

### Geometric Intuition

Picture two curves f and g. At a point x, each has its own tangent slope. The product rule combines:

- How steep g is (g′) weighted by f’s current height
- How steep f is (f′) weighted by g’s current height

Together they give the instantaneous slope of the product surface.

### Real-World ML/DS Applications

- **Feature interactions:** If a model score has a term x·y, gradient w.r.t. x is 1·y + x·0 = y, capturing how that interaction contributes.
- **Polynomial features:** For f(x)=x² and g(x)=3x, derivative of x²·3x uses product rule to yield 2x·3x + x²·3.
- **Neural network layers:** When computing gradients through a layer that multiplies activations (e.g., gating mechanisms), backprop uses the product rule.
- **Loss × regularizer:** If total loss is L_data(x)·L_reg(x), its gradient splits into data-gradient times reg-value plus reg-gradient times data-value.

### Practice Problems

1. By hand differentiate:
    
    a. d/dx [ (2x + 1) · sin(x) ]
    
    b. d/dx [ x² · e^x ]
    
    c. d/dx [ (ln x) · (x³) ]
    
2. Verify the derivative of f(x)=x³·cos(x) at x=π/4 by computing f′(x) via product rule and by central difference with h=1e-5.
3. In polynomial regression, if model output is ŷ(x)=w·x² + b·x, show how gradient ∂ŷ/∂x uses the product rule on each term.

### Python Exercises

```python
import numpy as np

# Define f, g and their derivatives
f  = lambda x: x**2
df = lambda x: 2*x

g  = lambda x: np.exp(x)
dg = lambda x: np.exp(x)

# Combined function and analytic derivative
h  = lambda x: f(x)*g(x)
dh = lambda x: f(x)*dg(x) + df(x)*g(x)

# Numerical derivative via central difference
def central_diff(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2*h)

# Test at a few points
xs = [0.0, 1.0, 2.0]
for x0 in xs:
    print(f"x={x0:.1f}, analytic={dh(x0):.5f}, numeric={central_diff(h, x0):.5f}")
```

```python
# Plot f(x)*g(x) and its derivative
import matplotlib.pyplot as plt

x_vals = np.linspace(-1, 3, 200)
plt.plot(x_vals, h(x_vals), label='h(x)=x^2·e^x')
plt.plot(x_vals, dh(x_vals), '--', label="h'(x)")
plt.legend()
plt.title('Product Rule: Function and Its Derivative')
plt.xlabel('x')
plt.show()
```

---

## Properties of the Derivative – Chain Rule

### Conceptual Overview

The chain rule tells you how to differentiate a composition of two (or more) functions.

If you have an outer function f and an inner function g, the rate at which f(g(x)) changes depends on how fast g changes at x and how fast f changes at g(x).

In ML, every time you stack layers or embed nonlinearities (e.g., sigmoid of a linear score), you use the chain rule to backpropagate gradients through each layer.

### Key Formula

```
d/dx [ f( g(x) ) ] = f′( g(x) ) * g′( x )
```

- f′(g(x)): derivative of the outer function evaluated at the inner function’s value
- g′(x): derivative of the inner function with respect to x

### Line-by-Line Breakdown

1. Start with the composite function:
    
    ```
    h(x) = f( g(x) )
    ```
    
2. Increment x by a small amount h:
    
    ```
    g(x+h) ≈ g(x) + g′(x)*h
    f(g(x+h)) ≈ f( g(x) + g′(x)*h )
    ```
    
3. Linearize f around g(x):
    
    ```
    f( g(x) + Δ ) ≈ f( g(x) ) + f′( g(x) ) * Δ
    where Δ = g′(x)*h
    
    ```
    
4. Combine and divide by h:
    
    ```
    [ f(g(x+h)) − f(g(x)) ] / h
    ≈ [ f′( g(x) ) * g′(x) * h ] / h
    = f′( g(x) ) * g′( x )
    ```
    

### Geometric Intuition

- Imagine walking along g’s curve: at x you have slope g′(x).
- Then feed your position g(x) into f, whose slope at that point is f′(g(x)).
- The chain rule multiplies these two slopes: how fast you move along g times how steep f is at the new point.

### Real-World ML/DS Applications

- **Neural network backpropagation:** Each layer applies a linear transform then activation. Gradients flow through both via the chain rule.
- **Logistic regression:** derivative of σ(w·x) is σ′(w·x)·(w·∂x), combining the sigmoid and linear parts.
- **Feature pipelines:** When you apply transformations like log(1+exp(x)), you differentiate the outer log and the inner exp.
- **Nested automatic differentiation:** In libraries like TensorFlow and PyTorch, chain rule underpins every composite gradient.

### Practice Problems

1. By hand differentiate:
    
    a. d/dx [ sin(3x²) ]
    
    b. d/dx [ ln(1 + e^(2x)) ]
    
    c. d/dx [ (3x + 1)⁵ ]
    
2. Compute the derivative of f(x) = tanh(2x + 1) using the chain rule and known derivative tanh′(u)=1−tanh²(u).
3. In a two-layer neural net with output y = σ( w₂·(σ(w₁x + b₁)) + b₂ ), derive ∂y/∂w₁ via the chain rule.

### Python Exercises

```python
import numpy as np

# Define inner and outer functions
g  = lambda x: 2*x + 1
f  = lambda u: np.sin(u)
h  = lambda x: f(g(x))

# Analytical derivatives
g_prime = lambda x: 2
f_prime = lambda u: np.cos(u)
h_prime = lambda x: f_prime(g(x)) * g_prime(x)

# Numerical derivative
def central_diff(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2*h)

# Test at points
for x0 in [0.0, 0.5, 1.0]:
    print(f"x={x0}, analytic={h_prime(x0):.5f}, numeric={central_diff(h, x0):.5f}")
```

```python
# Plot h(x) and its derivative
import matplotlib.pyplot as plt

xs = np.linspace(-2, 2, 200)
plt.plot(xs, h(xs), label='h(x)=sin(2x+1)')
plt.plot(xs, h_prime(xs), '--', label="h'(x)")
plt.legend()
plt.title('Composite Function and Its Derivative via Chain Rule')
plt.xlabel('x')
plt.show()
```

---

## Introduction to Optimization

### Why Optimization Matters in ML and DS

In machine learning and data science, you almost always pose a problem as “find the best parameters” that minimize a loss or maximize a performance measure.

Optimization is the math of making those choices:

- Tuning model weights to reduce prediction error
- Selecting hyperparameters to balance bias and variance
- Fitting probability distributions by maximizing likelihood

Without optimization, you can’t train models or extract insights from data.

### Formal Statement of an Optimization Problem

An optimization problem asks you to find the variable θ (which can be a number, vector, or matrix) that makes an objective function J(θ) as small (or as large) as possible.

```
Minimize J(θ)
with respect to θ ∈ ℝⁿ
```

- J(θ): objective (loss) you want to minimize
- θ: parameters of your model

If you instead want to maximize F(θ), you solve Minimize −F(θ), so everything stays in the “minimize” framework.

### Types of Optimization

- **Unconstrained vs. Constrained**
    
    Unconstrained: θ can be any real vector.
    
    Constrained: θ must satisfy some conditions (e.g., θ ≥ 0, sum(θ)=1).
    
- **Convex vs. Non-convex**
    
    Convex: objective has a single global minimum (nice bowl shape).
    
    Non-convex: multiple local minima (neural networks, deep learning).
    
- **Gradient-based vs. Gradient-free**
    
    Gradient-based: use derivatives to guide search (gradient descent, Newton).
    
    Gradient-free: use function values only (grid search, evolutionary algorithms).
    

### Core Optimization Method: Gradient Descent

Gradient descent is the workhorse for continuous, differentiable objectives.

### Update Rule

```
θ_new = θ_old − α * ∇J(θ_old)
```

- θ_old: current parameters
- ∇J(θ_old): gradient vector of partial derivatives of J at θ_old
- α (alpha): learning rate (step size)
- θ_new: updated parameters

### Breakdown of Terms

- **Gradient ∇J(θ)** gives the direction of steepest ascent.
- **Negative gradient** points to the steepest descent.
- **Learning rate α** scales how far you move along that direction each step.

### Geometric Intuition

Visualize J(θ) as a hilly landscape and θ as your position on it.

- The gradient tells you which way is “uphill.”
- Stepping in the negative gradient direction takes you “downhill” toward a minimum.
- With a convex bowl, repeated steps converge to the unique bottom.
- In rugged terrain (non-convex), you might get stuck in a local valley.

### Real-World ML Example: Linear Regression

Objective: mean squared error

```
J(w,b) = (1/m) Σᵢ [ w·xᵢ + b − yᵢ ]²
```

Gradient descent updates w and b:

```
dw = (2/m) Σᵢ (w·xᵢ + b − yᵢ) * xᵢ
db = (2/m) Σᵢ (w·xᵢ + b − yᵢ)

w_new = w_old − α * dw
b_new = b_old − α * db
```

Each iteration nudges parameters to reduce the average error.

### Practice Problems & Python Exercises

1. **Find the minimum of a quadratic**
    
    By hand, minimize f(x)=2x²−8x+3 by solving f′(x)=0, then verify f″(x)>0.
    
2. **Implement gradient descent on a simple quadratic**
    
    ```python
    import numpy as np
    
    # Define objective and gradient
    f  = lambda x: 2*x**2 - 8*x + 3
    df = lambda x: 4*x - 8
    
    # Hyperparameters
    alpha = 0.1
    n_steps = 50
    x = 0.0  # initial guess
    
    # Gradient descent loop
    for i in range(n_steps):
        grad = df(x)
        x = x - alpha * grad
    print("Approximate minimum at x =", x, "f(x) =", f(x))
    ```
    
3. **Linear regression from scratch**
    
    ```python
    # Toy data
    X = np.linspace(0, 10, 100)
    y = 2.5 * X + 1.0 + np.random.randn(100) * 2
    
    # Add bias term
    X_b = np.vstack([np.ones(len(X)), X]).T
    theta = np.zeros(2)  # [b, w]
    alpha = 0.01
    n_iter = 1000
    
    for _ in range(n_iter):
        gradients = (2/len(X_b)) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= alpha * gradients
    
    print("Learned parameters [b, w] =", theta)
    ```
    
4. **Plot convergence**
    
    Track f(x) over iterations for the simple quadratic and visualize how fast you descend.
    

---

## Optimization of Squared Loss – The One–Feature Power Plant Problem

### 1. Problem Setup

We have a single input feature (ambient temperature) and a single output (power generated).

We model their relationship with a straight line:

```
y_pred = w * x + b
```

- x: temperature
- y: actual power output
- w: slope (how power changes per degree)
- b: intercept (base power when x=0)

We want to choose w and b to minimize the average squared difference between predictions and true outputs.

### 2. Squared Loss Cost Function

Define the cost (objective) J(w,b) as the mean squared error:

```
J(w, b) = (1 / (2m)) * Σ_{i=1}^m [ (w * x_i + b) − y_i ]^2
```

Breakdown:

- m: number of data points
- w * x_i + b: predicted power for the i-th sample
- y_i: actual power of the i-th sample
- Squared error: [ prediction − actual ]^2
- 1/(2m): scales sum to an average and simplifies derivatives

### 3. Derivatives of the Cost

Compute partial derivatives to see how J changes when w or b shifts:

```
∂J/∂w = (1 / m) * Σ_{i=1}^m [ (w * x_i + b) − y_i ] * x_i
∂J/∂b = (1 / m) * Σ_{i=1}^m [ (w * x_i + b) − y_i ]
```

- Each summand is the error times the sensitivity (x_i for w, 1 for b)
- These gradients point in the direction of steepest increase; we’ll descend the negative.

### 4. Closed-Form Solution (Normal Equations)

Setting gradients to zero yields optimal parameters:

```
w = [ Σ (x_i − x̄)(y_i − ȳ ) ]  /  [ Σ (x_i − x̄)^2 ]
b = ȳ − w * x̄
```

- x̄ = (1/m) Σ x_i (mean of temperatures)
- ȳ = (1/m) Σ y_i (mean of power outputs)
- Numerator: covariance between x and y
- Denominator: variance of x

This solves for the best-fitting line in one pass.

### 5. Gradient Descent Algorithm

Rather than solve directly, you can iteratively update:

```
repeat until convergence:
  w := w − α * (1/m) * Σ [ (w*x_i + b) − y_i ] * x_i
  b := b − α * (1/m) * Σ [ (w*x_i + b) − y_i ]
```

- α: learning rate (step size)
- Each iteration moves parameters slightly downhill in cost.

### 6. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate the power plant data
np.random.seed(0)
m = 100
x = np.linspace(0, 40, m)                      # temperatures
y = 50 - 0.9 * x + np.random.randn(m) * 2.5    # power with noise

# Closed-form solution
x_mean, y_mean = x.mean(), y.mean()
num = np.sum((x - x_mean)*(y - y_mean))
den = np.sum((x - x_mean)**2)
w_closed = num / den
b_closed = y_mean - w_closed * x_mean

print("Closed-form w, b:", w_closed, b_closed)

# Gradient descent
alpha = 0.001
n_iter = 2000
w_gd, b_gd = 0.0, 0.0

for _ in range(n_iter):
    y_pred = w_gd * x + b_gd
    error = y_pred - y
    grad_w = (1/m) * np.dot(error, x)
    grad_b = (1/m) * np.sum(error)
    w_gd -= alpha * grad_w
    b_gd -= alpha * grad_b

print("GD solution w, b:", w_gd, b_gd)

# Plot data and fitted lines
plt.scatter(x, y, label='Data', alpha=0.6)
plt.plot(x, w_closed*x + b_closed, 'r-', label='Closed-form')
plt.plot(x, w_gd*x + b_gd, 'g--', label='Gradient Descent')
plt.xlabel('Temperature')
plt.ylabel('Power Output')
plt.legend()
plt.show()
```

### 7. Practice Exercises

- Derive ∂J/∂w and ∂J/∂b from the cost function step by step.
- Implement gradient descent with different α (e.g., 0.01, 0.0001) and observe convergence speed.
- Use real power plant dataset (e.g., UCI Combined Cycle Power Plant) to fit and evaluate.

### 8. Applications Beyond Power

- **Housing prices:** one feature (size) → price
- **Forecasting demand:** temperature → electricity load
- **Simple calibration:** sensor reading → true measurement

Mastering this one–feature squared-loss problem builds intuition for multi-variable least squares, ridge regression, and the foundation of generalized linear models.

---

## Optimization of Squared Loss – The Two-Feature Power Plant Problem

### 1. Problem Setup

We now have two input features—say temperatures at two nearby sites—and one output (power generated).

We model the relationship with a plane instead of a line:

```
y_pred = w₁·x₁ + w₂·x₂ + b
```

- x₁, x₂: features (e.g., temp at powerline 1 and temp at powerline 2)
- w₁, w₂: slopes (sensitivity of power to each temperature)
- b: intercept (base power when both temps are zero)
- y_pred: predicted power output

Our goal is to pick w₁, w₂, b to minimize the average squared error over m data points.

### 2. Squared Loss Cost Function

```
J(w₁, w₂, b) = (1 / (2m)) * Σ_{i=1}^m [ (w₁ x_{i1} + w₂ x_{i2} + b) − y_i ]²
```

- m: number of samples
- x_{i1}, x_{i2}: the two features for sample i
- y_i: actual power for sample i
- 1/(2m): normalizing factor that cancels 2 in the derivative

This J is a convex “bowl” in (w₁, w₂, b)-space.

### 3. Gradients (Partial Derivatives)

To find the steepest ascent, we compute partial derivatives:

```
∂J/∂w₁ = (1/m) * Σ_{i=1}^m [ (w₁ x_{i1} + w₂ x_{i2} + b) − y_i ] · x_{i1}

∂J/∂w₂ = (1/m) * Σ_{i=1}^m [ (w₁ x_{i1} + w₂ x_{i2} + b) − y_i ] · x_{i2}

∂J/∂b  = (1/m) * Σ_{i=1}^m [ (w₁ x_{i1} + w₂ x_{i2} + b) − y_i ]
```

- Each gradient measures how J changes per unit change in that parameter.
- We’ll use these in gradient descent to update (w₁, w₂, b).

### 4. Closed-Form Solution (Normal Equations)

Stack inputs into matrix X (m×3 with a column of ones) and outputs into y (m×1).

The optimal parameter vector θ = [b, w₁, w₂]ᵀ satisfies:

```
θ* = (Xᵀ X)⁻¹ Xᵀ y
```

Breakdown:

- X column 1 = all ones (for b), column 2 = x₁, column 3 = x₂
- Xᵀ X is a 3×3 matrix capturing variances and covariances
- Inverting Xᵀ X projects y onto the span of X

This gives the unique least-squares plane fitting the data.

### 5. Gradient Descent Algorithm

Iteratively update parameters by stepping opposite to the gradient:

```
repeat until convergence:
  w₁ ← w₁ − α * (1/m) Σ [error_i]·x_{i1}
  w₂ ← w₂ − α * (1/m) Σ [error_i]·x_{i2}
   b ←  b − α * (1/m) Σ [error_i]

where error_i = (w₁ x_{i1} + w₂ x_{i2} + b) − y_i
```

- α: learning rate
- Each step reduces J, moving the plane closer to the data points.

### 6. Python Implementation

```python
import numpy as np

# 1. Simulate data
np.random.seed(1)
m = 200
x1 = 20 + 10 * np.random.rand(m)                # temps at site 1
x2 = 25 + 5  * np.random.rand(m)                # temps at site 2
# true underlying relationship + noise
y  = 30 - 1.2*x1 + 0.8*x2 + np.random.randn(m) * 2

# 2. Closed-form solution
X = np.column_stack([np.ones(m), x1, x2])  # shape (m,3)
theta_closed = np.linalg.inv(X.T @ X) @ X.T @ y
b_c, w1_c, w2_c = theta_closed
print("Closed-form:", "b=", b_c, "w1=", w1_c, "w2=", w2_c)

# 3. Gradient descent
alpha, n_iter = 0.0005, 2000
b_g, w1_g, w2_g = 0.0, 0.0, 0.0

for _ in range(n_iter):
    y_pred = w1_g*x1 + w2_g*x2 + b_g
    error  = y_pred - y
    grad_b  = (1/m) * error.sum()
    grad_w1 = (1/m) * (error * x1).sum()
    grad_w2 = (1/m) * (error * x2).sum()
    b_g  -= alpha * grad_b
    w1_g -= alpha * grad_w1
    w2_g -= alpha * grad_w2

print("Gradient descent:", "b=", b_g, "w1=", w1_g, "w2=", w2_g)

# 4. Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, alpha=0.3, label='Data')
# Plot plane from closed-form
xx1, xx2 = np.meshgrid(np.linspace(x1.min(), x1.max(), 10),
                       np.linspace(x2.min(), x2.max(), 10))
yy = w1_c*xx1 + w2_c*xx2 + b_c
ax.plot_surface(xx1, xx2, yy, alpha=0.4, color='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('Two-Feature Plane Fit')
plt.legend()
plt.show()
```

### 7. Practice Problems

1. **Derive** the partials ∂J/∂w₁, ∂J/∂w₂, ∂J/∂b from J(w₁,w₂,b) step by step.
2. **Implement** gradient descent with different α (e.g., 0.001, 0.0001) and plot J over iterations.
3. **Compare** closed-form vs. gradient-descent results on a real dataset (UCI Power Plant Data).
4. **Extend** to three features: add a humidity feature x₃ and derive the normal equations for 4 parameters.

### 8. Geometric Intuition

- The cost J(w₁, w₂, b) is a convex bowl in 3D parameter space.
- The gradient is a vector pointing uphill—negative gradient points directly toward the bottom.
- The normal-equation solution finds the unique foot of the orthogonal projection of y onto the column space of X.

---

## Optimization of Squared Loss – The Three-Feature Power Plant Problem

### 1. Problem Setup

We now have three input features—temperatures at three powerlines—and one output (power generated).

Our linear model becomes a plane in 4D (three slopes + intercept):

```
y_pred = w₁·x₁ + w₂·x₂ + w₃·x₃ + b
```

- x₁, x₂, x₃: features (e.g., temps at three lines)
- w₁, w₂, w₃: sensitivities of output to each feature
- b: bias (base power when all x’s are zero)

We want to choose (w₁, w₂, w₃, b) to minimize the mean squared error over m samples.

### 2. Squared Loss Cost Function

```
J(w₁,w₂,w₃,b) = (1 / (2m)) * Σ_{i=1}^m [ (w₁ x_{i1} + w₂ x_{i2} + w₃ x_{i3} + b) − y_i ]²
```

- m: number of data points
- The factor ½ cancels when differentiating squared error

This J is a convex quadratic bowl in the 4-dimensional parameter space.

### 3. Gradients (Partial Derivatives)

Compute each partial to see how J changes with respect to each parameter:

```
∂J/∂w₁ = (1/m) * Σ_{i=1}^m [ (y_pred_i − y_i) · x_{i1} ]

∂J/∂w₂ = (1/m) * Σ_{i=1}^m [ (y_pred_i − y_i) · x_{i2} ]

∂J/∂w₃ = (1/m) * Σ_{i=1}^m [ (y_pred_i − y_i) · x_{i3} ]

∂J/∂b  = (1/m) * Σ_{i=1}^m [ (y_pred_i − y_i) ]
```

where

```
y_pred_i = w₁ x_{i1} + w₂ x_{i2} + w₃ x_{i3} + b
```

### 4. Closed-Form Solution (Normal Equations)

Stack inputs into matrix **X** (m×4):

- Column 1: all ones (for b)
- Column 2: x₁, Column 3: x₂, Column 4: x₃

The optimal parameter vector

```
θ = [b, w₁, w₂, w₃]ᵀ
```

satisfies:

```
θ* = (Xᵀ X)⁻¹ Xᵀ y
```

This gives the unique least-squares solution in one linear algebra step.

### 5. Gradient Descent Algorithm

Alternatively, iterate updates until convergence:

```
repeat:
  compute y_pred = w₁x₁ + w₂x₂ + w₃x₃ + b

  error_i = y_pred_i − y_i

  w₁ ← w₁ − α * (1/m) Σ error_i·x_{i1}
  w₂ ← w₂ − α * (1/m) Σ error_i·x_{i2}
  w₃ ← w₃ − α * (1/m) Σ error_i·x_{i3}
   b ←  b − α * (1/m) Σ error_i
```

- α: learning rate
- Each step moves parameters downhill in J

### 6. Python Implementation

```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(42)
m = 300
x1 = 20 + 10*np.random.rand(m)
x2 = 25 + 5 *np.random.rand(m)
x3 = 15 + 8 *np.random.rand(m)
# true relation + noise
y  = 40 - 1.0*x1 + 0.5*x2 + 0.8*x3 + np.random.randn(m)*3

# Closed-form solution
X = np.column_stack([np.ones(m), x1, x2, x3])    # shape (m,4)
theta_closed = np.linalg.inv(X.T @ X) @ X.T @ y
b_c, w1_c, w2_c, w3_c = theta_closed
print("Closed-form:", b_c, w1_c, w2_c, w3_c)

# Gradient descent
alpha, n_iter = 0.0005, 5000
b_g, w1_g, w2_g, w3_g = 0.0, 0.0, 0.0, 0.0

for _ in range(n_iter):
    y_pred = w1_g*x1 + w2_g*x2 + w3_g*x3 + b_g
    error  = y_pred - y
    grad_b  = error.mean()
    grad_w1 = (error * x1).mean()
    grad_w2 = (error * x2).mean()
    grad_w3 = (error * x3).mean()
    b_g  -= alpha * grad_b
    w1_g -= alpha * grad_w1
    w2_g -= alpha * grad_w2
    w3_g -= alpha * grad_w3

print("Gradient Descent:", b_g, w1_g, w2_g, w3_g)

# 3D Visualization of two features (fix x3 at its mean)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mask = np.abs(x3 - x3.mean()) < 1.0  # slice around mean
ax.scatter(x1[mask], x2[mask], y[mask], alpha=0.5)
# plane using closed-form, with x3 = x3.mean()
xx1, xx2 = np.meshgrid(np.linspace(x1.min(),x1.max(),10),
                       np.linspace(x2.min(),x2.max(),10))
xx3 = x3.mean()
yy = w1_c*xx1 + w2_c*xx2 + w3_c*xx3 + b_c
ax.plot_surface(xx1, xx2, yy, alpha=0.4, color='orange')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
plt.title('3-Feature Linear Fit (slice at x3 mean)')
plt.show()
```

### 7. Practice Exercises

1. Derive ∂J/∂w₁, ∂J/∂w₂, ∂J/∂w₃, ∂J/∂b step by step from the cost J.
2. Run gradient descent with different learning rates (e.g., 0.001, 0.0001) and plot J over iterations to observe convergence behavior.
3. Apply the closed-form and GD solutions to the UCI Combined Cycle Power Plant dataset (features: T, V, AP) and compare performance (MSE).
4. Extend to four features: derive normal equations for a 5×5 matrix XᵀX.

### 8. Geometric Intuition

- The cost surface J(w₁,w₂,w₃,b) is a 4D convex bowl.
- The gradient vector points uphill; stepping in its negative direction leads you toward the unique global minimum.
- The normal-equation solution directly computes the foot of the orthogonal projection of y onto the column space of X.

---

## Optimization of Log‐Loss, Part 1: Single‐Feature Logistic Regression

### 1. Problem Setup

We have a single input feature (x) (e.g., temperature) and a binary outcome (y\in{0,1}) (e.g., “power plant on/off”).

Our model uses a sigmoid (logistic) link:

```
ŷ = σ(z)  where  z = w·x + b
σ(z) = 1 / (1 + e^(−z))
```

- (w): weight (how strongly (x) influences the log‐odds)
- (b): bias (log‐odds when (x=0))
- ŷ: predicted probability of (y=1)

### 2. Log‐Loss (Binary Cross‐Entropy) Cost

We measure fit with the average negative log‐likelihood:

```
J(w,b) = −(1/m) Σ_{i=1}^m [ y_i·log(ŷ_i) + (1−y_i)·log(1−ŷ_i) ]
```

- (m): number of training examples
- ŷᵢ = σ(w·xᵢ + b)
- Penalizes confident mistakes heavily (log barrier near 0)

### 3. Gradients of the Cost

Compute partial derivatives to drive gradient descent:

```
∂J/∂w = (1/m) Σ_{i=1}^m (ŷ_i − y_i)·x_i

∂J/∂b = (1/m) Σ_{i=1}^m (ŷ_i − y_i)
```

**Why this works**

- For each sample, error = ŷᵢ − yᵢ
- Chain rule through σ: dσ/dz = σ(z)·[1−σ(z)] cancels with log‐loss derivative

### 4. Gradient Descent Algorithm

Iteratively update parameters:

```
repeat until convergence:
  compute ŷ_i = σ(w·x_i + b)  for all i

  grad_w = (1/m) Σ (ŷ_i − y_i)·x_i
  grad_b = (1/m) Σ (ŷ_i − y_i)

  w ← w − α·grad_w
  b ← b − α·grad_b
```

- α: learning rate (tune via experiments)
- Stop when J stops decreasing or after fixed epochs

### 5. Python Implementation

```python
import numpy as np

# Simulate data
np.random.seed(0)
m = 100
x = np.linspace(0,10,m)
# true decision boundary at x=5
y = (x + np.random.randn(m)) > 5
y = y.astype(int)

# Sigmoid
sigmoid = lambda z: 1/(1 + np.exp(-z))

# Hyperparameters
alpha = 0.1
n_iter = 2000
w, b = 0.0, 0.0

# Training loop
for _ in range(n_iter):
    z = w*x + b
    y_pred = sigmoid(z)
    error = y_pred - y
    grad_w = (error * x).mean()
    grad_b = error.mean()
    w -= alpha * grad_w
    b -= alpha * grad_b

print("Learned w, b:", w, b)
```

### 6. Practice Problems

1. Derive ∂J/∂w and ∂J/∂b step by step using the chain rule.
2. Experiment with different learning rates (e.g., 0.01, 0.5) and plot J over iterations.
3. Use real binary‐classification data (e.g., Iris setosa vs. versicolor on petal length) to fit a single‐feature logistic model.

---

## Optimization of Log‐Loss, Part 2

### 1. Multi‐Feature Logistic Regression

We generalize from one feature to an input vector **x**∈ℝⁿ with weights **w**∈ℝⁿ and bias b.

The model and loss are:

```
z = wᵀ x + b
ŷ = σ(z) = 1 / (1 + e^(−z))

J(w,b) = −(1/m) Σ [ yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ) ]
```

Using matrix notation, stack examples into X (m×n) and targets y (m×1). This lets us vectorize gradients efficiently.

### 2. Regularization (L2 Penalty)

To prevent overfitting, add an L2 penalty on the weights:

```
J_reg(w,b) = J(w,b) + (λ/(2m)) · ‖w‖²
```

- λ≥0 is the regularization strength
- Penalizing large w shrinks weights toward zero

The gradients become:

```
∇_w J_reg = (1/m)·Xᵀ(ŷ − y) + (λ/m)·w
∂J_reg/∂b  = (1/m)·Σ (ŷ − y)
```

### 3. Vectorized Gradient Descent

Implementing batch gradient descent with matrix operations:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss_and_gradients(X, y, w, b, lam):
    m = X.shape[0]
    z = X.dot(w) + b
    y_pred = sigmoid(z)
    loss = (-1/m) * (y*np.log(y_pred) + (1-y)*np.log(1-y_pred)).sum()
    loss += (lam/(2*m)) * np.dot(w, w)

    error = y_pred - y
    grad_w = (1/m) * X.T.dot(error) + (lam/m) * w
    grad_b = (1/m) * error.sum()
    return loss, grad_w, grad_b

# Gradient descent loop
alpha, lam, n_iter = 0.1, 0.01, 2000
w = np.zeros(n_features)
b = 0.0

for i in range(n_iter):
    loss, grad_w, grad_b = compute_loss_and_gradients(X, y, w, b, lam)
    w -= alpha * grad_w
    b -= alpha * grad_b
```

### 4. Newton’s Method (Second‐Order Update)

For faster, often quadratic convergence on convex problems, use Newton’s update:

```
θ = [b; w]   (dimension n+1)
H = Hessian of J_reg w.r.t. θ  ( (n+1)×(n+1) matrix )
g = gradient vector             (n+1)×1

θ_new = θ − H⁻¹ · g
```

For logistic loss, the Hessian H has blocks:

```
S = diag( σ(zᵢ)·(1−σ(zᵢ)) )   (m×m)

H = (1/m) · [ [1ᵀ S  X]ᵀ [1ᵀ S  X] ] + (λ/m)·diag(0, Iₙ)
```

In practice:

```python
# Build Hessian and gradient
p = sigmoid(z)                                # m×1
S = np.diag(p * (1-p))                        # m×m
X_aug = np.hstack([np.ones((m,1)), X])        # m×(n+1)

H = (1/m) * X_aug.T.dot(S).dot(X_aug)
H[1:,1:] += (lam/m) * np.eye(n_features)      # regularize w only

g = np.concatenate(([grad_b], grad_w))
theta -= np.linalg.solve(H, g)
```

### 5. Model Evaluation & Convergence

- **Loss trajectory:** plot J_reg over iterations to monitor convergence.
- **Accuracy / ROC‐AUC:** evaluate predictions ŷ against y.
- **Stopping criteria:** small change in loss or parameters, or maximum iterations.

Tuning α for gradient descent and verifying Hessian conditioning are key to stable convergence.

### 6. Practice Problems

- **Vectorize** a two‐class logistic regression on the Iris dataset (sepal length & width). Plot decision boundary.
- **Implement L1 regularization** (lasso) with subgradient descent: modify gradients for |w|.
- **Compare optimizers:** run SGD, momentum, AdaGrad, and Newton on the same data. Plot loss vs. iteration.

---