# Mathematics_c2_w2

## Introduction to Tangent Planes

### 1. Conceptual Overview

A tangent plane generalizes the idea of a tangent line (from single‐variable calculus) to surfaces defined by z = f(x, y).

At a point (x₀, y₀), the tangent plane is the best linear approximation to the surface.

Just as a tangent line “touches” a curve and shares its slope, the tangent plane “touches” the surface and shares its partial slopes in the x– and y–directions.

### 2. Definition of the Tangent Plane

Given a differentiable function

```
z = f(x, y)
```

the tangent plane at the point (x₀, y₀, z₀) where z₀ = f(x₀, y₀) is:

```
z = z₀
    + f_x(x₀, y₀) * (x − x₀)
    + f_y(x₀, y₀) * (y − y₀)
```

- f_x and f_y are the partial derivatives at (x₀, y₀).
- The plane matches the surface value and slopes in both directions.

### 3. Derivation via Linear Approximation

1. **Start** with the multivariable linearization:
    
    ```
    f(x, y) ≈ f(x₀, y₀)
            + f_x(x₀, y₀)·(x−x₀)
            + f_y(x₀, y₀)·(y−y₀)
    ```
    
2. **Interpret** that as the equation of a plane in (x, y, z).
3. **Rearrange** to isolate z:
    
    ```
    z = f(x₀, y₀)
        + f_x(x₀, y₀)·(x−x₀)
        + f_y(x₀, y₀)·(y−y₀)
    ```
    

### 4. Geometric Intuition

- **Partial slopes** f_x and f_y give the steepness of the surface in the x–axis and y–axis directions.
- The tangent plane “leans” according to those two slopes, forming a flat surface that just touches without crossing (locally).
- Zooming in infinitesimally on the surface at (x₀, y₀) makes the surface look like its tangent plane.

### 5. Worked Example

Function:

```
f(x, y) = x² + y²
```

Point:

```
(x₀, y₀) = (1, 2)
```

1. Compute z₀:
    
    ```
    z₀ = f(1, 2) = 1² + 2² = 5
    ```
    
2. Compute partials:
    
    ```
    f_x(x,y) = 2x    → f_x(1,2) = 2
    f_y(x,y) = 2y    → f_y(1,2) = 4
    ```
    
3. Plug into formula:
    
    ```
    z = 5 + 2·(x−1) + 4·(y−2)
      = 5 + 2x − 2 + 4y − 8
      = 2x + 4y - 5
    ```
    

So the tangent plane is **z = 2x + 4y − 5**.

### 6. Real-World ML/DS Applications

- **Newton’s method in multiple variables:** uses the tangent plane (first derivatives) and Hessian (second derivatives) to find roots or optima.
- **Local linear models:** approximate complex surfaces by planes for interpretability (LIME algorithm).
- **Feature sensitivity:** partial derivatives show how small changes in each feature jointly affect the prediction.
- **Gradient‐based solvers for PDEs and neural ODEs:** surfaces in (x,y,z) get linearized at each step.

### 7. Practice Problems

1. For f(x,y)=e^(x+y), find the tangent plane at (0,0).
2. Determine and classify where the tangent plane to f(x,y)=x³−3xy+y² is horizontal (flat).
3. Approximate f(1.1, 1.9) for f(x,y)=ln(x)+x·y by using the tangent plane at (1,2).

### 8. Python Exercise: Plotting a Tangent Plane

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define function and point
f = lambda x, y: x**2 + y**2
x0, y0 = 1, 2
z0    = f(x0, y0)
fx    = lambda x, y: 2*x
fy    = lambda x, y: 2*y

# Tangent plane
plane = lambda x, y: z0 + fx(x0,y0)*(x-x0) + fy(x0,y0)*(y-y0)

# Grid for plotting
xs = np.linspace(0, 2, 30)
ys = np.linspace(1, 3, 30)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)
Z_plane = plane(X, Y)

# Plot
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis', label='f(x,y)')
ax.plot_surface(X, Y, Z_plane, alpha=0.4, color='red', label='Tangent Plane')
ax.scatter([x0], [y0], [z0], color='black', s=50)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.title('Surface and Its Tangent Plane at (1,2)')
plt.show()
```

---

## Partial Derivatives – Part 1

### 1. What You Need to Know First

Before diving into partial derivatives, you should already be comfortable with:

- The derivative of a single-variable function (the slope of a curve):```d/dx [f(x)] = f′(x)```
- How to compute basic derivatives (power rule, sum rule, product rule, chain rule).
- The idea of holding everything else constant when we measure “rate of change.”

### 2. Conceptual Overview

Imagine a smooth landscape described by height = f(x, y).

- x and y are two directions (think east–west and north–south).
- Partial derivatives tell us how steep the hill is if we walk purely east–west (holding y fixed) or purely north–south (holding x fixed).

In data science terms:

- f could be a loss function in two parameters θ₁ and θ₂.
- ∂f/∂θ₁ tells you how much the loss changes if you tweak θ₁ a tiny bit, keeping θ₂ constant.
- That directional sensitivity is the backbone of gradient-based optimization.

### 3. Formal Definition

For a function f(x, y), the partial derivative with respect to x is:

```
∂f/∂x (x, y) = limit as h → 0 of [ f(x + h, y) – f(x, y) ] / h
```

Likewise, with respect to y:

```
∂f/∂y (x, y) = limit as h → 0 of [ f(x, y + h) – f(x, y) ] / h
```

**Step-by-step** for ∂f/∂x:

1. Pick a tiny step h along the x-axis.
2. Measure how f changes when x→x+h, but keep y exactly the same.
3. Divide that change by h to get “change per unit x.”
4. Let h shrink to zero for the instantaneous rate.

### 4. Why Partial Derivatives Matter in ML/DS

- **Gradient descent**: For a loss L(θ₁, θ₂,…), you update each θⱼ using ∂L/∂θⱼ.
- **Feature sensitivity**: In linear regression L(w,b) = mean((w x + b – y)²), ∂L/∂w = average error·x tells how loss shifts when you scale that feature.
- **Optimization checks**: Finite-difference gradient checking uses partial derivatives to catch bugs.

### 5. Worked Examples

### 5.1 Simple Polynomial

Let

```
f(x, y) = x²·y + 3·x·y² – sin(x)
```

- Treat y as constant when computing ∂/∂x:
    
    ```
    ∂f/∂x = 2·x·y   +  3·y²   –  cos(x)
    ```
    
- Treat x as constant for ∂/∂y:
    
    ```
    ∂f/∂y = x²     +  6·x·y     +  0
    ```
    

### 5.2 Logarithm Example

Let

```
f(x, y) = ln(x·y)
```

Use ln(a·b) = ln(a) + ln(b). Then:

- ∂f/∂x = 1/x
- ∂f/∂y = 1/y

### 6. Geometric Intuition

- **Slice in x-direction**: Set y=y₀ and draw the curve z=f(x,y₀). The slope of that curve at x₀ is ∂f/∂x(x₀, y₀).
- **Slice in y-direction**: Set x=x₀ and see slope ∂f/∂y(x₀, y₀).
- **Gradient vector** (∂f/∂x, ∂f/∂y) points uphill in the steepest direction and is perpendicular to level curves.

### 7. Practice Problems

### 7.1 By Hand

1. f(x,y) = x³ + 2xy + y
    - Compute ∂f/∂x and ∂f/∂y.
2. f(x,y) = e^(x·y)
    - Find ∂f/∂x (treat y constant) and ∂f/∂y.
3. f(x,y) = (x² + y²)^(1/2)
    - Compute ∂f/∂x and ∂f/∂y.

### 7.2 Numerical Approximation in Python

```python
import numpy as np

def partial_x(f, x, y, h=1e-5):
    return (f(x+h, y) - f(x-h, y)) / (2*h)

def partial_y(f, x, y, h=1e-5):
    return (f(x, y+h) - f(x, y-h)) / (2*h)

# Example function
f = lambda x, y: x**2 * y + 3*x*y**2 - np.sin(x)

x0, y0 = 1.0, 2.0
print("Numeric ∂f/∂x:", partial_x(f, x0, y0))
print("Numeric ∂f/∂y:", partial_y(f, x0, y0))

# Compare to analytic
analytic_dx = 2*x0*y0 + 3*y0**2 - np.cos(x0)
analytic_dy = x0**2 + 6*x0*y0
print("Analytic ∂f/∂x:", analytic_dx)
print("Analytic ∂f/∂y:", analytic_dy)
```

### 8. Real-World ML/DS Example

**Linear Regression Loss**

Loss per sample:

```
L_i(w, b) = (w·x_i + b – y_i)²
```

Partial wrt w:

```
∂L_i/∂w = 2·(w·x_i + b – y_i)·x_i
```

If xᵢ is a feature like “square footage,” ∂L/∂w tells you how adjusting the price-per-square-foot weight changes the error.

In code, summing over all m samples gives the gradient used in gradient descent.

### 9. Visual Description

Imagine a bowl-shaped surface L(w₁, w₂).

- At each (w₁, w₂), the partial derivatives ∂L/∂w₁ and ∂L/∂w₂ define the tilt in two perpendicular directions.
- Together they form the gradient arrow that points uphill; flipping its sign points you directly downhill toward the minimum.

---

## Partial Derivatives – Part 2

### 1. What You Should Already Know

You’ve mastered first-order partials (∂f/∂x, ∂f/∂y) from Part 1.

You can:

- Hold one variable constant while differentiating with respect to another.
- Compute basic single-variable derivatives (power rule, product rule, chain rule).

### 2. Higher-Order Partials

Just as in single-variable calculus you can take second or third derivatives, in multivariable calculus you can take **higher-order partials**.

For a function `f(x, y)`, the second-order partials are:

```
f_xx(x,y) = ∂/∂x [ ∂f/∂x ]    (second partial with respect to x twice)
f_yy(x,y) = ∂/∂y [ ∂f/∂y ]    (second partial with respect to y twice)
f_xy(x,y) = ∂/∂y [ ∂f/∂x ]    (first x then y)
f_yx(x,y) = ∂/∂x [ ∂f/∂y ]    (first y then x)
```

### Step-by-Step

1. Compute `∂f/∂x` as in Part 1.
2. Differentiate that result with respect to x again to get `f_xx`.
3. Or differentiate `∂f/∂x` with respect to y to get the mixed partial `f_xy`.
4. Repeat similarly for `f_yy` and `f_yx`.

### 3. Clairaut’s (Schwarz’s) Theorem

Under mild conditions (f and its partials up to second order are continuous near a point), **mixed partials are equal**:

```
f_xy(x, y) = f_yx(x, y)
```

**Why it matters**

- It guarantees symmetry of the Hessian matrix in optimization.
- Allows us to compute whichever order is easier.

### 4. The Hessian Matrix

The Hessian `H(f)` collects all second-order partials:

```
H(f)(x,y) = | f_xx   f_xy |
            | f_yx   f_yy |
```

Because `f_xy = f_yx`, `H` is symmetric.

**Use in ML**

- Measures curvature of a loss surface.
- In Newton’s method, you invert or solve systems with the Hessian to accelerate convergence.

### 5. Taylor Expansion in Two Variables

A second-order Taylor approximation of `f(x, y)` around `(x₀, y₀)` uses first and second partials:

```
f(x, y) ≈ f(x₀,y₀)
         + f_x(x₀,y₀)*(x−x₀)
         + f_y(x₀,y₀)*(y−y₀)
         + 1/2 * [ f_xx(x₀,y₀)*(x−x₀)²
                   + 2*f_xy(x₀,y₀)*(x−x₀)*(y−y₀)
                   + f_yy(x₀,y₀)*(y−y₀)² ]
```

### Meaning of Each Term

- The linear terms match the tangent plane (Part 1).
- The quadratic terms capture curvature in each direction and how x and y interact.

### 6. Worked Example

Let

```
f(x, y) = x² y + 3 x y² − sin(x y)
```

1. First-order partials (for reference):
    
    ```
    f_x = 2 x y + 3 y² − cos(x y) * y
    f_y = x² + 6 x y − cos(x y) * x
    ```
    
2. Second-order partials:
    
    ```
    f_xx = ∂/∂x [f_x]
          = 2 y +      d/dx[− y·cos(x y)]
          = 2 y + y²·sin(x y)
    
    f_yy = ∂/∂y [f_y]
          =       d/dy[x²] + 6 x + d/dy[− x·cos(x y)]
          = 0 + 6 x + x²·sin(x y)
    
    f_xy = ∂/∂y [f_x]
          = 2 x + 6 y − d/dy[y·cos(x y)]
          = 2 x + 6 y − [ cos(x y) + y·(−sin(x y)*x) ]
          = 2 x + 6 y − cos(x y) + x y·sin(x y)
    
    f_yx = ∂/∂x [f_y]
          (you’ll find it matches f_xy)
    ```
    

### 7. Practice Problems

### By Hand

1. For `f(x,y) = x³ + xy + y³`, compute `f_xx`, `f_yy`, and the mixed `f_xy`.
2. Verify `f_xy = f_yx` for `f(x,y) = ln(x+y)`.

### Numerical Verification in Python

```python
import numpy as np

def partial(f, x, y, var, h=1e-4):
    if var=='x':
        return (f(x+h,y)-f(x-h,y))/(2*h)
    else:
        return (f(x,y+h)-f(x,y-h))/(2*h)

def second_partial(f, x, y, vars):
    if vars=='xx':
        return (partial(f, x+1e-4, y, 'x') - partial(f, x-1e-4, y, 'x'))/(2e-4)
    if vars=='yy':
        return (partial(f, x, y+1e-4, 'y') - partial(f, x, y-1e-4, 'y'))/(2e-4)
    if vars=='xy':
        return (partial(f, x, y+1e-4, 'x') - partial(f, x, y-1e-4, 'x'))/(2e-4)

f = lambda x,y: x**2*y + 3*x*y**2 - np.sin(x*y)

x0, y0 = 1.2, 0.7
print("f_xx", second_partial(f, x0, y0, 'xx'))
print("f_yy", second_partial(f, x0, y0, 'yy'))
print("f_xy", second_partial(f, x0, y0, 'xy'))
print("f_yx", second_partial(f, x0, y0, 'xy'))  # same as f_xy
```

### 8. Real-World ML/DS Applications

- **Newton’s Method:** Uses Hessian and gradient to update parameters:
    
    ```
    θ_new = θ_old − H⁻¹ ∇f(θ_old)
    ```
    
- **Gaussian Approximations:** Second-order Taylor around a mode gives covariance = H⁻¹.
- **LIME (Local Interpretable Model-agnostic Explanations):** Fits a local linear model—first and second partials reveal feature interactions.

### 9. Geometric and Intuitive Interpretation

- **f_xx and f_yy** tell you how the slope in x or y is itself changing—concavity along each axis.
- **Mixed partial f_xy** tells how a change in x affects the slope in y, capturing interaction “twist.”
- **Positive curvature** (f_xx>0) hints at a local minimum in that direction; negative hints at a maximum.
- The Hessian’s eigenvalues classify points as minima, maxima, or saddle points.

---

## Gradients

### 1. Prerequisites

Before diving into gradients, ensure you’re comfortable with:

- Partial derivatives (∂f/∂x, ∂f/∂y)【partial derivatives part 1】
- Higher-order partials and the Hessian【partial derivatives part 2】
- Basic vector notation (vectors as ordered lists of numbers)

### 2. What Is a Gradient?

The **gradient** of a scalar-valued function (f) of several variables is the vector of all its first-order partial derivatives.

- It points in the direction of **steepest ascent** on the surface defined by (z = f(x_1,x_2,\dots,x_n)).
- Its length (magnitude) tells you how fast (f) is increasing in that direction.

In optimization, we move **against** the gradient (negative gradient) to descend the surface toward a minimum.

### 3. Formal Definition

For (f(x,y)), the gradient is

```
∇f(x,y) = [ ∂f/∂x ,  ∂f/∂y ]
```

For (f(x₁,x₂,…,xₙ)), it generalizes to

```
∇f(x₁,…,xₙ) = [ ∂f/∂x₁ ,  ∂f/∂x₂ ,  … ,  ∂f/∂xₙ ]
```

- Each component is the rate of change of (f) when you vary one coordinate, holding others fixed.
- The gradient itself is a function of the input point ((x₁,…,xₙ)).

### 4. Why Gradients Matter in ML/DS

- **Gradient Descent:** Update parameters (\theta) by
    
    ```
    θ_new = θ_old − α · ∇J(θ_old)
    ```
    
    where (J) is your loss and (\alpha) the learning rate.
    
- **Direction of Steepest Descent:** The negative gradient is the fastest way to reduce loss in one step.
- **Feature Sensitivity:** If (f(w,b)=\text{MSE}(w,b)), then (\partial f/∂w) tells how loss changes if you tweak weight (w).
- **Visualization:** In a 3D plot of (z=f(x,y)), gradient vectors lie perpendicular to level curves (contours).

### 5. Worked Example

Let

```
f(x, y) = x²·y + 3·x·y² − sin(x·y)
```

1. Compute first-order partials:
    
    ```
    ∂f/∂x = 2x·y + 3y² − y·cos(x·y)
    ∂f/∂y = x² + 6x·y − x·cos(x·y)
    ```
    
2. Form the gradient vector:
    
    ```
    ∇f(x,y) = [ 2x·y + 3y² − y·cos(x·y) ,
                x² + 6x·y − x·cos(x·y) ]
    ```
    
3. At the point (1,2):
    
    ```
    ∂f/∂x = 2·1·2 + 3·4 − 2·cos(2) = 4 + 12 − 2·cos(2)
    ∂f/∂y = 1 + 12 − 1·cos(2)      = 13 − cos(2)
    ∇f(1,2) ≈ [16 − 2·(−0.416), 13 − (−0.416)]
             ≈ [16.832, 13.416]
    ```
    

### 6. Visual Intuition

Imagine standing on a hill at point ((x,y)).

- Draw the direction in which the hill rises most steeply—that arrow is the gradient.
- The length of the arrow equals how fast altitude increases if you step in that exact direction.

On a contour map, these arrows are perpendicular to the concentric level lines.

### 7. Practice Problems

### 7.1 By Hand

1. (f(x,y) = x^3 − 3xy + y^2)
    - Find ∂f/∂x, ∂f/∂y, then write ∇f.
2. (f(x,y,z) = e^{xy} + z^2·\sin(x))
    - Compute ∂f/∂x, ∂f/∂y, ∂f/∂z and assemble ∇f.

### 7.2 Python Exploration

```python
import numpy as np

# Define f and its analytic gradient
f = lambda x, y: x**2 * y + 3*x*y**2 - np.sin(x*y)
grad_analytic = lambda x, y: np.array([
    2*x*y + 3*y**2 - y*np.cos(x*y),
    x**2 + 6*x*y - x*np.cos(x*y)
])

# Numerical gradient via central differences
def numeric_grad(f, x, y, h=1e-5):
    df_dx = (f(x+h,y) - f(x-h,y)) / (2*h)
    df_dy = (f(x,y+h) - f(x,y-h)) / (2*h)
    return np.array([df_dx, df_dy])

point = (1.0, 2.0)
print("Analytic grad:", grad_analytic(*point))
print("Numeric grad :", numeric_grad(f, *point))
```

### 8. Real-World ML Example

**Logistic Regression Loss**

For data matrix **X** (m×n), labels **y**, weights **w**, bias **b**:

```
z = X·w + b
ŷ = sigmoid(z)
J(w,b) = −(1/m) Σ [y·ln(ŷ) + (1−y)·ln(1−ŷ)]

Gradient:
∂J/∂w = (1/m)·Xᵀ(ŷ − y)
∂J/∂b = (1/m)·Σ(ŷ − y)

So ∇J = [∂J/∂w, ∂J/∂b] guides each GD update.
```

### 9. Practice with Gradient Descent

```python
# Simple gradient descent on f(x,y)=x^2 + y^2
import numpy as np

f = lambda v: v[0]**2 + v[1]**2
grad = lambda v: np.array([2*v[0], 2*v[1]])

v = np.array([3.0, -1.5])
alpha = 0.1
for i in range(20):
    v = v - alpha * grad(v)
    print(f"Iter {i:2d}: v={v}, f(v)={f(v):.4f}")
```

---

## Gradients and Maxima/Minima

### 1. Prerequisites

Before diving in, make sure you’ve mastered:

- First‐order partial derivatives and the **gradient** ∇f 【Gradients】
- Second‐order partials and the **Hessian** matrix H(f) 【Partial Derivatives Part 2】
- Basic single‐variable extremum tests (f′(x)=0 and f″(x) sign)

### 2. Conceptual Overview

In single‐variable calculus:

- A maximum or minimum occurs where f′(x)=0 (a “stationary point”).
- The sign of f″(x) tells you if it’s a valley (f″>0) or a hilltop (f″<0).

In multiple dimensions:

- **Stationary points** satisfy ∇f=0 (all partials zero).
- The Hessian H at that point—built from second partials—determines its nature:
    - **Positive definite H** ⇒ local minimum
    - **Negative definite H** ⇒ local maximum
    - **Indefinite H** ⇒ saddle point

This interplay between gradient and curvature guides optimization in ML: finding the lowest loss or highest log‐likelihood.

### 3. Stationary Points and Critical Conditions

Find **critical points** by solving:

```
∇f(x₁,…,xₙ) = 0
```

Concretely for f(x,y):

```
[ ∂f/∂x , ∂f/∂y ] = [ 0 , 0 ]
```

**Step‐by‐step**:

1. Compute ∂f/∂x and ∂f/∂y.
2. Solve the simultaneous equations ∂f/∂x=0 and ∂f/∂y=0.
3. Each solution (x₀,y₀) is a candidate for max, min, or saddle.

### 4. Second‐Order Test via the Hessian

At a critical point p=(x₀,…,xₙ), form the Hessian H:

```
H = [ ∂²f/∂xᵢ∂xⱼ ]  (n×n matrix)
```

For two variables:

```
H = | f_xx   f_xy |
    | f_yx   f_yy |
```

Evaluate eigenvalues λ₁, λ₂ of H(p), or use determinants:

1. Compute D = f_xx·f_yy − (f_xy)² at p.
2. Examine signs:
    - D > 0 and f_xx > 0 ⇒ local **minimum**
    - D > 0 and f_xx < 0 ⇒ local **maximum**
    - D < 0 ⇒ **saddle point**
    - D = 0 ⇒ test inconclusive

### 5. Real‐World ML/DS Applications

- **Loss surface analysis:** Identify minima of mean‐squared or cross‐entropy loss for model parameters.
- **Newton’s method:** Uses H⁻¹∇f to jump toward a local minimum more quickly than gradient descent.
- **Gaussian Laplace approximations:** Maximum of log‐posterior found by solving ∇log p=0 and using H for covariance.

### 6. Worked Example

Let

```
f(x,y) = x³ − 3x y + y²
```

1. Compute gradient:
    
    ```
    ∂f/∂x = 3x² − 3y
    ∂f/∂y = −3x + 2y
    ```
    
2. Solve ∇f=0:
    
    ```
    3x² − 3y = 0   → y = x²
    −3x + 2y = 0  → 2y = 3x  → y = 1.5x
    ```
    
    Combine: x² = 1.5x → x(x − 1.5)=0 → x=0 or x=1.5
    
    → points: (0,0) and (1.5, 2.25)
    
3. Hessian:
    
    ```
    f_xx = 6x      f_xy = −3
    f_yx = −3     f_yy = 2
    ```
    
4. Evaluate at (0,0):
    
    D = (6·0)*(2) − (−3)² = 0 − 9 = −9 < 0 → **saddle**
    
5. At (1.5,2.25):
    
    f_xx = 6·1.5 = 9
    
    D = 9·2 − 9 = 18 − 9 = 9 > 0 and f_xx>0 → **local minimum**
    

### 7. Practice Problems

1. **Find and classify** all critical points of
    
    ```
    f(x,y) = x² + xy + y² − 4x + 6y
    ```
    
2. **Use the determinant test** for
    
    ```
    f(x,y) = x⁴ − 2x²y + y²
    ```
    
3. **Multivariable**: Solve ∇f=0 forThen use eigenvalues of the 3×3 Hessian to classify.
    
    ```
    f(x,y,z) = x² + y² + z² − 2xy + 4yz − 6x + 2
    ```
    

### 8. Python Exercises

```python
import numpy as np

# Example function
def f(X):
    x, y = X
    return x**3 - 3*x*y + y**2

# Gradient and Hessian
def grad(X):
    x, y = X
    return np.array([3*x**2 - 3*y, -3*x + 2*y])

def hessian(X):
    x, y = X
    return np.array([[6*x, -3],
                     [-3,   2]])

# Find stationary points with Newton’s method from different starts
def find_stationary(start, lr=0.1, tol=1e-6):
    X = np.array(start, dtype=float)
    for _ in range(100):
        g = grad(X)
        H = hessian(X)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = lr * g
        X = X - delta
        if np.linalg.norm(g) < tol:
            break
    return X

starts = [(0.5,0.5), (2,2), (-1,1)]
for s in starts:
    pt = find_stationary(s)
    H = hessian(pt)
    eigs = np.linalg.eigvals(H)
    print("Start", s, "→ stationary", pt, "eigenvalues", eigs)
```

### 9. Geometric Intuition

- **Saddle point**: surface looks like a mountain pass—in one direction up, in another down.
- **Local minimum**: like the bottom of a bowl—curvature positive in all directions.
- **Local maximum**: like the top of a hill—curvature negative in all directions.

Plotting level sets (contours) around the critical point reveals their character:

- **Closed ellipses** for minima/maxima
- **Hyperbolas** for saddles

---

## Optimization Using Gradients – An Example

### 1. Conceptual Overview

Gradient‐based optimization finds the minimum of a smooth function by following the negative of its slope (gradient) at each point.

Think of standing on a hill and wanting to reach the bottom. You look around, see which direction decreases altitude fastest (that’s the negative gradient), take a small step that way, then repeat. Over many steps, you’ll arrive at the lowest point.

### 2. Problem Definition

We’ll work with a simple two‐variable “bowl” function whose minimum is known.

Function:

```
f(x, y) = (x − 3)^2 + (y + 2)^2
```

- Its lowest value is 0, achieved at (x, y) = (3, −2).
- We’ll start from an arbitrary point, compute the gradient, and iteratively update (x, y) to approach the minimum.

### 3. Computing the Gradient

The gradient ∇f is the vector of partial derivatives:

```
∇f(x, y) = [ ∂f/∂x ,  ∂f/∂y ]
```

For our f:

```
∂f/∂x = 2 · (x − 3)
∂f/∂y = 2 · (y + 2)
```

Each component tells how steeply f rises if we move only in the x‐ or y‐direction.

### 4. Gradient Descent Update Rule

We update (x, y) by stepping in the opposite direction of the gradient:

```
x_new = x_old − α · ∂f/∂x (x_old, y_old)
y_new = y_old − α · ∂f/∂y (x_old, y_old)
```

- α (alpha) is the **learning rate** or step size.
- If α is too large, we may overshoot the minimum.
- If α is too small, convergence will be slow.

### 5. Step‐by‐Step Update

1. **Initialize** (x₀, y₀) anywhere (e.g., (−5, 5)).
2. **Compute** gradient at current point:
    
    ```
    gx = 2·(x_old − 3)
    gy = 2·(y_old + 2)
    ```
    
3. **Update** coordinates:
    
    ```
    x_new = x_old − α·gx
    y_new = y_old − α·gy
    ```
    
4. **Repeat** steps 2–3 until the change in f(x, y) or in (x, y) is below a threshold.

### 6. Python Implementation and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
f = lambda x, y: (x - 3)**2 + (y + 2)**2
grad = lambda x, y: np.array([2*(x - 3), 2*(y + 2)])

# Hyperparameters
alpha = 0.1           # learning rate
n_steps = 30          # number of iterations
xy = np.array([-5.0, 5.0])  # starting point

# Store path for plotting
path = [xy.copy()]

for _ in range(n_steps):
    g = grad(xy[0], xy[1])
    xy = xy - alpha * g
    path.append(xy.copy())

path = np.array(path)

# Plot contours and descent path
x_vals = np.linspace(-6, 6, 200)
y_vals = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.figure(figsize=(6,6))
contours = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot(path[:,0], path[:,1], 'o-', color='red', label='Descent path')
plt.scatter([3], [-2], color='black', label='Minimum (3, -2)')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Gradient Descent on f(x,y)')
plt.legend()
plt.show()
```

Run this in Jupyter to see how the red markers march down the contours toward (3, −2).

### 7. Practice Problems

1. By hand, perform **3 gradient‐descent updates** on f(x,y) starting from (0,0) with α=0.2.
2. Modify the code to try different α values (0.01, 0.5, 1.0) and observe convergence or divergence.
3. Change f toand adapt the gradient. Run gradient descent and plot the new path.
    
    ```
    f(x,y) = 2(x − 1)^2 + 5(y + 3)^2
    ```
    
4. Extend to three variables:Implement gradient descent in Python to find (2, −1, 4).
    
    ```
    f(x,y,z) = (x−2)^2 + 2(y+1)^2 + 3(z−4)^2
    ```
    

### 8. Geometric Interpretation

- Contour lines on the plot are “level sets” where f is constant.
- The gradient at each point is **perpendicular** to the contour and points uphill.
- By stepping opposite to that vector, you move “downhill” toward the nearest minimum.

### 9. Real‐World ML Application

In linear regression we minimize the mean squared error

```
J(w, b) = (1/m) Σᵢ (w·xᵢ + b − yᵢ)²
```

The gradient w.r.t. w and b is

```
∂J/∂w = (2/m) Σᵢ (w·xᵢ + b − yᵢ)·xᵢ
∂J/∂b = (2/m) Σᵢ (w·xᵢ + b − yᵢ)
```

TensorFlow and PyTorch under the hood perform exactly these steps—compute gradients, step parameters—millions of times when training large models.

---

## Optimization Using Gradients – Analytical Method

### 1. Conceptual Overview

In gradient‐based optimization you can either

- **Iterate** using gradient descent (numerical steps), or
- **Solve analytically** by finding where the gradient vanishes (setting derivatives to zero).

The **analytical method** finds exact minima or maxima by:

1. Computing the gradient vector (all first partials).
2. Setting each component of the gradient equal to zero.
3. Solving the resulting system of equations.
4. Verifying with second‐derivative tests (Hessian).

This is the “closed-form” approach that underlies normal equations in linear regression.

### 2. One-Variable Example

### Function

```
f(x) = x^2 - 4x + 5
```

### Step 1: Compute derivative

```
d/dx [f(x)] = 2*x - 4
```

### Step 2: Set derivative to zero

```
2*x - 4 = 0
```

### Step 3: Solve for x

```
x* = 4 / 2
   = 2
```

### Step 4: Verify minimum

- Second derivative:
    
    ```
    d^2/dx^2 [f(x)] = 2  (> 0 → local minimum)
    ```
    
- Value at optimum:
    
    ```
    f(2) = 2^2 - 4·2 + 5 = 4 - 8 + 5 = 1
    ```
    

### 3. Two-Variable Example

### Function

```
f(x,y) = 2*x^2 + 3*y^2 - 4*x + 6*y + 10
```

### Step 1: Compute gradient

```
∇f(x,y) = [ ∂f/∂x , ∂f/∂y ]
∂f/∂x = 4*x - 4
∂f/∂y = 6*y + 6
```

### Step 2: Set each partial to zero

```
4*x - 4 = 0
6*y + 6 = 0
```

### Step 3: Solve for (x, y)

```
x* = 4 / 4 = 1
y* = -6 / 6 = -1
```

### Step 4: Verify via Hessian

```
H = [ f_xx  f_xy ]   =  [ 4   0 ]
    [ f_yx  f_yy ]      [ 0   6 ]

Eigenvalues: 4 > 0, 6 > 0  → positive definite → local minimum
```

Value at optimum:

```
f(1, -1) = 2(1)^2 + 3(1)^2 - 4·1 + 6·(-1) + 10
         = 2 + 3 - 4 - 6 + 10
         = 5
```

### 4. Analytic Solution for Linear Regression

For a dataset with inputs X (m×n) and targets y (m×1), and a linear model ŷ = X·w, the squared‐loss objective:

```
J(w) = (1/(2m)) ‖X·w - y‖^2
```

### Gradient

```
∇J(w) = (1/m)·Xᵀ (X·w - y)
```

### Set gradient to zero

```
Xᵀ (X·w - y) = 0
```

### Normal equations

```
Xᵀ X · w = Xᵀ y
```

### Closed-form solution

```
w* = (Xᵀ X)^(-1) Xᵀ y
```

This analytic formula directly computes the weight vector that minimizes the MSE.

### 5. Python Example: Solving Gradient=0

```python
import sympy as sp

# Symbolic variables
x, y = sp.symbols('x y')
f = 2*x**2 + 3*y**2 - 4*x + 6*y + 10

# Compute gradient
fx = sp.diff(f, x)
fy = sp.diff(f, y)

# Solve fx=0, fy=0
solution = sp.solve([fx, fy], [x, y])
x_star, y_star = solution[x], solution[y]

# Hessian and test
H = sp.hessian(f, (x, y))
eigs = H.subs({x: x_star, y: y_star}).eigenvals()

print("Critical point:", (x_star, y_star))
print("Hessian eigenvalues:", eigs)
```

### 6. Practice Problems

1. By hand, find and classify the stationary points of
    
    ```
    f(x,y) = x^2 + x*y + y^2 - 2x + 3y + 7
    ```
    
2. Derive the normal-equation solution for 2-feature linear regression (with bias term), and implement it in NumPy.
3. Use Sympy (or any CAS) to solve
    
    ```
    f(x,y,z) = x^2 + y^2 + z^2 - 2xy + 4yz - 3xz + 5x - y + 2
    ```
    
    Find critical (x,y,z), compute Hessian, and classify the point.
    

### 7. Geometric Interpretation

- **Setting ∇f=0** finds points where all tangent‐plane slopes vanish—“flat spots.”
- **Hessian definiteness** tells whether that flat spot is a basin (min), peak (max), or saddle.
- In ML, analytic solutions like normal equations exploit the same idea to “solve” for the minimum rather than iterating.

### 8. Real-World Applications

- **Ridge regression** extends the normal equations by adding λI to XᵀX:
    
    ```
    (XᵀX + λI)w = Xᵀy
    ```
    
- **Gaussian process regression** uses closed-form formulas for posterior means by solving linear systems of the form K·α = y.
- **Quadratic programming** in SVMs or portfolio optimization solves gradient‐zero conditions under constraints.

---

## Optimization Using Gradient Descent – One Variable (Part 1)

### 1. What You Need to Know First

You should already understand how to compute the derivative of a single‐variable function:

```
d/dx [ f(x) ] = f′(x)
```

This gives the instantaneous slope at any point x on the curve y=f(x). In gradient descent, that slope tells us which way is “uphill” and, by negating it, which way is “downhill.”

### 2. Conceptual Overview

Imagine you’re on a hilly road described by height = f(x). You want to find the lowest point in the valley.

- At your current position x₀, compute the slope f′(x₀).
- If the slope is positive, the hill is rising to the right, so you step left.
- If the slope is negative, you step right.
- You repeat small steps until you can’t go any lower.

This step‐by‐step process is gradient descent in one dimension.

### 3. The Update Rule Formula

```
x_new = x_old - α * f′(x_old)
```

- `x_old`: your current position on the x-axis
- `f′(x_old)`: slope of f at that position
- `α` (alpha): learning rate, a small positive constant
- `x_new`: updated position after one step

Step-by-step:

1. Evaluate the derivative f′(x_old).
2. Multiply by the learning rate α to scale your step.
3. Subtract that product from x_old to move downhill.

### 4. Real-World Data Science Example

In regularized linear regression, you might optimize the bias term b in:

```
L(b) = (1/m) Σ (w·xᵢ + b - yᵢ)²
```

Treating w as fixed, gradient descent on L(b) uses

```
db = d/db [L(b)] = (2/m) Σ (w·xᵢ + b - yᵢ)
b_new = b_old - α * db
```

This tunes the intercept to best center predictions on your training data.

### 5. Worked Example: Quadratic Function

Minimize

```
f(x) = x² - 4x + 5
```

1. Compute derivative:
    
    ```
    f′(x) = 2x - 4
    ```
    
2. Pick a starting point, say x₀ = 0, and a learning rate α = 0.1.
3. First update:
    
    ```
    x₁ = x₀ - α * f′(x₀)
       = 0  - 0.1 * (2*0 - 4)
       = 0  - 0.1 * (-4)
       = 0.4
    ```
    
4. Next update uses x₁ = 0.4:
    
    ```
    f′(0.4) = 2*0.4 - 4 = -3.2
    x₂ = 0.4 - 0.1 * (-3.2) = 0.72
    ```
    
5. Repeat until x converges near the true minimum at x=2.

### 6. Python Implementation

```python
import numpy as np

# Define f and its derivative
f  = lambda x: x**2 - 4*x + 5
df = lambda x: 2*x - 4

# Hyperparameters
alpha    = 0.1
n_steps  = 20
x        = 0.0  # start

# Gradient descent loop
trajectory = [x]
for _ in range(n_steps):
    grad = df(x)
    x   = x - alpha * grad
    trajectory.append(x)

print("Descent path:", trajectory)
```

Run this in Jupyter to see how x moves toward the minimum.

### 7. Practice Problems

- By hand, perform **5 gradient descent updates** on
    
    ```
    f(x) = (x - 3)**2 + 2
    ```
    
    starting at x₀ = 0 with α = 0.2.
    
- Modify the Python code above to:
    - Use α = 0.01 and α = 0.5, observe which converges and which diverges.
    - Track and plot f(x) at each step:
        
        ```python
        import matplotlib.pyplot as plt
        
        xs = trajectory
        plt.plot(xs, [f(x) for x in xs], '-o')
        plt.xlabel('x'); plt.ylabel('f(x)'); plt.title('Loss over iterations')
        plt.show()
        ```
        
- Apply gradient descent to find the minimum of
    
    ```
    f(x) = x**4 - 3*x**3 + 2
    ```
    
    and compare with the analytical solution.
    

### 8. Visual Intuition

Picture the curve y = f(x) on a 2D plot:

- At each x, draw the tangent line whose slope is f′(x).
- The update rule picks a point on that tangent and steps down the hill.
- Plotting the sequence of (x, f(x)) shows red dots walking downhill along the curve.

---

## Optimization Using Newton’s Method – One Variable (Method 2)

### 1. Prerequisites

You should already know how to compute the first derivative f′(x) (slope) and the second derivative f″(x) (curvature) of a single‐variable function【Optimization Using Gradient Descent – One Variable (Part 1)】.

Newton’s method uses both to jump directly toward the extremum instead of taking small fixed steps.

### 2. Conceptual Overview

Gradient descent takes a small step proportional to the slope f′(x). Newton’s method adjusts that step by dividing by the curvature f″(x), giving:

- **Large steps** when curvature is small (flat region)
- **Small steps** when curvature is large (steep region)

This often converges much faster—quadratically—near a minimum or maximum, as long as f″(x)≠0.

### 3. Update Rule and Breakdown

```
x_new = x_old - f′(x_old) / f″(x_old)
```

- `f′(x_old)`: current slope—direction and magnitude of ascent
- `f″(x_old)`: current curvature—how steeply the slope itself is changing
- Dividing slope by curvature scales the step, correcting for local shape
- Subtracting moves you toward a stationary point f′(x)=0

### 4. Step‐by‐Step Procedure

1. **Initialize** `x = x₀` near where you expect a minimum or maximum.
2. **Repeat** until convergence or small update:a. Compute `slope = f′(x)` and `curvature = f″(x)`.b. Update
    
    ```
    x ← x - slope / curvature
    ```
    
3. **Stop** when `|slope|` or `|x_new - x_old|` falls below a threshold.

### 5. Worked Example: Quadratic Function

Minimize

```
f(x) = x^2 - 4x + 5
```

1. Compute derivatives:
    
    ```
    f′(x) = 2x - 4
    f″(x) = 2
    ```
    
2. Newton update simplifies to:
    
    ```
    x_new = x_old - (2x_old - 4) / 2
          = x_old - x_old + 2
          = 2
    ```
    
3. In one step from any `x_old`, you jump exactly to the minimum `x=2`.

### 6. Real‐World ML/DS Example: Univariate Logistic Regression

We fit a single weight `w` for data `(xᵢ,yᵢ)` using log‐loss:

```
J(w) = -(1/m) Σ [ yᵢ·ln(σ(w·xᵢ)) + (1−yᵢ)·ln(1−σ(w·xᵢ)) ]
σ(z) = 1 / (1 + e^(−z))
```

### Derivatives

```
∂J/∂w  = (1/m) Σ (σ(w xᵢ) − yᵢ)·xᵢ
∂²J/∂w² = (1/m) Σ σ(w xᵢ)·(1−σ(w xᵢ))·xᵢ²
```

### Newton Update

```
w_new = w_old - (∂J/∂w) / (∂²J/∂w²)
```

This uses curvature of the loss to adapt the step size, often converging in only a few iterations.

### 7. Python Implementation

```python
import numpy as np

# Example function f(x) and its derivatives
f  = lambda x: x**2 - 4*x + 5
df = lambda x: 2*x - 4
d2f= lambda x: 2

# Newton's method parameters
x = 0.0         # initial guess
n_iter = 5

for i in range(n_iter):
    slope     = df(x)
    curvature = d2f(x)
    x_new     = x - slope / curvature
    print(f"Iter {i}: x={x:.6f}, f(x)={f(x):.6f}")
    if abs(slope) < 1e-6:
        break
    x = x_new

print("Estimated minimum at x =", x)
```

### 8. Practice Problems

1. Apply Newton’s method to
    
    ```
    f(x) = x^4 - 3*x^3 + 2
    ```
    
    Starting at `x₀=0.5`, run 5 iterations and compare to the true minimizer.
    
2. For univariate logistic regression on synthetic data:
    
    ```python
    x = np.linspace(-3,3,100)
    y = (x > 0).astype(int)
    ```
    
    Implement Newton’s method to find `w` that minimizes `J(w)` and plot convergence of `J(w)`.
    
3. Compare convergence of gradient descent (α=0.1) vs Newton’s method on `f(x) = (x-2)^2 + 1`.

### 9. Visual Intuition

- For a non‐quadratic f(x), graph f and draw the tangent line at `x_old`.
- The Newton update picks the root of that tangent (where the line crosses the x‐axis) as `x_new`.
- Repeating this “tangent‐root” step zooms in on the extremum much faster than stepping a fixed fraction of the slope.

---

## Optimization Using Gradient Descent – Two Variables (Part 1)

### 1. Prerequisites

Before we tackle two‐variable gradient descent, make sure you’re solid on:

- Single‐variable gradient descent: update rule
    
    ```
    x_new = x_old - α * f′(x_old)
    ```
    
- Partial derivatives ∂f/∂x and ∂f/∂y for functions f(x,y).
- Forming the gradient vector ∇f = [∂f/∂x, ∂f/∂y].

### 2. Conceptual Overview

When your objective f depends on two parameters (x, y), the surface z = f(x,y) is a 3D landscape. Gradient descent in two variables:

1. **Measures slope** in the x‐direction (∂f/∂x) and y‐direction (∂f/∂y).
2. **Moves** the point (x,y) opposite to that 2D slope—downhill on the surface.
3. **Repeats** until you settle near a valley (local minimum).

### 3. Update Rule in Two Variables

In code‐block form, the simultaneous updates are:

```
x_new = x_old - α * (∂f/∂x)(x_old, y_old)
y_new = y_old - α * (∂f/∂y)(x_old, y_old)
```

Breaking it down:

- `x_old`, `y_old`: current coordinates.
- `(∂f/∂x)(x_old, y_old)`: slope if you move slightly in x, keeping y fixed.
- `(∂f/∂y)(x_old, y_old)`: slope if you move slightly in y, keeping x fixed.
- `α`: learning rate (step size)—controls how big each move is.

### 4. Vectorized Notation

Stack (x, y) into a parameter vector θ:

```
θ = [ x
      y ]
∇f(θ) = [ ∂f/∂x
          ∂f/∂y ]
θ_new = θ_old - α · ∇f(θ_old)
```

This compact form extends naturally to n variables (θ∈ℝⁿ).

### 5. Worked Example: Simple Bowl

Let’s minimize

```
f(x,y) = x² + y²
```

– a perfectly symmetric “bowl” with its bottom at (0,0).

1. **Compute partials**:
    
    ```
    ∂f/∂x = 2·x
    ∂f/∂y = 2·y
    ```
    
2. **Write updates**:
    
    ```
    x_new = x_old - α * (2·x_old)
    y_new = y_old - α * (2·y_old)
    ```
    
3. **Interpret**:– If x>0, ∂f/∂x>0 → we move x downward toward 0.– If y<0, ∂f/∂y<0 → we move y upward toward 0.

### 6. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Define f and its gradient
f = lambda x, y: x**2 + y**2
grad = lambda x, y: np.array([2*x, 2*y])

# 2. Hyperparameters
alpha   = 0.1           # learning rate
n_steps = 25            # number of iterations
pt      = np.array([3.0, -4.0])  # starting point

# 3. Run gradient descent
path = [pt.copy()]
for _ in range(n_steps):
    g   = grad(pt[0], pt[1])
    pt -= alpha * g
    path.append(pt.copy())
path = np.array(path)

# 4. Plot contour and path
xs = np.linspace(-5, 5, 200)
ys = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

plt.figure(figsize=(6,6))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.plot(path[:,0], path[:,1], 'o-', color='red', label='GD path')
plt.scatter([0], [0], color='black', label='Minimum (0,0)')
plt.xlabel('x'); plt.ylabel('y')
plt.title('2D Gradient Descent on f(x,y)=x²+y²')
plt.legend()
plt.show()
```

### 7. Choosing the Learning Rate α

- **Too large**: steps overshoot the minimum, can diverge.
- **Too small**: convergence is painfully slow.
- Good practice:
    - Start with α≈0.1 or 0.01.
    - Monitor f(x,y) over iterations; if it bounces or grows, reduce α.

### 8. Practice Problems

1. By hand, perform **3 updates** of GD on
    
    ```
    f(x,y) = 2(x−1)² + 5(y+2)²
    ```
    
    starting at (x,y)=(4, −1) with α=0.1.
    
2. Implement the Python code above for
    
    ```
    f(x,y) = x² + x·y + y²
    ```
    
    Compute partials, plot the descent path from (3,3).
    
3. Experiment with different α values (0.01, 0.2) and observe which ones converge for each f.

### 9. Real‐World ML/DS Example

If your loss is

```
J(w,b) = (1/m) Σ (w·xᵢ + b - yᵢ)²
```

and you choose to optimize both w and b simultaneously:

- Partial w:
    
    ```
    ∂J/∂w = (2/m) Σ (w·xᵢ + b - yᵢ)·xᵢ
    ```
    
- Partial b:
    
    ```
    ∂J/∂b = (2/m) Σ (w·xᵢ + b - yᵢ)
    ```
    
- Updates:
    
    ```
    w_new = w_old - α * (∂J/∂w)
    b_new = b_old - α * (∂J/∂b)
    ```
    

Under the hood, frameworks like scikit‐learn or TensorFlow do exactly this for linear and logistic regression.

---

## Optimization Using Gradient Descent – Least Squares

### 1. Prerequisites

You should be comfortable with:

- Partial derivatives and gradients【Gradients】
- Gradient descent in one and two dimensions【Optimization Using Gradient Descent – One Variable (Part 1)】【Optimization Using Gradient Descent – Two Variables (Part 1)】
- Basic matrix–vector notation (vectors as lists, dot‐products)

### 2. Conceptual Overview

Least squares (mean squared error) is the workhorse loss for fitting linear models.

- We have data points ((x_i, y_i)).
- Our model predicts (\hat y_i = w \cdot x_i + b).
- We measure average squared error between predictions and targets.
- Gradient descent tweaks (w) and (b) to drive that error down.

This process underlies linear regression, polynomial fit, and many feature‐engineering pipelines in ML/DS.

### 3. The Least Squares Objective

For (m) samples with feature vectors (x_i\in\mathbb{R}^n) and targets (y_i), define

```
J(w, b) = (1 / (2m)) * sum_{i=1..m} [ (w · x_i + b - y_i)^2 ]
```

- `w` is an n-dimensional weight vector.
- `b` is a scalar bias.
- The factor `1/(2m)` simplifies gradients (cancels the 2).

### 4. Deriving the Gradients

We need partial derivatives of (J) w.r.t. each parameter.

### ∂J/∂w

```
∂J/∂w = (1 / m) * sum_{i=1..m} [ (w · x_i + b - y_i) * x_i ]
```

- Error term: `(w · x_i + b - y_i)`
- Multiply by `x_i` to see how each feature contributes

### ∂J/∂b

```
∂J/∂b = (1 / m) * sum_{i=1..m} [ (w · x_i + b - y_i) ]
```

- Same error, times 1 since `b` shifts the prediction equally for all features

### 5. Update Rules

Stack parameters into a single vector (\theta = [b; w]) and features into augmented vectors (\tilde x_i = [1; x_i]). The gradient descent step becomes:

```
for each iteration:
    w = w - α * (1/m) * sum_i [ (w·x_i + b - y_i) * x_i ]
    b = b - α * (1/m) * sum_i [ (w·x_i + b - y_i) ]
```

Or in vector form:

```
θ_new = θ_old - α * (1/m) * Xᵀ (X·w + b·1 - y)
```

- `X` is the m×n design matrix
- `1` is an m-vector of ones
- `y` is the m-vector of targets

### 6. Geometric Interpretation

- The loss surface (J(w,b)) is a convex “bowl” in ((w,b))-space.
- The gradient vector ([\partial J/∂w; ∂J/∂b]) points uphill; stepping opposite moves you downhill.
- Each iteration “slides” the plane (w·x + b) closer to the cloud of points ((x_i,y_i)).

### 7. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate toy data
np.random.seed(0)
m = 50
X = 2 * np.random.rand(m, 1)              # single feature
y = 4 + 3 * X[:, 0] + np.random.randn(m)  # true w=3, b=4 with noise

# 2. Hyperparameters
alpha   = 0.1
n_iter  = 1000
w       = np.random.randn(1)  # initialize weight
b       = 0.0
costs   = []

# 3. Gradient descent loop
for _ in range(n_iter):
    y_pred = w * X[:, 0] + b
    error  = y_pred - y
    grad_w  = (1/m) * np.dot(error, X[:, 0])
    grad_b  = (1/m) * error.sum()
    w      -= alpha * grad_w
    b      -= alpha * grad_b
    cost   = (1/(2*m)) * np.sum(error**2)
    costs.append(cost)

print("Learned w, b:", w[0], b)

# 4. Plot cost over iterations
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Convergence of Gradient Descent")
plt.show()

# 5. Plot fit
plt.scatter(X, y, label="Data")
plt.plot(X, w*X + b, color='red', label="Fitted line")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.show()
```

### 8. Practice Problems

1. **By hand**, derive ∂J/∂w and ∂J/∂b for a single feature regression.
2. **Experiment** in code: try different α (0.01, 0.5, 1.0) and observe convergence or divergence of cost.
3. **Extend** the code to two features:Fit both w₁ and w₂ using gradient descent.
    
    ```
    y = 5 + 2*x1 - x2 + noise
    ```
    
4. **Compare** your GD solution to the closed‐form normal equationby computing both and measuring prediction error.
    
    ```
    θ = (XᵀX)^(-1) Xᵀ y
    ```
    

### 9. Real-World ML/DS Example

- **House price prediction:** features like square footage and age of home → target price.
- **Demand forecasting:** weather variables → electricity load.
- **Calibration tasks:** sensor readings → actual values.

In all these, least squares and gradient‐based solvers are pillars of model fitting.

---

## Optimization Using Gradient Descent – Least Squares with Multiple Observations

### 1. Conceptual Overview

In real‐world regression you almost never fit a model to a single data point—you have **m** observations, each with one or more features.

Least squares with multiple observations finds the best linear relationship by minimizing the average squared error across all training examples.

Gradient descent then iteratively adjusts model parameters to “slide” the predicted line (or hyperplane) closer to the cloud of points.

### 2. Problem Setup

- You have **m** data points ((x^{(i)}, y^{(i)})), (i=1,2,\dots,m).
- Each input (x^{(i)}\in\mathbb{R}^n) is an n-dimensional feature vector.
- You posit a linear model with weights (w\in\mathbb{R}^n) and bias (b\in\mathbb{R}):
    
    ```
    y_pred^{(i)} = w·x^{(i)} + b
    ```
    

Your goal is to find ((w,b)) that minimize the average squared error across all m samples.

### 3. Least Squares Cost Function

Define the mean squared error (MSE) loss:

```
J(w, b) = (1/(2m)) * sum_{i=1..m} ( (w·x^{(i)} + b - y^{(i)})^2 )
```

- `w·x^{(i)}` is the dot-product between weights and features for the i-th sample.
- `b` is added to every prediction.
- The factor `1/(2m)` normalizes by the number of samples and cancels the `2` in differentiation.

### 4. Deriving the Gradients

To apply gradient descent, compute the partial derivatives (gradients) of (J) w.r.t. each parameter.

### 4.1 Gradient w.r.t. w

```
∂J/∂w = (1/m) * sum_{i=1..m} [ (w·x^{(i)} + b - y^{(i)}) * x^{(i)} ]
```

- For each sample, error = `(prediction - actual)`.
- Multiply the error by the feature vector `x^{(i)}` to see how each weight influences that error.
- Sum over all samples and average by `1/m`.

### 4.2 Gradient w.r.t. b

```
∂J/∂b = (1/m) * sum_{i=1..m} [ (w·x^{(i)} + b - y^{(i)}) ]
```

- Same error term, but since `b` shifts every prediction by 1, multiply by 1.
- Sum and average across m samples.

### 5. Gradient Descent Update Rules

Using learning rate α:

```
w_new = w_old - α * (1/m) * sum_{i=1..m} [ (w_old·x^{(i)} + b_old - y^{(i)}) * x^{(i)} ]

b_new = b_old - α * (1/m) * sum_{i=1..m} [ (w_old·x^{(i)} + b_old - y^{(i)}) ]
```

Or vectorized: stack samples into matrix **X** (m×n) and vector **y** (m×1), define **1** as an m-vector of ones:

```
predictions = X·w_old + b_old·1
errors      = predictions - y

grad_w = (1/m) * Xᵀ · errors
grad_b = (1/m) * sum(errors)

w_new = w_old - α·grad_w
b_new = b_old - α·grad_b
```

### 6. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(42)
m = 100
n = 2
X = np.random.randn(m, n)       # 2 features
true_w = np.array([2.5, -1.0])
true_b = 0.5
y = X.dot(true_w) + true_b + np.random.randn(m) * 0.5

# Hyperparameters
alpha   = 0.1
n_iter  = 500
w       = np.zeros(n)
b       = 0.0
costs   = []

# Gradient descent loop
for _ in range(n_iter):
    preds = X.dot(w) + b                    # shape (m,)
    errors = preds - y                      # shape (m,)
    grad_w = (1/m) * X.T.dot(errors)        # shape (n,)
    grad_b = (1/m) * np.sum(errors)         # scalar

    w -= alpha * grad_w
    b -= alpha * grad_b

    cost = (1/(2*m)) * np.sum(errors**2)
    costs.append(cost)

print("Learned w:", w, "b:", b)
print("True   w:", true_w, "b:", true_b)

# Plot cost convergence
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Gradient Descent Convergence")
plt.show()
```

### 7. Geometric Intuition

- **Each iteration** adjusts the hyperplane (w·x + b) to better align with the data cloud.
- **Errors** form a vector in ℝᵐ; projecting it back via **Xᵀ** computes how to steer each weight.
- The cost surface in (w, b)-space is a convex bowl—gradient descent “rolls” downhill toward its bottom.

### 8. Real-World ML/DS Applications

- **Multi-feature linear regression:** predicting housing prices from square footage, age, location coordinates.
- **Time-series forecasting:** using past lags as features to predict future values.
- **Feature-engineered models:** feeding polynomial and interaction terms into a gradient-descent solver.

### 9. Practice Problems

1. **By hand**, derive ∂J/∂w and ∂J/∂b for the case of **three observations** with a single feature.
2. **Modify** the Python code:
    - Try different learning rates α = [0.01, 0.5, 1.0] and plot their cost curves.
    - Change noise level in y and observe effect on convergence.
3. **Extend to mini-batch GD**: update parameters using batches of size 10 instead of full m each iteration.
4. **Compare** GD vs closed-form: compute normal-equation solutionand report mean squared error for both methods.
    
    ```python
    w_normal = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    b_normal = y.mean() - w_normal.dot(X.mean(axis=0))
    ```
    

---