# Mathematics_C1_W1

## Linear Algebra Applied 1

### Systems of Linear Equations

Welcome to our deep dive on **systems of linear equations**—the mathematical backbone of linear regression and so many ML algorithms.

### 1. What Is a System of Linear Equations?

A **linear equation** in two variables (x, y) looks like:

- It graphs as a straight line in the plane.
- Its general form:
where a, b, c are known numbers, and x, y are unknowns.
    
    ```
    a·x + b·y = c
    ```
    

A **system** is just multiple such equations that must hold simultaneously. For example:

- Equation 1: 2x + 3y = 6
- Equation 2: x − y = 1

The solution is the (x, y) point where these two lines intersect.

**Why it matters:**

- A unique intersection means one solution.
- Parallel lines → no solution.
- Same line twice → infinitely many solutions.

In ML, each training example gives you one linear equation; finding model weights means solving all equations together.

### 2. From Many Equations to Matrix Form

### 2.1 The “by hand” example (2×2)

Equations:

```
2x + 3y =  6
 x −  y =  1
```

Rewrite as a matrix equation **A x = b**:

```
[ [2, 3],
  [1, -1] ]   ·   [x, y]ᵀ   =   [6, 1]ᵀ
```

- **A** is the 2×2 **coefficient matrix**.
- **x** is the column vector of unknowns.
- **b** is the column vector of constants.

### 2.2 General n-feature linear model (explained below in detail)

With n inputs x₁,…,xₙ and weight vector w, plus bias b:

```
y = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
```

In **vector form**, augment x with a 1 for bias:

```
Let x̄ = [1, x₁, x₂, …, xₙ]ᵀ
    w̄ = [ b,  w₁,  w₂, …, wₙ]ᵀ

Then each example satisfies
w̄ᵀ x̄ = y
```

Collecting M examples into matrix **X** (M×(n+1)) and vector **y** (M×1):

```
X · w̄ = y
```

- **X** rows are individual examples [1, x₁, …, xₙ].
- **w̄** holds bias and weights.
- **y** is the targets.

### 3. What Each Symbol Means

```
X \\; w̄ = y
```

- **X**
    - Shape: M rows (data points) × (n+1) columns (features + bias).
    - Entry Xᵢⱼ is feature j of example i (or 1 if j=0 for bias).
- **w̄**
    - Shape: (n+1)×1.
    - First element = bias b; next n elements = weights w₁…wₙ.
- **y**
    - Shape: M×1.
    - Entry yᵢ is the observed target for example i.

Solving for w̄ means finding the bias and weights that make all M equations as close to true as possible.

### 4. Geometric & Intuitive Picture

- **2 variables →** two lines in the plane.
- **3 variables →** three planes in 3D; their common point (if unique) solves all three equations.
- **n variables →** n–dimensional hyperplanes.

Linear regression picks a single hyperplane that “best” fits scattered data points—solving a system in the least-squares sense.

### 5. Practice Problems & Python

### 5.1 Beginner Exercise: Solve by Hand & with NumPy

1. By hand, solve:
    
    ```
    3x + 2y =  8
     x −  y =  1
    ```
    
2. Confirm with Python:
    
    ```python
    import numpy as np
    
    A = np.array([[3, 2],
                  [1, -1]])
    b = np.array([8, 1])
    
    solution = np.linalg.solve(A, b)
    print(solution)  # [x, y]
    ```
    

### 5.2 ML-style: Linear Regression via Normal Equations

Given data points for wind speed (x) and power (y):

```python
import numpy as np

# Sample data
X_raw = np.array([ [5.0], [3.2], [7.1], [4.3] ])   # wind speeds
y      = np.array([1500,  800, 2000, 1200])       # power outputs

# Augment with bias column
X = np.hstack([ np.ones((X_raw.shape[0],1)), X_raw ])  # shape (4,2)

# Normal equation: w̄ = (Xᵀ X)⁻¹ Xᵀ y
XtX = X.T.dot(X)
XtX_inv = np.linalg.inv(XtX)
w_bar = XtX_inv.dot(X.T).dot(y)

print("Bias (b):", w_bar[0])
print("Weight (w):", w_bar[1])
```

**Try this in your notebook** to see bias and weight for the best-fit line.

### 6. Real-World Relevance in ML/DS

- **Feature selection:** Each equation shows how features combine linearly—knowing if your coefficient matrix is singular (non-invertible) warns you about collinearity.
- **Training linear models:** Most closed-form solutions (normal equations, ridge regression) reduce to solving a system **XᵀX w̄ = Xᵀ y**.
- **Optimization kick-off:** Understanding systems underlies gradient computation, Hessians, and Newton’s method.

---

## General n-Feature Linear Model

### Intuitive Explanation

Imagine you’re cooking and each feature (x₁, x₂, …, xₙ) is an ingredient, while each weight (w₁, w₂, …, wₙ) is how much of that ingredient to use. The bias (b) is like a fixed base flavor you always add. Combining ingredients in different proportions produces your final dish (y).

In ML, each feature might be a measurement (e.g., wind speed, temperature, humidity), and you learn how much each one contributes to your prediction.

### Core Formula

```
y = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
```

### Breakdown

- w₁, w₂, …, wₙ
weights: how strongly each feature influences the output
- x₁, x₂, …, xₙ
feature values for one data point
- b
bias: a constant term that shifts the prediction up or down
- y
predicted output (target)

### Vectorized Form

1. Add a constant “1” feature to handle bias.
2. Stack bias and weights into one vector.

```
x̄ = [1, x₁, x₂, …, xₙ]ᵀ
w̄ = [b, w₁, w₂, …, wₙ]ᵀ

Then for each example:
y = w̄ᵀ · x̄
```

- x̄ is (n+1)×1
- w̄ is (n+1)×1
- Their dot product gives y

### Dataset-Level Matrix Form

Collect M examples into a matrix:

```
X = [ x̄¹ᵀ; x̄²ᵀ; …; x̄ᴹᵀ ]   (shape M×(n+1))
y = [ y¹; y²; …; yᴹ ]         (shape M×1)

Model:
X · w̄ = y
```

Solving for w̄ finds the best bias and weights across all M examples.

### Real-World Example

Predict wind turbine power using three features:

- x₁ = wind speed
- x₂ = temperature
- x₃ = humidity

Model:

```
y = w₁·(wind speed) + w₂·(temperature) + w₃·(humidity) + b
```

Here, w₂ might be small if temperature has little effect, while w₁ could be large if wind speed dominates.

---

## Linear Algebra Applied 2

### Systems of Linear Equations

A system of linear equations is simply a collection of straight-line equations that share the same unknowns. In machine learning, each training example gives one linear equation; solving the system means finding model weights that best explain all examples at once.

### 1. What Is a Linear Equation?

A single linear equation in two variables x and y has the form:

```
a·x + b·y = c
```

Here:

- a and b are known numbers (coefficients)
- x and y are unknowns you want to solve for
- c is a known constant

Geometrically, each such equation corresponds to one straight line in the plane. Any point (x, y) on that line makes the equation true.

### 2. What Makes It a “System”?

When you stack multiple linear equations that share the same unknowns, you get a system. For example:

```
2·x + 3·y = 6
  x −   y = 1
```

- You’re looking for one pair (x, y) that satisfies **both** equations simultaneously.
- Graphically, the solution is the intersection point of two lines.

Possible outcomes:

- Exactly one intersection → one unique solution.
- Parallel lines → no solution.
- Same line twice → infinitely many solutions.

### 3. From Equations to Matrix Form

When you have many features (unknowns) and many examples (equations), it’s cleaner to use matrices and vectors.

### 3.1 Writing Ax = b

Suppose you have M data points and n features. Let w = [w₁, w₂, …, wₙ]ᵀ be the weights and b the bias. Each data point i has feature vector x⁽ᶦ⁾ = [x₁⁽ᶦ⁾, x₂⁽ᶦ⁾, …, xₙ⁽ᶦ⁾] and target y⁽ᶦ⁾. The linear model is:

```
w₁·x₁⁽ᶦ⁾ + w₂·x₂⁽ᶦ⁾ + … + wₙ·xₙ⁽ᶦ⁾ + b = y⁽ᶦ⁾
```

Stack all M equations into one matrix equation:

```
X · w̄ = y
```

where

```
X = [ [1, x₁⁽¹⁾, …, xₙ⁽¹⁾],
      [1, x₁⁽²⁾, …, xₙ⁽²⁾],
      …,
      [1, x₁⁽ᴹ⁾, …, xₙ⁽ᴹ⁾] ]  (shape M×(n+1))

w̄ = [ b, w₁, w₂, …, wₙ ]ᵀ       (shape (n+1)×1)

y = [ y⁽¹⁾, y⁽²⁾, …, y⁽ᴹ⁾ ]ᵀ      (shape M×1)
```

- The first column of X is all 1’s to multiply the bias term b.
- Solving for w̄ means finding the bias and all weights that best fit the data.

### 4. Singular vs. Nonsingular Systems

A square system (same number of equations as unknowns) has a **coefficient matrix** A.

```
A · w = b
```

- If det(A) ≠ 0, A is **nonsingular** (invertible) and there’s a unique solution:
    
    ```
    w = A⁻¹ · b
    ```
    
- If det(A) = 0, A is **singular** and the system either has no solution or infinitely many.

In regression, XᵀX plays the role of A. If XᵀX is singular (due to collinear features), you can’t invert it directly.

### 5. Practice Problems

1. Convert the “course scores” sentences into a system of three equations in a, c, p.
2. Write that system in matrix form A·x = b.
3. Compute det(A) to decide if it’s singular or not.
4. Solve for [a, c, p]ᵀ by hand or using Python.

### 6. Python Exercise

```python
import numpy as np

# Example from your “scores” problem
A = np.array([[ 1,  1, -1],
              [ 1, -1,  2],
              [ 4, -2,  1]])
b = np.array([ 6, 4, 10])

# Check if A is singular
det_A = np.linalg.det(A)
print("det(A) =", det_A)

# Solve if possible
if abs(det_A) > 1e-8:
    solution = np.linalg.solve(A, b)
    print("Solution [a, c, p]:", solution)
else:
    print("Matrix is singular; no unique solution.")
```

Try running this in your notebook. Observe how det(A) determines solvability.

### 7. Why This Matters in ML

- Linear regression reduces to solving XᵀX·w̄ = Xᵀy.
- Checking singularity of XᵀX warns you about redundant features (collinearity).
- Understanding these concepts lays the groundwork for Gaussian elimination, least-squares, pseudoinverses, and gradient-based solvers.

---

## Analogy using an eg:

1. Imagine you’re baking batches of cookies.
    - Each batch (one data point) uses several ingredients: flour, sugar, butter, etc.
    - You have a “recipe” that tells you how much of each ingredient to use.
2. In our ML “recipe,”
    - **Features** (x₁, x₂, … xₙ) are your ingredient amounts for one batch—say, cups of flour, spoons of sugar, sticks of butter.
    - **Weights** (w₁, w₂, … wₙ) are the recipe proportions—how strongly each ingredient contributes to the final taste.
    - **Bias** (b) is a fixed base flavor you always add (like a pinch of salt) no matter what.
    - **Output** (y) is how good your cookies turn out—your predicted score.

For one batch, the recipe is simply:

```
predicted_score = w₁·(cups_of_flour)
                + w₂·(spoons_of_sugar)
                + …
                + wₙ·(sticks_of_butter)
                + b
```

1. Now imagine you baked many batches. You’d have a table (spreadsheet) where each row lists the ingredient amounts for one batch, and you have one recipe that you apply to every row.
    - That table is called **X** (rows = batches, columns = ingredient amounts).
    - Your recipe proportions plus the base flavor form a list we call **w**.
    - When you “multiply” the table by the recipe list, you get a new list of predicted scores for every batch, which we call **y**.
2. Why use a table-and-list instead of writing each line separately?
    - It’s tidy—one operation gives all predictions at once.
    - It generalizes from two or three ingredients to hundreds easily.
3. So in simple terms:
    - Build a table **X** of your data (ingredient amounts).
    - Keep your recipe **w** (weights + bias) in a list.
    - Multiply them to get predictions **y**.

That’s the leap from “many equations” to “matrix form.” You’ve gone from writing one recipe line per batch to saying, “Take the whole table, apply this one recipe, and out pops all the predicted scores.”

### 1. Our “Cookie Table” (X)

Imagine you baked two batches. You tracked only flour (cups) and sugar (spoons):

| Batch | Flour (x₁) | Sugar (x₂) |
| --- | --- | --- |
| 1 | 2 | 1 |
| 2 | 4 | 3 |

We add a “1” column so we can bake the same base flavor into every batch:

| Batch | 1 (bias) | Flour | Sugar |
| --- | --- | --- | --- |
| 1 | 1 | 2 | 1 |
| 2 | 1 | 4 | 3 |

That whole grid is **X** (2×3).

### 2. Our “Recipe” (w)

You decide:

- Base flavor adds 5 taste-points → bias b = 5
- Each cup of flour adds 2 points → w₁ = 2
- Each spoon of sugar adds 0.5 points → w₂ = 0.5

Stack those into one list:

```
w = [b, w₁, w₂] = [5, 2, 0.5]
```

### 3. Multiply Table × Recipe → Predictions

Multiply each row of X by w (dot product):

Batch 1 prediction

= 1·5 + 2·2 + 1·0.5

= 5 + 4 + 0.5

= 9.5 taste-points

Batch 2 prediction

= 1·5 + 4·2 + 3·0.5

= 5 + 8 + 1.5

= 14.5 taste-points

All at once, that’s just matrix multiplication:

```
[ [1, 2, 1],
  [1, 4, 3] ]   ·   [5, 2, 0.5]ᵀ   =   [ 9.5, 14.5 ]ᵀ
```

- X is 2×3, w is 3×1, result is 2×1 (one prediction per batch).

### 4. Python Demo

Try this in your notebook:

```python
import numpy as np

# 1. Build the table X
X = np.array([
    [1, 2, 1],  # batch 1: [bias, flour, sugar]
    [1, 4, 3]   # batch 2
])

# 2. Define your recipe w
w = np.array([5, 2, 0.5])  # [bias, w1, w2]

# 3. Multiply to get predictions
y_pred = X.dot(w)

print("Predicted taste-points:", y_pred)
# → [ 9.5 14.5 ]
```

### Why This Matters

- **Rows of X** = each example’s features (plus a 1 for bias).
- **w** = your shared “recipe” (bias + weights).
- Multiplying X by w spits out one prediction per example.

That’s exactly how we go from many “ingredient lists” to one neat matrix equation in machine learning.

---

## System of sentences

### Systems of Linear Equations

Before diving into symbols, let’s anchor ourselves with the sentence analogy you just saw:

- A **system of sentences** gives you information by combining simple statements.
- A **system of linear equations** does the exact same, but in math language.

Here, each “sentence” is a linear equation—like “2x + 3y = 5”—and the goal is to uncover the hidden values (variables) that make every sentence true at once.

### 1. Intuition & Analogy

Imagine you have two clues about two unknowns—say, the prices of apples (x) and bananas (y):

- Clue 1: “Two apples plus three bananas cost 5 dollars.”
- Clue 2: “One apple minus one banana costs 1 dollar.”

Each clue alone leaves multiple possibilities. Together, they pin down exactly one pair \((x,y)\). That’s a **complete** (non-singular) system with a **unique solution**.

### 2. From Sentences to Equations

Translate the fruit clues into math:

```markdown
2x + 3y = 5
1x - 1y = 1
```

Here’s what each part means:

- `2x` – two times the value of x (apples)
- `+ 3y` – plus three times the value of y (bananas)
- `= 5` – all together equals 5 dollars
- The second line is a similar statement for a different combination.

When you have **n** variables and **n** independent equations, you can often solve for a unique vector \(x\).

### 3. Matrix Formulation

All at once, we write the system as:

```markdown
A x = b
```

where

```markdown
A = [[2, 3],
     [1, -1]],    x = [x, y]ᵀ,    b = [5, 1]ᵀ
```

Step-by-step:

1. **A** is the coefficient matrix—each row holds the multipliers of variables.
2. **x** is the unknown vector we seek.
3. **b** is the constants vector (right-hand sides).

Solving \(A x = b\) means finding x so that when A multiplies it, you get b.

### 4. Solution Types & Singularity

Depending on A and b, you get one of three behaviors:

- **Unique solution** (complete/non-singular): exactly as many independent equations as variables.
- **Infinite solutions** (redundant/singular): equations repeat or don’t add new info.
- **No solution** (contradictory/singular): equations conflict.

In our fruit example, \(\det(A) = 2\cdot(-1) - 3\cdot1 = -5\neq0\), so A is invertible and the system is non-singular.

### 5. Geometric Interpretation

- In 2D (two variables), each equation is a line in the plane.
- **Intersection** of two lines (if they cross once) is the unique solution.
- Parallel lines never meet (no solution).
- Same line twice means infinite overlap (infinite solutions).

Visualizing these intersections helps build geometric intuition for higher dimensions, too.

### 6. Real-World ML/DS Example

Linear regression boils down to solving a (usually over-determined) system:

```
X w = y
```

- **X**: data matrix (samples × features)
- **w**: weight vector we solve for
- **y**: target values

Finding \(w\) that best fits all equations (samples) uses least squares, which hinges on solving normal equations \(XᵀX w = Xᵀy\).

### 7. Practice Problems

### 7.1 Basic Pencil-and-Paper

1. Solve:
    - \(3x + 2y = 16\)
    - \(x - 4y = -2\)
2. Classify each system as non-singular, redundant, or contradictory:
    - \(x + y = 4\)
    \(2x + 2y = 8\)
    - \(x + y = 4\)
    \(2x + 2y = 9\)

### 7.2 Python Exercises

```python
import numpy as np

# Define A and b for the fruit example
A = np.array([[2, 3],
              [1, -1]])
b = np.array([5, 1])

# Solve the system
x = np.linalg.solve(A, b)
print("Solution [x, y]:", x)
```

Try modifying A and b to hit singular cases:

- Set rows equal (redundant).
- Make them proportional but with different b (contradictory).

Observe `np.linalg.LinAlgError` on singular matrices.

---

## Systems of Linear Equations

### 1. Equations as Information-Sentences

Imagine each equation as a simple sentence that conveys one piece of information:

- “An apple plus a banana costs \$10.”
- “An apple plus two bananas costs \$12.”

Alone, each gives many possibilities. Together, they narrow down to the true prices of apple (a) and banana (b).

A **system of linear equations** is just several such “sentences” sharing the same unknowns (variables). Solving the system means finding values for all variables that make **every** sentence true at once.

### 2. What Makes an Equation “Linear”?

A **linear equation** in \(n\) variables \(x_1, x_2, \dots, x_n\) has this form:

```markdown
a₁·x₁ + a₂·x₂ + … + aₙ·xₙ = b
```

- \(a_i\) are known coefficients (numbers).
- \(x_i\) are unknowns you want to find.
- \(b\) is the constant term.

### Step-by-Step of the Formula

```markdown
a₁·x₁ + a₂·x₂ + … + aₙ·xₙ = b
```

- **\(a_i·x_i\)**: multiply coefficient \(a_i\) by variable \(x_i\).
- **“+ … +”**: sum up all terms for each variable.
- **“= b”**: the total equals the constant \(b\).

### Real-World ML/DS Example

- In a simple linear regression with two features \(x_1, x_2\) and weight vector \(\mathbf{w}=[w_1,w_2]\), one prediction is

\[
w_1 x_1 + w_2 x_2 = y.
\]

Here, \(w_1,w_2\) are coefficients and \(y\) is the predicted value.

## 3. Defining a System of Linear Equations

A **system** simply stacks multiple linear equations:

```markdown
a₁₁ x₁ + a₁₂ x₂ + … + a₁ₙ xₙ = b₁
a₂₁ x₁ + a₂₂ x₂ + … + a₂ₙ xₙ = b₂
…
aₘ₁ x₁ + aₘ₂ x₂ + … + aₘₙ xₙ = bₘ
```

- \(m\) equations, \(n\) unknowns.
- The goal is to find the vector \(\mathbf{x} = [x₁,x₂,…,xₙ]ᵀ\) that satisfies **all**.

### 4. Types of Solutions

Depending on how much information the system carries, you get:

| Behavior | What Happens | Analogy |
| --- | --- | --- |
| Unique solution | Exactly one \(\mathbf{x}\) works | Complete (non-singular) |
| Infinite solutions | Many \(\mathbf{x}\) satisfy all equations | Redundant (singular) |
| No solution | Equations contradict; no \(\mathbf{x}\) exists | Contradictory (singular) |
- **Non-singular (complete)**: as many independent equations as unknowns; no redundancy or contradictions.
- **Singular**: either redundant or contradictory.

### 5. Matrix Formulation

Compactly write \(m\) equations in \(n\) unknowns as:

```markdown
A · x = b
```

- **A** is the \(m×n\) coefficient matrix \(\bigl[a_{ij}\bigr]\).
- **x** is the unknown vector \([x₁, x₂, …, xₙ]ᵀ\).
- **b** is the constant vector \([b₁, b₂, …, bₘ]ᵀ\).

### Example (2×2)

```markdown
A = [[2, 3],     x = [x, y]ᵀ,    b = [5, 1]ᵀ
     [1, -1]]
```

Solving \(A·x=b\) finds the unique \((x,y)\) that satisfies both equations.

### 6. Geometric Intuition

- In 2D, each linear equation is a straight line.
- **Unique solution**: two lines intersect at one point.
- **Infinite solutions**: two lines coincide.
- **No solution**: two lines are parallel and never meet.

Extends to planes and hyperplanes in higher dimensions.

### 7. ML/DS Connection: Regression & Feature Systems

In least-squares linear regression you solve normal equations:

```markdown
Xᵀ X · w = Xᵀ y
```

- **X**: data matrix (samples×features)
- **w**: weight vector (unknowns)
- **y**: target vector

Whether \(XᵀX\) is singular or not determines if the regression has a unique solution or needs regularization.

## 8. Practice Problems

### 8.1 Pencil-and-Paper

1. Solve for \(x,y\):
    
    \[
    \begin{cases}
    3x + 2y = 16\\
    x - 4y = -2
    \end{cases}
    \]
    
2. Classify as unique, infinite, or no solution:
    - \(\;x+y=4\)
    \(\;2x+2y=8\)
    - \(\;x+y=4\)
    \(\;2x+2y=9\)

### 8.2 Python in Jupyter

```python
import numpy as np

# System: 3x + 2y = 16, x - 4y = -2
A = np.array([[3, 2],
              [1, -4]])
b = np.array([16, -2])

# Solve
x = np.linalg.solve(A, b)
print("Solution [x, y]:", x)

```

- Modify A and b so rows are identical → expect `LinAlgError`.
- Change b to create a contradiction → check rank with `np.linalg.matrix_rank`.

### 8.3 Real-World Data Twist

You have features “size” and “age” of houses; you observe two aggregate prices:

- House A: \(2\)·size + \(1\)·age = \$300k
- House B: \(1\)·size + \(2\)·age = \$250k

Set up the system and solve for unit price of each feature.

---

## System of equations on Lines and Planes

A system of linear equations can be viewed geometrically: in two variables each equation is a line in the plane, in three variables each equation is a plane in 3D space, and in higher dimensions each equation defines a hyperplane.  Solving the system means finding the intersection of these geometric objects.

### 1. Lines in 2D

A single linear equation in two variables

```markdown
a·x + b·y = c
```

is the equation of a straight line.

- Coefficients \(a,b\) determine the slope and orientation.
- Constant \(c\) shifts the line’s position.

### Intersection Cases for Two Lines

- **Unique intersection (one point)**
Lines cross at exactly one point → one solution.
- **Parallel lines**
Same slope, different intercepts → no solution.
- **Coincident lines**
Exactly the same line → infinitely many solutions.

### 2. Planes in 3D

A single linear equation in three variables

```markdown
a·x + b·y + c·z = d
```

is the equation of a plane in \(\mathbb{R}^3\).

### Intersection Cases for Three Planes

1. **Unique point**
Three planes meet at exactly one common point → unique solution.
2. **Line intersection**
Two planes meet in a line, the third contains that line → infinitely many solutions.
3. **Coincident plane**
Two or more planes coincide → redundancy → infinitely many solutions.
4. **No common intersection**
At least one plane is parallel to intersection of others → no solution.

### 3. General Hyperplanes

In \(n\) dimensions, each equation

```markdown
a₁·x₁ + a₂·x₂ + … + aₙ·xₙ = b
```

defines a hyperplane.  Solving \(m\) such equations means finding the intersection of \(m\) hyperplanes in \(n\)-dimensional space.

- If hyperplanes meet in a single point: unique solution.
- If they overlap more than needed: infinitely many solutions.
- If they do not share a common point: no solution.

### 4. Link to Matrix Rank

Write the system as

```markdown
A · x = b
```

- **Row space** of \(A\) spans the normals of hyperplanes.
- **Rank** of \(A\) = number of independent hyperplanes.
- **rank = number of variables** → non-singular → unique solution.
- **rank < number of variables** → singular → infinite solutions if consistent.
- **inconsistent** (augmented matrix rank > rank of \(A\)) → no solutions.

### 5. Real-World ML/DS Examples

- **Linear regression** in feature space:
    
    Each sample’s prediction equation defines a hyperplane in weight space.
    
    Solving normal equations \(XᵀX w = Xᵀy\) finds the intersection point (optimal weights).
    
- **Decision boundary** in classification:
    
    A linear classifier defines a hyperplane \(\mathbf{w}·\mathbf{x} + b = 0\).
    
    Intersection with data geometry separates classes.
    

### 6. Practice Problems

### 6.1 Pencil-and-Paper

1. Two lines intersecting once:
    
    ```
    2x + 3y = 6
    x -  y = 1
    ```
    
2. Parallel lines (no solution):
    
    ```
    4x + 2y = 8
    2x +  y = 5
    ```
    
3. Coincident lines (infinite solutions):
    
    ```
    x + 2y = 4
    2x + 4y = 8
    ```
    
4. Three planes in 3D:
    - Unique point:
        
        ```
        x + y + z = 3
        2x - y + z = 2
        x + 2y - z = 1
        ```
        
    - Intersection line:
        
        ```
        x + y + z = 3
        2x + 2y + 2z = 6
        x - y + z = 1
        ```
        
    - No solution:
        
        ```
        x + y + z = 3
        x + y + z = 4
        x - y + z = 1
        ```
        

### 6.2 Python Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two lines: 2x+3y=6 and x-y=1
x = np.linspace(-1, 4, 100)
y1 = (6 - 2*x)/3
y2 = x - 1

plt.plot(x, y1, label='2x+3y=6')
plt.plot(x, y2, label='x-y=1')
plt.legend()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.scatter(*np.linalg.solve(np.array([[2,3],[1,-1]]), [6,1]), color='red')
plt.title('Intersection of Two Lines')
plt.show()
```

Plot planes in 3D using `mpl_toolkits.mplot3d` to see their intersections.

---

## Graphical Representation of 2-Variable Linear Systems

A system of two linear equations in variables \(x\) and \(y\) can be visualized as two straight lines in the plane. Their intersection (if any) gives the solution to the system.

### 1. Line Equations in the Plane

Every linear equation in \(x,y\) defines a line:

```markdown
a·x + b·y = c
```

You can rewrite it in **slope-intercept form**:

```markdown
y = m·x + b
```

where

- `m = –a/b` is the **slope** (rise over run),
- `b = c/b` is the **y-intercept** (where the line crosses the y-axis).

## 2. Intersection Cases

Given two lines:

```markdown
Line 1: a₁·x + b₁·y = c₁

Line 2: a₂·x + b₂·y = c₂
```

there are three possible scenarios:

- **Unique intersection**
    - Slopes differ (`m₁ ≠ m₂`).
    - Lines cross at exactly one point → one solution.
- **Parallel lines (no solution)**
    - Slopes equal (`m₁ = m₂`) but intercepts differ (`b₁ ≠ b₂`).
    - Lines never meet → system is inconsistent.
- **Coincident lines (infinite solutions)**
    - Equations are proportional (`a₁/a₂ = b₁/b₂ = c₁/c₂`).
    - Same line → every point on it satisfies both → infinitely many solutions.

### 3. Visual Intuition with ASCII

Imagine axes and two lines:

```
   y
   ↑
  6│         • Intersection
  4│        /
  2│  _____/______
    │       /
 -2│      /
    └────────────────→ x
```

- The crossing point ● is the unique solution to the system.

### 4. Plotting in Python

Use Matplotlib to draw and find intersections:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define line parameters: a x + b y = c
lines = [
    (2, -1,  4),   # 2x - y = 4  → y = 2x - 4
    (1,  1,  6)    #  x + y = 6  → y = -x + 6
]

# Generate x values
x = np.linspace(-1, 5, 200)

# Plot each line
for a, b, c in lines:
    y = (c - a*x) / b
    plt.plot(x, y, label=f'{a}x + {b}y = {c}')

# Compute intersection
A = np.array([[2, -1],[1, 1]])
b = np.array([4, 6])
sol = np.linalg.solve(A, b)
plt.scatter(*sol, color='red')
plt.text(sol[0]+0.1, sol[1]-0.3, f'({sol[0]:.2f}, {sol[1]:.2f})')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.title('Intersection of Two Lines')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 5. Real-World Analogy

- **Cost vs. Quantity**:
    - Line 1: \$2 per apple and \$1 per banana totals \$4 → `2x + 1y = 4`.
    - Line 2: \$1 per apple and \$1 per banana totals \$6 → `1x + 1y = 6`.
    Intersection gives the unique item prices.

### 6. Practice Problems

1. Solve and graph:
    
    ```
    3x + 2y = 12
    x  -  y = 1
    ```
    
2. Classify (no graph needed):
    
    System A:
    
    ```
    x + 2y = 5
    2x + 4y = 10
    ```
    
    System B:
    
    ```
    x + 2y = 5
    2x + 4y = 12
    ```
    
3. In Python, modify the code above to plot:
    - Two parallel lines (`slope 1`, different intercepts).
    - Two coincident lines (same slope & intercept).

---

## **System of Equations as Planes (3x3)**

In three variables \(x,y,z\), each linear equation

```markdown
a·x + b·y + c·z = d
```

defines a plane in 3D. Solving three such equations means finding the common intersection of three planes.

### 1. Plane Equation and Components

A single plane has the form:

```markdown
a·x + b·y + c·z = d
```

- **\([a,b,c]\)** is the **normal vector**, showing the plane’s orientation.
- **\(d\)** shifts the plane along its normal.

Every point \((x,y,z)\) on the plane satisfies this dot-product condition.

### 2. Intersection Scenarios of Three Planes

When you have three planes:

```markdown
Plane 1: a₁·x + b₁·y + c₁·z = d₁
Plane 2: a₂·x + b₂·y + c₂·z = d₂
Plane 3: a₃·x + b₃·y + c₃·z = d₃
```

you get four geometric possibilities:

1. **Unique point**
    
    All normals are independent → planes meet at exactly one point.
    
2. **Infinite line**
    
    Two planes intersect in a line, and the third plane contains that line.
    
3. **Infinite plane**
    
    At least two equations are proportional → two planes coincide, the third cuts through them.
    
4. **No intersection**
    
    One plane is parallel to the intersection of the other two → no common solution.
    

### 3. Matrix Formulation & Determinant

Stacking them gives:

```markdown
A · x = b
```

where

```markdown
A = [[a₁, b₁, c₁],
     [a₂, b₂, c₂],
     [a₃, b₃, c₃]],
x = [x, y, z]ᵀ,
b = [d₁, d₂, d₃]ᵀ
```

- If det(A) ≠ 0, normals are independent → **unique solution**.
- If det(A) = 0 but system consistent → **infinitely many** (line or plane).
- If inconsistent → **no solution**.

### 4. Geometric Intuition

- **Normals**: each plane’s normal vector points perpendicular to the surface.
- **Intersection**: solving is like finding the single point/line/plane that all normals “agree” on.
- Visualize with three sheets of paper in space—how they overlap determines solution type.

### 5. ML/DS Connection

In a regression with three features, solving the normal equations:

```markdown
Xᵀ X · w = Xᵀ y
```

gives three equations in weights \(w_1,w_2,w_3\). Whether \(XᵀX\) is invertible (non-singular) or not parallels the plane-intersection story.

### 6. Practice Problems

### 6.1 Pencil-and-Paper

1. Unique point:
    
    ```
    x +  y +  z = 6
    2x -  y +  z = 3
    x + 2y - 2z = -1
    ```
    
2. Intersection line:
    
    ```
    x +  y +  z = 6
    2x + 2y + 2z = 12
    x -  y +  z = 2
    ```
    
3. Coincident plane:
    
    ```
    x + 2y -  z = 4
    2x + 4y - 2z = 8
    x -  y +  z = 1
    ```
    
4. No solution:
    
    ```
    x +  y +  z = 6
    x +  y +  z = 7
    x -  y +  z = 2
    ```
    

### 6.2 Python Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define coefficient matrix A and RHS b
A = np.array([[1, 1, 1],
              [2, -1, 1],
              [1, 2, -2]])
b = np.array([6, 3, -1])

# Solve if possible
try:
    sol = np.linalg.solve(A, b)
    print("Unique solution (x, y, z):", sol)
except np.linalg.LinAlgError:
    print("Matrix is singular or inconsistent.")

# Create grid for plotting planes
xx, yy = np.meshgrid(np.linspace(0, 5, 10), np.linspace(0, 5, 10))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each plane: ax·x + ay·y + az·z = d  →  z = (d - ax·x - ay·y)/az
planes = [(1,1,1,6), (2,-1,1,3), (1,2,-2,-1)]
colors = ['red', 'green', 'blue']

for (a, b, c, d), col in zip(planes, colors):
    zz = (d - a*xx - b*yy) / c
    ax.plot_surface(xx, yy, zz, alpha=0.5, color=col)

# Mark intersection if unique
if 'sol' in locals():
    ax.scatter(*sol, color='black', s=50)
    ax.text(sol[0], sol[1], sol[2], 'Solution')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Intersection of Three Planes')
plt.show()
```

---

## Geometric Notion of Singularity

Singularity in linear systems means the hyperplanes defined by your equations fail to meet in exactly one point. Geometrically, this happens when the hyperplanes are too “aligned” (redundant) or “misaligned” (contradictory).

### 1. Singularity in 2D: Two Lines

Given two lines

```markdown
a₁·x + b₁·y = c₁
a₂·x + b₂·y = c₂
```

they are singular if:

- **Parallel (no intersection)**
slopes equal (`a₁/b₁ = a₂/b₂`) but intercepts differ → no solution
- **Coincident (infinite intersections)**
equations proportional (`a₁/a₂ = b₁/b₂ = c₁/c₂`) → infinite solutions

Non-singular if slopes differ (`a₁/b₁ ≠ a₂/b₂`) → exactly one intersection point.

### 2. Singularity in 3D: Three Planes

For three planes

```markdown
aᵢ·x + bᵢ·y + cᵢ·z = dᵢ   (i = 1,2,3)
```

singularity arises when the normals \([aᵢ,bᵢ,cᵢ]\) are linearly dependent:

- **Unique point** (non-singular)
normals independent → determinant ≠ 0
- **Line of intersections** (infinite solutions)
two normals independent, third lies in their span
- **Coincident plane** (infinite solutions)
two or three normals proportional
- **No common intersection** (no solution)
normals dependent but inconsistent right-sides

### 3. Determinant as Volume and Singularity

The determinant of a square matrix \(A\) measures the volume scaling of the linear map \(x↦A·x\).

For a 3×3 matrix

```markdown
A = [[a₁, b₁, c₁],
     [a₂, b₂, c₂],
     [a₃, b₃, c₃]]
```

the determinant is

```markdown
det(A) = a₁(b₂c₃ – b₃c₂)
       – b₁(a₂c₃ – a₃c₂)
       + c₁(a₂b₃ – a₃b₂)
```

- If `det(A) ≠ 0`, volume ≠ 0 → non-singular → unique intersection.
- If `det(A) = 0`, volume collapses → singular → intersection is a line, plane, or empty.

### 4. Rank, Nullity & Intersection Dimension

The **rank** of \(A\) equals the number of independent normals; **nullity** = number of free variables.

| rank | nullity | Geometric intersection |
| --- | --- | --- |
| n | 0 | unique point |
| n–1 | 1 | infinite line (1D) |
| n–2 | 2 | infinite plane (2D) |
| <n–2 | ≥3 | higher-dimensional affine |

The solution of the homogeneous system \(A·x=0\) is the nullspace; the solution of \(A·x=b\) is an affine shift of that nullspace.

### 5. Real-World ML/DS Connection

- In regression, if two features are collinear (one is a multiple of another), the data matrix \(X\) is singular.
- Singular \(XᵀX\) means no unique least-squares solution.
- Detect with determinant or rank; fix via regularization or feature elimination.

### 6. Practice Problems

1. Identify singularity type and solution dimension for each system:
    
    System A:
    
    ```
    x + 2y  = 5
    2x + 4y = 10
    ```
    
    System B:
    
    ```
    x + 2y  = 5
    2x + 4y = 12
    ```
    
    System C:
    
    ```
    x +  y +  z = 6
    2x + 2y + 2z = 12
    x -  y +  z = 2
    ```
    
    System D:
    
    ```
    x +  y +  z = 6
    x +  y +  z = 7
    x -  y +  z = 2
    ```
    
2. Compute determinant and nullspace in Python:
    
    ```python
    import numpy as np
    
    A = np.array([[1,2,0],
                  [2,4,0],
                  [1,-1,1]])
    print("det(A):", np.linalg.det(A))
    # Nullspace via SVD
    U, S, Vt = np.linalg.svd(A)
    null_mask = (S < 1e-10)
    N = Vt.T[:, null_mask]
    print("nullspace basis:", N)
    ```
    

---

## Singular vs Non-Singular Matrices

A **square matrix** \(A\in\mathbb{R}^{n\times n}\) is called **non-singular** (or **invertible**) if there exists a matrix \(A^{-1}\) such that

```markdown
A·A⁻¹ = A⁻¹·A = Iₙ
```

Otherwise, if no such inverse exists, \(A\) is **singular**.

### 1. Determinant Criterion

For any square matrix \(A\), its **determinant** \(\det(A)\) tells us if it’s singular:

```markdown
det(A) ≠ 0   ⇒   A is non-singular (invertible)
det(A) = 0   ⇒   A is singular (non-invertible)
```

Step-by-step:

1. Compute \(\det(A)\).
2. Check if it’s zero.
3. If non-zero, you can find \(A^{-1}\).
4. If zero, no inverse exists—rows (or columns) are linearly dependent.

### 2. Rank and Linear Independence

The **rank** of \(A\) is the number of linearly independent rows (or columns).

```markdown
rank(A) = n   ⇒   A is non-singular
rank(A) < n   ⇒   A is singular
```

- **Full rank** (\(=n\)) means no redundancy: every row/column adds new information.
- **Rank deficiency** means some rows/columns are redundant or contradictory.

### 3. Geometric Intuition

- A non-singular A maps R^n onto R^n without collapse—volumes scale by det(A).
- A singular \(A\) **collapses** R^n onto a lower-dimensional subspace (plane, line, or point).

### 4. Linear Systems Connection

Solving

```markdown
A · x = b
```

- **Non-singular**: unique solution \(x = A^{-1} b\).
- **Singular**:
    - If consistent: infinitely many solutions (solution space is an affine subspace).
    - If inconsistent: no solutions (equations contradict).

### 5. Real-World ML/DS Examples

- **Collinearity** in regression: if two features are exact multiples, \(XᵀX\) is singular ⇒ no unique least-squares weights.
- **PCA**: singular covariance means zero variance along some direction ⇒ lower-dimensional data.
- **Regularization**: adds \(\lambda I\) to make \(XᵀX + \lambda I\) non-singular.

### 6. Practice Problems

### 6.1 Pencil-and-Paper

1. Determine if each matrix is singular or non-singular. If non-singular, compute its inverse.
    
    Matrix A:
    
    ```markdown
    1  2
    3  5
    ```
    
    Matrix B:
    
    ```markdown
    1  2
    2  4
    ```
    
2. For each, solve \(A x = b\) or state no/inf solutions:
    
    System A:
    
    ```markdown
    1·x + 2·y = 5
    3·x + 5·y = 11
    ```
    
    System B:
    
    ```markdown
    1·x + 2·y = 5
    2·x + 4·y = 12
    ```
    

### 6.2 Python Exercises

```python
import numpy as np

# Define matrices
A = np.array([[1, 2],
              [3, 5]])
B = np.array([[1, 2],
              [2, 4]])

# Determinants
print("det(A):", np.linalg.det(A))
print("det(B):", np.linalg.det(B))

# Inverse of A
A_inv = np.linalg.inv(A)
print("A_inv:", A_inv)

# Solve Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print("Solution for A·x=b:", x)

# Attempt solve on B
try:
    y = np.linalg.solve(B, [5, 12])
    print("Solution for B·y:", y)
except np.linalg.LinAlgError as e:
    print("B is singular:", e)

# Rank check
print("rank(A):", np.linalg.matrix_rank(A))
print("rank(B):", np.linalg.matrix_rank(B))
```

---

## Linear Dependence and Independence

Linear dependence describes when one vector in a set can be written as a combination of the others. Independence means no vector in the set is “redundant.”

### 1. Intuition & Analogy

Think of vectors as directions or arrows.

- **Independent** arrows: none of them lie exactly on top of a line or plane formed by the others. You need all of them to “span” the space they cover.
- **Dependent** arrows: at least one arrow can be obtained by stretching, flipping, and adding the others—making it redundant.

### 2. Formal Definition

A set of vectors (v_1, v_2, … , v_k) in {R}^n is **linearly dependent** if there exist scalars (c_1, c_2, … ,c_k), **not all zero**, such that

```markdown
c₁·v₁ + c₂·v₂ + … + c_k·v_k = 0
```

Otherwise, if the **only** solution to that equation is \(c_1 = c_2 = … = c_k = 0\), the set is **linearly independent**.

### 3. Matrix & Rank Criterion

Stacking the vectors as columns in a matrix \(V\):

```markdown
V = [v₁  v₂  …  v_k]  (n×k)
```

- If **rank**\((V) = k\), columns are **independent**.
- If **rank**\((V) < k\), there is **dependence** (some column is redundant).

### 4. Geometric Interpretation

- In \(\mathbb{R}^2\):
    - Two independent vectors span the plane.
    - If they lie on the same line, they are dependent.
- In \(\mathbb{R}^3\):
    - Three independent vectors span all of 3D space.
    - If they lie in a plane, they are dependent.
    - If two lie on a line, only that line is spanned.

### 5. Connection to Systems & Singularity

- Columns of the **coefficient matrix** \(A\) in \(A·x=b\)
    - If columns **independent**, \(A\) is **non-singular** (invertible).
    - If columns **dependent**, \(A\) is **singular** (no unique solution possible).
- In regression, **collinear features** mean dependence→ unstable estimates.

### 6. Real-World ML/DS Example

Feature vectors representing similar measurements (e.g., inches vs. centimeters) are dependent.

- Dependence inflates variances of weight estimates in linear models.
- Detect with **variance inflation factor (VIF)** or rank checks.
- Fix by dropping or combining redundant features, or using **PCA**.

## 7. Python Exercises

```python
import numpy as np

# Define vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([1, 0, -1])

# Stack as columns
V = np.column_stack([v1, v2, v3])

# Compute rank
rank_V = np.linalg.matrix_rank(V)
print("Rank of V:", rank_V)    # rank < 3 → dependence

# Check if v2 is a multiple of v1
print("Is v2 dependent on v1?", np.allclose(2*v1, v2))
```

To find a non-trivial combination:

```python
# Solve V·c = 0
# Using SVD for nullspace
U, S, Vt = np.linalg.svd(V)
null_mask = (S < 1e-10)
null_space = Vt.T[:, null_mask]
print("Nullspace basis:", null_space)
```

### 8. Practice Problems

1. Determine dependence/independence of:
    
    ```markdown
    v₁ = [1, 0, 2],  v₂ = [2, 1, 3],  v₃ = [3, 1, 5]
    ```
    
2. For each set, compute the rank of the matrix of columns and state if the set is independent:
    
    System A:
    
    ```markdown
    [1, 0;  0, 1]                # 2×2 identity
    ```
    
    System B:
    
    ```markdown
    [1, 2; 2, 4]                 # two identical rows/columns
    ```
    
3. In a dataset with features \(\{X_1, X_2, X_3\}\), you observe \(X_3 = 3X_1 - 2X_2\).
    - Explain why the design matrix is singular.
    - Suggest a remedy for regression modeling.

---

## The Determinant

The **determinant** is a scalar value associated with a square matrix that captures two key ideas:

- **Invertibility**: whether the matrix has an inverse (non-zero determinant) or not (zero determinant).
- **Geometric scaling**: how areas (in 2D) or volumes (in 3D) change under the linear transformation defined by the matrix.

### 1. Determinant of a 2×2 Matrix

For a matrix

```markdown
A = [ [a, b],
      [c, d] ]
```

its determinant is

```markdown
det(A) = a·d – b·c
```

Step-by-step:

- Multiply the **main diagonal** entries: \(a \times d\).
- Multiply the **off diagonal** entries: \(b \times c\).
- Subtract the second product from the first: \(a d - b c\).

If `det(A) ≠ 0`, the transformation scales area by \(|\det(A)|\) and is invertible. If `det(A) = 0`, the two column (or row) vectors are collinear, area collapses to zero, and the matrix is singular.

### 2. Determinant of a 3×3 Matrix

For

```markdown
A = [ [a, b, c],
      [d, e, f],
      [g, h, i] ]
```

the determinant is

```markdown
det(A) = a·(e·i – f·h)
       – b·(d·i – f·g)
       + c·(d·h – e·g)
```

Breakdown:

1. **Take each element** of the first row (\(a,b,c\)).
2. **Compute its minor**: determinant of the 2×2 matrix you get by removing that element’s row and column.
3. **Apply alternating signs** \((+,-,+)\).
4. **Sum** the signed minors.

### 3. General Definition via Cofactor Expansion

For an \(n\times n\) matrix \(A\):

```markdown
det(A) = ∑_{j=1}^n (-1)^{1+j} · A_{1j} · det(M_{1j})
```

- \(A_{1j}\) is the element in row 1, column \(j\).
- \(M_{1j}\) is the \((n-1)×(n-1)\) **minor** obtained by removing row 1 and column \(j\).
- The sign \((-1)^{1+j}\) alternates across the row.

You can expand along any row or column, not just the first.

### 4. Key Properties

- **Multiplicative**: \(\det(AB) = \det(A)\det(B)\).
- **Row swaps** flip sign of the determinant.
- **Scaling** a row by \(k\) multiplies \(\det(A)\) by \(k\).
- **Determinant zero** ⇔ rows (or columns) are linearly dependent ⇔ matrix singular.

### 5. Geometric Interpretation

- **2D (area)**: a 2×2 matrix transforms a unit square into a parallelogram. Its area = \(|\det|\).
- **3D (volume)**: a 3×3 matrix transforms a unit cube into a parallelepiped. Its volume = \(|\det|\).
- **Sign**: positive det preserves orientation; negative det flips it.

### 6. ML/DS Connections

- **Invertibility of \(XᵀX\)** in linear regression:
\(\det(XᵀX)=0\) ⇒ features are collinear ⇒ no unique least-squares solution.
- **Change of variables** in probability (Jacobian determinant) rescales densities.
- **PCA**: covariance matrix singularity indicates redundant dimensions.

### 7. Practice Problems

### 7.1 Hand Calculations

1. Compute the determinant and state if the matrix is singular:
    
    ```
    A = [ [4, 2],
          [3, 1] ]
    ```
    
2. Compute for a 3×3:
    
    ```
    B = [ [1, 2, 3],
          [0, 1, 4],
          [5, 6, 0] ]
    ```
    
3. Expand along the second column of:
    
    ```
    C = [ [2, 0, 1],
          [3, 4, 5],
          [6, 7, 8] ]
    ```
    

### 7.2 Python Exercises

```python
import numpy as np

# Define matrices
A = np.array([[4, 2],
              [3, 1]])
B = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# Compute determinants
print("det(A):", np.linalg.det(A))
print("det(B):", np.linalg.det(B))

# Test singularity
print("A is singular?" , np.isclose(np.linalg.det(A), 0))
print("B is singular?" , np.isclose(np.linalg.det(B), 0))

# In regression: create XᵀX for collinear features
X = np.array([[1, 2],
              [2, 4],
              [3, 6]])  # second column = 2× first column
XtX = X.T.dot(X)
print("det(XᵀX):", np.linalg.det(XtX))
```

- Observe that `A` is non-singular (det≠0), `XᵀX` is singular (det≈0).

---

## Introduction to NumPy Arrays

NumPy arrays lie at the heart of scientific Python. They’re like Python lists on steroids—allowing you to store large, multi-dimensional data efficiently and perform fast, element-wise computations without explicit Python loops.

### 1. Creating Arrays

### From Python Lists

```python
import numpy as np

# 1D array
a = np.array([1, 2, 3, 4])

# 2D array (matrix)
B = np.array([[1, 2, 3],
              [4, 5, 6]])
```

### Common Constructors

```python
# Evenly spaced values
r = np.arange(0, 10, 2)        # [0, 2, 4, 6, 8]

# Linearly spaced
l = np.linspace(0, 1, 5)       # [0.  , 0.25, 0.5 , 0.75, 1.  ]

# All zeros or ones
Z = np.zeros((3, 4))           # 3×4 zero matrix
O = np.ones((2, 2))            # 2×2 ones matrix

# Random numbers
R = np.random.rand(2, 3)       # Uniform [0,1) 2×3 array

```

### 2. Array Attributes

Once you have an array `X`, inspect its metadata:

```python
X.shape    # tuple giving dimensions, e.g. (m, n)
X.ndim     # number of dimensions (axes)
X.size     # total number of elements (product of shape)
X.dtype    # data type of elements
```

- **Shape** `(m, n)` means `m` rows and `n` columns.
- **ndim** tells you if it’s a vector (1), matrix (2), or higher-dimensional.

### 3. Indexing & Slicing

NumPy lets you access subarrays without copy loops:

```python
A = np.arange(1, 17).reshape(4, 4)
# A =
# [[ 1,  2,  3,  4],
#  [ 5,  6,  7,  8],
#  [ 9, 10, 11, 12],
#  [13, 14, 15, 16]]

# Single element
x = A[1, 2]        # second row, third column → 7

# Entire row or column
row2 = A[2, :]     # [ 9, 10, 11, 12]
col1 = A[:, 1]     # [ 2,  6, 10, 14]

# Submatrix (block)
block = A[1:3, 2:4]
# [[ 7,  8],
#  [11, 12]]
```

- Use `start:stop` slices (stop is exclusive).
- Slicing returns a **view**—modifying `block` alters `A`.

### 4. Vectorized Operations

NumPy performs element-wise arithmetic across arrays of the same shape:

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

u + v       # [5, 7, 9]
u * v       # [ 4, 10, 18]
u - 2       # [ -1,  0,  1]
```

No `for` loops needed—under the hood C code runs in parallel.

### 5. Broadcasting

Broadcasting lets you combine arrays of different shapes by “stretching” the smaller one:

```python
# Add a 1D array to each row of a 3×3 matrix
M = np.ones((3, 3))
bias = np.array([0, 1, 2])

Y = M + bias
# Y =
# [[1, 2, 3],
#  [1, 2, 3],
#  [1, 2, 3]]
```

**Rule**: two dimensions are compatible if they are equal or one of them is 1.

### 6. Real-World ML/DS Example

Imagine you have a dataset of 100 samples with 5 features. You store it as a 2D NumPy array:

```python
X = np.random.rand(100, 5)   # shape (100, 5)
y = np.random.rand(100)      # target vector shape (100,)
```

- Compute feature means:
    
    ```python
    means = X.mean(axis=0)     # shape (5,)
    ```
    
- Center data:
    
    ```python
    X_centered = X - means     # broadcasts means across 100 rows
    ```
    

These steps underpin PCA, standardization, and regression workflows.

### 7. Practice Problems

### 7.1 Array Creation & Inspection

1. Create a 3×4 array of even numbers from 2 up to 24.
2. Print its shape, `ndim`, `size`, and `dtype`.

### 7.2 Indexing & Slicing

Given

```python
A = np.arange(1, 26).reshape(5, 5)
```

- Extract the third row and fourth column.
- Extract the 3×3 center block.

### 7.3 Vectorized Computation

1. Let
Compute `sin(x)` and `cos(x)` arrays without loops.
    
    ```python
    x = np.linspace(0, 2*np.pi, 100)
    ```
    
2. Given
Subtract the mean of each column from that column.
    
    ```python
    M = np.random.randint(1, 10, (4,4))
    ```
    

### 7.4 Broadcasting Challenge

Create a 4×3 matrix of zeros and add the vector `[1, 2, 3]` to each row using broadcasting. Verify the result.

---

## Linear Systems as Matrices

A system of linear equations can be compactly represented and manipulated using matrix notation. This allows us to apply powerful linear‐algebra tools—like Gaussian elimination or matrix inversion—in a systematic way.

### 1. From Equations to Matrix Form

Consider the system of \(m\) equations in \(n\) unknowns:

```
a₁₁ x₁ + a₁₂ x₂ + … + a₁ₙ xₙ = b₁
a₂₁ x₁ + a₂₂ x₂ + … + a₂ₙ xₙ = b₂
…
aₘ₁ x₁ + aₘ₂ x₂ + … + aₘₙ xₙ = bₘ
```

We bundle the coefficients \(a_{ij}\), variables \(x_j\), and constants \(b_i\) into three objects:

```markdown
A · x = b
```

where

```markdown
A = [aᵢⱼ]_{m×n},    x = [x₁, x₂, …, xₙ]ᵀ,    b = [b₁, b₂, …, bₘ]ᵀ
```

- **A** is the coefficient matrix (\(m\times n\)).
- **x** is the vector of unknowns (\(n\times1\)).
- **b** is the constants (right‐hand side) vector (\(m\times1\)).

Solving the system means finding **x** such that \(A·x = b\).

### 2. Augmented Matrix

To perform row operations, we form the **augmented matrix** \([A|b]\):

```markdown
[A | b] = [ a₁₁  a₁₂  …  a₁ₙ  |  b₁ ]
          [ a₂₁  a₂₂  …  a₂ₙ  |  b₂ ]
          [  ⋮     ⋮         ⋱   ⋮  ]
          [ aₘ₁  aₘ₂  …  aₘₙ  |  bₘ ]
```

Row operations on \([A|b]\) correspond to adding, swapping, or scaling whole equations. Gaussian elimination uses these operations to reduce \([A|b]\) to **row-echelon form** (and then to reduced form) to read off solutions.

### 3. Step-by-Step Breakdown of \(A·x=b\)

```markdown
A·x = b
```

1. **A**
    - Size \(m\times n\).
    - Row \(i\) holds all coefficients of equation \(i\).
2. **x**
    - Size \(n\times1\).
    - Each entry \(x_j\) is an unknown variable.
3. **b**
    - Size \(m\times1\).
    - Each entry \(b_i\) is the constant term of equation \(i\).
4. **Multiplication**
    - Row-by-column dot product:
    \((A·x)*i = \sum*{j=1}^n a_{ij}·x_j = b_i\).

### 4. Real-World ML/DS Example

In **linear regression** with \(m\) samples and \(n\) features:

```markdown
X · w = y
```

- **X** is \(m×n\) design matrix (each row is a sample’s feature vector).
- **w** is \(n×1\) weight vector to learn.
- **y** is \(m×1\) vector of observed targets.

The **normal equations** \(XᵀX·w = Xᵀy\) are another linear system in matrix form, solved to find the best-fit weights.

### 5. Practice Problems

### 5.1 Pencil-and-Paper

1. Write the matrix form \(A·x=b\) for:
    
    ```
    2x +  y −  z =  5
    x − 2y + 2z = −1
    3x +  y +  z =  4
    ```
    
2. Form the augmented matrix \([A|b]\) for the above system.
3. State whether the system can have a unique solution based on the shape of \(A\) (3×3 here).

### 5.2 Python in Jupyter

```python
import numpy as np

# Coefficients and constants
A = np.array([[ 2,  1, -1],
              [ 1, -2,  2],
              [ 3,  1,  1]])
b = np.array([5, -1, 4])

# Solve using NumPy
x = np.linalg.solve(A, b)
print("Solution x:", x)

# Form augmented matrix
Aug = np.hstack([A, b.reshape(-1, 1)])
print("Augmented [A|b]:\\n", Aug)
```

- Modify one row of **b** to create a contradictory system and observe `LinAlgError`.
- Change **A** to make it singular (e.g., duplicate a row) and check rank with `np.linalg.matrix_rank(A)`.

---