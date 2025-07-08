# Mathematics_c1_w2

## Solving Non-Singular Systems of Linear Equations

Whenever you have a set of linear equations in unknowns, you can write them compactly as

```
A x = b
```

where

- A is an n×n matrix of coefficients
- x is an n-vector of unknowns
- b is an n-vector of constants

A system is **non-singular** when A is invertible (determinant ≠ 0). That means there’s exactly one solution x.

### 1. Explanation

Imagine you have two locks and two keys. Each key affects both locks in some way. Your goal is to find the exact shape of each key so that both locks open perfectly. Here:

- Each lock ↔ one equation
- Each key ↔ one unknown

If the way keys interact with locks is “non-degenerate” (i.e., each lock-key relationship is independent enough), there’s exactly one pair of keys that opens both locks. That independence is what “non-singular” means.

To solve, you systematically eliminate one unknown at a time until you isolate each key’s shape.

### 2. Formulas and the Math Behind Them

### 2.1 The Inverse Solution

When A is invertible, the direct formula is:

```
x = A^{-1} b
```

where A^{-1} is the inverse of A.

- A^{-1}·A = I (the identity matrix)
- Multiplying both sides of A x = b by A^{-1} gives I x = A^{-1} b → x = A^{-1} b

### 2.2 Determinant Test

A matrix A is invertible ↔ det(A) ≠ 0.

```
det(A) ≠ 0
```

- det(A) is a scalar that measures “volume scaling” by A
- If det(A)=0, A flattens n-space into lower dimension (singular)

### 2.3 Gaussian Elimination

Instead of computing A^{-1}, you can apply row operations to reduce `[A | b]` to `[I | x]`.

Basic steps:

1. For each pivot row i from 1 to n:
    - Swap with a row below that has a nonzero entry in column i
    - Scale the pivot row so that the pivot = 1
    - Eliminate the entry in column i of every other row by subtracting a multiple of pivot row
2. At the end, the left side is I and the right side is x.

### 3. Practice Problems and Examples

### 3.1 By Hand

1. Solve the 2×2 system:
    
    3x1 + 2x2 = 8
    
    x1  − x2  = 1
    
2. Solve the 3×3 system:
    
    2x1 +  x2 −  x3 =  1
    
    x1  + 3x2 + 2x3 =  4
    
    3x1 + 2x2 + 5x3 = 10
    

*Solution Sketch for (1):*

- From second equation: x1 = 1 + x2
- Substitute in first: 3(1+x2)+2x2 = 8 → 3 +3x2+2x2 =8 →5x2=5 →x2=1
- Then x1=2

### 3.2 Python with NumPy

```python
import numpy as np

# Define A and b
A = np.array([[3, 2],
              [1,-1]], dtype=float)
b = np.array([8, 1], dtype=float)

# Method 1: direct inverse (not recommended for large n)
x_inv = np.linalg.inv(A).dot(b)

# Method 2: solve function
x = np.linalg.solve(A, b)

print("x via inverse:", x_inv)
print("x via solve:  ", x)
```

### 3.3 Real-World ML Task

- Feature transformation: If you want coefficients for a simple linear regression with two features, you solve (X^T X) w = X^T y. Here X^T X must be non-singular to get a unique weight vector w.

### 4. Visual/Geometric Intuition and ML Relevance

- In 2D, each linear equation is a line. A non-singular 2×2 system has two lines that intersect at exactly one point.
- In 3D, each equation is a plane. Three non-singular planes meet at one point.

Data scientists use this every day: when fitting linear models, checking that (X^T X) is non-singular ensures a unique best-fit. In more advanced settings, if (X^T X) is near-singular, you add regularization to restore stability.

---

## Different Methods to Solve Non-Singular Linear Systems

When A x = b has a unique solution (det(A)≠0), you can reach x in many ways. Below we’ll explore direct solvers, matrix decompositions, determinant-based formulas, and iterative techniques.

### 1. Overview

- **Gaussian Elimination**
    
    Systematically eliminate variables row by row until you isolate each unknown. Works in-place on `[A|b]`.
    
- **Cramer’s Rule**
    
    Express each xᵢ as a ratio of two determinants. Elegant but costly beyond n≈5.
    
- **LU Decomposition**
    
    Factor A into a lower (L) and upper (U) triangular matrix: A=LU. Then solve Ly=b (forward), Ux=y (back).
    
- **Cholesky Decomposition**
    
    For symmetric positive-definite A: A=LLᵀ. Halves work of general LU.
    
- **QR Decomposition**
    
    Factor A=Q R with orthonormal Q, upper R. Solve R x = Qᵀ b (back substitution).
    
- **SVD (Pseudoinverse)**
    
    A=U Σ Vᵀ → x=V Σ⁻¹ Uᵀ b. Always works, even near-singular, but costliest.
    
- **Iterative Methods**
    
    Jacobi, Gauss-Seidel, SOR, Conjugate Gradient. Build x^(k+1) from x^(k) until convergence.
    

### 2. Formulas and Math Behind Them

### 2.1 Gaussian Elimination

```
[A|b] → row-reduce → [I|x]
```

- Swap, scale, and eliminate to turn A into I.

• Back-substitute to read off x.

### 2.2 Cramer’s Rule

For A x = b in {R}ⁿ:

```
x_i = det(A_i) / det(A)
```

- A_i = A with column i replaced by b.

• det(A)≠0 ensures valid division.

### 2.3 LU Decomposition

```
A = L · U
Solve L·y = b   (forward sub)
Solve U·x = y   (back sub)
```

- L has 1s on diagonal, U is upper triangular.

### 2.4 Cholesky Decomposition

Applicable if A = Aᵀ and xᵀAx > 0 ∀x≠0:

```
A = L · Lᵀ
Solve L·y = b
Solve Lᵀ·x = y
```

### 2.5 QR Decomposition

```
A = Q · R
x = R⁻¹ · (Qᵀ · b)
```

- QᵀQ = I, R upper triangular

### 2.6 SVD and Pseudoinverse

```
A = U · Σ · Vᵀ
x = V · Σ⁻¹ · Uᵀ · b
```

- Σ⁻¹ takes reciprocal of nonzero singular values

### 2.7 Iterative Updates

Jacobi (split A=D+R):

```
x^(k+1) = D⁻¹ · (b − R · x^(k))
```

Conjugate Gradient (for SPD) builds x in Krylov subspace to minimize error in A-norm.

### 3. Practice Problems & Python Examples

### 3.1 By Hand Exercise

A = [[4,1,2],
[1,3,0],
[2,0,5]],

b = [9,7,18].

• Factor A=LU ● Solve Ly=b ● Solve Ux=y.

### 3.2 NumPy/SciPy Snippets

```python
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve
from scipy.sparse.linalg import cg

A = np.array([[4,1,2],
              [1,3,0],
              [2,0,5]], float)
b = np.array([9,7,18], float)

# Direct solve (LU under the hood)
x_direct = np.linalg.solve(A, b)

# Explicit LU
P, L, U = lu(A)
y = np.linalg.solve(L, P.dot(b))
x_lu = np.linalg.solve(U, y)

# Conjugate Gradient
x_cg, info = cg(A, b)

print("Direct:", x_direct)
print("LU    :", x_lu)
print("CG    :", x_cg)
```

### 3.3 Real-World Relevance

- LU: accelerating repeated solves with same A but different b (e.g. time-series forecasting).
- QR: stable regression when A has full column rank.
- CG: huge, sparse SPD systems from graph-based problems or finite-element ML.

### 4. Visual & Geometric Intuition

- **Gaussian / LU / Cholesky**
    
    Triangular solve = peeling an onion layer by layer: first one variable, then the next.
    
- **QR**
    
    Project b onto the span of A’s columns via orthonormal Q, then solve triangular system in R.
    
- **Iterative**
    
    Each update moves x^(k) closer to intersection of hyperplanes (equations), like gradually homing in.
    

| Method | Complexity | Best for |
| --- | --- | --- |
| Gaussian Elim | O(n³) | One-off small/medium systems |
| LU / Cholesky | O(n³/3) | Multiple b solves, SPD matrices |
| QR | O(n·m²) | Tall skinny A (m≫n), regression |
| SVD | O(n³) | Ill-conditioned, pseudo-inverse |
| Iterative (CG) | O(k·nnz) | Large sparse SPD, approximate x |

---

## Solving Systems with More Variables Than Equations

When you have fewer equations than unknowns (n variables but m equations with m < n), the system

```
A x = b
```

usually has infinitely many solutions. That’s because m hyperplanes in n-space intersect in an (n – m)-dimensional subspace rather than a single point.

### 1. Explanation

Think of two lines (equations) in three-dimensional space (three variables). Those two lines typically intersect in an entire line, not a single point. That line represents infinitely many solutions.

Key ideas you should know or revisit:

- Row reduction (Gaussian elimination)
- Concept of pivot vs free variables
- Span and null space of a matrix

In underdetermined systems, some variables become free parameters. You express the remaining variables in terms of these free parameters to describe all possible solutions.

### 2. Formulas and the Math Behind Them

### 2.1 Parametric Form via Row Reduction

After reducing `[A|b]` to row-echelon form, identify pivot columns (variables you can solve) and non-pivot columns (free variables). For a 3×5 example, you might get:

```
x1 + 2x3 − x5 = 4
    x2 −  x3 + 2x4 = 1
```

Choose x3, x4, x5 as parameters, say

```
x3 = t,
x4 = s,
x5 = r.
```

Then

```
x1 = 4 − 2t +  r
x2 = 1 +  t − 2s
x3 = t
x4 = s
x5 = r
```

The full solution is a 3-parameter family in {R}^5.

### 2.2 Minimal-Norm Solution via Moore-Penrose Pseudoinverse

To pick one solution (the one with smallest length), use the pseudoinverse A^+:

```
x_min = A^+ · b
```

Compute A^+ with SVD:

```
A = U · Σ · Vᵀ
Σ⁺ = take reciprocal of nonzero singular values
A^+ = V · Σ⁺ · Uᵀ
```

This x_min lies in the row-space of A and has minimal Euclidean norm among all solutions.

### 3. Practice Problems and Python Examples

### 3.1 By Hand Exercise

Solve the underdetermined system:

2x1 +  x2 − x3 + 0x4 = 3

3x2 + 2x3 + x4 = 5

Steps: row-reduce, identify free variables x3 and x4, express x1 and x2 in terms of parameters.

### 3.2 Python with NumPy

```python
import numpy as np

A = np.array([[2,1,-1,0],
              [0,3, 2,1]], float)
b = np.array([3,5], float)

# 1) General solution via row reduction (manually inspect)
# 2) Minimal-norm solution using pseudoinverse
x_min = np.linalg.pinv(A).dot(b)

print("Minimal-norm solution:", x_min)
```

### 3.3 Real-World ML Task

In feature-rich settings (like text with thousands of features), you often have more features than samples. The underdetermined linear system arises in:

- Sparse coding and compressed sensing
- Finding a weight vector that fits training labels exactly
- Initializing parameters before regularization

Using the pseudoinverse gives the smallest-energy solution before you apply L1 or L2 regularization.

### 4. Visual/Geometric Intuition and ML Relevance

- In {R}^3 with two planes, the intersection is a line: that line is all solutions.
- Free variables let you “move” along that line (or higher-dimensional subspace).

Data scientists leverage this by adding regularization (L2 shrinks the null-space component to zero) so that among infinitely many fits, you choose the simplest one. This insight underpins ridge regression, where you solve:

```
(Xᵀ X + λI) w = Xᵀ y
```

Here λ>0 makes the system full-rank and selects a unique w with controlled norm.

---

## Graphical Representation of 3-Variable Linear Systems

When you move from 2 to 3 variables, each linear equation

```
a1·x1 + a2·x2 + a3·x3 = b
```

becomes a plane in 3D. The solution to a system of three such equations is where those three planes intersect.

### 1. Beginner-Friendly Explanation

Each equation ↔ one flat sheet (plane) floating in space.

- If all three planes meet at a single point, you have exactly one solution.
- If they meet along a line (two planes intersect in a line, and the third contains that line), you have infinitely many solutions.
- If they don’t all share any common point or line, there’s no solution.

Imagine three sheets of paper in the air:

- Tilt them so they all cross at one spot → unique.
- Tilt so two share an entire edge and the third also contains that edge → infinite.
- Tilt so one sheet misses the others → none.

### 2. Formulas and the Math Behind Them

### 2.1 General Form of a Plane

```
a·x + b·y + c·z = d
```

- (a, b, c) is the plane’s normal vector (perpendicular direction).
- d sets how far the plane is from the origin along that normal.

### 2.2 System in Matrix Form

```
A = [[a11, a12, a13],
     [a21, a22, a23],
     [a31, a32, a33]]

x = [x1, x2, x3]ᵀ    b = [b1, b2, b3]ᵀ

A·x = b
```

- Each row of A·x=b is one plane.
- Solve by Gaussian elimination, LU, or direct inverse when det(A)≠0.

### 3. Practice Problems & Python Examples

### 3.1 By Hand Examples

1. **Unique Point**
    
    1·x + 1·y + 1·z = 6
    
    2·x − 1·y + 1·z = 3
    
    1·x + 2·y − 1·z = 2
    
2. **Infinite Solutions**
    
    x + y + z = 4
    
    2x + 2y + 2z = 8
    
    x − y + 0·z = 1
    
3. **No Solution**
    
    x + y + z = 3
    
    x + y + z = 4
    
    x − y + z = 1
    

*Sketch for (1):* eliminate to find x=1, y=2, z=3.

### 3.2 Python Visualization with Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt

# Define plane: ax + by + cz = d  as (a,b,c,d)
planes = [
    (1, 1, 1, 6),
    (2,-1, 1, 3),
    (1, 2,-1, 2),
]

# Create grid
xx, yy = np.meshgrid(range(0,5), range(0,5))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for a,b,c,d in planes:
    # solve for z: z = (d - a x - b y)/c
    zz = (d - a*xx - b*yy) / c
    ax.plot_surface(xx, yy, zz, alpha=0.5)

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.show()
```

Run this to see three semi-transparent planes and their single intersection point.

### 4. Visual/Geometric Intuition & ML Relevance

- **Intersections**
    
    • Point → unique solution.
    
    • Line → one degree of freedom → infinite solutions.
    
    • Parallel/misaligned → no solution.
    
- **In ML**
    
    • A classifier’s decision boundary in 3D is a plane.
    
    • In weight space, constraints from training points are half-spaces; their intersection is a feasible region.
    
    • Visualizing 3-variable intersections builds intuition for high-dimensional feature spaces sliced by hyperplanes (e.g., support vector machines).
    

---

## Matrix Row Reduction

Matrix row reduction (Gaussian elimination) is a systematic way to simplify a matrix to solve linear systems, find ranks, and determine null spaces using three basic operations on rows.

### 1. Explanation

Row reduction transforms a matrix step by step into a simpler form by applying elementary row operations:

- Swap two rows.
- Multiply a row by a nonzero scalar.
- Add a scalar multiple of one row to another row.

By doing this, you move toward **row echelon form (REF)**—where each leading nonzero entry (pivot) is to the right of the pivot above—and further to **reduced row echelon form (RREF)**—where every pivot is 1 and is the only nonzero entry in its column.

This process isolates variables in a system A x = b, reveals the rank of A, and identifies free variables for underdetermined systems.

### 2. Formulas and the Math Behind Them

### 2.1 Elementary Row Operations

```
Ri <-> Rj            // Swap row i and row j
Ri <- c · Ri         // Scale row i by nonzero constant c
Ri <- Ri + c·Rj      // Add c times row j to row i
```

- Swapping changes row order without affecting solutions.
- Scaling pivots to 1 makes back-substitution clear.
- Adding multiples eliminates entries below or above pivots.

### 2.2 Algorithm to Reach REF

1. Start at row 1, column 1.
2. Find a nonzero entry in or below the pivot position; swap it into the pivot row.
3. Scale the pivot row so the pivot entry = 1.
4. Eliminate all entries below the pivot by adding suitable multiples of the pivot row.
5. Move to the next row and column; repeat until all rows or columns are processed.

### 2.3 Further to RREF

1. After REF, start from the bottom pivot.
2. Eliminate all entries above each pivot by adding multiples of that pivot row.
3. Ensure every pivot is 1 and its column has zeros elsewhere.

### 3. Practice Problems and Python Examples

### 3.1 By Hand Exercise

Reduce the augmented matrix for the system

2x + y − z = 1

x  − y + 2z = 3

3x + 2y + z = 4

Augmented form:

```
[ 2  1 −1 | 1 ]
[ 1 −1  2 | 3 ]
[ 3  2  1 | 4 ]
```

Steps (outline):

1. Swap row1 and row2 to get pivot 1 in row1.
2. Eliminate below pivot in column 1.
3. Scale row2, eliminate below pivot in column 2.
4. Scale row3, back-substitute to clear above pivots.

### 3.2 Python with NumPy & SymPy

```python
import sympy as sp

# Define symbols and matrix
x, y, z = sp.symbols('x y z')
A = sp.Matrix([[2,1,-1, 1],
               [1,-1,2, 3],
               [3,2, 1, 4]])

# Compute RREF
rref_matrix, pivot_cols = A.rref()

print("RREF:")
print(rref_matrix)
print("Pivot columns:", pivot_cols)
```

This yields the reduced form and identifies which variables are basic vs free.

### 4. Visual/Geometric Intuition & ML Relevance

- Geometrically, row reduction finds where hyperplanes intersect by peeling off one variable at a time.
- REF reveals the dimension of the column space (number of pivots = rank).
- RREF directly gives the general solution or shows inconsistency (a row [0 0 …| c] with c≠0 means no solution).

In ML and data science, row reduction underlies:

- Computing ranks of feature matrices to detect multicollinearity.
- Finding null space for dimensionality reduction or feature selection.
- Solving normal equations (XᵀX) w = Xᵀy via Gaussian elimination when matrix inversion is too expensive.

---

## Row Operation That Preserves Singularity

A matrix is singular exactly when its determinant is zero. Among the three elementary row operations, **adding a multiple of one row to another** keeps the determinant—and hence the singularity property—exactly the same.

### 1. Explanation

When you add a scalar multiple of one row to another row, you’re performing a “shear” transformation on the space: you slide one dimension along another without stretching or flipping the entire shape.

- If the matrix was singular (flattened volume = 0), it stays flattened.
- If it was non-singular (nonzero volume), it keeps the same nonzero volume.

By contrast, swapping rows just flips the sign of the determinant, and scaling a row multiplies the determinant by that scale factor. Those still preserve whether det=0 or not, but they change its magnitude or sign. Only the row-addition move leaves the determinant exactly unchanged.

### 2. Formula and Determinant Behavior

Let A be an n×n matrix, and let Rᵢ ← Rᵢ + c·Rⱼ be the operation of adding c times row j to row i. In matrix form this is E·A, where E is the elementary shear matrix:

```
E = I + c·E_{ij}
```

Here E_{ij} is the matrix with a 1 in the (i,j) position and zeros elsewhere. Since det(E)=1, we have:

```
det(E·A) = det(E)·det(A) = 1 · det(A) = det(A)
```

- det(A)=0 → det(E·A)=0
- det(A)≠0 → det(E·A)≠0

Thus singularity (det=0) is preserved exactly.

### 3. Practice Problems & Python Examples

### 3.1 By Hand Exercise

1. Start with A =
This matrix is singular because row₂ = ½·row₁ → det=0.
    
    ```
    [2  4]
    [1  2]
    ```
    
2. Apply R₂ ← R₂ – 0.5·R₁ to get A′:
Check det(A′)=0·2 – 4·0 = 0.
    
    ```
    [2  4]
    [0  0]
    ```
    

*Verify:* A was singular and A′ remains singular.

### 3.2 Python Verification

```python
import numpy as np

# Original singular matrix
A = np.array([[2,4],
              [1,2]], float)
det_A = np.linalg.det(A)

# Create shear E that does R2 <- R2 - 0.5*R1
E = np.array([[1,   0],
              [-0.5, 1]], float)
A2 = E.dot(A)
det_A2 = np.linalg.det(A2)

print("det(A) :", det_A)
print("det(A2):", det_A2)
```

You’ll see both determinants are effectively zero.

### 4. Geometric Intuition & ML Relevance

- **Geometric View:** R₂ ← R₂ + c·R₁ is a shear: you “slide” one axis along another without stretching volumes. The parallelepiped volume (determinant) stays constant.
- **In ML/Data Science:** When doing Gaussian elimination on the normal equations (XᵀX)w = Xᵀy, shear steps let you simplify the system without accidentally changing whether (XᵀX) is invertible.

Understanding which operations leave singularity intact helps you track when a system gains or loses unique solutions—critical when diagnosing multicollinearity or choosing regularization strategies.

---

## The Rank of a Matrix

The **rank** of a matrix A measures how many independent directions (rows or columns) it spans. Equivalently, it’s the dimension of A’s column space (or row space).

### 1. Explanation

Rank tells you how “full” a matrix is.

- If A maps {R}ⁿ→{R}ᵐ, rank(A)=k means A really only uses a k-dimensional subspace of {R}ⁿ.
- Geometrically, think of A as squashing or rotating space:
    - rank=2 in {R}³ flattens 3-space onto a plane.
    - rank=1 flattens onto a line.
    - rank=3 (full for 3×3) preserves a full 3D volume (if m≥3).

Rank also equals the number of pivots you get when you row-reduce A to RREF.

### 2. Formulas and the Math Behind Them

### 2.1 Definition via Column Space

```
rank(A) = dim( span{ columns of A } )
```

- Count how many columns are linearly independent.

### 2.2 Definition via Row Reduction

```
rank(A) = number of nonzero rows in RREF(A)
```

- Perform row reduction until every pivot row has a leading 1 and zeros elsewhere.
- Each such row contributes one to the rank.

### 2.3 Equivalences

```
rank(A) = rank(Aᵀ)           // row rank = column rank
rank(A) ≤ min( m, n )        // for A of size m×n
```

### 3. Practice Problems and Python Examples

### 3.1 By Hand Exercises

1. Find rank of
    
    ```
    A = [ 1  2  3
          2  4  6
          1  0  1 ]
    ```
    
    Row-reduce and count nonzero rows.
    
2. Determine rank of
    
    ```
    B = [ 1 0 2  1
          0 1 3  2
          1 1 5  3 ]
    ```
    
    Identify pivot positions.
    

### 3.2 Python with NumPy & SymPy

```python
import numpy as np
import sympy as sp

# Example matrix
A = np.array([[1,2,3],
              [2,4,6],
              [1,0,1]], float)

print("NumPy rank:", np.linalg.matrix_rank(A))

# Sympy for RREF and pivot columns
M = sp.Matrix(A)
rref_M, pivots = M.rref()
print("Sympy RREF:\\n", rref_M)
print("Pivot columns:", pivots)
```

### 3.3 Real-World ML Task

- **Multicollinearity detection:** If feature matrix X has rank < number of features, some features are linear combos of others.
- **Low-rank approximation:** SVD truncates small singular values to compress data or denoise images.

### 4. Visual/Geometric Intuition & ML Relevance

- In {R}²:
    - rank=2 → two independent axes → full plane.
    - rank=1 → all points lie on a line.
    - rank=0 → everything maps to the origin.
- In ML:
    - PCA finds principal directions (columns of U in SVD), each direction corresponds to a singular value. The number of significant singular values ≈ rank of data.
    - Understanding rank helps you choose the right regularization or detect when (XᵀX) is invertible for linear regression.

| Rank | Geometric Action | ML Implication |
| --- | --- | --- |
| 0 | All vectors → 0 | Degenerate, no information |
| 1 | Maps onto a line | One dominant direction |
| k | Maps onto k-dimensional plane | Retains k features/variances |
| min(m,n) | Full map onto {R}^min(m,n) | No linear dependencies, invertible (if square) |

---

## The Rank of a Matrix in General

The **rank** of a matrix A measures the number of independent directions its rows or columns span. It tells you the maximum number of linearly independent rows or columns, and it reveals how “full” or “degenerate” the transformation A applies to a vector space is.

### 1. Explanation

The rank answers: “How many independent axes does this matrix use?”

Consider A as a machine that takes vectors in {R}ⁿ and outputs vectors in {R}ᵐ. If rank(A)=k, then no matter how many dimensions n you start with, A only truly uses k dimensions. All outputs lie inside a k-dimensional subspace of {R}ᵐ.

High rank (close to min(m,n)) means A preserves most of the input’s freedom. Low rank means A collapses many directions down to fewer.

### 2. Formulas and the Math Behind Them

### 2.1 Column-Space Definition

```
rank(A) = dim( span{ columns of A } )
```

Count how many columns are linearly independent.

### 2.2 Row-Reduction Definition

```
rank(A) = number of nonzero rows in RREF(A)
```

Perform row reduction to reduced row-echelon form (RREF) and count pivot rows.

### 2.3 Key Properties

rank(A) = rank(Aᵀ)

```
  Row rank equals column rank.
- ```text
rank(A) ≤ min(m, n)
```

For A of size m×n.

rank(A·B) ≤ min(rank(A), rank(B))

```
  Composition cannot increase rank.

---

## 3. Practice Problems & Python Examples

### 3.1 By Hand Exercises

- Determine rank of
```

A = [ 1  2  3 2  4  6 3  5  7 ]

```
Row-reduce to RREF and count nonzero rows.

- For
```

B = [ 1 0 2 1 0 0 0 0 3 0 6 3 ]

```
Identify pivot columns and compute rank.

### 3.2 Python with NumPy

```python
import numpy as np

A = np.array([[1,2,3],
            [2,4,6],
            [3,5,7]], float)

print("Rank of A:", np.linalg.matrix_rank(A))
```

---

## 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    
    ▪ rank=2 in {R}³ flattens 3D into a plane.
    
    ▪ rank=1 flattens into a line.
    
    ▪ rank=0 collapses everything to the origin.
    
- **In Machine Learning**
    
    ▪ Detect multicollinearity: if X’s feature matrix rank < number of features, some features are redundant.
    
    ▪ PCA: selects top k principal directions corresponding to largest singular values—effectively reducing to a rank-k approximation.
    
    ▪ Regularization: adds λI to XᵀX to boost rank and ensure invertibility in linear regression.
    

| rank | m×n action | ML implication |
| --- | --- | --- |
| 0 | all outputs → 0 | degenerate, no signal |
| k | maps to k-dimensional plane | retains k key features |
| min | full mapping | invertible if square, no redundancy |

---

## Row Echelon Form (REF)

Row echelon form is a structured way to simplify a matrix so that solving linear systems, computing rank, and finding pivots becomes straightforward.

### 1. Explanation

Imagine you’re organizing books on shelves of descending height:

- The tallest stack goes on the first shelf, the next-tallest on the second shelf, and so on.
- No book below a given shelf can be taller than the shelf above.

In a matrix in row echelon form:

- Any rows consisting entirely of zeros sit at the bottom.
- In each nonzero row, the first nonzero entry (the **pivot**) appears to the right of the pivot in the row above.
- All entries below a pivot are zero.

This “staircase” or “stepped” structure makes back-substitution easy: you start at the bottom pivot, solve that variable, then move up.

**Prior Knowledge to Revisit**

- Elementary row operations (swap, scale, add rows)
- Concept of pivots and elimination

### 2. 📐 Formulas and the Math Behind REF

### 2.1 Shape of an m×n Matrix in REF

```
[ *   *   *   * ]
[ 0   *   *   * ]
[ 0   0   *   * ]
[ 0   0   0   * ]
[ 0   0   0   0 ]
```

Here `*` denotes any nonzero or zero entry; zeros under each pivot are required.

### 2.2 Formal Conditions

- If row i is entirely zero, then every row j > i is also entirely zero.
- In each nonzero row i, let pivot(i) be the column index of the first nonzero entry. Then
where r is the number of nonzero rows.
    
    ```
    pivot(1) < pivot(2) < … < pivot(r)
    ```
    
- For every pivot at row i, all entries below it (in rows i+1…m) are zero.

### 2.3 Why REF Works

- By using row operations, we zero out entries below each pivot column, creating a triangular shape.
- Once in REF, each equation involves a new variable further to the right, so you can solve from bottom up.

**Real-World ML Example**

In linear regression, solving the normal equations

```
XᵀX · w = Xᵀy
```

can be done by factoring XᵀX into an upper-triangular U (via Cholesky) and then performing a back-substitution—essentially working with a system already in REF.

### 3. 🧠 Practice Problems and Examples

### 3.1 By-Hand Exercise

Reduce the matrix below to REF:

```
[ 2   4  −2  |  6 ]
[ 1   2   1  |  3 ]
[ 3   6   0  |  9 ]
```

Steps:

1. Swap or scale so your first pivot is 1 (e.g., swap row1 and row2).
2. Eliminate entries below pivot in column 1.
3. Move to row2, column2; pivot and eliminate below.
4. Continue until all rows below pivots are zero.

*Solution Sketch:*

After full elimination, you should see zeros below each leading entry, for example:

```
[ 1  2   1  |  3 ]
[ 0  0  −4  | −3 ]
[ 0  0   0  |  0 ]
```

### 3.2 Python with Sympy

```python
import sympy as sp

# Augmented matrix
M = sp.Matrix([[2,4,-2,6],
               [1,2, 1,3],
               [3,6, 0,9]])

# Compute row echelon form
REF = M.echelon_form()   # uses Gaussian elimination
print("REF:\\n", REF)
```

### 3.3 Real-World Data Science Task

You have features X (100×10) and labels y (100×1). To fit a linear model via normal equations:

1. Compute A = XᵀX (10×10), b = Xᵀy (10×1).
2. Row-reduce `[A|b]` to REF.
3. Back-substitute to find the weight vector w.

### 4. 📊 Visual/Geometric Intuition & ML Relevance

- **Geometric View:**
    
    Each row of the augmented matrix is an equation (hyperplane). REF peels away one dimension at a time, like slicing an onion, isolating one variable per “step” from bottom to top.
    
- **ML Connection:**
    - REF reveals the **rank** (number of pivots)—key to spotting multicollinearity in feature matrices.
    - Triangular systems (REF form) let you solve multiple right-hand sides quickly—critical for techniques like **Gaussian process regression** or **Kalman filters**.
    - In **feature selection**, REF shows which columns are redundant (zero columns below pivots).

| Step | Action | ML Implication |
| --- | --- | --- |
| Pivot finding | Identify independent feature | Detects multicollinearity |
| Eliminate below | Zero out lower entries | Builds triangular solve structure |
| Back-substitute | Solve variables from bottom up | Efficient weight computation |

---

## Row Echelon Form in General

Row echelon form (REF) organizes a matrix into a “staircase” shape that makes solving linear systems, computing rank, and identifying pivots straightforward.

### 1. Explanation

In row echelon form:

- All rows consisting entirely of zeros appear at the bottom.
- In each nonzero row, the first nonzero entry (the pivot) is strictly to the right of the pivot in the row above.
- Every entry below a pivot is zero.

Imagine a staircase of blocks stepping down from left to right. Each step is a pivot, and everything below each step is clear space. That pattern lets you solve variables from the bottom up by back-substitution.

### 2. Formal Definition and Properties

### 2.1 General Shape of an m×n Matrix in REF

```
[ *   *   *   *   * ]
[ 0   *   *   *   * ]
[ 0   0   *   *   * ]
[ 0   0   0   *   * ]
[ 0   0   0   0   0 ]
```

Here `*` denotes any entry (zero or nonzero), but the zeros under each pivot are required.

### 2.2 Conditions for REF

- If row i is all zeros, then every row j > i must also be all zeros.
- For each nonzero row i, let pivot(i) be the column index of its first nonzero entry. Then
where r is the number of nonzero rows.
    
    ```
    pivot(1) < pivot(2) < … < pivot(r)
    ```
    
- All entries below each pivot are zero.

### 2.3 Why REF Is Useful

- Reveals the **rank** of the matrix (number of pivots).
- Exposes **pivot columns** (basic variables) and **free columns** (parameters).
- Transforms A x = b into an upper-triangular augmented form for easy back-substitution.

### 3. Practice Problems and Python Examples

### 3.1 By-Hand Exercise

Reduce the augmented matrix for the system

```
2x +  y +  z =  5
x  + 2y −  z =  1
3x + 4y + 2z = 10
```

Augmented form:

```
[ 2  1  1 |  5 ]
[ 1  2 −1 |  1 ]
[ 3  4  2 | 10 ]
```

Steps:

1. Swap or scale to get a 1 in the (1,1) pivot.
2. Eliminate entries below the first pivot.
3. Move to row 2, column 2; pivot and eliminate below.
4. Continue until all entries below pivots are zero.

### 3.2 Python with Sympy

```python
import sympy as sp

# Define augmented matrix
M = sp.Matrix([[2,1,1,5],
               [1,2,-1,1],
               [3,4,2,10]])

# Compute row echelon form
REF = M.echelon_form()
print("REF:\\n", REF)
```

### 4. Visual/Geometric Intuition & ML Relevance

- Geometrically, REF **peels away** one dimension at a time: each pivot isolates a hyperplane intersection along a new axis.
- In machine learning:
    - REF identifies **multicollinearity** by showing which feature columns become redundant (zero rows).
    - Solving normal equations for linear regression often uses an upper-triangular (REF) factorization like Cholesky before back-substitution.
    - REF underpins algorithms that compute **rank**, **null space**, and **column space**, all key to dimensionality reduction and feature selection.

---

## The Gaussian Elimination Algorithm

Gaussian elimination is a systematic procedure to solve linear systems, compute matrix rank, and find inverses by transforming a matrix to an upper-triangular (row echelon) form, then back-substituting to find variables.

### 1. Explanation

Gaussian elimination works like peeling the layers off an onion: you eliminate one variable at a time, moving from the top row down, until the system becomes triangular.

First, you use row operations to zero out all entries below the current pivot (leading entry) in each column—this is the **forward elimination** phase. Once the matrix is in row echelon form, you solve for the bottom variable, then substitute upward—this is **back-substitution**.

Key row operations you’ll use:

- Swapping two rows.
- Multiplying a row by a nonzero constant.
- Adding a multiple of one row to another.

### 2. Formulas and the Math Behind Them

### 2.1 Forward Elimination Steps

For an n×n system A·x = b, form the augmented matrix [A|b]. Then for each pivot index i = 1…n–1:

1. **Pivoting**
    
    ```
    if A[i,i] == 0:
        swap row i with a lower row j where A[j,i] ≠ 0
    ```
    
2. **Eliminate below**
    
    For each row k = i+1…n:
    
    ```
    factor = A[k,i] / A[i,i]
    Row[k] ← Row[k] − factor · Row[i]
    ```
    

After these steps, all entries below the i-th pivot become zero.

### 2.2 Back-Substitution

Once you have an upper-triangular system U·x = c:

For i = n down to 1:

```
x[i] = (c[i] − Σ_{j=i+1 to n} U[i,j]·x[j]) / U[i,i]
```

Solve the last equation first, then move upward.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercise

Solve the system:

```
2x +  y +  z =  9
 x −  y + 2z =  8
3x + 2y −  z =  3
```

Steps outline:

1. Form `[A|b]`.
2. Use row1 as pivot: eliminate x from rows 2 and 3.
3. Use new row2 as pivot: eliminate y from row3.
4. Back-substitute to find z, then y, then x.

*Solution sketch:*

After forward elimination you get

```
[2  1  1 | 9 ]
[0 -1  1 | 4 ]
[0  0 -3 |-2]
```

Back-substitution gives z = 2/3, y = (4 − 1·(2/3))/(-1) = −10/3, x = (9 − 1·y − 1·z)/2.

### 3.2 Python Implementation

```python
import numpy as np

def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    # Forward elimination
    for i in range(n):
        # Pivot if zero
        if abs(A[i,i]) < 1e-12:
            for j in range(i+1, n):
                if abs(A[j,i]) > abs(A[i,i]):
                    A[[i,j]] = A[[j,i]]
                    b[i], b[j] = b[j], b[i]
                    break
        # Eliminate below
        for k in range(i+1, n):
            factor = A[k,i] / A[i,i]
            A[k,i:] -= factor * A[i,i:]
            b[k]    -= factor * b[i]
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - A[i,i+1:] @ x[i+1:]) / A[i,i]
    return x

# Example
A = np.array([[2,1,1],
              [1,-1,2],
              [3,2,-1]], float)
b = np.array([9,8,3], float)
x = gaussian_elimination(A, b)
print("Solution:", x)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    
    Each row is a hyperplane in {R}ⁿ. Forward elimination tilts and stacks these hyperplanes so they intersect in a clear “staircase” pattern, isolating one variable at each step.
    
- **In Machine Learning**
    
    Gaussian elimination underlies solving normal equations in linear regression:
    
    ```
    (XᵀX)·w = Xᵀy
    ```
    
    Forward elimination (or Cholesky, an optimized variant) computes w efficiently.
    

| Phase | Action | ML Implication |
| --- | --- | --- |
| Forward elimination | Zero out below-pivot entries | Builds upper-triangular solve for w |
| Pivoting | Row swaps for numerical stability | Prevents division by tiny numbers (conditioning) |
| Back substitution | Solve variables from bottom up | Computation of model parameters |

---

## Gaussian Elimination:Practical and Programming Examples

Gaussian elimination is the workhorse algorithm for solving linear systems, computing matrix inverses, and determining rank.

### 1. Beginner-Friendly Overview

Gaussian elimination “peels away” variables one at a time:

- **Forward elimination** zeros out entries below each pivot, turning the system into an upper-triangular form.
- **Back-substitution** then solves for unknowns from the bottom row up.

Think of aligning dominoes so each only depends on those above it; once the bottom falls (solved), it triggers the next, and so on.

### 2. Detailed Algorithm Steps

### 2.1 Forward Elimination

For an n×n system A·x = b, form the augmented matrix \[A|b\]. Then for each pivot index i from 1 to n–1:

1. **Partial Pivoting**
    - If A[i,i] = 0 (or very small), swap row i with a lower row j where |A[j,i]| is largest.
2. **Eliminate Below**
For each row k = i+1…n:
After this, all entries in column i below row i become zero.
    
    ```
    factor = A[k,i] / A[i,i]
    Row[k] ← Row[k] − factor · Row[i]
    ```
    

### 2.2 Back-Substitution

Once A is upper-triangular U and b is modified to c:

For i = n down to 1:

```
sum = Σ_{j=i+1 to n} U[i,j] · x[j]
x[i] = (c[i] − sum) / U[i,i]
```

### 3. Hand-Worked Example

Solve

```
2x + 3y −  z =  5
4x +  y + 2z = 11
−2x + y + 2z = −1
```

1. Form augmented matrix:
    
    ```
    [  2   3  −1 |  5 ]
    [  4   1   2 | 11 ]
    [ −2   1   2 | −1 ]
    ```
    
2. Pivot row1 (i=1). Eliminate rows 2–3:
    - Row2 ← Row2 − (4/2)·Row1 → [0, −5, 4 | 1]
    - Row3 ← Row3 − (−2/2)·Row1 → [0, 4, 1 | 4]
    
    ```
    [ 2  3  −1 |  5 ]
    [ 0 −5   4 |  1 ]
    [ 0  4   1 |  4 ]
    ```
    
3. Pivot row2 (i=2). Eliminate row3:
    - factor = 4 / (−5) = −0.8
    - Row3 ← Row3 − (−0.8)·Row2 → [0, 0, 4.2 | 4.8]
    
    ```
    [ 2   3    −1 |  5 ]
    [ 0  −5     4 |  1 ]
    [ 0   0   4.2 | 4.8]
    ```
    
4. Back-substitute:
    - z = 4.8 / 4.2 = 1.1429
    - y = (1 − 4·z) / (−5) = (1 − 4·1.1429)/(−5) = 0.3714
    - x = (5 − 3·y + z) / 2 = (5 − 3·0.3714 + 1.1429)/2 = 2.0571

Solution: x≈2.057, y≈0.371, z≈1.143.

### 4. Programming Examples

### 4.1 From Scratch in Python

```python
import numpy as np

def gaussian_elimination(A, b, tol=1e-12):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]

    # Forward elimination
    for i in range(n-1):
        # Partial pivoting
        max_row = np.argmax(abs(A[i:, i])) + i
        if abs(A[max_row, i]) < tol:
            raise ValueError("Matrix is singular or nearly singular")
        # Swap
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]

        # Eliminate rows below
        for k in range(i+1, n):
            factor = A[k, i] / A[i, i]
            A[k, i:] -= factor * A[i, i:]
            b[k]     -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if abs(A[i, i]) < tol:
            raise ValueError("Zero diagonal element")
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    return x

# Example usage
A = np.array([[2,3,-1],[4,1,2],[-2,1,2]])
b = np.array([5,11,-1])
solution = gaussian_elimination(A, b)
print("Solution:", solution)
```

### 4.2 Using NumPy’s Built-In Solver

```python
import numpy as np

A = np.array([[2,3,-1],[4,1,2],[-2,1,2]], float)
b = np.array([5,11,-1], float)

# Direct solve (LU under the hood)
x = np.linalg.solve(A, b)
print("NumPy solve result:", x)
```

### 4.3 LU Decomposition via SciPy

```python
import numpy as np
from scipy.linalg import lu_factor, lu_solve

A = np.array([[2,3,-1],[4,1,2],[-2,1,2]], float)
b = np.array([5,11,-1], float)

# Compute LU factorization
lu, piv = lu_factor(A)
x = lu_solve((lu, piv), b)
print("SciPy LU solve:", x)
```

### 4.4 Solving Multiple Right-Hand Sides

If you have A·X = B for many b’s, factor once:

```python
# B is shape (n, m)
B = np.random.rand(3, 10)
lu, piv = lu_factor(A)
X = lu_solve((lu, piv), B)
```

### 5. Performance and Numerical Stability

- **Complexity:** O(n³) for forward elimination + back-substitution.
- **Pivoting:** Partial pivoting controls rounding errors by avoiding tiny pivots.
- **Library Benefit:** NumPy/SciPy use highly optimized BLAS/LAPACK for speed and stability.

| Aspect | From Scratch | NumPy/SciPy |
| --- | --- | --- |
| Speed | Moderate | Very fast |
| Stability | Depends on pivoting | High |
| Multiple RHS | Needs reuse | LU factor reuse |

### 6. Data Science Applications

- **Linear Regression (Normal Equations):**
    
    ```
    (XᵀX) w = Xᵀy
    ```
    
    Gaussian elimination (or Cholesky) solves for w.
    
- **Kalman Filters & Control:**
    
    Solve update equations A·x = b in real time.
    
- **Computing Matrix Inverse:**
    
    Augment A with I and row-reduce to [I|A⁻¹].
    
- **Rank & Null Space Analysis:**
    
    REF reveals rank; RREF gives null-space basis for feature selection.
    

### 7. Practice Exercises

1. Implement complete pivoting (swap rows **and** columns) in your from-scratch solver.
2. Solve a 4×4 system by hand, showing every row operation.
3. Use NumPy to time solving A·x = b versus your pure-Python version for n=100, 200, 500.
4. Given X (100×20) and y, compare solutions of (XᵀX)w = Xᵀy via `np.linalg.solve` and `np.linalg.pinv`.

---