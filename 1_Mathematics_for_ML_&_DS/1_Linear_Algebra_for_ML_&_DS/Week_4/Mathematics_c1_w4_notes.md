# Mathematics_c1_w4

## Singularity and Rank of Linear Transformations

### 1. Explanation

A linear transformation from {R}^n to {R}^m is like a machine that takes an n-dimensional arrow (vector) in and produces an m-dimensional arrow out. Two key questions tell you how that machine behaves:

- **Singularity** asks: Does the machine crush some directions down to zero?If it does, the map is singular (it flattens space and loses information).
- **Rank** asks: How many independent directions survive the map?The rank is the number of dimensions in the output where inputs still spread out.

Imagine a sheet of clay (3-dimensional space).

- A non-singular map in {R}^3 might rotate, stretch, and shear the clay but never flatten it completely—volumes stay nonzero.
- A singular map squashes the clay onto a plane or line—some dimensions vanish.

**Prior knowledge to revisit**

- What a matrix does to a vector (matrix–vector multiplication)
- Determinant as a volume scaler (for square matrices)
- Row reduction (REF/RREF) to count pivots

### 2. Formulas and the Math Behind Them

```
# For a square n×n matrix A:
A is singular     if det(A) = 0
A is non-singular if det(A) != 0
```

Explanation: If the determinant of A is zero, at least one volume dimension has collapsed and A cannot be inverted.

```
# Definition of rank and nullity for an m×n matrix A:
rank(A)   = number of linearly independent columns of A
nullity(A)= n − rank(A)

# Rank-Nullity Theorem:
rank(A) + nullity(A) = n
```

Explanation:

The rank counts how many input directions survive in the output.

The nullity counts how many input directions get sent to zero.

Together they add up to n, the total number of input directions.

```
# Computing a 2×2 determinant example:
A = [[a, b],
     [c, d]]
det(A) = a*d - b*c
```

Explanation:

For a simple 2×2 matrix, det = a times d minus b times c. If that equals zero, the two rows/columns are linearly dependent.

**Real-World ML Example**

In linear regression, X is an (m×n) feature matrix:

- If rank(X) < n, some features are exact linear combinations of others (multicollinearity).
- XᵀX becomes singular and you can’t solve the normal equations uniquely unless you add regularization.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Determine singularity, rank, and nullity of
    
    ```
    A = [[1, 2, 3],
         [2, 4, 6],
         [1, 1, 1]]
    ```
    
    *Hint:* Row-reduce or notice row₂ = 2·row₁.
    
2. For
    
    ```
    B = [[1, 0],
         [0, 0]]
    ```
    
    compute det(B), rank(B), nullity(B). What does the transformation do to vectors in {R}^2?
    
3. Prove that a triangular n×n matrix with all nonzero diagonal entries is non-singular.

### 3.2 Python with NumPy

```python
import numpy as np

# Example matrices
A = np.array([[1,2,3],
              [2,4,6],
              [1,1,1]], float)
B = np.array([[1,0],
              [0,0]], float)

# Compute determinant (only square matrices)
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)

# Compute rank and nullity
rank_A = np.linalg.matrix_rank(A)
rank_B = np.linalg.matrix_rank(B)
nullity_A = A.shape[1] - rank_A
nullity_B = B.shape[1] - rank_B

print("A: det =", det_A, " rank =", rank_A, " nullity =", nullity_A)
print("B: det =", det_B, " rank =", rank_B, " nullity =", nullity_B)
```

### 4. Visual/Geometric Intuition and ML Relevance

- **Geometric View**
    - A non-singular map in {R}^3 sends a cube to a skewed parallelepiped of nonzero volume.
    - A singular map flattens that cube onto a plane (rank=2) or line (rank=1) or point (rank=0).
- **ML Applications**
    - **Feature Engineering**: Checking rank(X) tells you if any features are redundant.
    - **Regularization**: Adding λI to XᵀX ensures it becomes full-rank so you can compute a stable inverse.
    - **PCA**: Rank of the data covariance matrix equals the number of nonzero principal components.

| Property | Condition | ML Impact |
| --- | --- | --- |
| Singular | det(A)=0 or rank<n | Infinite or no solutions in regression |
| Non-singular (full) | det(A)!=0 or rank=n | Unique invertible transformation |
| Rank-deficient | rank(A)=k<n | Data lies in a k-dimensional subspace |

---

## Determinant as an Area

### 1. Explanation

When you take two vectors in the plane, think of them as the edges of a parallelogram.

The determinant of the 2×2 matrix built from those vectors tells you the signed area of that parallelogram.

- If the parallelogram has area 5, det = 5 or det = –5 (the sign shows orientation).
- A positive det means the two vectors go around in a counterclockwise order; a negative det means clockwise.
- If det = 0, the two vectors lie on the same line and the parallelogram has zero area.

**Prior knowledge to revisit**

- How to form a 2×2 matrix from two vectors
- Basic vector addition and scalar multiplication

### 2. Formulas and the Math Behind Them

```
# Given two vectors in R^2:
u = [u1, u2]
v = [v1, v2]

# Form the matrix A whose columns are u and v:
A = [ u1  v1
      u2  v2 ]

# The determinant of A:
det(A) = u1*v2 - u2*v1

# The area of the parallelogram spanned by u and v:
area = |det(A)|
```

Explanation:

- Multiply the top-left entry u1 by the bottom-right entry v2.
- Multiply the bottom-left entry u2 by the top-right entry v1.
- Subtract the second product from the first to get det(A).
- Take the absolute value to get the actual (non-signed) area.

**Real-World ML Example**

When changing variables in 2D probability density functions, the Jacobian determinant gives the area scaling factor so that total probability remains 1.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Compute the signed area and true area for vectors
    
    ```
    u = [3, 1]
    v = [1, 2]
    ```
    
    *Hint:* det = 3*2 – 1*1.
    
2. For
    
    ```
    u = [2, 0]
    v = [0, 5]
    ```
    
    what shape do u and v form? What is det, and why?
    
3. Show that swapping u and v flips the sign of det(A) but does not change the absolute area.

### 3.2 Python with NumPy

```python
import numpy as np

# Define vectors
u = np.array([3, 1], float)
v = np.array([1, 2], float)

# Build matrix with u, v as columns
A = np.column_stack((u, v))

# Compute signed area (determinant) and true area
signed_area = np.linalg.det(A)
true_area = abs(signed_area)

print("Matrix A:\n", A)
print("Signed area (det):", signed_area)
print("Parallelogram area:", true_area)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    
    Place u and v tail-to-tail at the origin. They form a parallelogram.
    
    The determinant measures how much that parallelogram covers relative to the unit square.
    
- **Area Scaling in Transformations**
    
    Any 2×2 matrix M acts on the plane by deforming shapes. A small square of area 1 becomes a parallelogram of area |det(M)|.
    
- **In Machine Learning**
    - **Jacobian Determinant**: In normalizing flows, you need det(Jacobian) to adjust densities when mapping data through a transformation.
    - **Feature Space Mapping**: 2D feature transformations can shrink or stretch local neighborhoods; det tells you how much the local area changes, affecting density estimation.
    - **PCA and Whitening**: When you apply PCA rotation and scaling, the determinant of the transformation tracks volume change in the feature space; whitening sets det to 1 to preserve volume.

| Concept | Formula | ML Application |
| --- | --- | --- |
| Signed area in R^2 | det = u1*v2 − u2*v1 | Orientation test for 2D data mapping |
| Absolute area |  | det |
| Area scaling factor | det(M) | Ensures probability conservation |

By seeing the determinant as an area, you’ll develop deeper intuition for how linear maps stretch, shear, and rotate data—crucial for density modeling, feature transformations, and geometric understanding in ML.

---

## Determinant of a Product

### 1. Explanation

When you apply one linear transformation after another, the way they stretch or shrink space multiplies.

- If matrix A scales areas (or volumes) by det(A), and B scales by det(B), doing A then B scales by det(A)·det(B).
- The rule is: the determinant of the product AB equals the product of the determinants.

This means you can break complex transformations into simpler parts and multiply their individual “volume‐scaling” effects.

### 2. Formulas and the Math Behind Them

```
# For any two square n×n matrices A and B:
det(A · B) = det(A) * det(B)
```

Explanation:

- det(·) returns a scalar that describes how a matrix scales volume.
- A·B means first apply B, then apply A.
- The combined volume change is det(A)*det(B).

```
# More identities:
det(I_n)          = 1
det(A^k)          = [det(A)]^k     # repeated multiplication
det(A^(-1))       = 1 / det(A)     # if A is invertible
det(B · A · B^-1) = det(A)          # similarity transform
```

Explanation:

- I_n is the n×n identity matrix, which does nothing (volume scale = 1).
- Raising A to a power multiplies its determinant that many times.
- Inverting A flips the scale.
- Conjugating A by B leaves its determinant unchanged.

**Real-World ML Example**

In **normalizing flows**, you compose simple invertible transformations f = f_L∘…∘f_1. To compute the change of density, you sum log-determinants:

```
log |det(df/dx)| = Σ_i log |det(J_i)|
```

where J_i is the Jacobian of f_i.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Let
    
    ```
    A = [[2, 1],
         [0, 3]]
    B = [[1, 2],
         [1, 1]]
    ```
    
    - Compute det(A) and det(B).
    - Form C = A·B, then compute det(C).
    - Verify det(C) = det(A)*det(B).
2. Show that if det(A)=0 then for any square B, det(A·B)=0 and det(B·A)=0.
3. For a 3×3 identity I and any invertible A, check det(A·I)=det(A) and det(I·A)=det(A).

### 3.2 Python with NumPy

```python
import numpy as np

# Define matrices
A = np.array([[2,1],[0,3]], float)
B = np.array([[1,2],[1,1]], float)

# Compute determinants
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)

# Compute product and its determinant
C = A.dot(B)
det_C = np.linalg.det(C)

print("det(A) =", det_A)
print("det(B) =", det_B)
print("det(A)*det(B) =", det_A*det_B)
print("det(A·B) =", det_C)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**In 2D, A turns a unit square into a parallelogram of area |det(A)|.Then B turns that parallelogram into another shape. The final area is the product of each step’s area change.
- **Machine Learning Uses**
    - **Normalizing Flows:** computing log-determinants of Jacobians for density estimation.
    - **Layer Composition:** in invertible neural networks, tracking volume change when stacking layers.
    - **Feature Transformations:** when preprocessing data via multiple linear steps, overall data‐space scaling is product of individual scales.

| Concept | Formula | ML Application |
| --- | --- | --- |
| det(A·B) | det(A)*det(B) | log-det sum in normalizing flows |
| det(A^k) | [det(A)]^k | repeated layer effect |
| det(A^(-1)) | 1/det(A) | reversing transformations |

Understanding this property lets you break down complex volume or probability‐density changes into simpler parts and recombine them by multiplication.

---

## Determinants of Inverses

### 1. Explanation

When a square matrix A stretches or shrinks space, it does so by a factor of det(A) in volume (area in 2D, length in 1D).

Its inverse A_inv must undo that change exactly, so it stretches or shrinks by the reciprocal factor, 1/det(A).

If A doubles every area (det(A)=2), then A_inv halves every area (det(A_inv)=1/2).

If A flips orientation (det(A)<0), A_inv flips back.

Think of pressing clay with a stamp that doubles its area—you then need a stamp that halves that doubled shape to return to the original.

### 2. Formulas and the Math Behind Them

```
# For any invertible n×n matrix A:
det(A · A_inv) = det(I_n) = 1

# Since det(A · A_inv) = det(A) * det(A_inv), we get:
det(A_inv) = 1 / det(A)
```

```
# 2×2 example:
A       = [[a, b],
           [c, d]]
det(A)  = a*d - b*c

A_inv   = (1/det(A)) * [[ d, -b],
                        [-c,  a]]
det(A_inv) = 1 / det(A)
```

Each symbol means:

- det(·) gives the volume‐scaling factor of a matrix.
- A_inv is the inverse of A, satisfying A · A_inv = I_n.
- I_n is the n×n identity matrix, with det(I_n)=1.

### 3. Practice Problems and Examples

### By-Hand Exercises

1. For
    
    ```
    A = [[4,  7],
         [2,  6]]
    ```
    
    a. Compute det(A).
    
    b. Use the 2×2 inverse formula to write A_inv.
    
    c. Compute det(A_inv) and check it equals 1/det(A).
    
2. Given a 3×3 diagonal matrix
    
    ```
    D = [[2, 0, 0],
         [0, 5, 0],
         [0, 0, 3]]
    ```
    
    a. Compute det(D).
    
    b. Write D_inv by inverting each diagonal entry.
    
    c. Compute det(D_inv) without expanding all entries.
    
3. Explain why if det(A)=0 then A has no inverse (so det(A_inv) does not exist).

### Python with NumPy

```python
import numpy as np

# Example matrix
A = np.array([[4, 7],
              [2, 6]], float)

# Compute determinant of A
det_A = np.linalg.det(A)

# Compute inverse of A
A_inv = np.linalg.inv(A)

# Compute determinant of A_inv
det_A_inv = np.linalg.det(A_inv)

print("det(A)       =", det_A)
print("1 / det(A)   =", 1 / det_A)
print("det(A_inv)   =", det_A_inv)
```

This code prints det(A), its reciprocal, and det(A_inv) to confirm they match.

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    
    A transforms a unit shape into one of volume |det(A)|.
    
    A_inv transforms that shape back to the unit shape. Its volume scale is |det(A_inv)| = 1/|det(A)|.
    
- **Machine Learning Uses**
    - **Normalizing Flows**: each layer’s Jacobian has det(J). The log-density update uses −log|det(J)| for inverse flows.
    - **Covariance Whitening**: transforming data by C^(−1/2) makes the covariance identity. The determinant of C^(−1/2) is 1/√det(C), restoring unit volume in feature space.
    - **Conditioning**: knowing det(A) and det(A_inv) helps assess numerical stability when solving linear systems or computing matrix inverses.

By understanding that det(A_inv) = 1/det(A), you gain intuitive control over volume changes and how inverse transformations recover original scales in data and models.

---

## Bases in Linear Algebra

### 1. Explanation

A **basis** is like the set of building-block directions that can create every arrow (vector) in a space.

Imagine you’re in a flat world (a 2D plane). You can pick two non-parallel streets as your “north” and “east” directions. Any location in the city can be reached by going c₁ blocks north and c₂ blocks east. Those two streets form a basis for the plane.

In general, for an n-dimensional space (like feature vectors of length n), a basis is a set of n vectors that:

- **Span** the space (any vector can be built from them).
- Are **linearly independent** (none of them can be made by mixing the others).

Prior knowledge to revisit:

- What it means to add and scale vectors.
- The concept of linear combinations and span.
- Linear independence (only the trivial combination gives zero).

### 2. Formulas and the Math Behind Them

```
# Definition of a basis for an n-dimensional space V:
Basis B = { v1, v2, …, vn }

# Span condition:
Span(B) = { c1*v1 + c2*v2 + … + cn*vn : c1,…,cn in R } = V

# Linear independence condition:
c1*v1 + c2*v2 + … + cn*vn = 0
    implies
c1 = c2 = … = cn = 0

# Coordinate representation of any x in V:
x = v1*c1 + v2*c2 + … + vn*cn
Coordinates of x relative to B: [x]_B = [c1, c2, …, cn]^T
```

Explanation:

- Span(B) means every vector in V is some mix of the basis vectors.
- Linear independence ensures uniqueness: no basis vector is redundant.
- The coordinates [c1,…,cn] tell you exactly how many “blocks” of each basis vector you need.

**Real-World ML Example**

In PCA you find an orthonormal basis {u₁,…,u_k} of top principal directions so that each data point x can be written as

```
x ≈ u1*(u1·x) + … + uk*(uk·x)
```

where the coordinates (uᵢ·x) compress x into k dimensions.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. In {R}^2, check if B = { [1, 1], [2, 3] } is a basis.
    - Compute whether they span the plane (solve c1*[1,1] + c2*[2,3] = [x,y]).
    - Check linear independence by solving c1*[1,1] + c2*[2,3] = [0,0].
2. In {R}^3, find a basis for the plane defined by x + y + z = 0.
    - Express general solution and pick two independent direction vectors.
3. Given basis B = { [1,0,0], [0,2,0], [0,0,3] }, express x = [4,6,9] in B-coordinates.

### 3.2 Python with NumPy

```python
import numpy as np

# Check independence and span for B in R2
v1 = np.array([1,1], float)
v2 = np.array([2,3], float)
M = np.column_stack((v1, v2))

# rank = number of independent columns
rank = np.linalg.matrix_rank(M)
print("Rank of [v1 v2]:", rank, "(=2 means basis in R2)")

# Express x in basis if invertible
x = np.array([5, 7], float)
if rank == 2:
    coords = np.linalg.solve(M, x)
    print("Coordinates of x in basis:", coords)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - Basis vectors are axes: they point along the core directions of your space.
    - Coordinates tell you how far to go along each axis to reach a point.
- **In Machine Learning**
    - **Feature Engineering:** choosing a good basis (feature set) that captures the essential patterns without redundancy.
    - **Dimensionality Reduction:** PCA, LDA, and other techniques find new bases where the first few axes explain most variance or class separation.
    - **Model Building:** many algorithms assume data expressed in an orthonormal basis for simplicity (e.g., whitened inputs for faster convergence in gradient descent).

| Concept | Basis Role | ML Application |
| --- | --- | --- |
| Span | Coverage of entire space | Ability to reconstruct data points |
| Independence | No redundancy among features | Avoid multicollinearity |
| Coordinates | Unique representation of vectors | Compact encoding in PCA embeddings |

Mastering bases will let you switch between coordinate systems, reduce dimensions effectively, and understand how data is represented and manipulated in every linear model you build. Experiment with different bases to see how they change the shape and structure of your data!

---

## Span in Linear Algebra

### 1. Explanation

Span is the set of all arrows you can make by mixing a given collection of arrows.

Imagine you have one street running east; you can walk east or west any distance—that’s the span of that one street (a line). Add a north–south street, and you can reach any corner in the city; those two streets span the whole plane.

In linear algebra, if you have vectors v₁, v₂, …, vk, their span is every possible combination of those vectors.

Prior knowledge to revisit:

- What it means to add vectors
- What scalar multiplication of a vector does
- How linear combinations work

### 2. Formulas and the Math Behind Them

```
# Definition of span for vectors v1, v2, …, vk in R^n:
Span({v1, v2, …, vk})
  = { c1*v1 + c2*v2 + … + ck*vk : c1, c2, …, ck are real numbers }

```

Explanation:

Each cᵢ is a real number (coefficient).

You multiply each vector vᵢ by its coefficient cᵢ, then add them all up.

The result is any vector you can reach by combining those basis directions.

```
# Special cases:
Span({v})          = all points on the line through v and the origin
Span({v1, v2})     = a plane through the origin if v1 and v2 are not collinear
Span({e1, e2, … en}) = R^n if you use the standard basis

```

Plain text explanation:

- One nonzero vector spans a line.
- Two independent vectors in R^3 span a plane, not necessarily the whole 3D space.
- n independent vectors in R^n span the entire R^n space.

**Real-World ML Example**

In feature engineering, if you have k transformed features that span a subspace of the original data space, you know any prediction based on those features lives in that k-dimensional subspace. PCA selects principal components whose span captures most data variance.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Given v1 = [1, 2] and v2 = [3, 6], describe Span({v1, v2}). Is it a line or the plane?
2. In R^3, let u = [1, 0, 0], v = [0, 1, 0]. What is Span({u, v})? Can you reach [2,3,1]? Why or why not?
3. Determine if w = [2, 5, 7] lies in Span({ [1,1,1], [0,1,2], [1,0,1] }). Show your work by solving for coefficients.

### 3.2 Python with NumPy

```python
import numpy as np

# Define basis vectors and target
V = np.column_stack(([1,1,1], [0,1,2], [1,0,1]))   # shape (3,3)
w = np.array([2,5,7], float)

# Solve V * c = w for coefficients c
# If rank(V) == rank([V|w]), w is in the span
augmented = np.column_stack((V, w))
rank_V = np.linalg.matrix_rank(V)
rank_aug = np.linalg.matrix_rank(augmented)

print("rank(V)     =", rank_V)
print("rank([V|w]) =", rank_aug)

if rank_V == rank_aug:
    c = np.linalg.lstsq(V, w, rcond=None)[0]
    print("w is in the span, coefficients c =", c)
else:
    print("w is not in the span of the given vectors.")
```

This checks whether w lies in the span and, if so, finds one combination of the basis vectors that produces w.

### 4. Visual/Geometric Intuition & ML Relevance

Span is all about coverage of space by chosen directions.

- One vector → a line through the origin
- Two independent vectors → a plane through the origin
- Three independent vectors in R^3 → the full space

| Scenario | Span Description | ML Application |
| --- | --- | --- |
| One feature vector | Line of feature scaling | Feature along a single direction |
| Two PCA directions | Plane capturing top variance | Dimensionality reduction to 2D |
| Full feature set | Entire input space | Model can represent any pattern in the data |

Data scientists use span to:

- Verify that feature transformations still cover the space of interest
- Ensure new features add independent information (expand span)
- Reduce dimension by choosing a smaller basis whose span retains most variance (PCA, autoencoders)

---

## Linear Span and Other Types of Span

### 1. Explanation

A **linear span** of a set of vectors is all the arrows you can make by freely adding and scaling those vectors.

Other “span” concepts change the rules on how you mix vectors:

- An **affine span** lets you mix vectors but forces the mix‐coefficients to add to 1—so you shift and combine without stretching the origin.
- A **conical span** (or positive span) only allows nonnegative scalings—think of shining a light from the origin through each vector.
- A **convex span** (convex hull) mixes vectors with nonnegative weights that sum to 1, so you get every point inside the shape formed by them.

**Prior knowledge to revisit**

- Vector addition and scalar multiplication
- Linear combinations and linear span
- How coefficients control combinations

### 2. Formulas and the Math Behind Them

```
# Linear span of v1…vk in R^n:
LinearSpan({v1,…,vk})
  = { c1*v1 + … + ck*vk
      : each ci is any real number }
```

Explanation:

You multiply each vector by any real number (positive, negative, zero) then add them.

```
# Affine span of v1…vk:
AffineSpan({v1,…,vk})
  = { c1*v1 + … + ck*vk
      : sum(ci) = 1, each ci is any real number }
```

Explanation:

Coefficients must sum to one. You move the origin and then mix along lines connecting the points.

```
# Conical span (positive cone) of v1…vk:
ConicalSpan({v1,…,vk})
  = { c1*v1 + … + ck*vk
      : each ci >= 0 }

```

Explanation:

You only allow nonnegative scalings, so you get all rays and sectors emanating from the origin through the vectors.

```
# Convex span (convex hull) of v1…vk:
ConvexSpan({v1,…,vk})
  = { c1*v1 + … + ck*vk
      : each ci >= 0 and sum(ci) = 1 }

```

Explanation:

You mix the vectors like probabilities—no vector can be weighted negatively, and the weights add to one, so you get every point inside the polygon (or polytope).

**Real-World ML Examples**

- PCA subspace is a **linear span** of top principal directions.
- Affine span defines solution sets of linear regression (intercept + slope directions).
- Nonnegative matrix factorization works in the **conical span** of basis parts.
- Convex combinations underlie clustering centers and mixture models (convex hull of cluster centroids).

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Let v1=[1,0], v2=[0,1] in the plane. Describe:
    - LinearSpan({v1,v2})
    - AffineSpan({v1,v2})
    - ConicalSpan({v1,v2})
    - ConvexSpan({v1,v2})
2. In R^2, show that w=[2,2] lies in ConvexSpan({[1,0],[0,3]}) or not. Solve for c1,c2.
3. In R^3, find AffineSpan of { [1,0,0], [0,1,0], [0,0,1] }. Is it the whole R^3 or a plane?

### 3.2 Python with NumPy

```python
import numpy as np
from itertools import product

v1 = np.array([1,0])
v2 = np.array([0,1])
points = []

# Generate convex combinations of v1 and v2
for c1 in np.linspace(0,1,11):
    c2 = 1-c1
    point = c1*v1 + c2*v2
    points.append(point)

points = np.array(points)
print("ConvexSpan points between v1 and v2:\n", points)

# Test membership in convex span for w=[0.4,0.6]
w = np.array([0.4,0.6])
# Solve c1*v1 + c2*v2 = w
# Here c1 + c2 = 1 and both >=0, so w is in convex span if w coords sum to 1 and each >=0
print("w in convex span?", np.all(w >= 0) and abs(w.sum() - 1) < 1e-6)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Linear span**: an entire line, plane, or higher-dim subspace through the origin.
- **Affine span**: a shifted line/plane that need not pass through the origin.
- **Conical span**: a “pie slice” or infinite wedge starting at the origin.
- **Convex span**: the filled-in polygon (polytope) between the points.

| Span Type | Constraints on ci | Geometric Shape | ML Use Case |
| --- | --- | --- | --- |
| Linear | ci ∈ R | Subspace through 0 | PCA subspaces |
| Affine | ci ∈ R, sum(ci)=1 | Shifted subspace | Regression solution space (with intercept) |
| Conical | ci ≥ 0 | Cone/wedge from 0 | Nonnegative matrix factorization |
| Convex | ci ≥ 0, sum(ci)=1 | Convex polytope | Mixture models, clustering hulls |

Understanding these spans helps you see how feature sets define spaces of predictions, how constraints on coefficients shape solution regions, and how data mixtures and decompositions work in ML pipelines.

---

## Eigenbases

### 1. Explanation

A **basis** is a set of vectors that can build every vector in a space by scaling and adding. An **eigenbasis** is a special basis made up of a matrix’s eigenvectors.

When you apply a matrix A to one of its eigenvectors v, the result is just a scaled version of v:

- That scale factor is the **eigenvalue** λ.
- The eigenvector v never changes direction—only its length.

If you can find n independent eigenvectors in n-dimensional space, those eigenvectors form an eigenbasis. Expressing any vector in that basis makes applying A as simple as stretching each coordinate by its eigenvalue.

**Prior knowledge to revisit**

- Matrix–vector multiplication
- Concept of eigenvalue and eigenvector (A·v = λ·v)
- What it means for vectors to be linearly independent and form a basis

### 2. Formulas and the Math Behind Them

```
# Eigenvalue equation for an n×n matrix A:
A · v = λ · v

# v is a nonzero vector in R^n
# λ is a scalar (eigenvalue)

# If A has n independent eigenvectors v1…vn, form matrix P:
P = [ v1  v2  …  vn ]

# Diagonalization relation:
P^(-1) · A · P = D

# D is a diagonal matrix with D_ii = λ_i
```

Explanation:

- You stack eigenvectors as columns of P.
- Multiplying A by P reorders A’s action into scaling along each eigenvector.
- Conjugating A by P produces a diagonal matrix D whose nonzero entries are the eigenvalues.

**Why it works**

1. A·v_i = λ_i·v_i makes each column of A·P equal to λ_i times the i-th column of P.
2. That is A·P = P·D.
3. Rearranging gives P^(-1)·A·P = D.

**Real-World ML Example**

In PCA, you compute the covariance matrix C and find its eigenbasis. Expressing data in that eigenbasis (principal components) aligns axes with directions of maximum variance, and D tells you how much variance each component carries.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **2×2 Diagonalization**
    
    ```
    A = [[4, 1],
         [2, 3]]
    ```
    
    - Find eigenvalues by solving det(A - λI) = 0.
    - Find eigenvectors v1, v2.
    - Show P^(-1)·A·P = D.
2. **Defective Matrix**
    
    ```
    B = [[2, 1],
         [0, 2]]
    ```
    
    - Compute eigenvalues.
    - Check whether you get two independent eigenvectors.
    - Conclude if B is diagonalizable.
3. **Symmetric Matrix**
    
    ```
    C = [[2, -1, 0],
         [-1, 2, -1],
         [0, -1, 2]]
    ```
    
    - Show C is symmetric.
    - Find its eigenvalues and orthonormal eigenvectors.
    - Confirm the eigenbasis is orthonormal.

### 3.2 Python with NumPy

```python
import numpy as np

# Define matrix A
A = np.array([[4, 1],
              [2, 3]], float)

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

print("Eigenvalues:", eigvals)
print("Eigenvectors (columns):\n", eigvecs)

# Form P and D
P = eigvecs
D = np.diag(eigvals)

# Check diagonalization
reconstructed = np.linalg.inv(P).dot(A).dot(P)
print("P^(-1) A P =\n", reconstructed)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - Eigenvectors are the “invariant directions” of A—they never rotate under A, only stretch.
    - Eigenbasis aligns your coordinate axes so that A acts by simple scaling on each axis.
- **Machine Learning Uses**
    - **PCA (Principal Component Analysis):** diagonalizes the covariance matrix to find principal directions.
    - **Spectral Clustering:** uses eigenbasis of a graph Laplacian to embed nodes for clustering.
    - **Dimensionality Reduction:** picking top k eigenvectors gives a k-dimensional subspace capturing most structure.

| Concept | Formula | ML Application |
| --- | --- | --- |
| Eigenvalue eqn | A·v = λ·v | PCA variance direction |
| Diagonalization | P^(-1)·A·P = D | Simplify repeated transforms |
| Orthonormal basis | P^T·P = I | Numerical stability in PCA |
| Defective case | less than n eigenvectors | Non-diagonalizable transforms |

By mastering eigenbases, you’ll be able to rotate and scale data along its natural axes, simplify complex transformations, and unlock powerful algorithms like PCA, spectral methods, and more.

---

## Eigenvalues and Eigenvectors

### 1. Explanation

A matrix is a machine that transforms vectors. An **eigenvector** is a special vector that only gets stretched or flipped by that machine—never turned to a new direction. The factor by which it stretches or flips is the **eigenvalue**.

Imagine you press clay along some directions. Some lines in the clay get stretched or squashed but don’t rotate. Those lines point along eigenvectors, and the stretch factor is the eigenvalue.

Prior knowledge to revisit:

- How a matrix multiplies a vector
- Linear combinations and direction
- Determinant as a volume or area scaler

### 2. Formulas and the Math Behind Them

```
# Eigenvalue equation for an n×n matrix A:
A · v = lambda · v

# Here:
# A      is an n×n matrix
# v      is a nonzero vector in R^n (the eigenvector)
# lambda is a scalar (the eigenvalue)
```

Explanation:

Multiply A by v and you get the same vector v scaled by lambda.

```
# To find eigenvalues, solve the characteristic equation:
det(A - lambda·I) = 0
```

Explanation:

Subtract lambda from each diagonal entry of A to form (A - lambda·I), take its determinant, and set it to zero. The solutions for lambda are the eigenvalues.

```
# For each eigenvalue lambda_i, find eigenvector v_i by solving:
(A - lambda_i·I) · v_i = 0
```

Explanation:

Plug each lambda_i into the matrix (A - lambda_i·I), then solve the homogeneous system. Nontrivial solutions v_i give the directions that remain invariant under A.

```
# If A has n independent eigenvectors v1…vn, you can diagonalize:
P = [v1  v2  …  vn]
P^(-1) · A · P = D

# D is diagonal with entries D_ii = lambda_i
```

Explanation:

Stack eigenvectors as columns of P. Conjugating A by P turns it into a diagonal matrix D whose entries are the eigenvalues.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **2×2 Matrix**
    
    ```
    A = [[2, 1],
         [1, 2]]
    ```
    
    a. Compute det(A - lambda·I).
    
    b. Solve for eigenvalues lambda.
    
    c. For each lambda, solve (A - lambda·I)·v = 0 to find eigenvectors.
    
2. **Defective Case**
    
    ```
    B = [[3, 1],
         [0, 3]]
    ```
    
    a. Find eigenvalue(s).
    
    b. Check how many independent eigenvectors B has.
    
    c. Explain why B cannot be diagonalized.
    
3. **Symmetric Matrix**
    
    ```
    C = [[4, -1, 0],
         [-1, 4, -1],
         [0, -1, 4]]
    ```
    
    a. Find eigenvalues and show they are real.
    
    b. Find an orthonormal set of eigenvectors.
    

### 3.2 Python with NumPy

```python
import numpy as np

# Example matrix A
A = np.array([[2, 1],
              [1, 2]], float)

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

print("Eigenvalues:", eigvals)
print("Eigenvectors (columns):\n", eigvecs)

# Verify A·v = lambda·v for first eigenpair
v0 = eigvecs[:, 0]
lambda0 = eigvals[0]
print("A·v0:", A.dot(v0))
print("lambda0·v0:", lambda0 * v0)
```

### 4. Visual/Geometric Intuition & ML Relevance

A 2×2 matrix acting on the plane will stretch along one direction and squash along another if it has two real eigenvalues. Those directions are eigenvectors.

In machine learning:

- **PCA** finds an eigenbasis of the covariance matrix. Principal components (eigenvectors) point along directions of maximum variance, and eigenvalues tell you how much variance each direction captures.
- **Spectral Clustering** uses eigenvectors of a graph Laplacian to embed nodes into a low-dimensional space where clusters separate.
- **Markov Chains** use eigenvalues of the transition matrix to understand long‐term behavior and mixing rates.

| Concept | Formula | ML Application |
| --- | --- | --- |
| Characteristic equation | det(A - lambda·I) = 0 | Finding PCA directions |
| Eigen decomposition | A = P·D·P^(-1) | Diagonalizing transforms |
| Real symmetric case | P^T·A·P = D | Orthonormal PCA components |
| Defective matrix | fewer than n eigenvectors | Indicates non-diagonalizable |

Mastering eigenvalues and eigenvectors gives you the tools to rotate, scale, and reshape data along its natural axes—key for dimensionality reduction, clustering, and understanding linear systems deeply.

---

## Calculating Eigenvalues and Eigenvectors

### 1. Explanation

Finding eigenvalues and eigenvectors tells you the special directions in which a matrix acts by pure scaling.

- An **eigenvalue** λ is a number such that when you apply the matrix A to some nonzero vector v, the result is exactly λ times v.
- The corresponding **eigenvector** v never changes direction under A—only its length changes.

Why we solve it step by step:

1. We look for λ so that A·v – λ·v = 0, which means (A – λ·I)·v = 0.
2. The equation det(A – λ·I) = 0 picks out exactly those λ where the system has a nontrivial solution v.
3. Once λ is known, we solve (A – λ·I)·v = 0 for v to find the direction.

Prior knowledge to revisit:

- Matrix–vector multiplication
- Determinant and why det = 0 signals nontrivial solutions
- Solving systems of linear equations

### 2. Formulas and the Math Behind Them

```
# Let A be an n×n matrix:
A = [ a11  a12  …  a1n
      a21  a22  …  a2n
      ⋮     ⋮        ⋮
      an1  an2  …  ann ]

# Form (A – lambda·I):
A – lambda·I = [ a11-lambda  a12       …  a1n
                 a21         a22-lambda …  a2n
                 ⋮            ⋮          ⋮
                 an1         an2       …  ann-lambda ]

# Characteristic equation:
det(A – lambda·I) = 0

# This is a polynomial in lambda of degree n.
# Solve it to find eigenvalues lambda_1, lambda_2, …, lambda_n.
```

Explanation:

- det(A – lambda·I) expands to a polynomial whose roots are the eigenvalues.
- Each lambda_i solves that polynomial exactly.

```
# For each eigenvalue lambda_i:
(A – lambda_i·I) · v = 0

# This is a homogeneous linear system.
# Solve for v (nonzero) by row reduction or other methods.
# The solutions v form the eigenspace for lambda_i.
```

Explanation:

- You plug in one eigenvalue at a time and solve the system to find the corresponding direction(s).
- The set of all v for one lambda forms a subspace called the eigenspace.

**Real-World ML Example**

PCA computes eigenvalues and eigenvectors of the data covariance matrix. The top k eigenvectors define a basis that captures the most variance. Eigenvalues tell you how much variance each direction carries.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **2×2 Example**
    
    ```
    A = [[4, 2],
         [1, 3]]
    ```
    
    a. Form A – lambda·I and compute det(A – lambda·I).
    
    b. Solve the quadratic to find lambda_1 and lambda_2.
    
    c. For each lambda, solve (A – lambda·I)·v = 0 to find eigenvectors v.
    
2. **3×3 Symmetric Example**
    
    ```
    B = [[2, 1, 0],
         [1, 2, 1],
         [0, 1, 2]]
    ```
    
    a. Write det(B – lambda·I).
    
    b. Find its three real eigenvalues.
    
    c. Solve for eigenvectors and verify they are orthogonal.
    
3. **Defective Matrix**
    
    ```
    C = [[5, 1],
         [0, 5]]
    ```
    
    a. Compute eigenvalues.
    
    b. Show there is only one independent eigenvector.
    
    c. Explain why C cannot be diagonalized.
    

### 3.2 Python with NumPy

```python
import numpy as np

# Define a sample matrix
A = np.array([[4, 2],
              [1, 3]], float)

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

print("Eigenvalues:", eigvals)
print("Eigenvectors (columns):\n", eigvecs)

# Verify A·v = lambda·v for each eigenpair
for i in range(len(eigvals)):
    v = eigvecs[:, i]
    lam = eigvals[i]
    Av = A.dot(v)
    lv = lam * v
    print(f"Check eigenpair {i}: A·v = {Av}, lambda·v = {lv}")
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    
    In 2D, A deforms the unit circle into an ellipse.
    
    The ellipse’s axes align with eigenvectors, and their lengths equal |eigenvalues|.
    
- **Machine Learning Uses**
    - **PCA**: eigenvectors of covariance matrix give principal components; eigenvalues rank their importance.
    - **Spectral Clustering**: eigenvectors of graph Laplacian embed data for clustering.
    - **Markov Chains**: dominant eigenvalue of transition matrix equals 1; its eigenvector gives the stationary distribution.
    - **Stability Analysis**: eigenvalues of a system’s Jacobian determine equilibrium behavior (growing, decaying, or oscillating modes).

| Step | Operation | ML Context |
| --- | --- | --- |
| Characteristic poly | det(A – lambda·I) = 0 | Variance spectrum in PCA |
| Solve for lambda | Find roots of degree-n polynomial | Identify dominant modes |
| Solve for v | Solve (A – lambda·I)·v = 0 | Directions of data spread |
| Assemble P, D | P^(-1)·A·P = D | Simplify repeated transforms |

Mastering the calculation of eigenvalues and eigenvectors equips you to analyze data geometry, design dimensionality‐reduction pipelines, and understand the behavior of complex systems in machine learning and beyond.

---

## On the Number of Eigenvectors

### 1. Explanation

Every eigenvalue λ of an n×n matrix A comes with an entire line (or higher-dimensional space) of eigenvectors—any nonzero scalar multiple of one eigenvector is also an eigenvector.

Key ideas:

- For each λ, the set of all v satisfying A·v = λ·v is called the **eigenspace**.
- The eigenspace is a vector space of dimension **geometric multiplicity** ≥ 1.
- Distinct eigenvalues always have independent eigenspaces, so you can get up to n independent eigenvectors.
- If you find n independent eigenvectors (sum of all geometric multiplicities = n), A is **diagonalizable** and those eigenvectors form a basis.
- When an eigenvalue appears k times in the characteristic polynomial (algebraic multiplicity), its eigenspace dimension can be anywhere from 1 to k. If it’s less than k, A is **defective**.

### 2. Formulas and the Math Behind Them

```
# Characteristic polynomial roots give eigenvalues:
det(A - lambda*I) = 0
# Suppose lambda_i is a root with algebraic multiplicity m_i.

# Geometric multiplicity of lambda_i:
geom_mult(lambda_i)
  = dimension of nullspace(A - lambda_i*I)
  = number of independent solutions v to (A - lambda_i*I)·v = 0

# Always:
1 ≤ geom_mult(lambda_i) ≤ alg_mult(lambda_i)

# Diagonalizability criterion:
sum_over_i [ geom_mult(lambda_i) ] = n   ↔   A is diagonalizable
```

Explanation:

- The characteristic equation has total degree n, so eigenvalues (with algebraic multiplicity) sum to n.
- For each λ_i, you solve (A - λ_i·I)·v = 0 to find its eigenspace dimension.
- If the eigenspace dimensions add up to n, you can stack all those eigenvectors into a matrix P that diagonalizes A.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **Distinct Eigenvalues (2×2)**
    
    ```
    A = [[3, 1],
         [0, 2]]
    ```
    
    - Solve det(A - lambda*I)=0 → lambda=3,2.
    - For each lambda, solve (A - lambda*I)·v=0.
    - Count independent eigenvectors (should be 2).
2. **Defective Case (2×2)**
    
    ```
    B = [[5, 1],
         [0, 5]]
    ```
    
    - Characteristic gives lambda=5 with alg_mult=2.
    - Solve (B - 5*I)·v=0 and show you get only one independent eigenvector.
    - Conclude geom_mult=1 < alg_mult=2, so B is not diagonalizable.
3. **Repeated but Full Geometric (3×3)**
    
    ```
    C = 2*I_3
    ```
    
    - All eigenvalues are 2 with alg_mult=3.
    - Solve (C - 2*I)·v=0 → every vector is in the eigenspace.
    - Geom_mult=3, so C is diagonalizable (trivially, it’s scalar).

### 3.2 Python with NumPy

```python
import numpy as np

def summarize_eigens(A):
    eigvals, eigvecs = np.linalg.eig(A)
    print("Eigenvalues:", eigvals)
    # Count how many independent eigenvectors per value
    unique = np.unique(np.round(eigvals, 6))
    for lam in unique:
        # Solve nullspace dimension
        M = A - lam*np.eye(A.shape[0])
        rank = np.linalg.matrix_rank(M)
        geom = A.shape[0] - rank
        alg = np.sum(np.isclose(eigvals, lam))
        print(f"lambda={lam}: alg_mult={alg}, geom_mult={geom}")

# Test on B
B = np.array([[5, 1],[0,5]], float)
summarize_eigens(B)
```

This prints algebraic vs geometric multiplicities and shows defectiveness.

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**Each eigenspace is a line (dimension 1) or plane (dimension 2), etc.When geom_mult=1 you get a single direction; when >1 you get a whole subspace of invariant directions.
- **In Machine Learning**
    - **PCA:** covariance matrices are symmetric so they have a full set of n orthogonal eigenvectors (no defect).
    - **Spectral Methods:** graph Laplacians yield eigenvectors whose count and multiplicity reveal connected components and clustering structure.
    - **Stability Analysis:** repeated eigenvalues with deficient eigenspaces can signal non‐generic or stiff systems (harder to optimize).

| Case | alg_mult | geom_mult | diagonalizable? | shape of eigenspace |
| --- | --- | --- | --- | --- |
| distinct lambdas | 1 each | 1 each | yes | lines |
| repeated, full space | k | k | yes | k-dim subspace |
| defective | k | d<k | no | d-dim subspace |

Master these counts and you’ll know exactly when you can peel a matrix into independent scaling directions, and when you hit a defective block that forces more advanced tools (Jordan form) or careful numerical work.

---

## Dimensionality Reduction and Projection

### 1. Explanation

Dimensionality reduction is the process of taking high-dimensional data (vectors with many features) and finding a lower-dimensional representation that retains as much important information as possible.

A core method is **projection**: dropping dimensions by “casting” each data point onto a lower-dimensional subspace. Imagine shining a light straight down onto a table—each 3D point above the table casts a 2D shadow on the surface. That shadow is a projection.

Key ideas to revisit:

- What a subspace is (span of basis vectors)
- How to add and scale vectors
- Orthogonal (right-angle) relationships

### 2. Formulas and the Math Behind Them

```
# Given:
#  x      in R^n           # original vector
#  V      is n×k matrix whose columns v1…vk form a basis of a k-dim subspace (k < n)

# If columns of V are orthonormal (V^T · V = I_k), the orthogonal projection of x onto span(V) is:
x_proj = V · (V^T · x)
```

Explanation:

- V^T · x computes k coordinates (dot products with each basis vector).
- Multiplying by V re-builds the vector inside the subspace.
- x_proj lies in the k-dim subspace and is the closest point in that subspace to x.

```
# More general projection (non-orthonormal basis):
# Let G = V^T · V  (a k×k Gram matrix, invertible if columns of V are independent)

x_proj = V · ( G^-1 · (V^T · x) )
```

Explanation:

- V^T · x gives raw coefficients.
- G^-1 · (V^T · x) corrects for basis vectors that aren’t orthonormal.
- V · (…) reconstructs the projection in R^n.

**PCA projection**

```
# Given data matrix X (m×n), centered so each column has mean 0:
C = (1/(m−1)) · X^T · X      # covariance matrix, n×n

# Compute top k eigenvectors U_k (n×k)
# Project any centered point x into k-dim space:
x_reduced = U_k^T · x        # coordinates in principal component space

# Reconstruct approximation in original space:
x_approx = U_k · x_reduced
```

Explanation:

- PCA finds directions (principal components) that capture maximum variance.
- U_k^T · x gives the k most important coordinates.
- U_k · x_reduced maps back to n dimensions with minimal error.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **Projection onto a line in R^2**
    - Let v = [3, 4]. Orthonormalize v: u = v / ||v||.
    - Project x = [5, 2] onto span({v}) using x_proj = u·(u^T·x).
    - Compute u, then x_proj by hand.
2. **Non-orthonormal basis**
    - Basis B = {[1, 1], [1, 2]}. Project x = [3, 4] onto span(B).
        1. Form V = [[1,1],[1,2]].
        2. Compute G = V^T·V and its inverse.
        3. Apply x_proj = V · (G^-1 · (V^T · x)).
3. **PCA projection**
    - Given centered points (in R^2):
        
        ```
        X = [[2,0],
             [0,2],
             [-2,0],
             [0,-2]]
        ```
        
    - Compute covariance C.
    - Find eigenvalues/eigenvectors.
    - Project points onto the top principal component.

### 3.2 Python with NumPy

```python
import numpy as np

# 1. Orthonormal projection in R^2
v = np.array([3, 4], float)
u = v / np.linalg.norm(v)               # normalize
x = np.array([5, 2], float)
x_proj = u * (u.dot(x))                # scalar times u
print("Projection of x onto v:", x_proj)

# 2. Projection onto non-orthonormal basis
V = np.array([[1, 1],
              [1, 2]], float)        # columns are basis vectors
x = np.array([3, 4], float)
G = V.T.dot(V)
coeffs = np.linalg.inv(G).dot(V.T.dot(x))
x_proj2 = V.dot(coeffs)
print("Projection onto span(B):", x_proj2)

# 3. PCA example
X = np.array([[2,0],[0,2],[-2,0],[0,-2]], float)
Xc = X - X.mean(axis=0)
C = (1/(Xc.shape[0]-1)) * Xc.T.dot(Xc)
eigvals, eigvecs = np.linalg.eig(C)
idx = np.argsort(eigvals)[::-1]        # sort descending
U1 = eigvecs[:, idx[:1]]               # top 1 component, shape (2,1)
# Project all points
X_reduced = Xc.dot(U1)                 # m×1
X_approx  = X_reduced.dot(U1.T)        # back to m×2
print("Reduced shape:", X_reduced.shape)
print("Reconstructed X:", X_approx)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - **Projection**: dropping a perpendicular from x onto the subspace gives the closest point.
    - **Dimensionality reduction**: you choose a subspace (line, plane, etc.) that best captures your data’s spread, then project onto it.
- **Machine Learning Uses**
    - **PCA** for compressing features, noise reduction, visualization in 2D/3D.
    - **Linear Discriminant Analysis (LDA)** projects data to maximize class separation.
    - **Autoencoders** learn nonlinear projections via neural networks, but the final bottleneck layer acts like a projection.
    - **Feature selection** can be seen as projecting data onto coordinate axes for interpretability.

| Method | Subspace Basis | Projection Formula | ML Application |
| --- | --- | --- | --- |
| Simple orthonormal line | single u (unit vector) | x_proj = u·(u^T·x) | PCA first component, denoising |
| Non-orthonormal span | columns of V | V·( (V^T·V)^-1·V^T·x ) | Custom feature subspace |
| PCA | top k eigenvectors | U_k^T·x | Dimensionality reduction, visualization |
| LDA | discriminant directions | W^T·x | Classification preprocessing |

By mastering projections and dimensionality reduction, you’ll be equipped to compress data, remove noise, visualize high-dimensional structures, and improve model performance in real-world ML tasks.

---

## Motivating PCA

### 1. Explanation

Imagine you have a dataset with dozens—or even hundreds—of features. Often many of those features are related or redundant. You’d like to:

- **Compress** the data into fewer dimensions without losing important patterns.
- **Remove noise** and redundant information.
- **Visualize** high-dimensional data in 2D or 3D.

Principal Component Analysis (PCA) finds new axes (directions) that:

1. Pass through the center of your data.
2. Are mutually at right angles (orthogonal).
3. Capture the maximum variance (spread) in descending order.

By projecting onto the first few principal components, you keep as much information (variance) as possible in a lower-dimensional space.

### 2. Formulas and the Math Behind PCA

```
# Given data X of shape (m, n):
# m = number of samples, n = number of original features

# 1. Center the data (each column has mean 0):
X_centered = X - mean(X, axis=0)

# 2. Compute the covariance matrix (n×n):
C = (1/(m−1)) * X_centered^T · X_centered

# 3. Eigen decomposition of C:
#    Find eigenvalues λ1 ≥ λ2 ≥ … ≥ λn and corresponding orthonormal eigenvectors u1…un:
C · u_i = λ_i · u_i

# 4. Form projection matrix U_k using top k eigenvectors:
U_k = [u1, u2, …, uk]   # shape (n, k)

# 5. Project any centered x into k-dimensional space:
x_reduced = U_k^T · x_centered

# 6. Reconstruct approximation in original space:
x_approx = U_k · x_reduced + mean(X, axis=0)
```

Explanation:

- Centering removes the offset so PCA axes pass through the data’s centroid.
- Covariance C measures how each pair of features varies together.
- Eigen decomposition finds directions u_i where data variance along u_i equals λ_i.
- Projecting on the first k components captures the largest λ_i variances.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **Small 2D Dataset**
    
    ```
    X = [[2, 0],
         [0, 2],
         [-2, 0],
         [0,-2]]
    ```
    
    a. Center X and compute its covariance matrix C.
    
    b. Compute eigenvalues and eigenvectors of C.
    
    c. Show that projecting onto the first component aligns with the direction of greatest spread.
    
2. **Redundant Features**
    
    ```
    X = [[1, 2, 3],
         [2, 4, 6],
         [3, 6, 9]]
    ```
    
    a. Notice the third column = 3×first.
    
    b. Center and compute C.
    
    c. Find eigenvalues (one should be zero) and interpret.
    
3. **Reconstruction Error**
    
    For the first dataset, project onto k=1 component and compute mean squared reconstruction error:
    
    ```
    error = (1/m) * sum ||x_centered - x_approx||^2
    ```
    

### 3.2 Python with NumPy

```python
import numpy as np

# Example dataset
X = np.array([[2, 0],
              [0, 2],
              [-2, 0],
              [0,-2]], float)

# 1. Center data
mean_X = X.mean(axis=0)
Xc = X - mean_X

# 2. Covariance matrix
C = (1/(Xc.shape[0]-1)) * Xc.T.dot(Xc)

# 3. Eigen decomposition
eigvals, eigvecs = np.linalg.eig(C)
idx = np.argsort(eigvals)[::-1]       # sort descending
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# 4. Project onto first principal component
U1 = eigvecs[:, :1]                   # shape (2,1)
X_reduced = Xc.dot(U1)                # shape (4,1)
X_approx  = X_reduced.dot(U1.T) + mean_X

print("Reduced (1D):\n", X_reduced)
print("Reconstructed X:\n", X_approx)

# 5. Reconstruction error
error = np.mean(np.sum((Xc - (X_approx - mean_X))**2, axis=1))
print("MSE reconstruction error:", error)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - Data points form a cloud. PCA finds the longest direction through that cloud (first component), then the next longest orthogonal direction, and so on.
    - Projecting onto those axes gives the “shadow” of the data onto the most informative dimensions.
- **Machine Learning Uses**
    - **Noise Reduction:** keep components with large eigenvalues, drop small‐variance directions that often capture noise.
    - **Feature Compression:** reduce storage and computation by working in a lower‐dimensional space.
    - **Visualization:** plot data in 2D or 3D principal component space to see clusters or trends.
    - **Preprocessing:** feeding PCA‐reduced features into downstream models can improve training speed and reduce overfitting.

| Step | Operation | ML Benefit |
| --- | --- | --- |
| Centering | subtract mean | aligns data with origin |
| Covariance | measure feature correlations | uncovers joint variability |
| Eigen decomposition | find principal directions | identifies key patterns |
| Projection | reduce to top k components | compresses and denoises |
| Reconstruction | map back to original space | quantifies information loss |

By motivating PCA this way, you see not just the formulas, but why it matters: it’s about finding the simplest axes that reveal your data’s true shape.

---

## Variance and Covariance

### 1. Explanation

**Variance** measures how spread out a single feature (variable) is around its mean.

Imagine exam scores: if all students score 80, variance is zero. If scores range from 50 to 100, variance is high.

**Covariance** measures how two features move together.

- Positive covariance means when one feature goes up, the other tends to go up.
- Negative covariance means one feature goes up when the other goes down.
- Zero covariance means they’re uncorrelated (no linear relationship).

Prior knowledge to revisit:

- How to compute a mean (average)
- What a data sample looks like (rows = examples, columns = features)
- Basic understanding of addition and multiplication of numbers

### 2. Formulas and the Math Behind Them

```
# Variance of a feature X with m samples:
mean_X = (1/m) * sum_{i=1 to m}( X[i] )
var(X)  = (1/m) * sum_{i=1 to m}( (X[i] - mean_X)^2 )
```

Explanation:

1. Compute the average mean_X.
2. Subtract mean_X from each sample to get deviations.
3. Square each deviation and average them to get variance.

```
# Covariance between features X and Y with m samples:
mean_X = (1/m) * sum_{i=1 to m}( X[i] )
mean_Y = (1/m) * sum_{i=1 to m]( Y[i] )
cov(X,Y)
  = (1/m) * sum_{i=1 to m}( (X[i] - mean_X) * (Y[i] - mean_Y) )

```

Explanation:

1. Center X and Y by subtracting their means.
2. Multiply the paired deviations and average to see how they co-vary.

```
# Covariance matrix for data matrix D (m×n):
# Each row of D is a sample, each column a feature.
D_centered = D - mean(D, axis=0)    # subtract column means
Cov = (1/m) * D_centered^T · D_centered  # n×n matrix

```

Explanation:

- Cov[i,j] is cov(feature_i, feature_j).
- The diagonal entries are variances of each feature.

**Real-World ML Example**

- In PCA you diagonalize the covariance matrix to find principal components.
- In feature scaling, you might divide by sqrt(var) (standard deviation) to normalize features.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Compute variance of X = [2, 4, 6, 8]:
    - mean_X = (2+4+6+8)/4
    - var(X) = average of squared deviations.
2. Compute covariance between X = [1,2,3] and Y = [2,4,6]:
    - mean_X, mean_Y
    - cov(X,Y).
3. Given D =
    
    ```
    [[1, 2],
     [3, 4],
     [5, 6]]
    ```
    
    a. Center D by subtracting column means.
    
    b. Compute Cov = (1/3) * D_centered^T · D_centered.
    

### 3.2 Python with NumPy

```python
import numpy as np

# Sample data
X = np.array([2, 4, 6, 8], float)
Y = np.array([1, 3, 5, 7], float)

# 1. Variance
mean_X = X.mean()
var_X = ((X - mean_X)**2).mean()
print("mean_X:", mean_X, " var(X):", var_X)

# 2. Covariance
mean_Y = Y.mean()
cov_XY = ((X - mean_X) * (Y - mean_Y)).mean()
print("mean_Y:", mean_Y, " cov(X,Y):", cov_XY)

# 3. Covariance matrix for multiple features
D = np.array([[1,2],[3,4],[5,6]], float)
D_centered = D - D.mean(axis=0)
Cov = (1 / D.shape[0]) * D_centered.T.dot(D_centered)
print("Covariance matrix:\n", Cov)

# Alternatively, use numpy built-in (normalizes by m-1 by default)
print("np.cov (by default uses m-1):\n", np.cov(D, rowvar=False))
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - Variance of X is the average squared distance of points along the X-axis.
    - Covariance of X and Y describes the tilt of the point cloud’s ellipsoid in the XY-plane.
    - Positive tilt → positive covariance; no tilt → zero covariance.
- **Machine Learning Uses**
    - **Feature Scaling:** divide by standard deviation (sqrt(var)) so features have unit variance.
    - **PCA:** covariance matrix eigenvalues tell you how much variance each principal component captures.
    - **Multicollinearity Detection:** near-zero determinant of covariance means features are linearly dependent.
    - **Gaussian Models:** covariance matrix defines the shape of multivariate Gaussian density.

| Concept | Formula | ML Application |
| --- | --- | --- |
| Variance | var(X) = mean((X-mean_X)^2) | Standardization and normalization |
| Covariance | cov(X,Y)=mean((X-mean_X)*(Y-mean_Y)) | Correlation analysis, PCA |
| Covariance mat | Cov = (1/m)D_c^T · D_c | Basis for PCA, Gaussian modeling |

Understanding variance and covariance is fundamental to measuring feature spread, relationships, and guiding techniques like PCA, normalization, and multivariate modeling.

---

## Covariance Matrix

### 1. Explanation

The covariance matrix is a table of how each feature in your data varies with every other feature.

Imagine you have two features—height and weight—for many people. A single covariance number tells you whether taller people tend to be heavier (positive covariance) or lighter (negative covariance).

When you have n features, you build an n×n matrix where each entry at row i, column j is the covariance between feature i and feature j. The diagonal entries are just the variances of each feature.

This matrix summarizes all pairwise relationships in one place.

### 2. Formulas and the Math Behind It

```
# Given data matrix D of shape (m, n):
#  m = number of samples
#  n = number of features

# Step 1: center each column (feature) by subtracting its mean
D_centered = D - mean(D, axis=0)    # result is (m, n)

# Step 2: compute covariance matrix (n×n)
Cov = (1/(m - 1)) * (D_centered^T · D_centered)
```

Explanation:

- mean(D, axis=0) computes a length-n vector of feature means.
- D_centered subtracts that mean vector from each row of D.
- Multiplying D_centered transpose by D_centered sums the product of paired deviations across all samples.
- Dividing by (m−1) gives the unbiased estimate of covariance.

```
# Elementwise definition:
Cov[i,j] = (1/(m - 1)) * sum_{k=1 to m}( (D[k,i] - mean_i) * (D[k,j] - mean_j) )
```

Explanation:

- Cov[i,j] measures how feature i and feature j move together.
- If Cov[i,j] is large and positive, i and j increase together; if large and negative, one increases as the other decreases.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Compute the covariance matrix for
    
    ```
    D = [[1, 2],
         [3, 4],
         [5, 6]]
    ```
    
    by centering and applying the formula.
    
2. Show that the covariance matrix is symmetric: Cov[i,j] = Cov[j,i].
3. If two features are identical (column 2 equals column 1), what does that do to the covariance matrix?

### 3.2 Python with NumPy

```python
import numpy as np

# Sample data (3 samples, 2 features)
D = np.array([[1, 2],
              [3, 4],
              [5, 6]], float)

# Manual computation
D_centered = D - D.mean(axis=0)
Cov_manual = (1 / (D.shape[0] - 1)) * D_centered.T.dot(D_centered)

# Using numpy built-in (rows are samples, columns are features)
Cov_np = np.cov(D, rowvar=False, bias=False)

print("Manual covariance matrix:\n", Cov_manual)
print("NumPy covariance matrix:\n", Cov_np)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - The covariance matrix describes the shape of the data cloud in feature space.
    - Eigenvectors of Cov point along principal axes of that cloud; eigenvalues give the squared lengths of those axes.
- **Machine Learning Uses**
    - **PCA**: diagonalize Cov to find principal components and reduce dimensions.
    - **Whitening**: transform data so Cov becomes the identity, removing feature correlations.
    - **Gaussian Models**: Cov defines the shape of multivariate normal distributions.
    - **Mahalanobis Distance**: uses the inverse Cov to measure distances accounting for feature variances and correlations.

| Aspect | Role of Covariance Matrix | ML Application |
| --- | --- | --- |
| Variance capture | Diagonal entries show individual feature spread | Feature normalization |
| Correlation measure | Off-diagonals show pairwise relationships | Detect multicollinearity |
| Principal axes | Eigen decomposition reveals natural directions | Dimensionality reduction with PCA |
| Distance metric | Inverse covariance defines Mahalanobis distance | Outlier detection, anomaly scoring |

Understanding the covariance matrix is a key step toward techniques like PCA, whitening, and probabilistic modeling, giving you a full picture of how your features interact.

---

## Principal Component Analysis (PCA) – Overview

### 1. Explanation

PCA is a way to take high-dimensional data (many features) and find a new set of axes that capture the most variation in the data.

Imagine a cloud of points in space. PCA finds the line that runs through the cloud in the direction of greatest spread, then the next line at right angles that captures the next largest spread, and so on.

By projecting data onto the first few of these new axes (principal components), you compress the data, remove noise, and reveal its core structure.

### 2. Formulas and the Math Behind PCA

```
# Let X be an m×n data matrix (m samples, n features).

# 1. Center each feature to mean zero:
X_centered = X - mean(X, axis=0)

# 2. Compute the covariance matrix (n×n):
C = (1/(m - 1)) * X_centered^T · X_centered

# 3. Solve the eigenproblem:
C · U = U · Lambda
# U is n×n of eigenvectors, Lambda is diagonal of eigenvalues

# 4. Select top k eigenvectors:
U_k = [u1, u2, …, uk]   # shape n×k

# 5. Project data into k dimensions:
X_reduced = X_centered · U_k  # result is m×k
```

Each symbol means:

- X is your original data.
- X_centered removes feature means so PCA axes pass through the data centroid.
- C measures how features vary together.
- U contains directions (principal components) sorted by variance (eigenvalues in Lambda).
- X_reduced gives coordinates in the reduced k-dimensional space.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **Small 2D cloud**
    
    ```
    X = [[2,0],[0,2],[-2,0],[0,-2]]
    ```
    
    a. Center X.
    
    b. Compute covariance C.
    
    c. Find eigenvalues and eigenvectors by hand.
    
    d. Project points onto the first component and plot.
    
2. **Three redundant features**
    
    ```
    X = [[1,2,3],[2,4,6],[3,6,9]]
    ```
    
    a. Center and compute C.
    
    b. Show one eigenvalue is zero.
    
    c. Interpret why one direction carries no new information.
    
3. **Reconstruction error**
    
    For the 2D cloud, project onto k=1, reconstruct X_approx, and compute
    
    ```
    error = (1/m) * sum ||x_centered - x_approx||^2
    ```
    

### 3.2 Python with NumPy

```python
import numpy as np

# Example data
X = np.array([[2,0],[0,2],[-2,0],[0,-2]], float)

# 1. Center data
mean_X = X.mean(axis=0)
Xc = X - mean_X

# 2. Covariance matrix
C = (1/(Xc.shape[0]-1)) * Xc.T.dot(Xc)

# 3. Eigen decomposition
eigvals, eigvecs = np.linalg.eig(C)
idx = np.argsort(eigvals)[::-1]
U = eigvecs[:, idx]     # sorted eigenvectors

# 4. Project onto first component
U1 = U[:, :1]            # n×1
X_reduced = Xc.dot(U1)   # m×1
X_approx  = X_reduced.dot(U1.T) + mean_X

print("Eigenvalues:", eigvals[idx])
print("Reduced shape:", X_reduced.shape)
print("Reconstruction error:",
      np.mean(np.sum((Xc - (X_approx - mean_X))**2, axis=1)))
```

### 4. Visual/Geometric Intuition & ML Relevance

- Geometric View
    - The first principal component is the axis through the data with maximum variance.
    - Subsequent components are orthogonal axes capturing the next highest variance.
- ML Applications
    - Noise reduction by discarding low-variance components.
    - Feature compression for faster training and reduced storage.
    - Visualization of high-dimensional data in 2D or 3D.
    - Preprocessing for downstream models to improve performance and prevent overfitting.

| PCA Step | Operation | Benefit in ML |
| --- | --- | --- |
| Centering | subtract mean | aligns data with origin |
| Covariance matrix | Xc^T · Xc | measures feature relationships |
| Eigen decomposition | C·U = U·Lambda | finds key variance directions |
| Projection | Xc · U_k | reduces dimension, keeps important info |
| Reconstruction | U_k·(U_k^T·Xc) + mean | quantifies information loss |

---

## PCA – Why It Works

### 1. Explanation

PCA finds the directions where your data spreads out the most and uses those directions as new axes.

By projecting onto these axes, you capture as much of the original variation as possible in fewer dimensions.

This works because the directions of maximum variance also minimize the average squared distance (reconstruction error) between your data points and their low-dimensional approximations.

### 2. Formulas and the Math Behind It

```
# Variance maximization view (one component):
Given centered data X (m×n), find unit vector u∈R^n that maximizes
  u* = argmax_{||u||=1}  (1/(m−1)) * sum_{i=1..m}( (u^T x_i)^2 )
```

Explanation:

- You look for a direction u of length 1.
- Project each data point x_i onto u via u^T x_i.
- The average of those squared projections is the variance along u.
- Maximizing that picks the axis of greatest spread.

```
# This leads to the eigenproblem:
C · u = λ · u
# where C = (1/(m−1)) * X^T · X is the covariance matrix
```

Explanation:

- The derivative of the variance objective yields C·u = λu.
- Solutions u are eigenvectors of C, with λ measuring the variance captured.

```
# Reconstruction error view (k components):
Find U_k (n×k, U_k^T·U_k = I_k) that minimizes
  || X - (X·U_k)·U_k^T ||_F^2

```

Explanation:

- You build a k-dimensional approximation by projecting X onto U_k and back.
- Minimizing the Frobenius norm of the difference selects U_k as the top k eigenvectors.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **One-Dimensional Variance**
    - Data: x = [2, 0, -2] in R^1 (centered).
    - Find variance along u = ±1. Verify PCA picks u = ±1 with λ equal to that variance.
2. **2D Cloud**
    
    ```
    X = [[2,0],[0,1],[-2,0],[0,-1]]
    ```
    
    - Center X.
    - Compute covariance C.
    - Solve C·u = λu for u and λ.
    - Check that u points along the long axis of the cloud.
3. **Reconstruction Error**
    - For the same X, project onto k=1 and compute
        
        ```
        error = || X_centered - (X_centered·u)·u^T ||_F^2
        ```
        
    - Show this is minimal compared to any other unit u.

### 3.2 Python with NumPy

```python
import numpy as np

# 2D example
X = np.array([[2,0],[0,1],[-2,0],[0,-1]], float)
Xc = X - X.mean(axis=0)

# Covariance
C = (1/(Xc.shape[0]-1)) * Xc.T.dot(Xc)

# Eigen decomposition
eigvals, eigvecs = np.linalg.eig(C)
idx = np.argsort(eigvals)[::-1]
u1, lambda1 = eigvecs[:, idx[0]], eigvals[idx[0]]

# Variance along u1
var_along_u1 = np.mean((Xc.dot(u1))**2)
print("First eigenvalue λ1:", lambda1)
print("Variance along u1:", var_along_u1)

# Reconstruction error for k=1
X_proj = Xc.dot(u1).reshape(-1,1) * u1.reshape(1,-1)
error = np.linalg.norm(Xc - X_proj, 'fro')**2
print("Reconstruction error k=1:", error)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - The top principal axis is the longest direction through the data cloud.
    - Projecting onto it gives the tightest “shadow” with minimal spread lost.
- **ML Applications**
    - **Noise Reduction**: low-variance directions often capture noise; PCA drops them.
    - **Feature Compression**: use first k components for a compact representation.
    - **Visualization**: plot data in the space of first two or three components.

| Viewpoint | Key Formula | Insight |
| --- | --- | --- |
| Variance maximization | max_{ | Eigenvectos capture max spread |
| Error minimization | min | Same U as top eigenvectors |
| Covariance diagonalize | C = U·Λ·U^T | Rotates data to uncorrelated axes |

By seeing PCA as solving these equivalent optimization problems, you’ll understand not just how to compute it, but why it extracts the most informative directions in your data.

---

## PCA – Mathematical Formulation

### 1. Explanation

Principal Component Analysis (PCA) finds a new coordinate system where data variation is maximized along the first axes.

- You start with data in {R}ⁿ and center it so its mean is zero.
- PCA then picks k orthonormal directions (principal components) so that projecting data onto those directions captures the most variance.
- Mathematically, this means you solve an optimization problem whose solution turns out to be the top k eigenvectors of the data covariance matrix.

Prior knowledge to revisit:

- How to center data (subtract the mean)
- Matrix–vector multiplication and inner products
- What an orthonormal set of vectors means

### 2. Formulas and the Math Behind It

```
# Given data matrix X (m×n), m samples and n features
# Center the data:
X_centered = X - mean(X, axis=0)       # shape (m, n)

# Covariance matrix (n×n):
C = (1/(m - 1)) * X_centered^T · X_centered
```

Explanation:

C[i,j] measures how feature i and feature j vary together across samples.

```
# Variance maximization for one component:
# Find unit vector u in R^n that maximizes variance along u
u* = argmax_{||u|| = 1}  u^T · C · u
```

Explanation:

Projecting each sample x_i onto u gives scalar u^T·x_i. Squaring and averaging measures variance along u.

```
# For k components, U_k is n×k with orthonormal columns:
U_k = argmax_{U^T · U = I_k}  trace( U^T · C · U )
```

Explanation:

The trace of U^T·C·U sums variances captured by each column of U. Constraining U^T·U=I_k ensures components are orthonormal.

```
# Equivalently, reconstruction error formulation:
# Minimize the Frobenius norm of the difference between original and projected data
U_k = argmin_{U^T · U = I_k}
      || X_centered - (X_centered · U)·U^T ||_F^2
```

Explanation:

Projecting and back-projecting gives the best k-dimensional approximation in least‐squares sense.

```
# Solution via eigen decomposition:
C · u_i = lambda_i · u_i   for i = 1…n
# Sort eigenvalues lambda_1 ≥ … ≥ lambda_n
# Principal components U_k = [u_1, u_2, …, u_k]
```

Explanation:

Each eigenvalue λ_i measures variance along u_i. Picking top k gives directions of greatest spread.

```
# Equivalent via SVD on X_centered (m×n):
X_centered = U · Σ · V^T
# Then V's first k columns are U_k, and Σ contains singular values
```

Explanation:

SVD on the data directly yields principal directions (V) and the strength of each (Σ).

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **3×2 Dataset**
    
    ```
    X = [[2, 0],
         [0, 2],
         [-2, 0]]
    ```
    
    a. Center X and compute C.
    
    b. Solve det(C - lambda·I) = 0 to find eigenvalues.
    
    c. Find eigenvectors u_1, u_2 and verify they are orthonormal.
    
    d. Project X onto u_1 and reconstruct with U_1·(U_1^T·X_centered).
    
2. **Reconstruction vs Variance**
    
    For the same X, compute
    
    ```
    error(u) = || X_centered - (X_centered·u)·u^T ||_F^2
    ```
    
    for u = [1,0] and u = u_1 from PCA. Show PCA u_1 gives smaller error.
    

### 3.2 Python with NumPy

```python
import numpy as np

# Sample data (4 samples, 3 features)
X = np.array([[2, 0, 1],
              [0, 2, 1],
              [-2, 0, 1],
              [0, -2, 1]], float)

# 1. Center data
mean_X = X.mean(axis=0)
Xc = X - mean_X

# 2. Covariance matrix
C = (1/(Xc.shape[0] - 1)) * Xc.T.dot(Xc)

# 3. Eigen decomposition
eigvals, eigvecs = np.linalg.eig(C)
idx = np.argsort(eigvals)[::-1]     # sort descending
eigvals = eigvals[idx]
U = eigvecs[:, idx]

# 4. Pick top k components
k = 2
U_k = U[:, :k]                      # shape (n, k)

# 5. Project and reconstruct
X_reduced = Xc.dot(U_k)             # m×k
X_approx  = X_reduced.dot(U_k.T) + mean_X

print("Top eigenvalues:", eigvals[:k])
print("Projection shape:", X_reduced.shape)
print("Reconstruction error:",
      np.linalg.norm(X - X_approx, 'fro')**2)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View**
    - Data in {R}^n forms a cloud. PCA rotates axes to align with the longest spread.
    - The first k axes form a k-dimensional subspace that best fits the cloud.
- **Machine Learning Uses**
    - **Noise Filtering:** discard small-variance components to remove noise.
    - **Compression:** store m×k coordinates instead of m×n features.
    - **Visualization:** plot X_reduced in 2D or 3D principal component space.
    - **Preprocessing:** reduce dimension before clustering or classification to speed up training.

| Formulation View | Objective | Solution |
| --- | --- | --- |
| Variance maximization | max_{ | eigenvector with largest lambda |
| Trace maximization | max_{U^T·U=I} trace(U^T·C·U) | top k eigenvectors |
| Error minimization | min | same top k eigenvectors |
| Data SVD | X = U·Σ·V^T | V's columns = principal axes |

Understanding these equivalent formulations shows why PCA finds the best low-dimensional representation: it balances capturing variance and minimizing reconstruction error through eigen decomposition or SVD.

---

## Discrete Dynamical Systems

### 1. Explanation

A discrete dynamical system describes how a state changes in steps (time t = 0, 1, 2, …).

You start with an initial state x₀ and apply a rule f repeatedly to get x₁, x₂, x₃, etc.:

Each step:

xₜ₊₁ = f(xₜ)

Think of a population of rabbits: xₜ is the population in year t. A rule f models births and deaths, so year t+1’s population depends only on year t.

Key ideas:

- **State**: a number or vector that captures the system at one step.
- **Map**: a function f that tells you how the state moves from one step to the next.
- **Orbit**: the sequence x₀, x₁, x₂, … tracing the system’s evolution.

### 2. Formulas and the Math Behind Them

### 2.1 Scalar Map

```
# One-dimensional system:
x_{t+1} = f(x_t)
# Example: logistic map
f(x) = r * x * (1 - x)
```

Explanation:

- xₜ is a single number between 0 and 1.
- r is a growth parameter.
- The term (1 – x) limits the population so it can’t grow unbounded.

### 2.2 Linear System

```
# Multi-dimensional linear system:
x_{t+1} = A · x_t
# A is an n×n matrix, x_t is in {R}^n
```

Explanation:

- Each step multiplies the state by A.
- Eigenvalues of A govern stability: if all |λ_i|<1, the origin is an attractor; if any |λ_i|>1, orbits grow without bound.

### 2.3 Nonlinear Systems and Fixed Points

```
# Fixed point condition:
x* = f(x*)
# Stability: study f′(x*) or Jacobian J_f at x* for multi-dim
```

Explanation:

- A fixed point x* satisfies x* = f(x*).In 1D, if |f′(x*)| < 1, the fixed point is stable (nearby orbits converge); if >1, it's unstable.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. **Logistic Map**
    
    ```
    x_{t+1} = 3.2 * x_t * (1 - x_t)
    ```
    
    a. Compute x₁, x₂, x₃ starting from x₀ = 0.1.
    
    b. Find fixed points: solve x = 3.2 x (1–x).
    
    c. Check stability: compute f′(x*) = 3.2 (1–2x*) at each fixed point.
    
2. **Linear 2D System**
    
    ```
    A = [[0.8, 0.1],
         [0.2, 0.9]]
    ```
    
    a. Find eigenvalues λ₁, λ₂.
    
    b. Determine if orbits converge to the origin.
    
    c. Compute x₂ from x₀ = [1, 0]^T.
    
3. **Nonlinear Iteration**
    
    ```
    f([x, y]) = [0.5 x + y^2, x - 0.5 y]
    ```
    
    a. Find fixed point(s).
    
    b. Compute Jacobian J_f at those points.
    
    c. Evaluate eigenvalues of J_f for stability.
    

### 3.2 Python with NumPy

```python
import numpy as np

# 1. Logistic map iteration
def logistic(x, r=3.2):
    return r * x * (1 - x)

x = 0.1
for t in range(5):
    x = logistic(x)
    print(f"x_{t+1} =", x)

# 2. Linear system iteration
A = np.array([[0.8, 0.1],
              [0.2, 0.9]])
x = np.array([1.0, 0.0])
for t in range(5):
    x = A.dot(x)
    print(f"x_{t+1} =", x)

# 3. Fixed points of logistic map
# Solve x = r x (1 - x) <=> r x^2 - r x + x = 0
coeffs = [3.2, -(3.2+1), 0]  # r, -(r+1), 0 for roots
roots = np.roots(coeffs)
print("Fixed points:", roots)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Phase Plots**
    - In 1D, plot xₜ₊₁ vs xₜ and the line y=x to find intersections (fixed points) and cobweb diagrams to see orbit behavior.
    - In 2D, plot orbits in the plane to see spirals, attractors, or repellers.
- **Machine Learning Uses**
    - **Recurrent Neural Networks (RNNs):** hidden state updates follow x_{t+1} = f(W x_t + b). Stability and attractors control memory and vanishing/exploding gradients.
    - **Markov Chains:** discrete state systems with transition matrix P where x_{t+1} = P x_t. Stationary distribution π solves π = P π (eigenvector with eigenvalue 1).
    - **Iterative Algorithms:** many optimizers (power method, EM, value iteration in reinforcement learning) are discrete dynamical systems; convergence depends on eigenvalues or contraction properties.

| System Type | Update Rule | Key Analysis Tool | ML Connection |
| --- | --- | --- | --- |
| Scalar nonlinear | x_{t+1}=f(x_t) | Fixed-point & f′ | Activation iterations |
| Linear vector | x_{t+1}=A x_t | Eigenvalues of A | Power method, feature iteration |
| Markov chain | x_{t+1}=P x_t | Stationary eigenvector | PageRank, RL transition model |
| RNN hidden state | h_{t+1}=σ(W h_t + U x_t) | Jacobian & stability | Sequence modeling |

By understanding discrete dynamical systems, you’ll analyze stability, design iterative ML algorithms, and interpret the long‐term behavior of recurrent models and Markov processes.

---