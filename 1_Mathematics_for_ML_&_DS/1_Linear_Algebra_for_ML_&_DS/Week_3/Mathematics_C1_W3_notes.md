# Mathematics_c1_w3

## Vectors and Their Properties

A **vector** is an ordered list of numbers representing magnitude and direction in space. In machine learning and data science, vectors model features, data points, gradients, and more.

### 1. Explanation

Think of a vector as an arrow:

- The **length** of the arrow shows how big the vector is.
- The **direction** it points shows which way.

In {R}², a vector v = [3, 2] is an arrow from the origin to the point (3, 2). In higher dimensions, you can’t draw it, but it still represents a direction and magnitude in feature space.

Key ideas to revisit:

- Coordinates: each entry vi is how far you move along axis i.
- Addition: placing two arrows tip-to-tail.
- Scaling: stretching or shrinking an arrow.

### 2. 📐 Formulas and Their Math

### 2.1 Vector Representation

```
v = [v1, v2, …, vn]
u = [u1, u2, …, un]
```

- vi, ui are components along each axis.

### 2.2 Addition and Scalar Multiplication

```
u + v = [u1 + v1, u2 + v2, …, un + vn]
c·v   = [c·v1,   c·v2,   …, c·vn]
```

- Addition combines direction and magnitude.
- Scaling c stretches (c>1) or flips (c<0) or shrinks (0<c<1).

### 2.3 Dot Product (Inner Product)

```
u · v = Σ_{i=1 to n} (ui * vi)
```

- Measures how much u and v point in the same direction.
- If u · v = 0, they’re orthogonal (at right angles).

### 2.4 Norm (Length or Magnitude)

```
||v|| = sqrt(v · v) = sqrt(Σ_{i=1 to n} vi^2)
```

- Distance from the origin to the tip of v.

### 2.5 Cross Product (3D Only)

```
u × v = [u2*v3 − u3*v2,
         u3*v1 − u1*v3,
         u1*v2 − u2*v1]
```

- Result is a vector perpendicular to both u and v.
- Only defined in {R}³.

### 2.6 Properties

- **Commutativity of addition:**`u + v = v + u`
- **Associativity of addition:**`(u + v) + w = u + (v + w)`
- **Distributivity:**`c·(u + v) = c·u + c·v`
- **Scalar associativity:**`(cd)·v = c·(d·v)`
- **Dot product symmetry:**`u · v = v · u`
- **Dot product linearity:**`u · (v + w) = u · v + u · w`

### 3. 🧠 Practice Problems & Python Examples

### 3.1 By-Hand Exercises

1. Given u = [2, –1, 3] and v = [–1, 4, 2], compute:
    - u + v
    - 3·u
    - u · v
    - ||u||
2. In {R}³, find a vector perpendicular to u = [1, 2, 3] and v = [4, 0, –1] using cross product.
3. If u · v = 0, and ||u|| = 3, ||v|| = 4, what is the angle between u and v?

*Solutions Sketch:*

1. u+v=[1,3,5], 3u=[6,–3,9], u·v=2*(–1)+(-1)*4+3*2=−2−4+6=0, ||u||=√(4+1+9)=√14.
2. u×v=[(–1)*(–1)−3*4, 3*4−1*(–1),1*0−2*4] = [1−12,12+1,0−8]=[–11,13,–8].
3. Angle θ: u·v = ||u||·||v||·cosθ → 0 = 3·4·cosθ → cosθ=0 → θ=90°.

### 3.2 Python with NumPy

```python
import numpy as np

u = np.array([2, -1, 3], float)
v = np.array([-1, 4, 2], float)

# Addition and scaling
print("u+v:", u + v)
print("3u :", 3 * u)

# Dot product and norms
dot = u.dot(v)
norm_u = np.linalg.norm(u)
angle = np.arccos(dot / (norm_u * np.linalg.norm(v))) * 180/np.pi

print("u·v   :", dot)
print("||u|| :", norm_u)
print("Angle :", angle, "degrees")

# Cross product
w = np.cross(u, v)
print("u×v   :", w)
```

### 3.3 Real-World ML Tasks

- **Cosine Similarity:**Measures text or embedding similarity via
    
    ```
    cos_sim(u,v) = (u·v) / (||u||·||v||)
    ```
    
- **Gradient Descent Updates:**Parameter vector θ updated by subtracting a scaled gradient:
    
    ```
    θ_new = θ_old − α·∇L(θ_old)
    ```
    
- **Feature Scaling:**Normalize feature vectors to unit length to prevent one feature dominating.

### 4. 📊 Visual/Geometric Intuition & ML Relevance

- In {R}² or {R}³, arrows show direction and length.
- **Dot product** = projection length × length of the other vector.
- **Orthogonality** (u·v=0) means features are uncorrelated.
- **Cross product** gives area and orientation of parallelogram spanned by two vectors.

| Property | Geometric View | ML Application |
| --- | --- | --- |
| Vector addition | Tip-to-tail arrows | Aggregating feature shifts |
| Scalar multiplication | Stretching/shrinking arrow | Learning rate scaling in optimization |
| Dot product | Projection length | Similarity measures, kernel methods |
| Norm (length) | Arrow’s magnitude | Regularization, normalization |
| Cross product | Perpendicular vector & area | 3D rotations, geometry-based features |

---

## Vector Operations

Vector operations let you combine, compare, and transform vectors in {R}ⁿ. These are the building blocks for feature manipulations, similarity measures, and optimization steps in data science and ML.

### 1. Explanation

Vectors are arrows in space. Operations on vectors let you:

- **Add/Subtract**: Place arrows tip-to-tail to combine effects or find differences.
- **Scale**: Stretch or shrink an arrow by multiplying by a number.
- **Dot Product**: Measure how much two arrows point in the same direction.
- **Norm (Length)**: Find how long an arrow is.
- **Cross Product (in {R}³)**: Produce a new arrow perpendicular to two given arrows.
- **Projection**: Drop a perpendicular from one arrow onto another to see how much one “shadows” the other.

Each operation has a clear geometric meaning and a direct use in ML tasks like comparing feature vectors or updating parameters.

### 2. Formulas and the Math Behind Them

```
u = [u1, u2, …, un]
v = [v1, v2, …, vn]
```

- **Addition**

```
u + v = [u1+v1, u2+v2, …, un+vn]
```

Combine two effects component-wise.

- **Subtraction**

```
u − v = [u1−v1, u2−v2, …, un−vn]
```

Find the difference arrow.

- **Scalar Multiplication**

```
c · v = [c·v1, c·v2, …, c·vn]
```

Stretch or flip by c.

- **Dot Product**

```
u · v = Σ_{i=1 to n} (ui * vi)
```

Sum of pairwise products; measures alignment.

- **Norm (Euclidean Length)**

```
||v|| = sqrt(Σ_{i=1 to n} vi^2)
```

Distance from origin.

- **Cross Product** (only in {R}³)

```
u × v = [u2·v3 − u3·v2,
         u3·v1 − u1·v3,
         u1·v2 − u2·v1]
```

Perpendicular arrow of area equal to parallelogram spanned by u and v.

- **Projection of u onto v**

```
proj_v(u) = (u·v / v·v) · v
```

Component of u in the direction of v.

### 3. Practice Problems and Examples

### 3.1 By-Hand Exercises

1. Let u = [2, –1, 3], v = [1, 4, 2]. Compute:
    - u + v
    - u − v
    - 2·u
    - u · v
    - ||u||
    - proj_v(u)
2. In {R}³, find u × v for u = [1, 0, 0], v = [0, 1, 0].
3. If u · v = 0 and u ≠ 0, what does that tell you about u and v?

*Solution Sketch for (1):*

- u+v = [3, 3, 5]
- u−v = [1, –5, 1]
- 2·u = [4, –2, 6]
- u·v = 2*1 + (–1)*4 + 3*2 = 2 – 4 + 6 = 4
- ||u|| = sqrt(4 + 1 + 9) = sqrt(14)
- proj_v(u) = (4 / (1+16+4))·[1,4,2] = (4/21)·[1,4,2]

### 3.2 Python with NumPy

```python
import numpy as np

u = np.array([2, -1, 3], float)
v = np.array([1,  4, 2], float)

# Operations
print("u+v       =", u+v)
print("u-v       =", u-v)
print("2·u       =", 2*u)
print("u·v       =", u.dot(v))
print("||u||     =", np.linalg.norm(u))
print("proj_v(u) =", (u.dot(v) / v.dot(v)) * v)

# Cross product example
u3 = np.array([1,0,0], float)
v3 = np.array([0,1,0], float)
print("u3×v3     =", np.cross(u3, v3))
```

### 3.3 Real-World ML Tasks

- **Cosine Similarity** between text embeddings u, v:
    
    ```
    cos_sim = (u·v) / (||u||·||v||)
    ```
    
- **Gradient Update** in optimization:
    
    ```
    θ_new = θ_old − α · ∇L(θ_old)
    ```
    
    uses vector subtraction and scalar multiplication.
    
- **Feature Scaling**: normalize each feature vector v to unit norm:
    
    ```
    v_norm = v / ||v||
    ```
    

### 4. Visual/Geometric Intuition & ML Relevance

- **Addition/Subtraction**: arrows tip-to-tail, showing combined or difference effect.
- **Dot Product**: projection length times magnitude; zero means orthogonal features (no correlation).
- **Norm**: arrow length; used for regularization (penalizing large weights).
- **Cross Product**: area and orientation; used in 3D geometry.
- **Projection**: shadow of one feature on another; fundamental in least-squares.

| Operation | Geometric View | ML Application |
| --- | --- | --- |
| u + v, u − v | Tip-to-tail arrows | Combining feature vectors |
| c · v | Stretch/shrink arrow | Learning rate scaling |
| u · v | Length of projection | Similarity, kernel computations |
|  |  | v |
| u × v | Perpendicular axis, area | 3D rotations, cross-feature patterns |
| proj_v(u) | Foot of perpendicular | Component analysis, dimensionality reduction |

---

## The Dot Product

The dot product (also called the inner product) is a way to multiply two vectors of the same length and get a single number. It measures how much the vectors point in the same direction.

### 1. Explanation

Imagine two arrows starting from the origin:

- If they point in exactly the same direction, their dot product is a large positive number.
- If they point at right angles, their dot product is zero.
- If they point in opposite directions, it’s a large negative number.

You can think of the dot product as the “shadow” one arrow casts onto the other: how much of u lies along v.

### 2. 📐 Formula and Step-by-Step Breakdown

```
u · v = u1*v1 + u2*v2 + … + un*vn
```

- u and v are vectors in {R}ⁿ with components u1…un and v1…vn.
- Multiply each pair of matching components (ui * vi).
- Sum up all those products to get a single scalar.

Why it works step by step:

1. Pair components along each axis.
2. Multiply to see how aligned they are on that axis.
3. Adding captures total alignment across all axes.

Real-world ML example: cosine similarity between text embeddings uses the dot product in its numerator.

### 3. 🧠 Practice Problems & Python Examples

### 3.1 By-Hand Exercises

1. Let u = [3, –1, 2] and v = [1, 4, 0]. Compute u·v.
2. Given u·v = 0 and u ≠ 0, what does that tell you about u and v?
3. Find the angle θ between u and v where
    
    ```
    cos(θ) = (u·v) / (||u|| * ||v||)
    ```
    

*Solution Sketch:*

1. 3*1 + (–1)*4 + 2*0 = 3 – 4 + 0 = –1
2. They are orthogonal (at right angles).
3. Compute norms and then θ = arccos(dot/(norms)).

### 3.2 Python with NumPy

```python
import numpy as np

u = np.array([3, -1, 2], float)
v = np.array([1,  4, 0], float)

# Dot product
dot = u.dot(v)

# Norms
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)

# Angle in degrees
cos_theta = dot / (norm_u * norm_v)
angle_deg = np.degrees(np.arccos(cos_theta))

print("u·v       =", dot)
print("||u||     =", norm_u)
print("||v||     =", norm_v)
print("Angle (°) =", angle_deg)
```

### 4. 📊 Visual/Geometric Intuition & ML Relevance

- Geometric view:
    - u·v = ||u||·||v||·cos(θ), so it ties directly to the angle θ between u and v.
    - Zero dot product → 90° angle → orthogonal features.
- Machine Learning uses:
    - **Cosine Similarity:**Measures similarity of high-dimensional feature vectors:
        
        ```
        cos_sim(u, v) = (u·v) / (||u||·||v||)
        ```
        
    - **Projection:**Project u onto v asUsed in least-squares and component analysis.
        
        ```
        proj_v(u) = (u·v / (v·v)) · v
        ```
        
    - **Kernel Methods:**Many kernels (like linear kernel) are just dot products in feature space.

| Concept | Formula | ML Application |
| --- | --- | --- |
| Dot product | u·v = Σ ui·vi | Similarity, projections |
| Zero dot product | u·v = 0 | Orthogonal (uncorrelated) |
| Projection | (u·v / v·v)·v | Least-squares, PCA |
| Cosine similarity | (u·v)/( |  |

---

## Geometric Dot Product

The geometric dot product expresses the dot product of two vectors in terms of their lengths and the angle between them.

### 1. Explanation

When you take the dot product of two vectors u and v, you’re measuring how much one vector points in the direction of the other. Geometrically, it’s the product of:

- The length (magnitude) of u
- The length of v
- The cosine of the angle θ between them

If θ is small (vectors point in similar directions), cos θ is close to 1 and the dot product is large and positive. If θ=90° (perpendicular), cos θ=0 and the dot product is zero. If θ>90° (opposite general directions), cos θ is negative and the dot product is negative.

### 2. Formulas and Step-by-Step Breakdown

```
u · v = ||u|| * ||v|| * cos(θ)
```

- u and v are vectors in {R}ⁿ.
- ||u|| = sqrt(u1^2 + u2^2 + … + un^2) is the length of u.
- ||v|| = sqrt(v1^2 + v2^2 + … + vn^2) is the length of v.
- θ is the angle between u and v, 0° ≤ θ ≤ 180°.

Why it works step by step:

1. Compute the lengths of each vector (||u|| and ||v||).
2. Measure the angle between them by imagining both arrows tail-to-tail at the origin.
3. Multiply lengths by cos θ to get how much one “drops” onto the other.

### 3. Practice Problems and Python Examples

### 3.1 By-Hand Exercises

1. Given u = [2, 2] and v = [3, 0], find θ using
    
    ```
    cos(θ) = (u · v) / (||u|| * ||v||)
    ```
    
2. Let u = [1, 2, 2], v = [2, 0, 1].
    - Compute ||u|| and ||v||.
    - Compute u·v.
    - Find θ in degrees.

*Answer Sketch for (1):*

- u·v = 2*3 + 2*0 = 6
- ||u|| = sqrt(4+4)=√8, ||v||=3
- cos(θ)=6/(3√8)=2/√8=√2/2 → θ=45°

### 3.2 Python with NumPy

```python
import numpy as np

def angle_between(u, v):
    dot = u.dot(v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cos_theta = dot / (norm_u * norm_v)
    # Clip to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return np.degrees(theta_rad)

# Example vectors
u = np.array([1, 2, 2], float)
v = np.array([2, 0, 1], float)

print("u·v       =", u.dot(v))
print("||u||     =", np.linalg.norm(u))
print("||v||     =", np.linalg.norm(v))
print("Angle (°) =", angle_between(u, v))
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View:**
    
    Place u and v tail-to-tail at the origin. The angle θ between them and their lengths form two sides of a triangle. The dot product equals the product of lengths times cos θ, which is the length of the projection of one onto the other.
    
- **Machine Learning Uses:**
    - **Cosine Similarity:**Widely used to compare text embeddings, user profiles, or image features.
        
        ```
        cos_sim(u, v) = (u · v) / (||u|| * ||v||)
        ```
        
    - **Projection in Least-Squares:**Projecting target vector onto column space via dot products drives the solution.
    - **Orthogonality Checks:**Zero dot product detects uncorrelated features or orthogonal directions in PCA and SVD.

| Concept | Formula | ML Application |
| --- | --- | --- |
| Geometric dot prod | u·v = | Understanding angles in feature spaces |
| Cosine similarity | (u·v)/( | Text/image similarity |
| Projection length | (u·v)/ | Component along v in regression |
| Orthogonality | u·v = 0 | Feature decorrelation |

---

## Multiplying a Matrix by a Vector

Multiplying a matrix by a vector is a way to transform one vector into another using the linear map encoded by the matrix. In machine learning, this operation underlies everything from feature transformations to neural network layers.

### 1. Explanation

When you multiply an m×n matrix A by an n-vector x, you combine each row of A with x to produce one entry of the result.

- Think of A as m “recipes” and x as n “ingredients.”
- Each row of A tells you how much of each ingredient to mix.
- The output is an m-vector whose entries are the results of those m mixes.

Geometrically, A maps the input x from {R}ⁿ into a point in {R}ᵐ by stretching, rotating, or projecting.

### 2. 📐 Formulas and Step-by-Step Breakdown

### 2.1 Definition

```
A = [ a11  a12  …  a1n
      a21  a22  …  a2n
      ⋮     ⋮        ⋮
      am1  am2  …  amn ]

x = [ x1, x2, …, xn ]ᵀ

y = A · x = [ y1, y2, …, ym ]ᵀ
```

Each component yi is:

```
yi = Σ_{j=1 to n} (aij * xj)      for i = 1…m
```

- aij is the entry in row i, column j of A
- xj is the j-th entry of vector x
- yi is the i-th entry of the output vector y

### 2.2 Why It Works

1. Multiply each element of row i by the corresponding element of x.
2. Sum those products to collapse n inputs into one output yi.
3. Repeat for each row to build the full m-vector y.

### 3. 🧠 Practice Problems and Python Examples

### 3.1 By-Hand Exercises

1. Let
    
    ```
    A = [ 2  1  0
          1  3 −1 ]
    x = [ 3,  2,  1 ]ᵀ
    ```
    
    Compute y = A · x by hand.
    
2. If A is 3×3 and x = [1, 0, −1]ᵀ, show which columns of A determine y.
3. Given A·x = y, explain how a zero row in A affects y.

### 3.2 Python with NumPy

```python
import numpy as np

# Define matrix A (2×3) and vector x (3,)
A = np.array([[2, 1, 0],
              [1, 3, -1]], float)
x = np.array([3, 2, 1], float)

# Multiply
y = A.dot(x)

print("A:\n", A)
print("x:", x)
print("y = A·x:", y)
```

### 3.3 Real-World ML Task

- **Feature Projection**
    
    If x is a feature vector of length n and A is a weight matrix of a linear layer in a neural network, y = A·x produces the next layer’s activations in m dimensions.
    
- **Linear Regression Prediction**
    
    With A = wᵀ (1×n) and x the feature vector, y = wᵀ x is a scalar prediction.
    

### 4. 📊 Geometric Intuition & ML Relevance

- **Geometric View**
    
    Each row of A is a hyperplane’s normal. Computing yi = row_i·x is projecting x onto that normal, then summing gives the signed distance.
    
- **Dimensionality Change**
    
    Matrix A can change dimension: it can embed {R}ⁿ into higher or lower {R}ᵐ, rotate, or shear the space.
    

| Operation | Interpretation | ML Application |
| --- | --- | --- |
| A·x | Linear map of x by A | Weight transformations in layers |
| Rows of A | Directions defining each output component | Feature extraction or projection axes |
| Zero row in A | That dimension in y is always zero | Dead neuron in a neural network |

---

## Vector Operations: Scalar Multiplication, Sum, and Dot Product

### 1. Explanation

Scalar multiplication, vector addition, and the dot product are the three core ways to combine or transform vectors:

- **Scalar multiplication** stretches or shrinks a vector by a number (the scalar), like zooming in or out on an arrow.
- **Vector addition** places two arrows tip-to-tail and draws a new arrow from the origin to the end, combining their effects.
- **Dot product** multiplies two vectors to produce a single number that measures how much they point in the same direction, like casting a shadow of one arrow onto another.

These operations let you scale features, merge signals, and quantify similarity in data science.

### 2. Formulas and Step-by-Step Breakdown

### 2.1 Scalar Multiplication

```
c · v = [c*v1, c*v2, …, c*vn]
```

- c is a real number (scalar).
- v = [v1, v2, …, vn] is an n-dimensional vector.
- Multiply each component of v by c to stretch (|c|>1), shrink (0<|c|<1), or flip (c<0) the vector.

### 2.2 Vector Addition (Sum)

```
u + v = [u1+v1, u2+v2, …, un+vn]
```

- u and v are both n-dimensional vectors.
- Add matching components to get a new vector that represents the combined effect.

### 2.3 Dot Product

```
u · v = u1*v1 + u2*v2 + … + un*vn
```

- Produces a single scalar.
- Measures alignment: large positive when u and v point similarly, zero when orthogonal, negative when opposite.

### 3. Practice Problems and Python Examples

### 3.1 By-Hand Exercises

1. Let v = [2, –3, 1]. Compute 3·v and –0.5·v.
2. Given u = [1, 4, 2] and v = [–2, 0, 3], find u + v and u − v.
3. For the same u and v, calculate u·v and interpret the sign and magnitude.

### 3.2 Python with NumPy

```python
import numpy as np

# Define vectors
v = np.array([2, -3, 1], float)
u = np.array([1,  4, 2], float)

# Scalar multiplication
print("3·v       =", 3 * v)
print("-0.5·v    =", -0.5 * v)

# Addition and subtraction
print("u + v     =", u + v)
print("u - v     =", u - v)

# Dot product
dot = u.dot(v)
print("u · v     =", dot)
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Scalar Multiplication:**
    - Geometric: stretches or flips the arrow.
    - ML: adjusts step size in gradient descent (learning rate times gradient).
- **Vector Addition:**
    - Geometric: tip-to-tail combination of arrows.
    - ML: combines feature effects or aggregates updates (e.g., momentum).
- **Dot Product:**
    - Geometric: ||u||·||v||·cos(θ), where θ is the angle between arrows.
    - ML: underlies cosine similarity in text/image embeddings and projection steps in least-squares.

| Operation | Geometric Action | ML Application |
| --- | --- | --- |
| c·v | Stretch/shrink/flip arrow | Learning-rate scaling |
| u + v, u − v | Combine/difference of arrows | Feature aggregation, gradient update |
| u · v | Projection length | Similarity measure, kernel methods |

---

## Matrices as Linear Transformations

A matrix can be viewed as a machine that takes an input vector and produces an output vector by stretching, rotating, reflecting, or projecting space. In this sense, every m×n matrix A defines a linear transformation from {R}ⁿ to {R}ᵐ.

### 1. Explanation

Think of a vector x as a point or arrow in n-dimensional space. When you apply the matrix A to x (compute y = A·x), you’re moving that point according to a fixed rule:

- Rows of A describe how much each output coordinate mixes the input coordinates.
- Columns of A show where the basis vectors of input space end up in output space.

This mapping has two key properties:

- **Additivity:** A·(u + v) = A·u + A·v
- **Homogeneity:** A·(c·v) = c·(A·v)

Because of these, A acts consistently on lines and planes through the origin, preserving their linear structure.

### 2. Formulas and the Math Behind Them

```
A = [ a11  a12  …  a1n
      a21  a22  …  a2n
      ⋮     ⋮        ⋮
      am1  am2  …  amn ]

x = [ x1, x2, …, xn ]ᵀ
```

Applying A to x gives y in {R}ᵐ:

```
y = A · x = [ y1, y2, …, ym ]ᵀ
where
yi = Σ_{j=1 to n} (aij * xj)
```

Step by step:

1. Multiply each entry xj of the input by the coefficient aij in row i.
2. Sum these products to get the i-th component yi of the output.
3. Repeat for i=1…m to build the full output vector y.

Since A·(u + v) = A·u + A·v and A·(c·v) = c·(A·v), A preserves lines through the origin and the origin itself.

### 3. Practice Problems and Python Examples

### 3.1 By-Hand Exercises

1. Let
    
    ```
    A = [ 1  2
          3  4 ]
    ```
    
    and x = [1, 3]ᵀ. Compute y = A·x.
    
2. Interpret column 1 of A as the image of the basis vector e₁ and column 2 as the image of e₂. Draw arrows in {R}² for those images.
3. Show that A·(u + v) = A·u + A·v for u = [2,–1]ᵀ, v = [0,1]ᵀ under the same A.

### 3.2 Python with NumPy

```python
import numpy as np

# Define a 3×2 matrix and two vectors in R^2
A = np.array([[2, 1],
              [0, 3],
              [1,-1]], float)
u = np.array([4, 2], float)
v = np.array([1, 5], float)

# Linear combination property
lhs = A.dot(u + v)
rhs = A.dot(u) + A.dot(v)

print("A·(u + v):", lhs)
print("A·u + A·v:", rhs)

# Visualize columns as images of basis vectors
e1 = np.array([1, 0], float)
e2 = np.array([0, 1], float)
print("A·e1:", A.dot(e1))
print("A·e2:", A.dot(e2))
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View:**
    
    Each column of A is where the corresponding basis axis in {R}ⁿ lands in {R}ᵐ. The entire transformation is determined by where these basis arrows go. Lines through the origin map to lines (or to the zero vector if in the kernel).
    
- **In Machine Learning:**
    - **Feature Projection:** Weight matrices in neural networks multiply input activations, transforming them into new feature spaces.
    - **Dimensionality Reduction:** Linear methods like PCA use matrices to project high-dimensional data onto lower-dimensional subspaces spanned by principal directions.
    - **Data Augmentation:** Transformations such as rotations or scalings in image processing are represented by specific matrices acting on pixel coordinate vectors.

| Aspect | Interpretation | ML Application |
| --- | --- | --- |
| Columns of A | Images of basis vectors | Learned feature directions |
| A·(u + v) | Distribute over sum (additivity) | Superposition in linear models |
| A·(c·v) | Factor out scalar (homogeneity) | Scaling features or parameter updates |
| Kernel of A | Inputs mapping to zero | Identifies redundant directions |

Matrices as linear transformations provide the foundation for understanding how data moves through algorithms and how model parameters shape that movement.

---

## Linear Transformations as Matrices

A **linear transformation** is a rule that takes any input vector in {R}ⁿ, stretches, rotates, reflects, or projects it, and outputs another vector in {R}ᵐ—always preserving sums and scalar multiples. When you fix a basis for the input and output spaces, you can pack all of its action into an m×n matrix.

### 1. Explanation

Imagine you have a machine that:

- Takes an input x with n knobs (coordinates).
- Turns each knob, scales the input, and combines them to produce an output y with m dials.

To fully describe the machine, you only need to know what it does to each of the n standard basis vectors (the ones that have a 1 in one coordinate and 0 elsewhere). You collect the m outputs as columns in a matrix. Feeding any new x into the matrix multiplies those outputs by the knobs’ settings and sums them—exactly recreating the machine’s rule.

Key points you should already know:

- What the standard basis e₁…eₙ are in {R}ⁿ.
- How matrix–vector multiplication works.
- The properties of additivity and homogeneity that define linearity.

### 2. 📐 Formulas and the Math Behind Them

### 2.1 Matrix Representation

If T: {R}ⁿ → {R}ᵐ is linear, define its matrix A by:

```
A = [ T(e₁)  T(e₂)  …  T(eₙ) ]
```

Here each T(eⱼ) is an m-vector, forming the j-th column of A. Then for any x = [x₁, x₂, …, xₙ]ᵀ:

```
T(x) = A · x = x₁·T(e₁) + x₂·T(e₂) + … + xₙ·T(eₙ)
```

### 2.2 Change of Basis

If B and C are bases for the domain and codomain, and P is the change-of-basis matrix from B to the standard basis, then the matrix of T in basis B→C is:

```
A_{B→C} = P_C^{-1} · A · P_B
```

- P_B converts B-coordinates to standard coordinates.
- P_C^{-1} converts standard outputs to C-coordinates.

### 2.3 Composition and Inverse

- **Composition:** If S: {R}ᵐ→{R}ᵏ has matrix B and T: {R}ⁿ→{R}ᵐ has matrix A, then
    
    ```
    S∘T has matrix B·A
    ```
    
- **Inverse:** If T is invertible, its matrix A is invertible, and
    
    ```
    T^{-1}(x) = A^{-1} · x
    ```
    

### 3. 🧠 Practice Problems & Python Examples

### 3.1 By-Hand Exercises

1. Define T: {R}²→{R}² that rotates every vector by 90° counterclockwise.
    - Find T(e₁) and T(e₂).
    - Write the 2×2 matrix A.
2. Let T: {R}³→{R}² be
    
    ```
    T([x, y, z]ᵀ) = [2x − y + 3z,  x + 4y − z]ᵀ
    ```
    
    - Determine A by evaluating T(e₁), T(e₂), T(e₃).
3. Given bases B = { [1,1], [1,−1] } and the standard basis E for {R}², find the change-of-basis matrix P_B→E and then the matrix of T (from exercise 1) in basis B→E.

### 3.2 Python with NumPy

```python
import numpy as np

# 1. Rotation by 90° CCW in R^2
A_rot = np.array([[0, -1],
                  [1,  0]], float)

# Test on a vector
v = np.array([3, 2], float)
print("Rotated v:", A_rot.dot(v))

# 2. Given linear map T in R^3→R^2
def T(v):
    x, y, z = v
    return np.array([2*x - y + 3*z, x + 4*y - z])

# Build matrix by applying T to standard basis
E3 = np.eye(3)
A_T = np.column_stack([T(E3[:, j]) for j in range(3)])
print("Matrix of T:\n", A_T)
```

### 4. 📊 Visual/Geometric Intuition & ML Relevance

- **Columns as Directions:**
    
    Each column of A shows where a basis axis ends up—visualize arrows from the origin to T(eⱼ).
    
- **Composing Layers:**
    
    In neural networks, each layer’s weight matrix is a linear transformation. Stacking layers is matrix multiplication chaining those transformations.
    
- **Feature Embedding:**
    
    Word embeddings or PCA projections are linear maps that send high-dimensional one-hot or raw data vectors into lower-dimensional semantic spaces via matrix multiplication.
    

| Concept | Matrix Interpretation | ML Example |
| --- | --- | --- |
| Basis images | Columns = T(eⱼ) arrows | Weight columns as learned feature detectors |
| Composition | Multiply matrices B·A | Layer stacking in deep nets |
| Change of basis | P_C⁻¹·A·P_B | Converting embeddings between coordinate systems |
| Inverse map | A⁻¹ | Decoding feature representations |

---

## Matrix Multiplication

Multiplying matrices combines two linear transformations into one, or applies one transformation after another. If

```
A is m×n
B is n×p
```

then their product

```
C = A · B
```

is an m×p matrix.

### 1. Explanation

Imagine A as a machine that converts inputs of length n into outputs of length m, and B as a machine that converts length-p inputs into length-n outputs. Doing B first and then A corresponds to feeding your p-length data through B, then sending that result into A. Matrix multiplication packages these two steps into one new machine C = A·B.

Key intuition:

- Rows of A tell you how to combine outputs from B.
- Columns of B tell you how each of B’s inputs flows into its outputs.
- When you multiply, each entry of C blends a row of A with a column of B to capture the full pipeline.

### 2. Formulas and the Math Behind Them

### 2.1 Definition of C = A·B

```
A = [ a11  a12 … a1n
      a21  a22 … a2n
      ⋮     ⋮      ⋮
      am1  am2 … amn ]

B = [ b11  b12 … b1p
      b21  b22 … b2p
      ⋮     ⋮      ⋮
      bn1  bn2 … bnp ]

C = A·B is m×p, with entries

cij = Σ_{k=1 to n} (aik * bkj)
```

- For each i (row of A) and j (column of B), multiply matching entries across that row and column and sum them.
- Requires the inner dimensions n to match.

### 2.2 Why It Works

1. Take row i of A: `[ai1, ai2, …, ain]`.
2. Take column j of B: `[b1j, b2j, …, bnj]`.
3. Multiply component-wise and sum to get cij.

This implements the composition of two linear maps: you first apply B to a basis vector, then apply A, and record the result.

### 3. Practice Problems and Python Examples

### 3.1 By-Hand Exercises

1. Let
    
    ```
    A = [ 1  2  0
          3 −1  4 ]
    B = [ 2  1
          0  3
          5 −2 ]
    ```
    
    Compute C = A·B by hand.
    
    – Row 1 of A · Col 1 of B = 1·2 + 2·0 + 0·5 = 2
    
    – Row 1 of A · Col 2 of B = 1·1 + 2·3 + 0·(−2) = 7
    
    – Continue for row 2.
    
2. If X is a 100×10 data matrix and W is a 10×1 weight vector, what is X·W? (Hint: it’s a 100×1 prediction vector.)
3. Show that (A·B)·x = A·(B·x) for a compatible vector x, verifying associativity.

### 3.2 Python with NumPy

```python
import numpy as np

# Define A (2×3) and B (3×2)
A = np.array([[1,2,0],
              [3,-1,4]], float)
B = np.array([[2,1],
              [0,3],
              [5,-2]], float)

# Method 1: dot
C1 = A.dot(B)

# Method 2: @ operator
C2 = A @ B

print("C via dot:\n", C1)
print("C via @  :\n", C2)
```

### 3.3 Real-World ML Tasks

- **Forward Pass in Neural Nets**
    
    Each layer’s activations h^(l+1) = W^(l)·h^(l) + b^(l) relies on matrix–vector multiplies. Batching many inputs X (batch×features) into W (features×hidden) gives batchwise outputs X·W.
    
- **Gram / Kernel Matrix**
    
    For data X (n×d), the Gram matrix K = X·Xᵀ (n×n) encodes all pairwise inner products, used in kernel methods and spectral clustering.
    
- **Chaining PCA**
    
    Projecting data into k principal components uses A = V_kᵀ (k×d); then new features Z = X·Aᵀ is d×k matrix multiplication.
    

### 4. Visual/Geometric Intuition & ML Relevance

- **Column-by-Column View**
    
    Each column j of C is A multiplied by column j of B:
    
    ```
    C[:, j] = A · (B[:, j])
    ```
    
    Meaning: you see where each basis direction of B goes under the full composition.
    
- **Row-by-Row Interpretation**
    
    Each row i of C combines rows of B, weighted by row i of A, building new hyperplanes in feature space.
    
- **Composition of Transformations**
    
    Geometrically, applying B then A is like first rotating/shearing by B, then rotating/shearing by A. The combined effect is a single linear map C.
    

| Property | Explanation | ML Context |
| --- | --- | --- |
| Associative | (A·B)·C = A·(B·C) | Deep networks stack multiple weight matrices |
| Distributive | A·(B + C) = A·B + A·C | Combining multiple feature pipelines |
| Non-commutative | A·B ≠ B·A in general | Order matters: layers apply in sequence |
| Identity | I·A = A·I = A | Preserves activations when needed |

---

## The Identity Matrix

The identity matrix acts like the number 1 for matrices: multiplying by it leaves any compatible matrix or vector unchanged.

### 1. Explanation

Think of the identity matrix I as the “do nothing” transformation. When you feed a vector x into I, it outputs x without any change. If you compose any transformation A with I (either before or after), you get A back:

- I·x = x
- A·I = I·A = A

In {R}³, I is a 3×3 grid with 1s along the main diagonal (top-left to bottom-right) and 0s elsewhere. Each basis vector eᵢ stays exactly where it is.

### 2. Formulas and Structure

### 2.1 Definition of Iₙ

For size n, the n×n identity matrix Iₙ has entries:

```
(Iₙ)_ij = { 1 if i = j
          { 0 otherwise
```

In block form for n=4:

```
I₄ = [1 0 0 0
      0 1 0 0
      0 0 1 0
      0 0 0 1]
```

### 2.2 Key Properties

- **Multiplicative Identity**:
    
    ```
    Iₘ·A = A = A·Iₙ
    ```
    
    for any A of size m×n.
    
- **Inverse Relationship**:
    
    When A is invertible,
    
    ```
    A·A⁻¹ = Iₙ  and  A⁻¹·A = Iₙ
    ```
    
- **Powers of I**:
    
    ```
    Iₙ^k = Iₙ
    ```
    
    for any integer k ≥ 1.
    

### 3. Practice Problems & Python Examples

### 3.1 By-Hand Exercises

1. Write I₃ by hand.
2. Verify for A = [ [2,3], [1,4] ] that I₂·A = A·I₂ = A.
3. If B is 3×3 and singular, what is I₃ – B? Is it singular or invertible?

### 3.2 Python with NumPy

```python
import numpy as np

# Create identity matrices
I2 = np.eye(2)
I4 = np.eye(4)

# Test A·I and I·A
A = np.array([[2,3],[1,4]], float)
print("I2·A =\n", I2.dot(A))
print("A·I2 =\n", A.dot(I2))

# Check invertible property
A_inv = np.linalg.inv(A)
print("A·A_inv =\n", A.dot(A_inv))
print("A_inv·A =\n", A_inv.dot(A))
```

### 4. Visual/Geometric Intuition & ML Relevance

- **Geometric View:**I maps every vector to itself—no rotation, scaling, or shear.
- **In Machine Learning:**
    - Initialization: identity-like initializations in residual networks (ResNets) let layers start as no-ops, easing training.
    - Regularization: subtracting a scaled identity (XᵀX + λI) improves conditioning in ridge regression.
    - Basis Alignment: I represents a coordinate system where each axis is orthonormal.

| Role of I | Context | ML Example |
| --- | --- | --- |
| Neutral element | A·I = I·A = A | Layer skip connections in ResNets |
| Inverse check | A·A⁻¹ = I | Validating numerical inversion |
| Regularization | XᵀX + λI | Ridge regression’s closed-form |

---

## Matrix Inverse

The **matrix inverse** of an invertible square matrix A is another matrix, denoted A⁻¹, that “undoes” the action of A. When you multiply A by A⁻¹ (in either order), you get the identity matrix:

```
A · A⁻¹ = I = A⁻¹ · A
```

Only matrices with nonzero determinant (det(A) ≠ 0) have inverses.

### 1. Explanation

Think of A as a machine that transforms inputs x into outputs y by y = A·x. The inverse matrix A⁻¹ is the “reverse machine”:

- Feed an output y into A⁻¹ to recover the original input x.
- A⁻¹ literally reverses every stretch, rotation, and shear that A applied.

Just as dividing by a number reverses multiplication by that number, multiplying by A⁻¹ reverses A.

### 2. 📐 Formulas and the Math Behind the Inverse

### 2.1 Definition

```
A is n×n and invertible ↔ ∃ A⁻¹ such that A·A⁻¹ = Iₙ and A⁻¹·A = Iₙ
```

- Iₙ is the n×n identity matrix.
- det(A) ≠ 0 is necessary and sufficient for A⁻¹ to exist.

### 2.2 2×2 Formula

For

```
A = [ a  b
      c  d ]
```

if det(A)=a·d − b·c ≠ 0, then

```
A⁻¹ = (1 / det(A)) · [  d  −b
                      −c   a ]
```

### 2.3 General Computation via Gauss–Jordan

1. Form the augmented matrix `[A | I]`.
2. Row-reduce until the left block becomes I.
3. The right block transforms into A⁻¹.

### 2.4 Relation to LU and Other Factorizations

- If A=LU (with L, U invertible), then A⁻¹ = U⁻¹·L⁻¹.
- For symmetric positive-definite A, Cholesky gives A=LLᵀ → A⁻¹ = L⁻ᵀ·L⁻¹.

### 3. 🧠 Practice Problems & Python Examples

### 3.1 By-Hand Exercise

Invert the matrix

```
A = [ 4  7
      2  6 ]
```

1. Compute det(A)=4·6 − 7·2 = 24 − 14 = 10.
2. Apply the 2×2 formula:
    
    ```
    A⁻¹ = (1/10) · [  6  −7
                    −2   4 ]
    ```
    

### 3.2 Python with NumPy & SymPy

```python
import numpy as np
import sympy as sp

A = np.array([[4,7],
              [2,6]], float)

# NumPy inverse
A_inv_np = np.linalg.inv(A)
print("NumPy inverse:\n", A_inv_np)

# Sympy exact inverse
M = sp.Matrix([[4,7],[2,6]])
A_inv_sp = M.inv()
print("Sympy inverse:\n", A_inv_sp)
```

### 3.3 Real-World ML Task

When solving normal equations in linear regression:

```
(Xᵀ X) · w = Xᵀ y
```

if XᵀX is invertible, you get the unique least-squares solution:

```
w = (Xᵀ X)⁻¹ · (Xᵀ y)
```

### 4. 📊 Visual/Geometric Intuition & ML Relevance

- **Geometric View:**
    
    A transforms space—A⁻¹ maps it back. Volumes scale by det(A); A⁻¹ scales by 1/det(A).
    
- **Numerical Considerations:**
    
    In high dimensions, directly computing A⁻¹ can be unstable or expensive.
    
    - Use specialized solvers (LU, Cholesky) to solve A·x = b without forming A⁻¹ explicitly.
    - Check condition number cond(A)=||A||·||A⁻¹|| to gauge numerical stability.
- **ML Implications:**
    - Ridge regression adds λI to make (XᵀX + λI) invertible and well-conditioned.
    - Inverting covariance matrices in Gaussian models or Kalman filters relies on stable factors rather than direct inversion.

| Aspect | Notes |
| --- | --- |
| Existence | Requires det(A) ≠ 0 |
| Computational cost | O(n³) for general inversion |
| Stability | Governed by condition number; pivoting helps LU |
| Practical use | Solve linear systems, compute sensitivity analyses |

---

## Which Matrices Have an Inverse?

A matrix is **invertible** (nonsingular) exactly when there’s another matrix that “undoes” its action. Below are the key characterizations, hands-on checks, and why it matters in data science.

### 1. Beginner-Friendly Explanation

- Only **square** matrices (n×n) can have a two-sided inverse A⁻¹ satisfying
    
    ```
    A · A⁻¹ = Iₙ  and  A⁻¹ · A = Iₙ
    ```
    
- Geometrically, an invertible matrix is a transformation that doesn’t “flatten” any dimension to zero. It can stretch, rotate, or shear space, but it never collapses it.

### 2. 📐 Formal Criteria

A square matrix A (size n×n) is invertible if and only if any of these equivalent conditions holds:

- **Nonzero Determinant**
    
    ```
    det(A) ≠ 0
    ```
    
- **Full Rank**
    
    ```
    rank(A) = n
    ```
    
    All rows (and columns) are linearly independent.
    
- **Unique Solution**
    
    The linear system
    
    ```
    A·x = b
    ```
    
    has exactly one solution for every b in {R}ⁿ.
    
- **No Zero Eigenvalue**
    
    All eigenvalues λᵢ of A satisfy λᵢ ≠ 0.
    
- **LU / Cholesky / QR Decompositions Exist without Row-Exchange Failures**
    
    Factorizations that rely on pivoting or positivity succeed without breakdown.
    

### 3. 🧠 Practice Problems & Programming Examples

### 3.1 By-Hand Exercises

1. Determine if each matrix is invertible. If yes, compute its determinant; if no, exhibit a linear dependency.
    
    ```
    A = [2  3
         1  4]
    B = [1  2  3
         2  4  6
         0  1  1]
    C = [1  0  0
         0  0  1
         0  1  0]
    ```
    
2. Prove that a triangular matrix with nonzero diagonal entries is invertible.

### 3.2 Python with NumPy

```python
import numpy as np

def is_invertible(A, tol=1e-12):
    return np.linalg.matrix_rank(A) == A.shape[0]

# Example matrices
A = np.array([[2,3],[1,4]], float)
B = np.array([[1,2,3],[2,4,6],[0,1,1]], float)
C = np.array([[1,0,0],[0,0,1],[0,1,0]], float)

for name, M in [('A',A),('B',B),('C',C)]:
    det = np.linalg.det(M) if M.shape[0]==M.shape[1] else None
    inv = None
    try:
        inv = np.linalg.inv(M)
    except:
        pass
    print(f"{name}: square={M.shape[0]==M.shape[1]}, det={det}, invertible={is_invertible(M)}")
    if inv is not None:
        print(f"{name}⁻¹:\n{inv}")
```

### 4. 📊 Visual/Geometric Intuition & ML Relevance

- In **2D**, an invertible 2×2 matrix maps the unit square to a parallelogram of nonzero area (determinant ≠ 0).
- In **ML**:
    - Inverting `(XᵀX)` in linear regression assumes features are independent (full rank).
    - **Regularization** (e.g., ridge) adds λI to make near-singular matrices invertible and stable.
    - **Whitening** transforms a covariance matrix C to I via C^(-1/2) when C is invertible.

| Condition | Interpretation | ML Implication |
| --- | --- | --- |
| det(A) ≠ 0 | Volume scaling ≠ 0 | Unique least-squares solution |
| rank(A)=n | No redundant rows/columns | No multicollinearity in features |
| λᵢ ≠ 0 | No fixed subspace collapsed to 0 | Reliable covariance inversion |

---

## Neural Networks and Matrices

Neural networks are, at their core, compositions of matrix operations and nonlinear activations. By understanding the matrix view, you’ll see how data flows, how learning happens, and how to implement efficient, scalable models.

### 1. Overview

A neural network is a sequence of layers. Each layer:

- Takes an input vector (or batch of vectors).
- Multiplies it by a **weight matrix** and adds a **bias vector**.
- Applies a nonlinear **activation**.

You can think of each layer as a machine that warps space (via the matrix) and then bends or clamps values (via the activation). Chaining these layers builds complex, flexible functions.

### 2. Matrix Representation of Layers

### 2.1 Single Fully-Connected Layer

Let

```
x     ∈ {R}^n      # input vector
W     ∈ {R}^{m×n}  # weight matrix
b     ∈ {R}^m      # bias vector
z     ∈ {R}^m      # pre-activation output
a     ∈ {R}^m      # post-activation output
```

Forward pass:

```
z = W · x + b
a = φ(z)
```

- φ(·) is an elementwise activation (ReLU, sigmoid, tanh, etc.).
- Each row of W is the normal of a hyperplane; b shifts it.

### 2.2 Stacking Multiple Layers

For L layers, index them by ℓ = 1…L:

```
x^(0) = input
z^(ℓ) = W^(ℓ) · x^(ℓ−1) + b^(ℓ)
x^(ℓ) = φ^(ℓ)( z^(ℓ) )
output = x^(L)
```

Here x^(ℓ) is the activation vector after layer ℓ.

### 3. Training via Loss and Backpropagation

### 3.1 Loss Function

Given a network output ŷ and true label y, choose a scalar loss L(ŷ, y) such as:

- **Mean Squared Error**:
    
    ```
    L = 1/2 · ||ŷ − y||^2
    ```
    
- **Cross-Entropy (classification)**:
    
    ```
    L = − Σ y_i · log(ŷ_i)
    ```
    

### 3.2 Computing Gradients with Matrices

To update W^(ℓ), b^(ℓ), compute partial derivatives via the chain rule. For a single example:

1. **Output layer**
    
    ```
    δ^(L) = ∂L/∂z^(L)   # vector of size m_L
    ```
    
2. **Backpropagate** for ℓ = L…1:
    
    ```
    ∂L/∂W^(ℓ) = δ^(ℓ) · (x^(ℓ−1))ᵀ    # outer product
    ∂L/∂b^(ℓ) = δ^(ℓ)               # sum over batch if needed
    δ^(ℓ−1) = (W^(ℓ))ᵀ · δ^(ℓ) * φ' (z^(ℓ−1))
    ```
    

All these are matrix–vector or matrix–matrix products, efficiently handled by libraries.

### 4. Python Example: Two-Layer Feedforward Network

```python
import numpy as np

# Activation and its derivative
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

# Initialize parameters
n_input, n_hidden, n_output = 4, 10, 3
W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2/n_input)
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(2/n_hidden)
b2 = np.zeros(n_output)

# Forward pass
def forward(x):
    z1 = W1.dot(x) + b1
    x1 = relu(z1)
    z2 = W2.dot(x1) + b2
    # Softmax for classification
    expz = np.exp(z2 - np.max(z2))
    y_hat = expz / expz.sum()
    return z1, x1, z2, y_hat

# Backward pass (one example)
def backward(x, y_true, z1, x1, z2, y_hat, lr=1e-2):
    # Compute gradients
    delta2 = y_hat.copy()
    delta2[y_true] -= 1           # ∂L/∂z2 for cross-entropy + softmax
    dW2 = np.outer(delta2, x1)
    db2 = delta2

    delta1 = W2.T.dot(delta2) * relu_grad(z1)
    dW1 = np.outer(delta1, x)
    db1 = delta1

    # Parameter update
    global W1, b1, W2, b2
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1
```

Use minibatches by replacing outer products with matrix multiplications over batches.

### 5. Mini-Batch and Vectorization

Stack B examples into matrix X (n_input×B). Then:

```
Z1 = W1 · X + b1[:, None]     # shape (n_hidden, B)
X1 = φ(Z1)
Z2 = W2 · X1 + b2[:, None]
Y_hat = softmax(Z2)           # shape (n_output, B)
```

Backprop becomes:

```
Δ2 = Y_hat − Y_true_onehot     # (n_output, B)
dW2 = (1/B) · Δ2 · X1ᵀ         # (n_output, n_hidden)
db2 = (1/B) · sum over columns of Δ2

Δ1 = W2ᵀ · Δ2 * φ'(Z1)         # (n_hidden, B)
dW1 = (1/B) · Δ1 · Xᵀ
db1 = (1/B) · sum over columns of Δ1
```

Vectorization turns many small dot products into a few large matrix multiplies, which GPUs execute extremely fast.

### 6. Initialization & Regularization

- **He/Xavier Initialization** sets weight scales to keep activations’ variance stable:
    
    ```
    W ~ N(0, 2/n_in)     # for ReLU
    W ~ N(0, 1/n_in)     # for tanh/sigmoid
    ```
    
- **Dropout**: multiply activations by random mask then scale:
    
    ```
    X1 *= (np.random.rand(*X1.shape) > p) / (1−p)
    ```
    
- **Weight decay** (L2 regularization) adds λ·W to gradient:
    
    ```
    dW += λ * W
    ```
    

### 7. Shapes and Complexity

| Matrix/Vector | Shape | Role |
| --- | --- | --- |
| X | (n_input, B) | Input batch |
| W1 | (n_hidden, n_input) | Layer-1 weights |
| b1 | (n_hidden,) | Layer-1 biases |
| Z1, X1 | (n_hidden, B) | Pre/post activations (layer 1) |
| W2 | (n_output, n_hidden) | Layer-2 weights |
| b2 | (n_output,) | Layer-2 biases |
| Z2, Ŷ | (n_output, B) | Pre/post activations (output layer) |

Each forward/backward pass costs roughly O(B·n_input·n_hidden + B·n_hidden·n_output).

### 8. Extensions and Advanced Layers

- **Convolutional Layers** replace W·X with sliding-window dot products—still matrix multiplies under the hood (via im2col).
- **Recurrent Layers** use shared Wxh and Whh matrices to propagate state over time.
- **Attention Mechanisms** compute queries, keys, and values via separate weight matrices Q, K, V, then combine them with softmaxed dot products.

### 9. Visual/Geometric Intuition & ML Relevance

- Each layer’s W projects inputs onto learned axes (features).
- Nonlinear activations carve space into regions—stacking them yields complex decision boundaries.
- Backprop adjusts W to align hyperplanes so that classes or outputs separate linearly in the last layer’s space.

---

###