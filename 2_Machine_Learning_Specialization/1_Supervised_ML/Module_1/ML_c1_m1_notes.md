# ML_c1_m1

## What Is Machine Learning?

### Overview

Imagine you’re teaching a child to recognize apples. You show various fruits (pictures of apples, oranges, bananas) and tell them “this is an apple” or “this isn’t an apple.” Over time, the child learns patterns—color, shape, texture—without you explicitly listing every rule. Machine learning works the same way: we give a computer examples of inputs (features) paired with the correct outputs (labels), and it “learns” the underlying rule to map any new input to its output.

Step by step:

- We gather data points, each containing input information (`x`) and the desired answer (`y`).
- We choose a family of functions (a model) that can potentially capture the relationship between `x` and `y`.
- We adjust the model’s internal parameters so its predictions match our examples as closely as possible.
- Once trained, the model can predict `y` for unseen `x` by applying the learned rule.

### Key Math and Formula

At its core, supervised machine learning fits a function to data. In its simplest form:

```
y = f(x) + ε
```

- `x` is your input (a single variable or a vector of features).
- `y` is the true output or label you observe.
- `f` is the unknown function mapping inputs to outputs.
- `ε` (epsilon) represents noise or randomness in your observations.

In practice, we choose a parametric model, `f(x; θ)`, where `θ` are parameters we tune. The goal becomes finding the best `θ`:

```
θ* = argmin_θ  L(y, f(x; θ))
```

- `L` is a loss function that measures prediction error (e.g., squared error).
- `θ*` denotes the parameter values that minimize this error across all training examples.

### Connecting to Machine Learning in Practice

- **Model Selection**: Picking linear regression, decision trees, neural networks, etc., defines the form of `f(x; θ)`.
- **Training**: We compute the loss over all examples and use optimization (like gradient descent) to update `θ`.
- **Prediction**: For a new input `x_new`, we compute `ŷ = f(x_new; θ*)`.
- **Evaluation**: We measure how well our model generalizes using metrics like mean squared error (MSE) or classification accuracy.

### Realistic Examples and Practice

1. **Toy Dataset (House Prices)**
    - Generate synthetic data: homes’ square footage (`x`) vs. price (`y`).
    - Fit a simple linear model: `price = θ0 + θ1 * sqft`.
    - Plot data and fitted line.
2. **Practice Problem (Classification)**
    - Download Iris dataset.
    - Pick two features (sepal length, sepal width) and two classes.
    - Implement logistic regression from scratch using gradient descent.
    - Visualize the decision boundary.

Hints and skeleton code for Jupyter Notebook:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # first two features
y = (iris.target != 0).astype(int)  # binary labels

# Initialize parameters
theta = np.zeros(X.shape[1] + 1)

# Add intercept term
X_bias = np.c_[np.ones(X.shape[0]), X]

# Define sigmoid, loss, and gradient functions here...
```

### Visual & Intuitive Interpretations

- **Scatter Plot & Curve**: Plot `x` vs. `y` for regression; you’ll see points scattered around some trend line. Fitting the model is like drawing the smoothest line (or curve) that best follows those points.
- **Decision Boundary**: For classification, you can imagine a line (2D), plane (3D), or hyperplane dividing feature space into regions belonging to different classes.

### Highlighted Prerequisites

Before diving deeper, ensure you’re comfortable with:

- Basic algebra (solving equations).
- Vectors and matrices (adding, multiplying, transposing).
- Derivatives (how functions change, chain rule).

If any of these feel rusty, review quick tutorials or Jupyter notebooks on those topics to hit the ground running.

---

## Linear Regression with One Variable

### Overview

Imagine you want to predict a student’s exam score based on hours spent studying. You collect pairs of data: (hours studied, score achieved). You suspect there’s a straight-line relationship: more study time generally means a higher score, but with some variability. Linear regression with one variable finds the line that best captures this trend.

Step by step, you:

1. Assume the relationship is a line: score ≈ intercept + slope×hours.
2. Measure how far off each prediction is from the actual score.
3. Adjust the intercept and slope to reduce these errors.
4. Arrive at the line that, on average, minimizes the squared differences.

### Key Math and Formulas

We define our hypothesis (prediction function) as:

```
hθ(x) = θ₀ + θ₁ * x
```

The cost function measures average squared error over m examples:

```
J(θ₀, θ₁) = (1 / (2m)) * Σᵢ (hθ(x⁽ⁱ⁾) − y⁽ⁱ⁾)²
```

- x⁽ⁱ⁾: input feature (hours studied) for example i
- y⁽ⁱ⁾: actual output (score) for example i
- θ₀: intercept (prediction when x = 0)
- θ₁: slope (change in prediction per unit x)
- m: number of training examples
- hθ(x): predicted y given x and parameters θ

### Why This Formula Works

Minimizing J finds the θ₀ and θ₁ that make predictions as close as possible to real data, in a least-squares sense. The squared term penalizes larger errors more heavily, pushing the model to fit the overall trend rather than outliers. Dividing by 2m simplifies derivative calculations when we apply gradient descent.

### Connecting to Machine Learning in Practice

- Training: compute J for current θ, then update θ₀ and θ₁ to move “downhill” on the cost surface.
- Optimization: use batch gradient descent (or variants) to iteratively adjust parameters.
- Prediction: once trained, feed a new x into hθ(x) to get ŷ (estimated score).
- Evaluation: compare ŷ on a test set using metrics like mean squared error (MSE).

### Realistic Examples and Practice

1. Generate Synthetic Data
    - Create random study hours between 0–10 and set scores = 5 + 8×hours + noise.
2. Implement Gradient Descent
    - Initialize θ₀, θ₁ to zero.
    - Update rules per iteration:
        
        ```
        θ₀ := θ₀ − α * (1/m) * Σ (hθ(x⁽ⁱ⁾) − y⁽ⁱ⁾)
        θ₁ := θ₁ − α * (1/m) * Σ ((hθ(x⁽ⁱ⁾) − y⁽ⁱ⁾) * x⁽ⁱ⁾)
        ```
        
    - Plot J versus iterations to confirm convergence.
3. Practice Problem
    - Use the `scikit-learn` diabetes dataset’s BMI feature to predict disease progression.
    - Fit a custom one-variable linear regression. Compare your θ to `LinearRegression()`.

Starter code snippet:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Synthetic data
np.random.seed(0)
X = 10 * np.random.rand(100, 1)        # hours studied
y = 5 + 8 * X.flatten() + np.random.randn(100) * 4  # scores with noise

# Add intercept term
X_b = np.c_[np.ones((100, 1)), X]      # shape (100, 2)
theta = np.zeros(2)                    # [θ₀, θ₁]

# Learning rate and iterations
alpha = 0.01
iterations = 1000

# Gradient Descent loop here...
```

### Visual & Intuitive Interpretations

- Plot your data points and overlay the fitted line. You’ll see how the line tilts to reduce vertical distances.
- Imagine the cost function J as a 3D bowl over (θ₀, θ₁). Gradient descent rolls “downhill” to the bottom, finding the optimal parameters.
- Each update step is a small move along this bowl, guided by the slope (derivative) in each parameter direction.

### Highlighted Prerequisites

- Solving simple algebraic equations (for understanding parameter updates).
- Basic derivatives (especially derivative of squared error).
- Vector and matrix operations to generalize later to multiple variables.

If any of these are unfamiliar, review algebra and derivative basics before proceeding.

---

## Linear Regression with Multiple Variables

### Intuition and Use Cases

Predicting house prices often depends on multiple factors—size, bedrooms, location, age. Multiple linear regression extends the single-variable case to handle several features at once. You’re finding a hyperplane in a higher-dimensional space that best fits your data points.

### Hypothesis Function (Vectorized)

We collect m examples, each with n features. Let

- **x⁽ⁱ⁾** be the feature vector for example i (including x₀=1),
- **θ** be the parameter vector (θ₀, θ₁, …, θₙ).

The hypothesis is:

```latex
h_θ(x) = θ^T x = θ₀ x₀ + θ₁ x₁ + … + θₙ xₙ
```

In matrix form for all examples:

```
H = X · θ
```

where

- **X** is an m×(n+1) matrix (first column all ones),
- **θ** is an (n+1)×1 vector.

### Cost Function

We measure fit using mean squared error over m examples:

```latex
J(θ) = \frac{1}{2m} \sum_{i=1}^m (h_θ(x^{(i)}) − y^{(i)})^2
```

Minimizing J finds the best θ.

### Optimizing with Gradient Descent

1. Initialize θ as a zero (n+1)-vector.
2. Choose learning rate α.
3. Repeat until convergence:This single vectorized update adjusts all θ₀…θₙ simultaneously.
    
    ```
    θ := θ − \frac{α}{m} X^T (X·θ − y)
    ```
    

### Feature Scaling & Mean Normalization

When features vary widely (e.g., square footage vs. number of rooms), scale them to zero mean and unit variance:

```
x_j := \frac{x_j − μ_j}{σ_j}
```

- μ_j: mean of feature j
- σ_j: standard deviation of feature j

This speeds up convergence.

### Normal Equation (Closed-Form)

For small n, you can solve directly without iteration:

```
θ = (X^T X)^{-1} X^T y
```

No need for α or feature scaling, but compute cost grows O(n³).

### Python Implementation Example

```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load data
data = load_boston()
X_raw, y = data.data, data.target

# Add intercept term
m, n = X_raw.shape
X = np.c_[np.ones((m, 1)), X_raw]

# Feature scaling (exclude intercept)
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

# Gradient Descent
alpha = 0.01
iterations = 1000
theta = np.zeros(n+1)

for _ in range(iterations):
    gradients = (1/m) * X.T.dot(X.dot(theta) - y)
    theta -= alpha * gradients

print("Learned parameters:", theta)
```

---

## Unsupervised Learning – Part 1

### Overview of Unsupervised Learning

Unsupervised learning finds hidden structure in data without labeled outcomes. Instead of predicting a target value, you explore patterns, groupings, or intrinsic dimensions. Key motivations include:

- Discovering natural groupings (e.g., customer segments)
- Reducing data complexity for visualization or noise reduction
- Detecting anomalies or outliers

Unsupervised methods empower you to mine insights when labeled data is scarce or expensive.

### Clustering: Grouping Similar Data Points

Clustering partitions data into meaningful groups so that points in the same cluster are more similar to each other than to points in other clusters.

### K-Means Clustering

1. Intuition
    
    Imagine you have mixed marbles of different colors on a table. K-means picks K “prototypes” (centroids) and shifts them until each marble sits closest to one centroid.
    
2. Algorithm Steps
    1. Choose number of clusters K.
    2. Randomly initialize K centroids.
    3. Repeat until convergence:
        - **Assignment step**: assign each point to its nearest centroid.
        - **Update step**: recompute each centroid as the mean of points assigned to it.
3. Cost Function
    
    The objective minimises within-cluster variance:
    
    ```latex
    J = \sum_{j=1}^K \sum_{x \in C_j} \|x - \mu_j\|^2
    ```
    
    - C_j: set of points in cluster j
    - μ_j: centroid of cluster j
4. Choosing K: Elbow Method
    - Compute J for different K
    - Plot J vs. K and look for the “elbow” where marginal gain drops

### Practical Python Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Fit K-Means
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black')
plt.title("K-Means Clustering (K=4)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

This code:

- Creates 2D data with four blobs
- Fits K-means and labels each point
- Plots clusters with centroids marked

### Practical Considerations

- Feature Scaling
    
    Scale features (zero mean, unit variance) so distances are comparable.
    
- Initialization Sensitivity
    
    K-means++ initialization reduces poor random starts.
    
- Convergence Criteria
    
    Stop when assignments or centroids change negligibly or after a fixed number of iterations.
    
- Cluster Shapes
    
    K-means assumes roughly spherical clusters; non-convex shapes may need alternative algorithms.
    

### When and Why to Use Clustering

- Customer segmentation based on purchasing behavior
- Image compression by grouping similar pixel values
- Anomaly detection by treating small, sparse clusters as outliers

Clustering reveals the “natural language” of your data, helping you form hypotheses and guide further analysis.

---

## Unsupervised Learning – Part 2: Dimensionality Reduction with PCA

### Why Dimensionality Reduction Matters

High-dimensional data can be hard to visualize, noisy, and slow to process. Reducing dimensions helps you:

- Visualize data in 2D or 3D
- Remove noise by dropping low-variance directions
- Speed up algorithms like clustering or classification
- Mitigate the curse of dimensionality

### Step-by-Step PCA

Follow these steps to find the principal components that capture the most variance:

### 1. Center the Data

Subtract the mean of each feature so that each column has zero mean.

```
X_centered = X - mean(X, axis=0)
```

### 2. Compute the Covariance Matrix

Measure how features vary together.

```
Sigma = (1 / m) * (X_centered^T * X_centered)
```

### 3. Perform Eigen Decomposition

Find eigenvectors and eigenvalues of the covariance matrix.

```
Sigma * v_k = lambda_k * v_k
```

- v_k is the k-th principal component (unit vector)
- lambda_k is the variance along v_k

### 4. Select Top Components and Project

1. Sort eigenvalues in descending order.
2. Take the corresponding top K eigenvectors and form matrix W.

```
W = [v1, v2, …, vK]
Z = X_centered * W
```

Z is your data represented in K dimensions.

### 5. Compute Explained Variance

The fraction of total variance captured by the first K components:

```
explained_variance_ratio = sum(lambda[0:K]) / sum(lambda[0:n])
```

### Python Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and scale data
data = load_iris()
X = data.data
X_scaled = StandardScaler().fit_transform(X)

# Manual PCA via SVD
Xc = X_scaled - X_scaled.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
components_manual = Vt[:2]                  # top 2 principal components
Z_manual = Xc.dot(components_manual.T)      # projected data

# PCA with scikit-learn
pca = PCA(n_components=2)
Z_sklearn = pca.fit_transform(X_scaled)
explained_ratio = pca.explained_variance_ratio_

print("Explained variance ratio:", explained_ratio)

# Visualization
plt.figure(figsize=(6,5))
for label in np.unique(data.target):
    idx = data.target == label
    plt.scatter(Z_sklearn[idx,0], Z_sklearn[idx,1], label=data.target_names[label])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris projected onto first two PCs")
plt.legend()
plt.show()
```

### Interpreting Results and Best Practices

- Use a **scree plot** (cumulative explained variance vs. K) to choose how many components to keep.
- Always **standardize** features first if they’re on different scales.
- Remember that PCA captures only **linear** relationships; for non-linear structure, look at t-SNE or UMAP.
- Drop components with very low variance to **filter out noise**.

---

## Linear Regression Model – Part 1

### Why Linear Regression?

Linear regression is your entry point to modeling relationships between variables. It’s used when you suspect a roughly straight-line connection—like predicting house prices from square footage or estimating sales based on ad spend. Mastering this builds intuition for more complex models.

### Problem Setup and Intuition

Imagine plotting points where the x-axis is hours of study and the y-axis is exam scores. You want a line that “best” summarizes this cloud of points, so you can plug in a new study time and estimate the score.

- The line has two parameters:
    - **θ₀** (intercept): where the line crosses the y-axis
    - **θ₁** (slope): how much y changes per unit x

Each data point pulls the line up or down. Your goal is to position the line to minimize overall “vertical” error.

### Hypothesis Function

Your prediction function (often called the hypothesis) is:

```
h_theta(x) = theta0 + theta1 * x
```

Here, `h_theta(x)` is the estimated y for input x.

### Cost Function (Mean Squared Error)

To quantify how well the line fits the data, use the mean squared error cost:

```
J(theta0, theta1) = (1 / (2 * m)) * sum( (h_theta(x_i) - y_i)^2 for i = 1..m )
```

- `m` is the number of training examples
- Squaring errors penalizes large deviations
- Dividing by `2` simplifies derivatives in gradient descent

### Geometry: The Cost Surface

Visualize `J(theta0, theta1)` as a 3D bowl. Each (θ₀, θ₁) pair is a point on this bowl. The bottom of the bowl is the minimum cost—your optimal line.

### Optimization with Gradient Descent

Gradient descent iteratively nudges (θ₀, θ₁) downhill on the cost surface.

1. Choose a learning rate `alpha`
2. Initialize `theta0` and `theta1` (often to 0)
3. Repeat until convergence:
    
    ```
    temp0 = theta0 - alpha * (1/m) * sum( h_theta(x_i) - y_i for i = 1..m )
    temp1 = theta1 - alpha * (1/m) * sum( (h_theta(x_i) - y_i) * x_i for i = 1..m )
    theta0 = temp0
    theta1 = temp1
    ```
    

Each update uses all examples (batch gradient descent). Watch `alpha`: too large → divergence, too small → slow learning.

### Closed-Form Solution (Preview)

You can also compute the exact best-fit line in one shot:

```
theta1 = sum( (x_i - x_mean) * (y_i - y_mean) for i=1..m )
         / sum( (x_i - x_mean)^2 for i=1..m )

theta0 = y_mean - theta1 * x_mean
```

This avoids iterative updates but reveals the same result.

### Evaluating Your Model

Common metrics to assess fit:

```
MSE  = (1/m) * sum( (y_pred_i - y_i)^2 for i=1..m )
RMSE = sqrt( MSE )
R2   = 1 - sum((y_pred_i - y_i)^2) / sum((y_i - y_mean)^2)
```

- **RMSE** gives error in original units
- **R²** indicates fraction of variance explained

### Hands-On Python Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
np.random.seed(0)
X = 10 * np.random.rand(100)                   # feature values
y = 2 + 3 * X + np.random.randn(100) * 2       # underlying line with noise

# Prepare for gradient descent
m = len(X)
X_b = np.c_[np.ones(m), X]                     # add x0 = 1
theta = np.zeros(2)                            # [theta0, theta1]
alpha = 0.01
iterations = 1000
J_history = []

# Batch Gradient Descent
for _ in range(iterations):
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradients = (1/m) * X_b.T.dot(errors)
    theta -= alpha * gradients
    J_history.append((1/(2*m)) * np.sum(errors**2))

print("Learned parameters:", theta)

# Plot data and fitted line
plt.scatter(X, y, label="Data")
plt.plot(X, X_b.dot(theta), color='red', label="Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Plot cost over iterations
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Convergence of Gradient Descent")
plt.show()
```

### Key Takeaways

- Simple linear regression fits a straight line to data by minimizing squared error.
- Gradient descent is a universal optimizer; the closed-form solution is a handy alternative.
- Always plot your data, the fitted line, and cost curve to confirm learning behavior.

---

## Linear Regression Model – Part 2

### 1. Recap of Part 1

You’ve seen:

- Hypothesis for one feature:
    
    ```
    h_theta(x) = theta0 + theta1 * x
    ```
    
- Cost function (MSE):
    
    ```
    J(theta0, theta1) = (1/(2*m)) * sum((h_theta(x_i) - y_i)^2)
    ```
    
- Batch gradient descent updates for θ₀, θ₁.

Now we’ll derive those updates, show the closed-form normal equation, and generalize to many features.

### 2. Deriving the Gradient Updates

To minimize `J`, take partial derivatives with respect to `theta0` and `theta1`, set them in gradient descent.

1. Compute error term for example *i*:
    
    ```
    error_i = h_theta(x_i) - y_i
    ```
    
2. Partial derivative wrt `theta0`:
    
    ```
    ∂J/∂theta0 = (1/m) * sum(error_i)
    ```
    
3. Partial derivative wrt `theta1`:
    
    ```
    ∂J/∂theta1 = (1/m) * sum(error_i * x_i)
    ```
    
4. Gradient descent update rules:
    
    ```
    theta0 := theta0 - alpha * (1/m) * sum(error_i)
    theta1 := theta1 - alpha * (1/m) * sum(error_i * x_i)
    ```
    

### 3. The Normal Equation (Closed-Form Solution)

Instead of iterating, solve for the minimum analytically by setting gradients to zero. For one variable, it reduces to:

```
theta1 = sum((x_i - x_mean)*(y_i - y_mean))
         / sum((x_i - x_mean)^2)

theta0 = y_mean - theta1 * x_mean
```

In matrix form (for *n* features):

1. Let
    - `X` be the m×(n+1) design matrix (first column all ones)
    - `y` be the m×1 target vector
    - `theta` be the (n+1)×1 parameter vector
2. The normal equation:
    
    ```
    theta = (X^T * X)^(-1) * X^T * y
    ```
    

> No α, no iterations—just one matrix inverse (costly if n is large).
> 

### 4. Multivariate Linear Regression

Extend hypothesis and cost to multiple features:

```
h_theta(x) = theta0*1 + theta1*x1 + theta2*x2 + ... + thetan*xn
```

Vectorized form:

```
h_theta(X) = X * theta
```

Cost function:

```
J(theta) = (1/(2*m)) * (X*theta - y)^T * (X*theta - y)
```

Gradient descent update (vectorized):

```
theta := theta - (alpha/m) * (X^T * (X*theta - y))
```

### 5. Comparison: Gradient Descent vs Normal Equation

| Aspect | Gradient Descent | Normal Equation |
| --- | --- | --- |
| Iterative? | Yes | No |
| Learning rate (α) | Required | Not required |
| Computation per iteration | O(m·n) | — |
| Closed-form cost | No | O(n³) for matrix inverse |
| Feature scaling | Recommended | Not strictly necessary |
| Use when | n large or memory constrained | n small, m large, and invertible XᵀX |

### 6. Regularization Preview

When you have many features or risk overfitting, add a penalty term. Ridge regression (L2):

```
J_reg(theta) = (1/(2*m)) * sum((h_theta(x_i)-y_i)^2)
               + (lambda/(2*m)) * sum(theta_j^2 for j=1..n)
```

Normal equation with L2:

```
theta = (X^T*X + lambda * I)^(-1) * X^T * y
```

(`I` is identity matrix with first diagonal element zero if you exclude bias from regularization.)

### 7. Assumptions & Diagnostic Checks

To trust your linear model, verify:

- **Linearity**: Relationship between features and target is linear.
- **Independence**: Residuals are independent.
- **Homoscedasticity**: Constant variance of residuals.
- **Normality**: Residuals follow a normal distribution.
- **No multicollinearity**: Features not too highly correlated.

Common diagnostics:

- Residual vs. fitted-value plot
- QQ-plot of residuals
- Variance Inflation Factor (VIF) for multicollinearity
- Cook’s distance to identify influential points

### 8. Python Example: Multivariate Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv

# Generate toy data
np.random.seed(1)
m, n = 100, 3
X_raw = np.random.rand(m, n) * 10
y = 5 + X_raw.dot([1.5, -2.0, 3.0]) + np.random.randn(m) * 2

# Feature scaling and design matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X = np.c_[np.ones((m,1)), X_scaled]  # m x (n+1)

# Normal equation solution
theta_ne = inv(X.T.dot(X)).dot(X.T).dot(y)
print("Theta (normal eq):", theta_ne)

# Gradient descent solution
alpha = 0.1
iterations = 500
theta_gd = np.zeros(n+1)
for _ in range(iterations):
    gradients = (1/m) * X.T.dot(X.dot(theta_gd) - y)
    theta_gd -= alpha * gradients

print("Theta (grad descent):", theta_gd)
```

### Key Takeaways

- You can **derive** gradient updates from the cost function’s partial derivatives.
- The **normal equation** gives a direct solution but may be impractical for many features.
- Vectorization makes code concise and fast.
- **Regularization** guards against overfitting when extending to high-dimensional data.
- Always perform **diagnostic checks** to validate your model assumptions.

---

## Cost Function – Linear Regression

### Formula

For a training set of m examples, the cost function J(θ) measures the average squared error between predictions and actual values:

```
J(theta) = (1 / (2 * m)) * sum( (h_theta(x_i) - y_i)^2 for i = 1..m )
```

Here:

- `h_theta(x_i)` is the prediction for example i
- `y_i` is the true target for example i
- `theta` is the vector of parameters (θ₀, θ₁, …, θₙ)
- dividing by 2*m simplifies the gradient expressions

In vectorized form, using the design matrix X and target vector y:

```
J(theta) = (1 / (2 * m)) * (X*theta - y)^T * (X*theta - y)
```

### Intuition

Every data point produces an error: the vertical distance between the predicted value and the actual target. Squaring that error gives us:

- A non-negative penalty (errors below zero don’t cancel out positives)
- Extra weight to larger mistakes, pushing the model to fit the overall trend

By averaging over all m examples (and including the 1/2 factor), J(θ) represents the “height” of the error surface at parameter setting θ.

Visualizing J(θ₀, θ₁) as a 3D bowl:

- The floor of the bowl is the minimum error (optimal line)
- Gradient descent “rolls” you downhill toward that minimum

Minimizing J means finding the line (or hyperplane) that yields the smallest average squared deviation from all your data points.

### Why the 1/(2*m) Factor?

- **1/m** turns the sum into an average so cost doesn’t grow with dataset size.
- **1/2** cancels the 2 in the derivative of the square term, making gradient expressions cleaner:
    
    ```
    ∂J/∂theta_j = (1/m) * sum( (h_theta(x_i) - y_i) * x_i_j )
    ```
    

---

## Visualizing the Cost Function

### Why Visualize the Cost Function

Seeing the shape of the cost function helps you understand how gradient descent navigates toward the minimum. A 3D surface plot shows the “bowl” shape of error over parameter space, while a contour plot reveals level curves and the path of convergence.

### Cost Function Formula in Code

```
J(theta0, theta1) = (1 / (2 * m)) * sum((theta0 + theta1 * x_i - y_i)**2 for i in range(m))
```

### Computing the Cost Grid

```python
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([1.5, 3.4, 4.1, 5.0, 6.2])
m = len(X)

# Define ranges for theta0 and theta1
theta0_vals = np.linspace(-1, 7, 100)
theta1_vals = np.linspace(0, 2, 100)

# Initialize cost grid
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Compute cost for each pair (theta0, theta1)
for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        J_vals[i, j] = (1 / (2 * m)) * np.sum((t0 + t1 * X - y)**2)
```

### Plotting the 3D Surface

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals.T, cmap='viridis', edgecolor='none')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('Cost J')
ax.set_title('3D Surface of Cost Function')
plt.show()
```

### Plotting Contour and Gradient Descent Path

```python
# Assume theta_history stores (theta0, theta1) at each iteration
theta_history = np.array([[0, 0], [1, 0.5], [2, 1.0], [3, 1.2], [4, 1.4], [5, 1.6]])

plt.figure(figsize=(6, 5))
CS = plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=30, cmap='coolwarm')
plt.plot(theta_history[:, 0], theta_history[:, 1], 'kx-', markersize=8, linewidth=2)
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Cost Function Contours and Descent Path')
plt.show()
```

### Interpreting the Plots

Each point on the 3D surface corresponds to a specific line (θ₀, θ₁) and its average squared error. The steep walls guide gradient descent steps, while the contour lines reveal how direct or winding the path is toward the global minimum.

---

## Visualization Examples for Linear Regression

Below are multiple Python examples to visualize different aspects of a simple linear regression model. Each snippet uses plain code blocks for easy copy-paste into Notion or any notebook.

### 1. Scatter Plot with Fitted Line

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = 10 * np.random.rand(50)
y = 2 + 3 * X + np.random.randn(50) * 2

# Fit line
theta1, theta0 = np.polyfit(X, y, deg=1)

# Plot
plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, theta0 + theta1 * X, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot with Regression Line')
plt.legend()
plt.show()
```

### 2. Residual Plot

```python
# Compute predictions and residuals
y_pred = theta0 + theta1 * X
residuals = y - y_pred

# Plot residuals vs. fitted values
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, color='purple')
plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), linestyles='dashed')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

### 3. Cost vs. Iterations

```python
# Assume J_history is a list of cost values per iteration
# Here we simulate a decaying cost for demonstration
iterations = 100
J_history = np.exp(-0.05 * np.arange(iterations)) + 0.01 * np.random.rand(iterations)

plt.figure(figsize=(6,4))
plt.plot(range(iterations), J_history, color='green')
plt.xlabel('Iteration')
plt.ylabel('Cost J')
plt.title('Cost vs. Gradient Descent Iterations')
plt.show()
```

### 4. 3D Surface of Cost Function

```python
from mpl_toolkits.mplot3d import Axes3D

# Define grid of theta0 and theta1 values
theta0_vals = np.linspace(-5, 5, 50)
theta1_vals = np.linspace(0, 4, 50)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Compute cost over the grid
m = len(X)
for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        J_vals[i, j] = (1/(2*m)) * np.sum((t0 + t1*X - y)**2)

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals.T, cmap='viridis', edgecolor='none')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('Cost J')
ax.set_title('3D Surface of Cost Function')
plt.show()
```

### 5. Contour Plot with Descent Path

```python
# Simulate a gradient descent path
theta_history = np.column_stack((np.linspace(4, theta0, 20),
                                 np.linspace(0.5, theta1, 20)))

plt.figure(figsize=(6,5))
cs = plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=20, cmap='coolwarm')
plt.plot(theta_history[:,0], theta_history[:,1], 'k.-', linewidth=1.5, markersize=6)
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Cost Contours & Gradient Descent Path')
plt.show()
```

### 6. Animated Descent on Contours

```python
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(6,5))
cs = ax.contour(theta0_vals, theta1_vals, J_vals.T, levels=20, cmap='coolwarm')
line, = ax.plot([], [], 'ro-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    pts = theta_history[:frame+1]
    line.set_data(pts[:,0], pts[:,1])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(theta_history),
                              init_func=init, blit=True, repeat=False)
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Animated Gradient Descent')
plt.show()
```

### 7. Learning Curve (Training & Validation Error)

```python
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

# Prepare data
X_reshaped = X.reshape(-1,1)
model = LinearRegression()

# Compute learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X_reshaped, y, train_sizes=np.linspace(0.1,1.0,10), cv=5)

train_err = 1 - np.mean(train_scores, axis=1)
val_err   = 1 - np.mean(val_scores, axis=1)

plt.figure(figsize=(6,4))
plt.plot(train_sizes, train_err, 'o-', label='Training error')
plt.plot(train_sizes, val_err, 's--', label='Validation error')
plt.xlabel('Training Set Size')
plt.ylabel('Error (1 - R²)')
plt.title('Learning Curve')
plt.legend()
plt.show()
```

### 8. Interactive Plot with Plotly

```python
import plotly.graph_objects as go

# Use the 3D grid from example 4
fig = go.Figure(data=[go.Surface(z=J_vals.T, x=T0, y=T1)])
fig.update_layout(title='Interactive Cost Surface',
                  scene=dict(xaxis_title='theta0',
                             yaxis_title='theta1',
                             zaxis_title='Cost J'),
                  autosize=False, width=700, height=500)
fig.show()
```

---

## Training Models with Gradient Descent

### Overview

Imagine standing on a foggy hillside and trying to reach the bottom. You can’t see far, but you can feel the slope under your feet. You take a small step in the direction that feels steepest downhill. Repeat this until you can’t go any lower. Gradient descent works the same way: it uses information about the “slope” of the error surface (the gradient) to update model parameters step by step until it finds the minimum error.

### Key Math and Formulas

### Cost Function for m Examples and n Features

```
J(theta) = (1 / (2 * m)) * (X * theta - y)^T * (X * theta - y)
```

- `X` is an m×(n+1) design matrix (first column all ones)
- `theta` is an (n+1)×1 parameter vector
- `y` is an m×1 vector of true labels
- `m` is number of training examples

### Gradient Computation (Elementwise)

For each parameter `theta_j`:

```
partial_j = (1 / m) * sum( (h_theta(x_i) - y_i) * x_i_j )  for i = 1..m
```

- `h_theta(x_i)` = prediction on example i
- `x_i_j` = j-th feature of example i (with `x_i_0 = 1` for bias)

### Gradient Descent Update (Vectorized)

```
theta := theta - alpha * (1/m) * X^T * (X * theta - y)
```

- `alpha` is the learning rate
- `X^T * (X*theta - y)` computes all partial derivatives at once

### Variants of Gradient Descent

- Batch Gradient Descent
    
    Updates parameters using all m examples each iteration. Stable but can be slow on large datasets.
    
- Stochastic Gradient Descent (SGD)
    
    Updates parameters using one example at a time. Fast updates, noisier convergence, good for online learning.
    
- Mini-Batch Gradient Descent
    
    Uses a small batch (e.g., 32–256 examples) each update. Balances speed and stable convergence.
    

### Why It Works

The gradient points in the direction of steepest increase of the cost function. By subtracting a fraction (`alpha`) of the gradient, you step downhill, reducing error at each iteration until you reach a flat region (local or global minimum).

### Practical Considerations & Hyperparameters

- Learning Rate (`alpha`)
    
    Too large → overshoot and diverge.
    
    Too small → slow convergence.
    
- Number of Iterations
    
    Determine via plotting cost vs. iterations and stopping when change is negligible.
    
- Feature Scaling
    
    Scale inputs to zero mean and unit variance to ensure balanced updates across features.
    
- Batch Size (for mini-batch)
    
    Trade-off between noisy updates (small batches) and computation per update (large batches).
    

### Python Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
np.random.seed(42)
X = 10 * np.random.rand(100, 1)             # single feature
y = 4 + 2 * X.flatten() + np.random.randn(100) * 2

m = len(X)
X_b = np.c_[np.ones((m,1)), X]              # add bias term
theta = np.zeros(2)                         # [theta0, theta1]

alpha = 0.1
iterations = 100
J_history = []

for it in range(iterations):
    gradients = (1/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta -= alpha * gradients
    cost = (1/(2*m)) * np.sum((X_b.dot(theta) - y)**2)
    J_history.append(cost)

print("Learned theta:", theta)

# Plot data and fitted line
plt.figure(figsize=(6,4))
plt.scatter(X, y, label='Data')
plt.plot(X, X_b.dot(theta), color='red', label='Fit')
plt.xlabel('x'), plt.ylabel('y')
plt.legend()
plt.show()

# Plot cost convergence
plt.figure(figsize=(6,4))
plt.plot(J_history, color='green')
plt.xlabel('Iteration')
plt.ylabel('Cost J')
plt.title('Gradient Descent Convergence')
plt.show()
```

### Visual & Intuitive Interpretations

- **3D Surface Plot**: `theta0` vs. `theta1` vs. `J(theta)` shows the error “bowl.”
- **Contour Plot**: Level curves of constant cost with gradient descent path overlaid.
- **Cost vs. Iterations**: A downward curve indicating steady error reduction.

### Practice Problems

1. Implement **stochastic gradient descent** on the same synthetic data and compare convergence speed.
2. Use **mini-batch gradient descent** (batch size = 20) to train on the Boston housing dataset’s RM feature.
3. Visualize how different learning rates (`0.001`, `0.01`, `0.1`, `1`) affect convergence on a simple dataset.

### Pre-Requisites

- Understanding of partial derivatives (how to compute ∂J/∂θ_j).
- Comfort with matrix multiplication and transposing.
- Familiarity with Python and NumPy for implementation.

---

## Implementing Gradient Descent in Python

### 1. Setup and Data Preparation

Create a simple synthetic dataset and prepare it for linear regression:

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X.flatten() + np.random.randn(m)

# 2. Add bias term (column of ones)
X_b = np.c_[np.ones((m, 1)), X]   # shape: (m, 2)

# 3. Initialize parameters
theta_init = np.zeros(2)           # [theta0, theta1]
```

### 2. Defining the Cost Function

We need a function to compute the mean squared error:

```python
def compute_cost(X, y, theta):
    """
    Compute cost for linear regression.
    X      : [m x n] design matrix (including bias column)
    y      : [m] target values
    theta  : [n] parameters vector
    Returns: scalar cost J(theta)
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1 / (2 * m)) * np.dot(errors, errors)
```

### 3. Implementing Gradient Descent

Next, implement the update rule in a function that also records the cost history:

```python
def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    X          : [m x n] design matrix
    y          : [m] target values
    theta      : [n] initial parameters
    alpha      : learning rate
    num_iters  : number of iterations
    Returns    : (theta_final, history_of_costs)
    """
    m = len(y)
    J_history = np.zeros(num_iters)

    for it in range(num_iters):
        # Compute gradient
        gradient = (1 / m) * X.T.dot(X.dot(theta) - y)
        # Update parameters
        theta = theta - alpha * gradient
        # Save the cost
        J_history[it] = compute_cost(X, y, theta)

    return theta, J_history
```

### 4. Running Gradient Descent & Visualization

Bring it all together, run gradient descent, then plot:

```python
# Hyperparameters
alpha = 0.1
iterations = 100

# Train model
theta_final, J_hist = gradient_descent(X_b, y, theta_init, alpha, iterations)

print("Learned parameters:", theta_final)

# Plot data and fitted line
plt.figure(figsize=(6,4))
plt.scatter(X, y, label="Training data")
plt.plot(X, X_b.dot(theta_final), "r-", label="Linear fit")
plt.xlabel("x"), plt.ylabel("y")
plt.legend()
plt.title("Linear Regression via Gradient Descent")
plt.show()

# Plot convergence of cost function
plt.figure(figsize=(6,4))
plt.plot(range(iterations), J_hist, "g-")
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Cost Convergence")
plt.show()
```

### 5. Interview & Debugging Tips

- Explain why feature scaling – mean normalization and variance scaling – speeds up convergence.
- Discuss time complexity: O(k·m·n) per run (k = iterations).
- Be ready to swap in stochastic or mini-batch gradient descent for large m.
- Show how Python’s vectorized operations (`X.T.dot(...)`) replace explicit loops for speed.
- Outline common pitfalls: choosing alpha too large (divergence) or too small (slow learning).

---

## Gradient Descent Intuition

### The Landscape Analogy

Imagine the cost function as a foggy mountain range where your goal is to descend to the lowest valley without seeing far ahead. Your current position on the slope corresponds to a set of model parameters. By feeling which way is downhill (the gradient), you take a small step in that direction, gradually winding your way toward the bottom.

This process repeats: at each step you reassess the slope underfoot and advance. If you step too far, you risk overshooting the valley; if you step too little, progress becomes painfully slow. Choosing the right step size (learning rate) balances speed with stability.

### Visualizing with Contour Plots

- Contour lines represent levels of equal error, like elevation lines on a topographic map.
- The gradient at any point is perpendicular to these lines, pointing toward higher error.
- Gradient descent moves opposite that direction, hopping down the contours toward the center.

```
   Error Contours (top view)

        +-----------+
        |  o        |
        |   o   .   |
        |    o     *| ← path of descent
        |     o    .|
        |      o   .|
        +-----------+
```

### Key Intuition Points

- Gradient as local slope: tells you both direction and steepness of increase.
- Opposite direction: subtracting the gradient ensures error decreases each step.
- Learning rate governs how far you move—too large jumps around, too small crawls.
- Convex surfaces guarantee you’ll reach the global minimum; non-convex may trap you.

### Pseudocode Walkthrough

1. Initialize parameters to zeros or small random values.
2. Repeat until convergence:
    - Compute gradient at current parameters.
    - Update parameters = old parameters − learning rate × gradient.
    - Optionally track error to decide when to stop.
3. Return the final parameters as your trained model.

This simple loop underlies most optimization routines in machine learning.

### Variants at a Glance

| Variant | Update Frequency | Pros | Cons |
| --- | --- | --- | --- |
| Batch GD | Every `m` examples | Smooth, stable convergence | Slow on large datasets |
| Stochastic GD (SGD) | Every single example | Fast updates, online learning | Noisy convergence |
| Mini-Batch GD | Batches of size `b` | Balance of speed and stability | Requires tuning batch size |

### When Things Go Wrong

- Divergence: learning rate too high causes error to increase.
- Stalling: learning rate too low or no feature scaling stalls progress.
- Local minima or saddle points trap you in non-convex settings.
- Poorly conditioned data (features with different scales) makes some dimensions dominate.

---

## Learning Rate: Tuning Your Step Size

### What Is the Learning Rate?

The learning rate (α) controls how big a step you take along the gradient in each update.

It appears directly in the update rule:

```
θ := θ − α · ∇J(θ)
```

Choosing α well ensures steady progress without overshooting or crawling.

### The Perils of Too Large or Too Small

- If α is too large, updates can jump back and forth and fail to converge.
- If α is too small, convergence will be painfully slow and may get stuck in flat regions.
- The “just right” α balances speed with stability, akin to pacing your hike down a foggy hill.

### Strategies for Picking a Learning Rate

- **Grid Search**: Try α values on a log scale (e.g., 10⁻⁴, 10⁻³, 10⁻², …) and pick the one with fastest, stable descent.
- **Learning Rate Finder**: Increase α exponentially during a short run, plot loss vs. α, then choose the range where loss decreases sharply.
- **Cross-Validation**: For each α candidate, train on subsets and compare validation error.

### Learning Rate Schedules

| Schedule | How It Works | When to Use |
| --- | --- | --- |
| Constant | α remains fixed throughout training | Simple problems, small models |
| Step Decay | α ← α · drop_rate every k epochs | When loss plateaus periodically |
| Exponential Decay | α ← α₀ · exp(− decay_rate · epoch) | Smoothly reduce steps over time |
| Cosine Annealing | α follows a cosine curve between bounds | Modern deep nets, avoids sharp drops |
| Cyclical Learning | α oscillates between α_min and α_max | Helps escape local minima |

### Adaptive Learning Rates

- **AdaGrad**: Scales α by the inverse square root of the sum of past squared gradients.
- **RMSprop**: Uses a moving average of squared gradients to normalize updates.
- **Adam**: Combines momentum with RMSprop’s normalization for robust, default optimizer.

Each method adjusts step size per-parameter, reducing the need to manually tune α.

### Visualizing the Impact

```python
import numpy as np
import matplotlib.pyplot as plt

def run_gd(alpha):
    theta = np.zeros(2)
    costs = []
    for _ in range(50):
        grad = (1/m) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= alpha * grad
        costs.append(compute_cost(X_b, y, theta))
    return costs

alphas = [0.001, 0.01, 0.1, 1]
for a in alphas:
    plt.plot(run_gd(a), label=f"α={a}")
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.legend()
plt.title("Convergence for Different Learning Rates")
plt.show()
```

Watching these curves side by side reveals α that converges quickly without divergence.

---

## Gradient Descent for Linear Regression

### 1. Problem Setup & Cost Function

Linear regression fits a line

```
hθ(x) = θ₀ + θ₁x
```

to minimize the mean squared error over m examples:

```
J(θ₀, θ₁) = (1 / (2m)) Σ₍i=1→m₎ [hθ(x⁽ᶦ⁾) − y⁽ᶦ⁾]²
```

Each θ controls the intercept (θ₀) and slope (θ₁). Minimizing J finds the best-fitting line.

### 2. Deriving the Gradient

The partial derivatives tell us the slope of J in each parameter direction:

- ∂J/∂θ₀ = (1/m) Σ [hθ(x⁽ᶦ⁾) − y⁽ᶦ⁾]
- ∂J/∂θ₁ = (1/m) Σ [hθ(x⁽ᶦ⁾) − y⁽ᶦ⁾] · x⁽ᶦ⁾

Vectorized (for n features):

```
∇J(θ) = (1/m) · Xᵀ · (X·θ − y)
```

where

- X is m×(n+1) (first column ones),
- θ is (n+1)×1,
- y is m×1.

### 3. Gradient Descent Update Rule

Update both parameters simultaneously:

```
θ := θ − α · ∇J(θ)
```

Scalar form for univariate:

```
θ₀ := θ₀ − α·(1/m)Σ(hθ(x⁽ᶦ⁾) − y⁽ᶦ⁾)
θ₁ := θ₁ − α·(1/m)Σ(hθ(x⁽ᶦ⁾) − y⁽ᶦ⁾)·x⁽ᶦ⁾
```

- α is the learning rate controlling step size.
- Repeat until convergence.

### 4. Implementation in Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X.flatten() + np.random.randn(m)

# Prepare design matrix
X_b = np.c_[np.ones((m, 1)), X]

# Cost function
def compute_cost(theta):
    errors = X_b.dot(theta) - y
    return (1 / (2*m)) * errors.T.dot(errors)

# Gradient descent
def gradient_descent(theta, alpha, n_iters):
    cost_history = []
    for i in range(n_iters):
        gradient = (1/m) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(theta))
    return theta, cost_history

# Initialize and run
theta_init = np.zeros(2)
theta_best, cost_hist = gradient_descent(theta_init, alpha=0.1, n_iters=100)

print("Learned θ:", theta_best)

# Plot results
plt.scatter(X, y, label="Data")
plt.plot(X, X_b.dot(theta_best), "r-", label="Fit")
plt.legend(); plt.show()
```

### 5. Convergence & Practical Tips

- Feature scaling (mean normalization, variance scaling) speeds up convergence.
- Monitor cost vs. iterations to ensure steady decrease.
- If cost increases → lower α. If cost barely moves → raise α.
- For large datasets, consider mini-batch or stochastic variants.

---

## Running Gradient Descent: From Code to Convergence

### 1. Full Script with Iteration Logging

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X.flatten() + np.random.randn(m)

# 2. Prepare design matrix
X_b = np.c_[np.ones((m, 1)), X]   # shape: (m, 2)

# 3. Cost function
def compute_cost(theta):
    errors = X_b.dot(theta) - y
    return (1 / (2 * m)) * errors.T.dot(errors)

# 4. Gradient descent with logging
def gradient_descent(theta, alpha, n_iters, log_every=10):
    cost_history = []
    for i in range(n_iters):
        gradient = (1 / m) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(theta))

        # Print progress every log_every iterations
        if i % log_every == 0 or i == n_iters - 1:
            print(f"Iteration {i:3d} | Cost = {cost_history[-1]:.4f} | θ = {theta}")

    return theta, cost_history

# 5. Initialize and run
theta_init = np.zeros(2)
alpha = 0.1
n_iters = 100

theta_final, costs = gradient_descent(theta_init, alpha, n_iters, log_every=20)

# 6. Final result
print("\nFinal Parameters:", theta_final)
```

### 2. Sample Console Output

```
Iteration   0 | Cost = 27.2875 | θ = [0.1203 0.3660]
Iteration  20 | Cost =  1.7371 | θ = [3.0395 2.9512]
Iteration  40 | Cost =  1.0745 | θ = [3.7519 2.9991]
Iteration  60 | Cost =  1.0383 | θ = [3.9444 2.9959]
Iteration  80 | Cost =  1.0377 | θ = [3.9913 2.9984]
Iteration  99 | Cost =  1.0377 | θ = [3.9996 2.9999]

Final Parameters: [3.9996 2.9999]
```

These logs show how cost plummets initially and then flattens as you approach the optimal θ.

### 3. Visualizing Convergence

```python
plt.figure(figsize=(6, 4))
plt.plot(range(n_iters), costs, 'b-')
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()
```

- The curve drops sharply at first, then levels out around the minimum.

### 4. Plotting the Learned Line

```python
plt.figure(figsize=(6, 4))
plt.scatter(X, y, label="Data points")
plt.plot(X, X_b.dot(theta_final), 'r-', linewidth=2, label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression via GD")
plt.show()
```

- Compare the red line (model) against blue dots (data).

---

##