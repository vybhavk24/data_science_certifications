# ML_c2_m4

## Decision trees

Decision trees are non-parametric supervised learning models that split data into homogeneous subsets via recursive binary (or multiway) splits, using criteria like information gain or Gini impurity. 

They’re intuitive, interpretable, and serve both classification and regression tasks. Properly regularized trees avoid overfitting and form the core of powerful ensemble methods like random forests and gradient boosting.

### Prerequisites

- Basic understanding of supervised learning
- Familiarity with classification vs. regression
- Knowledge of probability distributions and entropy
- Python, NumPy, and scikit-learn

### 1. Overview and Intuition

A decision tree mimics human decision-making: you ask a series of yes/no questions (feature thresholds) that funnel data points into leaf nodes with a final prediction. Each split aims to maximize “purity” of the resulting child nodes. Trees partition feature space into axis-aligned regions, making them easy to visualize and interpret.

### 2. Tree Structure and Terminology

- Root node: the topmost decision point
- Internal nodes: test a feature’s value against a threshold
- Branches: outcomes of a test (e.g., “yes” or “no”)
- Leaf nodes: terminal nodes with a class label (classification) or value (regression)
- Depth: number of splits on the longest path from root to a leaf

### 3. Splitting Criteria

### 3.1 Information Gain (Entropy)

```
Entropy(S) = - sum(p_i * log2(p_i)) for each class i in S

InfoGain(S, A) = Entropy(S)
                 - sum((|S_v|/|S|) * Entropy(S_v)) over all values v of feature A
```

### 3.2 Gini Impurity

```
Gini(S) = 1 - sum(p_i^2) for each class i in S

GiniGain(S, A) = Gini(S)
                 - sum((|S_v|/|S|) * Gini(S_v)) over all v
```

### 3.3 Variance Reduction (Regression)

```
Var(S) = (1/|S|) * sum((y_j - mean(S))^2) for j in S

VarReduction(S, A) = Var(S)
                     - sum((|S_v|/|S|) * Var(S_v)) over all v
```

### 4. Tree Building Algorithms

- ID3: uses information gain; handles only categorical features; no pruning
- C4.5: extends ID3 to continuous features, handles missing values, uses gain ratio
- CART: binary splits; uses Gini impurity for classification and variance reduction for regression; supports cost complexity pruning

### 5. Handling Different Data Types

- Categorical features: evaluate splits for each category or group categories via one-hot encoding
- Continuous features: sort unique values and test midpoints between adjacent values
- Missing values:
    - Surrogate splits (CART)
    - Imputation before training
    - Learn separate “missing” branch

### 6. Overfitting and Pruning

### 6.1 Pre-pruning (Early Stopping)

- `max_depth`: limit tree height
- `min_samples_split`: minimum samples to consider a split
- `min_samples_leaf`: minimum samples in a leaf
- `max_leaf_nodes`: cap number of leaves

### 6.2 Post-pruning

- Reduced error pruning: remove branches that don’t hurt validation accuracy
- Cost complexity pruning (CART): prune based on alpha parameter controlling complexity

### 7. Complexity Analysis

- Training time: O(n_samples × n_features × depth) on average
- Prediction time: O(depth) per sample
- Memory: O(n_nodes) where n_nodes grows exponentially with depth

### 8. Interpretability and Feature Importance

- Extract decision rules by tracing paths from root to leaves
- Compute feature importance as total impurity reduction contributed by splits on that feature
- Visualize tree graph to inspect split thresholds and class distributions

### 9. Regression Trees

- Use variance reduction as splitting criterion
- Leaf predicts mean (or median) of target values in that region
- Handle outliers by adjusting splitting metric (e.g., mean absolute error)

### 10. Implementing a Decision Tree from Scratch

1. Compute impurity of current node
2. Loop over features and all candidate thresholds:
    - Split data into left/right subsets
    - Compute weighted impurity of subsets
    - Record best split with maximum impurity reduction
3. Recurse on left and right subsets until stopping criteria met
4. Assign leaf value or class at terminal nodes

### 11. Using scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
clf = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
).fit(X_train, y_train)

# Regression
reg = DecisionTreeRegressor(
    criterion='squared_error',  # variance reduction
    max_depth=None,
    min_samples_split=5
).fit(X_train, y_train)
```

### 12. Hyperparameter Tuning

- Grid search over `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Use cross-validation with stratified splits (classification)
- Monitor training vs. validation error to detect overfitting

### 13. Visualizing the Tree

```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
tree.plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
```

### 14. Pros and Cons

- Pros:
    - Highly interpretable
    - Handles mixed data types
    - No feature scaling needed
    - Captures non-linear relationships
- Cons:
    - Prone to overfitting if not pruned
    - Unstable to small data perturbations
    - Axis-aligned splits may need many nodes for diagonal boundaries

### 15. Use Cases

- Credit scoring (classification)
- Customer churn prediction
- Medical diagnosis decision support
- Pricing or demand forecasting (regression)
- Feature discovery and rule extraction

---

## Decision Tree Learning Process

### Prerequisites

- Basic understanding of supervised learning
- Familiarity with classification vs. regression
- Knowledge of probability distributions and impurity measures
- Python, NumPy, and scikit-learn for code examples

### 1. High-Level Overview

A decision tree is built by recursively partitioning the training data into subsets that are increasingly homogeneous in their target values. The learning process consists of:

1. Choosing the best feature and threshold to split the data
2. Splitting the data into child nodes based on that test
3. Recursing on each child until stopping criteria are met
4. Optionally pruning the fully grown tree to avoid overfitting

### 2. Impurity and Split Evaluation

At each node, evaluate all possible splits and pick the one that maximizes impurity reduction.

### 2.1 Entropy and Information Gain

```
# Entropy of a set S with class proportions p_i
entropy(S) = - sum(p_i * log2(p_i) for each class i)

# Information gain of feature A
info_gain(S, A) = entropy(S)
                   - sum((|S_v|/|S|) * entropy(S_v) for each value v of A)
```

### 2.2 Gini Impurity and Gini Gain

```
# Gini impurity of a set S with class proportions p_i
gini(S) = 1 - sum(p_i**2 for each class i)

# Gini gain of feature A
gini_gain(S, A) = gini(S)
                   - sum((|S_v|/|S|) * gini(S_v) for each value v of A)
```

### 2.3 Variance Reduction (Regression)

```
# Variance of target values in S
variance(S) = sum((y_j - mean(S))**2 for j in S) / |S|

# Variance reduction for feature A
var_reduction(S, A) = variance(S)
                       - sum((|S_v|/|S|) * variance(S_v) for each value v)
```

### 3. Handling Feature Types

1. Continuous features
    - Sort unique values
    - Test splits at midpoints between adjacent values
2. Categorical features
    - For low-cardinality: evaluate every subset split
    - For high-cardinality: group rare categories or use one-hot encoding
3. Missing values
    - Impute before training
    - Learn a “missing” branch
    - Use surrogate splits (CART)

### 4. Recursive Tree-Building Algorithm

1. **Compute impurity** of current node’s data
2. **For each feature**a. For continuous: iterate candidate thresholdsb. For categorical: iterate possible category splitsc. Compute weighted impurity of left/right subsetsd. Track split with maximum impurity reduction
3. **Create an internal node** testing the best feature/threshold
4. **Split data** into left and right subsets
5. **Check stopping criteria** (see section 5)
    - If met, make a leaf node with predicted class/value
    - Else, recurse on each child subset

### 5. Stopping Criteria and Pre-Pruning

Stop recursion when any condition is met:

- Maximum depth reached (`max_depth`)
- Minimum samples to split (`min_samples_split`)
- Minimum samples per leaf (`min_samples_leaf`)
- No improvement in impurity reduction
- All targets are homogeneous in the node

### 6. Pseudocode

```
function build_tree(data, depth):
    if stopping_criteria_met(data, depth):
        return make_leaf(data)

    best_split = find_best_split(data)
    left_data, right_data = split(data, best_split)

    left_child  = build_tree(left_data, depth+1)
    right_child = build_tree(right_data, depth+1)

    return Node(split=best_split,
                left=left_child,
                right=right_child)
```

### 7. Post-Pruning (Cost Complexity Pruning)

1. Grow full tree without depth limit
2. For each subtree T located at node t, compute cost complexity:
    
    ```
    R_alpha(T) = R(T) + alpha * |leaves(T)|
    ```
    
    - R(T): total impurity of leaves
    - alpha: complexity parameter
3. Choose alpha via cross-validation
4. Prune subtrees that reduce R_alpha(T)

### 8. Computational Complexity

- At each node:
    - Continuous features cost O(n_samples × n_features × log n_samples) to sort and evaluate thresholds
    - Categorical splits depend on category count
- Total training time scales roughly O(n_samples × n_features × tree_depth)
- Prediction time per sample is O(tree_depth)

### 9. scikit-learn Implementation

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',          # or 'entropy'
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=0.01,            # cost complexity pruning
    random_state=42
)

clf.fit(X_train, y_train)
```

- Access the pruning path and effective alphas:

```python
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
```

### 10. From-Scratch Python Skeleton

```python
import numpy as np

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=None,
               min_samples_split=2, min_samples_leaf=1):
    # stopping
    if (max_depth and depth >= max_depth) or \
       len(y) < min_samples_split or \
       len(np.unique(y)) == 1:
        leaf_value = np.bincount(y).argmax()
        return DecisionTreeNode(value=leaf_value)

    # find best split
    best_feat, best_thresh = None, None
    best_gain = 0
    current_impurity = gini(y)
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for t in (thresholds[:-1] + thresholds[1:]) / 2:
            left_mask = X[:, feature] <= t
            y_left, y_right = y[left_mask], y[~left_mask]
            if len(y_left) < min_samples_leaf or \
               len(y_right) < min_samples_leaf:
                continue

            gain = information_gain(y, y_left, y_right, current_impurity)
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feature, t

    if best_gain == 0:
        leaf_value = np.bincount(y).argmax()
        return DecisionTreeNode(value=leaf_value)

    # split and recurse
    left_mask = X[:, best_feat] <= best_thresh
    left = build_tree(X[left_mask], y[left_mask],
                      depth+1, max_depth,
                      min_samples_split, min_samples_leaf)
    right = build_tree(X[~left_mask], y[~left_mask],
                       depth+1, max_depth,
                       min_samples_split, min_samples_leaf)

    return DecisionTreeNode(feature=best_feat,
                            threshold=best_thresh,
                            left=left, right=right)
```

### 11. Hyperparameter Tuning

- Grid search over:
    - `max_depth`, `min_samples_split`, `min_samples_leaf`
    - `max_features` (number of features to consider at each split)
    - `ccp_alpha` for pruning
- Use cross-validation to find the best parameter set

### 12. Interpretability and Feature Importance

- Extract decision rules by tracing paths from root to leaves
- Feature importance via total impurity reduction per feature
- Example to get importances in scikit-learn:

```python
importances = clf.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")
```

### 13. Common Pitfalls

- Overfitting when tree is too deep
- Underfitting when pruning or stopping criteria are too strict
- Poor splits on high-cardinality categorical variables
- Unstable model: small data changes yield different trees
- Ignoring feature scaling when using distance-based ensembles

---

## Measuring Purity in Decision Trees

### Pre-requisites

- Basic classification concepts and probability
- Understanding of tree splits and node homogeneity
- Familiarity with Python and NumPy for code snippets

### 1. Purity vs. Impurity

Every node in a decision tree contains a subset of training samples.

- **Purity**: how homogeneous the node’s labels are (all the same class → perfectly pure).
- **Impurity**: how mixed the labels are (multiple classes present → impure).

Decision tree learning seeks splits that **increase purity** or **decrease impurity**.

### 2. Common Impurity Measures

### 2.1 Gini Impurity

Measures the probability of misclassifying a random sample if labels were assigned by the node’s class distribution.

```
Let p[i] = fraction of samples in class i at the node.
gini = 1 - sum(p[i]**2 for i in classes)
```

- Range: 0 (pure) to 1 - 1/K (max impurity for K classes).
- Fast to compute (no logs).

### 2.2 Entropy (Information Entropy)

Quantifies the information content; highest when classes are equally likely.

```
entropy = - sum(p[i] * log2(p[i]) for i in classes if p[i] > 0)
```

- Range: 0 (pure) to log2(K) (uniform distribution over K classes).
- More sensitive to changes in class probabilities near 0 and 1.

### 2.3 Classification Error

Simple measure of misclassification at the node.

```
error = 1 - max(p[i] for i in classes)
```

- Range: 0 (pure) to (1 - 1/K).
- Often used for pruning rather than split selection.

### 3. Impurity Reduction (Information Gain)

To evaluate a split on feature A at threshold t:

1. Compute parent impurity:
    
    ```
    I_parent = impurity(parent_node)
    ```
    
2. Split data into left and right child nodes.
3. Compute weighted child impurity:
    
    ```
    I_children = (n_left / n_total) * impurity(left_node)
               + (n_right / n_total) * impurity(right_node)
    ```
    
4. **Information gain** (for entropy) or **Gini gain**:
    
    ```
    gain = I_parent - I_children
    ```
    

Pick the split that **maximizes gain**.

### 4. Regression: Variance Reduction

For regression trees, measure node impurity by variance of target values:

```
Let y_j be target values at the node, m = number of samples.

mean_y = sum(y_j for j in node) / m
variance = sum((y_j - mean_y)**2 for j in node) / m
```

Variance reduction when splitting:

```
var_reduction = variance(parent)
              - (n_left/n_total)*variance(left)
              - (n_right/n_total)*variance(right)
```

### 5. Comparison of Measures

| Measure | Formula | Range | Use Case |
| --- | --- | --- | --- |
| Gini Impurity | `1 - sum(p[i]**2)` | [0, 1 - 1/K] | Fast splits in CART |
| Entropy | `- sum(p[i]*log2(p[i]))` | [0, log2(K)] | ID3/C4.5 splits, more sensitive |
| Classification Error | `1 - max(p[i])` | [0, 1 - 1/K] | Pruning with minimal gain |
| Variance (regression) | `sum((y - mean_y)**2)/m` | [0, ∞) | Regression trees |

### 6. Example Calculation

```python
import numpy as np

# Sample class counts at a node: [30 ham, 10 spam]
counts = np.array([30, 10])
p = counts / counts.sum()  # [0.75, 0.25]

# Gini impurity
gini = 1 - np.sum(p**2)     # 1 - (0.75^2 + 0.25^2) = 0.375

# Entropy
entropy = -np.sum(p * np.log2(p))  # -(0.75*log2(0.75) + 0.25*log2(0.25)) ≈ 0.811

# Classification error
error = 1 - np.max(p)       # 1 - 0.75 = 0.25
```

### 7. Numerical Stability

- For entropy, guard against `log2(0)` by adding a small epsilon or filtering zero probabilities.
- Always compute `p[i]` with floating-point division.

```python
eps = 1e-12
entropy = -np.sum(p * np.log2(p + eps))
```

### 8. Multi-Class and Multi-Way Splits

- All measures extend naturally when K > 2.
- For categorical features with many categories, consider grouping or one-hot encoding to limit split candidates.

### 9. Computational Considerations

- **Gini vs. Entropy**: Gini is slightly faster (no log) and often yields similar splits.
- When K is large, computing all p[i] can be costly; use sparse counts or early stopping when gain is below threshold.

### 10. Advanced Impurity Measures

- **Tsallis Entropy** (q-parameterized):
    
    ```
    tsallis = (1 - sum(p[i]**q)) / (q - 1)
    ```
    
- **Rényi Entropy**:
    
    ```
    renyi = log(sum(p[i]**q)) / (1 - q)
    ```
    
- Useful for specialized splitting in information-theoretic or quantum-inspired trees.

### 11. Tuning Splits by Impurity Thresholds

- `min_impurity_decrease`: require gain ≥ threshold to split.
- `ccp_alpha` (cost complexity pruning): prunes splits that don’t sufficiently reduce impurity relative to tree size.

---

## Choosing a Split with Information Gain

### Prerequisites

- Basic classification and decision-tree concepts
- Understanding of probability and class distributions
- Familiarity with Python (NumPy)

### 1. Why Information Gain?

At each node of a decision tree, we want the split that **reduces uncertainty** about the target variable the most.

Information gain (IG) measures how much “uncertainty” (entropy) is removed by partitioning the data on a given feature.

### 2. Entropy: Quantifying Uncertainty

Entropy (H) at a node with (K) classes and class probabilities (p[i]) is:

```
H = - sum(p[i] * log2(p[i]) for i in 0..K-1 if p[i] > 0)
```

- (H=0) when the node is perfectly pure (all samples one class).
- (H) is maximal when classes are equally likely.

### 3. Information Gain: Definition

Given a split that divides a parent node into left and right children:

```
IG = H_parent
   - (n_left  / n_total) * H_left
   - (n_right / n_total) * H_right
```

- `H_parent` is entropy before split.
- `H_left`, `H_right` are entropies of child nodes.
- `n_left`, `n_right`, `n_total` are sample counts.

We choose the split with the **largest** IG.

### 4. Step-by-Step Example

Suppose at the parent node you have 10 samples: 6 of class 0, 4 of class 1.

```python
import numpy as np

y = np.array([0]*6 + [1]*4)
ps = np.bincount(y) / len(y)   # [0.6, 0.4]
H_parent = -np.sum(ps * np.log2(ps))  # ≈ 0.971 bits
```

Consider threshold on a feature that splits into:

- Left: [4 zeros, 1 one] → ps_L = [0.8, 0.2], H_left ≈ 0.722
- Right: [2 zeros, 3 ones] → ps_R = [0.4, 0.6], H_right ≈ 0.971

Compute IG:

```
nL, nR, n = 5, 5, 10
IG = 0.971
   - (5/10)*0.722
   - (5/10)*0.971
   = 0.971 - 0.361 - 0.486
   = 0.124 bits
```

### 5. Searching Splits on Continuous Features

1. **Sort** the feature values.
2. **Compute candidate thresholds** as midpoints between consecutive unique values.
3. **For each threshold**:
    - Partition `X_col <= t` and `X_col > t`.
    - Compute child entropies and IG.
4. **Select** the threshold with maximum IG.

```python
def best_threshold(X_col, y):
    sorted_idx = np.argsort(X_col)
    Xs, ys = X_col[sorted_idx], y[sorted_idx]
    thresholds = (Xs[:-1] + Xs[1:]) / 2
    best_ig, best_t = 0, None

    for t in thresholds:
        ig = information_gain(X_col, y, t)
        if ig > best_ig:
            best_ig, best_t = ig, t
    return best_t, best_ig
```

### 6. Splitting on Categorical Features

- **One-vs-Rest**: for each category (c), split ({x=c}) vs. ({x \neq c}).
- **Subset Enumeration**: test all non-empty subsets of categories (expensive if many).
- **Multi-way Split**: each category is its own branch—IG generalizes to (M) children:

```
IG = H_parent
   - sum( (n_j / n_total) * H_j for each child j )
```

### 7. Gain Ratio: Correcting High-Cardinality Bias

Information gain favors features with many values. Gain ratio normalizes by split’s intrinsic information:

```
SplitInfo = - sum((n_j/n_total) * log2(n_j/n_total) for j in children)
GainRatio = IG / SplitInfo   if SplitInfo > 0 else 0
```

Use in C4.5 to select splits when feature cardinality is high.

### 8. Computational Optimizations

- **Sort once per feature** and reuse sorted indices.
- **Cumulative counts**: precompute class counts up to each index to get child counts in O(1).
- **Early stopping**: if `IG <= min_impurity_decrease`, skip further splits.
- **Vectorization**: evaluate all thresholds in parallel with NumPy arrays.

### 9. Numerical Stability

Guard against (\log_2(0)):

```python
eps = 1e-12
ps = np.bincount(y) / len(y)
entropy = -np.sum(ps * np.log2(ps + eps))
```

Always use floating-point division.

### 10. Split-Related Hyperparameters

| Hyperparameter | Description |
| --- | --- |
| `min_samples_split` | Minimum samples required to consider splitting a node |
| `min_samples_leaf` | Minimum samples required to form a leaf node |
| `min_impurity_decrease` | Minimum IG required to make a split |
| `max_features` | Number of features to consider at each split (for random forests) |
| `ccp_alpha` | Complexity parameter for cost-complexity pruning (prune low-IG splits) |

Tuning these controls depth, overfitting, and computational cost.

### 11. Theoretical Connections

- **Mutual Information** between feature (X) and label (Y):[ I(X;Y) = H(Y) - H(Y\mid X) ] IG at a split is an empirical estimate of (I(\text{split};,Y)).
- **Relation to KL Divergence**: IG = KL divergence between joint and product distributions of split and label.
- **Minimum Description Length (MDL)**: IG corresponds to reduction in code length when encoding labels after split.

### 12. Alternative Split Criteria

| Criterion | Formula | Pros | Cons |
| --- | --- | --- | --- |
| Gini Impurity | `1 - sum(p[i]**2)` | Fast, no logs | Similar splits to IG |
| Classification Error | `1 - max(p[i])` | Simple | Less sensitive, poor split quality |
| Entropy (IG) | see above | Theoretically grounded | Slightly slower (log) |

### 13. Pseudocode: Finding Best Split at a Node

```
best_ig = 0
best_feature, best_threshold = None, None

for each feature f in features:
    if f is continuous:
        compute candidate thresholds T_f
    else:
        compute category splits S_f

    for each split candidate s in (T_f or S_f):
        partition data into left, right (or multi-way)
        compute IG_s
        if IG_s > best_ig:
            best_ig, best_feature, best_threshold = IG_s, f, s

return best_feature, best_threshold, best_ig
```

### 14. Practical Considerations

- **High-Cardinality** features: consider hashing or grouping before splitting.
- **Missing Values**: treat “missing” as its own category or use surrogate splits.
- **Overfitting**: deep trees with tiny IG splits generalize poorly—use pruning or hyperparameter tuning.

---

## Decision Tree Learning – Recursive Splitting

### Direct Answer

Recursive splitting is a top-down, greedy procedure that builds a decision tree by repeatedly partitioning the data at each node using the best feature split until stopping criteria are met.

### 1. Overview of Recursive Partitioning

At a high level, decision tree training for classification or regression follows these steps:

1. Start with the root node containing the full training set.
2. Evaluate all possible splits on all features.
3. Select the split that maximizes impurity reduction (information gain, Gini, variance reduction).
4. Partition the data into child nodes based on that split.
5. Recursively apply steps 2–4 to each child node.
6. Stop when a node is “pure” or other stopping criteria are reached, and label it as a leaf.

### 2. Node Impurity and Split Quality

You choose splits by computing an impurity measure before and after splitting:

```
For classification:
  parent_impurity = measure_impurity(y_parent)
  child_impurity = (n_left/n_total)*measure_impurity(y_left)
                 + (n_right/n_total)*measure_impurity(y_right)
  impurity_reduction = parent_impurity - child_impurity

For regression:
  parent_variance = var(y_parent)
  child_variance = (n_left/n_total)*var(y_left)
                 + (n_right/n_total)*var(y_right)
  variance_reduction = parent_variance - child_variance
```

Common measures for classification:

- Gini impurity
- Entropy (information gain)
- Classification error

### 3. Stopping Criteria (Pre-Pruning)

To prevent infinite growth and overfitting, stop splitting a node when any of these holds:

- Node is pure (all labels identical).
- Maximum tree depth `max_depth` is reached.
- Number of samples at node < `min_samples_split`.
- Impurity reduction < `min_impurity_decrease`.
- Number of samples in any child < `min_samples_leaf`.

### 4. Pseudocode for Recursive Splitting

```
function build_tree(X, y, depth=0):
    if stopping_condition(y, depth):
        return create_leaf(y)
    best_feature, best_threshold = find_best_split(X, y)
    X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
    left_subtree  = build_tree(X_left,  y_left,  depth+1)
    right_subtree = build_tree(X_right, y_right, depth+1)
    return create_node(best_feature, best_threshold, left_subtree, right_subtree)
```

### 5. Finding the Best Split

For each feature:

1. If continuous:a. Sort feature values.b. Generate thresholds at midpoints of adjacent values.
2. If categorical:a. Try one-vs-rest splits or enumerate subsets.
3. For each candidate split:a. Partition samples into left/right.b. Compute impurity or variance reduction.
4. Record the feature and threshold with maximum reduction.

### 6. Handling Edge Cases

- When a split produces an empty child, its reduction is zero.
- Use a small epsilon to avoid log(0) in entropy.
- Map rare categories in categorical splits to “other” to limit combinations.
- Treat missing values as their own category or use surrogate splits (find alternative features that mimic the primary split).

### 7. Post-Pruning (Cost-Complexity Pruning)

After building a full tree, you can prune bottom-up to reduce overfitting:

1. Compute for each non-leaf node:
    
    ```
    cost_complexity = impurity(node) + alpha * num_leaves(node)
    ```
    
2. Identify the weakest link (node whose removal yields smallest increase in cost).
3. Remove that subtree, making it a leaf.
4. Repeat for increasing `alpha` to get a pruning path.
5. Choose optimal tree via cross-validation.

### 8. Computational Complexity

- At each node: evaluating one continuous feature takes `O(n log n)` to sort + `O(n)` to scan thresholds.
- With `D` features: `O(D * n log n)` per node.
- Total cost grows with number of nodes; balancing via stopping criteria is key.

### 9. Hyperparameters That Control Splitting

| Hyperparameter | Role |
| --- | --- |
| `max_depth` | Maximum depth of the tree |
| `min_samples_split` | Minimum samples to attempt a split |
| `min_samples_leaf` | Minimum samples to allow at a leaf node |
| `min_impurity_decrease` | Minimum impurity reduction to split |
| `max_features` | Number of features to consider at each split (RF) |

Careful tuning balances bias and variance.

### 10. Multi-Class and Regression Trees

- For **multi-class** classification, impurity measures extend to K classes naturally.
- For **regression**, use variance reduction in place of impurity reduction.
- Leaf prediction: class majority in classification; mean of targets in regression.

### 11. Tree Interpretability and Feature Importance

- Each split defines a decision rule you can trace from root to leaf.
- **Feature importance** can be computed as the total impurity reduction contributed by each feature across all nodes.

```
feature_importance[f]
  = sum(impurity_reduction(node) for node where split_feature == f)
  normalized to sum to 1
```

### 12. Extensions of Recursive Splitting

- **Random Forests**: build many trees on bootstrap samples with random feature subsets at each split.
- **Gradient Boosted Trees**: build trees sequentially to correct errors of prior trees.
- **Oblique Trees**: splits on linear combinations of features (multivariate splits).
- **Conditional Inference Trees**: splits based on statistical tests to avoid bias.

---

## One-Hot Encoding of Categorical Features

### 1. Why and When to Encode Categorical Variables

Machine learning algorithms require numeric inputs. Categorical features—variables that take on a limited set of discrete values—must be converted into numeric form without introducing spurious ordering.

- **Nominal**: no natural order (e.g., color: red, green, blue)
- **Ordinal**: ordered categories (e.g., size: small, medium, large)

Encoding strategies:

- **Label encoding** assigns an integer to each category (can imply false ordering)
- **One-hot encoding** creates a binary column per category (maintains neutrality)

Use one-hot for nominal variables when:

- Your algorithm is linear or distance-based (e.g., logistic regression, k-NN).
- You want to avoid misleading the model with ordinal relationships.

### 2. One-Hot Encoding: Core Concept

For a feature `Color` with categories `['Red', 'Green', 'Blue']`, one-hot encoding transforms each sample into:

```
Color_Red  Color_Green  Color_Blue
    1           0            0
    0           1            0
    0           0            1
```

- Each original category becomes its own binary column.
- Exactly one column is “hot” (1) per sample, all others are 0.

### 3. Manual Implementation in Python

```python
import numpy as np

# Sample data
colors = np.array(['Red', 'Green', 'Blue', 'Green', 'Red'])

# Unique categories
categories = np.unique(colors)            # ['Blue', 'Green', 'Red']
cat_to_index = {c: i for i, c in enumerate(categories)}

# Initialize one-hot matrix
one_hot = np.zeros((len(colors), len(categories)), dtype=int)

# Fill matrix
for row, color in enumerate(colors):
    col = cat_to_index[color]
    one_hot[row, col] = 1

print(one_hot)
```

### 4. Using pandas.get_dummies

```python
import pandas as pd

df = pd.DataFrame({
    'Color': ['Red', 'Green', 'Blue', 'Green', 'Red'],
    'Size':  ['S',   'M',     'L',    'M',     'S']
})

# One-hot encode all categorical columns
df_ohe = pd.get_dummies(df)
print(df_ohe)
```

Options:

- `columns=[...]` to specify which columns to encode
- `drop_first=True` to avoid the dummy-variable trap (multicollinearity)

### 5. Using scikit-learn’s OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Sample DataFrame X, target y
X = pd.DataFrame({
    'Color': ['Red', 'Green', 'Blue'],
    'Size':  ['S',   'M',     'L']
})
y = [0, 1, 0]

# Pipeline: one-hot encode 'Color' and 'Size', then train classifier
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'),
     ['Color', 'Size'])
])

pipeline = Pipeline([
    ('encode', preprocessor),
    ('clf', LogisticRegression())
])

pipeline.fit(X, y)
```

Key parameters:

- `sparse=False` returns a dense array (default is sparse matrix)
- `drop='first'` drops the first category per feature to prevent multicollinearity
- `handle_unknown='ignore'` encodes unseen categories as all zeros

### 6. Handling Unseen Categories & Sparsity

- **handle_unknown='ignore'** ensures your pipeline won’t break if test data contains new categories.
- **sparse=True** (default) stores the one-hot result in a memory-efficient sparse matrix—critical for high-cardinality features.

```python
ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
X_encoded = ohe.fit_transform(X_train)
```

### 7. Dealing with High Cardinality

One-hot encoding can explode feature space if a categorical variable has many unique values.

| Issue | Mitigation |
| --- | --- |
| Curse of dimensionality | Feature hashing, embedding layers |
| Memory & compute bloat | Limit top-k frequent categories; group rest as “Other” |
| Overfitting on rare cats | Target (mean) encoding, smoothing |

### 8. Impact on Different Models

- **Linear Models & Distance Methods**
    
    Require one-hot to avoid false numeric ordering.
    
- **Tree-Based Models**
    
    Can handle label encoding but may still benefit from one-hot when splits on categories yield clearer branches.
    
- **Neural Networks**
    
    Often use embeddings instead of one-hot for very high-cardinality features—but one-hot remains a simple baseline.
    

### 9. Common Pitfalls & Best Practices

- **Dummy-Variable Trap**: including all one-hot columns plus an intercept can cause perfect multicollinearity. Use `drop='first'` or `drop='if_binary'`.
- **Pipeline Integration**: always fit encoders on training data only; apply to validation/test via a pipeline to avoid data leakage.
- **Missing Values**: fill or treat them as a separate category before encoding.
- **Memory Management**: prefer sparse output and incremental learning for large datasets.

### 10. Advanced Topics

- **Feature Hashing**: map categories to a fixed number of columns via a hash function—constant memory at the cost of collisions.
- **Target Encoding**: replace categories with aggregated target statistics (risk of leakage if not cross-validated).
- **Categorical Embedding**: learn low-dimensional dense representations inside a neural network.
- **Interaction Features**: one-hot encode pairwise category interactions to capture joint effects.

---

## Continuous Valued Features

### Pre-requisites

- Familiarity with numeric data types in tabular datasets
- Basic understanding of machine learning models (linear, tree, distance-based, neural nets)
- Python and libraries like NumPy, pandas, and scikit-learn

### 1. Definition and Importance

Continuous valued features take on any real value within a range (e.g., age, temperature, income).

- They capture fine-grained information but often require preprocessing.
- Proper handling ensures models converge faster and make fair comparisons across features.
- Poor scaling or unaddressed outliers can skew results, especially in distance-based methods.

### 2. Scaling and Normalization

### 2.1 Min–Max Scaling

Rescales values to a fixed range, typically [0, 1]:

```
X_scaled = (X - X_min) / (X_max - X_min)
```

### 2.2 Standardization (Z-Score)

Centers to zero mean and unit variance:

```
X_standard = (X - mean(X)) / std(X)
```

### 2.3 Robust Scaling

Uses median and interquartile range to reduce outlier impact:

```
X_robust = (X - median(X)) / (Q3(X) - Q1(X))
```

### 3. Outlier Handling

### 3.1 Clipping or Winsorizing

Limit values to percentiles:

```python
lower, upper = np.percentile(X, [1, 99])
X_clipped = np.clip(X, lower, upper)
```

### 3.2 Log and Power Transforms

Reduce skew by compressing large values:

```python
X_log = np.log1p(X)      # log(1 + X) for nonnegative X
X_sqrt = np.sqrt(X)      # square-root transform
```

### 4. Feature Transformations

### 4.1 Polynomial Features

Add interaction and power terms:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### 4.2 Binning (Discretization)

Convert continuous into categories:

```python
pd.cut(X, bins=5, labels=False)
```

### 4.3 Splines

Model smooth curves over intervals:

```python
from patsy import dmatrix
spline = dmatrix("bs(X, df=4, degree=3)", {"X": X})
```

### 5. Algorithm-Specific Considerations

- **Distance-Based (kNN, SVM)**
    
    Sensitive to feature scale; always scale before training.
    
- **Linear Models**
    
    Coefficients reflect feature importance when standardized.
    
- **Tree-Based (Decision Trees, Random Forests)**
    
    Largely invariant to monotonic transforms; scaling less critical.
    
- **Neural Networks**
    
    Benefit from zero-centered, unit-variance inputs for stable training.
    

### 6. Missing Value Imputation

Continuous features often have gaps. Common strategies:

- Mean or median imputation
- k-NN imputation using similar records
- Model-based imputation (e.g., iterative regression)

Always fit imputer on training data only and apply to validation/test within a pipeline.

### 7. Advanced Engineering Techniques

- **Quantile Transformer**: map distributions to uniform or normal.
- **Principal Component Analysis (PCA)**: reduce correlated continuous features.
- **Autoencoder Embeddings**: compress and denoise continuous inputs in a neural network.

### 8. Interview-Ready Best Practices

1. Always inspect distributions and outliers before modeling.
2. Choose scaling method based on model requirements and outlier presence.
3. Explain trade-offs between binning (interpretability) and polynomial/spline (flexibility).
4. Emphasize pipeline integration to prevent data leakage.

---

## Using Multiple Decision Trees

### Pre-requisites

- Understanding of single decision tree learning (splitting, impurity, pruning)
- Familiarity with overfitting and variance reduction concepts
- Basic Python and scikit-learn experience

### 1. Why Use Multiple Trees?

A single decision tree is easy to overfit, especially on noisy or small datasets. By combining many trees, you can:

- Reduce variance and improve generalization
- Smooth out individual tree quirks
- Achieve higher accuracy with modest tuning

### 2. Bagging and Random Forests

### 2.1 Bagging (Bootstrap Aggregating)

1. Create B bootstrap samples (random sampling with replacement) from the training data.
2. Train one decision tree on each bootstrap sample without pruning.
3. For prediction, average (regression) or majority-vote (classification) across trees.

```
For each tree b in 1..B:
    D_b = bootstrap_sample(D, size=N)
    tree_b = train_tree(D_b)
Predict(x):
    regression → mean(tree_b(x) for b)
    classification → mode(tree_b(x) for b)
```

### 2.2 Random Forest

Random forests add extra randomness to bagging:

- At each split, consider only a random subset of `mtry` features instead of all features.
- This de-correlates trees and further reduces variance.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',      # for classification
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

| Hyperparameter | Role |
| --- | --- |
| `n_estimators` | Number of trees in the forest |
| `max_features` | Features to consider at each split (`sqrt`, `log2`, or int) |
| `max_depth` | Maximum depth per tree (limits overfitting) |
| `min_samples_leaf` | Minimum samples required in a leaf |
| `bootstrap` | Whether to use bootstrap sampling |

### 3. Boosting: Sequential Tree Ensembles

Boosting trains trees sequentially, each one correcting its predecessor’s errors.

### 3.1 AdaBoost

- Initialize sample weights equally.
- For each tree:
    1. Train on weighted samples.
    2. Compute tree error and derive its weight.
    3. Increase weights on misclassified samples.
- Final prediction is weighted vote of all trees.

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
```

### 3.2 Gradient Boosting

- Each new tree fits to the negative gradient (residual) of the loss function.
- Additive model:
    
    ```
    F_0(x) = initial_prediction
    For m = 1 to M:
        residuals = -[dL(y, F_{m-1}(x)) / dF_{m-1}(x)]
        tree_m = train_tree(X, residuals)
        F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x)
    ```
    
- Enables flexible losses (squared error, logistic loss, Huber, etc.).

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
```

### 4. Extra Trees (Extremely Randomized Trees)

- Similar to random forests, but at each split:
    - Choose cut-points at random rather than optimizing split.
    - Often faster and can reduce variance further at the cost of a small bias increase.

```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
et.fit(X_train, y_train)
```

### 5. Stacking and Blending

Combine different models (including tree ensembles) via a meta-learner:

1. Split data into folds.
2. Train base models on training folds, predict on hold-out fold.
3. Use base-model predictions as features for a higher-level model.
4. Final prediction from the meta-model.

```
Level-0: many models → generate out-of-fold predictions
Level-1: train meta-model on those predictions → final output
```

Stacking often yields marginal gains when base models are diverse.

### 6. Hyperparameter Tuning

Common strategies:

- **Grid Search**: exhaustive over a parameter grid
- **Random Search**: sample random combinations
- **Bayesian Optimization**: tools like Optuna or Hyperopt
- **Early Stopping**: track validation loss to halt adding more trees

Example grid for random forest:

```python
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2']
}
```

### 7. Advantages and Trade-Offs

| Method | Variance Reduction | Bias Impact | Computation | Interpretability |
| --- | --- | --- | --- | --- |
| Single Tree | Low | Low bias | Fast train/predict | High |
| Bagging/RF | High | Slight bias ↑ | Moderate | Feature importances only |
| Boosting | Medium–High | Can reduce bias | Slower sequential | Low-medium |
| Extra Trees | Very high | More bias | Fast | Similar to RF |
- **Ensembles** are harder to interpret but deliver superior out-of-sample accuracy.
- **Boosting** can overfit if learning rate or tree depth is too high.
- **Bagging** excels on noisy, high-variance datasets.

### 8. Practical Tips

- Always set `random_state` for reproducibility.
- Use **out-of-bag (OOB)** score in random forests instead of a separate validation split:
    
    ```python
    rf = RandomForestClassifier(
        oob_score=True, n_estimators=200, random_state=42
    )
    print(rf.oob_score_)
    ```
    
- For large datasets, subsample (`subsample` in boosting, `max_samples` in bagging).
- Monitor training and validation performance to detect overfitting.
- Feature scaling is not required for tree-based methods.

---

## Sampling With Replacement

### 1. Basic Definition

Sampling with replacement means each time you draw an element from a dataset, you “put it back” before the next draw.

This allows the same record to appear multiple times in one sample.

It contrasts with sampling without replacement, where each drawn element is removed and cannot recur.

### 2. Sampling With vs Without Replacement

- Without replacement: each sample is unique; once drawn, an element is out of the pool.
- With replacement: each draw is independent, and elements can repeat.
- Replacement preserves the original distribution but changes the effective sample diversity.

### 3. Why Use Sampling With Replacement?

- It provides a simple way to approximate the sampling distribution of a statistic.
- It underpins the bootstrap method for confidence intervals and hypothesis testing.
- It drives ensemble methods like bagging and random forests to reduce variance.

### 4. Bootstrapping: Estimating Uncertainty

The bootstrap procedure:

1. Draw B bootstrap samples of size n (with replacement) from your data.
2. Compute your statistic (mean, median, model accuracy) on each sample.
3. Use the distribution of those B estimates to form confidence intervals or standard errors.

```python
import numpy as np

data = np.array([2.3, 5.1, 3.8, 4.4, 6.0])
n, B = len(data), 1000
boot_means = []
for _ in range(B):
    sample = np.random.choice(data, size=n, replace=True)
    boot_means.append(sample.mean())
ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)
print("95% CI:", ci_lower, ci_upper)
```

### 5. Bagging and Random Forests

Bagging (bootstrap aggregating) builds B models on B bootstrap samples:

```
for b in 1..B:
    D_b = bootstrap_sample(D, n)
    model_b = train_tree(D_b)
predict(x):
    classification → majority vote of model_b(x)
    regression     → average prediction of model_b(x)
```

Random forests add a random feature subset at each split on top of bagging to further de-correlate trees.

### 6. Out-Of-Bag (OOB) Estimation

About 36.8% of original samples are not picked in each bootstrap sample.

Those left-out samples serve as a built-in validation set per tree.

Average OOB error across trees approximates test-set performance without an explicit hold-out.

### 7. Mathematical Foundation

By sampling with replacement, the bootstrap approximates the sampling distribution of a statistic T:

```
T_bootstrap ≈ distribution of T(data)
```

As B → ∞, the empirical bootstrap distribution converges to the true sampling distribution under mild conditions (central limit theorem applies for smooth statistics).

### 8. Practical Implementation in scikit-learn

```python
from sklearn.utils import resample
# X_train, y_train are original arrays
X_boot, y_boot = resample(
    X_train, y_train,
    replace=True,           # sampling with replacement
    n_samples=len(X_train),
    random_state=42
)
```

You can plug `X_boot, y_boot` into any estimator’s `.fit()` method inside a loop to build a bagged ensemble.

### 9. Effects on Bias and Variance

- Bagging primarily **reduces variance**: averaging independent models smooths out fluctuations.
- It may **slightly increase bias** since each tree sees a smaller effective sample diversity.
- Overall generalization error often improves on noisy datasets.

### 10. Pitfalls and Best Practices

- Duplicates in a bootstrap sample reduce the “effective sample size.”
- Always set `random_state` for reproducibility.
- For highly imbalanced data, combine bootstrap with stratification to preserve class ratios.
- Monitor OOB error to detect when adding more trees stops helping.

### 11. Advanced Bootstrap Variants

- **Parametric bootstrap**: draw samples from an estimated parametric distribution instead of the raw data.
- **Block bootstrap**: for time series, sample contiguous blocks to preserve temporal dependence.
- **Bayesian bootstrap**: weight original observations with Dirichlet-distributed weights.

---

## Random Forest Algorithm

### Pre-requisites

- Understanding of decision trees (splitting, impurity, pruning)
- Basics of bias–variance trade-off and ensemble methods
- Python with scikit-learn for code illustrations

### 1. What Is a Random Forest?

A random forest is an ensemble of decision trees trained on bootstrap samples with random feature selection at each split. Predictions from all trees are aggregated—majority vote for classification or mean for regression—yielding high accuracy and robustness.

### 2. Why Use Random Forests?

- Reduces variance by averaging many uncorrelated trees
- Retains low bias of deep individual trees
- Handles large feature sets, missing values, and mixed data types
- Provides built-in estimates of feature importance and generalization error

### 3. Core Concepts

1. **Bootstrap Sampling**
    
    Draw N samples with replacement from the training set for each tree.
    
2. **Feature Bagging**
    
    At each node split, consider a random subset of `m` features (instead of all) to decorrelate trees.
    
3. **Aggregation**
    - Classification: majority vote of tree predictions
    - Regression: average of tree predictions

### 4. Algorithm Steps

```
Given training data D of size N and feature dimension D:
For b = 1 to B:
    1. Draw bootstrap sample D_b of size N (with replacement).
    2. Train a decision tree T_b on D_b:
       a. At each node, randomly select m features from D features.
       b. Find the best split on those m features (by Gini or entropy).
       c. Recurse until stopping criteria (max_depth, min_samples_leaf).
End For

To predict on new sample x:
    - Classification: return mode(T_b(x) for b=1..B)
    - Regression:  return mean(T_b(x) for b=1..B)
```

### 5. Splitting Criteria in Base Trees

Decision trees in the forest can use:

```
# Gini impurity for a node with class probabilities p[i]
gini = 1 - sum(p[i]**2 for i in classes)
```

```
# Entropy for a node with class probabilities p[i]
entropy = - sum(p[i] * log2(p[i]) for i in classes if p[i] > 0)
```

Regression trees use variance reduction:

```
# Variance at a node with target values y_j
mean_y   = sum(y_j for j in node) / m
variance = sum((y_j - mean_y)**2 for j in node) / m
```

### 6. Out-of-Bag (OOB) Estimation

- About 36.8% of samples are left out of each bootstrap.
- Use those OOB samples to estimate generalization error without a separate validation set.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)
print("OOB Accuracy:", rf.oob_score_)
```

### 7. Key Hyperparameters

| Parameter | Description |
| --- | --- |
| `n_estimators` | Number of trees in the forest |
| `max_features` | Number of features to consider at each split (`sqrt`, `log2`, int) |
| `max_depth` | Maximum depth per tree |
| `min_samples_split` | Minimum samples required to split an internal node |
| `min_samples_leaf` | Minimum samples required at a leaf node |
| `bootstrap` | Whether to sample with replacement (`True`) |
| `oob_score` | Whether to use OOB samples to estimate score |

Tuning these balances bias, variance, and computational cost.

### 8. Computational Complexity

- **Training time**: O(B · N · m · log N)
- **Prediction time**: O(B · tree_depth) per sample
- **Memory footprint**: O(B · average_tree_size)

### 9. Feature Importance

1. **Mean Decrease in Impurity**
    
    Sum of impurity reductions from splits on each feature, averaged over all trees.
    
    ```
    importance[f] = sum_over_trees(
                      sum_over_nodes_split_on_f(
                        (n_node / N) * (impurity_parent - impurity_children)
                      )
                    )
    ```
    
2. **Permutation Importance**
    
    Measure drop in OOB or validation score when a feature’s values are randomly permuted.
    
    ```python
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=10, random_state=0, n_jobs=-1
    )
    print(result.importances_mean)
    ```
    

### 10. Handling Data Types & Missing Values

- Numerical and categorical features: trees handle both without scaling.
- High-cardinality categories: consider grouping rare categories or using one-hot encoding.
- Missing values: impute before training or treat missing as its own category.

### 11. Variations & Extensions

- **Extra Trees**: choose random split thresholds, not optimal ones, for faster and more randomized forests.
- **Weighted Random Forests**: apply sample weights for class imbalance.
- **Rotation Forest**: apply PCA to feature subsets before each tree.
- **Quantile Regression Forests**: estimate conditional quantiles for uncertainty quantification.

### 12. Theoretical Properties

- **Variance Reduction**: averaging B uncorrelated models reduces variance roughly by 1/B.
- **Bias**: similar to individual deep trees (low bias).
- **Consistency**: under certain conditions (B→∞, trees deep, feature randomness), random forests are consistent estimators.

### 13. Practical Code Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Split data
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_tr, y_tr)
print("Test Accuracy:", rf.score(X_te, y_te))

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt','log2'],
    'max_depth': [None, 10, 20]
}
grid = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_tr, y_tr)
print("Best Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)
```

### 14. Pros and Cons

**Pros**

- High accuracy and robustness to overfitting
- Handles mixed data types and missing values
- Built-in feature importance and OOB error
- Parallelizable training and prediction

**Cons**

- Less interpretable than single trees
- Higher memory and compute requirements
- Can struggle with extremely high-cardinality categorical data

### 15. Applications

- Fraud detection, credit scoring, medical diagnosis
- Customer churn prediction, recommendation systems
- Environmental modeling, remote sensing
- Genomics, anomaly detection, and more

---

## XGBoost

### 1. Direct Answer

XGBoost (eXtreme Gradient Boosting) is an optimized implementation of gradient boosting that uses second-order gradients, regularization, and system-level optimizations (parallelism, cache awareness) to deliver fast, accurate, and scalable tree ensembles for classification and regression.

### 2. Prerequisites

- Understanding of decision trees and recursive splits
- Familiarity with gradient boosting concept
- Python experience with NumPy and pandas
- Basic comfort with optimizing loss functions

### 3. Gradient Boosting Refresher

1. **Ensemble** of weak learners (trees) added sequentially.
2. Each new tree fits the negative gradient (residual) of the loss function from prior ensemble.
3. Final prediction is sum of predictions from all trees.

```
F_0(x) = init_prediction
For m in 1..M:
    r_i = -[dL(y_i, F_{m-1}(x_i)) / dF_{m-1}(x_i)]
    Fit tree h_m to residuals r_i
    F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
```

### 4. XGBoost Core Innovations

- Uses both **first** and **second** derivatives of the loss (gradient and Hessian).
- Adds **regularization** on leaf weights to control complexity.
- Implements **approximate split finding** and **shrinkage** for efficiency.
- Supports **sparse input**, **missing-value handling**, and **parallel tree construction**.

### 5. Regularized Objective Function

XGBoost minimizes this at each iteration:

```
Obj = sum(L(y_i, ŷ_i))
    + sum(Ω(tree_t)) for t=1..T

Ω(tree) = γ * #leaves + 0.5 * λ * sum(w_j^2) for each leaf weight w_j
```

- `L` is the chosen loss (e.g., logistic, squared error).
- `γ` penalizes number of leaves.
- `λ` is L2 regularization on leaf weights.

### 6. Tree Structure and Leaf Score

When building a tree, the gain from splitting node `j` into `left` and `right` is:

```
gain = 0.5 * [G_L^2 / (H_L + λ) + G_R^2 / (H_R + λ) - G_j^2 / (H_j + λ)]
       - γ
```

- `G_*` = sum of gradients in the node(s)
- `H_*` = sum of Hessians in the node(s)
- We choose split with **maximum gain** ≥ 0.

### 7. Approximate Split Finding

For large data, XGBoost:

1. **Bucketizes** continuous features into bins.
2. **Histograms** gradients/Hessians per bin.
3. **Scans** histogram to compute split gain.

This reduces memory bandwidth and allows parallelization.

### 8. Shrinkage & Column Subsampling

- **Shrinkage**: scale leaf contributions by `learning_rate` (η) to slow learning.
- **Column Subsampling**: randomly select a fraction of features per tree or per split to reduce correlation.

### 9. Handling Missing Values

XGBoost automatically learns a **default direction** (left/right) for missing values per split, based on which choice yields higher gain.

### 10. System Optimizations

- **Multi-threading** on features and data partitions
- **Cache-aware** block structure for feature columns
- **Out-of-core** computation for datasets larger than memory
- **GPU support** for histogram construction and split finding

### 11. Python API and Code Examples

### 11.1 Basic Classification

```python
import xgboost as xgb

# Prepare DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# Parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'gamma': 0.1
}

# Training with evaluation
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(params, dtrain, num_boost_round=200,
                evals=watchlist, early_stopping_rounds=20)
```

### 11.2 scikit-learn Wrapper

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.1,
    use_label_encoder=False,
    random_state=0
)
model.fit(
    X_tr, y_tr,
    early_stopping_rounds=20,
    eval_set=[(X_te, y_te)],
    verbose=False
)
```

### 12. Parameter Categories

| Category | Parameters |
| --- | --- |
| Booster | `eta`, `gamma`, `max_depth`, `min_child_weight` |
| Regularization | `lambda`, `alpha` |
| Sampling | `subsample`, `colsample_bytree`, `colsample_bylevel` |
| Learning | `learning_rate`, `n_estimators`, `early_stopping_rounds` |
| System | `nthread`, `tree_method`, `gpu_id` |

### 13. Hyperparameter Tuning

1. **Tree parameters**: `max_depth`, `min_child_weight`
2. **Sampling parameters**: `subsample`, `colsample_bytree`
3. **Regularization**: `gamma`, `lambda`, `alpha`
4. **Learning rate schedule**: lower `eta` and increase `n_estimators`
5. **Early stopping** on validation set to prevent overfitting

Use `GridSearchCV` or `BayesianOptimization` focusing on one category at a time.

### 14. Cross-Validation & Early Stopping

```python
cv_results = xgb.cv(
    params, dtrain,
    num_boost_round=500,
    nfold=5,
    metrics='auc',
    early_stopping_rounds=20,
    as_pandas=True,
    seed=0
)
best_rounds = len(cv_results)
```

### 15. Feature Importance & Interpretation

- **Weight**: number of times a feature is used in splits
- **Gain**: average gain when a feature is used
- **Cover**: average number of samples affected by splits on the feature

```python
import matplotlib.pyplot as plt

xgb.plot_importance(bst, importance_type='gain')
plt.show()
```

For fine-grained insight, use **SHAP values** via the `shap` library.

### 16. Advanced Features

- **Monotone Constraints**: enforce feature’s prediction direction (`monotone_constraints` parameter).
- **DART Booster**: drop-out trees to reduce overfitting (`booster='dart'`).
- **Linear Booster**: linear models as base learners (`booster='gblinear'`).
- **Quantile Regression**: custom objective for quantile loss.

### 17. GPU & Distributed Training

- Set `tree_method='gpu_hist'` for GPU acceleration.
- Use `xgb.dask` module for distributed training on Dask clusters.
- Supports multi-GPU and MPI-based training for massive datasets.

### 18. Model Persistence & Integration

```python
# Save and load
bst.save_model('xgb_model.json')
bst2 = xgb.Booster()
bst2.load_model('xgb_model.json')

# Predicting
dnew = xgb.DMatrix(X_new)
y_pred = bst2.predict(dnew)
```

You can export to **ONNX** or **PMML** for integration in other environments.

### 19. Common Pitfalls & Best Practices

- Don’t set `learning_rate` too high—use early stopping.
- Beware of data leakage in cross-validation and early stopping.
- For sparse or large data, use `tree_method='hist'` or `'gpu_hist'`.
- Monitor training vs. validation metrics to detect overfitting.
- Always fix random seeds for reproducibility.

---

## When to Use Decision Trees

### Direct Answer

Use decision trees when you need an interpretable, non-parametric model that handles mixed data types, captures non-linear patterns and interactions automatically, and works without extensive feature preprocessing.

### 1. Data Characteristics & Problem Types

- **Mixed Feature Types**
    
    Works on categorical, ordinal, and continuous features without scaling or encoding (beyond simple label or one-hot).
    
- **Non-linear Relationships**
    
    Automatically partitions feature space into axis-aligned regions, capturing complex decision boundaries.
    
- **Interactions**
    
    Detects and models feature interactions without manual engineering (e.g., splits on `Age` then on `Income` capture joint effects).
    
- **Missing Values & Outliers**
    
    Robust to missing data (can learn “missing” branches) and insensitive to extreme outliers, unlike distance-based methods.
    

### 2. Interpretability & Rule Extraction

- **Human-Readable Rules**
    
    Each path from root to leaf is a clear “if-then” rule you can explain to stakeholders.
    
- **Feature Importance**
    
    Summarizes which variables drive decisions via impurity-reduction totals.
    
- **Debugging & Auditing**
    
    Easy to inspect splits and diagnose model behavior on specific subgroups.
    

### 3. When Not to Use Standalone Trees

- **High Variance / Overfitting**
    
    Single trees can overfit noisy data. If generalization is critical, consider ensembles (random forests, gradient boosting).
    
- **Smooth Continuous Boundaries**
    
    Axis-aligned splits produce step functions; if your target varies smoothly, linear or kernel methods may fit better.
    
- **Very High-Dimensional Sparse Data**
    
    Text or one-hot encoded features with thousands of columns may lead to overly complex trees; use regularized linear models or embeddings instead.
    

### 4. Common Use Cases

- **Risk Scoring & Credit Models**
    
    Easily justify “decline” or “approve” decisions with clear branching rules.
    
- **Medical Diagnosis & Decision Support**
    
    Clinicians prefer transparent decision paths over opaque black-box outputs.
    
- **Customer Segmentation & Churn Prediction**
    
    Detect key feature combinations driving segment membership.
    
- **Feature Discovery & Data Exploration**
    
    Quick baseline to uncover important splits and interactions before deeper modeling.
    

### 5. Algorithm & Resource Constraints

- **Fast Training & Inference**
    
    Efficient on moderate-sized datasets (up to hundreds of thousands of samples) without GPUs.
    
- **On-Device or Real-Time Systems**
    
    Shallow trees can run in milliseconds on low-power devices.
    
- **Prototype & Baseline Models**
    
    Ideal for a quick proof of concept before investing in complex pipelines.
    

### 6. Integration in Ensembles

Even if a standalone tree isn’t sufficient, it excels as a base learner:

```
Bagging (Random Forest):
  - Reduces variance by training many trees on bootstrap samples.

Boosting (XGBoost, LightGBM):
  - Sequentially fits trees to residuals for low bias and controlled variance.
```

Use tree ensembles when you need higher accuracy while retaining some interpretability via feature importance or Tree SHAP.

### 7. Tuning & Regularization

When using decision trees, control overfitting by:

```
max_depth            # limit tree height
min_samples_split    # minimum samples to split an internal node
min_samples_leaf     # minimum samples per leaf
min_impurity_decrease# require a minimum information gain
ccp_alpha            # cost-complexity pruning parameter
```

Tuned correctly, a single tree can generalize well on clean, well-separable data.

### 8. Summary Checklist

Use decision trees when you need:

- Immediate interpretability and rule extraction
- Handling of mixed and missing data without preprocessing
- Automatic detection of nonlinear patterns and feature interactions
- Fast prototyping or on-device inference

Avoid standalone trees when:

- Your dataset is extremely noisy or high-dimensional without clear axis-aligned structure
- You require the lowest possible test error—prefer ensembles
- You need truly smooth predictions over continuous space

---