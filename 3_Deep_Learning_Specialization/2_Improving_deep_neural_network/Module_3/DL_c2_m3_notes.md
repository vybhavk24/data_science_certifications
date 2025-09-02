# DL_c2_m3

## Hyperparameter Tuning Process

### 1. Direct Definition

Hyperparameter tuning is the systematic process of selecting the best set of hyperparameters (learning rate, batch size, regularization strength, network architecture, etc.) to optimize a model’s performance on a validation dataset.

### 2. Concept Intuition

- Hyperparameters sit “above” the learning algorithm—they’re not learned by gradient descent.
- They control model capacity, optimization behavior, and regularization.
- Tuning is like adjusting the knobs on a sound mixer: small tweaks produce big changes in the output quality.

Why it matters

- A well-tuned model generalizes better on unseen data.
- Poor choices can lead to slow convergence, overfitting, or underfitting.

### 3. Mathematical Breakdown

At training time, we minimize the regularized cost:

```python
J(θ; λ) = (1/m) * Σ_{i=1}^m L( ŷ^{(i)}, y^{(i)} )
        + (λ/(2m)) * Σ ||W^{[l]}||²
```

Where

- θ = ${W^{[1]}, b^{[1]}, …, W^{[L]}, b^{[L]}}$
- λ is the regularization hyperparameter
- L(·) is the loss (e.g., cross-entropy)
- m is number of training examples

Key hyperparameters:

- α (learning rate) appears in update:
    
    ```python
    W := W - α * dW
    b := b - α * db
    ```
    
- Batch size B influences noisy vs. smooth gradient estimates.
- Number of epochs E determines how many passes through the data.

### 4. Code & Practical Application

### 4.1 Grid Search (NumPy-style)

```python
import numpy as np

def train_model(alpha, lam, num_epochs=100):
    # initialize parameters W, b
    # run gradient descent for num_epochs using alpha, lam
    # return validation_accuracy
    pass

# Define grid
learning_rates = [0.001, 0.01, 0.1]
lambdas        = [0.0, 0.1, 1.0]

best_score = 0
best_params = {}

for α in learning_rates:
    for λ in lambdas:
        score = train_model(α, λ)
        if score > best_score:
            best_score = score
            best_params = {'alpha': α, 'lambda': λ}

print("Best hyperparameters:", best_params, "with val acc:", best_score)
```

### 4.2 Random Search (TensorFlow Keras example)

```python
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import random

def build_and_train(hp):
    model = tf.keras.Sequential([
        layers.Dense(hp['units'], activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    opt = optimizers.Adam(learning_rate=hp['lr'])
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    history = model.fit(X_train, y_train,
                        epochs=hp['epochs'],
                        batch_size=hp['batch_size'],
                        validation_data=(X_val, y_val),
                        verbose=0)
    return max(history.history['val_accuracy'])

param_dist = {
    'lr': [1e-4, 1e-3, 1e-2],
    'units': [16, 32, 64],
    'batch_size': [32, 64, 128],
    'epochs': [20, 50]
}

best = {'score': 0}
for _ in range(20):  # 20 random trials
    hp = {k: random.choice(v) for k, v in param_dist.items()}
    val_acc = build_and_train(hp)
    if val_acc > best['score']:
        best = {'params': hp, 'score': val_acc}

print("Random Search Best:", best)
```

### 5. Visualization / Geometry

ASCII depiction of hyperparameter landscape:

```
       Val Accuracy
    1.0 |        .           .
        |      .   .   .
        |   .           .      ← peaks are good hyperparam combos
      0 |________________________
         LR low        LR high →
```

- Each dot represents one (learning rate, λ) trial
- The surface is non-convex. We explore grids or random points to find peaks.

### 6. Common Pitfalls & Tips

- Searching too coarsely: you may miss the sweet spot. Start coarse, then refine around the best region.
- Over-tuning on test set: always reserve a final hold-out test set. Use validation only for tuning.
- Ignoring interactions: hyperparameters often interact (e.g., batch size vs. learning rate).
- Early stopping as a hyperparameter: saves compute and avoids overfitting.

### 7. Interview-Ready Insights

- Random search often outperforms grid search when only a few hyperparameters strongly influence performance.
- Learning rate usually has the biggest impact—tune it first.
- Bayesian optimization (e.g., Gaussian Processes) can find optima in fewer trials than brute force.
- Explain how cross-validation folds can be used in small‐data scenarios.

### 8. Practice Exercises

### Exercise 1: Grid Search on Synthetic Data

1. Generate a 2-class dataset with `sklearn.datasets.make_moons`.
2. Implement logistic regression with L2 regularization in NumPy.
3. Tune `alpha` ∈ [0.001, 0.005, 0.01, 0.05, 0.1] and `lambda` ∈ [0, 0.01, 0.1, 1].
4. Plot validation accuracy as a heatmap.

Hint: Use `np.meshgrid` for heatmap axes.

### Exercise 2: Random Search in PyTorch

1. Build a two-layer network in PyTorch for MNIST.
2. Define random search over learning rate, hidden units, batch size (at least 3 choices each).
3. Track best validation accuracy across 30 trials.
4. Visualize how accuracy varies with learning rate.

Hint: Use `torch.utils.data.DataLoader` and wrap trials in a for-loop.

---

## Using Appropriate Scale to Pick Hyperparameter

### 1. Direct Definition

Choosing an appropriate scale means defining the range and sampling distribution (linear, logarithmic, or power-law) for each hyperparameter so your search efficiently explores values that matter.

### 2. Concept Intuition

Hyperparameters such as learning rate, regularization λ, or dropout rate often span several orders of magnitude.

A linear sweep (e.g., 0.0001 to 0.1) places most samples near the high end, missing small but critical values.

Using a log scale spreads sampling evenly in log-space, so you cover, say, 1e-6, 1e-5, …, 1e-1 with equal density.

### 3. Mathematical Breakdown

Linear sampling

```
α = low + r * (high − low)
where r ∼ Uniform(0,1)
```

Log-uniform (log scale) sampling

```
log_low  = log(low)
log_high = log(high)
log_α    = log_low + r * (log_high − log_low)
α        = exp(log_α)
where r ∼ Uniform(0,1)
```

Power-law sampling (for exponent p)

```
α = ((high^p − low^p) * r + low^p)^(1/p)
```

### 4. Code & Practical Application

### 4.1 NumPy: log-uniform sampling

```python
import numpy as np

def sample_log_uniform(low, high, size=1):
    log_low, log_high = np.log(low), np.log(high)
    return np.exp(np.random.uniform(log_low, log_high, size))

# Example: 5 samples between 1e-6 and 1e-1
samples = sample_log_uniform(1e-6, 1e-1, size=5)
print("Learning rates:", samples)
```

### 4.2 TensorFlow Keras Tuner example

```python
from kerastuner import HyperParameters

hp = HyperParameters()
hp.Float("learning_rate",
         min_value=1e-6, max_value=1e-1,
         sampling="log")  # log-uniform sampling

# Later, pass hp into your tuner
```

### 5. Visualization / Geometry

Imagine plotting r on [0,1]:

Uniform sampling

```
r: 0.0 ──•─────────•─────────•────────•───────── 1.0
α: low──────────mid──────────high
```

Log-uniform sampling

```
r: 0.0 ─•───•──•───────────•───────── 1.0
α: 1e-6 1e-5 1e-4     1e-2    1e-1
```

On a log-scale x-axis, points are evenly spaced, ensuring you probe each order of magnitude.

### 6. Common Pitfalls & Tips

- Sampling linearly for hyperparams that vary exponentially (like learning rate) often wastes trials near the top end.
- Forgetting to transform back: always exponentiate after sampling log space.
- Overly narrow range: start broad (e.g., 1e-6 to 1e-1) then zoom in around promising values.
- Mixing scales: some hyperparameters (dropout rate) belong in [0,1]—use linear sampling there.

### 7. Interview-Ready Insights

- Explain why log-scale picks give better coverage when performance changes rapidly across orders of magnitude.
- Contrast grid search in log space versus linear: show how grid with log spacing reduces total combinations.
- Mention alternative distributions like quniform, qloguniform in libraries (e.g., Hyperopt).
- Discuss adaptive approaches (Bayesian optimization) that can learn an effective scale from early trials.

### 8. Practice Exercises

**Exercise 1: Log vs Linear Sampling Visualization**

1. Write code to generate 1000 samples of α ∈ [1e-4, 1] using linear and log-uniform.
2. Plot histograms of both sets on a log-scale x-axis.
3. Observe where points concentrate.

**Exercise 2: Hyperparameter Grid Refinement**

1. On a toy CNN, tune learning rate in two phases:
    - Phase 1: coarse log grid: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].
    - Phase 2: refine around best from Phase 1 with tighter log bounds.
2. Track training and validation accuracy.
3. Report how much faster you reach good performance with this two-stage strategy.

---

## Hyperparameter Tuning in Practice: Pandas vs. Caviar

### 1. Direct Definition

Hyperparameter tuning in practice involves choosing, organizing, and periodically reevaluating hyperparameters to maximize model performance under your compute constraints. 

Two major “schools of thought” dominate this process: the **Pandas** (manual, resource-light) approach and the **Caviar** (automated, compute-heavy) approach.

### 2. The Two Schools: Pandas vs. Caviar

### Pandas Approach (Manual Babysitting)

- Use when you have a large dataset but limited CPUs/GPUs.
- Train one model at a time, watch its learning curves, and nudge hyperparameters day by day.
- Example workflow:
    1. Day 0: initialize parameters randomly; start training.
    2. Day 1: observe cost J or error decrease; increase learning rate slightly.
    3. Day 2: check if new rate helps; adjust momentum or regularization next.
    4. Repeat—each day you manually tweak and observe.

### Caviar Approach (Parallel Search)

- Use when you have abundant compute (many GPUs/CPUs).
- Launch multiple training jobs in parallel, each with different hyperparameter settings.
- Automatically compare final validation metrics to pick the winner—no babysitting needed.
- Enables broad exploration (grid, random, Bayesian) at scale without daily manual checks.

### 3. Organizing Your Search Process

- Cross-domain intuition can guide initial ranges, but always retest periodically—even “good” settings go stale as data or infrastructure evolve.
- Split data into training, development (validation), and test sets. Tune exclusively on the dev set to avoid overfitting your hyperparameter choices.
- Start broad (e.g., learning rates from 1e-6 to 1e-1 on a log scale), then zoom in around top performers.

### 4. Code & Practical Application

### 4.1 Manual “Pandas” Loop in NumPy

```python
def train_and_eval(lr, momentum, num_epochs=5):
    # initialize W, b
    # run gradient descent with given lr, momentum
    # return val_accuracy
    pass

# Manual schedule of tweaks
settings = [
    {'lr': 0.001, 'momentum': 0.9},
    {'lr': 0.005, 'momentum': 0.9},
    {'lr': 0.005, 'momentum': 0.95},
]
best = {'acc': 0}
for cfg in settings:
    acc = train_and_eval(cfg['lr'], cfg['momentum'])
    if acc > best['acc']:
        best.update({'acc': acc, **cfg})
print("Best (manual):", best)
```

### 4.2 Automated “Caviar” Random Search with Keras Tuner

```python
from kerastuner import RandomSearch
import tensorflow as tf

def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=hp.Int('units', 32, 128, step=32),
            activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='caviar_search'
)

tuner.search(X_train, y_train,
             epochs=10,
             validation_data=(X_val, y_val))
print("Best hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)
```

### 5. Common Pitfalls & Tips

- **Stale Settings**: Reevaluate hyperparameters every few months; data drift or infrastructure changes can shift optimal values.
- **Overfitting the Dev Set**: Never peek at test performance until final evaluation.
- **Interaction Effects**: Remember that one hyperparameter’s effect can depend heavily on others—exhaustively tuning one at a time can mislead.
- **Compute Budget**: If you’re “Pandas,” plan time for manual checks; if you’re “Caviar,” watch out for infrastructure limits and job queuing delays.

### 6. Interview-Ready Insights

- Contrast manual (Pandas) vs. automated (Caviar) strategies by compute availability and project scale.
- Explain why a hybrid two-stage strategy (broad automated search → manual fine-tuning) often works best.
- Discuss how modern tools (Ray Tune, Optuna) let you dynamically switch between these approaches.
- Be prepared to justify your choice of strategy given a scenario’s data size, timeline, and hardware budget.

### 7. Practice Exercises

1. **Manual Tuning Simulation**
    - Use a small fully-connected network on MNIST.
    - Write a loop that runs 5 manual hyperparameter tweaks (learning rate & dropout), each time saving and plotting validation loss curves.
2. **Parallel Search with Limited GPUs**
    - Use `torch.multiprocessing` or `concurrent.futures` to launch 8 PyTorch training jobs in parallel, sampling learning rate from a log-uniform distribution and batch size from [32, 64, 128].
    - Aggregate the validation accuracies and identify the best configuration.
3. **Staleness Check**
    - Given a pretrained model and evolving data (e.g., adding noise or new classes), retrain at 0, 1, and 3 months. Record how previously optimal hyperparameters perform over time. Plot performance decay to justify periodic retuning.

---

## Normalizing Activations in a Network (Batch Normalization)

### 1. Direct Definition

Batch normalization (BatchNorm) is a technique that normalizes the inputs (activations) of each layer to have zero mean and unit variance within each mini-batch during training. It adds two learnable parameters per activation to allow the network to restore representation power.

### 2. Concept Intuition

BatchNorm addresses the problem of “internal covariate shift”—the idea that as earlier layers update, the distribution of inputs to later layers keeps changing, forcing them to constantly readapt. By normalizing activations:

- Layers see stable, standardized inputs
- Learning rates can be higher, speeding up convergence
- The network becomes less sensitive to initialization

You can think of it as keeping each layer’s activation histogram centered and spread consistently, like tuning the dials on each stage so downstream layers always receive familiar signals.

### 3. Mathematical Breakdown

For a single layer input vector x across a mini-batch of size m:

1. Compute batch mean and variance
    
    ```python
    mu     = (1/m) * sum(x_i for i in batch)
    var    = (1/m) * sum((x_i - mu)**2 for i in batch)
    ```
    
2. Normalize
    
    ```python
    x_hat  = (x - mu) / sqrt(var + epsilon)
    ```
    
3. Scale and shift (learnable parameters γ and β)
    
    ```python
    y      = gamma * x_hat + beta
    ```
    
    - ε (epsilon) stabilizes division
    - γ (scale) and β (shift) let the layer recover any needed distribution

Full forward pass per activation dimension:

```python
mu     = np.mean(x, axis=0)
var    = np.var(x, axis=0)
x_hat  = (x - mu) / np.sqrt(var + eps)
out    = gamma * x_hat + beta
```

Backward pass involves computing gradients through normalization steps—tracking dL/dout → dL/dx_hat → dL/dvar and dL/dmu. Detailed derivations ensure stable gradient flow.

### 4. Code & Practical Application

### 4.1 NumPy Implementation (Forward + Backward)

```python
import numpy as np

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps, self.mom = eps, momentum
        self.gamma = np.ones(dim)
        self.beta  = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var  = np.zeros(dim)

    def forward(self, x, train=True):
        if train:
            mu  = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = self.mom*self.running_mean + (1-self.mom)*mu
            self.running_var  = self.mom*self.running_var  + (1-self.mom)*var
        else:
            mu, var = self.running_mean, self.running_var

        x_hat = (x - mu) / np.sqrt(var + self.eps)
        out   = self.gamma * x_hat + self.beta

        # Cache for backward
        self.cache = (x, x_hat, mu, var)
        return out

    def backward(self, dout):
        x, x_hat, mu, var = self.cache
        m = x.shape[0]

        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta  = np.sum(dout, axis=0)

        dx_hat = dout * self.gamma
        dvar   = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        dmu    = np.sum(dx_hat * -1/np.sqrt(var + self.eps), axis=0) \
               + dvar * np.mean(-2*(x - mu), axis=0)

        dx = dx_hat/np.sqrt(var + self.eps) + dvar*2*(x-mu)/m + dmu/m

        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta  -= learning_rate * dbeta

        return dx
```

### 4.2 TensorFlow / Keras Usage

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(128, activation=None, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
```

### 4.3 PyTorch Usage

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.fc2(x)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 5. Visualization / Geometry

Imagine plotting the activation distribution before and after BatchNorm:

```
Before BN: wide, shifting bell curves per batch
                ┌─┐       ┌───┐
   activation   │ │   ┌─┐ │ │ └─┐
                └─┘ └─┘ └─┘

After BN: centered at zero, unit variance
                ┌───────────┐
 activation     │    ┌───┐  │
  distribution  │   │   │ └┼─┐
                └───┴───┴───┴─┘
```

On the loss surface, BatchNorm tends to smooth and convexify directions, letting you take larger steps along gradients.

### 6. Common Pitfalls & Tips

- Small batch sizes (e.g., < 16) can yield noisy estimates of mean/variance. Consider GroupNorm or LayerNorm instead.
- Always switch to `eval()`/`model.eval()` during inference so running statistics are used.
- Momentum hyperparameter in updating running stats often set between 0.9 and 0.99.
- Place BatchNorm before or after activation carefully—common pattern is Dense → BatchNorm → Activation.

### 7. Interview-Ready Insights

- Explain how BatchNorm reduces covariate shift and allows higher learning rates.
- Contrast BatchNorm with LayerNorm, InstanceNorm, and GroupNorm—when each shines (e.g., RNNs vs. CNNs vs. small-batch scenarios).
- Discuss how BatchNorm injects noise during training via batch statistics, acting like a regularizer.
- Be ready to derive backward-pass equations succinctly for dL/dx.

### 8. Practice Exercises

### Exercise 1: From-Scratch BatchNorm on MNIST

1. Build a two-layer fully connected network in NumPy.
2. Insert your `BatchNorm1d` layer between Dense layers and ReLU.
3. Train on a small subset of MNIST (e.g., 5k examples) and compare:
    - Learning curves with vs. without BatchNorm
    - Final validation accuracy

Hint: Track running_mean and running_var to switch between train/inference modes.

### Exercise 2: Visualizing Activation Distributions

1. Write a training loop on CIFAR-10 with a simple ConvNet in PyTorch.
2. After each epoch, capture pre-ReLU activations for one batch.
3. Plot histograms (use Matplotlib) of these activations before and after applying BatchNorm.
4. Observe how the distribution evolves over epochs.

### Exercise 3: Batch Size Sensitivity

1. Train the same Model+BatchNorm on Fashion-MNIST with batch sizes [8, 32, 128].
2. Record validation accuracy and training stability (loss spikes).
3. Analyze how batch size affects normalization quality and suggest alternative norms if needed.

---

## Fitting Batch Normalization into a Neural Network

### 1. Direct Definition

Batch normalization is inserted between a layer’s linear transformation and its activation.

It normalizes each mini-batch’s activations to zero mean and unit variance, then applies learned scale (γ) and shift (β).

### 2. Concept Intuition

Adding batch norm is like placing an auto-tuner after each layer so downstream layers always see inputs in a familiar range.

This stabilization speeds up convergence, allows larger learning rates, and reduces sensitivity to weight initialization.

### 3. Mathematical Breakdown

For a layer computing `Z = W·A_prev + b`, batch norm fits in as:

```python
# 1. Linear step
Z = W.dot(A_prev) + b

# 2. Batch statistics
mu     = np.mean(Z, axis=1, keepdims=True)
var    = np.var(Z, axis=1, keepdims=True)
Z_norm = (Z - mu) / np.sqrt(var + eps)

# 3. Scale and shift
Z_tilde = gamma * Z_norm + beta

# 4. Activation
A = activation(Z_tilde)
```

- `gamma` and `beta` are learnable vectors of shape `(units, 1)`.
- `eps` avoids division by zero.
- During inference, use running averages of `mu` and `var` instead of batch statistics.

### 4. Code & Practical Application

### 4.1 NumPy Example

```python
import numpy as np

class BatchNormLayer:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps, self.momentum = eps, momentum
        self.gamma = np.ones((dim, 1))
        self.beta  = np.zeros((dim, 1))
        self.running_mean = np.zeros((dim, 1))
        self.running_var  = np.zeros((dim, 1))

    def forward(self, Z, train=True):
        if train:
            mu  = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mu
            self.running_var  = self.momentum*self.running_var  + (1-self.momentum)*var
        else:
            mu, var = self.running_mean, self.running_var

        Z_norm   = (Z - mu) / np.sqrt(var + self.eps)
        out      = self.gamma * Z_norm + self.beta
        self.cache = (Z, Z_norm, mu, var)
        return out

    # backward omitted for brevity
```

Insert in a two-layer network:

```python
# Forward pass
Z1    = W1.dot(X) + b1
BN1   = bn1.forward(Z1, train=True)
A1    = relu(BN1)
Z2    = W2.dot(A1) + b2
A2    = softmax(Z2)
```

### 4.2 TensorFlow / Keras

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(128, input_shape=(input_dim,)),
    layers.BatchNormalization(),    # fit here
    layers.Activation('relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

### 4.3 PyTorch

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu= nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)       # fit here
        x = self.relu(x)
        return self.fc2(x)
```

### 5. Visualization / Geometry

Before batch norm, each layer’s output distribution shifts as weights update:

```
Batch 1:   ┌───┐      Batch 2:     ┌────┐
           │   │                   │    │
```

After batch norm, outputs stay centered:

```
          ┌────────┐
          │   •••  │
```

On the loss surface, normalizing flattens narrow valleys—gradient steps become more reliable.

### 6. Common Pitfalls & Tips

- Always call `model.eval()` (PyTorch) or `training=False` (Keras) for inference to use running stats.
- For very small batches (<16), consider LayerNorm or GroupNorm.
- Place BatchNorm before activation for stable gradient flow.
- Tune the momentum (default 0.9) when your data distribution shifts frequently.

### 7. Interview-Ready Insights

- Explain why BatchNorm reduces “internal covariate shift” and acts like a regularizer by adding noise.
- Contrast its use in CNNs versus RNNs and why LayerNorm often replaces it in sequence models.
- Discuss how γ and β let the network learn identity transforms if normalization hurts performance.

### 8. Practice Exercises

1. From-Scratch Integration
    - Build a three-layer NumPy network.
    - Insert your `BatchNormLayer` between each Dense and ReLU.
    - Train on a subset of MNIST; compare speed and accuracy vs. network without BatchNorm.
2. Framework Comparison
    - Implement the same CNN in TensorFlow and PyTorch, each with and without BatchNorm.
    - Train for 5 epochs on CIFAR-10 and record training loss curves.
    - Plot and analyze convergence differences.
3. Inference Mode Validation
    - Train a Keras model with BatchNorm.
    - During inference, feed one example at a time (`batch_size=1`) and compare predictions with `batch_size=64`.
    - Observe discrepancies if running stats weren’t updated correctly.

---

## Why Batch Normalization Works

### 1. Direct Explanation

Batch normalization speeds up and stabilizes training by normalizing each layer’s inputs. This reduces the “internal covariate shift,” smooths the optimization landscape, and injects a mild regularization effect.

### 2. Conceptual Intuition

Imagine each layer as a person receiving food ingredients whose flavors keep shifting day to day. They’d constantly relearn recipes. Batch norm is like pre–measuring and standardizing ingredients—every layer always gets the same baseline flavors, so they can focus on learning the task rather than re-adapting to shifting inputs.

- Stabilized inputs let you use higher learning rates.
- Reduces vanishing/exploding gradients by keeping activations in a well-conditioned range.
- Acts like a light regularizer: the noise in batch statistics prevents overfit.

### 3. Mathematical Explanation

Given pre-activation vector (Z\in\mathbb{R}^{m\times n}) (m examples, n units):

```python
mu   = mean(Z, axis=0)                 # shape: (n,)
var  = var(Z, axis=0)                  # shape: (n,)
Zhat = (Z - mu) / sqrt(var + eps)      # normalize

# rescale & shift
Y    = gamma * Zhat + beta             # learnable gamma, beta
```

Why this helps optimization:

- **Zero-centered inputs**: The gradient with respect to weights (W) becomes
    
    ```python
    dL/dW = (1/m) * (dL/dY * gamma / sqrt(var + eps)).T · A_prev
    ```
    
    which is less sensitive to scale of (A_{prev}).
    
- **Improved condition number**: By enforcing unit variance, the Hessian's eigenvalues cluster more tightly, making descent paths more direct and stable.

### 4. Code & Practical Application

### 4.1 Aggressive Learning Rate with BatchNorm

Try training a small network on MNIST with/without BatchNorm. Notice how with BatchNorm you can crank the LR much higher without divergence.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(use_bn, lr):
    layers_list = [
        layers.Dense(256, input_shape=(784,)),
    ]
    if use_bn:
        layers_list += [layers.BatchNormalization()]
    layers_list += [layers.Activation('relu'),
                    layers.Dense(10, activation='softmax')]

    model = models.Sequential(layers_list)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train with and without BN
for use_bn in [False, True]:
    model = build_model(use_bn, lr=1e-2)
    print("Use BN:", use_bn)
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
```

You’ll see the model without BN blows up or learns very slowly, while the BN model converges quickly.

### 5. Visualization

```
Loss Surface Cross-Section
                /
               /   ← narrow, jagged without BN
              /
             /
            /
           /    ← smoother, wider valleys with BN
          /
______/
```

Plotting a 2-D slice of the loss around a trained solution shows that BatchNorm flattens steep walls, letting SGD take larger, more consistent steps.

### 6. Common Pitfalls & Tips

- Very small batch sizes yield noisy estimates of mean/variance.
- Always switch to inference mode (`model.eval()` in PyTorch, `training=False` in Keras) so running averages are used.
- Placing BatchNorm before activation is standard; putting it after can degrade performance.
- The momentum for running stats (usually 0.9) may need tuning if your data distribution drifts.

### 7. Interview-Ready Insights

- Explain the original “internal covariate shift” motivation and how later research shifted focus to loss-surface smoothing as the main benefit.
- Discuss how BatchNorm injects noise in forward passes, acting similarly to dropout’s regularization.
- Contrast BatchNorm with LayerNorm and GroupNorm: why BatchNorm shines in convolutional models but struggles with tiny batches or sequence data.

### 8. Practice Exercises

1. **Gradient Stability Check**
    - Train a two-layer NumPy network on a toy dataset with and without your BatchNorm implementation.
    - After each weight update, log the norm of the gradients.
    - Plot gradient norms over iterations to see how BatchNorm stabilizes updates.
2. **Loss Surface Visualization**
    - Choose two random directions in parameter space around a trained model.
    - Evaluate and plot the loss on a grid in that 2D plane for models with and without BatchNorm.
    - Observe valley width and smoothness differences.
3. **Batch Size Sensitivity**
    - Using PyTorch, train the same CNN with batch sizes [8, 32, 128], both with and without BatchNorm.
    - Record best validation accuracies and training times.
    - Analyze how BatchNorm’s benefit changes as batch size shrinks.

---

## Batch Normalization at Test Time

### 1. Direct Definition

At test time (inference), BatchNorm layers use accumulated running statistics (running mean and running variance) computed during training instead of per-batch statistics. The layer then applies the same scale (γ) and shift (β) to produce a deterministic, stable output.

### 2. Concept Intuition

During training, each mini-batch’s mean and variance fluctuate, injecting a bit of noise that regularizes the model. But at inference, you want consistent predictions regardless of batch size or order of examples.

By freezing the normalization to the running estimates, you ensure each feature is centered and scaled the same way the network learned, avoiding the “jitter” you’d get if you recomputed statistics on a single test example or small batch.

### 3. Mathematical Breakdown

Given an input activation vector **x** of shape `(batch_size, features)`:

1. Retrieve stored statistics:
    
    ```python
    mu_running  = running_mean    # shape: (features,)
    var_running = running_var     # shape: (features,)
    eps         = 1e-5            # small constant
    ```
    
2. Normalize using running stats:
    
    ```python
    x_hat = (x - mu_running) / np.sqrt(var_running + eps)
    ```
    
3. Scale and shift:
    
    ```python
    y = gamma * x_hat + beta
    ```
    
- `gamma` and `beta` are the learned scale and shift parameters.
- No gradient updates happen at test time.

### 4. Code & Practical Application

### 4.1 NumPy-Style Inference Step

```python
def batchnorm_inference(x, running_mean, running_var, gamma, beta, eps=1e-5):
    x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    return gamma * x_hat + beta

# Example
import numpy as np
x_new           = np.random.randn(1, 128)       # one test example
running_mean    = np.zeros(128)                 # learned during training
running_var     = np.ones(128)                  # learned during training
gamma, beta     = np.ones(128), np.zeros(128)

y_pred = batchnorm_inference(x_new, running_mean, running_var, gamma, beta)
```

### 4.2 TensorFlow / Keras

```python
# Define model
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(64, input_shape=(100,)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dense(10)
])

# Train model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5)

# Inference: ensure training=False or call model.evaluate/predict
preds = model(X_test, training=False)
```

### 4.3 PyTorch

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(100, 64)
        self.bn   = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.out  = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)         # uses running stats in eval mode
        x = self.relu(x)
        return self.out(x)

model = Net()
# After training...
model.eval()               # switch to inference mode
with torch.no_grad():
    preds = model(torch.Tensor(X_test))
```

### 5. Visualization / Geometry

```
Training Phase:
  Batch 1:  ── Normalized by batch stats → noise injected
  Batch 2:  ── Different stats → different normalization

Inference Phase:
  All inputs ── Normalized by fixed running stats → stable output
```

On a feature-wise histogram:

- Training batches shift the mean/variance per batch.
- Inference histogram remains fixed around zero mean and unit variance.

### 6. Common Pitfalls & Tips

- Forgetting to switch modes:
    - Keras: call the layer/model with `training=False`.
    - PyTorch: call `model.eval()`, then `model.train()` when resuming training.
- Small test batches:
    - Batch size of 1 still uses the same running stats—don’t recompute per-example stats.
- Unupdated running stats:
    - If you never train long enough or use very small batches, running estimates may be poor. Consider accumulating stats on a hold-out set.

### 7. Interview-Ready Insights

- Explain why batch statistics are inappropriate at test time—single examples lead to wildly varying normalization.
- Discuss how γ and β allow the network to learn the ideal activation distribution and even recover the identity transform if needed.
- Contrast with LayerNorm (which normalizes per sample) and why LayerNorm is sometimes preferred in sequence models where batch sizes vary at inference.

### 8. Practice Exercises

1. From-Scratch Inference Check
    - Train a small NumPy network with your BatchNorm implementation on a toy dataset.
    - Save the running_mean and running_var.
    - Switch to inference mode, feed single examples, and confirm outputs never “jitter” compared to batch inputs.
2. Mode-Switching in Frameworks
    - Build a Keras model with BatchNorm and train it.
    - Call `model.predict()` vs. `model(X_test, training=True)` on the same inputs.
    - Plot the difference in outputs to see the effect of using batch vs. running stats.
3. Running Stats Quality
    - Train a PyTorch model with very small batch size (e.g., 4).
    - After training, evaluate on validation set to measure performance drop.
    - Retrain with a larger batch size or accumulate running stats over extra minibatches and compare improvements.

---

## Softmax Regression

### 1. Direct Definition

Softmax regression (a.k.a. multiclass logistic regression) generalizes logistic regression to handle (K) classes.

It uses the softmax function to convert raw scores (logits) into class probabilities and trains by minimizing the cross-entropy loss.

### 2. Concept Intuition

- For binary classification, logistic regression maps a score to ([0,1]). Softmax extends that to (K) classes, ensuring all predicted probabilities sum to 1.
- You can think of each class having its own “score” line $(W_k^T x + b_k)$. Softmax turns those scores into a probability distribution over classes.
- The model learns weight vectors $(W_k)$ so that the correct class’s score is higher than the others by a margin.

Why it matters

- It’s the foundation for the final layer in deep neural networks for classification.
- Understanding Softmax regression teaches you how multiclass decisions arise from linear models and how cross-entropy shapes weight updates.

### 3. Mathematical Breakdown

Given input feature vector $(x \in \mathbb{R}^n)$, parameters $(W \in \mathbb{R}^{K \times n})$, $(b \in \mathbb{R}^K)$, and one-hot label $(y \in {0,1}^K):$

1. **Logits**
    
    ```python
    z = W.dot(x) + b        # shape: (K,)
    ```
    
2. **Softmax probabilities**
    
    ```python
    exp_z = np.exp(z - np.max(z))
    probs = exp_z / np.sum(exp_z)   # shape: (K,)
    ```
    
3. **Cross-entropy loss**
    
    ```python
    L = -sum(y[k] * log(probs[k]) for k in range(K))
    ```
    
4. **Vectorized cost over m examples**
    
    ```python
    Z     = W.dot(X) + b[:, None]                     # X shape: (n, m)
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    P     = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)  # shape: (K, m)
    loss  = -np.sum(Y * np.log(P)) / m
    ```
    
5. **Gradients**
    
    ```python
    dZ    = (P - Y) / m           # shape: (K, m)
    dW    = dZ.dot(X.T)           # shape: (K, n)
    db    = np.sum(dZ, axis=1)    # shape: (K,)
    ```
    

### 4. Code & Practical Application

### 4.1 NumPy Implementation (Training Loop)

```python
import numpy as np

def softmax(z):
    z_max = np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def compute_loss_and_grads(X, Y, W, b):
    m = X.shape[1]
    Z = W.dot(X) + b[:, None]          # (K, m)
    P = softmax(Z)                     # (K, m)
    loss = -np.sum(Y * np.log(P)) / m

    dZ = (P - Y) / m                   # (K, m)
    dW = dZ.dot(X.T)                   # (K, n)
    db = np.sum(dZ, axis=1)            # (K,)
    return loss, dW, db

def train_softmax(X, Y, lr=0.1, num_epochs=1000):
    n, m = X.shape
    K    = Y.shape[0]
    W    = np.random.randn(K, n) * 0.01
    b    = np.zeros(K)

    for epoch in range(num_epochs):
        loss, dW, db = compute_loss_and_grads(X, Y, W, b)
        W -= lr * dW
        b -= lr * db
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}")
    return W, b
```

### 4.2 TensorFlow / Keras Example

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n,)),
    layers.Dense(K, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_split=0.1)
```

### 4.3 PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SoftmaxReg(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return nn.functional.log_softmax(self.linear(x), dim=1)

model = SoftmaxReg(n, K)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.NLLLoss()  # negative log likelihood

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(torch.from_numpy(X_train.T).float())
    loss    = criterion(outputs, torch.from_numpy(np.argmax(Y_train, axis=0)))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

### 5. Visualization / Geometry

Imagine a 2D input and 3 classes—each weight vector (W_k) defines a line. The region where class (k) is predicted is where (W_k^T x + b_k) is the largest.

```
Class 0 region   Class 1 region   Class 2 region
 ┌───────────┐   ┌───────────┐   ┌───────────┐
 │   \   /   │   │   \   /   │   │   \   /   │
 └────*──────┘   └────*──────┘   └────*──────┘
```

Decision boundaries are linear separators (hyperplanes), and softmax turns the distances to these hyperplanes into probabilities.

### 6. Common Pitfalls & Tips

- **Numerical stability**: Always subtract the max logit before exponentiating to avoid overflow.
- **One-hot encoding**: Ensure labels are properly one-hot for vectorized cross-entropy.
- **Learning rate sensitivity**: Softmax regression can diverge if the learning rate is too large. Start small (e.g., 0.1) and tune on a log scale.
- **Regularization**: Add L2 penalty on (W) to prevent overfitting, especially when (n) is large relative to data.

### 7. Interview-Ready Insights

- Be prepared to derive gradients of the cross-entropy + softmax in one shot:$[ \frac{\partial L}{\partial z_k} = p_k - y_k ]$
- Explain why softmax + cross-entropy gradient simplifies to (p - y).
- Contrast one-vs-all logistic regression versus softmax regression in terms of training and consistency.
- Discuss how softmax regression is the last layer in deep nets and how backprop flows through it.

### 8. Practice Exercises

**Exercise 1: Iris Dataset**

1. Load the Iris dataset (`sklearn.datasets.load_iris`).
2. Implement softmax regression in NumPy.
3. Train on two features (e.g., sepal length & width).
4. Plot decision boundaries and report accuracy.

**Exercise 2: MNIST Subset**

1. Take only digits 0–4 from MNIST.
2. Train softmax regression in TensorFlow on flattened images.
3. Tune learning rate and L2 regularization strength.
4. Plot training/validation accuracy curves.

**Exercise 3: Vectorization Challenge**

1. Write a non-vectorized version of the gradient computation (nested loops).
2. Profile its runtime on a synthetic dataset of size $(n=100, m=10{,}000)$.
3. Then write the fully vectorized version and compare speedups.

---

## Training a Softmax Classifier

### 1. Direct Definition

Training a softmax classifier means optimizing its weight matrix W and bias vector b so that, for each input x, the softmax outputs

```python
p = softmax(W · x + b)
```

match the true one-hot labels y by minimizing the cross-entropy loss over a training set.

### 2. Concept Intuition

- Each class k has a weight vector W_k. The score $z_k = W_k·x + b_k$ measures how much x belongs to class k.
- Softmax turns these scores into a probability distribution over classes.
- Training nudges W and b so that the correct class’s score is higher than all others—driving down loss.
- Using minibatch gradient descent, you update W and b in small steps, averaging the gradient over each batch for stability.

### 3. Mathematical Breakdown

Given m examples in X (shape (n, m)) and one-hot labels Y (shape (K, m)):

1. Forward pass
    
    ```python
    Z     = W.dot(X) + b[:, None]                        # (K, m)
    expZ  = np.exp(Z - np.max(Z, axis=0, keepdims=True)) # subtract max for stability
    P     = expZ / np.sum(expZ, axis=0, keepdims=True)   # softmax probs (K, m)
    ```
    
2. Cross-entropy cost
    
    ```python
    loss = -np.sum(Y * np.log(P)) / m
    ```
    
3. Backward pass (gradients)
    
    ```python
    dZ = (P - Y) / m                # (K, m)
    dW = dZ.dot(X.T) + (λ * W)      # add L2 regularization term
    db = np.sum(dZ, axis=1)         # (K,)
    ```
    
4. Parameter update
    
    ```python
    W -= lr * dW
    b -= lr * db
    ```
    

### 4. Code & Practical Application

### 4.1 NumPy: Minibatch Gradient Descent

```python
import numpy as np

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ    = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def train_softmax(X, Y, lr=0.1, λ=0.0, epochs=200, batch_size=64):
    n, m    = X.shape
    K       = Y.shape[0]
    W       = np.random.randn(K, n) * 0.01
    b       = np.zeros(K)
    for epoch in range(epochs):
        # Shuffle and minibatch
        perm = np.random.permutation(m)
        X_sh, Y_sh = X[:, perm], Y[:, perm]
        for i in range(0, m, batch_size):
            Xb = X_sh[:, i:i+batch_size]
            Yb = Y_sh[:, i:i+batch_size]

            # Forward
            Z  = W.dot(Xb) + b[:, None]
            P  = softmax(Z)
            loss = -np.sum(Yb * np.log(P)) / Xb.shape[1]

            # Backward
            dZ = (P - Yb) / Xb.shape[1]
            dW = dZ.dot(Xb.T) + λ * W
            db = np.sum(dZ, axis=1)

            # Update
            W -= lr * dW
            b -= lr * db

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, loss: {loss:.4f}")
    return W, b

# Example usage:
# X_train shape (n, m), Y_train one-hot shape (K, m)
# W, b = train_softmax(X_train, Y_train)
```

### 4.2 PyTorch: Using CrossEntropyLoss

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SoftmaxModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)  # logits

model     = SoftmaxModel(input_dim=n, num_classes=K)
criterion = nn.CrossEntropyLoss()   # combines LogSoftmax + NLLLoss
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=λ)

for epoch in range(200):
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        idx = permutation[i:i+batch_size]
        xb, yb = X_train[idx], y_train[idx]  # y_train: class indices, not one-hot

        optimizer.zero_grad()
        logits   = model(xb)
        loss      = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")
```

### 5. Visualization / Geometry

For a 2D input and 3 classes:

```
 Weight vectors W0, W1, W2 define 3 lines (hyperplanes).
 Each region where Wk·x + bk is largest is assigned class k.

 Decision boundaries are straight lines separating these regions.
```

During training, you can plot how these lines rotate and shift in 2D space, closing in on data clusters.

### 6. Common Pitfalls & Tips

- Subtract max logits before exponentiating to prevent overflow.
- Initialize W with small random values to break symmetry.
- Choose learning rate on a log scale (e.g., 1e-3 to 1).
- Add L2 regularization (weight decay) if you see overfitting.
- Monitor training vs. validation loss to detect underfitting/overfitting early.

### 7. Interview-Ready Insights

- Derive why the gradient of softmax plus cross-entropy simplifies to (P – Y).
- Explain how CrossEntropyLoss in PyTorch fuses LogSoftmax and NLLLoss for numerical stability.
- Contrast one-vs-all logistic regression (training separate binaries) with softmax regression (joint training) in terms of consistency and efficiency.
- Discuss why vectorization (batch operations) yields massive speedups over explicit loops.

### 8. Practice Exercises

1. **Iris Classification**
    - Load Iris (`sklearn.datasets`).
    - Implement `train_softmax` in NumPy on sepal‐petal features.
    - Visualize decision boundaries.
2. **MNIST Subset**
    - Restrict MNIST to digits {0,1,2}.
    - Train with minibatch SGD in NumPy.
    - Experiment with λ ∈ [0, 0.01, 0.1], plot validation accuracy vs. λ.
3. **Vectorization Challenge**
    - Write a naïve loop-based gradient calculation over m examples.
    - Profile runtime vs. the vectorized minibatch version for m=10 000.
    - Report speedup factors and discuss trade-offs.

---

## Deep Learning Frameworks

### 1. Direct Definition

A deep learning framework is a software library that provides building blocks—tensor operations, automatic differentiation, neural-network layers, optimizers, and GPU support—to design, train, and deploy deep neural networks with minimal boilerplate.

### 2. Concept Intuition

- At its core, a framework manages
    - efficient tensor arithmetic across CPUs/GPUs
    - automatic gradient computation (backpropagation)
    - model abstraction (layers, modules)
    - training loops, checkpointing, and deployment tooling
- Using a framework lets you focus on **model architecture** and **experiments**, not on low-level CUDA kernels or manual derivative code.

Why it matters

- Without a framework, every new layer, optimizer, or custom op must be hand-coded and optimized.
- Frameworks give you battle-tested performance, community-supported extensions, and integration with visualization and production ecosystems.

### 3. Core Components

1. **Tensors and Ops**
    - Multidimensional arrays with device awareness (CPU/GPU)
    - Broadcasted arithmetic, linear algebra, indexing
2. **Autograd / Gradient Engine**
    - Builds a graph of tensor operations at runtime or compile time
    - Computes gradients via reverse-mode automatic differentiation
3. **Model Abstractions**
    - Layers (Dense/Conv/RNN) or Modules that encapsulate parameters and forward logic
    - Sequential containers and functional APIs
4. **Optimizers & Schedulers**
    - SGD, Adam, RMSProp, plus learning-rate schedules and warm-up
5. **Data Pipeline**
    - Dataset and DataLoader abstractions for batching, shuffling, and parallel data loading
    - Preprocessing transforms (e.g., image augmentations)
6. **Utilities**
    - Checkpointing, early stopping, profiling, visualization (TensorBoard, torch.utils.tensorboard)
    - Export formats (SavedModel, ONNX) for serving

### 4. Code & Practical Application

### 4.1 TensorFlow 2.x (Eager + Functional API)

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# 1. Define model
inputs = tf.keras.Input(shape=(28,28,1))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPool2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs, outputs)

# 2. Compile
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Prepare data
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None] / 255.0
x_val   = x_val[..., None]   / 255.0

# 4. Train
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(x_val, y_val)
)

# 5. Save for serving
model.save('mnist_tf_model')
```

### 4.2 PyTorch (Dynamic Graph + Imperative API)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 1. Define model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1  = nn.Linear(32*13*13, 64)
        self.fc2  = nn.Linear(64, 10)
    def forward(self, x):
        x = torch.relu(self.pool(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = ConvNet().to('cuda')

# 2. Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 3. Data pipeline
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_ds, val_ds = random_split(full_dataset, [55000, 5000])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)

# 4. Training loop
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to('cuda'), yb.to('cuda')
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    # validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to('cuda'), yb.to('cuda')
            pred   = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    print(f"Epoch {epoch}: val_acc={correct/total:.4f}")
```

### 4.3 JAX (Functional + XLA Acceleration)

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# 1. Model
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# 2. Setup
def create_state(key, model, lr):
    params = model.init(key, jnp.ones([1, 784]))['params']
    tx     = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss   = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# 3. Data (flattened MNIST omitted for brevity)
# 4. Training loop: jitted functions iterate over batches
```

### 5. Visualization / Geometry

```
[ DataLoader ] → [ Forward Graph ] → [ Loss ] → [ Autograd Graph ] → [ Backward ] → [ Op Kernels ]
```

- **Static Graph (TF1.x, XLA)**: Graph compiled ahead; optimized end-to-end.
- **Dynamic Graph (PyTorch, TF2 eager)**: Graph built on the fly, easier to debug.

### 6. Common Pitfalls & Tips

- **Version mismatches**: TF2 vs TF1, CUDA/cuDNN versions, PyTorch–CUDA compatibility
- **Eager vs Graph**: Debugging is easier in eager mode—use `tf.function` to compile critical paths later
- **Memory leaks**: Retaining references to computation graphs (e.g., storing torch tensors in lists)
- **Reproducibility**: Set random seeds across frameworks, disable nondeterministic CuDNN ops if needed
- **Profiling**: Use TensorBoard profiler (`tf.profiler`) or PyTorch’s `torch.profiler` to identify bottlenecks

### 7. Interview-Ready Insights

- **Autograd mechanics**: Reverse-mode AD builds a tape of operations; backward pass traverses it to compute gradients.
- **Static vs Dynamic**: Static graphs enable global optimization (XLA, TensorRT), while dynamic graphs offer Python-native control flow.
- **Deployment**:
    - TensorFlow’s SavedModel → TensorFlow Serving or TensorFlow Lite
    - PyTorch’s TorchScript → LibTorch or ONNX export
    - JAX’s XLA → TPU acceleration
- **Ecosystem maturity**: TensorFlow has wide deployment tooling; PyTorch excels in research agility; JAX shines in composable function transforms (grad, jit, vmap).

### 8. Practice Exercises

1. **Custom Layer in Two Frameworks**
    - Implement a `ScaledLinear` layer that multiplies its weight matrix by a learned scalar.
    - Write it as a `tf.keras.layers.Layer` and as a `torch.nn.Module`.
    - Train a small MLP on synthetic data to verify both behave identically.
2. **Profiling Comparison**
    - Train the same CNN on CIFAR-10 for 5 epochs in TensorFlow and PyTorch.
    - Use each framework’s profiler to record forward/backward time and GPU utilization.
    - Compare training speed and memory footprint; write a short summary.
3. **ONNX Export & Inference**
    - Train a PyTorch model, export it to ONNX, and run inference with ONNX Runtime.
    - Measure latency on CPU vs GPU.
    - Discuss the steps needed to ensure exported graph matches training behavior (e.g., `model.eval()`).
4. **Graph vs Eager Debugging**
    - Write a small snippet in TensorFlow that fails in `@tf.function` but passes in eager mode.
    - Diagnose the cause and refactor code to be compatible with both.

---

## TensorFlow

### 1. Direct Definition

TensorFlow is an open-source library for numerical computation using data-flow graphs. Nodes represent mathematical operations, edges represent multi-dimensional data arrays (tensors). It provides high-level APIs (Keras) and low-level primitives, automatic differentiation, graph optimization, and deployment tools for CPUs, GPUs, and TPUs.

### 2. Concept Intuition

TensorFlow separates two phases:

- **Definition (Graph Construction)**
    
    You describe your computation as a graph of operations on tensors. This lets TensorFlow analyze and optimize the entire workflow before running.
    
- **Execution**
    
    The runtime schedules and executes operations efficiently on available hardware, parallelizing across machines or devices.
    

In TensorFlow 2.x, **eager execution** is enabled by default. You write code imperatively (like NumPy), and TensorFlow records operations under the hood, so you still get graph optimizations via `@tf.function` when you need them.

### 3. Mathematical Breakdown

At its core, TensorFlow implements reverse-mode automatic differentiation on a computation graph.

1. **Tensors**
    
    Multidimensional arrays, represented as `tf.Tensor`, with a static or partially known shape and a data type (e.g., `float32`).
    
2. **Operations (Ops)**
    
    Functions like matrix multiply, convolution, or activation build up the graph. Each op takes tensors as inputs and produces tensors as outputs.
    
3. **GradientTape**
    
    Records the forward pass on “watched” tensors. During the backward pass, it computes derivatives by traversing the recorded operations in reverse.
    
    ```python
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss   = loss_fn(y_true, y_pred)
    grads  = tape.gradient(loss, model.trainable_variables)
    ```
    
4. **Graph Tracing**
    
    Wrapping a Python function with `@tf.function` traces its computation into a static graph. This graph can be serialized, optimized, and executed repeatedly with low overhead.
    

### 4. Code & Practical Application

### 4.1 High-Level (Keras) Example

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
```

### 4.2 Low-Level (GradientTape) Example

```python
# Custom training loop with tf.GradientTape
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn   = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss   = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Validation
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
```

### 4.3 Data Pipeline with `tf.data`

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(128)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### 5. Visualization / Geometry

```
[ Python Function ]
        ↓  @tf.function traces
[ Computational Graph ]
        ↓  optimizations (XLA, pruning)
[ Device Executors ]
        ↓  parallel kernels on CPU/GPU/TPU
[ Scalars, Vectors, Matrices flowing through Ops ]
```

The graph abstraction lets TensorFlow fuse operations (e.g., batch norm + activation) and schedule them to minimize memory reads and writes.

### 6. Common Pitfalls & Tips

- Forgetting to set `training=True` in layers like BatchNormalization or Dropout during training leads to stale behavior.
- Over-eager use of `@tf.function` can hide Python errors; debug in eager mode first.
- Shape mismatches: TensorFlow enforces static shapes when tracing. Use `None` for dynamic batch dimensions.
- GPU memory growth: enable `tf.config.experimental.set_memory_growth` to allocate memory on demand.
- Mixed precision: use `tf.keras.mixed_precision.Policy` to speed up on modern GPUs, but watch for numeric stability.

### 7. Interview-Ready Insights

- Explain eager execution versus graph mode and the role of `tf.function` in bridging the two.
- Discuss how `tf.data` pipelines can be distributed across workers and prefetch to hide I/O latency.
- Describe TensorFlow’s SavedModel format for serving, and how it preserves both the graph and variable values.
- Contrast TensorFlow’s static graph optimizations (XLA) with PyTorch’s dynamic tracing approach (TorchScript).

### 8. Practice Exercises

1. **Custom Layer with `tf.Module`**
    - Implement a layer that scales its input by a learnable scalar and adds a bias.
    - Integrate it into a Keras model and confirm gradient updates via `model.trainable_variables`.
2. **Manual Gradient Checking**
    - Build a small MLP and write a training loop with `GradientTape`.
    - Numerically approximate gradients for one weight and compare to `tape.gradient` output.
3. **Advanced `tf.data` Pipeline**
    - Load CIFAR-10 using `tf.data`.
    - Apply random flips, rotations, and color jitter in parallel.
    - Benchmark throughput with and without `AUTOTUNE`.
4. **Graph Performance Profiling**
    - Wrap your training step in `@tf.function` and use the TensorFlow Profiler to identify bottlenecks.
    - Optimize by fusing small ops or adjusting batch size, then re-profile.

---