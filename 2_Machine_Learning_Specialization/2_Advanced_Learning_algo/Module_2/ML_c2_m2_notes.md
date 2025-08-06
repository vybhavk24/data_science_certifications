# ML_c2_m2

## TensorFlow Implementation

### Pre-requisites

- Python 3.x installed and basic familiarity with functions and classes
- TensorFlow 2.x installed (`pip install tensorflow`) and eager execution enabled by default
- Understanding of forward propagation, loss functions (cross-entropy or MSE), and gradient descent
- Familiarity with NumPy operations and data batching concepts

### 1. Core Intuition

Training a neural network means adjusting its weights so that its predictions match the true labels.

At each step you:

- Run a **forward pass** (compute predictions)
- Compute a **loss** measuring prediction error
- Perform a **backward pass** (compute gradients)
- **Update** weights using an optimizer (e.g., gradient descent)

TensorFlow automates gradient computation and offers both high-level and low-level APIs.

### 2. High-Level API: `tf.keras.Model`

TensorFlow’s Keras API lets you train models with two lines of code:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32
)
```

- **optimizer** handles weight updates
- **loss** computes error
- **metrics** track performance
- **fit()** runs the training loop internally

### 3. Under-the-Hood Math in Code Blocks

### Forward pass: compute logits and predictions

```python
logits = W1 @ X + b1
activations = relu(logits)
logits_out = W2 @ activations + b2
probs = softmax(logits_out)
```

### Loss and gradient descent update

```python
# cross-entropy loss for one batch
loss = -tf.reduce_mean(y_true * tf.math.log(probs))

# gradient descent update
W2.assign_sub(learning_rate * dW2)
b2.assign_sub(learning_rate * db2)
```

### 4. Custom Training Loop with `tf.GradientTape`

When you need full control:

```python
optimizer = tf.keras.optimizers.Adam(0.001)

for epoch in range(epochs):
    for X_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
```

- `GradientTape` records operations for automatic differentiation
- `apply_gradients` updates weights

### 5. Real-World ML Workflow Example

1. **Load and preprocess data**
    
    ```python
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_val = x_val.reshape(-1, 784) / 255.0
    ```
    
2. **Build model**
    
    ```python
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation="relu", input_shape=(784,)),
      tf.keras.layers.Dense(10, activation="softmax")
    ])
    ```
    
3. **Compile and train** (as shown above)
4. **Evaluate**
    
    ```python
    loss, acc = model.evaluate(x_val, y_val)
    print("Validation accuracy:", acc)
    ```
    
5. **Save**
    
    ```python
    model.save("mnist_model.h5")
    ```
    

### 6. Practice Problems

- Implement and train a 3-layer MLP on Fashion-MNIST using both `model.fit` and a custom loop. Compare training curves.
- Add L2 regularization and observe how validation accuracy changes.
- Experiment with learning rate schedules (`tf.keras.callbacks.LearningRateScheduler`) and plot loss vs lr.

### 7. Visual Insights

- Plot **training vs validation loss** per epoch to spot under- or over-fitting.
- Use **weight histograms** (`model.layers[0].get_weights()`) to see how weights evolve.
- Visualize **gradient norms** to detect vanishing or exploding gradients.

### 8. Common Pitfalls & Interview Tips

- **Data pipeline mismatch**: ensure training and inference preprocessing are identical.
- **Forgetting `model.trainable` flags**: frozen layers won’t update if set incorrectly.
- Remember to call `model.compile` before `fit` or `evaluate`.
- Interview question:“How does TensorFlow compute gradients automatically? Explain `GradientTape`.”

---

## Detailed Training Pipeline in TensorFlow

### 1. Data Pipeline Configuration

Before training, create a fast, reproducible input pipeline using `tf.data`.

```python
import tensorflow as tf

def load_and_preprocess(x, y):
    # Normalize images to [0,1]
    x = tf.cast(x, tf.float32) / 255.0
    # Potential augmentations (for images)
    x = tf.image.random_flip_left_right(x)
    return x, y

batch_size = 64
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=10_000, seed=42)
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
```

- Shuffle before batching to randomize each epoch
- Use `AUTOTUNE` for parallel mapping and prefetching

### 2. Weight Initialization

Proper initialization prevents vanishing/exploding signals.

| Layer Type | Common Initializer | Rationale |
| --- | --- | --- |
| Dense + ReLU | He Normal (`he_normal`) | Scales variance by 2/fan_in |
| Dense + tanh | Glorot Uniform (`glorot`) | Balances variance of inputs/outputs |
| Bias | Zeros | Start unbiased |

```python
tf.keras.layers.Dense(
    units=128,
    activation='relu',
    kernel_initializer='he_normal',
    bias_initializer='zeros'
)
```

### 3. Optimizer and Learning Rate Scheduling

Choosing and tuning optimizers drives convergence speed and stability.

1. **Adam**
    - Default: `learning_rate=1e-3`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7`
2. **SGD with Momentum**
    - `learning_rate=1e-2`, `momentum=0.9`
3. **Schedulers**
    - **Step Decay**: reduce LR every *k* epochs
    - **Cosine Decay**: smooth decay to zero
    - **OneCycle**: cycle up then down for faster convergence

```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.5
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### 4. Loss Function Selection

Your choice depends on output type:

- **Classification (multi-class)**:
    - `SparseCategoricalCrossentropy(from_logits=True)`
- **Regression**:
    - `MeanSquaredError()` or `MeanAbsoluteError()`

You can add **label smoothing** to mitigate overconfidence:

```python
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1
)
```

### 5. Custom Training Loop with GradientTape

Full control over each step:

```python
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    # Optional: gradient clipping
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)
    train_acc.update_state(y_batch, logits)

for epoch in range(1, epochs+1):
    train_loss.reset_states()
    train_acc.reset_states()
    for x_b, y_b in train_ds:
        train_step(x_b, y_b)
    for x_v, y_v in val_ds:
        val_logits = model(x_v, training=False)
        val_loss.update_state(loss_fn(y_v, val_logits))
        val_acc.update_state(y_v, val_logits)
    print(f"Epoch {epoch:02d}: "
          f"Train Loss={train_loss.result():.4f}, "
          f"Train Acc={train_acc.result():.4f}, "
          f"Val Loss={val_loss.result():.4f}, "
          f"Val Acc={val_acc.result():.4f}")
```

- Wrap heavy ops in `@tf.function` for graph mode
- Clip gradients to norm ≤ 1 to prevent exploding

### 6. Callbacks for Monitoring & Control

Keras callbacks automate checkpointing, early stopping, and logging:

```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', save_best_only=True, monitor='val_loss'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='logs', histogram_freq=1
    ),
    tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-3 * (0.5 ** (epoch // 5))
    )
]
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)
```

- **ModelCheckpoint** saves weights when validation loss improves
- **EarlyStopping** halts training to avoid overfitting
- **TensorBoard** visualizes metrics and weight histograms

### 7. Regularization Techniques

Combat overfitting by adding noise and constraints:

- **L2 Weight Decay**
    
    ```python
    tf.keras.layers.Dense(
        units=128,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )
    ```
    
- **Dropout**
    
    ```python
    tf.keras.layers.Dropout(rate=0.5)
    ```
    
- **Batch Normalization**Accelerates training and adds slight regularization.

### 8. Reproducibility & Determinism

Fixing seeds and controlling parallelism:

```python
import numpy as np
import random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Enable deterministic ops (may slow down)
tf.config.experimental.enable_op_determinism()
```

### 9. Debugging Training

- Inspect **gradient norms** per layer to catch vanishing/exploding issues.
- Plot **weight histograms** after each epoch in TensorBoard.
- Check for NaNs or infinities in loss/gradients.

### 10. Next-Level Optimization

- Mixed‐precision with `tf.keras.mixed_precision` for faster GPU throughput.
- Distributed training using `tf.distribute.MirroredStrategy`.
- Automated hyperparameter search via Keras Tuner or Optuna.

---

## Alternatives to the Sigmoid Activation

Exploring other activations helps you sidestep sigmoid’s vanishing gradients, non–zero-centered outputs, and slow convergence. Below is a catalog of popular replacements, their formulas, ranges, strengths, and caveats.

### 1. Rectified and Leaky Variants

### 1.1 ReLU (Rectified Linear Unit)

- Formula: `f(x) = max(0, x)`
- Range: `[0, ∞)`
- Pros:
    - Simple, cheap to compute
    - Alleviates vanishing gradients for x > 0
- Cons:
    - “Dying ReLU” where neurons get stuck at zero

### 1.2 Leaky ReLU

- Formula:
    
    ```
    f(x) = { x        if x >= 0
           { α·x     if x < 0
    ```
    
- Typical α: 0.01
- Range: `(-∞, ∞)`
- Pros:
    - Allows small gradient when x < 0
- Cons:
    - Still piecewise linear; negative slope is fixed

### 1.3 PReLU (Parametric ReLU)

- Like Leaky ReLU, but α is a learnable parameter per channel/layer
- Pros:
    - Learns negative slope to optimize gradient flow
- Cons:
    - Slightly more parameters; risk of overfitting on small data

### 2. Smooth Nonlinear Activations

### 2.1 ELU (Exponential Linear Unit)

- Formula:
    
    ```
    f(x) = { x                    if x >= 0
           { α·(exp(x) − 1)      if x < 0
    ```
    
- Range: `(−α, ∞)`
- Pros:
    - Negative values push mean activations toward zero
    - Smooth curve reduces bias shift
- Cons:
    - Requires exp computation
    - α typically needs tuning

### 2.2 SELU (Scaled ELU)

- ELU scaled to self-normalize:
    
    ```
    λ≈1.0507, α≈1.6733
    f(x) = λ·ELU(x; α)
    ```
    
- Pros:
    - Triggers self-normalizing networks under certain initializations
- Cons:
    - Strict requirements on network architecture and dropout

### 2.3 GELU (Gaussian Error Linear Unit)

- Formula:
    
    ```
    f(x) = x·Φ(x)
    where Φ(x) = 0.5·[1 + erf(x/√2)]
    ```
    
- Range: `(−∞, ∞)`
- Pros:
    - Smooth, non-monotonic; used in Transformers
- Cons:
    - More expensive than ReLU

### 3. Novel and Emerging Functions

### 3.1 Swish

- Formula: `f(x) = x·sigmoid(βx)` (β often set to 1)
- Range: `(−0.2785…, ∞)`
- Pros:
    - Smooth, non-monotonic; empirically outperforms ReLU on deep nets
- Cons:
    - Slight compute overhead

### 3.2 Mish

- Formula: `f(x) = x·tanh(softplus(x))`
- softplus(x) = ln(1 + eˣ)
- Range: `(−0.31…, ∞)`
- Pros:
    - Retains small negative values, smoother than Swish
- Cons:
    - Heavier math per activation

### 3.3 Hard-Sigmoid & Hard-Swish

- Approximations using piecewise linear segments for mobile/edge efficiency
- Pros:
    - Low compute cost, suited for quantization
- Cons:
    - Approximate curves may reduce representational power

### 4. Comparative Table

| Activation | Formula Snippet | Range | Compute | Key Benefit |
| --- | --- | --- | --- | --- |
| Sigmoid | `1/(1+e⁻ˣ)` | `(0,1)` | High | Probabilistic output |
| ReLU | `max(0,x)` | `[0,∞)` | Low | Speed, simplicity |
| Leaky ReLU | `αx (x<0), x (x>=0)` | `(-∞,∞)` | Low | Mitigates dying ReLU |
| PReLU | `αx (learned)` | `(-∞,∞)` | Low | Adaptive negative slope |
| ELU | `α(eˣ−1) (x<0), x (x>=0)` | `(−α,∞)` | Medium | Zero-centered activations |
| SELU | `λ·ELU(x;α)` | `(−λ·α,∞)` | Medium | Self-normalizing networks |
| GELU | `x·Φ(x)` | `(−∞,∞)` | High | Smooth non-monotonic, popular in NLP |
| Swish | `x·sigmoid(x)` | `(−0.28,∞)` | Medium | Improves deep-network accuracy |
| Mish | `x·tanh(ln(1+eˣ))` | `(−0.31,∞)` | High | Smooth, retains small negatives |

### 5. Practical Guidelines

- Use **ReLU** for most hidden layers to start.
- Swap in **Leaky/PReLU** if neurons die or you see sparse gradients.
- Try **ELU/SELU** for deeper nets needing zero-centered flows.
- In modern architectures (e.g., Transformers), test **GELU** or **Swish/Mish** for slight accuracy gains.
- For mobile or quantized models, consider **Hard-Swish** or **Hard-Sigmoid**.

### 6. Interview Tip

Explain how activation shapes gradient flow and representational capacity. You might be asked to derive derivative formulas (e.g., Swish’s gradient) or discuss why zero-centered activations speed convergence.

---

## Choosing Activation Functions

### 1. Match Output Activation to Your Task

Select the final layer activation based on problem type:

- Regression
    - Use a linear activation (`activation=None`) to predict unbounded continuous values.
- Binary Classification
    - Use `sigmoid` to map logits into (0,1); pair with `BinaryCrossentropy`.
- Multi-Class Classification
    - Use `softmax` across k units to produce a probability distribution; pair with `CategoricalCrossentropy`.

### 2. Hidden Layers: Default and Fallbacks

Follow a decision flow when picking hidden activations:

1. Start with **ReLU** for most dense or convolutional layers to encourage sparse, efficient gradients.
2. If you observe “dead” neurons (zero gradients), switch to **LeakyReLU** or **PReLU** for small negative slopes.
3. In very deep nets prone to covariate shift, consider **ELU** (with BatchNorm) or **SELU** (with specific initialization and no dropout).
4. For transformer-style or NLP models, test **GELU** for smoother, non-monotonic behavior.
5. On resource-constrained or quantized models, use **Hard-Swish** or **Hard-Sigmoid** for low-cost approximations.
6. If chasing incremental gains on large datasets, experiment with **Swish** or **Mish**, mindful of extra compute per activation.

### 3. Comparative Overview

| Activation | Range | Compute Cost | When to Use |
| --- | --- | --- | --- |
| ReLU | [0, ∞) | Low | Default for hidden layers |
| LeakyReLU | (−∞, ∞) | Low | Mitigating dead neurons |
| PReLU | (−∞, ∞) | Low–Medium | Adaptive negative slope |
| ELU | (−α, ∞) | Medium | Deeper nets + zero-centered outputs |
| SELU | (−λ·α, ∞) | Medium | Self-normalizing MLPs |
| GELU | (−∞, ∞) | High | Transformer/NLP architectures |
| Swish | (−0.28, ∞) | Medium | High-performance vision or NLP |
| Mish | (−0.31, ∞) | High | Extra smooth, small negative tail |
| Hard-Swish | (−0.25, ∞) | Low | Mobile/quantized models |

### 4. Code Snippets in TensorFlow

```python
import tensorflow as tf

# Example: swapping activations dynamically
def build_model(hidden_activation='relu', output_activation='softmax'):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=hidden_activation, input_shape=(784,)),
        tf.keras.layers.Dense(64, activation=hidden_activation),
        tf.keras.layers.Dense(10, activation=output_activation)
    ])

# Instantiate with Mish in hidden layers
model = build_model(hidden_activation=lambda x: x * tf.math.tanh(tf.math.softplus(x)),
                    output_activation='softmax')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### 5. Monitoring & Diagnostics

- Track **activation histograms** in TensorBoard to ensure outputs don’t saturate.
- Plot **gradient norms** to verify nonlinearity still propagates useful signals.
- Visualize **layer-wise mean and variance** post-activation to spot covariate shifts.

### 6. Activation as a Hyperparameter

Treat activations like learning rates—tune them:

- Use **Keras Tuner** or **Optuna** to search over a set of candidates (`['relu','elu','swish']`).
- Profile validation loss and training speed to find the best trade-off.

---

## Why We Need Activation Functions

---

Every neural network layer applies a function to its inputs. If that function is purely linear, no matter how many layers you stack, the model collapses into a single linear mapping. Activation functions break this linearity, unlock model expressiveness, and enable deep learning to solve complex tasks.

### 1. Linearity Bottleneck

When every layer is linear, stacking layers is redundant:

```
# One hidden layer
y = W2 · (W1 · x + b1) + b2

# Simplifies to a single linear map
y = (W2 · W1) · x + (W2 · b1 + b2)
```

Depth adds no power without non-linearity.

### 2. Injecting Non-Linearity

Activation functions apply an element-wise non-linear transform:

- They let networks approximate curves, surfaces, decision boundaries.
- They enable each layer to learn different features rather than repeat a linear map.

### 3. Universal Approximation

With at least one hidden layer and a non-linear activation, a network can approximate any continuous function on a bounded domain, given enough neurons. This is the **Universal Approximation Theorem**.

### 4. Hierarchical Feature Learning

Non-linear activations let each layer build on the last:

1. Early layers learn simple patterns (edges, colors).
2. Middle layers combine them into shapes or textures.
3. Deep layers detect objects or concepts.

Without non-linearity, this hierarchy collapses.

### 5. Training Dynamics and Gradient Flow

Activation choice shapes gradient propagation:

- **Saturating activations** (sigmoid, tanh) squash inputs, causing gradients near zero when inputs are large
- **Non-saturating activations** (ReLU, LeakyReLU) keep gradients alive for positive inputs
- **Zero-centered activations** (tanh, ELU) help faster convergence by balancing positive/negative outputs

Proper activations mitigate vanishing or exploding gradients.

### 6. Biological Inspiration

Real neurons fire only when inputs exceed thresholds. Activation functions mimic this gating:

- ReLU acts like a hard threshold at zero.
- Sigmoid/tanh act like smooth thresholds.

They decide “how much” signal passes forward.

### 7. Output-Specific Activations

Choose final-layer activations to match your task:

- Regression
    
    ```
    activation = linear  # outputs unbounded real values
    ```
    
- Binary classification
    
    ```
    activation = sigmoid  # maps to (0,1)
    ```
    
- Multi-class classification
    
    ```
    activation = softmax  # maps to a probability distribution
    ```
    

### 8. Activation Properties Comparison

| Activation | Range | Zero-Centered | Saturating | Cost |
| --- | --- | --- | --- | --- |
| Sigmoid | (0, 1) | No | Yes | High |
| Tanh | (−1, 1) | Yes | Yes | High |
| ReLU | [0, ∞) | No | No | Low |
| LeakyReLU | (−∞, ∞) | No | No | Low |
| ELU | (−α, ∞) | Yes | Partially | Medium |
| Swish | (−0.28, ∞) | Yes | No | Medium |
| Mish | (−0.31, ∞) | Yes | No | High |

### 9. Advanced Topics

1. **Parametric Activations**
    
    Learn negative slopes or scale factors (PReLU, Parametric Swish).
    
2. **Self-Normalizing Networks**
    
    SELU + special init + no dropout maintain zero mean/unit variance.
    
3. **RNN Gates**
    
    LSTMs/GRUs use sigmoid and tanh in gates to control memory flow.
    
4. **Dynamic Activations**
    
    AutoML techniques evolve novel activation shapes for specific tasks.
    

### 10. Practical Guidelines

1. Default to **ReLU** in hidden layers.
2. If neurons die (always zero), switch to **LeakyReLU** or **PReLU**.
3. For very deep MLPs, try **ELU** or **SELU** (with proper init).
4. In NLP/Transformers, test **GELU** or **Swish**.
5. For mobile/quantized models, use **Hard-Swish** or **Hard-Sigmoid**.

---

## ReLU Activation

### 1. Definition and Formula

ReLU (Rectified Linear Unit) applies an element-wise threshold at zero, passing positive inputs unchanged and zeroing negatives.

```
def relu(x):
    return max(0, x)
```

Its derivative is simple:

```
def relu_grad(x):
    return 1 if x > 0 else 0
```

### 2. Intuition and Role

ReLU serves as a gate that:

- Suppresses negative activations, creating sparse representations.
- Keeps positive values linear, avoiding saturation.
- Mimics biological neurons that don’t fire below a threshold.

This simplicity lets deep networks learn faster by preserving gradients for active neurons.

### 3. Key Properties

| Property | Description |
| --- | --- |
| Range | [0, ∞) |
| Zero-Centered Output | No |
| Saturating Regions | No for x>0; completely shuts off x<0 |
| Computational Cost | Very low (single comparison and multiply) |
| Sparsity | Encourages many zeros → efficient representations |

### 4. TensorFlow Implementation

### Using built-in activation

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Using ReLU layer

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])
```

### 5. Advantages

- Enables deep models by preventing vanishing gradients for x>0.
- Computation is cheap and parallelizable.
- Implicit feature selection via sparsity (many zeros).

### 6. Disadvantages & Mitigations

- **Dying ReLU**: neurons stuck at zero if inputs always negative.
    - Mitigation: use **LeakyReLU** or **PReLU** to allow a small gradient when x<0.
- **Not zero-centered**: outputs in [0,∞) can slow convergence.
    - Mitigation: apply **Batch Normalization** before or after ReLU.

### 7. Common Variants

| Variant | Formula | Notes |
| --- | --- | --- |
| LeakyReLU | `max(α·x, x)` where α≈0.01 | Prevents neurons from dying by non-zero slope |
| PReLU | `max(α·x, x)` with α learned during training | Adapts negative slope per channel or layer |
| ELU | `x if x>0 else α*(exp(x)-1)` | Smooth negative region, zero-centered |

### 8. Visualization

Plotting ReLU and its gradient helps internalize behavior:

```python
import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(-5, 5, 200)
ys = np.maximum(0, xs)
grads = (xs > 0).astype(float)

plt.figure(figsize=(8,4))
plt.plot(xs, ys, label='ReLU')
plt.plot(xs, grads, '--', label='Gradient')
plt.legend()
plt.title('ReLU and Its Gradient')
plt.show()
```

### 9. Best Practices

- Initialize weights with He initialization (`he_normal` or `he_uniform`).
- Combine with BatchNorm to mitigate non-zero-centered outputs.
- Use an appropriate learning rate (start around 1e-3 for Adam, 1e-2 for SGD+momentum).
- Monitor activation histograms in TensorBoard to detect dead neurons.

### 10. Debugging Tips

- If many neurons output zero for all inputs, reduce learning rate or switch to LeakyReLU.
- Inspect gradient norms per layer—zero gradients indicate inactive neurons.
- Try a small random input batch and print intermediate activations to see saturation.

### 11. Interview Tip

Be ready to:

- Derive ReLU’s gradient and explain why it avoids vanishing when x>0.
- Discuss the dying ReLU problem and propose alternatives.
- Explain how He initialization pairs with ReLU mathematically.

---

## Multiclass Classification

### 1. Problem Definition

Multiclass classification is the task of assigning each input to one of *K* mutually exclusive classes.

Unlike binary classification, where outputs are 0 or 1, here the model must discriminate among three or more categories.

Common examples include digit recognition (0–9), object classification (dog, cat, bird, …), or topic labeling (sports, politics, tech, …).

### 2. Label Encoding Strategies

Before feeding labels into a model, you must encode them numerically. Two main approaches:

- Integer encoding
    
    ```
    y = [2, 0, 3, 1, 2]  # values in range [0, K-1]
    ```
    
    Use with `SparseCategoricalCrossentropy`.
    
- One-hot encoding
    
    ```
    y = [[0,0,1,0],
         [1,0,0,0],
         [0,0,0,1],
         …]
    ```
    
    Use with `CategoricalCrossentropy`.
    

### 3. Softmax Activation

Softmax turns raw logits into a probability distribution over *K* classes:

```
softmax(z)i = exp(zi) / sum_j exp(zj)
```

- Ensures outputs sum to 1 and each lies in (0,1).
- Greater separation when one logit is higher than the rest.
- Numerically stabilized via subtracting max-logit:

```
shifted_z = z - max(z)
exp_z = exp(shifted_z)
softmax(z)i = exp_z[i] / sum(exp_z)
```

### 4. Loss Functions

### Categorical Crossentropy (one-hot)

```
loss = - sum_i y_true[i] * log(y_pred[i])
```

### Sparse Categorical Crossentropy (integer labels)

```
loss = - log(y_pred[label_index])
```

- Set `from_logits=True` if you pass raw logits instead of softmax outputs.
- Supports **label smoothing** to prevent overconfidence:

```python
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0.1
)
```

### 5. TensorFlow Implementation

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(K)  # raw logits
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
```

- If you add `activation='softmax'` in the last layer, use `from_logits=False`.
- Swap to `CategoricalCrossentropy` if you one-hot encode `y_train`.

### 6. Evaluation Metrics

Beyond overall accuracy, consider:

- **Top-K Accuracy**
    
    ```python
    tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    ```
    
- **Confusion Matrix**
    
    ```python
    tf.math.confusion_matrix(y_true, y_pred_labels)
    ```
    
- **Precision, Recall, F1-score** (per class)

Visualize these in TensorBoard or with `sklearn.metrics.classification_report`.

### 7. Common Pitfalls & Tips

- **Logit vs Probability** mismatch: ensure `from_logits` aligns with your model output.
- **Class Imbalance**: apply class weights or focal loss to emphasize rare classes.
- **Numerical Stability**: always implement softmax with log-sum-exp trick to avoid overflow.

### 8. Advanced Topics

- **Hierarchical Classification**: break classes into sub-groups (e.g., animal → mammal → dog).
- **Label Smoothing**: prevents model from becoming overconfident, improves generalization.
- **Focal Loss**: focuses learning on hard, misclassified examples.
- **Ensembles & Distillation**: combine multiple multiclass models or distill into a smaller network.

### 9. Interview Tips

- Derive the gradient of softmax + crossentropy in one step to show efficiency gains.
- Explain why softmax outputs sum to one and how it models categorical distributions.
- Discuss trade-offs between integer vs one-hot labels and their memory/computation costs.

---

## Softmax Activation

### Definition and Core Intuition

Softmax converts a vector of raw scores (logits) into a probability distribution over *K* classes.

Each output reflects how likely its class is relative to the others.

It’s used when you need mutually exclusive class probabilities that sum to 1.

### Mathematical Formula

```python
# z: shape (K, )
exp_z = [exp(zi) for zi in z]
sum_exp = sum(exp_z)
softmax = [exp_z[i] / sum_exp for i in range(K)]
```

- `z[i]`: raw score (logit) for class *i*
- `softmax[i]`: normalized probability for class *i*

### Numerical Stability Trick

Subtracting the maximum logit prevents overflow in `exp`:

```python
# stabilized softmax
z_max = max(z)
shifted_z = [zi - z_max for zi in z]
exp_z = [exp(zi) for zi in shifted_z]
sum_exp = sum(exp_z)
softmax = [exp_z[i] / sum_exp for i in range(K)]
```

This ensures no exponent exceeds the floating-point range.

### Key Properties

- Outputs sum to 1, forming a valid probability distribution.
- Amplifies relative differences: a small logit advantage yields higher probability.
- Differentiable everywhere, enabling gradient-based training.

### NumPy Implementation

```python
import numpy as np

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Example
logits = np.array([[2.0, 1.0, 0.1]])
probs = softmax(logits)
```

### TensorFlow Implementation

```python
import tensorflow as tf

logits = tf.constant([[2.0, 1.0, 0.1]])
probs = tf.nn.softmax(logits, axis=-1)
```

Use `from_logits=True` in loss functions when passing raw logits directly.

### Geometric Insight

Softmax maps the logit vector onto the *K*-dimensional simplex.

Imagine each class probability as a corner of a triangle (for *K*=3).

The model’s raw scores push you toward one corner; softmax normalizes that direction into a point on the simplex.

### Real-World Use Cases

- Multi-class image classification (e.g., CIFAR-10, ImageNet).
- Language modeling: predicting next word among a large vocabulary.
- Reinforcement learning: computing a stochastic policy over discrete actions.

### Practice Problems

- Implement softmax and cross-entropy loss from scratch on a small 3-class dataset.
- Plot how softmax outputs change as you vary one logit while freezing others.
- Compare training a TensorFlow model with and without `from_logits=True` in your loss.

### Common Pitfalls & Interview Tips

- Forgetting numerical stability leads to `inf` or `NaN`.
- Using softmax for non-mutually exclusive labels (use sigmoid instead).
- Be ready to derive the gradient of softmax combined with cross-entropy in one step.

---

## Neural Network with Softmax Output

### 1. Problem Setup

When you have *K* mutually exclusive classes, you use a softmax output layer to convert raw network scores into class probabilities.

- Input: feature vector *x*
- Output: probability distribution *p* over K classes
- Prediction: class with highest probability

### 2. Mathematical Model

### 2.1 Forward Pass Through Hidden Layers

For each layer *l* (excluding output):

```python
Zl = Wl · Al-1 + bl
Al = activation(Zl)            # e.g., ReLU, tanh
```

- `Wl` shape: (n_l, n_l-1)
- `bl` shape: (n_l, 1)
- `Al-1` shape: (n_l-1, m)

### 2.2 Output Layer with Softmax

```python
ZL = WL · A L-1 + bL           # raw logits, shape (K, m)
expZ = exp(ZL - max(ZL, axis=0))
softmax = expZ / sum(expZ, axis=0, keepdims=True)
```

- `softmax` shape: (K, m)
- Each column sums to 1 and lies in (0,1)

### 3. Loss Function: Categorical Cross-Entropy

For one-hot labels *Y*:

```python
loss = - sum(Y * log(softmax)) / m
```

For integer labels (sparse):

```python
loss = - sum(log(softmax[label_index, i])) / m
```

- Use `from_logits=False` when passing the softmax output to the loss.
- Use `from_logits=True` if you pass raw `ZL` to a specialized loss that applies softmax internally.

### 4. NumPy Implementation (Forward + Loss)

```python
import numpy as np

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def compute_loss(A_softmax, Y_onehot):
    m = Y_onehot.shape[1]
    loss = -np.sum(Y_onehot * np.log(A_softmax)) / m
    return loss

# Example shapes
m = 5          # number of examples
K = 3          # number of classes
ZL = np.random.randn(K, m)
Y_onehot = np.eye(K)[:, :m]

A_softmax = softmax(ZL)
loss = compute_loss(A_softmax, Y_onehot)
print("Softmax output shape:", A_softmax.shape)
print("Loss:", loss)
```

### 5. TensorFlow/Keras Implementation

### 5.1 Model Definition

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(K)                   # raw logits
])
```

### 5.2 Compilation

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

### 5.3 Training

```python
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=10,
          batch_size=32)
```

- If you add `activation='softmax'` in the last layer, set `from_logits=False`.

### 6. Backward Pass (Gradients)

The gradient of the loss wrt logits `ZL` is:

```python
dZL = A_softmax - Y_onehot      # shape (K, m) for one-hot labels
```

Subsequent parameter gradients:

```python
dWL = (1/m) · dZL · A L-1ᵀ
dbL = (1/m) · sum(dZL, axis=1, keepdims=True)
```

### 7. Numerical Stability and Best Practices

- **Subtract max logit per column** before exponentiating to avoid overflow.
- **Clip predictions** inside `log()` to [ε, 1] to avoid `log(0)`.
- **Batch Normalization** before or after softmax isn’t common—normalize hidden layers instead.
- Use **label smoothing** to improve generalization:

```python
tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
```

### 8. Advanced Variants

- **Temperature Scaling**
    
    ```python
    softmax(z / T)
    ```
    
    Higher *T* yields softer distributions; lower *T* sharpens.
    
- **Sparsemax**
    
    Produces sparse probability vectors, zeroing out small scores.
    
- **Focal Loss**
    
    Focuses training on hard examples by down-weighting well-classified ones.
    

### 9. Practice Problems

1. Implement a two-layer network with softmax output in NumPy and train on a toy 3-class dataset (`sklearn.datasets.make_classification`).
2. Compare training with raw logits + `from_logits=True` vs explicit softmax + `from_logits=False`.
3. Visualize how temperature affects output distributions for a fixed set of logits.

### 10. Common Pitfalls & Interview Tips

- Mixing up `from_logits` flag results in wrong gradients and poor training.
- Expect small numerical errors if you don’t stabilize softmax.
- Be prepared to derive why the gradient simplifies to `A_softmax - Y_onehot`.

---

## Improved Softmax Implementation in NumPy

### 1. Key Improvements

- Numerical stability by subtracting the maximum logit per sample
- Full batch‐wise vectorization
- Optional gradient/Jacobian‐vector product
- Clear type hints and docstrings
- Unit tests covering shapes, stability, and gradient correctness

### 2. NumPy Implementation

```python
import numpy as np
from typing import Tuple

def softmax(Z: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the softmax of each slice along the specified axis.

    Parameters
    ----------
    Z : np.ndarray
        Input logits, shape (..., batch_size).
    axis : int
        Axis along which to apply softmax (default=0).

    Returns
    -------
    np.ndarray
        Softmax probabilities, same shape as Z.
    """
    # 1. Subtract max for numerical stability
    Z_max = np.max(Z, axis=axis, keepdims=True)
    Z_stable = Z - Z_max

    # 2. Exponentiate and normalize
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=axis, keepdims=True)

def softmax_gradient(softmax_out: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """
    Compute gradient of loss w.r.t. logits Z given upstream gradient dA.

    Parameters
    ----------
    softmax_out : np.ndarray
        Softmax output, shape (K, m).
    dA : np.ndarray
        Upstream gradient w.r.t. softmax output, same shape as softmax_out.

    Returns
    -------
    np.ndarray
        Gradient w.r.t. input logits Z, same shape as input.
    """
    # For each sample column: dZ = S * (dA - sum(S * dA))
    dot = np.sum(softmax_out * dA, axis=0, keepdims=True)
    return softmax_out * (dA - dot)
```

### 3. Usage Example

```python
# Sample logits for 3 classes × 4 samples
Z = np.array([[2.0, 1.0, 0.1, -1.2],
              [1.0, 0.5, 0.3,  0.0],
              [0.1, 0.2, 0.4,  1.0]])

# Forward pass
A = softmax(Z, axis=0)
print("Softmax probabilities:\n", A)

# Upstream gradient (e.g., dL/dA = 1 for demonstration)
dA = np.ones_like(A)
dZ = softmax_gradient(A, dA)
print("Gradient w.r.t logits:\n", dZ)
```

### 4. Unit Tests

```python
def test_softmax_shape_and_sum():
    rng = np.random.RandomState(0)
    Z = rng.randn(5, 10)
    A = softmax(Z, axis=0)
    assert A.shape == Z.shape
    assert np.allclose(np.sum(A, axis=0), 1)

def test_softmax_numerical_stability():
    Z = np.array([[1000, -1000], [2000, -2000]])
    A = softmax(Z, axis=0)
    assert np.all(np.isfinite(A))
    assert np.allclose(np.sum(A, axis=0), 1)

def test_gradient_vs_numeric():
    rng = np.random.RandomState(1)
    Z = rng.randn(3, 4)
    A = softmax(Z, axis=0)
    dA = rng.randn(3, 4)
    dZ_auto = softmax_gradient(A, dA)

    # Numerical approximation
    eps = 1e-7
    dZ_num = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Zp = Z.copy(); Zp[i, j] += eps
            Zm = Z.copy(); Zm[i, j] -= eps
            Ap = softmax(Zp, axis=0)
            Am = softmax(Zm, axis=0)
            dZ_num[i, j] = np.sum((Ap - Am) * dA) / (2 * eps)

    assert np.allclose(dZ_auto, dZ_num, atol=1e-5)

if __name__ == "__main__":
    test_softmax_shape_and_sum()
    test_softmax_numerical_stability()
    test_gradient_vs_numeric()
    print("All softmax tests passed.")
```

---

## Advanced Optimization Techniques for Neural Networks

### 1. When to Reach for Advanced Optimizers

Basic optimizers like SGD with momentum and Adam often get you off the ground. But as you scale to deeper models or tighter accuracy margins, you’ll encounter plateaus, slow convergence, or generalization gaps. Advanced optimization techniques can help you:

- Escape sharp minima
- Adapt faster to changing curvature
- Stabilize training in large-batch or distributed settings
- Improve generalization through controlled noise

### 2. Adaptive Gradient Methods Beyond Adam

| Optimizer | Key Idea | Pros | Cons |
| --- | --- | --- | --- |
| AdamW | Decouples weight decay from gradient step | Better generalization | Requires tuning weight decay |
| RAdam | Rectifies Adam’s variance during warm-up | More stable early training | Slightly more compute |
| AdaBelief | Adapts step by “belief” in gradient change | Sharper convergence, robust | Extra hyperparameter (ε) |
| Nadam | Adam + Nesterov momentum | Smoother and faster updates | More moving averages to track |

### 3. Second-Order & Quasi-Newton Methods

1. **L-BFGS**
    - Builds a low-rank Hessian approximation
    - Suited for smaller models or fine-tuning
2. **K-FAC (Kronecker-Factored Approximate Curvature)**
    - Approximates natural gradient via layerwise Kronecker factors
    - Speeds up convergence on deep nets with moderate overhead
3. **Natural Gradient Descent**
    - Moves parameters in the direction of steepest descent under the Fisher metric
    - Often used in reinforcement learning (e.g., TRPO)

### 4. Learning Rate Schedules & Warm Restarts

- **Cosine Annealing**
    
    ```python
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t / T))
    ```
    
- **Cyclical Learning Rates**
    - Triangle or sawtooth patterns between two bounds
- **Warm Restarts (SGDR)**
    - Reset learning rate periodically to jump out of local minima
- **OneCycle Policy**
    - Ramp up then down in a single cycle for fast, high-accuracy training

### 5. Injecting Controlled Noise & Regularization

- **Gradient Noise Injection**
    
    ```python
    g_t ← g_t + Normal(0, σ² / (1 + t)^γ)
    ```
    
- **Label Smoothing**
    - Prevents over-confidence in softmax classifiers
- **Stochastic Depth & Dropconnect**
    - Randomly drop entire layers or weights during training

### 6. System-Level & Mixed-Precision Tricks

- **Mixed-Precision Training**
    - Use FP16 for forward/backward, FP32 for weight updates
- **Gradient Checkpointing**
    - Save memory by recomputing activations
- **Large-Batch Training**
    - Scale LR linearly with batch size + warmup
- **Distributed Optimizers**
    - LAMB, ZeRO for efficient multi-GPU/TPU scaling

### 7. Code Snippets

### 7.1 AdamW in NumPy

```python
def adamw_update(w, m, v, grad, t,
                 lr=1e-3, β1=0.9, β2=0.999, ε=1e-8, wd=1e-2):
    m = β1 * m + (1 - β1) * grad
    v = β2 * v + (1 - β2) * (grad ** 2)
    m_hat = m / (1 - β1**t)
    v_hat = v / (1 - β2**t)
    w -= lr * (m_hat / (np.sqrt(v_hat) + ε) + wd * w)
    return w, m, v
```

### 7.2 RAdam Warm-Up Correction

```python
def radam_update(w, m, v, grad, t, lr=1e-3, β1=0.9, β2=0.999, ε=1e-8):
    ρ_inf = 2/(1 - β2) - 1
    m = β1*m + (1-β1)*grad
    v = β2*v + (1-β2)*(grad**2)
    ρ_t = ρ_inf - 2*t*β2**t/(1-β2**t)
    if ρ_t > 4:
        r_t = np.sqrt(((ρ_t-4)*(ρ_t-2)*ρ_inf) / ((ρ_inf-4)*(ρ_inf-2)*ρ_t))
        w -= lr * r_t * m / (np.sqrt(v) + ε)
    else:
        w -= lr * m
    return w, m, v
```

### 8. Practice Problems

1. Implement K-FAC for a two-layer MLP and compare iteration count vs. AdamW on MNIST.
2. Benchmark one-cycle vs. cosine annealing on CIFAR-10 with ResNet-18.
3. Add gradient noise to SGD and visualize loss landscapes traversed.

---

## Additional Layer Types

### Overview

Deep networks leverage a variety of layer types to capture spatial patterns, temporal dependencies, context, and to regularize training. Adding the right layers can transform a simple MLP into a state-of-the-art vision, sequence, or graph model.

### Layer Summary Table

| Layer Type | Purpose | Typical Use Case | Keras Snippet |
| --- | --- | --- | --- |
| Convolutional | Local feature extraction | Image recognition, segmentation | `Conv2D(filters, kernel_size)` |
| Pooling | Spatial down-sampling | Reducing resolution, invariance | `MaxPool2D(pool_size)` |
| Batch Normalization | Activation distribution stabilization | Faster convergence, deeper nets | `BatchNormalization()` |
| Dropout | Random neuron masking | Regularization across domains | `Dropout(rate)` |
| Dense (Fully Connected) | Global feature mixing | Classification, regression heads | `Dense(units, activation)` |
| LSTM / GRU | Sequential memory | Time series, language modeling | `LSTM(units)` / `GRU(units)` |
| Attention | Context-aware weighting | Transformers, seq2seq, vision | `MultiHeadAttention(num_heads, key_dim)` |
| Embedding | Discrete → continuous vector mapping | NLP, categorical features | `Embedding(input_dim, output_dim)` |
| Layer Normalization | Per-sample normalization | RNNs, Transformers | `LayerNormalization()` |
| Graph Convolutional | Neighborhood aggregation | Graph-structured data | `GCNConv(out_channels)` (PyG) |

### 1. Convolutional Layers

Convolutional layers slide learnable kernels across spatial dimensions, sharing parameters to detect patterns.

- Kernel size (k×k), stride, padding control receptive field.
- Output shape formula for `Conv2D`:
    
    ```
    H_out = (H_in + 2*pad - k) // stride + 1
    W_out = (W_in + 2*pad - k) // stride + 1
    ```
    
- NumPy snippet for one 2D convolution (single filter):
    
    ```python
    def conv2d_single(X, W, stride=1, pad=0):
        # X: (h, w), W: (kh, kw)
        h, w = X.shape; kh, kw = W.shape
        X_p = np.pad(X, pad)
        out_h = (h + 2*pad - kh)//stride + 1
        out_w = (w + 2*pad - kw)//stride + 1
        Y = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                patch = X_p[i*stride:i*stride+kh, j*stride:j*stride+kw]
                Y[i, j] = np.sum(patch * W)
        return Y
    ```
    

### 2. Pooling Layers

Pooling reduces spatial dimension and builds translational invariance.

- Max Pooling picks the largest value in each window.
- Average Pooling computes the mean.
- Global Pooling collapses entire spatial map:
    
    ```python
    # Keras
    layers.GlobalMaxPool2D()
    layers.GlobalAveragePool2D()
    ```
    

### 3. Normalization Layers

Normalization stabilizes activations and gradients.

- **BatchNormalization** computes per-batch mean/variance:
    
    ```python
    layers.BatchNormalization()
    ```
    
- **LayerNormalization** normalizes across features for each sample:
    
    ```python
    layers.LayerNormalization()
    ```
    
- Benefits include higher learning rates and reduced covariate shift.

### 4. Regularization Layers

Prevent overfitting by injecting noise or dropping information.

- **Dropout:** zeros random activations:
    
    ```python
    layers.Dropout(0.5)
    ```
    
- **SpatialDropout2D:** drops entire feature maps in conv nets:
    
    ```python
    layers.SpatialDropout2D(0.3)
    ```
    
- **AlphaDropout:** maintains mean/variance for SELU activations.

### 5. Recurrent & Sequence Layers

Capture temporal or ordered dependencies.

- **LSTM:** Long short-term memory with gates controlling flow:
    
    ```python
    layers.LSTM(units, return_sequences=True)
    ```
    
- **GRU:** Gated recurrent unit, simpler than LSTM:
    
    ```python
    layers.GRU(units)
    ```
    
- **Bidirectional:** wrap an RNN to process forward and backward:
    
    ```python
    layers.Bidirectional(layers.LSTM(units))
    ```
    

### 6. Attention & Transformer Blocks

Learn to weigh input elements contextually.

- **MultiHeadAttention:** projects queries, keys, values:
    
    ```python
    attn = layers.MultiHeadAttention(num_heads=8, key_dim=64)
    output = attn(query, value, key)
    ```
    
- **Transformer Encoder Block** combines attention, normalization, and feed-forward:
    
    ```python
    def transformer_block(x):
        attn_out = attn(x, x, x)
        x1 = layers.Add()([x, attn_out])
        x2 = layers.LayerNormalization()(x1)
        ff = layers.Dense(ff_dim, activation='relu')(x2)
        x3 = layers.Add()([x2, ff])
        return layers.LayerNormalization()(x3)
    ```
    

### 7. Embedding Layers

Map discrete tokens into dense vectors.

- Input shape: `(batch_size, seq_len)`
- Output shape: `(batch_size, seq_len, embedding_dim)`

```python
layers.Embedding(input_dim=vocab_size, output_dim=128)
```

Embeddings can be pretrained (e.g., Word2Vec, GloVe) or learned end-to-end.

### 8. Graph Convolutional Layers

Generalize convolution to graph structures.

- Aggregate neighbor features via adjacency:
    
    ```python
    from torch_geometric.nn import GCNConv
    conv = GCNConv(in_channels, out_channels)
    x = conv(x, edge_index)
    ```
    
- Useful for social networks, molecules, recommendation systems.

### 9. Practice Problems

1. Build a small CNN combining `Conv2D`, `BatchNormalization`, `MaxPool2D`, and `Dropout`. Train on CIFAR-10.
2. Implement a simple Transformer encoder block from scratch in NumPy.
3. Apply a GCN to the CORA citation network and compare node classification accuracy with an MLP.

### 10. Interview Tips

- Be ready to explain how each layer affects gradient flow and representational power.
- Derive the forward and backward passes for BatchNormalization and Dropout.
- Discuss trade-offs: parameter sharing (CNN) vs. flexibility (Dense), memory footprint (RNN vs. Transformer).

---

## Derivatives and Backpropagation

### 1. Foundations of Derivatives

Derivatives measure how a function changes as its input changes. In neural networks, we use them to see how small changes in weights affect the loss.

- Single‐variable derivative
- Partial derivatives for multivariable functions
- Chain rule for compositions

### 1.1 Single‐Variable Derivative

The derivative of (f(x)) at (x) is the limit of its average rate of change:

```
f'(x) = lim_{h -> 0} (f(x + h) - f(x)) / h
```

Key differentiation rules:

```
# Power rule
d/dx [ x^n ] = n * x^(n-1)

# Sum rule
d/dx [ u + v ] = du/dx + dv/dx

# Product rule
d/dx [ u * v ] = u' * v + u * v'

# Quotient rule
d/dx [ u / v ] = (u' * v - u * v') / v^2

# Chain rule
d/dx [ u(v(x)) ] = u'(v(x)) * v'(x)
```

### 1.2 Partial Derivatives and Gradient

For a function of multiple variables (f(x,y, \dots)), the partial derivative w.r.t. one variable holds others constant:

```
∂f/∂x = lim_{h -> 0} (f(x + h, y) - f(x, y)) / h
```

The gradient is the vector of all partial derivatives:

```
grad f = [ ∂f/∂x, ∂f/∂y, … ]^T
```

### 2. Chain Rule in Neural Nets

Neural nets stack layers; each layer’s output becomes the next layer’s input. We apply the chain rule to propagate derivatives through this composition.

- **Scalar chain rule**
    
    ```
    If y = f(x) and L = g(y), then
    dL/dx = (dL/dy) * (dy/dx)
    ```
    
- **Vector chain rule**For a vector‐valued intermediate (y):
    
    ```
    ∂L/∂x = (∂y/∂x)^T · (∂L/∂y)
    ```
    

### 3. Computational Graph Perspective

Visualize every operation (addition, multiplication, activation) as a node in a graph.

- **Forward pass**: compute outputs and cache inputs/Zs
- **Backward pass**: start from loss node, apply chain rule at each node, propagate gradients backward

### 4. Backpropagation Algorithm

Backprop computes gradients for all weights/biases in a layerwise fashion.

1. **Forward pass**
    - For ℓ=1…L:
        
        ```
        Z[ℓ] = W[ℓ] · A[ℓ-1] + b[ℓ]
        A[ℓ] = activation(Z[ℓ])
        ```
        
    - Cache (A[ℓ-1], Z[ℓ], W[ℓ], b[ℓ])
2. **Initialize backward pass**
    - If using softmax + cross‐entropy:
        
        ```
        dZ[L] = A[L] - Y_one_hot
        ```
        
3. **Iterate ℓ=L…1**
    - Compute activation gradient:
        
        ```
        dA[ℓ] = from previous layer or dZ[ℓ+1] step
        dZ[ℓ] = dA[ℓ] ∘ activation_prime(Z[ℓ])
        ```
        
    - Compute parameter gradients:
        
        ```
        dW[ℓ] = (1/m) * dZ[ℓ] · A[ℓ-1]^T
        db[ℓ] = (1/m) * sum(dZ[ℓ], axis=1, keepdims=True)
        ```
        
    - Propagate to previous activations:
        
        ```
        dA[ℓ-1] = W[ℓ]^T · dZ[ℓ]
        ```
        

### 5. Activation Derivatives

Different activations have simple elementwise derivatives:

```
# ReLU
activation_prime(Z) = (Z > 0 ? 1 : 0)

# Sigmoid
σ(Z) = 1 / (1 + exp(-Z))
activation_prime(Z) = σ(Z) * (1 - σ(Z))

# Tanh
activation_prime(Z) = 1 - tanh(Z)^2
```

### 6. End‐to‐End NumPy Implementation

```python
import numpy as np

def forward_pass(X, parameters):
    caches = []
    A = X
    for l in range(1, L+1):
        W, b = parameters['W'+str(l)], parameters['b'+str(l)]
        Z = W.dot(A) + b
        if l < L:
            A = np.maximum(0, Z)        # ReLU
        else:
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / np.sum(expZ, axis=0, keepdims=True)  # Softmax
        caches.append((A, Z, W, b))
    return A, caches

def backward_pass(A_final, Y, caches):
    grads = {}
    m = Y.shape[1]
    # Initialize dZ for softmax + CE
    A_last, Z_last, _, _ = caches[-1]
    dZ = A_last - Y
    for l in reversed(range(1, L+1)):
        A_prev, Z_curr, W, b = caches[l-1] if l>1 else (X, None, None, None)
        grads['dW'+str(l)] = (1/m) * dZ.dot(A_prev.T)
        grads['db'+str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA_prev = W.T.dot(dZ)
            dZ = dA_prev * (Z_curr > 0)  # ReLU backward
    return grads
```

### 7. Gradient Checking

Validate your backprop by comparing analytical gradients to numerical approximations:

```python
def compute_numeric_gradient(f, params, eps=1e-7):
    num_grads = {}
    for key in params:
        grad = np.zeros_like(params[key])
        it = np.nditer(params[key], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = params[key][idx]
            params[key][idx] = orig + eps
            loss_plus = f(params)
            params[key][idx] = orig - eps
            loss_minus = f(params)
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            params[key][idx] = orig
            it.iternext()
        num_grads[key] = grad
    return num_grads
```

### 8. Extensions to Other Layers

Backprop follows the same chain‐rule pattern for:

- Convolutional layers
- Pooling layers (max/avg)
- BatchNorm and LayerNorm
- Dropout and regularization layers

Each has its own local derivative.

---