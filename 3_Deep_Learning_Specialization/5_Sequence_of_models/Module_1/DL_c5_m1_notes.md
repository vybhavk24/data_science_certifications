# DL_c5_m1

## Why sequence models?

### 1. Direct Definition

A sequence model is a type of neural network designed to process data where order matters. Instead of treating each input independently, it carries information forward in time (or position), enabling it to learn patterns that unfold sequentially.

### 2. Concept Intuition

Sequence models shine when your data has a natural order. Examples include:

- Text: The meaning of a sentence depends on the words that came before.
- Time series: Stock prices today depend on yesterday’s trends.
- Speech/audio: The sound at each moment shapes what comes next.
- Video: Each frame relates to the previous frames.

Why it matters

- Captures context: Maintaining a “memory” of past inputs lets the model understand dependencies.
- Variable lengths: You can feed in sequences of different lengths without retraining the entire network.
- Shared parameters: The same weights slide over time, reducing the number of parameters and improving generalization.

### 3. Mathematical Breakdown

At each time step t, an RNN cell updates its hidden state a[t] and produces an output ŷ[t]. The core equations are:

```python
# hidden state update
a[t] = activation(Wax @ x[t] + Waa @ a[t-1] + ba)

# output computation
ŷ[t] = softmax(Wya @ a[t] + by)
```

Variables

- x[t]: input vector at time t
- a[t]: hidden “memory” at time t
- ŷ[t]: predicted output (e.g., next word probabilities)
- Wax, Waa, Wya: weight matrices
- ba, by: bias vectors
- activation: elementwise nonlinearity (e.g., tanh or ReLU)

Why it works

- Wax embeds new information.
- Waa carries forward context.
- ba and by let the network shift decision boundaries.

### 4. Code & Practical Application

Below is a minimal NumPy implementation of a single RNN cell forward pass. Feed in a toy sequence and observe the hidden states.

```python
import numpy as np

def rnn_cell_forward(x_t, a_prev, Wax, Waa, ba):
    """
    x_t: (n_x,) input at time t
    a_prev: (n_a,) previous hidden state
    Returns: a_next (n_a,)
    """
    z = Wax.dot(x_t) + Waa.dot(a_prev) + ba
    a_next = np.tanh(z)
    return a_next

# toy dimensions
n_x, n_a = 3, 5
Wax = np.random.randn(n_a, n_x)
Waa = np.random.randn(n_a, n_a)
ba = np.zeros((n_a,))

# toy sequence: length 4
sequence = [np.random.randn(n_x) for _ in range(4)]
a = np.zeros((n_a,))

# forward through time
states = []
for x_t in sequence:
    a = rnn_cell_forward(x_t, a, Wax, Waa, ba)
    states.append(a)

print("Hidden states shape:", np.stack(states).shape)
```

Real-world context

- In TensorFlow / PyTorch you’d wrap this logic in a loop or use built-in `tf.keras.layers.SimpleRNN` or `nn.RNN`.
- During training, you apply backpropagation through time (BPTT) to update Wax, Waa, ba.

### 5. Visualization / Geometry

Think of the hidden state `a[t]` as a point moving through an nₐ-dimensional space. Each new input x[t]:

1. Applies an affine transformation (`Wax @ x[t] + Waa @ a[t-1] + ba`)
2. Squashes the result into the “memory manifold” via tanh
3. Moves the point to a new location that reflects both past and present

ASCII sketch for a 2D hidden state:

```
time t-1:    ● a[t-1]
             \
              \  -- Wax @ x[t]
               \
time t:     ● a[t]
```

As you train, the network learns a pathway through this space that separates sequence classes or predicts next values.

### 6. Common Pitfalls & Tips

- Ignoring chronology: Shuffling sequence order destroys temporal dependencies.
- Vanishing/exploding gradients: Long sequences can cause small gradients to vanish or large ones to blow up; you’ll address this with gating (LSTM/GRU) or gradient clipping.
- Padding & masking: When batching variable-length sequences, pad shorter ones and mask the padded steps so they don’t affect updates.
- Overfitting: Sequence models can overfit small datasets—use dropout between time steps or L2 regularization.

### 7. Interview-Ready Insights

- Why use shared weights in time? It reduces parameters and enforces time invariance.
- Explain “hidden state” in your own words: It’s the network’s evolving summary of all prior inputs.
- Trade-offs vs. feed-forward nets: RNNs capture order but are harder to train; feed-forward nets are simpler but oblivious to sequence.
- Real use-cases: speech recognition, machine translation, time-series forecasting, music generation.

### 8. Practice Exercises

1. Implement a sequence sum RNN:
    - Given a sequence of scalars `[x1, x2, …, xT]`, train an RNN to output the cumulative sum at each time.
    - Hint: Treat `x[t]` and `a[t-1]` as 1-dim vectors; mean-squared loss on `ŷ[t]`.
2. Character prediction toy:
    - Dataset: “hellohello…” repeated.
    - Task: Given one-hot of each character, predict the next one.
    - Walkthrough: Start with NumPy RNN cell, then upgrade to `tf.keras.layers.SimpleRNN` for training.

---

## Notation

### 1. Direct Definition

Notation in sequence models refers to the symbols, indices, and dimensional conventions used to describe inputs, hidden states, outputs, parameters, and sequence lengths. Having a clear, consistent notation ensures you and others can read formulas, implement code, and communicate ideas without mismatch or confusion.

### 2. Concept Intuition

Why notation matters:

- Clarity: You instantly know whether you’re talking about time step t or t−1, a vector or a matrix.
- Consistency: When you see `W_aa` you know it’s a weight matrix connecting previous hidden states to the next.
- Debugging: Shape mismatches and index errors vanish when you’ve mapped every symbol to a clear dimension.
- Collaboration: Interviewers and teammates immediately grasp your design when notation aligns with industry norms.

### 3. Notation Cheat-Sheet

| Symbol | Description | Shape |
| --- | --- | --- |
| x<sup>⟨t⟩</sup> | input vector at time step t | (nₓ,) |
| a<sup>⟨t⟩</sup> | hidden-state (memory) at time step t | (nₐ,) |
| ŷ<sup>⟨t⟩</sup> | predicted output at time step t | (n_y,) |
| y<sup>⟨t⟩</sup> | true label/output at time step t | (n_y,) |
| Wₐₓ | weights from input x to hidden state | (nₐ, nₓ) |
| Wₐₐ | weights from previous state to current state | (nₐ, nₐ) |
| W_yₐ | weights from hidden state to output | (n_y, nₐ) |
| bₐ | bias for hidden state | (nₐ,) |
| b_y | bias for output | (n_y,) |
| Tₓ | length of input sequence | scalar |
| T_y | length of output sequence | scalar |
| nₓ | dimension of each input vector | scalar |
| nₐ | number of units in hidden state | scalar |
| n_y | dimension of each output vector | scalar |

### 4. Mathematical Breakdown

At each time step t:

```python
z = W_ax @ x      + W_aa @ a_prev + b_a
a   = tanh(z)
ŷ   = softmax(W_ya @ a + b_y)
```

Here:

- `x` is x<sup>⟨t⟩</sup> ∈ ℝⁿₓ
- `a_prev` is a<sup>⟨t−1⟩</sup> ∈ ℝⁿₐ
- `a` becomes a<sup>⟨t⟩</sup>
- `ŷ` is ŷ<sup>⟨t⟩</sup> ∈ ℝⁿʸ

Each matrix-vector product’s dimensions align because of our notation table.

### 5. Code & Practical Application

Initialize parameters using the notation above:

```python
import numpy as np

def initialize_rnn(n_x, n_a, n_y):
    np.random.seed(1)
    W_ax = np.random.randn(n_a, n_x) * 0.01
    W_aa = np.random.randn(n_a, n_a) * 0.01
    W_ya = np.random.randn(n_y, n_a) * 0.01
    b_a  = np.zeros((n_a,))
    b_y  = np.zeros((n_y,))
    return { "W_ax": W_ax, "W_aa": W_aa,
             "W_ya": W_ya, "b_a": b_a, "b_y": b_y }

# Example dims
params = initialize_rnn(n_x=10, n_a=20, n_y=5)
for key in params:
    print(key, params[key].shape)
```

Running this you’ll see each weight and bias matches its intended shape.

### 6. Visualization / Geometry

ASCII sketch for two time steps:

```
          ┌──────────┐        ┌──────────┐
 x⟨t⟩ ──▶  │  RNN    │ ──▶ ŷ⟨t⟩
          │  cell   │
 a⟨t−1⟩──▶ │        │
          └──────────┘
               │
               ▼
             a⟨t⟩
```

Labels remind you which vectors flow where, and our notation assures dimensional compatibility at every arrow.

### 7. Common Pitfalls & Tips

- Mixing up superscripts ⟨t⟩ vs. subscripts: Always treat “t” as a time index, not a dimension.
- Shape mismatches: If you swap nₓ and nₐ, you’ll see dot-product errors immediately.
- Inconsistent naming: Choose either `W_ax` or `W_xa` and stick with it.
- Forgetting biases: `b_a` and `b_y` must match hidden/output dims exactly.

### 8. Interview-Ready Insights

- Articulate why Wₐₓ has shape (nₐ, nₓ): it projects input into the hidden-state space.
- Explain shared parameters: Wₐₓ and Wₐₐ apply at every time step, so they capture stationary dynamics.
- Discuss vectorized implementation: stack x<sup>⟨1…Tₓ⟩</sup> as a matrix X of shape (nₓ, Tₓ) and compute all z’s in one go if memory allows.

### 9. Practice Exercises

1. Given nₓ=8, nₐ=16, n_y=4, write out the shapes of all parameters and placeholders: x<sup>⟨t⟩</sup>, a<sup>⟨t⟩</sup>, Wₐₓ, Wₐₐ, W_yₐ, bₐ, b_y.
2. Implement a function `get_shapes(n_x, n_a, n_y)` that returns a dictionary of shapes for each symbol. Test it on random dims.
3. Draw a diagram of an RNN unrolled for Tₓ=3, labeling all x<sup>⟨t⟩</sup>, a<sup>⟨t⟩</sup>, and ŷ<sup>⟨t⟩</sup> to cement your notation skills.

---

## Recurrent neural network model

### 1. Direct Definition

A recurrent neural network (RNN) model is a neural architecture that processes a sequence of inputs x⟨1…Tₓ⟩ by maintaining and updating an internal hidden state a⟨t⟩ at each time step. The final or intermediate hidden states feed into outputs ŷ⟨t⟩, letting the network learn patterns that depend on order and context.

### 2. Concept Intuition

- Memory over time: At each step, the RNN “remembers” prior inputs via a hidden state vector.
- Stationary dynamics: The same weight matrices apply at every time step, so the network captures consistent sequential rules.
- Flexible lengths: You can feed in short or long sequences without changing the architecture.
- Causal processing: For tasks like language modeling or time-series forecasting, each prediction only uses past (and optionally future) context.

### 3. Mathematical Breakdown

At time step t:

```python
# Compute next hidden state
a[t] = tanh( W_ax @ x[t] + W_aa @ a[t-1] + b_a )

# Compute output prediction
ŷ[t] = softmax( W_ya @ a[t] + b_y )
```

Sequence-level loss (averaged over Tₓ steps and m examples):

```python
loss = -(1/m) * sum_i sum_{t=1}^{Tₓ}
           sum_{k=1}^{n_y}
             y_i[t][k] * log(ŷ_i[t][k])
```

Variables

- x[t] ∈ ℝⁿₓ: input at step t
- a[t] ∈ ℝⁿₐ: hidden state at step t
- ŷ[t] ∈ ℝⁿʸ: predicted output at step t
- W_ax ∈ ℝⁿₐ×ⁿₓ, W_aa ∈ ℝⁿₐ×ⁿₐ, W_ya ∈ ℝⁿʸ×ⁿₐ; b_a ∈ ℝⁿₐ, b_y ∈ ℝⁿʸ

### 4. Code & Practical Application

A minimal NumPy implementation of the full forward pass over a sequence:

```python
import numpy as np

def rnn_forward(X, a0, params):
    """
    X: list of T_x inputs, each shape (n_x,)
    a0: initial hidden state, shape (n_a,)
    params: dict with W_ax, W_aa, b_a, W_ya, b_y
    Returns:
      A: list of hidden states a[1…T_x]
      Y_hat: list of predictions ŷ[1…T_x]
    """
    W_ax, W_aa, b_a = params["W_ax"], params["W_aa"], params["b_a"]
    W_ya, b_y       = params["W_ya"], params["b_y"]

    a_prev = a0
    A, Y_hat = [], []

    for x_t in X:
        # hidden state update
        z = W_ax.dot(x_t) + W_aa.dot(a_prev) + b_a
        a_next = np.tanh(z)
        # output
        y_hat = np.exp(W_ya.dot(a_next) + b_y)
        y_hat /= np.sum(y_hat)
        # store
        A.append(a_next)
        Y_hat.append(y_hat)
        a_prev = a_next

    return A, Y_hat

# Example usage
n_x, n_a, n_y = 4, 6, 3
params = {
    "W_ax": np.random.randn(n_a, n_x)*0.01,
    "W_aa": np.random.randn(n_a, n_a)*0.01,
    "b_a":  np.zeros(n_a),
    "W_ya": np.random.randn(n_y, n_a)*0.01,
    "b_y":  np.zeros(n_y),
}
X = [np.random.randn(n_x) for _ in range(5)]
a0 = np.zeros(n_a)
A, Y_hat = rnn_forward(X, a0, params)
print("Forward pass outputs:", len(Y_hat), "steps.")

# In practice: wrap this in backpropagation through time (BPTT) and optimize parameters.
```

### 5. Visualization / Geometry

Unrolled RNN over Tₓ=3 time steps:

```
 x⟨1⟩ ─▶ [ RNN cell ] ─▶ a⟨1⟩ ─▶ ŷ⟨1⟩
            │
            ▼
 x⟨2⟩ ─▶ [ RNN cell ] ─▶ a⟨2⟩ ─▶ ŷ⟨2⟩
            │
            ▼
 x⟨3⟩ ─▶ [ RNN cell ] ─▶ a⟨3⟩ ─▶ ŷ⟨3⟩
```

Geometric view: each a⟨t⟩ is a point in ℝⁿₐ.

- W_ax projects x⟨t⟩ into hidden‐state space.
- W_aa rotates/scales the previous state.
- tanh nonlinearly squashes the result onto a hyper-surface, creating a trajectory through hidden‐state space that encodes the entire prefix x⟨1…t⟩.

### 6. Common Pitfalls & Tips

- Gradient issues: without gates, long dependencies suffer vanishing/exploding gradients.
- Time complexity: forward and backward pass scale with Tₓ; mini-batch across sequences requires padding and masking.
- Initialization: small random W’s prevent saturation of tanh; zero biases let the network freely discover dynamics.
- Overfitting: dropout between time steps and L2 regularization on W help generalize.

### 7. Interview-Ready Insights

- Explain why parameters are shared over time and how that enforces temporal stationarity.
- Describe how BPTT computes gradients through the unrolled graph and why vanishing gradients occur.
- Compare RNNs with LSTMs/GRUs: gating mechanisms solve gradient decay for long sequences.
- Discuss real applications: machine translation (sequence-to-sequence), speech recognition, anomaly detection in time series.

### 8. Practice Exercises

1. **Sequence Summation:** Using the `rnn_forward` above, implement `compute_loss(Y_hat, Y)` for mean-squared error and write gradient updates for W_ax, W_aa, W_ya via BPTT on a simple cumulative-sum task.
2. **Batch Processing:** Modify `rnn_forward` to accept a batch of sequences (padded) and apply masking so padded time steps don’t contribute to loss.
3. **Keras Quick Build:** Use `tf.keras.layers.SimpleRNN` to build an RNN model that classifies IMDB movie reviews. Compare training speed and accuracy with your NumPy model.

---

## Backpropagation Through Time (BPTT)

### 1. Direct Definition

Backpropagation Through Time (BPTT) is the algorithm that computes gradients of a recurrent neural network’s loss with respect to its parameters by unrolling the network across all time steps and applying the chain rule through this unrolled graph.

### 2. Concept Intuition

At each time step, an RNN cell’s hidden state depends not only on the current input but also on all previous inputs through the chain of hidden‐state updates.

- To update parameters (Wax, Waa, Wya, ba, by), we need to see how a small change in each weights affects the loss at every future time.
- BPTT “unrolls” the RNN for T steps, treats it like a very deep feed-forward net, and backpropagates errors from the end back to the beginning.
- This captures temporal credit assignment: it answers “Which time steps and which weights contributed most to the final error?”

### 3. Prerequisites & Refresher

Before diving in, ensure you’re comfortable with:

- Chain rule for multivariable functions
- Derivative of tanh: if a = tanh(z), then da/dz = 1 − a²
- Softmax + cross-entropy derivative: if ŷ = softmax(z), loss L = −∑y·log(ŷ), then dL/dz = ŷ − y

### 4. Mathematical Breakdown

Assume a sequence of length T, loss L is summed over all t:

```
L = ∑_{t=1…T} L[t]
```

At time step t:

```
a[t] = tanh(z[t]),   where z[t] = Wax·x[t] + Waa·a[t-1] + b_a
ŷ[t] = softmax(Wya·a[t] + b_y)
L[t] = − y[t]·log(ŷ[t])
```

We want ∂L/∂Waa. By chain rule:

```
∂L/∂Waa = ∑_{t=1…T} ∂L/∂Waa | at time t
```

For a single t:

```
dL/dWaa = (dL/da[t]) · (da[t]/dz[t]) · (dz[t]/dWaa)
```

Expand each term:

```python
# 1. dL/da[t]:
dL/da[t] = Wya^T · (ŷ[t] - y[t])   # gradient from output at step t

# 2. da[t]/dz[t]:
da[t]/dz[t] = 1 - a[t]**2         # elementwise

# 3. dz[t]/dWaa:
dz[t]/dWaa = a[t-1] (as outer-product)
```

But dL/da[t] itself depends on future steps, because a[t] influences a[t+1], a[t+2], …, a[T]. Therefore we define a “backpropagated error” δ[t] on the hidden state:

```python
# Initialize at final time
δ[T] = (Wya^T · (ŷ[T] - y[T])) * (1 - a[T]**2)

# Recursively for t = T-1 down to 1
δ[t] = (Wya^T · (ŷ[t] - y[t]) + Waa^T · δ[t+1]) * (1 - a[t]**2)
```

Then accumulate parameter gradients:

```python
# For Waa
dWaa = ∑_{t=1…T} δ[t] ⊗ a[t-1]    # outer product

# For Wax
dWax = ∑_{t=1…T} δ[t] ⊗ x[t]

# For b_a
db_a = ∑_{t=1…T} δ[t]

# For Wya
dWya = ∑_{t=1…T} (ŷ[t] - y[t]) ⊗ a[t]

# For b_y
db_y = ∑_{t=1…T} (ŷ[t] - y[t])
```

### 5. Code & Practical Application

Below is a NumPy sketch of BPTT gradients for one sequence:

```python
import numpy as np

def bptt(X, Y, params, A, Y_hat):
    """
    X: list of T inputs (n_x,)
    Y: list of T true outputs (n_y,)
    params: dict with Wax, Waa, Wya, b_a, b_y
    A: list of hidden states a[1..T]
    Y_hat: list of ŷ[1..T]
    Returns: gradients dict
    """
    Wax, Waa, Wya = params['Wax'], params['Waa'], params['Wya']
    b_a, b_y       = params['b_a'], params['b_y']
    n_a = Waa.shape[0]

    # initialize gradients
    dWax = np.zeros_like(Wax)
    dWaa = np.zeros_like(Waa)
    dWya = np.zeros_like(Wya)
    db_a = np.zeros_like(b_a)
    db_y = np.zeros_like(b_y)

    # initialize δ_next with zeros
    delta_next = np.zeros((n_a,))

    # iterate backwards through time
    for t in reversed(range(len(X))):
        # output layer gradient
        dy = Y_hat[t] - Y[t]
        dWya += np.outer(dy, A[t])
        db_y += dy

        # hidden layer error
        da = Wya.T.dot(dy) + Waa.T.dot(delta_next)
        dz = da * (1 - A[t]**2)

        # accumulate
        dWax += np.outer(dz, X[t])
        dWaa += np.outer(dz, A[t-1] if t > 0 else np.zeros(n_a))
        db_a  += dz

        # update δ_next
        delta_next = dz

    return {'dWax': dWax, 'dWaa': dWaa, 'dWya': dWya, 'db_a': db_a, 'db_y': db_y}
```

**Practical tips**

- Clip gradients element-wise (e.g., between −5 and 5) to avoid exploding gradients.
- Batch sequences with padding and masks; only backpropagate through actual time steps.

### 6. Visualization / Geometry

Unrolled RNN with gradient flow δ:

```
 x⟨1⟩ → [cell] → a⟨1⟩ → [cell] → … → a⟨T⟩
            ↖__ δ⟨1⟩  ↖__ δ⟨2⟩     ↖__ δ⟨T⟩
```

- δ[t] carries error signal back through the network in time.
- Each δ[t] is a vector in ℝⁿₐ that combines current output error plus “future” error from δ[t+1].

Geometrically, you’re tracing how a small weight perturbation ripples forward through time and back again to the loss, summing across all temporal paths.

### 7. Common Pitfalls & Tips

- Vanishing gradients: When ‖Waa‖ < 1, repeated multiplication drives δ[t]→0 for long ranges.
- Exploding gradients: When ‖Waa‖ > 1, δ[t] can blow up—use gradient clipping or spectral normalization.
- Time complexity: BPTT scales O(T·nₐ²) per sequence; truncated BPTT (backprop over K≪T steps) is a practical speedup.
- Memory: Storing all A[t] can be expensive; truncated BPTT or checkpointing can trade compute for memory.

### 8. Interview-Ready Insights

- Explain truncated BPTT: only backpropagate K steps to manage computation and memory.
- Describe why gates in LSTM/GRU stabilize gradient flow.
- Discuss how gradient clipping prevents NaN losses in long sequences.
- Illustrate temporal credit assignment: how does the network learn that an input at t=3 influenced the loss at t=10?

### 9. Practice Exercises

1. **Implement full training loop**
    - Use the `rnn_forward` and `bptt` functions.
    - Train on a cumulative sum sequence; monitor gradient norms, clip when they exceed a threshold.
2. **Truncated BPTT**
    - Modify `bptt` to only backpropagate K steps (e.g., K=5).
    - Compare training speed and convergence on a synthetic dataset of length T=100.
3. **Gradient Visualization**
    - For a toy RNN with nₐ=2, track δ[t] across time for a fixed sequence.
    - Plot each component of δ[t] vs. t to see vanishing/exploding behavior.

---

## Different Types of Recurrent Neural Networks

### 1. Direct Definition

Recurrent neural networks come in several architectural variants that trade off memory capacity, training stability, and context modeling.

Common types include:

- Vanilla (simple) RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional RNN
- Stacked (deep) RNN

### 2. Concept Intuition

Each variant addresses limitations of the basic RNN cell:

- Vanilla RNNs capture short-range patterns but suffer vanishing/exploding gradients over long sequences.
- LSTMs and GRUs introduce gating mechanisms to selectively remember or forget information, stabilizing gradient flow.
- Bidirectional RNNs process data in both forward and reverse order, giving each time step access to past and future context.
- Stacked RNNs (multiple layers) let the network learn hierarchical temporal features, from low-level patterns in layer 1 to high-level abstractions in deeper layers.

### 3. Mathematical Breakdown

### 3.1 Vanilla RNN Cell

```python
a[t] = tanh(W_ax @ x[t] + W_aa @ a[t-1] + b_a)
ŷ[t] = softmax(W_ya @ a[t] + b_y)
```

### 3.2 LSTM Cell

```python
# gates
f[t] = sigmoid(W_fx @ x[t] + W_fa @ a[t-1] + b_f)
i[t] = sigmoid(W_ix @ x[t] + W_ia @ a[t-1] + b_i)
o[t] = sigmoid(W_ox @ x[t] + W_oa @ a[t-1] + b_o)
c̃[t] = tanh   (W_cx @ x[t] + W_ca @ a[t-1] + b_c)

# cell state and hidden state
c[t] = f[t] * c[t-1] + i[t] * c̃[t]
a[t] = o[t] * tanh(c[t])
ŷ[t] = softmax(W_ya @ a[t] + b_y)
```

### 3.3 GRU Cell

```python
z[t] = sigmoid(W_zx @ x[t] + W_za @ a[t-1] + b_z)   # update gate
r[t] = sigmoid(W_rx @ x[t] + W_ra @ a[t-1] + b_r)   # reset gate
h̃[t] = tanh(   W_hx @ x[t] + W_ha @ (r[t] * a[t-1]) + b_h)
a[t] = (1 - z[t]) * a[t-1] + z[t] * h̃[t]
ŷ[t] = softmax(W_ya @ a[t] + b_y)
```

Variables

- x[t]: input vector at step t
- a[t], c[t]: hidden and cell states
- f, i, o, z, r: forget, input, output, update, reset gates
- W_*x, W_*a, b_*: weight matrices and biases

### 4. Code & Practical Application

### 4.1 Keras Example

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# build a bidirectional 2-layer GRU for sequence classification
model = models.Sequential([
    layers.Bidirectional(
        layers.GRU(64, return_sequences=True),
        input_shape=(None, feature_dim)
    ),
    layers.GRU(32),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

### 4.2 NumPy GRU Cell (single step)

```python
import numpy as np

def gru_cell_forward(x_t, a_prev, params):
    W_zx, W_za = params['W_zx'], params['W_za']
    W_rx, W_ra = params['W_rx'], params['W_ra']
    W_hx, W_ha = params['W_hx'], params['W_ha']
    b_z, b_r, b_h = params['b_z'], params['b_r'], params['b_h']

    z = sigmoid(W_zx.dot(x_t) + W_za.dot(a_prev) + b_z)
    r = sigmoid(W_rx.dot(x_t) + W_ra.dot(a_prev) + b_r)
    h_tilde = np.tanh(W_hx.dot(x_t) + W_ha.dot(r * a_prev) + b_h)
    a_next = (1 - z) * a_prev + z * h_tilde
    return a_next
```

### 5. Visualization / Geometry

```
             ┌──────────┐        ┌─────────┐
 x[t] ──▶    │  RNN    │ ──▶ a[t] │ Output  │
             │  Cell   │         └─────────┘
 a[t-1] ──▶  └▲─┬───┬─┘            ▲
             │ │   │              │
        gates│ │state updates     │
             │ └───┘              │
             └────────────────────┘
```

- In gated cells, each gate (sigmoid) carves out subspaces of memory to retain or discard.
- Bidirectional RNNs merge forward/backward trajectories in hidden-state space.

### 6. Common Pitfalls & Tips

- Forgetting `return_sequences=True` when stacking RNN layers.
- Not resetting states between sequences if `stateful=True`.
- Over-parameterization: LSTM has ~4× more weights than Vanilla RNN.
- Misusing bidirectional RNN in predictive tasks where future context isn’t available at inference.

### 7. Interview-Ready Insights

- Compare parameter counts: GRU ~75% of LSTM’s parameters but often matches its performance.
- When to choose a bidirectional RNN: tasks like POS tagging or speech where full sequence is known in advance.
- Explain how gates mitigate vanishing gradients by creating linear paths through time.
- Trade-off stacked vs. deep RNN: depth improves representation but can amplify gradient issues without careful initialization or normalization.

### 8. Practice Exercises

1. **Implement a vanilla vs. gated cell**
    - Code a simple RNN cell and a GRU cell in NumPy.
    - Train both on a toy task (e.g., binary sequence parity) and compare learning curves.
2. **Bidirectional sentiment classifier**
    - Use TensorFlow to build a bidirectional LSTM on IMDb reviews.
    - Measure accuracy difference versus a unidirectional LSTM.
3. **Gate visualization**
    - For a trained GRU on a toy dataset, record z[t] and r[t] across time.
    - Plot their average activation vs. t to see when the model chooses to update or reset.

---

## Language Models & Sequence Generation

### 1. Direct Definition

A language model (LM) assigns a probability to a sequence of tokens (words, characters, subwords). Sequence generation uses that model to produce new, coherent sequences by sampling or decoding token by token.

### 2. Concept Intuition

- Language is a chain of dependent events: each word depends on the ones before.
- An LM learns these dependencies so it can predict the next token given all prior tokens.
- Sequence generation “rolls out” those predictions into a full sentence or paragraph.
- Applications: autocomplete, machine translation (decoding), text summarization, code generation.

### 3. Mathematical Breakdown

### 3.1 Chain Rule for Sequences

For a token sequence x¹…xᵀ, the joint probability:

```python
P(x¹…xᵀ) = ∏_{t=1…T} P(xᵗ | x¹…x^{t-1})
```

### 3.2 Model Formulation

At each step t the RNN/Transformer computes:

```python
h[t] = f(h[t-1], x[t-1])            # hidden or contextual state
logits[t] = W_ho @ h[t] + b_o       # raw scores over vocabulary
P(xᵗ | x<ᵗ) = softmax(logits[t])     # probability distribution
```

Loss across a corpus of m sequences:

```python
loss = -(1/m) * ∑_{i=1…m} ∑_{t=1…Tᶦ}
           log P(xᶦᵗ | xᶦ¹…xᶦ⁽ᵗ⁻¹⁾)
```

### 4. Code & Practical Application

### 4.1 Building a Tiny Character-Level RNN LM in PyTorch

```python
import torch
import torch.nn as nn

class CharRNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn   = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h0):
        # x: (batch, seq_len), h0: (1, batch, hidden_dim)
        e = self.embed(x)                  # (batch, seq_len, embed_dim)
        out, h = self.rnn(e, h0)           # out: (batch, seq_len, hidden_dim)
        logits = self.fc(out)              # (batch, seq_len, vocab_size)
        return logits, h

# Example training loop snippet
model = CharRNNLM(vocab_size=50, embed_dim=32, hidden_dim=64)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
for epoch in range(10):
    h0 = torch.zeros(1, batch_size, 64)
    logits, h0 = model(x_batch, h0)
    # reshape for loss: (batch*seq, vocab)
    loss = lossf(logits.view(-1, logits.size(-1)), y_batch.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
```

### 4.2 Sampling from the Trained LM

```python
def sample(model, start_token, max_len=100, temperature=1.0):
    model.eval()
    idx = torch.tensor([[start_token]])
    h   = torch.zeros(1, 1, model.rnn.hidden_size)
    generated = [start_token]

    for _ in range(max_len):
        logits, h = model(idx, h)
        logits = logits[:, -1, :] / temperature
        probs  = torch.softmax(logits, dim=-1)
        idx    = torch.multinomial(probs, num_samples=1)
        token  = idx.item()
        generated.append(token)
        if token == eos_token: break

    return generated
```

### 5. Visualization / Geometry

```
 Step t-1:  … x[t-2] → [ RNN cell ] → h[t-1] → softmax → P(x[t] | …)
                                            │
 Step t:      x[t-1] → [ RNN cell ] → h[t]   └─ sample next token
```

- The hidden state h[t] is a point in ℝᵈ tracking all past context.
- Softmax projects h[t] onto a probability simplex of size |vocab|.
- Sampling “walks” from one simplex to the next, generating a path through token space.

### 6. Common Pitfalls & Tips

- **Exposure Bias**: During training you feed ground-truth tokens (teacher forcing), but at test time you feed your own samples—mismatch can degrade generation. Mitigate with scheduled sampling.
- **Repetition Loops**: Low temperature or greedy decoding can get stuck repeating tokens.
- **Choosing Temperature**:
    - High (>1.0): more randomness, risk of gibberish.
    - Low (<1.0): safer, but may be dull or repetitive.
- **Beam Search Trade-offs**: Wider beams find higher-likelihood sequences but cost more compute and can output generic text.
- **Vocabulary Size**: Large vocab slows softmax. Consider subword tokenization (BPE, WordPiece).

### 7. Interview-Ready Insights

- **Perplexity**: exponent of average negative log-likelihood; the lower the better.
- **Teacher Forcing**: speeds convergence by feeding true x[t] during training; explain its pros and cons.
- **Beam Search**: how it balances exploration/exploitation versus greedy.
- **Sampling Methods**:
    - Greedy
    - Stochastic (multinomial)
    - Top-k / nucleus (top-p) sampling
- **Evaluation Metrics**: BLEU, ROUGE for translation/summarization; why perplexity alone doesn’t guarantee quality.

### 8. Practice Exercises

1. **Character-Level LM on Tiny Shakespeare**
    - Dataset: first 10 KB of Shakespeare.
    - Task: train a 2-layer GRU LM (embed_dim=64, hidden=128).
    - Compute training & validation perplexity.
2. **Sampling Comparison**
    - Generate 5 samples with greedy, temperature=0.8, top-k=5, nucleus p=0.9.
    - Compare diversity and coherence.
3. **Scheduled Sampling Experiment**
    - Implement scheduled sampling: gradually replace ground-truth inputs with model’s own predictions during training.
    - Observe effects on validation perplexity and sample quality.

---

## Sampling Novel Sequences

### 1. Direct Definition

Sampling a novel sequence means generating new tokens from a trained language model by drawing from its learned probability distribution at each step, rather than always taking the single most likely token.

### 2. Concept Intuition

When you sample instead of greedily picking the arg-max, you introduce variety and creativity. Three controls shape how “inventive” your outputs are:

- Temperature: Scales logits before softmax. Higher temperature flattens the distribution (more random); lower sharpens it (more conservative).
- Top-k filtering: Keeps only the k most probable tokens, zeroing out the rest before sampling.
- Nucleus (top-p) filtering: Keeps the smallest set of tokens whose cumulative probability ≥ p.

Together these let you balance coherence versus diversity.

### 3. Mathematical Breakdown

Let logits be the raw scores from your model for the next token. Temperature scaling:

```python
scaled_logits = logits / T
probs = softmax(scaled_logits)
```

Here

- `T` = temperature (>0)
- `softmax(z_i) = exp(z_i) / sum_j exp(z_j)`

Top-k filter:

```python
# assume logits shape (vocab,)
values, indices = topk(logits, k)
threshold = values[-1]
filtered_logits = where(logits < threshold, -inf, logits)
```

Nucleus (top-p) filter:

```python
sorted_probs, sorted_idx = sort(probs, descending=True)
cumulative = cumsum(sorted_probs)
mask = cumulative > p
# keep tokens up to first mask
filtered_indices = sorted_idx[ : first_true(mask) + 1 ]
filtered_logits = where(token_idx in filtered_indices, logits, -inf)
```

After filtering, renormalize and sample:

```python
probs = softmax(filtered_logits)
next_token = multinomial(probs, num_samples=1)
```

### 4. Code & Practical Application

Below is a PyTorch snippet combining temperature, top-k, and top-p sampling:

```python
import torch
import torch.nn.functional as F

def sample_sequence(model, start_tokens, max_len, T=1.0, top_k=None, top_p=None):
    model.eval()
    tokens = start_tokens.clone()             # shape (1, seq_len)
    h = None                                   # initial hidden state

    for _ in range(max_len):
        logits, h = model(tokens, h)          # logits: (1, seq_len, vocab)
        logits = logits[:, -1, :] / T         # focus on last step
        if top_k:
            top_vals, _ = torch.topk(logits, top_k)
            kth = top_vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < kth,
                                 torch.tensor(-1e10, device=logits.device),
                                 logits)
        if top_p:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cum_probs > top_p
            # mask everything after first True
            cutoff_idx = torch.argmax(cutoff, dim=-1)
            mask = torch.arange(probs.size(-1))[None, :] > cutoff_idx[:, None]
            logits = logits.masked_fill(mask, -1e10)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == eos_token:
            break

    return tokens
```

**Real-world workflow**

1. Train your model with teacher forcing.
2. At inference, use the `sample_sequence` function with tuned `T`, `top_k`, or `top_p`.
3. Evaluate outputs qualitatively and via metrics like perplexity, diversity, or human judgment.

### 5. Visualization / Geometry

Imagine your logits as a point in ℝᵛ (vocabulary-dimensional space). Temperature rescales distances from that point to the center, flattening or sharpening the probability simplex. Top-k/top-p carve away regions of that simplex, restricting your sampling to a smaller, focused manifold. Sampling then picks a random direction within that allowed region.

```
           Original softmax simplex
                /‾‾‾‾‾‾‾‾‾‾‾\
        /\     /             \
       /  \   /               \
      /    \ /                 \
     -----------------------------
     Filtered simplex via top-p
```

### 6. Common Pitfalls & Tips

- Temperature too high → gibberish, loss of grammar.
- Temperature too low → repetitive, stale outputs.
- Top-k too small → misses valid tokens; too large → negates the filter.
- Top-p too low → overly narrow; too high → nearly unfiltered.
- Forgetting to renormalize after filtering leads to invalid distributions.
- Inconsistent tokenization between training and inference breaks sampling.

### 7. Interview-Ready Insights

- Explain why temperature divides logits instead of probabilities.
- Contrast top-k versus nucleus sampling: top-k fixes sample count, nucleus fixes cumulative probability.
- Discuss how these methods impact diversity and coherence and why nucleus often yields more human-like text.
- Mention repetition penalty or presence penalty for long generations.

### 8. Practice Exercises

1. **Implement & Compare**
    - Generate ten sequences (length = 50) with:a) greedyb) temperature = 0.7c) top_k = 40d) top_p = 0.9
    - Compare for diversity and coherence.
2. **Tune Parameters**
    - On a small dataset (e.g., IMDB reviews), sample 100 sequences with varying (T, top_k, top_p).
    - Plot human-rated fluency vs. diversity scores to find sweet spots.
3. **Diversity Metrics**
    - For each sampling method, compute self-BLEU (lower = more diverse).
    - Analyze trade-offs between perplexity and self-BLEU.

---

## Vanishing Gradients in RNNs

### 1. Direct Definition

Vanishing gradients occur when, during backpropagation through time, the gradient signals shrink exponentially as they’re propagated backward across many time steps. This makes earlier layers (or time steps) learn extremely slowly or not at all.

### 2. Concept Intuition

When you unroll an RNN over T steps, each hidden‐state update involves a multiplication by the recurrent weight matrix and activation derivative.

- If the product of their spectral norms is below 1, gradients shrink at every step.
- Long-range dependencies vanish because the “credit” for an error at time T barely reaches time step 1.
- As a result, the network fails to learn patterns spanning many time steps.

### 3. Mathematical Breakdown

Consider the gradient of the loss (L) at final time T with respect to the hidden state at time t:

```python
dL/da[t] = (W_aa.T @ dL/da[t+1]) * (1 - a[t]**2)
```

Unrolling this recursion back to step t=1:

```python
dL/da[1] = (W_aa.T)**(T-1) @ dL/da[T]
           * Π_{k=1…T-1} (1 - a[k]**2)
```

Here:

- `(W_aa.T)**(T-1)` denotes repeated matrix multiplication.
- `Π (1 - a[k]**2)` comes from successive tanh derivatives.
- If ‖W_aa‖ < 1 and `(1 - a[k]**2)` < 1, the overall norm decays exponentially with T.

### 4. Code & Practical Application

### 4.1 Measuring Gradient Norm Decay

```python
import numpy as np

# Toy recurrent weight and activation derivative
Waa = np.random.randn(5,5) * 0.5   # spectral norm < 1
deriv = lambda a: 1 - a**2

# simulate hidden states
a = [np.tanh(np.random.randn(5)) for _ in range(10)]
dl_da_T = np.random.randn(5)       # gradient at final step

# backpropagate gradient
dl_da = dl_da_T.copy()
norms = []
for t in reversed(range(len(a)-1)):
    dl_da = (Waa.T @ dl_da) * deriv(a[t])
    norms.append(np.linalg.norm(dl_da))

print("Gradient norms over time:", list(reversed(norms)))
```

Plot `norms` to see exponential decay toward zero. In real training, these tiny signals stall learning.

### 5. Visualization / Geometry

```
      δ[T]         δ[T-1]       δ[T-2]     …    δ[1]
       ●───W_aa^T───●──W_aa^T───●──W_aa^T───●
```

- Each arrow scales δ by W_aa^T and tanh′.
- When the scale factor <1, δ shrinks at every step, vanishing before it reaches the start.

Geometrically, you’re repeatedly projecting error vectors into smaller and smaller subspaces, collapsing long-term dependency signals.

### 6. Common Pitfalls & Tips

- Initializing Waa with small random values can exacerbate vanishing.
- Using tanh or sigmoid activations without gates compounds shrinkage.
- Ignoring gradient norms: always monitor them during training to detect vanishing.
- Batching long sequences without truncation can hide the problem until much later in training.

### 7. Interview-Ready Insights

- Explain why vanishing arises from repeated multiplication by Jacobians with spectral norm <1.
- Describe how gating in LSTM/GRU provides near-linear paths that preserve gradient magnitude.
- Discuss truncated BPTT: cutting sequence length trades gradient fidelity for stability.
- Contrast gradient clipping (addresses exploding gradients) with architectures that solve vanishing.

### 8. Practice Exercises

1. **Gradient Norm Tracking**
    - Implement an RNN forward and BPTT.
    - Log `||∂L/∂a[t]||` for t=1…T during training on sequences of increasing length.
    - Plot and interpret the decay rate.
2. **Activation Comparison**
    - Replace tanh with ReLU in your RNN cell.
    - Compare gradient norms over time. Do ReLU networks still vanish? Why or why not?
3. **Gated vs. Vanilla**
    - Train a vanilla RNN and a GRU on a toy “copy memory” task (recall a token from 50 steps ago).
    - Evaluate which model learns long-range dependencies and how vanishing affects convergence.

---

## Gated Recurrent Unit (GRU)

### 1. Direct Definition

A Gated Recurrent Unit (GRU) is a type of recurrent neural network cell that uses gating mechanisms to adaptively capture dependencies in sequence data. It streamlines the LSTM architecture by combining the forget and input gates into a single update gate, resulting in fewer parameters and often faster training.

### 2. Concept Intuition

A vanilla RNN blindly passes its hidden state through time, which causes vanishing gradients and limits learning of long-range patterns. GRUs introduce two gates:

- Update gate (z[t]) decides how much of the past state to keep versus replace with new content.
- Reset gate (r[t]) controls how much of the previous state to forget when computing new candidate information.

By adaptively blending old and new information, GRUs preserve important signals over longer spans without the full complexity of LSTMs.

### 3. Mathematical Breakdown

At each time step t:

```python
# 1. Update gate: how much past to keep
z[t] = sigmoid(W_zx @ x[t] + W_za @ a[t-1] + b_z)

# 2. Reset gate: how much past to forget
r[t] = sigmoid(W_rx @ x[t] + W_ra @ a[t-1] + b_r)

# 3. Candidate hidden state: new information
h̃[t] = tanh(W_hx @ x[t] + W_ha @ (r[t] * a[t-1]) + b_h)

# 4. Final hidden state: linear interpolation
a[t] = (1 - z[t]) * a[t-1] + z[t] * h̃[t]
```

Variables and shapes:

- x[t] ∈ ℝⁿₓ: input vector
- a[t], a[t-1] ∈ ℝⁿₐ: hidden state vectors
- z[t], r[t], h̃[t] ∈ ℝⁿₐ: update gate, reset gate, candidate state
- W_zx, W_rx, W_hx ∈ ℝⁿₐ×ⁿₓ: input-to-gate weights
- W_za, W_ra, W_ha ∈ ℝⁿₐ×ⁿₐ: hidden-to-gate weights
- b_z, b_r, b_h ∈ ℝⁿₐ: gate biases

Why it works:

- Multiplying r[t] * a[t-1] zeroes out irrelevant parts of the old state before combining with the new input.
- The update gate z[t] balances retaining the old state versus accepting the new candidate, creating shortcuts that carry gradients more directly backward.

### 4. Code & Practical Application

### 4.1 NumPy: Single GRU Cell

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gru_cell_forward(x_t, a_prev, params):
    W_zx, W_za = params['W_zx'], params['W_za']
    W_rx, W_ra = params['W_rx'], params['W_ra']
    W_hx, W_ha = params['W_hx'], params['W_ha']
    b_z, b_r, b_h = params['b_z'], params['b_r'], params['b_h']

    z = sigmoid(W_zx.dot(x_t) + W_za.dot(a_prev) + b_z)
    r = sigmoid(W_rx.dot(x_t) + W_ra.dot(a_prev) + b_r)
    h_tilde = np.tanh(W_hx.dot(x_t) + W_ha.dot(r * a_prev) + b_h)
    a_next = (1 - z) * a_prev + z * h_tilde

    return a_next

# Example dimensions
n_x, n_a = 4, 6
params = {
    'W_zx': np.random.randn(n_a, n_x)*0.01,
    'W_za': np.random.randn(n_a, n_a)*0.01,
    'b_z':  np.zeros(n_a),

    'W_rx': np.random.randn(n_a, n_x)*0.01,
    'W_ra': np.random.randn(n_a, n_a)*0.01,
    'b_r':  np.zeros(n_a),

    'W_hx': np.random.randn(n_a, n_x)*0.01,
    'W_ha': np.random.randn(n_a, n_a)*0.01,
    'b_h':  np.zeros(n_a),
}

x_t = np.random.randn(n_x)
a_prev = np.zeros(n_a)
a_next = gru_cell_forward(x_t, a_prev, params)
print("Next hidden state shape:", a_next.shape)
```

### 4.2 PyTorch: Stacked GRU for Sequence Classification

```python
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)              # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]               # last time step
        logits = self.fc(out)             # (batch, num_classes)
        return logits

# Usage
model = GRUClassifier(input_dim=100, hidden_dim=64, num_layers=2, num_classes=10)
print(model)
```

### 5. Visualization / Geometry

```
     ┌─────────────────────────────────┐
     │           GRU Cell             │
     │                                 │
x[t] │→─┐       ┌───────┐              │
    └─┬┴──┬─→ z  │       │              │
      │   └──→ r │       │              │
      │         └──┬────┘              │
      │            ↓                   │
      └─────────> ⊗ — h̃[t] ────┐       │
                  ↑            │       │
a[t-1] ────────■───┘            │       │
               ■─ bypass via (1−z)     │
                                  ↓    │
                               a[t]───┘

```

- The bypass “■” carries (1−z[t])*a[t−1] directly forward, preventing gradient shrinkage.
- Gates carve subspaces of the hidden dimension to update or reset.

### 6. Common Pitfalls & Tips

- Forgetting to initialize biases (especially b_z) so the update gate starts near 1 (carry behavior).
- Using tanh for candidate but not clipping; large activations can saturate and slow learning.
- Overlooking sequence padding and masking when batching variable lengths.
- Confusing hidden_dim and num_layers ordering in PyTorch (batch_first vs. seq_first).

### 7. Interview-Ready Insights

- GRU vs. LSTM: GRU merges forget/input into update gate, reducing parameters by ~25–33%. Often matches LSTM performance on moderate-dependency tasks.
- Gates as dynamic highways: update gate creates shortcuts akin to residual connections, smoothing gradient flow.
- When to choose GRU: faster convergence for shorter dependencies or limited compute; LSTM for very long or complex patterns.
- Relation to attention: GRU’s gating foreshadows attention’s selective focus over past states.

### 8. Practice Exercises

1. **Parity Task**
    - Generate sequences of 0/1 bits of length 20; label each position with parity of all seen bits (even=0, odd=1).
    - Train a GRU cell to predict parity at each step. Monitor accuracy and gate activations over time.
    - Hint: use teacher forcing and clip gradients.
2. **IMDb Sentiment Classification**
    - Use `nn.GRU` in PyTorch with embedding and dropout.
    - Compare performance vs. `nn.LSTM` on validation accuracy and training speed.
    - Plot training loss and inspect how many epochs each needs to converge.
3. **Gate Dynamics Visualization**
    - For a trained GRU on the parity task, record z[t] and r[t] across time steps for several sequences.
    - Plot average gate values vs. t to see where the model chooses to update or reset.

---

## Long Short-Term Memory (LSTM)

### 1. Direct Definition

A Long Short-Term Memory (LSTM) cell is a recurrent neural network unit that maintains two states—a cell state c⟨t⟩ and a hidden state a⟨t⟩—and uses three gates (forget, input, output) to control information flow. These gates allow LSTMs to learn long-range dependencies by preserving and updating memory without vanishing or exploding gradients.

### 2. Concept Intuition

- Memory highway: The cell state c⟨t⟩ acts like a conveyor belt that runs through time steps almost unchanged.
- Selective updates: Gates open or close pathways, deciding which information to forget, write, or expose.
- Long dependencies: By providing nearly linear gradient flow along c⟨t⟩, LSTMs can remember events from hundreds of steps ago.
- Versatility: They power tasks from language modeling and translation to time‐series forecasting and speech synthesis.

### 3. Mathematical Breakdown

At time step t, given input x⟨t⟩ and previous states a⟨t−1⟩, c⟨t−1⟩:

```python
# 1. Forget gate: decide what to discard
f[t] = sigmoid( W_fx @ x[t] + W_fa @ a[t-1] + b_f )

# 2. Input gate: decide what new to write
i[t] = sigmoid( W_ix @ x[t] + W_ia @ a[t-1] + b_i )

# 3. Candidate cell update
c_tilde[t] = tanh( W_cx @ x[t] + W_ca @ a[t-1] + b_c )

# 4. Update cell state by forgetting and adding
c[t] = f[t] * c[t-1] + i[t] * c_tilde[t]

# 5. Output gate: decide what to expose
o[t] = sigmoid( W_ox @ x[t] + W_oa @ a[t-1] + b_o )

# 6. Compute new hidden state
a[t] = o[t] * tanh( c[t] )
```

Variables and shapes

- x⟨t⟩ ∈ ℝⁿₓ: input vector
- a⟨t⟩ ∈ ℝⁿₐ: hidden state (output)
- c⟨t⟩ ∈ ℝⁿₐ: cell state (memory)
- f, i, o ∈ ℝⁿₐ: forget, input, output gates
- c_tilde ∈ ℝⁿₐ: candidate cell update
- W_*x ∈ ℝⁿₐ×ⁿₓ, W_*a ∈ ℝⁿₐ×ⁿₐ; b_* ∈ ℝⁿₐ

Why it works

- Sigmoid gates f, i, o ∈ [0,1] choose how much information flows.
- Linear path for c⟨t⟩ avoids repeated squashing, preserving gradient magnitude.
- Nonlinearities (tanh) inside gates and update add expressivity without blocking gradient flow entirely.

### 4. Code & Practical Application

### 4.1 NumPy: Single LSTM Cell Forward Pass

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_cell_forward(x_t, a_prev, c_prev, params):
    W_fx, W_fa, b_f = params['W_fx'], params['W_fa'], params['b_f']
    W_ix, W_ia, b_i = params['W_ix'], params['W_ia'], params['b_i']
    W_cx, W_ca, b_c = params['W_cx'], params['W_ca'], params['b_c']
    W_ox, W_oa, b_o = params['W_ox'], params['W_oa'], params['b_o']

    # Forget gate
    f = sigmoid(W_fx.dot(x_t) + W_fa.dot(a_prev) + b_f)
    # Input gate
    i = sigmoid(W_ix.dot(x_t) + W_ia.dot(a_prev) + b_i)
    # Candidate cell
    c_tilde = np.tanh(W_cx.dot(x_t) + W_ca.dot(a_prev) + b_c)
    # New cell state
    c_next = f * c_prev + i * c_tilde
    # Output gate
    o = sigmoid(W_ox.dot(x_t) + W_oa.dot(a_prev) + b_o)
    # New hidden state
    a_next = o * np.tanh(c_next)

    return a_next, c_next

# Example initialization
n_x, n_a = 3, 5
params = { key: np.random.randn(n_a, dim)*0.01
           for key, dim in [
             ('W_fx', n_x), ('W_fa', n_a),
             ('W_ix', n_x), ('W_ia', n_a),
             ('W_cx', n_x), ('W_ca', n_a),
             ('W_ox', n_x), ('W_oa', n_a)] }
for b in ['b_f','b_i','b_c','b_o']:
    params[b] = np.zeros(n_a)

x_t = np.random.randn(n_x)
a_prev = np.zeros(n_a)
c_prev = np.zeros(n_a)
a_next, c_next = lstm_cell_forward(x_t, a_prev, c_prev, params)
print("a_next shape:", a_next.shape, "c_next shape:", c_next.shape)
```

### 4.2 TensorFlow/Keras: Stacked LSTM for Text Classification

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_classifier(vocab_size, embed_dim, hidden_units, num_layers, num_classes):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embed_dim, mask_zero=True))
    for _ in range(num_layers):
        model.add(layers.LSTM(hidden_units, return_sequences=True))
    # Final LSTM without return_sequences
    model.add(layers.LSTM(hidden_units))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model = build_lstm_classifier(
    vocab_size=10000, embed_dim=128,
    hidden_units=64, num_layers=2,
    num_classes=5
)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

### 5. Visualization / Geometry

```
     ┌─────────────────────────────────────────────────┐
     │                     LSTM Cell                 │
     │                                                 │
x[t] │→─┐ f-gate ──┐ i-gate ──┐ c_tilde ──┐ o-gate ──┐│
    └─┬─▶───┐      └─▶───┐      └─▶───┐      └─▶───┐┬─┘
      │    │         │         │         │         │
      ▼    ▼         ▼         ▼         ▼         ▼
     ●─ c_prev ──►[ * f ]──►+──[ * i ]──►[ o * tanh ]─► a_next
                   ▲        ▲                      ▲
                   │        └──── cell-update ─────┘
                   └─────────── highway ────────────┘

```

- The “highway” (linear path) from c_prev to c_next carries gradients unchanged when f≈1, i≈0.
- Gates carve out subspaces dictating what information passes, is stored, or is output.

### 6. Common Pitfalls & Tips

- Forgetting to mask padded time steps when batching variable-length sequences.
- Initializing forget-gate bias b_f too low: f starts near 0 and erases memory prematurely.
- Using tanh on cell state c can still saturate; consider layer normalization inside LSTM.
- Over-stacking layers without dropout can overfit small datasets.

### 7. Interview-Ready Insights

- Explain how the forget gate f[t] stabilizes gradient flow by providing a (nearly) linear path.
- Compare parameter counts: LSTM has ~4× those of a vanilla RNN cell.
- Discuss peephole connections (optional): direct c→gate links for finer control.
- Contrast LSTM vs. GRU: LSTM’s separate input/forget gates give more flexibility at the cost of complexity.

### 8. Practice Exercises

1. Implement `lstm_cell_backward` in NumPy to compute gradients ∂L/∂W_* and ∂L/∂b_*. Validate by gradient checking on a toy sequence.
2. Copy-memory task: generate a sequence of length T with a marker, then a random pattern; train an LSTM to reproduce the pattern after the marker. Test for different T (e.g., 20, 50, 100).
3. Sentiment analysis with Keras LSTM: compare single-layer vs. two-layer LSTM on IMDb reviews. Plot training/validation loss and inspect overfitting.
4. Gate dynamics: for a trained LSTM on the copy-memory task, record f[t], i[t], o[t] across time. Plot their average activations vs. t to see how the cell chooses to remember or update.

---

## Bidirectional Recurrent Neural Network

### 1. Direct Definition

A bidirectional recurrent neural network (BiRNN) runs two parallel RNNs over an input sequence: one forward (left→right) and one backward (right→left). At each time step t, it concatenates the forward hidden state and backward hidden state to form a richer representation that captures both past and future context.

### 2. Concept Intuition

Standard RNNs can only use past context when making a prediction at time t. In many tasks—like part-of-speech tagging or named-entity recognition—the token at position t depends on both what came before and what comes after.

A BiRNN solves this by:

- Processing the sequence once in normal order to get forward states.
- Processing the same sequence in reverse to get backward states.
- Merging both so each time step “sees” the full context window.

### 3. Mathematical Breakdown

Given inputs x⟨1…T⟩, initial forward state a↗⟨0⟩ and backward state a↙⟨T+1⟩:

```python
# forward RNN (t = 1…T)
a↗[t] = tanh(W↗_ax @ x[t] + W↗_aa @ a↗[t-1] + b↗_a)

# backward RNN (t = T…1)
a↙[t] = tanh(W↙_ax @ x[t] + W↙_aa @ a↙[t+1] + b↙_a)

# concatenate states
a[t] = concatenate( a↗[t], a↙[t] )   # shape (2·n_a, )

# output prediction
ŷ[t] = softmax(W_y @ a[t] + b_y)
```

Shapes

- x[t] ∈ ℝⁿₓ
- a↗[t], a↙[t] ∈ ℝⁿₐ
- a[t] ∈ ℝ²ⁿₐ
- W↗_ax, W↙_ax ∈ ℝⁿₐ×ⁿₓ; W↗_aa, W↙_aa ∈ ℝⁿₐ×ⁿₐ
- W_y ∈ ℝⁿʸ×(2·nₐ); b↗_a, b↙_a ∈ ℝⁿₐ; b_y ∈ ℝⁿʸ

### 4. Code & Practical Application

### 4.1 TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True),
    layers.Bidirectional(
        layers.LSTM(64, return_sequences=True),
        input_shape=(None, )
    ),
    layers.TimeDistributed(layers.Dense(num_tags, activation='softmax'))
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

### 4.2 PyTorch

```python
import torch.nn as nn

class BiGRUTagger(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.bigru     = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.embedding(x), None
        out, _ = self.bigru(e)               # out: (batch, seq_len, 2*hidden_dim)
        logits = self.classifier(out)        # (batch, seq_len, num_tags)
        return logits
```

### 5. Visualization / Geometry

```
   x⟨1⟩ → [▶ RNN cell ▶] → a↗⟨1⟩ ─┐
                                │
   x⟨2⟩ → [▶ RNN cell ▶] → a↗⟨2⟩ ─┼─ concatenate → a⟨2⟩ → ŷ⟨2⟩
                                │
   x⟨3⟩ → [▶ RNN cell ▶] → a↗⟨3⟩ ─┘

   x⟨3⟩ ← [◀ RNN cell ◀] ← a↙⟨3⟩ ─┐
                                │
   x⟨2⟩ ← [◀ RNN cell ◀] ← a↙⟨2⟩ ─┼─ concatenate → a⟨2⟩ → ŷ⟨2⟩
                                │
   x⟨1⟩ ← [◀ RNN cell ◀] ← a↙⟨1⟩ ─┘
```

Each a⟨t⟩ lives in a 2·nₐ-dimensional space, tracing intertwined forward/backward trajectories that encode full-sequence context.

### 6. Common Pitfalls & Tips

- Doubling hidden size: Remember that the concatenated state is twice as large, so downstream layers and parameter counts increase accordingly.
- Future context leak: Don’t use BiRNNs in streaming or real-time tasks where future tokens aren’t available.
- Masking/padding: Must apply the same mask to both directions so padded positions don’t skew backward states.
- return_sequences: When stacking BiRNNs or using TimeDistributed outputs, always set `return_sequences=True`in intermediate layers.

### 7. Interview-Ready Insights

- Explain why bidirectional context boosts performance on sequence labeling and offline tasks but isn’t usable in live generation.
- Discuss parameter sharing: forward and backward RNNs do not share weights, so BiRNN has ~2× the parameters of a single directional RNN.
- Compare to attention: BiRNNs capture fixed-window future context; attention (or Transformers) can attend arbitrarily far ahead or behind.
- Mention fusion strategies: instead of concatenation, you can sum, average, or pass through a learnable projection.

### 8. Practice Exercises

1. **POS Tagging with BiLSTM**
    - Use Keras to build a bidirectional LSTM for part-of-speech tagging on the UD English Web Treebank.
    - Measure token-level accuracy versus a unidirectional LSTM baseline.
2. **Masking Test**
    - Create variable-length sequences and pad them.
    - Verify that backward hidden states at padded positions remain zero (or masked) when you apply `mask_zero=True`.
3. **Fusion Comparison**
    - Implement three fusion methods (concatenate, sum, average) over forward/backward states in a custom NumPy BiRNN.
    - Train on a toy named-entity recognition task and compare performance.

---

## Deep Recurrent Neural Networks (Deep RNNs)

### 1. Direct Definition

A Deep RNN is an architecture that stacks multiple recurrent layers on top of each other. At each time step, the hidden state of layer ℓ feeds as input to layer ℓ+1, enabling the network to learn hierarchical temporal representations.

### 2. Concept Intuition

Deepening an RNN adds capacity to model complex patterns over time:

- Layer 1 captures low‐level, short‐term features (e.g., phonemes in speech).
- Layer 2 abstracts those into mid‐level patterns (e.g., phoneme sequences into syllables).
- Higher layers encode long‐range or semantic relationships (e.g., entire words or phrases).

Stacking layers lets the network build on simpler temporal features to recognize richer, multi‐scale dependencies.

### 3. Mathematical Breakdown

For an L-layer deep RNN, at time step t:

```python
# Input to first layer
x[1][t] = x[t]

# For each layer ℓ = 1…L
a[ℓ][t] = tanh( W_ax[ℓ] @ x[ℓ][t]
                + W_aa[ℓ] @ a[ℓ][t-1]
                + b_a[ℓ] )

# Hidden of layer ℓ becomes input for next layer
x[ℓ+1][t] = a[ℓ][t]

# Final output (after layer L)
ŷ[t] = softmax( W_ya @ a[L][t] + b_y )
```

Shapes:

- x[t] ∈ ℝⁿₓ
- a[ℓ][t] ∈ ℝⁿₐℓ (hidden size of layer ℓ)
- W_ax[ℓ] ∈ ℝⁿₐℓ×n_inputℓ, W_aa[ℓ] ∈ ℝⁿₐℓ×ⁿₐℓ, b_a[ℓ] ∈ ℝⁿₐℓ

This hierarchy of transformations deepens both spatial (feature) and temporal processing.

### 4. Code & Practical Application

### 4.1 NumPy: Two-Layer RNN Forward Pass

```python
import numpy as np

def deep_rnn_forward(X, a0_layers, params):
    """
    X: list of T inputs x[t] shape (n_x,)
    a0_layers: list of initial hidden states [a0_1, a0_2]
    params: {
      'W_ax1', 'W_aa1', 'b_a1',
      'W_ax2', 'W_aa2', 'b_a2',
      'W_ya',  'b_y'
    }
    Returns hidden states and outputs
    """
    T = len(X)
    a_prev1, a_prev2 = a0_layers
    A1, A2, Y_hat = [], [], []

    for t in range(T):
        x_t = X[t]
        # Layer 1
        z1 = params['W_ax1'].dot(x_t) + params['W_aa1'].dot(a_prev1) + params['b_a1']
        a1 = np.tanh(z1)

        # Layer 2
        z2 = params['W_ax2'].dot(a1) + params['W_aa2'].dot(a_prev2) + params['b_a2']
        a2 = np.tanh(z2)

        # Output
        logits = params['W_ya'].dot(a2) + params['b_y']
        y_hat = np.exp(logits) / np.sum(np.exp(logits))

        A1.append(a1); A2.append(a2); Y_hat.append(y_hat)
        a_prev1, a_prev2 = a1, a2

    return A1, A2, Y_hat

# Example dims
n_x, n_a1, n_a2, n_y = 3, 5, 4, 2
params = {
    'W_ax1': np.random.randn(n_a1, n_x)*0.01,
    'W_aa1': np.random.randn(n_a1, n_a1)*0.01,
    'b_a1':  np.zeros(n_a1),
    'W_ax2': np.random.randn(n_a2, n_a1)*0.01,
    'W_aa2': np.random.randn(n_a2, n_a2)*0.01,
    'b_a2':  np.zeros(n_a2),
    'W_ya':  np.random.randn(n_y, n_a2)*0.01,
    'b_y':   np.zeros(n_y)
}
X = [np.random.randn(n_x) for _ in range(7)]
a0_layers = [np.zeros(n_a1), np.zeros(n_a2)]
A1, A2, Y_hat = deep_rnn_forward(X, a0_layers, params)
print("Outputs steps:", len(Y_hat))
```

### 4.2 Keras: Stacked RNN Layers

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    layers.SimpleRNN(128, return_sequences=True),
    layers.SimpleRNN(64,  return_sequences=True),
    layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

### 5. Visualization / Geometry

Unrolling a 2-layer RNN for 3 time steps:

```
Time →  t=1              t=2              t=3
       ┌─────────┐       ┌─────────┐       ┌─────────┐
 x[1] →│ RNN L1  │→a¹[1]  │ RNN L1  │→a¹[2]  │ RNN L1  │→a¹[3]
       └─┬───────┘       └─┬───────┘       └─┬───────┘
         ↓ a¹[t]            ↓ a¹[t]            ↓ a¹[t]
       ┌──┴──────┐       ┌──┴──────┐       ┌──┴──────┐
         RNN L2          RNN L2          RNN L2
       └─────────┘       └─────────┘       └─────────┘
             ↓                 ↓                 ↓
           ŷ[1]              ŷ[2]              ŷ[3]
```

Depth axis (layers) and time axis intertwine to form a grid of recurrent cells.

### 6. Common Pitfalls & Tips

- Forgetting `return_sequences=True` on all but the last RNN layer when stacking.
- Vanishing/exploding gradients worsen with both depth and sequence length; consider gating (LSTM/GRU), residual connections, or layer normalization.
- Sharp dimensionality changes between layers can disrupt training—match hidden sizes or use projection layers.
- Overfitting risk increases; apply dropout between layers and recurrent dropout within cells.

### 7. Interview-Ready Insights

- Explain how depth adds hierarchical temporal modeling but increases gradient propagation paths in both time and layers.
- Discuss residual connections in RNNs (e.g., adding a¹[t] to a²[t]) to stabilize training.
- Compare deep RNNs with deep feed-forward nets: both trade off capacity against gradient flow challenges.
- Relate to Transformers: depth in self-attention layers achieves similar hierarchical representation without recurrence.

### 8. Practice Exercises

1. **Stacked GRU for Sequence Classification**
    - Build a 3-layer GRU in PyTorch (`nn.GRU(return_sequences=True)` on first two layers).
    - Train on a toy task (e.g., classifying sine vs. square waves).
    - Monitor how training time and accuracy change when you add or remove layers.
2. **Residual Deep RNN**
    - Implement a 2-layer RNN in NumPy with a skip connection:
        
        ```
        a2[t] = tanh(W_ax2·a1[t] + W_aa2·a2[t−1] + b2) + a1[t]
        ```
        
    - Compare gradient norms and convergence against the vanilla stacked version.
3. **Layer Normalization**
    - Insert LayerNorm between hidden‐state affine transform and tanh in each layer.
    - Evaluate impact on RNN training stability for sequences of length 100.
4. **Visualization of Hidden Trajectories**
    - For nₐ=2 in layer 1 and layer 2, plot the trajectory of (a1[t], a2[t]) in 2D over time for a sample input.
    - Observe how depth changes the shape of the trajectory.

---