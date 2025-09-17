# DL_c5_m4

## Transformer network intuition

### 1. Direct Definition

The Transformer is a neural network architecture for sequence modeling that relies entirely on self-attention and position-wise feed-forward layers, eliminating recurrence and convolution. It processes all tokens in parallel, capturing long-range dependencies by learning how each position in the input attends to every other position.

### 2. Concept Intuition

Self-attention lets the model “look” at every token in a sequence when encoding a particular token. Instead of marching through time steps (as in RNNs), a token’s representation is updated by a weighted sum of all tokens—where weights are learned similarities.

By stacking these attention layers, the Transformer builds rich contextualized embeddings. Parallel computation and direct connections between any two positions let it handle long dependencies and very deep stacks efficiently.

### 3. Mathematical Breakdown

Prerequisites

- Matrix multiplication, dot product
- Softmax function

Key projections

```python
# X: (sequence_length, d_model)
Q = X @ W_Q    # (seq_len, d_k)
K = X @ W_K    # (seq_len, d_k)
V = X @ W_V    # (seq_len, d_v)
```

- W_Q, W_K, W_V: learnable matrices
- d_model: model dimension; typically d_k = d_v = d_model/num_heads

Scaled dot-product attention

```python
# scores: (seq_len, seq_len)
scores = Q @ K.T         # raw similarity
scores_scaled = scores / sqrt(d_k)
A = softmax(scores_scaled, axis=1)  # rows sum to 1
# output: (seq_len, d_v)
Attention = A @ V
```

Multi-head attention

```python
# for each head i
head_i = Attention(Q @ W_Qi, K @ W_Ki, V @ W_Vi)
# then concatenate all heads
MultiHead = concat(head_1, ..., head_h) @ W_O
```

### 4. Code & Practical Application

Numpy implementation of one head:

```python
import numpy as np

def scaled_dot_product_attention(X, W_Q, W_K, W_V):
    Q = X @ W_Q             # (n, d_k)
    K = X @ W_K             # (n, d_k)
    V = X @ W_V             # (n, d_v)

    scores = Q @ K.T        # (n, n)
    scores /= np.sqrt(Q.shape[1])
    A = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    return A @ V            # (n, d_v)

# toy example
X = np.random.rand(3, 4)    # 3 tokens, d_model=4
W_Q = np.random.rand(4, 4)
W_K = np.random.rand(4, 4)
W_V = np.random.rand(4, 4)
out = scaled_dot_product_attention(X, W_Q, W_K, W_V)
print(out.shape)  # (3, 4)
```

PyTorch built-in multi-head:

```python
import torch
from torch import nn

embed_dim, num_heads = 8, 2
mha = nn.MultiheadAttention(embed_dim, num_heads)

# shape: (seq_len, batch_size, embed_dim)
x = torch.randn(5, 1, embed_dim)
out, attn_weights = mha(x, x, x)
print(out.shape)          # (5, 1, 8)
print(attn_weights.shape) # (1, 5, 5)
```

### 5. Visualization / Geometry

Imagine each token’s embedding X[i] as a point in ℝᵈ. Multiplying by W_Q projects it to a query vector Q[i]—defining a direction you “ask” with. Multiplying by W_K projects every token to key vectors K[j]—potential “facts.” Dot-product Q[i]·K[j] measures relevance. After scaling and softmax, you get a convex combination over all V[j] projections.

Geometrically, each attention head learns its own coordinate system where similarity is a simple dot-product. The attention weight matrix A (n×n) can be plotted as a heatmap: rows are queries, columns keys, and the intensity shows how much information flows between positions.

### 6. Common Pitfalls & Tips

- Forgetting to divide by sqrt(d_k) leads to vanishing gradients when d_k is large.
- Mixing up shapes: Q, K, V must share the same seq_len dimension.
- Ignoring positional encoding: without it, order information is lost.
- Skipping mask in decoder: you’ll leak future tokens during training.
- Over-complicating: start with one attention head before scaling up.

### 7. Interview-Ready Insights

- Why no recurrence? Parallelism across tokens accelerates training on GPUs/TPUs.
- Complexity trade-off: O(n²·d_model) vs. O(n·d_model²) in RNNs; attention excels for moderate n but hits memory limits at very long sequences.
- Self-attention is content-based addressing—key idea inherited from Neural Turing Machines.
- Multi-head attention allows the model to capture different types of token relationships in parallel subspaces.
- Positional encoding injects order information via fixed sinusoids or learned embeddings.

### 8. Practice Exercises

1. Implement scaled dot-product attention from scratch using a tiny vocabulary of 4 tokens.
    
    Hint: Use one-hot vectors for X and identity matrices for W_Q/K/V to see pure attention patterns.
    
2. Visualize attention weights for a toy sentence, e.g., “I love machine learning”, by plotting the 4×4 heatmap.
    
    Hint: After computing A, use matplotlib’s `imshow(A, cmap='viridis')`.
    
3. Extend your numpy code to multi-head attention: split X into two heads, run each head separately, then concatenate.
    
    Hint: Use `np.split` on the feature dimension.
    
4. In PyTorch, replace the built-in `nn.MultiheadAttention` with your numpy version wrapped in `torch.from_numpy`/`torch.Tensor` and compare outputs.
5. Experiment: Vary d_k (e.g., 2, 4, 8) and observe how attention distributions sharpen or flatten.

---

## Self-Attention

### 1. Direct Definition

Self-attention is a mechanism that transforms an input sequence into a new sequence of the same length by letting each element “attend” to (i.e., weigh and aggregate) all elements—including itself—based on learned similarity scores. It uses three projections—queries (Q), keys (K), and values (V)—and computes weighted sums of V according to the similarity of Q to K.

### 2. Concept Intuition

Imagine you’re reading a sentence and want to understand the meaning of one word by looking at context words around it. Self-attention does exactly that: for each token, it asks “Which other tokens matter most to me right now?” and then pulls information from those tokens. This lets the model capture long-range dependencies in one shot, without stepping through tokens sequentially.

### 3. Mathematical Breakdown

Prerequisites refresh

- Matrix multiplication and transposition
- The softmax function for converting scores into probabilities

Key formulas (clean code blocks):

```python
# Given X shape: (seq_len, d_model)
# W_Q, W_K, W_V shapes: (d_model, d_k), (d_model, d_k), (d_model, d_v)

Q = X @ W_Q            # shape: (seq_len, d_k)
K = X @ W_K            # shape: (seq_len, d_k)
V = X @ W_V            # shape: (seq_len, d_v)

# raw attention scores
scores = Q @ K.T       # shape: (seq_len, seq_len)

# scale to stabilize gradients
scores_scaled = scores / sqrt(d_k)

# attention weights
A = softmax(scores_scaled, axis=1)  # each row sums to 1

# output of self-attention
output = A @ V        # shape: (seq_len, d_v)
```

Variable breakdown

- `seq_len`: number of tokens
- `d_model`: input embedding dimension
- `d_k`, `d_v`: dimensions for queries/keys and values
- `A[i, j]`: how much token *i* attends to token *j*

### 4. Code & Practical Application

### Numpy Implementation

```python
import numpy as np

def self_attention(X, W_Q, W_K, W_V):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T
    scores /= np.sqrt(Q.shape[1])

    A = np.exp(scores)
    A /= np.sum(A, axis=1, keepdims=True)

    return A @ V, A

# Toy example
X = np.array([
    [1.0, 0.0],   # token A
    [0.0, 1.0],   # token B
    [1.0, 1.0],   # token C
])
W_Q = np.eye(2)
W_K = np.eye(2)
W_V = np.eye(2)

output, attn_weights = self_attention(X, W_Q, W_K, W_V)
print("Output:", output)
print("Attention weights:\n", attn_weights)
```

### PyTorch Example

```python
import torch
from torch import nn

seq_len, batch, d_model = 4, 1, 8
mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=2)

x = torch.randn(seq_len, batch, d_model)  # (seq_len, batch, d_model)
out, attn = mha(x, x, x)                   # self-attention
print(out.shape)     # (4, 1, 8)
print(attn.shape)    # (1, 4, 4)
```

### 5. Visualization / Geometry

Each token embedding sits as a point in a high-dimensional space.

- **Queries** define “questions” in that space.
- **Keys** define “where to look.”
- **Values** carry the actual information.

A dot product Q[i]·K[j] measures alignment between the question from token *i* and the fact at token *j*. After softmax, you get a convex combination of the values V[j].

Plotting the attention matrix `A` as a heatmap (rows: queries, columns: keys) reveals which tokens interact and how strongly.

### 6. Common Pitfalls & Tips

- Forgetting `scores /= sqrt(d_k)` can lead to extremely large or small softmax gradients.
- Overlooking mask usage in decoder self-attention allows future tokens to leak.
- Using identical dimensions for d_k and seq_len can blow up memory for long sequences.
- Ignoring positional encodings makes the model permutation-invariant (loses order).

### 7. Interview-Ready Insights

1. Self-attention runs in O(n² · d) time, but is massively parallelizable—key for GPU/TPU performance.
2. It generalizes both content-based addressing (Neural Turing Machines) and non-local operations in vision.
3. Multi-head attention partitions the embedding space so different heads can focus on syntax, semantics, or positional patterns.
4. Without recurrence or convolution, every token sees all others in one layer—no vanishing gradient over time.

### 8. Practice Exercises

1. **One-Hot Test**
    - Use one-hot vectors as `X` and identity matrices for `W_Q`, `W_K`, `W_V`.
    - Observe which tokens attend to which, and explain why.
2. **Heatmap Visualization**
    - Compute attention weights for a toy sentence of length 5 with random embeddings.
    - Plot `A` with matplotlib’s `imshow` and annotate the strongest connections.
3. **Dimension Sweep**
    - Vary `d_k` through [2, 4, 16] while keeping `d_model=32`.
    - Track how the average entropy of each attention row changes—do smaller `d_k` lead to flatter distributions?
4. **Masking Experiment**
    - Implement causal masking so that each token only attends to past tokens (including itself).
    - Verify that the upper triangle of `A` is zero after softmax.
5. **PyTorch vs. Numpy**
    - Wrap your numpy self-attention in `torch.from_numpy` and compare its output to `nn.MultiheadAttention`with one head.
    - Confirm numerical closeness within a small epsilon.

---

## Multi-Head Attention

### 1. Direct Definition

Multi-head attention runs several self-attention mechanisms (“heads”) in parallel, each learning different projections of queries, keys, and values. It then concatenates their outputs and applies a final linear transformation, allowing the model to jointly attend to information from multiple representation subspaces.

### 2. Concept Intuition

Imagine reading a sentence through different “lenses.” One lens might focus on grammatical structure, another on semantic roles, a third on coreference, and so on. Each lens is a head: it transforms inputs into its own query, key, and value space, computes attention, and extracts distinct relationships. By combining all heads, the model captures varied patterns simultaneously.

### 3. Mathematical Breakdown

Given input X with shape (seq_len, d_model) and h heads:

```python
# For each head i = 1…h
Q_i = X @ W_Qi    # shape: (seq_len, d_k)
K_i = X @ W_Ki    # shape: (seq_len, d_k)
V_i = X @ W_Vi    # shape: (seq_len, d_v)

# Scaled dot-product attention
scores_i      = Q_i @ K_i.T
scores_i     /= sqrt(d_k)
A_i            = softmax(scores_i, axis=1)
head_i_output = A_i @ V_i   # shape: (seq_len, d_v)

# Concatenate all heads
concat = concat(head_1_output, …, head_h_output)  # shape: (seq_len, h * d_v)

# Final linear projection
MultiHeadOutput = concat @ W_O   # shape: (seq_len, d_model)
```

Variable dimensions

- d_model: total model dimension
- h: number of heads
- d_k = d_v = d_model / h (must divide evenly)
- W_Qi, W_Ki: (d_model, d_k)
- W_Vi: (d_model, d_v)
- W_O: (h * d_v, d_model)

### 4. Code & Practical Application

### Numpy Implementation

```python
import numpy as np

def multi_head_attention(X, W_Q, W_K, W_V, W_O, h):
    # X: (seq_len, d_model)
    seq_len, d_model = X.shape
    d_k = d_model // h

    # 1. Linear projections and split heads
    Q = X @ W_Q       # (seq_len, d_model)
    K = X @ W_K
    V = X @ W_V

    Qh = np.split(Q, h, axis=1)  # list of (seq_len, d_k)
    Kh = np.split(K, h, axis=1)
    Vh = np.split(V, h, axis=1)

    heads = []
    for Qi, Ki, Vi in zip(Qh, Kh, Vh):
        scores = Qi @ Ki.T
        scores /= np.sqrt(d_k)
        A = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        heads.append(A @ Vi)  # (seq_len, d_k)

    # 2. Concatenate and final proj
    concat = np.concatenate(heads, axis=1)  # (seq_len, d_model)
    return concat @ W_O                     # (seq_len, d_model)

# Toy data
seq_len, d_model, h = 5, 8, 2
X   = np.random.randn(seq_len, d_model)
W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)
W_O = np.random.randn(d_model, d_model)

out = multi_head_attention(X, W_Q, W_K, W_V, W_O, h)
print("Output shape:", out.shape)  # (5, 8)
```

### PyTorch Example

```python
import torch
from torch import nn

seq_len, batch, d_model, num_heads = 6, 2, 16, 4
mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

# shape for PyTorch: (seq_len, batch, d_model)
x = torch.randn(seq_len, batch, d_model)
out, attn_weights = mha(x, x, x)

print("Output:", out.shape)         # (6, 2, 16)
print("Attention weights:", attn_weights.shape)  # (2, 6, 6)
```

### 5. Visualization / Geometry

1. **Subspace Views**Each head projects tokens into its own d_k-dimensional subspace.
2. **Attention Heatmaps**You can plot A_i (seq_len×seq_len) for each head. Different heads highlight different token-to-token interactions.
3. **Concatenation**Geometrically, concatenation stacks these varied attention outputs side by side, forming a richer embedding in Rᵈ_model.

### 6. Common Pitfalls & Tips

- Ensure d_model is divisible by h; otherwise head dimensions break.
- Always include the sqrt(d_k) scaling; omitting it destabilizes training.
- Don’t forget the final W_O projection—otherwise heads can’t be recombined properly.
- Watch memory: multi-head attention uses O(h · seq_len²) space for score matrices.
- Experiment with small h and small seq_len first to debug shapes.

### 7. Interview-Ready Insights

- Multiple heads let the model attend to different types of relationships (syntax vs. semantics) in parallel.
- The final projection W_O merges these subspace features back into the model dimension.
- Choosing h trades off between representational diversity and computation/memory cost.
- In Vision Transformers, heads often learn specialized patterns like edges vs. textures vs. color blobs.
- Position-wise feed-forward layers and residual connections around multi-head blocks stabilize deep stacks.

### 8. Practice Exercises

1. **Single vs. Multi-Head**
    - Implement single-head attention (h=1) and multi-head (h=4).
    - Compare output variances: is multi-head richer?
2. **Head Specialization**
    - Train a toy Transformer on a small translation task.
    - Visualize each head’s attention heatmap and describe what each head focuses on.
3. **Dimensionality Sweep**
    - Keep d_model=32; vary h among [1, 2, 4, 8].
    - Measure training loss convergence speed. What’s the sweet spot?
4. **Mask Integration**
    - Add causal masking into your numpy multi-head function so heads can’t see future tokens.
    - Verify by plotting masked score matrices.
5. **Backward Pass Check**
    - Using PyTorch’s `autograd.gradcheck`, validate your PyTorch `nn.MultiheadAttention` gradients for a small input.
    - Understand how gradient flows through Q, K, V, and W_O layers.

---

## Transformer Network

### 1. Direct Definition

A Transformer is a sequence-to-sequence architecture built entirely on attention mechanisms and feed-forward networks. It comprises an encoder stack and a decoder stack, each layer combining multi-head self-attention, residual connections with layer normalization, and position-wise feed-forward sublayers. Positional encodings inject order information since there is no recurrence.

### 2. Concept Intuition

Instead of processing tokens one by one (as in RNNs), the Transformer simultaneously lets every token attend to all others.

- Encoder: Learns contextualized embeddings of the input by stacking self-attention and feed-forward layers.
- Decoder: Generates output tokens one at a time, masking future positions in its self-attention and attending to encoder outputs to incorporate source context.

This design captures long-range dependencies, is highly parallelizable, and scales to very deep networks.

### 3. Mathematical Breakdown

Assume input sequence of length n, embedding dimension d_model, h heads, feed-forward hidden size d_ff.

1. **Input embeddings + positional encoding**

```python
# X_token: (n, d_model)
PE = positional_encoding(n, d_model)    # same shape
X = X_token + PE                        # inject order info
```

1. **Encoder layer**

```python
# Self-attention sublayer
Z1 = MultiHeadAttention(X, X, X)        # (n, d_model)
Z1_res = LayerNorm(X + Z1)

# Feed-forward sublayer
FF = relu(Z1_res @ W1 + b1) @ W2 + b2   # shape: (n, d_model)
EncoderOut = LayerNorm(Z1_res + FF)
```

1. **Decoder layer**

```python
# Masked self-attention
D1 = MultiHeadAttention(Y, Y, Y, mask=future_mask)
D1_res = LayerNorm(Y + D1)

# Encoder-decoder attention
D2 = MultiHeadAttention(D1_res, EncoderOut, EncoderOut)
D2_res = LayerNorm(D1_res + D2)

# Feed-forward
FFd = relu(D2_res @ W1d + b1d) @ W2d + b2d
DecoderOut = LayerNorm(D2_res + FFd)
```

1. **Final linear + softmax**

```python
logits = DecoderOut @ W_vocab + b_vocab  # (n_out, d_vocab)
probs  = softmax(logits, axis=-1)
```

### 4. Code & Practical Application

### PyTorch Skeleton

```python
import torch
from torch import nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x_enc, x_dec, src_mask=None, tgt_mask=None):
        # Encoder self-attention
        attn1, _ = self.self_attn(x_enc, x_enc, x_enc, key_padding_mask=src_mask)
        enc = self.norm1(x_enc + attn1)

        # Decoder masked self-attention
        attn2, _ = self.self_attn(x_dec, x_dec, x_dec, attn_mask=tgt_mask)
        dec1 = self.norm2(x_dec + attn2)

        # Decoder cross-attention
        attn3, _ = self.cross_attn(dec1, enc, enc, key_padding_mask=src_mask)
        dec2 = self.norm3(dec1 + attn3)

        # Feed-forward
        ff_out = self.ff(dec2)
        dec_out = self.norm3(dec2 + ff_out)

        return enc, dec_out

# Usage for one layer:
# x_src: (seq_len_src, batch, d_model)
# x_tgt: (seq_len_tgt, batch, d_model)
layer = TransformerLayer(d_model=512, num_heads=8, d_ff=2048)
enc_out, dec_out = layer(x_src, x_tgt, src_mask, tgt_mask)
```

### 5. Visualization / Geometry

- **Data flow**:
    
    X_token → add positional encoding → encoder layers → encoder output
    
    Decoder receives shifted target + pos-enc, then:
    
    masked self-attention → cross-attention with encoder output → feed-forward → final predictions
    
- **Residual & norms**:
    
    Every sublayer adds its input (identity path) and then applies LayerNorm, stabilizing deep stacks.
    
- **Attention heatmaps**:
    
    Plot attention matrices from each head in encoder and decoder to see which positions attend to which. This reveals patterns like subject-verb relationships or coreference.
    

### 6. Common Pitfalls & Tips

- Missing positional encoding makes the model ignore token order.
- Incorrect masking in decoder leaks future information.
- Wrong dimension splits for multi-head attention (d_model must be divisible by num_heads).
- Forgetting to apply LayerNorm after residual adds can lead to training instability.
- Overlooking dropout between sublayers can cause overfitting in small datasets.

### 7. Interview-Ready Insights

- Transformers remove recurrence to achieve full parallelism over sequence length, drastically improving training speed on modern hardware.
- The O(n²·d_model) complexity comes from attention score matrices—attention excels for sequences up to a few thousand tokens; for longer inputs, use sparse or local attention variants.
- Encoder-decoder attention in the decoder bridges source and target, learning alignment patterns akin to classical attention in seq2seq RNNs but in parallel.
- Positional encodings can be fixed (sinusoidal) or learned; fixed encodings allow extrapolation to longer sequences.
- Residual connections + LayerNorm enable training of very deep stacks (up to hundreds of layers in variants like GPT-3 and BERT).

### 8. Practice Exercises

- **Mini Transformer**
    
    Implement a two-layer encoder and decoder from scratch using NumPy. Train on a toy copy-task (input equals output) to verify it learns identity mapping.
    
- **Masking Challenge**
    
    Generate a causal mask matrix for target length 6 and show that upper triangle entries are `-inf` before softmax.
    
- **Positional Encoding Plot**
    
    Compute sinusoidal positional encodings for positions [0…99] in 16 dimensions and plot a few dimensions as functions of position.
    
- **Ablation Study**
    
    Remove skip connections or LayerNorm from your PyTorch TransformerLayer and compare training convergence on a simple translation dataset.
    
- **Attention Analysis**
    
    Train a small Transformer on English-French word translation. For a test sentence, extract attention weights from each decoder head and visualize them as heatmaps to interpret which source words each head focuses on.
    

---

## Transformer Pre-processing

### 1. Direct Definition

Transformer pre-processing is the pipeline that converts raw input tokens (e.g., words or subwords) into numerical representations the model can consume. It includes tokenization, vocabulary lookup, sequence padding/truncation, creation of attention masks, and addition of positional encodings before feeding data into the encoder and decoder.

### 2. Concept Intuition

Raw text can’t flow through a neural network, so we first split it into discrete symbols—tokens—that capture meaning and frequency patterns. Each token is mapped to an integer ID via a fixed vocabulary. Sequences are then padded or truncated to a uniform length so we can batch them. Finally, we embed IDs into vectors and add positional information so the model knows token order.

### 3. Mathematical Breakdown

1. Token IDs

```python
# Given a vocabulary dict: token_to_id
tokens = ["The", "cat", "sat"]
input_ids = [token_to_id.get(t, token_to_id["[UNK]"]) for t in tokens]
# input_ids shape: (seq_len,)
```

1. Padding and attention mask

```python
# pad to max_len
padded = input_ids + [token_to_id["[PAD]"]] * (max_len - len(input_ids))
# mask: 1 for real tokens, 0 for padding
attention_mask = [1]*len(input_ids) + [0]*(max_len-len(input_ids))
# shapes: (max_len,), (max_len,)
```

1. Embeddings plus positional encoding

```python
# E: (vocab_size, d_model)
token_embeddings = E[padded]         # shape: (max_len, d_model)
pos_enc = positional_encoding(max_len, d_model)  # same shape
X = token_embeddings + pos_enc      # final input to Transformer
```

### 4. Code & Practical Application

### Simple Tokenizer + Positional Encoding

```python
import numpy as np

def build_simple_vocab(texts):
    vocab = {"[PAD]":0, "[UNK]":1}
    for text in texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe

# Example usage
texts = ["I love NLP", "Transformers rock"]
vocab = build_simple_vocab(texts)
ids = [[vocab.get(t, vocab["[UNK]"]) for t in txt.split()] for txt in texts]
max_len = 5
padded_ids = [seq + [vocab["[PAD]"]]*(max_len-len(seq)) for seq in ids]
attention_masks = [[1]*len(seq) + [0]*(max_len-len(seq)) for seq in ids]
E = np.random.randn(len(vocab), 16)  # d_model=16
pos_enc = positional_encoding(max_len, 16)
batch_X = [E[pid] + pos_enc for pid in padded_ids]
```

### PyTorch Embedding Example

```python
import torch
from torch import nn

vocab_size, d_model, max_len = 1000, 32, 10
embedding = nn.Embedding(vocab_size, d_model)
pos_enc = torch.tensor(positional_encoding(max_len, d_model), dtype=torch.float)

# batch_input: (batch_size, max_len)
batch_input = torch.randint(0, vocab_size, (2, max_len))
mask = (batch_input != 0).long()  # 0 is PAD id

token_emb = embedding(batch_input)  # (2, max_len, d_model)
X = token_emb + pos_enc.unsqueeze(0)  # broadcast pos encoding
```

### 5. Visualization / Geometry

- **Token ID Map**A table mapping tokens→IDs lets you inspect how words and subwords are indexed.
- **Attention Mask Matrix**Plot mask as a heatmap (1’s white, 0’s black) to see which positions are ignored during attention.
- **Positional Encoding Curves**Graph each dimension of `pos_enc[position, :]` versus position; sinusoids of varying wavelengths reveal how the model discerns order.

### 6. Common Pitfalls & Tips

- Mismatched vocab: Using a different tokenizer at inference can produce out-of-vocabulary errors.
- Forgetting to mask padding tokens leads the model to attend to meaningless positions.
- Inconsistent sequence lengths: Always fix `max_len` across training and inference.
- Learned vs. fixed positional encodings: fixed sinusoids generalize to longer sequences, learned ones may not.

### 7. Interview-Ready Insights

- Byte-Pair Encoding (BPE) and WordPiece strike a balance between vocabulary size and handling rare words via subword units.
- Attention masks differentiate padding positions and enforce causality in the decoder (causal mask vs. padding mask).
- Sinusoidal positional encodings allow extrapolation to longer sequences, while learned embeddings specialize to train-time lengths.
- Pre-processing speed and consistency directly impact model convergence and inference latency.

### 8. Practice Exercises

1. Build a BPE tokenizer for a small corpus (e.g., 50 sentences) and compare its vocabulary size to whitespace splitting.
2. Implement a function that generates both padding masks and causal masks for decoder inputs of varying lengths.
3. Plot the first four dimensions of positional encoding for positions 0–99 and explain why they look like waves.
4. Create a PyTorch `Dataset` and `DataLoader` that returns `(X, attention_mask)` batches ready for a Transformer encoder.
5. Experiment: swap fixed sinusoids with a learned `nn.Embedding(max_len, d_model)` for positions—train on toy data and compare validation loss.

---

## Transformer network application

### 1. Direct Definition

Transformer network application refers to using pretrained or from-scratch Transformer architectures to solve real-world tasks—ranging from text generation and classification to machine translation, summarization, vision, speech, and time-series forecasting—by adapting the encoder, decoder, or encoder-decoder stacks with task-specific heads.

### 2. Concept Intuition

At its core, a Transformer builds rich contextual representations by letting every position attend to every other.

- In language tasks, you can:
- Feed input text through the **encoder** and attach a classification head.
- Use the **decoder** for autoregressive text generation.
- Leverage the full **encoder-decoder** for translation or summarization.
- In vision, you split an image into patches, embed them like tokens, and process with the **encoder** (Vision Transformer).
- In time-series, you treat each time step as a token, attend across history, and predict the next value.

By swapping out the final layer and loss function, you adapt the same underlying attention machinery to diverse applications.

### 3. Mathematical Breakdown

Here’s how you turn encoder outputs into task outputs:

1. **Encoder Output**

```python
# X_in: (seq_len, batch, d_model)
enc_out, _ = transformer_encoder(X_in, src_mask)
# enc_out: (seq_len, batch, d_model)
```

1. **Classification Head**

```python
# take representation of special token (e.g., [CLS] at position 0)
h = enc_out[0, :, :]                   # shape: (batch, d_model)
logits = h @ W_clf + b_clf             # shape: (batch, n_classes)
probs  = softmax(logits, axis=1)
```

1. **Seq2Seq Generation**

```python
# at each timestep t
y_t_embed = embedding(y_pred_ids[:t])
dec_out, _ = transformer_decoder(y_t_embed, enc_out, tgt_mask, src_mask)
logits_t  = dec_out[-1] @ W_vocab + b_vocab  # shape: (batch, vocab_size)
y_pred_id = argmax(softmax(logits_t))
```

1. **Vision Transformer Patch Classification**

```python
# img_patches: (num_patches, batch, patch_dim)
patch_emb = patch_proj(img_patches)    # linear map to d_model
cls_token = cls_embed.unsqueeze(1).repeat(1, batch, 1)
X_img = concat([cls_token, patch_emb], axis=0) + pos_enc
enc_out, _ = transformer_encoder(X_img, None)
image_logits = enc_out[0] @ W_img + b_img  # class scores
```

### 4. Code & Practical Application

### Fine-Tuning a Pretrained BERT for Sentiment Classification

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

# 1. Prepare tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 2. Custom dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx],
                             padding='max_length',
                             truncation=True,
                             max_length=self.max_len,
                             return_tensors="pt")
        return { "input_ids": enc.input_ids.squeeze(),
                 "attention_mask": enc.attention_mask.squeeze(),
                 "labels": torch.tensor(self.labels[idx]) }

# 3. DataLoader
dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len=128)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. Training loop (simplified)
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
for epoch in range(3):
    for batch in loader:
        optim.zero_grad()
        outputs = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optim.step()
```

### 5. Visualization / Geometry

- **Attention Heatmaps**
    
    Extract `attentions` from the model (`output.attentions=True`) and plot each head’s matrix for a sample sentence. Rows = query positions, columns = key positions.
    
- **Feature Space**
    
    Use t-SNE or UMAP on encoder outputs for different classes to see cluster separation.
    
- **Gradient Flow**
    
    Visualize loss surface along two directions in the classification head’s weight space to gauge sharpness or flatness of the solution.
    

### 6. Common Pitfalls & Tips

- **Catastrophic Forgetting**: Fine-tuning too aggressively can destroy pretrained knowledge—use low learning rates and gradual unfreezing.
- **Sequence Length Mismatch**: Inference texts longer than your `max_len` get truncated—consider sliding windows or long-context models (Longformer, BigBird).
- **Over-Splitting Patches**: In ViT, very small patches increase sequence length quadratically—balance patch size vs. resolution.
- **Data Imbalance**: For classification, use weighted loss or up/down-sampling to handle skewed labels.
- **Memory Footprint**: Transformer’s O(n²) attention may blow up on long inputs—use sparse attention or chunking.

### 7. Interview-Ready Insights

1. **Pretrained vs. From Scratch**: Fine-tuning leverages massive pretraining (e.g., BERT, GPT-3) for data-scarce tasks. Training from scratch can be viable only with huge corpora.
2. **Encoder-Only**: Great for classification, sentence embedding, and information retrieval (e.g., BERT, RoBERTa).
3. **Decoder-Only**: Autoregressive generation (e.g., GPT series) excels at text completion, code generation, and dialog.
4. **Encoder-Decoder**: Sequence-to-sequence tasks like translation and summarization (e.g., T5, BART) use cross-attention to integrate source and target.
5. **Vision Transformer (ViT)**: Demonstrates that pure attention can rival CNNs when trained on large image datasets.
6. **Adapters & LoRA**: Parameter-efficient fine-tuning techniques that add small layers or low-rank updates instead of full model updates.

### 8. Practice Exercises

1. **BERT Fine-Tuning**
    - Using the SST-2 sentiment dataset, fine-tune a pretrained BERT for 2-class classification. Report accuracy and confusion matrix.
2. **Summarization with T5**
    - Load `t5-small`, fine-tune on a tiny dataset of (article, summary) pairs, and generate summaries. Evaluate with ROUGE-1.
3. **Vision Transformer on CIFAR-10**
    - Implement patch embedding (patch size 4×4), add positional encodings, stack a 4-layer encoder, and train on CIFAR-10 for image classification.
4. **Time-Series Forecasting**
    - Treat a univariate temperature time series as tokens, build an encoder-only Transformer to predict the next 7 days. Compare MSE against an LSTM baseline.
5. **Attention Analysis**
    - For a trained translation model, pick a source-target sentence pair and visualize decoder `cross-attention`heads. Describe alignment patterns head by head.

---