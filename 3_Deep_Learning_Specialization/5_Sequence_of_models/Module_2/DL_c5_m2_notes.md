# DL_c5_m2

## Word Representation

### 1. Direct Definition

Word representation is the process of converting each word in a vocabulary into a numerical vector so that machine-learning models can process and learn from text data.

### 2. Concept Intuition

- Text is inherently symbolic—models need numbers.
- A good representation captures meaning and similarity: “king” and “queen” should be closer than “king” and “apple.”
- Early methods (one-hot) treat words as distinct tokens with no notion of similarity.
- Distributed representations (dense vectors) embed words in a continuous space, revealing semantic and syntactic relationships.

### 3. Mathematical Breakdown

### 3.1 One-Hot Encoding

Every word ( w ) in a vocabulary of size ( V ) is represented by a vector ( x \in \mathbb{R}^V ) where exactly one element is 1 and the rest 0.

```python
# Suppose vocab = ["I", "love", "cats"]
V = 3
# one-hot for "love" (index 1)
x = [0, 1, 0]
```

### 3.2 Embedding Matrix

We learn (or initialize) an embedding matrix ( E \in \mathbb{R}^{V \times d} ). For a one-hot vector ( x ), the dense embedding ( e \in \mathbb{R}^d ) is:

```
e = E^T · x
```

Because ( x ) has exactly one “1” at index ( i ), this simply selects row ( i ) of ( E ).

- ( V ): vocabulary size
- ( d ): embedding dimension (e.g. 50, 100, 300)
- ( E[i] ): the d-dimensional vector for word index ( i )

### 4. Code & Practical Application

### 4.1 One-Hot Encoder (NumPy)

```python
import numpy as np

vocab = ["I", "love", "cats"]
word_to_index = {w:i for i,w in enumerate(vocab)}

def one_hot(word):
    V = len(vocab)
    x = np.zeros(V, dtype=np.float32)
    x[word_to_index[word]] = 1.0
    return x

# test
print(one_hot("cats"))  # [0. 0. 1.]
```

### 4.2 Embedding Lookup (NumPy)

```python
# Initialize random embeddings
V, d = len(vocab), 5
E = np.random.randn(V, d)

def embed(word):
    idx = word_to_index[word]
    return E[idx]  # equivalent to E.T @ one_hot(word)

# test
print("Embedding 'love':", embed("love"))
```

### 4.3 Using PyTorch Embedding Layer

```python
import torch
import torch.nn as nn

V, d = len(vocab), 5
embedding = nn.Embedding(num_embeddings=V, embedding_dim=d)

# Convert word to index tensor
idx_tensor = torch.tensor([word_to_index["love"]], dtype=torch.long)
e = embedding(idx_tensor)
print(e.shape)  # torch.Size([1, 5])
```

### 5. Visualization / Geometry

Imagine a 2D plane where each word-vector is a point:

- Similar words cluster together (e.g., “cat” and “dog”).
- Axes capture latent features (e.g., axis-1 might track “animalness,” axis-2 “royalty”).
- You can draw arrows showing vector arithmetic:
    - king − man + woman ≈ queen

```
y-axis ↑
       |
queen  •
       |
       |
king   •      man
       |\
       | \
       |  \
       |   • woman
       +------------→ x-axis
```

### 6. Common Pitfalls & Tips

- One-hot vectors are high-dimensional and sparse; they don’t capture similarity.
- Randomly initialized embeddings need training (via backprop); untrained embeddings don’t reflect semantics.
- Beware out-of-vocabulary (OOV) words; consider a special `<UNK>` token.
- Choosing ( d ): too small may underfit meaning; too large may overfit or waste compute.
- Always normalize or use cosine similarity when comparing vectors.

### 7. Interview-Ready Insights

- Explain why word embeddings beat one-hot: they capture distributional semantics (“you shall know a word by the company it keeps”).
- Mention count-based methods (e.g., PMI, SVD on co-occurrence) vs. prediction-based (word2vec, GloVe).
- Be ready to derive the embedding lookup gradient: dL/dE[idx] = dL/de.
- Discuss context windows: CBOW vs. Skip-Gram objectives.
- Talk about subword models (FastText) and contextual embeddings (ELMo, BERT).

### 8. Practice Exercises

1. Implement one-hot encoding and embedding lookup for a toy vocab of 5 words.
    - Hint: use a dict for indexing.
2. Using your NumPy embeddings, compute cosine similarity between every pair of words.
    - Hint: cosine(a,b) = (a·b) / (||a|| * ||b||).
3. Train a tiny skip-gram: given window size 1 on “I love cats,” predict “I” and “cats” from “love.”
    - Hint: optimize embeddings by gradient ascent on log-likelihood.
4. Visualize your 2-D embeddings with matplotlib’s scatter, labeling each point.

---

## Using Word Embeddings

### 1. Direct Definition

Using word embeddings means replacing each word in your text pipeline with a learned dense vector that captures its meaning, so downstream models (RNNs, CNNs, transformers) can process semantic information directly.

### 2. Concept Intuition

Word embeddings serve as the neural network’s “vocabulary lookup.”

They allow similar words to share parameters and generalize better.

When you feed embeddings into a sequence model, the network sees continuous geometry instead of sparse one-hots.

This geometry lets it spot analogies, synonyms, and contextual patterns.

### 3. Mathematical Breakdown

Given

- an embedding matrix `E` of shape `(V, d)`,
- an input sequence of word indices `[i1, i2, …, iT]`,

the embedding lookup step produces a matrix `X` of shape `(T, d)`:

```python
# pseudocode
X[t] = E[input_indices[t]]  # for t in 1…T
```

In vectorized form you can think of it as:

```
X = E[input_indices]
```

During backprop, if the loss ∂L/∂X is known, the gradient wrt `E` is simply:

```python
# gradient accumulation
for t in range(T):
    dE[input_indices[t]] += dX[t]
```

### 4. Code & Practical Application

### PyTorch Example

```python
import torch
import torch.nn as nn

# vocab size V, embedding dim d
V, d = 10000, 300
embedding = nn.Embedding(num_embeddings=V, embedding_dim=d)

# sample batch of tokenized sentences (batch_size=2, seq_len=5)
input_indices = torch.tensor([
    [10, 523, 34, 7, 102],
    [4, 78, 999, 1023, 1]
], dtype=torch.long)

# forward pass: (2, 5, 300)
embedded = embedding(input_indices)
```

### TensorFlow / Keras Example

```python
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

model = Sequential([
    Embedding(input_dim=V, output_dim=d, input_length=5),
    # followed by an RNN or Dense layers…
])
```

### 5. Visualization / Geometry

Embed a small vocab into 2D and plot:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# assume E_np is your (V, 2) embedding matrix
coords = TSNE(n_components=2).fit_transform(E_np[:100])
plt.scatter(coords[:,0], coords[:,1])
for i, word in enumerate(vocab[:100]):
    plt.text(coords[i,0], coords[i,1], word)
plt.show()
```

You’ll see clusters of related words—animals near animals, verbs near verbs. As you train downstream tasks, those points shift to capture task-specific meanings.

### 6. Common Pitfalls & Tips

- Forgetting to mask padding indices can skew embedding updates.
- Leaving embeddings untrained when fine-tuning a model often hurts performance.
- Using random initialization vs. pre-trained (GloVe, Word2Vec) affects convergence and accuracy.
- High embedding dims can overfit on small datasets; low dims may underrepresent nuance.

### 7. Interview-Ready Insights

- Explain why fine-tuning pre-trained embeddings on your task often yields better results than training from scratch.
- Be ready to describe negative sampling and how it speeds up skip-gram training.
- Discuss contextual embeddings (BERT) vs. static embeddings and when each is appropriate.
- Know how to implement and backpropagate through an embedding layer in code.

### 8. Practice Exercises

1. Load GloVe vectors (50-dim) for 5,000 words, then write a function that returns the cosine similarity between any two words.
    - Hint: normalize each embedding once, then use dot product.
2. Build a PyTorch model that classifies movie reviews using an `Embedding` layer followed by an `LSTM`. Train on a tiny IMDB sample.
    - Hint: pad sequences with a special index, and set `padding_idx` in the embedding layer.
3. Visualize the first 200 GloVe embeddings in 2D using t-SNE and color-code nouns vs. verbs.
    - Hint: assemble lists of nouns and verbs manually or via WordNet.

---

## Properties of Word Embeddings

### 1. Direct Definition

Word embedding properties are the characteristic behaviors and patterns exhibited by learned dense vectors that map words into a continuous space. These properties determine how embeddings capture semantic similarity, syntactic roles, and algebraic relationships among words.

### 2. Concept Intuition

Word embeddings aren’t arbitrary points—they form a geometry where:

- Similar words cluster: synonyms sit near each other.
- Linear relationships encode analogies: king − man + woman ≈ queen.
- Vector norms reflect word frequency or specificity.
- Smooth transitions capture gradations: “good,” “better,” “best” lie along a continuum.

### 3. Mathematical Breakdown

### 3.1 Cosine Similarity

Measures angle between vectors; 1 means identical direction.

```python
cosine(u, v) = (u · v) / (||u|| * ||v||)
```

### 3.2 Vector Arithmetic for Analogies

Given embeddings u, v, w, finding x for “u is to v as w is to x”:

```python
target = v - u + w
```

Then pick x whose embedding has highest cosine similarity with `target`.

### 3.3 Norm and Isotropy

- Norm: ||e|| = sqrt(e · e)
- Isotropy: embeddings spread uniformly in all directions; prevents hubs (vectors that are nearest neighbor to many others).

### 4. Code & Practical Application

```python
import numpy as np

# Suppose we have a dict word2vec mapping strings to np.arrays
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def analogy(u, v, w, word2vec):
    target = v - u + w
    best_word, best_score = None, -1
    for word, vec in word2vec.items():
        if word in [u_word, v_word, w_word]: continue
        score = cosine_similarity(target, vec)
        if score > best_score:
            best_word, best_score = word, score
    return best_word

# Example usage
u_word, v_word, w_word = "man", "king", "woman"
print("king–man+woman =", analogy("man","king","woman", word2vec))
```

### 5. Visualization / Geometry

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Select subset of embeddings for common words
words = ["king","queen","man","woman","apple","banana","fruit"]
vecs = np.array([word2vec[w] for w in words])
coords = TSNE(n_components=2).fit_transform(vecs)

plt.figure(figsize=(6,6))
for i, w in enumerate(words):
    x, y = coords[i]
    plt.scatter(x, y)
    plt.text(x+0.2, y+0.2, w)
plt.show()
```

Points will reveal two clusters: royalty analogies and fruit semantics.

### 6. Common Pitfalls & Tips

- High-dimensional hubs: some vectors dominate nearest-neighbor lists; apply dimensionality reduction or normalization.
- Ignoring norm: cosine similarity mitigates frequency biases over raw dot products.
- Static embeddings confuse polysemy: “bank” (river vs. finance) shares one vector. Contextual models (BERT) solve this.
- Overfitting when fine-tuning embeddings on small data can erase general semantic structure.

### 7. Interview-Ready Insights

- Describe intrinsic vs. extrinsic evaluation: word similarity tasks (e.g., WordSim-353) vs. downstream performance.
- Explain why isotropy matters: non-isotropic embeddings produce hubs and degrade similarity measures.
- Discuss how analogical relationships emerge from the training objective (skip-gram with negative sampling).
- Mention bias in embeddings: how gender or racial stereotypes can be encoded and need debiasing.

### 8. Practice Exercises

1. **Nearest Neighbors**: For a small pre-trained embedding subset, implement a function that returns the top-5 most similar words to a query.
    - Hint: normalize all vectors once and use dot products.
2. **Hubness Analysis**: Compute how many times each vector appears in the top-k neighbor lists; plot hub frequencies.
    - Hint: use `collections.Counter`.
3. **Analogy Test**: Given a file of analogy questions (e.g., “Paris France Berlin Germany”), write code to compute accuracy of your embedding on the test set.
    - Hint: skip questions containing OOV words.
4. **Polysemy Visualization**: Pick a polysemous word (“bank”). Using contextual embeddings (e.g., BERT), extract its vectors from different sentences and use t-SNE to see sense clusters.

---

## Embedding Matrix

### 1. Direct Definition

An embedding matrix is a learnable weight matrix of shape `(V, d)` that maps each of the `V` words in your vocabulary to a `d`-dimensional dense vector. Each row `E[i]` in this matrix is the embedding for the word with index `i`.

### 2. Concept Intuition

- Think of the embedding matrix as a giant lookup table: given a word index, you grab its row to get a vector that captures meaning.
- Rather than one-hot vectors, you train this table so semantically similar words end up with nearby rows in the `d`dimensional space.
- During training, gradient updates tweak only the rows corresponding to words seen in each batch—making learning efficient.
- Pre-trained embedding matrices (GloVe, Word2Vec) give you a head start with general semantics; fine-tuning adapts those semantics to your task.

### 3. Mathematical Breakdown

Assume:

- `V` = vocabulary size
- `d` = embedding dimension
- `E` ∈ ℝ^(V×d) = embedding matrix
- input index sequence `I = [i₁, i₂, …, i_T]`

### 3.1 Forward Lookup

```python
# For position t:
e_t = E[ i_t ]         # shape (d,)
# Vectorized for all T:
X = E[ I ]             # shape (T, d)
```

### 3.2 Backpropagation

If your loss `L` produces gradients `dX` (shape `(T, d)`), the gradient w.r.t. `E` is:

```python
dE = zeros_like(E)     # shape (V, d)
for t in range(T):
    dE[ I[t] ] += dX[t]
```

Only the rows for indices in `I` get updated each batch.

### 4. Code & Practical Application

### 4.1 NumPy From Scratch

```python
import numpy as np

V, d = 10000, 300
E = np.random.randn(V, d) * 0.01  # small random init

def embed_sequence(indices):
    # indices: list or array of ints length T
    return E[ indices ]            # shape (T, d)

# Backprop sketch
def backprop(indices, dX):
    # dX: gradient from upstream, shape (T, d)
    dE = np.zeros_like(E)
    for t, idx in enumerate(indices):
        dE[idx] += dX[t]
    return dE
```

### 4.2 PyTorch Embedding Layer

```python
import torch
import torch.nn as nn

V, d = 10000, 300
embedding = nn.Embedding(num_embeddings=V, embedding_dim=d, padding_idx=0)

# Example input batch: shape (batch_size, seq_len)
inputs = torch.tensor([[4, 10, 523],[7, 523, 1]], dtype=torch.long)
# Forward pass: shape (batch_size, seq_len, d)
embedded = embedding(inputs)
```

### 4.3 Loading Pre-trained Weights

```python
# Suppose glove_matrix is a numpy array shape (V, d)
embedding = nn.Embedding(V, d)
embedding.weight.data.copy_(torch.from_numpy(glove_matrix))
# Optionally freeze
embedding.weight.requires_grad = False
```

### 5. Visualization / Geometry

- **Heatmap of embedding rows**: Visualize `E[:100]` as a heatmap to inspect activation patterns across dimensions.
- **t-SNE / PCA**: Project `E[:500]` to 2D and scatter-plot—you’ll see clusters of nouns, verbs, or thematic groups.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

coords = PCA(n_components=2).fit_transform(E[:200])
plt.scatter(coords[:,0], coords[:,1])
for i, word in enumerate(vocab[:200]):
    plt.text(coords[i,0], coords[i,1], word)
plt.show()
```

### 6. Common Pitfalls & Tips

- Initializing `E` with too large a scale can slow training or cause divergence; use small random values or Xavier init.
- Forgetting to mask or set `padding_idx` means padding tokens also learn embeddings, skewing gradients.
- If you fine-tune pre-trained embeddings on a tiny dataset, you risk catastrophic forgetting of general semantics—consider freezing some layers.
- OOV words need an `<UNK>` index; never leave gaps in your embedding matrix.

### 7. Interview-Ready Insights

- Describe how gradient updates on `E` correspond to adjusting word semantics based on task loss.
- Explain the trade-off between training embeddings from scratch vs. using pre-trained ones and when to freeze or fine-tune.
- Be prepared to derive the shape and update rule for the embedding gradient in backprop.
- Discuss subword embeddings (FastText) vs. word-level and how they modify the embedding matrix concept.

### 8. Practice Exercises

1. **Build from Scratch**
    - Create an embedding matrix for a toy vocab of 50 words and implement `embed_sequence` and `backprop` as above.
    - Hint: simulate `dX` with random noise to check your gradient logic.
2. **Padding and Masking**
    - Extend your NumPy embedding so that index `0` is used for padding and never updated.
    - Hint: skip `if idx == 0` in your backprop loop.
3. **Freeze vs. Fine-Tune**
    - Load a small pre-trained embedding (e.g., GloVe 50-dim for 1,000 words) into PyTorch.
    - Train a simple classifier on a sentiment task twice: once freezing embeddings, once fine-tuning.
    - Compare validation accuracy and training curves.
4. **Embedding Heatmap**
    - Visualize the first 100 rows of your trained embedding matrix as a heatmap (`plt.imshow`).
    - Interpret which dimensions are most active for semantically related words (e.g., color, animal, emotion).

---

## Learning Word Embeddings

### 1. Direct Definition

Learning word embeddings is the process of training a model to map each word in a vocabulary into a dense, low-dimensional vector space such that geometric relationships among those vectors encode semantic and syntactic word relationships.

### 2. Concept Intuition

Learning embeddings turns discrete word tokens into continuous coordinates where:

- Words that appear in similar contexts end up close together.
- The model discovers axes (dimensions) corresponding to latent concepts like gender, tense, topic.
- Predictive methods (Word2Vec) train embeddings by letting words compete to predict their neighbors.
- Count-based methods (GloVe, LSA) derive embeddings from global co-occurrence statistics.

This learned geometry empowers downstream models to generalize across words that share meaning.

### 3. Mathematical Breakdown

### 3.1 Count-Based (GloVe)

Build a co-occurrence matrix `X` where `X[i,j]` = number of times word `j` appears in the context of word `i`.

The GloVe objective minimizes:

```python
J = sum_{i,j=1..V} f(X[i,j]) * (wi · wj + bi + bj - log(X[i,j]))^2
```

- `wi`, `wj` are the target and context embedding vectors (size d).
- `bi`, `bj` are scalar biases.
- `f(X)` is a weighting function, e.g.
    
    ```python
    f(x) = (x/x_max)**alpha   if x < x_max
         = 1                   otherwise
    ```
    

### 3.2 Predictive (Skip-Gram with Negative Sampling)

For a word pair `(center, context)`, maximize

```python
log σ(u_context · v_center)
+ sum_{k=1..K} E_{neg~P_n}[ log σ(-u_neg · v_center) ]
```

- `v_center` ∈ ℝ^d is the center-word embedding.
- `u_context` ∈ ℝ^d is the context-word embedding.
- `σ(x) = 1/(1 + exp(-x))` is the sigmoid.
- Negative samples drawn from unigram distribution `P_n(w) ∝ freq(w)^0.75`.
- `K` is number of negative samples per positive pair.

Backprop updates both `v_center` and `u_*` vectors to pull true contexts closer and push negatives away.

### 4. Code & Practical Application

### 4.1 Build Co-occurrence & SVD (NumPy)

```python
import numpy as np

corpus = ["i", "love", "nlp", "and", "i", "love", "dl"]
vocab = list(set(corpus))
V = len(vocab)
idx = {w:i for i,w in enumerate(vocab)}

# build co-occurrence with window size 1
X = np.zeros((V, V))
for i, w in enumerate(corpus):
    for j in [i-1, i+1]:
        if 0 <= j < len(corpus):
            X[idx[w], idx[corpus[j]]] += 1

# SVD
U, S, _ = np.linalg.svd(X+1e-8)
embeddings = U[:, :2]  # 2-dim vectors
```

### 4.2 Train Skip-Gram with Negative Sampling (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from random import choices

class SkipGramNS(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.center_emb = nn.Embedding(V, d)
        self.context_emb = nn.Embedding(V, d)
    def forward(self, center, context, neg_samples):
        v = self.center_emb(center)    # (batch, d)
        u = self.context_emb(context)  # (batch, d)
        pos_score = torch.mul(v, u).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score))

        neg_u = self.context_emb(neg_samples)  # (batch, K, d)
        neg_score = torch.bmm(neg_u, v.unsqueeze(2)).squeeze()
        neg_loss = -torch.log(torch.sigmoid(-neg_score)).sum(dim=1)

        return (pos_loss + neg_loss).mean()

# prepare training pairs and negatives...
model = SkipGramNS(V=len(vocab), d=50)
opt = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    opt.zero_grad()
    loss = model(center_batch, context_batch, neg_batch)
    loss.backward()
    opt.step()
```

### 5. Visualization / Geometry

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# assume embeddings is a np.array shape (V, 2)
coords = TSNE(n_components=2).fit_transform(embeddings)
plt.scatter(coords[:,0], coords[:,1])
for i, w in enumerate(vocab):
    plt.text(coords[i,0], coords[i,1], w)
plt.show()
```

You’ll see semantically related words cluster and linear analogies manifest as parallel vectors.

### 6. Common Pitfalls & Tips

- Unbalanced co-occurrence counts can dominate count-based methods; apply weighting `f(x)`.
- Too few negative samples slows convergence; too many wastes compute—`K=5…20` is common.
- Learning rate scheduling: start around 0.01 and decay over epochs.
- Rare words get poor embeddings; consider minimum frequency cutoff or subword models.
- Normalizing embeddings (`v/||v||`) helps when comparing with cosine similarity.

### 7. Interview-Ready Insights

- Compare count-based vs. predictive embeddings: global statistics vs. local context prediction.
- Explain why the negative sampling objective approximates the full softmax and its computational benefits.
- Discuss how hyperparameters (`window size`, `K`, `d`) influence embedding quality.
- Be ready to derive gradients for `v_center` and `u_context` under the skip-gram NS loss.

### 8. Practice Exercises

1. Implement GloVe’s loss and gradient descent on a tiny co-occurrence matrix (V≤10). Track loss over iterations.
2. Write a skip-gram sampler: given a corpus and window size 2, generate `(center, context)` pairs, then sample negatives.
3. Train skip-gram with your sampler and plot the cosine similarity between a chosen word and all others over epochs.
4. Evaluate embeddings on a small word analogy set (e.g., king–man+woman≈?). Compute your model’s accuracy.

Let me know how these exercises go or if you need further hints!

---

## Word2Vec

### 1. Direct Definition

Word2Vec is a predictive model that learns dense vector representations of words by training a shallow neural network to predict surrounding words in a corpus. It comes in two main flavors—Continuous Bag-of-Words (CBOW) and Skip-Gram—and uses efficient approximations like Negative Sampling to scale to large vocabularies.

### 2. Concept Intuition

- Skip-Gram treats each word as a “center” and tries to predict its nearby “context” words.
- CBOW reverses this: it averages context word vectors to predict the center word.
- Training these tasks forces words that share contexts to acquire similar vectors.
- Negative Sampling replaces the full softmax over the entire vocabulary with a small set of “noise” words, cutting computation dramatically.

### 3. Mathematical Breakdown

### 3.1 Skip-Gram Softmax Objective

For a center word ( w ) and context word ( c ):

```
P(c | w) = exp(u_c · v_w) / sum_{i=1..V} exp(u_i · v_w)
```

- `v_w` is the input (center) embedding of word `w`.
- `u_c` is the output (context) embedding of word `c`.
- Computing that denominator for every update is O(V), so we use approximations.

### 3.2 Negative Sampling

Negative Sampling replaces full softmax with this loss per positive pair `(w, c)`:

```python
loss = -log(sigmoid(u_c · v_w))
       - sum_{k=1..K} log(sigmoid(-u_neg[k] · v_w))
```

- `u_neg[k]` are embeddings of K “noise” words drawn from a unigram^0.75 distribution.
- This pushes `v_w` closer to true context `u_c` and away from random words.

### 4. Code & Practical Application

### 4.1 Skip-Gram with Negative Sampling in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from random import choices

class Word2VecSGNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)

    def forward(self, centers, contexts, negatives):
        v = self.center_emb(centers)                # (batch, d)
        u_pos = self.context_emb(contexts)          # (batch, d)
        pos_score = torch.mul(u_pos, v).sum(1)      # (batch,)
        pos_loss = -torch.log(torch.sigmoid(pos_score))

        u_neg = self.context_emb(negatives)         # (batch, K, d)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (batch, K)
        neg_loss = -torch.log(torch.sigmoid(-neg_score)).sum(1)

        return (pos_loss + neg_loss).mean()

# Example training loop sketch
model = Word2VecSGNS(vocab_size=10000, embed_dim=100)
opt = optim.Adam(model.parameters(), lr=0.001)

for centers, contexts, negatives in data_loader:
    opt.zero_grad()
    loss = model(centers, contexts, negatives)
    loss.backward()
    opt.step()
```

### 4.2 Training with Gensim

```python
from gensim.models import Word2Vec

# corpus: list of tokenized sentences
model = Word2Vec(sentences=corpus,
                 vector_size=100,
                 window=5,
                 min_count=2,
                 sg=1,           # 1 = skip-gram, 0 = CBOW
                 negative=5,     # number of negative samples
                 epochs=5)

# Access embedding for word "king"
vec_king = model.wv["king"]
```

### 5. Visualization / Geometry

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

words = ["king","queen","man","woman","apple","banana","orange","fruit"]
vecs = [model.wv[w] for w in words]
coords = TSNE(n_components=2).fit_transform(vecs)

plt.figure(figsize=(6,6))
for i, w in enumerate(words):
    x, y = coords[i]
    plt.scatter(x, y)
    plt.text(x+0.2, y+0.2, w)
plt.title("Word2Vec TSNE projection")
plt.show()
```

You’ll spot clusters (fruits together, royalty analogies forming parallel vectors).

### 6. Common Pitfalls & Tips

- Choosing `window` too small misses long-range dependencies; too large adds noise.
- Setting `min_count` too high drops rare but meaningful words; too low inflates vocabulary and slows training.
- Unbalanced sampling skews embeddings; use the 0.75 power weighting in negative sampling.
- Learning rate schedule: linearly decay or use adaptive optimizers to avoid overshooting.

### 7. Interview-Ready Insights

- Derive the gradient of the negative sampling loss w.r.t. `v_w` and `u_c`:
    
    ```python
    dL/dv_w = (sigmoid(u_c·v_w)-1) * u_c
              + sum_k sigmoid(u_neg[k]·v_w) * u_neg[k]
    ```
    
- Explain why negative sampling approximates full softmax yet retains quality for nearest-neighbor tasks.
- Contrast CBOW vs. Skip-Gram in terms of speed and performance on rare words.
- Discuss how hierarchical softmax offers another softmax approximation and when to use it.

### 8. Practice Exercises

1. **From Scratch Skip-Gram**
    - Build center-context pairs from a toy corpus. Implement negative sampling and train 2-D embeddings using only NumPy. Plot the loss curve.
2. **Hyperparameter Sweep**
    - Train Word2Vec on a small Wikipedia dump with varying `window`, `embed_dim`, `negative`. Report which combination yields highest word similarity on a validation set.
3. **Analogy Evaluation**
    - Given a list of analogy questions (`king man queen woman`), write code to compute accuracy of your gensim model using the vector arithmetic method.
4. **Subword Extension**
    - Integrate FastText’s character n-gram trick: break each word into 3–6 character grams and average their n-gram vectors. Compare performance on rare-word similarity tasks.

---

## Negative Sampling

### 1. Direct Definition

Negative sampling is an efficient training technique for word embedding models (like Skip-Gram) that replaces the expensive full softmax over a large vocabulary with a loss computed on a small set of “negative” words sampled from a noise distribution.

### 2. Concept Intuition

- In full softmax, predicting the true context word requires computing scores for every word in the vocabulary, which costs O(V) per update.
- Negative sampling sidesteps this by:
    - Pulling the embedding of a true context word closer to the center word.
    - Pushing embeddings of a handful of randomly chosen “noise” words away.
- You only update K + 1 word pairs (one positive, K negatives) instead of V pairs.
- Sampling noise words from a modified unigram distribution ensures common words appear more often as negatives, making the model learn to distinguish true contexts from frequent distractors.

### 3. Mathematical Breakdown

### 3.1 Full Softmax Loss (for comparison)

For a center word index `w` and its true context word index `c`:

```
P(c | w) = exp(u_c · v_w) / sum_{i=1..V} exp(u_i · v_w)
L_full = -log P(c | w)
```

Computing that denominator `sum_{i=1..V}` is the bottleneck when V is large.

### 3.2 Negative Sampling Loss

Given:

- `v_w` ∈ ℝᵈ: center-word embedding
- `u_c` ∈ ℝᵈ: context-word embedding
- K: number of negative samples
- {n₁, …, n_K}: indices of noise words drawn from distribution Pₙ

Define sigmoid function in code form:

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

Then the negative sampling loss for one (w, c) pair:

```python
loss = -log( sigmoid( u_c · v_w ) )
       - sum_{k=1..K} log( sigmoid( -u_{n_k} · v_w ) )
```

- The first term pulls true context closer (`u_c·v_w` large → sigmoid near 1).
- Each second term pushes noise word embeddings away (`u_{n_k}·v_w` small or negative).

### 3.3 Gradients

For one negative sample `n`:

```
dL/dv_w += (sigmoid(u_n · v_w) - 0) * u_n   if n is positive
dL/dv_w += (sigmoid(u_n · v_w) - 1) * u_n   if n is negative
```

More explicitly, for positive pair (w,c):

```python
grad_vw_pos = (sigmoid(u_c.dot(v_w)) - 1.0) * u_c
grad_uc_pos = (sigmoid(u_c.dot(v_w)) - 1.0) * v_w
```

For each negative sample `n_k`:

```python
grad_vw_neg = sigmoid(u_nk.dot(v_w)) * u_nk
grad_unk_neg = sigmoid(u_nk.dot(v_w)) * v_w
```

Summing these gives the total gradient updates for `v_w`, `u_c`, and each `u_{n_k}`.

### 4. Code & Practical Application

### 4.1 NumPy Implementation of One Training Step

```python
import numpy as np

def negative_sampling_step(v_w, u_c, u_neg, lr=0.01):
    # v_w: center vector shape (d,)
    # u_c: positive context vector shape (d,)
    # u_neg: negative samples shape (K, d)

    # compute scores
    pos_score = sigmoid(u_c.dot(v_w))
    neg_scores = sigmoid(-u_neg.dot(v_w))  # shape (K,)

    # compute loss (optional)
    loss = -np.log(pos_score) - np.sum(np.log(neg_scores))

    # gradients w.r.t. v_w
    grad_vw = (pos_score - 1.0) * u_c
    grad_vw += np.sum((1.0 - neg_scores)[:, np.newaxis] * u_neg, axis=0)

    # gradients w.r.t. u_c
    grad_uc = (pos_score - 1.0) * v_w

    # gradients w.r.t. each u_neg[k]
    grad_uneg = (1.0 - neg_scores)[:, np.newaxis] * v_w[np.newaxis, :]

    # update embeddings
    v_w -= lr * grad_vw
    u_c -= lr * grad_uc
    u_neg -= lr * grad_uneg

    return loss
```

### 4.2 PyTorch Skip-Gram with Negative Sampling

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, embed_dim)
        self.context = nn.Embedding(vocab_size, embed_dim)

    def forward(self, w, c, neg):
        v_w = self.center(w)                 # (batch, d)
        u_c = self.context(c)                # (batch, d)
        pos_score = torch.mul(u_c, v_w).sum(1)
        pos_loss = -torch.log(torch.sigmoid(pos_score))

        u_neg = self.context(neg)            # (batch, K, d)
        neg_score = torch.bmm(u_neg, v_w.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.log(torch.sigmoid(-neg_score)).sum(1)

        return (pos_loss + neg_loss).mean()

# Example usage:
model = SkipGramNS(vocab_size=5000, embed_dim=100)
optimizer = optim.Adam(model.parameters(), lr=0.002)

for centers, contexts, negatives in data_loader:
    optimizer.zero_grad()
    loss = model(centers, contexts, negatives)
    loss.backward()
    optimizer.step()
```

### 5. Visualization / Geometry

Visualize updates after one epoch on a tiny vocab:

1. Project embeddings to 2D (PCA or t-SNE).
2. Plot centers and contexts before and after training.
3. Notice true context pairs move closer, negative samples fan out.

```python
# assume E_before and E_after are (V, d) matrices
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

coords_b = PCA(n_components=2).fit_transform(E_before[:50])
coords_a = PCA(n_components=2).fit_transform(E_after[:50])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(coords_b[:,0], coords_b[:,1])
plt.title("Before Training")
plt.subplot(1,2,2)
plt.scatter(coords_a[:,0], coords_a[:,1])
plt.title("After 1 Epoch")
plt.show()
```

### 6. Common Pitfalls & Tips

- Picking K too small yields poor discrimination; too large wastes computation. Typical K=5–20.
- Noise distribution matters: unigram^0.75 gives better performance than uniform or raw frequency.
- Ensure negatives exclude true context words per batch.
- Learning rate scheduling can stabilize embedding training—decay as loss plateaus.
- Extremely rare words receive few updates; consider filtering by frequency or using subword methods.

### 7. Interview-Ready Insights

1. Explain why negative sampling approximates the softmax gradient only on sampled words, reducing cost from O(V) to O(K).
2. Derive the gradient of the loss w.r.t. center and context embeddings step by step.
3. Discuss the effect of the exponent 0.75 in the unigram noise distribution and its origin in the Word2Vec paper.
4. Compare negative sampling vs. hierarchical softmax: trade-offs in speed, implementation complexity, and performance.

### 8. Practice Exercises

1. Implement negative sampling training on a toy corpus (`["king","queen","man","woman","apple","fruit"]`). Plot the distance between “king” and “queen” over epochs.
    - Hint: precompute negative samples per (w,c) pair and reuse them.
2. Experiment with different noise exponents (0.5, 0.75, 1.0) and evaluate word similarity on a small set. Record which exponent yields highest cosine similarity on true synonyms.
3. Compare Skip-Gram NS vs. CBOW NS on a small dataset: train both models and report training time and nearest-neighbor quality.
4. Extend your NumPy implementation to support mini-batching. Ensure gradient updates accumulate correctly across the batch.

---

## GloVe Word Vectors

### 1. Direct Definition

GloVe (Global Vectors for Word Representation) is a count-based embedding method that builds a global word co-occurrence matrix and then factorizes it to learn dense word vectors. The resulting vectors capture both global statistics and local context relationships.

### 2. Concept Intuition

GloVe starts by counting how often each word appears near every other word in a large corpus.

It then seeks vectors whose dot product approximates the logarithm of those co-occurrence counts.

By fitting all pairs simultaneously, GloVe unifies global matrix factorization with the predictive power of local context methods.

Words that frequently co-occur end up with similar vectors, so semantic and syntactic patterns emerge geometrically.

### 3. Mathematical Breakdown

Given vocabulary size `V`, embedding dim `d`, and co-occurrence counts `X[i,j]`:

```python
# Weighting function
def f(x, x_max=100, alpha=0.75):
    return (x / x_max)**alpha if x < x_max else 1.0

# GloVe loss over all word pairs
J = 0
for i in range(V):
    for j in range(V):
        weight = f(X[i,j])
        term = dot(w[i], w_ctx[j]) + b[i] + b_ctx[j] - log(X[i,j])
        J += weight * term**2
```

- `w[i]`, `w_ctx[j]` ∈ ℝᵈ are target and context vectors.
- `b[i]`, `b_ctx[j]` are scalar biases.
- The model minimizes `J` via gradient descent.

### 4. Code & Practical Application

### 4.1 Building Co-Occurrence (NumPy)

```python
import numpy as np

def build_cooccurrence(corpus, window=2):
    vocab = list(set(corpus))
    idx = {w:i for i,w in enumerate(vocab)}
    V = len(vocab)
    X = np.zeros((V, V), dtype=np.int32)
    for pos, w in enumerate(corpus):
        for offset in range(-window, window+1):
            if offset == 0: continue
            j = pos + offset
            if 0 <= j < len(corpus):
                X[idx[w], idx[corpus[j]]] += 1
    return vocab, X
```

### 4.2 Simplified GloVe Training Loop

```python
def train_glove(X, d=50, lr=0.05, epochs=100, x_max=100, alpha=0.75):
    V = X.shape[0]
    W = np.random.randn(V, d) * 0.01
    W_ctx = np.random.randn(V, d) * 0.01
    b = np.zeros(V); b_ctx = np.zeros(V)
    for epoch in range(epochs):
        for i in range(V):
            for j in range(V):
                if X[i,j] == 0: continue
                weight = f(X[i,j], x_max, alpha)
                diff = np.dot(W[i], W_ctx[j]) + b[i] + b_ctx[j] - np.log(X[i,j])
                grad = 2 * weight * diff
                W[i]     -= lr * grad * W_ctx[j]
                W_ctx[j] -= lr * grad * W[i]
                b[i]     -= lr * grad
                b_ctx[j] -= lr * grad
    return W, W_ctx
```

### 4.3 Loading Pre-Trained GloVe into PyTorch

```python
import torch, torch.nn as nn

# Load glove.6B.100d.txt into glove_matrix (V,100)
embedding = nn.Embedding(num_embeddings=V, embedding_dim=100)
embedding.weight.data.copy_(torch.from_numpy(glove_matrix))
embedding.weight.requires_grad = False  # freeze if desired
```

### 5. Visualization / Geometry

Project the first 200 vectors to 2D with t-SNE or PCA to reveal clusters:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

coords = TSNE(n_components=2).fit_transform(W[:200])
plt.scatter(coords[:,0], coords[:,1], s=10)
for i, word in enumerate(vocab[:200]):
    plt.text(coords[i,0], coords[i,1], word, fontsize=6)
plt.show()
```

Analogies manifest as parallel vector differences (e.g., `king − man + woman ≈ queen`).

### 6. Common Pitfalls & Tips

- Setting `x_max` too low overweight rare co-occurrences; too high downplays informative counts.
- Poor initialization or large learning rates can diverge training—start small.
- Rare words have noisy statistics; apply a frequency cutoff or subword methods.
- Including both `w` and `w_ctx` vectors doubles storage—often you average them at the end.

### 7. Interview-Ready Insights

- Explain how GloVe’s objective relates to factorizing a shifted PMI matrix.
- Discuss the role of the weighting function `f(x)` in balancing rare vs. frequent pairs.
- Contrast GloVe with Word2Vec: count-based global statistics vs. local context prediction.
- Describe how you’d tune `d`, `x_max`, and `alpha` based on corpus size and domain.

### 8. Practice Exercises

1. Implement GloVe on a tiny corpus of 10 words. Vary `x_max` and `alpha`, then plot loss vs. epoch.
2. Load pre-trained GloVe (50d on 5k words), compute cosine similarity for synonyms vs. random pairs, and report averages.
3. Evaluate analogy accuracy on `king − man + woman` for your trained vectors.
4. Visualize how vector norms vary with word frequency—plot `||w[i]||` vs. frequency of word `i`.

---

## Sentiment Classification

### 1. Direct Definition

Sentiment classification is the task of assigning a label (e.g., positive, negative, neutral) or score to a piece of text that reflects its emotional or opinionated content. It typically uses a neural network that ingests word embeddings and outputs a sentiment probability.

### 2. Concept Intuition

- We convert each word into a dense vector so the model “sees” meaning, not just tokens.
- A sequence model (RNN, LSTM, GRU, or CNN) reads these embeddings and builds an internal representation of the entire sentence’s sentiment trajectory.
- The final representation is fed through a classifier (sigmoid or softmax) to predict sentiment.
- Good embeddings and sequence modeling capture negation (“not good”), intensifiers (“very happy”), and long-range dependencies (“I thought it would be bad, but it was great”).

### 3. Mathematical Breakdown

### 3.1 Embedding Lookup

```python
e_t = E[w_t]       # E: (V, d), w_t is index, e_t: (d,)
```

### 3.2 Sequence Model (LSTM)

```python
h_t, c_t = LSTM_Cell(e_t, (h_{t-1}, c_{t-1}))
# shapes: h_t, c_t: (hidden_dim,)
```

### 3.3 Classification Layer

For binary sentiment:

```python
z = W_y · h_T + b_y     # W_y: (1, hidden_dim), b_y: scalar
ŷ = sigmoid(z)          # ŷ ∈ (0,1)
```

### 3.4 Loss Function

```python
loss = -[ y*log(ŷ) + (1-y)*log(1-ŷ) ]
```

- (w_t) : t-th word index
- (e_t) : its embedding
- (h_T) : last hidden state after T steps
- (y) : true label (0 or 1)
- (ŷ) : predicted probability

### 4. Code & Practical Application

### 4.1 PyTorch Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, word2idx, max_len):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        idxs = [self.word2idx.get(w, self.word2idx['<UNK>'])
                for w in self.sentences[i].split()]
        # pad or truncate
        idxs = idxs[:self.max_len] + [self.word2idx['<PAD>']]*(self.max_len - len(idxs))
        return torch.tensor(idxs), torch.tensor(self.labels[i], dtype=torch.float32)

class SentimentModel(nn.Module):
    def __init__(self, V, d, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(V, d, padding_idx=0)
        self.lstm = nn.LSTM(d, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        e = self.embedding(x)              # (batch, seq_len, d)
        _, (h_T, _) = self.lstm(e)         # h_T: (1, batch, hidden_dim)
        logits = self.fc(h_T.squeeze(0))   # (batch, 1)
        return torch.sigmoid(logits).squeeze(1)

# Hyperparameters
d, hidden_dim, max_len = 100, 128, 50
batch_size, epochs, lr = 32, 5, 0.001

# Suppose train_sentences, train_labels, word2idx defined
train_ds = SentimentDataset(train_sentences, train_labels, word2idx, max_len)
loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = SentimentModel(V=len(word2idx), d=d, hidden_dim=hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
```

### 4.2 Using Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=V, output_dim=d, input_length=max_len, mask_zero=True),
    LSTM(hidden_dim),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, batch_size=32, epochs=5, validation_split=0.2)
```

### 5. Visualization / Geometry

- **Hidden State Trajectory:** Extract (h_t) for each position in a sentence. Project to 2D (PCA/t-SNE) to see sentiment evolution.
- **Decision Boundary:** For a 2-D toy embedding (d=2) and simple RNN, plot (h_T) of positive vs. negative examples—check separability by the sigmoid plane.

```python
# Example: project last hidden states
hs = []
ys = []
for x, y in loader:
    with torch.no_grad():
        _, (h_T, _) = model.lstm(model.embedding(x))
    hs.append(h_T.squeeze(0).numpy())
    ys.append(y.numpy())
hs = np.vstack(hs)
plt.scatter(hs[:,0], hs[:,1], c=np.concatenate(ys), cmap='coolwarm')
plt.title("Last hidden state projection by sentiment")
plt.show()
```

### 6. Common Pitfalls & Tips

- **Padding Mistakes:** Forgetting `padding_idx` in embeddings or not masking LSTM causes the model to learn from `<PAD>` tokens.
- **Class Imbalance:** If positives greatly outnumber negatives, use weighted loss or oversample minority class.
- **Overfitting:** Dropout on LSTM outputs or embeddings can improve generalization.
- **Sequence Length:** Too long leads to vanishing gradients; too short misses context. Pick a window that covers typical sentence length.
- **Batching Variable Length:** Use `pack_padded_sequence` and `pad_packed_sequence` in PyTorch for efficiency.

### 7. Interview-Ready Insights

- **Why LSTM/GRU:** They capture long-range dependencies and guard against vanishing/exploding gradients, unlike vanilla RNN.
- **Pooling Strategies:** Compare using last hidden state vs. average/max pooling over all (h_t).
- **Bidirectional RNNs:** Explain how a BiLSTM sees both past and future context before classification.
- **Attention Mechanisms:** Show how attention weights over hidden states can highlight sentiment-bearing words.
- **Transfer Learning:** Using pre-trained embeddings (GloVe/BERT) and fine-tuning can dramatically boost performance on small datasets.

### 8. Practice Exercises

1. **Toy Dataset**
    - Create 100 positive and 100 negative one-sentence samples (e.g., “I love X” vs. “I hate X”). Train a minimal LSTM model. Track train vs. validation accuracy.
2. **GloVe Initialization**
    - Load pre-trained GloVe 50-d vectors for your vocab, initialize `Embedding` layer, and compare convergence vs. random init.
3. **Bidirectional LSTM**
    - Extend the PyTorch model to a `nn.LSTM(..., bidirectional=True)` and adjust the classification layer accordingly. Measure any gain in accuracy.
4. **Attention Layer**
    - Implement a simple attention mechanism: learn a weight vector (a), compute scores (α_t = \tanh(W_a h_t)), normalize via softmax, and form context vector (\sum_t α_t h_t). Use it instead of `h_T` for classification.
5. **Error Analysis**
    - On a held-out test set, list misclassified examples. Identify patterns (negation, sarcasm, domain-specific language) and propose model or data fixes.

---

## Debiasing Word Embeddings

### 1. Direct Definition

Debiasing word embeddings is the process of identifying and removing unwanted societal biases (e.g., gender, race) encoded in pre-trained word vectors, so that downstream models make fairer predictions.

### 2. Concept Intuition

Word embeddings trained on text corpora capture not only semantics but also patterns of bias present in the data.

For example, “doctor” may lie closer to “man” than “woman.”

Debiasing finds a low-dimensional subspace that represents the bias (the “bias direction”) and then adjusts embeddings to remove that component for words that should be neutral.

This preserves the geometry of genuine semantic relationships while eliminating spurious associations.

### 3. Mathematical Breakdown

### 3.1 Identify Bias Direction

Build a list of definitional word pairs (e.g., (“he”,“she”), (“man”,“woman”)). Compute difference vectors and extract the top principal component:

```
diffs = [emb[w_pos] - emb[w_neg] for (w_pos, w_neg) in definitional_pairs]
# stack into matrix D of shape (N, d)
B = PCA(n_components=1).fit(diffs).components_[0]   # bias direction shape (d,)
```

### 3.2 Neutralize

For each “neutral” word embedding e, remove its projection onto B:

```
proj = np.dot(e, B) * B
e_neutral = e - proj
```

### 3.3 Equalize

For each pair of words that should be equidistant from bias (e.g., (“doctor”,“nurse”) if both are neutral professions), adjust so they are symmetric around the bias subspace:

```
m = (e1 + e2) / 2
m_proj = np.dot(m, B) * B
m_orth = m - m_proj

# component orthogonal to bias
e1_orth = e1 - np.dot(e1, B) * B
e2_orth = e2 - np.dot(e2, B) * B

# set equal distance
adjust = np.sqrt(1 - np.linalg.norm(m_orth)**2)
e1_equal = m_orth + adjust * (e1_orth - m_orth) / np.linalg.norm(e1_orth - m_orth)
e2_equal = m_orth + adjust * (e2_orth - m_orth) / np.linalg.norm(e2_orth - m_orth)
```

### 4. Code & Practical Application

```python
import numpy as np
from sklearn.decomposition import PCA

# Suppose word2idx, idx2word, and pre_trained_emb (V, d) loaded
def compute_bias_direction(pairs, emb):
    diffs = []
    for a, b in pairs:
        diffs.append(emb[word2idx[a]] - emb[word2idx[b]])
    D = np.stack(diffs)
    pca = PCA(n_components=1)
    pca.fit(D)
    return pca.components_[0]   # shape (d,)

def neutralize(word, emb, bias_dir):
    e = emb[word2idx[word]]
    proj = np.dot(e, bias_dir) * bias_dir
    emb[word2idx[word]] = e - proj

def equalize(pair, emb, bias_dir):
    i, j = word2idx[pair[0]], word2idx[pair[1]]
    e1, e2 = emb[i], emb[j]
    m = (e1 + e2) / 2
    m_proj = np.dot(m, bias_dir) * bias_dir
    m_orth = m - m_proj

    def_orth = lambda e: e - np.dot(e, bias_dir) * bias_dir
    e1o, e2o = def_orth(e1), def_orth(e2)

    # ensure unit norm in orthogonal subspace
    dist = np.linalg.norm(e1o - e2o) / 2
    if dist == 0:
        return
    e1_eq = m_orth + (dist / np.linalg.norm(e1o - m_orth)) * (e1o - m_orth)
    e2_eq = m_orth + (dist / np.linalg.norm(e2o - m_orth)) * (e2o - m_orth)

    emb[i], emb[j] = e1_eq, e2_eq
```

### PyTorch Integration

```python
import torch

# Assume glove_tensor shape (V, d)
bias_dir = torch.from_numpy(bias_dir_np).float()  # (d,)

# neutralize batch of indices
indices = torch.tensor([idx1, idx2, ...], dtype=torch.long)
E = glove_tensor.clone()
e = E[indices]                                        # (n, d)
proj = torch.outer(e @ bias_dir, bias_dir)            # (n, d)
E[indices] = e - proj
```

### 5. Visualization / Geometry

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# select words for plotting
words = ["man","woman","doctor","nurse","king","queen"]
embs = np.array([pre_trained_emb[word2idx[w]] for w in words])
embs_deb = np.copy(embs)

# apply neutralize/equalize steps to embs_deb here

coords = TSNE(n_components=2).fit_transform(embs)
coords_deb = TSNE(n_components=2).fit_transform(embs_deb)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(coords[:,0], coords[:,1])
for i, w in enumerate(words):
    plt.text(coords[i,0], coords[i,1], w)
plt.title("Original")

plt.subplot(1,2,2)
plt.scatter(coords_deb[:,0], coords_deb[:,1])
for i, w in enumerate(words):
    plt.text(coords_deb[i,0], coords_deb[i,1], w)
plt.title("Debiased")
plt.show()
```

Clusters of gendered words collapse onto the neutral subspace after debiasing.

### 6. Common Pitfalls & Tips

- Overzealous debiasing can remove legitimate semantic differences (e.g., actor vs. actress).
- Choosing definitional pairs poorly leads to an imprecise bias subspace.
- Rare words get noisy projections; consider frequency thresholds.
- Debiasing one bias type (gender) does not address others (race, religion).
- Evaluate using metrics like Word Embedding Association Test (WEAT) before and after.

### 7. Interview-Ready Insights

- Describe the two key steps: neutralize and equalize.
- Explain why the bias subspace is low-dimensional and how PCA finds it.
- Discuss limitations: static embeddings still carry indirect bias via neighbor shifts.
- Mention alternative methods: Hard Debias, INLP, adversarial training for contextual embeddings.
- Cite the seminal paper: “Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings,” EMNLP 2016.

### 8. Practice Exercises

1. **Gender Bias Direction**
    - Use a list of 10 gender definitional pairs to compute the bias direction on GloVe 100d.
    - Plot the norm of each embedding’s projection onto this direction.
2. **Neutralize Occupations**
    - Given a list of 20 profession words, neutralize their embeddings and measure cosine similarity to “he” and “she” before and after.
3. **Equalize Pairs**
    - Pick 5 equalize pairs (e.g., “doctor”/“nurse”, “scientist”/“teacher”). Apply equalization and verify that each pair is symmetric around the bias axis.
4. **WEAT Evaluation**
    - Implement the Word Embedding Association Test on a small word list before and after debiasing. Report changes in effect size.
5. **Extend to Race Bias**
    - Identify definitional pairs for race (e.g., “white”/“black”) and repeat neutralize/equalize. Observe which words shift most.

---