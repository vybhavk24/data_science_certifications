# DL_c5_m3

## Basic Models in Sequence Modeling

### 1. Direct Definition

A basic recurrent neural network (RNN) model is a neural architecture designed to process sequential data by maintaining a hidden “memory” state that’s updated at each step. Unlike feed-forward networks, an RNN shares parameters across time steps and captures dependencies in sequences of arbitrary length.

### 2. Concept Intuition

- You can think of an RNN as reading a sentence one word at a time, updating its internal “mental sketch” after each word.
- That sketch (the hidden state) carries forward everything learned so far, so the network can—in principle—remember earlier context when making later predictions.
- This ability to chain information across time makes RNNs well suited for tasks like language modeling, time-series forecasting, and speech recognition.

### 3. Mathematical Breakdown

```python
# At time step t:
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
y_t = W_hy · h_t + b_y
```

- x_t: input vector at time t
- h_{t-1}: previous hidden state
- W_xh: weights mapping input to hidden (shape: hidden_size × input_size)
- W_hh: recurrent weights (shape: hidden_size × hidden_size)
- b_h: bias for hidden state (shape: hidden_size)
- h_t: updated hidden state (shape: hidden_size)
- W_hy: weights mapping hidden to output (shape: output_size × hidden_size)
- b_y: bias for output (shape: output_size)
- tanh: elementwise activation that squashes values into (–1, 1)

**Why it works**

- W_xh projects the new input into the hidden space.
- W_hh carries the previous memory forward.
- Summing and applying tanh adds nonlinearity and prevents uncontrolled growth.

### 4. Code & Practical Application

### NumPy Implementation of One RNN Cell

```python
import numpy as np

class BasicRNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x_t, h_prev):
        z = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
        h_t = np.tanh(z)
        y_t = np.dot(self.W_hy, h_t) + self.b_y
        return h_t, y_t

# Toy example: predict next value in a simple sinusoid sequence
seq = [np.array([[np.sin(t)]]) for t in np.linspace(0, 2*np.pi, 50)]
rnn = BasicRNNCell(input_size=1, hidden_size=10, output_size=1)
h = np.zeros((10, 1))

for x in seq:
    h, y = rnn.forward(x, h)
    # compare y to the next true value to compute loss, then backprop (omitted for brevity)
```

You can swap to TensorFlow or PyTorch by using their built-in `tf.keras.layers.SimpleRNNCell` or `torch.nn.RNNCell`and running a full training loop with optimizers.

### 5. Visualization / Geometric Intuition

```
Time ──► t-1 ──► t ──► t+1
        │       │
      h_{t-1}  h_t  h_{t+1}
        │       │
      (W_hh) (W_hh)
        ▼       ▼
      Update hidden state via tanh.
```

- Each step applies an affine transformation + tanh, which geometrically is:
    1. Rotate/scale the hidden vector
    2. Add the new input contribution
    3. Squash into (–1, 1) hypercube
- Backpropagating through many steps multiplies lots of Jacobians, which can shrink (vanish) or blow up (explode).

### 6. Common Pitfalls & Tips

- Initializing W_hh too large leads to exploding gradients; too small causes vanishing gradients.
- Forgetting to reset or carry the hidden state correctly between sequences can leak information.
- Training very long sequences with a basic RNN is hard—consider truncated backpropagation through time.
- Always clip gradients to a reasonable range (e.g., [–5, 5]).

### 7. Interview-Ready Insights

- Explain that a basic RNN is a universal approximator for sequence-to-sequence mappings but struggles with long-term dependencies.
- Contrast depth in time (unfolding) vs. depth in space (stacked layers).
- Mention alternatives like LSTM/GRU that add gates to mitigate gradient issues.
- Be ready to derive the time-step update formula and discuss its computational complexity O(T·(hidden_size² + hidden_size·input_size)).

### 8. Practice Exercises

1. Implement `backward` for `BasicRNNCell` to compute gradients w.r.t. W_xh, W_hh, b_h given dL/dh_t and dL/dh_{t-1}.
    - Hint: use the derivative of tanh: 1 - h_t²
2. Create a mini-dataset of length-10 sequences where each next value is the sum of the previous two values. Train your RNN to predict the next number.
    - Hint: normalize your data to [–1, 1] for stable training.
3. Visualize hidden-state trajectories (h_1, h_2, …) in 2D by reducing hidden_size to 2. Plot how different input sequences carve paths in that space.

---

## Picking the Most Likely Sequence

### 1. Direct Definition

Decoding in sequence models is the process of choosing an output sequence [y₁, y₂, …, y_T] that maximizes the model’s conditional probability

```python
argmax_{y₁…y_T} P(y₁, y₂, …, y_T | x)
```

where x is the input (e.g., source sentence in translation, prefix in language modeling).

### 2. Concept Intuition

- A sequence model gives you at each time step t a probability distribution over the next token y_t.
- Picking the globally most likely full sequence means finding the sequence whose product of step-wise probabilities is largest.
- Exact search is exponential in T (vocabulary^T), so we use approximate algorithms:
    - **Greedy decoding** picks the single most likely token at each step—fast but myopic.
    - **Beam search** tracks the top-k partial sequences at each step—trades off compute for better global solutions.

### 3. Mathematical Breakdown

Let vocab size = V, beam width = B. We work in log space to avoid underflow:

```python
# For a candidate partial sequence y_{1:t} with log-prob scores
score(y_{1:t}) = sum_{i=1..t} log P(y_i | y_{1:i-1}, x)

# Greedy step at time t:
y_t = argmax_j log P(y_t = j | y_{1:t-1}, x)

# Beam search update:
# Given beams [(y_seq, score)] of size B, and next-step log-probs p of shape (V,)
new_beams = []
for seq, sc in beams:
    for j in range(V):
        new_seq = seq + [j]
        new_sc  = sc + p[j]
        add (new_seq, new_sc) to new_beams
# Keep top B by new_sc
beams = top_k(new_beams, k=B)
```

- Summing logs corresponds to multiplying probabilities.
- At the end, pick the beam with highest score, optionally normalizing by length.

### 4. Code & Practical Application

### Toy Beam Search in NumPy

```python
import numpy as np

def beam_search_step(beams, log_probs, B):
    # beams: list of (seq, score)
    all_candidates = []
    for seq, sc in beams:
        for token in range(log_probs.shape[1]):
            cand_seq = seq + [token]
            cand_sc  = sc + log_probs[len(seq), token]
            all_candidates.append((cand_seq, cand_sc))
    # select top B
    ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    return ordered[:B]

# simulate model log-probs for T=5, V=10
np.random.seed(0)
logits = np.random.randn(5, 10)
log_probs = logits - logits.max(axis=1, keepdims=True)
log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=1, keepdims=True))

# initialize beam
beams = [([], 0.0)]
B = 3
for t in range(log_probs.shape[0]):
    beams = beam_search_step(beams, log_probs, B)

best_seq, best_score = beams[0]
print("Best sequence:", best_seq)
print("Log-prob score:", best_score)
```

You can swap in a real language model’s `logits` at each step and wrap this logic in a loop that stops on an end-of-sequence token.

### 5. Visualization / Geometric Intuition

```
        Step t=1       Step t=2       Step t=3
beams    ┌───┐          ┌───────┐      ┌───────┐
  B=2    │A: -0.5│      │A→B: -1.2│    │A→B→C: -2.1│
         ├───┤          ├───────┤      ├───────┤
         │C: -0.7│      │C→A: -1.4│    │C→A→D: -2.6│
         └───┘          └───────┘      └───────┘
```

- Each beam is a path through a tree of possible tokens.
- Beam width B prunes the tree to the B highest-scoring paths at each level.
- In log-prob space, longer sequences accumulate more negative scores, so you can visualize beams diverging and some paths dying out early.

### 6. Common Pitfalls & Tips

- Without length normalization, beams often favor very short sequences (higher average log-prob).
- Picking a beam width B too small collapses to greedy; too large slows decoding and may include low-quality paths.
- Always include a special end-of-sequence (EOS) token and stop expanding beams that already ended.
- A small positive length penalty (e.g., dividing score by t^α) can improve fluency in translation/generation.

### 7. Interview-Ready Insights

- Describe why exact argmax over all sequences is intractable (combinatorial explosion V^T).
- Contrast greedy vs beam search: local optimum vs global search with limited breadth.
- Discuss log vs probability space to handle underflow.
- Explain length penalty to correct bias, and be ready to write the penalty formula:
    
    ```python
    score_norm = score / (len(seq) ** alpha)
    ```
    
- Mention alternative decoding methods: top-k sampling, nucleus (top-p) sampling for creative text generation.

### 8. Practice Exercises

1. Implement beam search with length normalization:
    - Use α=0.7, compare outputs with and without normalization.
    - Hint: divide cumulative log-prob by (t ** α).
2. Apply greedy and beam decoding on a pretrained small LSTM language model (e.g., character-level on “hello world” text).
    - Compare generated text quality at B=1 (greedy), B=3, B=5.
3. Explore sampling strategies: write a function that given a log-prob vector returns a token by:
    - Top-k sampling (restrict to k highest tokens)
    - Top-p (nucleus) sampling where you pick the smallest set whose cumulative prob ≥ p.
    - Hint: convert log-probs back to probs, sort, accumulate.

---

## Beam Search

### 1. Direct Definition

Beam search is a heuristic sequence decoding algorithm that, at each time step, keeps the top B partial hypotheses (the “beam”) ranked by their cumulative log-probability. Instead of committing to the single highest-probability token like greedy decoding, beam search explores multiple paths and prunes to the best B, trading off computation for better global sequence quality.

### 2. Concept Intuition

Imagine you’re navigating a maze with many forks.

- Greedy decoding picks the best turn at each fork and never looks back.
- Beam search carries B explorers; at each fork, each explorer splits into all possible next moves, then only the top B overall explorers continue.This lets you recover from a locally suboptimal choice, increasing the chance of finding the true highest-probability route through the maze of possible sequences.

### 3. Mathematical Breakdown

At time step t, we maintain B sequences with their scores:

```python
# beams: list of tuples (seq, score)
# seq = [y1, y2, …, y_{t-1}], score = sum_{i=1..t-1} log P(y_i | y_{<i}, x)

# Expand each beam by all tokens in vocab (size V)
new_beams = []
for seq, score in beams:
    log_probs = model.log_probabilities(seq, x)[t]      # shape (V,)
    for token in range(V):
        new_seq   = seq + [token]
        new_score = score + log_probs[token]            # add log P(token)
        new_beams.append((new_seq, new_score))

# Prune back to top B sequences
beams = sorted(new_beams, key=lambda b: b[1], reverse=True)[:B]
```

Key points:

- We sum log-probs for numerical stability (log a + log b = log (ab)).
- Beam width B controls breadth: B=1 is greedy, B→∞ approaches exact search.
- Optionally apply length normalization:
    
    ```python
    normalized_score = score / (len(seq) ** alpha)
    ```
    

### 4. Code & Practical Application

Below is a PyTorch-style implementation of beam search for an autoregressive model:

```python
import torch
import torch.nn.functional as F

def beam_search(model, encoder_outputs, beam_width=3, max_len=20, eos_token=2, alpha=0.7):
    # Initialize beams: (sequence, score)
    beams = [([model.sos_token], 0.0)]
    completed = []

    for t in range(max_len):
        all_candidates = []
        for seq, score in beams:
            if seq[-1] == eos_token:
                # Passed EOS: move to completed
                completed.append((seq, score))
                continue

            # Prepare decoder input and hidden state
            decoder_input = torch.tensor([seq], device=model.device)
            decoder_logits = model.decode(decoder_input, encoder_outputs)  # shape (1, V)
            log_probs = F.log_softmax(decoder_logits[:, -1, :], dim=-1).squeeze(0)

            # Expand
            topk_logps, topk_tokens = log_probs.topk(beam_width)
            for logp, token in zip(topk_logps.tolist(), topk_tokens.tolist()):
                new_seq   = seq + [token]
                new_score = score + logp
                # Apply length normalization
                norm_score = new_score / ((len(new_seq) ** alpha))
                all_candidates.append((new_seq, norm_score, new_score))

        # Keep top B normalized scores
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(seq, raw_score) for seq, _, raw_score in ordered]

        if not beams:
            break

    # Add any unfinished beams to completed
    completed += beams
    # Return the sequence with highest raw score
    best_seq, best_score = max(completed, key=lambda x: x[1])
    return best_seq, best_score
```

This function:

- Carries forward only non-EOS beams each step.
- Applies top-k expansion and prunes by normalized score.
- Returns the highest raw-score completed sequence.

### 5. Visualization / Geometric Intuition

```
                 vocab
                 ┌─────────┬─────────┐
          Step t  A: -0.1  B: -1.2  C: -3.0
           beams ┌───┐
             B=2 │[‹SOS›], 0.0│
                 └───┘
                    ↓ expand
            all candidates  [‹SOS›, A], -0.1  [‹SOS›, B], -1.2  [‹SOS›, C], -3.0
                    ↓ prune to top-2
           new beams  [‹SOS›, A], -0.1    [‹SOS›, B], -1.2

 Next step:
 each beam expands to V new nodes; the tree grows width=V then prunes back to B

```

Geometrically, each partial sequence is a path through a vast V-ary tree. Beam search clips the tree’s breadth to B most promising branches, trading exactness for tractability.

### 6. Common Pitfalls & Tips

- Forgetting log-probs: summing raw probabilities overflows or underflows easily.
- No length normalization: favors very short sequences (fewer negative logs).
- EOS handling: if you keep expanding completed beams, you waste capacity and may never finish.
- Beam width too small collapses to greedy; too large slows decoding and can introduce low-quality paths.
- Vocabulary size V large ⇒ expansion cost B×V. Consider vocabulary pruning or caching decoder states.

### 7. Interview-Ready Insights

- Explain time/space complexity per step: O(B·V) expansions, sorting O(B·V log (B·V)).
- Discuss why log-space is critical for long sequences.
- Describe length normalization and coverage penalty:
    
    ```python
    score_norm = raw_score / (len(seq) ** alpha)  # length penalty
    coverage = sum(attention_weights)            # for translation coverage
    score_cov  = score_norm + beta * coverage
    ```
    
- Compare beam search to sampling: deterministic vs stochastic exploration.
- Mention enhancements: diverse beam search, minimum Bayes-risk decoding.

### 8. Practice Exercises

1. **Implement Beam Search With and Without Length Normalization**
    - Train a toy character-level RNN on text (e.g., “hamlet.txt”).
    - Generate sequences at B=3: once dividing by t⁰.⁷, once raw.
    - Compare lengths and coherence.
2. **Analyze Beam Width vs Quality Trade-Off**
    - For B ∈ {1, 2, 5, 10}, decode from a small English→French transformer.
    - Measure average BLEU score and decoding time.
    - Plot BLEU vs time and identify diminishing returns.
3. **Implement Diverse Beam Search**
    - Split B=6 beams into 2 groups of 3; penalize tokens chosen by earlier groups.
    - Show how hypotheses in different groups become more varied.

---

## Refinements to Beam Search

### 1. Direct Definition

Refinements to beam search are heuristic tweaks and penalty terms added to the vanilla beam-search algorithm to:

- Counteract its biases (e.g., favoring short sequences)
- Encourage diverse outputs
- Enforce constraints (e.g., mandatory tokens)
- Improve speed or memory usage

### 2. Concept Intuition

- Vanilla beam search simply picks top B paths by cumulative log-prob. That can lead to overly short text, repetitive loops, or lack of diversity.
- By adding penalty terms or varying the beam adaptively, we guide the search toward sequences that are fluent, sufficiently long, non-redundant, and diverse.
- Think of each refinement as adding a new “compass” needle that nudges explorers in the maze toward different desirable traits.

### 3. Mathematical Refinements

### 3.1 Length Penalty

Compensates for log-prob accumulating more negative mass as sequence grows.

```python
# raw_score = sum_{i=1..t} log P(y_i|...)
normalized_score = raw_score / (t ** alpha)
```

– α ∈ [0,1]: 0 = no penalty (raw), 1 = full average

### 3.2 Coverage Penalty (for Attention Models)

Encourages the model to attend to all source tokens.

```python
coverage = sum_{i=1..T_dest, j=1..T_src} min(a_{ij}, 1 - a_{ij})
score_cov = normalized_score + beta * coverage
```

– a_{ij}: attention weight on source j at target step i

– β ≥ 0: strength of coverage reward

### 3.3 Diversity Penalty (Diverse Beam Search)

Splits beams into G groups and penalizes tokens chosen by earlier groups.

```python
# for group g, adjust log_probs:
log_probs_g[j] -= gamma * count_{g'<g}(token=j in beam g')
```

– γ ≥ 0: diversity strength

### 3.4 Constrained Beam Search

Enforces required tokens or structure via a mask M:

```python
# M(t, token) = -∞ if token forbidden at step t
adjusted_log_probs = log_probs + M[t]
```

### 3.5 Dynamic Beam Sizing

Adapt beam width B_t based on entropy H_t of distribution at step t:

```python
H_t = -sum_k p_t[k] * log p_t[k]
B_t = clip(round(c * H_t), min_B, max_B)
```

– c: scaling constant

### 4. Code & Practical Application

Below is a PyTorch-style pseudo-implementation incorporating length and coverage penalties plus diversity grouping.

```python
import torch
import torch.nn.functional as F

def refined_beam_search(model, encoder_states,
                        beam_width=5, max_len=30,
                        alpha=0.6, beta=0.1,
                        groups=2, gamma=0.5,
                        eos_token=2):
    # beams: [(seq, raw_score, coverage_vector)]
    beams = [([model.sos_token], 0.0, torch.zeros(encoder_states.size(1)))]
    completed = []

    for t in range(max_len):
        all_cands = []
        for g in range(groups):
            # select top beams for group g
            group_beams = beams[g::groups]
            for seq, raw_score, cov in group_beams:
                if seq[-1] == eos_token:
                    completed.append((seq, raw_score))
                    continue

                logits, attn = model.decode_step(seq, encoder_states)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                # coverage update
                cov_new = cov + attn.squeeze(0)
                cov_pen = beta * torch.sum(torch.min(cov_new, 1 - cov_new))

                # diversity penalty
                for token in range(log_probs.size(-1)):
                    count = sum(1 for b, *_ in beams[:g] if b[-1] == token)
                    log_probs[token] -= gamma * count

                # top-k expand
                topk_logp, topk_tok = log_probs.topk(beam_width)
                for logp, tok in zip(topk_logp.tolist(), topk_tok.tolist()):
                    new_seq = seq + [tok]
                    raw_new = raw_score + logp
                    # length norm
                    norm_score = raw_new / ((len(new_seq))**alpha)
                    score = norm_score + cov_pen
                    all_cands.append((new_seq, raw_new, cov_new, score))

        # prune by refined score
        ordered = sorted(all_cands, key=lambda x: x[3], reverse=True)[:beam_width]
        beams = [(s, r, c) for s, r, c, _ in ordered]
        if not beams:
            break

    completed += [(s, r) for s, r, _ in beams]
    best_seq, best_score = max(completed, key=lambda x: x[1])
    return best_seq
```

**Real-world tip:** Plug this into your Transformer or RNN decoder; track attention matrices for coverage; adjust hyperparameters α, β, γ on a validation set.

### 5. Visualization / Geometric Intuition

```
              refinement axes
                   ▲
     diversity  •──┼──•  coverage
                   │
               length bias
```

- Vanilla beam search sits at the center (no preference).
- Pushing toward “length” nudges beams downward (longer paths).
- Pushing toward “coverage” nudges beams right (uses all source tokens).
- Pushing toward “diversity” nudges beams upward (fills multiple groups).Each beam score is a dot-product of its raw log-prob vector with these penalty axes.

### 6. Common Pitfalls & Tips

- Over-penalizing length (α too high) yields run-on, ungrammatical outputs.
- Excessive coverage (β too large) forces unnatural attention shifts.
- Diversity γ close to zero collapses to standard beams; too large yields junk tokens.
- Constrained search masks can inadvertently block all valid continuations—always ensure a fallback token is allowed.
- Dynamic beam sizing adds overhead; tune c to avoid wild fluctuations.

### 7. Interview-Ready Insights

- Explain why and how length, coverage, and diversity penalties correct beam search’s biases.
- Be ready to write the combined scoring formula:
    
    ```python
    score = raw_score / (t**alpha) + beta * coverage − gamma * repetition_count
    ```
    
- Discuss time complexity: penalty terms add O(B·V) work but guide quality.
- Mention advanced variants:
    - **Minimum Bayes-Risk decoding** (risk-averse choice)
    - **Noisy parallel approximate decoding** (adds Gaussian noise to logits)
    - **Constrained decoding** via finite-state machines

### 8. Practice Exercises

1. **Tune Length vs Coverage**
    - On an English→German model, run beam search with (α,β) ∈ {(0.6,0.0), (0.6,0.1), (1.0,0.2)}.
    - Measure BLEU and average length. Plot trade-off curves.
2. **Implement Diverse Beam with Groups**
    - Split B=8 into groups G=4. Compare generated translations’ lexical overlap across groups.
    - Visualize top-2 candidates per group for a sample sentence.
3. **Dynamic Beam Width**
    - Write a beam search where B_t scales with entropy H_t.
    - Test on a toy char-RNN: see if uncertain steps (high H_t) spawn more beams, and confident steps shrink them.

---

## Error Analysis in Beam Search

### 1. Direct Definition

Error analysis in beam search is the systematic process of diagnosing why the sequences produced by beam search diverge from the desired (reference) outputs. It breaks down failures into “search errors” (beam didn’t include the reference) and “model errors” (beam included the reference but didn’t rank it highest), and measures phenomena like length bias, repetition, and coverage gaps.

### 2. Concept Intuition

- Beam search is a heuristic: it may prune the correct sequence (search error) or rank a suboptimal path above the correct one (model error).
- By quantifying these errors and their magnitudes—e.g., how often the gold sequence falls outside the beam or how badly beam outputs repeat tokens—we pinpoint whether to improve the search strategy or retrain the model itself.
- This targeted insight steers hyperparameter tuning (beam width, length penalty) and model adjustments (coverage mechanisms, repetition penalties).

### 3. Mathematical Breakdown

```python
# Given N examples, beam width B, and reference sequence ref_seq
# Let beam_k[i] be the top-k output for example i, and score(seq) its log-prob

# 1. Search Error Rate:
search_error = 1 - (count_i [ref_seq ∈ beam_B[i]] / N)

# 2. Model Error Rate (conditional on ref in beam):
model_error = count_i [ref_seq ∈ beam_B[i] and beam_1[i] ≠ ref_seq] \
            / count_i [ref_seq ∈ beam_B[i]]

# 3. Reference vs. Best Margin:
margin_i = score(ref_seq) - score(beam_1[i])

# 4. Length Ratio (hyp vs. ref):
length_ratio_i = len(beam_1[i]) / len(ref_seq)

# 5. Repetition Rate of n-grams in hypothesis:
rep_rate_i = count_repeated_ngrams(beam_1[i], n) / total_ngrams(beam_1[i], n)
```

- search_error pinpoints beam’s pruning mistakes.
- model_error reveals scoring mistakes by the model’s learned distribution.
- margin_i shows how much more log-prob the model assigns to its own choice versus the true sequence.
- length_ratio_i and rep_rate_i quantify biases toward short or repetitive outputs.

### 4. Code & Practical Application

```python
import numpy as np
from collections import Counter

def compute_error_metrics(all_refs, all_beams, all_scores, n=3):
    """
    all_refs: list of reference token lists
    all_beams: list of lists of hypotheses (each a list of tokens), size B
    all_scores: list of lists of log-prob scores matching all_beams
    """
    N = len(all_refs)
    B = len(all_beams[0])

    # Search & model errors
    ref_in_beam = [int(ref in beam) for ref, beam in zip(all_refs, all_beams)]
    search_error = 1 - sum(ref_in_beam)/N

    model_errors = []
    margins = []
    length_ratios = []
    rep_rates = []

    for ref, beam, scores in zip(all_refs, all_beams, all_scores):
        # margin: ref score minus top hypothesis score
        try:
            ref_score = scores[beam.index(ref)]
        except ValueError:
            ref_score = -np.inf
        top_score = scores[0]
        margins.append(ref_score - top_score)

        # model error conditional
        if ref in beam:
            model_errors.append(int(beam[0] != ref))

        # length ratio
        length_ratios.append(len(beam[0]) / len(ref))

        # repetition rate
        hy = beam[0]
        ngrams = [tuple(hy[i:i+n]) for i in range(len(hy)-n+1)]
        rep = sum(1 for g,c in Counter(ngrams).items() if c>1)
        total = len(ngrams) or 1
        rep_rates.append(rep/total)

    model_error = sum(model_errors)/sum(ref_in_beam or [1])
    return {
        'search_error': search_error,
        'model_error': model_error,
        'avg_margin': np.mean(margins),
        'avg_length_ratio': np.mean(length_ratios),
        'avg_rep_rate': np.mean(rep_rates)
    }
```

- Collect `all_beams` by running beam search on your validation set and storing the top B hypotheses and their log-prob scores.
- Use these metrics to decide: increase beam width (reduces search_error) or adjust length/diversity penalties (improves length_ratio, rep_rate).

### 5. Visualization / Geometric Intuition

```
             Model Error ↗
                  •
 search error •
               ↖
```

- Imagine a 2D plot where x-axis is search_error and y-axis is model_error.
- Points in the upper-left indicate mostly model errors (beam contains the reference but misranks it).
- Points in the lower-right indicate search errors dominate (beam pruned the reference).
- Length_ratio and rep_rate can be overlaid as color gradients to spot correlations (e.g., high rep_rate often co-occurs with high model_error).

### 6. Common Pitfalls & Tips

- Ignoring oracle analysis: always check if the reference is in the beam before blaming the model.
- Relying solely on BLEU: a high BLEU doesn’t reveal if beams are too short or overly repetitive.
- Aggregating over heterogeneous examples: separate error metrics by sequence length or domain to get finer-grained insights.
- Forgetting to calibrate scores before margin computation: ensure you compare raw (unnormalized) log-prob scores.

### 7. Interview-Ready Insights

- Explain the decomposition:“Total error = search_error + (1 – search_error) × model_error.”
- Describe margin analysis to quantify how “confidently wrong” the model is.
- Mention how oracle BLEU (best BLEU among beams) upper-bounds achievable performance given B.
- Discuss advanced fixes: minimum Bayes risk decoding to optimize expected task loss directly, or curriculum beam widening (increase B for longer sequences).

### 8. Practice Exercises

1. **Oracle vs Actual BLEU**
    - Run beam search (B=5) on a dev set. Compute BLEU of top-1 and oracle BLEU (best-scoring reference-like beam). Compare gap.
2. **Error Decomposition by Length**
    - Bucket sentences into short/medium/long. Compute search_error and model_error in each bucket. Analyze trends.
3. **Repetition Profiling**
    - For B=1 vs B=5, plot average rep_rate for n = 1…4. Visualize how beam width impacts repetition.
4. **Margin Histogram**
    - Plot a histogram of margins across examples. Identify how many margin_i < 0 (model prefers its own hypothesis over reference) versus margin_i ≥ 0.

---

## Attention Model Intuition

### 1. Direct Definition

Attention is a mechanism that lets a model dynamically weight and combine all elements in a sequence when producing each output. Rather than compressing the entire input into one fixed vector, attention computes a context vector as a weighted sum of “value” vectors, where the weights depend on similarity between a “query” and each “key.”

### 2. Concept Intuition

- Imagine translating a sentence word by word. For each target word, you glance back at only the relevant source words, not the entire sentence equally.
- Attention mimics this selective focus: each query “asks” which keys are most relevant, and gathers information from values accordingly.
- This dynamic routing of information overcomes RNN bottlenecks, capturing long-range dependencies and parallelizing sequence processing.

### 3. Mathematical Breakdown

```python
# Given query Q, keys K, values V (shapes: Q:(t_q,d), K:(t_k,d), V:(t_k,d_v))
scores = Q @ K.T                       # shape: (t_q, t_k)
scaled_scores = scores / sqrt(d)       # scale by sqrt(d)
weights = softmax(scaled_scores, axis=1)  # shape: (t_q, t_k)
context = weights @ V                  # shape: (t_q, d_v)
```

- `scores[i,j]` measures similarity between query i and key j.
- Scaling prevents extremely large dot-products when d grows.
- Softmax turns scores into a probability distribution over keys.
- Context is a weighted sum of values.

### 4. Code & Practical Application

### NumPy: Scaled Dot-Product Attention

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q.dot(K.T) / np.sqrt(d)
    # numeric stability
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return weights.dot(V), weights

# Toy example
Q = np.array([[1., 0., 1.]])           # 1 query
K = np.array([[1., 0., 0.],
              [0., 1., 1.],
              [1., 1., 0.]])
V = np.array([[1., 2.],
              [3., 4.],
              [5., 6.]])
context, attn_weights = scaled_dot_product_attention(Q, K, V)
print("Context:", context)
print("Weights:", attn_weights)
```

### PyTorch: Single‐Head Attention Layer

```python
import torch
import torch.nn.functional as F

class SingleHeadAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        Q = self.W_q(x_q)        # shape: (batch, t_q, d_model)
        K = self.W_k(x_kv)       # shape: (batch, t_k, d_model)
        V = self.W_v(x_kv)       # shape: (batch, t_k, d_model)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model**0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn
```

### 5. Visualization / Geometry

```
    Q ─┐
       ▼
   Dot product ──► scores matrix (t_q×t_k)
       │             ↑
       ▼             │ softmax
Attention weights ──► weights matrix (t_q×t_k)
       │
       ▼
 Weighted sum of V ──► context vectors
```

- Geometrically, queries and keys live in the same vector space.
- Dot product projects one onto another: large when aligned, small when orthogonal.
- Softmax turns these alignments into attention “heatmaps.”
- Context is a convex combination of values guided by that heatmap.

### 6. Common Pitfalls & Tips

- Missing scale factor `1/√d` leads to tiny gradients or peaked softmax.
- Forgetting numeric stability in softmax can overflow.
- Mixing up sequence lengths: ensure Q’s time-axis aligns with output length, K/V with input length.
- Not masking future positions (in decoder self-attention) allows information leak.
- Using dot-product on unnormalized vectors exaggerates length effects—consider layer normalization.

### 7. Interview-Ready Insights

- Explain why scaling by `√d` is critical: controls variance of dot-product as d grows.
- Contrast additive (Bahdanau) vs dot-product (Luong/Vaswani) attention:
    - Additive uses a feed-forward network on `[Q;K]`.
    - Dot-product is faster with optimized matrix multiplications.
- Describe self-attention’s O(T²·d) complexity and how multi-head splits d into parallel subspaces for richer representations.
- Be ready to derive backpropagation through softmax and discuss mask implementation.

### 8. Practice Exercises

1. **From Scratch:**
    - Implement additive attention:
        
        ```python
        scores = v.T @ tanh(W_q Q + W_k K)
        ```
        
    - Compare performance on a toy English→French corpus.
2. **Multi-Head Attention:**
    - Extend the `SingleHeadAttention` class to `MultiHeadAttention` with h heads.
    - Verify that concatenation of heads followed by a final linear layer recovers original `d_model`.
3. **Visualization:**
    - Use Matplotlib to plot attention weights for a sample sentence in a pretrained Transformer.
    - Interpret which source tokens the model focuses on for each target token.
4. **Masking Challenge:**
    - Implement causal mask for decoder self-attention: prevent positions > t attending to t.
    - Test that the mask enforces zeros above the diagonal in the weight matrix.

---

## Attention Model

### 1. Direct Definition

An attention model in sequence-to-sequence architectures computes, at each decoding step, a context vector as a weighted sum of encoder hidden states. The weights—attention scores—measure relevance between a decoder query and each encoder key/value pair, enabling the decoder to dynamically “focus” on different input positions.

### 2. Concept Intuition

- In vanilla encoder–decoder RNNs, all input information is squashed into a single fixed-length vector.
- Attention lets the decoder peek back at the full sequence of encoder states, not just the final one.
- You can imagine translating a sentence: for each target word, you glance at the specific source words you need, rather than trying to remember the whole sentence in your head at once.
- This selective focusing handles long-range dependencies, aligns input/output tokens, and improves gradients by creating shortcut connections.

### 3. Mathematical Breakdown

### 3.1 Additive (Bahdanau) Attention

```python
# Given decoder hidden state s_t (shape d), encoder states h_i (shape d)
score_i = v.T · tanh(W_s · s_t + W_h · h_i)    # scalar score for position i
α_i     = exp(score_i) / sum_j exp(score_j)   # softmax over positions
context = sum_i α_i * h_i                     # weighted sum of encoder states
```

### 3.2 Multiplicative (Luong) Attention

```python
# Given s_t (shape d), h_i (shape d)
score_i = s_t.T · W · h_i                     # faster dot-product variant
α_i     = softmax(score)                      # normalize scores
context = sum_i α_i * h_i
```

### 3.3 Scaled Dot-Product Attention (Transformer)

```python
# Q: queries (t_q×d), K: keys (t_k×d), V: values (t_k×d_v)
scores        = Q @ K.T                       # shape: (t_q, t_k)
scaled_scores = scores / sqrt(d)              # prevent large dot-products
weights       = softmax(scaled_scores, axis=1)  # attention map
context       = weights @ V                   # shape: (t_q, d_v)
```

### 4. Code & Practical Application

### NumPy: Bahdanau Attention

```python
import numpy as np

def bahdanau_attention(s_t, H_enc):
    # s_t: (d,); H_enc: (T, d)
    W_s = np.random.randn(d, d)
    W_h = np.random.randn(d, d)
    v   = np.random.randn(d)

    # compute scores
    scores = np.tanh(H_enc @ W_h.T + s_t @ W_s.T) @ v
    # numeric stability
    scores -= np.max(scores)
    weights = np.exp(scores) / np.sum(np.exp(scores))
    context = weights @ H_enc
    return context, weights

# toy encoder states and decoder state
T, d = 5, 4
H_enc = np.random.randn(T, d)
s_t   = np.random.randn(d)
context, attn_w = bahdanau_attention(s_t, H_enc)
print("Context vector:", context)
print("Attention weights:", attn_w)
```

### PyTorch: Integrating Attention in a Seq2Seq

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.decoder = nn.GRU(d_model + d_model, d_model, batch_first=True)
        self.attn_W  = nn.Linear(d_model * 2, d_model)
        self.attn_v  = nn.Linear(d_model, 1, bias=False)
        self.out     = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # src: (B, T_src, d), tgt: (B, T_tgt, d)
        enc_out, _ = self.encoder(src)  # (B, T_src, d)
        dec_h      = torch.zeros_like(enc_out[:, 0, :]).unsqueeze(0)
        outputs    = []
        for t in range(tgt.size(1)):
            # compute attention scores
            dec_input = tgt[:, t, :].unsqueeze(1)              # (B,1,d)
            dec_h_s   = dec_h.transpose(0,1)                   # (B,1,d)
            repeat_h  = dec_h_s.expand(-1, enc_out.size(1), -1)  # (B,T_src,d)

            energy = torch.tanh(self.attn_W(torch.cat((repeat_h, enc_out), dim=2)))  # (B,T_src,d)
            scores = self.attn_v(energy).squeeze(2)            # (B,T_src)
            weights= F.softmax(scores, dim=1).unsqueeze(1)     # (B,1,T_src)

            context= weights @ enc_out                         # (B,1,d)
            rnn_input = torch.cat((dec_input, context), dim=2)  # (B,1,2d)
            out, dec_h = self.decoder(rnn_input, dec_h)        # (B,1,d)
            token_logits = self.out(out.squeeze(1))            # (B, vocab)
            outputs.append(token_logits)
        return torch.stack(outputs, dim=1)  # (B, T_tgt, vocab)
```

### 5. Visualization / Geometry

```
        Decoder state s_t
               │
     ┌─────────┴─────────┐
     │                   │
 Compute similarity     Add bias & tanh   Normalize (softmax)
     │                   │                    │
    scores              └──► weights ◄───────┘
     │                          │
     │                          ▼
 Context = ∑ weights_i · encoder_state_i
```

- Geometrically, each encoder state is a vector in d-dim space.
- The decoder state projects into that same space.
- Attention computes alignment scores (high when vectors align), then blends encoder states into the context.

### 6. Common Pitfalls & Tips

- Forgetting to mask padded encoder positions leads to wasted attention on `<pad>`.
- Omitting the scaling factor `1/√d` in dot-product attention can lead to extremely peaked or flat softmax.
- Mixing up batch/time dimensions in implementations often causes shape errors.
- Not applying `detach()` or caching hidden states can blow up memory in PyTorch loops.
- In self-attention, failing to mask future positions leaks information and invalidates autoregressive generation.

### 7. Interview-Ready Insights

- Contrast additive vs multiplicative attention:
    - Additive is more flexible for small d; multiplicative is faster at large d.
- Explain why Transformers moved to multi-head attention:
    - Each head attends to different subspaces, enriching representation.
- Discuss complexity: O(T_src·T_tgt·d) per layer and strategies like sparse attention to scale to long sequences.
- Be ready to write the full forward pass of a single attention head and explain masking for both padding and causality.

### 8. Practice Exercises

1. Implement **Luong dot-product** attention in NumPy and compare its attention weights to the additive version on the same random data.
2. Extend the PyTorch `Seq2SeqAttention` class to **multi-head** attention: split d into h heads, compute in parallel, then concatenate. Verify outputs match single-head when h=1.
3. Visualize **attention heatmaps** on a toy translation task (e.g., English digits “one two three”). Plot a matrix of attention weights between source and generated tokens.
4. Implement **causal masking** in scaled dot-product attention: ensure position i cannot attend to j>i, and test on a random sequence to confirm upper-triangular mask is applied.

---

## Speech Recognition

### 1. Direct Definition

Speech recognition converts a raw audio waveform x[n] into a discrete symbol sequence y₁…y_T (phonemes, characters, or words). Modern end-to-end systems map time-frequency features through neural encoders and decoders—typically using Connectionist Temporal Classification (CTC) or sequence-to-sequence with attention—to produce text.

### 2. Concept Intuition

- An audio signal is a continuous wave; to process it, we slice it into overlapping frames, compute spectral features (e.g., Mel-spectrogram), and feed that time-series into a neural network.
- A CTC‐based model learns to align input frames to output tokens by inserting blank labels and summing over all valid frame‐to‐token alignments.
- An attention‐based seq2seq model learns to “focus” on relevant time frames for each output token, dynamically weighting encoder states.
- Both methods turn an unsegmented, variable‐length speech signal into a well-formed text sequence.

### 3. Mathematical Breakdown

### 3.1 STFT & Mel-Spectrogram

```python
# Short-time Fourier transform
X[t, k] = ∑_{n=0..N−1} x[n + t·hop] · w[n] · exp(−j·2π·k·n/N)

# Mel filterbank conversion
mel(f) = 2595 · log10(1 + f/700)
```

– x[n]: waveform samples

– w[n]: window (e.g., Hamming) length N

– hop: frame shift

– X[t, k]: complex spectrum at time frame t, frequency bin k

### 3.2 CTC Loss

```python
# Let π ∈ {labels ∪ blank}^{T'} be an alignment, B(π)=y collapse blanks & repeats
P(y | X) = ∑_{π ∈ B⁻¹(y)} ∏_{t=1..T'} P(π_t | X)

CTC_loss = −log P(y | X)
```

– T': number of input frames

– B maps alignments π to output y by removing blanks and repeated labels

### 3.3 Attention-based Decoder

```python
scores = Q @ K.T / sqrt(d)         # dot‐product attention
α = softmax(scores, axis=1)        # focus weights over time frames
context_t = α[t] @ V               # context vector for output t
```

– Q: decoder state; K,V: encoder outputs for all input frames

### 4. Code & Practical Application

### 4.1 Feature Extraction with Librosa

```python
import librosa
import numpy as np

y, sr = librosa.load('audio.wav', sr=16000)
# 25 ms window, 10 ms hop
mel_spec = librosa.feature.melspectrogram(
    y, sr=sr, n_fft=400, hop_length=160, n_mels=80
)
log_mel = np.log1p(mel_spec)  # (n_mels, T_frames)
```

### 4.2 PyTorch: CNN + BiLSTM + CTC Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechCTC(nn.Module):
    def __init__(self, n_mels, hidden_size, num_labels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(), nn.BatchNorm2d(64)
        )
        self.lstm = nn.LSTM(
            input_size=64*n_mels, hidden_size=hidden_size,
            num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_size*2, num_labels)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        b, _, n_mels, T = x.size()
        h = self.cnn(x)            # (B, 64, n_mels, T)
        h = h.permute(0,3,1,2)     # (B, T, 64, n_mels)
        h = h.reshape(b, T, -1)    # (B, T, 64*n_mels)
        o, _ = self.lstm(h)        # (B, T, 2*hidden)
        logits = self.fc(o)        # (B, T, num_labels)
        return F.log_softmax(logits, dim=2)
```

### 4.3 Training Loop with CTC Loss

```python
model = SpeechCTC(n_mels=80, hidden_size=256, num_labels=30)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in train_loader:
    feats, feat_lens, labels, label_lens = batch
    logp = model(feats)  # (B, T, V)
    # CTC expects (T, B, V)
    loss = ctc_loss(
        logp.permute(1,0,2), labels, feat_lens, label_lens
    )
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
```

### 5. Visualization / Geometry

- Plot **log-Mel spectrogram**: time on x-axis, Mel bands on y-axis, intensity=power.
- For CTC models, draw **alignment lattice**: each frame can emit blank or a label; valid paths flow through this grid.
- For attention decoders, visualize **attention map**: matrix T_out×T_in showing link strength between output tokens and input frames.

### 6. Common Pitfalls & Tips

- **Frame vs Token Rate Mismatch**: CTC assumes many frames map to few tokens—normalize lengths properly.
- **Blank Collapse Issues**: if blank dominates, reduce blank bias or add minimum label probability floor.
- **Overfitting**: audio datasets are small—use SpecAugment (time/frequency masking) and noise injection.
- **Alignment Drift**: attention models may misalign on long utterances—add guided‐attention or location penalties.
- **Gradient Vanishing** in deep LSTMs—use bidirectional layers and residual connections.

### 7. Interview-Ready Insights

- Contrast **CTC** vs **attention** decoders: CTC is alignment-free but assumes monotonic order; attention handles non-monotonic but requires decoder at inference.
- Explain **beam search with CTC**: prefix scoring, merging identical labels separated by blanks.
- Define **WER** (word error rate) and **CER** (character error rate):where S= substitutions, D= deletions, I= insertions, N= # words.
    
    ```
    WER = (S + D + I) / N
    ```
    
- Discuss **streaming ASR**: latency constraints, chunk-based attention or monotonic attention.

### 8. Practice Exercises

1. **Feature Exploration**
    - Extract log-Mel spectrograms for a set of WAV files. Plot and play them back to link audio to spectrogram patterns.
2. **CTC Model on Digits**
    - Use the Speech Commands “zero”…“nine” dataset. Train the CNN+BiLSTM+CTC model to predict digit labels. Evaluate CER.
3. **Greedy vs Beam CTC Decoding**
    - Implement greedy CTC decode (remove repeats/blanks) and beam search CTC decode with beam width B=5. Compare WER improvements.
4. **Attention ASR Prototype**
    - Replace the CTC head with an attention decoder. Visualize attention maps on a sample utterance and compare alignment quality.

---

## Trigger Word Detection

### 1. Direct Definition

Trigger word detection (keyword spotting) is the task of continuously monitoring an audio stream to spot the occurrence of a predefined “wake word” (e.g., “Alexa” or “Hey Siri”). It frames the problem as a binary classification over short time windows or as a sequence labeling task, outputting 1 when the trigger is present and 0 otherwise.

### 2. Concept Intuition

You can think of a rolling window sliding over the audio: each window is turned into a feature vector, fed through a lightweight neural network, and yields a probability that the trigger word appears within that window.

By tuning the window length, hop size, and model capacity, you balance detection accuracy, false-alarm rate, and latency.

In practice, you stream features in real time, maintain the model’s hidden state across windows, and raise a “wake” event when the probability crosses a threshold.

### 3. Mathematical Breakdown

```python
# For each time frame t, let h_t be the model’s last hidden layer output
z_t = W_h · h_t + b             # affine transform (shape: 1×1)
p_t = sigmoid(z_t)              # probability trigger present in window t

# Binary cross-entropy loss over T windows:
loss = - (1/T) * sum_{t=1..T} [y_t * log(p_t) + (1-y_t) * log(1-p_t)]
```

- h_t: feature representation at time t (from CNN/RNN)
- W_h: weight vector mapping h_t→logit
- b: scalar bias
- y_t ∈ {0,1}: ground-truth label for window t

### 4. Code & Practical Application

### 4.1 Feature Extraction (Librosa)

```python
import librosa
import numpy as np

def extract_log_mel(audio_path, sr=16000, n_mels=40, win=0.025, hop=0.010):
    y, _ = librosa.load(audio_path, sr=sr)
    n_fft   = int(win * sr)
    hop_len = int(hop * sr)
    mel_spec = librosa.feature.melspectrogram(
        y, sr=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
    )
    log_mel = np.log1p(mel_spec)  # shape: (n_mels, T_frames)
    return log_mel.T               # return shape: (T_frames, n_mels)
```

### 4.2 Keras Model for Streaming Detection

```python
import tensorflow as tf

def build_trigger_model(input_dim, rnn_units=32):
    inputs = tf.keras.Input(shape=(None, input_dim))  # streaming frames
    x = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.GRU(rnn_units, return_sequences=True)(x)
    logits = tf.keras.layers.Dense(1)(x)              # one logit per frame
    outputs = tf.keras.activations.sigmoid(logits)
    return tf.keras.Model(inputs, outputs)

# Compile
model = build_trigger_model(input_dim=40)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 5. Visualization / Geometry

- Plot the **spectrogram** with time on the x-axis and Mel bands on the y-axis.
- Overlay the model’s **probability curve** p_t over time: peaks align with the trigger.
- In a 2D latent space (e.g., t-SNE on h_t), windows containing the trigger cluster separately from background noise.

### 6. Common Pitfalls & Tips

- Class imbalance: far more “non-trigger” windows than trigger—use oversampling or weighted loss.
- Threshold selection: calibrate probability threshold on a validation set to minimize false alarms vs. miss rate.
- Window/hop trade-off: smaller hops reduce latency but increase compute.
- Streaming state: ensure RNN hidden states persist across inference calls to detect triggers that span window boundaries.

### 7. Interview-Ready Insights

- Describe sliding-window vs. frame-wise detection: window-wise aggregates context, frame-wise leverages sequence models.
- Explain why a small Conv1D+GRU model is preferred on-device for low latency and memory constraints.
- Discuss metrics: false-alarm rate (FAR), miss rate, and how to plot DET curves.
- Be ready to outline a real-time inference pipeline: feature buffer → model → threshold logic → event callback.

### 8. Practice Exercises

1. Build a **binary dataset** from Google Speech Commands: label “yes” as trigger, all other words and silence as negatives.
    - Hint: extract 1-second clips, assign “1” if keyword present anywhere.
2. Train the Conv1D+GRU model above. Plot **precision vs. recall** by varying the detection threshold.
3. Implement a **frame-smoothing** function: only fire a trigger if p_t exceeds threshold for ≥ k consecutive frames (e.g., k=3).
    - Hint: apply a moving window over the binary decisions.
4. Deploy your model in a simple **streaming simulator**: feed live audio in chunks, maintain RNN state, and log trigger times and latency.

---