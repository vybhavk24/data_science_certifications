# GenAI_m1

## Generative AI and LLMs

### 1. Direct definition

Generative AI uses models that produce new content (text, code, images) from learned patterns. Large Language Models (LLMs) are generative models trained on massive text corpora that map input token sequences to output token probabilities, enabling tasks like next-token prediction, completion, translation, summarization, and instruction following.

### 2. Concept intuition

- What it is: Think of an LLM as a very large conditional probability engine that, given a sequence of tokens, predicts the most plausible next tokens based on patterns in training data.
- Why it matters: LLMs generalize language structure, facts, and some reasoning patterns, enabling automation of writing, coding, question answering, and more.
- Analogy: Imagine a novelist who has read billions of books and learned styles, facts, and transitions; when given a prompt they continue in a statistically plausible, context-aware way.
- Key operational modes:
    - Pretraining: learn language by predicting tokens across huge corpora (unsupervised).
    - Fine-tuning: adapt the pretrained model to a narrower task or instruction-following using supervised or RL methods.
    - Inference: generate outputs using decoding algorithms (greedy, sampling, beam search).
- How components fit:
    - Tokenization converts text to discrete tokens.
    - Embeddings map tokens to vectors.
    - Transformer blocks (self-attention + feed-forward) mix context.
    - Final linear + softmax turn hidden vectors into token probabilities.

### 3. Mathematical breakdown

Core objective during autoregressive pretraining:

- For a token sequence x = [x1, x2, ..., xT] maximize log-likelihood

```
Loss = - sum_{t=1..T} log p(x_t | x_{1:t-1})
```

Model factorization (autoregressive):

```
p(x) = product_{t=1..T} p(x_t | x_{1:t-1})
```

Softmax for token probability from logits z_t (vector over vocab):

```
p(x_t = v | context) = exp(z_t[v]) / sum_{u in V} exp(z_t[u])
```

Embedding lookup and positional encoding:

```
h0[t] = Embedding[token_id_t] + PosEncoding[t]
```

Self-attention (single head) computations:

```
Q = h W_Q
K = h W_K
V = h W_V
attention_scores = Q K^T / sqrt(d_k)
attention_weights = softmax(attention_scores, axis=-1)
head_output = attention_weights V
```

Multi-head attention:

```
head_i = Attention(h W_Qi, h W_Ki, h W_Vi)
MHA_output = concat(head_1, ..., head_h) W_O
```

Transformer layer (residual + layer norm):

```
h' = LayerNorm(h + MHA_output)
h_out = LayerNorm(h' + FFN(h'))
```

Feed-forward network (position-wise):

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2    # often gelu instead of relu
```

Gradient update (SGD-like):

```
theta_{t+1} = theta_t - lr * grad_theta Loss
```

Cross-entropy loss for a batch:

```
CE = - 1/N sum_{i=1..N} sum_{t=1..T_i} log p(theta; x_{i,t} | x_{i,1:t-1})
```

Explain variables briefly:

- x_t: token at position t
- p(...): model predicted probability
- h: hidden states (batch, seq_len, d_model)
- W_Q, W_K, W_V, W_O: learned projection matrices
- d_k: dimensionality of keys/queries
- V: vocabulary
- theta: model parameters

### 4. Code and practical application

Minimal end-to-end example showing how to load a pretrained LLM, tokenize text, run forward pass to get probabilities, and generate text (Hugging Face + PyTorch). This is a toy workflow for local experimentation.

1. Install (if needed)

```bash
pip install transformers torch sentencepiece
```

1. Tokenize, forward pass, compute loss on a toy sequence

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "gpt2"  # small, good for local toy experiments
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]  # shape [1, seq_len]

# Forward pass: get logits
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
logits = outputs.logits  # shape [1, seq_len, vocab_size]
print("Loss:", loss.item())
```

1. Simple generation with sampling and temperature

```python
prompt = "In the future, AI will"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# top-k sampling with temperature
gen = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 30,
    do_sample=True,
    top_k=50,
    temperature=0.8,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(gen[0], skip_special_tokens=True))
```

1. Visualize attention maps (single example, small model)

```python
# return attentions
outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions  # tuple of length num_layers, each [batch, heads, seq, seq]
# inspect first layer, head 0
att_layer0_head0 = attentions[0][0,0].detach().numpy()  # shape [seq, seq]

```

Practical notes:

- For fine-tuning, prepare dataset of prompt–response pairs, set labels shifting the decoder inputs, use gradient accumulation for large batches, and use mixed precision (fp16) for speed.
- For instruction following, often use supervised fine-tuning on curated datasets, then optionally RLHF to align outputs to human preferences.

### 5. Visualization and geometric intuition

- Embeddings geometry: each token maps to a point in d-dimensional space. Nearby tokens share contexts; linear directions can encode features (tense, numeric, gender analogies).
- Attention as soft routing: attention weights form a matrix per head where each row shows how much a token attends to previous tokens. Visualize as heatmaps: diagonal strong attention indicates locality; off-diagonal shows long-range dependencies.
- Multi-head: different heads capture different relations (syntax, coreference, entities). Geometry: heads project into different subspaces, compute pairwise affinities, and recombine — like multiple lenses each emphasizing a different relationship.
- Layerwise transformations: early layers refine local patterns, later layers form higher-level abstractions and task-specific features. Visualize t-SNE/UMAP of hidden states across layers to see clustering by semantic role.
- Gradients: backprop flows from loss to parameters; steep gradients indicate high influence. Visualize gradient norms per layer to diagnose vanishing or exploding updates.

Quick plotting approach:

- Plot attention head heatmap using matplotlib imshow of attention matrix.
- Project token embeddings using PCA/UMAP to 2D to inspect clusters.

### 6. Common pitfalls and tips

- Confusing pretraining vs fine-tuning: pretraining builds general language knowledge; fine-tuning adapts to tasks and can quickly overfit if dataset small.
- Tokenization issues: different models use different tokenizers; misaligned tokenization changes input length and meaning.
- Over-reliance on perplexity: low perplexity doesn't guarantee factual or safe outputs.
- Decoding misuse: greedy decoding produces bland output; uncalibrated sampling can hallucinate. Tune temperature, top-k, or top-p.
- Catastrophic forgetting: fine-tuning without preservation strategies can erase general capabilities; use lower learning rates, regularization, or adapters.
- Training stability: large LR or wrong normalization can destabilize training; use layer norm, appropriate optimizers (AdamW), warmup schedules.
- Prompt sensitivity: small prompt wording changes can produce large output differences; treat prompt engineering as model calibration.
- Evaluation pitfalls: automatic metrics like BLEU/ROUGE often correlate poorly with human judgment for generation tasks.

### 7. Interview-ready insights

- Why Transformers beat RNNs for language: Transformers compute attention with parallelizable matrix ops enabling long-range dependency learning without recurrence and with better gradient flow.
- Autoregressive vs Masked models: Autoregressive (GPT) predict next token, good for generation; Masked (BERT) predict masked tokens using bidirectional context, good for representations but not straightforward generation without adaptation.
- Decoder-only, Encoder-only, Encoder-Decoder:
    - Decoder-only: optimized for generative tasks (GPT family).
    - Encoder-only: great for classification and retrieval (BERT).
    - Encoder-Decoder: sequence-to-sequence tasks (T5, BART) — good for translation, summarization.
- Scaling laws: Increasing model size, data, and compute tends to improve performance but with diminishing returns; trade-offs exist.
- Decoding choices and trade-offs:
    - Greedy: fast, deterministic, low diversity.
    - Beam search: better sequence-level probability but can be repetitive.
    - Sampling (top-k, top-p): adds diversity, needs temperature tuning.
- Regularization and optimization choices: AdamW with weight decay, learning-rate warmup, cosine decay, gradient clipping, mixed precision are standard.
- Evaluation strategy: combine automatic metrics with human evaluation for fluency, factuality, and alignment.
- Safety and alignment: RLHF and safety filters are industry-standard to guide behavior.

### 8. Practice exercises

Exercise 1 Quick refresher

- Task: Tokenize 5 sentences using GPT-2 tokenizer, print token ids and decoded output.
- Hint: Use tokenizer.encode and tokenizer.decode.

Exercise 2 Forward pass and next-token probability

- Task: For prompt "Climate change causes", compute the top-5 most probable next tokens and their probabilities.
- Hint: Run model(input_ids) to get logits, apply softmax on last position.

Exercise 3 Attention heatmap

- Task: For a short sentence of 8 tokens, extract attention from layer 0 head 0 and plot heatmap.
- Hint: Use matplotlib.imshow on outputs.attentions[0][0,0].detach().numpy().

Exercise 4 Small fine-tune (toy)

- Task: Fine-tune GPT-2 small to complete simple QA pairs like "Q: Capital of France? A: Paris" with 50 synthetic pairs.
- Hint: Create dataset of concatenated "Q: ... A:" strings, use Trainer API or manual loop with shifting labels. Use small epochs and low lr.

Exercise 5 Decoding comparison

- Task: Generate 5 continuations of the same prompt with greedy, beam (size 5), top-k (50), and top-p (0.9) sampling. Compare diversity and quality.
- Hint: Use model.generate arguments do_sample, num_beams, top_k, top_p, temperature.

Exercise 6 Visual geometry mini-project

- Task: Extract token embeddings for a small vocab subset, run PCA to 2D, and plot tokens to inspect semantic clustering (e.g., numbers, colors, animals).
- Hint: tokenizer.get_vocab() gives mapping; use model.transformer.wte.weight for GPT-2 embeddings.

---

## LLM use cases and tasks

### 1. Direct definition

Large Language Model (LLM) use cases are the practical applications where an LLM’s ability to model token sequences and conditional probabilities is applied to perform tasks such as generation, classification, transformation, retrieval, and interaction. Tasks are concrete formulations (inputs, outputs, constraints, evaluation) that map an LLM to a product or research objective.

### 2. Concept intuition

- What it is: Take the LLM’s core skill — mapping context to plausible continuations — and frame it as many problem templates: produce text, fill blanks, answer questions, transform form, rank items, extract structured facts, or drive actions.
- Why it matters: A single LLM architecture can power chatbots, content creation, search augmentation, code generation, summarization pipelines, assistants, retrieval-augmented generation (RAG), and more. Understanding tasks helps choose model type (decoder-only vs seq2seq vs encoder-only), pretraining/fine-tuning strategy, and decoding settings.
- Analogy: The LLM is a Swiss Army knife; tasks are the attachments you clip on. Same motor (probability model), different tool head (prompt, dataset, decoding, evaluation).
- High-level task families:
    - Generation (free, conditional)
    - Understanding / classification
    - Transformation / structured output
    - Retrieval and grounding (RAG)
    - Planning and control (agents)
    - Evaluation and scoring (metrics, safety)
    - Multimodal fusion (vision+language)
    - Code synthesis and reasoning

### 3. Mathematical breakdown (core templates)

Next-token generation objective (autoregressive):

```
Loss_gen = - sum_{t=1..T} log p(x_t | x_{1:t-1})
```

Conditional generation (given prompt c produce y):

```
p(y | c) = product_{t=1..|y|} p(y_t | c, y_{1:t-1})
Loss_cond = - sum_t log p(y_t | c, y_{<t})
```

Sequence-to-sequence cross-entropy (encoder-decoder):

```
Loss_seq2seq = - 1/N sum_{i=1..N} sum_{t=1..T_i} log p(y_{i,t} | x_i, y_{i,<t})
```

Classification with softmax over labels:

```
z = f_theta(x)              # model pooled logits for labels
p(y=k|x) = exp(z_k) / sum_j exp(z_j)
Loss_cls = - sum log p(y_true|x)
```

Contrastive retrieval scoring (embedding-based):

```
score(q, d) = q · d / (||q|| ||d||)
Loss_NCE = - log ( exp(score(q, d_pos)/tau) / sum_{d in batch} exp(score(q, d)/tau) )
```

RAG generation (retrieve R docs r1..rk, condition on them):

```
p(y|q) ≈ sum_{i=1..k} p(r_i | q) p(y | q, r_i)   # approximate marginalization
```

RLHF-style objective (policy gradient, simplified):

```
max_theta E_{y ~ pi_theta(.|x)} [ R(y) ]   # train policy to increase reward R(y)
```

Variables:

- x, c, q: inputs / context / query
- y: generated output sequence
- r_i: retrieved document
- tau: temperature in contrastive loss
- theta: model parameters
- p(r_i|q): retrieval probability or weighting

### 4. Code & practical application (mini-recipes)

Prereq install:

```bash
pip install transformers accelerate faiss-cpu datasets
```

A. Text generation (open-ended) — decoder-only

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Write a short product description for a smart water bottle:"
input_ids = tok(prompt, return_tensors="pt").input_ids
gen = model.generate(input_ids, max_length=120, do_sample=True, top_p=0.9, temperature=0.8)
print(tok.decode(gen[0], skip_special_tokens=True))
```

B. Summarization — seq2seq (T5/BART)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

article = "Long article text ..."
input_ids = tok("summarize: " + article, return_tensors="pt").input_ids
summary_ids = model.generate(input_ids, max_length=80, num_beams=4)
print(tok.decode(summary_ids[0], skip_special_tokens=True))
```

C. Classification via prompt / zero-shot (PET / prompt)

```python
# simple prompt classification with GPT-style scoring
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Review: The movie was fantastic and moving.\nSentiment:"
labels = [" positive", " negative"]
scores = []
for lab in labels:
    ids = tok(prompt + lab, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits
    # score = sum log probs for label tokens; simpler: look at probability of first token after prompt
    prob = torch.softmax(logits[0, -1], dim=-1)[ tok.encode(lab, add_special_tokens=False)[0] ]
    scores.append(prob.item())
print("Scores", dict(zip(labels, scores)))
```

D. Retrieval-Augmented Generation (RAG) sketch

- Build an embedding index for documents (FAISS), embed query, retrieve top-k, then condition LLM on prompt + retrieved passages.

```
pipeline:
1. docs -> encode -> index (FAISS)
2. q -> encode -> top-k ids
3. prompt = "Context: <doc1> <doc2> ... Query: q"
4. model.generate(prompt)
```

E. Code generation evaluation loop (safety & tests)

- Generate function
- Run unit tests in sandbox
- Score by pass/fail and complexity

### 5. Visualization / Geometry

- Generation probabilities: visualize token probability distribution for the next token as a bar chart; high-peaked → low uncertainty, flat distribution → high uncertainty.
- Classification decision boundary: for encoder pooled outputs, plot 2D PCA/UMAP of pooled vectors; clusters correspond to classes.
- RAG flow: diagram showing query → embedding space → nearest docs → augmented prompt → generation. Embedding geometry shows semantic neighborhoods.
- Attention & conditioning: show how adding retrieved context shifts attention patterns — attention heatmaps with rows pointing to retrieved context tokens.
- Token-level saliency: compute gradient of log-probability of chosen token wrt input embeddings to highlight influential tokens (saliency map).

Quick code to plot next-token distribution:

```python
import matplotlib.pyplot as plt
import torch, numpy as np

logits = outputs.logits[0, -1].detach()
probs = torch.softmax(logits, dim=-1).cpu().numpy()
topk = 20
ids = probs.argsort()[-topk:][::-1]
plt.bar(range(topk), probs[ids])
plt.xticks(range(topk), [tok.decode([i]) for i in ids], rotation=90)
plt.show()
```

### 6. Common pitfalls & tips

- Confusing task framing: many tasks can be solved either by prompting or by fine-tuning; choose based on data availability, latency, and update frequency.
- Hallucination in grounded tasks: always ground generative answers with retrieval or tool outputs when factuality matters.
- Latency vs quality: bigger models and beam search increase latency. Use model distillation, caching, or reranking to trade off.
- Evaluation mismatch: automatic metrics (ROUGE/BLEU/Perplexity) often misalign with user utility; include human evals or task-specific measures (unit tests for code, factuality checks for QA).
- Tokenization artifacts: label tokens for classification must align with tokenizer splits; use label-mapping when label tokens split into multiple subwords.
- Retrieval contamination: when fine-tuning or evaluating, ensure docs used in training are not in the retrieval corpus used at inference.
- Safety/guardrails: adopt prompt filtering, moderation endpoints, RLHF, rule-based fallbacks.
- Cost management: for production, prefer smaller specialized models + RAG over large LLM for many queries.

### 7. Interview-ready insights

- When to use RAG: if task requires up-to-date or niche factual knowledge and you want to avoid full model retraining. Explain the pipeline: index -> embed -> retrieve -> generate; mention approximate marginalization and reranking.
- Why prompt engineering matters: it changes the conditional distribution p(y|c). Small prompt changes can shift modes dramatically. For robust systems, combine prompt templates with few-shot exemplars and calibration.
- Fine-tune vs adaptation:
    - Full fine-tuning: better performance, higher maintenance cost, risk of forgetting.
    - Adapters / LoRA: parameter-efficient, reversible, lower compute and storage.
    - Prompt tuning: very low memory, sometimes lower performance.
- Evaluation design: propose both intrinsic (perplexity, accuracy) and extrinsic (task success, human ratings, downstream metrics) measures; include safety checks and A/B testing.
- Decoding trade-offs: explain how temperature, top-p, and beam width change exploration vs exploitation and show simple equations for sampling:

```
p_i = softmax(logits / temperature)
top-p: normalize probabilities after filtering cumulative prob threshold p
```

### 8. Practice (exercises + hints)

Exercise A — Generation vs Completion

- Task: Use GPT-2 to generate (a) a short marketing blurb given a product name and (b) complete an unfinished sentence. Compare top-5 next-token distributions for both prompts.
- Hint: Use model.generate and inspect logits at last step.

Exercise B — Summarization fine-tune (toy)

- Task: Fine-tune t5-small on 100 short article->summary pairs (synthetic). Evaluate ROUGE and qualitatively inspect errors.
- Hint: Use Hugging Face Trainer; set num_epochs=3, batch_size=8, lr=5e-5.

Exercise C — Build a tiny RAG system

- Task: Index 200 small FAQ passages with FAISS using SentenceTransformers embeddings, implement retrieval-augmented prompt and generate answers for 20 queries. Measure answer overlap with gold answers by exact-match or token F1.
- Hint: Use sentence-transformers/all-MiniLM-L6-v2 for embeddings and store normalized vectors.

Exercise D — Prompt robustness

- Task: Create 10 paraphrased prompts for the same question and generate model responses. Quantify response similarity using embedding cosine and comment on variance.
- Hint: Use SBERT to embed outputs and compute mean pairwise cosine.

Exercise E — Classification via LM scoring

- Task: Implement zero-shot sentiment classification by computing the conditional log-probability of label tokens appended after prompt and picking max. Compare with a small fine-tuned classifier.
- Hint: For multi-token labels, sum log-probs of token sequence.

Exercise F — Safety gating

- Task: Add a simple rule-based filter to redact outputs containing a list of banned phrases; log filtered outputs and compare occurrence rates with and without filter.
- Hint: Use deterministic regex checks; consider partial matches across tokens.

---

## Text generators before Transformers

### Direct definition

Text generators before Transformers are neural architectures that produced text sequences using recurrence or convolution to model temporal dependencies. Key families include n-gram and probabilistic models, Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) variants, sequence-to-sequence encoder‑decoder models, and early attention-augmented seq2seq. These models learned p(x) or p(y|x) by processing tokens sequentially and predicting next-token probabilities.

### Concept intuition

- What it is: A text generator maps prior tokens to probabilities over next tokens. Early models used explicit counts (n-grams). Neural approaches replaced counts with learned vector states that carry context forward step-by-step.
- Why it mattered: They demonstrated that neural networks could capture syntax, short- and medium-range semantics, and could be trained end-to-end for tasks like translation and summarization before Transformers made long-range dependencies easier to handle.
- Analogy: Think of RNNs as a person reading a sentence with a sticky note (the hidden state) that they update word-by-word; LSTMs give the person special compartments (gates) to selectively write, erase, and read from the note so important facts survive longer.
- Key evolutionary points:
    - n-gram models: simple statistical baselines with limited context and data sparsity.
    - RNN: learn distributed state, but struggle with long-term dependencies.
    - LSTM / GRU: gating mechanisms to mitigate vanishing gradients and keep information over longer spans.
    - Seq2Seq encoder-decoder: two RNNs, one encodes the input, the other decodes — introduced for translation and conditional generation.
    - Attention on top of seq2seq: allowed decoder to peek at encoder states dynamically, dramatically improving long-range alignment and translation quality.

### Mathematical breakdown

n‑gram maximum likelihood (baseline)

```
p(x) ≈ product_t p(x_t | x_{t-n+1:t-1})
MLE estimate: p(w_i | history) = count(history, w_i) / count(history)
```

RNN single-step update

```
h_t = f(W_h x_t + U_h h_{t-1} + b_h)
o_t = W_o h_t + b_o
p(x_t | x_{<t}) = softmax(o_t)
```

- h_t: hidden state, x_t: input vector (one-hot or embedding), f: nonlinearity (tanh).

Vanishing gradient motivation

```
dL/dh_t = dL/dh_{t+1} * diag(f'(a_{t+1})) * U_h^T
Repeated multiplication by U_h and f' can shrink or explode gradient exponentially with time steps.
```

LSTM cell (gates and state)

```
i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)     # input gate
f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)     # forget gate
o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)     # output gate
g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)        # candidate
c_t = f_t * c_{t-1} + i_t * g_t                # cell state
h_t = o_t * tanh(c_t)                          # hidden state
```

GRU (simpler gating)

```
z_t = sigmoid(W_z x_t + U_z h_{t-1})
r_t = sigmoid(W_r x_t + U_r h_{t-1})
h_tilde = tanh(W_h x_t + U_h (r_t * h_{t-1}))
h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
```

Seq2Seq encoder-decoder (basic)

```
# encoder processes source x_1..x_S -> final state h_S
h_enc_t = RNN_enc(x_t, h_enc_{t-1})
# decoder initialized with encoder final state, generates y_1..y_T
h_dec_t = RNN_dec(y_{t-1}, h_dec_{t-1})
p(y_t | y_{<t}, x) = softmax(W_o h_dec_t)
Loss = - sum_t log p(y_t | y_{<t}, x)
```

Additive attention (Bahdanau)

```
score_{t,s} = v^T tanh(W_1 h_dec_{t-1} + W_2 h_enc_s)
alpha_{t,s} = softmax_s(score_{t,s})
context_t = sum_s alpha_{t,s} * h_enc_s
h_dec_t = RNN_dec(y_{t-1}, h_dec_{t-1}, context_t)
```

Multiplicative attention (Luong)

```
score_{t,s} = h_dec_{t-1}^T W h_enc_s
alpha_{t} = softmax(score_t)
context_t = sum alpha_{t,s} * h_enc_s
```

Explain variables briefly:

- x_t, y_t: input and output tokens
- h: hidden states; c: LSTM cell state
- W, U, v: learnable weight matrices/vectors
- alpha: attention weights over encoder time steps

### Code and practical application

Two runnable PyTorch mini-examples: 1) LSTM next-token model; 2) Seq2Seq with attention (toy).

Tiny LSTM language model (toy dataset)

```python
# pip install torch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence

# toy vocab and dataset
vocab = ["<pad>","a","b","c","d","e"," "]
stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for s,i in stoi.items()}

def encode(s): return torch.tensor([stoi[ch] for ch in s], dtype=torch.long)

data = [encode("ab "), encode("abc"), encode("abd"), encode("abe")]
batch = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)  # shape [B, L]

class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.out = nn.Linear(d, vocab_size)
    def forward(self, x, hidden=None):
        e = self.embed(x)
        o, hidden = self.lstm(e, hidden)
        logits = self.out(o)
        return logits, hidden

model = TinyLSTM(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# train few steps
for epoch in range(200):
    opt.zero_grad()
    logits, _ = model(batch)
    # shift targets
    targets = batch
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    opt.step()
print("trained loss", loss.item())

# sample greedy from prompt "a"
prompt = torch.tensor([[stoi["a"]]], dtype=torch.long)
logits, hid = model(prompt)
probs = torch.softmax(logits[0, -1], dim=-1)
print("next-token probs:", {itos[i]: float(probs[i]) for i in range(len(vocab))})
```

Simple encoder-decoder with dot-product attention (inference skeleton)

```python
# simplified; production code requires batching, masks
class Encoder(nn.Module):
    def __init__(self, vocab_size, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.rnn = nn.GRU(d, d, batch_first=True, bidirectional=False)
    def forward(self, x):
        e = self.embed(x)
        out, h = self.rnn(e)
        return out, h  # out: [B, S, d], h: [1, B, d]

class Decoder(nn.Module):
    def __init__(self, vocab_size, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.rnn = nn.GRU(d + d, d, batch_first=True)  # input + context
        self.out = nn.Linear(d, vocab_size)
    def forward(self, y_prev, h_prev, enc_out):
        e = self.embed(y_prev)  # [B,1,d]
        # compute attention scores (dot)
        scores = torch.bmm(enc_out, h_prev.transpose(0,1).transpose(1,2))  # [B, S, 1]
        alpha = torch.softmax(scores, dim=1)  # [B, S, 1]
        context = torch.sum(alpha * enc_out, dim=1, keepdim=True)  # [B,1,d]
        rnn_in = torch.cat([e, context], dim=-1)
        out, h = self.rnn(rnn_in, h_prev)
        logits = self.out(out)
        return logits, h, alpha

# instantiate and run one step (toy shapes)
enc = Encoder(len(vocab))
dec = Decoder(len(vocab))
src = batch  # reuse tiny batch
enc_out, enc_h = enc(src)
y_prev = torch.tensor([[stoi[" "]]])  # start token
logits, dec_h, alpha = dec(y_prev, enc_h, enc_out)
print("logits shape", logits.shape, "alpha shape", alpha.shape)
```

Practical notes:

- Use teacher forcing during training for faster convergence.
- Implement masking for variable lengths.
- For real tasks use packed sequences, batching, scheduler, and gradient clipping.

### Visualization and geometric intuition

- Hidden state trajectory: visualize h_t over time projected with PCA/UMAP. For a repeated structure (e.g., numbers, dates), hidden states form trajectories; LSTM gates cause slower drift for remembered info.
- Attention alignment heatmap: for seq2seq with attention, plot alpha_{t,s} heatmap (decoder steps vs encoder tokens). Ideal translation shows near-diagonal or clear source-target alignments.
- Gate activations: plot i_t, f_t, o_t values over time to see when the model writes or forgets information.
- Vanishing gradient geometry: repeated multiplication by Jacobian shrinks vectors toward dominant eigenvector; visualize singular values of recurrent matrix U_h — small singular values indicate contraction.
- Sampling: examine next-token probability distribution—narrow (peaky) distributions lead to deterministic text; flatter distributions increase diversity and risk of mistakes.

Quick code snippet to plot attention heatmap (matplotlib)

```python
import matplotlib.pyplot as plt
alpha_np = alpha.detach().squeeze(-1).cpu().numpy()  # [B, S]
plt.imshow(alpha_np, cmap='viridis', aspect='auto')
plt.xlabel("encoder tokens")
plt.ylabel("decoder step (single step in this toy example)")
plt.colorbar()
plt.show()
```

### Common pitfalls and tips

- RNNs struggle with very long-range dependencies due to gradient decay; LSTMs help but are not magic.
- Teacher forcing mismatch: training with teacher forcing but sampling autoregressively at inference time can cause exposure bias. Use scheduled sampling or sequence-level objectives to mitigate.
- Overfitting on small data: recurrent models can memorize; use dropout between layers, weight decay, and early stopping.
- Training instability: exploding gradients common—use gradient clipping.
- Inattention to batching and masking: incorrect masking yields wrong alignments and loss computation.
- Inefficient parallelism: RNNs are sequential; they are slower to train compared to Transformer’s parallelizable attention.
- Attention in pre-transformer seq2seq adds computational cost but improves alignment; its implementation details (score function, normalization) affect performance.

### Interview-ready insights

- Why LSTMs were necessary: they solved vanishing gradients by adding a cell state with linear self-loop gated by forget gate, preserving information across many steps.
- Seq2Seq with attention vs plain encoder-decoder: attention relieves encoder of compressing whole source into single fixed vector; it provides soft-access to all encoder states and produces interpretable alignments.
- Exposure bias and solutions: explain teacher forcing and scheduled sampling; discuss reinforcement learning or minimum risk training for sequence-level objectives.
- Complexity trade-offs: RNNs are O(T) sequential operations per layer; attention in Transformers is O(T^2) in memory and compute per layer but parallelizable across positions.
- When RNNs still make sense: low-latency streaming inference, small-footprint models, or when computation constraints favor sequential small models.
- Gate math intuition: forget gate controls retain/erase; input gate controls writing; output gate controls exposure. Show how a persistent memory pattern can be implemented by setting f≈1, i≈0 for continuity.

### Practice exercises

Exercise 1 n-gram baseline

- Task: Implement a trigram model from a small corpus, compute next-token probabilities for a test sentence, and compare perplexities with an LSTM trained on the same corpus.
- Hint: Use smoothing like add-1 or Kneser-Ney for better generalization.

Exercise 2 Train LSTM next-token predictor

- Task: Train the TinyLSTM above on a small set of sentences (e.g., Project Gutenberg chapter excerpt) for next-token prediction. Log training loss and sample generated text after every epoch.
- Hint: Use torch.nn.utils.rnn.pack_padded_sequence for variable lengths. Try both teacher forcing and sampling evaluation.

Exercise 3 Seq2Seq translation toy

- Task: Create a synthetic dataset of simple paired sequences (e.g., mapping spelled numbers to words: "1 2 3" -> "one two three") and train an encoder-decoder with attention. Visualize attention alignments.
- Hint: Use cross-entropy loss with teacher forcing and plot alpha matrices for several samples.

Exercise 4 Diagnose vanishing/exploding gradients

- Task: Construct an RNN with a simple linear recurrent matrix; measure gradient norms across time steps for different spectral radii of recurrence matrix. Observe how gradient norms decay/grow.
- Hint: Initialize U_h with varying scale and compute dL/dh_t norms numerically.

Exercise 5 Scheduled sampling experiment

- Task: Train a seq2seq model with scheduled sampling (gradually replace teacher inputs with model samples during training) and compare BLEU/perplexity with plain teacher forcing.
- Hint: Implement a probability schedule that decays from 1.0 to 0.0 over epochs.

Exercise 6 Gate probing

- Task: For a trained LSTM on a task requiring remembering an element (e.g., copy task), log gate values (f_t, i_t, o_t) and visualize when information is stored or forgotten.
- Hint: Use hooks or modify forward to return gates; plot over time for multiple examples.

---

## Transformer architecture

### Direct definition

A Transformer is a neural network architecture that processes token sequences by computing pairwise attention between positions and mixing those results through position-wise feed-forward networks, using residual connections and layer normalization to enable deep, parallelizable learning of sequence representations for generation and understanding tasks.

### Concept intuition

- What it is: The Transformer replaces recurrence with attention: every token can directly attend to every other token using learned queries, keys, and values, allowing the model to capture long-range dependencies in a single layer pass.
- Why it matters: Transformers are highly parallelizable, scale effectively with data and compute, and form the basis of modern LLMs (GPT, BERT, T5), enabling superior performance on language tasks, strong transfer learning, and flexible encoder-decoder setups for seq2seq tasks.
- Visual analogy: Imagine each token as a person in a room who holds a question (query), listens for answers (keys from others), and collects knowledge (values). Attention is the polite conversation where each person weighs every other person's answers, then everyone updates their internal notes simultaneously.
- Core building blocks: token embeddings + positional encoding, multi-head self-attention, residual + layer norm, position-wise feed-forward network, stacking into layers, final projection to tasks.
- Typical variants:
    - Encoder-only (BERT): bidirectional context for classification/representations.
    - Decoder-only (GPT): autoregressive generation.
    - Encoder-decoder (T5, BART): conditional generation and seq2seq.

### Mathematical breakdown

Embedding and input:

```
h0[t] = Embedding(token_t) + PosEncoding(t)
```

Scaled dot-product attention (single head):

```
Q = H W_Q
K = H W_K
V = H W_V
scores = Q K^T / sqrt(d_k)          # shape [seq, seq]
A = softmax(scores, axis=-1)        # attention weights
head = A V                          # shape [seq, d_k]
```

Multi-head attention:

```
head_i = Attention(H W_Qi, H W_Ki, H W_Vi)   for i in 1..h
MHA_output = concat(head_1, ..., head_h) W_O
```

Transformer layer forward (pre-norm variant shown; post-norm similar with different stability properties):

```
# Multi-head attention block
x1 = LayerNorm(x + MHA(x))
# Feed-forward block
x2 = LayerNorm(x1 + FFN(x1))
```

Position-wise feed-forward network:

```
FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
```

Final token logits for language modeling:

```
logits = x_Layer * W_vocab^T + b_vocab
p(token) = softmax(logits)
Loss = - sum_t log p(x_t | x_{<t})
```

Why scale by sqrt(d_k):

```
score = (q·k) / sqrt(d_k)
```

Scaling keeps dot-product variance stable as d_k grows, preventing softmax from becoming too peaked or too flat.

Explain variables:

- H: input matrix of hidden states (seq_len × d_model).
- W_Q, W_K, W_V: projection matrices (d_model × d_k).
- d_k: dimensionality per head (d_model / num_heads).
- W_O: output projection (num_heads*d_k × d_model).
- GELU: Gaussian Error Linear Unit activation used in modern Transformers.
- LayerNorm: normalization applied over feature dimension.
- A: attention matrix with rows summing to 1 representing where each token attends.

Backprop essentials (gradient flow through residuals):

```
dL/dx flows through both the residual path and through MHA/FFN, improving gradient propagation compared to deep stacked RNNs.
```

Complexity:

```
Self-attention compute: O(seq_len^2 * d_model) per layer
FFN compute: O(seq_len * d_model^2) per layer
```

### Code & practical application

Minimal self-attention and single-layer Transformer block in PyTorch (copy-paste ready).

```python
# pip install torch einops
import torch
import torch.nn as nn
from einops import rearrange

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.scale = self.d_head ** 0.5

    def forward(self, x, mask=None):
        # x: [B, T, d_model]
        B, T, _ = x.shape
        Q = rearrange(self.wq(x), "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(self.wk(x), "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(self.wv(x), "b t (h d) -> b h t d", h=self.num_heads)
        scores = torch.einsum("b h i d, b h j d -> b h i j", Q, K) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, V)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.wo(out), attn  # returns logits and attention weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = SimpleSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights
```

Tiny Transformer encoder stack and forward pass:

```python
class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=3, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos[:T]
        attns = []
        for layer in self.layers:
            x, a = layer(x, mask=mask)
            attns.append(a)
        logits = self.out(x)
        return logits, attns
```

Example usage with toy data and next-token training loop:

```python
# toy example
vocab_size = 100
model = TinyTransformerEncoder(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# random toy batch
input_ids = torch.randint(0, vocab_size, (8, 16))
targets = input_ids.clone()

logits, attns = model(input_ids)
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
loss.backward()
optimizer.step()
print("loss", loss.item())
```

Extracting and visualizing attention:

```python
# attns: list of [B, heads, T, T] per layer
layer0_head2 = attns[0][0, 2].detach().cpu().numpy()  # for batch 0, head 2
# use matplotlib to heatmap this matrix
```

Practical deployment notes:

- Use rotary embeddings or learned absolute/relative positional encodings for longer context handling.
- Use efficient attention variants (sparse, sliding window, kernel-based) for very long sequences.
- For autoregressive decoding, cache past key/value projections per layer to avoid recomputing keys/values for earlier tokens during stepwise generation.

### Visualization / Geometry

- Attention matrix geometry: attention weight A is a stochastic matrix per token; each row is the receptive field for a token showing which positions influence its new representation. Visualize as heatmaps where bright columns indicate strong influence.
- Query-key similarity: attention score at (i, j) is proportional to cosine-like similarity between query at i and key at j. Visualize queries and keys in 2D PCA space to see clusters that produce strong cross-attention.
- Multi-head subspaces: each head projects into a distinct subspace; plot head outputs with PCA/UMAP to observe specialized roles (syntax, named-entity linking, long-range dependencies).
- Layerwise representation evolution: take hidden states for the same token across layers and project to 2D to inspect how its representation drifts from lexical to semantic/task-specific features.
- Residual + skip geometry: residual connections keep representations near input manifold early in training; FFN and attention perform controlled perturbations that progressively move vectors to new regions relevant to the task.
- Gradient flow visualization: plot gradient norm per layer during training to check for vanishing/exploding behavior; pre-norm variants typically show more stable deep gradient norms.

Suggested plotting code for attention heatmap:

```python
import matplotlib.pyplot as plt
att = attns[layer_idx][batch_idx, head_idx].detach().cpu().numpy()
plt.imshow(att, cmap="viridis", aspect="auto")
plt.colorbar()
plt.xlabel("key positions")
plt.ylabel("query positions")
plt.title(f"Layer {layer_idx} Head {head_idx}")
plt.show()
```

Suggested PCA projection of embeddings:

```python
from sklearn.decomposition import PCA
hs = hidden_states[layer_idx].detach().cpu().numpy().reshape(-1, d_model)  # (B*T, d)
p = PCA(n_components=2).fit_transform(hs)
plt.scatter(p[:,0], p[:,1], s=2)
plt.show()
```

### Common pitfalls & tips

- Masking errors: forgetting causal masks for decoder/autoregressive tasks causes leakage of future tokens and training collapse.
- Positional encoding misuse: absolute positional embeddings limit generalization to longer sequences than trained; relative/rotary encodings handle extrapolation better.
- Softmax saturation without scaling: omitting division by sqrt(d_k) causes unstable attention distributions when d_k large.
- LayerNorm placement: post-norm and pre-norm variants change stability and training dynamics; pre-norm is more stable for very deep models.
- Initialization and depth: poorly initialized projection matrices or deep stacks without pre-norm can lead to training instability.
- Memory & compute: full self-attention is O(T^2); long contexts require efficient attention approximations or chunking strategies.
- Caching for inference: failing to cache key/value tensors for autoregressive decoding results in quadratic compute at inference.
- Head redundancy: many heads can be redundant; pruning or head analysis can reduce compute while preserving performance.
- Overfitting FFN: FFN modules are parameter-heavy; use dropout and weight decay to combat overfitting.
- Training large Transformers: use AdamW, learning-rate warmup (linear or cosine), gradient clipping, and mixed precision (fp16/AMP).

### Interview-ready insights

- Why attention replaces recurrence: attention directly models pairwise dependencies with O(1) per-layer path length between any two tokens and enables full parallelization across positions.
- Why multi-head attention: multiple heads let the model attend to different types of relationships in parallel; concat+W_O lets the model recombine those subspace signals.
- Residuals + LayerNorm role: residuals help gradient flow across many layers; LayerNorm stabilizes feature distributions and speeds training convergence.
- Scaling laws and compute trade-offs: model performance scales with parameters, data, and compute; scaling increases context and reasoning abilities but costs memory and latency.
- Causal versus bidirectional masks: causal masks enforce autoregressive factorization for generation; bidirectional contexts (no causal mask) enable masked-language modeling and richer contextual representations.
- Positional encodings choices: sinusoidal was original; learned absolute embeddings are simple; relative/rotary encodings improve long-context generalization and translation alignment.
- Efficient attention methods: explain sparse attention, local window + global tokens, Performer (random feature attention), Linformer (low-rank), and Longformer/BigBird patterns as ways to reduce O(T^2) costs.
- Practical recipe for fine-tuning: freeze some early layers, use lower LR for pretrained weights, or use parameter-efficient methods (LoRA, adapters) to reduce compute and storage.
- Autoregressive caching: during generation, cache K/V per layer and reuse them to achieve O(T * d_model^2) instead of O(T^2) per generated token.

### Practice exercises

Exercise 1: Implement attention from scratch

- Task: Given random input H ∈ R^{T×d}, implement scaled-dot product attention and verify that each row of attention weights sums to 1.
- Hint: Use stable softmax and verify numerical stability by printing min/max of scores.

Exercise 2: Visualize head specialization

- Task: Train the TinyTransformerEncoder on a toy task (e.g., copy task or simple grammar) and plot attention heatmaps for several heads and layers to identify specialization.
- Hint: Use different seeds, and plot multiple samples to see consistent patterns.

Exercise 3: Causal masking bug hunt

- Task: Create a simple autoregressive training loop without a causal mask and observe collapse (model can cheat by looking into future). Then add mask and show proper learning.
- Hint: Build masks with torch.tril and pass to attention masking.

Exercise 4: Pre-norm vs post-norm comparison

- Task: Replace LayerNorm placement in TransformerBlock (pre-norm vs post-norm), train both shallow and deep stacks, and compare training loss stability and gradient norms.
- Hint: Track gradient norm of parameters in early vs late layers.

Exercise 5: Implement caching for autoregressive decoding

- Task: Modify SimpleSelfAttention forward to accept cached K/V tensors and demonstrate stepwise generation where only the new query is computed and attn uses cached keys/values.
- Hint: Store cumulative K/V tensors per layer; use concatenation along time axis for cache.

Exercise 6: Efficient attention prototype

- Task: Replace full attention with a local-window attention (each token attends to ±w neighbors) and measure throughput and memory for increasing sequence lengths compared to full attention.
- Hint: Implement attention using unfolding or masked banded scores for speed.

---

## Generating text with Transformers

### 1. Direct definition

Text generation with Transformers is the process of producing a target token sequence y = [y1..yT] given a context c by autoregressively modeling p(y | c) using a Transformer decoder (or decoder-only) that iteratively predicts next-token probabilities and samples or selects tokens via a decoding strategy.

### 2. Concept intuition

- What it is: At each step the model turns the current context (prompt + tokens generated so far) into a vector representation, computes a probability distribution over the vocabulary, then emits the next token and repeats.
- Why it matters: Transformers let each generated token attend to all prior tokens in parallel when computing the next-token distribution and efficiently scale to large models and long contexts.
- Simple mental image: Imagine writing a sentence while constantly re-reading everything written so far and weighing which next word best continues the idea; attention is that re-reading and weighing process.
- Two operational modes:
    - Training (teacher-forced): model sees ground-truth prefixes and learns to predict next tokens via cross-entropy.
    - Inference (autoregressive): model uses its own sampled/selected tokens as prefixes; decoding strategy controls quality/diversity/hallucination.

### 3. Mathematical breakdown

Autoregressive factorization:

```
p(y | c) = product_{t=1..T} p(y_t | c, y_{1:t-1})
```

Next-token softmax over logits z_t:

```
p(y_t = v | context) = softmax(z_t)[v] = exp(z_t[v]) / sum_u exp(z_t[u])
```

Logit computation (decoder-only Transformer, simplified):

```
h0 = Emb(token_ids) + PosEnc
for layer in 1..L:
    h = TransformerLayer(h)           # multi-head attention + FFN + residuals
z_t = Linear_vocab(h[t])             # logits for token position t
```

Cross-entropy loss (training):

```
Loss = - sum_{t=1..T} log p(y_t | c, y_{<t})
```

Decoding transforms logits to tokens; core operations for sampling with temperature T:

```
p_Token = softmax(z / temperature)
```

Top-k filtering:

```
keep top k logits; set others to -inf; renormalize softmax
```

Top-p (nucleus) filtering:

```
sort probs descending; keep smallest set with cumulative prob >= p; set rest to 0; renormalize
```

Beam search (outline):

- Maintain B partial hypotheses; at each step expand each hypothesis by top candidates, score by sum log-probs (optionally length-penalty), keep top B.

Greedy:

- Choose argmax token each step.

Sampling:

- Sample from p (after optional filtering/temperature) to introduce diversity.

Variables:

- z: logits vector (vocab_size)
- temperature: scalar >0 controlling sharpness (1 = original)
- top_k, top_p: filtering hyperparameters
- B: beam width

### 4. Code & practical application

Install

```bash
pip install transformers torch
```

Minimal generation examples (Hugging Face + PyTorch)

Load model/tokenizer:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
```

Greedy generation:

```python
prompt = "Climate change will"
input_ids = tok(prompt, return_tensors="pt").input_ids
gen_ids = model.generate(input_ids, max_length= input_ids.shape[1] + 30, do_sample=False)
print(tok.decode(gen_ids[0], skip_special_tokens=True))
```

Sampling with temperature, top-k, top-p:

```python
gen_ids = model.generate(
    input_ids,
    max_length=input_ids.shape[1] + 60,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
    eos_token_id=tok.eos_token_id,
)
for i, g in enumerate(gen_ids):
    print(f"Sample {i}:", tok.decode(g, skip_special_tokens=True))
```

Top-p implemented manually (inspect logits and sample one step):

```python
import torch.nn.functional as F

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]                   # [vocab]
    logits = logits / 0.8                            # temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    cutoff = (cumsum > 0.9).nonzero()[0].item()
    keep_idx = sorted_idx[: cutoff + 1]
    mask = torch.ones_like(probs) * 0.0
    mask[keep_idx] = probs[keep_idx]
    mask = mask / mask.sum()
    next_token = torch.multinomial(mask, num_samples=1)
```

Beam search usage via generate:

```python
beam_out = model.generate(input_ids, max_length= input_ids.shape[1]+40, num_beams=5, early_stopping=True)
print(tok.decode(beam_out[0], skip_special_tokens=True))
```

Practical tips:

- For deterministic generation in production, use greedy or beam with length penalty and reranking.
- For creative outputs, use sampling with tuned temperature and top-p.
- For factual Q&A, prefer beam with grounding (RAG) and output verification.
1. Caching K/V for stepwise decoding (sketch)
- When generating token-by-token, cache per-layer keys and values to avoid recomputing encoder projections for past tokens. HF Decoding uses `past_key_values` to speed up autoregressive looping.

### 5. Visualization / Geometry

- Next-token probability bar chart: shows model uncertainty; sharp peaks → confident picks; broad tails → uncertainty.
- Attention heatmap across layers/heads: inspect which tokens earlier positions each new token attends to; look for patterns:
    - Diagonal locality indicates strong recent-token dependency.
    - Off-diagonal patterns reveal long-range coreference or topic tracking.
- Beam search tree: visualize top hypotheses per time-step; beam collapse (all beams identical) shows lack of diversity.
- Trajectory of hidden states: project hidden vector for the generation token across layers (PCA/UMAP) to see representation drift from lexical to semantic.
- Sampling dynamics: plot entropy of p(y_t|...) over time—entropy spikes often when model “switches topics” or reaches uncertainty.

Quick plotting (next-token distribution):

```python
import matplotlib.pyplot as plt
probs = torch.softmax(logits, dim=-1).cpu().numpy()
topk = 20
ids = probs.argsort()[-topk:][::-1]
plt.bar(range(topk), probs[ids])
plt.xticks(range(topk), [tok.decode([int(i)]) for i in ids], rotation=90)
plt.show()
```

Attention heatmap:

```python
attn = attns[layer_idx][batch_idx, head_idx].cpu().numpy()
plt.imshow(attn, cmap="viridis")
plt.colorbar()
plt.xlabel("key positions")
plt.ylabel("query positions")
plt.show()
```

### 6. Common pitfalls & tips

- Exposure bias: training on ground-truth prefixes (teacher forcing) but sampling model prefixes at inference causes distribution shift; mitigation: scheduled sampling, data augmentation, or sequence-level training.
- Repetition loop and degeneration: sampling with high temperature often creates loops; length penalties, repetition penalties, or nucleus sampling tuned low help.
- Over-confident softmax: logits can be overly peaky and harmful for sampling; temperature lowers peakiness.
- Beam search hallucination: beam favors high-probability n-grams and can produce safe but bland or repetitive text; rerank outputs using external scoring (factuality, diversity).
- Tokenization sensitivity: outputs depend on token-level granularity. Multi-token label handling in scoring must account for subword splits.
- Deterministic vs diverse tradeoff: production NLG often prefers reliability over diversity; choose decoding accordingly.
- Caching mistakes: incorrect past_key_values alignment yields incoherent generations; ensure proper shape/time-order concatenation.
- Safety: model can generate toxic or false content — apply filters, grounding, and verification.
- Latency vs quality: larger models and complex decoding (big beams) increase latency; use distillation, quantization, or parameter-efficient adapters.

### 7. Interview-ready insights

- Why use temperature: dividing logits by temperature T (>0) before softmax rescales distribution: T<1 sharpens (more greedy), T>1 flattens (more diverse). Equation: p ∝ softmax(z/T).
- Top-k vs top-p: top-k constrains candidates to fixed-size; top-p (nucleus) adapts candidate set to probability mass, often preserving tail information for rare but plausible words.
- Beam search scoring nuance: cumulative log-prob favors shorter sequences; length penalty or normalized log-prob avoid bias. Standard formula:

```
score(hyp) = sum_t log p(y_t) / (len(hyp) ^ alpha)   # alpha length penalty
```

- Why caching matters: naive autoregressive recomputation is O(T^2) per generation; caching K/V reduces per-step compute to O(T) across layers and yields large speedups.
- Tradeoffs: Sampling for creative tasks; beam for optimization objective (log-prob); sampling+higher temperature for diversity but more hallucination risk.
- Measuring generation quality: combine automatic (BLEU, ROUGE, BERTScore, perplexity) with task-specific metrics and human evaluation. For factuality use retrieval/verification pipelines.
- Exposure bias solutions: scheduled sampling, professor forcing, Minimum Risk Training, or policy-gradient (RL) optimizing sequence-level metrics.

Short talking points for interviews:

- "Top-p sampling adapts candidate set by cumulative probability, avoiding arbitrary cutoffs of top-k."
- "Caching past key/values is essential for fast autoregressive decoding; it's why production decoders are orders of magnitude faster than naive loops."
- "Length normalization corrects beam’s preference for short sequences, avoiding truncated outputs."

### 8. Practice exercises

Exercise 1 — Next-token probe

- Task: For prompt "The capital of India is", compute top-10 next-token probabilities, show bar chart, and explain why the model chose the top token.
- Hint: inspect logits at last position and consider tokenization (e.g., " New", "Delhi").

Exercise 2 — Decoding comparison

- Task: Generate 5 continuations of same prompt with greedy, beam (width 5), top-k=50 sampling, top-p=0.9 sampling (temp 0.8). Compare diversity, coherence, and any hallucinations. Report qualitative observations.
- Hint: use num_return_sequences>1 for sampling and set seed to compare variability.

Exercise 3 — Implement top-p sampling by hand (one-step)

- Task: Using raw logits from the model, implement top-p filtering and sample the next token as shown in the manual code example above.
- Hint: sort probs and compute cumulative sum to pick cutoff.

Exercise 4 — Beam search with length penalty

- Task: Implement plain beam search (beam width B=3) for a tiny toy vocab and scoring logits, add length penalty alpha=0.7, and compare with greedy.
- Hint: maintain list of (sequence, score) and expand each step; prune to top-B.

Exercise 5 — Diagnose repetition

- Task: Prompt a model with a short lead and generate a long continuation (200 tokens) using high temperature (1.2) and low (0.7). Observe repetition modes. Try repetition_penalty (Hugging Face option) and report improvements.
- Hint: repetition_penalty multiplies logit of already-seen tokens to discourage repeats.

Exercise 6 — Caching practice

- Task: Use HF model with past_key_values to generate step-by-step while printing shapes of cached K/V tensors for each layer. Time naive vs cached generation for 50 tokens and report speedup.
- Hint: call model.generate vs manual loop with past_key_values; compare .inference_time or use time.time().

Exercise 7 — Grounded generation check

- Task: Implement a small RAG: index 100 short docs, retrieve top-3 for a query, form prompt = "Context: <docs> Query: <q>" and generate answer. Compare factuality to plain model answer.
- Hint: use sentence-transformers for embeddings and a small FAISS index.

---

## Prompting and prompt engineering

### 1. Direct definition

Prompting is the act of framing input (text, instructions, examples, context) given to an LLM so the model produces the desired output. Prompt engineering is the deliberate design, testing, and tuning of prompts (templates, few-shot examples, constraints, context) to reliably steer model behavior toward specific tasks, quality, and constraints.

### 2. Concept intuition

- What it is: A prompt is the model’s context — the conditioning that defines the distribution p(y | prompt). Good prompts shape that conditional distribution to concentrate probability mass on useful, correct, or safe outputs.
- Why it matters: For many applications you can get production-quality behavior without fine-tuning by carefully designing prompts; this is cheaper, faster, and preserves the base model’s generality.
- Analogy: Prompting is like crafting a short brief for an expert: clarity, examples, format constraints, and evaluation criteria dramatically change the expert’s deliverable.
- Key levers you control:
    - Task specification (instruction vs question vs blank fill)
    - Format constraints (output schema, JSON, bullet points)
    - Examples (zero-shot, few-shot, chain-of-thought)
    - Context grounding (retrieved documents, factual snippets)
    - Decoding hyperparameters (temperature, top-p, beams)
    - Post-processing (parse/validate outputs, apply filters)
- Two useful patterns:
    - Prompt templates: fixed structure with variable slots, reusable and testable.
    - Prompt chaining: break complex tasks into smaller prompts (decompose → solve → combine).

### 3. Mathematical breakdown (how prompts alter the model distribution)

Conditional probability under prompt c:

```
p(y | c) = product_{t=1..T} p(y_t | c, y_{<t})
```

Change in distribution by adding instruction I or examples E:

```
p(y | c = [I, E, prompt]) != p(y | c' = [prompt])
```

Few-shot as implicit posterior shaping:

```
p(y | I, (x1,y1), ..., (xk,yk), x_query) ≈ p_theta(y | context_with_examples)
```

Few-shot examples act as context-based demonstrations that bias token probabilities for the query.

Chain-of-thought (CoT) factorization:

```
p(answer | prompt) = sum_{chains} p(chain, answer | prompt)
```

Encouraging intermediate chain tokens increases probability mass on reasoning paths that lead to better answers.

1. Scoring labels via conditional likelihood (prompt-based classification):

```
score(label_j) = log p(label_tokens_j | prompt + input)
prediction = argmax_j score(label_j)
```

1. Calibration via logit-adjustment:

```
p_adj = softmax( logits - bias_label )
```

You can adjust logits or apply temperature scaling to calibrate probabilities.

Variables:

- c: prompt context (instruction, examples, retrieved docs)
- y: generated sequence
- k: number of few-shot examples
- label_tokens_j: tokenization of class label j

### 4. Code & practical application (templates, recipes, and experiments)

1. Prompt template (zero-shot instruction)

```python
prompt = """You are a helpful assistant.
Task: Summarize the following article in one sentence.
Article:
{article}

One-sentence summary:""".format(article=article_text)
```

1. Few-shot pattern

```python
examples = [
    ("Translate to French: I love apples.", "J'aime les pommes."),
    ("Translate to French: She is reading.", "Elle lit.")
]
few_shot = "\n".join([f"Input: {x}\nOutput: {y}" for x,y in examples])
prompt = few_shot + "\nInput: " + new_input + "\nOutput:"
```

1. Label scoring (zero-shot classification) — compute log-likelihood per label

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

def score_label(prompt, label_text):
    in_ids = tok(prompt, return_tensors="pt").input_ids
    lab_ids = tok(label_text, return_tensors="pt").input_ids
    ids = torch.cat([in_ids, lab_ids], dim=1)
    with torch.no_grad():
        logits = model(ids).logits
    # sum log-probs of label tokens conditioned on prompt
    lab_logits = logits[0, -lab_ids.shape[1]:-1]  # shift alignment
    lab_token_logprobs = torch.log_softmax(lab_logits, dim=-1)
    score = 0.0
    for i, tok_id in enumerate(lab_ids[0]):
        score += lab_token_logprobs[i, tok_id].item()
    return score
```

1. CoT prompting example (few-shot)

```
Example 1:
Q: If 3 machines take 3 hours to make 3 widgets, how long for 100 machines to make 100 widgets?
A: Each machine makes 1 widget in 3 hours, so 100 machines make 100 widgets in 3 hours. Answer: 3 hours.

Now answer:
Q: ...
A:
```

1. Constrained output JSON template

```
Prompt:
You will reply with JSON only. Keys: "summary" (string), "key_points" (list of strings).
Article:
{article}

Return:
```

Then parse the model output with a JSON parser and validate.

1. Prompt gradients (soft prompts / prompt tuning sketch)
- Soft prompt p (learnable embeddings) prepended to embeddings E:

```
h0 = concat( p (learned embeddings), Embedding(tokens) )
```

Optimization updates p by backprop while keeping main model frozen.

Practical tips:

- Always include an explicit output format and few examples when possible.
- Use retrieval to ground facts in prompts (RAG style).
- Log and version prompts; treat them like code with tests and A/B experiments.
- Use temperature and top-p with care: more aggressive sampling needs stronger output constraints.

### 5. Visualization / Geometric intuition

- Prompt as conditioning vector: think of the prompt as shifting the initial hidden representation manifold; examples shift which basin of attraction the decoding dynamics fall into.
- Few-shot as local attractors: supplying examples creates local clusters of hidden trajectories that guide generation towards demonstrated mapping patterns.
- CoT as path expansion: asking for intermediate steps expands probability mass over extended sequences (the chain), making correct final answers more likely when the model can generate valid intermediate reasoning tokens.
- Embedding-space view:
    - Hard prompt tokens alter initial token embeddings fed into the Transformer.
    - Soft prompts (learned embeddings) are explicit vectors that move hidden states in embedding space before model layers, effectively steering attention and final logits.
- Visual tools:
    - t-SNE/PCA of hidden states for the same query with different prompts to see how prompts change internal states.
    - Attention heatmaps showing how a prompt’s example tokens get attended to when generating the answer.

Small visual experiment:

- Feed same query with two prompts A and B to the model, capture layer L hidden states for the first generated token, reduce to 2D with PCA, and plot to see separation.

### 6. Common pitfalls & tips

- Ambiguous instruction: vague prompts produce inconsistent outputs; be explicit about format, constraints, and examples.
- Overly long prompts: long unstructured context can overwhelm and introduce irrelevant signals; use retrieval and summarization.
- Example contamination: few-shot examples must be representative and non-contradictory; contradictory examples confuse the model.
- Label tokenization leakage: when scoring labels by token likelihood, check whether label words split into multiple subwords and adjust scoring accordingly.
- CoT hallucination: chain-of-thought can produce plausible but incorrect intermediate steps; add verification or self-consistency checks.
- Prompt brittleness: small wording changes may produce big output changes; test prompts across paraphrases and edge cases.
- Output parsing fragility: never trust raw text for structured outputs; enforce schemas and validate.
- Implicit bias in examples: examples carry style and factual bias; be mindful of selection and diversity.
- Repetition from templates: overly prescriptive templates can cause generic or repetitive answers; balance structure with flexibility.
- Leakage & privacy: avoid including sensitive data in prompts; be conscious of data sent to the model.

Practical mitigation:

- Validate outputs automatically (schema validators, unit tests, factuality checks).
- Use ensemble prompts (multiple prompts, aggregate outputs) for robustness.
- Keep prompts short, template-driven, and include negative examples if needed (showing what *not* to do).
- For high-stakes tasks, combine prompting with retrieval and explicit verification steps.

### 7. Interview-ready insights

- Why few-shot works: the model conditions on demonstrations; in a very large pretrained model, examples bias internal activations and condition generation in-context without parameter updates.
- Soft prompts vs hard prompts:
    - Hard prompts: human-readable tokens you craft.
    - Soft prompts (prompt tuning): learned continuous embeddings updated via gradient descent—parameter-efficient for specialization.
- Chain-of-thought benefit: increasing latent reasoning tokens often leads to better multi-step problem solving, but it increases token usage and can hallucinate—use self-consistency and verification.
- Calibration and scoring: prompting for classification can be framed as likelihood scoring; to compare labels, compute conditional log-likelihoods and apply length normalization for multi-token labels.
- Prompt robustness testing: treat prompts like software: unit tests, edge cases, adversarial paraphrases, and A/B evaluation.
- Prompt engineering as productization: design prompts with operational constraints (latency, token cost, interpretability, safety), maintain versioning, monitor metrics (utility, hallucination rate), and guardrails (filters, rerankers).
- When to fine-tune instead: if prompts require very high accuracy across well-defined data or you have abundant labeled data, fine-tuning or adapters may be preferable for consistent behavior.
- Decoding interplay: prompts and decoding hyperparameters interact — a more constrained prompt may allow higher-temperature sampling, while open prompts may need low temperature to reduce hallucinations.

Concise technical note to mention:

- Self-consistency trick: sample multiple chain-of-thoughts, take majority vote on final answers across sampled chains to improve correctness.

### 8. Practice — exercises, small projects, and templates

Exercise 1 — Prompt tuning exploration (hard prompts)

- Task: Create three different zero-shot prompts to ask an LLM to extract the "main claim" from a short paragraph. Compare outputs and compute semantic similarity of results (use SBERT embeddings). Report which prompt is most robust across 10 paragraphs.
- Hint: vary explicitness, output format, and few-shot examples.

Exercise 2 — Few-shot classification with likelihood scoring

- Task: Implement label scoring where for each candidate label you compute conditional log-likelihood of the label given prompt+input and pick argmax. Test on a small sentiment dataset and compare with a fine-tuned classifier.
- Hint: For multi-token labels, sum token log-probs; normalize by label length if needed.

Exercise 3 — Chain-of-Thought vs Direct answer

- Task: For 20 multi-step math word problems, prompt the model with (a) direct answer instruction and (b) CoT few-shot examples. Compare accuracy. Then implement self-consistency: sample 5 CoT outputs per question and pick the most common final answer. Measure improvement.
- Hint: Use temperature >1 for CoT diversity in self-consistency.

Exercise 4 — Template + JSON parsing and validation

- Task: Design a prompt that instructs the model to return a fixed JSON schema for product metadata (title, price, categories). Generate outputs for 50 product descriptions and implement automated JSON parsing + schema validation. Measure parse success rate and common errors.
- Hint: Include "Return only valid JSON, no commentary" and give 2 valid examples in prompt.

Exercise 5 — Prompt chaining pipeline

- Task: Build a two-step pipeline: (1) short-list candidate facts from a document using a prompt that returns bullet points, (2) synthesize a final answer using those bullets. Compare with single-step direct prompt in terms of factuality and conciseness.
- Hint: Use retrieval to supply the document and enforce format on step 1.

Exercise 6 — Soft prompt tuning (advanced)

- Task: Freeze a small pretrained model and train a soft prompt (learnable embeddings) for a new micro-task (e.g., sentiment on niche domain with 200 examples) using gradient descent. Evaluate vs few-shot prompting.
- Hint: Use Hugging Face's prompt-tuning or implement learned prefix embeddings prepended to input embeddings.

Exercise 7 — Robustness & adversarial paraphrase test

- Task: Create adversarial paraphrases of prompts (10 variants) and measure output variance (embedding cosine). Then use ensemble prompting (3 templates) and majority voting to increase stability.
- Hint: Use back-translation or paraphrase models to generate variants.

---

## Generative configuration

### Direct definition

Generative configuration is the set of design choices, hyperparameters, and system components that determine how a pretrained language model is conditioned, decoded, constrained, and deployed to produce text. It covers prompt structure, decoding algorithm and its hyperparameters, grounding (retrieval/tooling), safety filters, resource/latency settings, and monitoring/metrics that together control quality, cost, and risk of generated outputs.

### Concept intuition

- What it is: Think of generative configuration as the model’s operating system settings plus the brief you give it. Two systems with the same model weights can behave very differently by changing a few knobs: prompt wording, temperature, top-p, beam width, repetition penalty, and whether the model has external grounding.
- Why it matters: Small config changes shift the distribution p(y | c). Good configuration trades off correctness, creativity, latency, cost, and safety for the real application. Bad configuration produces hallucinations, repetitive text, latency spikes, or unsafe outputs.
- Analogy: If the LLM is an orchestra, the generative configuration is the conductor’s score (prompt), tempo (temperature), number of soloists (beam width / num_return_sequences), and stage notes (safety filters and retrieval context). The same orchestra plays very different music under different conductors.

### Mathematical breakdown (core formulas & transforms)

1. Conditional generation factorization (reminder)

```
p(y | c) = ∏_{t=1..T} p(y_t | c, y_{<t})
```

1. Temperature scaling of logits z

```
z' = z / temperature
p = softmax(z')
```

- temperature < 1 → sharper distribution (less diverse)
- temperature > 1 → flatter distribution (more diverse)
1. Top-k truncation

```
keep indices K = topk_indices(z', k)
z'_i = z'_i  if i in K else -inf
p = softmax(z')
```

1. Top-p (nucleus) truncation

```
sort probs descending -> p_sorted
find smallest m s.t. sum_{i=1..m} p_sorted[i] >= p_threshold
keep those m tokens; set others to -inf; renormalize
```

1. Repetition penalty (logit adjustment)

```
for token in generated_set:
    if logit[token] > 0:
        logit[token] /= repetition_penalty
    else:
        logit[token] *= repetition_penalty
```

(Hugging Face implements different variants; conceptually scale logits of previously generated tokens to discourage repeats.)

1. Beam search scoring with length penalty

```
score(hyp) = (1 / (len(hyp) ^ alpha)) * sum_{t=1..T} log p(y_t | context)
```

- alpha > 0 penalizes short hypotheses
1. Reranking with external score (e.g., factuality or QA match)

```
final_score = λ1 * log_prob + λ2 * factuality_score + λ3 * retrieval_score
choose hypothesis with max final_score
```

1. Constrained decoding (hard constraints)
- Use token-level constraint masks C_t such that allowed tokens at time t satisfy C_t. Implement with logits masking:

```
z'_i = z_i if i in allowed_set else -inf
p = softmax(z')
```

1. Cost/latency rough model

```
latency ≈ compute_time_per_token * generated_tokens + overhead_cache
compute_time_per_token ∝ (d_model * seq_len * layers) / (parallelism)
cost ≈ tokens_generated * price_per_token_model
```

### Code & practical application (recipes and runnable examples)

Install prerequisites

```bash
pip install transformers accelerate torch sentencepiece
```

Hugging Face generation templates (common configs)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval().to("cpu")  # move to GPU if available

prompt = "Design a healthy 200-calorie snack for students:"

input_ids = tok(prompt, return_tensors="pt").input_ids

# Example config sets
configs = {
    "deterministic": {"do_sample": False, "num_beams": 1},
    "beam_search": {"num_beams": 5, "early_stopping": True, "length_penalty": 1.0},
    "diverse_sampling": {"do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 0.9, "num_return_sequences": 3},
    "conservative": {"do_sample": True, "top_p": 0.6, "temperature": 0.6},
}

for name, cfg in configs.items():
    out = model.generate(input_ids, max_length=input_ids.shape[1]+80, eos_token_id=tok.eos_token_id, **cfg)
    print(f"\n--- {name} ---\n", tok.decode(out[0], skip_special_tokens=True))
```

Manual one-step top-p sampling (inspect internals)

```python
import torch.nn.functional as F

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]  # last position
    temperature = 0.8
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    # top-p filter
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    cutoff = (cumsum > 0.9).nonzero()[0].item()
    candidate_idx = sorted_indices[:cutoff+1]
    candidate_probs = probs[candidate_idx]
    candidate_probs = candidate_probs / candidate_probs.sum()
    next_tok = candidate_idx[torch.multinomial(candidate_probs, 1)]
```

Constrained decoding by regex-like constraints (via token mask)

- Build allowed token ids set at each step, mask logits:

```python
allowed = set([tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]])
mask = torch.full_like(logits, -1e9)
mask[list(allowed)] = 0.0
logits_masked = logits + mask
probs = F.softmax(logits_masked, dim=-1)
```

Caching and stepwise generation (speed improvement)

- Use model.generate for built-in caching, or manual loop with past_key_values

```python
past = None
generated = input_ids
for _ in range(50):
    out = model(generated[:, -1:], past_key_values=past, use_cache=True)
    logits = out.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    past = out.past_key_values
```

RAG-style grounding (sketch)

- Embed docs, index with FAISS, retrieve top-k, form prompt:

```
prompt = "Context:\n<doc1>\n<doc2>\nQuestion: Q\nAnswer:"
```

- Then generate conditioned on prompt; optionally rerank beams by doc overlap or external verifier.
1. Safety filters and classifiers
- After generation: run a toxicity classifier and factuality verifier. If fail, rerun with conservative config or decline.

### Visualization & geometric intuition

- Entropy over time: plot entropy H(p_t) = -Σ p_t log p_t across generated tokens; spikes show uncertainty or topic shifts.
- Next-token distribution bar plots: compare distributions under different temperatures/top-p to see effect on mass concentration.
- Beam evolution tree: draw top-B hypotheses at each time step with scores; show how beams converge or collapse.
- Attention shift when adding retrieval: visualize attention maps before/after adding retrieved context — attention should increase on retrieved passages for grounding tokens.
- Cumulative cost curve: x-axis tokens generated, y-axis compute time or token cost — visualize tradeoffs for configuration choices.
- Sampling diversity heatmap: compute pairwise BLEU or embedding cosine across num_return_sequences samples to measure diversity for given config.

Quick code to compute entropy:

```python
import torch
import torch.nn.functional as F

probs = F.softmax(logits, dim=-1)
entropy = - (probs * torch.log(probs + 1e-12)).sum().item()
```

### Common pitfalls & tips

- Mis-tuning temperature: small changes can flip behavior; tune on validation prompts, not just eyeballing outputs.
- Confusing top-k and top-p effects: top-k fixes candidate count; top-p adapts mass — prefer top-p for adaptive diversity, top-k for speed control.
- Beam search degeneracy: beams often produce bland or repetitive text; rerank with external metrics (factuality, diversity) or use diverse beam search variants.
- Unintended constraints: constrained decoding can over-constrain leading to -inf logits for every candidate; always check mask coverage.
- Repetition loops: long-generation loops often repeat phrases; use repetition_penalty, min_length, and penalize recently-generated n-grams.
- Caching mistakes: wrong sequence concatenation or mismatch of past_key_values order creates incoherent outputs — test caching on short sequences.
- Costly configs in production: do_sample with many return sequences or large beams multiplies model calls — budget costs accordingly.
- Safety & hallucination: low temperature and retrieval help but do not eliminate hallucination; always add verification for high-stakes outputs.
- Tokenization leaks: format constraints relying on tokens must account for subword splits — validate token sequences rather than raw text when enforcing constraints.

Tips:

- Maintain a small validation set of prompts representing expected workload; measure hallucination, exactness, and latency across configs.
- Use parameter-efficient methods (LoRA/Adapters) rather than often retraining the model; use RAG for up-to-date factual answerability.
- For production, precompute embeddings and cache generated prefixes for frequent prompts.

### Interview-ready insights

- Temperature formula and effect: z' = z / T; softmax(z') sharpens/flattens distribution. Use T<1 for precision, T>1 for creativity.
- Top-k vs Top-p: top-p adapts candidate count to probability mass making it more robust across vocab/distribution shifts; top-k guarantees bounded compute.
- Beam search objective bias: cumulative log-prob favors short sequences; apply length penalty: score = sum_logprobs / (len^alpha).
- Reranking improves quality: generate N candidates (beams or samples), then rerank with task-specific scorers (QA-checker, exact match tests, secondary classifier) — commonly used in production.
- Caching reduces inference from O(T^2) to O(T) per token for autoregressive decoders by reusing past keys/values.
- Deterministic vs stochastic trade-offs: production assistants often prefer deterministic outputs for consistency; creative tools prefer stochastic sampling.
- Constrained decoding and safety: constrained decoding enforces formats and can prevent certain hallucinations but needs careful token-level design and fallback logic.

Concise example to state in interviews:

- "In production, I usually run a two-stage pipeline: (1) fast candidate generation with a conservative config (top-p 0.8, temp 0.7) and (2) rerank candidates using an external verifier that checks grounding, toxicity, and factuality, selecting the highest-scored non-violating candidate."

### Practice exercises (with hints)

Exercise 1 — Config sweep and evaluation

- Task: Choose 10 representative prompts. For each prompt, generate outputs under grids of temperature ∈ {0.4,0.7,1.0} and top_p ∈ {0.6,0.9,0.95}. Compute per-prompt metrics: average token entropy, pairwise sample diversity (embedding cosine), and human-rated coherence (1–3). Summarize best tradeoff for your application.
- Hint: Automate generation and store outputs; use sentence-transformers for embeddings.

Exercise 2 — Implement top-p by hand and compare with HF

- Task: For one prompt, implement manual one-step top-p sampling using logits (see manual snippet) and compare next-token distributions with HF generate using top_p to ensure parity.
- Hint: careful with float rounding when computing cumulative sum cutoff.

Exercise 3 — Beam search + reranking

- Task: Generate 8 beams (num_beams=8) for technical question answers. Implement a reranker that scores factuality by computing overlap of named entities with a retrieved knowledge snippet (simple exact-match). Compare final answer with plain highest-prob beam.
- Hint: use spaCy for NER and simple token overlap ratio.

Exercise 4 — Constrained JSON generation

- Task: Design constraint masks so the model can only output tokens that form valid JSON keys and values for a fixed schema. Generate 50 examples and compute parse success rate. If parse fails, implement two-step fallback: (A) ask model to correct the JSON; (B) run grammar-based repair.
- Hint: map allowed characters to token ids; consider subtokens for punctuation.

Exercise 5 — Caching performance comparison

- Task: Time naive autoregressive generation (recomputing full forward each token) vs cached past_key_values generation for 100 tokens. Report speedup for your environment.
- Hint: use torch.cuda.synchronize() if measuring on GPU and run multiple trials.

Exercise 6 — Safety loop with conservative re-gen

- Task: Generate an answer; if a toxicity classifier flags it, regenerate with a conservative config (temperature 0.5, top_p 0.5) and add "I cannot assist" fallback after 2 failed attempts. Log decisions.
- Hint: accumulate a counter for attempts and keep small timeout for classify step.

---

## AI Project Lifecycle

### 1. Direct definition

An AI project lifecycle is the end-to-end sequence of stages you follow to deliver a working, maintainable, and safe AI system: problem framing → data collection & labeling → prototyping & modeling → evaluation & validation → deployment & serving → monitoring & maintenance → governance & compliance → iteration & scaling.

### 2. Concept intuition

- What it is: a flow that turns a real-world problem into a reliable AI product. Each stage imposes technical, organizational, and risk constraints that determine what models are appropriate and how they must be integrated.
- Why it matters: most project failures come from skipping steps (poor data, missing evaluation, no monitoring, ignored safety). Treat models as components in a larger system that includes data pipelines, human review, and feedback loops.
- Analogy: building an AI product is like building a bridge — you need site surveys (data audit), blueprints (design), stress tests (evaluation), and ongoing inspections (monitoring). Skipping inspections causes collapse.
- Key cross-cutting concerns: reproducibility, versioning (data, code, model), cost, latency, privacy, and governance.

### 3. Mathematical & operational breakdown

Project-level objectives map to optimization objectives:

- Task loss (training objective)

```
Loss(theta) = E_{(x,y)~D_train}[ l(f_theta(x), y) ]   # e.g., cross-entropy, MSE
```

- Generalization and validation

```
Generalization gap = E_val[ loss ] - E_train[ loss ]
```

- Business metric mapping (example):

```
BusinessGain(policy) = sum_{i} weight_i * metric_i  # combine precision, latency, cost
```

- Resource and latency budget (rough constraints)

```
latency <= L_max
memory <= M_max
cost_per_query <= C_budget
```

- Threshold-based deployment rule (simple)

```
if eval_metric >= threshold and safety_checks_pass:
    promote_to_prod()
else:
    iterate()
```

- Monitoring drift detection (statistical test)
    - Population shift: compare feature distribution P_train(X) vs P_live(X), e.g., KL divergence

```
KL(P_train || P_live) = sum_i P_train(x_i) log(P_train(x_i) / P_live(x_i))
```

- Metric alerting: if rolling_mean(metric, window) deviates > k * std_dev → alert.
- A/B decision criterion (sample size for lift detection)

```
n ≈ 2 * (z_{1-α/2} + z_{1-β})^2 * σ^2 / Δ^2
```

(where σ^2 is variance of metric, Δ is effect size)

Explain variables: theta = model params; l = per-example loss; D_train = training data; L_max/M_max/C_budget = operational constraints; α/β = type-I/II error rates; Δ = minimal detectable improvement.

### 4. Code & practical application (recipes, checklists, and minimal scaffolds)

A. Project kickoff checklist (practical)

- Define problem statement, success metrics (business + ML), and guardrails.
- Data sources, ownership, sampling plan, labeling requirements.
- Compute/infra needs and privacy constraints.
- Baseline model and MVP scope (fast fail).

B. Minimal reproducible experiment scaffold (PyTorch-like pseudocode)

```python
# experiment scaffold
set_seed(42)
load_config("config.yaml")   # model, data paths, hyperparams
dataset = Dataset(data_path)
train_loader, val_loader = dataset.get_loaders(batch_size=config.bs)
model = build_model(config.model)
optimizer = AdamW(model.parameters(), lr=config.lr)
scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.steps)

for epoch in range(config.epochs):
    train_epoch(model, train_loader, optimizer)
    val_metrics = evaluate(model, val_loader)
    log_metrics(epoch, val_metrics)
    save_checkpoint_if_improved(model, val_metrics["primary_metric"])
```

C. Data versioning & lineage (tools and pattern)

- Use: Git-like for code, DVC/DeltaLake/MLFlow for data, model registry (MLflow, Hugging Face Hub) for model artifacts.
- Minimal pattern: store raw_data_hash, preprocessing_code_hash, label_version, train_config in experiment metadata so runs are reproducible.

D. Simple deployment flow (container + API)

- Package model (weights + tokenizer + config).
- Build lightweight inference wrapper that applies preprocessing, model.forward with batching & caching, postprocessing, and safety filters.
- Serve via a REST/gRPC gateway with health checks and rate limits.
- Provide a reject/fallback path (e.g., "I don't know" or human-in-the-loop).

E. Monitoring skeleton (pseudo)

```python
# on each request
log_request({prompt_hash, input_features_stats, model_confidence, latency, output_summary})
# offline batch job
compute_rolling_metrics(window=24h)
if anomaly_detected(metric, threshold): trigger_alert()
```

### 5. Visualization / Geometry (what to monitor and why)

- Model performance dashboards
    - Training curves: loss, accuracy, validation loss. Watch overfitting (divergence).
    - Learning rate and gradient norms per layer.
- Data & feature drift
    - Univariate histograms over time for critical features; population shifts visualized as overlayed KDEs.
    - Embedding-space drift: project sample embeddings (PCA/UMAP) from train vs live to detect distributional shift.
- Prediction diagnostics
    - Confusion matrices sliced by cohort (user segment, geography).
    - Calibration plots: predicted probability vs empirical accuracy (reliability diagram).
    - Perplexity or token-entropy over time for language models.
- Resource & cost
    - Latency percentiles (p50/p95/p99), memory usage, GPU utilization.
- Safety & content metrics
    - Toxicity rate, hallucination flags, flagged policy violations over time.
- A/B and causal impact visualization
    - Cumulative lift curves and confidence intervals for business KPI.

Quick plotting examples:

- Rolling metric plot: metric vs time with alert bands.
- Drift heatmap: pairwise KL divergences across features and cohorts.

### 6. Common pitfalls & practical tips

- Ambiguous success metric: ML accuracy ≠ business value. Always define concrete downstream metrics (revenue uplift, task completion).
- Data leakage: features derived from the target or future timestamps break evaluation. Strict feature cutoffs and time-based splits guard against it.
- Imbalanced evaluation: global averages hide cohort failures. Use slice-based metrics and fairness checks.
- Poor baseline selection: compare to simple baselines (rule-based, heuristics). Complexity should buy measurable value.
- Insufficient logging: no logs → no debugging. Log inputs, outputs, confidence, and environment context with sampling to keep costs manageable.
- Ignoring latency and cost: producing high-quality but slow/expensive models that can’t run in production wastes effort.
- Overfitting to validation: repeated tuning on same val set inflates expected performance. Keep a holdout test set and/or use nested CV.
- No rollback plan: deployments must be able to revert quickly; use canary releases and feature flags.
- Neglecting human-in-the-loop: for high-risk tasks, include human review, escalation paths, and incentives for label quality.
- Governance blind spots: ensure privacy, consent, and data retention policies are followed; maintain audit trails.

Tips:

- Start small with an MVP that proves value and failure modes; expand after stable signals.
- Automate CI for data checks, model training, and unit tests for preprocessing.
- Use parameter-efficient tuning (LoRA/adapters) for fast iteration when model size is large.
- Keep a "model card" and "data card" documenting intended use, limitations, and evaluation.

### 7. Interview-ready insights (concise, high-impact points)

- Metric design: Choose a primary business metric and a set of safety/fairness metrics. Explain expected trade-offs and acceptable guardrail thresholds.
- Experimentation: Use randomized A/B tests for causal inference. Know sample-size formula and how to interpret p-values and confidence intervals in the context of sequential testing.
- Data engineering: Emphasize data contracts, schema checks, null handling, and lineage to prevent silent production errors.
- MLOps: Reproducibility requires versioning of data, code, model, config, and seeds — name tools you’d use and why (DVC/MLflow/Hydra/Git).
- Deployment strategy: Canary rollouts + shadow testing for validating model behavior on live traffic without serving to users; autoscaling + batching to control cost/latency.
- Monitoring & alerts: Track both model performance and input distribution; detect drift early and have automated retraining or human review triggers.
- Safety & alignment: For generative models, use RAG, verification, filters, and human review in the critical path. Design fallbacks and escalation policies.
- Continuous learning: Decide between online learning, periodic retrain, and human-in-the-loop labeling based on data rate, cost, and stability needs.
- Tradeoffs: Present examples of latency vs accuracy, cost vs model size, and retraining frequency vs freshness — and how you’d choose based on SLAs.

### 8. Practice & project exercises

Exercise 1 — Build a minimal ML product (5–7 days)

- Goal: Deliver a small web service that answers domain FAQs using retrieval-augmented generation.
- Steps:
    1. Collect 500 domain docs (or scrape/curate).
    2. Build embeddings index (sentence-transformers + FAISS).
    3. Implement simple RAG pipeline: retrieve top-3, craft prompt, generate answer with a small LLM.
    4. Add simple safety filter: block answers containing banned phrases.
    5. Serve via FastAPI, add logging, and deploy locally with Docker.
- Deliverables: API, README, logs demonstrating example queries and metrics (latency, success).

Exercise 2 — Data pipeline + reproducible experiment

- Goal: Train a classifier (text) with full data lineage and experiment registry.
- Steps:
    1. Ingest dataset, store raw snapshot and compute hash.
    2. Implement deterministic preprocessing pipeline (tokenization, truncation).
    3. Train baseline model and store artifacts (weights, tokenizer, config).
    4. Register run in MLflow/DVC with metrics.
    5. Reproduce the run from artifacts only.
- Deliverables: Reproducible notebook and instructions to re-run experiment.

Exercise 3 — Monitoring & drift detection prototype

- Goal: Demonstrate drift detection and alerting on a deployed model.
- Steps:
    1. Deploy a toy sentiment model behind an API.
    2. Simulate live traffic with slightly shifted text distribution.
    3. Implement a nightly job that computes KL divergence per feature and model accuracy on sampled human labels.
    4. Trigger an alert when drift or accuracy drop exceeds threshold.
- Deliverables: Dashboard screenshots, alert logs, remediation plan.

Exercise 4 — Canary + rollback automation

- Goal: Implement a CI/CD flow with canary rollout and automated rollback.
- Steps:
    1. Build deployment pipeline that deploys new model to canary subset of traffic.
    2. Implement health checks and metric comparison to baseline.
    3. Automate rollback when canary metrics degrade beyond threshold.
- Deliverables: Pipeline config (e.g., GitHub Actions), scripts, and test logs showing rollback.

Exercise 5 — Ethics & governance audit

- Goal: Create a model card and data sheet for an NLP model and run fairness checks.
- Steps:
    1. Document intended use, limitations, training data sources, and known biases.
    2. Run slice analysis by demographic/cohort proxies, generate fairness metrics, and propose mitigations.
    3. Produce an audit report with remediation steps and monitoring plans.
- Deliverables: Model card, metrics, and mitigation plan.

---

## Gen AI use case: summarise dialogue

### Summarize dialogue — direct definition

A dialogue summarization system ingests a multi-turn conversational transcript and outputs a concise, coherent summary that captures the main points, speaker intents, decisions, and/or action items while preserving speaker roles and temporal ordering where relevant.

### Concept intuition

- What it is: compress many utterances into a short representation that keeps what matters and discards filler (backchannels, repeats).
- Why it matters: meeting notes, customer support case summaries, chat logs for moderation, helpdesk ticket creation, CRM enrichment, and concise display on mobile.
- Two useful summary types:
    - Extractive: pick salient utterances or phrases from the original dialogue.
    - Abstractive: generate a new concise text that paraphrases and condenses content.
- Key subtasks: speaker-role detection, topic segmentation, salient-turn identification, coreference resolution (who does “it” refer to), and factual grounding (dates, numbers, commitments).
- High-level pipeline options:
    - Prompting a large decoder model (RAG optional) for abstractive summaries.
    - Fine-tuning a seq2seq model (T5/BART/LED) on paired dialogue→summary data.
    - Hybrid: retrieve important turns, then abstractive model rewrites them.

### Mathematical breakdown

Objective (abstractive seq2seq)

```
Loss = - sum_{t=1..T_y} log p(y_t | x, y_{<t})
```

- x: dialogue token sequence; y: summary tokens.

Extractive scoring (binary labeling of turns)

```
For each turn i: score_i = sigmoid( f_theta(turn_i, context) )
Loss = BCE(score_i, label_i)
```

Importance weighting for training (optional)

```
Loss_weighted = - sum_i w_i * log p(y_i | context)
```

- w_i > 1 for critical tokens (dates, numbers, decisions).
1. Evaluation metrics
- ROUGE (n-gram overlap):

```
ROUGE-N = (sum overlaps of N-grams between system & reference) / (total N-grams in reference)
```

- BERTScore (embedding similarity): cosine similarity over token embeddings, better for paraphrase capture.
- Factuality metrics (QA-based): generate question-answer pairs from reference and check if model answers match.
- Human metrics: coherence, completeness, non-redundancy, factuality, and speaker attribution accuracy.
1. RAG marginalization idea (grounding)

```
p(y | x) ≈ p_gen(y | x)  # if model is grounded
Or: p(y|x) ≈ sum_{r in R_k(x)} p(r|x) p(y | x, r)
```

- r: retrieved context segments.

Variables: x = dialogue, y = summary, f_theta = scoring network, R_k = top-k retrieved passages.

### Code & practical application

Minimal abstractive recipe (Hugging Face, inference)

```bash
pip install transformers torch datasets sentence-transformers
```

Zero-shot summarization via prompt (decoder-only LLM)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

dialogue = """Alice: Hi team, the client wants the ETA moved up to next Monday.
Bob: That's tight; we need two more devs or reduce scope.
Alice: Can QA delay some checks to next sprint?
Claire: We can, but only non-blocking checks.
Alice: Okay, action: request two contractors and draft scope cut list."""

prompt = f"Summarize the dialogue below in 2-3 short sentences, include action items and owners.\n\nDialogue:\n{dialogue}\n\nSummary:"
input_ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(input_ids, max_length=input_ids.shape[1]+80, do_sample=True, top_p=0.9, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=True))
```

Note: for production use an instruction-tuned LLM (FLAN, Llama-instruct, or commercial instruction models).

Fine-tuning a seq2seq model (T5) on a dialogue dataset (outline)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

dataset = load_dataset("knkarthick/dialogsum")  # example dialogue summarization dataset
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def preprocess(ex):
    input_text = "summarize dialogue: " + ex["dialogue"]
    model_inputs = tokenizer(input_text, max_length=512, truncation=True)
    labels = tokenizer(ex["summary"], max_length=128, truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

ds = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
args = TrainingArguments(output_dir="t5-dialog-sum", per_device_train_batch_size=8, num_train_epochs=3, fp16=True)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"])
trainer.train()
```

Practical tips: use long-context models (LED, LongT5, or encoder-decoder with sliding windows) for long dialogues.

1. RAG + summarization (retrieve critical turns then summarize)
- Embed turns (SentenceTransformers), index with FAISS.
- Retrieve top-k turns for a given dialogue, concatenate to prompt with instruction, then generate summary.

Sketch:

```
1. Split dialogue into turns
2. Encode each turn -> vector
3. Index vectors with FAISS
4. For target dialogue: retrieve top-k salient turns (or use simple heuristics like TF-IDF or speaker-role weights)
5. Prompt model: "Using these highlighted turns, summarize..."
6. Generate and postprocess (normalize names, dates)
```

1. Post-processing & validation
- Normalize speaker names, canonicalize dates/numbers, ensure action items are bulletized, and run a factuality QA verifier:
    - Auto-generate questions: "Who is assigned to request contractors?" — answer via model or rule.
    - If verification fails, trigger rerun with stricter decoding or human-in-the-loop.

### Visualization & geometric intuition

- Saliency map: compute token-level gradients of log-prob of summary tokens wrt input embeddings to see which turns influenced the summary most.
- Attention heatmaps: inspect cross-attention (encoder-decoder) to see which source tokens each summary token attended to.
- Turn importance scatter: plot embedding-space distances between turn embeddings and summary embedding — turns closer to the summary are more influential.
- Summary compression curve: plot cumulative information retained (ROUGE/BERTScore) vs summary length to choose target length.
- Timeline view: show original turns on x-axis with highlighted spans that the model used (retrieval or attention peaks), plus final summary aligned below.

Quick code snippet for attention visualization (seq2seq):

```python
# after forward pass with output_attentions=True
# cross_attentions: tuple per layer [batch, tgt_len, src_len]
att = outputs.cross_attentions[-1][0].mean(axis=0)  # average over heads -> [tgt_len, src_len]
# plot heatmap of att for the first summary token or aggregate over target tokens
```

### Common pitfalls & tips

- Long dialogues exceed context: use segmentation + hierarchical summarization (summarize segments then summarize summaries).
- Speaker attribution errors: tokenization and name variants break ownership; canonicalize names and include speaker markers in prompt ("Alice:").
- Hallucination of facts: abstractive models may invent dates or commitments — always ground with RAG or apply verification.
- Repetitive or generic summaries: tuning the decoding (lower temperature, top-p) and using factuality rerankers reduces blandness.
- Evaluation mismatch: ROUGE favors lexical overlap but misses paraphrase quality — include embedding-based and human evals for coverage, correctness, and attribution.
- Data bias: training data may emphasize certain turn types; ensure dataset diversity across domains and participant roles.
- Label granularity: some references include full annotated action items; others only high-level summaries — be consistent when fine-tuning.
- Privacy: dialogues often contain PII; redact or apply differential privacy in training and deployment.

Practical mitigations:

- Use extract+abstractive hybrid: extract salient turns then generate; reduces hallucination.
- Enforce output schema: ask model to return JSON with fields {summary, action_items:[{text, owner, due_date}]} and validate.
- Use ensemble verification: generate multiple summaries, run fact-checker or QA-based verification, and pick the best.

### Interview-ready insights

- When to fine-tune vs prompt:
    - Prompting / instruction-tuned LLMs work well for low-volume or evolving domains.
    - Fine-tune seq2seq when you have moderate labeled pairs and need consistent, low-latency, cost-effective inference.
- Hierarchical summarization pattern:
    - Chunk long transcripts → per-chunk summaries → aggregate summary → optional compression step.
    - This reduces O(T^2) attention issues and helps with long contexts.
- Extractive pre-selection advantage:
    - Selecting top-k salient turns reduces noise and gives the abstractive model high-signal input; easier to verify and cheaper.
- Factuality strategy:
    - Ground outputs with retrieval and use QA-based factuality checks; for critical action items, require exact text match or human confirmation.
- Metrics to report:
    - ROUGE/BERTScore for automated tracking, human-rated factuality, speaker-attribution accuracy, action-item extraction precision/recall, and latency/cost per summary.
- Practical decode choices:
    - Use beam search + length penalty for concise, high-likelihood summaries; use constrained decoding or format instructions for structured outputs.

### Practice exercises

Zero-shot prompt engineering

- Task: Given 20 dialogues from a small dataset, design three prompt templates (short instruction, few-shot, and action-item-focused) and compare outputs for consistency and action-item extraction accuracy against references.
- Hint: Measure overlap for action items via token-level F1; use SBERT cosine for summary similarity.

Fine-tune T5 on DialogSum (toy)

- Task: Fine-tune T5-small on 1,000 dialogue→summary pairs, validate on 200 pairs, report ROUGE and BERTScore, and show 10 qualitative examples.
- Hint: Use max_input_length=512, max_target_length=128, batch_size=8, num_epochs=3, lr=3e-5.

Extractive + Abstractive hybrid

- Task: Build an extractor that scores turns by TF-IDF and a classifier, pick top-5 turns, then feed them to a generator to produce a summary. Compare with full-dialogue generation in speed and factuality.
- Hint: Use sentence-transformers for turn embeddings and cosine similarity with summary embeddings as a proxy.

Hierarchical summarization for long chat

- Task: Implement chunking (512-token windows), get chunk-level summaries, then a final merge summarizer to produce a concise result. Visualize attention to chunk summaries.
- Hint: Use sliding windows with overlap to avoid losing boundary information.

Action-item extraction and normalization

- Task: Prompt an LLM or fine-tune a classifier to extract (action_text, owner, due_date) triples from dialogues. Evaluate precision/recall on a small annotated set.
- Hint: For owner normalization, maintain a name-entity map (aliases → canonical name) built from dialogue metadata.

Factuality QA-check

- Task: Generate a summary, auto-generate 5 fact-check questions (Who, What, When), let an answer-extraction model answer from the original dialogue, and compare answers to the summary’s claims. Flag mismatches.
- Hint: Use an off-the-shelf QA model (SQuAD-style) for extraction.

---

## Pretraining large language models (LLMs)

### Direct definition

Pretraining large language models (LLMs) is the stage where a high-capacity neural network is trained on massive unlabeled text corpora to learn general-purpose language representations and a conditional next-token distribution. Pretraining establishes the base capabilities (syntax, semantics, world knowledge, reasoning priors) that later adaptation (fine-tuning, instruction tuning, RLHF, RAG) leverages.

### Concept intuition

- What it is: Teach a network to predict masked tokens or next tokens across billions of words so it internalizes statistical regularities of language and facts about the world. The model learns representations useful for many downstream tasks without supervised labels.
- Why it matters: Good pretraining yields models that transfer well with little supervision, generalize to new tasks via prompting or lightweight adaptation, and provide robust starting points for alignment and domain specialization.
- High-level design choices that shape capabilities:
    - Objective: autoregressive (next-token), masked (MLM), or span/masked-infilling.
    - Architecture: decoder-only, encoder-only, or encoder-decoder.
    - Data mix: web pages, books, code, dialog, scientific text; quality vs scale trade-offs.
    - Tokenization and vocab design: affects rare word handling and efficiency.
    - Optimization strategy: batch size, learning rate schedule, mixed precision, gradient accumulation, stability tricks.
- Analogy: Pretraining is like general education; it gives broad literacy, while fine-tuning is vocational training for a job.

### Mathematical breakdown

Core pretraining objectives

- Autoregressive next-token (causal) objective:

```
Given sequence x = [x1..xT]:
Loss = - sum_{t=1..T} log p_theta(x_t | x_{1:t-1})
p_theta(x_t | x_{<t}) = softmax( Linear(h_t) )
```

- Masked (BERT-style) masked language modeling objective:

```
Randomly mask positions M.
Loss = - sum_{i in M} log p_theta(x_i | x_{[1:T] \ M})
```

- Span / infilling (T5/BART):

```
Replace spans with sentinel tokens, train seq2seq to reconstruct spans.
Loss = - sum_{t} log p_theta(y_t | x_masked, y_{<t})
```

Model parameter update (mini-batch SGD/AdamW):

```
theta <- theta - lr * m_t / (sqrt(v_t) + eps)   # Adam-style update (simplified)
```

Perplexity (training monitor):

```
Perplexity = exp( (1/N) * sum_{i=1..N} -log p_theta(x_i) )
```

Effective batch and data sampling

- When mixing corpora with different sizes/quality, sampling probabilities pi can be weighted:

```
sample_document d with probability p(d) ∝ weight(source_of_d) * freq(d)^alpha
```

- Temperature scaling of sampling (alpha controls up/downsampling rare sources).

Scaling laws (empirical)

- Rough relation among model size (N parameters), dataset size (D tokens), and compute (C FLOPs):

```
Loss ≈ A * N^{-α} + B * D^{-β} + noise  # empirical power laws; α, β ~ small positive
```

- Practical implication: performance improves with proportional scaling of model, data, and compute; mismatch causes inefficiency.

Regularization and stability formulas

- Weight decay (AdamW) applied to parameters:

```
theta <- theta - lr * (grad + weight_decay * theta)
```

- Gradient clipping by norm:

```
g = grad; if ||g|| > clip_value: g = g * (clip_value / ||g||)
```

Checkpoint averaging / exponential moving average (EMA)

```
theta_avg = (1/k) * sum_{i=1..k} theta_i  # or EMA: theta_ema <- β * theta_ema + (1-β) * theta
```

- Useful to stabilize final weights and improve generalization.

Explain variables:

- theta: model parameters; h_t: hidden state at position t; lr: learning rate; α, β: scaling exponents; N, D, C: size/compute; p(d): sampling probability.

### Practical pretraining pipeline & code snippets

Notes: full-scale pretraining uses distributed GPU/TPU clusters. Below are compact, runnable sketches for small-scale experiments that illustrate key mechanics.

Data pipeline (tokenization, chunking, sampling)

```python
# sketch: tokenize and make contiguous blocks of tokens
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
texts = ["..."]  # list of documents
tokens = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
# flatten and chunk
flat = [tok for doc in tokens for tok in doc]
block_size = 1024
blocks = [flat[i:i+block_size] for i in range(0, len(flat)-block_size+1, block_size)]
# dataset yields fixed-length input_ids for training
```

Simple training loop (toy, single-process)

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

class BlockDataset(Dataset):
    def __init__(self, blocks): self.blocks = blocks
    def __len__(self): return len(self.blocks)
    def __getitem__(self, i): return torch.tensor(self.blocks[i], dtype=torch.long)

model = AutoModelForCausalLM.from_pretrained("gpt2")
ds = BlockDataset(blocks)
loader = DataLoader(ds, batch_size=4, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(3):
    for batch in loader:
        inputs = batch.cuda()
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
```

Mixed-source sampling (importance weighting)

```python
# assume corpora = {"web": web_blocks, "books": book_blocks}
# set up sampling probabilities
weights = {"web": 0.6, "books": 0.4}
# to produce training batch, sample source by weights, then sample a block within that source
```

Learning rate schedule with warmup + decay

```python
from transformers import get_cosine_schedule_with_warmup
optimizer = torch.optim.AdamW(...)
total_steps = epochs * len(loader)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
# call scheduler.step() each step

```

Gradient accumulation for large effective batch sizes

```python
accum_steps = 8
opt.zero_grad()
for step, batch in enumerate(loader):
    loss = model(batch, labels=batch).loss
    (loss / accum_steps).backward()
    if (step + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); scheduler.step(); opt.zero_grad()
```

Checkpoint averaging (stable final model)

```python
# save periodic checkpoints, then average last k checkpoints' weights elementwise
```

Practical tips:

- Use fp16 or bfloat16 mixed precision to reduce memory and increase throughput.
- Use distributed data parallelism (DDP) with gradient accumulation to simulate huge batches.
- Monitor gradient norms, loss scale, and ensure no NaNs appear.

### Visualization and geometric intuition

- Token embedding manifold: visualize token vectors by PCA/UMAP; semantically similar tokens cluster; positional encodings shift representations by token index.
- Layerwise representation drift: project hidden states for the same token across layers to see progressive abstraction (lexical → syntactic → semantic).
- Loss & training curves: plot training loss vs steps and validation loss/perplexity; watch for divergence indicating instability.
- Gradient norm per-layer: plot ||∇_θ L|| for each layer to detect imbalance; rescale via per-layer lrs or use AdamW.
- Data sampling effects: plot performance vs data mix (e.g., tokens from source A vs B) to see benefits of up/downsampling.
- Scaling experiments: plot validation loss vs model size or compute to observe empirical power-law behavior.

Quick code to log training metrics:

```python
# inside training loop
train_loss = loss.item()
print(f"step {step}, loss {train_loss}, grad_norm {grad_norm}")
# visualize later with TensorBoard or matplotlib
```

### Common pitfalls & tips

- Data quality vs scale: more data helps but noisy data (spam, boilerplate, PII) degrades behavior. Clean and filter (deduplication, remove low-quality sources).
- Tokenization mismatches: inconsistent tokenizers across pretraining and fine-tuning cause trouble; freeze tokenizer design early.
- Catastrophic under/oversampling: naive use of massive web corpora drowns niche, high-quality sources; use weighted sampling or curriculum.
- Stability at scale: large learning rates, poor initialization, or missing warmup cause divergence. Use conservative warmup and smaller peak LR when scaling up.
- Numerical issues: fp16 can produce NaNs—use dynamic loss scaling and careful ops to avoid instability.
- Overfitting to training artifacts: models memorize training data; deduplicate and remove near-duplicates to reduce leakage and memorization.
- Compute inefficiency: mismatched model/data scaling wastes compute—follow scaling law guidance to proportionally increase model/data/compute.
- Checkpoint management: enormous storage; implement pruning, merging, and smart checkpointing (keep only necessary).
- Safety and privacy: pretraining corpora may contain copyrighted, private, or toxic content—apply filters and redaction.

Practical mitigations:

- Use deduplication pipelines (exact & fuzzy), quality scoring, and domain-balancing.
- Warmup schedulers, gradient clipping, and checkpoint averaging improve stability.
- Monitor memorization risk by checking for verbatim training-extractable sequences and redact when required.

### Interview-ready insights

- Why objective choice matters:
    - Autoregressive (GPT) excels at generation and scaling; masked (BERT) yields strong bidirectional representations better for classification; seq2seq (T5) balances both and is convenient for conditional tasks.
- Data sampling strategy is an instrument: the same model can emphasize coding, math, or conversational skills by reweighting domain data during pretraining.
- Scaling laws: adding parameters without enough data yields diminishing returns; balance parameters, data, and compute for efficiency.
- Stability tricks: LR warmup and decay, AdamW, weight decay, layer-wise lr decay, gradient clipping, and FP16 with dynamic loss scaling are standard for stable large-scale runs.
- Efficiency: parameter-efficient methods (Mixture-of-Experts, sparsely activated layers) or MoE reduce inference cost for large capacity but complicate training & routing.
- Reproducibility: logging dataset snapshots, random seeds, and pipeline versions is essential for auditability and debugging.
- Memorization & privacy: LLMs memorize; use membership testing, deduplication, and data governance to reduce privacy risk.
- Pretraining-to-product mapping: pretraining is expensive; prefer modular pipelines (foundation model + RAG + adapters/LoRA) to adapt models without retraining from scratch.

### Practice exercises

Exercise 1 — Mini pretraining from scratch (toy)

- Task: Train a small transformer (1–3 layers, d_model=128) on a small corpus (e.g., Project Gutenberg chapter) using contiguous token blocks of length 128 for next-token prediction. Track loss and sample generated text after every epoch.
- Hints: use AdamW, lr ~ 1e-4, batch_size 8, warmup 100 steps, gradient clipping 1.0, fp16 optional.

Exercise 2 — Data-mix sampling experiment

- Task: Build two small sources (news-like and code-like). Pretrain two identical small models: (A) with uniform sampling, (B) with upsampled code. Evaluate perplexity on held-out news and code dev sets to see trade-off.
- Hints: compute per-source dev perplexities and plot the tradeoff curve.

Exercise 3 — Checkpoint averaging benefit

- Task: Save model checkpoints every N steps near the end of training (last 5 checkpoints). Evaluate single checkpoints vs their average on dev perplexity.
- Hints: implement simple parameter-wise averaging for k checkpoints.

Exercise 4 — Tokenizer and vocab ablation

- Task: Train two tokenizers: byte-pair (BPE) with vocab 30k vs 50k on same corpus. Pretrain a small model with each tokenizer and compare tokenization length distributions, training speed, and final perplexity.
- Hints: measure average tokens per sentence and effect on compute.

Exercise 5 — Stability debugging

- Task: Intentionally remove warmup or set lr 10x too large and observe training divergence (NaNs or exploding loss). Add warmup and gradient clipping back and demonstrate recovered stability.
- Hints: log gradient norms and loss scale to find collapse point.

Exercise 6 — Memorization test

- Task: Insert a unique synthetic long string into the training corpus. After training, prompt the model with the prefix and measure whether it regenerates the string verbatim (measures memorization risk).
- Hints: use multiple different prefixes and compute exact-match retrieval rate.

---

## Computational challenges of training large language models (LLMs)

### Direct definition

Computational challenges of training large language models (LLMs) are the hardware, software, algorithmic, and operational barriers that make training at scale expensive, slow, fragile, and resource-hungry — including massive FLOPs, memory pressure for parameters and activations, network/IO bottlenecks in distributed training, numerical instability, costly data pipelines, and operational (energy/cost) constraints.

### Key challenges

- Compute (FLOPs): Transformer training cost grows roughly with model size, sequence length, and dataset size; large models require petaflop-days of compute and many GPU/TPU-hours to converge.
- Memory (parameters + activations): Storing parameters, gradients, and per‑token activations exceeds a single device’s RAM for multi-billion-parameter models, forcing model/data sharding or activation recomputation (checkpointing) to fit in memory.
- Communication and bandwidth: Model- and pipeline-parallel training require high-throughput, low-latency interconnects; cross‑device communication (all-reduce, scatter/gather, KV exchange) becomes the dominant overhead as GPU compute scales.
- Data pipeline throughput: Feeding accelerators requires fast, scalable tokenization, sharding, prefetching, and I/O; starvation of accelerators due to slow data pipelines wastes expensive compute.
- Numerical stability and optimization: Large-batch training, FP16/bfloat16 mixed precision, and optimizer state management (AdamW moments) can cause NaNs, instability, or suboptimal convergence without warmup, clipping, or loss scaling.
- Cost, energy, and carbon: Wall-clock time and electricity costs drive engineering trade-offs (smaller batches, quantization, or model-efficiency techniques) and influence feasibility for organizations.

(References used above: general and practical challenge reports and bottleneck analyses.)

### Complexity formulas and resource models

- Per-layer self-attention compute cost per forward pass (per sequence of length T and model dim d):

```
O_attn ≈ O(T^2 * d)    # pairwise attention matrix multiply
```

- FFN (position-wise) cost per layer:

```
O_ffn ≈ O(T * d^2)     # two linear layers per token
```

- Total per-step compute (L layers):

```
FLOPs_per_step ≈ L * (c1 * T^2 * d + c2 * T * d^2)   # c1,c2 constants for matmuls/GEMMs
```

- Memory footprint (approx for training with batch B, seq T):

```
Mem ≈ Params_size + B * T * d * 4 bytes (activations) + Optimizer_states (≈ 2-3x params for Adam)
```

- Throughput tradeoff (idealized):

```
tokens_per_second ∝ (num_devices * device_FLOPS) / (FLOPs_per_token + comm_overhead)
```

Use these relations to reason about where to optimize (reduce T, d, L; change precision; reduce comm overhead).

### Practical mitigations and engineering patterns

1. Precision and compression
    - Mixed precision (FP16/bfloat16) with dynamic loss scaling reduces memory and increases throughput; combine with activation compression or quantization-aware techniques for checkpoints.
2. Memory-reduction tricks
    - Activation checkpointing (recompute activations on backward pass), gradient checkpointing, and offloading (stage activations to CPU/NVMe) trade compute for memory.
    - Parameter sharding (ZeRO stage 1/2/3) splits optimizer states, gradients, and parameters across devices to reduce per‑device memory footprint.
3. Parallelism strategies
    - Data parallelism: replicate model; scale with all-reduce for gradients (simple but memory-limited).
    - Tensor (operator) parallelism: slice large parameter matrices across devices to fit model layers.
    - Pipeline (layer) parallelism: partition layers across devices and stream micro‑batches to keep devices busy.
    - Hybrid approaches (tensor + pipeline + data) combine advantages at scale; though complexity and communication overhead grow with hybridization.
4. Communication engineering
    - Overlap compute and communication, use high-bandwidth interconnects (InfiniBand/NVLink), use fused all-reduce kernels, and tune micro-batch sizes to amortize communication.
5. Optimization recipes
    - Large-batch optimizers: use AdamW with careful hyperparameters, LR warmup, cosine decay, gradient clipping, and possible LAMB/Adafactor variants when memory matters.
6. Efficient model designs
    - Sparse / Mixture of Experts (MoE) to increase capacity without linear inference cost; efficient attention variants (local, sparse, Performer, Linformer) reduce T^2 costs for long context.
7. Data and pipeline engineering
    - Sharded, pre-tokenized datasets, streaming token blocks, deterministic seeding, deduplication, and balanced sampling (data weighting or curriculum) prevent I/O bottlenecks and training bias.
8. Cost & carbon management
    - Use spot instances, mixed clouds, and scheduling; checkpointing and early stopping; model distillation to create cheaper student models for inference.

### Code sketches

Activation checkpointing (PyTorch checkpoint API)

```python
import torch
from torch.utils.checkpoint import checkpoint

def block_forward(x, block):
    return block(x)

# inside model forward
x = input
for block in blocks:
    x = checkpoint(lambda t: block_forward(t, block), x)
# saves memory by recomputing block during backward
```

Simple ZeRO-like param sharding idea (conceptual)

```
# Use DeepSpeed or FairScale in practice. Conceptual:
# - Partition optimizer states and parameters across ranks
# - On forward: each rank has local param shard, compute local outputs
# - On backward: reduce grads across ranks only for shards you own
```

Overlap compute and communication (pseudo)

```python
# schedule: compute matmuls -> async all_reduce grads for earlier layers while computing backward for later layers
# use torch.distributed.all_reduce(..., async_op=True) and later op.wait()
```

Use battle-tested libraries: DeepSpeed, Megatron-LM, FairScale, Colossal-AI, TorchRec for production-scale runs.

### Diagnostics, visualization, and monitoring

- Measure and plot:
    - GPU utilization, memory usage, PCIe/NIC bandwidth, and DMA stalls.
    - Per-step time breakdown: forward, backward, optimizer step, communication.
    - Tokens/sec and loss vs steps curves; plateaus indicate underprovisioned data or poor LR scheduling.
    - Gradient and parameter norm per layer for exploding/vanishing signals.
- Tools:
    - NVIDIA Nsight/torch.cuda.profiler, nvtop, Prometheus+Grafana for infra metrics, and framework profilers (PyTorch Profiler).
- Practical check: if GPU utilization < 60% and CPU IO is high, the bottleneck is data pipeline; if utilization high but tokens/sec plateaus while all-reduce time increases with device count, communication dominates.

### Common pitfalls & tips

- Blind scaling: Increasing batch size or model size without proportional data or LR schedule adjustments leads to poor convergence or wasted compute.
- Ignoring comm overhead: naive parallelism often hits network bottlenecks; measure collective op costs and tune shard sizes and micro-batch sizes.
- Stability with FP16: without dynamic loss scaling and careful ops, FP16 can produce NaNs; test warmup schedules and gradient clipping early.
- Dataset quality vs quantity: more tokens help, but noisy/duplicated data causes memorization and harms generalization — deduplicate and score data quality.
- Checkpoint/IO strain: many checkpoints and heavy I/O can saturate storage; use incremental checkpoints and asynchronous upload.
- Underestimating cost: budget and energy impacts affect feasibility — plan for early stopping, experiment budgets, and distillation for cheaper inference.

### Interview-ready insights

- Bottleneck triad: compute (FLOPs), memory (parameters+activations), and communication (all‑reduce / KV passing) — optimize whichever dominates in your scale regime and profile to find it.
- Choose parallelism to match constraints: if memory is the limiter, use ZeRO stage; if single-layer matmuls exceed device dims, use tensor parallelism; if billions of layers, pipeline parallelism helps.
- Mixed precision + gradient checkpointing + ZeRO is the de facto recipe for fitting huge models on limited hardware; production stacks use DeepSpeed/Megatron for these primitives.
- Communication strategy matters: overlap, fused collectives, and high-bandwidth interconnects often give more gains than adding more GPUs.
- Efficiency alternatives: MoE, efficient attention, and distillation change the compute/quality frontier and are worth considering before naïve parameter scaling.

### Practice exercises

1. Activation checkpointing experiment
    - Train a small Transformer with and without checkpointing; measure peak GPU memory and training throughput; compare wall-clock to a target loss.
2. Data-pipeline profiling
    - Build a tokenized streaming loader for a text corpus; measure CPU/GPU utilization and tokens/sec as you vary num_workers, prefetch, and batch sizing.
3. Communication scaling study (small cluster)
    - Simulate data-parallel training across 2, 4, 8 GPUs; measure per-step time and all-reduce time, plot scaling efficiency, and show when adding GPUs yields diminishing returns.
4. ZeRO vs baseline memory comparison (use DeepSpeed)
    - Configure ZeRO stage 0/1/2/3 on a modest model and report per-device memory, max batch size, and tokens/sec.
5. FP16 stability troubleshooting
    - Break a training run by enabling FP16 without dynamic loss scaling, observe NaNs/divergence, then fix with grad scaling and mixed-precision autocast.

---

## Efficient multi-GPU compute strategies

### Direct summary

Efficient multi-GPU training for large models balances three bottlenecks: memory (parameters + activations + optimizer state), compute (FLOPs per token/step), and communication (synchronizing gradients, exchanging activations or KV caches). The practical strategies are: choose the right parallelism (data, tensor, pipeline, or hybrid), reduce memory via sharding/checkpointing/offload/precision, overlap compute and comms, and use engineering practices (micro-batching, fused collectives, profiler-driven tuning). Below are the actionable techniques, math/complexity intuition, code patterns, pitfalls, and exercises.

### 1. Core parallelism strategies

- Data Parallelism (DP)
    - What: replicate full model on each GPU; each GPU processes a different mini-batch and all-reduces gradients.
    - Pros: simple to implement (DDP), robust scaling for models that fit on a single device.
    - Cons: memory duplicates model/optimizer per GPU; all-reduce communication cost grows with device count.
    - Best when: model fits on a device and you want simple horizontal scaling.
- Tensor (Operator) Parallelism (TP)
    - What: split individual large tensors (weight matrices) across GPUs so single matrix multiplies are sharded (row/column splits).
    - Pros: enables single-layer compute to be distributed (useful for very wide layers).
    - Cons: requires careful implementation (synchronization around matmuls, pipelined collectives).
    - Best when: a layer’s weights exceed single-device capacity or GEMM ops dominate and must be split.
- Pipeline Parallelism (PP)
    - What: partition the model by layers and place different layer groups on different GPUs; stream micro‑batches (micro-batching) through the pipeline.
    - Pros: reduces per-device memory by distributing layers; keeps large models trainable across devices.
    - Cons: pipeline bubbles (utilization loss), requires micro-batch tuning and pipeline schedulers.
    - Best when: model depth is large and layers can be grouped sensibly.
- ZeRO / Parameter Sharding
    - What: shard optimizer state, gradients, and optionally parameters across data-parallel ranks (stage 1/2/3).
    - Pros: linear reduction in per-device memory (Stage 3 shards parameters too), often used with DP to scale large models.
    - Cons: introduces communication during forward/backward to gather/shard parameters; complexity handled by libraries.
    - Best when: memory is the limiting factor but you still want data-parallel semantics.
- Mixtures and Hybrids
    - What: combine DP + TP, DP + PP, or DP + TP + PP to match model size and cluster topology.
    - Pros: realistic at scale — each technique addresses different constraints.
    - Cons: complexity skyrockets; scheduling and comm tuning crucial.

### 2. Memory reduction patterns (trade compute for memory)

- Mixed precision (FP16 / bfloat16)
    - Reduce memory and increase throughput; use dynamic loss scaling to avoid underflows/NaNs.
    - Practical: use AMP (torch.cuda.amp) or framework-level bfloat16 support.
- Activation checkpointing (recompute)
    - Save memory by not storing intermediate activations for some layers; recompute them during backward pass.
    - Tradeoff: extra forward compute for large memory savings.
    - Use-case: deep models where activations dominate memory.
- Gradient / optimizer state sharding (ZeRO)
    - Shard optimizer states and gradients across ranks — biggest win for Adam-like optimizers with 2–3× parameter state.
    - Stage 3 shards parameters as well: near-linear per-GPU memory reduction.
- Parameter offloading (CPU/NVMe)
    - Offload optimizer states or rarely-used parameters to CPU/NVMe when GPU RAM is insufficient.
    - Tradeoff: higher latency and IO overhead; useful to fit very large models with limited GPU memory.
- Quantized or Low-Precision Checkpoints
    - Store checkpoints in int8 or float16; dequantize on load. Useful for checkpoint storage and slightly for memory if carefully integrated.

### 3. Communication engineering (make it fast and overlapped)

- All-reduce tuning and fused collectives
    - Use fused all-reduce kernels and NCCL tuning for large contiguous buffers rather than many small buffers.
    - Aggregate gradients into big tensors before all-reduce.
- Overlap compute and communication
    - Start async all-reduce for early-layer gradients while computing backward for later layers.
    - Use non-blocking collectives and schedule waits after compute is done.
- Topology-aware partitioning
    - Map tensor/pipeline partitions to GPUs so that heavy comm happens over NVLink or within the same host; minimize cross-rack traffic for fine-grained ops.
- Reduce communication frequency
    - Gradient accumulation to increase effective batch size and amortize all-reduce cost.
    - Gradient compression (sparsification or quantization) for extreme network-limited clusters (with careful error correction).

### 4. Scheduling, micro-batching, and utilization

- Micro-batching for pipeline parallelism
    - Split global batch into micro-batches and overlap pipeline stages; more micro-batches reduces pipeline bubble but increases memory.
    - Choose micro-batch count that balances utilization and memory constraints.
- Batch size vs learning dynamics
    - Larger effective batch sizes need LR scaling and warmup; use linear scaling rules or adaptive optimizers with tuned warmup.
- Layer grouping for pipelines
    - Group layers to balance compute across devices; aim for similar FLOPs per stage.
    - Use profiling to compute per-layer FLOPs and latency, then partition to minimize imbalance.

### 5. Practical software stacks and patterns (how to implement)

- DistributedTraining basics (PyTorch DDP)
    - Initialize torch.distributed, wrap model in DistributedDataParallel, use DistributedSampler, and perform per-step all-reduce via DDP.
- Use battle-tested libraries for scale
    - DeepSpeed, FairScale, Megatron-LM, Colossal-AI and similar provide ZeRO, TP, PP, and fused kernels. They handle complex bookkeeping and often expose simple config files.
- Checkpointing & fault tolerance
    - Save sharded checkpoints (reduce IO sizes); support resuming with partial loads.
    - Use frequent lightweight checkpoints (optimizer states optional) during long runs.
- Profiling and telemetry
    - Profile with PyTorch Profiler / Nsight to find hot spots (comms vs compute vs IO).
    - Log tokens/sec, GPU utilization, all-reduce time, memory high-water marks.

### 6. Key formulas and complexity intuition

- Self-attention cost per layer: O(T^2 * d)
- FFN cost per layer: O(T * d^2)
- Memory for activations ≈ B * T * d * 4 bytes (FP32) (scale down with FP16)
- Per-step FLOPs ≈ L * (c1*T^2*d + c2*T*d^2); adding layers L or sequence length T increases compute strongly.
- Effective communication cost for DP: O( log(N) * Params ) for all-reduce (dependent on algorithm and implementation). Sharding reduces per-GPU param memory but shifts some cost to gather/scatter on demand.

Use these to decide: if T is huge, optimize attention (sparse or chunked); if d is huge, consider TP; if params exceed GPU memory, ZeRO or offload.

### 7. Common pitfalls and debugging tips

- GPU underutilization
    - Symptom: low GPU utilization, high CPU or IO wait. Fix: pre-tokenize, increase num_workers, enable pinned memory, pipeline larger micro-batches, or tune data pipeline.
- Communication overhead dominates
    - Symptom: all-reduce time grows with GPU count. Fix: aggregate gradients, increase compute per all-reduce (larger batches), use fused ops, choose topology-aware mapping.
- NaNs with FP16
    - Use dynamic loss scaling; check problematic ops (softmax/logs/divisions) and cast to float32 where needed.
- Pipeline stalls (bubbles)
    - Increase micro-batches, better layer grouping, or reduce pipeline depth per stage.
- Memory OOM during backward
    - Use activation checkpointing, reduce per-GPU batch size, enable ZeRO, or offload.
- Incorrect past_key/past_key_values handling during inference across shards
    - Carefully manage KV caches per shard and ensure correct concatenation order.
- Reproducibility across multi-node
    - Set deterministic seeds per rank, fix dataloader shuffling seeds with offsets, and ensure consistent config across nodes.

### 8. Practical code snippets (patterns)

Basic DDP skeleton (PyTorch)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=batch_per_gpu, sampler=sampler, num_workers=4, pin_memory=True)

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        inputs = batch.to(device)
        loss = model(inputs).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

Activation checkpointing (PyTorch)

```python
from torch.utils.checkpoint import checkpoint

def forward_block(x):
    x = block_layer1(x)
    x = block_layer2(x)
    return x

# inside model forward
x = input
x = checkpoint(forward_block, x)  # recompute inside backward
```

Manual async overlap pseudo (conceptual)

```python
# in backward: after computing grad for layer L, kick off async all-reduce for grads of earlier layers
handle = dist.all_reduce_async(tensor)
# continue computing other layers' backward
handle.wait()
```

(Use library utilities or fused ops where possible.)

Using ZeRO-like sharding (conceptual API)

```python
# In practice use DeepSpeed or FairScale:
# model = deepspeed.initialize(model=model, config_params=ds_config)
# deepspeed handles parameter partitioning, optimizer sharding etc.
```

### 9. Interview-ready bullets

- Memory-first decision rule: if a model doesn’t fit on one GPU, use ZeRO stage 3 or TP to shard parameters; if per-layer matmuls exceed device dims, use TP; if model depth is very large, use pipeline stages.
- Overlap is king: overlapping communication with compute yields larger throughput gains than adding GPUs in many regimes.
- Profiling drives choices: always profile per-step forward/backward/optimizer/all-reduce to identify the dominant bottleneck and then choose the parallelism/optimization targeted at that bottleneck.
- Use existing engineering stacks: DeepSpeed/FairScale/Megatron/Colossal-AI implement complex strategies and are essential for production-scale efficiency.

### 10. Practice exercises

Exercise A — Activation checkpointing impact

- Run a small Transformer (e.g., 12 layers, d=512) with/without checkpointing and measure peak GPU memory and tokens/sec. Plot memory vs throughput tradeoff.

Exercise B — ZeRO stage comparison (via library)

- Configure ZeRO stage 1/2/3 on a toy model and record max batch size per GPU before OOM and tokens/sec. Observe per-stage memory savings.

Exercise C — Micro-batch pipeline tuning

- Build a pipeline-parallel split for a medium-depth model across 2 GPUs. Sweep micro-batch counts {1,2,4,8} and measure utilization and latency to find sweet spot.

Exercise D — Communication profiling

- Instrument a small DP training run across 2, 4 GPUs. Measure all-reduce time fraction vs compute time and show how gradient accumulation reduces communication overhead.

---

## Scaling laws and compute optimal models

### Direct definition

Scaling laws describe empirical power-law relationships between model size, dataset size, compute, and performance (typically loss). Compute-optimal models are model+data configurations that, for a fixed compute budget, minimize expected loss by choosing the best trade-off between number of parameters and number of training tokens.

### Concept intuition

- What it is: Instead of treating model size and dataset size independently, scaling laws tell you how performance improves as you scale parameters (N), tokens (D), or compute (C). They let you predict returns from spending compute on a bigger model versus more data or longer training.
- Why it matters: Training huge LLMs is costly; scaling laws guide resource allocation (how big to make the model and how many tokens to train on) so you get the most performance per dollar/TFLOP.
- Key insight: For many model families and tasks, loss L empirically follows a power-law form with respect to N and D; at fixed compute C there’s an optimal N* and D* (the compute-optimal frontier, e.g., Chinchilla).
- Analogy: Given a fixed budget to build a bridge, you must choose how much steel (model params) versus surveying (data) to buy — scaling laws quantify that trade-off.

### Mathematical breakdown

Empirical scaling law (simplified)

```
L(N, D) ≈ A * N^{-α_N} + B * D^{-α_D} + L_inf
```

- L: validation loss (e.g., cross-entropy)
- N: number of parameters
- D: number of training tokens
- A, B, α_N, α_D, L_inf: fit constants (α are positive exponents)

Compute (approx) for autoregressive training

```
C ≈ k * N * D       # compute proportional to params × tokens (k depends on model architecture)
```

Compute-optimal trade-off (intuitively)

- Given C fixed, choose N and D to minimize L subject to C ≈ k N D.
- Substitute D ≈ C / (k N) into L(N,D) and minimize over N:

```
L(N) ≈ A * N^{-α_N} + B * (C / (k N))^{-α_D} + L_inf
     = A * N^{-α_N} + B' * N^{α_D} * C^{-α_D} + L_inf
# find N* solving dL/dN = 0 (closed form when exponents known)
```

Resulting compute-optimal scaling (power-law relationship)

```
N* ∝ C^{α_D / (α_N + α_D)}
D* ∝ C^{α_N / (α_N + α_D)}
```

- Interpretation: exponents α_N and α_D determine how to split compute between model size and data. If data yields larger exponent (α_D bigger), favor more data; otherwise favor larger N.

Chinchilla-style rule (empirical example)

- Early practices trained very large N on relatively small D (e.g., GPT-3). Chinchilla showed that smaller models trained on more tokens gave better loss per compute.
- Rough empirical outcome: optimal tokens per parameter ratio is roughly constant; e.g., Chinchilla suggested ~20 tokens per parameter (varies by fit and family).

Marginal returns and diminishing improvements

```
ΔL / ΔC ∝ C^{-γ}   # diminishing returns; γ > 0
```

- As compute grows, each extra compute unit buys less improvement.

Notes:

- Constants and exponents must be fitted per model family, objective, tokenization, and data mix.
- Scaling laws are empirical; out-of-distribution factors (architecture changes, data quality, task shifts) alter constants.

### Code & practical application

Fit scaling law to small experiments (toy)

```python
# pip install numpy scipy
import numpy as np
from scipy.optimize import curve_fit

# example arrays from experiments (replace with real runs)
Ns = np.array([1e6, 5e6, 2e7, 8e7])       # parameters
Ds = np.array([1e7, 5e7, 2e8, 8e8])       # tokens
losses = np.array([4.2, 3.5, 3.0, 2.8])   # observed val loss

def model(vars, A, alphaN, B, alphaD, L_inf, kN, kD):
    N, D = vars
    return A * N**(-alphaN) + B * D**(-alphaD) + L_inf

popt, _ = curve_fit(model, (Ns, Ds), losses, p0=(1.0, 0.2, 1.0, 0.2, 1.0))
A, alphaN, B, alphaD, L_inf = popt
print("fitted exponents:", alphaN, alphaD)
```

Compute-optimal N* given compute budget C (analytical approximation)

```python
# given fitted alphaN, alphaD, A, B, k (approx)
C = 1e25  # example TFLOP-like unit
k = 1.0   # constant absorbed into C units
# derived proportionality (from math section)
ratio = alphaD / (alphaN + alphaD)
N_star = (C ** ratio)  # times constant factor from A,B,k omitted here; solve full eqn for exact value
```

Practical implementation: solve for N minimizing L(N) numerically using the fitted model and constraint D = C/(kN).

Planning experiments with cheap proxies

- Train smaller models (scaled N and D) and fit scaling law to extrapolate to larger C; this avoids training huge models end-to-end.

Use-cases in practice

- Budget planning: choose model size vs tokens to stay on compute-optimal frontier.
- Data acquisition strategy: determine whether to spend on more data collection/curation versus larger model capacity.
- Distillation scheduling: given compute-optimal teacher sizes, plan student sizes and distillation budgets to maximize per-dollar performance.

### Visualization / Geometry intuition

- Loss surfaces across (N, D):
    - Plot heatmap of loss L(N,D) for experiments; compute-optimal frontier is the ridge of minimal loss for each compute-sublevel.
- Frontier curve:
    - For fixed C, plot L(N) vs N; N* is the minimum; repeat for many C to see how N* grows.
- Marginal improvement plots:
    - Plot dL/dlogN and dL/dlogD to see where marginal returns balance.
- Token-per-parameter constant:
    - Plot D/N across fitted experiments — compute-optimal configurations tend to cluster around a constant ratio.

Quick plotting pseudo:

```python
import matplotlib.pyplot as plt
# L_grid computed from model for grid of N,D
plt.contourf(logN_grid, logD_grid, L_grid)
plt.plot(N_star_curve, D_star_curve, color='white')  # compute-optimal frontier
plt.xlabel("log N"); plt.ylabel("log D"); plt.colorbar(label="loss")
```

### Common pitfalls & tips

- Misapplying exponents: scaling exponents are model-family and dataset-dependent; copy-pasting Chinchilla numbers without fitting is risky.
- Ignoring data quality: scaling laws treat tokens as homogeneous; low-quality or duplicated tokens reduce effective α_D and break predictions.
- Hardware and software differences: compute C measured in FLOPs assumes ideal implementation; real wall-clock cost and engineering overhead vary.
- Overfitting to small-proxy fits: small-model behavior may not perfectly extrapolate due to emergent phenomena in large models.
- Constant factors matter: analytical proportionalities omit multiplicative constants (A, B, k) that shift N*; fitting is necessary for practical planning.
- Task-specific scaling: language modeling loss is one metric; downstream task performance (e.g., reasoning) may scale differently.
- Ignoring efficiency techniques: MoE, improved architectures, or data augmentation can change the shape of scaling curves.

Tips:

- Fit scaling laws using many data points spanning orders of magnitude in N and D.
- Control for tokenizer, compute measurement, and data mix when comparing runs.
- Use nested experiments: small grid to fit exponents → plan larger budget runs using extrapolation → validate with a mid-scale run.

### Interview-ready insights

- Core message: Given fixed compute, there's an optimal split between parameters and tokens; blindly scaling only parameters (or only data) is suboptimal.
- Chinchilla lesson: Best-performing models per FLOP trained smaller models on more data rather than the reverse; tokens-per-parameter is a key planning metric.
- Practical rule: Fit empirical scaling laws on your model family and data mix before committing large budgets. Use small-scale probes to predict large-scale behavior.
- Why it works: Power-law scaling emerges because model capacity and data both reduce irreducible error components; the exponents quantify marginal returns.
- When to deviate: Use MoE or architectural changes (sparsity) if you want more capacity for same compute; these change the effective scaling exponents.
- Cost vs performance: Scaling laws let you compute marginal cost per unit loss reduction, enabling rational budget allocation and comparison with alternatives (data curation, architecture search, distillation).

### Practice exercises

Exercise 1 — Fit scaling law from toy runs

- Task: Run a grid of small training runs varying N and D (3–5 sizes each), record validation loss, fit the two-term power-law model, and report α_N and α_D and their confidence intervals.
- Hint: Log-transform variables and perform linear regression on log-space to get initial estimates.

Exercise 2 — Find compute-optimal split numerically

- Task: Given fitted L(N,D) and compute formula C = k N D, numerically solve for N* and D* that minimize L under the constraint C. Plot L(N) for that C and mark the minimum.
- Hint: use scipy.optimize.minimize_scalar over log(N).

Exercise 3 — Data-quality ablation

- Task: Create two corpora: high-quality and noisy duplicates. Fit scaling laws separately and compare α_D. Observe how noisy data lowers data exponent and shifts N* toward larger models.
- Hint: simulate noisy data by injecting repeated boilerplate or random tokens.

Exercise 4 — Extrapolation validation

- Task: Fit scaling laws on small models, predict loss for a mid-size model, train that model, and measure prediction error. Report extrapolation accuracy and sources of mismatch.
- Hint: use cross-validation by leaving out one mid-scale point to test predictions.

Exercise 5 — Compute vs wall-clock analysis

- Task: For a set of configurations (N, D) compute estimated FLOPs C and convert to expected wall-clock given device TFLOPS and measured efficiency factor (e.g., 30–60%). Compare cost estimates for candidate configurations and choose compute-optimal in dollar terms.
- Hint: factor in I/O, communication overhead, and scheduling inefficiencies as multiplicative efficiency scalars.

---

## Pre‑training for domain adaptation

### Direct definition

Pre‑training for domain adaptation is the practice of continuing or tailoring a foundation model’s unsupervised pretraining on a curated, domain‑specific corpus (or modifying its tokenizer/data recipe) so the model’s internal representations and next‑token distribution capture domain terminology, style, and facts. The goal is to produce a foundation that (a) improves downstream domain tasks with less supervised data and (b) preserves general capabilities where needed.

### Concept intuition

- What it does: take a broadly pretrained LLM and expose it to lots of domain text (medical notes, legal filings, financial reports, customer support logs, code, scientific papers) so rare domain tokens, phrases, and co‑occurrence patterns become first‑class signals in its parameterization.
- Why it matters: domain data often contains specialized terms, idiosyncratic phrasing, and structured elements (tables, code) that general pretraining underrepresents. Domain continual pretraining (DACP / continued pretraining) makes the model more fluent and less prone to hallucinate in that domain.
- Two flavors:
    - Continued (continued unsupervised) pretraining: keep the same LM objective (causal or masked) and train on domain corpus for some steps. Lightweight and preserves pretraining inductive biases.
    - From-scratch domain pretraining or heavy adapter init: redesign tokenizer, vocab, or initialize embeddings differently and train longer (costly, higher risk but sometimes necessary when domain tokenization is poor).
- Complementary steps: tokenizer/vocab tweaks, mixed-source sampling, curriculum schedules, and then lightweight supervised fine-tuning or LoRA/adapters for end tasks.

### Mathematical breakdown (core formulas and practical choices)

Continued causal pretraining objective (same as base LM):

```
Loss = - sum_{t=1..T} log p_theta(x_t | x_{1:t-1})
```

Mixed-source sampling (to preserve general capability while adapting):

- Let S_base and S_domain be source corpora. Sample documents with probability:

```
p(sample from domain) = alpha
p(sample from base) = 1 - alpha
```

- Effective dataset seen by model is a mixture; alpha controls specialization vs generality.

Importance weighting / upsampling:

```
If doc d from source s has weight w_s:
  p(d) ∝ w_s / |S_s|
```

- Higher w_domain raises exposure to domain patterns.

Tokenizer/vocab change (if rebaking):

- New vocab reduces average token length for domain terms, reducing sequence length and improving modeling efficiency. If vocab change performed, you must reinitialize embeddings E_new and either:
    - re-learn embeddings from scratch
    - map old embeddings to new tokens via heuristics (e.g., mean of subtoken embeddings).

Catastrophic forgetting risk (informal)

- Training only on domain (alpha = 1) shifts parameter posterior:

```
theta_new ≈ argmin_theta E_{d~domain}[ -log p_theta(d) ]  # may diverge from theta_base
```

- To preserve general skill, use mixed sampling, smaller LR, or constraints (L2 to theta_base, KL regularization).

Regularization via KL constraint (stable continual pretraining)

```
min_theta E_{x~domain}[ -log p_theta(x) ] + λ * D_KL( p_theta || p_theta0 )  # θ0 initial weights
```

- Practically implemented via distillation loss or small learning rate and mixed data.

Variables:

- theta: model params; θ0: base weights; alpha: domain sampling weight; λ: regularization coefficient.

### Code & practical application (recipes and runnable sketches)

Prereqs:

```bash
pip install transformers datasets accelerate sentence-transformers faiss-cpu
```

A. Simple continued pretraining (causal LM) on domain corpus

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import math

base_model = "gpt2"          # use a suitable base (instruction / domain-agnostic)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# load domain texts as a Dataset of strings
ds = load_dataset("text", data_files={"train":"domain_texts.txt"})["train"]

# tokenization and chunking into blocks
def tokenize_and_chunk(examples, block_size=1024):
    tokens = tokenizer(examples["text"], truncation=False, add_special_tokens=False)
    all_ids = sum(tokens["input_ids"], [])
    blocks = [all_ids[i:i+block_size] for i in range(0, len(all_ids)-block_size+1, block_size)]
    return {"input_ids": blocks}

blocks = ds.map(lambda ex: tokenize_and_chunk(ex, block_size=512), batched=True, remove_columns=["text"])
blocks = blocks["train"]

# training args: small continued pretrain
args = TrainingArguments(
    output_dir="domain-continue",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,    # small LR for stability
    num_train_epochs=1,
    fp16=True,
    logging_steps=50,
    save_steps=500,
)

# simple data collator for LM
from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(model=model, args=args, train_dataset=blocks, data_collator=collator)
trainer.train()
```

B. Mixed-source sampling (to preserve general skills)

- Prepare two datasets: base_blocks and domain_blocks. Create an iterable dataset that samples from domain with probability alpha.
- Simple sampler sketch:

```python
import random
def mixed_generator(domain_blocks, base_blocks, alpha=0.3):
    while True:
        if random.random() < alpha:
            yield random.choice(domain_blocks)
        else:
            yield random.choice(base_blocks)
```

- Wrap generator into torch Dataset for Trainer.

C. Adapter / LoRA alternative (parameter-efficient)

- Instead of changing whole weights, keep base weights frozen and train tiny adapters (LoRA) on domain data. Hugging Face PEFT or peft library examples:

```bash
pip install peft
```

Then attach LoRA modules and train with low LR, small steps — much cheaper and avoids catastrophic forgetting.

D. Tokenizer update strategy (if necessary)

- If domain uses many terms that tokenize badly:
    1. Train SentencePiece/BPE on domain+sample base corpus.
    2. Rebuild tokenizer, compute mapping old_token -> new token ids (where possible).
    3. Initialize new embedding table E_new; for tokens that are subword merges of old tokens, initialize by averaging corresponding old embeddings.
    4. Continue pretraining (recommended: shorter LR, longer warmup).

E. Regularization: KL or distillation to base model

- Compute logits from base model θ0 and penalize divergence:

```python
# pseudo-loss
loss = ce_loss(p_theta, x) + lambda * KLDivLoss(logits_theta, logits_theta0)
```

- Practically compute logits from frozen base model and add small KL term to trainer.

### Visualization / Geometry intuition

- Embedding shift plot: select domain-specific tokens; plot their embedding vectors before and after domain pretraining (PCA/UMAP projection). Expect tokens to move and form domain clusters.
- Attention heatmaps change: cross-attention (or self-attention patterns) may show stronger intra-domain token dependencies after adaptation. Visualize attention matrices for same prompts pre/post adaptation.
- Loss and metric curves: monitor in-domain perplexity vs general-domain perplexity as training proceeds. Ideal path: in-domain loss drops while general loss holds stable (or degrades minimally).
- Parameter drift: plot L2 distance ||theta_t - theta0|| per layer to see where adaptation concentrates; heavy drift in embeddings and top layers is common.

Quick plotting snippet for embedding shift:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# E0, E1: numpy arrays [vocab_subset, d]
pca = PCA(n_components=2).fit_transform(np.vstack([E0, E1]))
plt.scatter(pca[:len(E0),0], pca[:len(E0),1], label="before")
plt.scatter(pca[len(E0):,0], pca[len(E0):,1], label="after")
plt.legend(); plt.show()
```

### Common pitfalls & tips

- Overfitting / catastrophic forgetting:
    - Pitfall: training only on domain drives model to lose general knowledge and fluency.
    - Mitigation: mixed-source sampling, small LR, short continued-pretrain steps, and regularization (KL to base). Monitor general-domain validation metrics.
- Tokenizer mismatch and embeddings:
    - Pitfall: poor tokenization of domain terms increases sequence length and hurts modeling efficiency. Re-tokenizing requires careful embedding reinitialization and more compute.
    - Mitigation: evaluate tokenization statistics (avg tokens per doc); only change tokenizer if benefit justifies cost.
- Data contamination and leakage:
    - Pitfall: accidentally including test or private production data in domain corpus causes leakage and privacy risk.
    - Mitigation: strict data lineage, deduplication, and held-out test sets; remove PII and copyrighted text as per policy.
- Data quality vs quantity:
    - Pitfall: flooding with noisy domain data (boilerplate, auto-generated content) reduces gains.
    - Mitigation: filter by quality signals (language model score, length thresholds, source reputation), deduplicate, and curate representative samples.
- Excessive training steps:
    - Pitfall: long continued pretraining with high alpha overfits; after small number of epochs the marginal gain often falls fast.
    - Mitigation: run small pilot runs to find diminishing returns; use early stopping on in-domain val set.
- Evaluation mismatch:
    - Pitfall: only measuring in-domain perplexity misses downstream task performance (e.g., extraction accuracy, summarization fidelity).
    - Mitigation: set downstream benchmarks (few-shot and fine-tuned tasks) to evaluate real utility.
- Compute & cost underestimation:
    - Pitfall: re-pretraining large models is expensive; LoRA/adapters or fine-tuning often cheaper and nearly as effective for many tasks.
    - Mitigation: compare ROI — small adapter tuning vs full continued pretraining.

### Interview-ready insights

- When to do continued pretraining:
    - Use continued pretraining when domain has large unlabeled corpora and you need improved base fluency/factuality in that domain, especially for many downstream tasks or for retrieval‑anchored RAG. If you only need one or two supervised tasks, adapters / LoRA or full fine-tuning may be more efficient.
- How to preserve general capabilities:
    - Keep a fraction of base data during continual pretraining, use small LR, short training schedules, or KL/distillation regularization. Monitor both in-domain and general validation metrics.
- Tokenization decision rule:
    - If domain tokens are frequent enough to change average tokenization length significantly (and that affects compute/seq_len), retrain tokenizer; otherwise, leave tokenizer unchanged and rely on embedding adaption.
- Practical recipe (starter template):
    1. Audit domain corpus: size, tokenization stats, duplication, PII, quality.
    2. Run a small continued-pretrain pilot (1–2 epochs on blocks) with alpha in {0.2,0.5,0.8}, monitor in-domain perplexity and general perplexity.
    3. If tokenization poor, consider new tokenizer + embedding init pilot.
    4. Compare full continued pretrain vs LoRA/adapters on downstream tasks; pick best cost/accuracy tradeoff.
    5. Apply RAG or retrieval post-adapt to ground facts if up-to-date knowledge needed.
- Cost-effective alternatives:
    - LoRA/adapters for parameter-efficient specialization; RAG for up-to-date facts; instruction fine-tuning on domain-labeled data for behavior shaping.

### Practice exercises

Exercise 1 — Pilot continued pretraining

- Task: Take a base model (e.g., t5-small or gpt2) and a domain corpus (~10k documents). Run continued pretraining for a small number of steps with mixed-source sampling (alpha ∈ {0.0,0.2,0.5}). Track in-domain perplexity and base validation perplexity. Report best alpha and training steps where marginal return drops.

Exercise 2 — Adapter vs continued-pretrain comparison

- Task: Train LoRA adapters on domain corpus (or task-labeled subset) and compare downstream performance (classification / extraction) against a small continued pretrain run. Measure GPU hours, final metric, and size of saved artifact.

Exercise 3 — Tokenizer ablation

- Task: Compute tokenization stats (avg tokens/doc, fraction of domain words split >1 subtoken) for base tokenizer. If > X% (choose 10%) of domain-specific phrases are split, retrain tokenizer on domain+small base mixture; initialize embeddings by averaging old embeddings. Fine-tune/continue pretraining and compare downstream metrics.

Exercise 4 — Regularized continual pretraining

- Task: Implement KL-regularized continual pretraining by computing frozen base logits for batches and adding λ * KL term to loss. Compare stability and generalization vs unregularized continued-pretrain.

Exercise 5 — RAG + domain pretrain hybrid

- Task: Build a RAG pipeline: (1) index domain docs with sentence embeddings, (2) continue-pretrain base LM on domain (short run), (3) generate answers for domain queries with and without retrieval; measure factuality and hallucination rate.

---