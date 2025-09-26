# GenAI_m2

## Instruction Fine-tuning

### Direct definition

Instruction fine-tuning is the supervised process of continuing to train a pretrained language model on a dataset of (instruction, input, response) examples so the model learns to follow human-style instructions and produce helpful, aligned outputs.

### Concept intuition

- What it does: converts a general pretrained LLM (which predicts next tokens from raw text) into a model that reliably follows explicit user instructions (e.g., "Summarize this", "Explain like I'm five", "Write code for X").
- Why it matters: base pretraining learns language patterns but not the mapping from an instruction + input to a desired action-oriented output; instruction fine-tuning teaches that mapping so the model is useful in interactive systems and aligned applications.
- Analogy: base pretraining is like learning a language by reading many books; instruction fine-tuning is guided apprenticeship — a teacher gives tasks and desired outputs so the student learns how to respond when asked.
- Variants: direct supervised instruction fine-tuning (SFT), iterative refinement with human feedback (RLHF uses a reward model and policy optimization on top of SFT), and multi-task instruction fine-tuning (training on many different instruction types to improve generalization).

### Mathematical breakdown

Supervised objective (cross-entropy language modeling on response tokens)

- Setup: each training example is (instruction I, optional input X, target response Y which is token sequence y1..yT).
- We concatenate prompt P = format(I, X) and let model generate tokens. We maximize log-likelihood of tokens in Y (teacher forcing).
- Loss (per example):

```python
# cross-entropy negative log-likelihood for a single example
L = - sum_{t=1..T} log p(y_t | P, y_{1:t-1}; theta)
```

- Where:
    - theta: model parameters
    - p(y_t | ...) : softmax probability output by model at timestep t
    - P: tokenized prompt containing instruction and input

Batch loss and optimization

```python
L_batch = (1/N) * sum_{i=1..N} L_i
theta <- theta - eta * grad_theta(L_batch)
```

- Where:
    - N: batch size
    - eta: learning rate
    - grad_theta: gradient computed by backprop

Label handling and causal masks

- For encoder-only or encoder-decoder models, target tokens are aligned differently (encoder-decoder uses decoder cross-entropy conditioned on encoder output). For causal LMs we ensure loss is computed only on response token positions, not the prompt context.

Regularization / stability

- Common additions: weight decay, gradient clipping, AdamW optimizer with betas, and learning-rate scheduling (warmup + cosine decay).

### Code & practical application

Goal: fine-tune a small causal LM (GPT-2 small or EleutherAI/gpt-neo-125M) with Hugging Face and PyTorch on an instruction dataset. The example uses supervised fine-tuning (SFT) only.

Minimal dataset format (JSONL)

- Each line:

```json
{"instruction":"Translate English to French", "input":"I love you.", "output":"Je t'aime."}
```

Data preparation (tokenize prompt and target; ensure loss only on target)

```python
# pip install transformers datasets accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

model_name = "gpt2"   # or a small causal LM
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Formatting function: create prompt and labels where prompt tokens are masked with -100
def format_example(example, max_length=512):
    instr = example.get("instruction","")
    inp = example.get("input","")
    out = example.get("output","")
    if inp:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    output_ids = tokenizer(out, return_tensors="pt").input_ids[0]

    input_ids = torch.cat([prompt_ids, output_ids], dim=0)
    # create labels: -100 for prompt part so loss is computed only on output tokens
    labels = input_ids.clone()
    labels[:prompt_ids.size(0)] = -100
    # attention mask
    attention_mask = torch.ones_like(input_ids)
    # trim/pad to max_length
    if input_ids.size(0) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
        attention_mask = attention_mask[-max_length:]
    else:
        pad_len = max_length - input_ids.size(0)
        input_ids = torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
        labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# Example usage with datasets
ds = load_dataset("json", data_files="instructions.jsonl", split="train")
ds = ds.map(lambda ex: format_example(ex), remove_columns=ds.column_names)
ds.set_format(type="torch")
```

Training loop (simple PyTorch / HF Trainer alternative)

```python
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

train_loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda x: {
    "input_ids": torch.stack([d["input_ids"] for d in x]),
    "labels": torch.stack([d["labels"] for d in x]),
    "attention_mask": torch.stack([d["attention_mask"] for d in x])
})

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 3
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} loss {loss.item():.4f}")
```

Inference (prompting the fine-tuned model)

```python
model.eval()
prompt = "### Instruction:\nSummarize the following text\n\n### Input:\nThe quick brown fox jumps over the lazy dog.\n\n### Response:\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generated = model.generate(input_ids, max_new_tokens=50, do_sample=True, top_k=50, top_p=0.95, temperature=0.8, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(generated[0][input_ids.size(1):], skip_special_tokens=True))
```

Notes:

- Use batching, gradient accumulation, and fp16 for larger models.
- For encoder-decoder (T5), format the instruction+input as encoder input and compute cross-entropy on decoder outputs.

### Visualization / Geometry

Embedding-space effect

- Before SFT: prompt embedding for "### Instruction: Translate" may lie near general language tokens; after SFT the model learns a specific manifold: prompts indicating instructions map the model into a region of parameter activations that produce instruction-following behavior.
- Visualize by projecting prompt prefixes (mean-pooled token embeddings or final hidden state for prompt) via t-SNE/UMAP before and after SFT to see clustering by instruction type.

Attention and response generation

- Inspect attention maps in decoder layers when producing the response; often you’ll see:
    - Strong causal attention to recent decoder tokens (autoregressive)
    - Cross-attention (in encoder-decoder) focusing on input segments that are relevant to the instruction.
- Practical plot: extract attention weights from model outputs (many HF models return attentions if configured) and plot with seaborn heatmap aligned to tokens.

Loss heatmap over tokens

- Plot per-token loss (by feeding prompt+response and computing logits vs labels) to see which tokens the model still struggles with after fine-tuning (helps find domain-specific vocabulary gaps).

Example snippet to get per-token loss:

```python
model.eval()
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=False)
    logits = out.logits  # shape [1, seq_len, vocab_size]
    # compute cross entropy per position
    import torch.nn.functional as F
    log_probs = F.log_softmax(logits, dim=-1)
    # gather log_probs for true tokens
    target = labels.clone()
    target[target==-100] = tokenizer.pad_token_id
    per_token_logprob = log_probs[0, torch.arange(logits.size(1)), target]
    per_token_loss = -per_token_logprob
```

### Common pitfalls & tips

- Masking prompts incorrectly: forgetting to set labels to -100 for prompt tokens will make the model try to predict the prompt, confusing learning. Always mask prompt positions for causal LMs.
- Catastrophic forgetting: aggressive LR or long SFT can make the model lose general language knowledge; mitigate with lower LR, few epochs, or mix in base-task data (replay).
- Overfitting small instruction datasets: use regularization, augmentation (paraphrase instructions), early stopping, and validation splits.
- Distribution mismatch at inference: training with a rigorous prompt format but receiving different prompts in production reduces performance — train on varied prompt phrasings.
- Tokenizer mismatch: ensure tokenizer.pad_token is set for causal models; otherwise generation and padding break.
- Privacy and safety: instruction datasets may contain toxic or private content — curate and filter training data and consider safety classifiers.

### Interview-ready insights

- Why SFT before RLHF? SFT gives a stable, instruction-following policy that RLHF can then refine with reward optimization; RLHF without SFT often destabilizes training.
- Loss choice: token-level cross-entropy is standard; however, sequence-level metrics are not directly optimized — RL or minimum-risk training can optimize sequence objectives.
- Prompt-format design matters: using canonical separators (Instruction / Input / Response) reduces ambiguity and improves generalization.
- Learning-rate strategy: small LR (1e-5 to 5e-5) for large pretrained models; larger LR will break pretrained representations.
- Compute-efficient tactics: LoRA / adapters / prefix tuning allow instruction fine-tuning by updating far fewer parameters and are practical for deployment.

### Practice exercises

Basic SFT (easy)

- Task: Fine-tune "gpt2-small" on a toy JSONL dataset of 100 instruction-response pairs (mix of translate, summarize, sentiment). Train 1–3 epochs and evaluate by hand with 10 prompts.
- Hint: Make sure labels are masked for prompt; use small batch size and save checkpoint.

Per-token loss inspection (medium)

- Task: After fine-tuning, compute per-token loss on a validation set and plot a heatmap of loss across tokens of several examples. Identify words causing high loss and expand training data with examples containing those words.
- Hint: Use the per-token loss snippet and seaborn. High loss often corresponds to rare tokens or formatting mistakes.

Robust prompts (medium)

- Task: Create 5 paraphrased prompt templates for the same instruction and fine-tune using all templates. Evaluate zero-shot generalization to unseen paraphrases.
- Hint: Data augmentation by paraphrasing reduces prompt-sensitivity.

Parameter-efficient tuning (advanced)

- Task: Implement LoRA (low-rank adapter) for attention weights and fine-tune only LoRA parameters on instructions. Compare final performance and compute/memory footprint vs full-finetuning.
- Hint: Use PEFT library (Parameter-Efficient Fine-Tuning) or implement low-rank updates on query/key/value projection matrices.

Safety-aware SFT (advanced)

- Task: Build a small filter or classifier to remove toxic outputs from the training targets. Fine-tune with and without filtered data and compare the model outputs on prompts likely to trigger toxicity.
- Hint: Use a simple toxicity classifier (e.g., Perspective API or a small HF toxicity model) to flag targets during preprocessing.

---

## Fine-tuning on a single task

### Direct definition

Fine-tuning on a single task is the supervised process of continuing training a pretrained language model on data for one narrowly defined task (e.g., sentiment classification, summarization, code generation) so the model adapts its representations and outputs specifically to that task while retaining useful general language knowledge.

### Concept intuition

- What it is: start from a pretrained LLM (rich language priors) and adjust its weights so it performs one target mapping reliably: input → task-specific output.
- Why it matters: full pretraining is expensive and task-agnostic; task-specific fine-tuning is sample-efficient and yields much better task performance with far less compute.
- Analogy: pretraining is learning a language and world model; single-task fine-tuning is learning a specialized skill (like legal drafting) by practicing only that task instead of retraining the entire brain.
- When to use it: you have a clear, labeled dataset for one task; you need a single production model optimized for that task; latency and accuracy trade-offs prioritize a single-purpose model.

### Mathematical breakdown

Supervised cross-entropy (classification or seq2seq)

- For classification (K classes), model outputs probability vector p(y | x; θ). Loss per example:

```
L = - sum_{k=1..K} 1[y=k] * log p(y=k | x; theta)
```

- For sequence generation (response y1..yT), causal LM cross-entropy on target tokens:

```
L = - sum_{t=1..T} log p(y_t | x, y_{1:t-1}; theta)
```

Batch update (SGD/AdamW)

```
L_batch = (1/N) * sum_{i=1..N} L(x_i, y_i)
theta <- theta - eta * grad_theta(L_batch)
```

Fine-tune vs. feature-extract

- Full fine-tune: update theta for all model parameters.
- Feature-extract (head-only): freeze base parameters and learn only a new task head (fewer params, faster, less catastrophic forgetting).

Regularization and stability

- Weight decay (decoupled AdamW), gradient clipping, learning-rate warmup. Typical LR ranges:
    - Full fine-tune (large LLMs): 1e-5 — 5e-5
    - Head-only: 1e-4 — 1e-3

Evaluation objective (task-specific)

- Classification: accuracy, F1, precision/recall.
- Generation: BLEU/ROUGE, but prefer human eval or task-specific metrics (e.g., EM for QA).

### Code & practical application

Plan: fine-tune a pretrained model on a single task end-to-end. I give two minimal, runnable examples: (A) classification (sentiment) using Hugging Face + PyTorch, (B) seq2seq summarization using T5. Both are copy-paste-ready for a notebook.

A. Binary classification (IMDb-like toy)

```python
# pip install transformers datasets accelerate evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load tiny dataset
ds = load_dataset("imdb", split="train[:1%]")  # toy subset
ds = ds.train_test_split(test_size=0.1, seed=42)
metric = evaluate.load("accuracy")

def preprocess(example):
    return tokenizer(example["text"], truncation=True)
ds = ds.map(preprocess, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./ft-sentiment",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
```

B. Sequence-to-sequence summarization (T5-small)

```python
# pip install transformers datasets accelerate rouge_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import evaluate

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

ds = load_dataset("cnn_dailymail", "3.0.0", split="train[:0.1%]")  # tiny
ds = ds.train_test_split(test_size=0.2, seed=42)
rouge = evaluate.load("rouge")

max_input = 512
max_target = 128

def preprocess(examples):
    inputs = ["summarize: " + t for t in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)
    labels = tokenizer(examples["highlights"], max_length=max_target, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./ft-summarize",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    learning_rate=3e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

def compute_metrics(pred):
    preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    refs = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    return rouge.compute(predictions=preds, references=refs)

trainer = Trainer(model, training_args, train_dataset=ds["train"], eval_dataset=ds["test"],
                  tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
trainer.evaluate()
```

Practical tips:

- Use small subsets first, verify pipeline, then scale.
- For larger models, use gradient accumulation, fp16, and/or DeepSpeed/FairScale.
- Save and test frequent checkpoints to avoid wasted runs.

### Visualization / Geometry

Embedding drift

- Visual check: save token embedding matrix (before and after fine-tune) and project a sample of tokens with UMAP or t-SNE to see how task-specific words move closer to task-relevant clusters.

Hidden representation change

- For classification, take CLS (or pooled) hidden vector before/after fine-tune; project examples colored by label. Fine-tuning should open class-separable manifolds.

Gradient flow view

- Visualize gradient norms per layer during training; often lower layers get smaller gradient updates in late fine-tuning for stable models.

Example: plot pooled hidden vectors with t-SNE

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

model.eval()
texts = ["I love this movie.", "Awful film, do not watch."] * 50
inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
with torch.no_grad():
    out = model(**{k: v.to(model.device) for k,v in inputs.items()}, output_hidden_states=True)
# For encoder-decoder use encoder last_hidden_state pooled method; for classification use logits or pooler.
hidden = out.hidden_states[-1][:,0,:].cpu().numpy()  # [batch, dim]
emb = TSNE(n_components=2).fit_transform(hidden)
plt.scatter(emb[:,0], emb[:,1], c=[0 if "love" in t else 1 for t in texts])
plt.show()
```

### Common pitfalls & tips

- Wrong label alignment: for seq2seq ensure labels are shifted to align with decoder outputs; HF Trainer handles this, but manual loops often err.
- Too high LR: causes catastrophic forgetting; start low and monitor validation loss.
- Overfitting small datasets: use early stopping, dropout, data augmentation, or freeze lower layers.
- Tokenizer truncation: losing important context leads to poor performance—inspect token lengths and adjust max_length.
- Imbalanced labels: use class weighting, sampling, or metrics like macro-F1.
- Poor evaluation metric: for generative tasks, BLEU/ROUGE often don’t reflect human utility—prefer task-specific measures or human review.

### Interview-ready insights

- When to full fine-tune vs. head-only vs. PEFT:
    - Full fine-tune: best performance when compute permits and dataset is ample.
    - Head-only: fast and stable for small datasets or when preserving pretrained behavior is important.
    - PEFT (LoRA, adapters, prefix-tuning): minimal storage/compute, great for many-task multi-model deployment.
- LR scheduling: short warmup then small decay stabilizes adaptation; a tiny LR can still produce large representation changes over many steps.
- Data efficiency tricks: data augmentation, label-smoothing, mixup for text (word dropout, back-translation), and replay of generic data to avoid forgetting.
- Debugging tips: check loss on a single batch (should decrease), check random sample predictions frequently, and use per-token loss to diagnose tokenization or vocabulary issues.

### Practice exercises

(Easy) Sentiment fine-tune check

- Task: Fine-tune a DistilBERT model on a 500-example sentiment dataset, freeze all transformer layers and train only the classification head for 3 epochs. Compare accuracy vs full fine-tune for 1 epoch.
- Hint: set require_grad=False for model.base_model.parameters() to freeze.

(Medium) Per-layer learning-rate

- Task: Implement discriminative learning rates: smaller LR for lower layers, larger for head. Fine-tune and compare final accuracy.
- Hint: in the optimizer, pass parameter groups like [{"params": base_layers, "lr": lr*0.1}, {"params": head_params, "lr": lr}].

(Medium) Data-efficiency augmentation

- Task: For a small summarization dataset (200 examples), create 3 paraphrases of each article prompt (use simple back-translation via a pretrained translation model or sentence paraphraser). Fine-tune T5 and report ROUGE improvements.
- Hint: HF Transformers has translation models; run on small batches to create augmented data offline.

(Advanced) LoRA vs full fine-tune

- Task: Using PEFT (pip install peft), apply LoRA to a causal LM and fine-tune on a code-generation task. Compare model size stored (adapter params) and generation quality on held-out examples.
- Hint: PEFT exposes simple APIs to wrap your model; measure GPU RAM and saved checkpoint size.

(Advanced) Catastrophic forgetting mitigation

- Task: Mix a small fraction (5–10%) of pretraining-like language modeling data into each fine-tune batch (replay) to reduce forgetting. Compare to a baseline without replay on general-language tasks like perplexity on held-out Wiki text.
- Hint: prepare a tiny LM dataset (wikitext-2) and combine datasets at the DataLoader level with sampling weights.

---

## Multi-task instruction fine-tuning

### Direct definition

Multi-task instruction fine-tuning is the supervised training procedure that continues training a pretrained language model on a diverse collection of (instruction, input, response) tasks simultaneously so the model learns a single policy that generalizes to many instruction types and to novel instructions at inference time.

### Concept intuition

- What it is and why it matters: instead of adapting a model to one narrowly-defined task, multi-task instruction fine-tuning exposes the model to many different tasks (summarization, translation, QA, classification, code generation, etc.) with a unified instruction format. The model learns a mapping from the *intent* expressed by an instruction to the appropriate behavior, improving zero-shot and few-shot generalization to unseen tasks and phrasing.
- Visual analogy: imagine training a chef on many cuisines with labeled recipes and instructions; after seeing many cooking tasks, the chef can reason about new recipes from new instructions. The model develops a task-understanding manifold where instruction tokens steer activations toward task-specific subspaces.
- Practical benefit: one deployed model can handle many user requests without separate fine-tuned weights per task, reducing engineering overhead and improving UX consistency.

### Mathematical breakdown

Training objective

- Every example i has Instruction I_i, Input X_i, Response Y_i (a token sequence y1..yT).
- Concatenate a formatted prompt P_i = format(I_i, X_i).
- Standard token-level cross-entropy objective for causal LM SFT:

```
L_i(theta) = - sum_{t=1..T} log p_theta(y_t | P_i, y_{1:t-1})
L_batch(theta) = (1/N) * sum_{i=1..N} L_i(theta)
theta <- theta - eta * grad_theta(L_batch)
```

- For encoder-decoder models the decoder is conditioned on encoder(P_i) and loss is cross-entropy on decoder outputs.

Weighted multi-task objective

- If tasks are heterogeneous or imbalanced, apply per-task weights w_task:

```
L_batch = (1/N) * sum_{i=1..N} w_task(i) * L_i
```

- Choices for w: inverse-frequency (upweight rare tasks), balanced, or learned.

Auxiliary losses for robustness

- Mix in auxiliary objectives to preserve pretraining knowledge or shape behavior:
    - Language modeling replay: L_LM to avoid forgetting
    - Supervised classifiers or constraint losses (e.g., toxicity penalty)
- Joint objective:

```
L_total = L_tasks + alpha * L_LM + beta * L_safety
```

- alpha, beta are scalars controlling trade-offs.

Curriculum and sampling

- The effective distribution the model learns from is determined by sampling strategy S over tasks. Mathematically the expected loss is:

```
E_{task ~ S} E_{(x,y)~D_task} [ L(task, x, y) ]
```

- Choosing S influences convergence and generalization.

### Code and practical application

Plan: build a compact, runnable multi-task instruction fine-tuning pipeline using Hugging Face Transformers and Datasets. We show: dataset format, formatter, balanced sampler, and training with Trainer. Use a small causal LM (gpt2) or T5 for seq2seq. Example uses T5 for clearer encoder-decoder conditioning.

Minimal multi-task dataset JSONL example lines

```json
{"task":"summarize","instruction":"Summarize the following","input":"Cats sleep a lot.","output":"Cats are known to sleep many hours a day."}
{"task":"translate","instruction":"Translate to French","input":"I love you.","output":"Je t'aime."}
{"task":"qa","instruction":"Answer the question concisely","input":"Q: What is the capital of France? Context: France is a European country.","output":"Paris."}
```

Data formatting and tokenization

```python
# pip install transformers datasets accelerate sentencepiece
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# load your jsonl files or construct synthetic dataset
ds = load_dataset("json", data_files={"train":"multi_task_train.jsonl","validation":"multi_task_val.jsonl"})

# Prompt template function supports multiple task styles for robustness
def make_prompt(example):
    task = example.get("task","")
    instr = example.get("instruction","").strip()
    inp = example.get("input","").strip()
    # multiple templates to reduce prompt brittleness
    templates = [
        f"{instr}\n\n{inp}",
        f"Task: {task}\nInstruction: {instr}\nInput: {inp}",
        f"{instr}\nInput:\n{inp}\nResponse:"
    ]
    return random.choice(templates)

max_input = 512
max_output = 128

def preprocess(batch):
    prompts = [make_prompt(x) for x in batch]
    model_inputs = tokenizer(prompts, max_length=max_input, truncation=True, padding="longest")
    labels = tokenizer(batch["output"], max_length=max_output, truncation=True, padding="longest")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
ds.set_format(type="torch")
```

Balanced sampling across tasks

```python
from collections import Counter
import numpy as np

# compute task frequencies and sampling weights
train = ds["train"]
tasks = train["task"]
freq = Counter(tasks)
weights = [1.0 / freq[t] for t in tasks]
# normalize
weights = np.array(weights) / np.sum(weights)

# create a sampler for DataLoader if using PyTorch training loop
from torch.utils.data import DataLoader, WeightedRandomSampler
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
loader = DataLoader(train, batch_size=8, sampler=sampler)
```

Training with Trainer (simple)

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./multi_task_ft",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-5,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    logging_steps=50,
    weight_decay=0.01,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
```

Inference across tasks

```python
def run_prompt(instruction, input_text, max_new_tokens=128):
    prompt = f"{instruction}\n\n{input_text}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(run_prompt("Translate to French", "I love you."))
print(run_prompt("Summarize the following", "Cats sleep a lot and like warm places."))
```

Practical notes

- Use multiple templates during training to reduce prompt-format sensitivity.
- Use a validation set per task and aggregate metrics per task for diagnostics.

### Visualization and geometry

Task-conditioned embedding manifold

- Extract encoder pooled representations for prompts from different tasks. Project with UMAP/t-SNE to see clustering by task and by instruction phrasing. Multi-task training should produce a manifold where instruction tokens move representations toward task clusters.

Attention patterns

- For encoder-decoder models inspect cross-attention when decoding; for summarization the decoder cross-attends broadly to content; for translation it may show alignment-like attention. Plot attention heatmaps with tokens on axes to compare across tasks.

Per-task loss landscape

- Plot per-task validation loss curves. Visualize how some tasks converge faster and how trade-offs occur (improving one task can hurt another if data imbalance or capacity limits exist).

Gradient attribution across tasks

- Compute gradient norms per layer aggregated by task to see which layers each task relies on more. Visualization helps decide freezing lower layers or using adapters.

Example snippet to collect encoder pooled states and project

```python
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model.eval()
examples = ds["validation"].select(range(200))
inputs = tokenizer([x["input_ids"] for x in examples], return_tensors="pt", padding=True)  # adapt if already tensors
with torch.no_grad():
    out = model.encoder(input_ids=inputs["input_ids"].to(model.device), attention_mask=inputs["attention_mask"].to(model.device))
    # mean pool encoder last hidden state
    pooled = out.last_hidden_state.mean(dim=1).cpu().numpy()

proj = TSNE(n_components=2).fit_transform(pooled)
colors = [hash(t)%10 for t in examples["task"]]
plt.scatter(proj[:,0], proj[:,1], c=colors, cmap="tab10")
plt.title("Encoder pooled states projected by task")
plt.show()
```

### Common pitfalls and tips

- Imbalanced tasks dominate learning: frequent tasks overshadow rare ones. Fix with sampling weights, up/downsampling, or per-task loss weights.
- Overfitting to templates: if training used a single prompt template, model will be brittle to new phrasing. Use multiple templates, paraphrases, and adversarial prompts.
- Catastrophic forgetting of general knowledge: mix in LM replay or use low alpha for auxiliary LM loss.
- Competing objectives across tasks: some tasks require brevity, others verbosity; control with task-specific decoding parameters or include task-specific tokens that encode desired length/format.
- Evaluation complexity: aggregate metrics can hide per-task failure modes. Always report per-task metrics and qualitative examples.
- Capacity limits: smaller models can’t perfectly model many tasks — consider adapters or multi-model routing.
- Safety across tasks: instruction tasks can induce harmful outputs in some tasks; apply filtering and include safety tasks or penalties in objective.

### Interview-ready insights

- Why multi-task works: shared representations and a common instruction format let the model learn a meta-mapping from *instruction semantics* to behavior, enabling zero-shot transfer to new tasks.
- Sampling is as important as model architecture: the effective training distribution shapes what the model can do. Explain inverse-frequency sampling, temperature-based sampling, or curriculum learning in answers.
- Parameter-efficient alternatives: adapters, LoRA, and prefix-tuning allow multi-task behavior by adding small task-specific modules rather than retraining full weights. They scale better when you want separate task footprints.
- Metrics design: for multi-task models, emphasize per-task metrics, calibration, and human evaluation over single aggregated numbers. Explain trade-offs between optimizing a single averaged loss versus per-task targeted tuning.
- Debugging multi-task failure: check per-task loss, per-task sample quality, tokenization issues, prompt format mismatch, and label noise before changing model hyperparameters.

### Practice exercises

Basic multi-task SFT (easy)

- Task: Create a small multi-task JSONL with three tasks (summarize, translate, QA) each 200 examples. Fine-tune flan-t5-small for 2 epochs and evaluate per-task metrics.
- Hint: use multiple prompt templates for each example.

Balanced sampling ablation (medium)

- Task: Train two models: one with naive dataset sampling (natural imbalance) and one with inverse-frequency sampling. Compare per-task validation loss and examples where the naive model fails.
- Hint: measure variance of per-task performance and produce plots.

Prompt robustness via augmentation (medium)

- Task: For a single task type, generate 5 paraphrased prompt templates and include them in multi-task training. Evaluate zero-shot performance on unseen paraphrase forms.
- Hint: use simple back-translation or a paraphrasing model offline to create variations.

Adapters versus full fine-tune (advanced)

- Task: Use PEFT adapters for each task (task-id conditioned) or a shared multi-task adapter and compare parameter counts and per-task performance to full fine-tuning. Report storage and runtime trade-offs.
- Hint: implement a routing token like "<TASK_SUMMARIZE>" at the start and add small per-task adapters.

Safety and task trade-off experiment (advanced)

- Task: Add a toxicity penalty term in the loss using a pretrained toxicity scorer: L_total = L_task + gamma * ToxicityScore(output). Compare outputs on potentially adversarial prompts with and without the penalty.
- Hint: use a small toxicity classifier to score generated tokens and backprop through a differentiable proxy or apply offline filtered targets.

---

## Scaling instruct models

### Direct definition

Scaling instruct models means increasing a model’s capability and instruction-following quality by systematically enlarging model size, dataset size/diversity, compute, and training techniques (architecture changes, optimization, parameter-efficiency, and system-level engineering) so the model generalizes better, follows instructions more robustly, and remains practical to train and serve.

### Concept intuition

- What “scaling” buys you: larger models and more diverse instruction data usually improve few/zero-shot generalization, reasoning, and robustness to novel prompts. Scaling also exposes failure modes (toxicity, hallucination), so scaling is paired with alignment methods (SFT, RLHF, reward models).
- Why multiple axes matter: capability is not only parameter count. Data quality, instruction diversity, compute budget, optimization stability, and inference latency all interact. Think of capacity (model params) like muscle, data like practice, compute as training time, and system engineering as the gym equipment that lets you lift heavier.
- Scaling trade-offs: bigger models are better but cost more to train/serve; parameter-efficient tuning (LoRA/adapters) and MoE split that trade-off by increasing capacity for sparse compute or swapping small task-specific modules.
- Real-world analogy: to build a world-class chef (instruction model), you can 1) give them more brainpower (bigger model), 2) expose them to many recipes and critiques (multi-task SFT/RLHF), 3) specialize with small toolkits (PEFT) for tasks they’ll repeat, and 4) design kitchens (systems) that let them cook fast for many customers.

### Mathematical breakdown

Scaling laws (rough empirical form)

- Empirical relationship (power-law) between loss, model size (N), dataset size (D), and compute (C):

```
L(N, D) ≈ a * N^(-α) + b * D^(-β) + const
```

- Where:
    - N: number of parameters
    - D: number of training tokens
    - α, β: empirical exponents (>0)
    - a, b: constants fit to experiments
- Interpretation: doubling parameters or data yields diminishing returns; best gains come from balanced scaling (N and D tuned together).

Compute-optimal frontier

- Given fixed compute C, optimal N and D roughly satisfy:

```
C ≈ k * N * D_eff
```

- And empirical optimal D scales roughly with sqrt(C) or other power; practical rule: increase D and N together rather than one alone.

Mixture-of-Experts (sparse) capacity

- MoE uses G experts; only k experts are active per token. Effective parameters:

```
N_effective = N_shared + G * N_expert
compute_per_token ∝ N_shared + k * N_expert
```

- Sparse routing reduces compute at inference while increasing representational capacity.

PEFT parameter count (LoRA)

- LoRA replaces a weight update W ∈ R^{d_out×d_in} with low-rank update:

```
Delta_W = A @ B  # A ∈ R^{d_out×r}, B ∈ R^{r×d_in}, r << min(d_out,d_in)
```

- Train params = r*(d_out + d_in) vs full d_out*d_in; large compression ratio.

Regularization and generalization trade-offs

- Weighted multi-task loss and replay (from instruction fine-tuning) remain:

```
L_total = Σ_t w_t * L_t + α * L_LM + β * L_safety
```

- Where task weighting and replay control forgetting and capacity allocation across tasks.

### Code & practical application

Goal: practical recipes for scaling an instruction model in four phases: data, model, efficient fine-tuning, and system-level training. Use Hugging Face + PEFT + DeepSpeed snippets.

Data scaling: build diverse instruction dataset

- Strategy: aggregate many sources (SFT corpora, instruction datasets), dedupe, filter toxicity, and apply prompt-templates + paraphrases.
- Toy augmentation example (paraphrase using a model locally):

```python
# paraphrase generation skeleton (run offline on small batches)
from transformers import pipeline
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")  # example
def augment_instruction(instr):
    out = paraphraser(f"paraphrase: {instr}", max_length=128, num_return_sequences=3)
    return [o['generated_text'] for o in out]
```

PEFT (LoRA) fine-tuning for large models (fast and practical)

- Install: pip install transformers accelerate peft
- Minimal LoRA wrap and training sketch:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "gpt2"  # scale to bigger models in real runs
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA config (rank r, alpha scaling)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # adapt to model layer names
)
model = get_peft_model(model, lora_config)
# then train model with HF Trainer or custom loop (only LoRA params require gradients)
```

- Benefits: drastically fewer trainable params, smaller checkpoints.

DeepSpeed ZeRO + FP16 for large-scale training

- Example DeepSpeed config (save as ds_config.json):

```json
{
  "train_batch_size": 64,
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  },
  "optimizer": {"type": "AdamW", "params": {"lr": 1e-5, "betas": [0.9,0.95], "eps": 1e-8, "weight_decay": 0.01}}
}
```

- Launch training with Accelerate or deepspeed launcher to enable memory scaling.

MoE concept and skeleton (using mixture experts frameworks)

- Implement routing in transformer blocks or use libraries (e.g., FastMoE, Mesh TensorFlow). Example conceptual code for gating (not runnable as-is):

```python
# gate: project token representation to logits over G experts and softmax select top-k
gate_logits = gate_proj(hidden_state)  # [batch, seq_len, G]
topk_idx = topk(gate_logits, k)  # indices of top-k experts per token
# route token representation to selected experts
```

- MoE training requires load balancing auxiliary loss to prevent expert collapse:

```
L_total = L_task + lambda_balance * L_balance
L_balance ≈ coefficient * variance_of_expert_load
```

Inference optimizations: quantization + distillation

- Post-training quantization (bitsandbytes or PyTorch native):

```bash
# example using bitsandbytes with HF transformers
pip install bitsandbytes
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("big-model", load_in_4bit=True, device_map="auto")
```

- Distillation: train a smaller student with teacher outputs (logits or sequence-level) to preserve instruction-following behavior.

RLHF scale considerations

- RLHF pipeline at scale: SFT -> preference data collection -> train reward model -> PPO policy optimization (with KL control). PPO requires careful hyperparameter tuning, clipping, and compute (many rollouts per update).

### Visualization / Geometry

- Capacity manifolds: project pooled prompt embeddings from small and scaled models; larger models show clearer task clusters and smoother interpolation between instruction types. Use UMAP/t-SNE on encoder/decoder pooled states.
- Parameter adaptation heatmaps: visualize which layers change most during large-scale SFT/SFT+RLHF by plotting per-layer cosine distance between pretrain and posttrain weights. This shows where adaptation concentrates.
- Expert routing visualization (MoE): for a given batch, plot token-to-expert assignment heatmap over tokens and layers to inspect load distribution and specialization.
- Gradient and loss surfaces: monitor gradient norms across layers and validation loss per task; scaling often shifts gradient mass to higher layers. Example code to compute layerwise grad norms:

```python
# after loss.backward()
layer_grads = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        layer = name.split('.')[0]
        layer_grads.setdefault(layer, []).append(param.grad.norm().item())
layer_grad_means = {layer: sum(v)/len(v) for layer,v in layer_grads.items()}
```

### Common pitfalls & tips

- Unbalanced scaling: increasing N without enough diverse D leads to overfitting or brittleness; follow compute-optimal scaling (increase data with model).
- Expert collapse in MoE: routers assign everything to a few experts; use auxiliary load-balancing loss and noisy gating.
- Catastrophic forgetting when fine-tuning at scale: include LM replay or low learning rates, and consider continual learning or adapters.
- Cost vs. latency trade-offs: large dense models are costly to serve; use quantization, distillation, or Mixture-of-Experts for sparse inference.
- Instability with mixed precision: use stable optimizers, gradient scaling, and correct eps/betas; validate gradient norms early.
- Safety and misalignment scale faster: bigger models can confidently hallucinate; invest in reward models, filtering, and post-generation checks.
- Checkpoint explosion: frequent full checkpoints are expensive—use parameter-efficient checkpoints (LoRA/adapters) and save via HF/PEFT-friendly formats.

### Interview-ready insights

- Balanced scaling beats unilateral scaling: explain why N, D, and compute should grow together using the scaling law intuition and compute-optimal frontier.
- Parameter-efficiency is often the right engineering choice: LO/LoRA/adapters let you scale abilities and deploy many specialized behaviors cheaply. Mention how Delta_W = A @ B reduces trainable params.
- MoE increases capacity cheaply but complicates systems: pros—huge capacity with sparse compute; cons—routing complexity, load balancing, and hardware inefficiencies.
- Systems matter as much as algorithms: ZeRO, DeepSpeed, tensor-slicing, and device mesh strategies enable training models impossible otherwise. Be ready to talk about ZeRO stages and how they trade memory vs communication.
- Distillation and quantization are the production levers: you can scale at training time but must compress for latency-sensitive serving. Distillation can preserve instruction-following behavior if teacher signals include instruction-conditioned outputs.
- RLHF at scale is a multi-component pipeline: collecting preferences, training reward models, and stabilizing PPO—explain KL-penalty to prevent drift from SFT policy.

### Practice exercises

Balanced scaling experiment (medium)

- Task: Pick a small model (T5-small). Create three training runs: (A) fixed dataset, scale model by 2× (simulate by training longer or using wider hidden dims), (B) double dataset size (paraphrases/augment), (C) both N and D doubled (increase training steps + augment). Compare validation loss and instruction generalization on held-out tasks.
- Hint: run on small subsets; measure per-task and aggregated metrics, and report compute used (steps × batch × model flop proxy).

LoRA practicality (easy → medium)

- Task: Apply LoRA to a transformer model and fine-tune on an instruction dataset. Measure checkpoint size, GPU memory, and time vs full fine-tune. Evaluate on held-out instructions.
- Hint: use peft and measure trainable param count via sum(p.numel() for p in model.parameters() if p.requires_grad).

MoE simulation (advanced)

- Task: Implement a tiny MoE layer in PyTorch with 4 experts and top-1 gating. Train it on a toy multi-task instruction dataset. Track expert utilization and add a load-balancing loss. Visualize token-to-expert assignments.
- Hint: start with small experts (2-layer MLP) and a softmax gating with LoadLoss = variance(normalized_loads).

Quantization & distillation (medium)

- Task: Take a fine-tuned instruction model and (A) quantize to 8-bit or 4-bit for inference, (B) distill to a smaller student via teacher logits or sequence-level loss. Compare generation quality and latency.
- Hint: use bitsandbytes for quantization and standard knowledge distillation losses for training the student.

RLHF mini-pipeline (advanced)

- Task: Build a small RLHF loop: collect human or synthetic preference pairs for two model outputs, train a reward model, and run PPO to improve a policy starting from SFT weights, with KL control to SFT. Evaluate improvement by preference accuracy.
- Hint: keep environment small, use synthetic annotators (e.g., heuristics) if human labels are unavailable, and use stable baselines or trl for PPO components.

---

## Model evaluation

### Direct definition

Model evaluation is the systematic measurement of a model’s behavior using quantitative and qualitative metrics, tests, and procedures so you can judge correctness, usefulness, robustness, safety, and calibration for the model’s intended tasks and deployment settings.

### Concept intuition

- Purpose: decide if a model is fit for purpose, diagnose failures, compare versions, and guide further training or deployment decisions.
- Why it matters for instruction and generative models: generative models produce open-ended outputs where simple token-level loss or perplexity doesn’t capture helpfulness, truthfulness, or safety; evaluation must measure task correctness, style, factuality, calibration, and failure modes.
- Two complementary views:
    - Behavioural evaluation: how the model performs on tasks people care about (accuracy, ROUGE, BLEU, human preference).
    - Diagnostic evaluation: internal signals and stress tests (per-token loss, calibration, adversarial prompts, hidden state probes).
- Practical trade-offs: automated metrics scale but are imperfect; human evaluation is expensive but essential for instruction-following and alignment.

### Mathematical breakdown

Core supervised metrics

- Accuracy for classification:

```
accuracy = (# correct predictions) / (total examples)
```

- Precision, Recall, F1 for imbalanced classes:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * (precision * recall) / (precision + recall)
```

Sequence generation metrics (token-level / n-gram)

- Cross-entropy / negative log-likelihood (NLL):

```
NLL = - sum_{t=1..T} log p(y_t | context)   # per-sequence
perplexity = exp(NLL / T)
```

- BLEU (n-gram precision with brevity penalty) and ROUGE (recall-oriented n-gram / longest-overlap) are computed from comparisons between generated and reference sequences.

Calibration and confidence

- Reliability diagram and expected calibration error (ECE):

```
# Partition predictions into M bins by confidence
ECE = sum_{m=1..M} (|bin_m|/N) * |acc(bin_m) - conf(bin_m)|
```

- Where acc(bin_m) is empirical accuracy in bin and conf(bin_m) is average predicted confidence.

Human preference modeling and ranking

- Preference probability from a learned reward model r_theta:

```
P(A > B) = sigmoid( r_theta(A) - r_theta(B) )
```

- Rank correlation (Spearman rho) between model scores and human ranks:

```
rho = 1 - (6 * sum d_i^2) / (n*(n^2 - 1))
```

Robustness and stress tests

- Adversarial flip rate:

```
flip_rate = (# examples where small perturbation changes output undesirably) / total
```

- Distribution shift gap:

```
gap = metric_in_domain - metric_out_of_domain
```

Composite and utility losses

- Combined evaluation objective for model selection:

```
score = w_task * task_metric - w_toxic * toxic_rate - w_halluc * hallucination_rate
```

- Weights chosen to reflect product priorities.

### Code and practical application

I use PyTorch + Hugging Face for examples. The snippets are minimal, copy-paste-ready.

A. Compute basic metrics for classification and generation

```python
# pip install evaluate transformers datasets rouge_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import numpy as np

# Example: summarization evaluation
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

ds = load_dataset("cnn_dailymail", "3.0.0", split="test[:200]")  # small subset
rouge = evaluate.load("rouge")

def generate_summary(text):
    inp = "summarize: " + text
    ids = tokenizer(inp, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=120)
    return tokenizer.decode(out[0], skip_special_tokens=True)

preds, refs = [], []
for ex in ds:
    pred = generate_summary(ex["article"])
    preds.append(pred)
    refs.append(ex["highlights"])

res = rouge.compute(predictions=preds, references=refs)
print(res)
```

B. Per-token loss and calibration

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

text = "The capital of France is Paris."
enc = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**enc, labels=enc["input_ids"])
    # loss is averaged; get per-token cross entropy
    logits = outputs.logits  # [1, seq_len, vocab]
    log_probs = F.log_softmax(logits, dim=-1)
    target = enc["input_ids"][0]
    per_token_logprob = log_probs[0, torch.arange(logits.size(1)), target]
    per_token_loss = -per_token_logprob.cpu().numpy()
print("Per-token loss:", per_token_loss)
```

C. Calibration ECE implementation

```python
import numpy as np

def compute_ece(confidences, labels, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if bin_mask.sum() == 0: continue
        acc = labels[bin_mask].mean()
        conf = confidences[bin_mask].mean()
        ece += (bin_mask.sum()/len(labels)) * abs(acc - conf)
    return ece
# confidences: predicted prob for chosen class; labels: 0/1
```

D. Automated factuality check with QA-based consistency

- Idea: to check if a generated statement S about context C is factual, convert S into questions and test answers against C using an extraction QA model. Implement as pipeline:
    1. Extract claims from S (or split into statements).
    2. Generate question(s) per claim (using a question generation model).
    3. Use an open-book QA or retrieval over context C; compare answers for support and score alignment.

E. Human preference evaluation scaffold

- Collect K model outputs per input with randomized order. Use a UI (streamlit or simple web form) to collect pairwise comparisons or ranking. Aggregate with Bradley-Terry or train a reward model using logistic loss:

```
Loss = - sum_i log sigmoid( r_theta(out_preferred) - r_theta(out_other) )
```

### Visualization and geometry

- Per-token loss heatmap: plot tokens on x-axis, per-token loss as a heat row. Use to find problematic tokens or places where model hallucinates.
- Embedding- and hidden-state drift plots: measure cosine distance between pre-finetune and post-finetune hidden states per-layer to see where adaptation focused.
- Attention diagnostic plots: visualize attention maps while the model generates the response to verify focus on input or instruction tokens.
- Calibration reliability diagram: plot predicted confidence bins vs empirical accuracy for a classifier or token-level next-token probabilities.
- Preference scatter plots: visualize reward model scores vs human preference probability to diagnose reward model misalignment.

Example per-token loss heatmap snippet

```python
import seaborn as sns
import matplotlib.pyplot as plt
# per_token_loss: array [seq_len]
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
plt.figure(figsize=(12,2))
sns.heatmap([per_token_loss], xticklabels=tokens, yticklabels=[], cmap="Reds")
plt.xticks(rotation=45, ha="right")
plt.show()
```

### Common pitfalls and tips

- Using sole automatic metrics: BLEU/ROUGE/PPL often miss hallucination, style, or instruction-following quality. Always pair with targeted tests and human eval.
- Test set leakage: ensure no overlap between training and evaluation, especially in instruction datasets where paraphrases leak task templates.
- Aggregated metrics hide failure modes: report per-task, per-input-length, per-domain, and percentile metrics (e.g., 10th percentile).
- Calibration confusion: next-token probability is not the same as semantic confidence for an open-ended answer — use separate calibration tests or Monte Carlo dropout/ensembles to estimate uncertainty.
- Reward model proxy limitations: reward models can be gamed; use diverse annotators, adversarial examples, and calibration checks.
- Overfitting to evaluation metrics: optimizing for ROUGE can reduce human preference; avoid metric-only training unless it matches product need.
- Human eval design mistakes: present randomized outputs, blind annotators to source, and provide clear rubrics that separate correctness, helpfulness, style, and safety.

### Interview-ready insights

- Metric selection must match product goals: retrieval QA needs exact-match and F1; summarization needs faithfulness and factuality checks beyond ROUGE; chat assistants need preference and safety metrics.
- Perplexity is useful for broad model quality but correlates weakly with instruction-following quality; prefer task-specific evals for fine-tuned models.
- Calibration matters in production: a confident wrong answer is worse than a cautious answer. Implement confidence thresholds and abstention mechanisms.
- Use hierarchical evaluation: automatic unit tests → large-scale automated metrics → targeted adversarial tests → human preference studies. This pipeline balances scale and fidelity.
- Reward models require careful validation: check that reward correlates with human preference and is robust to adversarial optimization; use KL control when optimizing policies.
- Reporting: provide broken-down metrics (by domain, prompt type, toxicity risk) and worst-case slices, not just averages.

### Practice exercises

Per-token diagnostic and heatmap

- Task: For a small GPT-2 fine-tuned on summarization, compute per-token loss for 50 examples and create a heatmap to find frequent high-loss tokens. Propose data edits to fix the high-loss tokens.
- Hint: high per-token loss often maps to rare or incorrectly-tokenized words; augment data or adjust tokenizer.

Calibration experiment

- Task: For a binary classification head you fine-tuned, compute ECE and plot a reliability diagram. Apply temperature scaling on the logits to recalibrate and report new ECE.
- Hint: fit temperature T by minimizing NLL on validation set: softmax(logits / T).

Factuality probe using QA

- Task: Given model-generated summaries for news articles, implement a QA-based factuality check: generate questions from each summary sentence, answer them from the original article using an extractive QA model, and compute support rate (fraction of claims supported).
- Hint: use a pretrained question-generation model and an extractive QA model from Hugging Face.

Preference model training

- Task: Collect synthetic preference pairs from heuristics (e.g., shorter, non-hallucinated outputs win) for 1k examples. Train a small reward model that predicts preference and evaluate Spearman correlation with held-out human judgments (if available).
- Hint: use logistic loss and compare predicted probabilities to empirical preference rates.

Robustness slicing

- Task: Create a battery of adversarial prompts (typos, negations, leading questions) for an instruction model. Measure flip_rate and accuracy before/after adversarial training. Report which prompt types cause most failures.
- Hint: automate paraphrase and negation injection, and use seed lists of tricky constructs.

---

## Benchmarks

### Direct definition

Benchmarks are standardized datasets, tasks, metrics, and evaluation protocols used to measure and compare model capabilities, generalization, calibration, and safety across models and training regimes.

### Concept intuition

- Purpose: provide objective, repeatable probes for capabilities (language understanding, reasoning, coding, instruction following, factuality) so engineers and researchers can measure progress and regressions.
- Why they matter: benchmarks guide model design, dataset collection, and tuning. They create common targets (GLUE, SuperGLUE, MMLU, HumanEval, BIG-bench, HELM, MT-Bench) and force reproducible evaluation across labs and products.
- Trade-offs: benchmarks are proxies — optimizing only for benchmarks can produce brittle, overfitted models that game metrics rather than genuinely improve usefulness or safety.

### Types of benchmarks and common examples

- Natural language understanding (NLU): GLUE, SuperGLUE (classification, inference, coref) — measure reasoning and comprehension.
- Knowledge and reasoning: MMLU (Multi-task Language Understanding), BigBench/Hard tasks — test broad knowledge and reasoning across domains.
- Instruction-following and conversational quality: MT-Bench, AlpacaEval, VicunaEval — often use human preference or pairwise comparisons.
- Code synthesis: HumanEval, MBPP — evaluate code generation correctness via unit tests.
- Safety and toxicity: RealToxicityPrompts, SafetyGym-style suites — measure harmful output rates.
- Robustness and adversarial: Dynabench, Adversarial NLI — test model stability to perturbations.
- Holistic evaluation suites: HELM (Holistic Evaluation of Language Models), BIG-bench — aggregate many tasks and provide standards for reporting.

### Core metrics (with formulas in code blocks)

- Accuracy (classification, multi-choice)

```
accuracy = correct_predictions / total_examples
```

- Exact Match (QA)

```
exact_match = (# predictions exactly equal to reference) / total_examples
```

- F1 (token-level overlap)

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)
```

- Pass@k (code eval via multiple samples)

```
pass_k = (# problems with >=1 correct solution in k samples) / total_problems
```

- BLEU / ROUGE (n-gram overlap) — standard libraries compute these.
- Preference win-rate (human eval, pairwise)

```
win_rate = (# times model_A preferred over model_B) / total_pairs
```

- Expected Calibration Error (ECE)

```
# implemented by binning confidences and averaging abs(conf - accuracy) weighted by bin size
```

### How to run common benchmark evaluations (practical code patterns)

GLUE / SuperGLUE quick evaluation (HF datasets + simple loop)

```python
# pip install datasets transformers evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import evaluate

model_name = "facebook/bart-large-mnli"  # example
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to("cuda")
metric = evaluate.load("glue", "mrpc")  # change task

ds = load_dataset("glue", "mrpc", split="validation")
def predict(batch):
    enc = tokenizer(batch["sentence1"], batch["sentence2"], truncation=True, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits
    preds = logits.argmax(-1).cpu().numpy()
    return preds

preds = []
labels = ds["label"]
for i in range(0, len(ds), 16):
    batch = ds[i:i+16]
    preds.extend(predict(batch))
print("Accuracy:", (np.array(preds)==np.array(labels)).mean())
```

MMLU-style multiple choice evaluation (prompt-based, few-shot)

- Pattern: format few-shot prompt + test question, generate logits for answer choices, compute accuracy.

```python
# pseudo-pattern
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def score_choices(prompt, choices):
    scores = []
    for c in choices:
        text = prompt + "\nAnswer: " + c
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            logits = model(ids).logits
        # score is sum log-prob of final choice tokens (or probability of choice token if single token)
        # implement careful masking to compute conditional probability of choice given prompt
        scores.append(computed_score)
    return scores
```

- Use existing evaluation utilities (lm-eval-harness or HF evaluation scripts) for robust handling of tokenization and scoring.

HumanEval / code correctness (unit-test based)

- Generate k completions, run unit tests in a sandbox, compute pass@k (take care with safety and timeouts).
- Use established harnesses (OpenAI HumanEval repo style) or HF evals.

Preference-based instruction benchmarks (MT-Bench, VicunaEval)

- Collect model responses A/B to same prompt, randomize order, gather human judgments, compute win rates and significance (bootstrap CIs). Use standard annotation UI or Mechanical Turk.

### Statistical rigor and significance

- Report confidence intervals: e.g., bootstrap on evaluation examples to get 95% CI for accuracy or win-rate.
- Paired tests for model comparisons (McNemar for classification, paired bootstrap for generation metrics).
- Report distributional metrics: median, 10th percentile, and worst-slices — averages hide rare catastrophic failures.

Bootstrap example (accuracy CI)

```python
import numpy as np
def bootstrap_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    accs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        accs.append((y_true[idx]==y_pred[idx]).mean())
    lo = np.percentile(accs, 100*alpha/2)
    hi = np.percentile(accs, 100*(1-alpha/2))
    return lo, hi
```

### Benchmark pitfalls and anti-patterns

- Overfitting to benchmarks: heavy tuning on benchmark validation or test sets leads to inflated progress. Prefer held-out, fresh benchmarks and locked test servers.
- Cherry-picked prompts and prompt-engineering bias: small prompt changes can flip results; report prompt templates and few-shot seeds.
- Single-number summaries: a single average metric conceals per-domain regressions and safety regressions.
- Dataset leakage: ensure training data did not include benchmark examples (especially for web-scale pretraining). When unsure, use time-sliced or provenance-filtered test sets.
- Misaligned metrics: ROUGE/BLEU vs human preference mismatch — always complement automatic metrics with targeted human evals.
- Non-reproducible setups: random seeds, temperature, decoding method, and evaluation sampling influence results—report them.

### Robustness, slices, and stress testing

- Slice by prompt length, domain, demographic terms, rare words, adversarial paraphrases.
- Use adversarial frameworks (Dynabench-style) to find failure modes interactively.
- Measure worst-percentile metrics (e.g., 5th percentile accuracy) and tail risks for safety-related outputs.

### Scaling benchmark strategies

- Hold out hard tasks that are not used in model tuning for robust generalization claims (e.g., BIG-bench Hard tasks).
- Use multi-metric dashboards: capability, calibration, toxicity, and latency together.
- Evaluate under resource constraints: latency, memory, quantized setups (deployed model variant).

### Interpreting benchmark results (interview-ready insights)

- Explain differences between intrinsic metrics (perplexity) and extrinsic task metrics (accuracy, pass@k) and why lowering PPL may not improve instruction following.
- Discuss data contamination: how to detect it (n-gram overlap, watermarking, provenance checks) and why it invalidates benchmark claims.
- Preference evaluations require proper randomization, rater guidelines, and inter-annotator agreement reporting (Cohen’s kappa or Krippendorff’s alpha).
- For code evals, unit-test-based pass@k is stronger than BLEU; emphasize sandboxed deterministic execution and handling of nondeterminism/time limits.
- For multi-model comparisons, use paired bootstrap tests and report effect sizes and CIs, not just p-values.

### Practical checklist for running a trustworthy benchmark

1. Curate dataset and confirm no training leakage.
2. Fix evaluation protocol: prompt templates, few-shot seeds, decoding params (temperature, max tokens, top_p/k), and sampling seeds.
3. Run multiple seeds and compute CI via bootstrap.
4. Report per-task metrics, slices (length, domain), and worst-percentiles.
5. Include human evaluation for subjective attributes (helpfulness, style, factuality).
6. Release code and exact prompts for reproducibility.

### Practice exercises

Reproduce GLUE subset

- Task: pick three GLUE tasks, run two models (e.g., bert-base and roberta-base) using identical evaluation pipelines, compute per-task accuracy and bootstrap CIs, and present a dashboard of results and interpretation.

MMLU few-shot ablation

- Task: evaluate a causal LM on MMLU with 0-, 1-, and 5-shot prompts. Report accuracy per subject area and analyze which subjects gain most from more shots.

HumanEval pass@k

- Task: fine-tune a small code model on MBPP, generate 100 completions per problem, run unit tests in a sandbox, compute pass@1, pass@5, pass@10, and measure time per sample.

Robustness slicing

- Task: for an instruction model, create 5 adversarial prompt transformations (typos, negation flip, misleading preface). Measure accuracy/hallucination rate per slice and propose data-augmentation strategies that close the gap.

Benchmark reproducibility report

- Task: document a full reproducible evaluation for one benchmark: code, seeds, prompt templates, decoding hyperparameters, dataset versions, and CI results. Publish as a short report with visualizations.

---

## Parameter Efficient Fine-Tuning (PEFT)

### Direct definition

Parameter-efficient fine-tuning (PEFT) refers to methods that adapt a pretrained model to a new task by updating only a small subset of parameters or by adding a small number of trainable parameters while keeping the bulk of the pretrained weights frozen, achieving strong task performance with far lower compute, storage, and memory cost than full fine-tuning.

### Concept intuition

- What it is: instead of changing every weight in a large model, PEFT either (a) adds tiny trainable modules (adapters, LoRA matrices, prompt vectors) or (b) updates just a few existing parameters (Bias/LayerNorm/embedding deltas). The pretrained backbone remains frozen.
- Why it matters for instruct/LLMs: models get huge; full fine-tuning is expensive and impractical for many tasks or many task endpoints. PEFT lets you maintain many task-specific behaviors cheaply (small checkpoints), reduces catastrophic forgetting, and speeds up training and deployment.
- Visual analogy: think of a neural network as a large orchestra; PEFT adds a few specialist musicians or adjusts a conductor’s subtle gestures to change the performance instead of replacing the whole orchestra.
- Typical gains: orders-of-magnitude fewer trainable parameters (e.g., 0.1%–3% of params), smaller checkpoints, often comparable performance to full fine-tuning when applied correctly.

### Main PEFT methods

- LoRA (Low-Rank Adaptation): add low-rank updates to key weight matrices (Q/K/V/Proj), training small A and B matrices so Delta_W = A @ B. Widely used for causal LMs and instruction fine-tuning.
- Adapters: small MLP blocks inserted between layers; only adapter parameters are trained. Common in encoder-decoder models.
- Prefix Tuning / Prompt Tuning: learn virtual token embeddings prepended to the input (prefix) or only train special prompt vectors; good for conditional generation and few-shot adaptation.
- BitFit / Bias-only tuning: only train bias terms (and sometimes LayerNorm) — extremely cheap, sometimes surprisingly effective for classification.
- Compacter / (Kronecker) Factored methods: compressors for adapter-like layers using structured low-rank parametrizations.
- Mixture: combine LoRA + adapters or bias-only + prefix for complementary behavior.

### Mathematical breakdown (clean code-style formulas)

Low-Rank update (LoRA)

```
Given a pretrained weight W ∈ R^{d_out × d_in},
We parameterize an additive update:
Delta_W = A @ B
where A ∈ R^{d_out × r}, B ∈ R^{r × d_in}, r << min(d_out, d_in)

Effective trainable params per matrix ≈ r*(d_out + d_in)
Full params per matrix = d_out * d_in
```

Adapter block (bottleneck MLP)

```
Given hidden x ∈ R^{d},
Adapter:  x' = x + s * Up( ReLU(Down(x)) )
Down: R^{d} -> R^{m}, Up: R^{m} -> R^{d}, with m << d
Trainable params ≈ d*m + m*d = 2*d*m
s: optional scalar (trainable or fixed) for scaling residual
```

Prefix tuning (continuous prompts)

```
Learn P ∈ R^{L_prefix × d_model} (virtual token embeddings)
At encoder/decoder attention, treat P as extra key/value (or key/value/query) vectors.
Trainable params ≈ L_prefix * d_model
```

Bias-only (BitFit)

```
Only bias vectors b of linear layers are trainable.
Update: W x + b -> W x + (b + delta_b) with delta_b trainable
Trainable params: sum of bias sizes
```

1. Combined objective (when mixing PEFT + SFT)

```
L_total = (1/N) * sum_{i=1..N} L_task(model(x_i; theta_frozen, phi_trainable), y_i)
# Only phi_trainable (PEFT params) are optimized via gradient descent
phi <- phi - eta * grad_phi(L_total)
```

### Code & practical application

Default: PyTorch + Hugging Face + PEFT library (peft). Two runnable examples: LoRA for a causal LM, and Adapters for an encoder-decoder. Use small models for local runs.

A. LoRA on a causal LM (GPT-2 small) — full pipeline

```python
# pip install transformers accelerate datasets peft bitsandbytes
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model (optionally quantize or prepare for k-bit)
model = AutoModelForCausalLM.from_pretrained(model_name)
# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  # check module names for your model
    lora_dropout=0.05,
    bias="none",  # or "all" to train biases too
    inference_mode=False,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Dataset: tiny toy data
ds = load_dataset("json", data_files={"train":"toy_instr.jsonl"})
def build_prompt(ex):
    prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex.get('input','')}\n\n### Response:\n"
    input_ids = tokenizer(prompt + ex["output"], truncation=True, max_length=512).input_ids
    # mask prompt tokens for loss
    prompt_len = len(tokenizer(prompt).input_ids)
    labels = [-100]*prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "labels": labels}

ds = ds["train"].map(build_prompt)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./lora_ft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,  # LoRA can often use larger lr than full fine-tune
    logging_steps=10,
    fp16=False
)
trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
trainer.train()

# Save tiny PEFT checkpoint (saves only LoRA weights)
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
```

B. Adapters (using AdapterHub-like pattern via PEFT or direct insertion)

```python
# pip install transformers peft datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, get_peft_model, LoraConfig  # adapters may need a different config in peft; many libs implement adapters
# For illustration we'll use LoRA-like adapter in seq2seq style:
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

adapter_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, adapter_config)
# proceed similarly with Trainer and DataCollatorForSeq2Seq
```

C. BitFit (bias-only) quick example

```python
model_name = "distilbert-base-uncased"
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# freeze all params
for p in model.parameters():
    p.requires_grad = False
# unfreeze biases and LayerNorms
for name, param in model.named_parameters():
    if "bias" in name or "LayerNorm" in name:
        param.requires_grad = True
# optimizer only sees trainable params
```

Practical tips:

- Inspect target_modules and layer names for your model; HF models have differing naming conventions.
- LoRA hyperparameters: r (rank, common 4–32), alpha (scaling), dropout (0–0.2). Larger r improves capacity but raises trainable param count.
- Save only adapter/LoRA params for lightweight checkpoints. PEFT library handles this with model.save_pretrained.
- LoRA often tolerates higher LR than full FT; tune on a small validation split.
- For multi-task instruction models, prefix tuning + LoRA can be complementary.

### Visualization / Geometry

- Low-rank interpretation: LoRA constrains updates to lie in an r-dimensional subspace per weight matrix. Visualize Delta_W's singular value spectrum: most energy concentrated in r modes.
- Adapter manifold: adapters add a small residual mapping that nudges hidden states into task-specific subspaces; visualize by projecting pooled hidden states before vs after adapter output with t-SNE/UMAP.
- Activation-wise view: freeze base model, forward inputs and inspect how small adapter outputs shift activations:
    - Plot per-layer cosine similarity between frozen-only activations and adapted activations to see where adapters exert influence.
- Gradient focus: plot gradient norms over PEFT params vs frozen model (should be zero) to confirm only intended params update.

Snippet to compute per-layer cosine shift after PEFT:

```python
import torch
from sklearn.metrics.pairwise import cosine_similarity
model.eval()
# get hidden states before applying PEFT (simulate by loading base model state)
# for simplicity, compute on a batch
inputs = tokenizer(["Example input text"], return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    out_adapt = model(**inputs, output_hidden_states=True)
# assume base_out obtained from base model (without PEFT)
# compute mean pooled hidden state per layer and cos similarity
def pool(h):
    return h.mean(dim=1).cpu().numpy()
for layer_idx, (h_base, h_adapt) in enumerate(zip(base_out.hidden_states, out_adapt.hidden_states)):
    v_base = pool(h_base)
    v_adapt = pool(h_adapt)
    sim = cosine_similarity(v_base, v_adapt).diagonal().mean()
    print(f"Layer {layer_idx} mean cos sim: {sim:.4f}")
```

### Common pitfalls & tips

- Wrong target_modules names: mismatch between LoRA target module names and model internals yields silent failures; check model.named_modules() to locate correct keys.
- Forgetting to set model.config.use_cache=False during training with past_key_values for some PEFT methods; caching can break gradient flow.
- Choosing rank r too low/too high: too low underfits task; too high negates PEFT benefit. Start with r in {4,8,16} and validate.
- Learning rate scale: LoRA and prefix tuning often use larger LRs than full fine-tune; tune carefully with small validation set (warmup helps).
- Not freezing base params: ensure all base parameters are frozen except intended ones (PEFT wrappers usually do this, but confirm).
- Incompatible quantization: some quantized-loading paths require special preparation (prepare_model_for_kbit_training). Mix peft + 4-bit quantization carefully.
- Evaluation mismatch: ensure inference loads adapter weights (get_peft_model with correct config) and that generation uses same decoding hyperparameters used in validation.

### Interview-ready insights

- Why PEFT works: pretrained layers encode rich priors; small, structured parameter additions or low-rank deltas can steer those priors for new tasks without rewriting the core representations. This is efficient because many downstream tasks require only low-dimensional adjustments.
- LoRA vs Adapters vs Prompt Tuning: LoRA modifies linear maps implicitly (like a learned low-rank delta) and works well for generative LMs; adapters explicitly change activations via small MLPs and are more modular; prefix/prompt tuning changes input-space conditioning and can be optimal for few-shot with frozen encoder/decoder. Choice depends on task type, compute, and deployment constraints.
- Storage & deployment: PEFT allows shipping tiny checkpoints per task (MBs), avoiding storage duplication of full models for every task. Orchestrate model + adapter loading at inference time for multi-task endpoints.
- When not to use PEFT: extremely divergent tasks requiring massive internal rewiring may need full fine-tuning. Also, if you want simple reproducibility and minimized engineering complexity, full fine-tune is straightforward but costly.
- Combining methods: LoRA + bias-only updates often beat single methods. For extreme low-storage, bias-only or prompt tuning can be practical baselines.

### Practice exercises (progressive)

LoRA basic (easy)

- Task: Fine-tune GPT-2 small with LoRA (r=8) on a 500-example instruction dataset. Compare trainable param count and saved checkpoint size vs full fine-tune.
- Hints:
    - Count trainable params: sum(p.numel() for p in model.parameters() if p.requires_grad)
    - LoRA checkpoints saved by peft are small (MBs). Verify generation on held-out prompts.

Adapter vs LoRA ablation (medium)

- Task: Train two models on the same seq2seq summarization subset: (A) adapters (bottleneck m=64), (B) LoRA (r=16). Compare validation ROUGE, training speed, and memory usage.
- Hints:
    - Use same training loop and batch size; track GPU memory with nvidia-smi or torch.cuda.max_memory_allocated().

BitFit baseline (easy → medium)

- Task: Implement BitFit for a classification head and compare performance to full fine-tune and LoRA for small data regime (e.g., 100–1000 examples). Report when BitFit suffices.
- Hints:
    - Freeze base model, unfreeze biases & LayerNorms, train classifier head.

PEFT + k-bit quantization (advanced)

- Task: Load a QLoRA-style 4-bit base model (prepare_model_for_kbit_training), attach LoRA, fine-tune on instruction data, and evaluate inference memory and latency vs non-quantized LoRA.
- Hints:
    - Use prepare_model_for_kbit_training from peft/transformers; ensure compatibility of quantization library (bitsandbytes).

Multi-task adapters with routing (advanced)

- Task: Create per-task adapter modules (one small adapter per task) and implement runtime routing to load the correct adapter per incoming task token. Fine-tune on a multi-task instruction dataset and measure per-task performance and adapter size.
- Hints:
    - Prepend a task-id token and use a simple mapping to swap adapter weights at inference or implement conditional adapter enabling.

---

## PEFT Techniques 1: LoRA

### Direct definition

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adapts pretrained transformer weights by learning small, low-rank additive updates to selected dense weight matrices instead of updating the full weight matrices. The base model weights remain frozen and only the low-rank factors are trained, drastically reducing number of trainable parameters and checkpoint size while retaining strong performance.

### Concept intuition

- What LoRA does: for a target weight matrix W (e.g., the query projection in attention), LoRA models the change as an additive low-rank matrix Delta_W = A @ B where A and B are small (rank r). During training you optimize A and B; at inference you either apply Delta_W to W or compute outputs by injecting the low-rank update on the fly.
- Why it matters: large LLMs have huge weight matrices; many downstream tasks require only low-dimensional changes to steer model behavior. LoRA constrains learning to a low-dimensional subspace that is expressive enough for adaptation but cheap to store and compute.
- Visual/analogy: imagine W as a high-dimensional map of behaviors. Instead of carving a completely new map, LoRA draws a few guiding vectors (A columns) and small linear combinations (B rows) that nudge the model in the required directions — like adding a few steering vectors rather than rewriting the whole control panel.
- When to pick LoRA: large causal LMs or encoder-decoder models where you want strong adaptation with tiny checkpoints, fast training, and low memory overhead. LoRA is a default for instruction tuning, multi-task adapters, and QLoRA-style quantized fine-tuning.

### Mathematical breakdown (copy-paste-friendly)

Base linear layer

```
Given input x ∈ R^{d_in}, weight W ∈ R^{d_out × d_in}, bias b ∈ R^{d_out},
the layer computes: y = W x + b
```

LoRA parameterization

```
We parameterize an additive low-rank update:
Delta_W = A @ B

A ∈ R^{d_out × r}
B ∈ R^{r × d_in}
r << min(d_out, d_in)
```

Forward with LoRA (training or merged)

```
y = (W + alpha/r * A @ B) x + b
# often scaled by alpha/r to control update magnitude
```

Trainable parameter counts

```
Trainable params per matrix ≈ r * (d_out + d_in)
Full params per matrix = d_out * d_in
```

Loss & optimization (standard supervised SFT)

```
L = (1/N) * sum_i loss(model(x_i; W + Delta_W), y_i)
Update only A, B (and optionally small bias terms)
```

Merging for inference (optional)

```
W_merged = W + alpha/r * A @ B
# then only W_merged is used; LoRA factors can be discarded if desired
```

Notes:

- Typical hyperparameters: r ∈ {4,8,16,32}, lora_alpha (scaling) ∈ {8,16,32}, lora_dropout ∈ [0.0,0.2].
- Target modules: for attention you typically target query/key/value/output projection matrices (names differ by model implementation).

### Code & practical application (PyTorch + Hugging Face + PEFT)

A. Minimal end-to-end example: apply LoRA to a causal LM, train on a tiny instruction dataset, run inference. This is copy-paste ready for a notebook.

```python
# pip install transformers datasets accelerate peft
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

model_name = "gpt2"  # switch to larger model for real runs
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
# ---------- LoRA config ----------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],  # check your model's module names
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
# ----------------------------------

# tiny toy dataset: JSONL with fields instruction, input, output
ds = load_dataset("json", data_files={"train":"toy_instr.jsonl"}, split="train")

def format_example(ex, max_len=512):
    instr = ex.get("instruction","")
    inp = ex.get("input","")
    out = ex.get("output","")
    if inp:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    out_ids = tokenizer(out, add_special_tokens=False).input_ids
    input_ids = prompt_ids + out_ids
    # labels: -100 for prompt so loss computed only on output
    labels = [-100] * len(prompt_ids) + out_ids
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]
    return {"input_ids": input_ids, "labels": labels}

ds = ds.map(format_example)
# pad in collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./lora_example",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator
)
trainer.train()

# Save only LoRA adapters (small)
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

# Inference: load base model and adapter
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("./lora_adapter")
model_peft = PeftModel.from_pretrained(base, "./lora_adapter").to("cuda")

prompt = "### Instruction:\nTranslate to French\n\n### Input:\nI love you.\n\n### Response:\n"
ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
gen = model_peft.generate(ids, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(gen[0], skip_special_tokens=True))
```

B. Practical tips for real runs

- Inspect model.named_modules() to find correct target_modules names (they differ by architecture and HF implementation).
- Use peft.get_peft_model which wraps the model and freezes base params automatically.
- Typical LR: LoRA often uses higher LR than full-finetune (e.g., 1e-4–5e-4) but validate on a small held-out set; use warmup steps.
- Use gradient accumulation and mixed precision for bigger models.
- Save and ship only the PEFT adapter directory — tiny and fast to load.

### Visualization / Geometry (how LoRA acts in representation space)

- Low-rank subspace: Delta_W = A @ B has rank ≤ r, so updates lie in an r-dimensional subspace of the full weight matrix space. Visualize by computing singular values of Delta_W; most energy concentrates in top r singular values.
- Activation shifts: compute hidden-state vectors before and after applying LoRA; measure cosine similarity per-layer to see where LoRA influences activations most (usually middle-to-upper layers for instruction tuning).
- t-SNE/UMAP: pool final-layer hidden states for a set of prompts and plot before vs after LoRA training; clusters for different instruction types often become more separable after adaptation.
- Per-parameter contribution: compute ||A @ B||_F per module to see which modules receive the largest adaptation energy.

Quick snippet: singular values of Delta_W (if you merged)

```python
import torch, numpy as np
W = model.base_model.transformer.h[0].attn.c_attn.weight.data.cpu()  # example path
# if merged: Delta = W_merged - W_orig
Delta = (W_merged - W_orig).cpu().numpy()
s = np.linalg.svd(Delta, compute_uv=False)
print("Top singular values:", s[:10])
```

### Common pitfalls & tips

- Wrong target module names: LoRA requires targeting the correct linear layers; mismatches lead to no effect. Inspect module names with `for n,m in model.named_modules(): print(n)` before config.
- Forgetting to freeze base model: get_peft_model usually freezes base params, but verify trainable params count to ensure only LoRA factors are trainable.
- Rank r selection: too small r underfits; too large reduces parameter-efficiency benefits. Start with r=8 or 16, compare validation loss and adapter size.
- Scaling alpha mismatch: default scaling alpha/r matters; tuning alpha helps control update magnitude. Use alpha ≈ r as a rule-of-thumb scaling.
- Dropout & regularization: use modest lora_dropout (0–0.1) on small noisy datasets to prevent overfitting.
- Quantization interactions: when combining LoRA with k-bit quantization (QLoRA), call prepare_model_for_kbit_training and follow library guidelines — otherwise gradients can break.
- Inference merging: merging LoRA into W reduces runtime overhead but prevents cheap switching between adapters for different tasks; keep adapters separate for multi-task endpoints.

### Interview-ready insights

- Why LoRA is effective: pretrained transformers are highly overparameterized; task-specific changes frequently inhabit a low-dimensional manifold. LoRA directly models that manifold with far fewer parameters, matching the inductive bias of small, focused adaptations.
- Trade-offs vs full fine-tune: LoRA gives huge storage and training efficiency gains and often matches or nearly matches full fine-tuning, but extreme domain shifts or tasks needing deep reconfiguration might still benefit from full-finetune.
- LoRA vs adapters vs prefix tuning: LoRA modifies internal linear maps (powerful for attention-heavy models), adapters alter activations via bottleneck MLPs (modular, layer-local), and prefix tuning conditions via learned context tokens (works well for frozen encoder-decoders). Choose by task type, deployment needs, and whether you need per-task modularity.
- Hyperparameters to report: r, lora_alpha, lora_dropout, target_modules, learning_rate, number of trainable params. These determine performance, checkpoint size, and generalization.

### Practice exercises

LoRA sanity check (easy)

- Task: Apply LoRA (r=8) to GPT-2 small and fine-tune on 200 short instruction examples for 1 epoch. Measure trainable param count and verify generation quality on 10 held-out prompts.
- Hint: Count trainable params with sum(p.numel() for p in model.parameters() if p.requires_grad).

Rank ablation (medium)

- Task: Train LoRA with r ∈ {4,8,16,32} on same dataset. Plot validation loss vs r and checkpoint size vs r. Interpret trade-offs and pick a recommended r for that data size.
- Hint: validation loss improvements will often saturate; choose smallest r with near-peak performance.

Target-module selection (medium)

- Task: Try targeting only query and value projections (q_proj, v_proj) vs targeting all projection matrices (q,k,v,o). Compare performance and training cost. Report which modules are most beneficial for instruction tuning on your dataset.
- Hint: inspect named_modules to map projection naming for your model.

LoRA + Quantization (advanced)

- Task: Implement QLoRA: load a 4-bit quantized model (prepare_model_for_kbit_training), attach LoRA, fine-tune on instructions, and compare memory usage and final performance to full 16-bit LoRA.
- Hint: use bitsandbytes + peft integration; follow recommended prepare steps to avoid gradient issues.

Multi-adapter switching (advanced)

- Task: Train separate LoRA adapters for 3 tasks (summarize, translate, QA). Implement runtime adapter switching based on a task token and evaluate per-task performance compared to a single multi-task LoRA trained jointly.
- Hint: save each adapter folder separately; load desired adapter into the base model with PeftModel.from_pretrained(base, adapter_path).

---

## PEFT Techniques 2: soft prompts

### Direct definition

Soft prompt tuning (a form of prompt-based PEFT) learns a small set of continuous prompt vectors (virtual tokens) that are prepended or injected into a frozen pretrained model; only those prompt vectors are trained while the model weights remain fixed, enabling task adaptation with very few parameters.

### Concept intuition

- What it is: instead of writing a natural-language prompt, soft prompts are learned embeddings that act like “virtual” prompt tokens the model attends to. They change the model’s behavior by shifting internal context without altering core weights.
- Why it matters: extremely parameter-efficient (often kilobytes to a few MBs), easy to store/swap, and particularly useful when you must keep the backbone frozen (e.g., shared large models or restricted compute).
- Analogy: imagine giving a musician a short piece of sheet music that subtly changes how they play the whole orchestra; soft prompts are that short sheet—tiny but able to steer the full performance.
- Variants: prefix-tuning (learn keys/values per attention layer), prompt-tuning (learn input token embeddings only), and P-Tuning/optimus-style layerwise prompts. Difference is where and how the learned vectors are injected.

### Mathematical breakdown (copy-paste friendly)

Basic prompt-injection forward

```
Let E: token -> embedding matrix (vocab_size × d).
A soft prompt P ∈ R^{L_p × d} is a trainable matrix (L_p virtual tokens).

For an input token sequence x = [x1..xT] with embeddings E(x),
we form input embeddings:
  X' = concat(P, E(x))  # shape (L_p + T, d)

Model processes X' through transformer layers unchanged.
Only P is updated by gradient descent; model parameters θ are fixed.
```

Loss and update

```
Given supervised loss L_task on model outputs y_hat:
  L = L_task(y_hat, y)
Update rule for soft prompts (phi):
  phi <- phi - eta * grad_phi(L)
No gradients update model parameters θ.
```

Prefix tuning (layerwise key/value)

```
Instead of only input embeddings, learn per-layer prefix key/value matrices:
  For layer l, Prefix_l: K_l ∈ R^{L_k × d_k}, V_l ∈ R^{L_v × d_v}
At attention, treat (K_l, V_l) as additional key/value entries the decoder/encoder can attend to.
Trainable params ≈ sum_l (L_k*d_k + L_v*d_v)
```

Parameter count (rough)

```
Prompt-tuning trainable params ≈ L_p * d
Prefix-tuning trainable params ≈ num_layers * L_prefix * d_per_layer
Typical L_p: 10–100; d: 512–2048 => small footprints.
```

### Code & practical application (PyTorch + Hugging Face; prompt-tuning and prefix example)

A. Prompt-tuning (learn virtual token embeddings prepended to input — simple)

```python
# pip install transformers datasets accelerate
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
model.eval()  # backbone stays frozen

# Soft prompt (trainable)
L_p = 20  # number of virtual tokens
d = model.config.d_model
soft_prompt = torch.randn(L_p, d, requires_grad=True, device="cuda")

# Helper: prepend soft prompt to encoder input embeddings and run the model
def encode_with_soft_prompt(input_texts):
    enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_ids = enc.input_ids  # we won't use these embeddings for the prefix
    # get input embeddings from model's embedding layer
    inputs_embeds = model.get_encoder().embed_tokens(input_ids)  # (batch, T, d)
    b = inputs_embeds.size(0)
    prompt_expanded = soft_prompt.unsqueeze(0).expand(b, -1, -1)  # (b, L_p, d)
    inputs_with_prompt = torch.cat([prompt_expanded, inputs_embeds], dim=1)  # (b, L_p+T, d)
    attention_mask = torch.cat([torch.ones(b, L_p, device="cuda"), enc.attention_mask], dim=1)
    return {"inputs_embeds": inputs_with_prompt, "attention_mask": attention_mask, "labels": None}

# Training loop skeleton (supervised)
optimizer = torch.optim.AdamW([soft_prompt], lr=5e-4)
for epoch in range(3):
    for batch in dataset:  # assume small dataset yields raw input_texts and targets
        model.eval()  # backbone frozen, but still compute gradients wrt soft_prompt
        enc = encode_with_soft_prompt(batch["input_texts"])
        labels = tokenizer(batch["targets"], return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        outputs = model(**enc, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

B. Prefix-tuning (learn keys/values per layer example sketch)

- Implementing full prefix requires hooking into each attention layer to prepend learned key/value tensors or supply past_key_values to decoder layers. Many libraries implement this; below is conceptual sketch (use peft or a prefix-tuning library in practice).

Notes:

- Use batching and gradient_accumulation for larger L_p.
- Often freeze model gradients to save memory and wrap optimizer only around prompt parameters.

C. Libraries

- Use "peft" or "transformers" extensions that implement prompt/prefix tuning to avoid low-level hooking. E.g., peft supports prefix/prompt configs similar to LoRA.

### Visualization / Geometry (intuition and diagnostics)

- Prompt embedding drift: visualize soft-prompt vectors projected into PCA/UMAP along with token embeddings to see whether prompt vectors occupy distinct regions or move toward task-specific token clusters.
- Activation shifts: measure cosine similarity between hidden states with and without the trained soft prompt to see how early layers vs late layers are influenced. Soft prompts often steer initial activations which then propagate.
- Attention patterns: for prefix-tuning (layerwise K/V), inspect attention weights to/from injected prefix entries — they act like global context tokens, and you can visualize which input tokens rely on prefixes.
- Decision boundary effect: for classification, compare class-separability of pooled encoder outputs with and without the soft prompt via t-SNE.

Quick snippet: compare cosine shift of encoder pooled outputs

```python
# compute pooled states with and without soft prompt for same inputs
with torch.no_grad():
    base_out = model.get_encoder()(input_ids=base_input_ids, attention_mask=base_attention_mask, output_hidden_states=True)
    with_prompt = model.get_encoder()(inputs_embeds=inputs_with_prompt, attention_mask=mask_with_prompt, output_hidden_states=True)
base_pooled = base_out.last_hidden_state.mean(dim=1).cpu()
prompt_pooled = with_prompt.last_hidden_state.mean(dim=1).cpu()
cos_sim = torch.nn.functional.cosine_similarity(base_pooled, prompt_pooled, dim=-1).mean().item()
print("Mean pooled cosine similarity shift:", 1.0 - cos_sim)
```

### Common pitfalls & tips

- Prompt length L_p trade-off: too short may underfit; too long increases params and overfits. Start small (10–50) and validate.
- Initialization matters: random initialization works, but initializing soft prompts from token embeddings of helpful seed phrases (e.g., "Summarize:") can speed convergence.
- Where to inject: prompt in encoder input is common for seq2seq; for causal LMs you may prepend to token embeddings or inject into key/value per layer (prefix). Architecture choice affects expressivity.
- Frozen model assumptions: soft prompts only steer existing representations; if base model lacks capacity for target behavior, prompts cannot fully compensate.
- Attention masking: when using inputs_embeds and prepended prompts, ensure attention_mask aligns (ones for prompt positions) so prompts are attended to.
- Overfitting to prompt format: if training always uses one natural-language instruction format, model may not generalize to other phrasings; mix templates and paraphrases.
- Transfer limitations: soft prompts may not generalize across very different domains/tasks; adapters or LoRA might be better for deep rewiring tasks.

### Interview-ready insights

- When to use soft prompts:
    - Very low-storage per-task budget (many tasks/endpoints).
    - When the backbone must remain strictly frozen for compliance or infrastructure reasons.
    - Rapid, low-cost experimentation with new behaviors.
- Limitations compared to LoRA/adapters:
    - Fewer trainable params and cheaper, but often less expressive; LoRA and adapters usually match or exceed prompt performance on complex tasks.
- Relationship to classic prompting:
    - Soft prompts learn a continuous embedding analogue of hand-crafted prompts — they can encode subtler instruction signals inaccessible to discrete prompts.
- Practical hyperparameters to report:
    - L_p (prompt length), injection point (encoder input vs decoder vs per-layer), learning rate, initialization method, and trainable param count.
- Deployability:
    - Soft prompts are extremely easy to swap at inference (just load small vectors) making them ideal for multi-tenant services where one large frozen model serves many tasks.

### Practice exercises

Prompt-tuning baseline (easy)

- Task: Implement prompt-tuning for flan-t5-small with L_p=20 on a small summarization set. Train only the prompt vectors and compare ROUGE with a small-head-only fine-tune.
- Hint: initialize soft prompt randomly and use a higher LR (e.g., 5e-4). Compare trainable param counts.

Initialization ablation (medium)

- Task: Compare three initializations for soft prompt: random, mean of embeddings for "summarize:", and embeddings of several manual seed tokens. Report convergence speed and final validation loss.
- Hint: use same optimizer and seeds for fair comparison.

Prefix-tuning attention inspection (medium)

- Task: Implement prefix-tuning (layerwise K/V prefixes) for a small encoder-decoder. Train on a translation subset and, at inference, visualize cross-attention matrices showing where prefix keys get attended.
- Hint: inspect decoder cross-attention weights and add prefix token labels on the axis for clarity.

Prompt length vs generalization (advanced)

- Task: Train prompt-tuning models with L_p ∈ {10, 20, 40, 80}. Test on in-distribution and out-of-distribution instructions (format paraphrases, domain shifts). Plot performance vs L_p and identify diminishing returns point.
- Hint: measure both task metric and a robustness metric (performance on paraphrased prompts).

Hybrid prompt + LoRA comparison (advanced)

- Task: Train (A) soft prompt only, (B) LoRA only (small r), and (C) prompt+LoRA. Compare metrics, train time, and stored checkpoint sizes. Conclude on trade-offs for your dataset.
- Hint: keep total trainable params in (A) and (B) roughly matched to compare efficiency.

---

## Fine tune a generative AI model for dialogue summarization

### Direct definition

Fine-tuning for dialogue summarization is the supervised process of adapting a pretrained sequence-to-sequence or causal language model to map multi-turn conversational inputs into concise, coherent summaries that capture key facts, decisions, and action items while preserving speaker intents and dialogue structure.

### Concept intuition

- What the task is: compress a conversation (chat, meeting, support call) into a short summary that is faithful, non-hallucinated, and structured (bullet points, decisions, or short paragraphs).
- Why it is different from regular summarization: dialogues have speaker turns, disfluencies, pronouns, interruptions, and often implicit context. The model must handle coreference, turn-level importance, and extractive vs abstractive trade-offs.
- Real-world use-cases: meeting notes, customer support summaries, CRM auto-filling, compliance logs.
- High-level design choices:
    - Model family: encoder-decoder (T5, FLAN-T5, BART) often easier for conditional summarization; causal LMs (GPT-family) work well with instruction prompts and SFT/LoRA.
    - Output format: label-oriented (bullets, action items) reduces hallucination and eases evaluation.
    - Data: supervised examples mapping raw dialogue -> gold summary; augment with role-aware templates, turn markers, and synthetic data (dialogue-to-summary) if scarce.
- Deployment constraints: latency, model size, ability to update adapter weights for new domains, and safety filters to avoid leaking PII.

### Mathematical breakdown

Problem formulation

- Input dialogue D is a sequence of tokens representing turns: D = [T_1, T_2, ..., T_M], where a turn T_j = (speaker_j, u_{j,1..n}). Target summary S is a token sequence y_1..y_T.

Supervised objective (seq2seq cross entropy)

```
L = - sum_{t=1..T} log p_theta(y_t | encoder(D), y_{1:t-1})
```

- theta: model parameters (or only PEFT params if using adapters/LoRA/prefix tuning).

Masking and loss weighting for long inputs

- If we chunk long dialogues, let segments s_1..s_K with targets S_1..S_K then total loss:

```
L_total = (1/K) * sum_{k=1..K} L(S_k | s_k)
```

- Optionally weight recent turns or decisions:

```
L_total = (1/K) * sum_k w_k * L(S_k | s_k)
```

with w_k > 1 for higher-priority chunks.

Evaluation metrics

- ROUGE variants for n-gram overlap:

```
ROUGE-1, ROUGE-2, ROUGE-L = standard library computations
```

- BERTScore for semantic overlap, and factuality metrics (QA-based support rate).
- For end-use, human preference and action item extraction precision/recall matter more than raw ROUGE.

Fine-tune with PEFT

- If using LoRA, parameterize selected weight matrices with Delta_W = A @ B and only optimize A,B:

```
theta = theta_frozen ∪ phi_trainable
phi = {A_i, B_i}
Update phi via gradient descent minimizing L.
```

### Code and practical application

I give a full runnable pipeline skeleton for a T5-style encoder-decoder fine-tune (Hugging Face Trainer) plus a LoRA example for efficiency on larger models. Replace toy file paths with your data.

A. Data format and preprocessing

- JSONL example line:

```json
{"id":"meeting-001","dialogue":"[Alice] Hi team... \n[Bob] We decided to ...","summary":"Decisions: Adopt X. Action items: Alice to draft plan."}
```

- Tokenization and dataset prep

```python
# pip install transformers datasets evaluate peft
from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "google/flan-t5-base"  # or "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def make_prompt(example):
    # keep speaker tokens intact to aid grounding
    dialogue = example["dialogue"].strip()
    # explicit instruction helps generalization
    prompt = "Summarize the dialogue. List decisions and action items if any.\n\nDialogue:\n" + dialogue + "\n\nSummary:\n"
    return prompt

def preprocess_function(examples, max_input_len=512, max_target_len=128):
    prompts = [make_prompt(x) for x in examples["dialogue"]]
    model_inputs = tokenizer(prompts, max_length=max_input_len, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=max_target_len, truncation=True, padding="max_length")
    # replace pad token id in labels with -100 for HF Trainer
    labels_input_ids = labels["input_ids"]
    labels_input_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in lab] for lab in labels_input_ids]
    model_inputs["labels"] = labels_input_ids
    return model_inputs

ds = load_dataset("json", data_files={"train":"dialogue_train.jsonl","validation":"dialogue_val.jsonl"})
ds = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)
ds.set_format(type="torch")
```

B. Training with Hugging Face Trainer (encoder-decoder)

```python
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import evaluate

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
rouge = evaluate.load("rouge")

def compute_metrics(pred):
    preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    refs = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    return rouge.compute(predictions=preds, references=refs)

training_args = TrainingArguments(
    output_dir="./dialogue_summarizer",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
```

C. LoRA variant for large models (efficient)

```python
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

model_name = "google/flan-t5-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],  # adapt names for T5 implementation
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
# then use Trainer as above; only LoRA params updated and saved_small checkpoints
```

D. Inference and decoding strategies

- For faithfulness, use constrained decoding and generate length limits and n-gram blocking for verbatim hallucination control.

```python
# greedy or beam
inputs = tokenizer("Summarize the dialogue...\n\nDialogue:\n" + dialogue, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=120, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
summary = tokenizer.decode(out[0], skip_special_tokens=True)
# sampling for diversity if needed: do_sample=True, top_p=0.9, temperature=0.8
```

E. Factuality QA-based post-check

- Convert summary claims into questions and check support against dialogue using an extractive QA model; if unsupported, flag or shorten summary.

```python
# outline: generate questions per sentence using a QG model, answer on dialogue with a QA model, compare answers.
# Use HF pipelines for question-generation and question-answering in small-scale prototyping.
```

### Visualization and geometry

Token-level loss heatmap

- Plot per-token loss for dialogue->summary pairs to find tokens/positions with high loss (speaker tokens, dates, rare named entities).

Hidden-state cluster plots

- Pool encoder last hidden states per dialogue and plot UMAP coloring by dialogue type (sales call vs engineering meeting). After fine-tuning, summaries of similar meeting types should cluster.

Attention inspection

- For encoder-decoder, visualize decoder cross-attention during summary token generation to check it attended to relevant turns and speaker utterances. Plot heatmap with decoder tokens on y-axis and dialogue tokens on x-axis.

Decision / action extraction alignment

- For outputs formatted as lists (Decisions:, Actions:), inspect token-level alignment to where those phrases map back into the input (use attention or token overlap) to debug missing action items.

Quick code to extract cross attention from HF model during generation

```python
# ensure model config output_attentions=True
model.config.output_attentions = True
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, output_attentions=True, return_dict_in_generate=True, num_beams=1, max_new_tokens=80)
# outputs.cross_attentions is nested per step and layer depending on model; some HF models return attentions in generative path only for certain APIs.
# Alternatively run a forced decode with model.generate then run model(decoder_input_ids=...) with output_attentions=True to inspect attentions.
```

### Common pitfalls and tips

- Input length truncation: dialogues can be long. Options: chunk with sliding window + hierarchical summarization, retrieveive preselection of salient turns, or use long-context models (Longformer, BigBird, or extended T5 variants).
- Speaker information loss: always keep speaker labels or special tokens like [Alice] to preserve referents and reduce hallucination on who did what.
- Hallucination and factuality: summarization models tend to hallucinate; reduce by using extractive supervision, constraints, or post-checkers (QA support rate).
- Small data overfitting: data augmentation (paraphrase summaries, backtranslation), label smoothing, and PEFT help reduce overfitting.
- Evaluation misalignment: ROUGE correlates weakly with user satisfaction. Use BERTScore, factuality probes, and human evaluation focusing on action items and correctness.
- Format brittleness: training with multiple output templates (bullets, paragraphs, short/long) makes the model robust to requested styles.
- Privacy and PII leakage: redact or anonymize training examples containing sensitive data; include an explicit "redact" stage if needed.

### Interview ready insights

- Model choice: encoder-decoder models simplify conditional generation and training with teacher forcing; causal LMs excel when instruction prompting and in-context few-shot behavior are desired. Explain trade-offs in latency and input-length handling.
- Handling long dialogues: discuss hierarchical summarization — first produce per-chunk abstracts, then summarize abstracts into a final summary — and retrieval-based selection to reduce compute.
- PEFT for production: LoRA or adapters let you maintain per-domain adapters for clients while using a single frozen backbone, minimizing storage and enabling fast switch. QLoRA enables training large models on a single GPU.
- Decoding controls to reduce hallucination: beam search with length penalty, n-gram blocking, constrained decoding with allowed lexicons for action items.
- Evaluation: use a mix of ROUGE/BERTScore, QA-based factuality, and targeted human evaluation on key axes (faithfulness, conciseness, action correctness). Report per-slice metrics and worst-case examples.
- Safety and legal: discuss PII filtering in training and inference and audit logging for generated summaries.

### Practice exercises

Baseline fine-tune (easy)

- Task: Fine-tune flan-t5-small on a 1k dialogue-summary pairs subset. Train for 3 epochs and report ROUGE-1/2/L and a few qualitative examples.
- Hints: keep speaker tokens, use DataCollatorForSeq2Seq, and validate on 200 held-out examples.

Hierarchical summarization (medium)

- Task: Implement a two-stage pipeline: chunk dialogues into 512-token windows, generate chunk summaries, then fine-tune a second model to summarize concatenated chunk summaries into a final summary. Compare to single-stage model on long dialogues.
- Hints: try overlapping chunks and include chunk index tokens (Chunk 1:, Chunk 2:) to preserve order.

Factuality QA pipeline (medium)

- Task: Build a small QA-based factuality checker: extract summary sentences, for each sentence generate a question, answer from the dialogue using an extractive QA model, and compute support ratio. Use this to filter/flag generated summaries in evaluation.
- Hints: use a pretrained question-generation model or simple heuristics for question templates.

PEFT LoRA experiment (medium)

- Task: Fine-tune a larger base (flan-t5-large) with LoRA (r=8) on the same dataset and compare performance, GPU memory, and checkpoint size versus full fine-tune on a small GPU.
- Hints: use peft.get_peft_model and compare sum(p.numel() for p in model.parameters() if p.requires_grad).

Style-conditioned summaries and evaluation (advanced)

- Task: Train the model to produce multiple summary styles based on an instruction token (e.g., "Short bullets", "Manager summary", "Action items only"). Evaluate correctness and style adherence; include human-rated helpfulness.
- Hints: prepend "Style: Short bullets" to prompt during training; sample varied styles during training to improve robustness.

Production readiness checklist (advanced)

- Task: From your fine-tuned model produce a deployment checklist: latency targets, quantization strategy (8-bit or 4-bit), adapter management, input redaction pipeline, human-in-the-loop review for flagged summaries, and monitoring metrics (hallucination rate, PII leakage rate, 99th percentile latency). Implement a small script that runs inference and flags summaries failing the QA-support threshold.

---