# Mathematics_c3_w1

## What Is Probability?

### Intuitive Definition

Probability measures how likely an event is to occur, on a scale from 0 (impossible) to 1 (certain).

Imagine spinning a wheel with colored sectors: if half the wheel is red, then the chance you land on red is “half,” or 0.5.

### Prerequisites You Should Know

- Basic set notation: understanding what a “sample space” (all possible outcomes) and an “event” (some subset of outcomes) are.
- Simple counting: knowing how to count elements in a set.

If those feel fuzzy, pause here and review sets and counting before continuing.

### Formal Definition

Let S be the set of all possible outcomes (the sample space).

Let A be an event (a subset of S).

For equally likely outcomes, the probability of A is:

```
P(A) = |A| / |S|
```

- `|A|` means the number of outcomes in event A.
- `|S|` means the number of outcomes in the entire sample space.

This ratio tells you what fraction of all possibilities make A happen.

### Step-by-Step Breakdown of the Formula

1. Identify the sample space S.
2. Count how many outcomes are in S → that’s `|S|`.
3. Identify your event A (the “favorable” outcomes).
4. Count how many outcomes are in A → that’s `|A|`.
5. Divide `|A|` by `|S|` to get a number between 0 and 1.

### Real-World Examples

- Coin flip: S = {Heads, Tails} → |S| = 2.
    - Event A = “Heads” → |A| = 1.
    - P(Heads) = 1/2 = 0.5
- Dice roll: S = {1, 2, 3, 4, 5, 6} → |S| = 6.
    - Event B = “rolling an even number” → B = {2, 4, 6}, |B| = 3.
    - P(even) = 3/6 = 0.5
- Email spam detection (frequency‐based):
    - Suppose in your dataset of 10,000 emails, 2,000 are labeled spam.
    - S = all emails (|S| = 10,000), A = spam (|A| = 2,000).
    - P(spam) = 2000/10000 = 0.2

### Visual & Geometric Intuition

Imagine a pie chart of the sample space:

- The whole pie is the sample space S.
- A slice proportional to |A| is your event A.
- The angle or area of that slice over the whole pie is P(A).

This helps you see probability as “area covered.”

### Practice Problems & Python Exercises

### 1. Probability of Drawing an Ace from a Deck

Problem: You have a standard 52-card deck. What’s the probability of drawing an Ace?

- S = 52 cards → |S| = 52
- A = 4 Aces → |A| = 4
- P(Ace) = 4/52 = 1/13 ≈ 0.0769

### Python Simulation

```python
import random

def estimate_ace_probability(trials=100_000):
    deck = list(range(52))          # label cards 0–51
    aces = set([0, 13, 26, 39])     # imagine these indices are Aces
    count = 0
    for _ in range(trials):
        card = random.choice(deck)
        if card in aces:
            count += 1
    return count / trials

print(estimate_ace_probability())
```

### 2. Simulating Coin Flips

Problem: Estimate P(3 heads in 5 flips).

### Python Code

```python
import random

def trial():
    flips = [random.choice([0,1]) for _ in range(5)]  # 1=heads, 0=tails
    return sum(flips) == 3

def estimate_probability(trials=100_000):
    return sum(trial() for _ in range(trials)) / trials

print(estimate_probability())
```

### 3. Email Spam Dataset Example

Problem: Given a CSV of emails with labels “spam” or “ham,” compute P(spam).

```python
import pandas as pd

df = pd.read_csv("emails.csv")     # assume column “label” has spam/ham
p_spam = (df["label"] == "spam").mean()
print("P(spam) =", p_spam)
```

### How Data Scientists Use This Daily

- When initializing model priors in Naive Bayes (P(feature|class)).
- To decide train/test splits if sampling randomly.
- In A/B testing, to calculate the chance that variation B outperforms A by random fluctuation.

---

## Complement of Probability

### Intuitive Explanation

Probability measures how likely an event A is to happen. The complement of A, written Aᶜ, is everything in the sample space where A does **not** occur.

Imagine a spinner divided into 4 equal wedges. If one wedge is blue (event A), then the chance of landing on blue is 1/4. The complement—landing on anything but blue—is the other 3 wedges, so 3/4.

### Formal Definition

If P(A) is the probability of event A, then the probability of its complement Aᶜ is:

```
P(Aᶜ) = 1 - P(A)
```

- `P(Aᶜ)` reads “probability of not A.”
- The total probability of all possible outcomes is always 1.
- Subtracting P(A) from 1 gives the probability of everything outside A.

### Step-by-Step Breakdown of the Formula

1. Ensure you know P(A), the chance event A happens.
2. Recall that the sum of probabilities over every possible outcome equals 1.
3. Subtract P(A) from 1 to get the chance A does not occur.
4. The result is automatically between 0 and 1, since 0 ≤ P(A) ≤ 1.

### Real‐World Examples

- Weather forecast: If P(rain) = 0.3, then P(no rain) = 1 - 0.3 = 0.7.
- Coin toss: P(heads) = 0.5, so P(tails) = 0.5.
- Email classification: If in your inbox P(spam) = 0.2, then P(ham) = 0.8.
- Defect rate: A factory has 5% defective parts (P(defect)=0.05), so P(good part)=0.95.

### Visual Intuition

Draw a circle representing the sample space. Shade the region for event A (proportion P(A)). The unshaded region is Aᶜ, with area proportional to 1 − P(A).

### Practice Problems & Python Exercises

### 1. Coin Flip Complement

Problem: What is P(not heads) in a fair coin flip?

Answer: P(heads)=0.5 ⇒ P(not heads)=1−0.5=0.5

### Simulation

```python
import random

def estimate_not_heads(trials=100_000):
    return sum(random.choice([0,1]) == 0 for _ in range(trials)) / trials

# Here 1=heads, 0=tails; not heads means tails
print("Estimated P(not heads):", estimate_not_heads())
```

### 2. Deck of Cards

Problem: From a 52-card deck, what is P(not drawing a spade)?

- P(spade) = 13/52 = 0.25
- P(not spade) = 1 − 0.25 = 0.75

### Simulation

```python
import random

def estimate_not_spade(trials=100_000):
    deck = list(range(52))
    spades = set(range(13))  # cards 0–12 are spades
    count = 0
    for _ in range(trials):
        if random.choice(deck) not in spades:
            count += 1
    return count / trials

print("Estimated P(not spade):", estimate_not_spade())
```

### 3. Email Spam Complement

Problem: Given a DataFrame `df` with column “label” ∈ {spam, ham}, compute P(ham).

```python
import pandas as pd

df = pd.read_csv("emails.csv")
p_spam = (df["label"] == "spam").mean()
p_ham = 1 - p_spam
print("P(ham) =", p_ham)
```

### How Data Scientists Use Complements

- In binary classification, P(class=0) = 1 − P(class=1).
- For A/B tests, if P(conversion|variant B) is known, the complement is the non-conversion rate.
- In survival analysis, P(failure) + P(survival) = 1.

---

## Sum of Probabilities for Disjoint Events

### Intuitive Explanation

Probability assigns a number between 0 and 1 to each event, representing its chance of occurring. When two events A and B are **disjoint** (also called **mutually exclusive**), they cannot both happen at the same time—there’s no overlap in outcomes.

Think of rolling a fair six-sided die:

- Event A = “roll a 1” (only outcome {1})
- Event B = “roll a 2” (only outcome {2})

Since you can’t roll both a 1 and a 2 in a single throw, A and B are disjoint. To find the probability of “rolling a 1 **or** a 2,” you simply add their individual probabilities.

### Formal Definition

For two disjoint events A and B in a sample space S:

```
P(A ∪ B) = P(A) + P(B)
```

More generally, for n pairwise disjoint events A₁, A₂, …, Aₙ:

```
P(A₁ ∪ A₂ ∪ … ∪ Aₙ) = P(A₁) + P(A₂) + … + P(Aₙ)
```

- `A ∪ B` means “A or B.”
- Since no outcome belongs to both A and B, we don’t risk double-counting.

### Step-by-Step Breakdown

1. Verify events are disjoint: ensure A ∩ B = ∅ (no common outcomes).
2. Compute P(A) and P(B) separately.
3. Add them: P(A) + P(B) gives the probability that either A or B happens.
4. For more events, repeat the addition across all disjoint events.

### Real-World Examples in ML & DS

- **Dice Roll**: P(roll 1 or 2 or 3) = 1/6 + 1/6 + 1/6 = 3/6 = 0.5.
- **Card Draw**: In a 52-card deck, hearts (13 cards) and spades (13 cards) are disjoint suits.P(heart ∪ spade) = 13/52 + 13/52 = 26/52 = 0.5.
- **Multiclass Labels**: If a dataset has three classes (A, B, C), which are mutually exclusive labels for each sample, thenP(label = A or B or C) = P(A) + P(B) + P(C) = 1 (covers the entire sample space).

### Visual & Geometric Intuition

Imagine the sample space S as a rectangle. Inside, draw two non-overlapping circles for A and B. The union A ∪ B is simply the combined area of both circles. Because there’s no intersection, the total area is the sum of each circle’s area.

### Practice Problems & Python Exercises

### 1. Dice Simulation

Problem: Estimate P(roll is 1, 2, or 3).

```python
import random

def estimate_1_2_3(trials=100_000):
    count = 0
    for _ in range(trials):
        roll = random.randint(1, 6)
        if roll in (1, 2, 3):
            count += 1
    return count / trials

print("Estimated P(1,2,3):", estimate_1_2_3())
```

### 2. Card Draw

Problem: Simulate drawing a card and estimate P(heart or spade).

```python
import random

def estimate_heart_spade(trials=100_000):
    suits = ["hearts", "spades", "diamonds", "clubs"]
    count = 0
    for _ in range(trials):
        if random.choice(suits) in ("hearts", "spades"):
            count += 1
    return count / trials

print("Estimated P(heart or spade):", estimate_heart_spade())
```

### 3. Email Category Probability

Problem: Given a DataFrame `df` with column “category” ∈ {“promotion”, “social”, “updates”, “primary”}, all disjoint:

```python
import pandas as pd

df = pd.read_csv("emails.csv")
# Compute P(promotion or social)
p_promotion = (df["category"] == "promotion").mean()
p_social    = (df["category"] == "social").mean()
p_union     = p_promotion + p_social
print("P(promotion or social) =", p_union)
```

### How Data Scientists Use Disjoint-Sum Rule

- **Softmax Outputs**: In multiclass classification, softmax probabilities for each class sum to 1 because classes are mutually exclusive.
- **Naive Bayes Priors**: Class priors P(Class=k) across all k sum to 1.
- **Event Counting**: When aggregating non-overlapping conditions (e.g., users in age brackets), summing probabilities gives overall coverage without double-counting.

---

## Sum of Probabilities for Joint Events

### Intuitive Explanation

When two events A and B can occur at the same time, they are not disjoint. To find the chance that **either** A **or** B happens, you can’t simply add P(A) and P(B) because you’d count the overlap twice. Instead, you subtract the probability of them both happening together.

### Formal Definition for Two Events

For any two events A and B in the same sample space:

```
P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
```

- `A ∪ B` means “A or B (or both).”
- `A ∩ B` means “A and B together.”
- Subtracting P(A ∩ B) corrects for the fact that outcomes in the overlap were counted twice.

### Step-by-Step Breakdown

1. Compute P(A): probability that event A occurs.
2. Compute P(B): probability that event B occurs.
3. Compute P(A ∩ B): probability that both A and B occur.
4. Add P(A) + P(B).
5. Subtract P(A ∩ B) to remove the double-counted overlap.
6. The result is P(A ∪ B), guaranteed between 0 and 1.

### Extension to Three Events

When you have three events A, B, C:

```
P(A ∪ B ∪ C) = P(A) + P(B) + P(C)
               − P(A ∩ B) − P(A ∩ C) − P(B ∩ C)
               + P(A ∩ B ∩ C)
```

Each pairwise intersection is subtracted, and the triple intersection is added back once.

### Real-World Examples

- Drawing from a deck:
    - A = “draw an Ace” → P(A) = 4/52
    - B = “draw a Heart” → P(B) = 13/52
    - A ∩ B = “draw the Ace of Hearts” → P(A ∩ B) = 1/52
    - P(A ∪ B) = 4/52 + 13/52 − 1/52 = 16/52 ≈ 0.3077
- Email classification:
    - A = “email contains word ‘free’” → P(A) measured from dataset
    - B = “email marked spam” → P(B) measured from dataset
    - P(A ∩ B) = proportion of emails that contain “free” **and** are spam
    - P(A ∪ B) = P(contains free) + P(spam) − P(both)
- Machine learning model alerts:
    - A = “feature X > threshold”
    - B = “feature Y > threshold”
    - P(A ∪ B) gives the probability that at least one alert triggers.

### Visual & Geometric Intuition

Draw two overlapping circles for A and B inside a rectangle (the sample space).

The union is the total shaded area of both circles.

Adding the areas of each circle covers the overlap twice, so you remove the overlap once to get the correct total.

### Practice Problems & Python Exercises

### 1. Deck of Cards: Ace or Heart

```python
import random

def estimate_ace_or_heart(trials=100_000):
    ranks = list(range(13))                     # 0–12 for Ace–King
    suits = ["hearts", "spades", "clubs", "diamonds"]
    count = 0
    for _ in range(trials):
        rank = random.choice(ranks)
        suit = random.choice(suits)
        is_ace   = (rank == 0)
        is_heart = (suit == "hearts")
        if is_ace or is_heart:
            count += 1
    return count / trials

print("Estimated P(Ace or Heart):", estimate_ace_or_heart())
```

Compare to the analytic result: 16/52 ≈ 0.3077.

### 2. Email DataFrame: “free” or “spam”

```python
import pandas as pd

df = pd.read_csv("emails.csv")  # columns: text, label (spam/ham)

p_free = df["text"].str.contains("free", case=False).mean()
p_spam = (df["label"] == "spam").mean()
p_both = ((df["text"].str.contains("free", case=False)) &
          (df["label"] == "spam")).mean()

p_union = p_free + p_spam - p_both
print("P(free or spam) =", p_union)
```

### 3. Simulated Feature Alerts

```python
import numpy as np

N = 100_000
X = np.random.normal(0, 1, N)
Y = np.random.normal(0, 1, N)

# A: X > 1.5, B: Y > 1.5
pA  = (X > 1.5).mean()
pB  = (Y > 1.5).mean()
pAB = ((X > 1.5) & (Y > 1.5)).mean()

p_union = pA + pB - pAB
print("P(A or B) ≈", p_union)
```

### How Data Scientists Use This Rule

- Combining non-exclusive conditions in exploratory analysis.
- Computing the probability of complex events in A/B testing.
- Adjusting metrics when counts overlap (e.g., users who clicked *or* scrolled).
- In probabilistic reasoning when events share dependencies, before moving to conditional probability.

---

## Independence (Independent Events)

### Intuitive Explanation

Two events A and B are independent when knowing that one occurred gives you no information about whether the other occurs.

Imagine flipping a fair coin (Event A = “heads”) and rolling a fair six-sided die (Event B = “roll a 4”). The coin flip result doesn’t change the chance of rolling a 4.

### Formal Definition

For events A and B in the same sample space:

```
P(A ∩ B) = P(A) * P(B)
```

Equivalently, if P(B) > 0:

```
P(A | B) = P(A)
```

- `P(A ∩ B)` is the probability both A and B occur.
- `P(A) * P(B)` is the product of their separate probabilities.
- `P(A | B)` is the probability of A given B has happened.

### Step-by-Step Breakdown

1. Compute P(A): the chance of A occurring on its own.
2. Compute P(B): the chance of B occurring on its own.
3. Multiply P(A) * P(B) to get the joint probability if independent.
4. Check the result against P(A ∩ B) from data or logic—if they match, A and B are independent.

### Real-World Examples

- Coin flip + die roll
    - P(heads) = 0.5, P(4) = 1/6 → P(heads ∩ 4) = 0.5 * 1/6 ≈ 0.0833.
- Feature independence in Naive Bayes
    - If features X₁ and X₂ are independent given class C, thenP(X₁, X₂ | C) = P(X₁ | C) * P(X₂ | C).
- Randomized controlled trials
    - Assignment to treatment (A) and baseline characteristic distribution (B) should be independent after proper randomization.

### Visual Intuition

Draw two circles A and B inside a rectangle for the sample space.

If they’re independent, the area of their overlap equals the product of each circle’s area fraction.

### Practice Problems & Python Exercises

### 1. Coin Flip & Die Roll Simulation

Estimate P(heads and roll = 4) and compare to P(heads)*P(4).

```python
import random

def estimate_independence(trials=100_000):
    count_joint = 0
    for _ in range(trials):
        flip = random.choice(["H", "T"])
        roll = random.randint(1, 6)
        if flip == "H" and roll == 4:
            count_joint += 1
    p_joint = count_joint / trials
    p_heads = 0.5
    p_four = 1/6
    print("Estimated P(H ∩ 4):", p_joint)
    print("Theoretical P(H)*P(4):", p_heads * p_four)

estimate_independence()
```

### 2. Dice Roll Independence Check

Simulate two dice rolls to verify P(both even) = P(even) * P(even).

```python
import random

def simulate_two_dice(trials=100_000):
    count_joint = 0
    for _ in range(trials):
        d1, d2 = random.randint(1,6), random.randint(1,6)
        if d1 % 2 == 0 and d2 % 2 == 0:
            count_joint += 1
    p_joint = count_joint / trials
    p_even = 3/6
    print("Estimated P(both even):", p_joint)
    print("Theoretical P(even)^2:", p_even**2)

simulate_two_dice()
```

### 3. Feature Independence in an Email Dataset

Check if the words “free” and “win” occur independently in spam emails.

```python
import pandas as pd

df = pd.read_csv("emails.csv")  # columns: text, label
spam = df[df["label"] == "spam"]["text"]

p_free = spam.str.contains("free", case=False).mean()
p_win  = spam.str.contains("win",  case=False).mean()
p_both = (spam.str.contains("free", case=False) &
          spam.str.contains("win",  case=False)).mean()

print("P(free) =", p_free)
print("P(win)  =", p_win)
print("P(free and win) =", p_both)
print("P(free)*P(win) =", p_free * p_win)
```

### How Data Scientists Use Independence

- Naive Bayes classification assumes feature independence given class.
- Deciding when you can factor joint distributions into simpler marginals.
- Simplifying Monte Carlo simulations by multiplying independent event probabilities.
- Validating randomization in experiments.

---

## Birthday Problem

### Intuitive Explanation

The birthday problem asks: in a group of n people, what is the chance that at least two share the same birthday?

Surprisingly, with just 23 people the probability exceeds 50%.

This “paradox” shows how quickly collision rates grow as you add more items into a fixed set of 365 days.

### Prerequisites You Should Know

- Complement rule (P(Aᶜ) = 1 − P(A)).
- Multiplication (product) rule for independent events.
- Basic counting of outcomes in a finite sample space.

If any of these feel unclear, review them before proceeding.

### Formal Formula

First compute the probability that **no two** share a birthday. Assuming 365 equally likely days and independence:

```
P_no_match(n) = ∏₍i=0₎ⁿ⁻¹ (365 − i) / 365
P_match(n)    = 1 − P_no_match(n)
```

- `∏₍i=0₎ⁿ⁻¹` means you multiply terms for i = 0 up to n−1.
- Each term `(365 − i) / 365` is the chance the (i+1)th person avoids all previous birthdays.

### Step‐by‐Step Breakdown

1. First person can have any birthday → probability 365/365 = 1.
2. Second person must avoid the first person’s day → (365 − 1)/365.
3. Third person must avoid the first two days → (365 − 2)/365.
4. Continue until the nth person: (365 − (n−1))/365.
5. Multiply all these “no‐collision” terms to get P_no_match(n).
6. Subtract from 1 to find the chance of **at least one** shared birthday.

### Probability vs. Group Size

| n | P(match) ≈ |
| --- | --- |
| 10 | 0.117 |
| 23 | 0.507 |
| 30 | 0.706 |
| 50 | 0.970 |
| 70 | 0.9992 |

### Real‐World Analogies

- Hash collisions: storing n keys into 365 slots, chance two map to the same slot.
- Random login tokens: probability two users get the same session ID.
- Data deduplication: chance two records share the same fingerprint.

### Visual Intuition

Picture a curve rising slowly at first, then steeply climbing near n=23, and leveling off as n→365. The inflection happens because the chance of any new person “colliding” grows with the number already in the room.

### Practice Problems & Python Exercises

### 1. Analytic Computation for n=23

```python
import math

def p_no_match(n, days=365):
    prob = 1.0
    for i in range(n):
        prob *= (days - i) / days
    return prob

n = 23
p_match = 1 - p_no_match(n)
print(f"P(at least one match) for n={n}:", p_match)
```

### 2. Monte Carlo Simulation

```python
import random

def estimate_birthday_match(n, trials=100_000):
    count = 0
    for _ in range(trials):
        birthdays = [random.randint(1, 365) for _ in range(n)]
        if len(set(birthdays)) < n:
            count += 1
    return count / trials

print("Estimated P(match) for n=23:", estimate_birthday_match(23))
```

### 3. Find Threshold for 99% Chance

```python
for n in range(1, 100):
    if 1 - p_no_match(n) >= 0.99:
        print("First n with ≥99% match:", n)
        break
```

### How Data Scientists Use This

- Evaluating collision risk in hash tables or bloom filters.
- Designing unique ID schemes to keep collision probability below a threshold.
- Assessing data quality when many records share limited categories.

---

## Conditional Probability – Part 1

### Intuitive Explanation

Conditional probability measures how likely an event A is, **given** that another event B has already occurred.

Imagine you have a deck of cards and someone tells you they’ve drawn a red card. You now only consider the 26 red cards, so the chance of drawing a Heart (13 out of 26) is 13/26 = 0.5.

### Prerequisites You Should Know

- Joint probability (P(A ∩ B)): the chance both A and B happen.
- Basic counting or frequency estimation.
- Complement and product rules for independent events (to contrast conditional vs. independent).

If any of these feel unclear, review them before proceeding.

### Formal Definition

The conditional probability of A given B, written P(A | B), is:

```
P(A | B) = P(A ∩ B) / P(B)
```

- `P(A ∩ B)` is the probability that **both** A and B occur.
- `P(B)` is the probability that B occurs (must be > 0).
- The ratio “zooms in” on the world where B happened and asks: out of that world, how often does A also happen?

### Step-by-Step Breakdown of the Formula

1. Identify your events A and B in the same sample space.
2. Compute P(B) — the chance B occurs on its own.
3. Compute P(A ∩ B) — the chance both A and B occur together.
4. Divide P(A ∩ B) by P(B). This rescales probabilities to the “universe” where B has happened.
5. The result is guaranteed between 0 and 1.

### Real-World Examples in ML & DS

- **Spam Filtering**: Let A = “email is spam,” B = “email contains the word ‘free.’”P(spam | free) tells you how common spam is among emails that include “free.”
- **Medical Testing**: A = “patient has disease,” B = “test is positive.”P(disease | positive) is diagnostic accuracy once you know the test flagged positive.
- **User Behavior**: A = “user clicks ad,” B = “user is on mobile device.”P(click | mobile) guides mobile-specific ad strategies.

### Visual & Geometric Intuition

Draw two overlapping circles for B and A inside a rectangle (the sample space).

Shade the overlap (A ∩ B). Then imagine focusing only on the B-circle—P(A | B) is the fraction of B’s area that overlaps with A.

### Practice Problems & Python Exercises

### 1. Dice Example: P(roll is even | roll > 3)

Analytic:

- B = {4,5,6} → P(B) = 3/6 = 0.5
- A ∩ B = {4,6} → P(A ∩ B) = 2/6 ≈ 0.333
- P(A | B) = (2/6) / (3/6) = 2/3 ≈ 0.666

### Python Simulation

```python
import random

def estimate_conditional(trials=100_000):
    count_B = 0
    count_A_and_B = 0
    for _ in range(trials):
        roll = random.randint(1, 6)
        if roll > 3:
            count_B += 1
            if roll % 2 == 0:
                count_A_and_B += 1
    return count_A_and_B / count_B

print("Estimated P(even | >3):", estimate_conditional())
```

### 2. Email Spam Example: P(spam | “free”)

```python
import pandas as pd

df = pd.read_csv("emails.csv")  # columns: text, label (spam/ham)

contains_free = df["text"].str.contains("free", case=False)
p_free = contains_free.mean()
p_spam_and_free = df[contains_free]["label"].eq("spam").mean()

p_spam_given_free = p_spam_and_free / p_free
print("P(spam | free) =", p_spam_given_free)
```

### 3. Titanic Survival Example: P(survived | passenger is female)

```python
import pandas as pd

df = pd.read_csv("titanic.csv")  # columns include 'Sex', 'Survived'
female = df["Sex"] == "female"
p_female = female.mean()
p_survived_and_female = df[female]["Survived"].mean()

p_survived_given_female = p_survived_and_female  # since we only look at females
print("P(survived | female) =", p_survived_given_female)
```

### How Data Scientists Use Conditional Probability

- Calculating **feature likelihoods** in Naive Bayes classifiers (P(feature | class)).
- Building **Markov models** where P(next state | current state) defines transitions.
- Splitting metrics by segments (e.g., conversion rate given user cohort).

---

## Conditional Probability – Part 2

### Quick Recap of Part 1

In Part 1, we defined conditional probability P(A | B) as the probability of A **given** B has occurred:

```
P(A | B) = P(A ∩ B) / P(B)
```

We saw how focusing on the universe where B happens “zooms in” and rescales probabilities.

### Law of Total Probability

### Intuitive Explanation

When you have a partition of your sample space into disjoint events B₁, B₂, …, Bₙ that cover every outcome, you can compute the overall probability of A by summing over each way A can occur under those conditions.

Think of diagnosing a disease in different age groups: the chance a person has the disease is the weighted sum of the chance in each age group times the proportion of people in that group.

### Formal Formula

Let B₁, B₂, …, Bₙ be disjoint and cover S (so ∪ᵢ Bᵢ = S). Then:

```
P(A) = ∑_{i=1}^n P(A | B_i) * P(B_i)
```

- Each term P(A | Bᵢ) is the probability of A in the “world” where Bᵢ happens.
- P(Bᵢ) is how likely that world is.
- Summing covers all possible worlds.

### Step‐by‐Step Breakdown

1. Identify your partition events B₁…Bₙ (disjoint and exhaustive).
2. For each i:
    - Compute P(Bᵢ).
    - Compute P(A | Bᵢ).
3. Multiply P(A | Bᵢ) by P(Bᵢ).
4. Sum over i from 1 to n.

### Real-World Example

Dataset with two user cohorts:

- B₁ = “mobile users” (P(B₁)=0.6)
- B₂ = “desktop users” (P(B₂)=0.4)
- P(click | mobile)=0.05, P(click | desktop)=0.08

Overall click rate:

```
P(click) = 0.05*0.6 + 0.08*0.4 = 0.05*0.6 + 0.08*0.4
         = 0.03 + 0.032
         = 0.062
```

### Python Exercise

```python
import numpy as np

p_mobile, p_desktop = 0.6, 0.4
p_click_mobile, p_click_desktop = 0.05, 0.08

p_click = p_click_mobile * p_mobile + p_click_desktop * p_desktop
print("Overall click probability:", p_click)
```

---

## Bayes’ Theorem

### Intuitive Explanation

Bayes’ theorem flips conditional probabilities: from P(A | B) to P(B | A). It tells you how to update your belief in hypothesis B when you see evidence A.

Imagine a medical test: you know how likely the test is positive if you have the disease, but you want how likely you have the disease given a positive test result.

### Formal Formula

For events A and B with P(B) > 0:

```
P(B | A) = [ P(A | B) * P(B) ] / P(A)
```

Using the law of total probability in the denominator when necessary:

```
P(B | A) = [ P(A | B) * P(B) ]
           / ∑_{i} P(A | B_i) * P(B_i)
```

### Step‐by‐Step Breakdown

1. Identify:
    - P(A | B): how likely evidence A is if hypothesis B is true.
    - P(B): your prior belief in B.
    - P(A): overall chance of seeing evidence A (use total probability if needed).
2. Multiply P(A | B) × P(B).
3. Divide by P(A).

### Real-World Examples

**Medical Test**

- P(positive | disease)=0.99 (sensitivity)
- P(disease)=0.01 (prevalence)
- P(positive | no disease)=0.05 (false positive rate)

First compute P(positive):

```
P(positive) = 0.99*0.01 + 0.05*0.99 = 0.0099 + 0.0495 = 0.0594
```

Then posterior probability of disease given positive:

```
P(disease | positive) = (0.99*0.01) / 0.0594 ≈ 0.1667
```

Only ~16.7% of positives actually have the disease.

### Python Simulation

```python
import random

def simulate_bayes(trials=100_000):
    disease_rate = 0.01
    true_pos_rate = 0.99
    false_pos_rate = 0.05
    count_positive_and_disease = 0
    count_positive = 0

    for _ in range(trials):
        has_disease = random.random() < disease_rate
        if has_disease:
            positive = random.random() < true_pos_rate
        else:
            positive = random.random() < false_pos_rate
        if positive:
            count_positive += 1
            if has_disease:
                count_positive_and_disease += 1

    return count_positive_and_disease / count_positive

print("Simulated P(disease | positive):", simulate_bayes())
```

---

## Conditional Independence

### Intuitive Explanation

Two events A and B are conditionally independent given C if, **once you know** C has occurred, knowing A gives you no extra information about B and vice versa.

For example, if you know someone’s age group (C), then their probability of liking rock music (A) and liking pop music (B) might be independent within that age group—even if overall they’re correlated.

### Formal Definition

A and B are conditionally independent given C if:

```
P(A ∩ B | C) = P(A | C) * P(B | C)
```

Equivalently:

```
P(A | B, C) = P(A | C)
```

### Use in Naive Bayes

Naive Bayes classifiers assume that features X₁…Xₖ are all conditionally independent given the class C. That simplifies the joint likelihood:

```
P(X₁, …, Xₖ | C) = ∏_{i=1}^k P(X_i | C)
```

This factorization makes inference tractable on high-dimensional data.

### Python Illustration

```python
import pandas as pd

df = pd.read_csv("emails.csv")  # columns: text, label

# Check conditional independence of words "free" and "win" given spam
spam_emails = df[df["label"] == "spam"]["text"]
p_free_given_spam = spam_emails.str.contains("free").mean()
p_win_given_spam  = spam_emails.str.contains("win").mean()
p_both_given_spam = (spam_emails.str.contains("free") &
                     spam_emails.str.contains("win")).mean()

print("P(free and win | spam):", p_both_given_spam)
print("P(free | spam)*P(win | spam):", p_free_given_spam * p_win_given_spam)
```

If the two sides align closely, “free” and “win” behave approximately independently within spam.

### Practice Problems

1. **Law of Total Probability**
    
    You have three factories producing widgets:
    
    - Factory A (30% of output), defect rate = 2%
    - Factory B (50% of output), defect rate = 3%
    - Factory C (20% of output), defect rate = 5%
    
    Compute the overall defect rate using the law of total probability.
    
2. **Bayes’ Theorem**
    
    In a certain town, 0.1% of people have a rare disease. A test has 95% sensitivity and 98% specificity.
    
    - Compute P(disease | positive).
3. **Conditional Independence**
    
    In a dataset of customer reviews, let A = “review mentions ‘fast shipping’,” B = “review mentions ‘good price’,” and C = “product category is electronics.”
    
    - Estimate from data whether A and B are conditionally independent given C using counts or Python.

### How Data Scientists Use These Concepts

- **Bayesian inference** for updating model parameters as new data arrives.
- **Mixture models** and hidden Markov models rely on the law of total probability.
- **Feature selection** by checking conditional independence to remove redundant variables.
- **Probabilistic graphical models** where edges represent dependencies and conditional independencies.

---

## Bayes’ Theorem – Prior and Posterior

### Intuitive Explanation

Imagine you’re a detective working on a case. You have an initial hunch (“prior belief”) about who the culprit might be. Then you gather evidence—fingerprints, witness statements, or a motive. Bayes’ theorem tells you how to update your initial hunch into a refined belief (“posterior probability”) once you see the evidence.

### Prerequisites You Should Know

- Joint probability: P(A and B)
- Conditional probability: P(A | B)
- Law of total probability: summing over all ways an event can happen

If any of these are fuzzy, review conditional probability before continuing.

### Formal Definition

Let H be a hypothesis (e.g., “email is spam”) and E be evidence (e.g., “email contains the word ‘free’”). The **prior** P(H) is your belief in H before seeing evidence. The **likelihood** P(E | H) is how probable evidence is if H is true. The **posterior**P(H | E) is your updated belief in H given E.

```
P(H | E) = ( P(E | H) * P(H) ) / P(E)
```

To compute P(E) when you have multiple hypotheses H₁,…,Hₙ:

```
P(E) = ∑_{i=1 to n} P(E | H_i) * P(H_i)
```

### Step-by-Step Breakdown

1. Identify hypothesis H and evidence E.
2. Compute prior P(H): initial belief in H before seeing E.
3. Compute likelihood P(E | H): chance of seeing E if H is true.
4. Compute marginal P(E): overall chance of seeing E across all hypotheses.
5. Multiply P(E | H) by P(H).
6. Divide by P(E) to get the posterior P(H | E).

### Real-World Examples

- Medical testing:
    - H = “patient has disease”
    - E = “test result is positive”
    - Prior P(H) = disease prevalence (e.g., 1%)
    - Likelihood P(E | H) = sensitivity (e.g., 99%)
    - Posterior P(H | E) = updated chance patient has disease given positive test
- Spam filtering:
    - H = “email is spam”
    - E = “email contains the word ‘buy’”
    - Prior P(H) from historical spam ratio
    - Likelihood P(E | H) from frequency of “buy” in spam
    - Posterior P(H | E) guides classification

### Geometric Intuition

Visualize two bars side by side:

- Bar 1 (hypothesis H): height = P(H)
- Bar 2 (hypothesis not‐H): height = P(not‐H)

Within each bar, shade the portion corresponding to P(E | H) and P(E | not‐H). The total shaded area across both bars is P(E). Bayes’ theorem tells you how much of the shaded area comes from bar 1 versus bar 2.

### Practice Problems & Python Exercises

### 1. Medical Test Posterior

Problem: A disease has 2% prevalence. Test sensitivity = 95%, false positive rate = 5%. Compute P(disease | positive).

Analytic:

- Prior P(H) = 0.02
- P(E | H) = 0.95
- P(E | not‐H) = 0.05
- P(E) = 0.95×0.02 + 0.05×0.98 = 0.0181 + 0.049 = 0.0671
- Posterior = (0.95×0.02) / 0.0671 ≈ 0.282

### Python Code

```python
p_disease = 0.02
p_pos_given_disease = 0.95
p_pos_given_healthy = 0.05

p_pos = p_pos_given_disease * p_disease \
        + p_pos_given_healthy * (1 - p_disease)

p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos
print("P(disease | positive):", p_disease_given_pos)
```

### 2. Email Spam Example

Problem: In your inbox, 20% of emails are spam. 40% of spam contains “free.” 5% of non‐spam contains “free.” Compute P(spam | “free”).

```python
p_spam = 0.20
p_free_given_spam = 0.40
p_free_given_ham = 0.05

p_free = p_free_given_spam * p_spam \
         + p_free_given_ham * (1 - p_spam)

p_spam_given_free = (p_free_given_spam * p_spam) / p_free
print("P(spam | free):", p_spam_given_free)
```

### 3. A/B Testing Posterior

Problem: You run two landing pages, A and B. Prior belief page A is better = 0.5. After 1,000 visits, page A converts 100 users, page B converts 120. Use Bayes’ theorem with Beta priors (use `scipy.stats.beta`) to compute posterior distributions over conversion rates and compare.

```python
from scipy.stats import beta

# Prior: Beta(1,1) uniform
alpha_prior, beta_prior = 1, 1

# Observed data
conv_A, trials_A = 100, 1000
conv_B, trials_B = 120, 1000

# Posterior parameters
post_A = beta(alpha_prior + conv_A, beta_prior + trials_A - conv_A)
post_B = beta(alpha_prior + conv_B, beta_prior + trials_B - conv_B)

# Mean of posterior
print("Posterior mean A:", post_A.mean())
print("Posterior mean B:", post_B.mean())
```

---

## Bayes’ Theorem – The Naive Bayes Model

### Intuitive Explanation

Naive Bayes turns Bayes’ theorem into a classifier for multiple features. It assumes that all features are **conditionally independent** given the class label. Despite this “naive” assumption, it often works surprisingly well in practice and is blazing fast.

### Formal Definition

For a data point with features X₁, X₂, …, Xₖ and classes C ∈ {c₁, c₂, …, cₙ}, the posterior for class cⱼ is:

```
P(C = c_j | X₁…X_k)
  = [ P(C=c_j) * ∏_{i=1 to k} P(X_i | C=c_j) ]
    / P(X₁…X_k)
```

Since P(X₁…X_k) is the same across classes, we compare **unnormalized scores**:

```
score(c_j) = P(C=c_j) * ∏_{i=1 to k} P(X_i | C=c_j)
```

Then predict the class with the highest score.

### Step-by-Step Breakdown

1. Compute class priors P(C=cⱼ) from training labels.
2. For each feature Xᵢ and class cⱼ, estimate likelihood P(Xᵢ | C=cⱼ) using frequencies (categorical) or densities (continuous).
3. Multiply priors by all likelihoods to get a score for each class.
4. Normalize if you need actual probabilities, or pick the class with the highest score directly.

### Real-World Examples

- **Spam filtering**: features are word‐presence flags.
- **Document classification**: TF–IDF scores turned into likelihood estimates.
- **Medical diagnosis**: symptoms as binary features, disease classes as labels.

### Graphical Intuition

Draw a probability tree:

- First branch on class C with weights P(c₁), P(c₂), …
- Then from each class node, branch on each feature X₁, X₂…The Naive Bayes assumption prunes all connections between features—they only connect through the class node—making the tree easy to compute.

### Practice Problems & Python Exercises

### 1. Text Classification with Multinomial Naive Bayes

Use scikit-learn to classify movie reviews as positive or negative.

```python
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
reviews = load_files("movie_reviews/")
X_train, X_test, y_train, y_test = train_test_split(
    reviews.data, reviews.target, test_size=0.2, random_state=0
)

# Vectorize text to word counts
vect = CountVectorizer(stop_words="english")
X_train_counts = vect.fit_transform([doc.decode("utf-8") for doc in X_train])
X_test_counts  = vect.transform([doc.decode("utf-8") for doc in X_test])

# Train Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
y_pred = clf.predict(X_test_counts)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

### 2. Gaussian Naive Bayes on Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

### 3. Build Naive Bayes from Scratch

```python
import numpy as np
import pandas as pd

def train_naive_bayes(df, feature_cols, label_col):
    classes = df[label_col].unique()
    priors = {c: (df[label_col]==c).mean() for c in classes}
    likelihoods = {}
    for c in classes:
        subset = df[df[label_col]==c]
        likelihoods[c] = {
            col: subset[col].value_counts(normalize=True).to_dict()
            for col in feature_cols
        }
    return priors, likelihoods

def predict_naive_bayes(priors, likelihoods, sample):
    scores = {}
    for c, prior in priors.items():
        score = np.log(prior)
        for col, value in sample.items():
            score += np.log(likelihoods[c][col].get(value, 1e-6))
        scores[c] = score
    return max(scores, key=scores.get)

# Example usage on small DataFrame
df = pd.DataFrame({
    "feature1": ["yes","yes","no","no","yes"],
    "feature2": ["high","low","medium","low","high"],
    "label":    ["spam","ham","ham","ham","spam"]
})

priors, likelihoods = train_naive_bayes(df, ["feature1","feature2"], "label")
sample = {"feature1":"yes", "feature2":"medium"}
print("Predicted label:", predict_naive_bayes(priors, likelihoods, sample))
```

### How Data Scientists Use Naive Bayes Daily

- Rapid prototyping of text classifiers when data is high-dimensional.
- Baseline models for comparison against more complex algorithms.
- Real-time scoring where model simplicity and speed are crucial.
- Feature engineering: independence assumption teaches you which features interact.

---

## Random Variables

### Intuitive Explanation

A random variable is simply a rule that assigns a number to each outcome of a random process. It turns the abstract “sample space” of possible outcomes into a numerical scale you can analyze.

- When you roll a die, you don’t care that the outcome is “face #4”; you think of that as the number 4.
- When you flip three coins, you might want to count how many heads appear—0, 1, 2, or 3.

That counting or labeling step is exactly what a random variable does: it **maps** each pure outcome to a number.

### Formal Definition

Let S be a sample space of all possible outcomes. A random variable X is a function:

```
X: S → ℝ
```

- For each outcome s in S, X(s) is a real number.
- We write X as uppercase (the random variable) and x as one of its numeric values.

### Types of Random Variables

| Type | Range of X | Example |
| --- | --- | --- |
| Discrete | Finite or countable | Number of heads in 3 coin flips {0,1,2,3} |
| Continuous | Uncountably infinite | Height of a person in centimeters |

### Key Formulas

### Discrete Random Variable (PMF)

The **probability mass function** p_X(x) gives the chance that X takes value x:

```
p_X(x) = P(X = x)
```

- You list each x in the range of X and assign a probability.
- All probabilities sum to 1:

```
∑_{x} p_X(x) = 1
```

### Continuous Random Variable (PDF)

The **probability density function** f_X(x) satisfies:

```
P(a ≤ X ≤ b) = ∫_a^b f_X(x) dx
```

- f_X(x) itself can be greater than 1, but the area under its curve across all real numbers equals 1:

```
∫_{-∞}^{∞} f_X(x) dx = 1
```

### Step-by-Step Breakdown

1. **Choose your experiment** (flip coins, roll dice, measure heights).
2. **Define X**: decide what number you’ll record for each outcome
    - Dice: X(“roll 4”) = 4
    - Coins: X(“HHT”) = 2 heads
    - Heights: X(person) = their height in cm
3. **Determine range**: list all possible x values.
4. **Build distribution**: assign probabilities (discrete) or densities (continuous).
5. **Check normalization**: sum of pmf = 1 or integral of pdf = 1.

### Real-World Examples

- **Customer Arrivals**: X = number of customers in an hour → often modeled as a **Poisson** discrete random variable.
- **Email Length**: X = number of words → discrete, skewed.
- **Daily Temperature**: X = measured in °C → continuous, often approximated by a **normal** distribution.
- **Document Vector Length**: X = magnitude of TF–IDF vector → continuous.

### Visual & Geometric Intuition

- Discrete X: visualize p_X(x) as bars at each x-value. The height is the probability.
- Continuous X: draw a smooth curve f_X(x). The area under the curve between a and b is P(a ≤ X ≤ b).

### Practice Problems & Python Exercises

### 1. Discrete: Heads in 3 Coin Flips

Problem: Define X = number of heads when flipping three fair coins.

- Range: {0,1,2,3}
- Compute p_X(x) analytically and simulate via Python.

```python
import itertools
from collections import Counter

# All coin‐flip outcomes
outcomes = list(itertools.product(['H','T'], repeat=3))
counts = [out.count('H') for out in outcomes]
pmf_analytical = Counter(counts)
for x in pmf_analytical:
    pmf_analytical[x] /= len(outcomes)
print("Analytic PMF:", pmf_analytical)

# Simulation
import random
trials = 100_000
sim_counts = Counter(sum(random.choice([0,1]) for _ in range(3)) for _ in range(trials))
pmf_sim = {x: sim_counts[x]/trials for x in sorted(sim_counts)}
print("Simulated PMF:", pmf_sim)
```

### 2. Continuous: Heights from Normal Distribution

Problem: Model adult heights as X ∼ Normal(µ=170, σ=10).

- Compute P(160 ≤ X ≤ 180).
- Plot the PDF.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = 170, 10
# Probability between 160 and 180
p_range = norm.cdf(180, mu, sigma) - norm.cdf(160, mu, sigma)
print("P(160 ≤ X ≤ 180):", p_range)

# Plot PDF
xs = np.linspace(130, 210, 400)
plt.plot(xs, norm.pdf(xs, mu, sigma))
plt.fill_between(xs, norm.pdf(xs, mu, sigma), where=(xs>=160)&(xs<=180), alpha=0.3)
plt.title("Height distribution (µ=170, σ=10)")
plt.xlabel("Height (cm)"); plt.ylabel("Density")
plt.show()
```

### 3. Build Empirical Distribution from Data

Problem: You have a CSV of session durations (in minutes). Estimate the distribution of X = session duration.

- Discrete approximation: bin into 1-minute intervals → frequency counts.
- Continuous estimation: plot a kernel density estimate.

```python
import pandas as pd
import seaborn as sns

df = pd.read_csv("sessions.csv")  # column “duration”
# Discrete‐style histogram
hist = df["duration"].value_counts().sort_index() / len(df)
print("Empirical PMF (first 10 bins):", hist.head(10))

# Continuous KDE
sns.kdeplot(df["duration"], fill=True)
```

### How Data Scientists Use Random Variables

- **Feature engineering**: treating each feature as a random variable and modeling its distribution.
- **Probabilistic models**: defining likelihoods for data under model assumptions.
- **Uncertainty quantification**: tracking distributions of predictions.
- **Simulation & sampling**: drawing from random variables to evaluate model performance under noise.

---

## Discrete Probability Distributions

### Intuitive Explanation

A discrete probability distribution tells you how likely each specific outcome of a random process is, when there are a countable number of outcomes.

Think of rolling a six-sided die: there are exactly six outcomes (1–6), and the distribution assigns a probability to each face.

You can visualize it as a bar chart where each bar’s height is the probability of that exact value.

### Formal Definition

Let X be a discrete random variable taking values in a countable set 𝒳. Its **probability mass function (PMF)** is:

```
p_X(x) = P(X = x)
```

This function must satisfy two properties:

- Non-negativity:
    
    ```
    p_X(x) ≥ 0   for every x ∈ 𝒳
    ```
    
- Normalization:
    
    ```
    ∑_{x∈𝒳} p_X(x) = 1
    ```
    

That sum covers every possible value of X.

### Step-by-Step Breakdown of the PMF

1. Identify the random variable X and list all possible values 𝒳 = {x₁, x₂, …}.
2. For each xᵢ in 𝒳, determine how often X equals xᵢ in the sample space (or estimate from data).
3. Compute p_X(xᵢ) = count of outcomes where X=xᵢ divided by total outcomes.
4. Verify that all p_X(xᵢ) sum to 1.
5. Use p_X(x) to answer questions like “What is P(X ≤ 3)?” by summing over appropriate x.

### Common Discrete Distributions

| Distribution | Support (𝒳) | PMF formula | Use Case |
| --- | --- | --- | --- |
| Bernoulli | {0,1} | `p_X(1)=p , p_X(0)=1−p` | Single yes/no trial (flip, click) |
| Binomial | {0,1,…,n} | `p_X(k)=C(n,k) p^k (1−p)^(n−k)` | Count of successes in n independent trials |
| Poisson | {0,1,2,…} | `p_X(k)=λ^k e^(−λ) / k!` | Rare events over fixed interval (arrivals, faults) |

### Real-World Examples

- **Email arrivals per hour** often follow a Poisson distribution with rate λ (average emails/hour).
- **Number of defective items** in a batch of n can be modeled as Binomial(n, p) when each item has independent defect probability p.
- **Click/no-click** on an ad is Bernoulli( p_click ), where p_click is the click-through rate.

### Visual & Geometric Intuition

Imagine drawing vertical bars at each x in 𝒳 on the x-axis. The bar’s height is p_X(x).

- For a fair die, six bars all have height 1/6.
- For a biased coin (Bernoulli), two bars at x=0 and x=1 have heights (1−p) and p.

Seeing the bars helps you grasp where most of the probability mass lies.

### Practice Problems & Python Exercises

### 1. Empirical PMF of Coin Flips

Problem: Flip a biased coin with p(heads)=0.3 five times. Empirically estimate the PMF of X = number of heads.

```python
import random
from collections import Counter

def empirical_pmf(trials=100_000, flips=5, p=0.3):
    counts = Counter()
    for _ in range(trials):
        heads = sum(random.random() < p for _ in range(flips))
        counts[heads] += 1
    pmf = {k: v / trials for k, v in counts.items()}
    return pmf

print(empirical_pmf())
```

### 2. Analytic vs. Simulated Binomial

Problem: Compare the analytic Binomial PMF with a simulation for n=10, p=0.4.

```python
import math
import random
from collections import Counter

def binomial_pmf(k, n, p):
    return math.comb(n, k) * p**k * (1-p)**(n-k)

# Analytic PMF
analytic = {k: binomial_pmf(k, 10, 0.4) for k in range(11)}

# Simulation
trials = 100_000
counts = Counter()
for _ in range(trials):
    counts[sum(random.random() < 0.4 for _ in range(10))] += 1
sim = {k: counts[k]/trials for k in range(11)}

print("k  Analytic    Simulated")
for k in range(11):
    print(f"{k:<2} {analytic[k]:.4f}    {sim[k]:.4f}")
```

### 3. Poisson Count Simulation

Problem: Model website hits per minute as Poisson(λ=3). Simulate and plot PMF.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

lm = 3.0
trials = 100_000
samples = np.random.poisson(lm, size=trials)
counts = Counter(samples)
pmf = {k: counts[k]/trials for k in sorted(counts)}

plt.bar(pmf.keys(), pmf.values())
plt.title("Empirical Poisson PMF (λ=3)")
plt.xlabel("Hits per minute")
plt.ylabel("Probability")
plt.show()
```

### How Data Scientists Use Discrete Distributions

- **Feature modeling**: count-based features (word counts, event counts) often assume Poisson or multinomial distributions.
- **Anomaly detection**: deviations from expected discrete counts signal outliers.
- **Naive Bayes classifiers**: use discrete distributions (multinomial/Bernoulli) to model word frequencies.
- **A/B testing**: modeling discrete success/failure trials with Binomial models to compare conversion rates.

---

## Binomial Distribution

### Intuitive Explanation

The binomial distribution models the number of “successes” you get in a fixed number of independent yes/no trials, each with the same probability of success.

For example, if you flip a biased coin 10 times with a 30% chance of heads, the binomial distribution tells you how likely it is to see exactly 0, 1, 2, …, or 10 heads.

### Formal Definition

Let X be the number of successes in n independent trials, each with success probability p. The **probability mass function**is:

```
p_X(k) = C(n, k) * p^k * (1 - p)^(n - k)
```

- `C(n, k)` is “n choose k” = n! / (k! (n - k)!)
- `p^k` is the probability of k successes
- `(1 - p)^(n - k)` is the probability of the remaining (n - k) failures
- k ranges from 0 up to n

### Step-by-Step Breakdown of the Formula

1. **n**: total number of trials.
2. **k**: number of successes you’re interested in.
3. **C(n, k)**: counts how many ways you can choose which k trials out of n are successes.
4. **p^k**: probability of those k trials all turning out successful.
5. **(1 − p)^(n − k)**: probability of the other (n − k) trials all failing.
6. Multiply those three parts to get the chance of exactly k successes in any order.

### Real-World Examples

- **A/B Testing**: Out of 500 visitors to variant A, 50 convert. Model the conversion count as Binomial(n=500, p≈0.1).
- **Email Campaign**: If your open rate is 20%, the number of opens in 1,000 emails is Binomial(n=1000, p=0.2).
- **Manufacturing**: A production line has a 1% defect rate. The number of defects in a batch of 200 items is Binomial(n=200, p=0.01).
- **Quality Control**: Counting the number of passed parts out of a fixed lot size when failure probability is known.

### Visual & Geometric Intuition

- Plot k on the x-axis from 0 to n.
- Draw a bar at height p_X(k) for each k.
- Shape is unimodal, peaking near k = n·p.
- Mean of X is n·p, variance is n·p·(1−p).

A typical binomial bar chart shows the concentration of probability around the expected number of successes.

### Practice Problems & Python Exercises

### 1. Compute Analytic PMF for n=10, p=0.4

```python
import math

def binomial_pmf(k, n, p):
    return math.comb(n, k) * p**k * (1 - p)**(n - k)

n, p = 10, 0.4
pmf = {k: binomial_pmf(k, n, p) for k in range(n + 1)}
print("Binomial PMF:", pmf)
```

### 2. Simulate Binomial Trials

```python
import numpy as np
from collections import Counter

n, p, trials = 10, 0.4, 100_000
samples = np.random.binomial(n, p, size=trials)
counts = Counter(samples)
empirical_pmf = {k: counts[k] / trials for k in range(n + 1)}
print("Empirical PMF:", empirical_pmf)
```

### 3. Cumulative Probability

Compute P(X ≤ 3) both analytically and via simulation:

```python
# Analytic cumulative
p_cum = sum(binomial_pmf(k, n, p) for k in range(4))
print("P(X ≤ 3) analytic:", p_cum)

# Simulation cumulative
p_cum_sim = sum(empirical_pmf[k] for k in range(4))
print("P(X ≤ 3) simulated:", p_cum_sim)
```

### 4. Plotting the PMF

```python
import matplotlib.pyplot as plt

ks = list(pmf.keys())
values = list(pmf.values())

plt.bar(ks, values)
plt.title("Binomial(n=10, p=0.4) PMF")
plt.xlabel("Number of Successes (k)")
plt.ylabel("P(X = k)")
plt.show()
```

### How Data Scientists Use the Binomial Distribution

- Modeling counts of binary events (clicks, opens, defects) in a fixed number of trials.
- Performing power analysis and sample size calculations for A/B tests.
- Constructing confidence intervals for proportions.
- Feature engineering: generating synthetic binary‐count features or simulating expected variability.

---

## Continuous Probability Distributions

### Intuitive Explanation

A continuous distribution models a random variable X that can take on any real value (often within an interval). Instead of assigning probabilities to exact points (which would all be zero), we describe how the probability “flows” over ranges using a **probability density function** (PDF).

You can think of the PDF as a height curve: the higher the curve over an interval, the more probability mass that interval carries. The total area under the curve is 1, just as the total probability across all outcomes is 1.

### Formal Definitions

### Probability Density Function (PDF)

```
f_X(x) is the PDF of X such that
P(a ≤ X ≤ b) = ∫_a^b f_X(x) dx
```

- f_X(x) ≥ 0 for all x.
- ∫_{−∞}^{∞} f_X(x) dx = 1.

### Cumulative Distribution Function (CDF)

```
F_X(x) = P(X ≤ x) = ∫_{−∞}^x f_X(t) dt
```

- F_X(−∞) = 0, F_X(+∞) = 1.
- F_X(x) is non-decreasing and right-continuous.

### Key Properties

1. Non-negativity:
    
    ```
    f_X(x) ≥ 0
    ```
    
2. Normalization:
    
    ```
    ∫_{−∞}^{∞} f_X(x) dx = 1
    ```
    
3. Probabilities over intervals:
    
    ```
    P(a ≤ X ≤ b) = ∫_a^b f_X(x) dx
    ```
    
4. Relationship to CDF:
    
    ```
    f_X(x) = d/dx [ F_X(x) ]
    ```
    

### Common Continuous Distributions

| Distribution | Support | PDF Formula | Use Case |
| --- | --- | --- | --- |
| Uniform(a, b) | x ∈ [a, b] | `f(x) = 1/(b - a)` | Random sampling over fixed range |
| Normal(µ,σ) | x ∈ (−∞, ∞) | `f(x) = (1/(σ√(2π))) * exp(−(x−µ)²/(2σ²))` | Measurement noise, errors in regression |
| Exponential(λ) | x ∈ [0, ∞) | `f(x) = λ * exp(−λ x)` | Time between Poisson events (arrivals, failures) |
| Gamma(k,θ) | x ∈ [0, ∞) | `f(x) = x^{k−1} exp(−x/θ) / (Γ(k) θ^k)` | Waiting time for k events |
| Beta(α,β) | x ∈ [0, 1] | `f(x) = x^{α−1}(1−x)^{β−1} / B(α,β)` | Priors for probabilities in Bayesian inference |

### Visual & Geometric Intuition

- Plot x on the horizontal axis.
- Draw the PDF curve f_X(x).
- The **area** under the curve from a to b represents P(a ≤ X ≤ b).
- Sharp peaks indicate values that are most likely; long tails indicate occasional extreme values.

### Practice Problems & Python Exercises

### 1. Uniform Distribution on [2, 5]

**Compute analytically** P(3 ≤ X ≤ 4):

```
P(3 ≤ X ≤ 4) = ∫_3^4 (1/(5−2)) dx = (4−3) / 3 = 1/3 ≈ 0.3333
```

**Python Simulation & Plot**

```python
import numpy as np
import matplotlib.pyplot as plt

a, b, trials = 2, 5, 100_000
samples = np.random.uniform(a, b, size=trials)

# Empirical P(3 ≤ X ≤ 4)
p_emp = ((samples >= 3) & (samples <= 4)).mean()
print("Empirical P(3 ≤ X ≤ 4):", p_emp)

# Plot PDF
xs = np.linspace(1.5, 5.5, 300)
pdf = np.where((xs >= a) & (xs <= b), 1/(b - a), 0)
plt.plot(xs, pdf, label="Uniform PDF")
plt.fill_between(xs, pdf, where=(xs>=3)&(xs<=4), alpha=0.3)
plt.title("Uniform(2,5)")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.show()
```

### 2. Normal Distribution N(µ=0, σ=1)

**Compute analytically** P(−1 ≤ X ≤ 1):

```python
from scipy.stats import norm
p = norm.cdf(1, 0, 1) - norm.cdf(-1, 0, 1)
print("P(-1 ≤ X ≤ 1):", p)  # ≈ 0.6827
```

**Simulation & Histogram**

```python
import numpy as np
import matplotlib.pyplot as plt

trials = 100_000
samples = np.random.normal(0, 1, size=trials)

# Empirical probability
p_emp = ((samples >= -1) & (samples <= 1)).mean()
print("Empirical P(-1 ≤ X ≤ 1):", p_emp)

# Plot histogram
plt.hist(samples, bins=50, density=True, alpha=0.6, label="Empirical")
xs = np.linspace(-4, 4, 300)
plt.plot(xs, norm.pdf(xs, 0, 1), 'r--', label="Theoretical PDF")
plt.title("Standard Normal Distribution")
plt.xlabel("x"); plt.ylabel("Density")
plt.legend()
plt.show()
```

### 3. Exponential Distribution with λ=0.5

**Compute analytically** P(X > 3):

```
P(X > 3) = ∫_3^∞ 0.5 exp(−0.5 x) dx = exp(−0.5 * 3) ≈ 0.2231
```

**Python Code**

```python
import numpy as np
from scipy.stats import expon

lam = 0.5
# Theoretical
p_theo = expon.sf(3, scale=1/lam)  # survival function
print("P(X > 3) theoretical:", p_theo)

# Simulation
samples = np.random.exponential(1/lam, size=100_000)
p_emp = (samples > 3).mean()
print("P(X > 3) empirical:", p_emp)
```

### 4. Empirical Density Estimation

Given a CSV `data.csv` of continuous measurements:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")  # column “value”
sns.histplot(df["value"], kde=True, stat="density")
plt.title("Empirical Density & KDE")
plt.xlabel("Value"); plt.ylabel("Density")
plt.show()
```

### How Data Scientists Use Continuous Distributions

- **Assuming Gaussian noise** in regression and classification likelihoods.
- **Modeling inter-arrival times** with Exponential or Gamma for queuing and reliability.
- **Feature transformation**: mapping raw data to approximate Normal via Box–Cox for algorithms that assume normality.
- **Anomaly detection**: flagging values in tail regions of fitted continuous models.

---

## Probability Density Function (PDF)

### Intuitive Explanation

A probability density function (PDF) describes how probability mass is “spread out” over a continuous variable.

Imagine measuring people’s heights: the PDF tells you how dense the probability is around each height value.

Rather than assigning a nonzero probability to an exact height (which would be zero), the PDF lets you compute probabilities over intervals.

### Formal Definition

For a continuous random variable X with PDF f_X(x):

```
f_X(x) ≥ 0
∫_{−∞}^{∞} f_X(x) dx = 1
P(a ≤ X ≤ b) = ∫_a^b f_X(x) dx
```

- f_X(x) is nonnegative everywhere.
- The total area under the curve is 1, ensuring all probabilities sum to 1.
- Probability of X falling between a and b equals the area under f_X(x) from a to b.

### Key Properties

- Non-negativity: f_X(x) ≥ 0 for all x.
- Normalization: ∫_{−∞}^{∞} f_X(x) dx = 1.
- Interval probabilities: you never ask for P(X = x) directly (it’s zero), but P(a ≤ X ≤ b) is meaningful.

### Step-by-Step Breakdown

1. **Identify X** and its support (e.g., all real numbers for a Normal distribution).
2. **Write down f_X(x)** using the distribution formula.
3. **Check normalization**: integrate f_X(x) over its support to confirm area = 1.
4. **Compute interval probability**: integrate f_X(x) between your desired bounds.

### Real-World Examples

- Heights of adult men (approximately Normal with µ≈175 cm, σ≈7 cm).
- Time between user clicks on a website (often modeled by Exponential with rate λ).
- Sensor noise in measurements (modeled as Gaussian around zero mean).

### Visual Intuition

Picture the PDF as a smooth curve above the x-axis.

The area under the curve between any two x-values (shaded region) equals the probability X lies in that interval.

### Practice Problems & Python Exercises

### 1. Uniform PDF on [0,1]

Compute P(0.2 ≤ X ≤ 0.7) analytically and by simulation:

```python
import numpy as np

# Analytic
p_analytic = 0.7 - 0.2

# Simulation
samples = np.random.uniform(0, 1, size=100_000)
p_empirical = ((samples >= 0.2) & (samples <= 0.7)).mean()

print("Analytic:", p_analytic, "Empirical:", p_empirical)
```

### 2. Normal PDF Integration

Estimate P(−1 ≤ X ≤ 2) for X∼N(0,1):

```python
from scipy.stats import norm

# Using CDF differences
p = norm.cdf(2, 0, 1) - norm.cdf(-1, 0, 1)
print("P(-1 ≤ X ≤ 2):", p)
```

### 3. Plotting a PDF

Plot the PDF of X∼Exponential(λ=0.5):

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

lam = 0.5
xs = np.linspace(0, 10, 400)
plt.plot(xs, expon.pdf(xs, scale=1/lam))
plt.title("Exponential PDF (λ=0.5)")
plt.xlabel("x"); plt.ylabel("f_X(x)")
plt.show()
```

---

## Cumulative Distribution Function (CDF)

### Intuitive Explanation

A cumulative distribution function (CDF) F_X(x) gives the probability that the random variable X is **at most** x.

It accumulates all probability mass up to that point, building a non-decreasing curve from 0 to 1.

### Formal Definition

For a continuous random variable X with PDF f_X(x), the CDF is:

```
F_X(x) = P(X ≤ x) = ∫_{−∞}^x f_X(t) dt
```

- F_X(x) starts at 0 as x→−∞ and approaches 1 as x→∞.
- It is right-continuous and non-decreasing.

### Relationship Between PDF and CDF

- **Derivative**: f_X(x) = d/dx [F_X(x)]
- **Integral**: F_X(x) = ∫_{−∞}^x f_X(t) dt

This duality links the instantaneous density (PDF) to the accumulated probability (CDF).

### Key Properties

- 0 ≤ F_X(x) ≤ 1 for all x.
- limₓ→−∞ F_X(x) = 0; limₓ→∞ F_X(x) = 1.
- P(a < X ≤ b) = F_X(b) − F_X(a).

### Real-World Examples

- **Service time**: F_X(t) tells you the chance a service completes within t seconds.
- **Customer wait time**: CDF shows percent of customers served before a given time.
- **Model evaluation**: computing percentiles (e.g., 90th percentile latency).

### Visual Intuition

The CDF is an S-shaped curve (for many distributions) creeping from 0 to 1.

At each x, the y-value equals the total probability mass left of x.

### Practice Problems & Python Exercises

### 1. Empirical CDF

Given data in `values`, plot the empirical CDF:

```python
import numpy as np
import matplotlib.pyplot as plt

values = np.random.normal(0, 1, size=1_000)
sorted_vals = np.sort(values)
cdf = np.arange(1, len(values)+1) / len(values)

plt.step(sorted_vals, cdf, where='post')
plt.title("Empirical CDF")
plt.xlabel("x"); plt.ylabel("F̂_X(x)")
plt.show()
```

### 2. Analytical vs. Empirical Normal CDF

```python
from scipy.stats import norm

# Analytic at x=1.5
print("F(1.5):", norm.cdf(1.5, 0, 1))

# Empirical
samples = np.random.normal(0, 1, size=100_000)
print("Empirical F(1.5):", (samples <= 1.5).mean())
```

### 3. Computing Percentiles

Find the 95th percentile of X∼Exp(λ=1):

```python
from scipy.stats import expon

pct95 = expon.ppf(0.95, scale=1)  # inverse CDF
print("95th percentile:", pct95)
```

### How Data Scientists Use CDFs

- Determining **percentile-based thresholds** for anomaly detection.
- Computing **quantile loss** in regression tasks.
- Designing **risk models** by estimating probabilities of exceeding critical values.
- Feature generation: using empirical CDF transforms to uniformize distributions.

---

## Uniform Distribution

### Intuitive Explanation

Imagine you pick a random point on a stick of length L without favoring any part. Every position is equally likely. That’s the continuous uniform distribution: it spreads probability evenly over an interval [a, b].

For example, if you generate a random number between 0 and 1 to shuffle data or initialize neural-network weights, you’re using a uniform distribution on [0, 1] (or [–r, r]).

### Formal Definition

Let X be a continuous random variable uniformly distributed on the interval [a, b]. Its probability density function (PDF) f_X(x) and cumulative distribution function (CDF) F_X(x) are:

```markdown
PDF: f_X(x) = 1 / (b - a)   for a ≤ x ≤ b
       f_X(x) = 0           otherwise
```

```markdown
CDF: F_X(x) = 0               for x < a
       F_X(x) = (x - a)/(b - a)   for a ≤ x ≤ b
       F_X(x) = 1               for x > b
```

### Key Properties

- Support: x ∈ [a, b]
- Total area under PDF = 1
- Mean: E[X] = (a + b) / 2
- Variance: Var[X] = (b - a)² / 12

### Step-by-Step Breakdown of the PDF

1. **Uniform density** means the height of f_X(x) is constant over [a, b].
2. To make total area = 1, height must be 1 divided by width (b−a).
3. Outside [a, b], probability is zero—X cannot take values there.

### Step-by-Step Breakdown of the CDF

1. For x below a, none of the mass has been “accumulated,” so F_X(x)=0.
2. Between a and x, the area under the PDF is rectangle width (x−a) times height 1/(b−a).
3. Above b, all mass is included, so F_X(x)=1.

### Real-World Examples in ML & DS

- **Parameter Initialization**: Weights in a neural network often start Uniform(–r, r) to break symmetry.
- **Random Sampling**: Shuffling indices uses uniform draws over [0, N).
- **Monte Carlo Integration**: Sampling x∼Uniform(a,b) to estimate ∫ₐᵇ g(x)dx.
- **Feature Scaling**: Transforming data to U(0,1) before training algorithms sensitive to scale.

### Practice Problems & Python Exercises

### 1. Simulate and Visualize Uniform PDF & CDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Parameters
a, b = 2.0, 5.0
dist = uniform(loc=a, scale=b-a)

# Sample points
xs = np.linspace(a - 1, b + 1, 400)

# Plot PDF
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(xs, dist.pdf(xs), 'b', lw=2)
plt.fill_between(xs, dist.pdf(xs), where=(xs>=a)&(xs<=b), alpha=0.3)
plt.title("PDF of Uniform(2,5)")
plt.xlabel("x"); plt.ylabel("f_X(x)")

# Plot CDF
plt.subplot(1,2,2)
plt.plot(xs, dist.cdf(xs), 'r', lw=2)
plt.title("CDF of Uniform(2,5)")
plt.xlabel("x"); plt.ylabel("F_X(x)")
plt.tight_layout()
plt.show()
```

### 2. Empirical vs. Theoretical Interval Probability

Compute P(3 ≤ X ≤ 4) analytically and by simulation:

```python
# Analytic
p_theoretical = (4 - 3) / (b - a)

# Simulation
samples = np.random.uniform(a, b, size=100_000)
p_empirical = ((samples >= 3) & (samples <= 4)).mean()

print(f"P(3 ≤ X ≤ 4) theoretical: {p_theoretical:.4f}")
print(f"P(3 ≤ X ≤ 4) empirical:  {p_empirical:.4f}")
```

### 3. Estimate Mean and Variance from Data

```python
# Simulate uniform samples
samples = np.random.uniform(a, b, size=100_000)

# Empirical
mean_emp = samples.mean()
var_emp  = samples.var(ddof=0)

# Theoretical
mean_theo = (a + b) / 2
var_theo  = (b - a)**2 / 12

print("Mean – theoretical vs empirical:", mean_theo, mean_emp)
print("Variance – theoretical vs empirical:", var_theo, var_emp)
```

### How Data Scientists Use Uniform Distributions

- **Baseline randomness**: when no prior structure exists, use uniform priors in Bayesian models.
- **A/B random assignment**: assign users evenly across variants.
- **Synthetic data generation**: create toy datasets with known bounds.
- **Hyperparameter search**: sample hyperparameters uniformly within chosen ranges.

---

## Normal Distribution

### Intuitive Explanation

Imagine you measure a large number of people’s heights. Most will cluster around the average, with fewer extremely tall or extremely short individuals. Plotting that frequency yields the familiar “bell curve.”

The normal distribution formalizes this pattern: it’s a continuous distribution where outcomes near the mean are most likely, and probabilities taper off symmetrically in both directions.

### Formal Definition

A random variable X is normally distributed with mean μ and standard deviation σ, written X∼N(μ, σ²), if its PDF is:

```
f_X(x) = 1 / (σ * sqrt(2 * pi)) * exp( - (x - μ)^2 / (2 * σ^2) )
```

Its CDF is:

```
F_X(x) = P(X ≤ x) = 0.5 * (1 + erf( (x - μ) / (σ * sqrt(2)) ))
```

- **μ** shifts the center (mean) of the bell.
- **σ** controls the spread (standard deviation).
- **erf(·)** is the “error function,” a special function capturing the integral of the Gaussian.

### Key Properties

- The curve is symmetric about x = μ.
- Total area under f_X(x) is 1.
- Approximately 68% of mass lies within one σ of μ, ~95% within two σ, ~99.7% within three σ.

### Step-by-Step Breakdown of the PDF

1. `1 / (σ * sqrt(2 * pi))`:
    - Ensures the total area under the curve equals 1.
    - `sqrt(2 * pi)` ≈ 2.5066 scales the height.
2. `exp( - (x - μ)^2 / (2 * σ^2) )`:
    - Assigns high density when x is close to μ (small squared difference).
    - Falls off quickly as |x − μ| grows, with rate controlled by σ² in the denominator.
3. Multiplying these gives the bell-shaped curve.

### Real-World Examples in ML & DS

- **Measurement Noise**: sensors often produce errors following a Normal distribution centered at zero.
- **Residuals in Regression**: ordinary least squares assumes residuals (errors) are Normally distributed.
- **Feature Standardization**: z-scoring a feature (subtract mean, divide by σ) maps it to a standard normal.
- **Gaussian Naive Bayes**: models continuous features Xᵢ | class c as N(μ*{c,i}, σ²*{c,i}).

### Visual & Geometric Intuition

- **PDF Plot**: the classic bell curve peaked at μ, width determined by σ.
- **68–95–99.7 Rule**: shade the region [μ–σ, μ+σ] (~68%), [μ–2σ, μ+2σ] (~95%) to see how quickly probability accumulates.
- **CDF Plot**: S-shaped curve rising from 0 to 1; the inflection point at x = μ.

### Practice Problems & Python Exercises

### 1. Compute Interval Probability

Problem: For X∼N(μ=10, σ=2), find P(8 ≤ X ≤ 12).

```python
from scipy.stats import norm

mu, sigma = 10, 2
p = norm.cdf(12, mu, sigma) - norm.cdf(8, mu, sigma)
print("P(8 ≤ X ≤ 12):", p)  # ≈ 0.6827
```

### 2. Monte Carlo Simulation

```python
import numpy as np

mu, sigma, trials = 10, 2, 100_000
samples = np.random.normal(mu, sigma, size=trials)
p_emp = ((samples >= 8) & (samples <= 12)).mean()
print("Empirical P(8 ≤ X ≤ 12):", p_emp)
```

### 3. Plotting the PDF and CDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = 0, 1
xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)

plt.figure(figsize=(10,4))

# PDF
plt.subplot(1,2,1)
plt.plot(xs, norm.pdf(xs, mu, sigma), 'b', lw=2)
plt.fill_between(xs, norm.pdf(xs, mu, sigma),
                 where=(xs>=-1)&(xs<=1), alpha=0.2)
plt.title("Standard Normal PDF")
plt.xlabel("x"); plt.ylabel("f_X(x)")

# CDF
plt.subplot(1,2,2)
plt.plot(xs, norm.cdf(xs, mu, sigma), 'r', lw=2)
plt.title("Standard Normal CDF")
plt.xlabel("x"); plt.ylabel("F_X(x)")

plt.tight_layout()
plt.show()
```

### 4. Z-Score Transformation

Problem: Convert X=15 from N(μ=10, σ=2) into a standard normal z-score and find its tail probability.

```python
from scipy.stats import norm

x, mu, sigma = 15, 10, 2
z = (x - mu) / sigma
p_tail = 1 - norm.cdf(z)
print("z-score:", z)          # 2.5
print("P(X > 15):", p_tail)   # ≈ 0.0062
```

### How Data Scientists Use the Normal Distribution

- **Hypothesis Testing**: many test statistics follow a normal or approximate-normal distribution under the null.
- **Confidence Intervals**: assuming Normal error lets you compute ± z*·σ/√n intervals.
- **Feature Engineering**: z-score standardization makes features comparable and improves convergence.
- **Anomaly Detection**: flag values lying far in the tails of a fitted Normal model.

---

## Chi-Squared Distribution

### Intuitive Explanation

The chi-squared distribution describes the distribution of the sum of squared standard normal variables.

If you draw k independent values from a Normal(0, 1) and square each one, then add them up, that sum follows a chi-squared with k degrees of freedom.

It models how “spread out” those squared deviations are, which underpins tests of variance and contingency‐table tests.

### Formal Definition

Let Z₁, Z₂, …, Z_k be independent Normal(0, 1) random variables. Define

```
X = Z₁² + Z₂² + … + Z_k²
```

Then X follows a chi-squared distribution with k degrees of freedom, written X ∼ χ²(k). Its PDF for x ≥ 0 is:

```
f_X(x; k) = 1 / (2^(k/2) * Gamma(k/2))
            * x^(k/2 - 1) * exp(-x/2)
```

The CDF uses the lower incomplete gamma function:

```
F_X(x; k) = γ(k/2, x/2) / Gamma(k/2)
```

- `Gamma(·)` is the gamma function.
- `γ(·,·)` is the lower incomplete gamma.

### Step-by-Step Breakdown of the PDF

1. **2^(k/2)** normalizes for degrees of freedom.
2. **Gamma(k/2)** ensures the total area under the curve is 1.
3. **x^(k/2 - 1)** shapes the curve: when k is small, it spikes near zero; larger k shifts mass right.
4. **exp(-x/2)** provides the exponential decay for large x.

### Real-World Examples in ML & DS

- **Feature selection**: the chi² statistic scores categorical features by measuring independence from target label.
- **Goodness-of-fit tests**: comparing observed vs. expected counts in bins (e.g., histogram of residuals).
- **Variance confidence intervals**: testing if population variance matches a specified value.

### Visual & Geometric Intuition

- For k=1, the PDF is heavily right-skewed, peaking near zero.
- As k increases, the distribution shifts right and becomes more symmetric, approaching Normal shape by k≈30.
- Plotting PDF curves for k=1, 2, 5, 10 shows how degrees of freedom stretch and smooth the curve.

### Practice Problems & Python Exercises

### 1. Plot PDF for Various k

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

xs = np.linspace(0, 20, 400)
for k in [1, 2, 5, 10]:
    plt.plot(xs, chi2.pdf(xs, df=k), label=f'k={k}')
plt.title("Chi‐squared PDFs")
plt.xlabel("x")
plt.ylabel("f_X(x)")
plt.legend()
plt.show()
```

### 2. Compute CDF and Quantiles

```python
from scipy.stats import chi2

k = 4
x = 7.78
print("CDF at x:", chi2.cdf(x, df=k))  # P(X ≤ x)

p = 0.95
print("95th percentile:", chi2.ppf(p, df=k))
```

### 3. Feature Selection with Chi-Squared

```python
from sklearn.feature_selection import chi2
import pandas as pd

# Assume df has categorical features encoded as counts and binary target y
X = pd.DataFrame({
    'feat1': [0,1,2,1,0,2,1],
    'feat2': [1,0,1,1,0,0,1]
})
y = [0,1,0,1,0,1,1]
chi2_scores, p_values = chi2(X, y)
print("Chi² scores:", chi2_scores)
print("p-values:", p_values)
```

### How Data Scientists Use the Chi-Squared Distribution

- Testing independence in contingency tables (e.g., click vs. device type).
- Selecting top k categorical features before training tree-based or linear models.
- Validating simulation outputs against theoretical distributions.
- Building confidence intervals for variance in process-control applications.

---

## Sampling from a Distribution

### Intuitive Explanation

Sampling from a distribution means generating random numbers that follow the shape of a target PDF or PMF.

This underlies Monte Carlo simulation, bootstrapping datasets, random initialization of model weights, and synthetic data generation.

### Common Sampling Methods

- **Direct (Library) Sampling**
    
    Use built-in functions (e.g., `numpy.random.normal`, `scipy.stats.expon.rvs`).
    
- **Inverse Transform Sampling**
    1. Draw u ∼ Uniform(0,1).
    2. Compute x = F⁻¹(u), where F⁻¹ is the inverse CDF (quantile function).
- **Rejection Sampling**
    1. Sample candidate x from an easy “proposal” distribution g(x).
    2. Accept with probability f(x)/(M·g(x)), where M≥sup_x f(x)/g(x).
    3. Repeat until enough samples accepted.

### Step-by-Step: Inverse Transform for Exponential

1. Uniform draw: u ∼ Uniform(0,1).
2. Exponential CDF: F(x)=1−exp(−λ x).
3. Solve for x: u=1−exp(−λ x) ⇒ x=−(1/λ)·ln(1−u).

Thus:

```python
import numpy as np

def sample_exponential(lam, size=1):
    u = np.random.rand(size)
    return -np.log(1 - u) / lam
```

### Practice Problems & Python Exercises

### 1. Sample Custom PDF with Inverse Transform

PDF f(x)=2x for x∈[0,1] (Beta(2,1)). Inverse CDF method:

```python
import numpy as np
import matplotlib.pyplot as plt

# f(x)=2x ⇒ F(x)=x^2 ⇒ F⁻¹(u)=sqrt(u)
u = np.random.rand(100_000)
samples = np.sqrt(u)

# Plot empirical vs theoretical
plt.hist(samples, bins=50, density=True, alpha=0.6)
xs = np.linspace(0,1,200)
plt.plot(xs, 2*xs, 'r--', label='PDF')
plt.legend(); plt.show()
```

### 2. Rejection Sampling for a Triangular PDF

Target f(x)=2x for x∈[0,1], proposal g(x)=Uniform(0,1), M=2:

```python
import numpy as np

def rejection_sample(n):
    samples = []
    while len(samples) < n:
        x = np.random.rand()
        u = np.random.rand()
        if u <= 2*x / 2:   # f(x)/(M*g(x)) = (2x)/(2*1)
            samples.append(x)
    return np.array(samples)

samps = rejection_sample(100_000)
```

### 3. Library Sampling & Analysis

```python
import numpy as np
from scipy.stats import gamma

# Sample from Gamma(k=2, θ=3)
data = gamma(a=2, scale=3).rvs(size=100_000)
print("Empirical mean, var:", data.mean(), data.var())
print("Theoretical mean, var:", 2*3, 2*3**2)
```

### How Data Scientists Use Sampling

- **Monte Carlo integration**: estimating integrals or expectations when analytic form is intractable.
- **Bootstrap**: drawing repeated samples with replacement to estimate confidence intervals.
- **Data augmentation**: generating new data points under assumed distributions.
- **Stochastic optimization**: minibatch sampling in gradient descent.

---