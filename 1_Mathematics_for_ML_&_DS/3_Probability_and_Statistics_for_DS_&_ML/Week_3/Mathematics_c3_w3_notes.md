# Mathematics_c3_w3

## Population and Sample

### 1. Core Definitions

A **population** is the complete set of all items, measurements, or observations you care about. In Data Science, it might be every customer, every transaction, or every pixel in an image.

A **sample** is a subset drawn from the population, used because measuring the entire population is often impossible, costly, or time-consuming.

A **parameter** is a numeric summary of the population (for example, the true average income of all households). A **statistic** is the same kind of summary computed on a sample (for example, the average income of 1,000 surveyed households).

### 2. Key Formulas

### 2.1 Population Mean and Sample Mean

The **population mean** (a parameter) is the average of all values in the population.

```
mu = (1 / N) * sum_{i=1 to N} x_i
```

- `mu` is the population mean.
- `N` is the number of items in the population.
- `x_i` is the value of the i-th item.

Step-by-step:

1. Sum every value in your population: `sum_{i=1 to N} x_i`.
2. Divide by the total count `N`.

The **sample mean** (a statistic) is the average of all values in your sample.

```
x_bar = (1 / n) * sum_{i=1 to n} x_i
```

- `x_bar` is the sample mean.
- `n` is the sample size.
- `x_i` is the value of the i-th sampled item.

Step-by-step:

1. Sum the values you actually collected: `sum_{i=1 to n} x_i`.
2. Divide by how many you collected: `n`.

### 2.2 Population Variance and Sample Variance

Variance measures how spread out values are around the mean.

Population variance (parameter):

```
sigma_squared = (1 / N) * sum_{i=1 to N} (x_i - mu)^2
```

Sample variance (statistic), using “n−1” for an unbiased estimate:

```
s_squared = (1 / (n - 1)) * sum_{i=1 to n} (x_i - x_bar)^2
```

- `sigma_squared` is true variance of the whole population.
- `s_squared` is variance estimated from a sample.
- The subtraction `(x_i - mean)` centers each value before squaring.
- Dividing by `n−1` corrects bias when estimating from limited data.

### 3. Visual Intuition

Imagine an urn filled with 10,000 colored balls (each ball’s color = a data value). That’s your population.

- If you could count and measure all balls, you’d compute `mu` exactly.
- Instead, you randomly draw 100 balls (your sample), measure them, and compute `x_bar`.

Plot a histogram of all 10,000 vs. a histogram of your 100 draws. You’ll see the sample histogram bouncing around the true shape. As you increase sample size, the sample histogram converges on the population’s.

### 4. Real-World ML & DS Applications

- **Mini-batch SGD**: At each step, you draw a sample (mini-batch) from your training set (a population) to approximate the true gradient.
- **Train/Test Split**: Your training set is a sample from the true data‐generation process; your test set is another sample used to estimate generalization.
- **Bootstrapping**: Repeatedly sample (with replacement) from your data to estimate the sampling distribution of a statistic (e.g., feature importance).
- **A/B Testing**: You compare statistics from two samples (version A vs. B) to infer which variation truly outperforms in the full user population.

### 5. Practice Problems & Python Exercises

### 5.1 Exercise 1: Manual Calculation

Dataset (population): heights in cm = [160, 165, 170, 155, 180]

1. Compute the population mean `mu` and population variance `sigma_squared`.
2. Now imagine you only measured the first three heights (sample = [160, 165, 170]). Compute `x_bar` and `s_squared`.

### 5.2 Exercise 2: Python with NumPy

```python
import numpy as np

# Full population
pop = np.array([160, 165, 170, 155, 180])
mu = pop.mean()
sigma2 = pop.var(ddof=0)

# Sample of size 3
sample = pop[:3]
x_bar = sample.mean()
s2 = sample.var(ddof=1)

print("Population mean:", mu)
print("Population variance:", sigma2)
print("Sample mean:", x_bar)
print("Sample variance:", s2)
```

- `ddof=0` computes population variance; `ddof=1` computes sample variance.

### 5.3 Exercise 3: Monte Carlo Sampling & CLT Preview

1. Generate a large “population” of 10,000 draws from a skewed distribution (e.g., exponential).
2. Take 1,000 samples of size 50 each; compute each sample’s mean.
3. Plot the distribution of those 1,000 sample means. Notice how it becomes bell-shaped, hinting at the Central Limit Theorem.

---

## Sample Mean

### 1. Quick Prerequisite Reminder

You already know that a **population** is the full set of items you care about, and a **sample** is a smaller subset you actually observe.

The **sample size**, denoted by *n*, is the number of observations in your sample.

### 2. What Is the Sample Mean?

The **sample mean** is the arithmetic average of the numbers in your sample.

Think of the sample mean as the “balancing point” of your data: if each data point were a weight on a ruler, the sample mean is where you’d place a fulcrum so the ruler stays level.

Why it matters in ML & DS:

- It’s the simplest estimate of the center of your data.
- It forms the basis for variance, standard deviation, and many statistical tests.
- In mini-batch gradient descent, each batch’s mean loss helps approximate the true gradient.

### 3. Formula and Breakdown

```
x_bar = (1 / n) * sum_{i=1 to n} x_i
```

- `x_bar` (read “x-bar”) is the sample mean.
- `n` is the number of observations in your sample.
- `x_i` is the i-th observed value in your sample.
- `sum_{i=1 to n} x_i` means add up all sample values from i = 1 to n.
- Dividing by `n` gives you the average.

Step by step:

1. List your n data points: x₁, x₂, …, xₙ.
2. Compute the sum: S = x₁ + x₂ + … + xₙ.
3. Divide by n: x_bar = S / n.

### 4. Geometric Intuition

Imagine plotting your sample points on a number line.

- The sample mean is the point where the total “distance” to the left equals the total “distance” to the right.
- In two dimensions, it’s the center of mass of points on a scatter plot.

Visualize a histogram of your sample:

- The bar heights show frequencies.
- Overlay a vertical line at x_bar.
- That line splits the total weight of all bars so the areas on each side are (roughly) balanced.

### 5. Real-World Examples in Data Science

1. **Feature Centering**
    - Before training, you often subtract the sample mean from each feature so that the new feature has mean zero.
    - This speeds up convergence in gradient-based optimizers.
2. **Batch Loss in Mini-Batch SGD**
    - For each mini-batch, you compute the average loss:x_bar = (1/m) * ∑loss_i
    - That average approximates the true loss over the entire training set.
3. **A/B Testing**
    - You compare the sample mean conversion rate of group A vs. group B to decide which version performs better.

### 6. Practice Problems & Python Exercises

### 6.1 Manual Calculation

Dataset: daily sales in units = [12, 15, 9, 20, 14]

1. Compute the sample mean.
2. Explain in one sentence why this average represents the center of your sales data.

### 6.2 Python: Compute and Compare

```python
import numpy as np

sales = np.array([12, 15, 9, 20, 14])
n = sales.size

# Compute sample mean
x_bar = sales.sum() / n
print("By formula:", x_bar)

# Using built-in
print("Using numpy.mean:", sales.mean())
```

- Verify both methods give the same result.
- Change one value (e.g., replace 9 with 30) and observe how the mean shifts.

### 6.3 Monte Carlo Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate a “population” of 50,000 exam scores (skewed)
population = np.random.exponential(scale=60, size=50000)

# Draw 1000 samples of size 30, compute their means
means = [np.random.choice(population, size=30).mean() for _ in range(1000)]

# Plot the sampling distribution of the mean
plt.hist(means, bins=30, edgecolor='k')
plt.axvline(np.mean(means), color='red', linestyle='--', label='Mean of sample means')
plt.xlabel('Sample mean')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

- Observe how the spread of sample means is much narrower than the spread of individual scores.
- This illustrates that the sample mean has lower variability than single observations.

---

## Sample Proportion

### 1. Prerequisite Reminder

You already know that a **population** is the full set of items you care about and a **sample** is the subset you actually observe.

You understand the **sample size** *n* and the **sample mean** (`x_bar`) as the average of numeric values.

Now we’ll focus on proportions—when each observation is a “success” or “failure” and we care about the fraction of successes.

### 2. What Is the Sample Proportion?

The **sample proportion**, denoted `p_hat`, is the fraction of observations in your sample that have a particular attribute (a “success”).

Analogy: imagine you draw 100 balls from a huge urn, where each ball is either red (success) or blue (failure). If 27 balls are red, your sample proportion of red balls is 27/100 = 0.27.

In data science, this could be the fraction of users who click an ad, the defect rate on an assembly line, or the proportion of emails marked spam.

### 3. Formula and Step-by-Step Breakdown

First, count the number of “successes” in your sample, call that `x`. Then divide by the sample size `n`:

```
p_hat = x / n
```

- `p_hat` is the sample proportion.
- `x` is the number of successes (observations with the attribute).
- `n` is the total number of observations in your sample.

Alternate form using indicator variables (I_i), where (I_i=1) if observation *i* is a success, else 0:

```
p_hat = (1 / n) * sum_{i=1 to n} I_i
```

- `sum_{i=1 to n} I_i` is just counting how many observations met the criterion.
- Dividing by `n` converts that count into a fraction or percentage.

### 4. Visual and Geometric Intuition

Imagine repeating the “urn draw” 1,000 times, always taking samples of size *n* and computing `p_hat`.

If you plot a histogram of those 1,000 `p_hat` values, you’ll see a bell-shaped curve centered near the true population proportion *p*.

As you increase *n*, the histogram of sample proportions tightens around *p*, illustrating that larger samples give more precise estimates.

### 5. Real-World DS & ML Applications

- A/B testing: Compare `p_hat_A` and `p_hat_B` to see which website version has a higher conversion rate.
- Class balance: Estimate the proportion of positive class labels in your training set before training a classifier.
- Quality control: Track the defect rate (fraction of faulty products) on a production line.
- Feature engineering: Compute the proportion of missing values in a column to decide if imputation or removal is needed.

### 6. Practice Problems & Python Exercises

### 6.1 Manual Calculation

Dataset: In a survey of 50 people, 18 say they prefer tea over coffee.

1. Compute the sample proportion `p_hat`.
2. Explain why dividing by 50 gives an estimate of the population preference.

### 6.2 Python: Computing a Sample Proportion

```python
import numpy as np

# 1 indicates "success" (prefers tea), 0 indicates "failure"
responses = np.array([1]*18 + [0]*32)
n = responses.size

# Method 1: count and divide
x = responses.sum()
p_hat = x / n

# Method 2: average of 0/1 array
p_hat_alt = responses.mean()

print("Sample proportion (p_hat):", p_hat)
print("Alternate calculation:", p_hat_alt)
```

- Verify both methods yield the same result.
- Change the number of successes and observe how `p_hat` shifts.

### 6.3 Simulation: Sampling Distribution of p_hat

```python
import numpy as np
import matplotlib.pyplot as plt

# True population proportion
p_true = 0.3
population_size = 100000

# Create a large population of 0/1
population = np.random.binomial(1, p_true, size=population_size)

# Draw many samples of size 50 and compute sample proportions
sample_size = 50
num_samples = 1000
p_hats = [np.random.choice(population, sample_size).mean()
          for _ in range(num_samples)]

# Plot histogram
plt.hist(p_hats, bins=20, edgecolor='k')
plt.axvline(p_true, color='red', linestyle='--', label='True p')
plt.xlabel('Sample proportion p_hat')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

- Observe how the sample proportions cluster around the true `p_true`.
- Notice the spread shrinks when you increase `sample_size`.

---

## Sample Variance

### 1. Prerequisite Reminder

You already understand:

- A sample is the subset of data points you observe, of size *n*.
- The sample mean, `x_bar`, is the average of those *n* observations.

Sample variance builds on these to measure how spread‐out your data are around the mean.

### 2. What Is Sample Variance?

Sample variance, written `s_squared`, quantifies the average squared distance of each observation from the sample mean.

Analogy: imagine each data point is a weight on a ruler balanced at `x_bar`. The squared distances show how much each weight “pulls” the ruler away. Squaring ensures all distances are positive and gives more emphasis to points far from the center.

### 3. Formula and Breakdown

```
s_squared = (1 / (n - 1)) * sum_{i=1 to n} (x_i - x_bar)^2
```

- `s_squared` is the sample variance.
- `n` is the number of observations in your sample.
- `x_i` is the i-th sample value.
- `x_bar` is the sample mean, computed as (1/n) * sum_{i=1 to n} x_i.
- The term `(x_i - x_bar)` measures each point’s deviation from the mean.
- Squaring `(x_i - x_bar)^2` makes deviations positive and amplifies larger gaps.
- Dividing by `(n – 1)` (instead of `n`) corrects bias—this “degrees of freedom” adjustment ensures that on average, `s_squared` equals the true population variance.

Step by step:

1. Compute the sample mean `x_bar`.
2. For each data point, find its deviation: `d_i = x_i - x_bar`.
3. Square each deviation: `d_i^2`.
4. Sum all squared deviations: `sum_{i=1 to n} d_i^2`.
5. Divide by `n – 1` to get `s_squared`.

### 4. Geometric Intuition

Picture your *n* data points on a number line and their center at `x_bar`.

- Each point’s deviation is a horizontal distance to `x_bar`.
- Squaring turns these distances into areas—large distances cover more “area,” pulling the average up.
- The variance is the average area per deviation, adjusted by `n–1`.

On a scatter plot, variance relates to the “cloud” of points around its center: a tight cloud means low variance; a wide cloud means high variance.

### 5. Real-World ML & DS Applications

- Feature Scaling: Many algorithms assume features have similar spread. You compute variance to standardize features (subtract mean, divide by standard deviation).
- Feature Selection: Variance thresholding removes features whose variance is below a cutoff—those add little information.
- PCA (Principal Component Analysis): PCA uses covariance (and hence variance) to find directions of maximum spread.
- Regularization Diagnostics: In Bayesian models, variances belong to prior/posterior distributions that control model flexibility.

### 6. Practice Problems & Python Exercises

### 6.1 Manual Calculation

Dataset: [5, 7, 3, 9]

1. Compute the sample mean `x_bar`.
2. Find each deviation, square them, sum them.
3. Divide by `(n – 1)` to get `s_squared`.

### 6.2 Python: Computing Sample Variance

```python
import numpy as np

data = np.array([5, 7, 3, 9])
n = data.size

# Method 1: manual
x_bar = data.mean()
s_squared_manual = ((data - x_bar)**2).sum() / (n - 1)

# Method 2: built-in with ddof=1
s_squared_builtin = data.var(ddof=1)

print("Manual sample variance:", s_squared_manual)
print("Built-in sample variance:", s_squared_builtin)
```

- Confirm both methods match.
- Change one value (e.g., replace 9 with 20) and observe how variance grows.

### 6.3 Simulation: Sampling Distribution of Sample Variance

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a “population” of 10000 values from a normal distribution
population = np.random.normal(loc=50, scale=10, size=10000)

# Draw 1000 samples of size 30 and compute their variances
sample_variances = [
    np.random.choice(population, 30).var(ddof=1)
    for _ in range(1000)
]

plt.hist(sample_variances, bins=30, edgecolor='k')
plt.xlabel('Sample variance')
plt.ylabel('Frequency')
plt.title('Sampling distribution of sample variance')
plt.show()
```

- Notice how sample variances scatter around the true population variance (~100).
- Increasing sample size tightens this distribution.

---

## Law of Large Numbers

### 1. Prerequisite Reminder

You already know that a **sample mean** `x_bar` is your best guess for the true population mean `mu` when you only see *n*observations.

You’ve seen that as *n* grows, the sample mean “settles down” more tightly around the true center.

The Law of Large Numbers (LLN) formalizes **why** and **how** that settling happens.

### 2. What Is the Law of Large Numbers?

The LLN states that as you increase your sample size, the sample average converges to the true population mean.

In plain terms: the more data you collect, the closer your average measurement gets to the real average.

Analogy:

Imagine flipping a fair coin. Early on—say after 10 flips—you might see 7 heads. That’s 0.7, quite far from the true probability 0.5.

After 1,000 flips, you might get 512 heads (0.512). Already much closer.

After 1,000,000 flips, you’ll see something like 500,123 heads (0.500123), almost spot on.

### 3. Formal Statement & Formula

### 3.1 Weak Law of Large Numbers

```
For any ε > 0:
  lim_{n → ∞} P( | x_bar_n – mu | < ε ) = 1
```

- `x_bar_n` is the sample mean of *n* observations.
- `mu` is the true population mean.
- `ε` is any small positive number (how tightly you want to be around `mu`).
- `P(...)` is probability.
- The statement says: as *n* goes to infinity, the probability that `x_bar_n` falls within `±ε` of `mu` goes to 1.

Step by step:

1. Pick how close you want your sample mean to the true mean (`ε`).
2. As you increase *n*, the chance that `|x_bar_n – mu|` exceeds `ε` shrinks toward zero.
3. In practice, large samples almost certainly produce averages very near `mu`.

### 3.2 Strong Law of Large Numbers (for completeness)

```
P( lim_{n → ∞} x_bar_n = mu ) = 1
```

- This stronger version says that with probability 1, the sequence of sample means actually converges to `mu` (not just in probability).

### 4. Geometric & Visual Intuition

Picture a plot of sample means against sample size *n*:

- On the x-axis: n = 10, 20, 50, …, 1000
- On the y-axis: observed x_bar_n

Early on, the line jumps around. As *n* grows, the jagged path narrows in on the horizontal line `y = mu`.

Histogram view:

- For *n* = 10, the histogram of repeated experiments is wide.
- For *n* = 100, it’s tighter.
- For *n* = 1000, it’s even more clustered around `mu`.

### 5. Real-World ML & DS Applications

- Mini-batch Gradient Estimates
    
    Each mini-batch gradient is an average of sample gradients. As batch size grows, you approximate the true gradient better, stabilizing training updates.
    
- Monte Carlo Simulation
    
    Estimating expectations (e.g., expected return in reinforcement learning) by averaging many random samples. The LLN guarantees your estimate improves with more samples.
    
- A/B Testing Stability
    
    With enough users in each group, your observed conversion rates converge to the true performance, reducing the risk of false positives.
    
- Estimating Data Set Statistics
    
    When computing feature means or variances on huge data sets, streaming algorithms rely on LLN to give accurate running estimates as more data flows in.
    

### 6. Practice Problems & Python Exercises

### 6.1 Coin-Flip Simulation

1. Simulate flipping a fair coin *n* times for increasing *n* = [10, 100, 1 000, 10 000].
2. Compute the sample proportion of heads each time.
3. Plot sample proportion vs. *n* and observe convergence to 0.5.

```python
import numpy as np
import matplotlib.pyplot as plt

true_p = 0.5
ns = [10, 100, 1000, 10000]
proportions = []

for n in ns:
    flips = np.random.binomial(1, true_p, size=n)
    proportions.append(flips.mean())

plt.plot(ns, proportions, marker='o')
plt.axhline(true_p, color='red', linestyle='--', label='True p')
plt.xscale('log')
plt.xlabel('Sample size n')
plt.ylabel('Sample proportion of heads')
plt.legend()
plt.show()
```

### 6.2 Exponential Distribution Averages

1. Generate a “population” of 100 000 samples from an exponential distribution with mean 5.
2. For *n* in [10, 50, 200, 1000], draw samples of size *n* and compute their means.
3. Repeat 1000 times for each *n*, plot histograms of means to see the tightening around 5.

### 6.3 Theoretical Exercise

Let `X_i` be i.i.d. with mean `mu` and finite variance. Show that `E[x_bar_n] = mu` and `Var(x_bar_n) = Var(X)/n`. Explain how the decreasing variance supports the LLN.

---

## Central Limit Theorem for Discrete Random Variables

### 1. Prerequisite Reminder

You know a **discrete random variable** takes countable values (like outcomes of a die).

You’ve seen the **sample mean** `x_bar` as the average of independent observations and the **Law of Large Numbers** telling you it converges to the true mean `mu`.

You also understand **expectation** `E[X] = mu` and **variance** `Var(X) = sigma_squared` for a single discrete variable.

### 2. Intuitive Explanation

Imagine rolling a fair six-sided die *n* times and summing the results.

For small *n*, your sum jumps around between  *n* and *6n*, and its shape is jagged.

As you increase *n*—say 30, 100, 1,000—the histogram of your sums smooths out into the familiar bell curve.

This happens no matter if the die is fair or loaded, or even if your outcomes are just zeros and ones (a Bernoulli variable).

### 3. Formal Statement & Formula

Let X₁, X₂, …, Xₙ be independent, identically distributed discrete random variables with

- Mean μ = E[Xᵢ]
- Variance σ² = Var(Xᵢ])

Define the standardized sum Zₙ:

```
Z_n = ( (sum_{i=1 to n} X_i) - n * mu ) / sqrt(n * sigma_squared)
```

- `sum_{i=1 to n} X_i` is the total across n draws.
- `n * mu` shifts that total by its expected value.
- `sqrt(n * sigma_squared)` scales the spread so Zₙ has unit variance.

The CLT says:

```
As n → ∞:
  P( a <= Z_n <= b ) → Phi(b) - Phi(a)
```

- `Phi` is the cumulative distribution function of a standard normal.
- In words, Zₙ’s distribution approaches N(0,1) for large n.

### 4. Geometric & Visual Intuition

Picture a series of bar charts showing sums of dice rolls:

- For n=5, the bars peak at 17 or 18 but still look jagged.
- For n=30, the bars form a smoother hump.
- For n=100, they trace out nearly a perfect bell.

Overlay a smooth normal curve on each histogram. You’ll see the discrete bars “fill in” that curve as n grows.

### 5. Real-World DS & ML Applications

- **Bootstrapping Discrete Data**
    
    When resampling counts or labels, the average of many bootstrap samples is approximated by a normal distribution for confidence intervals.
    
- **Ensemble Methods**
    
    Bagging aggregates discrete tree predictions; the distribution of average votes approaches normal, letting you estimate prediction uncertainty.
    
- **Event-Count Modeling**
    
    Summing daily website clicks (discrete counts) over weeks yields nearly normal behavior, justifying Gaussian-based control charts.
    
- **Approximate Inference**
    
    In probabilistic graphical models, sums of discrete latent variables can be approximated by normals to speed up calculations (e.g., variational methods).
    

### 6. Practice Problems & Python Exercises

### 6.1 Dice-Roll Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate_sum(n, trials=10000):
    sums = np.random.randint(1, 7, size=(trials, n)).sum(axis=1)
    return sums

for n in [5, 30, 100]:
    data = simulate_sum(n)
    plt.hist(data, bins=range(n, 6*n+2), density=True, alpha=0.6, edgecolor='k')
    mu = n * 3.5
    sigma = np.sqrt(n * ((35/12)))
    xs = np.linspace(n, 6*n, 200)
    plt.plot(xs, norm.pdf(xs, mu, sigma), 'r--')
    plt.title(f'Distribution of sum for n={n}')
    plt.show()
```

- Observe how the red normal curve fits bars better as *n* increases.

### 6.2 Bernoulli Sample-Mean Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

p = 0.3
for n in [10, 50, 200]:
    means = np.random.binomial(1, p, size=(10000, n)).mean(axis=1)
    plt.hist(means, bins=20, density=True, alpha=0.6, edgecolor='k')
    mu, sigma = p, np.sqrt(p*(1-p)/n)
    xs = np.linspace(0,1,200)
    plt.plot(xs, norm.pdf(xs, mu, sigma), 'r--')
    plt.title(f'Sample mean distribution for n={n}')
    plt.show()
```

- Notice the clustering of sample means around p and the narrowing spread.

### 6.3 Theoretical Exercise

- Prove that as n→∞, the moment generating function of Zₙ approaches that of N(0,1).
- Explain why discretization (integer outcomes) doesn’t prevent convergence to a continuous normal curve.

---

## Central Limit Theorem for Continuous Random Variables

### 1. Prerequisite Reminder

You know that a continuous random variable can take any value in an interval (for example, heights or weights).

You understand its **mean** μ = E[X] and **variance** σ² = Var(X).

You’ve seen the Law of Large Numbers ensuring the sample mean converges to μ as sample size *n* grows.

### 2. Intuitive Explanation

Imagine drawing *n* independent samples from a continuous distribution—say uniform on [0, 1] or exponential with mean 2.

For small *n*, the average of those values can wiggle widely.

As you increase *n*—to 30, 100, 1 000—the distribution of the sample average smooths into the bell curve, no matter the original shape.

### 3. Formal Statement & Formula

Let X₁, X₂, …, Xₙ be independent, identically distributed continuous random variables with

- mean μ = E[Xᵢ]
- variance σ² = Var(Xᵢ])

Define the standardized mean Zₙ:

```
Z_n = ( x_bar_n - mu ) / ( sigma / sqrt(n) )
```

- `x_bar_n` = (1/n) * sum_{i=1 to n} X_i
- `sigma / sqrt(n)` scales spread of the mean

The Central Limit Theorem states:

```
As n → ∞:
  Z_n converges in distribution to N(0, 1)
```

In practice, for large *n*,

```
x_bar_n ≈ Normal( mu,  sigma^2 / n )
```

### 4. Geometric & Visual Intuition

Visualize histograms of sample means for different *n*:

- For **uniform [0,1]**, original data is flat.
- For **n = 5**, sample-mean histogram is still somewhat flat-triangular.
- For **n = 30**, it becomes noticeably hump-shaped.
- For **n = 100**, it matches a smooth Gaussian curve.

Overlay the normal density on each histogram to see the fit improve as *n* grows.

### 5. Real-World DS & ML Applications

- Monte Carlo Integration
    
    Average many continuous random draws to approximate integrals or expected values; CLT gives error margins.
    
- Ensemble Predictions
    
    Averaging continuous outputs (regression scores) from multiple models yields a nearly normal distribution of the ensemble mean.
    
- Sensor Fusion
    
    Combining continuous readings (e.g., temperature sensors) reduces noise; the average reading’s error approximates normal.
    
- Loss Averaging in Training
    
    Computing mean squared error over mini-batches of continuous targets relies on CLT to justify using Gaussian-based stopping criteria.
    

### 6. Practice Problems & Python Exercises

### 6.1 Simulation with Uniform Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_sample_means(dist_func, params, ns, trials=5000):
    for n in ns:
        samples = dist_func(*params, size=(trials, n))
        means = samples.mean(axis=1)
        plt.hist(means, bins=30, density=True, alpha=0.6, edgecolor='k')
        mu = params[0] if dist_func is np.random.exponential else 0.5
        sigma = (1/np.sqrt(12)) if dist_func is np.random.rand else params[1]
        if dist_func is np.random.exponential:
            sigma = params[1]
        sigma_mean = sigma / np.sqrt(n)
        xs = np.linspace(means.min(), means.max(), 200)
        plt.plot(xs, norm.pdf(xs, mu, sigma_mean), 'r--')
        plt.title(f'n={n}')
        plt.show()

# Uniform [0,1]
plot_sample_means(lambda size: np.random.rand(*size), (), [5, 30, 100])
```

- Observe how the sample-mean histogram approaches the red normal curve.

### 6.2 Exponential Distribution Averages

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

ns = [10, 50, 200]
scale = 2.0  # mean of exponential
for n in ns:
    data = np.random.exponential(scale, size=(5000, n)).mean(axis=1)
    plt.hist(data, bins=30, density=True, alpha=0.6, edgecolor='k')
    mu, sigma = scale, scale
    sigma_mean = sigma / np.sqrt(n)
    xs = np.linspace(0, data.max(), 200)
    plt.plot(xs, norm.pdf(xs, mu, sigma_mean), 'r--')
    plt.title(f'Sample mean of exponential, n={n}')
    plt.show()
```

- Notice the convergence to normal for increasing *n*.

### 6.3 Theoretical Exercise

Show that for continuous Xᵢ with mgf M(t), the mgf of Zₙ, M_Zₙ(t), satisfies

```
M_Zₙ(t) = [ M( t / (sigma * sqrt(n)) ) ]^n * exp( - t * sqrt(n) * mu / sigma )
```

and that as n→∞, M_Zₙ(t) → exp( t^2 / 2 ), the mgf of N(0,1).

---

## Introduction to Point Estimation

### 1. What Is Point Estimation?

Point estimation is the task of using your sample data to produce a single best guess for an unknown population parameter.

You start with a sample of *n* observations and choose a formula (an estimator) that maps those observations to one number (the point estimate).

That point estimate is your “best shot” at the true value of a parameter—like the population mean, proportion, or variance—before you quantify your uncertainty.

### 2. Key Terms

- Estimator: A rule or function of the sample data that yields an estimate (for example, the sample mean formula).
- Estimate (Point Estimate): The actual number you compute from your data (for example, x̄ = 23.5).
- Parameter: The fixed but unknown quantity in the population you want to learn (for example, the true average height μ).
- Sampling Distribution: The probability distribution of your estimator over all possible samples of size *n*.

### 3. Desirable Properties of Estimators

An ideal estimator should have these properties:

- Unbiasedness: Its expected value equals the true parameter.
- Consistency: It converges to the true parameter as sample size grows.
- Efficiency: It has the smallest possible variance among unbiased estimators.
- Minimum Mean Squared Error (MMSE): It minimizes bias² + variance.

### 4. Formulas and Breakdown

### 4.1 Bias of an Estimator

```
Bias(θ̂) = E[θ̂] – θ
```

- `θ̂` is your estimator (random because it depends on the sample).
- `E[θ̂]` is its expected value.
- `θ` is the true parameter.
- An unbiased estimator has Bias(θ̂) = 0.

### 4.2 Mean Squared Error (MSE)

```
MSE(θ̂) = E[(θ̂ – θ)^2] = Var(θ̂) + [Bias(θ̂)]^2
```

- Measures average squared distance between estimator and true parameter.
- Lower MSE means better overall performance.

### 5. Examples of Common Estimators

### 5.1 Sample Mean as Estimator for Population Mean

```
Estimator:   θ̂ = x̄ = (1/n) * sum_{i=1 to n} x_i
```

- Unbiased: E[x̄] = μ.
- Variance: Var(x̄) = σ² / n (shrinks as n grows).

### 5.2 Proportion Estimator for Population Proportion

```
Estimator:   θ̂ = p̂ = x / n
```

- Unbiased: E[p̂] = p.
- Variance: Var(p̂) = p(1−p) / n.

### 5.3 Sample Variance as Estimator for Population Variance

```
Estimator:   θ̂ = s² = (1/(n−1)) * sum_{i=1 to n} (x_i − x̄)^2
```

- Unbiased: E[s²] = σ².
- Used for estimating spread around the mean.

### 6. Visual Intuition

Picture a dartboard’s bullseye as the true parameter θ.

- Each sample yields one dart (the point estimate).
- Unbiased but high-variance estimators scatter darts uniformly around the bullseye.
- Biased estimators shift darts away from center, even if they’re clustered tightly.
- Consistent estimators’ darts become more concentrated on the bullseye as sample size grows.

Plotting the sampling distribution of θ̂ for different n shows how it narrows and centers on θ over repeated experiments.

### 7. Real-World DS & ML Applications

- Estimating feature means and variances before standardization.
- Learning regression coefficients via maximum likelihood (MLE) or least squares (a point estimate for weights).
- Tuning hyperparameters: treating performance metrics on validation sets as point estimates of generalization.
- Bootstrapping: generating many θ̂ from resampled data to approximate its sampling distribution and assess estimator variability.

### 8. Practice Problems & Python Exercises

### 8.1 Manual Calculation

Data: [8, 12, 15, 7, 10]

1. Compute the sample mean x̄.
2. Compute its bias if the true μ = 11.
3. Compute sample variance s² and compare to true σ² = 10.

### 8.2 Python: Estimator Properties

```python
import numpy as np

def experiment(n, true_mu=5, true_sigma=2, trials=10000):
    estimates = []
    for _ in range(trials):
        sample = np.random.normal(true_mu, true_sigma, size=n)
        estimates.append(sample.mean())
    estimates = np.array(estimates)
    print("Bias:", estimates.mean() - true_mu)
    print("Variance:", estimates.var(ddof=0))
    print("MSE:", np.mean((estimates - true_mu)**2))

for n in [5, 20, 100]:
    print(f"n={n}")
    experiment(n)
    print()
```

- Observe bias near zero and variance decreasing with n.

### 8.3 Theory: MLE vs. Method of Moments

1. For an exponential distribution with density λe^(−λx), derive the MLE for λ.
2. Use the method of moments (set sample mean = 1/λ) and compare to the MLE.

---

## Maximum Likelihood Estimation

### 1. Prerequisites

You know that point estimation produces a single best guess for an unknown population parameter.

You’ve seen estimators like the sample mean and sample proportion, which map data to numbers.

Maximum Likelihood Estimation (MLE) is a systematic way to pick the estimator that makes your observed data most probable under a chosen model.

### 2. Conceptual Explanation

Imagine you have a glove mold that can stretch to different sizes (the parameter). You’ve observed several glove impressions (data). You want the mold size that would most likely produce those impressions.

In MLE:

1. You assume a probability model for your data, parameterized by θ.
2. You ask, “For each possible θ, how likely is it that I would have seen exactly my data?”
3. You choose the θ that maximizes that likelihood.

Analogy:

- Betting on a coin’s bias. You flip it 100 times, see 70 heads. You ask, “If the coin’s true bias were p, what’s the probability of seeing exactly 70 heads?” You pick the p that makes that probability highest.

### 3. Likelihood and Log-Likelihood

Given independent observations x₁, x₂, …, xₙ from a model with parameter θ:

1. **Likelihood**
    
    ```
    L(θ | x₁,…,xₙ) = ∏_{i=1 to n} f(x_i; θ)
    ```
    
    - `f(x_i; θ)` is the probability mass (discrete) or density (continuous) at xᵢ.
    - The product combines independent contributions.
2. **Log-Likelihood**
    
    ```
    ℓ(θ) = log L(θ | x₁,…,xₙ) = ∑_{i=1 to n} log f(x_i; θ)
    ```
    
    - Taking log turns products into sums.
    - Maximizing ℓ(θ) is equivalent to maximizing L(θ).

### 4. Finding the MLE

1. **Write down ℓ(θ)** for your model and data.
2. **Differentiate** ℓ(θ) with respect to θ to get the score function:
    
    ```
    S(θ) = dℓ(θ) / dθ
    ```
    
3. **Set the score to zero** and solve for θ:
    
    ```
    S(θ̂_MLE) = 0
    ```
    
4. **Check the second derivative** at θ̂_MLE to ensure it’s a maximum:
    
    ```
    d²ℓ(θ) / dθ² |_{θ=θ̂_MLE} < 0
    ```
    

### 5. Common MLE Examples

### 5.1 Bernoulli Distribution (Coin Toss)

Data: xᵢ ∈ {0,1}, i = 1…n, with success probability p.

- Likelihood:
    
    ```
    L(p) = p^{x₁} (1−p)^{1−x₁} × … × p^{xₙ} (1−p)^{1−xₙ}
         = p^x (1−p)^{n−x},  where x = ∑ xᵢ
    ```
    
- Log-Likelihood:
    
    ```
    ℓ(p) = x log p + (n−x) log(1−p)
    ```
    
- Score and solution:
    
    ```
    dℓ/dp = x/p − (n−x)/(1−p) = 0
    ⇒ p̂_MLE = x / n
    ```
    

### 5.2 Exponential Distribution

Data: xᵢ ≥ 0 from density f(x;λ) = λ e^{−λx}.

- Log-Likelihood:
    
    ```
    ℓ(λ) = ∑_{i=1 to n} [ log λ − λ x_i ]
          = n log λ − λ ∑ x_i
    ```
    
- Score and solution:
    
    ```
    dℓ/dλ = n/λ − ∑ x_i = 0
    ⇒ λ̂_MLE = n / ∑ x_i
    ```
    

### 5.3 Normal Distribution (Unknown Mean & Variance)

Data: xᵢ ∼ Normal(μ, σ²).

- Log-Likelihood:
    
    ```
    ℓ(μ,σ²) =
      − (n/2) log(2πσ²)
      − (1 / (2σ²)) ∑ (x_i − μ)²
    ```
    
- Solving for μ̂:
    
    ```
    ∂ℓ/∂μ = (1/σ²) ∑ (x_i − μ) = 0
    ⇒ μ̂_MLE = (1/n) ∑ x_i = x̄
    ```
    
- Solving for σ̂²:Note: This uses 1/n, not 1/(n−1).
    
    ```
    ∂ℓ/∂σ² = − (n/(2σ²)) + (1 / (2σ⁴)) ∑ (x_i − μ̂)² = 0
    ⇒ σ̂²_MLE = (1/n) ∑ (x_i − x̄)²
    ```
    

### 6. Geometric & Visual Intuition

Plot the likelihood function L(θ) (or ℓ(θ)) on the vertical axis against θ on the horizontal axis.

- Data produce a curve that peaks at θ̂_MLE.
- For small n, the curve is wide (high uncertainty).
- As n grows, the peak sharpens, reflecting more precise estimates.

Overlay different ℓ(θ) curves for various n to see how the maximum becomes more pronounced.

### 7. Applications in ML & DS

- **Logistic Regression**: Estimates class–conditional log-odds by maximizing the Bernoulli log-likelihood.
- **Naive Bayes**: Learns discrete feature probabilities via MLE of categorical distributions.
- **Gaussian Mixture Models**: Uses MLE in the EM algorithm to fit component means, variances, and weights.
- **Survival Analysis**: Uses MLE on exponential or Weibull models to estimate hazard rates.
- **Neural Networks**: Many loss functions (cross-entropy, MSE) arise from negative log-likelihoods.

### 8. Practice Problems & Python Exercises

### 8.1 Coin-Toss MLE (Bernoulli)

1. You observe 45 heads in 80 flips. Compute p̂_MLE by hand.
2. Write Python to confirm:

```python
import numpy as np

flips = np.array([1]*45 + [0]*35)
p_hat = flips.mean()
print("MLE p:", p_hat)
```

### 8.2 Exponential MLE

1. Given data [2.3, 0.7, 1.9, 3.5, 2.1], compute λ̂_MLE by hand.
2. Verify in Python:

```python
import numpy as np

data = np.array([2.3, 0.7, 1.9, 3.5, 2.1])
lambda_hat = data.size / data.sum()
print("MLE λ:", lambda_hat)
```

### 8.3 Normal MLE via `scipy.optimize`

Fit μ and σ to data sampled from Normal(5,2):

```python
import numpy as np
from scipy.optimize import minimize

# sample data
np.random.seed(0)
data = np.random.normal(5, 2, size=100)

# negative log-likelihood
def neg_log_lik(params):
    mu, sigma = params
    # add small epsilon for stability
    nll = 0.5*np.log(2*np.pi*sigma**2)*data.size + ((data-mu)**2).sum()/(2*sigma**2)
    return nll

# initial guess
init = [np.mean(data), np.std(data)]
bounds = [(None, None), (1e-6, None)]

result = minimize(neg_log_lik, init, bounds=bounds)
mu_mle, sigma_mle = result.x
print("MLE μ:", mu_mle)
print("MLE σ:", sigma_mle)
```

### 8.4 Likelihood Curve Visualization

1. Simulate 20 flips with unknown p.
2. Evaluate and plot L(p) for p in [0,1]:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

flips = np.random.binomial(1, 0.7, size=20)
x = flips.sum()
ns = flips.size

ps = np.linspace(0, 1, 200)
lik = ps**x * (1-ps)**(ns-x)

plt.plot(ps, lik)
plt.axvline(x/ns, color='red', linestyle='--', label='MLE')
plt.xlabel('p')
plt.ylabel('Likelihood')
plt.title('Likelihood of p given data')
plt.legend()
plt.show()
```

---

## Maximum Likelihood Estimation: Bernoulli Example

### 1. Conceptual Overview

A Bernoulli trial is a single “success/failure” experiment (e.g., coin flip, click/no-click).

We assume each trial has an unknown success probability *p* and we observe *n* independent outcomes.

MLE finds the value of *p* that makes the observed data most likely under the Bernoulli model.

### 2. Likelihood and Log-Likelihood

Given observations

x₁, x₂, …, xₙ ∈ {0,1}, let

- *x* = ∑ xᵢ (total successes)
- *n* = number of trials

The likelihood is:

```
L(p) = ∏_{i=1 to n} p^{x_i} (1−p)^{1−x_i}
     = p^x (1−p)^{n−x}
```

Taking the log gives the log-likelihood:

```
ℓ(p) = log L(p)
     = x·log(p) + (n−x)·log(1−p)
```

### 3. Derivation of the MLE

1. **Write ℓ(p)**
    
    ℓ(p) = x·log(p) + (n−x)·log(1−p)
    
2. **Differentiate ℓ(p)**
    
    ```
    dℓ/dp = x/p − (n−x)/(1−p)
    ```
    
3. **Set derivative to zero**
    
    ```
    x/p − (n−x)/(1−p) = 0
    ```
    
4. **Solve for p̂**
    
    ```
    x·(1−p) = (n−x)·p
    ⇒ x − x p = n p − x p
    ⇒ x = n p
    ⇒ p̂_MLE = x / n
    ```
    
5. **Check second derivative**
    
    The second derivative is negative for p in (0,1), confirming a maximum.
    

### 4. Interpretation and Real-World Use

- **Click-Through Rate (CTR):** Estimate the true click probability from a sample of ad impressions.
- **Conversion Rate:** Infer the purchase probability from trial customer data.
- **Spam Detection:** Estimate the probability an email is spam based on a labeled subset.

In each case, p̂ = successes / trials gives the most likely rate under the Bernoulli assumption.

### 5. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate n Bernoulli trials with true p=0.3
np.random.seed(0)
n = 100
true_p = 0.3
data = np.random.binomial(1, true_p, size=n)

# Compute MLE
x = data.sum()
p_hat = x / n
print(f"Observed successes: {x}/{n}")
print(f"MLE p_hat: {p_hat:.3f}")

# Plot likelihood curve
ps = np.linspace(0, 1, 200)
likelihood = ps**x * (1-ps)**(n-x)

plt.plot(ps, likelihood, label='Likelihood L(p)')
plt.axvline(p_hat, color='red', linestyle='--', label='MLE p_hat')
plt.xlabel('p')
plt.ylabel('L(p)')
plt.title('Bernoulli Likelihood')
plt.legend()
plt.show()
```

### 6. Practice Problems

1. You observe 18 clicks in 60 ad impressions. Find p̂_MLE by hand and confirm in Python.
2. Simulate 1,000 trials with true p=0.7 and plot the histogram of p̂ from 100-trial experiments. Describe the spread.
3. Show that the variance of p̂ is p(1−p)/n and discuss how n affects estimator precision.

---

## Maximum Likelihood Estimation: Gaussian Example

### 1. Conceptual Overview

We assume our data x₁, x₂, …, xₙ come from a Normal (Gaussian) distribution with unknown mean μ and variance σ².

Maximum Likelihood Estimation (MLE) finds the values μ̂ and σ̂² that maximize the likelihood of observing exactly x₁…xₙ under the Gaussian model.

In many ML tasks, we model residual errors or continuous feature distributions as Gaussian. MLE gives us the most plausible parameters for those distributions.

### 2. Likelihood and Log-Likelihood

For independent observations from Normal(μ, σ²), the joint density (likelihood) is:

```
L(μ, σ² | x)
  = ∏_{i=1 to n} [ 1 / sqrt(2πσ²) ]
    * exp( − (x_i − μ)² / (2σ²) )
```

Taking logs turns the product into a sum and simplifies:

```
ℓ(μ, σ²)
  = log L(μ, σ² | x)
  = − (n/2) * log(2πσ²)
    − (1 / (2σ²)) * ∑_{i=1 to n} (x_i − μ)²
```

- The first term penalizes large σ².
- The second term penalizes poor fit of μ to the data.

### 3. Derivation of MLEs

### 3.1 MLE for μ

Differentiate ℓ with respect to μ, set to zero:

```
∂ℓ/∂μ = (1/σ²) * ∑_{i=1 to n} (x_i − μ)  =  0
```

Solve for μ:

```
∑ (x_i − μ) = 0
n * μ = ∑ x_i
μ̂_MLE = (1/n) * ∑_{i=1 to n} x_i
```

The MLE for μ is the **sample mean**.

### 3.2 MLE for σ²

Differentiate ℓ with respect to σ², set to zero:

```
∂ℓ/∂σ²
  = − (n / (2σ²))
    + (1 / (2σ⁴)) * ∑ (x_i − μ)²
  = 0
```

Solve for σ²:

```
n / (2σ²) = (1 / (2σ⁴)) * ∑ (x_i − μ)²
σ̂²_MLE = (1/n) * ∑_{i=1 to n} (x_i − μ̂_MLE)²
```

Note that MLE uses **1/n**, not 1/(n−1). This makes σ̂² slightly biased downward for small samples.

### 4. Visual Intuition

- Plot ℓ(μ, σ²) as a surface in the (μ, σ²) plane. The peak occurs at (μ̂, σ̂²).
- For fixed σ², plot ℓ vs. μ: a concave quadratic peaking at the sample mean.
- As n increases, the peak sharpens, indicating more precise estimates.

### 5. Real-World DS & ML Applications

- **Regression Residuals**
    
    When fitting a linear regression, we often assume residuals are Gaussian. MLE for μ and σ² gives the estimated noise distribution.
    
- **Anomaly Detection**
    
    Model features as Gaussian; points far from μ̂ (in units of σ̂) are flagged as anomalies.
    
- **Gaussian Mixture Models (GMMs)**
    
    In the EM algorithm, each component’s μ and σ² are updated via MLE given soft-assignments of data points.
    
- **Sensor Modeling**
    
    Combine multiple continuous sensor readings; MLE yields best-fit parameters for fusion and filtering.
    

### 6. Practice Problems & Python Exercises

### 6.1 Manual Calculation

Data: [4.2, 5.0, 3.8, 4.5]

1. Compute μ̂_MLE by averaging the four values.
2. Compute σ̂²_MLE by summing squared deviations from μ̂ and dividing by 4.

### 6.2 Python: Direct Formulas

```python
import numpy as np

data = np.array([4.2, 5.0, 3.8, 4.5])
n = data.size

# MLE for mean
mu_hat = data.mean()

# MLE for variance (ddof=0 for 1/n)
sigma2_hat = ((data - mu_hat)**2).sum() / n

print("MLE μ:", mu_hat)
print("MLE σ²:", sigma2_hat)
```

### 6.3 Python: Optimize Log-Likelihood

```python
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
data = np.random.normal(10, 2, size=100)

def neg_log_lik(params):
    mu, sigma = params
    # enforce sigma>0
    if sigma <= 0:
        return np.inf
    n = data.size
    return (n/2)*np.log(2*np.pi*sigma**2) + ((data - mu)**2).sum()/(2*sigma**2)

init = [np.mean(data), np.std(data)]
bounds = [(None, None), (1e-6, None)]
res = minimize(neg_log_lik, init, bounds=bounds)

mu_mle, sigma_mle = res.x
print("Optimized μ:", mu_mle)
print("Optimized σ:", sigma_mle)
```

### 6.4 Likelihood Surface Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, size=50)
mus = np.linspace(-1, 1, 100)
sigmas = np.linspace(0.5, 1.5, 100)

LL = np.zeros((len(mus), len(sigmas)))
for i, m in enumerate(mus):
    for j, s in enumerate(sigmas):
        LL[i, j] = -(
            (data.size/2)*np.log(2*np.pi*s**2)
            + ((data - m)**2).sum()/(2*s**2)
        )

plt.contour(mus, sigmas, LL.T, levels=20)
plt.scatter(np.mean(data), np.std(data), color='red', label='Sample stats')
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('Log-Likelihood Contours')
plt.legend()
plt.show()
```

---

## Maximum Likelihood Estimation for a Gaussian Population

### 1. Problem Setup

You have *n* independent observations

x₁, x₂, …, xₙ ∈ ℝᵈ

assumed to come from a multivariate normal distribution with unknown mean vector μ ∈ ℝᵈ and unknown covariance matrix Σ ∈ ℝᵈˣᵈ.

Goal: find the values μ̂ and Σ̂ that maximize the likelihood of observing your data.

### 2. Likelihood and Log-Likelihood

The density of a single observation xᵢ under N(μ, Σ) is:

```
f(x_i; μ, Σ)
  = 1 / [ (2π)^(d/2) * |Σ|^(1/2) ]
    * exp( -½ (x_i − μ)ᵀ Σ⁻¹ (x_i − μ) )
```

The joint likelihood for all *n* observations:

```
L(μ, Σ)
  = ∏_{i=1 to n} f(x_i; μ, Σ)
```

Taking the log:

```
ℓ(μ, Σ)
  = − (n d / 2) log(2π)
    − (n / 2) log |Σ|
    − ½ ∑_{i=1 to n} (x_i − μ)ᵀ Σ⁻¹ (x_i − μ)
```

### 3. Derivation of MLEs

### 3.1 MLE for μ

1. Differentiate ℓ with respect to μ and set to zero:
    
    ```
    ∂ℓ/∂μ = Σ⁻¹ ∑_{i=1 to n} (x_i − μ) = 0
    ```
    
2. Solve:
    
    ```
    ∑_{i=1 to n} (x_i − μ) = 0
    n μ = ∑_{i=1 to n} x_i
    μ̂_MLE = (1/n) * ∑_{i=1 to n} x_i
    ```
    

The MLE of the mean vector is the sample centroid.

### 3.2 MLE for Σ

1. Plug μ̂ into ℓ and differentiate with respect to Σ⁻¹ (the natural parameter).
2. Set derivative to zero, leading to:
    
    ```
    Σ̂_MLE = (1/n) * ∑_{i=1 to n} (x_i − μ̂) (x_i − μ̂)ᵀ
    ```
    

This is the sample covariance matrix with **1/n** normalization (biased for small *n* but maximum-likelihood).

### 4. Geometric Interpretation

- μ̂ is the **centroid** of your *n* data points in ℝᵈ.
- Σ̂ captures the **scatter**: it’s the average outer product of deviations from the centroid.
- Directions of largest variance correspond to eigenvectors of Σ̂ with largest eigenvalues.

### 5. Python Implementation

```python
import numpy as np

# Simulated data: n samples in d dimensions
n, d = 100, 3
X = np.random.randn(n, d) + np.array([1, 2, 3])  # true mean = [1,2,3]

# MLE for mean
mu_hat = X.mean(axis=0)

# Centered data
X_centered = X - mu_hat

# MLE for covariance (1/n)
Sigma_hat = (X_centered.T @ X_centered) / n

print("MLE mean vector:", mu_hat)
print("MLE covariance matrix:\n", Sigma_hat)
```

- `axis=0` ensures we average over rows (samples).
- The outer-product average yields the covariance matrix.

### 6. Practice Problems & Exercises

1. **Manual Computation**
    
    Data in 2D:
    
    x₁=(1,2), x₂=(3,0), x₃=(2,4)
    
    - Compute μ̂ by hand.
    - Compute Σ̂ using 1/3 normalization.
2. **Python Exercise**
    
    Generate 500 samples from N([0,0], [[2,1],[1,2]]).
    
    - Compute μ̂ and Σ̂ via the code above.
    - Compare eigenvalues of Σ̂ with the true covariance.
3. **Visualization**
    - Plot the 2D samples and overlay the ellipse defined by ±2√λ contours of Σ̂.
    - Verify the ellipse encloses ~95% of the points.

### 7. Applications in ML & DS

- **Gaussian Mixture Models (GMMs):** MLE updates in the EM algorithm.
- **Anomaly Detection:** Mahalanobis distance based on μ̂ and Σ̂.
- **Linear Discriminant Analysis (LDA):** Within-class covariance estimation.
- **Principal Component Analysis (PCA):** Eigen-decomposition of Σ̂ for dimensionality reduction.

---

## Maximum Likelihood Estimation: Linear Regression

### 1. Problem Setup

You have *n* paired observations {(x₁, y₁), …, (xₙ, yₙ)}, where each xᵢ is a vector of *p* predictors and yᵢ is a continuous response.

You posit a linear model with Gaussian noise:

yᵢ = xᵢᵀβ + εᵢ,

where

- β ∈ ℝᵖ is an unknown coefficient vector
- εᵢ are independent noise terms, each ~ Normal(0, σ²)

Goal: Find the values β̂ and σ̂² that maximize the likelihood of observing your data under this model.

### 2. Likelihood and Log-Likelihood

### 2.1 Likelihood

Because each εᵢ ~ Normal(0, σ²),

yᵢ | xᵢ, β, σ² ~ Normal(xᵢᵀβ, σ²).

The joint likelihood over *n* samples is:

```
L(β, σ² | X, y)
  = ∏_{i=1 to n}
      [ 1 / sqrt(2πσ²) ]
      * exp( −(y_i − x_iᵀβ)² / (2σ²) )
```

Here

- X is the n×p design matrix whose i-th row is xᵢᵀ
- y is the n-vector of responses

### 2.2 Log-Likelihood

Taking the log turns the product into a sum:

```
ℓ(β, σ²)
  = log L
  = − (n/2) * log(2πσ²)
    − 1/(2σ²) * ∑_{i=1 to n} (y_i − x_iᵀβ)²
```

Breakdown:

- The first term penalizes large σ²
- The second term measures squared errors between predictions and observations

### 3. Derivation of the MLE

### 3.1 MLE for β

1. Treat σ² as fixed. Differentiate ℓ with respect to β:
    
    ```
    ∂ℓ/∂β = (1/σ²) * Xᵀ (y − Xβ)
    ```
    
2. Set the derivative to zero and solve for β:
    
    ```
    Xᵀ (y − Xβ) = 0
    ⇒ Xᵀ y = Xᵀ X β
    ⇒ β̂_MLE = (Xᵀ X)⁻¹ Xᵀ y
    ```
    

This is the familiar **ordinary least squares** solution.

### 3.2 MLE for σ²

1. Plug β̂ into the log-likelihood.
2. Differentiate ℓ with respect to σ² and set to zero:
    
    ```
    ∂ℓ/∂σ²
      = − (n/(2σ²)) + (1/(2σ⁴)) * ∑ (y_i − x_iᵀβ̂)²
      = 0
    ```
    
3. Solve:
    
    ```
    σ̂²_MLE
      = (1/n) * ∑_{i=1 to n} (y_i − x_iᵀβ̂)²
    ```
    

Note that this uses **1/n**; the unbiased estimator uses **1/(n−p)** instead.

### 4. Geometric Intuition

- Xβ̂ is the projection of y onto the column space of X.
- Residuals r = y − Xβ̂ are orthogonal to every column of X: Xᵀ r = 0.
- Minimizing squared errors ∑rᵢ² is equivalent to maximizing the Gaussian likelihood.

### 5. Real-World ML & DS Applications

- Predicting house prices from features (area, rooms, age).
- Modeling customer spend as a linear function of demographics.
- Trend estimation in time series (with basis expansion in X).
- Feature interpretation: each β̂ⱼ quantifies the effect of predictor j on the response.

### 6. Practice Problems & Python Exercises

### 6.1 Manual Computation

Data:

x₁ = [1, 2], y₁ = 4

x₂ = [1, 5], y₂ = 7

x₃ = [1, 3], y₃ = 5

1. Form X and y.
2. Compute XᵀX, Xᵀy.
3. Solve β̂ = (XᵀX)⁻¹Xᵀy by hand.

### 6.2 Python: Closed-Form Solution

```python
import numpy as np

# sample data
X = np.array([[1,2],
              [1,5],
              [1,3]], dtype=float)
y = np.array([4,7,5], dtype=float)

# MLE for beta
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
beta_hat = XtX_inv @ X.T @ y

# Predictions and residual variance
y_pred = X @ beta_hat
sigma2_hat = ((y - y_pred)**2).sum() / X.shape[0]

print("Beta_hat:", beta_hat)
print("Sigma2_hat:", sigma2_hat)
```

### 6.3 Python: Validation with scikit-learn

```python
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression(fit_intercept=False)  # intercept included in X
model.fit(X, y)
print("sklearn beta:", model.coef_)

# Compare residual variance
resid_var = ((y - model.predict(X))**2).sum() / len(y)
print("Residual variance:", resid_var)
```

### 6.4 Simulation: Sampling Variability

```python
import numpy as np
import matplotlib.pyplot as plt

true_beta = np.array([2.0, 0.5])
n, p = 100, len(true_beta)
trials = 500

beta_estimates = np.zeros((trials, p))
for t in range(trials):
    X = np.hstack((np.ones((n,1)), np.random.randn(n,1)))
    y = X @ true_beta + np.random.randn(n)*2.0
    beta_estimates[t] = np.linalg.inv(X.T@X) @ X.T @ y

plt.hist(beta_estimates[:,1], bins=20, edgecolor='k')
plt.title('Sampling distribution of β̂₂')
plt.xlabel('Estimate of β₂')
plt.show()
```

- Observe the spread of β̂ estimates around the true value.

---

## Regularization

### 1. Prerequisite Reminder

You already know how to fit an ordinary least squares (OLS) linear regression by maximizing the Gaussian log-likelihood, which yields the closed-form solution

```
β̂_OLS = (Xᵀ X)⁻¹ Xᵀ y
```

When you have many features or multicollinearity, OLS can overfit—coefficients explode to fit noise. Regularization tames that by adding a penalty term to the loss.

### 2. What Is Regularization?

Regularization is the technique of adding a penalty on coefficient size to your loss function. This penalty

- Reduces variance (prevents overfitting)
- Encourages smaller (and in some cases sparser) coefficients
- Trades a bit of bias for a large drop in variance, improving generalization

Two common forms:

- **Ridge (L2) Regularization**
- **Lasso (L1) Regularization**

### 3. Formulas and Breakdown

### 3.1 Ridge (L2) Regression

Minimize sum of squared errors plus λ times sum of squared coefficients:

```
β̂_ridge = argmin_β [
    ∑_{i=1 to n} (y_i − x_iᵀ β)²
  + λ * ∑_{j=1 to p} β_j²
]
```

- `∑(y_i − x_iᵀ β)²` is the usual residual sum of squares (RSS).
- `λ` ≥ 0 is the regularization strength (hyperparameter).
- `∑ β_j²` penalizes large β values equally in all directions.

Closed-form solution:

```
β̂_ridge = (Xᵀ X + λ I)⁻¹ Xᵀ y
```

- `I` is the p×p identity matrix.
- Adding `λ I` to `XᵀX` shrinks eigenvalues, making the inverse more stable.

### 3.2 Lasso (L1) Regression

Minimize RSS plus λ times sum of absolute coefficients:

```
β̂_lasso = argmin_β [
    ∑_{i=1 to n} (y_i − x_iᵀ β)²
  + λ * ∑_{j=1 to p} |β_j|
]
```

- `∑ |β_j|` encourages sparsity: some β_j become exactly zero.
- No closed-form; solved via coordinate descent or convex solvers.

### 4. Geometric & Visual Intuition

- In coefficient space, contours of equal RSS are ellipses centered at β̂_OLS.
- Ridge adds an L2 “ball” constraint (circle) around the origin; the solution is where an ellipse first touches the ball.
- Lasso adds an L1 “diamond” constraint; its corners lie on axes, so touching often happens at a corner (zero coefficient), creating sparsity.

### 5. Real-World ML & DS Applications

- **High-dimensional data** (p ≫ n): Ridge stabilizes estimates when XᵀX is singular.
- **Feature selection**: Lasso automatically drops irrelevant features by zeroing their coefficients.
- **Multicollinearity**: Ridge shares weight among correlated features instead of blowing one coefficient up.
- **Model interpretability**: Lasso yields simpler, interpretable models with fewer features.

### 6. Practice Problems & Python Exercises

### 6.1 Manual Ridge Calculation

Data:

```
X = [[1, 2],
     [1, 4],
     [1, 6]]
y = [3, 5, 7]
```

1. Form XᵀX and Xᵀy.
2. With λ = 1, compute (XᵀX + λI) and its inverse.
3. Compute β̂_ridge by matrix multiplication.

### 6.2 Python: Ridge vs. OLS

```python
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

# Simulate data
np.random.seed(0)
n, p = 100, 10
X = np.random.randn(n, p)
true_beta = np.array([2, -1, 0.5] + [0]*(p-3))
y = X @ true_beta + np.random.randn(n)*0.5

# Fit OLS
ols = LinearRegression(fit_intercept=False).fit(X, y)

# Fit Ridge
ridge = Ridge(alpha=1.0, fit_intercept=False).fit(X, y)

print("OLS coefficients:", np.round(ols.coef_, 3))
print("Ridge coefficients:", np.round(ridge.coef_, 3))
```

- Observe how Ridge shrinks coefficients toward zero relative to OLS.

### 6.3 Python: Lasso Sparsity

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, fit_intercept=False, max_iter=10000).fit(X, y)
print("Lasso coefficients:", np.round(lasso.coef_, 3))
print("Number of non-zero:", np.sum(lasso.coef_ != 0))
```

- Notice how many coefficients Lasso zeros out.

### 6.4 Cross-Validation for λ Selection

```python
from sklearn.linear_model import RidgeCV, LassoCV

# RidgeCV tests multiple alphas and selects the best by cross-validation
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10], fit_intercept=False).fit(X, y)
print("Best Ridge α:", ridge_cv.alpha_)

# LassoCV similarly
lasso_cv = LassoCV(alphas=[0.01,0.1,1,10], fit_intercept=False).fit(X, y)
print("Best Lasso α:", lasso_cv.alpha_)
```

---

## Bayesian Statistics: Frequentist vs Bayesian

### 1. Introduction

Statistics seeks to learn about unknown quantities (parameters) using observed data. Two major paradigms—**Frequentist**and **Bayesian**—differ in how they define probability, treat parameters, and draw inferences. Understanding their contrasts equips you to choose the right approach for modelling, uncertainty quantification, and decision-making in ML/DS.

### 2. Probability Interpretations

- Frequentist probability
    
    Probability of an event is its long-run relative frequency over repeated trials. Parameters (like a coin’s bias) are fixed but unknown; data vary over hypothetical repeats.
    
- Bayesian probability
    
    Probability quantifies **degree of belief** in an event or hypothesis, given current knowledge. Both data and parameters are treated as random variables; beliefs update with new evidence.
    

### 3. Frequentist Approach

- Treats parameter θ as fixed.
- Uses estimators (e.g., MLE) to find a single “best” θ̂ from data X.
- Constructs **confidence intervals**: ranges that would contain the true θ in a specified fraction of hypothetical repeated experiments.

Key steps:

1. Specify likelihood L(θ|X).
2. Maximize L to get θ̂.
3. Derive sampling distribution of θ̂ to build intervals or tests.

### 4. Bayesian Approach

- Treats parameter θ as a random variable with a **prior** distribution π(θ).
- Uses observed data X to compute a **posterior** distribution π(θ|X), summarizing updated beliefs.
- Reports **credible intervals**: intervals containing θ with a given posterior probability.

Key steps:

1. Specify prior π(θ).
2. Write likelihood L(X|θ).
3. Apply Bayes’ theorem to get posterior π(θ|X).
4. Derive point estimates (mean, median, MAP) or full predictive distributions.

### 5. Bayes’ Theorem: Formula & Breakdown

```
posterior  π(θ | X)  =  [ likelihood  L(X | θ)  *  prior  π(θ) ]
                       ------------------------------------------------
                                   evidence  π(X)
```

- **prior** π(θ): your belief about θ before seeing data.
- **likelihood** L(X|θ): probability of observed data given θ.
- **evidence** π(X) = ∫ L(X|θ) π(θ) dθ: normalizing constant.
- **posterior** π(θ|X): updated belief about θ after seeing data.

Step by step:

1. Compute the product L(X|θ)·π(θ).
2. Integrate over θ to find π(X).
3. Divide to normalize, yielding π(θ|X).

### 6. Geometric & Visual Intuition

Imagine plotting curves over θ:

- **Prior**: a curve reflecting belief before data.
- **Likelihood**: spikes where data are most probable under θ.
- **Posterior**: the product of prior and likelihood, then rescaled.

Overlaying these shows how data shift and sharpen your belief from prior to posterior.

### 7. Real-World ML & DS Applications

- **Bayesian Linear Regression**
    
    Place a Gaussian prior on weights β; posterior is Gaussian—yields uncertainty on predictions.
    
- **Naive Bayes Classifier**
    
    Treat class–conditional feature probabilities as random; use conjugate priors (e.g., Beta, Dirichlet) for closed-form updates.
    
- **Bayesian Neural Networks**
    
    Place priors on network weights and use variational inference or MCMC to approximate posteriors, capturing model uncertainty.
    
- **A/B Testing with Beta-Binomial**
    
    Model click rates with a Beta prior and Binomial likelihood; posterior Beta gives credible intervals for conversion rates per variant.
    

### 8. Practice Problems & Python Exercises

### 8.1 Coin-Flip Posterior with Beta Prior

Dataset: Observe x = 7 heads in n = 10 flips. Prior: Beta(α=2, β=2).

1. Derive the posterior distribution:
    
    ```
    posterior  ∝  p^x (1−p)^{n−x}  *  p^{α−1} (1−p)^{β−1}
              = p^{x+α−1} (1−p)^{n−x+β−1}
    ```
    
2. Identify parameters:
    
    ```
    posterior = Beta( α_post = x+α,  β_post = n−x+β )
    ```
    
3. Compute posterior mean and a 95% credible interval.

### 8.2 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Observed data
x, n = 7, 10
alpha_prior, beta_prior = 2, 2

# Posterior parameters
alpha_post = x + alpha_prior
beta_post  = n - x + beta_prior

# Posterior mean and 95% CI
mean_post = alpha_post / (alpha_post + beta_post)
ci_low, ci_high = beta.ppf([0.025, 0.975], alpha_post, beta_post)

print(f"Posterior mean: {mean_post:.3f}")
print(f"95% credible interval: [{ci_low:.3f}, {ci_high:.3f}]")

# Plot prior vs posterior
ps = np.linspace(0,1,200)
plt.plot(ps, beta.pdf(ps, alpha_prior, beta_prior), label='Prior')
plt.plot(ps, beta.pdf(ps, alpha_post, beta_post), label='Posterior')
plt.xlabel('p')
plt.ylabel('Density')
plt.title('Beta Prior and Posterior')
plt.legend()
plt.show()
```

### 8.3 Frequentist Comparison

Compute the MLE and a 95% confidence interval for p using the Normal approximation:

```python
p_hat = x / n
se = np.sqrt(p_hat*(1-p_hat)/n)
ci_norm = (p_hat - 1.96*se, p_hat + 1.96*se)
print(f"MLE p_hat: {p_hat:.3f}")
print(f"95% CI (approx): [{ci_norm[0]:.3f}, {ci_norm[1]:.3f}]")
```

Compare the Bayesian credible interval to the frequentist confidence interval.

---

## Bayesian Statistics: Maximum A Posteriori (MAP) Estimation

### 1. Conceptual Overview

Maximum A Posteriori (MAP) estimation picks the single parameter value that is most probable under your posterior distribution.

In Bayesian inference, you start with a prior belief about the parameter and update it with data via the likelihood. MAP then finds the mode of the resulting posterior.

Think of it as a compromise between pure Bayesian integration and simple point estimates: you leverage prior information, yet report just one “best” guess.

### 2. Prior, Likelihood, Posterior Recap

You’ve already seen Bayes’ theorem:

```
posterior π(θ | X) = [ L(X | θ) * π(θ) ] / π(X)
```

- π(θ) is the prior belief over θ
- L(X | θ) is the likelihood of data X under θ
- π(X) is a normalizing constant

The posterior π(θ | X) combines prior and likelihood into updated beliefs.

### 3. MAP Definition and Formula

MAP chooses the θ that maximizes the posterior:

```
θ̂_MAP = argmax_θ π(θ | X)
       = argmax_θ [ L(X | θ) * π(θ) ]
```

Since π(X) does not depend on θ, you can ignore it in the maximization:

```
θ̂_MAP = argmax_θ [ log L(X | θ) + log π(θ) ]
```

Breakdown:

- You add the log-likelihood and the log-prior
- The prior acts as a regularizer, pulling the estimate toward regions of high prior density
- When the prior is uniform, MAP reduces to Maximum Likelihood Estimation (MLE)

### 4. Visual Intuition

- Plot likelihood L(X|θ) as a curve over θ
- Plot prior π(θ) on the same axis
- The posterior is their product (or the sum of logs)
- The peak of the posterior curve marks θ̂_MAP

As you increase data, the likelihood sharpens and can dominate the prior, making MAP close to MLE.

### 5. Real-World ML & DS Applications

- **Ridge Regression**
    
    MAP with a Gaussian prior on coefficients (zero-mean, variance τ²) yields the ridge penalty λ ∥β∥₂².
    
- **Logistic Regression with Prior**
    
    Placing a Gaussian prior on weights transforms cross-entropy training into MAP with L2 regularization.
    
- **Sparse Models (Lasso)**
    
    A Laplace (double-exponential) prior on coefficients yields an L1 penalty in the MAP solution.
    
- **Naive Bayes with Dirichlet Prior**
    
    MAP estimates of class–conditional probabilities add “pseudo-counts” to avoid zeros.
    

### 6. Practice Problems & Python Exercises

### 6.1 Beta-Bernoulli MAP

Data: x = 7 heads in n = 10 flips. Prior: Beta(α = 2, β = 2).

1. Posterior is Beta(α + x, β + n − x) = Beta(9, 5).
2. MAP estimate is the mode of Beta(a, b): (a − 1)/(a + b − 2).
3. Compute θ̂_MAP by hand: (9 − 1)/(9 + 5 − 2) = 8/12 ≈ 0.667.

Python:

```python
from scipy.stats import beta

alpha_prior, beta_prior = 2, 2
x, n = 7, 10
a_post = alpha_prior + x
b_post = beta_prior + n - x

map_estimate = (a_post - 1) / (a_post + b_post - 2)
print("MAP estimate p̂:", map_estimate)
```

### 6.2 Gaussian Mean with Conjugate Prior

Assume observations X ~ Normal(μ, σ²) with known σ². Prior on μ: Normal(μ₀, τ²).

1. Likelihood log L ∝ −(n/2σ²)(x̄ − μ)²
2. Prior log π ∝ −(1/2τ²)(μ − μ₀)²
3. MAP solves:

```
μ̂_MAP = ( (n/σ²) x̄ + (1/τ²) μ₀ ) / ( n/σ² + 1/τ² )
```

Python:

```python
import numpy as np

# data
data = np.array([4.2, 5.0, 3.8, 4.5])
n = data.size
sigma2 = 1.0  # known
x_bar = data.mean()

# prior
mu0, tau2 = 0.0, 2.0

# MAP formula
mu_map = ( (n/sigma2)*x_bar + (1/tau2)*mu0 ) / ( n/sigma2 + 1/tau2 )
print("μ_MAP:", mu_map)
```

---

## Bayesian Statistics: Updating Priors

### 1. Conceptual Overview

Updating priors is the heart of Bayesian inference: you start with a **prior** belief about a parameter, observe data, and then form a **posterior** belief that blends prior information with evidence from the data.

Every time new data arrive, you repeat this process—using your previous posterior as the new prior—so your beliefs evolve coherently as evidence accumulates.

### 2. Bayes’ Theorem for Sequential Updating

At its core, one update step applies Bayes’ theorem:

```
posterior(θ | data)  ∝  likelihood(data | θ)  *  prior(θ)
```

Written as a normalization:

```
π(θ | X) = [ f(X | θ) ⋅ π(θ) ]  /  ∫ f(X | θ) ⋅ π(θ) dθ
```

- π(θ) is the prior density before seeing data X
- f(X | θ) is the likelihood of the data given θ
- π(θ | X) is the posterior density after observing X

For **sequential** updates, if you split data into two batches X₁ then X₂:

1. Compute posterior₁ from prior and X₁:
    
    ```
    π₁(θ) ∝ f(X₁ | θ) ⋅ π₀(θ)
    ```
    
2. Use π₁ as new prior and update with X₂:
    
    ```
    π₂(θ) ∝ f(X₂ | θ) ⋅ π₁(θ)
    ```
    

This yields the same result as a single update on combined data X₁ ∪ X₂.

### 3. Conjugate Priors: Closed-Form Updates

Conjugate priors make updating especially simple: the posterior belongs to the same family as the prior. Below are three canonical examples.

### 3.1 Beta–Bernoulli

- **Model**: observations xᵢ ∈ {0,1}, each ~ Bernoulli(p)
- **Prior**: p ~ Beta(α, β)

After observing *x* successes in *n* trials:

```
posterior  p | data  ~  Beta( α + x,  β + n − x )
```

Breakdown:

1. α + x adds the number of successes to the prior “success count”
2. β + (n − x) adds failures to the prior “failure count”

### 3.2 Dirichlet–Multinomial

- **Model**: *n* trials over *k* categories; counts c₁,…,cₖ ~ Multinomial(n, p)
- **Prior**: p = (p₁,…,pₖ) ~ Dirichlet(α₁,…,αₖ)

After observing counts c₁…cₖ:

```
posterior  p | data  ~  Dirichlet( α₁ + c₁, …, αₖ + cₖ )
```

### 3.3 Normal–Normal (Known Variance)

- **Model**: xᵢ ~ Normal(μ, σ²) with known σ²
- **Prior**: μ ~ Normal(μ₀, τ²)

After *n* observations with sample mean x̄:

```
posterior  μ | data  ~  Normal( μₙ,  τₙ² )

where
  τₙ² = ( 1/τ² + n/σ² )⁻¹
  μₙ   =  τₙ² ⋅ ( μ₀/τ² + n x̄/σ² )
```

### 4. Step-by-Step Update Example: Beta–Bernoulli

Let’s work through a concrete Beta–Bernoulli update.

1. **Prior**: α = 2, β = 2
2. **Data**: in 10 flips, x = 7 heads, 3 tails

Update:

```
α_post = α + x        = 2 + 7  = 9
β_post = β + (n - x)  = 2 + 3  = 5

Posterior  p | data  ~  Beta(9, 5)
```

- **Prior mean** = 2 / (2+2) = 0.5
- **Posterior mean** = 9 / (9+5) ≈ 0.643

Your belief shifts toward more probability of success after seeing mostly heads.

### 5. Python Exercises

### 5.1 Sequential Updating with Beta–Bernoulli

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Initial prior
alpha, beta_param = 2, 2

# Batch 1 data: 4 heads in 6 flips
x1, n1 = 4, 6
alpha1 = alpha + x1
beta1  = beta_param + (n1 - x1)

# Batch 2 data: 3 heads in 4 flips
x2, n2 = 3, 4
alpha2 = alpha1 + x2
beta2  = beta1 + (n2 - x2)

# Plot prior, intermediate posterior, final posterior
ps = np.linspace(0,1,200)
plt.plot(ps, beta.pdf(ps, alpha, beta_param), label='Prior')
plt.plot(ps, beta.pdf(ps, alpha1, beta1), label='Posterior₁')
plt.plot(ps, beta.pdf(ps, alpha2, beta2), label='Posterior₂')
plt.legend(); plt.xlabel('p'); plt.ylabel('Density')
plt.show()
```

- Verify that direct updating with all 7 heads of 10 flips gives the same final Beta(9,5).

### 5.2 Dirichlet–Multinomial Update

```python
import numpy as np
from scipy.stats import dirichlet

# Prior pseudo-counts for 3 categories
alpha = np.array([1, 1, 1])

# Observed counts
counts = np.array([30, 50, 20])

# Posterior parameters
alpha_post = alpha + counts

print("Posterior Dirichlet params:", alpha_post)
# Draw samples from posterior
samples = dirichlet.rvs(alpha_post, size=1000)
```

- Plot histograms of sampled p₁, p₂, p₃ to see updated category probabilities.

### 5.3 Normal–Normal Update

```python
import numpy as np

# Known variance
sigma2 = 4.0

# Prior on mu
mu0, tau2 = 0.0, 1.0

# Data
data = np.array([5.1, 4.9, 5.3, 5.0])
n = data.size
x_bar = data.mean()

# Posterior parameters
tau_n2 = 1 / (1/tau2 + n/sigma2)
mu_n   = tau_n2 * (mu0/tau2 + n*x_bar/sigma2)

print("Posterior mean μₙ:", mu_n)
print("Posterior variance τₙ²:", tau_n2)
```

- Compare posterior mean to sample mean x̄, noting the “shrinkage” toward μ₀.

### 6. Practice Problems

1. **Beta–Bernoulli**:
    
    Prior Beta(5, 5). First batch: 8 heads/12 flips. Second batch: 2 heads/8 flips.
    
    a) Compute posterior after each batch by hand.
    
    b) Plot the evolving densities.
    
2. **Dirichlet–Multinomial**:
    
    Prior Dirichlet(2,2,2). Observed counts [10, 0, 5].
    
    a) Compute the posterior vector of α’s.
    
    b) Find the posterior mean for each category.
    
3. **Normal–Normal**:
    
    Known σ² = 9. Prior μ₀ = 10, τ² = 4. Data: [12, 8, 11, 10, 9].
    
    a) Derive posterior μₙ and τₙ².
    
    b) Compare posterior mean to the sample mean.
    

---

## A Full Worked Bayesian Example: Estimating an Ad Click-Through Rate

### 1. Problem Setup

We want to estimate the true click-through rate *p* of a new online advertisement.

- Before launching, our **prior** belief about *p* is moderately uninformative: we think clicks are likely between 0.1 and 0.4, but we’re not certain.
- We model each ad impression as a Bernoulli trial: click = 1, no-click = 0.
- We then serve 100 impressions and observe 15 clicks.

Goal:

1. Update our prior to a **posterior** distribution for *p*.
2. Compute point estimates (mean, MAP).
3. Find a 95% **credible interval**.
4. Derive the **posterior predictive** distribution for clicks in the next 50 impressions.

### 2. Choosing a Conjugate Prior

For Bernoulli data, the **Beta** family is conjugate:

```
Prior:   p ~ Beta(α₀, β₀)
Likelihood:  L(data|p) = pˣ (1−p)ⁿ⁻ˣ
Posterior: p | data ~ Beta(α₀ + x, β₀ + n − x)
```

We set α₀=2, β₀=2 so that prior mean = 2/(2+2)=0.5 but with modest weight (four “imaginary” flips).

### 3. Posterior Update

We collect n=100 trials with x=15 clicks.

```
α_post = α₀ + x   = 2 + 15  = 17
β_post = β₀ + n−x = 2 + 100−15 = 87

Posterior: p | data ~ Beta(17, 87)
```

### 4. Point Estimates and Credible Interval

### 4.1 Posterior Mean

```
E[p | data] = α_post / (α_post + β_post)
            = 17 / (17 + 87)
            ≈ 0.163
```

### 4.2 MAP (Posterior Mode)

For a Beta(a,b) with a,b > 1:

```
p̂_MAP = (a − 1)/(a + b − 2)
       = (17 − 1)/(17 + 87 − 2)
       = 16 / 102
       ≈ 0.157
```

### 4.3 95% Credible Interval

Use the Beta percent-point function:

```python
from scipy.stats import beta
ci_low, ci_high = beta.ppf([0.025, 0.975], 17, 87)
```

Numerically,

```
95% CI ≈ [0.100, 0.232]
```

We are 95% “certain” p lies in this range, given our model and prior.

### 5. Posterior and Prior Visualization

Imagine plotting density curves over p∈[0,1]:

- Prior Beta(2,2): mild peak at p=0.5, broad spread.
- Posterior Beta(17,87): peak around 0.16, much narrower.

Visually, the likelihood from 15/100 pulls and sharpens the prior toward lower p.

### 6. Posterior Predictive Distribution

We want the distribution of future clicks *k* in *m* = 50 impressions:

### 6.1 Beta-Binomial Formula

```
P(k | data)
  =  C(m, k)  *  B(α_post + k,  β_post + m − k)
               / B(α_post, β_post)
```

- C(m,k) is “m choose k.”
- B(·,·) is the Beta function.

### 6.2 Predictive Mean and Variance

```
E[k] = m * α_post / (α_post + β_post) ≈ 50 * 17/104 ≈ 8.17
Var[k] = m * (α_post β_post / (α_post + β_post)²) * (α_post + β_post + m) / (α_post + β_post + 1)
```

So we expect about 8 clicks next 50 impressions, with calculable spread.

### 7. Full Python Walkthrough

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, betabinom

# 1. Data and prior
alpha0, beta0 = 2, 2
x, n = 15, 100

# 2. Posterior parameters
alpha_post = alpha0 + x
beta_post  = beta0 + (n - x)

# 3. Estimates and CI
post_mean = alpha_post / (alpha_post + beta_post)
map_est    = (alpha_post - 1) / (alpha_post + beta_post - 2)
ci_low, ci_high = beta.ppf([0.025, 0.975], alpha_post, beta_post)

print(f"Posterior mean: {post_mean:.3f}")
print(f"MAP:            {map_est:.3f}")
print(f"95% CI:         [{ci_low:.3f}, {ci_high:.3f}]")

# 4. Plot prior vs. posterior
ps = np.linspace(0,1,200)
plt.plot(ps, beta.pdf(ps, alpha0, beta0), label='Prior Beta(2,2)')
plt.plot(ps, beta.pdf(ps, alpha_post, beta_post),
         label=f'Posterior Beta({alpha_post},{beta_post})')
plt.axvline(post_mean, color='k', linestyle='--', label='Posterior mean')
plt.xlabel('p'); plt.ylabel('Density'); plt.legend()
plt.title('Prior vs Posterior')
plt.show()

# 5. Posterior predictive for m=50
m = 50
k = np.arange(0, m+1)
pred_pmf = betabinom.pmf(k, m, alpha_post, beta_post)

plt.bar(k, pred_pmf, color='skyblue', edgecolor='k')
plt.xlabel('Future clicks k')
plt.ylabel('P(k | data)')
plt.title(f'Posterior Predictive (m={m})')
plt.show()

print(f"Predictive mean ≈ { (m*post_mean):.2f} clicks")
```

### 8. Interpretation & Decision Making

- Our **best guess** for the click rate is ~16.3%.
- We’re 95% confident it lies between 10% and 23%.
- In the next 50 impressions, we expect ~8 clicks.
- If a business needs at least 10 clicks in 50, they can compute P(k≥10) under the predictive PMF to decide if the ad is acceptable.

### 9. Extensions & Next Steps

- Compare different priors (uniform Beta(1,1) vs informative Beta(5,5)) and see how much data are needed to overcome them.
- Incorporate sequential updates: process data in batches and confirm the final posterior is the same.
- Move to hierarchical Beta–Binomial if you run multiple ads and want to share strength across them.
- Explore non-conjugate models (e.g., logistic regression with Gaussian priors) via MCMC or variational inference.

---

## Relationship Between MLP, MLE, and Regularization

### 1. Viewing an MLP Through the Lens of Likelihood

An MLP (Multi-Layer Perceptron) is a parametric model that defines a conditional distribution over outputs given inputs,

 P(y | x; W)

where **W** denotes all network weights and biases.

When you train an MLP for classification or regression, you almost always choose a loss function that equals the **negative log-likelihood** of your data under that model.

### 1.1 Negative Log-Likelihood (NLL) Loss

For a dataset {(xᵢ, yᵢ)}ₙ, the NLL is

```
NLL(W) = − ∑_{i=1}^n log P(y_i | x_i; W)
```

- In **classification**, P(yᵢ|xᵢ;W) comes from a softmax output; minimizing NLL is equivalent to minimizing cross-entropy loss.
- In **regression** with Gaussian noise, P(yᵢ|xᵢ;W) ∼ Normal(f_W(xᵢ), σ²), so NLL becomes mean squared error (up to constants).

Minimizing NLL through gradient descent is therefore **Maximum Likelihood Estimation (MLE)** for an MLP’s parameters.

### 2. MLE ⇒ Training Objective for an MLP

1. **Model specification**
    
    Define your network f_W(x) and interpret its final layer as logits → probabilities or predictions.
    
2. **Write down the likelihood**
    
    ```
    L(W) = ∏_{i=1}^n P(y_i | x_i; W)
    ```
    
3. **Convert to loss**
    
    ```
    Loss(W) = − log L(W) = − ∑_{i=1}^n log P(y_i | x_i; W)
    ```
    
4. **Optimize with backprop**
    
    Use automatic differentiation and gradient-based optimizers (SGD, Adam) to find
    
    ```
    Ŵ_MLE = argmin_W Loss(W)
    ```
    

That procedure is literally **MLE** for a neural network.

### 3. Introducing Regularization as a Prior → MAP

Pure MLE can **overfit** when your MLP has millions of parameters. Regularization prevents weight explosion by adding a penalty term:

```
Regularized Loss(W)
  = − ∑ log P(y_i | x_i; W)
    + λ · R(W)
```

- **R(W)** is a penalty (e.g., ∥W∥₂² for weight decay).
- **λ** is the regularization strength.

### 3.1 Bayesian Interpretation: MAP Estimation

If you place a Gaussian prior on weights,

```
π(W) ∝ exp( − (λ/2) · ∥W∥₂² )
```

then **Maximum A Posteriori (MAP)** estimation solves

```
Ŵ_MAP = argmax_W [ log L(W) + log π(W) ]
      = argmin_W [ − log L(W) + (λ/2) ∥W∥₂² ]
```

Thus, **weight decay** in your MLP is exactly a Gaussian prior → MAP estimation.

### 4. Common Regularizers and Their Bayesian Counterparts

- **L2 (Weight Decay)**
    
    R(W) = ∑ wⱼ²
    
    ↔ Gaussian prior on each wⱼ
    
- **L1 (Lasso-style)**
    
    R(W) = ∑ |wⱼ|
    
    ↔ Laplace (double-exponential) prior on wⱼ → sparsity
    
- **Dropout**
    
    Randomly zeroing neurons during training
    
    ↔ Approximate Bayesian inference via variational methods
    

### 5. Practical Python Snippet

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss()
# weight_decay translates to L2 penalty (Gaussian prior)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for x_batch, y_batch in data_loader:
    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- **CrossEntropyLoss** = negative log-likelihood for classification.
- **weight_decay** = λ ∑ wⱼ² penalty added to the loss = MAP under a Gaussian prior.

### 6. Putting It All Together

1. **MLP + MLE**
    
    Training an MLP with NLL or MSE is exactly finding Ŵ_MLE.
    
2. **MLP + Regularization**
    
    Adding L2 or L1 penalties converts MLE into **MAP**, injecting prior beliefs to combat overfitting.
    
3. **Beyond MAP**
    
    More advanced regularizers (dropout, batch norm) can be viewed through Bayesian or information-theoretic lenses—approximating full posterior inference or constraining representations.
    

---