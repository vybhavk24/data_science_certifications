# Mathematics_c3_w4

## Confidence Intervals: Overview

### 1. What Is a Confidence Interval?

A confidence interval (CI) is a range of values, derived from sample data, that is likely to contain the true but unknown population parameter.

Rather than giving a single point estimate, a CI quantifies uncertainty by providing an interval around that estimate with a specified confidence level (for example, 95%).

### 2. Frequentist Interpretation

A 95% CI means that if you repeated your sampling procedure many times and built a CI each time, approximately 95% of those intervals would contain the true parameter.

Key points:

- The parameter is fixed; the interval is random.
- We do not say “there is a 95% probability the true value is in this one interval.”
- We say “95% of such intervals, constructed in this way, cover the true value.”

### 3. The General Formula

Most CI’s follow the pattern:

```
point_estimate  ±  critical_value  ×  standard_error
```

- **point_estimate**: statistic computed from your sample (mean, proportion, etc.).
- **critical_value**: number from a reference distribution (Normal or t) corresponding to your confidence level.
- **standard_error**: estimated standard deviation of the estimator.

### 4. CI for a Population Mean

### 4.1 Known Variance (σ known)

```
CI = x̄  ±  z_{α/2}  *  (σ / √n)
```

- `x̄` is the sample mean.
- `z_{α/2}` is the Normal critical value (for 95%, z_{0.025}=1.96).
- `σ` is the known population standard deviation.
- `n` is the sample size.

### 4.2 Unknown Variance (σ unknown)

```
CI = x̄  ±  t_{n−1, α/2}  *  (s / √n)
```

- `t_{n−1, α/2}` is the t-distribution critical value with `n−1` degrees of freedom.
- `s` is the sample standard deviation (unbiased estimate of σ).

### 5. CI for a Population Proportion

When each observation is success/failure, let `p̂ = x/n`:

```
CI = p̂  ±  z_{α/2}  *  √[ p̂ (1−p̂) / n ]
```

- Assumes `n p̂` and `n (1−p̂)` are both ≥ 5 for the Normal approximation to hold.
- For small samples or extreme p̂, consider Wilson or Agresti-Coull intervals.

### 6. Summary of Common Formulas

| Parameter | CI Formula | Reference Distribution |
| --- | --- | --- |
| Mean (σ known) | `x̄ ± z_{α/2} · (σ/√n)` | Normal |
| Mean (σ unknown) | `x̄ ± t_{n−1,α/2} · (s/√n)` | t |
| Proportion | `p̂ ± z_{α/2} · √[p̂(1−p̂)/n]` | Normal approx. |
| Difference of Means | `(x̄₁−x̄₂) ± z_{α/2}·√[σ₁²/n₁ + σ₂²/n₂]` (σ known) | Normal |
| Difference of Proportions | `(p̂₁−p̂₂) ± z_{α/2}·√[p̂₁(1−p̂₁)/n₁ + p̂₂(1−p̂₂)/n₂]` | Normal approx. |

### 7. Geometric & Visual Intuition

- Picture many repeated samples and their CIs on one plot.
- Each CI is a horizontal line segment around its sample estimate.
- About 95% of those segments cross the true parameter line; the others miss.

Overlaying intervals shows how larger samples (bigger `n`) produce narrower bands, reflecting increased precision.

### 8. Applications in ML & Data Science

- **Model Performance**: Report 95% CI for cross-validation accuracy or AUC.
- **Parameter Uncertainty**: Give CIs for regression coefficients to assess which features have reliable effects.
- **A/B Testing**: Compute CIs for conversion rates or lift to guide decision thresholds.
- **Feature Importance**: Bootstrap feature-importance scores and display their CIs to quantify stability.

### 9. Practice Problems & Python Examples

### 9.1 CI for the Mean (Unknown σ)

```python
import numpy as np
from scipy.stats import t

# Sample data: daily sales
data = np.array([12, 15, 9, 20, 14, 17, 13])
n = len(data)
x_bar = data.mean()
s = data.std(ddof=1)

alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df=n-1)
margin = t_crit * s / np.sqrt(n)

ci_lower = x_bar - margin
ci_upper = x_bar + margin

print(f"95% CI for mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

### 9.2 CI for a Proportion

```python
import numpy as np
from scipy.stats import norm

# Survey data: 40 out of 120 users clicked
x, n = 40, 120
p_hat = x / n

z_crit = norm.ppf(1 - 0.05/2)
se = np.sqrt(p_hat * (1 - p_hat) / n)

ci = (p_hat - z_crit*se, p_hat + z_crit*se)
print(f"95% CI for proportion: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### 9.3 Bootstrap CI for Median

```python
import numpy as np

data = np.random.exponential(scale=5, size=100)
medians = [np.median(np.random.choice(data, size=len(data), replace=True))
           for _ in range(5000)]
ci_lower, ci_upper = np.percentile(medians, [2.5, 97.5])
print(f"Bootstrap 95% CI for median: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

---

## Confidence Intervals: Adapting Levels and Widths

### 1. What “Changing the Interval” Means

A confidence interval (CI) is defined by three ingredients:

- A **point estimate** (e.g., sample mean or proportion)
- A **critical value** (from a reference distribution, tied to your chosen confidence level)
- A **standard error** (how much your estimator varies)

Changing the interval can mean:

- Altering the **confidence level** (e.g., 90%, 95%, 99%)
- Adjusting the **sample size** to achieve a narrower interval
- Switching the **reference distribution** or using an alternative interval method

### 2. Effect of Confidence Level

As you raise the confidence level, your critical value grows, widening the interval.

| Confidence Level | α | Critical Value (z<sub>α/2</sub>) |
| --- | --- | --- |
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | 1.960 |
| 99% | 0.01 | 2.576 |

Formula for a mean with known σ:

```
CI = x̄  ±  z_{α/2}  *  (σ / √n)
```

- Increasing z<sub>α/2</sub> (by lowering α) **widens** your interval.
- Decreasing z<sub>α/2</sub> (higher α) **narrows** your interval but lowers confidence.

### 3. Effect of Sample Size and Variability

The interval width (W) is:

```
W = 2 · z_{α/2} · SE
```

For a mean (σ known):

```
SE = σ / √n
⇒ W = 2·z_{α/2}·(σ/√n)
```

- **Larger n** ⇒ smaller SE ⇒ narrower CI.
- **Higher σ** ⇒ larger SE ⇒ wider CI.

You can solve for n to achieve a desired half-width *m*:

```
n = ( z_{α/2} · σ / m )²
```

### 4. Alternative Interval Types for Proportions

Beyond the Normal-approximation interval, you can choose:

- **Wilson score interval**: better coverage when n·p̂ is small
- **Agresti–Coull interval**: adds pseudo-counts to stabilize extreme proportions
- **Exact (Clopper–Pearson)**: inverts the Binomial test for guaranteed coverage

Each alternative trades off complexity for improved performance at small n or extreme p̂.

### 5. Python Examples

### 5.1 Computing CIs at Multiple Levels

```python
import numpy as np
from scipy.stats import norm

# Sample data
data = np.random.normal(loc=50, scale=10, size=30)
x_bar = data.mean()
sigma = data.std(ddof=0)
n = len(data)

for conf in [0.90, 0.95, 0.99]:
    alpha = 1 - conf
    z = norm.ppf(1 - alpha/2)
    se = sigma / np.sqrt(n)
    lower, upper = x_bar - z*se, x_bar + z*se
    print(f"{int(conf*100)}% CI: [{lower:.2f}, {upper:.2f}]")
```

### 5.2 Visualizing Interval Width vs Confidence Level

```python
import matplotlib.pyplot as plt

conf_levels = np.linspace(0.80, 0.99, 20)
widths = []
for conf in conf_levels:
    z = norm.ppf(1 - (1-conf)/2)
    widths.append(2*z*(sigma/np.sqrt(n)))

plt.plot(conf_levels*100, widths, marker='o')
plt.xlabel('Confidence Level (%)')
plt.ylabel('Interval Width')
plt.title('Width vs Confidence Level')
plt.show()
```

### 6. Summary

- **Higher confidence** ⇒ larger critical value ⇒ **wider** interval
- **Larger sample size** ⇒ smaller SE ⇒ **narrower** interval
- **Greater variability** ⇒ larger SE ⇒ **wider** interval
- Alternative intervals (Wilson, Agresti–Coull, exact) can improve coverage in edge cases

By balancing confidence level, sample size, and interval type, you tailor intervals to your precision and certainty needs.

---

## Confidence Intervals: Margin of Error

### 1. What Is the Margin of Error?

The **margin of error** (MOE) is half the width of a confidence interval. It quantifies the maximum expected difference between your sample estimate and the true population parameter at a given confidence level.

In formula form:

```
Confidence Interval = point_estimate ± margin_of_error
```

### 2. General MOE Formula

The margin of error is the product of:

- A **critical value** from a reference distribution
- The **standard error** (SE) of your point estimator

```
margin_of_error = critical_value  *  standard_error
```

- `critical_value` = z₍α/2₎ for Normal-based intervals or t₍n−1,α/2₎ for t-based
- `standard_error` depends on your estimator (mean, proportion, etc.)

### 3. MOE for a Population Mean

### 3.1 Known Population Variance (σ known)

```
MOE = z_{α/2} * ( σ / sqrt(n) )
CI = x̄ ± MOE
```

- `x̄` is the sample mean
- `σ` is the known population standard deviation
- `n` is the sample size
- `z_{α/2}` is the Normal critical value (e.g., 1.96 for 95% confidence)

### 3.2 Unknown Population Variance (σ unknown)

```
MOE = t_{n−1, α/2} * ( s / sqrt(n) )
CI = x̄ ± MOE
```

- `s` is the sample standard deviation (`ddof=1`)
- `t_{n−1, α/2}` is the t-distribution critical value with `n−1` degrees of freedom

### 4. MOE for a Population Proportion

When estimating a proportion `p̂ = x/n`:

```
MOE = z_{α/2} * sqrt[ p̂ * (1 − p̂) / n ]
CI = p̂ ± MOE
```

- `x` = number of successes, `n` = total trials
- Normal approximation applies when `n·p̂ ≥ 5` and `n·(1−p̂) ≥ 5`

### 5. How MOE Changes

- **Confidence Level**Higher confidence ⇒ larger critical value ⇒ wider MOE
- **Sample Size**Larger `n` ⇒ smaller SE ⇒ narrower MOE
- **Data Variability**Higher σ or p̂(1−p̂) ⇒ larger SE ⇒ wider MOE

You can solve for required sample size to achieve a target MOE (*m*):

```
n = ( z_{α/2} * σ / m )²      (for mean, σ known)
n = ( z_{α/2}² * p*(1−p) ) / m²  (for proportion, p ≈ estimated)
```

### 6. Python Examples

### 6.1 MOE for Sample Mean

```python
import numpy as np
from scipy.stats import norm, t

data = np.array([12, 15, 9, 20, 14, 17, 13])
n = len(data)
x_bar = data.mean()
s = data.std(ddof=1)

# 95% normal-based MOE (t instead of z)
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df=n-1)
moe_mean = t_crit * s / np.sqrt(n)

print(f"Sample mean: {x_bar:.2f}")
print(f"95% MOE (mean): ±{moe_mean:.2f}")
print(f"95% CI: [{x_bar-moe_mean:.2f}, {x_bar+moe_mean:.2f}]")
```

### 6.2 MOE for Sample Proportion

```python
import numpy as np
from scipy.stats import norm

x, n = 40, 120  # e.g., 40 successes out of 120
p_hat = x / n

z_crit = norm.ppf(1 - 0.05/2)
moe_prop = z_crit * np.sqrt(p_hat * (1-p_hat) / n)

print(f"Sample proportion: {p_hat:.3f}")
print(f"95% MOE (prop): ±{moe_prop:.3f}")
print(f"95% CI: [{p_hat-moe_prop:.3f}, {p_hat+moe_prop:.3f}]")
```

### 7. Practice Problems

1. **Mean MOE**
    
    You measure 50 CPU response times (ms), sample mean 200 ms, sample std = 20 ms. Compute 90% MOE and CI.
    
2. **Proportion MOE**
    
    In a survey of 250 users, 68 use feature X. Find the 99% MOE and CI for the true usage rate.
    
3. **Required n**
    
    You want a MOE no larger than 2 seconds for a mean with σ ≈ 10 s at 95% confidence. Calculate the needed sample size.
    

### 8. Visual Intuition

Imagine many repeated samples of size *n*. For each, draw the CI as a segment around its estimate.

- The length of each segment = 2·MOE.
- About 95% of segments will cover the true parameter.
- Taller bars (narrower segments) come from larger *n* or lower variability.

Plotting segment widths against *n* or confidence levels shows MOE shrinking or growing accordingly.

### 9. Summary

- Margin of error = critical value × standard error.
- It directly controls CI width and depends on confidence level, sample size, and variability.
- You can plan sample sizes to achieve a desired MOE.

---

## Confidence Intervals: Calculation Steps and Worked Example

### 1. Calculation Steps for a Confidence Interval

1. Identify the **parameter** and its **point estimate**
    - e.g., population mean μ with sample mean x̄, or proportion p with sample proportion p̂.
2. Choose the **confidence level** (1−α), e.g. 90%, 95%, 99%
    - Determines your critical value z₍α/2₎ (Normal) or t₍n−1,α/2₎ (Student’s t).
3. Determine the **reference distribution** and find the **critical value**
    - If σ is known or n large: use Normal → z₍α/2₎ from standard Normal table
    - If σ unknown and n small: use Student’s t with n−1 degrees of freedom
4. Compute the **standard error (SE)** of the estimator
    - For a mean (σ known):
        
        ```
        SE = σ / √n
        ```
        
    - For a mean (σ unknown):
        
        ```
        SE = s / √n
        ```
        
    - For a proportion:
        
        ```
        SE = √[ p̂ (1−p̂) / n ]
        ```
        
5. Compute the **margin of error (MOE)**
    
    ```
    MOE = critical_value × SE
    ```
    
6. Construct the **confidence interval**
    
    ```
    CI = point_estimate ± MOE
    ```
    

### 2. Worked Example: 95% CI for a Mean (σ Unknown)

**Scenario:**

You measure the reaction times (ms) of n = 25 participants.

- Sample mean:
    
    ```
    x̄ = 350 ms
    ```
    
- Sample standard deviation:
    
    ```
    s = 40 ms
    ```
    
- Desired confidence level: 95% → α = 0.05

### Step 1: Point Estimate

x̄ = 350 ms

### Step 2: Confidence Level & Critical Value

- Degrees of freedom: n−1 = 24
- For 95% CI, α/2 = 0.025
- From t‐table:
    
    ```
    t_{24, 0.025} ≈ 2.064
    ```
    

### Step 3: Standard Error

```
SE = s / √n = 40 / √25 = 40 / 5 = 8 ms
```

### Step 4: Margin of Error

```
MOE = t_{24,0.025} × SE = 2.064 × 8 ≈ 16.512 ms
```

### Step 5: Confidence Interval

```
CI = x̄ ± MOE
   = 350 ± 16.512
   = [333.488, 366.512] ms
```

**Interpretation:**

If we repeated this experiment many times, approximately 95% of the intervals constructed this way would contain the true mean reaction time.

### 3. Python Code Example

```python
import numpy as np
from scipy.stats import t

# Given data
n = 25
x_bar = 350.0
s = 40.0
alpha = 0.05

# Degrees of freedom
df = n - 1

# Critical t-value for 95% CI
t_crit = t.ppf(1 - alpha/2, df)

# Standard error
se = s / np.sqrt(n)

# Margin of error
moe = t_crit * se

# Confidence interval
ci_lower = x_bar - moe
ci_upper = x_bar + moe

print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}] ms")
```

### 4. Additional Notes

- For **large samples** (n ≥ 30) and unknown σ, the t‐distribution approaches Normal; you can often use z₍α/2₎.
- For **proportions**, replace point estimate with p̂ and SE with √[p̂(1−p̂)/n].
- For **small or skewed data**, consider **bootstrap intervals** by resampling.

---

## Calculating Sample Size

### 1. Why Determine Sample Size?

Before collecting data, you often need to know **how many observations** (*n*) will give you estimates precise enough for your goals.

- In A/B testing, you want enough users to detect a lift.
- For population means or proportions, you target a **margin of error** (*m*) at a given confidence level.
- In ML model evaluation, you choose test set size to narrow performance CIs.

### 2. Key Ingredients

1. **Confidence level** (1–α): how sure you want to be (e.g., 95%).
2. **Margin of error** *m*: maximum half‐width of the CI.
3. **Variability estimate**:
    - For means: population σ or pilot sample *s*.
    - For proportions: anticipated p (or worst‐case 0.5).
4. **Reference distribution**: Normal (z) or t (small n, unknown σ).

### 3. Sample Size for a Population Mean

### 3.1 Known σ (Normal‐based)

To achieve CI half‐width *m* at confidence level 1–α:

```
n = ( z_{α/2} * σ / m )^2
```

- `z_{α/2}` is the z‐score (e.g., 1.96 for 95%).
- Solve for n and round up to the next integer.

### 3.2 Unknown σ (Using Pilot s)

When σ is unknown, use estimated *s* from prior data:

```
n ≈ ( z_{α/2} * s / m )^2
```

If *n* is small, you could iterate using t_{n−1,α/2} instead of z, but the Normal approximation often suffices when planning.

### 4. Sample Size for a Proportion

To bound a proportion estimate `p̂` within ±*m*:

```
n = ( z_{α/2}^2 * p * (1−p) ) / m^2
```

- If *p* unknown, use *p*=0.5 for worst‐case (maximizes p(1−p)).
- Round up after computing.

### 5. Sample Size for Difference of Means

For two groups (equal n per group), targeting margin *m* on (x̄₁−x̄₂):

```
n = ( 2 * z_{α/2}^2 * σ^2 ) / m^2
```

- `σ^2` is common variance.
- If groups have different variances σ₁², σ₂²:

```
n = [ ( z_{α/2} )^2 * (σ₁^2 + σ₂^2) ] / m^2
```

### 6. Sample Size for Difference of Proportions

To estimate (p̂₁−p̂₂) within ±*m*:

```
n = ( z_{α/2}^2 * [p₁(1−p₁) + p₂(1−p₂)] ) / m^2
```

- If p₁,p₂ unknown, pilot estimates or worst‐case 0.5 can be used.

### 7. Python Examples

```python
import numpy as np
from scipy.stats import norm

def sample_size_mean(sigma, m, conf=0.95):
    z = norm.ppf(1 - (1-conf)/2)
    return int(np.ceil((z * sigma / m)**2))

def sample_size_prop(p, m, conf=0.95):
    z = norm.ppf(1 - (1-conf)/2)
    return int(np.ceil((z**2 * p * (1-p)) / m**2))

# Example: mean with sigma=10, m=2, conf=95%
print("n (mean):", sample_size_mean(10, 2, 0.95))

# Example: proportion with p=0.3, m=0.05, conf=95%
print("n (prop):", sample_size_prop(0.3, 0.05, 0.95))

# Worst-case proportion
print("n (prop worst-case):", sample_size_prop(0.5, 0.05, 0.95))
```

### 8. Worked Example

**Goal:** Estimate average session length with ±1 minute at 95% confidence. Pilot data show s≈4 minutes.

1. σ≈s=4, m=1, z₀.₀₂₅=1.96
2. `n ≈ (1.96 * 4 / 1)^2 = (7.84)^2 ≈ 61.5
⇒ n = 62`

You need at least 62 sessions measured.

---

## Difference Between Confidence and Probability

### 1. What Is Probability?

- Probability is a numerical measure of how likely an event is to occur.
- In the **frequentist** view, P(A) is the long-run fraction of times event A happens if you repeat an experiment infinitely.
- In the **Bayesian** view, P(A) expresses your degree of belief in event A given what you know.

Example:

- “The probability of a fair coin landing heads is 0.5.”
- “Given our prior and 10 observed flips, there’s a 90% chance the next flip is heads.”

### 2. What Is Confidence (in CIs)?

- **Confidence** refers specifically to the reliability of an interval-building procedure.
- A 95% confidence interval for a parameter means that if you repeated your sampling and interval construction many times, **95%** of those intervals would cover the true parameter.
- It **does not** mean there’s a 95% probability this one interval contains the parameter—because in the frequentist framework, the parameter is fixed and not random.

### 3. Key Distinctions

| Aspect | Probability | Confidence |
| --- | --- | --- |
| What’s random? | Events or parameters (Bayesian) | The interval (frequentist) |
| Subject of statement | “P(A) = 0.3” → event A | “95% of intervals cover true θ” |
| Interpretation | Degree of belief or long-run frequency | Long-run performance of the interval-building rule |
| Applicable to one trial? | Yes (you can say “there’s 30% chance…”) | No (can’t say “95% chance this θ is in here”) |

### 4. Concrete Example

Suppose you survey *n=100* users and observe 30 clicks. You build a 95% confidence interval for the true click-through rate *p*:

```
p̂ = 0.30
SE = sqrt[0.30*0.70/100] ≈ 0.0458
CI = 0.30 ± 1.96·0.0458
   = [0.210, 0.390]
```

- **Confidence statement**: If you took many samples of 100 users and built CIs this way, **95%** of those intervals would contain the true *p*.
- **Probability statement**: It’s **not** correct to say “there’s a 95% probability that *p* lies between 0.21 and 0.39.” Instead, we rely on the long-run coverage property.

### 5. When “Probability” Applies to Intervals

In a **Bayesian** framework, you treat the parameter *p* as random and compute a **credible interval**:

```
Posterior p | data ~ Beta(α + 30, β + 70)
95% credible interval: [0.22, 0.38]
```

- Here you **can** say “given our data and prior, there’s a 95% probability *p* lies in this interval.”
- That is a statement about the posterior distribution of *p*, not about repeated sampling.

### 6. Takeaways

- **Probability** quantifies uncertainty about events (or parameters in Bayesian inference).
- **Confidence** in CIs measures how well your interval-construction method works over many repeated samples.
- Never conflate a frequentist CI’s 95% confidence with a 95% probability for that one realized interval.

Understanding this difference is crucial for correct interpretation of statistical results.

---

## Confidence Interval for a Mean When σ Is Unknown

### 1. Prerequisite Reminder

Before we jump in, make sure you’re comfortable with:

- **Sample mean** (`x_bar`) and **sample standard deviation** (`s`) calculations.
- Basic **normal-based confidence intervals** when σ (population standard deviation) is known.
- The notion of a **sampling distribution** for `x_bar` and how its spread shrinks as sample size *n* grows.

### 2. Why Unknown σ Changes Things

In real life you almost never know the true σ. You replace it with your sample estimate `s`. That substitution adds extra uncertainty—especially with small samples—so the ratio

```
T = (x_bar − μ) / (s / √n)
```

no longer follows a standard Normal. Instead it follows a **Student’s t-distribution** with **ν = n − 1** degrees of freedom, which has “fatter tails” to account for the extra uncertainty in estimating σ.

### 3. The t-Distribution in Plain Language

- Think of the t-distribution as a family of bell curves indexed by degrees of freedom (ν = n − 1).
- With very few data points, the curve is wide and flat at the top, putting more probability far from zero.
- As n grows, ν grows, and the t-distribution converges to the Normal curve.

Visually, draw a series of curves for ν = 5, 10, 30, 100. Notice how the peak sharpens and tails thin out, approaching the Normal.

### 4. Formula in a Copy-Paste-Friendly Code Block

```markdown
CI = x_bar ± t_{ν, α/2} * ( s / sqrt(n) )
```

- `x_bar` : sample mean
- `s` : sample standard deviation (use ddof=1)
- `n` : sample size
- `ν = n − 1` : degrees of freedom for the t-distribution
- `t_{ν, α/2}` : critical value so that the two-tailed area beyond ±t equals α

### 5. Step-by-Step Explanation

1. **Compute x_bar**
    
    Add up your n observations and divide by n.
    
2. **Compute s**
    
    ```
    s = sqrt( (1/(n−1)) * sum (x_i − x_bar)^2 )
    ```
    
    This “n−1” denominator corrects bias when estimating σ.
    
3. **Choose confidence level**
    
    For 95% CI, α = 0.05 → α/2 = 0.025.
    
4. **Find t-critical**
    
    Use a table or software call:
    
    ```python
    from scipy.stats import t
    t_star = t.ppf(1 - alpha/2, df=n-1)
    ```
    
5. **Compute standard error (SE)**
    
    ```
    SE = s / sqrt(n)
    ```
    
6. **Compute margin of error (MOE)**
    
    ```
    MOE = t_star * SE
    ```
    
7. **Construct CI**
    
    ```
    Lower bound = x_bar − MOE
    Upper bound = x_bar + MOE
    ```
    

### 6. Real-World ML & DS Applications

- **Cross-Validation Error**
    
    When you report the average validation error over k folds, you don’t know the “true” variability. A t-interval around the mean error gives a realistic band for expected performance.
    
- **A/B Tests with Small Groups**
    
    If only a handful of users see variant B, use a t-interval on average metrics (e.g., time on page) to judge significance rather than a Normal approximation.
    
- **Feature Stability**
    
    Estimating the average of a noisy feature (like daily server latency) from a small pilot run—you need a t-interval to decide if you have enough data to trust that feature.
    

### 7. Visual & Geometric Intuition

- **Overlayed Curves**: Plot Normal vs t₅, t₁₀, t₃₀. See heavier tails in t₅ → wider CIs.
- **CI Bands**: Simulate many small samples, draw CIs on the same axis—all span a horizontal line for the true μ. Notice about 95% cover it, others miss.

### 8. Practice Problems & Python Exercises

### 8.1 Manual Calculation

Data (seconds):

```
[2.1, 2.4, 1.9, 2.3, 2.7, 2.2]
```

1. Compute n, x_bar, s.
2. For 95% CI, find ν and t* from a t-table.
3. Calculate SE, MOE, and the interval.

### 8.2 Python Code

```python
import numpy as np
from scipy.stats import t

# Sample data
data = np.array([2.1, 2.4, 1.9, 2.3, 2.7, 2.2])
n = len(data)

# 1. Compute x_bar and s
x_bar = data.mean()
s = data.std(ddof=1)

# 2. Confidence parameters
conf = 0.95
alpha = 1 - conf
df = n - 1

# 3. t-critical
t_star = t.ppf(1 - alpha/2, df)

# 4. SE and MOE
SE = s / np.sqrt(n)
MOE = t_star * SE

# 5. CI
ci_lower = x_bar - MOE
ci_upper = x_bar + MOE

print(f"n = {n}")
print(f"Sample mean = {x_bar:.3f}")
print(f"Sample std  = {s:.3f}")
print(f"{int(conf*100)}% CI    = [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### 8.3 Plotting the t-Distribution vs Normal

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

xs = np.linspace(-4, 4, 200)
plt.plot(xs, norm.pdf(xs), label='Standard Normal')
for df in [5, 10, 30]:
    plt.plot(xs, t.pdf(xs, df), label=f't (df={df})')
plt.legend()
plt.title('Normal vs t-Distributions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Observe how t₅ and t₁₀ have heavier tails ⇒ require larger critical values.

---

## Confidence Interval for a Mean When σ Is Unknown

### 1. Prerequisite Reminder

Before we jump in, make sure you’re comfortable with:

- **Sample mean** (`x_bar`) and **sample standard deviation** (`s`) calculations.
- Basic **normal-based confidence intervals** when σ (population standard deviation) is known.
- The notion of a **sampling distribution** for `x_bar` and how its spread shrinks as sample size *n* grows.

### 2. Why Unknown σ Changes Things

In real life you almost never know the true σ. You replace it with your sample estimate `s`. That substitution adds extra uncertainty—especially with small samples—so the ratio

```
T = (x_bar − μ) / (s / √n)
```

no longer follows a standard Normal. Instead it follows a **Student’s t-distribution** with **ν = n − 1** degrees of freedom, which has “fatter tails” to account for the extra uncertainty in estimating σ.

### 3. The t-Distribution in Plain Language

- Think of the t-distribution as a family of bell curves indexed by degrees of freedom (ν = n − 1).
- With very few data points, the curve is wide and flat at the top, putting more probability far from zero.
- As n grows, ν grows, and the t-distribution converges to the Normal curve.

Visually, draw a series of curves for ν = 5, 10, 30, 100. Notice how the peak sharpens and tails thin out, approaching the Normal.

### 4. Formula in a Copy-Paste-Friendly Code Block

```markdown
CI = x_bar ± t_{ν, α/2} * ( s / sqrt(n) )
```

- `x_bar` : sample mean
- `s` : sample standard deviation (use ddof=1)
- `n` : sample size
- `ν = n − 1` : degrees of freedom for the t-distribution
- `t_{ν, α/2}` : critical value so that the two-tailed area beyond ±t equals α

### 5. Step-by-Step Explanation

1. **Compute x_bar**
    
    Add up your n observations and divide by n.
    
2. **Compute s**
    
    ```
    s = sqrt( (1/(n−1)) * sum (x_i − x_bar)^2 )
    ```
    
    This “n−1” denominator corrects bias when estimating σ.
    
3. **Choose confidence level**
    
    For 95% CI, α = 0.05 → α/2 = 0.025.
    
4. **Find t-critical**
    
    Use a table or software call:
    
    ```python
    from scipy.stats import t
    t_star = t.ppf(1 - alpha/2, df=n-1)
    ```
    
5. **Compute standard error (SE)**
    
    ```
    SE = s / sqrt(n)
    ```
    
6. **Compute margin of error (MOE)**
    
    ```
    MOE = t_star * SE
    ```
    
7. **Construct CI**
    
    ```
    Lower bound = x_bar − MOE
    Upper bound = x_bar + MOE
    ```
    

### 6. Real-World ML & DS Applications

- **Cross-Validation Error**
    
    When you report the average validation error over k folds, you don’t know the “true” variability. A t-interval around the mean error gives a realistic band for expected performance.
    
- **A/B Tests with Small Groups**
    
    If only a handful of users see variant B, use a t-interval on average metrics (e.g., time on page) to judge significance rather than a Normal approximation.
    
- **Feature Stability**
    
    Estimating the average of a noisy feature (like daily server latency) from a small pilot run—you need a t-interval to decide if you have enough data to trust that feature.
    

### 7. Visual & Geometric Intuition

- **Overlayed Curves**: Plot Normal vs t₅, t₁₀, t₃₀. See heavier tails in t₅ → wider CIs.
- **CI Bands**: Simulate many small samples, draw CIs on the same axis—all span a horizontal line for the true μ. Notice about 95% cover it, others miss.

### 8. Practice Problems & Python Exercises

### 8.1 Manual Calculation

Data (seconds):

```
[2.1, 2.4, 1.9, 2.3, 2.7, 2.2]
```

1. Compute n, x_bar, s.
2. For 95% CI, find ν and t* from a t-table.
3. Calculate SE, MOE, and the interval.

### 8.2 Python Code

```python
import numpy as np
from scipy.stats import t

# Sample data
data = np.array([2.1, 2.4, 1.9, 2.3, 2.7, 2.2])
n = len(data)

# 1. Compute x_bar and s
x_bar = data.mean()
s = data.std(ddof=1)

# 2. Confidence parameters
conf = 0.95
alpha = 1 - conf
df = n - 1

# 3. t-critical
t_star = t.ppf(1 - alpha/2, df)

# 4. SE and MOE
SE = s / np.sqrt(n)
MOE = t_star * SE

# 5. CI
ci_lower = x_bar - MOE
ci_upper = x_bar + MOE

print(f"n = {n}")
print(f"Sample mean = {x_bar:.3f}")
print(f"Sample std  = {s:.3f}")
print(f"{int(conf*100)}% CI    = [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### 8.3 Plotting the t-Distribution vs Normal

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

xs = np.linspace(-4, 4, 200)
plt.plot(xs, norm.pdf(xs), label='Standard Normal')
for df in [5, 10, 30]:
    plt.plot(xs, t.pdf(xs, df), label=f't (df={df})')
plt.legend()
plt.title('Normal vs t-Distributions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Observe how t₅ and t₁₀ have heavier tails ⇒ require larger critical values.

---

## Defining Hypotheses

### 1. What Is a Hypothesis in Testing?

A hypothesis is a **claim about a population parameter** (like a mean, proportion, or difference) that you can check with data.

- Think of a hypothesis test as a courtroom drama:
    - The **null hypothesis (H₀)** is the “defendant”—the status quo or “no effect” claim (innocent until proven guilty).
    - The **alternative hypothesis (H₁ or Hₐ)** is what you suspect might be true instead (the “guilty” claim).

Your data act like evidence. You gather enough evidence to decide whether to **reject H₀** or **fail to reject H₀**, just like a jury’s verdict.

### 2. Null and Alternative Hypotheses

### 2.1 Null Hypothesis (H₀)

- Statement that the parameter equals a specific value or there is no effect.
- Example: “The average click-through rate is 5%.”

### 2.2 Alternative Hypothesis (Hₐ)

- What you want to show is true if H₀ is false.
- Can be:
    - **Two-sided**: parameter is different (greater or smaller)
    - **One-sided**: parameter is strictly greater or strictly smaller

**Example formulas** (for population mean μ compared to μ₀):

```markdown
# Two-sided test
H₀: μ = μ₀
Hₐ: μ ≠ μ₀

# One-sided “greater than” test
H₀: μ = μ₀
Hₐ: μ > μ₀

# One-sided “less than” test
H₀: μ = μ₀
Hₐ: μ < μ₀
```

- In A/B testing, you might set H₀: p₁ = p₂ and Hₐ: p₁ > p₂ if you suspect variant 1 outperforms variant 2.

### 3. How to Define Your Hypotheses

1. **Identify the parameter** you care about
    - Mean (μ), proportion (p), variance (σ²), difference of means, etc.
2. **State the null hypothesis (H₀)**
    - The default value (e.g., μ = 10, p = 0.2).
3. **Choose the alternative (Hₐ)**
    - Two-sided if you just want to detect any change.
    - One-sided if you expect a direction (greater or lesser).
4. **Ensure hypotheses are mutually exclusive and cover all possibilities**
    - H₀ ∪ Hₐ covers every possible true value.
    - H₀ ∩ Hₐ = ∅.

### 4. Real-World DS & ML Examples

- **A/B Testing Conversion Rate**
    - H₀: p_click_A = p_click_B
    - Hₐ: p_click_A ≠ p_click_B
- **Model Error Reduction**
    - H₀: average cross-val error = 0.10
    - Hₐ: average cross-val error < 0.10
- **Feature Importance Check**
    - H₀: coefficient β_j = 0 (no effect)
    - Hₐ: coefficient β_j ≠ 0 (feature matters)

Every time you set up a statistical test in your ML pipeline—whether comparing models or checking a feature—you’re defining these hypotheses first.

### 5. Practice Problems & Python Exercises

### 5.1 Coin-Flip Fairness

You flip a coin 60 times and get 40 heads.

1. Define H₀ and Hₐ for a two-sided test of fairness.
2. In Python, compute how many heads you’d expect under H₀.

```python
import numpy as np
from scipy.stats import binom_test

# Data
n, x = 60, 40

# Two-sided binomial test
p_value = binom_test(x, n, p=0.5, alternative='two-sided')
print("p-value:", p_value)
```

### 5.2 Mean Latency Against Target

You measure 20 response times (ms). You want to test if the true mean exceeds 200 ms.

1. H₀: μ = 200 ms; Hₐ: μ > 200 ms.
2. Simulate data, compute sample mean, then use a one-sample t-test:

```python
import numpy as np
from scipy.stats import ttest_1samp

# Simulated latencies
data = np.random.normal(210, 20, size=20)

# One-sided test (greater)
t_stat, p_value_two = ttest_1samp(data, popmean=200)
# convert two-sided p to one-sided
p_value_one = p_value_two / 2 if t_stat > 0 else 1 - p_value_two/2

print("t-stat:", t_stat, "one-sided p:", p_value_one)
```

### 6. Geometric & Visual Intuition

- **Sampling Distributions**
    - Under H₀, draw the distribution of your test statistic (e.g., t or z).
    - Shade the rejection regions in the tail(s).
- **Observed Statistic**
    - Plot the dot where your sample falls.
    - If it lands in the shaded region, you **reject H₀**.

Describing the picture helps cement why an extreme result under H₀ gives reason to reject it.

---

## Type I and Type II Errors

### 1. Conceptual Overview

In hypothesis testing, you make a decision based on data: either **reject H₀** (the null hypothesis) or **fail to reject H₀**. Because your decision is based on random samples, two kinds of mistakes can happen:

- A **Type I error** is a false positive: you reject H₀ when it is actually true.
- A **Type II error** is a false negative: you fail to reject H₀ when the alternative H₁ is actually true.

Think of it like a smoke alarm:

- Type I error: the alarm rings (reject H₀) when there’s no fire (H₀ true).
- Type II error: the alarm stays silent (fail to reject H₀) when there’s a fire (H₁ true).

### 2. Formal Definitions and Formulas

1. **Significance level (α)**
    
    The probability of a Type I error:
    
    ```markdown
    α = P(reject H₀ │ H₀ is true)
    ```
    
2. **Type II error rate (β)**
    
    The probability of a Type II error:
    
    ```markdown
    β = P(fail to reject H₀ │ H₁ is true)
    ```
    
3. **Power of the test (1−β)**
    
    The probability of correctly rejecting H₀ when H₁ is true:
    
    ```markdown
    power = 1 - β = P(reject H₀ │ H₁ is true)
    ```
    

Key trade‐off:

- Lowering α (being stricter about false positives) usually **increases** β (more false negatives).
- Increasing sample size *n* can **reduce both** α and β.

### 3. Step-by-Step Breakdown

1. **Set up H₀ and H₁**
    
    Example: test whether a coin is fair
    
    ```markdown
    H₀: p = 0.5
    H₁: p ≠ 0.5
    ```
    
2. **Choose α**
    
    Common choices are 0.05 or 0.01. This fixes the rejection region under H₀.
    
3. **Determine rejection region**
    
    Based on the sampling distribution under H₀ (e.g., Normal or t), find critical values so that
    
    ```markdown
    P(statistic in rejection region │ H₀ true) = α
    ```
    
4. **Compute test statistic** from your sample and compare to critical value
    - If it falls in the rejection region, you **reject H₀** (risking a Type I error with probability α).
    - Otherwise, you **fail to reject H₀** (risking a Type II error).
5. **Calculate β** (requires specifying a particular value under H₁)
    - Under H₁, determine the sampling distribution of your statistic if the true parameter equals some value—for example p = 0.6.
    - Compute
        
        ```markdown
        β = P(statistic not in rejection region │ H₁ true)
        ```
        

### 4. Geometric & Visual Intuition

- Draw the **sampling distribution** of your test statistic under H₀ (e.g., a bell curve centered at 0).
- Shade the tails beyond the critical value(s) — that area is α (Type I error).
- Draw the distribution under a specific alternative (e.g., a bell curve shifted right).
- Shade the area of that alternative curve that **does not** overlap the rejection tails — that area is β (Type II error).

Visually, increasing α (wider rejection tails) shrinks β; shifting distributions further apart (larger effect size) also shrinks β.

### 5. Real-World DS & ML Examples

- **Spam Filter Tuning**
    - H₀: “email is not spam.”
    - Type I error (α): flagging a good email as spam.
    - Type II error (β): letting spam through as good mail.
    - You set α low to avoid losing important emails, then collect more data to reduce β.
- **Model Performance Threshold**
    - H₀: model accuracy = 80%.
    - Type I error: concluding accuracy > 80% when it’s not.
    - Type II error: failing to detect an improvement to 85%.
    - Increase test sample size to reduce β so you reliably spot real improvements.

### 6. Practice Problems & Python Simulations

### 6.1 Coin-Flip Example

Test H₀: p = 0.5 vs H₁: p = 0.6 at α = 0.05 with n = 50 flips.

1. **Find critical region** using Normal approximation:
    
    ```python
    from scipy.stats import norm
    alpha = 0.05
    z_crit = norm.ppf(1 - alpha/2)  # two-sided
    ```
    
2. **Compute β** for true p = 0.6:
    
    ```python
    import numpy as np
    
    n = 50
    p0 = 0.5
    p1 = 0.6
    z_crit = norm.ppf(1 - 0.05/2)
    
    # critical number of heads
    se0 = np.sqrt(p0*(1-p0)/n)
    lower = p0 - z_crit*se0
    upper = p0 + z_crit*se0
    
    # compute beta: P(p_hat between lower and upper | true p1)
    se1 = np.sqrt(p1*(1-p1)/n)
    beta = norm.cdf((upper - p1)/se1) - norm.cdf((lower - p1)/se1)
    power = 1 - beta
    print(f"β ≈ {beta:.3f}, power ≈ {power:.3f}")
    ```
    

### 6.2 Simulation to Estimate α and β

```python
import numpy as np

def simulate_errors(p_true, p_test, n, alpha, trials=100000):
    # simulate Type I or II error rate
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha/2)
    # rejection bounds under H0 (p_test)
    se0 = np.sqrt(p_test*(1-p_test)/n)
    lower = p_test - z_crit*se0
    upper = p_test + z_crit*se0

    errors = 0
    for _ in range(trials):
        p_hat = np.sum(np.random.rand(n) < p_true) / n
        # Type I error if p_true == p_test and outside bounds
        # Type II error if p_true != p_test and inside bounds
        if ((p_true == p_test) and (p_hat < lower or p_hat > upper)) \
           or ((p_true != p_test) and (lower <= p_hat <= upper)):
            errors += 1
    return errors / trials

# Estimate α when true p = 0.5
alpha_emp = simulate_errors(0.5, 0.5, 50, 0.05)
# Estimate β when true p = 0.6
beta_emp  = simulate_errors(0.6, 0.5, 50, 0.05)
print(f"Empirical α ≈ {alpha_emp:.3f}, empirical β ≈ {beta_emp:.3f}")
```

### 7. Summary

- **Type I error (α)**: rejecting a true null.
- **Type II error (β)**: failing to reject a false null.
- **Power (1−β)**: probability to detect a real effect.
- Balancing α and β is key: choose α first, then adjust sample size to achieve desired β (power).

---

## Right-Tailed, Left-Tailed, and Two-Tailed Tests

### 1. Introduction

When you set up a hypothesis test, you choose not only your null (H₀) and alternative (H₁) hypotheses but also the **direction** of the test. The direction tells you where to look for evidence:

- **Right-tailed**: evidence that the parameter is **greater** than the H₀ value
- **Left-tailed**: evidence that the parameter is **less** than the H₀ value
- **Two-tailed**: evidence that the parameter is **different** (either greater or less) than the H₀ value

Choosing the correct tail type aligns your test with the question you’re asking, and it determines where you place your rejection region(s) in the sampling distribution.

### 2. Intuitive Analogy

Imagine you’re playing darts and you suspect the bullseye has moved:

- In a **right-tailed** test, you only care if the bullseye moved **up** on the board.
- In a **left-tailed** test, you only care if the bullseye moved **down**.
- In a **two-tailed** test, you care if it moved **up or down**.

You draw boundary lines on the dartboard (critical values) and see where your observed dart (test statistic) lands relative to those lines.

### 3. Formal Definitions and Decision Rules

### 3.1 Right-Tailed Test

- H₀: parameter = θ₀
- H₁: parameter > θ₀

Rejection region sits in the **upper** tail of the sampling distribution.

Decision rule:

```
compute test_statistic
if test_statistic > critical_value_right
  reject H₀
else
  fail to reject H₀
```

p-value:

```
p_value = P( Z ≥ observed_statistic │ H₀ )
```

### 3.2 Left-Tailed Test

- H₀: parameter = θ₀
- H₁: parameter < θ₀

Rejection region sits in the **lower** tail.

Decision rule:

```
compute test_statistic
if test_statistic < critical_value_left
  reject H₀
else
  fail to reject H₀
```

p-value:

```
p_value = P( Z ≤ observed_statistic │ H₀ )

```

### 3.3 Two-Tailed Test

- H₀: parameter = θ₀
- H₁: parameter ≠ θ₀

Rejection regions in **both** tails, each with area α/2.

Decision rule:

```
compute test_statistic
if |test_statistic| > critical_value_two
  reject H₀
else
  fail to reject H₀
```

p-value:

```
p_value = 2 × P( Z ≥ |observed_statistic| │ H₀ )
```

### 4. Test Statistic Formula

For a one-sample mean test with known σ, the test statistic is:

```markdown
z = ( x_bar - mu_0 ) / ( sigma / sqrt(n) )
```

- `x_bar` : sample mean
- `mu_0` : hypothesized mean under H₀
- `sigma` : population standard deviation
- `n` : sample size

Under H₀, `z` follows a standard Normal distribution.

### 5. Graphical & Geometric Intuition

- **Right-tailed**: shade the area under the Normal curve to the **right** of `z_crit`
- **Left-tailed**: shade the area under the curve to the **left** of `z_crit`
- **Two-tailed**: shade the two symmetric areas in both tails beyond ±`z_crit`

Plot the standard Normal curve, mark `z_crit` at `z_{α}` (one‐sided) or `z_{α/2}` (two‐sided), and see whether your observed `z` falls into the shaded region.

### 6. Real-World Examples

- **Right-Tailed (Quality Control)**
    
    H₀: defect rate = 2%
    
    H₁: defect rate > 2%
    
    You inspect 1,000 items and compute a proportion test statistic.
    
- **Left-Tailed (A/B Test Speed)**
    
    H₀: average load time = 2s
    
    H₁: average load time < 2s
    
    You measure 50 page loads and compute a t‐statistic using sample standard deviation.
    
- **Two-Tailed (Drug Efficacy)**
    
    H₀: mean blood pressure change = 0
    
    H₁: mean blood pressure change ≠ 0
    
    You run a clinical trial, measure before/after, and compute a paired t‐test.
    

### 7. Practice Problems & Python Exercises

### 7.1 Right-Tailed Z-Test for Proportion

You observe 90 successes in 200 trials. Test H₀: p = 0.3 vs H₁: p > 0.3 at α = 0.05.

```python
import numpy as np
from scipy.stats import norm

# Data
n, x = 200, 90
p0 = 0.3

# 1. Sample proportion
p_hat = x / n

# 2. Test statistic
se = np.sqrt(p0 * (1 - p0) / n)
z = (p_hat - p0) / se

# 3. Critical value and p-value
alpha = 0.05
z_crit = norm.ppf(1 - alpha)
p_value = 1 - norm.cdf(z)

print(f"z-statistic: {z:.3f}")
print(f"Critical z: {z_crit:.3f}")
print(f"p-value: {p_value:.3f}")
```

### 7.2 Left-Tailed T-Test for Mean

You have 12 measurements with sample mean 5.2 and sample sd 1.1. Test H₀: μ = 5 vs H₁: μ < 5 at α = 0.10.

```python
import numpy as np
from scipy.stats import t

# Data summary
n = 12
x_bar = 5.2
s = 1.1
mu0 = 5

# 1. Test statistic
se = s / np.sqrt(n)
t_stat = (x_bar - mu0) / se

# 2. Critical value and p-value
alpha = 0.10
df = n - 1
t_crit = t.ppf(alpha, df)
p_value = t.cdf(t_stat, df)

print(f"t-statistic: {t_stat:.3f}")
print(f"Critical t: {t_crit:.3f}")
print(f"p-value: {p_value:.3f}")
```

### 7.3 Two-Tailed Z-Test for Mean

You measure 50 sample values, get x_bar = 102, σ = 10. Test H₀: μ = 100 vs H₁: μ ≠ 100 at α = 0.05.

```python
import numpy as np
from scipy.stats import norm

# Data summary
n = 50
x_bar = 102
sigma = 10
mu0 = 100

# 1. Test statistic
se = sigma / np.sqrt(n)
z = (x_bar - mu0) / se

# 2. Critical value and p-value
alpha = 0.05
z_crit = norm.ppf(1 - alpha/2)
p_value = 2 * (1 - norm.cdf(abs(z)))

print(f"z-statistic: {z:.3f}")
print(f"Critical z: ±{z_crit:.3f}")
print(f"p-value: {p_value:.3f}")
```

---

## p-Value: In-Depth Understanding

### 1. Conceptual Explanation

A **p-value** quantifies how surprising your data are under the assumption that the null hypothesis (H₀) is true.

- Imagine you flip a coin 100 times expecting 50 heads (H₀: p=0.5).
- You observe 60 heads. A p-value asks:“If the coin were fair, what is the probability of observing 60 or more heads just by chance?”

A **small p-value** (e.g., 0.01) means such an extreme result is rare under H₀ → evidence **against** H₀.

A **large p-value** (e.g., 0.30) means the result is not unusual under H₀ → insufficient evidence to reject H₀.

### 2. Formal Definition and Formula

For a chosen test statistic *T* (like a z-score or t-statistic), the p-value is:

```markdown
p_value =
  P(T ≥ observed_T │ H₀ true)      # right-tailed test
  P(T ≤ observed_T │ H₀ true)      # left-tailed test
  2 × P(T ≥ |observed_T| │ H₀ true) # two-tailed test
```

- **observed_T**: the value you compute from your data.
- **P(…│H₀ true)**: probability under the null distribution.
- Multiply by 2 for two-sided alternatives.

### 3. Step-by-Step Breakdown

1. **Set up H₀ and H₁**, and choose one-, left-, or two-tailed.
2. **Compute the test statistic**
    
    ```
    z = (x_bar − μ₀) / (σ / √n)
    t = (x_bar − μ₀) / (s / √n)
    ```
    
3. **Determine the null distribution** of *T* (Standard Normal, Student’s t, Chi-square, etc.).
4. **Calculate the tail probability** beyond your observed statistic:
    - Right-tailed: area to the right of observed_T
    - Left-tailed: area to the left of observed_T
    - Two-tailed: twice the area in the more extreme tail
5. **Interpret**
    - If `p_value ≤ α` (e.g., 0.05), reject H₀.
    - If `p_value > α`, fail to reject H₀.

### 4. Real-World DS/ML Examples

- **Model Accuracy vs Baseline**
    
    Test if your classifier’s accuracy (e.g., 78% on 1,000 samples) is better than a 75% baseline. Compute a z-statistic for proportions and a one-tailed p-value.
    
- **Feature Coefficient Significance**
    
    In linear regression, each coefficient has a t-statistic. A small two-tailed p-value for βₖ indicates that feature *k* has a statistically significant effect.
    
- **A/B Test Lift**
    
    Compare conversion rates of two website variants with a two-sample z-test. The p-value shows whether observed lift is unlikely under no-difference (H₀: p₁ = p₂).
    

### 5. Geometric & Visual Intuition

1. **Draw the null distribution curve** (e.g., a bell curve centered at zero).
2. **Mark your observed_T** on the horizontal axis.
3. **Shade the tail area(s)**:
    - Right-tailed → shade to the right of observed_T
    - Left-tailed → shade to the left
    - Two-tailed → shade both tails beyond ±|observed_T|
4. The **shaded area** is the p-value—smaller area means more surprising result.

### 6. Practice Problems & Python Exercises

### 6.1 Coin-Flip Two-Tailed Test

You flip a coin 100 times and get 60 heads. Test H₀: p=0.5 vs H₁: p≠0.5 at α=0.05.

```python
import numpy as np
from scipy.stats import norm

# Data
n, x = 100, 60
p0 = 0.5

# 1. Compute p_hat and z-statistic
p_hat = x / n
se = np.sqrt(p0 * (1 - p0) / n)
z = (p_hat - p0) / se

# 2. Two-tailed p-value
p_value = 2 * (1 - norm.cdf(abs(z)))

print(f"z = {z:.3f}, p-value = {p_value:.3f}")
```

### 6.2 One-Sample t-Test on Model Error

You run 10-fold CV and get these errors:

`[0.23, 0.20, 0.22, 0.24, 0.21, 0.19, 0.25, 0.20, 0.23, 0.22]`

Test if mean error < 0.22 at α=0.05.

```python
import numpy as np
from scipy.stats import t

# Data
errors = np.array([0.23,0.20,0.22,0.24,0.21,0.19,0.25,0.20,0.23,0.22])
n = errors.size
mu0 = 0.22

# 1. Compute t-statistic
x_bar = errors.mean()
s = errors.std(ddof=1)
se = s / np.sqrt(n)
t_stat = (x_bar - mu0) / se

# 2. One-tailed p-value
p_value = t.cdf(t_stat, df=n-1)

print(f"t = {t_stat:.3f}, one-tailed p-value = {p_value:.3f}")
```

### 6.3 Two-Sample z-Test for Proportions

Group A: 120/400 click; Group B: 100/380 click. Test H₀: p₁ = p₂ vs H₁: p₁ > p₂.

```python
import numpy as np
from scipy.stats import norm

# Data
x1, n1 = 120, 400
x2, n2 = 100, 380

p1, p2 = x1/n1, x2/n2
p_pool = (x1 + x2) / (n1 + n2)
se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))

# z-statistic and one-tailed p-value
z = (p1 - p2) / se
p_value = 1 - norm.cdf(z)

print(f"z = {z:.3f}, one-tailed p-value = {p_value:.3f}")
```

### 7. Key Takeaways

- A p-value is the **probability of obtaining data as extreme (or more) as observed**, under H₀.
- **Small p-value** (≤ α) → data unlikely under H₀ → **reject H₀**.
- **Large p-value** (> α) → data consistent with H₀ → **fail to reject H₀**.
- It **does not** give the probability that H₀ is true or false.

---

## Critical Values

### 1. Prerequisite Reminder

Before we dive in, you should know:

- How to compute and interpret **test statistics** (e.g., z-score, t-statistic).
- The concept of a **reference distribution** (Standard Normal, Student’s t, χ², F).
- What a **significance level** (α) means in hypothesis testing or confidence intervals.

### 2. What Is a Critical Value?

A **critical value** is the cutoff point on your test statistic’s scale that separates the “fail to reject H₀” region from the “reject H₀” region.

- For a given α (e.g., 0.05), the critical value marks the tail area(s) of size α (one-sided) or α/2 (two-sided).
- Observing a test statistic more extreme than the critical value means your data are sufficiently surprising under H₀ — you reject H₀.

Analogy:

- Imagine a highway speed limit sign: any speed **above** the limit is illegal (reject H₀), any speed **at or below** is legal (fail to reject H₀). The posted speed is your critical value.

### 3. Generic Formula

For a test statistic T with cumulative distribution function (CDF) F₀ under H₀:

- **Right-tailed** at level α:
    
    ```markdown
    critical_value = F₀⁻¹(1 − α)
    ```
    
- **Left-tailed** at level α:
    
    ```markdown
    critical_value = F₀⁻¹(α)
    ```
    
- **Two-tailed** at level α:
    
    ```markdown
    critical_value = F₀⁻¹(1 − α/2)
    ±  F₀⁻¹(α/2)
    ```
    

Here F₀⁻¹ is the **quantile function** (inverse CDF) of the reference distribution.

### 4. Standard Normal (z) Critical Values

When T ~ Normal(0,1):

- Right-tailed α:
    
    ```markdown
    z_{α} = Φ⁻¹(1 − α)
    ```
    
- Two-tailed α:
    
    ```markdown
    ± z_{α/2} = ± Φ⁻¹(1 − α/2)
    ```
    

Common z-critical values:

| α (two-tailed) | z₍α/2₎ |
| --- | --- |
| 0.10 (90% CI) | 1.645 |
| 0.05 (95% CI) | 1.960 |
| 0.01 (99% CI) | 2.576 |

### 5. Student’s t Critical Values

When T ~ tₙ₋₁ (ν = n−1 degrees of freedom):

- Right-tailed α:
    
    ```markdown
    t_{ν,α} = t.ppf(1 − α, df=ν)
    ```
    
- Two-tailed α:
    
    ```markdown
    ± t_{ν,α/2} = ± t.ppf(1 − α/2, df=ν)
    ```
    

Because t-distributions have heavier tails for small ν, **t_{ν,α/2} > z_{α/2}** when n is small.

### 6. Critical Values for Other Tests

- **Chi-square (χ²) test** (goodness-of-fit, variance):
    
    ```markdown
    χ²_{df,α} = chi2.ppf(1 − α, df)
    ```
    
- **F-test** (variance ratio, ANOVA):
    
    ```markdown
    F_{df1,df2,α} = f.ppf(1 − α, df1, df2)
    ```
    

Each test plugs its own quantile function and degrees of freedom.

### 7. Step-by-Step: How to Compute a Critical Value

1. **Choose α** (e.g., 0.05) and tail type (one- or two-sided).
2. **Select reference distribution** under H₀ (Normal, t, χ², F).
3. **Determine degrees of freedom** (for t, χ², F).
4. **Compute quantile** with your statistical software or table:
    - Python: use `.ppf()` methods from `scipy.stats`.
    - R: use `qnorm()`, `qt()`, `qchisq()`, `qf()`.
5. **Use that critical value** in your decision rule: reject H₀ if your test statistic exceeds it (in absolute value for two-sided).

### 8. Real-World DS/ML Applications

- **A/B Testing**Determine z₀.₀₂₅ = 1.96 to build a 95% CI around difference in click rates.
- **Model Performance**Use t_{9,0.025} to test if 10-fold CV error differs significantly from a target.
- **Feature Selection**In linear regression, compare each coefficient’s t-statistic to t_{n−p,α/2} to decide inclusion.

### 9. Practice Problems & Python Code

### 9.1 Compute Z-Critical for Various α

```python
from scipy.stats import norm

for conf in [0.90, 0.95, 0.99]:
    alpha = 1 - conf
    z = norm.ppf(1 - alpha/2)  # two-sided
    print(f"{int(conf*100)}% two-sided z_crit = {z:.3f}")
```

### 9.2 Compute t-Critical for a Small Sample

```python
from scipy.stats import t

n = 12
df = n - 1
alpha = 0.05

t_two = t.ppf(1 - alpha/2, df)
t_right = t.ppf(1 - alpha, df)

print(f"t_{{11,0.025}} (two-sided) = {t_two:.3f}")
print(f"t_{{11,0.05}} (right-sided) = {t_right:.3f}")
```

### 9.3 Chi-Square Critical for Variance Test

```python
from scipy.stats import chi2

df = 15
alpha = 0.05

# For upper tail
chi2_right = chi2.ppf(1 - alpha, df)
# For two-sided variance CI, you need both tails
chi2_left  = chi2.ppf(alpha/2, df)

print(f"χ²_{{15,0.05}} = {chi2_right:.3f}")
print(f"χ²_{{15,0.025}} = {chi2_left:.3f}")
```

### 10. Visual & Geometric Intuition

- **Plot the reference PDF** (e.g., Normal).
- **Mark the critical value(s)** on the x-axis.
- **Shade the tail area(s)** corresponding to α or α/2.
- This visualization cements why values beyond that point are “unlikely” if H₀ is true.

### 11. Summary

- **Critical values** are quantiles of your test statistic’s null distribution that define rejection regions.
- They depend on α, tail type, and degrees of freedom (for t, χ², F).
- Compute them via quantile (ppf) functions in your software.

---

## Power of a Test and Interpreting Results

### Part 1: Power of a Test

### 1. Prerequisite Reminder

Before you dive into power, you should already understand:

- How to set up hypotheses (H₀, H₁).
- Type I error (α) and Type II error (β).
- Critical values and test statistics (z, t).
- p-values and decision rules for rejecting H₀.

### 2. Conceptual Overview

**Power** is the probability your test will **correctly detect** a real effect when the alternative hypothesis H₁ is true.

- It equals 1 − β, where β is the probability of a **Type II error** (failing to reject H₀ when H₁ is true).
- In everyday terms, power measures the sensitivity of your test.

Analogy:

Imagine a smoke alarm (your test) designed to detect fire (H₁).

- **Type I error (false alarm)**: the alarm rings when there’s no fire (α).
- **Type II error (missed fire)**: the alarm stays silent when there is a fire (β).
- **Power**: the alarm’s chance of ringing when there really is a fire (1 − β).

### 3. Key Formulas

### 3.1 Definition of Power

```markdown
Power = 1 − β
      = P(reject H₀ │ H₁ is true)
```

### 3.2 Power for a One-Sample z-Test (Known σ)

When testing H₀: μ = μ₀ vs H₁: μ = μ₁ > μ₀ at significance α, the power is:

```markdown
Power = 1 − Φ( z_crit − (μ₁ − μ₀)/(σ/√n) )
```

- `Φ(·)` is the standard Normal CDF.
- `z_crit = Φ⁻¹(1 − α)` is the right‐tail critical value.
- `(μ₁ − μ₀)/(σ/√n)` is the **standardized effect size**.

For a two‐sided test, replace `z_crit` with `z_{α/2}` and use both tails.

### 4. Step-by-Step Power Calculation

1. **Specify H₀ and H₁**, including a concrete alternative value μ₁ you care about (effect size).
2. **Choose α** (e.g., 0.05) → determine `z_crit` = Φ⁻¹(1−α) for right‐tailed or `z_{α/2}` for two‐tailed.
3. **Compute effect size**
    
    ```
    δ = (μ₁ − μ₀) / (σ/√n)
    ```
    
4. **Plug into formula**
    
    ```
    β = Φ( z_crit − δ )
    Power = 1 − β
    ```
    
5. **Interpret**: higher power means greater chance to detect that effect.

### 5. Geometric & Visual Intuition

1. Draw two Normal curves on the same axis:
    - **Under H₀**: centered at μ₀
    - **Under H₁**: centered at μ₁ (shifted right)
2. Shade the rejection region (area beyond `z_crit`) under the H₀ curve → that’s α.
3. Under the H₁ curve, shade the same region → that area is **power**.
4. The non‐shaded H₁ area to the left of the critical line is β (missed detections).

As μ₁ moves farther from μ₀, or as n increases (curves narrow), the shaded H₁ area (power) grows.

### 6. Real-World DS & ML Applications

- **A/B Testing**
    
    Plan how many users (n) you need to detect a minimum lift (μ₁ − μ₀) in conversion rate with 80% power at α=0.05.
    
- **Model Evaluation**
    
    When comparing two algorithms, compute the power to detect a 2% difference in cross-validation accuracy using paired t-tests.
    
- **Clinical Trials**
    
    Decide sample size so you have 90% power to detect a target mean improvement in a health metric.
    

### 7. Practice Problems & Python Code

### 7.1 Compute Power for a z-Test

You want 80% power to detect a mean increase from μ₀=100 to μ₁=105, σ=20, at α=0.05. How large n must be? Then verify power.

```python
import numpy as np
from scipy.stats import norm

# Given parameters
mu0, mu1, sigma = 100, 105, 20
alpha, target_power = 0.05, 0.80

# 1. z critical for right-tail
z_crit = norm.ppf(1 - alpha)

# 2. Solve for n: want 1 - Φ(z_crit - (mu1-mu0)/(sigma/√n)) = target_power
#    Let δ = (mu1 - mu0)/(sigma/√n) → δ = z_crit + z_power, where z_power = norm.ppf(target_power)
z_power = norm.ppf(target_power)
delta = z_crit + z_power

# 3. Compute n
n = ((sigma * delta) / (mu1 - mu0))**2
n = int(np.ceil(n))
print("Required n:", n)

# 4. Verify actual power with this n
delta_n = (mu1 - mu0) / (sigma / np.sqrt(n))
beta = norm.cdf(z_crit - delta_n)
power = 1 - beta
print("Achieved power:", power)
```

### 7.2 Simulate Power

Simulate many experiments to estimate empirical power:

```python
import numpy as np

def simulate_power(mu0, mu1, sigma, n, alpha, trials=10000):
    z_crit = norm.ppf(1 - alpha)
    rejections = 0
    for _ in range(trials):
        data = np.random.normal(mu1, sigma, size=n)
        z_stat = (data.mean() - mu0) / (sigma/np.sqrt(n))
        if z_stat > z_crit:
            rejections += 1
    return rejections / trials

emp_power = simulate_power(100, 105, 20, n, 0.05)
print("Empirical power:", emp_power)
```

## Part 2: Interpreting Results

### 1. Combining α, β, p-value, and Power

- **p-value ≤ α**
    
    You reject H₀. But with low power, you might still miss meaningful effects in future tests.
    
- **p-value > α**
    
    You fail to reject H₀. If power is low, that result may reflect an insensitive test rather than no effect.
    
- **High power (≥ 0.8)**
    
    A non‐significant result is stronger evidence that there truly is no effect of the size you specified.
    
- **Low power (< 0.5)**
    
    Even a large true effect might go undetected; non‐significant tests are inconclusive.
    

### 2. Interpreting a Complete Test Report

When you run an experiment or test, report:

1. **α (Type I rate)** you used (e.g., 0.05).
2. **n** (sample size) and **assumed effect size** for power calc.
3. **Observed test statistic** and **p-value**.
4. **Power** of the test for the effect size you care about.
5. **Conclusion** in context:
    - “We rejected H₀ at α=0.05 (p=0.02). With 80% power to detect a 5-unit increase, this suggests the true effect is likely ≥5.”
    - Or “We did not reject H₀ (p=0.12). However, with only 40% power to detect a 5-unit change, we cannot rule out an effect of that size.”

### 3. Common Pitfalls

- **Ignoring power** leads to overconfidence in negative results.
- **Interpreting p-value as power**: p-value does not tell you the probability the test will detect an effect in future.
- **Post-hoc power calculations** (using observed effect size) are often misleading—plan power **before** data collection.

---

## Student’s t-Distribution and t-Tests

### Part 1: Student’s t-Distribution

### 1. Motivation

When you estimate a population mean μ from a small sample (n < 30) and **σ is unknown**, replacing σ with the sample standard deviation `s` adds uncertainty. The ratio

```
T = (x̄ − μ) / (s / √n)
```

no longer follows a Normal curve but instead follows a **Student’s t-distribution** with ν = n−1 degrees of freedom.

That t-distribution has “fatter tails” than the Normal, widening confidence intervals and making hypothesis tests more conservative to account for extra variability.

### 2. Definition and PDF

The probability density function of the t-distribution with ν degrees of freedom is:

```markdown
f(t; ν) = Γ((ν + 1) / 2)
          ---------------------------------  *  (1 + t²/ν)^(-(ν + 1)/2)
          √(ν π) · Γ(ν / 2)
```

- `Γ(·)` is the Gamma function (generalizes factorial).
- ν = degrees of freedom; as ν → ∞, f(t; ν) → standard Normal PDF.

### 3. Key Properties

- **Mean:** 0 for ν > 1
- **Variance:** ν/(ν − 2) for ν > 2 (larger than 1 when ν small)
- **Shape:**
    - Heavier tails for small ν (more probability of extreme t).
    - Approaches the Normal curve as ν increases.

Visually, plot Normal and t₅, t₁₀, t₃₀ to see tails thinning.

### 4. When to Use

- **Confidence intervals** for μ when σ unknown and sample size small.
- **Hypothesis tests** (t-tests) for means under the same conditions.
- Whenever your statistic is standardized by s rather than σ.

## Part 2: t-Tests

t-tests use the t-distribution to test hypotheses about means. We cover:

1. One-sample t-test
2. Two-sample t-tests
3. Paired t-test

### 1. One-Sample t-Test

**Goal:** Test H₀: μ = μ₀ using a single sample of size n.

**Test statistic:**

```markdown
t = (x̄ − μ₀) / ( s / √n )
```

- `x̄` = sample mean
- `s` = sample standard deviation (ddof=1)
- `n` = sample size
- `ν = n − 1` degrees of freedom

**Decision:** Compare t to critical t_{ν,α/2} (two-sided) or compute p-value using tCDF.

### Steps

1. Compute `x̄`, `s`, and `t`.
2. Choose α (e.g., 0.05) and tail type.
3. Find `t_crit = t.ppf(1−α/2, df=ν)` (two-sided) or `t.ppf(1−α, df=ν)` (one-sided).
4. Reject H₀ if |t| > t_crit (two-sided) or t > t_crit (right-tailed) etc.
5. Or compute p-value:
    
    ```python
    from scipy.stats import t
    p_value_two = 2 * (1 - t.cdf(abs(t), df=ν))
    ```
    

### 2. Two-Sample t-Tests

### 2.1 Student’s t-Test (Equal Variances)

**Assumption:** The two groups have the same variance.

**Test statistic:**

```markdown
s_p² = [ (n₁−1)s₁² + (n₂−1)s₂² ] / (n₁ + n₂ − 2)

t = (x̄₁ − x̄₂)
    -----------------------
    s_p * √(1/n₁ + 1/n₂)

df = n₁ + n₂ − 2
```

- `x̄₁, x̄₂` = group means
- `s₁, s₂` = group sample standard deviations
- `n₁, n₂` = sample sizes

### 2.2 Welch’s t-Test (Unequal Variances)

**No equal‐variance assumption.**

```markdown
t = (x̄₁ − x̄₂)
    --------------------------
    √( s₁²/n₁ + s₂²/n₂ )

df ≈ [ (s₁²/n₁ + s₂²/n₂)² ]
     -------------------------
     [ (s₁⁴ / [n₁²(n₁−1)]) + (s₂⁴ / [n₂²(n₂−1)]) ]
```

Use this when variances differ or sample sizes are unequal.

### 3. Paired t-Test

When measurements come in pairs (before/after, matched subjects):

1. Compute differences `d_i = x_i(before) − x_i(after)`.
2. Let `d̄ = mean(d_i)` and `s_d = std(d_i)` with ddof=1.
3. Test statistic:

```markdown
t = d̄ / ( s_d / √n )
df = n − 1
```

This reduces to a one-sample test on differences.

### Part 3: Python Examples

```python
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

# One-sample t-test: is mean ≠ μ0?
data = np.random.randn(12)*2 + 5
t_stat, p_value = ttest_1samp(data, popmean=5)
print("One-sample:", t_stat, p_value)

# Two-sample (Welch by default)
group1 = np.random.randn(20) + 2
group2 = np.random.randn(15) + 2.5
t_stat2, p_value2 = ttest_ind(group1, group2, equal_var=False)
print("Welch two-sample:", t_stat2, p_value2)

# Paired t-test
before = np.random.randn(10)*1 + 100
after  = before + np.random.randn(10)*0.5 - 1
t_stat3, p_value3 = ttest_rel(before, after)
print("Paired:", t_stat3, p_value3)
```

### Part 4: Practice Problems

1. **One-Sample**
    
    A pilot study of n = 8 ML model run-times (seconds): `[1.2,1.5,1.3,1.4,1.6,1.1,1.3,1.2]`. Test if the true mean ≠ 1.0 at α=0.05.
    
2. **Two-Sample Equal Variance**
    
    Compare feature extraction times of two implementations:
    
    - Impl A (n₁=12): times in ms
    - Impl B (n₂=10): times in msTest H₀: μ_A = μ_B vs H₁: μ_A < μ_B at α=0.10.
3. **Welch’s Test**
    
    Same data but sample variances differ—use Welch’s test instead and compare results.
    
4. **Paired Test**
    
    Measure AUC before and after hyperparameter tuning for 10 cross-val runs. Test H₀: mean change = 0.
    

### Part 5: Real-World DS/ML Applications

- **Algorithm Comparison**: Paired t-test on cross-validation scores to decide if a new model improves performance.
- **AB Testing Metrics**: Two-sample t-test on average session times.
- **Feature Drift Detection**: One-sample t-test on mean feature value deviations from a baseline.

---

## Hypothesis Tests for Proportions

### 1. Prerequisite Reminder

Before you dive in, make sure you’re comfortable with:

- Computing a **sample proportion** `p_hat = x / n`.
- The **normal approximation** to the binomial: for large `n`,`p_hat` ≈ Normal(`p`, `p(1–p)/n`).
- Building a **standard error** for a proportion:`SE = sqrt(p*(1–p)/n)` under the null or use `p_hat` when estimating.

### 2. When and Why to Test Proportions

You use proportion tests when your data are counts of “success/failure” trials and you want to decide if the true success rate `p` equals or differs from a benchmark, or if two groups have different rates.

Common ML/DS scenarios:

- A/B test click-through rates.
- Classifier accuracy vs. baseline.
- Feature adoption rate comparisons between cohorts.

### 3. One‐Sample z-Test for a Proportion

### 3.1 Hypotheses

```
H₀: p = p₀
H₁ (two-sided): p ≠ p₀
H₁ (right-tailed): p > p₀
H₁ (left-tailed): p < p₀
```

### 3.2 Test Statistic

Under H₀, the standard error uses `p₀`:

```markdown
z = ( p_hat − p₀ )
    -------------------------
    sqrt( p₀ * (1 − p₀) / n )
```

- `p_hat = x / n` is sample proportion.
- `p₀` is the hypothesized proportion under H₀.
- `n` is the number of trials.

### 3.3 Decision Rule

1. Choose significance level `α` (e.g., 0.05).
2. For two-sided: reject H₀ if`|z| > z_{α/2}`, where `z_{α/2}` = 1.96 for α=0.05.
3. For one-sided: reject H₀ if
    - Right-tailed: `z > z_{α}`
    - Left-tailed: `z < −z_{α}`

### 3.4 Example: A/B Test Click Rate

- Group A: `x = 120` clicks in `n = 400` impressions → `p_hat = 0.30`
- Test H₀: `p = 0.25` vs H₁: `p > 0.25` at α=0.05.

```python
import numpy as np
from scipy.stats import norm

x, n, p0 = 120, 400, 0.25
p_hat = x / n
se = np.sqrt(p0 * (1 - p0) / n)
z = (p_hat - p0) / se
p_value = 1 - norm.cdf(z)

print(f"z-statistic = {z:.3f}")
print(f"one-sided p-value = {p_value:.3f}")
```

- If `z > 1.645`, reject H₀ at 5% one-sided.

### 4. Two‐Sample z-Test for Proportions

Compare rates between two groups:

### 4.1 Hypotheses

```
H₀: p₁ = p₂
H₁ (two-sided): p₁ ≠ p₂
H₁ (right-tailed): p₁ > p₂
H₁ (left-tailed): p₁ < p₂
```

### 4.2 Pooled Standard Error

Under H₀, combine successes:

```
p_pool = ( x₁ + x₂ ) / ( n₁ + n₂ )
```

### 4.3 Test Statistic

```markdown
z = ( p_hat1 − p_hat2 )
    -----------------------------------
    sqrt( p_pool * (1 − p_pool) * (1/n₁ + 1/n₂) )
```

- `p_hat1 = x₁ / n₁`, `p_hat2 = x₂ / n₂`.
- Use the same rejection rules as one-sample.

### 4.4 Example: Variant Comparison

- Variant A: 90/200 → `p_hat1 = 0.45`
- Variant B: 100/250 → `p_hat2 = 0.40`
- H₀: `p1 = p2`, α=0.05, two-sided.

```python
import numpy as np
from scipy.stats import norm

x1, n1 = 90, 200
x2, n2 = 100, 250
p1, p2 = x1/n1, x2/n2
p_pool = (x1 + x2) / (n1 + n2)
se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
z = (p1 - p2) / se
p_value = 2 * (1 - norm.cdf(abs(z)))

print(f"z = {z:.3f}, two-sided p-value = {p_value:.3f}")
```

### 5. Geometric & Visual Intuition

1. **One‐Sample:** Plot Normal distribution of `p_hat` under H₀ (mean `p₀`, sd `sqrt(p₀(1−p₀)/n)`). Shade the tail(s) beyond observed `p_hat`.
2. **Two‐Sample:** Plot two Normal curves for `p_hat1 − p_hat2` under H₀ (mean 0, sd pooled). Shade tails around observed difference.

These shaded areas correspond to p-values.

### 6. Conditions & Alternatives

- **Normal Approximation Validity:** Requires `n p₀ ≥ 5` and `n (1−p₀) ≥ 5` for one-sample; similarly for both groups in two-sample.
- **Small Samples or Extreme p̂:** Use **Wilson** or **Clopper–Pearson** intervals/tests instead of z.
- **Unpooled Test (approx):** If you suspect H₀ is false, you can use separate SEs (rare for hypothesis testing).

### 7. Practice Problems

1. **One-Sample Two-Sided Test**
    
    You observe `x = 30` successes in `n = 100`. Test H₀: `p = 0.25` vs H₁: `p ≠ 0.25` at α=0.05.
    
    - Compute `z`, p-value, and decision.
2. **Two-Sample One-Sided Test**
    
    Group A: 150/500 → `pA = 0.30`.
    
    Group B: 130/450 → `pB = 0.289`.
    
    Test H₀: `pA = pB` vs H₁: `pA > pB` at α=0.01.
    
    - Compute pooled SE, `z`, p-value, and decision.
3. **Python Simulation of Type I Error**
    
    Simulate 10 000 experiments under H₀: `p = 0.2`, `n = 50`.
    
    - Count fraction of times you reject at α=0.05 using one-sample z-test. Should approximate 0.05.

---

## Two-Sample t-Tests

### 1. Prerequisite Reminder

You should already know how to compute a sample mean (`x̄`), sample standard deviation (`s`), and perform a one‐sample t-test using the formula

```
t = (x̄ − μ₀) / (s / √n)
```

Also be familiar with degrees of freedom (`df = n−1`) and how to look up or compute critical t-values.

### 2. Why Two-Sample t-Tests?

When you have **two independent groups** and want to compare their means, a two-sample t-test evaluates whether the observed difference between `x̄₁` and `x̄₂` could arise by chance under the null hypothesis H₀: μ₁ = μ₂.

There are two main versions:

- **Pooled t-test** (assumes equal variances)
- **Welch’s t-test** (allows unequal variances)

### 3. Pooled t-Test (Equal Variances)

### 3.1 Formula

```markdown
# Pooled variance
s_p² = [ (n₁−1)·s₁² + (n₂−1)·s₂² ] / (n₁ + n₂ − 2)

# Test statistic
t = ( x̄₁ − x̄₂ )
    --------------------------
    s_p · sqrt( 1/n₁ + 1/n₂ )

# Degrees of freedom
df = n₁ + n₂ − 2
```

### 3.2 Step-by-Step Explanation

- Compute each group’s sample variance (`s₁²`, `s₂²`) and sample sizes (`n₁`, `n₂`).
- Pool variances into `s_p²` to get a more stable estimate under H₀.
- Find `t` by dividing the mean difference by the pooled standard error `s_p·√(1/n₁+1/n₂)`.
- Compare `t` to `t_{df, α/2}` for a two-sided test or compute a p-value with `df`.

### 4. Welch’s t-Test (Unequal Variances)

### 4.1 Formula

```markdown
# Test statistic
t = ( x̄₁ − x̄₂ )
    ---------------------------
    sqrt( s₁²/n₁ + s₂²/n₂ )

# Approximate degrees of freedom
df ≈ [ (s₁²/n₁ + s₂²/n₂)² ]
     --------------------------------------------
     [ s₁⁴/(n₁²·(n₁−1)) + s₂⁴/(n₂²·(n₂−1)) ]
```

### 4.2 Explanation

- You avoid pooling and treat each variance separately.
- The denominator is the square root of the sum of each variance scaled by its sample size.
- Degrees of freedom (`df`) are adjusted downward to reflect increased uncertainty.
- Use this when group variances differ or sample sizes are unbalanced.

### 5. Geometric & Visual Intuition

- Draw two Normal curves under H₀ (centered at the same mean) but each with its own spread.
- The rejection region sits in the tails beyond ±t-critical for a two-sided test.
- The test statistic `t` is the observed standardized difference—you reject if it falls in those tails.

### 6. Real-World DS/ML Examples

- **Model Comparison**
    
    Compare mean cross-validation errors of two algorithms over k folds using a paired or independent t-test.
    
- **Performance Benchmarks**
    
    Test whether a new C++ feature extraction routine is faster than the Python version, comparing mean runtimes.
    
- **User Behavior**
    
    Evaluate if two user cohorts have different average session lengths after deploying a UI change.
    

### 7. Practice Problems & Python Exercises

### 7.1 Pooled t-Test Example

Data:

- Group A runtimes (ms): 120, 115, 130, 125 (n₁=4)
- Group B runtimes (ms): 140, 135, 150, 145 (n₂=4)Test H₀: μ₁ = μ₂ vs H₁: μ₁ < μ₂ at α=0.05.

```python
import numpy as np
from scipy.stats import t

# Data
A = np.array([120,115,130,125])
B = np.array([140,135,150,145])
n1, n2 = A.size, B.size

# Sample stats
x1, x2 = A.mean(), B.mean()
s1, s2 = A.std(ddof=1), B.std(ddof=1)

# Pooled variance
s_p2 = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
s_p = np.sqrt(s_p2)

# Test statistic
t_stat = (x1 - x2) / (s_p * np.sqrt(1/n1 + 1/n2))
df = n1 + n2 - 2

# One-sided p-value
p_value = t.cdf(t_stat, df)

print(f"t = {t_stat:.3f}, df = {df}, p-value = {p_value:.3f}")
```

### 7.2 Welch’s t-Test Example

Same data but assume unequal variances; test H₀: μ₁ = μ₂ vs H₁: μ₁ < μ₂ at α=0.05.

```python
import numpy as np
from scipy.stats import ttest_ind

# Data
A = np.array([120,115,130,125])
B = np.array([140,135,150,145])

# Welch’s test (default equal_var=False)
t_stat, p_value = ttest_ind(A, B, equal_var=False, alternative='less')

print(f"t = {t_stat:.3f}, p-value = {p_value:.3f}")

```

---

## Paired t-Test

### 1. When to Use a Paired t-Test

A paired t-test is appropriate when you have **matched or dependent samples**—for example:

- Measurements before and after a treatment on the same subjects.
- Two techniques applied to identical experimental units.
- Sibling or twin studies comparing traits.

You’re interested in whether the **mean difference** between paired observations differs from zero (or some value).

### 2. Hypotheses

Let each pair yield a difference (d_i = x_{i,1} - x_{i,2}). You test:

```markdown
H₀: μ_d = 0       # mean difference is zero
H₁: μ_d ≠ 0       # two-sided
# Or one-sided:
# H₁: μ_d > 0     # first measurement larger
# H₁: μ_d < 0     # first measurement smaller
```

- μ_d is the true mean of the differences.

### 3. Test Statistic Formula

Compute differences (d_1, d_2, ..., d_n). Then:

```markdown
d_bar = (1/n) * sum_{i=1 to n} d_i

s_d = sqrt( (1/(n-1)) * sum_{i=1 to n} (d_i - d_bar)^2 )

t = d_bar / ( s_d / sqrt(n) )

df = n - 1
```

- (d_{bar}) is the sample mean of differences.
- (s_d) is the sample standard deviation of differences.
- (n) is the number of pairs.

### 4. Step-by-Step Procedure

1. **Compute differences**: (d_i = x_{i,1} - x_{i,2}).
2. **Calculate**:
    - Mean of differences: (d_{bar}).
    - Standard deviation of differences: (s_d).
3. **Define hypotheses** around μ_d (0 if no change).
4. **Compute t-statistic**:[ t = \frac{d_{bar}}{s_d / \sqrt{n}} ]
5. **Determine df**: (n - 1).
6. **Find critical t** or **compute p-value**:
    
    ```python
    from scipy.stats import t
    p_value_two = 2 * (1 - t.cdf(abs(t_stat), df))
    ```
    
7. **Decision**: reject H₀ if p-value ≤ α or if |t| > t_crit.

### 5. Real-World Examples

- **A/B Test (Same Users)**: Measure click rate before and after UI change for each user.
- **Model Performance**: Compare model A vs model B on same cross-validation folds (paired error differences).
- **Medical Studies**: Pre- and post-treatment measurements for blood pressure on same patients.

### 6. Python Implementation

```python
import numpy as np
from scipy.stats import ttest_rel

# Example: before vs after study hours and test scores
before = np.array([78, 82, 85, 90, 88])
after  = np.array([80, 85, 87, 92, 90])

# Compute differences manually
diffs = before - after
n = diffs.size
d_bar = diffs.mean()
s_d = diffs.std(ddof=1)
t_stat = d_bar / (s_d / np.sqrt(n))
df = n - 1

# p-value (two-sided)
from scipy.stats import t
p_value = 2 * (1 - t.cdf(abs(t_stat), df))

print(f"d_bar = {d_bar:.2f}, s_d = {s_d:.2f}")
print(f"t_stat = {t_stat:.3f}, df = {df}, p-value = {p_value:.3f}")

# Or using built-in
t_stat2, p_value2 = ttest_rel(before, after)
print(f"Built-in t_stat = {t_stat2:.3f}, p-value = {p_value2:.3f}")
```

### 7. Practice Problems

1. **Pre/Post Training**
    
    Students’ performance scores before and after a training session (n=12). Test if the training improved scores.
    
2. **Algorithm Comparison**
    
    Error rates of two classifiers measured on same 20 data splits. Test if classifier A has lower mean error.
    
3. **Bootstrapped Paired Test**
    
    Use bootstrap sampling on paired differences to compute a confidence interval for μ_d and compare with the t-test result.
    

### 8. Visual Intuition

- Plot a **histogram** of the differences (d_i).
- Mark the mean (d_{bar}) and overlay a t-distribution centered at 0 scaled by (s_d / √n).
- The area beyond (±|t|) gives p-value.

---

## ML Application: A/B Testing

### 1. Introduction

A/B testing is a controlled experiment comparing two (or more) variants—A (control) and B (treatment)—to see which performs better on a key metric. In ML and data science, A/B tests guide decisions on:

- Model deployments (e.g., new recommendation algorithm)
- UI changes affecting click‐through or conversion rates
- Feature rollouts (e.g., new search ranking variant)

You randomly split users or sessions between variants, collect outcomes, and apply statistical tests to decide if one variant truly outperforms the other.

### 2. Formulating Hypotheses

1. **Define metric** (e.g., conversion rate, average session time).
2. **Null hypothesis (H₀):** no difference between A and B.
3. **Alternative hypothesis (H₁):** B is better (one‐sided) or different (two‐sided).

Example for conversion rate `p`:

```markdown
H₀: p_A = p_B
H₁: p_A < p_B   # one‐sided: B has higher conversion
```

### 3. Designing the Experiment

- **Randomization:** assign each user to A or B at random.
- **Sample size:** compute upfront to ensure sufficient power (e.g., 80%).
- **Duration:** run long enough to reach required `n` and account for seasonality.
- **Data integrity:** track unique users, exclude bots, handle missing data.

### 4. Metric Selection

- **Binary outcome (conversion/no‐conversion):** use proportion tests.
- **Continuous metric (time on page, revenue):** use t-tests on means.
- **Count data (clicks per user):** consider Poisson or negative‐binomial tests.

### 5. Two‐Sample z-Test for Proportions

### 5.1 Formula

```markdown
p_hat_A = x_A / n_A
p_hat_B = x_B / n_B
p_pool  = (x_A + x_B) / (n_A + n_B)

z = (p_hat_A − p_hat_B)
    ----------------------------
    sqrt( p_pool*(1 − p_pool)*(1/n_A + 1/n_B) )
```

- `x_A`, `x_B` = number of conversions in each group
- `n_A`, `n_B` = number of users exposed to each variant

Reject H₀ if `z > z_{α}` (one‐sided) or `|z| > z_{α/2}` (two‐sided).

### 6. Step‐By‐Step Calculation

1. Compute `p_hat_A`, `p_hat_B`, and `p_pool`.
2. Calculate standard error:
    
    ```
    SE = sqrt( p_pool*(1 − p_pool)*(1/n_A + 1/n_B) )
    ```
    
3. Compute test statistic `z`.
4. Find critical value `z_crit = norm.ppf(1 − α)` (one‐sided) or `norm.ppf(1 − α/2)` (two‐sided).
5. Compare and decide:
    - If `z > z_crit`, reject H₀ and conclude B is better.
    - Else, fail to reject H₀.

### 7. Confidence Interval for Difference in Proportions

```markdown
diff = p_hat_B - p_hat_A

CI_lower = diff - z_{α/2} * sqrt( p_hat_A*(1−p_hat_A)/n_A + p_hat_B*(1−p_hat_B)/n_B )
CI_upper = diff + z_{α/2} * sqrt( p_hat_A*(1−p_hat_A)/n_A + p_hat_B*(1−p_hat_B)/n_B )
```

A 95% CI that does **not** include zero indicates a significant difference at `α=0.05`.

### 8. Python Code Example

```python
import numpy as np
from scipy.stats import norm

# Simulated data
n_A, x_A = 10000, 1200   # 12% conversion in A
n_B, x_B = 10000, 1320   # 13.2% conversion in B

# 1. Proportions
pA, pB = x_A/n_A, x_B/n_B
p_pool = (x_A + x_B) / (n_A + n_B)

# 2. Standard error and z
se_pool = np.sqrt(p_pool*(1-p_pool)*(1/n_A + 1/n_B))
z_stat = (pB - pA) / se_pool

# 3. p-value (one-sided)
alpha = 0.05
p_value = 1 - norm.cdf(z_stat)

print(f"pA = {pA:.3f}, pB = {pB:.3f}")
print(f"z_stat = {z_stat:.3f}, one-sided p = {p_value:.4f}")

# 4. 95% CI for diff
z_crit = norm.ppf(1 - alpha/2)
se_diff = np.sqrt(pA*(1-pA)/n_A + pB*(1-pB)/n_B)
ci_lower = (pB - pA) - z_crit*se_diff
ci_upper = (pB - pA) + z_crit*se_diff

print(f"95% CI for (pB - pA): [{ci_lower:.4f}, {ci_upper:.4f}]")
```

### 9. Sample Size and Power Planning

### 9.1 Sample Size for Proportions

To detect a minimum lift `m` with power `1−β` at level `α`:

```markdown
n ≈ [ z_{α/2} * sqrt(2*p0*(1−p0)) + z_{β} * sqrt(p1*(1−p1)+p0*(1−p0)) ]²
      ---------------------------------------------------------------
                       m²
```

- `p0` = baseline rate, `p1 = p0 + m`
- `z_{β}` = norm.ppf(power)

### 9.2 Python Snippet

```python
from scipy.stats import norm
import numpy as np

def sample_size_ab(p0, lift, alpha=0.05, power=0.8):
    p1 = p0 + lift
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta  = norm.ppf(power)
    term1 = z_alpha * np.sqrt(2*p0*(1-p0))
    term2 = z_beta  * np.sqrt(p1*(1-p1) + p0*(1-p0))
    n = ((term1 + term2)**2) / (lift**2)
    return int(np.ceil(n))

print("Per-group n:", sample_size_ab(0.12, 0.02))
```

### 10. Visualization & Interpretation

- **Bar chart** of `p_hat_A` vs `p_hat_B` with error bars from CIs.
- **Distribution of difference** under H₀ vs observed difference line.
- **Power curve** showing how power varies with sample size or lift.

### 11. Practice Problems

1. You observe 80/2000 → A, 100/2000 → B. Test H₀: p_A = p_B vs H₁: p_A < p_B at α=0.05.
    - Compute z, p-value, CI, and conclusion.
2. Feature rollout: baseline CTR=0.15. You want to detect a 0.02 lift with 90% power.
    - Calculate per-group sample size.
3. Continuous metric: compare average session time (A vs B) with two‐sample t-test.
    - Simulate data with μ_A=300s, μ_B=320s, σ≈50s, n=100 per group.

---