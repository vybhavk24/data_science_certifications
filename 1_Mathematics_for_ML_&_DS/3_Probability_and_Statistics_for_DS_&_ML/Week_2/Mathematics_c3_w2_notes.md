# Mathematics_c3_w2

## Expected Value (Expectation)

### Intuitive Explanation

Expected value is the “center of mass” of a distribution—the long-run average outcome if you repeat the random process many times.

It answers: on average, what number should I expect for my random variable?

Imagine rolling a fair six-sided die. Although any one roll could be 1, 2, …, or 6, if you roll thousands of times, the average tends toward 3.5. That 3.5 is the die’s expected value.

### Prerequisites You Should Know

- Definition of a random variable (discrete vs. continuous)
- PMF for discrete distributions and PDF + CDF for continuous
- Basic summation and integral calculus

### Formal Definitions

### Discrete Random Variable

For X taking values in a countable set 𝒳 with PMF p_X(x):

```markdown
E[X] = ∑_{x ∈ 𝒳} x * p_X(x)
```

### Continuous Random Variable

For X with PDF f_X(x):

```markdown
E[X] = ∫_{−∞}^{∞} x * f_X(x) dx
```

### Step-by-Step Breakdown of the Formula

1. **List the possible values** x of X (discrete) or identify the support of X (continuous).
2. **Weight each value** by its probability (p_X(x) or f_X(x)dx).
3. **Sum or integrate** the product x · probability over all x.
4. The result is a single number—the long-run average.

### Real-World Examples in ML & DS

- **Expected accuracy**: if a classifier has P(correct)=0.8, over 1000 samples you expect 800 correct predictions.
- **Expected revenue**: an ad click has value $2 with click-through rate 0.05 ⇒ expected value per impression = 2 × 0.05 = $0.10.
- **Expected loss**: in regression, the mean squared error is the expectation of (prediction − true)².
- **Expected customer arrivals**: if λ=3 per hour (Poisson), you expect 3 arrivals.

### Visual & Geometric Intuition

- **Discrete**: picture bars at each x. Multiply each bar’s x-coordinate by its height, then find the weighted average of all bars.
- **Continuous**: picture the PDF curve. Multiply the curve at each x by x, then compute the “area under x·f(x)” curve.

### Practice Problems & Python Exercises

### 1. Die Roll Expectation

Compute E[X] for a fair six-sided die.

```python
# Analytic
values = range(1, 7)
prob = 1/6
E = sum(x * prob for x in values)
print("E[X] =", E)  # 3.5
```

### 2. Empirical Verification

Simulate die rolls and estimate the average.

```python
import random
trials = 100_000
rolls = [random.randint(1, 6) for _ in range(trials)]
print("Simulated E[X]:", sum(rolls) / trials)
```

### 3. Continuous Example: Uniform[2, 5]

Compute E[X] = (a + b) / 2 = 3.5 analytically and by simulation.

```python
import numpy as np

# Analytic
a, b = 2, 5
E_theo = (a + b) / 2

# Simulation
samples = np.random.uniform(a, b, size=100_000)
E_emp = samples.mean()

print("Theoretical E[X]:", E_theo)
print("Empirical   E[X]:", E_emp)
```

### 4. Expected Value for a Binomial

For X ∼ Binomial(n=10, p=0.3), E[X] = n · p = 3. Verify simulation.

```python
import numpy as np

n, p = 10, 0.3
# Analytical expectation
print("E[X] theoretical:", n * p)

# Simulation
samples = np.random.binomial(n, p, size=100_000)
print("E[X] empirical:  ", samples.mean())
```

### How Data Scientists Use Expectation Daily

- **Feature expectations**: compute mean of each feature to center data (feature normalization).
- **Loss functions**: define expected loss to train models (e.g., expected log-loss in classification).
- **Decision theory**: choose actions that maximize expected reward or minimize expected cost.
- **A/B testing**: compare expected conversion rates between variants.

---

## Other Measures of Central Tendency: Median and Mode

### Intuitive Explanation

The median is the “middle” value that splits a distribution into two equal halves: 50% of the data lie below it and 50% lie above it.

The mode is the value that occurs most frequently in your data or has the highest density in a continuous distribution.

These measures can be more robust than the mean (expected value) when your data are skewed or contain outliers.

### Prerequisites You Should Know

- Definition of a random variable and its distribution (PMF for discrete, PDF/CDF for continuous)
- Cumulative distribution function (CDF) for continuous variables
- Basic sorting and counting operations

### Formal Definitions

Median (continuous or discrete):

```
xₘ = F_X⁻¹(0.5)
```

- F_X is the CDF of X.
- xₘ is any value where P(X ≤ xₘ) ≥ 0.5 and P(X ≥ xₘ) ≥ 0.5.

Mode (discrete):

```
mode = argmax_{x ∈ 𝒳} p_X(x)
```

Mode (continuous):

```
mode = argmax_x f_X(x)
```

- p_X(x) is the PMF for discrete X.
- f_X(x) is the PDF for continuous X.
- argmax returns the x with the highest probability or density.

### Step-by-Step Breakdown

Median:

1. Sort your data values in ascending order.
2. If you have an odd number of points, the median is the middle element.
3. If you have an even number, the median is the average of the two middle elements.
4. For a distribution, find x such that F_X(x) = 0.5 (or the smallest x with CDF ≥ 0.5).

Mode:

1. Tally the frequency of each distinct value (discrete) or estimate the density curve (continuous).
2. Identify the value(s) with the highest frequency or density.
3. If multiple values tie, there are multiple modes.

### Real-World Examples in ML & DS

- House price analysis: median sale price avoids distortion by extreme luxury listings.
- Response time logs: median latency gives a robust central measure under heavy-tailed delays.
- Categorical features (e.g., browser types): mode indicates the most common category.
- Kernel density estimation: mode of a continuous feature shows the most likely value.

### Visual & Geometric Intuition

- **Median**: imagine shading up to xₘ under the PDF curve so half the total area lies on each side.
- **Mode**: locate the highest peak of the PDF curve (continuous) or the tallest bar in a histogram (discrete).

### Practice Problems & Python Exercises

### 1. Compute Median and Mode of a List

```python
import statistics

data = [2, 5, 1, 3, 2, 5, 4]
median_value = statistics.median(data)
mode_value  = statistics.mode(data)

print("Median:", median_value)
print("Mode:  ", mode_value)
```

### 2. Empirical Median and Mode on a DataFrame

```python
import pandas as pd
from scipy import stats

df = pd.DataFrame({
    "age": [23, 45, 23, 30, 23, 38, 45, 30]
})

median_age = df["age"].median()
mode_age   = df["age"].mode().iloc[0]

# Continuous mode via kernel density estimate
kde = stats.gaussian_kde(df["age"])
xs  = np.linspace(df["age"].min(), df["age"].max(), 200)
mode_cont = xs[np.argmax(kde(xs))]

print("Median age:", median_age)
print("Mode age (discrete):", mode_age)
print("Mode age (continuous KDE):", mode_cont)
```

### 3. Median of a Continuous Distribution

For X ∼ Exponential(λ=0.5), the median solves F_X(xₘ)=0.5 ⇒ 1−e^(−λxₘ)=0.5 ⇒ xₘ=(ln2)/λ.

```python
import math

lam = 0.5
median_theo = math.log(2) / lam
print("Median of Exponential(0.5):", median_theo)
```

### How Data Scientists Use Median and Mode

- Replace missing values with median or mode to reduce bias from outliers.
- Summarize skewed distributions (income, response times) with median instead of mean.
- Identify most common categories via mode for categorical feature encoding.
- Detect multi-modality in data (e.g., product ratings clustering around multiple peaks).

---

## Expected Value of a Function

### Intuitive Explanation

When you apply a function g(…) to a random variable X, you get a new random outcome g(X). Its **expected value**, E[g(X)], is the long-run average of those transformed outcomes.

Think of X as daily temperature in °C and g(X)=X² as “squared temperature.” E[g(X)] tells you the average of the squares of daily temperatures—useful when modeling energy (which often grows like temperature²).

### Prerequisites You Should Know

- Definition of **expectation** for random variables.
- PMF (for discrete) or PDF/CDF (for continuous) of X.
- Basic sums and integrals.

If these feel unclear, review **expected value** before continuing.

### Formal Definitions

### Discrete Random Variable

For X with PMF p_X(x):

```
E[g(X)] = ∑_{x ∈ 𝒳} g(x) * p_X(x)
```

- Sum over all possible values x of X.
- Weight g(x) by the probability p_X(x).

### Continuous Random Variable

For X with PDF f_X(x):

```
E[g(X)] = ∫_{−∞}^{∞} g(x) * f_X(x) dx
```

- Integrate g(x)·f_X(x) across the support of X.
- The area under g(x)·f_X(x) gives the average of g(X).

### Step-by-Step Breakdown

1. **Identify g(x)**: your function of interest (e.g., g(x)=x², g(x)=ln(x), g(x)=I{x>c}).
2. **Obtain distribution of X** (p_X or f_X).
3. **Multiply** g(x) by the probability weight p_X(x) or density f_X(x).
4. **Sum** over x (discrete) or **integrate** over x (continuous).
5. The result is E[g(X)], guaranteed to exist when the sum/integral converges.

### Real-World Examples

- **Variance**: Var(X)=E[(X−μ)²] = E[g(X)] with g(x)=(x−μ)².
- **Risk in finance**: g(x)=max(0, K−x) for option payoff; E[g(X)] gives expected payoff.
- **Log-loss**: g(p)=−ln(p) in classification; E[−ln(P(y|X))] is expected log-loss.
- **Utility theory**: g(x)=√x mapping wealth to utility; E[√X] is expected utility.

### Visual & Geometric Intuition

- **Discrete**: bars at each x weighted by p_X(x). Multiply each bar’s height by g(x) on the y-axis, then take the weighted average of those g(x) values.
- **Continuous**: plot f_X(x) as a curve. Form a new curve g(x)·f_X(x). The area under g(x)·f_X(x) is E[g(X)].

### Practice Problems & Python Exercises

### 1. Second Moment of a Die

Compute E[X²] for X ∈ {1,…,6}, fair die.

```python
# Analytic
p = 1/6
E_x2 = sum((k**2) * p for k in range(1,7))
print("E[X^2] =", E_x2)  # should be 91/6 ≈ 15.1667
```

### 2. Log-Return Expectation

Simulate X ∼ Normal(μ=0.01, σ=0.02), compute E[g(X)] with g(x)=e^x.

```python
import numpy as np

mu, sigma, trials = 0.01, 0.02, 100_0000
samples = np.random.normal(mu, sigma, size=trials)
E_expX = np.mean(np.exp(samples))
print("E[e^X] empirical:", E_expX)
print("Theoretical:", np.exp(mu + 0.5*sigma**2))
```

### 3. Indicator Function

Let X be Uniform(0,1). Compute P(X > 0.7) via E[g(X)] with g(x)=I{x>0.7}.

```python
import numpy as np

trials = 100_000
samples = np.random.rand(trials)
g = (samples > 0.7).astype(float)
print("E[I{X>0.7}] empirical:", g.mean())  # ~0.3
```

### 4. Continuous Integral

For X ∼ Exponential(λ=2), compute E[X²]:

```python
from scipy.stats import expon

lam = 2
# Theoretical E[X^2] = 2/λ^2 = 2/(2^2) = 0.5
E_x2 = expon.moment(2, scale=1/lam)
print("E[X^2]:", E_x2)
```

### How Data Scientists Use This

- **Feature transformations**: computing E[g(X)] to normalize or engineer new features.
- **Model evaluation**: expected loss E[L(Y, Ŷ)] guides training objectives.
- **Bayesian inference**: posterior expectations of parameters E[θ | data].
- **Metrics**: expected precision, recall, or F1 score under different thresholds.

---

## Sum of Expectations

### Intuitive Explanation

Expectation is like a “long-run average” of a random variable. The **linearity of expectation** says that the average outcome of a sum of random quantities equals the sum of their individual averages—no matter how they interact.

Imagine you roll two dice. On average each die shows 3.5, so the sum of the two dice averages 3.5 + 3.5 = 7.

### Formal Definition

For any two random variables X and Y (discrete or continuous):

```markdown
E[X + Y] = E[X] + E[Y]
```

More generally, for n random variables X₁, X₂, …, Xₙ:

```markdown
E[ ∑_{i=1}^n X_i ] = ∑_{i=1}^n E[X_i]
```

And for constants a and b:

```markdown
E[a X + b] = a E[X] + b
```

### Step-by-Step Explanation

1. **Start with the sum**: consider Z = X + Y.
2. **Discrete case**:
    - E[Z] = ∑_{x,y} (x + y) · P(X=x, Y=y)
    - Split the sum: ∑ₓ₍ₓ,ᵧ₎ x·P + ∑₍ₓ,ᵧ₎ y·P → E[X] + E[Y].
3. **Continuous case**:
    - E[X+Y] = ∬ (x+y) f_{X,Y}(x,y) dx dy
    - Split the integral into two integrals → E[X] + E[Y].
4. **No independence needed**: joint distribution P(X,Y) cancels out when you split sums or integrals.
5. **Extend to more variables** by induction or repeated application of the two-term case.

### Real-World Examples

- **Coin flips**: Let Xᵢ be 1 if flip i is heads (p=heads), 0 otherwise. For n flips,E[sum of heads] = ∑E[Xᵢ] = n·p.
- **Daily revenue**: If revenue from each of k products is a random variable Rᵢ, total daily revenue R = ∑Rᵢ, thenE[R] = ∑E[Rᵢ].
- **Sensor network**: each sensor reports noise Nᵢ with zero mean. Sum of noises ∑Nᵢ has expectation 0.

### Visual Intuition

Imagine two histograms side by side for X and Y. Their centers (means) lie at μ_X and μ_Y. If you “add” the histograms by sliding one onto the other, the center of the combined shape is at μ_X + μ_Y.

### Practice Problems & Python Exercises

### 1. Two Dice Sum

Compute E[X+Y] for two fair six-sided dice.

```python
import itertools

# All pairs of rolls
pairs = list(itertools.product(range(1,7), repeat=2))
E_sum = sum((x+y) for x,y in pairs) / len(pairs)
print("E[X+Y] analytic:", E_sum)  # should be 7.0
```

### 2. Empirical Simulation

```python
import random

trials = 100_000
total = 0
for _ in range(trials):
    total += random.randint(1,6) + random.randint(1,6)
print("Simulated E[X+Y]:", total / trials)
```

### 3. Linear Combination

Let X ∼ Uniform(0,1), Y ∼ Uniform(2,5). Compute E[3X − 2 + Y].

```python
import numpy as np

# Theoretical
E_X, E_Y = 0.5, (2+5)/2
E_linear = 3*E_X - 2 + E_Y
print("Analytic E:", E_linear)

# Simulation
samples_X = np.random.rand(100_000)
samples_Y = np.random.uniform(2,5,100_000)
values = 3*samples_X - 2 + samples_Y
print("Empirical E:", values.mean())
```

### How Data Scientists Use This Daily

- **Aggregating features**: when summing component contributions, compute total expected effect easily.
- **A/B tests**: combining expected conversions across multiple segments by summing segment-level expectations.
- **Simulation**: predicting overall outcomes by summing expected values of subcomponents.
- **Unbiased estimators**: many estimators rely on linearity to prove E[estimator] = true parameter.

---

## Variance

### Intuitive Explanation

Variance measures how “spread out” the values of a random variable are around its mean.

Think of three exam-score distributions, all with mean 70:

- Class A: scores tightly packed between 68–72 (low variance).
- Class B: scores from 50–90 (high variance).
- Class C: half the students score 70, half score 70 (zero variance).

Variance answers: on average, how far (squared) does each outcome deviate from the mean?

### Prerequisites You Should Know

- Expected value (mean) of a random variable E[X].
- PMF for discrete variables or PDF for continuous variables.
- Basic algebra: squaring and summing/integrating.

### Formal Definitions

### Discrete Random Variable

For X taking values x₁, x₂, … with PMF p_X(x):

```
Var(X) = E[(X − μ)²]
       = ∑_{i} (x_i − μ)² * p_X(x_i)
where μ = E[X] = ∑_{i} x_i * p_X(x_i)
```

### Continuous Random Variable

For X with PDF f_X(x):

```
Var(X) = E[(X − μ)²]
       = ∫_{−∞}^{∞} (x − μ)² * f_X(x) dx
where μ = E[X] = ∫_{−∞}^{∞} x * f_X(x) dx
```

### Step-by-Step Breakdown of the Discrete Formula

1. Compute the mean
    
    ```
    μ = ∑_i x_i * p_X(x_i)
    ```
    
2. For each possible value x_i:
    - Find its deviation from the mean: (x_i − μ).
    - Square that deviation: (x_i − μ)².
3. Weight each squared deviation by its probability p_X(x_i).
4. Sum over all i to get Var(X).

This “average of squared deviations” ensures larger deviations contribute more.

### Alternate “Shortcut” Formula

Sometimes we use:

```
Var(X) = E[X²] − (E[X])²
```

- Compute E[X²] = ∑ x_i² p_X(x_i) (or ∫ x² f_X(x) dx).
- Subtract the square of the mean.

### Properties

- Var(X) ≥ 0 always.
- If Y = aX + b (constants a,b), then
    
    ```
    Var(Y) = a² * Var(X)
    ```
    
- For independent X and Y:
    
    ```
    Var(X + Y) = Var(X) + Var(Y)
    ```
    

### Real-World Examples in ML & DS

- **Feature scaling**: you standardize features viawhere σ_feature = √Var(feature).
    
    ```
    z = (x − μ_feature) / σ_feature
    ```
    
- **PCA (Principal Component Analysis)**: identifies directions of maximum variance.
- **Bias–Variance tradeoff**: variance quantifies how much model predictions fluctuate around their average.
- **A/B testing**: variance in conversions helps compute confidence intervals for lift.

### Visual & Geometric Intuition

- Plot the PDF or PMF of X.
- Draw vertical lines at the mean μ.
- At each x, the squared distance to μ is the horizontal distance squared.
- Variance is the weighted “area under the curve” of those squared distances.

### Practice Problems & Python Exercises

### 1. Variance of a Fair Die

Compute Var(X) for X ∈ {1,…,6}, fair.

```python
import math

# PMF: each face has prob 1/6
xs = range(1, 7)
p = 1/6

# Mean
mu = sum(x * p for x in xs)

# Variance
var = sum((x - mu)**2 * p for x in xs)
print("Mean μ:", mu)           # 3.5
print("Variance Var(X):", var) # 35/12 ≈ 2.9167
```

### 2. Simulate Binomial Variance

X ∼ Binomial(n=10, p=0.4). Theoretical variance = n·p·(1−p) = 10×0.4×0.6 = 2.4. Verify by simulation:

```python
import numpy as np

n, p = 10, 0.4
trials = 200_000
samples = np.random.binomial(n, p, size=trials)

print("Empirical Var(X):", samples.var(ddof=0))  # ddof=0 for population
print("Theoretical Var(X):", n * p * (1-p))
```

### 3. Variance of Uniform[2,5]

Theoretical Var = (b−a)²/12 = (3)²/12 = 0.75. Simulate:

```python
import numpy as np

a, b = 2, 5
samples = np.random.uniform(a, b, size=100_0000)

print("Empirical Var(X):", samples.var(ddof=0))
print("Theoretical Var(X):", (b - a)**2 / 12)
```

### 4. Empirical vs. Shortcut Formula

For X from an exponential distribution λ=0.5, verify Var(X)=1/λ²=4:

```python
import numpy as np

lam = 0.5
samples = np.random.exponential(1/lam, size=200_000)

# Method 1: direct definition
mu = samples.mean()
var1 = ((samples - mu)**2).mean()

# Method 2: shortcut E[X²] - (E[X])²
E2 = (samples**2).mean()
var2 = E2 - mu**2

print("Var from def:", var1)
print("Var from shortcut:", var2)
print("Theoretical Var:", 1/lam**2)
```

### How Data Scientists Use Variance

- **Standardization**: scale features to zero mean and unit variance before modeling.
- **Feature selection**: remove low-variance features that carry little information.
- **Uncertainty quantification**: variance of predictions under Bayesian models.
- **Ensemble methods**: reducing model variance via bagging and boosting.

---

## Standard Deviation

### Intuitive Explanation

Standard deviation measures how much the values of a random variable typically deviate from its mean.

If most data points lie close to the mean, standard deviation is small; if they spread out widely, it’s larger.

It has the same units as the original data, making it easier to interpret than variance (which is in squared units).

### Formal Definitions

### Population Standard Deviation

For a random variable X with mean μ:

```markdown
σ = sqrt( Var(X) )
  = sqrt( E[ (X − μ)² ] )
```

- σ (sigma) denotes population standard deviation.

### Discrete Case

```markdown
σ = sqrt( ∑_{i} (x_i − μ)² * p_X(x_i) )
```

### Continuous Case

```markdown
σ = sqrt( ∫_{−∞}^{∞} (x − μ)² * f_X(x) dx )
```

### Sample Standard Deviation

When you have a sample of n observations x₁…xₙ with sample mean x̄:

```markdown
s = sqrt( (1/(n − 1)) * ∑_{i=1}^n (x_i − x̄)² )
```

- s uses n−1 in the denominator for an unbiased estimate of σ.

### Step-by-Step Breakdown

1. Compute the mean (μ or x̄).
2. For each value, find its deviation from the mean and square it.
3. Average those squared deviations (use 1/n for population or 1/(n−1) for a sample).
4. Take the square root of that average to return to the original units.

### Real-World Examples in ML & DS

- In regression, standard deviation of residuals indicates model fit and confidence intervals.
- Feature scaling (z-score): subtract the mean and divide by standard deviation for each feature.
- Evaluating volatility in time-series (e.g., asset returns) uses rolling standard deviation.
- In A/B testing, standard deviation of conversion rates helps compute margin of error.

### Visual Intuition

Plot a bell curve (Normal distribution).

– The distance from the center (mean) to the inflection points on either side equals one standard deviation.

– Approximately 68% of the area lies within ±1σ, 95% within ±2σ, 99.7% within ±3σ.

### Practice Problems & Python Exercises

### 1. Die Roll Standard Deviation

Compute σ for a fair six-sided die analytically and by simulation.

```python
import math, random

# Analytic
values = range(1, 7)
p = 1/6
mu = sum(x * p for x in values)
var = sum((x - mu)**2 * p for x in values)
sigma_analytic = math.sqrt(var)
print("Analytic σ:", sigma_analytic)  # ≈1.7078

# Simulation
trials = 200_000
rolls = [random.randint(1,6) for _ in range(trials)]
print("Empirical σ:", (sum((r - mu)**2 for r in rolls)/trials)**0.5)
```

### 2. Binomial Distribution

For X ∼ Binomial(n=20, p=0.3), theoretical σ = √(n·p·(1−p)). Verify by simulation.

```python
import numpy as np

n, p = 20, 0.3
sigma_theo = math.sqrt(n * p * (1 - p))
samples = np.random.binomial(n, p, size=100_000)
print("Empirical σ:", samples.std(ddof=0))
print("Theoretical σ:", sigma_theo)
```

### 3. Sample vs. Population σ

Generate 1,000 samples from Uniform(0,1). Compare s (ddof=1) vs. σ (ddof=0).

```python
import numpy as np

data = np.random.rand(1000)
print("Population σ:", data.std(ddof=0))
print("Sample   s:", data.std(ddof=1))
```

### 4. Rolling Standard Deviation

Compute a 7-day rolling σ of daily returns in a pandas Series `returns`.

```python
import pandas as pd

# assume returns is a pandas Series of daily returns
rolling_sigma = returns.rolling(window=7).std(ddof=0)
print(rolling_sigma.head(10))
```

### How Data Scientists Use Standard Deviation Daily

- Standardizing inputs (zero mean, unit variance) for gradient-based models.
- Quantifying volatility in finance and algorithmic trading strategies.
- Defining control limits in statistical process control (±3σ rule).
- Detecting outliers as points beyond a certain number of σ from the mean.

---

## Sum of Gaussian Random Variables

### Intuitive Explanation

When you add two independent Gaussian (normal) random variables, the result is itself Gaussian.

Each variable brings its own “bell curve” of uncertainty, and summing them simply shifts the center and widens the spread.

Imagine measuring the same length with two different instruments:

- Instrument A has error ∼N(0, 1 mm²)
- Instrument B has error ∼N(0, 4 mm²)Adding their readings means the total error is a new normal distribution with variance 1 + 4.

### Formal Definition

Let

```
X ∼ N(μ₁, σ₁²)
Y ∼ N(μ₂, σ₂²)
```

be independent. Define

```
Z = X + Y
```

Then

```
Z ∼ N(μ₁ + μ₂,  σ₁² + σ₂²)
```

More generally, for n independent Gaussians Xᵢ∼N(μᵢ, σᵢ²):

```
S = ∑_{i=1}^n X_i
S ∼ N( ∑ μ_i,  ∑ σ_i² )
```

### PDF of the Sum

The PDF of Z is:

```
f_Z(z) = 1
       —————————————
       √[2π (σ₁²+σ₂²)]

       * exp( − (z − (μ₁+μ₂))²
               ———————————————
               2 (σ₁²+σ₂²)         )
```

- The new mean is μ₁+μ₂.
- The new variance is σ₁²+σ₂².

### Step-by-Step Breakdown

1. **Mean addition**: the center of Z sits at μ₁+μ₂ because expectations add:
    
    ```
    E[Z] = E[X] + E[Y] = μ₁ + μ₂
    ```
    
2. **Variance addition**: since X and Y are independent, their variances sum:
    
    ```
    Var(Z) = Var(X) + Var(Y) = σ₁² + σ₂²
    ```
    
3. **Normalization constant**: 1/√[2π(σ₁²+σ₂²)] makes the total area = 1.
4. **Exponent**: the squared deviation (z−(μ₁+μ₂))² is scaled by twice the new variance.

### Real-World Examples

- **Sensor fusion**: combining two noisy measurements yields a single Gaussian error.
- **Ensemble predictions**: summing independent model outputs (with Gaussian errors) remains Gaussian.
- **Brownian motion**: increments over non‐overlapping time intervals are independent normals; the total displacement over those intervals is Gaussian with summed variances.

### Visual & Geometric Intuition

- Plot two bell curves for N(μ₁, σ₁²) and N(μ₂, σ₂²).
- Their convolution (area‐wise sliding sum) yields a new bell curve.
- The new peak is at μ₁+μ₂, and the width (standard deviation) is √(σ₁²+σ₂²).

### Practice Problems & Python Exercises

### 1. Sum of Two Normals

Simulate X∼N(2, 1²) and Y∼N(−1, 2²), then plot the histogram of Z=X+Y against its analytic PDF.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu1, sigma1 = 2, 1
mu2, sigma2 = -1, 2
mu_z, sigma_z = mu1+mu2, np.sqrt(sigma1**2 + sigma2**2)

# Simulate
trials = 200_000
X = np.random.normal(mu1, sigma1, size=trials)
Y = np.random.normal(mu2, sigma2, size=trials)
Z = X + Y

# Plot histogram
plt.hist(Z, bins=100, density=True, alpha=0.6, label="Simulated")

# Plot analytic PDF
xs = np.linspace(mu_z-4*sigma_z, mu_z+4*sigma_z, 400)
plt.plot(xs, norm.pdf(xs, mu_z, sigma_z), 'r--', label="Analytic PDF")
plt.title("Sum of two Gaussians")
plt.xlabel("z"); plt.ylabel("Density")
plt.legend(); plt.show()
```

### 2. Sum of n Independent Gaussians

Verify that summing n standard normals yields N(0, n).

```python
import numpy as np

n = 5
trials = 100_000
S = np.sum(np.random.normal(0,1,(trials, n)), axis=1)

print("Empirical mean:", S.mean())
print("Empirical var: ", S.var(ddof=0), "≈", n)
```

### How Data Scientists Use This Rule

- **Error propagation**: combining uncertainties from multiple independent sources.
- **Model stacking**: when blending independent predictions, the aggregate error stays Gaussian.
- **Kalman filters**: predict‐update cycles rely on Gaussian sums for state estimation.
- **Signal processing**: noise accumulation remains Gaussian, simplifying analysis.

---

## Standardizing a Distribution

### Intuitive Explanation

Standardizing transforms any random variable X into a new variable Z that has mean 0 and standard deviation 1.

Imagine you have two exam scores—one out of 50 and one out of 200. Directly comparing them is unfair. By subtracting each score’s class average (mean) and dividing by its class standard deviation, you get “z-scores” on a common scale: how many standard deviations above or below each student is, regardless of the original scale.

### Formal Definition

Given a random variable X with mean μ and standard deviation σ, the standardized variable Z is:

```markdown
Z = (X − μ) / σ
```

- Z has E[Z] = 0 and Var(Z) = 1.
- Any value of X maps linearly to Z.

### Step-by-Step Breakdown of the Formula

1. **Compute μ = E[X]**Estimate the average of X across your data or distribution.
2. **Compute σ = √Var(X)**Measure the spread of X around μ.
3. **Subtract the mean**Form Y = X − μ so that Y has mean 0.
4. **Divide by σ**Z = Y / σ rescales Y so its standard deviation equals 1.

### Real-World Examples in ML & DS

- **Feature scaling**: Algorithms like k-NN or SVM perform better when features have zero mean and unit variance.
- **Comparing metrics**: Convert different KPIs (e.g., click rate vs. time on site) to z-scores to see which deviates most from its average.
- **Anomaly detection**: Points with |z| > 3 are often flagged as outliers—more than three standard deviations from the norm.
- **Hypothesis testing**: Test statistics (t-scores, z-scores) use standardization to reference tables of standard normal probabilities.

### Visual Intuition

1. Plot the original distribution’s histogram; note its center (μ) and spread (σ).
2. After standardizing, the histogram shifts so its peak lies at zero and its width corresponds to one unit (one σ).
3. Overlay the standard normal curve (bell centered at 0, σ=1) to see alignment.

### Practice Problems & Python Exercises

### 1. Standardize a Simulated Dataset

```python
import numpy as np

# Simulate X ~ Normal(µ=10, σ=2)
np.random.seed(0)
X = np.random.normal(10, 2, size=100_000)

# Compute mean and std
mu, sigma = X.mean(), X.std(ddof=0)

# Standardize
Z = (X - mu) / sigma

# Verify
print("Mean of Z:", Z.mean())
print("Std of Z: ", Z.std(ddof=0))
```

### 2. Feature Standardization in pandas

```python
import pandas as pd

df = pd.DataFrame({
    "height_cm": np.random.normal(170, 10, 1000),
    "weight_kg": np.random.normal(65, 15, 1000)
})

# Compute means and stds
means  = df.mean()
stds   = df.std(ddof=0)

# Standardize each column
df_std = (df - means) / stds
print(df_std.describe().loc[["mean","std"]])
```

### 3. Z-Score Outlier Detection

```python
import numpy as np

data = np.random.exponential(1, size=1000)
mu, sigma = data.mean(), data.std(ddof=0)
z_scores = (data - mu) / sigma

# Flag outliers beyond ±3σ
outliers = data[np.abs(z_scores) > 3]
print("Number of outliers:", len(outliers))
```

### How Data Scientists Use Standardization Daily

- Preprocessing pipelines: apply `StandardScaler` (scikit-learn) to train and test sets.
- Model interpretability: coefficients correspond to change in output per one standard deviation change in feature.
- Regularization: penalized models (Lasso, Ridge) converge faster when features share scale.
- Control charts: monitor process stability by tracking z-scores over time.

---

## Skewness

### Intuitive Explanation

Skewness measures the asymmetry of a distribution’s shape around its mean.

- A distribution with a long right tail is **positively skewed** (right‐skewed).
- A distribution with a long left tail is **negatively skewed** (left‐skewed).
- A perfectly symmetric distribution (like the Normal) has zero skewness.

### Formal Definition

Let X be a random variable with mean μ and standard deviation σ.

### Population Skewness

```
γ₁ = E\bigl[\bigl(\tfrac{X - μ}{σ}\bigr)^3\bigr]
```

- γ₁ (gamma one) is the third standardized moment.

### Sample Skewness

For a sample {x₁,…,xₙ} with mean (\bar x) and sample standard deviation s:

```
g₁ = \frac{n}{(n-1)(n-2)}
      \sum_{i=1}^n \bigl(\tfrac{x_i - \bar x}{s}\bigr)^3
```

- This corrects bias in small samples.

### Calculation Steps

1. Compute the mean:
    
    (\bar x = \tfrac{1}{n}\sum_i x_i)
    
2. Compute the standard deviation:
    
    (s = \sqrt{\tfrac{1}{n-1}\sum_i (x_i - \bar x)^2})
    
3. Compute the third moment:
    
    (\sum_i (x_i - \bar x)^3)
    
4. Plug into the sample skewness formula to get g₁.

## Interpretation

| Skewness (g₁ or γ₁) | Shape Description |
| --- | --- |
| < 0 | Left‐skewed (tail to the left) |
| ≈ 0 | Symmetric |
| > 0 | Right‐skewed (tail to the right) |

### Real-World Examples

- **Income distribution**: often right‐skewed—few individuals earn very high incomes.
- **Response times**: left‐skewed when there’s a minimum physiological limit.
- **Customer ratings**: may show skewness if most reviews cluster high but some very low.

### Visual Intuition

1. Draw a histogram or density plot.
2. Locate the tail that stretches further from the bulk of the data.
3. If the tail is on the right, skewness is positive; if on the left, negative.

### Practice Problems & Python Exercises

### 1. Compute Sample Skewness

```python
import numpy as np
from scipy.stats import skew

data = np.random.exponential(scale=2, size=1000)
# SciPy’s skew uses bias‐corrected g1 by default
print("Sample skewness:", skew(data, bias=False))
```

### 2. Simulate Known Skew

Generate right-skewed and left-skewed samples:

```python
# Right‐skewed: exponential
exp_data = np.random.exponential(1, size=10000)
print("Exponential skew:", skew(exp_data, bias=False))

# Left‐skewed: negative exponential
neg_exp = -exp_data
print("Negative‐exp skew:", skew(neg_exp, bias=False))
```

### 3. Visual Comparison

```python
import matplotlib.pyplot as plt
plt.hist(exp_data, bins=50, alpha=0.6, label="Right‐skewed")
plt.hist(neg_exp, bins=50, alpha=0.6, label="Left‐skewed")
plt.legend(); plt.show()
```

### How Data Scientists Use Skewness

- Detecting departure from normality before applying statistical tests.
- Deciding on transformations (log, square-root) to reduce skewness.
- Feature engineering: creating skewness-aware metrics for anomaly detection.
- Model diagnostics: checking residual skewness in regression models.

---

## Kurtosis

### Intuitive Explanation

Kurtosis measures how “tailed” or “peaked” a distribution is compared to a Normal distribution.

- A **leptokurtic** distribution has heavy tails and a sharp peak, indicating more frequent extreme values.
- A **platykurtic** distribution has light tails and a flatter peak, indicating fewer extremes.
- A **mesokurtic** distribution has the same tail weight as a Normal (kurtosis ≈ 3).

### Prerequisites You Should Know

- Definition of mean (μ) and standard deviation (σ)
- Concept of moments (especially fourth moment)
- CDF/PDF or PMF of your random variable
- Sample vs. population statistics

### Formal Definitions

**Population (raw) kurtosis**

[ \kappa = E\Bigl[\bigl(\tfrac{X - μ}{σ}\bigr)^4\Bigr] ]

**Excess kurtosis**

[ \gamma_2 = \kappa - 3 ]

- Subtracting 3 makes the Normal distribution have zero excess kurtosis.

**Sample kurtosis** (bias-corrected)

[ g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^n \Bigl(\tfrac{x_i - \bar{x}}{s}\Bigr)^4 ;-; \frac{3(n-1)^2}{(n-2)(n-3)} ]

- (n) is the sample size, (\bar{x}) the sample mean, and (s) the sample standard deviation.

### Step-by-Step Breakdown

1. Compute the mean: (\bar{x}) (or μ for population).
2. Compute the standard deviation: (s) (or σ).
3. Calculate each standardized deviation to the fourth power: (\bigl(\tfrac{x_i - \bar{x}}{s}\bigr)^4).
4. Sum these values and apply the sample-kurtosis formula (or integrate/sum and divide by n for population).
5. Subtract 3 if you want excess kurtosis, centering the Normal at zero.

### Types of Kurtosis

| Type | Excess Kurtosis γ₂ | Shape Characteristics |
| --- | --- | --- |
| Platykurtic | γ₂ < 0 | Flatter peak, light tails (e.g., Uniform) |
| Mesokurtic | γ₂ ≈ 0 | Moderate peak and tails (e.g., Normal) |
| Leptokurtic | γ₂ > 0 | Sharp peak, heavy tails (e.g., t-distribution with low df) |

### Real-World Examples

- **Financial returns** often exhibit positive excess kurtosis (fat tails), signaling higher crash risk.
- **Uniform noise** is platykurtic—extremes are rarer than in a Normal.
- **Student’s t-distribution** with small degrees of freedom is leptokurtic, modeling data prone to outliers.

### Visual & Geometric Intuition

- Plot density curves of a Uniform, Normal, and heavy-tailed distribution.
- Observe that heavy tails place more area far from the mean, while light tails concentrate area closer in.
- The fourth-power weighting amplifies deviations far from the center, making tails dominate the kurtosis measure.

### Practice Problems & Python Exercises

1. **Analytic Uniform Kurtosis**
    
    For X ∼ Uniform(a, b):
    
    [ \kappa = \frac{(b - a)^4}{80,\sigma^4} + 3;=;1.8, ] giving excess kurtosis γ₂ = −1.2. Verify by integration.
    
2. **Simulate Exponential Kurtosis**
    
    ```python
    import numpy as np
    from scipy.stats import kurtosis
    
    samples = np.random.exponential(scale=1.0, size=200_000)
    # Fisher’s definition: subtract 3 for excess kurtosis
    print("Excess kurtosis:", kurtosis(samples, bias=False))
    ```
    
3. **Compare Heavy-Tailed Distributions**
    
    ```python
    import numpy as np
    from scipy.stats import kurtosis
    
    n = 200_000
    normals = np.random.normal(size=n)
    t5      = np.random.standard_t(df=5, size=n)
    
    print("Normal γ₂:", kurtosis(normals, bias=False))
    print("t(5df)   γ₂:", kurtosis(t5, bias=False))
    ```
    

### How Data Scientists Use Kurtosis

- **Risk management**: flag distributions with high kurtosis as prone to extreme events.
- **Feature engineering**: detect non-Gaussian features and apply transformations (log, Box–Cox).
- **Model diagnostics**: check residual kurtosis in regression to validate Normal error assumptions.
- **Anomaly detection**: use kurtosis in rolling windows to spot shifts toward heavier tails.

---

## Quantiles and Box-Plots

### Intuitive Explanation

Quantiles partition a distribution into intervals with equal probability mass.

For example, the 0.5-quantile (median) splits data into two halves: 50% below and 50% above.

Box-plots visualize these key quantiles and extremes in a compact summary of a distribution’s shape and outliers.

### Formal Definitions

A random sample of size n sorted as x₍₁₎ ≤ x₍₂₎ ≤ … ≤ x₍ₙ₎ has quantiles defined by positions in the sorted list.

- The **p-quantile** Q(p) is a value such that at least p·100% of the data lie at or below Q(p).
- Common special cases:
    - Q(0.25): first quartile (Q₁)
    - Q(0.50): median (Q₂)
    - Q(0.75): third quartile (Q₃)

Different software interpolate between neighboring values when p·(n+1) is not an integer.

### Types of Quantiles

| Name | Notation | p-value | Interpretation |
| --- | --- | --- | --- |
| Minimum | Q(0) | 0 | Smallest observation |
| Lower quartile | Q(0.25) | 0.25 | 25% of data ≤ Q₁ |
| Median | Q(0.50) | 0.5 | Middle of the data |
| Upper quartile | Q(0.75) | 0.75 | 75% of data ≤ Q₃ |
| Maximum | Q(1) | 1 | Largest observation |

### Computing Quantiles (Step-by-Step)

1. Sort your data into ascending order.
2. Compute the target index i = p·(n + 1).
3. If i is an integer, set Q(p) = x₍ᵢ₎.
4. If i is not an integer, let k = ⌊i⌋ and d = i − k. Interpolate:Q(p) = (1 − d)·x₍ₖ₎ + d·x₍ₖ₊₁₎.
5. Edge cases: for p = 0 or p = 1, Q(p) equals the minimum or maximum.

### Box-Plot Construction

A box-plot (or box-and-whisker plot) graphically summarizes the five-number summary and potential outliers:

1. **Box** spans from Q₁ to Q₃.
2. **Median** line at Q₂ inside the box.
3. **Whiskers** extend to the most extreme values within 1.5·IQR of the quartiles, whereIQR = Q₃ − Q₁.
4. **Outliers** are points beyond the whiskers, plotted individually.

### Interpretation of a Box-Plot

- The **height of the box** (IQR) shows the bulk spread of the middle 50% of data.
- **Whisker length** indicates variability outside the central 50%.
- **Skewness** is visible if the median is off-center in the box or whiskers are unequal.
- **Outliers** flag unusually large or small observations for further investigation.

### Practice Problems & Python Exercises

### 1. Compute Quantiles by Hand

Given the data [3,7,8,5,12,14,21,13,18], sort it and compute Q₁, Q₂, Q₃ using interpolation.

### 2. Quantiles Using NumPy and pandas

```python
import numpy as np
import pandas as pd

data = np.array([3,7,8,5,12,14,21,13,18])
# NumPy quantiles
q_np = np.quantile(data, [0, 0.25, 0.5, 0.75, 1.0])
print("NumPy quantiles:", q_np)

# pandas Series
s = pd.Series(data)
print("pandas quantiles:\n", s.quantile([0, 0.25, 0.5, 0.75, 1.0]))
```

### 3. Draw a Box-Plot

```python
import matplotlib.pyplot as plt

plt.boxplot(data, vert=False, patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
plt.title("Box-Plot of Sample Data")
plt.xlabel("Value")
plt.show()
```

### 4. Compare Multiple Groups

Generate two groups: one Normal(0,1), another right-skewed (Exponential(1)), and plot side-by-side box-plots.

```python
import numpy as np
import matplotlib.pyplot as plt

group1 = np.random.normal(0,1, size=200)
group2 = np.random.exponential(1, size=200)

plt.boxplot([group1, group2], labels=['Normal','Exp'])
plt.ylabel("Value")
plt.title("Box-Plots Comparison")
plt.show()
```

### How Data Scientists Use Quantiles & Box-Plots

- **Outlier detection**: isolate data points beyond 1.5·IQR.
- **Feature transformation**: apply quantile normalization to map data to a uniform or normal distribution.
- **Summary statistics**: communicate distribution shape in reports and dashboards.
- **Quality control**: monitor metrics over time using box-plots to spot shifts or anomalies.

---

## Visualizing Data: KDE, Violin Plots, and QQ Plots

## Kernel Density Estimation (KDE)

### Intuitive Explanation

Kernel density estimation creates a smooth curve that approximates the underlying probability density of a continuous variable.

Instead of grouping data into bins (as in a histogram), KDE places a small “bump” (kernel) at each data point and sums them. The result is a continuous curve showing where observations cluster and where they thin out.

### Formal Definition

Given data points x₁, x₂, …, xₙ and a kernel function K (usually Gaussian) with bandwidth h:

```
f̂(x) = (1 / (n h))
       · ∑_{i=1}ⁿ K( (x − xᵢ) / h )
```

- K is a symmetric, nonnegative function integrating to 1.
- The bandwidth h controls smoothness: small h yields a spiky estimate; large h oversmooths.

### Step-by-Step Breakdown

1. Choose a kernel K (e.g., Gaussian: K(u)= (1/√(2π))·e^(−u²/2)).
2. Select a bandwidth h (via rules of thumb like Silverman’s rule or cross-validation).
3. For each evaluation point x:
    - Compute uᵢ = (x − xᵢ) / h for all i.
    - Sum K(uᵢ) over i, then divide by n·h to get f̂(x).
4. Plot the resulting f̂(x) curve over a suitable range.

### Real-World Examples

- Estimating the distribution of daily returns in finance to detect fat tails.
- Visualizing the density of gene expression levels in bioinformatics.
- Comparing customer waiting time distributions across service channels.

### Visual & Geometric Intuition

Imagine placing a small hill of width h and unit area centered at each data point. Adding all those hills yields a landscape: peaks where data are dense, valleys where data are sparse.

### Practice Problems & Python Exercises

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate skewed data
data = np.concatenate([np.random.normal(0, 1, 500),
                       np.random.normal(5, 0.5, 200)])

# Plot histogram + KDE
sns.histplot(data, bins=30, stat='density', alpha=0.4)
sns.kdeplot(data, bw_adjust=0.5, label='bw=0.5')
sns.kdeplot(data, bw_adjust=1.0, label='bw=1.0')
sns.kdeplot(data, bw_adjust=2.0, label='bw=2.0')
plt.legend(); plt.title("Histogram and KDE with Varying Bandwidth")
plt.show()
```

- Experiment with `bw_adjust` to see under- and over-smoothing.
- Use cross-validation (`sklearn.model_selection.GridSearchCV`) to select optimal h.

### How Data Scientists Use KDE

- Exploratory data analysis to spot multimodality or outliers.
- Density-based clustering (e.g., DBSCAN) to identify dense regions.
- Data augmentation: sampling from the estimated density.

---

## Violin Plots

### Intuitive Explanation

A violin plot combines a box-plot’s summary with a mirrored KDE on each side. It reveals distribution shape, spread, and outliers, along with median and interquartile range.

### Components of a Violin Plot

- The central box or marker shows the median (and optionally quartiles).
- The “violin” shape around it is the mirrored KDE of the data.
- Whiskers or inner markers indicate data extremes or outliers.

### Step-by-Step Breakdown

1. Compute the KDE of the data.
2. Mirror the KDE across a central axis to form the violin shape.
3. Overlay summary statistics (median line, IQR box) on top.
4. Repeat side by side for multiple groups for easy comparison.

### Real-World Examples

- Comparing test scores distribution across different classrooms.
- Visualizing feature distributions by target class in classification problems.
- Monitoring latency distributions for different API endpoints in production.

### Visual & Geometric Intuition

Picture a violin: the thickness at each height reflects the density of data at that value. Wider sections show many observations; narrow parts show few.

### Practice Problems & Python Exercises

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate two groups
group_a = np.random.normal(0, 1, 300)
group_b = np.random.exponential(1, 300)

# Create DataFrame
import pandas as pd
df = pd.DataFrame({
    'value': np.concatenate([group_a, group_b]),
    'group': ['A']*300 + ['B']*300
})

# Plot violin
sns.violinplot(x='group', y='value', data=df,
               inner='quartile', palette='pastel')
plt.title("Violin Plot of Two Groups")
plt.show()
```

- Toggle `inner` between `'box'`, `'quartile'`, and `None` for different summaries.
- Compare violin plots to box-plots to see additional distribution detail.

## How Data Scientists Use Violin Plots

- Side-by-side comparison of feature distributions by category.
- Detecting multimodal patterns within subgroups.
- Presenting distribution summaries in dashboards and reports.

---

## QQ (Quantile–Quantile) Plots

### Intuitive Explanation

A QQ plot compares the quantiles of your data to the quantiles of a theoretical distribution (often Normal). If the points lie roughly along a straight line, your data follow that distribution.

### Formal Definition

For sorted sample data x₍i₎ and theoretical distribution F with inverse CDF F⁻¹:

1. Compute plotting positions pᵢ = (i − 0.5) / n for i = 1…n.
2. Obtain theoretical quantiles qᵢ = F⁻¹(pᵢ).
3. Plot (qᵢ, x₍i₎).

### Step-by-Step Breakdown

1. Sort your data.
2. Choose a reference distribution (e.g., Normal(μ, σ)).
3. Compute empirical quantiles and theoretical quantiles.
4. Plot empirical vs. theoretical quantiles and add a 45° reference line.
5. Deviations from the line reveal skewness (curvature) or heavy tails (S-shape).

### Real-World Examples

- Verifying Normality of regression residuals before hypothesis testing.
- Checking if financial returns conform to a t-distribution.
- Validating model fit in probabilistic modeling by plotting data vs. posterior predictive.

### Visual & Geometric Intuition

Points on the 45° line indicate empirical quantile equals theoretical. Bowed curves imply skewness; S-shapes imply kurtosis differences.

### Practice Problems & Python Exercises

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Simulate data from a heavy-tailed distribution
data = np.random.standard_t(df=5, size=1000)

# QQ plot against Normal
stats.probplot(data, dist="norm", plot=plt)
plt.title("QQ Plot of t(5) vs. Normal")
plt.show()
```

- Replace `dist="norm"` with `dist="t"` to check fit to t-distribution.
- Use `stats.probplot` vs. manual computation of quantiles to understand internals.

### How Data Scientists Use QQ Plots

- Diagnosing departures from distributional assumptions (Normality, Exponentiality).
- Guiding choice of transformations (log, Box–Cox) to normalize data.
- Model validation in generative modeling and Bayesian posterior checks.

---

## Joint Distribution for Discrete Variables

### 1. What You Should Already Understand

Before diving in, make sure you’re comfortable with:

- Discrete random variables and their probability mass functions (PMFs):p_X(x) = P(X = x)
- Basic probability rules: sum rule, product rule (for independent events), and conditional probability.
- Counting outcomes in finite sample spaces (e.g., rolling dice, flipping coins).

If any of these feel fuzzy, review them first; we’ll build directly on those ideas here.

### 2. Intuitive Explanation

When two discrete random variables, X and Y, can each take on a set of values, the **joint distribution** describes how likely every possible pair (x, y) is.

- Imagine rolling two six-sided dice.
- X is the outcome of Die 1 (1–6), Y is the outcome of Die 2 (1–6).
- The joint distribution tells you P(X=3 and Y=5), P(X=2 and Y=2), etc., for all 36 ordered pairs.

This gives you a full picture of how X and Y behave together, not just separately.

### 3. Formal Definitions and Key Formulas

### 3.1 Joint PMF

The joint PMF of (X, Y) assigns a probability to each pair (x, y):

```markdown
p_{X,Y}(x, y) = P( X = x  and  Y = y )
```

It must satisfy:

- Non-negativity:
    
    ```
    p_{X,Y}(x, y) ≥ 0
    ```
    
- Total probability = 1:
    
    ```
    ∑_{x ∈ 𝒳} ∑_{y ∈ 𝒴} p_{X,Y}(x, y) = 1
    ```
    

Here 𝒳 and 𝒴 are the sets of possible values for X and Y.

### 3.2 Marginal PMFs

To get the behavior of X **alone** (ignore Y):

```markdown
p_X(x) = ∑_{y ∈ 𝒴} p_{X,Y}(x, y)
```

And for Y alone:

```markdown
p_Y(y) = ∑_{x ∈ 𝒳} p_{X,Y}(x, y)
```

### 3.3 Conditional PMF

The probability that X = x **given** Y = y is:

```markdown
p_{X|Y}(x | y) = p_{X,Y}(x, y) / p_Y(y)    when p_Y(y) > 0
```

This “zooms in” on the slice where Y = y and rescales probabilities to sum to 1.

### 3.4 Independence

X and Y are independent exactly when knowing Y tells you nothing about X:

```markdown
p_{X,Y}(x, y) = p_X(x) * p_Y(y)   for every x, y
```

If this holds, the joint PMF factors into the product of marginals.

### 4. Breaking Down the Formulas

1. **p_{X,Y}(x, y)**
    - “Probability that X is x **and** Y is y.”
    - You list every (x, y) pair and assign a number between 0 and 1 so that all pairs sum to 1.
2. **Marginalization**
    - To **ignore** Y, you add up the probabilities across all y for a fixed x.
    - This collapses the 2D table into the 1D PMF for X.
3. **Conditioning**
    - Fix Y = y; the total probability in that horizontal row is p_Y(y).
    - Divide each cell in the row by p_Y(y) so those conditional probabilities sum to 1.
4. **Checking independence**
    - Compare p_{X,Y}(x,y) to p_X(x)·p_Y(y).
    - If they match for all (x,y), X and Y are independent.

### 5. Real-World Data Science Examples

- **Text features**
    - X = presence of word “free” (0 or 1)
    - Y = presence of word “win” (0 or 1)
    - Joint PMF shows how often “free” and “win” co-occur in spam vs. ham.
- **Sensor networks**
    - X = temperature bin (low/medium/high)
    - Y = humidity bin
    - Joint PMF models how temperature and humidity vary together, feeding into probabilistic weather models.
- **User engagement**
    - X = number of pages viewed (0,1,2,…)
    - Y = number of clicks
    - Joint PMF helps estimate joint patterns for churn prediction or ad-recommendation models.

### 6. Geometric Interpretation

Visualize a table or heatmap:

| X \ Y | y₁ | y₂ | … |
| --- | --- | --- | --- |
| x₁ | p₁₁ | p₁₂ |  |
| x₂ | p₂₁ | p₂₂ |  |
| … | … | … |  |
- Each cell p_{i,j} = p_{X,Y}(xᵢ,yⱼ) is the probability of that pair.
- Marginals collapse rows or columns into sums.
- Conditional slices pick one row (or column) and rescale it.

### 7. Practice Problems & Python Exercises

### 7.1 Two Dice: Theoretical Joint PMF

1. List all 36 ordered pairs (x, y) for x, y ∈ {1,…,6}.
2. Assignfor each pair (fair dice).
    
    ```
    p_{X,Y}(x,y) = 1/36
    ```
    
3. Compute marginals:
    
    ```
    p_X(x) = 6*(1/36) = 1/6
    p_Y(y) = 6*(1/36) = 1/6
    ```
    
4. Verify independence: p_{X,Y}(x,y) = p_X(x)·p_Y(y).

### 7.2 Simulate Joint PMF in Python

```python
import numpy as np
from collections import Counter

# Simulate many trials of rolling two dice
trials = 200_000
rolls = np.random.randint(1, 7, size=(trials, 2))
pairs = [tuple(r) for r in rolls]

# Empirical joint counts
joint_counts = Counter(pairs)
joint_pmf = {pair: count / trials for pair, count in joint_counts.items()}

# Empirical marginal for X = first die
marginal_x = Counter(r[0] for r in rolls)
pX = {x: marginal_x[x] / trials for x in range(1,7)}

# Print P(X=3, Y=5), p_X(3), p_Y(5), and check independence
p_35 = joint_pmf.get((3,5), 0)
p3 = pX[3]
# Compute pY from marginals or similarly
print(f"P(X=3, Y=5): {p_35:.4f}")
print(f"P(X=3)*P(Y=5): {p3 * pX[5]:.4f}")
```

### 7.3 Email Feature Joint Distribution

Given a pandas DataFrame `df` with Boolean columns `has_free` and `has_win`:

```python
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    'has_free': np.random.choice([0,1], size=1000, p=[0.7,0.3]),
    'has_win':  np.random.choice([0,1], size=1000, p=[0.6,0.4])
})

# Joint frequencies
joint = df.groupby(['has_free','has_win']).size().unstack(fill_value=0)
joint_pmf = joint / len(df)

print("Joint PMF:\n", joint_pmf)

# Marginals
p_free = joint_pmf.sum(axis=1)
p_win  = joint_pmf.sum(axis=0)
print("\nMarginal P(has_free):\n", p_free)
print("\nMarginal P(has_win):\n", p_win)

# Conditional P(has_free=1 | has_win=1)
p_ffgw = joint_pmf.loc[1,1] / p_win[1]
print("\nP(free=1 | win=1):", p_ffgw)
```

### 8. How Data Scientists Use Joint Distributions

- **Feature interaction**: detect dependent features before applying ML algorithms that assume independence.
- **Naive Bayes**: builds joint likelihood via conditional marginals under the “naive” assumption.
- **Graphical models**: represent joint PMFs or PDFs in Bayesian networks and Markov random fields.
- **Contingency analysis**: chi-squared tests for independence in A/B testing or user-behavior studies.

---

## Joint Continuous Distributions

### 1. What You Should Already Understand

- Single‐variable PDF: f_X(x), normalization ∫f_X(x)dx=1.
- Continuous random variables and the notion of probability over intervals.
- Basic calculus: single and double integrals.
- Marginalization for discrete variables (summing); here we’ll replace sums by integrals.

### 2. Intuitive Explanation

A **joint continuous distribution** describes the density of two random variables X and Y together.

Instead of a 1D curve, you get a 2D “hill” or surface f<sub>X,Y</sub>(x,y).

The height at each (x,y) tells you how “densely” outcomes cluster around that point.

- Think of a topographic map: peaks where data are most likely, valleys where data are rare.
- Slicing this surface along x or y gives marginal densities; fixing y and moving along x gives conditional densities.

### 3. Formal Definitions and Key Formulas

### 3.1 Joint PDF and Normalization

```markdown
f_XY(x, y) >= 0    for all real x, y
integral_{y=-∞ to ∞} integral_{x=-∞ to ∞} f_XY(x, y) dx dy = 1
```

- f_XY(x, y) is the joint PDF of (X,Y).
- The double integral over the entire plane must equal 1.

### 3.2 Marginal PDFs

To “ignore” one variable, integrate it out:

```markdown
f_X(x) = integral_{y=-∞ to ∞} f_XY(x, y) dy
f_Y(y) = integral_{x=-∞ to ∞} f_XY(x, y) dx
```

These are the standalone densities of X and Y.

### 3.3 Conditional PDF

The density of X **given** Y=y:

```markdown
f_{X|Y}(x | y) = f_XY(x, y) / f_Y(y)    when f_Y(y) > 0
```

This rescales the joint along the line Y=y so that the area under x is 1.

### 3.4 Independence

X and Y are independent exactly when their joint PDF factors:

```markdown
f_XY(x, y) = f_X(x) * f_Y(y)    for all x, y
```

If this holds, knowing Y gives no information about X.

### 4. Breaking Down the Formulas

1. **f_XY(x, y) ≥ 0**
    
    Every point on the density surface is non-negative.
    
2. **Double integral = 1**
    
    Summing “heights × infinitesimal area” across the whole plane covers total probability.
    
3. **Marginalization**
    - To find f_X(x), we “collapse” the 2D surface by integrating along y.
    - Geometrically: project the density pile onto the x-axis.
4. **Conditioning**
    - Slice the surface at Y=y, giving a 1D curve along x.
    - Divide by the total area of that slice (f_Y(y)) to normalize.
5. **Independence**
    - If the surface is a perfect “product” of two independent hills, then X and Y don’t interact.
    - No ridge or valley linking x to y.

### 5. Real-World Data Science Examples

- **Bivariate Gaussian**: modeling two correlated features (e.g., height and weight).
- **Joint distribution of petal length & width** in the Iris dataset to visualize class separation.
- **Sensor fusion**: temperature and humidity readings modeled jointly to improve weather predictions.
- **Image analysis**: joint histogram of pixel intensities in two channels (e.g., R vs. G) for color balancing.

### 6. Geometric Interpretation

- Visualize f_XY(x,y) as a 3D surface or contour map.
- **Contour lines** connect points of equal density—like elevation lines on a topo map.
- **Slices** at y=y₀ are conditional densities f_{X|Y=y₀}(x).
- **Projections** onto axes yield marginals f_X(x) and f_Y(y).

### 7. Practice Problems & Python Exercises

### 7.1 Simulate & Visualize a Bivariate Gaussian

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
mean = [0, 0]
cov  = [[1.0, 0.8], [0.8, 1.0]]  # correlation 0.8

# Simulate
data = np.random.multivariate_normal(mean, cov, size=2000)
x, y = data[:,0], data[:,1]

# Joint scatter + KDE contours
sns.jointplot(x=x, y=y, kind='kde', fill=True, cmap='Blues')
plt.show()
```

- The contour plot shows level sets of f_XY(x,y).
- Changing `cov` to `[[1,0],[0,1]]` produces concentric circles (independence).

### 7.2 Compute Marginals from Samples

```python
# Empirical marginal of X via histogram
plt.hist(x, bins=30, density=True, alpha=0.5, label='empirical f_X')
# Overlay theoretical marginal N(0,1)
from scipy.stats import norm
xs = np.linspace(-4,4,200)
plt.plot(xs, norm.pdf(xs, 0, 1), 'r--', label='theoretical f_X')
plt.legend(); plt.show()
```

Repeat for y.

### 7.3 Estimate Conditional PDF f(X|Y=y0)

```python
# Choose a slice around Y=0 (±0.1)
mask = np.abs(y) < 0.1
x_slice = x[mask]

# Plot histogram of X|Y≈0
plt.hist(x_slice, bins=30, density=True, alpha=0.6, label='empirical f(X|Y≈0)')
# Theoretical normal with mean 0.8*y0=0 and var=1-0.8^2=0.36
from scipy.stats import norm
plt.plot(xs, norm.pdf(xs, 0, np.sqrt(1-0.8**2)), 'k--', label='theoretical')
plt.legend(); plt.show()
```

### 7.4 Joint Distribution in Iris Dataset

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset('iris')
# Plot joint distribution of petal_length vs petal_width colored by species
sns.jointplot(data=df, x='petal_length', y='petal_width',
              kind='kde', hue='species', fill=True, palette='muted')
plt.show()
```

- Compare shapes for each species; see how densities shift.

### 8. How Data Scientists Use Joint Continuous Distributions

- **Density estimation**: learn f_XY(x,y) nonparametrically (KDE, Gaussian mixtures).
- **Generative models**: sample new (x,y) from learned joint.
- **Feature decorrelation**: examine off-diagonal density for independence assumptions.
- **Anomaly detection**: flag points in low-density regions of the joint.
- **Bayesian inference**: joint posterior of (parameters, data) under continuous priors.

---

## Marginal and Conditional Distributions

### 1. What You Should Already Understand

- Joint distributions for discrete and continuous variables.
- PMF (probability mass function) for discrete, PDF (probability density function) for continuous.
- Basic sums (for discrete) and integrals (for continuous).

If you need to revisit joint PMF/PDF, see our previous sections on joint distributions.

### 2. Intuitive Explanation

**Marginal distribution** is what you get when you “ignore” one variable and focus on the other.

- Analogy: you have a table of sales by product (X) and region (Y).
- The marginal by product sums sales across all regions—so you see total product popularity.

**Conditional distribution** is how one variable behaves **given** a specific value of the other.

- Analogy: “Given region = East, what fraction of sales come from each product?”
- You zoom in on the East row of the table and rescale so those probabilities sum to 1.

### 3. Formal Formulas

### 3.1 Discrete Case

```markdown
# Joint PMF: p_{X,Y}(x,y)

# Marginal PMFs
p_X(x) = ∑_{y ∈ Y_values} p_{X,Y}(x, y)
p_Y(y) = ∑_{x ∈ X_values} p_{X,Y}(x, y)

# Conditional PMF
p_{X|Y}(x | y) = p_{X,Y}(x, y) / p_Y(y)  for p_Y(y) > 0
p_{Y|X}(y | x) = p_{X,Y}(x, y) / p_X(x)  for p_X(x) > 0
```

### 3.2 Continuous Case

```markdown
# Joint PDF: f_{X,Y}(x,y)

# Marginal PDFs
f_X(x) = ∫_{y=-∞ to ∞} f_{X,Y}(x, y) dy
f_Y(y) = ∫_{x=-∞ to ∞} f_{X,Y}(x, y) dx

# Conditional PDF
f_{X|Y}(x | y) = f_{X,Y}(x, y) / f_Y(y)  for f_Y(y) > 0
f_{Y|X}(y | x) = f_{X,Y}(x, y) / f_X(x)  for f_X(x) > 0
```

### 4. Step-by-Step Breakdown

1. **Marginalization (Discrete)**
    - Fix x, add up p_{X,Y}(x,y) over all y.
    - This collapses the 2D joint into the 1D distribution of X.
2. **Marginalization (Continuous)**
    - Fix x, integrate f_{X,Y}(x,y) over all y.
    - Area under the slice of the surface gives f_X(x).
3. **Conditioning**
    - “Given Y = y” means look at the slice p_{X,Y}(x,y) (or f_{X,Y}(x,y)).
    - Divide by the total probability of that slice (p_Y(y) or f_Y(y)) so it again sums/integrates to 1.
4. **Key point**: conditioning rescales, marginalization sums or integrates out the other variable.

### 5. Real-World ML & DS Examples

- **Naive Bayes**
    - Compute p(feature = x | class = c) via conditional PMF/PDF.
    - Multiply by p(class) (marginal) to get joint likelihood.
- **Feature Independence Checks**
    - Compare p_{X,Y}(x,y) vs.\ p_X(x)·p_Y(y) to test if two features are independent.
- **User Behavior Modeling**
    - X = time on site, Y = pages visited.
    - Marginal f_X reveals overall session length; conditional f_{Y|X}(pages | time=5 min) shows click patterns for 5-minute sessions.

### 6. Geometric & Visual Intuition

- **Marginal**: project the joint surface ( or heatmap ) down onto one axis.
- **Conditional**: slice the surface at y = y₀; what remains is a curve f_{X|Y=y₀}(x) after rescaling.

*Image description*: imagine a 3D hill; looking at it from the side (along Y) shows the marginal height vs.\ X. Cutting a vertical plane at Y = y₀ and normalizing that slice shows the conditional shape.

### 7. Practice Problems & Python Exercises

### 7.1 Discrete Example: Two Dice

**a. Theoretical Marginal & Conditional**

- Joint PMF: p_{X,Y}(x,y) = 1/36 for x,y ∈ {1…6}.
- Marginal: p_X(x) = 6·(1/36) = 1/6.
- Conditional: P(X=3 | Y=5) = p_{X,Y}(3,5)/p_Y(5) = (1/36)/(1/6) = 1/6.

**b. Empirical Simulation**

```python
import numpy as np
from collections import Counter

trials = 200_000
rolls = np.random.randint(1,7,(trials,2))
pairs = [tuple(r) for r in rolls]

# Joint PMF
joint = Counter(pairs)
joint_pmf = {k: v/trials for k,v in joint.items()}

# Marginal for Y
marginal_y = Counter(y for _,y in rolls)
pY = {y: marginal_y[y]/trials for y in range(1,7)}

# P(X=3, Y=5) and P(X=3|Y=5)
p_35 = joint_pmf[(3,5)]
p_x3_given_y5 = p_35 / pY[5]
print("P(X=3, Y=5):", p_35)
print("P(X=3|Y=5):", p_x3_given_y5)
```

### 7.2 Continuous Example: Bivariate Normal

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

# Simulate correlated normals
mean = [0,0]
cov  = [[1, 0.6],[0.6, 1]]
data = np.random.multivariate_normal(mean, cov, size=5000)
x, y = data[:,0], data[:,1]

# Empirical marginal f_X(x)
sns.histplot(x, bins=30, stat='density', alpha=0.5, label='empirical f_X')
xs = np.linspace(-4,4,200)
plt.plot(xs, norm.pdf(xs, 0, 1), 'r--', label='theoretical N(0,1)')
plt.legend(); plt.show()

# Empirical conditional f_{X|Y=1}(x)
y0 = 1.0
tol = 0.1
mask = (np.abs(y - y0) < tol)
x_slice = x[mask]
sns.kdeplot(x_slice, label=f'empirical f(X|Y≈{y0})')
# Theoretical conditional is N(mean=0.6*y0, sd=√(1−0.6²))
mean_cond = 0.6 * y0
sd_cond   = np.sqrt(1 - 0.6**2)
plt.plot(xs, norm.pdf(xs, mean_cond, sd_cond), 'k--', label='theoretical')
plt.legend(); plt.show()
```

### 8. How Data Scientists Use Marginals & Conditionals

- **Bayesian inference**: posterior p(θ | data) ∝ p(data | θ)·p(θ). Here p(data | θ) is a conditional distribution.
- **Feature selection**: compare p(feature) vs.\ p(feature | class) to choose discriminative features.
- **Anomaly detection**: flag points where p_{X,Y}(x,y) is very low, or where conditional p_{Y|X}(y | x) deviates strongly from marginal p_Y(y).

---

## Covariance of a Dataset

### 1. Intuitive Explanation

Covariance measures how two variables change together.

- A **positive** covariance means as one variable increases, the other tends to increase.
- A **negative** covariance means as one increases, the other tends to decrease.
- A **zero** (or near-zero) covariance suggests no linear relationship.

Imagine you record daily hours studied (X) and exam scores (Y). If higher study hours typically coincide with higher scores, X and Y have positive covariance.

### 2. Prerequisites You Should Already Understand

- **Mean (average)** of a list of numbers.
- **Deviation from the mean**: how far each observation is from its average.
- Basic Python or pandas for data handling (optional).

### 3. Formal Definitions and Formulas

### 3.1 Population Covariance

When you believe your data represent the entire population:

```markdown
cov(X, Y) =
  (1 / n)
  * ∑_{i=1 to n} (x_i − μ_X) * (y_i − μ_Y)
```

- `n` is number of data points.
- `μ_X` and `μ_Y` are the population means of X and Y.

### 3.2 Sample Covariance

When your data is a sample from a larger population (most common in DS/ML):

```markdown
cov_sample(X, Y) =
  (1 / (n − 1))
  * ∑_{i=1 to n} (x_i − x̄) * (y_i − ȳ)
```

- `x̄` and `ȳ` are the sample means.
- Dividing by `n−1` gives an unbiased estimate of the population covariance.

### 4. Step-by-Step Explanation of the Sample Covariance Formula

1. **Compute means**
    - `x̄ = (1/n) * ∑ x_i`
    - `ȳ = (1/n) * ∑ y_i`
2. **Compute deviations**
    - For each pair `(x_i, y_i)`, find
        - `dx_i = x_i − x̄`
        - `dy_i = y_i − ȳ`
3. **Multiply deviations**
    - For each i, compute the product `dx_i * dy_i`.
    - This product is positive when both deviations have the same sign.
4. **Sum products**
    - `sum_dev = ∑ (dx_i * dy_i)`
5. **Normalize**
    - Divide by `(n−1)` to get the final covariance:
        
        ```markdown
        cov_sample = sum_dev / (n − 1)
        ```
        

### 5. Real-World Data Science Examples

- **Feature relationships**
    - Check covariance between `age` and `income` to see if older customers earn more.
- **Multivariate Gaussian models**
    - Covariance matrix controls the shape of the joint PDF, used in classification or anomaly detection.
- **Principal Component Analysis (PCA)**
    - PCA diagonalizes the covariance matrix to find directions (eigenvectors) of maximal variance.

### 6. Visual & Geometric Interpretation

- **Scatter plot** of (X, Y):
    - **Sloped cluster** rising left-to-right → positive covariance.
    - **Sloped cluster** falling left-to-right → negative covariance.
    - **Circular cloud** → covariance near zero.
- **Ellipse** contours for a bivariate Gaussian:
    - The orientation of the ellipse’s major axis aligns with the sign and magnitude of covariance.

### 7. Practice Problems & Python Exercises

### 7.1 Manual Covariance Calculation

Data:

```python
X = [2, 4, 6, 8]
Y = [1, 3, 5, 10]
```

Steps:

1. Compute sample means `x̄` and `ȳ`.
2. Compute deviations and their products.
3. Sum and divide by `n−1`.

### 7.2 Covariance with NumPy

```python
import numpy as np

X = np.array([2, 4, 6, 8])
Y = np.array([1, 3, 5, 10])

# Manual using the formula
n = len(X)
x_mean, y_mean = X.mean(), Y.mean()
cov_manual = ((X - x_mean) * (Y - y_mean)).sum() / (n - 1)

# NumPy covariance matrix
cov_matrix = np.cov(X, Y, ddof=1)

print("Manual covariance:", cov_manual)
print("NumPy covariance matrix:\n", cov_matrix)
```

### 7.3 Covariance Matrix in pandas

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
df = sns.load_dataset('iris')[['sepal_length', 'sepal_width', 'petal_length']]

# Compute covariance matrix
cov_mat = df.cov()
print("Covariance matrix:\n", cov_mat)

# Visualize as heatmap
sns.heatmap(cov_mat, annot=True, cmap='coolwarm')
plt.title("Feature Covariance Matrix")
plt.show()
```

### 8. How Data Scientists Use Covariance Daily

- **Feature engineering**: decide which features to combine or drop based on covariance.
- **Multicollinearity detection**: high covariance between features can degrade linear model performance.
- **Dimensionality reduction**: covariance matrix eigenvalues guide how many principal components to keep.
- **Anomaly detection**: points with unusual joint behavior (low joint density) often show unexpected covariance patterns.

---

## Covariance of a Probability Distribution

### 1. Intuitive Explanation

Covariance tells you how two random variables, X and Y, move together on average.

If high values of X tend to go with high values of Y, covariance is positive.

If high X matches low Y, covariance is negative.

When they’re unrelated, covariance is zero.

This extends our dataset‐based concept to the **true** distribution: we weight every outcome by its probability instead of empirical frequency.

### 2. Prerequisites You Should Know

- Expected value (E[X]) for discrete sums or continuous integrals.
- Joint distribution pₓᵧ(x, y) or fₓᵧ(x, y).
- Basic sum notation and integral calculus.

If you need a quick review, revisit the sections on **expectation** and **joint distributions**.

### 3. Formal Definitions

### Discrete Random Variables

```markdown
Cov(X, Y)
  = E[(X - E[X]) * (Y - E[Y])]
  = ∑_{x ∈ 𝒳} ∑_{y ∈ 𝒴} (x - μ_X) * (y - μ_Y) * p_{X,Y}(x, y)
```

- μ_X = E[X] = ∑ x p_X(x)
- μ_Y = E[Y] = ∑ y p_Y(y)

### Continuous Random Variables

```markdown
Cov(X, Y)
  = ∫_{y} ∫_{x} (x - μ_X) * (y - μ_Y) * f_{X,Y}(x, y) dx dy
```

- μ_X = ∫ x f_X(x) dx
- μ_Y = ∫ y f_Y(y) dy

### Shortcut Formula

```markdown
Cov(X, Y) = E[X * Y] - E[X] * E[Y]
```

- Compute E[XY] via joint distribution, then subtract product of means.

### 4. Step-by-Step Breakdown

1. **Find means**:
    - μ_X = E[X]
    - μ_Y = E[Y]
2. **Center each variable**:
    - ΔX = X − μ_X
    - ΔY = Y − μ_Y
3. **Multiply centered terms**:
    - For each (x,y), compute ΔX·ΔY weighted by joint probability.
4. **Sum or integrate** over all (x,y) to get Cov(X,Y).
5. **Or** use the shortcut:
    - Compute E[XY] directly from pₓᵧ or fₓᵧ.
    - Subtract μ_X·μ_Y.

### 5. Real-World Examples

- **Feature co-movement**: in a multivariate Gaussian, covariance defines the tilt of contour ellipses.
- **Portfolio risk**: Cov(return_A, return_B) enters the variance of combined investments.
- **Sensor fusion**: combining measurements with known noise covariances for Kalman filters.

### 6. Geometric Interpretation

- Picture the joint density surface fₓᵧ(x,y).
- Subtract the means: shift the center to (0,0).
- Covariance is the “tilt” of that shifted surface.
- In 2D scatter: the best‐fit ellipse’s axes align with principal directions of covariance.

### 7. Practice Problems & Python Exercises

### 7.1 Discrete Joint PMF

Let X,Y each be 0 or 1 with joint PMF:

| X\Y | 0 | 1 |
| --- | --- | --- |
| 0 | 0.2 | 0.3 |
| 1 | 0.1 | 0.4 |
- Compute μ_X, μ_Y.
- Use the double sum formula to find Cov(X,Y).

### 7.2 Continuous Bivariate Normal

```python
import numpy as np

# Parameters
mean = [1, 2]
cov  = [[4, 1.5], [1.5, 9]]  # Var(X)=4, Var(Y)=9, Cov(X,Y)=1.5

# Simulate
data = np.random.multivariate_normal(mean, cov, size=200000)
X, Y = data[:,0], data[:,1]

# Empirical covariance
emp_cov = np.cov(X, Y, ddof=0)[0,1]
print("Empirical Cov(X,Y):", emp_cov)
print("Theoretical Cov(X,Y):", cov[0][1])
```

### 7.3 Shortcut Verification

- Compute E[XY] by averaging X*Y over your sample.
- Compute empirical means and verify
    
    ```
    Cov(X,Y) ≈ E[XY] − E[X]*E[Y]
    ```
    

### 8. How Data Scientists Use Covariance

- Building the **covariance matrix** for PCA and factor analysis.
- **Regularizing** covariance estimates in high dimensions (Ledoit–Wolf).
- **Whitening** features by decorrelating via covariance eigen decomposition.
- Measuring **co-movement** in time-series analysis.

---

## Covariance Matrix

### 1. What You Should Already Understand

- How to compute the **mean vector** of multiple variables.
- The concept of **covariance** between two variables, Cov(X, Y).
- Basic matrix operations and transposes.

If any of these feel unclear, revisit the sections on *mean*, *covariance*, and *matrix multiplication*.

### 2. Intuitive Explanation

A covariance matrix generalizes pairwise covariance to many variables at once.

- Imagine a dataset with columns: height, weight, and age.
- The covariance matrix shows how each pair of these variables moves together—height vs. weight, height vs. age, weight vs. age—and also each variable’s own variance on the diagonal.

Visually, it captures the “shape” of the cloud of points in multi-dimensional space.

### 3. Formal Definitions and Key Formulas

### 3.1 Population Covariance Matrix

For a random vector **X** = [X₁, X₂, …, Xₖ]ᵀ with mean vector **μ**, the covariance matrix Σ is:

```markdown
Σ = E[ (X − μ) (X − μ)ᵀ ]
```

- Σ is a k×k matrix.
- Entry Σ_{i,j} = Cov(Xᵢ, Xⱼ).

### 3.2 Sample Covariance Matrix

Given data matrix **X** of shape (n samples × k features), let **X̄** be the n×k matrix where each column is the sample mean x̄ᵢ repeated n times. Then the unbiased sample covariance matrix S is:

```markdown
S = (1 / (n − 1)) · (X − X̄)ᵀ  ·  (X − X̄)
```

- (X − X̄) is the centered data matrix.
- S[i, j] = sample covariance between feature i and j.

### 4. Step-by-Step Breakdown of the Sample Formula

1. **Compute column means**
    - x̄ᵢ = (1/n) · ∑*{t=1..n} X*{t,i} for each feature i.
2. **Center data**
    - Subtract x̄ᵢ from each entry in column i: (X − X̄).
3. **Multiply transposed centered data**
    - (X − X̄)ᵀ has shape (k × n).
    - (X − X̄) has shape (n × k).
    - Their product is (k × k).
4. **Scale by (1/(n−1))**
    - Ensures an unbiased estimate of the population covariance.

### 5. Real-World Data Science Examples

- **Principal Component Analysis (PCA)**
    - Perform eigen decomposition on the covariance matrix to find directions of maximal variance.
- **Multivariate Gaussian**
    - Σ defines the shape of elliptical contours in a Gaussian probability density.
- **Portfolio Optimization**
    - Covariance of asset returns is used to minimize portfolio risk.

### 6. Geometric Interpretation

- In 2D, covariance matrix Σ defines an **ellipse**:
    - The ellipse’s axes align with Σ’s eigenvectors.
    - Axis lengths are √eigenvalues.
- In higher dimensions, Σ defines a **hyper-ellipsoid** capturing data spread.

### 7. Practice Problems & Python Exercises

### 7.1 Manual Computation for a Small Dataset

```python
import numpy as np

# Sample data: each row is an observation, columns are features
X = np.array([
    [65, 120],
    [70, 150],
    [60, 115],
    [75, 160]
], dtype=float)

n, k = X.shape
means = X.mean(axis=0)           # compute feature means
X_centered = X - means           # center data

# Manual covariance matrix
S_manual = (X_centered.T @ X_centered) / (n - 1)
print("Manual S:\n", S_manual)
```

### 7.2 Using NumPy’s Built-in Function

```python
import numpy as np

# X as above
cov_matrix = np.cov(X, rowvar=False, ddof=1)
print("NumPy S:\n", cov_matrix)
```

### 7.3 Visualizing with a Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cov_matrix, annot=True, cmap='coolwarm',
            xticklabels=['height','weight'],
            yticklabels=['height','weight'])
plt.title("Sample Covariance Matrix Heatmap")
plt.show()
```

### 8. How Data Scientists Use Covariance Matrices

- **Feature Selection**: identify highly correlated features to drop or combine.
- **Dimensionality Reduction**: eigenvalues of Σ guide how many principal components to retain.
- **Anomaly Detection**: Mahalanobis distance uses Σ⁻¹ to measure deviations from the mean in multi-D.

---

## Correlation Coefficient

### 1. What You Should Already Understand

- How to compute **mean** and **standard deviation** of a variable.
- What **covariance** measures (how two variables vary together).
- Basic scatter-plot interpretation (patterns of co-movement).

If any of these feel unclear, revisit the sections on mean, standard deviation, and covariance.

### 2. Intuitive Explanation

Correlation is a **normalized** measure of how two variables move together.

- When X and Y both increase (or both decrease) in tandem, correlation is **positive**.
- When X increases while Y decreases, correlation is **negative**.
- When there’s no consistent linear relationship, correlation is near **zero**.

Imagine two dancers (variables): if they move in sync, their “dance” has high positive correlation; if one steps forward when the other steps back, it’s negative; if they just wander randomly, it’s near zero.

### 3. Formal Definitions and Key Formulas

### 3.1 Population (Pearson) Correlation

```markdown
ρ(X, Y) = Cov(X, Y) / (σ_X * σ_Y)
```

- `Cov(X, Y)` is the average product of centered X and Y.
- `σ_X` and `σ_Y` are the population standard deviations of X and Y.

### 3.2 Sample Correlation

For data pairs ((x_i, y_i)), (i=1…n):

```markdown
r =
  ∑_{i=1}^n (x_i − x̄) · (y_i − ȳ)
  ——————————————————————
  √[ ∑ (x_i − x̄)²  ·  ∑ (y_i − ȳ)² ]
```

- `x̄` and `ȳ` are the sample means.
- The numerator is the sum of products of deviations.
- The denominator rescales by each variable’s total squared deviation.

### 4. Step-by-Step Breakdown

1. **Center each variable**
    
    (\displaystyle dx_i = x_i - x̄,\quad dy_i = y_i - ȳ)
    
2. **Compute the numerator**
    
    (\displaystyle N = \sum_{i=1}^n dx_i \times dy_i)
    
3. **Compute each sum of squares**
    
    (\displaystyle S_X = \sum dx_i^2,\quad S_Y = \sum dy_i^2)
    
4. **Form the denominator**
    
    (\displaystyle D = \sqrt{S_X \times S_Y})
    
5. **Divide**
    
    (\displaystyle r = N / D)
    

### 5. Real-World Examples in Data Science & ML

- **Feature selection**: drop one of any pair of features with (|r|>0.9) to avoid multicollinearity.
- **PCA preprocessing**: examine correlation matrix to decide if whitening is needed.
- **Model diagnostics**: check correlation of residuals to detect pattern violations.
- **Recommender systems**: user–item rating correlations to find similar users or items.

### 6. Visual & Geometric Interpretation

- **Scatter plot**
    - A tight cloud along a rising line → high positive (r).
    - A tight cloud along a falling line → high negative (r).
    - A round cloud → (r\approx0).
- **Angle between centered vectors**
    - Stack ({dx_i}) into vector **u**, ({dy_i}) into **v**.
    - (r = \cos(\theta)) where (\theta) is the angle between **u** and **v**.

### 7. Practice Problems & Python Exercises

### 7.1 Manual Calculation

```python
# Data
X = [2, 4, 6, 8]
Y = [1, 3, 5, 10]

# Step-by-step
import math
n = len(X)
x_mean = sum(X)/n
y_mean = sum(Y)/n

# deviations
dx = [x - x_mean for x in X]
dy = [y - y_mean for y in Y]

# numerator and sums of squares
num = sum(dxi*dyi for dxi,dyi in zip(dx, dy))
sx  = sum(dxi**2 for dxi in dx)
sy  = sum(dyi**2 for dyi in dy)

r_manual = num / math.sqrt(sx * sy)
print("Manual r:", r_manual)
```

### 7.2 Using NumPy and pandas

```python
import numpy as np
import pandas as pd

# Simulate sample data
np.random.seed(0)
df = pd.DataFrame({
    'age': np.random.randint(20, 60, 100),
    'income': np.random.normal(50000, 15000, 100)
})

# pandas correlation
r_pandas = df['age'].corr(df['income'])
print("Pandas r:", r_pandas)

# NumPy version
r_numpy = np.corrcoef(df['age'], df['income'])[0,1]
print("NumPy r:", r_numpy)
```

### 7.3 Visualizing Correlation

```python
import matplotlib.pyplot as plt
plt.scatter(df['age'], df['income'], alpha=0.6)
plt.title(f"Scatter plot (r={r_pandas:.2f})")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()
```

### 8. How Data Scientists Use Correlation Coefficient

- **Exploratory Data Analysis (EDA)**: quick check for linear relationships.
- **Multicollinearity detection**: high (|r|) warns of redundant predictors in regression.
- **Feature engineering**: combine or transform highly correlated features.
- **Baseline models**: simple linear correlations can serve as quick predictive baselines.

---

## Multivariate Gaussian Distribution

### 1. What You Should Already Understand

- How a **univariate Normal** (bell curve) works: mean μ, variance σ², and its PDF.
- Basics of **vectors** and **matrices**: addition, transpose, determinant, and inverse.
- The **covariance matrix** Σ and how it captures variances and covariances between features.
- Computing the **Mahalanobis distance**: ((x-\mu)^\top \Sigma^{-1} (x-\mu)).

If any of these feel unfamiliar, pause and review linear algebra and the univariate Normal before proceeding.

### 2. Intuitive Explanation

- A univariate Normal is a 1-D “bell.” In higher dimensions, the Multivariate Gaussian is a **bell-shaped hill** in (\mathbb{R}^d).
- Its peak sits at the **mean vector** μ (the center of the hill).
- Its **spread** and **orientation** come from the covariance matrix Σ:
    - Directions with large variance stretch the hill out.
    - Correlations tilt the hill toward certain axes.
- Every “contour line” of equal density forms an **ellipse** (in 2D) or an **ellipsoid** (in higher D).

Use cases:

- Modeling joint behavior of several continuous features (e.g., height & weight).
- Gaussian discriminant analysis (GDA) for classification.
- Anomaly detection: points in low-density regions of the hill are outliers.

## 3. Formal Definition and Key Formula

For a random vector **X** in (\mathbb{R}^d) with mean vector **μ** (shape (d\times1)) and covariance matrix **Σ** ((d\times d), positive-definite), the PDF is:

```markdown
f_X(x) =
  1
  -------------------------------------------------
  sqrt( (2 · π)^d  ·  det(Σ) )
  · exp ⎡ –½ · (x – μ)ᵀ · Σ⁻¹ · (x – μ) ⎤
```

- **(2·π)^d**: extends the normalization constant to d dimensions.
- **det(Σ)**: scales volume according to how much Σ stretches or shrinks space.
- **Σ⁻¹**: “whitens” differences by accounting for scale and correlation.
- **Mahalanobis term** ((x−μ)ᵀ Σ⁻¹ (x−μ)): squared distance from x to μ in Σ-scaled space.

### 4. Breaking Down the Formula

1. **Normalization Constant**
    
    ```markdown
    Z = 1 / sqrt((2·π)^d · det(Σ))
    ```
    
    Ensures total probability integrates to 1 over (\mathbb{R}^d).
    
2. **Exponent**
    
    ```markdown
    exponent = –½ · (x – μ)ᵀ · Σ⁻¹ · (x – μ)
    ```
    
    Measures how far x sits from μ, weighted by Σ. Larger Mahalanobis distance ⇒ smaller density.
    
3. **Putting Together**
    
    ```markdown
    f_X(x) = Z · exp(exponent)
    ```
    
    Yields a smooth, bell-shaped density surface (or contour map).
    

### 5. Geometric Interpretation

- **Eigen-decomposition**: Σ = Q · Λ · Qᵀ
    - Q’s columns are **principal axes** of the ellipsoid.
    - Λ’s diagonal entries (λ₁…λ_d) are the **squared lengths** of those axes.
- **Contours** where ((x−μ)ᵀΣ⁻¹(x−μ) = c) are ellipsoids centered at μ.
    - (c=1) gives the one-standard-deviation boundary.
    - Higher c ⇒ larger ellipsoid (lower density).

### 6. Practice Problems & Python Exercises

### 6.1 Simulate & Estimate Parameters

```python
import numpy as np

# Parameters
mu     = np.array([2.0, -1.0])
Sigma  = np.array([[2.0, 0.8],
                   [0.8, 1.5]])

# Simulate samples
n = 5000
data = np.random.multivariate_normal(mu, Sigma, size=n)

# Empirical estimates
mu_emp    = data.mean(axis=0)
Sigma_emp = np.cov(data, rowvar=False, ddof=1)

print("True μ:", mu)
print("Empirical μ:", mu_emp)
print("True Σ:\n", Sigma)
print("Empirical Σ:\n", Sigma_emp)
```

### 6.2 Visualize Contours in 2D

```python
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Grid for plotting
x = np.linspace(-2, 6, 200)
y = np.linspace(-5, 4, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Compute PDF values
rv = multivariate_normal(mean=mu, cov=Sigma)
Z = rv.pdf(pos)

# Plot contours and scatter
plt.contour(X, Y, Z, levels=6, cmap='Blues')
plt.scatter(data[:,0], data[:,1], s=5, alpha=0.3)
plt.title("Bivariate Gaussian Contours")
plt.xlabel("X₁"); plt.ylabel("X₂")
plt.show()
```

### 6.3 Mahalanobis Distance & Outlier Detection

```python
import numpy as np

# Compute Σ⁻¹ once
Sigma_inv = np.linalg.inv(Sigma)

# Mahalanobis distances
diff = data - mu
m_dist_sq = np.sum(diff @ Sigma_inv * diff, axis=1)

# Flag points with distance² > threshold (e.g., 9 ≈ 3σ ellipse)
outliers = data[m_dist_sq > 9]
print("Number of outliers:", len(outliers))
```

### 6.4 Fit MVN to Real Data (Iris Petals)

```python
import seaborn as sns
from scipy.stats import multivariate_normal

iris = sns.load_dataset('iris')
# Use petal_length & petal_width
X = iris[['petal_length','petal_width']].values
mu_fit    = X.mean(axis=0)
Sigma_fit = np.cov(X, rowvar=False)

# Visualize for each species
for species, group in iris.groupby('species'):
    data_sp = group[['petal_length','petal_width']].values
    rv_sp = multivariate_normal(mean=data_sp.mean(axis=0),
                                cov=np.cov(data_sp, rowvar=False))
    sns.kdeplot(x=data_sp[:,0], y=data_sp[:,1],
                levels=3, label=species)

plt.title("Iris Petal Measurements by Multivariate Normal Contours")
plt.xlabel("Petal Length"); plt.ylabel("Petal Width")
plt.legend(); plt.show()
```

### 7. How Data Scientists Use Multivariate Gaussians

- **Gaussian Discriminant Analysis (GDA)**: classify by modeling class-conditional MVNs.
- **Anomaly Detection**: fit MVN to “normal” data; points in low-density regions flagged as anomalies.
- **PCA Whitening**: transform data so MVN becomes spherical (identity covariance).
- **Probabilistic PCA & Factor Analysis**: assume MVN latent structure to reduce dimension.
- **Bayesian Linear Regression**: posterior over weights is MVN.

---

##