# Collaborative Filtering Recommender System

## What This Project Does

Builds a movie recommender system from scratch using **Collaborative Filtering** and **TensorFlow**.

The model learns hidden user preferences and movie features purely from rating patterns — without ever asking users directly what they like.

---

## Project Structure

```
Part 1 — Simple Version (standalone)
├── 3 users, 4 movies, 2 features
├── Built from scratch to understand every line
├── For loop cost function (easy to read)
├── Vectorized cost function (used in training)
└── Runs without any extra files ✅

Part 2 — Full Version (requires Coursera lab files)
├── 443 users, 4778 movies, 100 features
├── Real MovieLens dataset
├── Personal ratings added
└── Top movie recommendations generated
```

---

## The Math

**Prediction formula:**
```
ŷ(i,j) = W(j) · X(i) + b(j)
```

**Cost function:**
```
J = (1/2) · Σ(i,j):r(i,j)=1 (W(j)·X(i) + b(j) - y(i,j))²
  + (λ/2) · Σ W²    ← regularize user preferences
  + (λ/2) · Σ X²    ← regularize movie features
```

**What the model learns:**
| Variable | Meaning |
|---|---|
| W (ω) | emotional preference fingerprint of each user |
| X | hidden identity of each movie |
| b | personal bias per user |

---

## Key Concepts

### Why Gaussian Initialization?
```
f(x) = (1/√2πσ²) · e^(−x²/2σ²)   σ = 0.1

Zero init  → identical gradients → learning collapses ❌
Gaussian   → diverse gradients   → each parameter learns ✅
```

### Why GradientTape?
```python
with tf.GradientTape() as tape:
    cost = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

grads = tape.gradient(cost, [X, W, b])
optimizer.apply_gradients(zip(grads, [X, W, b]))
```
TensorFlow automatically computes dJ/dW, dJ/dX, dJ/db — no manual calculus needed.

### Convergence
```
Iteration   0 | Cost: 9.60  ← random noise
Iteration  50 | Cost: 3.22  ← patterns emerging
Iteration 100 | Cost: 0.88  ← structure forming
Iteration 200 | Cost: 0.15  ← learned ✅
```

---

## How to Run

### Part 1 (standalone)
```bash
pip install numpy tensorflow
python collaborative_filtering_recommender.py
```

### Part 2 (requires Coursera lab files)
Run inside the Coursera lab environment where these files exist:
```
recsys_utils.py
small_movie_list.csv
```

---

## The Key Insight

> The scientific community calls W a *"latent factor vector."*
> What it truly captures is the **emotional lens** through which each person perceives the world of content.
>
> No survey conducted. No question asked. No preference declared.
> The mathematics inferred it — silently — from behaviour alone.

---

## Stack
`TensorFlow` · `GradientTape` · `Adam Optimizer` · `Matrix Factorization` · `Gaussian Initialization` · `L2 Regularization`

## Foundation
Andrew Ng — Machine Learning Specialization 
Dataset: MovieLens ml-latest-small — Harper & Konstan, 2015
