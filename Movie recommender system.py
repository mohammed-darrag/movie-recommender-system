"""
============================================================
 Movie Recommender System
 Collaborative Filtering with TensorFlow
============================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


print("=" * 60)
print(" Movie Recommender System")
print(" Collaborative Filtering with TensorFlow")
print(" 3 users | 4 movies | 2 features")
print("=" * 60)


# ============================================================
#  DATA
# ============================================================

# Ratings matrix — 0 means not rated yet
#          Titanic  Avengers  Notebook  Batman
Y = np.array([
    [5, 1, 4, 0],   # Alice
    [1, 5, 0, 4],   # Bob
    [0, 4, 2, 5],   # Carol
], dtype=np.float32)

# R = 1 if user rated the movie, 0 if not
R = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [0, 1, 1, 1],
], dtype=np.float32)

users  = ["Alice", "Bob", "Carol"]
movies = ["Titanic", "Avengers", "Notebook", "Batman"]

print("\nRatings matrix Y:")
print(f"{'':12}", end="")
for m in movies:
    print(f"{m:12}", end="")
print()
for i, u in enumerate(users):
    print(f"{u:12}", end="")
    for j in range(len(movies)):
        val = int(Y[i, j])
        print(f"{'?' if val == 0 else val:<12}", end="")
    print()

num_users    = Y.shape[0]   # 3
num_movies   = Y.shape[1]   # 4
num_features = 2            # hidden features to learn


# ============================================================
#  NORMALIZE Y
#  Subtract mean rating per movie (only from rated users)
#  This removes the bias of users who always rate high or low
# ============================================================

Ynorm = Y.copy()
Ymean = np.zeros(num_movies)

for j in range(num_movies):
    rated = R[:, j] == 1
    if rated.any():
        mean_val      = Y[rated, j].mean()
        Ymean[j]      = mean_val
        Ynorm[rated, j] -= mean_val

print(f"\nNormalized Y (Ynorm):")
print(np.round(Ynorm, 2))


# ============================================================
#  COST FUNCTION — FOR LOOP VERSION
#  Easy to read and understand
#
#  J = (1/2) * sum(W[j]·X[i] + b[j] - Y[i,j])²   ← main cost
#    + (λ/2) * sum(W²)                              ← regularize W
#    + (λ/2) * sum(X²)                              ← regularize X
# ============================================================

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Collaborative filtering cost function — for loop version.

    Args:
      X       (num_movies × num_features) : movie feature matrix
      W       (num_users  × num_features) : user preference matrix
      b       (1 × num_users)             : user bias vector
      Y       (num_movies × num_users)    : ratings matrix
      R       (num_movies × num_users)    : rated indicator matrix
      lambda_ (float)                     : regularization strength

    Returns:
      J (float) : total cost
    """
    nm, nu = Y.shape
    J = 0

    # Main cost — only where R[i,j] = 1
    for j in range(nu):
        for i in range(nm):
            if R[i, j] == 1:
                prediction = np.dot(W[j], X[i]) + b[0, j]
                J += 0.5 * (prediction - Y[i, j]) ** 2

    # Regularization — keep W and X values small
    for j in range(nu):
        J += (lambda_ / 2) * np.sum(W[j] ** 2)
    for i in range(nm):
        J += (lambda_ / 2) * np.sum(X[i] ** 2)

    return J


# ============================================================
#  COST FUNCTION — VECTORIZED VERSION
#  Used in training — computes all users and movies at once
#  Compatible with TensorFlow GradientTape
# ============================================================

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Collaborative filtering cost function — vectorized version.

    predictions = X @ W.T + b         shape (num_movies × num_users)
    error       = (predictions - Y)*R  zero out unrated pairs
    J = 0.5 * sum(error²) + regularization

    Args:
      X       (num_movies × num_features) : movie feature matrix
      W       (num_users  × num_features) : user preference matrix
      b       (1 × num_users)             : user bias vector
      Y       (num_movies × num_users)    : ratings matrix
      R       (num_movies × num_users)    : rated indicator matrix
      lambda_ (float)                     : regularization strength

    Returns:
      J (float) : total cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J


# ============================================================
#  INITIALIZE PARAMETERS
#  W, X → small random numbers from Gaussian distribution
#  b    → zeros (safe starting point for bias)
# ============================================================

tf.random.set_seed(42)
W = tf.Variable(tf.random.normal([num_users,  num_features], stddev=0.1, dtype=tf.float32))
X = tf.Variable(tf.random.normal([num_movies, num_features], stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.zeros([1, num_users], dtype=tf.float32))

print(f"\nInitial W (user preferences):\n{W.numpy().round(3)}")
print(f"Initial X (movie features):\n{X.numpy().round(3)}")
print(f"Initial b (user bias):\n{b.numpy().round(3)}")


# ============================================================
#  TRAINING LOOP
#  GradientTape records operations → computes dJ/dW, dJ/dX, dJ/db
#  Adam optimizer updates W, X, b every iteration
# ============================================================

optimizer  = keras.optimizers.Adam(learning_rate=0.1)
iterations = 200
lambda_    = 1.0

print(f"\nTraining...")
for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost = cofi_cost_func_v(
            X, W, b,
            tf.constant(Ynorm.T),   # shape (movies × users)
            tf.constant(R.T),
            lambda_)
    grads = tape.gradient(cost, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))
    if iter % 50 == 0:
        print(f"  Iteration {iter:3d} | Cost: {cost.numpy():.4f}")


# ============================================================
#  PREDICT
#  Use learned W, X, b to predict unrated movies
# ============================================================

predictions = tf.linalg.matmul(X, tf.transpose(W)) + b
predictions = tf.transpose(predictions).numpy()  # shape (users × movies)

print("\nPredicted ratings for unrated movies:")
print("-" * 40)
for i, user in enumerate(users):
    for j, movie in enumerate(movies):
        if R[i, j] == 0:
            score = predictions[i, j] + Ymean[j]  # restore mean
            print(f"  {user:8} → {movie:10}: {score:.2f} / 5")

print("\nFinal learned W (user emotional preferences):")
print(np.round(W.numpy(), 3))
print("\nFinal learned X (movie hidden features):")
print(np.round(X.numpy(), 3))
print("\nFinal learned b (user bias):")
print(np.round(b.numpy(), 3))
print("\nDone! Model converged successfully.")