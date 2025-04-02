import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def support_points_ccp(y, x, n, max_iter=500, tol=1e-4):
    """
    Compute support points using the convex-concave procedure (sp.ccp)
    for multi-dimensional data.

    Parameters:
      y : numpy array of shape (N, d)
          Fixed sample batch drawn from the target distribution (here, MNIST images).
      x : numpy array of shape (n, d)
          Initial support points.
      n : int
          Desired number of support points.
      max_iter : int
          Maximum number of iterations.
      tol : float
          Convergence tolerance.

    Returns:
      x : numpy array of shape (n, d)
          Final set of support points.
    """
    N = y.shape[0]
    eps = 1e-8  # small constant to prevent division by zero

    for iteration in range(max_iter):
        new_x = np.zeros_like(x)
        for i in range(n):
            xi = x[i]
            # Compute contributions from other support points (exclude index i)
            x_others = np.delete(x, i, axis=0)  # shape (n-1, d)
            diff_support = xi - x_others         # shape (n-1, d)
            # Compute Euclidean norms of differences along the rows
            norms_support = np.linalg.norm(diff_support, axis=1) + eps  # shape (n-1,)
            term_support = np.sum(diff_support / norms_support.reshape(-1, 1), axis=0)  # shape (d,)

            # Compute contributions from the fixed sample batch y (all MNIST images)
            diff_y = xi - y                       # shape (N, d)
            norms_y = np.linalg.norm(diff_y, axis=1) + eps  # shape (N,)
            term_y = np.sum(y / norms_y.reshape(-1, 1), axis=0)  # shape (d,)

            denom_y = np.sum(1.0 / norms_y)

            numerator = (N / n) * term_support + term_y
            new_x[i] = numerator / (denom_y + eps)

        # Check convergence using the maximum Euclidean distance between updates
        if np.max(np.linalg.norm(new_x - x, axis=1)) < tol:
            print(f"Converged in {iteration} iterations.")
            x = new_x
            break
        x = new_x

    return x

# Main script

# Load the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()
x_train_9 = x_train[y_train == 9]
y_train_9 = y_train[y_train == 9]
# # Normalize images to [0, 1]
# x_train = x_train.astype('float32') / 255.0
# Flatten each image from 28x28 to 784 dimensions
x_train_9 = x_train_9.reshape(-1, 28*28)

N = x_train_9.shape[0]  # number of MNIST images
# print(x_train_9.shape)
n = 100             # desired number of support points

# Two initializations for support points:
# Initialization 1: Randomly select n images from the dataset.
# indices = np.random.choice(N, n, replace=False)
# x1 = x_train[indices].copy()

# Initialization 2: Uniform random initialization over [0, 1] for each pixel.
x2 = np.random.uniform(0, 0.01, size=(n, 28*28))

# Compute support points using the modified CCP algorithm
# support_pts1 = support_points_ccp(x_train_9, x1, n)
support_pts2 = support_points_ccp(x_train_9, x2, n)

# Visualization: plot support points as 28x28 images in a grid.
def plot_support_points(support_points, title):
    grid_size = int(np.sqrt(len(support_points)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(support_points[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

# plot_support_points(support_pts1, "Support Points from x1 Initialization")
plot_support_points(support_pts2, "Support Points from x2 Initialization")
