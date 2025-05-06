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

    V = np.random.exponential(scale=10e-4, size=N)
    weights = V / np.sum(V)
    # weights *= N

    for iteration in range(max_iter):
        new_x = np.zeros_like(x)
        for i in range(n):
            xi = x[i]
            # Compute contributions from other support points (exclude index i)
            x_others = np.delete(x, i, axis=0)  # shape (n-1, d)
            diff_support = xi - x_others  # shape (n-1, d)
            # Compute Euclidean norms of differences along the rows
            norms_support = np.linalg.norm(diff_support, axis=1) + eps  # shape (n-1,)
            term_support = np.sum(
                diff_support / norms_support.reshape(-1, 1), axis=0
            )  # shape (d,)

            # Compute contributions from the fixed sample batch y (all MNIST images)
            diff_y = xi - y  # shape (N, d)
            norms_y = np.linalg.norm(diff_y, axis=1) + eps  # shape (N,)
            # term_y = np.sum(y / norms_y.reshape(-1, 1), axis=0)  # shape (d,)
            term_y = N * np.sum((weights[:, None] * y) / norms_y[:, None], axis=0)

            # denom_y = np.sum(1.0 / norms_y)
            denom_y = N * np.sum(weights / norms_y)

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
x_train_9 = x_train
y_train_9 = y_train
# Normalize images to [0, 1]
x_train_9 = x_train_9.astype('float32') / 255.0
# Flatten each image from 28x28 to 784 dimensions
x_train_9 = x_train_9.reshape(-1, 28 * 28)

N = x_train_9.shape[0]  # number of MNIST images
# print(x_train_9.shape)
n = 100  # desired number of support points

# Two initializations for support points:
# Initialization 1: Randomly select n images from the dataset.
# indices = np.random.choice(N, n, replace=False)
# x1 = x_train[indices].copy()

# Initialization 2: Uniform random initialization over [0, 1] for each pixel.
x2 = np.random.uniform(0, 0.01, size=(n, 28 * 28))

# Compute support points using the modified CCP algorithm
# support_pts1 = support_points_ccp(x_train_9, x1, n)
support_pts2 = support_points_ccp(x_train_9, x2, n)
mn, mx = support_pts2.min(), support_pts2.max()
support_pts2_rescaled = (support_pts2 - mn) / (mx - mn)
print(support_pts2_rescaled.max(), support_pts2_rescaled.min())
support_pts2_bin = (support_pts2_rescaled >= 0.5).astype('float32')

# print(support_pts2[:10,])
# Visualization: plot support points as 28x28 images in a grid.
def plot_support_points(support_points, title):
    grid_size = int(np.sqrt(len(support_points)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(support_points[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.show()


# plot_support_points(support_pts1, "Support Points from x1 Initialization")
# plot_support_points(support_pts2, "Support Points from x2 Initialization")
plot_support_points(support_pts2_bin, "Crisp Support Points")

# # Load the MNIST dataset (raw pixel values in the range [0, 255])
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Reshape images to (num_samples, 28, 28, 1)
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# # Convert labels to one-hot encoding
# y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
# y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# # Define a simple CNN model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train_cat, batch_size=128, epochs=5, validation_split=0.1)

# # Evaluate the model on the MNIST test set
# score = model.evaluate(x_test, y_test_cat, verbose=0)
# print(f"Test accuracy on raw MNIST data: {score[1] * 100:.2f}%")

# # Reshape generated images to (n, 28, 28, 1)
# generated_images = support_pts2.reshape(-1, 28, 28, 1)

# # Predict labels using the trained classifier
# predictions = model.predict(generated_images)
# predicted_labels = np.argmax(predictions, axis=1)

# # Since these images are meant to be digit 9, create the ground truth labels accordingly.
# ground_truth = 9 * np.ones_like(predicted_labels)

# # Compute accuracy: the percentage of generated images that the classifier labels as '9'
# accuracy = np.mean(predicted_labels == ground_truth) * 100
# print(f"Classifier accuracy on generated images: {accuracy:.2f}%")
