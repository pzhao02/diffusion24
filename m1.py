import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the image matrix
file_path = "/Users/peiqiz/DATA/M_1.csv"
matrix = pd.read_csv(file_path, header=None).values
matrix = matrix[1:, :].astype(float)  # Assuming first row is headers

# Normalize the image values to [-10, 10]
min_val, max_val = matrix.min(), matrix.max()
X_img = 20 * (matrix - min_val) / (max_val - min_val) - 10

# Image dimensions
img_height, img_width = X_img.shape
dim_X = img_width
n_samples = img_height
p = 40  # Basis functions
noise_std = 1.0  # Noise level

# Generate noisy observations
np.random.seed(42)
epsilon_img = np.random.normal(0, noise_std, (n_samples, dim_X))
Y_img = X_img + epsilon_img


# Define Cosine basis function
def cosine_basis(y, p):
    return np.array([np.cos(2 * np.pi * j * y / p) for j in range(1, p + 1)])


# Construct design matrix
Phi_img = np.array(
    [cosine_basis(Y_img[i, d], p) for i in range(n_samples) for d in range(dim_X)]
)

# Solve for theta_hat
Phi_T_img = Phi_img.T
theta_hat_img = np.linalg.inv(Phi_T_img @ Phi_img) @ Phi_T_img @ epsilon_img.flatten()

# Predict and reconstruct
epsilon_hat_img = (Phi_img @ theta_hat_img).reshape(n_samples, dim_X)
X_hat_img = Y_img - epsilon_hat_img

# Rescale back to original range
X_hat_rescaled = (X_hat_img + 10) / 20 * (max_val - min_val) + min_val

# Plot images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(Y_img, cmap="gray")
axs[0].set_title("Noisy Image")
axs[0].axis("off")

axs[1].imshow(X_hat_rescaled, cmap="gray")
axs[1].set_title("Reconstructed Image")
axs[1].axis("off")

plt.show()
