import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the image matrix
file_path = "/Users/peiqiz/DATA/M_1.csv"
matrix = pd.read_csv(file_path, header=None).values
matrix = matrix[1:, :].astype(float)  # Assuming first row is headers

# Image dimensions
img_height, img_width = matrix.shape
dim_X = img_width
n_samples = img_height
p = 20  # Basis functions
noise_std = 100.0  # Noise level

# Generate noisy observations
np.random.seed(42)
epsilon_img = np.random.normal(1, noise_std, (n_samples, dim_X))
Y_img = matrix + epsilon_img


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
print(theta_hat_img)
# Predict and reconstruct
epsilon_hat_img = (Phi_img @ theta_hat_img).reshape(n_samples, dim_X)
X_hat_img = Y_img - epsilon_hat_img

# Plot images
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(matrix, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(Y_img, cmap="gray")
axs[1].set_title("Noisy Image")
axs[1].axis("off")

axs[2].imshow(X_hat_img, cmap="gray")
axs[2].set_title("Reconstructed Image")
axs[2].axis("off")

plt.show()
