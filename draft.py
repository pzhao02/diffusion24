import numpy as np
import pandas as pd
import cv2

# Load the image matrix from CSV files
matrix = []
for n in range(1, 11):
    file_path = "/Users/peiqiz/DATA/M_" + str(n) + ".csv"
    img = pd.read_csv(file_path, header=None).values
    img = img[1:, :].astype(float)  # Assuming first row is headers
    matrix.append(img.flatten())
matrix = np.array(matrix)

# Image dimensions
img_height, img_width = matrix.shape
dim_X = img_width
n_samples = img_height
p = 1000  # Number of basis functions
noise_std = 100.0  # Noise level

# Generate noisy observations
np.random.seed(42)
epsilon_img = np.random.normal(1, noise_std, (n_samples, dim_X))
Y_img = matrix + epsilon_img

# Define cosine basis function
def cosine_basis(y, p):
    return np.array([np.cos(2 * np.pi * j * y / p) for j in range(1, p + 1)])

# Construct design matrix
Phi_img = np.array(
    [cosine_basis(Y_img[i, d], p) for i in range(n_samples) for d in range(dim_X)]
)

# Solve for theta_hat using the normal equation
Phi_T_img = Phi_img.T
theta_hat_img = np.linalg.inv(Phi_T_img @ Phi_img) @ Phi_T_img @ epsilon_img.flatten()

# Predict and reconstruct
epsilon_hat_img = (Phi_img @ theta_hat_img).reshape(n_samples, dim_X)
X_hat_img = Y_img - epsilon_hat_img

# For demonstration, select the first image and reshape (assuming 28x28 images)
img_shape = (28, 28)
orig_img = matrix[0, :].reshape(img_shape)
noisy_img = Y_img[0, :].reshape(img_shape)
recon_img = X_hat_img[0, :].reshape(img_shape)

# Normalize images to the 0-255 range and convert to uint8
def normalize_to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

orig_img = normalize_to_uint8(orig_img)
noisy_img = normalize_to_uint8(noisy_img)
recon_img = normalize_to_uint8(recon_img)

# Resize images for better viewing (e.g., scaling up by a factor of 10)
scale_factor = 10
new_size = (img_shape[1] * scale_factor, img_shape[0] * scale_factor)
orig_img_large = cv2.resize(orig_img, new_size, interpolation=cv2.INTER_NEAREST)
noisy_img_large = cv2.resize(noisy_img, new_size, interpolation=cv2.INTER_NEAREST)
recon_img_large = cv2.resize(recon_img, new_size, interpolation=cv2.INTER_NEAREST)

# Add labels to each image
cv2.putText(orig_img_large, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
cv2.putText(noisy_img_large, "Noisy", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
cv2.putText(recon_img_large, "Reconstructed", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

# Display the images in separate windows
cv2.imshow("Original Image", orig_img_large)
cv2.imshow("Noisy Image", noisy_img_large)
cv2.imshow("Reconstructed Image", recon_img_large)

cv2.waitKey(0)
cv2.destroyAllWindows()
