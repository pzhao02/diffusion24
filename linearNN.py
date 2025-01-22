import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_samples = 1000  # Number of observations
p = 10  # Number of basis functions
noise_std = 1.0  # Standard deviation of noise

# Generate true values X and noisy observations Y
np.random.seed(42)
X = np.random.uniform(-10, 10, n_samples)  # True values
epsilon = np.random.normal(0, noise_std, n_samples)  # Noise
Y = X + epsilon  # Observations


# Define Cosine basis functions
def cosine_basis(y, p):
    """Generate Cosine basis functions for y."""
    return np.array([np.cos(2 * np.pi * j * y / p) for j in range(1, p + 1)]).T


# Construct the design matrix Phi (n_samples x p)
Phi = np.array([cosine_basis(y, p) for y in Y])

# Explicit least-squares solution for theta_hat
Phi_T = Phi.T  # Transpose of Phi
theta_hat = np.linalg.inv(Phi_T @ Phi) @ Phi_T @ epsilon  # Least-squares solution

# Predict epsilon_hat for all samples
epsilon_hat = Phi @ theta_hat

# Reconstruct X_hat from Y
X_hat = Y - epsilon_hat

# Plot true vs reconstructed X
plt.figure(figsize=(10, 5))
plt.scatter(X, X_hat, alpha=0.5, label="Reconstructed vs True")
plt.plot([-10, 10], [-10, 10], "r--", label="Perfect Reconstruction")  # Diagonal line
plt.xlabel("True X")
plt.ylabel("Reconstructed X_hat")
plt.title("Reconstruction of X from Y using Cosine Basis Functions (Least Squares)")
plt.legend()
plt.show()

# Plot the noise prediction
plt.figure(figsize=(10, 5))
plt.scatter(epsilon, epsilon_hat, alpha=0.5, label="Predicted vs True Noise")
plt.plot([-3, 3], [-3, 3], "r--", label="Perfect Prediction")  # Diagonal line
plt.xlabel("True Noise (epsilon)")
plt.ylabel("Predicted Noise (epsilon_hat)")
plt.title("Noise Prediction using Cosine Basis Functions (Least Squares)")
plt.legend()
plt.show()
