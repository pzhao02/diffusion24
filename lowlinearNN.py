import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_samples = 1000  # Number of observations
dim_X = 10  # X is now 10-dimensional
p = 40  # Number of basis functions per dimension
noise_std = 1.0  # Standard deviation of noise

# Generate true values X (n_samples, dim_X) and noisy observations Y
np.random.seed(42)
X = np.random.uniform(-10, 10, (n_samples, dim_X))  # True values (10D)
epsilon = np.random.normal(0, noise_std, (n_samples, dim_X))  # 10D Noise
Y = X + epsilon  # Observations


# Define Fourier basis functions (cosine and sine)
def fourier_basis(y, p):
    """Generate Fourier basis functions for each dimension of y."""
    features = []
    for j in range(1, p + 1):
        features.append(np.cos(2 * np.pi * j * y / p))
        features.append(np.sin(2 * np.pi * j * y / p))
    return np.array(features)


def cosine_basis(y, p):
    """Generate Cosine basis functions for y."""
    return np.array([np.cos(2 * np.pi * j * y / p) for j in range(1, p + 1)])


# Construct the design matrix Phi so that each scalar measurement (each entry of Y)
# gets its own row. Thus, Phi will have shape (n_samples*dim_X, p).
Phi = np.array(
    [cosine_basis(Y[i, d], p) for i in range(n_samples) for d in range(dim_X)]
)

epsilon_flat = epsilon.flatten()

# Explicit least-squares solution for theta_hat
Phi_T = Phi.T  # Transpose of Phi
print(Phi.shape)
theta_hat = np.linalg.inv(Phi_T @ Phi) @ Phi_T @ epsilon_flat  # Solve for all dims

# Predict epsilon_hat for all scalar measurements
epsilon_hat_flat = Phi @ theta_hat  # Shape (n_samples*dim_X,)
# Reshape epsilon_hat back to (n_samples, dim_X)
epsilon_hat = epsilon_hat_flat.reshape(n_samples, dim_X)

# Reconstruct X_hat from Y
X_hat = Y - epsilon_hat

# Plot true vs reconstructed X for first dimension
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 1], X_hat[:, 1], alpha=0.5, label="Reconstructed vs True (dim 1)")
plt.plot([-10, 10], [-10, 10], "r--", label="Perfect Reconstruction")
plt.xlabel("True X (dim 1)")
plt.ylabel("Reconstructed X_hat (dim 1)")
plt.title("Reconstruction of X from Y using Fourier Basis Functions")
plt.legend()
plt.show()

# # Plot the noise prediction for first dimension
# plt.figure(figsize=(10, 5))
# plt.scatter(
#     epsilon[:, 0], epsilon_hat[:, 0], alpha=0.5, label="Predicted vs True Noise (dim 1)"
# )
# plt.plot([-3, 3], [-3, 3], "r--", label="Perfect Prediction")
# plt.xlabel("True Noise (epsilon) (dim 1)")
# plt.ylabel("Predicted Noise (epsilon_hat) (dim 1)")
# plt.title("Noise Prediction using Fourier Basis Functions (First Dimension)")
# plt.legend()
# plt.show()
