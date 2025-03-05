import numpy as np
import matplotlib.pyplot as plt


def support_points_ccp(y, x, n, max_iter=500, tol=1e-6):
    """
    Compute support points using the convex-concave procedure (sp.ccp)
    following the closed-form update rule in Equation (22) of the paper.

    Parameters:
      y : numpy array
          Fixed sample batch drawn from the target distribution F (size N).
      n : int
          Desired number of support points.
      max_iter : int
          Maximum number of iterations.
      tol : float
          Convergence tolerance.

    Returns:
      x : numpy array
          Final set of n support points.
    """
    N = len(y)
    # # Initialize support points by randomly selecting n distinct points from y
    # x1 = np.random.choice(y, n, replace=False).astype(float)
    # x2 = np.random.uniform(0, 0.000000001, n)
    eps = 1e-8  # small constant to prevent division by zero

    for iteration in range(max_iter):
        x_old = x.copy()
        new_x = np.zeros_like(x)

        for i in range(n):
            xi = x[i]

            # Compute contributions from the other support points (exclude index i)
            x_others = np.delete(x, i)
            diff_support = np.abs(xi - x_others)
            # Term from support points: sum_{j != i} (xi - x_j)/|xi - x_j|
            term_support = np.sum((xi - x_others) / (diff_support + eps))
            # Denom from support points: sum_{j != i} 1/|xi - x_j|
            # denom_support = np.sum(1.0 / (diff_support + eps))

            # Compute contributions from the fixed sample batch y
            diff_y = np.abs(xi - y)
            term_y = np.sum(y / (diff_y + eps))
            denom_y = np.sum(1.0 / (diff_y + eps))

            # Combine with weighting factor (N/n) for the support point contributions
            numerator = (N / n) * term_support + term_y
            # denominator = (N / n) * denom_support + denom_y
            denominator = denom_y

            new_x[i] = numerator / (denominator + eps)

        # Check convergence
        if np.max(np.abs(new_x - x)) < tol:
            print(f"Converged in {iteration} iterations.")
            x = new_x
            break
        x = new_x

    return x


# Main script
np.random.seed(42)  # For reproducibility
N = 1000  # Number of samples from standard normal distribution
n = 100  # Desired number of support points

# Generate 1000 samples from a standard normal distribution
y_samples = np.random.randn(N)
# Initialize support points by randomly selecting n distinct points from y
x1 = np.random.choice(y_samples, n, replace=False).astype(float)
x2 = np.random.uniform(0, 0.000000001, n)

# Compute support points using the CCP algorithm
support_pts1 = support_points_ccp(y_samples, x1, n)
support_pts2 = support_points_ccp(y_samples, x2, n)

# Plot the results: histogram of the 1000 samples and the support points overlaid.
plt.figure(figsize=(10, 5))
# plt.hist(y_samples, bins=30, density=True, alpha=0.5, label="Standard Normal Samples")
# # Plot support points along the x-axis with zero height (for visualization)
# # plt.scatter(np.sort(support_pts), np.zeros_like(support_pts), color='red', label="Support Points", zorder=5)
# plt.hist(support_pts1, bins=30, density=True, alpha=0.5, label="Support Points")
# plt.hist(support_pts2, bins=30, density=True, alpha=0.5, label="Support Points")
plt.scatter(
    np.sort(support_pts1),
    np.sort(support_pts2),
    color="red",
    label="Support Points",
    zorder=5,
)
plt.title("Support Points via CCP")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
