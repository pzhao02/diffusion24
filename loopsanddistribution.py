import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Number of simulations
num_simulations = 10000

# List to store the last reconstructed value from each simulation
last_reconstructed_values = []

# Simulation loop
for sim in range(num_simulations):
    # Simulation parameters
    T = 100  # Total time steps
    beta = np.linspace(0.0001, 0.02, T)  # Variance schedule
    alpha = 1 - beta
    alpha_cumprod = np.cumprod(alpha)  # Cumulative product of alpha_t

    # Define a real number as the starting point
    x0 = np.random.normal(loc=5.0, scale=1)  # Initial real number (you can change this)
    real_numbers = []

    # Forward process to add noise
    xt = x0
    epsilons = []  # Store the epsilon noise for each time step
    xt_values = []  # Store the x_t values for polynomial fitting

    for t in range(T):
        # Generate epsilon noise
        epsilon_t = np.random.normal(0, 1)
        epsilons.append(epsilon_t)

        # Forward process: equation (8)
        xt = np.sqrt(alpha_cumprod[t]) * x0 + np.sqrt(1 - alpha_cumprod[t]) * epsilon_t
        real_numbers.append(xt)
        xt_values.append([xt, t])  # Collect xt and t for regression

    epsilon_t_bar = (real_numbers[-1] - np.sqrt(alpha_cumprod[-1]) * x0) / np.sqrt(
        1 - alpha_cumprod[-1]
    )

    # Prepare polynomial regression to predict epsilon_theta as a function of x_t and t
    poly = PolynomialFeatures(degree=2, include_bias=False)
    xt_poly = poly.fit_transform(
        xt_values
    )  # Transform x_t and t into polynomial features

    # Fit a linear regression model to predict epsilon_t based on x_t and t
    regressor = LinearRegression()
    regressor.fit(xt_poly, epsilons)  # Fit the model to predict epsilon_t

    # Backward process using Equation (18) to find mu_q
    xt = real_numbers[-1]  # Start from the final noised value (xt at time T)
    reconstructed_values = []  # Store the reconstructed values for each backward step

    for t in reversed(range(1, T)):
        # Polynomial features for the current xt and t
        xt_poly_t = poly.transform([[xt, t]])

        # Predict epsilon_theta using the polynomial regression model
        epsilon_theta = regressor.predict(xt_poly_t)[0]

        # Equation (18) for mu_q calculation
        epsilon_t = epsilons[t]
        mu_q = (
            1
            / np.sqrt(alpha[t])
            * (xt - (1 - alpha[t]) / np.sqrt(1 - alpha_cumprod[t]) * epsilon_t)
        )
        mu_q_theta = (
            1
            / np.sqrt(alpha[t])
            * (xt - (1 - alpha[t]) / np.sqrt(1 - alpha_cumprod[t]) * epsilon_theta)
        )

        # Equation (19): backward denoising process
        sigma_q_t = np.sqrt(
            (1 - alpha_cumprod[t - 1]) * (1 - alpha[t]) / (1 - alpha_cumprod[t])
        )
        z = np.random.normal(0, 1)  # Standard Gaussian noise
        xt = mu_q_theta + sigma_q_t * z  # Update xt for the next step
        reconstructed_values.append(xt)

    # Store the last reconstructed value of this simulation
    last_reconstructed_values.append(reconstructed_values[-1])

# Print or analyze the 100 last reconstructed values
print(last_reconstructed_values)
# plt.figure(figsize=(8, 6))
# plt.hist(last_reconstructed_values, bins=15, edgecolor="black", alpha=0.7)
# plt.title("Histogram of Last Reconstructed Values from 100 Simulations")
# plt.xlabel("Last Reconstructed Value")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()
# Plot the histogram of last reconstructed values
plt.figure(figsize=(8, 6))
plt.hist(
    last_reconstructed_values,
    bins=15,
    edgecolor="black",
    alpha=0.7,
    density=True,
    label="Histogram",
)

# Generate values for the normal density curve
mean = 5
std_dev = 1
x = np.linspace(min(last_reconstructed_values), max(last_reconstructed_values), 100)
normal_density = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x - mean) / std_dev) ** 2
)

# Plot the normal density curve
plt.plot(
    x,
    normal_density,
    color="red",
    label=f"Normal Distribution (mean={mean}, std={std_dev})",
)

# Add titles and labels
plt.title("Histogram with Normal Density Curve")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
