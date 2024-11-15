import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T = 100  # Total time steps
beta = np.linspace(0.0001, 0.02, T)  # Variance schedule
alpha = 1 - beta
alpha_cumprod = np.cumprod(alpha)  # Cumulative product of alpha_t

# Define a real number as the starting point
x0 = 5.0  # Initial real number (you can change this)
real_numbers = []

# Forward process to add noise
xt = x0
epsilons = []  # Store the epsilon noise for each time step

for t in range(T):
    # Generate epsilon noise
    epsilon_t = np.random.normal(0, 1)
    epsilons.append(epsilon_t)

    # Forward process: equation (8)
    xt = np.sqrt(alpha_cumprod[t]) * x0 + np.sqrt(1 - alpha_cumprod[t]) * epsilon_t
    real_numbers.append(xt)

# Plot the forward process (diffusion) over time
plt.plot(real_numbers, label="Noised Real Number (Forward Process)")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("Forward Process: Adding Noise to Real Number")
plt.legend()
plt.show()

# Backward process using Equation (18) to find mu_q
xt = real_numbers[-1]  # Start from the final noised value (xt at time T)
x0_pred = xt / np.sqrt(
    alpha_cumprod[-1]
)  # A simple initial guess for x0 from equation (8)
print(x0_pred)
mu_qs = []  # Store mu_q values for each backward step
reconstructed_values = []  # Store the reconstructed values for each backward step

for t in reversed(range(1, T)):
    # Equation (18) for mu_q calculation
    epsilon_t = epsilons[t]
    mu_q = (
        1
        / np.sqrt(alpha[t])
        * (xt - (1 - alpha[t]) / np.sqrt(1 - alpha_cumprod[t]) * epsilon_t)
    )
    mu_qs.append(mu_q)

    # Equation (19): backward denoising process
    sigma_q_t = np.sqrt(
        (1 - alpha_cumprod[t - 1]) * (1 - alpha[t]) / (1 - alpha_cumprod[t])
    )
    z = np.random.normal(0, 1)  # Standard Gaussian noise
    xt = mu_q + sigma_q_t * z  # Update xt for the next step
    reconstructed_values.append(xt)

print(reconstructed_values[-1])
# Plot the backward process (denoising) over time
plt.plot(
    real_numbers + reconstructed_values[::1],
    label="Denoised Real Number (Backward Process)",
)
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("Backward Process: Denoising the Real Number")
plt.legend()
plt.show()
