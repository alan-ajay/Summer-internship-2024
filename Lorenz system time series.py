import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Lorenz system parameters
sigma = 10
r = 216.64
b = 8/3

# Lorenz system equations
def X_dot(t, X, sigma, r, b):
    x, y, z = X
    return [sigma * (y - x), x * (r - z) - y, x * y - b * z]

# Initial conditions and time span
init = [0.43527023,  22.43333004, 174.64552558]
t_span = (0, 50)
t_values = np.linspace(50, 55, 10000)  # Increased number of points for smoother plot
t_span2 = (50, 55)

# Solve the Lorenz system
sol1 = solve_ivp(X_dot, t_span, init, args=(sigma, r, b))
sol2 = solve_ivp(X_dot, t_span2, sol1.y[:, -1], args=(sigma, r, b), dense_output=True, t_eval=t_values)

# Extract the solution
y_values = sol2.t
x_values = sol2.y[1]

# Detect peaks in the y(t) time series
peaks, _ = find_peaks(x_values)  # Adjust height and distance as needed

# Extract peak values
peak_values = x_values[peaks]

# Create subplots
fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# Plot the peaks on the time series
ax2.plot(t_values, x_values, label=f'r = {r}, chaos', color='blue')
ax2.plot(t_values[peaks], x_values[peaks], 'ro', label='', markersize=5)
ax2.set_xlabel('t')
ax2.set_ylabel('y')
ax2.set_title('Detected Peaks in the Time Series')
ax2.legend( loc='upper right')
ax2.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Print peak values
print("Peak y values:", peak_values)
