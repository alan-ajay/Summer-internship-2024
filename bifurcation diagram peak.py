import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from tqdm import tqdm

# Parameters for the Lorenz system
sigma = 10.0
b = 8.0 / 3.0
r_start = 50
r_stop = 250
r_step = 2
initial_state = [0, 1, 0]  # Initial conditions [x0, y0, z0]

# Time spans
t_transient = 50  # Time to let the system settle
t_runtime = 100  # Time to record the peaks
t_eval_transient = np.linspace(0, t_transient, 1000)
t_eval_runtime = np.linspace(0, t_runtime, 10000)

def lorenz(t, state, sigma, r, b):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (r - z) - y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

def find_peaks_in_y(r_values, t_transient, t_runtime, initial_state):
    results = []
    current_state = initial_state

    # Initialize progress bar
    progress_bar = tqdm(total=len(r_values), desc='Finding peaks')

    # Solver options for high precision
    solver_options = {
        'method': 'DOP853',  # Implicit solver suitable for stiff problems
        'rtol': 1e-6,  # Relative tolerance
        'atol': 1e-9,  # Absolute tolerance
        'max_step': 0.05  # Maximum step size
    }

    for r in r_values:
        try:
            # Solve for transient period
            sol_transient = solve_ivp(lorenz, [0, t_transient], current_state, args=(sigma, r, b))
            final_state = sol_transient.y[:, -1]

            if r == 202:
                print(final_state)

            # Solve for runtime period
            sol_runtime = solve_ivp(lorenz, [0, t_runtime], final_state, args=(sigma, r, b),
                                    t_eval=t_eval_runtime)
            y = sol_runtime.y[1]

            # Find peaks in y
            peaks, _ = find_peaks(y)
            peak_values = y[peaks]

            for y_peak in peak_values:
                results.append([r, y_peak])

            current_state = final_state
        except Exception as e:
            print(f"Error occurred for r = {r}: {e}")

        progress_bar.update(1)

    progress_bar.close()

    return np.array(results)

# Generate r values for forward and backward directions
r_values_ascend = np.arange(r_start, r_stop + r_step, r_step)
r_values_descend = np.arange(r_stop, r_start - r_step, -r_step)

# Find peaks of y for each r value in ascending and descending order
peak_results_ascend = find_peaks_in_y(r_values_ascend, t_transient, t_runtime, initial_state)
peak_results_descend = find_peaks_in_y(r_values_descend, t_transient, t_runtime, initial_state)

# Plotting the bifurcation diagram
plt.figure(figsize=(12, 8))

# Plot ascending values in red
plt.plot(peak_results_ascend[:, 0], peak_results_ascend[:, 1], 'r.', markersize=0.3, label='r increasing')

# Plot descending values in blue
plt.plot(peak_results_descend[:, 0], peak_results_descend[:, 1], 'b.', markersize=0.3, label='r decreasing')

plt.xlabel('r')
plt.ylabel('y (Peaks)')
plt.title('Bifurcation Diagram of the Lorenz System')
plt.legend()
plt.grid(True)
plt.show()
