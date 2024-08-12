import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Define the equation of motion
def bead_on_loop(t, y, b, g, r, omega):
    phi, phi_dot = y
    phi_ddot = (-b * phi_dot - g * np.sin(phi) + r * omega**2 * np.sin(phi) * np.cos(phi)) / m
    return [phi_dot, phi_ddot]

# Set parameters and initial conditions
m = 1.0      # Mass of the bead
b = 0.9     # Damping coefficient
g = 9.81     # Acceleration due to gravity
r = 1.0      # Radius of the loop

# Define angular velocities
omega1 = 0.5
omega2 = np.sqrt(g / (r * np.cos(np.pi / 4)))

# Define initial conditions for all beads: [initial angle, initial angular velocity]
initial_conditions = [
    [np.pi * 3 / 4, 0.0],  # Starting at 3pi/4 radians
    [np.pi, 0.0],      # Starting at pi radians
    [np.pi * -3 / 4, 0.0]  # Starting at -3pi/4 radians
]

# Solve the differential equations for all initial conditions
t_span = (0, 10)  # Time span for the simulation
t_eval = np.linspace(*t_span, 1000)  # Time points at which to store the computed solution

solutions1 = [solve_ivp(bead_on_loop, t_span, ic, args=(b, g, r, omega1), t_eval=t_eval) for ic in initial_conditions]
solutions2 = [solve_ivp(bead_on_loop, t_span, ic, args=(b, g, r, omega2), t_eval=t_eval) for ic in initial_conditions]

# Function to create the animations
def create_animation(solutions, omega_value, ax, colors):
    hoop, = ax.plot([], [], 'k-', lw=2)
    beads = [ax.plot([], [], color, markersize=8)[0] for color in colors]
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        hoop.set_data([], [])
        for bead in beads:
            bead.set_data([], [])
        time_text.set_text('')
        return [hoop, *beads, time_text]

    def update(frame):
        hoop_x = r * np.cos(np.linspace(0, 2 * np.pi, 100))
        hoop_y = r * np.sin(np.linspace(0, 2 * np.pi, 100))
        hoop.set_data(hoop_x, hoop_y)

        for i, bead in enumerate(beads):
            angle = solutions[i].y[0, frame]
            bead_x = r * np.sin(angle)
            bead_y = -r * np.cos(angle)
            bead.set_data(bead_x, bead_y)

        time_text.set_text(time_template % (t_eval[frame]))
        return [hoop, *beads, time_text]

    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=25)
    return ani

# Create a combined figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Colors for the beads
colors = ['ro', 'bo', 'go']

# Set up animations for both cases
ani1 = create_animation(solutions1, omega1, ax1, colors)
ani2 = create_animation(solutions2, omega2, ax2, colors)

# Set titles and labels
ax1.set_xlim(-1.2 * r, 1.2 * r)
ax1.set_ylim(-1.2 * r, 1.2 * r)
ax1.set_aspect('equal')  # Set aspect ratio to be equal
ax1.set_title(f'$\omega = {omega1}$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.set_xlim(-1.2 * r, 1.2 * r)
ax2.set_ylim(-1.2 * r, 1.2 * r)
ax2.set_aspect('equal')  # Set aspect ratio to be equal
ax2.set_title(f'$\omega = {omega2:.2f}$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()

# Display the animations
plt.show()
