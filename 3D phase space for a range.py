import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import math

#constants
sigma = 10
beta = 8/3 
rho = 20

# System of differential equations
def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = (x * (rho - z) - y)
    dzdt = (x * y - beta * z)
    return [dxdt, dydt, dzdt]

# Solving the system of ODEs
def phase_portrait_3d(init, T, function, h=0.01):
    t_eval = np.linspace(0, T, 10000)
    sol = solve_ivp(function,[0, T], init, t_eval = t_eval, method='RK45')
    return sol.y[0], sol.y[1], sol.y[2]

# Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Initial conditions
divisions = 5
range = 1
z_range = np.linspace(-range, range, divisions)
y_range = np.linspace(-range, range, divisions)
x_range = np.linspace(-range, range, divisions)
T = 50

#Plotting the phase portrait for different initial conditions
for x, y, z in product(x_range, y_range, z_range):
    x_vals, y_vals, z_vals = phase_portrait_3d([x, y, z], T, lorenz)
    ax.plot(x_vals, y_vals, z_vals, color='b', linewidth=0.3)

# Coordinates and labels for the fixed points
points = [
    (0, 0, 0, 'O'),
    (math.sqrt(beta * (rho - 1)), math.sqrt(beta * (rho - 1)), rho - 1, 'C+'),
    (-math.sqrt(beta * (rho - 1)), -math.sqrt(beta * (rho - 1)), rho - 1, 'C-')
]

# Plotting and labeling the fixed points
for x, y, z, label in points:
    ax.scatter(x, y, z, c='r', marker='o', s=100)
    ax.text(x+1, y+1, z+1, label, fontsize=12, color='red')

ax.set_title('3D Phase Portrait for Different Initial Conditions')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
