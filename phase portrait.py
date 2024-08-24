import numpy as np
import matplotlib.pyplot as plt

def x_dot(x, v):
    return v 

def v_dot(x, v):
    return -x

def phase_plane(x_dot, v_dot, x_range, v_range, num_arrows):
    x_values = np.linspace(x_range[0], x_range[1], num_arrows)
    v_values = np.linspace(v_range[0], v_range[1], num_arrows)

    X, V = np.meshgrid(x_values, v_values)
    DX = x_dot(X, V)
    DV = v_dot(X, V)

    # M = np.hypot(DX, DV)
    # M[M == 0] = 1
    # DX /= M
    # DV /= M

    plt.quiver(X, V, DX, DV, angles='xy')
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('Phase Plane Plot')
    plt.grid()
    plt.show()

x_range = [-10, 10]
v_range = [-10, 10]
num_arrows = 20

phase_plane(x_dot, v_dot, x_range, v_range, num_arrows)
