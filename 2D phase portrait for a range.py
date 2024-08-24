import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def x_dot(x, y):
    return y

def y_dot(x, y):
    return -np.sin(x)

def rk4_step(x, y, x_dot, y_dot, h):
    k1 = x_dot(x, y)
    l1 = y_dot(x, y)

    k2 = x_dot(x + 0.5 * h * k1, y + 0.5 * h * l1)
    l2 = y_dot(x + 0.5 * h * k1, y + 0.5 * h * l1)

    k3 = x_dot(x + 0.5 * h * k2, y + 0.5 * h * l2)
    l3 = y_dot(x + 0.5 * h * k2, y + 0.5 * h * l2)

    k4 = x_dot(x + h * k3, y + h * l3)
    l4 = y_dot(x + h * k3, y + h * l3)

    K = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    L = (l1 + 2 * l2 + 2 * l3 + l4) / 6

    x += K * h
    y += L * h

    return x, y

def phase_portrait(init, T, x_dot, y_dot, h=0.01):
    x, y = init
    t_values = np.arange(0, T, h)
    x_values = [x]
    y_values = [y]
    plt.plot(x , y, 'bo', markersize = 3)
    for t in t_values[1:]:
        x, y = rk4_step(x, y, x_dot, y_dot, h)
        x_values.append(x)
        y_values.append(y)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.plot(x_values, y_values, 'b-', lw = .8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

plt.figure(figsize=(10, 8))
t =20

range = 5
step = .25

x_range = np.arange(-range, range, step)
y_range = np.arange(-range, range, step)

for x, y in product(x_range, y_range):
    phase_portrait((x, y), 1, x_dot, y_dot)

plt.title('Phase Portraits for Different Initial Conditions')
plt.show()
