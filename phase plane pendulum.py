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

def phase_portrait(init, T, x_dot, y_dot, h=0.01, clr = 'b',linewidth=0.5):
    x, y = init
    t_values = np.arange(0, T, h)
    x_values = [x]
    y_values = [y]
 
    # plt.plot(x,y,'bo')
    for t in t_values[1:]:  
        x, y = rk4_step(x, y, x_dot, y_dot, h)
        x_values.append(x)
        y_values.append(y)    
    # manifold_values = []
    # for x in x_range2:
    #   manifold_values.append(-np.log(-x))
    plt.ylim(-3, 3)
    plt.xlim(-7, 7)

    plt.plot(x_values, y_values, clr,linewidth=linewidth)
    # plt.plot(x_range2,manifold_values, 'r-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

plt.figure(figsize=(10, 8))
t =20

range = 5
step = .1

y_range = np.arange(-3, 3, step)
x_range = np.arange(-7, 7, step)
x_range2 = np.arange(-10,0,.001)

for x, y in product(x_range, y_range):
    phase_portrait((x, y), 1, x_dot, y_dot,linewidth=0.5)
phase_portrait((-np.pi+0.0001, 0.0001), t, x_dot, y_dot, clr = 'r',linewidth=1) 
phase_portrait((np.pi-0.0001, -0.0001), t, x_dot, y_dot, clr = 'r',linewidth=1) 
# # phase_portrait((0.0011, 0.01), t, x_dot, y_dot, clr = 'r') 
# phase_portrait((2.9, 2.15), t, x_dot, y_dot, clr = 'r',linewidth=1) 
plt.plot(0, 0, marker='o', markersize=10, markerfacecolor='red', markeredgewidth=2, markeredgecolor='red')
plt.plot(np.pi, 0, marker='o', markersize=10, markerfacecolor='None', markeredgewidth=2, markeredgecolor='red')
plt.plot(-np.pi, 0, marker='o', markersize=10, markerfacecolor='None', markeredgewidth=2, markeredgecolor='red')
plt.plot(2*np.pi, 0, marker='o', markersize=10, markerfacecolor='red', markeredgewidth=2, markeredgecolor='red')
plt.plot(-2*np.pi, 0, marker='o', markersize=10, markerfacecolor='red', markeredgewidth=2, markeredgecolor='red')
# plt.plot(3, 0, 'ro', label='(1, 0)')
plt.title('Phase Portrait for Different Initial Conditions')
plt.show()