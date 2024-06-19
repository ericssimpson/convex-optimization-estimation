import matplotlib.pyplot as plt
import numpy as np
from .optimization import solve_convex_set

def plt_halfspace(a, b, bbox, ax):
    if a[1] == 0:
        ax.axvline(b / a[0])
    else:
        x = np.linspace(bbox[0][0], bbox[0][1], 100)
        ax.plot(x, (b - a[0]*x) / a[1])

def plot_convex_set(A, b, bbox, ax=None):
    points, interior_point, hs = solve_convex_set(A, b, bbox, ax=ax)
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(bbox[0])
    ax.set_ylim(bbox[1])
    ax.fill(points[:, 0], points[:, 1], 'r')
    return points, interior_point, hs

def plot_inequalities(A, b, c, bbox, ax=None):
    points, interior_point, hs = plot_convex_set(A, b, bbox, ax=ax)
    for a_k, b_k in zip(A, b):
        plt_halfspace(a_k, b_k, bbox, ax)
    return points, interior_point, hs
