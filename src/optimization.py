import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
import pulp

def feasible_point(A, b):
    norm_vector = np.linalg.norm(A, axis=1)
    A_ = np.hstack((A, norm_vector[:, None]))
    b_ = b[:, None]
    c = np.zeros((A.shape[1] + 1,))
    c[-1] = -1
    res = linprog(c, A_ub=A_, b_ub=b[:, None], bounds=(None, None))
    return res.x[:-1]

def hs_intersection(A, b):
    interior_point = feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    return hs

def solve_convex_set(A, b, bbox):
    A_, b_ = add_bbox(A, b, *bbox)
    interior_point = feasible_point(A_, b_)
    hs = hs_intersection(A_, b_)
    points = hs.intersections
    hull = ConvexHull(points)
    return points[hull.vertices], interior_point, hs

def generate_linear_program(n, m, lower_entry_bound, upper_entry_bound):
    a_matrix = np.random.randint(lower_entry_bound, upper_entry_bound, size=(m, n))
    b_vector = np.random.randint(lower_entry_bound, upper_entry_bound, size=(m))
    c_vector = np.random.randint(lower_entry_bound, upper_entry_bound, size=(n))
    for i in range(len(b_vector)):
        if b_vector[i] <= 0:
            b_vector[i] = b_vector[i] * -1
            a_matrix[i] = a_matrix[i] * -1
    return a_matrix, b_vector, c_vector
