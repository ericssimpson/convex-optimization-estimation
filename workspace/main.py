import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog

import pulp

def feasible_point(A, b):
    # finds the center of the largest sphere fitting in the convex hull
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

def plt_halfspace(a, b, bbox, ax):
    if a[1] == 0:
        ax.axvline(b / a[0])
    else:
        x = np.linspace(bbox[0][0], bbox[0][1], 100)
        ax.plot(x, (b - a[0]*x) / a[1])

def add_bbox(A, b, xrange, yrange):
    A = np.vstack((A, [
        [-1,  0],
        [ 1,  0],
        [ 0, -1],
        [ 0,  1],
    ]))
    b = np.hstack((b, [-xrange[0], xrange[1], -yrange[0], yrange[1]]))
    return A, b

def solve_convex_set(A, b, bbox, ax=None):
    A_, b_ = add_bbox(A, b, *bbox)
    interior_point = feasible_point(A_, b_)
    hs = hs_intersection(A_, b_)
    points = hs.intersections
    hull = ConvexHull(points)
    return points[hull.vertices], interior_point, hs

def plot_convex_set(A, b, bbox, ax=None):
    # solve and plot just the convex set (no lines for the inequations)
    points, interior_point, hs = solve_convex_set(A, b, bbox, ax=ax)
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(bbox[0])
    ax.set_ylim(bbox[1])
    ax.fill(points[:, 0], points[:, 1], 'r')
    return points, interior_point, hs

def plot_inequalities(A, b, c, bbox, ax=None):
    # solve and plot the convex set,
    # the inequation lines, and
    # the interior point that was used for the halfspace intersections
    points, interior_point, hs = plot_convex_set(A, b, bbox, ax=ax)
    #ax.plot(*interior_point, 'o')
    print(interior_point)
    for a_k, b_k in zip(A, b):
        plt_halfspace(a_k, b_k, bbox, ax)
    return points, interior_point, hs

def generate_linear_program(n, m, lower_entry_bound, upper_entry_bound):

    a_matrix = np.random.randint(lower_entry_bound,upper_entry_bound,size = (m,n))
    b_vector = np.random.randint(lower_entry_bound,upper_entry_bound,size = (m))
    c_vector = np.random.randint(lower_entry_bound,upper_entry_bound,size = (n))

    # Making Ax <= b with b being positive
    for i in range(len(b_vector)):
        if b_vector[i] <= 0:
            b_vector[i] = b_vector[i] * -1
            a_matrix[i] = a_matrix[i] * -1
    
    return (a_matrix, b_vector, c_vector)

n = 2
m = 3

lower_entry_bound = -10
upper_entry_bound = 10

a_matrix, b_vector, c_vector = generate_linear_program(n,m,lower_entry_bound,upper_entry_bound)

print(f'A \n{a_matrix}\n')
print(f'b \n{b_vector}\n')
print(f'c \n{c_vector}\n')

plt.rcParams['figure.figsize'] = (10, 10)
bbox = [(-10, 10), (-10, 10)]
fig, ax = plt.subplots()
plot_inequalities(a_matrix, b_vector, c_vector, bbox, ax)

linear_program = pulp.LpProblem("Test Generation", pulp.LpMinimize)
#solver_list = pulp.listSolvers(onlyAvailable=True)
#print(solver_list)
solver = pulp.PULP_CBC_CMD()
#? Have to figure out a way to dynamically generate variables for LP or use a differnt method to check infeasible/unbounded
x1 = pulp.LpVariable('x1', cat='Continuous')
x2 = pulp.LpVariable('x2', cat='Continuous')
linear_program += c_vector[0] * x1 + c_vector[1] * x2, 'Z'
for i in range(len(a_matrix)):
    linear_program += a_matrix[i][0] * x1 + a_matrix[i][1] * x2 <= b_vector[i]
print(linear_program)
result = linear_program.solve(solver)