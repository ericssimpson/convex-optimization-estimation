import pulp as pl
import numpy as np

def generate_uniformly_random_linear_program(min=np.random.choice([0,1]), decision_variables=np.random.randint(-10,10), constraints=np.random.randint(-10,10), iteration=0):
    #? Randomly Determines Whether To Generate A Maximize or Minimize Problem
    if min == 1:
        linear_program = pl.LpProblem(f"Primal_Linear_Program_{iteration}", pl.LpMinimize)
    else:
        linear_program = pl.LpProblem(f"Primal_Linear_Program_{iteration}", pl.LpMaximize)

    #? Generates a Dictionary of n Decision Variables Then Randomizes Upper or Lower Bound From Zero
    decision_variables = pl.LpVariable.dicts("x", range(n))
    for variable in decision_variables:
        lower = np.random.choice([0,1])
        if lower == 1:
            decision_variables[variable].lowBound = 0
        else:
            decision_variables[variable].upBound = 0

    #? For Each Decision Varible Create a Random Constant Such That: 
    #? Objective Function = (Variable 1, Random Constant 1) + (Variable 2, Random Constant 2) + ... + (Variable i, Random Constant i)
    objective_function = pl.LpAffineExpression([(decision_variables[i], np.random.randint(-10,10)) for i in range(len(decision_variables))])

    #? Adds Objective Function To The Linear Program
    linear_program += objective_function

    #? Generates m Constraints With Random Constants Similar To Objective Function
    for i in range(m):
        temporary_function = pl.LpAffineExpression([(decision_variables[i], np.random.randint(-10,10)) for i in range(len(decision_variables))])
        temporary_constraint = pl.LpConstraint(temporary_function, np.random.choice([-1,1]), rhs=np.random.randint(-10,10)) #? Constraint Such That >= or <= Random Constant
        linear_program += temporary_constraint

    '''
    #? Checks if Linear Program is Unbounded
    solver = pulp.PULP_CBC_CMD()
    result = linear_program.solve(solver)
    status = pulp.LpStatus[linear_program.status]
    
    if status == "Unbounded":
        #? Generating the Dual of the Primal if Unbounded To Get Infeasible
        if linear_program.sense == 1: #? Setting Max or Min Problem | 1 == Minimize, -1 == Maximize
            dual = pl.LpProblem(f"Dual_Linear_Program_{iteration}", pl.LpMaximize)
        else:
            dual = pl.LpProblem(f"Dual_Linear_Program_{iteration}", pl.LpMinimize)
        
    '''

    return linear_program


distribution = [0,0,0,0] #? Feasible, Unbounded, Infeasible, Error

n, m = 2, 2
for i in range(100):
    temporary_lp = generate_uniformly_random_linear_program(decision_variables=n, constraints=m)
    solver = pl.PULP_CBC_CMD()
    result = temporary_lp.solve(solver)
    status = pl.LpStatus[temporary_lp.status]
    if status == "Optimal":
        distribution[0] += 1
    elif status == "Unbounded":
        distribution[1] += 1
    elif status == "Infeasible":
        distribution[2] += 1
    else:
        distribution[3] += 1
    temporary_lp = None

print(distribution)