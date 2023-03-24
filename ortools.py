from ortools.linear_solver import pywraplp


def LpProblemSolver():

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

    #initialize  the two variables
    x = solver.NumVar(0, solver.infinity(), 'x')
    y = solver.NumVar(0, solver.infinity(), 'y')

    #set the constraints
    n = int(input("number of constraints"))
    for i in range(n):

        a = int(input("Constraint for x "))
        b = int(input("Constraint for y "))
        comparison = int(input(">= ?"))
        comparison_number = int(input("comparison number ?"))
        if comparison == 0:
            solver.Add(a*x + b*y >= comparison_number)
        else:
            solver.Add(a*x + b*y <= comparison_number)


    # enter the objective function
    c = int(input("a"))
    d = int(input("b"))
    q = int(input("maximise ?"))
    if q == 0:
        solver.Maximize(c * x + d * y)
    else:
        solver.Minimize(c * x + d * y)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        print('x =', x.solution_value())
        print('y =', y.solution_value())
    else:
        print('error problem doesnt have optimal solution')