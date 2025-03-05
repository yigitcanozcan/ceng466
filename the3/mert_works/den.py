from sympy import symbols, Eq, solve

# Define the variables
x, y = symbols('x y', real=True)

# Define the equations
eq1 = Eq(0.367*x-y, -64+64*0.367)    # 2x + 3y = 6
eq2 = Eq(0.768*x-y, -12+12*0.768)                      # x = 4
# Solve the system
solution = solve((eq1, eq2), (x, y))
print(solution)  # This will print {x: 4, y: 2/3} as a dictionary


