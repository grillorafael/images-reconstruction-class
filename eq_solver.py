from sympy.solvers import solve
from sympy import Symbol

import sys

a = sys.argv[1]
b = sys.argv[2]
c = sys.argv[3]
d = sys.argv[4]
fe = sys.argv[5]
fl = sys.argv[6]

# print sys.argv

x = Symbol('x')
eq = ((x ** 2) / (1 + (fe * x) **2 )) + ( ((c * x + d) ** 2) / (((a * x + b) ** 2) + (fl ** 2) * ((c * x + x) ** 2)));
result = solve(eq, x)

print result
