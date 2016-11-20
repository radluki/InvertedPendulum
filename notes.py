import numpy as np
from sympy.matrices import Matrix

print('Jordan form presentation')
a = np.array([[5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]])
m = Matrix(a)
P, J = m.jordan_form()
# P*J*P.inv()-a
print(a)
print(J)
print(P)
print(P*J*P.inv()-a)







