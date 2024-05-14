import numpy as np
from numpy import dot, inner, outer

u = np.array([7,3])
v = np.array([2,1])
A = np.array([[1,2],[3,2]])
B = np.array([[1,1],[2,2]])
print(u, v)
print(A)
print(B)
print()

print(dot(u, v)) #standard inner product on vectors
print(dot(A, u)) #standard matrix operating on a vector
print(dot(A,B)) #standard matrix multiplication
print()

print(inner(u, v)) #standard inner product on vectors <v|u>
print(inner(A, u)) # standard matrix operating on vector A|u> 
print(inner(A, B)) # ATranspose(B)
print()

print(outer(u, v)) #standard outerproduct on vectors |u><v|
print(outer(v, u))
print(outer(A, u)) 
print(outer(A, B))