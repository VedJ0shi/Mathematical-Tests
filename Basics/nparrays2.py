import numpy as np
from numpy import sqrt, sin, diagonal, trace, argmax

'''arithmetic operations are broadcasted to all elements in an array '''

a = np.array([0, 4, 9, 16], float)
print(a/3)
print(a)
print(a-1)
print(sqrt(a))
print(sin(a))
print(argmax(a)) #index of largest element
print(list(a)) #since numpy arrays are iterables, they can be converted into a list

'''can perform math operations between individual elements''' 
print(a[0] + a[1] + a[2])
print(type(a[0])) #of type numpy.float (not a standard float)

'''can do matrix/2D array operations'''
M = np.array([[4,-2,1],[2,4,2], [1,-2,3]])
print(diagonal(M)) #prints array [4 4 3]
print(trace(M))