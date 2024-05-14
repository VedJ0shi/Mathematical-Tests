import numpy as np


'''arrays can be declared explicitly by passing a list into np.array(list, type)'''

vector = np.array([1, 2, 3])
floatvector = np.array([1, 2, 3], float) #can pass element type, too
matrix = np.array([[1, 2, 3], [4, 5, 6]])

print(vector)
print(floatvector)
print(matrix)

'''range can be emulated in arrays with np.arange(from, to, increment)'''
range_array = np.arange(0, 10, 2)
print(range_array)
print()

'''can access elements in array in the standard manner of any iterable'''
print(vector[0], vector[0:2])
print(matrix[1])
for n in vector:
    print(n)
print(iter(vector)) #associated iterator object