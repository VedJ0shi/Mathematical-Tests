import numpy as np

M = np.array([[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10 , 11, 12],
             [13, 14, 15, 16]])

print(M[0]) #first row
print(M[0][1:])
print(M[0, 1:]) #equivalent to prev line

diff = M[0, 1:] - 2*M[1, 1:] #row operation
print(diff) #equivalent to array([2, 3, 4]) - 2*array([6, 7, 8])

trace = M[0, 0] + M[1, 1] + M[2, 2] + M[3, 3]
same = M[0][0] + M[1][1] + M[2][2] + M[3][3]
print(trace)
print(same)

col1 = M[0:, 0]
col2 = M[0:, 1]
print(col1)
print(col2)