import numpy as np

def blur_interior(arr):
    '''transforms each interior pixel of arr to the average of itself and non-diagonal neighbors'''
    new = np.zeros(arr.shape)
    interior = arr[1:-1,1:-1]
    shifted_up = arr[ :-2, 1:-1]
    shifted_down = arr[2: ,1:-1 ]
    shifted_left = arr[1:-1, :-2]
    shifted_right = arr[1:-1, 2: ]
    blurred_interior = (interior + shifted_up + shifted_down + shifted_right + shifted_left)/5
    new[1:-1, 1:-1] = blurred_interior
    return new


