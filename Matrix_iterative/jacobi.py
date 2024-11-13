import numpy as np


class IterativeMethodException(Exception):
    pass


def jacobi_solver(A, b, max_iterations=100):
    if not A.dtype == 'float64':
        A = A.astype('float64')
    if not b.dtype == 'float64':
        b = b.astype('float64')
    n = len(b)
    J = np.zeros((n,n))
    c = np.zeros(n)
    for i in range(n):
        if A[i,i] == 0:
            k = i + 1
            while k < n and A[k,i] == 0:
                k = k + 1
            if k == n:
                raise IterativeMethodException('No rows to swap with; cannot proceed with Jacobi method on this matrix')
            swap = (A[i].copy(), b[i])
            A[i], b[i] = A[k], b[k]
            A[k], b[k] = swap      
        row = [-A[i,j]/A[i,i] if j != i else 0 for j in range(n)]
        J[i] = row
        c[i] = b[i]/A[i,i]
    
    def iterate(x, iteration=0, tol=1e-8):
        if iteration > max_iterations:
            raise IterativeMethodException('Cannot complete the iterations, limit reached')
        x_next = np.dot(J,x) + c #main step in algorithm (simultaneous displacement)
        print(x_next)
        if np.linalg.norm(x - x_next, ord=np.inf)/np.linalg.norm(x, ord=np.inf) < tol:
            return x_next
        return iterate(x_next, iteration=iteration+1)

    guess = np.ones(n)
    return iterate(guess)
 


    

    