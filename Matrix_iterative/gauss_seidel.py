import numpy as np


class IterativeMethodException(Exception):
    pass


def gaussseidel_solver(A, b, max_iterations=50):
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
        old = x.copy()
        if iteration > max_iterations:
            raise IterativeMethodException('Cannot complete the iterations, limit reached')
        def sub_iterate(i=0): #main step of algorithm (successive displacements)
            x[i] = np.dot(J[i], x) + c[i]
            if i == n - 1:
                return x
            return sub_iterate(i=i+1)
        x_next = sub_iterate()
        print(x_next)
        if np.linalg.norm(old - x_next, ord=np.inf)/np.linalg.norm(old, ord=np.inf) < tol:
            return x_next
        return iterate(x_next, iteration=iteration+1)
    
    guess = np.ones(n)
    return iterate(guess)
 
