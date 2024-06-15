import numpy as np

class SingularMatrixException(Exception):
    pass

def linear_solver(A, b):
    A = A.astype('f') 
    b = b.astype('f')
    n = len(b) 

    def gauss_elim(tol=1e-12):
        '''transforms A into upper triangular form via Gauss-elimination row operations;
        incorporates Gauss pivoting to maintain diagonal dominance at every step''' 

        s = [max(abs(A[i])) for i in range(n)] #greatest element of each row of A

        for k in range(0, n-1): 
            #row interchange if need:
            rel_sizes = [abs(A[j,k])/s[j] for j in range(k, n)]
            max_rel_size = max(rel_sizes)
            if max_rel_size == A[k,k]/s[k]:
                    print('no swap')
                    pass
            else:
                r = rel_sizes.index(max_rel_size) + k 
                print('r=', r)
                if abs(A[r,k]) < tol:
                    raise SingularMatrixException('Matrix is singular, cannot proceed')
                swap = (A[r].copy(), b[r], s[r])    #.copy() creates a shallow copy of the array
                A[r], b[r], s[r] = (A[k], b[k], s[k])
                A[k], b[k], s[k] = swap

            print(A)
            print(s)
            print()

            pivot = A[k,k]
            for j in range(k+1, n):
                if A[j,k] != 0.0: 
                    front = A[j, k]
                    A[j, k:] = A[j, k:] - (front/pivot)*A[k, k:] 
                    b[j] = b[j] - (front/pivot)*b[k]    
        return A, b 


    def back_substitute(U, c): 
        x = np.zeros(n)
        for k in range(n-1, -1, -1): 
            x[k] = (c[k] - np.dot(U[k, k+1:], x[k+1:]))/U[k,k]
        return x

    
    return back_substitute(*gauss_elim())
