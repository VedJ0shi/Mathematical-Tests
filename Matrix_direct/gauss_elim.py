import numpy as np

def linear_solver(A, b):
    if not A.dtype == 'float64':
        A = A.astype('float64') 
    if not b.dtype == 'float64':
        b = b.astype('float64')
    n = len(b) #also the number of rows in A; rows/columns are zero-indexed 

    def gauss_elim():
        '''transforms A into upper triangular form via Gauss-elimination row operations'''       
        
        for k in range(0, n-1): #from 0 to n-2 
            pivot = A[k, k]
            for j in range(k+1, n):
                if A[j,k] != 0.0: #checking all leading non-zero terms of kth column
                    front = A[j, k]
                    A[j, k:] = A[j, k:] - (front/pivot)*A[k, k:] #replace entire row
                    b[j] = b[j] - (front/pivot)*b[k]    
        return A, b #transformed


    def back_substitute(U, c): #expects U to be upper triangular
        x = np.zeros(n)
        for k in range(n-1, -1, -1): #from n-1 to 0
            x[k] = (c[k] - np.dot(U[k, k+1:], x[k+1:]))/U[k,k]
        return x

    
    return back_substitute(*gauss_elim())











