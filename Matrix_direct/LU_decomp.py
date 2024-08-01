import numpy as np

def LU_decomp(A):
    '''applies Doolittle's decomposition to A; results can be stored for later'''
    A = A.astype('f')
    n = len(A[0])
    L = np.zeros((n,n))
    for k in range(0, n-1):
        pivot = A[k,k]
        L[k,k] = 1 #Dolittle's L matrix has 1's along diagonal
        for j in range(k+1, n):
            if A[j,k] != 0:
                front = A[j, k]
                lam = front/pivot
                A[j, k:] = A[j, k:] - lam*A[k, k:]
                L[j,k] = lam #replaces initialized zero below the diagonal
    L[n-1,n-1] = 1
    return L, A    #A is transformed to upper trian; L is lower trian comprised of corresponding Gauss multipliers




def tri_solver(T, v, upper=True):
    '''solves Tu = v by substitution methods where T must be either upper or lower triangular'''
    T = T.astype('f')
    v = v.astype('f')
    n = len(v)
    u = np.zeros(n)
    if upper:
        for k in range(n-1, -1, -1):
            u[k] = (v[k] - np.dot(T[k, k+1:],u[k+1:]))/T[k,k]
    else: #if lower
        for k in range(0, n):
            u[k] = (v[k] - np.dot(T[k, :k], u[:k]))/T[k,k]  
    return u






