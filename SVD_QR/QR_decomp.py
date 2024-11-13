import numpy as np

def QR_decomp(A):
    if not A.dtype == 'float64':
        A = A.astype('float64')
    n, m = A.shape
    Q = np.empty((n,n))
    R = np.zeros((n, m))
    for i in range(n):
        u = A[:, i]
        for j in range(i-1, -1, -1):
            R[j, i] = np.dot(A[:, i], Q[:, j])
            u = u - R[j, i]*Q[:, j]
        Q[:, i] = u/np.linalg.norm(u)
        R[i, i] = np.dot(A[:, i], Q[:, i] )
    if m > n:
        for i in range(n, m):
            for j in range(n-1, -1, -1):
                R[j, i] = np.dot(A[:, i], Q[:, j])
    return Q, R
    