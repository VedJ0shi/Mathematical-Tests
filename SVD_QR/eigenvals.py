from QR_decomp import QR_decomp
import numpy as np

def make_similar(A):
    Q, R = QR_decomp(A)
    return R @ Q 

def find_eigenvals_QR(A, diff_target=1e-8):
    '''
    returns list of eigenvalues of A found from applying the iterative QR algorithm;
    eigenvalues will converge on the diagonal of the similar matrices
    '''
    A_ = make_similar(A) #A_=RQ is similar to A
    n = len(A) #A must be a square matrix
    corner = A_[-1, -1] #rightmost diag element converges fastest to eigenvalue
    diff = 1
    i = 0
    while diff > diff_target: #continues until convergence criteria satisfied
        A_ = make_similar(A_)
        diff = abs(corner - A_[-1,-1])
        corner = A_[-1,-1]
        i = i + 1
    print("# of iterations:", i)
    eigenvals = [A_[i,i] for i in range(n)] #A_ is the ith similar matrix computed in the iteration chain
    return eigenvals



      


    
