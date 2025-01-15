import numpy as np

class MatrixDegradationException(Exception):
    pass

def reduce_sensitivity(A, max_condition_num, min_capture=.75):
    '''
    returns truncated SVD product of A, satisfying a condition
    number less than or equal to max_condition_num while remaining
    above a minimum % contribution of captured singular
    values (min_capture); returned value has an increased 
    effective minimum singular value (sing_min) rendering it 
    less numerically sensitive (lower condition number)
    '''

    if not A.dtype == 'float64':
        A = A.astype('float64')
    
    U, s, V_ = np.linalg.svd(A, full_matrices=False)
    ssum = sum(s)
    i = len(s)-1
    capture = 1
    discard = 0
    condition_num  = s[0]/s[i] #k(A) = sing_max/sing_min
    while condition_num >= max_condition_num and capture >= min_capture:
        discard = discard + s[i]
        i = i - 1
        condition_num = s[0]/s[i]
        capture = (ssum-discard)/ssum

    if capture < min_capture:
        raise MatrixDegradationException('Matrix is degraded at this condition number; try increasing it')

    return U[:, :i+1]@np.diag(s[:i+1])@V_[:i+1, :]
    





