import numpy as np

def pca(dataset, min_variance_captured=.70):
    '''
    given a dataset matrix (data samples as rows), returns a 
    dimension-reduced dataset by projecting onto the least-squares
    subspace; new dimension is the smallest number of principal components
    to capture at least min_variance_captured %-variance of original dataset
    '''

    if not dataset.dtype == 'float64':
        dataset = dataset.astype('float64')
    
    mean_row = dataset.mean(axis=0)
    m = np.shape(dataset)[0] #number of samples
    n = np.shape(dataset)[1] #dimensionality of features
    X = dataset - np.array([mean_row for _ in range(m)]) #returns mean-centered dataset
    
    Cx = (1/(m-1))*(X.T @ X) #returns covariance matrix (square, Hermitian)
    eigvals, V = np.linalg.eig(Cx) #diagonalize to find principal components (unit eigvectors) and variances (eigvalues)
    order = np.argsort(eigvals)[::-1] #indices ordered corresponding to greatest to least
    variances = np.array([eigvals[i] for i in order]) #orders the variances
    V = np.array([V[:,i] for i in order]).T #orders the principal components

    j = 0
    total = sum(variances)
    variance_captured = variances[0] / total
    while variance_captured < min_variance_captured: #loop to compute how many components to keep
        j += 1
        variance_captured += variances[j] / total
    
    
    W = V[:, :j+1] #include only the minimal number of high-variance principal components

    X_pca = X @ W #dimensionality-reducing projection onto the subspace of principal components

    return X_pca #each row is a dimensionally reduced datapoint; total dataset captures required variance


























