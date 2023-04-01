import numpy as np

def simulatate_gaussian_vector(mu, sigma):

    d=len(sigma)
    z=np.random.normal(0, 1, d)
    C = np.linalg.cholesky(sigma)

    X=mu+np.matmul(C,z)
    return X
