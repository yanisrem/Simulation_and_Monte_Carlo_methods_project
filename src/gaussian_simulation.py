import numpy as np


def simulate_gaussian_vector(mu, sigma):
    d=len(sigma)
    z=np.random.normal(0, 1, d)
    C = np.linalg.cholesky(sigma)

    X=mu+np.dot(C,z)
    return np.array(X)
