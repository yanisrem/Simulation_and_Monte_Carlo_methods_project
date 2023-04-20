import numpy as np

def simulate_gaussian_vector(mu, sigma):
    """Simulation d'un vecteur gaussien

    Args:
        mu (np.array): vecteur de moyenne
        sigma (np.array): matrice de variances-covariances

    Returns:
        np.array: vecteur gaussien
    """
    d=len(sigma)
    z=np.random.normal(0, 1, d)
    C = np.linalg.cholesky(sigma)

    X=mu+np.dot(C,z)
    return np.array(X)


def generer_nech_gaussien_x_sachant_z(n, mu_z, sigma_z, sigma_x):
    """Simulation d'un n-échantillon de vecteurs gaussiens, de loi conditionnelle à z

    Args:
        n (int): nombre d'observations
        mu_z (np.array): vecteur de moyennes de la variable Z
        sigma_z (np.array): matrice de variances-covariances de la variable Z
        sigma_x (np.array): matrice de variances-covariances de la variable X

    Returns:
        np.array: n-échantillon de vecteurs gaussiens
    """
    echantillon_x=np.array([])

    for i in range(1,n+1):
        z=simulate_gaussian_vector(mu=mu_z, sigma=sigma_z)
        if i==1:
            echantillon_x=np.append(echantillon_x, simulate_gaussian_vector(mu=z, sigma=sigma_x)) #on sait simuler x|z
        else:
            echantillon_x=np.vstack((echantillon_x, simulate_gaussian_vector(mu=z, sigma=sigma_x)))
    
    return echantillon_x