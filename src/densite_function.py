import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def densite_Z(z, theta):
    """Densité p(z)

    Args:
        z (np.array): vecteur dans R^20
        theta (np.array): paramètre d'intérêt

    Returns:
        float: densité calculée au point z
    """
    return multivariate_normal.pdf(z, mean=theta, cov=np.identity(20))

def densite_X_sachant_Z(x, z):
    """Densité p(x|z)

    Args:
        x (np.array): vecteur dans R^20
        z (np.array): vecteur dans R^20
        theta (np.array): paramètre d'intérêt

    Returns:
        float: densité calculée au point x sachant z
    """
    return multivariate_normal.pdf(x, mean=z, cov=np.identity(20))


def densite_X(x, theta):
    """Densité p(x)

    Args:
        x (np.array): vecteur dans R^20
        theta (np.array): paramètre d'intérêt

    Returns:
        float: densité calculée au point x
    """
    return multivariate_normal.pdf(x, mean=theta, cov=2*np.identity(20))


def densite_Z_sachant_X(z,x,theta):
    """Densité p(z|x)

    Args:
        x (np.array): vecteur dans R^20
        z (np.array): vecteur dans R^20
        theta (np.array): paramètre d'intérêt

    Returns:
        float: densité calculée au point z sachant x
    """
    return multivariate_normal.pdf(z, mean=(theta+x)/2, cov=0.5*np.identity(20))


def densite_XZ(x,z,theta):
    """Densité jointe p(x,z)

    Args:
        x (np.array): vecteur dans R^20
        z (np.array): vecteur dans R^20
        theta (np.array): paramètre d'intérêt

    Returns:
        float: densité calculée au point (x,z)
    """
    return densite_X_sachant_Z(x,z)*densite_Z(z,theta)


def densite_q(A, b, z, x):
    """Loi de proposition q(z|x)

    Args:
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        z (np.array): vecteur dans R^20
        x (np.array): vecteur dans R^20

    Returns:
        float: loi de proposition calculée au point z sachant x
    """
    return multivariate_normal.pdf(z, mean=np.matmul(A,x)+b, cov=(2/3)*np.identity(20))

def w(z, x, theta, A,b):
    """Poids d'importance

    Args:
        z (np.array): vecteur dans R^20
        x (np.array): vecteur dans R^20
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20

    Returns:
        float: poids d'importance en z
    """
    return densite_XZ(x,z,theta)/densite_q(A, b, z, x)

