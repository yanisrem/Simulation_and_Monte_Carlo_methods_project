import numpy as np
from src.densite_function import *

def log_vraisemblance(x, theta):
    """Log-vraisemblance log(p(x))

    Args:
        x (np.array): point dans R^20
        theta (np.array): paramètre d'intérêt

    Returns:
        float: log-vraisemblance au point x
    """
    return np.log(densite_X(x, theta))

def gradient_log_vraisemblance(x, theta):
    """Gradient de log(p(x)) par rapport à theta

    Args:
        x (np.array): point dans R^20
        theta (np.array): paramètre d'intérêt

    Returns:
        float: gradient de la log-vraisemblance au point x
    """
    return 2*(x-theta)