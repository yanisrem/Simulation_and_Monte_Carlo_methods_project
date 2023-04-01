import numpy as np
from densite_function import *

def log_vraisemblance(x, theta):
    return np.log(densite_X(x, theta))

def gradient_log_vraisemblance(x, theta):
    return 2*(x-theta)