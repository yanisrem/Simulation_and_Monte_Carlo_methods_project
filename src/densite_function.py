import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def densite_Z(z, theta):
    return multivariate_normal.pdf(z, mean=theta, cov=np.identity(20))

def densite_X_sachant_Z(x, z):
    return multivariate_normal.pdf(x, mean=z, cov=np.identity(20))


def densite_X(x, theta):
    return multivariate_normal.pdf(x, mean=theta, cov=2*np.identity(20))


def densite_Z_sachant_X(z,x,theta):
    return multivariate_normal.pdf(z, mean=(theta+x)/2, cov=0.5*np.identity(20))


def densite_XZ(x,z,theta):
    return densite_X_sachant_Z(x,z)*densite_Z(z,theta)


def densite_q(A, b, z, x):
    return multivariate_normal.pdf(z, mean=np.array(np.dot(A,x)+b)[0], cov=(2/3)*np.identity(20))

def w(z, x, theta, A,b):
    return densite_XZ(x,z,theta)/densite_q(A, b, z, x)

