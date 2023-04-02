from src.densite_function import *
from src.gaussian_simulation import *
import numpy as np
from src.vraisemblance import *

def importance_sampling_logvraisemblance(k, theta, A, b, x):
    array_w=np.array([])
    i=0
    while i<k:
        z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]
        W_i=w(z=z_i, x=x, theta=theta, A=A,b=b)
        array_w= np.append (array_w, W_i)
        i+=1
    return(np.log(np.mean(array_w)))


def importance_sampling_gradientlogvraisemblance(k, theta, A, b, x):
    array_wi=np.array([])
    gradient=x-theta
    i=0
    while i<k:
        z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]

        W_i=w(z=z_i, x=x, theta=theta, A=A,b=b)

        array_wi= np.append(array_wi, W_i)
        i+=1
    return x-theta


def biais_IWAE_logvraisemblance():
    pass

def biais_IWAE_gradientlogvraisemblance():
    pass