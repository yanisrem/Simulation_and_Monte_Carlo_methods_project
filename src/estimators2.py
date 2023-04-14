from src.densite_function import *
from src.gaussian_simulation import *
import numpy as np
from src.vraisemblance import *
from scipy.stats import geom

########## Log-vraisemblance
#### IWAE

def importance_sampling_log_vraisemblance_from_array_w(array_w):
    return np.log(np.mean(array_w))

def importance_sampling_logvraisemblance(k, theta, A, b, x, return_weights=False):
    array_w=np.array([])
    i=0
    while i<k:
        z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]
        W_i=w(z=z_i, x=x, theta=theta, A=A,b=b)
        array_w= np.append (array_w, W_i)
        i+=1
    
    if return_weights:
        return importance_sampling_log_vraisemblance_from_array_w(array_w), array_w
    else:
        return importance_sampling_log_vraisemblance_from_array_w(array_w)

#### SUMO

def estimateur_SUMO_logvraisemblance(x, theta, A, b, r, k_max=1):
    K=np.random.geometric(p=r, size=1)[0]
    
    while K>k_max:
        K=np.random.geometric(p=r, size=1)[0]

    k=0
    array_delta_k=np.array([])
    array_proba_K_sup_k=np.array([])

    while k<K:

        if k==0:
            l_k1, array_w1=importance_sampling_logvraisemblance(k=k+1, theta=theta, A=A, b=b, x=x, return_weights=True)
            proba_k=1 #Convention: Si K suit une loi géométrique, P(K>=0)=1
        else:
            l_k1=importance_sampling_logvraisemblance(k=k+1, theta=theta, A=A, b=b, x=x, return_weights=False)
            proba_k=geom.cdf(k, p=r)

        l_k2=importance_sampling_logvraisemblance(k=k+2, theta=theta, A=A, b=b, x=x, return_weights=False)

        delta_k=l_k2-l_k1
        array_delta_k=np.append(array_delta_k, delta_k)
        array_proba_K_sup_k=np.append(array_proba_K_sup_k, proba_k)

        k+=1
    
    I_0=np.log(array_w1[0])
    SUMO=I_0+np.sum(array_delta_k/array_proba_K_sup_k)

    return SUMO

#### ML-SS

def estimateur_ML_SS_logvraisemblance(x, theta, A, b, r, k_max=1):
    K=np.random.geometric(p=r, size=1)[0]
    
    while K>k_max:
        K=np.random.geometric(p=r, size=1)[0]

    z_O=np.array([])
    z_E=np.array([])

    array_w=np.array([])
    array_w_O=np.array([])
    array_w_E=np.array([])

    i=1
    while i<=2**K:
        z_i_O=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]
        z_i_E=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]

        w_i_E=w(z=z_i_E, x=x, theta=theta, A=A,b=b)
        w_i_O=w(z=z_i_O, x=x, theta=theta, A=A,b=b)

        
        z_O= np.append(z_O, z_i_O)
        z_E= np.append(z_E, z_i_E)

        array_w_O=np.append(array_w_O, w_i_O)
        array_w_E=np.append(array_w_E, w_i_E)
        i+=1

    array_w=np.union1d(array_w_O, array_w_E)

    I_0=np.mean(np.log(array_w))

    IWAE_O=importance_sampling_log_vraisemblance_from_array_w(array_w_O)
    IWAE_E=importance_sampling_log_vraisemblance_from_array_w(array_w_E)
    IWAE_OUE=importance_sampling_log_vraisemblance_from_array_w(array_w)

    delta_K=IWAE_OUE-0.5*(IWAE_O+IWAE_E)

    SS=I_0+delta_K/geom.pmf(K, p=r)

    return SS


#### ML-RR

def estimateur_ML_RR_logvraisemblance(x, theta, A, b, r, k_max=1):
    K=np.random.geometric(p=r, size=1)[0]
    while K>k_max:
        K=np.random.geometric(p=r, size=1)[0]
        
    k=0
    array_delta_k=np.array([])
    array_proba_K_sup_k=np.array([])

    while k<K:

        z_O=np.array([])
        z_E=np.array([])

        array_w=np.array([])
        array_w_O=np.array([])
        array_w_E=np.array([])

        i=0
        while i<2**k:

            z_i_O=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]
            z_i_E=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]

            w_i_E=w(z=z_i_E, x=x, theta=theta, A=A,b=b)
            w_i_O=w(z=z_i_O, x=x, theta=theta, A=A,b=b)

            
            z_O= np.append(z_O, z_i_O)
            z_E= np.append(z_E, z_i_E)

            array_w_O=np.append(array_w_O, w_i_O)
            array_w_E=np.append(array_w_E, w_i_E)
            i+=1

        array_w=np.union1d(array_w_O, array_w_E)

        IWAE_O=importance_sampling_log_vraisemblance_from_array_w(array_w_O)
        IWAE_E=importance_sampling_log_vraisemblance_from_array_w(array_w_E)
        IWAE_OUE=importance_sampling_log_vraisemblance_from_array_w(array_w)

        delta_k=IWAE_OUE-0.5*(IWAE_O+IWAE_E)

        if k==0:
            I_0=np.mean(np.log(array_w))
            proba_k=1 #convention P(K>=0)=1
        else:
            proba_k=geom.cdf(k, p=r)

        array_delta_k=np.append(array_delta_k, delta_k)
        array_proba_K_sup_k=np.append(array_proba_K_sup_k, proba_k)

        k+=1
    
    RR=I_0+np.sum(array_delta_k/array_proba_K_sup_k)

    return RR


########## Gradient de la  log-vraisemblance

#### IWAE

def importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta):
    num=np.sum(np.array([w_i*(z_i-theta) for (w_i,z_i) in zip(array_w,array_z)]), axis=0)
    denom=np.sum(array_w)

    return num/denom

def importance_sampling_gradientlogvraisemblance(k, theta, A, b, x, return_array_w_z=False):

    array_w=np.array([])
    array_z=np.array([])

    i=0
    while i<k:
        z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]
        W_i=w(z=z_i, x=x, theta=theta, A=A,b=b)
        array_w= np.append(array_w, W_i)

        if i==0:
            array_z=np.append(array_z, z_i)
        else:
            array_z=np.vstack((array_z, z_i))
        i+=1
    
    if return_array_w_z:
        return array_w, array_z, importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)
    else:
        return importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)

#### SUMO

def estimateur_SUMO_gradientlogvraisemblance(x, theta, A, b, r, k_max=1):
    K=np.random.geometric(p=r, size=1)[0]
    
    while K>k_max:
        K=np.random.geometric(p=r, size=1)[0]

    k=0
    array_delta_k=np.array([])
    array_proba_K_sup_k=np.array([])

    while k<=K:
        if k==0:
            gradient_k2=importance_sampling_gradientlogvraisemblance(k=k+2, theta=theta, A=A, b=b, x=x, return_array_w_z=False)
            array_w1, array_z1, gradient_k1=importance_sampling_gradientlogvraisemblance(k=k+1, theta=theta, A=A, b=b, x=x, return_array_w_z=True)
            proba_k=1
            delta_k=gradient_k2-gradient_k1
            array_delta_k=np.append(array_delta_k, delta_k)
        else:
            gradient_k2=importance_sampling_gradientlogvraisemblance(k=k+2, theta=theta, A=A, b=b, x=x, return_array_w_z=False)
            gradient_k1=importance_sampling_gradientlogvraisemblance(k=k+1, theta=theta, A=A, b=b, x=x, return_array_w_z=False)
            delta_k=gradient_k2-gradient_k1
            proba_k=geom.cdf(k, p=r)
            array_delta_k=np.vstack((array_delta_k, delta_k))

        array_proba_K_sup_k=np.append(array_proba_K_sup_k, proba_k)

        k+=1
    
    I_0=array_z1-theta
    SUMO=I_0+np.sum(np.array([array_delta_k[i]/array_proba_K_sup_k[i] for i in range(len(array_proba_K_sup_k))]), axis=0)

    return SUMO

#### ML-SS

def estimateur_ML_SS_gradientlogvraisemblance(x, theta, A, b, r, k_max=1):
    K=np.random.geometric(p=r, size=1)[0]

    while K>k_max:
        K=np.random.geometric(p=r, size=1)[0]

    z_O=np.array([])
    z_E=np.array([])

    array_w=np.array([])
    array_w_O=np.array([])
    array_w_E=np.array([])

    i=0
    while i<2**K:
        z_i_O=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]
        z_i_E=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]

        w_i_E=w(z=z_i_E, x=x, theta=theta, A=A,b=b)
        w_i_O=w(z=z_i_O, x=x, theta=theta, A=A,b=b)

        if i==0:
            z_O= np.append(z_O, z_i_O)
            z_E= np.append(z_E, z_i_E)

        else:
            z_O= np.vstack((z_O, z_i_O))
            z_E= np.vstack((z_E, z_i_E))

        array_w_O=np.append(array_w_O, w_i_O)
        array_w_E=np.append(array_w_E, w_i_E)
        i+=1

    array_z=np.unique(np.vstack((z_O,z_E)), axis=0)
    array_w=np.union1d(array_w_O, array_w_E)


    I_0=np.mean(array_z)-theta

    IWAE_O=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_O, z_O, theta)
    IWAE_E=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_E, z_E, theta)
    IWAE_OUE=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)

    delta_K=IWAE_OUE-0.5*(IWAE_O+IWAE_E)

    SS=I_0+delta_K/geom.pmf(K, p=r)

    return SS

#### ML-RR
    
def estimateur_ML_RR_gradientlogvraisemblance(x, theta, A, b, r, k_max=1):
    K=np.random.geometric(p=r, size=1)[0]
    
    while K>k_max:
        K=np.random.geometric(p=r, size=1)[0]
    k=0
    array_delta_k=np.array([])
    array_proba_K_sup_k=np.array([])

    while k<K:

        z_O=np.array([])
        z_E=np.array([])

        array_w=np.array([])
        array_w_O=np.array([])
        array_w_E=np.array([])

        i=0
        while i<2**k:

            z_i_O=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]
            z_i_E=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))[0]

            w_i_E=w(z=z_i_E, x=x, theta=theta, A=A,b=b)
            w_i_O=w(z=z_i_O, x=x, theta=theta, A=A,b=b)

            if i==0:
                z_O=np.append(z_O, z_i_O)
                z_E=np.append(z_E, z_i_E)

            else:
                z_O= np.vstack((z_O, z_i_O))
                z_E= np.vstack((z_E, z_i_E))

            array_w_O=np.append(array_w_O, w_i_O)
            array_w_E=np.append(array_w_E, w_i_E)
            i+=1

        array_z=np.unique(np.vstack((z_O,z_E)), axis=0)
        array_w=np.union1d(array_w_O, array_w_E)

        IWAE_O=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_O, z_O, theta)
        IWAE_E=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_E, z_E, theta)
        IWAE_OUE=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)

        delta_k=IWAE_OUE-0.5*(IWAE_O+IWAE_E)

        if k==0:
            proba_k=1
            array_delta_k=np.append(array_delta_k, delta_k)
            I_0=np.mean(array_z)-theta
        else:
            proba_k=geom.cdf(k, p=r)
            array_delta_k=np.vstack((array_delta_k, delta_k))

        array_proba_K_sup_k=np.append(array_proba_K_sup_k, proba_k)

        k+=1
    
    RR=I_0+np.sum(np.array([array_delta_k[i]/array_proba_K_sup_k[i] for i in range(len(array_proba_K_sup_k))]), axis=0)

    return RR