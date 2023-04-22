from src.densite_function import *
from src.gaussian_simulation import *
import numpy as np
from src.vraisemblance import *
from scipy.stats import geom

########## Log-vraisemblance
#### IWAE

def importance_sampling_log_vraisemblance_from_array_w(array_w):
    """Estimateur IWAE de la log-vraisemblance à partir d'un échantillon w(Z_i) préalablement généré

    Args:
        array_w (np.array): array w(Z_i) où Z_i est généré selon la loi de proposition q(.|x)

    Returns:
        float: estimateur IWAE de la log-vraisemblance
    """
    return np.log(np.mean(array_w))

def importance_sampling_logvraisemblance(k, theta, A, b, x, return_weights=False):
    """Estimateur IWAE de la log-vraisemblance

    Args:
        k (int): nombre de variables Z_i tirées sous q(.|x)
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        x (np.array): point où la log-vraisemblance est estimée
        return_weights (bool, optional): si True, on retourne l'échantillon (w(Z_i)) généré. False par défaut

    Returns:
        float: estimateur IWAE de la log-vraisemblance
    """
    array_w=np.array([])

    for i in range(1,k+1): #i=1,...k
        z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))
        W_i=w(z=z_i, x=x, theta=theta, A=A,b=b)
        array_w= np.append(array_w, W_i)
    
    if return_weights:
        return importance_sampling_log_vraisemblance_from_array_w(array_w), array_w
    else:
        return importance_sampling_log_vraisemblance_from_array_w(array_w)

#### SUMO

def estimateur_SUMO_logvraisemblance(x, theta, A, b, r, k_max=None, l=0):
    """Estimateur SUMO de la log-vraisemblance

    Args:
        x (np.array): point où la log-vraisemblance est estimée
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        r (float dans [0,1]): paramètre de la loi géométrique utilisée pour tirer K
        k_max (int, optional): borne supérieure du support de la variable K. None par défaut
        l (int, optional): période de chauffe. 0 par défaut.

    Returns:
        float: estimateur SUMO de la log-vraisemblance
    """
    K=np.random.geometric(p=r, size=1)[0]
    
    if k_max!=None:
        while K>k_max:
            K=np.random.geometric(p=r, size=1)[0]

    array_delta_k_proba_k=np.array([])

    for k in range(K+1): #k=0,..,K

        l_k2, array_w2=importance_sampling_logvraisemblance(k=k+l+2, theta=theta, A=A, b=b, x=x, return_weights=True) #k+l+2
        l_k1=importance_sampling_log_vraisemblance_from_array_w(array_w2[:len(array_w2)-1]) #k+l+1

        if k==0:
            if l==0: #k+l+1=1
                I_0=l_k1
            else:
                I_0=importance_sampling_log_vraisemblance_from_array_w(array_w2[:len(array_w2)-2])
        
        proba_k=geom.sf(k=k, p=r)+geom.pmf(k=k, p=r)
        delta_k=l_k2-l_k1
        array_delta_k_proba_k=np.append(array_delta_k_proba_k, delta_k/proba_k)

    SUMO=I_0+np.sum(array_delta_k_proba_k)

    return SUMO

#### ML-SS

def estimateur_ML_SS_logvraisemblance(x, theta, A, b, r, k_max=None, l=0):
    """Estimateur ML-SS de la log-vraisemblance

    Args:
        x (np.array): point où la log-vraisemblance est estimée
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        r (float dans [0,1]): paramètre de la loi géométrique utilisée pour tirer K
        k_max (int, optional): borne supérieure du support de la variable K. None par défaut
        l (int, optional): période de chauffe. 0 par défaut.

    Returns:
        float: estimateur ML-SS de la log-vraisemblance
    """
    K=np.random.geometric(p=r, size=1)[0]

    if k_max!=None:
        while K>k_max:
            K=np.random.geometric(p=r, size=1)[0]
    array_w=np.array([])
    array_w_O=np.array([])
    array_w_E=np.array([])

    for i in range(1,2**(K+l+1)+1): #i=1,...,2^(K+l+1)

        z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))
        w_i=w(z=z_i, x=x, theta=theta, A=A,b=b)

        if i%2==0:
            array_w_O=np.append(array_w_O, w_i)
        else:    
            array_w_E=np.append(array_w_E, w_i)
        
        array_w=np.append(array_w, w_i)


    array_w=np.unique(array_w)
    I_0=importance_sampling_log_vraisemblance_from_array_w(array_w[:2**l]) #np.log(np.mean(array_w))

    IWAE_O=importance_sampling_log_vraisemblance_from_array_w(array_w_O)
    IWAE_E=importance_sampling_log_vraisemblance_from_array_w(array_w_E)
    IWAE_OUE=importance_sampling_log_vraisemblance_from_array_w(array_w)

    delta_K=IWAE_OUE-0.5*(IWAE_O+IWAE_E)
    SS=I_0+delta_K/geom.pmf(K, p=r)

    return SS

#### ML-RR

def estimateur_ML_RR_logvraisemblance(x, theta, A, b, r, k_max=None, l=0):
    """Estimateur ML-RR de la log-vraisemblance

    Args:
        x (np.array): point où la log-vraisemblance est estimée
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        r (float dans [0,1]): paramètre de la loi géométrique utilisée pour tirer K
        k_max (int, optional): borne supérieure du support de la variable K. None par défaut
        l (int, optional): période de chauffe. 0 par défaut.

    Returns:
        float: estimateur ML-RR de la log-vraisemblance
    """
    K=np.random.geometric(p=r, size=1)[0]
    if k_max!=None:
        while K>k_max:
            K=np.random.geometric(p=r, size=1)[0]
    array_delta_proba=np.array([])

    for k in range(K+1): #k=0,...,K

        array_w=np.array([])
        array_w_O=np.array([])
        array_w_E=np.array([])


        for i in range(1,2**(k+l+1)+1): #i=1,...,2^(k+l+1)

            z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))
            w_i=w(z=z_i, x=x, theta=theta, A=A,b=b)

            if i%2==0:
                array_w_O=np.append(array_w_O, w_i)
            else:    
                array_w_E=np.append(array_w_E, w_i)

            array_w=np.append(array_w, w_i)

        array_w=np.unique(array_w)

        IWAE_O=importance_sampling_log_vraisemblance_from_array_w(array_w_O)
        IWAE_E=importance_sampling_log_vraisemblance_from_array_w(array_w_E)
        IWAE_OUE=importance_sampling_log_vraisemblance_from_array_w(array_w)

        delta_k=IWAE_OUE-0.5*(IWAE_O+IWAE_E)

        if k==0:
            I_0=importance_sampling_log_vraisemblance_from_array_w(array_w[:2**l])

        proba_k=geom.sf(k=k, p=r)+geom.pmf(k=k, p=r)
        array_delta_proba=np.append(array_delta_proba, delta_k/proba_k)
    

    RR=I_0+np.sum(array_delta_proba)

    return RR


########## Gradient de la  log-vraisemblance

#### IWAE

def importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta):
    """Estimateur IWAE du gradient de la log-vraisemblance, calculé à partir de deux arrays (Z_i) et (w(Z_i)) générés préalablement

    Args:
        array_w (np.array): array w(Z_i) où Z_i est généré selon la loi de proposition q(.|x)
        array_z (np.array): array Z_i
        theta (np.array): paramètre d'intérêt

    Returns:
        np.array: estimateur IWAE du gradient de la log-vraisemblance
    """
    num=np.sum(np.array([w_i*(z_i-theta) for (w_i,z_i) in zip(array_w,array_z)]), axis=0)
    denom=np.sum(array_w)

    return num/denom

def importance_sampling_gradientlogvraisemblance(k, theta, A, b, x, return_array_w_z=False):
    """Estimateur IWAE du gradient de la log-vraisemblance

    Args:
        k (int): nombre de variables Z_i tirées sous q(.|x)
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        x (np.array): point où le gradient de la log-vraisemblance est estimé
        return_array_w_z (bool, optional): si True, on retourne les échantillons (w(Z_i)) et (Z_i) générés. False par défaut

    Returns:
        np.array: estimateur IWAE du gradient de la log-vraisemblance
    """

    array_w=np.array([])
    array_z=np.array([])

    for i in range(1,k+1): #i=1,...,k
        z_i=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))
        W_i=w(z=z_i, x=x, theta=theta, A=A,b=b)
        array_w= np.append(array_w, W_i)

        if i==1:
            array_z=np.append(array_z, z_i)
        else:
            array_z=np.vstack((array_z, z_i))
    
    if return_array_w_z:
        return array_w, array_z, importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)
    else:
        return importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)

#### SUMO

def estimateur_SUMO_gradientlogvraisemblance(x, theta, A, b, r, k_max=None, l=0):
    """Estimateur SUMO du gradient de la log-vraisemblance

    Args:
        x (np.array): point où le gradient de la log-vraisemblance est estimé
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        r (float dans [0,1]): paramètre de la loi géométrique utilisée pour tirer K
        k_max (int, optional): borne supérieure du support de la variable K. None par défaut
        l (int, optional): période de chauffe. 0 par défaut.

    Returns:
        np.array: estimateur SUMO du gradient de la log-vraisemblance
    """
    K=np.random.geometric(p=r, size=1)[0]
    
    if k_max!=None:
        while K>k_max:
            K=np.random.geometric(p=r, size=1)[0]

    array_delta_k_proba_k=np.array([])

    for k in range(K+1): #k=0,...,K
        if k==0:
            array_w2, array_z2, gradient_k2=importance_sampling_gradientlogvraisemblance(k=k+l+2, theta=theta, A=A, b=b, x=x, return_array_w_z=True) #k+l+2
            gradient_k1=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w2[:len(array_w2-1)], array_z2[:len(array_z2)-1], theta) #k+l+1
            delta_k=gradient_k2-gradient_k1
            proba_k=geom.sf(k=k, p=r)+geom.pmf(k=k, p=r)
            array_delta_k_proba_k=np.append(array_delta_k_proba_k, delta_k/proba_k)

            if l==0:
                I_0=gradient_k1
            else:
                I_0=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w2[:len(array_w2-2)], array_z2[:len(array_z2)-2], theta) #k+l+2-2=l

            array_w2, array_z2, gradient_k2=importance_sampling_gradientlogvraisemblance(k=k+l+2, theta=theta, A=A, b=b, x=x, return_array_w_z=True) #k+l+2
            gradient_k1=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w2[:len(array_w2-1)], array_z2[:len(array_z2)-1], theta)
            delta_k=gradient_k2-gradient_k1
            proba_k=geom.sf(k=k, p=r)+geom.pmf(k=k, p=r)
            array_delta_k_proba_k=np.vstack((array_delta_k_proba_k, delta_k/proba_k))
    
    SUMO=I_0+np.sum(array_delta_k_proba_k, axis=0)

    return SUMO

#### ML-SS

def estimateur_ML_SS_gradientlogvraisemblance(x, theta, A, b, r, k_max=None, l=0):
    """Estimateur ML-SS du gradient de la log-vraisemblance

    Args:
        x (np.array): point où le gradient de la log-vraisemblance est estimé
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        r (float dans [0,1]): paramètre de la loi géométrique utilisée pour tirer K
        k_max (int, optional): borne supérieure du support de la variable K. None par défaut
        l (int, optional): période de chauffe. 0 par défaut.

    Returns:
        np.array: estimateur ML-SS du gradient de la log-vraisemblance
    """
    np.seterr(divide='ignore', invalid='ignore')

    K=np.random.geometric(p=r, size=1)[0]

    if k_max!=None:
        while K>k_max:
            K=np.random.geometric(p=r, size=1)[0]

    z_O=np.array([])
    z_E=np.array([])

    array_w=np.array([])
    array_w_O=np.array([])
    array_w_E=np.array([])

    for i in range(1,2**(K+l)+1): #i=1,...,2^(K+l)
        z_i_O=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))
        z_i_E=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))

        w_i_E=w(z=z_i_E, x=x, theta=theta, A=A,b=b)
        w_i_O=w(z=z_i_O, x=x, theta=theta, A=A,b=b)

        if i==1:
            z_O= np.append(z_O, z_i_O)
            z_E= np.append(z_E, z_i_E)

        else:
            z_O= np.vstack((z_O, z_i_O))
            z_E= np.vstack((z_E, z_i_E))

        array_w_O=np.append(array_w_O, w_i_O)
        array_w_E=np.append(array_w_E, w_i_E)

    array_z=np.unique(np.vstack((z_O,z_E)), axis=0)
    array_w=np.union1d(array_w_O, array_w_E)

    I_0=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w[:2**l], array_z[:2**l], theta)

    IWAE_O=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_O, z_O, theta)
    IWAE_E=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_E, z_E, theta)
    IWAE_OUE=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)

    delta_K=IWAE_OUE-0.5*(IWAE_O+IWAE_E)

    SS=I_0+delta_K/geom.pmf(K, p=r)

    return SS

    
def estimateur_ML_RR_gradientlogvraisemblance(x, theta, A, b, r, k_max=None,l=0):
    """Estimateur ML-RR du gradient de la log-vraisemblance

    Args:
        x (np.array): point où e gradient de la log-vraisemblance est estimé
        theta (np.array): paramètre d'intérêt
        A (np.array): matrice de dimensions 20*20
        b (np.array): vecteur dans R^20
        r (float dans [0,1]): paramètre de la loi géométrique utilisée pour tirer K
        k_max (int, optional): borne supérieure du support de la variable K. None par défaut
        l (int, optional): période de chauffe. 0 par défaut.

    Returns:
        np.array: estimateur ML-RR du gradient de la log-vraisemblance
    """
    np.seterr(divide='ignore', invalid='ignore')

    K=np.random.geometric(p=r, size=1)[0]
    
    if k_max!=None:
        while K>k_max:
            K=np.random.geometric(p=r, size=1)[0]

    array_delta_k_proba_K=np.array([])

    for k in range(K+1): #k=0,...,K

        z_O=np.array([])
        z_E=np.array([])

        array_w=np.array([])
        array_w_O=np.array([])
        array_w_E=np.array([])

        for i in range(1,2**(k+l)+1): #i=1,...,2^(k+l)

            z_i_O=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))
            z_i_E=simulate_gaussian_vector(mu=np.matmul(A,x)+b, sigma=(2/3)*np.identity(20))

            w_i_E=w(z=z_i_E, x=x, theta=theta, A=A,b=b)
            w_i_O=w(z=z_i_O, x=x, theta=theta, A=A,b=b)

            if i==1:
                z_O=np.append(z_O, z_i_O)
                z_E=np.append(z_E, z_i_E)

            else:
                z_O= np.vstack((z_O, z_i_O))
                z_E= np.vstack((z_E, z_i_E))

            array_w_O=np.append(array_w_O, w_i_O)
            array_w_E=np.append(array_w_E, w_i_E)

        array_z=np.unique(np.vstack((z_O,z_E)), axis=0)
        array_w=np.union1d(array_w_O, array_w_E)

        IWAE_O=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_O, z_O, theta)
        IWAE_E=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w_E, z_E, theta)
        IWAE_OUE=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w, array_z, theta)

        delta_k=IWAE_OUE-0.5*(IWAE_O+IWAE_E)
        proba_k=geom.sf(k=k, p=r)+geom.pmf(k=k, p=r)

        if k==0:
            array_delta_k_proba_K=np.append(array_delta_k_proba_K, delta_k/proba_k)
            I_0=importance_sampling_gradientlog_vraisemblance_from_array_wz(array_w[:2**l], array_z[:2**l], theta)


        else:
            array_delta_k_proba_K=np.vstack((array_delta_k_proba_K, delta_k/proba_k))
    
    RR=I_0+np.sum(array_delta_k_proba_K, axis=0)

    return RR