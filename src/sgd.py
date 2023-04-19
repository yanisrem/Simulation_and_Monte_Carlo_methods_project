from src.estimators2 import *
from src.vraisemblance import *
import numpy as np

### SGD usuelle

def SGD(theta_init, learn_rate, echantillon, n_iter):
    #Step 1: mélanger l'échantillon
    i=0
    compteur=0
    np.random.shuffle(echantillon)
    theta=theta_init-learn_rate*(-1)*gradient_log_vraisemblance(echantillon[i], theta_init)

    i+=1
    compteur+=1
    #Tant qu'on n'atteint pas le nombre d'itérations fixé, on actualise theta
    while compteur<n_iter:
        # Si on a parcouru tout l'échantillon alors que le nombre d'itérations n'est pas atteint,
        # On remélange l'échantillon, on reparcourt l'échantillon en commençant par le début
        # Le compteur n'est pas réinitialisé
        if i==len(echantillon):
           i=0
           np.random.shuffle(echantillon)
           theta= theta - learn_rate*(-1)*gradient_log_vraisemblance(echantillon[i], theta)
           i+=1
           compteur+=1
        else:
            theta=theta-learn_rate*(-1)*gradient_log_vraisemblance(echantillon[i], theta)
            i+=1
            compteur+=1

    else:
        return theta

### SGD IAWE

def SGD_IAWE(theta_init, learn_rate, n_iter, A, b, echantillon, k=6):
    #Step 1: mélanger l'échantillon
    i=0
    compteur=0
    np.random.shuffle(echantillon)
    estimateur_gradient=importance_sampling_gradientlogvraisemblance(k=k, theta=theta_init, A=A, b=b, x=echantillon[i])
    if True in np.isnan(estimateur_gradient):
        estimateur_gradient=0
    theta=theta_init-learn_rate*estimateur_gradient

    i+=1
    compteur+=1
    #Tant qu'on n'atteint pas le nombre d'itérations fixé, on actualise theta
    while compteur<n_iter:
        # Si on a parcouru tout l'échantillon alors que le nombre d'itérations n'est pas atteint,
        # On remélange l'échantillon, on reparcourt l'échantillon en commençant par le début
        # Le compteur n'est pas réinitialisé
        if i==len(echantillon):
            i=0
            np.random.shuffle(echantillon)
            estimateur_gradient=importance_sampling_gradientlogvraisemblance(k=k, theta=theta, A=A, b=b, x=echantillon[i])
            if True in np.isnan(estimateur_gradient):
               estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1
        else:
            estimateur_gradient=importance_sampling_gradientlogvraisemblance(k=k, theta=theta, A=A, b=b, x=echantillon[i])
            if True in np.isnan(estimateur_gradient):
                estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1

    else:
        return theta

### SGD SUMO

def SGD_SUMO(theta_init, learn_rate, n_iter, A, b, echantillon, l=0):
    #Step 1: mélanger l'échantillon
    i=0
    compteur=0
    np.random.shuffle(echantillon)
    estimateur_gradient=estimateur_SUMO_gradientlogvraisemblance(theta=theta_init, A=A, b=b, x=echantillon[i], r=0.6, l=0)
    if True in np.isnan(estimateur_gradient):
        estimateur_gradient=0
    theta=theta_init-learn_rate*estimateur_gradient

    i+=1
    compteur+=1
    #Tant qu'on n'atteint pas le nombre d'itérations fixé, on actualise theta
    while compteur<n_iter:
        # Si on a parcouru tout l'échantillon alors que le nombre d'itérations n'est pas atteint,
        # On remélange l'échantillon, on reparcourt l'échantillon en commençant par le début
        # Le compteur n'est pas réinitialisé
        if i==len(echantillon):
            i=0
            np.random.shuffle(echantillon)
            estimateur_gradient=estimateur_SUMO_gradientlogvraisemblance(theta=theta, A=A, b=b, x=echantillon[i], r=0.6, l=l)
            if True in np.isnan(estimateur_gradient):
                estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1
            
        else:
            estimateur_gradient=estimateur_SUMO_gradientlogvraisemblance(theta=theta, A=A, b=b, x=echantillon[i], r=0.6, l=l)
            if True in np.isnan(estimateur_gradient):
                estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1

    else:
        return theta

### SGD RR

def SGD_RR(theta_init, learn_rate, n_iter, A, b, echantillon, l=6):
    #Step 1: mélanger l'échantillon
    i=0
    compteur=0
    np.random.shuffle(echantillon)
    estimateur_gradient=estimateur_ML_RR_gradientlogvraisemblance(x=echantillon[i], theta=theta_init, A=A, b=b, r=0.6, l=l)
    if True in np.isnan(estimateur_gradient):
        estimateur_gradient=0
    theta=theta_init-learn_rate*estimateur_gradient

    i+=1
    compteur+=1
    #Tant qu'on n'atteint pas le nombre d'itérations fixé, on actualise theta
    while compteur<n_iter:
        # Si on a parcouru tout l'échantillon alors que le nombre d'itérations n'est pas atteint,
        # On remélange l'échantillon, on reparcourt l'échantillon en commençant par le début
        # Le compteur n'est pas réinitialisé
        if i==len(echantillon):
            i=0
            np.random.shuffle(echantillon)
            estimateur_gradient=estimateur_ML_RR_gradientlogvraisemblance(x=echantillon[i], theta=theta, A=A, b=b, r=0.6, l=l)
            if True in np.isnan(estimateur_gradient):
                estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1
        else:
            estimateur_gradient=estimateur_ML_RR_gradientlogvraisemblance(x=echantillon[i], theta=theta, A=A, b=b, r=0.6, l=l)
            if True in np.isnan(estimateur_gradient):
                estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1

    else:
        return theta

### SGD SS

def SGD_SS(theta_init, learn_rate, n_iter, A, b, echantillon, l=6):
    #Step 1: mélanger l'échantillon
    i=0
    compteur=0
    np.random.shuffle(echantillon)
    estimateur_gradient=estimateur_ML_SS_gradientlogvraisemblance(x=echantillon[i], theta=theta_init, A=A, b=b, r=0.6, l=l)
    if True in np.isnan(estimateur_gradient):
        estimateur_gradient=0
    theta=theta_init-learn_rate*estimateur_gradient

    i+=1
    compteur+=1
    #Tant qu'on n'atteint pas le nombre d'itérations fixé, on actualise theta
    while compteur<n_iter:
        # Si on a parcouru tout l'échantillon alors que le nombre d'itérations n'est pas atteint,
        # On remélange l'échantillon, on reparcourt l'échantillon en commençant par le début
        # Le compteur n'est pas réinitialisé
        if i==len(echantillon):
            i=0
            np.random.shuffle(echantillon)
            estimateur_gradient=estimateur_ML_SS_gradientlogvraisemblance(x=echantillon[i], theta=theta, A=A, b=b, r=0.6, l=l)
            if True in np.isnan(estimateur_gradient):
                estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1
        else:
            estimateur_gradient=estimateur_ML_SS_gradientlogvraisemblance(x=echantillon[i], theta=theta, A=A, b=b, r=0.6, l=l)
            if True in np.isnan(estimateur_gradient):
                estimateur_gradient=0
            theta=theta-learn_rate*estimateur_gradient
            i+=1
            compteur+=1

    else:
        return theta