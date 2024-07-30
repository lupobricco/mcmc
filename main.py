import numpy as np
import pandas as pd
import os
import emcee
import multiprocessing
import scipy.integrate as integrate
import math
import matplotlib.pyplot as plt
import corner
plt.rcParams.update({'font.size': 12})
from tqdm import tqdm
import scipy.linalg as la
import likelyhood_functions

if __name__ == '__main__':
    # Params definition
    rho0 = 2.284
    rs = -0.824
    gamma = 0

    params = [rho0, rs, gamma]

    #p0 = np.random.uniform(low=[-3, -2, -3, -10], high=[5, 3, 3, 1], size=(nwalker, ndim)) #vettore con inizializzazione randomica dei parametri con beta
    p0 = np.random.uniform(low=[-3, -2, -3], high=[5, 3, 3], size=(likelyhood_functions.NWALKER, likelyhood_functions.NDIM)) #vettore con inizializzazione randomica dei parametri



    multiprocessing.set_start_method('spawn', force=True)

    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(likelyhood_functions.NWALKER, likelyhood_functions.NDIM, likelyhood_functions.log_likelyhood, pool=pool)
        sampler.run_mcmc(p0, likelyhood_functions.NITER, progress = True)

    
    samples=sampler.get_chain(discard=10,thin=2,flat=True)
    print(samples)