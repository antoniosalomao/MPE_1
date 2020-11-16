import pandas as pd
import numpy as np
from numpy import *
from scipy.stats import random_correlation
from scipy.optimize import minimize
import random

#-------------------------------------------------------------------------------------------------------#
#-----------#
# Functions #
#-----------#

def random_covar_matrix():
    # Volatility 
    vol_arr = np.array([0.75, 1.25, 1.45])
    vol_diag = np.diag(vol_arr)
    # Correlation
    corr = np.array([ [1,   0.25, 0.55], 
                      [0.25, 1,   0], 
                      [0.55, 0, 1]])
    # Covariance
    covar = vol_diag@corr@vol_diag
    return covar

#-------------------------------------------------------------------------------------------------------#
#------#
# Main #
#------#
                            
covar = random_covar_matrix()

# (2) Risk parity problem
# Minimum variance problem

minvar_results = []
minvar_trials = 1000
for n in range(minvar_trials):
    
    MV_opt_dict = {'fun': lambda weights_arr: (0.5)*weights_arr.T@covar@weights_arr,
                    'x0': np.random.uniform(low=0, high=(1), size=covar.shape[0]),
                'method': 'SLSQP',
                'bounds': [(0, 1) for x in range(covar.shape[0])],
           'constraints': [({'type': 'eq', 'fun': lambda weights_arr: 1 - np.sum(weights_arr)})]}

    opt_results = minimize(**MV_opt_dict)
    if (opt_results.success == True):
        minvar_fun = opt_results.fun
        minvar_weights = opt_results.x
        minvar_results.append(tuple((minvar_fun, minvar_weights)))

best_minvar_sol = sorted(minvar_results, key= lambda tup: tup[0])[0]
best_minvar_weights = best_minvar_sol[1]

'''
Minimum variance approach leads to portfolios with equal marginal risk contribtuions.
'''

minvar_port_vol = best_minvar_weights.T@covar@best_minvar_weights
mrc_arr = (best_minvar_weights@covar)/minvar_port_vol


print(mrc_arr)

