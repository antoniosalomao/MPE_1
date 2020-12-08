import pandas as pd
import numpy as np
import math
import scipy.optimize
from scipy.optimize import minimize

# Variance-Covarariance Matrix Q
Q = np.array([[1, -0.9, 0.6], 
              [-0.9, 1, -0.2],
              [0.6, -0.2, 4]])

# Risk-Parity: Allowing for negative weights

def RP_X_sum(params_rp, C):
    '''
    Sum(weights) constraint
    '''

    # Selecting portfolio weights
    X = params_rp[:3]
    sum_X = sum(X) - C

    return sum_X

def F_RP_X(params_rp, covariance_matrix, rho):
    '''
    Objective Function F
    '''

    # Loading variables
    Q = covariance_matrix
    X_n = Q.shape[0]

    # Slicing array
    X, theta = params_rp[:X_n], params_rp[-1]

    A = [math.pow(X[N]*Q[N]@X - theta, 2) for N, i in enumerate(X)]
    B = sum(A) + rho*X.T@Q@X

    return B


def RP_opt(covariance_matrix, LB_UB_x, X_sum, rho):
    '''
    Risk Parity optimization
    '''

    covariance_matrix = Q

    init_guess_X = np.random.uniform(low=LB_UB_x[0], high=LB_UB_x[1], size=len(Q))
    init_guess_theta = np.random.uniform(low=-10, high=10, size=1)
    init_guess_X_theta = np.array(list(init_guess_X) + list(init_guess_theta))

    bounds_X_theta = [LB_UB_x for _ in Q] + [(-1000, 1000)]

    opt_dict = { 'fun': F_RP_X,
                  'x0': init_guess_X_theta,
                'args': tuple((covariance_matrix, rho)),
              'bounds': bounds_X_theta,
         'constraints': {'type': 'eq', 'fun': RP_X_sum, 'args': [X_sum]}}

    opt_report = minimize(**opt_dict, tol=math.pow(10, -16), options={'maxiter': math.pow(10, 16)})

    return opt_report

# Minimum variance with risk parity
# Sequential min-variance risk parity algorithm
estimated_covar = Q
LB_UB_x = tuple((-2, 1))
rho_L = []
rho_start = 4
for k in list(range(0, 16)):
    rho_i = math.pow(10, rho_start)
    rho_start -= 1
    rho_L.append(rho_i)

RP_solutions = []
n_RP_trials = 10
for rho_i in rho_L:
    for trial in range(n_RP_trials):
        rp_opt_dict = { 'covariance_matrix': estimated_covar,
                                  'LB_UB_x': LB_UB_x,
                                    'X_sum': 1,
                                      'rho': rho_i}

        result = RP_opt(**rp_opt_dict)
        result_fun = result.fun
        result_X = result.x
        RP_X = result_X[:len(estimated_covar)]

        # Check RP solution
        RP_variance = RP_X@Q@RP_X
        RP_vol = math.pow(RP_variance, 0.5)
        RC_target = (RP_variance)/len(RP_X)
        RC_L = ([RP_X[N]*Q[N]@RP_X for N, i in enumerate(RP_X)])
        RC_tolerance = math.pow(10, -4)
        RC_test = [abs(RC_i - RC_target) for RC_i in RC_L]

        if ((all(i <= RC_tolerance for i in RC_test)) == True):
            RP_solutions.append(tuple((RP_X, RP_vol)))


RP_X_sorted = sorted(RP_solutions, key= lambda x: x[1])
RP_X = RP_X_sorted[0]

print(RP_X)



