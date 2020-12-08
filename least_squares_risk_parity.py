import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------#
# Functions #
#-----------#

def RP_X_sum(params_rp, Q, X_sum):
    '''
    Sum(weights) constraint
    '''

    # Selecting portfolio weights
    X_n = Q.shape[0]
    X = params_rp[:X_n]
    sum_X = sum(X) - X_sum

    return sum_X

def F_RP_X(params_rp, Q, rho):
    '''
    Objective Function F
    '''
    
    # Slicing array
    X_n = Q.shape[0]
    X, theta = params_rp[:X_n], params_rp[-1]

    A = [math.pow(X[N]*Q[N]@X - theta, 2) for N, i in enumerate(X)]
    B = sum(A) + rho*X.T@Q@X

    return B

def RP_opt(covariance_matrix, LB_UB_x, X_sum, rho):
    '''
    Risk Parity optimization
    '''
    # Loading variables
    LB_x, UB_x, Q = LB_UB_x[0], LB_UB_x[0], covariance_matrix
    LB_theta, UB_theta = -50, 50

    init_guess_X = np.random.uniform(low=LB_x , high=UB_x, size=len(Q))
    init_guess_theta = np.random.uniform(low=LB_theta, high=UB_theta, size=1)
    init_guess_X_theta = np.array(list(init_guess_X) + list(init_guess_theta))

    bounds_X_theta = [LB_UB_x for _ in Q] + [(LB_theta, UB_theta)]

    opt_dict = { 'fun': F_RP_X,
                  'x0': init_guess_X_theta,
                'args': tuple((Q, rho)),
              'bounds': bounds_X_theta,
         'constraints': {'type': 'eq', 'fun': RP_X_sum, 'args': tuple((Q, X_sum))},
                 'tol': math.pow(10, -14),
             'options': {'maxiter': math.pow(10, 6)}}

    opt_report = minimize(**opt_dict)

    return opt_report

def sequential_mv_rp(covariance_matrix, LB_x, UB_x, X_sum):
    '''
    Minimum variance with risk parity
    Sequential min-variance risk parity algorithm
    '''

    Q = covariance_matrix

    rho_L = []
    rho_start = 4
    for k in list(range(0, 12)):
        rho_i = math.pow(10, rho_start)
        rho_start -= 1
        rho_L.append(rho_i)

    RP_solutions = []
    n_RP_trials = 50
    for rho_i in rho_L:
        print('====== Rho i: {} '.format(rho_i))
        for trial in range(n_RP_trials):
            rp_opt_dict = { 'covariance_matrix': Q,
                                      'LB_UB_x': tuple((LB_x, UB_x)),
                                        'X_sum': X_sum,
                                          'rho': rho_i}

            result = RP_opt(**rp_opt_dict)
            result_fun = result.fun
            result_X = result.x
            RP_X = result_X[:len(Q)]

            # Check RP solution
            RP_variance = RP_X@Q@RP_X
            RP_vol = math.pow(RP_variance, 0.5)
            RC_target = (RP_variance)/len(RP_X)
            RC_L = ([RP_X[N]*Q[N]@RP_X for N, i in enumerate(RP_X)])
            RC_tolerance = math.pow(10, -5)
            RC_test = [abs(RC_i - RC_target) for RC_i in RC_L]

            if ((all(i <= RC_tolerance for i in RC_test)) == True):
                RP_solutions.append(tuple((RP_X, RP_vol)))

    RP_X_sorted = sorted(RP_solutions, key= lambda x: x[1])
    RP_X = RP_X_sorted[0]

    return RP_X_sorted, RP_X

#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
#------#
# Main #
#------#

# Variance-Covarariance Matrix Q
Q = np.array([[1, -0.9, 0.6], 
              [-0.9, 1, -0.2],
              [0.6, -0.2, 4]])

RP_param_dict = {'covariance_matrix': Q, 'LB_x': -1, 'UB_x': 1, 'X_sum': 1}
RP_report = sequential_mv_rp(**RP_param_dict)
RP_MV = RP_report[-1]

print(RP_MV)





