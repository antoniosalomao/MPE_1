import pandas as pd
import numpy as np
import math
import scipy.optimize
from scipy.optimize import minimize
from sympy import sin, cos, symbols, lambdify

# Variance-Covarariance Matrix Q
Q = np.array([[1, -0.9, 0.6], 
              [-0.9, 1, -0.2],
              [0.6, -0.2, 4]])

# Minimum Variance, Risk Parity Algorithm

# Risk contribution example
x = np.array([0.2, 0.3, 0.5])

rc_0 = x[0]*Q[0]@x
rc_1 = x[1]*Q[1]@x
rc_2 = x[2]*Q[2]@x

rc_sum = rc_0 + rc_1 + rc_2
port_variance = x.T@Q@x

print('Variance-Covariance Matrix: \n{}\n'.format(Q))
print('Portoflio weights: {}'.format(x))
print('Sum(RC): {}'.format(rc_sum))
print('Portoflio variance: {}'.format(port_variance))

# Objective function F

def weight(params_rp):

    X = params_rp[:3]
    sum_X = sum(X) - 1

    return sum_X

def F_RP(params_rp):
    X, theta = params_rp[:3], params_rp[-1]

    rho = math.pow(10, -8)
    A = [math.pow(X[N]*Q[N]@X - theta, 2) for N, i in enumerate(X)]
    B = sum(A) + rho*X.T@Q@X

    return B

initial_guess = [-2, -2, -2, 1]
bounds = [(-2, 1), (-2, 1), (-2, 1), (-1000, 1000)]

opt_dict = { 'fun': F_RP,
              'x0': initial_guess,
          'bounds': bounds,
     'constraints': {'type': 'eq', 'fun': weight}}

result = minimize(**opt_dict, tol=math.pow(10, -12))

x = np.array([0.57367876,  0.53099801, -0.10467676])

RC_i = ([x[N]*Q[N]@x for N, i in enumerate(x)])
print(RC_i)
print(sum(RC_i))
print((sum(x))**5)

A = [math.pow(x[N]*Q[N]@x - 1, 2) for N, i in enumerate(x)]

