import pandas as pd
import numpy as np
from numpy import *
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import seaborn as sns
from scipy.optimize import minimize
from operator import itemgetter

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#
#-----------#
# Functions #
#-----------#

def yfinance_df(tickers_L):
    '''
    Returns a dataframe from the yahoo finance library
    '''
    yfinance_dict_i = { 'tickers': sorted(tickers_L, reverse=False),
                          'start': '2015-01-01',
                            'end': '2020-10-30',
                       'interval': '1d'}

    df_main = yf.download(**yfinance_dict_i,)
    df_main.index = pd.to_datetime(df_main.index)

    return df_main

def get_ret_vol_mvutility(weights, d_ra, returns, covar):
    '''
    d_ra: risk aversion parameter
    d_ra --> infinity --> minimum variance portfolio

    *** arg max problem --> expected_return - d_ra * port_variance
    *** arg min problem --> (0.5)*port_variance - d_ra^(-1) * expected_return

    '''
    weights = np.array(weights)
    expected_return = returns@weights*252
    port_variance = weights.T@np.array(covar)*252@weights
    Q = (0.5)*port_variance - (d_ra**(-1))*expected_return
    return np.array([expected_return, port_variance, Q])

def target_vol(sigma, returns_vol_arg, covar_vol_arg):
    '''
    Sigma: Target volatility
    '''
    return lambda weights: get_ret_vol_mvutility(weights, d_ra=1, returns=returns_vol_arg, covar=covar_vol_arg)[1] - (sigma**2)

def check_sum(C):
    '''
    C: Max leverage
    '''
    return lambda weights: C - np.sum(abs(weights))

def get_bounds(weights, LB, UB):
    '''
    LB: Lower bound
    UB: Upper bound
    '''
    w_B = np.array(tuple([(LB, UB) for w in list(range(len(weights)))]))
    return w_B

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#
#-------------------#
# Equities Universe #
#-------------------#

eq_univ = [ 'VALE3.SA', 'ITUB4.SA', 'PETR4.SA', 'ABEV3.SA', 'RADL3.SA',
            'RENT3.SA', 'JBSS3.SA', 'EQTL3.SA', 'KLBN11.SA', 'TOTS3.SA']

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#
#-----------------------------------#
# Main - Mean-Variance Optimization #
#-----------------------------------#

# Fetching data
df_yf = yfinance_df(eq_univ)
# Slicing and cleaning DataFrame --> Price Series
p_options = ['Adj Close', 'Close']
df_ps = df_yf.loc[:, [p_options[1]]].ffill(axis=0)
df_ps.columns = df_ps.columns.droplevel()
df_ps.columns = [tick[:-3] for tick in list(df_ps.columns)]
# Computing log retuns
df_ret = np.log(df_ps).diff(1).fillna(method='ffill').dropna(how='any')
# Expected returns
df_exp_ret = pd.DataFrame(df_ret.mean(), columns=['Expected Return'])
ret_numpy = np.array(df_exp_ret['Expected Return'].tolist())
# Covariance of returns
df_covar = df_ret.cov()
covar_numpy = np.array(df_covar)
# Dictionary of expected returns
expected_returns_dict = {}
for i, row in df_exp_ret.iterrows():
    expected_returns_dict[i] = []
    expected_returns_dict[i].append(row['Expected Return'])

g_cons = ({'type': 'eq',
           'fun': target_vol(sigma=0.25, returns_vol_arg=df_exp_ret['Expected Return'], covar_vol_arg=df_covar)})
h_cons = ({'type': 'ineq',
           'fun': check_sum(C=1)})

eq_mvutility_L = []
s_n = 0
n_trials = 10
for i in range(n_trials):
    '''
    Attempting to find true global minima via iteration
    '''
    init_weights = np.random.uniform(low=0, high=0.15, size=(len(eq_univ)))
    G_bounds = get_bounds(weights=init_weights, LB=0, UB=0.15)

    opt_dict = { 'fun': lambda weights: get_ret_vol_mvutility(weights, d_ra=1, returns=ret_numpy, covar=covar_numpy)[2],
                    'x0': init_weights,
                'method': 'SLSQP',
                'bounds': G_bounds,
            'constraints': [g_cons, h_cons]}

    try:
        opt_results = minimize(**opt_dict)
        print(opt_results)
    except ValueError as e:
        continue

    opt_weights = opt_results.x
    opt_success = opt_results.success
    opt_mvutility = opt_results.fun

    if (opt_success == True):
        eq_mvutility_L.append(tuple((opt_weights, opt_mvutility)))
        s_n += 1
    else:
        continue

# Selecting 'best' solution
weights_max_mvutility = min(np.array(eq_mvutility_L), key=itemgetter(1))
Q_W = weights_max_mvutility[0]
Q_MV = weights_max_mvutility[1]
Q_return, Q_variance, Q_mv = get_ret_vol_mvutility(weights=Q_W, d_ra=1, returns=ret_numpy, covar=covar_numpy)

print('\n')
print('Q Return: {:.2f}'.format(Q_return))
print('Q Variance: {:.2f}'.format(Q_variance))
print('Q Utility: {:.2f}'.format(Q_MV*-1))

# Final allocation
df_mv_allocation = pd.DataFrame(data=Q_W, index=df_ps.columns, columns=['MV Weights'])

print(df_mv_allocation)

plt.figure(figsize=(10, 6))
plt.bar(x=df_mv_allocation.index, height=df_mv_allocation['MV Weights'])
plt.title('Mean-variance Weights')
plt.xlabel('Equities')
#plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------#
# Main - Risk Parity #
#--------------------#

# Risk Contribution
rc = []
rrc = []
for N, w in enumerate(Q_W):

    # Risk contribution
    rc_i = (252*w*np.sum([w_j*covar_numpy[N, n] for n, w_j in enumerate(Q_W)]))/(Q_variance**0.5)
    rc.append(rc_i)
    
    # Relative risk contribution
    rrc.append(rc_i/Q_variance**0.5)

risk_contribution_df = pd.DataFrame(data=None, columns=None, index=None)
risk_contribution_df['RC'] = rc
risk_contribution_df['RRC'] = rrc

print(risk_contribution_df)
print(sum(rc))


