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

    arg min problem --> d_ra * port_variance -------> d_ra^(-1) * expected_return

    '''
    weights = np.array(weights)
    expected_return = returns@weights*252
    port_variance = weights.T@np.array(covar)*252@weights
    Q = (0.5)*port_variance - (d_ra**(-1))*expected_return
    return np.array([expected_return, port_variance, Q])

def target_vol(sigma, returns_vol_arg, covar_vol_arg):
    '''
    sigma: target volatility
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
#----------#
# Equities #
#----------#

eq_0 = ['VALE3.SA', 'ITUB4.SA', 'PETR4.SA', 'ABEV3.SA', 'RADL3.SA']
eq_1 = ['RENT3.SA', 'JBSS3.SA', 'EQTL3.SA', 'KLBN11.SA', 'TOTS3.SA']
eq_univ_yf = eq_0 + eq_1

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#
#------#
# Main #
#------#

eq_dict = {'eq_0': [eq_0],
           'eq_1': [eq_1]}

for k, v in eq_dict.items():
    '''
    (1) Equities dictionary ...
        k: 'eq_i' (Equity groups)
        v[0]: equities
        v[1]: price series
        v[2]: return series
        v[3]: expected return
        v[4]: covariance matrix
        v[5]: volatility target
        v[6]: max. leverage

        v[7]: (2) Expected return dictionary ... 
                  k: equities
                  v[0]: expected returns
                  v[1]: optimal weight (individual)

        v[8]: [0]: optimal weights
              [1]: expected return
              [2]: volatility
    '''
    # Fetching data
    df_yf = yfinance_df(v[0])
    # Slicing and cleaning DataFrame --> Price Series
    p_options = ['Adj Close', 'Close']
    df_ps = df_yf.loc[:, [p_options[1]]].ffill(axis=0)
    df_ps.columns = df_ps.columns.droplevel()
    df_ps.columns = [tick[:-3] for tick in list(df_ps.columns)]
    # Appending back to dictionary
    eq_dict[k].append(df_ps)
    # Computing log retuns
    df_ret = np.log(df_ps).diff(1).fillna(method='ffill').dropna(how='any')
    eq_dict[k].append(df_ret)
    # Expected returns
    df_exp_ret = pd.DataFrame(df_ret.mean(), columns=['Expected Return'])
    eq_dict[k].append(df_exp_ret)
    # Covariance of returns
    df_covar = df_ret.cov()
    eq_dict[k].append(df_covar)
    # Dictionary of expected returns
    expected_returns_dict = {}
    for i, row in df_exp_ret.iterrows():
        expected_returns_dict[i] = []
        expected_returns_dict[i].append(row['Expected Return'])
    # Appending group constraints
    # (1) Target Volatility
    # (2) Leverage
    # (3) Individual weights (LB, UB)
    if k == 'eq_0':
        tuple_vol_leverage = tuple((0.40, 1))
        lb_ub = tuple((-0.25, 0.25))
        eq_dict[k].extend((tuple_vol_leverage, lb_ub))
    elif k == 'eq_1':
        tuple_vol_leverage = tuple((0.40, 1))
        lb_ub = tuple((-0.25, 0.25))
        eq_dict[k].extend((tuple_vol_leverage, lb_ub))
    # Appending 2nd dictionary to original dictionary
    eq_dict[k].append(expected_returns_dict)

# Equity group j

def node_optimization(eq_dict):
    for k in eq_dict:
        eq_j_main  = eq_dict[k]
        eq_j_L = eq_j_main[0]
        eq_j_return_series_df = eq_j_main[2]
        eq_j_covar = eq_j_main[4]
        eq_j_constraints = eq_j_main[5]
        eq_j_vol_target = eq_j_constraints[0]
        eq_j_max_leverage = eq_j_constraints[1]
        eq_j_bounds = eq_j_main[6]
        eq_j_dict = eq_j_main[7]

        # Expected Returns array
        eq_j_expected_ret = eq_j_main[7]
        eq_j_ret_arr = np.array([v[0] for v in eq_j_expected_ret.values()])

        # Apending group 0 constraints
        #eq_j_g_cons = ({'type': 'eq', 'fun': target_vol(sigma=eq_j_vol_target, returns_vol_arg=eq_j_ret_arr, covar_vol_arg=eq_j_covar)})
        eq_j_h_cons = ({'type': 'ineq', 'fun': check_sum(C=eq_j_max_leverage)})

        eq_j_mvutility_L = []
        s_n = 0
        n_trials = 10
        for i in range(n_trials):
            '''
            Attempting to find true global minima via iteration
            '''
            init_weights = np.random.uniform(low=eq_j_bounds[0], high=eq_j_bounds[1], size=(len(eq_j_L),))
            G_bounds = get_bounds(weights=init_weights, LB=eq_j_bounds[0], UB=eq_j_bounds[1])

            opt_dict = { 'fun': lambda weights: get_ret_vol_mvutility(weights, d_ra=1, returns=eq_j_ret_arr, covar=eq_j_covar)[2],
                          'x0': init_weights,
                      'method': 'SLSQP',
                      'bounds': G_bounds,
                 'constraints': [eq_j_h_cons]}

            try:
                opt_results = minimize(**opt_dict)
                #print(opt_results)
            except ValueError as e:
                continue

            opt_weights = opt_results.x
            opt_success = opt_results.success
            opt_mvutility = opt_results.fun

            if (opt_success == True):
                eq_j_mvutility_L.append(tuple((opt_weights, opt_mvutility)))
                s_n += 1
            else:
                continue

        # Selecting 'best' solution
        weights_max_mvutility = min(np.array(eq_j_mvutility_L), key=itemgetter(1))
        opt_W = weights_max_mvutility[0]
        opt_MV = weights_max_mvutility[1]

        # Appending (saving) optimal weights
        for N, (k_v) in enumerate(eq_j_dict.items()):
            opt_w_i = opt_W[N]
            eq_j_dict[k_v[0]].append(opt_w_i)
    
        # Computing portfolio metrics
        port_j_ret = get_ret_vol_mvutility(weights=opt_W, d_ra=1, returns=eq_j_ret_arr, covar=eq_j_covar)[0]
        port_j_vol = get_ret_vol_mvutility(weights=opt_W, d_ra=1, returns=eq_j_ret_arr, covar=eq_j_covar)[1]
        port_j_mvutility = get_ret_vol_mvutility(weights=opt_W, d_ra=1, returns=eq_j_ret_arr, covar=eq_j_covar)[2]
        port_j_tuple = tuple((opt_W, port_j_ret, port_j_vol, port_j_mvutility))
        eq_dict[k].append(port_j_tuple)

eq_0_main = node_optimization(eq_dict)

eq_0_main = eq_dict['eq_0']
eq_1_main = eq_dict['eq_1']

eq_0_port_dict = eq_0_main[8]
eq_1_port_dict = eq_1_main[8]

eq_0_dict = eq_0_main[7]
eq_1_dict = eq_1_main[7]

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#
#-----------#
# Functions #
#-----------#

# Combining weights?
# https://quant.stackexchange.com/questions/11200/calculate-correlation-between-two-sub-portfolios-and-the-combined-portfolio

print(eq_0_dict)
print(eq_1_dict)

eq_univ = [eq.replace('.SA', '') for eq in eq_univ_yf]

eq_0_port_W = []
eq_1_port_W = []

for eq in eq_univ:
    if eq in eq_0_dict:
        eq_w = eq_0_dict[eq][1]
        tuple_eq = tuple((eq, eq_w))
        eq_0_port_W.append(tuple_eq)
    elif eq not in eq_0_dict:
        tuple_eq = tuple((eq, 0))
        eq_0_port_W.append(tuple_eq)

for eq in eq_univ:
    if eq in eq_1_dict:
        eq_w = eq_1_dict[eq][1]
        tuple_eq = tuple((eq, eq_w))
        eq_1_port_W.append(tuple_eq)
    elif eq not in eq_1_dict:
        tuple_eq = tuple((eq, 0))
        eq_1_port_W.append(tuple_eq)

print(eq_0_port_W)
print(eq_1_port_W)


portfolio_weights_df = pd.DataFrame(index=eq_univ, columns=['Portfolio 1', 'Portfolio 2'])
portfolio_weights_df['Portfolio 1'] = [eq[1] for eq in eq_0_port_W]
portfolio_weights_df['Portfolio 2'] = [eq[1] for eq in eq_1_port_W]
print(portfolio_weights_df)

portfolio_weights_df.sort_index(inplace=True)
print(portfolio_weights_df)

main_df = yfinance_df(eq_univ_yf)
p_options = ['Adj Close', 'Close']
df_ps = main_df.loc[:, [p_options[1]]].ffill(axis=0)
df_ps.columns = df_ps.columns.droplevel()
df_ps.columns = [tick[:-3] for tick in list(df_ps.columns)]
df_ret = np.log(df_ps).diff(1).fillna(method='ffill').dropna(how='any')
 
df_covar = df_ret.cov()

covar_arr = np.array(df_covar)
port_covar_matrix = np.array(portfolio_weights_df).T@np.array(df_covar)@np.array(portfolio_weights_df)*252
print(port_covar_matrix)

'''
port_var =w*covar*w
'''

print(eq_0_port_dict)
print(eq_1_port_dict)


'''
Optimize lvl 2
'''













