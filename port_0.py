import pandas as pd
import numpy as np
from numpy import *
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import seaborn as sns
from scipy.optimize import minimize

# Fetching historic returns (10 BR stocks)
tickers_br = ['VALE3.SA', 'ITUB4.SA', 'PETR4.SA', 'ABEV3.SA', 'RADL3.SA',
              'RENT3.SA', 'JBSS3.SA', 'EQTL3.SA', 'KLBN11.SA', 'TOTS3.SA']

yfinance_dict = { 'tickers': sorted(tickers_br, reverse=False),
                    'start': '2015-01-01',
                      'end': '2019-12-31',
                 'interval': '1d'}

df_main = yf.download(**yfinance_dict,)
df_main.index = pd.to_datetime(df_main.index)

# Slicing and cleaning DataFrame --> Price Series
p_options = ['Adj Close', 'Close']
df_ps = df_main.loc[:, [p_options[1]]].ffill(axis=0)
df_ps.columns = df_ps.columns.droplevel()
df_ps.columns = [tick[:-3] for tick in list(df_ps.columns)]

print('\n')
print(df_ps.info())
print('\n')
print(df_ps.head())
print('\n')

# (1) Returns (log)
df_returns = np.log(df_ps).diff(1).dropna(how='all')
mu_returns = df_returns.mean()
mu_dict = {ticker:val for ticker, val in mu_returns.iteritems()}
for k, v in mu_dict.items():
    print('Ticker: {}'.format(k), '\t', 'Average return: {:.4f} %'.format(v*100))

# (2) Variance - Covariance Matrix (numpy) + correlation between asset returns
covar_returns = df_returns.cov()
correl_returns = df_returns.corr()
mask_covar = np.triu(np.ones_like(correl_returns, dtype=np.bool))
f, ax = plt.subplots(figsize=(10, 8))
cmap_i = sns.diverging_palette(h_neg=220, h_pos=10, as_cmap=True)
sns.heatmap(data=correl_returns, mask=mask_covar, cmap=cmap_i, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=True)
plt.show()

# Constrained Optimization (Mean-Variance Utility (missing lambda (risk aversion?))
# (-) Objective function        Q(w, f) = E[r] - (0.5)*(lambda)*Var[r] = expected_return - (0.5)*(lambda)*(port_variance)
# (i) Target volatility         g(w, f) = port_variance - sigma^2; g(w, f) = 0
# (ii) Max portfolio leverage   h(w, f) = np.sum(abs(weights)) - C; h(w, f) <= 0 

# Numerical Python - Ch. 6
# Sequential Least Squares SQuares Programming (SLSQP) Algorithm 

def get_ret_vol_mvutility(weights, d_ra):
    '''
    d_ra: risk aversion parameter
    '''
    weights = np.array(weights)
    expected_return = np.sum(np.array(mean(df_returns, axis=0))*weights*252)
    port_variance = weights.T@np.array(df_returns.cov())*252@weights
    Q = expected_return - d_ra*(0.5)*port_variance
    return np.array([expected_return, port_variance, Q])

def check_sum(C):
    '''
    C: Max leverage
    '''
    return lambda weights: C - np.sum(abs(weights))

def target_vol(sigma):
    '''
    sigma: target volatility
    '''
    return lambda weights: get_ret_vol_mvutility(weights, d_ra=1)[1] - (sigma**2)

def get_bounds(weights, LB, UB):
    '''
    LB: Lower bound
    UB: Upper bound
    '''
    w_B = tuple([[LB, UB] for w in list(range(len(weights)))])
    return w_B

g_cons = ({'type': 'eq', 
            'fun': target_vol(sigma=0.25)})
h_cons = ({'type': 'ineq', 
            'fun': check_sum(C=1.5)})
init_weights = np.full((1, len(mu_returns)), (1/len(mu_returns))).T 
G_bounds = get_bounds(weights=init_weights, LB=-0.2, UB=0.2)

opt_dict = { 'fun': lambda weights: get_ret_vol_mvutility(weights, d_ra=1)[2]*-1,
              'x0': init_weights,
          'method': 'SLSQP',
          'bounds': G_bounds,
     'constraints': [h_cons, g_cons]}

opt_results = minimize(**opt_dict)
opt_weights = opt_results.x
opt_check = get_ret_vol_mvutility(weights=opt_weights, d_ra=1)
df_final = pd.DataFrame(opt_weights, index=mu_returns.index, columns=['Optimal Weights'])

print('\n')
print(opt_results)
print('\n')
print('Portfolio Return: {:.4f}%'.format(opt_check[0]*100))
print('Portfolio Volatility: {:.4f}%'.format(np.sqrt(opt_check[1]*100)))
print('(?) MV Utility: {:.4f}'.format(opt_check[2]))
print('\n')
print(df_final)
print('\nSum: {}'.format(np.sum(opt_weights)))
