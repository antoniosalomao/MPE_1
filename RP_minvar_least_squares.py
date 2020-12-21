import pandas as pd
import numpy as np
import math
import scipy.optimize
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from matplotlib.pyplot import figure
import seaborn as sns

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------#
# Yahoo Finance #
#---------------#

def get_expected_ret_covar(tickers_L):
    '''
    Yahoo Finance
    DataFrames: Expected Returns, Covariance Matrix, Correlation Matrix
    '''
    yfinance_dict_i = {'tickers': sorted(tickers_L, reverse=False),
                         'start': '2015-01-01',
                           'end': '2019-12-31',
                      'interval': '1d'}

    df_main = yf.download(**yfinance_dict_i,)
    df_main.index = pd.to_datetime(df_main.index)

    # Slicing and cleaning DataFrame --> Price Series
    p_options = ['Adj Close', 'Close']
    df_ps = df_main.loc[:, [p_options[1]]].ffill(axis=0)
    df_ps.columns = df_ps.columns.droplevel()
    df_ps.columns = [col.replace('.SA', '') for col in list(df_ps.columns)]
    # Returns
    df_ret = np.log(df_ps).diff(1).fillna(method='ffill').dropna(how='any')
    df_exp_ret = pd.DataFrame(df_ret.mean(), columns=['Expected Return'])
    ret_numpy = np.array(df_exp_ret['Expected Return'].tolist())
    # Covariance
    df_covar = df_ret.cov()
    # Correlation
    df_correl = df_ret.corr()

    return df_exp_ret, df_covar, df_correl

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------#
# Optimization - Helper functions #
#---------------------------------#

def RP_X_sum(params_rp, covariance_matrix, C):
    '''
    Sum(weights) constraint
    '''
    X = params_rp[:len(covariance_matrix)]
    sum_X = sum(X) - C

    return sum_X

def F_RP_X(params_rp, covariance_matrix, rho):
    '''
    Objective Function F
    '''
    Q = covariance_matrix
    X, theta = params_rp[:Q.shape[0]], params_rp[-1]

    F = sum([math.pow(X[N]*Q[N]@X - theta, 2) for N, i in enumerate(X)]) + rho*X.T@Q@X

    return F

def RP_opt(covariance_matrix, LB_UB_x, X_sum, init_guess_X ,rho):
    '''
    Risk Parity optimization
    '''
    Q = covariance_matrix

    init_guess_theta = np.random.uniform(low=-10, high=10, size=1)
    init_guess_X_theta = np.array(list(init_guess_X) + list(init_guess_theta))

    bounds_X_theta = [LB_UB_x for _ in Q] + [(-1000, 1000)]

    opt_dict = { 'fun': F_RP_X,
                  'x0': init_guess_X_theta,
                'args': tuple((Q, rho)),
              'bounds': bounds_X_theta,
         'constraints': {'type': 'eq', 'fun': RP_X_sum, 'args': tuple((Q, X_sum))}}

    opt_report = minimize(**opt_dict, tol=math.pow(10, -16), options={'maxiter': math.pow(10, 16)})

    return opt_report

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------#
# Main Function #
#---------------#

def get_SMVRP_report_RP_solutions(covar, rho_L, rho_n_trials, LB_i, UB_i, weights_constraint, RP_RC_tolerance, rho_tolerance):
    '''
    Sequential Minimum Variance Risk Parity
    Complete RP Report
    '''
    all_solutions = []
    for N, rho_i in enumerate(rho_L):
        if (N == 0) | (len(all_solutions) == 0):
            initial_X = np.random.uniform(low=LB_i, high=UB_i, size=len(covar))
        elif (rho_i < rho_tolerance):
            rho_i = 0
        else:
            initial_X = all_solutions[-1][0]

        rp_opt_dict = { 'covariance_matrix': covar,
                                  'LB_UB_x': tuple((LB_i, UB_i)),
                                    'X_sum': weights_constraint,
                             'init_guess_X': initial_X, 
                                      'rho': rho_i}
        rho_i_solutions = []
        for trial in range(rho_n_trials):
            result = RP_opt(**rp_opt_dict)
            if (result.success == True):
                rho_i_solutions.append(tuple((result.x[:len(covar)], result.fun)))
        if len(rho_i_solutions) > 0:
            best_X = np.array(sorted(rho_i_solutions, key= lambda x: x[1])[0][0])
            port_variance= best_X@covar@best_X
            port_vol = math.pow(port_variance, 0.5)
            RC_L= ([best_X[N]*covar[N]@best_X for N, i in enumerate(best_X)])
            RRC_L = ([k/sum(RC_L) for k in RC_L])
            RC_target = port_variance/len(covar)
            distance_to_rp = sum([math.pow(RC_i - RC_target, 2) for N, RC_i in enumerate(RC_L)])
            target_herfindahl_index, herfindahl_index = (1/len(covar)),sum([math.pow(((best_X[N]*covar[N]@best_X)/port_variance), 2) for N, i in enumerate(best_X)])
            RC_report = tuple((best_X, rho_i, port_vol, RC_L, RRC_L, distance_to_rp, RC_target))
            all_solutions.append(RC_report)
            print('='*10, 'Rho: {:.12f}'.format(rho_i), '='*10)
            print('Hinderfahl Index: {:.20f}\t[Target: {:.2f}]'.format(herfindahl_index, target_herfindahl_index))
            print('Portfolio volatility: {:.10f} %'.format(port_vol))
            print('Distance to RisK Parity: {:.20f}'.format(distance_to_rp))
            print('Weights: ',  [float('{:.4f}'.format(x)) for x in best_X])
            print('RC: ', [float('{:.4f}'.format(x)) for x in RC_L], '\n')
        else:
            continue
        if rho_i == 0:
            break

    # RP Tolerance
    RP_solutions = []
    for N, portfolio in enumerate(all_solutions):
        RP_X, RP_rho, RP_vol, RP_RC, RP_RRC, RP_sqd_dist, RP_RC_target = portfolio
        RC_test = [abs(RC_i - RP_RC_target) for RC_i in RP_RC]
        if ((all(i <= RP_RC_tolerance for i in RC_test)) == True):
            RP_solutions.append(portfolio)
    
    # RP Report (sorted by rho)
    SMVRP_report = sorted(all_solutions, key=lambda x: x[1])

    func_result = tuple((SMVRP_report, RP_solutions))
    return func_result

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------#
# Fetching data #
#---------------#

eq_univ = ['VALE3.SA', 'ITUB4.SA', 'PETR4.SA', 'ABEV3.SA', 'RADL3.SA',
           'RENT3.SA', 'JBSS3.SA', 'EQTL3.SA', 'KLBN11.SA', 'TOTS3.SA']

yf_data = get_expected_ret_covar(tickers_L=eq_univ)
df_ret, df_covar, df_correl = yf_data[0]*252, yf_data[1]*252, yf_data[2]
ret, covar, correl = np.array(df_ret), np.array(df_covar), df_correl

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------#
# Main - Sequential min-variance RP #
#-----------------------------------#

covar_k1 = np.array([[1, -0.9, 0.6],
                     [-0.9, 1, -0.2],
                     [0.6, -0.2, 4]])

covar_k2 = np.array([[94.868, 33.750, 12.325, -1.178, 8.778],
                     [33.750, 445.642, 98.955, -7.901, 84.954],
                     [12.325, 98.955, 117.265, 0.503, 45.184],
                     [-1.178, -7.901, 0.503, 5.460, 1.057],
                     [8.778, 84.954, 45.184, 1.057, 34.126]])

rp_least_squares_dict = { 'covar': covar,
                          'rho_L': sorted([math.pow(2, i) for i in np.arange(-40, 15)], reverse=True),
                   'rho_n_trials': 25,
                           'LB_i': -1,
                           'UB_i': 1,
             'weights_constraint': 1,
                'RP_RC_tolerance': math.pow(10, -4),
                  'rho_tolerance': math.pow(10, -4)}

SMVRP_report_v0, RP_solutions_v0 = get_SMVRP_report_RP_solutions(**rp_least_squares_dict)
RP_X, RP_rho, RP_vol, RP_RC, RP_RRC, RP_sqd_dist, RP_target = SMVRP_report_v0[0]

print(RP_X)

rho_L = [k[1] for k in SMVRP_report_v0]
vol_L = [k[2] for k in SMVRP_report_v0]
sqd_rp_diff_L = [k[5] for k in SMVRP_report_v0]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------------#
# Matplotlib #
#------------#

# Asset allocation
plt.figure(figsize=(10, 8), dpi=115)
ax1 = plt.subplot(211)
ax1.bar(df_ret.index, RP_X, width=0.5)
ax1.set_ylim([0, 0.175])
plt.title('Figure 5 - Risk Parity Asset Allocation')
xlocs, xlabs = plt.xticks()
for i, v in enumerate(RP_X):
    ax1.text(xlocs[i] - 0.25, v + 0.0075, '{:.2f}%'.format(v*100), fontsize=8)

# Risk Contribution
ax2 = plt.subplot(212)
ax2.bar(df_ret.index, RP_RRC, width=0.30, label='Relative Risk Contribution', color='r', alpha=0.75)
ax2.set_ylim([0, 0.15])
plt.legend()
xlocs, xlabs = plt.xticks()
for i, v in enumerate(RP_RRC):
    ax2.text(xlocs[i] - 0.25, v + 0.0075, '{:.2f}%'.format(v*100), fontsize=8)

# Variance-Covariance Matrix
mask = np.triu(np.ones_like(df_covar, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title('Figure 3 - Covariance Matrix of Asset Returns')
sns.heatmap(df_covar, mask=mask, cmap=cmap, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=True)

# Correlation Matrix
mask = np.triu(np.ones_like(df_correl, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title('Figure 3 - Correlation Matrix of Asset Returns')
sns.heatmap(df_correl, mask=mask, cmap=cmap, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=True)

# Rho Dynamics
plt.figure(figsize=(10, 6), dpi=115)
plt.scatter(rho_L, sqd_rp_diff_L, c=vol_L, cmap='winter', label='Volatility')
plt.title('Figure 4 - Sequential Minimum Variance Risk Parity', size=12.5)
plt.ylabel(' "Distance" to Risk Parity', size=12.5, labelpad=2)
plt.xlabel(r'$ \/ \rho $',size=12.5)
plt.tick_params(axis='x', rotation=70)
cbar = plt.colorbar()
cbar.set_label('Volatility', size=12.5, labelpad=2)
plt.legend()
plt.gca().invert_xaxis()

# Multiple RP solutions
'''rho_L = [k[1] for k in RP_solutions]
vol_L = [k[2] for k in RP_solutions]
sqd_rp_diff_L = [k[5] for k in RP_solutions]'''

fig, ax3 = plt.subplots(figsize=(10, 8))
ax3.scatter(rho_L, vol_L, marker='x', color='r')
plt.title('Figure 6 - Risk Parity solutions over 20 trials', size=12.5)
ax3.set_ylabel('Volatility')
ax3.set_xlabel('Nth trial')
for i, txt in enumerate(vol_L):
    ax3.annotate('{:.2f}'.format(txt), (rho_L[i], vol_L[i]))
#plt.show()














