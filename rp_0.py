import pandas as pd
import numpy as np
from numpy import *
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import seaborn as sns
from scipy.optimize import minimize
from operator import itemgetter
import seaborn as sns

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#
#-----------#
# Functions #
#-----------#


def yfinance_df(tickers_L):
    '''
    Returns a dataframe from the yahoo finance library
    '''
    yfinance_dict_i = {'tickers': sorted(tickers_L, reverse=False),
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


eq_univ = ['VALE3.SA', 'ITUB4.SA', 'PETR4.SA', 'ABEV3.SA', 'RADL3.SA',
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
df_ps.columns = [col.replace('.SA', '') for col in list(df_ps.columns)]
# Computing log retuns
df_ret = np.log(df_ps).diff(1).fillna(method='ffill').dropna(how='any')
# Expected returns
df_exp_ret = pd.DataFrame(df_ret.mean(), columns=['Expected Return'])
ret_numpy = np.array(df_exp_ret['Expected Return'].tolist())
# Covariance of returns
df_covar = df_ret.cov()
covar_numpy = np.array(df_covar)
# Correlation of returns
df_correl = df_ret.corr()
# Dictionary of expected returns
expected_returns_dict = {}
for i, row in df_exp_ret.iterrows():
    expected_returns_dict[i] = []
    expected_returns_dict[i].append(row['Expected Return'])

g_cons = ({'type': 'eq',
           'fun': target_vol(sigma=0.25, returns_vol_arg=df_exp_ret['Expected Return'], covar_vol_arg=df_covar)})
h_cons = ({'type': 'eq',
           'fun': check_sum(C=1)})



# Covariance
mask = np.triu(np.ones_like(df_correl, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df_correl, mask=mask, cmap=cmap, center=0, square=True,linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=True)


LB_i = 0
UB_i = 0.1

eq_mvutility_L = []
s_n = 0
n_trials = 100
for i in range(n_trials):

    init_weights = np.random.uniform(low=LB_i, high=UB_i, size=(len(eq_univ)))
    G_bounds = get_bounds(weights=init_weights, LB=LB_i, UB=UB_i)

    opt_dict = { 'fun': lambda weights: get_ret_vol_mvutility(weights, d_ra=1, returns=ret_numpy, covar=covar_numpy)[2],
                  'x0': init_weights,
              'method': 'SLSQP',
              'bounds': G_bounds,
         'constraints': [h_cons]}

    try:
        opt_results = minimize(**opt_dict, tol=math.pow(10, -10))
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

# Final Mean-Variance Allocation
df_mv_allocation = pd.DataFrame(
    data=Q_W, index=df_ps.columns, columns=['MV Weights'])

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#
#-------------------------#
# Matplotlib - Intermezzo #
#-------------------------#

# Risk Contribution, Relative Risk Contribution
rc = []
rrc = []
for N, w in enumerate(Q_W):

    # Risk contribution
    rc_i = (252*w*np.sum([w_j*covar_numpy[N, n]
                          for n, w_j in enumerate(Q_W)]))/(Q_variance**0.5)
    rc.append(rc_i)

    # Relative risk contribution
    rrc.append(rc_i/Q_variance**0.5)

risk_contribution_df = pd.DataFrame(
    data=None, columns=None, index=df_ps.columns)
risk_contribution_df['RC'] = rc
risk_contribution_df['RRC'] = rrc

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(
    10, 8), sharex=True, sharey=True)

ax = axes[0]
ax.set_ylim([0, 0.20])
df_mv_allocation['MV Weights'].plot.bar(ax=ax, x=df_mv_allocation.index)
tile_ax_string = r' (1/N) Naive diversification: $ \/ \sum_{i=0} ^N \omega_i = 1, \omega_i = \frac{1}{N} \/ \forall \/ i $'
ax.set_title(size=14, label=tile_ax_string, y=1.25, pad=-27.5)
xlocs, xlabs = plt.xticks()
for i, v in enumerate(df_mv_allocation['MV Weights']):
    ax.text(xlocs[i] - 0.225, v + 0.01, '{:.2f}%'.format(v*100))

ax = axes[1]
risk_contribution_df['RRC'].plot.bar(
    ax=ax,  x=risk_contribution_df.index, title='Relative Risk Contribution')
ax.set_title(size=14, label=r'Relative Risk Contribution')
xlocs, xlabs = plt.xticks()
for i, v in enumerate(risk_contribution_df['RRC']):
    ax.text(xlocs[i] - 0.225, v + 0.01, '{:.2f}%'.format(v*100))

for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.tight_layout(pad=2.5)

#-----------------------------------------------------------------------------------------------------------------------------------------------------–-----------------------#

#--------------------#
# Main - Risk Parity #
#--------------------#

# Long-Only Formulation (Karush-Khun-Tucker conditions OK)

obj_fun_rp_weights = []
s = []
s_n = 0
n_trials = 100
ln_constraint = {'type': 'eq', 'fun': lambda y: np.sum(np.log(y)) - 1}
for i in range(n_trials):

    initial_guess_rp = np.random.uniform(low=0.01, high=5, size=(len(eq_univ)))
    G_bounds_rp = get_bounds(weights=initial_guess_rp, LB=0.01, UB=5)

    rp_opt_dict = {'fun': lambda y: math.pow((y.T@covar_numpy@y*252), 0.5),
                   'x0': initial_guess_rp,
                   'bounds': G_bounds_rp,
                   'constraints': [ln_constraint]}

    try:
        opt_results_rp = minimize(**rp_opt_dict, tol=math.pow(10, -100))
    except ValueError as e:
        continue

    opt_weights_rp = opt_results_rp.x
    opt_success_rp = opt_results_rp.success
    opt_stdev_rp = opt_results_rp.fun

    if (opt_success_rp == True):
        obj_fun_rp_weights.append(tuple((opt_weights_rp, opt_stdev_rp)))

weights_min_func = min(np.array(obj_fun_rp_weights), key=itemgetter(1))
rp_x_original = weights_min_func[0]

weight_sum = np.sum(rp_x_original)
rp_normalized = []

for w in rp_x_original:
    w_i = w/weight_sum
    rp_normalized.append(w_i)

rp_normalized = np.array(rp_normalized)
RP_variance = rp_normalized.T@covar_numpy@rp_normalized*252
RP_stdev = math.pow(RP_variance, 0.5)

RP_rc = []
RP_rrc = []

for N, w in enumerate(rp_normalized):

    # Risk contribution
    rc_i = (252*w*np.sum([w_j*covar_numpy[N, n] for n, w_j in enumerate(rp_normalized)]))/RP_stdev
    RP_rc.append(rc_i)

    # Relative risk contribution
    RP_rrc.append(rc_i/RP_stdev)

risk_contribution_df = pd.DataFrame(data=None, columns=None, index=df_ps.columns)
risk_contribution_df['RC'] = RP_rc
risk_contribution_df['RRC'] = RP_rrc
risk_contribution_df['Allocation'] = list(rp_normalized)

print(risk_contribution_df)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(
    10, 8), sharex=True, sharey=True)

ax = axes[0]
ax.set_ylim([0, 0.175])
risk_contribution_df['Allocation'].plot.bar(ax=ax, x=df_mv_allocation.index)
tile_ax_string = r' Risk-Parity Allocation: $ \/ \sum_{i=0} ^N \omega_i = 1, 0 \leq \omega_i \leq 1 \/ \forall \/ i $'
ax.set_title(size=14, label=tile_ax_string, y=1.25, pad=-27.5)
xlocs, xlabs = plt.xticks()
for i, v in enumerate(risk_contribution_df['Allocation']):
    ax.text(xlocs[i] - 0.225, v + 0.01, '{:.2f}%'.format(v*100))

ax = axes[1]
risk_contribution_df['RRC'].plot.bar(
    ax=ax,  x=risk_contribution_df.index, title='Relative Risk Contribution')
ax.set_title(size=14, label=r'Relative Risk Contribution')
xlocs, xlabs = plt.xticks()
for i, v in enumerate(risk_contribution_df['RRC']):
    ax.text(xlocs[i] - 0.225, v + 0.01, '{:.2f}%'.format(v*100))

for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.tight_layout(pad=2.5)
plt.show()
