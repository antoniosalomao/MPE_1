import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import seaborn as sns

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
new_cols = [tick[:-3] for tick in list(df_ps.columns)]
df_ps.columns = new_cols

print('\n')
print(df_ps.info())
print(df_ps.head())
print('\n')

# Calculating the (Sample) Variance-Covariance Matrix

# (1) Returns (log)
df_returns = np.log(df_ps).diff(1).dropna(how='all')
mu_returns = df_returns.mean()
mu_dict = {ticker:val for ticker, val in mu_returns.iteritems()}
for k, v in mu_dict.items():
    print('Ticker: {}'.format(k), '\t', 'Average return: {:.4f} %'.format(v*100))

# (2) Covariance Matrix (numpy) + correlation between asset returns
covar_returns = df_returns.cov()
correl_returns = df_returns.corr()
mask_covar = np.triu(np.ones_like(correl_returns, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap_i = sns.diverging_palette(h_neg=220, h_pos=10, as_cmap=True)
sns.heatmap(data=correl_returns, mask=mask_covar, cmap=cmap_i, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=True)
#plt.show()

# Arrays
returns = np.array([v for v in mu_dict.values()])
covar = np.array(covar_returns)
weights = np.random.uniform(low=0.1, high=0.15, size=(10,))
expected_return = returns.T@weights
port_variance = weights.T@covar@weights
port_stdev = np.sqrt(port_variance)

print(expected_return)
print(port_variance)
print(port_stdev)

# Optimization
# ...








