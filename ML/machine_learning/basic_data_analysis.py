#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ./machine_learning/helper_func import binned_scatter
#from machine_learning.helper_func import binned_scatter
import os


#%% Setup parameters and load data

area = 'CE'
res_folder = './Results/basic_data_analysis/{}/'.format(area)
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
data_folder = './prepared_data/{}/'.format(area)

input_actual = pd.read_hdf(data_folder+'input_actual.h5')
input_forecast = pd.read_hdf(data_folder+'input_forecast.h5')
outputs = pd.read_hdf(data_folder+'outputs.h5')

data = pd.concat([input_actual], axis=1)
#data = pd.concat([input_actual, input_forecast, outputs], axis=1)


#%% Add absolute values (to see more linear relations)

abs_cols_forecast =  'abs_' + input_forecast.filter(regex='\w*ramp\w*').columns
data[abs_cols_forecast] = input_forecast.filter(regex='\w*ramp\w*').abs()
abs_cols_act_1 =  'abs_' + input_actual.filter(regex='\w*ramp').columns
data[abs_cols_act_1] = input_actual.filter(regex='\w*ramp').abs()
abs_cols_act_2 = 'abs_' + input_actual.filter(regex='^forecast_error').columns
data[abs_cols_act_2] = input_actual.filter(regex='^forecast_error').abs()


#%% Calc correlations

corr_pearson = data.corr(method='pearson')
corr_kendall = data.corr(method='kendall')
corr_spearman = data.corr(method='spearman')

corr_spearman.to_csv(res_folder+'corr_spearman.csv')
corr_pearson.to_csv(res_folder+'corr_pearson.csv')


#%% Actual inputs correlation heatmap

fig, ax= plt.subplots(1,3, figsize=(10,15))

output_cols = ['f_ext', 'f_5_quantile', 'f_95_quantile', 'f_abs_max', 'f_abs_95_quantile',
               'f_rocof', 'f_abs_integral', 'f_integral']

input_cols = np.concatenate([input_actual.columns, abs_cols_act_1, abs_cols_act_2])

vmax = np.amax([corr_pearson.loc[input_cols, output_cols].abs().max().max(),
                corr_kendall.loc[input_cols, output_cols].abs().max().max(),
                corr_spearman.loc[input_cols, output_cols].abs().max().max()])
sns.heatmap(corr_pearson.loc[input_cols, output_cols], xticklabels=True, yticklabels=True,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, ax=ax[0], cbar=False, linewidths=1)
sns.heatmap(corr_kendall.loc[input_actual.columns, output_cols], xticklabels=True, yticklabels=False,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, ax=ax[1], cbar=False, linewidths=1)
sns.heatmap(corr_spearman.loc[input_actual.columns, output_cols], xticklabels=True, yticklabels=False,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, ax=ax[2], linewidths=1)
ax[0].set_title('Pearson')
ax[1].set_title('Kendall')
ax[2].set_title('Spearman')

plt.tight_layout()
plt.savefig(res_folder+'correlation_coeff_actual.svg', bbox_inches='tight')


#%% Forecast inputs correlation heatmap

fig, ax= plt.subplots(1,3, figsize=(10,8))

input_cols = np.concatenate([input_forecast.columns, abs_cols_forecast])

vmax = np.amax([corr_pearson.loc[input_cols, output_cols].abs().max().max(),
                corr_kendall.loc[input_cols, output_cols].abs().max().max(),
                corr_spearman.loc[input_cols, output_cols].abs().max().max()])
sns.heatmap(corr_pearson.loc[input_cols, output_cols], xticklabels=True, yticklabels=True,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, ax=ax[0], cbar=False, linewidths=1)
sns.heatmap(corr_kendall.loc[input_actual.columns, output_cols], xticklabels=True, yticklabels=False,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, ax=ax[1], cbar=False, linewidths=1)
sns.heatmap(corr_spearman.loc[input_actual.columns, output_cols], xticklabels=True, yticklabels=False,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, ax=ax[2], linewidths=1)
ax[0].set_title('Pearson')
ax[1].set_title('Kendall')
ax[2].set_title('Spearman')

plt.tight_layout()
plt.savefig(res_folder+'correlation_coeff_forecast.svg', bbox_inches='tight')


#%% Correlation heatmap for input features

fig= plt.figure(figsize=(10,10))

ind = list(input_actual.columns) + list(input_forecast.columns)
vmax = corr_pearson.loc[ind, ind].abs().max().max()
sns.heatmap(corr_pearson.loc[ind, ind], xticklabels=True, yticklabels=True,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, cbar=True, linewidths=1)

plt.savefig(res_folder+'correlation_coeff_inputs.svg', bbox_inches='tight')


# %% binned scatterplots of forecast data and certain outputs

for var in ['f_ext']:
    fig, ax= plt.subplots(3,6, figsize=(13,7))

    for i,col in enumerate(input_forecast.columns):
        binned_scatter(input_forecast[col], outputs.loc[:,var], col, corr_pearson.loc[:,var], ax.flatten()[i])

    ax[1,0].set_ylabel(var, fontsize=10)
    plt.tight_layout()

    plt.savefig(res_folder+'binned_scatter_plot_{}_frac20_forecast.png'.format(var),
                dpi=200,
                bbox_inches='tight')
    plt.clf()

# %% binned scatterplots of actual data and certain outputs

for var in ['f_ext']:
    fig, ax= plt.subplots(5,10,figsize=(30,12))

    for i,col in enumerate(input_actual.columns):
        binned_scatter(input_actual[col], outputs.loc[:,var], col, corr_pearson.loc[:,var], ax.flatten()[i])

    ax[2,0].set_ylabel(var, fontsize=10)
    plt.tight_layout()

    plt.savefig(res_folder+'binned_scatter_plot_{}_frac20_actual.png'.format(var),
                dpi=200,
                bbox_inches='tight')
    plt.clf()
