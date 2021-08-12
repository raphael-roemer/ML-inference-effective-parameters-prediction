#%%
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np
import os


#%% Setup and load raw input data

area = 'CE'  

actual_cols = ['load', 'gen_biomass', 'gen_lignite', 'gen_coal_gas', 'gen_gas',
               'gen_hard_coal', 'gen_oil', 'gen_oil_shale', 'gen_fossil_peat',
               'gen_geothermal', 'gen_pumped_hydro', 'gen_run_off_hydro',
               'gen_reservoir_hydro', 'gen_marine', 'gen_nuclear', 'gen_other_renew',
               'gen_solar', 'gen_waste', 'gen_wind_off', 'gen_wind_on', 'gen_other']

forecast_cols = ['load_day_ahead', 'scheduled_gen_total','prices_day_ahead',
                 'solar_day_ahead','wind_off_day_ahead', 'wind_on_day_ahead']


folder = './prepared_data/{}/'.format(area)
plot_folder = folder + 'documentation_data_cleansing/'

if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

raw_input_data = pd.read_hdf(folder + 'raw_input_data.h5')


#%% Inspection of data distribution as boxplot

plt.figure(figsize=(7,4))
sns.boxplot(data=(raw_input_data-raw_input_data.min())/(raw_input_data.max() -raw_input_data.min()),
            whis=(5,95))
plt.xticks(rotation=60,horizontalalignment='right')

plt.savefig(plot_folder+'raw_data_distribution_inspect_outliers.png', dpi=200,
            bbox_inches='tight')

#%% Inspection of data distribution as histograms

fig,ax=plt.subplots(figsize=(20,20))
raw_input_data.hist(log=True,ax=ax, bins=100)
plt.tight_layout()

plt.savefig(plot_folder+'raw_data_histograms.svg', bbox_inches='tight')

#%% plot all time series

fig,ax=plt.subplots(figsize=(8,20))
raw_input_data.plot(subplots=True, ax=ax)
plt.tight_layout()
plt.savefig(plot_folder+'all_raw_time_series.png', dpi=200, bbox_inches='tight')

#%% Inpect single time series at unrealistic points

vmin=10000
vmax=10000000
feature = 'total_gen'

for i in np.argwhere(raw_input_data.loc[:,feature].values<vmin)[:20,0]:
    plt.figure()
    plt.plot(raw_input_data.loc[:,feature].iloc[np.clip(i-500,0,None):i+500])
    plt.xticks(rotation=60,horizontalalignment='right')

for i in np.argwhere(raw_input_data.loc[:,feature].values>vmax)[:10,0]:
    plt.figure()
    plt.plot(raw_input_data.loc[:,feature].iloc[np.clip(i-500,0,None):i+500])
    plt.xticks(rotation=60,horizontalalignment='right')

#%% Split into forecast and actual data and convert index to local timezone of frequency data

input_forecast = raw_input_data.loc[:, raw_input_data.columns.intersection(forecast_cols)]
input_actual = raw_input_data.loc[:, raw_input_data.columns.intersection(actual_cols)]

tzs = {'GB':'GB', 'Nordic':'Europe/Helsinki', 'CE':'CET', 'DE':'CET', 'SE':'Europe/Helsinki', 'CH':'CET'}
input_actual.index = input_actual.index.tz_convert(tzs[area])
input_forecast.index = input_forecast.index.tz_convert(tzs[area])

#%% Processing and cleansing
# Mark data points that appear to be unrealistic in the raw_data_histograms
# and in the single time series plots


if area=='GB':
    input_actual.load[(input_actual.load<10000)] = np.nan
    input_forecast.scheduled_gen_total[(input_forecast.scheduled_gen_total<12000)] =np.nan
    input_forecast.load_day_ahead[(input_forecast.load_day_ahead<10000)] =np.nan
    input_actual.gen_nuclear[(input_actual.gen_nuclear<3000)] = np.nan
    input_forecast.solar_day_ahead[input_forecast.solar_day_ahead>10000]=np.nan
    
elif area=='Nordic':
    input_actual.gen_biomass[(input_actual.gen_biomass<260)] = np.nan
    input_actual.gen_other_renew[(input_actual.gen_other_renew>75)] = np.nan

    
elif area=='CE':
    input_actual.gen_waste[(input_actual.gen_waste>4000)] = np.nan
    input_forecast.prices_day_ahead[(input_forecast.prices_day_ahead>200)] = np.nan

elif area=='DE':
    input_actual.gen_reservoir_hydro[(input_actual.gen_reservoir_hydro>590)] = np.nan

elif area=='CH':
    input_actual.gen_solar[(input_actual.gen_solar>390)] = np.nan
    input_actual.load[(input_actual.load>12000)] = np.nan
    
elif area=='SE':
    input_actual.load[(input_actual.load<7000)] = np.nan
    input_forecast.scheduled_gen_total[(input_forecast.scheduled_gen_total<4500)] =np.nan
    input_forecast.scheduled_gen_total[(input_forecast.scheduled_gen_total>30000)] =np.nan
    input_forecast.wind_on_day_ahead[(input_forecast.wind_on_day_ahead>8000)] =np.nan  

# %% Additional derived inputs

# Time
input_forecast['month'] = input_forecast.index.month
input_forecast['weekday'] =  input_forecast.index.weekday
input_forecast['hour'] =  input_forecast.index.hour

# Total generation
input_actual['total_gen'] = input_actual.filter(regex='^gen').sum(axis='columns')


# Inertia proxy - Sum of all synchronous generation
input_actual['synchronous_gen'] = input_actual.total_gen- input_actual.loc[:,['gen_solar',
                                                                              'gen_wind_off',
                                                                              'gen_wind_on']].sum(axis=1)

# Ramps of load and total generation
input_forecast['load_ramp_day_ahead'] = input_forecast.load_day_ahead.diff()
input_actual['load_ramp'] = input_actual.load.diff()
input_forecast['total_gen_ramp_day_ahead'] = input_forecast.scheduled_gen_total.diff()
input_actual['total_gen_ramp'] = input_actual.total_gen.diff()

# Ramps of generaton types
input_forecast['wind_off_ramp_day_ahead'] = input_forecast.wind_off_day_ahead.diff()
input_forecast['wind_on_ramp_day_ahead'] = input_forecast.wind_on_day_ahead.diff()
input_forecast['solar_ramp_day_ahead'] = input_forecast.solar_day_ahead.diff()
gen_ramp_cols = input_actual.filter(regex='^gen').columns.str[4:] + '_ramp'
input_actual[gen_ramp_cols] = input_actual.filter(regex='^gen').diff()

# Price Ramps
input_forecast['price_ramp_day_ahead'] = input_forecast.prices_day_ahead.diff()

# Forecast errors
input_actual['forecast_error_wind_on'] = input_forecast.wind_on_day_ahead - input_actual.gen_wind_on
input_actual['forecast_error_wind_off'] = input_forecast.wind_off_day_ahead - input_actual.gen_wind_off
input_actual['forecast_error_solar'] = input_forecast.solar_day_ahead - input_actual.gen_solar
input_actual['forecast_error_total_gen'] = input_forecast.scheduled_gen_total - input_actual.total_gen
input_actual['forecast_error_load'] = input_forecast.load_day_ahead - input_actual.load
input_actual['forecast_error_load_ramp'] = input_forecast.load_ramp_day_ahead - input_actual.load_ramp
input_actual['forecast_error_total_gen_ramp'] = input_forecast.total_gen_ramp_day_ahead - input_actual.total_gen_ramp
input_actual['forecast_error_wind_off_ramp'] = input_forecast.wind_off_ramp_day_ahead - input_actual.wind_off_ramp
input_actual['forecast_error_wind_on_ramp'] = input_forecast.wind_on_ramp_day_ahead - input_actual.wind_on_ramp
input_actual['forecast_error_solar_ramp'] = input_forecast.solar_ramp_day_ahead - input_actual.solar_ramp



#%% Save data

input_actual.to_hdf(folder+'input_actual.h5',key='df')
input_forecast.to_hdf(folder+'input_forecast.h5',key='df')




#%%
