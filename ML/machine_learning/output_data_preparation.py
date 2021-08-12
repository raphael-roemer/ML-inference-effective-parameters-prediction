#%%
import pandas as pd
import numpy as np
import os
import csv

#%% General outputs to characterize hourly frequency deviations


# General parameter Setup
tzs = {'CE':'CET', 'Nordic':'Europe/Helsinki', 'GB':'GB'}
#start = pd.Timestamp('2015-01-01 00:00:00', tz='UTC')
#end = pd.Timestamp('2018-12-31 23:00:00', tz='UTC')
# just for 2018:
start = pd.Timestamp('2018-01-01 00:00:00', tz='UTC')
end = pd.Timestamp('2018-12-31 23:00:00', tz='UTC')


for area in ['CE']:

    #print(area)

    # Load frequency data

    folder = './prepared_data/{}/'.format(area)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load MLE results into output data frame
        #first load mleresults in a dataframe.

    index2 = pd.date_range(start, end, freq='1H', tz='UTC').tz_convert(tzs[area])

    # from csv-file:
    outputs=pd.read_csv('./prepared_data/mle/mleres18_re9ADDDe4.csv').set_index(index2)
    outputs2=pd.read_csv('./prepared_data/mle/mleres18.csv').set_index(index2)

    # or alternatively from hdf file (not needed):
    # ...

    #give names to columns:
    #input_actual = raw_input_data.loc[:, raw_input_data.columns.intersection(actual_cols)]
    #outputs.rename(columns = {'x1':'turnOn','x2':'posOfdbcrossing','x3':'g1','x4':'g2','x5':'q','x6':'r','x7':'D','x8':'mu_w_0','x9':'mu_a_0','x10':'var_ww_0','x11':'var_aa_0','x12':'rho_aw_0','x13':'RoCof','x14':'nadir','x15':'MeanDevInFirstHalf','x16':'Loglike'}, inplace = True)
    outputs.rename(columns = {'Column1':'turnOn','Column2':'posOfdbcrossing','Column3':'g1','Column4':'g2','Column5':'q','Column6':'r','Column7':'D','Column8':'mu_w_0','Column9':'mu_a_0','Column10':'var_ww_0','Column11':'var_aa_0','Column12':'rho_aw_0','Column13':'RoCof','Column14':'nadir','Column15':'MeanDevInFirstHalf','Column16':'Loglike','Column17':'vari','Column18':'nad','Column19':'RoCoFLong'}, inplace = True)
    #print(outputs.columns)
    #print(outputs.loc[:,'q'])
    #'turnOn','posOfdbcrossing','g1','g2','q','r','D','mu_w_0','mu_a_0','var_ww_0','var_aa_0','rho_aw_0','RoCof','nadir','MeanDevInFirstHalf', 'Loglike'

    # Save DataFrame into hdf5
    outputs.to_hdf(folder+'outputs18_ADDscaleALL.h5', key='df')
