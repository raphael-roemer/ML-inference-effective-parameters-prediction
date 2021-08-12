#%%
import pandas as pd
import numpy as np
from sklearn import model_selection
import os


#%% Setup parameters and load data

area = 'CE'
folder = './prepared_data/{}/'.format(area)

version_folder = folder + pd.Timestamp("today").strftime("%Y-%m-%d") + "_3" + '/'
if not os.path.exists(version_folder):
   os.makedirs(version_folder)


#targets = ['f_integral', 'f_rocof', 'f_ext']
targets = ['g1','g2','q','r','D','mu_w_0','mu_a_0','var_ww_0','var_aa_0','rho_aw_0','RoCof','nadir','MeanDevInFirstHalf','Loglike','vari','nad','RoCoFLong']


#pd.read_hdf(folder+'input_actual.h5')

X_actual = pd.read_hdf(folder+'input_actual.h5', key='df')
X = pd.read_hdf(folder + 'input_forecast.h5')
y = pd.read_hdf(folder + 'outputs18_ADDscaleALL.h5').loc[:,targets]

X=X.reindex(y.index)
X_actual=X_actual.reindex(y.index)


# %% Prepare train and test data for prediction of targets

# Drop nan values
valid_ind = ~pd.concat([X, X_actual, y], axis=1).isnull().any(axis=1)
X, X_actual, y = X[valid_ind], X_actual[valid_ind], y[valid_ind]
X_actual = X_actual.join(X)

# Extract one continuous time series for prediction inspection and remove it from test/train data
# muss für Inputs und outputs möglichst vollständig sein
X_test_cont = X.loc['2018-02-04':'2018-02-10']
X_test_act_cont = X_actual.loc['2018-02-04':'2018-02-10']
y_test_cont = y.loc['2018-02-04':'2018-02-10']
y_pred_cont = pd.DataFrame(index=y_test_cont.index)
ind = y.loc['2018-02-04':'2018-02-10'].index
X, X_actual, y = X.drop(index=ind), X_actual.drop(index=ind), y.drop(index=ind)

# Train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X_train_act = X_actual.loc[X_train.index]
X_test_act = X_actual.loc[X_test.index]
y_pred=pd.DataFrame(index=y_test.index)

# Save data
X_train.to_hdf(version_folder+'X_train.h5',key='df')
X_train_act.to_hdf(version_folder+'X_train_act.h5',key='df')
y_train.to_hdf(version_folder+'y_train.h5',key='df')
y_test.to_hdf(version_folder+'y_test.h5',key='df')
y_pred.to_hdf(version_folder+'y_pred.h5',key='df')
X_test.to_hdf(version_folder+'X_test.h5',key='df')
X_test_act.to_hdf(version_folder+'X_test_act.h5',key='df')
X_test_cont.to_hdf(version_folder+'X_test_cont.h5',key='df')
X_test_act_cont.to_hdf(version_folder+'X_test_act_cont.h5',key='df')
y_test_cont.to_hdf(version_folder+'y_test_cont.h5',key='df')
y_pred_cont.to_hdf(version_folder+'y_pred_cont.h5',key='df')
