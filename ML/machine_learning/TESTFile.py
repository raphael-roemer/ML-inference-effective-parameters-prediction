import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, learning_curve,ShuffleSplit, train_test_split
import os
import time
import shap
import xgboost as xgb

areas = ['CE']
data_version = '2021-06-18'
#targets = ['f_ext', 'f_rocof', 'f_integral'] # from output data
targets = ['q']


start_time = time.time()

area='CE'
print('---------------------------- ', area, ' ------------------------------------')

data_folder = './prepared_data/{}/{}/'.format(area,data_version)
#print(data_folder)
for target in targets:


        print('-------- ', target, ' --------')

        res_folder = './Results/model_fit/{}/version-{}/target_{}/'.format(area,data_version, target)
        print(res_folder)
        if not os.path.exists(res_folder):
                os.makedirs(res_folder)

        y_train = pd.read_hdf(data_folder+'y_train.h5').loc[:, target]
        y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:, target]
        print(y_train)
        print(y_test)
