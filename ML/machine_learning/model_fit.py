import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, learning_curve,ShuffleSplit, train_test_split
import os
import time
import shap
import xgboost as xgb

areas = ['CE']
data_version = '2021-07-14_3'
#targets = ['g1','g2','q','r','D','mu_w_0','mu_a_0','RoCof','nadir','MeanDevInFirstHalf','Loglike']
targets = ['MeanDevInFirstHalf']#['vari','nadir','nad','g1','g2','D','q','r']#,'RoCoFLong','mu_w_0','mu_a_0','RoCof','MeanDevInFirstHalf','Loglike']


start_time = time.time()
for area in areas:

    print('---------------------------- ', area, ' ------------------------------------')

    #data_folder = './prepared_data/{}/version-{}/'.format(area,data_version)
    data_folder = './prepared_data/{}/{}/'.format(area,data_version)

    for target in targets:

        print('-------- ', target, ' --------')

        res_folder = './Results/model_fit/{}/version-{}/target_{}/'.format(area,data_version, target)

        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        y_train = pd.read_hdf(data_folder+'y_train.h5').loc[:, target]
        y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:, target]

        if os.path.exists(res_folder+'y_pred.h5'):
            y_pred = pd.read_hdf(res_folder+'y_pred.h5')
            y_pred_cont = pd.read_hdf(res_folder+'y_pred_cont.h5')
        else:
            y_pred = pd.read_hdf(data_folder+'y_pred.h5') #contains only time index
            y_pred_cont = pd.read_hdf(data_folder+'y_pred_cont.h5') #contains only time index

        for actual in ['_act']:   #_act: full model, '': just day ahead

            # Load data
            X_train = pd.read_hdf(data_folder+'X_train{}.h5'.format(actual))
            X_test = pd.read_hdf(data_folder+'X_test{}.h5'.format(actual))
            X_test_cont = pd.read_hdf(data_folder+'X_test{}_cont.h5'.format(actual))


            # Daily profile prediction

            daily_profile = y_train.groupby(X_train.index.time).mean()
            y_pred['daily_profile'] = [daily_profile[time] for time in y_test.index.time]
            y_pred_cont['daily_profile'] = [daily_profile[time] for time in y_pred_cont.index.time]


            # Gradient boosting Regressor CV hyperparameter optimization

            X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train,
                                                                                      test_size=0.2)

            #params_grid = {
            #    'max_depth':[5],
            #    'learning_rate':[0.1],#[0.01,0.05,0.1, 0.2],
            #    'subsample':[1,0.7],#[1,0.7,0.4,0.1],
            #    #'reg_lambda':[ 0.1, 1, 10, 50]
            #}
            params_grid = {
                'max_depth':[2,3,5,7,9,11],
                'learning_rate':[0.01,0.05,0.1, 0.2],
                'subsample': [1,0.7,0.4,0.1] ,
                #'reg_lambda':[ 0.1, 1, 10],
                'min_child_weight':[1,5,10]
            }

            fit_params = {
                'eval_set':[(X_train_train, y_train_train),(X_train_val, y_train_val)],
                'early_stopping_rounds':20,
                'verbose':0
            }

            grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000,
                                                        verbosity=0),
                                    params_grid, verbose=1, n_jobs=-1, refit=False, cv=5)

            grid_search.fit(X_train_train, y_train_train, **fit_params)

            pd.DataFrame(grid_search.cv_results_).to_csv(res_folder+'cv_results_gtb{}.csv'.format(actual))
            pd.DataFrame(grid_search.best_params_,
                         index=['optimal']).to_csv(res_folder+'cv_best_params_gtb{}.csv'.format(actual))


            # Gradient boosting regression best model evaluation

            params=pd.read_csv(res_folder+'cv_best_params_gtb{}.csv'.format(actual),
                            index_col=[0]).to_dict('records')[0]


            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, **params)
            model.fit(X_train_train, y_train_train, **fit_params)


            shap_vals = shap.TreeExplainer(model).shap_values(X_test)
            np.save(res_folder + 'shap_values_gtb{}.npy'.format(actual), shap_vals)

            # shap_interact_vals = shap.TreeExplainer(model).shap_interaction_values(X_test)
            # np.save(res_folder + 'shap_interaction_values_gtb{}.npy'.format(actual), shap_interact_vals)

            y_pred['gtb{}'.format(actual)] = model.predict(X_test)
            y_pred_cont['gtb{}'.format(actual)] = model.predict(X_test_cont)



            # GTB learning curve for evaluating the fit

            # train_sizes, train_scores, valid_scores = learning_curve(model,
            #                                                         X_train, y_train,
            #                                                         cv=5, verbose=1, n_jobs=-1)
            # res = pd.DataFrame(index=np.arange(5), data={'train_sizes':train_sizes,
            #                                             'mean_train_scores':train_scores.mean(axis=-1),
            #                                             'mean_valid_scores': valid_scores.mean(axis=-1),
            #                                             'std_valid_scores': valid_scores.std(axis=-1),
            #                                             'std_train_scores': train_scores.std(axis=-1) })
            # res.to_csv(res_folder+'learning_curve_gtb{}.csv'.format(actual))

            # GTB prediction stages for evaluating the fit

            # res = pd.DataFrame(columns=['train_rmse','test_rmse'])
            # res.loc[:,'train_rmse'] = model.evals_result()['validation_0']['rmse']
            # res.loc[:,'test_rmse'] = model.evals_result()['validation_1']['rmse']
            # res.to_csv(res_folder+'staged_predict_gtb{}.csv'.format(actual))


        # Save prediction

        y_pred.to_hdf(res_folder+'y_pred.h5',key='df')
        y_pred_cont.to_hdf(res_folder+'y_pred_cont.h5',key='df')


print("Execution time: {}".format(time.time() - start_time))

# %%
