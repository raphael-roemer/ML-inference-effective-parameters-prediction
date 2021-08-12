# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
import os
import shap
from itertools import combinations_with_replacement

#%% Load test and prediction time series

version = '2021-07-14_3'
area = 'CE'
target =  'MeanDevInFirstHalf'
#target = ['g1','g2','q','r','D','mu_w_0','mu_a_0','RoCof','nadir','MeanDevInFirstHalf','Loglike']


data_folder = './prepared_data/{}/'.format(area) + version+'/'
fit_folder = './Results/model_fit/{}/version-'.format(area) + version + '/target_{}/'.format(target)
res_folder = './Results/model_analysis/{}/version-'.format(area) + version + '/target_{}/'.format(target)

if not os.path.exists(res_folder):
    os.makedirs(res_folder)


y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:,target]
y_pred = pd.read_hdf(fit_folder+'y_pred.h5')
X_test = pd.read_hdf(data_folder+'X_test.h5')
X_test_act = pd.read_hdf(data_folder+'X_test_act.h5')
y_pred_cont = pd.read_hdf(fit_folder+'y_pred_cont.h5')
y_test_cont = pd.read_hdf(data_folder+'y_test_cont.h5').loc[:,target]

#%% Load Shap values

X_test_display = X_test.copy()
X_test_display_act = X_test_act.copy()

for col in ['month', 'weekday', 'hour']:
    X_test_display.loc[:,col] = X_test.loc[:,col].apply(str)
    X_test_display_act.loc[:,col] = X_test_act.loc[:,col].apply(str)


shap_vals = np.load(fit_folder + 'shap_values_gtb.npy')
shap_vals_act  = np.load(fit_folder + 'shap_values_gtb_act.npy')

# shap_inter_vals = np.load(fit_folder + 'shap_interaction_values_gtb.npy')
# shap_inter_vals_act = np.load(fit_folder + 'shap_interaction_values_gtb_act.npy')


#%% Shap values analysis for forecast data and actual data

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
shap.summary_plot(shap_vals, X_test, plot_type='bar', show=False, plot_size=None)
plt.subplot(1,2,2)
shap.summary_plot(shap_vals, X_test, show=False, plot_size=None)
plt.tight_layout()
plt.savefig(res_folder + 'shap_summary_gtb.png', bbox_inches='tight')
#plt.show()
plt.clf()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
shap.summary_plot(shap_vals_act, X_test_act, plot_type='bar', show=False, plot_size=None)
plt.subplot(1,2,2)
shap.summary_plot(shap_vals_act, X_test_act, show=False, plot_size=None)
#plt.xlim([np.quantile(shap_vals_act.flatten(),0.00001),np.quantile(shap_vals_act.flatten(),0.99999)])
plt.tight_layout()
plt.savefig(res_folder + 'shap_summary_gtb_act.png', bbox_inches='tight')
#plt.show()
plt.clf()

n_plots = 30
for name in X_test.columns[np.argsort(np.abs(shap_vals).mean(axis=0))][-n_plots:]:
    x_jitter = 0
    if name in ['month', 'hour', 'weekday']:
        x_jitter=0.5
    xlower = X_test.loc[:,name].quantile(0.001)
    xupper =  X_test.loc[:,name].quantile(0.999)
    ylower = shap_vals[:,X_test.columns==name].min()
    yupper =  shap_vals[:,X_test.columns==name].max()
    shap.dependence_plot(name, shap_vals, X_test, display_features=X_test_display,
                         show=False, x_jitter=x_jitter)
    plt.xlim([xlower-np.abs(xupper-xlower)*0.1,xupper+np.abs(xupper-xlower)*0.1])
    plt.ylim([ylower-np.abs(yupper-ylower)*0.1,yupper+np.abs(yupper-ylower)*0.1])
    plt.savefig(res_folder + 'shap_dependence_plot_{}_gtb.png'.format(name), bbox_inches='tight')
    #plt.show()
    plt.clf()

n_plots=30
for name in X_test_act.columns[np.argsort(np.abs(shap_vals_act).mean(axis=0))][-n_plots:]:
    x_jitter = 0
    if name in ['month', 'hour', 'weekday']:
        x_jitter=0.5
    xlower = X_test_act.loc[:,name].quantile(0.005)
    xupper =  X_test_act.loc[:,name].quantile(0.995)
    ylower = shap_vals_act[:,X_test_act.columns==name].min()
    yupper =  shap_vals_act[:,X_test_act.columns==name].max()
    shap.dependence_plot(name, shap_vals_act, X_test_act, display_features=X_test_display_act,
                         show=False, x_jitter=x_jitter)
    plt.xlim([xlower-np.abs(xupper-xlower)*0.1,xupper+np.abs(xupper-xlower)*0.1])
    plt.ylim([ylower-np.abs(yupper-ylower)*0.1,yupper+np.abs(yupper-ylower)*0.1])
    plt.savefig(res_folder + 'shap_dependence_plot_{}_gtb_act.png'.format(name), bbox_inches='tight')
    #plt.show()
    plt.clf()

#%% Shap interaction values plots

# shap.summary_plot(shap_inter_vals,X_test, show=False)
# plt.savefig(res_folder + 'shap_summary_interactions_gtb.svg')
# plt.show()
# plt.clf

# tmp = np.abs(shap_inter_vals).sum(0)
# for i in range(tmp.shape[0]):
#     tmp[i,i] = 0
# inds = np.argsort(-tmp.sum(0))[:10]
# tmp2 = tmp[inds,:][:,inds]
# plt.figure(figsize=(12,12))
# plt.imshow(tmp2)
# plt.yticks(range(tmp2.shape[0]), X_test.columns[inds], rotation=50.4, horizontalalignment="right")
# plt.xticks(range(tmp2.shape[0]), X_test.columns[inds], rotation=50.4, horizontalalignment="left")
# plt.gca().xaxis.tick_top()
# plt.colorbar()
# plt.show()
# plt.savefig(res_folder+ 'shap_interaction_strengths_gtb.svg', bbox_inches='tight')
# plt.clf()

# for i,j in combinations(inds[:3],2):
#     name1 = X_test.columns[i]
#     name2 = X_test.columns[j]
#     x_jitter = 0
#     if name1 in ['month', 'hour', 'weekday']:
#         x_jitter=0.5
#     xlower = X_test.loc[:,name1].quantile(0.001)
#     xupper =  X_test.loc[:,name1].quantile(0.999)
#     ylower = np.quantile(shap_inter_vals[:,X_test.columns==name1,X_test.columns==name2],0.001)
#     yupper = np.quantile(shap_inter_vals[:,X_test.columns==name1,X_test.columns==name2],0.999)
#     shap.dependence_plot([name1,name2], shap_inter_vals, X_test, display_features=X_test_display,
#                          show=False, x_jitter=x_jitter)
#     plt.xlim([xlower-np.abs(xupper-xlower)*0.1,xupper+np.abs(xupper-xlower)*0.1])
#     plt.ylim([ylower-np.abs(yupper-ylower)*0.6,yupper+np.abs(yupper-ylower)*0.6])
#     plt.savefig(res_folder + 'shap_dependence_plot_{}_{}_gtb.svg'.format(name1,name2))
#     plt.show()
#     plt.clf()

# shap.summary_plot(shap_inter_vals_act,X_test_act,show=False)
# plt.savefig(res_folder + 'shap_summary_interactions_gtb_act.svg', bbox_inches='tight')
# plt.show()
# plt.clf()

# tmp = np.abs(shap_inter_vals_act).sum(0)
# for i in range(tmp.shape[0]):
#     tmp[i,i] = 0
# inds = np.argsort(-tmp.sum(0))[:10]
# tmp2 = tmp[inds,:][:,inds]
# plt.figure(figsize=(12,12))
# plt.imshow(tmp2)
# plt.yticks(range(tmp2.shape[0]), X_test_act.columns[inds], rotation=50.4, horizontalalignment="right")
# plt.xticks(range(tmp2.shape[0]), X_test_act.columns[inds], rotation=50.4, horizontalalignment="left")
# plt.gca().xaxis.tick_top()
# plt.colorbar()
# plt.savefig(res_folder+ 'shap_interaction_strengths_gtb_act.svg', bbox_inches='tight')
# plt.show()
# plt.clf()

# for i,j in combinations_with_replacement(inds[:10],2):
#     name2 = X_test_act.columns[i]
#     name1 = X_test_act.columns[j]
#     x_jitter = 0
#     if name1 in ['month', 'hour', 'weekday']:
#         x_jitter=0.5
#     xlower = X_test_act.loc[:,name1].quantile(0.005)
#     xupper =  X_test_act.loc[:,name1].quantile(0.995)
#     ylower = np.quantile(shap_inter_vals_act[:,X_test_act.columns==name1,X_test_act.columns==name2],0.001)
#     yupper = np.quantile(shap_inter_vals_act[:,X_test_act.columns==name1,X_test_act.columns==name2],0.999)
#     shap.dependence_plot([name1,name2], shap_inter_vals_act, X_test_act,
#                          display_features=X_test_display_act,
#                          show=False, x_jitter=x_jitter)
#     plt.xlim([xlower-np.abs(xupper-xlower)*0.1,xupper+np.abs(xupper-xlower)*0.1])
#     plt.ylim([ylower-np.abs(yupper-ylower)*0.6,yupper+np.abs(yupper-ylower)*0.6])
#     plt.savefig(res_folder + 'shap_dependence_plot_{}_{}_gtb_act.svg'.format(name1,name2))
#     plt.show()
#     plt.clf()


#%% Plot GTB hyper-parameter optimization results

cvres = pd.read_csv(fit_folder+'cv_results_gtb.csv')
cvres_act = pd.read_csv(fit_folder+'cv_results_gtb_act.csv')

fig, ax = plt.subplots(1, cvres_act.filter(regex='^param_').shape[1], figsize=(15, 3))
for i, key in enumerate(cvres_act.filter(regex='^param_').columns):
    cvres_data = cvres.groupby(key).apply(lambda x: x.loc[x.mean_test_score.idxmax()])
    cvres_act_data = cvres_act.groupby(key).apply(lambda x: x.loc[x.mean_test_score.idxmax()])
    #l1=ax.plot(cvres_data.index, cvres_data.mean_test_score/cvres_data.mean_test_score.max(), '.-',label='day-ahead data')
    #l2=ax.plot(cvres_act_data.index, cvres_act_data.mean_test_score/cvres_act_data.mean_test_score.max(),'.-',label='actual data')
    #ax.set_xlabel(key[6:], fontsize=10)
    l1=ax[i].plot(cvres_data.index, cvres_data.mean_test_score/cvres_data.mean_test_score.max(), '.-',label='day-ahead data')
    l2=ax[i].plot(cvres_act_data.index, cvres_act_data.mean_test_score/cvres_act_data.mean_test_score.max(),'.-',label='actual data')
    ax[i].set_xlabel(key[6:], fontsize=10)

#ax.set_ylabel('Max. relative score')
ax[0].set_ylabel('Max. relative score')
#ax[0].set_ylim((0.95,ax[1].get_ylim()[1]))
#ax.legend()
ax[0].legend()
plt.tight_layout()

plt.savefig(res_folder+'cv_results_gtb.png', bbox_inches='tight')



#%% Plot Learning curves summary

# plt.figure(figsize=(7,3.5))

# # plot models with forecast data
# plt.subplot(1,2,1)
# res = 0
# for i,model in enumerate(y_pred.filter(regex='.*(?<!_act)$').columns):
#     try:
#         res = pd.read_csv(fit_folder+'learning_curve_{}.csv'.format(model))
#     except:
#         continue
#     p1=plt.plot(res.train_sizes, res.mean_train_scores, '--', label=model)
#     p2=plt.errorbar(res.train_sizes, res.mean_valid_scores, res.std_valid_scores, c=p1[0].get_color())
# p1,=plt.plot(res.train_sizes, res.mean_train_scores, '--',  c='k', alpha=0)
# p2,=plt.plot(res.train_sizes, res.mean_valid_scores, c='k', alpha=0)
# #plt.yscale('log')
# l1=plt.legend(fontsize=10, ncol=2,bbox_to_anchor=(1.5, -0.7, 0.3, 0.5))
# l2=plt.gca().legend([p1,p2], ['Train', 'Test'], fontsize=10)
# l2.get_lines()[0].set_alpha(1)
# l2.get_lines()[1].set_alpha(1)
# plt.gca().add_artist(l1)
# plt.xlabel('Number of training samples')
# plt.ylabel('Score')
# plt.title('Day-ahead data')

# # plots models with actual data
# plt.subplot(1,2,2)
# for i,model in enumerate(y_pred.filter(regex='.*_act').columns):
#     try:
#         res = pd.read_csv(fit_folder+'learning_curve_{}.csv'.format(model))
#     except:
#         continue
#     p1=plt.plot(res.train_sizes, res.mean_train_scores, '--', label=model)
#     p2=plt.errorbar(res.train_sizes, res.mean_valid_scores, res.std_valid_scores, c=p1[0].get_color())
# plt.plot(res.train_sizes, res.mean_train_scores, '--',  c='k', alpha=0)
# plt.plot(res.train_sizes, res.mean_valid_scores, c='k', alpha=0)
# plt.xlabel('Number of training samples')
# plt.ylabel('Score')
# plt.title('Actual data')

# plt.tight_layout()
# plt.savefig(res_folder+'learning_curves_summary.svg', bbox_inches='tight')


#%% Hold-out continuous test series: Prediction comparison

#fig, axs = plt.subplots(y_pred_cont.drop(columns=['lassocv', 'lassocv_act'], errors='ignore').shape[1]//2-1, 2,figsize=(6,7), sharex=True, sharey=True)

#days = 3
#for i, method in enumerate(y_pred_cont.drop(columns=['lassocv_act'], errors='ignore').filter(regex='.*_act')):
#    axs[i,0].plot(y_test_cont.iloc[:48*days], alpha=0.3, label='test series')
#    axs[i,0].plot(y_pred_cont[method[:-4]].iloc[:48*days], alpha=0.8, label='prediction')
#    axs[i,0].set_title(method[:-4])
#    axs[i,1].plot(y_test_cont.iloc[:48*days], alpha=0.3)
#    axs[i,1].plot(y_pred_cont[method].iloc[:48*days], alpha=0.8)
#    axs[i,1].set_title(method)
#    axs[i,0].grid(axis='x')
#    axs[i,1].grid(axis='x')

#axs[0,0].legend(ncol=2, fontsize=7)
#axs[-1,0].tick_params(axis='x', rotation=70)
#axs[-1,1].tick_params(axis='x', rotation=70)

#fig.text(0.0, 0.5, '{}'.format(target), va='center', rotation='vertical')
#fig.text(0.5, -0.04, 'Date', ha='center')

#plt.tight_layout()

#plt.savefig(res_folder+'hold_out_series_prediction_comparison.svg', bbox_inches='tight')


# %% Model relative performance summary

plt.figure(figsize=(10,4))
errors = y_pred.add(-y_test, axis=0)
errors = errors.div(y_test, axis=0)
errors.boxplot(whis=[10,90], showfliers=False)
plt.ylabel(r'Prediction error $(y-\hat y) \cdot y^{-1}$', fontsize=14)
plt.xticks(rotation=60, ha='right')
plt.savefig(res_folder+'rel_pred_error_summary.png', bbox_inches='tight')

# %% Model absolute performance summary

plt.figure(figsize=(10,4))
errors = y_pred.add(-y_test, axis=0)
errors.abs().boxplot(whis=[10,90], showfliers=False)
plt.ylabel(r'Prediction error $|y-\hat y|$ [Hz]', fontsize=14)
plt.xticks(rotation=60, ha='right')
plt.savefig(res_folder+'abs_pred_error_summary.png', bbox_inches='tight')


# %% Model R^2 summary for single target

plt.figure(figsize=(3,4))
cols= y_pred.columns # .drop(columns=['lassocv', 'lassocv_act']).columns
scores=[r2_score(y_test, y_pred.loc[:, col]) for col in cols]
cols = cols[np.argsort(scores)]

for i, col in enumerate(cols):
    plt.plot(i, r2_score(y_test, y_pred.loc[:, col]), 'o', c='r')
    #plt.plot(i, explained_variance_score(y_test, y_pred.loc[:, col]), 's', c='r')

plt.xticks(np.arange(len(cols)), cols, rotation=50, ha='right')
plt.ylabel(r'R^2', fontsize=14)
plt.savefig(res_folder+'R2_score_summary.png', bbox_inches='tight')

#%% Comparison of R^2 score for all target time series


#targets = ['q']#['f_ext', 'f_rocof', 'f_integral']
#targets = ['g1','g2','q','r','D','mu_w_0','mu_a_0','RoCof','nadir','MeanDevInFirstHalf','Loglike']
targets = ['vari','nadir','MeanDevInFirstHalf','nad','g1','g2','D','q','r']

summary_folder = './Results/model_analysis/{}/version-'.format(area) + version + '/'

models= ['daily_profile', 'gtb', 'gtb_act']

scores = pd.DataFrame(columns=models, index=targets)
for i,targ in enumerate(targets):

    #data_folder = './prepared_data/{}/version-'.format(area) + version+'/'
    data_folder = './prepared_data/{}/'.format(area) + version+'/'
    fit_folder = './Results/model_fit/{}/version-'.format(area) + version + '/target_{}/'.format(targ)
    y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:,targ]
    y_pred = pd.read_hdf(fit_folder+'y_pred.h5')
    scores.loc[targ] = [r2_score(y_test, y_pred.loc[:, col]) for col in models]

scores.plot.bar(ylim=[-0.02,scores.max().max()+0.05], width=0.85, figsize=(6,3),
                color=plt.get_cmap('tab20')(range(20)))
plt.legend(bbox_to_anchor=(1.01, 0.5, 0.3, 0.5))
plt.xticks(np.arange(len(targets)), targets, rotation=50, ha='right')
plt.ylabel(r'$R^2$-score', fontsize=14)
plt.grid(axis='y')

plt.savefig(summary_folder+'R2_score_summary_for_targets.png', bbox_inches='tight')





#%%
