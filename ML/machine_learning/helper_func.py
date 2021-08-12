#%%
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoLocator, ScalarFormatter
from scipy.optimize import curve_fit 


#%%

def binned_scatter(x,y, x_name='x_name', corr_coefs=None, ax=None, boxplot=False, ylim_from_quantiles=True):
    if x_name=='gen_oil':
        cuts = pd.qcut(x, q=15, duplicates='drop')
    elif x_name in ['month', 'weekday', 'hour']:
        cuts = x
    else:
        cuts = pd.qcut(x, q=15, duplicates='drop')
    
    
    xmeans=x.groupby(cuts).mean()
    ymeans=y.groupby(cuts).mean()

    
    if ax==None:
        fig,ax=plt.subplots(1,1)
    
    
    ax.scatter(x.sample(frac=0.2, random_state=42),
               y.sample(frac=0.2, random_state=42), s=0.05, alpha=0.1)   
    
    c='orange'
    if boxplot:
        
        for i, (name,group) in enumerate(y.to_frame().groupby(cuts)):
            ax.boxplot(group.T, positions=xmeans.iloc[[i]], showfliers=False,whis=[10,90],
                       showmeans=False, patch_artist=True, 
                       widths=0.2*np.abs(xmeans.max()-xmeans.min())/ cuts.cat.categories.shape[0])

        xwidth=0.15
        ywidth=0.1
        
    else:
        
        ymeans_err=y.groupby(cuts).sem()
        ax.errorbar(xmeans, ymeans, yerr=ymeans_err, fmt='.', c='r', ms=5)
        #ax.plot(xmeans, y.groupby(cuts).median(), '.', c='r', ms=5)
        xwidth=0.1
        ywidth=0.3
    

    if ylim_from_quantiles:
        ywidth=0.
        yupper = y.quantile(0.9)
        ylower = y.quantile(0.1)
        ax.set_ylim([ylower - ywidth*np.abs(yupper-ylower), yupper + ywidth*np.abs(yupper-ylower)])
    else:
        ax.set_ylim([ymeans.min()-ywidth*np.abs(ymeans.min()-ymeans.max()),
                     ymeans.max()+ywidth*np.abs(ymeans.min()-ymeans.max())])
        
    ax.set_xlim([xmeans.min()-xwidth*np.abs(xmeans.min()-xmeans.max()),
                    xmeans.max()+xwidth*np.abs(xmeans.min()-xmeans.max())])

    

    ax.xaxis.set_major_locator(AutoLocator()) 
    ax.set_xticklabels(ax.get_xticks())
    ax.xaxis.set_major_formatter(ScalarFormatter())

    
    ax.set_xlabel(x_name)

    if corr_coefs is not None:
        ax.set_title(r'$\rho=${:.3}'.format(corr_coefs[x_name]), fontsize=9)
    return cuts

# %%


def calc_rocof(data,  smooth_window_size, lookup_window_size, method='increment_smoothing'):
  
    if data.index[0].minute!=0 or data.index[0].second!=0:
        print('Data is not starting with full hour!')
        return None
    
    full_hours = data.index[::3600]
    full_hours = full_hours[1:-1]
    
    result = pd.Series(index = full_hours)
    
    if method=='frequency_smoothing':

        for i in np.arange(len(full_hours)):

        
            smoothed_snipped = data.iloc[i*3600:(i+2)*3600].rolling(smooth_window_size, center=True).mean()

            df_dt = smoothed_snipped.diff(periods=5).iloc[3600-lookup_window_size:3600+lookup_window_size]
            
            
            if df_dt.isnull().any():
                result.iloc[i]=np.nan
            else:
                result.iloc[i] = df_dt[df_dt.abs().idxmax()] / 5.   

    if method=='increment_smoothing':

        for i in np.arange(len(full_hours)):
            
            df_dt = data.iloc[i*3600:(i+2)*3600].diff().rolling(smooth_window_size , center=True).mean()
            df_dt = df_dt.iloc[3600-lookup_window_size:3600+lookup_window_size]
                    
            if df_dt.isnull().any():
                result.iloc[i]=np.nan
            else:
                result.iloc[i] = df_dt[df_dt.abs().idxmax()]  

    return result


#%%


