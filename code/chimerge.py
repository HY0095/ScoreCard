import pandas as pd
import numpy as np
import scipy
from scipy import stats
from utils import cal_woe_iv


def ChiMerge(dataset, xname, target, binnum=5, maxcut=50):
    '''
    binnum: the number of bins output
    maxcut: initial bins number 
    '''
    
    dataset=dataset[[xname,target]]
    for val in list(set(dataset[xname])):
        dataset.loc[dataset[xname] == val, 'CutPoint'] = len(dataset.loc[(dataset[xname] == val) & (dataset[target] == 1)])/len(dataset.loc[dataset[xname] == val])
    # equifrequent cut the var into maxcut bins
    
    dist_vals = len(set(dataset[xname]))
    
    if dist_vals > maxcut:
        dataset["CutPoint"], breaks=pd.qcut(dataset['CutPoint'], q=maxcut, duplicates="drop", retbins=True)

    crosstab_dt = pd.crosstab(dataset['CutPoint'], dataset['target'])
    crosstab_dt.columns = ['Good', "Bad"]
    crosstab_dt.fillna(0, inplace=True)
    crosstab_dt.sort_values('CutPoint', inplace=True)
    crosstab_dt.reset_index(inplace=True)
    
    #calculate chi-square  merge the minimum chisquare
    while len(crosstab_dt) > binnum:
        chi_squares=[]
        for i in range(len(crosstab_dt)-1):
            chi_square = stats.chi2_contingency([crosstab_dt.iloc[i][1:3], crosstab_dt.iloc[i+1][1:3]])[0]
            chi_squares.append(chi_square)       
        # merge the minimum chisquare backwards
        i = chi_squares.index(min(chi_squares))
        crosstab_dt.iloc[i]['Good'] = crosstab_dt.iloc[i]['Good'] + crosstab_dt.iloc[i+1]['Good']
        crosstab_dt.iloc[i]['Bad'] = crosstab_dt.iloc[i]['Bad'] + crosstab_dt.iloc[i+1]['Bad']
        crosstab_dt = crosstab_dt.drop(labels=[i])
        crosstab_dt = crosstab_dt.reset_index(drop=True)

    return cal_woe_iv(crosstab_dt)                          
    
