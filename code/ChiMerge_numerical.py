import pandas as pd
import numpy as np
import scipy
from scipy import stats
def chi_bin(DF,var,target,binnum=5,maxcut=20):
    '''
    DF:data
    var:variable
    target:target / label
    binnum: the number of bins output
    maxcut: initial bins number 
    '''
    
    data=DF[[var,target]]
    #equifrequent cut the var into maxcut bins
    data["cut"],breaks=pd.qcut(data[var],q=maxcut,duplicates="drop",retbins=True)
    #count 1,0 in each bin
    count_1=data.loc[data[target]==1].groupby("cut")[target].count()
    count_0=data.loc[data[target]==0].groupby("cut")[target].count()
    #get bins value: min,max,count 0,count 1
    bins_value=[*zip(breaks[:maxcut-1],breaks[1:],count_0,count_1)]
    #define woe
    def woe_value(bins_value):
        df_woe=pd.DataFrame(bins_value)
        df_woe.columns=["min","max","count_0","count_1"]
        df_woe["total"]=df_woe.count_1+df_woe.count_0
        df_woe["bad_rate"]=df_woe.count_1/df_woe.total
        df_woe["woe"]=np.log((df_woe.count_0/df_woe.count_0.sum())/(df_woe.count_1/df_woe.count_1.sum()))
        return df_woe
    #define iv
    def iv_value(df_woe):
        rate=(df_woe.count_0/df_woe.count_0.sum())-(df_woe.count_1/df_woe.count_1.sum())
        iv=np.sum(rate * df_woe.woe)
        return iv
    #make sure every bin contain 1 and 0
    ##first bin merge backwards
    for i in range(len(bins_value)):
        if 0 in bins_value[0][2:]:
            bins_value[0:2]=[(
                bins_value[0][0],
                bins_value[1][1],
                bins_value[0][2]+bins_value[1][2],
                bins_value[0][3]+bins_value[1][3])]
            continue
    ##bins merge forwards
        if 0 in bins_value[i][2:]:
            bins_value[i-1:i+1]=[(
                bins_value[i-1][0],
                bins_value[i][1],
                bins_value[i-1][2]+bins_value[i][2],
                bins_value[i-1][3]+bins_value[i][3])]
            break
        else:
            break
    
    #calculate chi-square  merge the minimum chisquare       
    while len(bins_value)>binnum:
        chi_squares=[]
        for i in range(len(bins_value)-1):
            a=bins_value[i][2:]
            b=bins_value[i+1][2:]
            chi_square=scipy.stats.chi2_contingency([a,b])[0]
            chi_squares.append(chi_square)
    #merge the minimum chisquare backwards
        i = chi_squares.index(min(chi_squares))
                              
        bins_value[i:i+2]=[(
            bins_value[i][0],
            bins_value[i+1][1],
            bins_value[i][2]+bins_value[i+1][2],
            bins_value[i][3]+bins_value[i+1][3])]
        
        df_woe=woe_value(bins_value)
        
    #print bin number and iv
        print("箱数：{},iv:{:.6f}".format(len(bins_value),iv_value(df_woe)))
    #return bins and woe information 
    return woe_value(bins_value)                          
    
