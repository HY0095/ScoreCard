import numpy as np
import xlsxwriter
import time
import codecs
import pandas as pd
import math as math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
# %matplotlib inline
from IPython.display import SVG, HTML

from utils import *
from chimerge import ChiMerge
from collections import Counter

_OptmizeGrouping_doc = """
    Parameters
    ----------
    data: DataFrame
        raw data
    xname: string
        The independent variable 
    yname: string
        The dependnet variable, dim = n*p
    method: ['quantile', 'bucket']
        The default value is 'quantile'
    prebin_num: int
        prebin_num = 20 (default)
    coasrebin_num: int
        finebin_num = 5 (default)
    mingroupsize: float
        mingroupsize = 0.05 (default)
    """

# Interactive Grouping
class OptmizeGrouping(object):
    __doc__="""
    The Optmize Grouping Process
    %(Params_doc)s
    Notes
    ----
    """%{"Params_doc":_OptmizeGrouping_doc}

    def __init__(self, data, xname, yname, **kwargs):
        self.data = data[[xname, yname]]
        self.xname = xname
        self.target = yname
        self.nrow = data.shape[0]
        self.max_x_val = data[xname].max()
        self.mingroupsize = 0.05

        if 'mingroupsize' in kwargs.keys():
            self.mingroupsize = kwargs['mingroupsize']
        
        self.min_leaf_size = int(self.nrow * self.mingroupsize)

    def prebin(self, **kwargs):
        self.prebin_num = 20
        self.method = 'quantile'
        if 'prebin_num' in kwargs.keys():
            prebin_num = kwargs['prebin_num']
        
        if 'method' in kwargs.keys():
            method = kwargs['prebin_method']
        nandata = self.data[self.data[self.xname].isnull()]
        nonandata = self.data[~self.data[self.xname].isnull()]
        if (len(set(nonandata[self.xname][:1000])) >= self.prebin_num):
            if (self.method == 'quantile'):
                bincut = stats.mstats.mquantiles(nonandata[self.xname], prob=np.arange(0, 1, 1./self.prebin_num))
            # print(bincut)
            elif (self.method == 'bucket'):
                minvalue = min(nonandata[self.xname])
                maxvalue = max(nonandata[self.xname])
                bincut = np.arange(minvalue, maxvalue, 1.*(maxvalue-minvalue)/self.prebin_num)
            else:
                print("Error Message: Wrong Prebin mehtod ! Use: 'quantile' or 'bucket' ...")
                raise SystemExit
            bincut = list(set(bincut))   # remove duplicate
            bincut.sort()
            for i in np.arange(len(bincut)+1):
                if (i == 0):
                    nonandata.loc[nonandata[self.xname] <= bincut[i], self.xname+'_bin']= 'bin_'+str(i)                  
                    nonandata.loc[nonandata[self.xname] <= bincut[i], 'cut_point']= bincut[i]
                elif (i>1 and i<len(bincut)):
                    nonandata.loc[(nonandata[self.xname] > bincut[i-1])&(nonandata[self.xname]<=bincut[i]), self.xname+'_bin'] = 'bin_'+str(i) 
                    nonandata.loc[(nonandata[self.xname] > bincut[i-1])&(nonandata[self.xname]<=bincut[i]), 'cut_point'] = bincut[i]
                elif (i == len(bincut)):
                    nonandata.loc[nonandata[self.xname] > bincut[i-1], self.xname+'_bin'] = 'bin_'+str(i)
                    nonandata.loc[nonandata[self.xname] > bincut[i-1], "cut_point"] = max(nonandata[self.xname])           
        else:
            bincut = list(set(nonandata[self.xname]))
            bincut.sort()
            for i, v in enumerate(bincut):
                nonandata.loc[nonandata[self.xname] == i, self.xname+'_bin'] = 'bin_0'+str(i+1)
                nonandata.loc[nonandata[self.xname] == bincut[i], 'cut_point'] = v
        if nandata.shape[0] == 0:
            pass
        else:
            nandata[self.xname+'_bin'] = 'Null'
            nandata['cut_point'] = 'Null'
            
        return [nandata, nonandata]


    def coarsebin_interval(self, **kwargs):
        self.coarsebin_num = 8
        if 'coarsebin_num' in kwargs.keys():
            coarsebin_num = kwargs['coarsebin_num']
        best_split_point = list()
        [nandata, nonandata] = self.prebin()

        if nandata.shape[0] > 0:
            nandata[self.xname+'_bins'] = nandata['cut_point']
            res_nandata = pd.crosstab(nandata[self.xname+'_bins'], nandata[self.target])
            try:
                res_nandata.columns = ['Good', "Bad"]
                res_nandata.reset_index(self.xname+'_bins', inplace=True)                
            except (ValueError):
                print("Warning: Please Check missing value ! All of them = 1 or = 0 ")
                res_nandata = pd.DataFrame()
            except:
                print("Warning: Please Check missing value, something is wrong !!!")
                res_nandata = pd.DataFrame()
        else:
            res_nandata = pd.DataFrame()

        if nonandata.shape[0] == 0:
            print("Warnning: No Nan Data is Empty, Porc Coarse Bin Finished !!!")
            return nandata

        l0_split_sucess, l0_split_point, l00_split, l01_split = step_by_step_split(dataset=nonandata, feature='cut_point', target=self.target, min_leaf_size=self.min_leaf_size, best_split_point=[])
        
        if l0_split_sucess: # Layer 1 
            best_split_point.append(l0_split_sucess)
            if len(l00_split['cut_point'].value_counts()) > 1:
                l00_split_sucess, l00_split_point, l000_split, l001_split = step_by_step_split(dataset=l00_split, feature='cut_point', target=self.target, min_leaf_size =self.min_leaf_size, best_split_point=best_split_point)
                
                if l00_split_sucess:
                    best_split_point.append(l00_split_point)
                    if len(l000_split['cut_point'].value_counts()) > 1:
                        l000_split_sucess, l000_split_point, l0000_split, l0001_split = step_by_step_split(dataset=l000_split, feature='cut_point', target=self.target, min_leaf_size =self.min_leaf_size, best_split_point=best_split_point)
                        
                        if l000_split_sucess:
                            best_split_point.append(l000_split_point)
                
                    if len(l001_split['cut_point'].value_counts()) > 1:
                        l001_split_sucess, l001_split_point, l0010_split, l0011_split = step_by_step_split(dataset=l001_split, feature='cut_point', target=self.target, min_leaf_size =self.min_leaf_size, best_split_point=best_split_point)
                        if l001_split_sucess:
                            best_split_point.append(l001_split_point)

            if len(l01_split['cut_point'].value_counts()) > 1:                
                l01_split_sucess, l01_split_point, l010_split, l011_split = step_by_step_split(dataset=l01_split, feature='cut_point', target=self.target, min_leaf_size =self.min_leaf_size, best_split_point=best_split_point)

                if l01_split_sucess: # Layer 2 left
                    best_split_point.append(l01_split_point)
                    if len(l010_split['cut_point'].value_counts()) > 1:
                        l010_split_sucess, l010_split_point, l0100_split, l0101_split = step_by_step_split(dataset=l010_split, feature='cut_point', target=self.target, min_leaf_size =self.min_leaf_size, best_split_point=best_split_point)

                        if l010_split_sucess:
                            best_split_point.append(l010_split_point)

                    if len(l011_split['cut_point'].value_counts()) > 1:
                        l011_split_sucess, l011_split_point, l0110_split, l0111_split = step_by_step_split(dataset=l011_split, feature='cut_point', target=self.target, min_leaf_size =self.min_leaf_size, best_split_point=best_split_point)
                        if l011_split_sucess:
                            best_split_point.append(l011_split_point)

            # cal_Woe_table
            best_split_point.append(self.max_x_val)
            best_split_point = list(set(best_split_point))
            best_split_point.sort()
            final_split_len = len(best_split_point)

            for i in range(final_split_len):
                if i == 0:
                    nonandata.loc[nonandata.cut_point <= best_split_point[i], self.xname+'_bins'] = str(i+1) + ". ( -inf, "+str(best_split_point[i]) + " ]"
                else:
                    nonandata.loc[(nonandata.cut_point > best_split_point[i-1]) & (nonandata.cut_point <= best_split_point[i]), self.xname+'_bins'] = str(i+1) + ". (  "+str(best_split_point[i-1]) + " , "+ str(best_split_point[i]) +" ]"

            res_nonandata = pd.crosstab(nonandata[self.xname+'_bins'], nonandata[self.target])
            res_nonandata.columns = ['Good', "Bad"]
            res_nonandata.reset_index(self.xname+'_bins', inplace=True)
            res_nonandata.replace('Null', np.NaN).sort_values(self.xname+'_bins', ascending=True) 

        else:

            nonandata[self.xname+'_bins'] = 'others'
            res_nonandata = pd.crosstab(res_nonandata[self.xname+'_bins'], res_nonandata[self.target])
            res_nonandata.columns = ['Good', "Bad"]
            res_nonandata.reset_index(self.xname+'_bins', inplace=True)
            print("Warnning: Split data size < %s  !!!" % str(self.mingroupsize))
    
        res_woe = cal_woe_iv(pd.concat([res_nandata, res_nonandata]))        

        return res_woe


    def coarsebin_norminal(self, **kwargs):

        self.coarsebin_num = 8
        if 'coarsebin_num' in kwargs.keys():
            self.coarsebin_num = kwargs['coarsebin_num']

        dist_x_val = list(set(self.data[self.xname]))


        return ChiMerge(dataset=self.data, 
                        xname=self.xname, 
                        target=self.target, 
                        binnum=self.coarsebin_num
                    )
