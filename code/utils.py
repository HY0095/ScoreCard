import pandas as pd
import numpy as np
from math import log
import random


def entropy(data_classes, base=2):
    '''
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    '''
    if not isinstance(data_classes, pd.core.series.Series):
        raise AttributeError('input array should be a pandas series')
    classes = data_classes.unique()
    N = len(data_classes)
    ent = 0  # initialize entropy

    # iterate over classes
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        proportion = len(partition) / N
        #update entropy
        ent -= proportion * log(proportion, base)

    return ent

def cut_point_information_gain(dataset, cut_point, feature, target):
    '''
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature: column label of the numeric attribute values in data
    :param target: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    '''
    if not isinstance(dataset, pd.core.frame.DataFrame):
        raise AttributeError('input dataset should be a pandas data frame')

    entropy_full = entropy(dataset[target])  # compute entropy of full dataset (w/o split)

    #split data at cut_point
    data_left = dataset[dataset.cut_point <= cut_point]
    data_right = dataset[dataset.cut_point > cut_point]
    (N, N_left, N_right) = (len(dataset), len(data_left), len(data_right))

    gain = entropy_full - (N_left / N) * entropy(data_left[target]) - \
        (N_right / N) * entropy(data_right[target])

    return gain


def get_best_cut_point(dataset, feature, target):
    '''
        Selects the best cut point for a feature in a data partition based on information gain
        :param dataset: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
    '''
    cut_points = list(set(dataset[feature].values))
    range_min, range_max = min(cut_points), max(cut_points)
    
    candidates = [x for x in cut_points if (x > range_min) and (x < range_max)]

    if not candidates:
        return None
    gains = [(cut, cut_point_information_gain(dataset=dataset, cut_point=cut, feature=feature, target=target)) for cut in candidates]
    gains = sorted(gains, key=lambda x: x[1], reverse=True)

    return gains[0][0] #return cut point


def step_by_step_split(dataset, feature, target, min_leaf_size):

    split_point = get_best_cut_point(dataset=dataset, feature=feature, target=target)
    left_split = dataset[dataset[feature] <= split_point]
    right_split = dataset[dataset[feature] > split_point]
    if (left_split.shape[0] < min_leaf_size) or (right_split.shape[0] < min_leaf_size):
        split_sucess = False
    else:
        split_sucess = True
    return split_sucess, split_point, left_split, right_split



def cal_woe_iv(dataset):

    dataset['Total'] = dataset.Good + dataset.Bad
    dataset['Bad_Rate'] = dataset.Bad/ dataset.Total
    dataset['%Good'] = dataset.Good/ dataset.Good.sum()
    dataset['%Bad'] = dataset.Bad/ dataset.Bad.sum()
    dataset['%Good(cum)'] = dataset.Good.cumsum()/ dataset.Good.sum()
    dataset['%Bad(cum)'] = dataset.Good.cumsum()/ dataset.Good.sum()
    dataset['WOE'] = np.log(dataset['%Good'] / dataset['%Bad'])
    dataset['IV'] = dataset['WOE'] *(dataset['%Good'] - dataset['%Bad'])
    dataset['Odds'] = dataset.Good / dataset.Bad
    return dataset


# def chimerge(dataset, xname, yname, max_binnum, min_leaf_size):

    
    
    # while len