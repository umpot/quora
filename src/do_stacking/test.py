import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import json
from time import sleep
import traceback
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

stacking_data_fp='../stacking_data'
stacking_submit_data_fp='../stacking_submit_data'

######################################################################################
######################################################################################
######################################################################################
######################################################################################
data_folder = '../../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

folds_fp = os.path.join(data_folder, 'top_k_freq', 'folds.json')

def load_folds():
    folds = json.load(open(folds_fp))
    folds = [(int(k),(v['train'], v['test'])) for k,v in folds.iteritems()]
    folds = [x[1] for x in folds]

    return folds


def create_folds(df):
    folds = load_folds()

    return [
        (df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']])
        for x in range(len(folds))]

def shuffle_df(df, random_state=42):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################