import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

reload(sys)
sys.setdefaultencoding('utf-8')
import json

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'
question1, question2 = 'question1', 'question2'

data_folder = '../../../data/'
fp_train = data_folder + 'train.csv'


def fix_nans(df):
    def blja(s):
        if s != s:
            return ''
        return s

    for col in [question1, question2]:
        df[col] = df[col].apply(blja)

    return df


def load_train():
    return fix_nans(
        pd.read_csv(fp_train, index_col='id', encoding="utf-8")
    )



def gather_and_calculate_loss(name):
    fp_in = os.path.join('/home/ubik/Desktop', name)
    fp_out=os.path.join('../../stacking_data/', 'stacking_{}'.format(name), 'probs.csv')
    dfs=[]
    for l in os.listdir(fp_in):
        if os.path.isdir(os.path.join(fp_in, l)):
            fp = os.path.join(fp_in, l, 'probs.csv')
            df = pd.read_csv(fp, index_col='id')
            bl = df.copy()
            bl['z'] = 1-bl['prob']
            loss = log_loss(bl[TARGET], bl[['z', 'prob']])
            print loss
            dfs.append(df)

    df= pd.concat(dfs)
    df.to_csv(fp_out, index_label='id')
    return df

