import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import json
from time import sleep, time
import traceback

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

TARGET = 'label'
qid1, qid2 = 'qid1', 'qid2'


data_folder = '../../../data/'

topics_word2vec_fp=os.path.join(data_folder, 'topics', 'topics_word2vec.csv')

def shuffle_df(df, random_state=42):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]

def load_topics_w2v():
    df = pd.read_csv(topics_word2vec_fp)
    df = shuffle_df(df)
    blja = 'Unnamed: 0'
    if blja in df.columns:
        del df[blja]
    return df




def perform_xgb_cv():
    t =time()
    print "Loading data...."
    df = load_topics_w2v()
    print "Loaded!"
    print 'Time {}'.format(time()-t)
    folds =5
    seed = 42

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    losses = []
    n_est=[]
    for big_ind, small_ind in skf.split(np.zeros(len(df)), df[TARGET]):
        big = df.iloc[big_ind]
        small = df.iloc[small_ind]



        train_target = big[TARGET]
        del big[TARGET]
        train_arr = big

        test_target = small[TARGET]
        del small[TARGET]
        test_arr = small

        # estimator = xgb.XGBClassifier(n_estimators=10000,
        #                               subsample=0.6,
        #                               # colsample_bytree=0.8,
        #                               max_depth=7,
        #                               objective='binary:logistic',
        #                               learning_rate=0.02,
        #                               base_score=0.2)

        estimator = xgb.XGBClassifier(n_estimators=10000,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      max_depth=5,
                                      objective='binary:logistic',
                                      nthread=-1
                                      )
        print test_arr.columns.values
        print len(train_arr)
        print len(test_arr)
        eval_set = [(train_arr, train_target), (test_arr, test_target)]
        estimator.fit(
            train_arr, train_target,
            eval_set=eval_set,
            eval_metric='merror',
            verbose=True,
            early_stopping_rounds=150,
            verbose=True
        )


perform_xgb_cv()