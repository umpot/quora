from time import time

import pandas as pd
import numpy as np
import seaborn as sns
import re
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import json

from sklearn.model_selection import StratifiedKFold

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)


fp_train= '../../data/train.csv'
fp_test= '../../data/test.csv'

lemmas_train_fp='../../data/train_lemmas.csv'
tokens_train_fp='../../data/train_tokens.csv'
nlp_train_fp='../../data/postag_ner_train.json'

stems_train_fp='../../data/train_porter.csv'
stems_test_fp='../../data/test_porter.csv'

normalized_train_fp='../../data/train_normalized.csv'
common_words_train_fp = '../../data/train_common_words.csv'
length_train_fp = '../../data/train_length.csv'

METRICS_FP = [
    '../../data/train_metrics_bool_lemmas.csv',
    '../../data/train_metrics_bool_stems.csv',
    '../../data/train_metrics_bool_tokens.csv',
    '../../data/train_metrics_fuzzy_lemmas.csv',
    '../../data/train_metrics_fuzzy_stems.csv',
    '../../data/train_metrics_fuzzy_tokens.csv',
    '../../data/train_metrics_sequence_lemmas.csv',
    '../../data/train_metrics_sequence_stems.csv',
    '../../data/train_metrics_sequence_tokens.csv'
]


TARGET = 'is_duplicate'
qid1,  qid2 = 'qid1',  'qid2'

question1, question2 = 'question1', 'question2'
lemmas_q1, lemmas_q2 ='lemmas_q1', 'lemmas_q2'
stems_q1,stems_q2='stems_q1','stems_q2'
tokens_q1,tokens_q2='tokens_q1','tokens_q2'


def load_train():
    return pd.read_csv(fp_train, index_col='id')

def load__train_metrics():
    dfs = [pd.read_csv(fp, index_col='id') for fp in METRICS_FP]
    return pd.concat(dfs, axis=1)

def load_train_all():
    return pd.concat([
        load_train(),
        load_train_lemmas(),
        load_train_stems(),
        load_train_tokens(),
        load_train_lengths(),
        load_train_common_words(),
        load__train_metrics()
    ], axis=1)

def load_train_xgb():
    return pd.concat([
        load_train_lengths(),
        load_train_common_words(),
        load__train_metrics()
    ], axis=1)

def load_train_test():
    return pd.read_csv(fp_train, index_col='id'), pd.read_csv(fp_test, index_col='test_id')

def load_train_lemmas():
    df= pd.read_csv(lemmas_train_fp, index_col='id')
    df = df.fillna('')
    return df

def load_train_tokens():
    df= pd.read_csv(tokens_train_fp, index_col='id')
    df = df.fillna('')
    return df

def load_train_stems():
    df= pd.read_csv(stems_train_fp, index_col='id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter':'stems_q1', 'question2_porter':'stems_q2'})
    df = df.fillna('')
    return df

def load_train_common_words():
    df= pd.read_csv(common_words_train_fp, index_col='id')
    return df

def load_train_lengths():
    df= pd.read_csv(length_train_fp, index_col='id')
    return df

def load_train_normalized_train():
    return pd.read_csv(normalized_train_fp, index_col='id')



def plot_errors(imp):
    train_runs= [x['train'] for x in imp]
    test_runs= [x['test'] for x in imp]

    sz=len(train_runs[0])
    x_axis=range(sz)
    y_train = [np.mean([x[j] for x in train_runs]) for j in x_axis]
    y_test = [np.mean([x[j] for x in test_runs]) for j in x_axis]

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_train, label='train')
    ax.plot(x_axis, y_test, label='test')
    ax.legend()
    plt.show()

def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['logloss']
    results_on_train = estimator.evals_result()['validation_0']['logloss']
    return {
        'train': results_on_train,
        'test': results_on_test
    }

def perform_xgb_cv(df, target, n_estimators, max_depth):
    print '============================='
    print 'n_est {}'.format(n_estimators)
    print 'max_depth {}'.format(max_depth)
    print
    folds =5
    seed = int(time())
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    losses = []
    stats = []
    imp = []
    for big_ind, small_ind in skf.split(np.zeros(len(df)), target[TARGET]):
        big = df.iloc[big_ind]
        small = df.iloc[small_ind]
        train_arr, train_target = big, target.iloc[big_ind].values
        test_arr, test_target = small, target.iloc[small_ind].values

        estimator = xgb.XGBClassifier(n_estimators=n_estimators,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      max_depth=max_depth)
        eval_set = [(train_arr, train_target), (test_arr, test_target)]
        estimator.fit(train_arr, train_target, eval_set=eval_set, eval_metric='logloss', verbose=False)

        proba = estimator.predict_proba(test_arr)
        loss = log_loss(test_target, proba)
        print loss
        losses.append(loss)
        stats.append(xgboost_per_tree_results(estimator))
        imp.append(estimator.feature_importances_)
        # xgb.plot_importance(estimator)
        # plot_errors(stats)

    print 'avg {}'.format(np.mean(losses))

    return losses

def create_grid():
    nums = [200, 1000, 2000, 5000, 10000, 20000, 30000]
    depths = [3,4,5,8,10]
    grid = []
    for n in nums:
        for d in depths:
            grid.append([n,d])

    return grid

def perform_grid():
    grid = create_grid()
    df = load_train_xgb()
    target = load_train()[[TARGET]]
    res=[]
    for g in grid:
        n_estimators=g[0]
        max_depth=g[1]
        losses = perform_xgb_cv(df, target, n_estimators, max_depth)
        res.append({'n_estimators':n_estimators, 'max_depth':max_depth, 'losses':losses})
        with open('results_grid_xgb.json', 'w+') as f:
            json.dump(res, f)


perform_grid()