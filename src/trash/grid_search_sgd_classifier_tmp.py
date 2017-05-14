from time import time

import pandas as pd
import numpy as np
import seaborn as sns
import re
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
import json

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

def perform_sgd_cv(df, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 0.5, 0.9, 1],
        'en__l1_ratio': [0, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 1]
    }

    folder = StratifiedKFold(n_splits=5, shuffle=True)

    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(df[predictors],
                                  df[TARGET])

    return grid_search

def perform_grid():
    df = load_train_xgb()
    target = load_train()[[TARGET]]
    predictors = df.columns.values
    df[TARGET] = target
    perform_sgd_cv(df, predictors)


perform_grid()