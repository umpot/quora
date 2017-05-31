import pandas as pd
import numpy as np
import seaborn as sns
from time import time

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)


FEATURES = ['len_q1', 'len_q2', 'diff_len',
            'len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2',
            'common_words', 'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
            'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
            'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'wmd', 'norm_wmd',
            'cosine_distance', 'cityblock_distance', 'jaccard_distance',
            'canberra_distance', 'euclidean_distance', 'minkowski_distance',
            'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec',
            'kur_q2vec']#, 'qid1', 'qid2'



fp_train= '../../data/train.csv'
fp_test= '../../data/test.csv'

ab_train= '../../data/train_features.csv'
ab_test= '../../data/test_features.csv'

qid1,  qid2 = 'qid1',  'qid2'
TARGET = 'is_duplicate'


def load_train():
    df = pd.read_csv(ab_train)
    train_df = pd.read_csv(fp_train)

    return pd.merge(df, train_df[[qid1, qid2, TARGET]], left_index=True, right_index=True)

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

def perform_xgb_cv():
    df = load_train()
    folds =5
    seed = int(time())
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    losses = []
    stats = []
    imp = []
    for big_ind, small_ind in skf.split(np.zeros(len(df)), df[TARGET]):
        big = df.iloc[big_ind]
        small = df.iloc[small_ind]
        train_arr, train_target = big[FEATURES], big[TARGET]
        test_arr, test_target = small[FEATURES], small[TARGET]

        estimator = xgb.XGBClassifier(n_estimators=20000,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      max_depth=6)
        eval_set = [(train_arr, train_target), (test_arr, test_target)]
        estimator.fit(train_arr, train_target, eval_set=eval_set, eval_metric='logloss', verbose=False)

        proba = estimator.predict_proba(test_arr)
        loss = log_loss(test_target, proba)
        print loss
        losses.append(loss)
        stats.append(xgboost_per_tree_results(estimator))
        imp.append(estimator.feature_importances_)
        plot_errors(stats)

    print np.mean(losses)

perform_xgb_cv()