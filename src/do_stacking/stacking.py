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
from xgboost import plot_importance
from matplotlib import pyplot

reload(sys)
sys.setdefaultencoding('utf-8')

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

stacking_data_fp = '../../stacking_data'
stacking_submit_data_fp = '../../stacking_submit_data'




######################################################################################
######################################################################################
######################################################################################
######################################################################################
data_folder = '../../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

folds_fp = os.path.join(data_folder, 'top_k_freq', 'folds.json')

TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'

question1, question2 = 'question1', 'question2'


def load_train():
    df= pd.read_csv(fp_train, index_col='id', encoding="utf-8")
    cols_to_del = [qid1, qid2, question1, question2]
    for col in cols_to_del:
        del df[col]

    return df

def load_test():
    df = pd.read_csv(fp_test, index_col='test_id')
    cols_to_del = [question1, question2]
    for col in cols_to_del:
        del df[col]

    return df


def load_folds():
    folds = json.load(open(folds_fp))
    folds = [(int(k), (v['train'], v['test'])) for k, v in folds.iteritems()]
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

import pandas as pd
import numpy as np

TARGET = 'is_duplicate'

INDEX_PREFIX= 100000000
#old
{'pos': 0.369197853026293,
 'neg': 0.630802146973707}


#new
r1 = 0.174264424749
r0 = 0.825754788586

""""
p_old/(1+delta) = p_new

delta = (p_old/p_new)-1 = 1.1186071314214785
l = delta*N = 452241
"""

delta = 1.1186071314214785

def explore_target_ratio(df):
    return {
        'pos':1.0*len(df[df[TARGET]==1])/len(df),
        'neg':1.0*len(df[df[TARGET]==0])/len(df)
    }

def oversample_df(df, l, random_state):
    df_pos = df[df[TARGET]==1]
    df_neg = df[df[TARGET]==0]

    df_neg_sampl = df_neg.sample(l, random_state=random_state, replace=True)

    df=pd.concat([df_pos, df_neg, df_neg_sampl])
    df = shuffle_df(df, random_state)

    return df

def oversample(train_df, test_df, random_state):
    l_train = int(delta * len(train_df))
    l_test = int(delta * len(test_df))

    return oversample_df(train_df, l_train, random_state), oversample_df(test_df, l_test, random_state)


def oversample_submit(train_df, test_df, random_state=42):
    l_train = int(delta * len(train_df))

    return oversample_df(train_df, l_train, random_state),test_df

############################################################3
############################################################3
############################################################3

experiments = [
    'stacking_all1_deep',
    'stacking_no_emb_simpl_idf_light',
    'stacking_no_metrics_light',
    'stacking_no_metrics_only_glove_emb',
    'stacking_no_tfidf_light',
    'stacking_no_top_tokens_light',
    'stacking_only_glove_emb_light',
    'stacking_only_lex_emb_light',
    'stacking_only_word2vec_emb_light',
    'stacking_lstm',
    'stacking_random_forest_light',
    'stacking_lstm_glove_lemmas_re_stops_yes',
    'stacking_lstm_lex_question_re_stops_yes',
    'stacking_lstm_word2vec_lemmas_re_stops_yes',
    'stacking_lstm_glove_nouns_re_stops_no',
    'stacking_lstm_glove_verbs_re_stops_no'

]

def perform_xgb_cv():
    seed = 42

    folds = load_folds()
    df = load_stacking(experiments)

    losses = []

    for big_ind, small_ind in folds:
        big = df.iloc[big_ind]
        small = df.iloc[small_ind]

        print explore_target_ratio(big)
        print explore_target_ratio(small)

        big, small = oversample(big, small, seed)

        print explore_target_ratio(big)
        print explore_target_ratio(small)


        train_target = big[TARGET]
        del big[TARGET]
        train_arr = big

        test_target = small[TARGET]
        del small[TARGET]
        test_arr = small

        estimator = xgb.XGBClassifier(n_estimators=10000,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      max_depth=5,
                                      objective='binary:logistic',
                                      )
        print test_arr.columns.values

        eval_set = [(train_arr, train_target), (test_arr, test_target)]
        estimator.fit(
            train_arr, train_target,
            eval_set=eval_set,
            eval_metric='logloss',
            verbose=True,
            early_stopping_rounds=100
        )

        # plot_importance(estimator)
        # pyplot.show()

        proba = estimator.predict_proba(test_arr)
        loss = log_loss(test_target, proba)
        print loss
        losses.append(loss)

    print 'AVG={}'.format(np.mean(losses))


def load_stacking(exp_list):
    train_df = load_train()
    dfs = [(e, pd.read_csv(os.path.join(stacking_data_fp, e, 'probs.csv'), index_col='id') ) for e in exp_list]
    for e, df in dfs:
        train_df.loc[df.index, e] = df.loc[df.index, 'prob']

    return train_df

def load_submit_stacking(exp_list):
    test_df = load_test()
    dfs = [(e, pd.read_csv(os.path.join(stacking_submit_data_fp, 'submit_'+e, 'probs.csv'), index_col='test_id') ) for e in exp_list]
    for e, df in dfs:
        test_df.loc[df.index, e] = df.loc[df.index, 'prob']

    return test_df

def apply_stacking(name):
    random_state=42

    exp_list=experiments


    train_df = load_stacking(exp_list)
    test_df = load_submit_stacking(exp_list)

    print explore_target_ratio(train_df)
    train_df, test_df = oversample_submit(train_df, test_df)
    print explore_target_ratio(train_df)

    estimator = xgb.XGBClassifier(n_estimators=150,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  max_depth=5,
                                  objective='binary:logistic',
                                  )
    big, small = train_df, test_df

    train_target = big[TARGET]
    del big[TARGET]
    train_arr = big

    test_arr = small

    estimator.fit(
        train_arr, train_target,
        eval_metric='logloss',
        verbose=True
    )

    proba = estimator.predict_proba(test_arr)
    test_arr[TARGET] = proba[:,1]

    res = test_df[[TARGET]]
    res.to_csv('{}.csv'.format(name), index=True, index_label='test_id')


# perform_xgb_cv()
apply_stacking('stacking_with_noun_verb_lstm_150_5_0.8_0.8')
