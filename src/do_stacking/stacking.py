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

gc_host = '104.197.97.20'
local_host = '10.20.0.144'
user='ubik'
password='nfrf[eqyz'

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

experiments = ['stacking_lstm_glove_nouns_re_stops_no',
               'one_upper_magic_wh_common_words_lengths_500_0.6_0.6_5',
               'stacking_xgb_with_lstm_prob_deep',
               'stacking_lstm_glove_lemmas_re_stops_yes',
               'glove_metrics_lex_metrics_word2vec_metrics_500_0.8_0.8_5',
               'stacking_all1_deep',
               'stacking_no_top_tokens_light',
               'stacking_only_word2vec_emb_light',
               'topNs_avg_tok_freq_magic_500_0.8_0.8_5',
               'stacking_no_metrics_only_glove_emb',
               'lengths_common_words_500_0.8_0.8_5',
               'stacking_no_metrics_light',
               'stacking_lstm_word2vec_lemmas_re_stops_yes',
               'tfidf_new_magic_word2vec_metrics_500_0.6_0.6_5',
               'stacking_random_forest_light',
               'stacking_no_emb_simpl_idf_light',
               'stacking_lstm_lex_question_re_stops_yes',
               'metrics_500_0.8_0.8_5',
               'stacking_only_lex_emb_light',
               'stacking_no_tfidf_light',
               'tfidf_new_magic_500_0.6_0.6_5',
               'tfidf_new_magic_topNs_avg_tok_freq_500_0.6_0.6_5',
               'lengths_common_words_magic_500_0.8_0.8_5',
               'glove_metrics_500_0.8_0.8_5',
               'lengths_common_words_topNs_avg_tok_freq_500_0.6_0.6_5',
               'topNs_avg_tok_freq_500_0.8_0.8_5',
               'stacking_only_glove_emb_light',
               'stacking_lstm',
               'stacking_lstm_glove_verbs_re_stops_no',
               'glove_metrics_tfidf_new_500_0.8_0.8_5'
               ]+\
['new_top_uppers_magic_300_0.7_0.7_5',
 'glove_metrics_500_0.6_0.7_5',
 'top_7K_pair_freq_top_7K_x_None_freq_lstm_500_0.6_0.7_5',
 'pronoun_pairs_50_aux_pairs_50_magic_top_7K_pair_freq_top_7K_x_None_freq_300_0.6_0.7_4',
 'new_top_uppers_500_0.7_0.7_5',
 'top_7K_pair_freq_top_7K_x_None_freq_500_0.6_0.7_3',
 'diff_idf_new_top_uppers_pair_freq_lex_metrics_lengths_common_words_400_0.6_0.7_4',
 'lstm_magic_max_k_cores_400_0.6_0.7_3',
 'lstm_diff_idf_magic_500_0.7_0.7_5',
 'pronoun_pairs_50_aux_pairs_50_magic_300_0.6_0.7_4',
 'diff_idf_magic_500_0.7_0.7_5',
 'top_7K_pair_freq_magic_top_7K_x_None_freq_300_0.6_0.7_3',
 'common_words_lengths_diff_idf_magic_500_0.7_0.7_5',
 'new_top_uppers_magic_300_0.6_0.7_5',
 'diff_idf_500_0.7_0.7_5',
 'diff_idf_magic_max_k_cores_top_7K_x_None_freq_500_0.6_0.7_3',
 'glove_metrics_lstm_magic_500_0.6_0.7_5',
 'top_7K_pair_freq_top_7K_x_None_freq_pair_freq_pronoun_pairs_50_glove_metrics_400_0.6_0.7_4',
 'new_top_uppers_pair_freq_lex_metrics_500_0.6_0.7_5']+[
    'stacking_all3'
]





def perform_xgb_cv(name, max_depth, learning_rate, subsample, colsample_bytree):
    print 'name={},\n' \
          ' max_depth={},\n ' \
          'learning_rate={},\n' \
          ' subsample={},\n colsample_bytree={}'.\
        format(name, max_depth, learning_rate, subsample, colsample_bytree)
    seed = 42
    name = '{}_{}_{}_{}_{}'.format(
        name,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree)

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
                                      subsample=subsample,
                                      colsample_bytree=colsample_bytree,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      objective='binary:logistic',
                                      nthread=-1
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

        write_results(name, estimator, losses, train_arr)

    print '###############################'
    print '###############################'
    print
    print 'AVG={}'.format(np.mean(losses))
    print
    print '###############################'
    print '###############################'


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

    estimator = xgb.XGBClassifier(n_estimators=1130,
                                  subsample=0.5,
                                  colsample_bytree=0.9,
                                  max_depth=5,
                                  objective='binary:logistic',
                                  nthread=-1,
                                  learning_rate=0.01
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

def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['logloss']
    results_on_train = estimator.evals_result()['validation_0']['logloss']
    return {
        'train': results_on_train,
        'test': results_on_test
    }

def write_results(name, estimator, losses, train_arr):
    from pymongo import MongoClient

    features =train_arr.columns
    ii = estimator.feature_importances_
    per_tree_res = xgboost_per_tree_results(estimator)

    imp=[x.item() for x in ii]
    features=list(features)

    client = MongoClient(gc_host, 27017)
    client['admin'].authenticate(user, password)
    db = client['stacking_exp']
    collection = db[name]
    try:
        print 'INSERTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        collection.insert_one({
            'results': per_tree_res,
            'losses': losses,
            'AVG':np.mean(losses),
            'importance':imp,
            'features':features,
            'best_iteration':estimator.best_iteration,
            'best_score':estimator.best_score

        })
    except:
        print 'error in mongo'
        traceback.print_exc()
        raise
        # sleep(20)


# ['python', '-u', name, str(max_depth), str(learning_rate), str(subsample), str(colsample_bytree)])
#name, max_depth, learning_rate, subsample, colsample_bytree
# perform_xgb_cv(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))

# apply_stacking('stacking_with_many_weak_est_1130_5_0.5_0.9')
perform_xgb_cv('all3_test', 5, 0.02, 0.8, 0.8)