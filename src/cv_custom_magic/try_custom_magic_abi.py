import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys

from sklearn.model_selection import StratifiedKFold

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
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
stems_q1, stems_q2 = 'stems_q1', 'stems_q2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'
ner_q1, ner_q2 = 'ner_q1', 'ner_q2'
postag_q1, postag_q2 = 'postag_q1', 'postag_q2'

data_folder = '../../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

folds_fp = os.path.join(data_folder, 'top_k_freq', 'folds.json')


def load_folds():
    return json.load(open(folds_fp))


def create_folds(df):
    folds = load_folds()

    return [(df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']]) for x in range(len(folds))]

def split_into_folds(df, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    skf.get_n_splits(df, df[TARGET])
    res=[]
    for big_ind, small_ind in skf.split(df, df[TARGET]):
        res.append((df.iloc[big_ind], df.iloc[small_ind]))

    return res


lemmas_train_fp = os.path.join(data_folder, 'nlp', 'lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder, 'nlp', 'lemmas_test.csv')

tokens_train_fp = os.path.join(data_folder, 'nlp', 'tokens_train.csv')
tokens_test_fp = os.path.join(data_folder, 'nlp', 'tokens_test.csv')

postag_train_fp = os.path.join(data_folder, 'nlp', 'postag_train.csv')
postag_test_fp = os.path.join(data_folder, 'nlp', 'postag_test.csv')

ner_train_fp = os.path.join(data_folder, 'nlp', 'ner_train.csv')
ner_test_fp = os.path.join(data_folder, 'nlp', 'ner_test.csv')

stems_train_fp = os.path.join(data_folder, 'nlp', 'stems_train.csv')
stems_test_fp = os.path.join(data_folder, 'nlp', 'stems_test.csv')

tfidf_with_stops_train_fp = os.path.join(data_folder, 'tfidf','old', 'tokens_with_stop_words_tfidf_train.csv')
tfidf_with_stops_test_fp = os.path.join(data_folder, 'tfidf','old', 'tokens_with_stop_words_tfidf_test.csv')

magic_train_fp = os.path.join(data_folder, 'magic', 'magic_train.csv')
magic_test_fp = os.path.join(data_folder, 'magic', 'magic_test.csv')

magic2_train_fp = os.path.join(data_folder, 'magic', 'magic2_train.csv')
magic2_test_fp = os.path.join(data_folder, 'magic', 'magic2_test.csv')

common_words_train_fp = os.path.join(data_folder, 'basic', 'common_words_train.csv')
length_train_fp = os.path.join(data_folder, 'basic', 'lens_train.csv')

common_words_test_fp = os.path.join(data_folder, 'basic', 'common_words_test.csv')
length_test_fp = os.path.join(data_folder, 'basic', 'lens_test.csv')

TRAIN_METRICS_FP = [
    data_folder + 'distances/' + 'train_metrics_bool_lemmas.csv',
    data_folder + 'distances/' + 'train_metrics_bool_stems.csv',
    data_folder + 'distances/' + 'train_metrics_bool_tokens.csv',
    data_folder + 'distances/' + 'train_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/' + 'train_metrics_fuzzy_stems.csv',
    data_folder + 'distances/' + 'train_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/' + 'train_metrics_sequence_lemmas.csv',
    data_folder + 'distances/' + 'train_metrics_sequence_stems.csv',
    data_folder + 'distances/' + 'train_metrics_sequence_tokens.csv'
]

TEST_METRICS_FP = [
    data_folder + 'distances/' + 'test_metrics_bool_lemmas.csv',
    data_folder + 'distances/' + 'test_metrics_bool_stems.csv',
    data_folder + 'distances/' + 'test_metrics_bool_tokens.csv',
    data_folder + 'distances/' + 'test_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/' + 'test_metrics_fuzzy_stems.csv',
    data_folder + 'distances/' + 'test_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/' + 'test_metrics_sequence_lemmas.csv',
    data_folder + 'distances/' + 'test_metrics_sequence_stems.csv',
    data_folder + 'distances/' + 'test_metrics_sequence_tokens.csv'
]


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


def load_test():
    return fix_nans(
        pd.read_csv(fp_test, index_col='test_id')
    )


def load__train_metrics():
    dfs = [pd.read_csv(fp, index_col='id') for fp in TRAIN_METRICS_FP]
    return pd.concat(dfs, axis=1)


def load__test_metrics():
    dfs = [pd.read_csv(fp, index_col='test_id') for fp in TEST_METRICS_FP]
    return pd.concat(dfs, axis=1)


def load_train_all():
    return pd.concat([
        load_train(),
        load_train_lemmas(),
        load_train_stems(),
        load_train_tokens(),
        load_train_lengths(),
        load_train_common_words(),
        load__train_metrics(),
        load_train_tfidf()
    ], axis=1)


def load_train_nlp():
    return pd.concat([
        load_train(),
        load_train_postag(),
        load_train_lemmas(),
        load_train_stems(),
        load_train_tokens(),
        load_train_ner()
    ], axis=1)


def load_test_nlp():
    return pd.concat([
        load_test(),
        load_test_postag(),
        load_test_lemmas(),
        load_test_stems(),
        load_test_tokens(),
        load_test_ner()
    ], axis=1)


def load_test_all():
    return pd.concat([
        load_test(),
        load_test_lemmas(),
        load_test_stems(),
        load_test_tokens(),
        load_test_lengths(),
        load_test_common_words(),
        load__test_metrics(),
        load_test_tfidf()
    ], axis=1)


def load_train_test():
    return pd.read_csv(fp_train, index_col='id'), pd.read_csv(fp_test, index_col='test_id')


def load_train_lemmas():
    df = pd.read_csv(lemmas_train_fp, index_col='id')
    df = df.fillna('')
    for col in [lemmas_q1, lemmas_q2]:
        df[col] = df[col].apply(str)
    return df


def load_test_lemmas():
    df = pd.read_csv(lemmas_test_fp, index_col='test_id')
    df = df.fillna('')
    for col in [lemmas_q1, lemmas_q2]:
        df[col] = df[col].apply(str)
    return df


def load_train_tfidf():
    df = pd.read_csv(tfidf_with_stops_train_fp, index_col='id')
    return df


def load_train_tfidf_new():
    fps = [
        os.path.join(data_folder, 'tfidf', x) for x in ['train_dirty_lower_no_stops.csv',
                                                        'train_dirty_upper.csv',
                                                        'train_tokens_lower.csv',
                                                        'train_tokens_lower_no_stops.csv']
        ]
    return pd.concat(
        [pd.read_csv(fp, index_col='id') for fp in fps],
        axis=1)

def load_test_tfidf_new():
    fps = [
        os.path.join(data_folder, 'tfidf', x) for x in ['test_dirty_lower_no_stops.csv',
                                                        'test_dirty_upper.csv',
                                                        'test_tokens_lower.csv',
                                                        'test_tokens_lower_no_stops.csv']
        ]
    return pd.concat(
        [pd.read_csv(fp, index_col='test_id') for fp in fps],
        axis=1)


def load_test_tfidf():
    df = pd.read_csv(tfidf_with_stops_test_fp, index_col='test_id')
    return df


def load_train_tokens():
    df = pd.read_csv(tokens_train_fp, index_col='id')
    df = df.fillna('')
    return df


def load_test_tokens():
    df = pd.read_csv(tokens_test_fp, index_col='test_id')
    df = df.fillna('')
    return df


def load_train_postag():
    df = pd.read_csv(postag_train_fp, index_col='id')
    return df


def load_test_postag():
    df = pd.read_csv(postag_test_fp, index_col='test_id')
    return df


def load_train_ner():
    df = pd.read_csv(ner_train_fp, index_col='id')
    return df


def load_test_ner():
    df = pd.read_csv(ner_test_fp, index_col='test_id')
    return df


def load_train_magic():
    df = pd.concat([
        pd.read_csv(magic_train_fp, index_col='id')[['freq_question1', 'freq_question2']],
        pd.read_csv(magic2_train_fp, index_col='id')],
        axis=1
    )
    return df


def load_test_magic():
    df = pd.concat([
        pd.read_csv(magic_test_fp, index_col='test_id')[['freq_question1', 'freq_question2']],
        pd.read_csv(magic2_test_fp, index_col='test_id')],
        axis=1
    )
    return df


def load_train_stems():
    df = pd.read_csv(stems_train_fp, index_col='id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    for col in [stems_q1, stems_q2]:
        df[col] = df[col].apply(str)
    return df


def load_test_stems():
    df = pd.read_csv(stems_test_fp, index_col='test_id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    for col in [stems_q1, stems_q2]:
        df[col] = df[col].apply(str)
    return df


def load_train_common_words():
    df = pd.read_csv(common_words_train_fp, index_col='id')
    return df


def load_test_common_words():
    df = pd.read_csv(common_words_test_fp, index_col='test_id')
    return df


def load_train_lengths():
    df = pd.read_csv(length_train_fp, index_col='id')
    return df


def load_test_lengths():
    df = pd.read_csv(length_test_fp, index_col='test_id')
    return df


def shuffle_df(df, random_state=42):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]


def explore_target_ratio(df):
    return {
        'pos': 1.0 * len(df[df[TARGET] == 1]) / len(df),
        'neg': 1.0 * len(df[df[TARGET] == 0]) / len(df)
    }


# df = load_train_all()
######################################################################################
######################################################################################
######################################################################################
######################################################################################

# WH

wh_fp_train = os.path.join(data_folder, 'wh', 'wh_train.csv')
wh_fp_test = os.path.join(data_folder, 'wh', 'wh_test.csv')


def load_wh_train():
    df = pd.read_csv(wh_fp_train, index_col='id')
    return df


def load_wh_test():
    df = pd.read_csv(wh_fp_test, index_col='test_id')
    return df

    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################




# df = load_train_all()
######################################################################################
######################################################################################
######################################################################################
######################################################################################

#WH

wh_fp_train=os.path.join(data_folder, 'wh', 'wh_train.csv')
wh_fp_test=os.path.join(data_folder, 'wh', 'wh_test.csv')

def load_wh_train():
    df = pd.read_csv(wh_fp_train, index_col='id')
    return df

def load_wh_test():
    df = pd.read_csv(wh_fp_test, index_col='test_id')
    return df

######################################################################################
######################################################################################
######################################################################################
######################################################################################






######################################################################################
######################################################################################
######################################################################################
######################################################################################
upper_keywords_fp_train=os.path.join(data_folder, 'keywords', 'train_upper.csv')
upper_keywords_test=os.path.join(data_folder, 'keywords', 'test_upper.csv')

def load_upper_keywords_train():
    df = pd.read_csv(upper_keywords_fp_train, index_col='id')
    return df

def load_upper_keywords_test():
    df = pd.read_csv(upper_keywords_test, index_col='test_id')
    return df

######################################################################################
######################################################################################
######################################################################################
######################################################################################


######################################################################################
######################################################################################
######################################################################################
######################################################################################
one_upper_fp_train=os.path.join(data_folder, 'keywords', 'train_upper_freq_200.csv')
one_upper_fp_test=os.path.join(data_folder, 'keywords', 'test_upper_freq_200.csv')

def load_one_upper_train():
    df = pd.read_csv(one_upper_fp_train, index_col='id')
    return df

def load_one_upper_test():
    df = pd.read_csv(one_upper_fp_test, index_col='test_id')
    return df

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

def shuffle_df(df, random_state):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]

def oversample_df(df, l, random_state):
    df_pos = df[df[TARGET]==1]
    df_neg = df[df[TARGET]==0]

    df_neg_sampl = df_neg.sample(l, random_state=random_state, replace=True)

    df=pd.concat([df_pos, df_neg, df_neg_sampl])
    df = shuffle_df(df, random_state)

    return df

def oversample(train_df, test_df, random_state=42):
    l_train = int(delta * len(train_df))
    l_test = int(delta * len(test_df))

    return oversample_df(train_df, l_train, random_state), oversample_df(test_df, l_test, random_state)



############################################################3
############################################################3
############################################################3
train_avg_tokK_freq_fp=os.path.join(data_folder, 'top_k_freq', 'train_avg_K_tok_freq.csv')

def load_topNs_avg_tok_freq_train():
    return pd.read_csv(train_avg_tokK_freq_fp, index_col='id')

############################################################3
############################################################3
############################################################3
abi_train_fp = os.path.join(data_folder, 'abishek', 'abi_train.csv')
abi_test_fp = os.path.join(data_folder, 'abishek', 'abi_test.csv')


def load_abi_train():
    return pd.read_csv(abi_train_fp, index_col='id')

def load_abi_test():
    return pd.read_csv(abi_test_fp, index_col='test_id')

############################################################3
############################################################3
############################################################3
freq_question1,  freq_question2,  q1_q2_intersect='freq_question1',  'freq_question2',  'q1_q2_intersect'

q1_target_ratio, q2_target_ratio = 'q1_target_ratio', 'q2_target_ratio'
q1_as_q1_count, q1_as_q1_dups_num = 'q1_as_q1_count', 'q1_as_q1_dups_num'
q2_as_q2_count, q2_as_q2_dups_num = 'q2_as_q2_count', 'q2_as_q2_dups_num'
q1_as_q2_count, q1_as_q2_dups_num = 'q1_as_q2_count', 'q1_as_q2_dups_num'
q2_as_q1_count, q2_as_q1_dups_num = 'q2as_q1_count', 'q2_as_q1_dups_num'


q1_dup_freq, q2_dup_freq, q1_q2_dup_freq='q1_dup_freq', 'q2_dup_freq', 'q1_q2_dup_freq'


def custom_magic_with_update(bf_train, bf_test, update_df):
    q1_as_q1 = bf_train.groupby(question1)[TARGET].agg({q1_as_q1_count: 'count', q1_as_q1_dups_num: 'sum'})
    bf_train = pd.merge(bf_train, q1_as_q1, left_on=question1, right_index=True, how='left')

    q2_as_q2 = bf_train.groupby(question2)[TARGET].agg({q2_as_q2_count: 'count', q2_as_q2_dups_num: 'sum'})
    bf_train = pd.merge(bf_train, q2_as_q2, left_on=question2, right_index=True, how='left')

    suka_q1 = pd.merge(q1_as_q1, q2_as_q2, left_index=True, right_index=True, how='left')
    suka_q1 = suka_q1.rename(columns={q2_as_q2_count:q1_as_q2_count, q2_as_q2_dups_num:q1_as_q2_dups_num})
    suka_q1.fillna(0, inplace=True)
    bf_train = pd.merge(bf_train, suka_q1[[q1_as_q2_count, q1_as_q2_dups_num]], left_on=question1, right_index=True, how='left')

    suka_q2 = pd.merge(q2_as_q2, q1_as_q1, left_index=True, right_index=True, how='left')
    suka_q2 = suka_q2.rename(columns={q1_as_q1_count:q2_as_q1_count, q1_as_q1_dups_num:q2_as_q1_dups_num})
    suka_q2.fillna(0, inplace=True)
    bf_train = pd.merge(bf_train, suka_q2[[q2_as_q1_count, q2_as_q1_dups_num]], left_on=question2, right_index=True, how='left')

    bf_train[q1_dup_freq] = (bf_train[q1_as_q1_dups_num] + bf_train[q1_as_q2_dups_num]) / (bf_train[q1_as_q1_count] + bf_train[q1_as_q2_count])
    bf_train[q2_dup_freq] = (bf_train[q2_as_q1_dups_num] + bf_train[q2_as_q2_dups_num]) / (bf_train[q2_as_q1_count] + bf_train[q2_as_q2_count])

    bf_train[q1_q2_dup_freq] = (bf_train[q1_as_q1_dups_num] + bf_train[q1_as_q2_dups_num] + bf_train[q2_as_q1_dups_num] + bf_train[q2_as_q2_dups_num]) / \
                               (bf_train[q1_as_q1_count] + bf_train[q1_as_q2_count] + bf_train[q2_as_q1_count] + bf_train[q2_as_q2_count])


    bf_test = pd.merge(bf_test, q1_as_q1, left_on=question1, right_index=True, how='left')
    bf_test = pd.merge(bf_test, q2_as_q2, left_on=question2, right_index=True, how='left')


    bf_test = pd.merge(bf_test, suka_q1[[q1_as_q2_count, q1_as_q2_dups_num]], left_on=question1, right_index=True, how='left')
    bf_test = pd.merge(bf_test, suka_q2[[q2_as_q1_count, q2_as_q1_dups_num]], left_on=question2, right_index=True, how='left')
    bf_test.fillna(0, inplace=True)

    # print bf_test.columns.values

    bf_test[q1_dup_freq] = (bf_test[q1_as_q1_dups_num] + bf_test[q1_as_q2_dups_num]) / (bf_test[q1_as_q1_count] + bf_test[q1_as_q2_count])
    bf_test[q2_dup_freq] = (bf_test[q2_as_q1_dups_num] + bf_test[q2_as_q2_dups_num]) / (bf_test[q2_as_q1_count] + bf_test[q2_as_q2_count])

    bf_test[q1_q2_dup_freq] = (bf_test[q1_as_q1_dups_num] + bf_test[q1_as_q2_dups_num] + bf_test[q2_as_q1_dups_num] + bf_test[q2_as_q2_dups_num]) / \
                              (bf_test[q1_as_q1_count] + bf_test[q1_as_q2_count] + bf_test[q2_as_q1_count] + bf_test[q2_as_q2_count])


    if update_df is None:
        return bf_train, bf_test

    new_cols = [q1_dup_freq, q2_dup_freq, q1_q2_dup_freq]
    for col in new_cols:
        update_df.loc[bf_test.index, col] = bf_test.loc[bf_test.index, col]

    return bf_train, bf_test


def add_custom_magic_features_one_cv_fold(cv_train, cv_test):
    folds = split_into_folds(cv_train)
    for train, test in folds:
        custom_magic_with_update(train, test, update_df=cv_train)

    custom_magic_with_update(cv_train, cv_test, update_df=cv_test)



def add_custom_magic_features_submit(train_df, test_df, folds):
    for train, test in folds:
        custom_magic_with_update(train, test, update_df=train_df)

    custom_magic_with_update(train_df, test_df, update_df=test_df)


############################################################3
############################################################3
############################################################3
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import json
from time import sleep
import traceback

gc_host = '104.197.97.20'
local_host = '10.20.0.144'
user='ubik'
password='nfrf[eqyz'


def load_train_all_xgb():
    train_df = pd.concat([
        load_train(),
        load_train_lengths(),
        load_train_common_words(),
        load__train_metrics(),
        load_train_tfidf(),
        load_train_magic(),
        load_wh_train(),
        load_one_upper_train(),
        load_topNs_avg_tok_freq_train(),
        load_abi_train()
        # load_upper_keywords_train()
    ], axis=1)

    cols_to_del = [qid1, qid2, question1, question2]
    for col in cols_to_del:
        del train_df[col]

    return train_df


def load_train_all_xgb_no_drop_qs():
    train_df = pd.concat([
        load_train(),
        load_train_lengths(),
        load_train_common_words(),
        load__train_metrics(),
        load_train_tfidf(),
        load_train_magic(),
        load_wh_train(),
        load_one_upper_train(),
        load_topNs_avg_tok_freq_train(),
        load_abi_train()
        # load_upper_keywords_train()
    ], axis=1)

    return train_df


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

def out_loss(loss):
    print '====================================='
    print '====================================='
    print '====================================='
    print loss
    print '====================================='
    print '====================================='
    print '====================================='


def write_results(name,mongo_host, per_tree_res, losses, imp, features):
    from pymongo import MongoClient

    imp=[x.item() for x in imp]
    features=list(features)

    client = MongoClient(mongo_host, 27017)
    client['admin'].authenticate(user, password)
    db = client['xgb_cv']
    collection = db[name]
    try:
        collection.insert_one({
            'results': per_tree_res,
            'losses': losses,
            'importance':imp,
            'features':features
        })
    except:
        print 'error in mongo'
        traceback.print_exc()
        raise
        # sleep(20)

def drop_qs(df):
    cols_to_del = [qid1, qid2, question1, question2]
    for col in cols_to_del:
        del df[col]


def perform_xgb_cv(name, mongo_host):
    df = load_train_all_xgb_no_drop_qs()
    # df = load_train()
    folds =5
    seed = 42

    # skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    losses = []
    n_est=[]

    folds= create_folds(df)

    for big, small in folds:
        # big = df.iloc[big_ind]
        # small = df.iloc[small_ind]

        print explore_target_ratio(big)
        print explore_target_ratio(small)

        add_custom_magic_features_one_cv_fold(big, small)
        drop_qs(big)
        drop_qs(small)


        big, small = oversample(big, small, seed)

        print explore_target_ratio(big)
        print explore_target_ratio(small)

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
                                      )
        print test_arr.columns.values
        print len(train_arr)
        print len(test_arr)
        eval_set = [(train_arr, train_target), (test_arr, test_target)]
        estimator.fit(
            train_arr, train_target,
            eval_set=eval_set,
            eval_metric='logloss',
            verbose=True,
            early_stopping_rounds=150
        )

        proba = estimator.predict_proba(test_arr)

        loss = log_loss(test_target, proba)
        out_loss(loss)
        losses.append(loss)
        per_tree_res = xgboost_per_tree_results(estimator)
        ii = estimator.feature_importances_
        n_est.append(estimator.best_iteration)



        write_results(name, mongo_host, per_tree_res, losses, ii, train_arr.columns)


    out_loss('avg = {}'.format(np.mean([x['loss'] for x in losses])))


name='try_custom_magic_abi_0.8_0.8_5'
perform_xgb_cv(name, gc_host)



print '============================'
print 'DONE!'
print '============================'




