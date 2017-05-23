import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

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
ner_q1, ner_q2='ner_q1', 'ner_q2'
postag_q1, postag_q2='postag_q1', 'postag_q2'

data_folder = '../../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

lemmas_train_fp = os.path.join(data_folder,'nlp','lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder,'nlp','lemmas_test.csv')

tokens_train_fp = os.path.join(data_folder,'nlp','tokens_train.csv')
tokens_test_fp = os.path.join(data_folder,'nlp','tokens_test.csv')

postag_train_fp = os.path.join(data_folder,'nlp','postag_train.csv')
postag_test_fp = os.path.join(data_folder,'nlp','postag_test.csv')

ner_train_fp = os.path.join(data_folder,'nlp','ner_train.csv')
ner_test_fp = os.path.join(data_folder,'nlp','ner_test.csv')

stems_train_fp = os.path.join(data_folder,'nlp','stems_train.csv')
stems_test_fp = os.path.join(data_folder,'nlp','stems_test.csv')

tfidf_with_stops_train_fp = os.path.join(data_folder,'tfidf','tokens_with_stop_words_tfidf_train.csv')
tfidf_with_stops_test_fp = os.path.join(data_folder,'tfidf','tokens_with_stop_words_tfidf_test.csv')

magic_train_fp=os.path.join(data_folder, 'magic', 'magic_train.csv')
magic_test_fp=os.path.join(data_folder, 'magic', 'magic_test.csv')

magic2_train_fp = os.path.join(data_folder, 'magic', 'magic2_train.csv')
magic2_test_fp = os.path.join(data_folder, 'magic', 'magic2_test.csv')


common_words_train_fp = os.path.join(data_folder, 'basic','common_words_train.csv')
length_train_fp = os.path.join(data_folder, 'basic','lens_train.csv')

common_words_test_fp = os.path.join(data_folder, 'basic','common_words_test.csv')
length_test_fp = os.path.join(data_folder, 'basic','lens_test.csv')

TRAIN_METRICS_FP = [
    data_folder + 'distances/'+ 'train_metrics_bool_lemmas.csv',
    data_folder + 'distances/'+'train_metrics_bool_stems.csv',
    data_folder + 'distances/'+'train_metrics_bool_tokens.csv',
    data_folder + 'distances/'+'train_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/'+'train_metrics_fuzzy_stems.csv',
    data_folder + 'distances/'+'train_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/'+'train_metrics_sequence_lemmas.csv',
    data_folder + 'distances/'+'train_metrics_sequence_stems.csv',
    data_folder + 'distances/'+'train_metrics_sequence_tokens.csv'
]

TEST_METRICS_FP = [
    data_folder + 'distances/'+ 'test_metrics_bool_lemmas.csv',
    data_folder + 'distances/'+'test_metrics_bool_stems.csv',
    data_folder + 'distances/'+'test_metrics_bool_tokens.csv',
    data_folder + 'distances/'+'test_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/'+'test_metrics_fuzzy_stems.csv',
    data_folder + 'distances/'+'test_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/'+'test_metrics_sequence_lemmas.csv',
    data_folder + 'distances/'+'test_metrics_sequence_stems.csv',
    data_folder + 'distances/'+'test_metrics_sequence_tokens.csv'
]

def fix_nans(df):
    def blja(s):
        if s!=s:
            return ''
        return s

    for col in [question1, question2]:
        df[col]=df[col].apply(blja)

    return df

def load_train():
    return fix_nans(
        pd.read_csv(fp_train, index_col='id')
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
        # load_train_postag(),
        # load_train_lemmas(),
        # load_train_stems(),
        load_train_tokens()
        # load_train_ner()
    ], axis=1)

def load_test_nlp():
    return pd.concat([
        load_test(),
        # load_test_postag(),
        # load_test_lemmas(),
        # load_test_stems(),
        load_test_tokens()
        # load_test_ner()
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
        df[col]=df[col].apply(str)
    return df

def load_test_lemmas():
    df = pd.read_csv(lemmas_test_fp, index_col='test_id')
    df = df.fillna('')
    for col in [lemmas_q1, lemmas_q2]:
        df[col]=df[col].apply(str)
    return df


def load_train_tfidf():
    df = pd.read_csv(tfidf_with_stops_train_fp, index_col='id')
    return df

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
        df[col]=df[col].apply(str)
    return df

def load_test_stems():
    df = pd.read_csv(stems_test_fp, index_col='test_id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    for col in [stems_q1, stems_q2]:
        df[col]=df[col].apply(str)
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
        'pos':1.0*len(df[df[TARGET]==1])/len(df),
        'neg':1.0*len(df[df[TARGET]==0])/len(df)
    }



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
import json
from sklearn.model_selection import StratifiedKFold
from time import time
from dask.dataframe import from_pandas
import dask
from collections import Counter
dask.set_options(get=dask.multiprocessing.get)
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from joblib import Parallel, delayed

in_q1, in_q2='in_q1', 'in_q2'
tokens_set1, tokens_set2 = 'tokens_set1', 'tokens_set2'
inn='inn'

def get_all_tokens_counter(df):
    l = list(df[tokens_q1]) + list(df[tokens_q2])
    l = ' '.join(l).split()
    # l = filter(lambda s: s not in upper_stop_words, l)

    return Counter(l)

def get_top_N_tokens(df, N):
    c=get_all_tokens_counter(df)
    return [x[0] for x in c.most_common(N)]


def add_in_cols(df, w):
    w=w
    df[in_q2]=df[question2].apply(lambda s: w in s)
    df[in_q1]=df[question1].apply(lambda s: w in s)
    def m(x,y):
        if not x and not y:
            return 0
        elif x and y:
            return 1
        else:
            return -1

    df[inn] = df.apply(lambda s: m(s[in_q1], s[in_q2]), axis=1)


def create_frequencies(df, top_list):
    l = split_list(top_list, 5)
    res = Parallel(n_jobs=4, verbose=1)(delayed(get_freq_list)(ww, df) for ww in l)
    # for w in top_list:
    #     get_freq(w)
    res={x[0]:x[1] for x in flat_list(res)}

    return res


def flat_list(l):
    res=[]
    for x in l:
        res+=x

    return res


def split_list(l, N):
    res=[]
    sz = len(l)
    if sz%N==0:
        sz=sz/N
    else:
        sz=1+(sz/N)

    for j in range(sz):
        res.append(l[j*N: (j+1)*N])

    return res


def get_freq_list(l, df):
    return [get_freq(w, df) for w in l]


def get_freq(w, df):
    t=time()
    add_in_cols(df, w)
    print 'time {}'.format(time()-t)
    x = df[df[inn] == 1]
    y = df[df[inn] == -1]
    ratio_x = explore_target_ratio(x)
    ratio_y = explore_target_ratio(y)
    print w
    print 1, ratio_x
    print -1, ratio_y
    print '==========================='
    return w, {1: ratio_x, -1: ratio_y}

def add_frequencies_df(df, top_list, freq):

    def get_freq_tuple(a,b, w, freq):
        if w in a and w in b:
            return (1, freq[w][1]['pos'])
        if (w in a and w not in b) or (w in b and w not in a):
            return (-1, freq[w][-1]['pos'])
        return None

    def get_pos_from_tmp(t):
        if t is None:
            return None
        if t[0]==-1:
            return None
        return t[1]

    def get_neg_from_tmp(t):
        if t is None:
            return None
        if t[0]==1:
            return None
        return t[1]

    new_cols = []
    for w in top_list:
        df['tmp']=df.apply(lambda s: get_freq_tuple(s[tokens_set1], s[tokens_set2], w, freq), axis=1)

        col = '{}_both'.format(w)
        new_cols.append(col)
        df[col] = df['tmp'].apply(get_pos_from_tmp)

        col = '{}_only_one'.format(w)
        new_cols.append(col)
        df[col] = df['tmp'].apply(get_neg_from_tmp)

    return new_cols


def process_train_test_df(train_df, test_df, update_df, top_list):
    freq = create_frequencies(train_df, top_list)

    new_cols = add_frequencies_df(test_df, top_list, freq)
    if update_df is None:
        return
    for col in new_cols:
        update_df.loc[test_df.index, col]=test_df[col]

    return new_cols


def add_set_cols(df):
    df[tokens_set1]=df[tokens_q1].apply(lambda s: set(s.split()))
    df[tokens_set2]=df[tokens_q2].apply(lambda s: set(s.split()))


def write_top_frequencies(N):
    train_df, test_df = load_train_nlp(), load_test_nlp()
    train_df, test_df = shuffle_df(train_df, random_state=42), shuffle_df(test_df, random_state=42)
    train_df, test_df = train_df.head(5000), test_df.head(5000)
    add_set_cols(train_df)
    add_set_cols(test_df)

    top_list = get_top_N_tokens(train_df, N)
    print top_list

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    new_cols=None
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[TARGET]):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        new_cols = process_train_test_df(big, small, train_df, top_list)
        print train_df.columns


    fp=os.path.join(data_folder, 'keywords', 'train_top_{}_freq.csv'.format(N))
    train_df[new_cols].to_csv(fp, index_label='id')


    process_train_test_df(train_df, test_df, None, top_list)
    fp=os.path.join(data_folder, 'keywords', 'test_top_{}_freq.csv'.format(N))
    test_df[new_cols].to_csv(fp, index_label='test_id')


def write_folds():
    train_df = load_train()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    counter=0
    res={}
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[TARGET]):
        m={}
        res[counter]=m
        m['train']=list(big_ind)
        m['test']=list(small_ind)

    fp=os.path.join(data_folder, 'top_k_freq', 'folds.json')
    json.dump(res, open(fp, 'w+'))

write_folds()