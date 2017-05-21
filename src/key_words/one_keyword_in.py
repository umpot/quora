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


def load_train():
    return pd.read_csv(fp_train, index_col='id')

def load_test():
    return pd.read_csv(fp_test, index_col='test_id')


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
    sz = len(df)
    sz=1 if sz==0 else sz
    return {
        'pos':1.0*len(df[df[TARGET]==1])/sz,
        'neg':1.0*len(df[df[TARGET]==0])/sz
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
dask.set_options(get=dask.multiprocessing.get)
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
npartitions=4

top_upper_200_fp=os.path.join(data_folder, 'top_uppers_200.json')
in_q1, in_q2='in_upper_q1', 'in_upper_q2'
inn='inn_upper'
upper_plus_freq='upper_plus_freq_200'
upper_minus_freq='upper_minus_freq_200'

def get_top_upper(a,b, top_list, top_set):
    s=a+' '+b
    s=set(s.split())
    s=s.intersection(top_set)
    if len(s)==0:
        return None

    s=[(x, top_list.index(x)) for x in s]
    s.sort(key=lambda x: x[1])

    return s[0][0]

def create_frequencies(df, top_list):
    res = {}
    for w in top_list:
        t=time()
        add_in_cols(df,w)
        print 'time {}'.format(time()-t)
        x=df[df[inn]==1]
        y=df[df[inn]==-1]
        ratio_x = explore_target_ratio(x)
        ratio_y = explore_target_ratio(y)

        print w
        print 1, ratio_x
        print -1, ratio_y
        print '==========================='

        res[w]={1: ratio_x, -1: ratio_y}

    return res

def add_upper_frequencies_df(df, top_list, top_set, freq):
    # top_list= load_top_200_uppers()
    # top_set=set(top_list)
    # freq = create_frequencies(df, top_list)
    def get_upper_plus(a,b):
        up = get_top_upper(a,b, top_list, top_set)
        if up is None:
            return None
        if up in a and up in b:
            return freq[up][1]['pos']
        return None

    def get_upper_minus(a,b):
        up = get_top_upper(a,b, top_list, top_set)
        if up is None:
            return None
        if (up in a and up not in b) and (up in b and up not in a):
            return freq[up][-1]['pos']
        return None

    df[upper_plus_freq]=df.apply(lambda s: get_upper_plus(s[tokens_q1], s[tokens_q2]), axis=1)
    df[upper_minus_freq]=df.apply(lambda s: get_upper_minus(s[tokens_q1], s[tokens_q2]), axis=1)


def process_train_test_df(train_df, test_df, update_df):
    top_list= load_top_200_uppers()
    top_set=set(top_list)
    freq = create_frequencies(train_df, top_list)

    add_upper_frequencies_df(test_df, top_list, top_set, freq)
    if update_df is None:
        return
    for col in [upper_plus_freq, upper_minus_freq]:
        update_df.loc[test_df.index, col]=test_df[col]


def write_upper_frequencies():
    train_df, test_df = load_train_nlp(), load_test_nlp()
    train_df, test_df = shuffle_df(train_df, random_state=42), shuffle_df(test_df, random_state=42)
    train_df, test_df = train_df.head(100), test_df.head(100)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for big_ind, small_ind in skf.split(np.zeros(len(train_df)), train_df[TARGET]):
        big = train_df.iloc[big_ind]
        small = train_df.iloc[small_ind]
        process_train_test_df(big, small, train_df)
        print train_df.columns

    process_train_test_df(train_df, test_df, None)

    new_cols=[upper_plus_freq, upper_minus_freq]

    fp=os.path.join(data_folder, 'keywords', 'train_upper_freq_200.csv')
    train_df[new_cols].to_csv(fp, index_label='id')

    fp=os.path.join(data_folder, 'keywords', 'test_upper_freq_200.csv')
    test_df[new_cols].to_csv(fp, index_label='test_id')








def load_top_200_uppers():
    res= json.load(open(top_upper_200_fp))
    return [x[0] for x in res]

def add_in_cols(df, w):
    w=w
    df[in_q2]=from_pandas(df[question2], npartitions=npartitions).apply(lambda s: w in str(s)).compute()
    df[in_q1]=from_pandas(df[question1], npartitions=npartitions).apply(lambda s: w in str(s)).compute()
    def m(x,y):
        if not x and not y:
            return 0
        elif x and y:
            return 1
        else:
            return -1

    df[inn] = from_pandas(df, npartitions=npartitions).apply(lambda s: m(s[in_q1], s[in_q2]), axis=1).compute()


# def add_in_cols(df, w):
#     w=w
#     df[in_q2]=df[question2].apply(lambda s: w in str(s))
#     df[in_q1]=df[question1].apply(lambda s: w in str(s))
#     def m(x,y):
#         if not x and not y:
#             return 0
#         elif x and y:
#             return 1
#         else:
#             return -1
#
#     df[inn] = df.apply(lambda s: m(s[in_q1], s[in_q2]), axis=1)

def explore_for_keywords(df,ww):
    for w in ww:
        add_in_cols(df,w)
        x=df[df[inn]==1]
        y=df[df[inn]==-1]
        print w
        print '1: len={}, {}'.format(len(x), explore_target_ratio(x))
        print '-1: len={}, {}'.format(len(y), explore_target_ratio(y))
        print '======================================================'


write_upper_frequencies()