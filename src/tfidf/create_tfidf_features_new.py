import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
from collections import Counter
from nltk.corpus import stopwords
from dask.dataframe import from_pandas
import dask
dask.set_options(get=dask.multiprocessing.get)
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import functools
import ast

STOP_WORDS = set(stopwords.words('english'))

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

def materialize(s):
    if isinstance(s, set):
        return s
    return ast.literal_eval(s)

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
        pd.read_csv(fp_train, index_col='id', encoding="utf-8")
    )

def load_test():
    return fix_nans(
        pd.read_csv(fp_test, index_col='test_id')
    )

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

def load_train_tokens():
    df = pd.read_csv(tokens_train_fp, index_col='id')
    df = df.fillna('')
    return df

def load_test_tokens():
    df = pd.read_csv(tokens_test_fp, index_col='test_id')
    df = df.fillna('')
    return df

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


# def tfidf_word_match_share(q1, q2, stops, weights):
#     q1words = {}
#     q2words = {}
#     for word in q1:
#         if word not in stops:
#             q1words[word] = 1
#     for word in q2:
#         if word not in stops:
#             q2words[word] = 1
#     if len(q1words) == 0 or len(q2words) == 0:
#         # The computer-generated chaff includes a few questions that are nothing but stopwords
#         return 0
#
#     shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
#
#     w1 = [weights.get(w, 0) for w in q1words]
#     w2 = [weights.get(w, 0) for w in q2words]
#     total_weights = w1 + w2
#
#     R = np.sum(shared_weights) / np.sum(total_weights)
#     return R



def process_idf_statistics(q1, q2, weights):
    w1=[weights.get(x,0) for x in q1]
    w2=[weights.get(x,0) for x in q2]


    q_inter = q1.intersection(q2)
    w_inter = [weights.get(x,0) for x in q_inter]

    q_union = q1.union(q2)
    w_union = [weights.get(x,0) for x in q_union]

    # q1_diff = q1.difference(q2)
    # w1_diff = {weights.get(x,0) for x in q1_diff}
    #
    # q2_diff = q2.difference(q1)
    # w2_diff = {weights.get(x,0) for x in q2_diff}

    q1_empty=len(q1)==0
    q2_empty=len(q2)==0

    w1_max = None if q1_empty else np.max(w1)
    w2_max = None if q2_empty else np.max(w2)

    w1_min = None if q1_empty else np.min(w1)
    w2_min = None if q2_empty else np.min(w2)

    w1_mean = None if q1_empty else np.mean(w1)
    w2_mean = None if q2_empty else np.mean(w2)

    w1_std = None if q1_empty else np.std(w1)
    w2_std = None if q2_empty else np.std(w2)

    w1_sum = None if q1_empty else np.sum(w1)
    w2_sum = None if q2_empty else np.sum(w2)

    w1_sum_or_zero = 0 if q1_empty else w1_sum
    w2_sum_or_zero = 0 if q2_empty else w2_sum

    w_inter = 0 if len(q_inter)==0 else np.sum(w_inter)
    w_union = 0 if len(q_union)==0 else np.sum(w_union)

    w_share = w_inter

    den = w_union
    den = 1 if den==0 else den
    w_share_ratio_1 = (2*w_share)/den

    den = w1_sum_or_zero+w2_sum_or_zero
    den = 1 if den==0 else den
    w_share_ratio_2 = (2*w_share)/den

    den = np.max([w1_sum_or_zero, w2_sum_or_zero])
    den = 1 if den==0 else den
    w_share_ratio_3 = (w_share)/den

    w_means_log = None
    if w1_mean is not None and w2_mean is not None:
        if w2_mean==0 or w1_mean==0:
            w_means_log = None
        else:
            w_means_log = np.abs(np.log(w1_mean/w2_mean))


    return w1_max, w2_max,\
           w1_min, w2_min,\
           w1_mean, w2_mean,\
           w1_std, w2_std,\
           w1_sum, w2_sum,\
           w_share,\
           w_share_ratio_1, w_share_ratio_2,w_share_ratio_3,\
           w_means_log


def smooth_idf(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1.0 / (count + eps)

def std_idf(count, N):
    return np.log(N/count)

def create_weigths(df, col1, col2, idf_func):
    train_qs = list(df[col1]) + list(df[col2])

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: idf_func(count) for word, count in counts.items()}

    return weights


tmp1='tmp1'
tmp2='tmp2'

def apply_dirty_lower_no_stops(df):
    df[tmp1] = df[question1].apply(lambda s: {x for x in set(s.lower().split()) if x not in STOP_WORDS})
    df[tmp2] = df[question2].apply(lambda s: {x for x in set(s.lower().split()) if x not in STOP_WORDS})

def apply_dirty_upper(df):
    df[tmp1] = df[question1].apply(lambda s: set(s.split()))
    df[tmp2] = df[question2].apply(lambda s: set(s.split()))

def apply_tokens_lower(df):
    df[tmp1] = df[tokens_q1].apply(lambda s: set(s.lower().split()))
    df[tmp2] = df[tokens_q2].apply(lambda s: set(s.lower().split()))

def apply_tokens_lower_no_stops(df):
    df[tmp1] = df[tokens_q1].apply(lambda s: {x for x in set(s.lower().split()) if x not in STOP_WORDS})
    df[tmp2] = df[tokens_q2].apply(lambda s: {x for x in set(s.lower().split()) if x not in STOP_WORDS})


def write_tfidf_features(is_train, is_test, name1):
    print 'BEFORE train={}, test={}'.format(is_train, is_test)
    is_train == 'true'==is_train
    is_test == 'true'==is_test

    print 'AFTERR train={}, test={}'.format(is_train, is_test)

    return

    print 'Loading dfs...'
    train_df = load_train_nlp()
    if is_test:
        test_df = load_test_nlp()
    # train_df, test_df = train_df.head(100), test_df.head(100)

    prefix_map = {
       'dirty_lower_no_stops' :apply_dirty_lower_no_stops,
        'dirty_upper':apply_dirty_upper,
        'tokens_lower':apply_tokens_lower,
        'tokens_lower_no_stops':apply_tokens_lower_no_stops
    }

    cols_map = ['w1_max',
                'w2_max',
                'w1_min',
                'w2_min',
                'w1_mean' ,
                'w2_mean',
                'w1_std',
                'w2_std',
                'w1_sum',
                'w2_sum',
                'w_share',
                'w_share_ratio_1',
                'w_share_ratio_2',
                'w_share_ratio_3',
                'w_means_log'
                ]

    new_cols = []

    cols_map = {j: cols_map[j] for j in range(len(cols_map))}

    for name, preprocess in prefix_map.iteritems():
        if name!=name1:
            continue
        print name
        idfs = [smooth_idf, lambda s: std_idf(s, len(train_df))]
        print 'Preprocessing Train...'
        preprocess(train_df)
        if is_test:
            print 'Preprocessing Test...'
            preprocess(test_df)

        print 'Creating weights ....'
        weights_map={
            'std_idf':create_weigths(train_df, tmp1, tmp2, idfs[1]),
            'smooth_idf' : create_weigths(train_df, tmp1, tmp2, idfs[0])
        }
        for idf_name, weights in weights_map.iteritems():
            print idf_name
            #TRAIN
            if is_train:
                train_df['tmp'] = train_df[[tmp1, tmp2]].apply(
                    lambda s: process_idf_statistics(s[tmp1], s[tmp2], weights), axis=1)

                for k,v in cols_map.iteritems():
                    print v
                    col = '{}_{}_{}'.format(v, idf_name, name)
                    train_df[col]= train_df['tmp'].apply(lambda s: s[k])
                    new_cols.append(col)

            if is_test:
                test_df['tmp'] = test_df[[tmp1, tmp2]].apply(
                    lambda s: process_idf_statistics(s[tmp1], s[tmp2], weights), axis=1)

                for k,v in cols_map.iteritems():
                    col = '{}_{}_{}'.format(v, idf_name, name)
                    test_df[col]= test_df['tmp'].apply(lambda s: s[k])
                    if not is_train:
                        new_cols.append(col)


        train_fp = os.path.join(data_folder, 'tfidf', 'train_{}.csv'.format(name))
        test_fp = os.path.join(data_folder, 'tfidf', 'test_{}.csv'.format(name))

        if is_train:
            train_df[new_cols].to_csv(train_fp, index_label='id')

        if is_test:
            test_df[new_cols].to_csv(test_fp, index_label='test_id')


write_tfidf_features(sys.argv[1], sys.argv[2], sys.argv[3])
