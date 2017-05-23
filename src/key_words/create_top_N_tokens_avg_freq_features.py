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

def explore_target_ratio(df,w):
    sz = len(df)
    if sz==0:
        for x in range(100):
            print '================{}=================='.format(w)
    sz=max(1,sz)
    return {
        'pos':1.0*len(df[df[TARGET]==1]) / sz,
        'neg':1.0*len(df[df[TARGET]==0]) / sz
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


def add_set_cols(df):
    df[tokens_set1]=df[tokens_q1].apply(lambda s: set(s.split()))
    df[tokens_set2]=df[tokens_q2].apply(lambda s: set(s.split()))

folds_fp=os.path.join(data_folder, 'top_k_freq', 'folds.json')
topKtokens_fp=os.path.join(data_folder, 'top_k_freq', 'tokens.json')

out_of_fold_freq_fp = os.path.join(data_folder, 'top_k_freq', 'out_of_fold_freq.json')
train_freq_fp=os.path.join(data_folder, 'top_k_freq', 'train_freq.json')

out_of_fold_contains_fp= os.path.join(data_folder, 'top_k_freq', 'out_of_fold_contains.json')
test_contains_fp= os.path.join(data_folder, 'top_k_freq', 'test_contains.json')


out_of_fold_freq_sets_fp = os.path.join(data_folder, 'top_k_freq', 'out_of_fold_freq_sets.json')
test_freq_sets_fp = os.path.join(data_folder, 'top_k_freq', 'test_freq_sets.json')


train_avg_tokK_freq_fp=os.path.join(data_folder, 'top_k_freq', 'train_avg_K_tok_freq.csv')
test_avg_tokK_freq_fp=os.path.join(data_folder, 'top_k_freq', 'test_avg_K_tok_freq.csv')





def load_folds():
    return json.load(open(folds_fp))

def create_folds(df):
    folds = load_folds()

    return [(df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']]) for x in range(len(folds))]



def load_top_tokens():
    return json.load(open(topKtokens_fp))




def load_train_freq():
    res= json.load(open(train_freq_fp))
    res={word:
             {int(x): y['pos'] for x,y in word_res.iteritems()}
         for word, word_res in res.iteritems()}
    return res

def load_out_of_fold_freq():
    res= json.load(open(out_of_fold_freq_fp))
    res = {int(k):v for k,v in res.iteritems()}
    res={k:
             {word:
                  {int(x): y['pos'] for x,y in word_res.iteritems()}
              for word, word_res in v.iteritems()}
         for k,v in res.iteritems()}
    return res

def load_out_of_fold_contains():
    res= json.load(open(out_of_fold_contains_fp))
    res = {int(k):v for k,v in res.iteritems()}

    return res

def load_test_contains():
    res= json.load(open(test_contains_fp))
    # res = {int(k):v for k,v in res.iteritems()}

    return res



def get_from_fold(ind, folds):
    for train, test in folds:
        if ind in test.index:
            return test.loc[ind]

def get_freqs_from_fold(ind, folds, freq):
    for i in range(len(folds)):
        train, test = folds[i]
        if ind in test.index:
            return freq[i]



def write_test_freq_sets():
    Ns=[50, 100, 200, 500, 1000]
    tokens = load_top_tokens()
    train_df, test_df = load_train_nlp(), load_test_nlp()
    cont = load_test_contains()
    freq = load_train_freq()

    for w in tokens:
        bl = cont[w]
        a = set(bl['q1'])
        b=set(bl['q2'])
        plus = a.intersection(b)
        minus = a.symmetric_difference(b)
        cont[w]={1:plus, -1:minus}

    new_cols = []
    test_df['ind'] = test_df.index
    print 'Loaded'
    for N in Ns:
        print N
        toks = tokens[:N]
        blja = {ind:{1:set(), -1:set()} for ind in test_df.index}
        for w in toks:
            print w
            plus = cont[w][1]
            minus=cont[w][-1]

            for ind in plus:
                blja[ind][1].add(freq[w][1])

            for ind in minus:
                blja[ind][-1].add(freq[w][-1])

        col = 'freq_{}_plus'.format(N)
        new_cols.append(col)
        test_df[col] = test_df['ind'].apply(lambda s: blja[s][1])
        print col

        col = 'freq_{}_minus'.format(N)
        new_cols.append(col)
        test_df[col] = test_df['ind'].apply(lambda s: blja[s][-1])
        print col

    df = test_df[new_cols]
    df.to_csv(test_freq_sets_fp, index_label='test_id')


def write_out_of_fold_freq_sets():
    Ns=[50, 100, 200, 500, 1000]
    tokens = load_top_tokens()
    train_df, test_df = load_train_nlp(), load_test_nlp()

    folds = create_folds(train_df)
    contains=load_out_of_fold_contains()
    frequencies = load_out_of_fold_freq()

    for i in range(len(folds)):
        train, test = folds[i]
        cont = contains[i]

        for w in tokens:
            bl = cont[w]
            a = set(bl['q1'])
            b=set(bl['q2'])
            plus = a.intersection(b)
            minus = a.symmetric_difference(b)
            cont[w]={1:plus, -1:minus}



    new_cols = []
    train_df['ind'] = train_df.index
    print 'Loaded'
    for N in Ns:
        print N
        toks = tokens[:N]
        blja = {ind:{1:set(), -1:set()} for ind in train_df.index}
        for i in range(len(folds)):
            print 'fold_{}'.format(i)
            train, test = folds[i]
            cont = contains[i]
            freq=frequencies[i]



            for w in toks:
                print w
                plus = cont[w][1]
                minus=cont[w][-1]

                for ind in plus:
                    blja[ind][1].add(freq[w][1])

                for ind in minus:
                    blja[ind][-1].add(freq[w][-1])

        col = 'freq_{}_plus'.format(N)
        new_cols.append(col)
        train_df[col] = train_df['ind'].apply(lambda s: blja[s][1])
        print col

        col = 'freq_{}_minus'.format(N)
        new_cols.append(col)
        train_df[col] = train_df['ind'].apply(lambda s: blja[s][-1])
        print col

    df = train_df[new_cols]
    df.to_csv(out_of_fold_freq_sets_fp, index_label='id')

import ast
def load_out_of_fold_freq_set():
    bl = pd.read_csv(out_of_fold_freq_sets_fp, index_col='id')
    for col in bl.columns:
        bl[col] = bl[col].apply(lambda s: s.replace('set(', '').replace(')', ''))
        bl[col] = bl[col].apply(ast.literal_eval)

    return bl

def load_test_freq_set():
    bl = pd.read_csv(test_freq_sets_fp, index_col='test_id')
    for col in bl.columns:
        bl[col] = bl[col].apply(lambda s: s.replace('set(', '').replace(')', ''))
        bl[col] = bl[col].apply(ast.literal_eval)

    return bl


['freq_50_plus', 'freq_50_minus', 'freq_100_plus', 'freq_100_minus',
       'freq_200_plus', 'freq_200_minus', 'freq_500_plus',
       'freq_500_minus', 'freq_1000_plus', 'freq_1000_minus']


def mean_non_zero(s):
    s=filter(lambda x: x!=0, s)
    if len(s)==0:
        return None
    return np.mean(s)

def geometric_mean_non_zero(s):
    s=filter(lambda x: x!=0, s)
    if len(s)==0:
        return None

    s=[np.log(x) for x in s]
    s = np.mean(s)
    return np.exp(s)



npartitions=4
# def create_topNs_features():
#     df = load_out_of_fold_freq_set()
#     new_cols=[]
#     for col in df.columns:
#         new_col='{}_mean'.format(col)
#         print new_col
#         new_cols.append(new_col)
#         df[new_col] = from_pandas(df[col], npartitions=npartitions).apply(mean_non_zero).compute()
#
#         new_col='{}_g_mean'.format(col)
#         print new_col
#         new_cols.append(new_col)
#         df[new_col] = from_pandas(df[col], npartitions=npartitions).apply(geometric_mean_non_zero).compute()
#
#     df[new_cols].to_csv(train_avg_tokK_freq_fp, index_label='id')


def create_topNs_features_out_of_fold():
    df = load_out_of_fold_freq_set()
    print 'loaded'
    new_cols=[]
    for col in df.columns:
        new_col='{}_mean'.format(col)
        print new_col
        new_cols.append(new_col)
        df[new_col] = df[col].apply(mean_non_zero)

        new_col='{}_g_mean'.format(col)
        print new_col
        new_cols.append(new_col)
        df[new_col] = df[col].apply(geometric_mean_non_zero)

    df[new_cols].to_csv(train_avg_tokK_freq_fp, index_label='id')


# def create_topNs_features_test():
#     df = load_test_freq_set()
#     print 'loaded'
#     new_cols=[]
#     for col in df.columns:
#         new_col='{}_mean'.format(col)
#         print new_col
#         new_cols.append(new_col)
#         df[new_col] = df[col].apply(mean_non_zero)
#
#         new_col='{}_g_mean'.format(col)
#         print new_col
#         new_cols.append(new_col)
#         df[new_col] = df[col].apply(geometric_mean_non_zero)
#
#     df[new_cols].to_csv(test_avg_tokK_freq_fp, index_label='test_id')


def create_topNs_features_test():
    df = load_test_freq_set()
    print 'loaded'
    new_cols=[]
    for col in df.columns:
        new_col='{}_mean'.format(col)
        print new_col
        new_cols.append(new_col)
        df[new_col] = df[col].apply(mean_non_zero)

        new_col='{}_g_mean'.format(col)
        print new_col
        new_cols.append(new_col)
        df[new_col] = df[col].apply(geometric_mean_non_zero)

    df[new_cols].to_csv(test_avg_tokK_freq_fp, index_label='test_id')


def load_topNs_avg_tok_freq_train():
    return pd.read_csv(train_avg_tokK_freq_fp, index_col='id')

def load_topNs_avg_tok_freq_test():
    return pd.read_csv(test_avg_tokK_freq_fp, index_col='test_id')


write_test_freq_sets()
create_topNs_features_test()