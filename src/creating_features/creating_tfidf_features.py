import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import re
import os
from dask.dataframe import from_pandas
import dask
dask.set_options(get=dask.multiprocessing.get)
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

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


common_words_train_fp = os.path.join(data_folder, 'basic','common_words_train.csv')
length_train_fp = os.path.join(data_folder, 'basic','lens_train.csv')

common_words_test_fp = os.path.join(data_folder, 'basic','common_words_test.csv')
length_test_fp = os.path.join(data_folder, 'basic','lens_test.csv')

METRICS_FP = [
    data_folder + 'train_metrics_bool_lemmas.csv',
    data_folder + 'train_metrics_bool_stems.csv',
    data_folder + 'train_metrics_bool_tokens.csv',
    data_folder + 'train_metrics_fuzzy_lemmas.csv',
    data_folder + 'train_metrics_fuzzy_stems.csv',
    data_folder + 'train_metrics_fuzzy_tokens.csv',
    data_folder + 'train_metrics_sequence_lemmas.csv',
    data_folder + 'train_metrics_sequence_stems.csv',
    data_folder + 'train_metrics_sequence_tokens.csv'
]

TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'

question1, question2 = 'question1', 'question2'
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
stems_q1, stems_q2 = 'stems_q1', 'stems_q2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'


def load_train():
    return pd.read_csv(fp_train, index_col='id')

def load_test():
    return pd.read_csv(fp_test, index_col='test_id')


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


def load_train_tokens():
    df = pd.read_csv(tokens_train_fp, index_col='id')
    df = df.fillna('')
    return df

def load_test_tokens():
    df = pd.read_csv(tokens_test_fp, index_col='test_id')
    df = df.fillna('')
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


def load_train_lengths():
    df = pd.read_csv(length_train_fp, index_col='id')
    return df

#==============================================================================
#==============================================================================


def get_tf_idf_share_ratio(t1, t2, tfidf):
    t1=tfidf.transform([t1])
    t2=tfidf.transform([t2])
    s = t1+t2
    diff = (s-np.abs(t1-t2))/2

    s=np.sum(s)
    s=1 if s==0 else s

    diff=np.sum(diff)

    return diff/s


def get_tf_idf_share(t1, t2, tfidf):
    t1=tfidf.transform([t1])
    t2=tfidf.transform([t2])
    s = t1+t2
    diff = (s-np.abs(t1-t2))/2

    return np.sum(diff)




def write_tfidf_features_dfs(train_df, test_df, col1, col2, prefix, stopwords, fp):
    tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 1))
    for col in [col1, col2]:
        for df in [train_df, test_df]:
            df[col].fillna('', inplace=True)
    #sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
    def convert_series(s):
        return s[~s.isnull()].apply(lambda s: s.lower()).tolist()

    bl = [train_df[col1], train_df[col2], test_df[col1], test_df[col2]]
    corpus=[]
    for x in bl:
        corpus+=convert_series(x)
    print len(corpus)
    tfidf.fit_transform(corpus)
    print 'TfidfVectorizer has been built'

    for i, df in enumerate([train_df, test_df]):
        label = 'train' if i==0 else 'test'
        index_label = 'id' if i==0 else 'test_id'
        new_cols=[]

        tf_q1='tf_q1'
        tf_q2='tf_q2'

        df[tf_q1] = from_pandas(df[col1], npartitions=npartitions).apply(lambda s: tfidf.transform([s])).compute()
        print tf_q1
        df[tf_q2] = from_pandas(df[col2], npartitions=npartitions).apply(lambda s: tfidf.transform([s])).compute()
        print tf_q2

        def my_mean(s):
            return np.mean(tfidf.transform([s]).data)

        new_col = '{}_tfidf_mean_q1'.format(prefix)
        print label, new_col
        new_cols.append(new_col)
        df[new_col] = from_pandas(df[col1], npartitions=npartitions).apply(my_mean).compute()

        new_col = '{}_tfidf_mean_q2'.format(prefix)
        print label, new_col
        new_cols.append(new_col)
        df[new_col] = from_pandas(df[col2], npartitions=npartitions).apply(my_mean).compute()


        def my_sum(s):
            return np.sum(tfidf.transform([s]).data)

        new_col = '{}_tfidf_sum_q1'.format(prefix)
        print label, new_col
        new_cols.append(new_col)
        df[new_col] = from_pandas(df[col1], npartitions=npartitions).apply(my_sum).compute()

        new_col = '{}_tfidf_sum_q2'.format(prefix)
        print label, new_col
        new_cols.append(new_col)
        df[new_col] = from_pandas(df[col2], npartitions=npartitions).apply(my_sum).compute()


        new_col='{}_tfidf_share'.format(prefix)
        print label, new_col
        new_cols.append(new_col)
        df[new_col]=from_pandas(df[[col1, col2]], npartitions=npartitions).apply(
            lambda s: get_tf_idf_share(s[col1], s[col2], tfidf),
            axis=1,
            ).compute()

        new_col='{}_tfidf_share_ratio'.format(prefix)
        print label, new_col
        new_cols.append(new_col)
        df[new_col]=from_pandas(df[[col1, col2]], npartitions=npartitions).apply(
            lambda s: get_tf_idf_share_ratio(s[col1], s[col2],tfidf),
            axis=1
        ).compute()



        df = df[new_cols]
        df.to_csv(os.path.join(fp, '{}_tfidf_{}.csv'.format(prefix, label)), index_label=index_label)



npartitions=23

def write_tfidf_features():
    fp=os.path.join(data_folder,'tfidf')

    col1='tokens_q1'
    col2='tokens_q2'

    train_df = load_train_tokens()
    test_df = load_test_tokens()

    stopwords = ENGLISH_STOP_WORDS
    prefix='tokens_with_stop_words'
    write_tfidf_features_dfs(train_df, test_df, col1, col2, prefix, stopwords, fp)

    stopwords = None
    prefix='tokens'
    write_tfidf_features_dfs(train_df, test_df, col1, col2, prefix, stopwords, fp)


    col1='lemmas_q1'
    col2='lemmas_q2'

    train_df = load_train_lemmas()
    test_df = load_test_lemmas()

    stopwords = ENGLISH_STOP_WORDS
    prefix='lemmas_with_stop_words'
    write_tfidf_features_dfs(train_df, test_df, col1, col2, prefix, stopwords, fp)

    stopwords = None
    prefix='lemmas'
    write_tfidf_features_dfs(train_df, test_df, col1, col2, prefix, stopwords, fp)



write_tfidf_features()




