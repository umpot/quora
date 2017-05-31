import pandas as pd
import numpy as np
import seaborn as sns
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import os

stop_words = stopwords.words('english')

stop_words_with_PRON = set(stop_words)
stop_words_with_PRON.add('PRON')
stop_words_with_PRON.add('-PRON-')

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'

question1, question2 = 'question1', 'question2'
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
stems_q1, stems_q2 = 'stems_q1', 'stems_q2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'
postag_q1, postag_q2 = 'postag_q1', 'postag_q2'
ner_q1, ner_q2 = 'ner_q1', 'ner_q2'


data_folder = '../../../data/'

fp_train = os.path.join(data_folder,'train.csv')
fp_test = os.path.join(data_folder,'test.csv')

'tokens_train.csv', 'lemmas_train.csv', 'postag_train.csv', 'ner_train.csv'

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

def load_train_lemas_stems():
    return pd.concat([load_train_lemmas(), load_train_stems()], axis=1)

def load_test_lemas_stems():
    return pd.concat([load_test_lemmas(), load_test_stems()], axis=1)

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


def word_len(s):
    return len(s.split())


def word_len_except_stop(s):
    s = set(s.split())
    count = 0
    for w in s:
        if w not in stop_words_with_PRON:
            count += 1

    return count



def generate_lens(df, fp, index_label):
    df['len_char_q1'] = df[lemmas_q1].apply(len)
    df['len_char_q2'] = df[lemmas_q2].apply(len)
    df['len_char_diff'] = (df['len_char_q1'] - df['len_char_q2']).apply(abs)
    df['len_char_ratio'] = (df['len_char_q1']/df['len_char_q2'].apply(lambda s: 1 if s==0 else s))
    df['len_char_diff_log']=np.log(df['len_char_diff']+1)

    df['len_word_q1'] = df[lemmas_q1].apply(word_len)
    df['len_word_q2'] = df[lemmas_q2].apply(word_len)
    df['len_word_diff'] = (df['len_word_q1'] - df['len_word_q2']).apply(abs)
    df['len_word_ratio'] = (df['len_word_q1']/df['len_word_q2'].apply(lambda s: 1 if s==0 else s))
    df['len_word_diff_log']=np.log(df['len_word_diff']+1)

    df['len_word_expt_stop_q1'] = df[lemmas_q1].apply(word_len_except_stop)
    df['len_word_expt_stop_q2'] = df[lemmas_q2].apply(word_len_except_stop)
    df['len_word_expt_stop_diff'] = (df['len_word_expt_stop_q1'] - df['len_word_expt_stop_q2']).apply(abs)
    df['len_word_expt_stop_ratio'] = (df['len_word_expt_stop_q1']/df['len_word_expt_stop_q2'].apply(lambda s: 1 if s==0 else s))
    df['len_word_expt_stop_diff_log']=np.log(df['len_word_expt_stop_diff']+1)

    new_cols = [
        'len_char_q1','len_char_q2','len_char_diff','len_char_ratio','len_char_diff_log',
        'len_word_q1','len_word_q2','len_word_diff','len_word_ratio','len_word_diff_log',
        'len_word_expt_stop_q1','len_word_expt_stop_q2','len_word_expt_stop_diff','len_word_expt_stop_ratio','len_word_expt_stop_diff_log'
    ]

    df = df[new_cols]
    df.to_csv(fp, index_label=index_label)


def get_common_tokens_ratio(a,b):
    a = set(a.split())
    b= set(b.split())

    join = set(a)
    join.update(b)

    inter = a.intersection(b)

    return 1.0*len(inter)/len(join)

def get_common_tokens_num(a,b):
    a = set(a.split())
    b= set(b.split())
    inter = a.intersection(b)

    return len(inter)

def get_common_tokens_ratio_no_stops(a,b):
    a = set(a.split())
    b= set(b.split())

    join = set(a)
    join.update(b)
    join = join.difference(stop_words_with_PRON)
    join = len(join)
    join = 1 if join==0 else join

    inter = a.intersection(b)
    inter = inter.difference(stop_words_with_PRON)
    inter = len(inter)

    return 1.0*inter/join

def get_common_tokens_num_no_stops(a,b):
    a = set(a.split())
    b= set(b.split())
    inter = a.intersection(b)
    inter = inter.difference(stop_words_with_PRON)
    inter = len(inter)

    return inter

def get_stop_words_num(a):
    a=set(a.split())
    a= a.difference(stop_words_with_PRON)
    return len(a)

def get_stop_words_ratio(a):
    a=set(a.split())
    l = len(a)
    a= a.difference(stop_words_with_PRON)
    if l==0:
        return 0
    return 1.0*len(a)/l

def generate_common_words(df, fp, index_label):
    df['common_lemmas_num']=df[[lemmas_q1, lemmas_q2]].apply(lambda s: get_common_tokens_num(s[lemmas_q1], s[lemmas_q2]), axis=1)
    df['common_stems_num']=df[[stems_q1, stems_q2]].apply(lambda s: get_common_tokens_num(s[stems_q1], s[stems_q2]), axis=1)

    df['common_lemmas_num_no_stops']=df[[lemmas_q1, lemmas_q2]].apply(lambda s: get_common_tokens_num_no_stops(s[lemmas_q1], s[lemmas_q2]), axis=1)
    df['common_lemmas_ratio_no_stops']=df[[lemmas_q1, lemmas_q2]].apply(lambda s: get_common_tokens_ratio_no_stops(s[lemmas_q1], s[lemmas_q2]), axis=1)

    df['common_lemmas_ratio']=df[[lemmas_q1, lemmas_q2]].apply(lambda s: get_common_tokens_ratio(s[lemmas_q1], s[lemmas_q2]), axis=1)
    df['common_stems_ratio']=df[[stems_q1, stems_q2]].apply(lambda s: get_common_tokens_ratio(s[stems_q1], s[stems_q2]), axis=1)

    df['stop_words_num_q1']=df[lemmas_q1].apply(get_stop_words_num)
    df['stop_words_num_q2']=df[lemmas_q2].apply(get_stop_words_num)

    df['stop_words_ratio_q1']=df[lemmas_q1].apply(get_stop_words_ratio)
    df['stop_words_ratio_q2']=df[lemmas_q2].apply(get_stop_words_ratio)

    new_cols = ['common_lemmas_num', 'common_stems_num', 'common_lemmas_num_no_stops',
                'common_lemmas_ratio_no_stops','common_lemmas_ratio', 'common_stems_ratio',
                'stop_words_num_q1','stop_words_num_q2',
                'stop_words_ratio_q1','stop_words_ratio_q2'
                ]

    df = df[new_cols]
    df.to_csv(fp, index_label=index_label)


def write_basic_features_train():
    df = load_train_lemas_stems()
    index_label='id'

    fp=os.path.join(data_folder, 'basic', 'common_words_train.csv')
    generate_common_words(df, fp, index_label)

    fp=os.path.join(data_folder, 'basic', 'lens_train.csv')
    generate_lens(df, fp, index_label)


def write_basic_features_test():
    df = load_test_lemas_stems()
    index_label='test_id'

    fp=os.path.join(data_folder, 'basic', 'common_words_test.csv')
    generate_common_words(df, fp, index_label)

    fp=os.path.join(data_folder, 'basic', 'lens_test.csv')
    generate_lens(df, fp, index_label)


write_basic_features_test()