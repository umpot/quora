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


def add_in_cols(df, w):
    df[in_q2]=df[tokens_set2].apply(lambda s: w in s)
    df[in_q1]=df[tokens_set1].apply(lambda s: w in s)
    def m(x,y):
        if not x and not y:
            return 0
        elif x and y:
            return 1
        else:
            return -1

    df[inn] = df.apply(lambda s: m(s[in_q1], s[in_q2]), axis=1)

def create_frequencies(df, top_list):
    l = split_list(top_list, def_split)
    res = Parallel(n_jobs=n_jobs, verbose=1)(delayed(get_freq_list)(ww, df) for ww in l)
    # for w in top_list:
    #     get_freq(w)
    res={x[0]:x[1] for x in flat_list(res)}

    return res

def create_contains_data(df, top_list):
    l = split_list(top_list, def_split)
    res = Parallel(n_jobs=n_jobs, verbose=1)(delayed(get_contains_data_list)(ww, df) for ww in l)
    # for w in top_list:
    #     get_freq(w)
    res={x[0]:x[1] for x in flat_list(res)}

    return res

def get_contains_data_list(ww, df):
    return [get_contains_data(w, df) for w in ww]

def get_contains_data(w, df):
    print w
    return w, {
        'q1':list(df[df[tokens_set1].apply(lambda s: w in s)].index),
        'q2':list(df[df[tokens_set2].apply(lambda s: w in s)].index)
    }


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
    ratio_x = explore_target_ratio(x, w)
    ratio_y = explore_target_ratio(y, w)
    print w
    print 1, ratio_x
    print -1, ratio_y
    print '==========================='
    return w, {1: ratio_x, -1: ratio_y}


def add_set_cols(df):
    df[tokens_set1]=df[tokens_q1].apply(lambda s: set(s.split()))
    df[tokens_set2]=df[tokens_q2].apply(lambda s: set(s.split()))

folds_fp=os.path.join(data_folder, 'top_k_freq', 'folds.json')
topKtokens_fp=os.path.join(data_folder, 'top_k_freq', 'tokens.json')

out_of_fold_freq_fp = os.path.join(data_folder, 'top_k_freq', 'out_of_fold_freq.json')
train_freq_fp=os.path.join(data_folder, 'top_k_freq', 'train_freq.json')

out_of_fold_contains_fp= os.path.join(data_folder, 'top_k_freq', 'out_of_fold_contains.json')
test_contains_fp= os.path.join(data_folder, 'top_k_freq', 'test_contains.json')


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
        counter+=1

    json.dump(res, open(folds_fp, 'w+'))


def load_folds():
    return json.load(open(folds_fp))

def create_folds(df):
    folds = load_folds()

    return [(df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']]) for x in range(len(folds))]


def write_top_tokens():
    train_df = load_train_nlp()
    c=get_top_N_tokens(train_df, 1000)
    json.dump(c, open(topKtokens_fp, 'w+'))


def load_top_tokens():
    return json.load(open(topKtokens_fp))


def write_out_of_fold_frequencies():
    df = load_train_nlp()
    folds = create_folds(df)
    top_list = load_top_tokens()

    res={}

    for i, fold in enumerate(folds):
        df=fold[0]#train
        add_set_cols(df)
        freq = create_frequencies(df, top_list)
        res[i]=freq

    json.dump(res, open(out_of_fold_freq_fp, 'w+'))


def write_train_freq():
    df = load_train_nlp()
    top_list = load_top_tokens()
    add_set_cols(df)
    res = create_frequencies(df, top_list)
    json.dump(res, open(train_freq_fp, 'w+'))


def write_out_of_fold_test_contains():
    df = load_train_nlp()
    folds = create_folds(df)
    top_list = load_top_tokens()
    res={}
    for i, fold in enumerate(folds):
        df=fold[1]#test
        add_set_cols(df)
        freq = create_contains_data(df, top_list)
        res[i]=freq

    json.dump(res, open(out_of_fold_contains_fp, 'w+'))


def write_test_conatins():
    df = load_test_nlp()
    top_list = load_top_tokens()
    add_set_cols(df)
    res = create_contains_data(df, top_list)
    json.dump(res, open(test_contains_fp, 'w+'))

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

n_jobs=3
def_split=50

# write_folds()
# write_top_tokens()
# write_out_of_fold_frequencies()
# write_out_of_fold_test_contains()
# write_train_freq()
write_test_conatins()