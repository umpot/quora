import gensim
from nltk import Tree
import spacy
import ast
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os


import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
from dask.dataframe import from_pandas
import dask
dask.set_options(get=dask.multiprocessing.get)
import sys
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
ner_q1, ner_q2='ner_q1', 'ner_q2'
postag_q1, postag_q2='postag_q1', 'postag_q2'

data_folder = '../../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

folds_fp=os.path.join(data_folder, 'top_k_freq', 'folds.json')


def load_folds():
    return json.load(open(folds_fp))

def create_folds(df):
    folds = load_folds()

    return [(df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']]) for x in range(len(folds))]

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
    return {
        'pos':1.0*len(df[df[TARGET]==1])/len(df),
        'neg':1.0*len(df[df[TARGET]==0])/len(df)
    }



# df = load_train_all()
######################################################################################
######################################################################################
######################################################################################
######################################################################################

nlp = spacy.load('en')


vb_q1, vb_q2 = 'vb_q1', 'vb_q2'
nn_q1, nn_q2 = 'nn_q1', 'nn_q2'
no_stop_verbs_lemms_q1, no_stop_verbs_lemms_q2 = 'no_stop_verbs_lemms_q1', 'no_stop_verbs_lemms_q2'
nouns_lemmas_q1, nouns_lemmas_q2 = 'nouns_lemmas_q1', 'nouns_lemmas_q2'

adj_q1, adj_q2='adj_q1', 'adj_q2'
adv_q1, adv_q2='adv_q1', 'adv_q2'
adv_adj_q1, adv_adj_q2='adv_adj_q1', 'adv_adj_q2'

postag_q1, postag_q2='postag_q1', 'postag_q2'
TARGET = 'is_duplicate'

adj={'JJ', 'JJR', 'JJS'}
adverbs={'RBS', 'RBR', 'RB'}
verbs={'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB'}

adv_adj = {'RBS', 'RBR', 'RB','JJ', 'JJR', 'JJS'}

verb_stops={'be', 'do'}

nn={'NNPS', 'NNP', 'NNS', 'NN'}


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_




def materialize_cols(df, cols=(postag_q1, postag_q2)):
    for col in cols:
        df[col]=df[col].apply(ast.literal_eval)

def filter_in_set(l,  sset):
    return filter(lambda x: x[2] in sset, l)


def explore_verbs_no_stops(df):
    return df[[TARGET, no_stop_verbs_lemms_q1, no_stop_verbs_lemms_q2]]

def explore_nouns(df):
    return df[[TARGET, nouns_lemmas_q1, nouns_lemmas_q2]]

def explore_adjectives(df):
    return df[[TARGET, adj_q1, adj_q2]]

def explore_adverbs(df):
    return df[[TARGET, adv_q1, adv_q2]]

def explore_adv_adj(df):
    return df[[TARGET, adv_adj_q1, adv_adj_q2]]

def add_verbs_cols(df):
    def no_stops(s):
        return ' '.join([x[1] for x in s if x[1] not in verb_stops])

    df[vb_q1] = df[postag_q1].apply(lambda s: filter_in_set(s, verbs))
    df[vb_q2] = df[postag_q2].apply(lambda s: filter_in_set(s, verbs))
    df[no_stop_verbs_lemms_q1]=df[vb_q1].apply(no_stops)
    df[no_stop_verbs_lemms_q2]=df[vb_q2].apply(no_stops)

def flat_lemmas(s):
    return ' '.join([x[1] for x in s if x[1]])

def flat_list(s):
    res=[]
    for x in s:
        res+=x.split()

    return res

def add_nouns_cols(df):
    df[nn_q1] = df[postag_q1].apply(lambda s: filter_in_set(s, nn))
    df[nn_q2] = df[postag_q2].apply(lambda s: filter_in_set(s, nn))
    df[nouns_lemmas_q1]=df[nn_q1].apply(flat_lemmas)
    df[nouns_lemmas_q2]=df[nn_q2].apply(flat_lemmas)


def add_adj_cols(df):
    df[adj_q1] = df[postag_q1].apply(lambda s: filter_in_set(s, adj))
    df[adj_q2] = df[postag_q2].apply(lambda s: filter_in_set(s, adj))
    df[adj_q1]=df[adj_q1].apply(flat_lemmas)
    df[adj_q2]=df[adj_q2].apply(flat_lemmas)

def add_adv_cols(df):
    df[adv_q1] = df[postag_q1].apply(lambda s: filter_in_set(s, adverbs))
    df[adv_q2] = df[postag_q2].apply(lambda s: filter_in_set(s, adverbs))
    df[adv_q1]=df[adv_q1].apply(flat_lemmas)
    df[adv_q2]=df[adv_q2].apply(flat_lemmas)


def add_adv_adj_cols(df):
    df[adv_adj_q1] = df[postag_q1].apply(lambda s: filter_in_set(s, adv_adj))
    df[adv_adj_q2] = df[postag_q2].apply(lambda s: filter_in_set(s, adv_adj))
    df[adv_adj_q1]=df[adv_adj_q1].apply(flat_lemmas)
    df[adv_adj_q2]=df[adv_adj_q2].apply(flat_lemmas)



def get_verbs_counter(df):
    return Counter(flat_list(df[no_stop_verbs_lemms_q1])+flat_list(df[no_stop_verbs_lemms_q2]))

def get_tf_idf_share_ratio(t1, t2):
    # t1=tfidf.transform([t1])
    # t2=tfidf.transform([t2])
    if t1 is None and t2 is None:
        return None

    if t1 is None or t2 is None:
        return 0

    s = t1+t2
    diff = (s-np.abs(t1-t2))/2

    s=np.sum(s)
    s=1 if s==0 else s

    diff=np.sum(diff)

    return diff/s




def get_tf_idf_share(t1, t2):
    # t1=tfidf.transform([t1])
    # t2=tfidf.transform([t2])
    if t1 is None and t2 is None:
        return None

    if t1 is None or t2 is None:
        return 0


    s = t1+t2
    diff = (s-np.abs(t1-t2))/2

    return np.sum(diff)

def apply_vectorizer(s, tfidf):
    if len(s)==0:
        return None
    return tfidf.transform([s])

def create_vectorizer_df(df, cols):
    l=sum([df[col].tolist() for col in cols],[])
    tfidf = TfidfVectorizer()
    tfidf.fit(l)

    return tfidf




def wmd(s1, s2, model):
    stop_words={}

    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()

    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]

    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2, norm_model):
    stop_words={}

    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()

    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]

    return norm_model.wmdistance(s1, s2)


def sent2vec(s, model):
    stop_words={}
    words = s.split()
    words = [w for w in words if not w in stop_words]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


fp_model = os.path.join(data_folder, 'GoogleNews-vectors-negative300.bin')


def load_word2vec():
    model= gensim.models.KeyedVectors.load_word2vec_format(fp_model, binary=True)

    norm_model= gensim.models.KeyedVectors.load_word2vec_format(fp_model, binary=True)
    norm_model.init_sims(replace=True) # normalizes vectors

    return model, norm_model

    # model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    # data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
    #
    #
    # norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    # norm_model.init_sims(replace=True)
    # data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

# inf=float('inf')
# def wmd(x,y, model):
#     res = model.wmdistance(x, y)
#     if res==inf:
#         return None
#     return res

def common_num(a, b):
    a=set(a.split())
    b=set(b.split())

    if len(a)==0 and len(b)==0:
        return None
    return len(a.intersection(b))

def common_ratio(a, b):
    a=set(a.split())
    b=set(b.split())

    if len(a)==0 and len(b)==0:
        return None
    acp=set(a)
    acp.update(b)
    return 1.0*len(a.intersection(b))/len(acp)


def add_tfidf_features(df, col1, col2, tfidf, prefix):
    t1, t2 = 't1', 't2'
    df[t1] = from_pandas(df[col1], npartitions=npartitions).apply(lambda s: apply_vectorizer(s, tfidf)).compute()
    df[t2] = from_pandas(df[col2], npartitions=npartitions).apply(lambda s: apply_vectorizer(s, tfidf)).compute()

    new_cols = []

    col = '{}_num_q1'.format(prefix)
    df[col]=df[col1].apply(lambda s: len(s.split()))
    new_cols.append(col)

    col = '{}_num_q2'.format(prefix)
    df[col]=df[col2].apply(lambda s: len(s.split()))
    new_cols.append(col)

    col = '{}_common_num'.format(prefix)
    df[col]=df.apply(lambda s: common_num(s[col1], s[col2]), axis=1)
    new_cols.append(col)

    col = '{}_common_ratio'.format(prefix)
    df[col]=df.apply(lambda s: common_ratio(s[col1], s[col2]), axis=1)
    new_cols.append(col)

    col = '{}_tfidf_share'.format(prefix)
    df[col]= \
        df.apply(lambda s: get_tf_idf_share(s[t1], s[t2]), axis=1)
    new_cols.append(col)

    col = '{}_tfidf_share_ratio'.format(prefix)
    df[col]= \
        df.apply(lambda s: get_tf_idf_share_ratio(s[t1], s[t2]), axis=1)
    new_cols.append(col)

    return new_cols

def add_wmd_features(df, col1, col2, model, norm_model, prefix):
    new_cols = []

    col = '{}_wmd'.format(prefix)
    df[col] = df.apply(lambda s: wmd(s[col1], s[col2], model), axis=1)
    new_cols.append(col)

    col = '{}_norm_wmd'.format(prefix)
    df[col] = df.apply(lambda s: norm_wmd(s[col1], s[col2], norm_model), axis=1)
    new_cols.append(col)

    return new_cols


train_pos_metrics_fp=os.path.join(data_folder, 'pos_metrics', 'train_pos_metrics.csv')
test_pos_metrics_fp=os.path.join(data_folder, 'pos_metrics', 'test_pos_metrics.csv')

npartitions=20
def write_metrics_on_POS_features():
    train_df, test_df = load_train_nlp(), load_test_nlp()
    # train_df, test_df = train_df.head(1000), test_df.head(1000)
    for df in [train_df, test_df]:
        materialize_cols(df, [postag_q1, postag_q2])
        print 'materialization'
        add_nouns_cols(df)
        print 'adding nouns'
        add_verbs_cols(df)
        print 'adding verbs'

    new_cols=[]

    prefix='nouns'
    cols=[nouns_lemmas_q1, nouns_lemmas_q2]

    tfidf=create_vectorizer_df(train_df, cols)
    new_cols += add_tfidf_features(train_df, cols[0], cols[1], tfidf, prefix)
    # add_tfidf_features(test_df, cols[0], cols[1], tfidf, prefix)
    print 'tfidf nouns'

    prefix='verbs'
    cols=[no_stop_verbs_lemms_q1, no_stop_verbs_lemms_q2]

    tfidf=create_vectorizer_df(train_df, cols)
    new_cols +=add_tfidf_features(train_df, cols[0], cols[1], tfidf, prefix)
    # add_tfidf_features(test_df, cols[0], cols[1], tfidf, prefix)
    print 'tfidf verbs'


    model, norm_model = load_word2vec()
    print 'loaded word2vec'

    prefix='nouns'
    cols=[nouns_lemmas_q1, nouns_lemmas_q2]

    new_cols+=add_wmd_features(train_df, cols[0], cols[1], model, norm_model, prefix)
    # add_wmd_features(test_df, cols[0], cols[1], model, norm_model, prefix)
    print 'wmd nouns'

    prefix='verbs'
    cols=[no_stop_verbs_lemms_q1, no_stop_verbs_lemms_q2]

    new_cols+=add_wmd_features(train_df, cols[0], cols[1], model, norm_model, prefix)
    # add_wmd_features(test_df, cols[0], cols[1], model, norm_model, prefix)
    print 'wmd verbs'

    train_df[new_cols].to_csv(train_pos_metrics_fp, index_label='id')
    # test_df[new_cols].to_csv(test_pos_metrics_fp, index_label='test_id')



write_metrics_on_POS_features()
# def get_verbs(s):
#     doc = nlp(unicode(s))

