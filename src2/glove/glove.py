from time import time

import gensim
import os

from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import numpy as np
import pandas as pd
import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import sys


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

data_folder = '../../../data/'
fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'
lemmas_train_fp = os.path.join(data_folder, 'nlp', 'lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder, 'nlp', 'lemmas_test.csv')


lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
question1, question2 = 'question1', 'question2'
TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'



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
        pd.read_csv(fp_train, index_col='id')
    )


def load_test():
    return fix_nans(
        pd.read_csv(fp_test, index_col='test_id')
    )

def load_train_lemmas():
    df = pd.read_csv(lemmas_train_fp, index_col='id')
    df = df.fillna('')
    def del_pron(s):
        return str(s).replace('-PRON-', '')

    for col in [lemmas_q1, lemmas_q2]:
        df[col] = df[col].apply(del_pron)
    return df


def load_test_lemmas():
    df = pd.read_csv(lemmas_test_fp, index_col='test_id')
    df = df.fillna('')
    def del_pron(s):
        return str(s).replace('-PRON-', '')
    for col in [lemmas_q1, lemmas_q2]:
        df[col] = df[col].apply(del_pron)
    return df


def load_train_for_embeddings():
    return pd.concat([load_train(), load_train_lemmas()], axis=1)

def load_test_for_embeddings():
    return pd.concat([load_test(), load_test_lemmas()], axis=1)
##########################################################################
##########################################################################


glove_wmd_tokens='glove_wmd_tokens'
glove_norm_wmd_tokens='glove_norm_wmd_tokens'

glove_wmd_lemmas='glove_wmd_lemmas'
glove_norm_wmd_lemmas='glove_norm_wmd_lemmas'



counter=0
def wmd(s1, s2, model):
    global counter
    if counter%10000==0:
        print counter

    counter+=1


    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2, norm_model):
    global counter
    if counter%10000==0:
        print counter

    counter+=1


    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s, model):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def process_wmd_one_model(df, model, col1, col2, embed_name, operation, type_of_cols):
    print 'wmd'
    new_col = '{}_{}_{}'.format(embed_name, operation, type_of_cols)
    df[new_col] = df.apply(lambda row: wmd(row[col1], row[col2], model), axis=1)

def write_train(train_df, fp_train):
    del_trash_cols(train_df)
    train_df.to_csv(fp_train, index_label='id')


def write_test(test_df, fp_test):
    del_trash_cols(test_df)
    test_df.to_csv(fp_test, index_label='test_id')

def process_metrics(data, col1, col2, type_of_cols, embed_name, model):
    question1_vectors = np.zeros((data.shape[0], 300))
    error_count = 0

    for i, q in tqdm(enumerate(data[col1].values)):
        question1_vectors[i, :] = sent2vec(q, model)

    question2_vectors  = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data[col2].values)):
        question2_vectors[i, :] = sent2vec(q, model)

    col = 'cosine_distance_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

    col = 'cityblock_distance_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

    col = 'jaccard_distance_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                 np.nan_to_num(question2_vectors))]

    col = 'canberra_distance_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

    col = 'euclidean_distance_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

    col = 'minkowski_distance_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                      np.nan_to_num(question2_vectors))]

    col = 'braycurtis_distance_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]

    col = 'skew_q1vec_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [skew(x) for x in np.nan_to_num(question1_vectors)]

    col = 'skew_q2vec_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [skew(x) for x in np.nan_to_num(question2_vectors)]

    col = 'kur_q1vec_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]

    col = 'kur_q2vec_{}_{}'.format(type_of_cols, embed_name)
    print col
    data[col] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

def del_trash_cols(df):
    for col in [TARGET, question1, question2, lemmas_q1, lemmas_q2, qid1, qid2]:
        if col in df.columns:
            del df[col]


freebase_model_fp='/media/ubik/8cc52b52-b1bd-4eb6-a3ee-1b1a4adbc96c/freebase-vectors-skipgram1000-en.bin'
word2vec_model_fp = os.path.join(data_folder, 'GoogleNews-vectors-negative300.bin')
glove_model_fp = os.path.join(data_folder, 'glove.840B.300d_w2v.txt')
fasttext_model_fp=os.path.join(data_folder, 'wiki.en')
lex_model_fp = os.path.join(data_folder, 'lexvec.commoncrawl.300d.W+C.pos.vectors.gz')

############################################################################
word2vec_train_fp = os.path.join(data_folder, 'embeddings', 'word2vec_train.csv')
word2vec_test_fp = os.path.join(data_folder, 'embeddings', 'word2vec_test.csv')
def load_word2vec():
    model= gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_fp, binary=True)
    return model

def load_norm_word2vec():
    model= gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_fp, binary=True)
    model.init_sims(replace=True) # normalizes vectors

    return model
############################################################################


def load_freebase():
    model= gensim.models.KeyedVectors.load_word2vec_format(freebase_model_fp, binary=True)#, binary=True
    # model = Word2Vec.load_word2vec_format(freebase_model_fp, binary=True)
    return model

############################################################################
glove_train_fp = os.path.join(data_folder, 'embeddings', 'glove_train.csv')
glove_test_fp = os.path.join(data_folder, 'embeddings', 'glove_test.csv')
def load_glove():
    model= gensim.models.KeyedVectors.load_word2vec_format(glove_model_fp)#, binary=True
    # model.init_sims(replace=True) # normalizes vectors
    return model

def load_norm_glove():
    model= gensim.models.KeyedVectors.load_word2vec_format(glove_model_fp)#, binary=True
    model.init_sims(replace=True) # normalizes vectors
    return model

############################################################################
from gensim.models.wrappers import FastText

fasttext_train_fp = os.path.join(data_folder, 'embeddings', 'fasttext_train.csv')
fasttext_test_fp = os.path.join(data_folder, 'embeddings', 'fasttext_test.csv')


def load_fasttext():
    model = FastText.load_fasttext_format(fasttext_model_fp)
    return model

def load_norm_fasttext():
    model = FastText.load_fasttext_format(fasttext_model_fp)
    model.init_sims(replace=True)
    return model
############################################################################
lex_train_fp = os.path.join(data_folder, 'embeddings', 'lex_train.csv')
lex_test_fp = os.path.join(data_folder, 'embeddings', 'lex_test.csv')


def load_lex():
    model= gensim.models.KeyedVectors.load_word2vec_format(f)
    return model

def load_norm_lex():
    model= gensim.models.KeyedVectors.load_word2vec_format(f)
    model.init_sims(replace=True)
    return model
############################################################################
def get_model(name):
    if name=='glove':
        return load_glove()
    elif name == 'word2vec':
        return load_word2vec()
    elif name=='fasttext':
        return load_fasttext()
    elif name == 'lex':
        return load_lex()

def get_norm_model(name):
    if name=='glove':
        return load_norm_glove()
    elif name == 'word2vec':
        return load_norm_word2vec()
    elif name=='fasttext':
        return load_norm_fasttext()
    elif name == 'lex':
        return load_norm_lex()

def get_res_files_names(name):
    if name == 'glove':
        return glove_train_fp, glove_test_fp
    elif name == 'word2vec':
        return word2vec_train_fp, word2vec_test_fp
    elif name=='fasttext':
        return fasttext_train_fp, fasttext_test_fp
    elif name=='lex':
        return lex_train_fp, lex_test_fp

def process_paralell(train_test, embed_name, operation, type_of_cols):
    res_train_fp, res_test_fp = get_res_files_names(embed_name)

    if operation not in ['wmd', 'norm_wmd', 'metrics', 'combine']:
        raise Exception('{} blja!'.format(operation))
    if type_of_cols not in ['tokens', 'lemmas']:
        raise Exception('{} blja!'.format(type_of_cols))

    if train_test not in ['train', 'test']:
        raise Exception('{} blja!'.format(train_test))

    df = load_train_for_embeddings() if train_test == 'train' else load_test_for_embeddings()

    # df=df.head(15000)

    index='id' if train_test=='train' else 'test_id'

    if type_of_cols == 'tokens':
        col1, col2 = question1, question2
    else:
        col1, col2 = lemmas_q1, lemmas_q2

    if operation == 'metrics':

        print 'Loading model...'
        t=time()
        model = get_model(embed_name)
        print 'Loaded!!'
        print 'Time {}'.format(time()-t)

        process_metrics(df, col1, col2, type_of_cols, embed_name, model)
        del_trash_cols(df)
        df.to_csv('blja_{}_{}_metrics.csv'.format(train_test, type_of_cols), index_label=index)

    elif operation=='norm_wmd':

        print 'Loading model...'
        t=time()
        model = get_norm_model(embed_name)
        print 'Loaded!!'
        print 'Time {}'.format(time()-t)

        process_wmd_one_model(df, model, col1, col2, embed_name, operation, type_of_cols)
        del_trash_cols(df)
        df.to_csv('blja_{}_{}_norm_wmd.csv'.format(train_test, type_of_cols), index_label=index)

    elif operation=='wmd':

        t=time()
        print 'Loading model...'
        model = get_model(embed_name)
        print 'Loaded!!'
        print 'Time {}'.format(time()-t)

        process_wmd_one_model(df, model, col1, col2, embed_name, operation, type_of_cols)
        del_trash_cols(df)
        df.to_csv('blja_{}_{}_wmd.csv'.format(train_test, type_of_cols), index_label=index)

    elif operation=='combine':
        if train_test=='train':
            train_files = [
                              'blja_train_{}_metrics.csv'.format(x) for x in ['tokens', 'lemmas']
                          ]+[
                              'blja_{}_{}_wmd.csv'.format('train', x) for x in ['tokens', 'lemmas']
                              ]+[
                              'blja_{}_{}_norm_wmd.csv'.format('train', x) for x in ['tokens', 'lemmas']
                              ]

            dfs = [pd.read_csv(fp, index_col=index) for fp in train_files]
            df = pd.concat(dfs, axis=1)
            df.to_csv(res_train_fp, index_label='id')


        elif train_test=='test':
            test_files = [
                             'blja_test_{}_metrics.csv'.format(x) for x in ['tokens', 'lemmas']
                         ]+[
                             'blja_{}_{}_wmd.csv'.format('test', x) for x in ['tokens', 'lemmas']
                             ]+[
                             'blja_{}_{}_norm_wmd.csv'.format('test', x) for x in ['tokens', 'lemmas']
                             ]

            dfs = [pd.read_csv(fp, index_col=index) for fp in test_files]
            df = pd.concat(dfs, axis=1)
            df.to_csv(res_test_fp, index_label='test_id')


process_paralell(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])



