import json
import os
import re
import csv
import codecs
import traceback

import gensim
import numpy as np
import pandas as pd
import unicodedata

from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'

question1, question2 = 'question1', 'question2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'

data_folder = '../../../data/'
fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

magic_train_fp = os.path.join(data_folder, 'magic', 'magic_train.csv')
magic_test_fp = os.path.join(data_folder, 'magic', 'magic_test.csv')

magic2_train_fp = os.path.join(data_folder, 'magic', 'magic2_train.csv')
magic2_test_fp = os.path.join(data_folder, 'magic', 'magic2_test.csv')

lemmas_train_fp = os.path.join(data_folder, 'nlp', 'lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder, 'nlp', 'lemmas_test.csv')

tokens_train_fp = os.path.join(data_folder, 'nlp', 'tokens_train.csv')
tokens_test_fp = os.path.join(data_folder, 'nlp', 'tokens_test.csv')


folds_fp = os.path.join(data_folder, 'top_k_freq', 'folds.json')

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

word2vec_model_fp = os.path.join(data_folder, 'GoogleNews-vectors-negative300.bin')
glove_model_fp = os.path.join(data_folder, 'glove.840B.300d_w2v.txt')
fasttext_model_fp=os.path.join(data_folder, 'wiki.en')
lex_model_fp = os.path.join(data_folder, 'lexvec.commoncrawl.300d.W+C.pos.vectors.gz')

def load_word2vec():
    model= gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_fp, binary=True)
    return model

def load_lex():
    model= gensim.models.KeyedVectors.load_word2vec_format(lex_model_fp)
    return model

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                  rate_drop_dense)



def split_into_folds(df, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    res=[]
    for big_ind, small_ind in skf.split(np.zeros(len(df)), df[TARGET]):
        res.append((df.loc[big_ind], df.loc[small_ind]))

    return res

def get_dummy_folds(df):
    return split_into_folds(df)


def load_folds():
    return json.load(open(folds_fp))


def create_folds(df):
    folds = load_folds()

    return [
        (df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']])
        for x in range(len(folds))]

def fix_nans(df):
    def blja(s):
        if s != s:
            return ''
        s= unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
        return s

    for col in [question1, question2]:
        df[col] = df[col].apply(blja)

    return df

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

def load_train_tokens():
    df = pd.read_csv(tokens_train_fp, index_col='id')
    df = df.fillna('')
    return df


def load_test_tokens():
    df = pd.read_csv(tokens_test_fp, index_col='test_id')
    df = df.fillna('')
    return df

def load_train():
    df = pd.concat([
        pd.read_csv(fp_train, index_col='id', encoding="utf-8"),
        load_train_tokens(),
        load_train_lemmas(),
        pd.read_csv(magic_train_fp, index_col='id')[['freq_question1', 'freq_question2']],
        pd.read_csv(magic2_train_fp, index_col='id')],
        axis=1
    )

    return fix_nans(df)


def load_test():
    df = pd.concat([
        pd.read_csv(fp_test, index_col='test_id', encoding="utf-8"),
        load_test_tokens(),
        load_test_lemmas(),
        pd.read_csv(magic_test_fp, index_col='test_id')[['freq_question1', 'freq_question2']],
        pd.read_csv(magic2_test_fp, index_col='test_id')],
        axis=1
    )
    return fix_nans(df)

import subprocess
from time import gmtime, strftime, time


def get_time_str():
    return strftime("%Y_%m_%d__%H_%M_%S", gmtime())

def push_to_gs(name, descr):
    with open('descr.txt', 'w+') as f:
        f.writelines([descr])

    if name in os.listdir('.'):
        t=get_time_str()
        subprocess.call(['mv', name, '{}_fold_{}_{}'.format(name, t)])


    script_name = os.path.basename(__file__)
    subprocess.call(['python', '-u', 'compress_and_push_to_gs.py', name, script_name])


def done():
    print '============================'
    print 'DONE!'
    print '============================'

######################################################################################
######################################################################################
######################################################################################
######################################################################################

############################################################3
############################################################3
############################################################3
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

def oversample_df(df, l, random_state):
    df_pos = df[df[TARGET]==1]
    df_neg = df[df[TARGET]==0]

    df_neg_sampl = df_neg.sample(l, random_state=random_state, replace=True)

    df=pd.concat([df_pos, df_neg, df_neg_sampl])
    df = shuffle_df(df, random_state)

    return df

def oversample(train_df, test_df, random_state):
    l_train = int(delta * len(train_df))
    l_test = int(delta * len(test_df))

    return oversample_df(train_df, l_train, random_state), oversample_df(test_df, l_test, random_state)



def oversample_submit(train_df, test_df, random_state=42):
    l_train = int(delta * len(train_df))

    return oversample_df(train_df, l_train, random_state),test_df


def shuffle_df(df, random_state=42):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]

############################################################3
############################################################3
############################################################3
gc_host = '104.197.97.20'
local_host = '10.20.0.144'
user='ubik'
password='nfrf[eqyz'

def write_loss(f_num, type_of_cols, emb_type, remove_stop_words, loss):
    from pymongo import MongoClient
    name='lstm_{}_{}_re_stops_{}'.format(emb_type, type_of_cols, remove_stop_words)

    client = MongoClient(gc_host, 27017)
    client['admin'].authenticate(user, password)
    db = client['lstm_stacking_cv']
    collection = db[name]
    try:
        collection.insert_one({
            'loss': loss,
            'fold':f_num
        })
    except:
        print 'error in mongo'
        traceback.print_exc()
        # raise
        # sleep(20)



############################################################3
############################################################3
############################################################3

#token words, not lower, lemmas, top idf, nouns etc
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

def create_embed_index_word2vec():
    model = load_word2vec()
    word_index = {k:model[k] for k in model.vocab.keys()}

    return word_index

def create_embed_index_lex():
    model = load_word2vec()
    word_index = {k:model[k] for k in model.vocab.keys()}

    return word_index

def create_embed_index_glove():
    embed_index = {}
    f = open(glove_model_fp)
    count = 0
    for line in f:
        if count == 0:
            count = 1
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embed_index[word] = coefs
    f.close()

    print('Found %d word vectors of glove.' % len(embed_index))

    return embed_index

def get_emb_index(emb_type):
    if emb_type=='glove':
        return create_embed_index_glove()
    elif emb_type=='word2vec':
        return create_embed_index_word2vec()
    elif emb_type == 'lex':
        return create_embed_index_lex()
    raise Exception('Unknown emd_type {}'.format(emb_type))



def generate_data_for_lstm(cv_train, cv_test, col1, col2, remove_stops):
    cv_train['texts_1'] = cv_train[col1].apply(lambda s: text_to_wordlist(s, remove_stops))
    cv_train['texts_2'] = cv_train[col2].apply(lambda s: text_to_wordlist(s, remove_stops))
    texts_1=[x for x in cv_train['texts_1']]
    texts_2=[x for x in cv_train['texts_2']]
    train_labels = [x for x in cv_train[TARGET]]

    cv_test['texts_1'] = cv_test[col1].apply(lambda s: text_to_wordlist(s, remove_stops))
    cv_test['texts_2'] = cv_test[col2].apply(lambda s: text_to_wordlist(s, remove_stops))

    test_texts_1 = [x for x in cv_test['texts_1']]
    test_texts_2 = [x for x in cv_test['texts_2']]
    test_ids = [x for x in cv_test.index]
    test_labels = [x for x in cv_test[TARGET]]

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    train_labels = np.array(train_labels)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', train_labels.shape)

    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_ids = np.array(test_ids)

    leaks = cv_train[['q1_q2_intersect', 'freq_question1', 'freq_question2']]
    test_leaks = cv_test[['q1_q2_intersect', 'freq_question1', 'freq_question2']]

    ss = StandardScaler()
    ss.fit(np.vstack((leaks, test_leaks)))
    leaks = ss.transform(leaks)
    test_leaks = ss.transform(test_leaks)

    return data_1, data_2, leaks, \
           test_data_1, test_data_2, test_leaks, \
           train_labels, test_labels, test_ids, \
           word_index


def get_cols(type_of_cols):
    if type_of_cols == 'question':
        return question1, question2
    elif type_of_cols=='lemmas':
        return lemmas_q1, lemmas_q2
    raise Exception('Unknown type_of_cols {}'.format(type_of_cols))





def do_lstm_stacking(f_num, type_of_cols, emb_type, remove_stop_words):
    f_num=int(f_num)
    remove_stop_words = remove_stop_words == 'yes'

    col1, col2 = get_cols(type_of_cols)
    print 'Indexing...'
    embeddings_index = get_emb_index(emb_type)
    print 'Done indexing'

    # update_df = load_train()
    # folds = create_folds(update_df)

    update_df = load_train()
    update_df = update_df.head(5000)
    folds = get_dummy_folds(update_df)


    counter = 0
    for cv_train, cv_test in folds:
        print 'Len trainn {},Len test {}'.format(len(cv_train), len(cv_test))
        if f_num!=counter:
            print 'Skipping_{}'.format(counter)
            counter+=1
            continue
        else:
            counter+=1

        print explore_target_ratio(cv_train)
        print explore_target_ratio(cv_test)
        print '========================================'

        cv_train, cv_test = oversample(cv_train, cv_test, 42)

        print 'Len trainn {},Len test {}'.format(len(cv_train), len(cv_test))

        print explore_target_ratio(cv_train)
        print explore_target_ratio(cv_test)


        data_1, data_2, leaks, \
        test_data_1, test_data_2, test_leaks, \
        train_labels, test_labels, test_ids, word_index = \
            generate_data_for_lstm(cv_train, cv_test, col1, col2, remove_stop_words)

        ########################################
        ## prepare embeddings
        ########################################
        print('Preparing embedding matrix')

        nb_words = min(MAX_NB_WORDS, len(word_index))+1

        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        ########################################
        ## sample train/validation data
        ########################################

        data_1_train = np.vstack((data_1, data_2))
        data_2_train = np.vstack((data_2, data_1))
        leaks_train = np.vstack((leaks, leaks))
        labels_train = np.concatenate((train_labels, train_labels))

        data_1_val = np.vstack((test_data_1, test_data_2))
        data_2_val = np.vstack((test_data_2, test_data_1))
        leaks_val = np.vstack((test_leaks, test_leaks))
        labels_val = np.concatenate((test_labels, test_labels))



        ########################################
        ## define the model structure
        ########################################
        embedding_layer = Embedding(nb_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

        sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer(embedded_sequences_2)

        leaks_input = Input(shape=(leaks.shape[1],))
        leaks_dense = Dense(num_dense/2, activation=act)(leaks_input)

        merged = concatenate([x1, y1, leaks_dense])
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(num_dense, activation=act)(merged)
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)

        ########################################

        ########################################
        ## train the model
        ########################################

        model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], \
                      outputs=preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['acc'])
        #model.summary()
        print(STAMP)

        bst_model_path = STAMP + '.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


        hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train, \
                         validation_data=([data_1_val, data_2_val, leaks_val], labels_val), \
                         epochs=10, batch_size=2048, shuffle=True,callbacks=[model_checkpoint])


        preds = model.predict([test_data_1, test_data_2, test_leaks], batch_size=8192, verbose=1)
        preds += model.predict([test_data_2, test_data_1, test_leaks], batch_size=8192, verbose=1)
        preds /= 2

        cv_test['prob'] = preds
        cv_test['z'] = 1-cv_test['prob']

        loss = log_loss(cv_test[TARGET], cv_test[['z', 'prob']])
        write_loss(f_num, type_of_cols, emb_type, remove_stop_words, loss)

        cv_test = cv_test[~cv_test.index.duplicated(keep='first')]
        cv_test[[TARGET, 'prob']].to_csv('probs.csv', index_label='id')


descr= \
"""

"""

f_num, type_of_cols, emb_type, remove_stop_words = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

name='lstm_{}_{}_re_stops_{}_fold_{}'.format(emb_type, type_of_cols, remove_stop_words, f_num)

do_lstm_stacking(f_num, type_of_cols, emb_type, remove_stop_words)
push_to_gs(name, descr)
done()