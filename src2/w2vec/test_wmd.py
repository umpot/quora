import gensim
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import numpy as np

data_folder = '../../data/'

fp_model = os.path.join(data_folder, 'GoogleNews-vectors-negative300.bin')

def load_word2vec():
    model= gensim.models.KeyedVectors.load_word2vec_format(fp_model, binary=True)
    model.init_sims(replace=True) # normalizes vectors

    return model

def wmd(s1, s2, model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2, norm_model):
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