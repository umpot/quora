import gensim
from nltk import Tree
import textacy.extract as extr
import spacy
import textacy
import ast
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

nlp = spacy.load('en')

data_folder = '../../data/'

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

def print_tree(s):
    doc = nlp(s)
    print [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

def extract_subj_verb(s):
    doc=nlp(s)
    return [x for x in textacy.extract.subject_verb_object_triples(doc)]

def noun_chunks(s):
    s=unicode(s)
    doc=nlp(s)
    return list(extr.noun_chunks(doc))

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


def add_tfidf_features(df, col1, col2, prefix):
    tfidf=create_vectorizer_df(df, [col1, col2])
    t1, t2 = 't1', 't2'
    df[t1] = df[col1].apply(lambda s: apply_vectorizer(s, tfidf))
    df[t2] = df[col2].apply(lambda s: apply_vectorizer(s, tfidf))

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
    df[col]=\
        df.apply(lambda s: get_tf_idf_share(s[t1], s[t2]), axis=1)
    new_cols.append(col)

    col = '{}_tfidf_share_ratio'.format(prefix)
    df[col]=\
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




