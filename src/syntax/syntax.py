from nltk import Tree
import textacy.extract as extr
import spacy
import textacy
import ast
nlp = spacy.load('en')

vb_q1, vb_q2 = 'vb_q1', 'vb_q2'
no_stop_verbs_lemms_q1, no_stop_verbs_lemms_q2 = 'no_stop_verbs_lemms_q1', 'no_stop_verbs_lemms_q2'
postag_q1, postag_q2='postag_q1', 'postag_q2'
TARGET = 'is_duplicate'

adj={'JJ', 'JJR', 'JJS'}
adverbs={'RBS', 'RBR', 'RB'}
verbs={'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB'}

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

def materialize_cols(df, cols):
    for col in cols:
        df[col]=df[col].apply(ast.literal_eval)

def filter_in_set(l,  sset):
    return filter(lambda x: x[2] in sset, l)


def explore_verbs_no_stops(df):
    return df[[TARGET, no_stop_verbs_lemms_q1, no_stop_verbs_lemms_q2]]

def add_verbs_cols(df):
    def no_stops(s):
        return ' '.join([x[1] for x in s if x[1] not in verb_stops])

    # df[vb_q1] = df[postag_q1].apply(lambda s: filter_in_set(s, verbs))
    # df[vb_q2] = df[postag_q2].apply(lambda s: filter_in_set(s, verbs))
    df[no_stop_verbs_lemms_q1]=df[vb_q1].apply(no_stops)
    df[no_stop_verbs_lemms_q2]=df[vb_q2].apply(no_stops)

# def get_verbs(s):
#     doc = nlp(unicode(s))

