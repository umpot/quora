import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys

import spacy
from sklearn.model_selection import StratifiedKFold

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
ner_q1, ner_q2 = 'ner_q1', 'ner_q2'
postag_q1, postag_q2 = 'postag_q1', 'postag_q2'

data_folder = '../../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

lemmas_train_fp = os.path.join(data_folder, 'nlp', 'lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder, 'nlp', 'lemmas_test.csv')

tokens_train_fp = os.path.join(data_folder, 'nlp', 'tokens_train.csv')
tokens_test_fp = os.path.join(data_folder, 'nlp', 'tokens_test.csv')

postag_train_fp = os.path.join(data_folder, 'nlp', 'postag_train.csv')
postag_test_fp = os.path.join(data_folder, 'nlp', 'postag_test.csv')


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
        pd.read_csv(fp_train, index_col='id', encoding="utf-8")
    )


def load_test():
    return fix_nans(
        pd.read_csv(fp_test, index_col='test_id')
    )
''

def load_train_nlp():
    return pd.concat([
        load_train(),
        load_train_postag(),
        load_train_lemmas(),
        load_train_tokens(),
    ], axis=1)


def load_test_nlp():
    return pd.concat([
        load_test(),
        load_test_postag(),
        load_test_lemmas(),
        load_test_tokens(),
    ], axis=1)


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

def explore_target_ratio(df):
    return {
        'pos': 1.0 * len(df[df[TARGET] == 1]) / len(df),
        'neg': 1.0 * len(df[df[TARGET] == 0]) / len(df)
    }

##################################################################################
##################################################################################
# nlp = spacy.load('en')
#
#
# vb_q1, vb_q2 = 'vb_q1', 'vb_q2'
# nn_q1, nn_q2 = 'nn_q1', 'nn_q2'
# no_stop_verbs_lemms_q1, no_stop_verbs_lemms_q2 = 'no_stop_verbs_lemms_q1', 'no_stop_verbs_lemms_q2'
# nouns_lemmas_q1, nouns_lemmas_q2 = 'nouns_lemmas_q1', 'nouns_lemmas_q2'
#
# adj_q1, adj_q2='adj_q1', 'adj_q2'
# adv_q1, adv_q2='adv_q1', 'adv_q2'
# adv_adj_q1, adv_adj_q2='adv_adj_q1', 'adv_adj_q2'
#
# postag_q1, postag_q2='postag_q1', 'postag_q2'
# TARGET = 'is_duplicate'
#
# adj={'JJ', 'JJR', 'JJS'}
# adverbs={'RBS', 'RBR', 'RB'}
# verbs={'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB'}
#
# adv_adj = {'RBS', 'RBR', 'RB','JJ', 'JJR', 'JJS'}
#
# verb_stops={'be', 'do'}
#
# nn={'NNPS', 'NNP', 'NNS', 'NN'}

##################################################################################
##################################################################################
# aux
# be (am, are, is, was, were, being, been),
# can, could,\
# do (does, did),\
# have (has, had, having),\
# may, might, must, need, ought, shall, should, \
# will,\
# would

preposition=[
'aboard       '.strip(),
'about        '.strip(),
'above        '.strip(),
'across       '.strip(),
'after        '.strip(),
'against      '.strip(),
'along        '.strip(),
'amid         '.strip(),
'among        '.strip(),
'anti         '.strip(),
'around       '.strip(),
'as           '.strip(),
'at           '.strip(),
'before       '.strip(),
'behind       '.strip(),
'below        '.strip(),
'beneath      '.strip(),
'beside       '.strip(),
'besides      '.strip(),
'between      '.strip(),
'beyond       '.strip(),
'but          '.strip(),
'by           '.strip(),
'concerning   '.strip(),
'considering  '.strip(),
'despite      '.strip(),
'down         '.strip(),
'during       '.strip(),
'except       '.strip(),
'excepting    '.strip(),
'excluding    '.strip(),
'following    '.strip(),
'for          '.strip(),
'from         '.strip(),
'in           '.strip(),
'inside       '.strip(),
'into         '.strip(),
'like         '.strip(),
'minus        '.strip(),
'near         '.strip(),
'of           '.strip(),
'off          '.strip(),
'on           '.strip(),
'onto         '.strip(),
'opposite     '.strip(),
'outside      '.strip(),
'over         '.strip(),
'past         '.strip(),
'per          '.strip(),
'plus         '.strip(),
'regarding    '.strip(),
'round        '.strip(),
'save         '.strip(),
'since        '.strip(),
'than         '.strip(),
'through      '.strip(),
'to           '.strip(),
'toward       '.strip(),
'towards      '.strip(),
'under        '.strip(),
'underneath   '.strip(),
'unlike       '.strip(),
'until        '.strip(),
'up           '.strip(),
'upon         '.strip(),
'versus       '.strip(),
'via          '.strip(),
'with         '.strip(),
'within       '.strip(),
'without      '.strip()
]



S_preposition=set(preposition)

preposition1, preposition2='preposition1', 'preposition2'
preposition_pair = 'preposition_pair'

preposition_pair_target_freq = 'preposition_pair_target_freq'

def get_preposition_list(s):
    res=[]
    for t in s.lower().split():
        if t in S_preposition:
            res.append(t)

    return ' '.join(res)

def explore_preposition():
    df = pd.concat([
        load_train(),
        load_train_tokens()
    ], axis=1)

    create_preposition_features_df(df)
    create_preposition_pair_feature(df)
    return df[[TARGET, preposition1, preposition2, question1, question2]]

def explore_preposition_pair():
    df = explore_preposition()
    create_preposition_pair_feature(df)
    bl = df.groupby(preposition_pair)[TARGET].agg({'count': 'count', 'freq': 'mean'})
    bl.sort_values('count', ascending=False, inplace=True)

    return bl

def create_preposition_features_df(df):
    df[preposition1] = df[tokens_q1].apply(get_preposition_list)
    df[preposition2] = df[tokens_q2].apply(get_preposition_list)

def create_preposition_pair_feature(df):
    df[preposition_pair] = df[preposition1]+'##'+df[preposition2]


preposition_pairs_50_train_fp = os.path.join(data_folder, 'aux_pron', 'preposition_pairs_50_train.csv')
preposition_pairs_50_test_fp = os.path.join(data_folder, 'aux_pron', 'preposition_pairs_50_test.csv')

def write_preposition_pairs_features():
    train_df, test_df = pd.concat([load_train_tokens(), load_train()], axis=1), load_test_tokens()

    # train_df, test_df = train_df.head(5000), test_df.head(5000)

    create_preposition_features_df(train_df)
    create_preposition_features_df(test_df)

    create_preposition_pair_feature(train_df)
    create_preposition_pair_feature(test_df)

    bl = train_df.groupby(preposition_pair)[TARGET].count().sort_values(ascending=False)[:50]
    pairs_list = list(bl.index)
    pairs_set = set(pairs_list)

    bl = train_df.groupby(preposition_pair)[TARGET].mean()
    freq = {x:bl.loc[x] for x in pairs_set}
    # print freq

    def get_preposition_pair_freq(pair):
        return freq.get(pair, None)

    new_cols=[preposition_pair_target_freq]
    is_train=True
    for df in [train_df, test_df]:
        df[preposition_pair_target_freq] = df[preposition_pair].apply(get_preposition_pair_freq)
        for s in pairs_list:
            col = 'preposition_p_{}'.format(s)
            print col, is_train
            if is_train:
                new_cols.append(col)
            df[col]= df[preposition_pair].apply(lambda x: 1 if x == s else 0)

        is_train=False



    train_df[new_cols].to_csv(preposition_pairs_50_train_fp, index_label='id')
    test_df[new_cols].to_csv(preposition_pairs_50_test_fp, index_label='test_id')

# write_preposition_pairs_features()

