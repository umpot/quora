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
Demonstrative_Pronouns=['this', 'that', 'these', 'those']
Interrogative_Pronouns=[ 'who', 'whom', 'which', 'what', 'whose', 'whoever', 'whatever', 'whichever', 'whomever']
Relative_Pronouns=['who', 'whom', 'whose', 'which', 'that', 'what', 'whatever', 'whoever', 'whomever', 'whichever']

Subjective_Pronouns=['i', 'you', 'he', 'she', 'it', 'we', 'they']
Objective_Pronouns=['me', 'him', 'her', 'us','it', 'them']
Possessive_Pronouns=['mine', 'yours', 'his', 'hers', 'ours', 'theirs']
Reflexive_Pronouns=['myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves']

pronoun=Subjective_Pronouns+Objective_Pronouns+Possessive_Pronouns+Reflexive_Pronouns

Indefinite_Pronouns=[
    'anything', 'everybody', 'another', 'each', 'few', 'many', 'none', 'some',
    'all', 'any', 'anybody', 'anyone', 'everyone', 'everything', 'no one',
    'nobody', 'nothing', 'none', 'other', 'others', 'several',
    'somebody', 'someone', 'something', 'most', 'enough',
    'little', 'more', 'both', 'either', 'neither', 'one', 'much', 'such'
]

S_pronoun=set(pronoun)

pronoun1, pronoun2='pronoun1', 'pronoun2'
pronoun_pair = 'pronoun_pair'

pronoun_pair_target_freq = 'pronoun_pair_target_freq'

def get_pronoun_list(s):
    res=[]
    for t in s.lower().split():
        if t in S_pronoun:
            res.append(t)

    return ' '.join(res)

def explore_pronoun():
    df = pd.concat([
        load_train(),
        load_train_tokens()
    ], axis=1)

    create_pronoun_features_df(df)
    create_pronoun_pair_feature(df)
    return df[[TARGET, pronoun1, pronoun2, question1, question2]]

def explore_pronoun_pair():
    df = explore_pronoun()
    create_pronoun_pair_feature(df)
    bl = df.groupby(pronoun_pair)[TARGET].agg({'count': 'count', 'freq': 'mean'})
    bl.sort_values('count', ascending=False, inplace=True)

    return bl

def create_pronoun_features_df(df):
    df[pronoun1] = df[tokens_q1].apply(get_pronoun_list)
    df[pronoun2] = df[tokens_q2].apply(get_pronoun_list)

def create_pronoun_pair_feature(df):
    df[pronoun_pair] = df[pronoun1]+'##'+df[pronoun2]


pronoun_pairs_50_train_fp = os.path.join(data_folder, 'aux_pron', 'pronoun_pairs_50_train.csv')
pronoun_pairs_50_test_fp = os.path.join(data_folder, 'aux_pron', 'pronoun_pairs_50_test.csv')

def write_pronoun_pairs_features():
    train_df, test_df = pd.concat([load_train_tokens(), load_train()], axis=1), load_test_tokens()

    # train_df, test_df = train_df.head(5000), test_df.head(5000)

    create_pronoun_features_df(train_df)
    create_pronoun_features_df(test_df)

    create_pronoun_pair_feature(train_df)
    create_pronoun_pair_feature(test_df)

    bl = train_df.groupby(pronoun_pair)[TARGET].count().sort_values(ascending=False)[:50]
    pairs_list = list(bl.index)
    pairs_set = set(pairs_list)

    bl = train_df.groupby(pronoun_pair)[TARGET].mean()
    freq = {x:bl.loc[x] for x in pairs_set}
    # print freq

    def get_pronoun_pair_freq(pair):
        return freq.get(pair, None)

    new_cols=[pronoun_pair_target_freq]
    is_train=True
    for df in [train_df, test_df]:
        df[pronoun_pair_target_freq] = df[pronoun_pair].apply(get_pronoun_pair_freq)
        for s in pairs_list:
            col = 'pronoun_p_{}'.format(s)
            print col, is_train
            if is_train:
                new_cols.append(col)
            df[col]= df[pronoun_pair].apply(lambda x: 1 if x == s else 0)

        is_train=False



    train_df[new_cols].to_csv(pronoun_pairs_50_train_fp, index_label='id')
    test_df[new_cols].to_csv(pronoun_pairs_50_test_fp, index_label='test_id')

write_pronoun_pairs_features()

