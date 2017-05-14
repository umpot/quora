import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import json
from ast import literal_eval

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

TAGS = {u'CARDINAL',
        u'DATE',
        u'EVENT',
        u'FAC',
        u'GPE',
        u'LANGUAGE',
        u'LAW',
        u'LOC',
        u'MONEY',
        u'NORP',
        u'ORDINAL',
        u'ORG',
        u'PERCENT',
        u'PERSON',
        u'PRODUCT',
        u'QUANTITY',
        u'TIME',
        u'WORK_OF_ART'}


{u'CARDINAL': 64594,
 u'DATE': 57445,
 u'EVENT': 4758,
 u'FAC': 3773,
 u'GPE': 154751,
 u'LANGUAGE': 8085,
 u'LAW': 1435,
 u'LOC': 12055,
 u'MONEY': 5036,
 u'NORP': 41440,
 u'ORDINAL': 14377,
 u'ORG': 150700,
 u'PERCENT': 2686,
 u'PERSON': 97718,
 u'PRODUCT': 2978,
 u'QUANTITY': 2586,
 u'TIME': 5788,
 u'WORK_OF_ART': 8300}



def prepare_df(df):
    for col in [ner_q1, ner_q2]:
        df[col] = df[col].apply(literal_eval)


def all_tags(df):
    l = list(df[ner_q1])+list(df[ner_q2])
    tags = set()
    for it in l:
        for t in it:
            tags.add(t[0])

    return tags

def all_tags_counts(df):
    l = list(df[ner_q1])+list(df[ner_q2])
    tags = {k:0 for k in TAGS}
    for it in l:
        for t in it:
            tags[t[0]]+=1

    return tags