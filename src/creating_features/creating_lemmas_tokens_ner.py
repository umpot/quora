import spacy
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
import seaborn as sns
import re
import os

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

question1, question2 = 'question1', 'question2'
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
stems_q1, stems_q2 = 'stems_q1', 'stems_q2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'
postag_q1, postag_q2 = 'postag_q1', 'postag_q2'
ner_q1, ner_q2 = 'ner_q1', 'ner_q2'


data_folder = '../../../data/'

fp_train = os.path.join(data_folder,'train.csv')
fp_test = os.path.join(data_folder,'test.csv')

'tokens_train.csv', 'lemmas_train.csv', 'postag_train.csv', 'ner_train.csv'

lemmas_train_fp = os.path.join(data_folder,'nlp','lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder,'nlp','lemmas_test.csv')

tokens_train_fp = os.path.join(data_folder,'nlp','tokens_train.csv')
tokens_test_fp = os.path.join(data_folder,'nlp','tokens_test.csv')

postag_train_fp = os.path.join(data_folder,'nlp','postag_train.csv')
postag_test_fp = os.path.join(data_folder,'nlp','postag_test.csv')

ner_train_fp = os.path.join(data_folder,'nlp','ner_train.csv')
ner_test_fp = os.path.join(data_folder,'nlp','ner_test.csv')

def load_train():
    return pd.read_csv(fp_train, index_col='id')

def load_test():
    return pd.read_csv(fp_test, index_col='test_id')

nlp = spacy.load('en', parser=False)


POS_TAGS1_TO_SKIP=['SYM','SP', '-RRB-', 'LS', 'HYPH', ':', '-LRB-', '.', ',', '$', "'",'``', 'NFP']

counter=0
def process_df(df, tokens_fp, lemmas_fp, postag_fp, ner_fp, index_col, data_folder):
    def get_nlp(s):
        global counter
        counter+=1
        if counter%1000 == 0:
            print counter

        return nlp(str(s).decode("utf-8"))

    df['nlp_q1'] = df[question1].apply(get_nlp)
    df['nlp_q2'] = df[question2].apply(get_nlp)

    def get_ner(doc):
        return [(e.label_, str(e)) for e in doc.ents]

    df[ner_q1] = df['nlp_q1'].apply(get_ner)
    df[ner_q2] = df['nlp_q2'].apply(get_ner)

    fp = os.path.join(data_folder, ner_fp)
    new_cols = [ner_q1, ner_q2]
    df[new_cols].to_csv(fp, index_label=index_col)
    for col in new_cols:
        del df[col]

    def get_postag(doc):
        return [(word.text, word.lemma_,word.tag_) for word in doc if word.tag_ not in POS_TAGS1_TO_SKIP]

    df['nlp_q1'] = df['nlp_q1'].apply(get_postag)
    df['nlp_q2'] = df['nlp_q2'].apply(get_postag)
    df.rename(columns={'nlp_q1':'postag_q1', 'nlp_q2':'postag_q2'}, inplace=True)

    fp = os.path.join(data_folder, postag_fp)
    new_cols = ['postag_q1', 'postag_q2']
    df[new_cols].to_csv(fp, index_label=index_col)

    def get_tokens(a):
        return ' '.join([x[0] for x in a])

    def get_lemmas(a):
        return ' '.join([x[1] for x in a])

    df[tokens_q1]=df['postag_q1'].apply(get_tokens)
    df[tokens_q2]=df['postag_q2'].apply(get_tokens)

    fp = os.path.join(data_folder, tokens_fp)
    new_cols = [tokens_q1, tokens_q2]
    df[new_cols].to_csv(fp, index_label=index_col)
    for col in new_cols:
        del df[col]

    df[lemmas_q1]=df['postag_q1'].apply(get_lemmas)
    df[lemmas_q2]=df['postag_q2'].apply(get_lemmas)

    fp = os.path.join(data_folder, lemmas_fp)
    new_cols = [lemmas_q1, lemmas_q2]
    df[new_cols].to_csv(fp, index_label=index_col)
    for col in new_cols:
        del df[col]



def write_train():
    df = load_train()
    folder = os.path.join(data_folder, 'nlp')
    tokens_fp, lemmas_fp, postag_fp, ner_fp = \
    'tokens_train.csv', 'lemmas_train.csv', 'postag_train.csv', 'ner_train.csv'
    process_df(df, tokens_fp, lemmas_fp, postag_fp, ner_fp, 'id', folder)


def write_test():
    df = load_test()
    folder = os.path.join(data_folder, 'nlp')
    tokens_fp, lemmas_fp, postag_fp, ner_fp = \
        'tokens_test.csv', 'lemmas_test.csv', 'postag_test.csv', 'ner_test.csv'
    process_df(df, tokens_fp, lemmas_fp, postag_fp, ner_fp, 'test_id', folder)


write_train()
