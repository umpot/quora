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

TARGET = 'is_duplicate'
qid1, qid2 = 'qid1', 'qid2'

question1, question2 = 'question1', 'question2'
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
stems_q1, stems_q2 = 'stems_q1', 'stems_q2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'
ner_q1, ner_q2 = 'ner_q1', 'ner_q2'
postag_q1, postag_q2 = 'postag_q1', 'postag_q2'

data_folder = '../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

lemmas_train_fp = os.path.join(data_folder, 'nlp', 'lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder, 'nlp', 'lemmas_test.csv')

tokens_train_fp = os.path.join(data_folder, 'nlp', 'tokens_train.csv')
tokens_test_fp = os.path.join(data_folder, 'nlp', 'tokens_test.csv')

postag_train_fp = os.path.join(data_folder, 'nlp', 'postag_train.csv')
postag_test_fp = os.path.join(data_folder, 'nlp', 'postag_test.csv')

ner_train_fp = os.path.join(data_folder, 'nlp', 'ner_train.csv')
ner_test_fp = os.path.join(data_folder, 'nlp', 'ner_test.csv')

stems_train_fp = os.path.join(data_folder, 'nlp', 'stems_train.csv')
stems_test_fp = os.path.join(data_folder, 'nlp', 'stems_test.csv')

tfidf_with_stops_train_fp = os.path.join(data_folder, 'tfidf', 'tokens_with_stop_words_tfidf_train.csv')
tfidf_with_stops_test_fp = os.path.join(data_folder, 'tfidf', 'tokens_with_stop_words_tfidf_test.csv')

magic_train_fp = os.path.join(data_folder, 'magic', 'magic_train.csv')
magic_test_fp = os.path.join(data_folder, 'magic', 'magic_test.csv')

common_words_train_fp = os.path.join(data_folder, 'basic', 'common_words_train.csv')
length_train_fp = os.path.join(data_folder, 'basic', 'lens_train.csv')

common_words_test_fp = os.path.join(data_folder, 'basic', 'common_words_test.csv')
length_test_fp = os.path.join(data_folder, 'basic', 'lens_test.csv')

TRAIN_METRICS_FP = [
    data_folder + 'distances/' + 'train_metrics_bool_lemmas.csv',
    data_folder + 'distances/' + 'train_metrics_bool_stems.csv',
    data_folder + 'distances/' + 'train_metrics_bool_tokens.csv',
    data_folder + 'distances/' + 'train_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/' + 'train_metrics_fuzzy_stems.csv',
    data_folder + 'distances/' + 'train_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/' + 'train_metrics_sequence_lemmas.csv',
    data_folder + 'distances/' + 'train_metrics_sequence_stems.csv',
    data_folder + 'distances/' + 'train_metrics_sequence_tokens.csv'
]

TEST_METRICS_FP = [
    data_folder + 'distances/' + 'test_metrics_bool_lemmas.csv',
    data_folder + 'distances/' + 'test_metrics_bool_stems.csv',
    data_folder + 'distances/' + 'test_metrics_bool_tokens.csv',
    data_folder + 'distances/' + 'test_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/' + 'test_metrics_fuzzy_stems.csv',
    data_folder + 'distances/' + 'test_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/' + 'test_metrics_sequence_lemmas.csv',
    data_folder + 'distances/' + 'test_metrics_sequence_stems.csv',
    data_folder + 'distances/' + 'test_metrics_sequence_tokens.csv'
]


def load_train():
    return pd.read_csv(fp_train, index_col='id')


def load_test():
    return pd.read_csv(fp_test, index_col='test_id')


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
        df[col] = df[col].apply(str)
    return df


def load_test_lemmas():
    df = pd.read_csv(lemmas_test_fp, index_col='test_id')
    df = df.fillna('')
    for col in [lemmas_q1, lemmas_q2]:
        df[col] = df[col].apply(str)
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
    df = pd.read_csv(magic_train_fp, index_col='id')[['freq_question1', 'freq_question2']]
    return df


def load_test_magic():
    df = pd.read_csv(magic_test_fp, index_col='test_id')[['freq_question1', 'freq_question2']]
    return df


def load_train_stems():
    df = pd.read_csv(stems_train_fp, index_col='id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    for col in [stems_q1, stems_q2]:
        df[col] = df[col].apply(str)
    return df


def load_test_stems():
    df = pd.read_csv(stems_test_fp, index_col='test_id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    for col in [stems_q1, stems_q2]:
        df[col] = df[col].apply(str)
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


# df = load_train_all()
######################################################################################
######################################################################################
######################################################################################
######################################################################################

# WH

wh_fp_train = os.path.join(data_folder, 'wh', 'wh_train.csv')
wh_fp_test = os.path.join(data_folder, 'wh', 'wh_test.csv')


def load_wh_train():
    df = pd.read_csv(wh_fp_train, index_col='id')
    return df


def load_wh_test():
    df = pd.read_csv(wh_fp_test, index_col='test_id')
    return df


######################################################################################
######################################################################################
######################################################################################
######################################################################################
upper_stop_words =\
['What',
 'How',
 'Why',
 'Is',
 'Which',
 'Can',
 'I',
 'Who',
 'Do',
 'Where',
 'If',
 "What's",
 'Does',
 'Are',
 'Should',
 'When',
 'Will',
 'In',
 'My',
 "I'm",
 'Did',
 'Would',
 'Has',
 'Have',
 'Was',
 'Could',
 'As',
 'The',
 'A',
 'On',
 'For',
 'After',
 'Am',
 'At',
 'Were',
 'From',
 'With',
 "I've",
 'To',
 'What\xe2\x80\x99s',
 'Any',
 # 'Daniel',
 'We',
 'Since',
 "Who's",
 'It',
 'There',
 'According',
 'Now',
 # 'Indian',
 'Given',
 'what',
 'Difference',
 'While',
 'You',
 'Some',
 # 'India:',
 'During',
 'Whats',
 # 'Harvard',
 'People',
 'Between',
 'By',
 'Most',
 "How's",
 'So',
 'Whom',
 # 'Donald',
 'All',
 # 'Instagram',
 'This',
 'Being',
 'Someone',
 'Hi',
 'Two',
 'One',
 'I\xe2\x80\x99m',
 'An',
 'how',
 'Dating',
 # 'Quora:',
 'Under',
 'Besides',
 'Whose',
 "It's",
 'Life',
 'Whenever',
 # 'Android',
 'Suppose',
 'World',
 "Where's",
 'Every',
 'Whether',
 'Psychology',
 'Time',
 'Explain',
 'Without',
 'Best',
 # 'Star',
 # 'English'
 ]


def first_upper_counts(df):
    l = list(df[question1]) + list(df[question2])
    l = [str(x).split()[0] for x in l]
    m = {}
    for k in l:
        if k in m:
            m[k] += 1
        else:
            m[k] = 1

    m = [(k, v) for k, v in m.iteritems()]
    m.sort(key=lambda s: s[1], reverse=True)

    return m

def get_uppers(s):
    s=str(s).split()
    s=filter(lambda s: s[0].isupper(),s)
    s=filter(lambda s: s not in upper_stop_words, s)
    return ' '.join(s)

upper_q1='upper_q1'
upper_q2='upper_q2'


def add_upper_columns(df):
    df[upper_q1]=df[tokens_q1].apply(get_uppers)
    df[upper_q2]=df[tokens_q2].apply(get_uppers)
