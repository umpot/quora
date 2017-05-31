import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

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
postag_q1, postag_q2='postag_q1', 'postag_q2'

data_folder = '../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

lemmas_train_fp = os.path.join(data_folder,'nlp','lemmas_train.csv')
lemmas_test_fp = os.path.join(data_folder,'nlp','lemmas_test.csv')

tokens_train_fp = os.path.join(data_folder,'nlp','tokens_train.csv')
tokens_test_fp = os.path.join(data_folder,'nlp','tokens_test.csv')

postag_train_fp = os.path.join(data_folder,'nlp','postag_train.csv')
postag_test_fp = os.path.join(data_folder,'nlp','postag_test.csv')

ner_train_fp = os.path.join(data_folder,'nlp','ner_train.csv')
ner_test_fp = os.path.join(data_folder,'nlp','ner_test.csv')

stems_train_fp = os.path.join(data_folder,'nlp','stems_train.csv')
stems_test_fp = os.path.join(data_folder,'nlp','stems_test.csv')

tfidf_with_stops_train_fp = os.path.join(data_folder,'tfidf','tokens_with_stop_words_tfidf_train.csv')
tfidf_with_stops_test_fp = os.path.join(data_folder,'tfidf','tokens_with_stop_words_tfidf_test.csv')

magic_train_fp=os.path.join(data_folder, 'magic', 'magic_train.csv')
magic_test_fp=os.path.join(data_folder, 'magic', 'magic_test.csv')

magic2_train_fp = os.path.join(data_folder, 'magic', 'magic2_train.csv')
magic2_test_fp = os.path.join(data_folder, 'magic', 'magic2_test.csv')


common_words_train_fp = os.path.join(data_folder, 'basic','common_words_train.csv')
length_train_fp = os.path.join(data_folder, 'basic','lens_train.csv')

common_words_test_fp = os.path.join(data_folder, 'basic','common_words_test.csv')
length_test_fp = os.path.join(data_folder, 'basic','lens_test.csv')

TRAIN_METRICS_FP = [
    data_folder + 'distances/'+ 'train_metrics_bool_lemmas.csv',
    data_folder + 'distances/'+'train_metrics_bool_stems.csv',
    data_folder + 'distances/'+'train_metrics_bool_tokens.csv',
    data_folder + 'distances/'+'train_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/'+'train_metrics_fuzzy_stems.csv',
    data_folder + 'distances/'+'train_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/'+'train_metrics_sequence_lemmas.csv',
    data_folder + 'distances/'+'train_metrics_sequence_stems.csv',
    data_folder + 'distances/'+'train_metrics_sequence_tokens.csv'
]

TEST_METRICS_FP = [
    data_folder + 'distances/'+ 'test_metrics_bool_lemmas.csv',
    data_folder + 'distances/'+'test_metrics_bool_stems.csv',
    data_folder + 'distances/'+'test_metrics_bool_tokens.csv',
    data_folder + 'distances/'+'test_metrics_fuzzy_lemmas.csv',
    data_folder + 'distances/'+'test_metrics_fuzzy_stems.csv',
    data_folder + 'distances/'+'test_metrics_fuzzy_tokens.csv',
    data_folder + 'distances/'+'test_metrics_sequence_lemmas.csv',
    data_folder + 'distances/'+'test_metrics_sequence_stems.csv',
    data_folder + 'distances/'+'test_metrics_sequence_tokens.csv'
]


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

def load_test_nlp():
    return pd.concat([
        load_test(),
        load_test_postag(),
        load_test_lemmas(),
        load_test_stems(),
        load_test_tokens(),
        load_test_ner()
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
        df[col]=df[col].apply(str)
    return df

def load_test_lemmas():
    df = pd.read_csv(lemmas_test_fp, index_col='test_id')
    df = df.fillna('')
    for col in [lemmas_q1, lemmas_q2]:
        df[col]=df[col].apply(str)
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
    df = pd.concat([
        pd.read_csv(magic_train_fp, index_col='id')[['freq_question1', 'freq_question2']],
        pd.read_csv(magic2_train_fp, index_col='id')],
        axis=1
    )
    return df

def load_test_magic():
    df = pd.concat([
        pd.read_csv(magic_test_fp, index_col='test_id')[['freq_question1', 'freq_question2']],
        pd.read_csv(magic2_test_fp, index_col='test_id')],
        axis=1
    )
    return df


def load_train_stems():
    df = pd.read_csv(stems_train_fp, index_col='id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    for col in [stems_q1, stems_q2]:
        df[col]=df[col].apply(str)
    return df

def load_test_stems():
    df = pd.read_csv(stems_test_fp, index_col='test_id')
    df = df[['question1_porter', 'question2_porter']]
    df = df.rename(columns={'question1_porter': 'stems_q1', 'question2_porter': 'stems_q2'})
    df = df.fillna('')
    for col in [stems_q1, stems_q2]:
        df[col]=df[col].apply(str)
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

def shuffle_df(df, random_state=42):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]

def explore_target_ratio(df):
    return {
        'pos':1.0*len(df[df[TARGET]==1])/len(df),
        'neg':1.0*len(df[df[TARGET]==0])/len(df)
    }



# df = load_train_all()
######################################################################################
######################################################################################
######################################################################################
######################################################################################

#WH

wh_fp_train=os.path.join(data_folder, 'wh', 'wh_train.csv')
wh_fp_test=os.path.join(data_folder, 'wh', 'wh_test.csv')

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
from collections import Counter

freq_question1,  freq_question2,  q1_q2_intersect='freq_question1',  'freq_question2',  'q1_q2_intersect'


def get_all_questions_flat():
    train_df, test_df = load_train(), load_test()
    l= list(train_df[question1])+list(train_df[question2])+list(test_df[question1])+list(test_df[question2])
    return Counter(l)


def get_all_questions_flat_train():
    df = load_train()
    l= list(df[question1])+list(df[question2])
    return Counter(l)

def get_all_questions_flat_test():
    df = load_test()
    l= list(df[question1])+list(df[question2])
    return Counter(l)


def most_common_intersection(N):
    c_train = get_all_questions_flat_train()
    c_test=get_all_questions_flat_test()

    a=set([x[0] for x in c_train.most_common(N)])
    b=set([x[0] for x in c_test.most_common(N)])

    return a.intersection(b)

def filter_df_q1_q2(df, s):
    return df[(df[question1].apply(lambda x: str(x)==s))|(df[question2].apply(lambda x: str(x)==s))]

def filter_df_q1(df, s):
    return df[(df[question1].apply(lambda x: str(x)==s))]

def filter_df_q2(df, s):
    return df[(df[question1].apply(lambda x: str(x)==s))]

def explore_common_counts_test(c_train, c_test, N=100):
    for x in c_test.most_common(N):
        print x[0], c_train[x[0]], c_test[x[0]]


def explore_common_counts_train(c_train, c_test, N=100):
    for x in c_train.most_common(N):
        print x[0], c_train[x[0]], c_test[x[0]]


def drop_dups(df):
    return df[~df.index.duplicated(keep='first')]


def explore_magic_train():
    return pd.concat([
        load_train(),
        load_train_magic()
    ], axis=1)

def explore_magic_test():
    return pd.concat([
        load_test(),
        load_test_magic()
    ], axis=1)


def explore_counts_in_train_test(s):
    a = filter_df(train_df, s)
    b = filter_df(test_df, s)
    print 'train: {}'.format(len(a))
    ratio = explore_target_ratio(a)
    print 'ratio: {}'.format(ratio)
    print ''

    print 'test: {}'.format(len(b))
    # print 'ratio: {}'.format(explore_target_ratio(b))
    print ''

    return {'train_count':len(a), 'test_count':len(b), 'ratio':ratio}

def explore_counts_in_train_test_write(s, train_df, test_df):
    a = filter_df(train_df, s)
    b = filter_df(test_df, s)
    print 'train: {}'.format(len(a))
    ratio = explore_target_ratio(a)
    print 'ratio: {}'.format(ratio)
    print ''

    print 'test: {}'.format(len(b))
    # print 'ratio: {}'.format(explore_target_ratio(b))
    print ''

    return {'train_count':len(a), 'test_count':len(b), 'ratio':ratio}


def how_many_tops_from_train_are_in_test(N=100):
    c =get_all_questions_flat()
    ss = {x[0] for x in c.most_common(N)}
    return len(test_df[(test_df[question1].apply(lambda s: s in ss))|(test_df[question1].apply(lambda s: s in ss))])


def top_by_magic_target_ratios():
    c =get_all_questions_flat_train()
    for x in c.most_common(1000):
        print x
        print explore_counts_in_train_test(x[0])
        print '========================================='


def write_strange_counts(N=10):
    c =get_all_questions_flat_train()
    res=[]
    for x in c.most_common(1000):
        print x
        data = explore_counts_in_train_test(x[0])
        print data
        print '========================================='


question1_in_q1_train         =      'question1_in_q1_train'
question1_in_q2_train_test    =      'question1_in_q2_train_test'
question1_in_q2_train         =      'question1_in_q2_train'
question1_in_q1_train_test    =      'question1_in_q1_train_test'
question1_in_q_test           =      'question1_in_q_test'
question1_in_q2_test          =      'question1_in_q2_test'
question1_in_q1_test          =      'question1_in_q1_test'
question1_in_q_train          =      'question1_in_q_train'
question2_in_q2_train_test    =      'question2_in_q2_train_test'
question2_in_q1_train         =      'question2_in_q1_train'
question2_in_q2_train         =      'question2_in_q2_train'
question2_in_q1_train_test    =      'question2_in_q1_train_test'
question2_in_q_test           =      'question2_in_q_test'
question2_in_q2_test          =      'question2_in_q2_test'
question2_in_q1_test          =      'question2_in_q1_test'
question2_in_q_train          =      'question2_in_q_train'

def filter_nans(ser):
    def is_not_nan(s):
        return not s!=s and not s is None

    return filter(is_not_nan, ser)

def add_big_in_train_test_columns():
    train_df, test_df = load_train(), load_test()

    q1_train = Counter(train_df[question1])
    q2_train = Counter(train_df[question2])

    q1_test = Counter(test_df[question1])
    q2_test = Counter(test_df[question2])

    q_train = Counter(list(train_df[question1]) + list(train_df[question2]))
    q_test = Counter(list(test_df[question1]) + list(test_df[question2]))

    q1_train_test = Counter(list(train_df[question1]) + list(test_df[question1]))
    q2_train_test = Counter(list(train_df[question2]) + list(test_df[question2]))

    m = {
        'q1_train': q1_train,
        'q2_train': q2_train,
        'q1_test': q1_test,
        'q2_test': q2_test,
        'q_train': q_train,
        'q_test': q_test,
        'q1_train_test': q1_train_test,
        'q2_train_test': q2_train_test
    }

    for df in [train_df, test_df]:
        for col in [question1, question2]:
            for name, counter in m.iteritems():
                df['{}_in_{}'.format(col, name)] = df[col].apply(lambda s: counter[s])
                print col, name


    return train_df, test_df