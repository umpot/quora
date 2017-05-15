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
ner_q1, ner_q2='ner_q1', 'ner_q2'

data_folder = '../../../data/'
# data_folder = '../../data/'


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



# df = load_train_all()
######################################################################################
######################################################################################
######################################################################################
######################################################################################



TARGET = 'is_duplicate'

wh1='wh1'
wh2='wh2'
wh_list_1='wh_list_1'
wh_list_2='wh_list_2'
wh_same = 'wh_same'

def explore_target_ratio(df):
    return {
        'pos':1.0*len(df[df[TARGET]==1])/len(df),
        'neg':1.0*len(df[df[TARGET]==0])/len(df)
    }



questions_types=[
    'why', 'what', 'who', 'how', 'where', 'why', 'when', 'which'
]

modals=[
    'can',
    'could',
    'may',
    'might',
    'shall',
    'should',
    'will',
    'would',
    'must'
]


def add_the_same_wh_col(df):
    df[wh_same]=df[[wh1, wh2]].apply(lambda s: s[wh1] == s[wh2], axis=1)
    df[wh_same]=df[wh_same].apply(lambda s: 1 if s else 0)

def add_wh_cols(df):
    df[wh1] = df[lemmas_q1].apply(get_wh_type)
    df[wh2] = df[lemmas_q2].apply(get_wh_type)


def add_wh_list_cols(df):
    df[wh_list_1] = df[lemmas_q1].apply(get_wh_list)
    df[wh_list_2] = df[lemmas_q2].apply(get_wh_list)




def get_wh_type(s):
    s='' if s is None else str(s).lower()

    for w in questions_types:
        if s.startswith(w):
            return w

def get_wh_list(s):
    l=s.split()
    res=[]
    for t in l:
        if t in questions_types:
            res.append(t)

    return res


wh_fp_train=os.path.join(data_folder, 'wh', 'wh_train.csv')
wh_fp_test=os.path.join(data_folder, 'wh', 'wh_test.csv')

def load_wh_train():
    df = pd.read_csv(wh_fp_train, index_col='id')
    return df

def load_wh_test():
    df = pd.read_csv(wh_fp_test, index_col='test_id')
    return df


def load_exploring():
    df= pd.concat([load_train(), load_train_lemmas()], axis=1)
    add_wh_cols(df)
    add_wh_list_cols(df)

    return df


def write_wh_naive():
    train_df, test_df = load_train_lemmas(), load_test_lemmas()
    for df in [train_df, test_df]:
        add_wh_cols(df)
        add_the_same_wh_col(df)
        df[wh1]=df[wh1].apply(lambda s: -1 if s is None else questions_types.index(s))
        df[wh2]=df[wh2].apply(lambda s: -1 if s is None else questions_types.index(s))

    new_cols = [wh1, wh2, wh_same]

    index_label='id'
    df = train_df[new_cols]
    df.to_csv(wh_fp_train, index_label=index_label)

    index_label='test_id'
    df = test_df[new_cols]
    df.to_csv(wh_fp_test, index_label=index_label)



def get_most_frequent_start_words_ngrams(df, n):
    m=get_most_frequent_start_words_ngrams_map(df, n)

    m= [(k,v) for k,v in m.items()]
    m.sort(key=lambda s: s[1], reverse=True)

    return m


def get_ngram_prefix(s, n):
    s='' if s is None else str(s).lower()
    lst = s.split()
    if len(lst)<n:
        return tuple(lst)

    return tuple(lst[:n])

def get_most_frequent_start_words_ngrams_map(df, n):
    l = list(df[question1].apply(str))+list(df[question2].apply(str))

    m={}
    for i in l:
        pref = get_ngram_prefix(i, n)
        if pref is None:
            continue

        if pref in m:
            m[pref]+=1
        else:
            m[pref]=1

    return m

['pref_freq1_1',
 'pref_freq2_1',
 'pref_freq1_2',
 'pref_freq2_2',
 'pref_freq1_3',
 'pref_freq2_3',
 'pref_freq1_4',
 'pref_freq2_4']

def add_prefix_frequencies_cols_train(train_df, nums=(1,2,3,4)):
    df = train_df

    def get_prefix_count(m, pref):
        if pref in m:
            return m[pref]
        return 1

    new_cols = []

    for N in nums:
        m = get_most_frequent_start_words_ngrams_map(train_df, N)

        pref_freq1 = 'pref_freq1_{}'.format(N)
        new_cols.append(pref_freq1)
        df[pref_freq1]=df[question1].apply(lambda s: get_ngram_prefix(s, N))
        df[pref_freq1]=df[pref_freq1].apply(lambda s: get_prefix_count(m ,s))

        pref_freq2 = 'pref_freq2_{}'.format(N)
        new_cols.append(pref_freq2)
        df[pref_freq2]=df[question2].apply(lambda s: get_ngram_prefix(s, N))
        df[pref_freq2]=df[pref_freq2].apply(lambda s: get_prefix_count(m ,s))


    return new_cols


def get_comon_prefix_len(a,b):
    a='' if a is None else a
    b='' if b is None else b

    a=str(a).lower().split()
    b=str(b).lower().split()

    l = min(len(a), len(b))
    res = 0

    for i in range(l):
        if a[i]==b[i]:
            res+=1
        else:
            break

    return res

def get_comon_prefix_ratio(a,b):
    a='' if a is None else a
    b='' if b is None else b

    a=str(a).lower().split()
    b=str(b).lower().split()

    l = min(len(a), len(b))
    res = 0

    for i in range(l):
        if a[i]==b[i]:
            res+=1
        else:
            break

    max1 = max(len(a), len(b))
    max1=max(max1, 1)
    return 1.0*res / max1

def add_common_prefix_cols(df, col1, col2):
    df['common_prefix_len']=df[[col1, col2]].apply(lambda s: get_comon_prefix_len(s[col1], s[col2]), axis=1)
    df['common_prefix_ratio']=df[[col1, col2]].apply(lambda s: get_comon_prefix_ratio(s[col1], s[col2]), axis=1)


def add_prefix_frequencies_cols_train_test(train_df, test_df, nums=(1,2,3,4)):
    def get_prefix_count(m, pref):
        if pref in m:
            return m[pref]
        return 1

    new_cols = []

    for df in [train_df, test_df]:
        for N in nums:
            m = get_most_frequent_start_words_ngrams_map(train_df, N)

            pref_freq1 = 'pref_freq1_{}'.format(N)
            new_cols.append(pref_freq1)
            df[pref_freq1]=df[question1].apply(lambda s: get_ngram_prefix(s, N))
            df[pref_freq1]=df[pref_freq1].apply(lambda s: get_prefix_count(m ,s))

            pref_freq2 = 'pref_freq2_{}'.format(N)
            new_cols.append(pref_freq2)
            df[pref_freq2]=df[question2].apply(lambda s: get_ngram_prefix(s, N))
            df[pref_freq2]=df[pref_freq2].apply(lambda s: get_prefix_count(m ,s))

            pref_freq_log='pref_freq_log_{}'.format(N)
            new_cols.append(pref_freq_log)
            df[pref_freq_log] = np.abs(np.log(df[pref_freq1]/df[pref_freq2]))


    return new_cols

prefixes_fp_train=os.path.join(data_folder, 'wh', 'prefixes_train.csv')
prefixes_fp_test=os.path.join(data_folder, 'wh', 'prefixes_test.csv')
def write_prefixes():
    train_df, test_df = load_train(), load_test()
    add_prefix_frequencies_cols_train_test(train_df, test_df)
    add_common_prefix_cols(train_df, question1, question2)
    add_common_prefix_cols(test_df, question1, question2)
    new_cols = ['pref_freq1_1',
                'pref_freq2_1',
                'pref_freq1_2',
                'pref_freq2_2',
                'pref_freq1_3',
                'pref_freq2_3',
                'pref_freq1_4',
                'pref_freq2_4',
                'common_prefix_len',
                'common_prefix_ratio',
                'pref_freq_log_1',
                'pref_freq_log_2',
                'pref_freq_log_3',
                'pref_freq_log_4'
                ]



    index_label='id'
    df = train_df[new_cols]
    df.to_csv(prefixes_fp_train, index_label=index_label)

    index_label='test_id'
    df = test_df[new_cols]
    df.to_csv(prefixes_fp_test, index_label=index_label)


write_prefixes()

