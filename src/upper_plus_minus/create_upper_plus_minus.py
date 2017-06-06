import pandas as pd
import numpy as np
import seaborn as sns
import re
import os
import sys

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

folds_fp = os.path.join(data_folder, 'top_k_freq', 'folds.json')

new_qids_test_fp = os.path.join(data_folder,'magic' ,'new_quids.csv')
max_k_cores_fp = os.path.join(data_folder,'magic' ,'question_max_kcores.csv')

max_k_cores_train_fp=os.path.join(data_folder,'magic' ,'max_k_cores_train.csv')
max_k_cores_test_fp=os.path.join(data_folder,'magic' ,'max_k_cores_test.csv')

def load_folds():
    return json.load(open(folds_fp))


def create_folds(df):
    folds = load_folds()

    return [
        (df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']])
        for x in range(len(folds))]

def split_into_folds(df, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    res=[]
    for big_ind, small_ind in skf.split(np.zeros(len(df)), df[TARGET]):
        res.append((df.loc[big_ind], df.loc[small_ind]))

    return res


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

magic2_train_fp = os.path.join(data_folder, 'magic', 'magic2_train.csv')
magic2_test_fp = os.path.join(data_folder, 'magic', 'magic2_test.csv')

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



def load_max_k_cores_train():
    return pd.read_csv(max_k_cores_train_fp, index_col='id')


def load_max_k_cores_test():
    return pd.read_csv(max_k_cores_test_fp, index_col='test_id')


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


def load_train_tfidf():
    df = pd.read_csv(tfidf_with_stops_train_fp, index_col='id')
    return df


def load_train_tfidf_new():
    fps = [
        os.path.join(data_folder, 'tfidf', x) for x in ['train_dirty_lower_no_stops.csv',
                                                        'train_dirty_upper.csv',
                                                        'train_tokens_lower.csv',
                                                        'train_tokens_lower_no_stops.csv']
        ]
    return pd.concat(
        [pd.read_csv(fp, index_col='id') for fp in fps],
        axis=1)

def load_test_tfidf_new():
    fps = [
        os.path.join(data_folder, 'tfidf', x) for x in ['test_dirty_lower_no_stops.csv',
                                                        'test_dirty_upper.csv',
                                                        'test_tokens_lower.csv',
                                                        'test_tokens_lower_no_stops.csv']
        ]
    return pd.concat(
        [pd.read_csv(fp, index_col='test_id') for fp in fps],
        axis=1)


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


def shuffle_df(df, random_state=42):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]


def explore_target_ratio(df):
    return {
        'pos': 1.0 * len(df[df[TARGET] == 1]) / len(df),
        'neg': 1.0 * len(df[df[TARGET] == 0]) / len(df)
    }


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
from collections import defaultdict
from scipy.stats import kurtosis, skew, skewnorm
x_None= 'x_None'
RATIO = 'RATIO'

counter = 0


def add_target_ratio_to_m(m):
    for v in m.values():
        # print v
        for col, blja in v.iteritems():
            # print col, blja
            cnt = blja['count']
            t = blja[1]
            blja[RATIO]= (1.0*t)/cnt



def get_top_uppers_dict(df, top_N):
    res = defaultdict(dict)

    def make_upper_stats(x, y, top_N_set, result_set,  target):
        global counter
        counter+=1
        if counter%1000==0:
            print counter

        x = set(str(x).split())
        y = set(str(y).split())

        in_q1_not_in_2 = x.difference(y).intersection(top_N_set)
        in_q2_not_in_1 = y.difference(x).intersection(top_N_set)
        sym_diff=x.symmetric_difference(y).intersection(top_N_set)
        both = x.intersection(y).intersection(top_N_set)
        all_set = x.union(y).intersection(top_N_set)

        cols_map={
            'in_q1_not_in_2':in_q1_not_in_2,
            'in_q2_not_in_1':in_q2_not_in_1,
            'sym_diff':sym_diff,
            'both':both,
            'all':all_set
        }

        for col, ss in cols_map.iteritems():
            for a in ss:
                d=result_set[a]
                if col not in d:
                    d[col]={}
                    d[col]['count']=0
                    d[col][1]=0
                    d[col][0]=0

                d[col]['count']+=1
                d[col][target]+=1


    df.apply(lambda row: make_upper_stats(row[tokens_q1], row[tokens_q2],top_N, res, row[TARGET]), axis = 1)
    print 'Done2'

    add_target_ratio_to_m(res)

    return res

from collections import Counter
import json

def get_top_uppers(df):
    def filter_upper(s):
        return filter(lambda x: x[0].isupper(), s.split())
    tmp = df[tokens_q1]+' '+df[tokens_q2]
    tmp = tmp.apply(filter_upper)
    l=[]
    for x in tmp:
        l+=x
    return Counter(l)

def load_top_110_uppers():
    fp = os.path.join(data_folder, 'new_top_uppers', '1100_top_upper_tokens.json')
    return json.load(open(fp))

def add_upper_cols(df):
    top = load_top_110_uppers()
    top = set(top)
    def filter_upper(s):
        up = filter(lambda x: x[0].isupper(), s.split())
        up=set(up)
        return up.intersection(top)
    # def first_or_None(l):
    #     if len(l)==0:
    #         return None
    #     return l[0]
    for col in [tokens_q1, tokens_q2]:
        df[col] = df[col].apply(filter_upper)#.apply(first_or_None)


def get_count_map_from_freq_map(m):
    return {k:v['all']['count'] for k,v in m.iteritems()}

def get_most_frequent_upper(ss, count_m):


    l = [(x, count_m.get(x, None)) for x in ss if x in count_m]
    if len(l)==0:
        return None
    l.sort(key=lambda x:x[0], reverse=True)
    return l[0][0]


def process_top_N_uppers_plus_minus(df, m, update_df):
    global counter
    counter=0

    m_count = get_count_map_from_freq_map(m)

    upper_col='upper_col'

    add_upper_cols(df)

    def get_ratio_safe(m, top, col):
        x = m.get(top, None)
        if x is None:
            return None

        x = x.get(col, None)
        if x is None:
            return None
        return x[RATIO]


    def get_stat(row):
        a = row[tokens_q1]
        b = row[tokens_q2]
        u = a.union(b)
        if len(u)==0:
            return {
                'both':None,
                'in_q1_not_in_2':None,
                'in_q2_not_in_1':None,
                'sym_diff':None,

            }
        top = get_most_frequent_upper(u, m_count)
        if top is None:
            return {
                'both':None,
                'in_q1_not_in_2':None,
                'in_q2_not_in_1':None,
                'sym_diff':None,

            }
        if top in a and top in b:
            return {
                'both':get_ratio_safe(m, top, 'both'),
                'in_q1_not_in_2':None,
                'in_q2_not_in_1':None,
                'sym_diff':None,

            }
        elif top in a:
            return {
                'both':None,
                'in_q1_not_in_2':get_ratio_safe(m, top, 'in_q1_not_in_2'),
                'in_q2_not_in_1':None,
                'sym_diff':get_ratio_safe(m, top, 'sym_diff'),

            }
        elif top in b:
            return {
                'both':None,
                'in_q1_not_in_2':None,
                'in_q2_not_in_1':get_ratio_safe(m, top,'in_q2_not_in_1'),
                'sym_diff':get_ratio_safe(m, top,'sym_diff'),

            }
        else:
            return {
                'both':None,
                'in_q1_not_in_2':None,
                'in_q2_not_in_1':None,
                'sym_diff':None,

            }



    df[upper_col] = df.apply(get_stat, axis = 1)



    cols=[
        'in_q1_not_in_2',
        'in_q2_not_in_1',
        'sym_diff',
        'both'
    ]

    new_cols =[]
    for col in cols:
        new_col = 'top_1.1K_upper_{}'.format(col)
        new_cols.append(new_col)
        df[new_col] = df[upper_col].apply(lambda s: s[col])

    for col in new_cols:
        if update_df is not None:
            update_df.loc[df.index, col]=df.loc[df.index, col]

    return new_cols


new_top_uppers_train_fp = os.path.join(data_folder, 'new_top_uppers', 'new_top_uppers_train.csv')
new_top_uppers_test_fp = os.path.join(data_folder, 'new_top_uppers', 'new_top_uppers_test_fp.csv')


def write_top_1_1_uppers_plus_minus():
    train_df, test_df = load_train_nlp(), load_test_nlp()
    folds = create_folds(train_df)

    # train_df, test_df = load_train_nlp().head(5000), load_test_nlp().head(5000)
    # folds = split_into_folds(train_df, 7)

    topN=load_top_110_uppers()



    new_cols = None
    for train, test in folds:
        m = get_top_uppers_dict(train, topN)
        new_cols = process_top_N_uppers_plus_minus(test, m, train_df)

    train_df[new_cols].to_csv(new_top_uppers_train_fp, index_label='id')
    print 'Done TRAIN!'
    print '======================================='


    m = get_top_uppers_dict(train_df, topN)
    new_cols = process_top_N_uppers_plus_minus(test_df, m, None)
    test_df[new_cols].to_csv(new_top_uppers_test_fp, index_label='test_id')
    print 'Done TEST!'
    print '======================================='



write_top_1_1_uppers_plus_minus()