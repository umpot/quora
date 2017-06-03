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

new_qids_test_fp = os.path.join(data_folder, 'magic', 'new_quids.csv')
max_k_cores_fp = os.path.join(data_folder, 'magic', 'question_max_kcores.csv')

max_k_cores_train_fp = os.path.join(data_folder, 'magic', 'max_k_cores_train.csv')
max_k_cores_test_fp = os.path.join(data_folder, 'magic', 'max_k_cores_test.csv')


def load_folds():
    return json.load(open(folds_fp))


def create_folds(df):
    folds = load_folds()

    return [
        (df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']])
        for x in range(len(folds))]


def split_into_folds(df, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    res = []
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
        pd.concat(
            [pd.read_csv(fp_test, index_col='test_id'),
             pd.read_csv(new_qids_test_fp, index_col='test_id')],
            axis=1
        )
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
######################################################################################
######################################################################################
######################################################################################
######################################################################################



AUX = [
    'be',
    'am',
    'are',
    'is',
    'was',
    'were',
    'being',
    'been',
    'can',
    'could',
    'do',
    'does',
    'did',
    'have'
    'has',
    'had',
    'having',
    'may',
    'might',
    'must',
    'need',
    'ought',
    'shall',
    'should',
    'will',
    'would'
]

Demonstrative_Pronouns=['this', 'that', 'these', 'those']
Interrogative_Pronouns=[ 'who', 'whom', 'which', 'what', 'whose',
                         'whoever', 'whatever', 'whichever', 'whomever']
Relative_Pronouns=['who', 'whom', 'whose', 'which', 'that',
                   'what', 'whatever', 'whoever', 'whomever', 'whichever']

Subjective_Pronouns=['i', 'you', 'he', 'she', 'it', 'we', 'they']
Objective_Pronouns=['me', 'him', 'her', 'us','it', 'them']
Possessive_Pronouns=['mine', 'yours', 'his', 'hers', 'ours', 'theirs']
Reflexive_Pronouns=['myself', 'yourself', 'himself', 'herself',
                    'itself', 'ourselves', 'themselves']



Indefinite_Pronouns=[
    'anything', 'everybody', 'another', 'each', 'few', 'many', 'none', 'some',
    'all', 'any', 'anybody', 'anyone', 'everyone', 'everything', 'no one',
    'nobody', 'nothing', 'none', 'other', 'others', 'several',
    'somebody', 'someone', 'something', 'most', 'enough',
    'little', 'more', 'both', 'either', 'neither', 'one', 'much', 'such'
]

preposition=['aboard',
             'about',
             'above',
             'across',
             'after',
             'against',
             'along',
             'amid',
             'among',
             'anti',
             'around',
             'as',
             'at',
             'before',
             'behind',
             'below',
             'beneath',
             'beside',
             'besides',
             'between',
             'beyond',
             'but',
             'by',
             'concerning',
             'considering',
             'despite',
             'down',
             'during',
             'except',
             'excepting',
             'excluding',
             'following',
             'for',
             'from',
             'in',
             'inside',
             'into',
             'like',
             'minus',
             'near',
             'of',
             'off',
             'on',
             'onto',
             'opposite',
             'outside',
             'over',
             'past',
             'per',
             'plus',
             'regarding',
             'round',
             'save',
             'since',
             'than',
             'through',
             'to',
             'toward',
             'towards',
             'under',
             'underneath',
             'unlike',
             'until',
             'up',
             'upon',
             'versus',
             'via',
             'with',
             'within',
             'without']

SPACIAL_PREP = ['within','towards','toward',
                'through','outside','onto',
                'on', 'near', 'for', 'in', 'from',
                'between', 'beyond', 'by', 'above', 'across', 'around', 'as', 'at']
TIME_PREP=['in', 'for', 'during', 'before', 'after', 'since', 'until', 'while']

SHORT_PREP = ['in', 'on', 'at', 'as', 'of', 'by', 'but']

QUATITIES=['too', 'little', 'few', 'more', 'much', 'enough', 'fewer', 'lot', 'plenty', 'lots', 'less']

WH_QUESTIONS=['what', 'where', 'why', 'who', 'whome', 'whose', 'how', 'when', 'which']

######################################################################################
######################################################################################
######################################################################################
def first_or_none(s):
    s = s.split()
    if len(s) == 0:
        return None
    return s[0]


def add_first_token_cols(df, cols):
    new_cols = []
    for col in cols:
        new_col = 'first_tok_{}'.format(col)
        df[new_col] = df[col].apply(first_or_none)
        new_cols.append(new_col)

    return new_cols


######################################################################################
######################################################################################
######################################################################################
def add_from_set_cols(df, col1, col2, ss, ss_name):
    new_cols = []

    new_col = '{}_q1'.format(ss_name)
    df[new_col] = df[col1].apply(lambda x: ' '.join([y for y in x.lower().split() if y in ss]))
    new_cols.append(new_col)

    new_col = '{}_q2'.format(ss_name)
    df[new_col] = df[col2].apply(lambda x: ' '.join([y for y in x.lower().split() if y in ss]))
    new_cols.append(new_col)

    return new_cols


######################################################################################
######################################################################################
######################################################################################

def get_top_vals_freq_map(df, col, limit):
    group = df.groupby(col)[TARGET]
    bl = group.count().sort_values(ascending=False)
    bl = bl[bl >= limit].index
    means = group.mean()
    return {k: means[k] for k in bl}


def add_freq_cols(train_df, test_df, col1, col2, prefix, limit=300):
    p = '{}_pair'.format(prefix)
    train_df[p] = train_df[col1] + '***' + train_df[col2]
    test_df[p] = test_df[col1] + '***' + test_df[col2]

    new_cols = []

    for col in [col1, col2, p]:
        m = get_top_vals_freq_map(train_df, col, limit)
        new_col = '{}_freq'.format(col)
        new_cols.append(new_col)
        for df in [train_df, test_df]:
            df[new_col] = df[col].apply(lambda s: m.get(s, None))

    return new_cols


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################

def create_pairs_freq_features():
    train_df, test_df = load_train(), load_test()
    sets_map = {
        'aux':set(AUX),
        'subj_pron':set(Subjective_Pronouns),
        'obj_pron':set(Objective_Pronouns),
        'poss_pron':set(Possessive_Pronouns),
        'refl_pron':set(Reflexive_Pronouns),
        'ind_pron':set(Indefinite_Pronouns),
        'pers_pron':set(Subjective_Pronouns+Objective_Pronouns+Possessive_Pronouns+Reflexive_Pronouns),
        'spacial_prep':set(SPACIAL_PREP),
        'time_prep':set(TIME_PREP),
        'quant_prep':set(QUATITIES),
        'wh_quest':set(WH_QUESTIONS)
        
    }

    new_cols=[]
    for name, ss in sets_map.iteritems():
        cols=add_from_set_cols(train_df, question1, question2, ss, name)
        add_from_set_cols(test_df, question1, question2, ss, name)
        new_cols+=add_freq_cols(train_df, test_df, cols[0], cols[1], name)

    pair_freq_train_fp = os.path.join(data_folder, 'pair_freq', 'pair_freq_train_v1.csv')
    pair_freq_test_fp = os.path.join(data_folder, 'pair_freq', 'pair_freq_test_v1.csv')


    train_df[new_cols].to_csv(pair_freq_train_fp, index_label='id')
    test_df[new_cols].to_csv(pair_freq_test_fp, index_label='test_id')

    return train_df, test_df, new_cols

create_pairs_freq_features()
