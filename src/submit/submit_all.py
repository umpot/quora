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

data_folder = '../../../data/'

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


############################################################3
############################################################3
############################################################3
import pandas as pd
import numpy as np

TARGET = 'is_duplicate'

INDEX_PREFIX= 100000000
#old
{'pos': 0.369197853026293,
 'neg': 0.630802146973707}


#new
r1 = 0.174264424749
r0 = 0.825754788586

""""
p_old/(1+delta) = p_new

delta = (p_old/p_new)-1 = 1.1186071314214785
l = delta*N = 452241
"""

delta = 1.1186071314214785

def explore_target_ratio(df):
    return {
        'pos':1.0*len(df[df[TARGET]==1])/len(df),
        'neg':1.0*len(df[df[TARGET]==0])/len(df)
    }

def shuffle_df(df, random_state):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]

def oversample_df(df, l, random_state):
    df_pos = df[df[TARGET]==1]
    df_neg = df[df[TARGET]==0]

    df_neg_sampl = df_neg.sample(l, random_state=random_state, replace=True)

    df=pd.concat([df_pos, df_neg, df_neg_sampl])
    df = shuffle_df(df, random_state)

    return df

def oversample(train_df, test_df, random_state=42):
    l_train = int(delta * len(train_df))
    l_test = int(delta * len(test_df))

    return oversample_df(train_df, l_train, random_state), oversample_df(test_df, l_test, random_state)

def oversample_submit(train_df, test_df, random_state=42):
    l_train = int(delta * len(train_df))

    return oversample_df(train_df, l_train, random_state),test_df



############################################################3
############################################################3
############################################################3
trash_cols = [
    "w_share_ratio_2_std_idf_dirty_lower_no_stops",
    "w_share_ratio_2_smooth_idf_dirty_upper",
    "w_share_ratio_2_std_idf_tokens_lower_no_stops",
    "abi_jaccard_distance",
    "len_char_diff_log",
    "len_word_diff_log",
    "len_word_expt_stop_diff_log",
    "stop_words_num_q1",
    "stop_words_num_q2",
    "lemmas_kulsinski",
    "lemmas_dice",
    "lemmas_jaccard",
    "stems_kulsinski",
    "stems_dice",
    "stems_jaccard",
    "tokens_dice",
    "tokens_jaccard",
    "lemmas_partial_token_set_ratio",
    "stems_partial_token_set_ratio",
    "tokens_partial_token_set_ratio",
    "lemmas_distance.jaccard",
    "stems_distance.jaccard",
    "tokens_distance.jaccard",
    "w_share_ratio_2_smooth_idf_dirty_lower_no_stops",
    "w_share_ratio_2_std_idf_dirty_upper",
    "w_share_ratio_2_smooth_idf_tokens_lower",
    "w_share_ratio_2_std_idf_tokens_lower",
    "w_share_ratio_2_smooth_idf_tokens_lower_no_stops",
    "jaccard_distance_lemmas_glove",
    "jaccard_distance_lemmas_lex",
    "jaccard_distance_lemmas_word2vec",
    "jaccard_distance_tokens_glove",
    "jaccard_distance_tokens_lex"
]

def del_trash_cols(df):
    for col in trash_cols:
        if col in df:
            del df[col]

############################################################3
############################################################3
############################################################3
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






############################################################3
############################################################3
############################################################3
one_upper_fp_train=os.path.join(data_folder, 'keywords', 'train_upper_freq_200.csv')
one_upper_fp_test=os.path.join(data_folder, 'keywords', 'test_upper_freq_200.csv')

def load_one_upper_train():
    df = pd.read_csv(one_upper_fp_train, index_col='id')
    return df

def load_one_upper_test():
    df = pd.read_csv(one_upper_fp_test, index_col='test_id')
    return df

######################################################################################
######################################################################################
######################################################################################
######################################################################################
train_avg_tokK_freq_fp=os.path.join(data_folder, 'top_k_freq', 'train_avg_K_tok_freq.csv')
test_avg_tokK_freq_fp=os.path.join(data_folder, 'top_k_freq', 'test_avg_K_tok_freq.csv')

def load_topNs_avg_tok_freq_train():
    return pd.read_csv(train_avg_tokK_freq_fp, index_col='id')

def load_topNs_avg_tok_freq_test():
    return pd.read_csv(test_avg_tokK_freq_fp, index_col='test_id')
######################################################################################
######################################################################################
######################################################################################
######################################################################################
abi_train_fp = os.path.join(data_folder, 'abishek', 'abi_train.csv')
abi_test_fp = os.path.join(data_folder, 'abishek', 'abi_test.csv')


def load_abi_train():
    return pd.read_csv(abi_train_fp, index_col='id')

def load_abi_test():
    return pd.read_csv(abi_test_fp, index_col='test_id')

############################################################3
############################################################3
############################################################3

train_pos_metrics_fp=os.path.join(data_folder, 'pos_metrics', 'train_pos_metrics.csv')
test_pos_metrics_fp=os.path.join(data_folder, 'pos_metrics', 'test_pos_metrics.csv')

def load_metrics_on_pos_train():
    return pd.read_csv(train_pos_metrics_fp, index_col='id')

def load_metrics_on_pos_test():
    return pd.read_csv(test_pos_metrics_fp, index_col='test_id')


max_k_cores_train_fp=os.path.join(data_folder,'magic' ,'max_k_cores_train.csv')
max_k_cores_test_fp=os.path.join(data_folder,'magic' ,'max_k_cores_test.csv')


def load_max_k_cores_train():
    return pd.read_csv(max_k_cores_train_fp, index_col='id')


def load_max_k_cores_test():
    return pd.read_csv(max_k_cores_test_fp, index_col='test_id')
############################################################3
############################################################3
############################################################3
glove_train_fp = os.path.join(data_folder, 'embeddings', 'glove_train.csv')
glove_test_fp = os.path.join(data_folder, 'embeddings', 'glove_test.csv')

def load_glove_metrics_train():
    return pd.read_csv(glove_train_fp, index_col='id')


def load_glove_metrics_test():
    return pd.read_csv(glove_test_fp, index_col='test_id')
############################################################3
############################################################3
############################################################3
lex_train_fp = os.path.join(data_folder, 'embeddings', 'lex_train.csv')
lex_test_fp = os.path.join(data_folder, 'embeddings', 'lex_test.csv')

def load_lex_metrics_train():
    return pd.read_csv(lex_train_fp, index_col='id')


def load_lex_metrics_test():
    return pd.read_csv(lex_test_fp, index_col='test_id')
############################################################3
############################################################3
############################################################3
word2vec_train_fp = os.path.join(data_folder, 'embeddings', 'word2vec_train.csv')
word2vec_test_fp = os.path.join(data_folder, 'embeddings', 'word2vec_test.csv')


def load_word2vec_metrics_train():
    return pd.read_csv(word2vec_train_fp, index_col='id')


def load_word2vec_metrics_test():
    return pd.read_csv(word2vec_test_fp, index_col='test_id')
############################################################3
############################################################3
############################################################3

aux_pairs_50_train_fp = os.path.join(data_folder, 'aux_pron', 'aux_pairs_50_train.csv')
aux_pairs_50_test_fp = os.path.join(data_folder, 'aux_pron', 'aux_pairs_50_test.csv')
aux_pair_target_freq = 'aux_pair_target_freq'
def load_aux_pairs_50_train():
    return pd.read_csv(aux_pairs_50_train_fp, index_col='id')[[aux_pair_target_freq]]

def load_aux_pairs_50_test():
    return pd.read_csv(aux_pairs_50_test_fp, index_col='test_id')[[aux_pair_target_freq]]

############################################################3
############################################################3
############################################################3
embedings_list=['word2vec', 'glove', 'lex']
column_types = ['tokens', 'lemmas']
kur_pairs=[
    ('kur_q1vec_{}_{}'.format(col_type,emb), 'kur_q2vec_{}_{}'.format(col_type,emb))
    for col_type in column_types for emb in embedings_list
    ]

skew_pairs=[
    ('skew_q1vec_{}_{}'.format(col_type,emb), 'skew_q2vec_{}_{}'.format(col_type,emb))
    for col_type in column_types for emb in embedings_list
    ]


def add_kur_combinations(df):
    for col1, col2 in kur_pairs+skew_pairs:
        name = col1.replace('q1', '')
        df['{}_abs_diff'.format(name)]=np.abs(df[col1]-df[col2])
        df['{}_1div2_ratio'.format(name)]= df[col1]/df[col2]
        df['{}_log_ratio'.format(name)]= np.abs(np.log(df[col1]/df[col2]))
        df['{}_q1_ratio'.format(name)]=df[col1]/(df[col1]+df[col2])
        df['{}_q2_ratio'.format(name)]=df[col2]/(df[col1]+df[col2])

############################################################3
############################################################3
############################################################3
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import json
from time import sleep
import traceback

gc_host = '35.185.55.5'
local_host = '10.20.0.144'


def load_train_all_xgb():
    train_df = pd.concat([
        load_train(),
        load_train_lengths(),
        load_train_common_words(),
        load__train_metrics(),
        load_train_tfidf_new(),
        load_train_magic(),
        load_wh_train(),
        load_one_upper_train(),
        load_topNs_avg_tok_freq_train(),
        load_abi_train(),
        load_max_k_cores_train(),
        load_word2vec_metrics_train(),
        load_glove_metrics_train(),
        load_lex_metrics_train(),
        load_aux_pairs_50_train()
    ], axis=1)

    cols_to_del = [qid1, qid2, question1, question2]
    for col in cols_to_del:
        del train_df[col]

    return train_df

def load_test_all_xgb():
    test_df = pd.concat([
        load_test_lengths(),
        load_test_common_words(),
        load__test_metrics(),
        load_test_tfidf_new(),
        load_test_magic(),
        load_wh_test(),
        load_one_upper_test(),
        load_topNs_avg_tok_freq_test(),
        load_abi_test(),
        load_max_k_cores_test(),
        load_word2vec_metrics_test(),
        load_glove_metrics_test(),
        load_lex_metrics_test(),
        load_aux_pairs_50_test()
    ], axis=1)


    return test_df

def plot_errors(imp):
    train_runs= [x['train'] for x in imp]
    test_runs= [x['test'] for x in imp]

    sz=len(train_runs[0])
    x_axis=range(sz)
    y_train = [np.mean([x[j] for x in train_runs]) for j in x_axis]
    y_test = [np.mean([x[j] for x in test_runs]) for j in x_axis]

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_train, label='train')
    ax.plot(x_axis, y_test, label='test')
    ax.legend()
    plt.show()

def xgboost_per_tree_results(estimator):
    results_on_test = estimator.evals_result()['validation_1']['logloss']
    results_on_train = estimator.evals_result()['validation_0']['logloss']
    return {
        'train': results_on_train,
        'test': results_on_test
    }

def out_loss(loss):
    print '====================================='
    print '====================================='
    print '====================================='
    print loss
    print '====================================='
    print '====================================='
    print '====================================='


def write_results(name,mongo_host, per_tree_res, losses, imp, features):
    from pymongo import MongoClient

    imp=[x.item() for x in imp]
    features=list(features)

    client = MongoClient(mongo_host, 27017)
    db = client['xgb_cv']
    collection = db[name]
    try:
        collection.insert_one({
            'results': per_tree_res,
            'losses': losses,
            'importance':imp,
            'features':features
        })
    except:
        print 'error in mongo'
        traceback.print_exc()
        raise
        # sleep(20)

def get_all_cols_except_target(df):
    return set([x for x in df.columns if x!=TARGET])

def fix_train_columns(train_df, test_df):
    if(get_all_cols_except_target(train_df))!=get_all_cols_except_target(test_df):
        raise Exception('SETS of columns train/test are different')
    else:
        print 'SETS of cols are equal'


    train_df=train_df[[TARGET]+[x for x in test_df.columns]]


    ok = list(train_df.columns[1:])==list(test_df.columns)
    if not ok:
        raise Exception('Features LISTS for train/test are different')

    return train_df



def submit_xgb(name):
    seed=42
    train_df = load_train_all_xgb()
    test_df = load_test_all_xgb()

    for df in [train_df, test_df]:
        del_trash_cols(df)
        add_kur_combinations(df)

    train_df = fix_train_columns(train_df, test_df)

    # train_df, test_df = train_df.head(5000), test_df.head(5000)


    print explore_target_ratio(train_df)
    # print explore_target_ratio(small)

    train_df, test_df = oversample_submit(train_df, test_df, seed)

    print explore_target_ratio(train_df)
    # print explore_target_ratio(small)

    train_target = train_df[TARGET]
    del train_df[TARGET]
    train_arr = train_df

    print train_df.columns.values
    test_arr = test_df

    estimator = xgb.XGBClassifier(n_estimators=2800,
                                  subsample=0.6,
                                  # colsample_bytree=0.8,
                                  max_depth=7,
                                  objective='binary:logistic',
                                  learning_rate=0.02,
                                  base_score=0.2,
                                  nthread=-1)
    print test_arr.columns.values
    print len(train_arr)
    print len(test_arr)
    estimator.fit(
        train_arr, train_target,
        eval_metric='logloss',
        verbose=True
    )

    proba = estimator.predict_proba(test_arr)
    classes = [x for x in estimator.classes_]
    print 'classes {}'.format(classes)
    test_df[TARGET] = proba[:, 1]

    res = test_df[[TARGET]]
    res.to_csv('{}.csv'.format(name), index=True, index_label='test_id')




name='submit_all_31'
submit_xgb(name)


#911, 972, 947, 1003, 911
print '============================'
print 'DONE!'
print '============================'

