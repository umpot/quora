import os
import sys

import pandas as pd
import seaborn as sns

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

tfidf_with_stops_train_fp = os.path.join(data_folder,'tfidf','old' ,'tokens_with_stop_words_tfidf_train.csv')
tfidf_with_stops_test_fp = os.path.join(data_folder,'tfidf','old','tokens_with_stop_words_tfidf_test.csv')

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

######################################################################################
######################################################################################
######################################################################################
######################################################################################
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
    "w_share_ratio_2_smooth_idf_tokens_lower_no_stops"
]


def del_trash_cols(df):
    for col in trash_cols:
        if col in df:
            del df[col]

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
        if col1 not in df.columns:
            continue

        name = col1.replace('q1', '')
        df['{}_abs_diff'.format(name)]=np.abs(df[col1]-df[col2])
        df['{}_1div2_ratio'.format(name)]= df[col1]/df[col2]
        df['{}_log_ratio'.format(name)]= np.abs(np.log(df[col1]/df[col2]))
        df['{}_q1_ratio'.format(name)]=df[col1]/(df[col1]+df[col2])
        df['{}_q2_ratio'.format(name)]=df[col2]/(df[col1]+df[col2])


def preprocess_df(df):
    del_trash_cols(df)
    add_kur_combinations(df)

    # def blja(s):
    #     if s==None or s!=s or np.isnan(s) or np.isposinf(s) or np.isneginf(s):
    #         return -1
    #     return s
    # for col in df.columns:
    #     df[col] = df[col].apply(blja)

######################################################################################
######################################################################################
######################################################################################
######################################################################################
folds_fp = os.path.join(data_folder, 'top_k_freq', 'folds.json')

def load_folds():
    folds = json.load(open(folds_fp))
    folds = [(int(k),(v['train'], v['test'])) for k,v in folds.iteritems()]
    folds = [x[1] for x in folds]

    return folds


def create_folds(df):
    folds = load_folds()

    return [
        (df.loc[folds[str(x)]['train']], df.loc[folds[str(x)]['test']])
        for x in range(len(folds))]

def shuffle_df(df, random_state=42):
    np.random.seed(random_state)
    return df.iloc[np.random.permutation(len(df))]
######################################################################################
######################################################################################
######################################################################################
######################################################################################
T=None
def start():
    global T
    T = time()

def end(message=''):
    t = time()-T
    print '========================================================='
    print '========================================================='
    print '\n'
    print 'Time of {} = {}'.format(message, int(t))
    print '\n'
    print '========================================================='
    print '========================================================='

######################################################################################
######################################################################################
######################################################################################
######################################################################################

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

def oversample_df(df, l, random_state):
    df_pos = df[df[TARGET]==1]
    df_neg = df[df[TARGET]==0]

    df_neg_sampl = df_neg.sample(l, random_state=random_state, replace=True)

    df=pd.concat([df_pos, df_neg, df_neg_sampl])
    df = shuffle_df(df, random_state)

    return df

def oversample(train_df, test_df, random_state):
    l_train = int(delta * len(train_df))
    l_test = int(delta * len(test_df))

    return oversample_df(train_df, l_train, random_state), oversample_df(test_df, l_test, random_state)



############################################################3
############################################################3
############################################################3
import subprocess
from time import gmtime, strftime, time


def get_time_str():
    return strftime("%Y_%m_%d__%H_%M_%S", gmtime())

def push_to_gs(name, descr):
    with open('descr.txt', 'w+') as f:
        f.writelines([descr])

    if name in os.listdir('.'):
        t=get_time_str()
        subprocess.call(['mv', name, '{}_{}'.format(name, t)])


    script_name = os.path.basename(__file__)
    subprocess.call(['python', '-u', 'compress_and_push_to_gs.py', name, script_name])

######################################################################################
######################################################################################
######################################################################################
######################################################################################
gc_host = '104.197.97.20'
local_host = '10.20.0.144'
user='ubik'
password='nfrf[eqyz'

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


def done():
    print '============================'
    print 'DONE!'
    print '============================'


def write_results(name,mongo_host, per_tree_res, losses, imp, features):
    from pymongo import MongoClient

    imp=[x.item() for x in imp]
    features=list(features)

    client = MongoClient(mongo_host, 27017)
    client['admin'].authenticate(user, password)
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


def push_results_to_mongo(estimator, losses, mongo_host, name, test_arr, test_target, train_arr, proba):
    loss = log_loss(test_target, proba)
    out_loss(loss)
    losses.append(loss)
    per_tree_res = []
    ii = estimator.feature_importances_
    write_results(name, mongo_host, per_tree_res, losses, ii, train_arr.columns)

    out_loss('avg = {}'.format(np.mean(losses)))


######################################################################################
######################################################################################
######################################################################################
######################################################################################

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


######################################################################################
######################################################################################
######################################################################################
######################################################################################

def load__train_metrics():
    dfs = [pd.read_csv(fp, index_col='id') for fp in TRAIN_METRICS_FP]
    return pd.concat(dfs, axis=1)

def load__test_metrics():
    dfs = [pd.read_csv(fp, index_col='test_id') for fp in TEST_METRICS_FP]
    return pd.concat(dfs, axis=1)
######################################################################################
######################################################################################
######################################################################################
######################################################################################
def load_train_common_words():
    df = pd.read_csv(common_words_train_fp, index_col='id')
    return df

def load_test_common_words():
    df = pd.read_csv(common_words_test_fp, index_col='test_id')
    return df



######################################################################################
######################################################################################
######################################################################################
######################################################################################
def load_train_lengths():
    df = pd.read_csv(length_train_fp, index_col='id')
    return df

def load_test_lengths():
    df = pd.read_csv(length_test_fp, index_col='test_id')
    return df

######################################################################################
######################################################################################
######################################################################################
######################################################################################


def load_train_tfidf():
    df = pd.read_csv(tfidf_with_stops_train_fp, index_col='id')
    return df

def load_test_tfidf():
    df = pd.read_csv(tfidf_with_stops_test_fp, index_col='test_id')
    return df


######################################################################################
######################################################################################
######################################################################################
######################################################################################
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
upper_keywords_fp_train=os.path.join(data_folder, 'keywords', 'train_upper.csv')
upper_keywords_test=os.path.join(data_folder, 'keywords', 'test_upper.csv')

def load_upper_keywords_train():
    df = pd.read_csv(upper_keywords_fp_train, index_col='id')
    return df

def load_upper_keywords_test():
    df = pd.read_csv(upper_keywords_test, index_col='test_id')
    return df


######################################################################################
######################################################################################
######################################################################################
######################################################################################
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

############################################################3
############################################################3
############################################################3
abi_train_fp = os.path.join(data_folder, 'abishek', 'abi_train.csv')
abi_test_fp = os.path.join(data_folder, 'abishek', 'abi_test.csv')


def load_abi_train():
    return pd.read_csv(abi_train_fp, index_col='id')

def load_abi_test():
    return pd.read_csv(abi_test_fp, index_col='test_id')

############################################################3
############################################################3
############################################################3
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

train_pos_metrics_fp=os.path.join(data_folder, 'pos_metrics', 'train_pos_metrics.csv')
test_pos_metrics_fp=os.path.join(data_folder, 'pos_metrics', 'test_pos_metrics.csv')

def load_metrics_on_pos_train():
    return pd.read_csv(train_pos_metrics_fp, index_col='id')

def load_metrics_on_pos_test():
    return pd.read_csv(train_pos_metrics_fp, index_col='test_id')

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
import xgboost as xgb
from sklearn.metrics import log_loss
import json
import traceback



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


#STACKING
################################################3
################################################3
from sklearn.ensemble import RandomForestClassifier


def get_update_df():
    df = load_train()
    cols_to_del = [qid1, qid2, question1, question2]
    for col in cols_to_del:
        del df[col]

    return df

def perform_xgb_cv(name, mongo_host):
    seed = 42
    df = load_train_all_xgb()


    update_df = get_update_df()
    preprocess_df(df)

    df.replace([None, np.inf, -np.inf, np.nan, float('inf'), float('-inf')], -1, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    df.fillna(-1, inplace=True)
    df = df.apply(lambda x: np.float32(x))
    df.replace([None, np.inf, -np.inf, np.nan, float('inf'), float('-inf')], -1, inplace=True)

    folds = load_folds()

    losses = []
    counter = 0

    for big_ind, small_ind in folds:
        start()

        big = df.iloc[big_ind]
        small = df.iloc[small_ind]

        # big, small = big.head(1000), small.head(1000)


        print explore_target_ratio(big)
        print explore_target_ratio(small)

        big, small = oversample(big, small, seed)

        print explore_target_ratio(big)
        print explore_target_ratio(small)

        train_target = big[TARGET]
        del big[TARGET]
        train_arr = big

        test_target = small[TARGET]
        del small[TARGET]
        test_arr = small

        # estimator = xgb.XGBClassifier(n_estimators=2800,
        #                               subsample=0.6,
        #                               # colsample_bytree=0.8,
        #                               max_depth=7,
        #                               objective='binary:logistic',
        #                               learning_rate=0.02,
        #                               base_score=0.2,
        #                               nthread=-1)


        estimator = RandomForestClassifier(n_estimators=1000, verbose=10, n_jobs=-1)

        print test_arr.columns.values
        print len(train_arr)
        print len(test_arr), len(test_arr.index), len(set(test_arr.index))

        eval_set = [(train_arr, train_target), (test_arr, test_target)]

        estimator.fit(
            train_arr, train_target
        )

        proba = estimator.predict_proba(test_arr)
        print len(proba[:,1])
        print len(test_arr)

        test_arr['prob'] = proba[:,1]

        test_arr = test_arr[~test_arr.index.duplicated(keep='first')]

        update_df.loc[test_arr.index, 'prob']=test_arr.loc[test_arr.index, 'prob']

        loss = log_loss(test_target, proba)
        print 'Logloss {}'.format(loss)
        push_results_to_mongo(estimator, losses,
                              mongo_host, name, test_arr, test_target, train_arr, proba)

        end('fold {}'.format(counter))
        counter+=1


    update_df.to_csv('probs.csv', index_label='id')




descr= \
    """

    """


name='stacking_random_forest_light'


perform_xgb_cv(name, gc_host)
push_to_gs(name, descr)

done()


#STACKING
################################################3
################################################3




