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

    print l_train, l_test

    return oversample_df(train_df, l_train, random_state), oversample_df(test_df, l_test, random_state)



############################################################3
############################################################3
############################################################3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from fastFM.sgd import FMClassification
from pyfm import pylibfm
import xgboost as xgb
import pywFM

# train_df, test_df = load_train(), load_test()

def test_pywfm_on_bow():
    df = load_train()

    folds = create_folds(df)

    train, test = folds[0]

    train, test = oversample(train, test, 42)

    questions = list(train[question1])+list(train[question2])
    print 'Creating Vectorizer...'
    c = CountVectorizer(questions, binary=True, stop_words='english')
    print 'Fitting Vectorizer...'
    c.fit(questions)

    train_arr_q1 = c.transform(train[question1])
    train_arr_q2 = c.transform(train[question2])

    train_arr = train_arr_q1+train_arr_q2
    train_arr[train_arr==2]=1
    train_arr[train_arr==1]=-1

    test_arr_q1 = c.transform(test[question1])
    test_arr_q2 = c.transform(test[question2])

    test_arr = test_arr_q1+test_arr_q2
    test_arr[test_arr==2]=1
    test_arr[test_arr==1]=-1

    train_target = train[TARGET]
    test_target = test[TARGET]


    fm = pywFM.FM(task='classification',
                  num_iter=100,
                  verbose=10,r1_regularization=0.1, learn_rate=0.1)

    res = fm.run(train_arr, train_target, test_arr, test_target)
    prob = res.predictions
    prob_0 = [1-x for x in prob]
    # return res

    proba = np.array([prob_0, prob]).reshape(len(prob), 2)

    loss = log_loss(test[TARGET], proba)
    print loss

    print loss

def test_xgb_on_bow():
    df = load_train()

    folds = create_folds(df)

    train, test = folds[0]

    train, test = oversample(train, test, 42)

    questions = list(train[question1])+list(train[question2])
    print 'Creating Vectorizer...'
    c = CountVectorizer(questions, binary=True, stop_words='english')
    print 'Fitting Vectorizer...'
    c.fit(questions)

    train_arr_q1 = c.transform(train[question1])
    train_arr_q2 = c.transform(train[question2])

    train_arr = train_arr_q1+train_arr_q2
    train_arr[train_arr==2]=1
    train_arr[train_arr==1]=-1

    test_arr_q1 = c.transform(test[question1])
    test_arr_q2 = c.transform(test[question2])

    test_arr = test_arr_q1+test_arr_q2
    test_arr[test_arr==2]=1
    test_arr[test_arr==1]=-1

    xgb_params = {
        'objective': 'binary:logistic',
        'booster': 'gblinear',
        'eval_metric': 'logloss',
        'lambda':0.1,
        'eta': 0.01,
        # 'max_depth': 3,
        'subsample': 0.1,
        'colsample_bytree': 0.1,
        # 'min_child_weight': 5,
        'silent': 1
    }

    dTrain = xgb.DMatrix(train_arr, train[TARGET])
    dVal = xgb.DMatrix(test_arr, test[TARGET])

    res={}

    bst = xgb.train(xgb_params, dTrain,
                    1000,
                    [(dTrain,'train'), (dVal,'val')],
                verbose_eval=1,
                    early_stopping_rounds=50,
                    evals_result=res)
    loss = res['train']['logloss']
    print loss


def test_fastfmon_bow():
    df = load_train()

    folds = create_folds(df)

    train, test = folds[0]

    train, test = oversample(train, test, 42)
    for bl in [train, test]:
        bl[TARGET] = bl[TARGET].apply(lambda s:-1 if s==0 else 1)

    questions = list(train[question1])+list(train[question2])
    print 'Creating Vectorizer...'
    c = CountVectorizer(questions, binary=True, stop_words='english')
    print 'Fitting Vectorizer...'
    c.fit(questions)

    train_arr_q1 = c.transform(train[question1])
    train_arr_q2 = c.transform(train[question2])

    train_arr = train_arr_q1+train_arr_q2
    train_arr[train_arr==2]=1
    train_arr[train_arr==1]=-1

    test_arr_q1 = c.transform(test[question1])
    test_arr_q2 = c.transform(test[question2])

    test_arr = test_arr_q1+test_arr_q2
    test_arr[test_arr==2]=1
    test_arr[test_arr==1]=-1


    model = FMClassification(n_iter=1000, l2_reg=0.1)
    print 'Fitting model...'
    model.fit(train_arr, train[TARGET])


    proba = model.predict_proba(test_arr)

    loss = log_loss(test[TARGET], proba)
    print loss


def test_log_reg_on_bow():
    df = load_train()

    folds = create_folds(df)

    train, test = folds[0]
    train, test = oversample(train, test, 42)

    questions = list(train[question1])+list(train[question2])
    print 'Creating Vectorizer...'
    c = CountVectorizer(questions,
                        binary=True,
                        stop_words='english',
                        max_features=50000,
                        ngram_range=(1,2)
                        )
    print 'Fitting Vectorizer...'
    c.fit(questions)

    train_arr_q1 = c.transform(train[question1])
    train_arr_q2 = c.transform(train[question2])

    train_arr = train_arr_q1+train_arr_q2
    train_arr[train_arr==2]=1
    train_arr[train_arr==1]=-1

    test_arr_q1 = c.transform(test[question1])
    test_arr_q2 = c.transform(test[question2])

    test_arr = test_arr_q1+test_arr_q2
    test_arr[test_arr==2]=1
    test_arr[test_arr==1]=-1

    model = LogisticRegression(verbose=10, n_jobs=-1, penalty='l2')
    print 'Fitting model...'
    model.fit(train_arr, train[TARGET])
    proba = model.predict_proba(test_arr)

    loss = log_loss(test[TARGET], proba)
    print loss


test_log_reg_on_bow()

