import pandas as pd
import os

from fuzzywuzzy.fuzz import QRatio, WRatio, \
    partial_ratio, partial_token_set_ratio, partial_token_sort_ratio, \
    token_set_ratio, token_sort_ratio

# same length
from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, minkowski, seuclidean, \
    braycurtis, chebyshev, correlation, mahalanobis

from scipy.stats import skew, kurtosis

# boolean

from scipy.spatial.distance import dice, kulsinski, jaccard, \
    rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# sets
import distance
# from distance import levenshtein, sorensen, jaccard, nlevenshtein

question1, question2 = 'question1', 'question2'
lemmas_q1, lemmas_q2 = 'lemmas_q1', 'lemmas_q2'
stems_q1, stems_q2 = 'stems_q1', 'stems_q2'
tokens_q1, tokens_q2 = 'tokens_q1', 'tokens_q2'
postag_q1, postag_q2 = 'postag_q1', 'postag_q2'
ner_q1, ner_q2 = 'ner_q1', 'ner_q2'


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


def load_train():
    return pd.read_csv(fp_train, index_col='id')

def load_test():
    return pd.read_csv(fp_test, index_col='test_id')

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


def load_train_tokens():
    df = pd.read_csv(tokens_train_fp, index_col='id')
    df = df.fillna('')
    return df

def load_test_tokens():
    df = pd.read_csv(tokens_test_fp, index_col='test_id')
    df = df.fillna('')
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

BOOL_METRICS = {x.__name__: x for x in
                [dice, kulsinski, jaccard]}
FUZZY_METRICS = {x.__name__: x for x in
                 [QRatio, WRatio, partial_ratio, partial_token_set_ratio,
                  partial_token_sort_ratio, token_set_ratio, token_sort_ratio]}

REAL_METRICS = {x.__name__: x for x in
                [cosine, cityblock, canberra, euclidean, seuclidean,
                 braycurtis, chebyshev, correlation, mahalanobis]}
REAL_METRICS['minkowski_3']=lambda x,y: minkowski(x,y, 3)
REAL_STATISTICS={x.__name__: x for x in [skew, kurtosis]}

SEQUENCES_METRICS = {x.__name__: x for x in
                [distance.levenshtein, distance.sorensen, distance.nlevenshtein]}
SEQUENCES_METRICS['distance.jaccard']=distance.jaccard


def generate_bool_vectors(a, b):
    a=set(a)
    b=set(b)
    join = set(a)
    join.update(b)
    join=list(join)
    join.sort()
    return [x in a for x in join], [x in b for x in join]




def process_sets_metrics(df, col1, col2, prefix, fp, index_label):
    df['bool_vectors'] = df[[col1, col2]].apply(lambda s: generate_bool_vectors(s[col1].split(), s[col2].split()), axis=1)
    new_cols = []
    for name, func in BOOL_METRICS.iteritems():
        print name
        new_col = '{}_{}'.format(prefix, name)
        new_cols.append(new_col)
        def wrap_func(s):
            try:
                return func(s[0], s[1])
            except:
                print '{}=null'.format(name)
                return -1

        df[new_col] = df['bool_vectors'].apply(wrap_func)

    df[new_cols].to_csv(fp, index_label=index_label)


def process_fuzzy_metrics(df, col1, col2, prefix, fp, index_label):
    new_cols = []
    for name, func in FUZZY_METRICS.iteritems():
        print name
        new_col = '{}_{}'.format(prefix, name)
        new_cols.append(new_col)
        df[new_col] = df[[col1, col2]].apply(lambda s: func(s[col1],s[col2]), axis=1)

    df[new_cols].to_csv(fp, index_label=index_label)

def process_sequence_metrics(df, col1, col2, prefix, fp, index_label):
    new_cols = []
    for name, func in SEQUENCES_METRICS.iteritems():
        print name
        new_col = '{}_{}'.format(prefix, name)
        new_cols.append(new_col)
        df[new_col] = df[[col1, col2]].apply(lambda s: func(s[col1],s[col2]), axis=1)

    df[new_cols].to_csv(fp, index_label=index_label)



def write_train_distances():
    index_label = 'id'

    df = load_train_lemmas()
    cols = [lemmas_q1, lemmas_q2]
    prefix='lemmas'

    # fp=os.path.join(data_folder, 'distances', 'train_metrics_sequence_{}.csv'.format(prefix))
    # process_sequence_metrics(df, cols[0], cols[1], prefix, fp, index_label)
    #
    # fp=os.path.join(data_folder, 'distances', 'train_metrics_bool_{}.csv'.format(prefix))
    # process_sets_metrics(df, cols[0], cols[1], prefix, fp, index_label)
    #
    # fp=os.path.join(data_folder, 'distances', 'train_metrics_fuzzy_{}.csv'.format(prefix))
    # process_fuzzy_metrics(df, cols[0], cols[1], prefix, fp, index_label)



    df = load_train_stems()
    cols = [stems_q1, stems_q2]
    prefix='stems'

    fp=os.path.join(data_folder, 'distances', 'train_metrics_sequence_{}.csv'.format(prefix))
    process_sequence_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'train_metrics_bool_{}.csv'.format(prefix))
    process_sets_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'train_metrics_fuzzy_{}.csv'.format(prefix))
    process_fuzzy_metrics(df, cols[0], cols[1], prefix, fp, index_label)


    df = load_train_tokens()
    cols = [tokens_q1, tokens_q2]
    for c in cols:
        df[c] = df[c].apply(lambda s: s.lower())
    prefix='tokens'

    fp=os.path.join(data_folder, 'distances', 'train_metrics_sequence_{}.csv'.format(prefix))
    process_sequence_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'train_metrics_bool_{}.csv'.format(prefix))
    process_sets_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'train_metrics_fuzzy_{}.csv'.format(prefix))
    process_fuzzy_metrics(df, cols[0], cols[1], prefix, fp, index_label)


def write_test_distances():
    index_label = 'test_id'

    df = load_test_lemmas()
    cols = [lemmas_q1, lemmas_q2]
    prefix='lemmas'

    fp=os.path.join(data_folder, 'distances', 'test_metrics_sequence_{}.csv'.format(prefix))
    process_sequence_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'test_metrics_bool_{}.csv'.format(prefix))
    process_sets_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'test_metrics_fuzzy_{}.csv'.format(prefix))
    process_fuzzy_metrics(df, cols[0], cols[1], prefix, fp, index_label)



    df = load_test_stems()
    cols = [stems_q1, stems_q2]
    prefix='stems'

    fp=os.path.join(data_folder, 'distances', 'test_metrics_sequence_{}.csv'.format(prefix))
    process_sequence_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'test_metrics_bool_{}.csv'.format(prefix))
    process_sets_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'test_metrics_fuzzy_{}.csv'.format(prefix))
    process_fuzzy_metrics(df, cols[0], cols[1], prefix, fp, index_label)


    df = load_test_tokens()
    cols = [tokens_q1, tokens_q2]
    for c in cols:
        df[c] = df[c].apply(lambda s: s.lower())
    prefix='tokens'

    fp=os.path.join(data_folder, 'distances', 'test_metrics_sequence_{}.csv'.format(prefix))
    process_sequence_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'test_metrics_bool_{}.csv'.format(prefix))
    process_sets_metrics(df, cols[0], cols[1], prefix, fp, index_label)

    fp=os.path.join(data_folder, 'distances', 'test_metrics_fuzzy_{}.csv'.format(prefix))
    process_fuzzy_metrics(df, cols[0], cols[1], prefix, fp, index_label)


write_train_distances()

