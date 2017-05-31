import pandas as pd
import numpy as np
import seaborn as sns
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)


fp_train= '../../data/train.csv'
fp_test= '../../data/test.csv'

qid1,  qid2 = 'qid1',  'qid2'

BASIC_FEATURES=[
    'len_q1', 'len_q2', 'diff_len',
    'len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2',
    'common_words'
]

"""
Len:
len in words(with\without stopwords), chars(with\without spaces, punkt, spec symbols etc)
abs_diff_len, ratio_len, log(ratio_len) for different length
num of stopwords ?

=============================

Common words:

common num/ratio of tokens\stems\lemms with\without stop
================================

Tfid:
distances on tfidf ???
share of tfidf

==================================

word2vec, glove distances

norm_wmd/wmd


=================================
Unigrams \ bigrams \ treegrams
'unigram_jaccard'
'unigram_all_jaccard'   # count each char multiple times
'unigram_all_jaccard_max'  # norm by max

'trigram_jaccard'/'trigram_all_jaccard'/'trigram_all_jaccard_max'


'trigram_tfidf_cosine'
'trigram_tfidf_l2_euclidean'
'trigram_tfidf_l1_euclidean'
trigram_tf_l2_euclidean'

====================================
1wl_tfidf_cosine
1wl_tfidf_l2_euclidean



+6.639	trigram_all_jaccard
+3.131	bigram_all_jaccard_max
+1.075	unigram_all_jaccard_max
+1.017	m_w1l_tfidf_oof
+0.774	1wl_tfidf_cosine
+0.354	m_q1_q2_tf_oof
+0.266	bigram_jaccard
+0.244	trigram_tfidf_cosine
+0.221	m_vstack_svd_q1_q1_euclidean
+0.187	log_abs_diff_len1_len2
+0.141	len1
+0.130	1wl_tf_l2_euclidean
+0.024	unigram_jaccard
-0.010	trigram_tf_l2_euclidean
-0.044	m_vstack_svd_absdiff_q1_q2_oof
-0.074	m_vstack_svd_mult_q1_q2_oof
-0.099	m_q1_q2_tf_svd0
-0.099	m_q1_q2_tf_svd1
-0.105	m_q1_q2_tf_svd100_oof
-0.132	len2
-0.168	ratio_len1_len2
-0.339	m_vstack_svd_q1_q1_cosine
-0.351	abs_diff_len1_len2
-0.549	trigram_tfidf_l1_euclidean
-0.674	trigram_tfidf_l2_euclidean
-0.975	<BIAS>
-1.068	1wl_tfidf_l2_euclidean
-1.402	unigram_all_jaccard
-2.720	bigram_all_jaccard
-6.949	trigram_all_jaccard_max




'question1_nouns','question2_nouns','z_noun_match'

z_tfidf_sum1','z_tfidf_sum2','z_tfidf_mean1','z_tfidf_mean2'


'avg_world_len1', 'len_char_q1','len_word_q1','avg_world_len2','len_char_q2','len_word_q2',
'diff_avg_word','avg_world_len1','avg_world_len2'

	x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
"""

def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


data=None
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
data['diff_len'] = data.len_q1 - data.len_q2
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2'])