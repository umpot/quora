

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
from nltk.corpus import stopwords

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

data_folder = '../../data/'

fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'

train_orig =  pd.read_csv(fp_train, header=0)
test_orig =  pd.read_csv(fp_test, header=0)

ques = pd.concat([train_orig[['question1', 'question2']], \
                  test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')
print ques.shape

stops = set(stopwords.words("english"))
def word_match_share(q1, q2, stops=None):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


q_dict = defaultdict(dict)
for i in range(ques.shape[0]):
    wm = word_match_share(ques.question1[i], ques.question2[i], stops=stops)
    q_dict[ques.question1[i]][ques.question2[i]] = wm
    q_dict[ques.question2[i]][ques.question1[i]] = wm

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))
def q1_q2_wm_ratio(row):
    q1 = q_dict[row['question1']]
    q2 = q_dict[row['question2']]
    inter_keys = set(q1.keys()).intersection(set(q2.keys()))
    if(len(inter_keys) == 0): return 0.
    inter_wm = 0.
    total_wm = 0.
    for q,wm in q1.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    for q,wm in q2.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    if(total_wm == 0.): return 0.
    return inter_wm/total_wm

train_orig['q1_q2_wm_ratio'] = train_orig.apply(q1_q2_wm_ratio, axis=1, raw=True)
test_orig['q1_q2_wm_ratio'] = test_orig.apply(q1_q2_wm_ratio, axis=1, raw=True)

train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

#Saving
#################################################
import os
q1_q2_word_match_ratio_train_fp=os.path.join(data_folder,'magic', 'q1_q2_word_match_ratio_train.csv')
train_orig['q1_q2_wm_ratio'].to_csv(q1_q2_word_match_ratio_train_fp, index_label='id')

q1_q2_word_match_ratio_test_fp=os.path.join(data_folder,'magic', 'q1_q2_word_match_ratio_test.csv')
test_orig['q1_q2_wm_ratio'].to_csv(q1_q2_word_match_ratio_test_fp, index_label='test_id')
#################################################

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
temp = train_orig.q1_q2_intersect.value_counts()
sns.barplot(temp.index[:20], temp.values[:20])
plt.subplot(1,2,2)
train_orig['q1_q2_wm_ratio'].plot.hist()


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.violinplot(x='is_duplicate', y='q1_q2_wm_ratio', data = train_orig)
plt.subplot(1,2,2)
sns.violinplot(x='is_duplicate', y='q1_q2_intersect', data = train_orig)



train_orig.plot.scatter(x='q1_q2_intersect', y='q1_q2_wm_ratio', figsize=(12,6))
print(train_orig[['q1_q2_intersect', 'q1_q2_wm_ratio']].corr())
# q1_q2_intersect  q1_q2_wm_ratio
# q1_q2_intersect         1.000000        0.684574
# q1_q2_wm_ratio          0.684574        1.000000

train_feat = train_orig[['q1_q2_intersect', 'q1_q2_wm_ratio']]
test_feat = test_orig[['q1_q2_intersect', 'q1_q2_wm_ratio']]


train_feat.to_csv('new_magic_train.csv')
test_feat.to_csv('new_magic_test.csv')