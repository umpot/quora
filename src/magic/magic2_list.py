import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

data_folder = '../../data/'
fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'
question1, question2 = 'question1', 'question2'
new_col = 'q1_q2_intersect_list'

train_df =  pd.read_csv(fp_train, header=0)
test_df =  pd.read_csv(fp_test, header=0)

ques = pd.concat([train_df[['question1', 'question2']], \
                  test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')



q_dict = defaultdict(list)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].append(ques.question2[i])
    q_dict[ques.question2[i]].append(ques.question1[i])

# def q1_q2_intersect(row):
#     return len(
#         set(
#             q_dict[row['question1']]
#         ).intersection(
#             set(q_dict[row['question2']])
#         )
#     )


def multi_list_intersection_row(row):
    a=q_dict[row['question1']]
    b=q_dict[row['question2']]

    return multi_list_intersection(a,b)


def multi_list_intersection(a, b):
    inter = set(a).intersection(set(b))

    c_a = Counter(a)
    c_b = Counter(b)

    res=0
    for s in inter:
        res+=min(c_a[s], c_b[s])


    return res




train_df[new_col] = train_df.apply(multi_list_intersection_row, axis=1, raw=True)
test_df[new_col] = test_df.apply(multi_list_intersection_row, axis=1, raw=True)

#
# temp = train_orig.q1_q2_intersect.value_counts()
# sns.barplot(temp.index[:20], temp.values[:20])


magic2_train_fp = os.path.join(data_folder, 'magic', 'magic2_list_train.csv')
magic2_test_fp = os.path.join(data_folder, 'magic', 'magic2__list_test.csv')

new_cols=[new_col]
train_df[new_cols].to_csv(magic2_train_fp, index_label='id')
test_df[new_cols].to_csv(magic2_test_fp, index_label='test_id')


# train_feat = train_orig[['q1_q2_intersect']]
# test_feat = test_orig[['q1_q2_intersect']]


