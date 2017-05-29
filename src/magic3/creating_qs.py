import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)


data_folder = '../../data/'
fp_train = data_folder + 'train.csv'
fp_test = data_folder + 'test.csv'


train_orig =  pd.read_csv(fp_train, header=0)
test_orig =  pd.read_csv(fp_test, header=0)




# "id","qid1","qid2","question1","question2","is_duplicate"
df_id1 = train_orig[["qid1", "question1"]].drop_duplicates(keep="first").copy().reset_index(drop=True)
df_id2 = train_orig[["qid2", "question2"]].drop_duplicates(keep="first").copy().reset_index(drop=True)

df_id1.columns = ["qid", "question"]
df_id2.columns = ["qid", "question"]

print(df_id1.shape, df_id2.shape)

df_id = pd.concat([df_id1, df_id2]).drop_duplicates(keep="first").reset_index(drop=True)
print(df_id1.shape, df_id2.shape, df_id.shape)


import csv
dict_questions = df_id.set_index('question').to_dict()
dict_questions = dict_questions["qid"]

new_id = 538000 # df_id["qid"].max() ==> 537933

def get_id(question):
    global dict_questions
    global new_id

    if question in dict_questions:
        return dict_questions[question]
    else:
        new_id += 1
        dict_questions[question] = new_id
        return new_id

rows = []
max_lines = 10
if True:
    with open('../input/test.csv', 'r', encoding="utf8") as infile:
        reader = csv.reader(infile, delimiter=",")
        header = next(reader)
        header.append('qid1')
        header.append('qid2')

        if True:
            print(header)
            pos, max_lines = 0, 10*1000*1000
            for row in reader:
                # "test_id","question1","question2"
                question1 = row[1]
                question2 = row[2]

                qid1 = get_id(question1)
                qid2 = get_id(question2)
                row.append(qid1)
                row.append(qid2)

                pos += 1
                if pos >= max_lines:
                    break
                rows.append(row)