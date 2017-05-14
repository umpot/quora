import pandas as pd
import numpy as np
import seaborn as sns

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

fp_train= '../../data/train.csv'
fp_test= '../../data/test.csv'

qid1,  qid2 = 'qid1',  'qid2'


def load_train_test():
    return pd.read_csv(fp_train, index_col='id'), pd.read_csv(fp_test, index_col='test_id')

train_df, test_df = load_train_test()