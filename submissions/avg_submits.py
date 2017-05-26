import pandas as pd
import numpy as np
TARGET = 'is_duplicate'
TEST_ID = 'test_id'

def load_submission(fp):
    return pd.read_csv(fp, index_col='test_id')

def create_mean_submit(fps, name):
    dfs=[load_submission(fp) for fp in fps]

    df = sum(dfs)/len(dfs)

    df.to_csv('{}.csv'.format(name))


def create_geo_mean_submit(fps, name):
    dfs=[load_submission(fp).apply(np.log) for fp in fps]

    df = sum(dfs)/len(dfs)

    df = df.apply(np.exp)

    df.to_csv('{}.csv'.format(name))


fps=[
    'submit_abi_0.8_0.8_5_1000.csv',
    'submit_with_bi_pref_bay_freq_0.8_0.8_5_1150.csv'
]

name = 'abi_bi_pref_mean'

create_mean_submit(fps, name)