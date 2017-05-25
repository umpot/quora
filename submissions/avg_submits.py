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
    'submit_with_bay_tok_freq_0.8_0.8_5_1150.csv',
    'upper_1600_0.8_0.8_5_seed42.csv'
]

name = 'blja'

create_geo_mean_submit(fps, name)