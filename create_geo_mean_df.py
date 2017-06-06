import numpy as np
import pandas as pd

TARGET = 'is_duplicate'

def do_work(names, out_name):
    dfs = [pd.read_csv(fp, index_col='test_id') for fp in names]


