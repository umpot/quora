import os
import pandas as pd
TARGET = 'is_duplicate'

from sklearn.metrics import log_loss


def gather_and_calculate_loss(name):
    fp_in = os.path.join('/home/ubik/Desktop', name)
    fp_out=os.path.join('../../stacking_data/', 'stacking_{}'.format(name), 'probs.csv')
    dfs=[]
    for l in os.listdir(fp_in):
        if os.path.isdir(os.path.join(fp_in, l)):
            fp = os.path.join(fp_in, l, 'probs.csv')
            df = pd.read_csv(fp, index_col='id')
            bl = df.copy()
            bl['z'] = 1-bl['prob']
            loss = log_loss(bl[TARGET], bl[['z', 'prob']])
            print loss
            dfs.append(df)

    df= pd.concat(dfs)
    # df.to_csv(fp_out, index_label='id')
    return df
