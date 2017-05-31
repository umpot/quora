import json
import math
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyperopt.mongoexp import MongoTrials
from numpy import mean, std
from pymongo import MongoClient
from scipy.stats import normaltest
import pandas as pd
from sklearn.metrics import log_loss
import os

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)

gc_host = '104.197.97.20'
local_host = '10.20.0.144'
user='ubik'
password='nfrf[eqyz'

host = gc_host

client=MongoClient(host, 27017)
client['admin'].authenticate(user, password)


def plot_errors(name, fold=0):
    db = client['xgb_cv']
    results = db[name]
    results = [x['results'] for x in results.find()]
    train_runs= [x['train'] for x in results]
    test_runs= [x['test'] for x in results]

    sz=min(len(train_runs[fold]), len(test_runs[fold]))
    x_axis=range(sz)
    y_train = train_runs[fold]
    y_test = test_runs[fold]

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_train, label='train')
    ax.plot(x_axis, y_test, label='test')
    ax.legend()


def explore_importance(name, fold=0):
    db = client['xgb_cv']
    results = db[name]
    features = [x['features'] for x in results.find()][fold]

    importance = [x['importance'] for x in results.find()][fold]
    res = zip(features, importance)
    res.sort(key=lambda s:s[1], reverse=True)
    print res
    # res=res[:N]
    xs = [x[0] for x in res]
    ys=[x[1] for x in res]
    sns.barplot(xs, ys, orient='v')
    sns.plt.show()


def get_losses(name, fold=4):
    db = client['xgb_cv']
    results = db[name]
    losses = [x['losses'] for x in results.find()][fold]

    return losses