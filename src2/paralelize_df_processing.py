import subprocess
import pandas as pd
import sys

df=None

def init():
    global df
    df=None

init()

def split_df(df, n):
    l = len(df)
    l=l/n




def func(df):
    pass


def paralelize_df(df, func, n_threads, args):
    pass

def run_inner(n_threads, i):
    pass


def run(args):
    if len(args)==1:
        pass
    else:
        pass



run(sys.argv())