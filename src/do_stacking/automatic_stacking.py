from time import time
import subprocess

import numpy as np
from numpy import random

T=None
def start():
    global T
    T = time()

def end(message=''):
    t = time()-T
    print '========================================================='
    print '========================================================='
    print '\n'
    print 'Time of {} = {}'.format(message, int(t))
    print '\n'
    print '========================================================='
    print '========================================================='

#name, max_depth, learning_rate, subsample, colsample_bytree

seed = int(time())
random.seed(seed)

def generate_rnd_params():
    max_depth = random.randint(2,9)
    learning_rate = random.choice([0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.0005, 0.003, 0.007])
    subsample = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    colsample_bytree = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    return max_depth, learning_rate, subsample, colsample_bytree


name = 'with_many_weak_est'

def do_search():
    counter=0

    while True:
        start()
        max_depth, learning_rate, subsample, colsample_bytree = generate_rnd_params()
        subprocess.call(['python', '-u', name, str(max_depth), str(learning_rate), str(subsample), str(colsample_bytree)])
        counter+=1
        end('Evaluating {}'.format(counter))

do_search()