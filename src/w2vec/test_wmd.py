import gensim
import os

data_folder = '../../data/'

fp_model = os.path.join(data_folder, 'GoogleNews-vectors-negative300.bin')

def load_word2vec():
    model= gensim.models.KeyedVectors.load_word2vec_format(fp_model, binary=True)
    model.init_sims(replace=True) # normalizes vectors

    return model