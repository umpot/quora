import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
seed = 1024
np.random.seed(seed)
path = '../../data/'
import os

train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")

counter=0
def stem_str(x,stemmer=SnowballStemmer('english')):
    global counter
    counter+=1
    if counter%1000 == 0:
        print counter
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

porter = PorterStemmer()
snowball = SnowballStemmer('english')


print('Generate porter')
train['question1_porter'] = train['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question1_porter'] = test['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

train['question2_porter'] = train['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question2_porter'] = test['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

train.to_csv(os.path.join(path, 'nlp', 'stems_train.csv'), index_label='id')
test.to_csv(os.path.join(path, 'nlp', 'stems_test.csv'), index_label='test_id')

