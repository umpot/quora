import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem import WordNetLemmatizer
seed = 1024
np.random.seed(seed)
path = '../../data/'

train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")

counter = 0

def leamtize_str(x,lematizer=WordNetLemmatizer()):
    global counter
    if counter %1000 ==0:
        print counter
    counter+=1

    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = (" ").join([lematizer.lemmatize(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

lematizer = WordNetLemmatizer()


print('Generate lematizer')
train['question1_lematizer'] = train['question1'].astype(str).apply(lambda x:leamtize_str(x.lower(),lematizer))
test['question1_lematizer'] = test['question1'].astype(str).apply(lambda x:leamtize_str(x.lower(),lematizer))

train['question2_lematizer'] = train['question2'].astype(str).apply(lambda x:leamtize_str(x.lower(),lematizer))
test['question2_lematizer'] = test['question2'].astype(str).apply(lambda x:leamtize_str(x.lower(),lematizer))

train.to_csv(path+'train_lematizer.csv')
test.to_csv(path+'test_lematizer.csv')

