import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn import model_selection
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

trainDF = pd.read_csv('../input/train.csv')
trainDF = trainDF.dropna(how="any").reset_index(drop=True)

##########################################################33


featureExtractionStartTime = time.time()

maxNumFeatures = 300

# bag of letter sequences (chars)
BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures,
                                      analyzer='char', ngram_range=(1,2),
                                      binary=True, lowercase=True)
# bag of words
#BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=10, max_features=maxNumFeatures,
#                                      analyzer='word', ngram_range=(1,6), stop_words='english',
#                                      binary=True, lowercase=True)

BagOfWordsExtractor.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())

trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question1'])
trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question2'])
lables = np.array(trainDF.ix[:,'is_duplicate'])

featureExtractionDurationInMinutes = (time.time()-featureExtractionStartTime)/60.0
print("feature extraction took %.2f minutes" % (featureExtractionDurationInMinutes))
##########################################3333

crossValidationStartTime = time.time()

numCVSplits = 8
numSplitsToBreakAfter = 2

X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
#X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int) + \
#      trainQuestion1_BOW_rep.multiply(trainQuestion2_BOW_rep)
y = lables

logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag')

logRegAccuracy = []
logRegLogLoss = []
logRegAUC = []

print('---------------------------------------------')
stratifiedCV = model_selection.StratifiedKFold(n_splits=numCVSplits, random_state=2)
for k, (trainInds, validInds) in enumerate(stratifiedCV.split(X, y)):
    foldTrainingStartTime = time.time()

    X_train_cv = X[trainInds,:]
    X_valid_cv = X[validInds,:]

    y_train_cv = y[trainInds]
    y_valid_cv = y[validInds]

    logisticRegressor.fit(X_train_cv, y_train_cv)

    y_train_hat =  logisticRegressor.predict_proba(X_train_cv)[:,1]
    y_valid_hat =  logisticRegressor.predict_proba(X_valid_cv)[:,1]

    logRegAccuracy.append(accuracy_score(y_valid_cv, y_valid_hat > 0.5))
    logRegLogLoss.append(log_loss(y_valid_cv, y_valid_hat))
    logRegAUC.append(roc_auc_score(y_valid_cv, y_valid_hat))

    foldTrainingDurationInMinutes = (time.time()-foldTrainingStartTime)/60.0
    print('fold %d took %.2f minutes: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (k+1,
                                                                                       foldTrainingDurationInMinutes, logRegAccuracy[-1],logRegLogLoss[-1],logRegAUC[-1]))

    if (k+1) >= numSplitsToBreakAfter:
        break


crossValidationDurationInMinutes = (time.time()-crossValidationStartTime)/60.0

print('---------------------------------------------')
print('cross validation took %.2f minutes' % (crossValidationDurationInMinutes))
print('mean CV: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (np.array(logRegAccuracy).mean(),
                                                                 np.array(logRegLogLoss).mean(),
                                                                 np.array(logRegAUC).mean()))
print('---------------------------------------------')