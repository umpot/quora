import numpy as np

from keras.layers import Input, Dense, Convolution2D, Flatten

from keras.models import Model

import keras.backend as K

input1_tensor = lyr.Input(X1_val.shape[1:])

input2_tensor = lyr.Input(X2_val.shape[1:])

words_embedding_layer = lyr.Embedding(X1_train.max() + 1, 100)

seq_embedding_layer = model.layers[3]#model is the fitted model

seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))

merge_layer = multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])

features_model = Model([input1_tensor, input2_tensor], merge_layer)

features_model.compile(loss='mse', optimizer='adam')

z = features_model.predict([X1_val,X2_val])