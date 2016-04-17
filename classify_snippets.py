'''
Trains a LSTM classifier on Datafox Snippets.
Created by Derek C based on Keras examples.
'''
import numpy as np
import pandas as pd
import time as tm
import re
#  np.random.seed(14)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.sequence import pad_sequences

env = 'local'
# max_features = 19295 as determined by data, was originally 20000
# max_len = 105 as determined by data, was originally 80
batch_size = 32
epochs = 20

def vectorize(X, y, word_idx, max_snippet_len):
  X_vectors = map(lambda (snippet): [word_idx[w] for w in snippet], X)
  padded_X = pad_sequences(X_vectors, maxlen=max_snippet_len)

  label_index = dict((label, i) for i, label in enumerate(list(set(y))))
  y_vectors = [label_index[w] for w in y]
  one_hot_y = np_utils.to_categorical(y_vectors)

  return (padded_X, one_hot_y)

print('Loading data...')
initial_time = tm.time()
data = pd.read_pickle('data/training_data.p')
if env == 'local':
  data = data[0:5000]
X_raw = data['Content'].tolist()
y_raw = data['Tags'].tolist()
print "------------- Loaded in %0.2fs ---------------" % (tm.time() - initial_time)

print('Finding unique values ...')
checkpoint = tm.time()
vocab = reduce(lambda x, y: x | y, (set(snippet) for snippet in X_raw))
vocab_size = len(vocab) + 1
max_snippet_len = max(map(len, X_raw))
print "------------- Prepared in %0.2fs ---------------" % (tm.time() - checkpoint)

print('Vocabulary size: %d unique words') % vocab_size
print('Max snippet length: %d words') % max_snippet_len
print('Number of training inputs: %d') % len(X_raw)
print('Number of training labels: %d') % len(y_raw)

print('Vectorizing the word sequences...')
# the i++ is to reserve 0 for masking via pad_sequences
word_idx = dict((word, i + 1) for i, word in enumerate(vocab))
X_train, y_train = vectorize(X_raw, y_raw, word_idx, max_snippet_len)

# num = np.random.randint(0,5000)
# print('Here\'s a random snippet with its label and vectors (%d):') % num
# print X_raw[num]
# print y_raw[num]
# print X_train[num]
# print y_train[num]

print len(y_train)
print y_train[4:6]
print len(y_train[6])
print type(y_train[6])

print('Building and compiling model...')
checkpoint = tm.time()
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_snippet_len, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(21))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', metrics=['accuracy'])
print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)

print('Training phase ...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
        validation_split=0.2 , verbose=2)