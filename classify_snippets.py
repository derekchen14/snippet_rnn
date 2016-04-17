'''
Trains a LSTM classifier on Datafox Snippets.
Created by Derek C based on Keras examples.
'''
import numpy as np
import pandas as pd
import time as tm
import re
#  np.random.seed(14)  # for reproducibility

# from keras.preprocessing import sequence
from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.sequence import pad_sequences

env = 'local'
max_features = 20000
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

def vectorize(X, y, word_idx, max_snippet_len):
  X_vectors = map(lambda (snippet): [word_idx[w] for w in snippet], X)
  padded_X = pad_sequences(X_vectors, maxlen=max_snippet_len)

  label_index = dict((label, i+1) for i, label in enumerate(list(set(y))))
  y_vectors = [label_index[w] for w in y]
  one_hot_y = np_utils.to_categorical(y_vectors)  # might revisit if index 0 is reserved

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

# Reserve 0 for masking via pad_sequences
print('Vocabulary size: %d unique words') % vocab_size
print('Max snippet length: %d words') % max_snippet_len
print('Number of training inputs: %d') % len(X_raw)
print('Number of training labels: %d') % len(y_raw)

print('Vectorizing the word sequences...')
word_idx = dict((word, i + 1) for i, word in enumerate(vocab))
X_train, y_train = vectorize(X_raw, y_raw, word_idx, max_snippet_len)

num = np.random.randint(0,5000)
print('Here\'s a random snippet with its label and vectors (%d):') % num
print(X_raw[num])
print(y_raw[num])
print X_train[num]
print y_train[num]

# print('-')
# print('inputs: integer tensor of shape (samples, max_length)')
# print('inputs_train shape:', inputs_train.shape)
# print('inputs_test shape:', inputs_test.shape)
# print('-')
# print('queries: integer tensor of shape (samples, max_length)')
# print('queries_train shape:', queries_train.shape)
# print('queries_test shape:', queries_test.shape)
# print('-')
# print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
# print('answers_train shape:', answers_train.shape)
# print('answers_test shape:', answers_test.shape)
# print('-')
# print('Compiling...')


# print('Pad sequences (samples x time)')
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)

# print('Build model...')
# model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
# model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# print('Train...')
# print(X_train.shape)
# print(y_train.shape)
# model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
#           validation_data=(X_test, y_test))
# score, acc = model.evaluate(X_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)