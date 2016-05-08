'''
Trains a LSTM classifier on Datafox Snippets.
Created by Derek C based on Keras examples.
'''
import numpy as np
import pandas as pd
import time as tm
import re
import random

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.sequence import pad_sequences

env = 'local'
np.random.seed(14)
batch_size = 100 if env == 'local' else 32
epochs = 2 if env == 'local' else 20

cell_type = ['LSTM', 'GRU']
recurrent_dropout = [True, False]
unroll_dropout = [0.1, 0.3, 0.5]
optimizers = ['adam', 'rmsprop']
layers = [1, 2, 3]
cells_per_layer = [128, 256, 512]

def vectorize(X, y, word_idx, max_snippet_len):
  X_vectors = map(lambda (snippet): [word_idx[w] for w in snippet], X)
  padded_X = pad_sequences(X_vectors, maxlen=max_snippet_len)

  label_index = dict((label, i) for i, label in enumerate(list(set(y))))
  y_vectors = [label_index[w] for w in y]
  one_hot_y = np_utils.to_categorical(y_vectors)

  return (padded_X, one_hot_y)

def build_combinations():
  combos = []
  # Not pretty, but effective
  for cell in cell_type:
    for flag in recurrent_dropout:
      for step in unroll_dropout:
        for optimizer in optimizers:
          for layer in layers:
            for count in cells_per_layer:
              combos.append([cell, flag, step, optimizer, layer, count])
  # Limit the number of permutations for now so we don't spend forever training
  combinations = random.sample(combos, 5)
  return combinations


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

print('Vectorizing the word sequences...')
# the i++ is to reserve 0 for masking via pad_sequences
word_idx = dict((word, i + 1) for i, word in enumerate(vocab))
X_train, y_train = vectorize(X_raw, y_raw, word_idx, max_snippet_len)

'''
print('Vocabulary size: %d unique words') % vocab_size
print('Max snippet length: %d words') % max_snippet_len
print('Number of training inputs: %d') % len(X_raw)
print('Number of training labels: %d') % len(y_raw)

num = np.random.randint(0,5000)
print('Here\'s a random snippet with its label and vectors (%d):') % num
print X_raw[num]
print y_raw[num]
print X_train[num]
print y_train[num]

print len(y_train)
print y_train[4:6]
print len(y_train[6])
print type(y_train[6])

'''
combinations = build_combinations()

for combo in combinations:
  print("Building and compiling model with variables: %s" % combo)
  checkpoint = tm.time()
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length=max_snippet_len, dropout=0.2))
  if combo[0] == 'LSTM':
    model.add(LSTM(128, dropout_W=0.4, dropout_U=0.0))
  elif combo[0] == 'GRU':
    model.add(GRU(128, dropout_W=0.4, dropout_U=0.0))
  model.add(Dense(21))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',optimizer=combo[3], metrics=['accuracy'])
  print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)

  print('Training phase ...')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
          validation_split=0.2, verbose=2)