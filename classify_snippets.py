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
from keras.callbacks import Callback, EarlyStopping
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.sequence import pad_sequences

env = 'test'
np.random.seed(14)
batch_size = 100 if env == 'local' else 32
epochs = 7 if env == 'local' else 15

cell_type = ['LSTM', 'GRU'] # https://arxiv.org/abs/1412.3555
input_gate_dropout = [0.1, 0.3, 0.5] # http://arxiv.org/abs/1512.05287
cells_per_layer = [64, 128, 256] # https://github.com/karpathy/char-rnn
multilayer_rnn = [True, False]
recurrent_dropout = [0.0, 0.2] # http://arxiv.org/abs/1409.2329
optimizers = ['adam', 'rmsprop'] # http://arxiv.org/abs/1412.6980

def vectorize(X, y, word_idx, max_snippet_len):
  X_vectors = map(lambda (snippet): [word_idx[w] for w in snippet], X)
  padded_X = pad_sequences(X_vectors, maxlen=max_snippet_len)

  label_index = dict((label, i) for i, label in enumerate(list(set(y))))
  y_vectors = [label_index[w] for w in y]
  one_hot_y = np_utils.to_categorical(y_vectors)

  return (padded_X, one_hot_y)

def build_combinations():
  combos = []
  # for cell in cell_type:
  for step in input_gate_dropout:
    for count in cells_per_layer:
      # for layer in multilayer_rnn:
      for flag in recurrent_dropout:
        for optimizer in optimizers:
          combos.append(['GRU', step, count, False, flag, optimizer])
  return combos

print('Loading data...')
initial_time = tm.time()
data = pd.read_pickle('data/local_data.p')
if env == 'local':
  data = data[0:5000]
  cells_per_layer = [64, 80, 96]
elif env == 'test':
  data = data[0:10000]
X_raw = data['Content'].tolist()
y_raw = data['Tags'].tolist()
print "------------- Loaded in %0.2fs ---------------" % (tm.time() - initial_time)

print('Finding unique values ...')
checkpoint = tm.time()
vocab = reduce(lambda x, y: x | y, (set(snippet) for snippet in X_raw))
vocab_size = len(vocab) + 1
max_snippet_len = max(map(len, X_raw))
# the i++ is to reserve 0 for masking via pad_sequences
word_idx = dict((word, i + 1) for i, word in enumerate(vocab))
X_train, y_train = vectorize(X_raw, y_raw, word_idx, max_snippet_len)
print "------------- Prepared in %0.2fs ---------------" % (tm.time() - checkpoint)

combinations = build_combinations()

for combo in combinations:
  print("Building and compiling model ...")
  checkpoint = tm.time()
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length=max_snippet_len, dropout=0.2))
  # model.add(Embedding(vocab_size, combo[5], input_length=max_snippet_len, dropout=0.2))

  # if combo[0] == 'LSTM':
    # if combo[3]:
    #   model.add(LSTM(combo[2], return_sequences=True, dropout_W=combo[1], dropout_U=combo[4]))
    # model.add(LSTM(combo[2], dropout_W=combo[1], dropout_U=combo[4]))
  # elif combo[0] == 'GRU':
    # if combo[3]:
    #   model.add(GRU(combo[2], return_sequences=True, dropout_W=combo[1], dropout_U=combo[4]))
    # model.add(GRU(combo[2], dropout_W=combo[1], dropout_U=combo[4], go_backwards=True))
  model.add(GRU(combo[2], dropout_W=combo[1], dropout_U=combo[4]))

  model.add(Dense(21))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
      optimizer=combo[5], metrics=['accuracy'])
  print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)
  print("Cell type: %s" % combo[0], "Drop Input: %s" % combo[1], "Cell Count: %s" % combo[2])
  print("Multi-layer: %s" % combo[3], "Drop Recurrent: %s" % combo[4], "Optimizer: %s" % combo[5])

  # print('Training phase ...')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
          validation_split=0.2, verbose=2)
  model = None
  print " "