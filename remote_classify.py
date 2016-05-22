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

env = 'remote'
np.random.seed(14)
batch_size = 100 if env == 'local' else 32
epochs = 3 if env == 'local' else 15

cell_type = ['LSTM', 'GRU'] # https://arxiv.org/abs/1412.3555
input_gate_dropout = [0.5, 0.6] # http://arxiv.org/abs/1512.05287
cells_per_layer = [128, 256] # https://github.com/karpathy/char-rnn
multilayer_rnn = [True, False]
recurrent_dropout = [0.0, 0.2] # http://arxiv.org/abs/1409.2329
optimizers = ['adam', 'rmsprop'] # http://arxiv.org/abs/1412.6980
embedding_size = [128, 200, 300]
reverse_input = [True, False]

def vectorize(X, y, word_idx, max_snippet_len):
  X_vectors = map(lambda (snippet): [word_idx[w] for w in snippet], X)
  padded_X = pad_sequences(X_vectors, maxlen=max_snippet_len)

  label_index = dict((label, i) for i, label in enumerate(list(set(y))))
  y_vectors = [label_index[w] for w in y]
  one_hot_y = np_utils.to_categorical(y_vectors)

  return (padded_X, one_hot_y)

def build_combinations():
  combos = []
  combos.append([0.5, 256])
  combos.append([0.5, 512])
  combos.append([0.6, 256])
  combos.append([0.6, 512])
  combos.append([0.6, 768])
  combos.append([0.7, 512])
  combos.append([0.7, 768])
  # for step in input_gate_dropout:
  #   for count in cells_per_layer:
  #     for size in embedding_size:
  #       for flag in reverse_input:
  #         combos.append([step, count, size, flag])
  print len(combos)
  return combos

print('Loading data...')
initial_time = tm.time()
data = pd.read_pickle('data/orgppl_64k_data.p')
if env == 'local':
  data = data[0:5000]
else:
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

print('Vocabulary size: %d unique words') % vocab_size
print('Max snippet length: %d words') % max_snippet_len
print('Number of training inputs: %d') % len(X_raw)
print('Number of training labels: %d') % len(y_raw)
combinations = build_combinations()

for combo in combinations:
  print("Building and compiling model ...")
  checkpoint = tm.time()
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length=max_snippet_len, dropout=0.2))

  model.add(GRU(combo[1], dropout_W=combo[0], dropout_U=0.2))

  model.add(Dense(21))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', metrics=['accuracy'])
  print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)
  print("Dropout: %s" % combo[0], "Size: %s" % combo[1])
  print('Training phase ...')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
        validation_split=0.2, verbose=2)
  model = None