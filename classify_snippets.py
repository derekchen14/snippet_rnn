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
from keras.callbacks import Callback
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.sequence import pad_sequences

# env = 'local'
np.random.seed(14)
batch_size = 100 if env == 'local' else 32
epochs = 2 if env == 'local' else 10

cell_type = ['LSTM', 'GRU'] # https://arxiv.org/abs/1412.3555
recurrent_dropout = [True, False] # http://arxiv.org/abs/1409.2329
input_gate_dropout = [0.1, 0.3, 0.5] # http://arxiv.org/abs/1512.05287
optimizers = ['adam', 'rmsprop'] # http://arxiv.org/abs/1412.6980
cells_per_layer = [64, 128, 256] # https://github.com/karpathy/char-rnn

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
      for step in input_gate_dropout:
        for optimizer in optimizers:
          for count in cells_per_layer:
            combos.append([cell, flag, step, optimizer, count])
  return combos


print('Loading data...')
initial_time = tm.time()
data = pd.read_pickle('data/training_data.p')
if env == 'local':
  data = data[0:5000]
data = data[0:9000]
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

class show_results(Callback):
  def on_train_begin(self, logs={}):
      self.losses = []
  def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('val_loss'))

callback = show_results()
'''

combinations = build_combinations()

for combo in combinations:
  print("Building and compiling model ...")
  checkpoint = tm.time()
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length=max_snippet_len, dropout=0.2))

  drop_u = 0.2 if combo[1] else 0.0 # for dropout on recurrent connections
  drop_w = combo[2]  # for dropout on input gate connections

  if combo[0] == 'LSTM':
    model.add(LSTM(combo[4], dropout_W=drop_w, dropout_U=drop_u))
  elif combo[0] == 'GRU':
    model.add(GRU(combo[4], dropout_W=drop_w, dropout_U=drop_u))
  model.add(Dense(21))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
      optimizer=combo[3], metrics=['accuracy'])
  print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)
  print("Cell type: %s" % combo[0], "Drop Recurrent: %s" % combo[1], "Drop Input: %s" % combo[2], "Optimizer: %s" % combo[3], "Layers: %s" % combo[4])

  print('Training phase ...')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
          validation_split=0.2, verbose=2)
          # validation_split=0.2, callbacks=[callback], verbose=2)
  print " "