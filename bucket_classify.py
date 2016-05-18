'''
Trains a LSTM classifier on Datafox Snippets.
Created by Derek C based on Keras examples.
'''
import numpy as np
import pandas as pd
import time as tm
import re
import random
import itertools

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
epochs = 3 if env == 'local' else 15
num_buckets = 8

style = 'grid' # or 'random'
param_options = {
  'cell_type': ['LSTM', 'GRU'], # https://arxiv.org/abs/1412.3555
  'input_gate_dropout': [0.5, 0.6], # http://arxiv.org/abs/1512.05287
  'cells_per_layer': [128, 256], # https://github.com/karpathy/char-rnn
  'multilayer_rnn': [True, False],
  'recurrent_dropout': [0.0, 0.2], # http://arxiv.org/abs/1409.2329
  'optimizers': ['adam', 'rmsprop'], # http://arxiv.org/abs/1412.6980
  'embedding_size': [128, 200, 300],
  'reverse_input': [True, False]
}


def build_combinations():
  permutations, combinations = [], []
  for key,options in p.iteritems():
    permutations.append([{key:opt} for opt in options])
  for combo in itertools.product(*permutations):
    combinations.append( { k: v for d in combo for k, v in d.items() } )
  return combinations

def loadData(num_buckets):
  print('Loading data...')
  initial_time = tm.time()
  buckets = []
  for i in xrange(num_buckets):
    count = str(i+1)
    location = "data/buck"+count+".p"
    data = pd.read_pickle(location)
    buckets.append(data)
  print "------------- Loaded %d buckets in %0.2fs ---------------" % (len(buckets), (tm.time() - initial_time))
  return buckets

def prepareData(data, params):
  print('Finding unique values ...')
  checkpoint = tm.time()
  X_raw = data['Content'].tolist()
  y_raw = data['Tags'].tolist()

  vocab = reduce(lambda x, y: x | y, (set(snippet) for snippet in X_raw))
  vocab_size = len(vocab) + 1
  max_snippet_len = max(map(len, X_raw))
  # the i++ is to reserve 0 for masking via pad_sequences
  word_idx = dict((word, i + 1) for i, word in enumerate(vocab))
  params['maxlen'] = max_snippet_len
  params['vocab_size'] = vocab_size

  print('Vocabulary size: %d unique words') % vocab_size
  print('Max snippet length: %d words') % max_snippet_len
  print('Number of training inputs: %d') % len(X_raw)
  print('Number of training labels: %d') % len(y_raw)
  print "------------- Prepared in %0.2fs ---------------" % (tm.time() - checkpoint)

  return X_raw, y_raw, word_idx, params

def vectorize(X, y, word_idx, params):
  max_snippet_len = params['maxlen']
  vocab_size = params['vocab_size']

  X_vectors = map(lambda (snippet): [word_idx[w] for w in snippet], X)
  padded_X = pad_sequences(X_vectors, maxlen=max_snippet_len)

  label_index = dict((label, i) for i, label in enumerate(list(set(y))))
  y_vectors = [label_index[w] for w in y]
  one_hot_y = np_utils.to_categorical(y_vectors)

  return (padded_X, one_hot_y)

def trainNetwork(X_train, y_train, model, p):
  print("Building and compiling model ...")
  checkpoint = tm.time()
  model.add(Embedding(p['vocab_size'], 128, input_length=p['maxlen'], dropout=0.2))
  model.add(GRU(p['cell_count'], dropout_W=p['dropout'], dropout_U=0.2))

  model.add(Dense(21))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
        optimizer=p['optimizer'], metrics=['accuracy'])
  print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)
  print("Dropout: %s" % p['dropout'], "Cell Count: %s" % p['cell_count'])
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
        validation_split=0.2, verbose=2)
  return model

def main():
  combinations = build_combinations()
  buckets = loadData(num_buckets)

  for params in combinations:
    model = Sequential()
    for bucket in buckets:
      X, y, word_idx, params = prepareData(bucket, params)
      X_train, y_train = vectorize(X, y, word_idx, params)
      model = trainNetwork(X_train, y_train, model, params)
    model = None

if __name__ == "__main__":
  main()
