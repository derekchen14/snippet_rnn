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
grid_search = True  # False means 50 combos are randomly sampled
num_buckets = 8

param_options = {
  # 'cell_type': ['LSTM', 'GRU'], # https://arxiv.org/abs/1412.3555
  # 'dropout_W': [0.5, 0.6], # http://arxiv.org/abs/1512.05287
  'cell_count': [128, 256], # https://github.com/karpathy/char-rnn
  'recur_dropout': [0.0, 0.2], # http://arxiv.org/abs/1409.2329
  'masking': [True, False]
  # 'optimizers': ['rmsprop', 'adam', 'adadelta'] http://arxiv.org/abs/1412.6980
  # 'embedding_size': [128, 200, 300],
  # 'multilayer_rnn': [True, False],
  # 'reverse_input': [True, False]
}

def build_combinations(grid_search, param_options):
  permutations, combos = [], []
  for key,options in param_options.iteritems():
    permutations.append([{key:opt} for opt in options])
  for combo in itertools.product(*permutations):
    combos.append( { k: v for d in combo for k, v in d.items() } )
  try:
    combos = combos if grid_search else random.sample(combos, 50)
  except ValueError:
    print "Warning: Random search limits to top 50 combinations, but there are \
      only %d, so all combinations were returned." % len(combos)
  return combos

def loadData(num_buckets):
  print('Loading data...')
  initial_time = tm.time()
  buckets = pd.read_pickle("data/orgppl_10k_data.p")
  # buckets = []
  # for i in xrange(num_buckets):
  #   count = str(i+1)
  #   location = "data/buck"+count+".p"
  #   data = pd.read_pickle(location)
  #   buckets.append(data)
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
  params['num_classes'] = len(set(y_raw))

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
  print "---------"
  print p['maxlen']
  # The problem is that on the second iteration of the graph, we are adding more
  # layers to the model, but the model is already created.  Instead, we need
  # to find a way to swap the embedding layers since the input length has changed
  # in particular, the first bucket has length 10, while the second bucket has
  # length 20, this is obviouslsy not compatible
  model.add(Embedding(p['vocab_size'], 128, input_length=p['maxlen'],
    dropout=0.2, mask_zero=p['masking']))
  model.add(GRU(p['cell_count'], dropout_W=0.5, dropout_U=p['recur_dropout']))

  print p['num_classes']
  model.add(Dense(p['num_classes']))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', metrics=['accuracy'])
  print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)
  print p
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
        validation_split=0.2, verbose=2)
  return model

def main():
  combinations = build_combinations(grid_search, param_options)
  buckets = loadData(num_buckets)

  for params in combinations:
    model = Sequential()
    bucket = buckets   # for bucket in buckets:
    X, y, word_idx, params = prepareData(bucket, params)
    X_train, y_train = vectorize(X, y, word_idx, params)
    model = trainNetwork(X_train, y_train, model, params)
    print model.layers
    model = None

if __name__ == "__main__":
  main()
