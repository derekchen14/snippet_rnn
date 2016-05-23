import numpy as np
import pandas as pd
import time as tm

import itertools
import random
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

def combinations(grid_search, param_options):
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
  print "%d combinations being tested." % len(combos)
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

def prepareData(data, params={}):
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

def main():
  print "should not be main"

if __name__ == "__main__":
  main()
