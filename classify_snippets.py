'''
Trains a LSTM classifier on Datafox Snippets.
Created by Derek C based on Keras examples.
'''
import numpy as np
import time as tm
import builder
import results

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.callbacks import TensorBoard

env = 'local'
np.random.seed(14)
batch_size = 100 if env == 'local' else 32
epochs = 5 if env == 'local' else 15
num_buckets = 8
grid_search = True  # False means 50 combos are randomly sampled

param_options = {
  # 'cell_type': ['LSTM', 'GRU'], # https://arxiv.org/abs/1412.3555
  # 'dropout_W': [0.5, 0.6], # http://arxiv.org/abs/1512.05287
  'cell_count': [64, 128], # https://github.com/karpathy/char-rnn
  'recur_dropout': [0.0, 0.2], # http://arxiv.org/abs/1409.2329
  'masking': [False] #, True
  # 'optimizers': ['rmsprop', 'adam', 'adadelta'] http://arxiv.org/abs/1412.6980
  # 'embedding_size': [128, 200, 300],
  # 'multilayer_rnn': [True, False],
  # 'reverse_input': [True, False]
}

def compile_network(model, p):
  print("Building and compiling model ...")
  checkpoint = tm.time()
  model.add(Embedding(p['vocab_size'], 128, input_length=p['maxlen'],
    dropout=0.2, mask_zero=p['masking']))
  model.add(GRU(p['cell_count'], dropout_W=0.5, dropout_U=p['recur_dropout']))

  model.add(Dense(p['num_classes']))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', metrics=['accuracy'])
  print "------------- Model compiled in %0.2fs ---------------" % (tm.time() - checkpoint)
  print p
  return model


def train_network(X_train, y_train, model):
  history = results.SingleHistory()
  tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
        validation_split=0.2, verbose=1, callbacks=[history, tensorboard])
  model = None
  return history

def train_buckets(combinations, buckets, aggregator):
  for params in combinations:
    model = Sequential()
    for bucket in buckets:
      X, y, word_idx, params = builder.prepareData(buckets, params)
      X_train, y_train = builder.vectorize(X, y, word_idx, params)
      model = compile_network(model, params)
      history = train_network(X_train, y_train, model)
    model = None
    aggregator.add_history(history, params)

def main():
  combinations = builder.combinations(grid_search, param_options)
  data = builder.loadData(num_buckets)
  aggregator = results.Aggregator()

  if len(data) > 100:
    X, y, word_idx, attributes = builder.prepareData(data)
    X_train, y_train = builder.vectorize(X, y, word_idx, attributes)
    for params in combinations:
      params.update(attributes)
      model = compile_network(Sequential(), params)
      history = train_network(X_train, y_train, model)
      aggregator.add_history(history, params)
  else:
    train_buckets(combinations, data, aggregator)
  aggregator.store_results()

if __name__ == "__main__":
  main()
