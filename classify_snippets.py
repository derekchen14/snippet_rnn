'''
Trains a LSTM classifier on Datafox Snippets.
Created by Derek C based on Keras examples.
'''
import numpy as np
import time as tm
import builder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, EarlyStopping
from keras.layers.recurrent import LSTM, SimpleRNN, GRU

env = 'test'
np.random.seed(14)
batch_size = 100 if env == 'local' else 32
epochs = 3 if env == 'local' else 15
num_buckets = 8
grid_search = True  # False means 50 combos are randomly sampled

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

def trainNetwork(X_train, y_train, model, p):
  print("Building and compiling model ...")
  checkpoint = tm.time()
  print "---------"
  print p['maxlen']
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
  combinations = builder.combinations(grid_search, param_options)
  buckets = builder.loadData(num_buckets)

  for params in combinations:
    model = Sequential()
    bucket = buckets   # for bucket in buckets:
    X, y, word_idx, params = builder.prepareData(bucket, params)
    X_train, y_train = builder.vectorize(X, y, word_idx, params)
    model = trainNetwork(X_train, y_train, model, params)
    print model.layers
    model = None

if __name__ == "__main__":
  main()
