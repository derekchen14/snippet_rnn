from keras.callbacks import Callback
import pandas as pd
import datetime

class SingleHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []
    self.accuracies = []
  def on_epoch_end(self, batch, logs={}):
    self.losses.append(logs.get('val_loss'))
    self.accuracies.append(logs.get('val_acc'))
  def on_train_end(self, logs={}):
    results = pd.DataFrame([self.losses, self.accuracies])
    now = datetime.datetime.now()
    filename = "results/%02d%d_results.csv" % (now.month, now.day)
    results.to_csv(filename)

class Aggregator():
  def __init__(self):
    self.histories = []

  def add_history(self, history, params):
    history.losses.insert(0, "Loss")
    history.accuracies.insert(0, "Accuracy")
    result = (params, history.losses, history.accuracies)
    self.histories.append(result)

  def store_results(self):
    results = pd.DataFrame(self.histories)
    now = datetime.datetime.now()
    filename = "results/%02d%d_results.csv" % (now.month, now.day)
    results.to_csv(filename)