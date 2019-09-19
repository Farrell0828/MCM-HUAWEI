import keras 
import numpy as np 
from tqdm import tqdm 

class PCRR(keras.callbacks.Callback):
    def __init__(self, threshold, val_x, val_y):
        super(PCRR, self).__init__()
        self.threshold = threshold
        self.val_x = val_x
        self.val_y_true = val_y

    def on_epoch_end(self, epochs, logs={}):
        val_y_pred = self.model.predict(self.val_x, batch_size=4096)
        val_pcrr = self.pcrr(self.val_y_true, val_y_pred)
        logs.update({'val_pcrr': val_pcrr})
        print('Val PCRR: {}'.format(val_pcrr))

    def pcrr(self, y_true, y_pred):
        t = self.threshold
        tp, fp, fn = 0, 0, 0
        sub_len = 1000
        steps = np.ceil(len(y_true) / sub_len)
        for i in range(int(steps)):
            y_true_sub = y_true[i*sub_len:(i+1)*sub_len]
            y_pred_sub = y_pred[i*sub_len:(i+1)*sub_len]
            tp += ((y_true_sub < t) & (y_pred_sub < t)).sum()
            fp += ((y_true_sub >= t) & (y_pred_sub < t)).sum()
            fn += ((y_true_sub < t) & (y_pred_sub >= t)).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        print('precision: ', precision)
        print('recall: ', recall)
        pcrr = 2 * (precision * recall) / (precision + recall)
        return pcrr

class BatchLearningRateScheduler(keras.callbacks.Callback):
  """Learning rate scheduler.

  Arguments:
      schedule: a function that takes an batch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.
  """

  def __init__(self, schedule, verbose=0):
    super(BatchLearningRateScheduler, self).__init__()
    self.schedule = schedule
    self.verbose = verbose
    self.past_steps = 0   # record steps before current epoch
    self.steps = 0        # record steps of current epoch

  def on_batch_begin(self, batch, logs=None):
    batch += self.past_steps 
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    try:  # new API
      lr = float(keras.backend.get_value(self.model.optimizer.lr))
      lr = self.schedule(batch, lr)
    except TypeError:  # Support for old API for backward compatibility
      lr = self.schedule(batch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('At batch {} the output of the "schedule" function: {} should be float.'.format(batch, lr))
    keras.backend.set_value(self.model.optimizer.lr, lr)
    if self.verbose > 0:
      print('\nEpoch %05d: LearningRateScheduler reducing learning '
            'rate to %s.' % (batch + 1, lr))

  def on_batch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)
    self.steps += 1

  def on_epoch_end(self, epoch, logs=None):
    self.past_steps += self.steps
    self.steps = 0