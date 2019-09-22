import keras 
import numpy as np 
from tqdm import tqdm 
from math import sqrt
from sklearn.metrics import precision_score, recall_score, mean_squared_error 

class PCRR(keras.callbacks.Callback):
    def __init__(self, threshold, val_x, val_y):
        super(PCRR, self).__init__()
        self.threshold = threshold
        self.val_x = val_x
        self.val_y = val_y

    def on_epoch_end(self, epochs, logs={}):
        y_true = (self.val_y < self.threshold).astype(int)
        y_pred = self.model.predict(self.val_x, batch_size=10240)
        y_pred = (y_pred < self.threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        print('Epoch {} Precision Score: {}'.format(epochs+1, precision))
        recall = recall_score(y_true, y_pred)
        print('Epoch {} Recall Score: {}'.format(epochs+1, recall))
        pcrr = 2 * (precision * recall) / (precision + recall)
        print('Epoch {} PCRR Score: {}'.format(epochs+1, pcrr))


class RMSE(keras.callbacks.Callback):
    def __init__(self, val_x, val_y):
        super(RMSE, self).__init__()
        self.val_x = val_x
        self.val_y = val_y
        
    def on_epoch_end(self, epochs, logs={}):
        y_true = self.val_y
        y_pred = self.model.predict(self.val_x, batch_size=10240)
        rmse_score = self.rmse_np(y_true, y_pred)
        print('Epoch {} RMSE Score: {}'.format(epochs+1, rmse_score))

    def rmse_np(self, y_true, y_pred):
        return sqrt(mean_squared_error(y_true, y_pred))
