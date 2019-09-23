import numpy as np 
from tqdm import tqdm 
from keras import backend as K 
from sklearn.metrics import precision_score, recall_score, mean_squared_error 
from math import sqrt

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))
'''
def rmse_np(y_true, y_pred):
    sum_se = 0
    sub_len = 10000
    steps = np.ceil(len(y_true) / sub_len)
    for i in tqdm(range(int(steps))):
        y_true_sub = y_true[i*sub_len:(i+1)*sub_len]
        y_pred_sub = y_pred[i*sub_len:(i+1)*sub_len]
        sum_se += ((y_true_sub - y_pred_sub)**2).sum()
    return np.sqrt(sum_se / len(y_true))
'''
def rmse_np(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def pcrr4reg(y_true, y_pred):
    y_true = (y_true < -103).astype(int)
    y_pred = (y_pred < -103).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    pcrr = 2 * (precision * recall) / (precision + recall)
    return pcrr

def pcrr4cls(y_true, y_pred):
    y_true = (y_true >= 0.5).astype(int)
    y_pred = (y_pred >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    pcrr = 2 * (precision * recall) / (precision + recall)
    return pcrr