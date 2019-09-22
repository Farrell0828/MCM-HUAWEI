import numpy as np 
from tqdm import tqdm 
from keras import backend as K 
from sklearn.metrics import mean_squared_error
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

def pcrr4reg(threshold, y_true, y_pred):
    t = threshold
    tp, fp, fn = 0, 0, 0
    sub_len = 10000
    steps = np.ceil(len(y_true) / sub_len)
    for i in tqdm(range(int(steps))):
        y_true_sub = y_true[i*sub_len:(i+1)*sub_len]
        y_pred_sub = y_pred[i*sub_len:(i+1)*sub_len]
        tp += ((y_true_sub < t) & (y_pred_sub < t)).sum()
        fp += ((y_true_sub >= t) & (y_pred_sub < t)).sum()
        fn += ((y_true_sub < t) & (y_pred_sub >= t)).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    pcrr = 2 * (precision * recall) / (precision + recall)
    return pcrr

def pcrr4cls(y_true, y_pred):
    y_pred = (y_pred >= 0.5).astype(int)
    tp, fp, fn = 0, 0, 0
    sub_len = 10000
    steps = np.ceil(len(y_true) / sub_len)
    for i in tqdm(range(int(steps))):
        y_true_sub = y_true[i*sub_len:(i+1)*sub_len]
        y_pred_sub = y_pred[i*sub_len:(i+1)*sub_len]
        tp += ((y_true_sub == 1) & (y_pred_sub == 1)).sum()
        fp += ((y_true_sub == 0) & (y_pred_sub == 1)).sum()
        fn += ((y_true_sub == 1) & (y_pred_sub == 0)).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    pcrr = 2 * (precision * recall) / (precision + recall)
    return pcrr