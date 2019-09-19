from keras import backend as K 
from keras.losses import mean_squared_error 

def wmse(y_true, y_pred, threshold=-103, weights=5):
    mask = y_true < -103
    mse = mean_squared_error(y_true, y_pred)
    return weights * mse * mask + mse * ~mask
