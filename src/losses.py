from keras import backend as K 
from keras.losses import mean_squared_error 

def wmse(y_true, y_pred, threshold=-103, weights=50):
    pos_mask = K.cast(y_true < -103, 'float')
    neg_mask = K.cast(y_true >= -103, 'float')
    mse = mean_squared_error(y_true, y_pred)
    return weights * mse * pos_mask + mse * neg_mask
