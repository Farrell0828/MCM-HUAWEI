import os 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from keras import backend as K 
from keras.layers import Input, Dense, BatchNormalization, ReLU 
from keras.models import Model 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard 
from sklearn.model_selection import GroupKFold, train_test_split 
from glob import glob 
from callbacks import PCRR, BatchLearningRateScheduler 
from losses import wmse
from utils import rmse, rmse_np, pcrr 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

df = pd.read_csv('./data/train_v2.csv')
test_df = pd.read_csv('./data/test_v2.csv')
x_cols = [col for col in df.columns if col not in ['Cell Index', 'RSRP', 'RSRP_Poor']]
test_x = test_df[x_cols].values

n_folds = 5
gkf = GroupKFold(n_splits=n_folds)
fold = 0
test_y_pred = 0
rmse_scores = []
pcrr_scores = []

for train_idx, val_idx in gkf.split(df, df['RSRP_Poor'], df['Cell Index']):
    fold += 1
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_x = train_df[x_cols].values
    train_y = train_df['RSRP'].values
    val_x = val_df[x_cols].values
    val_y = val_df['RSRP'].values
    train_x = (train_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
    val_x = (val_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
    test_x = (test_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
    print('train_x.shape', train_x.shape)
    print('train_y.shape', train_y.shape)
    print('val_x.shape', val_x.shape)
    print('val_y.shape', val_y.shape)
    print('test_x.shape', test_x.shape)

    K.clear_session()
    input_tensor = Input(shape=(train_x.shape[1], ), name='input')
    x = Dense(128, name='fc1')(input_tensor)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)
    x = Dense(128, name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU(name='relu2')(x)
    x = Dense(1, name='output')(x)

    model = Model(inputs=input_tensor, outputs=x)
    optimizer = SGD(lr=1e-3, momentum=0.9)
    model.compile(loss=wmse, optimizer=optimizer, metrics=[rmse])

    checkpoint = ModelCheckpoint('./k-fold/model_v2/val_loss_best_fold_{}.h5'.format(fold), 
                                 save_weights_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(patience=5, min_delta=1e-4, verbose=1)
    early_stop = EarlyStopping(patience=8, min_delta=1e-4, verbose=1)
    tensorboard = TensorBoard('./logs/fold_{}'.format(fold))

    callbacks = [checkpoint, reduce_lr, early_stop, tensorboard]

    model.fit(x=train_x, y=train_y, 
              batch_size=1024, 
              epochs=10,
              validation_data=(val_x, val_y),
              callbacks=callbacks)

    model.load_weights('./k-fold/model_v2/val_loss_best_fold_{}.h5'.format(fold))
    val_y_pred = model.predict(val_x, batch_size=4096)
    print('Calculate final val RMSE and PCRR score...')
    rmse_score = rmse_np(val_y, val_y_pred)
    rmse_scores.append(rmse_score)
    print('Fold {} Val RMSE Score: {}'.format(fold, rmse_score))
    pcrr_score = pcrr(-103, val_y, val_y_pred)
    pcrr_scores.append(pcrr_score)
    print('Fold {} Val PCRR Score: {}'.format(fold, pcrr_score))

    test_y_pred += model.predict(test_x, batch_size=1024, verbose=1)

print('CV RMSE Score: ', np.array(rmse_scores).mean())
print('CV PCRR Score: ', np.array(pcrr_scores).mean())

test_y_pred /= n_folds
test_df['RSRP'] = test_y_pred
for cell_index in test_df['Cell Index'].unique():
    sub_df = test_df[test_df['Cell Index'] == cell_index]
    rsrp = sub_df[['RSRP']].values.tolist()
    f = open("./results/model_v2/test_{}.csv_result.txt".format(cell_index), "w")
    f.write(str({'RSRP': rsrp}))
    f.close()