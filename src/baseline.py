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
from callbacks import PCRR, BatchLearningRateScheduler 
from utils import rmse, pcrr 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

df = pd.read_csv('/home/fanqimen/projects/MCM-HUAWEI/data/train_v1.csv')
x_cols = [col for col in df.columns if col not in ['Cell Index', 'RSRP']]

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = list(gkf.split(df, df['RSRP'], df['Cell Index']))[0]
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

train_x = train_df[x_cols].values
train_y = train_df['RSRP'].values
val_x = val_df[x_cols].values
val_y = val_df['RSRP'].values
train_x = (train_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
val_x = (val_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
print('train_x.shape', train_x.shape)
print('train_y.shape', train_y.shape)
print('val_x.shape', val_x.shape)
print('val_y.shape', val_y.shape)

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
model.compile(loss='mse', optimizer=optimizer, metrics=[rmse])

checkpoint = ModelCheckpoint('./k-fold/val_loss_best.h5', save_weights_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(patience=5, min_delta=1e-4, verbose=1)
early_stop = EarlyStopping(patience=8, min_delta=1e-4, verbose=1)
tensorboard = TensorBoard('./logs/baseline')
pcrr_callback = PCRR(-103, val_x, val_y)

callbacks = [checkpoint, reduce_lr, early_stop, tensorboard]

model.fit(x=train_x, y=train_y, 
          batch_size=1024, 
          epochs=50,
          validation_data=(val_x, val_y),
          callbacks=callbacks)

model.load_weights('./k-fold/val_loss_best.h5')
val_y_pred = model.predict(val_x)
print('Calculate final PCRR score...')
pcrr_score = pcrr(-103, val_y, val_y_pred)
print('PCRR Score: ', pcrr_score)