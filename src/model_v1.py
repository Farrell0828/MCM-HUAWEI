import os 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import seaborn as sns 
import matplotlib.pyplot as plt 
from keras import backend as K 
from keras.layers import Input, Dense, BatchNormalization, ReLU 
from keras.models import Model 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard 
from sklearn.model_selection import GroupKFold, train_test_split 
from callbacks import PCRR, BatchLearningRateScheduler 
from utils import rmse, pcrr4reg, pcrr4cls  

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

df = pd.read_csv('/home/fanqimen/projects/MCM-HUAWEI/data/train_v1.csv')
x_cols = [col for col in df.columns if col not in ['Cell Index', 'Cell Clutter Index', 'Clutter Index', 'RSRP']]

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = list(gkf.split(df, df['RSRP'], df['Cell Index']))[0]
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

train_x = train_df[x_cols].values
train_y = train_df['RSRP'].values

val_x = val_df[x_cols].values
val_y = val_df['RSRP'].values
# train_x = (train_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
# val_x = (val_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
print('train_x.shape', train_x.shape)
print('train_y.shape', train_y.shape)
print('val_x.shape', val_x.shape)
print('val_y.shape', val_y.shape)

input_tensor = Input(shape=(train_x.shape[1], ), name='input_tensor')
x = Dense(128, name='fc1')(input_tensor)
x = BatchNormalization(name='bn1')(x)
x = ReLU(name='relu1')(x)
x = Dense(128, name='fc2')(x)
x = BatchNormalization(name='bn2')(x)
x = ReLU(name='relu2')(x)
reg_out = Dense(1, name='output')(x)

model = Model(inputs=input_tensor, outputs=reg_out)
optimizer = SGD(lr=1e-3, momentum=0.9)
model.compile(optimizer=optimizer, 
              loss='mse', 
              metrics=[rmse])

checkpoint = ModelCheckpoint('./k-fold/val_loss_best.h5', save_weights_only=True, verbose=1)
callbacks = [checkpoint]

model.fit(x=train_x, y=train_y,
          batch_size=1024, 
          epochs=1,
          validation_data=(val_x, val_y),
          callbacks=callbacks)

model.load_weights('./k-fold/val_loss_best.h5')

sess = K.get_session()
tf.saved_model.simple_save(sess, 'tf_model/saved_model', 
                           inputs={'input_tensor': model.input}, 
                           outputs={t.name: t for t in model.outputs})

plt.figure()
val_y_pred = model.predict(val_x)
print(val_y_pred[:100])
sns.distplot(val_y_pred)
plt.xlim(-91.8, -91.6)
plt.show()