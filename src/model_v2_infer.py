import os 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from keras import backend as K 
from keras import regularizers 
from keras.layers import Input, Dense, BatchNormalization, ReLU, Lambda 
from keras.models import Model 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler 
from sklearn.model_selection import GroupKFold, train_test_split 
from glob import glob 
from callbacks import PCRR, BatchLearningRateScheduler 
from losses import wmse
from utils import rmse, rmse_np, pcrr4reg 
import matplotlib.pyplot as plt 
import seaborn as sns 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_df = pd.read_csv('./data/train_v2.csv')
test_df = pd.read_csv('./data/test_v2.csv')
x_cols = [col for col in train_df.columns if col not in ['Cell Index', 'Cell Clutter Index', 'Clutter Index', 'RSRP']]
test_x = test_df[x_cols].values

print(train_df['RSRP'].isna().sum())

train_x = train_df[x_cols].values
train_y = train_df['RSRP'].values

print('train_x.shape', train_x.shape)
print('train_y.shape', train_y.shape)
print('test_x.shape', test_x.shape)

K.clear_session()
myInput = Input(shape=(train_x.shape[1], ), name='myInput')
x = BatchNormalization(name='bn0')(myInput)
x = Dense(128, name='fc1')(x)
x = BatchNormalization(name='bn1')(x)
x = ReLU(name='relu1')(x)
x = Dense(128, name='fc2')(x)
x = BatchNormalization(name='bn2')(x)
x = ReLU(name='relu2')(x)
myOutput = Dense(1, name='myOutput')(x)

model = Model(inputs=myInput, outputs=myOutput)
model.regularizers = [regularizers.l2(0.0005)]
optimizer = Adam(lr=1e-3)
model.compile(loss='mse', optimizer=optimizer, metrics=[rmse])

def scheduler(epoch, lr):
    if epoch > 0:
        return 0.1*lr
    else:
        return lr

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

model.fit(x=train_x, y=train_y, 
          batch_size=1024, 
          epochs=3, 
          callbacks=[lr_scheduler])
model.save('./model_v2_infer.h5')

sess = K.get_session()
tf.saved_model.simple_save(sess, 'tf_model/saved_model', 
                           inputs={'myInput': model.input}, 
                           outputs={t.name: t for t in model.outputs})

test_y_pred = model.predict(test_x, batch_size=4096, verbose=1)
plt.figure()
sns.distplot(test_y_pred)
plt.savefig('test_pred_dist.png')

if not os.path.exists('./results/'): os.mkdir('./results')
test_df['RSRP'] = test_y_pred
for cell_index in test_df['Cell Index'].unique():
    sub_df = test_df[test_df['Cell Index'] == cell_index]
    rsrp = sub_df[['RSRP']].values.tolist()
    f = open("./results/test_{}.csv_result.txt".format(cell_index), "w")
    f.write(str({'RSRP': rsrp}))
    f.close()

train_y_pred = model.predict(train_x, batch_size=10240)
train_df['pred'] = train_y_pred
train_df.to_csv('./data/train_v2_with_pred.csv')