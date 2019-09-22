import os 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from keras import backend as K 
from keras import regularizers 
from keras.layers import Input, Dense, BatchNormalization, ReLU, Lambda 
from keras.models import Model 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard 
from sklearn.model_selection import GroupKFold, train_test_split 
from glob import glob 
from callbacks import PCRR, BatchLearningRateScheduler 
from losses import wmse
from utils import rmse, rmse_np, pcrr4reg 
import matplotlib.pyplot as plt 
import seaborn as sns 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

df = pd.read_csv('./data/train_v2.csv')
test_df = pd.read_csv('./data/test_v2.csv')
x_cols = [col for col in df.columns if col not in ['Cell Index', 'Cell Clutter Index', 'Clutter Index', 'RSRP']]
test_x = test_df[x_cols].values

n_folds = 5
gkf = GroupKFold(n_splits=n_folds)
fold = 0
pcrr_scores = []

if not os.path.exists('./k-fold/'): os.mkdir('./k-fold')

for train_idx, val_idx in gkf.split(X=df, y=df['RSRP'], groups=df['Cell Index']):
    fold += 1
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_x = train_df[x_cols].values
    train_y = train_df['RSRP'].values
    train_y = (train_y < -103).astype(int)
    val_x = val_df[x_cols].values
    val_y = val_df['RSRP'].values
    val_y = (val_y < -103).astype(int)

    print('train_x.shape', train_x.shape)
    print('train_y.shape', train_y.shape)
    print('val_x.shape', val_x.shape)
    print('val_y.shape', val_y.shape)
    print('test_x.shape', test_x.shape)

    K.clear_session()
    input_tensor = Input(shape=(train_x.shape[1], ), name='input_tensor')
    x = BatchNormalization(name='bn0')(input_tensor)
    x = Dense(128, name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)
    x = Dense(128, name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU(name='relu2')(x)
    x = Dense(1, name='pred')(x)
    x = Lambda(lambda x: K.clip(x, -120, -60), name='cliped_pred')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.regularizers = [regularizers.l2(0.0005)]
    optimizer = Adam(lr=1e-4)
    model.compile(loss=wmse, optimizer=optimizer, metrics=[rmse])

    checkpoint = ModelCheckpoint('./k-fold/val_loss_best_fold_{}.h5'.format(fold), 
                                 save_weights_only=True, verbose=1)
    tensorboard = TensorBoard('./logs/fold_{}'.format(fold))

    # callbacks = [checkpoint]

    model.fit(x=train_x, y=train_y, 
              batch_size=1024, 
              epochs=1,
              validation_data=(val_x, val_y))

    # model.load_weights('./k-fold/val_loss_best_fold_{}.h5'.format(fold))
    val_y_pred = model.predict(val_x, batch_size=4096)
    plt.figure()
    sns.distplot(val_y_pred)
    plt.savefig('fold_{}_val_pred_dist.png'.format(fold))

    print('Calculate final val RMSE and PCRR score...')
    rmse_score = rmse_np(val_y, val_y_pred)
    rmse_scores.append(rmse_score)
    print('Fold {} Val RMSE Score: {}'.format(fold, rmse_score))
    pcrr_score = pcrr4reg(-103, val_y, val_y_pred)
    pcrr_scores.append(pcrr_score)
    print('Fold {} Val PCRR Score: {}'.format(fold, pcrr_score))

    test_y_pred = model.predict(test_x, batch_size=1024, verbose=1)
    final_pred.append(test_y_pred)
    plt.figure()
    sns.distplot(test_y_pred)
    plt.savefig('fold_{}_test_pred_dist.png'.format(fold))

print('CV RMSE Score: ', np.array(rmse_scores).mean())
print('CV PCRR Score: ', np.array(pcrr_scores).mean())

if not os.path.exists('./results/'): os.mkdir('./results')
final_pred = np.array(final_pred).mean(axis=0)
test_df['RSRP'] = final_pred
for cell_index in test_df['Cell Index'].unique():
    sub_df = test_df[test_df['Cell Index'] == cell_index]
    rsrp = sub_df[['RSRP']].values.tolist()
    f = open("./results/test_{}.csv_result.txt".format(cell_index), "w")
    f.write(str({'RSRP': rsrp}))
    f.close()