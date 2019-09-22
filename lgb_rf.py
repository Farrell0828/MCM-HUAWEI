#%%
import gc
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# 读数据，并进行处理
train_df = pd.read_csv("./src/data/train_v2.csv")
# train_df['d'] = ((train_df['Cell X'] - train_df['X'])**2 + (train_df['Cell Y'] - train_df['Y'])**2)**(1/2) * 0.001
# train_df = train_df[train_df['d'] >0]
# train_df = train_df[train_df['Height'] >0]
# train_df['lgd']=np.log10(train_df['d'])
# train_df['lgh']=np.log10(train_df['Height'])
# train_df['hb'] = train_df['Height'] + train_df['Cell Altitude'] - train_df['Altitude']
# train_df = train_df[train_df['hb'] > 0]
# train_df['hv'] = train_df['hb'] - train_df['d'] * np.tan(train_df['Electrical Downtilt'] + train_df['Mechanical Downtilt'])
# train_df['len'] = train_df['d'] / np.cos(train_df['Electrical Downtilt'] + train_df['Mechanical Downtilt'])
# train_df['lghb'] = np.log10(train_df['hb'])
# train_df['angle']=train_df['Electrical Downtilt'] + train_df['Mechanical Downtilt']
# train_df['lgd']=np.log10(train_df['d'])
# train_df['len2']=((train_df['d']**2+(train_df['hb'])**2)**(1/2))

X=train_df.drop('RSRP',axis=1)
y=train_df['RSRP']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
# lightGBM
from lightgbm import LGBMRegressor

reg = LGBMRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
loss = mean_squared_error(y_test, y_pred)

def CaculatePcrr(y_true,y_pred):
    t = -103
    tp = len(y_true[(y_true < t)&(y_pred < t)])
    fp = len(y_true[(y_true >= t)&(y_pred < t)])
    fn = len(y_true[(y_true < t) & (y_pred >= t)])
    precision =tp/(tp+fp)
    recall = tp/(tp+fn)
    pcrr = 2 * (precision * recall)/(precision + recall)
    return pcrr

print("----------loss_lgbm-------")
print(loss)
print("------importance--------")
print(reg.feature_importances_)
print("-----pcrr------")
print(CaculatePcrr(y_test,y_pred))

df = pd.DataFrame(X_train.columns.tolist(), columns=['feature'])
df['importance']=list(reg.feature_importances_)                
df = df.sort_values(by='importance',ascending=False)
print(df)

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
loss = mean_squared_error(y_test, y_pred)
print("----------loss_rf-------")
print(loss)
print("------importance--------")
print(reg.feature_importances_)
print("-----pcrr------")
print(CaculatePcrr(y_test,y_pred))

df2 = pd.DataFrame(X_train.columns.tolist(), columns=['feature'])
df2['importance']=list(reg.feature_importances_)                
df2 = df2.sort_values(by='importance',ascending=False)
print(df2)

df3 = pd.DataFrame(data=[[y_pred],[y_test]], columns=['pred', 'test'])
df3.to_csv('pred_RF_results2.csv')
print(df3)

