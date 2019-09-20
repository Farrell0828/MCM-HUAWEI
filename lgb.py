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
# 读数据，并进行处理
train_df = pd.read_csv("./data/train.csv")
X=train_df.drop('RSRP',axis=1)
y=train_df['RSRP']
#%%
history = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
# lightGBM
from lightgbm import LGBMRegressor

reg = LGBMRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
print(loss)
name = 'LGBMRegressor'
history.append([name,loss,end-start])

start = time.time()
reg = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=200,min_child_weight=10,
					subsample=0.7, colsample_bytree=0.7,reg_alpha=0, reg_lambda=0.5)
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'LGBMRegressor_s'
history.append([name,loss,end-start])


#%%
