#%%
import gc
import os
import datetime
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
# 线性回归：LR、Rigde(L2) 和 Lasso(L1)
from sklearn import linear_model

start = time.time()
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'LinearRegression'
history.append([name,loss,end-start])

start = time.time()
reg = linear_model.Ridge()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'Rigde'
history.append([name,loss,end-start])

#%%
# 集成模型：RF
from sklearn.ensemble import RandomForestRegressor

start = time.time()
reg = RandomForestRegressor()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'RandomForestRegressor'
history.append([name,loss,end-start])

start = time.time()
reg = RandomForestRegressor(n_estimators=200,random_state=0)
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'RandomForestRegressor_n_estimators=200'
history.append([name,loss,end-start])

from sklearn.ensemble.forest import ExtraTreeRegressor

start = time.time()
reg = ExtraTreeRegressor()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'ExtraTreeRegressor'
history.append([name,loss,end-start])


# 神经网络：MLP
from sklearn.neural_network import MLPRegressor

start = time.time()
reg = MLPRegressor()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = '	'
history.append([name,loss,end-start])

start = time.time()
reg = MLPRegressor(batch_size=50, hidden_layer_sizes=20, learning_rate_init=0.1,
				   max_iter=300,random_state=0,early_stopping=True)
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'MLPRegressor_s'
history.append([name,loss,end-start])

# SVM
from sklearn.svm import SVR, LinearSVR

start = time.time()
reg = SVR()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'SVR'
history.append([name,loss,end-start])

start = time.time()
reg = SVR(kernel='rbf',C=0.1, epsilon=0.1,max_iter=100)
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'SVR_s'
history.append([name,loss,end-start])

# XGBOOST
from xgboost.sklearn import XGBRegressor

start = time.time()
reg = XGBRegressor()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'XGBRegressor'
history.append([name,loss,end-start])

start = time.time()
reg = XGBRegressor(max_depth=4, n_estimators=500, min_child_weight=10,
				   subsample=0.7, colsample_bytree=0.7, reg_alpha=0, reg_lambda=0.5)
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'XGBRegressor_s'
history.append([name,loss,end-start])

# lightGBM
from lightgbm import LGBMRegressor

start = time.time()
reg = LGBMRegressor()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
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

from sklearn.neighbors.regression import KNeighborsRegressor

start = time.time()
reg = KNeighborsRegressor(n_neighbors=4,algorithm='kd_tree')
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'KNeighborsRegressor_s'
history.append([name,loss,end-start])

# 融合
start = time.time()
reg = LGBMRegressor()
reg.fit(X_train, y_train)
y_pred_lgb = reg.predict(X_test)
reg = linear_model.Lasso(alpha=2,max_iter=10)
reg.fit(X_train, y_train)
y_pred_lr = reg.predict(X_test)
y_pred = (y_pred_lgb + y_pred_lr) / 2
end = time.time()
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'LGBMRegressor+Lasso'
history.append([name,loss,end-start])


#%%
