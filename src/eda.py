#%%
import pandas as pd 
import numpy as np 
from glob import glob 

#%%
train_df = pd.read_csv('./data/train_v1.csv')
train_df

#%%
len(train_df['Cell Index'].unique())

#%%
import matplotlib.pyplot as plt 
import seaborn as sns 

#%%
plt.scatter(train_df['Cell X'], train_df['Cell Y'])

#%%
sns.distplot(train_df['RSRP'])

#%%
(train_df['RSRP'] < -103).mean()

#%%
(train_df['RSRP'] < -103).sum()

#%%
cell_xy = train_df[['Cell X', 'Cell Y']].drop_duplicates()

#%%
plt.scatter(cell_xy['Cell X'], cell_xy['Cell Y'])

#%%
test_df = pd.read_csv('./data/test_v1.csv')
test_df

#%%
cell_xy_test = test_df[['Cell X', 'Cell Y']].drop_duplicates()

#%%
plt.figure()
plt.scatter(cell_xy['Cell X'], cell_xy['Cell Y'], color='green')
plt.scatter(cell_xy_test['Cell X'], cell_xy_test['Cell Y'], color='red')
plt.show()

#%%
