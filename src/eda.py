#%%
import pandas as pd 
import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt 
import seaborn as sns 

#%%
train_csvs = glob('./data/train_set/*')
train_dfs = [pd.read_csv(train_csv) for train_csv in train_csvs]
train_df = pd.concat(train_dfs)
train_df = train_df.reset_index(drop=True)

test_csvs = glob('./data/test_set/*')
test_dfs = [pd.read_csv(test_csv) for test_csv in test_csvs]
test_df = pd.concat(test_dfs)
test_df = test_df.reset_index(drop=True)

#%%
user_map_cols = ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index']
cell_map_cols = ['Cell X', 'Cell Y', 'Cell Altitude', 'Cell Building Height', 'Cell Clutter Index']
#%%
user_map_info = pd.concat([train_df[user_map_cols], test_df[user_map_cols]]).drop_duplicates()
cell_map_info = pd.concat([train_df[cell_map_cols], test_df[cell_map_cols]]).drop_duplicates()
cell_map_info.columns = user_map_info.columns
global_map_info = pd.concat([user_map_info, cell_map_info]).drop_duplicates()
global_map_info


#%%
global_map_info.to_csv('./data/global_map_info.csv', index=False)

#%%
train_df[['X', 'Y']].max() - train_df[['X', 'Y']].min()

#%%
(train_df[['X', 'Y']].max() - train_df[['X', 'Y']].min()) * 5 / 1000

#%%
train_df['hb'] = train_df['Height'] + train_df['Cell Altitude'] - train_df['Altitude']

#%%
(train_df['hb'] <= 0).sum()

#%%
test_df['hb'] = test_df['Height'] + test_df['Cell Altitude'] - test_df['Altitude']
(test_df['hb'] <= 0).sum()

#%%
train_df[['Cell X', 'X']].values.min(axis=1)

#%%
train_df['left_dege'] = train_df[['Cell X', 'X']].values.min(axis=1)
train_df['right_edge'] = train_df[['Cell X', 'X']].values.max(axis=1)
train_df

#%%
train_df[['Cell X', 'X', 'left_dege', 'right_edge']]

#%%
