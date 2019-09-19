#%%
import pandas as pd 
import numpy as np 
from glob import glob 

#%%
train_csvs = glob('./data/train_set/*')
train_dfs = [pd.read_csv(train_csv) for train_csv in train_csvs]
train_df = pd.concat(train_dfs)
train_df = train_df.reset_index(drop=True)
train_df

#%%
test_csvs = glob('./data/test_set/*')
test_dfs = [pd.read_csv(test_csv) for test_csv in test_csvs]
test_df = pd.concat(test_dfs)
test_df = test_df.reset_index(drop=True)
test_df

#%%
test_cell_clutter_dummy = pd.get_dummies(pd.Categorical(test_df['Cell Clutter Index'], categories=range(1, 21)), prefix='CellClutterIndex_')
train_cell_clutter_dummy = pd.get_dummies(pd.Categorical(train_df['Cell Clutter Index'], categories=range(1, 21)), prefix='CellClutterIndex_')

test_clutter_dummy = pd.get_dummies(pd.Categorical(test_df['Clutter Index'], categories=range(1, 21)), prefix='ClutterIndex_')
train_clutter_dummy = pd.get_dummies(pd.Categorical(train_df['Clutter Index'], categories=range(1, 21)), prefix='ClutterIndex_')

#%%
test_df = test_df.merge(test_cell_clutter_dummy, left_index=True, right_index=True)
                 .merge(test_clutter_dummy, left_index=True, right_index=True)
del test_df['Cell Clutter Index']
del test_df['Clutter Index']
test_df

#%%
train_df = train_df.merge(test_cell_clutter_dummy, left_index=True, right_index=True)
                   .merge(train_clutter_dummy, left_index=True, right_index=True)
del train_df['Cell Clutter Index']
del train_df['Clutter Index']
train_df

#%%
train_df.to_csv('./data/train_v1.csv', index=False)
test_df.to_csv('./data/test_v1.csv', index=False)

#%%
