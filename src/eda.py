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

#%%
df = train_dfs[0]
df

#%%
user_cols = ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index']

df['X'].value_counts()

#%%
df['X'].map(df['X'].value_counts())

#%%
for col in user_cols:
    df[col + '_count'] = df[col].map(df[col].value_counts())

#%%
len(df)

#%%
len(df.drop_duplicates(['X', 'Y']))

#%%
df['mean_X'] = df['X'].mean()
df

#%%
a = 1, b = 2, c = 3

#%%
a = 1

#%%
b = 2

#%%
c = 3

#%%
d = [a, b, c]

#%%
for e in d:
    e += 4

#%%
d

#%%
sub_df = pd.read_csv('./data/train_v2_sub.csv')

#%%
sub_df

#%%
sub_df.columns

#%%
