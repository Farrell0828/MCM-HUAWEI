import pandas as pd 
import numpy as np 
from glob import glob 

train_csvs = glob('./data/train_set/*')
train_dfs = [pd.read_csv(train_csv) for train_csv in train_csvs]
train_df = pd.concat(train_dfs)
train_df = train_df.reset_index(drop=True)

test_csvs = glob('./data/test_set/*')
test_dfs = [pd.read_csv(test_csv) for test_csv in test_csvs]
test_df = pd.concat(test_dfs)
test_df = test_df.reset_index(drop=True)

unique_clutter_index = list(set(list(train_df['Cell Clutter Index'].unique()) 
                              + list(train_df['Clutter Index'].unique()) 
                              + list(test_df['Cell Clutter Index'].unique()) 
                              + list(test_df['Clutter Index'].unique())))
unique_clutter_index.sort()

def add_norm_fn(df, split='train'):
    df['hb'] = df['Height'] + df['Cell Altitude'] - df['Altitude']
    df = df[df['hb'] >= 0]
    df['d'] = ((df['Cell X'] - df['X'])**2 + (df['Cell Y'] - df['Y'])**2)**(1/2) * 0.001
    df['hv'] = df['hb'] - df['d'] * np.tan(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
    df['len'] = df['d'] / np.cos(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
    df['lghb'] = np.log10(df['hb'])

    if split == 'train': 
        df['RSRP_Poor'] = df['RSRP'] < -103
    return df 

def add_index_fn(df, unique_clutter_index):
    cell_clutter_dummy = pd.get_dummies(pd.Categorical(df['Cell Clutter Index'], categories=unique_clutter_index), prefix='CellClutterIndex')
    clutter_dummy = pd.get_dummies(pd.Categorical(df['Clutter Index'], categories=unique_clutter_index), prefix='ClutterIndex')
    df = (df.merge(cell_clutter_dummy, left_index=True, right_index=True)
            .merge(clutter_dummy, left_index=True, right_index=True))
    return df 

train_df = add_norm_fn(train_df, 'train')
test_df = add_norm_fn(test_df, 'test')

map_df = pd.read_csv('./data/global_map_info.csv')
map_df = add_index_fn(map_df, unique_clutter_index)

def func(row):
    left_edge = row[['Cell X', 'X']].min()
    right_edge = row[['Cell X', 'X']].max()
    down_edge = row[['Cell Y', 'Y']].min()
    up_edge = row[['Cell Y', 'Y']].max()
    info_df = map_df[(map_df['X'] >= left_edge) 
                   & (map_df['X'] <= right_edge)
                   & (map_df['Y'] >= down_edge) 
                   & (map_df['Y'] <= up_edge)]
    delta_x = right_edge - left_edge
    delta_y = up_edge - down_edge
    info_df['ab'] = ((row['Cell X'] - row['X'])**2 + (row['Cell Y'] - row['Y'])**2)**(1/2)
    info_df['ac'] = ((row['Cell X'] - info_df['X'])**2 + (row['Cell Y'] - info_df['Y'])**2)**(1/2)
    info_df['ab'] = ((row['X'] - info_df['X'])**2 + (row['Y'] - info_df['Y'])**2)**(1/2)
    info_df['w'] = 1 - (info_df['ac']+info_df['bc']-info_df['ab']) / (delta_x+delta_y-info_df['ab'])
    need_cols = [col for col in row.index if col not in ['X', 'Y', 'Clutter Index']]
    row[need_cols] = (info_df[need_cols] * (1 / info_df['d']))

def add_env_fn(df, map_df):
    df['left_edge'] = df[['Cell X', 'X']].values.min(axis=1)
    df['right_edge'] = df[['Cell X', 'X']].values.max(axis=1)
    df['down_edge'] = df[['Cell Y', 'Y']].values.min(axis=1)
    df['up_edge'] =df[['Cell Y', 'Y']].values.max(axis=1)


train_df.to_csv('./data/train_v3.csv', index=False)
test_df.to_csv('./data/test_v3.csv', index=False)
