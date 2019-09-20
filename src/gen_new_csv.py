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

def add_fn(df, unique_clutter_index, split='train'):
    
    df['hb'] = df['Height'] + df['Cell Altitude'] - df['Altitude']
    df = df[df['hb'] >= 0]
    df['d'] = ((df['Cell X'] - df['X'])**2 + (df['Cell Y'] - df['Y'])**2)**(1/2) * 0.001
    df['hv'] = df['hb'] - df['d'] * np.tan(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
    df['len'] = df['d'] / np.cos(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
    df['lghb'] = np.log10(df['hb'])

    cell_clutter_dummy = pd.get_dummies(pd.Categorical(df['Cell Clutter Index'], categories=unique_clutter_index), prefix='CellClutterIndex')
    clutter_dummy = pd.get_dummies(pd.Categorical(df['Clutter Index'], categories=unique_clutter_index), prefix='ClutterIndex')
    df = (df.merge(cell_clutter_dummy, left_index=True, right_index=True)
            .merge(clutter_dummy, left_index=True, right_index=True))

    if split == 'train': 
        df['RSRP_Poor'] = df['RSRP'] < -103
    return df 

train_df = add_fn(train_df, unique_clutter_index, 'train')
test_df = add_fn(test_df, unique_clutter_index, 'test')

train_df.to_csv('./data/train_v3.csv', index=False)
test_df.to_csv('./data/test_v3.csv', index=False)
