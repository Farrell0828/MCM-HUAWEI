import pandas as pd 
import numpy as np 
from glob import glob 
from tqdm import tqdm 
tqdm.pandas()
from multiprocessing import  Pool
from functools import partial

def add_norm_fn(df):
    df['hb'] = df['Height'] + df['Cell Altitude'] - df['Altitude']
    df = df[df['hb'] >= 0]
    df['d'] = ((df['Cell X'] - df['X'])**2 + (df['Cell Y'] - df['Y'])**2)**(1/2) * 0.001
    df['hv'] = df['hb'] - df['d'] * np.tan(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
    df['len'] = df['d'] / np.cos(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
    df['lghb'] = np.log10(df['hb'])
    return df 

def add_index_fn(df, unique_clutter_index):
    cell_clutter_dummy = pd.get_dummies(pd.Categorical(df['Cell Clutter Index'], categories=unique_clutter_index), prefix='CellClutterIndex')
    clutter_dummy = pd.get_dummies(pd.Categorical(df['Clutter Index'], categories=unique_clutter_index), prefix='ClutterIndex')
    df = (df.merge(cell_clutter_dummy, left_index=True, right_index=True)
            .merge(clutter_dummy, left_index=True, right_index=True))
    return df 

def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.progress_apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

def add_env_fn(row):
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
    info_df['bc'] = ((row['X'] - info_df['X'])**2 + (row['Y'] - info_df['Y'])**2)**(1/2)
    info_df['w'] = 1 - (info_df['ac']+info_df['bc']-info_df['ab']) / (delta_x+delta_y-info_df['ab'])
    return (info_df[need_cols] * info_df['w']).sum(axis=0)

if __name__ == '__main__':
    train_csvs = glob('./data/train_set/*')
    train_dfs = [pd.read_csv(train_csv) for train_csv in train_csvs]
    train_df = pd.concat(train_dfs)
    train_df = train_df.reset_index(drop=True)
    print('Read train done.')

    test_csvs = glob('./data/test_set/*')
    test_dfs = [pd.read_csv(test_csv) for test_csv in test_csvs]
    test_df = pd.concat(test_dfs)
    test_df = test_df.reset_index(drop=True)
    print('Read test done.')

    unique_clutter_index = list(set(list(train_df['Cell Clutter Index'].unique()) 
                                  + list(train_df['Clutter Index'].unique()) 
                                  + list(test_df['Cell Clutter Index'].unique()) 
                                  + list(test_df['Clutter Index'].unique())))
    unique_clutter_index.sort()

    train_df = add_norm_fn(train_df)
    print('Train add norm feature done.')
    test_df = add_norm_fn(test_df)
    print('Test add norm feature done.')
    train_df = add_index_fn(train_df, unique_clutter_index)
    print('Train add index feature done.')
    test_df = add_index_fn(test_df, unique_clutter_index)
    print('Test add index feature done.')

    map_df = pd.read_csv('./data/global_map_info.csv')
    print('Read map info done.')
    clutter_dummy = pd.get_dummies(pd.Categorical(map_df['Clutter Index'], categories=unique_clutter_index), prefix='ClutterIndex')
    map_df = map_df.merge(clutter_dummy, left_index=True, right_index=True)
    print('Map add index feature done.')

    need_cols = [col for col in map_df.columns if col not in ['X', 'Y', 'Clutter Index']]
    env_cols = ['env_' + col for col in need_cols]

    train_df[env_cols] = parallelize_on_rows(train_df, add_env_fn, 20)
    test_df[env_cols] = parallelize_on_rows(test_df, add_env_fn, 20)

    train_df.to_csv('./data/train_v4.csv', index=False)
    train_df.sample(n=1000).to_csv('./data/train_v4_sub.csv', index=False)
    test_df.to_csv('./data/test_v4.csv', index=False)
