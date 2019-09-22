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
    df['lgd'] = np.log10(df['d'] + 1)
    df['hv'] = df['hb'] - df['d'] * np.tan((df['Electrical Downtilt'] + df['Mechanical Downtilt']) * np.pi / 180)
    df['len'] = df['d'] / np.cos((df['Electrical Downtilt'] + df['Mechanical Downtilt']) * np.pi / 180)
    df['lghb'] = np.log10(df['hb'] + 1)
    return df 

def add_angle_fn(df):
    df['theta'] = np.arccos( (  np.sin(df['Azimuth']*np.pi/180) * (df['Y'] - df['Cell Y']) 
                                + np.cos(df['Azimuth']*np.pi/180) * (df['X'] - df['Cell X'])
                               ) / (
                                   np.sqrt( np.sin(df['Azimuth']*np.pi/180)**2 + np.cos(df['Azimuth']*np.pi/180)**2 )
                                 * np.sqrt( (df['X']-df['Cell X'])**2 + (df['Y']-df['Cell Y'])**2 )
                               )) / np.pi * 180
    return df

def add_count_fn(df, cols):
    for col in cols:
        df[col + '_count'] = df[col].map(df[col].value_counts())
    return df

def add_density_fn(df):
    df['n'] = len(df)
    df['area'] = ((df['X'].quantile(0.97) - df['X'].quantile(0.03))
                * (df['Y'].quantile(0.97) - df['Y'].quantile(0.03)))
    df['density'] = df['n'] / df['area']
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
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


if __name__ == '__main__':
    unique_clutter_index = [2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    user_cols = ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index']
    
    train_csvs = glob('./data/train_set/*')
    train_dfs = [pd.read_csv(train_csv) for train_csv in train_csvs]
    fn_added_train_dfs = []
    for train_df in tqdm(train_dfs):
        train_df = add_norm_fn(train_df)
        train_df = add_angle_fn(train_df)
        train_df = add_count_fn(train_df, user_cols)
        train_df = add_density_fn(train_df)
        train_df = add_index_fn(train_df, unique_clutter_index)
        train_df = train_df.fillna(train_df.mean())
        fn_added_train_dfs.append(train_df)
    train_all_dfs = pd.concat(fn_added_train_dfs, axis=0)
    train_all_dfs.to_csv('./data/train_v3.csv', index=False)
    train_all_dfs.sample(n=1000).to_csv('./data/train_v3_sub.csv', index=False)
    
    test_csvs = glob('./data/test_set/*')
    test_dfs = [pd.read_csv(test_csv) for test_csv in test_csvs]
    fn_added_test_dfs = []
    for test_df in test_dfs:
        test_df = add_norm_fn(test_df)
        test_df = add_angle_fn(test_df)
        test_df = add_count_fn(test_df, user_cols)
        test_df = add_density_fn(test_df)
        test_df = add_index_fn(test_df, unique_clutter_index)
        test_df = test_df.fillna(test_df.mean())
        fn_added_test_dfs.append(test_df)
    train_all_dfs = pd.concat(fn_added_test_dfs, axis=0)
    train_all_dfs.to_csv('./data/test_v3.csv', index=False)