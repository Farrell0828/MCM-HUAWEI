import pandas as pd 
import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt 
import seaborn as sns 

def rsrp_dist(train_df, title):
    plt.figure(figsize=(16, 12))
    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(train_df['X'], train_df['Y'], c=train_df['RSRP'], s=1, cmap=cmap)
    f.colorbar(points)
    cell_x = train_df.loc[0, 'Cell X']
    cell_y = train_df.loc[0, 'Cell Y']
    angle = train_df.loc[0, 'Azimuth']
    if angle == 90 or angle == 270:
        end_x = cell_x + 400
        end_y = cell_y
    else:
        end_x = cell_x + 250*np.tan(angle * np.pi / 180)
        end_y = cell_y + 250
    plt.scatter(cell_x, cell_y, c='r', s=50)
    plt.plot([cell_x, end_x], [cell_y, end_y], 'r--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(title+'.png', dpi=300)

def user_dist(train_df, title):
    plt.figure(figsize=(12, 9))
    plt.scatter(train_df['X'], train_df['Y'], c='g', s=1)
    plt.scatter(train_df['Cell X'], train_df['Cell Y'], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(title+'.png', dpi=300)


train_csvs = glob('./data/train_set/*')

train_df = pd.read_csv(train_csvs[960])
user_dist(train_df, 'user_dist_1')

train_df = pd.read_csv(train_csvs[961])
user_dist(train_df, 'user_dist_2')
rsrp_dist(train_df, 'rsrp_dist_3')