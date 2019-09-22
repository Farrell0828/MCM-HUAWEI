#%%
import pandas as pd 
import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt 
import seaborn as sns 


#%%
np.tan(45 * np.pi / 180)

#%%
a = np.array([[1], [2], [3]])
a

#%%
a.tolist()

#%%
b = a.tolist()
type(b[0])

#%%
type(b[0][0])

#%%
type(a[0][0])

#%%
a = np.array([1, 2, 3, np.nan, 5, np.nan, 7, 8, np.nan])
a

#%%
np.isnan(a)

#%%
a[np.isnan(a)]

#%%
a[np.isnan(a)] = 100
a

#%%
df = pd.read_csv('./data/train_v2.csv')

#%%
df['RSRP'].mean()

#%%
df['Cell Index']

#%%
df['Cell Index'].unique()

#%%
import matplotlib.pyplot as plt 
import seaborn as sns 

#%%
sns.distplot(df['Cell Index'].unique())

#%%
sns.distplot(df['Cell Index'])

#%%
x, y, z = np.random.rand(3, 100)
cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
points = ax.scatter(x, y, c=z, s=50, cmap=cmap)
f.colorbar(points)

#%%
train_csvs = glob('./data/train_set/*')

#%%
train_df = pd.read_csv(train_csvs[961])
print(train_df['Azimuth'])
plt.figure(figsize=(16, 12))
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(train_df['X'], train_df['Y'], c=train_df['RSRP'], s=1, cmap=cmap)
f.colorbar(points)
cell_x = train_df.loc[0, 'Cell X']
cell_y = train_df.loc[0, 'Cell Y']
angle = train_df.loc[0, 'Azimuth']
end_x = cell_x + 400*np.tan(angle * np.pi / 180)
end_y = cell_y + 400
plt.scatter(cell_x, cell_y, c='r', s=50)
plt.plot([cell_x, end_x], [cell_y, end_y], 'r--')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('rsrp_dist.png', dpi=300)

#%%
train_df['Azimuth']

#%%
train_df.loc[0, 'X']

#%%
train_df['theta'] = np.arccos( (  np.sin(df['Azimuth']*np.pi/180) * (df['Y'] - df['Cell Y']) 
                                + np.cos(df['Azimuth']*np.pi/180) * (df['X'] - df['Cell X'])
                               ) / (
                                   np.sqrt( np.sin(df['Azimuth']*np.pi/180)**2 + np.cos(df['Azimuth']*np.pi/180)**2 )
                                 * np.sqrt( (df['X']-df['Cell X'])**2 + (df['Y']-df['Cell Y'])**2 )
                               )) / np.pi * 180

#%%
train_df['theta']

#%%
def calc_theta(azimuth, cell_x, cell_y, x, y):
    return np.arccos( (  np.sin(azimuth*np.pi/180) * (y - cell_y) 
                                + np.cos(azimuth*np.pi/180) * (x - cell_x)
                               ) / (
                                   np.sqrt( np.sin(azimuth*np.pi/180)**2 + np.cos(azimuth*np.pi/180)**2 )
                                 * np.sqrt( (x-cell_x)**2 + (y-cell_y)**2 )
                               )) / np.pi * 180


#%%
calc_theta(30, 55, 55, 33, 33) / np.pi * 180

#%%
calc_theta(145, 55, 55, 33, 33) / np.pi * 180

#%%
train_df['theta'].max()

#%%
train_df = pd.read_csv(train_csvs[971])
plt.figure(figsize=(12, 9))
plt.scatter(train_df['X'], train_df['Y'], c='g', s=1)
plt.scatter(train_df['Cell X'], train_df['Cell Y'], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('user_dist_1.png', dpi=300)

#%%
