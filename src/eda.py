#%%
import pandas as pd 
import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt 
import seaborn as sns 

#%%
df = pd.read_csv('./data/train_v3.csv')

#%%
df.columns

cols = ['Cell X', 'Cell Y', 'Height', 'Azimuth',
       'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
       'RS Power', 'Cell Altitude', 'Cell Building Height',
       'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
       'Clutter Index', 'hb', 'd', 'lgd', 'hv', 'len', 'lghb', 'theta',
       'X_count', 'Y_count', 'Altitude_count', 'Building Height_count',
       'Clutter Index_count', 'n', 'area', 'density', 'RSRP']
#%%
len(cols)
#%%

#%%
figure = plt.figure(figsize=(12, 24), constrained_layout=True)
spec = figure.add_gridspec(ncols=4, nrows=8, hspace=0.2)
for row in range(8):
    for col in range(4):
        colu = cols[row*4+col]
        ax = figure.add_subplot(spec[row, col])
        sns.distplot(df[colu], ax=ax, label=colu)

#%%
df = pd.read_csv('ligthgbm.csv', index_col=0)

plt.figure(figsize=(8,12))
sns.barplot(x="importance",y="feature",data=df)
plt.tight_layout()
plt.savefig('lgbm_importances.png')

#%%
