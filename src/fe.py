import pandas as pd 

df = pd.read_csv('./data/train_v3_sub.csv')
print(type(df.sum(axis=0)))
row = df.iloc[0]
print(row.index)
print(type(row))
print(row[['X', 'Y']])
print(row[['X', 'Y']].max())
print(row[['X', 'Y']].min())
row[['X1, X2']] = row[['X', 'Y']]
print(row[['X1', 'X2']])