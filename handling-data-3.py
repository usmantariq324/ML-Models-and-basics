import  numpy as np
import pandas as pd
df=pd.read_csv('data_science_job.csv')
print(df.isnull().mean()*100)
cols=[var for var in df.columns if df[var].isnull().mean() <0.05 and df[var].isnull().mean() >0 ]
# print(cols)
len=len(df[cols].dropna())/len(df)
# print(len)
new_df=df[cols].dropna()
print(df.shape,new_df.shape)
