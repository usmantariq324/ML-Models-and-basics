import numpy as np
import pandas as pd
df=pd.read_csv('titanic_sample.csv')
# print(df['Number'].unique)
df['number_numrical']=pd.to_numeric(df['Number'],errors='coerce',downcast='integer')
df['number_categorical']=np.where(df['number_numrical'].isnull(),df['Number'],np.nan)
print(df)
df['cabin_num']=df['Cabin'].str.extract('(\d+)')
df['cabin_cat']=df['Cabin'].str[0]
print(df.head(3))
df['ticket_num']=df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num']=pd.to_numeric(df['ticket_num'],errors='coerce',downcast='integer')

df['ticket_cat']=df['Ticket'].apply(lambda s: s.split()[0])
df['number_categorical']=np.where(df['ticket_cat'].str.isdigit(),np.nan,df['ticket_cat'])
print(df.head(3))