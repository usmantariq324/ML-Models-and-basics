import numpy as np
import pandas as pd
date1=pd.read_csv('orders.csv')
time=pd.read_csv('messages.csv')
# print(date.head(3))
# print(time.head(3))
date1['date']=pd.to_datetime(date1['date'])
# print(date1.info())
date1['date_year']=date1['date'].dt.year

date1['date_month']=date1['date'].dt.month

date1['date_month_Name']=date1['date'].dt.month_name()

date1['date_days']=date1['date'].dt.day

date1['date_days_week']=date1['date'].dt.dayofweek

date1['date_days_name']=date1['date'].dt.day_name()

date1.drop(columns=['product_id','city_id','orders'],inplace=True)

# print(date1.head(3))

time['date']=pd.to_datetime(time['date'])

time['hour']=time['date'].dt.hour

time['min']=time['date'].dt.minute

time['sec']=time['date'].dt.second

print(time.head(3))
