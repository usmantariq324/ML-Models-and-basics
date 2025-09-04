from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

de=pd.read_csv('car_data_32brands.csv')
le=LabelEncoder()
df=pd.read_csv('reviews_education_purchased_500_simple.csv')
x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,0:2],df.iloc[:,-1],test_size=0.2)
oe=OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])
oe.fit(x_train)
x_train=oe.transform(x_train)
x_test=oe.transform(x_test)
print("Test data after transformation",x_test)
print("Train data after transformation",x_train)
le.fit(y_train)
y_train=le.transform(y_train)
y_test=le.transform(y_test)
print(de.head(5))
print(de['brand'].value_counts())
encode=pd.get_dummies(de, columns=["fuel", "owner"],dtype=int,drop_first=True)
print(encode.head())
X_train,X_test,Y_train,Y_test=train_test_split(de.iloc[:,0:4],de.iloc[:,-1],test_size=0.2)
ohe=OneHotEncoder(drop='first')
X_train_new=ohe.fit_transform(X_train[['fuel','owner']]).toarray()
X_test_new=ohe.fit_transform(X_test[['fuel','owner']]).toarray()
print(np.hstack((X_train[['brand','km_driven']].values,X_train_new)))
new_data=pd.read_csv('patient_data_numeric_fever.csv')
print(new_data.isnull().sum())
x_train,x_test,y_train,y_test=train_test_split(new_data.drop(columns=['has_covid']),new_data['has_covid'],test_size=0.2)
# print(x_train)
si=SimpleImputer()
x_train_fever=si.fit_transform(x_train['fever'])
x_test_fever=si.fit_transform(x_test['fever'])
oe=OrdinalEncoder(categories=[['Mild','Strong']])
x_train_cough=oe.fit_transform(x_train[['cough']])
x_test_cough=oe.fit_transform(x_test[['cough']])
ohe=OneHotEncoder(drop='first')
x_train_gender_city=ohe.fit_transform(x_train[['gender','city']]).toarray()
x_test_gender_city=ohe.fit_transform(x_test[['gender','city']]).toarray()
x_train_age=x_train.drop(columns=['gender','fever','cough','city'])
x_test_age=x_test.drop(columns=['gender','fever','cough','city'])

x_train_transfromed=np.concatenate(x_train_age,x_train_gender_city,x_train_fever,x_train_cough,axis=1)
x_test_transfromed=np.concatenate(x_test_age,x_test_gender_city,x_test_fever,x_test_cough,axis=1)
transformer=ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(drop='first'),['gender','city']),
],remainder='passthrough')
print(transformer.fit_transform(x_train).shape)