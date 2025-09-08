import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
df=pd.read_csv('tested.csv')
df['Family']=df['SibSp']+df['Parch']
df=df.drop(columns=['PassengerId','Pclass','SibSp','Sex','Cabin','Embarked','Name','Ticket','Parch'])
x=df.drop(columns=['Survived'])
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train['Age99']=x_train['Age'].fillna(99)
x_train['Age_-1']=x_train['Age'].fillna(-1)


x_train['Fare99']=x_train['Fare'].fillna(99)
x_train['Fare_-1']=x_train['Fare'].fillna(-1)
print(x_train.var())
imputer1=SimpleImputer(strategy='constant',fill_value=99)
imputer2=SimpleImputer(strategy='constant',fill_value=999)
#using sickit learn
trf1=ColumnTransformer([
    ('imputer1',imputer1,['Age']),
    ('imputer2',imputer2, ['Fare']),
],remainder='passthrough')
trf1.fit(x_train)