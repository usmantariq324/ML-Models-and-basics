import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn import set_config
from sklearn.model_selection import cross_val_score
import pickle
# here we have imported all the libraries used to create a pipeline

# we have used  pandas to read csv file and load our titanic dataset
# without pipeline code
df= pd.read_csv('tested.csv')
# here we have dropped some columns we don't and need and they don't contribute in our final output and training of model
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
# here we are splitting our dataset in to train and test category in which the train dataset is used to train the model and test data to test the model
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)
# here we are checking is there any null values in dataset
print(df.isnull().sum())
# simple imputer is used to fill the missing data and this frist chunk of code is explainig how difficult it is to do these steps without pipelines
si_age=SimpleImputer()
si_embarked=SimpleImputer(strategy='most_frequent')
#here we have filled  the missing values and transformed them
x_train_age=si_age.fit_transform(x_train[['Age']])
x_train_embarked=si_embarked.fit_transform(x_train[['Embarked']])
x_test_age=si_age.transform(x_test[['Age']])
x_test_embarked=si_embarked.transform(x_test[['Embarked']])
#here we used onehotencoder which is a library used to encode nominal categorical data and convert them into the format computer can understand
ohe_gender=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
ohe_embarked=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
# here we again fitted and transformed data
x_train_gender=ohe_gender.fit_transform(x_train[['Sex']])
x_train_embarked1=ohe_embarked.fit_transform(x_train[['Embarked']])
x_test_gender=ohe_gender.transform(x_test[['Sex']])
x_test_embarked1=ohe_embarked.transform(x_test[['Embarked']])
#here we dropped the columns which we have taken from the dataset to perform fit and transform so we can concatenate it later
x_train_rem=x_train.drop(columns=['Sex','Age','Embarked'])
x_test_rem=x_test.drop(columns=['Sex','Age','Embarked'])
# here we have concatenated all the columns from our dataset
x_train_transfromed=np.concatenate((x_train_rem,x_train_age,x_train_gender,x_train_embarked1),axis=1)
x_test_transfromed=np.concatenate((x_test_rem,x_test_age,x_test_gender,x_test_embarked1),axis=1)
# here we have used decision tree clasifier to predict survival on titanic dataset
clf=DecisionTreeClassifier()
clf.fit(x_train_transfromed,y_train)
y_pred=clf.predict(x_test_transfromed)
#here we have printed accuracy result to check how our code is doing is it more accuracte or it lacks
print(accuracy_score(y_test,y_pred))
# here we are preparing our model for the server side where anyone can use this code on the server basically we are converting this code in to binary
pickle.dump(ohe_gender,open('models/ohe_gender.pkl','wb'))
pickle.dump(ohe_embarked,open('models/ohe_embarked.pkl','wb'))
pickle.dump(clf,open('models/clf.pkl','wb'))

#pipelines code
# here we have done all the same code more efficiently in mannered  way and in steps
# here it is simplified how we can do this all code we have done before but more efficiently without any leakage
trf1=ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6]),
],remainder='passthrough')
trf2=ColumnTransformer([
    ('ohe_gender_embarked',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])

],remainder='passthrough')

trf3=ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])
trf4=SelectKBest(score_func=chi2,k=8)
trf5=DecisionTreeClassifier()

pipe=make_pipeline(trf1,trf2,trf3,trf4,trf5)
pipe.fit(x_train,y_train)


y_pred=pipe.predict(x_test)

print(pipe.named_steps)
# here we are runing cross validation on our model to check how much is it accurate
print(cross_val_score(pipe,x_train,y_train,cv=5,scoring='accuracy').mean())