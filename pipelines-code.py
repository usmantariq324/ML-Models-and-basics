import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
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
df= pd.read_csv('tested.csv')
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
# print(df.head(5))
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)
# print(x_train.head(2))
# print(y_train.head(2))
# print(df.isnull().sum())
# si_age=SimpleImputer()
# si_embarked=SimpleImputer(strategy='most_frequent')
# x_train_age=si_age.fit_transform(x_train[['Age']])
# x_train_embarked=si_embarked.fit_transform(x_train[['Embarked']])
#
# x_test_age=si_age.transform(x_test[['Age']])
# x_test_embarked=si_embarked.transform(x_test[['Embarked']])
#
# ohe_gender=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
# ohe_embarked=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
# x_train_gender=ohe_gender.fit_transform(x_train[['Sex']])
# x_train_embarked1=ohe_embarked.fit_transform(x_train[['Embarked']])
# x_test_gender=ohe_gender.transform(x_test[['Sex']])
# x_test_embarked1=ohe_embarked.transform(x_test[['Embarked']])
#
# x_train_rem=x_train.drop(columns=['Sex','Age','Embarked'])
# x_test_rem=x_test.drop(columns=['Sex','Age','Embarked'])
#
# x_train_transfromed=np.concatenate((x_train_rem,x_train_age,x_train_gender,x_train_embarked1),axis=1)
# x_test_transfromed=np.concatenate((x_test_rem,x_test_age,x_test_gender,x_test_embarked1),axis=1)
#
# clf=DecisionTreeClassifier()
# clf.fit(x_train_transfromed,y_train)
# y_pred=clf.predict(x_test_transfromed)
#
# print(accuracy_score(y_test,y_pred))
#
#
# pickle.dump(ohe_gender,open('models/ohe_gender.pkl','wb'))
# pickle.dump(ohe_embarked,open('models/ohe_embarked.pkl','wb'))
# pickle.dump(clf,open('models/clf.pkl','wb'))

#pipelines code
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

# print(pipe.named_steps)
print(cross_val_score(pipe,x_train,y_train,cv=5,scoring='accuracy').mean())