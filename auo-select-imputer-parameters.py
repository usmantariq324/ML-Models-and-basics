import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df=pd.read_csv('tested.csv')
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
x=df.drop(columns=['Survived'])
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
numerical_features=['Age','Fare']
categorical_features=['Embarked','Sex']
numerical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler()),
])

categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most-frequent')),
    ('ohe',OneHotEncoder(handle_unknown='ignore'))
])
preprocessor=ColumnTransformer(
    transformers=[
        ('num',numerical_transformer,numerical_features),
        ('cat',categorical_transformer,categorical_features),
])

clf=Pipeline(steps=[
     ('preprocessor',preprocessor),
    ('classifier',LogisticRegression()),
])

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
    'classifier__C': [0.1, 1.0, 10, 100]
}

grid_search=GridSearchCV(clf,param_grid,cv=10)
grid_search.fit(x_train,y_train)
print("Find best params:",grid_search.best_params_)


