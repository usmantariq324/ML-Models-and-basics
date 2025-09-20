import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv('linkedin_recommendation_dataset_clean.csv')

x=df.drop(columns=["Name","Skills_Count","Connections","Designation","Recommend"])
y=df['Recommend']
ordinal_features=['Education_Level','Experience_Level']
ordinal_mapping=[
    ['Bachelor','Master','PhD'],
    ['Junior','Mid','Senior']
]

onehot_features=['Technical_Expertise','Industry','Job_Search_Activity']
numeric_features=['Endorsements','Years_of_Experience','Location_Match']

prepocessor=ColumnTransformer(
transformers=[
    ('ord',OrdinalEncoder(categories=ordinal_mapping),ordinal_features),
    ('ohe',OneHotEncoder(handle_unknown='ignore'),onehot_features),
    ('num',StandardScaler(),numeric_features),
]
)

clf=Pipeline(
steps=[
    ("preproccessor",prepocessor),
    ("model",RandomForestClassifier())
]
)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Model Accuracy",accuracy_score(y_test,y_pred))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, x, y, cv=5)
print("CV Accuracy:", scores.mean())


edu = input("Enter Education Level (Bachelor/Master/PhD): ").strip()
exp = input("Enter Experience Level (Junior/Mid/Senior): ").strip()
tech = input("Enter Technical Expertise (AI Developer/Web Developer/Data Scientist/etc.): ").strip()
industry = input("Enter Industry (Tech/Finance/Healthcare/etc.): ").strip()
job_act = input("Enter Job Search Activity (High/Medium/Low): ").strip()
endorsements = int(input("Enter number of Endorsements: "))
years_exp = float(input("Enter Years of Experience: "))
location_match = int(input("Enter Location Match (1 = Yes, 0 = No): "))
new_data = pd.DataFrame([[
    edu, exp, tech, industry, job_act,
    endorsements, years_exp, location_match
]], columns=[
    'Education_Level','Experience_Level','Technical_Expertise',
    'Industry','Job_Search_Activity','Endorsements',
    'Years_of_Experience','Location_Match'
])
prediction= clf.predict(new_data)
if prediction[0]==1:
    print("This person will be Recommended")
else:
    print("No! It's not a good match")
x_new_num=prepocessor.transform(new_data)
x_train_num=prepocessor.transform(x_train)
similarity=cosine_similarity(x_new_num,x_train_num)
Names=df['Name'].values
similarity_scores = similarity.flatten()
top_k = 4
top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
top_matches = Names[top_indices]
print("Top recommended people:", top_matches)