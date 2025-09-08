import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
df=pd.read_csv('train.csv',usecols=['GarageQual','FireplaceQu','SalePrice'])
print(df.head(5))
print(df.isnull().mean()*100)
#WITHOUT USING IMPUTER LIBRARY
print(df['GarageQual'].mode())
fig=plt.figure()
ax=fig.add_subplot(121)
df[df['GarageQual']=='TA']['SalePrice'].plot(kind='kde',ax=ax)
df[df['GarageQual'].isnull()]['SalePrice'].plot(kind='kde',ax=ax,color='red')
lines,labels=ax.get_legend_handles_labels()
labels=['Houses with TA','House with NA']
ax.legend(lines,labels,loc='best')
plt.show()
temp=df[df['GarageQual']=='TA']['SalePrice']
df['GarageQual'].fillna('TA',inplace=True)
df['GarageQual'].value_counts().plot(kind='bar')
plt.show()
temp.plot(kind='kde',ax=ax)
df[df['GarageQual']=='TA']['SalePrice'].plot(kind='kde',ax=ax,color='red')
lines,labels=ax.get_legend_handles_labels()
labels=['Original variable','Imputed Variable']
ax.legend(lines,labels,loc='best')
plt.title('GarageQual')
plt.show()
#USING SIMPLE IMPUTER LIBRARY
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=['SalePrice']),df['SalePrice'],test_size=0.2)
imputer=SimpleImputer(strategy='most-frequent')
x_train=imputer.fit_transform(x_train)
x_test=imputer.fit_transform(x_train)