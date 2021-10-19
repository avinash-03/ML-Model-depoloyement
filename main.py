#  importing Librarys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from pickle import dump,load

# Load Dataset

url="http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Loading Dataset
df=pd.read_csv(url,names=names)

df.tail(11)

df.columns
test=[
{'sepal_length':5.1, 'sepal_width':3.5, 'petal_length':1.4, 'petal_width':0.2},
    {'sepal_length':6.9, 'sepal_width':3.1, 'petal_length':5.4, 'petal_width':2.1}
]

df.info()

# Checking null values
df.isna().sum()

df.describe()

# visulization

plt.figure(figsize=(10,8))
sns.scatterplot(x='sepal_length',y='sepal_width',data=df,hue='species')

plt.hist(df['species'])

# Label Encoding



# Selecting dependent and indepedent variable
x=df.drop('species',axis=1)
y=df['species']

# split data

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

# fit and train model

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("accuray is: ",accuracy_score(ytest,ypred))
cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print(classification_report(ytest,ypred))



# conclusion:
#In this task we have we explored the iris dataset and predicted the classification using RandomForestClassifier algorithm.

dump(model,open('model.pkl','wb'))