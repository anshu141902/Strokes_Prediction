# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()

df.shape

df.columns

df['work_type'].unique()

df.info()

df.describe()

sns.pairplot(df)

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

df.head()

df.groupby('gender')['hypertension'].agg('sum')

sns.countplot(df['hypertension'], label=True)

sns.barplot(df['gender'], df['hypertension'])

df.head()

sns.barplot(df['smoking_status'], df['hypertension'])

sns.barplot(df['work_type'], df['hypertension'])

sns.barplot(df['Residence_type'], df['hypertension'])

sns.barplot(df['ever_married'], df['hypertension'])

sns.barplot(df['smoking_status'], df['heart_disease'])

sns.barplot(df['work_type'], df['heart_disease'])

sns.barplot(df['Residence_type'], df['heart_disease'])

sns.barplot(df['ever_married'], df['heart_disease'])

sns.barplot(df['ever_married'], df['stroke'])

sns.barplot(df['Residence_type'], df['heart_disease'])

sns.barplot(df['work_type'], df['heart_disease'])

sns.barplot(df['smoking_status'], df['heart_disease'])

sns.histplot(df['age'], kde=True)

sns.boxplot(df['age'])

df.describe()

upper_viscous= 61.00+1.5*(25.00)
upper_viscous

lower_viscous= 61.00-1.5*(25.00)
lower_viscous

sns.histplot(df['age'], bins=30,kde=True)

sns.histplot(df['bmi'], bins=30,kde=True)

sns.histplot(df['avg_glucose_level'], bins=30,kde=True)

df.head()

df1 = df.copy().drop(['id'], axis=1)

df1.head()

work_type = pd.get_dummies(df1['work_type'])
work_type

Residence_type = pd.get_dummies(df1['Residence_type'])
Residence_type

smoking_status = pd.get_dummies(df1['smoking_status'])
smoking_status

df2 = df1.copy().drop(['smoking_status','Residence_type','Residence_type','work_type'], axis=1)
df2.head()

df3 = pd.concat([df2,work_type],axis=1)
df3

df4 = pd.concat([df3,Residence_type],axis=1)
df4.head()

df5 = pd.concat([df4,smoking_status],axis=1)
df5.head()

df5.dropna(how='any')

df5.info()

df[pd.isna(df['bmi'])]

df5.dropna(how='any', inplace=True)

df5.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df5.head()

df5['le_ever_married'] = le.fit_transform(df5['ever_married'])

df5['le_gender'] = le.fit_transform(df5['gender'])

df5.head()

df6 = df5.copy().drop(['gender','ever_married'], axis=1)

X = df6.drop('stroke', axis=1)

y=df6['stroke']
y.head()

X.columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

lr = LogisticRegression()

lr_pipe = Pipeline([('sc',StandardScaler()),('lr',LogisticRegression())])

lr_pipe.fit(X_train,y_train)

pred = lr_pipe.predict(X_test)
pred

lr_pipe.score(X_test,y_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(pred,y_test)
cm

from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([('sc',StandardScaler()),('rf',RandomForestClassifier())])

rf_pipe.fit(X_train,y_train)

pred = rf_pipe.predict(X_test)
pred

rf_pipe.score(X_test,y_test)

cm = confusion_matrix(pred,y_test)
cm

from xgboost import XGBClassifier

xg = XGBClassifier()

xg_pipe = Pipeline([('sc',StandardScaler()),('xg',XGBClassifier())])

xg_pipe.fit(X_train,y_train)

pred = xg_pipe.predict(X_test)
pred

xg_pipe.score(X_test,y_test)

cm = confusion_matrix(pred,y_test)
cm

import pickle

with open('model_pickle', 'wb') as f:
    pickle.dump(lr_pipe, f)
    

model = pickle.load(open('model_pickle', 'rb'))
model.predict(X_test)

model.score(X_test,y_test)
model.predict([X_test.iloc[221,:].values])

pred = model.predict([[84, 1, 1, 230.94, 180.25, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]])
print(pred)

pred=model.predict(X_test)

df = pd.DataFrame({'Y_test' : y_test, 'Prediction':pred})
df[df['Prediction']==1]



    
    
















