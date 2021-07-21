# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:20:52 2021

@author: 91892
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import pickle
model_lr_pipe = pickle.load(open('model_lr_pipe_pickle', 'rb'))
model_rf_pipe = pickle.load(open('model_rf_pipe_pickle', 'rb'))
model_svc_pipe = pickle.load(open('model_svc_pipe_pickle', 'rb'))
model_xg_pipe = pickle.load(open('model_xg_pipe_pickle', 'rb'))


st.title('Strokes Prediction')
st.subheader('by Anshu')
st.markdown('----')
st.balloons()


Age = int(st.number_input('Enter the Age: '))

Hypertension = int(st.slider('Are you suffering from hypertension: 1 = Yes & 0 = No', 0,1))

Heart_desease = int(st.slider('Are you suffering from Heart_Deseases: 1 = Yes & 0 = No', 0,1))

Average_Glucose_level = int(st.number_input('Enter the Average_Glucose_level: '))

BMI = int(st.number_input('Enter the BMI: '))

Govt_Job = int(st.slider('Govt Job: 1 = Yes & 0 = No', 0,1))

Never_worked = int(st.slider('Never_worked: 1 = Yes & 0 = No', 0,1))

Private_Job = int(st.slider('Private Job: 1 = Yes & 0 = No', 0,1))

Self_employed = int(st.slider('Self_employed: 1 = Yes & 0 = No', 0,1))

Children = int(st.slider('Children: 1 = Yes & 0 = No', 0,1))

Rural_Residence = int(st.slider('Rural_Residence: 1 = Yes & 0 = No', 0,1))

Residence_Urban = int(st.slider('Residence_Urban: 1 = Yes & 0 = No', 0,1))

Residence_Unknown = int(st.slider('Residence_Unknown: 1 = Yes & 0 = No', 0,1))

Formerly_Smokes = int(st.slider('Are you a Formerly_Smoker: 1 = Yes & 0 = No', 0,1))

Never_Smokes = int(st.slider('Never_Smokes: 1 = Yes & 0 = No', 0,1))

Smoker = int(st.slider('Smoker: 1 = Yes & 0 = No', 0,1))

Marital_Status = int(st.slider('Marital_Status: 1 = Married & 0 = Not Married', 0,1))

Gender = int(st.slider('Gender_type: 1 = Male & 0 = Female', 0,1))

activities=['Logistic Regression','Random Forest Classifier','Support Vector Classifier','XG Booster Classifier']
option=st.sidebar.selectbox('Which model would you like to use?',activities)

inputs=np.array([[Age, Hypertension, Heart_desease, Average_Glucose_level, BMI, Govt_Job, Never_worked, Private_Job, Self_employed, Children, Rural_Residence, Residence_Urban, Residence_Unknown, Formerly_Smokes, Never_Smokes, Smoker, Marital_Status, Gender]])


if option=='Logistic Regression':
    model = model_lr_pipe
                         
elif option=='Random Forest Classifier':
    model = model_rf_pipe
                               
elif option=='Support Vector Classifier':
    model = model_svc_pipe
                  
else:
    model = model_xg_pipe
           

#def predict_Strokes(Age, Hypertension, Heart_desease, Average_Glucose_level, BMI, Govt_Job, Never_worked, Private_Job, Self_employed, Children, Rural_Residence, Residence_Urban, Residence_Unknown, Formerly_Smokes, Never_Smokes, Smoker, Marital_Status, Gender):
#   input=np.array([[Age, Hypertension, Heart_desease, Average_Glucose_level, BMI, Govt_Job, Never_worked, Private_Job, Self_employed, Children, Rural_Residence, Residence_Urban, Residence_Unknown, Formerly_Smokes, Never_Smokes, Smoker, Marital_Status, Gender]]).astype(np.float64)
#    prediction=model_lr_pipe.predict(input)
#    return prediction

def predict_Strokes(Age, Hypertension, Heart_desease, Average_Glucose_level, BMI, Govt_Job, Never_worked, Private_Job, Self_employed, Children, Rural_Residence, Residence_Urban, Residence_Unknown, Formerly_Smokes, Never_Smokes, Smoker, Marital_Status, Gender):
                prediction=model.predict(inputs)
                return prediction

            
if st.button("Prediction"):
     prediction = predict_Strokes(Age, Hypertension, Heart_desease, Average_Glucose_level, BMI, Govt_Job, Never_worked, Private_Job, Self_employed, Children, Rural_Residence, Residence_Urban, Residence_Unknown, Formerly_Smokes, Never_Smokes, Smoker, Marital_Status, Gender)
    
if prediction == 1:
           st.error('You are in danger of having Stokes')
else:
           st.success('You are Safe from strokes')
            



