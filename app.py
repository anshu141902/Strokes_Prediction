# -*- coding: utf-8 -*-
"""
Created on Sun May  2 13:49:40 2021

@author: 91892
"""

from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *

import pickle
import numpy as np
model = pickle.load(open('model_pickle', 'rb'))
app = Flask(__name__)


def predict():
    Age = input("Enter the Age：", type=NUMBER)
    Hypertension = select('Are you suffering from hypertension', ['Yes', 'No'])
    if (Hypertension == 'yes'):
        Hypertension = 1
        
    else: 
        Hypertension = 0
        
    Heart_desease = select('Are you suffering from Heart_desease', ['Yes', 'No'])
    if (Heart_desease == 'yes'):
        Heart_desease = 1
        
    else:
        Heart_desease = 0
        
    Average_Glucose_level = input("Enter the Average_Glucose_level:", type=FLOAT)
    
    BMI = input("Enter the BMI:", type=FLOAT)
    
    Work_Type = select('Which type of job are you doing', ['Govt_job', 'Never_worked','Private','Self_employed','Children'])
    if (Work_Type == 'Govt_job'):
        Govt_job = 1
        Never_worked = 0
        Private = 0
        Self_employed = 0
        Children = 0

    elif(Work_Type == 'Never_worked'):
        Govt_job = 0
        Never_worked = 1
        Private = 0
        Self_employed = 0
        Children = 0
        
    elif(Work_Type == 'Private'):
        Govt_job = 0
        Never_worked = 0
        Private = 1
        Self_employed = 0
        Children = 0
        
    elif(Work_Type == 'Self-employed'):
        Govt_job = 0
        Never_worked = 0
        Private = 0
        Self_employed = 1
        Children = 0
        
    elif(Work_Type == 'Children'):
        Govt_job = 0
        Never_worked = 0
        Private = 0
        Self_employed = 0
        Children = 1
        
    else:
        Govt_job = 0
        Never_worked = 0
        Private = 0
        Self_employed = 0
        Children = 0
    
    Residence_Type = select('Residence Type', ['Rural', 'Urban','Unknown'])
    if (Residence_Type == 'Rural'):
        Rural = 1
        Urban = 0
        Unknown = 0
        

    elif(Residence_Type == 'Urban'):
        Rural = 0
        Urban = 1
        Unknown = 0
        
    elif(Residence_Type == 'Unknown'):
        Rural = 0
        Urban = 0
        Unknown = 1
        
    else:
        Rural = 0
        Urban = 0
        Unknown = 0
        
    Smoker_Type = select('Smoker Type', ['Formerly_Smokes', 'Never_Smokes','Smokes'])
    if (Smoker_Type == 'Formerly_Smokes'):
        Formerly_Smokes = 1
        Never_Smokes = 0
        Smokes = 0
        

    elif(Smoker_Type == 'Never_Smokes'):
        Formerly_Smokes = 0
        Never_Smokes = 1
        Smokes = 0
        
    elif(Smoker_Type == 'Smokes'):
        Formerly_Smokes = 0
        Never_Smokes = 0
        Smokes = 1
        
    else:
        Formerly_Smokes = 0
        Never_Smokes = 0
        Smokes = 0
    
    Marital_Status = select('Selet Married or Not Married', ['Married', 'Not_Married'])
    if (Marital_Status == 'Married'):
        Married = 1
        Not_Married = 0

    elif(Marital_Status == 'Not_Married'):
        Married = 0
        Not_Married = 1
        
    else:
        Married = 0
        Not_Married = 0
        
    Gender = select('Selet Gender', ['Male', 'Female'])
    if (Gender == 'Male'):
        Male = 1
        Female = 0

    elif(Gender == 'Female'):
        Male = 0
        Female = 1
        
    else:
        Male = 0
        Female = 0
        
    prediction = model.predict([[Age, Hypertension, Heart_desease, Average_Glucose_level, BMI, Work_Type, Residence_Type, Smoker_Type, Marital_Status, Gender]])
    output = prediction

    if (output == 0):
        put_text('You are safe')

    else:
        put_text('You are in danger of having stroke')

app.add_url_rule('/tool', 'webio_view', webio_view(predict),
            methods=['GET', 'POST', 'OPTIONS'])


#if __name__ == '__main__':
    #predict()

app.run(host='localhost', port=80)

#visit http://localhost/tool to open the PyWebIO application.

