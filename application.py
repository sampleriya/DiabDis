from flask import Flask, request, jsonify ,app,render_template
from flask import Response
from flask_cors import CORS

import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application
CORS(app)


scaler=pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        data = request.get_json()
        pregnancies = int(data['Pregnancies'])
        glucose = float(data['Glucose'])
        blood_pressure = float(data['BloodPressure'])
        skin_thickness = float(data['SkinThickness'])
        insulin = float(data['Insulin'])
        bmi = float(data['BMI'])
        diabetes_pedigree_function = float(data['DiabetesPedigreeFunction'])
        age = float(data['Age'])

        new_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'

        return ({"result": result})
        # return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")