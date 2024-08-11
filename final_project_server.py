# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:33:07 2024

@author: sndp_
"""
import pandas as pd
from flask import Flask, request, render_template
import joblib
import sys

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    n_features = []
    for i in request.form.values():
        n_features.append(i)
    cols = ['STREET1', 'STREET2', 'ROAD_CLASS',
       'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY',
       'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'AUTOMOBILE',
       'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER',
       'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY',
       'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140']
    print(n_features)
    features = pd.DataFrame([n_features],columns=cols)
    print(features)
    # trans_data= preprocessor.transform(features)
    # print(trans_data)
    prediction = lr.predict(features)
    print(prediction)
    result = "Fatal" if prediction[0] == 1 else "Non-Fatal"
    return render_template("result.html", prediction = result )

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('D:/vsCode/comp247/final-project/best_model.pkl') # Load "model.pkl"
    print ('Model loaded')
    preprocessor = joblib.load('D:/vsCode/comp247/final-project/preprocessor.pkl')
    print ('preprocessor loaded')
    
    
    app.run(port=port, debug=True)