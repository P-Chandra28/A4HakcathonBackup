from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/visualize")
def visualize():
    return render_template("vis.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data=CustomData(
        time_in_hospital= int(request.form.get("time_in_hospital")),
        n_lab_procedures= int(request.form.get("n_lab_procedures")),
        n_procedures= int(request.form.get("n_procedures")),
        n_medications= int(request.form.get("n_medications")),
        n_outpatient= int(request.form.get("n_outpatient")),
        n_inpatient= int(request.form.get("n_inpatient")),
        n_emergency= int(request.form.get("n_emergency")),
        medical_specialty= request.form.get("medical_specialty"),
        diag_1= request.form.get("diag_1"),
        diag_2= request.form.get("diag_2"),
        diag_3= request.form.get("diag_3"),
        glucose_test= request.form.get("glucose_test"),
        A1Ctest= request.form.get("A1Ctest"),
        change= request.form.get("change"),
        diabetes_med= request.form.get("diabetes_med"),
        age= request.form.get("age"),
        )
        pred_df=data.get_as_df()
        
        pred_pipeline=PredictPipeline()
        results=pred_pipeline.predicts(pred_df)
        probability = results[0][0] 
        if probability>0.70:
            risk="(High-risk Observed)"
        elif probability>0.30:
            risk="(Medium-risk Observed)"
        else:
            risk="(Low-risk Observed)"
        probability*=100
        return render_template(template_name_or_list="home.html",results=f"{probability:.2f}% chance of readmitting",riskvalue=risk)
    
if __name__=="__main__":
    app.run(host="0.0.0.0")