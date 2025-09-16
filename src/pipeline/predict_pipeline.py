import os
import sys
import joblib
import pandas as pd
from src.exception import CustomException
from src.utilis import load_object
from src.components.data_tranforming import DataTransformation,DataTransformationConfig
from src.components.data_preprocess import DataTrainingConfig,FeatureEngineering

class PredictPipeline:
    def __init__(self):
        pass

    def predicts(self,features: pd.DataFrame):
        model_path="artifacts\model.pkl"
        model=load_object(file_path=model_path)
        data = FeatureEngineering().perform_feature_engineering(features)
        data = DataTransformation().preprocess_data(data)
        # After preprocessing and before model.fit()
        import joblib
        feature_names = joblib.load("artifacts/feature_names.pkl")
        
        for col in feature_names:
            if col not in data.columns:
                data[col] = 0
        data = data[feature_names]


        data = data[feature_names]
        preds=model.predict_proba(data)

        return preds
class CustomData:
    def __init__(self,
    age        :  str,
    time_in_hospital:  int ,
  n_lab_procedures  :  int ,
  n_procedures      :  int ,
  n_outpatient     :    int ,
  n_medications    :    int ,
  n_inpatient      :    int ,
  n_emergency      :    int ,
  medical_specialty:    str,
  diag_1           :    str,
  diag_2           :    str,
  diag_3           :    str,
  glucose_test     :    str,
  A1Ctest          :    str,
  change           :    str,
  diabetes_med     :    str
    ):
        self.age= age
        self.time_in_hospital= time_in_hospital
        self.n_outpatient=n_outpatient
        self.n_lab_procedures= n_lab_procedures
        self.n_inpatient=n_inpatient
        self.n_procedures= n_procedures
        self.n_medications= n_medications
        self.n_emergency= n_emergency
        self.medical_specialty= medical_specialty
        self.diag_1= diag_1
        self.diag_2= diag_2
        self.diag_3= diag_3
        self.glucose_test= glucose_test
        self.A1Ctest= A1Ctest
        self.change= change
        self.diabetes_med= diabetes_med
    
    
    def get_as_df(self):
        try:
            custom_input={
            "age":[self.age],
            "time_in_hospital":[self.time_in_hospital],
            "n_lab_procedures":[self.n_lab_procedures],
            "n_procedures":[self.n_procedures],
            "n_medications":[self.n_medications],
            "n_outpatient":[self.n_outpatient],
            "n_inpatient":[self.n_inpatient],
            "n_emergency":[self.n_emergency],
            "medical_specialty":[self.medical_specialty],
            "diag_1":[self.diag_1],
            "diag_2":[self.diag_2],
            "diag_3":[self.diag_3],
            "glucose_test":[self.glucose_test],
            "A1Ctest":[self.A1Ctest],
            "change":[self.change],
            "diabetes_med":[self.diabetes_med],
            }
            df=pd.DataFrame(custom_input)
            

            
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
