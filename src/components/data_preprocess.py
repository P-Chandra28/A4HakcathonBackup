from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.utilis import simplify_age_group

from src.components.data_tranforming import DataTransformation,DataTransformationConfig
from src.components.model_training import ModelTrainer,ModelConfig

@dataclass
class DataTrainingConfig:
    '''raw_data_path: str=os.path.join("artifacts","data.csv")'''
    raw_data_path: str=os.path.join("artifacts","input.csv")

class FeatureEngineering:
    try:
        def __init__(self):
            self.ingestion_config=DataTrainingConfig()
    
        def perform_feature_engineering(self,df):
        
            logging.info("Starting feature engineering")
            df['num_diagnoses'] = df[['diag_1','diag_2','diag_3']].count(axis=1)

            df['total_med_procedures'] = df['n_lab_procedures'] + df['n_procedures']

            def simplify_age_group(age_range):
                if pd.isna(age_range): return 'Other'
                if age_range in ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)']:
                    return 'Young'
                elif age_range in ['[50-60)','[60-70)','[70-80)']:
                    return 'Middle-aged'
                else:
                    return 'Senior'
            df['age_group_simplified'] = df['age'].apply(simplify_age_group)
            logging.info("Age group simplified")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Feature engineering completed and database has been updated")

            return df

        def initiate_data_preprocessing(self):

            df = pd.read_csv("notebook\data\hospital_knnimputer.csv")
            logging.info("data is read into a dataframe")
            df=self.perform_feature_engineering(df)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Ingestion over")
            return(
                    self.ingestion_config.raw_data_path
                )
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    obj=FeatureEngineering()
    raw_path=obj.initiate_data_preprocessing()

    obj=DataTransformation()
    train_arr,test_arr=obj.initiate_data_transformation(raw_path=raw_path)

    obj=ModelTrainer()
    obj.train_and_evaluate_model(X=train_arr,y=test_arr)




        
        