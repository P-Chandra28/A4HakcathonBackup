from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utilis import save_object 
from imblearn.combine import SMOTEENN


@dataclass
class DataTransformationConfig:
    preprocessor_objfilepath=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tansformation_config=DataTransformationConfig()

    def preprocess_data(self,df:pd.DataFrame,target_col="readmitted"):
        logging.info("data preprocessing has begun")

        df = df.drop(columns=['age','diag_1', 'diag_2', 'diag_3'], axis=1)
        logging.info("Dropped unnecessary and modified columns")

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        logging.info("Cleaned column names to remove special characters")
        
        top_n_specialties = df['medical_specialty'].value_counts().nlargest(10).index.tolist()
        df['medical_specialty'] = df['medical_specialty'].apply(lambda x: x if x in top_n_specialties else 'Other')
        logging.info("Grouped less frequent medical specialties into different category")

        categorical_cols = ['medical_specialty','A1Ctest','change','diabetes_med','age_group_simplified','glucose_test']
        

        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logging.info("Applied one-hot encoding to categorical columns")
        df.columns = [str(c).replace('[','_').replace(']','_').replace('<','_').replace(',','_')
                    .replace('(','_').replace(')','_').replace(' ','_') for c in df.columns]
        
        return df
        
        

    def X_y_split(self,df:pd.DataFrame,target_col="readmitted"):
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            logging.info("Dropped unnecessary column 'Unnamed: 0'")

        
        
        df[target_col] = df[target_col].map({'no':0,'yes':1})
        logging.info("Mapped target column to numerical values")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        logging.info("Separated features and target variable")
        return X,y
        
    def initiate_data_transformation(self,raw_path):
        try:
            raw_data=pd.read_csv(raw_path)
            preprocess_data=self.preprocess_data(raw_data)

            train_data,test_data=self.X_y_split(preprocess_data,target_col="readmitted")         

            logging.info("data transformation is complete")

            save_object(
                file_path=self.data_tansformation_config.preprocessor_objfilepath,
                obj=self.preprocess_data
                )

            return(
                train_data,test_data
            )
        except Exception as e:
            raise CustomException(e,sys)
    
