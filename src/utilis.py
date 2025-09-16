import os
import sys
import dill
import pandas as pd


from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def simplify_age_group(age_range):
    if pd.isna(age_range): return 'Other'
    if age_range in ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)']:
        return 'Young'
    elif age_range in ['[50-60)','[60-70)','[70-80)']:
        return 'Middle-aged'
    else:
        return 'Senior'
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)