import sys
from src.logger import logging

def error(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error generated at {0} occured in script {1} in line {2}".format(str(error),file_name,exc_tb.tb_lineno)
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error(error_message,error_detail=error_details)

    def __str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        result=1/0
    except Exception as e:
        logging.info("Divided by zero")
        raise CustomException(e,sys)
