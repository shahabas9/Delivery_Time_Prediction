import os
import sys
import pandas as pd
import numpy as np
import math
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pickle
from datetime import datetime, timedelta

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        logging.info("Exception occured at saving the object")
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(x_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(x_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys) 
    
def load_object(file_path):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        logging.info("exception occured at loading the object")
        raise CustomException(e,sys)
    
def object_to_datetime(dataframe,col_name):
    try:

        dataframe[col_name]=pd.to_datetime(dataframe[col_name],infer_datetime_format=True)
    except Exception as e:
        logging.info("error occured when converting dattime colmn")
        raise CustomException(e,sys)
    
def read_data(filepath,object):
    try:
        data=pd.read_csv(os.path.join(filepath,object))
        return data
    except Exception as e:
        logging.info("error occured in reading in data")
        raise CustomException(e,sys)

def subtract_time(row):
    if row['Time_Orderd'] == '00:00':
        pickup_time = datetime.strptime(row['Time_Order_picked'], "%H:%M")
        time_diff = timedelta(hours=0, minutes=15)
        updated_time = pickup_time - time_diff
        return updated_time.strftime("%H:%M")
    else:
        return row['Time_Orderd']
    
def add_time(row):
    if row['Time_Order_picked'] == '00:00':
        orderd_time = datetime.strptime(row['Time_Orderd'], "%H:%M")
        time_diff = timedelta(hours=0, minutes=15)
        updated_time = orderd_time + time_diff
        return updated_time.strftime("%H:%M")
    else:
        return row['Time_Order_picked']


def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius_of_earth = 6371  # in kilometers
    distance = radius_of_earth * c

    return round(distance,2)
