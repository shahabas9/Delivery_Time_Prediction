import os 
import sys 
import pandas as pd 
import numpy as np 
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import Dataingestion
from src.components.data_transformation import Datatransformation
from src.components.model_training import ModelTrainer



# running this file:
if __name__=="__main__":
    obj=Dataingestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_tansformation=Datatransformation()
    train_arr,test_arr,object_path=data_tansformation.initiate_data_transformation(train_data_path,test_data_path)
    model_training=ModelTrainer()
    model_training.initiate_model_trainer(train_arr,test_arr)
