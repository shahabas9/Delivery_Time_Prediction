import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


#initialize the data ingestion configurtation

@dataclass
class Dataingestionconfig:
    raw_data_path:str=os.path.join('Artifacts','raw_data.csv')
    train_data_path:str=os.path.join('Artifacts','train_data.csv')
    test_data_path:str=os.path.join('Artifacts','test_data.csv')


# Create a class for data ingestion
class Dataingestion:
    def __init__(self):
        self.data_ingestion=Dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info("initializing the data ingestion method")
        try:
            file_path = os.path.join("/home/shahabas/shahabas/delivery_time_prediction/notebooks/data", "deliverytime.csv")
            df = pd.read_csv(file_path) 
            # df=pd.read_csv("notebooks/data/deliverytime.csv")
            logging.info("Dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.data_ingestion.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,index=False)
            logging.info("train test split")
            train_set,test_set=train_test_split(df,test_size=0.30)
            train_set.to_csv(self.data_ingestion.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion.test_data_path,index=False,header=True)
            logging.info("train test split completed")
            logging.info("data ingestion stage is completed")

            return(
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            )


        except Exception as e:
            logging.info("Exception occured at data ingestion method")
            raise CustomException(e,sys)