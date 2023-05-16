import os 
import sys 
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from src.utils import save_object,evaluate_model
from src.logger import logging
from src.exception import CustomException 
from dataclasses import dataclass 
import numpy as np 
import pandas as pd 


@dataclass
class Model_trainer_config:
    trained_model_file_path= os.path.join("Artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=Model_trainer_config()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("seperating dependant and independant variables from train and test data") 
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "ElasticNet":ElasticNet(),
                "Decisionregressor":DecisionTreeRegressor(random_state=0),
                "RandomForest":RandomForestRegressor(n_estimators=1000,bootstrap=True,oob_score=True),
                "gb_model" : GradientBoostingRegressor(n_estimators=1000)
                

            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )




        except Exception as e:
            logging.info("Error occcured at model training")
            raise CustomException(e,sys)
        
