import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import sys
from src.utils import save_object,subtract_time,add_time,calculate_distance


@dataclass
class Datatransformation_config:
    preprocessor_obj_file_path=os.path.join("Artifacts","preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.data_transformation_config=Datatransformation_config()
    
    def get_data_transformation_obj(self):
        try:
            
            logging.info("data transformation initiated")
            # numerical and categorical columns were separated
            numerical_columns=['Delivery_person_Age','distance']
            categorical_numerical=['Delivery_person_Ratings','Order_Year','Order_Month','Order_Day','orderd_hour','orderd_minute','orderd_picked_hour','orderd_picked_minute']
            categorical_columns=['Weather_conditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City','Vehicle_condition','multiple_deliveries']

            

            logging.info("pipeline initiated")
            # numerical pipeline
            num_pipeline1=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='mean')),
                ("scaler",StandardScaler())
                ]
            )
            num_pipeline2=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("scaler",StandardScaler())
                ]
            )


            # categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ('OneHotEncoder',OneHotEncoder(sparse=False)),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            # joining numerical and categorical pipeline
            preprocessor=ColumnTransformer([
                ("numerical_pipeline",num_pipeline1,numerical_columns),
                ("cat_numerical_pipeline",num_pipeline2,categorical_numerical),
                ("categorical_pipeline",cat_pipeline,categorical_columns)
            ])
            logging.info("pipeline completed")
            return preprocessor
            


            
        except Exception as e:
            logging.info("Exception occured at data transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Reading train and test data started")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test completed")
            logging.info(f"Train DataFrame head :\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head :\n{test_df.head().to_string()}")
            logging.info(" Getting preprocessor object")

            train_df['distance']=train_df.apply(lambda row: calculate_distance(row['Restaurant_latitude'],
                                                             row['Restaurant_longitude'],
                                                             row['Delivery_location_latitude'],
                                                             row['Delivery_location_longitude']), axis=1)


            test_df['distance']=test_df.apply(lambda row: calculate_distance(row['Restaurant_latitude'],
                                                             row['Restaurant_longitude'],
                                                             row['Delivery_location_latitude'],
                                                             row['Delivery_location_longitude']), axis=1)
            

            logging.info("Convert Order_Date column to datetime")
            train_df["Order_Date"] = pd.to_datetime(train_df["Order_Date"],dayfirst=True)
            test_df["Order_Date"] = pd.to_datetime(test_df["Order_Date"],dayfirst=True)

            logging.info("Extract year, month, and day into separate columns")
            train_df["Order_Year"] = train_df["Order_Date"].dt.year
            train_df["Order_Month"] = train_df["Order_Date"].dt.month
            train_df["Order_Day"] = train_df["Order_Date"].dt.day

            test_df["Order_Year"] = test_df["Order_Date"].dt.year
            test_df["Order_Month"] = test_df["Order_Date"].dt.month
            test_df["Order_Day"] = test_df["Order_Date"].dt.day

            logging.info("Extract hour and minute from time ordered and time picked order")
            train_df["Time_Orderd"]=train_df["Time_Orderd"].astype('str')
            train_df["Time_Orderd"]=train_df["Time_Orderd"].apply(lambda x : x[0] if "." in x else x)
            train_df["Time_Orderd"]=train_df["Time_Orderd"].apply(lambda x :'0'+x+":00" if x=='1' else x)
            train_df["Time_Orderd"]=train_df["Time_Orderd"].apply(lambda x :'01'+x[2:5] if x[0:2]=='24' else x)
            train_df["Time_Orderd"]=train_df["Time_Orderd"].apply(lambda x :'0' if x=='nan'else x)
            train_df["Time_Orderd"]=train_df["Time_Orderd"].apply(lambda x :'00:00' if x=='0'else x)

            train_df["Time_Order_picked"]=train_df["Time_Order_picked"].astype('str')
            train_df["Time_Order_picked"]=train_df["Time_Order_picked"].apply(lambda x : x[0] if "." in x else x)
            train_df["Time_Order_picked"]=train_df["Time_Order_picked"].apply(lambda x :'0'+x+":00" if x=='1' else x)
            train_df["Time_Order_picked"]=train_df["Time_Order_picked"].apply(lambda x :'01'+ x[2:5] if x[0:2]=='24' else x)
            train_df["Time_Order_picked"]=train_df["Time_Order_picked"].apply(lambda x :'0' if x=='nan'else x)
            train_df["Time_Order_picked"]=train_df["Time_Order_picked"].apply(lambda x :'00:00' if x=='0'else x)

            train_df['Time_Orderd'] = np.where(train_df['Time_Orderd'] == '00:00', train_df.apply(subtract_time, axis=1), train_df['Time_Orderd'])
            train_df['Time_Order_picked'] = np.where(train_df['Time_Order_picked'] == '00:00', train_df.apply(add_time, axis=1), train_df['Time_Order_picked'])

            test_df["Time_Orderd"]=test_df["Time_Orderd"].astype('str')
            test_df["Time_Orderd"]=test_df["Time_Orderd"].apply(lambda x : x[0] if "." in x else x)
            test_df["Time_Orderd"]=test_df["Time_Orderd"].apply(lambda x :'0'+x+":00" if x=='1' else x)
            test_df["Time_Orderd"]=test_df["Time_Orderd"].apply(lambda x :'01'+x[2:5] if x[0:2]=='24' else x)
            test_df["Time_Orderd"]=test_df["Time_Orderd"].apply(lambda x :'0' if x=='nan'else x)
            test_df["Time_Orderd"]=test_df["Time_Orderd"].apply(lambda x :'00:00' if x=='0'else x)

            test_df["Time_Order_picked"]=test_df["Time_Order_picked"].astype('str')
            test_df["Time_Order_picked"]=test_df["Time_Order_picked"].apply(lambda x : x[0] if "." in x else x)
            test_df["Time_Order_picked"]=test_df["Time_Order_picked"].apply(lambda x :'0'+x+":00" if x=='1' else x)
            test_df["Time_Order_picked"]=test_df["Time_Order_picked"].apply(lambda x :'01'+ x[2:5] if x[0:2]=='24' else x)
            test_df["Time_Order_picked"]=test_df["Time_Order_picked"].apply(lambda x :'0' if x=='nan'else x)
            test_df["Time_Order_picked"]=test_df["Time_Order_picked"].apply(lambda x :'00:00' if x=='0'else x)

            test_df['Time_Orderd'] = np.where(test_df['Time_Orderd'] == '00:00', test_df.apply(subtract_time, axis=1), test_df['Time_Orderd'])
            test_df['Time_Order_picked'] = np.where(test_df['Time_Order_picked'] == '00:00', test_df.apply(add_time, axis=1), test_df['Time_Order_picked'])

            train_df["orderd_hour"]=train_df["Time_Orderd"].apply(lambda x : x[0:2])
            train_df["orderd_minute"]=train_df["Time_Orderd"].apply(lambda x : x[3:])

            test_df["orderd_hour"]=test_df["Time_Orderd"].apply(lambda x : x[0:2])
            test_df["orderd_minute"]=test_df["Time_Orderd"].apply(lambda x : x[3:])

            train_df["orderd_picked_hour"]=train_df["Time_Order_picked"].apply(lambda x : x[0:2])
            train_df["orderd_picked_minute"]=train_df["Time_Order_picked"].apply(lambda x : x[3:])

            test_df["orderd_picked_hour"]=test_df["Time_Order_picked"].apply(lambda x : x[0:2])
            test_df["orderd_picked_minute"]=test_df["Time_Order_picked"].apply(lambda x : x[3:])


            logging.info("converting the new columns into int datatype")
            train_df["orderd_hour"]=train_df["orderd_hour"].astype(int)
            train_df["orderd_minute"]=train_df["orderd_minute"].astype(int)
            train_df["orderd_picked_hour"]=train_df["orderd_picked_hour"].astype(int)
            train_df["orderd_picked_minute"]=train_df["orderd_picked_minute"].astype(int)


            test_df["orderd_hour"]=test_df["orderd_hour"].astype(int)
            test_df["orderd_minute"]=test_df["orderd_minute"].astype(int)
            test_df["orderd_picked_hour"]=test_df["orderd_picked_hour"].astype(int)
            test_df["orderd_picked_minute"]=test_df["orderd_picked_minute"].astype(int)




            preprocessor_obj=self.get_data_transformation_obj()

            target_column_name='Time_taken (min)'
            drop_columns=[target_column_name,'ID', 'Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            logging.info(f"Train DataFrame head :\n{input_feature_train_df.head().to_string()}")
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            logging.info(f"Train DataFrame head :\n{input_feature_test_df.head().to_string()}")
            target_feature_test_df=test_df[target_column_name]
            

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing on train and test data")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            

            save_object(
                file_path=Datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("preprocessor pickle file saved")

            return(
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Exception occured at initiate_data_transformation")
            raise CustomException(e,sys)