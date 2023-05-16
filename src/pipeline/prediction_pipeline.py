import sys 
import os 
from src.exception import CustomException
from src.logger import logging 
from src.utils import load_object,calculate_distance
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass 
    def predict(self, features):
        try:
            preprocessor_path=os.path.join("/home/shahabas/shahabas/delivery_time_prediction/src/pipeline/Artifacts","preprocessor.pkl")
            model_path=os.path.join("/home/shahabas/shahabas/delivery_time_prediction/src/pipeline/Artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            distance = calculate_distance(features['Restaurant_latitude'], features['Restaurant_longitude'], features['Delivery_location_latitude'], features['Delivery_location_longitude'])
            features['distance'] = distance
            features["Order_Date"] = pd.to_datetime(features["Order_Date"],dayfirst=True)
            features['Order_Day'] = features["Order_Date"].dt.day
            features['Order_Month'] = features["Order_Date"].dt.month
            features['Order_Year'] = features["Order_Date"].dt.year
            features['orderd_hour'] = features['Time_Orderd'].str.split(':').str[0].astype(int)
            features['orderd_minute'] = features['Time_Orderd'].str.split(':').str[1].astype(int)
            features['orderd_picked_hour'] = features['Time_Order_picked'].str.split(':').str[0].astype(int)
            features['orderd_picked_minute'] = features['Time_Order_picked'].str.split(':').str[1].astype(int)
            drop_columns=['Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked']
            features=features.drop(columns=drop_columns,axis=1)
            
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return int(np.round(pred,decimals=0))

        


        except Exception as e:
            logging.info("exception occured at prediction")
            raise CustomException(e,sys) 
        
class CustomData:
    def __init__(self,
    Delivery_person_Age: float, 
    Delivery_person_Ratings: float, 
    Restaurant_latitude: float,
    Restaurant_longitude: float,
    Delivery_location_latitude: float,
    Delivery_location_longitude: float,
    Order_Date: str,
    Time_Orderd: str,
    Time_Order_picked: str,
    Weather_conditions: str,
    Road_traffic_density: str,
    Vehicle_condition: float, 
    Type_of_order: str,
    Type_of_vehicle: str,
    multiple_deliveries: float, 
    Festival: str, 
    City: str):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Restaurant_latitude = Restaurant_latitude
        self.Restaurant_longitude = Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Order_Date = Order_Date
        self.Time_Orderd = Time_Orderd
        self.Time_Order_picked = Time_Order_picked
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density 
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City
    
    def get_data_as_DataFrame(self):
        try:

            custom_input_data = {
                'Delivery_person_Age': [self.Delivery_person_Age], 
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Restaurant_latitude':[self.Restaurant_latitude],
                'Restaurant_longitude':[self.Restaurant_longitude],
                'Delivery_location_latitude':[self.Delivery_location_latitude],
                'Delivery_location_longitude':[self.Delivery_location_longitude],
                'Order_Date':[self.Order_Date],
                'Time_Orderd':[self.Time_Orderd],
                'Time_Order_picked':[self.Time_Order_picked],
                'Weather_conditions': [self.Weather_conditions],
                'Road_traffic_density': [self.Road_traffic_density],
                'Vehicle_condition': [self.Vehicle_condition],
                'Type_of_order': [self.Type_of_order],
                'Type_of_vehicle': [self.Type_of_vehicle],
                'multiple_deliveries': [self.multiple_deliveries], 
                'Festival': [self.Festival], 
                'City': [self.City],
                 
            }
            data = pd.DataFrame(custom_input_data)
            logging.info('Dataframe Gathered')
            return data
            
            
        except Exception as e:
            logging.info("exception occured at getting dataframe.")
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=CustomData(28,4.6,23.5841,74.1245,68.37194,33.791346,"20-02-2022","23:15","23:55","Fog","Jam",0,"Buffet","motorcycle",1,"No","Urban")
    df=obj.get_data_as_DataFrame()
    print(df.head())
    obj2=PredictPipeline()
    pre=obj2.predict(df)
    print(pre)
