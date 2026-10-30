import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import remove_duplicates
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

## Initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


## create a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')

        try:
            df=pd.read_csv('notebooks/Data/default_of_credit_card_clients.csv')
            logging.info('Dataset read as pandas dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Renaming column PAY_0 ->  PAY_1, Removing Duplicate records and dropping correlated features')
            df = df.rename(columns={"PAY_0": "PAY_1"})  
            df=remove_duplicates(df)
            df=df.drop(labels=['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis=1)

            logging.info('Train-Test split')
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.error("Error Occurred at Data Ingestion stage")
            raise CustomException(e,sys)


### Data Ingestion testing

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_tansformation=DataTransformation()
    train_arr,test_arr,preprocessor_path=data_tansformation.initiate_data_transformation(train_data_path,test_data_path)