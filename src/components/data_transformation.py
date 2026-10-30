import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            # Categorical and numerical columns segregation
            categorical_columns=['SEX','EDUCATION','MARRIAGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
            numerical_columns=['LIMIT_BAL','AGE','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

            logging.info('Pipeline creation stage')
            # Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ])
            #Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder(drop="if_binary",handle_unknown="ignore",sparse_output=False)) # scaling not required for one hot encoding
                ]
            )
                
            preprocessor=ColumnTransformer([
                ('numerical pipeline',num_pipeline,numerical_columns),
                ('categorical pipeline',cat_pipeline,categorical_columns)
            ])
            logging.info("Pipeline Completed")

            return preprocessor
        except Exception as e:
            logging.error("Error occurred Data Transformation object")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Reading Train-Test data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read Train-Test data completed')
            logging.info(f"Train DataFrame Head :\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head :\n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='default payment next month'

            input_feature_train_df=train_df.drop(columns='default payment next month')
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns='default payment next month')
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on traing and testing dataset')

            #transforming using preprocessor object

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor Pickel file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error("Error ocuured in Initiate Data Transformation")
            raise CustomException(e,sys)
