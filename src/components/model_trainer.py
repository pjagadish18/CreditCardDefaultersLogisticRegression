import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model
from dataclasses import dataclass
import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            "Logistic Regression":LogisticRegression(class_weight="balanced",max_iter=1000),
            "SVC":SVC(class_weight='balanced'),
            "Decision Tree":DecisionTreeClassifier(class_weight="balanced"),
            "Random Forest Classifier":RandomForestClassifier(class_weight="balanced",n_estimators=200)
            }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n====================================================================")
            logging.info(f'Model Report: {model_report}')

            # Select best model using F1 score for the default class
            best_model_name = max(model_report,key=lambda model_name: model_report[model_name]["f1"])

            best_model_score = model_report[best_model_name]["f1"]
            best_model = models[best_model_name]


            print(f"Best Model Found, Model Name: {best_model} and F1 score is: {best_model_score} ")
            print("\n==========================================================================")
            logging.info(f"Best Model Found, Model Name: {best_model} and F1 score is: {best_model_score} ")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.error("Error occurred at Model Training stage")
            raise CustomException(e,sys)