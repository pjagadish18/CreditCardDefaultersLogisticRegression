import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score, balanced_accuracy_score
import pickle
from src.exception import CustomException
from src.logger import logging




# Removing duplcate rows from data frame

def remove_duplicates(df):
    columns_to_check=[col for col in df.columns if col !='ID']
    df = df.drop_duplicates(subset=columns_to_check, keep="first", ignore_index=True)
    return df


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report=dict()

        for model_name,model in models.items():
            #Train model
            model.fit(X_train,y_train)

            #Predict Testing data
            y_test_pred=model.predict(X_test)

            # ROC-AUC should use probability of the positive/default class
            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = y_test_pred

            report[model_name] = {
                "f1": f1_score(y_test, y_test_pred),
                "recall_default": recall_score(y_test, y_test_pred),
                "roc_auc": roc_auc_score(y_test, y_test_proba),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred)
            }

        return report

    except Exception as e:
        logging.error("Exception Occured during model training")
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.error("Error occurred in Load Object")
        raise CustomException(e, sys)