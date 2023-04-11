import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import datetime as dt
from math import ceil
from Utils import *

pd.set_option('max_colwidth', None)

def data_loading():
    def_feature = pd.read_csv("input/Xente_Variable_Definitions.csv")
    raw_data = pd.read_csv("input/training.csv")
    X_test = pd.read_csv("input/test.csv")
    sample_submission = pd.read_csv("input/sample_submission.csv")
    raw_data['TransactionStartTime'] = pd.to_datetime(raw_data['TransactionStartTime'])
    X_test['TransactionStartTime'] = pd.to_datetime(X_test['TransactionStartTime'])
    return def_feature, raw_data, X_test, sample_submission

def get_data():
    def_feature, raw_data, X_test, sample_submission = data_loading()

    #Data cleaning
    data = raw_data.copy()
    data = data.dropna(axis=0) #Drop observations/rows with missing values
    data, cols_unique_value = delete_col_unique_val(data) #deal with unique values

    #Adding data
    data = adding_date_col(data, 'TransactionStartTime')
    X_test = adding_date_col(X_test, 'TransactionStartTime')

    #Data splitting
    y = data.FraudResult #The target label
    X = data.copy()
    X.drop(['FraudResult'], axis=1, inplace=True) #Only the features data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

    #Other data
    #Information on columns on raw_data
    info = pd.DataFrame(data = raw_data.dtypes)
    info.reset_index(inplace=True)
    info.rename({'index':'Column Name', 0: 'Dtype'}, axis=1, inplace=True)
    describe = def_feature.copy()
    describe = describe.merge(info)
    unique_val = []
    for col in list(describe["Column Name"]) : 
        unique_val.append(len(raw_data[col].unique()))
    describe["unique"]=unique_val
    describe = added_column(describe)
    object_cols = [col for col in train_X.columns if train_X[col].dtype == "object"]#liste of obejct columns
    
    return train_X, val_X, train_y, val_y, X, y, raw_data, X_test, data, describe, object_cols
