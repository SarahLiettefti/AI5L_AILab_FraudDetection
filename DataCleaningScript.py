import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
from math import ceil
from Utils import week_of_month

pd.set_option('max_colwidth', None)

class pre_processing:
    def __init__(self, report = False, random_state=0, one_hot_encoder=True):
        #data loading
        self.def_feature = pd.read_csv("input/Xente_Variable_Definitions.csv")
        self.raw_data = pd.read_csv("input/training.csv")
        self.X_test = pd.read_csv("input/test.csv")
        self.sample_submission = pd.read_csv("input/sample_submission.csv")

        #attribut initialization
        self.cols_unique_value = [] #Will be droped
        self.medium_cardianlity_cols = ["ProductId"]

        #Data transformation
        self.raw_data['TransactionStartTime'] = pd.to_datetime(self.raw_data['TransactionStartTime'])
        self.X_test['TransactionStartTime'] = pd.to_datetime(self.X_test['TransactionStartTime'])

        #Data cleaning
        self.data = self.raw_data.copy()
        self.data = self.data.dropna(axis=0) #Drop observations/rows with missing values
        self.data = self.delete_col_unique_val(self.data, report) #deal with unique values

        #Set the index to the Transactions Id
        self.data = self.transactioId_to_index(self.data)
        self.X_test = self.transactioId_to_index(self.X_test)

        #Adding data
        self.data = self.adding_date_col(self.data, 'TransactionStartTime')
        self.X_test = self.adding_date_col(self.X_test, 'TransactionStartTime')

        # "Cardinality" means the number of unique values in a column
        # Select categorical columns with relatively low cardinality (convenient but arbitrary)
        self.low_cardinality_cols = [cname for cname in self.data.columns if self.data[cname].nunique() < 15 and 
                                self.data[cname].dtype == "object"]
        
        #Transform string and categorical into numbers
        self.l_col_str = ["BatchId", "AccountId", "SubscriptionId", "CustomerId", "ProviderId", "ProductId", "ChannelId"]
        for col in self.l_col_str:
            self.data[['dc', 'new_col']] = self.data[col].str.split("_", expand = True)
            self.data.drop(['dc',col], inplace=True, axis=1)
            self.data.rename(columns={"new_col": col}, inplace=True)
            self.data[col] = self.data[col].astype('int')
            self.X_test[['dc', 'new_col']] = self.X_test[col].str.split("_", expand = True)
            self.X_test.drop(['dc',col], inplace=True, axis=1)
            self.X_test.rename(columns={"new_col": col}, inplace=True)
            self.X_test[col] = self.X_test[col].astype('int')
        
        #Data splitting
        self.y = self.data.FraudResult #The target label
        self.X = self.data.copy()
        self.X.drop(['FraudResult'], axis=1, inplace=True) #Only the features data
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, random_state = random_state)

        #Other data
        #Information on columns on raw_data
        ## P-e transformer ca en une fonction ? 
        info = pd.DataFrame(data = self.raw_data.dtypes)
        info.reset_index(inplace=True)
        info.rename({'index':'Column Name', 0: 'Dtype'}, axis=1, inplace=True)
        self.describe = self.def_feature.copy()
        self.describe = self.describe.merge(info)
        unique_val = []
        for col in list(self.describe["Column Name"]) : 
            unique_val.append(len(self.raw_data[col].unique()))
        self.describe["unique"]=unique_val
        self.describe = self.added_column(self.describe)
        self.cat_cols = [col for col in self.train_X.columns if self.train_X[col].dtype == "object"]#liste of obejct columns
        self.cat_cols.append("PricingStrategy")#pcq mm si c'est un chiffre il faut le considérer comme une catégorie
        
        if(report):
            print(self.describe)
            
        if one_hot_encoder:
            
            #low_cardinality_cols=["ProviderId", "ProductCategory", "ChannelId", "PricingStrategy"] 
            self.train_X[self.low_cardinality_cols] = self.train_X[self.low_cardinality_cols].astype(str) 
            self.val_X[self.low_cardinality_cols] = self.val_X[self.low_cardinality_cols].astype(str) 

            OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(self.train_X[self.low_cardinality_cols]))
            OH_cols_valid = pd.DataFrame(OH_encoder.transform(self.val_X[self.low_cardinality_cols]))
            # One-hot encoding removed index; put it back
            OH_cols_train.index = self.train_X.index
            OH_cols_valid.index = self.val_X.index

            # Remove categorical columns (will replace with one-hot encoding)
            #num_X_train = train_X.drop(cat_cols, axis=1)
            #num_X_valid = val_X.drop(cat_cols, axis=1)
            num_X_train = self.train_X.drop(self.low_cardinality_cols, axis=1)
            num_X_valid = self.val_X.drop(self.low_cardinality_cols, axis=1)

            OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
            OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

            # Problème de string
            OH_X_train.drop(['TransactionStartTime'], inplace=True, axis=1)
            OH_X_valid.drop(['TransactionStartTime'], inplace=True, axis=1)
            OH_X_train.columns = OH_X_train.columns.astype(str)
            OH_X_valid.columns = OH_X_valid.columns.astype(str)

            self.train_X = OH_X_train
            self.val_X = OH_X_valid


    def delete_col_unique_val(self, X, report = False):
        for col in X.columns : 
            if len(X[col].unique()) == 1:
                self.cols_unique_value.append(col)
        if report :
            print(f'The colomns : {self.cols_unique_value} have only one value and have been dropped')

        X.drop(self.cols_unique_value, axis=1, inplace=True)
        return X
    
    def transactioId_to_index(self, data):
        data[['dc', 'new_index']] = data.TransactionId.str.split("_", expand = True)
        data.drop(['dc','TransactionId'], inplace=True, axis=1)
        data.rename(columns={"new_index": "TransactionId"}, inplace=True)
        data.set_index('TransactionId', inplace=True)
        return data
    
    def adding_date_col(self, X, date_col):
        day = [i.day for i in X[date_col]]
        hour = [i.hour for i in X[date_col]]
        l_week = [week_of_month(x) for x in X[date_col].tolist()]
        X['Day'] = day
        X['Hour'] = hour
        X['week_day'] = X[date_col].dt.dayofweek
        X["weeks"] = l_week
        return X
    
    def added_column(self, describe):
        """
        Function to add information on  new columns and keep the describe's df updated
        If there is a missing data in one column, replace it with : np.nan
        """
        new_col = [
            ["Day", "The day in the month of the transaction", "int64", np.nan],
            ["Hour", "The hour of the transaction", "int64", np.nan],
            ["week_day", "The day of the week of the transaction", "int64", np.nan],
            ["weeks", "The week of the month of the transaction", "int64", np.nan]
        ]
        df1 = pd.DataFrame(new_col, columns=['Column Name', 'Definition', "Dtype", "unique"])
        describe = pd.concat([describe, df1])
        describe.reset_index(inplace=True)
        describe.drop(['index'], axis=1, inplace=True) #Only the features data
        return describe