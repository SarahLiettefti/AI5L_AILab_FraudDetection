import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import datetime as dt
from math import ceil
pd.set_option('max_colwidth', None)

def score_dataset(X_train, X_valid, y_train, y_valid):#fucntion to test but have to improve
    """
    Function to get an idea of performance
    To improve !! 
    """
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def delete_col_unique_val(X):
    """
    Delete the columns with unique values
    """
    cols_unique_value = []

    for col in X.columns : 
        if len(X[col].unique()) == 1:
            cols_unique_value.append(col)
        else :
            pass

    print(f'The colomns : {cols_unique_value} have only one value and have been dropped')

    X.drop(cols_unique_value, axis=1, inplace=True)
    return X, cols_unique_value

def week_of_month(dt):
    """
    Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)

    dom = dt.day
    week_day = first_day.weekday()#monday = 0
    adjusted_dom = dom + week_day
    
    if week_day>=5:
        return int(ceil(adjusted_dom/7.0)) - 1
    else:
        return int(ceil(adjusted_dom/7.0))

def adding_date_col(X, date_col):
    """
    Adding the date related columns
    """
    day = [i.day for i in X[date_col]]
    hour = [i.hour for i in X[date_col]]
    l_week = [week_of_month(x) for x in X[date_col].tolist()]
    X['Day'] = day
    X['Hour'] = hour
    X['week_day'] = X[date_col].dt.dayofweek
    X["weeks"] = l_week
    return X

def added_column(describe):
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