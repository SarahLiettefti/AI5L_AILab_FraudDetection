from sklearn.metrics import confusion_matrix, log_loss, matthews_corrcoef
import pandas as pd
import numpy as np

def precision(val_y, pred_y):
    cm = confusion_matrix(val_y, pred_y)
    tn, fp, fn, tp = cm.ravel()
    metric = tp / (tp+fp)
    return metric

def recall(val_y, pred_y):
    cm = confusion_matrix(val_y, pred_y)
    tn, fp, fn, tp = cm.ravel()
    metric = tp / (tp+fn)
    return metric

def f1score(val_y, pred_y):
    cm = confusion_matrix(val_y, pred_y)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    metric = 2*((precision*recall)/(precision+recall))
    return metric

def logloss(val_y, pred_y):
    metric = log_loss(val_y, pred_y)
    return metric

def mcc(val_y, pred_y):
    metric = matthews_corrcoef(val_y, pred_y)
    return metric


def report(val_y, pred_y, model, description, csvw = False):
    prec = precision(val_y, pred_y)
    re = recall(val_y, pred_y)
    f1 = f1score(val_y, pred_y)
    log = logloss(val_y, pred_y)
    mc = mcc(val_y, pred_y)
    d = {'Model': [model], 'Description': [description], 'Date':[pd.Timestamp.now()],
         'Precision' : [prec], 'Recall' : [re], 'F1-score' : [f1], 'LogLoss' : [log], 'Mcc' : [mc]}
    newdf = pd.DataFrame(data = d)
    if csvw : 
        newdf.to_csv('evaluationmetric.csv', mode='a', header=False)      
    return newdf

def listmetrics(val_y, pred_y, model, description):
    prec = precision(val_y, pred_y)
    re = recall(val_y, pred_y)
    f1 = f1score(val_y, pred_y)
    log = logloss(val_y, pred_y)
    mc = mcc(val_y, pred_y)
    d = [model, description, pd.Timestamp.now(), prec, re, f1, log, mc, np.nan, np.nan]
    return d

def listmetricsintodf(liste):
    df = pd.DataFrame(liste, columns=['Model', 'Description', 'Date', 'Precision', 'Recall','F1-score','LogLoss','Mcc','PublicScore','PrivateScore'])
    return df


def initreportcsv():
    d = {'Model': ["None"], 'Description': ['Table init'], 'Date':[pd.Timestamp.now()],
         'Precision' : [0], 'Recall' : [0], 'F1-score' : [0], 'LogLoss' : [0], 'Mcc' : [0]}
    df = pd.DataFrame(data=d)
    df.to_csv('evaluationmetric.csv', mode='a')

def showreportcsv():
    df = pd.read_csv('evaluationmetric.csv')
    df.drop("Unnamed: 0", axis=1, inplace = True)
    df.drop_duplicates(subset=["Model", "Description","Precision","Recall","F1-score","Mcc"], keep='last', inplace = True) 
    return df

def getscoring(model, X_test, name_file = "resultsfile.csv"):
    preds_val = model.predict(X_test)
    index_val = list(X_test.index.values.tolist())
    data = {'TransactionId': index_val,
            'FraudResult': preds_val}

    df = pd.DataFrame(data)

    df.to_csv(name_file, index=False)
    print("Done")
    
def testcode():
    print("codebien")