import pandas as pd
from sklearn.tree import DecisionTreeRegressor

print("helloSarah")

def_feature = pd.read_csv("input/Xente_Variable_Definitions.csv")
data = pd.read_csv("input/training.csv")
X_test = pd.read_csv("input/test.csv")
sample_submission = pd.read_csv("input/sample_submission.csv")

data = data.dropna(axis=0)
y = data.FraudResult
X = data.copy()
X.drop(['FraudResult'], axis=1, inplace=True)
X.describe()

first_model_decision_tree = DecisionTreeRegressor(random_state=1)
first_model_decision_tree.fit(X, y)
print("Making predictions for the following 5 data:")
print(X.head())
print("The predictions are")
print(first_model_decision_tree.predict(X.head()))