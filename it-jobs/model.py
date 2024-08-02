import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("employee_data.csv")


print(df.head())


df["Gender"] = [1 if x == 'F' else 0 for x in df["Gender"]]


print(df.head())


tree = DecisionTreeRegressor()

factors = ["Gender", "Experience (Years)"]
predict = "Salary"


xtrain, xtest, ytrain, ytest = train_test_split(df[factors], df[predict], test_size = 0.2)

tree.fit(xtrain, ytrain)

predictions = tree.predict(xtest)

print(mean_absolute_error(ytest, predictions))

print(tree.feature_importances_)

threshold = 10000
scores = []
for i in range(len(ytest)):
    if abs(ytest.iloc[i] - predictions[i]) < threshold:
        scores.append(True)
    else:
        scores.append(False)

print(scores.count(True)/len(scores))