import pandas as pd
import numpy as np
from utilities import sklearn_data_to_df, train_test_split_from_df
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LinearRegression

# Get data
data = load_boston()
df = sklearn_data_to_df(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split_from_df(df, 'target', test_size=0.2, random_state=0)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# Build a prediction model
linear_model = LinearRegression().fit(X_train, y_train)

# Get results
print(linear_model.get_params())
print(linear_model.coef_)
print(linear_model.intercept_)
print(linear_model.predict(X_test))
print(linear_model.score(X_test, y_test))
