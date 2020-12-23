import pandas as pd
import numpy as np
from utilities import sklearn_data_to_df, train_test_split_from_df
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Get data
data = load_iris()
df = sklearn_data_to_df(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split_from_df(df, 'target', test_size=0.2, random_state=0)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# Build a prediction model
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# Get results
print(knn.get_params(), '\n')
print(knn.predict(X_test), '\n')
print(knn.predict_proba(X_test), '\n')
print(knn.score(X_test, y_test), '\n')
print(knn.kneighbors_graph(X_test), '\n')


