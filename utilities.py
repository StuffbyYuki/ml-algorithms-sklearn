# Module that contains functions that can be used across all files
# df represents pandas DataFrame

import pandas as pd
import numpy as np
def sklearn_data_to_df(data):
    '''
        translate sklearn df to a pandas dataframe
        dependencies: numpy, pandas
        target attribute will be named "target"
    '''
    pandas_df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                        columns=np.append(data['feature_names'], ['target']))
    print(pandas_df.head())
    return pandas_df

from sklearn.model_selection import train_test_split
def train_test_split_from_df(df, target_col, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    '''
        split df into train and test
        utilizing sklearn "train_test_split"
        returns a tuple
    '''
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=shuffle,
                                                        stratify=stratify)
    return X_train, X_test, y_train, y_test
