# Module that contains functions that can be used across all files
import pandas as pd
import numpy as np

def sklearn_data_to_df(data):
    '''translate sklearn df to a pandas dataframe'''
    pandas_df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                        columns=data['feature_names'] + ['target'])
    return pandas_df
