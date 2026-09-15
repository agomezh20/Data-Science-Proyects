import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


#   Function defined to see if the correlation with the other 
#   numeric-only columns is significant
def high_prod_cost_corr(col: pd.DataFrame):
    df = pd.DataFrame({'prod_cost':raw_data['production_cost_per_unit'],
                       'column': col})
    return bool(abs(df.corr(numeric_only=True).iloc[0, 1]) > 0.05)


#   Define the main Data source and sub-DataFrames
present_file = Path(__file__).parent
file_route = present_file.parent / 'data' / 'Intelligent_Production_IIoT.csv'
raw_data = pd.read_csv(file_route)

raw_data.drop(labels='record_id', axis=1, inplace=True)

data_numeric_only = raw_data.iloc[:, 5:-1]
data_str_only = raw_data.iloc[:, np.r_[0:5, -1]]


#   Define the list that test the high level correlated columns
bool_list = []

for name, column in data_numeric_only.items():
    correlated = high_prod_cost_corr(column)
    bool_list.append(correlated)


#   Create the DataFrame acording to the previous list
data_prod_cost_corr = data_numeric_only.iloc[:, bool_list]

final_df = data_prod_cost_corr
final_df['shift'] = data_str_only['shift']
final_df['product_type'] = data_str_only['product_type']


#   Seperate the data in the features and the target
y = final_df['production_cost_per_unit']
X = final_df.drop('production_cost_per_unit', axis=1)


#   Express the string-type categorization with dummies
X = pd.get_dummies(X, columns=['shift', 'product_type'], drop_first=False)


#   Fil the Null values
X = X.fillna(X.median())