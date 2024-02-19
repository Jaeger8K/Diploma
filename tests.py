# fetch dataset
import pandas as pd
from ucimlrepo import fetch_ucirepo

from Utilities import export_csv

# Read the CSV files into DataFrames
a = pd.read_csv('test.csv', delimiter=';')
b = pd.read_csv('train.csv', delimiter=';')

# Merge the DataFrames
merged_df = pd.concat([a, b])

print(merged_df.columns)

# Export the merged DataFrame to a CSV file
merged_df.to_csv('bank_dataset.csv', index=False)