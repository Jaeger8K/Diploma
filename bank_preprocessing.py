import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np


def handle_NAN(df1, df2):
    # Combine features and target into one dataframe
    df = pd.concat([df1, df2], axis=1)

    # Print samples with NaN values in any feature
    # df_NAN= df[df.isnull().any(axis=1)]
    # print(df_NAN)

    # Drop samples with NaN values in any feature
    df = df.dropna()

    # Create two separate datasets
    clean_df2 = df[['income']]
    clean_df1 = df.drop('income', axis=1)

    return clean_df1, clean_df2


def preprocess_data(df1, df2):
    df1, df2 = handle_NAN(df1, df2)

    df1, df2 = handle_non_numerical_data(df1, df2)

    return normalise(df1), df2


def handle_non_numerical_data(df1, df2):
    columns = df1.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df1[column].dtype != np.int64 and df1[column].dtype != np.float64:
            column_contents = df1[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df1.loc[:, column] = df1[column].apply(convert_to_int)

    df2.loc[:, 'income'] = df2['income'].replace({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1})

    return df1, df2


def normalise(df1):
    columns_to_normalize = df1.columns[X.columns != 'sex']

    # Normalize all columns except 'sex'
    normalized_X_except_sex = (df1[columns_to_normalize] - df1[columns_to_normalize].mean()) / df1[
        columns_to_normalize].std()

    # Combine normalized columns with the 'sex' column
    normalized_X = pd.concat([normalized_X_except_sex, df1['sex']], axis=1)

    return normalized_X


def export(df1, df2):
    # Export the DataFrame to a CSV file
    df1.to_csv('normalized_bank_features.csv', index=False)
    df2.to_csv('y.csv', index=False)


def print_variable_info(X, y):
    # Print the column names
    print("X column Names:", X.columns)
    print("y column Names:", y.columns, "\n")

    # Assuming X is your DataFrame
    categories_per_column = {}

    for column in X.columns:
        categories_per_column[column] = X[column].unique()

    # Print the unique categories for each column
    for column, categories in categories_per_column.items():
        print(f"Categories in {column}: {categories}\n")

    # print(X['sex'])
    # print(y)


# fetch dataset
df = pd.read_csv('bank-full.csv', delimiter=';')

# print_variable_info(X, y)

# X, y = preprocess_data(X, y)

# export(X, y)

print(df.columns)
print(df)