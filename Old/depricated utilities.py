import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def bank_pie(features, classes, fav_pred, unfav_pred, my_title):
    """
    A function used for plotting a pie plot for the bank dataset.
    :param features: the dataframe holding the attributes of the dataset
    :param classes: the dataframe holding the classes of the dataset
    :param fav_pred: the favourable outcome
    :param unfav_pred: the unfavourable outcome
    :param my_title: a string displayed on the produced plot

    """
    plt.title(my_title)
    yes_young = sum(classes[features['age_young'] == True] == fav_pred)
    yes_elder = sum(classes[features['age_elder'] == True] == fav_pred)
    yes_adult = sum(classes[features['age_adult'] == True] == fav_pred)
    no_young = sum(classes[features['age_young'] == True] == unfav_pred)
    no_adult = sum(classes[features['age_adult'] == True] == unfav_pred)
    no_elder = sum(classes[features['age_elder'] == True] == unfav_pred)

    my_labels = ["yes_young", "yes_adult", "yes_elder", "no_young", "no_adult", "no_elder"]
    plt.pie(np.array([yes_young, yes_adult, yes_elder, no_young, no_adult, no_elder]), labels=my_labels,
            autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.show()


def double_data(dataframe, test_size, protected_attributes, class_label):
    """
    Currently not in use.

    """
    # Assuming df is your DataFrame
    numerical_columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()

    scaler.fit(dataframe[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    dataframe[numerical_columns] = scaler.transform(dataframe[numerical_columns])

    # Create a copy of the DataFrame
    dataframe_copy = dataframe.copy()

    # Replace values in the 'sex' column
    dataframe_copy['sex'] = dataframe_copy['sex'].replace({'Male': 'Female', 'Female': 'Male'})

    # Concatenate DataFrames
    dataframe = pd.concat([dataframe, dataframe_copy], ignore_index=True)

    X = dataframe.drop(class_label, axis=1)  # Drop the 'class' column to get features (X)
    y = dataframe[class_label]  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)

    return train_test_split(x_dummies, y, test_size=test_size, random_state=1)

def handle_age(data, label, low, high):
    """
    A simple function used for assigning values to the person_age label of the bank_dataset
    :param data: the dataframe containing the attributes of each instance
    :param label: the label which corresponds to the person_age column
    :param low: the lower bound that separates the 3 age groups
    :param low: the highest bound that separates the 3 age groups

    """
    # Define conditions and corresponding values
    conditions = [
        data[label] <= low,
        (data[label] > low) & (data[label] <= high),
        data[label] > high
    ]
    values = ['young', 'adult', 'elder']

    # Update the 'person_age' column based on conditions
    data[label] = np.select(conditions, values, default=data[label])
