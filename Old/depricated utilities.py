import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utilities import summary_plot, critical_region_test


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

def pre_plot_calculation(X_test, y_test, classifier, c_classifier, priv, unpriv, fav, unfav):
    """
    A function used for calculating the Disparate Impact Ratio,accuracy,recall and precision for multiple executions
    of ROC on 2 different versions of the same type of classifier, 1 is the counterfactually trained.
    Each time a different value of l is used.
    :param X_test: the dataframe holding the attributes of the test split
    :param y_test: the dataframe holding the classes of the test split
    :param classifier: a variable holding a classifier instance
    :param c_classifier: a variable holding a counterfactual instance of classifier
    :param priv: the privileged group
    :param unpriv: the unprivileged group
    :param fav: the favourable outcome
    :param unfav: the unfavourable outcome

    """
    l_values = []
    ROC_accuracy = []
    ROC_DIR = []
    ROC_samples = []
    CROC_accuracy = []
    CROC_DIR = []
    CROC_samples = []

    done_1 = 0
    done_2 = 0

    for i in range(1, int(sys.argv[3])):
        # accuracy, disparate_impact, precision, recall
        if done_1 == 0:

            a_1, b_1, c_1 = critical_region_test(X_test, y_test, classifier, unpriv, priv, unfav, fav, 0, i / 100, None)
            ROC_accuracy.append(a_1)
            ROC_DIR.append(b_1)
            ROC_samples.append(c_1)

            if b_1 > 1.0:
                done_1 = 1

        if done_2 == 0:
            a_2, b_2, c_2 = critical_region_test(X_test, y_test, c_classifier, unpriv, priv, unfav, fav, 0, i / 100, None)
            CROC_accuracy.append(a_2)
            CROC_DIR.append(b_2)
            CROC_samples.append(c_2)

            if b_2 > 1.0:
                done_2 = 1

        l_values.append(i / 100)

        if done_1 == 1 and done_2 == 1:
            break

    summary_plot(l_values, ROC_accuracy, CROC_accuracy, 'ROC', 'ROC+MOD', 'critical region', 'Accuracy',
                 'accuracy vs critical region')

    summary_plot(l_values, ROC_DIR, CROC_DIR, 'ROC', 'ROC+MOD', 'critical region', 'DIR', 'DIR vs critical region')

    summary_plot(l_values, ROC_samples, CROC_samples, 'ROC', 'ROC+MOD', 'critical region', 'samples',
                 'samples vs critical region')

def post_plot_calculation(X_test, y_test, classifier, priv, unpriv, fav, unfav):
    """
    A function used for calculating the Disparate Impact Ratio,accuracy,recall and precision for multiple executions
    of ROC and the attribute_swap + ROC algorithm on the same classifier. Each time a different value of l is used.
    :param X_test: the dataframe holding the attributes of the test split
    :param y_test: the dataframe holding the classes of the test split
    :param classifier: a variable holding a classifier instance
    :param priv: the privileged group
    :param unpriv: the unprivileged group
    :param fav: the favourable outcome
    :param unfav: the unfavourable outcome

    """
    l_values = []
    ROC_accuracy = []
    ROC_DIR = []
    ROC_samples = []
    SROC_accuracy = []
    SROC_DIR = []
    SROC_samples = []

    done_1 = 0
    done_2 = 0

    for i in range(1, int(sys.argv[3])):
        # accuracy, disparate_impact, precision, recall
        if done_1 == 0:
            a_1, b_1, c_1 = critical_region_test(X_test, y_test, classifier, unpriv, priv, unfav, fav, 0, i / 100, None)

            ROC_accuracy.append(a_1)
            ROC_DIR.append(b_1)
            ROC_samples.append(c_1)

            if b_1 > 1.0:
                done_1 = 1

        if done_2 == 0:
            a_2, b_2, c_2 = attribute_swap_and_critical(X_test, y_test, classifier, unpriv, priv, unfav, fav, 0,
                                                        i / 100, None)

            SROC_accuracy.append(a_2)
            SROC_DIR.append(b_2)
            SROC_samples.append(c_2)

            if b_2 > 1.0:
                done_2 = 1

        l_values.append(i / 100)

        if done_1 == 1 and done_2 == 1:
            break

    summary_plot(l_values, ROC_accuracy, SROC_accuracy, 'ROC', 'ROC+MOD', 'critical region', 'Accuracy',
                 'accuracy vs critical region')

    summary_plot(l_values, ROC_DIR, SROC_DIR, 'ROC', 'ROC+MOD', 'critical region', 'DIR', 'DIR vs critical region')

    summary_plot(l_values, ROC_samples, SROC_samples, 'ROC', 'ROC+MOD', 'critical region', 'samples',
                 'samples vs critical region')


def calculate_metrics(y_test, y_pred, x_test, priv, fav_out):
    """
    A function used for calculating the accuracy, precision, recall and Disparate Impact
    Ratio of a classifier.
    :param y_test: it holds the actual class labels of the test split
    :param y_test: it holds predicted class labels of the test split
    :param x_test: it holds attributes of the test split
    :param priv: it holds the label of the privileged group
    :param fav_out: it holds the label of the favourable outcome
    :return: the function returns the values of the Disparate Impact Ratio and accuracy of the classifier
    """

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{COLORS.GREEN}Accuracy of the classifier: {accuracy}{COLORS.ENDC}")

    # Calculate precision
    precision = precision_score(y_test, y_pred, pos_label=fav_out)
    print(f"{COLORS.BLUE}Precision of the classifier: {precision}{COLORS.ENDC}")

    # Calculate recall
    recall = recall_score(y_test, y_pred, pos_label=fav_out)
    print(f"{COLORS.YELLOW}Recall of the classifier: {recall}{COLORS.ENDC}")

    # a = ((y_pred == fav_out) & (x_test[priv] == False)).sum()
    # b = ((y_pred == fav_out) & (x_test[priv] == True)).sum()
    a = ((y_pred == fav_out) & (x_test[priv] == False)).sum() / (x_test[priv] == False).sum()
    b = ((y_pred == fav_out) & (x_test[priv] == True)).sum() / (x_test[priv] == True).sum()
    disparate_impact = a / b
    print(f"{COLORS.HEADER}Disparate Impact Ratio: {disparate_impact}{COLORS.ENDC}")
    # print(a, b)

    return accuracy, disparate_impact