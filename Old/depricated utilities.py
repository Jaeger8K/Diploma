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


def pie_plot(features, classes, fav_pred, unfav_pred, unpriv, labels, my_title):
    plt.title(my_title)
    a = sum(classes[features[unpriv] == True] == fav_pred)
    b = sum(classes[features[unpriv] == True] == unfav_pred)
    c = sum(classes[features[unpriv] == False] == fav_pred)
    d = sum(classes[features[unpriv] == False] == unfav_pred)

    plt.pie(np.array([a, c, b, d]), labels=labels, autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.gcf().set_size_inches(7, 4)

    plt.show()
def pre_crossval(data, data_c, model , priv, unpriv, fav, unfav, folds, crit_region, seed, pie):
    pie_functions = [adult_pie, crime_pie, bank_pie]

    x = data[0]
    y = data[1]
    x_c = data_c[0]
    y_c = data_c[1]

    k_fold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    classifier = choose_classifier(model)

    c_classifier = choose_classifier(model)

    ROC_accuracy = np.zeros(crit_region).tolist()
    ROC_DIR = np.zeros(crit_region).tolist()
    ROC_samples = np.zeros(crit_region).tolist()
    ROC_EQ_OP_D = np.zeros(crit_region).tolist()
    ROC_ST_P = np.zeros(crit_region).tolist()

    CROC_accuracy = np.zeros(crit_region).tolist()
    CROC_DIR = np.zeros(crit_region).tolist()
    CROC_samples = np.zeros(crit_region).tolist()
    CROC_EQ_OP_D = np.zeros(crit_region).tolist()
    CROC_ST_P = np.zeros(crit_region).tolist()

    l_values = [i / 100 for i in range(0, crit_region)]

    for index, ((train_indices, test_indices), (train_indices_c, test_indices_c)) \
            in enumerate(zip(k_fold.split(x), k_fold.split(x_c)), start=1):

        print(f"{COLORS.MAGENTA}\nFold:{index}{COLORS.ENDC}")

        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        pie_functions[pie](x_test, y_test, fav, unfav, f"Fold:{index} distribution")

        x_train_c, x_test_c = x_c.iloc[train_indices], x_c.iloc[test_indices]
        y_train_c, y_test_c = y_c.iloc[train_indices], y_c.iloc[test_indices]

        if pie == 1: #special case for crime dataset
            temp = fav
            fav = unfav
            unfav = temp
            priv = unpriv

        for i in range(0, crit_region):
            classifier.fit(x_train, y_train)
            c_classifier.fit(x_train_c, y_train_c)

            if i + 1 == crit_region:

                acc, DIR, samp, eq_op_d, st_p = critical_region_test(x_test, y_test, classifier, priv,
                                                                     unfav, fav, 0, i / 100, pie_functions[pie])
            else:
                acc, DIR, samp, eq_op_d, st_p = critical_region_test(x_test, y_test, classifier, priv,
                                                                     unfav, fav, 0, i / 100, None)

            ROC_accuracy[i] = ROC_accuracy[i] + acc
            ROC_DIR[i] = ROC_DIR[i] + DIR
            ROC_samples[i] = ROC_samples[i] + samp
            ROC_EQ_OP_D[i] = ROC_EQ_OP_D[i] + eq_op_d
            ROC_ST_P[i] = ROC_ST_P[i] + st_p

            print()

            if i + 1 == crit_region:
                acc, DIR, samp, eq_op_d, st_p = critical_region_test(x_test, y_test, c_classifier, priv, unfav,
                                                                     fav, 0, i / 100, pie_functions[pie])
            else:
                acc, DIR, samp, eq_op_d, st_p = critical_region_test(x_test, y_test, c_classifier, priv, unfav,
                                                                     fav, 0, i / 100, None)

            CROC_accuracy[i] = CROC_accuracy[i] + acc
            CROC_DIR[i] = CROC_DIR[i] + DIR
            CROC_samples[i] = CROC_samples[i] + samp
            CROC_EQ_OP_D[i] = CROC_EQ_OP_D[i] + eq_op_d
            CROC_ST_P[i] = CROC_ST_P[i] + st_p

        ROC_accuracy = [x / folds for x in ROC_accuracy]
        ROC_DIR = [x / folds for x in ROC_DIR]
        ROC_samples = [x / folds for x in ROC_samples]
        ROC_EQ_OP_D = [x / folds for x in ROC_EQ_OP_D]
        ROC_ST_P = [x / folds for x in ROC_ST_P]

        CROC_accuracy = [x / folds for x in CROC_accuracy]
        CROC_DIR = [x / folds for x in CROC_DIR]
        CROC_samples = [x / folds for x in CROC_samples]
        CROC_EQ_OP_D = [x / folds for x in CROC_EQ_OP_D]
        CROC_ST_P = [x / folds for x in CROC_ST_P]

    if crit_region != 1:
        summary_plot(l_values, ROC_accuracy, CROC_accuracy, 'ROC', 'ROC+MOD', 'critical region',
                     'Accuracy', 'accuracy vs critical region')

        summary_plot(l_values, ROC_DIR, CROC_DIR, 'ROC', 'ROC+MOD', 'critical region', 'DIR',
                     'DIR vs critical region')

        summary_plot(l_values, ROC_samples, CROC_samples, 'ROC', 'ROC+MOD', 'critical region',
                     'samples', 'samples vs critical region')

        summary_plot(l_values, ROC_EQ_OP_D, CROC_EQ_OP_D, 'ROC', 'ROC+MOD', 'critical region',
                     'samples', 'EQ of opportunity vs critical region')

        summary_plot(l_values, ROC_ST_P, CROC_ST_P, 'ROC', 'ROC+MOD', 'critical region',
                     'samples', 'Statistical Parity vs critical region')