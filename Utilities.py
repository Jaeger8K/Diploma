import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


# ANSI escape codes for colors
class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def handle_age(data, label, low, high):
    # Define conditions and corresponding values
    conditions = [
        data[label] <= low,
        (data[label] > low) & (data[label] <= high),
        data[label] > high
    ]
    values = ['young', 'adult', 'elder']

    # Update the 'person_age' column based on conditions
    data[label] = np.select(conditions, values, default=data[label])


def print_df_columns(columns):
    for column in columns:
        print(column)


def export_csv(dataframe, name):
    dataframe.to_csv(name, index=False)


def print_NAN(data):
    # Assuming 'data' is your DataFrame
    rows_with_nan = data[data.isna().any(axis=1)]
    print(rows_with_nan)


def preprocess_data(dataframe, test_size, class_label):
    # Assuming df is your DataFrame
    numerical_columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()

    scaler.fit(dataframe[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    dataframe[numerical_columns] = scaler.transform(dataframe[numerical_columns])

    X = dataframe.drop(class_label, axis=1)  # Drop the 'class' column to get features (X)
    y = dataframe[class_label]  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)
    # random_state = np.random.randint(1000)  # Generate a random integer
    return train_test_split(x_dummies, y, test_size=test_size, random_state=1)


def cross_validation_load(dataframe, class_label):
    # Assuming df is your DataFrame
    numerical_columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()

    scaler.fit(dataframe[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    dataframe[numerical_columns] = scaler.transform(dataframe[numerical_columns])

    X = dataframe.drop(class_label, axis=1)  # Drop the 'class' column to get features (X)
    y = dataframe[class_label]  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)

    return x_dummies, y


def choose_classifier(model_selection):
    m = 0

    if model_selection == "1":  # works
        m = LogisticRegression(random_state=16, max_iter=1000)

    elif model_selection == "2":  # works
        m = RandomForestClassifier(max_depth=5, random_state=0)

    elif model_selection == "3":  # works, needs more iterations, takes time
        m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1300)

    # elif model_selection == "4":  # works
    #    m = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)

    # elif model_selection == "4":  # works
    #    m = GaussianNB()

    # elif model_selection == "4":  # works 'LinearSVC' object has no attribute 'predict_proba'. Did you mean: '_predict_proba_lr'?
    #    m = LinearSVC(dual='auto')

    # elif model_selection == "6":  # needs work, 'KMeans' object has no attribute 'predict_proba'
    #    m = KMeans(n_clusters=2, random_state=0, n_init="auto")

    # elif model_selection == "8":  # needs work, doesnt terminate
    #    m = svm.SVC()

    return m


def double_data(dataframe, test_size, protected_attributes, class_label):
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


def calculate_metrics(y_test, y_pred, x_test, priv, fav_out):
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{colors.GREEN}Accuracy of the classifier: {accuracy}{colors.ENDC}")

    # Calculate precision
    precision = precision_score(y_test, y_pred, pos_label=fav_out)
    print(f"{colors.BLUE}Precision of the classifier: {precision}{colors.ENDC}")

    # Calculate recall
    recall = recall_score(y_test, y_pred, pos_label=fav_out)
    print(f"{colors.YELLOW}Recall of the classifier: {recall}{colors.ENDC}")

    a = ((y_pred == fav_out) & (x_test[priv] == False)).sum()
    b = ((y_pred == fav_out) & (x_test[priv] == True)).sum()
    disparate_impact = a / b
    print(f"{colors.HEADER}Disparate Impact Ratio: {disparate_impact}{colors.ENDC}")
    print(a, b)

    return accuracy, disparate_impact


def summary_plot(a, b, c, d, e, f, g, h):
    # Plotting
    plt.plot(a, b, label=d)
    plt.plot(a, c, label=e)

    # Adding labels and title
    plt.xlabel(f)
    plt.ylabel(g)
    plt.title(h)
    plt.legend()

    # Show plot
    plt.show()


def adult_pie(features, classes, fav_pred, unfav_pred, my_title):
    plt.title(my_title)
    rich_women = sum(classes[features['sex_Female'] == True] == fav_pred)
    poor_women = sum(classes[features['sex_Female'] == True] == unfav_pred)
    rich_men = sum(classes[features['sex_Female'] == False] == fav_pred)
    poor_men = sum(classes[features['sex_Female'] == False] == unfav_pred)

    my_labels = ["Rich women", "Rich men", "Poor women", "Poor men"]
    plt.pie(np.array([rich_women, rich_men, poor_women, poor_men]), labels=my_labels,
            autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.show()


def bank_pie(features, classes, fav_pred, unfav_pred, my_title):
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


def crime_pie(features, classes, fav_pred, unfav_pred, my_title):
    plt.title(my_title)
    rich_women = sum(classes[features['racepctblack_unprivileged'] == True] == fav_pred)
    poor_women = sum(classes[features['racepctblack_unprivileged'] == True] == unfav_pred)
    rich_men = sum(classes[features['racepctblack_unprivileged'] == False] == fav_pred)
    poor_men = sum(classes[features['racepctblack_unprivileged'] == False] == unfav_pred)

    my_labels = ["free blacks", "free whites", "jailed blacks", "jailed whites"]
    plt.pie(np.array([rich_women, rich_men, poor_women, poor_men]), labels=my_labels,
            autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.show()


def plot_calculation(X_test, y_test, classifier, priv, unpriv, fav, unfav):
    l_values = []
    ROC_accuracy = []
    ROC_DIR = []
    ROC_samples = []
    SROC_accuracy = []
    SROC_DIR = []
    SROC_samples = []

    for i in range(1, int(sys.argv[5])):
        # accuracy, disparate_impact, precision, recall
        a_1, b_1, c_1 = critical_region_test(X_test, y_test, classifier, unpriv, priv, unfav, fav, 0, i / 100, None)
        a_2, b_2, c_2 = attribute_swap_and_critical(X_test, y_test, classifier, unpriv, priv, unfav, fav, 0, i / 100, None)

        ROC_accuracy.append(a_1)
        SROC_accuracy.append(a_2)
        ROC_DIR.append(b_1)
        SROC_DIR.append(b_2)
        ROC_samples.append(c_1)
        SROC_samples.append(c_2)

        l_values.append(i / 100)

    summary_plot(l_values, ROC_accuracy, SROC_accuracy, 'ROC_accuracy', 'SROC_accuracy', 'l_values', 'Accuracy',
                 'Accuracy vs probability difference')

    summary_plot(l_values, ROC_DIR, SROC_DIR, 'ROC_DIR', 'SROC_DIR', 'l_values', 'DIR', 'DIR vs probability difference')

    summary_plot(l_values, ROC_samples, SROC_samples, 'ROC_samples', 'SROC_samples', 'l_values', 'samples', 'samples vs probability difference')


def partitioning(lower_bound, upper_bound, classifier_prob):
    indexes = [idx for idx, probabilities in enumerate(classifier_prob) if
               lower_bound < np.abs(probabilities[0] - probabilities[1]) < upper_bound]
    # print(len(indexes))

    return indexes


def attribute_swap_test(x, y, classifier, unpriv, priv, unfav, fav, print_function):
    x_copy = x.copy()
    x_copy[unpriv] = ~x_copy[unpriv]
    x_copy[priv] = ~x_copy[priv]

    pred2 = classifier.predict(x_copy)

    print("\nattribute_swap_test")

    # calculate_mertics(y_test, pred2, X_test, priv, fav)
    calculate_metrics(y, pred2, x_copy, priv, fav)

    # print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

    # adult_pie(X_test, pred2, '>50K', '<=50K', 'pred2 data')
    if print_function:
        print_function(x, pred2, fav, unfav, 'attribute_swap_test')


def critical_region_test(x, y, classifier, unpriv, priv, unfav, fav, lower_bound, upper_bound, print_function):
    pred4 = classifier.predict(x)

    indexes = partitioning(lower_bound, upper_bound, classifier.predict_proba(x))
    feature_part = x.iloc[indexes]

    for iteration_number, (index, row) in enumerate(feature_part.iterrows(), start=0):

        if row[priv]:
            pred4[indexes[iteration_number]] = unfav

        else:
            pred4[indexes[iteration_number]] = fav

    print(f"\ncritical_region_test l: {upper_bound}")
    print(f"{colors.RED}Elements in critical region: {len(indexes)}{colors.ENDC}")

    if print_function:
        print_function(x, pred4, fav, unfav, f'critical_region_test l: {upper_bound}')

    a, b = calculate_metrics(y, pred4, x, priv, fav)

    return a, b, len(indexes)


def attribute_swap_and_critical(x, y, classifier, unpriv, priv, unfav, fav, lower_bound, upper_bound, print_function):
    x_copy = x.copy()
    x_copy[unpriv] = ~x_copy[unpriv]
    x_copy[priv] = ~x_copy[priv]

    pred3 = classifier.predict(x_copy)

    indexes = partitioning(lower_bound, upper_bound, classifier.predict_proba(x_copy))
    feature_part = x_copy.iloc[indexes]

    for iteration_number, (index, row) in enumerate(feature_part.iterrows(), start=0):

        if row[unpriv]:
            pred3[indexes[iteration_number]] = unfav

        else:
            pred3[indexes[iteration_number]] = fav

    print(f"\nattribute_swap_and_critical l: {upper_bound}")
    print(f"{colors.RED}Elements in critical region: {len(indexes)}{colors.ENDC}")

    if print_function:
        print_function(x, pred3, fav, unfav, f'attribute_swap_and_critical l: {upper_bound}')

    a, b = calculate_metrics(y, pred3, x_copy, unpriv, fav)

    return a, b, len(indexes)
