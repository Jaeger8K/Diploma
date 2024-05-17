import inspect
import os
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
class COLORS:
    """
    A simple class containing the code of certain colors in order to make console output
    more easily interpretable
    """
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def print_df_columns(columns):
    """
    A simple function used for printing the columns of a dataframe

    """
    for column in columns:
        print(column)


def export_csv(dataframe, name):
    """
    A simple function used for exporting a dataframe into csv format

    """
    dataframe.to_csv(name, index=False)


def print_nan(data):
    """
    A simple function used for printing the samples that have NAN attribute values in any column

    """
    # Assuming 'data' is your DataFrame
    rows_with_nan = data[data.isna().any(axis=1)]
    print(rows_with_nan)


def preprocess_data(dataframe, test_size, class_label):
    """
    A function used for preprocessing each dataset. This includes normalizing numerical labels
    and performing one-hot encoding on categorical variables. A train test split is returned.
    :param dataframe: the dataframe that holds the dataset information
    :param test_size: a value specifying the size of the test split
    :param class_label: the column that specifies the class of each sample
    :return: a train/test split of the normalized dataset is returned

    """
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


def counterfactual_dataset(dataframe, test_size, class_label, prot_at_label):
    # Assuming df is your DataFrame
    numerical_columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()

    scaler.fit(dataframe[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    dataframe[numerical_columns] = scaler.transform(dataframe[numerical_columns])

    # Assuming df is your DataFrame and 'column_name' is the name of the column
    unique_values = dataframe[prot_at_label].unique()

    # Replace values in the 'sex' column
    dataframe[prot_at_label] = dataframe[prot_at_label].replace(
        {unique_values[0]: unique_values[1], unique_values[1]: unique_values[0]})
    # dataframe[prot_at_label] = dataframe[prot_at_label].replace({'Male': 'Female', 'Female': 'Male'})

    X = dataframe.drop(class_label, axis=1)  # Drop the 'class' column to get features (X)
    y = dataframe[class_label]  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)

    return train_test_split(x_dummies, y, test_size=test_size, random_state=1)


def cross_validation_load(dataframe, class_label):
    """
    A function used for preprocessing each dataset in order to perform 10 fold croos validation.
    :param dataframe: the dataframe that holds the dataset information
    :param class_label: the column that specifies the class of each sample
    :return: the function returns the modified attributes and class labels of the original dataset
    in separate data structures

    """

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
    """
    A function used for choosing among 3 classifiers.
    :param model_selection: the selection made by the user about which classifier to use
    :return: the function returns the instance of the chosen classifier

    """
    m = 0

    if model_selection == "1":  # works
        m = LogisticRegression(random_state=16, max_iter=1000)

    elif model_selection == "2":  # works
        m = RandomForestClassifier(max_depth=5, random_state=0)

    elif model_selection == "3":  # works, needs more iterations, takes time
        m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1300)

    elif model_selection == "4":  # works, needs more iterations, takes time
        m = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1300)

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


def summary_plot(a, b, c, d, e, f, g, h):
    """
    A function used for creating plots from the information gathered in the function plot_calculation.

    """
    # Plotting
    plt.plot(a[:len(b)], b, label=d)
    plt.plot(a[:len(c)], c, label=e)

    # Adding labels and title
    plt.xlabel(f, fontsize=14)
    plt.ylabel(g, fontsize=14)
    plt.title(h)
    plt.legend()

    # Show plot
    plt.show()


def adult_pie(features, classes, fav_pred, unfav_pred, my_title):
    """
    A function used for plotting a pie plot for the Adult dataset.
    :param features: the dataframe holding the attributes of the dataset
    :param classes: the dataframe holding the classes of the dataset
    :param fav_pred: the favourable outcome
    :param unfav_pred: the unfavourable outcome
    :param my_title: a string displayed on the produced plot

    """
    plt.title(my_title)
    rich_women = sum(classes[features['sex_Female'] == True] == fav_pred)
    poor_women = sum(classes[features['sex_Female'] == True] == unfav_pred)
    rich_men = sum(classes[features['sex_Female'] == False] == fav_pred)
    poor_men = sum(classes[features['sex_Female'] == False] == unfav_pred)

    my_labels = ["Rich women", "Rich men", "Poor women", "Poor men"]
    plt.pie(np.array([rich_women, rich_men, poor_women, poor_men]), labels=my_labels,
            autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.gcf().set_size_inches(7, 4)

    plt.show()


def crime_pie(features, classes, fav_pred, unfav_pred, my_title):
    """
    A function used for plotting a pie plot for the communities and crime dataset.
    :param features: the dataframe holding the attributes of the dataset
    :param classes: the dataframe holding the classes of the dataset
    :param fav_pred: the favourable outcome
    :param unfav_pred: the unfavourable outcome
    :param my_title: a string displayed on the produced plot

    """
    plt.title(my_title)
    a = sum(classes[features['racepctblack_unprivileged'] == True] == fav_pred)
    b = sum(classes[features['racepctblack_unprivileged'] == True] == unfav_pred)
    c = sum(classes[features['racepctblack_unprivileged'] == False] == fav_pred)
    d = sum(classes[features['racepctblack_unprivileged'] == False] == unfav_pred)

    my_labels = ["low crime black community", "low crime white community", "high crime black community",
                 "high crime white community"]
    plt.pie(np.array([a, c, b, d]), labels=my_labels, autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.gcf().set_size_inches(7, 4)

    plt.show()


def pie_plot(features, classes, fav_pred, unfav_pred, unpriv, labels, my_title):
    plt.title(my_title)
    a = sum(classes[features[unpriv] == True] == fav_pred)
    b = sum(classes[features[unpriv] == True] == unfav_pred)
    c = sum(classes[features[unpriv] == False] == fav_pred)
    d = sum(classes[features[unpriv] == False] == unfav_pred)

    plt.pie(np.array([a, c, b, d]), labels=labels, autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.gcf().set_size_inches(7, 4)

    plt.show()


def pre_plot_calculation(X_test, y_test, classifier, c_classifier, priv, unpriv, fav, unfav):
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
            a_2, b_2, c_2 = critical_region_test(X_test, y_test, c_classifier, unpriv, priv, unfav, fav, 0, i / 100,
                                                 None)
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
    of ROC and the attribute_swap + ROC algorithm. Each time a different value of l is used.
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


def partitioning(lower_bound, upper_bound, classifier_prob):
    """
    A function used for returning the indexes of the classifier decisions that have a difference lower than l
    :param lower_bound: the lowest value that the difference between the 2 predictions can have (typically 0)
    :param upper_bound: the highest value that the difference between the 2 predictions can have
    :param classifier_prob: a variable holding the probabilities of each sample belonging to either class
    :return: the function returns the indexes of the samples that the probability difference is between the
    lower_bound and upper_bound

    """

    indexes = [idx for idx, probabilities in enumerate(classifier_prob) if
               lower_bound < np.abs(probabilities[0] - probabilities[1]) < upper_bound]
    # print(len(indexes))

    return indexes


def attribute_swap_test(x, y, classifier, unpriv, priv, unfav, fav, print_function):
    """
    A function used for switching the protected attributes of the test data. Afterwards the classifier
    is tested with the altered data and Disparate Impact Ratio, Accuracy, Precision and Recall are calculated.
    :param x: a dataframe holding the attributes of the test dataset
    :param y: a dataframe holding the classes of the test dataset
    :param classifier: a instance of the chosen classifier
    :param unpriv: the unprivileged group
    :param priv: the privileged group
    :param unfav: the unfavourable outcome
    :param fav: the favourable outcome
    :param print_function: a function used for plotting different pie plots

    """
    x_copy = x.copy()
    x_copy[unpriv] = ~x_copy[unpriv]
    x_copy[priv] = ~x_copy[priv]

    pred2 = classifier.predict(x_copy)

    print("\nattribute_swap_test")

    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename

    # Extract just the name of the file
    caller_filename = os.path.basename(caller_filename)

    if caller_filename == 'postswap_adults.py' or caller_filename == 'postcross_adults.py':
        # calculate_mertics(y_test, pred2, X_test, priv, fav)
        calculate_metrics(y, pred2, x_copy, priv, fav)
    elif caller_filename == 'postswap_crime.py' or caller_filename == 'postcross_crime.py':
        # calculate_mertics(y_test, pred2, X_test, priv, fav)
        calculate_metrics(y, pred2, x_copy, unpriv, fav)

    if print_function == adult_pie:
        print_function(x, pred2, fav, unfav, 'attribute_swap_test')
    elif print_function == crime_pie:
        print_function(x, pred2, unfav, fav, 'attribute_swap_test')


def critical_region_test(x, y, classifier, unpriv, priv, unfav, fav, lower_bound, upper_bound, print_function):
    """
    A function used for applying the Reject Option-based Classification algorithm (ROC) to a classifier.
    :param x: a dataframe holding the attributes of the test dataset
    :param y: a dataframe holding the classes of the test dataset
    :param classifier: a instance of the chosen classifier
    :param unpriv: the unprivileged group
    :param priv: the privileged group
    :param unfav: the unfavourable outcome
    :param fav: the favourable outcome
    :param lower_bound: the lowest value that the difference between the 2 predictions can have (typically 0)
    :param upper_bound: the highest value that the difference between the 2 predictions can have
    :param print_function: a function used for plotting different pie plots

    """
    pred4 = classifier.predict(x)

    indexes = partitioning(lower_bound, upper_bound, classifier.predict_proba(x))
    feature_part = x.iloc[indexes]

    for iteration_number, (index, row) in enumerate(feature_part.iterrows(), start=0):

        if row[priv]:
            pred4[indexes[iteration_number]] = unfav

        else:
            pred4[indexes[iteration_number]] = fav

    print(f"\ncritical_region_test l: {upper_bound}")
    print(f"{COLORS.RED}Elements in critical region: {len(indexes)}{COLORS.ENDC}")

    if print_function == adult_pie:
        print_function(x, pred4, fav, unfav, f'critical_region_test l: {upper_bound}')
    elif print_function == crime_pie:
        print_function(x, pred4, unfav, fav, f'critical_region_test l: {upper_bound}')

    a, b = calculate_metrics(y, pred4, x, priv, fav)

    return a, b, len(indexes)


def attribute_swap_and_critical(x, y, classifier, unpriv, priv, unfav, fav, lower_bound, upper_bound, print_function):
    """
    A function used for applying both the Reject Option-based Classification algorithm to a classifier and
    switching the protected attributes.
    :param x: a dataframe holding the attributes of the test dataset
    :param y: a dataframe holding the classes of the test dataset
    :param classifier: a instance of the chosen classifier
    :param unpriv: the unprivileged group
    :param priv: the privileged group
    :param unfav: the unfavourable outcome
    :param fav: the favourable outcome
    :param lower_bound: the lowest value that the difference between the 2 predictions can have (typically 0)
    :param upper_bound: the highest value that the difference between the 2 predictions can have
    :param print_function: a function used for plotting different pie plots

    """
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
    print(f"{COLORS.RED}Elements in critical region: {len(indexes)}{COLORS.ENDC}")

    if print_function == adult_pie:
        print_function(x, pred3, fav, unfav, f'attribute_swap_and_critical l: {upper_bound}')
    elif print_function == crime_pie:
        print_function(x, pred3, unfav, fav, f'attribute_swap_and_critical l: {upper_bound}')

    a, b = calculate_metrics(y, pred3, x_copy, unpriv, fav)

    return a, b, len(indexes)
