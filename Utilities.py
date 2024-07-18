import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


# ANSI escape codes for colors
class COLORS:
    """
    A class containing ANSI escape codes for various colors and text styles
    to make console output more easily interpretable and visually appealing.
    """
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright text colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    # Bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'

    # Text styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'

    # Reset
    RESET = '\033[0m'

    # Aliases for backwards compatibility
    HEADER = BRIGHT_MAGENTA
    ENDC = RESET


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


def normalization(dataframe):
    """
    A function used for normalizing numerical labels.
    :param dataframe: the dataframe that holds the dataset information

    """

    # Assuming df is your DataFrame
    numerical_columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()

    scaler.fit(dataframe[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    dataframe[numerical_columns] = scaler.transform(dataframe[numerical_columns])

    return dataframe


def one_hot_encoding(dataframe, class_label):
    """
    A function used for performing one-hot encoding.
    :param dataframe: the dataframe that holds the dataset information
    :param class_label: the column that specifies the class of each sample

    """
    X = dataframe.drop(class_label, axis=1)  # Drop the 'class' column to get features (X)
    y = dataframe[class_label]  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)

    return x_dummies, y


def preprocess_data(dataframe, class_label):
    """
    A function used for preprocessing each dataset. This includes normalizing numerical labels
    and performing one-hot encoding. The modified dataset is returned.
    :param dataframe: the dataframe that holds the dataset information
    :param class_label: the column that specifies the class of each sample
    :return: the final dataset is returned

    """
    dataframe = normalization(dataframe)

    x_dummies, y = one_hot_encoding(dataframe, class_label)

    return [x_dummies, y]


def preprocess_counterfactual_dataset(dataframe, class_label, prot_at_label):
    """
    A function used for preprocessing each dataset. This includes normalizing numerical labels
    and performing one-hot encoding on categorical variables. In addition the values of the protected
    attribute are swapped so as to create a counterfactual dataset. The modified dataset is returned.
    :param dataframe: the dataframe that holds the dataset information
    :param class_label: the column that specifies the class of each sample
    :return: the final dataset is returned

    """
    dataframe = normalization(dataframe)

    # Assuming df is your DataFrame and 'column_name' is the name of the column
    unique_values = dataframe[prot_at_label].unique()

    # Replace values in the 'sex' column
    dataframe[prot_at_label] = dataframe[prot_at_label].replace(
        {unique_values[0]: unique_values[1], unique_values[1]: unique_values[0]})

    x_dummies, y = one_hot_encoding(dataframe, class_label)

    return [x_dummies, y]


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
    A function used for calculating the accuracy, Disparate Impact Ratio,
    Equal Opportunity Difference, and Statistical Parity Difference of a classifier.

    :param y_test: it holds the actual class labels of the test split
    :param y_pred: it holds predicted class labels of the test split
    :param x_test: it holds attributes of the test split
    :param priv: it holds the label of the privileged group
    :param fav_out: it holds the label of the favourable outcome
    :return: the function returns the values of the Disparate Impact Ratio, accuracy,
             Equal Opportunity Difference, and Statistical Parity Difference of the classifier
    """
    priv_mask = x_test[priv] == True
    unpriv_mask = x_test[priv] == False

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{COLORS.BRIGHT_GREEN}Accuracy : {accuracy}{COLORS.ENDC}")

    # Calculate Disparate Impact Ratio
    a = ((y_pred == fav_out) & (unpriv_mask)).sum() / (unpriv_mask).sum()
    b = ((y_pred == fav_out) & (priv_mask)).sum() / (priv_mask).sum()
    disparate_impact = a / b
    print(f"{COLORS.HEADER}Disparate Impact Ratio: {disparate_impact}{COLORS.ENDC}")

    # Calculate Equal Opportunity Difference
    cm_priv = confusion_matrix(y_test[priv_mask], y_pred[priv_mask])
    cm_unpriv = confusion_matrix(y_test[unpriv_mask], y_pred[unpriv_mask])
    tpr_priv = cm_priv[1, 1] / (cm_priv[1, 1] + cm_priv[1, 0])
    tpr_unpriv = cm_unpriv[1, 1] / (cm_unpriv[1, 1] + cm_unpriv[1, 0])
    equal_opp_diff = tpr_priv - tpr_unpriv
    print(f"{COLORS.BRIGHT_BLUE}Equal Opportunity Difference: {equal_opp_diff}{COLORS.ENDC}")

    # Calculate Statistical Parity Difference
    prob_pos_priv = (y_pred[priv_mask] == fav_out).mean()
    prob_pos_unpriv = (y_pred[unpriv_mask] == fav_out).mean()
    stat_parity_diff = prob_pos_priv - prob_pos_unpriv
    print(f"{COLORS.BRIGHT_YELLOW}Statistical Parity Difference: {stat_parity_diff}{COLORS.ENDC}")

    return accuracy, disparate_impact, equal_opp_diff, stat_parity_diff


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


def pre_crossval(data, data_c, model, priv, unpriv, fav, unfav, folds):
    x = data[0]
    y = data[1]
    x_c = data_c[0]
    y_c = data_c[1]

    k_fold = KFold(n_splits=folds, shuffle=True, random_state=42)

    classifier = choose_classifier(model)

    c_classifier = choose_classifier(model)

    ROC_accuracy = np.zeros(int(sys.argv[2])).tolist()
    ROC_DIR = np.zeros(int(sys.argv[2])).tolist()
    ROC_samples = np.zeros(int(sys.argv[2])).tolist()
    ROC_EQ_OP_D = np.zeros(int(sys.argv[2])).tolist()
    ROC_ST_P = np.zeros(int(sys.argv[2])).tolist()

    CROC_accuracy = np.zeros(int(sys.argv[2])).tolist()
    CROC_DIR = np.zeros(int(sys.argv[2])).tolist()
    CROC_samples = np.zeros(int(sys.argv[2])).tolist()
    CROC_EQ_OP_D = np.zeros(int(sys.argv[2])).tolist()
    CROC_ST_P = np.zeros(int(sys.argv[2])).tolist()

    l_values = [i / 100 for i in range(0, int(sys.argv[2]))]

    for (train_indices, test_indices), (train_indices_c, test_indices_c) in zip(k_fold.split(x), k_fold.split(x_c)):

        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        x_train_c, x_test_c = x_c.iloc[train_indices], x_c.iloc[test_indices]
        y_train_c, y_test_c = y_c.iloc[train_indices], y_c.iloc[test_indices]
        # accuracy, disparate_impact, precision, recall

        for i in range(0, int(sys.argv[2])):
            classifier.fit(x_train, y_train)

            acc, DIR, samp, eq_op_d, st_p = critical_region_test(x_test, y_test, classifier, unpriv, priv,
                                                           unfav, fav, 0, i / 100, None)

            ROC_accuracy[i] = ROC_accuracy[i] + acc
            ROC_DIR[i] = ROC_DIR[i] + DIR
            ROC_samples[i] = ROC_samples[i] + samp
            ROC_EQ_OP_D[i] = ROC_EQ_OP_D[i] + eq_op_d
            ROC_ST_P[i] = ROC_ST_P[i] + st_p

            c_classifier.fit(x_train_c, y_train_c)

            acc, DIR, samp, eq_op_d, st_p = critical_region_test(x_test, y_test, c_classifier, unpriv, priv, unfav,
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

    summary_plot(l_values, ROC_accuracy, CROC_accuracy, 'ROC', 'ROC+MOD', 'critical region', 'Accuracy', 'accuracy vs critical region')

    summary_plot(l_values, ROC_DIR, CROC_DIR, 'ROC', 'ROC+MOD', 'critical region', 'DIR', 'DIR vs critical region')

    summary_plot(l_values, ROC_samples, CROC_samples, 'ROC', 'ROC+MOD', 'critical region', 'samples', 'samples vs critical region')

    summary_plot(l_values, ROC_EQ_OP_D, CROC_EQ_OP_D, 'ROC', 'ROC+MOD', 'critical region', 'samples', 'EQ of opportunity vs critical region')

    summary_plot(l_values, ROC_ST_P, CROC_ST_P, 'ROC', 'ROC+MOD', 'critical region', 'samples', 'Statistical Parity vs critical region')

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

    a, b, c, d = calculate_metrics(y, pred4, x, priv, fav)

    return a, b, len(indexes), c, d


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
