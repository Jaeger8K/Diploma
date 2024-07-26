import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


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
        # m = DecisionTreeClassifier(random_state=42)
        # m = KNeighborsClassifier(n_neighbors=5)

    elif model_selection == "3":  # works, needs more iterations, takes time
        m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2200)

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
    # print(f"{COLORS.BRIGHT_GREEN}Accuracy : {accuracy}{COLORS.ENDC}")
    print(f"{COLORS.BRIGHT_GREEN}{accuracy}{COLORS.ENDC}")

    # Calculate Disparate Impact Ratio
    a = ((y_pred == fav_out) & (unpriv_mask)).sum() / (unpriv_mask).sum()
    b = ((y_pred == fav_out) & (priv_mask)).sum() / (priv_mask).sum()
    disparate_impact = a / b
    # print(f"{COLORS.HEADER}Disparate Impact Ratio: {disparate_impact}{COLORS.ENDC}")
    print(f"{COLORS.HEADER}{disparate_impact}{COLORS.ENDC}")

    # Calculate Equal Opportunity Difference
    cm_priv = confusion_matrix(y_test[priv_mask], y_pred[priv_mask])
    cm_unpriv = confusion_matrix(y_test[unpriv_mask], y_pred[unpriv_mask])
    tpr_priv = cm_priv[1, 1] / (cm_priv[1, 1] + cm_priv[1, 0])
    tpr_unpriv = cm_unpriv[1, 1] / (cm_unpriv[1, 1] + cm_unpriv[1, 0])
    equal_opp_diff = tpr_priv - tpr_unpriv
    # print(f"{COLORS.BRIGHT_BLUE}Equal Opportunity Difference: {equal_opp_diff}{COLORS.ENDC}")
    print(f"{COLORS.BRIGHT_BLUE}{equal_opp_diff}{COLORS.ENDC}")

    # Calculate Statistical Parity Difference
    prob_pos_priv = (y_pred[priv_mask] == fav_out).mean()
    prob_pos_unpriv = (y_pred[unpriv_mask] == fav_out).mean()
    stat_parity_diff = prob_pos_priv - prob_pos_unpriv
    # print(f"{COLORS.BRIGHT_YELLOW}Statistical Parity Difference: {stat_parity_diff}{COLORS.ENDC}")
    print(f"{COLORS.BRIGHT_YELLOW}{stat_parity_diff}{COLORS.ENDC}")

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
    plt.figure()
    plt.title(my_title)

    if 'sex_Female' in features.columns:
        class_column = 'sex_Female'
        unpriv_condition = features[class_column] == True
        priv_condition = features[class_column] == False
    elif 'sex' in features.columns:
        class_column = 'sex'
        unpriv_condition = features[class_column] == 'Female'
        priv_condition = features[class_column] != 'Female'
    else:
        raise ValueError("Neither 'sex_Female' nor 'sex' column found in features DataFrame")

    rich_women = sum(classes[unpriv_condition] == fav_pred)
    poor_women = sum(classes[unpriv_condition] == unfav_pred)
    rich_men = sum(classes[priv_condition] == fav_pred)
    poor_men = sum(classes[priv_condition] == unfav_pred)

    my_labels = ["Rich women", "Rich men", "Poor women", "Poor men"]

    plt.pie(np.array([rich_women, rich_men, poor_women, poor_men]),
            labels=my_labels,
            autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.gcf().set_size_inches(7, 4)


def crime_pie(features, classes, fav_pred, unfav_pred, my_title):
    """
    A function used for plotting a pie plot for the Adult dataset.

    :param features: the dataframe holding the attributes of the dataset
    :param classes: the dataframe holding the classes of the dataset
    :param fav_pred: the favourable outcome
    :param unfav_pred: the unfavourable outcome
    :param my_title: a string displayed on the produced plot
    """
    plt.figure()
    plt.title(my_title)
    if 'racepctblack_unprivileged' in features.columns:
        class_column = 'racepctblack_unprivileged'
        unpriv_condition = features[class_column] == True
        priv_condition = features[class_column] == False
    elif 'racepctblack' in features.columns:
        class_column = 'racepctblack'
        unpriv_condition = features[class_column] == 'unprivileged'
        priv_condition = features[class_column] != 'unprivileged'
    else:
        raise ValueError("Neither 'racepctblack_unprivileged' nor 'racepctblack' column found in features DataFrame")

    a = sum(classes[unpriv_condition] == fav_pred)
    b = sum(classes[unpriv_condition] == unfav_pred)
    c = sum(classes[priv_condition] == fav_pred)
    d = sum(classes[priv_condition] == unfav_pred)

    my_labels = ["low crime\nblack community", "low crime\nwhite community", "high crime\nblack community", "high crime\nwhite community"]

    plt.pie(np.array([a, c, b, d]), labels=my_labels, autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.gcf().set_size_inches(7, 4)


def bank_pie(features, classes, fav_pred, unfav_pred, my_title):
    """
    A function used for plotting a pie plot for the Adult dataset.

    :param features: the dataframe holding the attributes of the dataset
    :param classes: the dataframe holding the classes of the dataset
    :param fav_pred: the favourable outcome
    :param unfav_pred: the unfavourable outcome
    :param my_title: a string displayed on the produced plot
    """
    plt.figure()
    plt.title(my_title)

    if 'sex_female' in features.columns:
        class_column = 'sex_female'
        unpriv_condition = features[class_column] == True
        priv_condition = features[class_column] == False
    elif 'sex' in features.columns:
        class_column = 'sex'
        unpriv_condition = features[class_column] == 'female'
        priv_condition = features[class_column] != 'female'
    else:
        raise ValueError("Neither 'sex_female' nor 'sex' column found in features DataFrame")

    a = sum(classes[unpriv_condition] == fav_pred)
    b = sum(classes[unpriv_condition] == unfav_pred)
    c = sum(classes[priv_condition] == fav_pred)
    d = sum(classes[priv_condition] == unfav_pred)

    my_labels = ["female\ngood credit", "male\ngood credit", "female\nbad credit", "male\nbad credit"]

    plt.pie(np.array([a, c, b, d]), labels=my_labels, autopct=lambda p: '{:.0f}'.format(p * len(features) / 100))

    plt.gcf().set_size_inches(7, 4)


def pre_crossval(data, data_c, model, priv, unpriv, fav, unfav, folds, crit_region, seed, pie):
    pie_functions = [adult_pie, crime_pie, bank_pie]
    x, y = data
    x_c, y_c = data_c

    k_fold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    classifier = choose_classifier(model)
    c_classifier = choose_classifier(model)

    metrics = ['accuracy', 'DIR', 'samples', 'EQ_OP_D', 'ST_P']
    ROC_metrics = {m: [0] * crit_region for m in metrics}
    CROC_metrics = {m: [0] * crit_region for m in metrics}

    l_values = [i / 100 for i in range(crit_region)]

    for index, (train_indices, test_indices) in enumerate(k_fold.split(x), start=1):
        print(f"{COLORS.MAGENTA}\nFold:{index}{COLORS.ENDC}")

        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        pie_functions[pie](x_test, y_test, fav, unfav, f"Fold:{index} distribution")

        x_train_c, x_test_c = x_c.iloc[train_indices], x_c.iloc[test_indices]
        y_train_c, y_test_c = y_c.iloc[train_indices], y_c.iloc[test_indices]

        classifier.fit(x_train, y_train)
        c_classifier.fit(x_train_c, y_train_c)

        for i in range(crit_region):
            for clf, metric_dict in [(classifier, ROC_metrics), (c_classifier, CROC_metrics)]:
                print(metric_dict)
                pie_func = pie_functions[pie] if i + 1 == crit_region else None
                results = critical_region_test(x_test, y_test, clf, priv, unfav, fav, 0, i / 100, pie_func)

                for m, value in zip(metrics, results):
                    metric_dict[m][i] += value

    # Average the metrics
    for metric_dict in [ROC_metrics, CROC_metrics]:
        for m in metrics:
            metric_dict[m] = [x / folds for x in metric_dict[m]]

    # Plot results
    if crit_region != 1:
        for m in metrics:
            summary_plot(l_values, ROC_metrics[m], CROC_metrics[m], 'ROC', 'ROC+MOD', 'critical region',
                         m, f'{m} vs critical region')


def post_crossval(data, model, priv, unpriv, fav, unfav, folds, crit_region, seed):
    x = data[0]
    y = data[1]

    k_fold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    classifier = choose_classifier(model)

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

    for index, (train_indices, test_indices) in enumerate(k_fold.split(x), start=1):
        print(f"{COLORS.MAGENTA}\nFold:{index}{COLORS.ENDC}")

        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        for i in range(0, crit_region):
            classifier.fit(x_train, y_train)

            acc, DIR, samp, eq_op_d, st_p = critical_region_test(x_test, y_test, classifier, priv,
                                                                 unfav, fav, 0, i / 100, None)

            ROC_accuracy[i] = ROC_accuracy[i] + acc
            ROC_DIR[i] = ROC_DIR[i] + DIR
            ROC_samples[i] = ROC_samples[i] + samp
            ROC_EQ_OP_D[i] = ROC_EQ_OP_D[i] + eq_op_d
            ROC_ST_P[i] = ROC_ST_P[i] + st_p

            acc, DIR, samp, eq_op_d, st_p = attribute_swap_and_critical(x_test, y_test, classifier, unpriv, priv, unfav, fav, 0,
                                                                        i / 100, None)

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

        summary_plot(l_values, ROC_DIR, CROC_DIR, 'ROC', 'ROC+MOD', 'critical region',
                     'DIR', 'DIR vs critical region')

        summary_plot(l_values, ROC_samples, CROC_samples, 'ROC', 'ROC+MOD', 'critical region',
                     'samples', 'samples vs critical region')

        summary_plot(l_values, ROC_EQ_OP_D, CROC_EQ_OP_D, 'ROC', 'ROC+MOD', 'critical region',
                     'samples', 'EQ of opportunity vs critical region')

        summary_plot(l_values, ROC_ST_P, CROC_ST_P, 'ROC', 'ROC+MOD', 'critical region', 'samples',
                     'Statistical Parity vs critical region')


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


def critical_region_test(x, y, classifier, priv, unfav, fav, lower_bound, upper_bound, print_function):
    """
    A function used for applying the Reject Option-based Classification algorithm (ROC) to a classifier.
    :param x: a dataframe holding the attributes of the test dataset
    :param y: a dataframe holding the classes of the test dataset
    :param classifier: a instance of the chosen classifier
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

    if upper_bound != 0:
        print(f"\ncritical region : {upper_bound}")
        print(f"{COLORS.RED}Elements in critical region: {len(indexes)}{COLORS.ENDC}")

    if print_function is not None:
        if upper_bound != 0:
            print_function(x, pred4, fav, unfav, f'critical_region_test l: {upper_bound}')
        else:
            print_function(x, pred4, fav, unfav, '')

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

    print(f"\nCritical region : {upper_bound}")
    print(f"{COLORS.RED}Elements in critical region: {len(indexes)}{COLORS.ENDC}")

    if print_function == adult_pie:
        print_function(x, pred3, fav, unfav, f'attribute_swap_and_critical l: {upper_bound}')
    elif print_function == crime_pie:
        print_function(x, pred3, unfav, fav, f'attribute_swap_and_critical l: {upper_bound}')

    a, b, c, d = calculate_metrics(y, pred3, x_copy, unpriv, fav)

    return a, b, len(indexes), c, d
