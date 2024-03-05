import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)


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


def export_csv(dataframe, name):
    dataframe.to_csv(name, index=False)


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

    return train_test_split(x_dummies, y, test_size=test_size, random_state=1)


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


def choose_classifier(model_selection):
    m = 0

    if model_selection == "1":  # works
        m = LogisticRegression(random_state=16, max_iter=1000)

    elif model_selection == "2":  # works
        m = RandomForestClassifier(max_depth=5, random_state=0)

    # elif model_selection == "3":  # works
        # m = GaussianNB()

    # elif model_selection == "4":  # works
        # m = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)

    elif model_selection == "3":  # works, needs more iterations, takes time
        m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=350)

    # elif model_selection == "6":  # needs work, 'KMeans' object has no attribute 'predict_proba'
        # m = KMeans(n_clusters=2, random_state=0, n_init="auto")

    elif model_selection == "4":  # works 'LinearSVC' object has no attribute 'predict_proba'. Did you mean: '_predict_proba_lr'?
        m = LinearSVC(dual='auto')

    # elif model_selection == "8":  # needs work, doesnt terminate
    #    m = svm.SVC()

    return m


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
    print(a,b)


def show_metrics_adults(classifier):
    data = fetch_adult(as_frame=True)
    X = pd.get_dummies(data.data)
    y_true = (data.target == '>50K') * 1
    sex = data.data['sex']

    classifier.fit(X, y_true)

    y_pred = classifier.predict(X)

    adult_pie(X, y_true)

    # Analyze metrics using MetricFrame
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "false positive rate": false_positive_rate,
        "false negative rate": false_negative_rate,
        "selection rate": selection_rate,
        "count": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sex
    )
    metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[12, 8],
        title="Show all metrics",
    )

    plt.show()

    """    # Customize plots with ylim
        metric_frame.by_group.plot(
            kind="bar",
            ylim=[0, 1],
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Show all metrics with assigned y-axis range",
        )
            # Customize plots with colormap
        metric_frame.by_group.plot(
        kind="bar",
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[12, 8],
        colormap="Accent",
        title="Show all metrics in Accent colormap",
    )
        # Saving plots
    fig = metric_frame.by_group[["count"]].plot(
        kind="pie",
        subplots=True,
        layout=[1, 1],
        legend=False,
        figsize=[12, 8],
        title="Show count metric in pie chart",
    )

    # Don't save file during doc build
    if "__file__" in locals():
        fig[0][0].figure.savefig("filename.png")
        
        # Customize plots with kind (note that we are only plotting the "count" metric here because we are showing a pie chart)
    metric_frame.by_group[["count"]].plot(
        kind="pie",
        subplots=True,
        layout=[1, 1],
        legend=False,
        figsize=[12, 8],
        title="Show count metric in pie chart",
    )
    """


def partitioning(lower_bound, upper_bound, classifier_prob):
    indexes = [idx for idx, probabilities in enumerate(classifier_prob) if
               lower_bound < np.abs(probabilities[0] - probabilities[1]) < upper_bound]
    # print(len(indexes))

    return indexes


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

    print(f"\ncritical_region_test l = {upper_bound}")

    calculate_metrics(y, pred4, x, priv, fav)

    print_function(x, pred4, fav, unfav, f'critical_region_test l = {upper_bound}')


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

    print(f"\nattribute_swap_and_critical. l = {upper_bound}")

    calculate_metrics(y, pred3, x_copy, unpriv, fav)

    print_function(x, pred3, fav, unfav, f'attribute_swap_and_critical l = {upper_bound}')
