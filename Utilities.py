import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)


def export_csv(dataframe, name):
    dataframe.to_csv(name, index=False)


def preprocess_data(dataframe, test_size, class_label):

    scaler = MinMaxScaler()

    # Assuming df is your DataFrame
    numerical_columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    scaler.fit(dataframe[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    dataframe[numerical_columns] = scaler.transform(dataframe[numerical_columns])

    X = dataframe.drop(class_label, axis=1)  # Drop the 'class' column to get features (X)
    y = dataframe[class_label]  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)

    return train_test_split(x_dummies, y, test_size=test_size, random_state=1)


def preprocess_credit_risk(test_size):
    credit_risk_df = pd.read_csv('Datasets/credit_risk_dataset.csv')

    credit_risk_df = credit_risk_df.dropna()

    scaler = MinMaxScaler()

    numerical_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                         'loan_percent_income', 'cb_person_cred_hist_length']

    scaler.fit(credit_risk_df[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    credit_risk_df[numerical_columns] = scaler.transform(credit_risk_df[numerical_columns])

    X = credit_risk_df.drop('cb_person_default_on_file', axis=1)  # Drop the 'class' column to get features (X)
    y = credit_risk_df['cb_person_default_on_file']  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)

    return train_test_split(x_dummies, y, test_size=test_size, random_state=1)


def choose_classifier(model_selection):
    m = 0

    if model_selection == "1":  # works
        m = LogisticRegression(random_state=16, max_iter=1000)

    elif model_selection == "2":  # works
        m = RandomForestClassifier(max_depth=5, random_state=0)

    elif model_selection == "3":  # works
        m = GaussianNB()

    elif model_selection == "4":  # works
        m = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)

    elif model_selection == "5":  # works, needs more iterations, takes time
        m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=350)

    elif model_selection == "6":  # needs work, 'KMeans' object has no attribute 'predict_proba'
        m = KMeans(n_clusters=2, random_state=0, n_init="auto")

    elif model_selection == "7":  # works 'LinearSVC' object has no attribute 'predict_proba'. Did you mean: '_predict_proba_lr'?
        m = LinearSVC(dual='auto')

    # elif model_selection == "8":  # needs work, doesnt terminate
    #    m = svm.SVC()

    return m


def calculate_mertics(y_test, y_pred, x_test, unp_group, fav_out):
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the classifier:", accuracy)

    a = ((y_pred == fav_out) & (x_test[unp_group] == True)).sum()
    b = ((y_pred == fav_out) & (x_test[unp_group] == False)).sum()
    disparate_impact = a / b
    # print(((y_pred == '>50K') & (x_test['sex_Female'] == True)).sum())
    # print(((y_pred == '>50K') & (x_test['sex_Female'] == False)).sum())
    print("Disparate Impact Ratio:", disparate_impact)


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


def student_ttest(data, label, label_values):
    group1 = data[data[label] == label_values[0]]
    group1 = group1.drop(columns=[label])

    group2 = data[data[label] == label_values[1]]
    group2 = group2.drop(columns=[label])

    # Perform the t-test
    t_statistic, p_value = ttest_ind(group1, group2)

    print(data.columns)

    results_df = pd.DataFrame({'Attribute': group1.columns, 'T-statistic': t_statistic, 'P-value': p_value})

    # Sort the DataFrame by the absolute T-statistic value in descending order
    results_df['Abs_T-statistic'] = abs(results_df['T-statistic'])
    results_df = results_df.sort_values(by='Abs_T-statistic', ascending=False)

    print(results_df)

    temp = results_df.index.tolist()
