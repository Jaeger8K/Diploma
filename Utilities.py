import pandas as pd
from fairlearn.datasets import fetch_adult
from matplotlib import pyplot as plt
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


def preprocess_adults(data, test_size):
    # Assuming 'df' is your DataFrame with numerical columns to be normalized
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Specify the numerical columns to be normalized
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # Fit the scaler on the numerical data
    scaler.fit(data[numerical_columns])

    # Transform and replace the original numerical data with the normalized values
    data[numerical_columns] = scaler.transform(data[numerical_columns])

    X = data.drop('class', axis=1)  # Drop the 'class' column to get features (X)
    y = data['class']  # Extract the 'class' column as the target variable (y)

    x_dummies = pd.get_dummies(X)
    # y_dummies = pd.get_dummies(y)

    return train_test_split(x_dummies, y, test_size=test_size, random_state=1)


def choose_classifier(model_selection):
    m = 0

    if model_selection == "1":  # works
        m = LogisticRegression(random_state=16)

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


def test_adults(y_test, y_pred, x_test, reverse):
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the classifier:", accuracy)

    if not reverse:

        disparate_impact = ((y_pred == '>50K') & (x_test['sex_Female'] == True)).sum() / ((y_pred == '>50K') & (x_test['sex_Female'] == False)).sum()
        # print(((y_pred == '>50K') & (x_test['sex_Female'] == True)).sum())
        # print(((y_pred == '>50K') & (x_test['sex_Female'] == False)).sum())
        print("Disparate Impact Ratio:", disparate_impact)

    elif reverse:

        disparate_impact = ((y_pred == '>50K') & (x_test['sex_Female'] == False)).sum() / ((y_pred == '>50K') & (x_test['sex_Female'] == True)).sum()
        print("Disparate Impact Ratio:", disparate_impact)

    '''    # Calculate disparate impact ratio
        # Mean of favorable outcomes for protected group
        fav_outcomes_prot = (y_pred[x_test['sex_Female'] == True] == '>50K').mean()
        # Mean of favorable outcomes for non-protected group
        fav_outcomes_non_prot = (y_pred[x_test['sex_Female'] == False] == '>50K').mean()

        disparate_impact = fav_outcomes_prot / fav_outcomes_non_prot
        print("Disparate Impact Ratio:", disparate_impact)
        #print(len(y_pred[x_test['sex_Female'] == True] == '>50K'))
        #print(len(y_pred[x_test['sex_Female'] == False] == '>50K'))

        # Filter predictions where y_pred is '>50K' and X_test['sex_Female'] is True
        condition = (y_pred == '>50K') & (x_test['sex_Female'] == True)
        print("Sum of corresponding samples:", condition.sum())
        condition = (y_pred == '>50K') & (x_test['sex_Female'] == False)
        print("Sum of corresponding samples:", condition.sum())'''


def show_metrics_adults(classifier):
    data = fetch_adult(as_frame=True)
    X = pd.get_dummies(data.data)
    y_true = (data.target == '>50K') * 1
    sex = data.data['sex']

    classifier.fit(X, y_true)

    y_pred = classifier.predict(X)

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

    # Customize plots with ylim
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

    # Customize plots with kind (note that we are only plotting the "count" metric here because we are showing a pie chart)
    metric_frame.by_group[["count"]].plot(
        kind="pie",
        subplots=True,
        layout=[1, 1],
        legend=False,
        figsize=[12, 8],
        title="Show count metric in pie chart",
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
    plt.show()
