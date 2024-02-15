import pandas as pd
from matplotlib import pyplot as plt
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def show_metrics():
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


def disparate_impact_ratio(y_pred, sensitive_features):
    # Calculate the selection rate for each group
    selection_rates = {}
    # print(sensitive_features.unique())
    groups = sensitive_features.unique()
    for group in groups:
        mask = (sensitive_features == group)
        selection_rates[group] = sum(y_pred[mask]) / sum(mask)

    # Calculate the disparate impact ratio
    ratio = selection_rates[groups[1]] / selection_rates[groups[0]]
    return ratio


def predict(features, label, decision):
    # print(features['sex'])

    classifier = choose_classifier(decision)
    y_pred = classifier.predict(features)

    print()
    print(classifier)

    # Calculate the disparate impact
    di = disparate_impact_ratio(y_pred, sensitive_features=features['sex'])
    print("Disparate Impact Ratio:", di)

    # Calculate the accuracy
    accuracy = accuracy_score(label, y_pred)
    print("Accuracy of the classifier:", accuracy)

    # Perform 10-fold cross-validation
    # cross_val_scores = cross_val_score(classifier, X, y, cv=10)
    # Print the cross-validation scores
    # print("Cross-validation scores:", cross_val_scores, "\n")

    return y_pred


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

    elif model_selection == "5":  # works
        m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter = 1000)
        # m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=350)

    elif model_selection == "6":  # needs work, 'KMeans' object has no attribute 'predict_proba'
        m = KMeans(n_clusters=2, random_state=0, n_init="auto")

    elif model_selection == "7":  # works
        m = LinearSVC(dual='auto')

    # elif model_selection == "8":  # needs work, doesnt terminate
    # m = svm.SVC()

    m.fit(X_train, y_train)
    return m


# Load the CSV file into a DataFrame
X = pd.read_csv('normalized_features.csv')
y = pd.read_csv('y.csv')
y = y['income'].to_numpy()

male_tag = X.iloc[0]['sex']
female_tag = X.iloc[4]['sex']
poor_tag = y[0]
rich_tag = y[9]

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

classifier = "7"

print("\nStandard results ")

pred1 = predict(X_test, y_test, classifier)

X_test.loc[:, 'sex'] = X['sex'].replace({male_tag: female_tag, female_tag: male_tag})

print("\nResults when switching the protected values.")

pred2 = predict(X_test, y_test, classifier)

print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

# export(X)

# show_metrics()
