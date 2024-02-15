import pandas as pd
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import selection_rate, MetricFrame, false_positive_rate, false_negative_rate
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def show_metrics():
    # Plotting other metrics
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "false positive rate": false_positive_rate,
        "false negative rate": false_negative_rate,
        "selection rate": selection_rate,
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

    elif model_selection == "8":  # needs work, doesnt terminate
        m = svm.SVC()

    return m


data = fetch_adult(as_frame=True)

X = pd.get_dummies(data.data)
y_true = (data.target == '>50K') * 1
sex = data.data['sex']

# classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier = choose_classifier("RandomForest")
classifier.fit(X, y_true)
y_pred = classifier.predict(X)

print(classifier)

# Calculate the disparate impact
di = disparate_impact_ratio(y_pred, sensitive_features=sex)
print("Disparate Impact Ratio:", di)

# Calculate the accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy of the classifier:", accuracy)

