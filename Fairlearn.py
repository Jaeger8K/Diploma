import pandas as pd
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import selection_rate, MetricFrame, false_positive_rate, false_negative_rate
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier

data = fetch_adult(as_frame=True)

# print(data)

X = pd.get_dummies(data.data)

y_true = (data.target == '>50K') * 1
print(y_true)

sex = data.data['sex']
counts = sex.value_counts()

print(sex.value_counts(), "\n")

classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)

classifier.fit(X, y_true)

y_pred = classifier.predict(X)

mf = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=sex)

print("Overall accuracy score:", mf.overall, "\n")
print(mf.by_group, "\n")

sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sex)

print("Overall Selection Rate:", sr.overall, "\n")  # the percentage of the population which have ‘1’ as their label:
print(sr.by_group, "\n")

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

print(len(X))