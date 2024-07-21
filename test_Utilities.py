import pandas as pd
from fairlearn.datasets import fetch_adult
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import AdultDataset
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric

# Load the dataset
dataset = fetch_adult(as_frame=True)

# Split the data into train and test
train, test = train_test_split(dataset.frame, test_size=0.3, shuffle=True)



# Train a simple logistic regression model
X_train = train.drop(columns=['class'])
y_train = train['class']

X_test = test.drop(columns=['class'])
y_test = test['class']

clf = LogisticRegression(solver='liblinear', random_state=1)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the original accuracy
original_acc = accuracy_score(y_test, y_pred)
print(f"Original Accuracy: {original_acc:.4f}")

# Create the dataset object for the predicted labels
pred_dataset = test.copy()
pred_dataset.labels = y_pred.reshape(-1, 1)

# Apply Calibrated Equalized Odds Postprocessing
cpp = CalibratedEqOddsPostprocessing(privileged_groups=[{'sex': 1}],
                                     unprivileged_groups=[{'sex': 0}],
                                     cost_constraint="weighted",
                                     seed=1)

cpp = cpp.fit(test, pred_dataset)
pred_cpp = cpp.predict(pred_dataset)

# Evaluate the accuracy after postprocessing
accuracy_after_cpp = accuracy_score(test.labels, pred_cpp.labels)
print(f"Accuracy after CPP: {accuracy_after_cpp:.4f}")

# Evaluate the fairness metrics
metric_pred = BinaryLabelDatasetMetric(pred_dataset,
                                       unprivileged_groups=[{'sex': 0}],
                                       privileged_groups=[{'sex': 1}])

metric_pred_cpp = BinaryLabelDatasetMetric(pred_cpp,
                                           unprivileged_groups=[{'sex': 0}],
                                           privileged_groups=[{'sex': 1}])

print(f"Disparate impact before CPP: {metric_pred.disparate_impact()}")
print(f"Disparate impact after CPP: {metric_pred_cpp.disparate_impact()}")

# Additional fairness metrics can be evaluated using ClassificationMetric
classified_metric_cpp = ClassificationMetric(test, pred_cpp,
                                             unprivileged_groups=[{'sex': 0}],
                                             privileged_groups=[{'sex': 1}])

print(f"Equal opportunity difference after CPP: {classified_metric_cpp.equal_opportunity_difference()}")
print(f"Average odds difference after CPP: {classified_metric_cpp.average_odds_difference()}")