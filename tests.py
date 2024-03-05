from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.datasets import fetch_adult
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Load dataset (using Fairlearn's example dataset)
data = fetch_adult(as_frame=True)
X = data.data
y = (data.target == '>50K') * 1  # Binary classification task: '>50K' vs. '<=50K'

# Perform one-hot encoding for categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a logistic regression classifier on the dataset.
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Instantiate ThresholdOptimizer
fairness_constraints = {"sex": 0.05}  # Example fairness constraint: 5% difference in accuracy between ethnic groups
thop = ThresholdOptimizer(estimator=classifier, constraints=fairness_constraints, prefit=True)

print(X_test.columns)

# Post-process predictions
y_fair_pred = thop.predict(X_test, sensitive_features=X_test['sex'])

# Evaluate accuracy
accuracy_fair = accuracy_score(y_test, y_fair_pred)

print(f"Fairness-aware accuracy: {accuracy_fair}")