import pandas as pd
from sklearn.metrics import accuracy_score

from Utilities import  choose_classifier, preprocess_data

data = pd.read_csv('Datasets/credit_risk_dataset.csv')
data = data.dropna()

X_train, X_test, y_train, y_test = preprocess_data(data, 0.3, 'cb_person_default_on_file')

classifier = choose_classifier("1")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

accuracy = accuracy_score(y_test, pred1)
print("Accuracy of the classifier:", accuracy)
