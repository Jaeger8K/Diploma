import pandas as pd
from sklearn.metrics import accuracy_score

from Utilities import choose_classifier, preprocess_data

data = pd.read_csv('Datasets/bank_dataset.csv')
data = data.dropna()

X_train, X_test, y_train, y_test = preprocess_data(data, 0.3, 'y')

classifier = choose_classifier("4")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

accuracy = accuracy_score(y_test, pred1)
print("Accuracy of the classifier:", accuracy)
