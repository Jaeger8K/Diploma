import Utilities
from fairlearn.datasets import fetch_adult

data = fetch_adult(as_frame=True)

dataframe = data.frame

X_train, X_test, y_train, y_test = Utilities.preprocess_adults(dataframe, 0.3)

classifier = Utilities.choose_classifier("7")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

print(classifier)

Utilities.test_adults(y_test, pred1, X_test, False)

X_test['sex_Female'] = ~X_test['sex_Female']
X_test['sex_Male'] = ~X_test['sex_Male']

pred2 = classifier.predict(X_test)

print("\nResults after swapping the protected attribute values.\n")

Utilities.test_adults(y_test, pred2, X_test, True)

print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

# print(X_test[['sex_Female', 'sex_Male']])
# print(X_test[['sex_Female', 'sex_Male']])
