from fairlearn.datasets import fetch_adult

from Utilities import double_data, choose_classifier, adult_pie, calculate_metrics, preprocess_data

data = fetch_adult(as_frame=True)

dataframe = data.frame

X_train, X_test, y_train, y_test = double_data(dataframe, 0.3, ['sex_Female', 'sex_Male'], 'class')

test_X_train, test_X_test, test_y_train, test_y_test = preprocess_data(dataframe, 0.3, 'class')

classifier = choose_classifier("5")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(test_X_test)

adult_pie(test_X_test, test_y_test, '>50K', '<=50K', 'actual data')
adult_pie(test_X_test, pred1, '>50K', '<=50K', 'prediction')

print()
print(classifier)

calculate_metrics(test_y_test, pred1, test_X_test, 'sex_Female', '>50K')
