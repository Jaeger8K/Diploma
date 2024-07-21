import sys
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
from utilities import calculate_metrics, choose_classifier, preprocess_data, adult_pie, \
    post_plot_calculation, pie_plot

"""
:param sys.argv[1]: contains the size of the test split. values:[0.1 -0.9]
:param sys.argv[2]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[3]: contains highest value of the l parameter multiplied by 100. values:[1-90]

"""

# fetch dataset
data = fetch_adult(as_frame=True)
dataframe = data.frame

a = preprocess_data(dataframe, 'class')
X_train, X_test, y_train, y_test = train_test_split(a[0], a[1], test_size=float(sys.argv[1]), random_state=1)

classifier = choose_classifier(sys.argv[2])

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

pie_labels = ["Rich women", "Rich men", "Poor women", "Poor men"]
pie_plot(X_test, y_test, '>50K', '<=50K', 'sex_Female', pie_labels, 'test data')
pie_plot(X_test, pred1, '>50K', '<=50K', 'sex_Female', pie_labels, 'classifier predictions')

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'sex_Male', '>50K')

post_plot_calculation(X_test, y_test, classifier, 'sex_Male', 'sex_Female', '>50K', '<=50K')