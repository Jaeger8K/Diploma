import sys
from fairlearn.datasets import fetch_adult

from Utilities import calculate_metrics, choose_classifier, preprocess_data, attribute_swap_test, adult_pie, plot_calculation

"""
:param sys.argv[1]: contains the size of the test split. values:[0.1 -0.9]
:param sys.argv[2]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[3]: contains highest value of the l parameter multiplied by 100. values:[1-90]

"""

data = fetch_adult(as_frame=True)

dataframe = data.frame

X_train, X_test, y_train, y_test = preprocess_data(dataframe, float(sys.argv[1]), 'class')

classifier = choose_classifier(sys.argv[2])

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

adult_pie(X_test, y_test, '>50K', '<=50K', 'test data')
adult_pie(X_test, pred1, '>50K', '<=50K', 'classifier predictions')

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'sex_Male', '>50K')

# attribute_swap_test(X_test, y_test, classifier, 'sex_Male', 'sex_Female', '<=50K', '>50K', None)
# critical_region_test(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, sys.argv[3], None)
# attribute_swap_and_critical(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, sys.argv[3], None)

attribute_swap_test(X_test, y_test, classifier, 'sex_Male', 'sex_Female', '<=50K', '>50K', adult_pie)
# critical_region_test(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, 0.1, adult_pie)
# attribute_swap_and_critical(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, 0.1, adult_pie)

# show_metrics_adults(classifier)
# Print the name of the script being executed

plot_calculation(X_test, y_test, classifier, 'sex_Male', 'sex_Female', '>50K', '<=50K')