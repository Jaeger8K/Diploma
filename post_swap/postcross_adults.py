import sys
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import KFold
from Utilities import calculate_metrics, choose_classifier, attribute_swap_test, adult_pie, critical_region_test, \
     attribute_swap_and_critical, cross_validation_load

"""
:param sys.argv[1]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[2]: contains the l parameter of the ROC algorithm. values:[0.1-0.9]
:param sys.argv[3]: contains the l parameter of the modified algorithm. values:[0.1-0.9]

"""

# fetch dataset
data = fetch_adult(as_frame=True)
dataframe = data.frame

X, y = cross_validation_load(dataframe, 'class')

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

classifier = choose_classifier(sys.argv[1])

# Iterate over each fold
for train_indices, test_indices in k_fold.split(X):
    # Get the training and testing data for this fold
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    classifier.fit(X_train, y_train)
    pred1 = classifier.predict(X_test)

    # adult_pie(X_test, y_test, '>50K', '<=50K', 'actual data')
    # adult_pie(X_test, pred1, '>50K', '<=50K', 'unaltered results')

    print()
    print(classifier)

    # Calculate the metrics for the vanilla classifier.
    calculate_metrics(y_test, pred1, X_test, 'sex_Male', '>50K')

    # Calculate the metrics for the modified classifier.
    attribute_swap_test(X_test, y_test, classifier, 'sex_Male', 'sex_Female', '<=50K', '>50K', adult_pie)
    critical_region_test(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, float(sys.argv[2]), adult_pie)
    attribute_swap_and_critical(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, float(sys.argv[3]), adult_pie)
