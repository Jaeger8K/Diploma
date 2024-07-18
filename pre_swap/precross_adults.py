import sys
import pandas as pd
from fairlearn.datasets import fetch_adult

from Utilities import counterfactual_dataset, preprocess_data, choose_classifier, calculate_metrics, adult_pie,pre_plot_calculation

"""
:param sys.argv[1]: contains the size of the test split. values:[0.1 -0.9]
:param sys.argv[2]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[3]: contains highest value of the l parameter multiplied by 100. values:[1-90]

"""

data = fetch_adult(as_frame=True)

dataframe = data.frame
X_train, X_test, y_train, y_test = preprocess_data(dataframe, float(sys.argv[1]), 'class')
c_X_train, c_X_test, c_y_train, c_y_test = counterfactual_dataset(dataframe, float(sys.argv[1]), 'class', 'sex')

classifier = choose_classifier(sys.argv[2])

c_classifier = choose_classifier(sys.argv[2])

# test_classifier = choose_classifier(sys.argv[2])

# normally trained classifier
classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

# classifier trained with counterfactual data
c_classifier.fit(c_X_train, c_y_train)
c_pred1 = c_classifier.predict(X_test)

# test_classifier.fit(pd.concat([c_X_train, X_train], axis=0),pd.concat([c_y_train, y_train], axis=0))
# test_pred1 = test_classifier.predict(pd.concat([c_X_test, X_test], axis=0))

adult_pie(X_test, y_test, '>50K', '<=50K', 'test data')
adult_pie(X_test, pred1, '>50K', '<=50K', 'classifier predictions')
adult_pie(X_test, c_pred1, '>50K', '<=50K', 'counter classifier predictions')
# adult_pie(X_test, test_pred1, '>50K', '<=50K', 'retrained classifier predictions')

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'sex_Male', '>50K')
print()
calculate_metrics(y_test, c_pred1, X_test, 'sex_Male', '>50K')

pre_plot_calculation(X_test, y_test, classifier, c_classifier, 'sex_Male', 'sex_Female', '>50K', '<=50K')
# pre_plot_calculation(X_test, y_test, classifier, test_classifier, 'sex_Male', 'sex_Female', '>50K', '<=50K')
