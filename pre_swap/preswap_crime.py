import sys
import pandas as pd
from ucimlrepo import fetch_ucirepo
from Utilities import preprocess_data, choose_classifier, crime_pie, calculate_metrics,counterfactual_dataset, pre_plot_calculation

"""
:param sys.argv[1]: contains the size of the test split. values:[0.1 -0.9]
:param sys.argv[2]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[3]: contains highest value of the l parameter multiplied by 100. values:[1-90]

"""
# fetch dataset
communities_and_crime = fetch_ucirepo(id=183)

# data (as pandas dataframes)
X = communities_and_crime.data.features
y = communities_and_crime.data.targets

# Concatenate X and y along columns (axis=1)
dataframe = pd.concat([X, y], axis=1)

# Replace values in the 'ViolentCrimesPerPop' column
dataframe['ViolentCrimesPerPop'] = dataframe['ViolentCrimesPerPop'].apply(lambda x: 'Low_crime' if x < 0.3 else 'High_crime')
dataframe['racepctblack'] = dataframe['racepctblack'].apply(lambda x: 'privileged' if x < 0.06 else 'unprivileged')

X_train, X_test, y_train, y_test = preprocess_data(dataframe, float(sys.argv[1]), 'ViolentCrimesPerPop')
c_X_train, c_X_test, c_y_train, c_y_test = counterfactual_dataset(dataframe, float(sys.argv[1]), 'ViolentCrimesPerPop', 'racepctblack')

classifier = choose_classifier(sys.argv[2])

c_classifier = choose_classifier(sys.argv[2])

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

c_classifier.fit(c_X_train, c_y_train)
c_pred1 = c_classifier.predict(X_test)

crime_pie(X_test, y_test, 'Low_crime', 'High_crime', 'test data')
crime_pie(X_test, pred1, 'Low_crime', 'High_crime', 'classifier predictions')
crime_pie(X_test, c_pred1, 'Low_crime', 'High_crime', 'counter classifier predictions')

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'racepctblack_unprivileged', 'High_crime')
print()
calculate_metrics(y_test, c_pred1, X_test, 'racepctblack_unprivileged', 'High_crime')

pre_plot_calculation(X_test, y_test, classifier,c_classifier , 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime')
