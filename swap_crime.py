import sys

import pandas as pd
from ucimlrepo import fetch_ucirepo
from Utilities import preprocess_data, choose_classifier, crime_pie, calculate_metrics, attribute_swap_test, plot_calculation, \
    critical_region_test, attribute_swap_and_critical

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

classifier = choose_classifier(sys.argv[2])

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

crime_pie(X_test, y_test, 'Low_crime', 'High_crime', 'test data')
crime_pie(X_test, pred1, 'Low_crime', 'High_crime', 'classifier predictions')

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'racepctblack_unprivileged', 'High_crime')

attribute_swap_test(X_test, y_test, classifier, 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime', crime_pie)
# (X_test, y_test, classifier, 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime', 0, 0.15, crime_pie)
# attribute_swap_and_critical(X_test, y_test, classifier, 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime', 0, 0.15, crime_pie)

plot_calculation(X_test, y_test, classifier, 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime')