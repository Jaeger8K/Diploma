import sys
import pandas as pd
from sklearn.model_selection import KFold
from ucimlrepo import fetch_ucirepo
from Utilities import calculate_metrics, choose_classifier, attribute_swap_test, critical_region_test, \
    attribute_swap_and_critical, cross_validation_load, crime_pie

# fetch dataset
communities_and_crime = fetch_ucirepo(id=183)

# data (as pandas dataframes)
X = communities_and_crime.data.features
y = communities_and_crime.data.targets

# Concatenate X and y along columns (axis=1)
dataframe = pd.concat([X, y], axis=1)

# Replace values in the 'ViolentCrimesPerPop' column
dataframe['ViolentCrimesPerPop'] = dataframe['ViolentCrimesPerPop'].apply(lambda x: 'Low_crime' if x < 0.1 else 'High_crime')
dataframe['racepctblack'] = dataframe['racepctblack'].apply(lambda x: 'privileged' if x < 0.06 else 'unprivileged')

X, y = cross_validation_load(dataframe, 'ViolentCrimesPerPop')

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

classifier = choose_classifier(sys.argv[1])

# Iterate over each fold
for train_indices, test_indices in k_fold.split(X):
    # Get the training and testing data for this fold
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    classifier.fit(X_train, y_train)
    pred1 = classifier.predict(X_test)

    print()
    print(classifier)

    calculate_metrics(y_test, pred1, X_test, 'racepctblack_unprivileged', 'High_crime')

    attribute_swap_test(X_test, y_test, classifier, 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime', crime_pie)
    critical_region_test(X_test, y_test, classifier, 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime', 0, float(sys.argv[2]), crime_pie)
    attribute_swap_and_critical(X_test, y_test, classifier, 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime', 0, float(sys.argv[3]), crime_pie)
