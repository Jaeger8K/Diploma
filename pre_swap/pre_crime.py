import sys

import pandas as pd
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import KFold
from ucimlrepo import fetch_ucirepo

from Utilities import preprocess_data, choose_classifier, preprocess_counterfactual_dataset, pre_crossval

"""
:param sys.argv[1]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[2]: contains the l parameter of the ROC algorithm. values:[0.1-0.9]
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

a = preprocess_data(dataframe, 'ViolentCrimesPerPop')

b = preprocess_counterfactual_dataset(dataframe, 'ViolentCrimesPerPop', 'racepctblack')

pre_crossval(a, b, sys.argv[1], 'racepctblack_privileged', 'racepctblack_unprivileged', 'Low_crime', 'High_crime')