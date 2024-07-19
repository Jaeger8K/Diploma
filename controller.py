import sys
import pandas as pd
from fairlearn.datasets import fetch_adult
from ucimlrepo import fetch_ucirepo
from Utilities import preprocess_data, preprocess_counterfactual_dataset, pre_crossval, post_crossval

"""
:param sys.argv[1]: contains the choice of dataset. values:[1,2]
:param sys.argv[2]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[3]: contains the choice of algorithm. values:[1,2]
:param sys.argv[4]: contains highest value of the critical region. values:[1-90]
:param sys.argv[5]: contains the number of folds:[1-10]

"""


def dataset_selection(input):
    if input == 1:
        data = fetch_adult(as_frame=True)

        return data.frame, 'class', 'sex', 'sex_Male', 'sex_Female', '>50K', '<=50K'

    elif input == 2:
        communities_and_crime = fetch_ucirepo(id=183)

        # data (as pandas dataframes)
        X = communities_and_crime.data.features
        y = communities_and_crime.data.targets

        # Concatenate X and y along columns (axis=1)
        dataframe = pd.concat([X, y], axis=1)

        # Replace values in the 'ViolentCrimesPerPop' column
        dataframe['ViolentCrimesPerPop'] = dataframe['ViolentCrimesPerPop'].apply(lambda x: 'Low_crime' if x < 0.3 else 'High_crime')
        dataframe['racepctblack'] = dataframe['racepctblack'].apply(lambda x: 'privileged' if x < 0.06 else 'unprivileged')

        return dataframe, 'ViolentCrimesPerPop', 'racepctblack', 'racepctblack_privileged',\
            'racepctblack_unprivileged', 'Low_crime', 'High_crime'


dataframe, class_at, prot_at, priv, unpriv, fav, unfav = dataset_selection(int(sys.argv[1]))

a = preprocess_data(dataframe, class_at)

if int(sys.argv[3]) == 1:

    b = preprocess_counterfactual_dataset(dataframe, class_at, prot_at)

    pre_crossval(a, b, sys.argv[2], priv, unpriv, fav, unfav, int(sys.argv[5]), int(sys.argv[4]))

elif int(sys.argv[3]) == 2:
    post_crossval(a, sys.argv[2], priv, unpriv, fav, unfav, int(sys.argv[5]), int(sys.argv[4]))
