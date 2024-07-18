import sys
from fairlearn.datasets import fetch_adult
from Utilities import preprocess_data, preprocess_counterfactual_dataset, pre_crossval

"""
:param sys.argv[1]: contains the choice of classifier. values:[1,2,3,4]
:param sys.argv[2]: contains highest value of the l parameter multiplied by 100. values:[1-90]
:param sys.argv[3]: contains the number of folds:[1-10]

"""

# fetch dataset
data = fetch_adult(as_frame=True)
dataframe = data.frame

a = preprocess_data(dataframe, 'class')

b = preprocess_counterfactual_dataset(dataframe, 'class', 'sex')

pre_crossval(a, b, sys.argv[1], 'sex_Male', 'sex_Female', '>50K', '<=50K', int(sys.argv[3]))
