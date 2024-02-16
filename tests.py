# fetch dataset
from ucimlrepo import fetch_ucirepo

from Utilities import export_csv

adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

export_csv(X, 'test_output.csv')