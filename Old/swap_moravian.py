import numpy as np
import pandas as pd

from utilities import preprocess_data, choose_classifier, calculate_metrics


def initialize_data(d):
    # Drop the 'Employee_Name' column
    d.drop(columns=['Employee_Name'], inplace=True)

    # Fill NaN values in the 'DateofTermination' column with '30/12/2017'
    d['DateofTermination'] = d['DateofTermination'].fillna('30/12/2017')

    # Fill NaN values in the 'ManagerID' column with 'unknown'
    d['ManagerID'] = d['ManagerID'].fillna('unknown')

    # Assuming 'data' is your DataFrame
    d['Salary'] = np.where(d['Salary'] < 80000, '<=80K', '>80K')

    print(d['Salary'])

    return d


# Read the data from the CSV file
d = pd.read_csv('../Datasets/HR_Moravian.csv')
data = initialize_data(d)


X_train, X_test, y_train, y_test = preprocess_data(data, 0.3, 'Salary')

classifier = choose_classifier("2")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

# Assuming 'data' is your DataFrame
for column_name in X_test.columns:
    print(column_name)

print(X_train['Sex_F'])

calculate_metrics(y_test, pred1, X_test, 'Sex_F', '>80K')

