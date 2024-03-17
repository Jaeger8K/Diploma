import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from Utilities import choose_classifier, preprocess_data, handle_age

'''
cb_person_default_on_file could indicate whether a person has a default entry on their credit report.
A default entry typically occurs when a borrower fails to repay a debt as agreed, 
resulting in negative consequences for their credit history.
'''

data = pd.read_csv('../Datasets/credit_risk_dataset.csv')
data = data.dropna()

handle_age(data, 'person_age', 25, 55)

X_train, X_test, y_train, y_test = preprocess_data(data, 0.3, 'cb_person_default_on_file')

classifier = choose_classifier("1")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

accuracy = accuracy_score(y_test, pred1)
print("Accuracy of the classifier:", accuracy)

print(X_test.columns)

print(sum(y_test == 'N'))
# print(sum(y_test == 'Y'))
print(((pred1 == 'N') & (X_test['person_age_young'] == True)).sum(), (X_test['person_age_young'] == True).sum())
print(((pred1 == 'N') & (X_test['person_age_adult'] == True)).sum(), (X_test['person_age_adult'] == True).sum())
print(((pred1 == 'N') & (X_test['person_age_elder'] == True)).sum(), (X_test['person_age_elder'] == True).sum())

print(((pred1 == 'Y') & (X_test['person_age_young'] == True)).sum(), (X_test['person_age_young'] == True).sum())
print(((pred1 == 'Y') & (X_test['person_age_adult'] == True)).sum(), (X_test['person_age_adult'] == True).sum())
print(((pred1 == 'Y') & (X_test['person_age_elder'] == True)).sum(), (X_test['person_age_elder'] == True).sum())

# print(((pred1 == fav_out) & (x_test[unp_group] == True)).sum())

'''
calculate_credit_mertics(y_test, pred1, X_test, 'person_age_young', 'Y')
calculate_credit_mertics(y_test, pred1, X_test, 'person_age_elder', 'Y')
calculate_credit_mertics(y_test, pred1, X_test, 'person_age_adult', 'Y')
'''
