import pandas as pd

from Utilities import preprocess_data, choose_classifier, calculate_metrics, attribute_swap_test, adult_pie

data = pd.read_csv('../Datasets/ΙΒΜ_attrition.csv')

print(data.columns)

# print(data['Attrition'])
print(data['Gender'])

X_train, X_test, y_train, y_test = preprocess_data(data, 0.3, 'Attrition')

classifier = choose_classifier("3")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'Gender_Male', 'No')

attrition_counts = data['Attrition'].value_counts()
num_attrition_yes = attrition_counts.get('Yes', 0)
num_attrition_no = attrition_counts.get('No', 0)

print("Number of 'Attrition' with value 'Yes':", num_attrition_yes)
print("Number of 'Attrition' with value 'No':", num_attrition_no)

