import pandas as pd
from sklearn.metrics import accuracy_score

from Utilities import choose_classifier, preprocess_data, handle_age, bank_pie, partitioning, calculate_metrics, \
    critical_region_test


class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def attribute_swap_test(unpriv, priv, fav):
    for index, row in X_test_mod.iterrows():

        if row['age_adult']:
            X_test_mod.at[index, 'age_adult'] = False
            X_test_mod.at[index, 'age_elder'] = True
            X_test_mod.at[index, 'age_young'] = False
        elif row['age_young']:
            X_test_mod.at[index, 'age_adult'] = True
            X_test_mod.at[index, 'age_elder'] = False
            X_test_mod.at[index, 'age_young'] = False
        elif row['age_elder']:
            X_test_mod.at[index, 'age_adult'] = True
            X_test_mod.at[index, 'age_elder'] = False
            X_test_mod.at[index, 'age_young'] = False

    pred2 = classifier.predict(X_test_mod)

    print("\nattribute_swap_test.")

    calculate_metrics(y_test, pred2, X_test, unpriv, fav)

    # print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

    bank_pie(X_test, pred2, 'yes', 'no', 'attribute_swap_test')


def attribute_swap_and_critical(unpriv, priv, fav, lower_bound, upper_bound):
    pred3 = classifier.predict(X_test_mod)

    indexes = partitioning(lower_bound, upper_bound, classifier.predict_proba(X_test_mod))
    feature_part = X_test_mod.iloc[indexes]

    for iteration_number, (index, row) in enumerate(feature_part.iterrows(), start=0):

        if row['age_young']:
            pred3[indexes[iteration_number]] = 'yes'

        elif row['age_elder']:
            pred3[indexes[iteration_number]] = 'yes'

        elif row['age_adult']:
            pred3[indexes[iteration_number]] = 'no'

    print(f"\nResults after clearing critical region. l = {upper_bound}")
    print(f"{colors.RED}Elements in critical region: {len(indexes)}{colors.ENDC}")

    calculate_metrics(y_test, pred3, X_test_mod, unpriv, fav)

    bank_pie(X_test, pred3, 'yes', 'no', 'swapped values and critical region')


data = pd.read_csv('Datasets/bank_dataset.csv')
data = data.dropna()

handle_age(data, 'age', 30, 60)

X_train, X_test, y_train, y_test = preprocess_data(data, 0.3, 'y')

classifier = choose_classifier("1")

bank_pie(X_test, y_test, 'yes', 'no', 'actual data')

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)
bank_pie(X_test, pred1, 'yes', 'no', 'unaltered data')

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'age_adult', 'yes')

X_test_mod = X_test.copy()

# critical_region_test('age_elder', 'age_adult', 'yes', 0, 0.2)
attribute_swap_test('age_adult', 'age_elder', 'yes')
critical_region_test(X_test, y_test, classifier, 'age_elder', 'age_adult', 'no', 'yes', 0, 0.20, bank_pie)
attribute_swap_and_critical('age_elder', 'age_adult', 'yes', 0, 0.1)

# calculate_mertics(y_test, pred1, X_test, 'age_young', 'yes')
# calculate_mertics(y_test, pred1, X_test, 'age_adult', 'yes')
# calculate_mertics(y_test, pred1, X_test, 'age_elder', 'yes')


# accuracy = accuracy_score(y_test, pred1)
# print("Accuracy of the classifier:", accuracy)

# print(len(pred1))
# print("Prediction: yes", (pred1 == 'yes').sum())
# print("Prediction: no", (pred1 == 'no').sum(), "\n")
# print("Age: age_young", (X_train['age_young']).sum())
# print("Age: age_adult", (X_train['age_adult']).sum())
# print("Age: age_elder", (X_train['age_elder']).sum())
