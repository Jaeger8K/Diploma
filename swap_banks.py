import pandas as pd
from Utilities import choose_classifier, preprocess_data, handle_age, calculate_mertics, bank_pie


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

    # X_test_mod[unpriv] = ~X_test_mod[unpriv]
    # X_test_mod[priv] = ~X_test_mod[priv]

    pred2 = classifier.predict(X_test_mod)

    print("\nResults after swapping the protected attribute values.\n")

    calculate_mertics(y_test, pred2, X_test_mod, priv, fav)

    print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

    bank_pie(X_test, pred2, 'yes', 'no', 'pred2 data')


data = pd.read_csv('Datasets/bank_dataset.csv')
data = data.dropna()

handle_age(data, 'age', 30, 60)

X_train, X_test, y_train, y_test = preprocess_data(data, 0.3, 'y')

classifier = choose_classifier("2")

bank_pie(X_test, y_test, 'yes', 'no', 'Actual data')

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)
bank_pie(X_test, pred1, 'yes', 'no', 'pred1 data')

print()
print(classifier)

calculate_mertics(y_test, pred1, X_test, 'age_elder', 'yes')

X_test_mod = X_test.copy()

attribute_swap_test('age_elder', 'age_adult', 'yes')

# calculate_mertics(y_test, pred1, X_test, 'age_young', 'yes')
# calculate_mertics(y_test, pred1, X_test, 'age_adult', 'yes')
# calculate_mertics(y_test, pred1, X_test, 'age_elder', 'yes')

'''
accuracy = accuracy_score(y_test, pred1)
print("Accuracy of the classifier:", accuracy)

print(len(pred1))
print((pred1 == 'yes').sum())
print((pred1 == 'no').sum(), "\n")
print((X_train['age_young']).sum())
print((X_train['age_adult']).sum())
print((X_train['age_elder']).sum())
'''
