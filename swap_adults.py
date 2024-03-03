import numpy as np
from fairlearn.datasets import fetch_adult
from matplotlib import pyplot as plt

from Utilities import calculate_mertics, choose_classifier, show_metrics_adults, adult_pie, \
    preprocess_data, partitioning


def final_prediction(pred1, pred2, x_test, fav, unfav, unpriv):
    final_predictions = np.array([None] * len(pred2))

    counter = 0
    for value1, value2, (_, row) in zip(pred1, pred2, x_test.iterrows()):

        if value1 == value2:
            final_predictions[counter] = value1

        elif value1 == fav != value2 and row[unpriv] == True:
            final_predictions[counter] = value2

        elif value1 == unfav != value2 and row[unpriv] == True:
            final_predictions[counter] = value1

        elif value1 == fav != value2 and row[unpriv] == False:
            final_predictions[counter] = value1

        elif value1 == unfav != value2 and row[unpriv] == False:
            final_predictions[counter] = value2
        counter = counter + 1

    return final_predictions


def attribute_swap_test(unpriv, priv, fav, unfav):
    # X_test[unpriv] = ~X_test[unpriv]
    # X_test[priv] = ~X_test[priv]

    X_test_mod[unpriv] = ~X_test_mod[unpriv]
    X_test_mod[priv] = ~X_test_mod[priv]

    pred2 = classifier.predict(X_test_mod)

    print("\nResults after swapping the protected attribute values.\n")

    # calculate_mertics(y_test, pred2, X_test, priv, fav)
    calculate_mertics(y_test, pred2, X_test_mod, priv, fav)

    # print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

    # adult_pie(X_test, pred2, '>50K', '<=50K', 'pred2 data')
    adult_pie(X_test_mod, pred2, '>50K', '<=50K', 'swapped values for protected attribute')


'''
    final = final_prediction(pred1, pred2, X_test, fav, unfav, unpriv)

    calculate_mertics(y_test, final, X_test, priv, fav)

    print(f"\nDifferent predictions: {sum(final != pred2)}\n")

'''


def only_priv_attribute(unpriv, priv, fav, unfav):
    X_test_mod[unpriv] = False
    X_test_mod[priv] = True

    pred2 = classifier.predict(X_test_mod)

    # adult_pie(X_test, pred2, fav, unfav, "Second prediction distribution")

    print("\nResults after turning protected values into the privilegded group.\n")

    calculate_mertics(y_test, pred2, X_test, unpriv, fav)

    print(f"\nDifferent predictions: {sum(pred1 != pred2)}")


def only_unpriv_attribute(unpriv, priv, fav, unfav):
    X_test_mod[unpriv] = True
    X_test_mod[priv] = False

    pred2 = classifier.predict(X_test_mod)

    # adult_pie(X_test, pred2, fav, unfav, "Second prediction distribution")

    print("\nResults after turning protected values into the unprivilegded group.\n")

    calculate_mertics(y_test, pred2, X_test, unpriv, fav)

    print(f"\nDifferent predictions: {sum(pred1 != pred2)}")


def attribute_swap_and_critical(unpriv, priv, fav, lower_bound, upper_bound):
    pred3 = classifier.predict(X_test_mod)

    indexes = partitioning(lower_bound, upper_bound, classifier.predict_proba(X_test_mod))
    feature_part = X_test_mod.iloc[indexes]

    for iteration_number, (index, row) in enumerate(feature_part.iterrows(), start=0):

        if row['sex_Female']:
            pred3[indexes[iteration_number]] = '<=50K'

        elif row['sex_Male']:
            pred3[indexes[iteration_number]] = '>50K'

    print("\nResults after clearing critical region.\n")

    calculate_mertics(y_test, pred3, X_test_mod, priv, fav)

    adult_pie(X_test_mod, pred3, '>50K', '<=50K', 'swapped values and critical region')


def critical_region_test(unpriv, priv, fav, lower_bound, upper_bound, ):
    pred4 = classifier.predict(X_test)

    indexes = partitioning(lower_bound, upper_bound, classifier.predict_proba(X_test))
    feature_part = X_test.iloc[indexes]

    for iteration_number, (index, row) in enumerate(feature_part.iterrows(), start=0):

        if row['sex_Female']:
            pred4[indexes[iteration_number]] = '>50K'

        elif row['sex_Male']:
            pred4[indexes[iteration_number]] = '<=50K'

    print("\nResults for critical region alone.\n")

    calculate_mertics(y_test, pred4, X_test, unpriv, fav)

    adult_pie(X_test, pred4, '>50K', '<=50K', 'critical region test')


data = fetch_adult(as_frame=True)

dataframe = data.frame

X_train, X_test, y_train, y_test = preprocess_data(dataframe, 0.3, 'class')

classifier = choose_classifier("5")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

adult_pie(X_test, y_test, '>50K', '<=50K', 'actual data')
adult_pie(X_test, pred1, '>50K', '<=50K', 'unaltered results')

print()
print(classifier)

calculate_mertics(y_test, pred1, X_test, 'sex_Female', '>50K')

X_test_mod = X_test.copy()

attribute_swap_test('sex_Female', 'sex_Male', '>50K', '<=50K')
# only_priv_attribute('sex_Female', 'sex_Male', '>50K', '<=50K')
# only_unpriv_attribute('sex_Female', 'sex_Male', '>50K', '<=50K')
attribute_swap_and_critical('sex_Female', 'sex_Male', '>50K', 0, 0.35)
critical_region_test('sex_Female', 'sex_Male', '>50K', 0, 0.60)

# show_metrics_adults(classifier)
