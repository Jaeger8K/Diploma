import numpy as np
from fairlearn.datasets import fetch_adult
from Utilities import calculate_metrics, choose_classifier, preprocess_data, partitioning, attribute_swap_test, \
    adult_pie, critical_region_test, attribute_swap_and_critical

'''
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


def only_priv_attribute(unpriv, priv, fav, unfav):
    X_test_mod[unpriv] = False
    X_test_mod[priv] = True

    pred2 = classifier.predict(X_test_mod)

    # adult_pie(X_test, pred2, fav, unfav, "Second prediction distribution")

    print("\nResults after turning protected values into the privilegded group.\n")

    calculate_metrics(y_test, pred2, X_test, unpriv, fav)

    print(f"\nDifferent predictions: {sum(pred1 != pred2)}")


def only_unpriv_attribute(unpriv, priv, fav, unfav):
    X_test_mod[unpriv] = True
    X_test_mod[priv] = False

    pred2 = classifier.predict(X_test_mod)

    # adult_pie(X_test, pred2, fav, unfav, "Second prediction distribution")

    print("\nResults after turning protected values into the unprivilegded group.\n")

    calculate_metrics(y_test, pred2, X_test, unpriv, fav)

    print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

'''

data = fetch_adult(as_frame=True)

dataframe = data.frame

X_train, X_test, y_train, y_test = preprocess_data(dataframe, 0.3, 'class')

classifier = choose_classifier("2")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

adult_pie(X_test, y_test, '>50K', '<=50K', 'actual data')
adult_pie(X_test, pred1, '>50K', '<=50K', 'unaltered results')

print()
print(classifier)

calculate_metrics(y_test, pred1, X_test, 'sex_Male', '>50K')

# X_test_mod = X_test.copy()

attribute_swap_test(X_test, y_test, classifier, 'sex_Male', 'sex_Female', '<=50K', '>50K', adult_pie)
critical_region_test(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, 0.21, adult_pie)
attribute_swap_and_critical(X_test, y_test, classifier, 'sex_Female', 'sex_Male', '<=50K', '>50K', 0, 0.15, adult_pie)
# critical_region_test('sex_Female', 'sex_Male', '<=50K', '>50K', 0, 0.10, adult_pie)
# attribute_swap_and_critical('sex_Female', 'sex_Male', '>50K', 0, 0.35)


# show_metrics_adults(classifier)
