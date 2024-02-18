import numpy as np
from fairlearn.datasets import fetch_adult

from Utilities import calculate_mertics, preprocess_adults, choose_classifier, show_metrics_adults, adult_pie


def final_prediction(pred1, pred2, x_test, fav, unfav, unpriv):
    final_predictions = np.array([None] * len(pred2))

    # print(final_predictions)

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

    # print(final_predictions)
    return final_predictions


def attribute_swap_test(unpriv, priv, fav, unfav):

    # X_test[unpriv] = ~X_test[unpriv]
    # X_test[priv] = ~X_test[priv]

    X_test_mod[unpriv] = ~X_test_mod[unpriv]
    X_test_mod[priv] = ~X_test_mod[priv]

    pred2 = classifier.predict(X_test_mod)

    print("\nResults after swapping the protected attribute values.\n")

    calculate_mertics(y_test, pred2, X_test_mod, priv, fav)

    print(f"\nDifferent predictions: {sum(pred1 != pred2)}")


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


data = fetch_adult(as_frame=True)

dataframe = data.frame

X_train, X_test, y_train, y_test = preprocess_adults(dataframe, 0.3)

classifier = choose_classifier("1")

classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

print()
print(classifier)

calculate_mertics(y_test, pred1, X_test, 'sex_Female', '>50K')

X_test_mod = X_test.copy()

attribute_swap_test('sex_Female', 'sex_Male', '>50K', '<=50K')
only_priv_attribute('sex_Female', 'sex_Male', '>50K', '<=50K')
only_unpriv_attribute('sex_Female', 'sex_Male', '>50K', '<=50K')

# show_metrics_adults(classifier)
