import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import Utilities


def disparate_impact_ratio(y_pred, sensitive_features):
    # Calculate the selection rate for each group
    selection_rates = {}
    # print(sensitive_features.unique())
    groups = sensitive_features.unique()
    print(groups)
    for group in groups:
        mask = (sensitive_features == group)
        selection_rates[group] = sum(y_pred[mask]) / sum(mask)
        print(sum(y_pred[mask]), group)

    # Calculate the disparate impact ratio
    ratio = selection_rates[groups[1]] / selection_rates[groups[0]]
    print(selection_rates[groups[1]])
    print(selection_rates[groups[0]])
    return ratio


def predict(features, label, decision):
    # print(features['sex'])

    classifier = Utilities.choose_classifier(decision)
    y_pred = classifier.predict(features)

    print()
    print(classifier)

    # Calculate the disparate impact
    di = disparate_impact_ratio(y_pred, sensitive_features=features['sex'])
    print("Disparate Impact Ratio:", di)

    # Calculate the accuracy
    accuracy = accuracy_score(label, y_pred)
    print("Accuracy of the classifier:", accuracy)

    # Perform 10-fold cross-validation
    # cross_val_scores = cross_val_score(classifier, X, y, cv=10)
    # Print the cross-validation scores
    # print("Cross-validation scores:", cross_val_scores, "\n")

    return y_pred


# Load the CSV file into a DataFrame
X = pd.read_csv('normalized_adult_features.csv')
y = pd.read_csv('adult_y.csv')
y = y['income'].to_numpy()

male_tag = X.iloc[0]['sex']
female_tag = X.iloc[4]['sex']
poor_tag = y[0]
rich_tag = y[9]

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

classifier_choice = "1"

print("\nStandard results ")

pred1 = predict(X_test, y_test, classifier_choice)

X_test.loc[:, 'sex'] = X['sex'].replace({male_tag: female_tag, female_tag: male_tag})

print("\nResults when switching the protected values.")

pred2 = predict(X_test, y_test, classifier_choice)

print(f"\nDifferent predictions: {sum(pred1 != pred2)}")

# export(X)

# show_metrics()
