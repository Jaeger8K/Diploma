import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import pandas as pd
import csv


def choose_classifier(model_selection):
    m = 0

    if model_selection == "LogisticRegression":  # works
        m = LogisticRegression(random_state=16)
        m.fit(X_train, y_train)
    elif model_selection == "RandomForest":  # works
        m = RandomForestClassifier(max_depth=2, random_state=0)
        m.fit(X_train, y_train)
    elif model_selection == "GaussianNB":  # works
        m = GaussianNB()
        m = m.fit(X_train, y_train)
    elif model_selection == "Kmeans":  # needs work, 'KMeans' object has no attribute 'predict_proba'
        m = KMeans(n_clusters=2, random_state=0, n_init="auto")
        m.fit(X_train)
    elif model_selection == "LinearSVC":  # works 'LinearSVC' object has no attribute 'predict_proba'. Did you mean: '_predict_proba_lr'?
        m = LinearSVC(dual='auto')
        m.fit(X_train, y_train)
    elif model_selection == "MLPClassifier":  # works, needs more iterations, takes time
        m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=350)
        m.fit(X_train, y_train)
    elif model_selection == "SVM":  # needs work, doesnt terminate
        m = svm.SVC()
        m.fit(X_train, y_train)

    return m, m.predict(X_test)


def show_metrics(y_test, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy
    print(f'Accuracy: {accuracy}', "\n")

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sn.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    target_names = ['above 50k', 'below 50k']
    print(classification_report(y_test, y_pred, target_names=target_names))


def partition_results(lower_bound, upper_bound):
    indexes = [idx for idx, probabilities in enumerate(y_pred_proba) if
               lower_bound < np.abs(probabilities[0] - probabilities[1]) < upper_bound]
    prob = y_pred_proba[indexes]
    true_label = y_test[indexes]
    pred = y_pred[indexes]
    features = X_test.iloc[indexes]

    print(f"Samples with {lower_bound} < probability difference < {upper_bound} :", len(indexes))
    print(f"Men: {len(features[features['sex'] == male_tag])} Women: {len(features[features['sex'] == female_tag])}\n")

    return prob, true_label, pred, features


def show_probabilities(prob, new_prob):
    print("Old probabilities\n")
    count = 0
    for element in prob:
        print(element, end=" ")
        count += 1
        if count % 6 == 0:
            print()  # Change row after printing 6 elements
        if count == 50:
            break
    print()

    print("New probabilities\n")
    count = 0
    for element in new_prob:
        print(element, end=" ")
        count += 1
        if count % 6 == 0:
            print()  # Change row after printing 6 elements
        if count == 50:
            break
    print()


def analyse_results(pred, prob, new_pred, new_prob, true_label):
    # show_probabilities(prob, new_prob)

    equal = 0
    first = 0
    second = 0
    for elem1, elem2, elem3 in zip(pred, new_pred, true_label):
        if elem1 == elem2 == elem3:
            equal += 1
        elif elem1 == elem3:
            first += 1
        elif elem2 == elem3:
            second += 1

    print("Number of times first prediction is correct:", first)
    print("Number of times second prediction is correct:", second)
    print("Number of times both predictions are correct:", equal)


def extract_predictions():
    prob, true_label, pred, features = partition_results(0.2, 0.3)

    features.loc[:, 'sex'] = features['sex'].replace({male_tag: female_tag, female_tag: male_tag})

    # Calculate the new probabilities
    new_prob = model.predict_proba(features)

    data = []

    for i in range(len(pred)):
        data.append([prob[i][0], prob[i][1], new_prob[i][0], new_prob[i][1], true_label[i]])

    # Define the CSV filename
    csv_filename = 'test.csv'

    # Write the data to a CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header row
        csvwriter.writerow(['Prob_0', 'Prob_1', 'New_Prob_0', 'New_Prob_1', 'True_Label'])

        # Write the data rows
        csvwriter.writerows(data)

    print(f"Data written to {csv_filename}")


def information():
    print("\nDataset Information")

    print(X)
    print(y, len(y))

    print(
        f"\nGeneral information\n\nMale:{int(male_tag)}  Female:{int(female_tag)}\n<=50K:{poor_tag} >50K:{rich_tag}\n")
    print(model)
    # Perform 10-fold cross-validation
    cross_val_scores = cross_val_score(model, X, y, cv=10)

    # Print the cross-validation scores
    print("Cross-validation scores:", cross_val_scores, "\n")


def counterfactual(lower_bound, upper_bound):
    prob, true_label, pred, features = partition_results(lower_bound, upper_bound)

    features.loc[:, 'sex'] = features['sex'].replace({male_tag: female_tag, female_tag: male_tag})

    # Calculate the new predictions
    new_pred = model.predict(features)

    # Calculate the new probabilities
    new_prob = model.predict_proba(features)

    print(f"Samples with pred != true_labels: {sum(pred != true_label)}")
    print(f"Samples with new_pred != true_labels: {sum(new_pred != true_label)}")
    print(f"Samples with pred != new_pred: {sum(pred != new_pred)}\n")

    analyse_results(pred, prob, new_pred, new_prob, true_label)


def analysis():
    print("Analysis\n")
    print("Classifier: LogisticRegression")
    print(f"Test size: {test_size}")
    print(f"Total test subjects: {len(X_test)}")
    print(f"Total men: {sum((X_test['sex'] == male_tag))} Total women: {sum((X_test['sex'] == female_tag))}")
    print(f"Income<=50K: {sum((y_test == poor_tag))} Income>50K: {sum((y_test == rich_tag))}\n")

    print(f"Total false results: {sum(y_test != y_pred)}")
    print(f"False results for men: {sum((X_test['sex'] == male_tag) & (y_test != y_pred))}")
    print(f"False results for women: {sum((X_test['sex'] == female_tag) & (y_test != y_pred))}\n")

    margins = [i / 10 for i in range(1, 11)]
    test = model.predict_proba(X_test[(y_test != y_pred)])

    count_1 = []
    count_2 = []

    for margin in margins:
        count = sum(abs(y_pred_proba[:, 0] - y_pred_proba[:, 1]) < margin)
        count_1.append(sum(abs(y_pred_proba[:, 0] - y_pred_proba[:, 1]) < margin))
        print(f"Number of results with probability margin < {margin}: {count}")

    print("\n")

    for margin in margins:
        count = sum(abs(test[:, 0] - test[:, 1]) < margin)
        count_2.append(sum(abs(test[:, 0] - test[:, 1]) < margin))
        print(f"Number of false results with probability margin < {margin}: {count}")

    print("\n")

    percentages = [a / b for a, b in zip(count_2, count_1)]

    for margin in margins:
        print(f"False results / total results with probability margin < {margin}: {percentages[int(margin * 10 - 1)]}")

    print("\n")
    false_percentages = [x / sum(y_test != y_pred) for x in count_2]

    for margin in margins:
        print(
            f"False results / total false results with probability margin < {margin}: {false_percentages[int(margin * 10 - 1)]}")

    # Plotting
    plt.plot(margins, count_1, marker='o', linestyle='-', color='b')
    plt.title(f'Number of results vs. probability margin test size:{test_size}')
    plt.xlabel('probability margin')
    plt.ylabel('results')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.plot(margins, count_2, marker='o', linestyle='-', color='b')
    plt.title(f'Number of false results vs. probability margin test size:{test_size}')
    plt.xlabel('Number of false results')
    plt.ylabel('Counts')
    plt.grid(True)
    plt.show()

    # Plotting both percentages on the same graph
    plt.plot(margins, percentages, marker='o', linestyle='-', color='b', label='false results/ total results')
    plt.plot(margins, false_percentages, marker='o', linestyle='-', color='r',
             label='false results/ total false results')
    plt.title(f'Percentage of False Results vs. Total Results vs. Probability Margin   test size:{test_size}')
    plt.xlabel('Probability Margin')
    plt.ylabel('Percentage')
    plt.legend()  # Display legend to differentiate between Total Results and False Results
    plt.grid(True)
    plt.show()


'''
    # Plotting
    plt.plot(margins, percentages, marker='o', linestyle='-', color='b')
    plt.title('percentage of false results/ total results vs. probability margin')
    plt.xlabel('probability margin')
    plt.ylabel('percentage')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.plot(margins, false_percentages, marker='o', linestyle='-', color='b')
    plt.title('percentage of false results/ total false results vs. probability margin')
    plt.xlabel('probability margin')
    plt.ylabel('percentage')
    plt.grid(True)
    plt.show()
'''

# Load the CSV file into a DataFrame
X = pd.read_csv('normalized_features.csv')
y = pd.read_csv('y.csv')
y = y['income'].ravel()

male_tag = X.iloc[0]['sex']
female_tag = X.iloc[4]['sex']
poor_tag = y[0]
rich_tag = y[9]

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

model, y_pred = choose_classifier('LogisticRegression')

information()

y_pred_proba = model.predict_proba(X_test)

# show_metrics(y_test, y_pred)

# analysis()

counterfactual(0.0, 0.1)

# extract_predictions()
