from sklearn.metrics import accuracy_score, precision_score


def calculate_metrics(y_test, y_pred, x_test, priv, fav_out):
    """
    A function used for calculating the accuracy, precision, recall and Disparate Impact
    Ratio of a classifier.
    :param y_test: it holds the actual class labels of the test split
    :param y_test: it holds predicted class labels of the test split
    :param x_test: it holds attributes of the test split
    :param priv: it holds the label of the privileged group
    :param fav_out: it holds the label of the favourable outcome
    :return: the function returns the values of the Disparate Impact Ratio and accuracy of the classifier
    """

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{COLORS.GREEN}Accuracy of the classifier: {accuracy}{COLORS.ENDC}")

    # Calculate precision
    precision = precision_score(y_test, y_pred, pos_label=fav_out)
    print(f"{COLORS.BLUE}Precision of the classifier: {precision}{COLORS.ENDC}")

    # Calculate recall
    recall = recall_score(y_test, y_pred, pos_label=fav_out)
    print(f"{COLORS.YELLOW}Recall of the classifier: {recall}{COLORS.ENDC}")

    # a = ((y_pred == fav_out) & (x_test[priv] == False)).sum()
    # b = ((y_pred == fav_out) & (x_test[priv] == True)).sum()
    a = ((y_pred == fav_out) & (x_test[priv] == False)).sum() / (x_test[priv] == False).sum()
    b = ((y_pred == fav_out) & (x_test[priv] == True)).sum() / (x_test[priv] == True).sum()
    disparate_impact = a / b
    print(f"{COLORS.HEADER}Disparate Impact Ratio: {disparate_impact}{COLORS.ENDC}")
    # print(a, b)

    return accuracy, disparate_impact