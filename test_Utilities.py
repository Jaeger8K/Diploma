import sys
import multiprocessing
from functools import partial


def process_iteration(i, x, y, x_c, y_c, classifier, c_classifier, k_fold, critical_region_test, unpriv, priv, unfav, fav, done_1, done_2):
    acc = DIR = samp = 0
    acc_c = DIR_c = samp_c = 0

    for (train_indices, test_indices), (train_indices_c, test_indices_c) in zip(k_fold.split(x), k_fold.split(x_c)):
        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        x_train_c, x_test_c = x_c.iloc[train_indices], x_c.iloc[test_indices]
        y_train_c, y_test_c = y_c.iloc[train_indices], y_c.iloc[test_indices]

        if done_1 == 0:
            classifier.fit(x_train, y_train)
            acc_temp, DIR_temp, samp_temp = critical_region_test(x_test, y_test, classifier, unpriv, priv, unfav, fav, 0, i / 100, None)
            acc += acc_temp
            DIR += DIR_temp
            samp += samp_temp

        if done_2 == 0:
            c_classifier.fit(x_train_c, y_train_c)
            acc_temp, DIR_temp, samp_temp = critical_region_test(x_test, y_test, c_classifier, unpriv, priv, unfav, fav, 0, i / 100, None)
            acc_c += acc_temp
            DIR_c += DIR_temp
            samp_c += samp_temp

    return i, acc, DIR, samp, acc_c, DIR_c, samp_c


if __name__ == '__main__':
    num_iterations = int(sys.argv[3])  # Assuming the third argument is the number of iterations

    # Initialize necessary variables and lists
    ROC_accuracy = []
    ROC_DIR = []
    ROC_samples = []
    CROC_accuracy = []
    CROC_DIR = []
    CROC_samples = []
    l_values = []

    done_1 = done_2 = 0  # Initial condition flags

    # Initialize classifiers, k_fold, and critical_region_test function appropriately

    # Set up multiprocessing pool
    pool = multiprocessing.Pool()

    try:
        # Define a partial function for easier parallel processing
        partial_iteration = partial(process_iteration, x=x, y=y, x_c=x_c, y_c=y_c,
                                    classifier=classifier, c_classifier=c_classifier,
                                    k_fold=k_fold, critical_region_test=critical_region_test,
                                    unpriv=unpriv, priv=priv, unfav=unfav, fav=fav,
                                    done_1=done_1, done_2=done_2)

        # Map the function over the range of iterations
        results = pool.map(partial_iteration, range(1, num_iterations))

        # Process results
        for result in results:
            i, acc, DIR, samp, acc_c, DIR_c, samp_c = result

            if done_1 == 0:
                ROC_accuracy.append(acc / 10)
                ROC_DIR.append(DIR / 10)
                ROC_samples.append(samp / 10)
                if DIR / 10 > 1.0:
                    done_1 = 1

            if done_2 == 0:
                CROC_accuracy.append(acc_c / 10)
                CROC_DIR.append(DIR_c / 10)
                CROC_samples.append(samp_c / 10)
                if DIR_c / 10 > 1.0:
                    done_2 = 1

            l_values.append(i / 100)

            if done_1 == 1 and done_2 == 1:
                break

    finally:
        pool.close()
        pool.join()
