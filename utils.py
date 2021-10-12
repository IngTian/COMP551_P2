import math
from typing import List, Tuple, Callable, Dict, Any, Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from simple_chalk import chalk

np.set_printoptions(linewidth=200)

ScikitLearnModel = Union[DecisionTreeClassifier, KNeighborsClassifier]

CrossValidationMean = Dict[str, Any]


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=object)
    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        temp = cartesian(arrays[1:])
        if (len(temp.shape) == 1):
            temp = temp[:, np.newaxis]
        for start_index in [k * m for k in range(arrays[0].size)]:
            out[start_index:start_index + m, 1:] = temp
    return out


def read_data(path: str, columns: List[str], verbose=True) -> pd.DataFrame:
    """
    Read a csv file by the given path.

    :param path: This is the path.
    :param columns: The data columns.
    :param verbose: Decide whether to print logs.
    :return: A Pandas DataFrame object.
    """
    df = pd.read_csv(path)
    df.columns = columns

    if verbose:
        print(f'{chalk.bold.greenBright("The data set has successfully been loaded.")}\n'
              f'{chalk.bold("PATH: ")} {path}\n'
              f'{chalk.bold("-" * 15 + "PRINTING DATA PREVIEW" + "-" * 15)}\n'
              f'{df.head()}\n')

    return df


def preprocess_data(x: np.ndarray,
                    y: np.ndarray,
                    plugins: List[Callable[[np.ndarray, np.ndarray, bool], Tuple[np.ndarray, np.ndarray]]],
                    verbose=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-processing data by a series of plug-ins.
    :param x: Features, or raw data
    :param y: Labels, or placeholder
    :param plugins: A series of callable functions that process the data.
    :param verbose: Decide whether to print logs.
    :return: The first value is training X, whereas the second value is the label y.
    """
    number_of_instances, number_of_features = x.shape[0], x.shape[1]

    if verbose:
        print(f'{chalk.bold("-" * 15 + "STARTING PRE-PROCESSING DATA" + "-" * 15)}\n'
              f'{chalk.bold("TOTAL ENTRIES:  ")} {number_of_instances}\n'
              f'{chalk.bold("TOTAL FEATURES: ")} {number_of_features}\n')

    for plugin in plugins:
        x, y = plugin(x, y, verbose)

    return x, y


def get_best_model_parameter(
        model_parameters: Dict[str, List[Any]],
        model: Callable[[Any], ScikitLearnModel],
        x: np.ndarray,
        y: np.ndarray,
        cross_validator: Callable[[np.ndarray, np.ndarray, int, ScikitLearnModel], CrossValidationMean],
        verbose=True,
        method='f1'
) -> (Dict[str, Any], List):
    """
    Use a grid search to search for
    all possible combinations in the
    parameters' space, and select the
    best one given the f1 score from
    the validator.
    :param model_parameters: A defined dict of possible model parameters.
    :param model: A model constructor
    :param x: X
    :param y: y
    :param cross_validator: A cross validation function.
    :return: Selected best combination, and a list of tuples containing a combination and its corresponding result
    """

    model_parameter_keys = [*model_parameters.keys()]
    all_combinations = cartesian([model_parameters[key] for key in model_parameter_keys])

    best_combination = dict()
    best_macro_f1_score = 0
    best_weighted_f1_score = 0
    best_accuracy = 0
    results = []

    for combination in all_combinations:
        combination_input = dict()
        for key_index in range(len(combination)):
            combination_input[model_parameter_keys[key_index]] = combination[key_index]
        m = model(**combination_input)
        result = cross_validator(x, y, 5, m)

        results += [(combination_input, result)]

        if result['weighted f1'] > best_weighted_f1_score and method == 'weighted f1':
            best_combination = combination_input
            best_weighted_f1_score = result['weighted f1']
        elif result['macro f1'] > best_macro_f1_score and method == 'macro f1':
            best_combination = combination_input
            best_macro_f1_score = result['macro f1']
        elif result['accuracy'] > best_accuracy and method == 'accuracy':
            best_combination = combination_input
            best_accuracy = result['accuracy']

    if verbose:
        print(f'{chalk.bold("-" * 15 + "BEST PARAMETERS FOUND" + "-" * 15)}\n'
              f'{chalk.greenBright(best_combination)}\n')

    return best_combination, results


def cross_validate(x: np.ndarray, y: np.ndarray, n_fold: int, model: ScikitLearnModel) -> CrossValidationMean:
    """
    Implement a cross validate algorithm
    to check how the model performs.
    :param x: X
    :param y: y
    :param n_fold: N fold
    :param model: A model
    :return: The report.
    """
    complete_data = np.append(x, y[:, np.newaxis], axis=1)
    total_number_of_instances = complete_data.shape[0]
    bucket_size = total_number_of_instances // n_fold

    train_weighted_f1 = 0
    train_macro_f1 = 0
    train_accuracy = 0

    val_macro_f1 = 0
    val_weighted_f1 = 0
    val_accuracy = 0
    for i in range(n_fold):

        # Computing the validation set and training set.
        validation_set_start_index, validation_set_end_index = i * bucket_size, min(i * bucket_size + bucket_size,
                                                                                    total_number_of_instances) - 1
        validation_set = complete_data[validation_set_start_index:validation_set_end_index + 1]
        training_set = None
        if validation_set_start_index == 0:
            training_set = complete_data[validation_set_end_index + 1:]
        elif validation_set_end_index == total_number_of_instances - 1:
            training_set = complete_data[:validation_set_start_index]
        else:
            training_set = np.append(
                complete_data[:validation_set_start_index],
                complete_data[validation_set_end_index + 1:],
                axis=0
            )

        # Make Predictions
        fitted_model = model.fit(training_set[:, :-1], training_set[:, -1].astype(int))

        training_predictions = fitted_model.predict(training_set[:, :-1])
        training_report = classification_report(training_set[:, -1].astype(int), training_predictions.astype(int),
                                                output_dict=True, zero_division=0)

        validation_predictions = fitted_model.predict(validation_set[:, :-1])
        validation_report = classification_report(validation_set[:, -1].astype(int), validation_predictions.astype(int),
                                                  output_dict=True, zero_division=0)

        train_macro_f1 += training_report["macro avg"]["f1-score"]
        train_weighted_f1 += training_report["weighted avg"]["f1-score"]
        train_accuracy += training_report["accuracy"]

        val_macro_f1 += validation_report["macro avg"]["f1-score"]
        val_weighted_f1 += validation_report["weighted avg"]["f1-score"]
        val_accuracy += validation_report["accuracy"]

    return {
        "training macro f1": train_macro_f1 / n_fold,
        "training weighted f1": train_weighted_f1 / n_fold,
        "training accuracy": train_accuracy / n_fold,

        "accuracy": val_accuracy / n_fold,
        "macro f1": val_macro_f1 / n_fold,
        "weighted f1": val_weighted_f1 / n_fold,
    }


def calculate_inaccuracy(x: np.ndarray, y: np.ndarray, model: ScikitLearnModel) -> float:
    """
    Calculate inaccuracies.
    :param x: X
    :param y: y
    :param model: A scikit model
    :return: Inaccuracy rate
    """
    trial_predict = model.predict(x)
    number_of_instances = y.shape[0]
    return np.count_nonzero(np.not_equal(trial_predict, y)) / number_of_instances
