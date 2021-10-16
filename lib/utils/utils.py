from lib.utils.math_utils import cartesian
from lib.types.types import *
from typing import List, Tuple, Callable, Dict, Any
from sklearn.metrics import classification_report
import numpy as np
from simple_chalk import chalk
from tqdm import tqdm

np.set_printoptions(linewidth=200)

CrossValidationMean = Dict[str, Any]


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
        model: Callable[[Any], LearningModel],
        x: np.ndarray,
        y: np.ndarray,
        cross_validator: Callable[
            [np.ndarray, np.ndarray, int, LearningModel, np.ndarray, np.ndarray], CrossValidationMean],
        verbose=True,
        method='accuracy',
        val_x: np.ndarray = None,
        val_y: np.ndarray = None,
) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """
    Use a grid search to search for
    all possible combinations in the
    parameters' space, and select the
    best one given the f1 score from
    the validator.
    :param method:
    :param verbose:
    :param model_parameters: A defined dict of possible model parameters.
    :param model: A model constructor
    :param x: X
    :param y: y
    :param val_x: val_x
    :param val_y: val_y
    :param cross_validator: A cross validation function.
    :return: Selected best combination, and a list of tuples containing a combination and its corresponding result
    """
    if verbose:
        print(f'{chalk.bold("-" * 15 + "START FINDING BEST PARAMETERS" + "-" * 15)}\n')

    model_parameter_keys = [*model_parameters.keys()]
    all_combinations = cartesian([model_parameters[key] for key in model_parameter_keys])

    best_combination = dict()
    best_macro_f1_score = 0
    best_weighted_f1_score = 0
    best_accuracy = 0
    results = []

    for combination in tqdm(all_combinations, leave=True):
        combination_input = dict()
        for key_index in range(len(combination)):
            combination_input[model_parameter_keys[key_index]] = combination[key_index]
        m = model(**combination_input)
        result = cross_validator(x, y, 5, m, val_x, val_y)

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


def cross_validate(
        x: np.ndarray,
        y: np.ndarray,
        n_fold: int,
        model: LearningModel,
        val_x: np.ndarray = None,
        val_y: np.ndarray = None,
) -> CrossValidationMean:
    if val_x is not None and val_y is not None:
        return cross_validate_with_val_data(x, y, model, val_x, val_y)
    else:
        return cross_validate_with_n_fold(x, y, n_fold, model)


def cross_validate_with_val_data(
        x: np.ndarray,
        y: np.ndarray,
        model: LearningModel,
        val_x: np.ndarray = None,
        val_y: np.ndarray = None,
) -> CrossValidationMean:
    _, number_of_iterations = model.fit(x, y)
    training_predictions = model.predict(x)
    training_report = classification_report(y.astype(int), training_predictions.astype(int),
                                            output_dict=True, zero_division=0)

    validation_predictions = model.predict(val_x)
    validation_report = classification_report(val_y.astype(int), validation_predictions.astype(int),
                                              output_dict=True, zero_division=0)

    return {
        "training macro f1": training_report["macro avg"]["f1-score"],
        "training weighted f1": training_report["weighted avg"]["f1-score"],
        "training accuracy": training_report["accuracy"],
        "number_of_iterations_to_converge": number_of_iterations,
        "macro f1": validation_report["macro avg"]["f1-score"],
        "weighted f1": validation_report["weighted avg"]["f1-score"],
        "accuracy": validation_report["accuracy"],
    }


def cross_validate_with_n_fold(
        x: np.ndarray,
        y: np.ndarray,
        n_fold: int,
        model: LearningModel,
) -> CrossValidationMean:
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
        model.fit(training_set[:, :-1], training_set[:, -1].astype(int))

        training_predictions = model.predict(training_set[:, :-1])
        training_report = classification_report(training_set[:, -1].astype(int), training_predictions.astype(int),
                                                output_dict=True, zero_division=0)

        validation_predictions = model.predict(validation_set[:, :-1])
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


def calculate_inaccuracy(x: np.ndarray, y: np.ndarray, model: LearningModel) -> float:
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
