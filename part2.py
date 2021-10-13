from lib.utils.io_utils import read_csv
from lib.utils.utils import get_best_model_parameter, cross_validate
from lib.model.gd import UpdateWeightMethod
from lib.model.gd import LogisticRegression
import pprint as pp
from simple_chalk import chalk
from typing import List, Tuple, Callable, Dict, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import numpy as np
import matplotlib.pyplot as plt
import pathlib as path
import seaborn as sns


np.set_printoptions(linewidth=200)
ScikitLearnModel = Union[LogisticRegression]
CrossValidationMean = Dict[str, Any]

# region Global Settings
TRAINING_DATA_PATH = 'data_sets/data_A2/fake_news/fake_news_train.csv'
TRAINING_DATA_PATH = 'data_sets/data_A2/fake_news/fake_news_train.csv'
TRAINING_DATA_PATH = 'data_sets/data_A2/fake_news/fake_news_train.csv'
MODEL_PARAMETERS = {
    'max_iter': 500
}
BEST_MAX_FEATURES = None


# endregion

# region I/O Utils

def read_data(path_to_file: str, verbose=True) -> np.ndarray:
    """
    Read the file and return a numpy ndarray.

    :param path_to_file: This is the path.
    :param verbose: Decide whether to print logs.
    :return: A numpy ndarray.
    """
    result_data = np.loadtxt(path_to_file, dtype=str, delimiter='\n', encoding='latin1')

    if verbose:
        print(f'{chalk.bold("-" * 15 + "SUCCESSFULLY LOADED DATA" + "-" * 15)}\n')
        print(f'{chalk.bold.greenBright("SHAPE:")} {result_data.shape}\n')
        print(f'{chalk.bold("-" * 15 + "SHOWING THE FIRST 5 ROWS OF DATA" + "-" * 15)}\n')
        print(f'{result_data[:5]}\n')

    return result_data


# endregion

# region Preprocess Data
def preprocess_data(
        x: np.ndarray,
        y: np.ndarray,
        plugins: List[Callable[[np.ndarray, np.ndarray, bool], Tuple[np.ndarray, np.ndarray]]],
        verbose=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-processing data by a series of plug-ins.
    :param x: Features, or raw data
    :param y: Labels, or placeholder
    :param plugins: A series of callable functions that process the data.
    :param verbose: Decide whether to print logs.
    :return: The first value is training X, whereas the second value is the label y.
    """
    number_of_instances, number_of_features = x.shape[0], x.shape[1] if len(x.shape) > 1 else 1

    if verbose:
        print(f'{chalk.bold("-" * 15 + "STARTING PRE-PROCESSING DATA" + "-" * 15)}\n'
              f'{chalk.bold("TOTAL ENTRIES:  ")} {number_of_instances}\n'
              f'{chalk.bold("TOTAL FEATURES: ")} {number_of_features}\n')

    for plugin in plugins:
        x, y = plugin(x, y, verbose)

    return x, y


def test_data_split(data_set: np.ndarray, test_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the data set into
    training set and test set.
    :param data_set: The original data set.
    :param test_ratio: The test set ratio to split.
    :return:
    """
    test_data_size = int(len(data_set) * test_ratio)
    return data_set[test_data_size:], data_set[:test_data_size]


def visualize_features_impact_via_scatter_plot(
        x: np.ndarray,
        y: np.ndarray,
        feature_column_names: List[str],
        size=500,
        folder_path: path.Path = None,
) -> None:
    """
    Display the relationship between
    different feature-label pairs.
    :param x: Features
    :param y: Labels
    :param feature_column_names: The name for each feature column
    :param size: Target size to include
    :param folder_path: Target path to save images.
    :return:
    """
    number_of_instances, number_of_features = x.shape
    for feature_index in range(number_of_features):
        plt.scatter(x[:size, feature_index], y[:size])
        plt.xlabel(f'Feature {feature_column_names[feature_index]}')
        plt.ylabel(f'Result')
        plt.title(f'Result Vs. {feature_column_names[feature_index]}')
        plt.show()
        if folder_path:
            plt.savefig(folder_path / f'{feature_column_names[feature_index]}.png')


def visualize_feature_distribution(
        x: np.ndarray,
        y: np.ndarray,
        features: List[str],
        complete_feature_labels: np.ndarray,
        save_folder: path.Path = None,
) -> None:
    """
    Visualize distributions for various
    features.
    :param x: Features
    :param y: Categories
    :param features: Features to Display
    :param complete_feature_labels: The complete feature names for each column of input
    :param save_folder: Target directory to save images.
    :return:
    """
    possible_categories = np.unique(y)

    for feature in features:
        target_feature_vector = x[:, complete_feature_labels == feature]
        for label in possible_categories:
            feature_with_the_target = target_feature_vector[y == label]
            sns.distplot(
                feature_with_the_target,
                hist=False,
                kde=True,
                kde_kws={'linewidth': 3},
                label=label
            )
        plt.legend(prop={'size': 16}, title='Category')
        plt.title(f'Distrubtion with feature {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.show()
        if save_folder:
            plt.savefig(str(save_folder / f'{feature}.png'))


# endregion

# region Preprocess Plugins
def preprocess_count_uni_gram_occurrence(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
        max_features: int = BEST_MAX_FEATURES
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(max_features=max_features)

    transformed_x = vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Unigram Occurrence")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')

    return transformed_x, y


def preprocess_count_uni_gram_occurrence_excluding_stop_words(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
        max_features: int = BEST_MAX_FEATURES
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'), max_features=max_features)
    transformed_x = vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Unigram Occurrence Excluding Stopwords")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


def preprocess_count_uni_gram_occurrence_with_stemming(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
        max_features: int = BEST_MAX_FEATURES
) -> Tuple[np.ndarray, np.ndarray]:
    stemming_helper = SnowballStemmer('english')
    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(x[:, 0])
    analyzer = vectorizer.build_analyzer()

    new_vectorizer = CountVectorizer(
        analyzer=lambda s: (stemming_helper.stem(word) for word in analyzer(s)),
        max_features=max_features
    )
    transformed_x = new_vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Unigram Occurrence with Stemming")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


def preprocess_count_uni_gram_occurrence_with_lemmatization(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
        max_features: int = BEST_MAX_FEATURES
) -> Tuple[np.ndarray, np.ndarray]:
    lemmatizer = WordNetLemmatizer()
    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(x[:, 0])
    analyzer = vectorizer.build_analyzer()

    new_vectorizer = CountVectorizer(
        analyzer=lambda s: (lemmatizer.lemmatize(word) for word in analyzer(s)),
        max_features=max_features
    )
    transformed_x = new_vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Unigram Occurrence with Lemmatization")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


def preprocess_count_bigram_occurrence(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
        max_features: int = BEST_MAX_FEATURES
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=max_features)
    transformed_x = vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Bigram Occurrence")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


def preprocess_count_bigram_occurrence_excluding_stop_words(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
        max_features: int = BEST_MAX_FEATURES
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(
        stop_words=stopwords.words('english'),
        ngram_range=(2, 2),
        max_features=max_features
    )
    transformed_x = vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Unigram Occurrence Excluding Stopwords")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


def preprocess_null_process(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool
) -> Tuple[np.ndarray, np.ndarray]:
    return np.ones(y.shape), y


def preprocess_uni_bi(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
        max_features: int = BEST_MAX_FEATURES
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=max_features)
    transformed_x = vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Bigram Occurrence")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


# endregion

# region Evaluation

def cross_validate(
        x: np.ndarray,
        y: np.ndarray,
        model: ScikitLearnModel,
        n_fold: int = 5,
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
    x = x[:, np.newaxis] if len(x.shape) == 1 else x
    y = y[:, np.newaxis] if len(y.shape) == 1 else y
    complete_data = np.append(x, y, axis=1)
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


def evaluate_preprocess_techniques(
        x: np.ndarray,
        y: np.ndarray,
        control_preprocess_functions: List[Callable[[np.ndarray, np.ndarray, bool], Tuple[np.ndarray, np.ndarray]]],
        experimental_preprocess_sets: Dict[
            str, List[Callable[[np.ndarray, np.ndarray, bool], Tuple[np.ndarray, np.ndarray]]]],
        model: Callable[[Any], ScikitLearnModel],
        model_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Comparison between two preprocess sets.
    :param x:
    :param y:
    :param control_preprocess_functions:
    :param experimental_preprocess_sets:
    :param model:
    :param model_parameters:
    :return:
    """

    result = {
        "control": dict(),
        "experiment": dict()
    }

    control_x, control_y = preprocess_data(x.copy(), y.copy(), control_preprocess_functions, False)
    result['control'] = cross_validate(control_x, control_y, model(**model_parameters))

    for experimental_preprocess_name in experimental_preprocess_sets:
        experimental_x, experimental_y = preprocess_data(
            x.copy(),
            y.copy(),
            experimental_preprocess_sets[experimental_preprocess_name],
            False
        )
        result['experiment'][experimental_preprocess_name] = cross_validate(
            experimental_x,
            experimental_y,
            model(**model_parameters)
        )

    return result


def evaluate_preprocess_performance_by_max_token(
        x: np.ndarray,
        y: np.ndarray,
        model: Callable[[Any], ScikitLearnModel],
        model_parameters: Dict[str, Any],
        preprocessor: Callable[[np.ndarray, np.ndarray, bool], Tuple[np.ndarray, np.ndarray]],
        max_tokens: List[int],
        preprocess_name: str,
        verbose: bool = True
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[int]
]:
    training_accuracies, training_mf1, training_wf1 = list(), list(), list()
    val_accuracies, val_mf1, val_wf1 = list(), list(), list()

    for max_token in max_tokens:
        processed_x, processed_y = preprocessor(x.copy(), y.copy(), False, max_features=max_token)
        result = cross_validate(processed_x, processed_y, model(**model_parameters))
        training_accuracies.append(result['training accuracy'])
        training_mf1.append(result['training macro f1'])
        training_wf1.append(result['training weighted f1'])
        val_accuracies.append(result['accuracy'])
        val_mf1.append(result['macro f1'])
        val_wf1.append(result['weighted f1'])

    if verbose:
        plt.plot(max_tokens, training_accuracies, label='Training Accuracies')
        plt.plot(max_tokens, training_mf1, label='Training Macro f1')
        plt.plot(max_tokens, training_wf1, label='Training Weighted f1')
        plt.plot(max_tokens, val_accuracies, label='Validation Accuracies')
        plt.plot(max_tokens, val_mf1, label='Validation Macro f1')
        plt.plot(max_tokens, val_wf1, label='Validation Weighted f1')
        plt.xlabel('Number of Max Features')
        plt.ylabel('Measures')
        plt.legend()
        plt.title(f'Performance of the {preprocess_name} Versus Number of Max Features')
        plt.show()

    return training_accuracies, training_mf1, training_wf1, val_accuracies, val_mf1, val_wf1, max_tokens


# endregion

# region Main

if __name__ == '__main__':
    positive_data, negative_data = read_data(POSITIVE_FILE_PATH), read_data(NEGATIVE_FILE_PATH)

    # Prepare training and test data,
    # and combine the positive and negative
    # data set in each category into one.
    pos_training_x, pos_test_x = test_data_split(positive_data, 0.3)
    neg_training_x, neg_test_x = test_data_split(negative_data, 0.3)
    training_x, test_x = np.append(pos_training_x, neg_training_x), np.append(pos_test_x, neg_test_x)
    training_y, test_y = np.append(np.ones(pos_training_x.shape, dtype=int),
                                   np.zeros(neg_training_x.shape, dtype=int)), np.append(
        np.ones(pos_test_x.shape, dtype=int), np.zeros(neg_test_x.shape, dtype=int))
    training_data, test_data = np.append(training_x[:, np.newaxis], training_y[:, np.newaxis], axis=1), np.append(
        test_x[:, np.newaxis], test_y[:, np.newaxis], axis=1)
    np.random.shuffle(training_data) and np.random.shuffle(test_data)

    print(f'{chalk.greenBright.bold("-" * 15 + "COMPLETED PARSING DATA" + "-" * 15)}\n')
    print(f'{chalk.bold("TRAINING DATA SIZE:")} {len(training_data)}\n')
    print(f'{chalk.bold("TEST DATA SIZE:")} {len(test_data)}\n')

    # Compare various preprocessing techniques
    preprocess_techniques_set = {
        "unigram": [preprocess_count_uni_gram_occurrence],
        "unigram_wrt_stopwords": [preprocess_count_uni_gram_occurrence_excluding_stop_words],
        "unigram_stemming": [preprocess_count_uni_gram_occurrence_with_stemming],
        "unigram_lemmatization": [preprocess_count_uni_gram_occurrence_with_lemmatization],
        "bigram": [preprocess_count_bigram_occurrence],
        "bigram_wrt_stopwords": [preprocess_count_bigram_occurrence_excluding_stop_words],
        "uni_bi": [preprocess_uni_bi]
    }
    preprocess_comparison_result = evaluate_preprocess_techniques(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        [preprocess_null_process],
        preprocess_techniques_set,
        LogisticRegression,
        MODEL_PARAMETERS
    )
    max_tokens = [100, 1000, 3000, 5000, 8000, 10000, 20000, 50000, 100000]

    print(f'{chalk.bold("-" * 15 + "COMPLETED COMPARISON AMONG VARIOUS PREPROCESS TECHNIQUES" + "-" * 15)}\n')
    pprint.pprint(preprocess_comparison_result)

    # Evaluate the impact of max features on preprocess_count_uni_gram_occurrence
    evaluate_preprocess_performance_by_max_token(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        LogisticRegression,
        MODEL_PARAMETERS,
        preprocess_count_uni_gram_occurrence,
        max_tokens,
        'preprocess_count_uni_gram_occurrence'
    )

    # Evaluate the impact of max features on preprocess_count_uni_gram_occurrence_excluding_stop_words
    evaluate_preprocess_performance_by_max_token(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        LogisticRegression,
        MODEL_PARAMETERS,
        preprocess_count_uni_gram_occurrence_excluding_stop_words,
        max_tokens,
        'preprocess_count_uni_gram_occurrence_excluding_stop_words'
    )

    # Evaluate the impact of max features on preprocess_count_uni_gram_occurrence_with_stemming
    evaluate_preprocess_performance_by_max_token(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        LogisticRegression,
        MODEL_PARAMETERS,
        preprocess_count_uni_gram_occurrence_with_stemming,
        max_tokens,
        'preprocess_count_uni_gram_occurrence_with_stemming'
    )

    # Evaluate the impact of max features on preprocess_count_uni_gram_occurrence_with_lemmatization
    evaluate_preprocess_performance_by_max_token(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        LogisticRegression,
        MODEL_PARAMETERS,
        preprocess_count_uni_gram_occurrence_with_lemmatization,
        max_tokens,
        'preprocess_count_uni_gram_occurrence_with_lemmatization'
    )

    # Evaluate the impact of max features on preprocess_count_bigram_occurrence
    evaluate_preprocess_performance_by_max_token(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        LogisticRegression,
        MODEL_PARAMETERS,
        preprocess_count_bigram_occurrence,
        max_tokens,
        'preprocess_count_bigram_occurrence'
    )

    # Evaluate the impact of max features on preprocess_count_bigram_occurrence_excluding_stop_words
    evaluate_preprocess_performance_by_max_token(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        LogisticRegression,
        MODEL_PARAMETERS,
        preprocess_count_bigram_occurrence_excluding_stop_words,
        max_tokens,
        'preprocess_count_bigram_occurrence_excluding_stop_words'
    )

    # Evaluate the impact of max features on preprocess_uni_bi
    evaluate_preprocess_performance_by_max_token(
        training_data[:, :-1],
        training_data[:, -1].astype(int),
        LogisticRegression,
        MODEL_PARAMETERS,
        preprocess_uni_bi,
        max_tokens,
        'preprocess_uni_bi'
    )

    # Apply the best preprocess techniques and apply it in
    # the test set.
    best_techniques, best_macro_f1, best_technique_name = list(), 0, ""
    for key in preprocess_comparison_result['experiment']:
        experimental_result = preprocess_comparison_result['experiment'][key]
        if experimental_result['macro f1'] > best_macro_f1:
            best_macro_f1 = experimental_result['macro f1']
            best_techniques = preprocess_techniques_set[key]
            best_technique_name = key

    print(f'{chalk.bold.greenBright("THE BEST PREPROCESS TECHNIQUE IS")} {best_technique_name}.\n')

    combined_data = np.append(training_data, test_data, axis=0)
    combined_data_x, combined_data_y = preprocess_data(
        combined_data[:, :-1],
        combined_data[:, -1].astype(int),
        best_techniques
    )

    best_preprocessed_training_x = combined_data_x[:len(training_data)]
    best_preprocessed_training_y = combined_data_y[:len(training_data)]
    best_preprocessed_test_x = combined_data_x[len(training_data):]
    best_preprocessed_test_y = combined_data_y[len(training_data):]

    model = LogisticRegression(**MODEL_PARAMETERS)
    model.fit(best_preprocessed_training_x, best_preprocessed_training_y)
    test_result = classification_report(
        best_preprocessed_test_y,
        model.predict(best_preprocessed_test_x),
        target_names=np.array(['Negative Reviews', 'Positive Reviews'])
    )

    print(test_result)

# endregion

if __name__ == '__main__':
    # region Read Data
    training_data = read_csv("data_sets/data_A2/fake_news/fake_news_train.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(training_data.shape)}')
    test_data = read_csv("data_sets/data_A2/fake_news/fake_news_test.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(test_data.shape)}')
    val_data = read_csv("data_sets/data_A2/fake_news/fake_news_val.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(val_data.shape)}')

    # endregion
