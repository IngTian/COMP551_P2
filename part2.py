import json

from lib.utils.io_utils import read_csv
from lib.utils.utils import preprocess_data, get_best_model_parameter, cross_validate
from typing import List, Tuple, Callable, Dict, Any, Union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import numpy as np
from simple_chalk import chalk


# region Preprocess Data Functions
def preprocess_count_uni_gram_occurrence(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer()

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
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
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
) -> Tuple[np.ndarray, np.ndarray]:
    stemming_helper = SnowballStemmer('english')
    vectorizer = CountVectorizer()
    vectorizer.fit(x[:, 0])
    analyzer = vectorizer.build_analyzer()

    new_vectorizer = CountVectorizer(
        analyzer=lambda s: (stemming_helper.stem(word) for word in analyzer(s))
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
) -> Tuple[np.ndarray, np.ndarray]:
    lemmatizer = WordNetLemmatizer()
    vectorizer = CountVectorizer()
    vectorizer.fit(x[:, 0])
    analyzer = vectorizer.build_analyzer()

    new_vectorizer = CountVectorizer(
        analyzer=lambda s: (lemmatizer.lemmatize(word) for word in analyzer(s))
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
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(ngram_range=(2, 2))
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
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(
        stop_words=stopwords.words('english'),
        ngram_range=(2, 2)
    )
    transformed_x = vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Unigram Occurrence Excluding Stopwords")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


def preprocess_uni_bi(
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    transformed_x = vectorizer.fit_transform(x[:, 0]).toarray()
    if verbose:
        print(f'{chalk.greenBright("Completed Extract Features with Bigram Occurrence")}\n'
              f'{chalk.bold("Shape:")} {transformed_x.shape}\n'
              f'{transformed_x[:2]}')
    return transformed_x, y


# endregion

# region Main

if __name__ == '__main__':
    # Read Data
    training_data = read_csv("data_sets/data_A2/fake_news/fake_news_train.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(training_data.shape)}')
    test_data = read_csv("data_sets/data_A2/fake_news/fake_news_test.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(test_data.shape)}')
    val_data = read_csv("data_sets/data_A2/fake_news/fake_news_val.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(val_data.shape)}')

    # Preprocess Data
    preprocesses = [
        preprocess_uni_bi,
        preprocess_count_bigram_occurrence,
        preprocess_count_bigram_occurrence_excluding_stop_words,
        preprocess_count_uni_gram_occurrence,
        preprocess_count_uni_gram_occurrence_with_lemmatization,
        preprocess_count_uni_gram_occurrence_with_stemming,
        preprocess_count_uni_gram_occurrence_excluding_stop_words,
    ]

    processed_data_sets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = list()

    print(f'{chalk.bold("-" * 15 + "START PREPROCESSING" + "-" * 15)}\n')
    for preprocess in tqdm(preprocesses):
        train = preprocess_data(
            np.copy(training_data[:, :-1]),
            np.copy(training_data[:, -1]),
            [
                preprocess
            ],
            verbose=False
        )
        val = preprocess_data(
            np.copy(val_data[:, :-1]),
            np.copy(val_data[:, -1]),
            [
                preprocess
            ],
            verbose=False
        )
        test = preprocess_data(
            np.copy(test_data[:, :-1]),
            np.copy(test_data[:, -1]),
            [
                preprocess
            ],
            verbose=False
        )
        train = np.append(
            train[0] if train[0].ndim > 1 else train[0][:, None],
            train[1] if train[1].ndim > 1 else train[1][:, None],
            axis=1)
        val = np.append(
            val[0] if val[0].ndim > 1 else val[0][:, None],
            val[1] if val[1].ndim > 1 else val[1][:, None],
            axis=1)
        test = np.append(
            test[0] if test[0].ndim > 1 else test[0][:, None],
            test[1] if test[1].ndim > 1 else test[1][:, None],
            axis=1)
        processed_data_sets.append((train, val, test))

    print(f'{chalk.bold("-" * 15 + "DATA PREPROCESS COMPLETED" + "-" * 15)}\n')

    # Run grid search for each preprocessed data
    # and pick the best combination
    results: Dict[str, Any] = dict()
    params = {
        "max_iter": [500, 1000, 5000, 10000],
        "multi_class": ["ovr"],
    }

    print(f'{chalk.bold("-" * 15 + "STARTING RUNNING TESTS" + "-" * 15)}\n')

    for preprocess_idx in tqdm(range(len(preprocesses))):
        preprocess = preprocesses[preprocess_idx]
        train_data, val_data, test_data = processed_data_sets[preprocess_idx]
        name_of_the_preprocess = preprocess.__name__
        preprocess_performance = get_best_model_parameter(
            params,
            LogisticRegression,
            train_data[:, :-1],
            train_data[:, -1],
            cross_validate,
            val_x=val_data[:, :-1],
            val_y=val_data[:, -1]
        )

        results[name_of_the_preprocess] = preprocess_performance

    # Save results
    print(f'{chalk.bold("-" * 15 + "COMPLETED" + "-" * 15)}\n')
    f = open("./output/part2.json", 'w')
    f.write(json.dumps(results, indent=4))
    f.close()

# endregion
