import json
from lib.utils.io_utils import read_csv
from lib.utils.utils import preprocess_data, get_best_model_parameter, cross_validate
from typing import List, Tuple, Callable, Dict, Any, Union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
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
def preprocess_corpora(
        training_corpus: np.ndarray,
        validation_corpus: np.ndarray,
        test_corpus: np.ndarray,
        vectorizer: CountVectorizer
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    training_x, training_y = training_corpus[:, 0], training_corpus[:, -1]
    validation_x, validation_y = validation_corpus[:, 0], validation_corpus[:, -1]
    test_x, test_y = test_corpus[:, 0], test_corpus[:, -1]

    transformed_training_x = vectorizer.transform(training_x).toarray()
    transformed_validation_x = vectorizer.transform(validation_x).toarray()
    transformed_test_x = vectorizer.transform(test_x).toarray()

    transformed_training = np.append(
        transformed_training_x if transformed_training_x.ndim > 1 else transformed_training_x[:, np.newaxis],
        training_y if training_y.ndim > 1 else training_y[:, np.newaxis], axis=1)
    transformed_validation = np.append(
        transformed_validation_x if transformed_validation_x.ndim > 1 else transformed_validation_x[:, np.newaxis],
        validation_y if validation_y.ndim > 1 else validation_y[:, np.newaxis], axis=1)
    transformed_test = np.append(
        transformed_test_x if transformed_test_x.ndim > 1 else transformed_test_x[:, np.newaxis],
        test_y if test_y.ndim > 1 else test_y[:, np.newaxis], axis=1)

    return transformed_training, transformed_validation, transformed_test


def preprocess_get_unigram_vectorizer(
        corpus: np.ndarray,
        params: Dict = None
) -> CountVectorizer:
    vectorizer = CountVectorizer(**params)
    vectorizer.fit(corpus)
    return vectorizer


def preprocess_get_unigram_ecl_stopwords_vectorizer(
        corpus: np.ndarray,
        params: Dict = None
) -> CountVectorizer:
    vectorizer = CountVectorizer(**params, stop_words=stopwords.words('english'))
    vectorizer.fit(corpus)
    return vectorizer


def preprocess_get_unigram_with_stemming_vectorizer(
        corpus: np.ndarray,
        params: Dict = None
) -> CountVectorizer:
    stemming_helper = SnowballStemmer('english')
    vectorizer = CountVectorizer(**params)
    vectorizer.fit(corpus)
    analyzer = vectorizer.build_analyzer()

    new_vectorizer = CountVectorizer(
        **params,
        analyzer=lambda s: (stemming_helper.stem(word) for word in analyzer(s)),
    )

    new_vectorizer.fit(corpus)

    return new_vectorizer


def preprocess_get_unigram_with_lemmatization_vectorizer(
        corpus: np.ndarray,
        params: Dict = None
) -> CountVectorizer:
    lemmatizer = WordNetLemmatizer()
    vectorizer = CountVectorizer(**params)
    vectorizer.fit(corpus)
    analyzer = vectorizer.build_analyzer()

    new_vectorizer = CountVectorizer(
        **params,
        analyzer=lambda s: (lemmatizer.lemmatize(word) for word in analyzer(s)),
    )

    new_vectorizer.fit(corpus)

    return new_vectorizer


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
    max_features_dict = {
        'max_features': 10000
    }

    vectorizer_generators = [
        preprocess_get_unigram_vectorizer,
        preprocess_get_unigram_ecl_stopwords_vectorizer,
        preprocess_get_unigram_with_stemming_vectorizer,
        preprocess_get_unigram_with_lemmatization_vectorizer
    ]

    processed_data_sets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = list()

    print(f'{chalk.bold("-" * 15 + "START PREPROCESSING" + "-" * 15)}\n')
    for generator in tqdm(vectorizer_generators):
        vectorizer = generator(training_data[:, 0], max_features_dict)
        processed_data_sets.append(preprocess_corpora(training_data, val_data, test_data, vectorizer))

    print(f'\n\n{chalk.bold("-" * 15 + "DATA PREPROCESS COMPLETED" + "-" * 15)}\n')

    # Run grid search for each preprocessed data
    # and pick the best combination
    results: Dict[str, Any] = dict()

    print(f'\n\n{chalk.bold("-" * 15 + "STARTING RUNNING TESTS" + "-" * 15)}\n')

    for generator_idx in tqdm(range(len(vectorizer_generators))):
        preprocess = vectorizer_generators[generator_idx]
        train_data, val_data, test_data = processed_data_sets[generator_idx]
        name_of_the_preprocess = preprocess.__name__

        model = LogisticRegression(max_iter=1000)
        model.fit(train_data[:, :-1].astype(int), train_data[:, -1].astype(int))

        validation_report = classification_report(
            val_data[:, -1].astype(int),
            model.predict(val_data[:, :-1].astype(int)),
            output_dict=True,
            zero_division=0
        )

        results[name_of_the_preprocess] = validation_report

    # Save results
    print(f'\n\n{chalk.bold("-" * 15 + "COMPLETED" + "-" * 15)}\n')
    f = open("./output/part2.json", 'w')
    f.write(json.dumps(results, indent=4))
    f.close()

# endregion
