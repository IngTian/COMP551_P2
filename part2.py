import json
from lib.utils.io_utils import read_csv
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import nltk

nltk.download('wordnet')
from nltk.stem import SnowballStemmer
import numpy as np
from simple_chalk import chalk


# region Preprocess Data Functions
def preprocess_get_vectorizer_with_stemming(
        corpus: np.ndarray,
        params: Dict = None
) -> CountVectorizer:
    stemming_helper = SnowballStemmer('english')
    vectorizer = CountVectorizer(**params)
    vectorizer.fit(corpus[:, 0], corpus[:, 1])
    analyzer = vectorizer.build_analyzer()

    new_vectorizer = CountVectorizer(
        **params,
        analyzer=lambda s: (stemming_helper.stem(word) for word in analyzer(s)),
    )

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

    params = [
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": True,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": True,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": False,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": False,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": True,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": True,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": False,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": False,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": False,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": True,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": True,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": False,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": False,
                "lowercase": False,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": True,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": True,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": False,
                "ngram_range": (1, 1)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
        {
            "stemming": True,
            "use_tfidf": False,
            "vectorizer_params": {
                "binary": True,
                "lowercase": False,
                "ngram_range": (1, 2)
            },
            "tfidf_params": {},
            "clf_params": {
                "max_iter": 1000,
                "solver": "sag"
            }
        },
    ]

    # Running results
    results = []

    for param_combination in tqdm(params):
        use_stemming = param_combination['stemming']
        use_tfidf = param_combination['use_tfidf']
        tfidf_params = param_combination['tfidf_params']
        vectorizer_params = param_combination['vectorizer_params']
        clf_params = param_combination['clf_params']

        vectorizer, text_clf = None, None

        if use_stemming:
            vectorizer = preprocess_get_vectorizer_with_stemming(training_data, vectorizer_params)
        else:
            vectorizer = CountVectorizer(**vectorizer_params)

        if not use_tfidf:
            text_clf = Pipeline([
                ("vect", vectorizer),
                ("clf", LogisticRegression(**clf_params))
            ])
        else:
            text_clf = Pipeline([
                ("vect", vectorizer),
                ("tfidf", TfidfTransformer(**tfidf_params)),
                ("clf", LogisticRegression(**clf_params))
            ])

        text_clf.fit(training_data[:, 0], training_data[:, 1].astype(int))

        result = {}
        result["params"] = param_combination
        result["val"] = classification_report(
            val_data[:, -1].astype(int),
            text_clf.predict(val_data[:, 0]).astype(int),
            output_dict=True,
            zero_division=0
        )
        result["test"] = classification_report(
            test_data[:, -1].astype(int),
            text_clf.predict(test_data[:, 0]).astype(int),
            output_dict=True,
            zero_division=0
        )

        results.append(result)

    f = open("./output/part2.json", 'w')
    f.write(json.dumps(results, indent=4))
    f.close()

# endregion
