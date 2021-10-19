import numpy as np
from lib.utils.math_utils import sigmoid
from typing import Dict, Tuple, List, Any
from simple_chalk import chalk
from lib.types.types import LearningModel


def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.count_nonzero(y_pred == y_true) / len(y_pred))


class LogisticRegression(LearningModel):

    def __init__(
            self,
            add_bias: bool = True,
            learning_rate: float = .1,
            epsilon: float = 2.5e-2,
            verbose: bool = False,
            mini_batch: int = None,
            momentum: float = None,
            epoch: int = None,
            accuracy_record_num: int = 20
    ):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # to get the tolerance for the norm of gradients
        self.mini_batch = mini_batch
        self.momentum = momentum
        self.verbose = verbose
        self.epoch = epoch
        self.accuracy_record_num = accuracy_record_num

        # Model Parameters.
        self.weights = None
        self.history_gradients = None

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            **kwargs
    ) -> Tuple[
        LearningModel,
        int,
        float,
        bool,
        List[Tuple[int, float, float]]
    ]:

        # Prepare X
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            number_of_instances = x.shape[0]
            x = np.column_stack([x, np.ones(number_of_instances)])

        # Set up parameters
        val_x, val_y = kwargs["val_x"], kwargs["val_y"]
        number_of_instances, number_of_features = x.shape
        self.weights = np.zeros(number_of_features)
        self.history_gradients = list()
        raw_gradients, epoch_run = np.inf, 0
        accuracy_record = list()
        accuracy_check_point = self.epoch // self.accuracy_record_num

        while epoch_run < self.epoch:
            # Get the segmented training
            # sets based on batch size.
            training_sets = self.separate_training_data(x, y)

            for batch in training_sets:
                raw_gradients = self.gradient(batch[0], batch[1])
                self.update_weights(raw_gradients)

            epoch_run += 1

            if epoch_run % accuracy_check_point == 0:
                accuracy_record.append(
                    (
                        epoch_run,
                        calculate_accuracy(self.predict(x), y),
                        calculate_accuracy(self.predict(val_x), val_y)
                    )
                )

            if np.linalg.norm(raw_gradients) <= self.epsilon:
                break

        if self.verbose:
            print(
                f'{chalk.bold("-" * 15 + "COMPLETED FITTING" + "-" * 15)}\n'
                f'EPOCHS: {chalk.green.bold(epoch_run)}\n'
                f'GRADIENT CHANGE: {chalk.yellowBright.bold(np.linalg.norm(raw_gradients))}\n'
                f'FINAL WEIGHTS: {chalk.blueBright(self.weights)}\n')

        return self, epoch_run, np.linalg.norm(raw_gradients), np.linalg.norm(
            raw_gradients) <= self.epsilon, accuracy_record

    def update_weights(self, raw_gradients: np.ndarray) -> None:
        """
        Given the raw_gradients
        :param raw_gradients:
        :return:
        """

        g = self.update_weights_momentum(len(self.history_gradients), raw_gradients) \
            if self.momentum \
            else raw_gradients

        g = self.learning_rate * g

        self.history_gradients.append(g)

        self.weights = self.weights - g

    def update_weights_momentum(self, t: int, raw_gradients: np.ndarray) -> np.ndarray:
        if t == 0:
            return (1 - self.momentum) * raw_gradients
        else:
            return self.momentum * self.history_gradients[t - 1] + (1 - self.momentum) * raw_gradients

    def separate_training_data(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Separate the training data into
        segments according to the batch size.
        :param self:
        :param x: X
        :param y: y
        :return: A list of segmented batches.
        """
        if self.mini_batch >= x.shape[0]:
            return [(x, y)]

        complete_data = np.append(x if x.ndim > 1 else x[:, None], y if y.ndim > 1 else y[:, None], axis=1)
        np.random.shuffle(complete_data)

        result = list()

        while complete_data.shape[0] > 0:
            if complete_data.shape[0] >= self.mini_batch:
                result.append((complete_data[:self.mini_batch, :-1], complete_data[:self.mini_batch, -1]))
                complete_data = complete_data[self.mini_batch:]
            else:
                result.append((complete_data[:, :-1], complete_data[:, -1]))
                break

        return result

    def predict(self, x):
        """
        Make predictions based on the weights given.
        :param x: X
        :return: The predicted y.
        """
        # Prepare X
        if x.ndim == 1:
            x = x[:, None]
        number_of_tests = x.shape[0]
        if self.add_bias and x.shape[1] != len(self.weights):
            x = np.column_stack([x, np.ones(number_of_tests)])

        # Make predictions
        result = sigmoid(np.dot(x, self.weights))
        positive = result > 0.5
        result[positive] = 1
        result[~positive] = 0
        return result.astype(int)

    def get_params(self):
        return self.__dict__

    def set_params(self, new_params: Dict[str, Any]):
        for k, v in new_params.items():
            if k == 'add_bias':
                self.add_bias = v
            elif k == 'learning_rate':
                self.learning_rate = v
            elif k == 'epsilon':
                self.epsilon = v
            elif k == 'verbose':
                self.verbose = v
            elif k == 'mini_batch':
                self.mini_batch = v
            elif k == 'momentum':
                self.momentum = v
        return True

    def gradient(self, x, y):
        """
        Calculate the gradient of logistic regression.
        :param x:
        :param y:
        :return:
        """
        number_of_instances, number_of_features = x.shape
        yh = sigmoid(np.dot(x, self.weights))  # predictions  size N
        grad = np.dot(x.T, yh - y) / number_of_instances  # divide by N because cost is mean over N points
        return grad

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        z = np.dot(x, self.weights)
        return float(np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z))))
