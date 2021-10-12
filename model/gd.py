import numpy as np
from utils.math_utils import sigmoid
from typing import Dict, Tuple, List, Any
from enum import Enum
from simple_chalk import chalk


class UpdateWeightMethod(Enum):
    REGULAR = 1
    MOMENTUM = 2


class LogisticRegression:

    def __init__(
            self,
            add_bias: bool = True,
            learning_rate: float = .1,
            epsilon: float = 1e-4,
            max_iterations: int = 1e5,
            verbose: bool = True,
            mini_batch_ratio: float = 1,
            momentum: float = None,
            update_weight_method: UpdateWeightMethod = UpdateWeightMethod.REGULAR
    ):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # to get the tolerance for the norm of gradients
        self.max_iterations = max_iterations  # maximum number of iteration of gradient descent
        self.mini_batch_ratio = mini_batch_ratio
        self.momentum = momentum
        self.update_weights_method = update_weight_method
        self.verbose = verbose

        # Model Parameters.
        self.weights = None
        self.history_gradients = None

    def fit(self, x, y):

        # Prepare X
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            number_of_instances = x.shape[0]
            x = np.column_stack([x, np.ones(number_of_instances)])

        # Get the segmented training
        # sets based on batch size.
        training_sets = self.separate_training_data(x, y)

        # Set up parameters
        number_of_instances, number_of_features = x.shape
        self.weights = np.zeros(number_of_features)
        self.history_gradients = list()
        current_gradient, iterations_run = np.inf, 0

        # the code snippet below is for gradient descent
        while np.linalg.norm(current_gradient) > self.epsilon and iterations_run < self.max_iterations:
            training_set = training_sets[iterations_run % len(training_sets)]
            current_gradient = self.gradient(training_set[0], training_set[1])
            self.history_gradients.append(current_gradient)
            self.update_weights(current_gradient)
            iterations_run += 1

        if self.verbose:
            print(
                f'{chalk.bold("-" * 15 + "COMPLETED FITTING" + "-" * 15)}\n'
                f'NUMBER OF ITERATIONS: {chalk.green.bold(iterations_run)} FINAL GRADIENT NORM: {chalk.yellowBright.bold(np.linalg.norm(current_gradient))}\n'
                f'FINAL WEIGHTS: {chalk.blueBright(self.weights)}\n')

        return self

    def update_weights(self, raw_gradients: np.ndarray) -> None:
        """
        Given the raw_gradients
        :param raw_gradients:
        :return:
        """
        if self.update_weights_method == UpdateWeightMethod.MOMENTUM:
            # TODO: Use momentum to update weights
            beta = momentum
            T = self.history_gradients.size
            for t in np.arange(1, T):
                self.weights += self.history_gradients[-t] * (1 - beta) * beta**(T - t)
        elif self.update_weights_method == UpdateWeightMethod.REGULAR:
            # TODO: Update weights regularly
            self.weights = self.weights - self.learning_rate * current_gradient 

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
        complete_data = np.append(x if x.ndim > 1 else x[:, None], y if y.ndim > 1 else y[:, None], axis=1)
        batch_size = int(x.shape[0] * self.mini_batch_ratio)

        if batch_size > x.shape[0]:
            raise ValueError("Batch size is larger than the number of instances.")

        result = list()

        while complete_data.shape[0] > 0:
            if complete_data.shape[0] >= batch_size:
                result.append((complete_data[:batch_size, :-1], complete_data[:batch_size, -1]))
                complete_data = complete_data[batch_size:]

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
        if self.add_bias:
            x = np.column_stack([x, np.ones(number_of_tests)])

        # Make predictions
        return sigmoid(np.dot(x, self.weights))

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
            elif k == 'max_iterations':
                self.max_iterations = v
            elif k == 'verbose':
                self.verbose = v
            elif k == 'mini_batch_ratio':
                self.mini_batch_ratio = v
            elif k == 'momentum':
                self.momentum = v
            elif k == 'update_weights_method':
                self.update_weights_method = v
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
