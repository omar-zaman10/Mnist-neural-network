import json

import numpy as np
from mnist_loader import load_data


class NeuralNet:
    def __init__(self):
        self._nn_structure = [784, 16, 16, 10]

        self._layers = len(self._nn_structure)

        self._weights = {}  # Weights set up as layer: weight matrix, wji
        self._biases = {}  # Biases set up as layer: bias vector

        self._h = {}  # Activations
        self._z = {}

        self._delta_w = {}  # dictionary which has delta wji matrix for each layer
        self._delta_b = {}  # dictionary which has delta b vectors for each layer

        self._batch_size = 1000
        self._training_data = None
        self._input_data = None
        self._labels = None

        self._max_iter = 2000
        self._alpha = 2.5

    def _initialise_data(self, training_data):
        self._training_data = training_data
        self._input_data = training_data[0]
        self._labels = training_data[1]

    def _initialise_weights(self):
        for l in range(1, self._layers):
            # randn used to generate numbers between -0.5 and 0.5 for the weights and biases
            self._weights[l] = np.random.randn(
                self._nn_structure[l], self._nn_structure[l - 1]
            )  # Creates weight matrix in each layer
            self._biases[l] = np.random.randn(
                self._nn_structure[l]
            )  # Creates bias vector in each layer

    def _initialise_deltas(self):
        for l in range(1, self._layers):
            self._delta_w[l] = np.zeros(
                (self._nn_structure[l], self._nn_structure[l - 1])
            )
            self._delta_b[l] = np.zeros(self._nn_structure[l])

    def _give_vectorised_label(self, y):
        lab = np.zeros(10)
        lab[y] = 1.0
        return lab

    def _give_batch(self, k):
        return self._input_data[k : k + self._batch_size], self._labels[
            k : k + self._batch_size
        ]

    def _reshuffle(self):
        randomize = np.arange(len(self._input_data))
        np.random.shuffle(randomize)
        self._input_data = self._input_data[randomize]
        self._labels = self._labels[randomize]

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _sigmoid_gradient(self, x):
        f = self._sigmoid(x)
        return (1.0 - f) * f

    def _feed_forward(self, x):
        self._h[1] = x

        for l in range(1, len(self._nn_structure)):
            z_l = (
                np.matmul(self._weights[l], self._h[l]) + self._biases[l]
            )  # W h + b = z

            self._h[l + 1] = self._sigmoid(z_l)  # f(z) = h

            self._z[l + 1] = z_l

        # self._h[self._layers] = z_l
        # self._softmax_output_layer()

        self._normalise_output_layer()

    def _softmax_output_layer(self):
        """Use instead of _sigmoid activation layer and normalisation"""
        self._h[self._layers] = np.exp(self._h[self._layers])

        factor = sum(self._h[self._layers])
        self._h[self._layers] /= factor

    def _normalise_output_layer(self):
        factor = sum(self._h[self._layers])
        self._h[self._layers] /= factor

    def _cost_function(self, x, y):
        """Cost for a single image"""

        self._feed_forward(x)
        return np.linalg.norm(y - self._h[self._layers])

    def _batch_cost(self, input_batch, labels_batch):
        """Cost of a batch of images"""

        c = 0
        for x, y in zip(input_batch, labels_batch):
            y = self._give_vectorised_label(y)
            c += self._cost_function(x, y)

        c /= len(labels_batch)
        return c

    def _entire_cost(self):
        c = 0
        for x, y in zip(self._input_data, self._labels):
            y = self._give_vectorised_label(y)
            c += self._cost_function(x, y)

        c /= len(self._labels)
        return c

    def _cost_derivative(self, y):
        return self._h[self._layers] - y

    def _backprop(self, y):
        y = self._give_vectorised_label(y)

        cost_grad = self._cost_derivative(y)

        output_deltas = cost_grad * self._sigmoid_gradient(
            self._z[self._layers]
        )  # delta = dC/dh_l * dh_l/dz_l

        deltas = {self._layers: output_deltas}

        for l in range(self._layers - 1, 0, -1):
            # Chain rule derivative dC/db_l = delta
            self._delta_b[l] += deltas[l + 1]

            # Chain rule derivative dC/dW = delta * h_l
            self._delta_w[l] += np.dot(
                deltas[l + 1][:, np.newaxis], np.transpose(self._h[l][:, np.newaxis])
            )

            # Backpropagation for deltas across different layers
            if l > 1:
                deltas[l] = np.dot(
                    np.transpose(self._weights[l]), deltas[l + 1]
                ) * self._sigmoid_gradient(self._z[l])

    def _gradient_descent(self):
        m = self._batch_size

        for l in range(1, self._layers):
            self._weights[l] += -1.0 / m * self._alpha * self._delta_w[l]
            self._biases[l] += -1.0 / m * self._alpha * self._delta_b[l]

    def train(self, training_data):
        self._initialise_data(training_data)
        self._initialise_weights()
        self.load_from_json()

        num_batches = len(input_data) / self._batch_size

        for i in range(self._max_iter):
            if i % num_batches == 0:
                self._reshuffle()

            self._initialise_deltas()

            k = (i * self._batch_size) % len(input_data)

            input_batch, labels_batch = self._give_batch(k)

            for x, y in zip(input_batch, labels_batch):
                self._feed_forward(x)
                self._backprop(y)

            self._gradient_descent()

            c = self._batch_cost(input_batch, labels_batch)

            if i % 10 == 0:
                print(f"Cost {c.round(8)} for iteration: {i}")

    def save_to_json(self):
        """Save weights and biases after training as json file"""

        weights = {key: value.tolist() for key, value in self._weights.items()}
        biases = {key: value.tolist() for key, value in self._biases.items()}

        with open("data/weights.json", "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=4)

        with open("data/biases.json", "w", encoding="utf-8") as f:
            json.dump(biases, f, ensure_ascii=False, indent=4)

        print("Weights and biases saved as json files!")

    def load_from_json(self):
        """Load weights from json file"""

        with open("data/weights.json") as w:
            weights = json.load(w)

        with open("data/biases.json") as b:
            biases = json.load(b)

        self._weights = {int(key): np.array(value) for key, value in weights.items()}
        self._biases = {int(key): np.array(value) for key, value in biases.items()}

    def evaluate_training(self):
        accuracy = 0

        i = 0
        for x, y in zip(self._input_data, self._labels):
            self._feed_forward(x)
            soft_output = self._h[self._layers]
            prediction = list(soft_output).index(max(soft_output))

            if prediction == y:
                accuracy += 1
            i += 1
        print(f"Accuracy was {100.0 * accuracy / len(self._input_data)} %")
        return accuracy / len(self._input_data)

    def evaluate(self, test_data):
        input_data, labels = test_data[0], test_data[1]

        accuracy = 0
        i = 0
        for x, y in zip(input_data, labels):
            self._feed_forward(x)
            soft_output = self._h[self._layers]
            prediction = list(soft_output).index(max(soft_output))

            if prediction == y:
                accuracy += 1
            i += 1

        print(f"Accuracy was {100.0 * accuracy / len(input_data)} %")
        return accuracy / len(input_data)


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data()

    input_data = test_data[0]
    labels = test_data[1]

    neural_net = NeuralNet()

    neural_net.train(training_data)
    neural_net.evaluate_training()

    neural_net.save_to_json()
