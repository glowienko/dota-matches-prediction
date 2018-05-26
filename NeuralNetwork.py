from numpy import random, exp, dot, array, power, zeros


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:

    def init_layers(self, first_layer_neurons_number, second_layer_neurons_number, third_layer_neurons_number):
        self.input_layer_neuron_number = first_layer_neurons_number
        self.hidden_layer_neuron_number = second_layer_neurons_number
        self.output_layer_neuron_number = third_layer_neurons_number

    def init_weights(self):
        self.first_weights = 2 * random.random((self.hidden_layer_neuron_number, self.input_layer_neuron_number)) - 1
        self.second_weights = 2 * random.random((self.output_layer_neuron_number, self.hidden_layer_neuron_number)) - 1

    def feed_forward(self, inputData):
        first_result = self.sigmoid(dot(self.first_weights, inputData))
        result = self.sigmoid(dot(self.second_weights, first_result))
        return result

    def backpropagate(self, inputData, outputData):
        g2 = [zeros(weight.shape) for weight in self.second_weights]
        g1 = [zeros(weight.shape) for weight in self.first_weights]

        A1 = inputData
        Z1 = dot(self.first_weights, A1)
        A2 = sigmoid(Z1)
        Z2 = dot(self.second_weights, A2)
        A3 = sigmoid(Z2)

        d2 = (A3 - outputData) * sigmoid_derivative(Z2)
        g2 = dot(d2, A2.transpose())

        d1 = dot(self.second_weights.transpose(), d2) * sigmoid_derivative(Z1)
        g1 = dot(d1, A1.transpose())

        return g1, g2

    def train(self):
        inputData = array([[1], [0]])
        outputData = array([[1], [0]])
        self.init_layers(2, 3, 2)
        self.init_weights()

        self.backpropagate(inputData, outputData)
        # for i in range(1000000000):
        #
        #     result_output = self.feed_forward(inputData)
        #
        #     error = output - result_output
        #     square_error = (power(error, 2)) / 2
        #     derivative = self.sigmoid_derivative(sum(square_error))
        #     # if derivative > 0:
        #     #     self.first_weights -= (derivative * 0.000001)
        #     #     self.second_weights -= (derivative * 0.000001)
        #     # elif derivative < 0:
        #     #     self.first_weights += (derivative * 0.000001)
        #     #     self.second_weights += (derivative * 0.000001)
        #     # else:
        #     #     break
        #
        #     if i % 100000 == 0:
        #         print("result: ", result_output)
        #         print("error: ", square_error)
        #         print("weights1: ", self.first_weights)
        #         print("weights2: ", self.second_weights)
        #         print("derivative: ", derivative)
        #         print()


# output_delta = error * self.sigmoid_derivative(result_output)
# self.first_weights += dot(first_layer, output_delta)
# self.second_weights += dot(first_layer, output_delta)

# print("Output After Training:")
# print(result_output)


neural_network = NeuralNetwork()
# neural_network.init_layers(4, 3)
# neural_network.init_weights()
#
# result = neural_network.feed_forward(array([0.1, 0.1, 0.9, 0.9]).transpose())
# print()
# print()
# print("Result: ")
# print(result)

neural_network.train()
