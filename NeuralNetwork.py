from numpy import random, exp, dot, array, power


class NeuralNetwork:

    def init_layers(self, first_layer_neurons_number, second_layer_neurons_number, third_layer_neurons_number):
        self.input_layer_neuron_number = first_layer_neurons_number
        self.hidden_layer_neuron_number = second_layer_neurons_number
        self.output_layer_neuron_number = third_layer_neurons_number

    def init_weights(self):
        self.first_weights = 2 * random.random((self.hidden_layer_neuron_number, self.input_layer_neuron_number)) - 1
        self.second_weights = 2 * random.random((self.output_layer_neuron_number, self.hidden_layer_neuron_number)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def feed_forward(self, input):
        first_result = self.sigmoid(dot(self.first_weights, input))
        return self.sigmoid(dot(self.second_weights, first_result))

    def train(self):
        input = array([1]).transpose()
        output = array([2]).transpose()

        for i in range(10):
            self.init_layers(1, 1, 1)
            self.init_weights()

            result_output = self.feed_forward(input)

            error = output - result_output
            square_error = power(error, 2)

            print("result: ", result_output)
            print("error: ", square_error)
            print("weights1: ", self.first_weights)
            print("weights2: ", self.second_weights)
            print()
            print()

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
