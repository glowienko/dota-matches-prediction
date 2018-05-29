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
        first_result = sigmoid(dot(self.first_weights, inputData))
        result = sigmoid(dot(self.second_weights, first_result))
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

    def updateWeights(self, gradient1, gradient2, eta):
        for i in range(len(self.first_weights)):
            for x in range(len(self.first_weights[i])):
                self.first_weights[i, x] = self.first_weights[i, x] - eta * gradient1[i, x]

        for i in range(len(self.second_weights)):
            for x in range(len(self.second_weights[i])):
                self.second_weights[i, x] = self.second_weights[i, x] - eta * gradient2[i, x]

    def train(self):

        for i in range(300000):
            inputData = array([[0]])
            outputData = array([[1], [0], [0]])


            g1, g2 = self.backpropagate(inputData, outputData)
            self.updateWeights(g1, g2, 0.001)

            inputData = array([[1]])
            outputData = array([[0], [1], [0]])

            g1, g2 = self.backpropagate(inputData, outputData)
            self.updateWeights(g1, g2, 0.001)

            inputData = array([[2]])
            outputData = array([[0], [0], [1]])

            g1, g2 = self.backpropagate(inputData, outputData)
            self.updateWeights(g1, g2, 0.001)


neural_network = NeuralNetwork()

neural_network.init_layers(1, 10, 3)
neural_network.init_weights()

result1a = neural_network.feed_forward([[0]])
result1b = neural_network.feed_forward([[1]])
result1c = neural_network.feed_forward([[2]])
print(f'Result1:\n'
      f'0: {result1a}\n'
      f'1: {result1b}\n'
      f'2: {result1c}')

neural_network.train()

result2a = neural_network.feed_forward([[0]])
result2b = neural_network.feed_forward([[1]])
result2c = neural_network.feed_forward([[2]])
print(f'Result2:\n'
      f'0: {result2a}\n'
      f'1: {result2b}\n'
      f'2: {result2c}')



