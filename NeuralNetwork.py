from numpy import random, exp, dot, array, power, zeros, newaxis
from random import shuffle
import json, codecs


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

    def init_biases(self):
        self.second_biases = 2 * random.random((self.hidden_layer_neuron_number, 1)) - 1
        self.third_biases = 2 * random.random((self.output_layer_neuron_number, 1)) - 1

    def init_parameters(self, eta, epochs, batch_size):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size

    def feed_forward(self, inputData):
        inputData = array(inputData)[newaxis].transpose()
        first_result = sigmoid(dot(self.first_weights, inputData) + self.second_biases)
        result = sigmoid(dot(self.second_weights, first_result) + self.third_biases)
        return result

    def backpropagate(self, inputData, outputData):
        # g2 = [zeros(weight.shape) for weight in self.second_weights]
        # g1 = [zeros(weight.shape) for weight in self.first_weights]

        A1 = inputData
        Z1 = dot(self.first_weights, A1) + self.second_biases
        A2 = sigmoid(Z1)
        Z2 = dot(self.second_weights, A2) + self.third_biases
        A3 = sigmoid(Z2)

        d2 = (A3 - outputData) * sigmoid_derivative(Z2)
        g2 = dot(d2, A2.transpose())

        d1 = dot(self.second_weights.transpose(), d2) * sigmoid_derivative(Z1)
        g1 = dot(d1, A1.transpose())

        return g1, g2, d1, d2

    def updateWeights(self, gradient1, gradient2, biases2, biases3):
        for i in range(len(self.first_weights)):
            for x in range(len(self.first_weights[i])):
                self.first_weights[i, x] = self.first_weights[i, x] - self.eta * gradient1[i, x] / self.batch_size

        for i in range(len(self.second_weights)):
            for x in range(len(self.second_weights[i])):
                self.second_weights[i, x] = self.second_weights[i, x] - self.eta * gradient2[i, x] / self.batch_size

        self.second_biases = self.second_biases - self.eta * biases2 / self.batch_size
        self.third_biases = self.third_biases - self.eta * biases3 / self.batch_size

    def train(self, trainData):

        for i in range(self.epochs):

            shuffle(trainData)

            for batch_num in range(0, len(trainData), self.batch_size):

                g1 = zeros(self.first_weights.shape)
                g2 = zeros(self.second_weights.shape)
                b2 = zeros(self.second_biases.shape)
                b3 = zeros(self.third_biases.shape)

                for data in trainData[batch_num:batch_num + self.batch_size]:
                    inputData = array(data[0])[newaxis].transpose()
                    outputData = array(data[1])[newaxis].transpose()

                    tg1, tg2, tb2, tb3 = self.backpropagate(inputData, outputData)
                    g1 = g1 + tg1
                    g2 = g2 + tg2
                    b2 = b2 + tb2
                    b3 = b3 + tb3

                self.updateWeights(g1, g2, b2, b3)

    def saveState(self, fileName):
        state = {'first_weights': self.first_weights.tolist(),
                 'second_weights': self.second_weights.tolist(),
                 'second_biases': self.second_biases.tolist(), 'third_biases': self.third_biases.tolist()}
        json.dump(state, codecs.open(fileName, 'w', encoding='utf-8'))

    def loadState(self, fileName):
        state = codecs.open(fileName, 'r', encoding='utf-8').read()
        state = json.loads(state)
        self.first_weights = array(state['first_weights'])
        self.second_weights = array(state['second_weights'])
        self.second_biases = array(state['second_biases'])
        self.third_biases = array(state['third_biases'])
