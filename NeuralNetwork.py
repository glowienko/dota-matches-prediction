from numpy import random, exp, dot, array, power, zeros, newaxis, concatenate
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
        self.hidden_weights = 2 * random.random(
            (self.hidden_layer_neuron_number, self.input_layer_neuron_number + 1)) - 1
        self.output_weights = 2 * random.random(
            (self.output_layer_neuron_number, self.hidden_layer_neuron_number + 1)) - 1

    def init_parameters(self, eta, epochs, batch_size):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size

    def feed_forward(self, y0):
        y0 = array(y0)[newaxis].transpose()
        y0 = concatenate((y0, [[1]]))
        s1 = dot(self.hidden_weights, y0)
        y1 = sigmoid(s1)
        y1 = concatenate((y1, [[1]]))
        s2 = dot(self.output_weights, y1)
        y2 = s2  # activation function is linear
        return y2

    def backpropagate(self, inputData, expectedOutput):

        #             sig          f.liniowa
        # y0  ->  s1  ->  y1  ->  s2  ->  y2  -> q
        # w1  ->          w2  ->

        y0 = inputData
        y0 = concatenate((y0, [[1]]))
        s1 = dot(self.hidden_weights, y0)
        y1 = sigmoid(s1)
        y1 = concatenate((y1, [[1]]))
        s2 = dot(self.output_weights, y1)
        y2 = s2  # activation function is linear

        dq_dy2 = (y2 - expectedOutput)
        dq_ds2 = dq_dy2
        dq_dw2 = dot(dq_ds2, y1.transpose())

        dq_dy1 = dot(self.output_weights.transpose(), dq_ds2)
        dq_ds1 = dq_dy1[:-1] * sigmoid_derivative(s1)
        dq_dw1 = dot(dq_ds1, y0.transpose())

        return dq_dw1, dq_dw2

    def updateWeights(self, gradient1, gradient2):
        for i in range(len(self.hidden_weights)):
            for x in range(len(self.hidden_weights[i])):
                self.hidden_weights[i, x] = self.hidden_weights[i, x] - self.eta * gradient1[i, x] / self.batch_size

        for i in range(len(self.output_weights)):
            for x in range(len(self.output_weights[i])):
                self.output_weights[i, x] = self.output_weights[i, x] - self.eta * gradient2[i, x] / self.batch_size

    def train(self, trainData):

        for i in range(self.epochs):

            if i % 10000 == 0:
                print('In progress: ', i)

            shuffle(trainData)

            for batch_num in range(0, len(trainData), self.batch_size):

                g1 = zeros(self.hidden_weights.shape)
                g2 = zeros(self.output_weights.shape)

                for data in trainData[batch_num:batch_num + self.batch_size]:
                    inputData = array(data[0])[newaxis].transpose()
                    outputData = array(data[1])[newaxis].transpose()

                    tg1, tg2 = self.backpropagate(inputData, outputData)
                    g1 = g1 + tg1
                    g2 = g2 + tg2

                self.updateWeights(g1, g2)

    def saveState(self, fileName):
        state = {'first_weights': self.hidden_weights.tolist(),
                 'second_weights': self.output_weights.tolist()}
        json.dump(state, codecs.open(fileName, 'w', encoding='utf-8'))

    def loadState(self, fileName):
        state = codecs.open(fileName, 'r', encoding='utf-8').read()
        state = json.loads(state)
        self.hidden_weights = array(state['first_weights'])
        self.output_weights = array(state['second_weights'])
