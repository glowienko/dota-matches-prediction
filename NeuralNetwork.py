from numpy import random, exp, dot, array, power, zeros, newaxis, concatenate, sqrt
from random import shuffle
import json, codecs
from scipy.special import expit


class NeuralNetwork:

    def init_layers(self, first_layer_neurons_number, second_layer_neurons_number, third_layer_neurons_number):
        self.input_layer_neuron_number = first_layer_neurons_number
        self.hidden_layer_neuron_number = second_layer_neurons_number
        self.output_layer_neuron_number = third_layer_neurons_number

    def init_weights(self):
        self.hidden_weights = 4 * random.random(
            (self.hidden_layer_neuron_number, self.input_layer_neuron_number + 1)) - 2
        self.output_weights = 0 * random.random(
            (self.output_layer_neuron_number, self.hidden_layer_neuron_number + 1))

    def init_parameters(self, eta, epochs, batch_size, beta):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta

    def sigmoid(self, x):
        return expit(x)

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, y0):
        y0 = array(y0)[newaxis].transpose()
        y0 = concatenate((y0, [[1]]))
        s1 = dot(self.hidden_weights, y0)
        y1 = self.sigmoid(s1)
        y1 = concatenate((y1, [[1]]))
        s2 = dot(self.output_weights, y1)
        y2 = s2  # activation function is linear
        return self.normalize(1, 0, y2)

    def backpropagate(self, inputData, expectedOutput):

        #             sig          f.liniowa
        # y0  ->  s1  ->  y1  ->  s2  ->  y2  -> q
        # w1  ->          w2  ->

        y0 = inputData
        y0 = concatenate((y0, [[1]]))
        s1 = dot(self.hidden_weights, y0)
        y1 = self.sigmoid(s1)
        y1 = concatenate((y1, [[1]]))
        s2 = dot(self.output_weights, y1)
        y2 = self.normalize(1, 0, s2)  # activation function is linear

        dq_dy2 = (y2 - expectedOutput)
        dq_ds2 = dq_dy2
        dq_dw2 = dot(dq_ds2, y1.transpose())

        dq_dy1 = dot(self.output_weights.transpose(), dq_ds2)
        dq_ds1 = dq_dy1[:-1] * self.sigmoid_derivative(s1)
        dq_dw1 = dot(dq_ds1, y0.transpose())

        return dq_dw1, dq_dw2

    def updateWeights(self, gradient1, gradient2, momentum1, momentum2):
        momentum1 = momentum1 * self.beta + (1 - self.beta) * gradient1 / self.batch_size
        momentum2 = momentum2 * self.beta + (1 - self.beta) * gradient2 / self.batch_size
        self.hidden_weights = self.hidden_weights - self.eta * momentum1
        self.output_weights = self.output_weights - self.eta * momentum2

    def evaluate(self, train_data):
        success = 0
        failure = 0

        for training_set in train_data:
            out = self.feed_forward(array(training_set[0]))
            if training_set[1][0] > training_set[1][1] and out[0][0] > out[1][0]:
                success += 1
            elif training_set[1][0] < training_set[1][1] and out[0][0] < out[1][0]:
                success += 1
            else:
                failure += 1
        return (success / (failure + success)) * 100

    def train(self, trainData):

        for i in range(self.epochs):
            momentum1 = zeros(self.hidden_weights.shape)
            momentum2 = zeros(self.output_weights.shape)

            if i % 1 == 0:
                print('Epoch: ', i, ' | Score: ', self.evaluate(trainData))
                # print('In progress: ', i)

            random.shuffle(trainData)

            for batch_num in range(0, len(trainData), self.batch_size):

                g1 = zeros(self.hidden_weights.shape)
                g2 = zeros(self.output_weights.shape)

                for data in trainData[batch_num:batch_num + self.batch_size]:
                    inputData = array(data[0])[newaxis].transpose()
                    outputData = array(data[1])[newaxis].transpose()

                    tg1, tg2 = self.backpropagate(inputData, outputData)
                    g1 = g1 + tg1
                    g2 = g2 + tg2

                self.updateWeights(g1, g2, momentum1, momentum2)

    def normalize(self, oldMax, oldMin, value):
        oldRange = oldMax - oldMin
        newRange = sqrt(self.hidden_layer_neuron_number) + sqrt(self.hidden_layer_neuron_number)
        return (((value - oldMin) * newRange) / oldRange) - sqrt(self.hidden_layer_neuron_number)

    def saveState(self, fileName):
        state = {'first_weights': self.hidden_weights.tolist(),
                 'second_weights': self.output_weights.tolist()}
        json.dump(state, codecs.open(fileName, 'w', encoding='utf-8'))

    def loadState(self, fileName):
        state = codecs.open(fileName, 'r', encoding='utf-8').read()
        state = json.loads(state)
        self.hidden_weights = array(state['first_weights'])
        self.output_weights = array(state['second_weights'])
