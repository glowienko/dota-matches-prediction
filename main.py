import numpy as num_py
from NeuralNetwork import NeuralNetwork

def sigmoid(x):
    return 1 / (1 + num_py.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


num_py.random.seed(1)

trainingInput = num_py.array([[0, 0, 1],
                              [0, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]])

expectedTrainingOutput = num_py.array([[0, 0, 1, 1]]).transpose()

weights = 2 * num_py.random.random((3, 1)) - 1

firstLayer = 0
resultOutput = 0
for i in range(10000):
    firstLayer = trainingInput
    resultOutput = sigmoid(num_py.dot(firstLayer, weights))  # forward propagation

    firstLayerError = expectedTrainingOutput - resultOutput

    outputDelta = firstLayerError * sigmoid_derivative(resultOutput)

    weights += num_py.dot(firstLayer.transpose(), outputDelta)  # update weights

print("Output After Training:")
print(resultOutput)
