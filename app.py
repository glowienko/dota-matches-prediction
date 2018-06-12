from NeuralNetwork import NeuralNetwork
from TrainingDataLoader import TrainingDataLoader
import cProfile
import time
from numpy import array

neural_network = NeuralNetwork()
data_loader = TrainingDataLoader()

training_data = data_loader.loadFromFile('training_sets')
teams_data = data_loader.loadFromFile('teams')

print(len(training_data))

neural_network.init_layers(24, 100, 2)
neural_network.init_weights()
neural_network.init_parameters(0.01, 10000, 10, 0.9)
# neural_network.loadState('long_time')

result1 = neural_network.feed_forward(array([1602.65, 1.465625, 7259, 1.221311475409836, 7335, 1.6231884057971016, 7484, 1.7678571428571428, 7463, 1.6818181818181819, 7625, 1.8571428571428572, 1585.75, 1.2222222222222223, 7915, 1.8421052631578947, 7333, 1.2950819672131149, 7917, 1.8421052631578947, 7914, 1.8421052631578947, 7915, 1.8421052631578947]))
print('result 1 : \n', result1)

# profiling
# cProfile.run('neural_network.train(training_data)', 'standard_python')

start = time.time()
neural_network.train(array(training_data))
duration = time.time() - start

neural_network.saveState('Saved_states/momentum')

result2 = neural_network.feed_forward(array([1602.65, 1.465625, 7259, 1.221311475409836, 7335, 1.6231884057971016, 7484, 1.7678571428571428, 7463, 1.6818181818181819, 7625, 1.8571428571428572, 1585.75, 1.2222222222222223, 7915, 1.8421052631578947, 7333, 1.2950819672131149, 7917, 1.8421052631578947, 7914, 1.8421052631578947, 7915, 1.8421052631578947]))
print('result 2 : \n', result2)

print('Duration: ', duration)
print('Lets predicate some matches!')
# while True:
#     print('1. Show available teams')
#
#     choice = input('Choice:')
#     if choice == '1':
#         template = '{0:10}{1:10}{2:30}'
#         for key, value in teams_data.items():
#             print(template.format(key, ':', str(value)))
