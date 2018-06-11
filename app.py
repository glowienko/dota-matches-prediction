from NeuralNetwork import NeuralNetwork
from TrainingDataLoader import TrainingDataLoader
import cProfile
import time
from numpy import array

neural_network = NeuralNetwork()
data_loader = TrainingDataLoader()

training_data = data_loader.loadTrainingSetsFromFile('training_data2')
teams_data = data_loader.loadTeamsFromFile('teams_map2')

print(len(training_data))

neural_network.init_layers(6, 20, 2)
neural_network.init_weights()
neural_network.init_parameters(0.001, 1000, 20)
# neural_network.loadState('long_time')

result1 = neural_network.feed_forward(array([27, 10, 1147.18, 16, 6, 1160.66]))
print('result 1 : \n', result1)

# profiling
# cProfile.run('neural_network.train(training_data)', 'standard_python')

start = time.time()
neural_network.train(array(training_data))
duration = time.time() - start

result2 = neural_network.feed_forward(array([27, 10, 1147.18, 16, 6, 1160.66]))
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
