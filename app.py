from NeuralNetwork import NeuralNetwork
from TrainingDataLoader import TrainingDataLoader
import cProfile
import time
from numpy import array

neural_network = NeuralNetwork()
data_loader = TrainingDataLoader()

training_data = data_loader.loadFromFile('training_sets2')
teams_data = data_loader.loadFromFile('teams2')
matches_data = data_loader.loadFromFile('matches2')

print(len(training_data))
print(len(teams_data))
print(len(matches_data))

neural_network.init_layers(24, 100, 2)
neural_network.init_weights()
neural_network.init_parameters(0.01, 10000, 10, 0.9)
# neural_network.loadState('long_time')

result1 = neural_network.feed_forward(array([0.8013250000000001, 0.5944233206590621, 0.7259, 0.5498154981549815, 0.7335, 0.6187845303867403, 0.7484, 0.6387096774193548, 0.7463, 0.6271186440677966, 0.7625, 0.65, 0.792875, 0.6111111111111112, 0.7915, 0.6481481481481481, 0.7333, 0.5642857142857143, 0.7917, 0.6481481481481481, 0.7914, 0.6481481481481481, 0.7915, 0.6481481481481481]))
print('result 1 : \n', result1)

# profiling
# cProfile.run('neural_network.train(training_data)', 'standard_python')

start = time.time()
neural_network.train(array(training_data))
duration = time.time() - start

neural_network.saveState('Saved_states/momentum')

result2 = neural_network.feed_forward(array([0.8013250000000001, 0.5944233206590621, 0.7259, 0.5498154981549815, 0.7335, 0.6187845303867403, 0.7484, 0.6387096774193548, 0.7463, 0.6271186440677966, 0.7625, 0.65, 0.792875, 0.6111111111111112, 0.7915, 0.6481481481481481, 0.7333, 0.5642857142857143, 0.7917, 0.6481481481481481, 0.7914, 0.6481481481481481, 0.7915, 0.6481481481481481]))
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
