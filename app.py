from NeuralNetwork import NeuralNetwork
from TrainingDataLoader import TrainingDataLoader

neural_network = NeuralNetwork()
data_loader = TrainingDataLoader()

data_loader.generateTrainingDataFile()
training_data = data_loader.loadTrainingSetsFromFile('dota_training_set')

print(training_data)

neural_network.init_layers(8, 28, 2)
neural_network.init_weights()
neural_network.init_parameters(0.1, 50000, 10)

result1 = neural_network.feed_forward([4, 6, 2, 5, 1043.88, 1029.81, 31, 22])
print('result 1 : \n', result1)

neural_network.train(training_data)

result2 = neural_network.feed_forward([4, 6, 2, 5, 1043.88, 1029.81, 31, 22])
print('result 2 : \n', result2)

print('Lets predicate some matches!')
print('Give first team name')
first_team = input()
print('Give second team name')
second_team = input()

network_input = data_loader.get_network_input_for_teams(first_team, second_team)
results = neural_network.feed_forward(network_input)

print('Your match predicion:\n')
print('team: ', first_team, ' has ', results[0], '% for win:\n')
print('team: ', second_team, ' has ', results[1], '% for win:\n')


# trainInput = [[1, 0, 0, 0, 0, 0],
#               [1, 1, 0, 0, 0, 0],
#               [1, 0, 0, 1, 0, 0],
#               [1, 0, 0, 1, 1, 0],
#               [1, 0, 0, 0, 1, 0],
#               [1, 1, 0, 1, 0, 0],
#               [1, 1, 0, 1, 1, 0],
#               [1, 1, 0, 0, 1, 0],
#               [0, 1, 0, 1, 0, 0],
#               [0, 1, 0, 1, 1, 0],
#               [1, 0, 1, 0, 0, 0],
#               [1, 1, 1, 0, 0, 0],
#               [1, 0, 1, 1, 0, 0],
#               [1, 0, 1, 1, 1, 0],
#               [1, 0, 1, 0, 1, 0],
#               [1, 1, 1, 1, 0, 0],
#               [1, 1, 1, 1, 1, 0],
#               [1, 1, 1, 0, 1, 0],
#               [0, 1, 1, 1, 0, 0],
#               [0, 1, 1, 1, 1, 0],
#               [1, 0, 1, 0, 0, 1],
#               [1, 1, 1, 0, 0, 1],
#               [0, 1, 0, 1, 1, 1],
#               [1, 0, 1, 1, 0, 1],
#               [1, 0, 1, 1, 1, 1],
#               [1, 0, 1, 0, 1, 1]]
#
# trainOutput = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
#
# trainData = [(x, y) for x, y in zip(trainInput, trainOutput)]
#
# neural_network.init_layers(6, 30, 26)
# neural_network.init_weights()
# neural_network.init_parameters(0.01, 100000, 26)
#
# result1a = neural_network.feed_forward([1, 1, 1, 0, 0, 0])
# print("------------------------\n"
#       "result1a: ", result1a)
#
# neural_network.train(trainData)
#
# result2a = neural_network.feed_forward([1, 1, 1, 0, 0, 0])
# print("------------------------\n"
#       "result2a: ", result2a)


# trainData = [([0], [1, 0, 0]), ([1], [0, 1, 0]), ([2], [0, 0, 1])]
#
# neural_network.init_layers(1, 2, 3)
# neural_network.init_weights()
# neural_network.init_parameters(0.1, 30000, 3)
# neural_network.saveState('przed')
#
# # neural_network.loadState('init')
#
# result1a = neural_network.feed_forward([0])
# result1b = neural_network.feed_forward([1])
# result1c = neural_network.feed_forward([2])
# print(f'Result1:\n'
#       f'0: {result1a}\n'
#       f'1: {result1b}\n'
#       f'2: {result1c}')
#
# neural_network.train(trainData)
# # neural_network.saveState('init')
# neural_network.saveState('po')
#
# result2a = neural_network.feed_forward([0])
# result2b = neural_network.feed_forward([1])
# result2c = neural_network.feed_forward([2])
# print(f'Result2:\n'
#       f'0: {result2a}\n'
#       f'1: {result2b}\n'
#       f'2: {result2c}')
