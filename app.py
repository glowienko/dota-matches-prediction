from NeuralNetwork import NeuralNetwork
from TrainingDataLoader import TrainingDataLoader
import time
from numpy import array
import tkinter as tk
from tkinter import ttk

neural_network = NeuralNetwork()
data_loader = TrainingDataLoader()

training_data = data_loader.loadFromFile('training_sets2')
teams_data = data_loader.loadFromFile('teams2')
players_data = data_loader.loadFromFile('players2')

neural_network.init_layers(24, 50, 2)
neural_network.init_weights()
neural_network.init_parameters(0.001, 500, 10, 0.9)


start = time.time()
neural_network.train(array(training_data))
duration = time.time() - start

#
# print('Duration: ', duration)
# print('Lets predicate some matches!')
#
# teams_choices = ['{}.{}'.format(key, value['name']) for key, value in teams_data.items()]
# players_choices = ['{}.{}'.format(key, value['name']) for key, value in players_data.items()]
#
#
# def make_player_combo(r, c, players, root_window):
#     player_combo = ttk.Combobox(root_window, values=players, state='readonly')
#     player_combo.grid(row=r, column=c)
#     return player_combo
#
#
# root = tk.Tk()
# root.title('Dota2 matches predictions')
#
# team_label_a = tk.Label(root, text='Team A:')
# team_label_a.grid(row=0, column=0)
#
# default_team_a = tk.StringVar()
# default_team_a.set(teams_choices[0])
# team_combo_a = ttk.Combobox(root, values=teams_choices, textvariable=default_team_a, state='readonly')
# team_combo_a.grid(row=0, column=1)
#
# team_label_b = tk.Label(root, text='Team B:')
# team_label_b.grid(row=0, column=2)
#
# default_team_b = tk.StringVar()
# default_team_b.set(teams_choices[1])
# team_combo_b = ttk.Combobox(root, values=teams_choices, textvariable=default_team_b, state='readonly')
# team_combo_b.grid(row=0, column=3)
#
# player_label_a = tk.Label(root, text='Players A:')
# player_label_a.grid(row=1, column=0)
#
# a_0 = make_player_combo(1, 1, players_choices, root)
# a_1 = make_player_combo(2, 1, players_choices, root)
# a_2 = make_player_combo(3, 1, players_choices, root)
# a_3 = make_player_combo(4, 1, players_choices, root)
# a_4 = make_player_combo(5, 1, players_choices, root)
#
# radiant_players = [a_0, a_1, a_2, a_3, a_4]
#
# player_label_b = tk.Label(root, text='Players B:')
# player_label_b.grid(row=1, column=2)
# b_0 = make_player_combo(1, 3, players_choices, root)
# b_1 = make_player_combo(2, 3, players_choices, root)
# b_2 = make_player_combo(3, 3, players_choices, root)
# b_3 = make_player_combo(4, 3, players_choices, root)
# b_4 = make_player_combo(5, 3, players_choices, root)
#
# dire_players = [b_0, b_1, b_2, b_3, b_4]
#
# space_label = tk.Label(root, text='--------')
# space_label.grid(row=6, column=2)
#
# result_title = tk.Label(root, text='Result:')
# result_title.grid(row=7, column=2)
#
# result_label = tk.Label(root)
# result_label.grid(row=7, column=3)
#
#
# def check():
#     input_data = list()
#
#     try:
#         radiant_team = team_combo_a.get().split('.')[0]
#         radiant_team = teams_data[radiant_team]
#         input_data.append(data_loader.normalize(2000, 0, radiant_team['rating']))
#         input_data.append(radiant_team['wins'] / (radiant_team['losses'] + radiant_team['wins']))
#         for i in radiant_players:
#             player_id = i.get().split('.')[0]
#             player = players_data.get(str(player_id))
#             input_data.append(data_loader.normalize(10000, 0, player['mmr']))
#             input_data.append(player['win'] / (player['lose'] + player['win']))
#
#         dire_team = team_combo_b.get().split('.')[0]
#         dire_team = teams_data[dire_team]
#         input_data.append(data_loader.normalize(2000, 0, dire_team['rating']))
#         input_data.append(dire_team['wins'] / (dire_team['losses'] + dire_team['wins']))
#         for i in dire_players:
#             player_id = i.get().split('.')[0]
#             player = players_data.get(str(player_id))
#             input_data.append(data_loader.normalize(10000, 0, player['mmr']))
#             input_data.append(player['win'] / (player['lose'] + player['win']))
#
#         result = neural_network.feed_forward(input_data)
#         result_label.config(text=str(result))
#     except:
#         result_label.config(text='Wrong input!')
#
#
# check_button = tk.Button(root, text='Check!', command=check)
# check_button.grid(row=7, column=1)
#
# root.mainloop()
