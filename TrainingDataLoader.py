import codecs
import json
from collections import defaultdict
from time import sleep

import requests


class TrainingDataLoader:
    base_api_url = 'https://api.opendota.com/api'
    pro_matches = '/proMatches'
    teams = '/teams'
    players = '/players'
    last_id = 2944966018
    query = '?less_than_match_id='

    # https_proxy = "193.239.38.229:8080"
    # proxyDict = {
    #     "https": https_proxy
    # }

    def create_team_players_url(self, team_id):
        return self.create_team_url(team_id) + self.players

    def create_team_url(self, team_id):
        return self.base_api_url + self.teams + '/' + str(team_id)

    def save_to_file(self, file_name, object_to_save):
        json.dump(object_to_save, codecs.open(file_name, 'w', encoding='utf-8'))

    def count_winratio(self, player):
        return player['wins'] / player['games_played'] * 100

    def evaluate_match_results(self, match):
        try:
            if match['radiant_win']:
                return [1, 0]
            return [0, 1]
        except:
            return []

    def evaluate_input_layer(self, match):
        try:
            r_team_id = match['radiant_team_id']
            d_team_id = match['dire_team_id']

            r_team_info = requests.get(self.create_team_url(r_team_id)).json()
            d_team_info = requests.get(self.create_team_url(d_team_id)).json()
            r_team_players = requests.get(self.create_team_players_url(r_team_id)).json()
            d_team_players = requests.get(self.create_team_players_url(d_team_id)).json()
            sleep(5)

            filter(lambda x: x['is_current_team_member'] == True, r_team_players)
            filter(lambda x: x['is_current_team_member'] == True, d_team_players)

            return ([
                        r_team_info['wins'],
                        r_team_info['losses'],
                        r_team_info['rating'],
                        self.count_winratio(r_team_players.__next__()),
                        self.count_winratio(r_team_players.__next__()),
                        self.count_winratio(r_team_players.__next__()),
                        self.count_winratio(r_team_players.__next__()),
                        self.count_winratio(r_team_players.__next__()),
                        d_team_info['wins'],
                        d_team_info['losses'],
                        d_team_info['rating'],
                        self.count_winratio(d_team_players.__next__()),
                        self.count_winratio(d_team_players.__next__()),
                        self.count_winratio(d_team_players.__next__()),
                        self.count_winratio(d_team_players.__next__()),
                        self.count_winratio(d_team_players.__next__())
                    ],
                    {
                        r_team_id:
                            (r_team_info['name'],
                             [r_team_info['wins'],
                              r_team_info['losses'],
                              r_team_info['rating']]
                             ),

                        d_team_id:
                            (d_team_info['name'],
                             [d_team_info['wins'],
                              d_team_info['losses'],
                              d_team_info['rating']]
                             )
                    })
        except:
            return []

    def generateTrainingDataFile(self, teamfile, trainingfile, team_strike_file):

        dota_training_data_list = list()
        teams_dict = dict()
        teams_last_games = defaultdict(list)

        for i in range(10000):
            matches = requests.get(self.base_api_url + self.pro_matches + self.query + str(self.last_id)).json()
            sleep(1)

            for index, match in enumerate(matches):
                input_layer = self.evaluate_input_layer(match)
                expected_output = self.evaluate_match_results(match)
                self.last_id = match['match_id']

                if not input_layer or not expected_output:
                    continue

                teams_dict.update(input_layer[1])
                teams_last_games[match['radiant_team_id']].append(match['radiant_win'])
                teams_last_games[match['dire_team_id']].append(not match['radiant_win'])
                training_set = (input_layer[0], expected_output)
                dota_training_data_list.append(training_set)
                self.save_to_file(trainingfile, dota_training_data_list)
                self.save_to_file(teamfile, teams_dict)
                self.save_to_file(team_strike_file, teams_last_games)
                print(index, ': ', training_set)
            print('i:', i)

        print("api fetching ended!")
        self.save_to_file(trainingfile, dota_training_data_list)
        self.save_to_file(teamfile, teams_dict)
        self.save_to_file(team_strike_file, teams_last_games)

    def loadTrainingSetsFromFile(self, filename):
        training_data = codecs.open(filename, 'r', encoding='utf-8').read()
        return json.loads(training_data)

    def loadTeamsFromFile(self, filename):
        team_data = codecs.open(filename, 'r', encoding='utf-8').read()
        return json.loads(team_data)

    def get_network_input_for_teams(self, first_team, second_team):
        pass


#loader = TrainingDataLoader()
#loader.generateTrainingDataFile('teams_map_my_new_1', 'training_data_my_new_1', 'teams_strike_my_new_1')
