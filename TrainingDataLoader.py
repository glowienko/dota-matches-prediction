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

    def generateTeams(self, filename, teams_number):
        sleep(1)
        teams = requests.get('https://api.opendota.com/api/teams').json()
        teams_dict = defaultdict(dict)
        for i in range(teams_number):
            item = dict(teams[i])
            team_id = item.pop('team_id')
            teams_dict[team_id].update(item)
            print(item)
        self.save_to_file(filename, teams_dict)

    def generateMatches(self, matches_file, teams_file, teams, matches_per_team):
        teams_dict = defaultdict(dict)
        teams_dict.update(teams)

        matches_dict = defaultdict(dict)
        for team in teams.keys():
            sleep(1)
            matches = requests.get('https://api.opendota.com/api/teams/{}/matches'.format(team)).json()
            for i in range(matches_per_team):
                try:
                    sleep(1)
                    match = requests.get(
                        'https://api.opendota.com/api/matches/{}'.format(matches[i]['match_id'])).json()
                    match_id = match.pop('match_id')
                    match_stat = dict()
                    radiant_team_id = match['radiant_team_id']
                    dire_team_id = match['dire_team_id']
                    match_stat.update({'dire_team_id': dire_team_id})
                    match_stat.update({'radiant_team_id': radiant_team_id})
                    match_stat.update({'radiant_win': match['radiant_win']})

                    if str(radiant_team_id) not in teams_dict:
                        sleep(1)
                        team = requests.get('https://api.opendota.com/api/teams/{}'.format(radiant_team_id)).json()
                        team_id = team.pop('team_id')
                        teams_dict[team_id].update(team)

                    elif str(dire_team_id) not in teams_dict:
                        sleep(1)
                        team = requests.get('https://api.opendota.com/api/teams/{}'.format(dire_team_id)).json()
                        team_id = team.pop('team_id')
                        teams_dict[team_id].update(team)

                    for x in range(10):
                        match_stat.update({'player{}'.format(x): match['players'][x]['account_id']})
                        match_stat.update({'isRadiant{}'.format(x): match['players'][x]['isRadiant']})
                    matches_dict[match_id].update(match_stat)
                    print(match_stat)
                except:
                    continue
        self.save_to_file(matches_file, matches_dict)
        self.save_to_file(teams_file, teams_dict)

    def generatePlayers(self, filename, matches):
        players_dict = defaultdict(dict)
        for match in matches.values():
            for i in range(10):
                try:
                    player_dict = dict()
                    player_id = match['player{}'.format(i)]
                    if player_id in players_dict:
                        continue
                    sleep(1)
                    player = requests.get('https://api.opendota.com/api/players/{}'.format(player_id)).json()
                    sleep(1)
                    wl = requests.get('https://api.opendota.com/api/players/{}/wl?date=50'.format(player_id)).json()
                    player_dict.update({'name': player['profile']['name']})
                    player_dict.update({'mmr': player['mmr_estimate']['estimate']})
                    player_dict.update({'win': wl['win']})
                    player_dict.update({'lose': wl['lose']})
                    players_dict[player_id].update(player_dict)
                    print(player_dict)
                except:
                    continue
        self.save_to_file(filename, players_dict)

    def generateTrainingData(self, filename, matches, players, teams):
        training_sets = list()
        for match in matches.values():
            input_data = list()

            radiant_team = teams.get(str(match['radiant_team_id']))
            input_data.append(self.normalize(2000, 0, radiant_team['rating']))
            input_data.append(radiant_team['wins'] / (radiant_team['losses'] + radiant_team['wins']))
            for i in range(10):
                if not match['isRadiant{}'.format(i)]:
                    continue
                player_id = match['player{}'.format(i)]
                player = players.get(str(player_id))
                input_data.append(self.normalize(10000, 0, player['mmr']))
                input_data.append(player['win'] / (player['lose'] + player['win']))

            dire_team = teams.get(str(match['dire_team_id']))
            input_data.append(self.normalize(2000, 0 ,dire_team['rating']))
            input_data.append(dire_team['wins'] / (dire_team['losses'] + dire_team['losses']))
            for i in range(10):
                if match['isRadiant{}'.format(i)]:
                    continue
                player_id = match['player{}'.format(i)]
                player = players.get(str(player_id))
                input_data.append(self.normalize(10000, 0, player['mmr']))
                input_data.append(player['win'] / (player['lose'] + player['win']))

            result = self.evaluate_match_results(match)
            training_sets.append((input_data, result))
        self.save_to_file(filename, training_sets)

    def normalize(self, oldMax, oldMin, value):
        oldRange = oldMax - oldMin
        newRange = 1
        return (((value - oldMin) * newRange) / oldRange) + 0

    def loadFromFile(self, filename):
        training_data = codecs.open(filename, 'r', encoding='utf-8').read()
        return json.loads(training_data)

    def get_network_input_for_teams(self, first_team, second_team):
        pass

loader = TrainingDataLoader()
loader.generateTeams('teams2', 50)
teams = loader.loadFromFile('teams2')
loader.generateMatches('matches2', 'teams2', teams, 10)
matches = loader.loadFromFile('matches2')
loader.generatePlayers('players2', matches)
players = loader.loadFromFile('players2')
teams = loader.loadFromFile('teams2')
loader.generateTrainingData('training_sets2', matches, players, teams)


# loader.generateTrainingDataFile('teams_map_my_new_1', 'training_data_my_new_1', 'teams_strike_my_new_1')
