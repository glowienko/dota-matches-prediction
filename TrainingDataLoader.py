import codecs
import json
from time import sleep

import requests


class TrainingDataLoader:
    base_api_url = 'https://api.opendota.com/api'
    pro_matches = '/proMatches'  # list of matches - playing teams id, match duration, match result
    teams = '/teams'  # list of teams - id, rating, wins, losses
    last_id = 3942922478
    query = '?less_than_match_id='

    def evaluate_match_results(self, match):
        try:
            result = match['radiant_win']

            if result:
                return [1, 0]
            return [0, 1]
        except:
            return []

    def evaluate_input_layer(self, match):
        try:
            r_team_id = match['radiant_team_id']
            d_team_id = match['dire_team_id']
            r_team_info = requests.get(self.base_api_url + self.teams + '/' + str(r_team_id)).json()
            sleep(1)
            d_team_info = requests.get(self.base_api_url + self.teams + '/' + str(d_team_id)).json()
            sleep(1)

            return ([
                        r_team_info['wins'],
                        d_team_info['wins'],
                        r_team_info['losses'],
                        d_team_info['losses'],
                        r_team_info['rating'],
                        d_team_info['rating']],
                    {
                        r_team_id:
                            (r_team_info['name'],
                             [r_team_info['wins'],
                              r_team_info['losses'],
                              r_team_info['rating']]),

                        d_team_id:
                            (d_team_info['name'],
                             [d_team_info['wins'],
                              d_team_info['losses'],
                              d_team_info['rating']])
                    })
        except:
            return []

    def save_to_file(self, file_name, object_to_save):
        # first we should write line with : y0 = 8, y1 = eg. 20, y2 = 2
        json.dump(object_to_save, codecs.open(file_name, 'w', encoding='utf-8'))
        print('training sets saved to file!')

    def generateTrainingDataFile(self, teamfile, trainingfile):

        dota_training_data_list = list()
        name_id_dict = dict()

        for i in range(1):
            matches = requests.get(
                self.base_api_url + self.pro_matches + self.query + str(self.last_id - i * 100000)).json()
            sleep(1)

            for index, match in enumerate(matches):
                input_layer = self.evaluate_input_layer(match)
                expected_output = self.evaluate_match_results(match)

                if not input_layer:
                    continue
                if not expected_output:
                    continue

                training_set = (input_layer[0], expected_output)
                dota_training_data_list.append(training_set)
                name_id_dict.update(input_layer[1])
                print(index, ': ', training_set)

        print("api fetching ended!")
        self.save_to_file(trainingfile, dota_training_data_list)
        self.save_to_file(teamfile, name_id_dict)

    def loadTrainingSetsFromFile(self, filename):
        training_data = codecs.open(filename, 'r', encoding='utf-8').read()
        return json.loads(training_data)

    def loadTeamsFromFile(self, filename):
        team_data = codecs.open(filename, 'r', encoding='utf-8').read()
        return json.loads(team_data)

    def get_network_input_for_teams(self, first_team, second_team):
        pass

# loader = TrainingDataLoader()
# loader.generateTrainingDataFile('short_team', 'short_training')
