import codecs
import json

import requests


class TrainingDataLoader:
    base_api_url = 'https://api.opendota.com/api'
    pro_matches = '/proMatches'  # list of matches - playing teams id, match duration, match result
    teams = '/teams'  # list of teams - id, rating, wins, losses

    def evaluate_match_results(self, match):
        result = match['radiant_win']

        if result:
            return [1, 0]
        return [0, 1]

    def evaluate_input_layer(self, match):
        r_team_id = match['radiant_team_id']
        d_team_id = match['dire_team_id']

        try:
            r_team_info = requests.get(self.base_api_url + self.teams + '/' + str(r_team_id)).json()
            d_team_info = requests.get(self.base_api_url + self.teams + '/' + str(d_team_id)).json()

            return [r_team_info['wins'],
                    d_team_info['wins'],
                    r_team_info['losses'],  # wins and losses around may be like 1k or more
                    d_team_info['losses'],
                    r_team_info['rating'],
                    d_team_info['rating']  # may be over 1000
                    # match['radiant_score'],  # be careful with all params !
                    # match['dire_score']  # may need normalization !!1
                    ]
        except:
            return []

    def save_to_file(self, file_name, object_to_save):
        # first we should write line with : y0 = 8, y1 = eg. 20, y2 = 2
        json.dump(object_to_save, codecs.open(file_name, 'w', encoding='utf-8'))
        print('training sets saved to file!')

    def generateTrainingDataFile(self):
        matches = requests.get(self.base_api_url + self.pro_matches).json()
        dota_training_data_list = list()

        for index, match in enumerate(matches):
            input_layer = self.evaluate_input_layer(match)
            expected_output = self.evaluate_match_results(match)

            if not input_layer:
                continue

        training_set = (input_layer, expected_output)
        dota_training_data_list.append(training_set)
        print(index, ': ', training_set)

        print("api fetching ended!")
        self.save_to_file("dota_training_set", dota_training_data_list)

    def loadTrainingSetsFromFile(self, filename):
        training_data = codecs.open(filename, 'r', encoding='utf-8').read()
        return json.loads(training_data)
