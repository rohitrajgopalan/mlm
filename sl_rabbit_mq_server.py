import json
import numpy as np
import pandas as pd
import pickle
from os.path import join, realpath, dirname, isfile, isdir
from os import mkdir
from datetime import datetime
from sklearn.metrics import mean_absolute_error

from rabbit_mq_server import RabbitMQServer
from mlm_utils import *


class SLRabbitMQServer(RabbitMQServer):
    message_types = ['text_messages', 'tactical_graphics', 'sos', 'blue_spots', 'red_spots']
    context_types = ['sos_operational_context', 'distance_to_enemy_context', 'distance_to_enemy_aggregator']

    result_cols = ['combination_id', 'regressor', 'pre_processing_type', 'use_default_params', 'num_runs', 'mae']
    results_dir = join(dirname(realpath('__file__')), 'results')
    datasets_dir = join(dirname(realpath('__file__')), 'datasets')

    models = {
        'sos_operational_context': {
            'model': None,
            'current_combination': -1,
            'features': ['Seconds Since Last Sent SOS'],
            'label': 'Multiplier',
            'cols': ['Seconds Since Last Sent SOS', 'Multiplier'],
            'cols_to_json': {
                'Seconds Since Last Sent SOS': 'secondsSinceLastSentSOS',
                'Multiplier': 'multiplier'
            },
            'actual_values': [],
            'predicted_values': []
        },
        'distance_to_enemy_context': {
            'model': None,
            'current_combination': -1,
            'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Multiplier',
            'cols': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                     '#5 Nearest', 'Multiplier'],
            'cols_to_json': {},
            'actual_values': [],
            'predicted_values': []
        },
        'distance_to_enemy_aggregator': {
            'model': None,
            'current_combination': -1,
            'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Multiplier',
            'cols': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                     '#5 Nearest', 'Multiplier'],
            'cols_to_json': {},
            'actual_values': [],
            'predicted_values': []
        },
        'text_messages': {
            'model': None,
            'current_combination': -1,
            'features': ['Age of Message'],
            'label': 'Penalty',
            'cols': ['Age of Message', 'Penalty'],
            'cols_to_json': {
                'Age of Message': 'ageOfMessage',
                'Penalty': 'penalty',
            },
            'actual_values': [],
            'predicted_values': []
        },
        'tactical_graphics': {
            'model': None,
            'current_combination': -1,
            'features': ['Age of Message'],
            'label': 'Score (Lazy)',
            'cols': ['Age of Message', 'Score (Lazy)'],
            'cols_to_json': {
                'Age of Message': 'ageOfMessage',
                'Score (Lazy)': 'score',
            },
            'actual_values': [],
            'predicted_values': []
        },
        'sos': {
            'model': None,
            'current_combination': -1,
            'features': ['Age of Message', 'Number of blue Nodes'],
            'label': 'Score',
            'cols': ['Age of Message', 'Number of blue Nodes', 'Score'],
            'cols_to_json': {
                'Age of Message': 'ageOfMessage',
                'Number of blue Nodes': 'numBlueNodes',
                'Score': 'score'
            },
            'actual_values': [],
            'predicted_values': []
        },
        'blue_spots': {
            'model': None,
            'current_combination': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance', 'Is Affected'],
            'label': 'Score',
            'cols': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                     'Average Hierarchical distance', 'Is Affected', 'Score'],
            'cols_to_json': {
                'Distance since Last Update': 'distanceSinceLastUpdate',
                'Number of blue Nodes': 'numBlueNodes',
                'Average Distance': 'averageDistance',
                'Average Hierarchical distance': 'averageHierarchicalDistance',
                'Is Affected': 'isAffected',
                'Score': 'score'
            },
            'actual_values': [],
            'predicted_values': []
        },
        'red_spots': {
            'model': None,
            'current_combination': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance', 'Is Affected'],
            'label': 'Score',
            'cols': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                     'Average Hierarchical distance', 'Is Affected', 'Score'],
            'cols_to_json': {
                'Distance since Last Update': 'distanceSinceLastUpdate',
                'Number of blue Nodes': 'numBlueNodes',
                'Average Distance': 'averageDistance',
                'Average Hierarchical distance': 'averageHierarchicalDistance',
                'Is Affected': 'isAffected',
                'Score': 'score'
            },
            'actual_values': [],
            'predicted_values': []
        }
    }

    MODEL_CREATION = 'SET_MODEL'
    COST = 'UPDATE_MODEL'
    RESULTS = 'WRITE_RESULTS'
    TRAIN = 'TRAIN_DATA'
    DATA = 'EXPORT_DATA'
    SETUP = 'SETUP_DATA'

    def __init__(self):
        super().__init__(queue_name='sl_request_queue', durable=True)
        self.combinations = get_scikit_model_combinations()
        self.writing_results = False
        self.writing_data = False
        self.num_runs_index = self.result_cols.index("num_runs") - 1
        self.mae_index = self.result_cols.index("mae") - 1
        for model_type in self.models:
            self.models[model_type].update({'results': pd.read_csv(
                join(self.results_dir, '{0}.csv'.format(model_type)), index_col=self.result_cols[0])})
        self.setup_data()

    def setup_data(self):
        if not isdir(self.datasets_dir):
            mkdir(self.datasets_dir)

        for model_type in self.models:
            self.models[model_type].update(
                {'data': pd.DataFrame(columns=self.models[model_type]['cols']), 'data_tuples': []})
            datasets_dir_model_type = join(self.datasets_dir, model_type)
            if not isdir(datasets_dir_model_type):
                mkdir(datasets_dir_model_type)

    def export_data(self):
        for model_type in self.models:
            if len(self.models[model_type]['data_tuples']) > 0:
                self.models[model_type]['data'].to_csv(join(self.datasets_dir, model_type,
                                                            '{0}_{1}.csv'.format(model_type,
                                                                                 datetime.now().strftime("%Y%m%d%H%M%S"))),
                                                       index=False)

    def _get_reply(self, request):
        request_body = request['requestBody']
        request_type = request['requestType']

        print(request_body)

        if request_type == self.COST:
            self.writing_results = False
            self.writing_data = False

            message_type = request_body['messageType'].lower()
            new_data_row = {}
            features = self.models[message_type]['features']
            cols_to_json = self.models[message_type]['cols_to_json']
            test_input = []
            for feature in features:
                if feature in cols_to_json:
                    feature_value = request_body[cols_to_json[feature]]
                    new_data_row.update({feature: feature_value})
                    test_input.append(feature_value)

            actual_score = calculate_raw_score(message_type, new_data_row)
            predicted_score = self.models[message_type]['model'].predict(np.array([test_input]))[0]

            self.models[message_type]['actual_values'].append(actual_score)
            self.models[message_type]['predicted_values'].append(predicted_score)

            if message_type == 'sos':
                return predicted_score
            else:
                doe_multiplier = 1
                sos_multiplier = 1

                seconds_since_last_sent_sos = request_body['secondsSinceLastSOS']
                if seconds_since_last_sent_sos != 1e6:
                    actual_sos_multiplier = calculate_sos_operational_context_mutliplier(
                        seconds_since_last_sent_sos)
                    predicted_sos_multiplier = self.models['sos_operational_context']['model'].predict(
                        np.array([seconds_since_last_sent_sos]).reshape(-1, 1))[0]
                    self.models['sos_operational_context']['actual_values'].append(actual_sos_multiplier)
                    self.models['sos_operational_context']['predicted_values'].append(predicted_sos_multiplier)
                    sos_multiplier = predicted_sos_multiplier

                nearest_values = request_body['nearestValues']
                nearest_values = np.array(nearest_values)
                nearest_values = np.sort(nearest_values)
                nearest_values = nearest_values[:5]

                if 1e6 not in nearest_values:
                    new_data_row = {}
                    for i in range(5):
                        new_data_row.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                    if message_type == 'red_spots':
                        actual_aggregate_multiplier = calculate_distance_to_enemy_aggregator(nearest_values)
                        predicted_aggregate_multiplier = \
                            self.models['distance_to_enemy_aggregator']['model'].predict(nearest_values.reshape(-1, 5))[
                                0]
                        self.models['distance_to_enemy_aggregator']['actual_values'].append(actual_aggregate_multiplier)
                        self.models['distance_to_enemy_aggregator']['predicted_values'].append(
                            predicted_aggregate_multiplier)
                        doe_multiplier = predicted_aggregate_multiplier
                    else:
                        actual_context_multiplier = calculate_distance_to_enemy_aggregator(nearest_values)
                        predicted_context_multiplier = \
                            self.models['distance_to_enemy_context']['model'].predict(nearest_values.reshape(-1, 5))[0]

                        self.models['distance_to_enemy_context']['actual_values'].append(actual_context_multiplier)
                        self.models['distance_to_enemy_context']['predicted_values'].append(
                            predicted_context_multiplier)
                        doe_multiplier = predicted_context_multiplier

                return predicted_score * sos_multiplier * doe_multiplier
        elif request_type == self.SETUP:
            self.writing_results = False
            self.writing_data = False
            self.setup_data()
            return 1

        elif request_type == self.MODEL_CREATION:
            self.writing_results = False
            self.writing_data = False

            combination_id = int(request_body['combinationID'])
            if 0 <= combination_id < len(self.combinations):
                for model_type in self.models:
                    self.models[model_type].update({'results': pd.read_csv(
                        join(self.results_dir, '{0}.csv'.format(model_type)), index_col=self.result_cols[0])})
                    if self.models[model_type]['current_combination'] == -1 or self.models[model_type][
                        'current_combination'] != combination_id:
                        model_name = join(dirname(realpath('__file__')), 'models', model_type,
                                          '{0}.pkl'.format(combination_id))
                        self.models[model_type].update(
                            {'model': pickle.load(open(model_name, 'rb')), 'current_combination': combination_id})
                        self.models[model_type].update({'actual_values': [], 'predicted_values': []})
                return 1
            else:
                return 0

        elif request_type == self.RESULTS:
            if not self.writing_results:
                self.writing_results = True
                self.write_results()
            return 1

        elif request_type == self.TRAIN:

            message_type = request_body['messageType'].lower()

            nearest_values = request_body['nearestValues']
            nearest_values = np.array(nearest_values)
            nearest_values = np.sort(nearest_values)

            doe_multiplier = 1
            sos_multiplier = 1

            if 1e6 not in nearest_values:
                nearest_values = nearest_values[:5]
                new_data_row_distance_to_enemy = {}

                for i in range(5):
                    new_data_row_distance_to_enemy.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})

                if message_type == 'red_spots':
                    distance_to_enemy_aggregator = calculate_distance_to_enemy_aggregator(nearest_values)
                    new_data_row_distance_to_enemy.update({'Multiplier': distance_to_enemy_aggregator})
                    distance_to_enemy_aggregator_tuple = (
                        nearest_values[0], nearest_values[1], nearest_values[2], nearest_values[3], nearest_values[4],
                        distance_to_enemy_aggregator)
                    doe_multiplier = distance_to_enemy_aggregator
                    if distance_to_enemy_aggregator_tuple not in self.models['distance_to_enemy_aggregator'][
                        'data_tuples']:
                        self.models['distance_to_enemy_aggregator']['data'] = \
                            self.models['distance_to_enemy_aggregator']['data'].append(
                                new_data_row_distance_to_enemy, ignore_index=True)
                        self.models['distance_to_enemy_aggregator'][
                            'data_tuples'].append(distance_to_enemy_aggregator_tuple)
                else:
                    distance_to_enemy_multiplier = calculate_distance_to_enemy_multiplier(nearest_values)
                    new_data_row_distance_to_enemy.update({'Multiplier': distance_to_enemy_multiplier})
                    distance_to_enemy_context_tuple = (
                        nearest_values[0], nearest_values[1], nearest_values[2], nearest_values[3], nearest_values[4],
                        distance_to_enemy_multiplier)
                    doe_multiplier = distance_to_enemy_multiplier
                    if distance_to_enemy_context_tuple not in self.models['distance_to_enemy_context'][
                        'data_tuples']:
                        self.models['distance_to_enemy_context']['data'] = \
                            self.models['distance_to_enemy_context']['data'].append(
                                new_data_row_distance_to_enemy,
                                ignore_index=True)
                        self.models['distance_to_enemy_context']['data_tuples'].append(
                            distance_to_enemy_context_tuple)

            seconds_since_last_sos_ = request_body['secondsSinceLastSOS']
            if seconds_since_last_sos_ != 1e6:
                predicted_sos_multiplier = calculate_sos_operational_context_mutliplier(seconds_since_last_sos_)
                sos_operational_context_tuple = (seconds_since_last_sos_, predicted_sos_multiplier)
                sos_multiplier = predicted_sos_multiplier
                if sos_operational_context_tuple not in self.models['sos_operational_context']['data_tuples']:
                    self.models['sos_operational_context']['data'] = \
                        self.models['sos_operational_context']['data'].append(
                            {'Seconds Since Last Sent SOS': seconds_since_last_sos_,
                             'Multiplier': predicted_sos_multiplier},
                            ignore_index=True)
                    self.models['sos_operational_context']['data_tuples'].append(sos_operational_context_tuple)

            new_data_row = {}
            new_data_tuple_list = []
            features = self.models[message_type]['features']
            cols_to_json = self.models[message_type]['cols_to_json']

            for feature in features:
                if feature in cols_to_json:
                    feature_value = request_body[cols_to_json[feature]]
                    new_data_row.update({feature: feature_value})
                    new_data_tuple_list.append(feature_value)

            raw_message_score = calculate_raw_score(message_type, request_body)
            new_data_row.update({self.models[message_type]['label']: raw_message_score})
            new_data_tuple_list.append(raw_message_score)
            new_data_tuple = tuple(new_data_tuple_list)

            if new_data_tuple not in self.models[message_type]['data_tuples']:
                self.models[message_type]['data'] = self.models[message_type]['data'].append(
                    new_data_row, ignore_index=True)
                self.models[message_type]['data_tuples'].append(new_data_tuple)

            if message_type == 'sos':
                return raw_message_score
            else:
                return raw_message_score * sos_multiplier * doe_multiplier

        elif request_type == self.DATA:
            if not self.writing_data:
                self.writing_data = True
                self.export_data()
            return 1

    def write_results(self):
        for model_type in self.models:
            actual_values = self.models[model_type]['actual_values']
            predicted_values = self.models[model_type]['predicted_values']

            current_combination = int(self.models[model_type]['current_combination'])

            if len(actual_values) > 0 and len(predicted_values) > 0:
                mae = mean_absolute_error(actual_values, predicted_values)

                num_runs = int(self.models[model_type]['results'].iat[current_combination, self.num_runs_index])
                old_mae = float(self.models[model_type]['results'].iat[current_combination, self.mae_index])

                num_runs += 1

                self.models[model_type]['results'].iat[current_combination, self.num_runs_index] = num_runs
                self.models[model_type]['results'].iat[current_combination, self.mae_index] = round(
                    (old_mae + mae) / num_runs, 3)

            self.models[model_type]['results'].to_csv(join(self.results_dir, '{0}.csv'.format(model_type)))


# Before running the server, pull the RabbitMQ docker image:
#  docker pull rabbitmq:3.8.5
#  once you've got the image, run it: docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.8.5

if __name__ == '__main__':
    server = SLRabbitMQServer()
    server.start()
