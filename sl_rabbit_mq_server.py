import json
import numpy as np
import pandas as pd
import atexit
from os.path import join, realpath, dirname, isfile, isdir
from os import mkdir
from datetime import datetime

from scikit_model import ScikitModel
from rabbit_mq_server import RabbitMQServer
from mlm_utils import *


class SLRabbitMQServer(RabbitMQServer):
    message_types = ['text_messages', 'tactical_graphics', 'sos', 'blue_spots', 'red_spots']
    context_types = ['sos_operational_context', 'distance_to_enemy_context', 'distance_to_enemy_aggregator']
    result_cols = ['combination_id', 'regressor', 'pre_processing_type', 'num_runs',
                   'mae', 'r2']
    results_dir = join(dirname(realpath('__file__')), 'results')
    datasets_dir = join(dirname(realpath('__file__')), 'datasets')

    context_type_models = {
        'sos_operational_context': {
            'model': None,
            'current_combination': -1,
            'features': ['Seconds Since Last Sent SOS'],
            'label': 'Multiplier',
            'cols': ['Seconds Since Last Sent SOS', 'Multiplier'],
            'cols_to_json': {
                'Seconds Since Last Sent SOS': 'secondsSinceLastSentSOS',
                'Multiplier': 'multiplier'
            }
        },
        'distance_to_enemy_context': {
            'model': None,
            'current_combination': -1,
            'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Multiplier',
            'cols': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest', 'Multiplier'],
            'cols_to_json': {}
        },
        'distance_to_enemy_aggregator': {
            'model': None,
            'current_combination': -1,
            'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Multiplier',
            'cols': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                     '#5 Nearest', 'Multiplier'],
            'cols_to_json': {}
        }
    }

    message_type_models = {
        'text_messages': {
            'model': None,
            'current_combination': -1,
            'features': ['Age of Message'],
            'label': 'Penalty',
            'cols': ['Age of Message', 'Penalty'],
            'cols_to_json': {
                'Age of Message': 'ageOfMessage',
                'Penalty': 'penalty',
            }
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
            }
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
            }
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
            }
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
            }
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

        for message_type in self.message_types:
            self.message_type_models[message_type].update({'results': pd.read_csv(
                join(self.results_dir, '{0}.csv'.format(message_type)), usecols=self.result_cols,
                index_col=self.result_cols[0])})

        for context_type in self.context_types:
            self.context_type_models[context_type].update({'results': pd.read_csv(
                join(self.results_dir, '{0}.csv'.format(context_type)), usecols=self.result_cols,
                index_col=self.result_cols[0])})

        self.writing_results = False
        self.writing_data = False

    def setup_data(self):
        if not isdir(self.datasets_dir):
            mkdir(self.datasets_dir)

        for message_type in self.message_types:
            self.message_type_models[message_type].update({'data': pd.DataFrame(columns=self.message_type_models[message_type]['cols'])})
            self.message_type_models[message_type].update({'data_tuples': []})
            datasets_dir_message_type = join(self.datasets_dir, message_type)
            if not isdir(datasets_dir_message_type):
                mkdir(datasets_dir_message_type)

        for context_type in self.context_types:
            self.context_type_models[context_type].update({'data': pd.DataFrame(columns=self.context_type_models[context_type]['cols'])})
            self.context_type_models[context_type].update({'data_tuples': []})
            datasets_dir_context_type = join(self.datasets_dir, context_type)
            if not isdir(datasets_dir_context_type):
                mkdir(datasets_dir_context_type)

    def export_results(self):
        for message_type in self.message_types:
            self.message_type_models[message_type]['results'].to_csv(
                join(self.results_dir, '{0}.csv'.format(message_type)))

        for context_type in self.context_types:
            self.context_type_models[context_type]['results'].to_csv(
                join(self.results_dir, '{0}.csv'.format(context_type)))

    def export_data(self):
        for message_type in self.message_types:
            self.message_type_models[message_type]['data'].to_csv(
                join(self.datasets_dir, message_type, '{0}_{1}.csv'.format(message_type, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

        for context_type in self.context_types:
            self.context_type_models[context_type]['data'].to_csv(
                join(self.datasets_dir, context_type, '{0}_{1}.csv'.format(context_type, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

    def _get_reply(self, request):
        request_body = request['requestBody']
        request_type = request['requestType']

        print(request_body)

        if request_type == self.COST:
            self.writing_results = False
            self.writing_data = False
            message_type = request_body['messageType'].lower()

            # Attempt to predict raw score
            predicted_raw_score = True
            predicted = -1
            if self.message_type_models[message_type]['model'] is None:
                predicted_raw_score = False
            else:
                try:
                    new_data_row = {}
                    features = self.message_type_models[message_type]['features']
                    cols_to_json = self.message_type_models[message_type]['cols_to_json']
                    for feature in features:
                        if feature in cols_to_json:
                            new_data_row.update({feature: request_body[cols_to_json[feature]]})
                    predicted = self.message_type_models[message_type]['model'].predict(new_data_row)
                except:
                    predicted_raw_score = False
            # Attempt to predict multiplier otherwise calculate actual
            # If we couldn't predict raw score, then we should calculate overall score
            if not predicted_raw_score:
                return calculate_score(message_type, request_body)
            else:
                nearest_values = request_body['nearestValues']
                nearest_values = np.array(nearest_values)
                nearest_values = np.sort(nearest_values)
                nearest_values = nearest_values[:5]
                if self.context_type_models['distance_to_enemy_context']['model'] is None:
                    distance_to_enemy_context_multiplier = calculate_distance_to_enemy_multiplier(nearest_values)
                else:
                    try:
                        new_data_row = {}
                        for i in range(5):
                            new_data_row.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                        distance_to_enemy_context_multiplier = self.context_type_models['distance_to_enemy_context'][
                            'model'].predict(new_data_row)
                    except:
                        distance_to_enemy_context_multiplier = calculate_distance_to_enemy_multiplier(nearest_values)

                if self.context_type_models['distance_to_enemy_aggregator']['model'] is None:
                    distance_to_enemy_aggregate_multiplier = calculate_distance_to_enemy_aggregator(nearest_values)
                else:
                    try:
                        new_data_row = {}
                        for i in range(5):
                            new_data_row.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                        distance_to_enemy_aggregate_multiplier = \
                            self.context_type_models['distance_to_enemy_aggregator']['model'].predict(
                                new_data_row)
                    except:
                        distance_to_enemy_aggregate_multiplier = calculate_distance_to_enemy_aggregator(nearest_values)

                if self.context_type_models['sos_operational_context']['model'] is None:
                    sos_multiplier = calculate_sos_operational_context_mutliplier(request_body['secondsSinceLastSOS'])
                else:
                    try:
                        new_data_row = {
                            'Seconds Since Last Sent SOS': request_body['secondsSinceLastSOS']}
                        sos_multiplier = self.context_type_models['sos_operational_context']['model'].predict(
                            new_data_row)
                    except:
                        sos_multiplier = calculate_sos_operational_context_mutliplier(
                            request_body['secondsSinceLastSOS'])

                if message_type in ['blue_spots', 'tactical_graphics', 'text_messages']:
                    final_score = predicted * sos_multiplier * distance_to_enemy_context_multiplier
                    # print('Message Type: {0}, Raw Score: {1}, SOS Multiplier: {2}, Distance to Enemy: {3}, Final Score: {4}'.format(message_type, predicted,sos_multiplier, distance_to_enemy_context_multiplier, final_score))
                    return final_score
                elif message_type == 'red_spots':
                    final_score = predicted * sos_multiplier * distance_to_enemy_aggregate_multiplier
                    # print('Message Type: red_spots, Raw Score: {0}, SOS Multiplier: {1}, Distance to Enemy Aggregator: {2}, Final Score: {3}'.format(predicted, sos_multiplier, distance_to_enemy_aggregate_multiplier, final_score))
                    return final_score
                else:
                    # print('Message Type: sos, Score: {0}'.format(predicted))
                    return predicted
        elif request_type == self.SETUP:
            self.writing_results = False
            self.writing_data = False
            self.setup_data()

        elif request_type == self.MODEL_CREATION:
            self.writing_results = False
            self.writing_data = False
            all_message_models_created = np.empty(len(self.message_types), dtype=np.bool)
            for i, message_type in enumerate(self.message_types):
                all_message_models_created[i] = self.create_model_for_message_type(message_type,
                                                                                   request_body) == 1
            all_context_models_created = np.empty(len(self.context_types), dtype=np.bool)
            for i, context_type in enumerate(self.context_types):
                all_context_models_created[i] = self.create_model_for_context_type(context_type,
                                                                                   request_body) == 1

            return 1 if all_context_models_created.all() and all_message_models_created.all() else 0

        elif request_type == self.RESULTS:
            if not self.writing_results:
                self.writing_results = True
                self.write_results(request_body)
                self.export_results()
            return 1

        elif request_type == self.TRAIN:
            message_type = request_body['messageType'].lower()
            nearest_values = request_body['nearestValues']
            nearest_values = np.array(nearest_values)
            nearest_values = np.sort(nearest_values)
            nearest_values = nearest_values[:5]

            new_data_row_distance_to_enemy = {}
            new_data_row_distance_to_enemy_context = new_data_row_distance_to_enemy
            new_data_row_distance_to_enemy_aggregator = new_data_row_distance_to_enemy


            for i in range(5):
                new_data_row_distance_to_enemy.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
            distance_to_enemy_multiplier = calculate_distance_to_enemy_multiplier(nearest_values)
            new_data_row_distance_to_enemy_context.update(
                {'Multiplier': distance_to_enemy_multiplier})
            distance_to_enemy_aggregator = calculate_distance_to_enemy_aggregator(nearest_values)
            new_data_row_distance_to_enemy_aggregator.update(
                {'Multiplier': distance_to_enemy_aggregator})

            distance_to_enemy_context_tuple = (
            nearest_values[0], nearest_values[1], nearest_values[2], nearest_values[3], nearest_values[4],
            distance_to_enemy_multiplier)
            if distance_to_enemy_context_tuple not in self.context_type_models['distance_to_enemy_context']['data_tuples']:
                self.context_type_models['distance_to_enemy_context']['data'] = \
                    self.context_type_models['distance_to_enemy_context']['data'].append(
                        new_data_row_distance_to_enemy_context,
                        ignore_index=True)
                self.context_type_models['distance_to_enemy_context']['data_tuples'].append(distance_to_enemy_context_tuple)

            distance_to_enemy_aggregator_tuple = (
            nearest_values[0], nearest_values[1], nearest_values[2], nearest_values[3], nearest_values[4],
            distance_to_enemy_aggregator)
            if distance_to_enemy_aggregator_tuple not in self.context_type_models['distance_to_enemy_aggregator'][
                'data_tuples']:
                self.context_type_models['distance_to_enemy_aggregator']['data'] = \
                    self.context_type_models['distance_to_enemy_aggregator']['data'].append(
                        new_data_row_distance_to_enemy_aggregator, ignore_index=True)
                self.context_type_models['distance_to_enemy_aggregator'][
                    'data_tuples'].append(distance_to_enemy_aggregator_tuple)

            sos_mutliplier = calculate_sos_operational_context_mutliplier(request_body['secondsSinceLastSOS'])
            sos_operational_context_tuple = (request_body['secondsSinceLastSOS'], sos_mutliplier)
            if sos_operational_context_tuple not in self.context_type_models['sos_operational_context']['data_tuples']:
                self.context_type_models['sos_operational_context']['data'] = self.context_type_models['sos_operational_context']['data'].append({'Seconds Since Last Sent SOS': request_body['secondsSinceLastSOS'], 'Multiplier': sos_mutliplier}, ignore_index=True)
                self.context_type_models['sos_operational_context']['data_tuples'].append(sos_operational_context_tuple)

            new_data_row = {}
            new_data_tuple_list = []
            features = self.message_type_models[message_type]['features']
            cols_to_json = self.message_type_models[message_type]['cols_to_json']

            for feature in features:
                if feature in cols_to_json:
                    feature_value = request_body[cols_to_json[feature]]
                    new_data_row.update({feature: feature_value})
                    new_data_tuple_list.append(feature_value)

            raw_message_score = calculate_raw_score(message_type, request_body)
            new_data_row.update({self.message_type_models[message_type]['label']: raw_message_score})
            new_data_tuple_list.append(raw_message_score)
            new_data_tuple = tuple(new_data_tuple_list)

            if new_data_tuple not in self.message_type_models[message_type]['data_tuples']:
                self.message_type_models[message_type]['data'] = self.message_type_models[message_type]['data'].append(new_data_row, ignore_index=True)
                self.message_type_models[message_type]['data_tuples'].append(new_data_tuple)

            return calculate_score(message_type, request_body)
        elif request_type == self.DATA:
            if not self.writing_data:
                self.writing_data = True
                self.export_data()
            return 1

    def write_results(self, request_body):
        current_combination_id = int(request_body['combinationID'])

        if current_combination_id < 0 or current_combination_id >= len(self.combinations):
            return

        for message_type in self.message_types:
            if self.message_type_models[message_type]['model'] is not None:
                r2 = self.message_type_models[message_type]['model'].calculate_score_with_metric('r2')
                mae = self.message_type_models[message_type]['model'].calculate_score_with_metric('mae')
            else:
                r2 = -1
                mae = -1

            if mae > -1:
                num_runs = self.message_type_models[message_type]['results'].iat[
                    current_combination_id, 2]
                old_mae = self.message_type_models[message_type]['results'].iat[
                    current_combination_id, 3]
                old_r2 = self.message_type_models[message_type]['results'].iat[
                    current_combination_id, 4]

                num_runs = int(num_runs)
                old_mae = float(old_mae)
                old_r2 = float(old_r2)

                num_runs += 1

                self.message_type_models[message_type]['results'].iat[
                    current_combination_id, 2] = num_runs
                self.message_type_models[message_type]['results'].iat[
                    current_combination_id, 3] = round((old_mae + mae) / num_runs, 3)
                self.message_type_models[message_type]['results'].iat[
                    current_combination_id, 4] = round((old_r2 + r2) / num_runs, 3)

        for context_type in self.context_types:
            if self.context_type_models[context_type]['model'] is not None:
                r2 = self.context_type_models[context_type]['model'].calculate_score_with_metric('r2')
                mae = self.context_type_models[context_type]['model'].calculate_score_with_metric('mae')
            else:
                r2 = -1
                mae = -1

            if mae > -1:
                num_runs = self.context_type_models[context_type]['results'].iat[
                    current_combination_id, 2]
                old_mae = self.context_type_models[context_type]['results'].iat[
                    current_combination_id, 3]
                old_r2 = self.context_type_models[context_type]['results'].iat[
                    current_combination_id, 4]

                num_runs = int(num_runs)
                old_mae = float(old_mae)
                old_r2 = float(old_r2)

                num_runs += 1

                self.context_type_models[context_type]['results'].iat[
                    current_combination_id, 2] = num_runs
                self.context_type_models[context_type]['results'].iat[
                    current_combination_id, 3] = round((old_mae + mae) / num_runs, 3)
                self.context_type_models[context_type]['results'].iat[
                    current_combination_id, 4] = round((old_r2 + r2) / num_runs, 3)

    def create_model_for_message_type(self, message_type, request_body):
        try:
            combination_id = int(request_body['combinationID'])
            if 0 <= combination_id < len(self.combinations):
                if self.message_type_models[message_type]['model'] is None or \
                        self.message_type_models[message_type]['model'].current_id != combination_id:
                    model = ScikitModel(combination_id, message_type,
                                        self.message_type_models[message_type]['features'],
                                        self.message_type_models[message_type]['label'])
                    self.message_type_models[message_type].update({'model': model})
                    self.message_type_models[message_type]['model'].current_id = combination_id
                return 1
            else:
                return 0
        except:
            return 0

    def create_model_for_context_type(self, context_type, request_body):
        try:
            combination_id = int(request_body['combinationID'])
            if 0 <= combination_id < len(self.combinations):
                if self.context_type_models[context_type]['model'] is None or \
                        self.context_type_models[context_type]['model'].current_id != combination_id:
                    model = ScikitModel(combination_id, context_type,
                                        self.context_type_models[context_type]['features'],
                                        self.context_type_models[context_type]['label'])
                    self.context_type_models[context_type].update({'model': model})
                    self.context_type_models[context_type]['model'].current_id = combination_id
                return 1
            else:
                return 0
        except:
            return 0



# Before running the server, pull the RabbitMQ docker image:
#  docker pull rabbitmq:3.8.5
#  once you've got the image, run it: docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.8.5

if __name__ == '__main__':
    server = SLRabbitMQServer()
    server.start()
