import json
import numpy as np
import pandas as pd
import atexit
from os.path import join, realpath, dirname, isfile, isdir
from os import mkdir

from scikit_model import ScikitModel
from rabbit_mq_server import RabbitMQServer
from mlm_utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

result_cols = ['regressor', 'polynomial_degree', 'scaling_type', 'enable_normalization', 'use_grid_search', 'num_runs',
               'mae', 'mse']


class SLRabbitMQServer(RabbitMQServer):
    message_types = ['text_messages', 'tactical_graphics', 'sos', 'blue_spots', 'red_spots']
    context_types = ['sos_operational_context', 'distance_to_enemy_context', 'distance_to_enemy_aggregator']

    context_models = {
        'sos_operational_context': {
            'model': None,
            'current_combination': -1,
            'features': ['Seconds Since Last Sent SOS'],
            'label': 'Multiplier',
            'cols_to_json': {
                'Seconds Since Last Sent SOS': 'secondsSinceLastSentSOS',
                'Multiplier': 'multiplier'
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'distance_to_enemy_context': {
            'model': None,
            'current_combination': -1,
            'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Multiplier',
            'results': pd.DataFrame(columns=result_cols)
        },
        'distance_to_enemy_aggregator': {
            'model': None,
            'current_combination': -1,
            'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Multiplier',
            'results': pd.DataFrame(columns=result_cols)
        }
    }

    message_type_models = {
        'text_messages': {
            'model': None,
            'current_combination': -1,
            'features': ['Age of Message'],
            'label': 'Penalty',
            'cols_to_json': {
                'Age of Message': 'ageOfMessage',
                'Penalty': 'penalty',
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'tactical_graphics': {
            'model': None,
            'current_combination': -1,
            'features': ['Age of Message'],
            'label': 'Score (Lazy)',
            'cols_to_json': {
                'Age of Message': 'ageOfMessage',
                'Score (Lazy)': 'score',
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'sos': {
            'model': None,
            'current_combination': -1,
            'features': ['Age of Message', 'Number of blue Nodes'],
            'label': 'Score',
            'cols_to_json': {
                'Age of Message': 'ageOfMessage',
                'Number of blue Nodes': 'numBlueNodes',
                'Score': 'score'
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'blue_spots': {
            'model': None,
            'current_combination': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance', 'Is Affected'],
            'label': 'Score',
            'cols_to_json': {
                'Distance since Last Update': 'distanceSinceLastUpdate',
                'Number of blue Nodes': 'numBlueNodes',
                'Average Distance': 'averageDistance',
                'Average Hierarchical distance': 'averageHierarchicalDistance',
                'Is Affected': 'isAffected',
                'Score': 'score'
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'red_spots': {
            'model': None,
            'current_combination': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance', 'Is Affected'],
            'label': 'Score',
            'cols_to_json': {
                'Distance since Last Update': 'distanceSinceLastUpdate',
                'Number of blue Nodes': 'numBlueNodes',
                'Average Distance': 'averageDistance',
                'Average Hierarchical distance': 'averageHierarchicalDistance',
                'Is Affected': 'isAffected',
                'Score': 'score'
            },
            'results': pd.DataFrame(columns=result_cols)
        }
    }

    MODEL_CREATION = 'SET_MODEL'
    COST = 'UPDATE_MODEL'

    def __init__(self):
        super().__init__(queue_name='sl_request_queue', durable=True)
        for message_type in self.message_types:
            combinations = get_scikit_model_combinations_with_polynomials(
                len(self.message_type_models[message_type]['features']))
            self.message_type_models[message_type].update({
                'combinations': combinations})
            for combination in combinations:
                method_name, degree, scaling_type, enable_normalization, use_grid_search = combination
                self.message_type_models[message_type]['results'] = self.message_type_models[message_type][
                    'results'].append({'regressor': method_name,
                                       'polynomial_degree': degree,
                                       'scaling_type': scaling_type.name,
                                       'enable_normalization': 'Yes' if enable_normalization else 'No',
                                       'use_grid_search': 'Yes' if use_grid_search else 'No',
                                       'num_runs': 0,
                                       'mae': 0,
                                       'mse': 0}, ignore_index=True)
        for context_type in self.context_types:
            combinations = get_scikit_model_combinations_with_polynomials(
                len(self.context_models[context_type]['features']))
            self.context_models[context_type].update({'combinations': combinations})
            for combination in combinations:
                method_name, degree, scaling_type, enable_normalization, use_grid_search = combination
                self.context_models[context_type]['results'] = self.context_models[context_type]['results'].append(
                    {'regressor': method_name,
                     'polynomial_degree': degree,
                     'scaling_type': scaling_type.name,
                     'enable_normalization': 'Yes' if enable_normalization else 'No',
                     'use_grid_search': 'Yes' if use_grid_search else 'No',
                     'num_runs': 0,
                     'mae': 0,
                     'mse': 0}, ignore_index=True)

    def write_results(self):
        results_dir = join(dirname(realpath('__file__')), 'results')
        if not isdir(results_dir):
            mkdir(results_dir)

        for message_type in self.message_types:
            self.message_type_models[message_type]['results'].to_csv(join(results_dir, '{0}.csv'.format(message_type)))

        for context_type in self.context_types:
            self.context_models[context_type]['results'].to_csv(join(results_dir, '{0}.csv'.format(context_type)))

    def _get_reply(self, request):
        request_body = request['requestBody']
        request_type = request['requestType']

        if request_type == self.COST:
            message_type = request_body['messageType'].lower()

            # Attempt to predict raw score
            predicted_raw_score = True
            predicted = -1
            if self.message_type_models[message_type]['model'] is None:
                predicted_raw_score = False
            else:
                try:
                    feature_value_dict_message = {}
                    features = self.message_type_models[message_type]['features']
                    cols_to_json = self.message_type_models[message_type]['cols_to_json']
                    for feature in features:
                        if feature in cols_to_json:
                            feature_value_dict_message.update({feature: request_body[cols_to_json[feature]]})
                    predicted = self.message_type_models[message_type]['model'].predict_then_fit(
                        feature_value_dict_message)
                except:
                    predicted_raw_score = False
            # Attempt to predict multiplier otherwise calculate actual
            # If we couldn't predict raw score, then we should calculate overall score
            if not predicted_raw_score:
                return calculate_score(message_type, request_body)
            else:
                distance_to_enemy_context_multiplier = -1
                nearest_values = request_body['nearestValues']
                nearest_values = np.array(nearest_values)
                nearest_values = np.sort(nearest_values)
                nearest_values = nearest_values[:5]
                if self.context_models['distance_to_enemy_context']['model'] is None:
                    distance_to_enemy_context_multiplier = calculate_distance_to_enemy_multiplier(nearest_values)
                else:
                    try:
                        feature_value_dict = {}
                        for i in range(5):
                            feature_value_dict.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                            distance_to_enemy_context_multiplier = self.context_models['distance_to_enemy_context'][
                                'model'].predict_then_fit(feature_value_dict)
                    except:
                        distance_to_enemy_context_multiplier = calculate_distance_to_enemy_multiplier(nearest_values)

                distance_to_enemy_aggregate_multiplier = -1
                if self.context_models['distance_to_enemy_aggregator']['model'] is None:
                    distance_to_enemy_aggregate_multiplier = calculate_distance_to_enemy_aggregator(nearest_values)
                else:
                    try:
                        feature_value_dict = {}
                        for i in range(5):
                            feature_value_dict.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                            distance_to_enemy_aggregate_multiplier = \
                            self.context_models['distance_to_enemy_aggregator'][
                                'model'].predict_then_fit(feature_value_dict)
                    except:
                        distance_to_enemy_aggregate_multiplier = calculate_distance_to_enemy_aggregator(nearest_values)

                if self.context_models['sos_operational_context']['model'] is None:
                    sos_multiplier = calculate_sos_operational_context_mutliplier(request_body['secondsSinceLastSOS'])
                else:
                    try:
                        feature_value_dict = {'Seconds Since Last SOS': request_body['secondsSinceLastSOS']}
                        sos_multiplier = self.context_models['sos_operational_context']['model'].predict_then_fit(
                            feature_value_dict)
                    except:
                        sos_multiplier = calculate_sos_operational_context_mutliplier(
                            request_body['secondsSinceLastSOS'])

                if message_type in ['blue_spots', 'tactical_graphics', 'text_messages']:
                    return predicted * sos_multiplier * distance_to_enemy_context_multiplier
                elif message_type == 'red_spots':
                    return predicted * sos_multiplier * distance_to_enemy_aggregate_multiplier
                else:
                    return predicted

        elif request_type == self.MODEL_CREATION:
            all_message_models_created = np.empty(self.message_types, dtype=np.bool)
            for i, message_type in enumerate(self.message_types):
                all_message_models_created[i] = self.create_model_for_message_type(message_type,
                                                                                   request_body) == 1
            all_context_models_created = np.empty(self.context_types, dtype=np.bool)
            for i, context_type in enumerate(self.context_types):
                all_context_models_created[i] = self.create_model_for_context_type(context_type,
                                                                                   request_body) == 1

            return 1 if all_context_models_created.all() and all_message_models_created.all() else 0

    def create_model_for_message_type(self, message_type, request_body):
        if self.message_type_models[message_type]['model'] is not None:
            mse = self.message_type_models[message_type]['model'].calculate_score_with_metric('mse')
            mae = self.message_type_models[message_type]['model'].calculate_score_with_metric('mae')
        else:
            mse = -1
            mae = -1

        combinations = self.message_type_models[message_type]['combinations']

        if 0 <= self.message_type_models[message_type]['current_combination'] < len(combinations):
            if mse > -1 and mae > -1:
                num_runs = self.message_type_models[message_type]['results'].iat[
                    self.message_type_models[message_type]['current_combination'], 5]
                old_mae = self.message_type_models[message_type]['results'].iat[
                    self.message_type_models[message_type]['current_combination'], 6]
                old_mse = self.message_type_models[message_type]['results'].iat[
                    self.message_type_models[message_type]['current_combination'], 7]

                num_runs += 1

                self.message_type_models[message_type]['results'].iat[
                    self.message_type_models[message_type]['current_combination'], 5] = num_runs
                self.message_type_models[message_type]['results'].iat[
                    self.message_type_models[message_type]['current_combination'], 6] = (old_mae + mae) / num_runs
                self.message_type_models[message_type]['results'].iat[
                    self.message_type_models[message_type]['current_combination'], 7] = (old_mse + mse) / num_runs
        try:
            if request_body['useBest'] == 1:
                model = self.get_best_model_for_message_type(message_type, request_body['metricType'])
                self.message_type_models[message_type].update({'model': model})
                return 1
            else:
                if 0 <= request_body['combinationID'] < len(combinations):
                    if self.message_type_models[message_type]['model'] is None or self.message_type_models[message_type]['model'].current_id != request_body['combinationID']:
                        model = ScikitModel(request_body['combinationID'], message_type,
                                            self.message_type_models[message_type]['features'],
                                            self.message_type_models[message_type]['label'])
                        self.message_type_models[message_type].update({'model': model})
                        self.message_type_models[message_type]['current_combination'] = request_body['combinationID']
                    return 1
                else:
                    return 0

        except:
            return 0

    def create_model_for_context_type(self, context_type, request_body):
        if self.context_models[context_type]['model'] is not None:
            mse = self.context_models[context_type]['model'].calculate_score_with_metric('mse')
            mae = self.context_models[context_type]['model'].calculate_score_with_metric('mae')
        else:
            mse = -1
            mae = -1

        combinations = self.context_models[context_type]['combinations']

        if 0 <= self.context_models[context_type]['current_combination'] < len(combinations):
            if mse > -1 and mae > -1:
                num_runs = self.context_models[context_type]['results'].iat[
                    self.context_models[context_type]['current_combination'], 5]
                old_mae = self.context_models[context_type]['results'].iat[
                    self.context_models[context_type]['current_combination'], 6]
                old_mse = self.context_models[context_type]['results'].iat[
                    self.context_models[context_type]['current_combination'], 7]

                num_runs += 1

                self.context_models[context_type]['results'].iat[
                    self.context_models[context_type]['current_combination'], 5] = num_runs
                self.context_models[context_type]['results'].iat[
                    self.context_models[context_type]['current_combination'], 6] = (old_mae + mae) / num_runs
                self.context_models[context_type]['results'].iat[
                    self.context_models[context_type]['current_combination'], 7] = (old_mse + mse) / num_runs
        try:
            if request_body['useBest'] == 1:
                model = self.get_best_model_for_context_type(message_type, request_body['metricType'])
                self.context_models[context_type].update({'model': model})
                return 1
            else:
                if 0 <= request_body['combinationID'] < len(combinations):
                    if self.context_models[context_type]['model'] is None or self.context_models[context_type]['model'].current_id != request_body['combinationID']:
                        model = ScikitModel(request_body['combinationID'], context_type,
                                            self.context_models[context_type]['features'],
                                            self.context_models[context_type]['label'])
                        self.context_models[context_type].update({'model': model})
                        self.context_models[context_type]['current_combination'] = request_body['combinationID']
                    return 1
                else:
                    return 0
        except:
            return 0

    def get_best_model(self, results, metric_type, combinations, sheet_name, features, label):
        lowest_val = np.min(results[metric_type])
        metric_index = 6 if metric_type == 'mae' else 7
        for i in range(len(combinations)):
            if results.iat[i, metric_index] == lowest_val:
                return ScikitModel(i, sheet_name, features, label)
        return None

    def get_best_model_for_message_type(self, message_type, metric_type):
        return self.get_best_model(self.message_type_models[message_type]['results'], metric_type,
                                   self.message_type_models[message_type]['combinations'], message_type,
                                   self.message_type_models[message_type]['features'],
                                   self.message_type_models[message_type]['label'])

    def get_best_model_for_context_type(self, context_type, metric_type):
        return self.get_best_model(self.context_models[context_type]['results'], metric_type,
                                   self.context_models[context_type]['combinations'], context_type,
                                   self.context_models[context_type]['features'],
                                   self.context_models[context_type]['label'])


# Before running the server, pull the RabbitMQ docker image:
#  docker pull rabbitmq:3.8.5
#  once you've got the image, run it: docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.8.5

if __name__ == '__main__':
    server = SLRabbitMQServer()
    server.start()
