import json
import numpy as np
import pandas as pd
from os.path import join, realpath, dirname, isfile, isdir

from supervised_learning.common import MethodType, ScalingType
from scikit_model import ScikitModel
from rabbit_mq_server import RabbitMQServer
from mlm_utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

result_cols = ['regressor',
               'polynomial_degree', 'polynomial_interaction_only', 'polynomial_include_bias',
               'scaling_type', 'enable_normalization', 'use_grid_search', 'mae', 'mse']


class SLRabbitMQServer(RabbitMQServer):
    message_types = ['text_messages', 'tactical_graphics', 'sos', 'blue_spots', 'red_spots']

    message_type_models = {
        'text_messages': {
            'model': None,
            'combinations': [],
            'current_combination_index': -1,
            'features': ['Age of Message'],
            'label': 'Penalty',
            'cols_to_json': {
                'Age of Message': 'age_of_message',
                'Penalty': 'penalty',
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'tactical_graphics': {
            'model': None,
            'combinations': [],
            'current_combination_index': -1,
            'features': ['Age of Message'],
            'label': 'Score (Lazy)',
            'cols_to_json': {
                'Age of Message': 'age_of_message',
                'Score (Lazy)': 'score',
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'sos': {
            'model': None,
            'combinations': [],
            'current_combination_index': -1,
            'features': ['Age of Message', 'Number of blue Nodes'],
            'label': 'Score',
            'cols_to_json': {
                'Age of Message': 'age_of_message',
                'Number of blue Nodes': 'num_blue_nodes',
                'Score': 'score'
            },
            'results': pd.DataFrame(columns=result_cols)
        },
        'blue_spots': {
            'model': None,
            'combinations': [],
            'current_combination_index': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance'],
            'label': 'Score',
            'cols_to_json': {
                'Distance since Last Update': 'distance_since_last_update',
                'Number of blue Nodes': 'num_blue_nodes',
                'Average Distance': 'average_distance',
                'Average Hierarchical distance': 'average_hierarchical_distance',
                'Score': 'score'
            },
            'pre_trained_results': None,
            'results': pd.DataFrame(columns=result_cols)
        },
        'red_spots': {
            'model': None,
            'combinations': [],
            'current_combination_index': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Score',
            'cols_to_json': {
                'Distance since Last Update': 'distance_since_last_update',
                'Number of blue Nodes': 'num_blue_nodes',
                'Average Distance': 'average_distance',
                'Average Hierarchical distance': 'average_hierarchical_distance',
                'Score': 'score'
            },
            'results': pd.DataFrame(columns=result_cols)
        }
    }

    MODEL_CREATION = 'SET_MODEL'
    COST = 'UPDATE_MODEL'

    def __init__(self):
        for message_type in self.message_type_models:
            self.message_type_models[message_type].update(
                {'combinations': get_scikit_model_combinations_with_polynomials(MethodType.Regression,
                                                                                len(self.message_type_models[
                                                                                        message_type]['features']))})
        super().__init__(queue_name='sl_request_queue')

    def _get_reply(self, request):
        request_body = request['requestBody']
        request_type = request['requestType']

        if request_type == self.COST:
            message_type = request_body['message_type'].lower()
            feature_value_dict = {}
            if message_type == 'red_spots':
                nearest_values = request_body['nearest_values']
                nearest_values = np.array(nearest_values)
                nearest_values = np.sort(nearest_values)
                nearest_values = nearest_values[:5]
                request_body.update({'nearest_values': nearest_values})
                for i in range(5):
                    feature_value_dict.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
            label, actual = calculate_score(message_type, request_body)

            features = self.message_type_models[message_type]['features']
            cols_to_json = self.message_type_models[message_type]['cols_to_json']

            if self.message_type_models[message_type]['model'] is None:
                return -1
            else:
                try:
                    for feature in features:
                        if feature in cols_to_json:
                            feature_value_dict.update({feature: request_body[cols_to_json[feature]]})
                    return self.message_type_models[message_type]['model'].predict_then_fit(feature_value_dict)
                except:
                    return actual
        elif request_type == self.MODEL_CREATION:
            model_created = np.empty(5, dtype=np.bool)
            for i, message_type in enumerate(self.message_types):
                model_created[i] = self.create_model(message_type,
                                                     request_body['use_best'] == 1,
                                                     request_body['metric']) == 1
            return 1 if model_created.all() else 0

    def create_model(self, message_type, use_best, metric_type):
        results = self.message_type_models[message_type]['results']
        mse = self.message_type_models[message_type]['model'].calculate_score(message_type, 'mse')
        mae = self.message_type_models[message_type]['model'].calculate_score(message_type, 'mae')
        combinations = self.message_type_models[message_type]['combinations']
        if self.message_type_models[message_type]['current_combination_index'] in range(0, len(combinations)):
            combination = combinations[self.message_type_models[message_type]['current_combination_index']]
            method_name, degree, interaction_only, include_bias, scaling_type, enable_normalization, use_grid_search = combination
            if mse > -1 and mae > -1:
                results = results.append({'regressor': method_name,
                                          'polynomial_degree': degree,
                                          'polynomial_interaction_only': 'Yes' if interaction_only else 'No',
                                          'polynomial_include_bias': 'Yes' if include_bias else 'No',
                                          'scaling_type': scaling_type.name,
                                          'enable_normalization': 'Yes' if enable_normalization else 'No',
                                          'use_grid_search': 'Yes' if use_grid_search else 'No',
                                          'mae': mae,
                                          'mse': mse}, ignore_index=True)
                self.message_type_models[message_type].update({'results': results})
        try:
            if use_best:
                model = self.get_best_model(message_type, metric_type)
                self.message_type_models[message_type].update({'model': model, 'actual': [], 'predicted': []})
                return 1
            else:
                self.message_type_models[message_type]['current_combination_index'] += 1
                if self.message_type_models[message_type]['current_combination_index'] in range(0, len(combinations)):
                    combination = combinations[self.message_type_models[message_type]['current_combination_index']]
                    model = ScikitModel(combination, message_type, self.message_type_models[message_type]['features'],
                                        self.message_type_models[message_type]['label'])
                    self.message_type_models[message_type].update({'model': model})
                    return 1
                else:
                    return 0
        except:
            return 0

    def get_best_model(self, message_type, metric_type):
        results = self.message_type_models[message_type]['results']
        lowest_val = np.min(results[metric_type])
        for index, row in results.iterrows():
            if row[metric] == lowest_val:
                combination = (row['regressor'], row['polynomial_degree'], row['polynomial_interaction_only'], row['polynomial_include_bias'],
                               row['scaling_type'], row['enable_normalization'], row['use_grid_search'])
                return ScikitModel(combination, message_type, self.message_type_models[message_type]['features'], self.message_type_models[message_type]['label'])


# Before running the server, pull the RabbitMQ docker image:
#  docker pull rabbitmq:3.8.5
#  once you've got the image, run it: docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.8.5

if __name__ == '__main__':
    server = SLRabbitMQServer()
    server.start()
