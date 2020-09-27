import json
import numpy as np
import pandas as pd
from os.path import join, realpath, dirname, isfile, isdir

from supervised_learning.common import MethodType, ScalingType

from rabbit_mq_server import RabbitMQServer
from mlm_utils import calculate_score, get_scikit_model_combinations, generate_scikit_model, load_training_data
from sklearn.metrics import mean_squared_error

result_cols = ['Regressor', 'Scaling Type', 'Enable Normalization', 'Use Default Params', 'Cross Validation',
               'Mean Squared Error']


class SLRabbitMQServer(RabbitMQServer):
    message_type_models = {
        'text_messages': {
            'model': None,
            'current_combination_index': -1,
            'features': ['Age of Message'],
            'label': 'Penalty',
            'cols': ['Age of Message', 'Penalty'],
            'cols_to_json': {
                'Age of Message': 'age_of_message',
                'Penalty': 'penalty',
            },
            'actual': [],
            'predicted': [],
            'results': pd.DataFrame(columns=result_cols)
        },
        'tactical_graphics': {
            'model': None,
            'current_combination_index': -1,
            'features': ['Age of Message'],
            'label': 'Score (Lazy)',
            'cols': ['Age of Message', 'Score (Lazy)'],
            'cols_to_json': {
                'Age of Message': 'age_of_message',
                'Score (Lazy)': 'score',
            },
            'actual': [],
            'predicted': [],
            'results': pd.DataFrame(columns=result_cols)
        },
        'sos': {
            'model': None,
            'current_combination_index': -1,
            'features': ['Age of Message', 'Number of blue Nodes'],
            'label': 'Score',
            'cols': ['Age of Message', 'Number of blue Nodes', 'Score'],
            'cols_to_json': {
                'Age of Message': 'age_of_message',
                'Number of blue Nodes': 'num_blue_nodes',
                'Score': 'score'
            },
            'actual': [],
            'predicted': [],
            'results': pd.DataFrame(columns=result_cols)
        },
        'blue_spots': {
            'model': None,
            'current_combination_index': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance'],
            'label': 'Score',
            'cols': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                     'Average Hierarchical distance',
                     'Score'],
            'cols_to_json': {
                'Distance since Last Update': 'distance_since_last_update',
                'Number of blue Nodes': 'num_blue_nodes',
                'Average Distance': 'average_distance',
                'Average Hierarchical distance': 'average_hierarchical_distance',
                'Score': 'score'
            },
            'actual': [],
            'predicted': [],
            'results': pd.DataFrame(columns=result_cols)
        },
        'red_spots': {
            'model': None,
            'current_combination_index': -1,
            'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                         'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                         '#5 Nearest'],
            'label': 'Score',
            'cols': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                     'Average Hierarchical distance',
                     '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Score'],
            'cols_to_json': {
                'Distance since Last Update': 'distance_since_last_update',
                'Number of blue Nodes': 'num_blue_nodes',
                'Average Distance': 'average_distance',
                'Average Hierarchical distance': 'average_hierarchical_distance',
                'Score': 'score'
            },
            'actual': [],
            'predicted': [],
            'results': pd.DataFrame(columns=result_cols)
        }
    }

    combinations = []

    MODEL_CREATION = 0
    COST = 1
    MEAN_SQUARED_ERROR = 2

    def __init__(self):
        self.combinations = get_scikit_model_combinations(MethodType.Regression)
        super().__init__(queue_name='sl_request_queue')

    def _get_reply(self, request):
        message_type = request['message_type']
        request_type = request['request_type']

        if request_type == self.COST:
            new_data = {}
            if message_type == 'red_spots':
                nearest_values = request['nearest_values']
                nearest_values = np.array(nearest_values)
                nearest_values = np.sort(nearest_values)
                nearest_values = nearest_values[:5]
                request.update({'nearest_values': nearest_values})
                for i in range(5):
                    new_data.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
            label, actual = calculate_score(message_type, request)

            features = self.message_type_models[message_type]['features']
            cols_to_json = self.message_type_models[message_type]['cols_to_json']

            if self.message_type_models[message_type]['model'] is None:
                return json.dumps({'message_type': message_type, label: actual, 'using_model': 0})
            else:
                try:
                    test_input = []
                    for feature in features:
                        if feature in cols_to_json:
                            value = request[cols_to_json[feature]]
                            new_data.update({feature: value})
                            test_input.append(value)
                    test_input = np.array([test_input])
                    predicted = self.message_type_models[message_type]['model'].predict(test_input)
                    self.message_type_models[message_type]['predicted'].append(predicted)
                    self.message_type_models[message_type]['actual'].append(actual)
                    self.message_type_models[message_type]['model'].add(new_data, actual)
                    return json.dumps({'message_type': message_type, label: predicted, 'using_model': 1})
                except:
                    return json.dumps({'message_type': message_type, label: actual, 'using_model': 0})
        elif request_type == self.MEAN_SQUARED_ERROR:
            score = self.calculate_mean_squared_error(message_type)
            return json.dumps({'message_type': message_type, 'mean_squared_error': score})
        elif request_type == self.MODEL_CREATION:
            score = self.calculate_mean_squared_error(message_type)
            if self.message_type_models[message_type]['current_combination_index'] in range(0, len(
                    self.combinations)) and score > -1:
                combination = self.combinations[self.message_type_models[message_type]['current_combination_index']]
                method_name, scaling_type, enable_normalization, use_grid_search, cv = combination
                results = self.message_type_models[message_type]['results']
                results = results.append({'Regressor': method_name,
                                          'Scaling Type': scaling_type.name,
                                          'Enable Normalization': 'Yes' if enable_normalization else 'No',
                                          'Use Default Params': 'No' if use_grid_search else 'Yes',
                                          'Cross Validation': cv,
                                          'Mean Squared Error': score}, ignore_index=True)
                self.message_type_models[message_type].update({'results': results})
            try:
                if 'use_best' in request:
                    use_best = request['use_best'] == 1
                else:
                    use_best = False

                if use_best:
                    model = self.get_best_model(message_type)
                    self.message_type_models[message_type].update({'model': model, 'actual': [], 'predicted': []})
                    return json.dumps({'message_type': message_type, 'model_created': 1})
                else:
                    self.message_type_models[message_type]['current_combination_index'] += 1
                    if self.message_type_models[message_type]['current_combination_index'] in range(0, len(
                            self.combinations)):
                        combination = self.combinations[
                            self.message_type_models[message_type]['current_combination_index']]
                        method_name, scaling_type, enable_normalization, use_grid_search, cv = combination
                        training_data = load_training_data(self.message_type_models[message_type]['cols'], message_type)
                        model = generate_scikit_model(MethodType.Regression, training_data, method_name, scaling_type,
                                                      enable_normalization, use_grid_search, cv)
                        self.message_type_models[message_type].update({'model': model, 'actual': [], 'predicted': []})
                        return json.dumps({'message_type': message_type, 'model_created': 1})
                    else:
                        return json.dumps({'message_type': message_type, 'model_created': 0})
            except:
                return json.dumps({'message_type': message_type, 'model_created': 0})

    def calculate_mean_squared_error(self, message_type):
        actual_outputs = self.message_type_models[message_type]['actual']
        predicted_outputs = self.message_type_models[message_type]['predicted']
        if len(actual_outputs) > 0 and len(predicted_outputs) > 0:
            score = mean_squared_error(actual_outputs, predicted_outputs)
        else:
            score = -1
        return score

    def get_best_model(self, message_type):
        results = self.message_type_models[message_type]['results']
        if len(results.index) == 0:
            if isdir(join(dirname(realpath('__file__')), 'results')) and isfile(
                    join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(message_type))):
                results = pd.read_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(message_type)),
                                      usecols=result_cols)
                self.message_type_models[message_type].update({'results': results})
        if len(results.index) == 0:
            return None
        else:
            lowest_mse = np.min(results['Mean Squared Error'])
            training_data = load_training_data(self.message_type_models[message_type]['cols'], message_type)
            for index, row in results.iterrows():
                if row['Mean Squared Error'] == lowest_mse:
                    return generate_scikit_model(MethodType.Regression, training_data, row['Regressor'],
                                                 ScalingType.get_type_by_name(row['Scaling Type']),
                                                 row['Enable Normalization'] == 'Yes',
                                                 row['Use Default Params'] == 'No', row['Cross Validation'])
            return None


# Before running the server, pull the RabbitMQ docker image:
#  docker pull rabbitmq:3.8.5
#  once you've got the image, run it: docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.8.5

if __name__ == '__main__':
    server = SLRabbitMQServer()
    server.start()
