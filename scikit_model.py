import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlm_utils import *
import pickle


class ScikitModel:
    def __init__(self, combination_id, sheet_name, features, label):
        model_name = join(dirname(realpath('__file__')), 'models', sheet_name, '{0}.pkl'.format(combination_id))
        self.current_id = combination_id
        self.pipeline = pickle.load(open(model_name, 'rb'))
        self.sheet_name = sheet_name
        self.features = features
        self.label = label
        self.actual_values = []
        self.predicted_values = []
        # print('Loaded {0}.pkl for {1}'.format(combination_id, sheet_name))

    def predict(self, feature_values_dict):
        test_input = []
        # print('Feature values for {0}: {1}'.format(self.sheet_name, feature_values_dict))
        actual_value = -1
        if self.sheet_name in ['text_messages', 'tactical_graphics', 'sos', 'blue_spots', 'red_spots']:
            actual_value = calculate_raw_score(self.sheet_name, feature_values_dict)
        elif self.sheet_name in ['sos_operational_context', 'distance_to_enemy_context',
                                 'distance_to_enemy_aggregator']:
            actual_value = calculate_raw_multiplier(self.sheet_name, feature_values_dict)
        # print('Actual Value for {0}:{1}'.format(self.sheet_name, actual_value))
        for feature in feature_values_dict:
            if feature in self.features:
                test_input.append(feature_values_dict[feature])
        test_input = np.array([test_input])
        predicted_value = self.pipeline.predict(test_input)[0]
        # print('Predicted Value for {0}:{1}'.format(self.sheet_name, predicted_value))
        self.actual_values.append(actual_value)
        self.predicted_values.append(predicted_value)
        return predicted_value

    def calculate_score_with_metric(self, metric_type):
        if len(self.actual_values) > 0 and len(self.predicted_values) > 0:
            if metric_type in ['mae', 'mean_absolute_error']:
                return mean_absolute_error(self.actual_values, self.predicted_values)
            elif metric_type in ['mse', 'mean_squared_error']:
                return mean_squared_error(self.actual_values, self.predicted_values)
            else:
                return -1
        else:
            return -1
