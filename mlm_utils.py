import math
from os.path import dirname, realpath, join
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from supervised_learning.common import load_from_directory, MethodType, ScalingType, get_scaler_by_type, regressors, \
    classifiers, select_method
import pandas as pd
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper

try:
    import keras as K
except ImportError:
    from tensorflow import keras as K


def calculate_score(message_type, args):
    if message_type == 'text_messages':
        return calculate_text_message_penalty(
            args['Age of Message'] if 'Age of Message' in args else args['age_of_message'])
    elif message_type == 'tactical_graphics':
        return calculate_tactical_graphics_score(
            args['Age of Message'] if 'Age of Message' in args else args['age_of_message'])
    elif message_type == 'sos':
        return calculate_sos_score(
            args['Age of Message'] if 'Age of Message' in args else args['age_of_message'],
            args['Number of blue Nodes'] if 'Number of blue Nodes' in args else args['num_blue_nodes'])
    elif message_type == 'blue_spots':
        return calculate_blue_spots_score(
            args['Distance since Last Update'] if 'Distance since Last Update' in args else args[
                'distance_since_last_update'],
            args['Number of blue Nodes'] if 'Number of blue Nodes' in args else args['num_blue_nodes'],
            args['Average Distance'] if 'Average Distance' in args else args['average_distance'],
            args['Average Hierarchical distance'] if 'Average Hierarchical distance' in args else args[
                'average_hierarchical_distance'])
    elif message_type == 'red_spots':
        if 'nearest_values' in args:
            nearest_values = args['nearest_values']
        else:
            nearest_values = []
            for i in range(5):
                nearest_values.append(args['#{0} Nearest'.format(i + 1)])

        return calculate_red_spots_score(
            args['Distance since Last Update'] if 'Distance since Last Update' in args else args[
                'distance_since_last_update'],
            args['Number of blue Nodes'] if 'Number of blue Nodes' in args else args['num_blue_nodes'],
            args['Average Distance'] if 'Average Distance' in args else args['average_distance'],
            args['average_hierarchical_distance'],
            nearest_values)


def get_scikit_model_combinations(method_type):
    combinations = []
    methods = regressors if method_type == MethodType.Regression else classifiers
    for method_name in methods:
        for scaling_type in ScalingType.all():
            for enable_normalization in [False, True]:
                for use_grid_search in [False, True]:
                    cross_validations = list(range(2, 11)) if use_grid_search else [1]
                    for cv in cross_validations:
                        combinations.append((method_name, scaling_type, enable_normalization, use_grid_search, cv))
    return combinations


def get_scikit_model_combinations_with_polynomials(method_type, num_features):
    combinations = []
    methods = regressors if method_type == MethodType.Regression else classifiers
    degrees = [1] if num_features == 1 else list(range(2, 6))
    interaction_only_flags = [False] if num_features == 1 else [False, True]
    include_bias_flags = [True] if num_features == 1 else [False, True]
    for method_name in methods:
        for degree in degrees:
            for interaction_only in interaction_only_flags:
                for include_bias in include_bias_flags:
                    for scaling_type in ScalingType.all():
                        for enable_normalization in [False, True]:
                            for use_grid_search in [False, True]:
                                combinations.append((method_name, degree, interaction_only, include_bias, scaling_type,
                                                     enable_normalization, use_grid_search))
    return combinations


def get_nn_model_combinations(method_type):
    combinations = []
    output_activations = ['linear', 'softplus'] if method_type == MethodType.Regression else ['sigmoid', 'softmax']
    for alpha in range(2, 11):
        for scaling_type in ScalingType.all():
            for enable_normalization in [False, True]:
                for output_activation in output_activations:
                    combinations.append((alpha, scaling_type, enable_normalization, output_activation))
    return combinations


def calculate_blue_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                               average_hierarchical_distance, look_ahead_time_in_seconds=10, distance_error_base=0.1):
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    score_for_all_nodes = num_blue_nodes * error_penalty
    distance_modifier = math.pow(1 - 0.2, (average_distance / 100) - 1)
    h_distance_modifier = math.pow(1 - 0.2, average_hierarchical_distance)
    score = score_for_all_nodes * distance_modifier * h_distance_modifier
    return round(score, 5)


def calculate_red_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                              average_hierarchical_distance, nearest_values, look_ahead_time_in_seconds=10,
                              distance_error_base=0.2):
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    score_for_all_nodes = num_blue_nodes * error_penalty
    multipliers = np.where(nearest_values < 100, 0.25, 0)
    multiplier_sum = np.sum(multipliers) + 1

    h_distance_modifier = 0.2 - (0.04 * average_hierarchical_distance)
    distance_modifier = 1 if average_distance <= 100 else math.pow(1 - h_distance_modifier,
                                                                   (average_distance / 100) - 1)
    score = score_for_all_nodes * distance_modifier * h_distance_modifier * multiplier_sum
    return round(score, 5)


def calculate_distance_to_enemy_multiplier(nearest_values):
    multipliers = np.where(nearest_values < 100, 1 - (nearest_values / 100), 0)
    return np.sum(multipliers) + 1


def calculate_text_message_penalty(age_of_message, start_penalty=49.625, decay=5 / 60):
    penalty = start_penalty - (decay * age_of_message)
    return round(penalty, 5)


def calculate_tactical_graphics_score(age_of_message, start_cum_message_score=49.925, decay=1 / 60, mutliplier=3):
    score = (start_cum_message_score - (age_of_message * decay)) * mutliplier
    return round(score, 5)


def calculate_sos_score(age_of_message, num_blue_nodes, base=20, decay=4 / 60):
    cum_message_score = 0
    for i in range(10):
        cum_message_score += (base - ((age_of_message + i) * decay))
    cum_message_score = max(0, cum_message_score)
    score = num_blue_nodes * cum_message_score
    return round(score, 5)


def calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos):
    return 2 if seconds_since_last_sent_sos < 121 else 1


def generate_neural_network(method_type, num_samples, num_features, alpha, output_activation='linear'):
    num_hidden_layers = int(num_samples / (alpha * (num_features + 1)))
    num_hidden_nodes = int(np.sqrt(num_features))
    model = K.models.Sequential()
    for i in range(num_hidden_layers):
        if i == 0:
            dense = K.layers.Dense(num_hidden_nodes, activation='relu', kernel_initializer='he_uniform',
                                   input_shape=(num_features,))
        else:
            dense = K.layers.Dense(num_hidden_nodes, activation='relu', kernel_initializer='he_uniform')
        model.add(dense)
    model.add(K.layers.Dense(1, activation=output_activation))
    model.compile(optimizer='adam', loss='mse' if method_type == MethodType.Regression else 'categorical_crossentropy')
    return model


def generate_scikit_model(method_type, data, model_name, scaling_type=ScalingType.NONE, enable_normalization=False,
                          use_grid_search=True, cv=0):
    return SupervisedLearningHelper.choose_helper(method_type, scaling_type, enable_normalization, data=data,
                                                  use_grid_search=use_grid_search, choosing_method=model_name,
                                                  cv=cv)


def load_training_data(cols, sheet_name):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    return load_from_directory(train_data_files_dir, cols, True, sheet_name)


def make_pipeline(combination, method_type):
    pipeline_list = []
    method_name, degree, interaction_only, include_bias, scaling_type, enable_normalization, use_grid_search = combination
    print(
        'Method: {0}\nPolynomial:\nDegree:{1}\nInteraction Only: {2}\nInclude Bias: {3}\nScaling Type: {4}\nEnable Normalization: {5}\nUsing Grid Search: {6}\n'.format(
            method_name, degree, 'Yes' if interaction_only else 'No', 'Yes' if include_bias else 'No',
            scaling_type.name,
            'Yes' if enable_normalization else 'No', 'Yes' if use_grid_search else 'No'))
    if degree > 1:
        pipeline_list.append(
            ('poly', PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)))
    scaler = get_scaler_by_type(scaling_type)
    if scaler is not None:
        pipeline_list.append(('scaler', scaler))
    normalizer = Normalizer() if enable_normalization else None
    normalizer_added = False
    if normalizer is not None and not use_grid_search:
        normalizer_added = True
        pipeline_list.append(('normalizer', normalizer))
    if not normalizer_added and normalizer is None and (
            use_grid_search and method_name not in ['Linear Regression', 'Lasso', 'Ridge', 'Elastic Net']):
        pipeline_list.append(('normalizer', normalizer))
    method = select_method(choosing_method=method_name, use_grid_search=use_grid_search, cv=10,
                           enable_normalization=enable_normalization, method_type=method_type)
    pipeline_list.append(('method', method))
    return Pipeline(pipeline_list)


def train_data(sheet_name, features, label, split=True):
    data_files_dir = join(dirname(realpath('__file__')), 'datasets', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)
    data = load_from_directory(data_files_dir, cols, True, sheet_name)
    if split:
        X = data[features]
        y = data[label]
        return X, y
    else:
        return data
