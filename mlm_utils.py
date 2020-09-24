import math
from os.path import dirname, realpath, join
import numpy as np
from sklearn.preprocessing import Normalizer
from supervised_learning.common import load_from_directory, MethodType, ScalingType, get_scaler_by_type
import pandas as pd
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper

try:
    import keras as K
except ImportError:
    from tensorflow import keras as K


def calculate_score(message_type, **args):
    if message_type == 'text_messages':
        return calculate_text_message_penalty(args['age_of_message'])
    elif message_type == 'tactical_graphics':
        return calculate_tactical_graphics_score(args['age_of_message'])
    elif message_type == 'sos':
        return calculate_sos_score(args['age_of_message'], args['num_blue_nodes'])
    elif message_type == 'red_spots':
        return calculate_red_spots_score(args['distance_since_last_update'], args['num_blue_nodes'],
                                         args['average_distance'], args['average_hierarchical_distance'],
                                         args['nearest_values'])
    elif message_type == 'blue_spots':
        return calculate_blue_spots_score(args['distance_since_last_update'], args['num_blue_nodes'],
                                          args['average_distance'], args['average_hierarchical_distance'])


def calculate_multiplier(context_type, **args):
    if context_type == 'distance_to_enemy':
        return calculate_distance_to_enemy_multiplier(args['nearest_values'])
    elif context_type == 'sos_operational_context':
        return calculate_sos_operational_context_mutliplier(args['seconds_since_last_sos'])


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


def load_best_scikit_model(training_data, sheet_name, method_type=MethodType.Regression):
    try:
        df = pd.read_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)))
        if method_type == MethodType.Regression:
            lowest_mse = np.min(df['Mean Squared Error'])
            for index, row in df.iterrows():
                if float(row['Mean Squared Error']) == lowest_mse:
                    return generate_scikit_model(method_type, training_data, row['Regressor'], row['Scaling Type'],
                                                 row['Enable Normalization'] == 'Yes',
                                                 row['Use Default Params'] == 'No'), lowest_mse, get_scaler_by_type(
                        row['Scaling Type']), Normalizer() if row['Enable Normalization'] == 'Yes' else None
        else:
            highest_accuracy = np.max(df['Accuracy'])
            for index, row in df.iterrows():
                if float(row['Accuracy']) == highest_accuracy:
                    return generate_scikit_model(method_type, training_data, row['Regressor'], row['Scaling Type'],
                                                 row['Enable Normalization'] == 'Yes',
                                                 row[
                                                     'Use Default Params'] == 'No'), highest_accuracy, get_scaler_by_type(
                        row['Scaling Type']), Normalizer() if row['Enable Normalization'] == 'Yes' else None
    except:
        return None, -1, None, False


def load_best_nn_model(training_data, sheet_name, features, method_type=MethodType.Regression):
    try:
        df = pd.read_csv(join(dirname(realpath('__file__')), 'results', '{0}_nn.csv'.format(sheet_name)))
        if method_type == MethodType.Regression:
            lowest_mse = np.min(df['Mean Squared Error'])
            for index, row in df.iterrows():
                if float(row['Mean Squared Error']) == lowest_mse:
                    return generate_neural_network(method_type, len(training_data.index), len(features),
                                                   row['Alpha'], row['Output Activation'],
                                                   ), lowest_mse, get_scaler_by_type(
                        row['Scaling Type']), Normalizer() if row['Enable Normalization'] == 'Yes' else None
        else:
            highest_accuracy = np.max(df['Accuracy'])
            for index, row in df.iterrows():
                if float(row['Accuracy']) == highest_accuracy:
                    return generate_neural_network(method_type, len(training_data.index), len(features),
                                                   row['Alpha'], row['Output Activation'],
                                                   ), highest_accuracy, get_scaler_by_type(
                        row['Scaling Type']), Normalizer() if row['Enable Normalization'] == 'Yes' else None
    except:
        return None, -1, None, False


def load_best_model(sheet_name, cols, cols_to_types, method_type=MethodType.Regression):
    training_data = load_training_data(cols, cols_to_types, sheet_name)
    sk_model, sk_score, sk_scaler, sk_normalizer = load_best_scikit_model(training_data, sheet_name, method_type)
    features = cols[0: len(cols) - 1]
    label = cols[-1]
    nn_model, nn_score, nn_scaler, nn_normalizer = load_best_nn_model(training_data, sheet_name, features, method_type)
    train_inputs = training_data[features]
    train_outputs = training_data[label]
    if nn_scaler is not None:
        train_inputs = nn_scaler.fit_transform(train_inputs)
    if nn_normalizer is not None:
        train_inputs = nn_normalizer.fit_transform(train_inputs)
    nn_model.fit(train_inputs, train_outputs)
    if sk_score == -1 and nn_score > -1:
        return nn_model, nn_scaler, nn_normalizer, 'nn'
    elif nn_score == -1 and sk_score > -1:
        return sk_model, sk_scaler, sk_normalizer, 'scikit'
    else:
        if method_type == MethodType.Regression:
            if sk_score < nn_score:
                return sk_model, sk_scaler, sk_normalizer, 'scikit'
            else:
                return nn_model, nn_scaler, nn_normalizer, 'nn'
        else:
            if sk_score > nn_score:
                return sk_model, sk_scaler, sk_normalizer, 'scikit'
            else:
                return nn_model, nn_scaler, nn_normalizer, 'nn'


def load_training_data(cols, cols_to_types, sheet_name):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, 0, cols_to_types)
    return training_data
