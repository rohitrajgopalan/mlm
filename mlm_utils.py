import math
from os.path import dirname, realpath, join
import numpy as np
from supervised_learning.common import load_from_directory, MethodType
import pandas as pd
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper


def calculate_blue_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                               average_hierarchical_distance, look_ahead_time_in_seconds, distance_error_base):
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    score_for_all_nodes = num_blue_nodes * error_penalty
    distance_modifier = math.pow(1 - 0.2, (average_distance / 100) - 1)
    h_distance_modifier = math.pow(1 - 0.2, average_hierarchical_distance)
    score = score_for_all_nodes * distance_modifier * h_distance_modifier
    return round(score, 5)


def calculate_red_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                              average_hierarchical_distance, look_ahead_time_in_seconds, distance_error_base,
                              nearest_values):
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
    multipliers = np.where(nearest_values >= 100, 0, 1 - (nearest_values / 100))
    return np.sum(multipliers) + 1


def calculate_text_message_penalty(age_of_message, start_penalty, decay):
    penalty = start_penalty - (decay * age_of_message)
    return round(penalty, 5)


def calculate_tactical_graphics_score(age_of_message, start_cum_message_score, decay, mutliplier):
    score = (start_cum_message_score - (age_of_message * decay)) * mutliplier
    return round(score, 5)


def calculate_sos_score(age_of_message, num_blue_nodes, base, decay):
    cum_message_score = 0
    for i in range(10):
        cum_message_score += (base - ((age_of_message + i) * decay))
    cum_message_score = max(0, cum_message_score)
    score = num_blue_nodes * cum_message_score
    return round(score, 5)


def calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos):
    return 2 if seconds_since_last_sent_sos < 121 else 1


def load_model(sheet_name, cols, cols_to_types):
    model = None
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, 0, cols_to_types)
    try:
        df = pd.read_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)))
        lowest_mse = np.min(df['Mean Squared Error'])
        for index, row in df.iterrows():
            if float(row['Mean Squared Error']) == lowest_mse:
                model = SupervisedLearningHelper.choose_helper(MethodType.Regression,
                                                               row['Enable Scaling'] == 'Yes',
                                                               row['Enable Normalization'] == 'Yes',
                                                               data=training_data,
                                                               use_grid_search=row['Use Default Params'] == 'No',
                                                               choosing_method=row['Regressor'])
                break

    except:
        pass
    finally:
        if model is None:
            model = SupervisedLearningHelper.choose_helper(MethodType.Regression, False, False, data=training_data,
                                                           use_grid_search=False,
                                                           choosing_method='random')
    return model
