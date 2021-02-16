import math
from os import listdir
from os.path import dirname, realpath, join, isfile
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import enum

regressors = {'LinearRegression': LinearRegression(n_jobs=-1),
              'DecisionTree': DecisionTreeRegressor(),
              'RandomForest': RandomForestRegressor(n_jobs=-1)}


def load_from_directory(files_dir, cols=[], concat=False, sheet_name='', header_index=0, cols_to_types={}):
    data_files = [join(files_dir, f) for f in listdir(files_dir) if
                  isfile(join(files_dir, f))]
    df_from_each_file = []
    for f in data_files:
        df = None
        if f.endswith(".csv"):
            if not bool(cols_to_types):
                df = pd.read_csv(f, usecols=cols, dtype=cols_to_types)
            else:
                df = pd.read_csv(f, usecols=cols)
        elif f.endswith(".xls") or f.endswith(".xlsx"):
            if not bool(cols_to_types):
                if len(sheet_name) > 0:
                    df = pd.read_excel(f, sheet_name=sheet_name, header=header_index, usecols=cols, dtype=cols_to_types)
                else:
                    df = pd.read_excel(f, usecols=cols, dtype=cols_to_types)
            else:
                if len(sheet_name) > 0:
                    df = pd.read_excel(f, sheet_name=sheet_name, header=header_index, usecols=cols)
                else:
                    df = pd.read_excel(f, usecols=cols)
        df = df.dropna()
        if df is None:
            continue
        df_from_each_file.append(df)

    return pd.concat(df_from_each_file, ignore_index=True) if concat else df_from_each_file


def select_method(choosing_method, use_default_params=True, model_type=''):
    if choosing_method == "LinearRegression":
        return get_linear_regression('default' if use_default_params else model_type)
    elif choosing_method == "DecisionTree":
        return get_decision_tree('default' if use_default_params else model_type)
    elif choosing_method == "RandomForest":
        return get_random_forest('default' if use_default_params else model_type)


def get_linear_regression(model_type):
    if model_type == "default":
        return LinearRegression(n_jobs=-1)
    elif model_type in ["blue_spots", "sos_operational_context", "sos", "tactical_graphics", "text_messages"]:
        return LinearRegression(n_jobs=-1, normalize=True, fit_intercept=True)
    else:
        return LinearRegression(n_jobs=-1, normalize=True, fit_intercept=False)


def get_random_forest(model_type):
    if model_type == "sos":
        return RandomForestRegressor(n_jobs=-1, bootstrap=True, max_depth=32, max_features='auto', n_estimators=2000)
    elif model_type in ["tactical_graphics", "text_messages"]:
        return RandomForestRegressor(n_jobs=-1, bootstrap=True, max_depth=16, max_features='auto', n_estimators=2000)
    elif model_type in ["sos_operational_context", "distance_to_enemy_context", "distance_to_enemy_aggregator"]:
        return RandomForestRegressor(n_jobs=-1, bootstrap=False, max_depth=2, max_features='auto', n_estimators=100)
    elif model_type == "blue_spots":
        return RandomForestRegressor(n_jobs=-1, bootstrap=True, max_depth=16, max_features='auto', n_estimators=100)
    elif model_type == "red_spots":
        return RandomForestRegressor(n_jobs=-1, bootstrap=True, max_depth=50, max_features='auto', n_estimators=400)
    else:
        return RandomForestRegressor(n_jobs=-1)


def get_decision_tree(model_type):
    if model_type == "sos":
        return DecisionTreeRegressor(criterion='mse', max_depth=8, max_features='auto', max_leaf_nodes=100,
                                     min_samples_leaf=20, min_samples_split=10, splitter='best')
    elif model_type in ["tactical_graphics", "blue_spots"]:
        return DecisionTreeRegressor(criterion='mae', max_depth=6, max_features='auto', max_leaf_nodes=20,
                                     min_samples_leaf=20, min_samples_split=10, splitter='best')
    elif model_type == "text_messages":
        return DecisionTreeRegressor(criterion='mse', max_depth=6, max_features='auto', max_leaf_nodes=20,
                                     min_samples_leaf=20, min_samples_split=10, splitter='best')
    elif model_type in ["sos_operational_context", "distance_to_enemy_context", "distance_to_enemy_aggregator"]:
        return DecisionTreeRegressor(criterion='mse', max_depth=2, max_features='auto', max_leaf_nodes=5,
                                     min_samples_leaf=20, min_samples_split=10, splitter='best')
    elif model_type == "red_spots":
        return DecisionTreeRegressor(criterion='mae', max_depth=2, max_features='auto', max_leaf_nodes=5,
                                     min_samples_leaf=20, min_samples_split=10, splitter='best')
    else:
        return DecisionTreeRegressor()


class PreProcessingType(enum.Enum):
    SCALING = 1,
    NORMALIZATION = 2,
    NONE = 3

    @staticmethod
    def all():
        return [PreProcessingType.SCALING, PreProcessingType.NORMALIZATION, PreProcessingType.NONE]


def calculate_raw_score(message_type, args):
    if message_type == 'text_messages':
        return calculate_text_message_penalty(
            args['Age of Message'] if 'Age of Message' in args else args['ageOfMessage'])
    elif message_type == 'tactical_graphics':
        return calculate_tactical_graphics_score(
            args['Age of Message'] if 'Age of Message' in args else args['ageOfMessage'])
    elif message_type == 'sos':
        return calculate_sos_score(
            args['Age of Message'] if 'Age of Message' in args else args['ageOfMessage'],
            args['Number of blue Nodes'] if 'Number of blue Nodes' in args else args['numBlueNodes'])
    elif message_type == 'blue_spots':
        return calculate_blue_spots_score(
            args['Distance since Last Update'] if 'Distance since Last Update' in args else args[
                'distanceSinceLastUpdate'],
            args['Number of blue Nodes'] if 'Number of blue Nodes' in args else args['numBlueNodes'],
            args['Average Distance'] if 'Average Distance' in args else args['averageDistance'],
            args['Average Hierarchical distance'] if 'Average Hierarchical distance' in args else args[
                'averageHierarchicalDistance'],
            args['Is Affected'] if 'Is Affected' in args else args['isAffected'])
    elif message_type == 'red_spots':
        return calculate_red_spots_score(
            args['Distance since Last Update'] if 'Distance since Last Update' in args else args[
                'distanceSinceLastUpdate'],
            args['Number of blue Nodes'] if 'Number of blue Nodes' in args else args['numBlueNodes'],
            args['Average Distance'] if 'Average Distance' in args else args['averageDistance'],
            args['Average Hierarchical distance'] if 'Average Hierarchical distance' in args else args[
                'averageHierarchicalDistance'],
            args['Is Affected'] if 'Is Affected' in args else args['isAffected'])


def calculate_raw_multiplier(context_type, args):
    if 'distance_to_enemy' in context_type:
        if 'nearestValues' in args:
            nearest_values = args['nearestValues']
        else:
            nearest_values = np.zeros(5)
            for i in range(5):
                nearest_values[i] = args['#{0} Nearest'.format(i + 1)]
        if context_type == 'distance_to_enemy_context':
            return calculate_distance_to_enemy_multiplier(nearest_values)
        else:
            return calculate_distance_to_enemy_aggregator(nearest_values)
    elif context_type == 'sos_operational_context':
        return calculate_sos_operational_context_mutliplier(args['Seconds Since Last Sent SOS']
                                                            if 'Seconds Since Last Sent SOS' in args
                                                            else args['secondsSinceLastSOS'])


def calculate_score(message_type, args):
    raw_score = calculate_raw_score(message_type, args)
    if message_type in ['blue_spots', 'tactical_graphics', 'text_messages']:
        return raw_score * calculate_raw_multiplier('distance_to_enemy_context', args) * calculate_raw_multiplier(
            'sos_operational_context', args)
    elif message_type == 'red_spots':
        return raw_score * calculate_raw_multiplier('distance_to_enemy_aggregator', args) * calculate_raw_multiplier(
            'sos_operational_context', args)
    else:
        return raw_score


def get_scikit_model_combinations():
    combinations = []
    for method_name in regressors:
        for pre_processing_type in PreProcessingType.all():
            # combinations.append((method_name, pre_processing_type))
            for use_default_params in [True, False]:
                combinations.append((method_name, pre_processing_type, use_default_params))
    return combinations


def calculate_blue_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                               average_hierarchical_distance, is_affected=0,
                               look_ahead_time_in_seconds=10, distance_error_base=0.1):
    if is_affected == 0:
        error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
        score_for_all_nodes = num_blue_nodes * error_penalty
        distance_modifier = math.pow(1 - 0.2, (average_distance / 100) - 1)
        h_distance_modifier = math.pow(1 - 0.2, average_hierarchical_distance)
        score = score_for_all_nodes * distance_modifier * h_distance_modifier
        return round(score, 5)
    else:
        return 0


def calculate_red_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                              average_hierarchical_distance, is_affected=0,
                              look_ahead_time_in_seconds=10,
                              distance_error_base=0.2):
    if is_affected == 0:
        error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
        score_for_all_nodes = num_blue_nodes * error_penalty

        h_distance_modifier = 0.2 - (0.04 * average_hierarchical_distance)
        distance_modifier = 1 if average_distance <= 100 else math.pow(1 - h_distance_modifier,
                                                                       (average_distance / 100) - 1)
        score = score_for_all_nodes * distance_modifier * h_distance_modifier
        return round(score, 5)
    else:
        return 0


def calculate_distance_to_enemy_multiplier(nearest_values):
    nearest_values = np.array(nearest_values)
    multipliers = np.where(nearest_values < 600, (1 - (nearest_values / 600)) * 10, 0)
    return np.sum(multipliers) + 1


def calculate_distance_to_enemy_aggregator(nearest_values):
    nearest_values = np.array(nearest_values)
    multipliers = np.where(nearest_values < 600, 2.5, 0)
    return np.sum(multipliers) + 1


def calculate_text_message_penalty(age_of_message, start_penalty=49.625, decay=5 / 60):
    penalty = start_penalty - (decay * age_of_message)
    return round(penalty, 5)


def calculate_tactical_graphics_score(age_of_message, start_cum_message_score=49.925, decay=1 / 60, mutliplier=3):
    return (start_cum_message_score - (age_of_message * decay)) * mutliplier


def calculate_sos_score(age_of_message, num_blue_nodes, base=20, decay=4 / 60):
    cum_message_score = 0
    for i in range(10):
        cum_message_score += (base - ((age_of_message + i) * decay))
    cum_message_score = max(0, cum_message_score)
    score = num_blue_nodes * cum_message_score
    return round(score, 5)


def calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos):
    return 6 if seconds_since_last_sent_sos < 300 else 1


def load_training_data(cols, sheet_name):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    return load_from_directory(train_data_files_dir, cols, True, sheet_name)


def make_pipeline(combination, model_type=''):
    pipeline_list = []
    method_name, pre_processing_type, use_default_params = combination
    #method_name, pre_processing_type = combination
    if pre_processing_type == PreProcessingType.SCALING:
        pipeline_list.append(('scaler', StandardScaler()))
    elif pre_processing_type == PreProcessingType.NORMALIZATION:
        pipeline_list.append(('normalizer', Normalizer()))
    # method = select_method(choosing_method=method_name, use_default_params=use_default_params, model_type=model_type)
    method = select_method(method_name, use_default_params, model_type)
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
