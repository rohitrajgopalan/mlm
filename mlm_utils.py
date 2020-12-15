import math
from os import listdir
from os.path import dirname, realpath, join, isfile
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.tree import DecisionTreeRegressor
import enum

regressors = {'LinearRegression': LinearRegression(n_jobs=-1),
              'DecisionTree': DecisionTreeRegressor(),
              'RandomForest': RandomForestRegressor(n_jobs=-1),
              'ExtraTrees': ExtraTreesRegressor(n_jobs=-1)}


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


def select_method(choosing_method, use_grid_search=True, enable_normalization=False, cv=0):
    chosen_method = None
    methods = regressors
    if choosing_method == 'random':
        chosen_method = randomly_select_method(methods)
    else:
        for method_name in methods.keys():
            if method_name.lower() == choosing_method.lower():
                chosen_method = methods[method_name]
                break
    if choosing_method == 'LinearRegression':
        chosen_method = LinearRegression(normalize=enable_normalization, n_jobs=-1)
    if use_grid_search:
        params = get_testable_parameters(chosen_method)
        return set_up_gridsearch(chosen_method, params, cv)
    else:
        return chosen_method


def randomly_select_method(methods):
    key = random.choice(list(methods.keys()))
    return methods[key]


def set_up_gridsearch(method, params, cv):
    if not bool(params):
        return GridSearchCV(method, param_grid=params, cv=cv,
                            scoring='neg_mean_squared_error',
                            verbose=0, n_jobs=-1, refit=True)
    else:
        return method


def get_testable_parameters(method_name):
    if method_name in ['ExtraTrees', 'RandomForest']:
        return {'max_features': ['auto', 'log2', 'sqrt'],
                'criterion': ['mse', 'mae']}
    elif method_name == 'DecisionTree':
        return {'max_features': ['auto', 'log2', 'sqrt'],
                'criterion': ['mse', 'mae', 'friedman_mse'],
                'splitter': {'best', 'random'}}
    else:
        return {}


class ScalingType(enum.Enum):
    STANDARD = 1,
    MAX_ABS = 2,
    MIN_MAX = 3,
    ROBUST = 4,
    NONE = 5,

    @staticmethod
    def all():
        return [ScalingType.NONE, ScalingType.STANDARD, ScalingType.MAX_ABS, ScalingType.MIN_MAX, ScalingType.ROBUST]

    @staticmethod
    def get_type_by_name(name):
        for scaling_type in ScalingType.all():
            if scaling_type.name.lower() == name.lower():
                return scaling_type


def get_scaler_by_type(scaling_type):
    if scaling_type == ScalingType.STANDARD:
        return StandardScaler()
    elif scaling_type == ScalingType.ROBUST:
        return RobustScaler()
    elif scaling_type == ScalingType.MIN_MAX:
        return MinMaxScaler()
    elif scaling_type == ScalingType.MAX_ABS:
        return MaxAbsScaler()
    else:
        return None


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
            nearest_values = []
            for i in range(5):
                nearest_values.append(args['#{0} Nearest'.format(i + 1)])
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
        return raw_score * calculate_raw_multiplier('distance_to_enemy_context', args) * calculate_raw_multiplier('sos_operational_context', args)
    elif message_type == 'red_spots':
        return raw_score * calculate_raw_multiplier('distance_to_enemy_aggregator', args) * calculate_raw_multiplier('sos_operational_context', args)
    else:
        return raw_score


def get_scikit_model_combinations_with_polynomials(num_features=1):
    combinations = []
    degrees = [1]
    for method_name in regressors:
        for degree in degrees:
            for scaling_type in ScalingType.all():
                for enable_normalization in [False, True]:
                    for use_grid_search in [False, True]:
                        combinations.append((method_name, degree, scaling_type,
                                             enable_normalization, use_grid_search))
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
    multipliers = np.where(nearest_values < 600, (1 - (nearest_values / 600))*10, 0)
    return np.sum(multipliers) + 1


def calculate_distance_to_enemy_aggregator(nearest_values):
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


def make_pipeline(combination):
    pipeline_list = []
    method_name, degree, scaling_type, enable_normalization, use_grid_search = combination
    if degree > 1:
        pipeline_list.append(
            ('poly', PolynomialFeatures(degree=degree)))
    scaler = get_scaler_by_type(scaling_type)
    if scaler is not None:
        pipeline_list.append(('scaler', scaler))
    normalizer = Normalizer() if enable_normalization else None
    normalizer_added = False
    if normalizer is not None and not use_grid_search:
        normalizer_added = True
        pipeline_list.append(('normalizer', normalizer))
    if not normalizer_added and normalizer is not None and (
            use_grid_search and method_name not in ['Linear Regression']):
        pipeline_list.append(('normalizer', normalizer))
    method = select_method(choosing_method=method_name, use_grid_search=use_grid_search, cv=10,
                           enable_normalization=enable_normalization)
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
