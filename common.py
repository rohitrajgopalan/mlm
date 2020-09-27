import sys
from os.path import dirname, realpath, join

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import Normalizer
from supervised_learning.common import MethodType, load_from_directory, ScalingType, \
    get_scaler_by_type, regressors, classifiers, select_method
from sklearn.model_selection import train_test_split

from mlm_utils import generate_neural_network, generate_scikit_model, get_scikit_model_combinations, load_training_data

sys.setrecursionlimit(10000)


def test_on_nn(sheet_name, features, label, method_type=MethodType.Regression):
    df_results = pd.DataFrame(
        columns=['Alpha', 'Output Activation', 'Scaling Type', 'Enable Normalization',
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy'])
    training_data, train_inputs, train_outputs, test_inputs, test_outputs = train_test_data(sheet_name, features, label)
    normalizer = Normalizer()

    for alpha in range(2, 11):
        for scaling_type in ScalingType.all():
            scaler = get_scaler_by_type(scaling_type)
            for enable_normalization in [False, True]:
                for output_activation in ['linear', 'softplus']:
                    model = generate_neural_network(method_type, len(training_data.index), len(features), alpha,
                                                    output_activation)
                    if scaler is not None:
                        train_inputs = scaler.fit_transform(train_inputs)
                        test_inputs = scaler.transform(test_inputs)
                    if enable_normalization:
                        train_inputs = normalizer.fit_transform(train_inputs)
                        test_inputs = normalizer.transform(test_inputs)
                    model.fit(train_inputs, train_outputs)
                    actual_outputs = model.predict(test_inputs)
                    score = mean_squared_error(test_outputs,
                                               actual_outputs) if method_type == MethodType.Regression else accuracy_score(
                        test_outputs, actual_outputs)
                    df_results = df_results.append(
                        {'Alpha': alpha,
                         'Output Activation': output_activation,
                         'Scaling Type': scaling_type.name,
                         'Enable Normalization': 'Yes' if enable_normalization else 'No',
                         'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy': score},
                        ignore_index=True)
    df_results.to_csv(join(dirname(realpath('__file__')), 'results', '{0}_nn.csv'.format(sheet_name)), index=False)


def train_test_data(sheet_name, features, label):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    test_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'test', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)

    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name)
    train_inputs = training_data[features]
    train_outputs = training_data[label]
    test_data = load_from_directory(test_data_files_dir, cols, True, sheet_name)
    test_inputs = test_data[features]
    test_outputs = test_data[label]

    return training_data, train_inputs, train_outputs, test_inputs, test_outputs


def train_data(sheet_name, features, label):
    data_files_dir = join(dirname(realpath('__file__')), 'datasets', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)
    data = load_from_directory(data_files_dir, cols, True, sheet_name)
    X = data[features]
    y = data[label]
    return X, y


def test_on_methods(sheet_name, features, label, method_type):
    df_results = pd.DataFrame(
        columns=['Regressor' if method_type == MethodType.Regression else 'Classifier', 'Scaling Type',
                 'Enable Normalization', 'Use Default Params', 'Cross Validation', 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy'])
    X, y = train_data(sheet_name, features, label)
    combinations = get_scikit_model_combinations(method_type)
    for combination in combinations:
        method_name, scaling_type, enable_normalization, use_grid_search, cv = combination
        scaler = get_scaler_by_type(scaling_type)
        normalizer = Normalizer() if enable_normalization else None
        try:
            model = select_method(choosing_method=method_name, use_grid_search=use_grid_search, cv=cv, enable_normalization=enable_normalization, method_type=method_type)
            test_sizes = [0.005, 0.01, 0.05, 0.1, 0.2]
            scores = np.zeros(len(test_sizes))
            for i, test_size in enumerate(test_sizes):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
                if scaler is not None:
                    X_train = scaler.fit_transform(X_train, y_train)
                    X_test = scaler.transform(X_test)

                if normalizer is not None:
                    X_train = normalizer.fit_transform(X_train, y_train)
                    X_test = normalizer.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = mean_squared_error(y_pred, y_test) if method_type == MethodType.Regression else accuracy_score(y_pred, y_test)
                scores[i] = score
            df_results = df_results.append(
                {'Regressor' if method_type == MethodType.Regression else 'Classifier': method_name,
                 'Scaling Type': scaling_type.name,
                 'Enable Normalization': 'Yes' if enable_normalization else 'No',
                 'Use Default Params': 'No' if use_grid_search else 'Yes',
                 'Cross Validation': cv,
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy': np.mean(scores)},
                ignore_index=True)
        except:
            continue

    df_results.to_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)), index=False)


def test_on_regressors(sheet_name, features, label):
    test_on_methods(sheet_name, features, label, MethodType.Regression)


def test_on_classifiers(sheet_name, features, label):
    test_on_methods(sheet_name, features, label, MethodType.Classification)
