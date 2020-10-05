import sys
from os.path import dirname, realpath, join

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, precision_score
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from supervised_learning.common import MethodType, load_from_directory, ScalingType, \
    get_scaler_by_type, regressors, classifiers, select_method
from sklearn.model_selection import train_test_split

from mlm_utils import generate_neural_network, generate_scikit_model, get_scikit_model_combinations, load_training_data, \
    get_nn_model_combinations

sys.setrecursionlimit(10000)

test_sizes = [0.005, 0.01, 0.05, 0.1, 0.2]


def test_on_nn(sheet_name, features, label, method_type=MethodType.Regression):
    df_results = pd.DataFrame(
        columns=['Regressor' if method_type == MethodType.Regression else 'Classifier', 'Scaling Type',
                 'Enable Normalization', 'Use Default Params', 'Cross Validation',
                 'Mean Absolute Error' if method_type == MethodType.Regression else 'Precision',
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy'])
    X, y = train_data(sheet_name, features, label)
    combinations = get_nn_model_combinations(method_type)
    for combination in combinations:
        alpha, scaling_type, enable_normalization, output_activation = combination
        scaler = get_scaler_by_type(scaling_type)
        normalizer = Normalizer() if enable_normalization else None
        try:
            scores = np.zeros((2, len(test_sizes)))
            for i, test_size in enumerate(test_sizes):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
                model = generate_neural_network(method_type, len(X_train), len(features), alpha, output_activation)
                if scaler is not None:
                    X_train = scaler.fit_transform(X_train, y_train)
                    X_test = scaler.transform(X_test)

                if normalizer is not None:
                    X_train = normalizer.fit_transform(X_train, y_train)
                    X_test = normalizer.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores[0, i] = mean_absolute_error(y_pred,
                                                   y_test) if method_type == MethodType.Regression else precision_score(
                    y_pred, y_test)
                scores[1, i] = mean_squared_error(y_pred,
                                                  y_test) if method_type == MethodType.Regression else accuracy_score(
                    y_pred, y_test)
            df_results = df_results.append(
                {'Regressor' if method_type == MethodType.Regression else 'Classifier': method_name,
                 'Scaling Type': scaling_type.name,
                 'Enable Normalization': 'Yes' if enable_normalization else 'No',
                 'Use Default Params': 'No' if use_grid_search else 'Yes',
                 'Cross Validation': cv,
                 'Mean Absolute Error' if method_type == MethodType.Regression else 'Precision': np.mean(scores[0]),
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy': np.mean(scores[1])},
                ignore_index=True)
        except:
            continue
    df_results.to_csv(join(dirname(realpath('__file__')), 'results', '{0}_pre_trained_nn.csv'.format(sheet_name)), index=False)


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
                 'Enable Normalization', 'Use Default Params', 'Cross Validation',
                 'Mean Absolute Error' if method_type == MethodType.Regression else 'Precision',
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy'])
    X, y, = train_data(sheet_name, features, label)
    if len(features) > 1:
        poly = PolynomialFeatures()
        X = poly.fit_transform(X, y)
    combinations = get_scikit_model_combinations(method_type)
    for combination in combinations:
        method_name, scaling_type, enable_normalization, use_grid_search, cv = combination
        scaler = get_scaler_by_type(scaling_type)
        normalizer = Normalizer() if enable_normalization else None
        try:
            model = select_method(choosing_method=method_name, use_grid_search=use_grid_search, cv=cv,
                                  enable_normalization=enable_normalization, method_type=method_type)
            scores = np.zeros((2, len(test_sizes)))
            for i, test_size in enumerate(test_sizes):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
                if scaler is not None:
                    X_train = scaler.fit_transform(X_train, y_train)
                    X_test = scaler.transform(X_test)

                if normalizer is not None and method_name not in ['Linear Regression', 'Lasso', 'Ridge', 'Elastic Net']:
                    X_train = normalizer.fit_transform(X_train, y_train)
                    X_test = normalizer.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores[0, i] = mean_absolute_error(y_pred, y_test) if method_type == MethodType.Regression else precision_score(y_pred, y_test)
                scores[1, i] = mean_squared_error(y_pred, y_test) if method_type == MethodType.Regression else accuracy_score(
                    y_pred, y_test)
            df_results = df_results.append(
                {'Regressor' if method_type == MethodType.Regression else 'Classifier': method_name,
                 'Scaling Type': scaling_type.name,
                 'Enable Normalization': 'Yes' if enable_normalization else 'No',
                 'Use Default Params': 'No' if use_grid_search else 'Yes',
                 'Cross Validation': cv,
                 'Mean Absolute Error' if method_type == MethodType.Regression else 'Precision': np.mean(scores[0]),
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy': np.mean(scores[1])},
                ignore_index=True)
        except:
            continue
    
    if method_type == MethodType.Regression:
        df_filtered = df_results[df_results['Mean Squared Error'] <= 10.0]
    else:
        df_filtered = df_results[df_results['Accuracy'] >= 0.9]
    df_filtered.to_csv(join(dirname(realpath('__file__')), 'results', '{0}_pre_trained.csv'.format(sheet_name)), index=False)


def test_on_regressors(sheet_name, features, label):
    test_on_methods(sheet_name, features, label, MethodType.Regression)


def test_on_classifiers(sheet_name, features, label):
    test_on_methods(sheet_name, features, label, MethodType.Classification)
