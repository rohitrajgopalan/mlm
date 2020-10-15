import math
from os import mkdir
from os.path import isdir, join, dirname, realpath

import numpy
import pandas
from sklearn.model_selection import cross_validate
from supervised_learning.common import MethodType

from mlm_utils import train_data, get_scikit_model_combinations_with_polynomials, make_pipeline

test_sizes = [0.005, 0.01, 0.05, 0.1, 0.2]


def test_on_methods(sheet_name, features, label, method_type):
    df_results = pd.DataFrame(columns=['regressor' if method_type == MethodType.Regression else 'classifier',
                                       'polynomial_degree', 'polynomial_interaction_only', 'polynomial_include_bias',
                                       'scaling_type', 'enable_normalization', 'use_grid_search',
                                       'train_mae' if method_type == MethodType.Regression else 'train_precision',
                                       'train_mse' if method_type == MethodType.Regression else 'train_accuracy',
                                       'test_mae' if method_type == MethodType.Regression else 'test_precision',
                                       'test_mse' if method_type == MethodType.Regression else 'test_accuracy'])
    X, y, = train_data(sheet_name, features, label)
    combinations = get_scikit_model_combinations_with_polynomials(method_type, len(features))
    for combination in combinations:
        method_name, degree, interaction_only, include_bias, scaling_type, enable_normalization, use_grid_search = combination
        print('Developing a pipeline for {0}:'.format(sheet_name))
        pipeline = make_pipeline(combination, method_type)
        scores = cross_validate(pipeline, X, y, cv=10,
                                scoring=('neg_mean_squared_error',
                                         'neg_mean_absolute_error') if method_type == MethodType.Regression
                                else ('accuracy', 'precision_macro'),
                                return_train_score=True)
        df_results = df_results.append({
            'regressor' if method_type == MethodType.Regression else 'classifier': method_name,
            'polynomial_degree': degree,
            'polynomial_interaction_only': 'Yes' if interaction_only else 'No',
            'polynomial_include_bias': 'Yes' if include_bias else 'No',
            'scaling_type': scaling_type.name,
            'enable_normalization': 'Yes' if enable_normalization else 'No',
            'use_grid_search': 'Yes' if use_grid_search else 'No',
            'train_mae': math.fabs(np.mean(scores['train_neg_mean_absolute_error'])),
            'train_mse': math.fabs(np.mean(scores['train_neg_mean_squared_error'])),
            'test_mae': math.fabs(np.mean(scores['test_neg_mean_absolute_error'])),
            'test_mse': math.fabs(np.mean(scores['test_neg_mean_squared_error']))
        }, ignore_index=True)
    df_results.to_csv(join(dirname(realpath('__file__')), 'results', '{0}_pre_trained.csv'.format(sheet_name)),
                      index=False)


def test_on_regressors(sheet_name, features, label):
    test_on_methods(sheet_name, features, label, MethodType.Regression)


def test_on_classifiers(sheet_name, features, label):
    test_on_methods(sheet_name, features, label, MethodType.Classification)


models = [
    {'sheet_name': 'text_messages',
     'features': ['Age of Message'],
     'label': 'Penalty'},
    {'sheet_name': 'tactical_graphics',
     'features': ['Age of Message'],
     'label': 'Score (Lazy)'},
    {'sheet_name': 'sos',
     'features': ['Age of Message', 'Number of blue Nodes'],
     'label': 'Score'},
    {'sheet_name': 'blue_spots',
     'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                  'Average Hierarchical distance'],
     'label': 'Score'},
    {'sheet_name': 'red_spots',
     'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                  'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                  '#5 Nearest'],
     'label': 'Score'}
]

if not isdir(join(dirname(realpath('__file__')), 'results')):
    mkdir(join(dirname(realpath('__file__')), 'results'))

for model in models:
    test_on_regressors(model['sheet_name'], model['features'], model['label'])
