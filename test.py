import math
from os import mkdir
from os.path import isdir, join, dirname, realpath

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from mlm_utils import train_data, get_scikit_model_combinations, make_pipeline


def test_on_regressors(sheet_name, features, label):
    df_results = pd.DataFrame(columns=['regressor',
                                       'scaling_type', 'enable_normalization', 'use_grid_search',
                                       'train_mae',
                                       'train_mse',
                                       'test_mae',
                                       'test_mse'])
    X, y, = train_data(sheet_name, features, label)
    # combinations = get_scikit_model_combinations_with_polynomials(len(features))
    combinations = get_scikit_model_combinations()
    for combination in combinations:
        method_name, degree, scaling_type, enable_normalization, use_grid_search = combination
        pipeline = make_pipeline(combination)
        scores = cross_validate(pipeline, X, y, cv=10,
                                scoring=('neg_mean_squared_error',
                                         'neg_mean_absolute_error'),
                                return_train_score=True)
        df_results = df_results.append({
            'regressor': method_name,
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


models = [
    # {'sheet_name': 'text_messages',
    #  'features': ['Age of Message'],
    #  'label': 'Penalty'},
    # {'sheet_name': 'tactical_graphics',
    #  'features': ['Age of Message'],
    #  'label': 'Score (Lazy)'},
    # {'sheet_name': 'sos_operational_context',
    #  'features': ['Seconds Since Last Sent SOS'],
    #  'label': 'Multiplier'
    #  },
    # {'sheet_name': 'sos',
    #  'features': ['Age of Message', 'Number of blue Nodes'],
    #  'label': 'Score'},
    {'sheet_name': 'blue_spots',
     'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                  'Average Hierarchical distance'],
     'label': 'Score'},
    # {'sheet_name':'distance_to_enemy',
    #  'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest'],
    #  'label': 'Multiplier',
    # },
    # {'sheet_name': 'red_spots',
    #  'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
    #               'Average Hierarchical distance'],
    #  'label': 'Score'}
]

if not isdir(join(dirname(realpath('__file__')), 'results')):
    mkdir(join(dirname(realpath('__file__')), 'results'))

for model in models:
    test_on_regressors(model['sheet_name'], model['features'], model['label'])
