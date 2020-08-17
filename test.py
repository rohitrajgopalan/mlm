from os import mkdir
from os.path import dirname, realpath, join, isdir

from common import get_regressors_with_best_mse, show_max_min

models = [
    {'sheet_name': 'text_messages',
     'features': ['Age of Message'],
     'label': 'Penalty',
     'header': 0,
     'cols_to_types': {
         'Age of Message': 'int16',
         'Penalty': 'int8'
     }},
    {'sheet_name': 'sos_operational_context',
     'features': ['Seconds Since Last Sent SOS'],
     'label': 'Multiplier',
     'header': 2,
     'cols_to_types': {
         'Seconds Since Last Sent SOS': 'int16',
         'Multiplier': 'int8'
     }},
    {'sheet_name': 'tactical_graphics',
     'features': ['Age of Message'],
     'label': 'Score (Lazy)',
     'header': 0,
     'cols_to_types': {
         'Age of Message': 'int16',
         'Score (Lazy)': 'int16'
     }},
    {'sheet_name': 'sos',
     'features': ['Age of Message', 'Number of blue Nodes'],
     'label': 'Score',
     'header': 2,
     'cols_to_types': {
         'Age of Message': 'int16',
         'Number of blue Nodes': 'int8',
         'Score': 'int16'
     }},
    {'sheet_name': 'blue_spots',
     'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                  'Average Hierarchical distance'],
     'label': 'Score',
     'header': 2,
     'cols_to_types': {
         'Distance since Last Update': 'int16',
         'Number of blue Nodes': 'int8',
         'Average Distance': 'int16',
         'Average Hierarchical distance': 'int8',
         'Score': 'int16'
     }},
    {'sheet_name': 'distance_to_enemy',
     'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest'],
     'label': 'Multiplier',
     'header': 2,
     'cols_to_types': {
         '#1 Nearest': 'int16',
         '#2 Nearest': 'int16',
         '#3 Nearest': 'int16',
         '#4 Nearest': 'int16',
         '#5 Nearest': 'int16',
         'Multiplier': 'int8'
     }},
    # {'sheet_name': 'red_spots',
    #  'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
    #               'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
    #               '#5 Nearest'],
    #  'label': 'Score',
    #  'header': 2,
    #  'cols_to_types': {
    #      'Distance since Last Update': 'int16',
    #      'Number of blue Nodes': 'int8',
    #      'Average Distance': 'int16',
    #      'Average Hierarchical distance': 'int8',
    #      '#1 Nearest': 'int16',
    #      '#2 Nearest': 'int16',
    #      '#3 Nearest': 'int16',
    #      '#4 Nearest': 'int16',
    #      '#5 Nearest': 'int16',
    #      'Score': 'int16'
    #  }}
]

if not isdir(join(dirname(realpath('__file__')), 'results')):
    mkdir(join(dirname(realpath('__file__')), 'results'))

for model in models:
    best_regressors = get_regressors_with_best_mse(model['sheet_name'], model['features'], model['label'],
                                                   model['header'], model['cols_to_types'])
    print('Best Regressors for {0}'.format(model['sheet_name']))
    for item in best_regressors:
        print('{0} {1} scaling and {2} normalization'.format(item[0], 'with' if item[1] else 'without',
                                                             'with' if item[2] else 'without'))
    # show_max_min(model['sheet_name'], model['features'], model['label'], model['cols_to_types'])
