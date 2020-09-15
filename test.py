from os import mkdir
from os.path import isdir


from common import *

models = [
    {'sheet_name': 'text_messages',
     'features': ['Age of Message'],
     'label': 'Penalty',
     'header': 0,
     'cols_to_types': {
         'Age of Message': 'int16',
         'Penalty': 'float8'
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
         'Score (Lazy)': 'float16'
     }},
    {'sheet_name': 'sos',
     'features': ['Age of Message', 'Number of blue Nodes'],
     'label': 'Score',
     'header': 2,
     'cols_to_types': {
         'Age of Message': 'int16',
         'Number of blue Nodes': 'int8',
         'Score': 'float16'
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
         'Multiplier': 'float8'
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
         'Score': 'float32'
     }},
    {'sheet_name': 'red_spots',
     'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                  'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                  '#5 Nearest'],
     'label': 'Score',
     'header': 2,
     'cols_to_types': {
         'Distance since Last Update': 'int16',
         'Number of blue Nodes': 'int8',
         'Average Distance': 'int16',
         'Average Hierarchical distance': 'int8',
         '#1 Nearest': 'int16',
         '#2 Nearest': 'int16',
         '#3 Nearest': 'int16',
         '#4 Nearest': 'int16',
         '#5 Nearest': 'int16',
         'Score': 'float32'
     }}
]

if not isdir(join(dirname(realpath('__file__')), 'results')):
    mkdir(join(dirname(realpath('__file__')), 'results'))

for model in models:
    if model['sheet_name'] in ['text_messages', 'tactical_graphics', 'sos', 'distance_to_enemy']:
        test_on_regressors(model['sheet_name'], model['features'], model['label'], model['header'], model['cols_to_types'])
    elif model['sheet_name'] == 'sos_operational_context':
        test_on_classifiers(model['sheet_name'], model['features'], model['label'], model['header'], model['cols_to_types'])
    else:
        test_with_svr(model['sheet_name'], model['features'], model['label'], model['header'], model['cols_to_types'])
        test_on_nn(model['sheet_name'], model['features'], model['label'], model['header'], model['cols_to_types'])
