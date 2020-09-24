from os import mkdir
from os.path import isdir


from common import *

models = [
    {'sheet_name': 'text_messages',
     'features': ['Age of Message'],
     'label': 'Penalty'},
    # {'sheet_name': 'sos_operational_context',
    #  'features': ['Seconds Since Last Sent SOS'],
    #  'label': 'Multiplier'},
    {'sheet_name': 'tactical_graphics',
     'features': ['Age of Message'],
     'label': 'Score (Lazy)'},
    {'sheet_name': 'sos',
     'features': ['Age of Message', 'Number of blue Nodes'],
     'label': 'Score'},
    # {'sheet_name': 'distance_to_enemy',
    #  'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest'],
    #  'label': 'Multiplier'},
    # {'sheet_name': 'blue_spots',
    #  'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
    #               'Average Hierarchical distance'],
    #  'label': 'Score'},
    # {'sheet_name': 'red_spots',
    #  'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
    #               'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
    #               '#5 Nearest'],
    #  'label': 'Score'}
]

if not isdir(join(dirname(realpath('__file__')), 'results')):
    mkdir(join(dirname(realpath('__file__')), 'results'))

for model in models:
    if model['sheet_name'] == 'sos_operational_context':
        test_on_classifiers(model['sheet_name'], model['features'], model['label'])
    else:
        test_on_regressors(model['sheet_name'], model['features'], model['label'])
    # test_on_nn(model['sheet_name'], model['features'], model['label'])
