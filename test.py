from pandas import ExcelWriter
from datetime import datetime
from common import test_each_regressor
from os import mkdir
from os.path import dirname, realpath, join, isdir
models = [{'sheet_name': 'text_messages',
           'features': ['Age of Message'],
           'label': 'Penalty'},
          {'sheet_name': 'sos_operational_context',
           'features': ['Seconds Since Last Sent SOS'],
           'label': 'Multiplier'},
          {'sheet_name': 'tactical_graphics',
           'features': ['Age of Message'],
           'label': 'Score (Lazy)'},
          {'sheet_name': 'sos',
           'features': ['Age of Message', 'Number of blue Nodes'],
           'label': 'Score'},
          # {'sheet_name': 'blue_spots',
          #  'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
          #               'Average Hierarchical distance'],
          #  'label': 'Score'},
          # {'sheet_name': 'distance_to_enemy',
          #  'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest'],
          #  'label': 'Multiplier'},
          # {'sheet_name': 'red_spots',
          #  'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
          #               'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
          #               '#5 Nearest'],
          #  'label': 'Score'}
          ]

if not isdir(join(dirname(realpath('__file__')), 'results')):
    mkdir(join(dirname(realpath('__file__')), 'results'))
try:
    with ExcelWriter(join(dirname(realpath('__file__')), 'results', 'results_{0}.xlsx'.format(datetime.now().strftime("%Y%m%d%H%M%S")))) as excel_writer:
        for model in models:
            test_each_regressor(model['sheet_name'], model['features'], model['label'], excel_writer)
except ModuleNotFoundError:
    for model in models:
        test_each_regressor(model['sheet_name'], model['features'], model['label'])
