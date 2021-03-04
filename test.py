from os import mkdir
from os.path import isdir, join, dirname, realpath
import numpy as np
import pandas as pd

message_types = ['text_messages', 'tactical_graphics', 'sos', 'blue_spots', 'red_spots']
context_types = ['sos_operational_context', 'distance_to_enemy_context', 'distance_to_enemy_aggregator']
results_dir = join(dirname(realpath('__file__')), 'results')

for model_type in context_types + message_types:
    results = pd.read_csv(join(results_dir, '{0}.csv'.format(model_type)))

    results_filtered = results[results['mae'] == results['mae'].min()]
    best_combinations = np.array(results_filtered['combination_id'])
    print('Best Models of {0}: {1}'.format(model_type, best_combinations))
