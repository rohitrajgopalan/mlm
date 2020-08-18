import pandas as pd
import numpy as np
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
from mlm_utils import calculate_distance_to_enemy_multiplier

max_nearest_values = 1000
generate_test_data = True
cols = ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Multiplier']


def generate_data(data_type, max_num_files, max_rows):
    if not isdir(join(dirname(realpath('__file__')), 'datasets')):
        mkdir(join(dirname(realpath('__file__')), 'datasets'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', data_type)):
        mkdir(join(dirname(realpath('__file__')), 'datasets', data_type))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', data_type, 'distance_to_enemy')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', data_type, 'distance_to_enemy'))

    for _ in range(max_num_files):
        df = pd.DataFrame(columns=cols)
        num_rows = 0
        while num_rows < max_rows:
            nearest_values = np.random.randint(0, max_nearest_values + 1, (5,))
            new_data = {'Multiplier': calculate_distance_to_enemy_multiplier(nearest_values)}
            for i in range(5):
                new_data.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
            try:
                df = df.append(new_data, ignore_index=True)
                num_rows += 1
            except ValueError:
                print('Duplicate Distance to Enemy Row found for {0}'.format(new_data))
                continue
        df.to_csv(join(dirname(realpath('__file__')), 'datasets', data_type, 'distance_to_enemy',
                       'distance_to_enemy_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))),
                  index=False)


generate_data('train', 40, 1000)
if generate_test_data:
    generate_data('test', 1, 100)
