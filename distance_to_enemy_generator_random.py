import pandas as pd
import numpy as np
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
from mlm_utils import calculate_distance_to_enemy_multiplier, calculate_distance_to_enemy_aggregator

max_nearest_values = 2000
rand_generator = np.random.RandomState(0)
cols = ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Multiplier']


def generate_data(max_num_files, max_rows):
    tuple_list1 = []
    tuple_list2 = []
    if not isdir(join(dirname(realpath('__file__')), 'datasets')):
        mkdir(join(dirname(realpath('__file__')), 'datasets'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'distance_to_enemy_context')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'distance_to_enemy_context'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'distance_to_enemy_aggregator')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'distance_to_enemy_aggregator'))

    for _ in range(max_num_files):
        df1 = pd.DataFrame(columns=cols)
        df2 = pd.DataFrame(columns=cols)
        num_rows = 0
        while num_rows < max_rows:
            applied_context = True
            applied_aggregator = True

            nearest_values = rand_generator.randint(0, max_nearest_values + 1, (5,))
            multiplier = calculate_distance_to_enemy_multiplier(nearest_values)
            aggregator = calculate_distance_to_enemy_aggregator(nearest_values)
            new_data1 = {'Multiplier': multiplier}
            new_data2 = {'Multiplier': aggregator}
            for i in range(5):
                new_data1.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                new_data2.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
            new_data_tuple1 = (
                nearest_values[0], nearest_values[1], nearest_values[2], nearest_values[3], nearest_values[4],
                multiplier)
            new_data_tuple2 = (
                nearest_values[0], nearest_values[1], nearest_values[2], nearest_values[3], nearest_values[4],
                aggregator)
            if new_data_tuple1 in tuple_list1:
                applied_context = False
                print('Duplicate Distance to Enemy Row found for {0}'.format(new_data1))
            if new_data_tuple2 in tuple_list2:
                applied_aggregator = False
                print('Duplicate Distance to Enemy Row found for {0}'.format(new_data2))

            if applied_context and applied_aggregator:
                df1 = df1.append(new_data1, ignore_index=True)
                df2 = df2.append(new_data2, ignore_index=True)
                num_rows += 1
            else:
                continue

        df1.to_csv(join(dirname(realpath('__file__')), 'datasets', 'distance_to_enemy_context',
                        'distance_to_enemy_context_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))),
                   index=False)
        df2.to_csv(join(dirname(realpath('__file__')), 'datasets', 'distance_to_enemy_aggregator',
                        'distance_to_enemy_aggregator_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))),
                   index=False)


generate_data(1000, 1000)
