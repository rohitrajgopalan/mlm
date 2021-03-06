from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir

import numpy as np
import pandas as pd

from mlm_utils import calculate_red_spots_score

min_distance_since_last_update = 0
max_distance_since_last_update = 3000
min_blue_nodes = 0
max_blue_nodes = 120
min_average_distance = 1000
max_average_distance = 10000
min_average_hierarchical_distance = 0
max_average_hierarchical_distance = 4
max_nearest_values = 1000

cols = ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance', 'Average Hierarchical distance',
        'Is Affected', 'Score']

rand_generator = np.random.RandomState(0)


def generate_data(max_num_files, max_rows):
    tuple_list = []
    if not isdir(join(dirname(realpath('__file__')), 'datasets')):
        mkdir(join(dirname(realpath('__file__')), 'datasets'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'red_spots')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'red_spots'))

    for _ in range(max_num_files):
        df = pd.DataFrame(columns=cols)
        num_rows = 0
        while num_rows < max_rows:
            distance_since_last_update = rand_generator.randint(min_distance_since_last_update,
                                                                max_distance_since_last_update + 1)
            new_data = {'Distance since Last Update': distance_since_last_update}
            num_blue_nodes = rand_generator.randint(min_blue_nodes, max_blue_nodes + 1)
            new_data.update({'Number of blue Nodes': num_blue_nodes})
            average_distance = rand_generator.randint(min_average_distance, max_average_distance + 1)
            new_data.update({'Average Distance': average_distance})
            is_affected = rand_generator.randint(0, 2)
            new_data.update({'Is Affected': is_affected})
            average_hierarchical_distance = rand_generator.randint(min_average_hierarchical_distance,
                                                                   max_average_hierarchical_distance + 1)
            score = calculate_red_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                                              average_hierarchical_distance, is_affected)
            new_data.update({'Average Hierarchical distance': average_hierarchical_distance, 'Score': score})
            new_data_tuple = (
                distance_since_last_update, num_blue_nodes, average_distance, average_hierarchical_distance, is_affected, score)
            if new_data_tuple in tuple_list:
                print('Duplicate Red Spots Row found for {0}'.format(new_data))
                continue
            try:
                tuple_list.append(new_data_tuple)
                df = df.append(new_data, ignore_index=True)
                num_rows += 1
            except ValueError:
                print('Duplicate Red Spots Row found for {0}'.format(new_data))
                continue

        df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'red_spots',
                       'red_spots_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))),
                  index=False)


generate_data(50, 1000)
