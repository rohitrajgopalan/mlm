from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir

import numpy as np
import pandas as pd

from mlm_utils import calculate_blue_spots_score

min_distance_since_last_update = 0
max_distance_since_last_update = 3000
min_blue_nodes = 0
max_blue_nodes = 120
min_average_distance = 1000
max_average_distance = 10000
min_average_hierarchical_distance = 0
max_average_hierarchical_distance = 10

cols = ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance', 'Average Hierarchical distance',
        'Score']

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))
if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))
if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'blue_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'blue_spots'))
if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))
if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'blue_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'blue_spots'))

rand_generator = np.random.RandomState(0)

increment = 5

for distance_since_last_update in np.arange(min_distance_since_last_update, max_distance_since_last_update + 1,
                                            increment):
    new_data = {'Distance since Last Update': distance_since_last_update}
    for num_blue_nodes in np.arange(min_blue_nodes, max_blue_nodes + 1, increment):
        new_data.update({'Number of blue Nodes': num_blue_nodes})
        df_train = pd.DataFrame(columns=cols)
        for average_distance in np.arange(min_average_distance, max_average_distance + 1, increment):
            new_data.update({'Average Distance': average_distance})
            for average_hierarchical_distance in np.arange(min_average_hierarchical_distance,
                                                           max_average_hierarchical_distance + 1, increment):
                score = calculate_blue_spots_score(distance_since_last_update, num_blue_nodes, average_distance,
                                                   average_hierarchical_distance)
                new_data.update({'Average Hierarchical distance': average_hierarchical_distance, 'Score': score})
                df_train = df_train.append(new_data, ignore_index=True)
        df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'blue_spots',
                             'blue_spots_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))),
                        index=False)
