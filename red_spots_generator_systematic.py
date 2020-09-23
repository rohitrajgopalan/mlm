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
        '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Score']

rand_generator = np.random.RandomState(0)

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'red_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'red_spots'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'red_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'red_spots'))

increment = 5
for distance_since_last_update in np.arange(min_distance_since_last_update, max_distance_since_last_update + 1,
                                            increment):
    new_data = {'Distance since Last Update': distance_since_last_update}
    for num_blue_nodes in np.arange(min_blue_nodes, max_blue_nodes + 1, increment):
        new_data.update({'Number of blue Nodes': num_blue_nodes})
        for average_distance in np.arange(min_average_distance, max_average_distance + 1, increment):
            new_data.update({'Average Distance': average_distance})
            for average_hierarchical_distance in range(min_average_hierarchical_distance,
                                                       max_average_hierarchical_distance + 1):
                new_data.update({'Average Hierarchical distance': average_hierarchical_distance})
                for nearest_value1 in np.arange(0, max_nearest_values + 1, increment):
                    for nearest_value2 in np.arange(0, max_nearest_values + 1, increment):
                        for nearest_value3 in np.arange(0, max_nearest_values + 1, increment):
                            for nearest_value4 in np.arange(0, max_nearest_values + 1, increment):
                                df_train = pd.DataFrame(columns=cols)
                                for nearest_value5 in np.arange(0, max_nearest_values + 1, increment):
                                    nearest_values = [nearest_value1, nearest_value2, nearest_value3, nearest_value4,
                                                      nearest_value5]
                                    nearest_values = np.array(nearest_values)
                                    score = calculate_red_spots_score(distance_since_last_update, num_blue_nodes,
                                                                      average_distance, average_hierarchical_distance,
                                                                      nearest_values)
                                    new_data = {'Score': score}
                                    for i in range(5):
                                        new_data.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                                    df_train = df_train.append(new_data, ignore_index=True)
                                df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'red_spots',
                                                     'red_spots_{0}.csv'.format(
                                                         datetime.now().strftime("%Y%m%d%H%M%S"))),
                                                index=False)

for distance_since_last_update in np.arange(min_distance_since_last_update, max_distance_since_last_update + 1,
                                            increment):
    new_data = {'Distance since Last Update': rand_generator.randint(distance_since_last_update + 1,
                                                                     distance_since_last_update + increment)}
    for num_blue_nodes in np.arange(min_blue_nodes, max_blue_nodes + 1, increment):
        new_data.update(
            {'Number of blue Nodes': rand_generator.randint(num_blue_nodes + 1, num_blue_nodes + increment)})
        for average_distance in np.arange(min_average_distance, max_average_distance + 1, increment):
            new_data.update(
                {'Average Distance': rand_generator.randint(average_distance + 1, average_distance + increment)})
            for average_hierarchical_distance in range(min_average_hierarchical_distance,
                                                       max_average_hierarchical_distance + 1):
                new_data.update({'Average Hierarchical distance': average_hierarchical_distance})
                for nearest_value1 in np.arange(0, max_nearest_values + 1, increment):
                    for nearest_value2 in np.arange(0, max_nearest_values + 1, increment):
                        for nearest_value3 in np.arange(0, max_nearest_values + 1, increment):
                            for nearest_value4 in np.arange(0, max_nearest_values + 1, increment):
                                df_test = pd.DataFrame(columns=cols)
                                for nearest_value5 in np.arange(0, max_nearest_values + 1, increment):
                                    nearest_values = [
                                        rand_generator.randint(low=nearest_value1 + 1, high=nearest_value1 + increment),
                                        rand_generator.randint(low=nearest_value2 + 1, high=nearest_value2 + increment),
                                        rand_generator.randint(low=nearest_value3 + 1, high=nearest_value3 + increment),
                                        rand_generator.randint(low=nearest_value4 + 1, high=nearest_value4 + increment),
                                        rand_generator.randint(low=nearest_value5 + 1, high=nearest_value5 + increment)]
                                    nearest_values = np.array(nearest_values)
                                    score = calculate_red_spots_score(distance_since_last_update, num_blue_nodes,
                                                                      average_distance, average_hierarchical_distance,
                                                                      nearest_values)
                                    new_data = {'Score': score}
                                    for i in range(5):
                                        new_data.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                                    df_test = df_test.append(new_data, ignore_index=True)
                                df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'red_spots',
                                                    'red_spots_{0}.csv'.format(
                                                        datetime.now().strftime("%Y%m%d%H%M%S"))),
                                               index=False)
