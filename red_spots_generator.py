import pandas as pd
import numpy as np
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import math

look_ahead_time_in_seconds = 10
distance_error_base = 0.2
min_distance_since_last_update = 0
max_distance_since_last_update = 3000
min_blue_nodes = 12
max_blue_nodes = 120
min_average_distance = 1000
max_average_distance = 10000
min_average_hierarchical_distance = 0
max_average_hierarchical_distance = 10
max_nearest_values = 1000

cols = ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance', 'Average Hierarchical distance', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Score']

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'red_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'red_spots'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'red_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'red_spots'))

file_counter = 0
for distance_since_last_update in range(min_distance_since_last_update, max_distance_since_last_update+1):
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    new_data = {'Distance since Last Update': distance_since_last_update}
    for num_blue_nodes in range(min_blue_nodes, max_blue_nodes+1):
        score_for_all_nodes = num_blue_nodes * error_penalty
        new_data.update({'Number of blue Nodes': num_blue_nodes})
        for average_distance in range(min_average_distance, max_average_distance+1):
            new_data.update({'Average Distance': average_distance})
            for average_hierarchical_distance in range(min_average_hierarchical_distance, max_average_hierarchical_distance+1):
                h_distance_modifier = 0.2 - (0.04 * average_hierarchical_distance)
                distance_modifier = 1 if average_distance <= 100 else math.pow(1 - h_distance_modifier, (average_distance / 100) - 1)
                new_data.update({'Average Hierarchical distance': average_hierarchical_distance})
                for nearest_1 in range(max_nearest_values + 1):
                    multiplier_1 = 0.25 if nearest_1 < 100 else 0
                    new_data.update({'#1 Nearest': nearest_1})
                    for nearest_2 in range(max_nearest_values + 1):
                        multiplier_2 = 0.25 if nearest_2 < 100 else 0
                        new_data.update({'#2 Nearest': nearest_2})
                        for nearest_3 in range(max_nearest_values + 1):
                            multiplier_3 = 0.25 if nearest_3 < 100 else 0
                            new_data.update({'#3 Nearest': nearest_3})
                            for nearest_4 in range(max_nearest_values + 1):
                                multiplier_4 = 0.25 if nearest_4 < 100 else 0
                                df = pd.DataFrame(columns=cols)
                                new_data.update({'#4 Nearest': nearest_4})
                                for nearest_5 in range(max_nearest_values + 1):
                                    multiplier_5 = 0.25 if nearest_5 < 100 else 0
                                    multiplier_sum = 1 + multiplier_1 + multiplier_2 + multiplier_3 + multiplier_4 + multiplier_5
                                    score = score_for_all_nodes * distance_modifier * h_distance_modifier * multiplier_sum
                                    new_data.update({'#5 Nearest': nearest_5, 'Score': score})
                                    df = df.append(new_data, ignore_index=True)
                                df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'red_spots', 'red_spots_{0}_{1}.csv'.format(file_counter+1, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
                                file_counter += 1

max_test_rows = 100
num_test_rows = 0
df_test = pd.DataFrame(columns=cols)
rand_generator = np.random.RandomState(0)
while num_test_rows < max_test_rows:
    nearest_values = np.random.randint(0, max_nearest_values+1, (5, ))
    multipliers = np.where(nearest_values < 100, 0.25, 0)
    multiplier_sum = np.sum(multipliers)+1
    distance_since_last_update = rand_generator.randint(min_distance_since_last_update, max_distance_since_last_update+1)
    new_data = {'Distance since Last Update': distance_since_last_update}
    num_blue_nodes = rand_generator.randint(min_blue_nodes, max_blue_nodes+1)
    new_data.update({'Number of blue Nodes': num_blue_nodes})
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    score_for_all_nodes = num_blue_nodes * error_penalty
    average_distance = rand_generator.randint(min_average_distance, max_average_distance+1)
    new_data.update({'Average Distance': average_distance})
    average_hierarchical_distance = rand_generator.randint(min_average_hierarchical_distance, max_average_hierarchical_distance+1)
    h_distance_modifier = 0.2 - (0.04 * average_hierarchical_distance)
    distance_modifier = 1 if average_distance <= 100 else math.pow(1 - h_distance_modifier,
                                                                   (average_distance / 100) - 1)
    score = score_for_all_nodes * distance_modifier * h_distance_modifier * multiplier_sum
    new_data.update({'Average Hierarchical distance': average_hierarchical_distance, 'Score': score})
    for i in range(5):
        new_data.update({'#{0} Nearest'.format(i+1): nearest_values[i]})
    try:
        df_test = df_test.append(new_data, ignore_index=True)
        num_test_rows += 1
    except ValueError:
        continue
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'red_spots', 'red_spots_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
