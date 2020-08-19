import pandas as pd
import numpy as np
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import math

look_ahead_time_in_seconds = 10
distance_error_base = 0.1
min_distance_since_last_update = 0
max_distance_since_last_update = 3000
min_blue_nodes = 12
max_blue_nodes = 120
min_average_distance = 1000
max_average_distance = 10000
min_average_hierarchical_distance = 0
max_average_hierarchical_distance = 10

cols = ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance', 'Average Hierarchical distance', 'Score']

max_test_rows = 100
num_test_rows = 0
rand_generator = np.random.RandomState(0)
df_test = pd.DataFrame(columns=cols)
while num_test_rows < max_test_rows:
    distance_since_last_update = rand_generator.randint(min_distance_since_last_update, max_distance_since_last_update+1)
    new_data = {'Distance since Last Update': distance_since_last_update}
    num_blue_nodes = rand_generator.randint(min_blue_nodes, max_blue_nodes+1)
    new_data.update({'Number of blue Nodes': num_blue_nodes})
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    score_for_all_nodes = num_blue_nodes * error_penalty
    average_distance = rand_generator.randint(min_average_distance, max_average_distance+1)
    new_data.update({'Average Distance': average_distance})
    distance_modifier = math.pow(1 - 0.2, (average_distance / 100) - 1)
    average_hierarchical_distance = rand_generator.randint(min_average_hierarchical_distance, max_average_hierarchical_distance+1)
    h_distance_modifier = math.pow(1 - 0.2, average_hierarchical_distance)
    score = score_for_all_nodes * distance_modifier * h_distance_modifier
    new_data.update({'Average Hierarchical distance': average_hierarchical_distance, 'Score': score})
    try:
        df_test = df_test.append(new_data, ignore_index=True)
        num_test_rows += 1
    except ValueError:
        continue
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'blue_spots', 'blue_spots_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

