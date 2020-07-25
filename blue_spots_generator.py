import pandas as pd
from datetime import datetime
from os.path import dirname, realpath, join
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

cols = ['Distance since Last Update', 'Error Penalty', 'Number of blue Nodes', 'Score for all Nodes', 'Average Distance', 'Distance Modifier', 'Average Hierarchical distance', 'H Distance Modifier', 'Score']

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'blue_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'blue_spots'))

for distance_since_last_update in range(min_distance_since_last_update, max_distance_since_last_update+1):
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    new_data = {'Distance since Last Update': distance_since_last_update, 'Error Penalty': error_penalty}
    for num_blue_nodes in range(min_blue_nodes, max_blue_nodes+1):
        score_for_all_nodes = num_blue_nodes * error_penalty
        new_data.update({'Number of blue Nodes': num_blue_nodes, 'Score for all Nodes': score_for_all_nodes})
        for average_distance in range(min_average_distance, max_average_distance+1):
            distance_modifier = math.pow(1-0.2, (average_distance/100)-1)
            df = pd.DataFrame(columns=cols)
            new_data.update({'Average Distance': average_distance, 'Distance Modifier': distance_modifier})
            for average_hierarchical_distance in range(min_average_hierarchical_distance, max_average_hierarchical_distance+1):
                h_distance_modifier = math.pow(1-0.2, average_hierarchical_distance)
                score = score_for_all_nodes * distance_modifier * h_distance_modifier
                new_data.update({'Average Hierarchical distance': average_hierarchical_distance, 'H Distance Modifier': h_distance_modifier, 'Score': score})
                df = df.append(new_data, ignore_index=True)
            df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'blue_spots', 'blue_spots_{0}_{1}.csv'.format((distance_since_last_update+num_blue_nodes+average_distance)+1, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
