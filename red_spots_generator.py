import pandas as pd
from datetime import datetime
from os.path import dirname, realpath, join
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

cols = ['Distance since Last Update', 'Error Penalty', 'Number of blue Nodes', 'Score for all Nodes', 'Average Distance', 'Distance Modifier', 'Average Hierarchical distance', 'H Distance Modifier', '#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Multiplier #1', 'Multiplier #2', 'Multiplier #3', 'Multiplier #4', 'Multiplier #5', 'Multiplier', 'Score']

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'red_spots')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'red_spots'))

for distance_since_last_update in range(min_distance_since_last_update, max_distance_since_last_update+1):
    error_penalty = distance_since_last_update * look_ahead_time_in_seconds * distance_error_base
    new_data = {'Distance since Last Update': distance_since_last_update, 'Error Penalty': error_penalty}
    for num_blue_nodes in range(min_blue_nodes, max_blue_nodes+1):
        score_for_all_nodes = num_blue_nodes * error_penalty
        new_data.update({'Number of blue Nodes': num_blue_nodes, 'Score for all Nodes': score_for_all_nodes})
        for average_distance in range(min_average_distance, max_average_distance+1):
            new_data.update({'Average Distance': average_distance})
            for average_hierarchical_distance in range(min_average_hierarchical_distance, max_average_hierarchical_distance+1):
                h_distance_modifier = 0.2 - (0.04 * average_hierarchical_distance)
                distance_modifier = 1 if average_distance <= 100 else math.pow(1 - h_distance_modifier, (average_distance / 100) - 1)
                new_data.update({'Average Hierarchical distance': average_hierarchical_distance, 'H Distance Modifier': h_distance_modifier, 'Distance Modifier': distance_modifier})
                for nearest_1 in range(max_nearest_values + 1):
                    multiplier_1 = 0.25 if nearest_1 < 100 else 0
                    new_data.update({'#1 Nearest': nearest_1, 'Multiplier #1': multiplier_1})
                    for nearest_2 in range(max_nearest_values + 1):
                        multiplier_2 = 0.25 if nearest_2 < 100 else 0
                        new_data.update({'#2 Nearest': nearest_2, 'Multiplier #2': multiplier_2})
                        for nearest_3 in range(max_nearest_values + 1):
                            multiplier_3 = 0.25 if nearest_3 < 100 else 0
                            new_data.update({'#3 Nearest': nearest_3, 'Multiplier #3': multiplier_3})
                            for nearest_4 in range(max_nearest_values + 1):
                                multiplier_4 = 0.25 if nearest_4 < 100 else 0
                                df = pd.DataFrame(columns=cols)
                                new_data.update({'#4 Nearest': nearest_4, 'Multiplier #4': multiplier_4})
                                for nearest_5 in range(max_nearest_values + 1):
                                    multiplier_5 = 0.25 if nearest_5 < 100 else 0
                                    multiplier_sum = 1 + multiplier_1 + multiplier_2 + multiplier_3 + multiplier_4 + multiplier_5
                                    score = score_for_all_nodes * distance_modifier * h_distance_modifier * multiplier_sum
                                    new_data.update({'#5 Nearest': nearest_5, 'Multiplier #5': multiplier_5,
                                                     'Multiplier': multiplier_sum, 'Score': score})
                                    df = df.append(new_data, ignore_index=True)
                                df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'red_spots', 'red_spots_{0}_{1}.csv'.format(distance_since_last_update+num_blue_nodes+average_distance+average_hierarchical_distance+nearest_1+nearest_2+nearest_3+nearest_4+1, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
