import pandas as pd
import numpy as np
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
from mlm_utils import calculate_distance_to_enemy_multiplier

max_nearest_values = 1000
rand_generator = np.random.RandomState(0)
cols = ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Multiplier']

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'distance_to_enemy')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'distance_to_enemy'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'distance_to_enemy')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'distance_to_enemy'))

increment = 5
for nearest_value1 in np.arange(0, max_nearest_values + 1, increment):
    for nearest_value2 in np.arange(0, max_nearest_values + 1, increment):
        for nearest_value3 in np.arange(0, max_nearest_values + 1, increment):
            for nearest_value4 in np.arange(0, max_nearest_values + 1, increment):
                df_train = pd.DataFrame(columns=cols)
                for nearest_value5 in np.arange(0, max_nearest_values + 1, increment):
                    nearest_values = [nearest_value1, nearest_value2, nearest_value3, nearest_value4, nearest_value5]
                    nearest_values = np.array(nearest_values)
                    multiplier = calculate_distance_to_enemy_multiplier(nearest_values)
                    new_data = {'Multiplier': multiplier}
                    for i in range(5):
                        new_data.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                    df_train = df_train.append(new_data, ignore_index=True)
                df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'distance_to_enemy',
                                     'distance_to_enemy_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))),
                                index=False)

for nearest_value1 in np.arange(0, max_nearest_values + 1, increment):
    for nearest_value2 in np.arange(0, max_nearest_values + 1, increment):
        for nearest_value3 in np.arange(0, max_nearest_values + 1, increment):
            for nearest_value4 in np.arange(0, max_nearest_values + 1, increment):
                df_test = pd.DataFrame(columns=cols)
                for nearest_value5 in np.arange(0, max_nearest_values + 1, increment):
                    nearest_values = [rand_generator.randint(low=nearest_value1 + 1, high=nearest_value1 + increment),
                                      rand_generator.randint(low=nearest_value2 + 1, high=nearest_value2 + increment),
                                      rand_generator.randint(low=nearest_value3 + 1, high=nearest_value3 + increment),
                                      rand_generator.randint(low=nearest_value4 + 1, high=nearest_value4 + increment),
                                      rand_generator.randint(low=nearest_value5 + 1, high=nearest_value5 + increment)]
                    nearest_values = np.array(nearest_values)
                    multiplier = calculate_distance_to_enemy_multiplier(nearest_values)
                    new_data = {'Multiplier': multiplier}
                    for i in range(5):
                        new_data.update({'#{0} Nearest'.format(i + 1): nearest_values[i]})
                    df_test = df_test.append(new_data, ignore_index=True)
                df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'distance_to_enemy',
                                    'distance_to_enemy_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))),
                               index=False)
