import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
max_nearest_values = 1000
cols = ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Multiplier']

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'distance_to_enemy')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'distance_to_enemy'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'distance_to_enemy')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'distance_to_enemy'))

file_counter = 0
for nearest_1 in range(max_nearest_values+1):
    multiplier_1 = 0 if nearest_1 >= 100 else 1 - (nearest_1/100)
    new_data = {'#1 Nearest': nearest_1}
    for nearest_2 in range(max_nearest_values+1):
        multiplier_2 = 0 if nearest_2 >= 100 else 1 - (nearest_2/100)
        new_data.update({'#2 Nearest': nearest_2})
        for nearest_3 in range(max_nearest_values+1):
            multiplier_3 = 0 if nearest_3 >= 100 else 1 - (nearest_3 / 100)
            new_data.update({'#3 Nearest': nearest_3})
            for nearest_4 in range(max_nearest_values+1):
                multiplier_4 = 0 if nearest_4 >= 100 else 1 - (nearest_4 / 100)
                new_data.update({'#4 Nearest': nearest_4})
                df = pd.DataFrame(columns=cols)
                for nearest_5 in range(max_nearest_values+1):
                    multiplier_5 = 0 if nearest_5 >= 100 else 1 - (nearest_5/100)
                    multiplier_sum = 1 + multiplier_1 + multiplier_2 + multiplier_3 + multiplier_4 + multiplier_5
                    new_data.update({'#5 Nearest': nearest_5, 'Multiplier': multiplier_sum})
                    df = df.append(new_data, ignore_index=True)
                df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'distance_to_enemy', 'distance_to_enemy_{0}_{1}.csv'.format(file_counter+1, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
                file_counter += 1

max_test_rows = 100
num_test_rows = 0
df_test = pd.DataFrame(columns=cols)
while num_test_rows < max_test_rows:
    new_data = {}
    try:
        df_test = df_test.append(new_data, ignore_index=True)
        num_test_rows += 1
    except ValueError:
        continue
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'distance_to_enemy', 'distance_to_enemy_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

