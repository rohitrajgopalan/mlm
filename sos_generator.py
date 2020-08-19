import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import numpy as np
from mlm_utils import calculate_sos_score

base = 20
decay = 4 / 60
min_age_of_message = 0
max_age_of_message = 350
min_num_blue_nodes = 0
max_num_blue_nodes = 120

cols = ['Age of Message', 'Number of blue Nodes', 'Score']

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'sos')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'sos'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'sos')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'sos'))

file_counter = 0
for age_of_message in range(min_age_of_message, max_age_of_message + 1):
    df_train = pd.DataFrame(columns=cols)
    new_data = {'Age of Message': age_of_message}
    for num_blue_nodes in range(min_num_blue_nodes, max_num_blue_nodes + 1):
        score = calculate_sos_score(age_of_message, num_blue_nodes, base, decay)
        new_data.update({'Number of blue Nodes': num_blue_nodes, 'Score': score})
        df_train = df_train.append(new_data, ignore_index=True)
    df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'sos', 'sos_{0}_{1}.csv'.format(file_counter + 1, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
    file_counter += 1

rand_generator = np.random.RandomState(0)
max_test_cases = 100
num_test_cases = 0

df_test = pd.DataFrame(columns=cols)
tuple_list = []
while num_test_cases < max_test_cases:
    age_of_message = rand_generator.randint(low=min_age_of_message, high=max_age_of_message)
    num_blue_nodes = rand_generator.randint(low=min_num_blue_nodes, high=max_num_blue_nodes)
    score = calculate_sos_score(age_of_message, num_blue_nodes, base, decay)
    new_data = {'Age of Message': age_of_message, 'Number of blue Nodes': num_blue_nodes, 'Score': score}
    new_data_tuple = (age_of_message, num_blue_nodes, score)
    if new_data_tuple in tuple_list:
        print('Duplicate SOS Row found for {0}'.format(new_data))
        continue
    try:
        tuple_list.append(new_data_tuple)
        df_test = df_test.append(new_data, ignore_index=True)
        num_test_cases += 1
    except ValueError:
        print('Duplicate SOS Row found for {0}'.format(new_data))
        continue
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'sos', 'sos_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
