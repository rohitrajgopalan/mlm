import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import numpy as np
from mlm_utils import calculate_sos_score

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

df_train = pd.DataFrame(columns=cols)
increment = 5
for age_of_message in np.arange(min_age_of_message, max_age_of_message+1, increment):
    new_data = {'Age of Message': age_of_message}
    for num_blue_nodes in np.arange(min_num_blue_nodes, max_num_blue_nodes+1, increment):
        score = calculate_sos_score(age_of_message, num_blue_nodes)
        new_data.update({'Number of blue Nodes': num_blue_nodes, 'Score': score})
        df_train = df_train.append(new_data, ignore_index=True)
df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'sos', 'sos_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

rand_generator = np.random.RandomState(0)

df_test = pd.DataFrame(columns=cols)
for age_of_message_raw in np.arange(min_age_of_message, max_age_of_message+1, increment):
    age_of_message = rand_generator.randint(low=age_of_message_raw+1, high=age_of_message_raw + increment)
    new_data = {'Age of Message': age_of_message}
    for num_blue_nodes_raw in np.arange(min_num_blue_nodes, max_num_blue_nodes+1, increment):
        num_blue_nodes = rand_generator.randint(low=num_blue_nodes_raw+1, high=num_blue_nodes_raw + increment)
        score = calculate_sos_score(age_of_message, num_blue_nodes)
        new_data.update({'Number of blue Nodes': num_blue_nodes, 'Score': score})
        df_test = df_test.append(new_data, ignore_index=True)
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'sos', 'sos_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)