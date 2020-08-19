import pandas as pd
import numpy as np
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
from mlm_utils import calculate_sos_operational_context_mutliplier

max_number_of_seconds = 240
cols = ['Seconds Since Last Sent SOS', 'Multiplier']

df_train = pd.DataFrame(columns=cols)
df_test = pd.DataFrame(columns=cols)

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'sos_operational_context')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'sos_operational_context'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'sos_operational_context')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'sos_operational_context'))

for seconds_since_last_sent_sos in range(max_number_of_seconds+1):
    df_train = df_train.append({'Seconds Since Last Sent SOS': seconds_since_last_sent_sos, 'Multiplier': calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos)}, ignore_index=True)
df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'sos_operational_context', 'sos_operational_context_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

rand_generator = np.random.RandomState(0)
max_test_cases = 100

num_test_cases = 0
tuple_list = []
while num_test_cases < max_test_cases:
    seconds_since_last_sent_sos = rand_generator.randint(low=0, high=max_number_of_seconds)
    mutliplier = calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos)
    new_data = {'Seconds Since Last Sent SOS': seconds_since_last_sent_sos,
                'Multiplier': mutliplier}
    new_data_tuple = (seconds_since_last_sent_sos, mutliplier)
    if new_data_tuple in tuple_list:
        print('Duplicate SOS Operational Context Row found for {0}'.format(new_data))
        continue
    try:
        tuple_list.append(new_data_tuple)
        df_test = df_test.append(new_data, ignore_index=True)
        num_test_cases += 1
    except ValueError:
        print('Duplicate SOS Operational Context Row found for {0}'.format(new_data))
        continue
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'sos_operational_context', 'sos_operational_context_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
