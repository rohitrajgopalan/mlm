import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import numpy as np

age_of_message = 0
decay = 5 / 60
start_penalty = 49.625

cols = ['Age of Message', 'Penalty']
df_train = pd.DataFrame(columns=cols)
df_test = pd.DataFrame(columns=cols)
penalty = start_penalty

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'text_messages')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'text_messages'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'text_messages')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'text_messages'))

while penalty >= 0:
    penalty = start_penalty - (decay * age_of_message)
    if penalty < 0:
        break
    df_train = df_train.append({'Age of Message': age_of_message, 'Penalty': penalty},
                               ignore_index=True)
    age_of_message += 1
df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'text_messages',
                     'text_messages_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

rand_generator = np.random.RandomState(0)
max_test_cases = 100

num_test_cases = 0
max_age_of_messages = age_of_message
while num_test_cases < max_test_cases:
    age_of_message = rand_generator.randint(low=0, high=max_age_of_messages)
    try:
        df_test = df_test.append(
            {'Age of Message': age_of_message, 'Penalty': start_penalty - (decay * age_of_message)}, ignore_index=True)
        num_test_cases += 1
    except ValueError:
        continue
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'text_messages',
                    'text_messages_{0}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S'))), index=False)
