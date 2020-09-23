import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import numpy as np
from mlm_utils import calculate_tactical_graphics_score

age_of_message = 0
cols = ['Age of Message', 'Score (Lazy)']
df_train = pd.DataFrame(columns=cols)
df_test = pd.DataFrame(columns=cols)

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'tactical_graphics')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'train', 'tactical_graphics'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'tactical_graphics')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'test', 'tactical_graphics'))

score = 49.925 * 3
while score >= 0:
    score = calculate_tactical_graphics_score(age_of_message)
    if score < 0:
        break
    df_train = df_train.append({'Age of Message': age_of_message, 'Score (Lazy)': score}, ignore_index=True)
    age_of_message += 1
df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'train', 'tactical_graphics', 'tactical_graphics_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)


rand_generator = np.random.RandomState(0)
max_test_cases = 100

num_test_cases = 0
max_age_of_messages = age_of_message
tuple_list = []
while num_test_cases < max_test_cases:
    age_of_message = rand_generator.randint(low=0, high=max_age_of_messages)
    score = calculate_tactical_graphics_score(age_of_message)
    new_data = {'Age of Message': age_of_message, 'Score (Lazy)': score}
    new_data_tuple = (age_of_message, score)
    if score < 0:
        print('Score is negative for Age of message value {0}'.format(age_of_message))
        continue
    elif new_data_tuple in tuple_list:
        print('Duplicate Tactical Graphics Row found for {0}'.format(new_data))
        continue
    try:
        tuple_list.append(new_data_tuple)
        df_test = df_test.append(new_data, ignore_index=True)
        num_test_cases += 1
    except ValueError:
        print('Duplicate Tactical Graphics Row found for {0}'.format(new_data))
        continue
df_test.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test', 'tactical_graphics', 'tactical_graphics_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)