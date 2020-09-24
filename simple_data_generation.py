import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import numpy as np
from mlm_utils import calculate_score

text_message_cols = ['Age of Message', 'Penalty']
tactical_graphics_cols = ['Age of Message', 'Score (Lazy)']
sos_cols = ['Age of Message', 'Number of blue Nodes', 'Score']

rand_generator = np.random.RandomState(0)


def generate_data(is_test):
    if not isdir(join(dirname(realpath('__file__')), 'datasets')):
        mkdir(join(dirname(realpath('__file__')), 'datasets'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'text_messages')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'text_messages'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'tactical_graphics')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'tactical_graphics'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'sos')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'sos'))

    text_message_data = pd.DataFrame(columns=text_message_cols)
    tactical_graphics_data = pd.DataFrame(columns=tactical_graphics_cols)
    sos_data = pd.DataFrame(columns=sos_cols)

    for age_of_message_raw in np.arange(0, 595 if is_test else 596, 5 if is_test else 1):
        age_of_message = rand_generator.randint(low=age_of_message_raw,
                                                high=age_of_message_raw + 5) if is_test else age_of_message_raw
        text_message_score = calculate_score('text_messages', age_of_message=age_of_message)
        if text_message_score > 0:
            text_message_data = text_message_data.append(
                {'Age of Message': age_of_message, 'Penalty': text_message_score}, ignore_index=True)
        tactical_graphics_score = calculate_score('tactical_graphics', age_of_message=age_of_message)
        if tactical_graphics_score > 0:
            tactical_graphics_data = tactical_graphics_data.append(
                {'Age of Message': age_of_message, 'Score (Lazy)': tactical_graphics_score}, ignore_index=True)
        if age_of_message > 350:
            continue
        for num_blue_nodes_raw in np.arange(0, 120 if is_test else 121, 5 if is_test else 1):
            num_blue_nodes = rand_generator.randint(low=num_blue_nodes_raw,
                                                    high=num_blue_nodes_raw + 5) if is_test else num_blue_nodes_raw
            sos_score = calculate_score('sos', age_of_message=age_of_message, num_blue_nodes=num_blue_nodes)
            sos_data = sos_data.append(
                {'Age of Message': age_of_message, 'Number of blue Nodes': num_blue_nodes, 'Score': sos_score}, ignore_index=True)

    text_message_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'text_messages',
             'text_messages_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
    tactical_graphics_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'tactical_graphics',
             'tactical_graphics_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
    sos_data.to_csv(join(dirname(realpath('__file__')), 'datasets', 'test' if is_test else 'train', 'sos',
                         'sos_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)


for is_test in [False, True]:
    generate_data(is_test)
