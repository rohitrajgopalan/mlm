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


def generate_data():
    if not isdir(join(dirname(realpath('__file__')), 'datasets')):
        mkdir(join(dirname(realpath('__file__')), 'datasets'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'text_messages')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'text_messages'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics'))

    if not isdir(join(dirname(realpath('__file__')), 'datasets', 'sos')):
        mkdir(join(dirname(realpath('__file__')), 'datasets', 'sos'))

    text_message_data = pd.DataFrame(columns=text_message_cols)
    tactical_graphics_data = pd.DataFrame(columns=tactical_graphics_cols)
    sos_data = pd.DataFrame(columns=sos_cols)

    for age_of_message in np.arange(0, 596, 1):
        new_text_message = {'Age of Message': age_of_message}
        new_tactical_graphic = {'Age of Message': age_of_message}
        new_sos = {'Age of Message': age_of_message}
        text_message_score = calculate_score('text_messages', new_text_message)
        new_text_message.update({'Penalty': text_message_score})
        if text_message_score > 0:
            text_message_data = text_message_data.append(new_text_message, ignore_index=True)

        tactical_graphics_score = calculate_score('tactical_graphics', new_tactical_graphic)
        new_tactical_graphic.update({'Score (Lazy)': tactical_graphics_score})
        if tactical_graphics_score > 0:
            tactical_graphics_data = tactical_graphics_data.append(new_tactical_graphic, ignore_index=True)

        if age_of_message > 350:
            continue
        for num_blue_nodes in np.arange(0, 121, 1):
            new_sos.update({'Number of blue Nodes': num_blue_nodes})
            sos_score = calculate_score('sos', new_sos)
            new_sos.update({'Score': sos_score})
            sos_data = sos_data.append(new_sos, ignore_index=True)

    text_message_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'text_messages',
             'text_messages_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
    tactical_graphics_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics',
             'tactical_graphics_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
    sos_data.to_csv(join(dirname(realpath('__file__')), 'datasets', 'sos',
                         'sos_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)


generate_data()
