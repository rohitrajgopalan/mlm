import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
import numpy as np
from mlm_utils import calculate_raw_score

text_message_cols = ['Age of Message', 'Penalty']
tactical_graphics_cols = ['Age of Message', 'Score (Lazy)']
sos_cols = ['Age of Message', 'Number of blue Nodes', 'Score']


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

    # tactical graphics
    age_of_message = 0
    new_tactical_graphic = {'Age of Message': age_of_message}
    tactical_graphics_score = calculate_raw_score('tactical_graphics', new_tactical_graphic)
    new_tactical_graphic.update({'Score (Lazy)': tactical_graphics_score})
    tactical_graphics_data = tactical_graphics_data.append(new_tactical_graphic, ignore_index=True)
    age_of_message = 1
    while tg_score > 0:
        new_tactical_graphic = {'Age of Message': age_of_message}
        tactical_graphics_score = calculate_raw_score('tactical_graphics', new_tactical_graphic)
        if tactical_graphics_score < 0:
            break
        new_tactical_graphic.update({'Score (Lazy)': tactical_graphics_score})
        tactical_graphics_data = tactical_graphics_data.append(new_tactical_graphic, ignore_index=True)
        age_of_message += 1
    tactical_graphics_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics',
             'tactical_graphics_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

    # text message
    for age_of_message_text in range(age_of_message):
        new_text_message = {'Age of Message': age_of_message_text}
        text_message_score = calculate_raw_score('text_messages', new_text_message)
        if text_message_score < 0:
            break
        new_text_message.update({'Penalty': text_message_score})
        text_message_data = text_message_data.append(new_text_message, ignore_index=True)
    text_message_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'text_messages',
             'text_messages_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

    # sos
    for age_of_message_sos in range(age_of_message):
        new_sos = {'Age of Message': age_of_message_sos}
        sos_data = pd.DataFrame(columns=sos_cols)
        for num_blue_nodes in range(121):
            new_sos.update({'Number of blue Nodes': num_blue_nodes})
            sos_score = calculate_raw_score('sos', new_sos)
            new_sos.update({'Score': sos_score})
            sos_data = sos_data.append(new_sos, ignore_index=True)
        sos_data.to_csv(join(dirname(realpath('__file__')), 'datasets', 'sos',
                             'sos_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)


generate_data()
