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
    sos_data = pd.DataFrame(columns=sos_cols)

    for age_of_message in range(601):
        new_tactical_graphic = {'Age of Message': age_of_message}
        tactical_graphics_score = calculate_raw_score('tactical_graphics', new_tactical_graphic)
        new_tactical_graphic.update({'Score (Lazy)': tactical_graphics_score})
        tactical_graphics_data = tactical_graphics_data.append(new_tactical_graphic, ignore_index=True)

        if age_of_message <= 595:
            new_text_message = {'Age of Message': age_of_message}
            text_message_score = calculate_raw_score('text_messages', new_text_message)
            new_text_message.update({'Penalty': text_message_score})
            text_message_data = text_message_data.append(new_text_message, ignore_index=True)

        new_sos = {'Age of Message': age_of_message, 'Number of blue Nodes': 12}
        new_sos.update({'Score': calculate_raw_score('sos', new_sos)})
        sos_data = sos_data.append(new_sos, ignore_index=True)

    tactical_graphics_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics',
             'tactical_graphics_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
    text_message_data.to_csv(
        join(dirname(realpath('__file__')), 'datasets', 'text_messages',
             'text_messages_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
    sos_data.to_csv(join(dirname(realpath('__file__')), 'datasets', 'sos',
                         'sos_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)


generate_data()
