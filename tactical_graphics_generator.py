import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir


age_of_message = 0
multiplier = 3
start_cum_message_score = 49.925
decay = 1/60

cols = ['Age of Message', 'Score (Lazy)']
df = pd.DataFrame(columns=cols)

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics'))

cum_message_cost = start_cum_message_score
while cum_message_cost >= 0:
    cum_message_cost = start_cum_message_score - (age_of_message * decay)
    if cum_message_cost < 0:
        break
    score = cum_message_cost * multiplier
    df = df.append({'Age of Message': age_of_message, 'Score (Lazy)': score}, ignore_index=True)
    age_of_message += 1
df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'tactical_graphics', 'tactical_graphics_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
