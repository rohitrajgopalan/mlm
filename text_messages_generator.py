import pandas as pd
from datetime import datetime
from os.path import dirname, realpath, join

age_of_message = 0
decay = 5 / 60
start_penalty = 49.625

cols = ['Age of Message', 'Penalty']
df = pd.DataFrame(columns=cols)
penalty = start_penalty

while penalty >= 0:
    penalty = start_penalty - (decay * age_of_message)
    if penalty < 0:
        break
    df = df.append({'Age of Message': age_of_message, 'Penalty': penalty},
                   ignore_index=True)
    age_of_message += 1
df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'text_messages',
               'text_messages_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
