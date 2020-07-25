import pandas as pd
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir

max_number_of_seconds = 240
cols = ['Seconds Since Last Sent SOS', 'Multiplier']

df = pd.DataFrame(columns=cols)

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'sos_operational_context')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'sos_operational_context'))

for seconds_since_last_sent_sos in range(max_number_of_seconds+1):
    df = df.append({'Seconds Since Last Sent SOS': seconds_since_last_sent_sos, 'Multiplier': 2 if seconds_since_last_sent_sos < 121 else 1}, ignore_index=True)
df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'sos_operational_context', 'sos_operational_context_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
