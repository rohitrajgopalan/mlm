import pandas as pd
import numpy as np
from datetime import datetime
from os import mkdir
from os.path import dirname, realpath, join, isdir
from mlm_utils import calculate_sos_operational_context_mutliplier

max_number_of_seconds = 600
cols = ['Seconds Since Last Sent SOS', 'Multiplier']

df_train = pd.DataFrame(columns=cols)
df_test = pd.DataFrame(columns=cols)

if not isdir(join(dirname(realpath('__file__')), 'datasets')):
    mkdir(join(dirname(realpath('__file__')), 'datasets'))

if not isdir(join(dirname(realpath('__file__')), 'datasets', 'sos_operational_context')):
    mkdir(join(dirname(realpath('__file__')), 'datasets', 'sos_operational_context'))

for seconds_since_last_sent_sos in range(max_number_of_seconds+1):
    df_train = df_train.append({'Seconds Since Last Sent SOS': seconds_since_last_sent_sos, 'Multiplier': calculate_sos_operational_context_mutliplier(seconds_since_last_sent_sos)}, ignore_index=True)
df_train.to_csv(join(dirname(realpath('__file__')), 'datasets', 'sos_operational_context', 'sos_operational_context_{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)

