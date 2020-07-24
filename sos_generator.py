import pandas as pd
from datetime import datetime
from os.path import dirname, realpath, join

base = 20
decay = 4 / 60
min_age_of_message = 0
max_age_of_message = 350
min_num_blue_nodes = 0
max_num_blue_nodes = 120

cols = ['Age of Message', 'Number of blue Nodes', 'Cum Message Score', 'Score']

for age_of_message in range(min_age_of_message, max_age_of_message + 1):
    cum_message_score = 0
    for i in range(10):
        cum_message_score += (base - ((age_of_message + i) * decay))
    cum_message_score = max(0, cum_message_score)
    df = pd.DataFrame(columns=cols)
    new_data = {'Age of Message': age_of_message, 'Cum Message Score': cum_message_score}
    for num_blue_nodes in range(min_num_blue_nodes, max_num_blue_nodes + 1):
        score = num_blue_nodes * cum_message_score
        new_data.update({'Number of blue Nodes': num_blue_nodes, 'Score': score})
        df = df.append(new_data, ignore_index=True)
    df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'sos', 'sos_{0}_{1}.csv'.format(age_of_message+1, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
