import pandas as pd
from datetime import datetime
from os.path import dirname, realpath, join
max_nearest_values = 1000
cols = ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest', '#5 Nearest', 'Multiplier #1', 'Multiplier #2', 'Multiplier #3', 'Multiplier #4', 'Multiplier #5', 'Multiplier']


for nearest_1 in range(max_nearest_values+1):
    multiplier_1 = 0 if nearest_1 >= 100 else 1 - (nearest_1/100)
    new_data = {'#1 Nearest': nearest_1, 'Multiplier #1': multiplier_1}
    for nearest_2 in range(max_nearest_values+1):
        multiplier_2 = 0 if nearest_2 >= 100 else 1 - (nearest_2/100)
        new_data.update({'#2 Nearest': nearest_2, 'Multiplier #2': multiplier_2})
        for nearest_3 in range(max_nearest_values+1):
            multiplier_3 = 0 if nearest_3 >= 100 else 1 - (nearest_3 / 100)
            new_data.update({'#3 Nearest': nearest_3, 'Multiplier #3': multiplier_3})
            for nearest_4 in range(max_nearest_values+1):
                multiplier_4 = 0 if nearest_4 >= 100 else 1 - (nearest_4 / 100)
                new_data.update({'#4 Nearest': nearest_4, 'Multiplier #4': multiplier_4})
                df = pd.DataFrame(columns=cols)
                for nearest_5 in range(max_nearest_values+1):
                    multiplier_5 = 0 if nearest_5 >= 100 else 1 - (nearest_5/100)
                    multiplier_sum = 1 + multiplier_1 + multiplier_2 + multiplier_3 + multiplier_4 + multiplier_5
                    new_data.update({'#5 Nearest': nearest_5, 'Multiplier #5': multiplier_5, 'Multiplier': multiplier_sum})
                    df = df.append(new_data, ignore_index=True)
                df.to_csv(join(dirname(realpath('__file__')), 'datasets', 'distance_to_enemy', 'distance_to_enemy_{0}_{1}.csv'.format(nearest_1+nearest_2+nearest_3+nearest_4+1, datetime.now().strftime("%Y%m%d%H%M%S"))), index=False)
