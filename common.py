from os.path import dirname, realpath, join

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from supervised_learning.common import MethodType, load_from_directory, regressors
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper


def show_max_min(sheet_name, features, label, cols_to_types):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)
    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, cols_to_types)
    print('Viewing max and min of {0}'.format(sheet_name))
    for col in cols:
        print('{0}: max is {1}, min is {2}'.format(col, training_data[col].max(), training_data[col].min()))


def get_regressors_with_best_mse(sheet_name, features, label, header_index, cols_to_types):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    test_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'test', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)

    df_results = pd.DataFrame(columns=['Regressor', 'Enable Scaling', 'Enable Normalization', 'Mean Squared Error'])

    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)

    for regressor in regressors:
        for enable_scaling in [False, True]:
            for enable_normalization in [False, True]:
                model = SupervisedLearningHelper.choose_helper(MethodType.Regression, enable_scaling,
                                                               enable_normalization, data=training_data,
                                                               choosing_method=regressor)
                test_data = load_from_directory(test_data_files_dir, cols, True, sheet_name, header_index)
                inputs = test_data[features]
                expected_outputs = test_data[label]
                actual_outputs = model.predict(inputs)
                mse = mean_squared_error(expected_outputs, actual_outputs)

                df_results = df_results.append({'Regressor': regressor,
                                                'Enable Scaling': 'Yes' if enable_scaling else 'No',
                                                'Enable Normalization': 'Yes' if enable_normalization else 'No',
                                                'Mean Squared Error': mse}, ignore_index=True)
    df_results.to_csv(join(dirname(realpath('__file__')), 'results',
                           '{0}.csv'.format(sheet_name)),
                      index=False)
    mse_min = np.min(df_results['Mean Squared Error'])
    df_with_mse_min = df_results.loc[df_results['Mean Squared Error'] == mse_min]
    best_regressors = []
    for index, row in df_with_mse_min.iterrows():
        best_regressors.append((row['Regressor'], row['Enable Scaling'] == 'Yes', row['Enable Normalization'] == 'Yes'))
    return best_regressors
