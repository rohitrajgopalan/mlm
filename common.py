from os.path import dirname, realpath, join

import pandas as pd
from sklearn.metrics import mean_squared_error
from supervised_learning.common import MethodType, load_from_directory, regressors
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper


def get_regressors_with_mse_less_than_one(sheet_name, features, label, header_index, cols_to_types):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    test_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'test', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)

    df_results = pd.DataFrame(columns=['Regressor', 'Enable Scaling', 'Enable Normalization', 'Mean Squared Error'])

    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    test_data = load_from_directory(test_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    inputs = test_data[features]
    expected_outputs = test_data[label]

    for regressor in regressors:
        for enable_scaling in [False, True]:
            for enable_normalization in [False, True]:
                model = SupervisedLearningHelper.choose_helper(MethodType.Regression, enable_scaling,
                                                               enable_normalization, data=training_data,
                                                               choosing_method=regressor)
                actual_outputs = model.predict(inputs)
                mse = mean_squared_error(expected_outputs, actual_outputs)
                df_results = df_results.append({'Regressor': regressor,
                                                'Enable Scaling': 'Yes' if enable_scaling else 'No',
                                                'Enable Normalization': 'Yes' if enable_normalization else 'No',
                                                'Mean Squared Error': mse}, ignore_index=True)
    df_results.to_csv(join(dirname(realpath('__file__')), 'results',
                           '{0}.csv'.format(sheet_name)),
                      index=False)
    df_selected_rows = df_results.loc[df_results['Mean Squared Error'] < 1.0]
    best_regressors = []
    for index, row in df_selected_rows.iterrows():
        best_regressors.append((row['Regressor'], row['Enable Scaling'] == 'Yes', row['Enable Normalization'] == 'Yes', row['Mean Squared Error']))
    return best_regressors
