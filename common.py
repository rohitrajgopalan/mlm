from os.path import dirname, realpath, join

import pandas as pd
from sklearn.metrics import mean_squared_error
from supervised_learning.common import MethodType, load_from_directory, regressors
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper


def get_regressors_with_mse_less_than_ten(sheet_name, features, label, header_index, cols_to_types):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    test_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'test', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)

    df_results = pd.DataFrame(
        columns=['Regressor', 'Enable Scaling', 'Enable Normalization', 'Use Default Params', 'Mean Squared Error'])

    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    test_data = load_from_directory(test_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    inputs = test_data[features]
    expected_outputs = test_data[label]

    for regressor in regressors:
        for use_grid_search in [False, True]:
            for enable_scaling in [False, True]:
                for enable_normalization in [False, True]:
                    print(
                        'Message Type: {0}, Regressor: {1}, Scaling?: {2}, Normalization?: {3}, Using Grid Search?: {4}'.format(
                            sheet_name,
                            regressor,
                            'Yes' if enable_scaling else 'No',
                            'Yes' if enable_normalization else 'No',
                            'Yes' if use_grid_search else 'No'))
                    try:
                        model = SupervisedLearningHelper.choose_helper(MethodType.Regression, enable_scaling,
                                                                       enable_normalization, data=training_data,
                                                                       use_grid_search=use_grid_search,
                                                                       choosing_method=regressor)
                        actual_outputs = model.predict(inputs)
                        mse = mean_squared_error(expected_outputs, actual_outputs)
                        df_results = df_results.append({'Regressor': regressor,
                                                        'Enable Scaling': 'Yes' if enable_scaling else 'No',
                                                        'Enable Normalization': 'Yes' if enable_normalization else 'No',
                                                        'Use Default Params': 'No' if use_grid_search else 'Yes',
                                                        'Mean Squared Error': mse}, ignore_index=True)
                    except:
                        continue

    df_selected_rows = df_results.loc[df_results['Mean Squared Error'] <= 10.0]
    df_selected_rows.to_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)), index=False)
