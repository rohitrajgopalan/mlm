from os.path import dirname, realpath, join

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from supervised_learning.common import MethodType, load_from_directory, run_with_different_methods, regressors
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper

metrics_regressors = ['Explained Variance', 'Max Error', 'Mean Absolute Error', 'Mean Squared Error',
                      'Root Mean Squared Error',
                      'Median Absolute Error', 'R2 Score']


def get_regressors_with_best_mse(sheet_name, features, label, header_index, use_test_data=True):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    test_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'test', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)

    df_results = pd.DataFrame(columns=['Regressor', 'Enable Scaling', 'Enable Normalization', 'Mean Squared Error'])

    for regressor in regressors:
        for enable_scaling in [False, True]:
            for enable_normalization in [False, True]:
                model = SupervisedLearningHelper.choose_helper(MethodType.Regression, train_data_files_dir, features,
                                                               label,
                                                               {}, enable_scaling, None,
                                                               regressor, sheet_name, enable_normalization,
                                                               header_index)
                if use_test_data:
                    test_data = load_from_directory(test_data_files_dir, cols, {}, True, sheet_name, header_index)
                    inputs = test_data[features]
                    expected_outputs = test_data[label]
                    actual_outputs = model.predict(inputs)
                    mse = mean_squared_error(expected_outputs, actual_outputs)
                else:
                    df_each_train_file = load_from_directory(train_data_files_dir, cols, {}, False, sheet_name,
                                                             header_index)
                    mse = 0
                    for df in df_each_train_file:
                        inputs = df[features]
                        expected_outputs = df[label]
                        actual_outputs = model.predict(inputs)
                        mse += mean_squared_error(expected_outputs, actual_outputs)
                    mse = mse / len(df_each_train_file)

                df_results = df_results.append({'Regressor': regressor,
                                                'Enable Scaling': 'Yes' if enable_scaling else 'No',
                                                'Enable Normalization': 'Yes' if enable_normalization else 'No',
                                                'Mean Squared Error': mse}, ignore_index=True)
    df_results.to_csv(join(dirname(realpath('__file__')), 'results',
                           '{0}_{1}.csv'.format(sheet_name, 'test' if use_test_data else 'train')),
                      index=False)
    mse_min = np.min(df_results['Mean Squared Error'])
    df_with_mse_min = df_results.loc[df_results['Mean Squared Error'] == mse_min]
    best_regressors = []
    for index, row in df_with_mse_min.iterrows():
        best_regressors.append((row['Regressor'], row['Enable Scaling'] == 'Yes', row['Enable Normalization'] == 'Yes'))
    return best_regressors


def test_each_regressor(sheet_name, features, label, excel_writer=None):
    data_files = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)
    df_from_each_file = load_from_directory(data_files, cols, {}, False, sheet_name)
    df_models = []
    for enable_scaling in [False, True]:
        for enable_normalization in [False, True]:
            models_data = run_with_different_methods(MethodType.Regression, df_from_each_file, enable_scaling,
                                                     enable_normalization=enable_normalization)
            df_models.append(pd.DataFrame(models_data, columns=models_data.keys()))
    df_models_combined = pd.concat(df_models, ignore_index=True)
    print('Results for {0}'.format(sheet_name))
    for metric in metrics_regressors:
        if metric == 'R2 Score':
            best_row_idx = np.argmax(df_models_combined[metric])
        else:
            best_row_idx = np.argmin(df_models_combined[metric])
        best_row = df_models_combined.iloc[best_row_idx]
        print('Model with best {0} is {1} {2} scaling and {3} normalization. Value was {4}'.format(metric,
                                                                                                   best_row['Model'],
                                                                                                   'with' if best_row[
                                                                                                                 'Enable Scaling'] == 'Yes' else 'without',
                                                                                                   'with' if best_row[
                                                                                                                 'Enable Normalization'] == 'Yes' else 'without',
                                                                                                   best_row[metric]))
    if excel_writer is None:
        df_models_combined.to_csv(join(dirname(realpath('__file__')), 'results', '{0}_{1}.csv'.format(sheet_name)),
                                  index=False)
    else:
        df_models_combined.to_excel(excel_writer, sheet_name)
