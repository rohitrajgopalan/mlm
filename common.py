from os.path import dirname, realpath, join

import pandas as pd
import numpy as np

from supervised_learning.common import MethodType, load_from_directory, run_with_different_methods

metrics_regressors = ['Explained Variance', 'Max Error', 'Mean Absolute Error', 'Mean Squared Error',
                      'Root Mean Squared Error',
                      'Median Absolute Error', 'R2 Score']


def test_each_regressor(sheet_name, features, label, excel_writer=None):
    data_files = join(dirname(realpath('__file__')), 'datasets', sheet_name)
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
        print('Model with best {0} is {1} {2} scaling and {3} normalization. Value was {4}'.format(metric, best_row['Model'], 'with' if best_row['Enable Scaling'] == 'Yes' else 'without', 'with' if best_row['Enable Normalization'] == 'Yes' else 'without', best_row[metric]))
    if excel_writer is None:
        df_models_combined.to_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)),
                                  index=False)
    else:
        df_models_combined.to_excel(excel_writer, sheet_name)
