from os.path import dirname, realpath, join

import pandas as pd
from supervised_learning.common import MethodType, load_from_directory, run_with_different_methods


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
    if excel_writer is None:
        df_models_combined.to_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)), index=False)
    else:
        df_models_combined.to_excel(excel_writer, sheet_name)
