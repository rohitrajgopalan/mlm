from os.path import dirname, realpath, join

import pandas as pd
from sklearn.metrics import mean_squared_error
from supervised_learning.common import MethodType, load_from_directory, regressors
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper
from neural_network.network_types import *


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
        best_regressors.append((row['Regressor'], row['Enable Scaling'] == 'Yes', row['Enable Normalization'] == 'Yes',
                                row['Mean Squared Error']))
    return best_regressors


def train_neural_network(sheet_name, features, label, header_index, cols_to_types):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    test_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'test', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)

    network_args = {
        'optimizer_type': NetworkOptimizer.ADAM,
        'loss_function': 'mse',
        'dense_layer_info_list': [
            {'num_units': 400,
             'kernel_initializer': NetworkInitializationType.ORTHOGONAL,
             'bias_initializer': NetworkInitializationType.ZEROS,
             'activation_function': NetworkActivationFunction.RELU},
            {'kernel_initializer': NetworkInitializationType.ORTHOGONAL,
             'bias_initializer': NetworkInitializationType.ZEROS,
             'num_units': 400,
             'activation_function': NetworkActivationFunction.RELU},
            {'kernel_initializer': NetworkInitializationType.ORTHOGONAL,
             'bias_initializer': NetworkInitializationType.ZEROS,
             'num_units': 400,
             'activation_function': NetworkActivationFunction.RELU},
            {'activation_function': NetworkActivationFunction.LINEAR}
        ]}

    df_results = pd.DataFrame(
        columns=['Number of Hidden layers', 'Enable Scaling', 'Enable Normalization', 'Mean Squared Error'])

    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    test_data = load_from_directory(test_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    inputs = test_data[features]
    expected_outputs = test_data[label]

    max_hidden_layers = 4

    dense_layer_info_list = [{'activation_function': NetworkActivationFunction.LINEAR}]
    for num_hidden_layers in range(0, max_hidden_layers + 1):
        i = 0
        hidden_layer_list = []
        while i < num_hidden_layers:
            hidden_layer_list.append({'num_units': 400,
                                      'kernel_initializer': NetworkInitializationType.ORTHOGONAL,
                                      'bias_initializer': NetworkInitializationType.ZEROS,
                                      'activation_function': NetworkActivationFunction.RELU})
        network_args.update({'dense_layer_info_list': hidden_layer_list.extend(dense_layer_info_list)})
        for enable_scaling in [False, True]:
            for enable_normalization in [False, True]:
                model = SupervisedLearningHelper.choose_helper(MethodType.Regression, enable_scaling,
                                                               enable_normalization, data=training_data,
                                                               dl_args=network_args, num_inputs=len(features))
                actual_outputs = model.predict(inputs)
                mse = mean_squared_error(expected_outputs, actual_outputs)
                df_results = df_results.append({'Number of Hidden layers': num_hidden_layers,
                                                'Enable Scaling': 'Yes' if enable_scaling else 'No',
                                                'Enable Normalization': 'Yes' if enable_normalization else 'No',
                                                'Mean Squared Error': mse}, ignore_index=True)
    df_results.to_csv(join(dirname(realpath('__file__')), 'results',
                           '{0}_nn.csv'.format(sheet_name)),
                      index=False)
