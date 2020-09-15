from os.path import dirname, realpath, join
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from supervised_learning.common import MethodType, load_from_directory, regressors, classifiers, ScalingType
from supervised_learning.supervised_learning_helper import SupervisedLearningHelper
import numpy as np
import keras as K
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler

sys.setrecursionlimit(10000)


def test_on_nn(sheet_name, features, label, header_index, cols_to_types):
    df_results = pd.DataFrame(
        columns=['Alpha', 'Output Activation', 'Scaling Type', 'Enable Normalization', 'Mean Squared Error'])
    training_data, train_inputs, train_outputs, test_inputs, test_outputs = train_test_data(sheet_name, features, label,
                                                                                            header_index, cols_to_types)
    num_hidden_nodes = int(np.sqrt(len(features)))

    normalizer = Normalizer()

    for alpha in range(2, 11):
        num_hidden_layers = int(len(training_data.index) / (alpha * (len(features) + 1)))
        for scaling_type in [ScalingType.NONE, ScalingType.STANDARD, ScalingType.ROBUST]:
            if scaling_type == ScalingType.STANDARD:
                scaler = StandardScaler()
            elif scaling_type == ScalingType.ROBUST:
                scaler = RobustScaler()
            else:
                scaler = None
            for enable_normalization in [False, True]:
                for output_activation in ['linear', 'softplus']:
                    model = K.models.Sequential()
                    for i in range(num_hidden_layers):
                        if i == 0:
                            dense = K.layers.Dense(num_hidden_nodes, activation='relu', kernel_initializer='he_normal',
                                                   bias_initializer='zeros', input_shape=(len(features),))
                        else:
                            dense = K.layers.Dense(num_hidden_nodes, activation='relu', kernel_initializer='he_normal',
                                                   bias_initializer='zeros')
                        model.add(dense)
                    model.add(K.layers.Dense(1, activation=output_activation))
                    model.compile(optimizer='adam', loss='mse')
                    if scaler is not None:
                        train_inputs = scaler.fit_transform(train_inputs)
                        test_inputs = scaler.transform(test_inputs)
                    if enable_normalization:
                        train_inputs = normalizer.fit_transform(train_inputs)
                        test_inputs = normalizer.transform(test_inputs)
                    model.fit(train_inputs, train_outputs)
                    actual_outputs = model.predict(test_inputs)
                    score = mean_squared_error(test_outputs,
                                               actual_outputs)
                    df_results = df_results.append(
                        {'Alpha': alpha,
                         'Output Activation': output_activation,
                         'Scaling Type': scaling_type.name,
                         'Enable Normalization': 'Yes' if enable_normalization else 'No',
                         'Mean Squared Error': score},
                        ignore_index=True)
    df_results.to_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)), index=False)


def train_test_data(sheet_name, features, label, header_index, cols_to_types):
    train_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'train', sheet_name)
    test_data_files_dir = join(dirname(realpath('__file__')), 'datasets', 'test', sheet_name)
    cols = [feature for feature in features]
    cols.append(label)

    training_data = load_from_directory(train_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    train_inputs = training_data[features]
    train_outputs = training_data[label]
    test_data = load_from_directory(test_data_files_dir, cols, True, sheet_name, header_index, cols_to_types)
    test_inputs = test_data[features]
    test_outputs = test_data[label]

    return training_data, train_inputs, train_outputs, test_inputs, test_outputs


def test_on_methods(sheet_name, features, label, header_index, cols_to_types, method_type):
    df_results = pd.DataFrame(
        columns=['Regressor' if method_type == MethodType.Regression else 'Classifier', 'Scaling Type',
                 'Enable Normalization', 'Use Default Params',
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy'])
    training_data, train_inputs, train_outputs, test_inputs, test_outputs = train_test_data(sheet_name, features, label,
                                                                                            header_index, cols_to_types)
    methods = regressors if method_type == MethodType.Regression else classifiers
    for method in methods:
        for use_grid_search in [False, True]:
            for scaling_type in [ScalingType.NONE, ScalingType.STANDARD, ScalingType.ROBUST]:
                for enable_normalization in [False, True]:
                    try:
                        model = SupervisedLearningHelper.choose_helper(method_type, enable_normalization, data=training_data,
                                                                       scaling_type=scaling_type,
                                                                       use_grid_search=use_grid_search,
                                                                       choosing_method=method)
                        actual_outputs = model.predict(test_inputs)
                        score = mean_squared_error(test_outputs,
                                                   actual_outputs) if method_type == MethodType.Regression else accuracy_score(
                            test_outputs, actual_outputs)
                        df_results = df_results.append(
                            {'Regressor' if method_type == MethodType.Regression else 'Classifier': method,
                             'Scaling Type': scaling_type.name,
                             'Enable Normalization': 'Yes' if enable_normalization else 'No',
                             'Use Default Params': 'No' if use_grid_search else 'Yes',
                             'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy': score},
                            ignore_index=True)
                    except:
                        continue

    df_results.to_csv(join(dirname(realpath('__file__')), 'results', '{0}.csv'.format(sheet_name)), index=False)


def test_on_regressors(sheet_name, features, label, header_index, cols_to_types):
    test_on_methods(sheet_name, features, label, header_index, cols_to_types, MethodType.Regression)


def test_on_classifiers(sheet_name, features, label, header_index, cols_to_types):
    test_on_methods(sheet_name, features, label, header_index, cols_to_types, MethodType.Classification)
