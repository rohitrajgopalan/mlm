from mlm_utils import *
import os
from os import mkdir
from os.path import dirname, realpath, isdir, isfile, join
import pickle

context_types = {
    'sos_operational_context': {
        'features': ['Seconds Since Last Sent SOS'],
        'label': 'Multiplier',
    },
    'distance_to_enemy_context': {
        'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                     '#5 Nearest'],
        'label': 'Multiplier',
    },
    'distance_to_enemy_aggregator': {
        'features': ['#1 Nearest', '#2 Nearest', '#3 Nearest', '#4 Nearest',
                     '#5 Nearest'],
        'label': 'Multiplier',
    }
}

message_types = {
    'text_messages': {
        'features': ['Age of Message'],
        'label': 'Penalty',
    },
    'tactical_graphics': {
        'features': ['Age of Message'],
        'label': 'Score (Lazy)',
    },
    'sos': {
        'features': ['Age of Message', 'Number of blue Nodes'],
        'label': 'Score',
    },
    'blue_spots': {
        'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                     'Average Hierarchical distance', 'Is Affected'],
        'label': 'Score',
    },
    'red_spots': {
        'features': ['Distance since Last Update', 'Number of blue Nodes', 'Average Distance',
                     'Average Hierarchical distance', 'Is Affected'],
        'label': 'Score',
    }
}

combinations = get_scikit_model_combinations_with_polynomials()
datasets_dir = join(dirname(realpath('__file__')), 'datasets')
models_dir = join(dirname(realpath('__file__')), 'models')

if not isdir(models_dir):
    mkdir(models_dir)


def save_models(model_name, features, label):
    model_dir = join(models_dir, model_name)
    if not isdir(model_dir):
        mkdir(model_dir)
    X, y = train_data(model_name, features, label)
    for i, combination in enumerate(combinations):
        pipeline_model = make_pipeline(combination)
        pipeline_model.fit(X, y)
        file_name = join(model_dir, '{0}.pkl'.format(i))
        pickle.dump(pipeline_model, open(file_name, 'wb'))


for context_type in context_types:
    save_models(context_type, context_types[context_type]['features'], context_types[context_type]['label'])

for message_type in message_types:
    save_models(message_type, message_types[message_type]['features'], message_types[message_type]['label'])