import json
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed


little_columns = {
    'version',
    'castles_placed',
    'molds', 'sand', 'filled', 'movie_id', 'options',
    'animals', 'round_target.size', 'round_target.type', 'round_target.animal',
    'item_type', 'position', 'animal', 'correct',
    'misses', 'holding_shell', 'has_water', 'shells', 'holes',
    'shell_size', 'hole_position', 'cloud', 'cloud_size',
    'water_level', 'time_played', 'houses', 'dinosaurs',
    'dinosaur', 'dinosaurs_placed', 'house.size', 'house.position',
    'rocket', 'height', 'launched', 'flowers', 'flower',
    'growth', 'stumps', 'destination', 'session_duration',
    'exit_type', 'distance', 'target_distances',
    'round_prompt', 'target_size', 'resources', 'object_type',
    'group', 'bug', 'buglength', 'stage_number', 'hat',
    'caterpillar', 'hats', 'caterpillars', 'bird_height', 'target_containers', 'container_type',
    'containers', 'current_containers', 'total_containers', 'toy_earned', 'object', 'previous_jars', 'bottles',
    'bottle.amount', 'bottle.color', 'jar', 'jar_filled', 'tutorial_step', 'hats_placed',
    'toy', 'diet', 'target_weight', 'weight', 'scale_weight', 'scale_contents',
    'target_water_level', 'buckets', 'target_bucket', 'mode', 'prompt', 'round_number',
    'bucket', 'buckets_placed', 'cauldron', 'layout.left.chickens',
    'layout.left.pig', 'layout.right.chickens', 'layout.right.pig', 'side',
}


def extract_small_feature(data, i):
    stash_data = data[["installation_id", 'game_session', 'type']]
    event_data = pd.io.json.json_normalize(data.event_data.apply(json.loads))
    event_data['coordinates'] = event_data['coordinates.stage_height'].astype(np.str) + event_data['coordinates.stage_width'].astype(np.str)
    event_data = event_data.drop(
        ['coordinates.stage_height', 'coordinates.stage_width'], axis=1)
    event_data['source'] = event_data['source'].astype(np.str)

    learge_df = event_data[
        ['description', 'identifier', 'media_type', 'coordinates', 'source']
        ]

    zero_df = pd.DataFrame(
        np.zeros([event_data.shape[0], len(little_columns)]),
        columns=little_columns, dtype=np.uint8)
    columns = little_columns & set(event_data.columns)
    zero_df.loc[:, columns] = np.where(
        event_data[columns].isna().values, np.uint8(0), np.uint8(1)
        ).astype(np.uint8)

    extract_df = pd.concat([zero_df, learge_df], axis=1)
    extract_df = pd.concat([extract_df, stash_data], axis=1)
    return extract_df, i


def extract_event_data(train_event, test_event) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print('---extract_test---')
    extracted_train = Parallel(n_jobs=-1)(
             [delayed(extract_small_feature)(data, i) for i, data in enumerate(train_event)]
             )
    extracted_train.sort(key=lambda x: x[1])
    extracted_train = [t[0] for t in extracted_train]
    extracted_train = pd.concat(extracted_train)
    
    del train_event

    extracted_test = Parallel(n_jobs=-1)(
             [delayed(extract_small_feature)(data, i) for i, data in enumerate(test_event)]
             )
    extracted_test.sort(key=lambda x: x[1])
    extracted_test = [t[0] for t in extracted_test]
    extracted_test = pd.concat(extracted_test)

    del test_event

    extracted_train_columns = set(extracted_train.columns)
    extracted_test_columns = set(extracted_test.columns)

    total_columns = list(
        extracted_train_columns.intersection(extracted_test_columns)
        )

    extracted_train = extracted_train[total_columns]
    extracted_test = extracted_test[total_columns]

    assert len(extracted_train) == len(extracted_test), f'extracted_train: {len(extracted_train)}, extracted_test: {len(extracted_test)}'

    # label_encodeing
    extracted_train, extracted_test = label_encode(
        extracted_train, extracted_test, 'description')

    extracted_train, extracted_test = label_encode(
        extracted_train, extracted_test, 'identifier')

    # extracted_train, extracted_test = label_encode(
    #     extracted_train, extracted_test, 'media_type')

    # extracted_train, extracted_test = label_encode(
    #     extracted_train, extracted_test, 'coordinates')

    # extracted_train, extracted_test = label_encode(
    #     extracted_train, extracted_test, 'source')

    print(extracted_test.columns)
    assert len(extracted_train.columns) == len(extracted_test.columns), f'extracted_train not equal extracted_test.'

    return extracted_train, extracted_test


def label_encode(extracted_event_train, extracted_event_test, column):
    print(f'---label_encoding({column})---')
    labelencoder = LabelEncoder()
    extracted_event_train[column] = extracted_event_train[column].astype(np.str)
    extracted_event_test[column] = extracted_event_test[column].astype(np.str)

    print(f'1')
    print(extracted_event_train[column].values)
    labelencoder.fit(extracted_event_train[column].values)
    print(f'2')
    le_dict = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    extracted_event_train[column] = extracted_event_train[column].apply(lambda x: le_dict.get(x, -1))
    extracted_event_test[column] = extracted_event_test[column].apply(lambda x: le_dict.get(x, -1))
    return extracted_event_train, extracted_event_test
