import json
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def extract_event_code(train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print('--extract_event_code--')
    drop_columns = [
        'event_code',
        'event_count',
        'game_time',
        ]
    print('---extract_train---')
    extracted_train = pd.io.json.json_normalize(train_df.event_data.apply(json.loads))
    extracted_train = extracted_train.drop(drop_columns, axis=1)
    extracted_train_columns = set(extracted_train.columns)
    print('---extract_test---')
    extracted_test = pd.io.json.json_normalize(test_df.event_data.apply(json.loads))
    extracted_test = extracted_test.drop(drop_columns, axis=1)
    extracted_test_columns = set(extracted_test.columns)
    total_columns = list(extracted_train_columns.intersection(extracted_test_columns))

    extracted_train = extracted_train[total_columns]
    extracted_test = extracted_test[total_columns]

    assert len(extracted_train) == len(extracted_test), f'extracted_train: {len(extracted_train)}, extracted_test: {len(extracted_test)}'

    # 10%以下しか存在しないcolumnを取り出す
    little_column = extracted_train.count()[extracted_train.count() < len(train_df)/10].index

    # nan値なら0, それ以外なら1とする。
    extracted_train.loc[:, little_column] = np.where(
        extracted_train[little_column].isna().values, 0, 1)
    extracted_test.loc[:, little_column] = np.where(
        extracted_test[little_column].isna().values, 0, 1)

    print('---label_encoding---')
    # label_encodeing
    extracted_train, extracted_test = label_encode(
        extracted_train, extracted_test, 'description')

    extracted_train, extracted_test = label_encode(
        extracted_train, extracted_test, 'identifier')

    extracted_train, extracted_test = label_encode(
        extracted_train, extracted_test, 'media_type')

    extracted_train['coordinates'] = extracted_train['coordinates.stage_height'].astype(np.str) + extracted_train['coordinates.stage_width'].astype(np.str)
    extracted_test['coordinates'] = extracted_test['coordinates.stage_height'].astype(np.str) + extracted_test['coordinates.stage_width'].astype(np.str)

    extracted_train, extracted_test = label_encode(
        extracted_train, extracted_test, 'coordinates')
    extracted_train = extracted_train.drop(
        ['coordinates.stage_height', 'coordinates.stage_width'], axis=1)
    extracted_test = extracted_test.drop(
        ['coordinates.stage_height', 'coordinates.stage_width'], axis=1)

    extracted_train['source'] = extracted_train['source'].astype(np.str)
    extracted_test['source'] = extracted_test['source'].astype(np.str)
    extracted_train, extracted_test = label_encode(
        extracted_train, extracted_test, 'source')
    
    train_df = pd.concat([train_df, extracted_train], axis=1)
    test_df = pd.concat([test_df, extracted_test], axis=1)

    print(extracted_test.columns)
    assert len(extracted_train.columns) == len(extracted_test.columns), f'extracted_train not equal extracted_test.'
    return train_df, test_df


def label_encode(extracted_event_train, extracted_event_test, column):
    labelencoder = LabelEncoder()
    extracted_event_train[column] = extracted_event_train[column].apply(lambda x: 'nan' if x is np.NaN else x)
    extracted_event_test[column] = extracted_event_test[column].apply(lambda x: 'nan' if x is np.NaN else x)
    labelencoder.fit(extracted_event_train[column].values)
    le_dict = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    extracted_event_train[column] = extracted_event_train[column].apply(lambda x: le_dict.get(x, -1))
    extracted_event_test[column] = extracted_event_test[column].apply(lambda x: le_dict.get(x, -1))
    # extracted_event_train[column] = labelencoder.transform(
    #     extracted_event_train[column].values)
    # extracted_event_test[column] = labelencoder.transform(
    #     extracted_event_test[column].values)
    return extracted_event_train, extracted_event_test
