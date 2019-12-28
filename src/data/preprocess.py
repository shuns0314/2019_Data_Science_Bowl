from typing import List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from code.src.features.make_feature import GetData


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               train_labels_df: pd.DataFrame):
    """前処理のメイン関数."""
    activities_map = encode_title(train_df, test_df)
    train_df['title'] = train_df['title'].map(activities_map)
    test_df['title'] = test_df['title'].map(activities_map)
    train_labels_df['title'] = train_labels_df['title'].map(activities_map)

    compiled_train = compile_history_data(
        df=train_df, total_num=17000)
    compiled_test = compile_history_data(
        df=test_df, total_num=1000, test_set=True)

    compiled_train.to_csv('/code/data/processed/proceeded_train.csv')
    compiled_test.to_csv('/code/data/processed/proceeded_test.csv')


def encode_title(train_df: pd.DataFrame,
                 test_df: pd.DataFrame
                 ) -> Tuple(pd.DataFrame, pd.DataFrame):
    """各DataFrameのtitleを番号に変換するためのmethod.

    return:
    activites_map = {'Sandcastle Builder (Activity)': 0,
      'Dino Drink': 1,
      'Mushroom Sorter (Assessment)': 2,
      'Crystal Caves - Level 2': 3,...}
    """
    # setに入れる重複がなくなる.
    train_title = set(train_df['title'].unique())
    test_title = set(test_df['title'].unique())
    # unionは和集合
    list_of_user_activities = list(train_title.union(test_title))
    activities_map = dict(
        zip(list_of_user_activities, np.arange(len(list_of_user_activities)))
        )
    return activities_map


def make_event_code(activities_map: dict):
    """
    return:
    event_code={0: 4100,
    1: 4100,
    2: 4100,...} <- {title: event_code}
    """
    win_code = dict(
        zip(
            activities_map.values(),
            (4100*np.ones(len(activities_map))).astype('int'))
        )
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    return win_code


def compile_history_data(df: pd.DataFrame,
                         total_num: int,
                         test_set: bool = False) -> pd.DataFrame:
    """過去のデータを、installation_idごとのデータにまとめる."""
    compiled_data: List = []
    get_data = GetData()

    for _, (_, user_sample) in tqdm(
            enumerate(df.groupby('installation_id', sort=False)),
            total=total_num):
        compiled_data += get_data(user_sample, test_set=test_set)

    return pd.DataFrame(compiled_data)
