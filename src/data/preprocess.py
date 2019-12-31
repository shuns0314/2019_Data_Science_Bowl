import argparse
from typing import Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from src.features.make_feature import GetData


parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, default=False)

# /code/src/features/make_feature.py
RAW_PATH = "/code/data/raw"


def main():
    """main."""
    args = parser.parse_args()
    if args.debug is True:
        nrows = 10000
        print("DEBUG MODE: rows=10000")
    else:
        nrows = None
    train_df = pd.read_csv(f"{RAW_PATH}/train.csv", nrows=nrows)
    test_df = pd.read_csv(f"{RAW_PATH}/test.csv", nrows=nrows)
    train_labels_df = pd.read_csv(f"{RAW_PATH}/train_labels.csv", nrows=nrows)

    preprocess(train_df, test_df, train_labels_df)


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               train_labels_df: pd.DataFrame):
    """前処理のメイン関数."""
    activities_map = encode_title(train_df, test_df)
    assert len(activities_map) == 44, f'想定値: 44, 入力値: {len(activities_map)}'

    train_df['title'] = train_df['title'].map(activities_map)
    test_df['title'] = test_df['title'].map(activities_map)
    train_labels_df['title'] = train_labels_df['title'].map(activities_map)
    win_code = make_event_code(activities_map)
    assert len(win_code) == 44, f'想定値: 44, 入力値: {len(win_code)}'

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    compile_history = CompileHistory(win_code=win_code)
    compiled_train = compile_history.compile_history_data(train_df)

    compile_history = CompileHistory(win_code=win_code, test_set=True)
    compiled_test = compile_history.compile_history_data(test_df)

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    compiled_train.to_csv(f'/code/data/processed/proceeded_train_{now}.csv')
    compiled_test.to_csv(f'/code/data/processed/proceeded_test_{now}.csv')


def encode_title(train_df: pd.DataFrame,
                 test_df: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


class CompileHistory:
    def __init__(self, win_code, test_set: bool = False):
        self.win_code = win_code
        self.test_set = test_set

    def compile_history_data(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        """過去のデータを、installation_idごとのデータにまとめる."""
        get_data = GetData(win_code=self.win_code, test_set=self.test_set)
        compiled_data = Parallel(n_jobs=-1)(
            [delayed(self.get_data_for_sort)(
                user_sample, i, get_data, installation_id
                ) for i, (installation_id, user_sample) in enumerate(
                    df.groupby('installation_id', sort=False)
                    )]
        )
        compiled_data.sort(key=lambda x: x[1])
        compiled_data = [t[0] for t in compiled_data]
        if self.test_set is False:
            compiled_data = [data for inner_data in compiled_data for data in inner_data]
        # print(compiled_data)

        return pd.DataFrame(compiled_data)

    def get_data_for_sort(self,
                          data: pd.DataFrame,
                          i: int,
                          get_data: GetData,
                          installation_id: str) -> Tuple[pd.DataFrame, int]:
        compiled_data = get_data.process(data, installation_id)
        # print(f"compiled_data: {compiled_data}")
        return compiled_data, i


if __name__ == "__main__":
    main()
