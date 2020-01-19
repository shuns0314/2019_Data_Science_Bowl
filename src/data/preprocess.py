import argparse
from typing import Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from src.features.make_feature import GetData
from src.features.tf_idf import EventClusterizer
from src.features.preprocess_feature_main import (
    MutualPreprocessFeatures, PreprocessFeatures)


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


def preprocess(train: pd.DataFrame,
               test: pd.DataFrame,
               train_labels: pd.DataFrame):
    """前処理のメイン関数."""
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # event_idの変換
    event_clusterizer = EventClusterizer(
        info_cluster_num=10, args_cluster_num=20
        )
    event_cluster = event_clusterizer.process()
    train = pd.merge(train, event_cluster,  left_on='event_id', right_index=True).sort_index()
    # train.drop('event_id', axis=1, inplace=True)
    # train = train.rename(columns={'clusters': 'event_id'})

    test = pd.merge(test, event_cluster,  left_on='event_id', right_index=True).sort_index()
    # test.drop('event_id', axis=1, inplace=True)
    # test = test.rename(columns={'clusters': 'event_id'})

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    # print(list_of_event_code)
    # print(train['event_id'])
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    list_of_info_clusters = list(set(train['info_clusters'].unique()).union(set(test['info_clusters'].unique())))
    list_of_args_clusters = list(set(train['args_clusters'].unique()).union(set(test['args_clusters'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    train['date'] = train['timestamp'].dt.date
    train['month'] = train['timestamp'].dt.month
    train['hour'] = train['timestamp'].dt.hour
    train['dayofweek'] = train['timestamp'].dt.dayofweek
    test['date'] = test['timestamp'].dt.date
    test['month'] = test['timestamp'].dt.month
    test['hour'] = test['timestamp'].dt.hour
    test['dayofweek'] = test['timestamp'].dt.dayofweek

    list_of_date = list(set(train['date'].unique()).union(set(test['date'].unique())))
    list_of_month = list(set(train['month'].unique()).union(set(test['month'].unique())))
    list_of_hour = list(set(train['hour'].unique()).union(set(test['hour'].unique())))
    list_of_dayofweek = list(set(train['dayofweek'].unique()).union(set(test['dayofweek'].unique())))

    print('list_of_date')
    print(list_of_date)

    print('title_column')
    print(train.head())
    print('asses_title')
    print(assess_titles)
    print('list_of_event_code')
    print(list_of_event_code)
    print('list_of_event_id')
    print(list_of_event_id)
    print('activities_labels')
    print(activities_labels)
    print('all_title_event_code')
    print(all_title_event_code)

    compile_history = CompileHistory(
        win_code=win_code,
        assess_titles=assess_titles,
        list_of_event_code=list_of_event_code,
        list_of_event_id=list_of_event_id,
        list_of_info_clusters=list_of_info_clusters,
        list_of_args_clusters=list_of_args_clusters,
        activities_labels=activities_labels,
        all_title_event_code=all_title_event_code,
        list_of_date=list_of_date,
        list_of_month=list_of_month,
        list_of_hour=list_of_hour,
        list_of_dayofweek=list_of_dayofweek,
        )
    compiled_train = compile_history.compile_history_data(train)

    compile_history = CompileHistory(
        win_code=win_code,
        assess_titles=assess_titles,
        list_of_event_code=list_of_event_code,
        list_of_event_id=list_of_event_id,
        list_of_info_clusters=list_of_info_clusters,
        list_of_args_clusters=list_of_args_clusters,
        activities_labels=activities_labels,
        all_title_event_code=all_title_event_code,
        list_of_date=list_of_date,
        list_of_month=list_of_month,
        list_of_hour=list_of_hour,
        list_of_dayofweek=list_of_dayofweek,
        test_set=True)
    compiled_test = compile_history.compile_history_data(test)

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    compiled_train.to_csv(f'/code/data/processed/proceeded_train_{now}.csv')
    compiled_test.to_csv(f'/code/data/processed/proceeded_test_{now}.csv')


def encode_title(train_df: pd.DataFrame,
                 test_df: pd.DataFrame
                 ) -> dict:
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
    def __init__(self,
                 win_code,
                 assess_titles,
                 list_of_event_code,
                 list_of_event_id,
                 list_of_info_clusters,
                 list_of_args_clusters,
                 activities_labels,
                 all_title_event_code,
                 list_of_date,
                 list_of_month,
                 list_of_hour,
                 list_of_dayofweek,
                 test_set: bool = False,):

        self.win_code = win_code
        self.assess_titles = assess_titles
        self.list_of_event_code = list_of_event_code
        self.list_of_event_id = list_of_event_id
        self.list_of_info_clusters = list_of_info_clusters
        self.list_of_args_clusters = list_of_args_clusters
        self.activities_labels = activities_labels
        self.all_title_event_code = all_title_event_code
        self.list_of_date = list_of_date
        self.list_of_month = list_of_month
        self.list_of_hour = list_of_hour
        self.list_of_dayofweek = list_of_dayofweek
        self.test_set = test_set

    def compile_history_data(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        """過去のデータを、installation_idごとのデータにまとめる."""
        print(self.list_of_event_code)
        get_data = GetData(
            win_code=self.win_code,
            assess_titles=self.assess_titles,
            list_of_event_code=self.list_of_event_code,
            list_of_event_id=self.list_of_event_id,
            list_of_info_clusters=self.list_of_info_clusters,
            list_of_args_clusters=self.list_of_args_clusters,
            activities_labels=self.activities_labels,
            all_title_event_code=self.all_title_event_code,
            list_of_date=self.list_of_date,
            list_of_month=self.list_of_month,
            list_of_hour=self.list_of_hour,
            list_of_dayofweek=self.list_of_dayofweek,
            test_set=self.test_set,
            )
        compiled_data = Parallel(n_jobs=-1, verbose=10)(
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
