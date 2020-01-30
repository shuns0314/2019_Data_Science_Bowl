# val_loss(seed=3): 0.5717663690568516
# val_loss(seed=3): kaggle_kernel 0.5600336315838049
# import pickle
# import glob
from typing import Tuple, Dict
import random
from collections import Counter, defaultdict
import json

import lightgbm as lgb
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


# /code/src/features/make_feature.py
RAW_PATH = "/code/data/raw"

SEED = 3


###############################################################################
# compile
###############################################################################

def compile_main(train: pd.DataFrame,
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
    train = pd.merge(train, event_cluster, left_on='event_id', right_index=True).sort_index()
    # train.drop('event_id', axis=1, inplace=True)
    # train = train.rename(columns={'clusters': 'event_id'})

    test = pd.merge(test, event_cluster, left_on='event_id', right_index=True).sort_index()
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
    # list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    # activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(
        set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
            set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    # train['world'] = train['world'].map(activities_world)
    # test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    train['hour'] = train['timestamp'].dt.hour
    train['dayofweek'] = train['timestamp'].dt.dayofweek
    test['hour'] = test['timestamp'].dt.hour
    test['dayofweek'] = test['timestamp'].dt.dayofweek

    list_of_hour = list(set(train['hour'].unique()).union(set(test['hour'].unique())))
    list_of_dayofweek = list(set(train['dayofweek'].unique()).union(set(test['dayofweek'].unique())))

    del train_labels
    compile_history = CompileHistory(
        win_code=win_code,
        assess_titles=assess_titles,
        list_of_event_code=list_of_event_code,
        list_of_event_id=list_of_event_id,
        list_of_info_clusters=list_of_info_clusters,
        list_of_args_clusters=list_of_args_clusters,
        activities_labels=activities_labels,
        all_title_event_code=all_title_event_code,
        list_of_hour=list_of_hour,
        list_of_dayofweek=list_of_dayofweek,
        )
    train = compile_history.compile_history_data(train)

    compile_history = CompileHistory(
        win_code=win_code,
        assess_titles=assess_titles,
        list_of_event_code=list_of_event_code,
        list_of_event_id=list_of_event_id,
        list_of_info_clusters=list_of_info_clusters,
        list_of_args_clusters=list_of_args_clusters,
        activities_labels=activities_labels,
        all_title_event_code=all_title_event_code,
        list_of_hour=list_of_hour,
        list_of_dayofweek=list_of_dayofweek,
        test_set=True)
    test = compile_history.compile_history_data(test)

    return train, test


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


class EventClusterizer:
    def __init__(self, info_cluster_num=10, args_cluster_num=8):
        self.info_cluster_num = info_cluster_num
        self.args_cluster_num = args_cluster_num
        self.specs_df = pd.read_csv(f'{RAW_PATH}/specs.csv')

    def process(self):
        self.specs_df['info_clusters'] = self.vectorize(
            columns='info', cluster_num=self.info_cluster_num)
        self.specs_df['args_clusters'] = self.vectorize(
            columns='args',  cluster_num=self.args_cluster_num)
        self.specs_df = self.specs_df.set_index('event_id')
        return self.specs_df[['info_clusters', 'args_clusters']]

    def vectorize(self, columns, cluster_num):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.specs_df[columns].values)
        clusters = KMeans(
            n_clusters=cluster_num,
            random_state=SEED,
            ).fit_predict(X.toarray())
        clusters = [f'{columns}_' + str(i) for i in clusters]
        return clusters


class GetData():
    """各installation_idのおける過去のゲームの実績をまとめるmethod."""

    def __init__(self,
                 win_code,
                 assess_titles,
                 list_of_event_code,
                 list_of_event_id,
                 list_of_info_clusters,
                 list_of_args_clusters,
                 activities_labels,
                 all_title_event_code,
                 list_of_hour,
                 list_of_dayofweek,
                 test_set=False):

        self.win_code = win_code
        self.assess_titles = assess_titles
        self.list_of_event_code = list_of_event_code
        self.list_of_event_id = list_of_event_id
        self.list_of_info_clusters = list_of_info_clusters
        self.list_of_args_clusters = list_of_args_clusters
        self.activities_labels = activities_labels
        self.all_title_event_code = all_title_event_code
        # self.list_of_date = list_of_date
        # self.list_of_month = list_of_month
        self.list_of_hour = list_of_hour
        self.list_of_dayofweek = list_of_dayofweek

        self.user_activities_count = {
            'Clip': 0,
            'Activity': 0,
            'Assessment': 0,
            'Game': 0
            }

        # self.nearly_user_activities_count = {
        #     'nearly_Clip': 0,
        #     'nearly_Activity': 0,
        #     'nearly_Assessment': 0,
        #     'nearly_Game': 0
        #     }

        self.last_activity = 0
        self.test_set = test_set
        self.count_actions = 0

        # print(self.list_of_event_code)
        self.event_code_count: Dict[str, int] = {ev: 0 for ev in self.list_of_event_code}
        self.event_id_count: Dict[str, int] = {eve: 0 for eve in self.list_of_event_id}
        self.info_clusters_count: Dict[str, int] = {eve: 0 for eve in self.list_of_info_clusters}
        self.args_clusters_count: Dict[str, int] = {eve: 0 for eve in self.list_of_args_clusters}
        self.title_count: Dict[str, int] = {eve: 0 for eve in self.activities_labels.values()}
        # self.title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in self.all_title_event_code}
        # self.date_count: Dict[str, int] = {f'date_{eve}': 0 for eve in self.list_of_date}
        # self.month_count: Dict[str, int] = {f'month_{eve}': 0 for eve in self.list_of_month}
        self.hour_count: Dict[str, int] = {f'hour_{eve}': 0 for eve in self.list_of_hour}
        self.dayofweek_count: Dict[str, int] = {f'dayofweek_{eve}': 0 for eve in self.list_of_dayofweek}

        self.total_duration = 0
        self.frequency = 0

        self.game_mean_event_count = 0
        self.accumulated_game_miss = 0

        self.game_round = []
        # self.game_duration = []
        # self.game_level = []

        self.good_comment = 0
        # self.coordinates = 0
        # self.coordinates = 0
        # self.description_val = 0
        # self.description = 0

        # self.coordinates_x = []
        # self.coordinates_y = []
        # self.size = []

    def process(self, user_sample, installation_id):

        all_assessments = []
        get_assesments = GetAssessmentFeature(self.win_code,
                                              self.assess_titles,
                                              self.list_of_event_code,
                                              self.list_of_event_id,
                                              self.list_of_info_clusters,
                                              self.list_of_args_clusters,
                                              self.activities_labels,
                                              self.all_title_event_code,
                                              test_set=self.test_set)
        first_session = user_sample.iloc[0, user_sample.columns.get_loc('timestamp')]
        # まずgame_sessionでgroupbyする
        for i, (session_id, session) in enumerate(
                user_sample.groupby('game_session', sort=False)):
            session_type = session['type'].iloc[0]
            # print(session_type)
            # game_session数が1以下を省く
            if self.test_set is True:
                second_condition = True
            else:
                if len(session) > 1:
                    second_condition = True
                else:
                    second_condition = False

            # gameを初めて開始してからの時間を計測
            # if i == 0:
            features: dict = self.user_activities_count.copy()

            if session_type == "Game":

                self.game_mean_event_count = (self.game_mean_event_count + session['event_count'].iloc[-1])/2.0

                game_s = session[session.event_code == 2030]
                misses_cnt = self.count_miss(game_s)
                self.accumulated_game_miss += misses_cnt

                try:
                    game_round_ = json.loads(session['event_data'].iloc[-1])["round"]
                    self.game_round.append(game_round_)

                except:
                    pass

                # try:
                #     game_duration_ = json.loads(session['event_data'].iloc[-1])["duration"]
                #     self.game_duration.append(game_duration_)
                # except:
                #     pass

                # try:
                #     game_level_ = json.loads(session['event_data'].iloc[-1])["level"]
                #     self.game_level.append(game_level_)
                # except:
                #     pass


            # session typeがAssessmentのやつだけ、カウントする。
            if (session_type == 'Assessment') & (second_condition):
                features = get_assesments.process(session, features)

                if features is not None:
                    features['installation_id'] = installation_id
                    # 特徴量に前回までのゲームの回数を追加
                    features['count_actions'] = self.count_actions

                    features.update(self.event_code_count.copy())
                    features.update(self.event_id_count.copy())
                    features.update(self.info_clusters_count.copy())
                    features.update(self.args_clusters_count.copy())
                    # features.update(self.date_count.copy())
                    # features.update(self.month_count.copy())
                    features.update(self.hour_count.copy())
                    features.update(self.dayofweek_count.copy())
                    # features.update(self.nearly_user_activities_count.copy())

                    features.update(self.title_count.copy())
                    #  features.update(self.title_event_code_count.copy())

                    features['total_duration'] = self.total_duration
                    features['frequency'] = self.frequency

                    features['game_mean_event_count'] = self.game_mean_event_count
                    features['accumulated_game_miss'] = self.accumulated_game_miss
                    features['mean_game_round'] = np.mean(self.game_round) if len(self.game_round) != 0 else 0
                    # features['max_game_round'] = np.max(self.game_round) if len(self.game_round) != 0 else 0
                    # features['sum_game_round'] = np.max(self.game_round) if len(self.game_round) != 0 else 0
                    # features['mean_game_duration'] = np.mean(self.game_duration) if len(self.game_duration) != 0 else 0
                    # features['max_game_duration'] = np.max(self.game_duration) if len(self.game_duration) != 0 else 0
                    # features['sum_game_duration'] = np.sum(self.game_duration) if len(self.game_duration) != 0 else 0
                    # features['std_game_duration'] = np.std(self.game_duration) if len(self.game_duration) != 0 else 0
                    # features['mean_game_level'] = np.mean(self.game_level) if len(self.game_level) != 0 else 0
                    # features['max_game_level'] = np.max(self.game_level) if len(self.game_level) != 0 else 0
                    # features['sum_game_level'] = np.sum(self.game_level) if len(self.game_level) != 0 else 0

                    # features['mean_coordinates_x'] = np.nanmean(self.coordinates_x) if len(self.coordinates_x) != 0 else 0
                    # features['std_coordinates_x'] = np.nanstd(self.coordinates_x) if len(self.coordinates_x) != 0 else 0
                    # features['mean_coordinates_y'] = np.nanmean(self.coordinates_y) if len(self.coordinates_y) != 0 else 0
                    # features['std_coordinates_y'] = np.nanstd(self.coordinates_y) if len(self.coordinates_y) != 0 else 0

                    # features['mean_size'] = np.nanmean(self.size) if len(self.size) != 0 else 0
                    # features['max_size'] = np.nanmax(self.size) if len(self.size) != 0 else 0

                    features['good_comment'] = self.good_comment

                    # features['description'] = self.description
                    # features['coordinates'] = self.coordinates
                    # features['description_val'] = self.description_val

                    all_assessments.append(features)


            # coordinates = session['event_data'].str.contains('coordinates').sum()
            # self.coordinates += coordinates

            # description = session['event_data'].str.contains('description').sum()
            # self.description += description

            # event_data = pd.io.json.json_normalize(session.event_data.apply(json.loads))
            # try:
            #     self.coordinates_x += (event_data['coordinates.x']/event_data['coordinates.stage_width']).to_list()
            # except:
            #     pass
            # try:
            #     self.coordinates_y += (event_data['coordinates.y']/event_data['coordinates.stage_height']).to_list()
            # except:
            #     pass
            # try:
            #     self.size += event_data['size'].to_list()
            # except:
            #     pass
            # try:
            #     self.description_val = len(set(event_data['description']))
            # except:
            #     pass

            good_comment_list = ['Good', 'good', 'cool', 'Cool', 'Nice', 'nice', 'Great', 'great', 'Amaging']
            for comment in good_comment_list:
                self.good_comment += session['event_data'].str.contains(comment).sum()

            # print(session.iloc[-1, session.columns.get_loc('timestamp')])
            self.total_duration = (session.iloc[-1, session.columns.get_loc('timestamp')] - first_session).seconds
            self.count_actions += len(session)
            if self.total_duration == 0:
                self.frequency = 0
            else:
                self.frequency = self.count_actions / self.total_duration

            self.event_code_count = self.update_counters(
                session, self.event_code_count, "event_code")
            self.event_id_count = self.update_counters(
                session, self.event_id_count, "event_id")
            self.info_clusters_count = self.update_counters(
                session, self.info_clusters_count, "info_clusters")
            self.args_clusters_count = self.update_counters(
                session, self.args_clusters_count, "args_clusters")
            self.title_count = self.update_counters(
                session, self.title_count, 'title')
            # self.date_count = self.update_counters(
            #     session, self.date_count, "date")
            # self.month_count = self.update_counters(
            #     session, self.month_count, "month")
            self.hour_count = self.update_counters(
                session, self.hour_count, "hour")
            self.dayofweek_count = self.update_counters(
                session, self.dayofweek_count, "dayofweek")

            # self.title_event_code_count = self.update_counters(
            #     session, self.title_event_code_count, 'title_event_code')

            # second_conditionがFalseのときは、user_activities_countのみ増える。
            if self.last_activity != session_type:
                # self.nearly_user_activities_count *= 0.8
                # self.nearly_user_activities_count[f'nearly_{session_type}'] += 1

                self.user_activities_count[session_type] += 1
                self.last_activitiy = session_type

        if self.test_set:
            return all_assessments[-1]
        return all_assessments

    def update_counters(self, session, counter: dict, col: str):
        num_of_session_count = Counter(session[col])
        for k in num_of_session_count.keys():
            x = k
            if col == 'title':
                x = self.activities_labels[k]
            if col in ['hour', 'month', 'date', 'dayofweek']:
                x = f'{col}_{x}'
            counter[x] += num_of_session_count[k]
        return counter

    def count_miss(self, df):
        cnt = 0
        for e in range(len(df)):
            x = df['event_data'].iloc[e]
            y = json.loads(x)['misses']
            cnt += y
        return cnt


class GetAssessmentFeature:

    def __init__(self,
                 win_code,
                 assess_titles,
                 list_of_event_code,
                 list_of_event_id,
                 list_of_info_clusters,
                 list_of_args_clusters,
                 activities_labels,
                 all_title_event_code,
                 test_set=False):

        self.list_of_event_code = list_of_event_code
        self.list_of_event_id = list_of_event_id
        self.list_of_info_clusters = list_of_info_clusters
        self.list_of_args_clusters = list_of_args_clusters
        self.activities_labels = activities_labels
        self.all_title_event_code = all_title_event_code
        self.assess_titles = assess_titles

        self.test_set = test_set
        self.win_code = win_code
        self.accuracy_groups = {'acc_0': 0, 'acc_1': 0, 'acc_2': 0, 'acc_3': 0}
        self.mean_accuracy_group = 0  # accuracy_groupsの平均
        self.count_accuracy = 0
        self.count_correct_attempts = 0  # 成功
        self.count_uncorrect_attempts = 0  # 失敗
        self.counter = 0
        self.durations = []
        self.true_attempts = 0
        self.false_attempts = 0

        self.last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    def process(self, session, features):
        all_attempts = session.query(
                f"event_code == {self.win_code[session['title'].iloc[0]]}"
                )
        assert type(all_attempts) == pd.DataFrame

        features['session_title'] = session['title'].iloc[0]
        session_title = session['title'].iloc[0]
        # print(self.activities_labels[session_title])
        session_title_text = self.activities_labels[session_title]
        # 特徴量に前回までの正解数と失敗数追加
        features = self.add_count_attempts(features, all_attempts)

        # 特徴量に前回までのaccuracyを追加
        count_acc = self.count_accuracy/self.counter if self.counter > 0 else 0
        features['count_accuracy'] = count_acc
        accuracy = self.calc_accuracy(self.true_attempts, self.false_attempts)
        self.count_accuracy += accuracy

        features.update(self.last_accuracy_title.copy())
        # print(accuracy)
        self.last_accuracy_title['acc_' + session_title_text] = accuracy

        # 特徴量に前回までの平均ゲーム時間を追加
        # features = self.add_duration_mean(features, session)

        # 特徴量に今回のacc_groupを追加
        features = self.add_accuracy_group(features, accuracy)

        # 特徴量に前回までのacc_groupの平均を追加
        if self.counter > 0:
            mean_acc_gp = self.mean_accuracy_group/self.counter
        else:
            mean_acc_gp = 0
        features['mean_accuracy_group'] = mean_acc_gp

        # 特徴量に前回までのacc_groupの数を追加
        features.update(self.accuracy_groups)
        self.mean_accuracy_group += features['accuracy_group']
        self.accuracy_groups[f"acc_{features['accuracy_group']}"] += 1

        self.counter += 1

        # trainで試行回数が0のものを除外する。
        if self.test_set is True:
            return features
        else:
            if self.true_attempts + self.false_attempts == 0:
                # print(0)
                pass
            else:
                # print(features)
                return features

    # def add_duration_mean(self, df, session):
    #     if self.durations == []:
    #         df['duration_mean'] = 0
    #         df['duration_max'] = 0
    #         df['duration_std'] = 0
    #         df['duration_median'] = 0

    #     else:
    #         df['duration_mean'] = np.mean(self.durations)
    #         df['duration_max'] = np.max(self.durations)
    #         df['duration_std'] = np.std(self.durations)
    #         df['duration_median'] = np.median(self.durations)

        # self.durations.append(
        #     (session.iloc[-1, session.columns.get_loc('timestamp')] - session.iloc[0, session.columns.get_loc('timestamp')]).seconds
        #     )
        # return df

    def add_accuracy_group(self,
                           df: pd.DataFrame,
                           accuracy: int) -> pd.DataFrame:
        if accuracy == 0:
            df['accuracy_group'] = 0
        elif accuracy == 1:
            df['accuracy_group'] = 3
        elif accuracy == 0.5:
            df['accuracy_group'] = 2
        else:
            df['accuracy_group'] = 1
        return df

    def add_count_attempts(self,
                           df: pd.DataFrame,
                           all_attempts: pd.DataFrame):
        """result: 'true' or 'false'."""
        # correct
        df['count_correct_attempts'] = self.count_correct_attempts
        self.true_attempts = all_attempts['event_data'].str.contains('true').sum()
        self.count_correct_attempts += self.true_attempts

        # uncorrect
        df['count_uncorrect_attempts'] = self.count_uncorrect_attempts
        self.false_attempts = all_attempts['event_data'].str.contains('false').sum()
        self.count_uncorrect_attempts += self.false_attempts
        return df

    def calc_accuracy(self, true, false):
        accuracy = true/(true+false) if (true+false) != 0 else 0
        return accuracy


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
        self.list_of_hour = list_of_hour
        self.list_of_dayofweek = list_of_dayofweek
        self.test_set = test_set

    def compile_history_data(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        """過去のデータを、installation_idごとのデータにまとめる."""
        get_data = GetData(
            win_code=self.win_code,
            assess_titles=self.assess_titles,
            list_of_event_code=self.list_of_event_code,
            list_of_event_id=self.list_of_event_id,
            list_of_info_clusters=self.list_of_info_clusters,
            list_of_args_clusters=self.list_of_args_clusters,
            activities_labels=self.activities_labels,
            all_title_event_code=self.all_title_event_code,
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


###############################################################################
# feature_preprocessing
###############################################################################
EVENT_ID_LIST = ["c51d8688", "d51b1749", "cf82af56", "30614231", "f93fc684", "1b54d27f", "a8a78786", "37ee8496", "51102b85", "2a444e03", "01ca3a3c", "acf5c23f", "86ba578b", "a8cc6fec", "160654fd", "65a38bf7", "c952eb01", "06372577", "d88e8f25", "0413e89d", "8d7e386c", "58a0de5c", "4bb2f698", "5be391b5", "ab4ec3a4", "b2dba42b", "2b9272f4", "7f0836bf", "5154fc30", "3afb49e6", "9ce586dd", "16dffff1", "2b058fe3", "a7640a16", "3bfd1a65", "bfc77bd6","003cd2ee","fbaf3456","19967db1","df4fe8b6","5290eab1","3bb91dda","6f8106d9","5f0eb72c","5c2f29ca","7d5c30a2","90d848e0","3babcb9b","53c6e11a","d9c005dd","15eb4a7d","15a43e5b","f71c4741","895865f3","b5053438","7da34a02","a5e9da97","a44b10dc","17ca3959","ab3136ba","76babcde","1375ccb7","67439901","363d3849","bcceccc6","90efca10","a6d66e51","90ea0bac","7ab78247","25fa8af4","dcaede90","65abac75","15f99afc","47f43a44","8fee50e2","51311d7a","c6971acf","56bcd38d","c74f40cd","d3268efa","c189aaf2","7ec0c298","9e6b7fb5","5c3d2b2f","e4d32835","0d1da71f","abc5811c","e37a2b78","6d90d394","3d63345e","b012cd7f","792530f8","26a5a3dd","d2e9262e","598f4598","00c73085","7cf1bc53","46b50ba8","ecc36b7f","363c86c9","5859dfb6","8ac7cce4","ecc6157f","daac11b0","4a09ace1","47026d5f","9b4001e4","67aa2ada","f7e47413","cf7638f3","a2df0760","e64e2cfd","119b5b02","1bb5fbdb","1beb320a","7423acbc","14de4c5d","587b5989","828e68f9","3d0b9317","4b5efe37","fcfdffb6","a76029ee","1575e76c","bdf49a58","b2e5b0f1","bd701df8","250513af","6bf9e3e1","7dfe6d8a","9e34ea74","6cf7d25c","1cc7cfca","e79f3763","5b49460a","37c53127","ea296733","b74258a0","02a42007","9b23e8ee","bb3e370b","6f4bd64e","05ad839b","611485c5","a592d54e","c7f7f0e1","857f21c0","6077cc36","3dcdda7f","5e3ea25a","e7561dd2","f6947f54","9e4c8c7b","8d748b58","9de5e594","c54cf6c5","28520915","9b01374f","55115cbd","1f19558b","56cd3b43","795e4a37","83c6c409","3ee399c3","a8876db3","30df3273","77261ab5","4901243f","7040c096","1af8be29","f5b8c21a","499edb7c","6c930e6e","1cf54632","f56e0afc","6c517a88","d38c2fd7","2ec694de","562cec5f","832735e1","736f9581","b1d5101d","36fa3ebe","3dfd4aa4","1c178d24","2dc29e21","4c2ec19f","38074c54","ca11f653","f32856e4","89aace00","29bdd9ba","4ef8cdd3","d2278a3b","0330ab6a","46cd75b4","6f445b57","5e812b27","4e5fc6f5","df4940d3","33505eae","f806dc10","4a4c3d21","cb6010f8","022b4259","99ea62f3","461eace6","ac92046e","0a08139c","fd20ea40","ea321fb1","e3ff61fb","c7128948","31973d56","2c4e6db0","b120f2ac","731c0cbe","d3f1e122","9d4e7b25","cb1178ad","5de79a6a","0d18d96c","a8efe47b","cdd22e43","392e14df","93edfe2e","3ccd3f02","b80e5e84","262136f4","a5be6304","e57dd7af","d3640339","565a3990","beb0a7b9","884228c8","bc8f2793","73757a5e","709b1251","c7fe2a55","3bf1cf26","c58186bf","ad148f58","cfbd47c8","6f4adc4b","e694a35b","222660ff","0086365d","3d8c61b0","37db1c2f","f50fc6c1","8f094001","2a512369","7ad3efc6","e4f1efe6","8af75982","08fd73f3","a1e4395d","3323d7e9","47efca07","5f5b2617","6043a2b4","7d093bf9","907a054b","5e109ec3","cc5087a3","8d84fa81","4074bac2","4d911100","84b0e0c8","45d01abe","0db6d71d","29f54413","1325467d","99abe2bb","f3cd5473","91561152","dcb1663e","f28c589a","0ce40006","7525289a","17113b36","29a42aea","db02c830","ad2fc29c","d185d3ea","37937459","804ee27f","28a4eb9a","49ed92e9","56817e2b","86c924c4","e5c9df6f","b7dc8128","3edf6747","8b757ab8","93b353f2","a52b92d5","9c5ef70c","dcb55a27","d06f75b5","44cb4907","c2baf0bd","27253bdc","a1bbe385","2fb91ec1","155f62a4","ec138c1c","c1cac9a2","eb2c19cd","3ddc79c3","a0faea5d","070a5291","7fd1ac25","69fdac0a","28ed704e","c277e121","13f56524","e720d930","3393b68b","f54238ee","63f13dd7","e9c52111","763fc34e","26fd2d99","7961e599","9d29771f","84538528","77ead60d","3b2048ee","923afab1","3a4be871","9ed8f6da","74e5f8a7","a1192f43","e5734469","d122731b","04df9b66","5348fd84","92687c59","532a2afb","2dcad279","77c76bc5","71e712d8","a16a373e","3afde5dd","d2659ab4","88d4a5be","6088b756","85d1b0de","08ff79ad","15ba1109","7372e1a5","9ee1c98c","71fe8f75","6aeafed4","756e5507","d45ed6a1","1996c610","de26c3a6","1340b8d7","5d042115","e7e44842","2230fab4","9554a50b","4d6737eb","bd612267","3bb91ced","28f975ea","b7530680","b88f38da","5a848010","85de926c","c0415e5c","a29c5338","d88ca108","16667cc5","87d743c1","d02b7a8e","e04fb33d","bbfe0445","ecaab346","48349b14","5dc079d8","e080a381"]
EVENT_CODE_LIST  = [2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
                    4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
                    2040, 4090, 4220, 4095, 4050]
TITLE_LIST = ["All Star Sorting","Welcome to Lost Lagoon!","Scrub-A-Dub","Costume Box","Magma Peak - Level 1","Bubble Bath","Crystal Caves - Level 2","Tree Top City - Level 3","Slop Problem","Happy Camel","Dino Drink","Watering Hole (Activity)","Heavy, Heavier, Heaviest","Honey Cake","Crystal Caves - Level 1","Air Show","Chicken Balancer (Activity)","Flower Waterer (Activity)","Cart Balancer (Assessment)","Mushroom Sorter (Assessment)","Leaf Leader","Dino Dive","Pan Balance","Cauldron Filler (Assessment)","Treasure Map","Crystals Rule","Lifting Heavy Things","Rulers","Fireworks (Activity)","Pirate's Tale","12 Monkeys","Balancing Act","Bottle Filler (Activity)","Chest Sorter (Assessment)","Sandcastle Builder (Activity)","Crystal Caves - Level 3","Egg Dropper (Activity)","Magma Peak - Level 2","Tree Top City - Level 2","Ordering Spheres","Chow Time","Bug Measurer (Activity)","Bird Measurer (Assessment)","Tree Top City - Level 1"]
ACC_ASSESSMENT = ["acc_Mushroom Sorter (Assessment)", "acc_Chest Sorter (Assessment)", "acc_Bird Measurer (Assessment)", "acc_Cart Balancer (Assessment)", "acc_Cauldron Filler (Assessment)"]
ACC_NO = ["acc_0", "acc_1", "acc_2", "acc_3"]
DROP_FEATURES = ['count_correct_attempts', 'total_duration', 'count_uncorrect_attempts']


def preprocess_features_main(train_df, test_df):

    preprocess = PreprocessFeatures()
    train = preprocess.process(train_df)
    test = preprocess.process(test_df)

    time_preprocess = PreprocessTime()
    train = time_preprocess.process(train)
    test = time_preprocess.process(test)

    mutual_preprocess = MutualPreprocessFeatures()
    train, test = mutual_preprocess.process(train, test)

    train.drop(DROP_FEATURES, axis=1, inplace=True)
    test.drop(DROP_FEATURES, axis=1, inplace=True)

    train.columns = train.columns.str.replace(',', '')
    test.columns = test.columns.str.replace(',', '')


    # train.to_csv('data/transformation/trans_train.csv')
    # test.to_csv('data/transformation/trans_test.csv')
    return train, test

class MutualPreprocessFeatures:
    
    def process(self, train: pd.DataFrame, test: pd.DataFrame):
        # train, test, label = self.list_label_encoder(train, test, ['description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['frequency_label', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['mean_accuracy_group_label', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['count_action_label', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['session_title', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['session_title', 'count_action_label'])
        # train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['session_title', 'count_accuracy_label'])
        train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['session_title', 'mean_accuracy_group_label'])
        train, test = self.frequency_encoder(train, test, label)
        return train, test

    def list_label_encoder(self, train, test, columns: list):
        le = LabelEncoder()
        train_description_val_session_title = train[columns].apply(lambda x: str([i for i in x]), axis=1)
        test_description_val_session_title = test[columns].apply(lambda x: str([i for i in x]), axis=1)
        le.fit(train_description_val_session_title)
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        label_name = '_'.join(columns)
        label_name = f'label_{label_name}'
        train[label_name] = train_description_val_session_title.apply(lambda x: le_dict.get(x, -1))
        test[label_name] = test_description_val_session_title.apply(lambda x: le_dict.get(x, -1))
        return train, test, label_name

    def frequency_encoder(self, train, test, label_name):
        freq = train[label_name].value_counts()
        train[f'count_{label_name}'] = train[label_name].map(freq)
        test[f'count_{label_name}'] = test[label_name].map(freq)
        return train, test


class PreprocessFeatures:

    def process(self, df):
        df['frequency_label'] = df.frequency.apply(lambda x: self.frequency_label_encoder(np.log1p(x)))
        df.drop('frequency', axis=1, inplace=True)
        df['count_action_label'] = df.count_actions.apply(lambda x: self.bins_count_action_label_encoder(np.log1p(x)))
        df.drop('count_actions', axis=1, inplace=True)
        df['count_accuracy_label'] = df.count_accuracy.apply(lambda x: self.count_accuracy_label_encoder(x))
        df.drop('count_accuracy', axis=1, inplace=True)
        df['mean_accuracy_group_label'] = df.mean_accuracy_group.apply(lambda x: self.mean_accuracy_group_label_encoder(x))
        df.drop('mean_accuracy_group', axis=1, inplace=True)
        
        sum_type = df[['Game', 'Clip', 'Activity']].sum(axis=1)
        df['ratio_Game'] = df['Game']/sum_type
        df['ratio_Clip'] = df['Clip']/sum_type
        df['ratio_Activity'] = df['Activity']/sum_type
        # df['ratio_Assessment'] = df['Assessment']/sum_type
        df.drop(['Assessment', 'Game', 'Clip', 'Activity'], axis=1, inplace=True)

        """game_sessionごとではなく、instration_idごとのcountとか"""
        # df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        # df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        # df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        # df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        # df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
        #                                 4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
        #                                 2040, 4090, 4220, 4095]].sum(axis=1)
        df['sum_event_code_2000'] = df.loc[:, [2050, 2060, 2070, 2075, 2080, 2081, 2083, 2000, 2010, 2020, 2025, 2030, 2035, 2040]].sum(axis=1)
        df['sum_event_code_3000'] = df.loc[:, [3110, 3120, 3121, 3010, 3020, 3021]].sum(axis=1)
        df['sum_event_code_4000'] = df.loc[:, [4100, 4230, 4235, 4110, 4010, 4020, 4021, 4022, 4025, 4030, 4031, 4035, 4040, 4045, 4070, 4080, 4090, 4220, 4095, 4050]].sum(axis=1)

        df['event_id_non_zero_count'] = df.loc[:, EVENT_ID_LIST].apply(lambda x: len(x.to_numpy().nonzero()[0]), axis=1)
        df['event_code_non_zero_count'] = df.loc[:, EVENT_CODE_LIST].apply(lambda x: len(x.to_numpy().nonzero()[0]), axis=1)
        df['title_non_zero_count'] = df.loc[:, TITLE_LIST].apply(lambda x: len(x.to_numpy().nonzero()[0]), axis=1)
        # df['good_comment_ratio'] = df['good_comment'] / df['description']

        sum_event_code = df[['sum_event_code_2000', 'sum_event_code_3000', 'sum_event_code_4000']].sum(axis=1)
        # df['ratio_event_code_2000'] = df['sum_event_code_2000']/sum_event_code
        df['ratio_event_code_3000'] = df['sum_event_code_3000']/sum_event_code
        df['ratio_event_code_4000'] = df['sum_event_code_4000']/sum_event_code
        df.drop(['sum_event_code_2000','ratio_event_code_3000', 'ratio_event_code_4000'], axis=1, inplace=True)
        
        # df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        # df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')
        # df.drop('sum_event_code_count', axis=1, inplace=True)
        
        return df

    def count_accuracy_label_encoder(self, x):
        if x == 0:
            y = 0
        elif x > 0 and x <= 0.34:
            y = 1
        elif x > 0.34 and x <= 0.67:
            y = 2
        elif x > 0.67 and x < 1:
            y = 3
        elif x == 1:
            y = 4
        return y

    def mean_accuracy_group_label_encoder(self, x):
        if x == 0:
            y = 0
        elif x > 0 and x < 1.0:
            y = 1
        elif x >= 1.0 and x < 2.0:
            y = 2
        elif x >= 2.0 and x < 3.0:
            y = 3
        elif x == 3.0:
            y = 4
        return y

    def bins_count_action_label_encoder(self, x):
        if x < 1.0:
            y = 0
        elif x >= 1.0 and x < 2.0:
            y = 1
        elif x >= 2.0 and x < 3.0:
            y = 2
        elif x >= 3.0 and x < 4.0:
            y = 3
        elif x >= 4.0 and x < 5.0:
            y = 4
        elif x >= 5.0 and x < 6.0:
            y = 5
        elif x >= 6.0 and x < 7.0:
            y = 6
        elif x >= 7.0 and x < 8.0:
            y = 7
        elif x >= 8.0 and x < 9.0:
            y = 8
        else:
            y = 9
        return y

    def frequency_label_encoder(self, x):
        if x <= 0.1:
            y = 0
        elif x > 0.1 and x <= 0.2:
            y = 1
        elif x > 0.2 and x <= 0.3:
            y = 2
        elif x > 0.3 and x <= 0.4:
            y = 3
        elif x > 0.4 and x <= 0.5:
            y = 4
        elif x > 0.5 and x <= 1.0:
            y=5
        else:
            y=6
        return y


class PreprocessTime:

    def process(self, df):
        lank = 5
        new_columns = [f'event_id_lank{i}' for i in range(lank)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=EVENT_ID_LIST)
        df.drop(EVENT_ID_LIST, axis=1, inplace=True)

        new_columns = [f'EVENT_CODE_LIST_lank{i}' for i in range(lank)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=EVENT_CODE_LIST)
        df.drop(EVENT_CODE_LIST, axis=1, inplace=True)
        
        new_columns = [f'TITLE_LIST_lank{i}' for i in range(lank)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=TITLE_LIST)
        df.drop(TITLE_LIST, axis=1, inplace=True)

        new_columns = [f'ACC_ASSESSMENT_lank{i}' for i in range(5)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=ACC_ASSESSMENT)
        df.drop(ACC_ASSESSMENT, axis=1, inplace=True)

        new_columns = [f'ACC_NO_lank{i}' for i in range(4)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=ACC_NO)
        df.drop(ACC_NO, axis=1, inplace=True)

        # df = self.common_process(df, 'date', lank=3)
        df = self.common_process(df, 'hour', lank=3)
        # df = self.common_process(df, 'month', lank=1)
        df = self.common_process(df, 'dayofweek', lank=1)
        df = self.common_process(df, 'args', lank=3)
        df = self.common_process(df, 'info', lank=3)
        
        return df

    def common_process(self, df, word_in_colums: str, lank: int):
        columns = df.columns[df.columns.str.contains(word_in_colums)]
        # df[f'mean_{word_in_colums}_count'] = df[columns].mean(axis=1)
        df[f'std_{word_in_colums}_count'] = df[columns].std(axis=1)
        df[f'non_zero_{word_in_colums}'] = (df[columns] == 0).sum(axis=1)
        new_columns = [f'{word_in_colums}_lank{i}' for i in range(lank)]
        print(new_columns)
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=columns)
        df.drop(columns, axis=1, inplace=True)
        return df

    def best_columns(self, df, lank, columns):
        df = df[columns]
        rename_columns = {date: i for i, date in enumerate(columns)}
        df = df.rename(columns=rename_columns)
        best_df = df.rank(axis=1, ascending=False).apply(
            lambda x: pd.Series(list(x.sort_values()[0:lank].index)), axis=1
            )
        return best_df

###############################################################################
# loss
###############################################################################

def lgb_qwk(preds, data):
    params = {
        'threshold_0': 1.12,
        'threshold_1': 1.62,
        'threshold_2': 2.20
        },

    func = np.frompyfunc(threshold, 2, 1)
    test_pred = func(preds, params)
    loss = qwk(test_pred, data.get_label())
    return 'qwk', loss, True


def qwk(a1, a2):
    """
    Source:
    https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def threshold(x, params):
    if x < params['threshold_0']:
        y = 0
    elif x < params['threshold_1']:
        y = 1
    elif x < params['threshold_2']:
        y = 2
    else:
        y = 3
    return y


###############################################################################
# closs validation
###############################################################################

def stratified_group_k_fold(X, y, groups, k, seed=None):
    np.random.seed(seed)
    # ラベルの数をカウント
    labels_num = np.max(y) + 1
    # 各グループのラベルの数をカウントする
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1
    # 各フォールドのラベルの数をカウント
    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)

    for i in range(k):
        test_k = i
        # val_k = i+1 if i+1 != k else 0
        # print(val_k)
        train_groups = all_groups - groups_per_fold[test_k]  # - groups_per_fold[val_k]
        # val_groups = groups_per_fold[val_k]
        test_groups = groups_per_fold[test_k]
        # print(test_groups)
        # train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        # val_indices = [i for i, g in enumerate(groups) if g in val_groups]
        # test_indices = {str(g): [i for i, g in enumerate(groups) if g in test_groups]}

        def choice_ind(group):
            indices = []
            n_g = None
            list_ = []
            for i, g in enumerate(groups):
                if g in group:
                    if n_g is not None and n_g != g:
                        indices.append(list_)
                        list_ = []
                    list_.append(i)
                    n_g = g

            indices = [np.random.choice(i) for i in indices]
            return indices

        train_indices = choice_ind(train_groups)
        test_indices = choice_ind(test_groups)
        # print(train_indices)
        yield train_indices, test_indices  # val_indices,


###############################################################################
# post processing
###############################################################################


###############################################################################
# predict
###############################################################################

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    }

PARAMS = {
        'threshold_0': 1.12,
        'threshold_1': 1.62,
        'threshold_2': 2.20
        }


def train_and_predict(train_df, test_df):
    """main."""
    train_df.columns = train_df.columns.str.replace(',', '')
    test_df.columns = test_df.columns.str.replace(',', '')

    num_fold = 4
    total_val_pred = np.zeros([train_df.shape[0], ])
    test_list = []
    importance_list = []

    kfold = KFold(num_fold, random_state=77)
    seed_list = [77, 5, 8, 16, 42, 10,
                 777, 98, 356, 561, 36,
                 56, 1, 212, 444, 612]
    for fold_ind, (train_ind, val_ind) in enumerate(
            kfold.split(train_df)):

        loss_list = []
        val_list = []

        for seed in seed_list:
            loss, val_pred, test_pred, all_importance = inner_train_and_predict(
                train_df.iloc[train_ind], train_df.iloc[val_ind], test_df,
                seed=seed)
            print(loss)
            loss_list.append(loss)
            val_list.append(val_pred)
            test_list.append(test_pred)
            importance_list.append(all_importance)

        loss = np.mean(loss_list)
        val_pred = pd.concat(val_list, axis=1)
        total_val_pred[val_ind] = val_pred.mean(axis=1)

    test_pred = pd.concat(test_list, axis=1)
    test_pred = test_pred.mean(axis=1)

    func = np.frompyfunc(threshold, 2, 1)
    final_val_pred = func(total_val_pred, PARAMS)
    loss = qwk(final_val_pred, train_df['accuracy_group'])
    print(loss)
    pred_df = func(test_pred, PARAMS)
    return pred_df


def inner_train_and_predict(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            test_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    num_fold = 4
    groups = np.array(train_df['installation_id'])

    y = train_df['accuracy_group'].reset_index(drop=True)
    x = train_df.drop('accuracy_group', axis=1).reset_index(drop=True)
    x = x.drop('installation_id', axis=1)

    # val_y = val_df['accuracy_group']
    val_x = val_df.drop('accuracy_group', axis=1)
    val_x = val_x.drop('installation_id', axis=1)

    test_x = test_df.drop('accuracy_group', axis=1)
    test_x = test_x.drop('installation_id', axis=1)

    total_inner_val_pred = np.zeros([y.shape[0], ])
    total_outer_val_pred = np.zeros([val_x.shape[0], num_fold])
    total_test_pred = np.zeros([test_df.shape[0], num_fold])

    all_importance = []
    inner_val_index_all = []

    for fold_ind, (train_ind, inner_val_ind) in enumerate(
            stratified_group_k_fold(
                X=x, y=y, groups=groups, k=num_fold, seed=seed)):
        # # lgb
        model, y_pred, importance, test_x, val_x = lgb_model(
            x, y, train_ind, inner_val_ind, test_x, seed=seed, val_x=val_x)
        total_inner_val_pred[inner_val_ind] = y_pred

        val_pred = model.predict(
            val_x, num_iteration=model.best_iteration)
        total_outer_val_pred[:, fold_ind] = val_pred

        test_pred = model.predict(
            test_x, num_iteration=model.best_iteration)
        total_test_pred[:, fold_ind] = test_pred

        all_importance.append(importance)
        inner_val_index_all += inner_val_ind

    all_importance = pd.concat(all_importance, axis=1)

    # print(total_val_pred)
    func = np.frompyfunc(threshold, 2, 1)
    y = y[inner_val_index_all].values
    total_val_pred = total_inner_val_pred[inner_val_index_all]

    loss = qwk(func(total_val_pred, PARAMS), y)

    total_outer_val_pred = pd.DataFrame(total_outer_val_pred).mean(axis=1)
    total_test_pred = pd.DataFrame(total_test_pred).mean(axis=1)

    return loss, total_outer_val_pred, total_test_pred, all_importance


def lgb_model(x, y, train_ind, test_ind, test_x, seed, val_x=None):
    x_train = x.iloc[train_ind]
    y_train = y.iloc[train_ind]
    x_test = x.iloc[test_ind]
    y_test = y.iloc[test_ind]

    # target_encoding = TargetEncoding()
    # if val_x is not None:
    #     x_train, x_test, test_x, val_x = target_encoding.process(
    #         x_train, y_train, x_test, test_x,
    #         columns=FEATURES_FOR_TARGET_ENCODING,
    #         seed=seed, val_x=val_x)
    # else:
    #     x_train, x_test, test_x = target_encoding.process(
    #         x_train, y_train, x_test, test_x,
    #         columns=FEATURES_FOR_TARGET_ENCODING,
    #         seed=seed)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)

    model = lgb.train(params=LGB_PARAMS,
                      train_set=lgb_train,
                      valid_sets=lgb_val,
                      feval=lgb_qwk,
                      num_boost_round=1000,
                      early_stopping_rounds=50)

    y_pred = model.predict(x_test, num_iteration=model.best_iteration)

    importance = pd.DataFrame(model.feature_importance('gain'),
                              index=x_train.columns)
    return model, y_pred, importance, test_x, val_x


###############################################################################
# predict
###############################################################################

def submit_main(pred_df):
    submission = pd.read_csv(f'{RAW_PATH}/sample_submission.csv')
    submission['accuracy_group'] = pred_df.astype(int)
    print(submission.dtypes)
    submission.to_csv(f'submission.csv', index=False)


###############################################################################
# main
###############################################################################

def main():
    """main."""
    print('--read_csv--')
    train_df = pd.read_csv(f"{RAW_PATH}/train.csv")
    test_df = pd.read_csv(f"{RAW_PATH}/test.csv")
    train_labels_df = pd.read_csv(f"{RAW_PATH}/train_labels.csv")

    print('--compile--')
    compiled_train, compiled_test = compile_main(
        train_df, test_df, train_labels_df)

    print('--preprocess--')
    compiled_train, compiled_test = preprocess_features_main(
        compiled_train, compiled_test)
    # print(compiled_test.shape)

    print('--predict--')
    pred_df = train_and_predict(compiled_train, compiled_test)

    print('--submit--')
    submit_main(pred_df)


if __name__ == "__main__":
    main()