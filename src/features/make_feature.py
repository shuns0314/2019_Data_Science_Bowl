import json
from collections import Counter
from typing import Dict

import pandas as pd
import numpy as np


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
                 list_of_date,
                 list_of_month,
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
        self.game_duration = []
        self.game_level = []

        self.good_comment = 0
        self.coordinates = 0
        self.coordinates = 0
        self.description_val = 0
        self.description = 0

        self.coordinates_x = []
        self.coordinates_y = []
        self.size = []

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

                try:
                    game_duration_ = json.loads(session['event_data'].iloc[-1])["duration"]
                    self.game_duration.append(game_duration_)
                except:
                    pass

                try:
                    game_level_ = json.loads(session['event_data'].iloc[-1])["level"]
                    self.game_level.append(game_level_)
                except:
                    pass


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
                    features.update(self.nearly_user_activities_count.copy())

                    features.update(self.title_count.copy())
                    #  features.update(self.title_event_code_count.copy())

                    features['total_duration'] = self.total_duration
                    features['frequency'] = self.frequency

                    features['game_mean_event_count'] = self.game_mean_event_count
                    features['accumulated_game_miss'] = self.accumulated_game_miss
                    features['mean_game_round'] = np.mean(self.game_round) if len(self.game_round) != 0 else 0
                    features['max_game_round'] = np.max(self.game_round) if len(self.game_round) != 0 else 0
                    # features['sum_game_round'] = np.max(self.game_round) if len(self.game_round) != 0 else 0
                    features['mean_game_duration'] = np.mean(self.game_duration) if len(self.game_duration) != 0 else 0
                    features['max_game_duration'] = np.max(self.game_duration) if len(self.game_duration) != 0 else 0
                    features['sum_game_duration'] = np.sum(self.game_duration) if len(self.game_duration) != 0 else 0
                    features['std_game_duration'] = np.std(self.game_duration) if len(self.game_duration) != 0 else 0
                    features['mean_game_level'] = np.mean(self.game_level) if len(self.game_level) != 0 else 0
                    # features['max_game_level'] = np.max(self.game_level) if len(self.game_level) != 0 else 0
                    # features['sum_game_level'] = np.sum(self.game_level) if len(self.game_level) != 0 else 0

                    features['mean_coordinates_x'] = np.nanmean(self.coordinates_x) if len(self.coordinates_x) != 0 else 0
                    features['std_coordinates_x'] = np.nanstd(self.coordinates_x) if len(self.coordinates_x) != 0 else 0
                    features['mean_coordinates_y'] = np.nanmean(self.coordinates_y) if len(self.coordinates_y) != 0 else 0
                    features['std_coordinates_y'] = np.nanstd(self.coordinates_y) if len(self.coordinates_y) != 0 else 0

                    features['mean_size'] = np.nanmean(self.size) if len(self.size) != 0 else 0
                    features['max_size'] = np.nanmax(self.size) if len(self.size) != 0 else 0

                    features['good_comment'] = self.good_comment

                    features['description'] = self.description
                    features['coordinates'] = self.coordinates
                    features['description_val'] = self.description_val

                    all_assessments.append(features)


            coordinates = session['event_data'].str.contains('coordinates').sum()
            self.coordinates += coordinates

            description = session['event_data'].str.contains('description').sum()
            self.description += description

            event_data = pd.io.json.json_normalize(session.event_data.apply(json.loads))
            try:
                self.coordinates_x += (event_data['coordinates.x']/event_data['coordinates.stage_width']).to_list()
            except:
                pass
            try:
                self.coordinates_y += (event_data['coordinates.y']/event_data['coordinates.stage_height']).to_list()
            except:
                pass
            try:
                self.size += event_data['size'].to_list()
            except:
                pass
            try:
                self.description_val = len(set(event_data['description']))
            except:
                pass

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
        features = self.add_duration_mean(features, session)

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

    def add_duration_mean(self, df, session):
        if self.durations == []:
            df['duration_mean'] = 0
            df['duration_max'] = 0
            df['duration_std'] = 0
            df['duration_median'] = 0

        else:
            df['duration_mean'] = np.mean(self.durations)
            df['duration_max'] = np.max(self.durations)
            df['duration_std'] = np.std(self.durations)
            df['duration_median'] = np.median(self.durations)

        self.durations.append(
            (session.iloc[-1, session.columns.get_loc('timestamp')] - session.iloc[0, session.columns.get_loc('timestamp')]).seconds
            )
        return df

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
